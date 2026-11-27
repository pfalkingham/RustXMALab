//! The Settings dialog (modal).
//!
//! Ported control by control from `SettingsDialog.ui` and ui/SettingsDialog.cpp. Three
//! checkable buttons on the left ("General", "Calibration", "Tracking") choose the page shown
//! on the right (`on_pushButton_General_clicked` and the others); "Reset to Default" is
//! disabled and has no handler in the C++ either.
//!
//! Each control writes its setting as soon as it changes, as the C++ handlers do, so the
//! interface settings take effect at once. The settings that change computed results are
//! part of the project here (`xma_model::AlgorithmSettings`, parity ledger A10). They show
//! the open project's values, or this machine's defaults with no project open
//! (`UiSettings::algorithm_base`). The ones the Tracking page offers (E73 to E79,
//! `docs/cpp-editing.md` section 11) are enabled: a change is returned as a
//! [`SettingsPatch`]; the application remembers it as this machine's default and, with a
//! project open, applies it to the project as an undoable edit (`Edit::SetSettings`, which
//! recomputes what depends on it). A number being dragged or typed is returned when the
//! drag or the typing ends, so that one change is one undo step. The Calibration page's
//! detection method, its two thresholds, "Fix principal" and the three checkerboard settings
//! (the two check boxes and the invert-axis radio buttons) are enabled the same way.
//!
//! Closing with "Use custom workspace" checked and no path is refused with the C++ error
//! (`closeEvent`). "Set marker status colors" closes the dialog and opens the Status colours
//! dialog (`on_pushButton_MarkerStatus_clicked`).

use egui::Ui;
use xma_model::{AlgorithmSettings, SettingValue, SettingsPatch};

use super::form;
use crate::settings::UiSettings;

/// The three pages.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Page {
    #[default]
    General,
    Calibration,
    Tracking,
}

/// Items of `comboBox_DetectionMethodForCalibration`, in form order.
pub const DETECTION_ITEMS: [&str; 6] = [
    "none",
    "default Xray marker",
    "dark Blobs ",
    "white marker",
    "corner detection",
    "white Blobs",
];

/// The tooltip of each item of `comboBox_DetectionMethodForCalibration`.
///
/// The C++ sets seven item tooltips by detection-method number on a combo box of six items
/// in another order (index 0 is "none"), so most items showed another method's tooltip.
/// Here each item has the tooltip of the method it selects (`GLCameraView::mousePressEvent`
/// maps item 1 to 4 onto methods 0 to 3 and item 5 onto method 6).
pub const DETECTION_TIPS: [&str; 6] = [
    "Turns off automatic snapping. Marker stays exactly where clicked",
    "Optimized for dark, spherical beads in X-rays",
    "Uses OpenCV SimpleBlobDetector to find circular blobs",
    "Inverts image to track bright white beads on a dark background",
    "Uses Shi-Tomasi corner finder for sharp angles (e.g. checkerboards)",
    "OpenCV SimpleBlobDetector for bright circular blobs",
];

/// `comboBox_DetectionMethodForCalibration`: `crate::qt::combo_box` with each item's
/// tooltip ([`DETECTION_TIPS`], parity ledger U37). Returns the newly chosen index, if any.
///
/// Ported from ui/SettingsDialog.cpp, `SettingsDialog::SettingsDialog` (`:112-119`, the
/// item tooltips, here on the items they describe).
fn detection_combo(ui: &mut Ui, selected: usize, label: egui::Id) -> Option<usize> {
    let mut chosen = None;
    let font = egui::TextStyle::Button.resolve(ui.style());
    // Qt's AdjustToContents: wide enough for the longest item (as `qt::combo_box`).
    let longest = DETECTION_ITEMS
        .iter()
        .map(|s| {
            ui.painter()
                .layout_no_wrap((*s).to_owned(), font.clone(), egui::Color32::BLACK)
                .size()
                .x
        })
        .fold(0.0_f32, f32::max);
    let r = egui::ComboBox::from_id_salt(("comboBox_DetectionMethodForCalibration", label))
        .selected_text(DETECTION_ITEMS.get(selected).copied().unwrap_or(""))
        .width(longest + 34.0)
        .show_ui(ui, |ui| {
            for (i, (it, tip)) in DETECTION_ITEMS.iter().zip(DETECTION_TIPS).enumerate() {
                if ui
                    .selectable_label(i == selected, *it)
                    .on_hover_text(tip)
                    .clicked()
                    && i != selected
                {
                    chosen = Some(i);
                }
            }
        });
    let chosen = chosen.or_else(|| {
        crate::qt::combo_wheel(
            ui,
            &r.response,
            Some(selected),
            DETECTION_ITEMS.len(),
            |_| true,
        )
    });
    let _ = r.response.labelled_by(label);
    chosen
}

/// Items of `comboBox_TriangulationMethod`.
pub const TRIANGULATION_ITEMS: [&str; 5] = [
    "minimal least square reprojection (A*x = 0)",
    "iterative minimal least square reprojection (A*x = 0)",
    "minimal least square reprojection (A*x = B)",
    "iterative minimal least square reprojection (A*x = B)",
    "minimal least square distance to projected view rays",
];

/// The dialog's own state.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SettingsDialog {
    pub page: Page,
    /// A number setting being dragged or typed, by its C++ name: shown in its field, and
    /// returned when the drag or the typing ends.
    pub pending: Option<(&'static str, SettingValue)>,
}

/// What the dialog asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SettingsRequest {
    /// The window's close button (or Escape).
    Close,
    /// "Set marker status colors".
    StatusColors,
    /// "...": choose the workspace folder.
    ChooseWorkspace,
}

/// The error `closeEvent` shows when a custom workspace is on without a path.
pub const WORKSPACE_ERROR: &str =
    "You have to select a custom workspace or disable the custom workspace";

/// May the dialog close (`closeEvent`)?
#[must_use]
pub fn may_close(s: &UiSettings) -> bool {
    !(s.custom_workspace_path && s.workspace_path.is_empty())
}

impl SettingsDialog {
    /// Draws the dialog. `changed` is set when an interface setting changed; `patch`
    /// receives the algorithm settings the user changed (E73 to E79).
    pub fn ui(
        &mut self,
        ui: &mut Ui,
        s: &mut UiSettings,
        alg: &AlgorithmSettings,
        changed: &mut bool,
        patch: &mut SettingsPatch,
    ) -> Option<SettingsRequest> {
        let mut req = None;
        ui.set_width(820.0);
        if form::dialog_title(ui, "Settings") {
            req = Some(SettingsRequest::Close);
        }
        // The page area: as tall as the tallest page (Tracking, 21 rows) at the current
        // font, within the window, the same for every page so that the dialog does not move
        // when the page changes; it scrolls when the window is too low.
        let row = ui
            .spacing()
            .interact_size
            .y
            .max(ui.text_style_height(&egui::TextStyle::Body))
            + ui.spacing().item_spacing.y;
        let screen = ui.ctx().content_rect().height();
        let page_h = (row * 23.0).min(screen - 160.0).max(160.0);
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.set_width(110.0);
                ui.add_space(30.0);
                for (p, t) in [
                    (Page::General, "General"),
                    (Page::Calibration, "Calibration"),
                    (Page::Tracking, "Tracking"),
                ] {
                    let h = ui.spacing().interact_size.y;
                    if ui
                        .add(
                            egui::Button::new(t)
                                .selected(self.page == p)
                                .min_size(egui::vec2(110.0, h)),
                        )
                        .clicked()
                    {
                        if self.page != p
                            && let Some((name, v)) = self.pending.take()
                        {
                            patch.set(name, v);
                        }
                        self.page = p;
                    }
                }
            });
            ui.vertical(|ui| {
                form::styled_frame(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .id_salt("settings_page")
                        .max_height(page_h)
                        .min_scrolled_height(page_h)
                        .auto_shrink([false, false])
                        .show(ui, |ui| match self.page {
                            Page::General => {
                                if general(ui, s, changed) {
                                    req = Some(SettingsRequest::ChooseWorkspace);
                                }
                            }
                            Page::Calibration => {
                                calibration(ui, s, alg, changed, patch, &mut self.pending);
                            }
                            Page::Tracking => {
                                if tracking(ui, s, alg, changed, patch, &mut self.pending) {
                                    req = Some(SettingsRequest::StatusColors);
                                }
                            }
                        });
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                    form::button(ui, false, "Reset to Default", 130.0);
                });
            });
        });
        // A number edit ends when nothing is dragged and no field is being typed in, or when
        // the dialog closes or changes page.
        let editing =
            ui.ctx().input(|i| i.pointer.any_down()) || ui.ctx().egui_wants_keyboard_input();
        if !editing || req.is_some() {
            self.flush(patch);
        }
        req
    }

    /// Returns the pending number edit, if any, in `patch`.
    pub fn flush(&mut self, patch: &mut SettingsPatch) {
        if let Some((name, v)) = self.pending.take() {
            patch.set(name, v);
        }
    }
}

/// A check box showing an algorithm setting; a change goes into `patch` under `name`.
fn algorithm_check(ui: &mut Ui, v: bool, text: &str, name: &str, patch: &mut SettingsPatch) {
    if let Some(v) = form::check_box(ui, true, v, text) {
        patch.set(name, SettingValue::Bool(v));
    }
}

/// The value a number field shows: the pending edit of that setting, or the stored value.
fn shown(
    pending: Option<&(&'static str, SettingValue)>,
    name: &str,
    stored: SettingValue,
) -> SettingValue {
    match pending {
        Some((n, v)) if *n == name => *v,
        _ => stored,
    }
}

/// A check box bound to a setting.
fn setting_check(ui: &mut Ui, v: &mut bool, text: &str, changed: &mut bool) {
    if ui.checkbox(v, text).changed() {
        *changed = true;
    }
}

/// The General page; returns true when "..." is clicked.
fn general(ui: &mut Ui, s: &mut UiSettings, changed: &mut bool) -> bool {
    let mut choose = false;
    setting_check(
        ui,
        &mut s.auto_confirm_pending_changes,
        "Automatically update data when changes are pending without asking",
        changed,
    );
    setting_check(
        ui,
        &mut s.confirm_quit_xmalab,
        "Confirmation required to quit RustXMALab",
        changed,
    );
    if ui
        .checkbox(
            &mut s.custom_workspace_path,
            "Use custom workspace to save data imported from XMA Portal",
        )
        .changed()
    {
        // on_checkBox_Workspace_stateChanged: unchecking clears the path.
        if !s.custom_workspace_path {
            s.workspace_path.clear();
        }
        *changed = true;
    }
    ui.horizontal(|ui| {
        let l = ui.label("Workspace path").id;
        let en = s.custom_workspace_path;
        let mut text = s.workspace_path.clone();
        ui.add_enabled(
            en,
            egui::TextEdit::singleline(&mut text)
                .interactive(false)
                .desired_width(ui.available_width() - 40.0),
        )
        .labelled_by(l);
        if ui
            .add_enabled(en, egui::Button::new("..."))
            .accessible_name("Choose workspace path")
            .clicked()
        {
            choose = true;
        }
    });
    ui.horizontal(|ui| {
        ui.label("Number of frames to advance when '5' or '6' is clicked");
        if let Some(v) = form::spin_box(
            ui,
            true,
            i64::from(s.frame_advance),
            1,
            99,
            50.0,
            "Number of frames to advance",
        ) {
            s.frame_advance = v as i32;
            *changed = true;
        }
    });
    setting_check(
        ui,
        &mut s.export_all_enabled,
        "Enable Option ' Export all for external (requires restart of RustXMALab)",
        changed,
    );
    setting_check(
        ui,
        &mut s.disable_image_search,
        "Disable searching and relinking for images when they are not found at startup",
        changed,
    );
    setting_check(
        ui,
        &mut s.recompute_when_saving,
        "Update trials when saving if they require recomputation",
        changed,
    );
    // Not in SettingsDialog.ui: the autosave interval (parity ledger U54), after the last
    // ported control of the page, below a separator so that the ported page is unchanged.
    ui.separator();
    ui.horizontal(|ui| {
        ui.label("Autosave a recovery copy of a changed dataset every");
        if let Some(v) = form::spin_box(
            ui,
            true,
            i64::from(s.autosave_minutes),
            0,
            120,
            50.0,
            "Autosave interval in minutes",
        ) {
            s.autosave_minutes = u32::try_from(v).unwrap_or(0);
            *changed = true;
        }
        ui.label("minutes (0: off)");
    });
    choose
}

/// Holds `v`, the value being edited in the field of `name`, in `pending`. A value another
/// field still holds there is moved into `patch` first, so that it is not lost when the user
/// goes straight from one number field to another.
fn hold(
    pending: &mut Option<(&'static str, SettingValue)>,
    patch: &mut SettingsPatch,
    name: &'static str,
    v: SettingValue,
) {
    if let Some((other, old)) = pending.take_if(|(n, _)| *n != name) {
        patch.set(other, old);
    }
    *pending = Some((name, v));
}

/// A number field of an algorithm setting, written when its edit ends (`pending`).
fn algorithm_int(
    ui: &mut Ui,
    name: &'static str,
    stored: i64,
    label: &str,
    pending: &mut Option<(&'static str, SettingValue)>,
    patch: &mut SettingsPatch,
) {
    let v = match shown(pending.as_ref(), name, SettingValue::Int(stored)) {
        SettingValue::Int(v) => v,
        _ => stored,
    };
    if let Some(v) = form::spin_box(ui, true, v, 0, 99, 50.0, label) {
        hold(pending, patch, name, SettingValue::Int(v));
    }
}

fn calibration(
    ui: &mut Ui,
    s: &mut UiSettings,
    alg: &AlgorithmSettings,
    changed: &mut bool,
    patch: &mut SettingsPatch,
    pending: &mut Option<(&'static str, SettingValue)>,
) {
    setting_check(
        ui,
        &mut s.auto_calib_after_reference,
        "Automatically run calibration after selection of last reference point",
        changed,
    );
    ui.horizontal(|ui| {
        let l = ui.label("Use Detection Method when Alt key is held:").id;
        let idx = usize::try_from(alg.detection_method_for_calibration.code())
            .unwrap_or(0)
            .min(DETECTION_ITEMS.len() - 1);
        // on_comboBox_DetectionMethodForCalibration_currentIndexChanged: the item index.
        let r = ui.scope(|ui| detection_combo(ui, idx, l));
        let chosen = r.inner;
        r.response
            .on_hover_text("Select the computer vision algorithm used to snap to the marker.");
        if let Some(i) = chosen
            && i != idx
        {
            patch.set("DetectionMethodForCalibration", SettingValue::Int(i as i64));
        }
    });
    // on_spinBox_IdentificationThreshold_valueChanged, on_spinBox_OutlierThreshold_valueChanged.
    ui.horizontal(|ui| {
        let label = "Distance threshold during automatic point identification";
        ui.label(label);
        algorithm_int(
            ui,
            "IdentificationThresholdCalibration",
            i64::from(alg.identification_threshold_calibration),
            label,
            pending,
            patch,
        );
    });
    ui.horizontal(|ui| {
        let label = "Reprojection error threshold for a considering point as outlier";
        ui.label(label);
        algorithm_int(
            ui,
            "OutlierThresholdForCalibration",
            i64::from(alg.outlier_threshold_for_calibration),
            label,
            pending,
            patch,
        );
    });
    setting_check(
        ui,
        &mut s.hide_warnings_during_calibration,
        "do not show warnings during calibration",
        changed,
    );
    setting_check(
        ui,
        &mut s.show_advanced_calibration,
        "show advanced options",
        changed,
    );
    // on_checkBox_DisableCheckerboardDetection_stateChanged: read when the calibration page
    // changes camera, frame or workspace (`checkerboardManual`).
    algorithm_check(
        ui,
        alg.disable_checkerboard_detection,
        "Disable automatic Checkerboard detection",
        "DisableCheckerboardDetection",
        patch,
    );
    // on_checkBox_DisableCheckerboardRefinement_stateChanged: read by the manual corners.
    algorithm_check(
        ui,
        alg.disable_checkerboard_refinement,
        "Disable corner refinement during manual checkerboard placement",
        "DisableCheckerboardRefinement",
        patch,
    );
    // on_radioButton_Checkerboard{No,X,Y}Invert_clicked: 0, 1, 2; read when a board is built
    // (a new dataset, or a project opened).
    ui.horizontal(|ui| {
        ui.label("Invert an axis during checkerboard calibration so that z is up");
        let a = alg.checkerboard_inverted_axis;
        for (value, text, checked) in [
            (0, "none", a != 1 && a != 2),
            (1, "x", a == 1),
            (2, "y", a == 2),
        ] {
            if form::radio(ui, true, checked, text) && !checked {
                patch.set("CheckerboadInvertedAxis", SettingValue::Int(value));
            }
        }
    });
    // on_checkBox_FixPrincipal_clicked.
    algorithm_check(
        ui,
        alg.fix_principal,
        "Fix principal at image center during calibration",
        "FixPrincipal",
        patch,
    );
}

/// The Tracking page; returns true when "Set marker status colors" is clicked.
fn tracking(
    ui: &mut Ui,
    s: &mut UiSettings,
    alg: &AlgorithmSettings,
    changed: &mut bool,
    patch: &mut SettingsPatch,
    pending: &mut Option<(&'static str, SettingValue)>,
) -> bool {
    setting_check(
        ui,
        &mut s.center_detail_view,
        "Always center detail window",
        changed,
    );
    setting_check(
        ui,
        &mut s.advanced_crosshair_detail_view,
        "Show an advanced crosshair in the detail window",
        changed,
    );
    setting_check(
        ui,
        &mut s.draw_projected_2d_positions_for_all_points,
        "Draw projected 2d positions for all points",
        changed,
    );
    setting_check(
        ui,
        &mut s.show_epi_line_detail_view,
        "Show the epipolar line in the detail view",
        changed,
    );
    setting_check(
        ui,
        &mut s.show_3d_point_detail_view,
        "Show the 3d reconstructed point in the detail window",
        changed,
    );
    ui.horizontal(|ui| {
        ui.label("Precision for the drawing of an epipolar line");
        if let Some(v) = form::spin_box(
            ui,
            true,
            i64::from(s.epipolar_line_precision),
            0,
            999,
            50.0,
            "Precision for the drawing of an epipolar line",
        ) {
            s.epipolar_line_precision = v as i32;
            *changed = true;
        }
    });
    // E73: SettingsDialog::on_comboBox_TriangulationMethod_currentIndexChanged.
    ui.horizontal(|ui| {
        let l = ui.label("Triangulation method").id;
        if let Some(i) = form::labelled_combo(
            ui,
            true,
            l,
            "comboBox_TriangulationMethod",
            usize::from(alg.triangulation_method).min(4),
            &TRIANGULATION_ITEMS,
            None,
        ) {
            patch.set("TriangulationMethod", SettingValue::Int(i as i64));
        }
    });
    // E77: on_doubleSpinBox_MaxError_valueChanged, which stores the value as a C++ float.
    ui.horizontal(|ui| {
        ui.label("Maximum reprojection error before point is marked lost");
        let name = "MaximumReprojectionError";
        let v = match shown(
            pending.as_ref(),
            name,
            SettingValue::Float(alg.maximum_reprojection_error),
        ) {
            SettingValue::Float(v) => v,
            _ => alg.maximum_reprojection_error,
        };
        if let Some(v) = form::double_spin_box(
            ui,
            true,
            v,
            0.0..=99.99,
            2,
            60.0,
            "Maximum reprojection error before point is marked lost",
        ) {
            hold(
                pending,
                patch,
                name,
                SettingValue::Float(f64::from(v as f32)),
            );
        }
    });
    // E79: on_checkBox_RetrackOptimizedTrackedPoints_stateChanged,
    // on_checkBox_TrackInterpolated_stateChanged.
    algorithm_check(
        ui,
        alg.retrack_optimized_tracked_points,
        "Retrack tracked and optimized points",
        "RetrackOptimizedTrackedPoints",
        patch,
    );
    algorithm_check(
        ui,
        alg.track_interpolated_points,
        "Override interpolated points during tracking",
        "TrackInterpolatedPoints",
        patch,
    );
    setting_check(
        ui,
        &mut s.show_colored_marker_ids,
        "Show Marker ids in the status color of the marker",
        changed,
    );
    setting_check(
        ui,
        &mut s.show_colored_marker_cross,
        "Show Marker Cross in the status color of the marker",
        changed,
    );
    // E74 to E76: on_checkBox_optimize2D_clicked, on_checkBox_DisableRBComputeAdvanced_clicked,
    // on_checkBox_Filter3DPoints_clicked.
    algorithm_check(
        ui,
        alg.optimize_rigid_body,
        "Optimze Rigid Body transformations based on the detected 2d positions",
        "OptimizeRigidBody",
        patch,
    );
    algorithm_check(
        ui,
        alg.disable_rb_compute_advanced,
        "Disable computation of rigid bodies with less than 3 2d correspondences",
        "DisableRBComputeAdvanced",
        patch,
    );
    algorithm_check(
        ui,
        alg.filter_3d_points,
        "Filter3D points instead of transformations. This will disable all advanced features for computation and optimizations of transformations",
        "Filter3DPoints",
        patch,
    );
    // E78: on_spinBox_DefaultMarkerThreshold_valueChanged (the form sets no range: 0 to 99).
    ui.horizontal(|ui| {
        ui.label("Default detection threshold for new markers");
        let name = "DefaultMarkerThreshold";
        let v = match shown(
            pending.as_ref(),
            name,
            SettingValue::Int(i64::from(alg.default_marker_threshold)),
        ) {
            SettingValue::Int(v) => v,
            _ => i64::from(alg.default_marker_threshold),
        };
        if let Some(v) = form::spin_box(
            ui,
            true,
            v,
            0,
            99,
            50.0,
            "Default detection threshold for new markers",
        ) {
            hold(pending, patch, name, SettingValue::Int(v));
        }
    });
    setting_check(
        ui,
        &mut s.show_ids_in_detail,
        "Show marker id's in detail window",
        changed,
    );
    ui.add_space(8.0);
    form::wide_button(ui, true, "Set marker status colors").clicked()
}
