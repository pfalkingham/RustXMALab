//! The Toolbox page of the Calibration workspace (cube or checkerboard).
//!
//! Ported control by control from `WizardCalibrationCubeFrame.ui` and
//! ui/WizardCalibrationCubeFrame.cpp (`setDialog`, `loadCalibrationSettings`,
//! `updateFrameList`, `reloadManualPoints`, `setupManualPoints`, the combo handlers).
//! Layout, a styled frame round the page, with the frame list on the right when shown:
//!
//! | Row | Control | Text |
//! |---|---|---|
//! | 0 | `label` | the instruction for the active camera and frame |
//! | 0-7, right | `framelist` › `frametreeWidget` | calibration images with at least one calibrated camera |
//! | 1 | `checkBoxManual`, `checkBoxFrameList` | "manual mode", "frame list" |
//! | 2 | `frameReferences` › `groupBox` "references" | four rows of radio button, tool button (reference index and name), check box, label (clicked position) |
//! | 3 | `frameManual` › `groupBox_2` "manual references" | one row per calibration-object point: radio button, label, check box |
//! | 4 | `frameModifyCalibration` | `groupBoxMouseClickMode` "mouseclick mode" (two hint labels), `groupBoxVisualization` "visualization" (points, image, text), `groupBoxOptimization` "advanced" ("optimize camera setup", "is optimized", "use camera distortion model"), `pushButtonRemoveOutlierAutomatically` "set outlier by distance" |
//! | 6 | `groupBoxDelete` | "Reset Frame", "Reset Camera", "Delete Frame" |
//! | 7 | `pushButton` | "compute calibration" / "recompute calibration" |
//!
//! Which parts show (`setDialog`) depends on the undistortion state, whether the active
//! camera's active calibration image is calibrated, manual mode, the calibration object
//! (cube or checkerboard) and the settings `ShowAdvancedCalibration` and
//! `DisableCheckerboardDetection`. The page is shown only for an internal calibration
//! (`WizardDockWidget::workspaceChanged`).
//!
//! The visualization combos change the display choices in [`AppState`] (C31); "frame
//! list" and a click in it are the page's own (C30). Every other control raises
//! `Action::Calib` (`crate::calibration_ui`, carried out in `app/calibration.rs`), or
//! `Action::OptimizeCameraSetup` (C37): the reference radio and tool buttons (C13, C14),
//! "manual mode" and its rows (C25, C26, C28), "is optimized" (C36), "use camera
//! distortion model" (C35), "Reset Frame", "Reset Camera", "Delete Frame" (C32 to C34) and
//! the main button (C17, C18, C22, C29). What the reference rows and the manual rows show
//! (clicked positions, the checked radio button, manual mode) is the engine's session state
//! (`xma_engine::Session`); the clicks that set it are in
//! `camera_view/overlay_calibration.rs`.

use egui::Color32;
use xma_model::{CalibrationObject, CalibrationType};

use crate::actions::Action;
use crate::calibration_ui::CalibAction;
use crate::dialogs::form;
use crate::panels::Cx;
use crate::state::{
    AppState, CalibrationVisImage, CalibrationVisPoints, CalibrationVisText, Changes,
    UndistortionState,
};

/// Items of `comboBoxPoints`.
pub const POINTS_ITEMS: [&str; 6] = [
    "none",
    "all detected points(distorted)",
    "detected cube points (distorted)",
    "projected cube points (distorted)",
    "detected cube points (undistorted)",
    "projected cube points (undistorted)",
];

/// Items of `comboBoxImage`.
pub const IMAGE_ITEMS: [&str; 2] = ["distorted", "undistorted"];

/// Items of `comboBoxText`.
pub const TEXT_ITEMS: [&str; 5] = ["none", "id", "error", "error (coloredcoded)", "markerNames"];

const POINTS: [CalibrationVisPoints; 6] = [
    CalibrationVisPoints::None,
    CalibrationVisPoints::DetectedAll,
    CalibrationVisPoints::DetectedCubeDistorted,
    CalibrationVisPoints::ProjectedCubeDistorted,
    CalibrationVisPoints::DetectedCubeUndistorted,
    CalibrationVisPoints::ProjectedCubeUndistorted,
];

const TEXTS: [CalibrationVisText; 5] = [
    CalibrationVisText::None,
    CalibrationVisText::Id,
    CalibrationVisText::Error,
    CalibrationVisText::ErrorColor,
    CalibrationVisText::MarkerName,
];

fn points_index(p: CalibrationVisPoints) -> usize {
    POINTS.iter().position(|x| *x == p).unwrap_or(0)
}

fn image_index(i: CalibrationVisImage) -> usize {
    usize::from(i == CalibrationVisImage::Undistorted)
}

fn text_index(t: CalibrationVisText) -> usize {
    TEXTS.iter().position(|x| *x == t).unwrap_or(0)
}

/// Ported from ui/WizardCalibrationCubeFrame.cpp, on_comboBoxImage_currentIndexChanged:
/// cube points follow the image into its space.
pub fn choose_image(state: &mut AppState, idx: usize) -> Changes {
    let mut ch = Changes::NONE;
    let pts = points_index(state.calibration_vis_points);
    if idx == 0 {
        if pts == 4 {
            ch |= choose_points(state, 2);
        } else if pts == 5 {
            ch |= choose_points(state, 3);
        }
    } else if idx == 1 {
        if pts == 2 {
            ch |= choose_points(state, 4);
        } else if pts == 3 {
            ch |= choose_points(state, 5);
        }
    }
    let img = if idx == 1 {
        CalibrationVisImage::Undistorted
    } else {
        CalibrationVisImage::Distorted
    };
    ch |= state.change_calibration_vis_image(img, false);
    ch
}

/// Ported from ui/WizardCalibrationCubeFrame.cpp, on_comboBoxPoints_currentIndexChanged:
/// the image follows the points into their space.
pub fn choose_points(state: &mut AppState, idx: usize) -> Changes {
    let mut ch = Changes::NONE;
    let img = image_index(state.calibration_vis_image);
    if (1..=3).contains(&idx) && img == 1 {
        ch |= state.change_calibration_vis_image(CalibrationVisImage::Distorted, false);
    } else if (idx == 4 || idx == 5) && img == 0 {
        ch |= state.change_calibration_vis_image(CalibrationVisImage::Undistorted, false);
    }
    ch |= state.change_calibration_vis_points(POINTS[idx.min(5)], false);
    ch
}

/// Hint for a correspondence click: COMMAND on macOS, Ctrl elsewhere. The C++ says
/// COMMAND on every system but Windows, so Linux users were told the wrong key.
#[must_use]
pub fn correspondence_hint() -> &'static str {
    if cfg!(target_os = "macos") {
        "Add a correspondance by using COMMAND+click."
    } else {
        "Add a correspondance by using Ctrl+click."
    }
}

/// The colours of the frame list (`updateFrameList`): calibrated, set from other frames.
const LIST_CALIBRATED: Color32 = Color32::from_rgb(0, 200, 100);
const LIST_FROM_OTHER: Color32 = Color32::from_rgb(0, 100, 200);

/// What `setDialog` shows, worked out from the document and the page's own state.
#[derive(Clone, Debug, Default, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct PageView {
    pub label: Option<&'static str>,
    pub manual_check: bool,
    pub frame_list: bool,
    pub references: bool,
    pub manual: bool,
    pub modify: bool,
    pub optimization: bool,
    pub distortion_check: bool,
    pub reset_buttons: bool,
    pub push_button: Option<&'static str>,
}

/// The page's own state (the C++ widgets' state that is not in [`AppState`] or in the
/// engine's session).
#[derive(Debug, Default)]
pub struct CalibrationPage {
    /// `checkBoxFrameList`.
    pub frame_list: bool,
    /// The label keeps its text when `setDialog` does not set one (manual mode).
    label: &'static str,
    /// The constructor's `comboBoxPoints->setCurrentIndex(2)` has been applied.
    constructed: bool,
}

impl CalibrationPage {
    /// The constructor's effect on the state: points "detected cube points (distorted)".
    pub fn construct(&mut self, cx: &mut Cx<'_>) {
        if !self.constructed {
            self.constructed = true;
            cx.change(|s| choose_points(s, 2));
        }
    }

    /// Ported from ui/WizardCalibrationCubeFrame.cpp, setDialog: which parts are shown.
    #[must_use]
    pub fn view(&self, cx: &Cx<'_>) -> PageView {
        let mut v = PageView::default();
        if cx.state.undistortion == UndistortionState::NotUndistorted {
            v.label = Some("You first have to perform an undistortion!");
            return v;
        }
        v.frame_list = self.frame_list;
        let (Some(doc), Some(cam), Some(frame)) = (
            cx.doc,
            cx.state.active_camera,
            cx.state.active_frame_calibration,
        ) else {
            return v;
        };
        let p = doc.project();
        let Some(camera) = p.cameras.get(cam) else {
            return v;
        };
        // `checkBoxManual` (C25) is the session's manual mode.
        let manual = doc.session().manual_mode;
        v.optimization = true;
        v.distortion_check = cx.settings.show_advanced_calibration;
        v.manual_check = true;
        let checkerboard = matches!(
            p.calibration_object,
            Some(CalibrationObject::Checkerboard { .. })
        );
        // `checkerboardManual`: the setting, or a failed detection (the engine's session).
        let checkerboard_manual = doc.checkerboard_manual();
        let calibrated = camera
            .calibration_images
            .get(frame)
            .is_some_and(|im| im.calibrated == 1);
        if calibrated {
            v.manual = manual;
            v.label = Some("Modify calibration");
            v.modify = true;
            v.reset_buttons = true;
            v.push_button = Some("recompute calibration");
        } else {
            if manual {
                v.manual = true;
            } else if !checkerboard {
                v.label = Some("Select the references");
                v.references = true;
            } else if checkerboard_manual {
                v.label = Some(
                    "Click on the four interior corners of the checkerboard. Start with the upper left corner of the chessboard",
                );
                v.references = true;
            } else {
                v.label = Some("Click on the interior upper left corner of the chessboard");
            }
            v.push_button = Some("compute calibration");
        }
        v
    }

    /// Draws the page.
    pub fn ui(&mut self, ui: &mut egui::Ui, cx: &mut Cx<'_>) {
        self.construct(cx);
        if cx.facts.calibration != CalibrationType::Internal {
            return;
        }
        let v = self.view(cx);
        if let Some(l) = v.label {
            self.label = l;
        }
        form::styled_frame(ui, |ui| {
            ui.horizontal_top(|ui| {
                let list_w = if v.frame_list {
                    60.0 + 14.0 * cx.facts.n_cameras as f32
                } else {
                    0.0
                };
                let left_w = (ui.available_width() - list_w - 8.0).max(120.0);
                ui.vertical(|ui| {
                    ui.set_width(left_w);
                    self.left_column(ui, cx, &v);
                });
                if v.frame_list {
                    ui.vertical(|ui| {
                        ui.set_width(list_w);
                        frame_list(ui, cx);
                    });
                }
            });
        });
    }

    fn left_column(&mut self, ui: &mut egui::Ui, cx: &mut Cx<'_>, v: &PageView) {
        ui.vertical_centered(|ui| {
            ui.add(egui::Label::new(self.label).wrap());
        });
        let manual = cx.doc.is_some_and(|d| d.session().manual_mode);
        ui.horizontal(|ui| {
            if v.manual_check
                && let Some(m) = form::check_box(ui, true, manual, "manual mode")
            {
                cx.raise(Action::Calib(CalibAction::ManualMode(m)));
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if let Some(f) = form::check_box(ui, true, self.frame_list, "frame list") {
                    self.frame_list = f;
                }
            });
        });
        if v.references {
            references(ui, cx);
        }
        if v.manual {
            manual_references(ui, cx);
        }
        if v.modify {
            form::group_box(ui, "mouseclick mode", |ui| {
                ui.vertical_centered(|ui| {
                    ui.label("Toggle Outlier by double clicking on a point.");
                    ui.label(correspondence_hint());
                });
            });
            form::group_box(ui, "visualization", |ui| {
                visualization(ui, cx);
            });
            if v.optimization {
                form::group_box(ui, "advanced", |ui| {
                    ui.horizontal_wrapped(|ui| {
                        let w = (ui.available_width() - 130.0).max(120.0);
                        // C37: always enabled where shown (the dialog is another package's).
                        if form::button(ui, true, "optimize camera setup", w).clicked() {
                            cx.raise(Action::OptimizeCameraSetup);
                        }
                        // C36: checked and enabled only when every camera is optimized;
                        // unchecking asks, then removes the optimization.
                        let opt = cx.facts.cameras_optimized;
                        if form::check_box(ui, opt, opt, "is optimized") == Some(false) {
                            cx.raise(Action::Calib(CalibAction::RemoveOptimization));
                        }
                    });
                    if v.distortion_check {
                        let has = cx
                            .doc
                            .zip(cx.state.active_camera)
                            .and_then(|(d, c)| d.project().cameras.get(c))
                            .is_some_and(|c| c.has_model_distortion);
                        // C35.
                        if let Some(on) =
                            form::check_box(ui, true, has, "use camera distortion model")
                        {
                            cx.raise(Action::Calib(CalibAction::LensModel(on)));
                        }
                    }
                });
            }
        }
        form::group_box(ui, "", |ui| {
            ui.horizontal_wrapped(|ui| {
                let n = if v.reset_buttons { 3.0 } else { 1.0 };
                let gap = ui.spacing().item_spacing.x;
                let w = ((ui.available_width() - gap * (n - 1.0)) / n - 1.0).max(60.0);
                if v.reset_buttons {
                    if form::button(ui, true, "Reset Frame", w).clicked() {
                        cx.raise(Action::Calib(CalibAction::ResetFrame));
                    }
                    if form::button(ui, true, "Reset Camera", w).clicked() {
                        cx.raise(Action::Calib(CalibAction::ResetCamera));
                    }
                }
                if form::button(ui, true, "Delete Frame", w).clicked() {
                    cx.raise(Action::Calib(CalibAction::DeleteFrame));
                }
            });
        });
        if let Some(t) = v.push_button
            && form::wide_button(ui, true, t).clicked()
        {
            cx.raise(Action::Calib(CalibAction::Compute));
        }
    }
}

/// What one row of `groupBox` "references" shows.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceRow {
    /// `radioButtonReferenceN` checked: the next click sets this reference (C13).
    pub checked: bool,
    /// `toolButtonReferenceN`: "<index>  <name>", or "<index>  CUSTOM INDEX" (C14).
    pub tool_text: String,
    /// `checkBoxReferenceN` (disabled): the reference has been clicked.
    pub set: bool,
    /// `labelReferenceN`: the clicked position, "x / y" with 2 decimals (C15).
    pub position: String,
}

/// The four rows of `groupBox` "references" (`loadCalibrationSettings`,
/// `on_toolButtonReferenceN_clicked`, `addCalibrationReference`, `resetReferences`), from the
/// session's references. With fewer than four reference ids in the file the tool buttons
/// keep no text, as in XMALab.
#[must_use]
pub fn reference_rows(doc: &xma_engine::Document) -> Vec<ReferenceRow> {
    let names: Option<&[String]> = match doc.project().calibration_object.as_ref() {
        Some(CalibrationObject::Frame(f)) if f.reference_ids.len() >= 4 => Some(&f.reference_names),
        _ => None,
    };
    let s = doc.session();
    s.references
        .iter()
        .enumerate()
        .map(|(i, r)| ReferenceRow {
            checked: s.next_reference == i,
            tool_text: match names {
                _ if r.custom => format!("{}  CUSTOM INDEX", r.point_id + 1),
                Some(n) => format!(
                    "{}  {}",
                    r.point_id + 1,
                    n.get(i).map_or("", String::as_str)
                ),
                None => String::new(),
            },
            set: r.position.is_some(),
            position: r
                .position
                .map(|(x, y)| format!("{x:.2} / {y:.2}"))
                .unwrap_or_default(),
        })
        .collect()
}

/// `groupBox` "references": the four reference beads of a cube.
fn references(ui: &mut egui::Ui, cx: &mut Cx<'_>) {
    let rows = cx.doc.map(reference_rows).unwrap_or_default();
    form::group_box(ui, "references", |ui| {
        egui::Grid::new("calib_references")
            .num_columns(4)
            .show(ui, |ui| {
                for (i, row) in rows.iter().enumerate() {
                    let n = i + 1;
                    if ui
                        .add(egui::RadioButton::new(row.checked, ""))
                        .accessible_name(format!("Reference {n}"))
                        .clicked()
                    {
                        cx.raise(Action::Calib(CalibAction::ChooseReference(i)));
                    }
                    if ui
                        .add(egui::Button::new(row.tool_text.as_str()).frame_when_inactive(false))
                        .accessible_name(format!("Reference {n} index"))
                        .clicked()
                    {
                        cx.raise(Action::Calib(CalibAction::ReferenceIndex(i)));
                    }
                    let mut c = row.set;
                    ui.add_enabled(false, egui::Checkbox::without_text(&mut c))
                        .accessible_name(format!("Reference {n} set"));
                    ui.label(row.position.as_str())
                        .accessible_name(format!("Reference {n} position"));
                    ui.end_row();
                }
            });
    });
}

/// `groupBox_2` "manual references" (`setupManualPoints`, `reloadManualPoints`): one row
/// per calibration-object point with its detected position and inlier flag in the active
/// image.
fn manual_references(ui: &mut egui::Ui, cx: &mut Cx<'_>) {
    let Some(doc) = cx.doc else { return };
    let p = doc.project();
    // `setupManualPoints`: one row per point of the cube or the checkerboard, named
    // "<i+1> <name>" (a board's corner "x , y").
    let Some(object) = &p.calibration_object else {
        form::group_box(ui, "manual references", |_| {});
        return;
    };
    let names = object.point_names();
    // The lists as XMALab holds them: for an image manual mode has just prepared, n unset
    // points, which the project receives with the first manual edit (ledger A164).
    let image = cx
        .state
        .active_camera
        .zip(cx.state.active_frame_calibration)
        .and_then(|(c, fr)| doc.calibration_image_lists(c, fr));
    let image = image.as_deref();
    let checked_point = doc.session().manual_point;
    let n = object.n_points();
    let inliers: Vec<bool> = (0..n)
        .map(|i| {
            image
                .and_then(|im| im.inlier.get(i))
                .is_some_and(|v| *v > 0)
        })
        .collect();
    let mut raise = None;
    form::group_box(ui, "manual references", |ui| {
        egui::ScrollArea::vertical()
            .id_salt("calib_manual_scroll")
            .max_height(260.0)
            .show(ui, |ui| {
                egui::Grid::new("calib_manual")
                    .num_columns(3)
                    .show(ui, |ui| {
                        for i in 0..n {
                            let name = names.get(i).map_or("", String::as_str);
                            let text = format!("{} {name}", i + 1);
                            // C26.
                            if ui
                                .add(egui::RadioButton::new(i == checked_point, text))
                                .clicked()
                            {
                                raise = Some(CalibAction::ManualPoint(i));
                            }
                            let pos = image
                                .and_then(|im| im.detected.get(i))
                                .map(|pt| format!("{} / {}", form::qnum(pt.x), form::qnum(pt.y)))
                                .unwrap_or_default();
                            ui.label(pos);
                            // C28: every check box's state is written.
                            let mut inlier = inliers[i];
                            if ui
                                .add(egui::Checkbox::without_text(&mut inlier))
                                .accessible_name(format!("Point {} inlier", i + 1))
                                .changed()
                            {
                                let mut all = inliers.clone();
                                all[i] = inlier;
                                raise = Some(CalibAction::ManualInliers(all));
                            }
                            ui.end_row();
                        }
                    });
            });
    });
    if let Some(a) = raise {
        cx.raise(Action::Calib(a));
    }
}

/// `groupBoxVisualization`: points, image, text.
fn visualization(ui: &mut egui::Ui, cx: &mut Cx<'_>) {
    let w = (ui.available_width() - 70.0).max(120.0);
    egui::Grid::new("calib_vis").num_columns(2).show(ui, |ui| {
        let l = ui.label("points   ").id;
        let sel = points_index(cx.state.calibration_vis_points);
        if let Some(i) = form::labelled_combo(
            ui,
            true,
            l,
            "calib_comboBoxPoints",
            sel,
            &POINTS_ITEMS,
            Some(w),
        ) {
            cx.change(|s| choose_points(s, i));
        }
        ui.end_row();
        let l = ui.label("image").id;
        let sel = image_index(cx.state.calibration_vis_image);
        if let Some(i) = form::labelled_combo(
            ui,
            true,
            l,
            "calib_comboBoxImage",
            sel,
            &IMAGE_ITEMS,
            Some(w),
        ) {
            cx.change(|s| choose_image(s, i));
        }
        ui.end_row();
        let l = ui.label("text").id;
        let sel = text_index(cx.state.calibration_vis_text);
        if let Some(i) =
            form::labelled_combo(ui, true, l, "calib_comboBoxText", sel, &TEXT_ITEMS, Some(w))
        {
            cx.change(|s| s.change_calibration_vis_text(TEXTS[i.min(4)], false));
        }
        ui.end_row();
    });
}

/// One row of the frame list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameListRow {
    /// Calibration image, 0-based.
    pub image: usize,
    /// Per camera: the `isCalibrated` value (1 calibrated, 2 set from other frames).
    pub cameras: Vec<i32>,
}

/// Ported from ui/WizardCalibrationCubeFrame.cpp, updateFrameList: the calibration images
/// in which at least one camera is calibrated.
#[must_use]
pub fn frame_list_rows(doc: &xma_engine::Document) -> Vec<FrameListRow> {
    let p = doc.project();
    let n = p.cameras.first().map_or(0, |c| c.calibration_images.len());
    (0..n)
        .filter_map(|i| {
            let cameras: Vec<i32> = p
                .cameras
                .iter()
                .map(|c| c.calibration_images.get(i).map_or(0, |im| im.calibrated))
                .collect();
            cameras
                .iter()
                .any(|c| *c != 0)
                .then_some(FrameListRow { image: i, cameras })
        })
        .collect()
}

/// `framelist`: a click on a row makes its image the active one.
fn frame_list(ui: &mut egui::Ui, cx: &mut Cx<'_>) {
    let Some(doc) = cx.doc else { return };
    let rows = frame_list_rows(doc);
    let active = cx.state.active_frame_calibration;
    egui::Frame::NONE
        .fill(ui.visuals().extreme_bg_color)
        .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
        .inner_margin(egui::Margin::same(2))
        .show(ui, |ui| {
            ui.set_min_height(160.0);
            ui.spacing_mut().item_spacing.y = 0.0;
            for r in rows {
                let text = (r.image + 1).to_string();
                let w = ui.available_width();
                let (rect, resp) =
                    ui.allocate_exact_size(egui::vec2(w, 18.0), egui::Sense::click());
                let name = format!("Calibration frame {text}");
                resp.widget_info(|| {
                    egui::WidgetInfo::selected(
                        egui::Role::ListItem,
                        true,
                        active == Some(r.image),
                        &name,
                    )
                });
                if active == Some(r.image) {
                    ui.painter()
                        .rect_filled(rect, 0.0, ui.visuals().selection.bg_fill);
                }
                let font = egui::TextStyle::Body.resolve(ui.style());
                ui.painter().text(
                    rect.left_center() + egui::vec2(4.0, 0.0),
                    egui::Align2::LEFT_CENTER,
                    &text,
                    font,
                    ui.visuals().text_color(),
                );
                let mut x = rect.max.x - 14.0 * r.cameras.len() as f32;
                for c in &r.cameras {
                    let color = match c {
                        1 => Some(LIST_CALIBRATED),
                        2 => Some(LIST_FROM_OTHER),
                        _ => None,
                    };
                    if let Some(col) = color {
                        let sw = egui::Rect::from_min_size(
                            egui::pos2(x + 3.0, rect.min.y + 1.0),
                            egui::vec2(8.0, 16.0),
                        );
                        ui.painter().rect_filled(sw, 0.0, col);
                    }
                    x += 14.0;
                }
                if resp.clicked() {
                    // frametreeWidgetitemClicked.
                    cx.change(|s| s.change_active_frame_calibration(Some(r.image), false));
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combos_keep_each_other_consistent() {
        let mut s = AppState::default();
        choose_points(&mut s, 2);
        assert_eq!(
            s.calibration_vis_points,
            CalibrationVisPoints::DetectedCubeDistorted
        );
        choose_image(&mut s, 1);
        assert_eq!(
            s.calibration_vis_points,
            CalibrationVisPoints::DetectedCubeUndistorted
        );
        assert_eq!(s.calibration_vis_image, CalibrationVisImage::Undistorted);
        choose_points(&mut s, 3);
        assert_eq!(s.calibration_vis_image, CalibrationVisImage::Distorted);
        choose_image(&mut s, 1);
        assert_eq!(
            s.calibration_vis_points,
            CalibrationVisPoints::ProjectedCubeUndistorted
        );
    }
}
