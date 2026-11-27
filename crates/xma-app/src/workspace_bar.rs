//! The workspace bar: workspace, trial, camera visibility and view layout.
//!
//! Ported control by control from `WorkspaceNavigationFrame.ui` and
//! ui/WorkspaceNavigationFrame.cpp. One row, left to right (grid columns of the form):
//!
//! | Col | Control | Text |
//! |---|---|---|
//! | 0 | `label` | "Workspace : " |
//! | 1 | `comboBoxWorkspace` | Undistortion / Calibration / Marker tracking |
//! | 2 | `label_Optimized` | "Camera calibration is not optimized! !" in rgb(170, 0, 0) |
//! | 3 | spacer | |
//! | 4 | `label_Trial` | "Trial : " |
//! | 5 | `comboBoxTrial` | trial names |
//! | 8 | `toolButtonAddTrial` | "+" |
//! | 9 | `toolButtonTrialSettings` | settings.png |
//! | 11 | spacer | |
//! | 13 | `label_2` | "View : " |
//! | 14 | `comboBoxViewspace` | the four layouts |
//! | 15 | `toolButtonCameraSettings` | settings.png |
//!
//! Behaviour:
//! - `setUndistortionCalibration`: without calibration the workspace label and combo are
//!   hidden; with calibration but no undistortion grid, "Undistortion" is removed.
//! - `setTrialVisible`: the trial controls show in Marker tracking when the cameras are
//!   calibrated or the dataset has no calibration.
//! - `workspaceChanged`: the view combo and camera button are enabled except in Marker
//!   tracking without trials; `label_Optimized` shows in Marker tracking unless every
//!   camera is optimized or there is at most one calibration image.
//! - `activeTrialChanged`: trial settings are disabled for the default trial.
//! - The bar is hidden until a dataset is open (`MainWindow.cpp:182,379`).
//!
//! Here the bar sits at the top of the window under the menu bar (PLAN.md section 5, "One
//! workspace bar, at the top"), not inside the central area as in XMALab 3.

use crate::actions::Action;
use crate::icons::Icon;
use crate::panels::Cx;
use crate::state::{DisplayLayout, Workspace};
use xma_model::CalibrationType;

/// The workspaces the combo box offers (`setUndistortionCalibration`).
#[must_use]
pub fn workspace_items(calibration: CalibrationType, has_undistortion: bool) -> Vec<Workspace> {
    if calibration != CalibrationType::None && !has_undistortion {
        vec![Workspace::Calibration, Workspace::MarkerTracking]
    } else {
        Workspace::ALL.to_vec()
    }
}

/// What the bar shows, from the state and the project (the C++ slots, computed).
#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
pub struct BarModel {
    pub workspace_visible: bool,
    pub workspaces: Vec<Workspace>,
    pub optimized_warning: bool,
    pub trial_visible: bool,
    pub trial_settings_enabled: bool,
    pub view_enabled: bool,
}

impl BarModel {
    #[must_use]
    pub fn new(cx: &Cx<'_>) -> Self {
        let f = cx.facts;
        let tracking = cx.state.workspace == Workspace::MarkerTracking;
        let trials = f.n_trials() > 0;
        Self {
            workspace_visible: f.calibration != CalibrationType::None,
            workspaces: workspace_items(f.calibration, f.has_undistortion),
            optimized_warning: tracking && !(f.cameras_optimized || f.nb_images_calibration <= 1),
            trial_visible: tracking && f.calibrated_or_none(),
            trial_settings_enabled: cx
                .state
                .active_trial
                .and_then(|t| f.trial_is_default.get(t))
                .is_some_and(|d| !d),
            view_enabled: !tracking || trials,
        }
    }
}

/// Draws the bar.
pub fn ui(ui: &mut egui::Ui, cx: &mut Cx<'_>) {
    let model = BarModel::new(cx);
    crate::qt::styled_frame(ui, |ui| {
        ui.horizontal(|ui| {
            if model.workspace_visible {
                let l = ui.label("Workspace : ").id;
                let labels: Vec<&str> = model.workspaces.iter().map(|w| w.label()).collect();
                let sel = model
                    .workspaces
                    .iter()
                    .position(|w| *w == cx.state.workspace);
                if let Some(i) =
                    crate::qt::combo_box(ui, "comboBoxWorkspace", sel, &labels, None, Some(l))
                {
                    cx.raise(Action::SetWorkspace(model.workspaces[i]));
                }
            }
            if model.optimized_warning {
                ui.label(
                    egui::RichText::new("Camera calibration is not optimized! !")
                        .color(crate::theme::warning_text(ui.visuals())),
                );
            }
            // The right-hand part, laid out from the right; the trial group sits in the
            // middle of the remaining space (the two expanding spacers of the form).
            let right_w = view_group_width(ui);
            let trial_w = if model.trial_visible {
                trial_group_width(ui, cx)
            } else {
                0.0
            };
            let sp = ui.spacing().item_spacing.x;
            let free = (ui.available_width() - right_w - trial_w).max(0.0);
            if model.trial_visible {
                ui.add_space(free / 2.0);
                trial_group(ui, cx, &model);
                ui.add_space((free / 2.0 - sp).max(0.0));
            } else {
                ui.add_space((free - sp).max(0.0));
            }
            view_group(ui, cx, &model);
        });
    });
}

fn trial_names(cx: &Cx<'_>) -> Vec<String> {
    cx.facts.trial_names.clone()
}

fn trial_group_width(ui: &egui::Ui, cx: &Cx<'_>) -> f32 {
    let font = egui::TextStyle::Button.resolve(ui.style());
    let longest = trial_names(cx)
        .iter()
        .map(|s| {
            ui.painter()
                .layout_no_wrap(s.clone(), font.clone(), egui::Color32::BLACK)
                .size()
                .x
        })
        .fold(0.0_f32, f32::max);
    // Label, combo box (text and arrow), "+" and the settings button, with egui's spacing.
    let sp = ui.spacing();
    let button = 16.0 + 2.0 * sp.button_padding.x;
    54.0 + longest
        + sp.icon_width
        + 2.0 * sp.button_padding.x
        + 2.0 * button
        + 4.0 * sp.item_spacing.x
}

fn view_group_width(ui: &egui::Ui) -> f32 {
    let font = egui::TextStyle::Button.resolve(ui.style());
    let longest = DisplayLayout::ALL
        .iter()
        .map(|d| {
            ui.painter()
                .layout_no_wrap(d.label().to_owned(), font.clone(), egui::Color32::BLACK)
                .size()
                .x
        })
        .fold(0.0_f32, f32::max);
    // Label, combo box (text and arrow) and the settings button, with egui's spacing.
    let sp = ui.spacing();
    let button = 16.0 + 2.0 * sp.button_padding.x;
    44.0 + longest + sp.icon_width + 2.0 * sp.button_padding.x + button + 3.0 * sp.item_spacing.x
}

fn trial_group(ui: &mut egui::Ui, cx: &mut Cx<'_>, model: &BarModel) {
    let l = ui.label("Trial : ").id;
    let names = trial_names(cx);
    let labels: Vec<&str> = names.iter().map(String::as_str).collect();
    if let Some(i) = crate::qt::combo_box(
        ui,
        "comboBoxTrial",
        cx.state.active_trial,
        &labels,
        None,
        Some(l),
    ) {
        cx.raise(Action::SetActiveTrial(i));
    }
    let add = Action::AddTrial;
    let acx_enabled = add.implemented();
    if ui
        .add_enabled(
            acx_enabled,
            egui::Button::new("+").min_size(egui::vec2(22.0, 0.0)),
        )
        .clicked()
    {
        cx.raise(add);
    }
    let settings = Action::TrialSettings;
    let img = cx.icons.image(ui.ctx(), Icon::Settings, 16.0);
    if ui
        .add_enabled(
            model.trial_settings_enabled && settings.implemented(),
            egui::Button::image(img).frame_when_inactive(false),
        )
        .on_hover_text("Trial settings")
        .clicked()
    {
        cx.raise(settings);
    }
}

fn view_group(ui: &mut egui::Ui, cx: &mut Cx<'_>, model: &BarModel) {
    let l = ui.label("View : ").id;
    let labels: Vec<&str> = DisplayLayout::ALL.iter().map(|d| d.label()).collect();
    let sel = DisplayLayout::ALL
        .iter()
        .position(|d| *d == cx.state.display);
    let chosen = ui
        .add_enabled_ui(model.view_enabled, |ui| {
            crate::qt::combo_box(ui, "comboBoxViewspace", sel, &labels, None, Some(l))
        })
        .inner;
    if let Some(i) = chosen {
        cx.raise(Action::SetDisplay(DisplayLayout::ALL[i]));
    }
    let cam = Action::CameraSettings;
    let img = cx.icons.image(ui.ctx(), Icon::Settings, 16.0);
    if ui
        .add_enabled(
            model.view_enabled && cam.implemented(),
            egui::Button::image(img).frame_when_inactive(false),
        )
        .on_hover_text("Camera settings")
        .clicked()
    {
        cx.raise(cam);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn undistortion_removed_without_grid() {
        assert_eq!(
            workspace_items(CalibrationType::Internal, false),
            vec![Workspace::Calibration, Workspace::MarkerTracking]
        );
        assert_eq!(workspace_items(CalibrationType::Internal, true).len(), 3);
    }
}
