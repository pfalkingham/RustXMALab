//! The Toolbox page of the Undistortion workspace.
//!
//! Ported control by control from `WizardUndistortionFrame.ui` and
//! ui/WizardUndistortionFrame.cpp. Layout (one column, a styled frame round the page):
//!
//! | Row | Control | Text |
//! |---|---|---|
//! | 0 | `label` | "You first need to perform an Undistortion" / "Modify Undistortion" |
//! | 1 | `frameModifyUndistortion` › `groupBoxMouseClickMode` "mouseclick mode" | radio buttons `radioButtonMouseClickNone` "None", `radioButtonMouseClickOutlier` "toggle outlier", `radioButtonMouseClickCenter` "set center" |
//! | 1 | `frameModifyUndistortion` › `groupBoxVisualization` "visualization" | `label_2` "points   " with `comboBoxPoints`, `label_3` "image" with `comboBoxImage` |
//! | 1 | `frameModifyUndistortion` › `pushButtonRemoveOutlierAutomatically` | "remove outlier at border" |
//! | 2 | `pushButton` | "compute undistortion" / "recompute undistortion" |
//!
//! Before an undistortion (`undistortionChanged(NOTUNDISTORTED)`) the "toggle outlier"
//! radio button, the visualization group and the outlier button are hidden. Once undistorted
//! they show, the image combo is set to "undistorted" and the points combo to "grid points
//! (undistorted)", and "None" is checked.
//!
//! The visualization combos change [`AppState`]'s undistortion display choices and keep
//! each other consistent as the C++ handlers do (C9, C10). The mouse-mode radio buttons
//! (C3 to C5, "set center" asks first once undistorted), "remove outlier at border" (C8,
//! its two prompts) and the main button (C1, C2) raise `Action::Calib` or the
//! `CalibrationEdit` (`crate::calibration_ui`, `app/calibration.rs`); the clicks of the
//! camera views in each mouse mode are in `camera_view/overlay_undistortion.rs`.

use crate::actions::Action;
use crate::calibration_ui::CalibAction;
use crate::dialogs::form;
use crate::editing::EditRequest;
use crate::panels::Cx;
use crate::state::{
    AppState, Changes, UndistortionMouseMode, UndistortionState, UndistortionVisImage,
    UndistortionVisPoints,
};

/// Items of `comboBoxPoints`.
pub const POINTS_ITEMS: [&str; 5] = [
    "none",
    "detected (distorted)",
    "grid points (distorted)",
    "grid points (undistorted)",
    "reference points (undistorted)",
];

/// Items of `comboBoxImage`.
pub const IMAGE_ITEMS: [&str; 7] = [
    "distorted",
    "undistorted",
    "distortion map (scaled angles)",
    "distortion map (angles)",
    "distortion map (distance)",
    "distortion map (horizontal displacement)",
    "distortion map (vertical displacement)",
];

const POINTS: [UndistortionVisPoints; 5] = [
    UndistortionVisPoints::None,
    UndistortionVisPoints::Detected,
    UndistortionVisPoints::GridDistorted,
    UndistortionVisPoints::GridUndistorted,
    UndistortionVisPoints::Reference,
];

const IMAGES: [UndistortionVisImage; 7] = [
    UndistortionVisImage::Distorted,
    UndistortionVisImage::Undistorted,
    UndistortionVisImage::MapAngleScaled,
    UndistortionVisImage::MapAngles,
    UndistortionVisImage::MapDistance,
    UndistortionVisImage::MapHorizontal,
    UndistortionVisImage::MapVertical,
];

fn points_index(p: UndistortionVisPoints) -> usize {
    POINTS.iter().position(|x| *x == p).unwrap_or(0)
}

fn image_index(i: UndistortionVisImage) -> usize {
    IMAGES.iter().position(|x| *x == i).unwrap_or(0)
}

/// Ported from ui/WizardUndistortionFrame.cpp, on_comboBoxImage_currentIndexChanged: an
/// image choice that does not match the points' space moves the points to the other space.
pub fn choose_image(state: &mut AppState, idx: usize) -> Changes {
    let mut ch = Changes::NONE;
    let pts = points_index(state.undistortion_vis_points);
    if idx == 0 && pts == 3 {
        ch |= choose_points(state, 2);
    } else if idx == 1 && pts == 2 {
        ch |= choose_points(state, 3);
    }
    ch |= state.change_undistortion_vis_image(IMAGES[idx.min(6)], false);
    ch
}

/// Ported from ui/WizardUndistortionFrame.cpp, on_comboBoxPoints_currentIndexChanged.
pub fn choose_points(state: &mut AppState, idx: usize) -> Changes {
    let mut ch = Changes::NONE;
    let img = image_index(state.undistortion_vis_image);
    if (idx == 1 || idx == 2) && img == 1 {
        ch |= state.change_undistortion_vis_image(UndistortionVisImage::Distorted, false);
    } else if (idx == 3 || idx == 4) && img == 0 {
        ch |= state.change_undistortion_vis_image(UndistortionVisImage::Undistorted, false);
    }
    ch |= state.change_undistortion_vis_points(POINTS[idx.min(4)], false);
    ch
}

/// Ported from ui/WizardUndistortionFrame.cpp, undistortionChanged(UNDISTORTED): image
/// "undistorted", points "grid points (undistorted)", mouse mode "None".
///
/// The C++ checks the "None" radio button with `setChecked`, which does not run its click
/// handler, so the state kept the previous mouse mode while the page showed "None"; here
/// the state follows the page.
pub fn on_undistorted(state: &mut AppState) -> Changes {
    let mut ch = choose_image(state, 1);
    ch |= choose_points(state, 3);
    ch |= state.change_undistortion_mouse_mode(UndistortionMouseMode::None, false);
    ch
}

/// Draws the page.
pub fn ui(ui: &mut egui::Ui, cx: &mut Cx<'_>) {
    let undistorted = cx.state.undistortion == UndistortionState::Undistorted;
    form::styled_frame(ui, |ui| {
        ui.vertical_centered(|ui| {
            ui.label(if undistorted {
                "Modify Undistortion"
            } else {
                "You first need to perform an Undistortion"
            });
        });
        // frameModifyUndistortion (a frame without a border).
        form::group_box(ui, "mouseclick mode", |ui| {
            ui.horizontal(|ui| {
                let mode = cx.state.undistortion_mouse_mode;
                let mut choose = |m: UndistortionMouseMode| {
                    cx.raise(Action::Calib(CalibAction::MouseMode(m)));
                };
                if form::radio(ui, true, mode == UndistortionMouseMode::None, "None") {
                    choose(UndistortionMouseMode::None);
                }
                ui.add_space(12.0);
                // "toggle outlier" shows once undistorted (`undistortionChanged`).
                if undistorted
                    && form::radio(
                        ui,
                        true,
                        mode == UndistortionMouseMode::ToggleOutlier,
                        "toggle outlier",
                    )
                {
                    choose(UndistortionMouseMode::ToggleOutlier);
                }
                // C5: asks first once undistorted (`app/calibration.rs`).
                if form::radio(
                    ui,
                    true,
                    mode == UndistortionMouseMode::SetCenter,
                    "set center",
                ) {
                    choose(UndistortionMouseMode::SetCenter);
                }
            });
        });
        if undistorted {
            form::group_box(ui, "visualization", |ui| {
                let w = (ui.available_width() - 70.0).max(120.0);
                egui::Grid::new("undist_vis").num_columns(2).show(ui, |ui| {
                    let l = ui.label("points   ").id;
                    let sel = points_index(cx.state.undistortion_vis_points);
                    if let Some(i) = form::labelled_combo(
                        ui,
                        true,
                        l,
                        "undist_comboBoxPoints",
                        sel,
                        &POINTS_ITEMS,
                        Some(w),
                    ) {
                        cx.change(|s| choose_points(s, i));
                    }
                    ui.end_row();
                    let l = ui.label("image").id;
                    let sel = image_index(cx.state.undistortion_vis_image);
                    if let Some(i) = form::labelled_combo(
                        ui,
                        true,
                        l,
                        "undist_comboBoxImage",
                        sel,
                        &IMAGE_ITEMS,
                        Some(w),
                    ) {
                        cx.change(|s| choose_image(s, i));
                    }
                    ui.end_row();
                });
            });
            if form::wide_button(ui, true, "remove outlier at border").clicked() {
                cx.raise(Action::Calib(CalibAction::RemoveOutliers));
            }
        }
        // C1, C2: one edit, which the engine runs as "compute" or "recompute" by the
        // session's undistortion state; it runs on a worker thread (`app/long_edit.rs`).
        let main = form::wide_button(
            ui,
            true,
            if undistorted {
                "recompute undistortion"
            } else {
                "compute undistortion"
            },
        );
        if main.clicked() {
            cx.request(EditRequest::Apply(xma_engine::Edit::Calibration(
                xma_engine::CalibrationEdit::ComputeUndistortion,
            )));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combos_keep_each_other_consistent() {
        let mut s = AppState::default();
        on_undistorted(&mut s);
        assert_eq!(s.undistortion_vis_image, UndistortionVisImage::Undistorted);
        assert_eq!(
            s.undistortion_vis_points,
            UndistortionVisPoints::GridUndistorted
        );
        // Distorted image: undistorted grid points become distorted ones.
        choose_image(&mut s, 0);
        assert_eq!(
            s.undistortion_vis_points,
            UndistortionVisPoints::GridDistorted
        );
        // Undistorted points: the image follows.
        choose_points(&mut s, 4);
        assert_eq!(s.undistortion_vis_image, UndistortionVisImage::Undistorted);
        choose_points(&mut s, 1);
        assert_eq!(s.undistortion_vis_image, UndistortionVisImage::Distorted);
    }
}
