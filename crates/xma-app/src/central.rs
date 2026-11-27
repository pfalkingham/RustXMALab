//! The central area: the start screen, the trial start frame, or the camera views with the
//! frame bar under them.
//!
//! Ported from `MainWindow.ui` (`startMainFrame`, `startTrialFrame`, `imageMainFrame`,
//! `imageScrollArea`) and ui/MainWindow.cpp (`relayoutCameras`, `setCameraViewWidgetTitles`,
//! `updateRecentFiles`, `workspaceChanged`).
//!
//! - **No dataset**: the start screen, a centred column: the Recent Files group (one button
//!   per recent file), then the buttons "New dataset", "New dataset with external
//!   calibration", "New trial without calibration", "Open .xma file", "Open .xma trial file
//!   with calibration from another xma-file" (each at least 150 × 40).
//! - **Marker tracking without trials**: the trial start frame: "Add trial" and "Add Default
//!   Trial", or "You first have to calibrate your cameras!" when the cameras are not
//!   calibrated.
//! - **Otherwise**: the visible cameras in the layout of the View combo box ([`arrange`]),
//!   each drawn by [`crate::camera_view::show`], and the frame bar ([`crate::frame_bar`]).

use crate::actions::Action;
use crate::camera_view::{self, CameraSlot};
use crate::panels::Cx;
use crate::state::{DisplayLayout, Workspace};
use xma_model::CalibrationType;

/// The rows of camera views for a layout: which cameras go in which row.
///
/// Ported from ui/MainWindow.cpp, MainWindow::relayoutCameras (`:549-593`). For the 1-,
/// 2- and 3-row layouts there are `rows` rows (all of them, even empty ones, each given
/// equal height), `ceil(n / rows)` views per row, and the free slots (`rows × width - n`)
/// are skipped once the cameras still to place fit the remaining slots exactly. (The C++
/// picks rows through `QSplitter::children()`, which `QSplitterPrivate::insertWidget`
/// reorders with `lower()`; the net effect is that camera 0 is in the top row.) Full height
/// is one row that scrolls sideways.
#[must_use]
pub fn arrange(visible: &[usize], layout: DisplayLayout) -> Vec<Vec<usize>> {
    let rows = match layout {
        DisplayLayout::FullHeight | DisplayLayout::OneRow => 1,
        DisplayLayout::TwoRows => 2,
        DisplayLayout::ThreeRows => 3,
    };
    let mut out = vec![Vec::new(); rows];
    let n = visible.len();
    if n == 0 {
        return out;
    }
    let width = n.div_ceil(rows);
    let free = rows * width - n;
    let mut count = 0usize;
    for (placed, cam) in visible.iter().enumerate() {
        if n < placed + free {
            count += 1;
        }
        let row = count / width;
        if let Some(r) = out.get_mut(row) {
            r.push(*cam);
        }
        count += 1;
    }
    out
}

/// The image name in a camera view's title.
///
/// Ported from ui/MainWindow.cpp, MainWindow::setCameraViewWidgetTitles: in Calibration
/// (internal calibration) the active calibration image's file name; in Undistortion the
/// grid image's file name or "No undistortion Grid loaded"; in Marker tracking the active
/// trial's current frame (`Trial::getActiveFilename`, empty for the default trial).
#[must_use]
pub fn image_name(cx: &Cx<'_>, camera: usize) -> String {
    let Some(doc) = cx.doc else {
        return String::new();
    };
    let p = doc.project();
    let Some(cam) = p.cameras.get(camera) else {
        return String::new();
    };
    let file = |f: &str| {
        let f = crate::session::adjust_path_to_os(f);
        std::path::Path::new(&f)
            .file_name()
            .map_or(f.clone(), |n| n.to_string_lossy().into_owned())
    };
    match cx.state.workspace {
        Workspace::Calibration if p.calibration_type == CalibrationType::Internal => cam
            .calibration_images
            .get(cx.state.active_frame_calibration.unwrap_or(0))
            .map(|im| file(&im.filename))
            .unwrap_or_default(),
        Workspace::Calibration => String::new(),
        Workspace::Undistortion => match &cam.undistortion {
            Some(u) => file(&u.image_filename),
            None => "No undistortion Grid loaded".to_owned(),
        },
        Workspace::MarkerTracking => {
            let Some((t, view)) = cx.state.active_trial.zip(cx.state.trial_view()) else {
                return String::new();
            };
            let Some(trial) = p.trials.get(t) else {
                return String::new();
            };
            if trial.is_default {
                return String::new();
            }
            trial
                .videos
                .get(camera)
                .and_then(|v| {
                    if v.filenames.len() > 1 {
                        v.filenames.get(view.active_frame)
                    } else {
                        v.filenames.first()
                    }
                })
                .map(|f| file(f))
                .unwrap_or_default()
        }
    }
}

/// Draws the central area.
pub fn ui(ui: &mut egui::Ui, cx: &mut Cx<'_>) {
    let full = ui.available_rect_before_wrap();
    if cx.doc.is_none() {
        start_screen(ui, cx);
        ui.allocate_rect(full, egui::Sense::hover());
        return;
    }
    if cx.state.workspace == Workspace::MarkerTracking && cx.facts.n_trials() == 0 {
        trial_start_frame(ui, cx);
        ui.allocate_rect(full, egui::Sense::hover());
        return;
    }
    let bar_h = if crate::frame_bar::visible(cx.state, cx.facts) {
        crate::frame_bar::height(ui.style()) + ui.spacing().item_spacing.y
    } else {
        0.0
    };
    let views = egui::Rect::from_min_max(full.min, egui::pos2(full.max.x, full.max.y - bar_h));
    let mut views_ui = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(views)
            .id_salt("camera_views"),
    );
    camera_views(&mut views_ui, cx, views);
    if bar_h > 0.0 {
        let bar = egui::Rect::from_min_max(
            egui::pos2(full.min.x, views.max.y + ui.spacing().item_spacing.y),
            full.max,
        );
        let mut bar_ui = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(bar)
                .id_salt("frame_bar")
                .layout(egui::Layout::left_to_right(egui::Align::Center)),
        );
        crate::frame_bar::ui(&mut bar_ui, cx);
    }
    ui.allocate_rect(full, egui::Sense::hover());
}

fn slot(cx: &Cx<'_>, camera: usize, full_height: bool) -> CameraSlot {
    CameraSlot {
        camera,
        camera_name: cx
            .facts
            .camera_names
            .get(camera)
            .cloned()
            .unwrap_or_default(),
        image_name: image_name(cx, camera),
        active: cx.state.active_camera == Some(camera),
        full_height,
    }
}

/// One view in its rectangle: active background, press-to-activate, then the view.
fn view_in(ui: &mut egui::Ui, cx: &mut Cx<'_>, rect: egui::Rect, s: &CameraSlot) {
    if s.active {
        ui.painter()
            .rect_filled(rect, 0.0, crate::theme::active_camera_bg(ui.visuals()));
    }
    // Only a press on the view itself, as only those reach `GLCameraView::mousePressEvent`:
    // in the part of the view that is not clipped away (a scrolled-out view lies under the
    // docked panels), and with no other layer (a menu, popup, modal or floating window) on
    // top at that point (as the Plot widget checks, `widgets/plot`).
    let visible = rect.intersect(ui.clip_rect());
    let layer = ui.layer_id();
    let pressed = ui.is_enabled()
        && ui
            .ctx()
            .input(|i| {
                i.pointer
                    .any_pressed()
                    .then(|| i.pointer.interact_pos())
                    .flatten()
            })
            .is_some_and(|p| {
                visible.contains(p) && ui.ctx().layer_id_at(p).is_none_or(|l| l == layer)
            });
    if pressed && !s.active {
        let cam = s.camera;
        cx.change(|st| st.change_active_camera(Some(cam), false));
    }
    let mut child = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(rect.shrink(6.0))
            .id_salt(("camera_view", s.camera)),
    );
    child.set_clip_rect(rect.intersect(ui.clip_rect()));
    camera_view::show(&mut child, cx, s);
}

fn camera_views(ui: &mut egui::Ui, cx: &mut Cx<'_>, rect: egui::Rect) {
    let visible: Vec<usize> = (0..cx.facts.n_cameras)
        .filter(|c| cx.state.camera_visible.get(*c).copied().unwrap_or(true))
        .collect();
    if cx.state.display == DisplayLayout::FullHeight {
        // Each view as wide as its image at full height, side by side, scrolling sideways.
        let doc = cx.doc;
        let title_h = crate::camera_view::header_height(ui.style());
        egui::ScrollArea::horizontal()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    for cam in &visible {
                        let aspect = doc
                            .and_then(|d| d.project().cameras.get(*cam))
                            .filter(|c| c.height > 0)
                            .map_or(1.0, |c| c.width as f32 / c.height as f32);
                        let w = ((rect.height() - title_h) * aspect + 12.0).max(80.0);
                        let (r, _) = ui.allocate_exact_size(
                            egui::vec2(w, rect.height()),
                            egui::Sense::hover(),
                        );
                        let s = slot(cx, *cam, true);
                        view_in(ui, cx, r, &s);
                    }
                });
            });
        return;
    }
    let rows = arrange(&visible, cx.state.display);
    let n_rows = rows.len().max(1);
    let row_h = rect.height() / n_rows as f32;
    for (ri, row) in rows.iter().enumerate() {
        if row.is_empty() {
            continue;
        }
        let w = rect.width() / row.len() as f32;
        for (ci, cam) in row.iter().enumerate() {
            let r = egui::Rect::from_min_size(
                egui::pos2(rect.min.x + w * ci as f32, rect.min.y + row_h * ri as f32),
                egui::vec2(w, row_h),
            );
            let s = slot(cx, *cam, false);
            view_in(ui, cx, r, &s);
        }
    }
}

/// A start-screen button: at least 150 × 40, disabled when its action is.
fn big_button(ui: &mut egui::Ui, text: &str, width: f32, enabled: bool) -> bool {
    ui.add_enabled(
        enabled,
        egui::Button::new(text).min_size(egui::vec2(width, 40.0)),
    )
    .clicked()
}

fn start_screen(ui: &mut egui::Ui, cx: &mut Cx<'_>) {
    let buttons: [(&str, Action); 5] = [
        ("New dataset", Action::NewProject),
        (
            "New dataset with external calibration",
            Action::NewDatasetExternalCalibration,
        ),
        (
            "New trial without calibration",
            Action::NewTrialWithoutCalibration,
        ),
        ("Open .xma file", Action::LoadProject),
        (
            "Open .xma trial file with calibration from another xma-file",
            Action::LoadProjectWithDifferentCalibration,
        ),
    ];
    let font = egui::TextStyle::Button.resolve(ui.style());
    let mut width: f32 = 150.0;
    for (t, _) in &buttons {
        width = width.max(
            ui.painter()
                .layout_no_wrap((*t).to_owned(), font.clone(), egui::Color32::BLACK)
                .size()
                .x
                + 24.0,
        );
    }
    for f in &cx.settings.recent_files {
        width = width.max(
            ui.painter()
                .layout_no_wrap(f.clone(), font.clone(), egui::Color32::BLACK)
                .size()
                .x
                + 40.0,
        );
    }
    let width = width.min(ui.available_width() - 20.0);
    let recent = cx.settings.recent_files.clone();
    ui.vertical_centered(|ui| {
        ui.add_space((ui.available_height() * 0.2).min(120.0));
        ui.allocate_ui(egui::vec2(width + 20.0, 0.0), |ui| {
            egui::Frame::group(ui.style())
                .inner_margin(egui::Margin::same(8))
                .show(ui, |ui| {
                    ui.set_width(width);
                    ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                        ui.label("Recent Files");
                        for (i, f) in recent.iter().enumerate() {
                            let a = Action::OpenRecentFile(i);
                            if ui
                                .add_enabled(
                                    a.implemented(),
                                    egui::Button::new(f.as_str()).min_size(egui::vec2(width, 0.0)),
                                )
                                .clicked()
                            {
                                cx.raise(a);
                            }
                        }
                    });
                });
        });
        ui.add_space(30.0);
        for (t, a) in buttons {
            if big_button(ui, t, width, a.implemented()) {
                cx.raise(a);
            }
            ui.add_space(2.0);
        }
    });
}

fn trial_start_frame(ui: &mut egui::Ui, cx: &mut Cx<'_>) {
    ui.vertical_centered(|ui| {
        ui.add_space(ui.available_height() * 0.35);
        if cx.facts.calibrated_or_none() {
            if big_button(ui, "Add trial", 150.0, Action::AddTrial.implemented()) {
                cx.raise(Action::AddTrial);
            }
            if big_button(
                ui,
                "Add Default Trial",
                150.0,
                Action::AddDefaultTrial.implemented(),
            ) {
                cx.raise(Action::AddDefaultTrial);
            }
        } else {
            ui.label("You first have to calibrate your cameras!");
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrangement_matches_cpp_arithmetic() {
        let cams = |n: usize| (0..n).collect::<Vec<_>>();
        assert_eq!(arrange(&cams(2), DisplayLayout::OneRow), vec![vec![0, 1]]);
        assert_eq!(
            arrange(&cams(4), DisplayLayout::TwoRows),
            vec![vec![0, 1], vec![2, 3]]
        );
        assert_eq!(
            arrange(&cams(3), DisplayLayout::TwoRows),
            vec![vec![0, 1], vec![2]]
        );
        assert_eq!(
            arrange(&cams(5), DisplayLayout::TwoRows),
            vec![vec![0, 1, 2], vec![3, 4]]
        );
        assert_eq!(
            arrange(&cams(5), DisplayLayout::ThreeRows),
            vec![vec![0, 1], vec![2, 3], vec![4]]
        );
        // 7 cameras in 3 rows of 3: the last camera skips a slot.
        assert_eq!(
            arrange(&cams(7), DisplayLayout::ThreeRows),
            vec![vec![0, 1, 2], vec![3, 4, 5], vec![6]]
        );
        // Fewer cameras than rows: the extra rows stay empty.
        assert_eq!(
            arrange(&cams(2), DisplayLayout::ThreeRows),
            vec![vec![0], vec![1], vec![]]
        );
        assert_eq!(
            arrange(&[], DisplayLayout::TwoRows),
            vec![Vec::<usize>::new(), vec![]]
        );
    }

    #[test]
    fn free_slot_skip_moves_last_camera() {
        // 4 cameras, 3 rows: width 2, 2 free; the skip starts once n < placed + free.
        assert_eq!(
            arrange(&[0, 1, 2, 3], DisplayLayout::ThreeRows),
            vec![vec![0, 1], vec![2], vec![3]]
        );
    }
}
