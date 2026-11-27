//! The Detailed View panel: one magnified view per camera, centred on the active marker.
//!
//! Ported from `DetailViewDockWidget.ui` / `ui/DetailViewDockWidget.cpp` (the panel: the
//! cameras side by side in `imageMainFrame`, between two spacers, each shown while its
//! camera is visible) and `CameraViewDetailWidget.ui` / `ui/CameraViewDetailWidget.cpp`
//! (one camera): a styled frame with
//!
//! | Row | Controls |
//! |---|---|
//! | 1 | `label_3` "Scale ", `horizontalSliderScale` (0 to 300, step 10, page 100), `doubleSpinBoxScale` (0 to 3, step 0.05, 2 decimals) |
//! | 2 | `label` "Bias ", `horizontalSliderBias` (-100 to 100, step 10, page 100), `doubleSpinBoxBias` (-1 to 1, step 0.01, 2 decimals) |
//!
//! and under it the 200 × 200 `glCameraView` in detail mode: zoom 500 %, the camera's image
//! (the same texture as the camera view), and the marker-tracking overlays of a detail view
//! (`super::super::camera_view::overlay_tracking`, with `detail` set: the crosshair of the
//! active marker, and the projected point, epipolar lines and numbers as the "detail view"
//! settings say).
//!
//! The slider and the spin box show the same value (slider = 100 × value); a double click
//! on the label or the slider resets scale to 1 and bias to 0 (`DoubleClickReset`).
//!
//! Centring (`GLCameraView::centerViewToPoint`, marker tracking only): on the active
//! marker's point in the camera, else its prediction from the frames before, else from the
//! frames after ([`xma_engine::view::overlays::detail_center`]); with the zoom reset to
//! 500 % when the workspace, the trial or the active marker changes, without on a frame
//! change. With the `CenterDetailView` setting the view follows the marker every frame and
//! the wheel does nothing.
//!
//! Scale and bias: the C++ stores them per camera in the project
//! (`Camera::getImageAdjustment(Camera::DetailView)`, saved with it). Here the view shows
//! [`DetailViewState::adjust`], read from the project when it opens; every change is also
//! written to the document's camera through an edit request
//! ([`crate::editing::EditRequest::ImageAdjustment`], `AdjustedView::Detail`), so that Save
//! stores it (parity ledger U46). Undo and redo leave it alone.

use egui::PointerButton;
use xma_model::{AdjustedView, Camera};

use super::{AppEvent, Cx, Panel, PanelId};
use crate::frames::Frames;
use crate::state::Workspace;
use crate::widgets::image_view::{ImageAdjust, ImageView, ImageViewEvent, ImageViewState};

/// Size of a detail view (`glCameraView` minimum and maximum size).
pub const DETAIL_VIEW_SIZE: f32 = 200.0;

/// Width of one camera's column (the form is 220 wide with 5-point margins).
const COLUMN_W: f32 = 210.0;

/// The interface state of one camera's detail view.
pub struct DetailViewState {
    pub image: ImageViewState,
    /// Scale and bias (`Camera::DetailView`).
    pub adjust: ImageAdjust,
}

impl DetailViewState {
    /// A new detail view (`CameraViewDetailWidget::CameraViewDetailWidget`), with the
    /// camera's stored scale and bias (`restoreImageAdjustment`).
    #[must_use]
    pub fn new(camera: &Camera) -> Self {
        let a = camera.adjustment(AdjustedView::Detail);
        let mut image = ImageViewState::detail();
        image.set_image_size([camera.width.max(1), camera.height.max(1)]);
        Self {
            image,
            adjust: ImageAdjust {
                scale: a.scale,
                bias: a.bias,
            },
        }
    }
}

/// The Detailed View panel.
#[derive(Debug, Default)]
pub struct DetailViewPanel;

/// Re-centres every detail view on the active marker.
///
/// Ported from `ui/GLCameraView.cpp`, `GLCameraView::centerViewToPoint(resetZoom)`.
pub fn center_views(cx: &mut Cx<'_>, reset_zoom: bool) {
    let Some(doc) = cx.doc else { return };
    center_on_active(cx.frames, doc, cx.state, reset_zoom);
}

/// [`center_views`] on the frame service's view state, for the application (after a click
/// that placed a point).
pub fn center_on_active(
    frames: &mut Frames,
    doc: &xma_engine::Document,
    state: &crate::state::AppState,
    reset_zoom: bool,
) {
    if state.workspace != Workspace::MarkerTracking {
        return;
    }
    let (Some(t), Some(view)) = (state.active_trial, state.trial_view().copied()) else {
        return;
    };
    if doc.project().trials.get(t).is_none() {
        return;
    }
    for (c, d) in frames.views.details.iter_mut().enumerate() {
        center_detail(d, doc, t, c, &view, reset_zoom);
    }
}

/// One detail view of [`center_on_active`]: camera `c`'s view centred on the active marker
/// of trial `t`; where the marker has no position there, with `reset_zoom` the zoom is reset
/// around the present centre, otherwise nothing changes.
fn center_detail(
    d: &mut DetailViewState,
    doc: &xma_engine::Document,
    t: usize,
    c: usize,
    view: &crate::state::TrialView,
    reset_zoom: bool,
) {
    let p = view
        .active_marker
        .and_then(|m| xma_engine::view::overlays::detail_center(doc, t, c, view.active_frame, m));
    match p {
        Some(p) => d.image.center_on((p.x, p.y), reset_zoom),
        None if reset_zoom => {
            let off = d.image.transform().offset;
            d.image.center_on((-off[0], -off[1]), true);
        }
        None => {}
    }
}

impl Panel for DetailViewPanel {
    fn id(&self) -> PanelId {
        PanelId::DetailedView
    }

    fn on_event(&mut self, event: &AppEvent, cx: &mut Cx<'_>) {
        match event {
            AppEvent::ProjectOpened => center_views(cx, true),
            AppEvent::Changed(ch) => {
                // CameraViewDetailWidget: workspaceChanged, activeTrialChanged and
                // activePointChanged re-centre with the zoom reset; activeFrameTrialChanged
                // without (and not while drawing is disabled).
                if ch.workspace || ch.active_trial || ch.selection {
                    center_views(cx, true);
                } else if ch.active_frame_trial && !cx.state.disable_draw {
                    center_views(cx, false);
                }
            }
            AppEvent::ProjectClosed => {}
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, cx: &mut Cx<'_>) {
        let Some(doc) = cx.doc else { return };
        let n = doc.project().cameras.len();
        if cx.frames.views.details.len() != n {
            cx.frames.project_opened(doc);
        }
        let follow =
            cx.settings.center_detail_view && cx.state.workspace == Workspace::MarkerTracking;
        egui::ScrollArea::both()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    // The two spacers of the form centre the cameras.
                    let shown: Vec<usize> = (0..n)
                        .filter(|c| cx.state.camera_visible.get(*c).copied().unwrap_or(true))
                        .collect();
                    let w = shown.len() as f32 * (COLUMN_W + 10.0);
                    ui.add_space(((ui.available_width() - w) * 0.5).max(4.0));
                    for c in shown {
                        ui.allocate_ui_with_layout(
                            egui::vec2(COLUMN_W, ui.available_height()),
                            egui::Layout::top_down(egui::Align::Center),
                            |ui| {
                                ui.set_width(COLUMN_W);
                                camera_detail(ui, cx, c, follow);
                            },
                        );
                        ui.add_space(10.0);
                    }
                });
            });
    }
}

/// One `CameraViewDetailWidget`.
fn camera_detail(ui: &mut egui::Ui, cx: &mut Cx<'_>, c: usize, follow: bool) {
    let Some(doc) = cx.doc else { return };
    let name = doc.project().cameras[c].name.clone();
    crate::qt::styled_frame(ui, |ui| {
        let state = &mut cx.frames.views.details[c];
        let mut changed = false;
        egui::Grid::new(("detail_adjust", c))
            .num_columns(3)
            .show(ui, |ui| {
                changed |= adjust_row(
                    ui,
                    &format!("{name} detail"),
                    "Scale",
                    &mut state.adjust.scale,
                    0.0..=3.0,
                    0.05,
                    1.0,
                    (0, 300),
                );
                ui.end_row();
                changed |= adjust_row(
                    ui,
                    &format!("{name} detail"),
                    "Bias",
                    &mut state.adjust.bias,
                    -1.0..=1.0,
                    0.01,
                    0.0,
                    (-100, 100),
                );
                ui.end_row();
            });
        if changed {
            let a = state.adjust;
            cx.edits.push(crate::editing::EditRequest::ImageAdjustment {
                camera: c,
                view: AdjustedView::Detail,
                scale: a.scale,
                bias: a.bias,
            });
        }
    });
    ui.add_space(5.0);
    let key = Frames::image_key(c);
    let store = cx.frames.store().cloned();
    let has_image = cx
        .frames
        .cameras
        .get(c)
        .is_some_and(|i| i.want.is_some() && !i.failed);
    // GLCameraView::paintGL: with "centre detail view" the view follows the marker.
    if follow {
        center_one(cx, c);
    }
    let state = &mut cx.frames.views.details[c];
    state.image.set_center_locked(follow);
    let adjust = state.adjust;
    let mut view = ImageView::new(&mut state.image)
        .adjust(adjust)
        .size(egui::vec2(DETAIL_VIEW_SIZE, DETAIL_VIEW_SIZE));
    if let Some(store) = &store
        && has_image
    {
        view = view.texture(store, key);
    }
    let out = view.show(ui);
    let _ = out
        .response
        .clone()
        .accessible_name(format!("{name} detail view"));
    let painter = out.painter(ui);
    if cx.state.workspace == Workspace::MarkerTracking {
        crate::camera_view::overlay_tracking::draw(&painter, &out.transform, cx, c, true);
    }
    for e in &out.events {
        if let ImageViewEvent::Pressed { .. } | ImageViewEvent::DoubleClicked { .. } = e {
            // GLCameraView::mousePressEvent: any press makes the camera active.
            cx.change(|s| s.change_active_camera(Some(c), false));
        }
        // A left click places the active marker exactly there, status MANUAL (E3, also with
        // Ctrl); Shift + click selects the nearest marker (E4).
        if let ImageViewEvent::Pressed {
            button: PointerButton::Primary,
            modifiers,
            pos,
        } = *e
        {
            match crate::tracking_ui::click(doc, cx.state, c, (pos.x, pos.y), modifiers, true) {
                crate::tracking_ui::ClickOutcome::Request(r) => cx.request(r),
                crate::tracking_ui::ClickOutcome::Select(m) => {
                    cx.change(|s| s.select_marker(Some(m)));
                }
                crate::tracking_ui::ClickOutcome::Nothing => {}
            }
        }
    }
}

fn center_one(cx: &mut Cx<'_>, c: usize) {
    let Some(doc) = cx.doc else { return };
    let (Some(t), Some(view)) = (cx.state.active_trial, cx.state.trial_view().copied()) else {
        return;
    };
    center_detail(&mut cx.frames.views.details[c], doc, t, c, &view, false);
}

/// One row: label, slider (100 × value), spin box. Double click on the label or the slider
/// resets the value. Returns true when the value changed.
///
/// Ported from `ui/CameraViewDetailWidget.cpp`, `on_horizontalSlider*_valueChanged`,
/// `on_doubleSpinBox*_valueChanged` and the `DoubleClickReset` filters. The C++ moves the
/// slider to `(int)(value * 100)`, truncating, which then sets the spin box back to the
/// truncated value (0.29 becomes 0.28); here the slider rounds.
#[allow(clippy::too_many_arguments)]
pub fn adjust_row(
    ui: &mut egui::Ui,
    prefix: &str,
    label: &str,
    value: &mut f64,
    range: std::ops::RangeInclusive<f64>,
    step: f64,
    reset: f64,
    slider_range: (i32, i32),
) -> bool {
    let before = *value;
    let l = ui.add(egui::Label::new(format!("{label} ")).sense(egui::Sense::click()));
    let mut s = (*value * 100.0).round() as i32;
    ui.spacing_mut().slider_width = 80.0;
    let slider = ui
        .add(
            egui::Slider::new(&mut s, slider_range.0..=slider_range.1)
                .show_value(false)
                .step_by(1.0),
        )
        .accessible_name(format!("{prefix} {} slider", label.to_lowercase()));
    if slider.changed() {
        *value = 0.01 * f64::from(s);
    }
    let mut v = *value;
    let spin = ui
        .add_sized(
            [48.0, ui.spacing().interact_size.y],
            egui::DragValue::new(&mut v)
                .range(range)
                .speed(step)
                .fixed_decimals(2),
        )
        .accessible_name(format!("{prefix} {}", label.to_lowercase()));
    if spin.changed() {
        *value = (v * 100.0).round() / 100.0;
    }
    // DoubleClickReset on the label and the slider (the slider senses drags only).
    let double = ui.input(|i| i.pointer.button_double_clicked(PointerButton::Primary));
    if l.double_clicked() || slider.double_clicked() || (double && slider.contains_pointer()) {
        *value = reset;
    }
    *value != before
}
