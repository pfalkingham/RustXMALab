//! One camera view in the central area: the port of `CameraViewWidget` (form
//! `CameraViewWidget.ui`, `ui/CameraViewWidget.cpp`) around the image widget, which stands
//! in for `GLCameraView`.
//!
//! The central area (`crate::central`) arranges the views and calls [`show`] once per
//! visible camera with a [`CameraSlot`]; it also draws the light-grey background of the
//! active camera and makes a camera active when a mouse button is pressed on its view.
//!
//! The form, top to bottom:
//!
//! - `frame` (styled panel): `imageTitleLabel` ("<camera> - <image>"), `label_2` "Zoom ",
//!   `spinBoxZoom` (0 to 999), `label_3` " % ", `toolButtonFitZoom` (checkable, checked,
//!   icon `arrow_out_longer`), `toolButtonInfo` (checkable, icon `About`); under them
//!   `frameInfo`, shown while the info button is checked, holding the info frame of the
//!   current workspace ([`info_frames`]);
//! - `frame_calibration` (the external-calibration entry, Calibration workspace with an
//!   external calibration, in place of the image; [`info_frames::external_calibration`]);
//! - `glCameraView`: the image with its overlays.
//!
//! What the view draws, by workspace (`GLCameraView::paintGL`):
//!
//! - Marker tracking: the trial frame, the rigid-body meshes ([`overlay_mesh`]) and the
//!   overlays of [`xma_engine::view::overlays::tracking_overlay`] ([`overlay_tracking`]).
//! - Calibration (internal): the calibration image and [`overlay_calibration::draw`].
//! - Undistortion: the grid image and [`overlay_undistortion::draw`], or "No undistortion
//!   grid loaded" in red.
//!
//! Scale and bias: the C++ keeps one pair per camera for the calibration view and one for
//! the marker-tracking view in `Camera::ImageAdjustment` (saved in the project) and applies
//! the current workspace's pair (`CameraViewWidget::restoreImageAdjustment`); the
//! undistortion view always shows scale 1, bias 0. Here the pairs are view state
//! ([`CameraViewState::adjust`]), read from the project when it opens; the info frames edit
//! them. Nothing is written to the project in this version.

pub mod info_frames;
pub mod overlay_calibration;
pub mod overlay_mesh;
pub mod overlay_tracking;
pub mod overlay_undistortion;

use egui::{Color32, PointerButton};
use xma_model::{AdjustedView, CalibrationType, Camera};

use crate::frames::Frames;
use crate::icons::Icon;
use crate::panels::Cx;
use crate::state::Workspace;
use crate::widgets::image_view::{
    ImageAdjust, ImageView, ImageViewEvent, ImageViewState, ViewChange, ViewTransform,
};

/// What the central area knows about a view.
#[derive(Clone, Debug, PartialEq)]
pub struct CameraSlot {
    /// Camera index in the project.
    pub camera: usize,
    /// The camera's name.
    pub camera_name: String,
    /// The image name for the title (`MainWindow::setCameraViewWidgetTitles`).
    pub image_name: String,
    /// This camera is the active camera.
    pub active: bool,
    /// Full-height layout: the view is as wide as the image at full height, and the views
    /// scroll sideways (`CameraViewWidget::setMinimumWidthGL(true)`).
    pub full_height: bool,
}

impl CameraSlot {
    /// The title row's text: "<camera name> - <image name>" (`CameraViewWidget::setImageName`).
    #[must_use]
    pub fn title(&self) -> String {
        format!("{} - {}", self.camera_name, self.image_name)
    }
}

/// The interface state of one camera view.
pub struct CameraViewState {
    /// Zoom, offsets and fit (`GLCameraView`), with the zoom box value.
    pub image: ImageViewState,
    /// `toolButtonInfo` checked: `frameInfo` shown.
    pub info_open: bool,
    /// Scale and bias of the calibration view and of the marker-tracking view, indexed by
    /// [`AdjustedView::Calibration`] and [`AdjustedView::Digitization`]
    /// (`Camera::getImageAdjustment`).
    pub adjust: [ImageAdjust; 2],
    /// Rigid-body mesh transparency, 0 to 1 (`GLCameraView::transparency`, 0.5; Ctrl +
    /// wheel in marker tracking). Used by the mesh overlay ([`overlay_mesh`]).
    pub transparency: f64,
    /// "Draw transparent bone models" (`GLCameraView::renderTransparentModels`).
    pub transparent_models: bool,
}

impl CameraViewState {
    /// The state of a newly set-up view (`CameraViewWidget::CameraViewWidget`), with the
    /// camera's stored scale and bias.
    #[must_use]
    pub fn new(camera: &Camera) -> Self {
        let adj = |v: AdjustedView| {
            let a = camera.adjustment(v);
            ImageAdjust {
                scale: a.scale,
                bias: a.bias,
            }
        };
        let mut image = ImageViewState::camera();
        image.set_image_size([camera.width.max(1), camera.height.max(1)]);
        Self {
            image,
            info_open: false,
            adjust: [
                adj(AdjustedView::Calibration),
                adj(AdjustedView::Digitization),
            ],
            transparency: 0.5,
            transparent_models: true,
        }
    }

    /// The scale and bias shown in `workspace` (`CameraViewWidget::restoreImageAdjustment`):
    /// identity in Undistortion.
    #[must_use]
    pub fn adjust_for(&self, workspace: Workspace) -> ImageAdjust {
        match workspace {
            Workspace::Undistortion => ImageAdjust::default(),
            Workspace::Calibration => self.adjust[AdjustedView::Calibration as usize],
            Workspace::MarkerTracking => self.adjust[AdjustedView::Digitization as usize],
        }
    }
}

/// Lines one device pixel wide over a camera image, in image coordinates: the segments,
/// crosses and diagonal crosses of the overlays (`GLCameraView` draws them with
/// `glLineWidth(1)`).
pub(crate) struct Pen<'a> {
    painter: &'a egui::Painter,
    t: &'a ViewTransform,
    px: f32,
}

impl<'a> Pen<'a> {
    pub(crate) fn new(painter: &'a egui::Painter, t: &'a ViewTransform) -> Self {
        Self {
            painter,
            t,
            px: 1.0 / t.pixels_per_point,
        }
    }

    /// A segment from `a` to `b`.
    pub(crate) fn seg(&self, a: (f64, f64), b: (f64, f64), color: Color32) {
        self.painter.line_segment(
            [self.t.image_to_screen(a), self.t.image_to_screen(b)],
            egui::Stroke::new(self.px, color),
        );
    }

    /// A cross of half-size `r` at `p`: the horizontal segment, then the vertical one.
    pub(crate) fn cross(&self, p: (f64, f64), r: f64, color: Color32) {
        let (x, y) = p;
        self.seg((x - r, y), (x + r, y), color);
        self.seg((x, y - r), (x, y + r), color);
    }

    /// A diagonal cross (an x) of half-size 5 at `p`: top left to bottom right, then top
    /// right to bottom left.
    pub(crate) fn ex(&self, p: (f64, f64), color: Color32) {
        let (x, y) = p;
        self.seg((x - 5.0, y - 5.0), (x + 5.0, y + 5.0), color);
        self.seg((x + 5.0, y - 5.0), (x - 5.0, y + 5.0), color);
    }
}

/// Font of text drawn over a camera image, from the Qt point size XMALab draws it in (1.3
/// points per Qt point). It is part of the image overlay, not of the window's look, so it
/// keeps XMALab's proportions.
#[must_use]
pub fn qt_font(points: f32) -> egui::FontId {
    egui::FontId::proportional(points * 1.3)
}

/// Height of a view's header (the framed title row) and the space under it, in points.
#[must_use]
pub fn header_height(style: &egui::Style) -> f32 {
    style.spacing.interact_size.y
        + egui::Frame::group(style).total_margin().sum().y
        + style.spacing.item_spacing.y
}

/// The accessible names of a view's controls, for tests: "<camera name> zoom" and so on.
#[must_use]
pub fn control_name(camera_name: &str, control: &str) -> String {
    format!("{camera_name} {control}")
}

/// Draws one camera view filling `ui`.
pub fn show(ui: &mut egui::Ui, cx: &mut Cx<'_>, slot: &CameraSlot) {
    let cam = slot.camera;
    let Some(doc) = cx.doc else {
        return;
    };
    let Some(camera) = doc.project().cameras.get(cam) else {
        return;
    };
    if cx.frames.views.cameras.len() <= cam {
        cx.frames.project_opened(doc);
    }
    let rect = ui.available_rect_before_wrap();
    let workspace = cx.state.workspace;
    let external = workspace == Workspace::Calibration
        && doc.project().calibration_type == CalibrationType::External;

    ui.vertical(|ui| {
        ui.spacing_mut().item_spacing.y = 0.0;
        // `frame`: the title row and, when the info button is checked, `frameInfo`.
        crate::qt::styled_frame(ui, |ui| {
            ui.set_width(ui.available_width());
            if slot.active {
                // Text on the active view's highlighted background: egui's text colour for
                // selected items, readable in both themes. (The C++ writes white on light grey
                // in its dark theme, which cannot be read; that is not copied.)
                let c = crate::theme::active_camera_text(ui.visuals());
                ui.visuals_mut().override_text_color = Some(c);
            }
            title_row(ui, cx, slot);
            if cx.frames.views.cameras[cam].info_open {
                ui.separator();
                info_frames::show(ui, cx, cam);
            }
        });
        ui.add_space(ui.spacing().item_spacing.y);
        if external {
            info_frames::external_calibration(ui, cx, cam);
            return;
        }
        image(ui, cx, slot, camera);
    });
    ui.allocate_rect(rect, egui::Sense::hover());
}

/// The title row: `imageTitleLabel`, "Zoom ", the zoom box, " % ", fit and info.
fn title_row(ui: &mut egui::Ui, cx: &mut Cx<'_>, slot: &CameraSlot) {
    let cam = slot.camera;
    let h = ui.spacing().interact_size.y;
    ui.horizontal(|ui| {
        ui.set_min_height(h);
        let title = slot.title();
        // The controls are laid out right to left so the title takes what is left.
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let state = &mut cx.frames.views.cameras[cam];
            let icon_size = 16.0;
            let info = ui
                .add(
                    egui::Button::image(cx.icons.image(ui.ctx(), Icon::About, icon_size))
                        .frame_when_inactive(state.info_open)
                        .selected(state.info_open),
                )
                .accessible_name(control_name(&slot.camera_name, "info"));
            if info.clicked() {
                // CameraViewWidget::on_toolButtonInfo_clicked.
                state.info_open = !state.info_open;
            }
            let fit_on = state.image.auto_zoom();
            let fit = ui
                .add(
                    egui::Button::image(cx.icons.image(ui.ctx(), Icon::ArrowOutLonger, icon_size))
                        .frame_when_inactive(fit_on)
                        .selected(fit_on),
                )
                .accessible_name(control_name(&slot.camera_name, "fit"));
            if fit.clicked() {
                // CameraViewWidget::on_toolButtonFitZoom_clicked: setAutoZoom(checked).
                state.image.set_auto_zoom(!fit_on);
            }
            ui.label(" % ");
            let mut zoom = i64::from(state.image.zoom_percent());
            // spinBoxZoom, range 0 to 999 (on_spinBoxZoom_valueChanged: setZoom).
            let mut changed = false;
            ui.allocate_ui_with_layout(
                egui::vec2(56.0, h),
                egui::Layout::left_to_right(egui::Align::Center),
                |ui| {
                    changed = crate::qt::spin_box(
                        ui,
                        &mut zoom,
                        0,
                        999,
                        40.0,
                        &control_name(&slot.camera_name, "zoom"),
                    );
                },
            );
            if changed {
                state.image.set_zoom_percent(zoom as i32);
            }
            ui.label("Zoom ");
            ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                ui.add(egui::Label::new(title).truncate());
            });
        });
    });
}

/// `glCameraView`: the image and what is drawn over it.
fn image(ui: &mut egui::Ui, cx: &mut Cx<'_>, slot: &CameraSlot, camera: &Camera) {
    let cam = slot.camera;
    let workspace = cx.state.workspace;
    let key = Frames::image_key(cam);
    let store = cx.frames.store().cloned();
    let message = cx.frames.cameras.get(cam).and_then(|c| c.message.clone());
    let derived = match workspace {
        Workspace::Undistortion => overlay_undistortion::image_key(cx, cam),
        Workspace::Calibration => overlay_calibration::image_key(cx, cam),
        Workspace::MarkerTracking => None,
    };
    let state = &mut cx.frames.views.cameras[cam];
    // GLCameraView::setCamera: the image size is the camera's.
    state
        .image
        .set_image_size([camera.width.max(1), camera.height.max(1)]);
    let adjust = state.adjust_for(workspace);
    let has_image = cx
        .frames
        .cameras
        .get(cam)
        .is_some_and(|c| c.want.is_some() && !c.failed);
    let state = &mut cx.frames.views.cameras[cam];
    let mut view = ImageView::new(&mut state.image).adjust(adjust);
    if let Some(store) = &store
        && has_image
    {
        view = view.texture(store, derived.unwrap_or(key));
    }
    let out = view.show(ui);
    let response = out
        .response
        .clone()
        .accessible_name(control_name(&slot.camera_name, "image"));
    let painter = out.painter(ui);
    let t = out.transform;

    if workspace != Workspace::MarkerTracking {
        cx.frames.views.meshes.overlay.remove(&cam);
    }
    match workspace {
        Workspace::Undistortion => {
            if camera.undistortion.is_none() {
                // GLCameraView::drawTexture: renderTextCentered, 24 pt red.
                centered_message(
                    &painter,
                    &t,
                    "No undistortion grid loaded",
                    24.0,
                    crate::theme::VIEW_MESSAGE,
                );
            } else {
                overlay_undistortion::draw(&painter, &t, cx, cam);
            }
        }
        Workspace::Calibration => overlay_calibration::draw(&painter, &t, cx, cam),
        Workspace::MarkerTracking => {
            // GLCameraView::paintGL: the meshes over the image, then the points.
            overlay_mesh::draw(&painter, &t, cx, cam);
            overlay_tracking::draw(&painter, &t, cx, cam, false);
        }
    }
    if let Some(m) = message {
        centered_message(&painter, &t, &m, 12.0, crate::theme::VIEW_MESSAGE);
    }

    for event in &out.events {
        match workspace {
            Workspace::Undistortion if camera.undistortion.is_some() => {
                overlay_undistortion::on_event(cx, cam, event);
            }
            Workspace::Calibration => overlay_calibration::on_event(cx, cam, event),
            _ => {}
        }
        match *event {
            ImageViewEvent::Pressed {
                button: PointerButton::Primary,
                modifiers,
                pos,
            } => {
                if cx.state.disable_draw {
                    continue;
                }
                // GLCameraView::mousePressEvent, marker tracking: Ctrl + click adds a point
                // (E1), a click moves the active marker with detection (E2), Shift + click
                // selects the nearest marker (E4); `crate::tracking_ui::click`. Calibration
                // and undistortion clicks: their overlays' `on_event`.
                if workspace == Workspace::MarkerTracking {
                    click_at(cx, cam, (pos.x, pos.y), modifiers);
                }
            }
            ImageViewEvent::CtrlWheel { angle_delta } if workspace == Workspace::MarkerTracking => {
                // GLCameraView::wheelEvent with Ctrl: mesh transparency.
                let s = &mut cx.frames.views.cameras[cam];
                s.transparency = (s.transparency + angle_delta / 2400.0).clamp(0.0, 1.0);
            }
            ImageViewEvent::ViewChanged(ViewChange::Wheel) => {
                // The wheel makes the camera active too.
                cx.change(|s| s.change_active_camera(Some(cam), false));
            }
            _ => {}
        }
    }
    let _ = response;
}

/// The C key: every camera view (hidden ones too) is centred on the active marker of the
/// frame shown, without changing its zoom or its fit flag: the marker's point in that
/// camera, else its prediction from the previous frames, else from the next frames; a view
/// with none of them, or no active marker, stays as it is. Marker tracking only.
///
/// Ported from `ui/MainWindow.cpp`, `MainWindow::centerViews`, and
/// `ui/GLCameraView.cpp`, `GLCameraView::centerViewToPoint(false)`.
pub fn center_on_active(
    frames: &mut Frames,
    doc: &xma_engine::Document,
    state: &crate::state::AppState,
) {
    if state.workspace != Workspace::MarkerTracking {
        return;
    }
    let (Some(t), Some(view)) = (state.active_trial, state.trial_view().copied()) else {
        return;
    };
    let Some(m) = view.active_marker else { return };
    for (c, v) in frames.views.cameras.iter_mut().enumerate() {
        if let Some(p) = xma_engine::view::overlays::detail_center(doc, t, c, view.active_frame, m)
        {
            v.image.center_on((p.x, p.y), false);
        }
    }
}

/// A left click in Marker tracking (E1, E2, E4): the request, or the marker selected
/// (`WizardDigitizationFrame::selectDigitizationPoint`, then `PointsDockWidget::selectPoint`).
/// The detail views are centred after a selection here (`DetailViewDockWidget::centerViews`)
/// and after a placed point by the application, once the point is written.
fn click_at(cx: &mut Cx<'_>, cam: usize, pos: (f64, f64), modifiers: egui::Modifiers) {
    let Some(doc) = cx.doc else { return };
    match crate::tracking_ui::click(doc, cx.state, cam, pos, modifiers, false) {
        crate::tracking_ui::ClickOutcome::Request(r) => cx.request(r),
        crate::tracking_ui::ClickOutcome::Select(m) => {
            cx.change(|s| s.select_marker(Some(m)));
        }
        crate::tracking_ui::ClickOutcome::Nothing => {}
    }
}

/// Text centred in the view (`GLCameraView::renderTextCentered` and the warnings).
pub fn centered_message(
    painter: &egui::Painter,
    t: &crate::widgets::image_view::ViewTransform,
    text: &str,
    qt_points: f32,
    color: Color32,
) {
    let r = t.view_rect();
    let galley = painter.layout(
        text.to_owned(),
        qt_font(qt_points),
        color,
        (r.width() - 16.0).max(40.0),
    );
    let pos = r.center() - galley.size() / 2.0;
    painter.galley(pos, galley, color);
}
