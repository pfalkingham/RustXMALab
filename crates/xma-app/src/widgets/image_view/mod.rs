//! The image view: one image with pan and zoom, drawn on the GPU. It is the base
//! of every camera view and of the Detailed View panel, and knows nothing about
//! projects, markers or the engine.
//!
//! Ported from `ui/GLCameraView.cpp` (the image, its scale and bias, the view
//! transform and the mouse handling). Overlays (markers, epipolar lines, rigid
//! bodies, grid points, the detail view crosshair, texts) are drawn by the
//! caller with egui's painter on top of the image, in image coordinates through
//! the returned [`ViewTransform`].
//!
//! - [`ImageStore`] owns the textures, keyed by an [`egui::Id`] the caller
//!   chooses. [`ImageStore::upload`] hands a frame to an upload thread and never
//!   blocks; frames are dropped, not queued, when the thread is busy.
//! - [`ImageViewState`] holds one view's zoom, offsets and "fit" flag:
//!   [`ImageViewState::camera`] for a camera view (with the zoom box),
//!   [`ImageViewState::detail`] for a detail view.
//! - [`ImageView`] is the widget. It draws the image, applies pan (right drag)
//!   and zoom (wheel; a factor per notch, [`WheelZoom`]), and returns [`ImageViewOutput`]: the transform and what the
//!   user did ([`ImageViewEvent`]). It changes nothing else; left clicks with
//!   their modifiers, double clicks and Ctrl + wheel are only reported.
//!
//! # Example: a camera view
//!
//! ```no_run
//! use eframe::egui;
//! use xma_app::widgets::image_view::{
//!     DisplayImage, ImageAdjust, ImageStore, ImageView, ImageViewEvent, ImageViewState,
//! };
//!
//! struct CameraView {
//!     key: egui::Id,
//!     state: ImageViewState,
//!     adjust: ImageAdjust,
//! }
//!
//! fn new_frame(store: &ImageStore, view: &CameraView, frame: xma_video::Frame, bit_depth: u8) {
//!     // Returns at once; the upload thread writes the texture and requests a repaint.
//!     store.upload(view.key, DisplayImage::gray(frame, bit_depth));
//! }
//!
//! fn show(ui: &mut egui::Ui, store: &ImageStore, view: &mut CameraView, marker: (f64, f64)) {
//!     // Zoom box and fit button, as in CameraViewWidget.
//!     ui.horizontal(|ui| {
//!         let mut percent = view.state.zoom_percent();
//!         if ui.add(egui::DragValue::new(&mut percent).range(0..=999)).changed() {
//!             view.state.set_zoom_percent(percent);
//!         }
//!         let mut fit = view.state.auto_zoom();
//!         if ui.toggle_value(&mut fit, "fit").changed() {
//!             view.state.set_auto_zoom(fit);
//!         }
//!     });
//!     let out = ImageView::new(&mut view.state)
//!         .texture(store, view.key)
//!         .adjust(view.adjust)
//!         .show(ui);
//!
//!     // Overlay in image coordinates: a cross centred on pixel `marker`.
//!     let painter = out.painter(ui);
//!     let c = out.transform.image_to_screen(marker);
//!     let stroke = egui::Stroke::new(1.0, egui::Color32::GREEN);
//!     painter.line_segment([c - egui::vec2(5.0, 0.0), c + egui::vec2(5.0, 0.0)], stroke);
//!     painter.line_segment([c - egui::vec2(0.0, 5.0), c + egui::vec2(0.0, 5.0)], stroke);
//!
//!     for event in &out.events {
//!         match event {
//!             ImageViewEvent::Pressed { button: egui::PointerButton::Primary, modifiers, pos }
//!                 if modifiers.command =>
//!             {
//!                 // Ctrl + left click: add a marker at `pos` (image coordinates).
//!                 let _ = pos;
//!             }
//!             ImageViewEvent::CtrlWheel { angle_delta } => {
//!                 // Marker tracking: mesh transparency += angle_delta / 2400.
//!                 let _ = angle_delta;
//!             }
//!             _ => {}
//!         }
//!     }
//! }
//! ```
//!
//! A detail view is the same with [`ImageViewState::detail`], re-centred with
//! [`ImageViewState::center_on`] when the frame or the active marker changes.
//!
//! # Coordinates
//!
//! Image coordinates have pixel (0, 0) top-left and pixel centres at integers;
//! pixel (x, y) covers [x - 0.5, x + 0.5] × [y - 0.5, y + 0.5]. That is also the
//! C++ view's convention (its image quad runs from -0.5 to width - 0.5), so a
//! marker at (x, y) is drawn over the same pixel as in XMALab, and a click maps
//! to the same image position as `GLCameraView::mousePressEvent` computes.
//! The zoom ratio counts device (physical) pixels, as the C++ does with
//! `devicePixelRatio`: 100 % shows one image pixel per physical pixel.

mod render;
mod state;
mod store;
mod transform;

pub use render::{Filtering, ImageAdjust, grey_level};
pub use state::{ImageViewState, ViewChange, ViewKind, WheelZoom, ZOOM_PERCENT_MAX};
pub use store::{DisplayImage, ImageStore, TextureInfo, UploadStats};
pub use transform::{
    ANGLE_DELTA_PER_NOTCH, DETAIL_ZOOM_RATIO, ImagePos, MIN_ZOOM_RATIO, ViewTransform,
    WHEEL_ZOOM_PER_ANGLE_UNIT,
};

use eframe::egui_wgpu;
use egui::{Color32, Id, Modifiers, MouseWheelUnit, PointerButton, Sense, Vec2};

use render::{DrawParams, ImageCallback, Uniforms, ViewGpu};

/// Something the user did in the view, in the order it happened this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ImageViewEvent {
    /// A mouse button went down over the view (Qt's `mousePressEvent`). The
    /// second press of a double click is reported as [`Self::DoubleClicked`]
    /// instead, as Qt does. Any press makes the camera active in XMALab.
    Pressed {
        button: PointerButton,
        modifiers: Modifiers,
        pos: ImagePos,
    },
    /// The second press of a double click (Qt's `mouseDoubleClickEvent`).
    /// `pos` is computed like a press; XMALab's double-click handler leaves out
    /// the press's -0.5, so it works half a pixel right of and below `pos`
    /// (`GLCameraView::mouseDoubleClickEvent`).
    DoubleClicked {
        button: PointerButton,
        modifiers: Modifiers,
        pos: ImagePos,
    },
    /// A button pressed in this view was released.
    Released {
        button: PointerButton,
        modifiers: Modifiers,
        pos: ImagePos,
    },
    /// Wheel with Ctrl (Command on macOS) held: the view does not zoom. In marker
    /// tracking XMALab changes the mesh transparency by `angle_delta / 2400`.
    /// Not reported for a detail view with the centre locked.
    CtrlWheel { angle_delta: f64 },
    /// The view transform changed by user input or by a fit on resize.
    ViewChanged(ViewChange),
}

/// What [`ImageView::show`] returns.
pub struct ImageViewOutput {
    /// The response for the view's rectangle (sense click and drag).
    pub response: egui::Response,
    /// The transform after this frame's input, as the image was drawn.
    pub transform: ViewTransform,
    /// Input this frame, in order.
    pub events: Vec<ImageViewEvent>,
    /// Image position under the pointer, if the pointer is over the view.
    pub hover: Option<ImagePos>,
    /// Size of the image shown (or of the state's image size without a texture).
    pub image_size: Option<[u32; 2]>,
}

impl ImageViewOutput {
    /// A painter clipped to the view, for overlays drawn after the image.
    pub fn painter(&self, ui: &egui::Ui) -> egui::Painter {
        ui.painter_at(self.response.rect)
    }
}

/// The image view widget. Build it each frame around the view's state.
pub struct ImageView<'a> {
    state: &'a mut ImageViewState,
    texture: Option<(&'a ImageStore, Id)>,
    adjust: ImageAdjust,
    filtering: Filtering,
    reduce_to_8bit: bool,
    background: Option<Color32>,
    size: Option<Vec2>,
}

impl<'a> ImageView<'a> {
    pub fn new(state: &'a mut ImageViewState) -> Self {
        Self {
            state,
            texture: None,
            adjust: ImageAdjust::default(),
            filtering: Filtering::default(),
            reduce_to_8bit: false,
            background: None,
            size: None,
        }
    }

    /// The image to show: the texture stored under `key`. Without a texture (or
    /// before its first upload) nothing is drawn, as XMALab draws nothing when
    /// there is no image; input still works if the state has an image size.
    pub fn texture(mut self, store: &'a ImageStore, key: Id) -> Self {
        self.texture = Some((store, key));
        self
    }

    /// Scale and bias (default identity).
    pub fn adjust(mut self, adjust: ImageAdjust) -> Self {
        self.adjust = adjust;
        self
    }

    /// Interpolation (default [`Filtering::Linear`], as XMALab 3).
    pub fn filtering(mut self, filtering: Filtering) -> Self {
        self.filtering = filtering;
        self
    }

    /// Show deep images reduced to 8 bits first, as the C++ program does
    /// (default off: full depth, parity ledger V1).
    pub fn reduce_to_8bit(mut self, on: bool) -> Self {
        self.reduce_to_8bit = on;
        self
    }

    /// Colour painted outside the image. Default `None`: nothing is painted
    /// there, so whatever is behind the view shows through, as with XMALab's
    /// clear colour (0, 0, 0, 0).
    pub fn background(mut self, color: Option<Color32>) -> Self {
        self.background = color;
        self
    }

    /// Size of the view in points; default all available space.
    pub fn size(mut self, size: Vec2) -> Self {
        self.size = Some(size);
        self
    }

    pub fn show(self, ui: &mut egui::Ui) -> ImageViewOutput {
        let Self {
            state,
            texture,
            adjust,
            filtering,
            reduce_to_8bit,
            background,
            size,
        } = self;
        let size = size.unwrap_or_else(|| ui.available_size());
        // GLCameraView::setMinimumSize(50, 50).
        let size = size.max(Vec2::splat(1.0));
        let (rect, response) = ui.allocate_exact_size(size, Sense::click_and_drag());
        let ppp = ui.ctx().pixels_per_point();

        let tex = texture.and_then(|(store, key)| store.texture(key).map(|t| (store, t)));
        if let Some((_, t)) = &tex {
            state.set_image_size([t.info.width, t.info.height]);
        }

        let mut events = Vec::new();
        let size_px = [
            f64::from(rect.width()) * f64::from(ppp),
            f64::from(rect.height()) * f64::from(ppp),
        ];
        if state.place(rect.min, size_px, ppp) {
            events.push(ImageViewEvent::ViewChanged(ViewChange::Fit));
        }

        handle_input(ui, state, &response, &mut events);

        let transform = state.transform();
        if let Some((store, t)) = tex {
            if state.gpu.as_ref().is_none_or(|g| g.store_id != store.id()) {
                state.gpu = Some(ViewGpu::new(store.device(), store.pipelines(), store.id()));
            }
            if let Some(gpu) = &state.gpu {
                let params = DrawParams {
                    transform,
                    adjust,
                    filtering,
                    reduce_to_8bit,
                    background,
                };
                let pipelines = store.pipelines();
                let callback = ImageCallback {
                    pipeline: pipelines.pipeline.clone(),
                    buffer: gpu.buffer.clone(),
                    view_bind_group: gpu.bind_group.clone(),
                    texture_bind_group: t.bind_group.clone(),
                    uniforms: Uniforms::new(&params, t.meta, pipelines.srgb_target),
                };
                ui.painter_at(rect)
                    .add(egui_wgpu::Callback::new_paint_callback(rect, callback));
            }
        } else if let Some(c) = background {
            ui.painter_at(rect).rect_filled(rect, 0.0, c);
        }

        let hover = response
            .hover_pos()
            .filter(|p| rect.contains(*p))
            .map(|p| transform.screen_to_image(p));
        ImageViewOutput {
            response,
            transform,
            events,
            hover,
            image_size: state.image_size(),
        }
    }
}

fn button_index(b: PointerButton) -> usize {
    match b {
        PointerButton::Primary => 0,
        PointerButton::Secondary => 1,
        PointerButton::Middle => 2,
        PointerButton::Extra1 => 3,
        PointerButton::Extra2 => 4,
    }
}

/// Applies this frame's pointer and wheel events to the state, in order.
///
/// Ported from `ui/GLCameraView.cpp`: `mousePressEvent`, `mouseMoveEvent`,
/// `mouseDoubleClickEvent`, `wheelEvent`.
fn handle_input(
    ui: &egui::Ui,
    state: &mut ImageViewState,
    response: &egui::Response,
    out: &mut Vec<ImageViewEvent>,
) {
    let rect = response.rect;
    let over = response.contains_pointer();
    let (line_speed, dbl_delay, click_dist) = ui.ctx().options(|o| {
        (
            o.input_options.line_scroll_speed,
            o.input_options.max_double_click_delay,
            o.input_options.max_click_dist,
        )
    });
    let (raw, time, latest, secondary_down) = ui.input(|i| {
        (
            i.events.clone(),
            i.time,
            i.pointer.latest_pos(),
            i.pointer.secondary_down(),
        )
    });
    let mut pointer = latest;
    let mut panned = false;
    for event in &raw {
        match *event {
            egui::Event::PointerMoved(p) => {
                pointer = Some(p);
                if state.panning && state.pressed[1] {
                    let t = state.transform();
                    state.pan_to(t.screen_to_view_px(p));
                    panned = true;
                }
            }
            egui::Event::PointerButton {
                pos,
                button,
                pressed: true,
                modifiers,
            } => {
                pointer = Some(pos);
                if !(over && rect.contains(pos)) {
                    continue;
                }
                let idx = button_index(button);
                state.pressed[idx] = true;
                let view_px = state.transform().screen_to_view_px(pos);
                let double = state.last_press.is_some_and(|(t, p, b)| {
                    b == button && time - t <= dbl_delay && (pos - p).length() <= click_dist
                });
                if double {
                    // Qt delivers the second press as a double click only; the
                    // pan anchor keeps its value from the first press.
                    state.last_press = None;
                    if button == PointerButton::Secondary {
                        state.panning = true;
                    }
                    out.push(ImageViewEvent::DoubleClicked {
                        button,
                        modifiers,
                        pos: state.transform().view_px_to_image(view_px),
                    });
                } else {
                    state.last_press = Some((time, pos, button));
                    let right = button == PointerButton::Secondary;
                    if right {
                        state.panning = true;
                    }
                    let image = state.press(view_px, right);
                    out.push(ImageViewEvent::Pressed {
                        button,
                        modifiers,
                        pos: image,
                    });
                }
            }
            egui::Event::PointerButton {
                pos,
                button,
                pressed: false,
                modifiers,
            } => {
                pointer = Some(pos);
                let idx = button_index(button);
                if state.pressed[idx] {
                    state.pressed[idx] = false;
                    if button == PointerButton::Secondary {
                        state.panning = false;
                    }
                    let view_px = state.transform().screen_to_view_px(pos);
                    out.push(ImageViewEvent::Released {
                        button,
                        modifiers,
                        pos: state.transform().view_px_to_image(view_px),
                    });
                }
            }
            egui::Event::MouseWheel {
                unit,
                delta,
                modifiers,
                ..
            } => {
                let Some(p) = pointer else { continue };
                if !(over && rect.contains(p)) {
                    continue;
                }
                if state.kind() == ViewKind::Detail && state.center_locked() {
                    continue;
                }
                // Qt angleDelta: 120 per notch, positive when the wheel turns away
                // from the user (egui: positive y moves content down, the same sign).
                let notches = match unit {
                    MouseWheelUnit::Line => f64::from(delta.y),
                    MouseWheelUnit::Point => f64::from(delta.y / line_speed),
                    MouseWheelUnit::Page => f64::from(delta.y * rect.height() / line_speed),
                };
                let angle_delta = notches * ANGLE_DELTA_PER_NOTCH;
                if angle_delta == 0.0 {
                    continue;
                }
                if modifiers.command {
                    out.push(ImageViewEvent::CtrlWheel { angle_delta });
                } else {
                    let anchor = state.transform().screen_to_view_px(p);
                    if state.wheel(angle_delta, anchor) {
                        out.push(ImageViewEvent::ViewChanged(ViewChange::Wheel));
                    }
                }
            }
            egui::Event::PointerGone => pointer = None,
            _ => {}
        }
    }
    // A release outside the window is not always reported.
    if state.panning && !secondary_down {
        state.panning = false;
        state.pressed[1] = false;
    }
    if panned {
        out.push(ImageViewEvent::ViewChanged(ViewChange::Pan));
    }
}
