//! Per-view state: zoom, offsets, "fit", the zoom box coupling, and the mouse
//! handling of `GLCameraView`, kept independent of egui's input so that it can
//! be tested as plain arithmetic.
//!
//! Ported from `ui/GLCameraView.cpp` and the zoom box wiring in
//! `ui/CameraViewWidget.cpp`.

use egui::Pos2;

use super::render::ViewGpu;
use super::transform::{
    ANGLE_DELTA_PER_NOTCH, DETAIL_ZOOM_RATIO, ImagePos, MIN_ZOOM_RATIO, ViewTransform,
    WHEEL_ZOOM_PER_ANGLE_UNIT,
};

/// Largest value of the zoom box (`spinBoxZoom`, range 0 to 999).
pub const ZOOM_PERCENT_MAX: i32 = 999;

/// Which of the two uses of `GLCameraView` a view is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewKind {
    /// A camera view in the central area: starts at 100 % with "fit" on, and has
    /// a zoom box whose integer value feeds back into the zoom ratio.
    Camera,
    /// A view in the Detailed View panel (`setDetailedView`): starts at 500 %
    /// with "fit" off and has no zoom box.
    Detail,
}

/// How a wheel step changes the zoom ratio.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WheelZoom {
    /// `ratio * (1 - angleDelta / 1000)`: each notch zooms by the same factor
    /// (0.88 in, 1.12 out), at any zoom.
    #[default]
    Factor,
    /// XMALab 3's step, kept for reference: `ratio * 1 - angleDelta / 1000`,
    /// which subtracts 0.12 per notch whatever the zoom. An operator-precedence
    /// bug in `GLCameraView::wheelEvent`: steps are huge when zoomed in (from
    /// ratio 0.2, one notch in jumps to the 999 % limit) and tiny when zoomed out.
    Xmalab3Linear,
}

/// What changed the view transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewChange {
    /// Right-button drag.
    Pan,
    /// Mouse wheel zoom about the cursor.
    Wheel,
    /// "Fit" applied because the view was resized (or first shown) with fit on.
    Fit,
}

/// State of one image view. Owned by the caller, one per view; the widget
/// [`super::ImageView`] reads and updates it each frame.
pub struct ImageViewState {
    kind: ViewKind,
    zoom_ratio: f64,
    offset: [f64; 2],
    autozoom: bool,
    /// Value of the zoom box (`spinBoxZoom`); `None` for a detail view.
    zoom_box: Option<i32>,
    /// `camera_width`, `camera_height`.
    image_size: Option<[u32; 2]>,
    view_size_px: Option<[f64; 2]>,
    view_min: Pos2,
    pixels_per_point: f32,
    center_locked: bool,
    wheel_zoom: WheelZoom,
    /// `prev_x`, `prev_y`: zoom ratio times the pointer position at the last
    /// right press or pan step, in device pixels.
    pan_prev: [f64; 2],
    pub(crate) panning: bool,
    pub(crate) pressed: [bool; 5],
    pub(crate) last_press: Option<(f64, Pos2, egui::PointerButton)>,
    pub(crate) gpu: Option<ViewGpu>,
}

impl ImageViewState {
    /// A camera view: zoom 100 %, "fit" on (applied as soon as the view has a
    /// size), offsets zero.
    ///
    /// Ported from `ui/GLCameraView.cpp`, `GLCameraView::GLCameraView`
    /// (`setZoomRatio(1.0, true)`) and the zoom box's initial value 100.
    pub fn camera() -> Self {
        Self::new(ViewKind::Camera, 1.0, true, Some(100))
    }

    /// A detail view: zoom 500 %, "fit" off, offsets zero.
    ///
    /// Ported from `ui/GLCameraView.cpp`, `GLCameraView::setDetailedView`.
    pub fn detail() -> Self {
        Self::new(ViewKind::Detail, DETAIL_ZOOM_RATIO, false, None)
    }

    fn new(kind: ViewKind, zoom_ratio: f64, autozoom: bool, zoom_box: Option<i32>) -> Self {
        Self {
            kind,
            zoom_ratio,
            offset: [0.0, 0.0],
            autozoom,
            zoom_box,
            image_size: None,
            view_size_px: None,
            view_min: Pos2::ZERO,
            pixels_per_point: 1.0,
            center_locked: false,
            wheel_zoom: WheelZoom::default(),
            pan_prev: [0.0, 0.0],
            panning: false,
            pressed: [false; 5],
            last_press: None,
            gpu: None,
        }
    }

    pub fn kind(&self) -> ViewKind {
        self.kind
    }

    /// The current transform. Before the view has been shown its size is 50 by 50
    /// device pixels (`GLCameraView`'s initial `window_width`).
    pub fn transform(&self) -> ViewTransform {
        ViewTransform {
            zoom_ratio: self.zoom_ratio,
            offset: self.offset,
            view_size_px: self.view_size_px.unwrap_or([50.0, 50.0]),
            view_min: self.view_min,
            pixels_per_point: self.pixels_per_point,
        }
    }

    /// Whether "fit" (the checked fit button, `autozoom`) is on.
    pub fn auto_zoom(&self) -> bool {
        self.autozoom
    }

    /// The value the zoom box shows. For a detail view, which has no zoom box,
    /// the value computed the same way from the zoom ratio.
    pub fn zoom_percent(&self) -> i32 {
        self.zoom_box
            .unwrap_or_else(|| ViewTransform::zoom_percent_of(self.zoom_ratio))
    }

    /// Image size in pixels (`camera_width`, `camera_height`), if known.
    pub fn image_size(&self) -> Option<[u32; 2]> {
        self.image_size
    }

    /// Sets the image size (`GLCameraView::setCamera`). The widget also sets it
    /// from the texture it shows. If "fit" is on and the view has a size, the
    /// image is fitted when the size changes.
    pub fn set_image_size(&mut self, size: [u32; 2]) {
        if self.image_size != Some(size) {
            self.image_size = Some(size);
            if self.autozoom && self.view_size_px.is_some() {
                self.set_zoom_to_fit();
            }
        }
    }

    /// For a detail view with XMALab's "centre detail view" setting on: the wheel
    /// is ignored and the caller re-centres the view on the marker every frame
    /// with [`Self::center_on`].
    ///
    /// Ported from `ui/GLCameraView.cpp`, `GLCameraView::wheelEvent` and `paintGL`.
    pub fn set_center_locked(&mut self, locked: bool) {
        self.center_locked = locked;
    }

    pub fn center_locked(&self) -> bool {
        self.center_locked
    }

    /// How the wheel zooms (default [`WheelZoom::Factor`]).
    pub fn set_wheel_zoom(&mut self, mode: WheelZoom) {
        self.wheel_zoom = mode;
    }

    pub fn wheel_zoom(&self) -> WheelZoom {
        self.wheel_zoom
    }

    /// Sets the zoom ratio and the fit flag, with the zoom box feedback of a
    /// camera view: the new ratio is shown as `floor(100 / ratio + 0.5)` %, and if
    /// that changes the zoom box value, the box sets the ratio back to
    /// 100 / value. So after a wheel step or a fit, a camera view's ratio is
    /// 100 / (integer percent), unless the box already showed that percent.
    ///
    /// Ported from `ui/GLCameraView.cpp`, `GLCameraView::setZoomRatio`, with
    /// `CameraViewWidget::zoomChanged` and `on_spinBoxZoom_valueChanged`.
    pub fn set_zoom_ratio(&mut self, zoom_ratio: f64, autozoom: bool) {
        let z = if zoom_ratio > MIN_ZOOM_RATIO {
            zoom_ratio
        } else {
            MIN_ZOOM_RATIO
        };
        if self.zoom_ratio != z {
            self.zoom_ratio = z;
            let percent = ViewTransform::zoom_percent_of(z);
            if let Some(shown) = self.zoom_box
                && percent != shown
            {
                let value = percent.clamp(0, ZOOM_PERCENT_MAX);
                if value != shown {
                    self.zoom_box = Some(value);
                    // valueChanged -> setZoom(value) with the fit flag as it is now.
                    let current = self.autozoom;
                    self.set_zoom_ratio(Self::ratio_of_percent(value), current);
                }
            }
        }
        self.autozoom = autozoom;
    }

    /// The zoom box was changed to `percent`: the ratio becomes 100 / percent,
    /// about the view centre; the fit flag is unchanged.
    ///
    /// Ported from `ui/GLCameraView.cpp`, `GLCameraView::setZoom`.
    pub fn set_zoom_percent(&mut self, percent: i32) {
        let percent = percent.clamp(0, ZOOM_PERCENT_MAX);
        if self.zoom_box.is_some() {
            self.zoom_box = Some(percent);
        }
        let current = self.autozoom;
        self.set_zoom_ratio(Self::ratio_of_percent(percent), current);
    }

    /// 100 / percent. The zoom box allows 0, which in the C++ makes the ratio
    /// infinite (the image disappears and the next wheel step makes the offsets
    /// NaN); here 0 is treated as 1.
    fn ratio_of_percent(percent: i32) -> f64 {
        100.0 / f64::from(percent.max(1))
    }

    /// The fit button: on fits the image and keeps fitting it on every resize;
    /// off only clears the flag.
    ///
    /// Ported from `ui/GLCameraView.cpp`, `GLCameraView::setAutoZoom`.
    pub fn set_auto_zoom(&mut self, on: bool) {
        let z = self.zoom_ratio;
        self.set_zoom_ratio(z, on);
        if on {
            self.set_zoom_to_fit();
        }
    }

    /// Fits the image: ratio = max(width ratio, height ratio) (through the zoom
    /// box feedback), image top-left corner at the view's top-left corner.
    /// Does nothing until both the image size and the view size are known.
    ///
    /// Ported from `ui/GLCameraView.cpp`, `GLCameraView::setZoomToFit`.
    pub fn set_zoom_to_fit(&mut self) {
        let (Some(image), Some(_)) = (self.image_size, self.view_size_px) else {
            return;
        };
        let z = self.transform().fit_zoom_ratio(image);
        let autozoom = self.autozoom;
        self.set_zoom_ratio(z, autozoom);
        self.offset = self.transform().fit_offsets();
    }

    /// Puts image point (`x`, `y`) at the view centre as the C++ does, by setting
    /// the offsets to (-x, -y). The view centre then shows (x - 0.5, y - 0.5), so
    /// the point itself is drawn 0.5 / zoom ratio device pixels right of and below
    /// the centre (2.5 device pixels at 500 %). With `reset_zoom` the zoom ratio
    /// returns to 0.2 (500 %) and "fit" goes off.
    ///
    /// Ported from `ui/GLCameraView.cpp`, `GLCameraView::centerViewToPoint`.
    pub fn center_on(&mut self, p: impl Into<ImagePos>, reset_zoom: bool) {
        let p = p.into();
        self.offset = [-p.x, -p.y];
        if reset_zoom {
            self.set_zoom_ratio(DETAIL_ZOOM_RATIO, false);
        }
    }

    /// Sets the offsets directly (`x_offset`, `y_offset`).
    pub fn set_offset(&mut self, offset: [f64; 2]) {
        self.offset = offset;
    }

    /// The view's placement on screen for this frame. A change of size in device
    /// pixels with "fit" on fits the image again; without "fit" the zoom ratio
    /// and the image point at the view centre stay as they are.
    /// Returns true when a fit was applied.
    ///
    /// Ported from `ui/GLCameraView.cpp`, `GLCameraView::resizeGL`.
    pub fn place(&mut self, view_min: Pos2, view_size_px: [f64; 2], pixels_per_point: f32) -> bool {
        self.view_min = view_min;
        self.pixels_per_point = pixels_per_point;
        if self.view_size_px != Some(view_size_px) {
            self.view_size_px = Some(view_size_px);
            if self.autozoom && self.image_size.is_some() {
                self.set_zoom_to_fit();
                return true;
            }
        }
        false
    }

    /// Image position of a press at device pixel position `p` in the view, and
    /// the start of a pan for the right button.
    ///
    /// Ported from `ui/GLCameraView.cpp`, `GLCameraView::mousePressEvent`.
    pub fn press(&mut self, p: [f64; 2], right_button: bool) -> ImagePos {
        if right_button {
            self.pan_prev = [self.zoom_ratio * p[0], self.zoom_ratio * p[1]];
        }
        self.transform().view_px_to_image(p)
    }

    /// The pointer moved to device pixel position `p` with the right button held
    /// after a press in this view: pan so the image follows the pointer, then
    /// clamp (`clampXY`).
    ///
    /// Ported from `ui/GLCameraView.cpp`, `GLCameraView::mouseMoveEvent`.
    pub fn pan_to(&mut self, p: [f64; 2]) {
        let z = self.zoom_ratio;
        self.offset[0] -= self.pan_prev[0] - z * p[0];
        self.offset[1] -= self.pan_prev[1] - z * p[1];
        self.pan_prev = [z * p[0], z * p[1]];
        if let Some(size) = self.image_size {
            let t = self.transform().clamped(size);
            self.offset = t.offset;
        }
    }

    /// A wheel step of Qt `angle_delta` units (120 per notch, positive away from
    /// the user) with the cursor at device pixel position `anchor`: zoom ratio
    /// times (1 - angle_delta / 1000) (times 0.88^(1000/120), about 0.344, for an
    /// event of 1000 units or more), or minus angle_delta / 1000 with
    /// [`WheelZoom::Xmalab3Linear`] (through the zoom box feedback), fit off, the
    /// image point under the cursor kept in place. Returns false when the wheel is
    /// ignored (a detail view with the centre locked). Ctrl + wheel is handled by
    /// the caller and never reaches here.
    ///
    /// Ported from `ui/GLCameraView.cpp`, `GLCameraView::wheelEvent`.
    pub fn wheel(&mut self, angle_delta: f64, anchor: [f64; 2]) -> bool {
        if self.kind == ViewKind::Detail && self.center_locked {
            return false;
        }
        let zoom_prev = self.zoom_ratio;
        let step = angle_delta * WHEEL_ZOOM_PER_ANGLE_UNIT;
        let z = match self.wheel_zoom {
            // An event of 1000 units or more (a coalesced fast wheel, a page-unit
            // event) would make the factor zero or negative and jump to the 999 %
            // limit; it zooms in as 1000 units of 120-unit notches would, each
            // times 0.88 (review 12).
            WheelZoom::Factor if step >= 1.0 || step.is_nan() => {
                let per_notch = 1.0 - ANGLE_DELTA_PER_NOTCH * WHEEL_ZOOM_PER_ANGLE_UNIT;
                self.zoom_ratio * per_notch.powf(1000.0 / ANGLE_DELTA_PER_NOTCH)
            }
            WheelZoom::Factor => self.zoom_ratio * (1.0 - step),
            WheelZoom::Xmalab3Linear => self.zoom_ratio * 1.0 - step,
        };
        self.set_zoom_ratio(z, false);
        self.offset = self.transform().offsets_after_zoom_about(zoom_prev, anchor);
        true
    }
}
