//! The view transform of a camera view: zoom ratio and offsets exactly as
//! `GLCameraView` keeps them, and conversions between egui screen points and
//! image coordinates.
//!
//! Ported from `ui/GLCameraView.cpp`. The C++ view sets
//!
//! ```text
//! glOrtho(-0.5*Z*W - xo - 0.5,  0.5*Z*W - xo - 0.5,     // left, right
//!          0.5*Z*H - yo - 0.5, -0.5*Z*H - yo - 0.5)     // bottom, top
//! ```
//!
//! where `Z` is `zoomRatio` (image pixels per device pixel), `W` and `H` the view
//! size in device pixels and `xo`, `yo` the offsets, and draws the image as the
//! quad from (-0.5, -0.5) to (width - 0.5, height - 0.5). So image pixel `i`
//! covers [i - 0.5, i + 0.5]: pixel centres are at integer image coordinates,
//! the convention of this project, and a point at device pixel position `p`
//! from the view's top-left corner is the image point `Z * p - (0.5 * Z * W + xo) - 0.5`,
//! which is the formula `mousePressEvent` uses for clicks.

use egui::{Pos2, Rect, Vec2};

/// Smallest zoom ratio (image pixels per device pixel): 999 %, the zoom box maximum.
///
/// Ported from `ui/GLCameraView.cpp`, `GLCameraView::setZoomRatio`.
pub const MIN_ZOOM_RATIO: f64 = 100.0 / 999.0;

/// Zoom ratio a detail view starts with and returns to when it is re-centred
/// with a zoom reset: 500 %.
///
/// Ported from `ui/GLCameraView.cpp`, `GLCameraView::setDetailedView` and
/// `GLCameraView::centerViewToPoint`.
pub const DETAIL_ZOOM_RATIO: f64 = 0.2;

/// Wheel zoom per unit of Qt's wheel `angleDelta` (120 units per notch): 0.12
/// per notch, as a factor of the zoom ratio by default (see
/// [`super::WheelZoom`]).
///
/// Ported from `ui/GLCameraView.cpp`, `GLCameraView::wheelEvent`.
pub const WHEEL_ZOOM_PER_ANGLE_UNIT: f64 = 1.0 / 1000.0;

/// Qt's `angleDelta` for one wheel notch.
pub const ANGLE_DELTA_PER_NOTCH: f64 = 120.0;

/// A point in image coordinates: pixel (0, 0) is the top-left pixel, pixel
/// centres are at integer coordinates, x to the right and y down.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ImagePos {
    pub x: f64,
    pub y: f64,
}

impl ImagePos {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

impl From<Pos2> for ImagePos {
    fn from(p: Pos2) -> Self {
        Self::new(f64::from(p.x), f64::from(p.y))
    }
}

impl From<[f64; 2]> for ImagePos {
    fn from(p: [f64; 2]) -> Self {
        Self::new(p[0], p[1])
    }
}

impl From<(f64, f64)> for ImagePos {
    fn from(p: (f64, f64)) -> Self {
        Self::new(p.0, p.1)
    }
}

impl From<ImagePos> for Pos2 {
    fn from(p: ImagePos) -> Self {
        Pos2::new(p.x as f32, p.y as f32)
    }
}

/// Where a view is on screen and how it maps image coordinates onto it.
///
/// A plain value: copy it out of [`super::ImageViewOutput`] and use it to draw
/// overlays in image coordinates or to convert pointer positions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewTransform {
    /// `zoomRatio`: image pixels per device (physical) pixel. 100 % in the zoom
    /// box is 1.0, one image pixel per physical screen pixel.
    pub zoom_ratio: f64,
    /// `x_offset`, `y_offset` of `GLCameraView`. The centre of the view shows
    /// the image point (-x_offset - 0.5, -y_offset - 0.5).
    pub offset: [f64; 2],
    /// View size in device pixels (`window_width * devicePixelRatio`, ...).
    pub view_size_px: [f64; 2],
    /// Top-left corner of the view in egui points.
    pub view_min: Pos2,
    /// egui's `pixels_per_point` (Qt's `devicePixelRatio`).
    pub pixels_per_point: f32,
}

impl Default for ViewTransform {
    fn default() -> Self {
        Self {
            zoom_ratio: 1.0,
            offset: [0.0, 0.0],
            view_size_px: [50.0, 50.0],
            view_min: Pos2::ZERO,
            pixels_per_point: 1.0,
        }
    }
}

impl ViewTransform {
    /// Image point under the device pixel position `p` measured from the view's
    /// top-left corner.
    ///
    /// Ported from `ui/GLCameraView.cpp`, `GLCameraView::mousePressEvent`
    /// (`zoomRatio * (pos * devicePixelRatio) - (window * devicePixelRatio * zoomRatio * 0.5 + offset) - 0.5`).
    pub fn view_px_to_image(&self, p: [f64; 2]) -> ImagePos {
        let z = self.zoom_ratio;
        ImagePos::new(
            z * p[0] - (self.view_size_px[0] * z * 0.5 + self.offset[0]) - 0.5,
            z * p[1] - (self.view_size_px[1] * z * 0.5 + self.offset[1]) - 0.5,
        )
    }

    /// Device pixel position, from the view's top-left corner, of an image point.
    /// Inverse of [`Self::view_px_to_image`].
    pub fn image_to_view_px(&self, p: ImagePos) -> [f64; 2] {
        let z = self.zoom_ratio;
        [
            (p.x + 0.5 + self.offset[0]) / z + 0.5 * self.view_size_px[0],
            (p.y + 0.5 + self.offset[1]) / z + 0.5 * self.view_size_px[1],
        ]
    }

    /// Device pixel position within the view of a screen point (egui points).
    pub fn screen_to_view_px(&self, p: Pos2) -> [f64; 2] {
        let ppp = f64::from(self.pixels_per_point);
        [
            (f64::from(p.x) - f64::from(self.view_min.x)) * ppp,
            (f64::from(p.y) - f64::from(self.view_min.y)) * ppp,
        ]
    }

    /// Screen point (egui points) of a device pixel position within the view.
    pub fn view_px_to_screen(&self, p: [f64; 2]) -> Pos2 {
        let ppp = f64::from(self.pixels_per_point);
        Pos2::new(
            (f64::from(self.view_min.x) + p[0] / ppp) as f32,
            (f64::from(self.view_min.y) + p[1] / ppp) as f32,
        )
    }

    /// Image point under a screen point (egui points).
    pub fn screen_to_image(&self, p: Pos2) -> ImagePos {
        self.view_px_to_image(self.screen_to_view_px(p))
    }

    /// Screen point (egui points) at which an image point is drawn. Use this to
    /// draw overlays: a cross at `image_to_screen((x, y))` sits on the centre of
    /// pixel (x, y) when x and y are integers.
    pub fn image_to_screen(&self, p: impl Into<ImagePos>) -> Pos2 {
        self.view_px_to_screen(self.image_to_view_px(p.into()))
    }

    /// Screen rectangle covered by image pixel (`x`, `y`): from (x - 0.5, y - 0.5)
    /// to (x + 0.5, y + 0.5).
    pub fn pixel_rect_on_screen(&self, x: i64, y: i64) -> Rect {
        let (x, y) = (x as f64, y as f64);
        Rect::from_min_max(
            self.image_to_screen((x - 0.5, y - 0.5)),
            self.image_to_screen((x + 0.5, y + 0.5)),
        )
    }

    /// The view's rectangle in egui points.
    pub fn view_rect(&self) -> Rect {
        let ppp = self.pixels_per_point;
        Rect::from_min_size(
            self.view_min,
            Vec2::new(
                self.view_size_px[0] as f32 / ppp,
                self.view_size_px[1] as f32 / ppp,
            ),
        )
    }

    /// The image point at the centre of the view.
    pub fn center(&self) -> ImagePos {
        ImagePos::new(-self.offset[0] - 0.5, -self.offset[1] - 0.5)
    }

    /// Zoom ratio that fits an image of `image_size` pixels in the view: the larger
    /// of width ratio and height ratio.
    ///
    /// Ported from `ui/GLCameraView.cpp`, `GLCameraView::setZoomToFit`.
    pub fn fit_zoom_ratio(&self, image_size: [u32; 2]) -> f64 {
        let rw = f64::from(image_size[0]) / self.view_size_px[0];
        let rh = f64::from(image_size[1]) / self.view_size_px[1];
        if rw > rh { rw } else { rh }
    }

    /// Offsets `setZoomToFit` sets for the current zoom ratio. They put the
    /// image's top-left corner (-0.5, -0.5) at the view's top-left corner; the
    /// image is not centred.
    ///
    /// Ported from `ui/GLCameraView.cpp`, `GLCameraView::setZoomToFit`.
    pub fn fit_offsets(&self) -> [f64; 2] {
        [
            -0.5 * (self.zoom_ratio * self.view_size_px[0]),
            -0.5 * (self.zoom_ratio * self.view_size_px[1]),
        ]
    }

    /// Offsets after the zoom ratio changed from `zoom_prev` to the current one
    /// with the cursor at device pixel position `anchor`, so that the image point
    /// under the cursor stays there.
    ///
    /// Ported from `ui/GLCameraView.cpp`, `GLCameraView::wheelEvent`.
    pub fn offsets_after_zoom_about(&self, zoom_prev: f64, anchor: [f64; 2]) -> [f64; 2] {
        let dz = zoom_prev - self.zoom_ratio;
        [
            self.offset[0] + dz * (0.5 * self.view_size_px[0] - anchor[0]),
            self.offset[1] + dz * (0.5 * self.view_size_px[1] - anchor[1]),
        ]
    }

    /// Clamps the offsets after a pan: an image narrower than the view stays
    /// wholly inside it, and a wider image always covers the view.
    ///
    /// Ported from `ui/GLCameraView.cpp`, `GLCameraView::clampXY`.
    pub fn clamped(mut self, image_size: [u32; 2]) -> Self {
        let z = self.zoom_ratio;
        for ((o, &size), &view) in self
            .offset
            .iter_mut()
            .zip(&image_size)
            .zip(&self.view_size_px)
        {
            let cam = f64::from(size);
            let half = 0.5 * (z * view);
            if cam < view * z {
                if *o < -half {
                    *o = -half;
                }
                if *o > -cam + half {
                    *o = -cam + half;
                }
            } else {
                if *o > -half {
                    *o = -half;
                }
                if *o < -cam + half {
                    *o = -cam + half;
                }
            }
        }
        self
    }

    /// The zoom box value for a zoom ratio: `floor(100 / zoomRatio + 0.5)`.
    ///
    /// Ported from `ui/GLCameraView.cpp`, `GLCameraView::setZoomRatio`.
    pub fn zoom_percent_of(zoom_ratio: f64) -> i32 {
        (100.0 / zoom_ratio + 0.5).floor() as i32
    }
}
