//! One step of XMALab's 2D template tracker, a port of `processing/MarkerTracking.cpp`
//! (CPU branch; the OpenCL branch is not ported).
//!
//! A template of `2 (size + 3) + 1` pixels is cut around the marker's position in the frame
//! tracked from ([`cut_template`]); in the next frame a region of `2 (size + 33) + 1`
//! pixels is cut around the predicted position, the template is correlated with it
//! (`TM_CCORR_NORMED`), the correlation is stretched to `0 .. 100 - maxPenalty`, a penalty
//! surface growing away from the predicted position and scaled by `maxPenalty` is
//! subtracted, and the best place wins ([`track_step_with`]). The result is a whole pixel; the
//! marker's prediction and the detection run afterwards belong to the caller.
//!
//! Every entry point takes a [`TrackCompat`]: the caller names the mode (the corrected
//! behaviour, [`TrackCompat::FIXED`], or XMALab 3's, [`TrackCompat::XMALAB3`]).
//!
//! In the XMALab 3 mode every stage reproduces the C++ with OpenCV 4.13 (no IPP) to the bit;
//! see `crates/xma-vision/tests/tracking_step.rs`.
//!
//! Images deeper than 8 bits (parity ledger V1, `crate::depth`): [`cut_template_u16`] and
//! [`track_step_u16_with`] read the image on the 8-bit scale and run the same step on `f32`
//! patches: the template is cut with OpenCV's float `getRectSubPix` (its sub-pixel values
//! are not rounded to whole grey levels), the search region is an exact copy, and the
//! correlation is OpenCV's `matchTemplate` for `CV_32F` inputs (transformed in double
//! precision). Everything after the correlation is shared with the 8-bit step.

use crate::depth::on_8bit_scale;
use crate::image::{Image, ImageF32, ImageU8, ImageU16};
use crate::imgproc::arith::{min_max_loc_f32, normalize_minmax_f32};
use crate::imgproc::subpix::{
    get_sub_image, get_sub_image_32f, get_sub_image_at, get_sub_image_at_32f,
};
use crate::imgproc::template::{TemplatePixel, match_template_impl};

/// Half-width of the search, in pixels, around the predicted position (`searchArea`).
pub const SEARCH_AREA: i32 = 30;

/// Which faults of XMALab 3's tracker step to reproduce. Each field is `true` for the C++
/// behaviour. [`Default`] and [`TrackCompat::FIXED`] fix both; [`TrackCompat::XMALAB3`] is
/// the C++. (xma-model's `CppCompat` holds the same two switches under the same names.)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TrackCompat {
    /// Ledger A64: the penalty surface is centred at `cols / 2` (30.5 for the 61-wide
    /// surface), half a pixel right of and below the search centre (the prediction rounded
    /// to a whole pixel), whatever the prediction's fraction. Fixed: centred on the predicted
    /// position itself, at surface index `x_pred - off_x - size - 3` (and the same in y),
    /// which lies in `[29.5, 30.5)` ([`penalty_surface_at`]).
    pub penalty_off_centre: bool,
    /// Ledger A65: the search region's offset is `(int)(v + 0.5)`, which rounds towards
    /// zero for a negative `v`. Fixed: `floor(v + 0.5)`.
    pub window_round_toward_zero: bool,
}

impl TrackCompat {
    /// Both faults fixed: the default.
    pub const FIXED: TrackCompat = TrackCompat {
        penalty_off_centre: false,
        window_round_toward_zero: false,
    };

    /// Everything as XMALab 3 does it.
    pub const XMALAB3: TrackCompat = TrackCompat {
        penalty_off_centre: true,
        window_round_toward_zero: true,
    };
}

/// The template cut from the frame tracked from (the state the `MarkerTracking`
/// constructor sets up): 8-bit ([`cut_template`]), or `f32` on the 8-bit scale for a deeper
/// image ([`cut_template_u16`]).
#[derive(Debug, Clone, PartialEq)]
pub struct MarkerTemplate<T = u8> {
    /// `size` of the C++: the marker's size rounded to a whole pixel, at least 5.
    pub size: i32,
    /// The `2 (size + 3) + 1` square around the marker (`templ`).
    pub image: Image<T>,
}

/// `size` of `MarkerTracking`: `(int)(marker size + 0.5)`, at least 5. The conversion
/// truncates toward zero, as the C cast does.
///
/// Ported from processing/MarkerTracking.cpp, MarkerTracking::MarkerTracking.
pub fn template_size(marker_size: f64) -> i32 {
    let size = (marker_size + 0.5) as i32;
    if size < 5 { 5 } else { size }
}

/// Cuts the template around the marker's position `(x_from, y_from)` in the frame tracked
/// from. `marker_size` is `Marker::getSize()`.
///
/// Ported from processing/MarkerTracking.cpp, MarkerTracking::MarkerTracking.
///
/// A template whose half-width `size + 3` exceeds
/// [`crate::imgproc::subpix::max_window_half`] of the frame (a marker size of more than
/// about 1000 pixels, an infinite one, or one beyond the `int` range, where `size + 3`
/// overflows in the C++) is not cut: its image is empty (0 by 0), and every step with it
/// finds nothing (parity ledger A167). A marker's mean size is at most 50 (only sizes from 1
/// to 50 are counted), so the engine never asks for one.
pub fn cut_template(
    frame_from: &ImageU8,
    x_from: f64,
    y_from: f64,
    marker_size: f64,
) -> MarkerTemplate {
    let size = template_size(marker_size);
    MarkerTemplate {
        size,
        image: get_sub_image(frame_from, size.saturating_add(3), x_from, y_from),
    }
}

/// [`cut_template`] for a `bit_depth`-bit frame (8 to 16 bits in `u16`): the image is read
/// on the 8-bit scale (`value / 2^(bit_depth - 8)`) and the template cut with OpenCV's float
/// `getRectSubPix` (`CV_32F` to `CV_32F`), so its sub-pixel values keep their fractions
/// where the 8-bit template rounds them to whole grey levels.
///
/// Ported from processing/MarkerTracking.cpp, MarkerTracking::MarkerTracking (extends
/// [`cut_template`]), with OpenCV imgproc/src/samplers.cpp, getRectSubPix for `CV_32F`
/// ([`get_sub_image_32f`]).
///
/// An over-large marker size gives an empty template, as in [`cut_template`].
///
/// # Panics
/// If `bit_depth` is not 8 to 16.
pub fn cut_template_u16(
    frame_from: &ImageU16,
    bit_depth: u8,
    x_from: f64,
    y_from: f64,
    marker_size: f64,
) -> MarkerTemplate<f32> {
    let size = template_size(marker_size);
    MarkerTemplate {
        size,
        image: get_sub_image_32f(
            &on_8bit_scale(frame_from, bit_depth),
            size.saturating_add(3),
            x_from,
            y_from,
        ),
    }
}

/// What the marker's prediction gave for the frame tracked to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prediction {
    /// The return value of `Marker::getMarkerPrediction` (0 or 1: weak prediction, 2: from
    /// the two previous frames, and so on). The predicted position is passed separately.
    Returned(i32),
    /// A position override was given (`m_hasOverride`); the C++ then counts the prediction
    /// as 1.
    Override,
}

impl Prediction {
    /// The value the C++ compares with 1.
    fn value(self) -> i32 {
        match self {
            Prediction::Returned(v) => v,
            Prediction::Override => 1,
        }
    }
}

/// The outcome of one tracking step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackStep {
    /// The tracked position (`x_to`, `y_to`): whole pixels.
    pub x: i32,
    /// See [`TrackStep::x`].
    pub y: i32,
    /// Top-left corner of the search region in the frame (`off_x`, `off_y`).
    pub off_x: i32,
    /// See [`TrackStep::off_x`].
    pub off_y: i32,
    /// The penalty actually applied: the marker's, divided by 3 (integer division) when
    /// the prediction is weak.
    pub max_penalty: i32,
    /// The best place in the penalised correlation (`maxLoc`), relative to the region.
    pub max_loc: (usize, usize),
    /// The value there (`maxVal`).
    pub max_val: f32,
}

/// The intermediate surfaces of a step, for checking against the reference.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackSurfaces<T = u8> {
    /// The search region (`ROI_to`).
    pub roi: Image<T>,
    /// `matchTemplate` result.
    pub correlation: ImageF32,
    /// The correlation after `normalize(0, 100 - maxPenalty, NORM_MINMAX)`.
    pub normalized: ImageF32,
    /// The normalised penalty surface.
    pub penalty: ImageF32,
    /// `penalty * maxPenalty` (`springforce`).
    pub spring: ImageF32,
    /// `normalized - spring`, where the maximum is taken.
    pub penalised: ImageF32,
}

/// One step of the tracker: finds `template` in `frame_to` near the predicted position
/// `(x_pred, y_pred)` (or the override position, with [`Prediction::Override`]).
/// `max_penalty` is `Marker::getMaxPenalty()`; `compat` names the mode
/// ([`TrackCompat::FIXED`] for the corrected step, [`TrackCompat::XMALAB3`] for the C++).
///
/// Returns `None` when the search region is smaller than the template (`result_cols` or
/// `result_rows` not positive). The C++ then leaves the point at the predicted position
/// and still marks it tracked; that cannot happen with the sizes the C++ uses (the result
/// is always 61 by 61), so it is the caller's to handle. Also `None` when the predicted
/// position is NaN, infinite or so far out that the C++'s `int` arithmetic overflows
/// (ledger A160), when the template or the search region is beyond the window bound
/// (an empty template from [`cut_template`], ledger A167), and when `100 - max_penalty`
/// overflows `int` (`max_penalty` of `i32::MIN + 100` or less, ledger A168).
///
/// Ported from processing/MarkerTracking.cpp, MarkerTracking::trackMarker_thread (CPU
/// branch, `else` at line 245).
pub fn track_step_with(
    template: &MarkerTemplate,
    frame_to: &ImageU8,
    x_pred: f64,
    y_pred: f64,
    prediction: Prediction,
    max_penalty: i32,
    compat: TrackCompat,
) -> Option<TrackStep> {
    track_step_surfaces_with(
        template,
        frame_to,
        x_pred,
        y_pred,
        prediction,
        max_penalty,
        compat,
    )
    .map(|(s, _)| s)
}

/// [`track_step_with`] with its intermediate surfaces.
///
/// Ported from processing/MarkerTracking.cpp, MarkerTracking::trackMarker_thread (CPU
/// branch).
pub fn track_step_surfaces_with(
    template: &MarkerTemplate,
    frame_to: &ImageU8,
    x_pred: f64,
    y_pred: f64,
    prediction: Prediction,
    max_penalty: i32,
    compat: TrackCompat,
) -> Option<(TrackStep, TrackSurfaces)> {
    let (used_size, off_x, off_y) = search_region(template.size, x_pred, y_pred, compat)?;
    let roi = get_sub_image_at(frame_to, used_size, off_x, off_y);
    track_in_region(
        template,
        roi,
        (x_pred, y_pred),
        (off_x, off_y),
        prediction,
        max_penalty,
        compat,
    )
}

/// [`track_step_with`] for a `bit_depth`-bit frame (8 to 16 bits in `u16`) and a template
/// from [`cut_template_u16`]: the search region is cut from the frame on the 8-bit scale
/// (`value / 2^(bit_depth - 8)`, an exact copy at whole-pixel offsets) and correlated with
/// the float template by OpenCV's `matchTemplate` for `CV_32F` inputs; the normalisation,
/// penalty and maximum are those of [`track_step_with`], in the mode `compat` names.
///
/// Ported from processing/MarkerTracking.cpp, MarkerTracking::trackMarker_thread (CPU
/// branch; extends [`track_step_with`]), with OpenCV imgproc/src/templmatch.cpp,
/// cv::matchTemplate for `CV_32F` ([`crate::imgproc::template::match_template_ccorr_normed_f32`]).
///
/// # Panics
/// If `bit_depth` is not 8 to 16.
#[allow(clippy::too_many_arguments)]
pub fn track_step_u16_with(
    template: &MarkerTemplate<f32>,
    frame_to: &ImageU16,
    bit_depth: u8,
    x_pred: f64,
    y_pred: f64,
    prediction: Prediction,
    max_penalty: i32,
    compat: TrackCompat,
) -> Option<TrackStep> {
    track_step_surfaces_u16_with(
        template,
        frame_to,
        bit_depth,
        x_pred,
        y_pred,
        prediction,
        max_penalty,
        compat,
    )
    .map(|(s, _)| s)
}

/// [`track_step_u16_with`] with its intermediate surfaces.
///
/// Ported from processing/MarkerTracking.cpp, MarkerTracking::trackMarker_thread (CPU
/// branch; extends [`track_step_surfaces_with`]).
///
/// # Panics
/// If `bit_depth` is not 8 to 16.
#[allow(clippy::too_many_arguments)]
pub fn track_step_surfaces_u16_with(
    template: &MarkerTemplate<f32>,
    frame_to: &ImageU16,
    bit_depth: u8,
    x_pred: f64,
    y_pred: f64,
    prediction: Prediction,
    max_penalty: i32,
    compat: TrackCompat,
) -> Option<(TrackStep, TrackSurfaces<f32>)> {
    let (used_size, off_x, off_y) = search_region(template.size, x_pred, y_pred, compat)?;
    let roi = get_sub_image_at_32f(&on_8bit_scale(frame_to, bit_depth), used_size, off_x, off_y);
    track_in_region(
        template,
        roi,
        (x_pred, y_pred),
        (off_x, off_y),
        prediction,
        max_penalty,
        compat,
    )
}

/// Half-width and top-left corner of the search region:
/// `used_size = size + searchArea + 3` and `(int)(x - used_size + 0.5)`. In XMALab 3 the cast
/// truncates toward zero, so a negative offset is rounded up rather than to nearest (parity
/// ledger A65, [`TrackCompat::window_round_toward_zero`]); fixed, it is `floor(v + 0.5)`.
///
/// Ported from processing/MarkerTracking.cpp, MarkerTracking::trackMarker_thread.
///
/// `None` (ledger A160) when the C++ cannot compute the region in `int` (a prediction that is
/// NaN, infinite or beyond about +-2^31: undefined behaviour); the step then finds nothing.
fn search_region(
    size: i32,
    x_pred: f64,
    y_pred: f64,
    compat: TrackCompat,
) -> Option<(i32, i32, i32)> {
    let used_size = size.checked_add(SEARCH_AREA + 3)?;
    let int_range = f64::from(i32::MIN)..f64::from(i32::MAX);
    let vx = x_pred - f64::from(used_size) + 0.5;
    let vy = y_pred - f64::from(used_size) + 0.5;
    if !int_range.contains(&vx) || !int_range.contains(&vy) {
        return None;
    }
    let round = |v: f64| {
        if compat.window_round_toward_zero {
            v as i32
        } else {
            v.floor() as i32
        }
    };
    let off_x = round(vx);
    let off_y = round(vy);
    // The window, and the template's offset added to it (`off + size + 3 + max_loc`), stay
    // within it too.
    crate::imgproc::subpix::window_in_int_range(i64::from(off_x), i64::from(off_y), used_size)
        .then_some((used_size, off_x, off_y))
}

/// The step after the search region is cut: correlation, normalisation, penalty, maximum.
/// Shared by the 8-bit and the deep step. `pred` is the predicted position (the search
/// centre), `off` the region's top-left corner.
///
/// Ported from processing/MarkerTracking.cpp, MarkerTracking::trackMarker_thread (CPU
/// branch, `else` at line 245).
fn track_in_region<P: TemplatePixel + Default>(
    template: &MarkerTemplate<P>,
    roi: Image<P>,
    pred: (f64, f64),
    (off_x, off_y): (i32, i32),
    prediction: Prediction,
    max_penalty: i32,
    compat: TrackCompat,
) -> Option<(TrackStep, TrackSurfaces<P>)> {
    let size = template.size;
    // `maxPenalty /= 3` on an int: integer division, toward zero.
    let max_penalty = if prediction.value() <= 1 {
        max_penalty / 3
    } else {
        max_penalty
    };

    // Ledger A168: `100 - maxPenalty` is an `int` in the C++; a penalty of `INT_MIN + 100`
    // or less (a project file can hold any `int`) overflows it, and finds nothing.
    let top = 100i32.checked_sub(max_penalty)?;

    let templ = &template.image;
    // Ledger A167: a template or a search region beyond the window bound is empty (not cut).
    if templ.is_empty() || roi.is_empty() {
        return None;
    }
    let result_cols = roi.width() as i64 - templ.width() as i64 + 1;
    let result_rows = roi.height() as i64 - templ.height() as i64 + 1;
    if result_cols <= 0 || result_rows <= 0 {
        return None;
    }

    // Ledger A64: XMALab 3's surface is centred half a pixel off the search centre; fixed,
    // it is centred on the prediction: result index `i` is the template centred at pixel
    // `off + size + 3 + i`, so the prediction is at index `pred - off - size - 3`.
    let penalty = if compat.penalty_off_centre {
        penalty_surface(result_rows as usize, result_cols as usize)
    } else {
        let lead = f64::from(size + 3);
        penalty_surface_at(
            result_rows as usize,
            result_cols as usize,
            pred.0 - f64::from(off_x) - lead,
            pred.1 - f64::from(off_y) - lead,
        )
    };
    let correlation = match_template_impl(&roi, templ);
    let normalized = normalize_minmax_f32(&correlation, 0.0, f64::from(top));
    // cv::multiply(penaltyMat, Scalar((float)maxPenalty), springforce)
    let mp = max_penalty as f32;
    let spring = penalty.map(|v| v * mp);
    // result = result - springforce
    let penalised = ImageF32::from_vec(
        normalized.width(),
        normalized.height(),
        normalized
            .data()
            .iter()
            .zip(spring.data())
            .map(|(a, b)| a - b)
            .collect(),
    );
    let mm = min_max_loc_f32(&penalised);
    let (max_val, (mx, my)) = (mm.max_val, mm.max_loc);
    let step = TrackStep {
        x: mx as i32 + off_x + size + 3,
        y: my as i32 + off_y + size + 3,
        off_x,
        off_y,
        max_penalty,
        max_loc: (mx, my),
        max_val,
    };
    Some((
        step,
        TrackSurfaces {
            roi,
            correlation,
            normalized,
            penalty,
            spring,
            penalised,
        },
    ))
}

/// The penalty surface: `exp(d^2 / (2 sigma^2))` with `d` the distance from
/// `(cols / 2, rows / 2)` and `sigma = 1.5 cols`, in `f64`, stored as `f32`, then stretched
/// to `0 .. 1` with `normalize(NORM_MINMAX)`. (The C++ keeps one per size in a cache; the
/// values are the same.)
///
/// Allocates the surface asked for (`rows * cols` values, twice); the tracker asks for the
/// result's size, at most the search region, which the window bound limits (ledger A167).
///
/// Ported from processing/MarkerTracking.cpp, getNormalizedPenaltySurface.
pub fn penalty_surface(rows: usize, cols: usize) -> ImageF32 {
    let mut m = ImageF32::new(cols, rows);
    let halfcol = 0.5 * cols as f64;
    let halfrow = 0.5 * rows as f64;
    let sigma = halfcol * 3.0;
    let inv_2sigma_sq = 1.0 / (2.0 * sigma * sigma);
    for i in 0..rows {
        let di = halfrow - i as f64;
        let di_sq = di * di;
        for (j, v) in m.row_mut(i).iter_mut().enumerate() {
            let dj = halfcol - j as f64;
            *v = ((di_sq + dj * dj) * inv_2sigma_sq).exp() as f32;
        }
    }
    normalize_minmax_f32(&m, 0.0, 1.0)
}

/// [`penalty_surface`] centred on the search centre, `((cols - 1) / 2, (rows - 1) / 2)`
/// (index 30 of the 61-wide surface): [`penalty_surface_at`] for a prediction at a whole
/// pixel.
///
/// Allocates the surface asked for (`rows * cols` values, twice); the tracker asks for the
/// result's size, at most the search region, which the window bound limits (ledger A167).
///
/// Ported from processing/MarkerTracking.cpp, getNormalizedPenaltySurface, with the centre
/// corrected.
pub fn penalty_surface_centred(rows: usize, cols: usize) -> ImageF32 {
    penalty_surface_at(
        rows,
        cols,
        0.5 * (cols as f64 - 1.0),
        0.5 * (rows as f64 - 1.0),
    )
}

/// [`penalty_surface`] centred on the surface position `(centre_col, centre_row)`, which
/// may be fractional: the predicted position in surface indices, so that the penalty is
/// symmetric about the prediction itself (parity ledger A64, the fixed behaviour). Same
/// `sigma` (`1.5 cols`), same stretch to `0 .. 1`.
///
/// Allocates the surface asked for (`rows * cols` values, twice); the tracker asks for the
/// result's size, at most the search region, which the window bound limits (ledger A167).
///
/// Ported from processing/MarkerTracking.cpp, getNormalizedPenaltySurface, with the centre
/// corrected.
pub fn penalty_surface_at(rows: usize, cols: usize, centre_col: f64, centre_row: f64) -> ImageF32 {
    let mut m = ImageF32::new(cols, rows);
    let sigma = 0.5 * cols as f64 * 3.0;
    let inv_2sigma_sq = 1.0 / (2.0 * sigma * sigma);
    for i in 0..rows {
        let di = centre_row - i as f64;
        let di_sq = di * di;
        for (j, v) in m.row_mut(i).iter_mut().enumerate() {
            let dj = centre_col - j as f64;
            *v = ((di_sq + dj * dj) * inv_2sigma_sq).exp() as f32;
        }
    }
    normalize_minmax_f32(&m, 0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_image(c: &xma_oracle::Case, key: &str) -> ImageF32 {
        let noipp = format!("{key}.noipp");
        let key = if c.outputs().contains(&noipp.as_str()) {
            noipp.as_str()
        } else {
            key
        };
        let im = c.image(key);
        ImageF32::from_slice(im.cols, im.rows, im.f32())
    }

    #[test]
    fn integer_arithmetic_of_the_cpp() {
        assert_eq!(template_size(2.4), 5);
        assert_eq!(template_size(5.4999), 5);
        assert_eq!(template_size(5.5), 6);
        assert_eq!(template_size(9.6), 10);
        assert_eq!(template_size(-3.0), 5);
        assert_eq!(Prediction::Override.value(), 1);
    }

    #[test]
    fn normalize_matches_opencv() {
        // The matchTemplate cases store normalize(result, 0, 50, NORM_MINMAX).
        let all = xma_oracle::cases("matchtemplate");
        assert!(all.len() >= 5);
        for c in &all {
            if !c.outputs().contains(&"normalized") {
                continue;
            }
            let res = f32_image(c, "result");
            let want = f32_image(c, "normalized");
            let got = normalize_minmax_f32(&res, 0.0, 50.0);
            assert_eq!(got, want, "{}", c.name);
        }
    }

    #[test]
    fn max_loc_first_in_row_order() {
        for c in xma_oracle::cases("minmaxloc") {
            let src = f32_image(&c, "src");
            let m = min_max_loc_f32(&src);
            let (v, (x, y)) = (m.max_val, m.max_loc);
            assert_eq!(f64::from(v), c.scalar("minmaxloc", "max_val"), "{}", c.name);
            assert_eq!(x as f64, c.scalar("minmaxloc", "max_x"), "{}", c.name);
            assert_eq!(y as f64, c.scalar("minmaxloc", "max_y"), "{}", c.name);
        }
    }
}
