//! Epipolar lines: where a point seen by one camera may lie in another camera's image.
//!
//! Ported from `core/Marker.cpp`, `Marker::getEpipolarLine` (the polyline the camera views
//! draw in the marker-tracking workspace) and `Marker::getEpipolarLineUndistorted` (the
//! normalised line in the ideal image, used by the tracker's epipolar checks).
//!
//! # The construction
//!
//! With `P1`, `P2` the 3x4 projection matrices of the two cameras (for the trial's
//! reference calibration image) extended to 4x4 by the row `(0, 0, 0, 1)`, the C++ forms
//! `proj = P2 * P1⁻¹` and maps two homogeneous points of the origin's ray, `(u, v, 1, 1)`
//! and `(100u, 100v, 100, 1)`, where `(u, v)` is the origin point with every distortion
//! removed (`Camera::undistortPoint(pt, true)`). The two images `p1`, `p2` define the line
//! in the destination's ideal (undistorted) image.
//!
//! [`epipolar_line`] then returns what is drawn in the destination's raw image:
//!
//! - a destination without distortion: the straight segment across the image;
//! - otherwise the line sampled in the ideal image, each sample at least `precision`
//!   pixels from the last one kept (the `EpipolarLinePrecision` setting), mapped into the
//!   raw image with `Camera::undistortPoint(pt, false)`, plus one closing point.
//!
//! # Fixed C++ defects ([`EpipolarOptions::xmalab3`] restores them)
//!
//! 1. The sampled line is kept where `0 < y < width` (the C++ names the camera's width
//!    `height`); fixed: `0 < y < height`.
//! 2. The line is sampled as `y = m x + y0` in steps of 0.1 in `x`. A vertical line
//!    (`p2.x == p1.x`) gets `m = 0`, a horizontal line through the wrong place; a steep
//!    line is sampled every `0.1 * m` pixels along `y`, so it is coarse or empty. Fixed: a
//!    line steeper than 45° is sampled in steps of 0.1 in `y` instead (x as a function of
//!    y); lines of slope at most 1 are sampled exactly as in the C++.
//! 3. The C++ curves the line only when the destination has an undistortion-grid object
//!    (`Camera::hasUndistortion`). A camera with a lens model and no grid gets a straight
//!    line over a distorted image; a camera whose grid is loaded but not computed gets no
//!    line at all (every sample is "unchanged" by the grid and dropped). Fixed: the line is
//!    curved when the destination has a computed grid or a lens model, and the test that
//!    drops samples the grid leaves unchanged (the grid's "not covered" answer) applies
//!    only when a computed grid is present.

use crate::camera::CameraModel;
use crate::linalg::invert;
use crate::{Matrix3x4, Point2};
use nalgebra::{Matrix4, Vector4};

/// One camera as the epipolar construction reads it.
#[derive(Clone, Copy, Debug)]
pub struct EpipolarCamera<'a> {
    /// Camera matrix, lens model and fitted undistortion grid.
    pub model: &'a CameraModel,
    /// `Camera::getProjectionMatrix(trial->getReferenceCalibrationImage())`.
    pub projection: &'a Matrix3x4,
    /// Image width in pixels (`Camera::getWidth`).
    pub width: u32,
    /// Image height in pixels (`Camera::getHeight`).
    pub height: u32,
    /// `Camera::hasUndistortion()`: an undistortion-grid object exists, computed or not.
    /// Read only by [`EpipolarOptions::xmalab3`].
    pub has_undistortion_grid: bool,
}

/// How [`epipolar_line`] samples a curved line.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpipolarOptions {
    /// Minimum distance in pixels (in the ideal image) between kept samples: the
    /// `EpipolarLinePrecision` setting (default 5).
    pub precision: f64,
    /// Reproduce XMALab 3 exactly, with the defects listed in the module documentation.
    pub xmalab3: bool,
}

impl Default for EpipolarOptions {
    fn default() -> Self {
        Self {
            precision: 5.0,
            xmalab3: false,
        }
    }
}

/// `proj = [P2; 0 0 0 1] * [P1; 0 0 0 1]⁻¹`.
///
/// Ported from `core/Marker.cpp`, `Marker::getEpipolarLine` (`:1757-1765`). The inverse is
/// OpenCV's `Mat::inv()` (`DECOMP_LU`, [`invert`]). Returns `None` when the 4x4 matrix is
/// singular (OpenCV returns zeros there, and the C++ then draws a degenerate line).
fn origin_to_destination(p1: &Matrix3x4, p2: &Matrix3x4) -> Option<Matrix4<f64>> {
    let extend = |p: &Matrix3x4| {
        let mut m = Matrix4::zeros();
        for r in 0..3 {
            for c in 0..4 {
                m[(r, c)] = p[(r, c)];
            }
        }
        m[(3, 3)] = 1.0;
        m
    };
    let inv = invert(&extend(p1));
    if !inv.ok {
        return None;
    }
    let inv = Matrix4::from_fn(|r, c| inv.x[(r, c)]);
    Some(extend(p2) * inv)
}

/// The two points `p1`, `p2` of the epipolar line in the destination's ideal image.
fn line_points(
    origin: &EpipolarCamera<'_>,
    destination: &EpipolarCamera<'_>,
    point: Point2,
) -> Option<(Point2, Point2, f64, f64)> {
    let proj = origin_to_destination(origin.projection, destination.projection)?;
    // Ledger A19 fixed: a source point with no undistorted position has no line.
    let u = origin.model.try_undistort_point(point, true, true, true)?;
    let o1 = proj * Vector4::new(u.x, u.y, 1.0, 1.0);
    let o2 = proj * Vector4::new(100.0 * u.x, 100.0 * u.y, 100.0, 1.0);
    let p1 = Point2::new(o1[0] / o1[2], o1[1] / o1[2]);
    let p2 = Point2::new(o2[0] / o2[2], o2[1] / o2[2]);
    Some((p1, p2, o1[2], o2[2]))
}

/// The polyline drawn in `destination`'s raw image for `point` (raw image coordinates) seen
/// by `origin`. Empty when the line cannot be formed (a singular origin matrix, or a point
/// whose ray maps to infinity) or, for a curved line, when no sample falls in the image.
///
/// Ported from `core/Marker.cpp`, `Marker::getEpipolarLine(int, int, cv::Point2d)`, with the
/// fixes listed in the module documentation unless `options.xmalab3`. The caller checks
/// `NO_CALIBRATION` (the C++ returns an empty line then).
#[must_use]
pub fn epipolar_line(
    origin: &EpipolarCamera<'_>,
    destination: &EpipolarCamera<'_>,
    point: Point2,
    options: EpipolarOptions,
) -> Vec<Point2> {
    let Some((p1, p2, _, _)) = line_points(origin, destination, point) else {
        return Vec::new();
    };
    if options.xmalab3 {
        return line_xmalab3(destination, p1, p2, options.precision);
    }
    if !(p1.x.is_finite() && p1.y.is_finite() && p2.x.is_finite() && p2.y.is_finite()) {
        return Vec::new();
    }
    let dx = p2.x - p1.x;
    let dy = p2.y - p1.y;
    if dx == 0.0 && dy == 0.0 {
        return Vec::new();
    }
    let w = f64::from(destination.width);
    let h = f64::from(destination.height);
    let model = destination.model;
    let has_grid = model.grid.is_some();
    let curved = has_grid || model.intrinsics.distortion.is_some();
    // Steep lines are parametrised by y (fix 2).
    let steep = dy.abs() > dx.abs();
    // The line as `b = slope * a + intercept`, with (a, b) = (x, y), or (y, x) when steep.
    let (slope, intercept, a_max, b_max) = if steep {
        let s = dx / dy;
        (s, p2.x - p2.y * s, h, w)
    } else {
        let s = dy / dx;
        (s, p2.y - p2.x * s, w, h)
    };
    let at = |a: f64| {
        let b = slope * a + intercept;
        if steep {
            Point2::new(b, a)
        } else {
            Point2::new(a, b)
        }
    };
    if !curved {
        return vec![at(0.0), at(a_max)];
    }
    let inside = |a: f64| {
        let b = slope * a + intercept;
        b > 0.0 && b < b_max
    };
    let map = |pt: Point2| -> Option<Point2> {
        let out = model.undistort_point(pt, false, true, true);
        // The grid returns its input where it does not cover the point; a sample with no
        // finite position is skipped.
        (out.x.is_finite() && out.y.is_finite() && (!has_grid || (out.x != pt.x && out.y != pt.y)))
            .then_some(out)
    };
    let dist2 = options.precision * options.precision;
    let mut line = Vec::new();
    let mut prev = Point2::new(-10000.0, -10000.0);
    let mut a = 0.0;
    while a <= a_max {
        if inside(a) {
            let pt = at(a);
            if (prev.x - pt.x).powi(2) + (prev.y - pt.y).powi(2) > dist2
                && let Some(out) = map(pt)
            {
                line.push(out);
                prev = pt;
            }
        }
        a += 0.1;
    }
    let mut a = a_max;
    while a >= 0.0 {
        if inside(a)
            && let Some(out) = map(at(a))
        {
            line.push(out);
            break;
        }
        a -= 0.1;
    }
    line
}

/// The C++ sampling, defects included (module documentation).
fn line_xmalab3(
    destination: &EpipolarCamera<'_>,
    p1: Point2,
    p2: Point2,
    precision: f64,
) -> Vec<Point2> {
    let denom = p2.x - p1.x;
    let m = if denom != 0.0 {
        (p2.y - p1.y) / denom
    } else {
        0.0
    };
    let y0 = p2.y - p2.x * m;
    let width = f64::from(destination.width);
    let mut epiline = Vec::new();
    if !destination.has_undistortion_grid {
        epiline.push(Point2::new(0.0, y0));
        epiline.push(Point2::new(width, m * width + y0));
        return epiline;
    }
    // Defect 1: the camera's width, named `height` in the C++.
    let height = width;
    let model = destination.model;
    let dist = precision * precision;
    let mut prev = Point2::new(-10000.0, -10000.0);
    let mut p = 0.0;
    while p <= width {
        let pt = Point2::new(p, m * p + y0);
        if pt.y > 0.0
            && pt.y < height
            && ((prev.x - pt.x) * (prev.x - pt.x) + (prev.y - pt.y) * (prev.y - pt.y)) > dist
        {
            let t = model.undistort_point(pt, false, true, true);
            if t.x != pt.x && t.y != pt.y {
                epiline.push(t);
                prev = pt;
            }
        }
        p += 0.1;
    }
    let mut p = width;
    while p >= 0.0 {
        let pt = Point2::new(p, m * p + y0);
        if pt.y > 0.0 && pt.y < height {
            let t = model.undistort_point(pt, false, true, true);
            if t.x != pt.x && t.y != pt.y {
                epiline.push(t);
                break;
            }
        }
        p -= 0.1;
    }
    epiline
}

/// The epipolar line in the destination's ideal image as `(a, b, c)` with `a² + b² = 1`, so
/// that `a x + b y + c` is the signed distance of an undistorted point `(x, y)` from it.
/// `None` where the C++ returns false (a projected depth below 1e-12, or coincident points).
///
/// Ported from `core/Marker.cpp`, `Marker::getEpipolarLineUndistorted`. The caller checks
/// `NO_CALIBRATION`.
#[must_use]
pub fn epipolar_line_undistorted(
    origin: &EpipolarCamera<'_>,
    destination: &EpipolarCamera<'_>,
    point: Point2,
) -> Option<[f64; 3]> {
    let (p1, p2, z1, z2) = line_points(origin, destination, point)?;
    if z1.abs() < 1e-12 || z2.abs() < 1e-12 {
        return None;
    }
    let a = p1.y - p2.y;
    let b = p2.x - p1.x;
    let c = p1.x * p2.y - p2.x * p1.y;
    let n = (a * a + b * b).sqrt();
    if n < 1e-12 {
        return None;
    }
    Some([a / n, b / n, c / n])
}

/// Distance in the destination's ideal image between `destination_point` (raw) and the
/// epipolar line of `origin_point`; -1 when there is no line, or when `destination_point`
/// has no undistorted position (ledger A19, fixed mode).
///
/// Ported from `core/Marker.cpp`, `Marker::distanceToEpipolarLine`.
#[must_use]
pub fn distance_to_epipolar_line(
    origin: &EpipolarCamera<'_>,
    destination: &EpipolarCamera<'_>,
    origin_point: Point2,
    destination_point: Point2,
) -> f64 {
    let Some(l) = epipolar_line_undistorted(origin, destination, origin_point) else {
        return -1.0;
    };
    let Some(u) = destination
        .model
        .try_undistort_point(destination_point, true, true, true)
    else {
        return -1.0;
    };
    (l[0] * u.x + l[1] * u.y + l[2]).abs()
}

/// The point of the epipolar line of `origin_point` nearest to `destination_point` (raw
/// image coordinates): the foot of the perpendicular in the destination's ideal image,
/// mapped back into the raw image (`Camera::undistortPoint(foot, false)`).
/// `destination_point` unchanged when there is no line, or when `destination_point` has no
/// undistorted position (ledger A19, fixed mode).
///
/// Ported from `core/Marker.cpp`, `Marker::projectOntoEpipolarLine` (`:1973-1981`). The
/// caller checks `NO_CALIBRATION`.
#[must_use]
pub fn project_onto_epipolar_line(
    origin: &EpipolarCamera<'_>,
    destination: &EpipolarCamera<'_>,
    origin_point: Point2,
    destination_point: Point2,
) -> Point2 {
    let Some(l) = epipolar_line_undistorted(origin, destination, origin_point) else {
        return destination_point;
    };
    let Some(u) = destination
        .model
        .try_undistort_point(destination_point, true, true, true)
    else {
        return destination_point;
    };
    let d = l[0] * u.x + l[1] * u.y + l[2];
    let foot = Point2::new(u.x - d * l[0], u.y - d * l[1]);
    destination.model.undistort_point(foot, false, true, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{Intrinsics, RigidTransform, projection_matrix};
    use crate::{Matrix3, Point3, Vector3};

    fn model(distortion: Option<[f64; 8]>) -> CameraModel {
        CameraModel {
            intrinsics: Intrinsics {
                camera_matrix: Matrix3::new(1000.0, 0.0, 512.0, 0.0, 1000.0, 512.0, 0.0, 0.0, 1.0),
                distortion,
            },
            grid: None,
            compat: Default::default(),
        }
    }

    fn pose(ry: f64, tx: f64) -> RigidTransform {
        RigidTransform {
            rotation: Vector3::new(0.0, ry, 0.0),
            translation: Vector3::new(tx, 0.0, 1000.0),
        }
    }

    fn cam<'a>(m: &'a CameraModel, p: &'a Matrix3x4) -> EpipolarCamera<'a> {
        EpipolarCamera {
            model: m,
            projection: p,
            width: 1024,
            height: 1024,
            has_undistortion_grid: false,
        }
    }

    /// Distance of `q` from the infinite line through `a` and `b`.
    fn dist_to_line(a: Point2, b: Point2, q: Point2) -> f64 {
        let d = b - a;
        ((q - a).perp(&d)).abs() / d.norm()
    }

    #[test]
    fn straight_line_passes_through_the_projection_of_every_ray_point() {
        let (m1, m2) = (model(None), model(None));
        let (pose1, pose2) = (pose(0.0, 0.0), pose(0.6, -300.0));
        let (pr1, pr2) = (
            projection_matrix(&m1.intrinsics.camera_matrix, &pose1),
            projection_matrix(&m2.intrinsics.camera_matrix, &pose2),
        );
        let (c1, c2) = (cam(&m1, &pr1), cam(&m2, &pr2));
        for world in [
            Point3::new(10.0, 20.0, 5.0),
            Point3::new(-50.0, 33.0, -80.0),
        ] {
            let x1 = m1.project_point(&pose1, world);
            let x2 = m2.project_point(&pose2, world);
            let line = epipolar_line(&c1, &c2, x1, EpipolarOptions::default());
            assert_eq!(line.len(), 2);
            assert_eq!(line[0].x, 0.0);
            assert_eq!(line[1].x, 1024.0);
            let d = dist_to_line(line[0], line[1], x2);
            assert!(d < 1e-6, "distance {d}");
            // The normalised form agrees.
            let dd = distance_to_epipolar_line(&c1, &c2, x1, x2);
            assert!(dd < 1e-6, "distance {dd}");
            // A point off the line is at its true distance.
            let off = Point2::new(x2.x + 3.0, x2.y);
            let l = epipolar_line_undistorted(&c1, &c2, x1).unwrap();
            assert!((l[0] * l[0] + l[1] * l[1] - 1.0).abs() < 1e-12);
            let expect = (3.0 * l[0]).abs();
            assert!((distance_to_epipolar_line(&c1, &c2, x1, off) - expect).abs() < 1e-6);
        }
    }

    #[test]
    fn projection_onto_the_line_is_the_nearest_point_on_it() {
        let d = Some([-0.2, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        for distortion in [None, d] {
            let (m1, m2) = (model(distortion), model(distortion));
            let (pose1, pose2) = (pose(0.0, 0.0), pose(0.6, -300.0));
            let (pr1, pr2) = (
                projection_matrix(&m1.intrinsics.camera_matrix, &pose1),
                projection_matrix(&m2.intrinsics.camera_matrix, &pose2),
            );
            let (c1, c2) = (cam(&m1, &pr1), cam(&m2, &pr2));
            let world = Point3::new(10.0, 20.0, 5.0);
            let x1 = m1.project_point(&pose1, world);
            let x2 = m2.project_point(&pose2, world);
            // A point on the line stays where it is.
            let on = project_onto_epipolar_line(&c1, &c2, x1, x2);
            assert!((on - x2).norm() < 1e-6, "{on} {x2}");
            // A point off the line lands on it, and no farther (in the ideal image) than
            // the distance to the line.
            let off = Point2::new(x2.x + 4.0, x2.y - 2.5);
            let p = project_onto_epipolar_line(&c1, &c2, x1, off);
            assert!(distance_to_epipolar_line(&c1, &c2, x1, p) < 1e-6);
            let u_off = m2.undistort_point(off, true, true, true);
            let u_p = m2.undistort_point(p, true, true, true);
            let dist = distance_to_epipolar_line(&c1, &c2, x1, off);
            assert!(((u_off - u_p).norm() - dist).abs() < 1e-6);
        }
    }

    #[test]
    fn vertical_line_is_drawn_vertically() {
        // Camera 2 displaced along x only, same orientation: epipolar lines are horizontal.
        // Displaced along y: vertical, which the C++ draws as a horizontal line (m = 0).
        let (m1, m2) = (model(None), model(None));
        let pose1 = pose(0.0, 0.0);
        let pose2 = RigidTransform {
            rotation: Vector3::zeros(),
            translation: Vector3::new(0.0, -200.0, 1000.0),
        };
        let pr1 = projection_matrix(&m1.intrinsics.camera_matrix, &pose1);
        let pr2 = projection_matrix(&m2.intrinsics.camera_matrix, &pose2);
        let (c1, c2) = (cam(&m1, &pr1), cam(&m2, &pr2));
        let world = Point3::new(30.0, 40.0, 50.0);
        let x1 = m1.project_point(&pose1, world);
        let x2 = m2.project_point(&pose2, world);
        let line = epipolar_line(&c1, &c2, x1, EpipolarOptions::default());
        assert_eq!(line.len(), 2);
        assert!((line[0].x - line[1].x).abs() < 1e-6, "{line:?}");
        assert!(dist_to_line(line[0], line[1], x2) < 1e-6);
        let cpp = epipolar_line(
            &c1,
            &c2,
            x1,
            EpipolarOptions {
                xmalab3: true,
                ..EpipolarOptions::default()
            },
        );
        // The C++ result: from x = 0 to x = width at a constant y.
        assert_eq!(cpp[0].x, 0.0);
        assert_eq!(cpp[0].y, cpp[1].y);
    }

    #[test]
    fn curved_line_with_lens_model_follows_the_distortion() {
        let d = Some([-0.2, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let (m1, m2) = (model(d), model(d));
        let (pose1, pose2) = (pose(0.0, 0.0), pose(0.5, -250.0));
        let pr1 = projection_matrix(&m1.intrinsics.camera_matrix, &pose1);
        let pr2 = projection_matrix(&m2.intrinsics.camera_matrix, &pose2);
        let (c1, c2) = (cam(&m1, &pr1), cam(&m2, &pr2));
        let world = Point3::new(60.0, -90.0, 20.0);
        let x1 = m1.project_point(&pose1, world);
        let x2 = m2.project_point(&pose2, world);
        let line = epipolar_line(&c1, &c2, x1, EpipolarOptions::default());
        assert!(line.len() > 10, "{} points", line.len());
        // Samples in the ideal image are at least `precision` apart; mapped, they stay
        // close to that.
        // The raw projection x2 lies on the polyline (within the chord error).
        let best = line
            .windows(2)
            .map(|w| {
                let (a, b) = (w[0], w[1]);
                let t = ((x2 - a).dot(&(b - a)) / (b - a).norm_squared()).clamp(0.0, 1.0);
                (a + (b - a) * t - x2).norm()
            })
            .fold(f64::INFINITY, f64::min);
        assert!(best < 0.05, "distance to polyline {best}");
        // The C++ draws this camera's line straight (no grid object): fix 3.
        let cpp = epipolar_line(
            &c1,
            &c2,
            x1,
            EpipolarOptions {
                xmalab3: true,
                ..EpipolarOptions::default()
            },
        );
        assert_eq!(cpp.len(), 2);
    }

    #[test]
    fn singular_origin_gives_no_line() {
        let m = model(None);
        let zero = Matrix3x4::zeros();
        let p = projection_matrix(&m.intrinsics.camera_matrix, &pose(0.0, 0.0));
        let line = epipolar_line(
            &cam(&m, &zero),
            &cam(&m, &p),
            Point2::new(1.0, 2.0),
            EpipolarOptions::default(),
        );
        assert!(line.is_empty());
        assert!(
            epipolar_line_undistorted(&cam(&m, &zero), &cam(&m, &p), Point2::new(1.0, 2.0))
                .is_none()
        );
    }
}
