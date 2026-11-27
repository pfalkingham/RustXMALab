//! What the trial update does for one marker at one frame: triangulation, reprojection and
//! the 2D error, and the mean marker size computed on load.
//!
//! Ported from `core/Marker.cpp`. The per-frame functions are pure: they read the marker
//! and an immutable [`FrameContext`] and return a [`MarkerFramePatch`] that says exactly
//! which values the C++ writes at that frame and which it leaves untouched. The caller
//! applies the patches, so frames can later be computed in parallel without shared mutable
//! state.

use xma_model::{Marker, MarkerStatus, Point2, Point3};
use xma_vision::Matrix3x4;
use xma_vision::camera::{CameraModel, RigidTransform, projection_matrix};
use xma_vision::triangulation::{Observation, TriangulationMethod, triangulate};

/// One camera as the trial update sees it: its model and its pose for the trial's
/// reference calibration image.
#[derive(Clone, Debug)]
pub struct FrameCamera {
    pub model: CameraModel,
    /// The pose of calibration image `referenceCalibrationImage`, or `None` when the camera
    /// has no such image (the C++ indexes out of bounds there: undefined behaviour).
    pub pose: Option<RigidTransform>,
    /// `K [R | t]` for `pose` (`Camera::getProjectionMatrix`).
    pub projection: Option<Matrix3x4>,
    /// `Camera::isCalibrated()`. Read by triangulation when ledger A34 is fixed.
    pub calibrated: bool,
}

impl FrameCamera {
    /// Builds the camera for one trial.
    ///
    /// Ported from core/Camera.cpp, Camera::getProjectionMatrix (the calibration image chosen
    /// by `Trial::getReferenceCalibrationImage`). The camera counts as calibrated; set
    /// [`FrameCamera::calibrated`] otherwise.
    #[must_use]
    pub fn new(model: CameraModel, pose: Option<RigidTransform>) -> Self {
        let projection = pose
            .as_ref()
            .map(|p| projection_matrix(&model.intrinsics.camera_matrix, p));
        Self {
            model,
            pose,
            projection,
            calibrated: true,
        }
    }
}

/// Everything the per-frame marker step reads besides the marker itself. Immutable during
/// an update.
#[derive(Clone, Debug)]
pub struct FrameContext {
    /// One per camera, in project order.
    pub cameras: Vec<FrameCamera>,
    /// True for a `NO_CALIBRATION` project (`Project::getCalibration() == NO_CALIBRATION`).
    pub no_calibration: bool,
    /// The `TriangulationMethod` setting.
    pub method: TriangulationMethod,
    /// Ledger A34: triangulation uses uncalibrated cameras, and a degenerate result keeps
    /// the old point or writes infinities or NaN (XMALab 3). Fixed (false): calibrated
    /// cameras only, and a degenerate (`None` or non-finite) result leaves the point
    /// undefined.
    pub triangulation_unchecked: bool,
    /// Ledger A35: projections are written for an undefined 3D point (XMALab 3). Fixed
    /// (false): none is written.
    pub project_undefined_points: bool,
}

/// What one marker's update writes at one frame. `None` (or a `None` entry) means the C++
/// leaves the stored value untouched.
#[derive(Clone, Debug, PartialEq)]
pub struct MarkerFramePatch {
    pub frame: usize,
    /// `status3D[frame]`; `None` when the marker was only reprojected.
    pub status3d: Option<MarkerStatus>,
    /// `points3D[frame]`, written only by a successful triangulation.
    pub point3d: Option<Point3>,
    /// `points2D_projected[c][frame]` per camera, written by `reprojectPoint`.
    pub projected: Vec<Option<Point2>>,
    /// `error2D[c][frame]` per camera, written by `updateError`.
    pub error: Vec<Option<f64>>,
}

impl MarkerFramePatch {
    fn untouched(frame: usize, n_cameras: usize) -> Self {
        Self {
            frame,
            status3d: None,
            point3d: None,
            projected: vec![None; n_cameras],
            error: vec![None; n_cameras],
        }
    }

    /// Writes the patch into the marker.
    pub fn apply(&self, marker: &mut Marker) {
        let f = self.frame;
        if let Some(s) = self.status3d {
            marker.status3d[f] = s;
        }
        if let Some(p) = self.point3d {
            marker.points3d[f] = p;
        }
        for (c, view) in marker.views.iter_mut().enumerate() {
            if let Some(Some(p)) = self.projected.get(c) {
                view.projected[f] = *p;
            }
            if let Some(Some(e)) = self.error.get(c) {
                view.error[f] = *e;
            }
        }
    }
}

/// The per-frame marker step of the trial update: triangulate when the marker is flagged,
/// otherwise only reproject the stored 3D point (parity ledger, item F9).
///
/// Ported from processing/UpdateTrialFrame.cpp, UpdateTrialFrame::process (the marker loop,
/// `:51-60`), which calls `Marker::reconstruct3DPoint(frame, true)` or
/// `Marker::reprojectPoint(frame)`.
#[must_use]
pub fn update_marker_frame(ctx: &FrameContext, marker: &Marker, frame: usize) -> MarkerFramePatch {
    if marker.requires_recomputation {
        reconstruct_3d_point(ctx, marker, frame)
    } else {
        let mut patch = MarkerFramePatch::untouched(frame, marker.n_cameras());
        reproject_point(
            ctx,
            marker,
            marker.points3d[frame],
            marker.status3d[frame],
            &mut patch,
        );
        patch
    }
}

/// Triangulates one frame.
///
/// Ported from core/Marker.cpp, Marker::reconstruct3DPoint with `updateAll = true` (so the
/// rigid bodies are not touched), and the common part of the five methods it dispatches to
/// (`reconstruct3DPointZisserman` and the others, `:239-616`):
///
/// - `status3D[frame]` is set to `UNDEFINED` first, always;
/// - a `NO_CALIBRATION` project stops there;
/// - the cameras used are those with `status2D > 0`, in camera order, whether calibrated or
///   not in XMALab 3 (parity ledger, item A34; fixed: calibrated cameras only); each point
///   goes through `Camera::undistortPoint(p, true)` (when that has no result, ledger A19
///   fixed, the camera is not used);
/// - with fewer than two such cameras, or when methods 0 and 1 meet a zero homogeneous
///   coordinate, nothing else is written: `points3D`, the projections and the errors keep
///   their old values (the point is undefined). Methods 2 to 4 can give infinities or NaN,
///   and parallel rays give a finite but meaningless point, which XMALab 3 writes; fixed,
///   a result that is not finite, lies behind a camera used, or comes from rays closer
///   than [`MIN_RAY_ANGLE`] is treated as no result;
/// - on success `points3D` is written, `status3D` becomes the smallest participating 2D
///   status (starting from `MANUAL_AND_OPTIMIZED`), and `reprojectPoint` runs.
///
/// A camera with no pose at the reference calibration image (undefined behaviour in the C++)
/// is left out of the triangulation.
#[must_use]
pub fn reconstruct_3d_point(ctx: &FrameContext, marker: &Marker, frame: usize) -> MarkerFramePatch {
    let mut patch = MarkerFramePatch::untouched(frame, marker.n_cameras());
    patch.status3d = Some(MarkerStatus::Undefined);
    if ctx.no_calibration {
        return patch;
    }
    let mut status = MarkerStatus::ManualAndOptimized;
    let mut observations = Vec::new();
    let mut used = Vec::new();
    for (c, view) in marker.views.iter().enumerate() {
        let s = view.status[frame];
        if !s.has_data() {
            continue;
        }
        let Some(cam) = ctx.cameras.get(c) else {
            continue;
        };
        if !ctx.triangulation_unchecked && !cam.calibrated {
            continue;
        }
        let Some(projection) = cam.projection else {
            continue;
        };
        // Ledger A19 fixed: a point whose lens undistortion has no result is not used, as
        // if the camera did not see the marker at this frame.
        let Some(point) = cam
            .model
            .try_undistort_point(view.points[frame], true, true, true)
        else {
            continue;
        };
        if s < status {
            status = s;
        }
        observations.push(Observation { projection, point });
        used.push(c);
    }
    let result = triangulate(ctx.method, &observations)
        .filter(|p| ctx.triangulation_unchecked || well_conditioned(ctx, &used, &observations, *p));
    if let Some(p) = result {
        patch.point3d = Some(p);
        patch.status3d = Some(status);
        reproject_point(ctx, marker, p, status, &mut patch);
    }
    patch
}

/// The smallest angle between viewing rays that a fixed-mode triangulation accepts
/// (ledger A34): 1e-4 radian (about 0.0057 degree). Over the 58327 marker points seen by
/// two or more calibrated cameras in the reference datasets, the smallest largest angle
/// between their viewing rays is 1.763 radian (101 degrees, test
/// `a34_smallest_ray_angle_in_the_reference_data`), so real points are four orders of
/// magnitude above the threshold, while rays that are parallel up to rounding (about
/// 1e-12) are far below it.
pub const MIN_RAY_ANGLE: f64 = 1e-4;

/// The viewing ray of an undistorted image point in world coordinates:
/// `Rᵀ K⁻¹ (u, v, 1)`, normalised. `None` when the camera has no pose or `K` is singular.
fn ray_direction(cam: &FrameCamera, point: Point2) -> Option<xma_vision::Vector3> {
    let pose = cam.pose.as_ref()?;
    let k_inv = cam.model.intrinsics.camera_matrix.try_inverse()?;
    let r = xma_vision::camera::rodrigues_to_matrix(&pose.rotation);
    let d = r.transpose() * (k_inv * xma_vision::Vector3::new(point.x, point.y, 1.0));
    let n = d.norm();
    (n > 0.0 && n.is_finite()).then(|| d / n)
}

/// The largest angle, in radians, between the viewing rays of the given cameras' points
/// (`cameras[i]` sees `points[i]`); `None` with fewer than two usable rays.
#[must_use]
pub fn largest_ray_angle(cameras: &[&FrameCamera], points: &[Point2]) -> Option<f64> {
    let rays: Vec<_> = cameras
        .iter()
        .zip(points)
        .filter_map(|(c, p)| ray_direction(c, *p))
        .collect();
    let mut best: Option<f64> = None;
    for i in 0..rays.len() {
        for j in i + 1..rays.len() {
            let a = rays[i].cross(&rays[j]).norm().atan2(rays[i].dot(&rays[j]));
            best = Some(best.map_or(a, |b: f64| b.max(a)));
        }
    }
    best
}

/// Ledger A34 fixed: a triangulated point is accepted only when it is finite, lies in
/// front of every camera used (positive third component of `P (X, 1)`), and the viewing
/// rays are not parallel (largest angle between them at least [`MIN_RAY_ANGLE`]).
fn well_conditioned(
    ctx: &FrameContext,
    used: &[usize],
    observations: &[Observation],
    p: Point3,
) -> bool {
    if !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite()) {
        return false;
    }
    let in_front = observations.iter().all(|o| {
        let m = &o.projection;
        let w = m[(2, 0)] * p.x + m[(2, 1)] * p.y + m[(2, 2)] * p.z + m[(2, 3)];
        w > 0.0
    });
    if !in_front {
        return false;
    }
    let cams: Vec<&FrameCamera> = used.iter().map(|&c| &ctx.cameras[c]).collect();
    let points: Vec<Point2> = observations.iter().map(|o| o.point).collect();
    largest_ray_angle(&cams, &points).is_some_and(|a| a >= MIN_RAY_ANGLE)
}

/// Projects the 3D point into every camera and updates the 2D errors.
///
/// Ported from core/Marker.cpp, Marker::reprojectPoint (`:1573-1590`): for a
/// `NO_CALIBRATION` project the projection is the 2D point itself; otherwise
/// `Camera::projectPoint` of `points3D[frame]` for **every** camera, whatever the 3D status,
/// so an undefined point projects its stale or placeholder value (parity ledger, item A35;
/// fixed, without [`FrameContext::project_undefined_points`], no projection is written for
/// an undefined point and the stored ones are left as they are). Then [`update_error`]. `point3d` and `status3d` are the values the marker holds at this
/// frame once the patch is applied.
pub fn reproject_point(
    ctx: &FrameContext,
    marker: &Marker,
    point3d: Point3,
    status3d: MarkerStatus,
    patch: &mut MarkerFramePatch,
) {
    let frame = patch.frame;
    if !ctx.no_calibration && !ctx.project_undefined_points && !status3d.has_data() {
        update_error(marker, status3d, patch);
        return;
    }
    for (c, view) in marker.views.iter().enumerate() {
        patch.projected[c] = if ctx.no_calibration {
            Some(view.points[frame])
        } else {
            ctx.cameras
                .get(c)
                .and_then(|cam| cam.pose.as_ref().map(|pose| (cam, pose)))
                .map(|(cam, pose)| cam.model.project_point(pose, point3d))
        };
    }
    update_error(marker, status3d, patch);
}

/// Ported from core/Marker.cpp, Marker::updateError (`:1693-1706`): only when the 3D status
/// is above 0, and only for cameras whose 2D status is above 0, the error is the Euclidean
/// distance between the 2D point and its projection, as
/// `sqrt(dx * dx + dy * dy)`. Other errors keep their old value.
pub fn update_error(marker: &Marker, status3d: MarkerStatus, patch: &mut MarkerFramePatch) {
    if !status3d.has_data() {
        return;
    }
    let frame = patch.frame;
    for (c, view) in marker.views.iter().enumerate() {
        if view.status[frame].has_data() {
            let p = view.points[frame];
            let q = patch.projected[c].unwrap_or(view.projected[frame]);
            let dx = p.x - q.x;
            let dy = p.y - q.y;
            patch.error[c] = Some((dx * dx + dy * dy).sqrt());
        }
    }
}

/// Mean detected marker size and the size range.
///
/// Ported from core/Marker.cpp, Marker::updateMeanSize (`:1708-1736`). `meanSize` is the
/// mean of the counted sizes, and is left unchanged when there are none. Cameras in order,
/// frames in order, as the C++ loops.
///
/// The counted sizes and the size range (parity ledger, item A1):
///
/// - with `range_from_sum` (XMALab 3, [`xma_model::CppCompat::size_range_from_sum`]): the
///   sizes strictly between 1 and 50 are counted; the range is the largest `|size - sum|`
///   over **all** entries (unset ones included), where `sum` is the sum of the counted
///   sizes, not their mean, so `Marker::isValid`'s size test never rejects anything;
/// - fixed: the sizes in the closed interval [1, 50] are counted, the interval
///   `Marker::isValid` accepts (`size < 1 || size > 50` rejects, `core/Marker.cpp:1018`), so
///   a size it accepts is measured against a range it is part of; the range is the largest
///   `|size - mean|` over the counted sizes only, with `mean` their mean; 0 when there are
///   none.
pub fn update_mean_size(marker: &mut Marker, range_from_sum: bool) {
    let counted = |s: f64| {
        if range_from_sum {
            s > 1.0 && s < 50.0
        } else {
            (1.0..=50.0).contains(&s)
        }
    };
    let mut sum = 0.0_f64;
    let mut count = 0_i32;
    for view in &marker.views {
        for &s in &view.size {
            if counted(s) {
                sum += s;
                count += 1;
            }
        }
    }
    if count > 0 {
        marker.mean_size = sum / f64::from(count);
    }
    let mut range = 0.0_f64;
    if range_from_sum {
        for view in &marker.views {
            for &s in &view.size {
                let d = (s - sum).abs();
                if d > range {
                    range = d;
                }
            }
        }
    } else if count > 0 {
        let mean = sum / f64::from(count);
        for view in &marker.views {
            for &s in &view.size {
                if counted(s) {
                    range = range.max((s - mean).abs());
                }
            }
        }
    }
    marker.size_range = range;
}

#[cfg(test)]
mod tests {
    use super::*;
    use xma_model::{Matrix3, Vector3};
    use xma_vision::camera::Intrinsics;

    fn camera(tx: f64) -> FrameCamera {
        let mut k = Matrix3::identity();
        k[(0, 0)] = 1000.0;
        k[(1, 1)] = 1000.0;
        k[(0, 2)] = 512.0;
        k[(1, 2)] = 512.0;
        FrameCamera::new(
            CameraModel {
                compat: xma_vision::camera::ModelCompat::XMALAB3,
                intrinsics: Intrinsics {
                    camera_matrix: k,
                    distortion: None,
                },
                grid: None,
            },
            Some(RigidTransform {
                rotation: Vector3::new(0.0, 0.1 * tx, 0.0),
                translation: Vector3::new(tx, 0.0, 50.0),
            }),
        )
    }

    fn context() -> FrameContext {
        FrameContext {
            cameras: vec![camera(-5.0), camera(5.0)],
            no_calibration: false,
            method: TriangulationMethod::ZissermanIncremental,
            triangulation_unchecked: true,
            project_undefined_points: true,
        }
    }

    #[test]
    fn one_view_writes_only_the_status() {
        let ctx = context();
        let mut m = Marker::new(2, 1, 8);
        m.points3d[0] = Point3::new(1.0, 2.0, 3.0);
        m.status3d[0] = MarkerStatus::Tracked;
        m.views[0].points[0] = Point2::new(500.0, 500.0);
        m.views[0].status[0] = MarkerStatus::Set;
        m.views[0].projected[0] = Point2::new(7.0, 8.0);
        m.views[0].error[0] = 0.5;
        let before = m.clone();
        let patch = reconstruct_3d_point(&ctx, &m, 0);
        assert_eq!(patch.status3d, Some(MarkerStatus::Undefined));
        assert_eq!(patch.point3d, None);
        assert!(patch.projected.iter().all(Option::is_none));
        assert!(patch.error.iter().all(Option::is_none));
        patch.apply(&mut m);
        assert_eq!(m.status3d[0], MarkerStatus::Undefined);
        assert_eq!(m.points3d[0], before.points3d[0]);
        assert_eq!(m.views, before.views);
    }

    #[test]
    fn two_views_take_the_lowest_status_and_reproject() {
        let ctx = context();
        let x = Point3::new(1.0, -2.0, 3.0);
        let mut m = Marker::new(2, 1, 8);
        for (c, s) in [(0, MarkerStatus::Manual), (1, MarkerStatus::Tracked)] {
            let cam = &ctx.cameras[c];
            m.views[c].points[0] = cam.model.project_point(cam.pose.as_ref().unwrap(), x);
            m.views[c].status[0] = s;
        }
        let patch = reconstruct_3d_point(&ctx, &m, 0);
        assert_eq!(patch.status3d, Some(MarkerStatus::Tracked));
        let p = patch.point3d.expect("triangulated");
        assert!((p - x).norm() < 1e-9);
        assert!(patch.projected.iter().all(Option::is_some));
        assert!(patch.error.iter().all(|e| e.is_some_and(|e| e < 1e-6)));
    }

    #[test]
    fn reprojection_writes_every_camera_but_errors_only_for_seen_points() {
        let ctx = context();
        let mut m = Marker::new(2, 1, 8);
        m.requires_recomputation = false;
        m.points3d[0] = Point3::new(0.0, 0.0, 0.0);
        m.status3d[0] = MarkerStatus::Set;
        m.views[1].points[0] = Point2::new(100.0, 100.0);
        m.views[1].status[0] = MarkerStatus::Set;
        let patch = update_marker_frame(&ctx, &m, 0);
        assert_eq!(patch.status3d, None);
        assert!(patch.projected.iter().all(Option::is_some));
        assert_eq!(patch.error[0], None);
        assert!(patch.error[1].is_some());

        // An undefined 3D point is still projected (ledger A35); errors are not touched.
        m.status3d[0] = MarkerStatus::Undefined;
        let patch = update_marker_frame(&ctx, &m, 0);
        assert!(patch.projected.iter().all(Option::is_some));
        assert!(patch.error.iter().all(Option::is_none));
    }

    /// Ledger A34 fixed: an uncalibrated camera is not used, so one calibrated view is not
    /// enough; XMALab 3 triangulates with it. A non-finite result is no result.
    #[test]
    fn a34_uncalibrated_cameras_are_not_triangulated() {
        let x = Point3::new(1.0, -2.0, 3.0);
        let mut ctx = context();
        ctx.cameras[1].calibrated = false;
        let mut m = Marker::new(2, 1, 8);
        for c in 0..2 {
            let cam = &ctx.cameras[c];
            m.views[c].points[0] = cam.model.project_point(cam.pose.as_ref().unwrap(), x);
            m.views[c].status[0] = MarkerStatus::Tracked;
        }
        let cpp = reconstruct_3d_point(&ctx, &m, 0);
        assert!(cpp.point3d.is_some());
        ctx.triangulation_unchecked = false;
        let fixed = reconstruct_3d_point(&ctx, &m, 0);
        assert_eq!(fixed.point3d, None);
        assert_eq!(fixed.status3d, Some(MarkerStatus::Undefined));
        // Two calibrated views at the same point: method 2 to 4 degenerate cases are not
        // reachable from consistent cameras here; a NaN 2D point gives a non-finite result.
        ctx.cameras[1].calibrated = true;
        m.views[0].points[0] = Point2::new(f64::NAN, 1.0);
        for method in [
            TriangulationMethod::ZissermanIncremental,
            TriangulationMethod::from_setting(2),
            TriangulationMethod::from_setting(4),
        ] {
            ctx.method = method;
            let p = reconstruct_3d_point(&ctx, &m, 0);
            assert_eq!(p.point3d, None, "{method:?}");
            assert_eq!(p.status3d, Some(MarkerStatus::Undefined));
        }
    }

    /// Ledger A35 fixed: no projection for an undefined point.
    #[test]
    fn a35_undefined_points_are_not_projected() {
        let mut ctx = context();
        ctx.project_undefined_points = false;
        let mut m = Marker::new(2, 1, 8);
        m.requires_recomputation = false;
        m.points3d[0] = Point3::new(0.0, 0.0, 0.0);
        m.status3d[0] = MarkerStatus::Undefined;
        m.views[1].points[0] = Point2::new(100.0, 100.0);
        m.views[1].status[0] = MarkerStatus::Set;
        let patch = update_marker_frame(&ctx, &m, 0);
        assert!(patch.projected.iter().all(Option::is_none));
        assert!(patch.error.iter().all(Option::is_none));
        m.status3d[0] = MarkerStatus::Set;
        let patch = update_marker_frame(&ctx, &m, 0);
        assert!(patch.projected.iter().all(Option::is_some));
    }

    #[test]
    fn no_calibration_copies_points_and_leaves_3d_undefined() {
        let mut ctx = context();
        ctx.no_calibration = true;
        let mut m = Marker::new(2, 1, 8);
        for c in 0..2 {
            m.views[c].points[0] = Point2::new(10.0 + c as f64, 20.0);
            m.views[c].status[0] = MarkerStatus::Set;
        }
        let patch = reconstruct_3d_point(&ctx, &m, 0);
        assert_eq!(patch.status3d, Some(MarkerStatus::Undefined));
        assert_eq!(patch.point3d, None);
        assert!(patch.projected.iter().all(Option::is_none));

        m.requires_recomputation = false;
        m.status3d[0] = MarkerStatus::Set;
        let patch = update_marker_frame(&ctx, &m, 0);
        assert_eq!(patch.projected[1], Some(Point2::new(11.0, 20.0)));
        assert_eq!(patch.error, vec![Some(0.0), Some(0.0)]);
    }

    #[test]
    fn mean_size_keeps_the_sum_bug() {
        let mut m = Marker::new(1, 4, 8);
        m.views[0].size = vec![2.0, 4.0, -1.0, 60.0];
        update_mean_size(&mut m, true);
        assert_eq!(m.mean_size, 3.0);
        // The range is measured from the sum (6), not the mean: |60 - 6| = 54.
        assert_eq!(m.size_range, 54.0);

        let mut empty = Marker::new(1, 2, 8);
        update_mean_size(&mut empty, true);
        assert_eq!(empty.mean_size, -1.0);
        assert_eq!(empty.size_range, 1.0);
    }

    /// Ledger A1 fixed: the range is measured from the mean, over the counted sizes only.
    #[test]
    fn a1_size_range_from_the_mean_of_counted_sizes() {
        let mut m = Marker::new(2, 3, 8);
        m.views[0].size = vec![2.0, 4.0, -1.0];
        m.views[1].size = vec![60.0, 6.0, 0.5];
        update_mean_size(&mut m, false);
        assert_eq!(m.mean_size, 4.0);
        assert_eq!(m.size_range, 2.0);
        // The same marker in XMALab 3: |-1 - 12| = 13 from the sum, unset entries included,
        // and 60 - 12 = 48 is the largest.
        update_mean_size(&mut m, true);
        assert_eq!(m.mean_size, 4.0);
        assert_eq!(m.size_range, 48.0);

        // The boundary sizes 1 and 50 are counted when fixed (the interval `isValid` accepts),
        // not in XMALab 3.
        let mut m = Marker::new(1, 4, 8);
        m.views[0].size = vec![5.0, 1.0, 50.0, 50.0001];
        update_mean_size(&mut m, false);
        assert_eq!(m.mean_size, 56.0 / 3.0);
        assert_eq!(m.size_range, 50.0 - 56.0 / 3.0);
        update_mean_size(&mut m, true);
        assert_eq!(m.mean_size, 5.0);

        // No counted size: the mean is kept, the range is 0.
        let mut empty = Marker::new(1, 2, 8);
        empty.mean_size = 3.5;
        update_mean_size(&mut empty, false);
        assert_eq!(empty.mean_size, 3.5);
        assert_eq!(empty.size_range, 0.0);
    }
}
