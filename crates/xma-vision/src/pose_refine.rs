//! Refinement of a rigid-body pose on 2D reprojection error.
//!
//! Work package: rigid-body poses. Ported from `processing/RigidBodyPoseOptimization.cpp`,
//! the generated expressions in `processing/optimizationFuncs/`
//! (`projFuncWithDistortionPoseSeperateDimensions.h`,
//! `jacFuncWithDistortionPoseSeperateDimensions.h`) and
//! `core/RigidBody.cpp:481-541` (`addDummyPointsForOptimization`), solved with
//! [`crate::levmar::dlevmar_der`].
//!
//! # What the C++ does, and this port with it
//!
//! - Observations ([`refinement_observations`]), for each camera `c` in order: every body
//!   marker with a 2D point in `c` (any camera, calibrated or not, any positive status,
//!   whether or not it has a 3D point; ledger A29), target the undistorted 2D point, model
//!   point the marker's reference point; then **all** virtual points projected into **every
//!   calibrated** camera ([`virtual_point_observations`]). So each virtual point is added
//!   (number of cameras) x (number of calibrated cameras) times (ledger A2). The C++ adds
//!   one whose world position is unknown at the origin, with its reference position at the
//!   origin (ledger A3); by default it is left out here ([`RigidCompat`]).
//! - The parameters are the inverse of the pose (body to world), `R' = R⁻¹` by LU inverse,
//!   `t' = −R⁻¹ t` ([`invert_pose`]).
//! - The residual of an observation is observed minus modelled pixel position, with the
//!   camera's k1, k2, p1, p2, k3 applied to the model point although the target is fully
//!   undistorted (ledger A4; by default the coefficients are zero here, which is the plain
//!   pinhole projection, [`RigidCompat`]), and without skew or k4 to k6 (ledger A30)
//!   ([`projection_residual`], [`projection_jacobian`]: the generated expressions term by
//!   term, in the same order).
//! - `dlevmar_der` with a zero measurement vector, `n = 2 x observations`, `itmax = 10000`
//!   and [`Options::XMALAB_POSE`]; the return value and `info` are ignored and `p` is used
//!   whatever the stop reason. With fewer than 3 observations levmar refuses (`n < m`)
//!   without touching `p`, so the result is the pose inverted twice (ledger A28).
//!
//! # Floating point
//!
//! The generated expressions are evaluated as C evaluates them (left to right, no fused
//! multiply-add), with `sqrt`, `sin`, `cos` and `pow` from the platform's math library, as
//! the C++ calls them.

#![allow(
    non_snake_case,
    clippy::too_many_lines,
    reason = "the generated expressions keep the C++ names and length so they can be compared line by line"
)]

use crate::camera::{RigidTransform, matrix_to_rodrigues, rodrigues_to_matrix};
use crate::cmath::{cos, pow, sin, sqrt};
use crate::levmar::{Options, dlevmar_der};
use crate::rigid::{
    PoseCamera, RigidBodyFrame, RigidCompat, VirtualPointFrame, camera_projection, gemm_3x3_3x1,
    gemm_3x4_4x1, inv3,
};
use crate::{Matrix3, Point2, Point3, Vector3};

/// `m_iterations` of `RigidBodyPoseOptimization`: levmar's `itmax`.
pub const ITERATIONS: usize = 10000;

/// One observation of the refinement: a target pixel position in a camera, and the point in
/// body coordinates that should project there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RefineObservation {
    /// Index of the camera (`cameraIdx`).
    pub camera: usize,
    /// Target position (`Pts2D`): the undistorted 2D point, or a virtual point's projection.
    pub image: Point2,
    /// Model point in body coordinates (`Pts3D`).
    pub model: Point3,
}

/// What the refinement reads of a camera: the reference calibration image's rotation and
/// translation vectors, the camera matrix and the first five distortion coefficients.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RefineCamera {
    /// Rotation vector, world to camera (`cameraRotationVector`).
    pub rotation: Vector3,
    /// Translation vector, world to camera (`cameraTranslationVector`).
    pub translation: Vector3,
    /// Camera matrix; only `fx`, `fy`, `cx`, `cy` are read.
    pub camera_matrix: Matrix3,
    /// k1, k2, p1, p2, k3 (`distortion[c].at<double>(0..5, 0)`).
    pub distortion: [f64; 5],
    /// The camera matrix's skew `(0, 1)` when the projection includes it (ledger A30
    /// fixed); 0 as in the C++, which ignores it.
    pub skew: f64,
}

impl RefineCamera {
    /// The values `RigidBodyPoseOptimization`'s constructor stores for a camera
    /// (`processing/RigidBodyPoseOptimization.cpp:179-182`). A camera without model
    /// distortion holds zero coefficients (`core/Camera.cpp:67`, `:373`). Without
    /// [`RigidCompat::lens_distortion`] every camera gets zero coefficients (ledger A4);
    /// without [`RigidCompat::refinement_without_skew`] the skew is kept (ledger A30).
    #[must_use]
    pub fn from_pose_camera(cam: &PoseCamera, options: RigidCompat) -> Self {
        let d = if options.lens_distortion {
            cam.model.intrinsics.distortion.unwrap_or([0.0; 8])
        } else {
            [0.0; 8]
        };
        RefineCamera {
            rotation: cam.pose.rotation,
            translation: cam.pose.translation,
            camera_matrix: cam.model.intrinsics.camera_matrix,
            distortion: [d[0], d[1], d[2], d[3], d[4]],
            skew: if options.refinement_without_skew {
                0.0
            } else {
                cam.model.intrinsics.camera_matrix[(0, 1)]
            },
        }
    }
}

/// Appends the observations of every virtual point in every calibrated camera and returns
/// how many were added.
///
/// Ported from `core/RigidBody.cpp:481-541`, `RigidBody::addDummyPointsForOptimization`:
/// the world position and the reference position pass through `cv::Point3f` (only with
/// [`RigidCompat::single_precision`], ledger A23); a point with
/// no world position keeps both at (0, 0, 0) and is still added with
/// [`RigidCompat::virtual_points_at_origin`] (ledger A3), and is left out without it; the
/// target is
/// `P (X, 1)` divided by its third component, with `P` the camera's projection matrix and
/// no distortion; a zero third component skips that camera. Tier A.
pub fn virtual_point_observations(
    virtual_points: &[VirtualPointFrame],
    cameras: &[PoseCamera],
    options: RigidCompat,
    out: &mut Vec<RefineObservation>,
) -> usize {
    virtual_points_into(virtual_points, cameras, None, options, out)
}

/// [`virtual_point_observations`] into every calibrated camera, or only into `camera`.
fn virtual_points_into(
    virtual_points: &[VirtualPointFrame],
    cameras: &[PoseCamera],
    camera: Option<usize>,
    options: RigidCompat,
    out: &mut Vec<RefineObservation>,
) -> usize {
    let mut count = 0;
    for v in virtual_points {
        let (src, dst) = match v.world {
            Some(w) => (options.round(w), options.round(v.reference)),
            None if options.virtual_points_at_origin => (Point3::origin(), Point3::origin()),
            None => continue,
        };
        for (c, cam) in cameras.iter().enumerate() {
            if !cam.calibrated || camera.is_some_and(|only| only != c) {
                continue;
            }
            let p = camera_projection(cam);
            let pt_out = gemm_3x4_4x1(&p, [src.x, src.y, src.z, 1.0]);
            let z = pt_out[2];
            if z != 0.0 {
                out.push(RefineObservation {
                    camera: c,
                    image: Point2::new(pt_out[0] / z, pt_out[1] / z),
                    model: dst,
                });
                count += 1;
            }
        }
    }
    count
}

/// The observations of the refinement, in the C++ order.
///
/// Ported from `processing/RigidBodyPoseOptimization.cpp:162-185`, the constructor of
/// `RigidBodyPoseOptimization`: for each camera, the body markers with a 2D point in it
/// (undistorted point, reference point `getReferencePoints()[p]` in double), then
/// [`virtual_point_observations`]. Tier A.
///
/// With `options` as XMALab 3: every camera's markers (ledger A29), and every virtual
/// point in every calibrated camera after each camera (ledger A2). Fixed: only calibrated
/// cameras, each with its markers and then each virtual point once.
#[must_use]
pub fn refinement_observations(
    frame: &RigidBodyFrame,
    cameras: &[PoseCamera],
    options: RigidCompat,
) -> Vec<RefineObservation> {
    let mut obs = Vec::new();
    for c in 0..cameras.len() {
        if !options.uncalibrated_cameras_in_refinement && !cameras[c].calibrated {
            continue;
        }
        for (p, marker) in frame.markers.iter().enumerate() {
            if let Some(view) = marker.views.get(c).copied().flatten() {
                obs.push(RefineObservation {
                    camera: c,
                    image: view.undistorted,
                    model: frame.reference_points[p],
                });
            }
        }
        let only = (!options.virtual_points_per_camera_pair).then_some(c);
        virtual_points_into(&frame.virtual_points, cameras, only, options, &mut obs);
    }
    obs
}

/// Inverts a pose: `R' = Rodrigues(Rodrigues(r).inv())`, `t' = −Rodrigues(r).inv() · t`.
///
/// Ported from `processing/RigidBodyPoseOptimization.cpp:187-191` (before the solve) and
/// `:237-241` (`setOutput`, after it): `Mat::inv()` is the LU inverse (zeros for a singular
/// matrix), the unary minus becomes `alpha = -1` of the `gemm`. Tier A.
#[must_use]
pub fn invert_pose(pose: &RigidTransform) -> RigidTransform {
    let rot = rodrigues_to_matrix(&pose.rotation);
    let rot_inv = inv3(&rot);
    let rotation = matrix_to_rodrigues(&rot_inv);
    let t = [
        pose.translation[0],
        pose.translation[1],
        pose.translation[2],
    ];
    let trans = gemm_3x3_3x1(&rot_inv, t, -1.0, None, 0.0);
    RigidTransform {
        rotation,
        translation: Vector3::new(trans[0], trans[1], trans[2]),
    }
}

/// The two residuals of one observation for the parameters `params` (Rodrigues vector and
/// translation of the body-to-world pose): observed minus modelled position, in pixels.
///
/// Ported from `processing/RigidBodyPoseOptimization.cpp:119-159`,
/// `RigidBodyPoseOptimization::projError`, with
/// `processing/optimizationFuncs/projFuncWithDistortionPoseSeperateDimensions.h` translated
/// term by term. Tier A.
#[must_use]
pub fn projection_residual(
    params: &[f64],
    obs: &RefineObservation,
    cam: &RefineCamera,
) -> [f64; 2] {
    let RxCam = cam.rotation[0];
    let RyCam = cam.rotation[1];
    let RzCam = cam.rotation[2];
    let txCam = cam.translation[0];
    let tyCam = cam.translation[1];
    let tzCam = cam.translation[2];
    let fx = cam.camera_matrix[(0, 0)];
    let fy = cam.camera_matrix[(1, 1)];
    let cx = cam.camera_matrix[(0, 2)];
    let cy = cam.camera_matrix[(1, 2)];
    let [k1, k2, p1, p2, k3] = cam.distortion;

    let RxBody = params[0];
    let RyBody = params[1];
    let RzBody = params[2];
    let txBody = params[3];
    let tyBody = params[4];
    let tzBody = params[5];

    let X = obs.model.x;
    let Y = obs.model.y;
    let Z = obs.model.z;

    let u = obs.image.x;
    let v = obs.image.y;

    let mut a0 = [0.0f64; 2];

    // --- projFuncWithDistortionPoseSeperateDimensions.h ---
    let t2 = RxCam * RxCam;
    let t3 = RyCam * RyCam;
    let t4 = RzCam * RzCam;
    let t5 = t2 + t3 + t4;
    let t6 = sqrt(t5);
    let t7 = RxBody * RxBody;
    let t8 = RyBody * RyBody;
    let t9 = RzBody * RzBody;
    let t10 = t7 + t8 + t9;
    let t11 = sqrt(t10);
    let t12 = sin(t11);
    let t13 = 1.0 / sqrt(t10);
    let t14 = cos(t11);
    let t15 = t14 - 1.0;
    let t16 = 1.0 / t10;
    let t17 = sin(t6);
    let t18 = 1.0 / sqrt(t5);
    let t19 = cos(t6);
    let t20 = t19 - 1.0;
    let t21 = 1.0 / t5;
    let t22 = RxBody * t12 * t13;
    let t23 = RyBody * RzBody * t15 * t16;
    let t24 = RzBody * t12 * t13;
    let t25 = RyBody * t12 * t13;
    let t26 = RxBody * RzBody * t15 * t16;
    let t28 = RyCam * t17 * t18;
    let t29 = RxCam * RzCam * t20 * t21;
    let t30 = RxBody * RyBody * t15 * t16;
    let t31 = t24 + t30;
    let t32 = Y * t31;
    let t33 = t25 - t26;
    let t34 = Z * t33;
    let t35 = t8 + t9;
    let t36 = t15 * t16 * t35;
    let t37 = t36 + 1.0;
    let t38 = X * t37;
    let t39 = -t32 + t34 + t38 + txBody;
    let t40 = t22 + t23;
    let t41 = Z * t40;
    let t42 = t7 + t9;
    let t43 = t15 * t16 * t42;
    let t44 = t43 + 1.0;
    let t45 = Y * t44;
    let t46 = t25 + t26;
    let t47 = X * t46;
    let t48 = t22 - t23;
    let t49 = Y * t48;
    let t50 = t7 + t8;
    let t51 = t15 * t16 * t50;
    let t52 = t51 + 1.0;
    let t53 = Z * t52;
    let t54 = -t47 + t49 + t53 + tzBody;
    let t56 = RzCam * t17 * t18;
    let t57 = RxCam * RyCam * t20 * t21;
    let t68 = t56 + t57;
    let t69 = t28 - t29;
    let t70 = t54 * t69;
    let t71 = t3 + t4;
    let t72 = t20 * t21 * t71;
    let t73 = t72 + 1.0;
    let t74 = t39 * t73;
    let t27 =
        t70 + t74 + txCam - t68 * (-t41 + t45 + tyBody + X * (t24 - RxBody * RyBody * t15 * t16));
    let t55 = RxCam * t17 * t18;
    let t58 = t24 - t30;
    let t59 = X * t58;
    let t60 = -t41 + t45 + t59 + tyBody;
    let t63 = RyCam * RzCam * t20 * t21;
    let t80 = t55 + t63;
    let t81 = t54 * t80;
    let t82 = t56 - t57;
    let t83 = t39 * t82;
    let t84 = t2 + t4;
    let t85 = t20 * t21 * t84;
    let t86 = t85 + 1.0;
    let t87 = t60 * t86;
    let t61 = -t81 + t83 + t87 + tyCam;
    let t62 = t28 + t29;
    let t64 = t2 + t3;
    let t65 = t20 * t21 * t64;
    let t66 = t65 + 1.0;
    let t67 = t54 * t66;
    let t91 = t60 * t68;
    let t75 = t70 + t74 - t91 + txCam;
    let t76 = t55 - t63;
    let t77 = t60 * t76;
    let t89 = t39 * t62;
    let t78 = t67 + t77 - t89 + tzCam;
    let t79 = 1.0 / (t78 * t78);
    let t88 = t61 * t61;
    let t90 = t79 * t88;
    let t92 = t75 * t75;
    let t93 = t79 * t92;
    let t94 = t90 + t93;
    let t95 = k3 * t94;
    let t96 = k2 + t95;
    let t97 = t94 * t96;
    let t98 = k1 + t97;
    let t99 = t94 * t98;
    let t100 = t99 + 1.0;
    let t101 = 1.0 / t78;
    a0[0] = -cx + u
        - fx * (p2 * (t90 + (t27 * t27) * t79 * 3.0)
            + t75 * t100 * t101
            + p1 * t61 * t75 * t79 * 2.0);
    let yd = p1 * (t93 + t79 * t88 * 3.0) + t61 * t100 * t101 + p2 * t61 * t75 * t79 * 2.0;
    a0[1] = -cy + v - fy * yd;
    // --- end ---

    // Ledger A30 fixed: the projection's skew term, `u = fx x + s y + cx`.
    if cam.skew != 0.0 {
        a0[0] -= cam.skew * yd;
    }

    a0
}

/// The 2 x 6 Jacobian of [`projection_residual`] with respect to the parameters.
///
/// Ported from `processing/RigidBodyPoseOptimization.cpp:69-117`,
/// `RigidBodyPoseOptimization::projErrorJac`, with
/// `processing/optimizationFuncs/jacFuncWithDistortionPoseSeperateDimensions.h` translated
/// term by term. Tier A.
#[must_use]
pub fn projection_jacobian(
    params: &[f64],
    obs: &RefineObservation,
    cam: &RefineCamera,
) -> [[f64; 6]; 2] {
    let RxCam = cam.rotation[0];
    let RyCam = cam.rotation[1];
    let RzCam = cam.rotation[2];
    let txCam = cam.translation[0];
    let tyCam = cam.translation[1];
    let tzCam = cam.translation[2];
    let fx = cam.camera_matrix[(0, 0)];
    let fy = cam.camera_matrix[(1, 1)];
    let cx = cam.camera_matrix[(0, 2)];
    let cy = cam.camera_matrix[(1, 2)];
    let [k1, k2, p1, p2, k3] = cam.distortion;

    let RxBody = params[0];
    let RyBody = params[1];
    let RzBody = params[2];
    let txBody = params[3];
    let tyBody = params[4];
    let tzBody = params[5];

    let X = obs.model.x;
    let Y = obs.model.y;
    let Z = obs.model.z;

    let u = obs.image.x;
    let v = obs.image.y;
    let _ = (cx, cy, u, v);

    let mut a0 = [[0.0f64; 6]; 2];

    // --- jacFuncWithDistortionPoseSeperateDimensions.h ---
    let t2 = RxCam * RxCam;
    let t3 = RzCam * RzCam;
    let t4 = RyCam * RyCam;
    let t5 = t2 + t3 + t4;
    let t6 = RxBody * RxBody;
    let t7 = RyBody * RyBody;
    let t8 = RzBody * RzBody;
    let t9 = t6 + t7 + t8;
    let t10 = sqrt(t9);
    let t11 = sin(t10);
    let t12 = 1.0 / pow(t9, 3.0 / 2.0);
    let t13 = cos(t10);
    let t14 = t13 - 1.0;
    let t15 = 1.0 / t9;
    let t16 = t6 + t8;
    let t17 = 1.0 / (t9 * t9);
    let t18 = sqrt(t5);
    let t19 = cos(t18);
    let t20 = t19 - 1.0;
    let t21 = 1.0 / t5;
    let t22 = t7 + t8;
    let t23 = RxBody * RzBody * t13 * t15;
    let t24 = RyBody * t6 * t11 * t12;
    let t25 = RyBody * t6 * t14 * t17 * 2.0;
    let t26 = sin(t18);
    let t27 = 1.0 / sqrt(t5);
    let t28 = 1.0 / sqrt(t9);
    let t29 = t6 * t11 * t12;
    let t30 = RxBody * RyBody * RzBody * t11 * t12;
    let t31 = RxBody * RyBody * RzBody * t14 * t17 * 2.0;
    let t32 = t6 + t7;
    let t33 = RxBody * RyBody * t13 * t15;
    let t34 = RzBody * t6 * t11 * t12;
    let t35 = RzBody * t6 * t14 * t17 * 2.0;
    let t36 = RxCam * t26 * t27;
    let t37 = RyCam * RzCam * t20 * t21;
    let t38 = t36 + t37;
    let t39 = RzCam * t26 * t27;
    let t87 = RxCam * RyCam * t20 * t21;
    let t40 = t39 - t87;
    let t41 = RyBody * t11 * t28;
    let t42 = RxBody * RzBody * t14 * t15;
    let t43 = t2 + t3;
    let t44 = t20 * t21 * t43;
    let t45 = t44 + 1.0;
    let t46 = RzBody * t11 * t28;
    let t47 = RxBody * RyBody * t14 * t15;
    let t48 = RxBody * t11 * t28;
    let t49 = t46 + t47;
    let t50 = t41 - t42;
    let t51 = Z * t50;
    let t52 = t14 * t15 * t22;
    let t53 = t52 + 1.0;
    let t54 = X * t53;
    let t101 = Y * t49;
    let t55 = t51 + t54 - t101 + txBody;
    let t56 = t46 - t47;
    let t57 = X * t56;
    let t58 = RyBody * RzBody * t14 * t15;
    let t59 = t48 + t58;
    let t60 = t14 * t15 * t16;
    let t61 = t60 + 1.0;
    let t62 = Y * t61;
    let t93 = Z * t59;
    let t63 = t57 + t62 - t93 + tyBody;
    let t64 = t41 + t42;
    let t65 = t14 * t15 * t32;
    let t66 = t65 + 1.0;
    let t67 = Z * t66;
    let t68 = RxBody * t11 * t12 * t22;
    let t69 = RxBody * t14 * t17 * t22 * 2.0;
    let t70 = t68 + t69;
    let t71 = RxBody * RzBody * t11 * t12;
    let t91 = RyBody * t14 * t15;
    let t72 = -t23 + t24 + t25 + t71 - t91;
    let t73 = Y * t72;
    let t74 = RxBody * RyBody * t11 * t12;
    let t75 = RyCam * t26 * t27;
    let t76 = RxCam * RzCam * t20 * t21;
    let t77 = t11 * t28;
    let t78 = t6 * t13 * t15;
    let t79 = -t29 + t30 + t31 + t77 + t78;
    let t80 = Y * t79;
    let t81 = RxBody * t11 * t12 * t32;
    let t82 = RxBody * t14 * t17 * t32 * 2.0;
    let t88 = RxBody * t14 * t15 * 2.0;
    let t83 = t81 + t82 - t88;
    let t113 = RzBody * t14 * t15;
    let t84 = -t33 + t34 + t35 + t74 - t113;
    let t85 = X * t84;
    let t112 = Z * t83;
    let t86 = t80 + t85 - t112;
    let t89 = RxBody * t11 * t12 * t16;
    let t90 = RxBody * t14 * t16 * t17 * 2.0;
    let t92 = t39 + t87;
    let t94 = t75 - t76;
    let t95 = t48 - t58;
    let t96 = Y * t95;
    let t108 = X * t64;
    let t97 = t67 + t96 - t108 + tzBody;
    let t98 = t3 + t4;
    let t99 = t20 * t21 * t98;
    let t100 = t99 + 1.0;
    let t102 = t75 + t76;
    let t103 = t36 - t37;
    let t104 = t63 * t103;
    let t105 = t2 + t4;
    let t106 = t20 * t21 * t105;
    let t107 = t106 + 1.0;
    let t109 = t97 * t107;
    let t123 = t55 * t102;
    let t110 = t104 + t109 - t123 + tzCam;
    let t111 = 1.0 / (t110 * t110);
    let t114 = t29 + t30 + t31 - t77 - t78;
    let t115 = Z * t114;
    let t116 = -t88 + t89 + t90;
    let t117 = t23 + t24 + t25 - t71 - t91;
    let t118 = X * t117;
    let t128 = Y * t116;
    let t119 = t115 + t118 - t128;
    let t120 = t94 * t97;
    let t121 = t55 * t100;
    let t135 = t63 * t92;
    let t122 = t120 + t121 - t135 + txCam;
    let t124 = t86 * t107;
    let t125 = t33 + t34 + t35 - t74 - t113;
    let t126 = Z * t125;
    let t142 = X * t70;
    let t127 = t73 + t126 - t142;
    let t129 = t103 * t119;
    let t148 = t102 * t127;
    let t130 = t124 + t129 - t148;
    let t131 = t40 * t55;
    let t132 = t45 * t63;
    let t137 = t38 * t97;
    let t133 = t131 + t132 - t137 + tyCam;
    let t134 = 1.0 / (t110 * t110 * t110);
    let t136 = t122 * t122;
    let t138 = t133 * t133;
    let t139 = t111 * t136;
    let t140 = t111 * t138;
    let t141 = t139 + t140;
    let t143 = t86 * t94;
    let t144 = k3 * t141;
    let t145 = k2 + t144;
    let t146 = t100 * t127;
    let t153 = t92 * t119;
    let t147 = t143 + t146 - t153;
    let t149 = t45 * t119;
    let t150 = t40 * t127;
    let t158 = t38 * t86;
    let t151 = t149 + t150 - t158;
    let t152 = t111 * t133 * t151 * 2.0;
    let t154 = t111 * t122 * t147 * 2.0;
    let t159 = t130 * t134 * t136 * 2.0;
    let t160 = t130 * t134 * t138 * 2.0;
    let t155 = t152 + t154 - t159 - t160;
    let t156 = t141 * t145;
    let t157 = k1 + t156;
    let t161 = 1.0 / t110;
    let t162 = t141 * t157;
    let t163 = t162 + 1.0;
    let t164 = RzBody * t7 * t11 * t12;
    let t165 = RzBody * t7 * t14 * t17 * 2.0;
    let t166 = t7 * t11 * t12;
    let t167 = RyBody * RzBody * t13 * t15;
    let t168 = RxBody * t7 * t11 * t12;
    let t169 = RxBody * t7 * t14 * t17 * 2.0;
    let t170 = t7 * t13 * t15;
    let t171 = t30 + t31 + t77 - t166 + t170;
    let t172 = Z * t171;
    let t173 = RyBody * t11 * t12 * t22;
    let t174 = RyBody * t14 * t17 * t22 * 2.0;
    let t186 = RyBody * t14 * t15 * 2.0;
    let t175 = t173 + t174 - t186;
    let t176 = RyBody * RzBody * t11 * t12;
    let t183 = RxBody * t14 * t15;
    let t177 = -t167 + t168 + t169 + t176 - t183;
    let t178 = Y * t177;
    let t198 = X * t175;
    let t179 = t172 + t178 - t198;
    let t180 = RyBody * t11 * t12 * t16;
    let t181 = RyBody * t14 * t16 * t17 * 2.0;
    let t182 = t180 + t181;
    let t184 = -t33 + t74 - t113 + t164 + t165;
    let t185 = Z * t184;
    let t187 = RyBody * t11 * t12 * t32;
    let t188 = RyBody * t14 * t17 * t32 * 2.0;
    let t189 = t33 - t74 - t113 + t164 + t165;
    let t190 = Y * t189;
    let t191 = t30 + t31 - t77 + t166 - t170;
    let t192 = X * t191;
    let t193 = -t186 + t187 + t188;
    let t199 = Z * t193;
    let t194 = t190 + t192 - t199;
    let t195 = t167 + t168 + t169 - t176 - t183;
    let t196 = X * t195;
    let t201 = Y * t182;
    let t197 = t185 + t196 - t201;
    let t200 = t107 * t194;
    let t202 = t103 * t197;
    let t209 = t102 * t179;
    let t203 = t200 + t202 - t209;
    let t204 = t100 * t179;
    let t205 = t94 * t194;
    let t208 = t92 * t197;
    let t206 = t204 + t205 - t208;
    let t207 = t40 * t179;
    let t210 = t45 * t197;
    let t215 = t38 * t194;
    let t211 = t207 + t210 - t215;
    let t212 = t111 * t133 * t211 * 2.0;
    let t213 = t111 * t122 * t206 * 2.0;
    let t216 = t134 * t136 * t203 * 2.0;
    let t217 = t134 * t138 * t203 * 2.0;
    let t214 = t212 + t213 - t216 - t217;
    let t218 = RyBody * t8 * t11 * t12;
    let t219 = RyBody * t8 * t14 * t17 * 2.0;
    let t220 = t8 * t13 * t15;
    let t221 = RxBody * t8 * t11 * t12;
    let t222 = RxBody * t8 * t14 * t17 * 2.0;
    let t223 = t8 * t11 * t12;
    let t224 = t30 + t31 - t77 - t220 + t223;
    let t225 = Y * t224;
    let t226 = RzBody * t11 * t12 * t22;
    let t227 = RzBody * t14 * t17 * t22 * 2.0;
    let t240 = RzBody * t14 * t15 * 2.0;
    let t228 = t226 + t227 - t240;
    let t229 = t167 - t176 - t183 + t221 + t222;
    let t230 = Z * t229;
    let t250 = X * t228;
    let t231 = t225 + t230 - t250;
    let t232 = RzBody * t11 * t12 * t32;
    let t233 = RzBody * t14 * t17 * t32 * 2.0;
    let t234 = t232 + t233;
    let t235 = -t167 + t176 - t183 + t221 + t222;
    let t236 = X * t235;
    let t237 = t23 - t71 - t91 + t218 + t219;
    let t238 = Y * t237;
    let t245 = Z * t234;
    let t239 = t236 + t238 - t245;
    let t241 = RzBody * t11 * t12 * t16;
    let t242 = RzBody * t14 * t16 * t17 * 2.0;
    let t243 = -t23 + t71 - t91 + t218 + t219;
    let t244 = Z * t243;
    let t246 = t30 + t31 + t77 + t220 - t223;
    let t247 = X * t246;
    let t248 = -t240 + t241 + t242;
    let t252 = Y * t248;
    let t249 = t244 + t247 - t252;
    let t251 = t107 * t239;
    let t253 = t103 * t249;
    let t260 = t102 * t231;
    let t254 = t251 + t253 - t260;
    let t255 = t100 * t231;
    let t256 = t94 * t239;
    let t259 = t92 * t249;
    let t257 = t255 + t256 - t259;
    let t258 = t40 * t231;
    let t261 = t45 * t249;
    let t266 = t38 * t239;
    let t262 = t258 + t261 - t266;
    let t263 = t111 * t133 * t262 * 2.0;
    let t264 = t111 * t122 * t257 * 2.0;
    let t267 = t134 * t136 * t254 * 2.0;
    let t268 = t134 * t138 * t254 * 2.0;
    let t265 = t263 + t264 - t267 - t268;
    let t269 = t40 * t111 * t133 * 2.0;
    let t270 = t102 * t134 * t138 * 2.0;
    let t271 = t100 * t111 * t122 * 2.0;
    let t272 = t102 * t134 * t136 * 2.0;
    let t273 = t269 + t270 + t271 + t272;
    let t274 = t103 * t134 * t138 * 2.0;
    let t275 = t92 * t111 * t122 * 2.0;
    let t276 = t103 * t134 * t136 * 2.0;
    let t278 = t45 * t111 * t133 * 2.0;
    let t277 = t274 + t275 + t276 - t278;
    let t279 = t107 * t134 * t138 * 2.0;
    let t280 = t38 * t111 * t133 * 2.0;
    let t281 = t107 * t134 * t136 * 2.0;
    let t283 = t94 * t111 * t122 * 2.0;
    let t282 = t279 + t280 + t281 - t283;
    let t284 = t145 * t155;
    let t285 = k3 * t141 * t155;
    let t286 = t284 + t285;
    let t287 = t141 * t286;
    let t288 = t155 * t157;
    let t289 = t287 + t288;
    let t290 = t145 * t214;
    let t291 = k3 * t141 * t214;
    let t292 = t290 + t291;
    let t293 = t141 * t292;
    let t294 = t157 * t214;
    let t295 = t293 + t294;
    let t296 = t145 * t265;
    let t297 = k3 * t141 * t265;
    let t298 = t296 + t297;
    let t299 = t141 * t298;
    let t300 = t157 * t265;
    let t301 = t299 + t300;
    let t302 = t145 * t273;
    let t303 = k3 * t141 * t273;
    let t304 = t302 + t303;
    let t305 = t141 * t304;
    let t306 = t157 * t273;
    let t307 = t305 + t306;
    let t308 = t145 * t277;
    let t309 = k3 * t141 * t277;
    let t310 = t308 + t309;
    let t311 = t141 * t310;
    let t312 = t157 * t277;
    let t313 = t311 + t312;
    let t314 = t145 * t282;
    let t315 = k3 * t141 * t282;
    let t316 = t314 + t315;
    let t317 = t141 * t316;
    let t318 = t157 * t282;
    let t319 = t317 + t318;
    a0[0][0] = -fx
        * (p2 * (t152 - t160 + t111 * t122 * t147 * 6.0 - t130 * t134 * t136 * 6.0)
            + t147 * t161 * t163
            + t122 * t161 * t289
            + p1 * t111 * t122 * t151 * 2.0
            + p1 * t111 * t133 * t147 * 2.0
            - t111 * t122 * t130 * t163
            - p1 * t122 * t130 * t133 * t134 * 4.0);
    a0[0][1] = -fx
        * (p2 * (t212 - t217 + t111 * t122 * t206 * 6.0 - t134 * t136 * t203 * 6.0)
            + t161 * t163 * t206
            + t122 * t161 * t295
            + p1 * t111 * t122 * t211 * 2.0
            + p1 * t111 * t133 * t206 * 2.0
            - t111 * t122 * t163 * t203
            - p1 * t122 * t133 * t134 * t203 * 4.0);
    a0[0][2] = -fx
        * (p2 * (t263 - t268 + t111 * t122 * t257 * 6.0 - t134 * t136 * t254 * 6.0)
            + t161 * t163 * t257
            + t122 * t161 * t301
            + p1 * t111 * t122 * t262 * 2.0
            + p1 * t111 * t133 * t257 * 2.0
            - t111 * t122 * t163 * t254
            - p1 * t122 * t133 * t134 * t254 * 4.0);
    a0[0][3] = -fx
        * (p2 * (t269 + t270 + t100 * t111 * t122 * 6.0 + t102 * t134 * t136 * 6.0)
            + t100 * t161 * t163
            + t122 * t161 * t307
            + p1 * t40 * t111 * t122 * 2.0
            + p1 * t100 * t111 * t133 * 2.0
            + t102 * t111 * t122 * t163
            + p1 * t102 * t122 * t133 * t134 * 4.0);
    a0[0][4] = fx
        * (p2
            * (t274 - t45 * t111 * t133 * 2.0
                + t92 * t111 * t122 * 6.0
                + t103 * t134 * t136 * 6.0)
            + t92 * t161 * t163
            + t122 * t161 * t313
            - p1 * t45 * t111 * t122 * 2.0
            + p1 * t92 * t111 * t133 * 2.0
            + t103 * t111 * t122 * t163
            + p1 * t103 * t122 * t133 * t134 * 4.0);
    a0[0][5] = fx
        * (p2 * (t279 + t280 - t94 * t111 * t122 * 6.0 + t107 * t134 * t136 * 6.0)
            - t94 * t161 * t163
            + t122 * t161 * t319
            + p1 * t38 * t111 * t122 * 2.0
            - p1 * t94 * t111 * t133 * 2.0
            + t107 * t111 * t122 * t163
            + p1 * t107 * t122 * t133 * t134 * 4.0);
    a0[1][0] = -fy
        * (p1 * (t154 - t159 + t111 * t133 * t151 * 6.0 - t130 * t134 * t138 * 6.0)
            + t151 * t161 * t163
            + t133 * t161 * t289
            + p2 * t111 * t122 * t151 * 2.0
            + p2 * t111 * t133 * t147 * 2.0
            - t111 * t130 * t133 * t163
            - p2 * t122 * t130 * t133 * t134 * 4.0);
    a0[1][1] = -fy
        * (p1 * (t213 - t216 + t111 * t133 * t211 * 6.0 - t134 * t138 * t203 * 6.0)
            + t161 * t163 * t211
            + t133 * t161 * t295
            + p2 * t111 * t122 * t211 * 2.0
            + p2 * t111 * t133 * t206 * 2.0
            - t111 * t133 * t163 * t203
            - p2 * t122 * t133 * t134 * t203 * 4.0);
    a0[1][2] = -fy
        * (p1 * (t264 - t267 + t111 * t133 * t262 * 6.0 - t134 * t138 * t254 * 6.0)
            + t161 * t163 * t262
            + t133 * t161 * t301
            + p2 * t111 * t122 * t262 * 2.0
            + p2 * t111 * t133 * t257 * 2.0
            - t111 * t133 * t163 * t254
            - p2 * t122 * t133 * t134 * t254 * 4.0);
    a0[1][3] = -fy
        * (p1 * (t271 + t272 + t40 * t111 * t133 * 6.0 + t102 * t134 * t138 * 6.0)
            + t40 * t161 * t163
            + t133 * t161 * t307
            + p2 * t40 * t111 * t122 * 2.0
            + p2 * t100 * t111 * t133 * 2.0
            + t102 * t111 * t133 * t163
            + p2 * t102 * t122 * t133 * t134 * 4.0);
    a0[1][4] = fy
        * (p1 * (t275 + t276 - t45 * t111 * t133 * 6.0 + t103 * t134 * t138 * 6.0)
            - t45 * t161 * t163
            + t133 * t161 * t313
            - p2 * t45 * t111 * t122 * 2.0
            + p2 * t92 * t111 * t133 * 2.0
            + t103 * t111 * t133 * t163
            + p2 * t103 * t122 * t133 * t134 * 4.0);
    a0[1][5] = fy
        * (p1 * (t281 - t283 + t38 * t111 * t133 * 6.0 + t107 * t134 * t138 * 6.0)
            + t38 * t161 * t163
            + t133 * t161 * t319
            + p2 * t38 * t111 * t122 * 2.0
            - p2 * t94 * t111 * t133 * 2.0
            + t107 * t111 * t133 * t163
            + p2 * t107 * t122 * t133 * t134 * 4.0);
    // --- end ---

    // Ledger A30 fixed: the residual's skew term `-s y_d` (see [`projection_residual`]); its
    // derivative is `s / fy` times the second row's, which is `-fy dy_d/dp`.
    if cam.skew != 0.0 {
        let ratio = cam.skew / fy;
        let row1 = a0[1];
        for (j0, j1) in a0[0].iter_mut().zip(row1) {
            *j0 += ratio * j1;
        }
    }

    a0
}

/// Refines a pose (world to body) on the observations' reprojection error and returns the
/// refined pose (world to body).
///
/// Ported from `processing/RigidBodyPoseOptimization.cpp:187-277` (the constructor's initial
/// guess, `optimizeRigidBodySetup` and `setOutput`): [`invert_pose`], `dlevmar_der` with
/// [`projection_residual`] and [`projection_jacobian`], a zero measurement vector of length
/// `2 x observations`, [`ITERATIONS`] and [`Options::XMALAB_POSE`], then [`invert_pose`] of
/// the result. With [`RigidCompat::refine_failure_ignored`] (XMALab 3) the solver's return
/// value and `info` are ignored; with fewer than 3 observations it refuses and the result
/// is the input inverted twice (ledger A28). Fixed: when the solver refuses or fails (an
/// error return: too few observations, a singular matrix, NaN or infinity in the residual)
/// or gives a non-finite result, `pose` is returned exactly as it was.
/// `cameras` is indexed by [`RefineObservation::camera`]. Tier A.
/// Oracle: `trial_<t>/rb_pose_optimized.csv`.
#[must_use]
pub fn refine_pose(
    pose: &RigidTransform,
    observations: &[RefineObservation],
    cameras: &[RefineCamera],
    compat: RigidCompat,
) -> RigidTransform {
    let start = invert_pose(pose);
    let mut p = [
        start.rotation[0],
        start.rotation[1],
        start.rotation[2],
        start.translation[0],
        start.translation[1],
        start.translation[2],
    ];
    let n = 2 * observations.len();
    let x = vec![0.0f64; n];
    let func = |p: &[f64], hx: &mut [f64]| {
        for (i, o) in observations.iter().enumerate() {
            let r = projection_residual(p, o, &cameras[o.camera]);
            hx[2 * i] = r[0];
            hx[2 * i + 1] = r[1];
        }
    };
    let jacf = |p: &[f64], jac: &mut [f64]| {
        for (i, o) in observations.iter().enumerate() {
            let j = projection_jacobian(p, o, &cameras[o.camera]);
            for (row, values) in j.iter().enumerate() {
                jac[i * 12 + row * 6..i * 12 + row * 6 + 6].copy_from_slice(values);
            }
        }
    };
    let result = dlevmar_der(
        func,
        jacf,
        &mut p,
        Some(&x),
        n,
        ITERATIONS,
        Some(&Options::XMALAB_POSE),
    );
    // XMALab 3 ignores the return value and info (ledger A28).
    let failed = match &result {
        Ok(info) => info.return_value() < 0,
        Err(_) => true,
    };
    if !compat.refine_failure_ignored && (failed || p.iter().any(|v| !v.is_finite())) {
        return *pose;
    }
    invert_pose(&RigidTransform {
        rotation: Vector3::new(p[0], p[1], p[2]),
        translation: Vector3::new(p[3], p[4], p[5]),
    })
}
