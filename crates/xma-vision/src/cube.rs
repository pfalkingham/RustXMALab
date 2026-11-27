//! Calibration of one camera from one image of the calibration cube: the camera matrix and
//! pose from the detected beads and three or four reference clicks
//! ([`compute_pose_and_cam`]), the pose alone for a calibrated camera ([`compute_pose`]), and
//! the pose handed over from another camera's image ([`set_pose`]).
//!
//! Work package "cube" (phase 4). Ported from `processing/CubeCalibration.cpp` (all of it but
//! the Qt threading and the `_threadFinished` handlers, whose data is the returned
//! [`CubeResult`]). Survey: `docs/cpp-calibration.md` sections 3.4 to 3.6. Reference values:
//! the hook tables of the calibration runs (`oracle/FORMAT.md`, "Calibration runs").
//!
//! # What the caller passes
//!
//! The C++ constructor (`CubeCalibration.cpp:51-93`) undistorts the four reference clicks
//! and every detected bead with `Camera::undistortPoint(p, true)` (grid, then the lens model
//! when it is on, refined). Here the caller does that and passes the undistorted points in
//! [`CubeInput`]: the clicks (with whether each was set) and the detected beads
//! (`CalibrationImage::getDetectedPointsAll`, in detection order), with the calibration
//! object's bead positions (`CalibrationObject::getFrameSpecifications`, scaled). All work
//! here is in undistorted coordinates, and every returned point is too.
//!
//! # The random draws
//!
//! The six-point and five-point searches draw from the C library's `rand()`
//! (`getRandomReferences`, `:292`, `:300`). They take a [`CRand`] by `&mut`: glibc's
//! generator, the same sequence on every platform. XMALab never seeds it (ledger A5): a
//! session starts from `srand(1)` and each calibration continues the sequence.
//!
//! # `cv::calibrateCamera`
//!
//! [`compute_pose_and_cam`] calls [`calib3d::calibrate_camera`]; [`compute_pose_and_cam_with`]
//! takes the call as a function argument ([`CalibrateCameraFn`]) so that the port can be
//! checked against the recorded calls of the reference runs.
//!
//! # Floating point
//!
//! Every function follows its source operation by operation, at the same width: the DLT,
//! the matching and the decomposition in `double`; `calibrateCamera`, `projectPoints` and
//! EPnP on single-precision points as the C++ passes them (survey CD27). Matrix products go
//! through the ports of OpenCV's `gemm` in [`crate::rigid`]. The DLT's null vector comes from
//! [`linalg::solve_z`] (OpenCV's Jacobi SVD); the reference build hands it to LAPACK for 13
//! or more points (ledger A37), so the refits from all matched beads agree with XMALab 3 to
//! rounding, not to the bit.
//!
//! # Faults of XMALab 3
//!
//! Fixed by default, kept behind [`CubeCompat`] (parity ledger A110 to A119). Undefined
//! behaviour and endless loops of the C++ are refused in both modes ([`CubeError`]).

// The port keeps the C++'s index loops so it can be checked line by line against it.
#![allow(clippy::needless_range_loop)]

use crate::calib3d::{self, CalibrateError, CalibrateFlags, CalibrateOutput};
use crate::camera::{self, RigidTransform};
use crate::crand::CRand;
use crate::linalg;
use crate::rigid::{gemm_3x4_4x1, gemm_small};
use crate::{Matrix3, Matrix3x4, Point2, Point3, Vector3};
use nalgebra::{DMatrix, Matrix4};

/// Iterations of the six-point search with four references (`CubeCalibration.cpp:160`).
pub const LOOPS_FOUR_REFERENCES: u32 = 100_000;
/// Iterations of the six-point search with three references (`:161`).
pub const LOOPS_THREE_REFERENCES: u32 = 1_000_000;
/// Iterations of the five-point pose search with three references (`:214`).
pub const LOOPS_POSE_THREE_REFERENCES: u32 = 100_000;
/// Iterations of the five-point pose search otherwise (`:215`).
pub const LOOPS_POSE: u32 = 1_000;
/// The threshold (pixels) of the six-point search, its DLT error and its matching
/// (`:160-161`).
pub const SEARCH_THRESHOLD: f64 = 10.0;

/// `cv::calibrateCamera` as [`calib3d::calibrate_camera`] declares it: object points, image
/// points, image size (width, height), camera matrix, distortion, flags.
pub type CalibrateCameraFn<'f> = dyn FnMut(
        &[Vec<nalgebra::Point3<f32>>],
        &[Vec<nalgebra::Point2<f32>>],
        (usize, usize),
        &Matrix3,
        &[f64],
        CalibrateFlags,
    ) -> Result<CalibrateOutput, CalibrateError>
    + 'f;

/// Which faults of XMALab 3's `CubeCalibration` to reproduce. Each field is `true` for the
/// C++ behaviour. [`CubeCompat::fixed`] (the default) fixes all; [`CubeCompat::xmalab3`]
/// is the C++.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CubeCompat {
    /// Ledger A110 (survey CD33): a reference counts as clicked when its **undistorted**
    /// position has `x > 0 && y > 0` (`CubeCalibration.cpp:154`, `:208`, `:275`), and the
    /// unset references (-1, -1) are undistorted too, so one can be used as a click, and a
    /// click undistorted to `x <= 0` or `y <= 0` is dropped. Fixed: a reference is used
    /// exactly when it was set ([`CubeReference::set`]).
    pub reference_from_undistorted_position: bool,
    /// Ledger A111 (survey CD15): `refineResults` marks the result calibrated (or the pose
    /// computed) after its loop whatever its last refit achieved (`:552-562`): a final
    /// `calibrateOpenCV` refused (9 inliers needed) or rejected by OpenCV leaves the DLT
    /// decomposition as the camera matrix with the pose of the previous refit; a final EPnP
    /// not run (fewer than 5 inliers) keeps the previous pose. Fixed: the result fails when
    /// the final refit does not run or is rejected.
    pub success_after_failed_refit: bool,
    /// Ledger A112 (survey CD28): with the setting `FixPrincipal` the principal point is
    /// fixed at `0.5 * (width + 1)`, `0.5 * (height + 1)` (`:384-385`), the image centre in
    /// 1-based pixel numbering. Fixed: `0.5 * (width - 1)`, `0.5 * (height - 1)`, the
    /// centre with pixel centres at 0-based integers.
    pub principal_point_one_based: bool,
    /// Ledger A156: `refineResults` does not match again after its final refit
    /// (`:552-560`), so the matches, inlier flags, inlier count and errors it returns are
    /// those of the camera and pose before that refit (only the projections and the errors
    /// of the old matches are recomputed). Fixed: after a final refit that succeeded, the
    /// projections of the returned camera and pose are matched again with the thresholds of
    /// the C++'s own last rematch in the loop (`setCorrespondances` at the identification
    /// threshold, then `selectCorrespondances` at the outlier threshold), and those matches,
    /// flags, count and errors are returned.
    pub inliers_of_previous_camera: bool,
}

impl CubeCompat {
    /// Every fault fixed: the default.
    pub const FIXED: CubeCompat = CubeCompat {
        reference_from_undistorted_position: false,
        success_after_failed_refit: false,
        principal_point_one_based: false,
        inliers_of_previous_camera: false,
    };

    /// Everything as XMALab 3 does it.
    pub const XMALAB3: CubeCompat = CubeCompat {
        reference_from_undistorted_position: true,
        success_after_failed_refit: true,
        principal_point_one_based: true,
        inliers_of_previous_camera: true,
    };

    /// [`CubeCompat::FIXED`].
    #[must_use]
    pub const fn fixed() -> Self {
        Self::FIXED
    }

    /// [`CubeCompat::XMALAB3`].
    #[must_use]
    pub const fn xmalab3() -> Self {
        Self::XMALAB3
    }
}

/// The settings `CubeCalibration` reads, and the camera's image size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CubeSettings {
    /// `IdentificationThresholdCalibration` (default 15): the matching radius in pixels of
    /// the refinement (`setCorrespondances`); half of it for the first matching and for the
    /// pose search.
    pub identification_threshold: i32,
    /// `OutlierThresholdForCalibration` (default 10): a match is an inlier when its distance
    /// is below this (`selectCorrespondances`).
    pub outlier_threshold: i32,
    /// `FixPrincipal` (default false): the principal point is fixed at the image centre.
    pub fix_principal: bool,
    /// `Camera::getWidth`.
    pub width: i32,
    /// `Camera::getHeight`.
    pub height: i32,
}

impl CubeSettings {
    /// XMALab's default settings for a camera of this size.
    #[must_use]
    pub const fn defaults(width: i32, height: i32) -> Self {
        CubeSettings {
            identification_threshold: 15,
            outlier_threshold: 10,
            fix_principal: false,
            width,
            height,
        }
    }
}

/// One of the four reference clicks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CubeReference {
    /// The calibration-object point (0-based) the click marks (`referencesID`).
    pub point_id: i32,
    /// The click, undistorted with `Camera::undistortPoint(p, true)`. For an unset
    /// reference XMALab passes (-1, -1) undistorted; only [`CubeCompat::reference_from_undistorted_position`]
    /// reads it then.
    pub position: Point2,
    /// The reference was clicked: its raw click has `x >= 0 && y >= 0`, the test of the
    /// "compute calibration" button (`ui/WizardCalibrationCubeFrame.cpp:607-610`).
    pub set: bool,
}

/// What a `CubeCalibration` is constructed from.
#[derive(Clone, Copy, Debug)]
pub struct CubeInput<'a> {
    /// The calibration object's beads (`coords3D`).
    pub object_points: &'a [Point3],
    /// Every detected bead of the image, undistorted (`alldetectedPoints`).
    pub detected: &'a [Point2],
    /// The four reference clicks, in slot order.
    pub references: [CubeReference; 4],
    pub settings: CubeSettings,
    pub compat: CubeCompat,
    /// `None`: XMALab's iteration counts. `Some(n)`: the search runs `n` iterations
    /// instead (for tests of the search's first iterations).
    pub search_loops: Option<u32>,
}

/// Which `CubeCalibration` entry point produced a result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CubeMode {
    /// `computePoseAndCam`: the camera was not calibrated.
    PoseAndCam,
    /// `computePose`: the camera is calibrated.
    Pose,
    /// `setPose`: the pose comes from another camera's image.
    SetPose,
}

/// What a `CubeCalibration` hands back to its `_threadFinished` handler.
#[derive(Clone, Debug, PartialEq)]
pub struct CubeResult {
    pub mode: CubeMode,
    /// `calibrated`: in [`CubeMode::PoseAndCam`] the camera was calibrated by this image.
    /// In the other modes the camera was calibrated before (always `true`).
    pub calibrated: bool,
    /// `poseComputed`.
    pub pose_computed: bool,
    /// `maxinlier`: the number of inliers of the last selection.
    pub max_inlier: i32,
    /// `cameramatrix`: the new camera matrix ([`CubeMode::PoseAndCam`]) or the camera's.
    pub camera_matrix: Matrix3,
    /// `rotationvector`, `translationvector`: world to camera.
    pub pose: RigidTransform,
    /// `detectedPoints`: per calibration-object point the detected bead matched to it, or
    /// (-1, -1). Empty after a failed refinement.
    pub detected: Vec<Point2>,
    /// `projectedPoints`: per calibration-object point its projection. Empty after a failed
    /// refinement.
    pub projected: Vec<Point2>,
    /// `Inlier`, per calibration-object point.
    pub inlier: Vec<bool>,
    /// `error`, per calibration-object point: the distance between the projected and the
    /// detected point (10000 when nothing was matched).
    pub error: Vec<f64>,
}

impl CubeResult {
    /// Whether the handler writes the result back: `calibrated` for
    /// [`CubeMode::PoseAndCam`] (`computePoseAndCam_threadFinished`, `:125`),
    /// `poseComputed` otherwise (`computePose_threadFinished`, `:182`).
    #[must_use]
    pub fn succeeded(&self) -> bool {
        match self.mode {
            CubeMode::PoseAndCam => self.calibrated,
            CubeMode::Pose | CubeMode::SetPose => self.pose_computed,
        }
    }
}

/// Inputs the C++ would read out of range, or with which its search would never end.
/// Refused in both modes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CubeError {
    /// A used reference names a calibration-object point that does not exist (the C++
    /// reads `coords3D` out of range).
    ReferenceOutOfRange { slot: usize, point_id: i32 },
    /// The random search cannot draw its set: fewer unused calibration-object points or
    /// detected beads than it needs (the C++ loops for ever, or divides by zero with no
    /// detected bead).
    CannotDraw {
        needed: usize,
        object_points_free: usize,
        detected: usize,
    },
}

impl std::fmt::Display for CubeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CubeError::ReferenceOutOfRange { slot, point_id } => write!(
                f,
                "reference {} names point {} of the calibration object, which has no such point",
                slot + 1,
                point_id + 1
            ),
            CubeError::CannotDraw {
                needed,
                object_points_free,
                detected,
            } => write!(
                f,
                "the search needs {needed} more points, but there are {object_points_free} \
                 unused calibration-object points and {detected} detected beads"
            ),
        }
    }
}

impl std::error::Error for CubeError {}

// ---------------------------------------------------------------------------------------
// Trace: the values the reference build's hooks record
// ---------------------------------------------------------------------------------------

/// A hypothesis of the six-point search that improved the inlier count
/// (`hooks/ransac_accepted.csv`).
#[derive(Clone, Debug, PartialEq)]
pub struct RansacAccepted {
    pub loop_index: u32,
    /// The six calibration-object point ids, references first.
    pub ids3d: Vec<i32>,
    /// The six detected-bead indices, -1 for a reference click.
    pub ids2d: Vec<i32>,
    /// Mean reprojection error of the six-point DLT.
    pub projection_error: f64,
    pub n_inlier: i32,
    pub best_before: i32,
    /// The six-point DLT projection.
    pub projection: Matrix3x4,
}

/// A round of `computeProjectionFromInliers` (`hooks/ransac_refine_rounds.csv`).
#[derive(Clone, Debug, PartialEq)]
pub struct RansacRefineRound {
    pub loop_index: u32,
    pub round: u32,
    pub n_matched: i32,
    pub max_inlier: i32,
    pub n_correspondences: usize,
    /// The projection refitted from the matches, normalised.
    pub projection: Matrix3x4,
    pub camera_matrix: Matrix3,
}

/// A `cv::calibrateCamera` call (`hooks/calibrate_camera*.csv`; the hook records
/// successful calls only).
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrateCall {
    pub object_points: Vec<nalgebra::Point3<f32>>,
    pub image_points: Vec<nalgebra::Point2<f32>>,
    pub flags: CalibrateFlags,
    pub camera_matrix_in: Matrix3,
    pub output: Result<CalibrateOutput, CalibrateError>,
}

/// `calibrateFromInliers` before `calibrateOpenCV` (`hooks/k_from_projection.csv`).
#[derive(Clone, Debug, PartialEq)]
pub struct KFromProjection {
    pub projection: Matrix3x4,
    pub camera_matrix: Matrix3,
}

/// A round of `refineResults` (`hooks/refine_rounds.csv`): round 0 the first matching,
/// rounds 1, 2, ... the loop, round -1 the final refit. -1 where the hook writes -1.
#[derive(Clone, Debug, PartialEq)]
pub struct RefineRound {
    pub with_camera: bool,
    pub round: i32,
    pub matched_before: i32,
    pub matched: i32,
    pub inliers: i32,
    pub camera_matrix: Matrix3,
    pub pose: RigidTransform,
}

/// A hypothesis of the five-point pose search that improved the inlier count
/// (`hooks/ransac_pose_accepted.csv`).
#[derive(Clone, Debug, PartialEq)]
pub struct RansacPoseAccepted {
    pub loop_index: u32,
    pub ids3d: Vec<i32>,
    pub ids2d: Vec<i32>,
    pub n_inlier: i32,
    pub best_before: i32,
    pub pose: RigidTransform,
}

/// After the pose search (`hooks/ransac_pose_result.csv`).
#[derive(Clone, Debug, PartialEq)]
pub struct RansacPoseResult {
    pub n_best: usize,
    pub max_inlier: i32,
    pub pose: RigidTransform,
}

/// `setPose_thread` after its first matching (`hooks/set_pose.csv`).
#[derive(Clone, Debug, PartialEq)]
pub struct SetPoseTrace {
    pub transformation: Matrix4<f64>,
    pub pose: RigidTransform,
    pub n_matched: i32,
}

/// The intermediate values of one calibration, in the order the C++ produces them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CubeTrace {
    pub ransac_accepted: Vec<RansacAccepted>,
    pub ransac_refine_rounds: Vec<RansacRefineRound>,
    pub calibrate_calls: Vec<CalibrateCall>,
    pub k_from_projection: Vec<KFromProjection>,
    pub refine_rounds: Vec<RefineRound>,
    pub ransac_pose_accepted: Vec<RansacPoseAccepted>,
    pub ransac_pose_result: Option<RansacPoseResult>,
    pub set_pose: Option<SetPoseTrace>,
}

// ---------------------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------------------

/// Which references are used: with [`CubeCompat::reference_from_undistorted_position`] those
/// whose undistorted position has `x > 0 && y > 0` (`CubeCalibration.cpp:154`), otherwise
/// those that were set.
#[must_use]
pub fn references_used(references: &[CubeReference; 4], compat: CubeCompat) -> [bool; 4] {
    let mut used = [false; 4];
    for (u, r) in used.iter_mut().zip(references) {
        *u = if compat.reference_from_undistorted_position {
            r.position.x > 0.0 && r.position.y > 0.0
        } else {
            r.set
        };
    }
    used
}

/// Iterations of the search each entry point runs for `count` used references
/// (`CubeCalibration.cpp:160-161`, `:214-215`): [`CubeMode::PoseAndCam`] 100,000 with four
/// or more, 1,000,000 with three, none with fewer; [`CubeMode::Pose`] 100,000 with three,
/// 1,000 otherwise; [`CubeMode::SetPose`] none.
#[must_use]
pub fn search_loops(mode: CubeMode, count: usize) -> u32 {
    match mode {
        CubeMode::PoseAndCam => match count {
            c if c > 3 => LOOPS_FOUR_REFERENCES,
            3 => LOOPS_THREE_REFERENCES,
            _ => 0,
        },
        CubeMode::Pose => {
            if count == 3 {
                LOOPS_POSE_THREE_REFERENCES
            } else {
                LOOPS_POSE
            }
        }
        CubeMode::SetPose => 0,
    }
}

/// Calibrates an uncalibrated camera from one cube image: the six-point search, the camera
/// matrix from its projection, `cv::calibrateCamera`, and the refinement.
///
/// Ported from `processing/CubeCalibration.cpp`, `CubeCalibration::computePoseAndCam_thread`
/// (`:149-166`) with the constructor (`:51-93`). Uses [`calib3d::calibrate_camera`].
///
/// # Errors
/// [`CubeError`] for inputs the C++ cannot handle (see there).
pub fn compute_pose_and_cam(
    input: &CubeInput<'_>,
    rng: &mut CRand,
    trace: Option<&mut CubeTrace>,
) -> Result<CubeResult, CubeError> {
    compute_pose_and_cam_with(input, rng, &mut calib3d::calibrate_camera, trace)
}

/// [`compute_pose_and_cam`] with the `cv::calibrateCamera` call supplied by the caller.
///
/// # Errors
/// As [`compute_pose_and_cam`].
pub fn compute_pose_and_cam_with(
    input: &CubeInput<'_>,
    rng: &mut CRand,
    calibrate: &mut CalibrateCameraFn<'_>,
    trace: Option<&mut CubeTrace>,
) -> Result<CubeResult, CubeError> {
    let mut cube = Cube::new(input, false, Some(calibrate), trace);
    let count = cube.used.iter().filter(|&&u| u).count();
    let loops = input
        .search_loops
        .unwrap_or_else(|| search_loops(CubeMode::PoseAndCam, count));
    // if (count > 3) setupCorrespondancesRansac(100000, 10.0);
    // else if (count == 3) setupCorrespondancesRansac(1000000, 10.0);
    if count >= 3 {
        cube.check(6, loops > 0)?;
        cube.setup_correspondances_ransac(loops, SEARCH_THRESHOLD, rng);
    }
    if cube.calibrate_opencv() {
        cube.refine_results(true);
    }
    Ok(cube.result(CubeMode::PoseAndCam))
}

/// The pose of a cube image in a calibrated camera: the five-point pose search, then the
/// refinement.
///
/// Ported from `processing/CubeCalibration.cpp`, `CubeCalibration::computePose` (`:168-178`,
/// `cameramatrix` = the camera's) and `computePose_thread` (`:203-218`).
///
/// # Errors
/// [`CubeError`] for inputs the C++ cannot handle (see there).
pub fn compute_pose(
    input: &CubeInput<'_>,
    camera_matrix: &Matrix3,
    rng: &mut CRand,
    trace: Option<&mut CubeTrace>,
) -> Result<CubeResult, CubeError> {
    let mut cube = Cube::new(input, true, None, trace);
    cube.camera_matrix = *camera_matrix;
    let count = cube.used.iter().filter(|&&u| u).count();
    let loops = input
        .search_loops
        .unwrap_or_else(|| search_loops(CubeMode::Pose, count));
    cube.check(5, loops > 0)?;
    cube.setup_correspondances_ransac_pose(loops, rng);
    cube.refine_results(false);
    Ok(cube.result(CubeMode::Pose))
}

/// The pose of a cube image in a calibrated camera from a transformation found through
/// another camera, then matched and refined.
///
/// Ported from `processing/CubeCalibration.cpp`, `CubeCalibration::setPose` (`:220-231`) and
/// `setPose_thread` (`:233-261`). `transformation` is camera to world, 4x4
/// (`CalibrationImage::getTransformationMatrix` composed as `calibrateOtherFrames` does,
/// `ui/WizardCalibrationCubeFrame.cpp:807-902`). No random draws. The references are not
/// read.
///
/// # Errors
/// None at present; the signature matches the other entry points.
pub fn set_pose(
    input: &CubeInput<'_>,
    camera_matrix: &Matrix3,
    transformation: &Matrix4<f64>,
    trace: Option<&mut CubeTrace>,
) -> Result<CubeResult, CubeError> {
    let mut cube = Cube::new(input, true, None, trace);
    cube.camera_matrix = *camera_matrix;
    cube.set_pose_thread(transformation);
    Ok(cube.result(CubeMode::SetPose))
}

/// Draws from `rng` exactly what the search of `mode` would draw for `input`, without
/// computing anything: for replaying the C library's sequence through an earlier
/// calibration of the same session (the sequence depends only on the references, the
/// number of beads and detections, and the iteration count).
///
/// # Errors
/// As the entry point of `mode`.
pub fn skip_search_draws(
    input: &CubeInput<'_>,
    mode: CubeMode,
    rng: &mut CRand,
) -> Result<(), CubeError> {
    let calibrated = mode != CubeMode::PoseAndCam;
    let cube = Cube::new(input, calibrated, None, None);
    let count = cube.used.iter().filter(|&&u| u).count();
    let loops = input
        .search_loops
        .unwrap_or_else(|| search_loops(mode, count));
    let nb = if mode == CubeMode::PoseAndCam { 6 } else { 5 };
    if mode == CubeMode::PoseAndCam && count < 3 {
        return Ok(());
    }
    cube.check(nb, loops > 0)?;
    for _ in 0..loops {
        let _ = cube.get_random_references(nb, rng);
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------
// The CubeCalibration object
// ---------------------------------------------------------------------------------------

/// The state of one `CubeCalibration`, with its members' names.
struct Cube<'a, 'c, 't> {
    coords3d: &'a [Point3],
    all_detected: &'a [Point2],
    /// `selectedPoints`, `selectedPointsID`.
    selected: [Point2; 4],
    selected_id: [i32; 4],
    /// The reference test (`selectedPoints[i].x > 0 && selectedPoints[i].y > 0`, or set).
    used: [bool; 4],
    settings: CubeSettings,
    compat: CubeCompat,

    maxinlier: i32,
    pose_computed: bool,
    calibrated: bool,
    projected: Vec<Point2>,
    detected: Vec<Point2>,
    error: Vec<f64>,
    inlier: Vec<bool>,
    projection: Matrix3x4,
    rvec: Vector3,
    tvec: Vector3,
    camera_matrix: Matrix3,

    calibrate: Option<&'c mut CalibrateCameraFn<'c>>,
    trace: Option<&'t mut CubeTrace>,
    /// The search loop and refine round of the trace's `ransac_refine_rounds`.
    trace_loop: u32,
    trace_round: u32,
}

/// `euclideanDist(p, q)`: `diff = p - q; sqrt(diff.x * diff.x + diff.y * diff.y)`.
///
/// Ported from `CubeCalibration.cpp:319-323`.
fn euclidean_dist(p: Point2, q: Point2) -> f64 {
    let dx = p.x - q.x;
    let dy = p.y - q.y;
    (dx * dx + dy * dy).sqrt()
}

/// `projection * (X, Y, Z, 1)` and the division by the third coordinate
/// (`CubeCalibration.cpp:858-863`, `:901-906`, `:976-981`).
fn project_with(projection: &Matrix3x4, p: Point3) -> Point2 {
    let q = gemm_3x4_4x1(projection, [p.x, p.y, p.z, 1.0]);
    Point2::new(q[0] / q[2], q[1] / q[2])
}

/// The camera matrix from a projection, and `C(2,2)` of `C = B * B^T` (B the left 3x3).
/// Element (0, 1) is the skew `g`; `-999` stands for a square root of a value that is not
/// positive, as in the C++.
///
/// Ported from `CubeCalibration.cpp:1031-1067` (the same code at `:601-636`); the same
/// operations are `Calibration::setInitialByReferences` in `processing/Calibration.cpp`,
/// which then sets element (0, 1) to 0.
#[must_use]
pub fn decompose(projection: &Matrix3x4) -> (Matrix3, f64) {
    let mut b = [[0.0f64; 3]; 3];
    let mut bt = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            b[i][j] = projection[(i, j)];
        }
    }
    // transpose(B, BT)
    for i in 0..3 {
        for j in 0..3 {
            bt[j][i] = b[i][j];
        }
    }
    // cv::Mat C = B * BT;
    let c = gemm_small(&b, &bt);
    let u0 = c[0][2] / c[2][2];
    let v0 = c[1][2] / c[2][2];
    let ku = c[0][0] / c[2][2];
    let kc = c[0][1] / c[2][2];
    let kv = c[1][1] / c[2][2];
    let b_ = if (kv - v0 * v0) > 0.0 {
        (kv - v0 * v0).sqrt()
    } else {
        -999.0
    };
    let g = (kc - u0 * v0) / b_;
    let a = if ku - u0 * u0 - g * g > 0.0 {
        (ku - u0 * u0 - g * g).sqrt()
    } else {
        -999.0
    };
    // cameramatrix(2, 2) = C(2,2) / C(2,2): 1, or NaN for 0, infinite or NaN C(2,2).
    #[allow(clippy::eq_op)]
    let k22 = c[2][2] / c[2][2];
    let k = Matrix3::new(a, g, u0, 0.0, b_, v0, 0.0, 0.0, k22);
    (k, c[2][2])
}

/// `projection = projection / sqrt(C(2,2))`: OpenCV's `MatExpr` scales by `1./s`
/// (`convertTo(m, -1, alpha, 0)`: `p * alpha + 0`).
///
/// Ported from `CubeCalibration.cpp:674`, `:1069`; OpenCV 4.13
/// `modules/core/src/matrix_expressions.cpp` (`operator /`, `MatOp_AddEx::assign`) and
/// `convert_scale.simd.hpp` (`cvt_64f`).
fn normalise(projection: &mut Matrix3x4, c22: f64) {
    let alpha = 1.0 / c22.sqrt();
    for v in projection.iter_mut() {
        *v = *v * alpha + 0.0;
    }
}

/// The DLT: the projection from at least six 3D-2D pairs, and its mean reprojection error;
/// 1000 (and `projection` unchanged) for fewer than six pairs or unequal counts.
///
/// Ported from `CubeCalibration.cpp:790-875`, `CubeCalibration::computeProjection`.
/// `cv::SVD::solveZ` is [`linalg::solve_z`] (ledger A37 for 13 or more pairs).
fn compute_projection(pt2d: &[Point2], pt3d: &[Point3], projection: &mut Matrix3x4) -> f64 {
    if pt2d.len() != pt3d.len() || pt2d.len() < 6 {
        return 1000.0;
    }
    let n = pt2d.len();
    let mut a = DMatrix::<f64>::zeros(2 * n, 12);
    for i in 0..n {
        let (x, y, z) = (pt3d[i].x, pt3d[i].y, pt3d[i].z);
        let (u, v) = (pt2d[i].x, pt2d[i].y);
        let r = 2 * i;
        a[(r, 0)] = x;
        a[(r, 1)] = y;
        a[(r, 2)] = z;
        a[(r, 3)] = 1.0;
        a[(r, 8)] = -u * x;
        a[(r, 9)] = -u * y;
        a[(r, 10)] = -u * z;
        a[(r, 11)] = -u;
        a[(r + 1, 4)] = x;
        a[(r + 1, 5)] = y;
        a[(r + 1, 6)] = z;
        a[(r + 1, 7)] = 1.0;
        a[(r + 1, 8)] = -v * x;
        a[(r + 1, 9)] = -v * y;
        a[(r + 1, 10)] = -v * z;
        a[(r + 1, 11)] = -v;
    }
    let f = linalg::solve_z(&a);
    for r in 0..3 {
        for c in 0..4 {
            projection[(r, c)] = f[r * 4 + c];
        }
    }
    let mut d = 0.0;
    for i in 0..n {
        let pt2di = project_with(projection, pt3d[i]);
        d += euclidean_dist(pt2di, pt2d[i]);
    }
    #[allow(clippy::cast_precision_loss)]
    let n_f = n as f64;
    d / n_f
}

/// The matching shared by `computeInlier`, `computeProjectionFromInliers` and
/// `setCorrespondances`: each projected bead takes the nearest detected bead within
/// `threshold` that no other bead holds more closely; passes repeat while a bead was taken
/// from another. Returns per projected bead its detected index and distance, and the count.
///
/// Ported from `CubeCalibration.cpp:731-786` (the same loop at `:886-931`, `:954-1020`).
fn match_beads(
    projected: &[Point2],
    detected: &[Point2],
    threshold: f64,
) -> (Vec<Option<(usize, f64)>>, i32) {
    let mut mask_dist = vec![threshold; detected.len()];
    loop {
        let mut out = Vec::with_capacity(projected.len());
        let mut inlier = 0;
        let mut duplicates = false;
        for &pt2d in projected {
            let mut distmin = threshold;
            let mut idx = 0usize;
            for (j, &pttmp) in detected.iter().enumerate() {
                let d = euclidean_dist(pt2d, pttmp);
                if d < distmin && d < threshold && d <= mask_dist[j] {
                    distmin = d;
                    idx = j;
                }
            }
            if distmin < threshold {
                if distmin < mask_dist[idx] {
                    duplicates = true;
                }
                mask_dist[idx] = distmin;
                out.push(Some((idx, distmin)));
                inlier += 1;
            } else {
                out.push(None);
            }
        }
        if !duplicates {
            return (out, inlier);
        }
    }
}

/// Ported from `CubeCalibration.cpp:877-934`, `CubeCalibration::computeInlier`: the number
/// of calibration-object points whose projection with `projection` matches a detected bead.
fn compute_inlier(
    coords3d: &[Point3],
    all_detected: &[Point2],
    projection: &Matrix3x4,
    threshold: f64,
) -> i32 {
    let projected: Vec<Point2> = coords3d
        .iter()
        .map(|&p| project_with(projection, p))
        .collect();
    match_beads(&projected, all_detected, threshold).1
}

/// Iterations of the six-point search per batch (see [`Cube::setup_correspondances_ransac`]).
const SEARCH_BATCH: usize = 8192;

/// Iterations below which the six-point search stays on the calling thread.
const SEARCH_MIN_PARALLEL: u32 = 1024;

/// Threads for a six-point search of `loop_max` iterations: the machine's parallelism
/// (read once per process), one for a short search.
fn search_workers(loop_max: u32) -> usize {
    use std::sync::OnceLock;
    static THREADS: OnceLock<usize> = OnceLock::new();
    if loop_max < SEARCH_MIN_PARALLEL {
        return 1;
    }
    *THREADS
        .get_or_init(|| std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get))
}

/// One draw of the six-point search: `getRandomReferences` (points 2D, 3D, ids 3D, ids 2D).
type SixDraw = (Vec<Point2>, Vec<Point3>, Vec<i32>, Vec<i32>);

/// The pure part of one six-point iteration.
#[derive(Clone, Copy, Debug)]
struct SixEval {
    /// `computeProjection`'s mean error.
    error: f64,
    /// The projection it computed.
    projection: Matrix3x4,
    /// `computeInlier`, computed only when `error < threshold` (0 otherwise, not read).
    inlier: i32,
}

impl Default for SixEval {
    fn default() -> Self {
        SixEval {
            error: 0.0,
            projection: Matrix3x4::zeros(),
            inlier: 0,
        }
    }
}

/// The DLT of one draw, and its inlier count when its error is below `threshold`
/// (`CubeCalibration.cpp:476-480`). A draw always has six pairs, so `computeProjection`
/// overwrites the whole projection and the value the C++ carries over from the previous
/// iteration is never read.
fn evaluate_six(
    coords3d: &[Point3],
    all_detected: &[Point2],
    draw: &SixDraw,
    threshold: f64,
) -> SixEval {
    let mut projection = Matrix3x4::zeros();
    let error = compute_projection(&draw.0, &draw.1, &mut projection);
    let inlier = if error < threshold {
        compute_inlier(coords3d, all_detected, &projection, threshold)
    } else {
        0
    };
    SixEval {
        error,
        projection,
        inlier,
    }
}

fn to_f32_3(p: Point3) -> nalgebra::Point3<f32> {
    #[allow(clippy::cast_possible_truncation)]
    nalgebra::Point3::new(p.x as f32, p.y as f32, p.z as f32)
}

fn to_f32_2(p: Point2) -> nalgebra::Point2<f32> {
    #[allow(clippy::cast_possible_truncation)]
    nalgebra::Point2::new(p.x as f32, p.y as f32)
}

/// OpenCV's integer value of the flags `calibrateOpenCV` passes, for the trace.
#[must_use]
pub fn opencv_flags(flags: CalibrateFlags) -> i32 {
    let mut v = 0;
    if flags.use_intrinsic_guess {
        v += 1;
    }
    if flags.fix_principal_point {
        v += 4;
    }
    if flags.zero_tangent_dist {
        v += 8;
    }
    for (k, bit) in [32, 64, 128, 2048, 4096, 8192].iter().enumerate() {
        if flags.fix_k[k] {
            v += bit;
        }
    }
    if flags.rational_model {
        v += 16384;
    }
    v
}

impl<'a, 'c, 't> Cube<'a, 'c, 't> {
    /// Ported from `CubeCalibration.cpp:51-93` (the constructor), with the points already
    /// undistorted by the caller. `calibrated` is the camera's flag.
    fn new(
        input: &CubeInput<'a>,
        calibrated: bool,
        calibrate: Option<&'c mut CalibrateCameraFn<'c>>,
        trace: Option<&'t mut CubeTrace>,
    ) -> Self {
        let mut selected = [Point2::origin(); 4];
        let mut selected_id = [0i32; 4];
        for (i, r) in input.references.iter().enumerate() {
            selected[i] = r.position;
            selected_id[i] = r.point_id;
        }
        Cube {
            coords3d: input.object_points,
            all_detected: input.detected,
            selected,
            selected_id,
            used: references_used(&input.references, input.compat),
            settings: input.settings,
            compat: input.compat,
            maxinlier: 0,
            pose_computed: false,
            calibrated,
            projected: Vec::new(),
            detected: Vec::new(),
            error: Vec::new(),
            inlier: Vec::new(),
            projection: Matrix3x4::zeros(),
            rvec: Vector3::zeros(),
            tvec: Vector3::zeros(),
            camera_matrix: Matrix3::zeros(),
            calibrate,
            trace,
            trace_loop: 0,
            trace_round: 0,
        }
    }

    /// Refuses what the C++ would read out of range or loop on for ever: a used reference's
    /// id outside the calibration object, and (when `draws`) a search that cannot complete
    /// a set of `nb` points.
    fn check(&self, nb: usize, draws: bool) -> Result<(), CubeError> {
        let n3d = self.coords3d.len();
        let mut ids: Vec<i32> = Vec::new();
        for i in 0..4 {
            if self.used[i] {
                let id = self.selected_id[i];
                if id < 0 || usize::try_from(id).is_ok_and(|v| v >= n3d) {
                    return Err(CubeError::ReferenceOutOfRange {
                        slot: i,
                        point_id: id,
                    });
                }
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
        }
        if draws {
            let set = self.used.iter().filter(|&&u| u).count();
            let needed = nb.saturating_sub(set);
            let free = n3d - ids.len();
            if needed > 0 && (free < needed || self.all_detected.len() < needed) {
                return Err(CubeError::CannotDraw {
                    needed,
                    object_points_free: free,
                    detected: self.all_detected.len(),
                });
            }
        }
        Ok(())
    }

    fn result(self, mode: CubeMode) -> CubeResult {
        CubeResult {
            mode,
            calibrated: self.calibrated,
            pose_computed: self.pose_computed,
            max_inlier: self.maxinlier,
            camera_matrix: self.camera_matrix,
            pose: RigidTransform {
                rotation: self.rvec,
                translation: self.tvec,
            },
            detected: self.detected,
            projected: self.projected,
            inlier: self.inlier,
            error: self.error,
        }
    }

    fn pose(&self) -> RigidTransform {
        RigidTransform {
            rotation: self.rvec,
            translation: self.tvec,
        }
    }

    /// The used references, then random pairs of an unused calibration-object point and an
    /// unused detected bead until `nb_points` pairs. Records the ids (`last_ids`).
    ///
    /// Ported from `CubeCalibration.cpp:264-317`, `CubeCalibration::getRandomReferences`.
    /// `rand() % coords3DSize`, then `rand() % alldetectedPointsSize`, two draws per
    /// attempt.
    fn get_random_references(
        &self,
        nb_points: usize,
        rng: &mut CRand,
    ) -> (Vec<Point2>, Vec<Point3>, Vec<i32>, Vec<i32>) {
        let mut pt2d = Vec::with_capacity(nb_points);
        let mut pt3d = Vec::with_capacity(nb_points);
        let mut idx_coords3d: Vec<i32> = Vec::with_capacity(nb_points);
        let mut idx_keypoints: Vec<i32> = Vec::with_capacity(nb_points);
        for i in 0..4 {
            if self.used[i] {
                let id = self.selected_id[i];
                idx_coords3d.push(id);
                #[allow(clippy::cast_sign_loss)]
                pt3d.push(self.coords3d[id as usize]);
                pt2d.push(self.selected[i]);
                idx_keypoints.push(-1);
            }
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let n3d = self.coords3d.len() as i32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let n2d = self.all_detected.len() as i32;
        while idx_coords3d.len() < nb_points {
            let ran_3d = rng.next() % n3d;
            let okay_3d = !idx_coords3d.contains(&ran_3d);
            let ran_2d = rng.next() % n2d;
            let okay_2d = !idx_keypoints.contains(&ran_2d);
            if okay_2d && okay_3d {
                idx_coords3d.push(ran_3d);
                #[allow(clippy::cast_sign_loss)]
                pt3d.push(self.coords3d[ran_3d as usize]);
                idx_keypoints.push(ran_2d);
                #[allow(clippy::cast_sign_loss)]
                pt2d.push(self.all_detected[ran_2d as usize]);
            }
        }
        (pt2d, pt3d, idx_coords3d, idx_keypoints)
    }

    /// The six-point search.
    ///
    /// Ported from `CubeCalibration.cpp:468-502`, `CubeCalibration::setupCorrespondancesRansac`.
    ///
    /// The C++ runs its iterations one after another. Here the iterations are taken in
    /// batches: the sets of a batch are drawn one after another from `rng` (the same draws
    /// in the same order), the pure part of each iteration (the DLT, and the inlier count
    /// when its error is below the threshold) is computed on [`search_workers`] threads,
    /// and the acceptance is then replayed in loop order. The pure part reads only the
    /// draw, the calibration-object points and the detected beads, none of which an
    /// acceptance changes, so every number and the trace are those of the sequential loop,
    /// whatever the number of threads.
    fn setup_correspondances_ransac(&mut self, loop_max: u32, threshold: f64, rng: &mut CRand) {
        let threads = search_workers(loop_max);
        self.setup_correspondances_ransac_in(
            loop_max,
            threshold,
            rng,
            threads,
            SEARCH_BATCH,
            usize::MAX,
        );
    }

    /// [`Cube::setup_correspondances_ransac`] on `threads` threads (1: the calling thread
    /// alone) in batches of `batch` iterations. Starting a thread fails (as when the
    /// operating system refuses one) after `max_spawns` threads have started in a batch:
    /// `usize::MAX` in the program, smaller only in tests. A thread that cannot be started
    /// leaves its share to the threads that did start and to the calling thread, with the
    /// same result.
    fn setup_correspondances_ransac_in(
        &mut self,
        loop_max: u32,
        threshold: f64,
        rng: &mut CRand,
        threads: usize,
        batch: usize,
        max_spawns: usize,
    ) {
        let batch = batch.max(1);
        let threads = threads.clamp(1, batch);
        self.maxinlier = 0;
        let (coords3d, all_detected) = (self.coords3d, self.all_detected);
        let mut draws: Vec<SixDraw> = Vec::new();
        let mut evals: Vec<SixEval> = Vec::new();
        let mut lp0 = 0u32;
        while lp0 < loop_max {
            let n = usize::try_from(loop_max - lp0).map_or(batch, |r| r.min(batch));
            draws.clear();
            for _ in 0..n {
                draws.push(self.get_random_references(6, rng));
            }
            evals.clear();
            evals.resize(n, SixEval::default());
            let eval = |d: &[SixDraw], e: &mut [SixEval]| {
                for (d, e) in d.iter().zip(e.iter_mut()) {
                    *e = evaluate_six(coords3d, all_detected, d, threshold);
                }
            };
            if threads == 1 {
                eval(&draws, &mut evals);
            } else {
                let chunk = n.div_ceil(threads).max(1);
                // The chunks are taken from a shared index by the workers and by the
                // calling thread, so a worker that cannot be started leaves its chunk to
                // the others (each iteration's result depends on its draw alone).
                let parts: Vec<std::sync::Mutex<Option<_>>> = draws
                    .chunks(chunk)
                    .zip(evals.chunks_mut(chunk))
                    .map(|p| std::sync::Mutex::new(Some(p)))
                    .collect();
                let next = std::sync::atomic::AtomicUsize::new(0);
                let work = || loop {
                    let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(slot) = parts.get(k) else { break };
                    let part = slot
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .take();
                    if let Some((d, e)) = part {
                        eval(d, e);
                    }
                };
                std::thread::scope(|s| {
                    for started in 0..parts.len() - 1 {
                        if started >= max_spawns
                            || std::thread::Builder::new().spawn_scoped(s, work).is_err()
                        {
                            break;
                        }
                    }
                    work();
                });
            }
            // The acceptance, in loop order.
            for (k, (draw, ev)) in draws.iter_mut().zip(&evals).enumerate() {
                #[allow(clippy::cast_possible_truncation)]
                let lp = lp0 + k as u32;
                let tmp_projection = ev.projection;
                let error_projection = ev.error;
                if error_projection < threshold {
                    let inlier = ev.inlier;
                    if inlier > self.maxinlier {
                        if let Some(t) = self.trace.as_deref_mut() {
                            t.ransac_accepted.push(RansacAccepted {
                                loop_index: lp,
                                ids3d: std::mem::take(&mut draw.2),
                                ids2d: std::mem::take(&mut draw.3),
                                projection_error: error_projection,
                                n_inlier: inlier,
                                best_before: self.maxinlier,
                                projection: tmp_projection,
                            });
                        }
                        self.trace_loop = lp;
                        self.trace_round = 0;
                        self.maxinlier = inlier;
                        self.projection = tmp_projection;
                        if self.maxinlier > 0 {
                            self.compute_projection_from_inliers(threshold);
                        }
                    }
                }
            }
            #[allow(clippy::cast_possible_truncation)]
            {
                lp0 += n as u32;
            }
        }
    }

    /// The projections of every calibration-object point with `projection`.
    fn project_all(&self, projection: &Matrix3x4) -> Vec<Point2> {
        self.coords3d
            .iter()
            .map(|&p| project_with(projection, p))
            .collect()
    }

    /// Matches with the current projection, refits it from the matches by the DLT, and
    /// decomposes it into the camera matrix; repeats while the match count grows.
    ///
    /// Ported from `CubeCalibration.cpp:936-1071`,
    /// `CubeCalibration::computeProjectionFromInliers`.
    fn compute_projection_from_inliers(&mut self, threshold: f64) {
        let mut inlier_changed = true;
        while inlier_changed {
            inlier_changed = false;
            let projected = self.project_all(&self.projection);
            let (matches, inlier) = match_beads(&projected, self.all_detected, threshold);
            let mut corr3d = Vec::new();
            let mut corr2d = Vec::new();
            self.detected.clear();
            self.inlier.clear();
            self.error.clear();
            for (i, m) in matches.iter().enumerate() {
                match *m {
                    Some((idx, distmin)) => {
                        corr3d.push(self.coords3d[i]);
                        corr2d.push(self.all_detected[idx]);
                        self.detected.push(self.all_detected[idx]);
                        self.inlier.push(true);
                        self.error.push(distmin);
                    }
                    None => {
                        self.detected.push(Point2::new(-1.0, -1.0));
                        self.inlier.push(false);
                        self.error.push(10000.0);
                    }
                }
            }
            self.projected = projected;

            if inlier > self.maxinlier {
                self.maxinlier = inlier;
                inlier_changed = true;
            }

            // The return value is not read: with fewer than six matches the projection
            // stays.
            let _ = compute_projection(&corr2d, &corr3d, &mut self.projection);
            let (k, c22) = decompose(&self.projection);
            self.camera_matrix = k;
            normalise(&mut self.projection, c22);

            let (lp, round) = (self.trace_loop, self.trace_round);
            self.trace_round += 1;
            if let Some(t) = self.trace.as_deref_mut() {
                t.ransac_refine_rounds.push(RansacRefineRound {
                    loop_index: lp,
                    round,
                    n_matched: inlier,
                    max_inlier: self.maxinlier,
                    n_correspondences: corr2d.len(),
                    projection: self.projection,
                    camera_matrix: self.camera_matrix,
                });
            }
        }
    }

    /// `cv::calibrateCamera` on the matched inliers, starting from the current camera
    /// matrix; on success the pose and the projections. Returns `poseComputed`, or `false`
    /// with 8 or fewer pairs.
    ///
    /// Ported from `CubeCalibration.cpp:325-421`, `CubeCalibration::calibrateOpenCV`
    /// (`singleFocal` is always false in XMALab: every call uses the default).
    fn calibrate_opencv(&mut self) -> bool {
        let mut corr3d = Vec::new();
        let mut corr2d = Vec::new();
        for i in 0..self.detected.len() {
            if self.inlier[i] && self.detected[i].x > -1.0 && self.detected[i].y > -1.0 {
                corr3d.push(self.coords3d[i]);
                corr2d.push(self.detected[i]);
            }
        }
        if corr3d.len() <= 8 {
            return false;
        }
        let object_points: Vec<nalgebra::Point3<f32>> =
            corr3d.iter().map(|&p| to_f32_3(p)).collect();
        let image_points: Vec<nalgebra::Point2<f32>> =
            corr2d.iter().map(|&p| to_f32_2(p)).collect();

        let width = self.settings.width;
        let height = self.settings.height;
        // The C++ computes `width - 1` and `width + 1` in `int`; in `f64` they are exact for
        // every `int` and do not overflow at the ends of its range.
        let (w, h) = (f64::from(width), f64::from(height));
        let k = &mut self.camera_matrix;
        if k[(0, 2)] < 0.0 {
            k[(0, 2)] = 1.0;
        }
        if k[(0, 2)] > w {
            k[(0, 2)] = w - 1.0;
        }
        if k[(1, 2)] < 0.0 {
            k[(1, 2)] = 0.0;
        }
        if k[(1, 2)] > h {
            k[(1, 2)] = h - 1.0;
        }
        k[(0, 1)] = 0.0;

        // CALIB_USE_INTRINSIC_GUESS + CALIB_FIX_K1 + CALIB_FIX_K2 + CALIB_FIX_K3 +
        // CALIB_ZERO_TANGENT_DIST
        let mut flags = CalibrateFlags {
            use_intrinsic_guess: true,
            fix_principal_point: false,
            zero_tangent_dist: true,
            fix_k: [true, true, true, false, false, false],
            rational_model: false,
        };
        if self.settings.fix_principal {
            flags.fix_principal_point = true;
            if self.compat.principal_point_one_based {
                k[(0, 2)] = 0.5 * (w + 1.0);
                k[(1, 2)] = 0.5 * (h + 1.0);
            } else {
                k[(0, 2)] = 0.5 * (w - 1.0);
                k[(1, 2)] = 0.5 * (h - 1.0);
            }
        }
        let k_in = *k;
        // std::vector<float> distCoefs(5, 0.0f)
        let dist = [0.0f64; 5];
        let objects = [object_points];
        let images = [image_points];
        #[allow(clippy::cast_sign_loss)]
        let size = (width.max(0) as usize, height.max(0) as usize);
        let out = match self.calibrate.as_deref_mut() {
            Some(f) => f(&objects, &images, size, &k_in, &dist, flags),
            None => Err(CalibrateError::NotPorted),
        };
        match &out {
            Ok(o) if !o.poses.is_empty() => {
                self.camera_matrix = o.camera_matrix;
                self.rvec = o.poses[0].rotation;
                self.tvec = o.poses[0].translation;
                self.pose_computed = true;
                self.reproject_and_compute_error();
            }
            _ => {
                self.pose_computed = false;
            }
        }
        if let Some(t) = self.trace.as_deref_mut() {
            let [object_points] = objects;
            let [image_points] = images;
            t.calibrate_calls.push(CalibrateCall {
                object_points,
                image_points,
                flags,
                camera_matrix_in: k_in,
                output: out,
            });
        }
        self.pose_computed
    }

    /// Projects every calibration-object point with the current pose and camera matrix (in
    /// `float`, no distortion), and recomputes the errors of the current matches.
    ///
    /// Ported from `CubeCalibration.cpp:423-466`, `CubeCalibration::reprojectAndComputeError`.
    fn reproject_and_compute_error(&mut self) {
        let object_points: Vec<nalgebra::Point3<f32>> =
            self.coords3d.iter().map(|&p| to_f32_3(p)).collect();
        let image_points = camera::project_points_f32(
            &object_points,
            &self.rvec,
            &self.tvec,
            &self.camera_matrix,
            &[],
        );
        self.projected = image_points
            .iter()
            .map(|p| Point2::new(f64::from(p.x), f64::from(p.y)))
            .collect();
        for i in 0..self.detected.len() {
            self.error[i] = euclidean_dist(self.projected[i], self.detected[i]);
        }
    }

    /// Ported from `CubeCalibration.cpp:504-525`, `CubeCalibration::selectCorrespondances`.
    fn select_correspondances(&mut self, threshold: f64) -> i32 {
        if self.error.len() != self.inlier.len() {
            return 0;
        }
        let mut count = 0;
        for i in 0..self.inlier.len() {
            if self.error[i] < threshold {
                self.inlier[i] = true;
                count += 1;
            } else {
                self.inlier[i] = false;
            }
        }
        count
    }

    /// Alternates refits (camera and pose, or pose alone) and matchings while the inlier
    /// count grows, then refits once more.
    ///
    /// Ported from `CubeCalibration.cpp:527-571`, `CubeCalibration::refineResults`.
    fn refine_results(&mut self, with_camera_refinement: bool) {
        let ident = f64::from(self.settings.identification_threshold);
        let outlier = f64::from(self.settings.outlier_threshold);
        self.maxinlier = self.set_correspondances(0.5 * ident, true);
        self.push_refine_round(with_camera_refinement, 0, -1, self.maxinlier, -1);

        if self.maxinlier > 5 {
            let mut inlier_tmp = 0;
            let mut round = 0;
            while inlier_tmp == 0 || inlier_tmp < self.maxinlier {
                inlier_tmp = self.maxinlier;
                if with_camera_refinement {
                    self.calibrate_from_inliers();
                } else {
                    self.set_pose_from_inlier();
                }
                let matched = self.set_correspondances(ident, false);
                self.maxinlier = self.select_correspondances(outlier);
                round += 1;
                self.push_refine_round(
                    with_camera_refinement,
                    round,
                    inlier_tmp,
                    matched,
                    self.maxinlier,
                );
            }

            if with_camera_refinement {
                self.calibrated = true;
            }
            self.pose_computed = true;

            let ok = if with_camera_refinement {
                self.calibrate_from_inliers()
            } else {
                self.set_pose_from_inlier()
            };
            self.push_refine_round(
                with_camera_refinement,
                -1,
                self.maxinlier,
                -1,
                self.maxinlier,
            );
            if !ok && !self.compat.success_after_failed_refit {
                self.fail(with_camera_refinement);
            } else if ok && !self.compat.inliers_of_previous_camera {
                // Ledger A156 fixed: the matching of the returned camera and pose.
                self.set_correspondances(ident, false);
                self.maxinlier = self.select_correspondances(outlier);
            }
        } else {
            self.fail(with_camera_refinement);
        }
    }

    /// The failure branch of `refineResults` (`CubeCalibration.cpp:564-570`).
    fn fail(&mut self, with_camera_refinement: bool) {
        self.projected.clear();
        self.detected.clear();
        if with_camera_refinement {
            self.calibrated = false;
        }
        self.pose_computed = false;
    }

    fn push_refine_round(
        &mut self,
        with_camera: bool,
        round: i32,
        matched_before: i32,
        matched: i32,
        inliers: i32,
    ) {
        let (k, pose) = (self.camera_matrix, self.pose());
        if let Some(t) = self.trace.as_deref_mut() {
            t.refine_rounds.push(RefineRound {
                with_camera,
                round,
                matched_before,
                matched,
                inliers,
                camera_matrix: k,
                pose,
            });
        }
    }

    /// EPnP on the inlier pairs, when the lists agree in length. Returns whether the pose
    /// was computed.
    ///
    /// Ported from `CubeCalibration.cpp:573-595`, `CubeCalibration::setPoseFromInlier`.
    fn set_pose_from_inlier(&mut self) -> bool {
        if self.inlier.len() == self.detected.len() && self.coords3d.len() == self.inlier.len() {
            let mut pt2d = Vec::new();
            let mut pt3d = Vec::new();
            for i in 0..self.inlier.len() {
                if self.inlier[i] {
                    pt3d.push(self.coords3d[i]);
                    pt2d.push(self.detected[i]);
                }
            }
            return self.compute_pose(&pt2d, &pt3d);
        }
        false
    }

    /// The DLT from the inliers, normalised, decomposed into the camera matrix, then
    /// `calibrateOpenCV`. Returns its result.
    ///
    /// Ported from `CubeCalibration.cpp:597-639`, `CubeCalibration::calibrateFromInliers`.
    fn calibrate_from_inliers(&mut self) -> bool {
        self.compute_projection_matrix_from_inlier();
        let (k, _) = decompose(&self.projection);
        self.camera_matrix = k;
        let (p, k) = (self.projection, self.camera_matrix);
        if let Some(t) = self.trace.as_deref_mut() {
            t.k_from_projection.push(KFromProjection {
                projection: p,
                camera_matrix: k,
            });
        }
        self.calibrate_opencv()
    }

    /// Ported from `CubeCalibration.cpp:641-679`,
    /// `CubeCalibration::computeProjectionMatrixFromInlier`.
    fn compute_projection_matrix_from_inlier(&mut self) {
        let mut corr3d = Vec::new();
        let mut corr2d = Vec::new();
        for i in 0..self.detected.len() {
            if self.inlier[i] && self.detected[i].x > -1.0 && self.detected[i].y > -1.0 {
                corr3d.push(self.coords3d[i]);
                corr2d.push(self.detected[i]);
            }
        }
        if compute_projection(&corr2d, &corr3d, &mut self.projection) < 1000.0 {
            let (_, c22) = decompose(&self.projection);
            normalise(&mut self.projection, c22);
        }
    }

    /// `cv::solvePnP(..., SOLVEPNP_EPNP)` on single-precision points with the camera
    /// matrix and no distortion, then the projections; only for a calibrated camera and at
    /// least five pairs. Returns whether the pose was computed.
    ///
    /// Ported from `CubeCalibration.cpp:681-721`, `CubeCalibration::computePose`.
    fn compute_pose(&mut self, pt2d: &[Point2], pt3d: &[Point3]) -> bool {
        if self.calibrated && pt2d.len() == pt3d.len() && pt2d.len() >= 5 {
            let object_points: Vec<nalgebra::Point3<f32>> =
                pt3d.iter().map(|&p| to_f32_3(p)).collect();
            let image_points: Vec<nalgebra::Point2<f32>> =
                pt2d.iter().map(|&p| to_f32_2(p)).collect();
            if let Some(pose) =
                crate::epnp::solve_pnp_epnp(&object_points, &image_points, &self.camera_matrix, &[])
            {
                self.rvec = pose.rotation;
                self.tvec = pose.translation;
                self.reproject_and_compute_error();
                return true;
            }
            self.pose_computed = false;
        }
        false
    }

    /// Matches the projections to the detected beads; with `set_as_inliers` the matches
    /// become the inliers. Returns the match count, or 0 (changing nothing) when there are
    /// no projections of every point or no detected bead.
    ///
    /// Ported from `CubeCalibration.cpp:723-788`, `CubeCalibration::setCorrespondances`.
    fn set_correspondances(&mut self, threshold: f64, set_as_inliers: bool) -> i32 {
        if self.projected.len() != self.coords3d.len() || self.all_detected.is_empty() {
            return 0;
        }
        let (matches, inlier) = match_beads(&self.projected, self.all_detected, threshold);
        self.detected.clear();
        if set_as_inliers {
            self.inlier.clear();
        }
        self.error.clear();
        for m in &matches {
            match *m {
                Some((idx, distmin)) => {
                    self.detected.push(self.all_detected[idx]);
                    if set_as_inliers {
                        self.inlier.push(true);
                    }
                    self.error.push(distmin);
                }
                None => {
                    self.detected.push(Point2::new(-1.0, -1.0));
                    if set_as_inliers {
                        self.inlier.push(false);
                    }
                    self.error.push(10000.0);
                }
            }
        }
        inlier
    }

    /// The five-point pose search.
    ///
    /// Ported from `CubeCalibration.cpp:1073-1119`,
    /// `CubeCalibration::setupCorrespondancesRansacPose` (its `threshold` argument is not
    /// read).
    fn setup_correspondances_ransac_pose(&mut self, loop_max: u32, rng: &mut CRand) {
        let half = 0.5 * f64::from(self.settings.identification_threshold);
        self.maxinlier = 0;
        let mut pt2d_best: Vec<Point2> = Vec::new();
        let mut pt3d_best: Vec<Point3> = Vec::new();
        for lp in 0..loop_max {
            let (pt2d, pt3d, ids3d, ids2d) = self.get_random_references(5, rng);
            self.compute_pose(&pt2d, &pt3d);
            let inlier = self.set_correspondances(half, true);
            if inlier > self.maxinlier {
                let pose = self.pose();
                if let Some(t) = self.trace.as_deref_mut() {
                    t.ransac_pose_accepted.push(RansacPoseAccepted {
                        loop_index: lp,
                        ids3d,
                        ids2d,
                        n_inlier: inlier,
                        best_before: self.maxinlier,
                        pose,
                    });
                }
                pt2d_best.clear();
                pt3d_best.clear();
                self.maxinlier = inlier;
                for i in 0..self.inlier.len() {
                    if self.inlier[i] {
                        pt2d_best.push(self.detected[i]);
                        pt3d_best.push(self.coords3d[i]);
                    }
                }
            }
        }

        if self.maxinlier >= 4 {
            self.compute_pose(&pt2d_best, &pt3d_best);
            self.reproject_and_compute_error();
            self.maxinlier =
                self.set_correspondances(f64::from(self.settings.identification_threshold), true);
        }
        let (n_best, maxinlier, pose) = (pt2d_best.len(), self.maxinlier, self.pose());
        if let Some(t) = self.trace.as_deref_mut() {
            t.ransac_pose_result = Some(RansacPoseResult {
                n_best,
                max_inlier: maxinlier,
                pose,
            });
        }
    }

    /// Ported from `CubeCalibration.cpp:233-261`, `CubeCalibration::setPose_thread`.
    /// `transformationmatrix.inv()` is [`linalg::invert`] (`DECOMP_LU`; all zeros for a
    /// singular matrix, as OpenCV), `cv::Rodrigues` [`camera::matrix_to_rodrigues`].
    fn set_pose_thread(&mut self, transformation: &Matrix4<f64>) {
        let trans_inv = linalg::invert(transformation).x;
        let mut rotation = Matrix3::zeros();
        for i in 0..3 {
            for j in 0..3 {
                rotation[(i, j)] = trans_inv[(i, j)];
            }
        }
        self.rvec = camera::matrix_to_rodrigues(&rotation);
        for j in 0..3 {
            self.tvec[j] = trans_inv[(j, 3)];
        }
        self.reproject_and_compute_error();
        self.maxinlier =
            self.set_correspondances(f64::from(self.settings.identification_threshold), true);
        let (pose, n) = (self.pose(), self.maxinlier);
        if let Some(t) = self.trace.as_deref_mut() {
            t.set_pose = Some(SetPoseTrace {
                transformation: *transformation,
                pose,
                n_matched: n,
            });
        }
        self.set_pose_from_inlier();
        self.refine_results(false);
    }
}

#[cfg(test)]
mod tests {
    //! Review 13 (performance round, calibration.md finding 3): the batched, threaded
    //! six-point search against the sequential loop it replaced, kept here as the reference.

    use super::*;

    /// The six-point search as it was before the performance round of 20 November 2026:
    /// one iteration after another on the calling thread.
    fn reference_search(
        cube: &mut Cube<'_, '_, '_>,
        loop_max: u32,
        threshold: f64,
        rng: &mut CRand,
    ) {
        let mut tmp_projection = Matrix3x4::zeros();
        cube.maxinlier = 0;
        for lp in 0..loop_max {
            let (pt2d, pt3d, ids3d, ids2d) = cube.get_random_references(6, rng);
            let error_projection = compute_projection(&pt2d, &pt3d, &mut tmp_projection);
            if error_projection < threshold {
                let projected = cube.project_all(&tmp_projection);
                let inlier = match_beads(&projected, cube.all_detected, threshold).1;
                if inlier > cube.maxinlier {
                    if let Some(t) = cube.trace.as_deref_mut() {
                        t.ransac_accepted.push(RansacAccepted {
                            loop_index: lp,
                            ids3d,
                            ids2d,
                            projection_error: error_projection,
                            n_inlier: inlier,
                            best_before: cube.maxinlier,
                            projection: tmp_projection,
                        });
                    }
                    cube.trace_loop = lp;
                    cube.trace_round = 0;
                    cube.maxinlier = inlier;
                    cube.projection = tmp_projection;
                    if cube.maxinlier > 0 {
                        cube.compute_projection_from_inliers(threshold);
                    }
                }
            }
        }
    }

    /// Everything the search leaves behind, floats as bits.
    #[derive(Debug, PartialEq, Eq)]
    struct After {
        maxinlier: i32,
        trace_loop: u32,
        trace_round: u32,
        state: Vec<u64>,
        inlier: Vec<bool>,
        trace: String,
        rng: CRand,
    }

    fn bits<'x>(v: impl IntoIterator<Item = &'x f64>) -> impl Iterator<Item = u64> {
        v.into_iter().map(|x| x.to_bits())
    }

    /// Runs the search (`None`: the reference; `Some((threads, batch, max_spawns))`: the
    /// new one, with starting a thread failing after `max_spawns` threads) and returns what
    /// it left.
    fn run(
        input: &CubeInput<'_>,
        seed: u32,
        loops: u32,
        how: Option<(usize, usize, usize)>,
    ) -> After {
        let mut trace = CubeTrace::default();
        let mut rng = CRand::new(seed);
        let mut cube = Cube::new(input, false, None, Some(&mut trace));
        cube.check(6, true).unwrap();
        match how {
            None => reference_search(&mut cube, loops, SEARCH_THRESHOLD, &mut rng),
            Some((threads, batch, max_spawns)) => cube.setup_correspondances_ransac_in(
                loops,
                SEARCH_THRESHOLD,
                &mut rng,
                threads,
                batch,
                max_spawns,
            ),
        }
        let mut state: Vec<u64> = bits(cube.projection.iter()).collect();
        state.extend(bits(cube.camera_matrix.iter()));
        state.extend(bits(&cube.error));
        for p in cube.detected.iter().chain(&cube.projected) {
            state.extend([p.x.to_bits(), p.y.to_bits()]);
        }
        state.push(cube.detected.len() as u64);
        state.push(cube.projected.len() as u64);
        let (maxinlier, trace_loop, trace_round, inlier) = (
            cube.maxinlier,
            cube.trace_loop,
            cube.trace_round,
            cube.inlier.clone(),
        );
        drop(cube);
        // `{:?}` of f64 prints the shortest string that reads back to the same bits (and
        // NaN, signed zeros and infinities distinctly), so equal strings are equal traces.
        let trace = format!("{trace:?}");
        After {
            maxinlier,
            trace_loop,
            trace_round,
            state,
            inlier,
            trace,
            rng,
        }
    }

    /// Accepted loop indices of a run's trace (from the trace string's `loop_index: n`).
    fn accepted_loops(trace: &str) -> Vec<u32> {
        let head = trace.split("ransac_refine_rounds").next().unwrap();
        head.match_indices("loop_index: ")
            .map(|(i, m)| {
                let rest = &head[i + m.len()..];
                let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap();
                rest[..end].parse().unwrap()
            })
            .collect()
    }

    /// Compares the search with the reference for several thread counts and batch sizes,
    /// and (review 13c) with starting a thread failing at once, after one worker and after
    /// several; returns whether some acceptance fell on the last iteration of a batch of 2
    /// or more.
    fn compare(input: &CubeInput<'_>, seed: u32, loops: u32, label: &str) -> bool {
        let want = run(input, seed, loops, None);
        let mut late = false;
        for batch in [1, 2, 3, 7, 64, 1000, SEARCH_BATCH] {
            for threads in [1, 2, 3, 5, 16] {
                let got = run(input, seed, loops, Some((threads, batch, usize::MAX)));
                assert_eq!(got, want, "{label}: {threads} threads, batch {batch}");
            }
            for (threads, max_spawns) in [(2, 0), (3, 0), (3, 1), (5, 1), (16, 0), (16, 1), (16, 4)]
            {
                let got = run(input, seed, loops, Some((threads, batch, max_spawns)));
                assert_eq!(
                    got, want,
                    "{label}: {threads} threads failing after {max_spawns}, batch {batch}"
                );
            }
            if batch > 1 {
                late |= accepted_loops(&want.trace)
                    .iter()
                    .any(|&l| (l as usize) % batch == batch - 1);
            }
        }
        // The production entry (machine's thread count, production batch).
        let mut cube_trace = CubeTrace::default();
        let mut rng = CRand::new(seed);
        let mut cube = Cube::new(input, false, None, Some(&mut cube_trace));
        cube.setup_correspondances_ransac(loops, SEARCH_THRESHOLD, &mut rng);
        assert_eq!(cube.maxinlier, want.maxinlier, "{label}: production entry");
        drop(cube);
        assert_eq!(
            format!("{cube_trace:?}"),
            want.trace,
            "{label}: production entry"
        );
        assert_eq!(rng, want.rng, "{label}: production entry");
        late
    }

    /// A synthetic cube image: a 4 x 4 x 4 lattice projected by a camera, some beads
    /// missing, jittered, or added as clutter.
    fn synthetic(rng: &mut CRand, n_refs: usize) -> (Vec<Point3>, Vec<Point2>, [CubeReference; 4]) {
        let mut u = || f64::from(rng.next()) / f64::from(i32::MAX);
        let mut obj = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    obj.push(Point3::new(
                        6.5 * f64::from(i),
                        6.5 * f64::from(j),
                        6.5 * f64::from(k),
                    ));
                }
            }
        }
        let f = 1500.0 + 500.0 * u();
        let p = Matrix3x4::new(
            f,
            0.0,
            500.0,
            100.0,
            0.0,
            f,
            400.0,
            80.0,
            0.0,
            0.0,
            1.0,
            150.0 + 50.0 * u(),
        );
        let mut det = Vec::new();
        let mut proj = Vec::new();
        for &q in &obj {
            let x = project_with(&p, q);
            proj.push(x);
            if u() < 0.8 {
                det.push(Point2::new(x.x + u() - 0.5, x.y + u() - 0.5));
            }
        }
        for _ in 0..8 {
            det.push(Point2::new(1000.0 * u(), 800.0 * u()));
        }
        // Shuffle the detections.
        for i in (1..det.len()).rev() {
            let j = (u() * (i + 1) as f64) as usize % (i + 1);
            det.swap(i, j);
        }
        let mut refs = [CubeReference {
            point_id: 0,
            position: Point2::new(-1.0, -1.0),
            set: false,
        }; 4];
        for (s, id) in [0usize, 3, 12, 63].into_iter().enumerate().take(n_refs) {
            refs[s] = CubeReference {
                point_id: id as i32,
                position: proj[id],
                set: true,
            };
        }
        (obj, det, refs)
    }

    fn input<'a>(obj: &'a [Point3], det: &'a [Point2], refs: [CubeReference; 4]) -> CubeInput<'a> {
        CubeInput {
            object_points: obj,
            detected: det,
            references: refs,
            settings: CubeSettings::defaults(1024, 1024),
            compat: CubeCompat::XMALAB3,
            search_loops: None,
        }
    }

    /// The first `cases` synthetic cubes (the same draws whatever `cases` is), each searched
    /// for `loops` iterations.
    fn six_point_search_synthetic(cases: usize, loops: u32) {
        let mut draws_rng = CRand::new(7);
        let mut late = false;
        for case in 0..cases {
            let (obj, det, refs) = synthetic(&mut draws_rng, 3 + case % 2);
            let inp = input(&obj, &det, refs);
            late |= compare(&inp, 1 + case as u32, loops, &format!("synthetic {case}"));
        }
        assert!(late, "no acceptance on the last iteration of a batch");
    }

    /// Both numbers of references, fewer iterations: below the 1,024 at which the
    /// production entry goes parallel, which `review13_six_point_search_edges` (2,000) and
    /// `review13_six_point_search_reference_data` reach.
    #[test]
    fn review13_six_point_search_synthetic() {
        six_point_search_synthetic(2, 400);
    }

    #[test]
    #[ignore = "long comparison run; cargo test -- --ignored _long"]
    fn review13_six_point_search_synthetic_long() {
        six_point_search_synthetic(12, 3000);
    }

    /// Degenerate inputs: detected beads with NaN, infinities, signed zeros and
    /// subnormals, and duplicated beads.
    #[test]
    fn review13_six_point_search_edges() {
        let mut draws_rng = CRand::new(11);
        let (obj, mut det, refs) = synthetic(&mut draws_rng, 3);
        det.extend([
            Point2::new(f64::NAN, 1.0),
            Point2::new(f64::INFINITY, f64::NEG_INFINITY),
            Point2::new(-0.0, 0.0),
            Point2::new(f64::MIN_POSITIVE / 4.0, -f64::MIN_POSITIVE / 8.0),
        ]);
        let d0 = det[0];
        det.push(d0);
        compare(&input(&obj, &det, refs), 5, 2000, "edges");
        // Few beads: every draw nearly the same.
        let few: Vec<Point2> = det.iter().copied().take(3).collect();
        compare(&input(&obj, &few, refs), 6, 500, "three beads");
        // Zero iterations.
        compare(&input(&obj, &det, refs), 7, 0, "no iteration");
    }

    /// The reference calibration `c05_three_refs` camera 0 (three references) and
    /// `c02_cube_s1` camera 0 (four), shortened searches.
    #[test]
    fn review13_six_point_search_reference_data() {
        use xma_oracle::Table;
        for (run, loops) in [("c05_three_refs", 60_000), ("c02_cube_s1", 20_000)] {
            let dir = xma_oracle::oracle_root()
                .join("data/pig/calibration/runs")
                .join(run)
                .join("a");
            if !dir.join("hooks/cube_start.csv").is_file() {
                assert!(!xma_oracle::oracle_required(), "{run} missing");
                eprintln!("skipping: {run} not present");
                return;
            }
            let start = Table::read(dir.join("hooks/cube_start.csv"));
            let first = start.rows().next().unwrap();
            let line = first.i64("line");
            let mine = |t: &Table| -> Vec<usize> {
                t.rows()
                    .enumerate()
                    .filter(|(_, r)| r.i64("line") == line)
                    .map(|(i, _)| i)
                    .collect()
            };
            let objs = Table::read(dir.join("s2_undistorted/calibration_object_points.csv"));
            let obj: Vec<Point3> = objs
                .rows()
                .map(|r| Point3::new(r.f64("x"), r.f64("y"), r.f64("z")))
                .collect();
            let dt = Table::read(dir.join("hooks/cube_detected_undistorted.csv"));
            let rows: Vec<_> = dt.rows().collect();
            let det: Vec<Point2> = mine(&dt)
                .into_iter()
                .map(|i| Point2::new(rows[i].f64("x"), rows[i].f64("y")))
                .collect();
            let rt = Table::read(dir.join("hooks/cube_references.csv"));
            let rrows: Vec<_> = rt.rows().collect();
            let mut refs = [CubeReference {
                point_id: 0,
                position: Point2::origin(),
                set: false,
            }; 4];
            for i in mine(&rt) {
                let r = &rrows[i];
                let position = Point2::new(r.f64("x_undistorted"), r.f64("y_undistorted"));
                refs[r.usize("slot")] = CubeReference {
                    point_id: i32::try_from(r.i64("point_id")).unwrap(),
                    position,
                    set: position.x >= 0.0 && position.y >= 0.0,
                };
            }
            let inp = input(&obj, &det, refs);
            compare(&inp, 1, loops, run);
        }
    }
}
