//! Multi-camera optimisation of a calibrated camera set-up ("optimize camera setup").
//!
//! Work package: calibration (phase 4). Ported from `processing/MultiCameraCalibration.cpp`
//! (`MultiCameraCalibration`) and the generated expressions in
//! `processing/optimizationFuncs/` ([`funcs`] submodule), solved with
//! [`crate::levmar::dlevmar_der`].
//!
//! # What the C++ does, and this port with it
//!
//! - **Frames** ([`setup`]): every calibration frame in which at least one camera's image is
//!   calibrated (`calibrated == 1`), in order.
//! - **Observations**: for each camera, each listed frame where its image is calibrated
//!   (`== 1`), each calibration-object point whose inlier flag is positive: the detected
//!   point after the grid undistortion only (`UndistortionObject::transformPoint(p, true)`;
//!   the caller passes those points, [`MulticamImage::points`]).
//! - **Parameters**: per camera Rodrigues rotation, translation, fx, fy, cx, cy and, for
//!   [`Method::NormWithDistortion`], k1, k2, p1, p2, k3; then per listed frame the object's
//!   Rodrigues rotation and translation. Nothing is held fixed.
//! - **Initial values**: the reference frame is the first listed frame in which every
//!   camera is calibrated; each camera's pose is its image's pose there, its rotation
//!   multiplied on the right by `Rodrigues(0.01, 0.01, 0.01)`. Without such a frame the
//!   reference is the **last** listed frame in which camera 0 is calibrated, and cameras not
//!   calibrated there are placed through a frame they share with a camera already placed:
//!   `T_c1_ref = T_c1(f) T_c2(f)^-1 T_c2_ref`, perturbed the same way. Each frame's object
//!   pose is the mean, over the cameras calibrated in it, of the Rodrigues vector and the
//!   translation of `T_cam^-1 T_image`.
//! - **Residuals**: [`Method::Norm`] one residual per observation, the distance between the
//!   observed and the projected point; [`Method::Separate`] its x and y; and
//!   [`Method::NormWithDistortion`] the distance with OpenCV's 5-coefficient lens model.
//!   The target vector is zero. Analytic Jacobians.
//! - **Solver** ([`solve`]): `dlevmar_der` with `{initial, 1e-30, 1e-30, 1e-40}` and the
//!   dialog's iteration count.
//! - **Written back** ([`write_back`]): per camera fx, fy, cx, cy (the rest of the camera
//!   matrix kept), for method 2 the five coefficients, for methods 0 and 1 the lens model
//!   switched off; per listed frame and **every** camera the pose `T_cam T_frame`, and an
//!   image that was not calibrated becomes `calibrated = 2`.
//!
//! # Defects fixed ([`MulticamCompat`])
//!
//! Parity ledger A120 to A127. The default fixes them; [`MulticamCompat::xmalab3`] keeps
//! the C++ behaviour wherever it is defined. Where the C++ reads uninitialised memory or
//! indexes out of range the port never reproduces it: [`setup`] refuses with a
//! [`MulticamError`] in both modes, or (the six Jacobian entries the method-1 header never
//! assigns, A123) uses the correct value 0 in both modes.
//!
//! # Floating point
//!
//! Operation by operation as the C++: the `cv::Mat` products through OpenCV's small-matrix
//! `gemm` order, `t_cam.inv() * t_chess` as the `cv::solve(.., DECOMP_LU)` OpenCV evaluates
//! it to, `T_c2_a.inv()` as `cv::invert(.., DECOMP_LU)`, the generated expressions without
//! fused multiply-add and with the C library's `sqrt`, `sin`, `cos` and `pow`.
//! The solver itself: see [`crate::levmar`] for which parts of the reference build's
//! LAPACK and BLAS are reproduced; with more than 1023 Jacobian entries levmar's `J^T J`
//! goes through BLAS `dgemm`, which the port reproduces only for six parameters.

pub mod funcs;

use std::fmt;

use crate::camera::{RigidTransform, matrix_to_rodrigues, project_points_f32, rodrigues_to_matrix};
use crate::levmar::{Info, LevmarError, Options, StopReason, dlevmar_der};
use crate::linalg::{invert, solve_lu};
use crate::rigid::{divide_by_norm, gemm_small};
use crate::{Matrix3, Point2, Point3, Vector3};

pub use funcs::{
    jacobian_norm, jacobian_norm_distortion, jacobian_xy, residual_norm, residual_norm_distortion,
    residual_xy,
};

/// The three methods of the Optimization dialog (`ui/OptimizationDialog.cpp`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Method {
    /// 0, "Without Modeldistortion - norm".
    Norm,
    /// 1, "Without Modeldistortion - x and y seperate".
    Separate,
    /// 2, "With Modeldistortion - norm".
    NormWithDistortion,
}

impl Method {
    /// The method for the dialog's index; any other value is method 0, as the C++'s
    /// `default: case 0:` (`processing/MultiCameraCalibration.cpp:269-284`).
    #[must_use]
    pub fn from_index(index: i32) -> Method {
        match index {
            1 => Method::Separate,
            2 => Method::NormWithDistortion,
            _ => Method::Norm,
        }
    }

    /// The dialog's index.
    #[must_use]
    pub fn index(self) -> i32 {
        match self {
            Method::Norm => 0,
            Method::Separate => 1,
            Method::NormWithDistortion => 2,
        }
    }

    /// `withDistortion`.
    #[must_use]
    pub fn with_distortion(self) -> bool {
        self == Method::NormWithDistortion
    }

    /// `seperateDimensions`.
    #[must_use]
    pub fn separate_dimensions(self) -> bool {
        self == Method::Separate
    }

    /// `nbCamParams`: 15 with distortion, else 10.
    #[must_use]
    pub fn camera_params(self) -> usize {
        if self.with_distortion() { 15 } else { 10 }
    }

    /// `getStep()`: residuals per observation.
    #[must_use]
    pub fn step(self) -> usize {
        if self.separate_dimensions() { 2 } else { 1 }
    }
}

/// Which defects of XMALab 3's multi-camera optimisation to reproduce. Each field is
/// `true` for the C++ behaviour.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MulticamCompat {
    /// Ledger A121: the chain test is `f_to_use > 0` (`MultiCameraCalibration.cpp:474`),
    /// so frame 0 never links two cameras. Fixed: any frame found links them.
    pub chain_skips_frame_zero: bool,
    /// Ledger A122: the chain takes the placed camera's pose at the reference frame from
    /// its own image there (`:479`), which is not calibrated when that camera was itself
    /// placed through the chain. XMALab 3: that image's pose if it holds one (else
    /// [`MulticamError::UndefinedPose`]). Fixed: the pose the camera was placed with (before
    /// the perturbation), which for a camera calibrated at the reference frame is the same.
    pub chain_uses_reference_image_pose: bool,
    /// Ledger A123: the norm Jacobians (methods 0 and 2) multiply by a coordinate
    /// difference's sign computed as `d / fabs(d)`, which is 0/0 = NaN when that difference
    /// is exactly 0, so the whole row is NaN although the derivative exists (and levmar
    /// then never accepts a step). Fixed: the sign is 0 there, which gives the derivative.
    pub jacobian_nan_at_zero_difference: bool,
    /// Ledger A124: method 2 applies the new coefficients (which rebuilds the lens maps and
    /// undistorts the points) before setting the new camera matrix, so the maps and the
    /// undistorted points use the old camera matrix. Fixed: the new one.
    /// [`CameraUpdate::lens_maps_camera_matrix`] says which.
    pub lens_maps_with_previous_camera_matrix: bool,
    /// Ledger A125: method 2 optimises k1, k2, p1, p2, k3 and keeps the camera's k4, k5, k6,
    /// which the optimised model does not have. Fixed: k4 to k6 written as 0, so the stored
    /// model is the one optimised.
    pub keeps_rational_coefficients: bool,
    /// Ledger A126: the trials are not flagged for recomputation after the calibration
    /// changed. Fixed: [`MulticamResult::trials_need_recompute`] is true.
    pub trials_not_flagged: bool,
    /// Ledger A127: the parameters are written back whatever the solver did: when it
    /// refuses (fewer residuals than parameters) the initial values are written (and the
    /// lens model switched off for methods 0 and 1), and non-finite values are written as
    /// they are. Fixed: [`optimize`] returns [`MulticamError::SolverRefused`] or
    /// [`MulticamError::NotFinite`] and nothing is to be written. Also (fixed, same row): a
    /// non-finite input value read by the residuals ([`MulticamError::NonFiniteInput`]) and
    /// a run that accepted no step and stopped with one of levmar's errors (a singular
    /// matrix or invalid values, stops 4 and 7: [`MulticamError::NothingOptimised`]) are
    /// errors.
    pub failure_written_back: bool,
    /// Ledger A158: a camera, or a listed frame's object pose, with fewer residuals than
    /// the parameters it adds is optimised and written back, although its data do not
    /// determine it (levmar only compares the totals). Fixed: [`optimize`] returns
    /// [`MulticamError::CameraUnderdetermined`] or [`MulticamError::FrameUnderdetermined`]
    /// (see [`check_determined`] for the counting rule).
    pub underdetermined_written_back: bool,
}

impl MulticamCompat {
    /// Every defect fixed: the default.
    #[must_use]
    pub const fn fixed() -> MulticamCompat {
        MulticamCompat {
            chain_skips_frame_zero: false,
            chain_uses_reference_image_pose: false,
            jacobian_nan_at_zero_difference: false,
            lens_maps_with_previous_camera_matrix: false,
            keeps_rational_coefficients: false,
            trials_not_flagged: false,
            failure_written_back: false,
            underdetermined_written_back: false,
        }
    }

    /// Everything as XMALab 3 does it, where it is defined.
    #[must_use]
    pub const fn xmalab3() -> MulticamCompat {
        MulticamCompat {
            chain_skips_frame_zero: true,
            chain_uses_reference_image_pose: true,
            jacobian_nan_at_zero_difference: true,
            lens_maps_with_previous_camera_matrix: true,
            keeps_rational_coefficients: true,
            trials_not_flagged: true,
            failure_written_back: true,
            underdetermined_written_back: true,
        }
    }
}

/// What the optimisation reads of a camera.
#[derive(Clone, Debug, PartialEq)]
pub struct MulticamCamera {
    /// `Camera::getCameraMatrix()`; fx, fy, cx, cy are optimised, the rest is kept.
    pub camera_matrix: Matrix3,
    /// `Camera::getDistortionCoefficiants()`: k1, k2, p1, p2, k3, k4, k5, k6 (zeros when the
    /// camera never had a lens model).
    pub distortion: [f64; 8],
    /// `Camera::hasModelDistortion()`.
    pub model_distortion: bool,
}

/// What the optimisation reads of one calibration image (one camera, one frame).
#[derive(Clone, Debug, PartialEq)]
pub struct MulticamImage {
    /// `CalibrationImage::isCalibrated()`: 1 calibrated, 2 set by an earlier optimisation,
    /// 0 not, -1 failed.
    pub calibrated: i32,
    /// The pose the image holds (world to camera), or `None` when it holds none (the C++'s
    /// matrices created without values). Must be `Some` when `calibrated == 1`.
    pub pose: Option<RigidTransform>,
    /// `getInliers()`, one per calibration-object point: 1 inlier, 0 outlier, -1 not
    /// detected.
    pub inliers: Vec<i32>,
    /// `getDetectedPoints()` passed through the grid undistortion only
    /// (`UndistortionObject::transformPoint(p, true)` when the camera's grid is computed,
    /// else unchanged), one per calibration-object point. Only inliers are read.
    pub points: Vec<Point2>,
}

/// Everything [`setup`] needs.
#[derive(Clone, Copy, Debug)]
pub struct MulticamInput<'a> {
    /// One per camera, in project order.
    pub cameras: &'a [MulticamCamera],
    /// `images[camera][frame]`, every camera with the same number of calibration frames.
    pub images: &'a [Vec<MulticamImage>],
    /// The calibration object's points (`CalibrationObject::getFrameSpecifications()`).
    pub object_points: &'a [Point3],
    /// The dialog's method.
    pub method: Method,
    /// levmar's `itmax`: the dialog's "Number of iterations" (10000 from the simple prompt).
    pub iterations: usize,
    /// levmar's `opts[0]`: the dialog's "Scale factor for initial λ" (0.01).
    pub initial_mu: f64,
}

/// One observation: a detected inlier of a calibrated image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Observation {
    /// Camera index (`camIdx`).
    pub camera: usize,
    /// Index into [`MulticamSetup::frames`] (`frameIdx`), not the frame number.
    pub frame_index: usize,
    /// Calibration-object point index (`pts3DIdx`).
    pub point: usize,
    /// The grid-undistorted detected point (`Pts2D`).
    pub image: Point2,
}

/// The problem as the C++'s constructor leaves it before `dlevmar_der`.
#[derive(Clone, Debug, PartialEq)]
pub struct MulticamSetup {
    /// The method.
    pub method: Method,
    /// Number of cameras.
    pub n_cameras: usize,
    /// The listed frames (`frames`), frame numbers in order.
    pub frames: Vec<usize>,
    /// The reference frame (`refIdx`), a frame number.
    pub reference_frame: usize,
    /// The observations, camera by camera, frame by frame, point by point.
    pub observations: Vec<Observation>,
    /// The calibration object's points (`Pts3D`).
    pub object_points: Vec<Point3>,
    /// The initial parameter vector (`p`).
    pub params: Vec<f64>,
}

impl MulticamSetup {
    /// `nbParams`.
    #[must_use]
    pub fn n_params(&self) -> usize {
        self.method.camera_params() * self.n_cameras + 6 * self.frames.len()
    }

    /// `nbPoints` after `* getStep()`: the number of residuals.
    #[must_use]
    pub fn n_residuals(&self) -> usize {
        self.observations.len() * self.method.step()
    }

    /// Offset of camera `c`'s parameters in the vector.
    #[must_use]
    pub fn camera_offset(&self, c: usize) -> usize {
        self.method.camera_params() * c
    }

    /// Offset of the parameters of listed frame `f` (index into [`Self::frames`]).
    #[must_use]
    pub fn frame_offset(&self, f: usize) -> usize {
        6 * f + self.method.camera_params() * self.n_cameras
    }
}

/// Why the optimisation cannot run, or (fixed mode) why its result is not to be used.
#[derive(Clone, Debug, PartialEq)]
pub enum MulticamError {
    /// No camera at all.
    NoCameras,
    /// The cameras do not have the same number of calibration images (`images[c].len()`).
    FrameCountMismatch,
    /// No image is calibrated (`calibrated == 1`): nothing to optimise. (The C++ indexes
    /// the images with -1.)
    NoCalibratedImage,
    /// A camera could not be placed: it shares no calibrated frame with a placed camera
    /// (ledger A120; with XMALab 3's chain also A121, A122). The C++ reads an empty matrix
    /// for its parameters, or, when camera 0 is calibrated in no frame, indexes the images
    /// with -1. Both modes.
    CameraNotPlaced {
        /// The camera.
        camera: usize,
    },
    /// An image the set-up reads holds no pose: a calibrated image without one (bad
    /// input), or in the XMALab 3 mode the reference image of a camera placed through the
    /// chain (ledger A122; the C++ reads uninitialised values there).
    UndefinedPose {
        /// The camera.
        camera: usize,
        /// The frame.
        frame: usize,
    },
    /// A calibrated image has fewer inlier flags or points than the calibration object
    /// has points.
    ImageListsTooShort {
        /// The camera.
        camera: usize,
        /// The frame.
        frame: usize,
    },
    /// Fixed mode, ledger A127: the solver refused (fewer residuals than parameters).
    SolverRefused(LevmarError),
    /// Fixed mode, ledger A127: the solver's parameters are not finite.
    NotFinite,
    /// Fixed mode, ledger A127: a value the residuals read is NaN or infinite (an inlier
    /// detection, a calibration-object point or a starting parameter).
    NonFiniteInput {
        /// Which value.
        what: String,
    },
    /// Fixed mode, ledger A127: the solver accepted no step and stopped with an error (a
    /// singular matrix or invalid values), so its parameters are the start values.
    NothingOptimised {
        /// levmar's stop reason.
        stop: crate::levmar::StopReason,
    },
    /// Fixed mode, ledger A158: a camera has fewer residuals than its parameters.
    CameraUnderdetermined {
        /// The camera.
        camera: usize,
        /// Its residuals (inlier observations times residuals per observation).
        residuals: usize,
        /// Its parameters.
        parameters: usize,
    },
    /// Fixed mode, ledger A158: a listed frame's object pose has fewer residuals than its 6
    /// parameters.
    FrameUnderdetermined {
        /// The frame number.
        frame: usize,
        /// Its residuals.
        residuals: usize,
    },
}

impl fmt::Display for MulticamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MulticamError::NoCameras => write!(f, "there is no camera"),
            MulticamError::FrameCountMismatch => {
                write!(
                    f,
                    "the cameras have different numbers of calibration images"
                )
            }
            MulticamError::NoCalibratedImage => write!(f, "no calibration image is calibrated"),
            MulticamError::CameraNotPlaced { camera } => write!(
                f,
                "camera {} shares no calibrated frame with the other cameras",
                camera + 1
            ),
            MulticamError::UndefinedPose { camera, frame } => write!(
                f,
                "the calibration image of camera {} in frame {} holds no pose",
                camera + 1,
                frame + 1
            ),
            MulticamError::ImageListsTooShort { camera, frame } => write!(
                f,
                "the calibration image of camera {} in frame {} has too few points",
                camera + 1,
                frame + 1
            ),
            MulticamError::SolverRefused(e) => write!(f, "the optimisation refused: {e}"),
            MulticamError::NotFinite => write!(f, "the optimisation gave non-finite values"),
            MulticamError::NonFiniteInput { what } => {
                write!(f, "the optimisation cannot run: {what} is not finite")
            }
            MulticamError::NothingOptimised { stop } => write!(
                f,
                "the optimisation changed nothing: the solver stopped at the start values \
                 (levmar stop {})",
                stop.code()
            ),
            MulticamError::CameraUnderdetermined {
                camera,
                residuals,
                parameters,
            } => write!(
                f,
                "camera {} has too few inlier points for the optimisation: {residuals} \
                 residuals for its {parameters} parameters",
                camera + 1
            ),
            MulticamError::FrameUnderdetermined { frame, residuals } => write!(
                f,
                "frame {} has too few inlier points for the optimisation: {residuals} \
                 residuals for the 6 parameters of its pose",
                frame + 1
            ),
        }
    }
}

impl std::error::Error for MulticamError {}

type Mat4 = [[f64; 4]; 4];

/// `getTransformationMatrix`: `[R | t; 0 0 0 1]` with `R` by `cv::Rodrigues`.
///
/// Ported from `processing/MultiCameraCalibration.cpp:632-661`,
/// `MultiCameraCalibration::getTransformationMatrix`.
fn transformation_matrix(r: &Vector3, t: &Vector3) -> Mat4 {
    let rm = rodrigues_to_matrix(r);
    let mut m = [[0.0; 4]; 4];
    for (i, row) in m.iter_mut().enumerate().take(3) {
        for (j, v) in row.iter_mut().enumerate().take(3) {
            *v = rm[(i, j)];
        }
        row[3] = t[i];
    }
    m[3][3] = 1.0;
    m
}

/// `getRotationVector`: `cv::Rodrigues` of the upper-left 3x3.
///
/// Ported from `processing/MultiCameraCalibration.cpp:663-681`.
fn rotation_vector(m: &Mat4) -> Vector3 {
    matrix_to_rodrigues(&Matrix3::from_fn(|i, j| m[i][j]))
}

/// `getTranslation`: the last column's first three entries.
///
/// Ported from `processing/MultiCameraCalibration.cpp:683-694`.
fn translation(m: &Mat4) -> Vector3 {
    Vector3::new(m[0][3], m[1][3], m[2][3])
}

fn to_na(m: &Mat4) -> nalgebra::Matrix4<f64> {
    nalgebra::Matrix4::from_fn(|i, j| m[i][j])
}

fn from_dm(m: &nalgebra::DMatrix<f64>) -> Mat4 {
    let mut out = [[0.0; 4]; 4];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = m[(i, j)];
        }
    }
    out
}

/// `a.inv()` assigned to a `Mat`: `cv::invert(a, DECOMP_LU)`.
fn inv4(a: &Mat4) -> Mat4 {
    from_dm(&invert(&to_na(a)).x)
}

/// `a.inv() * b` for two `Mat`s: `MatOp_Invert::matmul` makes it `cv::solve(a, b,
/// DECOMP_LU)`.
fn inv_times(a: &Mat4, b: &Mat4) -> Mat4 {
    from_dm(&solve_lu(&to_na(a), &to_na(b)).x)
}

/// The camera rotation XMALab starts from: `Rodrigues(Rodrigues(r) * Rodrigues(0.01, 0.01,
/// 0.01))` ("to make sure rotationmatrices are not identity").
///
/// Ported from `processing/MultiCameraCalibration.cpp:374-381` (and `:421-428`, `:483-489`).
fn perturbed(r: &Vector3) -> Vector3 {
    let rot_mat = rodrigues_to_matrix(&Vector3::new(0.01, 0.01, 0.01));
    let rot_mat2 = rodrigues_to_matrix(r);
    let a = [
        [rot_mat2[(0, 0)], rot_mat2[(0, 1)], rot_mat2[(0, 2)]],
        [rot_mat2[(1, 0)], rot_mat2[(1, 1)], rot_mat2[(1, 2)]],
        [rot_mat2[(2, 0)], rot_mat2[(2, 1)], rot_mat2[(2, 2)]],
    ];
    let b = [
        [rot_mat[(0, 0)], rot_mat[(0, 1)], rot_mat[(0, 2)]],
        [rot_mat[(1, 0)], rot_mat[(1, 1)], rot_mat[(1, 2)]],
        [rot_mat[(2, 0)], rot_mat[(2, 1)], rot_mat[(2, 2)]],
    ];
    let p = gemm_small(&a, &b);
    matrix_to_rodrigues(&Matrix3::from_fn(|i, j| p[i][j]))
}

fn image_pose(
    input: &MulticamInput<'_>,
    c: usize,
    f: usize,
) -> Result<RigidTransform, MulticamError> {
    input.images[c][f].pose.ok_or(MulticamError::UndefinedPose {
        camera: c,
        frame: f,
    })
}

/// Builds the problem: listed frames, observations, reference frame and initial
/// parameters.
///
/// Ported from `processing/MultiCameraCalibration.cpp:264-624`, the constructor of
/// `MultiCameraCalibration`, with the defects of ledger A121 and A122 fixed unless
/// `compat` says otherwise, and refusing (both modes) where the C++ is undefined (A120).
pub fn setup(
    input: &MulticamInput<'_>,
    compat: MulticamCompat,
) -> Result<MulticamSetup, MulticamError> {
    let method = input.method;
    let n_cameras = input.cameras.len();
    if n_cameras == 0 || input.images.len() != n_cameras {
        return Err(if n_cameras == 0 {
            MulticamError::NoCameras
        } else {
            MulticamError::FrameCountMismatch
        });
    }
    let n_images = input.images[0].len();
    if input.images.iter().any(|v| v.len() != n_images) {
        return Err(MulticamError::FrameCountMismatch);
    }
    let cal = |c: usize, f: usize| input.images[c][f].calibrated == 1;
    let n_points = input.object_points.len();

    // :299-309
    let frames: Vec<usize> = (0..n_images)
        .filter(|&f| (0..n_cameras).any(|c| cal(c, f)))
        .collect();
    if frames.is_empty() {
        return Err(MulticamError::NoCalibratedImage);
    }

    // :314-339
    let mut observations = Vec::new();
    for c in 0..n_cameras {
        for (f_, &f) in frames.iter().enumerate() {
            if !cal(c, f) {
                continue;
            }
            let img = &input.images[c][f];
            if img.inliers.len() < n_points {
                return Err(MulticamError::ImageListsTooShort {
                    camera: c,
                    frame: f,
                });
            }
            for k in 0..n_points {
                if img.inliers[k] > 0 {
                    let image = *img.points.get(k).ok_or(MulticamError::ImageListsTooShort {
                        camera: c,
                        frame: f,
                    })?;
                    observations.push(Observation {
                        camera: c,
                        frame_index: f_,
                        point: k,
                        image,
                    });
                }
            }
        }
    }

    // :349-508, the camera poses. `placed[c]` keeps the unperturbed pose at the reference
    // frame (fixed chain, A122).
    let mut cam_rot = vec![Vector3::zeros(); n_cameras];
    let mut cam_trans = vec![Vector3::zeros(); n_cameras];
    let mut placed: Vec<Option<Mat4>> = vec![None; n_cameras];

    let full = frames
        .iter()
        .copied()
        .find(|&f| (0..n_cameras).all(|c| cal(c, f)));
    let reference_frame;
    if let Some(r) = full {
        reference_frame = r;
        for c in 0..n_cameras {
            let pose = image_pose(input, c, r)?;
            cam_rot[c] = perturbed(&pose.rotation);
            cam_trans[c] = pose.translation;
            placed[c] = Some(transformation_matrix(&pose.rotation, &pose.translation));
        }
    } else {
        // :397-404: the last listed frame where camera 0 is calibrated. Without one the
        // C++ indexes the images with -1 (A120): camera 0 has no calibrated image and can
        // never be placed, so both modes refuse.
        reference_frame = frames
            .iter()
            .copied()
            .rfind(|&f| cal(0, f))
            .ok_or(MulticamError::CameraNotPlaced { camera: 0 })?;
        let r = reference_frame;
        // :417-439
        for c in 0..n_cameras {
            if cal(c, r) {
                let pose = image_pose(input, c, r)?;
                cam_rot[c] = perturbed(&pose.rotation);
                cam_trans[c] = pose.translation;
                placed[c] = Some(transformation_matrix(&pose.rotation, &pose.translation));
            }
        }
        // :447-507
        let count_missing = |placed: &[Option<Mat4>]| placed.iter().filter(|p| p.is_none()).count();
        let mut missing_new = count_missing(&placed);
        loop {
            let missing = missing_new;
            for c in 0..n_cameras {
                if placed[c].is_some() {
                    continue;
                }
                let mut found: Option<(usize, usize)> = None;
                'search: for &f in &frames {
                    for (c2, pl) in placed.iter().enumerate() {
                        if pl.is_some() && cal(c, f) && cal(c2, f) {
                            found = Some((c2, f));
                            break 'search;
                        }
                    }
                }
                let Some((c2, f)) = found else { continue };
                if compat.chain_skips_frame_zero && f == 0 {
                    continue;
                }
                let p2a = image_pose(input, c2, f)?;
                let p1a = image_pose(input, c, f)?;
                let t_c2_a = transformation_matrix(&p2a.rotation, &p2a.translation);
                let t_c1_a = transformation_matrix(&p1a.rotation, &p1a.translation);
                let t_c2_ref = if compat.chain_uses_reference_image_pose {
                    let p = image_pose(input, c2, r)?;
                    transformation_matrix(&p.rotation, &p.translation)
                } else {
                    placed[c2].expect("c2 is placed")
                };
                let t_c1_ref = gemm_small(&gemm_small(&t_c1_a, &inv4(&t_c2_a)), &t_c2_ref);
                cam_rot[c] = perturbed(&rotation_vector(&t_c1_ref));
                cam_trans[c] = translation(&t_c1_ref);
                placed[c] = Some(t_c1_ref);
            }
            missing_new = count_missing(&placed);
            if !(missing_new > 0 && missing != missing_new) {
                break;
            }
        }
        if let Some(c) = placed.iter().position(Option::is_none) {
            return Err(MulticamError::CameraNotPlaced { camera: c });
        }
    }

    // :510-531, the object pose per frame.
    let mut frame_poses = Vec::with_capacity(frames.len());
    for &f in &frames {
        let mut r_vec = [0.0f64; 3];
        let mut t_vec = [0.0f64; 3];
        let mut count = 0usize;
        for c in 0..n_cameras {
            if cal(c, f) {
                let t_cam = transformation_matrix(&cam_rot[c], &cam_trans[c]);
                let pose = image_pose(input, c, f)?;
                let t_chess = transformation_matrix(&pose.rotation, &pose.translation);
                let rel = inv_times(&t_cam, &t_chess);
                let rv = rotation_vector(&rel);
                let tv = translation(&rel);
                for i in 0..3 {
                    r_vec[i] += rv[i];
                    t_vec[i] += tv[i];
                }
                count += 1;
            }
        }
        // `r_vec / count`: MatOp_AddEx with alpha = 1./count.
        divide_by_norm(&mut r_vec, count as f64);
        divide_by_norm(&mut t_vec, count as f64);
        frame_poses.push((r_vec, t_vec));
    }

    // :555-595, the parameter vector.
    let ncp = method.camera_params();
    let mut params = vec![0.0; ncp * n_cameras + 6 * frames.len()];
    for c in 0..n_cameras {
        let o = ncp * c;
        let k = &input.cameras[c].camera_matrix;
        params[o] = cam_rot[c][0];
        params[o + 1] = cam_rot[c][1];
        params[o + 2] = cam_rot[c][2];
        params[o + 3] = cam_trans[c][0];
        params[o + 4] = cam_trans[c][1];
        params[o + 5] = cam_trans[c][2];
        params[o + 6] = k[(0, 0)];
        params[o + 7] = k[(1, 1)];
        params[o + 8] = k[(0, 2)];
        params[o + 9] = k[(1, 2)];
        if method.with_distortion() {
            params[o + 10..o + 15].copy_from_slice(&input.cameras[c].distortion[..5]);
        }
    }
    for (f_, (r_vec, t_vec)) in frame_poses.iter().enumerate() {
        let o = 6 * f_ + ncp * n_cameras;
        params[o..o + 3].copy_from_slice(r_vec);
        params[o + 3..o + 6].copy_from_slice(t_vec);
    }

    Ok(MulticamSetup {
        method,
        n_cameras,
        frames,
        reference_frame,
        observations,
        object_points: input.object_points.to_vec(),
        params,
    })
}

fn slice6(p: &[f64]) -> &[f64; 6] {
    p[..6].try_into().expect("6 values")
}

/// The residual vector at `p` (`ros`): `observations x step` values.
///
/// Ported from `processing/MultiCameraCalibration.cpp:56-64` (`ros`) and `:198-261`
/// (`MultiCameraCalibration::projError`).
pub fn residuals(s: &MulticamSetup, p: &[f64], hx: &mut [f64]) {
    for (i, o) in s.observations.iter().enumerate() {
        let co = s.camera_offset(o.camera);
        let frame = slice6(&p[s.frame_offset(o.frame_index)..]);
        let x = s.object_points[o.point];
        match s.method {
            Method::Norm => {
                let cam: &[f64; 10] = p[co..co + 10].try_into().expect("10 values");
                hx[i] = residual_norm(cam, frame, x, o.image);
            }
            Method::Separate => {
                let cam: &[f64; 10] = p[co..co + 10].try_into().expect("10 values");
                let r = residual_xy(cam, frame, x, o.image);
                hx[2 * i] = r[0];
                hx[2 * i + 1] = r[1];
            }
            Method::NormWithDistortion => {
                let cam: &[f64; 15] = p[co..co + 15].try_into().expect("15 values");
                hx[i] = residual_norm_distortion(cam, frame, x, o.image);
            }
        }
    }
}

/// The Jacobian at `p` (`jacros`), row-major, `residuals x params`.
///
/// Ported from `processing/MultiCameraCalibration.cpp:66-74` (`jacros`) and `:76-191`
/// (`MultiCameraCalibration::projErrorJac`). The six entries the method-1 header leaves
/// unassigned are 0 in both modes; the sign of a zero coordinate difference follows
/// [`MulticamCompat::jacobian_nan_at_zero_difference`] (both ledger A123).
pub fn jacobian(s: &MulticamSetup, p: &[f64], jac: &mut [f64], compat: MulticamCompat) {
    let nan_at_zero = compat.jacobian_nan_at_zero_difference;
    let m = s.n_params();
    let ncp = s.method.camera_params();
    for (i, o) in s.observations.iter().enumerate() {
        let co = s.camera_offset(o.camera);
        let fo = s.frame_offset(o.frame_index);
        let frame = slice6(&p[fo..]);
        let x = s.object_points[o.point];
        match s.method {
            Method::Norm | Method::NormWithDistortion => {
                let row = &mut jac[i * m..(i + 1) * m];
                row.iter_mut().for_each(|v| *v = 0.0);
                let a0: Vec<f64> = if s.method == Method::Norm {
                    let cam: &[f64; 10] = p[co..co + 10].try_into().expect("10 values");
                    jacobian_norm(cam, frame, x, o.image, nan_at_zero).to_vec()
                } else {
                    let cam: &[f64; 15] = p[co..co + 15].try_into().expect("15 values");
                    jacobian_norm_distortion(cam, frame, x, o.image, nan_at_zero).to_vec()
                };
                row[co..co + ncp].copy_from_slice(&a0[..ncp]);
                row[fo..fo + 6].copy_from_slice(&a0[ncp..ncp + 6]);
            }
            Method::Separate => {
                let rows = &mut jac[2 * i * m..2 * (i + 1) * m];
                rows.iter_mut().for_each(|v| *v = 0.0);
                let cam: &[f64; 10] = p[co..co + 10].try_into().expect("10 values");
                let a0 = jacobian_xy(cam, frame, x, o.image);
                for (r, a) in a0.iter().enumerate() {
                    rows[m * r + co..m * r + co + 10].copy_from_slice(&a[..10]);
                    rows[m * r + fo..m * r + fo + 6].copy_from_slice(&a[10..16]);
                }
            }
        }
    }
}

/// levmar's options as the C++ sets them (`:794-799`): `{initial, 1e-30, 1e-30, 1e-40}`
/// (`opts[4]`, `LM_DIFF_DELTA`, is not read by `dlevmar_der`).
#[must_use]
pub fn solver_options(initial_mu: f64) -> Options {
    Options {
        mu: initial_mu,
        eps1: 1e-30,
        eps2: 1e-30,
        eps3: 1e-40,
    }
}

/// Runs the solver from `s.params` and returns the final parameters with levmar's result.
/// `observe` is called after every function evaluation with the residual vector (the
/// reference build's `multicam_evals` hook).
///
/// Ported from `processing/MultiCameraCalibration.cpp:792-821`,
/// `MultiCameraCalibration::optimizeCameraSetup_thread`: `dlevmar_der(ros, jacros, p, x,
/// nbParams, nbPoints, iterations, opts, info, NULL, NULL, this)` with `x` zero. When levmar
/// refuses, `p` is returned unchanged with the error.
pub fn solve(
    s: &MulticamSetup,
    iterations: usize,
    initial_mu: f64,
    compat: MulticamCompat,
    mut observe: impl FnMut(&[f64]),
) -> (Vec<f64>, Result<Info, LevmarError>) {
    let n = s.n_residuals();
    let x = vec![0.0f64; n];
    let mut p = s.params.clone();
    let func = |p: &[f64], hx: &mut [f64]| {
        residuals(s, p, hx);
        observe(hx);
    };
    let jacf = |p: &[f64], jac: &mut [f64]| jacobian(s, p, jac, compat);
    let info = dlevmar_der(
        func,
        jacf,
        &mut p,
        Some(&x),
        n,
        iterations,
        Some(&solver_options(initial_mu)),
    );
    (p, info)
}

/// What is written to a camera.
#[derive(Clone, Debug, PartialEq)]
pub struct CameraUpdate {
    /// The new camera matrix (`setCameraMatrix`): fx, fy, cx, cy from the solution, the
    /// other entries (skew included) as they were.
    pub camera_matrix: Matrix3,
    /// Method 2: the new coefficients (`setDistortionCoefficiants`: lens model on).
    /// Methods 0 and 1: `None`, the lens model is switched off (`resetDistortion`: if it was
    /// on, coefficients set to zero and the model off; if it was off, nothing changes).
    pub distortion: Option<[f64; 8]>,
    /// Method 2: the camera matrix with which the lens maps are built and the detected and
    /// projected points undistorted when the coefficients are applied (ledger A124: the old
    /// one in XMALab 3, the new one fixed). `None` for methods 0 and 1.
    pub lens_maps_camera_matrix: Option<Matrix3>,
}

/// What is written to one calibration image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageUpdate {
    /// Camera.
    pub camera: usize,
    /// Frame number.
    pub frame: usize,
    /// The new pose (`setMatrices`), world to camera.
    pub pose: RigidTransform,
    /// The new `calibrated` value: 1 if it was 1, else 2.
    pub calibrated: i32,
}

/// The result of the optimisation, for the caller to write back.
#[derive(Clone, Debug, PartialEq)]
pub struct MulticamResult {
    /// The problem as set up (frames, reference frame, observations, initial parameters).
    pub setup: MulticamSetup,
    /// The final parameter vector.
    pub params: Vec<f64>,
    /// levmar's information, `None` when it refused (XMALab 3 mode only; the C++ then
    /// writes the initial parameters back).
    pub info: Option<Info>,
    /// One per camera.
    pub cameras: Vec<CameraUpdate>,
    /// Every listed frame, every camera, frame by frame.
    pub images: Vec<ImageUpdate>,
    /// Whether the trials must be flagged for recomputation (ledger A126: false in XMALab 3).
    pub trials_need_recompute: bool,
}

/// Turns a parameter vector into what is written to the cameras and images.
///
/// Ported from `processing/MultiCameraCalibration.cpp:706-779`,
/// `MultiCameraCalibration::optimizeCameraSetup_threadFinished` (the products `t_cam *
/// t_chess` in OpenCV's small-matrix `gemm` order), with ledger A124 to A126 fixed unless
/// `compat` says otherwise.
#[must_use]
pub fn write_back(
    input: &MulticamInput<'_>,
    s: &MulticamSetup,
    p: &[f64],
    info: Option<Info>,
    compat: MulticamCompat,
) -> MulticamResult {
    let ncp = s.method.camera_params();
    let mut cameras = Vec::with_capacity(s.n_cameras);
    let mut cam_t = Vec::with_capacity(s.n_cameras);
    for c in 0..s.n_cameras {
        let o = ncp * c;
        let old = input.cameras[c].camera_matrix;
        let mut k = old;
        k[(0, 0)] = p[o + 6];
        k[(1, 1)] = p[o + 7];
        k[(0, 2)] = p[o + 8];
        k[(1, 2)] = p[o + 9];
        let (distortion, lens_maps_camera_matrix) = if s.method.with_distortion() {
            let mut d = input.cameras[c].distortion;
            d[..5].copy_from_slice(&p[o + 10..o + 15]);
            if !compat.keeps_rational_coefficients {
                d[5..].iter_mut().for_each(|v| *v = 0.0);
            }
            let maps = if compat.lens_maps_with_previous_camera_matrix {
                old
            } else {
                k
            };
            (Some(d), Some(maps))
        } else {
            (None, None)
        };
        cameras.push(CameraUpdate {
            camera_matrix: k,
            distortion,
            lens_maps_camera_matrix,
        });
        cam_t.push(transformation_matrix(
            &Vector3::new(p[o], p[o + 1], p[o + 2]),
            &Vector3::new(p[o + 3], p[o + 4], p[o + 5]),
        ));
    }

    let mut images = Vec::with_capacity(s.frames.len() * s.n_cameras);
    for (f_, &f) in s.frames.iter().enumerate() {
        let o = s.frame_offset(f_);
        let t_chess = transformation_matrix(
            &Vector3::new(p[o], p[o + 1], p[o + 2]),
            &Vector3::new(p[o + 3], p[o + 4], p[o + 5]),
        );
        for (c, t_cam) in cam_t.iter().enumerate() {
            let m = gemm_small(t_cam, &t_chess);
            images.push(ImageUpdate {
                camera: c,
                frame: f,
                pose: RigidTransform {
                    rotation: rotation_vector(&m),
                    translation: translation(&m),
                },
                calibrated: if input.images[c][f].calibrated == 1 {
                    1
                } else {
                    2
                },
            });
        }
    }

    MulticamResult {
        setup: s.clone(),
        params: p.to_vec(),
        info,
        cameras,
        images,
        trials_need_recompute: !compat.trials_not_flagged,
    }
}

/// The whole optimisation: [`setup`], [`solve`], [`write_back`].
///
/// Ported from `processing/MultiCameraCalibration.cpp` (constructor,
/// `optimizeCameraSetup_thread`, `optimizeCameraSetup_threadFinished`). Fixed (ledger
/// A127): a refusal of the solver or non-finite parameters give an error and nothing is
/// to be written; XMALab 3 writes them back.
pub fn optimize(
    input: &MulticamInput<'_>,
    compat: MulticamCompat,
) -> Result<MulticamResult, MulticamError> {
    optimize_observed(input, compat, |_| {})
}

/// [`optimize`], calling `observe` with the residual vector after each function
/// evaluation.
pub fn optimize_observed(
    input: &MulticamInput<'_>,
    compat: MulticamCompat,
    observe: impl FnMut(&[f64]),
) -> Result<MulticamResult, MulticamError> {
    let s = setup(input, compat)?;
    if !compat.failure_written_back {
        check_finite_input(&s)?;
    }
    // A problem with fewer residuals than parameters in all is levmar's refusal (A127);
    // otherwise each camera and frame is checked (A158).
    if !compat.underdetermined_written_back && s.n_residuals() >= s.n_params() {
        check_determined(&s)?;
    }
    let (p, info) = solve(&s, input.iterations, input.initial_mu, compat, observe);
    let info = match info {
        Ok(i) => Some(i),
        Err(e) if !compat.failure_written_back => return Err(MulticamError::SolverRefused(e)),
        Err(_) => None,
    };
    if !compat.failure_written_back {
        if p.iter().any(|v| !v.is_finite()) {
            return Err(MulticamError::NotFinite);
        }
        if let Some(i) = &info
            && p == s.params
            && matches!(
                i.stop,
                StopReason::SingularMatrix | StopReason::InvalidValues
            )
        {
            return Err(MulticamError::NothingOptimised { stop: i.stop });
        }
    }
    Ok(write_back(input, &s, &p, info, compat))
}

/// Ledger A127, fixed: every value the residuals read is finite: the inlier detections, the
/// calibration-object points they use, and the starting parameters.
///
/// # Errors
/// [`MulticamError::NonFiniteInput`] naming the first value that is not.
pub fn check_finite_input(s: &MulticamSetup) -> Result<(), MulticamError> {
    for o in &s.observations {
        if !(o.image.x.is_finite() && o.image.y.is_finite()) {
            return Err(MulticamError::NonFiniteInput {
                what: format!(
                    "the detection of point {} by camera {} in frame {}",
                    o.point + 1,
                    o.camera + 1,
                    s.frames[o.frame_index] + 1
                ),
            });
        }
        let q = s.object_points[o.point];
        if !(q.x.is_finite() && q.y.is_finite() && q.z.is_finite()) {
            return Err(MulticamError::NonFiniteInput {
                what: format!("calibration-object point {}", o.point + 1),
            });
        }
    }
    if let Some(i) = s.params.iter().position(|v| !v.is_finite()) {
        return Err(MulticamError::NonFiniteInput {
            what: format!("starting parameter {i}"),
        });
    }
    Ok(())
}

/// Ledger A158, fixed: every camera and every listed frame's object pose is determined by
/// enough residuals. The counting rule: a camera's residuals are its observations (inlier
/// points of its calibrated images in the listed frames) times the residuals per
/// observation ([`Method::step`]: 2 for method 1, else 1), and must be at least its
/// parameters ([`Method::camera_params`]: 10, or 15 with distortion); a listed frame's
/// residuals are the observations of every camera in that frame times the step, and must
/// be at least 6. A count equal to the parameters is accepted.
///
/// # Errors
/// [`MulticamError::CameraUnderdetermined`] for the first camera short of residuals, else
/// [`MulticamError::FrameUnderdetermined`] for the first frame.
pub fn check_determined(s: &MulticamSetup) -> Result<(), MulticamError> {
    let step = s.method.step();
    let mut per_camera = vec![0usize; s.n_cameras];
    let mut per_frame = vec![0usize; s.frames.len()];
    for o in &s.observations {
        per_camera[o.camera] += step;
        per_frame[o.frame_index] += step;
    }
    let ncp = s.method.camera_params();
    if let Some(c) = per_camera.iter().position(|&n| n < ncp) {
        return Err(MulticamError::CameraUnderdetermined {
            camera: c,
            residuals: per_camera[c],
            parameters: ncp,
        });
    }
    if let Some(f) = per_frame.iter().position(|&n| n < 6) {
        return Err(MulticamError::FrameUnderdetermined {
            frame: s.frames[f],
            residuals: per_frame[f],
        });
    }
    Ok(())
}

/// The projections stored for an image after the optimisation: every calibration-object
/// point through the image's pose and the camera matrix, without lens distortion, in
/// single precision (`cv::Point3f` in, `cv::Point2f` out), widened back to `f64`.
///
/// Ported from `processing/MultiCameraCalibration.cpp:824-880`,
/// `MultiCameraCalibration::reproject`, for one image; it is
/// [`crate::camera::project_points_f32`] with no coefficients. The caller applies it to
/// every image of the camera with `calibrated > 0` (cube) or with detected points
/// (checkerboard), and stores the result as the undistorted projections.
#[must_use]
pub fn reproject(
    object_points: &[Point3],
    pose: &RigidTransform,
    camera_matrix: &Matrix3,
) -> Vec<Point2> {
    let pts: Vec<nalgebra::Point3<f32>> = object_points
        .iter()
        .map(|p| nalgebra::Point3::new(p.x as f32, p.y as f32, p.z as f32))
        .collect();
    project_points_f32(&pts, &pose.rotation, &pose.translation, camera_matrix, &[])
        .into_iter()
        .map(|q| Point2::new(f64::from(q.x), f64::from(q.y)))
        .collect()
}
