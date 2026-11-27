//! Image operations and geometry as plain functions: undistortion,
//! calibration, detection, tracking, rigid bodies, filtering and the solver.
//! Functions take points, matrices and images, and return the same.
//!
//! Dependency rule: must NOT depend on `xma-model` (checked by
//! `cargo xtask depcheck`). Never touches the GUI or global state.
//!
//! Status: each module is one phase 1 work package, ported from the C++ function its
//! documentation names and tested against the reference outputs (`oracle/FORMAT.md`). A
//! module whose documentation says "not yet ported" is still a placeholder.

pub mod beads;
pub mod calib3d;
pub mod camera;
pub mod chessboard;
pub mod cmath;
pub mod crand;
pub mod cube;
pub mod denoise;
pub mod depth;
pub mod detection;
pub mod dot_fit;
pub mod epipolar;
pub mod epnp;
pub mod filter;
pub mod grid_match;
pub mod image;
pub mod imgproc;
pub mod joint_tracking;
pub mod levmar;
pub mod linalg;
pub mod multicam;
pub mod pose_refine;
pub mod remap;
pub mod rigid;
pub mod tracking;
pub mod triangulation;
pub mod undistortion;
pub mod visual_filter;

/// Name of this crate, used to check the workspace builds and links.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");

/// A point in an image, in pixels.
pub type Point2 = nalgebra::Point2<f64>;
/// A point in 3D.
pub type Point3 = nalgebra::Point3<f64>;
/// A 3D vector: a translation, or a Rodrigues rotation vector.
pub type Vector3 = nalgebra::Vector3<f64>;
/// A 3x3 matrix.
pub type Matrix3 = nalgebra::Matrix3<f64>;
/// A 3x4 projection matrix.
pub type Matrix3x4 = nalgebra::Matrix3x4<f64>;

#[cfg(test)]
mod tests {
    #[test]
    fn crate_name() {
        assert_eq!(super::CRATE_NAME, "xma-vision");
    }
}
