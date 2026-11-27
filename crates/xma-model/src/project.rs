//! The project: cameras, calibration object, trials and result-affecting settings.
//!
//! Ported from `core/Project.h` and `core/CalibrationObject.h`. In the C++ both are global
//! singletons; here the project is an ordinary value owned by the document.

use std::sync::Arc;

use crate::Point3;
use crate::camera::Camera;
use crate::settings::AlgorithmSettings;
use crate::status::CalibrationType;
use crate::trial::Trial;

/// The object used to calibrate the cameras.
#[derive(Clone, Debug, PartialEq)]
pub enum CalibrationObject {
    /// A planar checkerboard (`CalibrationObject::setCheckerboard`). Its points are the
    /// inner corners, [`checkerboard_points`]; it has no reference points.
    Checkerboard {
        /// Inner corners per row ("Nb squares horizontal"; `cv::findChessboardCorners`'s
        /// pattern width).
        horizontal_squares: u32,
        /// Inner corners per column ("Nb squares vertical").
        vertical_squares: u32,
        square_size: f64,
        /// `CheckerboadInvertedAxis` when the board was built (created or opened): 1 negates
        /// x, 2 negates y, anything else neither. The C++ reads the setting in
        /// `setCheckerboard`; changing the setting later does not change the board.
        inverted_axis: i32,
    },
    /// A 3D calibration frame ("cube") of beads at known positions.
    Frame(CalibrationFrame),
}

/// The largest number of corners a checkerboard may have here (1000 x 1000). The New
/// dataset dialog allows 99 x 99; a project file can give any count, and a board beyond this
/// is refused when the file is read (ledger F74).
pub const MAX_CHECKERBOARD_POINTS: u64 = 1_000_000;

/// The largest number of frames a trial (`nbImages`), or a camera's calibration sequence
/// (`Frames`), may have here. A project file can give any count, and the per-frame arrays
/// are allocated from it, so a project beyond this is refused when the file is read
/// (ledger F84).
pub const MAX_FRAMES: u64 = 10_000_000;

/// The largest product of per-frame array counts one trial may have here: cameras x
/// frames x markers, and on their own frames x events and frames x virtual points (the
/// rigid bodies have [`MAX_TRIAL_BODY_CELLS`]). The frames are counted as the arrays reach
/// them, rows of the data files beyond the trial's frame count included. A project beyond
/// this is refused when the file is read (ledger F84).
pub const MAX_TRIAL_CELLS: u64 = 250_000_000;

/// The largest product frames x rigid bodies one trial may have here. A rigid body holds
/// about 162 bytes per frame (two pose tracks and the errors), so this is about 8 GB, where
/// [`MAX_TRIAL_CELLS`] body cells would be about 40 GB. A project beyond this is refused when
/// the file is read (ledger F84).
pub const MAX_TRIAL_BODY_CELLS: u64 = 50_000_000;

/// The points of a checkerboard: row `y` outer, column `x` inner, at
/// `(inv_x · x · size, inv_y · y · size, 0)` with `inv_x` -1 when `inverted_axis` is 1 and
/// `inv_y` -1 when it is 2. Empty for a board of more than [`MAX_CHECKERBOARD_POINTS`].
///
/// Ported from core/CalibrationObject.cpp, CalibrationObject::setCheckerboard.
#[must_use]
pub fn checkerboard_points(
    horizontal: u32,
    vertical: u32,
    square_size: f64,
    inverted_axis: i32,
) -> Vec<Point3> {
    if u64::from(horizontal) * u64::from(vertical) > MAX_CHECKERBOARD_POINTS {
        return Vec::new();
    }
    let inv_x: i64 = if inverted_axis == 1 { -1 } else { 1 };
    let inv_y: i64 = if inverted_axis == 2 { -1 } else { 1 };
    let mut out = Vec::with_capacity((horizontal * vertical) as usize);
    for y in 0..vertical {
        for x in 0..horizontal {
            // `inv_x * x * squareSize`: an integer product (at most 1e6 here), then double.
            #[allow(clippy::cast_precision_loss)]
            let (px, py) = ((inv_x * i64::from(x)) as f64, (inv_y * i64::from(y)) as f64);
            out.push(Point3::new(px * square_size, py * square_size, 0.0));
        }
    }
    out
}

impl CalibrationObject {
    /// The object's points in their order (`CalibrationObject::getFrameSpecifications`).
    #[must_use]
    pub fn points(&self) -> Vec<Point3> {
        match self {
            CalibrationObject::Frame(f) => f.points.clone(),
            CalibrationObject::Checkerboard {
                horizontal_squares,
                vertical_squares,
                square_size,
                inverted_axis,
            } => checkerboard_points(
                *horizontal_squares,
                *vertical_squares,
                *square_size,
                *inverted_axis,
            ),
        }
    }

    /// The number of points, without building them.
    #[must_use]
    pub fn n_points(&self) -> usize {
        match self {
            CalibrationObject::Frame(f) => f.points.len(),
            CalibrationObject::Checkerboard {
                horizontal_squares,
                vertical_squares,
                ..
            } => {
                let n = u64::from(*horizontal_squares) * u64::from(*vertical_squares);
                if n > MAX_CHECKERBOARD_POINTS {
                    0
                } else {
                    n as usize
                }
            }
        }
    }

    /// The name of each point (`CalibrationObject::getMarkerNames`): a frame's names, or
    /// "x , y" for a checkerboard's corner (`setCheckerboard`).
    #[must_use]
    pub fn point_names(&self) -> Vec<String> {
        match self {
            CalibrationObject::Frame(f) => f.point_names.clone(),
            CalibrationObject::Checkerboard {
                horizontal_squares,
                vertical_squares,
                ..
            } => {
                if self.n_points() == 0 {
                    return Vec::new();
                }
                let mut v = Vec::with_capacity(self.n_points());
                for y in 0..*vertical_squares {
                    for x in 0..*horizontal_squares {
                        v.push(format!("{x} , {y}"));
                    }
                }
                v
            }
        }
    }

    /// True for a checkerboard (`isCheckerboard`).
    #[must_use]
    pub fn is_checkerboard(&self) -> bool {
        matches!(self, CalibrationObject::Checkerboard { .. })
    }
}

/// A 3D calibration frame, read from a frame-specification file and a reference file.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationFrame {
    /// Bead positions, in file order (C++ `frameSpecifications`).
    pub points: Vec<Point3>,
    /// Name of each bead, where the file gives one.
    pub point_names: Vec<String>,
    /// Indices into `points` of the beads the user clicks to start a calibration: the
    /// 1-based ids of the `.ref` file minus one, as the C++ stores them (`int`). An id of 0
    /// in the file gives -1, which the C++ keeps.
    pub reference_ids: Vec<i32>,
    pub reference_names: Vec<String>,
    /// The scale factors of a frame-specification header (`s2,s3,s0.5,...`) that
    /// `xma_io::calibration_object::load_coords` multiplied into the positions. A project
    /// holds the scaled positions, and the project file stores them under a plain header
    /// (as the C++'s `saveCoords` writes them), so in a created or opened project this is
    /// (1, 1, 1): nothing is left to apply (ledger F47).
    pub scale: Point3,
    /// Beads appear white on dark, not dark on light.
    pub white_blobs: bool,
    pub frame_specification_filename: String,
    pub references_filename: String,
}

/// A project.
///
/// Cameras and trials are behind `Arc` for cheap snapshots; mutate with `Arc::make_mut`.
#[derive(Clone, Debug, PartialEq)]
pub struct Project {
    pub cameras: Vec<Arc<Camera>>,
    pub trials: Vec<Arc<Trial>>,
    pub calibration_type: CalibrationType,
    pub calibration_object: Option<CalibrationObject>,
    /// Settings that change computed results. XMALab 3 keeps these per machine; here they
    /// travel with the project (parity ledger, item A10).
    pub algorithm_settings: AlgorithmSettings,
    /// Creation date as the C++ wrote it.
    pub date_created: String,
    /// Index of the trial that was active when the project was saved.
    pub active_trial: usize,
    /// XMA Portal study metadata, kept as raw XML lines so it round-trips.
    pub portal_metadata: Vec<String>,
}

impl Project {
    /// An empty project.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cameras: Vec::new(),
            trials: Vec::new(),
            calibration_type: CalibrationType::None,
            calibration_object: None,
            algorithm_settings: AlgorithmSettings::default(),
            date_created: String::new(),
            active_trial: 0,
            portal_metadata: Vec::new(),
        }
    }
}

impl Default for Project {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkerboard_points_rows_outer_and_inverted_axes() {
        let p = checkerboard_points(3, 2, 2.5, 0);
        assert_eq!(p.len(), 6);
        assert_eq!(p[1], Point3::new(2.5, 0.0, 0.0));
        assert_eq!(p[3], Point3::new(0.0, 2.5, 0.0));
        assert_eq!(p[5], Point3::new(5.0, 2.5, 0.0));
        let x = checkerboard_points(3, 2, 2.5, 1);
        assert_eq!(x[5], Point3::new(-5.0, 2.5, 0.0));
        let y = checkerboard_points(3, 2, 2.5, 2);
        assert_eq!(y[5], Point3::new(5.0, -2.5, 0.0));
        // Neither 1 nor 2: no axis inverted.
        assert_eq!(checkerboard_points(3, 2, 2.5, 3), p);
        let board = CalibrationObject::Checkerboard {
            horizontal_squares: 3,
            vertical_squares: 2,
            square_size: 2.5,
            inverted_axis: 0,
        };
        assert_eq!(board.points(), p);
        assert_eq!(board.n_points(), 6);
        assert_eq!(board.point_names()[5], "2 , 1");
        assert!(board.is_checkerboard());
        let huge = CalibrationObject::Checkerboard {
            horizontal_squares: u32::MAX,
            vertical_squares: u32::MAX,
            square_size: 1.0,
            inverted_axis: 0,
        };
        assert!(huge.points().is_empty());
        assert_eq!(huge.n_points(), 0);
        assert!(huge.point_names().is_empty());
    }

    #[test]
    fn cloning_a_project_shares_cameras_and_trials() {
        let mut project = Project::new();
        project
            .cameras
            .push(Arc::new(Camera::new("Camera 1".to_owned())));
        project.trials.push(Arc::new(Trial::default_trial()));
        let snapshot = project.clone();
        assert!(Arc::ptr_eq(&project.cameras[0], &snapshot.cameras[0]));
        assert!(Arc::ptr_eq(&project.trials[0], &snapshot.trials[0]));
        Arc::make_mut(&mut project.cameras[0]).calibrated = true;
        assert!(!snapshot.cameras[0].calibrated);
    }
}
