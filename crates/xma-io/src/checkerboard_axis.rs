//! Ledger F75: the inverted axis of a checkerboard read from a project file that does not
//! store it (XMALab's own projects; this program writes it, ledger F73), recognised from
//! the stored calibration.
//!
//! XMALab builds the board with the setting `CheckerboadInvertedAxis` in force when the
//! project is opened. A project calibrated with another setting then has its board mirrored
//! against its stored poses. Each stored pose was computed for the board's own axis, so of
//! the three boards (no axis, x, y inverted) the right one is the one whose points, through
//! each calibrated image's stored pose and its camera's matrix (and lens model), land on that
//! image's stored detected points; a mirrored board lands tens to hundreds of pixels away.
//!
//! The rule: the mean distance over every inlier of every calibrated image is computed for
//! each axis; the best is taken when it is below 5 px and below a fifth of the second best.
//! Otherwise (no calibrated image, or no clear winner) the setting stays. When the axis taken
//! differs from the setting, a warning says so.

use xma_model::project::checkerboard_points;
use xma_model::{CalibrationObject, Point2};
use xma_vision::camera::project_points;

use crate::project_read::LoadedProject;

/// The mean reprojection distance (pixels) of the board with `axis` over the inliers of the
/// calibrated images, and the number of points; `None` without any.
fn mean_error(loaded: &LoadedProject, h: u32, v: u32, size: f64, axis: i32) -> Option<f64> {
    let pts = checkerboard_points(h, v, size, axis);
    if pts.is_empty() {
        return None;
    }
    let mut sum = 0.0;
    let mut n = 0usize;
    for cam in &loaded.project.cameras {
        let dist: Vec<f64> = if cam.has_model_distortion {
            cam.distortion.to_vec()
        } else {
            Vec::new()
        };
        for im in &cam.calibration_images {
            if im.calibrated <= 0 || im.detected.len() != pts.len() {
                continue;
            }
            let proj = project_points(
                &pts,
                &im.rotation,
                &im.translation,
                &cam.camera_matrix,
                &dist,
            );
            for ((p, d), inl) in proj.iter().zip(&im.detected).zip(&im.inlier) {
                if *inl <= 0 || !p.x.is_finite() || !p.y.is_finite() {
                    continue;
                }
                let e: Point2 = *d;
                sum += ((p.x - e.x).powi(2) + (p.y - e.y).powi(2)).sqrt();
                n += 1;
            }
        }
    }
    #[allow(clippy::cast_precision_loss)]
    (n > 0).then(|| sum / n as f64)
}

/// The axis the stored calibration fits, by the rule of the module documentation; `None`
/// when it cannot be told.
#[must_use]
pub fn fitted_axis(loaded: &LoadedProject) -> Option<i32> {
    let Some(CalibrationObject::Checkerboard {
        horizontal_squares,
        vertical_squares,
        square_size,
        ..
    }) = loaded.project.calibration_object
    else {
        return None;
    };
    let mut errs: Vec<(f64, i32)> = (0..3)
        .filter_map(|a| {
            mean_error(loaded, horizontal_squares, vertical_squares, square_size, a).map(|e| (e, a))
        })
        .filter(|(e, _)| e.is_finite())
        .collect();
    if errs.len() < 3 {
        return None;
    }
    errs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (best, axis) = errs[0];
    (best < 5.0 && best * 5.0 < errs[1].0).then_some(axis)
}

/// Sets the board's axis to the one its stored calibration fits, with a warning when that
/// is not the setting's. Nothing changes when it cannot be told.
pub fn recognise(loaded: &mut LoadedProject) {
    let Some(axis) = fitted_axis(loaded) else {
        return;
    };
    let normalised = |a: i32| if a == 1 || a == 2 { a } else { 0 };
    if let Some(CalibrationObject::Checkerboard { inverted_axis, .. }) =
        &mut loaded.project.calibration_object
        && normalised(*inverted_axis) != axis
    {
        loaded.warnings.push(format!(
            "The checkerboard's inverted axis is taken as {} (the stored calibration fits it), \
             not as the setting CheckerboadInvertedAxis ({}): the file does not store it",
            axis_name(axis),
            axis_name(normalised(*inverted_axis))
        ));
        *inverted_axis = axis;
    }
}

fn axis_name(a: i32) -> &'static str {
    match a {
        1 => "x",
        2 => "y",
        _ => "none",
    }
}
