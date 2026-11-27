//! Reprojection errors: File > Export > Reprojection Errors.
//!
//! Ported from core/Trial.cpp, `Trial::saveReprojectionErrors` (`:2018-2053`); the GUI
//! (`ui/MainWindow.cpp:1989-1998`) asks only for a folder.

use super::{ExportError, ExportFile, check_trial, g, padded};
use crate::cfmt::MISSING_NAN;
use xma_model::{CppCompat, Trial};

/// One marker's reprojection errors: one line per frame `0..n_frames`, one value per camera
/// (`n_cameras`, the project's camera count), no header. A value is written when the 2D
/// status is above 0 **and** the 3D status is not 0 (negative statuses count), else `NaN`.
///
/// Ported from core/Trial.cpp:2027-2049. With no cameras every line is empty and no line end
/// is written.
///
/// # Panics
/// When the input fails the checks of [`reprojection_errors`], which callers run first.
#[must_use]
pub fn reprojection_error_file(trial: &Trial, marker: usize, n_cameras: usize) -> String {
    let m = &trial.markers[marker];
    let mut out = String::new();
    for f in 0..trial.n_frames {
        for j in 0..n_cameras {
            if m.views[j].status[f].code() > 0 && m.status3d[f].code() != 0 {
                out.push_str(&g(m.views[j].error[f]));
            } else {
                out.push_str(MISSING_NAN);
            }
            out.push(if j != n_cameras - 1 { ',' } else { '\n' });
        }
    }
    out
}

/// The file name of one marker's reprojection errors:
/// `Marker%03d_<description>_reprojectionError.csv`.
///
/// Ported from core/Trial.cpp:2023.
///
/// Fixed (ledger F25), an empty description is replaced by `marker%03d`.
#[must_use]
pub fn reprojection_error_file_name(trial: &Trial, marker: usize, compat: CppCompat) -> String {
    let d = &trial.markers[marker].description;
    format!(
        "Marker{}_{}_reprojectionError.csv",
        padded(marker),
        super::file_description(
            d,
            super::marker_export_name(d, marker),
            compat.export_details
        )
    )
}

/// One file per marker, in marker order.
///
/// Ported from core/Trial.cpp, Trial::saveReprojectionErrors.
///
/// # Errors
/// When a marker holds fewer frames or cameras than the trial (undefined behaviour in the
/// C++); nothing is written.
pub fn reprojection_errors(
    trial: &Trial,
    n_cameras: usize,
    compat: CppCompat,
) -> Result<Vec<ExportFile>, ExportError> {
    check_trial(trial, Some(n_cameras))?;
    Ok((0..trial.markers.len())
        .map(|i| {
            ExportFile::new(
                reprojection_error_file_name(trial, i, compat),
                reprojection_error_file(trial, i, n_cameras),
            )
        })
        .collect())
}
