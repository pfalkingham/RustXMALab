//! The undistortion lookup table and grid points: part of File > Export > Export All.
//!
//! Ported from core/UndistortionObject.cpp, `UndistortionObject::exportData` (`:888-921`),
//! and core/Project.cpp, `Project::exportLUT` (`:572-583`). Export All also writes the DLT
//! coefficients and MayaCam version 1, which are not ported.

use std::sync::Arc;

use xma_model::{Camera, UndistortionGrid};
use xma_vision::undistortion::LookupTable;

use super::{ExportFile, complete_base_name, g};
use crate::cfmt::opencv_csv_f32;

/// The `_LUT.csv` text: `cv::format(mapX, FMT_CSV)`, `std::endl`, `cv::format(mapY,
/// FMT_CSV)`, `std::endl`.
///
/// Ported from core/UndistortionObject.cpp:890-894. OpenCV's CSV formatter writes `%.8g` per
/// `float`, `", "` between values and `\n` after each row of a matrix with more than one
/// row, ignoring the stream's precision ([`opencv_csv_f32`]); each map is followed by
/// `std::endl`, so by an empty line. `table` is the map built from the grid's distorting
/// model (`LocalUndistortion::createLookupTable`, [`xma_vision::undistortion::create_lookup_table`]
/// of `GridModel::distort` at the grid image's size).
#[must_use]
pub fn lut_file(table: &LookupTable) -> String {
    let mut out = opencv_csv_f32(&table.map_x, table.height, table.width);
    out.push('\n');
    out.push_str(&opencv_csv_f32(&table.map_y, table.height, table.width));
    out.push('\n');
    out
}

/// `x,y` per line of the given points where the grid's inlier flag is set, in order.
fn inlier_points(points: &[xma_model::Point2], inlier: &[bool]) -> String {
    let mut out = String::new();
    for (count, &is_inlier) in inlier.iter().enumerate() {
        if is_inlier {
            let p = points[count];
            out.push_str(&g(p.x));
            out.push(',');
            out.push_str(&g(p.y));
            out.push('\n');
        }
    }
    out
}

/// The `_INPTS.csv` text: the distorted grid points (`points_grid_distorted`) of the inliers.
///
/// Ported from core/UndistortionObject.cpp:896-906.
#[must_use]
pub fn inlier_points_file(grid: &UndistortionGrid) -> String {
    inlier_points(&grid.grid_distorted, &grid.grid_inlier)
}

/// The `_BSPTS.csv` text: the reference grid points (`points_grid_references`) of the inliers.
///
/// Ported from core/UndistortionObject.cpp:908-918.
#[must_use]
pub fn base_points_file(grid: &UndistortionGrid) -> String {
    inlier_points(&grid.grid_references, &grid.grid_inlier)
}

/// The undistortion files of Export All: for each camera, in order, whose grid undistortion
/// is computed, `<grid image base name>_LUT.csv`, `_INPTS.csv` and `_BSPTS.csv`.
///
/// Ported from core/Project.cpp, Project::exportLUT. `tables[c]` is camera `c`'s lookup table
/// (see [`lut_file`]); a camera with a computed grid but no table (the engine could not fit
/// the grid) is skipped, where the C++ would write whatever maps it holds.
#[must_use]
pub fn export_lut(cameras: &[Arc<Camera>], tables: &[Option<LookupTable>]) -> Vec<ExportFile> {
    let mut files = Vec::new();
    for (c, cam) in cameras.iter().enumerate() {
        let Some(grid) = cam.undistortion.as_ref().filter(|g| g.computed) else {
            continue;
        };
        let Some(Some(table)) = tables.get(c) else {
            continue;
        };
        let base = complete_base_name(&grid.image_filename);
        files.push(ExportFile::new(format!("{base}_LUT.csv"), lut_file(table)));
        files.push(ExportFile::new(
            format!("{base}_INPTS.csv"),
            inlier_points_file(grid),
        ));
        files.push(ExportFile::new(
            format!("{base}_BSPTS.csv"),
            base_points_file(grid),
        ));
    }
    files
}
