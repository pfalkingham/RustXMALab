//! 2D points: File > Export 2D Points.
//!
//! Ported from core/Trial.cpp, `Trial::save2dPoints` (`:1708-1915`), with the options
//! `ui/MainWindow.cpp`, `MainWindow::on_actionExport2D_Points_triggered` (`:1808-1854`)
//! passes from the export dialog's settings.

use std::sync::Arc;

use xma_model::{Camera, CppCompat, Trial};
use xma_vision::camera::CameraModel;

use super::{ExportError, ExportFile, ExportOutput, check_trial, g, marker_export_name, padded};
use crate::cfmt::MISSING_NAN;

/// Options of the 2D export. The default is what the GUI passes by default
/// (`core/Settings.cpp:176-187`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Points2dOptions {
    /// One file for all markers (`Export2DMulti`, default) or one file per marker
    /// (`Export2DSingle`).
    pub one_file: bool,
    /// Raw image coordinates (`Export2DDistorted`, default) or undistorted with the grid
    /// and lens model (`Export2DUndistorted`).
    pub distorted: bool,
    /// Add 1 to x and y (`Export2DCount1`); default off, counting from 0.
    pub offset1: bool,
    /// Write `height - y - 1` (`Export2DYUp`); default off, y down.
    pub y_invert: bool,
    /// Header row (`Export2DHeader`, default on).
    pub header: bool,
    /// Offset columns (`Export2DOffsetCols`), default off.
    pub offset_columns: bool,
    /// Which faults of XMALab 3 to reproduce: [`CppCompat::offset_header`] (ledger F2: with
    /// offset columns, the header's offset fields `cam<j>_offset` numbered from 0 with no
    /// separators; fixed, `cam<j+1>_offset,` per camera, one field per data column; data
    /// rows are the same either way) and [`CppCompat::export_details`] (ledger F25: file
    /// names with an empty description, line ends without markers). Default: fixed.
    pub compat: CppCompat,
    /// Only this camera (the C++ `id`; the GUI's 2D export passes -1, every camera; "All
    /// Trials for External" passes one camera).
    pub camera: Option<usize>,
}

impl Default for Points2dOptions {
    fn default() -> Self {
        Self {
            one_file: true,
            distorted: true,
            offset1: false,
            y_invert: false,
            header: true,
            offset_columns: false,
            compat: CppCompat::FIXED,
            camera: None,
        }
    }
}

/// The cameras written: all, or the one asked for.
fn camera_range(opts: &Points2dOptions, n_cameras: usize) -> std::ops::Range<usize> {
    match opts.camera {
        Some(c) => c..c + 1,
        None => 0..n_cameras,
    }
}

/// The offset columns' header fields.
///
/// Ported from core/Trial.cpp:1719-1725 (and `:1823-1829`): with `cpp`, `cam<j>_offset` for
/// each camera, numbered from 0, with no separator at all (ledger F2). Fixed (the default):
/// `cam<j+1>_offset,` for each camera, numbered from 1 as the marker columns are, each
/// followed by the comma that separates it from the next field.
fn offset_header(out: &mut String, cams: std::ops::Range<usize>, cpp: bool) {
    for j in cams {
        out.push_str("cam");
        out.push_str(&(if cpp { j } else { j + 1 }).to_string());
        out.push_str("_offset");
        if !cpp {
            out.push(',');
        }
    }
}

/// `0,` for each camera at the start of a data row (`outfile << 0.0 << ","`).
fn offset_values(out: &mut String, cams: std::ops::Range<usize>) {
    for _ in cams {
        out.push_str(&g(0.0));
        out.push(',');
    }
}

/// `<name>_cam<j+1>_X,<name>_cam<j+1>_Y`.
fn header_pair(out: &mut String, name: &str, j: usize) {
    let cam = (j + 1).to_string();
    out.push_str(name);
    out.push_str("_cam");
    out.push_str(&cam);
    out.push_str("_X,");
    out.push_str(name);
    out.push_str("_cam");
    out.push_str(&cam);
    out.push_str("_Y");
}

/// What every value of a 2D export reads.
#[derive(Clone, Copy)]
struct Ctx<'a> {
    trial: &'a Trial,
    cameras: &'a [Arc<Camera>],
    models: &'a [CameraModel],
    opts: &'a Points2dOptions,
}

/// `x,y` of one marker in one camera at one frame, or `NaN,NaN`.
///
/// Ported from core/Trial.cpp:1768-1799: only for `status2D > 0`; the raw point, or
/// `Camera::undistortPoint(pt, true)` (grid then lens, with refinement); then
/// `y = height - y - 1` if inverted; then `+1` to both if `offset1`. All in `double`.
/// An undistorted point that has no result (ledger A19, fixed mode) is written `NaN,NaN`,
/// as a missing point.
fn value_pair(out: &mut String, ctx: &Ctx<'_>, marker: usize, j: usize, frame: usize) {
    let Ctx {
        trial,
        cameras,
        models,
        opts,
    } = *ctx;
    let view = &trial.markers[marker].views[j];
    // Ledger A19 fixed: a point whose lens undistortion has no result is written as a
    // missing point (`NaN,NaN`), not as a formatted NaN.
    let point = (view.status[frame].code() > 0)
        .then(|| {
            let raw = view.points[frame];
            if opts.distorted {
                Some(raw)
            } else {
                models[j].try_undistort_point(raw, true, true, true)
            }
        })
        .flatten();
    if let Some(p) = point {
        let (mut x, mut y) = (p.x, p.y);
        if opts.y_invert {
            // `getHeight() - y - 1`: int converted to double, then left to right.
            y = f64::from(cameras[j].height) - y - 1.0;
        }
        if opts.offset1 {
            y += 1.0;
            x += 1.0;
        }
        out.push_str(&g(x));
        out.push(',');
        out.push_str(&g(y));
    } else {
        out.push_str(MISSING_NAN);
        out.push(',');
        out.push_str(MISSING_NAN);
    }
}

/// The one-file 2D export: every marker, marker-major, each camera's `X,Y`.
///
/// Ported from core/Trial.cpp, Trial::save2dPoints, the `onefile` branch (`:1713-1813`).
/// `opts.one_file` is not read. `cameras` are the project's cameras (their count and image
/// heights are used); `models` are their fitted models, read only when
/// `opts.distorted` is false (`Camera::undistortPoint(pt, true)`).
///
/// Every frame `0..n_frames` is written, with no frame column.
///
/// # Panics
/// When the input fails the checks of [`points_2d`], which callers run first. With no markers the header
/// has no line end and each row is only its offset columns, if any, with no line end.
#[must_use]
pub fn points_2d_file(
    trial: &Trial,
    cameras: &[Arc<Camera>],
    models: &[CameraModel],
    opts: &Points2dOptions,
) -> String {
    let ctx = Ctx {
        trial,
        cameras,
        models,
        opts,
    };
    let cams = camera_range(opts, cameras.len());
    let end = cams.end;
    let n_markers = trial.markers.len();
    let mut out = String::new();
    if n_markers == 0 && !opts.compat.export_details {
        // Ledger F25 fixed: the offset columns, if any, each line ended.
        let cols = if opts.offset_columns { cams.len() } else { 0 };
        if cols == 0 {
            return out;
        }
        if opts.header {
            let mut h = String::new();
            offset_header(&mut h, cams.clone(), false);
            h.pop();
            out.push_str(&h);
            out.push('\n');
        }
        for _ in 0..trial.n_frames {
            let mut row = String::new();
            offset_values(&mut row, cams.clone());
            row.pop();
            out.push_str(&row);
            out.push('\n');
        }
        return out;
    }
    if opts.header {
        if opts.offset_columns {
            offset_header(&mut out, cams.clone(), opts.compat.offset_header);
        }
        for (i, m) in trial.markers.iter().enumerate() {
            let name = marker_export_name(&m.description, i);
            for j in cams.clone() {
                header_pair(&mut out, &name, j);
                if i != n_markers - 1 || j != end - 1 {
                    out.push(',');
                } else {
                    out.push('\n');
                }
            }
        }
    }
    for f in 0..trial.n_frames {
        if opts.offset_columns {
            offset_values(&mut out, cams.clone());
        }
        for i in 0..n_markers {
            for j in cams.clone() {
                value_pair(&mut out, &ctx, i, j, f);
                if i != n_markers - 1 || j != end - 1 {
                    out.push(',');
                } else {
                    out.push('\n');
                }
            }
        }
    }
    out
}

/// The 2D export of one marker, as written in the one-file-per-marker mode.
///
/// Ported from core/Trial.cpp, Trial::save2dPoints, the per-marker branch (`:1816-1913`).
#[must_use]
pub fn points_2d_marker_file(
    trial: &Trial,
    marker: usize,
    cameras: &[Arc<Camera>],
    models: &[CameraModel],
    opts: &Points2dOptions,
) -> String {
    let ctx = Ctx {
        trial,
        cameras,
        models,
        opts,
    };
    let cams = camera_range(opts, cameras.len());
    let end = cams.end;
    let mut out = String::new();
    if opts.header {
        if opts.offset_columns {
            offset_header(&mut out, cams.clone(), opts.compat.offset_header);
        }
        let name = marker_export_name(&trial.markers[marker].description, marker);
        for j in cams.clone() {
            header_pair(&mut out, &name, j);
            out.push(if j != end - 1 { ',' } else { '\n' });
        }
    }
    for f in 0..trial.n_frames {
        if opts.offset_columns {
            offset_values(&mut out, cams.clone());
        }
        for j in cams.clone() {
            value_pair(&mut out, &ctx, marker, j, f);
            out.push(if j != end - 1 { ',' } else { '\n' });
        }
    }
    out
}

/// The file name of one marker's 2D export: `Marker%03d_<description>_points2d.csv`.
///
/// Ported from core/Trial.cpp:1818.
#[must_use]
pub fn points_2d_marker_file_name(trial: &Trial, marker: usize, compat: CppCompat) -> String {
    let d = &trial.markers[marker].description;
    format!(
        "Marker{}_{}_points2d.csv",
        padded(marker),
        super::file_description(d, marker_export_name(d, marker), compat.export_details)
    )
}

/// The 2D export as the GUI writes it: one file, or one file per marker.
///
/// Ported from core/Trial.cpp, Trial::save2dPoints.
///
/// # Errors
/// When the camera asked for does not exist, a model is missing for an undistorted export,
/// or a marker holds fewer frames or cameras than the trial (undefined behaviour in the
/// C++); nothing is written.
pub fn points_2d(
    trial: &Trial,
    cameras: &[Arc<Camera>],
    models: &[CameraModel],
    opts: &Points2dOptions,
) -> Result<ExportOutput, ExportError> {
    if let Some(c) = opts.camera.filter(|&c| c >= cameras.len()) {
        return Err(ExportError::NoSuchCamera(c));
    }
    if !opts.distorted && models.len() < cameras.len() {
        return Err(ExportError::Inconsistent(format!(
            "{} camera models for {} cameras",
            models.len(),
            cameras.len()
        )));
    }
    check_trial(trial, Some(cameras.len()))?;
    Ok(if opts.one_file {
        ExportOutput::File(points_2d_file(trial, cameras, models, opts))
    } else {
        ExportOutput::Folder(
            (0..trial.markers.len())
                .map(|i| {
                    ExportFile::new(
                        points_2d_marker_file_name(trial, i, opts.compat),
                        points_2d_marker_file(trial, i, cameras, models, opts),
                    )
                })
                .collect(),
        )
    })
}
