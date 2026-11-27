//! 3D points: File > Export 3D Points.
//!
//! Ported from core/Trial.cpp, `Trial::save3dPoints` (`:1529-1706`), with the options
//! `ui/MainWindow.cpp`, `MainWindow::save3DPoints` (`:1723-1796`) passes, and the filtering of
//! core/Marker.cpp, `Marker::filterMarker` and `Marker::filterData`.

use std::fmt;

use xma_model::{CppCompat, Marker, MarkerStatus, Point3, Trial};
use xma_vision::filter::{
    ButterworthLowPass, CutoffCorrection, FILTER_ORDER, can_filter, filter_runs, runs,
};

use super::{
    ExportError, ExportFile, ExportOutput, FrameRange, check_markers, check_range, check_trial,
    export_frames, g, marker_export_name, padded,
};
use crate::cfmt::{MISSING_NAN, format_qt_number};

/// A marker's 3D points and 3D statuses, one entry per frame.
pub type MarkerSeries = (Vec<Point3>, Vec<MarkerStatus>);

/// Options of the 3D export. The default is what the GUI passes by default: one file,
/// header, no frame column, unfiltered (`core/Settings.cpp:189-194`; the "filter?" question
/// answered no).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Points3dOptions {
    /// One file for all markers (`Export3DMulti`, default) or one per marker
    /// (`Export3DSingle`).
    pub one_file: bool,
    /// Header row (`Export3DHeader`, default on).
    pub header: bool,
    /// Low-pass cutoff in hertz; the data are filtered only when it is above 0 (the C++
    /// `filterFrequency`, -1 when the user declines filtering). The GUI's dialog offers the
    /// trial's cutoff, rounded to one decimal (`QInputDialog::getDouble`'s default).
    pub filter_frequency: f64,
    /// A `Frame` column (1-based) and only these frames (`Export3DOffsetCols`, default off,
    /// which writes every frame and no frame column).
    pub frames: Option<FrameRange>,
    /// Which faults of XMALab 3 to reproduce: [`CppCompat::export_details`] (ledger F25:
    /// file names with an empty description, line ends without markers) and
    /// [`CppCompat::filter_cutoff_single_pass`] (ledger A7: the filter's cutoff is not
    /// corrected for the two passes). Default: fixed.
    pub compat: CppCompat,
}

impl Default for Points3dOptions {
    fn default() -> Self {
        Self {
            one_file: true,
            header: true,
            filter_frequency: -1.0,
            frames: None,
            compat: CppCompat::FIXED,
        }
    }
}

/// `Marker::filterMarker` refused the cutoff for a marker: the C++ returns false before
/// writing anything and the GUI shows an error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilterRefused {
    /// Index of the first marker refused.
    pub marker: usize,
    pub cutoff: f64,
    pub recording_speed: f64,
}

impl fmt::Display for FilterRefused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cannot filter marker {} at {} Hz with a recording speed of {} Hz",
            self.marker + 1,
            self.cutoff,
            self.recording_speed
        )
    }
}

impl std::error::Error for FilterRefused {}

/// A marker's 3D points filtered at `cutoff`: the points, and the status of each frame.
///
/// Ported from core/Marker.cpp, Marker::filterMarker (`:1116-1162`) and Marker::filterData
/// (`:1515-1569`). Every frame starts as (-1000, -1000, -1000) and `UNDEFINED`; refused
/// (returns `None`) unless [`can_filter`] with the trial's recording speed; otherwise the
/// runs of consecutive frames `0..n_frames` with `status3D > 0` are filtered (runs of 12 or
/// fewer copied, ledger A8) and keep their input status. The filter is designed with
/// `correction` (ledger A7: [`CutoffCorrection::SinglePass`] is XMALab 3,
/// [`super::cutoff_correction`] gives the one for a compatibility mode); the refusal test is
/// made on the entered cutoff either way.
#[must_use]
pub fn filter_marker(
    marker: &Marker,
    cutoff: f64,
    recording_speed: f64,
    n_frames: usize,
    correction: CutoffCorrection,
) -> Option<MarkerSeries> {
    let len = marker.status3d.len();
    let mut points = vec![Point3::new(-1000.0, -1000.0, -1000.0); len];
    let mut status = vec![MarkerStatus::Undefined; len];
    if !can_filter(cutoff, recording_speed) {
        return None;
    }
    let valid: Vec<bool> = marker.status3d[..n_frames]
        .iter()
        .map(|s| s.code() > 0)
        .collect();
    let runs = runs(&valid);
    let filter = ButterworthLowPass::zero_phase(FILTER_ORDER, cutoff, recording_speed, correction);
    let mut out = [vec![-1000.0; len], vec![-1000.0; len], vec![-1000.0; len]];
    for (k, o) in out.iter_mut().enumerate() {
        let values: Vec<f64> = marker.points3d.iter().map(|p| p[k]).collect();
        filter_runs(&filter, &values, &runs, o);
    }
    for run in &runs {
        for f in run.clone() {
            points[f] = Point3::new(out[0][f], out[1][f], out[2][f]);
            status[f] = marker.status3d[f];
        }
    }
    Some((points, status))
}

/// The data a 3D export writes for each marker: copied, or filtered.
///
/// Ported from core/Trial.cpp:1531-1556. Checks the input first (see [`check_input`]).
fn export_data(
    trial: &Trial,
    markers: &[usize],
    opts: &Points3dOptions,
) -> Result<Vec<MarkerSeries>, ExportError> {
    check_input(trial, markers, opts)?;
    let filter_frequency = opts.filter_frequency;
    let correction = super::cutoff_correction(opts.compat);
    let data: Result<Vec<MarkerSeries>, FilterRefused> = markers
        .iter()
        .map(|&i| {
            let m = &trial.markers[i];
            if filter_frequency > 0.0 {
                filter_marker(
                    m,
                    filter_frequency,
                    trial.recording_speed,
                    trial.n_frames,
                    correction,
                )
                .ok_or(FilterRefused {
                    marker: i,
                    cutoff: filter_frequency,
                    recording_speed: trial.recording_speed,
                })
            } else {
                let n = trial.n_frames;
                Ok((m.points3d[..n].to_vec(), m.status3d[..n].to_vec()))
            }
        })
        .collect();
    Ok(data?)
}

/// What the C++ would read out of bounds on: a marker that does not exist, a marker with
/// fewer 3D frames than the trial, a frame range past the trial.
fn check_input(
    trial: &Trial,
    markers: &[usize],
    opts: &Points3dOptions,
) -> Result<(), ExportError> {
    check_markers(trial, markers)?;
    check_trial(trial, None)?;
    check_range(opts.frames, trial.n_frames)
}

/// The header name of a marker: description or `marker%03d`, with `_<cutoff>Hz` when
/// filtered (`QString::number`, 6 significant digits).
fn header_name(trial: &Trial, marker: usize, filter_frequency: f64) -> String {
    let mut name = marker_export_name(&trial.markers[marker].description, marker);
    if filter_frequency > 0.0 {
        name.push('_');
        name.push_str(&format_qt_number(filter_frequency));
        name.push_str("Hz");
    }
    name
}

fn header_triple(out: &mut String, name: &str) {
    for (k, axis) in ["_X", "_Y", "_Z"].iter().enumerate() {
        if k > 0 {
            out.push(',');
        }
        out.push_str(name);
        out.push_str(axis);
    }
}

fn value_triple(out: &mut String, points: &[Point3], status: &[MarkerStatus], f: usize) {
    if status[f].code() <= 0 {
        out.push_str(MISSING_NAN);
        out.push(',');
        out.push_str(MISSING_NAN);
        out.push(',');
        out.push_str(MISSING_NAN);
    } else {
        let p = points[f];
        out.push_str(&g(p.x));
        out.push(',');
        out.push_str(&g(p.y));
        out.push(',');
        out.push_str(&g(p.z));
    }
}

/// The one-file 3D export of the given markers, in the order given.
///
/// Ported from core/Trial.cpp, Trial::save3dPoints, the `onefile` branch (`:1559-1631`).
/// `opts.one_file` is not read.
///
/// # Errors
/// When filtering is asked for and refused for a marker, a marker does not exist or holds
/// fewer frames than the trial, or the frame range goes past the trial (undefined behaviour
/// in the C++). Nothing is written.
pub fn points_3d_file(
    trial: &Trial,
    markers: &[usize],
    opts: &Points3dOptions,
) -> Result<String, ExportError> {
    let data = export_data(trial, markers, opts)?;
    if markers.is_empty() && !opts.compat.export_details {
        return Ok(super::empty_items_text(
            opts.header,
            opts.frames
                .map(|_| export_frames(opts.frames, trial.n_frames)),
        ));
    }
    let mut out = String::new();
    if opts.header {
        if opts.frames.is_some() {
            out.push_str("Frame,");
        }
        for (k, &i) in markers.iter().enumerate() {
            header_triple(&mut out, &header_name(trial, i, opts.filter_frequency));
            out.push(if k != markers.len() - 1 { ',' } else { '\n' });
        }
    }
    for f in export_frames(opts.frames, trial.n_frames) {
        if opts.frames.is_some() {
            out.push_str(&(f + 1).to_string());
            out.push(',');
        }
        for (k, (points, status)) in data.iter().enumerate() {
            value_triple(&mut out, points, status, f);
            out.push(if k != data.len() - 1 { ',' } else { '\n' });
        }
    }
    Ok(out)
}

/// The 3D export of one marker, as written in the one-file-per-marker mode.
///
/// Ported from core/Trial.cpp, Trial::save3dPoints, the per-marker branch (`:1632-1694`).
///
/// # Errors
/// As [`points_3d_file`].
pub fn points_3d_marker_file(
    trial: &Trial,
    marker: usize,
    opts: &Points3dOptions,
) -> Result<String, ExportError> {
    let data = export_data(trial, &[marker], opts)?;
    let (points, status) = &data[0];
    Ok(marker_file_text(trial, marker, points, status, opts))
}

fn marker_file_text(
    trial: &Trial,
    marker: usize,
    points: &[Point3],
    status: &[MarkerStatus],
    opts: &Points3dOptions,
) -> String {
    let mut out = String::new();
    if opts.header {
        if opts.frames.is_some() {
            out.push_str("Frame,");
        }
        header_triple(&mut out, &header_name(trial, marker, opts.filter_frequency));
        out.push('\n');
    }
    for f in export_frames(opts.frames, trial.n_frames) {
        if opts.frames.is_some() {
            out.push_str(&(f + 1).to_string());
            out.push(',');
        }
        value_triple(&mut out, points, status, f);
        out.push('\n');
    }
    out
}

/// The file name of one marker's 3D export:
/// `Marker%03d_<description>_points3d[_<cutoff>Hz].csv`.
///
/// Ported from core/Trial.cpp:1637-1642.
#[must_use]
pub fn points_3d_marker_file_name(
    trial: &Trial,
    marker: usize,
    filter_frequency: f64,
    compat: CppCompat,
) -> String {
    let d = &trial.markers[marker].description;
    let mut name = format!(
        "Marker{}_{}_points3d",
        padded(marker),
        super::file_description(d, marker_export_name(d, marker), compat.export_details)
    );
    if filter_frequency > 0.0 {
        name.push('_');
        name.push_str(&format_qt_number(filter_frequency));
        name.push_str("Hz");
    }
    name.push_str(".csv");
    name
}

/// The 3D export as the GUI writes it: one file, or one file per marker. All markers are
/// filtered first; if any is refused, nothing is written.
///
/// Ported from core/Trial.cpp, Trial::save3dPoints. The GUI's own checks come before it
/// (`ui/MainWindow.cpp:1731-1742`): with filtering asked for and a recording speed of 0 or
/// less it shows an error and exports nothing.
///
/// # Errors
/// As [`points_3d_file`].
pub fn points_3d(
    trial: &Trial,
    markers: &[usize],
    opts: &Points3dOptions,
) -> Result<ExportOutput, ExportError> {
    if opts.one_file {
        return points_3d_file(trial, markers, opts).map(ExportOutput::File);
    }
    let data = export_data(trial, markers, opts)?;
    Ok(ExportOutput::Folder(
        markers
            .iter()
            .zip(&data)
            .map(|(&i, (points, status))| {
                ExportFile::new(
                    points_3d_marker_file_name(trial, i, opts.filter_frequency, opts.compat),
                    marker_file_text(trial, i, points, status, opts),
                )
            })
            .collect(),
    ))
}
