//! The export files of XMALab 3, byte for byte as the C++ writes them on Linux.
//!
//! Each export is a function from the model (plus an options struct whose `Default` is what
//! the GUI passes by default) to the file's text. Where the C++ writes several files, a
//! second function returns the set with the names the C++ gives them. Nothing here touches
//! the file system: the caller writes the text, so the GUI's file dialogs and the CLI stay in
//! charge of where files go.
//!
//! | Export | Text | Set of files | Ported from |
//! |---|---|---|---|
//! | 2D points | [`points2d::points_2d_file`], [`points2d::points_2d_marker_file`] | [`points2d::points_2d`] | `Trial::save2dPoints` |
//! | 3D points | [`points3d::points_3d_file`], [`points3d::points_3d_marker_file`] | [`points3d::points_3d`] | `Trial::save3dPoints` |
//! | Rigid-body transformations | [`rigid_bodies::transformations_file`], [`rigid_bodies::transformations_body_file`] | [`rigid_bodies::transformations`] | `Trial::saveRigidBodyTransformations` |
//! | MayaCam 2 | [`mayacam::maya_cam_v2`] | [`mayacam::maya_cams_v2`] | `Camera::saveMayaCamVersion2`, `Project::exportMayaCamVersion2` |
//! | Reprojection errors | [`reprojection::reprojection_error_file`] | [`reprojection::reprojection_errors`] | `Trial::saveReprojectionErrors` |
//! | Marker-to-marker distances | [`precision::marker_to_marker_distances`] | — | `Trial::saveMarkerToMarkerDistances` |
//! | Precision info | [`precision::precision_info`] | — | `Trial::savePrecisionInfo` |
//! | Files written at project save | — | [`precision::project_save_files`] | `ProjectFileIO::saveProject` |
//! | Undistortion | [`undistortion::lut_file`], [`undistortion::inlier_points_file`], [`undistortion::base_points_file`] | [`undistortion::export_lut`] | `UndistortionObject::exportData`, `Project::exportLUT` |
//! | Events | [`events::event_file`] | [`events::events`] | `EventData::saveData` |
//!
//! # Common rules (`docs/cpp-recompute.md`, section 6)
//!
//! - Numbers: `std::ofstream` with `precision(17)`, so `%.17g` for doubles
//!   ([`crate::cfmt::format_g17`]) and plain decimal for integers. Missing values are the
//!   literal `NaN` ([`crate::cfmt::MISSING_NAN`]).
//! - Separator `,` with no space, no quoting or escaping of names.
//! - Lines end with `\n` (the C++ opens files in text mode, so `\r\n` on Windows; the
//!   reference files and this port use `\n`).
//! - File names inside a folder are built as the C++ concatenates them: zero-padded 1-based
//!   indices ([`crate::cfmt::format_int_padded`]), the description as stored, even when
//!   empty (`Marker001__points3d.csv`).
//!
//! # Input the C++ would read out of bounds on
//!
//! The functions that produce a whole export (the "set of files" column, and the
//! one-file functions that take a frame range) check their input first and return an
//! [`ExportError`] instead of writing anything: a frame range past the trial, an index that
//! does not exist, a marker with fewer frames or cameras than the trial ([`check_trial`]).
//! The per-item helpers (`*_marker_file`, `*_body_file`, `maya_cam_v2`) document their
//! preconditions and are meant to be called after those checks.
//!
//! Derived data the model does not hold (the fitted camera models for undistorted 2D points,
//! the undistortion lookup table) is passed in as plain `xma-vision` types. Rigid-body poses
//! and errors are read from the model as they are; the C++ recomputes the exported bodies
//! before writing the transformations (ledger A36), which is the engine's job.

pub mod blender_scene;
pub mod events;
pub mod mayacam;
pub mod points2d;
pub mod points3d;
pub mod precision;
pub mod reprojection;
pub mod rigid_bodies;
pub mod undistortion;

#[cfg(test)]
mod tests;

use crate::cfmt::{format_g17, format_int_padded};

/// One file of an export that writes several, named as the C++ names it inside the folder
/// the user chose.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportFile {
    /// File name relative to the output folder.
    pub name: String,
    /// The whole text of the file.
    pub contents: String,
}

impl ExportFile {
    fn new(name: String, contents: String) -> Self {
        Self { name, contents }
    }
}

/// What an export with a "one file / one file per item" option produces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExportOutput {
    /// One file, written to the path the user chose (the C++ save-file dialog).
    File(String),
    /// Several files, written into the folder the user chose (the C++ folder dialog).
    Folder(Vec<ExportFile>),
}

impl ExportOutput {
    /// The text of a one-file export. Panics for a folder.
    #[must_use]
    pub fn into_file(self) -> String {
        match self {
            Self::File(text) => text,
            Self::Folder(_) => panic!("export produced a folder, not one file"),
        }
    }

    /// The files of a folder export. Panics for one file.
    #[must_use]
    pub fn into_folder(self) -> Vec<ExportFile> {
        match self {
            Self::Folder(files) => files,
            Self::File(_) => panic!("export produced one file, not a folder"),
        }
    }
}

/// A frame range for the exports that have a frame column: 0-based, both ends included
/// (the C++ `start` and `stop`, which the GUI sets to `from - 1` and `to - 1` of its
/// From/To dialog).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameRange {
    pub start: usize,
    pub stop: usize,
}

impl FrameRange {
    /// The frames of the range; empty when `stop < start`, as the C++ loop is.
    pub(crate) fn frames(self) -> std::ops::Range<usize> {
        self.start..self.stop + 1
    }
}

/// The frames an export with an optional frame column writes: the range when the frame
/// column is on, otherwise every frame of the trial (the C++ resets `start` and `stop`).
fn export_frames(range: Option<FrameRange>, n_frames: usize) -> std::ops::Range<usize> {
    match range {
        Some(r) => r.frames(),
        None => 0..n_frames,
    }
}

/// Why an export was not written. Each case is one the C++ either refuses with a message
/// (the filter) or would run into undefined behaviour on (reading past a vector); here
/// nothing is written and the caller gets the reason.
#[derive(Clone, Debug, PartialEq)]
pub enum ExportError {
    /// `Marker::filterMarker` refused the cutoff (3D export).
    FilterRefused(points3d::FilterRefused),
    /// A frame range that goes past the trial: `start..=stop` (0-based, inclusive) or
    /// `from..to` (half-open), against the trial's `n_frames`.
    FrameRange {
        /// First frame asked for (0-based).
        first: usize,
        /// One past the last frame asked for.
        end: usize,
        /// Frames in the trial.
        n_frames: usize,
    },
    /// A marker index that does not exist.
    NoSuchMarker(usize),
    /// A rigid-body index that does not exist.
    NoSuchBody(usize),
    /// A camera index that does not exist.
    NoSuchCamera(usize),
    /// The model is inconsistent in a way the C++ would read out of bounds on (a marker with
    /// fewer frames or cameras than the trial, a body naming a marker that does not exist).
    Inconsistent(String),
}

impl std::fmt::Display for ExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FilterRefused(e) => write!(f, "{e}"),
            Self::FrameRange {
                first,
                end,
                n_frames,
            } => write!(
                f,
                "frames {} to {} are outside the trial, which has {n_frames} frames",
                first + 1,
                end
            ),
            Self::NoSuchMarker(i) => write!(f, "marker {} does not exist", i + 1),
            Self::NoSuchBody(i) => write!(f, "rigid body {} does not exist", i + 1),
            Self::NoSuchCamera(i) => write!(f, "camera {} does not exist", i + 1),
            Self::Inconsistent(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for ExportError {}

impl From<points3d::FilterRefused> for ExportError {
    fn from(e: points3d::FilterRefused) -> Self {
        Self::FilterRefused(e)
    }
}

/// Checks an inclusive range `start..=stop` against a trial of `n_frames` frames. A range
/// with `stop < start` writes nothing in the C++ and is accepted.
pub(crate) fn check_range(range: Option<FrameRange>, n_frames: usize) -> Result<(), ExportError> {
    match range {
        Some(r) if r.start <= r.stop && r.stop >= n_frames => Err(ExportError::FrameRange {
            first: r.start,
            end: r.stop + 1,
            n_frames,
        }),
        _ => Ok(()),
    }
}

/// Checks a half-open range `from..to` against a trial of `n_frames` frames. A range with
/// `to <= from` writes no value in the C++ and is accepted.
pub(crate) fn check_half_open(from: usize, to: usize, n_frames: usize) -> Result<(), ExportError> {
    if from < to && to > n_frames {
        Err(ExportError::FrameRange {
            first: from,
            end: to,
            n_frames,
        })
    } else {
        Ok(())
    }
}

/// The cutoff correction of the low-pass filter for a compatibility mode: corrected for the
/// two passes by default, as XMALab 3 in its mode (parity ledger, item A7,
/// [`xma_model::CppCompat::filter_cutoff_single_pass`]).
#[must_use]
pub fn cutoff_correction(compat: xma_model::CppCompat) -> xma_vision::filter::CutoffCorrection {
    if compat.filter_cutoff_single_pass {
        xma_vision::filter::CutoffCorrection::SinglePass
    } else {
        xma_vision::filter::CutoffCorrection::BothPasses
    }
}

/// Checks that every marker of the trial holds at least `n_frames` frames of 3D data and of
/// 2D data in `n_cameras` cameras (when given), and that every rigid body names existing
/// markers.
///
/// The C++ sizes every marker to the trial when it loads or creates it; a model where this
/// does not hold would make the C++ read out of bounds in every export.
///
/// # Errors
/// [`ExportError::Inconsistent`] naming the first marker or body that does not fit.
pub fn check_trial(trial: &xma_model::Trial, n_cameras: Option<usize>) -> Result<(), ExportError> {
    let n = trial.n_frames;
    for (i, m) in trial.markers.iter().enumerate() {
        let short_3d = m.points3d.len() < n || m.status3d.len() < n;
        let bad_views = n_cameras.is_some_and(|c| {
            m.views.len() < c
                || m.views[..c].iter().any(|v| {
                    v.points.len() < n
                        || v.status.len() < n
                        || v.error.len() < n
                        || v.projected.len() < n
                })
        });
        if short_3d || bad_views {
            return Err(ExportError::Inconsistent(format!(
                "marker {} holds fewer frames or cameras than the trial ({n} frames{})",
                i + 1,
                n_cameras.map_or(String::new(), |c| format!(", {c} cameras"))
            )));
        }
    }
    for (k, b) in trial.rigid_bodies.iter().enumerate() {
        if let Some(&m) = b.marker_indices.iter().find(|&&m| m >= trial.markers.len()) {
            return Err(ExportError::Inconsistent(format!(
                "rigid body {} names marker {}, which does not exist",
                k + 1,
                m + 1
            )));
        }
    }
    Ok(())
}

/// Checks marker indices against the trial.
pub(crate) fn check_markers(
    trial: &xma_model::Trial,
    markers: &[usize],
) -> Result<(), ExportError> {
    match markers.iter().find(|&&m| m >= trial.markers.len()) {
        Some(&m) => Err(ExportError::NoSuchMarker(m)),
        None => Ok(()),
    }
}

/// Checks rigid-body indices against the trial.
pub(crate) fn check_bodies(trial: &xma_model::Trial, bodies: &[usize]) -> Result<(), ExportError> {
    match bodies.iter().find(|&&b| b >= trial.rigid_bodies.len()) {
        Some(&b) => Err(ExportError::NoSuchBody(b)),
        None => Ok(()),
    }
}

/// `%.17g`, the format of every `double` written to an export.
fn g(value: f64) -> String {
    format_g17(value)
}

/// A 1-based index padded to three digits: `QString("%1").arg(i + 1, 3, 10, QChar('0'))`.
fn padded(index: usize) -> String {
    format_int_padded(index as u64 + 1, 3)
}

/// The name an export uses for a marker: its description, or `marker%03d` (1-based) when the
/// description is empty.
///
/// Ported from core/Trial.cpp, Trial::save2dPoints (`:1730-1737`) and Trial::save3dPoints
/// (`:1572-1579`).
#[must_use]
pub fn marker_export_name(description: &str, index: usize) -> String {
    if description.is_empty() {
        format!("marker{}", padded(index))
    } else {
        description.to_owned()
    }
}

/// The name an export uses for a rigid body: its description, or `RigidBody%03d` (1-based)
/// when the description is empty.
///
/// Ported from core/Trial.cpp, Trial::saveRigidBodyTransformations (`:971-979`).
#[must_use]
pub fn body_export_name(description: &str, index: usize) -> String {
    if description.is_empty() {
        format!("RigidBody{}", padded(index))
    } else {
        description.to_owned()
    }
}

/// The description that goes into a per-item file name: as stored in XMALab 3
/// ([`xma_model::CppCompat::export_details`], ledger F25: `Marker001__points3d.csv` for an
/// empty one); fixed, an empty description is replaced by `default`, the name the export's
/// header gives the item (`marker%03d`, `RigidBody%03d`).
pub(crate) fn file_description(description: &str, default: String, cpp: bool) -> String {
    if description.is_empty() && !cpp {
        default
    } else {
        description.to_owned()
    }
}

/// The text of a one-file export with no items (no marker or body): XMALab 3 writes the
/// frame column fields with no line end (ledger F25); fixed, each line is ended. `header`
/// says whether there is a header row, `frames` the frame column's frames (`None` without
/// the column, which leaves the file empty).
pub(crate) fn empty_items_text(header: bool, frames: Option<std::ops::Range<usize>>) -> String {
    let mut out = String::new();
    if let Some(frames) = frames {
        if header {
            out.push_str("Frame\n");
        }
        for f in frames {
            out.push_str(&(f + 1).to_string());
            out.push('\n');
        }
    }
    out
}

/// `QFileInfo(path).completeBaseName()` on Linux: the file name (after the last `/`) up to
/// but not including its last `.`; the whole file name when it has no `.`.
///
/// Ported from Qt 6, `QFileSystemEntry::completeBaseName`, as `CalibrationImage::getFilenameBase`
/// and `UndistortionObject::getFilenameBase` call it.
#[must_use]
///
/// The name is taken after the last `/` **or** `\\` on every system: Qt holds paths with `/`,
/// this program holds them as the system gives them (`\\` on Windows), and a project file
/// may come from another system (review 8, finding 11).
pub fn complete_base_name(path: &str) -> String {
    crate::project_write::complete_base_name(path).to_owned()
}

/// One part of a path that an export or the denoising builds from a name the project holds
/// (a trial's or camera's name, a marker's or event's description, a video's file name), as
/// it is written: never a folder above the one chosen, never two parts, and in the default
/// mode a name every system can create (ledger F70).
///
/// - Refused in both modes (`Err`, nothing written under it): an empty name, `.`, `..`, and,
///   in the XMALab 3 mode, a name holding `/` or `\\`. XMALab writes them as they are: a
///   trial named `..` writes into the folder above the one chosen, a description with `/`
///   into a subfolder (or nowhere, when that folder does not exist).
/// - Replaced in the default mode (each by `_`): `/`, `\\`, `:`, `*`, `?`, `"`, `<`, `>`,
///   `|`, control characters, and dots and spaces at the end (Windows drops them); a name
///   whose part before the first dot is a Windows device name (`CON`, `PRN`, `AUX`, `NUL`,
///   `COM1` to `COM9`, `LPT1` to `LPT9`, any case) gets `_` in front. The XMALab 3 mode
///   keeps such names as XMALab writes them (on Linux all of them can be created).
///
/// # Errors
/// The sentence saying why the name is refused.
pub fn safe_component(name: &str, xmalab3: bool) -> Result<String, String> {
    if name.is_empty() || name == "." || name == ".." {
        return Err(format!(
            "\"{name}\" cannot be the name of a file or folder; not written"
        ));
    }
    if xmalab3 {
        if name.contains(['/', '\\']) {
            return Err(format!(
                "\"{name}\" holds a folder separator, so it would be written into another \
                 folder; not written"
            ));
        }
        return Ok(name.to_owned());
    }
    let mut out: String = name
        .chars()
        .map(|c| {
            if c.is_control() || matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') {
                '_'
            } else {
                c
            }
        })
        .collect();
    let kept = out.trim_end_matches(['.', ' ']).len();
    let trailing = out.len() - kept;
    if trailing > 0 {
        out.truncate(kept);
        out.push_str(&"_".repeat(trailing));
    }
    let stem = out.split('.').next().unwrap_or("").to_ascii_uppercase();
    let reserved = matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || ((stem.starts_with("COM") || stem.starts_with("LPT"))
            && stem.len() == 4
            && stem.as_bytes()[3].is_ascii_digit()
            && stem.as_bytes()[3] != b'0');
    if reserved {
        out.insert(0, '_');
    }
    Ok(out)
}

/// The last part of a path, after the last `/` or `\\` (either separator on every system).
#[must_use]
pub fn file_name_of(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or("")
}

#[cfg(test)]
mod helper_tests {
    use super::*;

    #[test]
    fn safe_components() {
        for bad in ["", ".", ".."] {
            assert!(safe_component(bad, false).is_err(), "{bad:?}");
            assert!(safe_component(bad, true).is_err(), "{bad:?}");
        }
        assert!(safe_component("a/b", true).is_err());
        assert!(safe_component("a\\b", true).is_err());
        assert_eq!(safe_component("a:b", true).unwrap(), "a:b");
        assert_eq!(safe_component("a/b\\c:d", false).unwrap(), "a_b_c_d");
        assert_eq!(safe_component("trial. ", false).unwrap(), "trial__");
        assert_eq!(safe_component("con", false).unwrap(), "_con");
        assert_eq!(safe_component("LPT1.txt", false).unwrap(), "_LPT1.txt");
        assert_eq!(safe_component("COM0", false).unwrap(), "COM0");
        assert_eq!(safe_component("console", false).unwrap(), "console");
        assert_eq!(safe_component("x\ty\u{7f}", false).unwrap(), "x_y_");
        assert_eq!(safe_component("Trial 1", false).unwrap(), "Trial 1");
        assert_eq!(safe_component("...", false).unwrap(), "___");
    }

    #[test]
    fn base_names_by_either_separator() {
        assert_eq!(complete_base_name("C:\\data\\cam1.avi"), "cam1");
        assert_eq!(complete_base_name("data\\cam1.avi"), "cam1");
        assert_eq!(complete_base_name("/d/cam1.tar.avi"), "cam1.tar");
        assert_eq!(file_name_of("a/b\\c"), "c");
    }

    #[test]
    fn names() {
        assert_eq!(marker_export_name("", 0), "marker001");
        assert_eq!(marker_export_name("", 999), "marker1000");
        assert_eq!(marker_export_name("Tip", 4), "Tip");
        assert_eq!(body_export_name("", 1), "RigidBody002");
        assert_eq!(body_export_name("Skull", 1), "Skull");
    }

    #[test]
    fn base_names() {
        assert_eq!(
            complete_base_name("/a/b.c/Calibration_cam1.tif"),
            "Calibration_cam1"
        );
        assert_eq!(complete_base_name("archive.tar.gz"), "archive.tar");
        assert_eq!(complete_base_name("noext"), "noext");
        assert_eq!(
            complete_base_name("/data/video.avi/Frame000012"),
            "Frame000012"
        );
        assert_eq!(complete_base_name(".hidden"), "");
        assert_eq!(complete_base_name(""), "");
    }

    #[test]
    fn frame_ranges() {
        assert_eq!(export_frames(None, 3), 0..3);
        assert_eq!(
            export_frames(Some(FrameRange { start: 2, stop: 4 }), 9),
            2..5
        );
        assert!(
            export_frames(Some(FrameRange { start: 4, stop: 2 }), 9)
                .next()
                .is_none()
        );
    }
}
