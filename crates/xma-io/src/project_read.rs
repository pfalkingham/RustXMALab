//! Reading `.xma` project files.
//!
//! Ported from `ui/ProjectFileIO.cpp` (`ProjectFileIO::loadProject`,
//! `ProjectFileIO::readProjectFile`, `ProjectFileIO::unzipFromFileToFolder`) and the load
//! functions they reach in `core/`. The result is what the C++ holds when
//! `ProjectFileIO::loadProject` returns, before the interface recomputes anything
//! (undistortion fits, mean marker sizes, 3D points, rigid-body poses: see the module
//! documentation of each item in the work-package report).
//!
//! The archive is read in memory, without the C++'s temporary folder (parity ledger F5).
//! A path in `project.xml` is resolved as the C++ resolves `basedir + separator + path`:
//! relative to the archive root, with `\` and `/` both taken as separators.
//!
//! Where the C++ would read out of bounds (a CSV row with too few values, a marker ID
//! beyond the marker list, a missing camera matrix) it has undefined behaviour; this
//! reader returns a [`ReadError`] naming the entry, line and column instead. Where the C++
//! silently tolerates something (a referenced file that is absent, a colour it cannot
//! parse) the reader does the same and adds a line to [`LoadedProject::warnings`].

use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::sync::Arc;

use xma_model::{
    AdjustedView, CalibrationFrame, CalibrationImage, CalibrationObject, CalibrationSequence,
    CalibrationType, Camera, DetectionMethod, Event, Interpolation, Marker, MarkerStatus,
    MarkerView, Matrix3, Point2, Point3, Project, Rgb, RigidBody, Trial, UndistortionGrid, Vector3,
    VideoSource, VirtualPoint, VirtualPointSource,
};
use xma_model::{AlgorithmSettings, CppCompat, SETTING_NAMES, SettingSource, SettingsPatch};

use crate::cparse::{
    LineReader, c_int, lines, qstring, qt_color, qt_to_double, qt_to_int, split_commas,
    stream_ints, stream_values, stream_values_nonfinite,
};
use crate::image_size::{ImageSize, image_size};
use crate::settings_entry::{SETTINGS_ENTRY, parse_settings};
use crate::xml::{Token, line_column, tokenize};
use crate::zip::{Archive, Entry, ZipError};

/// A loaded project and what the model does not hold.
#[derive(Clone, Debug)]
pub struct LoadedProject {
    pub project: Project,
    /// Archive entries the model does not hold (calibration and grid images, `log.html`,
    /// `xma_metadata.xml`, `MarkerDistances_*` and `PrecisionInfo_*` files, anything
    /// unknown), in archive order, so that a later save can carry them.
    pub attachments: Vec<Attachment>,
    /// The `Version` attribute of `<Project>`.
    pub file_version: FileVersion,
    /// The archive comment as stored (`XROMM Project File` for files XMALab writes).
    pub archive_comment: Vec<u8>,
    /// Per trial, in project order: where the frame count and recording speed came from.
    pub trial_sources: Vec<TrialSource>,
    /// Things the C++ loader tolerates silently and a user might want to know.
    pub warnings: Vec<String>,
    /// Where each setting of `project.algorithm_settings` came from, in
    /// [`xma_model::SETTING_NAMES`] order: [`ReadOptions::base_settings`] (`Default`), the
    /// project's settings entry ([`crate::settings_entry`], `Project`) or
    /// [`ReadOptions::overrides`] (`CommandLine`).
    pub settings_sources: Vec<(&'static str, SettingSource)>,
    /// The settings the project's settings entry gives, as read; empty without one.
    pub stored_settings: SettingsPatch,
    /// The compatibility mode the project was read in ([`ReadOptions::compat`]); a document
    /// built from this keeps using it.
    pub compat: CppCompat,
}

/// An archive entry carried through unchanged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attachment {
    /// The entry as recorded in the central directory (raw name, method, time,
    /// attributes, comment).
    pub entry: Entry,
    /// The name decoded as the C++ decodes it ([`Entry::decoded_name`]).
    pub name: String,
    /// The uncompressed data.
    pub data: Vec<u8>,
}

/// The project file version.
#[derive(Clone, Debug, PartialEq)]
pub struct FileVersion {
    /// The attribute text, or `None` when there is no `Version` attribute (or no
    /// `<Project>` element).
    pub text: Option<String>,
    /// The number the C++ compares with 0.2 and 0.3 (`QString::toDouble`; 0 when absent).
    pub value: f64,
}

/// Where a trial's frame count and recording speed came from, and which of its markers had
/// no `<Marker>` element.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrialSource {
    /// `nbImages` was in the file. Otherwise the C++ takes it from the first camera's
    /// video when the trial is constructed.
    pub n_frames_from_file: bool,
    /// `recordingSpeed` was in the file. Otherwise the C++ takes the first video's frame
    /// rate.
    pub recording_speed_from_file: bool,
    /// Markers (0-based) that have a line in `MarkerDescription.txt` but no `<Marker>`
    /// element, for example from a blank last line. The C++ never runs `Marker::load` for
    /// them, so their size range is left uninitialised (ledger A52); every marker a save
    /// writes has an element.
    pub markers_without_element: Vec<usize>,
}

/// Frame count and rate of a video file, as the C++ video readers report them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VideoInfo {
    /// `VideoStream::getNbImages`; -1 when the file cannot be opened.
    pub n_frames: i32,
    /// `VideoStream::getFPS`.
    pub fps: f64,
}

/// Options for [`read_project_with`]. What the C++ loader takes from outside the project
/// file: the file system (videos, meshes) and the per-machine settings. The default has
/// no probes, XMALab's default settings and no overrides.
#[derive(Clone, Default)]
pub struct ReadOptions<'a> {
    /// Opens a video file (`.avi`, `.mp4`, `.mov`, `.cine`) and reports its length and rate.
    /// Called only for a trial that lacks `nbImages` or `recordingSpeed`, as old project
    /// files do. Without it, or when it returns `None`, the reader does what the C++ does
    /// for a video it cannot open: 0 frames and a rate of 0, with a warning.
    pub video_info: Option<VideoProbe<'a>>,
    /// Says whether a rigid body's mesh file loads, given the `Meshmodel` attribute exactly
    /// as stored. [`mesh_opens`] is the C++ test. Its answer is recorded in
    /// [`RigidBody::mesh_readable`], and the path, `DrawMeshmodel` and `MeshScale` are kept
    /// either way (parity ledger F18, fixed). With [`CppCompat::drop_unreadable_mesh`]
    /// (XMALab 3) a mesh whose file does not load is dropped as the C++ drops it: no mesh,
    /// drawing off and scale 1, and the two attributes not read. Without a probe the reader
    /// keeps the path and the two attributes, leaves `mesh_readable` unset and adds a
    /// warning.
    pub mesh_loads: Option<MeshProbe<'a>>,
    /// The settings where the project's settings entry ([`crate::settings_entry`]) gives
    /// none: XMALab's per-machine settings. Default: XMALab's defaults. Loading reads one
    /// of them, `DefaultMarkerThreshold`: the threshold offset of a marker whose
    /// `ThresholdOffset` attribute is absent, or which has no `<Marker>` element
    /// (`Marker::Marker`, `core/Marker.cpp:50`; default 8,
    /// `core/Settings.cpp:217`, as [`AlgorithmSettings::default`] holds it).
    pub base_settings: AlgorithmSettings,
    /// Settings that override both (the command line). Applied after the project's.
    pub overrides: SettingsPatch,
    /// Which faults of XMALab 3 to reproduce; reading looks at
    /// [`CppCompat::drop_unreadable_mesh`] (ledger F18). Default: fixed.
    pub compat: CppCompat,
}

impl ReadOptions<'_> {
    /// [`ReadOptions::base_settings`] with `DefaultMarkerThreshold` set: XMALab's defaults
    /// otherwise.
    #[must_use]
    pub fn with_marker_threshold(threshold: i32) -> Self {
        let mut o = Self::default();
        o.base_settings.default_marker_threshold = threshold;
        o
    }
}

/// A function that opens a camera's video and reports its frame count and rate.
pub type VideoProbe<'a> = &'a dyn Fn(&VideoSource) -> Option<VideoInfo>;

/// A function that says whether a mesh file, named as the project stores it, loads.
pub type MeshProbe<'a> = &'a dyn Fn(&str) -> bool;

/// The C++ test of whether a rigid body's mesh loads, for [`ReadOptions::mesh_loads`].
///
/// Ported from gl/GLMLoader.cpp, GLMLoader::load, through core/RigidBody.cpp,
/// RigidBody::addMeshModel, and core/RigidBodyObj.cpp, RigidBodyObj::vboSet. The loader
/// returns null, and the mesh is dropped, only when `fopen(path.toUtf8(), "r")` fails.
/// Every file that opens gives a mesh, whatever it contains: an empty file, a file that is
/// not OBJ, and (on Linux and macOS) a directory all give a mesh with no triangles. So the
/// test is whether the path opens for reading:
///
/// - the path is the attribute as stored, not resolved against the archive; a relative
///   path is relative to the program's working directory;
/// - XMALab writes absolute paths, so a project moved to another machine loses its meshes
///   (ledger F18).
///
/// On Windows the C++ passes UTF-8 bytes to `fopen`, which reads them in the ANSI code page,
/// so a path with non-ASCII characters fails there; this function opens it. Malformed OBJ
/// content can make the C++ loader read out of bounds; that is not reproduced either.
#[must_use]
pub fn mesh_opens(path: &str) -> bool {
    std::fs::File::open(path).is_ok()
}

impl fmt::Debug for ReadOptions<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReadOptions")
            .field("video_info", &self.video_info.map(|_| "<fn>"))
            .field("mesh_loads", &self.mesh_loads.map(|_| "<fn>"))
            .field("base_settings", &self.base_settings)
            .field("overrides", &self.overrides)
            .field("compat", &self.compat)
            .finish()
    }
}

/// Why a project could not be read.
#[derive(Debug)]
pub enum ReadErrorKind {
    /// The file could not be read.
    Io(std::io::Error),
    /// The zip container is malformed or uses something unsupported.
    Zip(ZipError),
    /// The archive has no `project.xml` and is not an XMA Portal archive.
    NotAProject,
    /// An XMA Portal archive (the C++ `loadProject` returns 1 and offers to create a
    /// project from it). Reading those is out of scope.
    PortalArchive,
    /// `project.xml` is not well-formed or not UTF-8.
    Xml(String),
    /// Data the C++ cannot read without undefined behaviour, or the model cannot hold.
    Data(String),
}

/// A load failure, with the archive entry and, where it applies, the 1-based line and
/// column (for CSV files, the column is the 1-based value index).
#[derive(Debug)]
pub struct ReadError {
    pub kind: ReadErrorKind,
    pub entry: Option<String>,
    pub line: Option<usize>,
    pub column: Option<usize>,
}

impl ReadError {
    fn new(kind: ReadErrorKind) -> Self {
        Self {
            kind,
            entry: None,
            line: None,
            column: None,
        }
    }

    fn data(
        entry: &str,
        line: Option<usize>,
        column: Option<usize>,
        msg: impl Into<String>,
    ) -> Self {
        Self {
            kind: ReadErrorKind::Data(msg.into()),
            entry: Some(entry.to_owned()),
            line,
            column,
        }
    }
}

impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(e) = &self.entry {
            write!(f, "{e}")?;
            if let Some(l) = self.line {
                write!(f, ", line {l}")?;
            }
            if let Some(c) = self.column {
                write!(f, ", column {c}")?;
            }
            write!(f, ": ")?;
        }
        match &self.kind {
            ReadErrorKind::Io(e) => write!(f, "{e}"),
            ReadErrorKind::Zip(e) => write!(f, "{e}"),
            ReadErrorKind::NotAProject => write!(f, "not an XMALab project (no project.xml)"),
            ReadErrorKind::PortalArchive => write!(f, "an XMA Portal archive, not a project"),
            ReadErrorKind::Xml(m) | ReadErrorKind::Data(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for ReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            ReadErrorKind::Io(e) => Some(e),
            ReadErrorKind::Zip(e) => Some(e),
            _ => None,
        }
    }
}

/// Whether a frame count (a trial's `nbImages`, a calibration sequence's `Frames`) is
/// within [`xma_model::project::MAX_FRAMES`] (ledger F84). Not in the C++, which allocates
/// whatever the file says.
#[must_use]
pub fn frames_within_limit(n_frames: usize) -> bool {
    u64::try_from(n_frames).is_ok_and(|n| n <= xma_model::project::MAX_FRAMES)
}

/// Whether the product of the counts that size one trial's per-frame arrays (cameras x
/// frames x markers, frames x rigid bodies, ...) is within
/// [`xma_model::project::MAX_TRIAL_CELLS`] (ledger F84). The product is computed without
/// overflow: one that does not fit in 64 bits is beyond the limit; any zero count makes
/// it zero. Not in the C++.
#[must_use]
pub fn cells_within_limit(counts: &[usize]) -> bool {
    product(counts).is_some_and(|p| p <= xma_model::project::MAX_TRIAL_CELLS)
}

/// The product of `counts`, `None` when it does not fit in 64 bits; any zero count makes it
/// zero even where the others overflow.
fn product(counts: &[usize]) -> Option<u64> {
    if counts.contains(&0) {
        return Some(0);
    }
    counts
        .iter()
        .try_fold(1u64, |acc, &c| acc.checked_mul(u64::try_from(c).ok()?))
}

/// One element of a `<Trial>` that the reader acts on (ledger F84).
enum TrialElement {
    /// The index of an `<Event>` start tag.
    Event(usize),
    /// The index of a `<Marker>` start tag.
    Marker(usize),
    /// The index of a `<RigidBody>` start tag and of the `<DummyMarker>` start tags read
    /// with it.
    RigidBody { start: usize, dummies: Vec<usize> },
}

/// The elements of the `<Trial>` whose start tag is at `start`, in reading order, and the
/// index of its end tag (or the last token). Ported from the element loop of
/// ui/ProjectFileIO.cpp, ProjectFileIO::readProjectFile: inside a trial an `<Event>`, a
/// `<Marker>` or a `<RigidBody>` start tag is read, anything else is passed over; a rigid
/// body reads every `<DummyMarker>` up to the first `</RigidBody>`, and nothing else in it
/// (an event, a marker, a nested trial or body inside a body is passed over); the trial
/// ends at the first `</Trial>` outside a body.
///
/// Ledger F84: the size check and the reading loop of [`Loader::read_trial`] both walk this
/// list, so the elements counted are the elements read, whatever the nesting.
fn trial_elements(tokens: &[Token], start: usize) -> (Vec<TrialElement>, usize) {
    let last = tokens.len().saturating_sub(1);
    let mut elements = Vec::new();
    let mut i = start;
    while i < tokens.len() && !tokens[i].is_end("Trial") {
        if let Token::Start { name, .. } = &tokens[i] {
            match name.as_str() {
                "Event" => elements.push(TrialElement::Event(i)),
                "Marker" => elements.push(TrialElement::Marker(i)),
                "RigidBody" => {
                    let body = i;
                    let mut dummies = Vec::new();
                    while i < tokens.len() && !tokens[i].is_end("RigidBody") {
                        if tokens[i].is_start("DummyMarker") {
                            dummies.push(i);
                        }
                        i += 1;
                    }
                    i = i.min(last);
                    elements.push(TrialElement::RigidBody {
                        start: body,
                        dummies,
                    });
                }
                _ => {}
            }
        }
        i += 1;
    }
    (elements, i.min(last))
}

/// Calls `f(row, line)` for each line of `data` as `safeGetline` reads it, without
/// collecting them.
fn for_each_line(data: &[u8], mut f: impl FnMut(usize, &[u8]) -> bool) {
    let mut r = LineReader::new(data);
    let mut lc = 0;
    loop {
        let (line, eof) = r.next_line();
        if eof || !f(lc, line) {
            return;
        }
        lc += 1;
    }
}

/// The number of lines of `data` (as [`lines`] splits it) for which `keep` holds.
fn count_lines(data: &[u8], keep: impl Fn(&[u8]) -> bool) -> usize {
    let mut n = 0;
    for_each_line(data, |_, line| {
        n += usize::from(keep(line));
        true
    });
    n
}

/// An upper bound on the number of lines of `data` as [`lines`] splits it: its `\n` and
/// `\r` bytes plus one (every line but the last ends with one of `\n`, `\r` or `\r\n`).
/// One pass over the bytes, without splitting; it only decides whether
/// [`Loader::check_trial_size`] needs the exact lengths. Ledger F84.
fn line_bound(data: &[u8]) -> usize {
    // Counted in blocks of 255 bytes with a byte-wide counter, which the compiler
    // vectorises (several times faster than a `usize` per byte).
    let mut n = 1;
    for block in data.chunks(255) {
        let mut ends = 0u8;
        for &b in block {
            ends += u8::from((b == b'\n') | (b == b'\r'));
        }
        n += usize::from(ends);
    }
    n
}

/// The length an event's flags reach when [`Loader::read_event`] reads `data` into a trial
/// of `n_frames` frames: a row with a value at or beyond the end appends one flag
/// (core/EventData.cpp, EventData::loadData, `push_back`). Ledger F84.
fn event_length(data: &[u8], n_frames: usize) -> usize {
    let mut len = n_frames;
    for_each_line(data, |lc, line| {
        if len <= lc && !stream_values(line).is_empty() {
            len += 1;
        }
        true
    });
    len
}

/// The length a marker's arrays reach when [`load_marker_2d`] reads the 2D points `data`
/// into a marker of `len` frames and `n_cams` cameras: a row with values at or beyond the end
/// adds one frame (core/Marker.cpp, Marker::load, `addFrame`); the count stops where the
/// reader stops with an error. Ledger F84.
fn marker_2d_length(data: &[u8], nonfinite: bool, n_cams: usize, mut len: usize) -> usize {
    for_each_line(data, |lc, line| {
        let v = values(line, nonfinite);
        if v.is_empty() {
            return true;
        }
        if n_cams == 0 {
            return false;
        }
        if len <= lc {
            len += 1;
        }
        lc < len && v.len() >= 2 * n_cams
    });
    len
}

/// The number of coordinate rows [`Loader::read_dummy_marker`] keeps from `data` (after the
/// header line, the rows of three fields). Ledger F84.
fn coordinate_rows(data: &[u8]) -> usize {
    let mut rows = 0;
    for_each_line(data, |lc, line| {
        if lc > 0 && split_commas(&qstring(line)).len() == 3 {
            rows += 1;
        }
        true
    });
    rows
}

/// Whether a virtual point whose references file holds `refs` takes its positions from a
/// coordinates file (core/RigidBody.cpp, RigidBody::addDummyPoint: no rigid-body index, or
/// -1, after x, y and z on the second line).
fn reads_coordinates(refs: &[u8]) -> bool {
    let fields = second_line_fields(refs);
    fields.len() >= 3 && (fields.len() == 3 || qt_to_int(&fields[3]) == -1)
}

/// Loads a `.xma` project.
///
/// Ported from ui/ProjectFileIO.cpp, ProjectFileIO::loadProject.
pub fn read_project(path: &Path) -> Result<LoadedProject, ReadError> {
    read_project_with(path, &ReadOptions::default())
}

/// [`read_project`] with options.
pub fn read_project_with(
    path: &Path,
    options: &ReadOptions<'_>,
) -> Result<LoadedProject, ReadError> {
    let bytes = std::fs::read(path).map_err(|e| ReadError::new(ReadErrorKind::Io(e)))?;
    read_project_bytes(&bytes, options)
}

/// [`read_project`] from the bytes of a `.xma` file.
pub fn read_project_bytes(
    bytes: &[u8],
    options: &ReadOptions<'_>,
) -> Result<LoadedProject, ReadError> {
    let mut files = Files::open(bytes)?;
    // The settings first: loading reads `DefaultMarkerThreshold`.
    let mut warnings = Vec::new();
    let stored_settings = match files.take(SETTINGS_ENTRY) {
        Some(data) => {
            let (patch, w) = parse_settings(&data);
            warnings.extend(w);
            patch
        }
        None => SettingsPatch::default(),
    };
    let mut settings = options.base_settings.clone();
    let mut settings_sources: Vec<(&'static str, SettingSource)> = SETTING_NAMES
        .iter()
        .map(|&(n, _)| (n, SettingSource::Default))
        .collect();
    for (patch, source) in [
        (&stored_settings, SettingSource::Project),
        (&options.overrides, SettingSource::CommandLine),
    ] {
        for (name, value) in &patch.values {
            settings.set(name, *value).map_err(|e| {
                ReadError::new(ReadErrorKind::Data(match source {
                    SettingSource::Project => format!("{SETTINGS_ENTRY}: {e}"),
                    _ => e,
                }))
            })?;
            if let Some(s) = settings_sources.iter_mut().find(|(n, _)| *n == name) {
                s.1 = source;
            }
        }
    }
    let Some(xml_bytes) = files.take("project.xml") else {
        return Err(ReadError::new(if files.is_portal_archive() {
            ReadErrorKind::PortalArchive
        } else {
            ReadErrorKind::NotAProject
        }));
    };
    let xml_text = std::str::from_utf8(&xml_bytes).map_err(|e| ReadError {
        kind: ReadErrorKind::Xml(format!("not valid UTF-8: {e}")),
        entry: Some("project.xml".into()),
        line: Some(
            line_column(
                &String::from_utf8_lossy(&xml_bytes[..e.valid_up_to()]),
                e.valid_up_to(),
            )
            .0,
        ),
        column: None,
    })?;
    let tokens = tokenize(xml_text).map_err(|e| ReadError {
        kind: ReadErrorKind::Xml(e.message),
        entry: Some("project.xml".into()),
        line: Some(e.line),
        column: Some(e.column),
    })?;
    files.nonfinite = !options.compat.nonfinite_values_kept;
    let mut loader = Loader {
        files,
        options,
        xml_text,
        tokens: &tokens,
        project: Project::new(),
        calibration_frame: None,
        trials: Vec::new(),
        camera_ids: Vec::new(),
        warnings,
        checkerboard_axis_stored: false,
    };
    // `Project::Project` starts with an internal calibration (`core/Project.cpp`).
    loader.project.calibration_type = CalibrationType::Internal;
    loader.project.algorithm_settings = settings;
    let (file_version, active_trial) = loader.read_project_file()?;
    let axis_stored = loader.checkerboard_axis_stored;
    let mut loaded = loader.finish(file_version, active_trial)?;
    // Ledger F49: each NaN or infinity read is stored as its field's "not set" value.
    if !options.compat.nonfinite_values_kept {
        let lines = crate::finite::clear_non_finite(&mut loaded.project);
        loaded.warnings.extend(lines);
    }
    // Ledger F75: a checkerboard without its axis in the file (XMALab's own projects) takes
    // the axis its stored poses fit, in the default mode.
    if !axis_stored && !options.compat.checkerboard_axis_from_setting {
        crate::checkerboard_axis::recognise(&mut loaded);
    }
    loaded.settings_sources = settings_sources;
    loaded.stored_settings = stored_settings;
    loaded.compat = options.compat;
    Ok(loaded)
}

// ---------------------------------------------------------------------------------------
// Archive access

/// Entry data by resolved name, with a record of what the model consumed.
struct Files {
    entries: Vec<Entry>,
    names: Vec<String>,
    data: Vec<Vec<u8>>,
    /// Normalised name to the index of the last entry with that name (later entries
    /// overwrite earlier ones when the C++ extracts them).
    index: HashMap<String, usize>,
    used: Vec<bool>,
    comment: Vec<u8>,
    missing: Vec<String>,
    /// Ledger F49: data rows are read with [`stream_values_nonfinite`] (the default mode)
    /// rather than as `in >> double` reads them (XMALab 3).
    nonfinite: bool,
}

/// Normalises a path as the C++ file system calls see `basedir + separator + path`.
fn normalize(path: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for p in path.split(['/', '\\']) {
        match p {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other),
        }
    }
    parts.join("/")
}

/// The file name part of a path (`QFileInfo::fileName`).
fn file_name(path: &str) -> String {
    normalize(path).rsplit('/').next().unwrap_or("").to_owned()
}

impl Files {
    /// Ported from ui/ProjectFileIO.cpp, ProjectFileIO::unzipFromFileToFolder.
    fn open(bytes: &[u8]) -> Result<Self, ReadError> {
        let archive = Archive::new(bytes).map_err(|e| ReadError::new(ReadErrorKind::Zip(e)))?;
        let mut files = Files {
            entries: Vec::new(),
            names: Vec::new(),
            data: Vec::new(),
            index: HashMap::new(),
            used: Vec::new(),
            comment: archive.comment().to_vec(),
            missing: Vec::new(),
            nonfinite: false,
        };
        for (i, entry) in archive.entries().enumerate() {
            let name = entry.decoded_name();
            let data = archive.read(i).map_err(|e| ReadError {
                kind: ReadErrorKind::Zip(e),
                entry: Some(name.clone()),
                line: None,
                column: None,
            })?;
            if !entry.is_dir() {
                files.index.insert(normalize(&name), i);
            }
            files.entries.push(entry.clone());
            files.names.push(name);
            files.data.push(data);
            files.used.push(false);
        }
        Ok(files)
    }

    /// The data of a file the model will hold; marks it as consumed.
    fn take(&mut self, path: &str) -> Option<Vec<u8>> {
        let i = *self.index.get(&normalize(path))?;
        self.used[i] = true;
        Some(self.data[i].clone())
    }

    /// The data of a file the model holds, or empty with a warning when absent (the C++
    /// reads a missing file as an empty stream).
    fn take_or_empty(&mut self, path: &str) -> (String, Vec<u8>) {
        let name = normalize(path);
        match self.take(path) {
            Some(d) => (name, d),
            None => {
                self.missing.push(name.clone());
                (name, Vec::new())
            }
        }
    }

    /// The data of a file that stays an attachment (an image).
    fn peek(&self, path: &str) -> Option<&[u8]> {
        self.index
            .get(&normalize(path))
            .map(|&i| self.data[i].as_slice())
    }

    /// `loadProject`'s test for an XMA Portal archive: the first directory (by name)
    /// contains `File_Metadata/XMALab_Files-metadata.xml`.
    fn is_portal_archive(&self) -> bool {
        let mut dirs: Vec<&str> = self
            .names
            .iter()
            .filter_map(|n| n.split_once(['/', '\\']).map(|(d, _)| d))
            .filter(|d| !d.is_empty())
            .collect();
        dirs.sort_by_key(|d| d.to_lowercase());
        dirs.first().is_some_and(|d| {
            self.peek(&format!("{d}/File_Metadata/XMALab_Files-metadata.xml"))
                .is_some()
        })
    }
}

// ---------------------------------------------------------------------------------------
// Loader state

/// A marker while loading. Status, method and interpolation codes stay as read until the
/// version remaps have run.
struct RawMarker {
    description: String,
    method: i32,
    max_penalty: i32,
    /// The `TrackingPenalty` attribute as read, before `setMaxPenalty` caps it.
    penalty_attribute: Option<i32>,
    size_override: i32,
    threshold_offset: i32,
    requires_recomputation: bool,
    reference_3d: Option<Point3>,
    points2d: Vec<Vec<Point2>>,
    status2d: Vec<Vec<i32>>,
    size: Vec<Vec<f64>>,
    points3d: Vec<Point3>,
    status3d: Vec<i32>,
    interpolation: Vec<i32>,
    /// Entries the status and interpolation codes came from, for error messages.
    origin: String,
    /// A `<Marker>` element named this marker (so the C++ ran `Marker::load` on it).
    has_element: bool,
    /// The frames counted for it before reading (ledger F84): the trial's, or more from
    /// the rows of its 2D points files.
    planned_frames: usize,
}

impl RawMarker {
    /// Ported from core/Marker.cpp, Marker::Marker and Marker::init. `threshold_offset` is
    /// the setting `DefaultMarkerThreshold`.
    fn new(n_cameras: usize, n_frames: usize, threshold_offset: i32) -> Self {
        Self {
            description: String::new(),
            method: 0,
            max_penalty: xma_model::marker::DEFAULT_MAX_PENALTY,
            penalty_attribute: None,
            size_override: -1,
            threshold_offset,
            requires_recomputation: true,
            reference_3d: None,
            points2d: vec![vec![Point2::new(-2.0, -2.0); n_frames]; n_cameras],
            status2d: vec![vec![0; n_frames]; n_cameras],
            size: vec![vec![-1.0; n_frames]; n_cameras],
            points3d: vec![Point3::new(-1000.0, -1000.0, -1000.0); n_frames],
            status3d: vec![0; n_frames],
            interpolation: vec![0; n_frames],
            origin: String::new(),
            has_element: false,
            planned_frames: n_frames,
        }
    }

    fn n_frames(&self) -> usize {
        self.status3d.len()
    }

    /// Ported from core/Marker.cpp, Marker::addFrame.
    fn add_frame(&mut self) {
        for c in 0..self.points2d.len() {
            self.points2d[c].push(Point2::new(-2.0, -2.0));
            self.status2d[c].push(0);
            self.size[c].push(-1.0);
        }
        self.points3d.push(Point3::new(-1000.0, -1000.0, -1000.0));
        self.status3d.push(0);
        self.interpolation.push(0);
    }

    /// Room for `extra` more frames in every array, so that [`Self::add_frame`] allocates no
    /// more than the frames counted (ledger F84).
    fn reserve_frames(&mut self, extra: usize) {
        for c in 0..self.points2d.len() {
            self.points2d[c].reserve_exact(extra);
            self.status2d[c].reserve_exact(extra);
            self.size[c].reserve_exact(extra);
        }
        self.points3d.reserve_exact(extra);
        self.status3d.reserve_exact(extra);
        self.interpolation.reserve_exact(extra);
    }

    /// Ported from core/Marker.cpp, Marker::setMaxPenalty.
    fn set_max_penalty(&mut self, value: i32) {
        self.max_penalty = if value == 125 { 50 } else { value.min(100) };
    }

    /// Ported from core/Marker.cpp, Marker::updateToProject12 and Marker::updateStatus12.
    fn update_to_project_12(&mut self) {
        fn update_status_12(old: i32) -> i32 {
            match old {
                -2 => MarkerStatus::Deleted.code(),
                -1 => MarkerStatus::Lost.code(),
                1..=3 => MarkerStatus::Predicted.code(),
                4 => MarkerStatus::Tracked.code(),
                5 => MarkerStatus::Set.code(),
                6 => MarkerStatus::Manual.code(),
                _ => MarkerStatus::Undefined.code(),
            }
        }
        for i in 0..self.status3d.len() {
            self.status3d[i] = update_status_12(self.status3d[i]);
            for c in 0..self.status2d.len() {
                self.status2d[c][i] = update_status_12(self.status2d[c][i]);
            }
        }
        self.method = match self.method {
            4 => 0,
            5 => 2,
            6 => 4,
            m => m,
        };
    }

    /// Ported from core/Marker.cpp, Marker::updateToProject13 (integer arithmetic).
    ///
    /// XMALab 3 (`cpp`) divides the already capped penalty by 125 before multiplying by 50,
    /// so every marker ends with 0 (ledger F13). Fixed: a penalty read from the file is
    /// converted from the old scale (125) to the new one (50) first, `value * 50 / 125`
    /// truncated as the `int` assignment would, and then capped by `setMaxPenalty`; a marker
    /// without the attribute keeps the default.
    fn update_to_project_13(&mut self, cpp: bool) {
        if cpp {
            self.max_penalty = self.max_penalty / 125 * 50;
        } else if let Some(v) = self.penalty_attribute {
            let converted = i64::from(v) * 50 / 125;
            self.set_max_penalty(i32::try_from(converted).unwrap_or(i32::MAX));
        }
    }

    fn into_model(self) -> Result<Marker, ReadError> {
        let origin = self.origin.clone();
        let status = |code: i32, what: &str| {
            MarkerStatus::from_code(code).ok_or_else(|| {
                ReadError::data(
                    &origin,
                    None,
                    None,
                    format!("{what} code {code} is not a marker status"),
                )
            })
        };
        let n_frames = self.n_frames();
        let mut views = Vec::with_capacity(self.points2d.len());
        for c in 0..self.points2d.len() {
            let mut v = MarkerView::new(n_frames);
            v.points = self.points2d[c].clone();
            v.size = self.size[c].clone();
            v.status = self.status2d[c]
                .iter()
                .map(|&s| status(s, "2D status"))
                .collect::<Result<_, _>>()?;
            views.push(v);
        }
        let method = DetectionMethod::from_code(self.method).ok_or_else(|| {
            ReadError::data(
                "project.xml",
                None,
                None,
                format!(
                    "marker '{}': detection method {} is not known",
                    self.description, self.method
                ),
            )
        })?;
        let status3d = self
            .status3d
            .iter()
            .map(|&s| status(s, "3D status"))
            .collect::<Result<_, _>>()?;
        let interpolation = self
            .interpolation
            .iter()
            .map(|&m| {
                Interpolation::from_code(m).ok_or_else(|| {
                    ReadError::data(
                        &origin,
                        None,
                        None,
                        format!("interpolation code {m} is not known"),
                    )
                })
            })
            .collect::<Result<_, _>>()?;
        let mut m = Marker::new(views.len(), n_frames, self.threshold_offset);
        m.description = self.description;
        m.method = method;
        m.max_penalty = self.max_penalty;
        m.size_override = self.size_override;
        m.requires_recomputation = self.requires_recomputation;
        m.reference_3d = self.reference_3d;
        m.views = views;
        m.points3d = self.points3d;
        m.status3d = status3d;
        m.interpolation = interpolation;
        Ok(m)
    }
}

/// A trial while loading.
struct RawTrial {
    trial: Trial,
    markers: Vec<RawMarker>,
    bodies: Vec<RigidBody>,
    source: TrialSource,
    /// The cells its per-frame arrays hold so far (ledger F84).
    cells: TrialCells,
    /// The frames each array reaches, counted before reading, by the index of its
    /// element's start tag ([`Loader::check_trial_size`]).
    planned: HashMap<usize, usize>,
    /// `planned` holds the exact lengths (computed because the cheap bounds passed a
    /// limit), so the reader reserves them; otherwise they are upper bounds only, and the
    /// arrays grow as rows are read, as they did before ledger F84.
    planned_exact: bool,
}

/// Ledger F84: the frames each per-frame array of a trial reaches, as counted before
/// reading ([`Loader::plan_trial`]).
#[derive(Default)]
struct TrialPlan {
    /// The frames each array reaches, by the index of its element's start tag.
    planned: HashMap<usize, usize>,
    /// The lengths are exact, not upper bounds.
    exact: bool,
    /// The first array beyond `MAX_FRAMES`: its element, name, file and frames.
    too_long: Option<(usize, String, String, usize)>,
    events: usize,
    event_cells: u64,
    virtual_points: usize,
    virtual_point_cells: u64,
    /// Cameras x the frames of every marker.
    marker_cells: u64,
}

impl TrialPlan {
    fn within_limits(&self) -> bool {
        let limit = xma_model::project::MAX_TRIAL_CELLS;
        self.too_long.is_none()
            && self.event_cells <= limit
            && self.virtual_point_cells <= limit
            && self.marker_cells <= limit
    }
}

/// Ledger F84: the cells of one trial's per-frame arrays allocated so far, counted with
/// the frames each array reaches: cameras x frames of every marker, the frames of every
/// event, and of every virtual point.
#[derive(Default)]
struct TrialCells {
    markers: u64,
    events: u64,
    virtual_points: u64,
}

struct Loader<'a, 'o> {
    files: Files,
    options: &'o ReadOptions<'o>,
    xml_text: &'a str,
    tokens: &'a [Token],
    project: Project,
    calibration_frame: Option<CalibrationFrame>,
    trials: Vec<RawTrial>,
    /// Camera numbers from the project's portal metadata (`Project::cameraIDs`).
    camera_ids: Vec<i32>,
    warnings: Vec<String>,
    /// A checkerboard's `InvertedAxis` attribute was read (ledger F73).
    checkerboard_axis_stored: bool,
}

const XML: &str = "project.xml";

/// Image sizes of one camera while it is read.
struct CameraImages {
    calibration: Vec<ImageSize>,
    grid: Option<ImageSize>,
}

impl Loader<'_, '_> {
    fn xml_error(&self, index: usize, msg: impl Into<String>) -> ReadError {
        let (line, column) = match &self.tokens[index] {
            Token::Start { offset, .. } => {
                let (l, c) = line_column(self.xml_text, *offset);
                (Some(l), Some(c))
            }
            _ => (None, None),
        };
        ReadError {
            kind: ReadErrorKind::Data(msg.into()),
            entry: Some(XML.into()),
            line,
            column,
        }
    }

    fn non_negative(&self, index: usize, value: i32, what: &str) -> Result<usize, ReadError> {
        usize::try_from(value)
            .map_err(|_| self.xml_error(index, format!("{what} is {value}; it cannot be negative")))
    }

    fn warn(&mut self, msg: String) {
        self.warnings.push(msg);
    }

    /// Ported from ui/ProjectFileIO.cpp, ProjectFileIO::readProjectFile.
    fn read_project_file(&mut self) -> Result<(FileVersion, i32), ReadError> {
        let tokens = self.tokens;
        let mut version = FileVersion {
            text: None,
            value: 0.0,
        };
        let mut saw_project = false;
        let mut active_trial = -1;
        let mut old_flip_images = false;
        let mut i = 0;
        while i < tokens.len() {
            if let Token::Start { name, .. } = &tokens[i] {
                let t = &tokens[i];
                match name.as_str() {
                    "Project" => {
                        saw_project = true;
                        let v = t.attr("Version");
                        version = FileVersion {
                            text: matches!(t, Token::Start { attrs, .. } if attrs.iter().any(|(k, _)| k == "Version"))
                                .then(|| v.to_owned()),
                            value: qt_to_double(v),
                        };
                        let a = t.attr("ActiveTrial");
                        if !a.is_empty() {
                            active_trial = qt_to_int(a);
                        }
                        let date = t.attr("DateCreated");
                        if !date.is_empty() {
                            self.project.date_created = date.to_owned();
                        }
                        let meta = t.attr("MetaData");
                        if !meta.is_empty() {
                            self.load_project_metadata(meta);
                        }
                        let flip = t.attr("FlipImages");
                        if !flip.is_empty() {
                            old_flip_images = qt_to_int(flip) != 0;
                        }
                        let ct = t.attr("CalibrationType");
                        if !ct.is_empty() {
                            let code = qt_to_int(ct);
                            self.project.calibration_type = CalibrationType::from_code(code)
                                .ok_or_else(|| {
                                    self.xml_error(
                                        i,
                                        format!("calibration type {code} is not known"),
                                    )
                                })?;
                        }
                    }
                    "Camera" => i = self.read_camera(i, old_flip_images)?,
                    "CalibrationObject" => self.read_calibration_object(i)?,
                    "Trial" => i = self.read_trial(i)?,
                    _ => {}
                }
            }
            i += 1;
        }
        if !saw_project {
            self.warn("project.xml has no <Project> element; read as version 0".into());
        }
        Ok((version, active_trial))
    }

    /// The `MetaData` attribute of `<Project>`: the portal study data.
    ///
    /// Ported from core/Project.cpp, Project::setXMLData and Project::parseXMLData (the
    /// camera numbers only; the other fields are derived from the lines kept). The numbers
    /// are read up to the first error, as `QXmlStreamReader` reads them.
    fn load_project_metadata(&mut self, path: &str) {
        let Some(data) = self.files.take(path) else {
            self.files.missing.push(normalize(path));
            return;
        };
        let text = qstring(&data);
        let lines: Vec<String> = text_lines(&text);
        let (meta, error) = crate::portal::parse_lines_with_error(&lines);
        if let Some(e) = error {
            self.warn(format!(
                "portal metadata {}: {} (line {}); read up to there, as XMALab reads it",
                normalize(path),
                e.message,
                e.line
            ));
        }
        self.project.portal_metadata = lines;
        // `Project::parseXMLData`: every camera number of the metadata (XMALab 3 mode,
        // `CppCompat::portal_camera_ids_from_all_files`). Default (ledger F59, F64): the
        // numbers of its calibration files, as a dataset made here from a Portal project or
        // given this metadata numbers its cameras, else every camera number.
        let mut ids: Vec<i32> = if self.options.compat.portal_camera_ids_from_all_files {
            Vec::new()
        } else {
            meta.project_files()
                .iter()
                .map(|f| f.camera_number)
                .collect()
        };
        ids.sort_unstable();
        ids.dedup();
        if ids.is_empty() {
            ids = meta.all_camera_numbers;
        }
        self.camera_ids = ids;
    }

    /// One `<Camera>` element. Returns the index of its end tag.
    fn read_camera(&mut self, start: usize, old_flip_images: bool) -> Result<usize, ReadError> {
        let tokens = self.tokens;
        let t = &tokens[start];
        let mut cam = Camera::new(t.attr("Name").to_owned());
        cam.is_light_camera = qt_to_int(t.attr("isLightCamera")) != 0;
        cam.calibrated = qt_to_int(t.attr("isCalibrated")) != 0;
        let optimized = t.attr("isOptimized");
        if !optimized.is_empty() {
            cam.optimized = qt_to_int(optimized) != 0;
        }
        let flipped = t.attr("isFlipped");
        if !flipped.is_empty() {
            cam.flipped = qt_to_int(flipped) != 0;
        }
        if old_flip_images {
            cam.flipped = true;
        }
        for (v, prefix) in [
            (AdjustedView::Calibration, "Calibration"),
            (AdjustedView::Digitization, "Digitization"),
            (AdjustedView::Detail, "Detail"),
        ] {
            let scale = t.attr(&format!("{prefix}Scale"));
            if !scale.is_empty() {
                cam.image_adjustments[v as usize].scale = qt_to_double(scale);
            }
            let bias = t.attr(&format!("{prefix}Bias"));
            if !bias.is_empty() {
                cam.image_adjustments[v as usize].bias = qt_to_double(bias);
            }
        }
        if cam.calibrated {
            self.load_camera_matrix(&mut cam, t.attr("CameraMatrix"))?;
            let undist = t.attr("UndistortionParameter");
            if !undist.is_empty() {
                self.load_undistortion_param(&mut cam, undist);
            }
        }

        let mut images = CameraImages {
            calibration: Vec::new(),
            grid: None,
        };
        let mut i = start + 1;
        while i < tokens.len() && !tokens[i].is_end("Camera") {
            if let Token::Start { name, .. } = &tokens[i] {
                match name.as_str() {
                    "UndistortionGrid" => {
                        i = self.read_undistortion_grid(i, &mut cam, &mut images)?
                    }
                    "CalibrationSequence" => {
                        let t = &tokens[i];
                        // The video's path as stored (normally absolute), as the C++ keeps
                        // it; it is not an archive entry, so it is not resolved.
                        let filename = t.attr("Filename").to_owned();
                        let frames = self.non_negative(i, qt_to_int(t.attr("Frames")), "Frames")?;
                        let width = self.non_negative(i, qt_to_int(t.attr("Width")), "Width")?;
                        let height = self.non_negative(i, qt_to_int(t.attr("Height")), "Height")?;
                        // Ledger F84 (not in the C++): a sequence beyond `MAX_FRAMES` frames,
                        // or a camera whose calibration frames would pass it, is refused
                        // before any frame is added.
                        let limit = xma_model::project::MAX_FRAMES;
                        if !frames_within_limit(frames) {
                            return Err(self.xml_error(
                                i,
                                format!(
                                    "camera '{}' has a calibration sequence of {frames} frames, more than this program accepts ({limit})",
                                    cam.name
                                ),
                            ));
                        }
                        let total = cam.calibration_images.len().saturating_add(frames);
                        if !frames_within_limit(total) {
                            return Err(self.xml_error(
                                i,
                                format!(
                                    "camera '{}' has {total} calibration frames, more than this program accepts ({limit})",
                                    cam.name
                                ),
                            ));
                        }
                        // Ported from core/CalibrationSequence.cpp,
                        // CalibrationSequence::setCalibrationSequence.
                        for _ in 0..frames {
                            cam.calibration_images
                                .push(CalibrationImage::new(String::new()));
                            images.calibration.push(ImageSize::Missing);
                        }
                        cam.calibration_sequence = Some(CalibrationSequence {
                            filename,
                            n_frames: frames,
                            width: width as u32,
                            height: height as u32,
                        });
                    }
                    "CalibrationImage" => {
                        i = self.read_calibration_image(i, &mut cam, &mut images)?
                    }
                    _ => {}
                }
            }
            i += 1;
        }

        if !cam.calibration_images.is_empty() {
            self.set_resolutions(start, &mut cam, &images)?;
        }

        // Ported from core/Project.cpp, Project::addCamera (portal camera numbers).
        let idx = self.project.cameras.len();
        if let Some(&id) = self.camera_ids.get(idx) {
            cam.name = format!("Camera {id}");
            cam.portal_id = id;
        }
        self.project.cameras.push(Arc::new(cam));
        Ok(i.min(tokens.len().saturating_sub(1)))
    }

    /// Ported from core/Camera.cpp, Camera::loadCameraMatrix.
    fn load_camera_matrix(&mut self, cam: &mut Camera, path: &str) -> Result<(), ReadError> {
        let (entry, data) = self.files.take_or_empty(path);
        cam.camera_matrix = read_matrix3(&entry, &data, "camera matrix", self.files.nonfinite)?;
        Ok(())
    }

    /// Ported from core/Camera.cpp, Camera::loadUndistortionParam.
    fn load_undistortion_param(&mut self, cam: &mut Camera, path: &str) {
        let (_, data) = self.files.take_or_empty(path);
        let rows = value_rows(&data, self.files.nonfinite);
        for y in 0..8 {
            cam.distortion[y] = rows.get(y).map_or(0.0, |(_, r)| r[0]);
        }
        cam.has_model_distortion = true;
    }

    /// One `<UndistortionGrid>`. Returns the index of its end tag.
    ///
    /// Ported from core/Camera.cpp, Camera::loadUndistortionImage, and
    /// core/UndistortionObject.cpp (`setCenter`, `loadPoints*`, `loadGridPointsInlier`).
    fn read_undistortion_grid(
        &mut self,
        start: usize,
        cam: &mut Camera,
        images: &mut CameraImages,
    ) -> Result<usize, ReadError> {
        let tokens = self.tokens;
        let t = &tokens[start];
        let path = t.attr("Filename");
        images.grid = Some(image_size(self.files.peek(path)));
        let mut grid = UndistortionGrid {
            image_filename: file_name(path),
            computed: qt_to_int(t.attr("isComputed")) != 0,
            points_detected: Vec::new(),
            grid_distorted: Vec::new(),
            grid_references: Vec::new(),
            grid_inlier: Vec::new(),
            center: None,
        };
        let mut i = start;
        while i < tokens.len() && !tokens[i].is_end("UndistortionGrid") {
            let t = &tokens[i];
            if let Token::Start { name, .. } = t {
                match name.as_str() {
                    "Center" => {
                        grid.center = Some(Point2::new(
                            qt_to_double(t.attr("x")),
                            qt_to_double(t.attr("y")),
                        ));
                    }
                    "PointsDetected" => {
                        grid.points_detected = self.load_points(t.attr("Filename"))?
                    }
                    "GridPointsDistorted" => {
                        grid.grid_distorted = self.load_points(t.attr("Filename"))?
                    }
                    "GridPointsReferences" => {
                        grid.grid_references = self.load_points(t.attr("Filename"))?
                    }
                    "GridPointsInlier" => {
                        let (_, data) = self.files.take_or_empty(t.attr("Filename"));
                        grid.grid_inlier = value_rows(&data, false)
                            .iter()
                            .map(|(_, r)| r[0] > 0.5)
                            .collect();
                    }
                    _ => {}
                }
            }
            i += 1;
        }
        cam.undistortion = Some(grid);
        Ok(i.min(tokens.len().saturating_sub(1)))
    }

    /// One `<CalibrationImage>`. Returns the index of its end tag.
    ///
    /// Ported from ui/ProjectFileIO.cpp, ProjectFileIO::readProjectFile, and
    /// core/CalibrationImage.cpp (`loadPoints*`, `loadRotationMatrix`,
    /// `loadTranslationVector`).
    fn read_calibration_image(
        &mut self,
        start: usize,
        cam: &mut Camera,
        images: &mut CameraImages,
    ) -> Result<usize, ReadError> {
        let tokens = self.tokens;
        let t = &tokens[start];
        let text = t.attr("Filename");
        let index = if text == "external" {
            cam.calibration_images
                .push(CalibrationImage::new("external".into()));
            images
                .calibration
                .push(image_size(self.files.peek("external")));
            cam.width = self.non_negative(start, qt_to_int(t.attr("Width")), "Width")? as u32;
            cam.height = self.non_negative(start, qt_to_int(t.attr("Height")), "Height")? as u32;
            cam.calibration_images.len() - 1
        } else if !text.is_empty() {
            cam.calibration_images
                .push(CalibrationImage::new(file_name(text)));
            images.calibration.push(image_size(self.files.peek(text)));
            cam.calibration_images.len() - 1
        } else {
            let frame = qt_to_int(t.attr("Frame"));
            usize::try_from(frame)
                .ok()
                .filter(|&f| f < cam.calibration_images.len())
                .ok_or_else(|| {
                    self.xml_error(
                        start,
                        format!(
                            "calibration image frame {frame} does not exist ({} images)",
                            cam.calibration_images.len()
                        ),
                    )
                })?
        };
        cam.calibration_images[index].calibrated = qt_to_int(t.attr("isCalibrated"));
        let mut i = start;
        while i < tokens.len() && !tokens[i].is_end("CalibrationImage") {
            let t = &tokens[i];
            if let Token::Start { name, .. } = t {
                let path = t.attr("Filename");
                match name.as_str() {
                    "PointsDetectedAll" => {
                        cam.calibration_images[index].detected_all = self.load_points(path)?
                    }
                    "PointsDetected" => {
                        cam.calibration_images[index].detected = self.load_points(path)?
                    }
                    "Inlier" => {
                        let (_, data) = self.files.take_or_empty(path);
                        cam.calibration_images[index].inlier = value_rows(&data, false)
                            .iter()
                            .map(|(_, r)| c_int(r[0]))
                            .collect();
                    }
                    "RotationMatrix" => {
                        let (entry, data) = self.files.take_or_empty(path);
                        let m =
                            read_matrix3(&entry, &data, "rotation matrix", self.files.nonfinite)?;
                        cam.calibration_images[index].rotation = matrix_to_rodrigues(&m);
                    }
                    "TranslationVector" => {
                        let (entry, data) = self.files.take_or_empty(path);
                        let rows = value_rows(&data, self.files.nonfinite);
                        if rows.len() < 3 {
                            return Err(ReadError::data(
                                &entry,
                                None,
                                None,
                                format!("a translation vector needs 3 rows; found {}", rows.len()),
                            ));
                        }
                        cam.calibration_images[index].translation =
                            Vector3::new(rows[0].1[0], rows[1].1[0], rows[2].1[0]);
                    }
                    _ => {}
                }
            }
            i += 1;
        }
        Ok(i.min(tokens.len().saturating_sub(1)))
    }

    /// `UndistortionObject::loadPoints` and `CalibrationImage::loadPoints`: one point per
    /// row that has values.
    fn load_points(&mut self, path: &str) -> Result<Vec<Point2>, ReadError> {
        let (entry, data) = self.files.take_or_empty(path);
        value_rows(&data, self.files.nonfinite)
            .into_iter()
            .map(|(line, r)| {
                if r.len() < 2 {
                    Err(ReadError::data(
                        &entry,
                        Some(line),
                        Some(2),
                        "a point needs 2 values",
                    ))
                } else {
                    Ok(Point2::new(r[0], r[1]))
                }
            })
            .collect()
    }

    /// Ported from core/Camera.cpp, Camera::setResolutions (the resolution and its check;
    /// the lens undistortion map it also builds is derived).
    fn set_resolutions(
        &mut self,
        start: usize,
        cam: &mut Camera,
        images: &CameraImages,
    ) -> Result<(), ReadError> {
        let external = self.project.calibration_type == CalibrationType::External;
        let mut unknown = false;
        if !external {
            let dims = match &cam.calibration_sequence {
                Some(s) => Some((s.width, s.height)),
                None => images.calibration[0].dims(),
            };
            match dims {
                Some((w, h)) => {
                    cam.width = w;
                    cam.height = h;
                }
                None => unknown = true,
            }
        }
        let mut mismatch = false;
        if !external && cam.calibration_sequence.is_none() {
            for s in &images.calibration {
                match s.dims() {
                    Some(d) => mismatch |= !unknown && d != (cam.width, cam.height),
                    None => unknown = true,
                }
            }
        }
        if let Some(g) = images.grid {
            match g.dims() {
                Some(d) => mismatch |= d != (cam.width, cam.height),
                None => unknown = true,
            }
        }
        if unknown {
            self.warn(format!(
                "{}: an image format was not recognised, so its resolution ({}x{} used) was not checked",
                cam.name, cam.width, cam.height
            ));
        }
        if mismatch && !unknown {
            return Err(self.xml_error(start, format!("{} : Resolutions do not match", cam.name)));
        }
        Ok(())
    }

    /// Ported from ui/ProjectFileIO.cpp, ProjectFileIO::readProjectFile (`CalibrationObject`),
    /// and core/CalibrationObject.cpp, CalibrationObject::loadCoords.
    fn read_calibration_object(&mut self, index: usize) -> Result<(), ReadError> {
        let t = &self.tokens[index];
        if qt_to_int(t.attr("isPLanar")) != 0 {
            let h = self.non_negative(
                index,
                qt_to_int(t.attr("HorizontalSquares")),
                "HorizontalSquares",
            )?;
            let v = self.non_negative(
                index,
                qt_to_int(t.attr("VerticalSquares")),
                "VerticalSquares",
            )?;
            // Ledger F74: a board beyond `MAX_CHECKERBOARD_POINTS` corners is refused (the
            // C++ builds every point).
            if (h as u64) * (v as u64) > xma_model::project::MAX_CHECKERBOARD_POINTS {
                return Err(self.xml_error(
                    index,
                    format!(
                        "a checkerboard of {h} x {v} corners is larger than this program accepts ({} corners)",
                        xma_model::project::MAX_CHECKERBOARD_POINTS
                    ),
                ));
            }
            // `setCheckerboard` reads `CheckerboadInvertedAxis` (the settings are read
            // first). Ledger F73: the board's own axis, written by this program, when
            // present.
            let setting = self.project.algorithm_settings.checkerboard_inverted_axis;
            let stored = t.attr("InvertedAxis");
            let inverted_axis =
                if !self.options.compat.checkerboard_axis_from_setting && !stored.is_empty() {
                    self.checkerboard_axis_stored = true;
                    qt_to_int(stored)
                } else {
                    setting
                };
            self.project.calibration_object = Some(CalibrationObject::Checkerboard {
                horizontal_squares: h as u32,
                vertical_squares: v as u32,
                square_size: qt_to_double(t.attr("SquareSize")),
                inverted_axis,
            });
            return Ok(());
        }
        let spec_path = t.attr("FrameSpecifications").to_owned();
        let ref_path = t.attr("References").to_owned();
        // The singleton keeps `whiteBlobs` and the scale from an earlier load; a project
        // file with one calibration object starts from the constructor's values.
        let mut frame = self
            .calibration_frame
            .take()
            .unwrap_or_else(crate::calibration_object::empty_frame);
        frame.frame_specification_filename = file_name(&spec_path);
        frame.references_filename = file_name(&ref_path);
        let (entry, spec) = self.files.take_or_empty(&spec_path);
        let (_, refs) = self.files.take_or_empty(&ref_path);
        crate::calibration_object::load_coords_into(&mut frame, &spec, &refs)
            .map_err(|e| ReadError::data(&entry, Some(e.line), Some(e.column), e.message))?;
        self.project.calibration_object = Some(CalibrationObject::Frame(frame.clone()));
        self.calibration_frame = Some(frame);
        Ok(())
    }

    /// One `<Trial>` element. Returns the index of its end tag.
    fn read_trial(&mut self, start: usize) -> Result<usize, ReadError> {
        let tokens = self.tokens;
        let t = &tokens[start];
        let name = t.attr("Name").to_owned();
        let folder = format!("{name}/");
        let n_cameras = self.project.cameras.len();
        let mut source = TrialSource {
            n_frames_from_file: !t.attr("nbImages").is_empty(),
            recording_speed_from_file: !t.attr("recordingSpeed").is_empty(),
            markers_without_element: Vec::new(),
        };
        let mut trial = if name == "Default" {
            // Ported from core/Trial.cpp, Trial::Trial().
            Trial::default_trial()
        } else {
            self.construct_trial(&name, &folder, &source)?
        };
        let constructed_default = trial.is_default;
        if constructed_default {
            source.n_frames_from_file = true;
            source.recording_speed_from_file = true;
        }

        trial.start_frame =
            self.non_negative(start, qt_to_int(t.attr("startFrame")), "startFrame")?;
        trial.end_frame = self.non_negative(start, qt_to_int(t.attr("endFrame")), "endFrame")?;
        // Ledger F68: -1 is what XMALab's Trial dialog stores without a frame calibrated
        // in every camera.
        let reference = qt_to_int(t.attr("referenceCalibration"));
        trial.reference_calibration_image = if reference == -1 {
            xma_model::trial::NO_REFERENCE_CALIBRATION
        } else {
            self.non_negative(start, reference, "referenceCalibration")?
        };
        let speed = t.attr("recordingSpeed");
        if !speed.is_empty() {
            trial.recording_speed = qt_to_double(speed);
        }
        let cutoff = t.attr("cutOffFrequency");
        if !cutoff.is_empty() {
            trial.cutoff_frequency = qt_to_double(cutoff);
        }
        let interp = t.attr("interpolate3D");
        if !interp.is_empty() {
            trial.interpolate_3d = qt_to_int(interp) != 0;
        }
        let nb = t.attr("nbImages");
        if !nb.is_empty() && !constructed_default {
            // Trial::setNbImagesFromConfig does nothing for the default trial.
            trial.n_frames = self.non_negative(start, qt_to_int(nb), "nbImages")?;
        }
        let (elements, end) = trial_elements(tokens, start);
        let (planned, planned_exact) =
            self.check_trial_size(start, &name, &folder, trial.n_frames, n_cameras, &elements)?;

        let mut markers = self.load_markers(
            start,
            &name,
            &format!("{folder}data/MarkerDescription.txt"),
            n_cameras,
            trial.n_frames,
        )?;
        let bodies = self.load_rigid_bodies(
            start,
            &name,
            &format!("{folder}data/RigidBodies.txt"),
            &mut markers,
            trial.n_frames,
        )?;

        let meta = t.attr("MetaData");
        if !meta.is_empty() {
            match self.files.take(meta) {
                Some(d) => trial.portal_metadata = text_lines(&qstring(&d)),
                None => self.files.missing.push(normalize(meta)),
            }
        }
        let is_default = t.attr("Default");
        if !is_default.is_empty() {
            trial.is_default = qt_to_int(is_default) != 0;
        }
        let from_default = t.attr("FromDefault");
        if !from_default.is_empty() {
            trial.is_copy_from_default = qt_to_int(from_default) != 0;
        }

        let marker_cells = product(&[n_cameras, trial.n_frames, markers.len()]);
        let mut raw = RawTrial {
            trial,
            markers,
            bodies,
            source,
            cells: TrialCells {
                markers: marker_cells.unwrap_or(u64::MAX),
                ..TrialCells::default()
            },
            planned,
            planned_exact,
        };
        for element in &elements {
            match element {
                TrialElement::Event(i) => self.read_event(*i, &mut raw)?,
                TrialElement::Marker(i) => self.read_marker(*i, &mut raw)?,
                TrialElement::RigidBody { start, dummies } => {
                    self.read_rigid_body(*start, dummies, &mut raw)?;
                }
            }
        }
        self.trials.push(raw);
        Ok(end)
    }

    /// Ledger F84 (not in the C++): before any per-frame array of the trial is allocated,
    /// refuses a trial of more than `MAX_FRAMES` frames; one where an event, a marker or a
    /// virtual point would reach more than `MAX_FRAMES` frames from the rows of its data
    /// file; one whose events, virtual points (`<DummyMarker>` in its `<RigidBody>`
    /// elements), or cameras x markers, times the frames each reaches, pass
    /// `MAX_TRIAL_CELLS`; and one whose frames x rigid bodies (non-empty lines of
    /// `RigidBodies.txt`) pass `MAX_TRIAL_BODY_CELLS`.
    ///
    /// The elements are those [`Self::read_trial`] reads (`elements`, from
    /// [`trial_elements`]). The length each array reaches is bounded first by the lines of
    /// its file; when those bounds pass a limit, it is computed exactly, by functions that
    /// follow the reader's rules for growing the arrays ([`event_length`],
    /// [`marker_2d_length`], [`coordinate_rows`]), on the same archive entries, and the
    /// refusal rests on those. Returns the lengths used (bounds or exact) by the index of
    /// each element's start tag (an `<Event>`, a `<Marker>` of a marker that exists, a
    /// `<DummyMarker>`), and whether they are exact: the reader refuses a row beyond them,
    /// reserves them only when they are exact (a bound counts lines that hold no row, so
    /// below the limits the arrays grow as rows are read), and checks the totals again as
    /// it allocates ([`TrialCells`]), so nothing is allocated beyond what was counted here.
    fn check_trial_size(
        &self,
        start: usize,
        name: &str,
        folder: &str,
        n_frames: usize,
        n_cameras: usize,
        elements: &[TrialElement],
    ) -> Result<(HashMap<usize, usize>, bool), ReadError> {
        use xma_model::project::{MAX_TRIAL_BODY_CELLS, MAX_TRIAL_CELLS};
        if !frames_within_limit(n_frames) {
            return Err(self.xml_error(
                start,
                format!(
                    "trial '{name}' has {n_frames} frames, more than this program accepts ({})",
                    xma_model::project::MAX_FRAMES
                ),
            ));
        }
        let peek = |path: &str| self.files.peek(path).unwrap_or(&[]);
        let n_markers = count_lines(peek(&format!("{folder}data/MarkerDescription.txt")), |_| {
            true
        });
        let n_bodies = count_lines(peek(&format!("{folder}data/RigidBodies.txt")), |l| {
            !l.is_empty()
        });
        // A cheap pass first, with each array's length bounded by the line ends of its file
        // (every line a row); only when that passes a limit, the exact lengths, which the
        // refusal then rests on.
        let mut plan = self.plan_trial(n_frames, n_cameras, n_markers, elements, false);
        if !plan.within_limits() {
            plan = self.plan_trial(n_frames, n_cameras, n_markers, elements, true);
        }
        if let Some((index, what, file, len)) = &plan.too_long {
            self.check_array(*index, name, what, file, *len)?;
        }
        let TrialPlan {
            planned,
            exact,
            events,
            event_cells,
            virtual_points,
            virtual_point_cells,
            marker_cells,
            ..
        } = plan;
        self.check_total(
            start,
            name,
            "frames x events",
            &[n_frames, events],
            event_cells,
            MAX_TRIAL_CELLS,
        )?;
        self.check_total(
            start,
            name,
            "frames x virtual points",
            &[n_frames, virtual_points],
            virtual_point_cells,
            MAX_TRIAL_CELLS,
        )?;
        self.check_cells(
            start,
            name,
            "frames x rigid bodies",
            &[n_frames, n_bodies],
            MAX_TRIAL_BODY_CELLS,
        )?;
        self.check_total(
            start,
            name,
            "cameras x frames x markers",
            &[n_cameras, n_frames, n_markers],
            marker_cells,
            MAX_TRIAL_CELLS,
        )?;
        Ok((planned, exact))
    }

    /// Ledger F84: the frames each array of the trial reaches from the rows of its data
    /// file, for [`Self::check_trial_size`]. `exact`: computed as the reader grows the arrays
    /// ([`event_length`], [`marker_2d_length`], [`coordinate_rows`]); otherwise bounded by
    /// [`line_bound`], never less than the lines of the file (a row is a line).
    fn plan_trial(
        &self,
        n_frames: usize,
        n_cameras: usize,
        n_markers: usize,
        elements: &[TrialElement],
        exact: bool,
    ) -> TrialPlan {
        let peek = |path: &str| self.files.peek(path).unwrap_or(&[]);
        let n_lines = line_bound;
        let mut plan = TrialPlan {
            exact,
            ..TrialPlan::default()
        };
        let too_long = |plan: &mut TrialPlan, index: usize, what: String, file: &str, len| {
            if plan.too_long.is_none() && !frames_within_limit(len) {
                plan.too_long = Some((index, what, file.to_owned(), len));
            }
        };
        let mut marker_frames = Vec::new();
        for element in elements {
            match element {
                TrialElement::Event(i) => {
                    let t = &self.tokens[*i];
                    let file = t.attr("Filename");
                    let len = if exact {
                        event_length(peek(file), n_frames)
                    } else {
                        n_lines(peek(file)).max(n_frames)
                    };
                    too_long(
                        &mut plan,
                        *i,
                        format!("event '{}'", t.attr("Name")),
                        file,
                        len,
                    );
                    plan.planned.insert(*i, len);
                    plan.events += 1;
                    plan.event_cells = plan.event_cells.saturating_add(len as u64);
                }
                TrialElement::Marker(i) => {
                    if marker_frames.is_empty() && n_markers > 0 {
                        marker_frames = vec![n_frames; n_markers];
                    }
                    let t = &self.tokens[*i];
                    let id = usize::try_from(qt_to_int(t.attr("ID"))).ok();
                    // A marker that does not exist is refused by the reader.
                    if let Some(len) = id.and_then(|m| marker_frames.get_mut(m)) {
                        let file = t.attr("FilenamePoints2D");
                        *len = if exact {
                            marker_2d_length(peek(file), self.files.nonfinite, n_cameras, *len)
                        } else {
                            n_lines(peek(file)).max(*len)
                        };
                        let what = format!("marker {}", qt_to_int(t.attr("ID")) + 1);
                        too_long(&mut plan, *i, what, file, *len);
                        plan.planned.insert(*i, *len);
                    }
                }
                TrialElement::RigidBody { dummies, .. } => {
                    for &d in dummies {
                        let t = &self.tokens[d];
                        let file = t.attr("PointCoordinates");
                        let len = if !reads_coordinates(peek(t.attr("PointReferences"))) {
                            n_frames
                        } else if exact {
                            coordinate_rows(peek(file)).max(n_frames)
                        } else {
                            n_lines(peek(file)).saturating_sub(1).max(n_frames)
                        };
                        let what = format!("virtual point '{}'", t.attr("Name"));
                        too_long(&mut plan, d, what, file, len);
                        plan.planned.insert(d, len);
                        plan.virtual_points += 1;
                        plan.virtual_point_cells =
                            plan.virtual_point_cells.saturating_add(len as u64);
                    }
                }
            }
        }
        plan.marker_cells = if marker_frames.is_empty() {
            product(&[n_cameras, n_frames, n_markers]).unwrap_or(u64::MAX)
        } else {
            marker_frames
                .iter()
                .fold(0u64, |acc, &f| acc.saturating_add(f as u64))
                .saturating_mul(n_cameras as u64)
        };
        plan
    }

    /// Ledger F84: refuses an array of the trial `trial` (`what`, as `"event 'E'"`) that
    /// would reach `len` frames, more than `MAX_FRAMES`, from the rows of `file`.
    fn check_array(
        &self,
        index: usize,
        trial: &str,
        what: &str,
        file: &str,
        len: usize,
    ) -> Result<(), ReadError> {
        if frames_within_limit(len) {
            return Ok(());
        }
        Err(self.xml_error(
            index,
            format!(
                "trial '{trial}': {what} would have {len} frames from the rows of '{}', more \
                 than this program accepts ({})",
                normalize(file),
                xma_model::project::MAX_FRAMES
            ),
        ))
    }

    /// Ledger F84: refuses a trial whose cells of one family (`what`, as
    /// `"frames x events"`) number `total` (the frames each array reaches, rows beyond the
    /// trial's frame count included), more than `limit`. When no array grows beyond the
    /// trial's frames, `total` is the product of `counts` and the sentence gives them.
    fn check_total(
        &self,
        start: usize,
        name: &str,
        what: &str,
        counts: &[usize],
        total: u64,
        limit: u64,
    ) -> Result<(), ReadError> {
        if total <= limit {
            return Ok(());
        }
        if product(counts) == Some(total) {
            return self.check_cells(start, name, what, counts, limit);
        }
        Err(self.xml_error(
            start,
            format!(
                "trial '{name}' is too large: {what}, counted with the rows of its data files, \
                 is {total}, more than this program accepts ({limit})"
            ),
        ))
    }

    /// Ledger F84: refuses a trial whose `counts` (named by `what`, as `"frames x events"`)
    /// multiply to more than `limit`.
    fn check_cells(
        &self,
        start: usize,
        name: &str,
        what: &str,
        counts: &[usize],
        limit: u64,
    ) -> Result<(), ReadError> {
        if product(counts).is_some_and(|p| p <= limit) {
            return Ok(());
        }
        let values: Vec<String> = counts.iter().map(usize::to_string).collect();
        Err(self.xml_error(
            start,
            format!(
                "trial '{name}' is too large: {what} is {}, more than this program accepts \
                 ({limit})",
                values.join(" x "),
            ),
        ))
    }

    /// Ledger F84, at the point of allocation: adds `by` cells to `used`, the running total
    /// of one family of the trial `trial`, and refuses when it would pass `limit`, or when
    /// the array (`what`) would reach `len` frames, more than `MAX_FRAMES`. The check in
    /// [`Self::check_trial_size`] refuses the same files first, from the same counts; this
    /// one makes sure nothing is allocated past a limit even if the two ever disagreed.
    #[allow(clippy::too_many_arguments)]
    fn allocate_cells(
        &self,
        index: usize,
        trial: &str,
        what: &str,
        len: usize,
        used: &mut u64,
        by: u64,
        limit: u64,
    ) -> Result<(), ReadError> {
        let total = used.checked_add(by).filter(|&t| t <= limit);
        match total {
            Some(t) if frames_within_limit(len) => {
                *used = t;
                Ok(())
            }
            _ => Err(self.xml_error(
                index,
                format!(
                    "trial '{trial}' is too large: {what} would have {len} frames, which passes \
                     a limit of this program ({} frames, {limit} cells)",
                    xma_model::project::MAX_FRAMES
                ),
            )),
        }
    }

    /// Ported from core/Trial.cpp, Trial::Trial(QString trialname, QString folder): the
    /// video file lists and, where the file lacks them, frame count and rate from the first
    /// camera's video.
    fn construct_trial(
        &mut self,
        name: &str,
        folder: &str,
        source: &TrialSource,
    ) -> Result<Trial, ReadError> {
        let mut trial = Trial::default_trial();
        trial.name = name.to_owned();
        trial.is_default = false;
        trial.requires_recomputation = true;
        trial.n_frames = 0;
        for cam in &self.project.cameras {
            let path = format!("{folder}{}_filenames_absolute.txt", cam.name);
            let (_, data) = self.files.take_or_empty(&path);
            let filenames: Vec<String> = lines(&data).into_iter().map(qstring).collect();
            if filenames.is_empty() {
                // XMALab reads `filenameList.at(0)` of the empty list here (core/Trial.cpp:154),
                // undefined behaviour (a crash), so it cannot open such a file; it never
                // writes one either, as only its Default trial has no videos. This program
                // writes one for a trial built without videos (a new or synthetic project),
                // so it reads it back: the trial has no video for this camera, and its frame
                // count and rate come from the project file.
                self.warnings.push(format!(
                    "trial '{name}' has no video file for {} (XMALab cannot open such a \
                     project); the trial is read without images",
                    cam.name
                ));
            }
            trial.videos.push(VideoSource {
                filenames,
                metadata: Vec::new(),
            });
        }
        let Some(first) = trial.videos.first() else {
            return Err(ReadError::data(
                XML,
                None,
                None,
                format!("trial '{name}' has no cameras"),
            ));
        };
        // Trial::setNbImages and the video readers: an image sequence has one frame per
        // file and no rate; a video file reports both when it opens. No file: nothing
        // (the project file's values apply).
        let (n, fps) = if first.filenames.is_empty() {
            (0, 0.0)
        } else if first.filenames.len() > 1 || !is_video_file(&first.filenames[0]) {
            (first.filenames.len() as i32, 0.0)
        } else if source.n_frames_from_file && source.recording_speed_from_file {
            (0, 0.0)
        } else {
            match self.options.video_info.and_then(|f| f(first)) {
                Some(info) => (info.n_frames, info.fps),
                None => {
                    self.warnings.push(format!(
                        "trial '{name}': the file does not store {}; the C++ reads it from {}, which was not opened here (0 used)",
                        if source.n_frames_from_file { "the recording speed" } else { "the frame count" },
                        first.filenames[0]
                    ));
                    (-1, 0.0)
                }
            }
        };
        trial.n_frames = usize::try_from(n).unwrap_or(0);
        trial.recording_speed = fps;
        trial.start_frame = 1;
        trial.end_frame = trial.n_frames;
        Ok(trial)
    }

    /// Ported from core/Trial.cpp, Trial::loadMarkers. Ledger F84 (not in the C++): cameras
    /// x frames x description lines beyond `MAX_TRIAL_CELLS` is refused before any marker is
    /// built.
    fn load_markers(
        &mut self,
        start: usize,
        trial: &str,
        path: &str,
        n_cameras: usize,
        n_frames: usize,
    ) -> Result<Vec<RawMarker>, ReadError> {
        let (_, data) = self.files.take_or_empty(path);
        let threshold = self.project.algorithm_settings.default_marker_threshold;
        let lines = lines(&data);
        self.check_cells(
            start,
            trial,
            "cameras x frames x markers",
            &[n_cameras, n_frames, lines.len()],
            xma_model::project::MAX_TRIAL_CELLS,
        )?;
        Ok(lines
            .into_iter()
            .map(|l| {
                let mut m = RawMarker::new(n_cameras, n_frames, threshold);
                m.description = qstring(l);
                m
            })
            .collect())
    }

    /// Ported from core/Trial.cpp, Trial::loadRigidBodies. Ledger F84 (not in the C++):
    /// frames x rigid bodies (non-empty lines) beyond `MAX_TRIAL_BODY_CELLS` is refused
    /// before any body is built (and, by [`Self::check_trial_size`], before the markers are).
    fn load_rigid_bodies(
        &mut self,
        start: usize,
        trial: &str,
        path: &str,
        markers: &mut [RawMarker],
        n_frames: usize,
    ) -> Result<Vec<RigidBody>, ReadError> {
        let (entry, data) = self.files.take_or_empty(path);
        let lines = lines(&data);
        let n_bodies = lines.iter().filter(|l| !l.is_empty()).count();
        self.check_cells(
            start,
            trial,
            "frames x rigid bodies",
            &[n_frames, n_bodies],
            xma_model::project::MAX_TRIAL_BODY_CELLS,
        )?;
        let mut bodies = Vec::new();
        for (k, line) in lines.into_iter().enumerate() {
            if line.is_empty() {
                continue;
            }
            let mut body = RigidBody::new(n_frames);
            let open = line.iter().position(|&b| b == b'[');
            let close = line.iter().position(|&b| b == b']');
            body.description = qstring(&line[..open.unwrap_or(line.len())]);
            // `line.substr(line.find('[') + 1, line.find(']') - 1)`, with npos arithmetic.
            let from = open.map_or(0, |p| p + 1);
            let count = close.map_or(usize::MAX - 1, |c| c.wrapping_sub(1));
            let indices = &line[from..from.saturating_add(count).min(line.len())];
            clear_point_idx(&mut body, markers);
            for value in stream_ints(indices) {
                let idx = usize::try_from(value - 1)
                    .ok()
                    .filter(|&x| x < markers.len())
                    .ok_or_else(|| {
                        ReadError::data(
                            &entry,
                            Some(k + 1),
                            None,
                            format!("marker {value} does not exist ({} markers)", markers.len()),
                        )
                    })?;
                add_point_idx(&mut body, idx, markers);
            }
            bodies.push(body);
        }
        Ok(bodies)
    }

    /// Ported from ui/ProjectFileIO.cpp, ProjectFileIO::readProjectFile (`Event`), and
    /// core/EventData.cpp, EventData::loadData. Ledger F84: the flags, rows beyond the
    /// trial's frames included, are counted before they are allocated.
    fn read_event(&mut self, index: usize, raw: &mut RawTrial) -> Result<(), ReadError> {
        let t = &self.tokens[index];
        let name = t.attr("Name").to_owned();
        let colour_text = t.attr("Color");
        let color = qt_color(colour_text).unwrap_or_else(|| {
            self.warnings.push(format!(
                "event '{name}': colour '{colour_text}' not understood; black used"
            ));
            Rgb::new(0, 0, 0)
        });
        let (_, bytes) = self.files.take_or_empty(t.attr("Filename"));
        let n_frames = raw.trial.n_frames;
        let len = raw.planned.get(&index).copied().unwrap_or(n_frames);
        self.allocate_cells(
            index,
            &raw.trial.name,
            &format!("event '{name}'"),
            len,
            &mut raw.cells.events,
            len as u64,
            xma_model::project::MAX_TRIAL_CELLS,
        )?;
        // Reserved only when `len` is exact (ledger F84); otherwise as before F84.
        let mut data = Vec::with_capacity(if raw.planned_exact { len } else { n_frames });
        data.resize(n_frames, false);
        for (lc, line) in lines(&bytes).into_iter().enumerate() {
            let v = stream_values(line);
            if let Some(&first) = v.first() {
                let flag = c_int(first) != 0;
                if data.len() <= lc {
                    if data.len() >= len {
                        return Err(uncounted_row(&normalize(t.attr("Filename")), lc));
                    }
                    data.push(flag);
                } else {
                    data[lc] = flag;
                }
            }
        }
        raw.trial.events.push(Event {
            name,
            color,
            draw: qt_to_int(t.attr("Draw")) != 0,
            data,
        });
        Ok(())
    }

    /// Ported from ui/ProjectFileIO.cpp, ProjectFileIO::readProjectFile (`Marker`).
    fn read_marker(&mut self, index: usize, raw: &mut RawTrial) -> Result<(), ReadError> {
        let t = &self.tokens[index];
        let id = qt_to_int(t.attr("ID"));
        let n_markers = raw.markers.len();
        let Some(m) = usize::try_from(id)
            .ok()
            .and_then(|i| raw.markers.get_mut(i))
        else {
            return Err(self.xml_error(
                index,
                format!("marker ID {id} does not exist ({n_markers} markers)"),
            ));
        };
        m.origin = normalize(t.attr("FilenameStatus2D"));
        m.has_element = true;
        // Ledger F84: the frames the 2D points add beyond the marker's are counted before
        // they are allocated, and reserved when the count is exact.
        // Charged as the count before reading charged it: the frames planned beyond those
        // planned for this marker before.
        let n_cams = m.points2d.len();
        let before = m.n_frames();
        let len = raw.planned.get(&index).copied().unwrap_or(before);
        self.allocate_cells(
            index,
            &raw.trial.name,
            &format!("marker {}", id + 1),
            len,
            &mut raw.cells.markers,
            (len.saturating_sub(m.planned_frames) as u64).saturating_mul(n_cams as u64),
            xma_model::project::MAX_TRIAL_CELLS,
        )?;
        m.planned_frames = m.planned_frames.max(len);
        if raw.planned_exact {
            m.reserve_frames(len.saturating_sub(before));
        }
        load_marker_2d(
            &mut self.files,
            m,
            len,
            t.attr("FilenamePoints2D"),
            t.attr("FilenameStatus2D"),
            t.attr("FilenameSize"),
        )?;
        let reference = t.attr("Reference3DPoint");
        if !reference.is_empty() {
            // Ported from core/Marker.cpp, Marker::loadReference3DPoint: the first line only.
            let (_, data) = self.files.take_or_empty(reference);
            let (line, _) = LineReader::new(&data).next_line();
            let v = values(line, self.files.nonfinite);
            if v.len() == 3 {
                m.reference_3d = Some(Point3::new(v[0], v[1], v[2]));
            }
        }
        let penalty = t.attr("TrackingPenalty");
        if !penalty.is_empty() {
            m.penalty_attribute = Some(qt_to_int(penalty));
            m.set_max_penalty(qt_to_int(penalty));
        }
        let method = t.attr("DetectionMethod");
        if !method.is_empty() {
            m.method = qt_to_int(method);
        }
        let legacy = t.attr("InterpolationMethod");
        if !legacy.is_empty() {
            let new_method = match qt_to_int(legacy) {
                1 | 4 => 10,
                // 3 and 6 were cubic; the C++ maps them to linear (20) as well.
                2 | 3 | 5 | 6 => 20,
                _ => 0,
            };
            for f in 0..raw.trial.n_frames.min(m.interpolation.len()) {
                m.interpolation[f] = new_method;
            }
        }
        let size_override = t.attr("SizeOverride");
        if !size_override.is_empty() {
            m.size_override = qt_to_int(size_override);
        }
        let threshold = t.attr("ThresholdOffset");
        if !threshold.is_empty() {
            m.threshold_offset = qt_to_int(threshold);
        }
        let rr = t.attr("RequiresRecomputation");
        let p3 = t.attr("FilenamePoints3D");
        let s3 = t.attr("FilenameStatus3D");
        if !rr.is_empty() && !p3.is_empty() && !s3.is_empty() {
            m.requires_recomputation = qt_to_int(rr) != 0;
            load_marker_3d(&mut self.files, m, p3, s3)?;
        }
        let interp = t.attr("FilenameInterpolation");
        if !interp.is_empty() {
            // Ported from core/Marker.cpp, Marker::loadInterpolation.
            let (entry, data) = self.files.take_or_empty(interp);
            for (lc, line) in lines(&data).into_iter().enumerate() {
                if let Some(&v) = stream_values(line).first() {
                    let n = m.interpolation.len();
                    let slot = m.interpolation.get_mut(lc).ok_or_else(|| {
                        ReadError::data(
                            &entry,
                            Some(lc + 1),
                            None,
                            format!("more rows than the marker's {n} frames"),
                        )
                    })?;
                    *slot = c_int(v);
                }
            }
        }
        Ok(())
    }

    /// Ported from ui/ProjectFileIO.cpp, ProjectFileIO::readProjectFile (`RigidBody`), and
    /// core/RigidBody.cpp (`load`, `loadOptimized`, `addDummyPoint`). `dummies` are the
    /// `<DummyMarker>` start tags read with it ([`trial_elements`]).
    fn read_rigid_body(
        &mut self,
        start: usize,
        dummies: &[usize],
        raw: &mut RawTrial,
    ) -> Result<(), ReadError> {
        let tokens = self.tokens;
        let t = &tokens[start];
        let id = qt_to_int(t.attr("ID"));
        let n_bodies = raw.bodies.len();
        let Some(body_index) = usize::try_from(id).ok().filter(|&b| b < n_bodies) else {
            return Err(self.xml_error(
                start,
                format!("rigid body ID {id} does not exist ({n_bodies} bodies)"),
            ));
        };
        let names = t.attr("ReferenceNames");
        let points = t.attr("ReferencePoints3D");
        let body = &mut raw.bodies[body_index];
        if !names.is_empty() && !points.is_empty() {
            body.has_optimized_coordinates = false;
            let (_, data) = self.files.take_or_empty(points);
            body.reference_points = read_xyz_lines(&data);
            body.reference_points_original = body.reference_points.clone();
            let (_, data) = self.files.take_or_empty(names);
            body.reference_names = lines(&data)
                .into_iter()
                .filter(|l| !l.is_empty())
                .map(qstring)
                .collect();
            body.references_set = 2;
        } else {
            reset_references(body, &raw.markers);
        }
        let optimized = t.attr("ReferencePoints3DOptimized");
        if !optimized.is_empty() {
            body.has_optimized_coordinates = true;
            let n = body.reference_points.len();
            if body.reference_points_original.len() < n {
                return Err(self.xml_error(
                    start,
                    "rigid body has fewer original reference points than points",
                ));
            }
            body.reference_points_original[..n].copy_from_slice(&body.reference_points);
            let (_, data) = self.files.take_or_empty(optimized);
            body.reference_points = read_xyz_lines(&data);
        }
        let visible = t.attr("Visible");
        if !visible.is_empty() {
            body.visible = qt_to_int(visible) != 0;
        }
        let colour = t.attr("Color");
        if !colour.is_empty() {
            body.color = qt_color(colour).unwrap_or_else(|| {
                self.warnings.push(format!(
                    "rigid body {id}: colour '{colour}' not understood; black used"
                ));
                Rgb::new(0, 0, 0)
            });
        }
        let ov = t.attr("OverrideCutOffFrequency");
        if !ov.is_empty() {
            body.override_cutoff_frequency = qt_to_int(ov) != 0;
        }
        let cutoff = t.attr("CutOffFrequency");
        if !cutoff.is_empty() {
            body.cutoff_frequency = qt_to_double(cutoff);
        }
        let mesh = t.attr("Meshmodel");
        // Ported from core/RigidBody.cpp, RigidBody::addMeshModel: in the C++ the mesh, and
        // the next two attributes, are kept only if the OBJ file loads (see `mesh_opens`).
        // Fixed (ledger F18): kept whether or not it loads, with the answer recorded.
        let readable = match self.options.mesh_loads {
            Some(probe) if !mesh.is_empty() => Some(probe(mesh)),
            Some(_) => None,
            None => {
                if !mesh.is_empty() {
                    self.warnings.push(format!(
                        "rigid body {id}: mesh '{mesh}' kept without checking that it loads; the C++ drops a mesh whose file cannot be opened"
                    ));
                }
                None
            }
        };
        let keep = readable != Some(false) || !self.options.compat.drop_unreadable_mesh;
        if !mesh.is_empty() && keep {
            body.mesh_model = Some(mesh.to_owned());
            body.mesh_readable = readable;
            let draw = t.attr("DrawMeshmodel");
            if !draw.is_empty() {
                body.draw_mesh_model = qt_to_int(draw) != 0;
            }
            let scale = t.attr("MeshScale");
            if !scale.is_empty() {
                body.mesh_scale = qt_to_double(scale);
            }
        }

        for &i in dummies {
            let n_frames = raw.bodies[body_index].n_frames();
            let planned = raw.planned.get(&i).copied();
            let point = self.read_dummy_marker(
                i,
                n_frames,
                planned,
                raw.planned_exact,
                &raw.trial.name,
                &mut raw.cells,
            )?;
            raw.bodies[body_index].virtual_points.push(point);
        }
        Ok(())
    }

    /// Ported from core/RigidBody.cpp, RigidBody::addDummyPoint. Ledger F84: the
    /// coordinates, rows beyond the trial's frames included, were counted before reading
    /// (`planned`, the length they reach); they are added to `cells` (of the trial `trial`)
    /// before they are allocated, and a row beyond `planned` is refused. `planned` is
    /// reserved only when `reserve` (it is exact, not a bound).
    fn read_dummy_marker(
        &mut self,
        index: usize,
        n_frames: usize,
        planned: Option<usize>,
        reserve: bool,
        trial: &str,
        cells: &mut TrialCells,
    ) -> Result<VirtualPoint, ReadError> {
        let t = &self.tokens[index];
        let name = t.attr("Name").to_owned();
        let refs = t.attr("PointReferences").to_owned();
        let refs2 = match t.attr("PointReferences2") {
            "" => refs.clone(),
            other => other.to_owned(),
        };
        let coords_path = t.attr("PointCoordinates").to_owned();

        let (entry, data) = self.files.take_or_empty(&refs);
        let fields = second_line_fields(&data);
        if fields.len() < 3 {
            return Err(ReadError::data(
                &entry,
                Some(2),
                Some(fields.len() + 1),
                "a virtual point needs x, y and z",
            ));
        }
        let reference = Point3::new(
            qt_to_double(&fields[0]),
            qt_to_double(&fields[1]),
            qt_to_double(&fields[2]),
        );
        let marker_id = if fields.len() > 3 {
            qt_to_int(&fields[3])
        } else {
            -1
        };

        let source = if marker_id == -1 {
            let (_, data) = self.files.take_or_empty(&coords_path);
            let len = planned.unwrap_or(n_frames);
            self.allocate_cells(
                index,
                trial,
                &format!("virtual point '{name}'"),
                len,
                &mut cells.virtual_points,
                len as u64,
                xma_model::project::MAX_TRIAL_CELLS,
            )?;
            let mut reader = LineReader::new(&data);
            reader.next_line();
            let mut coords = Vec::with_capacity(if reserve { len } else { 0 });
            let mut row = 0;
            loop {
                let (line, eof) = reader.next_line();
                if eof {
                    break;
                }
                row += 1;
                let text = qstring(line);
                let f = split_commas(&text);
                if f.len() == 3 {
                    if coords.len() >= len {
                        return Err(uncounted_row(&normalize(&coords_path), row));
                    }
                    coords.push(if f.contains(&"NaN") {
                        None
                    } else {
                        Some(Point3::new(
                            qt_to_double(f[0]),
                            qt_to_double(f[1]),
                            qt_to_double(f[2]),
                        ))
                    });
                }
            }
            if coords.len() < n_frames {
                coords.resize(n_frames, None);
            }
            VirtualPointSource::Coordinates(coords)
        } else {
            // Counted as one cell per frame, as in the check before reading.
            self.allocate_cells(
                index,
                trial,
                &format!("virtual point '{name}'"),
                n_frames,
                &mut cells.virtual_points,
                n_frames as u64,
                xma_model::project::MAX_TRIAL_CELLS,
            )?;
            let body = usize::try_from(marker_id).map_err(|_| {
                ReadError::data(
                    &entry,
                    Some(2),
                    Some(4),
                    format!("rigid body index {marker_id} cannot be negative"),
                )
            })?;
            let (entry2, data) = self.files.take_or_empty(&refs2);
            let f = second_line_fields(&data);
            if f.len() < 3 {
                return Err(ReadError::data(
                    &entry2,
                    Some(2),
                    Some(f.len() + 1),
                    "a virtual point needs x, y and z",
                ));
            }
            VirtualPointSource::OtherBody {
                body,
                point: Point3::new(
                    qt_to_double(&f[0]),
                    qt_to_double(&f[1]),
                    qt_to_double(&f[2]),
                ),
            }
        };
        Ok(VirtualPoint {
            name,
            reference,
            source,
        })
    }

    /// The end of `readProjectFile`: the active trial, the version remaps, and the model.
    fn finish(
        mut self,
        file_version: FileVersion,
        active_trial: i32,
    ) -> Result<LoadedProject, ReadError> {
        let v = file_version.value;
        let mut sources = Vec::new();
        for raw in std::mem::take(&mut self.trials) {
            let RawTrial {
                mut trial,
                mut markers,
                bodies,
                mut source,
                cells: _,
                planned: _,
                planned_exact: _,
            } = raw;
            source.markers_without_element = markers
                .iter()
                .enumerate()
                .filter(|(_, m)| !m.has_element)
                .map(|(i, _)| i)
                .collect();
            for m in &mut markers {
                if v < 0.2 {
                    m.update_to_project_12();
                }
                if v < 0.3 {
                    m.update_to_project_13(self.options.compat.old_penalty_zero);
                }
            }
            trial.markers = markers
                .into_iter()
                .map(|m| m.into_model().map(Arc::new))
                .collect::<Result<_, _>>()?;
            trial.rigid_bodies = bodies.into_iter().map(Arc::new).collect();
            self.project.trials.push(Arc::new(trial));
            sources.push(source);
        }
        let n = self.project.trials.len();
        self.project.active_trial = match usize::try_from(active_trial) {
            Ok(a) if a < n => a,
            _ => n.saturating_sub(1),
        };

        let mut missing = std::mem::take(&mut self.files.missing);
        missing.dedup();
        for m in missing {
            self.warnings.push(format!(
                "'{m}' is referenced but not in the archive; read as empty"
            ));
        }
        let Files {
            entries,
            names,
            data,
            used,
            comment,
            ..
        } = self.files;
        let attachments = entries
            .into_iter()
            .zip(names)
            .zip(data)
            .zip(used)
            .filter(|(_, used)| !used)
            .map(|(((entry, name), data), _)| Attachment { entry, name, data })
            .collect();
        Ok(LoadedProject {
            project: self.project,
            attachments,
            file_version,
            archive_comment: comment,
            trial_sources: sources,
            warnings: self.warnings,
            settings_sources: Vec::new(),
            stored_settings: SettingsPatch::default(),
            compat: CppCompat::default(),
        })
    }
}

// ---------------------------------------------------------------------------------------
// Helpers shared by the element readers

/// `QTextStream::readLine` until null: lines without their terminators.
fn text_lines(text: &str) -> Vec<String> {
    lines(text.as_bytes()).into_iter().map(qstring).collect()
}

/// Rows of a `for (double v; in >> v; comma(in))` loop that produced values, with their
/// 1-based line numbers.
fn value_rows(data: &[u8], nonfinite: bool) -> Vec<(usize, Vec<f64>)> {
    lines(data)
        .into_iter()
        .enumerate()
        .filter_map(|(i, l)| {
            let v = values(l, nonfinite);
            (!v.is_empty()).then_some((i + 1, v))
        })
        .collect()
}

/// One data row: as `in >> double` reads it, or with `nonfinite` also the words for NaN and
/// the infinities ([`stream_values_nonfinite`], ledger F49).
fn values(line: &[u8], nonfinite: bool) -> Vec<f64> {
    if nonfinite {
        stream_values_nonfinite(line)
    } else {
        stream_values(line)
    }
}

/// A 3 × 3 matrix written as three rows of three values.
fn read_matrix3(
    entry: &str,
    data: &[u8],
    what: &str,
    nonfinite: bool,
) -> Result<Matrix3, ReadError> {
    let rows = value_rows(data, nonfinite);
    let mut m = Matrix3::zeros();
    for y in 0..3 {
        let (line, row) = rows.get(y).ok_or_else(|| {
            ReadError::data(
                entry,
                None,
                None,
                format!("a {what} needs 3 rows; found {}", rows.len()),
            )
        })?;
        if row.len() < 3 {
            return Err(ReadError::data(
                entry,
                Some(*line),
                Some(row.len() + 1),
                format!("a {what} row needs 3 values"),
            ));
        }
        for x in 0..3 {
            m[(y, x)] = row[x];
        }
    }
    Ok(m)
}

/// Lines split on commas with exactly three fields, read with `QString::toDouble`
/// (`RigidBody::load`, `RigidBody::loadOptimized`).
fn read_xyz_lines(data: &[u8]) -> Vec<Point3> {
    lines(data)
        .into_iter()
        .filter_map(|l| {
            let text = qstring(l);
            let f = split_commas(&text);
            (f.len() == 3)
                .then(|| Point3::new(qt_to_double(f[0]), qt_to_double(f[1]), qt_to_double(f[2])))
        })
        .collect()
}

/// The fields of the line left after two `safeGetline` calls.
fn second_line_fields(data: &[u8]) -> Vec<String> {
    let mut r = LineReader::new(data);
    r.next_line();
    let (line, _) = r.next_line();
    split_commas(&qstring(line))
        .into_iter()
        .map(str::to_owned)
        .collect()
}

/// `QFileInfo::suffix` is one of the extensions `Trial::Trial` opens as a video.
fn is_video_file(path: &str) -> bool {
    let name = file_name(path);
    let suffix = name.rsplit_once('.').map_or("", |(_, s)| s);
    ["cine", "avi", "mp4", "mov"]
        .iter()
        .any(|e| suffix.eq_ignore_ascii_case(e))
}

/// Ported from core/Marker.cpp, Marker::load. Ledger F84: a row that would take the marker
/// beyond `max_frames`, the frames counted before reading, is refused.
fn load_marker_2d(
    files: &mut Files,
    m: &mut RawMarker,
    max_frames: usize,
    points: &str,
    status: &str,
    size: &str,
) -> Result<(), ReadError> {
    let n_cams = m.points2d.len();
    let nonfinite = files.nonfinite;
    let (entry, data) = files.take_or_empty(points);
    for (lc, line) in lines(&data).into_iter().enumerate() {
        let v = values(line, nonfinite);
        if v.is_empty() {
            continue;
        }
        if n_cams == 0 {
            return Err(ReadError::data(
                &entry,
                Some(lc + 1),
                None,
                "2D points for a project without cameras",
            ));
        }
        if m.points2d[0].len() <= lc {
            if m.n_frames() >= max_frames {
                return Err(uncounted_row(&entry, lc));
            }
            m.add_frame();
        }
        if lc >= m.points2d[0].len() {
            return Err(ReadError::data(
                &entry,
                Some(lc + 1),
                None,
                "a row beyond the end follows an empty row",
            ));
        }
        if v.len() < 2 * n_cams {
            return Err(ReadError::data(
                &entry,
                Some(lc + 1),
                Some(v.len() + 1),
                format!("{} values needed (x and y per camera)", 2 * n_cams),
            ));
        }
        for c in 0..n_cams {
            m.points2d[c][lc] = Point2::new(v[2 * c], v[2 * c + 1]);
        }
    }
    let (entry, data) = files.take_or_empty(status);
    for (lc, line) in lines(&data).into_iter().enumerate() {
        let v = stream_values(line);
        if v.is_empty() || n_cams == 0 {
            continue;
        }
        check_row(&entry, lc, v.len(), n_cams, m.n_frames())?;
        for (track, &value) in m.status2d.iter_mut().zip(&v) {
            track[lc] = c_int(value);
        }
    }
    let (entry, data) = files.take_or_empty(size);
    for (lc, line) in lines(&data).into_iter().enumerate() {
        let v = values(line, nonfinite);
        if v.is_empty() || n_cams == 0 {
            continue;
        }
        check_row(&entry, lc, v.len(), n_cams, m.n_frames())?;
        for (track, &value) in m.size.iter_mut().zip(&v) {
            track[lc] = value;
        }
    }
    Ok(())
}

/// Ported from core/Marker.cpp, Marker::load3DPoints.
fn load_marker_3d(
    files: &mut Files,
    m: &mut RawMarker,
    points: &str,
    status: &str,
) -> Result<(), ReadError> {
    let nonfinite = files.nonfinite;
    let (entry, data) = files.take_or_empty(points);
    for (lc, line) in lines(&data).into_iter().enumerate() {
        let v = values(line, nonfinite);
        if v.is_empty() {
            continue;
        }
        check_row(&entry, lc, v.len(), 3, m.n_frames())?;
        m.points3d[lc] = Point3::new(v[0], v[1], v[2]);
    }
    let (entry, data) = files.take_or_empty(status);
    for (lc, line) in lines(&data).into_iter().enumerate() {
        let v = stream_values(line);
        if v.is_empty() {
            continue;
        }
        check_row(&entry, lc, v.len(), 1, m.n_frames())?;
        m.status3d[lc] = c_int(v[0]);
    }
    Ok(())
}

/// Ledger F84: a row of `entry` (`lc`, 0-based) that would grow an array beyond the frames
/// counted before reading. Not reached: the count and the reader use the same functions on
/// the same data; this keeps the allocation bounded even if they ever disagreed.
fn uncounted_row(entry: &str, lc: usize) -> ReadError {
    ReadError::data(
        entry,
        Some(lc + 1),
        None,
        "a row beyond the frames counted before reading (an internal error)",
    )
}

/// A row the C++ would write out of bounds.
fn check_row(
    entry: &str,
    lc: usize,
    n_values: usize,
    needed: usize,
    n_frames: usize,
) -> Result<(), ReadError> {
    if lc >= n_frames {
        return Err(ReadError::data(
            entry,
            Some(lc + 1),
            None,
            format!("more rows than the marker's {n_frames} frames"),
        ));
    }
    if n_values < needed {
        return Err(ReadError::data(
            entry,
            Some(lc + 1),
            Some(n_values + 1),
            format!("{needed} values needed"),
        ));
    }
    Ok(())
}

/// Ported from core/RigidBody.cpp, RigidBody::setReferenceMarkerReferences.
fn set_reference_marker_references(body: &mut RigidBody, markers: &[RawMarker]) {
    if !body
        .marker_indices
        .iter()
        .all(|&i| markers[i].reference_3d.is_some())
    {
        return;
    }
    for (k, &i) in body.marker_indices.iter().enumerate() {
        let Some(p) = markers[i].reference_3d else {
            continue;
        };
        if let Some(slot) = body.reference_points.get_mut(k) {
            *slot = p;
        }
        if let Some(slot) = body.reference_points_original.get_mut(k) {
            *slot = p;
        }
        if let Some(slot) = body.reference_names.get_mut(k) {
            *slot = format!("{} - FromMarker", markers[i].description);
        }
    }
    body.references_set = 1;
}

/// Ported from core/RigidBody.cpp, RigidBody::resetReferences (the stored data; the pose
/// recomputation and filtering it starts are derived).
fn reset_references(body: &mut RigidBody, markers: &[RawMarker]) {
    let origin = Point3::new(0.0, 0.0, 0.0);
    for k in 0..body.reference_names.len() {
        body.reference_names[k].clear();
        if let Some(p) = body.reference_points.get_mut(k) {
            *p = origin;
        }
        if let Some(p) = body.reference_points_original.get_mut(k) {
            *p = origin;
        }
    }
    body.references_set = 0;
    body.has_optimized_coordinates = false;
    set_reference_marker_references(body, markers);
}

/// Ported from core/RigidBody.cpp, RigidBody::clearPointIdx (which leaves
/// `points3D_original` as it is).
fn clear_point_idx(body: &mut RigidBody, markers: &[RawMarker]) {
    body.marker_indices.clear();
    body.reference_points.clear();
    body.reference_names.clear();
    reset_references(body, markers);
}

/// Ported from core/RigidBody.cpp, RigidBody::addPointIdx (with `recompute` false).
fn add_point_idx(body: &mut RigidBody, idx: usize, markers: &[RawMarker]) {
    let origin = Point3::new(0.0, 0.0, 0.0);
    body.marker_indices.push(idx);
    body.reference_points.push(origin);
    body.reference_points_original.push(origin);
    body.reference_names.push(String::new());
    reset_references(body, markers);
}

/// Converts a stored rotation matrix to a Rodrigues vector, as `cv::Rodrigues` does on
/// load (`core/CalibrationImage.cpp`, `CalibrationImage::loadRotationMatrix`): the matrix
/// is first replaced by the nearest rotation (`U Vᵀ` of its SVD).
fn matrix_to_rodrigues(m: &Matrix3) -> Vector3 {
    xma_vision::camera::matrix_to_rodrigues(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ledger F84: the cheap bound is never below the lines the reader sees, whatever the
    /// line endings (review 14e).
    #[test]
    fn review14e_line_bound_is_an_upper_bound() {
        let cases: &[&[u8]] = &[
            b"",
            b"a",
            b"a\n",
            b"\n",
            b"\r",
            b"\r\n",
            b"\n\r",
            b"a\rb\r\nc\n\nd",
            b"\r\r\r\n\n\n",
            b"1,2\r\n3,4\r\n",
            b"x\n\r\n\r\ry",
        ];
        for &c in cases {
            let n = count_lines(c, |_| true);
            assert!(line_bound(c) >= n, "{c:?}: {} < {n}", line_bound(c));
            assert!(line_bound(c) >= lines(c).len());
        }
        // Every byte sequence of length up to 6 over the alphabet {a, \n, \r}.
        let alphabet = b"a\n\r";
        for len in 0..=6u32 {
            for k in 0..3usize.pow(len) {
                let mut v = Vec::new();
                let mut x = k;
                for _ in 0..len {
                    v.push(alphabet[x % 3]);
                    x /= 3;
                }
                assert!(line_bound(&v) >= count_lines(&v, |_| true), "{v:?}");
            }
        }
    }

    #[test]
    fn rodrigues_of_known_rotations() {
        let angle = 0.7_f64;
        let rz = Matrix3::new(
            angle.cos(),
            -angle.sin(),
            0.0,
            angle.sin(),
            angle.cos(),
            0.0,
            0.0,
            0.0,
            1.0,
        );
        let v = matrix_to_rodrigues(&rz);
        assert!((v - Vector3::new(0.0, 0.0, angle)).norm() < 1e-15);
        assert_eq!(matrix_to_rodrigues(&Matrix3::identity()), Vector3::zeros());
        let half_turn = Matrix3::new(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0);
        let v = matrix_to_rodrigues(&half_turn);
        assert!(
            (v - Vector3::new(std::f64::consts::PI, 0.0, 0.0)).norm() < 1e-12,
            "{v}"
        );
    }

    #[test]
    fn paths_normalise_like_the_cpp() {
        assert_eq!(normalize("Trial 1\\data\\M.csv"), "Trial 1/data/M.csv");
        assert_eq!(normalize("/a//b/./c"), "a/b/c");
        assert_eq!(normalize("a/../b"), "b");
        assert_eq!(file_name("Camera 1/x.tif"), "x.tif");
        assert!(is_video_file("/v/Camera1.AVI"));
        assert!(!is_video_file("/v/f001.png"));
    }
}
