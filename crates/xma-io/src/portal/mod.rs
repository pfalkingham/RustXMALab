//! Files downloaded from the XMA Portal: a trial (`.xmatrial`, or a `.zip`) and a project
//! archive (a `.zip` or `.xma` without `project.xml`).
//!
//! The XMA Portal is a web service; this module reads the files a user has downloaded from
//! it and never contacts it.
//!
//! # The format as XMALab 3 reads it
//!
//! Ported from ui/ProjectFileIO.cpp (`loadProject` `:275-307`, `loadXMAPortalTrial`
//! `:870-1014`, `loadXMALabProject` `:1016-1142`, `addMetaData` `:1423-1448`), core/Project.cpp
//! (`setXMLData`, `parseXMLData`), core/Trial.cpp (`setXMLData`, `setFrameRateFromXML`) and
//! core/VideoStream.cpp (`parseXMLData`).
//!
//! - The archive is a zip. XMALab extracts all of it into a temporary folder and takes the
//!   first folder there, by name ignoring case (`QDir::entryList(NoDotAndDotDot | Dirs)`,
//!   which leaves out hidden folders), as the archive's **top folder** ([`top_folder`]).
//!   Files outside it are not used. A zip made on macOS has a `__MACOSX` folder, which
//!   sorts first; here it is skipped and the first folder holding the metadata file is
//!   taken (ledger F63; XMALab 3 mode as XMALab).
//! - The archive is a Portal archive when `<top>/File_Metadata/XMALab_Files-metadata.xml`
//!   exists ([`METADATA_FILE`]). An archive with `project.xml` at its root is a project
//!   and is never read as a Portal archive.
//! - The metadata file is XML. XMALab reads it element by element, whatever the root
//!   element and the nesting ([`Metadata::parse`]): `StudyName` and `TrialName` outside a
//!   `File` element (the last of each); each `File` element's `Filename`, `FileID` and
//!   `FileCategory` (at any depth inside it), and inside its `metadata` element
//!   `cameraNumber` (`cam1`, `cam2`: the number after removing every `cam`, by
//!   `QString::toInt`, 0 when that is not a number) and `FrameRate`; the project-level
//!   elements `Repository`, `StudyID`, `TrialID`, `TrialNumber`, `TrialType`, `lab`,
//!   `attribcomment`, `ts`, `trialDate`, and per file `Instrument`, `ShutterSpeed`, `kV`,
//!   `EDR`, `mA`, `SID`, `MagLevel`, `Radiationtype`, `Pulsewidth`, `FileDescription` and
//!   `Lab`, which are kept as the metadata's lines and shown in Project Metadata.
//! - `Filename` is a path relative to the top folder.
//! - A **trial** ([`Metadata::trial_files`]): every `File` with a camera number of 0 or more
//!   and a file name; the cameras are numbered in the order of their camera numbers.
//! - A **project** ([`Metadata::project_files`]): every `File` whose category is
//!   `XCalibObject` or `CalibObject` (a calibration image) or `XCalibGrid` or
//!   `DistortionGrid` (an undistortion grid), with a camera number of 1 or more; camera
//!   `camN` is the dialog's camera in the order of the numbers. The calibration object is
//!   the one `*.csv` and the one `*.ref` file directly in the top folder, each only when
//!   there is exactly one ([`cube_files`]).
//!
//! # Safety against hostile archives
//!
//! The archive is read with [`FileArchive`]: only the central directory and the entries
//! asked for are read, each streamed to its file, so any size of archive is safe for
//! memory; the metadata file is read into memory only up to [`MAX_METADATA_BYTES`]. An
//! entry whose name climbs out of the folder it is extracted to (`..`) or is absolute is
//! refused, and nothing is extracted (ledger F55, both modes); XMALab 3 writes such an
//! entry where its path leads.

pub mod fixture;

use std::collections::BTreeSet;
use std::fmt;
use std::path::{Path, PathBuf};

use crate::cparse::{lines, qstring, qt_to_double, qt_to_int};
use crate::xml::{Token, read_element_text, tokenize_lenient_limited, tokenize_limited};

/// The elements `Project::parseXMLData` reads with `readElementText()`.
const PROJECT_FIELDS: [&str; 12] = [
    "Repository",
    "StudyName",
    "StudyID",
    "TrialName",
    "TrialID",
    "TrialNumber",
    "TrialType",
    "lab",
    "attribcomment",
    "ts",
    "trialDate",
    "cameraNumber",
];

/// The most tokens (start tags, end tags, texts) a metadata file may hold: 500 000. A Portal
/// file lists each of its files in about 40 tokens, so this is some 12 000 files; a real one
/// holds a few to a few hundred (ledger F78: without it, 64 MiB of `<a/>` held 2.9 GB).
pub const MAX_METADATA_TOKENS: usize = 500_000;
use crate::zip::{FileArchive, FileArchiveError};

/// The metadata file, relative to the top folder.
pub const METADATA_FILE: &str = "File_Metadata/XMALab_Files-metadata.xml";

/// The largest metadata file read (64 MiB).
pub const MAX_METADATA_BYTES: u64 = 64 << 20;

/// XMALab's error for an archive that is not a Portal archive
/// (`ProjectFileIO::loadXMAPortalTrial`).
pub const NOT_VALID: &str = "Not a valid xma file from the portal";

/// Why a Portal file could not be read or extracted.
#[derive(Debug)]
pub enum PortalError {
    /// The file (or a file being written) could not be read or written.
    Io {
        path: PathBuf,
        error: std::io::Error,
    },
    /// The zip archive is malformed or uses what is not supported.
    Archive(FileArchiveError),
    /// The archive has no folder with `File_Metadata/XMALab_Files-metadata.xml` (XMALab:
    /// "Not a valid xma file from the portal").
    NotPortal,
    /// The metadata file is not well-formed XML, or too large.
    Metadata(String),
    /// An entry or a name from the metadata would be written outside the folder it
    /// belongs in.
    UnsafePath(String),
    /// The extraction was stopped by its caller.
    Stopped,
}

impl fmt::Display for PortalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PortalError::Io { path, error } => write!(f, "{}: {error}", path.display()),
            PortalError::Archive(e) => write!(f, "{e}"),
            PortalError::NotPortal => write!(f, "{NOT_VALID}"),
            PortalError::Metadata(m) => write!(f, "{METADATA_FILE}: {m}"),
            PortalError::UnsafePath(p) => write!(
                f,
                "{p}: the path leads outside the folder it is extracted to; nothing was extracted"
            ),
            PortalError::Stopped => write!(f, "stopped"),
        }
    }
}

impl std::error::Error for PortalError {}

fn io_err(path: &Path, error: std::io::Error) -> PortalError {
    PortalError::Io {
        path: path.to_path_buf(),
        error,
    }
}

// ---------------------------------------------------------------------------------------
// Entry names

/// The parts of an entry name, split at `/` and `\` (QuaZip names use `/`; a name written
/// on Windows may use `\`, which the C++ build there takes as a separator), with `.` and
/// empty parts left out. `Err` for a name that is absolute (a leading separator, a drive
/// `C:`) or has a `..` part.
pub fn entry_parts(name: &str) -> Result<Vec<&str>, String> {
    if name.starts_with(['/', '\\']) {
        return Err(name.to_owned());
    }
    let mut out = Vec::new();
    for p in name.split(['/', '\\']) {
        match p {
            "" | "." => {}
            other if !part_allowed(other) => return Err(name.to_owned()),
            other => out.push(other),
        }
    }
    Ok(out)
}

/// The Windows device names (`CON`, `PRN`, `AUX`, `NUL`, `COM1` to `COM9`, `LPT1` to
/// `LPT9`, also with the superscript digits ¹ ² ³), which name a device in any folder,
/// with or without an extension.
fn reserved_device(part: &str) -> bool {
    let stem = part
        .split('.')
        .next()
        .unwrap_or(part)
        .trim_end()
        .to_uppercase();
    if matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL") {
        return true;
    }
    let mut c = stem.chars();
    let head: String = c.by_ref().take(3).collect();
    let rest: String = c.collect();
    (head == "COM" || head == "LPT")
        && matches!(
            rest.as_str(),
            "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "\u{b9}" | "\u{b2}" | "\u{b3}"
        )
}

/// One part of an entry name that may be written on every system the program is built for
/// (ledger F55): not `..`; none of `:` (a drive, or an NTFS stream, in any part), `<`, `>`,
/// `"`, `|`, `?`, `*` or a control character (below U+0020); not ending in `.` or a space
/// (which Windows drops, so two names would be one file); not a Windows device name.
fn part_allowed(part: &str) -> bool {
    part != ".."
        && !part
            .chars()
            .any(|c| matches!(c, ':' | '<' | '>' | '"' | '|' | '?' | '*') || (c as u32) < 0x20)
        && !part.ends_with(['.', ' '])
        && !reserved_device(part)
}

/// A relative path read from the metadata (`Filename`) or a folder name (`StudyName`,
/// `TrialName`): its parts, refused when it climbs out (`..`) or is absolute.
pub fn safe_relative(name: &str) -> Result<PathBuf, PortalError> {
    let parts = entry_parts(name).map_err(PortalError::UnsafePath)?;
    Ok(parts.iter().collect())
}

/// The archive's top folder (`filesList[0]` of `loadProject`): the first folder, by name
/// ignoring case, that extracting `names` creates at the root; hidden folders (a name
/// starting with `.`) are left out, as `QDir::entryList` leaves them out on Unix.
#[must_use]
pub fn top_folder<'a>(names: impl IntoIterator<Item = &'a str>) -> Option<String> {
    top_folders(names).into_iter().next()
}

/// The folders extracting `names` creates at the root, in `QDir::entryList`'s order (by name
/// ignoring case, hidden ones left out); [`top_folder`] is the first.
#[must_use]
pub fn top_folders<'a>(names: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    let mut dirs: BTreeSet<(String, String)> = BTreeSet::new();
    for n in names {
        let Ok(parts) = entry_parts(n) else { continue };
        // A folder entry ("a/") or a file inside a folder ("a/b").
        let is_dir = n.ends_with(['/', '\\']);
        if parts.len() >= 2 || (is_dir && !parts.is_empty()) {
            let d = parts[0];
            if !d.starts_with('.') {
                dirs.insert((d.to_lowercase(), d.to_owned()));
            }
        }
    }
    dirs.into_iter().map(|(_, d)| d).collect()
}

/// The folder macOS's archiver adds to a zip it makes (resource forks); it holds no data.
pub const MACOS_FOLDER: &str = "__MACOSX";

/// The top folder XMALab 3 takes ([`top_folder`]) when `macos_folder_taken`; otherwise
/// (ledger F63) the first folder other than [`MACOS_FOLDER`] that holds
/// `File_Metadata/XMALab_Files-metadata.xml`, else the first other than it.
fn choose_top(names: &[String], macos_folder_taken: bool) -> Option<String> {
    let all = top_folders(names.iter().map(String::as_str));
    if macos_folder_taken {
        return all.into_iter().next();
    }
    // One pass over the names: the folders that hold the metadata file.
    let holders: BTreeSet<&str> = names
        .iter()
        .filter(|n| !n.ends_with(['/', '\\']))
        .filter_map(|n| {
            let p = entry_parts(n).ok()?;
            (p.len() == 3 && p[1] == "File_Metadata" && p[2] == "XMALab_Files-metadata.xml")
                .then_some(p[0])
        })
        .collect();
    let candidates: Vec<String> = all
        .into_iter()
        .filter(|d| !d.eq_ignore_ascii_case(MACOS_FOLDER))
        .collect();
    candidates
        .iter()
        .find(|d| holders.contains(d.as_str()))
        .or_else(|| candidates.first())
        .cloned()
}

// ---------------------------------------------------------------------------------------
// The metadata

/// One `File` element of the metadata.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PortalFile {
    /// `Filename` (the last one in the element), relative to the top folder.
    pub filename: String,
    /// `FileID` by `QString::toInt`; -1 when absent.
    pub file_id: i32,
    /// `FileCategory`.
    pub category: String,
    /// `metadata/cameraNumber` as written (`cam1`); `None` when absent.
    pub camera_number: Option<String>,
    /// `metadata/FrameRate` by `QString::toDouble`; `None` when absent.
    pub frame_rate: Option<f64>,
}

impl PortalFile {
    /// The camera number as XMALab reads it: `cameraNumber` with every `cam` removed, by
    /// `QString::toInt`; `None` when the file has no `cameraNumber`.
    #[must_use]
    pub fn camera(&self) -> Option<i32> {
        self.camera_number
            .as_ref()
            .map(|c| qt_to_int(&c.replace("cam", "")))
    }
}

/// What a calibration file of a project archive is (`loadXMALabProject`'s `type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalibrationFileKind {
    /// `XCalibObject` or `CalibObject`: a calibration image (or video).
    Image,
    /// `XCalibGrid` or `DistortionGrid`: an undistortion grid.
    Grid,
}

/// A file of a project archive matched to a camera.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectFile {
    /// The camera number from the metadata (`camN`: N, 1 or more).
    pub camera_number: i32,
    pub kind: CalibrationFileKind,
    /// `Filename`, relative to the top folder.
    pub filename: String,
}

/// The metadata file of a Portal archive.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Metadata {
    /// The file's lines, as `QTextStream::readLine` gives them: what a project or trial
    /// keeps (`Project::xml_data`, `Trial::xml_data`) and saves as `metadata.xml`.
    pub lines: Vec<String>,
    /// `StudyName` (outside `File` elements; the last one).
    pub study_name: String,
    /// `TrialName` (outside `File` elements; the last one).
    pub trial_name: String,
    /// The `File` elements, in document order.
    pub files: Vec<PortalFile>,
    /// Every `cameraNumber` in the document as a number, in increasing order, each once
    /// (`Project::parseXMLData`'s `cameraIDs`).
    pub all_camera_numbers: Vec<i32>,
}

/// One `File` element read from the tokens starting at `start`; returns it and the index
/// of its end tag.
///
/// Ported from ui/ProjectFileIO.cpp, the `File` loops of loadXMAPortalTrial
/// (`:916-951`) and loadXMALabProject (`:1052-1100`), and core/Trial.cpp,
/// Trial::setFrameRateFromXML (`:2129-2156`).
fn read_file(tokens: &[Token], start: usize) -> (PortalFile, usize, bool) {
    let mut f = PortalFile {
        file_id: -1,
        ..PortalFile::default()
    };
    let mut i = start + 1;
    // `readElementText()` of a leaf; false when it met a child element (Qt's error).
    let leaf = |i: usize| read_element_text(tokens, i);
    while i < tokens.len() && !tokens[i].is_end("File") {
        if let Token::Start { name, .. } = &tokens[i] {
            let (text, ok) = match name.as_str() {
                "Filename" | "FileID" | "FileCategory" => leaf(i),
                _ => (String::new(), true),
            };
            match name.as_str() {
                "Filename" => f.filename = text,
                "FileID" => f.file_id = qt_to_int(&text),
                "FileCategory" => f.category = text,
                "metadata" => {
                    let mut k = i + 1;
                    while k < tokens.len() && !tokens[k].is_end("metadata") {
                        if tokens[k].is_start("cameraNumber") {
                            let (t, ok) = leaf(k);
                            f.camera_number = Some(t);
                            if !ok {
                                return (f, k, false);
                            }
                        } else if tokens[k].is_start("FrameRate") {
                            let (t, ok) = leaf(k);
                            f.frame_rate = Some(qt_to_double(&t));
                            if !ok {
                                return (f, k, false);
                            }
                        }
                        k += 1;
                    }
                    i = k;
                }
                _ => {}
            }
            if !ok {
                return (f, i, false);
            }
        }
        i += 1;
    }
    (f, i, true)
}

impl Metadata {
    /// Reads the metadata file's bytes.
    ///
    /// # Errors
    /// [`PortalError::Metadata`] when the file is not well-formed XML. XMALab reads such a
    /// file up to the error and, when the error is inside a `File` element (an element not
    /// closed), loops for ever (ledger F56, both modes).
    pub fn parse(data: &[u8]) -> Result<Self, PortalError> {
        // The lines are split only once the document has been read (a refused one costs no
        // copy of them).
        let text = String::from_utf8_lossy(data);
        let tokens = tokenize_limited(&text, MAX_METADATA_TOKENS).map_err(|e| {
            PortalError::Metadata(format!(
                "line {}, column {}: {}",
                e.line, e.column, e.message
            ))
        })?;
        let text_lines: Vec<String> = lines(data).into_iter().map(qstring).collect();
        Ok(Self::from_tokens(text_lines, &tokens))
    }

    /// The metadata from its lines and tokens.
    fn from_tokens(text_lines: Vec<String>, tokens: &[Token]) -> Self {
        let mut m = Metadata {
            lines: text_lines,
            ..Metadata::default()
        };
        // `Project::parseXMLData`'s pass: every `cameraNumber`, the reading stopping at the
        // first `readElementText()` that meets a child element.
        let mut numbers = BTreeSet::new();
        for (k, t) in tokens.iter().enumerate() {
            if let Token::Start { name, .. } = t
                && PROJECT_FIELDS.contains(&name.as_str())
            {
                let (text, ok) = read_element_text(tokens, k);
                if name == "cameraNumber" {
                    numbers.insert(qt_to_int(&text.replace("cam", "")));
                }
                if !ok {
                    break;
                }
            }
        }
        m.all_camera_numbers = numbers.into_iter().collect();
        // The loaders' pass (`loadXMAPortalTrial`, `loadXMALabProject`).
        let mut i = 0;
        while i < tokens.len() {
            if let Token::Start { name, .. } = &tokens[i] {
                match name.as_str() {
                    "StudyName" | "TrialName" => {
                        let (text, ok) = read_element_text(tokens, i);
                        if name == "StudyName" {
                            m.study_name = text;
                        } else {
                            m.trial_name = text;
                        }
                        if !ok {
                            break;
                        }
                    }
                    "File" => {
                        let (f, end, ok) = read_file(tokens, i);
                        m.files.push(f);
                        if !ok {
                            break;
                        }
                        i = end;
                    }
                    _ => {}
                }
            }
            i += 1;
        }
        m
    }

    /// The files of a trial archive: `(camera number, Filename)` for every `File` with a
    /// camera number of 0 or more and a non-empty name, in document order.
    ///
    /// Ported from ui/ProjectFileIO.cpp, ProjectFileIO::loadXMAPortalTrial (`:916-951`).
    #[must_use]
    pub fn trial_files(&self) -> Vec<(i32, String)> {
        self.files
            .iter()
            .filter_map(|f| {
                let c = f.camera()?;
                (c >= 0 && !f.filename.is_empty()).then(|| (c, f.filename.clone()))
            })
            .collect()
    }

    /// The calibration files of a project archive, in document order.
    ///
    /// Ported from ui/ProjectFileIO.cpp, ProjectFileIO::loadXMALabProject (`:1052-1100`).
    /// A `File` without a `Filename` is left out (XMALab takes the top folder itself as the
    /// file, whose creation then fails; ledger F57, both modes).
    #[must_use]
    pub fn project_files(&self) -> Vec<ProjectFile> {
        self.files
            .iter()
            .filter_map(|f| {
                let kind = match f.category.as_str() {
                    "XCalibObject" | "CalibObject" => CalibrationFileKind::Image,
                    "XCalibGrid" | "DistortionGrid" => CalibrationFileKind::Grid,
                    _ => return None,
                };
                let c = f.camera()?;
                (c >= 1 && !f.filename.is_empty()).then(|| ProjectFile {
                    camera_number: c,
                    kind,
                    filename: f.filename.clone(),
                })
            })
            .collect()
    }

    /// The recording speed the metadata gives a trial: `Some(rate)` when every `File`
    /// element has a `FrameRate` and all are the same positive number; `None` otherwise
    /// (XMALab then shows "Framerate could not be set. ..."), and also with no `File` at all.
    /// A rate that is not finite is refused unless `nonfinite_accepted` (ledger F52's
    /// XMALab 3 mode, in which `inf` is taken).
    ///
    /// Ported from core/Trial.cpp, Trial::setFrameRateFromXML (`:2111-2179`). XMALab reads
    /// the lines joined without separators (`xml_data.join("")`); for a file whose elements
    /// do not span lines the result is the same.
    #[must_use]
    pub fn frame_rate(&self, nonfinite_accepted: bool) -> Option<f64> {
        let rates: Vec<f64> = self
            .files
            .iter()
            .map(|f| f.frame_rate.unwrap_or(-1.0))
            .collect();
        let first = *rates.first()?;
        #[allow(clippy::float_cmp)]
        let equal = rates.iter().all(|&r| r == first);
        (equal && first > 0.0 && (first.is_finite() || nonfinite_accepted)).then_some(first)
    }
}

/// The metadata of the lines a project or trial keeps, read as XMALab reads them (joined
/// without separators, up to the first error).
///
/// # Errors
/// None at present (the signature keeps room for one).
pub fn parse_lines(lines: &[String]) -> Result<Metadata, PortalError> {
    Ok(parse_lines_with_error(lines).0)
}

/// [`parse_lines`] with the error the reading stopped at, if any: the first XML error of
/// the joined lines, or the token limit [`MAX_METADATA_TOKENS`] (ledger F78) when they hold
/// more.
#[must_use]
pub fn parse_lines_with_error(lines: &[String]) -> (Metadata, Option<crate::xml::XmlError>) {
    // `xml_data.join("")`, read up to its first error, as `QXmlStreamReader` reads it
    // (metadata held twice, ledger F60, is read in its first copy).
    let (tokens, error) = tokenize_lenient_limited(&lines.concat(), MAX_METADATA_TOKENS);
    (Metadata::from_tokens(lines.to_vec(), &tokens), error)
}

// ---------------------------------------------------------------------------------------
// The archive

/// An opened Portal archive: its directory and its metadata.
#[derive(Debug)]
pub struct PortalArchive {
    archive: FileArchive,
    path: PathBuf,
    top: String,
    /// Index and normalised path (from the top folder) of each entry under the top folder
    /// that extraction writes (the last entry of each name).
    files: Vec<(usize, PathBuf)>,
    metadata: Metadata,
}

/// The decoded names of an archive's entries, each checked ([`entry_parts`]).
fn checked_names(archive: &FileArchive) -> Result<Vec<String>, PortalError> {
    archive
        .entries()
        .map(|e| {
            let n = e.decoded_name();
            entry_parts(&n)
                .map(|_| n.clone())
                .map_err(PortalError::UnsafePath)
        })
        .collect()
}

impl PortalArchive {
    /// Opens a Portal archive and reads its metadata.
    ///
    /// # Errors
    /// [`PortalError::NotPortal`] for an archive that is not one (no folder, no metadata
    /// file, or `project.xml` at its root: a project), [`PortalError::UnsafePath`] for an
    /// entry name that climbs out of the archive or is absolute, and the others for a file
    /// that cannot be read.
    pub fn open(path: &Path) -> Result<Self, PortalError> {
        Self::open_with(path, false)
    }

    /// [`PortalArchive::open`]; `macos_folder_taken` reproduces XMALab 3's choice of the
    /// top folder (`CppCompat::portal_macos_folder_taken`, ledger F63).
    ///
    /// # Errors
    /// As [`PortalArchive::open`].
    pub fn open_with(path: &Path, macos_folder_taken: bool) -> Result<Self, PortalError> {
        Self::probe_with(path, macos_folder_taken)?.ok_or(PortalError::NotPortal)
    }

    /// [`PortalArchive::open`], with `Ok(None)` for a readable archive that is not a
    /// Portal archive (an XMALab project, or another zip).
    ///
    /// Ported from ui/ProjectFileIO.cpp, ProjectFileIO::loadProject (`:288-306`).
    ///
    /// # Errors
    /// As [`PortalArchive::open`], except [`PortalError::NotPortal`].
    pub fn probe(path: &Path) -> Result<Option<Self>, PortalError> {
        Self::probe_with(path, false)
    }

    /// [`PortalArchive::probe`] with XMALab 3's choice of the top folder or the fixed one
    /// ([`PortalArchive::open_with`]).
    ///
    /// # Errors
    /// As [`PortalArchive::probe`].
    pub fn probe_with(path: &Path, macos_folder_taken: bool) -> Result<Option<Self>, PortalError> {
        let mut archive = FileArchive::open(path).map_err(|e| match e {
            FileArchiveError::Io(error) => io_err(path, error),
            other => PortalError::Archive(other),
        })?;
        let names: Vec<String> = archive.entries().map(|e| e.decoded_name()).collect();
        let is_project = names
            .iter()
            .any(|n| entry_parts(n).is_ok_and(|p| p == ["project.xml"]));
        if is_project {
            return Ok(None);
        }
        let Some(top) = choose_top(&names, macos_folder_taken) else {
            return Ok(None);
        };
        let meta_path: PathBuf = [top.as_str(), "File_Metadata", "XMALab_Files-metadata.xml"]
            .iter()
            .collect();
        let mut meta_index = None;
        let mut files: Vec<(usize, PathBuf)> = Vec::new();
        // The place of each name in `files`: a later entry of the same name replaces it.
        let mut place: std::collections::HashMap<PathBuf, usize> = std::collections::HashMap::new();
        for (i, n) in names.iter().enumerate() {
            let Ok(parts) = entry_parts(n) else { continue };
            let p: PathBuf = parts.iter().collect();
            let is_dir = n.ends_with(['/', '\\']);
            if !is_dir && p == meta_path {
                meta_index = Some(i);
            }
            if !is_dir && parts.len() >= 2 && parts[0] == top {
                let rel: PathBuf = parts[1..].iter().collect();
                match place.get(&rel) {
                    Some(&k) => files[k].0 = i,
                    None => {
                        place.insert(rel.clone(), files.len());
                        files.push((i, rel));
                    }
                }
            }
        }
        let Some(mi) = meta_index else {
            return Ok(None);
        };
        // Every name is checked before anything is used: XMALab extracts them all.
        checked_names(&archive)?;
        let metadata_data = archive
            .read_limited(mi, MAX_METADATA_BYTES)
            .map_err(|e| match e {
                FileArchiveError::Io(error) => io_err(path, error),
                other => PortalError::Archive(other),
            })?;
        let metadata = Metadata::parse(&metadata_data)?;
        Ok(Some(Self {
            archive,
            path: path.to_path_buf(),
            top,
            files,
            metadata,
        }))
    }

    /// The archive file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The top folder's name.
    #[must_use]
    pub fn top(&self) -> &str {
        &self.top
    }

    /// The metadata.
    #[must_use]
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    /// The files extraction writes, relative to the top folder.
    pub fn files(&self) -> impl Iterator<Item = &Path> {
        self.files.iter().map(|(_, p)| p.as_path())
    }

    /// The bytes extraction writes.
    #[must_use]
    pub fn total_size(&self) -> u64 {
        self.files
            .iter()
            .filter_map(|(i, _)| self.archive.entry(*i))
            .map(|e| e.uncompressed_size)
            .fold(0u64, u64::saturating_add)
    }

    /// Extracts the top folder's contents into `dest` (created if needed), as XMALab's
    /// unzip into its temporary folder and `littleHelper::copyPath` into the trial folder
    /// do: subfolders kept, a file that already exists in `dest` is left as it is
    /// (`QFile::copy` does not overwrite). Each file is written under a temporary name and
    /// renamed when complete. `progress(done, total)` is called as bytes are written;
    /// returning false stops the extraction ([`PortalError::Stopped`]).
    ///
    /// # Errors
    /// The first file that cannot be read or written; what was extracted before stays.
    pub fn extract(
        &mut self,
        dest: &Path,
        progress: &mut dyn FnMut(u64, u64) -> bool,
    ) -> Result<Extraction, PortalError> {
        let mut out = Extraction {
            folder: dest.to_path_buf(),
            written: Vec::new(),
            kept: Vec::new(),
            created_dirs: Vec::new(),
        };
        match self.extract_into(&mut out, progress) {
            Ok(()) => Ok(out),
            Err(e) => {
                // Ledger F55: a failed extraction leaves nothing it wrote.
                undo(&out.written, &out.created_dirs);
                Err(e)
            }
        }
    }

    fn extract_into(
        &mut self,
        out: &mut Extraction,
        progress: &mut dyn FnMut(u64, u64) -> bool,
    ) -> Result<(), PortalError> {
        let dest = out.folder.clone();
        create_dirs(&dest, &mut out.created_dirs)?;
        let total = self.total_size();
        let mut done = 0u64;
        let files = self.files.clone();
        for (index, rel) in files {
            let target = dest.join(&rel);
            if target.exists() {
                out.kept.push(target);
                if let Some(e) = self.archive.entry(index) {
                    done = done.saturating_add(e.uncompressed_size);
                }
                continue;
            }
            if let Some(parent) = target.parent() {
                create_dirs(parent, &mut out.created_dirs)?;
            }
            write_entry(&mut self.archive, index, &target, &mut |n| {
                progress(done.saturating_add(n), total)
            })?;
            if let Some(e) = self.archive.entry(index) {
                done = done.saturating_add(e.uncompressed_size);
            }
            out.written.push(target);
        }
        Ok(())
    }
}

/// Creates `dir` and its missing ancestors, recording each folder created (outermost
/// first), so that [`undo`] can take back exactly what an import made.
///
/// # Errors
/// A folder that cannot be created.
pub fn create_dirs(dir: &Path, created: &mut Vec<PathBuf>) -> Result<(), PortalError> {
    let mut missing = Vec::new();
    let mut d = Some(dir);
    while let Some(p) = d {
        if p.as_os_str().is_empty() || p.is_dir() {
            break;
        }
        missing.push(p.to_path_buf());
        d = p.parent();
    }
    for p in missing.into_iter().rev() {
        std::fs::create_dir(&p).map_err(|e| io_err(&p, e))?;
        created.push(p);
    }
    Ok(())
}

/// Removes the files an import wrote and then the folders it created (innermost first, each
/// only when empty): never a file or folder that was there before.
pub fn undo(written: &[PathBuf], created_dirs: &[PathBuf]) {
    for f in written.iter().rev() {
        let _ = std::fs::remove_file(f);
    }
    for d in created_dirs.iter().rev() {
        let _ = std::fs::remove_dir(d);
    }
}

/// What [`PortalArchive::extract`] did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Extraction {
    /// The folder extracted into.
    pub folder: PathBuf,
    /// The files written.
    pub written: Vec<PathBuf>,
    /// The folders created (outermost first).
    pub created_dirs: Vec<PathBuf>,
    /// The files that existed already and were left as they were.
    pub kept: Vec<PathBuf>,
}

/// Streams entry `index` to `target` through a temporary file next to it.
fn write_entry(
    archive: &mut FileArchive,
    index: usize,
    target: &Path,
    progress: &mut dyn FnMut(u64) -> bool,
) -> Result<(), PortalError> {
    let mut part = target.as_os_str().to_owned();
    part.push(".xma-part");
    let part = PathBuf::from(part);
    let file = std::fs::File::create(&part).map_err(|e| io_err(&part, e))?;
    let mut w = std::io::BufWriter::new(file);
    let r = archive.copy_to(index, &mut w, progress);
    let r = r.and_then(|_| {
        use std::io::Write;
        w.flush().map_err(FileArchiveError::Io)
    });
    drop(w);
    match r {
        Ok(_) => std::fs::rename(&part, target).map_err(|e| {
            let _ = std::fs::remove_file(&part);
            io_err(target, e)
        }),
        Err(e) => {
            let _ = std::fs::remove_file(&part);
            Err(match e {
                FileArchiveError::Io(error) => io_err(target, error),
                FileArchiveError::Zip(crate::zip::ZipError::BadData(m)) if m == "stopped" => {
                    PortalError::Stopped
                }
                other => PortalError::Archive(other),
            })
        }
    }
}

/// Extracts a zip held in a trial (an image sequence, `seq.zip`) into `folder` (files
/// already there are kept, as for the archive) and removes the zip. The caller passes only a
/// zip its own import wrote (ledger F79). On a failure, what it wrote is removed and the zip
/// kept.
///
/// Ported from ui/ProjectFileIO.cpp, loadXMAPortalTrial (`:994-999`) and
/// unzipFromFileToFolder. Entry names are checked as for the archive.
///
/// # Errors
/// As [`PortalArchive::extract`]; the zip is kept when it could not be extracted.
pub fn extract_nested(zip: &Path, folder: &Path) -> Result<Extraction, PortalError> {
    let mut archive = FileArchive::open(zip).map_err(|e| match e {
        FileArchiveError::Io(error) => io_err(zip, error),
        other => PortalError::Archive(other),
    })?;
    let names = checked_names(&archive)?;
    let mut out = Extraction {
        folder: folder.to_path_buf(),
        written: Vec::new(),
        kept: Vec::new(),
        created_dirs: Vec::new(),
    };
    let mut run = || -> Result<(), PortalError> {
        create_dirs(folder, &mut out.created_dirs)?;
        for (i, n) in names.iter().enumerate() {
            if n.ends_with(['/', '\\']) {
                continue;
            }
            let rel = safe_relative(n)?;
            if rel.as_os_str().is_empty() {
                continue;
            }
            let target = folder.join(rel);
            if target.exists() {
                out.kept.push(target);
                continue;
            }
            if let Some(parent) = target.parent() {
                create_dirs(parent, &mut out.created_dirs)?;
            }
            write_entry(&mut archive, i, &target, &mut |_| true)?;
            out.written.push(target);
        }
        Ok(())
    };
    if let Err(e) = run() {
        undo(&out.written, &out.created_dirs);
        return Err(e);
    }
    drop(archive);
    if let Err(e) = std::fs::remove_file(zip) {
        undo(&out.written, &out.created_dirs);
        return Err(io_err(zip, e));
    }
    Ok(out)
}

/// The cube files of a project archive's top folder (`loadXMALabProject` `:1125-1141`):
/// the `.csv` and the `.ref` file directly in `folder`, each only when there is exactly one
/// (case-insensitive extensions, as `QDir::entryList`'s name filters match).
#[must_use]
pub fn cube_files(folder: &Path) -> (Option<PathBuf>, Option<PathBuf>) {
    let mut csv = Vec::new();
    let mut refs = Vec::new();
    if let Ok(rd) = std::fs::read_dir(folder) {
        for e in rd.filter_map(Result::ok) {
            if !e.file_type().is_ok_and(|t| t.is_file()) {
                continue;
            }
            let name = e.file_name().to_string_lossy().to_lowercase();
            if name.ends_with(".csv") {
                csv.push(e.path());
            } else if name.ends_with(".ref") {
                refs.push(e.path());
            }
        }
    }
    let one = |v: Vec<PathBuf>| (v.len() == 1).then(|| v.into_iter().next()).flatten();
    (one(csv), one(refs))
}

/// Where the import of a Portal trial puts its files (`loadXMAPortalTrial` `:957-980`).
#[derive(Clone, Copy, Debug)]
pub struct Destination<'a> {
    /// Settings > General "Use custom workspace to save data imported from XMA Portal"
    /// with its folder; `None` when off (or when its folder is empty).
    pub workspace: Option<&'a Path>,
    /// The open project's file, if it has one.
    pub project_file: Option<&'a Path>,
    /// The Portal file.
    pub portal_file: &'a Path,
}

impl Destination<'_> {
    /// The folder: `<workspace>/<StudyName>/<TrialName>` with a custom workspace, else
    /// `<the project file's folder>/<TrialName>`, else `<the Portal file's
    /// folder>/<TrialName>`.
    ///
    /// # Errors
    /// [`PortalError::UnsafePath`] when the study or trial name climbs out (`..`) or is
    /// absolute (ledger F55: XMALab 3 creates the folder where it leads).
    pub fn folder(&self, study_name: &str, trial_name: &str) -> Result<PathBuf, PortalError> {
        let trial = safe_relative(trial_name)?;
        let base_of = |f: &Path| {
            std::path::absolute(f)
                .unwrap_or_else(|_| f.to_path_buf())
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_default()
        };
        Ok(match self.workspace.filter(|w| !w.as_os_str().is_empty()) {
            Some(w) => w.join(safe_relative(study_name)?).join(trial),
            None => match self.project_file {
                Some(p) => base_of(p).join(trial),
                None => base_of(self.portal_file).join(trial),
            },
        })
    }
}

/// The folder a nested zip is extracted to: its path without the `.zip` ending (any case).
/// `None` for a name that does not end in `.zip`.
///
/// XMALab 3 tests `endsWith("zip")` (case-sensitive) and removes every `.zip` anywhere in
/// the full path (`QString::replace`), so a folder named `a.zip.d` above the trial sends
/// the images elsewhere, while `CameraBoxTrial::setFilename` looks for them at the path
/// without its last four characters (ledger F58, both modes).
#[must_use]
pub fn nested_zip_folder(path: &Path) -> Option<PathBuf> {
    let s = path.to_string_lossy();
    let n = s.len();
    (n > 4 && s.is_char_boundary(n - 4) && s[n - 4..].eq_ignore_ascii_case(".zip"))
        .then(|| PathBuf::from(&s[..n - 4]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_that_climb_out_are_refused() {
        assert!(entry_parts("a/../../b").is_err());
        assert!(entry_parts("../b").is_err());
        assert!(entry_parts("/etc/passwd").is_err());
        assert!(entry_parts("\\\\server\\x").is_err());
        assert!(entry_parts("C:\\x").is_err());
        assert!(entry_parts("C:x").is_err());
        assert_eq!(entry_parts("a/./b\\c").unwrap(), vec!["a", "b", "c"]);
        assert_eq!(entry_parts("a/..b").unwrap(), vec!["a", "..b"]);
    }

    /// Ledger F55: the parts no system the program is built for may be given.
    #[test]
    fn windows_forms_are_refused_in_every_part() {
        for bad in [
            "a/C:x",
            "Top/x:stream",
            "Top/a.",
            "Top/a..",
            "Top/a ",
            "Top/dir./f.tif",
            "Top/CON",
            "Top/con.txt",
            "Top/Aux.avi",
            "Top/NUL ",
            "Top/COM1",
            "Top/lpt9.png",
            "Top/COM\u{b9}",
            "Top/a?b",
            "Top/a*b",
            "Top/a|b",
            "Top/a\"b",
            "Top/a<b",
            "Top/a>b",
            "Top/a\u{1}b",
        ] {
            assert!(entry_parts(bad).is_err(), "{bad:?} accepted");
            assert!(safe_relative(bad).is_err(), "{bad:?}");
        }
        for good in [
            "Top/CONSOLE.txt",
            "Top/COM10",
            "Top/LPT0",
            "Top/con1.avi",
            "Top/.hidden",
            "Top/a b.avi",
            "Top/caf\u{e9}.tif",
        ] {
            assert!(entry_parts(good).is_ok(), "{good:?} refused");
        }
    }

    #[test]
    fn top_folder_is_first_by_name_ignoring_case() {
        assert_eq!(
            top_folder(["b/x", "A/y", ".hidden/z", "root.txt"]),
            Some("A".to_owned())
        );
        assert_eq!(top_folder(["root.txt"]), None);
        assert_eq!(top_folder(["d/"]), Some("d".to_owned()));
    }

    #[test]
    fn metadata_elements_as_xmalab_reads_them() {
        let xml = br#"<?xml version="1.0"?>
<data><StudyName>S</StudyName><TrialName>T1</TrialName>
<File><Filename>c2.avi</Filename><FileID>7</FileID><FileCategory>Trial</FileCategory>
<metadata><cameraNumber>cam2</cameraNumber><FrameRate>250</FrameRate></metadata></File>
<File><Filename>c1.avi</Filename><metadata><cameraNumber>cam1</cameraNumber><FrameRate>250</FrameRate></metadata><TrialName>inside</TrialName></File>
<File><Filename>x.txt</Filename></File>
<File><Filename>g.tif</Filename><FileCategory>DistortionGrid</FileCategory><metadata><cameraNumber>cam1</cameraNumber></metadata></File>
<File><FileCategory>CalibObject</FileCategory><metadata><cameraNumber>cam1</cameraNumber></metadata></File>
</data>"#;
        let m = Metadata::parse(xml).unwrap();
        assert_eq!(m.study_name, "S");
        assert_eq!(m.trial_name, "T1");
        assert_eq!(m.files.len(), 5);
        assert_eq!(m.files[0].file_id, 7);
        assert_eq!(m.files[1].file_id, -1);
        assert_eq!(
            m.trial_files(),
            vec![
                (2, "c2.avi".to_owned()),
                (1, "c1.avi".to_owned()),
                (1, "g.tif".to_owned())
            ]
        );
        assert_eq!(
            m.project_files(),
            vec![ProjectFile {
                camera_number: 1,
                kind: CalibrationFileKind::Grid,
                filename: "g.tif".into()
            }]
        );
        assert_eq!(m.all_camera_numbers, vec![1, 2]);
        // The third and fourth files have no rate.
        assert_eq!(m.frame_rate(false), None);
        assert_eq!(m.lines.len(), 9);
    }

    #[test]
    fn frame_rate_rules() {
        let f = |r: Option<f64>| PortalFile {
            frame_rate: r,
            ..PortalFile::default()
        };
        let m = |v: Vec<PortalFile>| Metadata {
            files: v,
            ..Metadata::default()
        };
        assert_eq!(
            m(vec![f(Some(250.0)), f(Some(250.0))]).frame_rate(false),
            Some(250.0)
        );
        assert_eq!(
            m(vec![f(Some(250.0)), f(Some(100.0))]).frame_rate(false),
            None
        );
        assert_eq!(m(vec![f(Some(0.0))]).frame_rate(false), None);
        assert_eq!(m(vec![]).frame_rate(false), None);
        assert_eq!(m(vec![f(Some(f64::INFINITY))]).frame_rate(false), None);
        assert_eq!(
            m(vec![f(Some(f64::INFINITY))]).frame_rate(true),
            Some(f64::INFINITY)
        );
    }

    #[test]
    fn camera_number_text() {
        let f = |s: Option<&str>| PortalFile {
            camera_number: s.map(str::to_owned),
            ..PortalFile::default()
        };
        assert_eq!(f(Some("cam3")).camera(), Some(3));
        assert_eq!(f(Some("Cam3")).camera(), Some(0));
        assert_eq!(f(Some("")).camera(), Some(0));
        assert_eq!(f(None).camera(), None);
    }

    #[test]
    fn malformed_metadata_is_an_error() {
        assert!(matches!(
            Metadata::parse(b"<data><File><Filename>a</Filename>"),
            Err(PortalError::Metadata(_))
        ));
    }

    #[test]
    fn nested_zip_names() {
        assert_eq!(
            nested_zip_folder(Path::new("/a.zip.d/T/seq.ZIP")),
            Some(PathBuf::from("/a.zip.d/T/seq"))
        );
        assert_eq!(nested_zip_folder(Path::new("/a/seqzip")), None);
        assert_eq!(nested_zip_folder(Path::new("/a/c.avi")), None);
    }

    #[test]
    fn destination_rules() {
        let d = Destination {
            workspace: Some(Path::new("/ws")),
            project_file: Some(Path::new("/p/x.xma")),
            portal_file: Path::new("/dl/t.xmatrial"),
        };
        assert_eq!(d.folder("S", "T").unwrap(), PathBuf::from("/ws/S/T"));
        let d = Destination {
            workspace: Some(Path::new("")),
            ..d
        };
        assert_eq!(d.folder("S", "T").unwrap(), PathBuf::from("/p/T"));
        let d = Destination {
            workspace: None,
            project_file: None,
            ..d
        };
        assert_eq!(d.folder("S", "T").unwrap(), PathBuf::from("/dl/T"));
        assert!(d.folder("S", "../T").is_err());
        assert!(d.folder("S", "/T").is_err());
        let d = Destination {
            workspace: Some(Path::new("/ws")),
            ..d
        };
        assert!(d.folder("../S", "T").is_err());
    }
}
