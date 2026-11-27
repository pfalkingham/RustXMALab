//! Writing `.xma` project files.
//!
//! Ported from `ui/ProjectFileIO.cpp` (`ProjectFileIO::saveProject`,
//! `ProjectFileIO::writeProjectFile`, `ProjectFileIO::zipFromFolderToFile`,
//! `ProjectFileIO::recurseAddDir`) and the save functions they reach in `core/`. The result
//! is, entry for entry, what XMALab 3 on Linux writes from the same state: the same entry
//! names in the same order, the same bytes in every entry the model holds, the same zip
//! flags and archive comment. What the model does not hold (images, `log.html`,
//! `xma_metadata.xml`, `PrecisionInfo_<trial>.txt`, `MarkerDistances_<trial>.txt`) comes
//! ready-made from [`Extras`] and is placed where the C++ puts it.
//!
//! # How the C++ saves, and how this module reproduces it
//!
//! The C++ writes every file into `$TMP/XROMM_tmp/`, then zips that folder. Here the files
//! are collected in memory (parity ledger F5) in the order the C++ writes them, into a
//! model of that folder (`Folder`) that behaves as the file system does for the C++:
//!
//! - a file is written only into a folder the C++ created with `QDir::mkpath` (an event
//!   name containing `/` names a folder that does not exist, so the C++ writes nothing);
//! - a later file with the same path replaces an earlier one (an event named
//!   `Marker000points2d` is overwritten by the marker file);
//! - paths are resolved as the file system resolves them (`a//b`, `a/./b`, `a/../b`); a
//!   path that leaves the folder is written outside it by the C++ and is not archived.
//!
//! The folder is then archived as `recurseAddDir` collects it: depth first, each folder's
//! entries in `QDir`'s default order (by name, ignoring case: see [`qt_name_order`]), files
//! and folders interleaved, hidden names (starting with `.`) left out because `QDir`'s
//! default filter skips them. Empty folders give no entry.
//!
//! Line ends are `\n` on every platform (the Linux build's), path separators in
//! `project.xml` are `/` (ledger F5), and numbers in data files have 17 significant digits
//! (ledger F10). Archive entries are written by [`crate::zip::write_archive`].
//!
//! # When a file is written
//!
//! As in the C++ (`saveProject` and the functions it calls):
//! - `project.xml` always; `xma_metadata.xml` always in the C++ (here from [`Extras`]);
//!   `projectMetaData/metadata.xml` when the project has portal study data;
//! - per camera, in `<camera>/data/`: the camera matrix when calibrated, the lens distortion
//!   when calibrated and the lens model is in use; the undistortion grid image in
//!   `<camera>/` whenever the camera has a grid, and its four CSV files when the grid is
//!   computed; per calibration image, the image in `<camera>/` when the camera has no
//!   calibration sequence and the calibration is internal, and its five CSV files when the
//!   image is calibrated;
//! - `CalibrationObject/<spec>` and `CalibrationObject/<ref>` when the calibration object is
//!   a frame (not a checkerboard), whatever the calibration type;
//! - per trial: `PrecisionInfo_<trial>.txt` and `MarkerDistances_<trial>.txt` when the trial
//!   does not require recomputation (here from [`Extras`]); `<trial>/metadata.xml` with
//!   portal data; `<trial>/<camera>_filenames_absolute.txt` except for the default trial;
//!   in `<trial>/data/`: `MarkerDescription.txt`, `RigidBodies.txt`, one `<event>.csv` per
//!   event, per marker the five track files, the interpolation file and, when set, the
//!   reference point; per rigid body the reference names and points when the references
//!   were set from a file (`references_set == 2`), the optimised points when optimised, and
//!   per virtual point its reference file and either its coordinate file (imported
//!   coordinates) or its second reference file (a point of another body);
//! - `log.html` always (here from [`Extras`], empty when not supplied);
//! - not in the C++: the settings entry [`crate::settings_entry::SETTINGS_ENTRY`], always,
//!   last in the archive, with `project.algorithm_settings` (ledger A10). It is the one
//!   entry XMALab 3 does not write; it ignores it on loading and drops it on saving.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fmt::Write as _;
use std::path::Path;

use xma_model::{
    AdjustedView, CalibrationObject, CalibrationType, Camera, CppCompat, Marker, Project,
    RigidBody, Trial, VirtualPointSource,
};

use crate::cfmt::{
    format_bool, format_g17, format_int, format_int_padded, format_qt_number, push_g17,
};
use crate::project_read::{Attachment, LoadedProject};
use crate::settings_entry::{SETTINGS_ENTRY, write_settings};
use crate::xml::XmlWriter;
use crate::zip::{DosDateTime, NewEntry, ZipError, encode_cp866, prepare_archive, write_archive};

/// The archive comment XMALab writes (`ui/ProjectFileIO.cpp`, `saveProject`).
pub const ARCHIVE_COMMENT: &str = "XROMM Project File";

/// The `Version` attribute XMALab 3 writes.
pub const PROJECT_FILE_VERSION: &str = "0.3";

/// Archive entries the caller supplies ready-made, by name (as the C++ names them in the
/// archive, with `/` separators): calibration and grid images, `log.html`,
/// `xma_metadata.xml`, `PrecisionInfo_<trial>.txt` and `MarkerDistances_<trial>.txt`.
///
/// The writer takes from here only what the C++ would write for the project being saved,
/// and ignores the rest ([`ProjectEntries::unused_extras`] lists it).
///
/// Each entry's data is owned ([`ExtraFiles::insert`]) or borrowed
/// ([`ExtraFiles::insert_borrowed`], [`ExtraFiles::borrowing`]): a save can then pass the
/// images a document holds without copying them. [`Extras`] owns all its data.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExtraFiles<'a> {
    files: BTreeMap<String, Cow<'a, [u8]>>,
}

/// [`ExtraFiles`] that owns its data.
pub type Extras = ExtraFiles<'static>;

impl<'a> ExtraFiles<'a> {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds or replaces an entry. The name is resolved as a path (`\` is a separator,
    /// `a//b` is `a/b`).
    pub fn insert(&mut self, name: &str, data: Vec<u8>) {
        self.files
            .insert(resolve(name).unwrap_or_default(), Cow::Owned(data));
    }

    /// [`ExtraFiles::insert`] of data that stays where it is.
    pub fn insert_borrowed(&mut self, name: &str, data: &'a [u8]) {
        self.files
            .insert(resolve(name).unwrap_or_default(), Cow::Borrowed(data));
    }

    /// The attachments that are files (not folders), borrowed: what
    /// `Extras::from(attachments)` holds, without copying the data.
    #[must_use]
    pub fn borrowing(attachments: &'a [Attachment]) -> Self {
        let mut extras = Self::new();
        for a in attachments {
            if !a.entry.is_dir() {
                extras.insert_borrowed(&a.name, &a.data);
            }
        }
        extras
    }

    /// The data of an entry, if supplied.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&[u8]> {
        self.files
            .get(&resolve(name).unwrap_or_default())
            .map(|d| &**d)
    }

    /// Names of the entries, sorted.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.files.keys().map(String::as_str)
    }
}

impl From<&[Attachment]> for Extras {
    fn from(attachments: &[Attachment]) -> Self {
        let mut extras = Extras::new();
        for a in attachments {
            if !a.entry.is_dir() {
                extras.insert(&a.name, a.data.clone());
            }
        }
        extras
    }
}

impl From<Vec<Attachment>> for Extras {
    fn from(attachments: Vec<Attachment>) -> Self {
        let mut extras = Extras::new();
        for a in attachments {
            if !a.entry.is_dir() {
                extras.insert(&a.name, a.data);
            }
        }
        extras
    }
}

impl From<&LoadedProject> for Extras {
    fn from(loaded: &LoadedProject) -> Self {
        Extras::from(loaded.attachments.as_slice())
    }
}

/// Options for writing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WriteOptions {
    /// Modification time of every entry. The C++ stamps each entry with the time its
    /// temporary file was written, in local time; a save normally falls in one or two
    /// two-second steps. Taking it from here makes output reproducible.
    pub modified: DosDateTime,
    /// The trials to save, by index, in the order to save them. `None` saves every trial,
    /// as "Save" and "Save As" do; `Some` is "Save subset of trials"
    /// (`MainWindow::saveProjectAs(true)`), which saves the chosen trials and leaves
    /// `ActiveTrial` as the index in the whole project.
    pub trials: Option<Vec<usize>>,
    /// Which faults of XMALab 3 to reproduce; writing looks at
    /// [`CppCompat::drop_unreadable_mesh`] (ledger F18): with it, a body's mesh is written
    /// only when its file could be read ([`RigidBody::mesh_readable`] not `Some(false)`), as
    /// the C++ writes only a mesh it holds. Default: fixed, every mesh path written.
    pub compat: CppCompat,
}

impl Default for WriteOptions {
    /// 1 January 1980, 00:00:00 (the zip epoch); every trial.
    fn default() -> Self {
        Self {
            modified: DosDateTime::from_fields(1980, 1, 1, 0, 0, 0).unwrap_or_default(),
            trials: None,
            compat: CppCompat::default(),
        }
    }
}

/// Why a project could not be written.
#[derive(Debug)]
pub enum WriteError {
    /// The file could not be written.
    Io(std::io::Error),
    /// The archive cannot be built (an entry of 4 GiB or more, more than 65535 entries).
    Zip(ZipError),
    /// An image the C++ would write is not in [`Extras`] (images are not encoded here).
    MissingImage(String),
    /// The model is inconsistent in a way the C++ would read out of bounds on.
    Data(String),
}

impl fmt::Display for WriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WriteError::Io(e) => write!(f, "{e}"),
            WriteError::Zip(e) => write!(f, "{e}"),
            WriteError::MissingImage(name) => {
                write!(f, "image '{name}' is needed and was not supplied")
            }
            WriteError::Data(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for WriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WriteError::Io(e) => Some(e),
            WriteError::Zip(e) => Some(e),
            _ => None,
        }
    }
}

/// The files of a project, in archive order, before they are zipped.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProjectEntries {
    /// Name (with `/` separators) and data of each entry, in the order the C++ archives them.
    pub files: Vec<(String, Vec<u8>)>,
    /// Files the C++ would write but that could not be produced, because they are neither in
    /// the model nor in [`Extras`] (`xma_metadata.xml`, `PrecisionInfo_*`,
    /// `MarkerDistances_*`, `log.html`). `log.html` is written empty in that case, as the
    /// C++ writes an empty log; the others are left out.
    pub not_written: Vec<String>,
    /// Extras that are not part of the project as saved, sorted by name.
    pub unused_extras: Vec<String>,
    /// Files the C++ would write but would leave out of the archive or not write at all:
    /// hidden names, paths outside the project folder, files in folders that do not exist.
    pub dropped: Vec<String>,
}

/// [`ProjectEntries`] whose data may be borrowed from the [`ExtraFiles`] it was built
/// from ([`project_entries_borrowed`]), so that no image is copied before it is compressed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BorrowedEntries<'e> {
    /// As [`ProjectEntries::files`].
    pub files: Vec<(String, Cow<'e, [u8]>)>,
    /// As [`ProjectEntries::not_written`].
    pub not_written: Vec<String>,
    /// As [`ProjectEntries::unused_extras`].
    pub unused_extras: Vec<String>,
    /// As [`ProjectEntries::dropped`].
    pub dropped: Vec<String>,
}

impl BorrowedEntries<'_> {
    /// The entries with every file's data owned.
    #[must_use]
    pub fn into_owned(self) -> ProjectEntries {
        ProjectEntries {
            files: self
                .files
                .into_iter()
                .map(|(n, d)| (n, d.into_owned()))
                .collect(),
            not_written: self.not_written,
            unused_extras: self.unused_extras,
            dropped: self.dropped,
        }
    }
}

/// Writes a project to a `.xma` file.
///
/// The entries are compressed in memory (each entry's data borrowed, not copied), then
/// written to a temporary file next to `path`, which is renamed over `path`, so an existing
/// file is replaced only by a complete one. (The C++ deletes the old file first, then zips
/// into its place.) Every error of the archive is known before the file is created.
///
/// Ported from ui/ProjectFileIO.cpp, ProjectFileIO::saveProject.
pub fn write_project(
    path: &Path,
    project: &Project,
    extras: &ExtraFiles<'_>,
    options: &WriteOptions,
) -> Result<(), WriteError> {
    let entries = project_entries_borrowed(project, extras, options)?;
    write_borrowed_entries(path, &entries, options.modified)
}

/// Zips entries already built by [`project_entries`] and writes them to `path` as
/// [`write_project`] does. For a caller that wants the entries' report
/// ([`ProjectEntries::not_written`] and the rest) as well as the file.
pub fn write_entries(
    path: &Path,
    entries: &ProjectEntries,
    modified: DosDateTime,
) -> Result<(), WriteError> {
    write_zipped(path, &entries.files, modified)
}

/// [`write_entries`] for entries built by [`project_entries_borrowed`].
pub fn write_borrowed_entries(
    path: &Path,
    entries: &BorrowedEntries<'_>,
    modified: DosDateTime,
) -> Result<(), WriteError> {
    write_zipped(path, &entries.files, modified)
}

/// Zips `files` as [`zip_entries`] does and writes the archive to `path` without building
/// it in one buffer.
fn write_zipped<D: AsRef<[u8]>>(
    path: &Path,
    files: &[(String, D)],
    modified: DosDateTime,
) -> Result<(), WriteError> {
    let names = cp866_names(files);
    let list = new_entries(files, &names, modified);
    let archive = prepare_archive(&list, ARCHIVE_COMMENT.as_bytes()).map_err(WriteError::Zip)?;
    write_file_atomically(path, |f| {
        let mut w = std::io::BufWriter::with_capacity(1 << 16, f);
        archive.write_to(&mut w)?;
        std::io::Write::flush(&mut w)?;
        Ok(())
    })
}

/// Writes to a temporary file next to `path`, then renames it over `path`.
///
/// The temporary file is flushed to the disk before the rename, and removed when any step
/// fails, so a failed write leaves the previous file as it was and no partial file behind
/// (parity ledger U54: XMALab 3 deletes the old file first and writes the archive in its
/// place). Each call uses its own temporary name, so two writes in one process do not meet.
fn write_file_atomically(
    path: &Path,
    write: impl FnOnce(&mut std::fs::File) -> std::io::Result<()>,
) -> Result<(), WriteError> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let tmp = path.with_file_name(format!(".{file_name}.{}.{n}.tmp", std::process::id()));
    let written = std::fs::File::create(&tmp).and_then(|mut f| {
        write(&mut f)?;
        f.sync_all()
    });
    let result = written.and_then(|()| std::fs::rename(&tmp, path));
    result.map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        WriteError::Io(e)
    })
}

/// The bytes of a `.xma` file for a project. See [`write_project`].
pub fn write_project_bytes(
    project: &Project,
    extras: &ExtraFiles<'_>,
    options: &WriteOptions,
) -> Result<Vec<u8>, WriteError> {
    let entries = project_entries_borrowed(project, extras, options)?;
    zip_files(&entries.files, options.modified)
}

/// Zips entries as `zipFromFolderToFile` does, with the archive comment
/// [`ARCHIVE_COMMENT`] and names in code page 866 (ledger F6).
pub fn zip_entries(entries: &ProjectEntries, modified: DosDateTime) -> Result<Vec<u8>, WriteError> {
    zip_files(&entries.files, modified)
}

/// [`zip_entries`] of the files of [`ProjectEntries`] or [`BorrowedEntries`].
fn zip_files<D: AsRef<[u8]>>(
    files: &[(String, D)],
    modified: DosDateTime,
) -> Result<Vec<u8>, WriteError> {
    let names = cp866_names(files);
    let list = new_entries(files, &names, modified);
    write_archive(&list, ARCHIVE_COMMENT.as_bytes()).map_err(WriteError::Zip)
}

fn cp866_names<D>(files: &[(String, D)]) -> Vec<Vec<u8>> {
    files.iter().map(|(n, _)| encode_cp866(n)).collect()
}

fn new_entries<'a, D: AsRef<[u8]>>(
    files: &'a [(String, D)],
    names: &'a [Vec<u8>],
    modified: DosDateTime,
) -> Vec<NewEntry<'a>> {
    files
        .iter()
        .zip(names)
        .map(|((_, data), name)| NewEntry {
            name,
            data: data.as_ref(),
            modified,
        })
        .collect()
}

/// The files of a project in archive order, with their contents.
///
/// Ported from ui/ProjectFileIO.cpp, ProjectFileIO::saveProject and
/// ProjectFileIO::zipFromFolderToFile.
pub fn project_entries(
    project: &Project,
    extras: &ExtraFiles<'_>,
    options: &WriteOptions,
) -> Result<ProjectEntries, WriteError> {
    project_entries_borrowed(project, extras, options).map(BorrowedEntries::into_owned)
}

/// [`project_entries`] with the data of the entries taken from `extras` borrowed, not
/// copied.
pub fn project_entries_borrowed<'e>(
    project: &Project,
    extras: &'e ExtraFiles<'_>,
    options: &WriteOptions,
) -> Result<BorrowedEntries<'e>, WriteError> {
    // Ledger F49: in the default mode no NaN or infinity is written; each is written as its
    // field's "not set" value (`crate::finite`), which the reader reads back as it is.
    let cleared;
    let project = if !options.compat.nonfinite_values_kept && crate::finite::has_non_finite(project)
    {
        let mut p = project.clone();
        crate::finite::clear_non_finite(&mut p);
        cleared = p;
        &cleared
    } else {
        project
    };
    let trial_indices: Vec<usize> = match &options.trials {
        Some(list) => list.clone(),
        None => (0..project.trials.len()).collect(),
    };
    let mut trials = Vec::with_capacity(trial_indices.len());
    for &t in &trial_indices {
        trials.push(project.trials.get(t).ok_or_else(|| {
            WriteError::Data(format!(
                "trial {t} does not exist ({} trials)",
                project.trials.len()
            ))
        })?);
    }
    check_project(project, &trials)?;

    let mut s = Saver {
        folder: Folder::default(),
        extras,
        used: BTreeSet::new(),
        not_written: Vec::new(),
    };

    // saveProject: project.xml, then the portal summary.
    s.write(
        "project.xml",
        write_project_file(project, &trials, options.compat).into_bytes(),
    );
    s.write_extra("xma_metadata.xml");

    let has_study_data = !project.portal_metadata.is_empty();
    if has_study_data {
        s.folder.mkpath("projectMetaData");
        s.write(
            "projectMetaData/metadata.xml",
            text_lines(&project.portal_metadata),
        );
    }

    // Cameras.
    for cam in &project.cameras {
        let dir = cam.name.as_str();
        s.folder.mkpath(dir);
        s.folder.mkpath(&format!("{dir}/data"));
        save_camera(&mut s, project, cam)?;
    }

    // Calibration object.
    if let Some(CalibrationObject::Frame(frame)) = &project.calibration_object {
        s.folder.mkpath("CalibrationObject");
        // Ported from core/CalibrationObject.cpp, CalibrationObject::saveCoords.
        let mut spec = String::from(if frame.white_blobs {
            "white\n"
        } else {
            "x y z\n"
        });
        for (i, p) in frame.points.iter().enumerate() {
            let name = frame.point_names.get(i).map_or("", String::as_str);
            spec.push_str(&format!(
                "{},{},{},{name}\n",
                format_g17(p.x),
                format_g17(p.y),
                format_g17(p.z)
            ));
        }
        s.write(
            &format!("CalibrationObject/{}", frame.frame_specification_filename),
            spec.into_bytes(),
        );
        let mut refs = String::new();
        for (id, name) in frame.reference_ids.iter().zip(&frame.reference_names) {
            refs.push_str(&format!("{} {name}\n", format_int(i64::from(*id) + 1)));
        }
        s.write(
            &format!("CalibrationObject/{}", frame.references_filename),
            refs.into_bytes(),
        );
    }

    // Trials.
    for trial in &trials {
        save_trial(&mut s, project, trial)?;
    }

    // ConsoleDockWidget::save
    if extras.get("log.html").is_some() {
        s.write_extra("log.html");
    } else {
        s.not_written.push("log.html".to_owned());
        s.write("log.html", Vec::new());
    }

    let (mut files, dropped) = s.folder.archive_order_borrowed();
    // Not written by XMALab: the settings that change results (ledger A10), last, so that
    // every entry XMALab writes keeps its place. XMALab 3 ignores the entry on loading and
    // drops it on saving ([`crate::settings_entry`]).
    files.push((
        SETTINGS_ENTRY.to_owned(),
        Cow::Owned(write_settings(&project.algorithm_settings)),
    ));
    let unused_extras = extras
        .names()
        .filter(|n| !s.used.contains(*n))
        .map(str::to_owned)
        .collect();
    Ok(BorrowedEntries {
        files,
        not_written: s.not_written,
        unused_extras,
        dropped,
    })
}

// ---------------------------------------------------------------------------------------
// The temporary folder

/// Resolves a path relative to the project folder as the file system would: `/` and
/// (for names coming from a Windows build) `\` separate, empty and `.` components vanish,
/// `..` goes up. `None` when the path leaves the folder.
fn resolve(path: &str) -> Option<String> {
    resolve_split(path, &['/', '\\'])
}

/// Resolves a path the C++ writes, with `/` as the only separator (Linux).
fn resolve_linux(path: &str) -> Option<String> {
    resolve_split(path, &['/'])
}

/// [`resolve`] with the given separators.
fn resolve_split(path: &str, separators: &[char]) -> Option<String> {
    let mut parts: Vec<&str> = Vec::new();
    for p in path.split(separators) {
        match p {
            "" | "." => {}
            ".." => {
                parts.pop()?;
            }
            other => parts.push(other),
        }
    }
    Some(parts.join("/"))
}

/// A file of the folder: its path and data.
type FolderFile<'e> = (String, Cow<'e, [u8]>);

/// The C++'s `$TMP/XROMM_tmp/` folder, in memory. A file taken from [`ExtraFiles`] is
/// borrowed from it.
#[derive(Default)]
struct Folder<'e> {
    dirs: BTreeSet<String>,
    files: BTreeMap<String, Cow<'e, [u8]>>,
    dropped: Vec<String>,
}

impl<'e> Folder<'e> {
    /// `QDir().mkpath(path)`: the folder and every folder above it.
    fn mkpath(&mut self, path: &str) {
        let Some(p) = resolve_linux(path) else {
            return;
        };
        let mut cur = String::new();
        for part in p.split('/').filter(|s| !s.is_empty()) {
            if !cur.is_empty() {
                cur.push('/');
            }
            cur.push_str(part);
            if self.files.contains_key(&cur) {
                return; // a file is in the way; mkpath fails
            }
            self.dirs.insert(cur.clone());
        }
    }

    /// `std::ofstream` (or `QFile`, `cv::imwrite`) on `path`: replaces the file, or writes
    /// nothing when the folder does not exist or the path is a folder.
    fn write(&mut self, path: &str, data: impl Into<Cow<'e, [u8]>>) {
        let data = data.into();
        let Some(p) = resolve_linux(path) else {
            self.dropped.push(path.to_owned());
            return;
        };
        let parent = p.rsplit_once('/').map_or("", |(d, _)| d);
        let parent_ok = parent.is_empty() || self.dirs.contains(parent);
        if p.is_empty() || !parent_ok || self.dirs.contains(&p) {
            self.dropped.push(path.to_owned());
            return;
        }
        self.files.insert(p, data);
    }

    /// [`Folder::archive_order_borrowed`] with the data owned.
    #[cfg(test)]
    fn archive_order(self) -> (Vec<(String, Vec<u8>)>, Vec<String>) {
        let (files, dropped) = self.archive_order_borrowed();
        let files = files
            .into_iter()
            .map(|(n, d)| (n, d.into_owned()))
            .collect();
        (files, dropped)
    }

    /// The files as `recurseAddDir` lists them, and the names it leaves out.
    fn archive_order_borrowed(self) -> (Vec<FolderFile<'e>>, Vec<String>) {
        #[derive(Default)]
        struct Node<'e> {
            children: BTreeMap<String, Node<'e>>,
            file: Option<Cow<'e, [u8]>>,
        }
        let mut root = Node::default();
        let mut dropped = self.dropped;
        for (path, data) in self.files {
            if path.split('/').any(|c| c.starts_with('.')) {
                dropped.push(path);
                continue;
            }
            let mut node = &mut root;
            for part in path.split('/') {
                node = node.children.entry(part.to_owned()).or_default();
            }
            node.file = Some(data);
        }
        fn walk<'e>(node: Node<'e>, prefix: &str, out: &mut Vec<FolderFile<'e>>) {
            let mut children: Vec<(String, Node<'e>)> = node.children.into_iter().collect();
            children.sort_by(|a, b| qt_name_order(&a.0, &b.0));
            for (name, child) in children {
                let path = if prefix.is_empty() {
                    name
                } else {
                    format!("{prefix}/{name}")
                };
                match child.file {
                    Some(data) => out.push((path, data)),
                    None => walk(child, &path, out),
                }
            }
        }
        let mut out = Vec::new();
        walk(root, "", &mut out);
        (out, dropped)
    }
}

/// `QDir`'s default order (`QDir::Name | QDir::IgnoreCase`): names compared as
/// `QString::compare(.., Qt::CaseInsensitive)`, which folds each character's case and
/// compares UTF-16 code units.
///
/// Case is folded with Rust's lower-case mapping where it gives one character, which is
/// Qt's simple case folding for every character except a few (for example `ſ`, which Qt
/// folds to `s`). Names equal apart from case come out in an order that depends on the
/// file system in the C++ (checked with Qt 6.11 on tmpfs: the reverse of creation order);
/// here they are ordered by their UTF-16 code units, so the result is reproducible.
#[must_use]
pub fn qt_name_order(a: &str, b: &str) -> std::cmp::Ordering {
    fn fold(s: &str) -> Vec<u16> {
        let mut out = Vec::with_capacity(s.len());
        let mut buf = [0u16; 2];
        for c in s.chars() {
            let mut lower = c.to_lowercase();
            let folded = match (lower.next(), lower.next()) {
                (Some(l), None) => l,
                _ => c,
            };
            out.extend_from_slice(folded.encode_utf16(&mut buf));
        }
        out
    }
    fold(a)
        .cmp(&fold(b))
        .then_with(|| a.encode_utf16().cmp(b.encode_utf16()))
}

struct Saver<'e, 'x> {
    folder: Folder<'e>,
    extras: &'e ExtraFiles<'x>,
    used: BTreeSet<String>,
    not_written: Vec<String>,
}

impl Saver<'_, '_> {
    fn write(&mut self, path: &str, data: Vec<u8>) {
        self.folder.write(path, Cow::Owned(data));
    }

    /// A file the C++ always writes and the model does not hold.
    fn write_extra(&mut self, name: &str) {
        let key = resolve(name).unwrap_or_default();
        match self.extras.get(&key) {
            Some(data) => {
                self.used.insert(key);
                self.folder.write(name, Cow::Borrowed(data));
            }
            None => self.not_written.push(key),
        }
    }

    /// An image the C++ writes with `cv::imwrite`.
    fn write_image(&mut self, name: &str) -> Result<(), WriteError> {
        let key = resolve(name).unwrap_or_default();
        match self.extras.get(&key) {
            Some(data) => {
                self.used.insert(key);
                self.folder.write(name, Cow::Borrowed(data));
                Ok(())
            }
            None => Err(WriteError::MissingImage(key)),
        }
    }
}

// ---------------------------------------------------------------------------------------
// Checks

/// Inconsistencies the C++ would index out of bounds on when saving.
fn check_project(project: &Project, trials: &[&std::sync::Arc<Trial>]) -> Result<(), WriteError> {
    for trial in trials {
        if trial.videos.len() > project.cameras.len() {
            return Err(WriteError::Data(format!(
                "trial '{}' has {} videos for {} cameras",
                trial.name,
                trial.videos.len(),
                project.cameras.len()
            )));
        }
        for (k, body) in trial.rigid_bodies.iter().enumerate() {
            if body.has_optimized_coordinates
                && body.references_set == 2
                && body.reference_points_original.len() < body.reference_points.len()
            {
                return Err(WriteError::Data(format!(
                    "trial '{}', rigid body {k}: fewer original reference points than points",
                    trial.name
                )));
            }
            if let Some(&m) = body
                .marker_indices
                .iter()
                .find(|&&m| m >= trial.markers.len())
            {
                return Err(WriteError::Data(format!(
                    "trial '{}', rigid body {k}: marker {m} does not exist",
                    trial.name
                )));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------
// project.xml

/// `QString::number(int)`, `QString::number(bool)`.
fn qn(v: impl Into<i64>) -> String {
    format_int(v.into())
}

/// `QString::number(uint)` of a count or size held as `usize`.
fn qnu(v: usize) -> String {
    v.to_string()
}

/// The text of `project.xml`.
///
/// Ported from ui/ProjectFileIO.cpp, ProjectFileIO::writeProjectFile.
/// A number in an XML attribute: `QString::number` (6 significant digits) as XMALab 3
/// writes it ([`CppCompat::six_digit_attributes`], ledger F4), or `%.17g` (fixed), which
/// `QString::toDouble` reads back exactly.
fn attr_number(compat: CppCompat, v: f64) -> String {
    if compat.six_digit_attributes {
        format_qt_number(v)
    } else {
        format_g17(v)
    }
}

fn write_project_file(
    project: &Project,
    trials: &[&std::sync::Arc<Trial>],
    compat: CppCompat,
) -> String {
    let mut w = XmlWriter::new();
    w.start_document();
    w.start_element("Project");
    w.attribute("Version", PROJECT_FILE_VERSION);
    w.attribute("DateCreated", &project.date_created);
    // `State::getActiveTrial()` is -1 while the project has no trial (a new project, or
    // one whose trials were all deleted), and XMALab writes it so.
    if project.trials.is_empty() {
        w.attribute("ActiveTrial", "-1");
    } else {
        w.attribute("ActiveTrial", &qnu(project.active_trial));
    }
    if !project.portal_metadata.is_empty() {
        w.attribute("MetaData", "projectMetaData/metadata.xml");
    }
    w.attribute("CalibrationType", &qn(project.calibration_type.code()));

    for cam in &project.cameras {
        write_camera_xml(&mut w, project, cam, compat);
    }

    if project.calibration_type == CalibrationType::Internal {
        w.start_element("CalibrationObject");
        match &project.calibration_object {
            Some(CalibrationObject::Checkerboard {
                horizontal_squares,
                vertical_squares,
                square_size,
                inverted_axis,
            }) => {
                w.attribute("isPLanar", "1");
                w.attribute("HorizontalSquares", &qn(*horizontal_squares));
                w.attribute("VerticalSquares", &qn(*vertical_squares));
                w.attribute("SquareSize", &attr_number(compat, *square_size));
                // Ledger F73: the board's inverted axis, which XMALab 3 does not store
                // (and ignores when reading).
                if !compat.checkerboard_axis_from_setting {
                    w.attribute("InvertedAxis", &qn(*inverted_axis));
                }
            }
            Some(CalibrationObject::Frame(frame)) => {
                w.attribute("isPLanar", "0");
                w.attribute(
                    "FrameSpecifications",
                    &format!(
                        "CalibrationObject/{}",
                        file_name(&frame.frame_specification_filename)
                    ),
                );
                w.attribute(
                    "References",
                    &format!(
                        "CalibrationObject/{}",
                        file_name(&frame.references_filename)
                    ),
                );
            }
            None => {
                // The C++ singleton is a frame with empty file names until one is loaded.
                w.attribute("isPLanar", "0");
                w.attribute("FrameSpecifications", "CalibrationObject/");
                w.attribute("References", "CalibrationObject/");
            }
        }
        w.end_element();
    }

    for trial in trials {
        write_trial_xml(&mut w, project, trial, compat);
    }
    w.end_document()
}

/// `QFileInfo::fileName`: the part after the last `/` (Qt's separator), or `\\` (a path
/// held as Windows gives it).
fn file_name(path: &str) -> &str {
    crate::export::file_name_of(path)
}

/// `QFileInfo::completeBaseName`: the file name up to its last `.`.
pub(crate) fn complete_base_name(path: &str) -> &str {
    let name = file_name(path);
    name.rsplit_once('.').map_or(name, |(base, _)| base)
}

/// The name the C++ gives a calibration image: its file name, or `Frame` and a six-digit
/// index for a frame of a calibration sequence (`core/CalibrationSequence.cpp`,
/// `setCalibrationSequence`).
fn calibration_image_name(cam: &Camera, index: usize) -> String {
    if cam.calibration_sequence.is_some() {
        format!("Frame{}", format_int_padded(index as u64, 6))
    } else {
        file_name(&cam.calibration_images[index].filename).to_owned()
    }
}

fn write_camera_xml(w: &mut XmlWriter, project: &Project, cam: &Camera, compat: CppCompat) {
    let name = &cam.name;
    w.start_element("Camera");
    w.attribute("Name", name);
    w.attribute("isLightCamera", &format_bool(cam.is_light_camera));
    w.attribute("isCalibrated", &format_bool(cam.calibrated));
    w.attribute("isOptimized", &format_bool(cam.optimized));
    w.attribute("isFlipped", &format_bool(cam.flipped));
    for (view, prefix) in [
        (AdjustedView::Calibration, "Calibration"),
        (AdjustedView::Digitization, "Digitization"),
        (AdjustedView::Detail, "Detail"),
    ] {
        let adj = cam.adjustment(view);
        if adj.scale != 1.0 {
            w.attribute(&format!("{prefix}Scale"), &attr_number(compat, adj.scale));
        }
        if adj.bias != 0.0 {
            w.attribute(&format!("{prefix}Bias"), &attr_number(compat, adj.bias));
        }
    }
    if cam.calibrated {
        w.attribute(
            "CameraMatrix",
            &format!("{name}/data/{name}_CameraMatrix.csv"),
        );
        if cam.has_model_distortion {
            w.attribute(
                "UndistortionParameter",
                &format!("{name}/data/{name}_UndistortionParameter.csv"),
            );
        }
    }

    if let Some(grid) = &cam.undistortion {
        let base = complete_base_name(&grid.image_filename);
        w.start_element("UndistortionGrid");
        w.attribute(
            "Filename",
            &format!("{name}/{}", file_name(&grid.image_filename)),
        );
        w.attribute("isComputed", &format_bool(grid.computed));
        if grid.computed {
            if let Some(c) = grid.center {
                w.start_element("Center");
                w.attribute("x", &attr_number(compat, c.x));
                w.attribute("y", &attr_number(compat, c.y));
                w.end_element();
            }
            for (element, suffix) in [
                ("PointsDetected", "PointsDetected"),
                ("GridPointsDistorted", "GridPointsDistorted"),
                ("GridPointsReferences", "GridPointsReferences"),
                ("GridPointsInlier", "GridPointsInlier"),
            ] {
                w.start_element(element);
                w.attribute("Filename", &format!("{name}/data/{base}_{suffix}.csv"));
                w.end_element();
            }
        }
        w.end_element();
    }

    let sequence = cam.calibration_sequence.as_ref();
    if let Some(seq) = sequence {
        w.start_element("CalibrationSequence");
        w.attribute("Filename", &seq.filename);
        w.attribute("Frames", &qnu(seq.n_frames));
        w.attribute("Width", &qn(seq.width));
        w.attribute("Height", &qn(seq.height));
        w.end_element();
    }
    for (count, image) in cam.calibration_images.iter().enumerate() {
        let calibrated = image.calibrated > 0;
        if sequence.is_some() {
            if calibrated {
                w.start_element("CalibrationImage");
                w.attribute("Frame", &qnu(count));
                w.attribute("isCalibrated", &qn(image.calibrated));
            }
        } else {
            w.start_element("CalibrationImage");
            if project.calibration_type == CalibrationType::External {
                w.attribute("Filename", "external");
                w.attribute("Width", &qn(cam.width));
                w.attribute("Height", &qn(cam.height));
            } else {
                w.attribute(
                    "Filename",
                    &format!("{name}/{}", file_name(&image.filename)),
                );
            }
            w.attribute("isCalibrated", &qn(image.calibrated));
        }
        if calibrated {
            let base = calibration_image_base(cam, count);
            for (element, suffix) in [
                ("PointsDetectedAll", "PointsDetectedAll"),
                ("PointsDetected", "PointsDetected"),
                ("Inlier", "PointsInlier"),
                ("RotationMatrix", "RotationMatrix"),
                ("TranslationVector", "TranslationVector"),
            ] {
                w.start_element(element);
                w.attribute("Filename", &format!("{name}/data/{base}_{suffix}.csv"));
                w.end_element();
            }
        }
        if sequence.is_none() || calibrated {
            w.end_element();
        }
    }
    w.end_element();
}

/// `CalibrationImage::getFilenameBase` and the prefix of its CSV file names.
pub(crate) fn calibration_image_base(cam: &Camera, index: usize) -> String {
    complete_base_name(&calibration_image_name(cam, index)).to_owned()
}

pub(crate) fn marker_file(trial: &str, k: usize, what: &str) -> String {
    format!(
        "{trial}/data/Marker{}{what}.csv",
        format_int_padded(k as u64, 3)
    )
}

pub(crate) fn body_file(trial: &str, k: usize, what: &str) -> String {
    format!(
        "{trial}/data/RigidBody{}{what}.csv",
        format_int_padded(k as u64, 3)
    )
}

pub(crate) fn dummy_file(trial: &str, k: usize, p: usize, what: &str) -> String {
    format!(
        "{trial}/data/RigidBody{}DummyMarker{}{what}.csv",
        format_int_padded(k as u64, 3),
        format_int_padded(p as u64, 3)
    )
}

fn write_trial_xml(w: &mut XmlWriter, _project: &Project, trial: &Trial, compat: CppCompat) {
    let t = &trial.name;
    w.start_element("Trial");
    w.attribute("Name", t);
    w.attribute("Default", &format_bool(trial.is_default));
    w.attribute("FromDefault", &format_bool(trial.is_copy_from_default));
    w.attribute("startFrame", &qnu(trial.start_frame));
    w.attribute("endFrame", &qnu(trial.end_frame));
    // Ledger F68: XMALab's -1.
    w.attribute(
        "referenceCalibration",
        &if trial.reference_calibration_image == xma_model::trial::NO_REFERENCE_CALIBRATION {
            "-1".to_owned()
        } else {
            qnu(trial.reference_calibration_image)
        },
    );
    w.attribute(
        "recordingSpeed",
        &attr_number(compat, trial.recording_speed),
    );
    w.attribute(
        "cutOffFrequency",
        &attr_number(compat, trial.cutoff_frequency),
    );
    w.attribute("interpolate3D", &format_bool(trial.interpolate_3d));
    w.attribute("nbImages", &qnu(trial.n_frames));
    if !trial.portal_metadata.is_empty() {
        w.attribute("MetaData", &format!("{t}/metadata.xml"));
    }
    for e in &trial.events {
        w.start_element("Event");
        w.attribute("Name", &e.name);
        w.attribute("Color", &e.color.to_hex());
        w.attribute("Draw", &format_bool(e.draw));
        w.attribute("Filename", &format!("{t}/data/{}.csv", e.name));
        w.end_element();
    }
    for (k, m) in trial.markers.iter().enumerate() {
        w.start_element("Marker");
        w.attribute("Description", &m.description);
        w.attribute("ID", &qnu(k));
        w.attribute("TrackingPenalty", &qn(m.max_penalty));
        w.attribute("DetectionMethod", &qn(m.method.code()));
        w.attribute("SizeOverride", &qn(m.size_override));
        w.attribute("ThresholdOffset", &qn(m.threshold_offset));
        w.attribute(
            "RequiresRecomputation",
            &format_bool(m.requires_recomputation),
        );
        if m.reference_3d.is_some() {
            w.attribute("Reference3DPoint", &marker_file(t, k, "reference3Dpoint"));
        }
        w.attribute("FilenamePoints2D", &marker_file(t, k, "points2d"));
        w.attribute("FilenameStatus2D", &marker_file(t, k, "status2d"));
        w.attribute("FilenameSize", &marker_file(t, k, "size"));
        w.attribute("FilenamePoints3D", &marker_file(t, k, "points3d"));
        w.attribute("FilenameStatus3D", &marker_file(t, k, "status3d"));
        w.attribute("FilenameInterpolation", &marker_file(t, k, "interpolation"));
        w.end_element();
    }
    for (k, b) in trial.rigid_bodies.iter().enumerate() {
        w.start_element("RigidBody");
        w.attribute("Description", &b.description);
        w.attribute("ID", &qnu(k));
        if b.references_set == 2 {
            w.attribute("ReferenceNames", &body_file(t, k, "ReferenceNames"));
            w.attribute("ReferencePoints3D", &body_file(t, k, "ReferencePoints3d"));
        }
        if b.has_optimized_coordinates {
            w.attribute(
                "ReferencePoints3DOptimized",
                &body_file(t, k, "ReferencePoints3d_optimized"),
            );
        }
        w.attribute("Visible", &format_bool(b.visible));
        w.attribute("Color", &b.color.to_hex());
        w.attribute(
            "OverrideCutOffFrequency",
            &format_bool(b.override_cutoff_frequency),
        );
        w.attribute("CutOffFrequency", &attr_number(compat, b.cutoff_frequency));
        // Ported from core/RigidBody.cpp, RigidBody::hasMeshModel as `saveProject` tests it:
        // the C++ holds (and so writes) only a mesh whose file loaded (ledger F18).
        let unreadable = compat.drop_unreadable_mesh && b.mesh_readable == Some(false);
        if let Some(mesh) = b.mesh_model.as_ref().filter(|_| !unreadable) {
            w.attribute("Meshmodel", mesh);
            w.attribute("DrawMeshmodel", &format_bool(b.draw_mesh_model));
            w.attribute("MeshScale", &attr_number(compat, b.mesh_scale));
        }
        for (p, v) in b.virtual_points.iter().enumerate() {
            w.start_element("DummyMarker");
            w.attribute("Name", &v.name);
            w.attribute("PointReferences", &dummy_file(t, k, p, "PointReferences"));
            w.attribute("PointReferences2", &dummy_file(t, k, p, "PointReferences2"));
            w.attribute("PointCoordinates", &dummy_file(t, k, p, "PointCoordinates"));
            w.end_element();
        }
        w.end_element();
    }
    w.end_element();
}

// ---------------------------------------------------------------------------------------
// Data files

/// Lines each followed by `\n` (`QTextStream << line << '\n'`, `std::endl`).
fn text_lines<S: AsRef<str>>(lines: &[S]) -> Vec<u8> {
    let mut out = String::new();
    for l in lines {
        out.push_str(l.as_ref());
        out.push('\n');
    }
    out.into_bytes()
}

/// `x, y` per line (`CalibrationImage::savePoints`, `UndistortionObject::savePoints`).
fn points_comma_space(points: &[xma_model::Point2]) -> Vec<u8> {
    let mut out = String::new();
    for p in points {
        push_g17(&mut out, p.x);
        out.push_str(", ");
        push_g17(&mut out, p.y);
        out.push('\n');
    }
    out.into_bytes()
}

/// A 3 × 3 matrix, rows on lines, separator `", "`.
fn matrix_rows(m: &xma_model::Matrix3) -> Vec<u8> {
    let mut out = String::new();
    for i in 0..3 {
        out.push_str(&format!(
            "{}, {}, {}\n",
            format_g17(m[(i, 0)]),
            format_g17(m[(i, 1)]),
            format_g17(m[(i, 2)])
        ));
    }
    out.into_bytes()
}

/// `x,y,z` per line.
fn xyz_lines<'a>(points: impl IntoIterator<Item = &'a xma_model::Point3>) -> String {
    let mut out = String::new();
    for p in points {
        push_g17(&mut out, p.x);
        out.push(',');
        push_g17(&mut out, p.y);
        out.push(',');
        push_g17(&mut out, p.z);
        out.push('\n');
    }
    out
}

fn xyz(p: &xma_model::Point3) -> String {
    format!(
        "{},{},{}",
        format_g17(p.x),
        format_g17(p.y),
        format_g17(p.z)
    )
}

/// Ported from core/Camera.cpp, Camera::save, and core/CalibrationSequence.cpp,
/// CalibrationSequence::save (with the save functions of `CalibrationImage` and
/// `UndistortionObject` they call).
fn save_camera(s: &mut Saver<'_, '_>, project: &Project, cam: &Camera) -> Result<(), WriteError> {
    let dir = &cam.name;
    if cam.calibrated {
        // Camera::saveCameraMatrix
        s.write(
            &format!("{dir}/data/{dir}_CameraMatrix.csv"),
            matrix_rows(&cam.camera_matrix),
        );
        if cam.has_model_distortion {
            // Camera::saveUndistortionParam: 8 values, one per line.
            let mut out = String::new();
            for v in cam.distortion {
                out.push_str(&format_g17(v));
                out.push('\n');
            }
            s.write(
                &format!("{dir}/data/{dir}_UndistortionParameter.csv"),
                out.into_bytes(),
            );
        }
    }

    if let Some(grid) = &cam.undistortion {
        s.write_image(&format!("{dir}/{}", file_name(&grid.image_filename)))?;
        if grid.computed {
            let base = complete_base_name(&grid.image_filename);
            s.write(
                &format!("{dir}/data/{base}_PointsDetected.csv"),
                points_comma_space(&grid.points_detected),
            );
            s.write(
                &format!("{dir}/data/{base}_GridPointsDistorted.csv"),
                points_comma_space(&grid.grid_distorted),
            );
            s.write(
                &format!("{dir}/data/{base}_GridPointsReferences.csv"),
                points_comma_space(&grid.grid_references),
            );
            // UndistortionObject::saveGridPointsInlier
            let mut out = String::new();
            for &inlier in &grid.grid_inlier {
                out.push_str(if inlier { "1\n" } else { "0\n" });
            }
            s.write(
                &format!("{dir}/data/{base}_GridPointsInlier.csv"),
                out.into_bytes(),
            );
        }
    }

    // CalibrationSequence::save
    for (index, image) in cam.calibration_images.iter().enumerate() {
        if cam.calibration_sequence.is_none()
            && project.calibration_type == CalibrationType::Internal
        {
            s.write_image(&format!("{dir}/{}", calibration_image_name(cam, index)))?;
        }
        if image.calibrated > 0 {
            let base = calibration_image_base(cam, index);
            // CalibrationImage::savePointsInlier
            let mut inlier = String::new();
            for v in &image.inlier {
                inlier.push_str(&format_int(i64::from(*v)));
                inlier.push('\n');
            }
            s.write(
                &format!("{dir}/data/{base}_PointsInlier.csv"),
                inlier.into_bytes(),
            );
            s.write(
                &format!("{dir}/data/{base}_PointsDetected.csv"),
                points_comma_space(&image.detected),
            );
            s.write(
                &format!("{dir}/data/{base}_PointsDetectedAll.csv"),
                points_comma_space(&image.detected_all),
            );
            // CalibrationImage::saveRotationMatrix: cv::Rodrigues, vector to matrix.
            let rotation = xma_vision::camera::rodrigues_to_matrix(&image.rotation);
            s.write(
                &format!("{dir}/data/{base}_RotationMatrix.csv"),
                matrix_rows(&rotation),
            );
            // CalibrationImage::saveTranslationVector
            let t = &image.translation;
            s.write(
                &format!("{dir}/data/{base}_TranslationVector.csv"),
                format!(
                    "{}\n{}\n{}\n",
                    format_g17(t[0]),
                    format_g17(t[1]),
                    format_g17(t[2])
                )
                .into_bytes(),
            );
        }
    }
    Ok(())
}

/// The trial part of ui/ProjectFileIO.cpp, ProjectFileIO::saveProject.
fn save_trial(s: &mut Saver<'_, '_>, project: &Project, trial: &Trial) -> Result<(), WriteError> {
    let t = &trial.name;
    if !trial.requires_recomputation {
        s.write_extra(&format!("PrecisionInfo_{t}.txt"));
        s.write_extra(&format!("MarkerDistances_{t}.txt"));
    }
    s.folder.mkpath(t);
    if !trial.portal_metadata.is_empty() {
        // Trial::saveXMLData
        s.write(
            &format!("{t}/metadata.xml"),
            text_lines(&trial.portal_metadata),
        );
    }
    // Trial::save
    if !trial.is_default {
        for (i, video) in trial.videos.iter().enumerate() {
            let cam = &project.cameras[i].name;
            s.write(
                &format!("{t}/{cam}_filenames_absolute.txt"),
                text_lines(&video.filenames),
            );
        }
    }
    s.folder.mkpath(&format!("{t}/data"));
    // Trial::saveMarkers
    let descriptions: Vec<&str> = trial
        .markers
        .iter()
        .map(|m| m.description.as_str())
        .collect();
    s.write(
        &format!("{t}/data/MarkerDescription.txt"),
        text_lines(&descriptions),
    );
    // Trial::saveRigidBodies
    let mut bodies = String::new();
    for b in &trial.rigid_bodies {
        bodies.push_str(&b.description);
        bodies.push('[');
        let ids: Vec<String> = b
            .marker_indices
            .iter()
            .map(|&i| format_int(i as i64 + 1))
            .collect();
        bodies.push_str(&ids.join(","));
        bodies.push_str("]\n");
    }
    s.write(&format!("{t}/data/RigidBodies.txt"), bodies.into_bytes());
    // EventData::saveData: one bool per line.
    for e in &trial.events {
        s.write(
            &format!("{t}/data/{}.csv", e.name),
            crate::export::events::event_file(e).into_bytes(),
        );
    }
    for (k, m) in trial.markers.iter().enumerate() {
        save_marker(s, t, k, m);
    }
    for (k, b) in trial.rigid_bodies.iter().enumerate() {
        save_rigid_body(s, t, k, b);
    }
    Ok(())
}

/// Ported from core/Marker.cpp, Marker::save, Marker::save3DPoints,
/// Marker::saveInterpolation and Marker::saveReference3DPoint.
fn save_marker(s: &mut Saver<'_, '_>, t: &str, k: usize, m: &Marker) {
    // `Marker::save` loops over the frames of camera 0; with no camera the C++ reads out
    // of bounds. Here the three files are then empty.
    let n_frames = m.views.first().map_or(0, |v| v.points.len());
    let mut points = String::new();
    let mut status = String::new();
    let mut size = String::new();
    for j in 0..n_frames {
        for (i, v) in m.views.iter().enumerate() {
            if i > 0 {
                points.push(',');
                status.push(',');
                size.push(',');
            }
            let p = v.points.get(j).copied().unwrap_or_default();
            push_g17(&mut points, p.x);
            points.push(',');
            push_g17(&mut points, p.y);
            let st = v.status.get(j).copied().unwrap_or_default();
            // `format_int`, appended.
            let _ = write!(status, "{}", st.code());
            push_g17(&mut size, v.size.get(j).copied().unwrap_or_default());
        }
        points.push('\n');
        status.push('\n');
        size.push('\n');
    }
    s.write(&marker_file(t, k, "points2d"), points.into_bytes());
    s.write(&marker_file(t, k, "status2d"), status.into_bytes());
    s.write(&marker_file(t, k, "size"), size.into_bytes());

    s.write(
        &marker_file(t, k, "points3d"),
        xyz_lines(&m.points3d).into_bytes(),
    );
    let mut status3d = String::new();
    for st in &m.status3d {
        let _ = write!(status3d, "{}", st.code());
        status3d.push('\n');
    }
    s.write(&marker_file(t, k, "status3d"), status3d.into_bytes());

    let mut interpolation = String::new();
    for i in &m.interpolation {
        let _ = write!(interpolation, "{}", i.code());
        interpolation.push('\n');
    }
    s.write(
        &marker_file(t, k, "interpolation"),
        interpolation.into_bytes(),
    );

    if let Some(p) = &m.reference_3d {
        s.write(
            &marker_file(t, k, "reference3Dpoint"),
            format!("{}\n", xyz(p)).into_bytes(),
        );
    }
}

/// Ported from core/RigidBody.cpp, RigidBody::save, RigidBody::saveOptimized (without
/// header, as `saveProject` calls it) and RigidBody::saveDummy.
fn save_rigid_body(s: &mut Saver<'_, '_>, t: &str, k: usize, b: &RigidBody) {
    if b.references_set == 2 {
        s.write(
            &body_file(t, k, "ReferenceNames"),
            text_lines(&b.reference_names),
        );
        // The original points when optimised, else the working set; as many as the working
        // set has (checked in `check_project`).
        let points = if b.has_optimized_coordinates {
            &b.reference_points_original[..b.reference_points.len()]
        } else {
            &b.reference_points[..]
        };
        s.write(
            &body_file(t, k, "ReferencePoints3d"),
            xyz_lines(points).into_bytes(),
        );
    }
    if b.has_optimized_coordinates {
        s.write(
            &body_file(t, k, "ReferencePoints3d_optimized"),
            xyz_lines(&b.reference_points).into_bytes(),
        );
    }
    for (p, v) in b.virtual_points.iter().enumerate() {
        let index: i64 = match &v.source {
            VirtualPointSource::OtherBody { body, .. } => *body as i64,
            VirtualPointSource::Coordinates(_) => -1,
        };
        s.write(
            &dummy_file(t, k, p, "PointReferences"),
            format!("x,y,z\n{},{}\n", xyz(&v.reference), format_int(index)).into_bytes(),
        );
        match &v.source {
            VirtualPointSource::Coordinates(coords) => {
                let mut out = String::from("x,y,z\n");
                for c in coords {
                    match c {
                        Some(p) => out.push_str(&xyz(p)),
                        None => out.push_str("NaN,NaN,NaN"),
                    }
                    out.push('\n');
                }
                s.write(&dummy_file(t, k, p, "PointCoordinates"), out.into_bytes());
            }
            VirtualPointSource::OtherBody { point, .. } => {
                s.write(
                    &dummy_file(t, k, p, "PointReferences2"),
                    format!("x,y,z\n{}\n", xyz(point)).into_bytes(),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qt_order_folds_case_and_compares_utf16() {
        let mut names = vec![
            "b", "B", "_x", "a.csv", "A_b", "\u{c4}", "\u{e4}", "z", "Z1", "\u{e9}", "\u{c9}", "e",
            "f", "[", "a b", "a-b", "a~", "\u{3a9}", "\u{3c9}",
        ];
        names.sort_by(|a, b| qt_name_order(a, b));
        // Qt 6.11 lists the same names (in its own order among names equal apart from case)
        // as: [ _x a b a-b a.csv A_b a~ b B e f z Z1 ä Ä é É ω Ω
        let folded: Vec<String> = names.iter().map(|n| n.to_lowercase()).collect();
        assert_eq!(
            folded,
            [
                "[", "_x", "a b", "a-b", "a.csv", "a_b", "a~", "b", "b", "e", "f", "z", "z1",
                "\u{e4}", "\u{e4}", "\u{e9}", "\u{e9}", "\u{3c9}", "\u{3c9}"
            ]
        );
        assert_eq!(
            qt_name_order("PrecisionInfo_T.txt", "project.xml"),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            qt_name_order("log.html", "MarkerDistances_T.txt"),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn folder_behaves_like_the_file_system() {
        let mut f = Folder::default();
        f.mkpath("T/data");
        f.write("T/data/a.csv", b"1".to_vec());
        f.write("T/data/a.csv", b"2".to_vec());
        f.write("T/data/x/b.csv", b"3".to_vec()); // no such folder
        f.write("T//data/./c.csv", b"4".to_vec());
        f.write("../outside", b"5".to_vec());
        f.write("T/.hidden.csv", b"6".to_vec());
        f.write("T/data", b"7".to_vec()); // a folder
        let (files, dropped) = f.archive_order();
        assert_eq!(
            files,
            vec![
                ("T/data/a.csv".to_owned(), b"2".to_vec()),
                ("T/data/c.csv".to_owned(), b"4".to_vec())
            ]
        );
        assert_eq!(dropped.len(), 4, "{dropped:?}");
    }

    #[test]
    fn names_of_files() {
        assert_eq!(complete_base_name("Camera 1/cal.01.tif"), "cal.01");
        assert_eq!(complete_base_name("Frame000003"), "Frame000003");
        assert_eq!(file_name("a/b/c.tif"), "c.tif");
    }
}
