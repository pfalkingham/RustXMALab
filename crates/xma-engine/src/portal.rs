//! XMA Portal import: a trial from a `.xmatrial` file (the New trial dialog's "Load XMA
//! Portal file", C60), a new dataset from a Portal project archive (C63), and the study
//! metadata a project or trial keeps ("Add metadata from XMAPortal file").
//!
//! The archive format is [`xma_io::portal`]. Ported from ui/ProjectFileIO.cpp
//! (`loadXMAPortalTrial`, `loadXMALabProject`, `addMetaData`), ui/NewTrialDialog.cpp
//! (`createTrial`'s `setXMLData` and `setFrameRateFromXML`), ui/NewProjectDialog.cpp
//! (`addCalibrationImage`, `addGridImage`, `setupBasedOnMissingParameters`),
//! ui/CameraBoxTrial.cpp (`setFilename`), core/Project.cpp (`addCamera`, `setXMLData`) and
//! core/Trial.cpp (`setXMLData`, `setFrameRateFromXML`).
//!
//! The work that reads and writes files ([`import_trial`], [`import_project`]) is plain
//! and may run on any thread; it changes no document. The changes to a document are
//! edits ([`crate::Edit::NewTrialFromPortal`], [`crate::Edit::ChangeTrialDataFromPortal`],
//! [`crate::Edit::SetPortalMetadata`]), each one undo step, and
//! [`Document::new_project_from_portal`] for a new dataset.
//!
//! # Fixed C++ defects
//!
//! Ledger F53 to F60: an archive without a folder crashes XMALab (F53); a Portal trial with
//! more cameras than the dialog has boxes reads and writes past the boxes (F54); entry
//! names and study or trial names that climb out of their folder are written there (F55,
//! both modes refused); malformed metadata hangs XMALab (F56); a `File` without a
//! `Filename` gives the top folder as a calibration image (F57); a nested zip is extracted
//! to a folder found by replacing every `.zip` in the full path (F58); the cameras' portal
//! ids come from every camera number in the metadata (F59, compat); metadata added again
//! is appended (F60, compat). The interface's defects (the file deleted after a failed
//! import, the Portal file taken as the new dataset's file, a calibration video left in a
//! removed temporary folder) are ledger U151 to U153.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use xma_io::portal::{
    CalibrationFileKind, Metadata, cube_files, extract_nested, nested_zip_folder, safe_relative,
};

/// The archive format, for the crates that use the engine only.
pub use xma_io::portal::{Destination, NOT_VALID, PortalArchive, PortalError, fixture};

use crate::create::{CalibrationTarget, CreateError, NewCameraSpec, NewProjectSpec, NewTrialSpec};
use crate::document::{Document, OpenOptions};
use crate::edit::{EditError, EditOutcome, EditResult};
use crate::undistort_sequence::{SortName, names_less};

/// The question before "Load XMA Portal file" (`NewTrialDialog::on_pushButton_LoadXMA_clicked`).
pub const LOAD_CONFIRM: &str = "The data will be extracted to the folder where your xmatrial-file is located in case you did not setup a workspace. Also please make sure that you added enough cameras to the trial before importing. Do you want to proceed?";
/// The question after the Portal file is chosen.
pub const DELETE_QUESTION: &str = "Do you want to delete the .xmatrial file after the import? Click Yes if you want to delete it. No if you do not want to delete it.";
/// The first question of "Add metadata from XMAPortal file" (`MetaDataInfo::on_pushButtonAdd_clicked`).
pub const METADATA_CONFIRM: &str = "Are you sure you want to manually add metadata from the portal? Please make sure that it uses the same number of cameras, that the cameras are in the right order and that the data used is the same as in the xml-file.";
/// Its item dialog's title.
pub const METADATA_ITEM_TITLE: &str = "Select the trial you want to add xml data";
/// Its item dialog's label (with XMALab's trailing space).
pub const METADATA_ITEM_LABEL: &str = "Trial ";
/// The item standing for the project's own metadata.
pub const CALIBRATION_ITEM: &str = "Calibration";

/// The question before replacing study data ("Add metadata from XMAPortal file").
#[must_use]
pub fn override_text(item: &str) -> String {
    format!(
        "There is already data associated with the trial {item}. Are you sure you want to override it?"
    )
}

/// The items of "Select the trial you want to add xml data": "Calibration" when the
/// project has a calibration, then every trial's name (`MetaDataInfo::on_pushButtonAdd_clicked`).
#[must_use]
pub fn metadata_items(project: &xma_model::Project) -> Vec<String> {
    let mut items = Vec::new();
    if project.calibration_type != xma_model::CalibrationType::None {
        items.push(CALIBRATION_ITEM.to_owned());
    }
    items.extend(project.trials.iter().map(|t| t.name.clone()));
    items
}

/// What item `index` of [`metadata_items`] stands for: `Some(None)` the project's own
/// metadata, `Some(Some(t))` trial `t`; `None` out of range. Ledger U155: XMALab 3 takes the
/// item by its text, so a trial named "Calibration" can never be chosen (its item gives the
/// project's metadata), and a text matching no trial would leave `MetaDataInfo.cpp:166`'s
/// `Trial*` uninitialised (no list item can do that); here the item is taken by its place.
#[must_use]
pub fn metadata_target(project: &xma_model::Project, index: usize) -> Option<Option<usize>> {
    let calibrated = project.calibration_type != xma_model::CalibrationType::None;
    if calibrated && index == 0 {
        return Some(None);
    }
    let t = index - usize::from(calibrated);
    (t < project.trials.len()).then_some(Some(t))
}

/// XMALab's error when a trial's frame rate could not be taken from its metadata
/// (`NewTrialDialog::createTrial`).
pub const FRAME_RATE_NOT_SET: &str = "Framerate could not be set. Either the framerates of the files differ or Framerates have not been defined";

/// The image types `CameraBoxTrial::setFilename` lists in a nested zip's folder.
const SEQUENCE_TYPES: [&str; 5] = ["png", "tif", "bmp", "jpeg", "jpg"];

/// The files a camera box gets for one Portal file (`CameraBoxTrial::setFilename`): a
/// name ending in `.zip` (any case) is an image sequence, the images (`*.png`, `*.tif`,
/// `*.bmp`, `*.jpeg`, `*.jpg`, any case, files only) of the folder of the same name without
/// `.zip`, in natural order (`littleHelper::compareNames`); anything else is that one file.
///
/// Ported from ui/CameraBoxTrial.cpp, CameraBoxTrial::setFilename (`:55-77`).
#[must_use]
pub fn camera_files(path: &Path) -> Vec<PathBuf> {
    let Some(folder) = nested_zip_folder(path) else {
        return vec![path.to_path_buf()];
    };
    let dir = std::path::absolute(&folder).unwrap_or(folder);
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map(|rd| {
            rd.filter_map(Result::ok)
                .filter(|e| e.file_type().is_ok_and(|t| t.is_file() && !t.is_symlink()))
                .filter(|e| {
                    let n = e.file_name().to_string_lossy().to_lowercase();
                    SEQUENCE_TYPES.iter().any(|x| n.ends_with(&format!(".{x}")))
                })
                .map(|e| e.path())
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    natural_order(files)
}

/// The files in the order of a stable insertion by `compareNames`: each file, in the order
/// given, goes before the first file already placed that it comes before. `compareNames`
/// does not order every set of names consistently, so this rule (not another sort) decides
/// the order. Each name is prepared once ([`SortName`]), and the start every name shares is
/// not compared again.
fn natural_order(files: Vec<PathBuf>) -> Vec<PathBuf> {
    let texts: Vec<String> = files
        .iter()
        .map(|f| f.to_string_lossy().into_owned())
        .collect();
    let names: Vec<SortName<'_>> = texts.iter().map(|t| SortName::new(t)).collect();
    let common = names.first().map_or(0, |first| {
        names
            .iter()
            .map(|n| first.common_prefix(n))
            .min()
            .unwrap_or(0)
    });
    let mut order: Vec<usize> = Vec::with_capacity(names.len());
    for (k, s) in names.iter().enumerate() {
        let at = order
            .iter()
            .position(|&g| names_less(s, &names[g], common))
            .unwrap_or(order.len());
        order.insert(at, k);
    }
    let mut files: Vec<Option<PathBuf>> = files.into_iter().map(Some).collect();
    order.into_iter().filter_map(|k| files[k].take()).collect()
}

/// The line edit's text `CameraBoxTrial::setFilename` shows for a zip: the folder of its
/// images (the path without `.zip`); `None` for another file.
#[must_use]
pub fn nested_zip_text(path: &Path) -> Option<String> {
    nested_zip_folder(path).map(|f| f.to_string_lossy().into_owned())
}

/// One file of a Portal trial, extracted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportedFile {
    /// `cameraNumber` (`camN`: N).
    pub number: i32,
    /// The camera index it goes to: the place of its number among the trial's numbers.
    pub index: usize,
    /// `folder/Filename`, as `setCam` is given it.
    pub path: PathBuf,
    /// What the camera box takes ([`camera_files`]).
    pub files: Vec<PathBuf>,
}

/// A Portal trial extracted (what `loadXMAPortalTrial` does before it fills the dialog).
#[derive(Clone, Debug, PartialEq)]
pub struct TrialImport {
    /// Where it was extracted.
    pub folder: PathBuf,
    /// `TrialName`.
    pub trial_name: String,
    /// `StudyName`.
    pub study_name: String,
    /// The trial's files in metadata order.
    pub files: Vec<ImportedFile>,
    /// The metadata file the dialog keeps (`setXmlMetadata`): the extracted copy.
    pub metadata_file: PathBuf,
    /// Its lines, read when the import ends.
    pub metadata: Vec<String>,
    /// For the Console: files that were there already and were kept.
    pub messages: Vec<String>,
}

/// Imports a Portal trial: opens the archive, extracts its top folder into the folder
/// `dest` gives (created), extracts each of its files that is a zip (an image sequence)
/// into the folder of the same name and removes the zip, and reads the metadata.
/// `progress` gets the fraction done; returning false stops.
///
/// Ported from ui/ProjectFileIO.cpp, ProjectFileIO::loadXMAPortalTrial (`:870-1014`).
///
/// # Errors
/// [`PortalError::NotPortal`] ("Not a valid xma file from the portal") for an archive that
/// is not a Portal archive, an empty one included (ledger F53: XMALab 3 reads the first
/// folder of none); the other [`PortalError`]s for an archive that cannot be read or
/// extracted.
pub fn import_trial(
    path: &Path,
    dest: &Destination<'_>,
    compat: xma_model::CppCompat,
    progress: &mut dyn FnMut(f32) -> bool,
) -> Result<TrialImport, PortalError> {
    let mut archive = PortalArchive::open_with(path, compat.portal_macos_folder_taken)?;
    let meta = archive.metadata().clone();
    // Everything that can refuse the import is checked before anything is written (ledger
    // F55): the folder's names and every file name.
    let folder = dest.folder(&meta.study_name, &trial_folder_name(&meta, archive.top())?)?;
    let trial_files = meta.trial_files();
    let rels: Vec<(i32, PathBuf)> = trial_files
        .iter()
        .map(|(c, n)| safe_relative(n).map(|r| (*c, r)))
        .collect::<Result<_, _>>()?;
    let x = archive.extract(&folder, &mut |done, total| {
        progress(if total == 0 {
            1.0
        } else {
            done as f32 / total as f32
        })
    })?;
    let mut written = x.written.clone();
    let mut created = x.created_dirs.clone();
    let fail = |written: &[PathBuf], created: &[PathBuf], e: PortalError| {
        xma_io::portal::undo(written, created);
        Err(e)
    };
    let mut messages: Vec<String> = x
        .kept
        .iter()
        .map(|k| format!("{} exists already and was kept", k.display()))
        .collect();
    let numbers: std::collections::BTreeSet<i32> = rels.iter().map(|(c, _)| *c).collect();
    let mut files = Vec::new();
    for (number, rel) in &rels {
        let file = folder.join(rel);
        // Ledger F79: only a zip this import wrote is unpacked and removed.
        if let Some(seq) = nested_zip_folder(&file)
            && written.contains(&file)
        {
            match extract_nested(&file, &seq) {
                Ok(n) => {
                    written.retain(|w| *w != file);
                    written.extend(n.written);
                    created.extend(n.created_dirs);
                }
                Err(e) => return fail(&written, &created, e),
            }
        }
        let index = numbers.iter().position(|n| n == number).unwrap_or(0);
        files.push(ImportedFile {
            number: *number,
            index,
            files: camera_files(&file),
            path: file,
        });
    }
    let metadata_file = folder.join(xma_io::portal::METADATA_FILE);
    let metadata = match std::fs::read(&metadata_file) {
        Ok(d) => match Metadata::parse(&d) {
            Ok(m) => m.lines,
            Err(e) => return fail(&written, &created, e),
        },
        Err(e) => {
            messages.push(format!("{}: {e}", metadata_file.display()));
            meta.lines.clone()
        }
    };
    Ok(TrialImport {
        folder,
        trial_name: meta.trial_name,
        study_name: meta.study_name,
        files,
        metadata_file,
        metadata,
        messages,
    })
}

/// The name of the folder a trial is extracted to: `TrialName`, or, when it is empty, white
/// space or `.` (which would make the folder the Portal file's own, the project file's or
/// the study's), the archive's top folder (ledger F79). XMALab 3 extracts into that folder
/// itself.
///
/// # Errors
/// A name that climbs out or is not allowed (ledger F55).
pub fn trial_folder_name(meta: &Metadata, top: &str) -> Result<String, PortalError> {
    let rel = safe_relative(&meta.trial_name)?;
    Ok(
        if rel.as_os_str().is_empty() || meta.trial_name.trim().is_empty() {
            top.to_owned()
        } else {
            meta.trial_name.clone()
        },
    )
}

/// Which camera box each file of a Portal trial goes to.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TrialAssignment {
    /// The trial name to show (`setTrialName`, only when the trial has a file).
    pub trial_name: Option<String>,
    /// `(box, files, text)` in metadata order; a later file for the same box replaces an
    /// earlier one, as `setCam` does.
    pub cameras: Vec<(usize, Vec<PathBuf>, PathBuf)>,
    /// Boxes the dialog must have (more than it had when it can grow).
    pub boxes: usize,
    /// Files no box was found for (Console lines).
    pub skipped: Vec<String>,
}

/// The dialog's filling (`loadXMAPortalTrial` `:986-1006`): the file of camera number `n`
/// goes to the box of its index among the trial's numbers when the project has no cameras,
/// or that camera has no portal id, or has this one; otherwise it is not set.
///
/// `portal_ids` are the project's cameras' portal ids (empty when it has none). `boxes`
/// is the number of camera boxes; `can_grow` is true for "New trial without calibration",
/// whose number of cameras the user sets. Ledger F54: XMALab 3 reads and writes past the
/// boxes and the cameras for a trial with more cameras than the dialog has boxes; here the
/// dialog without calibration gets the boxes it needs, and the dialog of a project with
/// cameras leaves the file out with a Console line.
#[must_use]
pub fn assign_trial(
    import: &TrialImport,
    portal_ids: &[i32],
    boxes: usize,
    can_grow: bool,
) -> TrialAssignment {
    let mut a = TrialAssignment {
        boxes,
        ..TrialAssignment::default()
    };
    for f in &import.files {
        a.trial_name = Some(import.trial_name.clone());
        let id = f.index;
        let fits = id < boxes || can_grow;
        let in_project = portal_ids.is_empty() || id < portal_ids.len();
        if !fits || !in_project {
            a.skipped.push(format!(
                "{}: camera cam{} of the Portal trial has no camera in the dataset; the file was not used",
                f.path.display(),
                f.number
            ));
            continue;
        }
        let matches = portal_ids.is_empty() || portal_ids[id] == -1 || portal_ids[id] == f.number;
        if !matches {
            continue;
        }
        a.boxes = a.boxes.max(id + 1);
        a.cameras.retain(|(b, _, _)| *b != id);
        a.cameras.push((id, f.files.clone(), f.path.clone()));
    }
    a
}

// ---------------------------------------------------------------------------------------
// A new dataset from a Portal project archive

/// A folder removed when the last reference to it goes (the temporary folder a Portal
/// project is extracted to; `MainWindow::newProjectFinished`'s `removeTmpDir`).
#[derive(Debug, PartialEq, Eq)]
pub struct TemporaryFolder(pub PathBuf);

impl Drop for TemporaryFolder {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// One camera of a Portal project, as the New dataset dialog is filled with it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProjectCamera {
    /// `camN`: N.
    pub number: i32,
    /// `addCalibrationImage`'s files, in the order added.
    pub images: Vec<PathBuf>,
    /// `addGridImage`: the last grid given.
    pub grid: Option<PathBuf>,
}

/// A Portal project extracted (`loadXMALabProject`).
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectImport {
    /// The archive.
    pub source: PathBuf,
    /// Where it was extracted (its top folder's contents).
    pub folder: PathBuf,
    /// The archive's top folder's name.
    pub top: String,
    /// The cameras in the order of their numbers; index i is the dialog's box i.
    pub cameras: Vec<ProjectCamera>,
    /// The one `.csv` of the top folder (`setCalibrationCubeCSV`).
    pub frame_specification: Option<PathBuf>,
    /// The one `.ref` of the top folder (`setCalibrationCubeREF`).
    pub references: Option<PathBuf>,
    /// The metadata, which the project keeps (`loadProjectMetaData`).
    pub metadata: Metadata,
    /// The temporary folder, removed when the import is dropped; `None` when the files
    /// were extracted to stay (a calibration video, ledger U153).
    pub temporary: Option<Arc<TemporaryFolder>>,
}

fn is_video(name: &str) -> bool {
    let s = name.to_ascii_lowercase();
    [".cine", ".avi", ".mp4", ".mov"]
        .iter()
        .any(|e| s.ends_with(e))
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A new temporary folder name (`QDir::tempPath()/XROMM_tmp` in XMALab, one for the whole
/// program; here one per import).
fn temporary_folder() -> PathBuf {
    let n = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("XROMM_portal_{}_{n}", std::process::id()))
}

/// Imports a Portal project archive: opens it, extracts its top folder into a temporary
/// folder (XMALab's `XROMM_tmp`), and collects what fills the New dataset dialog.
///
/// Ledger U153: when a calibration file is a video, which a dataset refers to by its path
/// and does not copy, the files are extracted where a trial of the archive would be
/// (`dest`, with the archive's `TrialName`) and kept; XMALab 3 removes its temporary
/// folder after creating the dataset, which then refers to a file that is gone.
///
/// Ported from ui/ProjectFileIO.cpp, ProjectFileIO::loadProject (`:286-302`) and
/// ProjectFileIO::loadXMALabProject (`:1016-1142`).
///
/// # Errors
/// As [`import_trial`].
pub fn import_project(
    path: &Path,
    dest: &Destination<'_>,
    compat: xma_model::CppCompat,
    progress: &mut dyn FnMut(f32) -> bool,
) -> Result<ProjectImport, PortalError> {
    let mut archive = PortalArchive::open_with(path, compat.portal_macos_folder_taken)?;
    let meta = archive.metadata().clone();
    let top = archive.top().to_owned();
    let files = meta.project_files();
    let keep = files.iter().any(|f| is_video(&f.filename));
    // Checked before anything is written (ledger F55).
    let rels: Vec<PathBuf> = files
        .iter()
        .map(|f| safe_relative(&f.filename))
        .collect::<Result<_, _>>()?;
    let (folder, temporary) = if keep {
        (
            dest.folder(&meta.study_name, &trial_folder_name(&meta, &top)?)?,
            None,
        )
    } else {
        let t = temporary_folder();
        (t.clone(), Some(Arc::new(TemporaryFolder(t))))
    };
    archive.extract(&folder, &mut |done, total| {
        progress(if total == 0 {
            1.0
        } else {
            done as f32 / total as f32
        })
    })?;
    let numbers: std::collections::BTreeSet<i32> = files.iter().map(|f| f.camera_number).collect();
    let mut cameras: Vec<ProjectCamera> = numbers
        .iter()
        .map(|&n| ProjectCamera {
            number: n,
            ..ProjectCamera::default()
        })
        .collect();
    for (f, rel) in files.iter().zip(&rels) {
        let id = numbers
            .iter()
            .position(|&n| n == f.camera_number)
            .unwrap_or(0);
        let p = folder.join(rel);
        match f.kind {
            CalibrationFileKind::Image => cameras[id].images.push(p),
            CalibrationFileKind::Grid => cameras[id].grid = Some(p),
        }
    }
    let (frame_specification, references) = cube_files(&folder);
    Ok(ProjectImport {
        source: path.to_path_buf(),
        folder,
        top,
        cameras,
        frame_specification,
        references,
        metadata: meta,
        temporary,
    })
}

impl ProjectImport {
    /// The cameras' portal ids: in the default mode each camera's number; in the XMALab 3
    /// mode the camera numbers of the whole metadata in increasing order (ledger F59).
    #[must_use]
    pub fn portal_ids(&self, compat: xma_model::CppCompat) -> Vec<i32> {
        if compat.portal_camera_ids_from_all_files {
            self.metadata.all_camera_numbers.clone()
        } else {
            self.cameras.iter().map(|c| c.number).collect()
        }
    }

    /// The New dataset dialog as `loadXMALabProject` and `setupBasedOnMissingParameters`
    /// leave it, for `xma`: one camera per Portal camera (the dialog starts with two boxes,
    /// which a one-camera archive leaves with an empty second box), a camera without a grid
    /// a light camera, the calibration images sorted (`QStringList::sort`), Cube when a
    /// `.csv` or a `.ref` was found, else Checkerboard with the dialog's zeros.
    #[must_use]
    pub fn spec(&self) -> NewProjectSpec {
        let cameras = self
            .cameras
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let mut images = c.images.clone();
                images.sort_by(|a, b| {
                    a.to_string_lossy()
                        .encode_utf16()
                        .cmp(b.to_string_lossy().encode_utf16())
                });
                NewCameraSpec {
                    name: format!("Camera {}", i + 1),
                    images,
                    light_camera: c.grid.is_none(),
                    grid: c.grid.clone(),
                    flipped: false,
                }
            })
            .collect();
        let target = if self.frame_specification.is_none() && self.references.is_none() {
            CalibrationTarget::Checkerboard {
                horizontal_squares: 0,
                vertical_squares: 0,
                square_size: 0.0,
            }
        } else {
            CalibrationTarget::Cube {
                frame_specification: self.frame_specification.clone().unwrap_or_default(),
                references: self.references.clone().unwrap_or_default(),
            }
        };
        NewProjectSpec { cameras, target }
    }
}

impl Document {
    /// File > New dataset from a Portal project, after the dialog's OK: as
    /// [`Document::new_project`], then each camera named "Camera <portal id>" with that id
    /// (`Project::addCamera`, `Camera::setPortalId`) and the metadata kept as the
    /// project's study data. The document has no file (ledger U152: XMALab 3 takes the
    /// Portal file as the dataset's file, so Save overwrites the download).
    ///
    /// # Errors
    /// As [`Document::new_project`].
    pub fn new_project_from_portal(
        spec: &NewProjectSpec,
        import: &ProjectImport,
        options: &OpenOptions,
    ) -> Result<Document, CreateError> {
        let ids = import.portal_ids(options.compat);
        let mut spec = spec.clone();
        for (cam, id) in spec.cameras.iter_mut().zip(&ids) {
            cam.name = format!("Camera {id}");
        }
        let mut doc = Document::new_project(&spec, options)?;
        for (cam, id) in doc.project.cameras.iter_mut().zip(&ids) {
            Arc::make_mut(cam).portal_id = *id;
        }
        doc.project.portal_metadata = import.metadata.lines.clone();
        Ok(doc)
    }
}

impl Document {
    /// File > New trial without calibration, with the files of a Portal trial: as
    /// [`Document::new_without_calibration`], then the trial's metadata and the frame rate
    /// from it (`NewTrialDialog::createTrial`). Returns the document and XMALab's error
    /// when the frame rate could not be set (the dataset is made all the same).
    ///
    /// # Errors
    /// As [`Document::new_without_calibration`].
    pub fn new_without_calibration_from_portal(
        spec: &crate::create::NoCalibrationSpec,
        metadata: &[String],
        options: &OpenOptions,
    ) -> Result<(Document, Option<String>), CreateError> {
        let mut doc = Document::new_without_calibration(spec, options)?;
        let outcome = trial_metadata(&mut doc, 0, metadata);
        Ok((doc, outcome.refused))
    }
}

// ---------------------------------------------------------------------------------------
// The edits

/// Metadata set on a project or a trial: replaced, or appended in the XMALab 3 mode
/// (ledger F60).
fn set_lines(target: &mut Vec<String>, lines: &[String], compat: xma_model::CppCompat) {
    if compat.portal_metadata_appended {
        target.extend(lines.iter().cloned());
    } else {
        *target = lines.to_vec();
    }
}

/// `Trial::setXMLData`, then `Trial::setFrameRateFromXML`: the recording speed from the
/// metadata, or XMALab's error (the trial keeps its speed).
fn trial_metadata(doc: &mut Document, trial: usize, lines: &[String]) -> EditOutcome {
    let compat = doc.compat();
    let t = Arc::make_mut(&mut doc.project.trials[trial]);
    set_lines(&mut t.portal_metadata, lines, compat);
    if t.portal_metadata.is_empty() {
        return EditOutcome::none();
    }
    let rate = xma_io::portal::parse_lines(&t.portal_metadata)
        .ok()
        .and_then(|m| m.frame_rate(compat.nonfinite_imports_accepted));
    match rate {
        Some(r) => {
            t.recording_speed = r;
            EditOutcome::none()
        }
        None => EditOutcome::refused(FRAME_RATE_NOT_SET),
    }
}

/// [`crate::Edit::NewTrialFromPortal`]: [`crate::Edit::NewTrial`], then the metadata and
/// the frame rate from it.
///
/// Ported from ui/NewTrialDialog.cpp, NewTrialDialog::createTrial (`:163-193`).
pub(crate) fn new_trial(doc: &mut Document, spec: &NewTrialSpec, lines: &[String]) -> EditResult {
    crate::create::new_trial(doc, spec)?;
    let t = doc.project.trials.len() - 1;
    Ok(trial_metadata(doc, t, lines))
}

/// [`crate::Edit::ChangeTrialDataFromPortal`]: [`crate::Edit::ChangeTrialData`], then the
/// metadata and the frame rate from it.
///
/// Ported from ui/NewTrialDialog.cpp, NewTrialDialog::createTrial (`:131-151`).
pub(crate) fn change_trial_data(
    doc: &mut Document,
    trial: usize,
    spec: &NewTrialSpec,
    lines: &[String],
) -> EditResult {
    let outcome = crate::create::change_trial_data(doc, trial, spec)?;
    if outcome.refused.is_some() {
        return Ok(outcome);
    }
    Ok(trial_metadata(doc, trial, lines))
}

/// [`crate::Edit::SetPortalMetadata`]: "Add metadata from XMAPortal file" (`addMetaData`:
/// `Project::setXMLData` or `Trial::setXMLData`). The project's cameras get their portal ids
/// from it at once (ledger F64; XMALab 3 mode: when the project is opened again).
pub(crate) fn set_metadata(
    doc: &mut Document,
    trial: Option<usize>,
    lines: &[String],
) -> EditResult {
    let compat = doc.compat();
    match trial {
        None => {
            set_lines(&mut doc.project.portal_metadata, lines, compat);
            // Ledger F64: the cameras get their portal ids at once (XMALab 3: when the
            // project is opened again; the name "Camera <id>" follows then, here too).
            if !compat.portal_ids_left_unset
                && let Ok(m) = xma_io::portal::parse_lines(&doc.project.portal_metadata)
            {
                let mut ids: Vec<i32> = m.project_files().iter().map(|f| f.camera_number).collect();
                ids.sort_unstable();
                ids.dedup();
                if ids.is_empty() {
                    ids = m.all_camera_numbers;
                }
                for (cam, id) in doc.project.cameras.iter_mut().zip(ids) {
                    Arc::make_mut(cam).portal_id = id;
                }
            }
        }
        Some(t) => {
            let Some(tr) = doc.project.trials.get_mut(t) else {
                return Err(EditError::Invalid(format!("no trial {t}")));
            };
            set_lines(&mut Arc::make_mut(tr).portal_metadata, lines, compat);
        }
    }
    Ok(EditOutcome::none())
}

/// The metadata of a Portal file, for "Add metadata from XMAPortal file" (`addMetaData`):
/// `None` when the file is not a Portal archive, as XMALab then adds nothing.
///
/// # Errors
/// An archive that cannot be read, or whose metadata is malformed.
pub fn read_metadata(
    path: &Path,
    compat: xma_model::CppCompat,
) -> Result<Option<Vec<String>>, PortalError> {
    match PortalArchive::open_with(path, compat.portal_macos_folder_taken) {
        Ok(a) => Ok(Some(a.metadata().lines.clone())),
        Err(PortalError::NotPortal) => Ok(None),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn import(files: &[(i32, usize)]) -> TrialImport {
        TrialImport {
            folder: PathBuf::from("/t"),
            trial_name: "T".into(),
            study_name: "S".into(),
            files: files
                .iter()
                .map(|&(number, index)| ImportedFile {
                    number,
                    index,
                    path: PathBuf::from(format!("/t/c{number}.avi")),
                    files: vec![PathBuf::from(format!("/t/c{number}.avi"))],
                })
                .collect(),
            metadata_file: PathBuf::from("/t/m.xml"),
            metadata: Vec::new(),
            messages: Vec::new(),
        }
    }

    #[test]
    fn u155_metadata_item_by_place() {
        let mut p = xma_model::Project::new();
        p.calibration_type = xma_model::CalibrationType::Internal;
        let mut t = xma_model::Trial::default_trial();
        t.name = "Calibration".into();
        t.is_default = false;
        p.trials.push(Arc::new(t));
        assert_eq!(metadata_items(&p), vec!["Calibration", "Calibration"]);
        assert_eq!(metadata_target(&p, 0), Some(None));
        assert_eq!(metadata_target(&p, 1), Some(Some(0)));
        assert_eq!(metadata_target(&p, 2), None);
        p.calibration_type = xma_model::CalibrationType::None;
        assert_eq!(metadata_target(&p, 0), Some(Some(0)));
    }

    #[test]
    fn assignment_by_rank_and_portal_id() {
        let i = import(&[(2, 1), (1, 0)]);
        let a = assign_trial(&i, &[], 2, false);
        assert_eq!(a.trial_name.as_deref(), Some("T"));
        assert_eq!(a.cameras.len(), 2);
        assert_eq!(a.cameras[0].0, 1);
        // A camera whose portal id differs is not set.
        let a = assign_trial(&i, &[1, 5], 2, false);
        assert_eq!(a.cameras.len(), 1);
        assert_eq!(a.cameras[0].0, 0);
        assert!(a.skipped.is_empty());
        // F54: a third camera with two boxes.
        let i = import(&[(1, 0), (2, 1), (3, 2)]);
        let a = assign_trial(&i, &[-1, -1], 2, false);
        assert_eq!(a.cameras.len(), 2);
        assert_eq!(a.skipped.len(), 1);
        let a = assign_trial(&i, &[], 1, true);
        assert_eq!(a.boxes, 3);
        assert_eq!(a.cameras.len(), 3);
        // No files: the name is not set.
        let a = assign_trial(&import(&[]), &[], 2, false);
        assert_eq!(a.trial_name, None);
    }
}

#[cfg(test)]
mod review13_tests {
    //! Review 13 (performance round, engine-tracking-view.md finding 5): the Portal sequence
    //! order and `compare_names2` against the code they replaced, kept here as references.

    use super::*;
    use crate::undistort_sequence::compare_names2;

    /// `compare_names2` before 20 November 2026.
    fn reference_compare(s1: &str, s2: &str) -> bool {
        fn number_less(a: &str, b: &str) -> bool {
            fn value(s: &str) -> &str {
                if s.chars().all(|c| c.is_ascii_digit()) {
                    s.trim_start_matches('0')
                } else {
                    ""
                }
            }
            let (a, b) = (value(a), value(b));
            a.len().cmp(&b.len()).then_with(|| a.cmp(b)).is_lt()
        }
        let a: Vec<char> = s1.chars().collect();
        let b: Vec<char> = s2.chars().collect();
        let lower = |c: char| c.to_lowercase().next().unwrap_or(c);
        let num = |c: char| c.is_numeric();
        let mut i = 0;
        while i < a.len() && i < b.len() && lower(a[i]) == lower(b[i]) {
            i += 1;
        }
        i += 1;
        if i < a.len() && i < b.len() {
            let k = i - 1;
            if !num(a[k]) || !num(b[k]) {
                if num(a[k]) {
                    return false;
                }
                if num(b[k]) {
                    return true;
                }
                return s1.encode_utf16().lt(s2.encode_utf16());
            }
            let mut j = k;
            while j > 0 && num(a[j - 1]) {
                j -= 1;
            }
            let n: String = a[j..k].iter().collect();
            let run = |v: &[char]| -> String { v[k..].iter().take_while(|c| num(**c)).collect() };
            number_less(&format!("{n}{}", run(&a)), &format!("{n}{}", run(&b)))
        } else {
            a.len() < b.len()
        }
    }

    /// The insertion of `camera_files` before 20 November 2026.
    fn reference_order(files: Vec<PathBuf>) -> Vec<PathBuf> {
        let mut sorted: Vec<PathBuf> = Vec::with_capacity(files.len());
        for f in files {
            let s = f.to_string_lossy().into_owned();
            let at = sorted
                .iter()
                .position(|g| reference_compare(&s, &g.to_string_lossy()))
                .unwrap_or(sorted.len());
            sorted.insert(at, f);
        }
        sorted
    }

    struct Lcg(u64);
    impl Lcg {
        fn below(&mut self, n: usize) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            usize::try_from(self.0 >> 33).unwrap() % n.max(1)
        }
    }

    /// Characters that exercise every branch: ASCII digits, non-ASCII numerics (one with a
    /// lower case), letters in both cases (one whose lower case has two characters),
    /// separators, and characters outside the BMP.
    const ALPHABET: [char; 26] = [
        '0', '0', '1', '2', '9', '5', 'a', 'A', 'b', 'B', 'x', 'X', '.', '_', '-', ' ', '/', 'Ⅻ',
        'ⅻ', '²', '٣', 'É', 'é', 'İ', 'ß', '𝟘',
    ];

    fn random_name(r: &mut Lcg) -> String {
        let mut s = String::new();
        for _ in 0..r.below(6) {
            match r.below(4) {
                // A number group of 1 to 7 digits, maybe zero-padded.
                0 => {
                    for _ in 0..1 + r.below(7) {
                        s.push(['0', '0', '1', '3', '7', '9'][r.below(6)]);
                    }
                }
                1 => s.push_str(["img", "IMG", "Img", "cam", "tif"][r.below(5)]),
                _ => s.push(ALPHABET[r.below(ALPHABET.len())]),
            }
        }
        s
    }

    /// A random set: random names, some derived from others (case changed, a digit
    /// changed, digits added or removed, cut short), maybe in a common folder.
    fn random_set(r: &mut Lcg) -> Vec<String> {
        let n = r.below(40);
        let mut names: Vec<String> = Vec::new();
        for _ in 0..n {
            if names.is_empty() || r.below(2) == 0 {
                names.push(random_name(r));
                continue;
            }
            let base: Vec<char> = names[r.below(names.len())].chars().collect();
            let mut v = base.clone();
            match r.below(5) {
                0 => {
                    v = v
                        .iter()
                        .map(|c| {
                            if r.below(2) == 0 {
                                c.to_uppercase().next().unwrap_or(*c)
                            } else {
                                c.to_lowercase().next().unwrap_or(*c)
                            }
                        })
                        .collect();
                }
                1 if !v.is_empty() => {
                    let i = r.below(v.len());
                    v[i] = ALPHABET[r.below(6)];
                }
                2 => {
                    let i = r.below(v.len() + 1);
                    v.insert(i, ALPHABET[r.below(5)]);
                }
                3 if !v.is_empty() => {
                    v.truncate(r.below(v.len()));
                }
                _ => v.push_str_chars(&random_name(r)),
            }
            names.push(v.into_iter().collect());
        }
        let folder = ["", "/data/Seq1/", "/d/", "C:\\x\\"][r.below(4)];
        names.into_iter().map(|n| format!("{folder}{n}")).collect()
    }

    trait PushStr {
        fn push_str_chars(&mut self, s: &str);
    }
    impl PushStr for Vec<char> {
        fn push_str_chars(&mut self, s: &str) {
            self.extend(s.chars());
        }
    }

    /// `sets` random sets (the same ones whatever `sets` is), every pair in each.
    fn compare_names2_sets(sets: usize) {
        let mut r = Lcg(5);
        let mut pairs = 0;
        for _ in 0..sets {
            let set = random_set(&mut r);
            for a in &set {
                for b in &set {
                    assert_eq!(compare_names2(a, b), reference_compare(a, b), "{a:?} {b:?}");
                    pairs += 1;
                }
            }
        }
        for (a, b) in [
            ("", ""),
            ("", "1"),
            ("1", ""),
            ("a", "A"),
            ("A", "a"),
            ("Ⅻ1", "ⅻ2"),
        ] {
            assert_eq!(compare_names2(a, b), reference_compare(a, b), "{a:?} {b:?}");
        }
        assert!(pairs > 100_000 * sets / 4000);
    }

    #[test]
    fn review13_compare_names2() {
        compare_names2_sets(400);
    }

    #[test]
    #[ignore = "long comparison run; cargo test -- --ignored _long"]
    fn review13_compare_names2_long() {
        compare_names2_sets(4000);
    }

    /// `sets` random sets (the same ones whatever `sets` is), ordered both ways.
    fn portal_sequence_order_sets(sets: usize) {
        let mut r = Lcg(9);
        let mut inconsistent = 0;
        for _ in 0..sets {
            let mut set: Vec<PathBuf> = random_set(&mut r).into_iter().map(PathBuf::from).collect();
            if r.below(2) == 0 {
                set.sort();
            }
            let want = reference_order(set.clone());
            assert_eq!(natural_order(set.clone()), want, "{set:?}");
            // Sets the comparator does not order consistently are among them.
            let names: Vec<String> = set
                .iter()
                .map(|p| p.to_string_lossy().into_owned())
                .collect();
            let cmp = |a: &String, b: &String| reference_compare(a, b);
            if names.iter().any(|a| {
                names.iter().any(|b| {
                    names.iter().any(|c| cmp(a, b) && cmp(b, c) && !cmp(a, c))
                        || (!cmp(a, b)
                            && !cmp(b, a)
                            && names
                                .iter()
                                .any(|c| !cmp(b, c) && !cmp(c, b) && (cmp(a, c) || cmp(c, a))))
                })
            }) {
                inconsistent += 1;
            }
        }
        assert!(
            inconsistent > 50 * sets / 6000,
            "only {inconsistent} inconsistent sets"
        );
        assert_eq!(natural_order(Vec::new()), Vec::<PathBuf>::new());
        let one = vec![PathBuf::from("")];
        assert_eq!(natural_order(one.clone()), reference_order(one));
    }

    #[test]
    fn review13_portal_sequence_order() {
        portal_sequence_order_sets(600);
    }

    /// Named `_sets_long`: `review13_portal_sequence_order_long` below is the long run of
    /// another test.
    #[test]
    #[ignore = "long comparison run; cargo test -- --ignored _long"]
    fn review13_portal_sequence_order_sets_long() {
        portal_sequence_order_sets(6000);
    }

    /// An `n`-image zero-padded sequence, sorted as `camera_files` gives it and shuffled.
    fn padded_sequence_order(n: usize) {
        let mut files: Vec<PathBuf> = (0..n)
            .map(|i| PathBuf::from(format!("/data/trial 1/cam1/img.{i:05}.tif")))
            .collect();
        files.sort();
        let want = reference_order(files.clone());
        assert_eq!(natural_order(files.clone()), want);
        let mut r = Lcg(3);
        for i in (1..files.len()).rev() {
            files.swap(i, r.below(i + 1));
        }
        assert_eq!(natural_order(files.clone()), reference_order(files));
    }

    /// 500 images (both sorts are insertion sorts, so the length reaches no other branch).
    #[test]
    fn review13_portal_sequence_order_padded() {
        padded_sequence_order(500);
    }

    /// The 10,000-image sequence.
    #[test]
    #[ignore = "long comparison run; cargo test -- --ignored _long"]
    fn review13_portal_sequence_order_long() {
        padded_sequence_order(10_000);
    }
}
