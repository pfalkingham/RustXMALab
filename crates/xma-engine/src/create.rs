//! Creating projects and trials from raw files: File > New dataset, New dataset (external
//! calibration), New trial without calibration, Open Dataset with different calibration,
//! the "+" (new trial) button, "Add Default Trial" and "Change Trial Data".
//!
//! The specifications hold what the dialogs collect, already in the order the dialogs give
//! (C49: `QStringList::sort`; C57, C58: `littleHelper::compareNames`). The checks the
//! dialogs make before accepting (`isComplete`) are [`NewProjectSpec::check`] and
//! [`NewTrialSpec::check`], so that the window and `xma` refuse the same input with the
//! same sentence. What each action does in the C++ is in `docs/cpp-calibration.md`,
//! section 7 (C47 to C63).
//!
//! # Threads and errors
//!
//! Every function here reads the files it is given (images, videos, the calibration
//! object's files) and may take seconds (a long calibration video is opened to count its
//! frames). The C++ runs New dataset and Open with different calibration on a worker
//! thread behind the progress dialog "Create new dataset" / "Load dataset <file>"; the
//! functions are plain and may be called from any thread. [`CreateError::Incomplete`] and
//! [`CreateError::Refused`] carry XMALab's error-dialog sentence; [`CreateError::Io`] a
//! message naming the file.
//!
//! # Images
//!
//! Calibration and grid images are kept as archive entries (the document's attachments),
//! as XMALab keeps them in memory and writes them into the project at the first save
//! (ledger V6): an 8-bit file is kept byte for byte (XMALab re-encodes it with OpenCV,
//! the same pixels for a lossless format), a deeper file is reduced to 8 bits as XMALab
//! reduces it and encoded in the format its name gives. Videos are referenced by absolute
//! path and never copied.
//!
//! # Fixed C++ defects
//!
//! Each is reproduced in the XMALab 3 mode where the C++ behaviour is defined (ledger
//! F34 to F39, A128): a resolution mismatch at creation (F34), calibration videos of
//! different lengths (F35), trial videos of different lengths (F36), Change Trial Data's
//! arrays and its rename on refusal (F37), Open with different calibration's missing checks
//! (F38, both modes), the Default trial's uninitialised values (F39, both modes) and the
//! external calibration's frame count (A128).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use xma_io::calibration_object::{load_coords, reference_entries};
use xma_io::zip::{DosDateTime, Entry, Method};
use xma_io::{Attachment, FileVersion, LoadedProject, TrialSource};
use xma_model::{
    AlgorithmSettings, CalibrationImage, CalibrationObject, CalibrationSequence, CalibrationType,
    Camera, Project, SettingsPatch, Trial, UndistortionGrid, VideoSource,
};
use xma_video::open::{OpenOptions as VideoOptions, open_source, open_video};
use xma_video::{Pixels, VideoCompat};

use crate::document::{Document, OpenOptions, read_project};
use crate::edit::{EditError, EditOutcome, EditResult};
use crate::session::{Session, Workspace};

/// One camera box of the New dataset dialog (`ui/CameraBox.cpp`, C49 and C50).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NewCameraSpec {
    /// The box's title: "Camera 1", "Camera 2", ...
    pub name: String,
    /// The calibration images, or one video file.
    pub images: Vec<PathBuf>,
    /// "Light Camera" chosen instead of "XRay Camera".
    pub light_camera: bool,
    /// The undistortion grid image: `Some` only for an X-ray camera whose "Undistortion
    /// grid" box is checked. `Some` of an empty path is a checked box without a file
    /// ("Missing Undistortiongrid").
    pub grid: Option<PathBuf>,
    /// "Flip images horizontally".
    pub flipped: bool,
}

/// The calibration target chosen in the New dataset dialog (C51).
#[derive(Clone, Debug, PartialEq)]
pub enum CalibrationTarget {
    /// "Cube": the frame specification (`.csv`) and the reference points (`.ref`).
    Cube {
        frame_specification: PathBuf,
        references: PathBuf,
    },
    /// "Checkerboard": the corners per row and per column ("Nb squares horizontal",
    /// "Nb squares vertical": `cv::findChessboardCorners`'s pattern size) and the square
    /// size (`CalibrationObject::setCheckerboard`).
    Checkerboard {
        horizontal_squares: u32,
        vertical_squares: u32,
        square_size: f64,
    },
}

/// What the New dataset dialog collects (C47 to C52).
#[derive(Clone, Debug, PartialEq)]
pub struct NewProjectSpec {
    pub cameras: Vec<NewCameraSpec>,
    pub target: CalibrationTarget,
}

/// What the New trial dialog collects (C55 to C58, and C61 "Change Trial Data").
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NewTrialSpec {
    /// `lineEditTrialName`.
    pub name: String,
    /// Per camera, in camera order: its files (one video, or the images of a sequence).
    pub cameras: Vec<Vec<PathBuf>>,
}

/// What "New trial without calibration" collects (C54): the camera boxes' titles and the
/// trial.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoCalibrationSpec {
    /// "Camera 1", "Camera 2", ...: one per camera box.
    pub camera_names: Vec<String>,
    pub trial: NewTrialSpec,
}

/// Why nothing was created.
#[derive(Debug)]
pub enum CreateError {
    /// The dialog's own check refused the input; the text is the C++ error dialog's.
    Incomplete(String),
    /// The C++ refuses while creating ("<camera> : Resolutions do not match", "Width and
    /// Height of the videos does is not equal to the calibration images. Cannot create
    /// Trial."); the text is its message.
    Refused(String),
    /// A file could not be read.
    Io(String),
    /// Not built yet, or deferred (XMA Portal).
    NotImplemented(&'static str),
}

impl std::fmt::Display for CreateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CreateError::Incomplete(t) | CreateError::Refused(t) | CreateError::Io(t) => {
                write!(f, "{t}")
            }
            CreateError::NotImplemented(t) => write!(f, "{t} is not implemented"),
        }
    }
}

impl std::error::Error for CreateError {}

/// The sentence the C++ shows when a trial's videos do not have the calibration images'
/// size (`ui/NewTrialDialog.cpp:173`).
pub const TRIAL_SIZE_REFUSED: &str = "Width and Height of the videos does is not equal to the calibration images. Cannot create Trial.";

/// The same for Change Trial Data (`ui/NewTrialDialog.cpp:149`).
pub const TRIAL_CHANGE_SIZE_REFUSED: &str = "Width and Height of the videos does is not equal to the calibration images. Cannot change data for Trial.";

impl NewProjectSpec {
    /// `NewProjectDialog::isComplete` (C52).
    ///
    /// Ported from ui/NewProjectDialog.cpp, NewProjectDialog::isComplete (`:173-232`),
    /// NewProjectDialog::referencesValid (`:150-171`) and ui/CameraBox.cpp,
    /// CameraBox::isComplete (`:111-131`). The C++ indexes the first box unchecked; here no
    /// camera at all is refused.
    ///
    /// # Errors
    /// The sentence of the C++ error dialog for the first check that fails.
    pub fn check(&self) -> Result<(), String> {
        let Some(first) = self.cameras.first() else {
            return Err("A new dataset needs at least one camera".to_owned());
        };
        let n = first.images.len();
        for cam in &self.cameras {
            if cam.images.is_empty() {
                return Err(format!("{} is incomplete : Missing Images", cam.name));
            }
            if !cam.light_camera
                && let Some(g) = &cam.grid
                && g.as_os_str().is_empty()
            {
                return Err(format!(
                    "{} is incomplete : Missing Undistortiongrid",
                    cam.name
                ));
            }
            if cam.images.len() != n {
                return Err(format!(
                    "{} has wrong number of images : {} instead of {n}",
                    cam.name,
                    cam.images.len()
                ));
            }
        }
        match &self.target {
            CalibrationTarget::Cube {
                frame_specification,
                references,
            } => {
                if frame_specification.as_os_str().is_empty() {
                    return Err(
                        "Calibrationtarget Cube is incomplete : Frame specifications missing"
                            .to_owned(),
                    );
                }
                if references.as_os_str().is_empty() {
                    return Err(
                        "Calibrationtarget Cube is incomplete : Reference Points missing"
                            .to_owned(),
                    );
                }
                let data = std::fs::read(references).unwrap_or_default();
                if reference_entries(&data) != 4 {
                    return Err("Reference file is invalid. Please ensure that it has 4 entries and that the format is correct".to_owned());
                }
            }
            CalibrationTarget::Checkerboard {
                horizontal_squares,
                vertical_squares,
                square_size,
            } => {
                if *horizontal_squares == 0 {
                    return Err("Calibrationtarget Checkerboard is incomplete : Nb squares horizontal missing".to_owned());
                }
                if *vertical_squares == 0 {
                    return Err("Calibrationtarget Checkerboard is incomplete : Nb squares vertical missing".to_owned());
                }
                if square_size.is_nan() || *square_size <= 0.0 {
                    return Err(
                        "Calibrationtarget Checkerboard is incomplete : Squaresize missing"
                            .to_owned(),
                    );
                }
            }
        }
        Ok(())
    }
}

impl NewTrialSpec {
    /// The New trial dialog's checks on OK (C56 and the row after C60): the name differs
    /// from every other trial's (`replacing`: the trial being changed, whose own name is
    /// allowed), every camera has files, every camera has as many files as the first. An
    /// empty name is accepted, as in the C++.
    ///
    /// Ported from ui/NewTrialDialog.cpp, NewTrialDialog::isComplete (`:202-249`), and
    /// ui/CameraBoxTrial.cpp, CameraBoxTrial::isComplete (`:110-121`). The camera boxes are
    /// titled with the project's camera names.
    ///
    /// # Errors
    /// The sentence of the C++ error dialog for the first check that fails.
    pub fn check(&self, project: &Project, replacing: Option<usize>) -> Result<(), String> {
        let name_taken = project
            .trials
            .iter()
            .enumerate()
            .any(|(i, t)| Some(i) != replacing && t.name == self.name);
        if name_taken {
            return Err(format!(
                "Please choose another Trialname. A trial with the name {} already exists.",
                self.name
            ));
        }
        if self.cameras.len() != project.cameras.len() {
            return Err(format!(
                "The trial needs files for {} cameras; {} given",
                project.cameras.len(),
                self.cameras.len()
            ));
        }
        let n = self.cameras.first().map_or(0, Vec::len);
        for (i, files) in self.cameras.iter().enumerate() {
            let name = &project.cameras[i].name;
            if files.is_empty() {
                return Err(format!("{name} is incomplete : Missing Images"));
            }
            if files.len() != n {
                return Err(format!(
                    "{name} has wrong number of images : {} instead of {n}",
                    files.len()
                ));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------------------

/// The readers' options for a document mode: XMALab 3's readers in its mode.
pub(crate) fn video_options(compat: xma_model::CppCompat) -> VideoOptions {
    VideoOptions {
        compat: if compat == xma_model::CppCompat::XMALAB3 {
            VideoCompat::XMALAB3
        } else {
            VideoCompat::FIXED
        },
        ..VideoOptions::default()
    }
}

/// `QFileInfo(path).fileName()`.
fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// The path as the dialogs give it: absolute.
fn absolute(path: &Path) -> PathBuf {
    std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf())
}

/// An archive entry holding `data`, for the document's attachments.
fn attachment(name: String, data: Vec<u8>) -> Attachment {
    Attachment {
        entry: Entry {
            name: name.as_bytes().to_vec(),
            comment: Vec::new(),
            method: Method::Deflated,
            flags: 0,
            modified: DosDateTime { date: 0, time: 0 },
            version_made_by: 0,
            version_needed: 20,
            internal_attributes: 0,
            external_attributes: 0,
            extra: Vec::new(),
            crc32: 0,
            compressed_size: 0,
            uncompressed_size: data.len() as u64,
        },
        name,
        data,
    }
}

/// A still image as the C++ loads it (`Image::Image`): its size and the bytes the project
/// stores for it (see the module documentation).
fn still_image(path: &Path, compat: &VideoCompat) -> Result<(u32, u32, Vec<u8>), CreateError> {
    let bytes =
        std::fs::read(path).map_err(|e| CreateError::Io(format!("{}: {e}", path.display())))?;
    let frame = xma_video::image_file::decode_image_with(&bytes, path, compat)
        .map_err(|e| CreateError::Io(e.to_string()))?;
    let stored = match &frame.pixels {
        Pixels::Gray8(_) => bytes,
        Pixels::Gray16(_) => {
            xma_video::image_file::encode_gray8(frame.width, frame.height, &frame.to_gray8(), path)
                .map_err(|e| CreateError::Io(e.to_string()))?
        }
    };
    Ok((frame.width, frame.height, stored))
}

/// `QFileInfo::suffix` test of `CalibrationSequence::loadImages`: a calibration video.
/// Ledger F46, fixed: a calibration frame for a new project lists at least one bead, and
/// every reference id names one of its beads (ids below 0, from a 0 in the file, are kept
/// as the C++ keeps them; the cube calibration refuses them, ledger A114).
fn check_frame(
    frame: &xma_model::CalibrationFrame,
    spec_path: &Path,
    refs_path: &Path,
) -> Result<(), CreateError> {
    if frame.points.is_empty() {
        return Err(CreateError::Refused(format!(
            "The frame specification file {} lists no bead",
            spec_path.display()
        )));
    }
    let n = frame.points.len();
    for (k, &id) in frame.reference_ids.iter().enumerate() {
        if usize::try_from(id).is_ok_and(|i| i >= n) {
            return Err(CreateError::Refused(format!(
                "Reference {} of {} is bead {}, but the frame specification file {} lists {n} \
                 bead{}",
                k + 1,
                refs_path.display(),
                i64::from(id) + 1,
                spec_path.display(),
                if n == 1 { "" } else { "s" }
            )));
        }
    }
    Ok(())
}

fn is_calibration_video(path: &Path) -> bool {
    let s = path.to_string_lossy().to_ascii_lowercase();
    [".cine", ".avi", ".mp4", ".mov"]
        .iter()
        .any(|e| s.ends_with(e))
}

/// The settings a new document starts with: the base settings with the overrides.
fn initial_settings(options: &OpenOptions) -> Result<AlgorithmSettings, CreateError> {
    let mut s = options.base_settings.clone();
    options
        .overrides
        .apply(&mut s)
        .map_err(CreateError::Incomplete)?;
    Ok(s)
}

/// A document for a project built here, with its attachments and the session state of a
/// new project.
fn new_document(project: Project, attachments: Vec<Attachment>, options: &OpenOptions) -> Document {
    let session = Session::new_project(&project);
    let mut doc = Document::from_loaded(LoadedProject {
        project,
        attachments,
        file_version: FileVersion {
            text: Some("1.0".to_owned()),
            value: 1.0,
        },
        archive_comment: b"XROMM Project File".to_vec(),
        trial_sources: Vec::new(),
        warnings: Vec::new(),
        settings_sources: Vec::new(),
        stored_settings: SettingsPatch::default(),
        compat: options.compat,
    });
    *doc.session_mut() = session;
    // Nothing of it is on disk yet: closing asks "Save changes?" and autosave keeps a
    // recovery copy (ledger U53).
    doc.mark_unsaved();
    doc
}

/// What one camera's trial files hold.
struct TrialStream {
    files: Vec<String>,
    frames: usize,
    width: u32,
    height: u32,
    fps: f64,
}

/// Opens each camera's files of a trial as `Trial::Trial` does (more than one file an image
/// sequence, one file a video or a one-image sequence) and reads their frame count, size
/// and rate.
///
/// Ported from core/Trial.cpp, Trial::Trial(QString, std::vector<QStringList>&)
/// (`:60-117`): `nbImages` is camera 0's count, `recordingSpeed` camera 0's rate (0 for
/// images).
fn open_streams(
    spec: &NewTrialSpec,
    compat: xma_model::CppCompat,
) -> Result<Vec<TrialStream>, String> {
    let options = video_options(compat);
    spec.cameras
        .iter()
        .map(|files| {
            let files: Vec<PathBuf> = files.iter().map(|f| absolute(f)).collect();
            let source = open_source(&files, &options).map_err(|e| e.to_string())?;
            let info = source.info();
            Ok(TrialStream {
                files: files
                    .iter()
                    .map(|f| f.to_string_lossy().into_owned())
                    .collect(),
                frames: info.frame_count,
                width: info.width,
                height: info.height,
                fps: info.frame_rate.unwrap_or(0.0),
            })
        })
        .collect()
}

/// Ledger F36: the cameras' frame counts compared (refused unless the XMALab 3 mode
/// reproduces the C++, which compares none).
fn check_lengths(
    streams: &[TrialStream],
    names: &[String],
    compat: xma_model::CppCompat,
    ending: &str,
) -> Result<(), String> {
    if compat.trial_video_lengths_unchecked {
        return Ok(());
    }
    let Some(first) = streams.first() else {
        return Ok(());
    };
    for (i, s) in streams.iter().enumerate().skip(1) {
        if s.frames != first.frames {
            return Err(format!(
                "{} has {} frames and {} has {}: the videos of a trial must have the same number of frames. {ending}",
                names[i], s.frames, names[0], first.frames
            ));
        }
    }
    Ok(())
}

/// `Trial::checkTrialImageSizeValid` (`core/Trial.cpp:2432-2441`): each camera's first
/// frame has its camera's size.
fn sizes_valid(project: &Project, streams: &[TrialStream]) -> bool {
    streams
        .iter()
        .zip(&project.cameras)
        .all(|(s, c)| s.width == c.width && s.height == c.height)
}

/// A new trial from its streams (`Trial::Trial`).
fn trial_from_streams(name: &str, streams: Vec<TrialStream>) -> Trial {
    let n = streams.first().map_or(0, |s| s.frames);
    let fps = streams.first().map_or(0.0, |s| s.fps);
    Trial {
        name: name.to_owned(),
        videos: streams
            .into_iter()
            .map(|s| VideoSource {
                filenames: s.files,
                metadata: Vec::new(),
            })
            .collect(),
        markers: Vec::new(),
        rigid_bodies: Vec::new(),
        events: Vec::new(),
        n_frames: n,
        start_frame: 1,
        end_frame: n,
        reference_calibration_image: 0,
        recording_speed: fps,
        cutoff_frequency: 0.0,
        interpolate_3d: false,
        is_default: false,
        is_copy_from_default: false,
        requires_recomputation: true,
        portal_metadata: Vec::new(),
    }
}

fn source_of_new_trial() -> TrialSource {
    TrialSource {
        n_frames_from_file: true,
        recording_speed_from_file: true,
        markers_without_element: Vec::new(),
    }
}

impl Document {
    /// Appends a trial built here and keeps the loader's per-trial records in step.
    fn push_trial(&mut self, trial: Trial) -> usize {
        let n = self.project.trials.len();
        self.project.trials.push(Arc::new(trial));
        let extras = self.extras_mut();
        if extras.trial_sources.len() == n {
            extras.trial_sources.push(source_of_new_trial());
        }
        n
    }

    /// What the C++ runs after appending a trial: `changeActiveTrial(index, true)` (an
    /// update of that trial), `changeActiveFrameTrial(activeFrame, true)` and
    /// `changeWorkspace(DIGITIZATION, true)` (an update of the active trial).
    ///
    /// `changeActiveFrameTrial` reaches the frame bar while it still shows the Calibration
    /// workspace's calibration frames (`SequenceNavigationFrame::activeFrameChanged` sets its
    /// spin box, whose handler runs `changeFrame` in the Calibration workspace): the trial's
    /// active frame becomes the active calibration frame when it is one
    /// (`changeActiveFrameCalibration(f, true)`, with its `setDialog`).
    ///
    /// Ported from ui/NewTrialDialog.cpp, NewTrialDialog::createTrial (`:176-181`) and
    /// on_pushButton_Default_clicked (`:274-284`), and ui/SequenceNavigationFrame.cpp
    /// (`activeFrameChanged`, `on_spinBoxFrame_valueChanged`, `changeFrame`).
    fn activate_new_trial(&mut self, index: usize) {
        self.select_trial(index);
        // A new trial's active frame is its first (`Trial::Trial`).
        let f = 0;
        let s = self.session();
        if s.workspace == Workspace::Calibration
            && f < s.n_calibration_images
            && f != s.active_frame_calibration
        {
            let camera = s.active_camera;
            self.set_active_calibration_image(camera, f);
        }
        self.set_workspace(Workspace::Tracking, true);
    }
}

// ---------------------------------------------------------------------------------------
// New documents
// ---------------------------------------------------------------------------------------

impl Document {
    /// File > New dataset after the dialog's OK (`NewProjectDialog::createProject`, C52):
    /// cameras with their calibration images and grids, the calibration object; nothing
    /// computed, no trials.
    ///
    /// Ported from ui/NewProjectDialog.cpp, NewProjectDialog::createProject (`:123-149`),
    /// core/Camera.cpp (`Camera::Camera`, `loadImages`, `loadUndistortionImage`,
    /// `setResolutions`), core/CalibrationSequence.cpp, CalibrationSequence::loadImages
    /// (`:116-148`), core/Project.cpp, Project::addCamera, and what
    /// ui/MainWindow.cpp, MainWindow::newProjectFinished and setupProjectUI set (the
    /// [`Session`]).
    ///
    /// The dialog's check ([`NewProjectSpec::check`]) is run first. A camera whose images
    /// and grid do not share one size is refused with "<camera> : Resolutions do not
    /// match" (ledger F34: in the XMALab 3 mode the project is created with the cameras
    /// before it and no calibration object, and the sentence is in
    /// [`Document::warnings`]); calibration videos of different lengths are refused
    /// (ledger F35).
    ///
    /// # Errors
    /// See [`CreateError`].
    pub fn new_project(
        spec: &NewProjectSpec,
        options: &OpenOptions,
    ) -> Result<Document, CreateError> {
        spec.check().map_err(CreateError::Incomplete)?;
        // Ledger F74: a board beyond what the program accepts (the dialog allows 99 x 99).
        if let CalibrationTarget::Checkerboard {
            horizontal_squares: h,
            vertical_squares: v,
            ..
        } = &spec.target
            && u64::from(*h) * u64::from(*v) > xma_model::project::MAX_CHECKERBOARD_POINTS
        {
            return Err(CreateError::Refused(format!(
                "a checkerboard of {h} x {v} corners is larger than this program accepts ({} corners)",
                xma_model::project::MAX_CHECKERBOARD_POINTS
            )));
        }
        let compat = options.compat;
        let vopts = video_options(compat);
        let mut project = Project::new();
        project.calibration_type = CalibrationType::Internal;
        project.algorithm_settings = initial_settings(options)?;
        let mut attachments = Vec::new();
        let mut partial: Option<String> = None;
        let mut frames_first: Option<(usize, bool)> = None;

        for box_ in &spec.cameras {
            let mut cam = Camera::new(box_.name.clone());
            cam.flipped = box_.flipped;
            let mut sizes: Vec<(u32, u32)> = Vec::new();
            let mut cam_attachments = Vec::new();
            let first = &box_.images[0];
            if first.as_os_str().is_empty() {
                cam.calibration_images
                    .push(CalibrationImage::new(String::new()));
            } else if is_calibration_video(first) {
                let path = absolute(first);
                let source = open_video(&path, &vopts)
                    .map_err(|e| CreateError::Io(format!("{}: {e}", path.display())))?;
                let info = source.info();
                for _ in 0..info.frame_count {
                    cam.calibration_images
                        .push(CalibrationImage::new(String::new()));
                }
                cam.calibration_sequence = Some(CalibrationSequence {
                    filename: path.to_string_lossy().into_owned(),
                    n_frames: info.frame_count,
                    width: info.width,
                    height: info.height,
                });
            } else {
                for image in &box_.images {
                    let (w, h, data) = still_image(image, &vopts.compat)?;
                    let name = file_name(image);
                    sizes.push((w, h));
                    cam_attachments.push(attachment(format!("{}/{name}", cam.name), data));
                    cam.calibration_images.push(CalibrationImage::new(name));
                }
            }
            let mut grid_size = None;
            if !box_.light_camera
                && let Some(g) = &box_.grid
            {
                let (w, h, data) = still_image(g, &vopts.compat)?;
                let name = file_name(g);
                grid_size = Some((w, h));
                cam_attachments.push(attachment(format!("{}/{name}", cam.name), data));
                cam.undistortion = Some(UndistortionGrid {
                    image_filename: name,
                    computed: false,
                    points_detected: Vec::new(),
                    grid_distorted: Vec::new(),
                    grid_references: Vec::new(),
                    grid_inlier: Vec::new(),
                    center: None,
                });
            }
            // Camera::setResolutions.
            let (w, h) = match &cam.calibration_sequence {
                Some(s) => (s.width, s.height),
                None => sizes.first().copied().unwrap_or((0, 0)),
            };
            cam.width = w;
            cam.height = h;
            let mismatch =
                sizes.iter().any(|&s| s != (w, h)) || grid_size.is_some_and(|s| s != (w, h));
            if mismatch {
                let msg = format!("{} : Resolutions do not match", cam.name);
                if compat.partial_project_on_resolution_mismatch {
                    partial = Some(msg);
                    break;
                }
                return Err(CreateError::Refused(msg));
            }
            // Ledger F35: the dialog compares numbers of files, so a calibration video
            // counts as one image whatever its length.
            let count = cam.calibration_images.len();
            let any_video =
                cam.calibration_sequence.is_some() || frames_first.is_some_and(|(_, v)| v);
            match frames_first {
                None => frames_first = Some((count, cam.calibration_sequence.is_some())),
                Some((m, _))
                    if m != count && any_video && !compat.calibration_video_lengths_unchecked =>
                {
                    return Err(CreateError::Refused(format!(
                        "{} has wrong number of images : {count} instead of {m}",
                        cam.name
                    )));
                }
                Some(_) => {}
            }
            cam.is_light_camera = box_.light_camera;
            attachments.extend(cam_attachments);
            project.cameras.push(Arc::new(cam));
        }

        if partial.is_none()
            && let CalibrationTarget::Checkerboard {
                horizontal_squares,
                vertical_squares,
                square_size,
            } = &spec.target
        {
            // CalibrationObject::setCheckerboard, with the setting `CheckerboadInvertedAxis`
            // as it is now.
            project.calibration_object = Some(CalibrationObject::Checkerboard {
                horizontal_squares: *horizontal_squares,
                vertical_squares: *vertical_squares,
                square_size: *square_size,
                inverted_axis: project.algorithm_settings.checkerboard_inverted_axis,
            });
        }
        if partial.is_none()
            && let CalibrationTarget::Cube {
                frame_specification: spec_path,
                references: refs_path,
            } = &spec.target
        {
            let read = |p: &Path| {
                std::fs::read(p).map_err(|e| CreateError::Io(format!("{}: {e}", p.display())))
            };
            let mut frame = load_coords(
                &read(spec_path)?,
                &read(refs_path)?,
                &file_name(spec_path),
                &file_name(refs_path),
            )
            .map_err(|e| CreateError::Io(format!("{}: {e}", spec_path.display())))?;
            if !compat.frame_without_beads_accepted {
                check_frame(&frame, spec_path, refs_path)?;
            }
            // Ledger F52: a bead coordinate that is NaN or infinite (after the header's
            // scale) is refused; XMALab 3 keeps it.
            if !compat.nonfinite_imports_accepted
                && let Some(i) = frame
                    .points
                    .iter()
                    .position(|p| !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite()))
            {
                return Err(CreateError::Io(format!(
                    "{}, line {}: a bead coordinate is not a finite number",
                    spec_path.display(),
                    i + 2
                )));
            }
            // Ledger F47: the positions are scaled; the project keeps no scale to apply.
            frame.scale = xma_model::Point3::new(1.0, 1.0, 1.0);
            project.calibration_object = Some(CalibrationObject::Frame(frame));
        }
        let mut doc = new_document(project, attachments, options);
        if let Some(msg) = partial {
            doc.add_warning(msg);
        }
        Ok(doc)
    }

    /// File > New dataset (external calibration) with `n_cameras` cameras (C53): light
    /// cameras "Camera 1" to "Camera n", each with one calibration image without a file,
    /// size 0 × 0 until the external calibration is applied.
    ///
    /// Ported from ui/MainWindow.cpp, MainWindow::newProjectExternalCalibration
    /// (`:648-676`). The number of calibration frames is 1 (ledger A128: 0 in the XMALab 3
    /// mode, as the C++ counts before giving each camera its image).
    ///
    /// # Errors
    /// [`CreateError::Incomplete`] outside the input dialog's range, 1 to 20.
    pub fn new_external(n_cameras: usize, options: &OpenOptions) -> Result<Document, CreateError> {
        if !(1..=20).contains(&n_cameras) {
            return Err(CreateError::Incomplete(format!(
                "The number of cameras must be between 1 and 20, not {n_cameras}"
            )));
        }
        let mut project = Project::new();
        project.calibration_type = CalibrationType::External;
        project.algorithm_settings = initial_settings(options)?;
        for i in 0..n_cameras {
            let mut cam = Camera::new(format!("Camera {}", i + 1));
            cam.is_light_camera = true;
            // Ledger F48: the project file stores and reads back the name "external".
            let name = if options.compat.external_calibration_image_unnamed {
                String::new()
            } else {
                "external".to_owned()
            };
            cam.calibration_images.push(CalibrationImage::new(name));
            project.cameras.push(Arc::new(cam));
        }
        let mut doc = new_document(project, Vec::new(), options);
        if options.compat.external_calibration_frame_count_zero {
            doc.session_mut().n_calibration_images = 0;
        }
        Ok(doc)
    }

    /// File > New trial without calibration (C54): one camera per box, then the trial.
    ///
    /// Ported from ui/MainWindow.cpp, on_actionNew_trial_without_calibration_triggered
    /// (`:1614-1646`) and ui/NewTrialDialog.cpp, NewTrialDialog::createTrial (`:153-194`):
    /// the cameras are neither flipped nor light cameras and take the size of their first
    /// frame (`Trial::setCameraSizes`); the trial is made active and updated, then Marker
    /// tracking is shown twice (by `createTrial` and by `setupProjectUI`), each an update of
    /// the active trial.
    ///
    /// # Errors
    /// [`CreateError::Incomplete`] for the dialog's checks, [`CreateError::Refused`] for
    /// videos of different lengths (ledger F36), [`CreateError::Io`] for a file that does
    /// not open.
    pub fn new_without_calibration(
        spec: &NoCalibrationSpec,
        options: &OpenOptions,
    ) -> Result<Document, CreateError> {
        let mut project = Project::new();
        project.calibration_type = CalibrationType::None;
        project.algorithm_settings = initial_settings(options)?;
        for name in &spec.camera_names {
            project.cameras.push(Arc::new(Camera::new(name.clone())));
        }
        if project.cameras.is_empty() {
            return Err(CreateError::Incomplete(
                "A trial needs at least one camera".to_owned(),
            ));
        }
        spec.trial
            .check(&project, None)
            .map_err(CreateError::Incomplete)?;
        let streams = open_streams(&spec.trial, options.compat).map_err(CreateError::Io)?;
        check_lengths(
            &streams,
            &spec.camera_names,
            options.compat,
            "Cannot create Trial.",
        )
        .map_err(CreateError::Refused)?;
        for (cam, s) in project.cameras.iter_mut().zip(&streams) {
            let cam = Arc::make_mut(cam);
            cam.width = s.width;
            cam.height = s.height;
        }
        project
            .trials
            .push(Arc::new(trial_from_streams(&spec.trial.name, streams)));
        let mut doc = new_document(project, Vec::new(), options);
        doc.extras_mut().trial_sources = vec![source_of_new_trial()];
        doc.activate_new_trial(0);
        // setupProjectUI: changeWorkspace(DIGITIZATION, true).
        doc.set_workspace(Workspace::Tracking, true);
        doc.history = crate::history::History::default();
        // A new dataset: nothing of it is on disk yet (ledger U53).
        doc.mark_unsaved();
        Ok(doc)
    }

    /// File > Open Dataset with different calibration (C62): the cameras, grids,
    /// calibration and calibration object of `calibration`, the trials of `trials`; the
    /// document's path is `trials`.
    ///
    /// Ported from ui/ProjectFileIO.cpp, ProjectFileIO::loadProject (`:275-330`),
    /// readTrials and loadTrials (`:333-681`), then the normal end of loading. Each trial is
    /// flagged for recomputation. Fixed in both modes (ledger F38): the trials file must
    /// have the same cameras (number and names) as the calibration file (the C++ reads each
    /// trial's videos by the calibration project's camera names and crashes when one is
    /// missing), and a trial whose reference calibration frame does not exist in the
    /// calibration file gets frame 0, with a warning (the C++ reads out of range).
    ///
    /// # Errors
    /// [`CreateError::Io`] when either file cannot be read, [`CreateError::Refused`] when
    /// the cameras differ.
    pub fn open_with_different_calibration(
        calibration: &Path,
        trials: &Path,
        options: &OpenOptions,
    ) -> Result<Document, CreateError> {
        let read = |p: &Path| {
            read_project(p, options).map_err(|e| CreateError::Io(format!("{}: {e}", p.display())))
        };
        let mut calib = read(calibration)?;
        let other = read(trials)?;
        let ours: Vec<&str> = calib
            .project
            .cameras
            .iter()
            .map(|c| c.name.as_str())
            .collect();
        let theirs: Vec<&str> = other
            .project
            .cameras
            .iter()
            .map(|c| c.name.as_str())
            .collect();
        if ours != theirs {
            return Err(CreateError::Refused(format!(
                "The trials of {} were recorded with the cameras {theirs:?}, but the calibration of {} has the cameras {ours:?}. The cameras must be the same, with the same names.",
                trials.display(),
                calibration.display()
            )));
        }
        let n_frames = calib
            .project
            .cameras
            .iter()
            .map(|c| c.calibration_images.len())
            .min()
            .unwrap_or(0);
        let mut warnings = Vec::new();
        let mut new_trials = Vec::new();
        for t in &other.project.trials {
            let mut t = t.as_ref().clone();
            // XMALab's -1 ("no reference", ledger F68) is kept as XMALab keeps it.
            if let Some(r) = t
                .reference_image()
                .filter(|&r| !t.is_default && r >= n_frames.max(1))
            {
                warnings.push(format!(
                    "{}: reference calibration frame {} does not exist in {}; frame 1 is used",
                    t.name,
                    r + 1,
                    calibration.display()
                ));
                t.reference_calibration_image = 0;
            }
            t.requires_recomputation = true;
            for m in &mut t.markers {
                let m = Arc::make_mut(m);
                m.requires_recomputation = true;
                // ProjectFileIO::loadTrials reads no `FilenamePoints3D` / `FilenameStatus3D`
                // (readProjectFile does): the 3D points are a new marker's until the
                // update computes them.
                let n = m.points3d.len();
                m.points3d = vec![
                    xma_model::Point3::new(
                        xma_model::marker::UNSET_3D,
                        xma_model::marker::UNSET_3D,
                        xma_model::marker::UNSET_3D
                    );
                    n
                ];
                m.status3d = vec![xma_model::MarkerStatus::Undefined; n];
                m.error3d = vec![0.0; n];
            }
            new_trials.push(Arc::new(t));
        }
        if calib.project.active_trial >= new_trials.len() {
            calib.project.active_trial = 0;
        }
        calib.project.trials = new_trials;
        calib.trial_sources = other.trial_sources;
        calib.warnings.extend(warnings);
        let mut doc = Document::from_loaded(calib);
        doc.set_path(trials);
        doc.update_after_load();
        doc.history = crate::history::History::default();
        // The combination is in neither file; its path is the trials file, which Save
        // writes over (ledger U53).
        doc.mark_unsaved();
        Ok(doc)
    }
}

// ---------------------------------------------------------------------------------------
// The edits
// ---------------------------------------------------------------------------------------

/// [`crate::Edit::NewTrial`] (C55 to C58): the New trial dialog's OK.
///
/// Ported from ui/NewTrialDialog.cpp, NewTrialDialog::createTrial (`:153-194`, a project
/// with calibration) and ui/MainWindow.cpp, MainWindow::newTrial (`:1091-1105`): the
/// dialog's checks, the trial built from its files, refused with XMALab's sentence when a
/// first frame does not have its camera's size, appended, made active and updated, then
/// Marker tracking shown (an update of the active trial). Videos of different lengths are
/// refused (ledger F36; XMALab 3 mode: camera 0's length).
pub(crate) fn new_trial(doc: &mut Document, spec: &NewTrialSpec) -> EditResult {
    spec.check(&doc.project, None)
        .map_err(EditError::Rejected)?;
    let compat = doc.compat();
    let streams = open_streams(spec, compat).map_err(|e| EditError::Io {
        path: spec
            .cameras
            .first()
            .and_then(|c| c.first())
            .cloned()
            .unwrap_or_default(),
        message: e,
    })?;
    let names: Vec<String> = doc.project.cameras.iter().map(|c| c.name.clone()).collect();
    check_lengths(&streams, &names, compat, "Cannot create Trial.").map_err(EditError::Rejected)?;
    if !sizes_valid(&doc.project, &streams) {
        return Err(EditError::Rejected(TRIAL_SIZE_REFUSED.to_owned()));
    }
    let index = doc.push_trial(trial_from_streams(&spec.name, streams));
    doc.activate_new_trial(index);
    Ok(EditOutcome::none())
}

/// [`crate::Edit::AddDefaultTrial`] (C59).
///
/// Ported from ui/NewTrialDialog.cpp, NewTrialDialog::on_pushButton_Default_clicked
/// (`:274-284`; the button is disabled when a Default trial exists or the project has no
/// calibration, so the edit is invalid then). `Trial::Trial()` leaves the recording speed,
/// cutoff, reference calibration frame, 3D interpolation and recomputation flag
/// uninitialised (ledger F39); here they are 0 and false in both modes.
pub(crate) fn add_default_trial(doc: &mut Document) -> EditResult {
    if doc.project.trials.iter().any(|t| t.is_default) {
        return Err(EditError::Invalid(
            "the project already has a Default trial".to_owned(),
        ));
    }
    if doc.project.calibration_type == CalibrationType::None {
        return Err(EditError::Invalid(
            "a project without calibration has no Default trial".to_owned(),
        ));
    }
    let index = doc.push_trial(Trial::default_trial());
    doc.activate_new_trial(index);
    Ok(EditOutcome::none())
}

/// [`crate::Edit::ChangeTrialData`] (C61): the trial dialog's "Change Trial Data".
///
/// Ported from core/Trial.cpp, Trial::changeTrialData (`:224-285`), ui/NewTrialDialog.cpp,
/// NewTrialDialog::createTrial (`:139-158`) and what ui/WorkspaceNavigationFrame.cpp does
/// with `TRIALDIALOGCHANGE` (`:363-369`: Marker tracking shown, an update of the active
/// trial). The name and files are replaced, the frame count is camera 0's, start 1 and end
/// the count. Fixed (ledger F37): the markers' and events' per-frame arrays take the new
/// length and the trial is flagged for recomputation (XMALab 3 mode: neither, except that
/// arrays shorter than the new count are lengthened in both modes, as the C++ would read
/// past them); a change refused for its videos' size changes nothing (XMALab 3 mode: the
/// trial is renamed first, and the outcome carries the sentence). Videos of different
/// lengths are refused (ledger F36).
pub(crate) fn change_trial_data(
    doc: &mut Document,
    trial: usize,
    spec: &NewTrialSpec,
) -> EditResult {
    let Some(t) = doc.project.trials.get(trial) else {
        return Err(EditError::Invalid(format!("no trial {trial}")));
    };
    if t.is_default {
        return Err(EditError::Invalid(
            "the Default trial has no videos to change".to_owned(),
        ));
    }
    spec.check(&doc.project, Some(trial))
        .map_err(EditError::Rejected)?;
    let compat = doc.compat();
    let streams = open_streams(spec, compat).map_err(|e| EditError::Io {
        path: spec
            .cameras
            .first()
            .and_then(|c| c.first())
            .cloned()
            .unwrap_or_default(),
        message: e,
    })?;
    let names: Vec<String> = doc.project.cameras.iter().map(|c| c.name.clone()).collect();
    check_lengths(&streams, &names, compat, "Cannot change data for Trial.")
        .map_err(EditError::Rejected)?;
    if !sizes_valid(&doc.project, &streams) {
        if compat.change_trial_data_renames_on_refusal {
            Arc::make_mut(&mut doc.project.trials[trial]).name = spec.name.clone();
            return Ok(EditOutcome::refused(TRIAL_CHANGE_SIZE_REFUSED));
        }
        return Err(EditError::Rejected(TRIAL_CHANGE_SIZE_REFUSED.to_owned()));
    }
    let n = streams.first().map_or(0, |s| s.frames);
    let t = Arc::make_mut(&mut doc.project.trials[trial]);
    t.name = spec.name.clone();
    t.videos = streams
        .into_iter()
        .map(|s| VideoSource {
            filenames: s.files,
            metadata: Vec::new(),
        })
        .collect();
    t.n_frames = n;
    t.start_frame = 1;
    t.end_frame = n;
    let keep = compat.change_trial_data_keeps_arrays;
    for m in &mut t.markers {
        if keep && m.n_frames() >= n {
            continue;
        }
        let m = Arc::make_mut(m);
        resize_marker(m, n);
        // Fixed (ledger F37; the author, 21 November 2026, "Recompute at once"): the mean
        // size and size range of the arrays as cut, the values a reopen gives
        // (`Document::from_loaded`). XMALab 3 mode: kept (only lengthened arrays get here,
        // and the C++ recomputes nothing). The mean size is not saved, so a reopen starts
        // from a new marker's -1 (`Marker::new`), which `update_mean_size` leaves when no
        // counted size is left; the size range it always sets.
        if !keep {
            m.mean_size = -1.0;
            crate::markers::update_mean_size(m, compat.size_range_from_sum);
        }
    }
    for e in &mut t.events {
        if keep && e.data.len() >= n {
            continue;
        }
        e.data.resize(n, false);
    }
    if !keep {
        doc.set_trial_requires_recomputation(trial, true);
    }
    doc.set_workspace(Workspace::Tracking, true);
    Ok(EditOutcome::none())
}

/// A marker's per-frame arrays at `n` frames: cut, or lengthened with the values of a new
/// marker (`Marker::Marker`).
fn resize_marker(m: &mut xma_model::Marker, n: usize) {
    use xma_model::{MarkerStatus, Point2, Point3};
    for v in &mut m.views {
        v.points.resize(
            n,
            Point2::new(xma_model::marker::UNSET_2D, xma_model::marker::UNSET_2D),
        );
        v.projected.resize(
            n,
            Point2::new(xma_model::marker::UNSET_2D, xma_model::marker::UNSET_2D),
        );
        v.status.resize(n, MarkerStatus::Undefined);
        v.error.resize(n, 0.0);
        v.size.resize(n, xma_model::marker::UNSET_SIZE);
    }
    m.points3d.resize(
        n,
        Point3::new(
            xma_model::marker::UNSET_3D,
            xma_model::marker::UNSET_3D,
            xma_model::marker::UNSET_3D,
        ),
    );
    m.status3d.resize(n, MarkerStatus::Undefined);
    m.error3d.resize(n, 0.0);
    m.interpolation.resize(n, xma_model::Interpolation::None);
}
