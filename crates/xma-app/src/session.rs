//! The open project: loading it on a worker thread, finding a trial's video files when they
//! are not where the project says, closing it, and the window title.
//!
//! Ported from ui/MainWindow.cpp:
//!
//! - `MainWindow::loadProject(fileName)` (`:763-785`): remember the folder and the file in
//!   the recent files, start the load on a worker thread, and show the progress bar
//!   "Load dataset <file>" (non-cancelable, so the window is disabled while it runs).
//!   Here the worker runs [`Document::open_with`], which includes the after-load steps
//!   (`Document::update_after_load`), and then the search for missing video files below.
//! - `MainWindow::loadProjectFinished` and `UndistortionAfterloadProjectFinished`: the
//!   interface part of what follows a load is in `crate::app` (`XmaApp::project_loaded`).
//! - `MainWindow::checkTrialImagePaths` (`:1130-1220`): for each trial that is not the
//!   default trial, and each camera, if the first file of the camera's video does not
//!   exist, search the project file's folder and all its subfolders for a file of the same
//!   name ([`find_file_recursive`], a port of `listFiles`); if found, the camera's video
//!   files are taken from that folder. If not found, the C++ asks for a folder (image
//!   sequences) or a file (a single video file) with a file dialog titled "<name> not found.
//!   Enter a new directory for <name>" / "... Enter a file for <name>"; here the same
//!   request is queued as a dialog ([`MissingVideo`]) that offers the file dialog. All of
//!   this is skipped when the `DisableImageSearch` setting is on.
//!
//!   The C++ rewrites the trial's file names (`Trial::changeImagePath`). This viewer does
//!   not change the document: the replacement folder or file is kept here
//!   ([`Session::video_files`] gives the files to read) and is not saved.
//! - `ProjectFileIO::loadProject`'s Portal branch (`:293-302`): a file with no
//!   `project.xml` whose first folder holds `File_Metadata/XMALab_Files-metadata.xml` is an
//!   XMA Portal project. The worker recognises it before reading the archive as a project
//!   (only its directory is read), extracts it (`xma_engine::portal::import_project`) and
//!   hands the result to [`Session::take_portal_project`]; the application then opens the
//!   New dataset dialog filled from it (`MainWindow::newProjectFromXMALab`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use xma_engine::{Document, OpenOptions};

use crate::jobs::{JobHandle, Jobs, Polled, Worker};

/// Where a camera's video files were found after loading.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VideoLocation {
    /// The paths stored in the project exist.
    AsStored,
    /// The files were found in another folder (search, or chosen by the user): every
    /// stored file name is taken from this folder.
    Folder(PathBuf),
    /// A single video file chosen by the user.
    File(PathBuf),
    /// Not found.
    Missing,
}

/// A camera whose video was not found, waiting for the user.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MissingVideo {
    pub trial: usize,
    pub camera: usize,
    /// File name (without folder) of the first frame or of the video file.
    pub file_name: String,
    /// More than one file: an image sequence, so a folder is asked for.
    pub sequence: bool,
}

impl MissingVideo {
    /// The file dialog title of the C++ (`MainWindow.cpp:1181,1206`).
    #[must_use]
    pub fn prompt(&self) -> String {
        if self.sequence {
            format!(
                "{0} not found. Enter a new directory for {0}",
                self.file_name
            )
        } else {
            format!("{0} not found. Enter a file for {0}", self.file_name)
        }
    }
}

/// What the worker sends back.
pub struct LoadResult {
    pub path: PathBuf,
    pub result: Result<Document, String>,
    /// Per (trial, camera): where the video is. Filled only when the load succeeded.
    pub videos: BTreeMap<(usize, usize), VideoLocation>,
    /// Cameras still missing, in trial and camera order.
    pub missing: Vec<MissingVideo>,
    /// Messages for the log (what the search found).
    pub messages: Vec<String>,
}

/// What the loading worker sends: a project, or an XMA Portal project archive extracted
/// for the New dataset dialog.
enum Loaded {
    Project(Box<LoadResult>),
    Portal(Box<PortalProjectResult>),
}

/// An XMA Portal project archive opened in place of a project: the import, or why it
/// failed (the file's name and the message).
pub type PortalProjectResult = Result<xma_engine::portal::ProjectImport, String>;

/// The open project and the load in progress.
#[derive(Default)]
pub struct Session {
    pub doc: Option<Document>,
    /// Where each trial camera's video is.
    pub videos: BTreeMap<(usize, usize), VideoLocation>,
    /// Where the videos of trials no longer in the project were, by trial name and camera,
    /// so that undoing a trial deletion finds them again (`crate::app` trials).
    pub video_stash: BTreeMap<(String, usize), VideoLocation>,
    loading: Option<Worker<Loaded>>,
    /// The algorithm settings a project that stores none is opened with
    /// (`OpenOptions::base_settings`): this machine's defaults
    /// (`UiSettings::algorithm_base`), set by the application before each load.
    pub base_settings: xma_model::AlgorithmSettings,
    /// The custom workspace for XMA Portal data ("Use custom workspace to save data
    /// imported from XMA Portal"), set by the application before each load.
    pub portal_workspace: Option<PathBuf>,
    /// A Portal project archive opened in place of a project ([`Session::take_portal_project`]).
    portal_project: Option<PortalProjectResult>,
}

impl Session {
    #[must_use]
    pub fn is_loading(&self) -> bool {
        self.loading.is_some()
    }

    /// Starts loading a project on a worker thread, with a progress job titled as the C++
    /// ("Load dataset <file>").
    ///
    /// `read_from` reads another file in place of `path`: the project's recovery file
    /// (`crate::saving`). The document then has `path` as its file and is marked as not
    /// saved, since it differs from that file.
    pub fn start_load(
        &mut self,
        path: &Path,
        read_from: Option<&Path>,
        jobs: &mut Jobs,
        disable_image_search: bool,
    ) {
        self.start_load_from(path, Source::File(read_from), jobs, disable_image_search);
    }

    /// File > Open Dataset with different calibration (C62): starts loading the trials of
    /// `trials` with the calibration of `calibration` on the worker thread
    /// (`Document::open_with_different_calibration`), with the progress job "Load dataset
    /// <trials file>" as in XMALab 3. The document's file is `trials`.
    pub fn start_load_with_calibration(
        &mut self,
        trials: &Path,
        calibration: &Path,
        jobs: &mut Jobs,
        disable_image_search: bool,
    ) {
        self.start_load_from(
            trials,
            Source::Calibration(calibration),
            jobs,
            disable_image_search,
        );
    }

    fn start_load_from(
        &mut self,
        path: &Path,
        source: Source<'_>,
        jobs: &mut Jobs,
        disable_image_search: bool,
    ) {
        let (read_from, calibration) = match source {
            Source::File(r) => (r.map(Path::to_path_buf), None),
            Source::Calibration(c) => (None, Some(c.to_path_buf())),
        };
        let job = jobs.start(format!("Load dataset {}", path.display()), false);
        let path = path.to_path_buf();
        let base = self.base_settings.clone();
        let workspace = self.portal_workspace.clone();
        let spawn = Worker::spawn("xma-load", &job, move |ctx_job| {
            if read_from.is_none()
                && calibration.is_none()
                && let Some(r) = portal_worker(&path, workspace.as_deref(), ctx_job)
            {
                return Loaded::Portal(Box::new(r));
            }
            let res = load_worker(
                &path,
                read_from.as_deref(),
                calibration.as_deref(),
                disable_image_search,
                base,
            );
            Loaded::Project(Box::new(res))
        });
        self.loading = Some(match spawn {
            Ok(w) => w,
            Err(e) => {
                jobs.finish(job.id());
                Worker::ready(
                    job,
                    Loaded::Project(Box::new(LoadResult {
                        path: PathBuf::new(),
                        result: Err(format!("Cannot start the loading thread: {e}")),
                        videos: BTreeMap::new(),
                        missing: Vec::new(),
                        messages: Vec::new(),
                    })),
                )
            }
        });
    }

    /// Loads synchronously (tests and `--screenshot` runs that want the result at once).
    pub fn load_blocking(
        &mut self,
        path: &Path,
        jobs: &mut Jobs,
        disable_image_search: bool,
    ) -> LoadResult {
        let job = jobs.start(format!("Load dataset {}", path.display()), false);
        let r = load_worker(
            path,
            None,
            None,
            disable_image_search,
            self.base_settings.clone(),
        );
        jobs.finish(job.id());
        r
    }

    /// The finished load, if the worker is done. Ends the progress job.
    pub fn poll(&mut self, jobs: &mut Jobs) -> Option<LoadResult> {
        let polled = self.loading.as_ref()?.poll(jobs);
        if !matches!(polled, Polled::Running) {
            self.loading = None;
        }
        match polled {
            Polled::Done(Loaded::Project(r)) => Some(*r),
            Polled::Done(Loaded::Portal(r)) => {
                self.portal_project = Some(*r);
                None
            }
            Polled::Running => None,
            Polled::Stopped => Some(LoadResult {
                path: PathBuf::new(),
                result: Err("The loading thread stopped unexpectedly.".into()),
                videos: BTreeMap::new(),
                missing: Vec::new(),
                messages: Vec::new(),
            }),
        }
    }

    /// The XMA Portal project archive the last load found instead of a project, once.
    pub fn take_portal_project(&mut self) -> Option<PortalProjectResult> {
        self.portal_project.take()
    }

    /// Ported from ui/MainWindow.cpp, MainWindow::closeProject (the document part).
    pub fn close(&mut self) {
        self.doc = None;
        self.videos.clear();
        self.video_stash.clear();
    }

    /// The files of a trial camera's video, with any replacement folder or file applied.
    #[must_use]
    pub fn video_files(&self, trial: usize, camera: usize) -> Vec<PathBuf> {
        let Some(doc) = &self.doc else {
            return Vec::new();
        };
        let Some(v) = doc
            .project()
            .trials
            .get(trial)
            .and_then(|t| t.videos.get(camera))
        else {
            return Vec::new();
        };
        match self.videos.get(&(trial, camera)) {
            Some(VideoLocation::Folder(dir)) => v
                .filenames
                .iter()
                .map(|f| dir.join(file_name_of(f)))
                .collect(),
            Some(VideoLocation::File(f)) => vec![f.clone()],
            _ => v
                .filenames
                .iter()
                .map(|f| PathBuf::from(adjust_path_to_os(f)))
                .collect(),
        }
    }

    /// The user chose a folder or file for a missing video.
    ///
    /// For an image sequence the choice is either a folder holding the images
    /// ([`VideoLocation::Folder`]) or one video file used in their place
    /// ([`VideoLocation::File`], new in this program); a missing video file is replaced by a
    /// file.
    pub fn resolve_missing(&mut self, m: &MissingVideo, chosen: PathBuf) {
        let loc = if m.sequence && !chosen.is_file() {
            VideoLocation::Folder(chosen)
        } else {
            VideoLocation::File(chosen)
        };
        self.videos.insert((m.trial, m.camera), loc);
    }
}

/// The window title.
///
/// Ported from ui/MainWindow.cpp, MainWindow::updateWindowTitle: "RustXMALab <version>"
/// (XMALab 3: "XMALab <version>"; parity ledger U173), prefixed by "<project base name> - "
/// when a project is open. The version is this program's.
#[must_use]
pub fn window_title(project: Option<&Path>) -> String {
    let title = format!("{} {}", crate::APP_NAME, env!("CARGO_PKG_VERSION"));
    match project {
        Some(p) => format!("{} - {title}", project_name(p)),
        None => title,
    }
}

/// The title of the window with an open document: [`window_title`], with a `*` after the
/// project's name while it has unsaved changes (not in XMALab 3, parity ledger U53), and
/// "untitled" for a project that has no file yet.
#[must_use]
pub fn document_title(project: Option<&Path>, modified: bool) -> String {
    let name = project.map_or_else(|| "untitled".to_owned(), project_name);
    let star = if modified { "*" } else { "" };
    format!(
        "{name}{star} - {} {}",
        crate::APP_NAME,
        env!("CARGO_PKG_VERSION")
    )
}

/// `QFileInfo::completeBaseName`: the file name up to its last dot.
#[must_use]
pub fn project_name(p: &Path) -> String {
    let name = p
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    name.rsplit_once('.')
        .map_or(name.as_str(), |(b, _)| b)
        .to_owned()
}

/// Where a load reads from: the file itself or a recovery copy, or (C62) the trials of the
/// file with another file's calibration.
enum Source<'a> {
    File(Option<&'a Path>),
    Calibration(&'a Path),
}

/// `loadProject`'s Portal branch: `None` when `path` is not an XMA Portal archive (a
/// project, another file); otherwise its extraction for the New dataset dialog, the
/// progress shown on `job`.
fn portal_worker(
    path: &Path,
    workspace: Option<&Path>,
    job: &JobHandle,
) -> Option<PortalProjectResult> {
    match xma_engine::portal::PortalArchive::probe(path) {
        Ok(Some(_)) => {}
        Ok(None) | Err(_) => return None,
    }
    let dest = xma_engine::portal::Destination {
        workspace,
        project_file: None,
        portal_file: path,
    };
    Some(
        xma_engine::portal::import_project(
            path,
            &dest,
            xma_model::CppCompat::default(),
            &mut |f| {
                job.set_progress(f.min(0.99));
                true
            },
        )
        .map_err(|e| format!("{}: {e}", path.display())),
    )
}

fn load_worker(
    path: &Path,
    read_from: Option<&Path>,
    calibration: Option<&Path>,
    disable_image_search: bool,
    base_settings: xma_model::AlgorithmSettings,
) -> LoadResult {
    let options = OpenOptions {
        base_settings,
        ..OpenOptions::default()
    };
    let opened = match calibration {
        Some(c) => {
            Document::open_with_different_calibration(c, path, &options).map_err(|e| e.to_string())
        }
        None => Document::open_with(read_from.unwrap_or(path), &options).map_err(|e| e.to_string()),
    };
    let result = opened.map(|mut doc| {
        if read_from.is_some() {
            doc.set_path(path);
            doc.mark_unsaved();
        }
        doc
    });
    let mut videos = BTreeMap::new();
    let mut missing = Vec::new();
    let mut messages = Vec::new();
    if let (Ok(_), Some(r)) = (&result, read_from) {
        messages.push(format!("Opened the recovery copy {}", r.display()));
    }
    if let Ok(doc) = &result
        && !disable_image_search
    {
        check_trial_image_paths(doc, path, &mut videos, &mut missing, &mut messages);
    }
    LoadResult {
        path: path.to_path_buf(),
        result,
        videos,
        missing,
        messages,
    }
}

/// Ported from littleHelper::adjustPathToOS (core/HelperFunctions.cpp): both separators
/// become the platform's. The engine's function, which every reader of stored paths uses.
pub use xma_engine::adjust_path_to_os;

fn file_name_of(f: &str) -> String {
    let f = adjust_path_to_os(f);
    Path::new(&f)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or(f)
}

/// Ported from ui/MainWindow.cpp, MainWindow::checkTrialImagePaths (see the module
/// documentation). Relative stored paths are taken relative to the project's folder.
pub fn check_trial_image_paths(
    doc: &Document,
    project_file: &Path,
    videos: &mut BTreeMap<(usize, usize), VideoLocation>,
    missing: &mut Vec<MissingVideo>,
    messages: &mut Vec<String>,
) {
    let project_dir = std::path::absolute(project_file)
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or_default();
    for (t, trial) in doc.project().trials.iter().enumerate() {
        if trial.is_default {
            continue;
        }
        for (c, video) in trial.videos.iter().enumerate() {
            let Some(first) = video.filenames.first() else {
                continue;
            };
            let stored = PathBuf::from(adjust_path_to_os(first));
            let stored = if stored.is_absolute() {
                stored
            } else {
                project_dir.join(stored)
            };
            if stored.exists() {
                videos.insert((t, c), VideoLocation::AsStored);
                continue;
            }
            let name = file_name_of(first);
            match find_file_recursive(&project_dir, &name) {
                Some(dir) => {
                    messages.push(format!(
                        "{}: {} not found; using {}",
                        trial.name,
                        stored.display(),
                        dir.join(&name).display()
                    ));
                    videos.insert((t, c), VideoLocation::Folder(dir));
                }
                None => {
                    messages.push(format!("{}: {} not found", trial.name, stored.display()));
                    videos.insert((t, c), VideoLocation::Missing);
                    missing.push(MissingVideo {
                        trial: t,
                        camera: c,
                        file_name: name,
                        sequence: video.filenames.len() > 1,
                    });
                }
            }
        }
    }
}

/// Ported from ui/MainWindow.cpp, `listFiles`: depth-first search of `dir` for an entry
/// named `name`; returns the folder that holds the first one found. Entries are visited in
/// name order ignoring case (`QDir`'s default sort), files and folders together.
#[must_use]
pub fn find_file_recursive(dir: &Path, name: &str) -> Option<PathBuf> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .collect();
    entries.sort_by_key(|e| e.file_name().to_string_lossy().to_lowercase());
    for e in entries {
        let path = e.path();
        if e.file_name().to_string_lossy() == name {
            return Some(dir.to_path_buf());
        }
        // Symbolic links to folders are not followed (QDir lists them, but following them
        // could loop).
        if e.file_type().is_ok_and(|t| t.is_dir())
            && let Some(found) = find_file_recursive(&path, name)
        {
            return Some(found);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn title_like_cpp() {
        let v = env!("CARGO_PKG_VERSION");
        assert_eq!(window_title(None), format!("RustXMALab {v}"));
        assert_eq!(
            window_title(Some(Path::new("/a/pig.data.xma"))),
            format!("pig.data - RustXMALab {v}")
        );
        assert_eq!(
            document_title(Some(Path::new("/a/pig.data.xma")), true),
            format!("pig.data* - RustXMALab {v}")
        );
        assert_eq!(
            document_title(Some(Path::new("/a/pig.xma")), false),
            format!("pig - RustXMALab {v}")
        );
        assert_eq!(
            document_title(None, true),
            format!("untitled* - RustXMALab {v}")
        );
    }

    #[test]
    fn recursive_search_finds_nested_file() {
        let base = std::env::temp_dir().join(format!("xma_app_search_{}", std::process::id()));
        let deep = base.join("B").join("c");
        std::fs::create_dir_all(&deep).unwrap();
        std::fs::create_dir_all(base.join("a")).unwrap();
        std::fs::write(deep.join("frame_001.png"), b"x").unwrap();
        assert_eq!(
            find_file_recursive(&base, "frame_001.png"),
            Some(deep.clone())
        );
        assert_eq!(find_file_recursive(&base, "nope.png"), None);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn missing_prompt_texts() {
        let m = MissingVideo {
            trial: 0,
            camera: 1,
            file_name: "x.avi".into(),
            sequence: false,
        };
        assert_eq!(m.prompt(), "x.avi not found. Enter a file for x.avi");
    }
}
