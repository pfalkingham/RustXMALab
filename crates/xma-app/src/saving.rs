//! Saving the document: Save, Save As and Save subdataset as on a worker thread, autosave
//! to a recovery file, and the file dialogs they use.
//!
//! Ported from ui/MainWindow.cpp, MainWindow::saveProject (`:971-995`), saveProjectAs
//! (`:997-1053`) and saveProjectFinished (`:1222-1229`), with the differences of parity
//! ledger U53 and U54:
//!
//! - The archive is written by a **copy** of the document (`Document` shares its parts, so
//!   the copy is cheap) on a worker thread while the Progress panel shows "Save dataset as
//!   <file>" and the window is disabled, as in XMALab 3. When the worker finishes, the
//!   session's document takes the saved state and is marked saved at the revision that was
//!   written ([`SaveJob::revision`]).
//! - The file is written to a temporary file next to the target and moved into place only
//!   when complete (`xma_io::project_write`), so a failed save leaves the previous file as
//!   it was. A failure is reported in the error dialog and the Console, and the document
//!   keeps its file name and its modified state (XMALab 3 reports from the worker thread and
//!   goes on as if the save had worked, `docs/cpp-editing.md` D8).
//! - **Autosave** (not in XMALab 3): while the document is modified, a copy is written every
//!   `AutosaveMinutes` minutes (default 5, 0 switches it off) to its recovery file
//!   ([`recovery_path`]), on a worker thread and without the Progress panel. The recovery
//!   file is removed after a successful save and when the project is closed without unsaved
//!   changes. Opening a project whose recovery file is newer than the project file asks
//!   whether to open the recovered version.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use xma_engine::{Document, SaveOptions, SaveReport, SaveTime};

use crate::jobs::{JobHandle, Jobs, Polled, Worker};

/// Default of the `AutosaveMinutes` setting.
pub const DEFAULT_AUTOSAVE_MINUTES: u32 = 5;

/// What is written to the recovery file's name between the project's base name and `.xma`.
pub const RECOVERY_SUFFIX: &str = ".recovery.xma";

/// The name of the recovery file of a project never saved, in the settings folder.
pub const UNTITLED_RECOVERY: &str = "untitled.recovery.xma";

/// The recovery file of a project: next to the project file, the base name followed by
/// `.recovery.xma` (`pig.xma` → `pig.recovery.xma`), so that relative video paths and the
/// search for missing videos work for it as for the project. A project never saved uses
/// `untitled.recovery.xma` in the settings folder (`config_dir`); `None` when there is none.
#[must_use]
pub fn recovery_path(project: Option<&Path>, config_dir: Option<&Path>) -> Option<PathBuf> {
    match project {
        Some(p) => {
            let name = p.file_name()?.to_string_lossy().into_owned();
            // QFileInfo::completeBaseName: the name up to the last dot.
            let base = name.rsplit_once('.').map_or(name.as_str(), |(b, _)| b);
            Some(p.with_file_name(format!("{base}{RECOVERY_SUFFIX}")))
        }
        None => config_dir.map(|d| d.join(UNTITLED_RECOVERY)),
    }
}

/// The recovery file of `project` if it exists and is newer than the project file (or the
/// project file is gone).
#[must_use]
pub fn newer_recovery(project: &Path) -> Option<PathBuf> {
    let r = recovery_path(Some(project), None)?;
    let rm = std::fs::metadata(&r).ok()?.modified().ok()?;
    match std::fs::metadata(project).and_then(|m| m.modified()) {
        Ok(pm) if pm >= rm => None,
        _ => Some(r),
    }
}

/// Removes a recovery file; a missing file is not an error.
pub fn remove_recovery(path: &Path) -> std::io::Result<()> {
    match std::fs::remove_file(path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
        _ => Ok(()),
    }
}

/// Ported from ui/MainWindow.cpp, `ensureExtension` as `saveProjectAs` calls it: `.xma` is
/// appended when the name ends in neither `.xma` nor `.zip` (ignoring case).
#[must_use]
pub fn ensure_extension(path: PathBuf) -> PathBuf {
    let lower = path.to_string_lossy().to_lowercase();
    if lower.ends_with(".xma") || lower.ends_with(".zip") {
        path
    } else {
        let mut s = path.into_os_string();
        s.push(".xma");
        PathBuf::from(s)
    }
}

/// Which save.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SaveKind {
    /// File > Save dataset.
    Save,
    /// File > Save dataset as: the document takes the new file name.
    SaveAs,
    /// File > Save subdataset as: these trials only; the document keeps its file name and
    /// its modified state.
    Subset(Vec<usize>),
}

/// What the save worker sends back.
pub struct SaveOutcome {
    pub path: PathBuf,
    pub kind: SaveKind,
    /// The document's revision when the copy was taken.
    pub revision: u64,
    /// The saved copy (with the file name and creation date the save set) and the report,
    /// or the error's text.
    pub result: Result<(Document, SaveReport), String>,
}

/// A save running on a worker thread.
pub struct SaveJob {
    worker: Worker<SaveOutcome>,
    pub path: PathBuf,
    pub revision: u64,
}

impl SaveJob {
    /// Starts writing `doc` (a copy of the session's document) to `path`. The progress job
    /// is "Save dataset as <file>" and disables the window, as in XMALab 3 (all three saves
    /// use that title).
    #[must_use]
    pub fn start(
        mut doc: Document,
        path: PathBuf,
        kind: SaveKind,
        recompute_when_saving: bool,
        jobs: &mut Jobs,
    ) -> SaveJob {
        let job = jobs.start(format!("Save dataset as {}", path.display()), false);
        let revision = doc.revision();
        let p = path.clone();
        let work = move |_: &JobHandle| {
            let options = SaveOptions {
                recompute_when_saving,
                time: SaveTime::now(),
                trials: match &kind {
                    SaveKind::Subset(t) => Some(t.clone()),
                    _ => None,
                },
            };
            let result = match kind {
                SaveKind::Save => doc.save(&options),
                SaveKind::SaveAs | SaveKind::Subset(_) => doc.save_as(&p, &options),
            };
            SaveOutcome {
                path: p,
                kind,
                revision,
                result: result.map(|r| (doc, r)).map_err(|e| e.to_string()),
            }
        };
        let worker = match Worker::spawn("xma-save", &job, work) {
            Ok(w) => w,
            // The closure is gone: poll reports the thread's failure.
            Err(e) => Worker::ready(
                job,
                SaveOutcome {
                    path: path.clone(),
                    kind: SaveKind::Save,
                    revision,
                    result: Err(format!("Cannot start the saving thread: {e}")),
                },
            ),
        };
        SaveJob {
            worker,
            path,
            revision,
        }
    }

    /// The finished save, if the worker is done. Ends the progress job.
    pub fn poll(&mut self, jobs: &mut Jobs) -> Option<SaveOutcome> {
        match self.worker.poll(jobs) {
            Polled::Done(r) => Some(r),
            Polled::Running => None,
            Polled::Stopped => Some(SaveOutcome {
                path: self.path.clone(),
                kind: SaveKind::Save,
                revision: self.revision,
                result: Err("The saving thread stopped unexpectedly.".to_owned()),
            }),
        }
    }

    /// Waits for the worker (tests).
    pub fn wait(&mut self, jobs: &mut Jobs) -> SaveOutcome {
        loop {
            if let Some(r) = self.poll(jobs) {
                return r;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

/// What the autosave worker sends back.
pub struct AutosaveOutcome {
    /// The recovery file written.
    pub path: PathBuf,
    /// The revision written.
    pub revision: u64,
    pub result: Result<(), String>,
}

/// The autosave timer and the write in progress.
pub struct Autosave {
    /// When the interval is counted from: opening, the last save or autosave.
    pub since: Instant,
    /// The revision last written to the recovery file.
    pub written: Option<u64>,
    running: Option<Worker<AutosaveOutcome>>,
    /// The last failure, so that a failing autosave writes one Console line, not one per
    /// interval.
    pub last_error: Option<String>,
}

impl Default for Autosave {
    fn default() -> Self {
        Autosave {
            since: Instant::now(),
            written: None,
            running: None,
            last_error: None,
        }
    }
}

impl Autosave {
    /// Starts the interval again (a project opened or saved).
    pub fn reset(&mut self) {
        self.since = Instant::now();
        self.written = None;
    }

    #[must_use]
    pub fn is_running(&self) -> bool {
        self.running.is_some()
    }

    /// Time left before the next autosave is due, `None` when none is due (off, or nothing
    /// new to write).
    #[must_use]
    pub fn due_in(&self, minutes: u32, modified: bool, revision: u64) -> Option<Duration> {
        if minutes == 0 || !modified || self.written == Some(revision) {
            return None;
        }
        let interval = Duration::from_secs(u64::from(minutes) * 60);
        Some(interval.saturating_sub(self.since.elapsed()))
    }

    /// Starts writing `doc` (a copy) to `path` on a worker thread.
    pub fn start(&mut self, mut doc: Document, path: PathBuf) {
        let revision = doc.revision();
        self.since = Instant::now();
        let work = move || {
            let options = SaveOptions {
                recompute_when_saving: false,
                time: SaveTime::now(),
                trials: None,
            };
            let result = (|| {
                if let Some(dir) = path.parent() {
                    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
                }
                doc.save_as(&path, &options)
                    .map(|_| ())
                    .map_err(|e| e.to_string())
            })();
            AutosaveOutcome {
                path,
                revision,
                result,
            }
        };
        match Worker::spawn_untracked("xma-autosave", work) {
            Ok(w) => self.running = Some(w),
            Err(e) => self.last_error = Some(e.to_string()),
        }
    }

    /// The finished autosave, if any.
    pub fn poll(&mut self) -> Option<AutosaveOutcome> {
        match self.running.as_ref()?.try_recv() {
            Polled::Running => None,
            // A thread that stopped without a result reports nothing.
            Polled::Done(r) => {
                self.running = None;
                Some(r)
            }
            Polled::Stopped => {
                self.running = None;
                None
            }
        }
    }

    /// Waits for the running autosave (tests).
    pub fn wait(&mut self) -> Option<AutosaveOutcome> {
        while self.running.is_some() {
            if let Some(r) = self.poll() {
                return Some(r);
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        None
    }
}

/// What the file dialogs answer. The application asks [`FileChooser::save_file`] instead
/// of calling the operating system's dialog itself, so that tests can answer.
#[derive(Debug, Default)]
pub enum FileChooser {
    /// The operating system's file dialog (`rfd`).
    #[default]
    Native,
    /// Answers given in advance, in order (tests); `None` is Cancel. When empty, Cancel.
    Scripted(std::collections::VecDeque<Option<PathBuf>>),
    /// Answers given in advance, in order, each a list of chosen paths (empty: Cancel;
    /// several: `getOpenFileNames`), with every question recorded (tests check the titles
    /// and filters). When no answer is left, Cancel.
    Script(ScriptedFiles),
}

/// The answers and the record of [`FileChooser::Script`].
#[derive(Debug, Default)]
pub struct ScriptedFiles {
    /// The answers still to give, in order.
    pub answers: std::collections::VecDeque<Vec<PathBuf>>,
    /// Every question asked, in order.
    pub asked: Vec<FileQuestion>,
}

/// Which file dialog was asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileQuestionKind {
    /// `getOpenFileName`.
    OpenFile,
    /// `getOpenFileNames`.
    OpenFiles,
    /// `getSaveFileName`.
    SaveFile,
    /// `getExistingDirectory`.
    PickFolder,
}

/// One file dialog asked for, as [`FileChooser::Script`] records it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileQuestion {
    pub kind: FileQuestionKind,
    pub title: String,
    /// The filters: name and extensions.
    pub filters: Vec<(String, Vec<String>)>,
    /// Where the dialog starts.
    pub start: Option<PathBuf>,
}

impl ScriptedFiles {
    fn ask(
        &mut self,
        kind: FileQuestionKind,
        title: &str,
        start: Option<&Path>,
        filters: &[(&str, &[&str])],
    ) -> Vec<PathBuf> {
        self.asked.push(FileQuestion {
            kind,
            title: title.to_owned(),
            filters: filters
                .iter()
                .map(|(n, e)| ((*n).to_owned(), e.iter().map(|x| (*x).to_owned()).collect()))
                .collect(),
            start: start.map(Path::to_path_buf),
        });
        self.answers.pop_front().unwrap_or_default()
    }
}

impl FileChooser {
    /// A [`FileChooser::Script`] with these answers (each a list of chosen paths; empty is
    /// Cancel).
    #[must_use]
    pub fn script(answers: Vec<Vec<PathBuf>>) -> Self {
        FileChooser::Script(ScriptedFiles {
            answers: answers.into(),
            asked: Vec::new(),
        })
    }

    /// The questions a [`FileChooser::Script`] was asked (empty for the others).
    #[must_use]
    pub fn asked(&self) -> &[FileQuestion] {
        match self {
            FileChooser::Script(s) => &s.asked,
            _ => &[],
        }
    }

    /// `QFileDialog::getSaveFileName(parent, title, start, "Dataset (*.xma *.zip)")`.
    pub fn save_file(&mut self, title: &str, start: Option<&Path>) -> Option<PathBuf> {
        match self {
            FileChooser::Native => {
                let mut dlg = rfd::FileDialog::new()
                    .set_title(title)
                    .add_filter("Dataset", &["xma", "zip"]);
                if let Some(s) = start {
                    if s.is_dir() {
                        dlg = dlg.set_directory(s);
                    } else {
                        if let Some(d) = s.parent() {
                            dlg = dlg.set_directory(d);
                        }
                        if let Some(n) = s.file_name() {
                            dlg = dlg.set_file_name(n.to_string_lossy());
                        }
                    }
                }
                dlg.save_file()
            }
            FileChooser::Scripted(answers) => answers.pop_front().flatten(),
            FileChooser::Script(s) => s
                .ask(
                    FileQuestionKind::SaveFile,
                    title,
                    start,
                    &[("Dataset", &["xma", "zip"])],
                )
                .into_iter()
                .next(),
        }
    }

    /// `QFileDialog::getOpenFileName(parent, title, start, "<filter> (*.ext ...)")`.
    /// Scripted: the next answer.
    pub fn open_file(
        &mut self,
        title: &str,
        start: Option<&Path>,
        filter: (&str, &[&str]),
    ) -> Option<PathBuf> {
        match self {
            FileChooser::Native => native_dialog(title, start, filter).pick_file(),
            FileChooser::Scripted(answers) => answers.pop_front().flatten(),
            FileChooser::Script(s) => s
                .ask(FileQuestionKind::OpenFile, title, start, &[filter])
                .into_iter()
                .next(),
        }
    }

    /// `QFileDialog::getOpenFileNames`: several files; empty for Cancel. Scripted: the next
    /// answer, as one file.
    pub fn open_files(
        &mut self,
        title: &str,
        start: Option<&Path>,
        filter: (&str, &[&str]),
    ) -> Vec<PathBuf> {
        match self {
            FileChooser::Native => native_dialog(title, start, filter)
                .pick_files()
                .unwrap_or_default(),
            FileChooser::Scripted(answers) => answers.pop_front().flatten().into_iter().collect(),
            FileChooser::Script(s) => s.ask(FileQuestionKind::OpenFiles, title, start, &[filter]),
        }
    }

    /// `QFileDialog::getSaveFileName` with a filter other than the dataset one.
    /// Scripted: the next answer.
    pub fn save_file_filtered(
        &mut self,
        title: &str,
        start: Option<&Path>,
        filter: (&str, &[&str]),
    ) -> Option<PathBuf> {
        match self {
            FileChooser::Native => native_dialog(title, start, filter).save_file(),
            FileChooser::Scripted(answers) => answers.pop_front().flatten(),
            FileChooser::Script(s) => s
                .ask(FileQuestionKind::SaveFile, title, start, &[filter])
                .into_iter()
                .next(),
        }
    }

    /// `QFileDialog::getOpenFileName` with several filters (none: any file).
    /// Scripted: the next answer.
    pub fn open_file_filters(
        &mut self,
        title: &str,
        start: Option<&Path>,
        filters: &[(&str, &[&str])],
    ) -> Option<PathBuf> {
        match self {
            FileChooser::Native => {
                let mut dlg = start_at(rfd::FileDialog::new().set_title(title), start);
                for (name, exts) in filters {
                    dlg = dlg.add_filter(*name, exts);
                }
                dlg.pick_file()
            }
            FileChooser::Scripted(answers) => answers.pop_front().flatten(),
            FileChooser::Script(s) => s
                .ask(FileQuestionKind::OpenFile, title, start, filters)
                .into_iter()
                .next(),
        }
    }

    /// `QFileDialog::getExistingDirectory`. Scripted: the next answer.
    pub fn pick_folder(&mut self, title: &str, start: Option<&Path>) -> Option<PathBuf> {
        match self {
            FileChooser::Native => {
                start_at(rfd::FileDialog::new().set_title(title), start).pick_folder()
            }
            FileChooser::Scripted(answers) => answers.pop_front().flatten(),
            FileChooser::Script(s) => s
                .ask(FileQuestionKind::PickFolder, title, start, &[])
                .into_iter()
                .next(),
        }
    }
}

/// The operating system's file dialog with a title, a start (a folder, or a file whose
/// folder and name are used) and one filter.
fn native_dialog(title: &str, start: Option<&Path>, filter: (&str, &[&str])) -> rfd::FileDialog {
    let dlg = rfd::FileDialog::new()
        .set_title(title)
        .add_filter(filter.0, filter.1);
    start_at(dlg, start)
}

/// The dialog's start: a folder, or a file whose folder and name are used.
fn start_at(mut dlg: rfd::FileDialog, start: Option<&Path>) -> rfd::FileDialog {
    if let Some(s) = start {
        if s.is_dir() {
            dlg = dlg.set_directory(s);
        } else {
            if let Some(d) = s.parent().filter(|d| d.is_dir()) {
                dlg = dlg.set_directory(d);
            }
            if let Some(n) = s.file_name() {
                dlg = dlg.set_file_name(n.to_string_lossy());
            }
        }
    }
    dlg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovery_file_names() {
        assert_eq!(
            recovery_path(Some(Path::new("/a/pig.data.xma")), None),
            Some(PathBuf::from("/a/pig.data.recovery.xma"))
        );
        assert_eq!(
            recovery_path(None, Some(Path::new("/cfg"))),
            Some(PathBuf::from("/cfg/untitled.recovery.xma"))
        );
        assert_eq!(recovery_path(None, None), None);
    }

    #[test]
    fn default_extension_like_cpp() {
        assert_eq!(ensure_extension("a/b".into()), PathBuf::from("a/b.xma"));
        assert_eq!(ensure_extension("a/b.XMA".into()), PathBuf::from("a/b.XMA"));
        assert_eq!(ensure_extension("a/b.zip".into()), PathBuf::from("a/b.zip"));
        assert_eq!(
            ensure_extension("a/b.txt".into()),
            PathBuf::from("a/b.txt.xma")
        );
    }

    #[test]
    fn autosave_due_only_when_modified_and_new() {
        let a = Autosave::default();
        assert_eq!(a.due_in(0, true, 3), None);
        assert_eq!(a.due_in(5, false, 3), None);
        assert!(a.due_in(5, true, 3).unwrap() > Duration::from_secs(200));
        let b = Autosave {
            written: Some(3),
            ..Autosave::default()
        };
        assert_eq!(b.due_in(5, true, 3), None);
    }

    #[test]
    fn scripted_chooser_answers_in_order() {
        let mut c = FileChooser::Scripted([Some(PathBuf::from("x")), None].into());
        assert_eq!(c.save_file("t", None), Some(PathBuf::from("x")));
        assert_eq!(c.save_file("t", None), None);
        assert_eq!(c.save_file("t", None), None);
    }
}
