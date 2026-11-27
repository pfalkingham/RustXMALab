//! XMA Portal import in the window: the New trial dialog's "Load XMA Portal file" (C60)
//! run as the job "Load trial <file>", the New dataset dialog filled from a Portal project
//! opened with File > Open (C63), and Project Metadata's "Add metadata from XMAPortal file"
//! (its flow is `dialogs::portal_metadata`).
//!
//! Ported from ui/NewTrialDialog.cpp (`on_pushButton_LoadXMA_clicked`, `LoadXMAFinished`),
//! ui/MainWindow.cpp (`loadProjectFinished`'s result 1, `newProjectFromXMALab`) and
//! ui/NewProjectDialog.cpp (`addCalibrationImage`, `addGridImage`,
//! `setCalibrationCubeCSV`, `setCalibrationCubeREF`, `setupBasedOnMissingParameters`). The
//! work is `xma_engine::portal`.
//!
//! Differences from XMALab 3 (ledger U151 to U155):
//!
//! - "Delete after import" deletes the file only when the import succeeded (U151: XMALab 3
//!   deletes it after an archive that was not valid too).
//! - The dataset made from a Portal project has no file, so Save asks where to save (U152:
//!   XMALab 3 takes the Portal file as the dataset's file, so Save overwrites the download).
//! - The dialog is filled on the interface thread after the job (XMALab 3 fills its widgets
//!   from the worker thread); the open dataset is kept until a new one exists, as for New
//!   dataset (ledger U55 and the `app/create.rs` notes).

use std::path::PathBuf;

use xma_engine::portal::{Destination, ProjectImport, TrialImport, import_trial};

use super::XmaApp;
use crate::dialogs::Dialog;
use crate::dialogs::new_project::{CameraBox, NewProjectDialog};
use crate::dialogs::new_trial::{NewTrialDialog, NewTrialMode};
use crate::jobs::{Polled, Worker};
use crate::session::PortalProjectResult;

/// "Load XMA Portal file" running on a worker thread.
pub struct PortalTrialJob {
    worker: Worker<Result<TrialImport, String>>,
    dialog: Box<NewTrialDialog>,
    file: PathBuf,
    delete: bool,
}

/// The progress text (`on_pushButton_LoadXMA_clicked`: "Load trial " + the file).
#[must_use]
pub fn load_trial_text(file: &std::path::Path) -> String {
    format!("Load trial {}", file.display())
}

/// The New dataset dialog filled from a Portal project (`loadXMALabProject`, then
/// `setupBasedOnMissingParameters`): the boxes grow to the cameras found (two to start
/// with), each camera's calibration images (sorted, `QStringList::sort`) and its grid, the
/// cube's two files when found; then a camera without a grid becomes a light camera, and
/// with neither cube file Checkerboard is chosen.
#[must_use]
pub fn project_dialog(import: ProjectImport) -> NewProjectDialog {
    let mut d = NewProjectDialog::new();
    for (i, c) in import.cameras.iter().enumerate() {
        while d.cameras.len() <= i {
            d.plus();
        }
        let b: &mut CameraBox = &mut d.cameras[i];
        if !c.images.is_empty() {
            b.set_images(c.images.clone());
        }
        if let Some(g) = &c.grid {
            b.grid = Some(g.clone());
        }
    }
    d.frame_specification
        .clone_from(&import.frame_specification);
    d.references.clone_from(&import.references);
    for b in &mut d.cameras {
        if b.grid.is_none() {
            b.light = true;
        }
    }
    if d.frame_specification.is_none() && d.references.is_none() {
        // `radioButtonCheckerboard->click()`.
        d.checkerboard = true;
    }
    d.portal = Some(import);
    d
}

impl XmaApp {
    /// Starts "Load XMA Portal file" for `dialog` (`on_pushButton_LoadXMA_clicked` after its
    /// questions): the extraction on a worker thread with the job "Load trial <file>".
    pub(super) fn load_portal_trial(
        &mut self,
        dialog: Box<NewTrialDialog>,
        file: PathBuf,
        delete: bool,
    ) {
        let workspace = self.settings.portal_workspace();
        // The project file decides the folder only for a trial of the open dataset; "New
        // trial without calibration" makes a new one (XMALab 3 has closed the old one).
        let project_file = match dialog.mode {
            NewTrialMode::NoCalibration => None,
            _ => self
                .session
                .doc
                .as_ref()
                .and_then(|d| d.path().map(std::path::Path::to_path_buf)),
        };
        let job = self.jobs.start(load_trial_text(&file), false);
        let f = file.clone();
        let spawned = Worker::spawn("xma-portal", &job, move |wake| {
            let dest = Destination {
                workspace: workspace.as_deref(),
                project_file: project_file.as_deref(),
                portal_file: &f,
            };
            import_trial(&f, &dest, xma_model::CppCompat::default(), &mut |p| {
                wake.set_progress(p.min(0.99));
                true
            })
            .map_err(|e| e.to_string())
        });
        let worker = match spawned {
            Ok(w) => w,
            Err(e) => {
                self.jobs.finish(job.id());
                let msg = format!("Cannot start the import thread: {e}");
                self.log.error(msg.clone());
                self.dialogs.error(msg);
                self.dialogs.open(Dialog::NewTrial(dialog));
                return;
            }
        };
        self.portal_trial = Some(PortalTrialJob {
            worker,
            dialog,
            file,
            delete,
        });
    }

    /// "Load XMA Portal file" is running.
    #[must_use]
    pub fn is_loading_portal_trial(&self) -> bool {
        self.portal_trial.is_some()
    }

    /// Collects a finished "Load XMA Portal file" (once per frame): the dialog comes back
    /// filled (`LoadXMAFinished`), or with XMALab's error over it; the file is deleted when
    /// asked and the import succeeded (ledger U151).
    pub(super) fn poll_portal_trial(&mut self) {
        let Some(p) = &self.portal_trial else { return };
        let r = match p.worker.poll(&mut self.jobs) {
            Polled::Done(r) => r,
            Polled::Running => return,
            Polled::Stopped => Err("The import thread stopped unexpectedly.".to_owned()),
        };
        let Some(PortalTrialJob {
            mut dialog,
            file,
            delete,
            ..
        }) = self.portal_trial.take()
        else {
            return;
        };
        match r {
            Ok(import) => {
                let ids: Vec<i32> = match (dialog.mode, self.session.doc.as_ref()) {
                    (NewTrialMode::NoCalibration, _) | (_, None) => Vec::new(),
                    (_, Some(d)) => d.project().cameras.iter().map(|c| c.portal_id).collect(),
                };
                for m in &import.messages {
                    self.log.write(m.clone(), crate::log::LogLevel::Output);
                }
                for m in dialog.fill_from_portal(&import, &ids) {
                    self.log.error(m);
                }
                if delete && let Err(e) = std::fs::remove_file(&file) {
                    self.log.error(format!("{}: {e}", file.display()));
                }
            }
            Err(e) => {
                self.log.error(e.clone());
                self.dialogs.error(e);
            }
        }
        self.dialogs.open(Dialog::NewTrial(dialog));
    }

    /// File > Open found an XMA Portal project (`loadProjectFinished`, result 1): the New
    /// dataset dialog filled from it (`newProjectFromXMALab`), or the error.
    pub(super) fn portal_project_loaded(&mut self, r: PortalProjectResult) {
        self.state.loading = false;
        match r {
            Ok(import) => {
                self.log.write(
                    format!("XMA Portal project {}", import.source.display()),
                    crate::log::LogLevel::Output,
                );
                self.dialogs
                    .open(Dialog::NewProject(Box::new(project_dialog(import))));
            }
            Err(e) => {
                let msg = format!("Could not open {e}");
                self.log.error(msg.clone());
                self.dialogs.error(msg);
            }
        }
    }
}
