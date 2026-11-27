//! Creating projects and trials: File > New dataset (C47 to C52), New dataset (external
//! calibration) (C53), New trial without calibration (C54), the workspace bar's "+" and the
//! start screen's "Add trial" and "Add Default Trial" (C55 to C59), the Trial dialog's
//! "Change Trial Data" (C61) and File > Open Dataset with different calibration (C62).
//!
//! Ported from ui/MainWindow.cpp (`newProject`, `newProjectFinished`,
//! `newProjectExternalCalibration`, `on_actionNew_trial_without_calibration_triggered`,
//! `newTrial`, `on_pushButtonDefaultTrial_clicked`, `loadProjectWithDifferentCalibration`),
//! ui/NewTrialDialog.cpp (`createTrial`, `on_pushButton_Default_clicked`) and
//! ui/TrialDialog.cpp (`on_pushButton_ChangeTrialData_clicked`). The dialogs are
//! `dialogs::new_project` and `dialogs::new_trial`; the engine's side is
//! `xma_engine::create`.
//!
//! Differences from XMALab 3 (proposed ledger rows in the appnew report):
//!
//! - The open dataset is replaced only when the new one has been created. XMALab 3 closes
//!   it before its dialog, so Cancel leaves no dataset (C47) or an empty one taken for an
//!   open dataset (C54, CD31), and a failed creation leaves a half-made one (CD8).
//! - Before replacing the open dataset each action asks "Save changes?" when it has unsaved
//!   changes, as Close and Open do (ledger U55); New dataset (external calibration), which
//!   asks nothing in XMALab 3, does the same (CD30).
//! - A creation refused by the engine (`CreateError`) is shown in the error dialog from the
//!   interface thread; the open dataset stays as it was, and the New dataset or New trial
//!   dialog comes back with what was entered (XMALab 3 shows the error from the worker
//!   thread, goes on with a partial dataset and drops the dialog, CD8).
//! - Open Dataset with different calibration stops when the first file dialog is cancelled
//!   (XMALab 3 asks for the second file anyway).

use std::collections::BTreeMap;

use xma_engine::create::{CreateError, NewProjectSpec, NewTrialSpec};
use xma_engine::{Document, Edit, OpenOptions};

use super::XmaApp;
use crate::actions::Action;
use crate::dialogs::file_names::DATASET_FILES;
use crate::dialogs::new_project::NewProjectDialog;
use crate::dialogs::new_trial::{NewTrialAnswer, NewTrialDialog};
use crate::dialogs::{Dialog, IntPurpose};
use crate::editing::EditRequest;
use crate::jobs::{Polled, Worker};
use crate::log::LogLevel;
use crate::state::Workspace;

/// The progress text of the creation (`MainWindow::newProject`).
pub const CREATE_TEXT: &str = "Create new dataset";
/// The question before a trial without calibration.
pub const NO_CALIBRATION_TEXT: &str = "Are you sure you want to create a trial without calibration? There will be no possibility to add a calibration afterwards.";
/// The first file dialog of Open Dataset with different calibration.
pub const CALIBRATION_FILE_TITLE: &str = "Select dataset for the calibration";
/// The second file dialog of Open Dataset with different calibration.
pub const TRIALS_FILE_TITLE: &str = "Select dataset for the trials";

/// A creation running on a worker thread.
pub struct CreateJob {
    /// The document, and XMALab's error to show once it is open (a Portal trial's frame
    /// rate that could not be set).
    worker: Worker<Result<(Document, Option<String>), CreateError>>,
    /// The dialog shown again when the creation fails.
    retry: Option<Dialog>,
}

impl XmaApp {
    /// The New actions and the trial buttons, from `dispatch`. Returns false for other
    /// actions.
    pub(super) fn dispatch_create(&mut self, a: &Action, confirmed: bool) -> bool {
        match a {
            Action::NewProject => {
                if !self.unsaved_guard(a, confirmed) {
                    self.dialogs.open(Dialog::NewProject(Box::default()));
                }
            }
            Action::NewDatasetExternalCalibration => {
                if !self.unsaved_guard(a, confirmed) {
                    // QInputDialog::getInt(this, "Set number of Cameras", "Cameras:", 2, 1, 20).
                    self.dialogs.open(Dialog::GetInt {
                        title: "Set number of Cameras".into(),
                        label: "Cameras:".into(),
                        value: 2,
                        min: 1,
                        max: 20,
                        purpose: IntPurpose::ExternalCameras,
                    });
                }
            }
            Action::NewTrialWithoutCalibration => {
                if !self.unsaved_guard(a, confirmed) {
                    // Its Yes comes back as Confirmed(Confirmed(NewTrialWithoutCalibration)),
                    // which `dispatch` passes here as `Confirmed(..)` (below).
                    self.confirm(
                        NO_CALIBRATION_TEXT.into(),
                        Action::Confirmed(Box::new(Action::NewTrialWithoutCalibration)),
                    );
                }
            }
            Action::Confirmed(inner) if **inner == Action::NewTrialWithoutCalibration => {
                self.dialogs.open(Dialog::NewTrial(Box::new(
                    NewTrialDialog::without_calibration(),
                )));
            }
            Action::LoadProjectWithDifferentCalibration => {
                if !self.unsaved_guard(a, confirmed) {
                    self.open_with_different_calibration();
                }
            }
            Action::AddTrial => {
                if let Some(doc) = self.session.doc.as_ref() {
                    self.dialogs
                        .open(Dialog::NewTrial(Box::new(NewTrialDialog::new(doc))));
                }
            }
            Action::AddDefaultTrial => self.default_trial(),
            _ => return false,
        }
        true
    }

    /// OK in the camera-count prompt (C53): the dataset is made at once (as in XMALab 3).
    pub(super) fn create_external(&mut self, n_cameras: i32) {
        let n = usize::try_from(n_cameras.clamp(1, 20)).unwrap_or(1);
        match Document::new_external(n, &self.create_options()) {
            Ok(doc) => self.created(doc),
            Err(e) => self.creation_failed(&e, None),
        }
    }

    /// OK in the New dataset dialog (C52): the dataset is made on a worker thread while the
    /// Progress panel shows "Create new dataset".
    pub(super) fn create_project(&mut self, spec: NewProjectSpec, dialog: Box<NewProjectDialog>) {
        let options = self.create_options();
        // A dialog filled from an XMA Portal project (C63) creates as C52 does, then keeps
        // the Portal metadata and camera numbers (`crate::app::portal`).
        if let Some(import) = dialog.portal.clone() {
            self.start_create(Dialog::NewProject(dialog), move || {
                Document::new_project_from_portal(&spec, &import, &options)
            });
            return;
        }
        self.start_create(Dialog::NewProject(dialog), move || {
            Document::new_project(&spec, &options)
        });
    }

    /// The New trial dialog's answers.
    pub(super) fn new_trial_answer(&mut self, a: NewTrialAnswer) {
        match a {
            NewTrialAnswer::NewTrial(spec) => {
                let before = self.trial_count();
                self.apply_new_trial(spec);
                self.trial_added(before);
            }
            NewTrialAnswer::Change { trial, spec } => {
                let before = self.session.doc.as_ref().map(Document::changes);
                self.apply_change_trial_data(trial, spec);
                if self.session.doc.as_ref().map(Document::changes) != before {
                    self.trial_data_changed(trial);
                }
            }
            NewTrialAnswer::NoCalibration { spec, dialog } => {
                let options = self.create_options();
                if let Some(metadata) = dialog.portal_metadata.clone() {
                    self.start_create_with_notice(Dialog::NewTrial(dialog), move || {
                        Document::new_without_calibration_from_portal(&spec, &metadata, &options)
                    });
                    return;
                }
                // XMALab 3 makes this one on the interface thread; reading the videos'
                // sizes can take a while, so it runs as New dataset does.
                self.start_create(Dialog::NewTrial(dialog), move || {
                    Document::new_without_calibration(&spec, &options)
                });
            }
            NewTrialAnswer::DefaultTrial => self.default_trial(),
            NewTrialAnswer::LoadPortal {
                dialog,
                file,
                delete,
            } => self.load_portal_trial(dialog, file, delete),
            NewTrialAnswer::PortalTrial { spec, metadata } => {
                let before = self.trial_count();
                self.apply_request(EditRequest::Apply(Edit::NewTrialFromPortal {
                    spec,
                    metadata,
                }));
                self.trial_added(before);
            }
            NewTrialAnswer::PortalChange {
                trial,
                spec,
                metadata,
            } => {
                let before = self.session.doc.as_ref().map(Document::changes);
                self.apply_request(EditRequest::Apply(Edit::ChangeTrialDataFromPortal {
                    trial,
                    spec,
                    metadata,
                }));
                if self.session.doc.as_ref().map(Document::changes) != before {
                    self.trial_data_changed(trial);
                }
            }
        }
    }

    /// The start screen's and the dialog's "Add Default Trial" (C59).
    fn default_trial(&mut self) {
        if self.session.doc.is_none() {
            return;
        }
        let before = self.trial_count();
        self.apply_default_trial();
        self.trial_added(before);
    }

    /// `Edit::NewTrial(spec)` (C55 to C58: `NewTrialDialog::createTrial`).
    pub fn apply_new_trial(&mut self, spec: NewTrialSpec) {
        self.apply_request(EditRequest::Apply(Edit::NewTrial(spec)));
    }

    /// `Edit::AddDefaultTrial` (C59: `NewTrialDialog::on_pushButton_Default_clicked`).
    pub fn apply_default_trial(&mut self) {
        self.apply_request(EditRequest::Apply(Edit::AddDefaultTrial));
    }

    /// `Edit::ChangeTrialData { trial, spec }` (C61: `Trial::changeTrialData`).
    pub fn apply_change_trial_data(&mut self, trial: usize, spec: NewTrialSpec) {
        self.apply_request(EditRequest::Apply(Edit::ChangeTrialData { trial, spec }));
    }

    pub(super) fn trial_count(&self) -> usize {
        self.session
            .doc
            .as_ref()
            .map_or(0, |d| d.project().trials.len())
    }

    /// After a trial was added: it becomes the active trial, at its frame, in Marker
    /// tracking (`createTrial`, `on_pushButton_Default_clicked`, `MainWindow::newTrial`).
    fn trial_added(&mut self, before: usize) {
        self.sync_document();
        let n = self.trial_count();
        if n <= before {
            return;
        }
        let t = n - 1;
        self.changes |= self.state.change_active_trial(Some(t), true);
        self.changes |= self.state.change_workspace(Workspace::MarkerTracking, true);
        let frame = self.state.trials.get(t).map_or(0, |v| v.active_frame);
        self.changes |= self.state.change_active_frame_trial(Some(frame), true);
    }

    /// After "Change Trial Data": the trial's files are new, so videos found elsewhere
    /// for the old files are forgotten; Marker tracking at the trial's frame
    /// (`WorkspaceNavigationFrame`, `TRIALDIALOGCHANGE`).
    fn trial_data_changed(&mut self, trial: usize) {
        if let Some(name) = self.facts.trial_names.get(trial) {
            let name = name.clone();
            self.session.video_stash.retain(|(n, _), _| *n != name);
        }
        self.trial_files_replaced(trial);
    }

    /// [`XmaApp::trial_data_changed`] without forgetting the stashed videos (after the
    /// relink of "Denoise Images", `app/images.rs`).
    pub(super) fn trial_files_replaced(&mut self, trial: usize) {
        self.session.videos.retain(|(t, _), _| *t != trial);
        self.sync_document();
        self.changes |= self.state.change_workspace(Workspace::MarkerTracking, true);
        if self.state.active_trial == Some(trial) {
            let frame = self.state.trials.get(trial).map_or(0, |v| v.active_frame);
            self.changes |= self.state.change_active_frame_trial(Some(frame), true);
        }
    }

    /// The two file dialogs, then the load on the worker thread used for opening (C62).
    fn open_with_different_calibration(&mut self) {
        let start = self.last_dir();
        let Some(calibration) =
            self.file_chooser
                .open_file(CALIBRATION_FILE_TITLE, start.as_deref(), DATASET_FILES)
        else {
            return;
        };
        let Some(trials) =
            self.file_chooser
                .open_file(TRIALS_FILE_TITLE, start.as_deref(), DATASET_FILES)
        else {
            return;
        };
        if self.session.doc.is_some() {
            self.close_project();
        }
        self.begin_load(&trials);
        let disable = self.settings.disable_image_search;
        self.session
            .start_load_with_calibration(&trials, &calibration, &mut self.jobs, disable);
    }

    fn create_options(&self) -> OpenOptions {
        OpenOptions {
            base_settings: self.settings.algorithm_base(),
            ..OpenOptions::default()
        }
    }

    fn start_create(
        &mut self,
        retry: Dialog,
        work: impl FnOnce() -> Result<Document, CreateError> + Send + 'static,
    ) {
        self.start_create_with_notice(retry, move || work().map(|d| (d, None)));
    }

    /// [`XmaApp::start_create`] for a creation that may end with an error to show once
    /// the dataset is open.
    fn start_create_with_notice(
        &mut self,
        retry: Dialog,
        work: impl FnOnce() -> Result<(Document, Option<String>), CreateError> + Send + 'static,
    ) {
        let job = self.jobs.start(CREATE_TEXT, false);
        let worker = match Worker::spawn("xma-create", &job, move |_| work()) {
            Ok(w) => w,
            Err(e) => {
                self.jobs.finish(job.id());
                let msg = format!("Cannot start the creating thread: {e}");
                self.log.error(msg.clone());
                self.dialogs.error(msg);
                return;
            }
        };
        self.create = Some(CreateJob {
            worker,
            retry: Some(retry),
        });
    }

    /// A creation is running.
    #[must_use]
    pub fn is_creating(&self) -> bool {
        self.create.is_some()
    }

    /// Collects a finished creation (once per frame).
    pub(super) fn poll_create(&mut self) {
        let Some(c) = &self.create else { return };
        let r = match c.worker.poll(&mut self.jobs) {
            Polled::Done(r) => r,
            Polled::Running => return,
            Polled::Stopped => Err(CreateError::Io(
                "The creating thread stopped unexpectedly.".into(),
            )),
        };
        let mut c = self.create.take().expect("a creation runs");
        match r {
            Ok((doc, notice)) => {
                self.created(doc);
                if let Some(n) = notice {
                    self.log.error(n.clone());
                    self.dialogs.error(n);
                }
            }
            Err(e) => self.creation_failed(&e, c.retry.take()),
        }
    }

    /// Waits for the running creation and collects it (tests).
    pub fn wait_for_create(&mut self) {
        while self.create.is_some() {
            self.poll_create();
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    /// A `CreateError`: the error dialog (and the Console); the open dataset is unchanged;
    /// the dialog comes back behind the error.
    fn creation_failed(&mut self, e: &CreateError, retry: Option<Dialog>) {
        let msg = e.to_string();
        self.log.error(msg.clone());
        self.dialogs.error(msg);
        if let Some(d) = retry {
            self.dialogs.open(d);
        }
    }

    /// The new document becomes the open project (`newProjectFinished`, `setupProjectUI`):
    /// the open one is closed, active camera 0, calibration frame 0, the Undistortion
    /// workspace when a camera has a grid, else Calibration (Marker tracking without
    /// calibration).
    fn created(&mut self, doc: Document) {
        if self.session.doc.is_some() {
            self.close_project();
        }
        self.session.base_settings = self.settings.algorithm_base();
        self.log.write(CREATE_TEXT, LogLevel::Output);
        self.state.loading = true;
        self.show_document(doc, BTreeMap::new(), Vec::new());
    }
}
