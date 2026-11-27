//! Edits that take long ("compute undistortion", "compute calibration", the recalibration of
//! all frames, "optimize camera setup"), run without blocking the window.
//!
//! XMALab 3 runs these as `ThreadedProcessing` jobs that write the project from their worker
//! threads while the window can draw it (`docs/cpp-calibration.md` section 9, CD34), with
//! the progress dialog's indeterminate bar and no cancel. Here the edit is applied to a copy
//! of the document on a worker thread; the Progress panel shows the edit's
//! text (`calibration_ui::progress_text`) and the window is disabled meanwhile (a job that cannot
//! be cancelled disables it, as the creation job does), so nothing else can change the
//! project. When the edit succeeds the copy, which holds the edit as one undo step, becomes
//! the open document and its outcome is shown as an immediate edit's is
//! (`XmaApp::calibration_outcome`); when it fails the document stays as it was and the error is shown as
//! any edit error is (`editing::report_error`: XMALab's error dialog for its sentences).
//! There is no cancel, as in XMALab.

use xma_engine::{Document, Edit, EditError, EditResult};

use super::XmaApp;
use crate::jobs::{Polled, Worker};

/// What follows a successful long edit (the workspace change or save that asked the
/// pending-changes question).
pub(crate) type AfterLongEdit = Box<dyn FnOnce(&mut XmaApp)>;

/// A long edit running on a worker thread.
pub struct LongEdit {
    worker: Worker<(Document, EditResult)>,
    /// The text the Progress panel shows.
    text: String,
    /// The edit, when the job is one (`None` for [`XmaApp::start_long_job`]).
    edit: Option<Edit>,
    then: Option<AfterLongEdit>,
}

impl std::fmt::Debug for LongEdit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LongEdit")
            .field("text", &self.text)
            .field("edit", &self.edit)
            .finish_non_exhaustive()
    }
}

impl XmaApp {
    /// Applies `edit` to a copy of the document on a worker thread while the Progress panel
    /// shows `text`; the window is disabled until it ends. On success the copy becomes the
    /// open document (one undo step); on an error the document is unchanged and the error
    /// is shown (XMALab's error dialog for a sentence of XMALab's). Refused, with a Console
    /// line, while another long edit, a tracking run or a refinement is open.
    pub fn start_long_edit(&mut self, edit: Edit, text: &str) {
        self.start_long(text, Some(edit.clone()), None, move |doc| doc.apply(&edit));
    }

    /// [`XmaApp::start_long_edit`] for any work on the copy of the document: `work` runs on
    /// the worker thread and must leave the copy as it should become (one undo step for
    /// one edit). Tests use it with edits that take a while or fail.
    pub fn start_long_job(
        &mut self,
        text: &str,
        work: impl FnOnce(&mut Document) -> EditResult + Send + 'static,
    ) {
        self.start_long(text, None, None, work);
    }

    pub(super) fn start_long_then(&mut self, edit: Edit, text: &str, then: AfterLongEdit) {
        self.start_long(text, Some(edit.clone()), Some(then), move |doc| {
            doc.apply(&edit)
        });
    }

    fn start_long(
        &mut self,
        text: &str,
        edit: Option<Edit>,
        then: Option<AfterLongEdit>,
        work: impl FnOnce(&mut Document) -> EditResult + Send + 'static,
    ) {
        if self.long_edit.is_some() || self.tracking.is_some() || self.refine.is_some() {
            self.log.error(format!(
                "\"{text}\" cannot start while another computation runs."
            ));
            return;
        }
        // The session's display state goes with the copy (the engine reads the active
        // camera and frame).
        self.state_to_session();
        let Some(doc) = &self.session.doc else {
            return;
        };
        let mut copy = doc.clone();
        let job = self.jobs.start(text, false);
        let spawned = Worker::spawn("xma-long-edit", &job, move |_| {
            let r = work(&mut copy);
            (copy, r)
        });
        let worker = match spawned {
            Ok(w) => w,
            Err(e) => {
                self.jobs.finish(job.id());
                let msg = format!("Cannot start the computing thread: {e}");
                self.log.error(msg.clone());
                self.dialogs.error(msg);
                return;
            }
        };
        self.state.disable_draw = true;
        self.long_edit = Some(LongEdit {
            worker,
            text: text.to_owned(),
            edit,
            then,
        });
    }

    /// A long edit is running.
    #[must_use]
    pub fn is_computing(&self) -> bool {
        self.long_edit.is_some()
    }

    /// The text of the running long edit, if any.
    #[must_use]
    pub fn long_edit_text(&self) -> Option<&str> {
        self.long_edit.as_ref().map(|l| l.text.as_str())
    }

    /// Collects a finished long edit (once per frame).
    pub(super) fn poll_long_edit(&mut self) {
        let Some(l) = &self.long_edit else { return };
        let (copy, result) = match l.worker.try_recv() {
            Polled::Done(r) => (Some(r.0), r.1),
            Polled::Running => return,
            Polled::Stopped => (
                None,
                Err(EditError::Invalid(
                    "The computing thread stopped unexpectedly.".into(),
                )),
            ),
        };
        let l = self.long_edit.take().expect("a long edit runs");
        self.long_edit_done(l, copy, result);
    }

    /// Forgets a running long edit (the project is closed or the program quits): its
    /// result is not used.
    pub(super) fn drop_long_edit(&mut self) {
        if let Some(l) = self.long_edit.take() {
            l.worker.finish(&mut self.jobs);
            self.state.disable_draw = false;
        }
    }

    /// Waits for the running long edit and collects it (tests).
    pub fn wait_for_long_edit(&mut self) {
        while self.long_edit.is_some() {
            self.poll_long_edit();
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    fn long_edit_done(&mut self, l: LongEdit, copy: Option<Document>, result: EditResult) {
        l.worker.finish(&mut self.jobs);
        self.state.disable_draw = false;
        match result {
            Ok(outcome) => {
                if let (Some(copy), Some(doc)) = (copy, self.session.doc.as_mut()) {
                    *doc = copy;
                }
                self.calibration_outcome(l.edit.as_ref(), outcome);
                self.calibration_edited();
                if let Some(then) = l.then {
                    then(self);
                }
            }
            Err(e) => self.report_edit_error(&e),
        }
    }
}
