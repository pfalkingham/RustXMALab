//! File > Undistort sequence: the dialog (`dialogs/undistort_sequence.rs`) and the run.
//!
//! The queue runs on a worker thread (`xma_engine::undistort_sequence::UndistortJob`) as a
//! cancelable job "Undistort Sequence" in the Progress panel, which shows the item and frame
//! it is at; the project is not changed. When it ends, one Console line per item says what was
//! written, and the frames skipped (another size, unreadable, not written) and the items
//! that could not be processed are listed in a message box (XMALab skips them silently,
//! ledger U122). Cancel stops before the next frame; every file written is complete. XMALab 3
//! runs the queue on the interface thread with no progress and no cancel (ledger U121).

use xma_engine::undistort_sequence::{UndistortItem, UndistortJob, UndistortReport};

use super::XmaApp;
use super::images::{MAX_LINES, problems_text};
use crate::dialogs::Dialog;
use crate::dialogs::undistort_sequence::{TITLE, UndistortSequenceDialog};
use crate::jobs::{Polled, Worker};

/// A run of the queue on a worker thread.
#[derive(Debug)]
pub struct UndistortRun {
    worker: Worker<UndistortReport>,
    items: Vec<UndistortItem>,
}

impl XmaApp {
    /// File > Undistort sequence: opens the dialog.
    ///
    /// Ported from ui/MainWindow.cpp, MainWindow::on_actionUndistort_sequence_triggered.
    pub(super) fn open_undistort_sequence(&mut self) {
        let Some(doc) = &self.session.doc else { return };
        let last = self.last_dir();
        self.dialogs.open(Dialog::UndistortSequence(Box::new(
            UndistortSequenceDialog::new(doc, last),
        )));
    }

    /// Starts undistorting `items` on a worker thread. Refused, with a Console line, while
    /// another run is going on.
    pub fn start_undistort(&mut self, items: Vec<UndistortItem>) {
        if self.undistort.is_some() {
            self.log
                .error("Undistort sequence cannot start while another one runs.".to_owned());
            return;
        }
        let Some(doc) = &self.session.doc else { return };
        let job = match UndistortJob::new(doc, items.clone()) {
            Ok(j) => j,
            Err(e) => {
                self.log.error(e.clone());
                self.dialogs.error(e);
                return;
            }
        };
        let handle = self.jobs.start(TITLE, true);
        let spawned = Worker::spawn("xma-undistort", &handle, move |worker| {
            job.run(
                &mut |p| {
                    let done = if p.frames == 0 {
                        0.0
                    } else {
                        p.frame as f32 / p.frames as f32
                    };
                    worker.set_progress((p.item as f32 + done) / p.items.max(1) as f32);
                    if p.frames > 0 {
                        worker.set_message(format!(
                            "{TITLE}: item {} of {}, frame {} of {}",
                            p.item + 1,
                            p.items,
                            (p.frame + 1).min(p.frames),
                            p.frames
                        ));
                    }
                },
                &|| worker.is_canceled(),
            )
        });
        let worker = match spawned {
            Ok(w) => w,
            Err(e) => {
                self.jobs.finish(handle.id());
                let msg = format!("Cannot start the undistortion thread: {e}");
                self.log.error(msg.clone());
                self.dialogs.error(msg);
                return;
            }
        };
        self.undistort = Some(UndistortRun { worker, items });
    }

    /// An undistortion run is going on.
    #[must_use]
    pub fn is_undistorting(&self) -> bool {
        self.undistort.is_some()
    }

    /// Collects a finished run (once per frame).
    pub(super) fn poll_undistort(&mut self) {
        let Some(run) = &self.undistort else { return };
        let report = match run.worker.poll(&mut self.jobs) {
            Polled::Done(r) => Some(r),
            Polled::Running => return,
            Polled::Stopped => None,
        };
        let run = self.undistort.take().expect("a run is going on");
        match report {
            Some(r) => self.undistort_done(&run.items, &r),
            None => {
                let msg = "The undistortion thread stopped unexpectedly.".to_owned();
                self.log.error(msg.clone());
                self.dialogs.error(msg);
            }
        }
    }

    /// Waits for the running undistortion and collects it (tests).
    pub fn wait_for_undistort(&mut self) {
        while self.undistort.is_some() {
            self.poll_undistort();
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    /// The Console lines and, when frames were skipped or items failed, the message box.
    fn undistort_done(&mut self, items: &[UndistortItem], r: &UndistortReport) {
        for (it, ir) in items.iter().zip(&r.items) {
            match &ir.error {
                Some(e) => self
                    .log
                    .error(format!("{TITLE}: {}: not processed: {e}", it.list_text())),
                None => self.log.info(format!(
                    "{TITLE}: {}: {} of {} frames written, {} skipped",
                    it.list_text(),
                    ir.written.len(),
                    ir.frames,
                    ir.skipped.len()
                )),
            }
        }
        if r.cancelled {
            self.log.info(format!(
                "{TITLE}: cancelled after {} files; {} items not processed.",
                r.written(),
                items
                    .len()
                    .saturating_sub(r.items.iter().filter(|i| i.completed).count())
            ));
        }
        let problems = r.problems(items);
        if !problems.is_empty() {
            let text = problems_text(&problems, "items");
            // Only the problems the message box leaves out go to the Console.
            for p in problems.iter().skip(MAX_LINES) {
                self.log.info(format!("{TITLE}: {p}"));
            }
            self.dialogs.open(Dialog::Message {
                title: TITLE.to_owned(),
                text,
            });
        }
    }
}
