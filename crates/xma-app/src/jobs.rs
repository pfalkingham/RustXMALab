//! Background jobs and their progress, shown in the Progress panel.
//!
//! Ported from ui/ProgressDialog.cpp (`ProgressDialog`, the Progress panel). The C++ has one
//! progress bar: `showProgressbar(min, max, title, cancelable)` shows the panel with the
//! title as its window title, disables the whole main window (or, for a cancelable job,
//! only the Toolbox, Points, Plot, frame bar, workspace bar and Detailed View, so Cancel
//! stays usable); `setProgress` moves the bar; `closeProgressbar` hides the panel and
//! enables everything again. `min == max == 0` is an indeterminate ("busy") bar.
//!
//! Here a job is started with [`Jobs::start`], which returns a [`JobHandle`] that a worker
//! thread keeps to report progress and to see whether Cancel was pressed; the job ends when
//! [`Jobs::finish`] is called (normally by the code that started it, when the worker's
//! result arrives). Several jobs may run at once; the panel shows the oldest.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, mpsc};

/// Identifies a running job.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JobId(u64);

#[derive(Debug)]
struct Shared {
    /// Progress in `[0, 1]` as `f32` bits; `u32::MAX` while indeterminate.
    progress: AtomicU32,
    canceled: AtomicBool,
    /// Replaces the title while set (a worker may describe its current step).
    message: Mutex<Option<String>>,
}

/// What a worker holds: report progress, read the cancel flag. Cheap to clone; `Send`.
#[derive(Clone, Debug)]
pub struct JobHandle {
    id: JobId,
    shared: Arc<Shared>,
    repaint: Repaint,
}

/// The context to wake when a worker reports, shared by every job: a job started before
/// the window exists (the project load from the command line) still wakes it once it does.
type Repaint = Arc<Mutex<Option<egui::Context>>>;

/// Asks for a frame of the main window. A worker thread's plain `request_repaint` asks for
/// the viewport egui is drawing at that moment, which may be a floating panel's window,
/// so the main window would wait for its next event.
fn wake(repaint: &Repaint) {
    if let Some(ctx) = repaint.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
        ctx.request_repaint_of(egui::ViewportId::ROOT);
    }
}

impl JobHandle {
    #[must_use]
    pub fn id(&self) -> JobId {
        self.id
    }

    /// Ported from ui/ProgressDialog.cpp, ProgressDialog::setProgress, with the value as a
    /// fraction in `[0, 1]` instead of a value in `[min, max]`.
    pub fn set_progress(&self, fraction: f32) {
        let f = fraction.clamp(0.0, 1.0);
        self.shared.progress.store(f.to_bits(), Ordering::Relaxed);
        wake(&self.repaint);
    }

    /// Replaces the panel title while the job runs.
    pub fn set_message(&self, text: impl Into<String>) {
        if let Ok(mut m) = self.shared.message.lock() {
            *m = Some(text.into());
        }
        wake(&self.repaint);
    }

    /// `ProgressDialog::getIsCanceled`.
    #[must_use]
    pub fn is_canceled(&self) -> bool {
        self.shared.canceled.load(Ordering::Relaxed)
    }
}

/// A running job as the Progress panel shows it.
#[derive(Clone, Debug)]
pub struct JobInfo {
    pub id: JobId,
    /// The window title of the C++ panel (the `key` of `showProgressbar`).
    pub title: String,
    pub cancelable: bool,
    shared: Arc<Shared>,
}

impl JobInfo {
    /// `None` while indeterminate.
    #[must_use]
    pub fn progress(&self) -> Option<f32> {
        let bits = self.shared.progress.load(Ordering::Relaxed);
        (bits != u32::MAX).then(|| f32::from_bits(bits))
    }

    /// The title, or the worker's current message.
    #[must_use]
    pub fn display_title(&self) -> String {
        self.shared
            .message
            .lock()
            .ok()
            .and_then(|m| m.clone())
            .unwrap_or_else(|| self.title.clone())
    }

    #[must_use]
    pub fn is_canceled(&self) -> bool {
        self.shared.canceled.load(Ordering::Relaxed)
    }

    /// Ported from ui/ProgressDialog.cpp, ProgressDialog::on_cancelButton_clicked.
    pub fn cancel(&self) {
        self.shared.canceled.store(true, Ordering::Relaxed);
    }
}

/// The running jobs.
#[derive(Debug, Default)]
pub struct Jobs {
    running: Vec<JobInfo>,
    next: u64,
    repaint: Repaint,
}

impl Jobs {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Lets workers request a repaint when they report progress.
    pub fn set_repaint_context(&mut self, ctx: egui::Context) {
        *self.repaint.lock().unwrap_or_else(|e| e.into_inner()) = Some(ctx);
    }

    /// Ported from ui/ProgressDialog.cpp, ProgressDialog::showProgressbar. The bar starts
    /// indeterminate.
    pub fn start(&mut self, title: impl Into<String>, cancelable: bool) -> JobHandle {
        self.next += 1;
        let id = JobId(self.next);
        let shared = Arc::new(Shared {
            progress: AtomicU32::new(u32::MAX),
            canceled: AtomicBool::new(false),
            message: Mutex::new(None),
        });
        self.running.push(JobInfo {
            id,
            title: title.into(),
            cancelable,
            shared: shared.clone(),
        });
        JobHandle {
            id,
            shared,
            repaint: self.repaint.clone(),
        }
    }

    /// Ported from ui/ProgressDialog.cpp, ProgressDialog::closeProgressbar.
    pub fn finish(&mut self, id: JobId) {
        self.running.retain(|j| j.id != id);
    }

    /// The job the panel shows (the oldest), if any.
    #[must_use]
    pub fn current(&self) -> Option<&JobInfo> {
        self.running.first()
    }

    #[must_use]
    pub fn running(&self) -> &[JobInfo] {
        &self.running
    }

    #[must_use]
    pub fn is_busy(&self) -> bool {
        !self.running.is_empty()
    }

    /// A non-cancelable job runs: the C++ disables the whole main window.
    #[must_use]
    pub fn blocks_window(&self) -> bool {
        self.running.iter().any(|j| !j.cancelable)
    }
}

/// What [`Worker::poll`] found.
#[derive(Debug)]
pub enum Polled<T> {
    /// The worker has not sent its result yet.
    Running,
    /// The worker's result.
    Done(T),
    /// The thread ended without sending a result (it panicked). Each caller says so in its
    /// own words ("the … thread stopped unexpectedly"), or not at all.
    Stopped,
}

/// A worker thread's result channel and the job it reports to: the part every background
/// task of the window has in common. The thread sends its one result, then sets the job's
/// bar to 1, which wakes the window to collect it.
pub struct Worker<T> {
    rx: mpsc::Receiver<T>,
    job: Option<JobHandle>,
}

impl<T> std::fmt::Debug for Worker<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Worker")
            .field("job", &self.job)
            .finish_non_exhaustive()
    }
}

impl<T: Send + 'static> Worker<T> {
    /// Runs `work` on a thread named `name` for the job `job` (whose handle `work` gets, to
    /// report progress and read Cancel). An error is the operating system's refusal to start
    /// the thread; the job is left running for the caller to end.
    pub fn spawn(
        name: impl Into<String>,
        job: &JobHandle,
        work: impl FnOnce(&JobHandle) -> T + Send + 'static,
    ) -> std::io::Result<Self> {
        let (tx, rx) = mpsc::channel();
        let wake = job.clone();
        std::thread::Builder::new()
            .name(name.into())
            .spawn(move || {
                let _ = tx.send(work(&wake));
                wake.set_progress(1.0);
            })?;
        Ok(Worker {
            rx,
            job: Some(job.clone()),
        })
    }

    /// Runs `work` on a thread named `name`, with no job in the Progress panel.
    pub fn spawn_untracked(
        name: impl Into<String>,
        work: impl FnOnce() -> T + Send + 'static,
    ) -> std::io::Result<Self> {
        let (tx, rx) = mpsc::channel();
        std::thread::Builder::new()
            .name(name.into())
            .spawn(move || {
                let _ = tx.send(work());
            })?;
        Ok(Worker { rx, job: None })
    }

    /// A worker whose result is already there (a thread that could not start, reported
    /// on the next poll as if the thread had sent it).
    #[must_use]
    pub fn ready(job: JobHandle, result: T) -> Self {
        let (tx, rx) = mpsc::channel();
        let _ = tx.send(result);
        Worker { rx, job: Some(job) }
    }
}

impl<T> Worker<T> {
    /// The result, if the worker is done, without ending the job.
    pub fn try_recv(&self) -> Polled<T> {
        match self.rx.try_recv() {
            Ok(r) => Polled::Done(r),
            Err(mpsc::TryRecvError::Empty) => Polled::Running,
            Err(mpsc::TryRecvError::Disconnected) => Polled::Stopped,
        }
    }

    /// The result, if the worker is done; the job ends when it is ([`Jobs::finish`]).
    pub fn poll(&self, jobs: &mut Jobs) -> Polled<T> {
        let r = self.try_recv();
        if !matches!(r, Polled::Running) {
            self.finish(jobs);
        }
        r
    }

    /// Ends the job (the result is no longer wanted, or has been collected).
    pub fn finish(&self, jobs: &mut Jobs) {
        if let Some(j) = &self.job {
            jobs.finish(j.id());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_and_cancel_round_trip() {
        let mut jobs = Jobs::new();
        let h = jobs.start("Load dataset x", true);
        assert!(jobs.is_busy());
        assert!(!jobs.blocks_window());
        assert_eq!(jobs.current().unwrap().progress(), None);
        h.set_progress(0.25);
        assert_eq!(jobs.current().unwrap().progress(), Some(0.25));
        jobs.current().unwrap().cancel();
        assert!(h.is_canceled());
        jobs.finish(h.id());
        assert!(!jobs.is_busy());
    }

    /// A job started before the window exists (the load of a project named on the command
    /// line) wakes the main window when it reports, once the context is known.
    #[test]
    fn a_job_started_before_the_window_wakes_it() {
        let mut jobs = Jobs::new();
        let h = jobs.start("Load dataset x", false);
        let ctx = egui::Context::default();
        let woken = Arc::new(Mutex::new(Vec::new()));
        let w = woken.clone();
        ctx.set_request_repaint_callback(move |info| {
            w.lock().unwrap().push(info.viewport_id);
        });
        jobs.set_repaint_context(ctx.clone());
        assert!(woken.lock().unwrap().is_empty());
        h.set_progress(1.0);
        assert_eq!(*woken.lock().unwrap(), [egui::ViewportId::ROOT]);
    }
}
