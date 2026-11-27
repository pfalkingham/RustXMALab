//! The jobs of package "images": File > Export > Undistorted Trial images and All Trials
//! for External (`xma_engine::trial_images`), and the Trial dialog's "Denoise Images"
//! (`xma_engine::denoise`), each on worker threads (File > Export > Blender scene, in
//! `app/blender.rs`, runs as one of these exports) with a cancelable entry in the Progress
//! panel; the project changes only when "relink Trial to denoised images" is checked and
//! every camera finished (`Edit::ChangeTrialData`, one undo step).
//!
//! Differences (ledger U161 to U163): XMALab runs the two exports on the interface thread
//! (All Trials for External with an indeterminate progress bar, Undistorted Trial images with
//! none) and its denoising jobs with an indeterminate bar and no Cancel; here each shows its
//! camera and frame and can be cancelled, files are written whole, and what could not be
//! written is listed in a message box and the Console (XMALab says nothing).

use std::path::{Path, PathBuf};

use xma_engine::Edit;
use xma_engine::denoise::{DenoiseJob, DenoiseReport, DenoiseSettings, relink_spec};
use xma_engine::trial_images::{
    AllTrialsJob, AllTrialsReport, TrialImageFormat, TrialImagesJob, TrialImagesReport,
    TrialImagesSpec,
};
use xma_vision::visual_filter::VisualFilter;

use super::XmaApp;
use crate::dialogs::Dialog;
use crate::dialogs::exports::{ExportAction, Flow};
use crate::dialogs::from_to::ImageFormat;
use crate::dialogs::images::{DENOISE_FOLDER_TITLE, DenoiseDialog, ImagesAnswer};
use crate::editing::EditRequest;
use crate::jobs::{JobHandle, Polled, Worker};

/// Progress titles.
pub const TRIAL_IMAGES_TITLE: &str = "Export Undistorted Trial images";
/// `ProgressDialog::showProgressbar(0, 0, "Exporting all for external processing")`.
pub const ALL_TRIALS_TITLE: &str = "Exporting all for external processing";
/// `ThreadedProcessing("Denoise Trial")`.
pub const DENOISE_TITLE: &str = "Denoise Trial";
/// Problem lines shown in the message box at most (also for Undistort sequence).
pub(super) const MAX_LINES: usize = 30;

/// The text of a message box listing what was not written: "<n> frames or <what> were not
/// written:", at most [`MAX_LINES`] problems, then how many more the Console has.
pub(super) fn problems_text(problems: &[String], what: &str) -> String {
    let mut text = format!("{} frames or {what} were not written:\n", problems.len());
    for p in problems.iter().take(MAX_LINES) {
        text.push_str(p);
        text.push('\n');
    }
    if problems.len() > MAX_LINES {
        text.push_str(&format!(
            "... and {} more (see the Console)",
            problems.len() - MAX_LINES
        ));
    }
    text
}

/// What an export job reports.
#[derive(Debug)]
pub(super) enum ExportReport {
    Trial(TrialImagesReport),
    All(AllTrialsReport),
    /// File > Export > Blender scene (`app/blender.rs`).
    Blender(xma_engine::blender_scene::BlenderSceneReport),
}

/// The work of an export, run on its thread with the job's handle.
pub(super) type ExportWork = dyn FnOnce(&JobHandle) -> ExportReport + Send;

/// An export running on a worker thread.
#[derive(Debug)]
struct ExportRun {
    worker: Worker<ExportReport>,
    title: &'static str,
}

/// One camera's denoising job.
#[derive(Debug)]
struct CameraRun {
    worker: Worker<DenoiseReport>,
    frames: usize,
    report: Option<DenoiseReport>,
}

/// A "Denoise Images" run: one job per camera.
#[derive(Debug)]
struct DenoiseRun {
    trial: usize,
    trial_name: String,
    folder: PathBuf,
    relink: bool,
    /// The document the run was started on (`XmaApp::doc_generation`): the relink is
    /// applied to that document only.
    document: u64,
    cameras: Vec<CameraRun>,
}

/// The package's running jobs.
#[derive(Debug, Default)]
pub struct ImagesRuns {
    exports: Vec<ExportRun>,
    denoise: Option<DenoiseRun>,
}

impl ImagesRuns {
    /// Something is running.
    #[must_use]
    pub fn busy(&self) -> bool {
        !self.exports.is_empty() || self.denoise.is_some()
    }
}

fn format_of(f: ImageFormat) -> TrialImageFormat {
    match f {
        ImageFormat::Jpg => TrialImageFormat::Jpg,
        ImageFormat::Tif => TrialImageFormat::Tif,
        ImageFormat::Avi => TrialImageFormat::Avi,
    }
}

impl XmaApp {
    /// The visual filter settings of this machine.
    fn filter_settings(&self) -> VisualFilter {
        let s = &self.settings;
        VisualFilter {
            krad: s.visual_filter_krad,
            gsigma: s.visual_filter_gsigma,
            img_wt: s.visual_filter_img_wt,
            blur_wt: s.visual_filter_blur_wt,
            gamma: s.visual_filter_gamma,
        }
    }

    /// Starts Undistorted Trial images or All Trials for External into `dir` (from
    /// `dialogs::exports::write`, once the folder is chosen).
    pub fn start_images_export(&mut self, flow: &Flow, dir: &Path) {
        let Some(doc) = self.session.doc.as_ref() else {
            return;
        };
        let filter = self.filter_settings();
        let (title, work): (&'static str, Box<ExportWork>) = match flow.action {
            ExportAction::TrialImages => {
                let Some(c) = flow.images else { return };
                let spec = TrialImagesSpec {
                    trial: flow.trial,
                    first: c.first,
                    last: c.last,
                    format: format_of(c.format),
                    filtered: c.filtered,
                    filter,
                    camera: None,
                };
                let files = self.trial_files(flow.trial).cameras;
                let job = match TrialImagesJob::new(doc, spec, &files, dir) {
                    Ok(j) => j,
                    Err(e) => {
                        // Saving the Default trial's images: XMALab writes nothing.
                        self.log
                            .info(format!("{TRIAL_IMAGES_TITLE}: {e}; nothing written"));
                        return;
                    }
                };
                (
                    TRIAL_IMAGES_TITLE,
                    Box::new(move |h: &JobHandle| {
                        ExportReport::Trial(job.run(
                            &mut |p| {
                                let done = if p.frames == 0 {
                                    0.0
                                } else {
                                    p.frame as f32 / p.frames as f32
                                };
                                h.set_progress((p.camera as f32 + done) / p.cameras.max(1) as f32);
                                h.set_message(format!(
                                    "{TRIAL_IMAGES_TITLE}: camera {} of {}, frame {} of {}",
                                    p.camera + 1,
                                    p.cameras,
                                    (p.frame + 1).min(p.frames),
                                    p.frames
                                ));
                            },
                            &|| h.is_canceled(),
                        ))
                    }),
                )
            }
            ExportAction::AllTrials => {
                let n = doc.project().trials.len();
                let files: Vec<Vec<Vec<PathBuf>>> =
                    (0..n).map(|t| self.trial_files(t).cameras).collect();
                let job = AllTrialsJob::new(
                    doc,
                    dir,
                    &|t| files.get(t).cloned().unwrap_or_default(),
                    filter,
                );
                (
                    ALL_TRIALS_TITLE,
                    Box::new(move |h: &JobHandle| {
                        ExportReport::All(job.run(
                            &mut |p| {
                                h.set_progress(
                                    (p.trial * p.cameras + p.camera) as f32
                                        / (p.trials * p.cameras).max(1) as f32,
                                );
                                h.set_message(format!(
                                    "{ALL_TRIALS_TITLE}: trial {} of {}, camera {}{}",
                                    p.trial + 1,
                                    p.trials,
                                    p.camera,
                                    if p.frames > 0 {
                                        format!(
                                            ", frame {} of {}",
                                            (p.frame + 1).min(p.frames),
                                            p.frames
                                        )
                                    } else {
                                        String::new()
                                    }
                                ));
                            },
                            &|| h.is_canceled(),
                        ))
                    }),
                )
            }
            _ => return,
        };
        self.start_export_run(title, work);
    }

    /// Runs `work` on a worker thread as the cancelable job `title` in the Progress panel;
    /// its report is collected by [`XmaApp::poll_images`].
    pub(super) fn start_export_run(&mut self, title: &'static str, work: Box<ExportWork>) {
        let handle = self.jobs.start(title, true);
        let worker = match Worker::spawn("xma-images", &handle, work) {
            Ok(w) => w,
            Err(e) => {
                self.jobs.finish(handle.id());
                let msg = format!("Cannot start the export thread: {e}");
                self.log.error(msg.clone());
                self.dialogs.error(msg);
                return;
            }
        };
        self.images.exports.push(ExportRun { worker, title });
    }

    /// The answers of the Trial dialog's "Denoise Images" and of `DenoiseDialog`.
    pub(super) fn images_answer(&mut self, a: ImagesAnswer) {
        match a {
            ImagesAnswer::DenoiseAsked { trial } => {
                // WorkspaceNavigationFrame: the folder first, then the dialog.
                let start = self.last_dir();
                let Some(folder) = self
                    .file_chooser
                    .pick_folder(DENOISE_FOLDER_TITLE, start.as_deref())
                else {
                    return;
                };
                self.dialogs
                    .open(Dialog::Denoise(Box::new(DenoiseDialog::new(trial, folder))));
            }
            ImagesAnswer::Denoise {
                trial,
                folder,
                settings,
            } => {
                // Settings::setLastUsedDirectory(outputPath, true) on OK.
                let abs = std::path::absolute(&folder).unwrap_or_else(|_| folder.clone());
                self.settings.last_directory_used = abs.to_string_lossy().into_owned();
                self.settings_dirty = true;
                self.start_denoise(trial, &folder, &settings);
            }
        }
    }

    /// Starts one "Denoise Trial" job per camera. Refused, with a Console line, while
    /// another denoising runs.
    pub fn start_denoise(&mut self, trial: usize, folder: &Path, settings: &DenoiseSettings) {
        if self.images.denoise.is_some() {
            self.log
                .error("Denoise Images cannot start while another denoising runs.".to_owned());
            return;
        }
        let Some(doc) = self.session.doc.as_ref() else {
            return;
        };
        let files = self.trial_files(trial).cameras;
        let n = doc.project().cameras.len();
        let mut jobs = Vec::new();
        for c in 0..n {
            let f = files.get(c).cloned().unwrap_or_default();
            match DenoiseJob::new(doc, trial, c, &f, folder, settings) {
                Ok(j) => jobs.push(j),
                Err(e) => {
                    let msg = format!("{DENOISE_TITLE}: {e}");
                    self.log.error(msg.clone());
                    self.dialogs.error(msg);
                    return;
                }
            }
        }
        let trial_name = doc.project().trials[trial].name.clone();
        let mut cameras = Vec::new();
        for job in jobs {
            let handle = self.jobs.start(DENOISE_TITLE, true);
            let frames = job.frames();
            let cam = job.camera();
            let spawned = Worker::spawn(format!("xma-denoise-{cam}"), &handle, move |worker| {
                job.run(
                    &mut |p| {
                        if p.frames > 0 {
                            worker.set_progress(p.frame as f32 / p.frames as f32);
                        }
                        worker.set_message(format!(
                            "{DENOISE_TITLE}: camera {}, frame {} of {}",
                            cam + 1,
                            (p.frame + 1).min(p.frames),
                            p.frames
                        ));
                    },
                    &|| worker.is_canceled(),
                )
            });
            let worker = match spawned {
                Ok(w) => w,
                Err(e) => {
                    self.jobs.finish(handle.id());
                    self.log
                        .error(format!("Cannot start the denoising thread: {e}"));
                    continue;
                }
            };
            cameras.push(CameraRun {
                worker,
                frames,
                report: None,
            });
        }
        self.images.denoise = Some(DenoiseRun {
            trial,
            trial_name,
            folder: folder.to_path_buf(),
            relink: settings.relink,
            document: self.doc_generation,
            cameras,
        });
    }

    /// A job of this package is running.
    #[must_use]
    pub fn images_busy(&self) -> bool {
        self.images.busy()
    }

    /// Collects finished jobs (once per frame).
    pub(super) fn poll_images(&mut self) {
        let mut done = Vec::new();
        let mut i = 0;
        while i < self.images.exports.len() {
            let r = match self.images.exports[i].worker.poll(&mut self.jobs) {
                Polled::Done(r) => Some(r),
                Polled::Running => {
                    i += 1;
                    continue;
                }
                Polled::Stopped => None,
            };
            let run = self.images.exports.remove(i);
            done.push((run.title, r));
        }
        for (title, r) in done {
            match r {
                Some(ExportReport::Trial(r)) => self.trial_images_done(&r),
                Some(ExportReport::All(r)) => self.all_trials_done(&r),
                Some(ExportReport::Blender(r)) => self.blender_scene_done(&r),
                None => {
                    let msg = format!("{title}: the export thread stopped unexpectedly.");
                    self.log.error(msg.clone());
                    self.dialogs.error(msg);
                }
            }
        }
        self.poll_denoise();
    }

    /// Waits for every job of this package and collects them (tests).
    pub fn wait_for_images(&mut self) {
        while self.images.busy() {
            self.poll_images();
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    pub(super) fn problems_box(&mut self, title: &str, problems: &[String]) {
        if problems.is_empty() {
            return;
        }
        let text = problems_text(problems, "files");
        for p in problems {
            self.log.info(format!("{title}: {p}"));
        }
        self.dialogs.open(Dialog::Message {
            title: title.to_owned(),
            text,
        });
    }

    fn trial_images_done(&mut self, r: &TrialImagesReport) {
        for c in &r.cameras {
            self.log.write(
                format!(
                    "{TRIAL_IMAGES_TITLE}: camera {}: {} files in {}",
                    c.camera + 1,
                    c.written.len(),
                    c.folder.display()
                ),
                crate::log::LogLevel::Output,
            );
        }
        if r.cancelled {
            self.log.info(format!(
                "{TRIAL_IMAGES_TITLE}: cancelled after {} files.",
                r.written()
            ));
        }
        self.problems_box(TRIAL_IMAGES_TITLE, &r.problems());
    }

    fn all_trials_done(&mut self, r: &AllTrialsReport) {
        self.log.write(
            format!("{ALL_TRIALS_TITLE}: {} files written", r.written),
            crate::log::LogLevel::Output,
        );
        if r.cancelled {
            self.log.info(format!("{ALL_TRIALS_TITLE}: cancelled."));
        }
        if let Some(e) = &r.error {
            self.log.error(format!("{ALL_TRIALS_TITLE}: {e}"));
            self.dialogs.error(e.clone());
        }
        self.problems_box(ALL_TRIALS_TITLE, &r.problems);
    }

    fn poll_denoise(&mut self) {
        let Some(run) = &mut self.images.denoise else {
            return;
        };
        for c in &mut run.cameras {
            if c.report.is_some() {
                continue;
            }
            match c.worker.poll(&mut self.jobs) {
                Polled::Done(r) => c.report = Some(r),
                Polled::Running => {}
                Polled::Stopped => {
                    c.report = Some(DenoiseReport {
                        error: Some("the denoising thread stopped unexpectedly".to_owned()),
                        ..DenoiseReport::default()
                    });
                }
            }
        }
        if self
            .images
            .denoise
            .as_ref()
            .is_some_and(|r| r.cameras.iter().all(|c| c.report.is_some()))
        {
            let run = self.images.denoise.take().expect("a run");
            self.denoise_done(&run);
        }
    }

    /// Every camera's job has ended: the Console lines, the problems, then the relink.
    ///
    /// Ported from ui/WorkspaceNavigationFrame.cpp, changeDenoiseTrialDataAfterDenoise (the
    /// slot connected to the last job's `signal_finished`).
    fn denoise_done(&mut self, run: &DenoiseRun) {
        let mut problems = Vec::new();
        let reports: Vec<DenoiseReport> = run
            .cameras
            .iter()
            .filter_map(|c| c.report.clone())
            .collect();
        for r in &reports {
            self.log.write(
                format!(
                    "{DENOISE_TITLE}: camera {}: {} frames denoised, {} copied, {} files in {}",
                    r.camera + 1,
                    r.denoised,
                    r.copied,
                    r.written.len(),
                    xma_engine::denoise::camera_folder(&run.folder, r.camera).display()
                ),
                crate::log::LogLevel::Output,
            );
            if let Some(e) = &r.error {
                problems.push(format!("Camera {}: {e}", r.camera + 1));
            }
            if r.cancelled {
                self.log.info(format!(
                    "{DENOISE_TITLE}: camera {} cancelled.",
                    r.camera + 1
                ));
            }
        }
        let frames = run.cameras.first().map_or(0, |c| c.frames);
        self.problems_box(DENOISE_TITLE, &problems);
        if !run.relink {
            return;
        }
        // The files are written; the trial they replace belongs to a project that was closed
        // (another, or the same file opened again, may be open now).
        if run.document != self.doc_generation {
            self.log.info(format!(
                "{DENOISE_TITLE}: the trial \"{}\" is not relinked: its project was closed.",
                run.trial_name
            ));
            return;
        }
        let Some(doc) = self.session.doc.as_ref() else {
            return;
        };
        if !xma_engine::denoise::relink_due(doc, &reports, frames) {
            self.log.info(format!(
                "{DENOISE_TITLE}: the trial is not relinked: not every camera finished."
            ));
            return;
        }
        // The trial, if still there under its name.
        let trial = if doc.project().trials.get(run.trial).map(|t| &t.name) == Some(&run.trial_name)
        {
            run.trial
        } else if let Some(t) = doc
            .project()
            .trials
            .iter()
            .position(|t| t.name == run.trial_name)
        {
            t
        } else {
            self.log
                .info(format!("{DENOISE_TITLE}: the trial is gone; not relinked."));
            return;
        };
        let Some(spec) = relink_spec(doc, trial, &run.folder, &reports) else {
            return;
        };
        let revision = doc.revision();
        self.apply_request(EditRequest::Apply(Edit::ChangeTrialData { trial, spec }));
        if self
            .session
            .doc
            .as_ref()
            .is_some_and(|d| d.revision() != revision)
        {
            // As after "Change Trial Data", except that the stashed videos of deleted
            // trials are kept: the old files' replacements are forgotten and Marker
            // tracking shows the trial at its frame.
            self.trial_files_replaced(trial);
        }
    }
}
