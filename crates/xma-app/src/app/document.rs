//! The application's handling of the document: the edit requests, the tracking run, undo
//! and redo bookkeeping, bringing the interface up to date after a change, saving and
//! autosave. See `crate::editing` and `crate::saving` for the rules; this file is where
//! the application carries them out.

use std::path::{Path, PathBuf};

use xma_engine::edit::FrameRange;
use xma_engine::tracking::{self, Placement, TrackError, TrackRun};
use xma_engine::{Document, EditResult};

use super::XmaApp;
use crate::editing::{self, EditRequest, ErrorReport, TrackingRun, TrialFiles};
use crate::log::LogLevel;
use crate::saving::{self, AutosaveOutcome, SaveJob, SaveKind};
use crate::session::document_title;
use crate::state::{Changes, ProjectFacts, TrialView};

/// Where the active trial went when the document's trials changed
/// (`XmaApp::follow_trial_list`).
pub(super) enum TrialMove {
    /// Still there, at this index.
    Moved(usize),
    /// Deleted.
    Gone,
}

/// The identity of each camera's shared data: it changes when an edit changes the camera
/// (its calibration, grid or images).
pub(super) fn camera_ids(doc: &Document) -> Vec<usize> {
    doc.project()
        .cameras
        .iter()
        .map(|c| std::sync::Arc::as_ptr(c) as usize)
        .collect()
}

/// Per trial, the model's start frame, end frame and number of frames.
pub(super) fn trial_ranges(doc: &Document) -> Vec<(usize, usize, usize)> {
    doc.project()
        .trials
        .iter()
        .map(|t| (t.start_frame, t.end_frame, t.n_frames))
        .collect()
}

impl XmaApp {
    // -----------------------------------------------------------------------------------
    // Edit requests.

    /// Handles the queued edit requests in order (after the actions of the frame).
    pub fn handle_edit_requests(&mut self) {
        for r in self.edits.take() {
            self.apply_request(r);
        }
    }

    /// Carries out one edit request against the session's document (`crate::editing`).
    /// Tests call it directly; panels raise requests with `Cx::request`.
    pub fn apply_request(&mut self, r: EditRequest) {
        if self.session.doc.is_none() {
            return;
        }
        if let Some(log) = &mut self.request_log {
            log.push(r.clone());
        }
        // A tracking run, a refinement or a long edit in progress holds the project.
        // (The copy being computed replaces the document when it is done.)
        if self.long_edit.is_some() {
            self.log
                .error("The project cannot be changed while a computation runs.");
            return;
        }
        let tracking = self.tracking.is_some() || self.refine.is_some();
        let refused_while_tracking = |app: &mut Self| {
            app.log
                .error("The project cannot be changed while tracking runs; stop it first.");
        };
        match r {
            // C44: its errors are shown as the error dialog (`app/calibration_ext.rs`).
            // (C42, the optimisation, is a long edit: `apply_calibration_request` below.)
            EditRequest::Apply(xma_engine::Edit::Calibration(
                edit @ xma_engine::calibration::CalibrationEdit::ApplyExternal { .. },
            )) => self.apply_calibration_ext(edit),
            EditRequest::Apply(edit) => {
                if tracking {
                    return refused_while_tracking(self);
                }
                // Undistortion and calibration edits: long ones on a worker thread, and the
                // window follows the session they leave (`app/calibration.rs`).
                if self.apply_calibration_request(&edit) {
                    return;
                }
                let res = self.doc_mut().apply(&edit);
                self.edit_result(res);
            }
            EditRequest::ApplyAll { label, edits } => {
                if tracking {
                    return refused_while_tracking(self);
                }
                let res = self.doc_mut().apply_all(&label, &edits);
                self.edit_result(res);
            }
            EditRequest::ImageAdjustment {
                camera,
                view,
                scale,
                bias,
            } => {
                // Not an undo step and not a change of what panels show: the views already
                // show the values. Only the title (modified) follows; `Document::changes`,
                // which `sync_document` follows, does not move.
                self.doc_mut()
                    .set_image_adjustment(camera, view, scale, bias);
            }
            EditRequest::PlacePoint {
                trial,
                camera,
                frame,
                x,
                y,
                placement,
            } => {
                if tracking {
                    return refused_while_tracking(self);
                }
                let files = self.trial_files(trial);
                let Some(doc) = self.session.doc.as_mut() else {
                    return;
                };
                let mut images = editing::trial_images(&mut self.frames, &files);
                let res = tracking::place_point(
                    doc,
                    images.as_mut(),
                    trial,
                    camera,
                    frame,
                    x,
                    y,
                    placement,
                );
                drop(images);
                match res {
                    Ok(placed) => {
                        // WizardDigitizationFrame::addDigitizationPoint: the marker chosen
                        // (or created) by Ctrl+click becomes the active one.
                        if (placed.marker_added || matches!(placement, Placement::AddNext { .. }))
                            && self.state.active_trial == Some(trial)
                        {
                            self.changes |= self.state.select_marker(Some(placed.marker));
                        }
                        // GLCameraView::mousePressEvent: after a click in a camera view (E1,
                        // E2; not the Detailed View, E3) the detail views are centred.
                        if !matches!(placement, Placement::Manual(_)) {
                            self.center_detail_views();
                        }
                        // moveDigitizationPoint: Detection Settings shows the detection at
                        // the clicked point (E2, E3; not E1).
                        if let Placement::Detect(m) | Placement::Manual(m) = placement {
                            self.detection
                                .clicked(&self.state, trial, m, camera, frame, (x, y));
                        }
                    }
                    Err(e) => self.report_edit_error(&e),
                }
            }
            EditRequest::StartTracking(request) => {
                if tracking {
                    return;
                }
                let button = crate::tracking_ui::TrackButton::of_request(&request);
                // The multi-camera kinds run with the method's default parameters but for
                // the four shown in the window (`tracking_ui::multi_params`, ledgers U169
                // and U170); the other kinds do not read the parameters.
                let doc = self.session.doc.as_mut().expect("a document is open");
                let params = crate::tracking_ui::multi_params(&self.state);
                match TrackRun::start(doc, request, params) {
                    Ok(tracker) => {
                        doc.begin_step(editing::TRACKING_STEP);
                        self.state.disable_draw = true;
                        self.state.running_track = Some(button);
                        self.tracking = Some(TrackingRun::new(tracker));
                    }
                    // XMALab writes these lines to the Console and does not start.
                    Err(TrackError::Refused(text)) => {
                        self.log.write(text, LogLevel::Output);
                    }
                    Err(e @ (TrackError::Invalid(_) | TrackError::NotImplemented(_))) => {
                        self.log.error(e.to_string());
                    }
                }
            }
            EditRequest::StopTracking => {
                if let Some(run) = &mut self.tracking {
                    run.tracker.stop();
                }
            }
            EditRequest::RefinePoints {
                trial,
                markers,
                first,
                last,
                which,
            } => {
                if tracking {
                    return refused_while_tracking(self);
                }
                self.start_refine(trial, markers, first, last, which);
            }
            EditRequest::Ask(ask) => {
                if tracking {
                    return refused_while_tracking(self);
                }
                self.dialogs.open(crate::dialogs::Dialog::Ask(ask));
            }
        }
    }

    fn doc_mut(&mut self) -> &mut Document {
        self.session.doc.as_mut().expect("a document is open")
    }

    /// The outcome of an applied edit: messages to the Console, errors where XMALab shows
    /// them.
    pub(super) fn edit_result(&mut self, res: EditResult) {
        match res {
            Ok(outcome) => {
                for m in outcome.messages {
                    self.log.write(m, LogLevel::Output);
                }
                // Only the XMALab 3 mode sets this (an edit that changed the project and
                // then shows XMALab's error); the window runs in the default mode.
                if let Some(text) = outcome.refused {
                    self.dialogs.error(text);
                }
            }
            Err(e) => self.report_edit_error(&e),
        }
    }

    pub(super) fn report_edit_error(&mut self, e: &xma_engine::EditError) {
        match editing::report_error(e) {
            ErrorReport::Dialog(text) => self.dialogs.error(text),
            ErrorReport::Console(text) => self.log.error(text),
        }
    }

    /// The files of each camera of a trial, as the frame service reads them.
    pub fn trial_files(&self, trial: usize) -> TrialFiles {
        let n = self
            .session
            .doc
            .as_ref()
            .map_or(0, |d| d.project().cameras.len());
        TrialFiles {
            trial,
            cameras: (0..n).map(|c| self.session.video_files(trial, c)).collect(),
            flips: self.session.doc.as_ref().map_or_else(Vec::new, |d| {
                d.project().cameras.iter().map(|c| c.flipped).collect()
            }),
        }
    }

    // -----------------------------------------------------------------------------------
    // Tracking run.

    /// One step of the tracking run, then, when the run has ended, its undo step closed.
    pub fn step_tracking(&mut self) {
        let Some(mut run) = self.tracking.take() else {
            return;
        };
        // WizardDigitizationFrame::activePointChanged: a single-marker run stops when the
        // active marker changes.
        if let tracking::TrackKind::Single(m) = run.tracker.request().kind
            && self.state.active_trial == Some(run.tracker.request().trial)
            && self
                .state
                .trial_view()
                .is_some_and(|v| v.active_marker != Some(m))
        {
            run.tracker.stop();
        }
        if !run.finished() {
            let trial = run.tracker.request().trial;
            let files = self.trial_files(trial);
            let Some(doc) = self.session.doc.as_mut() else {
                return;
            };
            let mut images = editing::trial_images(&mut self.frames, &files);
            let report = run.tracker.step(doc, images.as_mut());
            drop(images);
            // A multi-camera run's lines (one per point refused) would flood the Console: it
            // writes a summary when it ends instead (`editing::multi_summary`).
            if !run.tracker.request().kind.is_multi() {
                for m in &report.messages {
                    self.log.write(m.clone(), LogLevel::Output);
                }
            }
            // Show the frame the run reached.
            if self.state.active_trial == Some(trial) {
                self.changes |= self
                    .state
                    .change_active_frame_trial(Some(report.frame), false);
            }
            run.record(report);
        }
        if run.finished() {
            self.end_tracking(run);
        } else {
            self.tracking = Some(run);
        }
    }

    /// Stops the tracking run, if any, and closes its undo step at once.
    pub fn finish_tracking(&mut self) {
        if let Some(mut run) = self.tracking.take() {
            run.tracker.stop();
            self.end_tracking(run);
        }
        self.finish_refine();
    }

    fn end_tracking(&mut self, run: TrackingRun) {
        if let Some(doc) = self.session.doc.as_mut()
            && doc.step_open()
        {
            doc.commit_step();
        }
        self.state.disable_draw = false;
        self.state.running_track = None;
        if let Some(doc) = self.session.doc.as_ref() {
            for line in editing::multi_summary(doc, &run) {
                self.log.write(line, LogLevel::Output);
            }
        }
        if let Some(text) = run.tracker.finished().and_then(editing::stop_message) {
            self.log.error(text);
        }
    }

    /// Stops the tracking run before its next step, if one is open (Escape, Stop, Play, a
    /// workspace or trial change: `WizardDigitizationFrame::stopTracking`).
    pub fn stop_tracking_run(&mut self) {
        if let Some(run) = &mut self.tracking {
            run.tracker.stop();
        }
    }

    /// Re-centres the Detailed View's views on the active marker without resetting their
    /// zoom (`DetailViewDockWidget::centerViews`).
    fn center_detail_views(&mut self) {
        if let Some(doc) = self.session.doc.as_ref() {
            crate::panels::detail_view::center_on_active(&mut self.frames, doc, &self.state, false);
        }
    }

    // -----------------------------------------------------------------------------------
    // Refinement (E21).

    /// Starts "Refine center of selected Points by polynomial fit" as a run of one frame per
    /// screen frame (`editing::RefineRun`).
    fn start_refine(
        &mut self,
        trial: usize,
        markers: Vec<usize>,
        first: usize,
        last: usize,
        which: tracking::RefineWhich,
    ) {
        let Some(doc) = self.session.doc.as_mut() else {
            return;
        };
        // The checks of refine_points, before anything is opened.
        let n = doc.project().trials.get(trial).map_or(0, |t| t.n_frames);
        let n_markers = doc
            .project()
            .trials
            .get(trial)
            .map_or(0, |t| t.markers.len());
        if markers.is_empty() || first > last {
            return;
        }
        if last >= n || markers.iter().any(|&m| m >= n_markers) {
            self.log.error(format!(
                "Refine: frames {} to {} or markers {markers:?} are not in trial {}",
                first + 1,
                last + 1,
                trial + 1
            ));
            return;
        }
        doc.begin_step("Refine Points");
        let job = self.jobs.start("Refining center", true);
        job.set_progress(0.0);
        self.state.disable_draw = true;
        self.refine = Some(editing::RefineRun {
            trial,
            markers,
            which,
            next: first,
            last,
            first,
            frame_before: (self.state.active_trial == Some(trial))
                .then_some(self.state.active_frame_trial)
                .flatten(),
            count: 0,
            job,
        });
    }

    /// One frame of the refinement, then, when it has ended (all frames, Cancel or an error),
    /// its undo step closed with what was refined.
    pub fn step_refine(&mut self) {
        let Some(mut run) = self.refine.take() else {
            return;
        };
        let mut done = run.next > run.last || run.job.is_canceled();
        if !done {
            let f = run.next;
            // The C++ makes each frame the active frame while it refines it.
            if self.state.active_trial == Some(run.trial) {
                self.changes |= self.state.change_active_frame_trial(Some(f), false);
            }
            let files = self.trial_files(run.trial);
            if let Some(doc) = self.session.doc.as_mut() {
                let mut images = editing::trial_images(&mut self.frames, &files);
                let res = tracking::refine_points(
                    doc,
                    images.as_mut(),
                    run.trial,
                    &run.markers,
                    FrameRange { first: f, last: f },
                    run.which,
                    &mut |_| true,
                );
                drop(images);
                match res {
                    Ok(c) => run.count += c,
                    Err(e) => {
                        self.report_edit_error(&e);
                        done = true;
                    }
                }
            }
            run.next += 1;
            let total = (run.last - run.first + 1) as f32;
            run.job.set_progress((run.next - run.first) as f32 / total);
            done |= run.next > run.last;
        }
        if done {
            self.end_refine(run);
        } else {
            self.refine = Some(run);
        }
    }

    /// Closes the refinement's undo step (keeping the frames refined so far), ends its
    /// job and shows the frame shown before it.
    pub fn end_refine(&mut self, run: editing::RefineRun) {
        if let Some(doc) = self.session.doc.as_mut()
            && doc.step_open()
        {
            doc.commit_step();
        }
        self.jobs.finish(run.job.id());
        self.state.disable_draw = false;
        if let Some(f) = run.frame_before
            && self.state.active_trial == Some(run.trial)
        {
            self.changes |= self.state.change_active_frame_trial(Some(f), false);
        }
    }

    /// Ends a refinement in progress at once (closing or saving the project).
    pub fn finish_refine(&mut self) {
        if let Some(run) = self.refine.take() {
            self.end_refine(run);
        }
    }

    // -----------------------------------------------------------------------------------
    // Keeping the interface in step with the document.

    /// Once per frame: the undo and redo labels, and, when [`Document::changes`] moved,
    /// [`XmaApp::document_changed`].
    pub(super) fn sync_document(&mut self) {
        let Some(doc) = &self.session.doc else {
            self.seen_revision = None;
            return;
        };
        if self.state.undo_label.as_deref() != doc.undo_label() {
            self.state.undo_label = doc.undo_label().map(str::to_owned);
        }
        if self.state.redo_label.as_deref() != doc.redo_label() {
            self.state.redo_label = doc.redo_label().map(str::to_owned);
        }
        let modified = doc.is_modified();
        if modified && !self.was_modified {
            // The autosave interval counts from the first unsaved change.
            self.autosave.since = std::time::Instant::now();
        }
        self.was_modified = modified;
        if self.seen_revision != Some(doc.changes()) {
            self.seen_revision = Some(doc.changes());
            self.document_changed();
        }
    }

    /// Brings what shows project data up to date after an edit, undo or redo (what XMALab 3
    /// does with `redrawGL`, `reloadListFromObject` and `updateMarkers` after each edit):
    /// the project facts (trials, cameras), the per-trial view state (start and end frames
    /// when the project's changed, frame and active marker or body kept in range), the
    /// active trial and the selection kept in range, then [`Changes::document`] for the
    /// panels.
    pub fn document_changed(&mut self) {
        // The per-trial interface state follows the trials, not their indices, when trials
        // were deleted, imported or brought back by undo (`follow_trial_list`).
        let old_active = self.state.active_trial;
        let moved = self.follow_trial_list();
        let Some(doc) = &self.session.doc else {
            return;
        };
        let facts = ProjectFacts::from_doc(doc);
        if facts != self.facts {
            self.facts = facts;
        }
        // The cameras changed (a calibration edit, its undo or redo): the derived images
        // and the views' cached grid and calibration points are made again.
        let cameras = camera_ids(doc);
        if cameras != self.seen_cameras {
            self.seen_cameras = cameras;
            self.frames.calibration_changed();
        }
        let trials = &doc.project().trials;
        let ranges = trial_ranges(doc);
        let mut ch = Changes {
            document: true,
            ..Changes::NONE
        };
        self.state.trials.truncate(trials.len());
        for (i, t) in trials.iter().enumerate() {
            let fresh = TrialView::from_trial(t);
            if i >= self.state.trials.len() {
                self.state.trials.push(fresh);
                continue;
            }
            let v = &mut self.state.trials[i];
            if self.seen_trials.get(i) != Some(&ranges[i]) {
                v.start_frame = fresh.start_frame;
                v.end_frame = fresh.end_frame;
                v.n_frames = fresh.n_frames;
                ch.trial_range = true;
            }
            if v.active_frame >= v.n_frames.max(1) {
                v.active_frame = v.n_frames.saturating_sub(1);
            }
            // D15 fixed: when the start or end frame changed, the trial's frame is moved into
            // the new range (XMALab 3 compares the 1-based end with the 0-based frame and can
            // leave it one past the end, docs/cpp-editing.md D15).
            if self.seen_trials.get(i) != Some(&ranges[i]) && v.start_frame <= v.end_frame {
                let (first, last) = (
                    v.start_frame.saturating_sub(1),
                    v.end_frame.saturating_sub(1),
                );
                v.active_frame = v.active_frame.clamp(first, last);
            }
            if v.active_marker.is_some_and(|m| m >= t.markers.len()) {
                v.active_marker = None;
                ch.selection = true;
            }
            if v.active_body.is_some_and(|b| b >= t.rigid_bodies.len()) {
                v.active_body = None;
                ch.selection = true;
            }
        }
        self.seen_trials = ranges;
        let n_cameras = doc.project().cameras.len();
        self.state.camera_visible.resize(n_cameras, true);
        match moved {
            // The active trial is still there, at another index.
            Some(TrialMove::Moved(t)) => ch |= self.state.change_active_trial(Some(t), true),
            // The active trial is gone: the one the edit made active (E62, the same index or
            // the last one), as a new trial (the trial update and the frame follow).
            Some(TrialMove::Gone) => {
                let t =
                    (!trials.is_empty()).then(|| doc.project().active_trial.min(trials.len() - 1));
                ch |= self.state.change_active_trial(t, true);
            }
            None => {}
        }
        match self.state.active_trial {
            Some(t) if t >= trials.len() => {
                let last = trials.len().checked_sub(1);
                ch |= self.state.change_active_trial(last, false);
            }
            _ => {}
        }
        // The frame on screen follows the active trial's frame (a new range, D15).
        if self.state.active_trial == old_active
            && let Some(v) = self.state.trial_view().copied()
            && self
                .state
                .active_frame_trial
                .is_some_and(|f| f != v.active_frame)
        {
            ch |= self
                .state
                .change_active_frame_trial(Some(v.active_frame), false);
        }
        if let Some(t) = self.state.active_trial.and_then(|t| trials.get(t)) {
            let (nm, nb) = (t.markers.len(), t.rigid_bodies.len());
            let before = self.state.selection.clone();
            self.state.selection.markers.retain(|m| *m < nm);
            self.state.selection.bodies.retain(|b| *b < nb);
            ch.selection |= before != self.state.selection;
        }
        // An undo or redo of an undistortion or calibration puts back the session's
        // undistortion state and the calibration frames: the pages follow.
        ch |= self.follow_session_computed();
        self.changes |= ch;
    }

    /// Changes algorithm settings from the Settings dialog (E73 to E79): remembered as this
    /// machine's defaults (`UiSettings::algorithm_defaults`, used for projects that store no
    /// value), and, with a project open, applied to it as one undoable edit
    /// (`Edit::SetSettings`, which recomputes what depends on them and is saved with the
    /// project, parity ledger A10). In XMALab 3 these are per-machine settings only.
    pub fn set_algorithm_settings(&mut self, patch: xma_model::SettingsPatch) {
        self.settings.remember_algorithm_settings(&patch);
        self.settings_dirty = true;
        if self.session.doc.is_some() {
            self.edits
                .push(EditRequest::Apply(xma_engine::Edit::SetSettings(patch)));
        }
    }

    /// Removes a recovery file, writing a failure to the Console.
    pub(super) fn remove_recovery_file(&self, path: &Path) {
        if let Err(e) = saving::remove_recovery(path) {
            self.log.error(format!(
                "Cannot remove the recovery file {}: {e}",
                path.display()
            ));
        }
    }

    /// The window title: the project's name, `*` while it has unsaved changes.
    pub(super) fn compute_title(&self) -> String {
        match &self.session.doc {
            Some(doc) => document_title(doc.path(), doc.is_modified()),
            None => crate::session::window_title(None),
        }
    }

    // -----------------------------------------------------------------------------------
    // Saving.

    /// The file dialog of Save As and Save subdataset as (`MainWindow::saveProjectAs`):
    /// "Save dataset as", starting at the project's file or the last used folder, `.xma`
    /// appended when the name has neither `.xma` nor `.zip`; the folder and the file are
    /// remembered (last used folder, recent files) before saving, as the C++ does.
    pub(super) fn choose_save_path(&mut self) -> Option<PathBuf> {
        let start = self
            .session
            .doc
            .as_ref()
            .and_then(|d| d.path().map(Path::to_path_buf))
            .or_else(|| self.last_dir());
        let chosen = self
            .file_chooser
            .save_file("Save dataset as", start.as_deref())?;
        let path = saving::ensure_extension(chosen);
        let abs = std::path::absolute(&path).unwrap_or_else(|_| path.clone());
        self.settings.set_last_used_directory_from_file(&abs);
        self.settings.add_to_recent_files(&abs.to_string_lossy());
        self.settings_dirty = true;
        Some(abs)
    }

    /// Starts a save of a copy of the document on a worker thread.
    pub(super) fn start_save(&mut self, path: PathBuf, kind: SaveKind) {
        if self.save_job.is_some() {
            return;
        }
        self.finish_tracking();
        // Ledger A210: the copy starts with no Console lines of its own, so that those of
        // the save's recomputation (`save_done`) are not written twice.
        self.write_pending_console_lines();
        let Some(doc) = &self.session.doc else {
            return;
        };
        self.save_changes = doc.changes();
        self.save_job = Some(SaveJob::start(
            doc.clone(),
            path,
            kind,
            self.settings.recompute_when_saving,
            &mut self.jobs,
        ));
    }

    /// True while a save runs.
    #[must_use]
    pub fn is_saving(&self) -> bool {
        self.save_job.is_some()
    }

    /// Waits for the running save and handles its result (tests).
    pub fn wait_for_save(&mut self) {
        if let Some(job) = &mut self.save_job {
            let out = job.wait(&mut self.jobs);
            self.save_job = None;
            self.save_done(out);
        }
    }

    pub(super) fn poll_save(&mut self) {
        let Some(job) = &mut self.save_job else {
            return;
        };
        if let Some(out) = job.poll(&mut self.jobs) {
            self.save_job = None;
            self.save_done(out);
        }
    }

    /// `MainWindow::saveProjectFinished`, and what XMALab 3 does not: the document takes
    /// the saved state, a failure is reported and changes nothing (ledger U54).
    fn save_done(&mut self, out: saving::SaveOutcome) {
        let after = self.after_save.take();
        match out.result {
            Ok((mut saved, report)) => {
                // Ledger A210: the lines of the poses "RecomputeWhenSaving" computed on the
                // copy, which may not become the document.
                for l in saved.take_capped_lines() {
                    self.write_capped_line(l);
                }
                for f in report.not_written.iter().chain(&report.dropped) {
                    self.log
                        .error(format!("Not saved in {}: {f}", out.path.display()));
                }
                if matches!(out.kind, SaveKind::Save | SaveKind::SaveAs)
                    && let Some(doc) = self.session.doc.as_mut()
                {
                    let old = doc.path().map(Path::to_path_buf);
                    // `changes` also moves for what is not an undo step (a trial selected
                    // in the frame the save started, review 12c), which the revision does not
                    // see: that change is kept, and the document is marked saved at the
                    // revision written, so it is unmodified as it would be had no save run.
                    if doc.revision() == out.revision
                        && doc.changes() == self.save_changes
                        && !doc.step_open()
                    {
                        // Nothing changed meanwhile: the saved copy, with the creation
                        // date and the file name the save set, becomes the document.
                        *doc = saved;
                    } else {
                        doc.set_path(&out.path);
                        doc.mark_saved_at(out.revision);
                    }
                    let modified = doc.is_modified();
                    self.was_modified = modified;
                    self.autosave.reset();
                    if !modified {
                        self.remove_recovery(old.as_deref());
                        self.remove_recovery(Some(&out.path));
                    }
                }
                if let Some(a) = after {
                    self.actions.push(a);
                }
            }
            Err(e) => {
                let msg = format!("Could not save {}: {e}", out.path.display());
                self.log.error(msg.clone());
                self.dialogs.error(format!(
                    "{msg}\nThe previous file, if there was one, is unchanged."
                ));
            }
        }
    }

    // -----------------------------------------------------------------------------------
    // Autosave.

    /// The folder for the recovery file of a project never saved: the settings folder.
    fn recovery_dir(&self) -> Option<PathBuf> {
        self.settings_path
            .as_ref()
            .and_then(|p| p.parent().map(Path::to_path_buf))
    }

    /// The recovery file of the open document.
    #[must_use]
    pub fn recovery_file(&self) -> Option<PathBuf> {
        let doc = self.session.doc.as_ref()?;
        saving::recovery_path(doc.path(), self.recovery_dir().as_deref())
    }

    pub(super) fn remove_recovery(&self, project: Option<&Path>) {
        if let Some(r) = saving::recovery_path(project, self.recovery_dir().as_deref())
            && let Err(e) = saving::remove_recovery(&r)
        {
            self.log.error(format!(
                "Cannot remove the recovery file {}: {e}",
                r.display()
            ));
        }
    }

    /// Starts an autosave when one is due; asks for a repaint when one will be.
    pub(super) fn autosave_tick(&mut self, ctx: &egui::Context) {
        let Some(doc) = &self.session.doc else {
            return;
        };
        if !self.autosave_timer
            || self.save_job.is_some()
            || self.autosave.is_running()
            || doc.step_open()
        {
            return;
        }
        match self.autosave.due_in(
            self.settings.autosave_minutes,
            doc.is_modified(),
            doc.revision(),
        ) {
            Some(d) if d.is_zero() => {
                self.autosave_now();
            }
            Some(d) => ctx.request_repaint_after(d),
            None => {}
        }
    }

    /// Starts an autosave at once if the document has unsaved changes (the timer calls it;
    /// tests too). Returns whether one started.
    pub fn autosave_now(&mut self) -> bool {
        let Some(path) = self.recovery_file() else {
            return false;
        };
        let Some(doc) = &self.session.doc else {
            return false;
        };
        if !doc.is_modified() || doc.step_open() || self.autosave.is_running() {
            return false;
        }
        self.autosave.start(doc.clone(), path);
        true
    }

    /// Waits for a running autosave and handles its result (tests).
    pub fn wait_for_autosave(&mut self) {
        if let Some(out) = self.autosave.wait() {
            self.autosave_done(out);
        }
    }

    pub(super) fn poll_autosave(&mut self) {
        if let Some(out) = self.autosave.poll() {
            self.autosave_done(out);
        }
    }

    pub(super) fn autosave_done(&mut self, out: AutosaveOutcome) {
        match out.result {
            Ok(()) => {
                self.autosave.last_error = None;
                self.autosave.written = Some(out.revision);
                // Saved or closed meanwhile: the copy is not wanted.
                let still_wanted = self.session.doc.as_ref().is_some_and(Document::is_modified)
                    && self.recovery_file().as_deref() == Some(out.path.as_path());
                if !still_wanted && let Err(e) = saving::remove_recovery(&out.path) {
                    self.log.error(format!(
                        "Cannot remove the recovery file {}: {e}",
                        out.path.display()
                    ));
                }
            }
            Err(e) => {
                if self.autosave.last_error.as_deref() != Some(e.as_str()) {
                    self.log
                        .error(format!("Autosave to {} failed: {e}", out.path.display()));
                }
                self.autosave.last_error = Some(e);
            }
        }
    }
}
