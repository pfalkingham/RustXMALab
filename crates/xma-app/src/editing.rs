//! The edit path: how the interface changes the project.
//!
//! Not a port of one C++ file. In XMALab 3 every widget writes to the model itself and then
//! calls the refreshes it needs (`MainWindow::redrawGL`, `PointsDockWidget::reloadListFromObject`,
//! `PlotWindow::updateMarkers`; `docs/cpp-editing.md` 1.2). Here a panel, a dialog or the
//! camera views raise an [`EditRequest`] through [`crate::panels::Cx::request`]; the
//! application handles the queue after the frame, in one place
//! (`XmaApp::handle_edit_requests`, app.rs), against the session's document:
//!
//! | Request | What the application does |
//! |---|---|
//! | [`EditRequest::Apply`], [`EditRequest::ApplyAll`] | `Document::apply` / `apply_all` (one undo step); the outcome's messages go to the Console; [`EditError::Rejected`] opens the error dialog with XMALab's text (`ErrorDialog::showErrorDialog`); `Io` also opens it; `Invalid` and `NotImplemented` go to the Console |
//! | [`EditRequest::ImageAdjustment`] | `Document::set_image_adjustment`: saved with the project and marks it modified, not an undo step, no refresh (ledger U46) |
//! | [`EditRequest::PlacePoint`] | `xma_engine::tracking::place_point` (one undo step); for Ctrl+click the marker it chose becomes the active one |
//! | [`EditRequest::StartTracking`] | opens one undo step, makes a [`xma_engine::tracking::TrackRun`] (XMALab's tracker, or for the multi-camera kinds the multi-camera tracker with its default parameters but for the four check boxes on the tracking page, [`crate::tracking_ui::multi_params`]), then one `TrackRun::step` per screen frame ([`TrackingRun`]) |
//! | [`EditRequest::StopTracking`] | `TrackRun::stop`: the run ends before its next step (also Escape) |
//! | [`EditRequest::RefinePoints`] | `xma_engine::tracking::refine_points`, one frame per screen frame inside one undo step ([`RefineRun`]), with the "Refining center" progress and its Cancel |
//! | [`EditRequest::Ask`] | opens the question as a modal dialog ([`crate::dialogs::ask`]); OK or Yes raises the requests and actions its answer gives |
//!
//! After anything that changed the project's data the application brings what shows it up to
//! date (`XmaApp::document_changed`): it notices that `Document::revision` moved, rebuilds
//! the project facts, keeps the per-trial view state and the selection in range, and raises
//! [`crate::state::Changes::document`] so that every panel gets `AppEvent::Changed` with it.
//!
//! The image access the tracker and the detector need ([`trial_images`]) reads through the
//! frame service (`crate::trial_images`): the image each camera shows at a frame, at the
//! source's depth, mirrored for a flipped camera, waiting for the decode.

use std::path::PathBuf;

use xma_engine::tracking::{
    Direction, Placement, RefineWhich, StepReport, StopReason, TrackRequest, TrackRun, TrialImages,
};
use xma_engine::{Edit, EditError};
use xma_model::AdjustedView;

use crate::frames::Frames;

/// One request to change the project. See the module documentation for what the
/// application does with each.
#[derive(Clone, Debug, PartialEq)]
pub enum EditRequest {
    /// One edit, one undo step named [`Edit::label`].
    Apply(Edit),
    /// Several edits as one undo step named `label`; the first error takes back all.
    ApplyAll { label: String, edits: Vec<Edit> },
    /// A view's scale and bias (E5, E6; ledger U46): the camera views' info frames
    /// (`Calibration`, `Digitization`) and the Detailed View (`Detail`).
    ImageAdjustment {
        camera: usize,
        view: AdjustedView,
        scale: f64,
        bias: f64,
    },
    /// A click that places a point (E1 to E3). Frame 0-based, image coordinates.
    PlacePoint {
        trial: usize,
        camera: usize,
        frame: usize,
        x: f64,
        y: f64,
        placement: Placement,
    },
    /// Starts a tracking run (E8 to E11). Ignored while a run is going on.
    StartTracking(TrackRequest),
    /// Stops the tracking run before its next step (Escape, the Stop button, a second press
    /// of the running button).
    StopTracking,
    /// "Refine center of selected Points by polynomial fit" (E21).
    RefinePoints {
        trial: usize,
        markers: Vec<usize>,
        /// Frames, 0-based, inclusive.
        first: usize,
        last: usize,
        which: RefineWhich,
    },
    /// Asks the user first (a confirmation, a choice, a name, a colour); the answer becomes
    /// the requests and actions of [`crate::dialogs::ask::Answered`]. Opened as a modal
    /// dialog by the application; nothing changes until it is answered.
    Ask(Box<crate::dialogs::ask::Ask>),
}

/// The ordered queue of edit requests, like [`crate::actions::ActionQueue`].
#[derive(Clone, Debug, Default)]
pub struct EditQueue {
    queue: Vec<EditRequest>,
}

impl EditQueue {
    pub fn push(&mut self, r: EditRequest) {
        self.queue.push(r);
    }

    /// Takes everything queued so far.
    pub fn take(&mut self) -> Vec<EditRequest> {
        std::mem::take(&mut self.queue)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    #[must_use]
    pub fn pending(&self) -> &[EditRequest] {
        &self.queue
    }
}

/// The undo-step name of a tracking run.
pub const TRACKING_STEP: &str = "Track Points";

/// A tracking run the application advances one [`TrackRun::step`] per screen frame, inside
/// one undo step opened when it started (`Document::begin_step`) and closed when it ends
/// (`Document::commit_step`). While it runs `AppState::disable_draw` is set, as XMALab 3
/// sets `State::disableDraw`, so clicks in the views place nothing and Undo, Redo and saving
/// are disabled.
#[derive(Debug)]
pub struct TrackingRun {
    pub tracker: TrackRun,
    /// The last report, for tests and the Progress title.
    pub last: Option<StepReport>,
    /// Points the run's steps wrote (`StepReport::tracked`), for the multi-camera summary.
    pub written: usize,
    /// Cells the run's steps left empty (`StepReport::left_empty`, multi-camera runs only),
    /// for the multi-camera summary.
    pub left_empty: usize,
    /// Earlier points the run's steps left as they were where they found none
    /// (`StepReport::kept_older`, multi-camera runs only), for the multi-camera summary.
    pub kept_older: usize,
    /// Earlier points the run's steps reset (`StepReport::removed_older`, multi-camera runs
    /// only), for the multi-camera summary.
    pub removed_older: usize,
    /// Points the run's steps placed by fitting two touching dots apart
    /// (`StepReport::fitted_apart`, multi-camera runs with `split_touching` only), for the
    /// multi-camera summary.
    pub fitted_apart: usize,
}

impl TrackingRun {
    /// A run just started.
    #[must_use]
    pub fn new(tracker: TrackRun) -> Self {
        Self {
            tracker,
            last: None,
            written: 0,
            left_empty: 0,
            kept_older: 0,
            removed_older: 0,
            fitted_apart: 0,
        }
    }

    /// True once the run has ended (by itself or by [`TrackRun::stop`]).
    #[must_use]
    pub fn finished(&self) -> bool {
        self.tracker.finished().is_some()
    }

    /// Counts a step's report.
    pub fn record(&mut self, report: StepReport) {
        self.written += report.tracked.len();
        self.left_empty += report.left_empty;
        self.kept_older += report.kept_older;
        self.removed_older += report.removed_older;
        self.fitted_apart += report.fitted_apart;
        self.last = Some(report);
    }
}

/// The Console lines for the end of a multi-camera run (new in this program): one line for
/// the run (frames covered, points written, points left empty, and, when either is not zero,
/// the earlier points kept where the run found none and those removed because they disagree
/// with the run's own points: `StepReport::kept_older`, `StepReport::removed_older`; and,
/// when not zero, the points placed by fitting two touching dots apart:
/// `StepReport::fitted_apart`), then
/// one line per marker
/// that was lost and not found again, naming the frame it was last found at. A run's
/// per-point lines (`StepReport::messages`, one per point refused because its dot was in
/// doubt) are not written in the window; `xma track` prints them. Empty for a run that did
/// no step. Frames and markers 1-based, as everywhere in the interface.
///
/// "Left empty" is what the engine counts (`StepReport::left_empty`, summed over the steps):
/// at each frame the run reached (not its start frame), the cells of the markers the run had
/// by then (`MultiTracker::has_had`) in the cameras it tracks (shown, calibrated, with a
/// pose: `MultiTracker::tracked_cameras`) that held no point after that step, not counting
/// cells marked untrackable (the run may never write there). "Lost" means the run lost it:
/// a marker the run has had whose last frame with a point in the run's own record
/// (`MultiTracker::run_holds`: its start points, the points it wrote, the protected points
/// it reached, in a camera it tracks), over the frames from the start to the frame reached,
/// is before the frame reached. Earlier points the run left as they were
/// (`MultiTrackParams::keep_unconfirmed`) do not count: a marker the run lost at frame 20
/// whose older points remain to the end is reported as lost after frame 20.
#[must_use]
pub fn multi_summary(doc: &xma_engine::Document, run: &TrackingRun) -> Vec<String> {
    let request = run.tracker.request();
    let steps = run.tracker.steps_done();
    let Some(multi) = run.tracker.multi() else {
        return Vec::new();
    };
    if steps == 0 {
        return Vec::new();
    }
    let Some(trial) = doc.project().trials.get(request.trial) else {
        return Vec::new();
    };
    let start = request.start_frame;
    let end = run.tracker.frame();
    let forward = request.direction == Direction::Forward;
    // Frames in run order, start first.
    let frames: Vec<usize> = if forward {
        (start..=end).collect()
    } else {
        (end..=start).rev().collect()
    };
    let mut lost = Vec::new();
    for m in (0..trial.markers.len()).filter(|&m| multi.has_had(m)) {
        let last_found = frames
            .iter()
            .copied()
            .rfind(|&f| multi.run_holds(doc, m, f));
        if let Some(f) = last_found
            && f != end
        {
            lost.push((m, f));
        }
    }
    let how = match run.tracker.finished() {
        Some(StopReason::Stopped) => ", stopped",
        _ => "",
    };
    let (a, b) = if forward { (start, end) } else { (end, start) };
    let mut first = format!(
        "Multi-camera tracking {}, frames {} to {}{how}: {} points written, {} left empty",
        if forward { "forward" } else { "backward" },
        a + 1,
        b + 1,
        run.written,
        run.left_empty,
    );
    if run.kept_older > 0 || run.removed_older > 0 {
        first.push_str(&format!(
            "; {} earlier points kept where the run found none, {} removed because they \
             disagree with the run's own points",
            run.kept_older, run.removed_older
        ));
    }
    if run.fitted_apart > 0 {
        first.push_str(&format!(
            "; {} points placed by fitting two touching dots apart",
            run.fitted_apart
        ));
    }
    first.push('.');
    let mut lines = vec![first];
    for (m, f) in lost {
        lines.push(format!(
            "Multi-camera tracking: marker {} ({}) was lost after frame {} and not found again.",
            m + 1,
            trial.markers[m].description,
            f + 1
        ));
    }
    lines
}

/// The Console text for the end of a run, if it says something.
#[must_use]
pub fn stop_message(reason: &StopReason) -> Option<String> {
    match reason {
        StopReason::Failed(text) => Some(text.clone()),
        StopReason::InvalidPoint => {
            Some("Tracking stopped: the tracked point is not valid.".into())
        }
        StopReason::SingleStep
        | StopReason::Boundary
        | StopReason::EndFrame
        | StopReason::Stopped => None,
    }
}

/// Where an [`EditError`] goes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ErrorReport {
    /// The error dialog (`ErrorDialog::showErrorDialog`) with this text.
    Dialog(String),
    /// A line in the Console.
    Console(String),
}

/// Ported rule: XMALab shows an error dialog where it refuses an edit
/// ([`EditError::Rejected`]); a file that cannot be read also gets the dialog. An edit
/// that does not apply or is not available yet goes to the Console.
#[must_use]
pub fn report_error(e: &EditError) -> ErrorReport {
    match e {
        EditError::Rejected(text) => ErrorReport::Dialog(text.clone()),
        EditError::Io { .. } => ErrorReport::Dialog(e.to_string()),
        EditError::Invalid(_) | EditError::NotImplemented(_) => ErrorReport::Console(e.to_string()),
    }
}

/// The files of each camera of a trial, taken before the document is borrowed for writing
/// (the tracker holds `&mut Document` while it reads images).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TrialFiles {
    pub trial: usize,
    /// Per camera, the files to read (`Session::video_files`).
    pub cameras: Vec<Vec<PathBuf>>,
    /// Per camera, whether its images are mirrored (`Camera::flipped`).
    pub flips: Vec<bool>,
}

/// The images of a trial for the tracker and the detector, read through the frame service
/// (`crate::trial_images::FrameImages`).
pub fn trial_images<'a>(
    frames: &'a mut Frames,
    files: &'a TrialFiles,
) -> Box<dyn TrialImages + 'a> {
    Box::new(crate::trial_images::FrameImages::new(frames, files))
}

/// The polynomial refinement (E21) in progress: one frame per screen frame, inside one undo
/// step opened when it started, with the "Refining center" progress and its Cancel (which
/// keeps the frames already refined, as XMALab 3 does). Each frame is shown while it is
/// refined and the frame shown before comes back at the end
/// (`MarkerTreeWidget::action_RefinePointsPolynomialFit_triggered`).
#[derive(Debug)]
pub struct RefineRun {
    pub trial: usize,
    pub markers: Vec<usize>,
    pub which: RefineWhich,
    /// The next frame to refine, 0-based.
    pub next: usize,
    /// The last frame, 0-based, inclusive.
    pub last: usize,
    /// The first frame (for the progress).
    pub first: usize,
    /// The frame shown before the run, shown again after it.
    pub frame_before: Option<usize>,
    /// Points refined so far.
    pub count: usize,
    /// The Progress panel's job.
    pub job: crate::jobs::JobHandle,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_go_where_xmalab_shows_them() {
        assert_eq!(
            report_error(&EditError::Rejected("No".into())),
            ErrorReport::Dialog("No".into())
        );
        assert!(matches!(
            report_error(&EditError::Invalid("x".into())),
            ErrorReport::Console(_)
        ));
        assert!(matches!(
            report_error(&EditError::NotImplemented("y")),
            ErrorReport::Console(_)
        ));
    }

    #[test]
    fn queue_keeps_order() {
        let mut q = EditQueue::default();
        q.push(EditRequest::StopTracking);
        q.push(EditRequest::Apply(Edit::ApplyDefaultToTrials));
        assert_eq!(q.pending().len(), 2);
        let v = q.take();
        assert_eq!(v[0], EditRequest::StopTracking);
        assert!(q.is_empty());
    }
}
