//! [`TrackRun`]: a tracking run of any kind, the one type through which the window and
//! `xma track` start and step a run. The existing kinds run on [`Tracker`] (XMALab's
//! tracker); the multi-camera kinds ([`TrackKind::Multi`], [`TrackKind::MultiSelected`],
//! new in this program) on [`MultiTracker`] with the method's default parameters unless
//! others are given.

use xma_model::{CalibrationType, CppCompat};
use xma_vision::joint_tracking::MultiTrackParams;

use super::multi::{
    MultiMarkers, MultiRequest, MultiTracker, too_few_calibrated, too_few_shown, tracked_cameras,
};
use super::{StepReport, StopReason, TrackError, TrackKind, TrackRequest, Tracker, TrialImages};
use crate::document::Document;

/// A tracking run in progress, of any kind.
#[derive(Clone, Debug)]
pub enum TrackRun {
    /// One of XMALab's kinds (single, selected, all).
    Classic(Tracker),
    /// Multi-camera tracking.
    Multi {
        /// The request as given (its kind is [`TrackKind::Multi`] or
        /// [`TrackKind::MultiSelected`]).
        request: TrackRequest,
        /// Boxed: it is several times the size of the other variant.
        tracker: Box<MultiTracker>,
    },
}

impl TrackRun {
    /// Starts a run with the multi-camera method's default parameters. Nothing is tracked until [`TrackRun::step`].
    ///
    /// # Errors
    /// As [`Tracker::new`] and [`MultiTracker::new`].
    pub fn new(doc: &Document, request: TrackRequest) -> Result<TrackRun, TrackError> {
        Self::start(doc, request, MultiTrackParams::default())
    }

    /// Starts a run: the existing kinds as [`Tracker::new`], the multi-camera kinds as
    /// [`MultiTracker::new`] with `params`.
    ///
    /// # Errors
    /// As [`Tracker::new`] and [`MultiTracker::new`].
    pub fn start(
        doc: &Document,
        request: TrackRequest,
        params: MultiTrackParams,
    ) -> Result<TrackRun, TrackError> {
        let markers = match &request.kind {
            TrackKind::Multi => MultiMarkers::All,
            TrackKind::MultiSelected(list) => MultiMarkers::Selected(list.clone()),
            _ => return Tracker::new(doc, request).map(TrackRun::Classic),
        };
        let tracker = MultiTracker::new(
            doc,
            MultiRequest {
                trial: request.trial,
                markers,
                direction: request.direction,
                continuous: request.continuous,
                start_frame: request.start_frame,
                end_frame: request.end_frame,
                visible_cameras: request.visible_cameras.clone(),
                params,
            },
        )?;
        Ok(TrackRun::Multi {
            request,
            tracker: Box::new(tracker),
        })
    }

    /// The request the run was started with.
    #[must_use]
    pub fn request(&self) -> &TrackRequest {
        match self {
            TrackRun::Classic(t) => t.request(),
            TrackRun::Multi { request, .. } => request,
        }
    }

    /// The frame the run is at, 0-based.
    #[must_use]
    pub fn frame(&self) -> usize {
        match self {
            TrackRun::Classic(t) => t.frame(),
            TrackRun::Multi { tracker, .. } => tracker.frame(),
        }
    }

    /// Why the run ended, once it has.
    #[must_use]
    pub fn finished(&self) -> Option<&StopReason> {
        match self {
            TrackRun::Classic(t) => t.finished(),
            TrackRun::Multi { tracker, .. } => tracker.finished(),
        }
    }

    /// Number of frame steps done so far.
    #[must_use]
    pub fn steps_done(&self) -> usize {
        match self {
            TrackRun::Classic(t) => t.steps_done(),
            TrackRun::Multi { tracker, .. } => tracker.steps_done(),
        }
    }

    /// Ends the run before its next step. A step in progress is never cut short.
    pub fn stop(&mut self) {
        match self {
            TrackRun::Classic(t) => t.stop(),
            TrackRun::Multi { tracker, .. } => tracker.stop(),
        }
    }

    /// One frame step ([`Tracker::step`], [`MultiTracker::step`]).
    pub fn step(&mut self, doc: &mut Document, images: &mut dyn TrialImages) -> StepReport {
        match self {
            TrackRun::Classic(t) => t.step(doc, images),
            TrackRun::Multi { tracker, .. } => tracker.step(doc, images),
        }
    }

    /// Steps until the run ends ([`Tracker::run`], [`MultiTracker::run`]).
    pub fn run(
        &mut self,
        doc: &mut Document,
        images: &mut dyn TrialImages,
        progress: &mut dyn FnMut(&StepReport) -> bool,
    ) -> StepReport {
        match self {
            TrackRun::Classic(t) => t.run(doc, images, progress),
            TrackRun::Multi { tracker, .. } => tracker.run(doc, images, progress),
        }
    }

    /// The multi-camera tracker of a multi-camera run (what it has had, the cameras it
    /// tracks); `None` for the other kinds.
    #[must_use]
    pub fn multi(&self) -> Option<&MultiTracker> {
        match self {
            TrackRun::Classic(_) => None,
            TrackRun::Multi { tracker, .. } => Some(tracker),
        }
    }
}

/// Why multi-camera tracking cannot run on trial `trial` of `doc` with the cameras shown
/// as `visible` (a camera beyond the list counts as shown), as one sentence for the
/// interface; `None` when it can. The conditions are those [`MultiTracker::new`] refuses on
/// before it looks at the markers and frames requested: the XMALab 3 mode, a trial that is
/// missing, the Default trial, fewer than two calibrated cameras (with a pose for the
/// trial), fewer than two of them shown ([`tracked_cameras`]), or a trial without markers.
#[must_use]
pub fn multi_unavailable(doc: &Document, trial: usize, visible: &[bool]) -> Option<String> {
    if doc.compat() == CppCompat::XMALAB3 {
        return Some(
            "Multi-camera tracking is not part of XMALab 3, and this project is open in the \
             XMALab 3 compatibility mode."
                .to_owned(),
        );
    }
    let Some(t) = doc.project().trials.get(trial) else {
        return Some("Multi-camera tracking needs a trial.".to_owned());
    };
    if t.is_default {
        return Some("The Default trial holds no tracking data.".to_owned());
    }
    // As `MultiTracker::new` counts them (this is asked every time the window draws the
    // page; `tracked_cameras` copies no camera model).
    let count = |v: Vec<bool>| v.into_iter().filter(|&b| b).count();
    let n_cal = if doc.project().calibration_type == CalibrationType::None {
        0
    } else {
        count(tracked_cameras(doc, trial, &[]))
    };
    if n_cal < 2 {
        return Some(too_few_calibrated(n_cal));
    }
    let shown = count(tracked_cameras(doc, trial, visible));
    if shown < 2 {
        return Some(too_few_shown(shown, n_cal));
    }
    if t.markers.is_empty() {
        return Some("Multi-camera tracking needs markers in the trial.".to_owned());
    }
    None
}
