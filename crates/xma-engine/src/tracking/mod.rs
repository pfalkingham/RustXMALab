//! Marker detection and tracking on a document: placing a point, the tracking loop, the
//! polynomial point refinement and the Detection Settings preview.
//!
//! Ported from `ui/WizardDigitizationFrame.cpp` (the loop), `processing/MarkerTracking.cpp`
//! and `processing/MarkerDetection.cpp` (what each job reads from the project and writes
//! back, [`jobs`]). Merge-aware tracking (XMALab 3's `processing/MergeMonitor.cpp`, kinds 5
//! and 6) was removed from this program on 26 November 2026, superseded by the multi-camera
//! kind (parity ledger A226).
//! `docs/cpp-tracking.md` describes the C++; the image work itself is in
//! `xma_vision::{detection, tracking}`.
//!
//! [`multi`] is a tracking method new in this program (multi-camera tracking, all cameras
//! and markers chosen together), run by its own [`MultiTracker`], so that the existing
//! kinds and their tests are untouched. Its two kinds ([`TrackKind::Multi`],
//! [`TrackKind::MultiSelected`]) are requested like the others; [`TrackRun`] ([`run`])
//! starts and steps a run of any kind, and is what the window and `xma track` use.
//! [`Tracker`] refuses the multi-camera kinds.
//!
//! # How the loop differs from the C++ in structure
//!
//! The C++ loop is a chain of Qt signals with jobs on worker threads. Here one call of
//! [`Tracker::step`] does one whole frame step and returns: template cut at the start
//! frame, tracker for every (marker, camera) pair, detection at the points tracked, the
//! validity check. The caller (the window, once per screen frame, or
//! `xma track`) calls it until it reports a stop. Trackers and detections of a step run on
//! several threads; their results are written afterwards in a fixed order, so nothing
//! depends on thread timing.
//!
//! # Where the C++ result depends on job order or timing (`docs/cpp-tracking.md` 7)
//!
//! - **Job finish order** (T20): each finished job's `setPoint` recomputes the frame of its
//!   marker and, in XMALab 3, the first rigid body that lists it. 2D points, statuses,
//!   sizes, 3D points and mean sizes do not depend on the order; a body's pose can, when a
//!   body is not the first body of all its markers or has a virtual point driven by another
//!   body. Here the results are written in the order the jobs were started (marker by
//!   marker, camera by camera), which is one order the C++ can finish in. In the default
//!   mode every body that depends on a written marker is recomputed after each write
//!   (ledger A31), so the last write leaves every pose as a full recomputation gives it.
//! - **Shared job counters** (T6) and **input between steps** (T7): one step is one call,
//!   and the window starts nothing while a run goes on, so a step's check always runs, once,
//!   after all its jobs.
//! - **Stopping while jobs run** (T11, section 7 row 4, ledger A86): [`Tracker::stop`] takes effect
//!   between steps; a step is never cut short, so the step's points are always validated
//!   (the C++ skips that validation after a stop that arrives while jobs run).
//! - **Steps with nothing to track** (T8): a continuous run goes on through such frames as
//!   the C++ does, without its recursion.
//!
//! # Undo
//!
//! Nothing here opens an undo step for a whole run: the caller brackets a run with
//! [`Document::begin_step`] and [`Document::commit_step`], so a run is one undo step however
//! many frames it covers. [`place_point`] and [`refine_points`] are one step each.

use std::fmt;
use std::sync::Arc;

use xma_model::{DetectionMethod, MarkerStatus, Point2};
use xma_video::Frame;
use xma_vision::image::ImageU8;

use crate::document::Document;
use crate::edit::ops::check_trial;
use crate::edit::{EditError, FrameRange};

pub mod images;
pub mod jobs;
pub mod multi;
pub mod run;

pub use images::TrialSources;
pub use jobs::{
    AlgorithmImage, DetectionJob, DetectionResult, ImageRef, JobTemplate, TemplateImage,
    TrackerJob, TrackerResult, algorithm_image, algorithm_image_for, algorithm_image_in,
    detect_options, detect_preview, full_depth, marker_prediction, refine_point, track_compat,
};
pub use multi::{MultiMarkers, MultiRequest, MultiTracker, tracked_cameras};
pub use run::{TrackRun, multi_unavailable};

/// The images of a trial, as the tracker and detector read them.
///
/// The window implements this on its frame cache, `xma track` on the trial's video sources
/// ([`TrialSources`]).
pub trait TrialImages {
    /// The image camera `camera` shows at `frame` (0-based), at the depth the source holds,
    /// mirrored when the camera is flipped: what `VideoStream::getImage` holds in the C++
    /// after the frame is loaded.
    ///
    /// # Errors
    /// A message when the frame cannot be read.
    fn image(&mut self, camera: usize, frame: usize) -> Result<Arc<Frame>, String>;
}

/// Which markers a run tracks (the C++ `trackType`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TrackKind {
    /// Type 1, "single": one marker (the active one). An invalid point stops the run.
    Single(usize),
    /// Type 2, "selected": the markers selected in the Points panel when the run starts.
    Selected(Vec<usize>),
    /// Type 3, "all".
    All,
    /// Multi-camera tracking ([`multi`]), all markers. New in this program, no C++
    /// `trackType`; run by [`TrackRun`], refused by [`Tracker`].
    Multi,
    /// Multi-camera tracking, the markers selected in the Points panel when the run starts.
    MultiSelected(Vec<usize>),
}

impl TrackKind {
    /// The C++ `trackType`; 0 for the multi-camera kinds, which XMALab 3 does not have.
    #[must_use]
    pub fn code(&self) -> i32 {
        match self {
            TrackKind::Single(_) => 1,
            TrackKind::Selected(_) => 2,
            TrackKind::All => 3,
            TrackKind::Multi | TrackKind::MultiSelected(_) => 0,
        }
    }

    /// True for the multi-camera kinds, new in this program.
    #[must_use]
    pub fn is_multi(&self) -> bool {
        matches!(self, TrackKind::Multi | TrackKind::MultiSelected(_))
    }
}

/// Direction of a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Backward,
}

/// What a tracking button or shortcut asks for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrackRequest {
    pub trial: usize,
    pub kind: TrackKind,
    pub direction: Direction,
    /// False for the one-step buttons, true for the continuous ones.
    pub continuous: bool,
    /// The frame the run starts from (the frame on screen), 0-based.
    pub start_frame: usize,
    /// A continuous run stops after the step that reaches this frame (0-based), as a user
    /// stopping it there would; `None` runs to the trial's first or last frame. Not a C++
    /// control: `xma track --to` and the reference runs' `--end`. Equal to the start frame,
    /// the run does no step and ends with [`StopReason::EndFrame`]; behind the start in the
    /// run's direction, [`Tracker::new`] refuses it ([`TrackError::Invalid`]).
    pub end_frame: Option<usize>,
    /// Per camera, whether it is shown. Hidden cameras are not tracked, detected or
    /// validated (`docs/cpp-tracking.md` 3.10). A camera beyond the list counts as shown.
    pub visible_cameras: Vec<bool>,
}

/// Why a run ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// A one-step request did its step.
    SingleStep,
    /// The run reached the trial's end frame (forward) or start frame (backward).
    Boundary,
    /// The run reached [`TrackRequest::end_frame`].
    EndFrame,
    /// Type 1 only: the tracked point failed the validity check.
    InvalidPoint,
    /// The caller asked the run to stop ([`Tracker::stop`]).
    Stopped,
    /// A frame could not be read, or the request does not apply. The text is for the Console.
    Failed(String),
}

/// What one step did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepReport {
    /// The frame the run is at after the step, 0-based: the frame to show.
    pub frame: usize,
    /// `Some` when the run has ended; no further step does anything.
    pub finished: Option<StopReason>,
    /// Lines for the Console.
    pub messages: Vec<String>,
    /// The (marker, camera) pairs the step tracked, in the order they were written.
    pub tracked: Vec<(usize, usize)>,
    /// The (marker, camera) points the step detected again.
    pub detected: Vec<(usize, usize)>,
    /// The (marker, camera) points the step's check found invalid (each reset, with the
    /// other cameras below `SET` when the reprojection error was too large).
    pub invalid: Vec<(usize, usize)>,
    /// Multi-camera tracking only (0 for the other kinds): the cells at the step's frame of
    /// the markers the run has had by the end of the step, in the cameras it tracks, that
    /// hold no point after the step (status not above `UNDEFINED`), except cells marked
    /// `UNTRACKABLE`, where the run may never write.
    pub left_empty: usize,
    /// Multi-camera tracking only (0 for the other kinds): the cells at the step's frame that
    /// held a point before the step, got none from the run, and were left as they are
    /// (`MultiTrackParams::keep_unconfirmed`; module documentation of
    /// [`multi`], step 6).
    pub kept_older: usize,
    /// Multi-camera tracking only (0 for the other kinds): the cells at the step's frame that
    /// held a point before the step and were reset to undefined by it (with
    /// `keep_unconfirmed` on, those the run's own points contradict; with it off, every
    /// reset).
    pub removed_older: usize,
    /// Multi-camera tracking only (0 for the other kinds): the points written at the step's
    /// frame from a candidate fitted apart from a touching dot
    /// (`MultiTrackParams::split_touching`; module documentation of [`multi`], step 3).
    pub fitted_apart: usize,
}

/// Why a run could not start.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TrackError {
    /// An index is out of range or the request does not apply. The text is for a log.
    Invalid(String),
    /// XMALab writes this line to the Console and does not start.
    Refused(String),
    /// Not ported yet.
    NotImplemented(&'static str),
}

impl fmt::Display for TrackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrackError::Invalid(text) | TrackError::Refused(text) => f.write_str(text),
            TrackError::NotImplemented(what) => write!(f, "{what} is not available yet"),
        }
    }
}

impl std::error::Error for TrackError {}

/// A tracking run in progress: the request and the frame it has reached.
#[derive(Clone, Debug)]
pub struct Tracker {
    request: TrackRequest,
    frame: usize,
    finished: Option<StopReason>,
    steps_done: usize,
    /// The last step's end-frame images as the algorithms read them, so that the next step,
    /// whose start frame is that end frame, does not convert them again. Each is reused
    /// only for the very frame it was made from (the same shared [`Frame`] returned again
    /// for the same camera and frame, in the same depth mode); released when the run ends.
    kept: Vec<KeptImage>,
}

/// A converted image kept from one step to the next ([`Tracker::kept`]).
#[derive(Clone)]
struct KeptImage {
    camera: usize,
    frame: usize,
    /// The decoded frame the image was made from; held so that its identity cannot be
    /// taken by another frame while the image is kept.
    source: Arc<Frame>,
    full_depth: bool,
    image: Arc<AlgorithmImage>,
}

impl fmt::Debug for KeptImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KeptImage")
            .field("camera", &self.camera)
            .field("frame", &self.frame)
            .field("full_depth", &self.full_depth)
            .finish_non_exhaustive()
    }
}

impl Tracker {
    /// Starts a run. Nothing is tracked until [`Tracker::step`].
    ///
    /// Refused as the C++ refuses (`trackSinglePoint`); also refused, in both
    /// modes, a start frame outside the trial's frames (the C++ can reach one only through
    /// `docs/cpp-tracking.md` T5, and then tracks past the trial until it reads past its
    /// arrays).
    ///
    /// # Errors
    /// See [`TrackError`].
    pub fn new(doc: &Document, request: TrackRequest) -> Result<Tracker, TrackError> {
        let t = doc
            .project()
            .trials
            .get(request.trial)
            .ok_or_else(|| TrackError::Invalid(format!("no trial {}", request.trial)))?;
        if t.is_default {
            return Err(TrackError::Invalid(
                "the Default trial holds no tracking data".to_owned(),
            ));
        }
        let n_markers = t.markers.len();
        let first = t.start_frame.saturating_sub(1);
        let last = t
            .end_frame
            .saturating_sub(1)
            .min(t.n_frames.saturating_sub(1));
        if request.start_frame < first || request.start_frame > last {
            return Err(TrackError::Invalid(format!(
                "frame {} is outside the trial's frames {} to {}",
                request.start_frame + 1,
                first + 1,
                last + 1
            )));
        }
        // Review 11: an end frame behind the start in the run's direction is refused; the
        // stop test after each step is equality, so the run would otherwise go on to the
        // trial's boundary. An end frame equal to the start is a run of no steps
        // ([`Tracker::step`]).
        if let Some(end) = request.end_frame {
            let behind = match request.direction {
                Direction::Forward => end < request.start_frame,
                Direction::Backward => end > request.start_frame,
            };
            if behind {
                return Err(TrackError::Invalid(format!(
                    "end frame {} is behind the start frame {} in the run's direction",
                    end + 1,
                    request.start_frame + 1
                )));
            }
        }
        let check = |list: &[usize]| -> Result<(), TrackError> {
            match list.iter().find(|&&m| m >= n_markers) {
                Some(m) => Err(TrackError::Invalid(format!("no marker {m}"))),
                None => Ok(()),
            }
        };
        match &request.kind {
            // WizardDigitizationFrame.cpp:265 (T19: `<=` admits the count itself).
            TrackKind::Single(m) => check(&[*m])?,
            TrackKind::Selected(list) => check(list)?,
            TrackKind::All => {}
            TrackKind::Multi | TrackKind::MultiSelected(_) => {
                return Err(TrackError::Invalid(
                    "multi-camera tracking runs through TrackRun, not Tracker".to_owned(),
                ));
            }
        }
        Ok(Tracker {
            frame: request.start_frame,
            request,
            finished: None,
            steps_done: 0,
            kept: Vec::new(),
        })
    }

    /// The request the run was started with.
    #[must_use]
    pub fn request(&self) -> &TrackRequest {
        &self.request
    }

    /// The frame the run is at, 0-based.
    #[must_use]
    pub fn frame(&self) -> usize {
        self.frame
    }

    /// Why the run ended, once it has.
    #[must_use]
    pub fn finished(&self) -> Option<&StopReason> {
        self.finished.as_ref()
    }

    /// Number of frame steps done so far.
    #[must_use]
    pub fn steps_done(&self) -> usize {
        self.steps_done
    }

    /// Ends the run before its next step (Escape, the Stop button, a second press of the
    /// running button). A step in progress is never cut short.
    pub fn stop(&mut self) {
        if self.finished.is_none() {
            self.finished = Some(StopReason::Stopped);
        }
        self.kept.clear();
    }

    fn forward(&self) -> bool {
        self.request.direction == Direction::Forward
    }

    fn visible(&self, camera: usize) -> bool {
        self.request
            .visible_cameras
            .get(camera)
            .copied()
            .unwrap_or(true)
    }

    fn report(&self, messages: Vec<String>) -> StepReport {
        StepReport {
            frame: self.frame,
            finished: self.finished.clone(),
            messages,
            tracked: Vec::new(),
            detected: Vec::new(),
            invalid: Vec::new(),
            left_empty: 0,
            kept_older: 0,
            removed_older: 0,
            fitted_apart: 0,
        }
    }

    /// The trial's frame where the run stops (`track()`'s `==` tests).
    ///
    /// Ported from ui/WizardDigitizationFrame.cpp, WizardDigitizationFrame::track
    /// (`:710-719`).
    fn at_boundary(&self, doc: &Document) -> bool {
        let t = doc.trial(self.request.trial);
        if self.forward() {
            self.frame == t.end_frame.saturating_sub(1)
        } else {
            self.frame == t.start_frame.saturating_sub(1)
        }
    }

    /// One frame step, as described in the module documentation. Writes the tracked points
    /// into the document. After the run has ended this does nothing and reports the end.
    ///
    /// Ported from ui/WizardDigitizationFrame.cpp: `track` (`:696-744`) before and after the
    /// step, then `trackSinglePoint`, `trackRB`, `trackAll` (the trackers),
    /// `trackSinglePointFinished`, `trackAllFinished` (the detections) and `checkIfValid`
    /// (the check).
    pub fn step(&mut self, doc: &mut Document, images: &mut dyn TrialImages) -> StepReport {
        if self.finished.is_some() {
            return self.report(Vec::new());
        }
        // A run whose end frame is its start frame does no step (review 11).
        if self.steps_done == 0 && self.request.end_frame == Some(self.frame) {
            self.finished = Some(StopReason::EndFrame);
            return self.report(Vec::new());
        }
        // track() before the first step: a run started at its boundary does nothing.
        if self.steps_done == 0 && self.at_boundary(doc) {
            self.finished = Some(StopReason::Boundary);
            return self.report(Vec::new());
        }
        let mut report = match self.do_step(doc, images) {
            Ok(r) => r,
            Err(msg) => {
                self.finished = Some(StopReason::Failed(msg));
                return self.report(Vec::new());
            }
        };
        // track() after the step (and the test driver's stop at its end frame).
        if self.finished.is_none() {
            if self.request.end_frame == Some(self.frame) {
                self.finished = Some(StopReason::EndFrame);
            } else if !self.request.continuous {
                self.finished = Some(StopReason::SingleStep);
            } else if self.at_boundary(doc) {
                self.finished = Some(StopReason::Boundary);
            }
        }
        if self.finished.is_some() {
            self.kept.clear();
        }
        report.frame = self.frame;
        report.finished = self.finished.clone();
        report
    }

    /// The markers a kind tracks, in order.
    fn kind_markers(&self, doc: &Document) -> Vec<usize> {
        match &self.request.kind {
            TrackKind::Single(m) => vec![*m],
            TrackKind::Selected(v) => v.clone(),
            TrackKind::All => (0..doc.trial(self.request.trial).markers.len()).collect(),
            // Refused by `Tracker::new`.
            TrackKind::Multi | TrackKind::MultiSelected(_) => Vec::new(),
        }
    }

    /// The step itself; `Err` when an image cannot be read (nothing written then).
    #[allow(clippy::too_many_lines)]
    fn do_step(
        &mut self,
        doc: &mut Document,
        images: &mut dyn TrialImages,
    ) -> Result<StepReport, String> {
        // Taken now, so that a step that fails keeps nothing.
        let kept = std::mem::take(&mut self.kept);
        let trial = self.request.trial;
        let compat = doc.compat();
        let forward = self.forward();
        let s = self.frame;
        let n_frames = doc.trial(trial).n_frames;
        let e = if forward {
            s + 1
        } else {
            s.checked_sub(1).ok_or("no frame before the first")?
        };
        if e >= n_frames {
            return Err(format!("frame {} is past the trial's frames", e + 1));
        }
        let markers = self.kind_markers(doc);
        let settings = doc.settings();
        let r_max = if settings.retrack_optimized_tracked_points {
            MarkerStatus::TrackedAndOptimized
        } else {
            MarkerStatus::Tracked
        };
        let track_interpolated = settings.track_interpolated_points;
        let n_cameras = doc.project().cameras.len();

        // The pairs (WizardDigitizationFrame.cpp:278-282, :384-390, :496-500, :576-580).
        let mut pairs: Vec<(usize, usize)> = Vec::new(); // marker, camera
        for &m in &markers {
            for c in 0..n_cameras {
                let v = &doc.trial(trial).markers[m].views[c];
                let (from, to) = (v.status[s], v.status[e]);
                if self.visible(c)
                    && from > MarkerStatus::Undefined
                    && to > MarkerStatus::Untrackable
                    && to <= r_max
                    && !(to == MarkerStatus::Interpolated && !track_interpolated)
                {
                    pairs.push((m, c));
                }
            }
        }

        // The images: the start frame (each template's own frame), and the end frame of every
        // shown camera (State::changeActiveFrameTrial loads them all). Every image is still
        // asked of `images` as before; the conversion is taken from the last step's when
        // `images` returns the very frame it was made from.
        let full = jobs::full_depth(compat);
        let mut cache: Vec<KeptImage> = Vec::new();
        let mut load = |cache: &mut Vec<KeptImage>, c: usize, f: usize| -> Result<usize, String> {
            if let Some(i) = cache.iter().position(|k| (k.camera, k.frame) == (c, f)) {
                return Ok(i);
            }
            let source = images.image(c, f)?;
            let image = kept
                .iter()
                .find(|k| {
                    (k.camera, k.frame, k.full_depth) == (c, f, full)
                        && Arc::ptr_eq(&k.source, &source)
                })
                .map_or_else(
                    || Arc::new(jobs::algorithm_image_arc(&source, compat)),
                    |k| Arc::clone(&k.image),
                );
            cache.push(KeptImage {
                camera: c,
                frame: f,
                source,
                full_depth: full,
                image,
            });
            Ok(cache.len() - 1)
        };
        let mut jobs_t = Vec::with_capacity(pairs.len());
        for &(m, c) in &pairs {
            // The template is cut from the image of the frame being left (`s`), at the
            // position the marker has there (MarkerTracking.cpp:129-135).
            let i = load(&mut cache, c, s)?;
            jobs_t.push(TrackerJob::new(
                &doc.trial(trial).markers[m],
                m,
                c,
                s,
                e,
                forward,
                &cache[i].image,
            ));
        }
        let mut end_images: Vec<Option<usize>> = vec![None; n_cameras];
        if !pairs.is_empty() {
            for (c, slot) in end_images.iter_mut().enumerate() {
                if self.visible(c) {
                    *slot = Some(load(&mut cache, c, e)?);
                }
            }
        }

        // The trackers (MarkerTracking::trackMarker_thread), then their writes
        // (trackMarker_threadFinished: setPoint(.., TRACKED)), in the order started.
        let tc = jobs::track_compat(compat);
        // A source that changed kind between the two frames (module documentation of
        // [`jobs`]): the 8-bit end image taken onto the full-depth path once per camera,
        // not once per tracker; `TrackerJob::run` does the same with it.
        let end_deep: Vec<Option<AlgorithmImage>> = end_images
            .iter()
            .enumerate()
            .map(|(c, slot)| match slot.map(|i| &*cache[i].image) {
                Some(AlgorithmImage::U8(im))
                    if jobs_t.iter().any(|j| {
                        j.camera == c && matches!(j.template.image, TemplateImage::Deep(_))
                    }) =>
                {
                    Some(AlgorithmImage::Deep {
                        image: im.map(u16::from),
                        bit_depth: 8,
                    })
                }
                _ => None,
            })
            .collect();
        let results = {
            let t = doc.trial(trial);
            let cache = &cache;
            let end_images = &end_images;
            let end_deep = &end_deep;
            jobs::par_map(&jobs_t, |job| {
                let img = match (&job.template.image, &end_deep[job.camera]) {
                    (TemplateImage::Deep(_), Some(deep)) => deep,
                    _ => &*cache[end_images[job.camera].expect("loaded")].image,
                };
                job.run(&t.markers[job.marker], img, tc)
            })
        };
        drop(end_deep);
        let tracked: Vec<(usize, usize)> = jobs_t.iter().map(|j| (j.marker, j.camera)).collect();
        let writes: Vec<Write> = jobs_t
            .iter()
            .zip(&results)
            .map(|(job, r)| Write {
                marker: job.marker,
                camera: job.camera,
                point: r.point,
                status: MarkerStatus::Tracked,
                size: None,
            })
            .collect();
        doc.write_results(trial, e, &writes)
            .map_err(|err| err.to_string())?;

        // The detections (trackSinglePointFinished, trackAllFinished).
        // With no tracker the C++ goes straight to the check.
        let mut detect: Vec<(usize, usize)> = Vec::new();
        if !tracked.is_empty() {
            if compat.redetect_all_tracked {
                // Ledger A62: every TRACKED point of the kind's markers (type 2 as type 3,
                // its trackers being connected to trackAllFinished).
                let det_markers: Vec<usize> = match &self.request.kind {
                    TrackKind::Selected(_) | TrackKind::All => {
                        (0..doc.trial(trial).markers.len()).collect()
                    }
                    _ => markers.clone(),
                };
                for &m in &det_markers {
                    for c in 0..n_cameras {
                        if self.visible(c)
                            && doc.trial(trial).markers[m].views[c].status[e]
                                == MarkerStatus::Tracked
                        {
                            detect.push((m, c));
                        }
                    }
                }
            } else {
                detect.clone_from(&tracked);
            }
        }
        let wraps = compat.threshold_offset_wraps;
        let jobs_d: Vec<DetectionJob> = detect
            .iter()
            .map(|&(m, c)| {
                let mk = &doc.trial(trial).markers[m];
                DetectionJob::new(mk, m, c, e, mk.mean_size * 2.0, true, wraps)
            })
            .collect();
        for job in &jobs_d {
            if end_images[job.camera].is_none() {
                end_images[job.camera] = Some(load(&mut cache, job.camera, e)?);
            }
        }
        let options = jobs::detect_options(doc);
        let found = {
            let cache = &cache;
            let end_images = &end_images;
            jobs::par_map(&jobs_d, |job| {
                job.run(
                    &cache[end_images[job.camera].expect("loaded")].image,
                    &options,
                )
            })
        };
        let writes: Vec<Write> = jobs_d
            .iter()
            .zip(&found)
            .map(|(job, r)| Write {
                marker: job.marker,
                camera: job.camera,
                point: r.point,
                status: job.status,
                size: Some(r.size),
            })
            .collect();
        doc.write_results(trial, e, &writes)
            .map_err(|err| err.to_string())?;

        // The check (checkIfValid).
        let messages = Vec::new();
        let mut invalid = Vec::new();
        let check: Vec<(usize, usize)> = if compat.redetect_all_tracked {
            markers
                .iter()
                .flat_map(|&m| (0..n_cameras).map(move |c| (m, c)))
                .filter(|&(_, c)| self.visible(c))
                .collect()
        } else {
            tracked.clone()
        };
        let mut valid = true;
        for (m, c) in check {
            let mk = &doc.trial(trial).markers[m];
            if mk.views[c].status[e] == MarkerStatus::Tracked
                && mk.method != DetectionMethod::NoDetection
            {
                let ok = doc.is_valid_point(trial, m, c, e);
                if !ok {
                    invalid.push((m, c));
                }
                valid = ok && valid;
            }
        }
        if !valid && matches!(self.request.kind, TrackKind::Single(_)) {
            self.finished = Some(StopReason::InvalidPoint);
        }
        self.frame = e;
        self.steps_done += 1;
        // The end frame is the next step's start frame.
        cache.retain(|k| k.frame == e);
        self.kept = cache;
        Ok(StepReport {
            frame: e,
            finished: None,
            messages,
            tracked,
            detected: detect,
            invalid,
            left_empty: 0,
            kept_older: 0,
            removed_older: 0,
            fitted_apart: 0,
        })
    }

    /// Steps until the run ends, calling `progress` after every step; the run stops when
    /// `progress` returns false. Returns the last report.
    pub fn run(
        &mut self,
        doc: &mut Document,
        images: &mut dyn TrialImages,
        progress: &mut dyn FnMut(&StepReport) -> bool,
    ) -> StepReport {
        loop {
            let report = self.step(doc, images);
            if report.finished.is_some() {
                return report;
            }
            if !progress(&report) {
                self.stop();
            }
        }
    }
}

/// One job's result to write: the size first when there is one (`setSize`), then the
/// point and status (`setPoint`).
#[derive(Clone, Copy, Debug)]
struct Write {
    marker: usize,
    camera: usize,
    point: Point2,
    status: MarkerStatus,
    size: Option<f64>,
}

impl Document {
    /// Writes a batch of job results at one frame, as their "finished" handlers do.
    ///
    /// With XMALab 3's recomputation after a point edit (ledger A31: the first body only;
    /// A34: a degenerate triangulation keeps the old point), each result is written as the
    /// C++ handler writes it, `setSize` then `setPoint` with its recomputation, in the
    /// order given. Otherwise the result of that sequence does not depend on the order or on
    /// the intermediate recomputations, so the 2D data and sizes are written first, then each
    /// marker's mean size and 3D point once, then every body that depends on a written
    /// marker once (with the continuity step and the stale mark, as
    /// [`Document::reset_rigid_body_by_marker`] does), which gives the same state.
    ///
    /// Ported from processing/MarkerTracking.cpp, MarkerTracking::trackMarker_threadFinished
    /// (`:297-307`) and processing/MarkerDetection.cpp,
    /// MarkerDetection::detectMarker_threadFinished (`:428-441`).
    fn write_results(
        &mut self,
        trial: usize,
        frame: usize,
        writes: &[Write],
    ) -> Result<(), EditError> {
        let compat = self.compat();
        if writes.is_empty() {
            return Ok(());
        }
        if compat.edit_first_body_only || compat.triangulation_unchecked {
            for w in writes {
                if let Some(size) = w.size {
                    self.set_size(trial, w.marker, w.camera, frame, size)?;
                }
                self.set_point(trial, w.marker, w.camera, frame, w.point, w.status)?;
            }
            return Ok(());
        }
        for w in writes {
            crate::edit::ops::check_cell(self, trial, w.marker, w.camera, frame)?;
        }
        // Ledger A97: a run's writes move the change count like any other change.
        self.note_change();
        let mut markers: Vec<usize> = Vec::new();
        let mut sized: Vec<usize> = Vec::new();
        for w in writes {
            let m = self.marker_mut(trial, w.marker);
            let v = &mut m.views[w.camera];
            if let Some(size) = w.size {
                v.size[frame] = size;
                if !sized.contains(&w.marker) {
                    sized.push(w.marker);
                }
            }
            v.points[frame] = w.point;
            v.status[frame] = w.status;
            if !markers.contains(&w.marker) {
                markers.push(w.marker);
            }
        }
        for &m in &sized {
            crate::markers::update_mean_size(self.marker_mut(trial, m), compat.size_range_from_sum);
        }
        let ctx = self.frame_context(trial);
        let mut bodies: Vec<usize> = Vec::new();
        for &m in &markers {
            let patch =
                crate::markers::reconstruct_3d_point(&ctx, &self.trial(trial).markers[m], frame);
            patch.apply(self.marker_mut(trial, m));
            bodies.extend(crate::edit::ops::bodies_affected_by(self.trial(trial), m));
        }
        bodies.sort_unstable();
        bodies.dedup();
        if bodies.is_empty() {
            return Ok(());
        }
        self.recompute_bodies_at_frame(trial, frame, &bodies);
        if !compat.edits_not_refiltered {
            for &b in &bodies {
                self.continue_rotations(trial, b, frame);
            }
        }
        self.mark_bodies_stale(trial, &bodies, compat);
        Ok(())
    }

    /// `Marker::isValid(camera, frame)`: the tracked point's size (below 1, above 50, or
    /// more than 3 size ranges from the mean, ledger A1), position (outside the camera's
    /// image) and reprojection error (above `MaximumReprojectionError`). The first test that
    /// fails resets the point (`Marker::reset`); the error test resets every camera whose
    /// status at the frame is below `SET` instead (ledger A70). Returns whether all passed.
    ///
    /// Ported from core/Marker.cpp, Marker::isValid (`:1000-1040`).
    pub(crate) fn is_valid_point(
        &mut self,
        trial: usize,
        marker: usize,
        camera: usize,
        frame: usize,
    ) -> bool {
        let (size, mean, range, p, error, n_cam) = {
            let m = &self.trial(trial).markers[marker];
            let v = &m.views[camera];
            (
                v.size[frame],
                m.mean_size,
                m.size_range,
                v.points[frame],
                v.error[frame],
                m.n_cameras(),
            )
        };
        let cam = &self.project().cameras[camera];
        let (w, h) = (f64::from(cam.width), f64::from(cam.height));
        let bad_point = size < 1.0
            || size > 50.0
            || (size - mean).abs() > 3.0 * range
            || p.x < 0.0
            || p.x >= w
            || p.y < 0.0
            || p.y >= h;
        if bad_point {
            let _ = self.reset_point(trial, marker, camera, frame);
            return false;
        }
        if error > self.settings().maximum_reprojection_error {
            for c in 0..n_cam {
                if self.trial(trial).markers[marker].views[c].status[frame] < MarkerStatus::Set {
                    let _ = self.reset_point(trial, marker, c, frame);
                }
            }
            return false;
        }
        true
    }
}

/// How a click places a point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// Ctrl+click in a camera view (E1): the first marker from the active one on (marker 0
    /// when none is active) without a point in this camera at this frame (status exactly
    /// `UNDEFINED`), a new marker if there is none, then detection; status `SET`.
    AddNext {
        /// The active marker, where the search starts (`Trial::activeMarkerIdx`).
        active: Option<usize>,
    },
    /// Click in a camera view (E2): the given marker, then detection; status `SET`.
    Detect(usize),
    /// Click in the Detailed View (E3): the given marker exactly where clicked; status
    /// `MANUAL`.
    Manual(usize),
}

/// What [`place_point`] did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placed {
    /// The marker that received the point (for [`Placement::AddNext`], the one chosen or
    /// created; the interface makes it the active marker).
    pub marker: usize,
    /// True when a marker was created.
    pub marker_added: bool,
    /// The stored point.
    pub x: f64,
    pub y: f64,
}

/// Places a point from a click, as one undo step. Ported from
/// `WizardDigitizationFrame::addDigitizationPoint` and `moveDigitizationPoint` with the
/// `MarkerDetection` job they start (`docs/cpp-editing.md` 2.1, 2.2), and
/// `Trial::setActiveToNextUndefinedMarker` (`core/Trial.cpp:421-438`) for
/// [`Placement::AddNext`]: `setPoint(.., SET or MANUAL)`, then for a detecting placement
/// the detection with search area 30 (`setSize`, `setPoint(.., SET)`).
///
/// # Errors
/// See [`EditError`]; the document is unchanged on an error.
#[allow(clippy::too_many_arguments)]
pub fn place_point(
    doc: &mut Document,
    images: &mut dyn TrialImages,
    trial: usize,
    camera: usize,
    frame: usize,
    x: f64,
    y: f64,
    placement: Placement,
) -> Result<Placed, EditError> {
    check_trial(doc, trial)?;
    let t = doc.trial(trial);
    if camera >= doc.project().cameras.len() {
        return Err(EditError::Invalid(format!("no camera {camera}")));
    }
    if frame >= t.n_frames {
        return Err(EditError::Invalid(format!("no frame {frame}")));
    }
    let n_markers = t.markers.len();
    match placement {
        Placement::Detect(m) | Placement::Manual(m) if m >= n_markers => {
            return Err(EditError::Invalid(format!("no marker {m}")));
        }
        Placement::AddNext { active: Some(a) } if a > n_markers => {
            return Err(EditError::Invalid(format!("no marker {a}")));
        }
        _ => {}
    }
    let image = if matches!(placement, Placement::Manual(_)) {
        None
    } else {
        let f = images.image(camera, frame).map_err(EditError::Invalid)?;
        Some(jobs::algorithm_image_arc(&f, doc.compat()))
    };
    doc.transact("Set Point", |doc| {
        let (marker, added) = match placement {
            Placement::Detect(m) | Placement::Manual(m) => (m, false),
            Placement::AddNext { active } => {
                let t = doc.trial(trial);
                let mut i = active.unwrap_or(0);
                while i < t.markers.len()
                    && t.markers[i].views[camera].status[frame] != MarkerStatus::Undefined
                {
                    i += 1;
                }
                if i == t.markers.len() {
                    (crate::edit::body_ops::add_marker(doc, trial), true)
                } else {
                    (i, false)
                }
            }
        };
        let status = if matches!(placement, Placement::Manual(_)) {
            MarkerStatus::Manual
        } else {
            MarkerStatus::Set
        };
        doc.set_point(trial, marker, camera, frame, Point2::new(x, y), status)?;
        if let Some(image) = &image {
            let job = DetectionJob::new(
                &doc.trial(trial).markers[marker],
                marker,
                camera,
                frame,
                30.0,
                false,
                doc.compat().threshold_offset_wraps,
            );
            let r = job.run(image, &jobs::detect_options(doc));
            doc.set_size(trial, marker, camera, frame, r.size)?;
            doc.set_point(trial, marker, camera, frame, r.point, job.status)?;
        }
        let p = doc.trial(trial).markers[marker].views[camera].points[frame];
        Ok(Placed {
            marker,
            marker_added: added,
            x: p.x,
            y: p.y,
        })
    })
}

/// Which points the polynomial refinement refines (the choices of E21).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefineWhich {
    /// "set from Mainwindow + tracked".
    SetAndTracked,
    /// "only tracked".
    TrackedOnly,
    /// "All (including set from detailwindow)".
    All,
}

impl RefineWhich {
    /// The highest status refined.
    fn max_status(self) -> MarkerStatus {
        match self {
            RefineWhich::SetAndTracked => MarkerStatus::SetAndOptimized,
            RefineWhich::TrackedOnly => MarkerStatus::TrackedAndOptimized,
            RefineWhich::All => MarkerStatus::ManualAndOptimized,
        }
    }
}

/// "Refine center of selected Points by polynomial fit" (E21), as one undo step.
/// `progress` is called with each frame about to be refined; returning false stops before
/// that frame and keeps the frames already refined. Returns the number of points refined.
///
/// For each frame of the range, every camera, and each marker in the order given: a point
/// whose status is between `TRACKED` and the choice's highest status is refined with the
/// marker's mean size as radius (5 when it is -1) as a dark marker unless the method is 2
/// or 6; on success the status is raised to its `_AND_OPTIMIZED` form, the point written,
/// then the fitted radius written as the size (which changes the mean size the next point
/// starts from). Every camera reads its own image at the frame, hidden or not
/// (`docs/cpp-tracking.md` T17: the C++ refines a hidden camera on whatever image it last
/// loaded; that is not reproduced in either mode).
///
/// Ported from ui/MarkerTreeWidget.cpp,
/// MarkerTreeWidget::action_RefinePointsPolynomialFit_triggered (`:262-358`), with
/// processing/MarkerDetection.cpp, MarkerDetection::refinePointPolynomialFit
/// (`xma_vision::detection::refine_point_polynomial_fit`, or its full-depth form on a frame
/// deeper than 8 bits outside the XMALab 3 mode: [`jobs::refine_point`], ledger V1).
///
/// # Errors
/// See [`EditError`]; on an error (an index out of range, an image that cannot be read)
/// the document is unchanged.
pub fn refine_points(
    doc: &mut Document,
    images: &mut dyn TrialImages,
    trial: usize,
    markers: &[usize],
    frames: FrameRange,
    which: RefineWhich,
    progress: &mut dyn FnMut(usize) -> bool,
) -> Result<usize, EditError> {
    check_trial(doc, trial)?;
    for &m in markers {
        crate::edit::ops::check_frames(doc, trial, m, frames)?;
    }
    let max = which.max_status();
    let n_cameras = doc.project().cameras.len();
    let compat = jobs::detect_options(doc).compat;
    let mode = doc.compat();
    doc.transact("Refine Points", |doc| {
        let mut count = 0;
        if frames.first > frames.last {
            return Ok(0);
        }
        for i in frames.first..=frames.last {
            if !progress(i) {
                break;
            }
            for c in 0..n_cameras {
                let mut image: Option<AlgorithmImage> = None;
                for &m in markers {
                    let mk = &doc.trial(trial).markers[m];
                    let status = mk.views[c].status[i];
                    if status < MarkerStatus::Tracked || status > max {
                        continue;
                    }
                    let pt = mk.views[c].points[i];
                    let mut radius = mk.mean_size;
                    #[allow(clippy::float_cmp)]
                    if radius == -1.0 {
                        radius = 5.0;
                    }
                    let dark = !matches!(
                        mk.method,
                        DetectionMethod::WhiteMarker | DetectionMethod::WhiteBlob
                    );
                    if image.is_none() {
                        let f = images.image(c, i).map_err(EditError::Invalid)?;
                        image = Some(jobs::algorithm_image_arc(&f, mode));
                    }
                    let img = image.as_ref().expect("loaded");
                    let Some(r) = jobs::refine_point(img, (pt.x, pt.y), radius, dark, compat)
                    else {
                        continue;
                    };
                    let raised = match status {
                        MarkerStatus::Tracked => MarkerStatus::TrackedAndOptimized,
                        MarkerStatus::Set => MarkerStatus::SetAndOptimized,
                        MarkerStatus::Manual => MarkerStatus::ManualAndOptimized,
                        other => other,
                    };
                    doc.set_point(trial, m, c, i, Point2::new(r.point.0, r.point.1), raised)?;
                    doc.set_size(trial, m, c, i, r.radius)?;
                    count += 1;
                }
            }
        }
        Ok(count)
    })
}

/// The Detection Settings preview (`DetectionSettings::update`): the stage images of the
/// detection at a point, for a marker's method and threshold. Nothing is written.
#[derive(Clone, Debug, PartialEq)]
pub struct DetectionPreview {
    /// Side length of every stage image.
    pub size: usize,
    /// The stage images in the order of the dialog's combo box ("Input", "Blurred
    /// Orignial", ...), 8-bit, row by row. Six when no contour was selected, seven ("Detected")
    /// when one was (the C++ then indexes past its list if "Detected" is chosen).
    pub stages: Vec<Vec<u8>>,
    /// The detected point in image coordinates, when one was found.
    pub detected: Option<(f64, f64)>,
}

/// Computes the Detection Settings preview at a point: `detectionPoint` on the camera's
/// image at `frame` with search area 30 (the C++ passes 30.5 to an `int`), the input size
/// of the detection jobs and the marker's threshold offset as a signed `int` (this path
/// never wrapped, ledger A15), with the crosshair drawn when `crosshair` ("draw
/// crosshair"). `None` when the marker's method has no preview (the C++ shows "Method not
/// supported": methods other than 0, 2 and 5). A frame deeper than 8 bits is read at its
/// depth outside the XMALab 3 mode ([`jobs::detect_preview`], ledger V1); the stage images
/// are 8-bit in both paths.
///
/// Ported from ui/DetectionSettings.cpp, DetectionSettings::update (`:120-139`).
///
/// # Errors
/// A message when the frame cannot be read or an index is out of range.
#[allow(clippy::too_many_arguments)]
pub fn detection_preview(
    doc: &Document,
    images: &mut dyn TrialImages,
    trial: usize,
    marker: usize,
    camera: usize,
    frame: usize,
    x: f64,
    y: f64,
    crosshair: bool,
) -> Result<Option<DetectionPreview>, String> {
    let mk = doc
        .project()
        .trials
        .get(trial)
        .and_then(|t| t.markers.get(marker))
        .ok_or_else(|| format!("no marker {marker} in trial {trial}"))?;
    if camera >= mk.n_cameras() {
        return Err(format!("no camera {camera}"));
    }
    let method = match mk.method {
        DetectionMethod::XrayMarker => xma_vision::detection::DetectionMethod::Default,
        DetectionMethod::WhiteMarker => xma_vision::detection::DetectionMethod::White,
        DetectionMethod::XrayMarkerCenterOfMass => xma_vision::detection::DetectionMethod::Centroid,
        _ => return Ok(None),
    };
    let input_size = jobs::input_size(mk);
    let threshold = f64::from(mk.threshold_offset);
    let f = images.image(camera, frame)?;
    let image = jobs::algorithm_image_arc(&f, doc.compat());
    let (d, stages) = jobs::detect_preview(
        &image,
        method,
        (x, y),
        30,
        input_size as i32,
        threshold,
        crosshair,
        &jobs::detect_options(doc),
    );
    let found = stages.len() == 7;
    Ok(Some(DetectionPreview {
        size: stages.first().map_or(0, ImageU8::width),
        stages: stages.into_iter().map(ImageU8::into_vec).collect(),
        detected: found.then_some(d.point),
    }))
}
