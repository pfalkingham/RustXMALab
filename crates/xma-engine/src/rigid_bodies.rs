//! What the trial update and its finalize do for rigid bodies: the pose of every body at a
//! frame, rotation continuity, filtering in both modes, and the error measures.
//!
//! Ported from `core/RigidBody.cpp` (`computePose`, `transformPoint`, `updateError`,
//! `recomputeTransformations`, `makeRotationsContinous`, `filterTransformations`,
//! `filterData`), `core/Marker.cpp` (`filterMarker`, `filterData`) and
//! `processing/UpdateTrialFrame.cpp` (the body loop). The algorithms themselves are in
//! [`xma_vision::rigid`] and [`xma_vision::filter`]; this module gathers their inputs from
//! the model and keeps the C++ state rules: what is written, what is left stale, and which
//! flags are read when.
//!
//! # Frames are independent
//!
//! `RigidBody::computePose(frame)` reads, at that frame only, the markers' 3D points and
//! statuses and their 2D points and statuses, and, through virtual points, the other bodies'
//! pose flags and poses at that frame; it writes only that frame of this body (and, through
//! the on-demand computation in `transformPoint`, that frame of other bodies). The per-frame
//! functions here therefore work on a [`FrameStates`]: a copy of every body's state at one
//! frame, owned by the caller, so frames can be computed in any order or in parallel and
//! written back afterwards. No shared mutable state.
//!
//! # Virtual points that refer to another body
//!
//! `RigidBody::transformPoint` calls `computePose(frame)` on the other body when that body's
//! `poseComputed[frame]` is clear (`core/RigidBody.cpp:297-300`), then uses its pose if the
//! flag is now set. [`transform_point`] does the same on the [`FrameStates`]. When the bodies
//! refer to each other in a cycle (or a body to itself) the C++ recurses until the stack
//! overflows; here the inner call finds the body already being computed, does not recurse,
//! and reports no pose, and [`FrameStates::cycle`] is set so that the caller can warn.
//!
//! The C++ calls `transformPoint` once for the correspondences of `computePose`, then again
//! once per camera in the refinement's constructor (`addDummyPointsForOptimization`). Here
//! the world position is found once and used for both: the later calls find the other body
//! in the state the first call left, so they would compute the same position (a body that
//! failed is computed again and fails again, with the same writes), except in a cycle.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use xma_model::{FitError, MarkerStatus, Point3, Pose, RigidBody, Trial, VirtualPointSource};
use xma_vision::camera::RigidTransform;
use xma_vision::filter::{
    ButterworthLowPass, CutoffCorrection, FILTER_ORDER, can_filter, filter_run, runs,
};
use xma_vision::pose_refine::{RefineCamera, refine_pose, refinement_observations};
use xma_vision::rigid::{
    MarkerFrame, PoseCamera, PoseErrors, PoseSettings, RigidBodyFrame, View, VirtualPointFrame,
    body_to_world, compute_pose as vision_compute_pose, fit_rigid_transform,
    make_rotations_continuous as vision_make_rotations_continuous, pose_errors,
};

/// Everything the rigid-body step reads besides the trial. Immutable during an update; the
/// C++ reads the settings when each function runs, so build it at the start of each entry
/// point.
#[derive(Clone, Debug)]
pub struct PoseContext {
    /// One per camera, in project order, with the pose of the trial's reference calibration
    /// image.
    pub cameras: Vec<PoseCamera>,
    /// `DisableRBComputeAdvanced` and `OptimizeRigidBody`, and the refinement's faults to
    /// reproduce (ledger A3, A4).
    pub settings: PoseSettings,
    /// `Filter3DPoints`: the filter mode of `filterTransformations`.
    pub filter_3d_points: bool,
    /// How the low-pass filter's cutoff is corrected for its two passes (ledger A7;
    /// [`CutoffCorrection::SinglePass`] in the XMALab 3 mode).
    pub filter_correction: CutoffCorrection,
    /// True for a `NO_CALIBRATION` project.
    pub no_calibration: bool,
    /// Where [`compute_pose`] notes the frames whose pose from 2D rays had more than
    /// [`xma_vision::rigid::MAX_RAY_COMBINATIONS`] combinations (ledger A210). Records
    /// nothing when built with `Default` ([`Document::pose_context`] connects it to the
    /// document, which turns the frames into Console lines).
    ///
    /// [`Document::pose_context`]: crate::Document::pose_context
    pub capped: CappedFrames,
}

/// The frames noted as (trial, body, frame), 0-based, in order.
type CappedSet = BTreeSet<(usize, usize, usize)>;

/// Notes, for one trial, the frames whose pose from 2D rays was capped (ledger A210):
/// shared by every copy of a [`PoseContext`], so that frames computed anywhere (in
/// parallel too) reach the document's [`CappedLog`].
#[derive(Clone, Debug, Default)]
pub struct CappedFrames {
    sink: Option<(usize, Arc<Mutex<CappedSet>>)>,
}

impl CappedFrames {
    /// Notes that `body` at `frame` was capped. Does nothing when not connected.
    pub fn record(&self, body: usize, frame: usize) {
        if let Some((trial, set)) = &self.sink {
            lock(set).insert((*trial, body, frame));
        }
    }
}

fn lock(set: &Mutex<CappedSet>) -> std::sync::MutexGuard<'_, CappedSet> {
    set.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The frames a [`CappedLog`] held at some moment.
#[derive(Clone, Debug, Default)]
pub(crate) struct CappedSnapshot(pub(crate) CappedSet);

/// The document's list of capped frames (ledger A210), filled through the
/// [`CappedFrames`] of its pose contexts and emptied into Console lines. A clone is a
/// separate list with the same frames (a cloned document does not share it).
#[derive(Debug, Default)]
pub(crate) struct CappedLog(Arc<Mutex<CappedSet>>);

impl Clone for CappedLog {
    fn clone(&self) -> Self {
        CappedLog(Arc::new(Mutex::new(self.snapshot())))
    }
}

impl CappedLog {
    /// A recorder for one trial's pose context.
    pub(crate) fn recorder(&self, trial: usize) -> CappedFrames {
        CappedFrames {
            sink: Some((trial, Arc::clone(&self.0))),
        }
    }

    /// The frames noted so far.
    pub(crate) fn snapshot(&self) -> CappedSet {
        lock(&self.0).clone()
    }

    /// Puts back the frames of a [`CappedLog::snapshot`].
    pub(crate) fn restore(&self, set: CappedSet) {
        *lock(&self.0) = set;
    }

    /// Empties the list and returns what it held.
    pub(crate) fn take(&self) -> CappedSet {
        std::mem::take(&mut *lock(&self.0))
    }
}

/// One Console line for capped frames (ledger A210; [`crate::Document::take_capped_lines`]):
/// the trial and the rigid body it is about (0-based indices into the project as it was
/// when the line was taken), and its text. The indices let the interface tell apart lines
/// of trials or bodies with the same name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CappedLine {
    /// The trial's index.
    pub trial: usize,
    /// The rigid body's index in the trial.
    pub body: usize,
    /// The line, naming the body, the trial and the frames.
    pub text: String,
}

/// The Console lines for capped frames (ledger A210): one per (trial, body), listing its
/// frames 1-based with runs compressed ("frames 12, 15-230").
pub(crate) fn capped_lines(project: &xma_model::Project, capped: &CappedSet) -> Vec<CappedLine> {
    let mut by_body: std::collections::BTreeMap<(usize, usize), Vec<usize>> =
        std::collections::BTreeMap::new();
    for &(trial, body, frame) in capped {
        by_body.entry((trial, body)).or_default().push(frame);
    }
    by_body
        .into_iter()
        .map(|((trial, body), frames)| {
            let t = project.trials.get(trial);
            let body_name = match t.and_then(|t| t.rigid_bodies.get(body)) {
                Some(b) if !b.description.is_empty() => {
                    format!("Rigid body {} ({})", body + 1, b.description)
                }
                _ => format!("Rigid body {}", body + 1),
            };
            let trial_name = t.map_or_else(|| format!("{}", trial + 1), |t| t.name.clone());
            let (word, verb) = if frames.len() == 1 {
                ("frame", "has")
            } else {
                ("frames", "have")
            };
            CappedLine {
                trial,
                body,
                text: format!(
                    "{body_name}, trial {trial_name}: {word} {} {verb}{CAPPED_TAIL}",
                    frame_runs(&frames)
                ),
            }
        })
        .collect()
}

/// What ends every line of [`capped_lines`], after the frames and the verb.
const CAPPED_TAIL: &str = " more than 1,000,000 combinations of positions for the markers \
     seen by one camera only; those markers were not used for the pose there.";

/// Sorted 0-based frames as 1-based runs: `[11, 14, 15, 16]` gives "12, 15-17".
fn frame_runs(frames: &[usize]) -> String {
    let mut parts = Vec::new();
    let mut i = 0;
    while i < frames.len() {
        let mut j = i;
        while j + 1 < frames.len() && frames[j + 1] == frames[j] + 1 {
            j += 1;
        }
        if i == j {
            parts.push(format!("{}", frames[i] + 1));
        } else {
            parts.push(format!("{}-{}", frames[i] + 1, frames[j] + 1));
        }
        i = j + 1;
    }
    parts.join(", ")
}

/// One body's raw (unfiltered) state at one frame: `poseComputed[frame]`,
/// `rotationvectors[frame]` with `translationvectors[frame]`, and the four unfiltered errors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyFrameState {
    pub computed: bool,
    /// The stored pose; stale (or zero) when `computed` is false.
    pub pose: Pose,
    pub errors: FitError,
}

/// Every body's raw state at one frame, owned by the per-frame computation.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameStates {
    pub frame: usize,
    /// One per body, in index order.
    pub bodies: Vec<BodyFrameState>,
    /// True when a virtual point led back to a body already being computed at this frame
    /// (a cycle; stack overflow in the C++).
    pub cycle: bool,
    /// Bodies whose `computePose` is running, innermost last.
    active: Vec<usize>,
}

impl FrameStates {
    /// Reads every body's state at `frame` from the trial. A body shorter than the frame
    /// (only possible outside the trial update, which sizes them) reads as zero, as
    /// `addFrame` would create it.
    #[must_use]
    pub fn read(trial: &Trial, frame: usize) -> Self {
        let bodies = trial
            .rigid_bodies
            .iter()
            .map(|b| BodyFrameState {
                computed: b.raw.valid.get(frame).copied().unwrap_or(false),
                pose: b.raw.poses.get(frame).copied().unwrap_or(Pose::ZERO),
                errors: b.raw.errors.get(frame).copied().unwrap_or_default(),
            })
            .collect();
        Self {
            frame,
            bodies,
            cycle: false,
            active: Vec::new(),
        }
    }

    /// Writes every body's state back into the trial, growing a short body first
    /// (`addFrame`).
    pub fn write(&self, trial: &mut Trial) {
        for (body, s) in trial.rigid_bodies.iter_mut().zip(&self.bodies) {
            write_state(body, self.frame, s);
        }
    }

    /// Writes the state of the listed bodies only back into the trial (indices out of range
    /// are skipped), growing a short body first.
    pub fn write_bodies(&self, trial: &mut Trial, bodies: &[usize]) {
        for &b in bodies {
            if let (Some(body), Some(s)) = (trial.rigid_bodies.get_mut(b), self.bodies.get(b)) {
                write_state(body, self.frame, s);
            }
        }
    }
}

/// Writes one body's state at frame `f`, growing a short body first; the body is not
/// touched (nor copied) when it already holds `s` there ([`stored_equals`]).
fn write_state(body: &mut std::sync::Arc<RigidBody>, f: usize, s: &BodyFrameState) {
    if stored_equals(body, f, s) {
        return;
    }
    let body = std::sync::Arc::make_mut(body);
    body.grow_to(f + 1);
    body.raw.valid[f] = s.computed;
    body.raw.poses[f] = s.pose;
    body.raw.errors[f] = s.errors;
}

/// True when a body already holds `s` at frame `f`, bit for bit: a NaN error equals a NaN
/// with the same bits, and `-0.0` differs from `0.0`, so that writing is skipped exactly
/// when it would change nothing (`==` would always see NaN as changed and copy the body).
pub(crate) fn stored_equals(body: &RigidBody, f: usize, s: &BodyFrameState) -> bool {
    let pose = |p: &Pose| {
        [
            p.rotation.x,
            p.rotation.y,
            p.rotation.z,
            p.translation.x,
            p.translation.y,
            p.translation.z,
        ]
        .map(f64::to_bits)
    };
    let errors = |e: &FitError| [e.mean_2d, e.sd_2d, e.mean_3d, e.sd_3d].map(f64::to_bits);
    body.raw.valid.get(f) == Some(&s.computed)
        && body
            .raw
            .poses
            .get(f)
            .is_some_and(|p| pose(p) == pose(&s.pose))
        && body
            .raw
            .errors
            .get(f)
            .is_some_and(|e| errors(e) == errors(&s.errors))
}

fn to_transform(p: &Pose) -> RigidTransform {
    RigidTransform {
        rotation: p.rotation,
        translation: p.translation,
    }
}

fn to_pose(t: &RigidTransform) -> Pose {
    Pose {
        rotation: t.rotation,
        translation: t.translation,
    }
}

fn to_fit_error(e: PoseErrors) -> FitError {
    FitError {
        mean_2d: e.mean_2d,
        sd_2d: e.sd_2d,
        mean_3d: e.mean_3d,
        sd_3d: e.sd_3d,
    }
}

/// The body's markers and reference points at one frame, as `computePose`, the refinement,
/// `RigidBodyPoseFrom2D`, EPnP and `updateError` read them.
///
/// - `reference_points` is the working set (`RigidBody::points3D`, the optimised one when
///   loaded);
/// - a marker's 3D point is present when `status3D[frame] > UNDEFINED`;
/// - a view is present when `status2D[camera][frame] > UNDEFINED`, with the stored (raw)
///   point and `Camera::undistortPoint(point, true)`.
///
/// Where the C++ reads out of bounds (more marker indices than reference points or the
/// reverse, a marker index outside the trial, a frame past a marker's end) this reads the
/// common prefix, or a marker with no data.
#[must_use]
pub fn body_frame(
    ctx: &PoseContext,
    trial: &Trial,
    body: &RigidBody,
    frame: usize,
    virtual_points: Vec<VirtualPointFrame>,
) -> RigidBodyFrame {
    let n = body.marker_indices.len().min(body.reference_points.len());
    let markers = body.marker_indices[..n]
        .iter()
        .map(|&idx| {
            let Some(marker) = trial.markers.get(idx) else {
                return MarkerFrame {
                    point3d: None,
                    views: vec![None; ctx.cameras.len()],
                };
            };
            MarkerFrame {
                point3d: marker.point3d(frame),
                views: ctx
                    .cameras
                    .iter()
                    .enumerate()
                    .map(|(c, cam)| {
                        // Ledger A19 fixed: a point whose lens undistortion has no result is
                        // not used, as if the camera did not see it at this frame.
                        marker.point2d(c, frame).and_then(|raw| {
                            Some(View {
                                raw,
                                undistorted: cam
                                    .model
                                    .try_undistort_point(raw, true, true, true)?,
                            })
                        })
                    })
                    .collect(),
            }
        })
        .collect();
    RigidBodyFrame {
        reference_points: body.reference_points[..n].to_vec(),
        markers,
        virtual_points,
    }
}

/// The world position of a point given in another body's reference frame, computing that
/// body's pose first if its flag is clear.
///
/// Ported from core/RigidBody.cpp, RigidBody::transformPoint (`:295-317`, unfiltered):
/// `if (!poseComputed[frame]) computePose(frame);` then `Rᵀ (x − t)` when
/// `poseComputed[frame]`. Returns `None` where the C++ returns false. A body index outside
/// the trial (undefined behaviour in the C++) gives `None`; so does a cycle (see the module
/// documentation).
pub fn transform_point(
    ctx: &PoseContext,
    trial: &Trial,
    states: &mut FrameStates,
    body: usize,
    point: Point3,
) -> Option<Point3> {
    if body >= states.bodies.len() {
        return None;
    }
    if !states.bodies[body].computed {
        if states.active.contains(&body) {
            states.cycle = true;
            return None;
        }
        compute_pose(ctx, trial, states, body);
    }
    let s = &states.bodies[body];
    s.computed
        .then(|| body_to_world(&to_transform(&s.pose), point))
}

/// The world position of each virtual point of a body at the frame, in order, as
/// `computePose` finds them (`core/RigidBody.cpp:1094-1118`): another body's
/// [`transform_point`] (`dummyRBIndex >= 0`), else the per-frame coordinates when set
/// (`dummypointsCoordsSet`), else none.
pub fn virtual_point_frames(
    ctx: &PoseContext,
    trial: &Trial,
    states: &mut FrameStates,
    body: usize,
) -> Vec<VirtualPointFrame> {
    let frame = states.frame;
    trial.rigid_bodies[body]
        .virtual_points
        .iter()
        .map(|v| {
            let world = match &v.source {
                VirtualPointSource::OtherBody { body: other, point } => {
                    transform_point(ctx, trial, states, *other, *point)
                }
                VirtualPointSource::Coordinates(coords) => coords.get(frame).copied().flatten(),
            };
            VirtualPointFrame {
                reference: v.reference,
                world,
            }
        })
        .collect()
}

/// Computes one body's pose at the frame.
///
/// Ported from core/RigidBody.cpp, RigidBody::computePose (`:1065-1229`):
///
/// 1. `poseComputed[frame] = 0`;
/// 2. a `NO_CALIBRATION` project returns here, without updating the errors;
/// 3. a body that is not `initialised` computes nothing;
/// 4. otherwise the correspondences, the fallbacks and the refinement
///    ([`xma_vision::rigid::compute_pose`]) on the frame's marker data and the virtual
///    points' world positions ([`virtual_point_frames`], which may compute other bodies);
///    a pose found is stored, and where none is found the stored pose is left as it was;
/// 5. `poseComputed[frame]` is set to whether a pose was found, and `updateError(frame)`
///    writes the four unfiltered errors (zero without a pose).
pub fn compute_pose(ctx: &PoseContext, trial: &Trial, states: &mut FrameStates, body: usize) {
    states.bodies[body].computed = false;
    if ctx.no_calibration {
        return;
    }
    let rb = &trial.rigid_bodies[body];
    let frame = states.frame;
    states.active.push(body);
    let mut pose = None;
    let rb_frame = if rb.initialised {
        let virtual_points = virtual_point_frames(ctx, trial, states, body);
        let f = body_frame(ctx, trial, rb, frame, virtual_points);
        let outcome = vision_compute_pose(&f, &ctx.cameras, ctx.settings);
        if outcome.combinations_capped {
            ctx.capped.record(body, frame);
        }
        pose = outcome.pose;
        Some(f)
    } else {
        None
    };
    states.active.pop();

    let s = &mut states.bodies[body];
    if let Some(p) = pose {
        s.pose = to_pose(&p);
        s.computed = true;
    }
    s.errors = match (pose, rb_frame) {
        (Some(_), Some(f)) => to_fit_error(pose_errors(
            &f,
            &ctx.cameras,
            pose.as_ref(),
            ctx.settings.compat,
        )),
        _ => FitError::default(),
    };
}

/// The rigid-body part of one frame of the trial update.
///
/// Ported from processing/UpdateTrialFrame.cpp, UpdateTrialFrame::process (`:62-70`):
/// every body's `poseComputed[frame]` is cleared, then `computePose(frame)` runs for each
/// body in index order. Reads `trial` at this frame only (the markers must already hold
/// this frame's marker step) and returns every body's state at the frame.
#[must_use]
pub fn update_frame(ctx: &PoseContext, trial: &Trial, frame: usize) -> FrameStates {
    let mut states = FrameStates::read(trial, frame);
    for s in &mut states.bodies {
        s.computed = false;
    }
    for b in 0..states.bodies.len() {
        compute_pose(ctx, trial, &mut states, b);
    }
    states
}

/// Refines the pose a body holds in `states` once more, from that pose, and writes the
/// refined pose and its errors (`updateError`) into `states`. Returns false, changing
/// nothing, when the refinement is off, the body is not initialised or has no pose there.
/// The virtual points' world positions are taken from `states` ([`virtual_point_frames`]).
///
/// Not a step of any C++ function: this is what one interleaving of a C++ data race leaves
/// (parity ledger, item A56). While a trial update runs, the interface thread draws the
/// frame shown (`RigidBody::draw2D`, core/RigidBody.cpp, `:2669-2700`), and a visible body's
/// virtual point that refers to another body calls `transformPoint` on it (`:2692`), which
/// runs `computePose` when that body's flag is clear (`:297-300`), concurrently with the
/// worker thread computing the same body and frame. When one thread's refinement is set up
/// (`RigidBodyPoseOptimization`'s constructor reads the held pose) after the other has
/// written its refined pose, it refines that pose again. `xma parity` replays it to
/// identify such values in a reference dump.
pub fn refine_again(
    ctx: &PoseContext,
    trial: &Trial,
    states: &mut FrameStates,
    body: usize,
) -> bool {
    let Some(rb) = trial.rigid_bodies.get(body) else {
        return false;
    };
    if !ctx.settings.optimize || !rb.initialised || !states.bodies[body].computed {
        return false;
    }
    let virtual_points = virtual_point_frames(ctx, trial, states, body);
    let f = body_frame(ctx, trial, rb, states.frame, virtual_points);
    let observations = refinement_observations(&f, &ctx.cameras, ctx.settings.compat);
    let cameras: Vec<RefineCamera> = ctx
        .cameras
        .iter()
        .map(|c| RefineCamera::from_pose_camera(c, ctx.settings.compat))
        .collect();
    let pose = refine_pose(
        &to_transform(&states.bodies[body].pose),
        &observations,
        &cameras,
        ctx.settings.compat,
    );
    let s = &mut states.bodies[body];
    s.pose = to_pose(&pose);
    s.errors = to_fit_error(pose_errors(
        &f,
        &ctx.cameras,
        Some(&pose),
        ctx.settings.compat,
    ));
    true
}

/// Ported from core/RigidBody.cpp, RigidBody::makeRotationsContinous (`:1594-1613`), on
/// frames `0..n_frames` of the raw poses (through
/// [`xma_vision::rigid::make_rotations_continuous`]).
pub fn make_rotations_continuous(trial: &mut Trial, body: usize) {
    let n = trial.n_frames;
    let rb = std::sync::Arc::make_mut(&mut trial.rigid_bodies[body]);
    rb.grow_to(n);
    let mut rotations: Vec<_> = rb.raw.poses[..n].iter().map(|p| p.rotation).collect();
    vision_make_rotations_continuous(&mut rotations, &rb.raw.valid[..n]);
    for (p, r) in rb.raw.poses[..n].iter_mut().zip(rotations) {
        p.rotation = r;
    }
}

/// Every frame of one body, in order, then the continuity step.
///
/// Ported from core/RigidBody.cpp, RigidBody::recomputeTransformations (`:1584-1592`).
/// Unlike the trial update, other bodies' flags are not cleared first, so a virtual point
/// that refers to another body uses that body's stored pose when its flag is set, and
/// computes it only when the flag is clear (parity ledger, item A36). Returns whether any
/// frame met a cycle of virtual points.
pub fn recompute_transformations(ctx: &PoseContext, trial: &mut Trial, body: usize) -> bool {
    let n = trial.n_frames;
    for rb in &mut trial.rigid_bodies {
        if rb.n_frames() < n {
            std::sync::Arc::make_mut(rb).grow_to(n);
        }
    }
    let all: Vec<FrameStates> = (0..n)
        .map(|frame| {
            let mut states = FrameStates::read(trial, frame);
            compute_pose(ctx, trial, &mut states, body);
            states
        })
        .collect();
    let mut cycle = false;
    for states in &all {
        states.write(trial);
        cycle |= states.cycle;
    }
    make_rotations_continuous(trial, body);
    cycle
}

/// The cutoff a body is filtered with: its own when it overrides the trial's.
///
/// Ported from core/RigidBody.cpp (`:1702`, `:1659`, `:1912`).
#[must_use]
pub fn body_cutoff(trial: &Trial, body: &RigidBody) -> f64 {
    if body.override_cutoff_frequency {
        body.cutoff_frequency
    } else {
        trial.cutoff_frequency
    }
}

/// Low-pass filters a marker's 3D series.
///
/// Ported from core/Marker.cpp, Marker::filterMarker (`:1116-1162`) and Marker::filterData
/// (`:1515-1569`): the output has one entry per input status, (-1000, -1000, -1000) and
/// `UNDEFINED`; unless the cutoff is refused ([`can_filter`], with the trial's recording
/// speed), every run of consecutive frames `0..n_frames` with status above 0 is copied
/// (12 frames or fewer, ledger A8) or filtered with the order 4 Butterworth on x, y and z,
/// with the input statuses copied. Returns `None` where the C++ returns false (the outputs
/// are then the placeholders, as the C++ leaves them); the caller decides whether to use
/// them. The filter is designed with `correction` (ledger A7; the refusal test is made on
/// the entered cutoff either way).
///
/// Returns the filtered points, the filtered statuses, and whether filtering went ahead.
#[must_use]
pub fn filter_marker(
    cutoff: f64,
    recording_speed: f64,
    n_frames: usize,
    points: &[Point3],
    status: &[MarkerStatus],
    correction: CutoffCorrection,
) -> (Vec<Point3>, Vec<MarkerStatus>, bool) {
    let mut out_points = vec![Point3::new(-1000.0, -1000.0, -1000.0); status.len()];
    let mut out_status = vec![MarkerStatus::Undefined; status.len()];
    if !can_filter(cutoff, recording_speed) {
        return (out_points, out_status, false);
    }
    let n = n_frames.min(status.len()).min(points.len());
    let valid: Vec<bool> = status[..n].iter().map(|s| s.has_data()).collect();
    let filter = ButterworthLowPass::zero_phase(FILTER_ORDER, cutoff, recording_speed, correction);
    for run in runs(&valid) {
        let x: Vec<f64> = points[run.clone()].iter().map(|p| p.x).collect();
        let y: Vec<f64> = points[run.clone()].iter().map(|p| p.y).collect();
        let z: Vec<f64> = points[run.clone()].iter().map(|p| p.z).collect();
        let (x, y, z) = (
            filter_run(&filter, &x),
            filter_run(&filter, &y),
            filter_run(&filter, &z),
        );
        for (k, f) in run.enumerate() {
            out_points[f] = Point3::new(x[k], y[k], z[k]);
            out_status[f] = status[f];
        }
    }
    (out_points, out_status, true)
}

/// Filters one body's poses and updates the filtered errors of every frame.
///
/// Ported from core/RigidBody.cpp, RigidBody::filterTransformations (`:1699-1949`) and
/// RigidBody::filterData (`:1615-1697`). The cutoff is [`body_cutoff`].
///
/// Pose mode (`Filter3DPoints` off): every `poseFiltered` is cleared; unless the cutoff is
/// refused, every run of frames with `poseComputed` is copied (12 frames or fewer, ledger
/// A8) or filtered component by component, tx, ty, tz and the three Rodrigues components
/// (ledger A9), and its frames get `poseFiltered = 1`. Frames outside the runs keep their
/// old filtered pose.
///
/// `Filter3DPoints` mode ([`filter_3d_points`]): the markers and virtual points are filtered
/// and the pose refitted per frame (ledger A32).
///
/// Both end with `updateError(frame, true)` for every frame: the filtered pose against the
/// unfiltered marker positions, zeros without a filtered pose. Returns whether a cycle of
/// virtual points was met.
pub fn filter_transformations(ctx: &PoseContext, trial: &mut Trial, body: usize) -> bool {
    let n = trial.n_frames;
    std::sync::Arc::make_mut(&mut trial.rigid_bodies[body]).grow_to(n);
    let mut cycle = false;
    if ctx.filter_3d_points {
        cycle = filter_3d_points(ctx, trial, body);
    } else {
        let cutoff = body_cutoff(trial, &trial.rigid_bodies[body]);
        let speed = trial.recording_speed;
        let rb = std::sync::Arc::make_mut(&mut trial.rigid_bodies[body]);
        for v in &mut rb.filtered.valid[..n] {
            *v = false;
        }
        if can_filter(cutoff, speed) {
            let filter =
                ButterworthLowPass::zero_phase(FILTER_ORDER, cutoff, speed, ctx.filter_correction);
            for run in runs(&rb.raw.valid[..n]) {
                let raw = &rb.raw.poses[run.clone()];
                let component = |get: &dyn Fn(&Pose) -> f64| -> Vec<f64> {
                    filter_run(&filter, &raw.iter().map(get).collect::<Vec<_>>())
                };
                let t: [Vec<f64>; 3] = std::array::from_fn(|k| component(&|p| p.translation[k]));
                let r: [Vec<f64>; 3] = std::array::from_fn(|k| component(&|p| p.rotation[k]));
                for (i, f) in run.enumerate() {
                    let p = &mut rb.filtered.poses[f];
                    for k in 0..3 {
                        p.translation[k] = t[k][i];
                        p.rotation[k] = r[k][i];
                    }
                    rb.filtered.valid[f] = true;
                }
            }
        }
    }
    update_filtered_errors(ctx, trial, body);
    cycle
}

/// `updateError(frame, true)` for every frame `0..n_frames`.
///
/// Ported from core/RigidBody.cpp, RigidBody::filterTransformations (`:1945-1948`) and
/// RigidBody::updateError with `filtered` true.
fn update_filtered_errors(ctx: &PoseContext, trial: &mut Trial, body: usize) {
    let n = trial.n_frames;
    let errors: Vec<FitError> = {
        let rb = &trial.rigid_bodies[body];
        (0..n)
            .map(|frame| {
                if !rb.filtered.valid[frame] {
                    return FitError::default();
                }
                let f = body_frame(ctx, trial, rb, frame, Vec::new());
                let pose = to_transform(&rb.filtered.poses[frame]);
                to_fit_error(pose_errors(
                    &f,
                    &ctx.cameras,
                    Some(&pose),
                    ctx.settings.compat,
                ))
            })
            .collect()
    };
    let rb = std::sync::Arc::make_mut(&mut trial.rigid_bodies[body]);
    rb.filtered.errors[..n].copy_from_slice(&errors);
}

/// The `Filter3DPoints` mode of `filterTransformations`, without the error update.
///
/// Ported from core/RigidBody.cpp, RigidBody::filterTransformations (`:1701-1904`):
///
/// 1. each body marker's 3D series through [`filter_marker`]; a refused cutoff leaves every
///    frame undefined, so no frame is refitted and the body has no filtered pose, as in
///    pose mode (ledger A32: the refusal is all or nothing, one cutoff and one recording
///    speed serving every marker and virtual point of the body, so this is not a defect);
/// 2. each virtual point's world series, frame by frame: another body's [`transform_point`]
///    with its **unfiltered** pose (computing it where its flag is clear), or the stored
///    coordinates, through `cv::Point3f` (ledger A23: only in the XMALab 3 mode), status
///    `SET`; else (-2, -2, -2) and `UNDEFINED`;
///    then [`filter_marker`] (the C++ calls it on `getMarkers()[pointsIdx[i]]` with the
///    virtual point's index, which only reaches the trial; out of range it is undefined
///    behaviour, and here the trial is used directly);
/// 3. per frame, the rigid fit ([`fit_rigid_transform`], the same arithmetic as the inline
///    copy) of the filtered marker points with status above `UNDEFINED` and the filtered
///    virtual points with status non-zero, through `cv::Point3f` (XMALab 3 mode), onto their
///    reference
///    positions: with 3 or more, the filtered pose and `poseFiltered = true`; otherwise
///    `poseFiltered = false` and the filtered pose is left as it was. No 2D fallback, EPnP
///    or refinement;
/// 4. the 150 degree continuity rule on the filtered rotation vectors.
fn filter_3d_points(ctx: &PoseContext, trial: &mut Trial, body: usize) -> bool {
    let n = trial.n_frames;
    let speed = trial.recording_speed;
    let cutoff = body_cutoff(trial, &trial.rigid_bodies[body]);

    let rb = std::sync::Arc::clone(&trial.rigid_bodies[body]);
    let n_markers = rb.marker_indices.len().min(rb.reference_points.len());

    // 1. Markers.
    let markers: Vec<(Vec<Point3>, Vec<MarkerStatus>)> = rb.marker_indices[..n_markers]
        .iter()
        .map(|&idx| match trial.markers.get(idx) {
            Some(m) => {
                let (p, s, _) = filter_marker(
                    cutoff,
                    speed,
                    n,
                    &m.points3d,
                    &m.status3d,
                    ctx.filter_correction,
                );
                (p, s)
            }
            None => (Vec::new(), Vec::new()),
        })
        .collect();

    // 2. Virtual points, frame by frame (transformPoint may compute other bodies).
    let mut cycle = false;
    let mut series: Vec<(Vec<Point3>, Vec<MarkerStatus>)> =
        vec![(Vec::with_capacity(n), Vec::with_capacity(n)); rb.virtual_points.len()];
    for frame in 0..n {
        let mut states = FrameStates::read(trial, frame);
        for (i, v) in rb.virtual_points.iter().enumerate() {
            let world = match &v.source {
                VirtualPointSource::OtherBody { body: other, point } => {
                    transform_point(ctx, trial, &mut states, *other, *point)
                }
                VirtualPointSource::Coordinates(coords) => coords.get(frame).copied().flatten(),
            };
            let (p, s) = match world {
                Some(w) => (ctx.settings.compat.round(w), MarkerStatus::Set),
                None => (Point3::new(-2.0, -2.0, -2.0), MarkerStatus::Undefined),
            };
            series[i].0.push(p);
            series[i].1.push(s);
        }
        cycle |= states.cycle;
        states.write(trial);
    }
    let virtuals: Vec<(Vec<Point3>, Vec<MarkerStatus>)> = series
        .iter()
        .map(|(p, s)| {
            let (p, s, _) = filter_marker(cutoff, speed, n, p, s, ctx.filter_correction);
            (p, s)
        })
        .collect();

    // 3. The refit per frame.
    let fits: Vec<Option<Pose>> = (0..n)
        .map(|frame| {
            let mut src = Vec::new();
            let mut dst = Vec::new();
            for (i, (p, s)) in markers.iter().enumerate() {
                if s.get(frame).is_some_and(|s| s.has_data()) {
                    src.push(ctx.settings.compat.round(p[frame]));
                    dst.push(ctx.settings.compat.round(rb.reference_points[i]));
                }
            }
            for (v, (p, s)) in rb.virtual_points.iter().zip(&virtuals) {
                if s[frame] != MarkerStatus::Undefined {
                    src.push(ctx.settings.compat.round(p[frame]));
                    dst.push(ctx.settings.compat.round(v.reference));
                }
            }
            if dst.len() >= 3 {
                fit_rigid_transform(&src, &dst).map(|t| to_pose(&t))
            } else {
                None
            }
        })
        .collect();

    let rb = std::sync::Arc::make_mut(&mut trial.rigid_bodies[body]);
    rb.grow_to(n);
    for (frame, fit) in fits.into_iter().enumerate() {
        match fit {
            Some(p) => {
                rb.filtered.poses[frame] = p;
                rb.filtered.valid[frame] = true;
            }
            None => rb.filtered.valid[frame] = false,
        }
    }

    // 4. Continuity on the filtered rotations.
    let mut rotations: Vec<_> = rb.filtered.poses[..n].iter().map(|p| p.rotation).collect();
    vision_make_rotations_continuous(&mut rotations, &rb.filtered.valid[..n]);
    for (p, r) in rb.filtered.poses[..n].iter_mut().zip(rotations) {
        p.rotation = r;
    }
    cycle
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::Arc;
    use xma_model::{Marker, Matrix3, Vector3, VirtualPoint};
    use xma_vision::camera::{CameraModel, Intrinsics};

    pub(crate) fn camera(tx: f64) -> PoseCamera {
        let mut k = Matrix3::identity();
        k[(0, 0)] = 1000.0;
        k[(1, 1)] = 1000.0;
        k[(0, 2)] = 512.0;
        k[(1, 2)] = 512.0;
        PoseCamera {
            model: CameraModel {
                compat: xma_vision::camera::ModelCompat::XMALAB3,
                intrinsics: Intrinsics {
                    camera_matrix: k,
                    distortion: None,
                },
                grid: None,
            },
            pose: RigidTransform {
                rotation: Vector3::new(0.0, 0.1 * tx, 0.0),
                translation: Vector3::new(tx, 0.0, 50.0),
            },
            calibrated: true,
            reference_image_calibrated: true,
        }
    }

    fn context(disable_advanced: bool) -> PoseContext {
        PoseContext {
            cameras: vec![camera(-5.0), camera(5.0)],
            settings: PoseSettings {
                disable_advanced,
                optimize: false,
                ..PoseSettings::default()
            },
            filter_3d_points: false,
            filter_correction: CutoffCorrection::default(),
            no_calibration: false,
            capped: CappedFrames::default(),
        }
    }

    /// The rigid motion of every body: a turn about z of 0.01 rad per frame and a shift of
    /// 0.1 along x per frame. At frame 0 world and reference coordinates coincide.
    pub(crate) fn moved(p: Point3, frame: usize) -> Point3 {
        let a = 0.01 * frame as f64;
        let (s, c) = a.sin_cos();
        Point3::new(
            c * p.x - s * p.y + 0.1 * frame as f64,
            s * p.x + c * p.y,
            p.z,
        )
    }

    /// Base positions: markers 0 to 3 for body A, 4 and 5 for body B.
    pub(crate) const BASE: [[f64; 3]; 6] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.5],
        [-1.0, -1.0, 0.0],
        [-1.0, 0.0, 1.0],
    ];

    /// The world point of body B's virtual point at frame 0 (and its position in both
    /// bodies' reference frames).
    pub(crate) const VIRTUAL: [f64; 3] = [2.0, -1.0, 0.5];

    pub(crate) fn base(m: usize) -> Point3 {
        Point3::new(BASE[m][0], BASE[m][1], BASE[m][2])
    }

    /// A marker that follows the motion, seen by both cameras at every frame, with its 3D
    /// point set.
    pub(crate) fn marker(cams: &[PoseCamera], m: usize, n_frames: usize) -> Marker {
        let mut marker = Marker::new(cams.len(), n_frames, 8);
        marker.requires_recomputation = false;
        for f in 0..n_frames {
            let x = moved(base(m), f);
            marker.points3d[f] = x;
            marker.status3d[f] = MarkerStatus::Tracked;
            for (c, cam) in cams.iter().enumerate() {
                marker.views[c].points[f] = cam.model.project_point(&cam.pose, x);
                marker.views[c].status[f] = MarkerStatus::Tracked;
            }
        }
        marker
    }

    pub(crate) fn body(markers: &[usize], n_frames: usize) -> RigidBody {
        let mut b = RigidBody::new(n_frames);
        b.marker_indices = markers.to_vec();
        b.reference_points = markers.iter().map(|&m| base(m)).collect();
        b.reference_points_original = b.reference_points.clone();
        b.reference_names = vec![String::new(); markers.len()];
        b.references_set = 2;
        b.initialised = true;
        b
    }

    pub(crate) fn virtual_to(other: usize) -> VirtualPoint {
        let v = Point3::new(VIRTUAL[0], VIRTUAL[1], VIRTUAL[2]);
        VirtualPoint {
            name: "v".to_owned(),
            reference: v,
            source: VirtualPointSource::OtherBody {
                body: other,
                point: v,
            },
        }
    }

    /// A trial of `n_frames` with the six markers, body A (markers 0 to 3) and body B
    /// (markers 4 and 5 and a virtual point in body A), A at index `a`.
    fn trial(cams: &[PoseCamera], n_frames: usize, a_first: bool) -> (Trial, usize, usize) {
        let mut t = Trial::default_trial();
        t.n_frames = n_frames;
        for m in 0..6 {
            t.markers.push(Arc::new(marker(cams, m, n_frames)));
        }
        let (a, b) = if a_first { (0, 1) } else { (1, 0) };
        let body_a = body(&[0, 1, 2, 3], n_frames);
        let mut body_b = body(&[4, 5], n_frames);
        body_b.virtual_points.push(virtual_to(a));
        if a_first {
            t.rigid_bodies = vec![Arc::new(body_a), Arc::new(body_b)];
        } else {
            t.rigid_bodies = vec![Arc::new(body_b), Arc::new(body_a)];
        }
        (t, a, b)
    }

    fn update_all(ctx: &PoseContext, t: &mut Trial) -> bool {
        let states: Vec<FrameStates> = (0..t.n_frames).map(|f| update_frame(ctx, t, f)).collect();
        let mut cycle = false;
        for s in &states {
            s.write(t);
            cycle |= s.cycle;
        }
        cycle
    }

    /// Distance between where a pose puts the reference points and where they are.
    fn pose_residual(pose: &Pose, frame: usize, markers: &[usize]) -> f64 {
        markers
            .iter()
            .map(|&m| {
                let w = body_to_world(&to_transform(pose), base(m));
                (w - moved(base(m), frame)).norm()
            })
            .fold(0.0, f64::max)
    }

    fn undefine_3d(t: &mut Trial, markers: &[usize], frame: usize) {
        for &m in markers {
            Arc::make_mut(&mut t.markers[m]).status3d[frame] = MarkerStatus::Undefined;
        }
    }

    #[test]
    fn a_virtual_point_in_a_later_body_computes_that_body_on_demand() {
        let ctx = context(true);
        let (mut t, a, b) = trial(&ctx.cameras, 3, false);
        assert_eq!((a, b), (1, 0));
        // At frame 1, B's own pose needs A's: computed on demand from B's turn, then A is
        // computed again in its own turn, with the same result.
        let mut states = FrameStates::read(&t, 1);
        compute_pose(&ctx, &t, &mut states, b);
        let a_on_demand = states.bodies[a];
        assert!(a_on_demand.computed, "A computed on demand");
        assert!(states.bodies[b].computed);
        assert!(!update_all(&ctx, &mut t));
        for f in 0..3 {
            let (pa, pb) = (&t.rigid_bodies[a].raw, &t.rigid_bodies[b].raw);
            assert!(pa.valid[f] && pb.valid[f], "frame {f}");
            assert!(pose_residual(&pa.poses[f], f, &[0, 1, 2, 3]) < 1e-5);
            assert!(pose_residual(&pb.poses[f], f, &[4, 5]) < 1e-5);
        }
        assert_eq!(t.rigid_bodies[a].raw.poses[1], a_on_demand.pose);
        assert_eq!(t.rigid_bodies[a].raw.errors[1], a_on_demand.errors);
        assert_eq!(t.rigid_bodies[b].raw.poses[1], states.bodies[b].pose);
    }

    #[test]
    fn a_virtual_point_in_an_earlier_body_gives_the_same_poses() {
        let ctx = context(true);
        let (mut later, a1, b1) = trial(&ctx.cameras, 3, false);
        let (mut earlier, a0, b0) = trial(&ctx.cameras, 3, true);
        update_all(&ctx, &mut later);
        update_all(&ctx, &mut earlier);
        assert_eq!(earlier.rigid_bodies[a0].raw, later.rigid_bodies[a1].raw);
        assert_eq!(earlier.rigid_bodies[b0].raw, later.rigid_bodies[b1].raw);
        assert!(earlier.rigid_bodies[b0].raw.valid.iter().all(|&v| v));
    }

    #[test]
    fn a_virtual_point_in_a_body_with_no_pose_is_left_out() {
        let ctx = context(true);
        let (mut t, a, b) = trial(&ctx.cameras, 3, true);
        // Body B has a stale pose at frame 2 from an earlier computation.
        let stale = Pose {
            rotation: Vector3::new(0.1, 0.2, 0.3),
            translation: Vector3::new(1.0, 2.0, 3.0),
        };
        {
            let rb = Arc::make_mut(&mut t.rigid_bodies[b]);
            rb.raw.poses[2] = stale;
            rb.raw.valid[2] = true;
            rb.raw.errors[2].mean_3d = 9.0;
        }
        // A has only two 3D points at frame 2 and no fallback: no pose, so B has two
        // correspondences and no pose either.
        undefine_3d(&mut t, &[0, 1], 2);
        update_all(&ctx, &mut t);
        let (ra, rb) = (&t.rigid_bodies[a].raw, &t.rigid_bodies[b].raw);
        assert_eq!((ra.valid[2], rb.valid[2]), (false, false));
        assert!(ra.valid[1] && rb.valid[1]);
        // No pose: the stored pose is left as it was, the errors are zeroed.
        assert_eq!(rb.poses[2], stale);
        assert_eq!(rb.errors[2], FitError::default());
        assert_eq!(ra.poses[2], Pose::ZERO);
    }

    #[test]
    fn coordinates_count_only_where_they_are_set() {
        let ctx = context(true);
        let (mut t, _, b) = trial(&ctx.cameras, 3, true);
        let v = Point3::new(VIRTUAL[0], VIRTUAL[1], VIRTUAL[2]);
        Arc::make_mut(&mut t.rigid_bodies[b]).virtual_points[0].source =
            VirtualPointSource::Coordinates(vec![Some(moved(v, 0)), None, Some(moved(v, 2))]);
        update_all(&ctx, &mut t);
        assert_eq!(t.rigid_bodies[b].raw.valid, vec![true, false, true]);
    }

    #[test]
    fn a_virtual_point_on_a_body_out_of_range_has_no_pose() {
        // core/RigidBody.cpp:1099,491,1729 index the trial's bodies with the stored index
        // unchecked (undefined behaviour); here such a point has no world position.
        let ctx = context(true);
        let (mut t, a, b) = trial(&ctx.cameras, 2, true);
        let mut states = FrameStates::read(&t, 0);
        assert_eq!(
            transform_point(&ctx, &t, &mut states, 7, base(0)),
            None,
            "body 7 of 2"
        );
        assert!(!states.cycle);
        Arc::make_mut(&mut t.rigid_bodies[b]).virtual_points[0] = virtual_to(7);
        assert!(!update_all(&ctx, &mut t));
        // B keeps two markers of its own and no fallback: no pose, zero errors; A unchanged.
        let (ra, rb) = (&t.rigid_bodies[a].raw, &t.rigid_bodies[b].raw);
        assert_eq!(rb.valid, vec![false, false]);
        assert_eq!(rb.errors[1], FitError::default());
        assert_eq!(ra.valid, vec![true, true]);
        // The Filter3DPoints refit reads it the same way: no virtual point series.
        let ctx = PoseContext {
            filter_3d_points: true,
            ..ctx
        };
        assert!(!filter_transformations(&ctx, &mut t, b));
        assert!(!t.rigid_bodies[b].filtered.valid[0]);
    }

    #[test]
    fn a_cycle_of_virtual_points_is_cut_and_reported() {
        let ctx = context(true);
        let (mut t, a, b) = trial(&ctx.cameras, 2, true);
        // A refers back to B, and A has only two markers of its own at frame 0.
        Arc::make_mut(&mut t.rigid_bodies[a])
            .virtual_points
            .push(virtual_to(b));
        undefine_3d(&mut t, &[0, 1], 0);
        assert!(update_all(&ctx, &mut t));
        assert!(!t.rigid_bodies[a].raw.valid[0]);
        assert!(t.rigid_bodies[a].raw.valid[1]);
    }

    #[test]
    fn a_body_that_is_not_initialised_gets_no_pose_and_zero_errors() {
        let ctx = context(false);
        let (mut t, a, _) = trial(&ctx.cameras, 2, true);
        {
            let rb = Arc::make_mut(&mut t.rigid_bodies[a]);
            rb.initialised = false;
            rb.raw.errors[0].mean_2d = 5.0;
        }
        update_all(&ctx, &mut t);
        let r = &t.rigid_bodies[a].raw;
        assert_eq!(r.valid, vec![false, false]);
        assert_eq!(r.errors[0], FitError::default());
    }

    #[test]
    fn a_body_with_no_reference_points_gets_no_pose() {
        let ctx = context(false);
        let (mut t, _, _) = trial(&ctx.cameras, 2, true);
        // Initialised (loaded from empty reference files) but with no markers and no
        // reference points: no correspondences, no camera with 4 points.
        let mut empty = RigidBody::new(2);
        empty.references_set = 2;
        empty.initialised = true;
        empty.raw.errors[1].mean_3d = 1.0;
        t.rigid_bodies.push(Arc::new(empty));
        update_all(&ctx, &mut t);
        let r = &t.rigid_bodies[2].raw;
        assert_eq!(r.valid, vec![false, false]);
        assert_eq!(r.errors[1], FitError::default());
    }

    #[test]
    fn recompute_transformations_uses_other_bodies_stored_poses() {
        // Ledger A36. After a full update, A's markers move at frame 1 without A being
        // recomputed. Recomputing B alone uses A's stored pose (its flag is set); the full
        // update recomputes A first.
        let ctx = context(true);
        let (mut t, a, b) = trial(&ctx.cameras, 3, true);
        update_all(&ctx, &mut t);
        for m in 0..4 {
            let marker = Arc::make_mut(&mut t.markers[m]);
            marker.points3d[1] = moved(base(m), 2);
        }
        let a_before = t.rigid_bodies[a].raw.clone();

        let mut alone = t.clone();
        recompute_transformations(&ctx, &mut alone, b);
        assert_eq!(alone.rigid_bodies[a].raw, a_before, "A is not touched");

        let mut full = t.clone();
        update_all(&ctx, &mut full);
        assert_ne!(full.rigid_bodies[a].raw.poses[1], a_before.poses[1]);
        let (pb_alone, pb_full) = (
            alone.rigid_bodies[b].raw.poses[1],
            full.rigid_bodies[b].raw.poses[1],
        );
        assert_ne!(pb_alone, pb_full);
        assert_eq!(
            alone.rigid_bodies[b].raw.poses[0],
            full.rigid_bodies[b].raw.poses[0]
        );

        // With A's flag clear at frame 1, recomputing B computes A there on demand.
        let mut cleared = t.clone();
        Arc::make_mut(&mut cleared.rigid_bodies[a]).raw.valid[1] = false;
        recompute_transformations(&ctx, &mut cleared, b);
        assert_eq!(
            cleared.rigid_bodies[a].raw.poses[1],
            full.rigid_bodies[a].raw.poses[1]
        );
        assert!(cleared.rigid_bodies[a].raw.valid[1]);
        assert_eq!(cleared.rigid_bodies[b].raw.poses[1], pb_full);
    }

    fn filter_trial(n_frames: usize, speed: f64, cutoff: f64) -> (PoseContext, Trial) {
        let ctx = context(true);
        let (mut t, _, _) = trial(&ctx.cameras, n_frames, true);
        t.recording_speed = speed;
        t.cutoff_frequency = cutoff;
        update_all(&ctx, &mut t);
        (ctx, t)
    }

    fn stale_filtered(t: &mut Trial, body: usize) -> Pose {
        let stale = Pose {
            rotation: Vector3::new(0.5, 0.5, 0.5),
            translation: Vector3::new(7.0, 8.0, 9.0),
        };
        let rb = Arc::make_mut(&mut t.rigid_bodies[body]);
        for f in 0..rb.filtered.poses.len() {
            rb.filtered.poses[f] = stale;
            rb.filtered.valid[f] = true;
            rb.filtered.errors[f].mean_3d = 3.0;
        }
        stale
    }

    #[test]
    fn the_pose_filter_refuses_without_a_usable_cutoff() {
        for f3d in [false, true] {
            for (speed, cutoff) in [(100.0, 0.0), (100.0, 50.0), (100.0, 60.0), (0.0, 10.0)] {
                let (mut ctx, mut t) = filter_trial(20, speed, cutoff);
                ctx.filter_3d_points = f3d;
                let stale = stale_filtered(&mut t, 0);
                filter_transformations(&ctx, &mut t, 0);
                let fl = &t.rigid_bodies[0].filtered;
                assert!(fl.valid.iter().all(|&v| !v), "f3d {f3d} cutoff {cutoff}");
                assert!(fl.poses.iter().all(|&p| p == stale), "left as they were");
                assert!(fl.errors.iter().all(|&e| e == FitError::default()));
            }
        }
    }

    #[test]
    fn the_body_cutoff_overrides_the_trial_cutoff() {
        let (ctx, mut t) = filter_trial(20, 100.0, 0.0);
        {
            let rb = Arc::make_mut(&mut t.rigid_bodies[0]);
            rb.override_cutoff_frequency = true;
            rb.cutoff_frequency = 10.0;
        }
        filter_transformations(&ctx, &mut t, 0);
        assert!(t.rigid_bodies[0].filtered.valid.iter().all(|&v| v));
        // Body 1 has no override: refused at the trial's 0.
        filter_transformations(&ctx, &mut t, 1);
        assert!(t.rigid_bodies[1].filtered.valid.iter().all(|&v| !v));
    }

    #[test]
    fn the_pose_filter_copies_short_runs_and_filters_long_ones() {
        let (ctx, mut t) = filter_trial(30, 100.0, 10.0);
        // A gap at frame 12: runs 0..12 (12 frames, copied) and 13..30 (17, filtered).
        undefine_3d(&mut t, &[0, 1], 12);
        update_all(&ctx, &mut t);
        assert!(!t.rigid_bodies[0].raw.valid[12]);
        filter_transformations(&ctx, &mut t, 0);
        let rb = &t.rigid_bodies[0];
        assert!(!rb.filtered.valid[12]);
        for f in 0..12 {
            assert!(rb.filtered.valid[f]);
            assert_eq!(rb.filtered.poses[f], rb.raw.poses[f], "frame {f} copied");
        }
        assert!((13..30).all(|f| rb.filtered.valid[f]));
        assert_ne!(rb.filtered.poses[20], rb.raw.poses[20]);
        assert!(rb.filtered.errors[20].mean_3d > 0.0);
    }

    #[test]
    fn the_3d_filter_refits_from_filtered_markers_and_virtual_points() {
        let (mut ctx, mut t) = filter_trial(30, 100.0, 10.0);
        ctx.filter_3d_points = true;
        // Body B: two markers and the virtual point in A; with a gap in A at frame 5 the
        // virtual point has runs 0..5 (copied) and 6..30 (filtered).
        undefine_3d(&mut t, &[0, 1], 5);
        update_all(&ctx, &mut t);
        filter_transformations(&ctx, &mut t, 1);
        let rb = &t.rigid_bodies[1];
        assert!(!rb.filtered.valid[5], "only two filtered points at frame 5");
        assert!((0..30).filter(|&f| f != 5).all(|f| rb.filtered.valid[f]));
        // Frames 0..5: the markers are filtered over 0..30 but the virtual point is copied,
        // so the refit is close to, not equal to, the raw pose.
        assert!(pose_residual(&rb.filtered.poses[2], 2, &[4, 5]) < 1e-2);
    }

    #[test]
    fn filter_marker_keeps_placeholders_outside_runs() {
        let n = 20;
        let points: Vec<Point3> = (0..n).map(|f| Point3::new(f as f64, 0.0, 1.0)).collect();
        let mut status = vec![MarkerStatus::Tracked; n];
        status[3] = MarkerStatus::Undefined;
        let (p, s, ok) = filter_marker(
            10.0,
            100.0,
            n,
            &points,
            &status,
            CutoffCorrection::default(),
        );
        assert!(ok);
        assert_eq!(s[3], MarkerStatus::Undefined);
        assert_eq!(p[3], Point3::new(-1000.0, -1000.0, -1000.0));
        assert_eq!(p[1], points[1], "run of 3 copied");
        assert_eq!(s[10], MarkerStatus::Tracked);
        let (p, s, ok) = filter_marker(
            50.0,
            100.0,
            n,
            &points,
            &status,
            CutoffCorrection::default(),
        );
        assert!(!ok);
        assert!(s.iter().all(|&s| s == MarkerStatus::Undefined));
        assert!(p.iter().all(|p| p.x == -1000.0));
    }

    /// Ledger A7: the pose filter and the marker filter design the filter at the corrected
    /// cutoff by default and at the entered cutoff with `SinglePass` (XMALab 3), exactly as
    /// `ButterworthLowPass::zero_phase` and `ButterworthLowPass::new` do.
    #[test]
    fn the_filters_use_the_context_cutoff_correction() {
        let (mut ctx, mut t) = filter_trial(40, 100.0, 10.0);
        for correction in [CutoffCorrection::BothPasses, CutoffCorrection::SinglePass] {
            ctx.filter_correction = correction;
            filter_transformations(&ctx, &mut t, 0);
            let filter = match correction {
                CutoffCorrection::BothPasses => {
                    let d = xma_vision::filter::design_cutoff(4, 10.0, 100.0, correction);
                    assert!(d > 10.0);
                    ButterworthLowPass::new(FILTER_ORDER, d, 100.0)
                }
                CutoffCorrection::SinglePass => ButterworthLowPass::new(FILTER_ORDER, 10.0, 100.0),
            };
            let rb = &t.rigid_bodies[0];
            let tx: Vec<f64> = rb.raw.poses[..40]
                .iter()
                .map(|p| p.translation[0])
                .collect();
            let want = filter_run(&filter, &tx);
            let got: Vec<f64> = rb.filtered.poses[..40]
                .iter()
                .map(|p| p.translation[0])
                .collect();
            assert_eq!(got, want, "{correction:?}");

            let points: Vec<Point3> = (0..40)
                .map(|f| Point3::new((f as f64 * 0.7).sin(), 0.0, 1.0))
                .collect();
            let status = vec![MarkerStatus::Tracked; 40];
            let (p, _, ok) = filter_marker(10.0, 100.0, 40, &points, &status, correction);
            assert!(ok);
            let x: Vec<f64> = points.iter().map(|p| p.x).collect();
            let want = filter_run(&filter, &x);
            assert_eq!(p.iter().map(|p| p.x).collect::<Vec<_>>(), want);
        }
    }
}
