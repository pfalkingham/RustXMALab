//! Ports of the rigid-body and trial model calls that the structure, body and import edits
//! share: `Trial::addMarker`, `removeMarker`, `addRigidBody`, `removeRigidBody`,
//! `RigidBody::addPointIdx`, `removePointIdx`, `updatePointIdx`, `clearPointIdx`,
//! `resetReferences`, `setReferenceMarkerReferences`, `computeCoordinateSystemAverage`,
//! `setReferenceFromFile` (with `fitAndComputeError`), `setReferenceFromFrame`,
//! `setOptimized`, `addDummyPoint`, `clearAllDummyPoints` and `copyData`, and the readers of
//! the files these edits take.
//!
//! # Recomputation
//!
//! Each model call here does what its C++ counterpart does to the data. What it recomputes
//! depends on the mode:
//!
//! - **XMALab 3 recomputation** (`CppCompat::edits_not_refiltered`, ledger A60): every call
//!   runs exactly the recomputation its C++ counterpart runs ([`cpp_recompute`]:
//!   `recomputeTransformations`, with or without `filterTransformations`, one body at a time
//!   as in ledger A36), so a sequence of calls leaves what the C++ sequence leaves, stale
//!   poses included.
//! - **Default**: the calls recompute nothing; the edit runs inside
//!   [`keeping_bodies_current`], which afterwards recomputes, as the full trial update does
//!   (every body's pose flag cleared first, so driving bodies are computed afresh; ledger
//!   A36 fixed), every body whose pose inputs changed and every body whose virtual points
//!   depend on one of those, makes their rotations continuous, and marks them stale so that
//!   [`Document::settle`] filters them when the step is committed. The result is what a full recomputation of the
//!   trial's bodies gives, provided the trial was up to date before the edit.
//!
//! The ports that are algorithms rather than recomputation (the pose computations inside
//! `computeCoordinateSystemAverage` and `setOptimized`) run in both modes.
//!
//! # Defects of `docs/cpp-editing.md` section 15 handled here
//!
//! Fixed by default. The XMALab 3 behaviour is kept, behind a `CppCompat` field, where
//! the C++ editing reference shows it (`CppCompat::remove_point_by_value`, ledger A77;
//! `virtual_points_keep_body_index`, A78; `clear_point_idx_keeps_original`, A79;
//! `import_2d_empty_cell_zero`, F30). Fixed in both modes, as no reference reaches them:
//! `copyData` copying the override flag as the cutoff and the source's `initialised`
//! (A80, A81), `setOptimized` using frames whose pose computation failed (A82), and the
//! trial import abandoning the trials after a declined Default trial (F31).

use std::path::Path;
use std::sync::Arc;

use xma_io::cparse::{LineReader, qstring, qt_to_double, qt_to_int, split_commas};
use xma_model::{Marker, MarkerStatus, Point3, RigidBody, Trial, VirtualPoint, VirtualPointSource};
use xma_vision::camera::rodrigues_to_matrix;
use xma_vision::rigid::alignment_error;

use super::{EditError, EditOutcome, EditResult};
use crate::document::Document;
use crate::rigid_bodies::{self, FrameStates, PoseContext};

// ---------------------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------------------

/// True when edits recompute exactly what the C++ calls recompute (ledger A60 kept).
#[must_use]
pub(crate) fn cpp_recomputation(doc: &Document) -> bool {
    doc.compat().edits_not_refiltered
}

// ---------------------------------------------------------------------------------------
// Access and validation
// ---------------------------------------------------------------------------------------

/// `Invalid` unless the trial exists.
pub(crate) fn check_trial(doc: &Document, trial: usize) -> Result<(), EditError> {
    let n = doc.project.trials.len();
    if trial < n {
        Ok(())
    } else {
        Err(EditError::Invalid(format!(
            "trial {trial} does not exist ({n} trials)"
        )))
    }
}

/// `Invalid` unless the trial and the body exist.
pub(crate) fn check_body(doc: &Document, trial: usize, body: usize) -> Result<(), EditError> {
    check_trial(doc, trial)?;
    let n = doc.project.trials[trial].rigid_bodies.len();
    if body < n {
        Ok(())
    } else {
        Err(EditError::Invalid(format!(
            "rigid body {body} does not exist ({n} bodies in trial {trial})"
        )))
    }
}

/// `Invalid` unless every marker exists and none is named twice.
pub(crate) fn check_markers(
    doc: &Document,
    trial: usize,
    markers: &[usize],
) -> Result<(), EditError> {
    check_trial(doc, trial)?;
    let n = doc.project.trials[trial].markers.len();
    for (k, &m) in markers.iter().enumerate() {
        if m >= n {
            return Err(EditError::Invalid(format!(
                "marker {m} does not exist ({n} markers in trial {trial})"
            )));
        }
        if markers[..k].contains(&m) {
            return Err(EditError::Invalid(format!("marker {m} is named twice")));
        }
    }
    Ok(())
}

/// The trial, for writing.
pub(crate) fn trial_mut(doc: &mut Document, trial: usize) -> &mut Trial {
    Arc::make_mut(&mut doc.project.trials[trial])
}

/// One body, for writing.
pub(crate) fn body_mut(doc: &mut Document, trial: usize, body: usize) -> &mut RigidBody {
    Arc::make_mut(&mut trial_mut(doc, trial).rigid_bodies[body])
}

/// The first body of the trial that lists the marker: the one whose row holds the marker in
/// the Points tree.
#[must_use]
pub(crate) fn body_of_marker(trial: &Trial, marker: usize) -> Option<usize> {
    trial
        .rigid_bodies
        .iter()
        .position(|b| b.marker_indices.contains(&marker))
}

/// A file's bytes, or [`EditError::Io`].
pub(crate) fn read_file(path: &Path) -> Result<Vec<u8>, EditError> {
    std::fs::read(path).map_err(|e| EditError::Io {
        path: path.to_path_buf(),
        message: e.to_string(),
    })
}

/// The 3D status of a marker at a frame; `UNDEFINED` for a marker or frame out of range
/// (undefined behaviour in the C++).
fn status3d(trial: &Trial, marker: usize, frame: usize) -> MarkerStatus {
    trial
        .markers
        .get(marker)
        .and_then(|m| m.status3d.get(frame).copied())
        .unwrap_or(MarkerStatus::Undefined)
}

// ---------------------------------------------------------------------------------------
// Recomputation
// ---------------------------------------------------------------------------------------

/// `recomputeTransformations()` and, with `filter`, `filterTransformations()` on one body,
/// in the XMALab 3 recomputation only (see the module documentation).
pub(crate) fn cpp_recompute(doc: &mut Document, trial: usize, body: usize, filter: bool) {
    if cpp_recomputation(doc) {
        doc.recompute_transformations(trial, body);
        if filter {
            doc.filter_transformations(trial, body);
        }
    }
}

/// Runs an edit of one trial and, in the default mode, brings the trial's bodies up to date
/// afterwards ([`refresh_bodies`]): the bodies whose pose inputs changed, the bodies listed
/// in `force`, and every body with a virtual point that depends on one of them are
/// recomputed and marked stale; bodies whose filter settings alone changed are marked stale.
/// In the XMALab 3 recomputation the edit's own calls have recomputed what the C++ does.
pub(crate) fn keeping_bodies_current(
    doc: &mut Document,
    trial: usize,
    force: &[usize],
    f: impl FnOnce(&mut Document) -> EditResult,
) -> EditResult {
    check_trial(doc, trial)?;
    let before = Arc::clone(&doc.project.trials[trial]);
    let mut outcome = f(doc)?;
    if !cpp_recomputation(doc)
        && trial < doc.project.trials.len()
        && refresh_bodies(doc, trial, &before, force)
    {
        outcome.messages.push(format!(
            "trial {trial}: virtual points of rigid bodies refer to each other in a cycle; \
             the inner reference has no pose"
        ));
    }
    Ok(outcome)
}

/// True when a marker's data that a pose reads are the same.
fn same_marker(a: Option<&Arc<Marker>>, b: Option<&Arc<Marker>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            Arc::ptr_eq(a, b)
                || (a.views == b.views && a.points3d == b.points3d && a.status3d == b.status3d)
        }
        _ => false,
    }
}

/// True when everything `computePose` reads of a body is the same in both trials (the
/// markers compared by content, so a renumbering is not a change).
fn same_pose_inputs(before: &Trial, old: &RigidBody, after: &Trial, new: &RigidBody) -> bool {
    old.initialised == new.initialised
        && old.reference_points == new.reference_points
        && old.virtual_points == new.virtual_points
        && old.marker_indices.len() == new.marker_indices.len()
        && old
            .marker_indices
            .iter()
            .zip(&new.marker_indices)
            .all(|(&a, &b)| same_marker(before.markers.get(a), after.markers.get(b)))
}

/// The default mode's recomputation after an edit (see [`keeping_bodies_current`]). Returns
/// whether a cycle of virtual points was met.
///
/// Not a port: the C++ recomputes after each call one body at a time without clearing
/// other bodies' flags (ledger A36), or not at all (ledger A60).
pub(crate) fn refresh_bodies(
    doc: &mut Document,
    trial: usize,
    before: &Trial,
    force: &[usize],
) -> bool {
    let t = &doc.project.trials[trial];
    let n = t.rigid_bodies.len();
    let mut recompute = vec![false; n];
    let mut stale = vec![false; n];
    for (b, new) in t.rigid_bodies.iter().enumerate() {
        match before.rigid_bodies.get(b) {
            None => recompute[b] = true,
            Some(old) => {
                // Raw poses written during the edit (the pose computations of
                // `computeCoordinateSystemAverage` and `setOptimized`) are recomputed too.
                let same_raw = Arc::ptr_eq(old, new) || old.raw == new.raw;
                if !same_pose_inputs(before, old, t, new) || !same_raw {
                    recompute[b] = true;
                } else if old.override_cutoff_frequency != new.override_cutoff_frequency
                    || old.cutoff_frequency != new.cutoff_frequency
                    || old.filtered != new.filtered
                {
                    stale[b] = true;
                }
            }
        }
    }
    for &b in force {
        if b < n {
            recompute[b] = true;
        }
    }
    // Bodies whose virtual points follow a recomputed body.
    loop {
        let mut grew = false;
        for b in 0..n {
            if recompute[b] {
                continue;
            }
            let depends = t.rigid_bodies[b].virtual_points.iter().any(|v| {
                matches!(v.source, VirtualPointSource::OtherBody { body, .. } if body < n && recompute[body])
            });
            if depends {
                recompute[b] = true;
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    let set: Vec<usize> = (0..n).filter(|&b| recompute[b]).collect();
    let cycle = if set.is_empty() {
        false
    } else {
        recompute_bodies_fully(doc, trial, &set)
    };
    for b in 0..n {
        if recompute[b] || stale[b] {
            doc.mark_body_stale(trial, b);
        }
    }
    cycle
}

/// The raw poses of the given bodies (index order) at every frame, as the trial update
/// computes them: at each frame **every** body's flag is cleared, then each listed body is
/// computed in index order, so a virtual point driven by another body uses that body's
/// pose computed afresh at the frame (on demand), never its stored pose, whose rotation the
/// continuity step may have replaced. Only the listed bodies' states are written; another
/// body computed on demand keeps its stored state. Then the continuity step on each listed
/// body's series; no filtering. Returns whether a cycle of virtual points was met.
///
/// Not a port: `UpdateTrialFrame::process` (processing/UpdateTrialFrame.cpp, `:62-70`)
/// restricted to the bodies an edit changed.
pub(crate) fn recompute_bodies_fully(doc: &mut Document, trial: usize, bodies: &[usize]) -> bool {
    let ctx = doc.pose_context(trial);
    let t = trial_mut(doc, trial);
    let n = t.n_frames;
    for rb in &mut t.rigid_bodies {
        if rb.n_frames() < n {
            Arc::make_mut(rb).grow_to(n);
        }
    }
    let t: &Trial = t;
    let all: Vec<FrameStates> = (0..n)
        .map(|frame| {
            let mut states = FrameStates::read(t, frame);
            for s in &mut states.bodies {
                s.computed = false;
            }
            for &b in bodies {
                rigid_bodies::compute_pose(&ctx, t, &mut states, b);
            }
            states
        })
        .collect();
    let t = trial_mut(doc, trial);
    let mut cycle = false;
    for s in &all {
        s.write_bodies(t, bodies);
        cycle |= s.cycle;
    }
    // The continuity step now rather than at `settle`: a later edit of the same step that
    // writes one frame (`continue_rotations`) expects the stored series to have been through
    // it already. The step reads only the body's own series, so this is what the update's
    // finalize gives.
    for &b in bodies {
        if b < t.rigid_bodies.len() {
            rigid_bodies::make_rotations_continuous(t, b);
        }
    }
    cycle
}

/// `computePose(frame)` on one body, written back into the trial. `ctx` is
/// [`Document::pose_context`] of the trial, built by the caller once for a loop that changes
/// none of what it is built from (cameras, settings, the trial's reference image).
///
/// Ported from core/RigidBody.cpp, RigidBody::computePose (`:1065-1229`), through
/// [`rigid_bodies::compute_pose`].
fn compute_pose_at(
    ctx: &PoseContext,
    doc: &mut Document,
    trial: usize,
    body: usize,
    frame: usize,
) -> bool {
    let t = &doc.project.trials[trial];
    let mut states = FrameStates::read(t, frame);
    rigid_bodies::compute_pose(ctx, t, &mut states, body);
    let computed = states.bodies[body].computed;
    states.write(trial_mut(doc, trial));
    computed
}

// ---------------------------------------------------------------------------------------
// Trial: markers and bodies
// ---------------------------------------------------------------------------------------

/// Appends a marker and returns its index.
///
/// Ported from core/Trial.cpp, Trial::addMarker (`:461-465`) and core/Marker.cpp,
/// Marker::Marker (`:45-58`): threshold offset from `DefaultMarkerThreshold`, penalty 50,
/// method 0, no size override, flagged for recomputation. The active marker is the
/// interface's. Nothing is recomputed.
pub(crate) fn add_marker(doc: &mut Document, trial: usize) -> usize {
    let n_cameras = doc.project.cameras.len();
    let threshold = doc.project.algorithm_settings.default_marker_threshold;
    let t = trial_mut(doc, trial);
    let n_frames = t.n_frames;
    t.markers
        .push(Arc::new(Marker::new(n_cameras, n_frames, threshold)));
    t.markers.len() - 1
}

/// Removes a marker: from the list, then from every body (`removePointIdx`), whose later
/// indices drop by one (`updatePointIdx`).
///
/// Ported from core/Trial.cpp, Trial::removeMarker (`:467-479`).
pub(crate) fn remove_marker(doc: &mut Document, trial: usize, marker: usize) {
    trial_mut(doc, trial).markers.remove(marker);
    for b in 0..doc.project.trials[trial].rigid_bodies.len() {
        remove_point_idx(doc, trial, b, marker);
        update_point_idx(doc, trial, b, marker);
    }
}

/// Appends an empty rigid body (red, not drawn, no references). Nothing is recomputed.
///
/// Ported from core/Trial.cpp, Trial::addRigidBody (`:447-451`), and core/RigidBody.cpp,
/// RigidBody::RigidBody (`:65-80`).
pub(crate) fn add_rigid_body(doc: &mut Document, trial: usize) -> usize {
    let t = trial_mut(doc, trial);
    let n_frames = t.n_frames;
    t.rigid_bodies.push(Arc::new(RigidBody::new(n_frames)));
    t.rigid_bodies.len() - 1
}

/// Removes a rigid body. Nothing is recomputed. Returns Console lines for the virtual points
/// removed with it.
///
/// Ported from core/Trial.cpp, Trial::removeRigidBody (`:454-458`). D6 fixed (unless
/// [`xma_model::CppCompat::virtual_points_keep_body_index`], ledger A78): every other body's virtual point driven by
/// the removed body is deleted, since its position can no longer be computed, and driver
/// indices above it are lowered by one, so each point keeps following the same body.
pub(crate) fn remove_rigid_body(doc: &mut Document, trial: usize, body: usize) -> Vec<String> {
    let keep = doc.compat().virtual_points_keep_body_index;
    let t = trial_mut(doc, trial);
    t.rigid_bodies.remove(body);
    let mut messages = Vec::new();
    if keep {
        return messages;
    }
    for (b, rb) in t.rigid_bodies.iter_mut().enumerate() {
        let touches = rb.virtual_points.iter().any(
            |v| matches!(v.source, VirtualPointSource::OtherBody { body: d, .. } if d >= body),
        );
        if !touches {
            continue;
        }
        let rb = Arc::make_mut(rb);
        rb.virtual_points.retain(|v| match v.source {
            VirtualPointSource::OtherBody { body: d, .. } if d == body => {
                messages.push(format!(
                    "Virtual point \"{}\" of rigid body {} was deleted: it followed the deleted rigid body {}",
                    v.name,
                    b + 1,
                    body + 1
                ));
                false
            }
            _ => true,
        });
        for v in &mut rb.virtual_points {
            if let VirtualPointSource::OtherBody { body: d, .. } = &mut v.source
                && *d > body
            {
                *d -= 1;
            }
        }
    }
    messages
}

// ---------------------------------------------------------------------------------------
// Rigid body: markers
// ---------------------------------------------------------------------------------------

/// Appends a marker to a body with a zero reference point and an empty name, then
/// `resetReferences` (so every reference of the body is discarded), then, with `recompute`,
/// recomputes and filters again.
///
/// Ported from core/RigidBody.cpp, RigidBody::addPointIdx (`:217-229`).
pub(crate) fn add_point_idx(
    doc: &mut Document,
    trial: usize,
    body: usize,
    marker: usize,
    recompute: bool,
) {
    let rb = body_mut(doc, trial, body);
    rb.marker_indices.push(marker);
    rb.reference_points.push(Point3::origin());
    rb.reference_points_original.push(Point3::origin());
    rb.reference_names.push(String::new());
    reset_references(doc, trial, body);
    if recompute {
        cpp_recompute(doc, trial, body, true);
    }
}

/// Removes a marker from a body with its reference point and name; the other references
/// are kept. Recomputes and filters when the marker was in the body. Returns whether it
/// was.
///
/// Ported from core/RigidBody.cpp, RigidBody::removePointIdx (`:231-243`). D5 fixed (unless
/// [`xma_model::CppCompat::remove_point_by_value`], ledger A77): the reference point, original point and name at
/// the marker's position are removed, not every element equal to them. Where the lists are
/// shorter than the marker list (undefined behaviour in the C++) only what exists is
/// removed.
pub(crate) fn remove_point_idx(
    doc: &mut Document,
    trial: usize,
    body: usize,
    marker: usize,
) -> bool {
    let by_value = doc.compat().remove_point_by_value;
    let Some(pos) = doc.project.trials[trial].rigid_bodies[body]
        .marker_indices
        .iter()
        .position(|&m| m == marker)
    else {
        return false;
    };
    let rb = body_mut(doc, trial, body);
    if by_value {
        if let Some(v) = rb.reference_points.get(pos).copied() {
            rb.reference_points.retain(|p| *p != v);
        }
        if let Some(v) = rb.reference_points_original.get(pos).copied() {
            rb.reference_points_original.retain(|p| *p != v);
        }
        if let Some(v) = rb.reference_names.get(pos).cloned() {
            rb.reference_names.retain(|p| *p != v);
        }
        rb.marker_indices.retain(|&m| m != marker);
    } else {
        // Every position of the marker (a body read from a file can list one twice), last
        // first.
        let positions: Vec<usize> = rb
            .marker_indices
            .iter()
            .enumerate()
            .filter(|&(_, &m)| m == marker)
            .map(|(k, _)| k)
            .collect();
        for &k in positions.iter().rev() {
            if k < rb.reference_points.len() {
                rb.reference_points.remove(k);
            }
            if k < rb.reference_points_original.len() {
                rb.reference_points_original.remove(k);
            }
            if k < rb.reference_names.len() {
                rb.reference_names.remove(k);
            }
            rb.marker_indices.remove(k);
        }
    }
    cpp_recompute(doc, trial, body, true);
    true
}

/// Lowers by one every marker index of the body above `marker` (after that marker was
/// deleted); recomputes and filters when any changed.
///
/// Ported from core/RigidBody.cpp, RigidBody::updatePointIdx (`:245-260`).
pub(crate) fn update_point_idx(doc: &mut Document, trial: usize, body: usize, marker: usize) {
    if !doc.project.trials[trial].rigid_bodies[body]
        .marker_indices
        .iter()
        .any(|&m| m > marker)
    {
        return;
    }
    for m in &mut body_mut(doc, trial, body).marker_indices {
        if *m > marker {
            *m -= 1;
        }
    }
    cpp_recompute(doc, trial, body, true);
}

/// Empties the body's marker list, reference points and names, then `resetReferences`.
///
/// Ported from core/RigidBody.cpp, RigidBody::clearPointIdx (`:204-210`), which leaves
/// `points3D_original` as it is; fixed unless [`xma_model::CppCompat::clear_point_idx_keeps_original`] (ledger A79).
pub(crate) fn clear_point_idx(doc: &mut Document, trial: usize, body: usize) {
    let keep = doc.compat().clear_point_idx_keeps_original;
    let rb = body_mut(doc, trial, body);
    rb.marker_indices.clear();
    rb.reference_points.clear();
    rb.reference_names.clear();
    if !keep {
        rb.reference_points_original.clear();
    }
    reset_references(doc, trial, body);
}

// ---------------------------------------------------------------------------------------
// Rigid body: references
// ---------------------------------------------------------------------------------------

/// Clears the references (names empty, points and original points zero, not set, not
/// initialised, not optimised), takes them from the markers when every marker of the body
/// has a reference point, then recomputes and filters.
///
/// Ported from core/RigidBody.cpp, RigidBody::resetReferences (`:262-280`). Where the point
/// lists are shorter than the names (undefined behaviour in the C++) only what exists is
/// cleared.
pub(crate) fn reset_references(doc: &mut Document, trial: usize, body: usize) {
    let rb = body_mut(doc, trial, body);
    let origin = Point3::origin();
    for k in 0..rb.reference_names.len() {
        rb.reference_names[k].clear();
        if let Some(p) = rb.reference_points.get_mut(k) {
            *p = origin;
        }
        if let Some(p) = rb.reference_points_original.get_mut(k) {
            *p = origin;
        }
    }
    rb.references_set = 0;
    rb.initialised = false;
    rb.has_optimized_coordinates = false;
    set_reference_marker_references(trial_mut(doc, trial), body);
    cpp_recompute(doc, trial, body, true);
}

/// When every marker of the body has a reference point: the reference points (original
/// and working) become those, the names "<description> - FromMarker", and the body is
/// initialised with `referencesSet` 1. A body without markers qualifies. Nothing is
/// recomputed.
///
/// Ported from core/RigidBody.cpp, RigidBody::setReferenceMarkerReferences (`:543-560`) and
/// RigidBody::allReferenceMarkerReferencesSet (`:282-293`). A marker index outside the trial
/// (undefined behaviour in the C++) counts as a marker without a reference point.
pub(crate) fn set_reference_marker_references(trial: &mut Trial, body: usize) {
    let rb = &trial.rigid_bodies[body];
    let refs: Option<Vec<(Point3, String)>> = rb
        .marker_indices
        .iter()
        .map(|&m| {
            let marker = trial.markers.get(m)?;
            Some((marker.reference_3d?, marker.description.clone()))
        })
        .collect();
    let Some(refs) = refs else {
        return;
    };
    let rb = Arc::make_mut(&mut trial.rigid_bodies[body]);
    for (k, (p, name)) in refs.into_iter().enumerate() {
        if let Some(slot) = rb.reference_points.get_mut(k) {
            *slot = p;
        }
        if let Some(slot) = rb.reference_points_original.get_mut(k) {
            *slot = p;
        }
        if let Some(slot) = rb.reference_names.get_mut(k) {
            *slot = format!("{name} - FromMarker");
        }
    }
    rb.initialised = true;
    rb.references_set = 1;
}

fn sub(a: Point3, b: Point3) -> Point3 {
    Point3::new(a.x - b.x, a.y - b.y, a.z - b.z)
}

/// `cv::norm(Point3d)`: `sqrt(x*x + y*y + z*z)`.
fn norm(p: Point3) -> f64 {
    (p.x * p.x + p.y * p.y + p.z * p.z).sqrt()
}

/// `Point3d::cross`.
fn cross(a: Point3, b: Point3) -> Point3 {
    Point3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

fn divide(p: Point3, s: f64) -> Point3 {
    Point3::new(p.x / s, p.y / s, p.z / s)
}

/// `cv::gemm(R, v, 1, c, beta)` for a 3x3 `R` stored by rows: `t = r0*v0 + r1*v1 + r2*v2`
/// left to right, stored as `t*1 + c*beta` (OpenCV's small-matrix branch; with no `c` it
/// reads zeros with `beta` 0).
fn gemm(r: &[[f64; 3]; 3], v: Point3, c: Point3, beta: f64) -> Point3 {
    let v = [v.x, v.y, v.z];
    let c = [c.x, c.y, c.z];
    let mut d = [0.0f64; 3];
    for i in 0..3 {
        let t = r[i][0] * v[0] + r[i][1] * v[1] + r[i][2] * v[2];
        d[i] = t * 1.0 + c[i] * beta;
    }
    Point3::new(d[0], d[1], d[2])
}

fn rows(m: &xma_model::Matrix3) -> [[f64; 3]; 3] {
    [
        [m[(0, 0)], m[(0, 1)], m[(0, 2)]],
        [m[(1, 0)], m[(1, 1)], m[(1, 2)]],
        [m[(2, 0)], m[(2, 1)], m[(2, 2)]],
    ]
}

/// `rotationMatrix * x + cv::Mat(t)`: a point of the world mapped into a body's frame by
/// its pose.
fn world_to_body(pose: &xma_model::Pose, x: Point3) -> Point3 {
    let r = rows(&rodrigues_to_matrix(&pose.rotation));
    let t = Point3::new(pose.translation.x, pose.translation.y, pose.translation.z);
    gemm(&r, x, t, 1.0)
}

/// The shape of points in their own frame: centred on their mean (divided by `count`, the
/// C++ `pointsIdx.size()`), then, with two or more, rotated so that the first point lies
/// on the x axis and the second in the x-z plane (rows x, z, y). Returns the points and
/// whether the rotation was applied (which makes the body initialised).
///
/// Ported from core/RigidBody.cpp, RigidBody::computeCoordinateSystem (`:891-976`) and the
/// same steps in RigidBody::computeCoordinateSystemAverage (`:741-814`).
fn shape_in_own_frame(points: &[Point3], count: usize) -> (Vec<Point3>, bool) {
    let mut center = Point3::origin();
    for p in points {
        center = Point3::new(center.x + p.x, center.y + p.y, center.z + p.z);
    }
    if !points.is_empty() {
        center = divide(center, count as f64);
    }
    if points.len() >= 2 {
        let mut px = sub(points[0], center);
        let mut pxy = sub(points[1], center);
        px = divide(px, norm(px));
        pxy = divide(pxy, norm(pxy));
        let mut py = cross(px, pxy);
        py = divide(py, norm(py));
        let pz = cross(py, px);
        let r = [[px.x, px.y, px.z], [pz.x, pz.y, pz.z], [py.x, py.y, py.z]];
        let out = points
            .iter()
            .map(|p| gemm(&r, sub(*p, center), Point3::origin(), 0.0))
            .collect();
        (out, true)
    } else {
        (points.iter().map(|p| sub(*p, center)).collect(), false)
    }
}

/// When the body's references are not set and every marker is valid in 3D at the frame,
/// the reference points become the markers' shape at that frame ([`shape_in_own_frame`]).
///
/// Ported from core/RigidBody.cpp, RigidBody::computeCoordinateSystem (`:891-976`).
fn compute_coordinate_system(trial: &mut Trial, body: usize, frame: usize) {
    let rb = &trial.rigid_bodies[body];
    if rb.references_set != 0 {
        return;
    }
    if rb
        .marker_indices
        .iter()
        .any(|&m| !status3d(trial, m, frame).has_data())
    {
        return;
    }
    let points: Vec<Point3> = rb
        .marker_indices
        .iter()
        .map(|&m| trial.markers[m].points3d[frame])
        .collect();
    let count = rb.marker_indices.len();
    let (shape, rotated) = shape_in_own_frame(&points, count);
    let rb = Arc::make_mut(&mut trial.rigid_bodies[body]);
    rb.reference_points = shape;
    if rotated {
        rb.initialised = true;
    }
}

/// When the body's references are not set: the reference points become the mean shape of
/// the markers over the frames where every marker is valid in 3D and a pose is found
/// (each frame's markers mapped into the body by that frame's pose), in the body's own
/// frame; then every frame's pose is computed again.
///
/// Ported from core/RigidBody.cpp, RigidBody::computeCoordinateSystemAverage (`:673-821`):
/// for each frame `computeCoordinateSystem(f)` (which replaces the reference points with
/// that frame's shape when every marker is valid there) then `computePose(f)`; nothing
/// more when no frame qualified.
pub(crate) fn compute_coordinate_system_average(doc: &mut Document, trial: usize, body: usize) {
    if doc.project.trials[trial].rigid_bodies[body].references_set != 0 {
        return;
    }
    let n = doc.project.trials[trial].rigid_bodies[body].n_frames();
    let markers = doc.project.trials[trial].rigid_bodies[body]
        .marker_indices
        .clone();
    // The loops below change only the trial's bodies (this one's references, and poses),
    // none of what the context is built from.
    let ctx = doc.pose_context(trial);
    let mut mean: Vec<Point3> = Vec::new();
    let mut count = 0usize;
    for f in 0..n {
        compute_coordinate_system(trial_mut(doc, trial), body, f);
        if !compute_pose_at(&ctx, doc, trial, body, f) {
            continue;
        }
        let t = &doc.project.trials[trial];
        if markers.iter().any(|&m| !status3d(t, m, f).has_data()) {
            continue;
        }
        count += 1;
        let pose = t.rigid_bodies[body].raw.poses[f];
        for (i, &m) in markers.iter().enumerate() {
            let pt = world_to_body(&pose, t.markers[m].points3d[f]);
            if count == 1 {
                mean.push(pt);
            } else {
                mean[i] = Point3::new(mean[i].x + pt.x, mean[i].y + pt.y, mean[i].z + pt.z);
            }
        }
    }
    if count == 0 {
        return;
    }
    for p in &mut mean {
        *p = divide(*p, count as f64);
    }
    let (shape, rotated) = shape_in_own_frame(&mean, markers.len());
    let rb = body_mut(doc, trial, body);
    rb.reference_points = shape;
    if rotated {
        rb.initialised = true;
    }
    for f in 0..n {
        compute_pose_at(&ctx, doc, trial, body, f);
    }
}

/// The mean distance between `src` and `dst` mapped back by the least-squares rigid fit of
/// `src` onto `dst`; 0 with fewer than 3 points.
///
/// Ported from core/RigidBody.cpp, RigidBody::fitAndComputeError (`:1340-1450`), through
/// [`alignment_error`] (the same arithmetic). Lists of different lengths (an out-of-range
/// read in the C++) give infinity, so they are never chosen. The C++ also sets
/// `initialised`; the caller does.
#[must_use]
pub(crate) fn fit_and_compute_error(src: &[Point3], dst: &[Point3]) -> f64 {
    if src.len() < 3 {
        return 0.0;
    }
    if src.len() != dst.len() {
        return f64::INFINITY;
    }
    alignment_error(src, dst) / dst.len() as f64
}

/// The most permutations' fits [`best_permutation`] evaluates before it gives up.
pub(crate) const PERMUTATION_FIT_BUDGET: u64 = 400_000;
/// The most search nodes [`best_permutation`] visits before it gives up.
pub(crate) const PERMUTATION_NODE_BUDGET: u64 = 20_000_000;

/// The search of [`best_permutation`].
struct PermutationSearch<'a> {
    src: &'a [Point3],
    dst: &'a [Point3],
    /// Distances between the file's points and between the reference points.
    dist_src: Vec<Vec<f64>>,
    dist_dst: Vec<Vec<f64>>,
    best_error: f64,
    best: Vec<usize>,
    /// A permutation's error known to be reachable (the C++ never accepts one above 100
    /// either); subtrees whose lower bound exceeds it are not entered.
    bound: f64,
    fits: u64,
    nodes: u64,
    exhausted: bool,
}

impl PermutationSearch<'_> {
    fn evaluate(&mut self, perm: &[usize]) -> f64 {
        self.fits += 1;
        let pts: Vec<Point3> = perm.iter().map(|&i| self.src[i]).collect();
        fit_and_compute_error(&pts, self.dst)
    }

    /// A lower bound of the mean fit error of every completion of `perm`: for positions
    /// `i`, `j` assigned, `| |s_i - s_j| - |d_i - d_j| | <= e_i + e_j` (an orthogonal map
    /// keeps distances), so over disjoint pairs the sum of these differences is at most the
    /// sum of all residuals, `n` times the mean.
    fn lower_bound(&self, perm: &[usize]) -> f64 {
        let k = perm.len();
        if k < 2 {
            return 0.0;
        }
        let mut pairs: Vec<(f64, usize, usize)> = Vec::with_capacity(k * (k - 1) / 2);
        for i in 0..k {
            for j in i + 1..k {
                let w = (self.dist_src[perm[i]][perm[j]] - self.dist_dst[i][j]).abs();
                pairs.push((w, i, j));
            }
        }
        pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut used = vec![false; k];
        let mut sum = 0.0;
        for (w, i, j) in pairs {
            if !used[i] && !used[j] && w.is_finite() {
                used[i] = true;
                used[j] = true;
                sum += w;
            }
        }
        sum / self.src.len() as f64
    }

    fn limit(&self) -> f64 {
        let b = self.best_error.min(self.bound);
        b + b.abs() * 1e-6 + 1e-9
    }

    fn search(&mut self, perm: &mut Vec<usize>, used: &mut [bool]) {
        if self.exhausted {
            return;
        }
        self.nodes += 1;
        if self.fits >= PERMUTATION_FIT_BUDGET || self.nodes >= PERMUTATION_NODE_BUDGET {
            self.exhausted = true;
            return;
        }
        let n = self.src.len();
        if perm.len() == n {
            let e = self.evaluate(perm);
            if e < self.best_error {
                self.best_error = e;
                self.best = perm.clone();
            }
            return;
        }
        for c in 0..n {
            if used[c] {
                continue;
            }
            perm.push(c);
            used[c] = true;
            if self.lower_bound(perm) <= self.limit() {
                self.search(perm, used);
            }
            perm.pop();
            used[c] = false;
            if self.exhausted {
                return;
            }
        }
    }
}

/// A permutation from matching each point's sorted distances to the others (greedy), to
/// bound the search early.
fn signature_guess(dist_src: &[Vec<f64>], dist_dst: &[Vec<f64>]) -> Vec<usize> {
    let n = dist_src.len();
    let sig = |d: &[Vec<f64>], i: usize| {
        let mut v: Vec<f64> = (0..n).filter(|&j| j != i).map(|j| d[i][j]).collect();
        v.sort_by(f64::total_cmp);
        v
    };
    let ss: Vec<Vec<f64>> = (0..n).map(|i| sig(dist_src, i)).collect();
    let sd: Vec<Vec<f64>> = (0..n).map(|i| sig(dist_dst, i)).collect();
    let mut costs: Vec<(f64, usize, usize)> = Vec::with_capacity(n * n);
    for (pos, d) in sd.iter().enumerate() {
        for (c, s) in ss.iter().enumerate() {
            let cost: f64 = d.iter().zip(s).map(|(a, b)| (a - b).abs()).sum();
            costs.push((cost, pos, c));
        }
    }
    costs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut perm = vec![usize::MAX; n];
    let mut taken = vec![false; n];
    for (_, pos, c) in costs {
        if perm[pos] == usize::MAX && !taken[c] {
            perm[pos] = c;
            taken[c] = true;
        }
    }
    perm
}

/// The permutation of `src` (position `j` takes `src[perm[j]]`) whose
/// [`fit_and_compute_error`] onto `dst` is smallest, and that error; the first in the order
/// of `std::next_permutation` from the identity among equal errors, as the C++ keeps the
/// first strictly smaller error it meets, starting from 1000000. Returns an empty
/// permutation and 1000000 when no permutation's error is below that. `None` when the
/// search gave up ([`PERMUTATION_FIT_BUDGET`], [`PERMUTATION_NODE_BUDGET`]).
///
/// Ported from core/RigidBody.cpp, RigidBody::setReferenceFromFile (`:2512-2539`), which
/// fits all n! permutations. Here the permutations are visited in the same order, but a
/// prefix is not completed when a lower bound of its error ([`PermutationSearch`]) exceeds
/// both the smallest error met so far and the error of a first guess (or 100, above which
/// the C++ rejects the file anyway). Every permutation the C++ could keep is therefore
/// still evaluated, so the result is the C++ one. With exact or nearly exact matches the
/// bound prunes at the second or third position, so bodies of 20 and more markers take
/// milliseconds; when the reference points are degenerate (all at one place: references
/// never set and no frame with every marker valid) nothing can be pruned and the search is
/// exhaustive, which stays within the budget up to 9 markers (9! = 362,880 fits); the
/// caller then takes the file's order ([`set_reference_from_file`], ledger A93).
pub(crate) fn best_permutation(src: &[Point3], dst: &[Point3]) -> Option<(Vec<usize>, f64)> {
    let n = src.len();
    let dist = |p: &[Point3]| -> Vec<Vec<f64>> {
        (0..p.len())
            .map(|i| (0..p.len()).map(|j| norm(sub(p[i], p[j]))).collect())
            .collect()
    };
    let dist_src = dist(src);
    let dist_dst = if dst.len() == n {
        dist(dst)
    } else {
        // Every fit gives infinity (fit_and_compute_error); no bound.
        vec![vec![f64::NAN; n]; n]
    };
    let mut s = PermutationSearch {
        src,
        dst,
        dist_src,
        dist_dst,
        best_error: 1_000_000.0,
        best: Vec::new(),
        bound: 100.0,
        fits: 0,
        nodes: 0,
        exhausted: false,
    };
    if n >= 3 && dst.len() == n {
        let guess = signature_guess(&s.dist_src, &s.dist_dst);
        let e = s.evaluate(&guess);
        if e.is_finite() {
            s.bound = s.bound.min(e);
        }
        s.fits = 0;
    }
    let mut perm = Vec::with_capacity(n);
    let mut used = vec![false; n];
    s.search(&mut perm, &mut used);
    if s.exhausted {
        return None;
    }
    Some((s.best, s.best_error))
}

/// The dialog text for `setReferenceFromFile`'s -1.
pub const REFERENCE_FILE_MISMATCH: &str = "Marker of the rigid body and the references supplied in the file are different or there has been a problem loading the file.";
/// The dialog text for `setReferenceFromFile`'s -2.
pub const REFERENCE_FILE_NOT_FITTED: &str =
    "The data in the marker could not be fitted. Please check the used correspondances.";

/// Reads a references file (`<name>_x,<name>_y,<name>_z,...` then the coordinates in the
/// same order): the names (the first of each three with "_x" removed) and the points, or
/// `None` when the file is empty. Shared by `setReferenceFromFile` and
/// `Trial::loadMarkersFromCSV`.
///
/// Ported from core/RigidBody.cpp, RigidBody::setReferenceFromFile (`:2480-2511`) and
/// core/Trial.cpp, Trial::loadMarkersFromCSV (`:787-813`): two `safeGetline` calls, each
/// line split at commas, values read with `QString::toDouble`.
pub(crate) fn read_reference_lines(data: &[u8]) -> (Vec<String>, Vec<String>) {
    let mut r = LineReader::new(data);
    let (names, _) = r.next_line();
    let (coords, _) = r.next_line();
    let names = qstring(names);
    let coords = qstring(coords);
    (
        split_commas(&names)
            .into_iter()
            .map(str::to_owned)
            .collect(),
        split_commas(&coords)
            .into_iter()
            .map(str::to_owned)
            .collect(),
    )
}

/// Point `i` of a references file: the name and the point.
pub(crate) fn reference_entry(names: &[String], coords: &[String], i: usize) -> (String, Point3) {
    (
        names[3 * i].replace("_x", ""),
        Point3::new(
            qt_to_double(&coords[3 * i]),
            qt_to_double(&coords[3 * i + 1]),
            qt_to_double(&coords[3 * i + 2]),
        ),
    )
}

/// True when every current reference point is the same point: the references were never
/// set and no frame has every marker in 3D, so `computeCoordinateSystemAverage` left the
/// mean shape at one place. Every order of a file's points then fits it equally (to
/// rounding).
fn degenerate_references(points: &[Point3]) -> bool {
    points.len() >= 3 && points.iter().all(|p| *p == points[0])
}

/// The Console line when a file's points are taken in their order (ledger A93).
fn file_order_message(body: usize, n: usize) -> String {
    format!(
        "Rigid body {}: its reference points were never set and no frame has all {n} of its markers in 3D, so every order of the file's points fits equally; the file's points were taken in their order (the first for the body's first marker, and so on).",
        body + 1
    )
}

/// Sets a body's references from a file: the file's points are matched to the current
/// reference points by the best permutation ([`best_permutation`]), and become the
/// reference points (original and working) and names in that order; references set 2, not
/// optimised, initialised; then recompute and filter. Returns Console lines.
///
/// Ported from core/RigidBody.cpp, RigidBody::setReferenceFromFile (`:2478-2563`). Errors:
/// a file that cannot be read is [`EditError::Io`]; -1 (field counts not 3 per marker) and
/// -2 (best mean error above 100) are [`EditError::Rejected`] with the dialog's text.
/// Nothing is changed on an error (the step is taken back).
///
/// Degenerate current references ([`degenerate_references`], ledger A93): every order fits
/// equally and the C++ keeps whichever rounding favours. Default: the file's order is taken
/// without a search, with a Console line. XMALab 3 mode
/// ([`xma_model::CppCompat::exhaustive_degenerate_reference_search`]): the exhaustive search
/// and its tie rule as long as it stays within the budget (up to 9 markers, where it gives
/// the C++ result), and beyond it the file's order, with the line. A search on references
/// that are not degenerate and that gives up (only possible from 10 markers on, when the
/// references neither match the file closely nor differ from it enough to prune) is
/// `Rejected` with its own text.
pub(crate) fn set_reference_from_file(
    doc: &mut Document,
    trial: usize,
    body: usize,
    path: &Path,
) -> Result<Vec<String>, EditError> {
    let data = read_file(path)?;
    let (names, coords) = read_reference_lines(&data);
    let n = doc.project.trials[trial].rigid_bodies[body]
        .marker_indices
        .len();
    if names.len() != 3 * n || coords.len() != 3 * n {
        return Err(EditError::Rejected(REFERENCE_FILE_MISMATCH.to_owned()));
    }
    let (file_names, file_points): (Vec<String>, Vec<Point3>) =
        (0..n).map(|i| reference_entry(&names, &coords, i)).unzip();
    if !doc.compat().nonfinite_imports_accepted {
        for p in &file_points {
            finite_point(*p, path)?;
        }
    }
    let dst = doc.project.trials[trial].rigid_bodies[body]
        .reference_points
        .clone();
    let degenerate = dst.len() == n && degenerate_references(&dst);
    let search = !degenerate || doc.compat().exhaustive_degenerate_reference_search;
    let mut messages = Vec::new();
    let found = if search {
        best_permutation(&file_points, &dst)
    } else {
        None
    };
    let (best, error) = match found {
        Some(found) => found,
        None if degenerate => {
            messages.push(file_order_message(body, n));
            ((0..n).collect(), 0.0)
        }
        None => {
            return Err(EditError::Rejected(format!(
                "The best match of the {n} points of the file to the rigid body's markers could not be found in reasonable time."
            )));
        }
    };
    if error > 100.0 || best.len() != n {
        return Err(EditError::Rejected(REFERENCE_FILE_NOT_FITTED.to_owned()));
    }
    let rb = body_mut(doc, trial, body);
    rb.reference_points = best.iter().map(|&i| file_points[i]).collect();
    rb.reference_points_original = rb.reference_points.clone();
    rb.reference_names = best.iter().map(|&i| file_names[i].clone()).collect();
    rb.has_optimized_coordinates = false;
    rb.references_set = 2;
    rb.initialised = true;
    cpp_recompute(doc, trial, body, true);
    Ok(messages)
}

/// The dialog text when a frame does not have every marker.
pub const REFERENCE_FRAME_INCOMPLETE: &str = "Cannot set references. All markers of the rigid body have to be detected in the selected frame!";

/// Sets a body's references from the markers' 3D points at a frame (names
/// "<description>_Frame<frame + 1>"), references set 2, initialised, not optimised; then
/// recompute and filter. `Rejected` when a marker is not valid in 3D there.
///
/// Ported from core/RigidBody.cpp, RigidBody::setReferenceFromFrame (`:2565-2591`).
pub(crate) fn set_reference_from_frame(
    doc: &mut Document,
    trial: usize,
    body: usize,
    frame: usize,
) -> Result<(), EditError> {
    let t = &doc.project.trials[trial];
    if frame >= t.n_frames {
        return Err(EditError::Invalid(format!(
            "frame {frame} is outside the trial ({} frames)",
            t.n_frames
        )));
    }
    let rb = &t.rigid_bodies[body];
    if rb
        .marker_indices
        .iter()
        .any(|&m| !status3d(t, m, frame).has_data())
    {
        return Err(EditError::Rejected(REFERENCE_FRAME_INCOMPLETE.to_owned()));
    }
    let (points, names): (Vec<Point3>, Vec<String>) = rb
        .marker_indices
        .iter()
        .map(|&m| {
            let marker = &t.markers[m];
            (
                marker.points3d[frame],
                format!("{}_Frame{}", marker.description, frame + 1),
            )
        })
        .unzip();
    let rb = body_mut(doc, trial, body);
    rb.reference_points = points.clone();
    rb.reference_points_original = points;
    rb.reference_names = names;
    rb.has_optimized_coordinates = false;
    rb.initialised = true;
    rb.references_set = 2;
    cpp_recompute(doc, trial, body, true);
    Ok(())
}

/// "optimized from tracked": the reference points go back to the original ones; when
/// `optimized` and the references are set, they become the mean, over the frames with a
/// pose where every marker is tracked or better in 3D (status above `INTERPOLATED`), of the
/// markers' 3D points mapped into the body by that frame's pose, recomputed first with the
/// original references. Not recomputed afterwards (the caller does).
///
/// Ported from core/RigidBody.cpp, RigidBody::setOptimized (`:823-889`). Ledger A82 fixed
/// in both modes: a frame whose recomputation finds no pose is left out (the C++ uses it
/// with the pose stored before, `:837-848`).
pub(crate) fn set_optimized(doc: &mut Document, trial: usize, body: usize, optimized: bool) {
    {
        let rb = body_mut(doc, trial, body);
        rb.has_optimized_coordinates = false;
        rb.reference_points = rb.reference_points_original.clone();
        if rb.references_set == 0 || !optimized {
            return;
        }
    }
    let n = doc.project.trials[trial].rigid_bodies[body].n_frames();
    let markers = doc.project.trials[trial].rigid_bodies[body]
        .marker_indices
        .clone();
    // The loop changes only the trial's body poses, none of what the context is built from.
    let ctx = doc.pose_context(trial);
    let mut good = Vec::new();
    for f in 0..n {
        let t = &doc.project.trials[trial];
        if !t.rigid_bodies[body]
            .raw
            .valid
            .get(f)
            .copied()
            .unwrap_or(false)
        {
            continue;
        }
        if markers
            .iter()
            .any(|&m| status3d(t, m, f).code() <= MarkerStatus::Interpolated.code())
        {
            continue;
        }
        let computed = compute_pose_at(&ctx, doc, trial, body, f);
        if computed {
            good.push(f);
        }
    }
    if good.is_empty() {
        return;
    }
    let t = &doc.project.trials[trial];
    let rb = &t.rigid_bodies[body];
    let mut sums = vec![Point3::origin(); markers.len()];
    for &f in &good {
        let pose = rb.raw.poses[f];
        for (i, &m) in markers.iter().enumerate() {
            let p = world_to_body(&pose, t.markers[m].points3d[f]);
            sums[i] = Point3::new(sums[i].x + p.x, sums[i].y + p.y, sums[i].z + p.z);
        }
    }
    let count = good.len() as f64;
    let points: Vec<Point3> = sums.into_iter().map(|p| divide(p, count)).collect();
    let rb = body_mut(doc, trial, body);
    rb.reference_points = points;
    rb.has_optimized_coordinates = true;
}

// ---------------------------------------------------------------------------------------
// Rigid body: virtual points
// ---------------------------------------------------------------------------------------

/// The fields of a file's second line (`littleHelper::safeGetline` twice, split at
/// commas).
fn second_line(data: &[u8]) -> Vec<String> {
    let mut r = LineReader::new(data);
    r.next_line();
    let (line, _) = r.next_line();
    split_commas(&qstring(line))
        .into_iter()
        .map(str::to_owned)
        .collect()
}

/// A point from the first three fields, or `Invalid` (the C++ reads past the list).
fn point_of(fields: &[String], path: &Path) -> Result<Point3, EditError> {
    if fields.len() < 3 {
        return Err(EditError::Invalid(format!(
            "{}: the second line must hold x, y and z",
            path.display()
        )));
    }
    Ok(Point3::new(
        qt_to_double(&fields[0]),
        qt_to_double(&fields[1]),
        qt_to_double(&fields[2]),
    ))
}

/// Ledger F52: `Invalid` naming the file's second line when a coordinate is NaN or
/// infinite (`nan`, `inf`, which `QString::toDouble` reads).
fn finite_point(p: Point3, path: &Path) -> Result<Point3, EditError> {
    if p.x.is_finite() && p.y.is_finite() && p.z.is_finite() {
        Ok(p)
    } else {
        Err(EditError::Invalid(format!(
            "{}, line 2: a coordinate is not a finite number",
            path.display()
        )))
    }
}

/// Ledger F52: the first line (1-based) of a virtual point's coordinate file whose row
/// [`read_virtual_coordinates`] reads as a point with a NaN or infinite coordinate.
pub(crate) fn virtual_coordinates_non_finite_line(data: &[u8]) -> Option<usize> {
    let mut r = LineReader::new(data);
    r.next_line();
    let mut n = 1;
    loop {
        let (line, eof) = r.next_line();
        if eof {
            return None;
        }
        n += 1;
        let text = qstring(line);
        let f = split_commas(&text);
        if f.len() == 3 && !f.contains(&"NaN") && f.iter().any(|v| !qt_to_double(v).is_finite()) {
            return Some(n);
        }
    }
}

/// The per-frame coordinates of an animated virtual point: a header line, then `x,y,z` per
/// frame; a line whose three fields include "NaN" is a frame without a value, lines with
/// another number of fields are skipped (so later rows move up a frame); missing frames at
/// the end, up to `n_frames`, have no value.
///
/// Ported from core/RigidBody.cpp, RigidBody::addDummyPoint (`:1978-2010`).
pub(crate) fn read_virtual_coordinates(data: &[u8], n_frames: usize) -> Vec<Option<Point3>> {
    let mut r = LineReader::new(data);
    r.next_line();
    let mut coords = Vec::new();
    loop {
        let (line, eof) = r.next_line();
        if eof {
            break;
        }
        let text = qstring(line);
        let f = split_commas(&text);
        if f.len() == 3 {
            coords.push(if f.contains(&"NaN") {
                None
            } else {
                Some(Point3::new(
                    qt_to_double(f[0]),
                    qt_to_double(f[1]),
                    qt_to_double(f[2]),
                ))
            });
        }
    }
    if coords.len() < n_frames {
        coords.resize(n_frames, None);
    }
    coords
}

/// Where an added virtual point's position comes from, as `addDummyPoint` takes it: the
/// chosen body (or none) and the second and coordinate files.
pub(crate) struct DummySource<'a> {
    /// The body chosen in the dialog, or `None` for an animated point.
    pub body: Option<usize>,
    /// The second CT file (the point in the driving body's frame).
    pub reference2: Option<&'a Path>,
    /// The file of per-frame coordinates.
    pub coordinates: Option<&'a Path>,
}

/// Adds a virtual point to a body. Nothing is recomputed (the caller does). Returns Console
/// lines.
///
/// Ported from core/RigidBody.cpp, RigidBody::addDummyPoint (`:1956-2031`): the point in
/// this body's frame is the second line of the reference file. For a point driven by a
/// body, the second line of the second file is the point in that body's frame. An animated
/// point whose coordinate file is not given reads no rows (the C++ opens an empty name), so
/// it has no value at any frame. A second line with fewer than three fields (the C++ reads
/// past the list, undefined behaviour) is `Invalid`.
///
/// The reference file's optional fourth value (the driving body's index, as `saveDummy`
/// writes it; -1 for an animated point) replaces the dialog's choice in the C++ (ledger A83):
///
/// - Default: the input given to the edit decides; a fourth value that names another source
///   is ignored, with a Console line saying so.
/// - [`xma_model::CppCompat::virtual_point_file_chooses_body`] (XMALab 3): the fourth value
///   decides where the C++ is defined: another body of the trial (the point follows it, at
///   the position the second file gives), or -1 for a point chosen to follow a body (an
///   animated point with no coordinates, as no coordinate file was asked for). Where the C++
///   crashes or reads out of range, the edit is `Rejected`: a body index for a point chosen
///   to be animated by coordinates (no second file was asked for; `coords_list.at(1)` of an
///   empty line), and an index that is not -1 nor a body of the trial.
pub(crate) fn add_dummy_point(
    doc: &mut Document,
    trial: usize,
    body: usize,
    name: &str,
    reference_file: &Path,
    source: &DummySource<'_>,
) -> Result<Vec<String>, EditError> {
    let fields = second_line(&read_file(reference_file)?);
    let reference = point_of(&fields, reference_file)?;
    let refuse = !doc.compat().nonfinite_imports_accepted;
    if refuse {
        finite_point(reference, reference_file)?;
    }
    let chosen: i64 = source.body.map_or(-1, |b| b as i64);
    let mut marker_id = chosen;
    let mut messages = Vec::new();
    let t = &doc.project.trials[trial];
    let n_frames = t.rigid_bodies[body].n_frames();
    let n_bodies = t.rigid_bodies.len();
    let describe = |id: i64| {
        if id == -1 {
            "animated by a file of coordinates".to_owned()
        } else {
            format!("driven by rigid body {}", id + 1)
        }
    };
    if fields.len() > 3 {
        let value = i64::from(qt_to_int(&fields[3]));
        if doc.compat().virtual_point_file_chooses_body {
            if value != -1 && usize::try_from(value).map_or(true, |b| b >= n_bodies) {
                return Err(EditError::Rejected(format!(
                    "The virtual point file {} names rigid body index {value} in its fourth column, but the trial has {n_bodies} rigid bodies (indices 0 to {}, or -1 for a point animated by coordinates).",
                    reference_file.display(),
                    n_bodies.saturating_sub(1)
                )));
            }
            if value != -1 && source.reference2.is_none() {
                return Err(EditError::Rejected(format!(
                    "The virtual point file {} names rigid body index {value} in its fourth column, but the point was chosen to be animated by coordinates, so no file gives its position in that body (XMALab 3 crashes here).",
                    reference_file.display()
                )));
            }
            marker_id = value;
        } else if value != chosen {
            messages.push(format!(
                "Virtual point \"{name}\": the fourth value of {} ({value}, a point {}) was ignored; the point is {} as chosen.",
                reference_file.display(),
                describe(value),
                describe(chosen)
            ));
        }
    }
    let source = if marker_id == -1 {
        let coords = match source.coordinates {
            Some(p) => {
                let data = read_file(p)?;
                let coords = read_virtual_coordinates(&data, n_frames);
                if refuse && let Some(line) = virtual_coordinates_non_finite_line(&data) {
                    return Err(EditError::Invalid(format!(
                        "{}, line {line}: a coordinate is not a finite number",
                        p.display()
                    )));
                }
                coords
            }
            None => vec![None; n_frames],
        };
        VirtualPointSource::Coordinates(coords)
    } else {
        let driver = usize::try_from(marker_id)
            .ok()
            .filter(|&b| b < n_bodies)
            .ok_or_else(|| {
                EditError::Invalid(format!(
                    "rigid body {marker_id} does not exist ({n_bodies} bodies)"
                ))
            })?;
        let Some(path2) = source.reference2 else {
            return Err(EditError::Invalid(format!(
                "{}: the point follows rigid body {} but no file gives its position there",
                reference_file.display(),
                driver + 1
            )));
        };
        let point = point_of(&second_line(&read_file(path2)?), path2)?;
        if refuse {
            finite_point(point, path2)?;
        }
        VirtualPointSource::OtherBody {
            body: driver,
            point,
        }
    };
    body_mut(doc, trial, body)
        .virtual_points
        .push(VirtualPoint {
            name: name.to_owned(),
            reference,
            source,
        });
    Ok(messages)
}

// ---------------------------------------------------------------------------------------
// Copying a body
// ---------------------------------------------------------------------------------------

/// Makes body `body` of `trial` a copy of `src` (a body of another trial): description,
/// markers (each through `addPointIdx` without its second recomputation), references when
/// the source has them (no recomputation after), mesh, virtual points driven by bodies,
/// optimised flag, cutoff override and cutoff, colour, visibility, initialised. Used by
/// "Apply to trials" (E27) and "Copy From trial" (E70).
///
/// Ported from core/RigidBody.cpp, RigidBody::copyData (`:82-137`), with D4 fixed:
///
/// - the C++ tests `this->dummyRBIndex[i]` of a list it has just cleared (an out-of-range
///   read); the intended test of the source's index is used: virtual points driven by a
///   body are copied, those animated by per-frame coordinates (data of the source trial's
///   frames) are not;
/// - the cutoff frequency is copied, not the override flag (ledger A80, both modes);
/// - the body is initialised when its references are set, not as the source was (ledger
///   A81, both modes).
///
/// The mesh: as `addMeshModel` with the source's file name, which removes the body's mesh
/// first; in XMALab 3 (`CppCompat::drop_unreadable_mesh`, ledger F18) a mesh whose file
/// does not open is not kept, and the draw flag is copied only with a mesh. Reference lists
/// shorter or longer than the marker list (undefined behaviour in the C++) are copied as
/// far as both go.
pub(crate) fn copy_data(doc: &mut Document, trial: usize, body: usize, src: &RigidBody) {
    let drop_unreadable = doc.compat().drop_unreadable_mesh;
    body_mut(doc, trial, body).description = src.description.clone();
    clear_point_idx(doc, trial, body);
    for &m in &src.marker_indices {
        add_point_idx(doc, trial, body, m, false);
    }
    let rb = body_mut(doc, trial, body);
    if src.references_set != 0 {
        rb.references_set = src.references_set;
        for (dst, s) in rb.reference_names.iter_mut().zip(&src.reference_names) {
            dst.clone_from(s);
        }
        for (i, s) in src.reference_points.iter().enumerate() {
            if let Some(d) = rb.reference_points.get_mut(i) {
                *d = *s;
            }
            if let (Some(d), Some(s)) = (
                rb.reference_points_original.get_mut(i),
                src.reference_points_original.get(i),
            ) {
                *d = *s;
            }
        }
    }
    let loads = match &src.mesh_model {
        Some(path) => !drop_unreadable || xma_io::project_read::mesh_opens(path),
        None => false,
    };
    if loads {
        rb.mesh_model.clone_from(&src.mesh_model);
        rb.mesh_readable = src.mesh_readable;
        rb.draw_mesh_model = src.draw_mesh_model;
    } else {
        rb.mesh_model = None;
        rb.mesh_readable = None;
        if !drop_unreadable {
            rb.draw_mesh_model = src.draw_mesh_model;
        }
    }
    rb.virtual_points = src
        .virtual_points
        .iter()
        .filter(|v| matches!(v.source, VirtualPointSource::OtherBody { .. }))
        .cloned()
        .collect();
    rb.has_optimized_coordinates = src.has_optimized_coordinates;
    rb.override_cutoff_frequency = src.override_cutoff_frequency;
    rb.cutoff_frequency = src.cutoff_frequency;
    rb.color = src.color;
    rb.mesh_scale = src.mesh_scale;
    rb.visible = src.visible;
    rb.initialised = rb.references_set != 0;
}

/// `Marker::setMaxPenalty`: 125 (an old erroneous default) becomes 50, values above 100
/// become 100.
///
/// Ported from core/Marker.cpp, Marker::setMaxPenalty (`:1081-1089`).
#[must_use]
pub(crate) fn capped_penalty(value: i32) -> i32 {
    if value == 125 { 50 } else { value.min(100) }
}

/// An outcome with the given Console lines.
#[must_use]
pub(crate) fn outcome(messages: Vec<String>) -> EditOutcome {
    EditOutcome {
        messages,
        refused: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f64, y: f64, z: f64) -> Point3 {
        Point3::new(x, y, z)
    }

    /// Every permutation in `std::next_permutation` order, as the C++ loop.
    fn brute_force(src: &[Point3], dst: &[Point3]) -> (Vec<usize>, f64) {
        fn next_permutation(v: &mut [usize]) -> bool {
            let n = v.len();
            if n < 2 {
                return false;
            }
            let mut i = n - 1;
            while i > 0 && v[i - 1] >= v[i] {
                i -= 1;
            }
            if i == 0 {
                v.reverse();
                return false;
            }
            let mut j = n - 1;
            while v[j] <= v[i - 1] {
                j -= 1;
            }
            v.swap(i - 1, j);
            v[i..].reverse();
            true
        }
        let mut perm: Vec<usize> = (0..src.len()).collect();
        let mut best = Vec::new();
        let mut error = 1_000_000.0;
        loop {
            let pts: Vec<Point3> = perm.iter().map(|&i| src[i]).collect();
            let e = fit_and_compute_error(&pts, dst);
            if e < error {
                error = e;
                best = perm.clone();
            }
            if !next_permutation(&mut perm) {
                break;
            }
        }
        (best, error)
    }

    /// A deterministic pseudo-random sequence (no dependency).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    fn cloud(rng: &mut Lcg, n: usize) -> Vec<Point3> {
        (0..n)
            .map(|_| p(rng.next() * 10.0, rng.next() * 10.0, rng.next() * 10.0))
            .collect()
    }

    fn shuffle(rng: &mut Lcg, v: &mut [Point3]) -> Vec<usize> {
        let mut order: Vec<usize> = (0..v.len()).collect();
        for i in (1..v.len()).rev() {
            let j = (rng.next() * (i + 1) as f64) as usize;
            order.swap(i, j);
        }
        let copy = v.to_vec();
        for (k, &o) in order.iter().enumerate() {
            v[k] = copy[o];
        }
        order
    }

    /// The pruned search gives the C++ result on every case small enough to enumerate: exact
    /// matches, noisy ones, mismatched shapes, ties from symmetric shapes and degenerate
    /// (coincident) references.
    #[test]
    fn pruned_search_equals_full_enumeration() {
        let mut rng = Lcg(7);
        for n in 1..=7 {
            for case in 0..6 {
                let dst = cloud(&mut rng, n);
                let mut src = match case {
                    // A rigid copy with noise.
                    0 | 1 => dst
                        .iter()
                        .map(|q| p(q.y + 3.0 + rng.next() * 0.01, -q.x, q.z - 1.0))
                        .collect(),
                    // Unrelated points.
                    2 => cloud(&mut rng, n),
                    // Coincident references (never set).
                    3 => cloud(&mut rng, n),
                    // A symmetric shape (ties).
                    4 => (0..n)
                        .map(|k| {
                            let a = k as f64 * std::f64::consts::TAU / n as f64;
                            p(a.cos(), a.sin(), 0.0)
                        })
                        .collect(),
                    _ => dst.clone(),
                };
                shuffle(&mut rng, &mut src);
                let dst = match case {
                    3 => vec![Point3::origin(); n],
                    4 => (0..n)
                        .map(|k| {
                            let a = k as f64 * std::f64::consts::TAU / n as f64;
                            p(a.cos() * 2.0, a.sin() * 2.0, 1.0)
                        })
                        .collect(),
                    _ => dst,
                };
                let expected = brute_force(&src, &dst);
                let got = best_permutation(&src, &dst).expect("within budget");
                assert_eq!(got.0, expected.0, "n {n} case {case}");
                assert_eq!(got.1.to_bits(), expected.1.to_bits(), "n {n} case {case}");
            }
        }
    }

    /// Large bodies stay fast when the file matches the body: the permutation that undoes
    /// the shuffle is found.
    #[test]
    fn large_bodies_are_matched_quickly() {
        let mut rng = Lcg(11);
        for n in [10, 15, 20, 30] {
            let dst = cloud(&mut rng, n);
            let mut src: Vec<Point3> = dst
                .iter()
                .map(|q| p(q.x + 0.001 * rng.next(), q.y + 5.0, q.z))
                .collect();
            let order = shuffle(&mut rng, &mut src);
            let start = std::time::Instant::now();
            let (perm, e) = best_permutation(&src, &dst).expect("within budget");
            let elapsed = start.elapsed();
            // Position j of the reference takes the file's point that came from j.
            for (j, &k) in perm.iter().enumerate() {
                assert_eq!(order[k], j, "n {n}");
            }
            assert!(e < 0.01);
            assert!(elapsed.as_secs_f64() < 2.0, "n {n}: {elapsed:?}");
        }
    }

    /// With degenerate references nothing can be pruned: up to 9 points the full search
    /// runs; beyond the budget the search gives up instead of hanging.
    #[test]
    fn degenerate_references_give_up_beyond_the_budget() {
        let mut rng = Lcg(3);
        let src = cloud(&mut rng, 9);
        assert!(best_permutation(&src, &vec![Point3::origin(); 9]).is_some());
        let src = cloud(&mut rng, 12);
        let start = std::time::Instant::now();
        assert!(best_permutation(&src, &vec![Point3::origin(); 12]).is_none());
        assert!(start.elapsed().as_secs_f64() < 20.0);
    }

    #[test]
    fn fewer_than_three_points_keep_the_file_order() {
        let src = [p(1.0, 0.0, 0.0), p(0.0, 1.0, 0.0)];
        let dst = [p(0.0, 1.0, 0.0), p(1.0, 0.0, 0.0)];
        assert_eq!(best_permutation(&src, &dst), Some((vec![0, 1], 0.0)));
    }

    #[test]
    fn penalty_is_capped_as_the_cpp() {
        assert_eq!(capped_penalty(125), 50);
        assert_eq!(capped_penalty(200), 100);
        assert_eq!(capped_penalty(-3), -3);
        assert_eq!(capped_penalty(70), 70);
    }

    #[test]
    fn virtual_coordinates_are_read_as_the_cpp() {
        let data = b"x,y,z\n1,2,3\nNaN,0,0\n1,2\n4,5,6\r\n7,8,nan";
        let c = read_virtual_coordinates(data, 6);
        assert_eq!(c.len(), 6);
        assert_eq!(c[0], Some(p(1.0, 2.0, 3.0)));
        assert_eq!(c[1], None);
        assert_eq!(c[2], Some(p(4.0, 5.0, 6.0)));
        // "nan" (lower case) is not "NaN": a value, NaN.
        assert!(c[3].is_some_and(|q| q.z.is_nan()));
        assert_eq!(c[4], None);
    }
}
