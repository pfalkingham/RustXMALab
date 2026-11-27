//! Ports of the C++ model calls that edits share: the marker-level calls of
//! `docs/cpp-editing.md` section 1.1 (`Marker::setPoint`, `Marker::setSize`, `Marker::reset`,
//! `Marker::resetMultipleFrames`, `Marker::interpolate`) and `Trial::resetRigidBodyByMarker`,
//! each with the recomputation the C++ call triggers (XMALab 3 mode) or that keeps derived
//! data consistent (default mode).
//!
//! These are functions on the [`Document`], used by the edits of [`super`], by the other
//! editing files and by the tracking loop, which calls [`Document::set_point`] and
//! [`Document::set_size`] for every tracked point. Each call does the work of one frame
//! (one marker's size statistics for [`Document::set_size`]); nothing here works on a whole
//! trial per call.
//!
//! # What a point edit recomputes
//!
//! After the 2D data of a marker at a frame change, the marker's 3D point at that frame is
//! triangulated again (`Marker::reconstruct3DPoint(frame)`), then rigid bodies at that frame:
//!
//! - **XMALab 3** ([`CppCompat::edit_first_body_only`], ledger A31): `computePose(frame)` on
//!   the first body that lists the marker, with the other bodies' pose flags as stored
//!   (`Trial::resetRigidBodyByMarker`); nothing for a `NO_CALIBRATION` project, where
//!   `reconstruct3DPoint` returns first.
//! - **Fixed**: every body that lists the marker, and every body with a virtual point
//!   driven by one of those (repeatedly), as the trial update computes them
//!   ([`Document::recompute_bodies_at_frame`]).
//!
//! Then, unless [`CppCompat::edits_not_refiltered`] (ledger A60), the continuity step is
//! carried on from that frame ([`Document::continue_rotations`]) and those bodies are marked
//! stale, so that [`Document::settle`] filters them once when the step is committed.

use std::collections::BTreeSet;
use std::sync::Arc;

use xma_model::marker::{UNSET_2D, UNSET_3D, UNSET_SIZE};
use xma_model::{
    CppCompat, Interpolation, Marker, MarkerStatus, Point2, Point3, VirtualPointSource,
};
use xma_vision::filter::cubic_spline;

use super::{CameraSel, EditError, FrameRange};
use crate::document::Document;
use crate::markers::{reconstruct_3d_point, update_mean_size};

/// Checks a trial index; the Default trial holds no tracking data.
pub(crate) fn check_trial(doc: &Document, trial: usize) -> Result<(), EditError> {
    let t = doc
        .project
        .trials
        .get(trial)
        .ok_or_else(|| EditError::Invalid(format!("no trial {trial}")))?;
    if t.is_default {
        return Err(EditError::Invalid(
            "the Default trial holds no tracking data".to_owned(),
        ));
    }
    Ok(())
}

/// Checks a trial and marker index (any trial, the Default one included).
pub(crate) fn check_marker(doc: &Document, trial: usize, marker: usize) -> Result<(), EditError> {
    let t = doc
        .project
        .trials
        .get(trial)
        .ok_or_else(|| EditError::Invalid(format!("no trial {trial}")))?;
    if marker >= t.markers.len() {
        return Err(EditError::Invalid(format!(
            "trial {trial} has no marker {marker}"
        )));
    }
    Ok(())
}

/// Checks a trial (not the Default one), a marker and a frame range of its data, and that
/// every camera of the marker holds every frame.
pub(crate) fn check_frames(
    doc: &Document,
    trial: usize,
    marker: usize,
    frames: FrameRange,
) -> Result<(), EditError> {
    check_trial(doc, trial)?;
    check_marker(doc, trial, marker)?;
    let m = &doc.project.trials[trial].markers[marker];
    let n = marker_frames(m);
    if frames.first <= frames.last && frames.last >= n {
        return Err(EditError::Invalid(format!(
            "frame {} is outside marker {marker}'s {n} frames",
            frames.last
        )));
    }
    Ok(())
}

/// Checks one cell: trial (not the Default one), marker, camera and frame.
pub(crate) fn check_cell(
    doc: &Document,
    trial: usize,
    marker: usize,
    camera: usize,
    frame: usize,
) -> Result<(), EditError> {
    check_frames(doc, trial, marker, FrameRange::one(frame))?;
    check_camera(doc, trial, marker, CameraSel::One(camera))
}

/// Checks a camera selection against a marker's cameras.
pub(crate) fn check_camera(
    doc: &Document,
    trial: usize,
    marker: usize,
    cameras: CameraSel,
) -> Result<(), EditError> {
    if let CameraSel::One(c) = cameras {
        let n = doc.project.trials[trial].markers[marker].n_cameras();
        if c >= n {
            return Err(EditError::Invalid(format!("no camera {c} (of {n})")));
        }
    }
    Ok(())
}

/// The number of frames every array of a marker holds.
fn marker_frames(m: &Marker) -> usize {
    let mut n = m.n_frames().min(m.points3d.len()).min(m.error3d.len());
    for v in &m.views {
        n = n
            .min(v.points.len())
            .min(v.projected.len())
            .min(v.status.len())
            .min(v.error.len())
            .min(v.size.len());
    }
    n
}

/// The cameras of a selection, in order.
pub(crate) fn cameras_of(sel: CameraSel, n_cameras: usize) -> std::ops::Range<usize> {
    match sel {
        CameraSel::All => 0..n_cameras,
        CameraSel::One(c) => c..c + 1,
    }
}

/// True for a status that is a negative mark (`UNTRACKABLE`, `DELETED`, `LOST`).
fn is_mark(s: MarkerStatus) -> bool {
    s < MarkerStatus::Undefined
}

/// Known frames for interpolation: `status >= INTERPOLATED`.
fn known(s: MarkerStatus) -> bool {
    s >= MarkerStatus::Interpolated
}

impl Document {
    /// One marker of a trial, for writing; moves [`Document::changes`] (every write of a
    /// marker's data goes through here, inside an edit or not: a tracking run's writes).
    pub(crate) fn marker_mut(&mut self, trial: usize, marker: usize) -> &mut Marker {
        self.note_change();
        let t = Arc::make_mut(&mut self.project.trials[trial]);
        Arc::make_mut(&mut t.markers[marker])
    }

    /// Writes a 2D point and its status, then recomputes the frame.
    ///
    /// Ported from core/Marker.cpp, Marker::setPoint (`:152-159`): `points2D` and `status2D`
    /// are written (not the size), then `reconstruct3DPoint(frame)` ([`Document::reconstruct_frame`]).
    ///
    /// # Errors
    /// [`EditError::Invalid`] for an index out of range or the Default trial.
    pub(crate) fn set_point(
        &mut self,
        trial: usize,
        marker: usize,
        camera: usize,
        frame: usize,
        point: Point2,
        status: MarkerStatus,
    ) -> Result<(), EditError> {
        check_cell(self, trial, marker, camera, frame)?;
        let m = self.marker_mut(trial, marker);
        m.views[camera].points[frame] = point;
        m.views[camera].status[frame] = status;
        self.reconstruct_frame(trial, marker, frame);
        Ok(())
    }

    /// Writes a detected size, then the mean size and size range.
    ///
    /// Ported from core/Marker.cpp, Marker::setSize (`:628-632`) and Marker::updateMeanSize
    /// ([`update_mean_size`], ledger A1 behind [`CppCompat::size_range_from_sum`]). Nothing
    /// else is recomputed.
    ///
    /// # Errors
    /// [`EditError::Invalid`] for an index out of range or the Default trial.
    pub(crate) fn set_size(
        &mut self,
        trial: usize,
        marker: usize,
        camera: usize,
        frame: usize,
        size: f64,
    ) -> Result<(), EditError> {
        check_cell(self, trial, marker, camera, frame)?;
        let from_sum = self.compat().size_range_from_sum;
        let m = self.marker_mut(trial, marker);
        m.views[camera].size[frame] = size;
        update_mean_size(m, from_sum);
        Ok(())
    }

    /// Deletes one cell's data and recomputes the frame.
    ///
    /// Ported from core/Marker.cpp, Marker::reset (`:1042-1054`): status `UNDEFINED`, 2D
    /// point and projected point (-2, -2), size -1, `error2D` 0; then
    /// `reconstruct3DPoint(frame)` and `updateMeanSize`.
    ///
    /// # Errors
    /// [`EditError::Invalid`] for an index out of range or the Default trial.
    pub(crate) fn reset_point(
        &mut self,
        trial: usize,
        marker: usize,
        camera: usize,
        frame: usize,
    ) -> Result<(), EditError> {
        check_cell(self, trial, marker, camera, frame)?;
        let from_sum = self.compat().size_range_from_sum;
        self.marker_mut(trial, marker).views[camera].reset(frame);
        self.reconstruct_frame(trial, marker, frame);
        update_mean_size(self.marker_mut(trial, marker), from_sum);
        Ok(())
    }

    /// Deletes a marker's data over a frame range, or marks it untrackable.
    ///
    /// Ported from core/Marker.cpp, Marker::resetMultipleFrames (`:857-902`). For each frame
    /// `first..=last` and each selected camera: 2D point (-2, -2), status `UNTRACKABLE` or
    /// `UNDEFINED`, `error2D` 0, size -1, projected y -2 and, unless
    /// [`CppCompat::reset_keeps_projected_x`] (ledger A68), projected x -2 (XMALab 3 writes
    /// the 2D x twice instead); then the 3D point (-1000, -1000, -1000), `status3D`
    /// `UNDEFINED`, `error3D` 0 and `reconstruct3DPoint(frame)`, so the other cameras can
    /// still triangulate the frame. `updateMeanSize` once at the end (also for an empty
    /// range, as the C++). The rigid bodies after each frame's triangulation go through `d`
    /// ([`Deferred`]: per frame in the XMALab 3 mode, once at the end otherwise, ledger A216).
    ///
    /// # Errors
    /// [`EditError::Invalid`] for an index out of range or the Default trial.
    pub(crate) fn reset_frames(
        &mut self,
        d: &mut Deferred,
        trial: usize,
        marker: usize,
        cameras: CameraSel,
        frames: FrameRange,
        untrackable: bool,
    ) -> Result<(), EditError> {
        check_frames(self, trial, marker, frames)?;
        check_camera(self, trial, marker, cameras)?;
        let compat = self.compat();
        let status = if untrackable {
            MarkerStatus::Untrackable
        } else {
            MarkerStatus::Undefined
        };
        if frames.first <= frames.last {
            for f in frames.first..=frames.last {
                let m = self.marker_mut(trial, marker);
                for c in cameras_of(cameras, m.n_cameras()) {
                    let v = &mut m.views[c];
                    v.points[f] = Point2::new(UNSET_2D, UNSET_2D);
                    if !compat.reset_keeps_projected_x {
                        v.projected[f].x = UNSET_2D;
                    }
                    v.projected[f].y = UNSET_2D;
                    v.status[f] = status;
                    v.error[f] = 0.0;
                    v.size[f] = UNSET_SIZE;
                }
                m.points3d[f] = Point3::new(UNSET_3D, UNSET_3D, UNSET_3D);
                m.status3d[f] = MarkerStatus::Undefined;
                m.error3d[f] = 0.0;
                self.deferred_reconstruct_frame(d, trial, marker, f);
            }
        }
        update_mean_size(self.marker_mut(trial, marker), compat.size_range_from_sum);
        Ok(())
    }

    /// Triangulates one frame of a marker again and recomputes the rigid bodies at that
    /// frame (see the module documentation for which, in each mode).
    ///
    /// Ported from core/Marker.cpp, Marker::reconstruct3DPoint with `updateAll = false`
    /// (`:209-237`; the triangulation is [`reconstruct_3d_point`]), then
    /// [`Document::reset_rigid_body_by_marker`]. Indices must be valid.
    pub(crate) fn reconstruct_frame(&mut self, trial: usize, marker: usize, frame: usize) {
        let ctx = self.frame_context(trial);
        let patch = reconstruct_3d_point(&ctx, &self.project.trials[trial].markers[marker], frame);
        patch.apply(self.marker_mut(trial, marker));
        // Marker.cpp:212-213: a NO_CALIBRATION project returns before the bodies.
        if ctx.no_calibration && self.compat().edit_first_body_only {
            return;
        }
        self.reset_rigid_body_by_marker(trial, marker, frame);
    }

    /// Recomputes the rigid bodies at a frame after one of their markers changed there.
    ///
    /// Ported from core/Trial.cpp, Trial::resetRigidBodyByMarker (`:1514-1527`): with
    /// [`CppCompat::edit_first_body_only`] (ledger A31), `computePose(frame)` on the first
    /// body (in index order) whose marker list holds the marker, with every other body's
    /// state as stored (a virtual point driven by a body whose flag is clear computes that
    /// body there and then, and that is written too). Fixed: every body that lists the
    /// marker, and every body with a virtual point driven by one of those, repeatedly, as
    /// the trial update computes them ([`Document::recompute_bodies_at_frame`]). Unless
    /// [`CppCompat::edits_not_refiltered`] (ledger A60), the bodies recomputed are marked
    /// stale.
    pub(crate) fn reset_rigid_body_by_marker(&mut self, trial: usize, marker: usize, frame: usize) {
        let compat = self.compat();
        let bodies = if compat.edit_first_body_only {
            let t = &self.project.trials[trial];
            let first = t.rigid_bodies.iter().position(|b| {
                b.marker_indices
                    .iter()
                    .any(|&i| i < t.markers.len() && i == marker)
            });
            let Some(b) = first else {
                return;
            };
            self.compute_pose_as_stored(trial, frame, b);
            vec![b]
        } else {
            let bodies = bodies_affected_by(&self.project.trials[trial], marker);
            self.recompute_bodies_at_frame(trial, frame, &bodies);
            if !compat.edits_not_refiltered {
                for &b in &bodies {
                    self.continue_rotations(trial, b, frame);
                }
            }
            bodies
        };
        self.mark_bodies_stale(trial, &bodies, compat);
    }

    /// `computePose(frame)` on one body with every body's state as stored, writing what it
    /// changed (the body, and other bodies computed on demand by its virtual points).
    ///
    /// Ported from core/RigidBody.cpp, RigidBody::computePose (`:1065-1229`), called alone.
    pub(crate) fn compute_pose_as_stored(&mut self, trial: usize, frame: usize, body: usize) {
        let ctx = self.pose_context(trial);
        let t = &self.project.trials[trial];
        let mut states = crate::rigid_bodies::FrameStates::read(t, frame);
        crate::rigid_bodies::compute_pose(&ctx, t, &mut states, body);
        let t = Arc::make_mut(&mut self.project.trials[trial]);
        states.write(t);
    }

    /// Marks bodies stale unless the mode leaves edits unfiltered (ledger A60).
    pub(crate) fn mark_bodies_stale(&mut self, trial: usize, bodies: &[usize], compat: CppCompat) {
        if compat.edits_not_refiltered {
            return;
        }
        for &b in bodies {
            self.mark_body_stale(trial, b);
        }
    }

    /// Fills a marker's gaps by each frame's interpolation method.
    ///
    /// Ported from core/Marker.cpp, Marker::interpolate (`:1217-1481`), with every write
    /// through [`Document::set_point`] in the C++ order (its rigid bodies through `d`,
    /// [`Deferred`]: per write in the XMALab 3 mode, once at the end otherwise, ledger A216):
    ///
    /// 1. Reset: with [`CppCompat::interpolate_clears_marks`] (ledger A61) every cell whose
    ///    status is at or below `INTERPOLATED` (the negative marks included) becomes
    ///    (-2, -2) `UNDEFINED`. Fixed: only `PREDICTED` and `INTERPOLATED` cells; the
    ///    marks and the `UNDEFINED` cells are left, and a marked cell is never written by
    ///    the steps below (a mark inside a gap does not end it).
    /// 2. Repeat (frames with method `REPEAT`, from frame 1): in 2D, a cell after a known
    ///    one (`>= INTERPOLATED`) takes its point; with `interpolate_3d`, every camera not
    ///    known at the frame takes the projection of the previous frame's 3D point when that
    ///    is known.
    /// 3. Linear: gaps bounded by known frames, filled at the frames whose method is
    ///    `LINEAR`, per camera in 2D or on the 3D point and projected.
    /// 4. Cubic: a not-a-knot spline ([`cubic_spline`]) through every known frame, at the
    ///    gap frames whose method is `CUBIC`, per camera in 2D (x and y) or on the 3D point.
    ///    With [`CppCompat::cubic_interpolation_stops`] (ledger A33) XMALab 3 returns from
    ///    the whole function at the first camera with fewer than 4 known frames
    ///    (`:1371-1372`), with no gap (`:1395-1396`), or where the spline fails
    ///    (`:1398-1401`); fixed, that camera is skipped. No frame before 0 is read when frame 0 is a known cubic frame (ledger
    ///    A47; the C++ reads `status2D[c][-1]`).
    ///
    /// # Errors
    /// [`EditError::Invalid`] for an index out of range or the Default trial.
    pub(crate) fn interpolate_marker(
        &mut self,
        d: &mut Deferred,
        trial: usize,
        marker: usize,
    ) -> Result<(), EditError> {
        check_trial(self, trial)?;
        check_marker(self, trial, marker)?;
        let compat = self.compat();
        let clear_marks = compat.interpolate_clears_marks;
        // Ledger A33 (and :1395-1401): XMALab 3 returns from the whole function.
        let cubic_stops = compat.cubic_interpolation_stops;
        let interpolate_3d = self.project.trials[trial].interpolate_3d;
        let (n, n_cam) = {
            let m = &self.project.trials[trial].markers[marker];
            (marker_frames(m).min(m.interpolation.len()), m.n_cameras())
        };
        let unset = Point2::new(UNSET_2D, UNSET_2D);

        // 1. Reset.
        for c in 0..n_cam {
            for i in 0..n {
                let s = self.project.trials[trial].markers[marker].views[c].status[i];
                let reset = if clear_marks {
                    s <= MarkerStatus::Interpolated
                } else {
                    s == MarkerStatus::Predicted || s == MarkerStatus::Interpolated
                };
                if reset {
                    self.deferred_set_point(
                        d,
                        trial,
                        marker,
                        c,
                        i,
                        unset,
                        MarkerStatus::Undefined,
                    )?;
                }
            }
        }

        let methods = self.project.trials[trial].markers[marker]
            .interpolation
            .clone();
        let requires = |m: Interpolation| methods.iter().take(n).skip(1).any(|&x| x == m);
        // A cell the steps below may write: not known, and not a mark (fixed mode).
        let fillable = |s: MarkerStatus| !known(s) && (clear_marks || !is_mark(s));
        let status2d = |doc: &Document, c: usize, f: usize| {
            doc.project.trials[trial].markers[marker].views[c].status[f]
        };

        // 2. Repeat.
        if requires(Interpolation::Repeat) {
            for (f, &method) in methods.iter().enumerate().take(n).skip(1) {
                if method != Interpolation::Repeat {
                    continue;
                }
                if interpolate_3d {
                    let mk = &self.project.trials[trial].markers[marker];
                    if known(mk.status3d[f - 1]) {
                        let p3 = mk.points3d[f - 1];
                        for c in 0..n_cam {
                            if fillable(status2d(self, c, f))
                                && let Some(p) = self.project_into(trial, c, p3)
                            {
                                self.deferred_set_point(
                                    d,
                                    trial,
                                    marker,
                                    c,
                                    f,
                                    p,
                                    MarkerStatus::Interpolated,
                                )?;
                            }
                        }
                    }
                } else {
                    for c in 0..n_cam {
                        let v = &self.project.trials[trial].markers[marker].views[c];
                        if known(v.status[f - 1]) && fillable(v.status[f]) {
                            let p = v.points[f - 1];
                            self.deferred_set_point(
                                d,
                                trial,
                                marker,
                                c,
                                f,
                                p,
                                MarkerStatus::Interpolated,
                            )?;
                        }
                    }
                }
            }
        }

        // 3. Linear. A gap's ends are read before its frames are written, which changes
        // neither (each write recomputes its own frame only).
        if requires(Interpolation::Linear) {
            if interpolate_3d {
                let mut f = 1;
                while f < n {
                    let gap = {
                        let mk = &self.project.trials[trial].markers[marker];
                        linear_gap(&mk.status3d[..n], f)
                            .map(|(start, end)| (start, end, mk.points3d[start], mk.points3d[end]))
                    };
                    if let Some((start, end, a, b)) = gap {
                        while f < end {
                            if methods[f] == Interpolation::Linear {
                                let p = (f - start) as f64 / (end - start) as f64;
                                let x = Point3::new(
                                    a.x + p * (b.x - a.x),
                                    a.y + p * (b.y - a.y),
                                    a.z + p * (b.z - a.z),
                                );
                                for c in 0..n_cam {
                                    if fillable(status2d(self, c, f))
                                        && let Some(q) = self.project_into(trial, c, x)
                                    {
                                        self.deferred_set_point(
                                            d,
                                            trial,
                                            marker,
                                            c,
                                            f,
                                            q,
                                            MarkerStatus::Interpolated,
                                        )?;
                                    }
                                }
                            }
                            f += 1;
                        }
                    }
                    f += 1;
                }
            } else {
                for c in 0..n_cam {
                    let mut f = 1;
                    while f < n {
                        let gap = {
                            let v = &self.project.trials[trial].markers[marker].views[c];
                            linear_gap(&v.status[..n], f)
                                .map(|(start, end)| (start, end, v.points[start], v.points[end]))
                        };
                        if let Some((start, end, a, b)) = gap {
                            while f < end {
                                if methods[f] == Interpolation::Linear
                                    && fillable(status2d(self, c, f))
                                {
                                    let p = (f - start) as f64 / (end - start) as f64;
                                    let q =
                                        Point2::new(a.x + p * (b.x - a.x), a.y + p * (b.y - a.y));
                                    self.deferred_set_point(
                                        d,
                                        trial,
                                        marker,
                                        c,
                                        f,
                                        q,
                                        MarkerStatus::Interpolated,
                                    )?;
                                }
                                f += 1;
                            }
                        }
                        f += 1;
                    }
                }
            }
        }

        // 4. Cubic.
        if requires(Interpolation::Cubic) {
            if interpolate_3d {
                let (knots, at, xs, ys, zs) = {
                    let mk = &self.project.trials[trial].markers[marker];
                    let status = &mk.status3d[..n];
                    let knots: Vec<usize> = (0..n).filter(|&f| known(status[f])).collect();
                    if knots.len() < 4 {
                        return Ok(());
                    }
                    let at = cubic_gap_frames(&methods, status, &knots, n);
                    if at.is_empty() {
                        return Ok(());
                    }
                    let x1: Vec<f64> = knots.iter().map(|&f| f as f64).collect();
                    let x2: Vec<f64> = at.iter().map(|&f| f as f64).collect();
                    let coord = |k: usize| -> Vec<f64> {
                        knots.iter().map(|&f| mk.points3d[f][k]).collect()
                    };
                    let (Some(xs), Some(ys), Some(zs)) = (
                        cubic_spline(&x1, &coord(0), &x2),
                        cubic_spline(&x1, &coord(1), &x2),
                        cubic_spline(&x1, &coord(2), &x2),
                    ) else {
                        return Ok(());
                    };
                    (knots, at, xs, ys, zs)
                };
                let _ = knots;
                for (k, &f) in at.iter().enumerate() {
                    let x = Point3::new(xs[k], ys[k], zs[k]);
                    for c in 0..n_cam {
                        if fillable(status2d(self, c, f))
                            && let Some(q) = self.project_into(trial, c, x)
                        {
                            self.deferred_set_point(
                                d,
                                trial,
                                marker,
                                c,
                                f,
                                q,
                                MarkerStatus::Interpolated,
                            )?;
                        }
                    }
                }
            } else {
                for c in 0..n_cam {
                    // Marker.cpp:1371-1372 (ledger A33), :1395-1396 and :1398-1401: XMALab 3
                    // returns from the whole function where this camera is skipped.
                    let fill = {
                        let v = &self.project.trials[trial].markers[marker].views[c];
                        let status = &v.status[..n];
                        let knots: Vec<usize> = (0..n).filter(|&f| known(status[f])).collect();
                        if knots.len() < 4 {
                            None
                        } else {
                            let at = cubic_gap_frames(&methods, status, &knots, n);
                            if at.is_empty() {
                                None
                            } else {
                                let x1: Vec<f64> = knots.iter().map(|&f| f as f64).collect();
                                let x2: Vec<f64> = at.iter().map(|&f| f as f64).collect();
                                let yx: Vec<f64> = knots.iter().map(|&f| v.points[f].x).collect();
                                let yy: Vec<f64> = knots.iter().map(|&f| v.points[f].y).collect();
                                match (cubic_spline(&x1, &yx, &x2), cubic_spline(&x1, &yy, &x2)) {
                                    (Some(xs), Some(ys)) => Some((at, xs, ys)),
                                    _ => None,
                                }
                            }
                        }
                    };
                    let Some((at, xs, ys)) = fill else {
                        if cubic_stops {
                            return Ok(());
                        }
                        continue;
                    };
                    for (k, &f) in at.iter().enumerate() {
                        if fillable(status2d(self, c, f)) {
                            let q = Point2::new(xs[k], ys[k]);
                            self.deferred_set_point(
                                d,
                                trial,
                                marker,
                                c,
                                f,
                                q,
                                MarkerStatus::Interpolated,
                            )?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Starts the follow-up record of an edit over many cells ([`Deferred`]): batched unless
    /// the mode does the follow-up after every point
    /// ([`CppCompat::point_edits_follow_up_per_point`], ledger A216) or recomputes the first
    /// body only ([`CppCompat::edit_first_body_only`], ledger A31, whose result depends on
    /// the order of the writes).
    pub(crate) fn deferred(&self, trial: usize) -> Deferred {
        let compat = self.compat();
        let batched = !compat.point_edits_follow_up_per_point && !compat.edit_first_body_only;
        Deferred {
            ctx: batched.then(|| self.frame_context(trial)),
            touched: std::collections::BTreeMap::new(),
            sized: BTreeSet::new(),
        }
    }

    /// [`Document::set_point`] inside an edit over many cells. Per point (XMALab 3 mode):
    /// that call. Batched: the point and status are written and the marker's 3D point at
    /// the frame triangulated at once (as `set_point` does, with the edit's one
    /// [`crate::FrameContext`]), and the rigid bodies are left to [`Document::finish_deferred`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn deferred_set_point(
        &mut self,
        d: &mut Deferred,
        trial: usize,
        marker: usize,
        camera: usize,
        frame: usize,
        point: Point2,
        status: MarkerStatus,
    ) -> Result<(), EditError> {
        let Some(ctx) = &d.ctx else {
            return self.set_point(trial, marker, camera, frame, point, status);
        };
        check_cell(self, trial, marker, camera, frame)?;
        let m = self.marker_mut(trial, marker);
        m.views[camera].points[frame] = point;
        m.views[camera].status[frame] = status;
        let patch = reconstruct_3d_point(ctx, &self.project.trials[trial].markers[marker], frame);
        patch.apply(self.marker_mut(trial, marker));
        d.touched.entry(frame).or_default().insert(marker);
        Ok(())
    }

    /// [`Document::reset_point`] inside an edit over many cells. Per point (XMALab 3 mode):
    /// that call. Batched: the cell is reset and the marker's 3D point at the frame
    /// triangulated at once; the rigid bodies and the mean size are left to
    /// [`Document::finish_deferred`].
    pub(crate) fn deferred_reset_point(
        &mut self,
        d: &mut Deferred,
        trial: usize,
        marker: usize,
        camera: usize,
        frame: usize,
    ) -> Result<(), EditError> {
        let Some(ctx) = &d.ctx else {
            return self.reset_point(trial, marker, camera, frame);
        };
        check_cell(self, trial, marker, camera, frame)?;
        self.marker_mut(trial, marker).views[camera].reset(frame);
        let patch = reconstruct_3d_point(ctx, &self.project.trials[trial].markers[marker], frame);
        patch.apply(self.marker_mut(trial, marker));
        d.touched.entry(frame).or_default().insert(marker);
        d.sized.insert(marker);
        Ok(())
    }

    /// [`Document::reconstruct_frame`] inside an edit over many cells. Per point (XMALab 3
    /// mode): that call. Batched: the marker's 3D point at the frame is triangulated at
    /// once, and the rigid bodies are left to [`Document::finish_deferred`]. Indices must
    /// be valid.
    pub(crate) fn deferred_reconstruct_frame(
        &mut self,
        d: &mut Deferred,
        trial: usize,
        marker: usize,
        frame: usize,
    ) {
        let Some(ctx) = &d.ctx else {
            self.reconstruct_frame(trial, marker, frame);
            return;
        };
        let patch = reconstruct_3d_point(ctx, &self.project.trials[trial].markers[marker], frame);
        patch.apply(self.marker_mut(trial, marker));
        d.touched.entry(frame).or_default().insert(marker);
    }

    /// The follow-up work an edit over many cells left ([`Deferred`]), once: the mean size
    /// and size range of each marker whose sizes were reset, then, frame by frame in
    /// increasing order, every rigid body whose pose depends on a marker written at that
    /// frame, recomputed as the trial update computes it, with the continuity step carried
    /// on from the frame and the stale mark (as [`Document::reset_rigid_body_by_marker`]
    /// does for one point). Nothing when the edit ran per point.
    ///
    /// Not a step of XMALab (ledger A216). The result is that of the per-point order: a
    /// body's raw pose at a frame depends only on the final marker data at that frame (and
    /// the bodies driving it, computed afresh there), and the continuity step applied from
    /// each recomputed frame in increasing order leaves the series a full continuity pass
    /// gives, as for the writes of a tracking run (`tracking::write_results`). Two
    /// differences: an edit that resets every counted size of a marker keeps the mean size
    /// from before the edit (per point: the last remaining size's), since
    /// [`update_mean_size`] leaves the mean when no size is counted; and where a body's
    /// final computation at a frame finds no pose, the pose stored there (never read: the
    /// frame is not valid) is the one from before the edit, where per point it is the last
    /// one an intermediate computation found.
    pub(crate) fn finish_deferred(&mut self, trial: usize, d: Deferred) {
        if d.ctx.is_none() {
            return;
        }
        let compat = self.compat();
        for &m in &d.sized {
            update_mean_size(self.marker_mut(trial, m), compat.size_range_from_sum);
        }
        for (frame, markers) in d.touched {
            let t = &self.project.trials[trial];
            let mut bodies: Vec<usize> = Vec::new();
            for &m in &markers {
                bodies.extend(bodies_affected_by(t, m));
            }
            bodies.sort_unstable();
            bodies.dedup();
            if bodies.is_empty() {
                continue;
            }
            self.recompute_bodies_at_frame(trial, frame, &bodies);
            if !compat.edits_not_refiltered {
                for &b in &bodies {
                    self.continue_rotations(trial, b, frame);
                }
            }
            self.mark_bodies_stale(trial, &bodies, compat);
        }
    }

    /// A 3D point projected into a camera with the trial's reference calibration image
    /// (`Camera::projectPoint(pt, referenceCalibrationImage)`); `None` for a camera without
    /// that image (undefined behaviour in the C++).
    pub(crate) fn project_into(&self, trial: usize, camera: usize, p: Point3) -> Option<Point2> {
        let reference = self.project.trials[trial].reference_calibration_image;
        let cam = self.project.cameras.get(camera)?;
        let pose = crate::document::reference_pose(cam, reference)?;
        Some(self.camera(camera).model.project_point(&pose, p))
    }
}

/// The follow-up work of an edit over many cells, kept for the end in the default mode
/// (ledger A216; [`Document::deferred`], [`Document::finish_deferred`]).
pub(crate) struct Deferred {
    /// The trial's triangulation inputs when batched; `None` when every point does its own
    /// follow-up.
    ctx: Option<crate::FrameContext>,
    /// Frame → markers written there.
    touched: std::collections::BTreeMap<usize, BTreeSet<usize>>,
    /// Markers whose sizes were reset.
    sized: BTreeSet<usize>,
}

/// The linear gap that starts at frame `f` (`f - 1` known, `f` not), as
/// `Marker::interpolate` finds it (core/Marker.cpp, `:1306-1316`): `(start, end)` with
/// `start = f - 1` and `end` the next known frame; `None` when `f` starts no gap or no known
/// frame ends it.
fn linear_gap(status: &[MarkerStatus], f: usize) -> Option<(usize, usize)> {
    if f == 0 || f >= status.len() || known(status[f]) || !known(status[f - 1]) {
        return None;
    }
    let mut end = f;
    while end < status.len() && !known(status[end]) {
        end += 1;
    }
    (end < status.len()).then_some((f - 1, end))
}

/// The frames a cubic gap fills, as `Marker::interpolate` collects them
/// (core/Marker.cpp, `:1376-1392` and `:1426-1442`): from the first known frame up to (not
/// including) the last, a frame whose method is `CUBIC` and whose previous frame is known
/// starts a run that continues while the method is `CUBIC` and the frame unknown; a run
/// ended by a known frame is filled. Ledger A47 fixed: frame 0 has no previous frame.
fn cubic_gap_frames(
    methods: &[Interpolation],
    status: &[MarkerStatus],
    knots: &[usize],
    n: usize,
) -> Vec<usize> {
    let mut at = Vec::new();
    let last = knots[knots.len() - 1];
    let mut f = knots[0];
    while f < last {
        if methods[f] == Interpolation::Cubic && f > 0 && known(status[f - 1]) {
            let mut end = f;
            while end < n && methods[end] == Interpolation::Cubic && !known(status[end]) {
                end += 1;
            }
            if end < n && known(status[end]) {
                while f < end {
                    at.push(f);
                    f += 1;
                }
            }
        }
        f += 1;
    }
    at
}

/// The bodies whose pose at a frame depends on a marker: those that list it, and,
/// repeatedly, those with a virtual point driven by one of them. In index order.
pub(crate) fn bodies_affected_by(trial: &xma_model::Trial, marker: usize) -> Vec<usize> {
    let start: Vec<usize> = trial
        .rigid_bodies
        .iter()
        .enumerate()
        .filter(|(_, b)| b.marker_indices.contains(&marker))
        .map(|(i, _)| i)
        .collect();
    with_dependents(trial, &start)
}

/// The given bodies and, repeatedly, every body with a virtual point driven by one of them.
/// In index order; indices out of range are dropped.
pub(crate) fn with_dependents(trial: &xma_model::Trial, bodies: &[usize]) -> Vec<usize> {
    let n = trial.rigid_bodies.len();
    let mut set: BTreeSet<usize> = bodies.iter().copied().filter(|&b| b < n).collect();
    loop {
        let before = set.len();
        for (i, b) in trial.rigid_bodies.iter().enumerate() {
            if set.contains(&i) {
                continue;
            }
            let driven = b.virtual_points.iter().any(|v| {
                matches!(v.source, VirtualPointSource::OtherBody { body, .. } if set.contains(&body))
            });
            if driven {
                set.insert(i);
            }
        }
        if set.len() == before {
            break;
        }
    }
    set.into_iter().collect()
}

/// True when a body is on a cycle of virtual points: it follows itself through other
/// bodies' virtual points.
#[must_use]
pub(crate) fn on_virtual_point_cycle(trial: &xma_model::Trial, body: usize) -> bool {
    let followers: Vec<usize> = trial
        .rigid_bodies
        .iter()
        .enumerate()
        .filter(|(_, b)| {
            b.virtual_points.iter().any(
                |v| matches!(v.source, VirtualPointSource::OtherBody { body: d, .. } if d == body),
            )
        })
        .map(|(i, _)| i)
        .collect();
    with_dependents(trial, &followers).contains(&body)
}

/// A cycle of virtual points in a trial, if there is one: bodies `c[0]`, `c[1]`, ... where
/// each has a virtual point driven by the next and the last one by `c[0]` (a body driven by
/// itself is a cycle of one). The first cycle met from the lowest body index. Driver indices
/// out of range are not followed.
///
/// Not a port: XMALab recurses without end on such a cycle (`RigidBody::transformPoint`
/// calls `computePose` on the driving body, core/RigidBody.cpp:297-300).
#[must_use]
pub fn virtual_point_cycle(trial: &xma_model::Trial) -> Option<Vec<usize>> {
    let n = trial.rigid_bodies.len();
    let drivers = |b: usize| -> Vec<usize> {
        let mut d: Vec<usize> = trial.rigid_bodies[b]
            .virtual_points
            .iter()
            .filter_map(|v| match v.source {
                VirtualPointSource::OtherBody { body, .. } if body < n => Some(body),
                _ => None,
            })
            .collect();
        d.sort_unstable();
        d.dedup();
        d
    };
    // 0 unvisited, 1 on the current path, 2 done.
    let mut state = vec![0u8; n];
    for root in 0..n {
        if state[root] != 0 {
            continue;
        }
        let mut path: Vec<usize> = vec![root];
        let mut next: Vec<(Vec<usize>, usize)> = vec![(drivers(root), 0)];
        state[root] = 1;
        while let Some((ds, k)) = next.last_mut() {
            if *k == ds.len() {
                let b = path.pop().expect("path");
                state[b] = 2;
                next.pop();
                continue;
            }
            let d = ds[*k];
            *k += 1;
            match state[d] {
                1 => {
                    let at = path.iter().position(|&b| b == d).expect("on the path");
                    return Some(path[at..].to_vec());
                }
                0 => {
                    state[d] = 1;
                    path.push(d);
                    next.push((drivers(d), 0));
                }
                _ => {}
            }
        }
    }
    None
}

/// The sentence that refuses an edit creating a cycle of virtual points.
#[must_use]
pub(crate) fn cycle_message(trial: &xma_model::Trial, cycle: &[usize]) -> String {
    let name = |b: usize| {
        let d = trial
            .rigid_bodies
            .get(b)
            .map_or("", |rb| rb.description.as_str());
        if d.is_empty() {
            format!("rigid body {}", b + 1)
        } else {
            format!("rigid body {} ({d})", b + 1)
        }
    };
    let links: Vec<String> = (0..cycle.len())
        .map(|i| {
            format!(
                "{} follows {}",
                name(cycle[i]),
                name(cycle[(i + 1) % cycle.len()])
            )
        })
        .collect();
    format!(
        "Virtual points cannot form a cycle: {}. A rigid body cannot be animated by itself through virtual points.",
        links.join(", ")
    )
}

/// `Rejected` when an edit has created a cycle of virtual points in a trial that had none
/// (`before` holds the trials as they were before the edit; a trial that is the same
/// allocation is unchanged). A trial that already held a cycle (a project opened so) is
/// left alone: its cycle is cut where the computation meets it, with a warning.
pub(crate) fn refuse_new_cycles(
    before: &[std::sync::Arc<xma_model::Trial>],
    after: &[std::sync::Arc<xma_model::Trial>],
) -> Result<(), EditError> {
    for (i, t) in after.iter().enumerate() {
        let old = before.get(i);
        if old.is_some_and(|o| Arc::ptr_eq(o, t)) {
            continue;
        }
        if old.is_some_and(|o| virtual_point_cycle(o).is_some()) {
            continue;
        }
        if let Some(cycle) = virtual_point_cycle(t) {
            return Err(EditError::Rejected(cycle_message(t, &cycle)));
        }
    }
    Ok(())
}
