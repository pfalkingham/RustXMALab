//! 2D point data: set, delete, mark untrackable, exchange, interpolate.
//!
//! Each function ports the action its [`super::Edit`] variant names
//! (`docs/cpp-editing.md`); the model calls they end in are in [`super::ops`]. Every index
//! is checked first, so an edit that fails changes nothing (the step takes back anything
//! written before an error, [`Document::apply`]).

#![allow(clippy::too_many_arguments)]

use xma_model::{Interpolation, MarkerStatus, Point2};

use super::ops::{cameras_of, check_camera, check_frames, check_marker, check_trial};
use super::{
    BodyErrorKind, CameraSel, EditError, EditOutcome, EditResult, ErrorTarget, FrameRange,
};
use crate::document::Document;
use crate::markers::update_mean_size;

/// E3 (a `MANUAL` placement in the Detailed View) and every other single point write.
///
/// Ported from ui/WizardDigitizationFrame.cpp, WizardDigitizationFrame::moveDigitizationPoint
/// (`:159-190`, the `setPoint` it makes), through `Marker::setPoint`
/// ([`Document::set_point`]). The marker size is not changed.
pub(super) fn set_point(
    doc: &mut Document,
    trial: usize,
    marker: usize,
    camera: usize,
    frame: usize,
    x: f64,
    y: f64,
    status: MarkerStatus,
) -> EditResult {
    doc.set_point(trial, marker, camera, frame, Point2::new(x, y), status)?;
    Ok(EditOutcome::none())
}

/// What a finished detection writes: the size, then the point.
///
/// Ported from processing/MarkerDetection.cpp, MarkerDetection::detectMarker_threadFinished
/// (`:420-441`): `setSize(cam, frame, size)` then `setPoint(cam, frame, x, y, status)`.
pub(super) fn set_detected_point(
    doc: &mut Document,
    trial: usize,
    marker: usize,
    camera: usize,
    frame: usize,
    x: f64,
    y: f64,
    size: f64,
    status: MarkerStatus,
) -> EditResult {
    doc.set_size(trial, marker, camera, frame, size)?;
    doc.set_point(trial, marker, camera, frame, Point2::new(x, y), status)?;
    Ok(EditOutcome::none())
}

/// E18 (Points panel "Delete Data from selected points"), E44 (Delete in the plot) and
/// E45 (Ctrl+U, untrackable).
///
/// Ported from ui/MarkerTreeWidget.cpp, MarkerTreeWidget::action_ResetPoints_triggered
/// (`:484-508`) and ui/PlotWindow.cpp, PlotWindow::deleteData (`:649-699`) and
/// PlotWindow::setUntrackable (`:3067-3117`): `Marker::resetMultipleFrames` on each marker
/// in the order given ([`Document::reset_frames`]). The C++ indexes the plot's marker and
/// body lists unchecked (`docs/cpp-editing.md` D19); here every marker index is checked
/// before anything is written. The plot's types 4 and 5 then press "Update" (E48), which
/// the interface sends as a separate edit.
pub(super) fn reset_frames(
    doc: &mut Document,
    trial: usize,
    markers: &[usize],
    cameras: CameraSel,
    frames: FrameRange,
    untrackable: bool,
) -> EditResult {
    check_trial(doc, trial)?;
    for &m in markers {
        check_frames(doc, trial, m, frames)?;
        check_camera(doc, trial, m, cameras)?;
    }
    let mut d = doc.deferred(trial);
    for &m in markers {
        doc.reset_frames(&mut d, trial, m, cameras, frames, untrackable)?;
    }
    doc.finish_deferred(trial, d);
    Ok(EditOutcome::none())
}

/// E47: deletes every point of a frame whose error is above the threshold, over every
/// frame of the trial (not limited to the start and end frames).
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::deleteAllAboveError (`:1369-1424`) and
/// PlotWindow::isFrameAboveError (`:548-627`):
///
/// - [`ErrorTarget::Marker`]: the frame's error is the mean `error2D` over the cameras whose
///   2D status and the 3D status are above `UNDEFINED` (0 when none), or that of one camera
///   under the same test; `Marker::reset` on every camera, or on the one camera.
/// - [`ErrorTarget::Body`]: the body's mean error at the frame, by `kind`: 0 unfiltered,
///   1 filtered, 2 the larger of the two, on the 3D or the 2D error as `error` says (the
///   plot's `comboBoxRigidBodyError`). Then `Marker::reset` on every camera of every marker
///   of the body, markers in body order.
///
/// The frames are tested as the C++ loop meets them; a reset changes the errors of its own
/// frame only, so testing every frame first gives the same frames. In the default mode the
/// rigid bodies and mean sizes are recomputed once at the end (ledger A216,
/// [`super::ops::Deferred`]); a marker whose every counted size is reset keeps its mean size
/// from before the edit (XMALab 3: the last remaining size).
pub(super) fn reset_above_error(
    doc: &mut Document,
    trial: usize,
    target: &ErrorTarget,
    threshold: f64,
) -> EditResult {
    check_trial(doc, trial)?;
    let t = &doc.project.trials[trial];
    let n = t.n_frames;
    let n_cam = doc.project.cameras.len();
    // (frame, markers, cameras) to reset, in order.
    let mut cells: Vec<(usize, Vec<usize>, CameraSel)> = Vec::new();
    match *target {
        ErrorTarget::Marker { marker, cameras } => {
            check_marker(doc, trial, marker)?;
            check_frames(
                doc,
                trial,
                marker,
                FrameRange {
                    first: 0,
                    last: n.saturating_sub(1),
                },
            )?;
            check_camera(doc, trial, marker, cameras)?;
            let m = &t.markers[marker];
            for f in 0..n {
                let seen = |c: usize| {
                    m.views[c].status[f] > MarkerStatus::Undefined
                        && m.status3d[f] > MarkerStatus::Undefined
                };
                let error = match cameras {
                    CameraSel::All => {
                        let used: Vec<usize> = (0..m.n_cameras()).filter(|&c| seen(c)).collect();
                        let mut e = 0.0;
                        for &c in &used {
                            e += m.views[c].error[f];
                        }
                        if used.is_empty() {
                            e
                        } else {
                            e / used.len() as f64
                        }
                    }
                    CameraSel::One(c) => {
                        if seen(c) {
                            m.views[c].error[f]
                        } else {
                            0.0
                        }
                    }
                };
                if error > threshold {
                    cells.push((f, vec![marker], cameras));
                }
            }
        }
        ErrorTarget::Body { body, kind, error } => {
            let rb = t.rigid_bodies.get(body).ok_or_else(|| {
                EditError::Invalid(format!("trial {trial} has no rigid body {body}"))
            })?;
            if kind > 2 {
                return Err(EditError::Invalid(format!(
                    "no rigid-body error kind {kind}"
                )));
            }
            for &m in &rb.marker_indices {
                check_frames(
                    doc,
                    trial,
                    m,
                    FrameRange {
                        first: 0,
                        last: n.saturating_sub(1),
                    },
                )?;
            }
            let get = |track: &xma_model::PoseTrack, f: usize| {
                track.errors.get(f).map_or(0.0, |e| match error {
                    BodyErrorKind::Error2d => e.mean_2d,
                    BodyErrorKind::Error3d => e.mean_3d,
                })
            };
            for f in 0..n {
                let raw = get(&rb.raw, f);
                let filtered = get(&rb.filtered, f);
                let value = match kind {
                    0 => raw,
                    1 => filtered,
                    _ => filtered.max(raw),
                };
                if value > threshold {
                    cells.push((f, rb.marker_indices.clone(), CameraSel::All));
                }
            }
        }
    }
    let mut d = doc.deferred(trial);
    for (f, markers, cameras) in cells {
        for m in markers {
            for c in cameras_of(cameras, n_cam) {
                doc.deferred_reset_point(&mut d, trial, m, c, f)?;
            }
        }
    }
    doc.finish_deferred(trial, d);
    Ok(EditOutcome::none())
}

/// E22: exchanges the tracked data of two markers over cameras and frames.
///
/// Ported from ui/MarkerTreeWidget.cpp, MarkerTreeWidget::action_ChangePoint_triggered
/// (`:360-451`): for each camera, then each frame, `setPoint` on the first marker with the
/// second's point and status, then on the second with the first's; then every body that
/// lists either marker is recomputed (`recomputeTransformations`) and filtered.
///
/// Ledger A67 ([`xma_model::CppCompat::swap_points_only`]): XMALab 3 swaps points and
/// statuses only; fixed, the detected sizes are swapped with them (and the mean sizes
/// updated). The 2D errors are derived and the interpolation methods describe the marker,
/// not the data, so neither is swapped in either mode. In the default mode the bodies are
/// recomputed once per frame after the point writes (ledger A216) and filtered when the step
/// is committed,
/// which gives what a full update gives; in the XMALab 3 mode the C++ recomputation and
/// filtering run (ledger A36: one body at a time).
pub(super) fn swap_marker_data(
    doc: &mut Document,
    trial: usize,
    marker: usize,
    other: usize,
    cameras: CameraSel,
    frames: FrameRange,
) -> EditResult {
    check_frames(doc, trial, marker, frames)?;
    check_frames(doc, trial, other, frames)?;
    check_camera(doc, trial, marker, cameras)?;
    check_camera(doc, trial, other, cameras)?;
    if marker == other {
        return Err(EditError::Invalid(
            "a marker's data cannot be exchanged with itself".to_owned(),
        ));
    }
    let compat = doc.compat();
    let bodies: Vec<usize> = doc.project.trials[trial]
        .rigid_bodies
        .iter()
        .enumerate()
        .filter(|(_, b)| b.marker_indices.contains(&marker) || b.marker_indices.contains(&other))
        .map(|(i, _)| i)
        .collect();
    let n_cam = doc.project.trials[trial].markers[marker].n_cameras();
    let mut d = doc.deferred(trial);
    if frames.first <= frames.last {
        for c in cameras_of(cameras, n_cam) {
            for f in frames.first..=frames.last {
                let read = |doc: &Document, m: usize| {
                    let v = &doc.project.trials[trial].markers[m].views[c];
                    (v.points[f], v.status[f], v.size[f])
                };
                let (p1, s1, z1) = read(doc, marker);
                let (p2, s2, z2) = read(doc, other);
                if !compat.swap_points_only {
                    doc.marker_mut(trial, marker).views[c].size[f] = z2;
                    doc.marker_mut(trial, other).views[c].size[f] = z1;
                }
                doc.deferred_set_point(&mut d, trial, marker, c, f, p2, s2)?;
                doc.deferred_set_point(&mut d, trial, other, c, f, p1, s1)?;
            }
        }
    }
    if !compat.swap_points_only {
        for m in [marker, other] {
            update_mean_size(doc.marker_mut(trial, m), compat.size_range_from_sum);
        }
    }
    doc.finish_deferred(trial, d);
    if compat.edits_not_refiltered {
        for b in bodies {
            doc.recompute_transformations(trial, b);
            doc.filter_transformations(trial, b);
        }
    }
    Ok(EditOutcome::none())
}

/// E13 (the active marker) and E14 (every marker of the trial): fills gaps.
///
/// Ported from ui/WizardDigitizationFrame.cpp,
/// WizardDigitizationFrame::on_toolButton_InterpolateActive_clicked (`:1284-1292`) and
/// on_toolButton_InterpolateAll_clicked (`:1294-1301`): `Marker::interpolate` on each
/// marker in the order given ([`Document::interpolate_marker`]).
pub(super) fn interpolate(doc: &mut Document, trial: usize, markers: &[usize]) -> EditResult {
    check_trial(doc, trial)?;
    for &m in markers {
        check_marker(doc, trial, m)?;
    }
    let mut d = doc.deferred(trial);
    for &m in markers {
        doc.interpolate_marker(&mut d, trial, m)?;
    }
    doc.finish_deferred(trial, d);
    Ok(EditOutcome::none())
}

/// E46: sets the interpolation method of a frame range; gaps are filled by
/// [`super::Edit::Interpolate`].
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::setInterpolation (`:3007-3065`):
/// `Marker::setInterpolation(f, method)` for each frame of the range (core/Marker.cpp,
/// `:904-908`), then `updateHasInterpolation` (derived here,
/// [`xma_model::Marker::has_interpolation`]). Nothing is recomputed.
pub(super) fn set_interpolation(
    doc: &mut Document,
    trial: usize,
    marker: usize,
    frames: FrameRange,
    method: Interpolation,
) -> EditResult {
    check_trial(doc, trial)?;
    check_marker(doc, trial, marker)?;
    let len = doc.project.trials[trial].markers[marker]
        .interpolation
        .len();
    if frames.first <= frames.last && frames.last >= len {
        return Err(EditError::Invalid(format!(
            "frame {} is outside marker {marker}'s {len} frames",
            frames.last
        )));
    }
    if frames.first > frames.last {
        return Ok(EditOutcome::none());
    }
    let unchanged = doc.project.trials[trial].markers[marker].interpolation
        [frames.first..=frames.last]
        .iter()
        .all(|&m| m == method);
    if !unchanged {
        let m = doc.marker_mut(trial, marker);
        for f in frames.first..=frames.last {
            m.interpolation[f] = method;
        }
    }
    Ok(EditOutcome::none())
}
