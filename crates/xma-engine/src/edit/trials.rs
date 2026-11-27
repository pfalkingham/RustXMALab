//! Trial properties, start and end frame, deleting trials, the link to the Default trial
//! (`docs/cpp-editing.md` sections 4.3 and 9).

use std::sync::Arc;

use xma_model::{Marker, RigidBody, Trial};

use super::{EditError, EditOutcome, EditResult, TrialProperties};
use crate::document::Document;

/// `Invalid` ("no trial N") unless the trial exists, the Default trial included.
pub(super) fn check_trial(doc: &Document, trial: usize) -> Result<(), EditError> {
    if trial >= doc.project.trials.len() {
        return Err(EditError::Invalid(format!("no trial {trial}")));
    }
    Ok(())
}

fn trial_mut(doc: &mut Document, trial: usize) -> &mut Trial {
    Arc::make_mut(&mut doc.project.trials[trial])
}

/// A frequency the Trial dialog's spin boxes can hold: 0 to 10000.
fn check_frequency(what: &str, v: f64) -> Result<(), EditError> {
    if v.is_finite() && (0.0..=10000.0).contains(&v) {
        Ok(())
    } else {
        Err(EditError::Invalid(format!(
            "{what} {v} is outside 0 to 10000"
        )))
    }
}

/// E57 to E61: the Trial dialog's OK, or "Force Update Trial".
///
/// Ported from ui/TrialDialog.cpp, TrialDialog::isComplete (`:111-138`) and
/// on_pushButton_Update_clicked (`:178-183`), and what ui/WorkspaceNavigationFrame.cpp does
/// with the result (`:370-381`):
///
/// - a recording speed or cutoff that differs from the stored one is written
///   (`TRIALDIALOGUPDATEFILTER`); `interpolate_3d` is written; a reference calibration
///   image that differs is written (`TRIALDIALOGUPDATE`). The image must be calibrated in
///   every camera (the only ones the dialog lists), otherwise [`EditError::Invalid`];
/// - `TRIALDIALOGUPDATE` (a new reference image, or `force_update`): the trial is flagged
///   and fully updated ([`Document::update_trial`]);
/// - `TRIALDIALOGUPDATEFILTER` only: XMALab 3 recomputes and filters every body, one at a
///   time (`Trial::recomputeAndFilterRigidBodyTransformations`, ledger A36); in the default
///   mode the raw poses do not depend on these two values, so every body is refiltered when
///   the step is committed (ledger A60).
///
/// The dialog's own defects are the interface's: it stores a new reference image when it
/// opens (D13; the interface sends that as this edit only when the user confirms), and its
/// spin boxes round to 2 decimals (D14; the interface leaves a value `None` unless the user
/// changed it). The dialog is not offered for the Default trial ([`EditError::Invalid`]).
pub(super) fn set_properties(
    doc: &mut Document,
    trial: usize,
    properties: &TrialProperties,
) -> EditResult {
    check_trial(doc, trial)?;
    let t = &doc.project.trials[trial];
    if t.is_default {
        return Err(EditError::Invalid(
            "the Default trial has no trial settings".to_owned(),
        ));
    }
    let p = properties;
    if let Some(v) = p.recording_speed {
        check_frequency("recording speed", v)?;
    }
    if let Some(v) = p.cutoff_frequency {
        check_frequency("cutoff frequency", v)?;
    }
    if let Some(r) = p.reference_calibration {
        let cams = &doc.project.cameras;
        let listed = !cams.is_empty()
            && r < cams[0].calibration_images.len()
            && cams.iter().all(|c| {
                c.calibration_images
                    .get(r)
                    .is_some_and(|im| im.is_calibrated())
            });
        if !listed {
            return Err(EditError::Invalid(format!(
                "calibration image {r} is not calibrated in every camera"
            )));
        }
    }
    let speed = p.recording_speed.filter(|&v| v != t.recording_speed);
    let cutoff = p.cutoff_frequency.filter(|&v| v != t.cutoff_frequency);
    let reference = p
        .reference_calibration
        .filter(|&r| r != t.reference_calibration_image);
    let interpolate = p.interpolate_3d.filter(|&v| v != t.interpolate_3d);
    let update_filter = speed.is_some() || cutoff.is_some();
    let update = reference.is_some() || p.force_update;
    if speed.is_some() || cutoff.is_some() || reference.is_some() || interpolate.is_some() {
        let t = trial_mut(doc, trial);
        if let Some(v) = speed {
            t.recording_speed = v;
        }
        if let Some(v) = cutoff {
            t.cutoff_frequency = v;
        }
        if let Some(v) = interpolate {
            t.interpolate_3d = v;
        }
        if let Some(r) = reference {
            t.reference_calibration_image = r;
        }
    }
    if update {
        doc.set_trial_requires_recomputation(trial, true);
        doc.update_trial(trial);
    } else if update_filter {
        if doc.compat().edits_not_refiltered {
            doc.recompute_and_filter_rigid_body_transformations(trial);
        } else {
            for b in 0..doc.project.trials[trial].rigid_bodies.len() {
                doc.mark_body_stale(trial, b);
            }
        }
    }
    Ok(EditOutcome::none())
}

/// E63: the start frame, 1-based, from 1 to the end frame.
///
/// Ported from ui/SequenceNavigationFrame.cpp, SequenceNavigationFrame::on_toolButtonFrameStart_clicked
/// (`:448-459`) and core/Trial.cpp, Trial::setStartFrame. Nothing is recomputed (the update
/// and filtering use every frame). Moving the active frame into the range is the
/// interface's (D15 is the interface's comparison).
pub(super) fn set_start_frame(doc: &mut Document, trial: usize, frame: usize) -> EditResult {
    check_trial(doc, trial)?;
    let end = doc.project.trials[trial].end_frame;
    if frame < 1 || frame > end {
        return Err(EditError::Invalid(format!(
            "start frame {frame} is outside 1 to {end}"
        )));
    }
    if doc.project.trials[trial].start_frame != frame {
        trial_mut(doc, trial).start_frame = frame;
    }
    Ok(EditOutcome::none())
}

/// E64: the end frame, 1-based, from the start frame to the number of frames.
///
/// Ported from ui/SequenceNavigationFrame.cpp, SequenceNavigationFrame::on_toolButtonFrameEnd_clicked
/// (`:461-473`) and core/Trial.cpp, Trial::setEndFrame. As [`set_start_frame`].
pub(super) fn set_end_frame(doc: &mut Document, trial: usize, frame: usize) -> EditResult {
    check_trial(doc, trial)?;
    let t = &doc.project.trials[trial];
    if frame < t.start_frame || frame > t.n_frames {
        return Err(EditError::Invalid(format!(
            "end frame {frame} is outside {} to {}",
            t.start_frame, t.n_frames
        )));
    }
    if t.end_frame != frame {
        trial_mut(doc, trial).end_frame = frame;
    }
    Ok(EditOutcome::none())
}

/// E62 (the Trial dialog's "Delete Trial") and E65 (File > Delete multiple trials).
///
/// Ported from core/Project.cpp, Project::deleteTrial (`:269-274`), with what
/// ui/WorkspaceNavigationFrame.cpp (`:382-405`) and ui/MainWindow.cpp,
/// MainWindow::on_actionDelete_multiple_trials_triggered (`:2202-2243`) do to the active
/// trial. The trials are removed (each index once, in any order given). D7 (the C++ reads
/// the name of a deleted trial) does not arise. The active trial stays the same trial when
/// it is kept; when it is deleted, the trial now at its index becomes active, or the last
/// one. XMALab 3's "Delete multiple trials" only clamps the index, so deleting a trial before
/// the active one makes the next trial active (ui/MainWindow.cpp, `:2224-2225`); that is not
/// reproduced. The trial update of the new active trial is the interface's
/// ([`Document::select_trial`]).
pub(super) fn delete_trials(doc: &mut Document, trials: &[usize]) -> EditResult {
    for &t in trials {
        check_trial(doc, t)?;
    }
    let mut list = trials.to_vec();
    list.sort_unstable();
    list.dedup();
    if list.is_empty() {
        return Ok(EditOutcome::none());
    }
    let n_before = doc.project.trials.len();
    let active = doc.project.active_trial;
    let new_index = |i: usize| -> Option<usize> {
        if list.binary_search(&i).is_ok() {
            None
        } else {
            Some(i - list.iter().filter(|&&d| d < i).count())
        }
    };
    let remaining = n_before - list.len();
    let new_active = match new_index(active) {
        Some(i) => i,
        None => {
            let shifted = active - list.iter().filter(|&&d| d < active).count();
            shifted.min(remaining.saturating_sub(1))
        }
    };
    for &i in list.iter().rev() {
        doc.project.trials.remove(i);
    }
    doc.project.active_trial = new_active;
    if doc.extras.trial_sources.len() == n_before {
        let sources = &mut doc.extras_mut().trial_sources;
        for &i in list.iter().rev() {
            sources.remove(i);
        }
    }
    doc.remap_trials(&new_index);
    Ok(EditOutcome::none())
}

/// E28: removes the link to the Default trial.
///
/// Ported from ui/PointsDockWidget.cpp, PointsDockWidget::on_checkBoxIsFromMaster_clicked
/// (`:307-317`): `Trial::setIsCopyFromDefault(false)`. A trial that is not linked:
/// [`EditError::Invalid`].
pub(super) fn unlink_from_default(doc: &mut Document, trial: usize) -> EditResult {
    check_trial(doc, trial)?;
    if !doc.project.trials[trial].is_copy_from_default {
        return Err(EditError::Invalid(format!(
            "trial {trial} is not linked to the Default trial"
        )));
    }
    trial_mut(doc, trial).is_copy_from_default = false;
    Ok(EditOutcome::none())
}

/// E27 "Apply to trials": copies the Default trial's markers and bodies to every trial
/// linked to it.
///
/// Ported from ui/PointsDockWidget.cpp, PointsDockWidget::on_pushButtonApply_clicked
/// (`:319-376`). For every linked trial, in order:
///
/// 1. marker `i` of the Default trial: added if missing (`Trial::addMarker`, threshold from
///    `DefaultMarkerThreshold`); description, marker reference point (when set) and
///    detection method copied (penalty, size override and threshold are not);
/// 2. body `i`: added if missing (`Trial::addRigidBody`), then `RigidBody::copyData`
///    ([`super::body_ops::copy_data`]);
/// 3. extra bodies, then extra markers, removed from the end;
/// 4. `Trial::setRequiresRecomputation(true)`.
///
/// XMALab 3 starts no update there (the trial is updated when it is next selected); in the
/// default mode each linked trial is updated at once ([`Document::update_trial`]), so its
/// derived data are those of a full update. No Default trial: [`EditError::Invalid`].
pub(super) fn apply_default_to_trials(doc: &mut Document) -> EditResult {
    let Some(d) = doc.project.trials.iter().position(|t| t.is_default) else {
        return Err(EditError::Invalid(
            "the project has no Default trial".to_owned(),
        ));
    };
    let default = Arc::clone(&doc.project.trials[d]);
    let threshold = doc.project.algorithm_settings.default_marker_threshold;
    let n_cameras = doc.project.cameras.len();
    let linked: Vec<usize> = (0..doc.project.trials.len())
        .filter(|&i| i != d && doc.project.trials[i].is_copy_from_default)
        .collect();
    for &ti in &linked {
        // 1. Markers.
        {
            let t = trial_mut(doc, ti);
            let n_frames = t.n_frames;
            for (i, src) in default.markers.iter().enumerate() {
                if i >= t.markers.len() {
                    t.markers
                        .push(Arc::new(Marker::new(n_cameras, n_frames, threshold)));
                }
                let m = &t.markers[i];
                let reference = src.reference_3d.or(m.reference_3d);
                if m.description != src.description
                    || m.reference_3d != reference
                    || m.method != src.method
                {
                    let m = Arc::make_mut(&mut t.markers[i]);
                    m.description = src.description.clone();
                    m.reference_3d = reference;
                    m.method = src.method;
                }
            }
        }
        // 2. Bodies.
        for (i, src) in default.rigid_bodies.iter().enumerate() {
            if i >= doc.project.trials[ti].rigid_bodies.len() {
                let t = trial_mut(doc, ti);
                let n = t.n_frames;
                t.rigid_bodies.push(Arc::new(RigidBody::new(n)));
            }
            super::body_ops::copy_data(doc, ti, i, src);
        }
        // 3. Extra bodies, then extra markers, from the end.
        {
            let n_markers = default.markers.len();
            let t = trial_mut(doc, ti);
            t.rigid_bodies.truncate(default.rigid_bodies.len());
            if t.markers.len() > n_markers {
                t.markers.truncate(n_markers);
                // `Trial::removeMarker` takes removed markers out of the bodies
                // (`removePointIdx`, by position here, D5 fixed). A copied body lists only
                // the Default trial's markers, so this matters only for a Default trial
                // whose bodies list markers it does not have.
                for b in &mut t.rigid_bodies {
                    if b.marker_indices.iter().any(|&m| m >= n_markers) {
                        let b = Arc::make_mut(b);
                        let keep: Vec<bool> =
                            b.marker_indices.iter().map(|&m| m < n_markers).collect();
                        retain_by(&mut b.marker_indices, &keep);
                        retain_by(&mut b.reference_points, &keep);
                        retain_by(&mut b.reference_points_original, &keep);
                        retain_by(&mut b.reference_names, &keep);
                    }
                }
            }
        }
        // 4.
        doc.set_trial_requires_recomputation(ti, true);
        if !doc.compat().edits_not_refiltered {
            doc.update_trial(ti);
        }
    }
    Ok(EditOutcome::none())
}

/// Keeps the entries of `v` whose flag is set (entries past the flags are kept).
fn retain_by<T>(v: &mut Vec<T>, keep: &[bool]) {
    let mut i = 0;
    v.retain(|_| {
        let k = keep.get(i).copied().unwrap_or(true);
        i += 1;
        k
    });
}
