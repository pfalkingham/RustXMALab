//! How many markers and rigid bodies a trial has, and which marker is in which body: the
//! Points panel's context menu (E15 for bodies, E16, E17, E19), drag and drop (E23) and the
//! "Set Number" buttons (E24, E25), as `docs/cpp-editing.md` section 4 describes them.
//!
//! Each edit makes the model calls of its C++ handler in the same order
//! ([`super::body_ops`]); what is recomputed follows the mode, as described there.

use super::body_ops::{
    self, add_marker as add_marker_op, add_point_idx, add_rigid_body, body_of_marker, check_body,
    check_markers, check_trial, keeping_bodies_current, outcome, remove_marker, remove_point_idx,
    remove_rigid_body,
};
use super::{EditError, EditOutcome, EditResult};
use crate::document::Document;

/// Appends a marker to the trial (`Trial::addMarker`). Nothing is recomputed.
///
/// Ported from core/Trial.cpp, Trial::addMarker (`:461-465`).
pub(super) fn add_marker(doc: &mut Document, trial: usize) -> EditResult {
    check_trial(doc, trial)?;
    add_marker_op(doc, trial);
    Ok(EditOutcome::none())
}

/// E19 "Delete selected points": each marker that is in a body is first removed from it
/// (`removePointIdx`), then the markers are deleted highest index first
/// (`Trial::removeMarker`), which renumbers the later markers in every body.
///
/// Ported from ui/MarkerTreeWidget.cpp, MarkerTreeWidget::action_DeletePoints_triggered
/// (`:209-237`). The body a marker is removed from first is the first body listing it (the
/// row the C++ reads its parent from); `removeMarker` then removes it from any other.
pub(super) fn remove_markers(doc: &mut Document, trial: usize, markers: &[usize]) -> EditResult {
    check_markers(doc, trial, markers)?;
    keeping_bodies_current(doc, trial, &[], |doc| {
        for &m in markers {
            if let Some(b) = body_of_marker(&doc.project.trials[trial], m) {
                remove_point_idx(doc, trial, b, m);
            }
        }
        let mut sorted = markers.to_vec();
        sorted.sort_unstable();
        for &m in sorted.iter().rev() {
            remove_marker(doc, trial, m);
        }
        Ok(EditOutcome::none())
    })
}

/// E24 "Set Number Markers": markers are appended (`addMarker`) or the last one removed
/// (`removeMarker`) until the trial has `count`.
///
/// Ported from ui/PointsDockWidget.cpp, PointsDockWidget::on_pushButtonSetNumberMarkers_clicked
/// (`:449-476`). The dialog's limit (1000) is the interface's.
pub(super) fn set_marker_count(doc: &mut Document, trial: usize, count: usize) -> EditResult {
    check_trial(doc, trial)?;
    keeping_bodies_current(doc, trial, &[], |doc| {
        while doc.project.trials[trial].markers.len() < count {
            add_marker_op(doc, trial);
        }
        while doc.project.trials[trial].markers.len() > count {
            let last = doc.project.trials[trial].markers.len() - 1;
            remove_marker(doc, trial, last);
        }
        Ok(EditOutcome::none())
    })
}

/// E25 "Set Number Rigid Bodies": bodies are appended (`addRigidBody`) or the last one
/// removed (`removeRigidBody`) until the trial has `count`. The C++ recomputes nothing;
/// with D6 fixed, virtual points that followed a removed body are deleted
/// ([`body_ops::remove_rigid_body`]) and, in the default mode, the bodies that had them are
/// recomputed.
///
/// Ported from ui/PointsDockWidget.cpp,
/// PointsDockWidget::on_pushButtonSetNumberRigidBodies_clicked (`:478-504`).
pub(super) fn set_rigid_body_count(doc: &mut Document, trial: usize, count: usize) -> EditResult {
    check_trial(doc, trial)?;
    keeping_bodies_current(doc, trial, &[], |doc| {
        let mut messages = Vec::new();
        while doc.project.trials[trial].rigid_bodies.len() < count {
            add_rigid_body(doc, trial);
        }
        while doc.project.trials[trial].rigid_bodies.len() > count {
            let last = doc.project.trials[trial].rigid_bodies.len() - 1;
            messages.extend(remove_rigid_body(doc, trial, last));
        }
        Ok(outcome(messages))
    })
}

/// E16 "Create Rigid Body from selected": one new body, to which each marker that is in no
/// body yet is added in order (`addPointIdx`, which resets the body's references and
/// recomputes and filters it). Markers already in a body are skipped, with a Console line
/// for each. `Invalid` when none is free (the C++ then creates nothing).
///
/// Ported from ui/MarkerTreeWidget.cpp, MarkerTreeWidget::action_CreateRigidBody_triggered
/// (`:510-538`).
pub(super) fn create_rigid_body(doc: &mut Document, trial: usize, markers: &[usize]) -> EditResult {
    check_markers(doc, trial, markers)?;
    let t = &doc.project.trials[trial];
    let (free, taken): (Vec<usize>, Vec<usize>) = markers
        .iter()
        .partition(|&&m| body_of_marker(t, m).is_none());
    if free.is_empty() {
        return Err(EditError::Invalid(
            "none of the markers is free: each is already in a rigid body".to_owned(),
        ));
    }
    let messages = taken
        .iter()
        .map(|&m| {
            format!(
                "Marker {} is already in rigid body {} and was not added.",
                m + 1,
                body_of_marker(t, m).map_or(0, |b| b + 1)
            )
        })
        .collect();
    keeping_bodies_current(doc, trial, &[], |doc| {
        let body = add_rigid_body(doc, trial);
        for &m in &free {
            add_point_idx(doc, trial, body, m, true);
        }
        Ok(outcome(messages))
    })
}

/// E17 "Add selected to Rigid Body" (`body` given) and E23 drag and drop: each marker in a
/// body is removed from it (`removePointIdx`); then, with a target, added to it
/// (`addPointIdx`, which resets the target's references and recomputes and filters it). With
/// `None` a marker in no body is left as it is.
///
/// A marker moved to the body it is already in is left out of the move (ledger U79, decided
/// by the author 16 November 2026), so a move of only such markers changes nothing and makes
/// no undo step. XMALab 3 ([`xma_model::CppCompat::move_to_own_body_readds`]) removes and
/// adds it again: the body's references are reset and the marker moves to the end of its
/// list.
///
/// Ported from ui/MarkerTreeWidget.cpp, MarkerTreeWidget::action_AddToRigidBody_triggered
/// (`:540-587`) and MarkerTreeWidget::dropEvent (`:589-689`).
pub(super) fn move_markers_to_body(
    doc: &mut Document,
    trial: usize,
    markers: &[usize],
    body: Option<usize>,
) -> EditResult {
    check_markers(doc, trial, markers)?;
    if let Some(b) = body {
        check_body(doc, trial, b)?;
    }
    let readds = doc.compat().move_to_own_body_readds;
    keeping_bodies_current(doc, trial, &[], |doc| {
        for &m in markers {
            if let Some(b) = body
                && !readds
                && doc.project.trials[trial].rigid_bodies[b]
                    .marker_indices
                    .contains(&m)
            {
                continue;
            }
            if let Some(old) = body_of_marker(&doc.project.trials[trial], m) {
                remove_point_idx(doc, trial, old, m);
            }
            if let Some(b) = body {
                add_point_idx(doc, trial, b, m, true);
            }
        }
        Ok(EditOutcome::none())
    })
}

/// E15 for a rigid body: sets its description. Nothing is recomputed.
///
/// Ported from ui/MarkerTreeWidget.cpp, MarkerTreeWidget::action_ChangeDescription_triggered
/// (`:182-207`).
pub(super) fn set_body_description(
    doc: &mut Document,
    trial: usize,
    body: usize,
    description: &str,
) -> EditResult {
    check_body(doc, trial, body)?;
    if doc.project.trials[trial].rigid_bodies[body].description != description {
        body_ops::body_mut(doc, trial, body).description = description.to_owned();
    }
    Ok(EditOutcome::none())
}
