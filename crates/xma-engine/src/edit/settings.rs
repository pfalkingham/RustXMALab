//! Algorithm settings, with the recomputation each change needs (E73 to E79,
//! `docs/cpp-editing.md` section 11). The project stores them (ledger A10), so a change is
//! an edit and can be undone.

use xma_model::SettingsPatch;

use super::{EditError, EditOutcome, EditResult};
use crate::document::Document;

/// E73 to E79: changes settings and recomputes what depends on them.
///
/// Ported from ui/SettingsDialog.cpp (`:127-392`), where each control writes its setting as
/// it changes. Only `TriangulationMethod` triggers anything there
/// (SettingsDialog::on_comboBox_TriangulationMethod_currentIndexChanged, `:313-325`): when
/// the method changes, every trial is flagged for recomputation and the active one is
/// updated. That is what the XMALab 3 mode ([`xma_model::CppCompat::edits_not_refiltered`])
/// does; the other settings take effect at the next computation that reads them (ledger
/// A60, `docs/cpp-editing.md` D10).
///
/// In the default mode a changed setting that changes results recomputes what depends on
/// it, in every trial except the Default one:
///
/// - `TriangulationMethod`: every trial flagged and fully updated ([`Document::update_trial`]);
/// - `OptimizeRigidBody`, `DisableRBComputeAdvanced`: every body's poses recomputed as the
///   update computes them, then made continuous and filtered
///   (`Document::recompute_poses_as_update`); the markers' stored 3D points are not
///   recomputed (ledger F9). A trial flagged for recomputation is left for its update;
/// - `Filter3DPoints`: every body refiltered when the step is committed.
///
/// The other settings (E77 to E79 and the rest) affect later tracking, detection or loading
/// only. A patch the settings refuse (an unknown name, a value of the wrong kind or out of
/// range) is [`EditError::Invalid`] and changes nothing.
pub(super) fn set_settings(doc: &mut Document, patch: &SettingsPatch) -> EditResult {
    let before = doc.project.algorithm_settings.clone();
    let mut after = before.clone();
    patch.apply(&mut after).map_err(EditError::Invalid)?;
    if after == before {
        return Ok(EditOutcome::none());
    }
    let triangulation = after.triangulation_method != before.triangulation_method;
    let poses = after.optimize_rigid_body != before.optimize_rigid_body
        || after.disable_rb_compute_advanced != before.disable_rb_compute_advanced;
    let filter = after.filter_3d_points != before.filter_3d_points;
    doc.project.algorithm_settings = after;

    let n_trials = doc.project.trials.len();
    if doc.compat().edits_not_refiltered {
        if triangulation {
            let active = doc.project.active_trial;
            for t in 0..n_trials {
                doc.set_trial_requires_recomputation(t, true);
                if t == active {
                    doc.update_trial(t);
                }
            }
        }
        return Ok(EditOutcome::none());
    }

    let real: Vec<usize> = (0..n_trials)
        .filter(|&t| !doc.project.trials[t].is_default)
        .collect();
    if triangulation {
        for t in 0..n_trials {
            doc.set_trial_requires_recomputation(t, true);
        }
        for &t in &real {
            doc.update_trial(t);
        }
    } else if poses {
        for &t in &real {
            if !doc.project.trials[t].requires_recomputation {
                doc.recompute_poses_as_update(t);
            }
        }
    } else if filter {
        for &t in &real {
            if doc.project.trials[t].requires_recomputation {
                continue;
            }
            for b in 0..doc.project.trials[t].rigid_bodies.len() {
                doc.mark_body_stale(t, b);
            }
        }
    }
    Ok(EditOutcome::none())
}
