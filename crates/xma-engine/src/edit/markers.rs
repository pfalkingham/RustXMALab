//! Marker properties: description, detection method, threshold, size override, penalty.
//!
//! None of these recomputes anything in either mode: they affect later detection and
//! tracking only (`docs/cpp-editing.md` sections 4.1, 5.1 and 8).

use xma_model::DetectionMethod;

use super::ops::check_marker;
use super::{EditError, EditOutcome, EditResult, MarkerProperties};
use crate::document::Document;

/// E15 for a marker.
///
/// Ported from ui/MarkerTreeWidget.cpp, MarkerTreeWidget::action_ChangeDescription_triggered
/// (`:182-207`): `Marker::setDescription`. The action is not offered in a trial linked to
/// the Default trial (`:160`, `:174`), so there it is [`EditError::Invalid`].
pub(super) fn set_description(
    doc: &mut Document,
    trial: usize,
    marker: usize,
    description: &str,
) -> EditResult {
    check_marker(doc, trial, marker)?;
    if doc.project.trials[trial].is_copy_from_default {
        return Err(EditError::Invalid(
            "a trial linked to the Default trial takes its descriptions from it".to_owned(),
        ));
    }
    if doc.project.trials[trial].markers[marker].description != description {
        doc.marker_mut(trial, marker).description = description.to_owned();
    }
    Ok(EditOutcome::none())
}

/// E20 (several markers, Points panel) and E55 (Detection Settings).
///
/// Ported from ui/MarkerTreeWidget.cpp, MarkerTreeWidget::action_ChangeDetectionMethod_triggered
/// (`:239-260`) and ui/DetectionSettings.cpp, DetectionSettings::on_comboBox_Method_currentIndexChanged
/// (`:76-83`): `Marker::setMethod` on each marker. Every index is checked first. The Detection
/// Settings window cannot change the method of a marker using methods 1, 3, 4 or 6
/// (`docs/cpp-editing.md` D12); that is the window's defect, not the model's, and the edit
/// applies to any marker.
pub(super) fn set_method(
    doc: &mut Document,
    trial: usize,
    markers: &[usize],
    method: DetectionMethod,
) -> EditResult {
    for &m in markers {
        check_marker(doc, trial, m)?;
    }
    for &m in markers {
        if doc.project.trials[trial].markers[m].method != method {
            doc.marker_mut(trial, m).method = method;
        }
    }
    Ok(EditOutcome::none())
}

/// The Marker dialog's OK (E29), and E56 (the threshold offset from Detection Settings).
///
/// Ported from ui/MarkerDialog.cpp, MarkerDialog::isComplete (`:79-87`), with
/// core/Marker.cpp, Marker::setMaxPenalty (`:1081-1089`: 125 becomes 50, above 100 becomes
/// 100). The values must lie in the dialog's ranges (threshold offset 0 to 100, size
/// override -1 to 50, penalty 0 to 100), otherwise [`EditError::Invalid`]. As
/// `isComplete`, the method is not written in a trial linked to the Default trial, and the
/// other three are not written in the Default trial (the dialog disables those controls).
pub(super) fn set_properties(
    doc: &mut Document,
    trial: usize,
    marker: usize,
    properties: &MarkerProperties,
) -> EditResult {
    check_marker(doc, trial, marker)?;
    let p = properties;
    if let Some(v) = p.threshold_offset
        && !(0..=100).contains(&v)
    {
        return Err(EditError::Invalid(format!(
            "threshold offset {v} is outside 0 to 100"
        )));
    }
    if let Some(v) = p.size_override
        && !(-1..=50).contains(&v)
    {
        return Err(EditError::Invalid(format!(
            "size override {v} is outside -1 to 50"
        )));
    }
    if let Some(v) = p.max_penalty
        && !(0..=100).contains(&v)
    {
        return Err(EditError::Invalid(format!(
            "penalty {v} is outside 0 to 100"
        )));
    }
    // MarkerDialog.cpp:81-84: the Default trial keeps penalty, size override and threshold
    // (its spin boxes are disabled), a trial linked to it keeps the method.
    let t = &doc.project.trials[trial];
    let (is_default, linked) = (t.is_default, t.is_copy_from_default);
    let p = &MarkerProperties {
        method: p.method.filter(|_| !linked),
        threshold_offset: p.threshold_offset.filter(|_| !is_default),
        size_override: p.size_override.filter(|_| !is_default),
        max_penalty: p.max_penalty.filter(|_| !is_default),
    };
    let current = &doc.project.trials[trial].markers[marker];
    let penalty = p.max_penalty.map(super::body_ops::capped_penalty);
    let changes = p.method.is_some_and(|v| v != current.method)
        || p.threshold_offset
            .is_some_and(|v| v != current.threshold_offset)
        || p.size_override.is_some_and(|v| v != current.size_override)
        || penalty.is_some_and(|v| v != current.max_penalty);
    if changes {
        let m = doc.marker_mut(trial, marker);
        if let Some(v) = penalty {
            m.max_penalty = v;
        }
        if let Some(v) = p.method {
            m.method = v;
        }
        if let Some(v) = p.size_override {
            m.size_override = v;
        }
        if let Some(v) = p.threshold_offset {
            m.threshold_offset = v;
        }
    }
    Ok(EditOutcome::none())
}
