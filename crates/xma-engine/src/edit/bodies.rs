//! One rigid body: correspondences, reference points, optimised references, virtual points,
//! drawing and filter settings (the Rigid body dialog, E30 to E43, `docs/cpp-editing.md`
//! section 5.2), and the plot's "Update" (E48).
//!
//! Each edit makes the model calls of its C++ handler ([`super::body_ops`]); what is
//! recomputed follows the mode, as described there. In the XMALab 3 recomputation:
//!
//! | Edit | Recomputed |
//! |---|---|
//! | E30 correspondences, E36 to E41 | nothing (D10) |
//! | E31 from file, E32 from frame, E33 reset | the body's poses, then filtered |
//! | E34 optimized, E35 update, E42 add, E43 clear virtual points | the body's poses, not filtered |
//! | E48 Update | every body in index order, each recomputed then filtered (A36) |

use std::path::Path;

use xma_io::project_read::mesh_opens;
use xma_model::Rgb;

use super::body_ops::{
    DummySource, add_dummy_point, body_mut, check_body, check_trial,
    compute_coordinate_system_average, cpp_recomputation, cpp_recompute, keeping_bodies_current,
    outcome, reset_references as reset_references_op, set_optimized as set_optimized_op,
    set_reference_from_file, set_reference_from_frame,
};
use super::ops::{cycle_message, virtual_point_cycle};
use super::{EditError, EditOutcome, EditResult, VirtualPointInput};
use crate::document::Document;

/// The dialog text for duplicate correspondences.
pub const CORRESPONDENCES_DUPLICATE: &str =
    "Each Marker can only be used once. Please correct your selection.";

/// E30: reference point `j` is matched with marker `markers[j]`, one of the body's own
/// markers (the combo boxes list only those). `Rejected` with the dialog's text when a
/// marker is chosen twice; `Invalid` when the count differs or a marker is not in the body.
/// The C++ recomputes nothing (D10); the default mode recomputes the body.
///
/// Ported from ui/RigidBodyDialog.cpp, RigidBodyDialog::isComplete and
/// RigidBodyDialog::setRigidBodyIdxByDialog (`:213-259`), and core/RigidBody.cpp,
/// RigidBody::setPointIdx.
pub(super) fn set_correspondences(
    doc: &mut Document,
    trial: usize,
    body: usize,
    markers: &[usize],
) -> EditResult {
    check_body(doc, trial, body)?;
    let current = &doc.project.trials[trial].rigid_bodies[body].marker_indices;
    if markers.len() != current.len() {
        return Err(EditError::Invalid(format!(
            "rigid body {body} has {} markers, {} given",
            current.len(),
            markers.len()
        )));
    }
    for (k, m) in markers.iter().enumerate() {
        if markers[..k].contains(m) {
            return Err(EditError::Rejected(CORRESPONDENCES_DUPLICATE.to_owned()));
        }
        if !current.contains(m) {
            return Err(EditError::Invalid(format!(
                "marker {m} is not in rigid body {body}"
            )));
        }
    }
    if current == markers {
        return Ok(EditOutcome::none());
    }
    keeping_bodies_current(doc, trial, &[], |doc| {
        body_mut(doc, trial, body).marker_indices = markers.to_vec();
        Ok(EditOutcome::none())
    })
}

/// E31 "Set Body from File", after the confirmation: references not set any more
/// (`setReferencesSet(0)`), the mean shape over the trial (`computeCoordinateSystemAverage`),
/// then the file's points matched to it (`setReferenceFromFile`, which recomputes and
/// filters). The correspondences the dialog applies first are a separate edit
/// ([`super::Edit::SetBodyCorrespondences`]).
///
/// Ported from ui/RigidBodyDialog.cpp, RigidBodyDialog::on_pushButton_setFromFile_clicked
/// (`:285-321`). Ledger A76 (D18) fixed: a rejected file leaves the body as it was; the
/// error is `Rejected` with the dialog's text, or `Io`. With
/// [`xma_model::CppCompat::rejected_reference_file_keeps_reset`] the C++ behaviour: the
/// body keeps `referencesSet` 0 and the mean shape it was given before the file was read
/// (nothing recomputed after), and for a file whose fit was rejected (-2) also the
/// `initialised` flag `fitAndComputeError` set (core/RigidBody.cpp:1446); the edit then
/// succeeds with the dialog's text in [`EditOutcome::refused`] (a file that cannot be read
/// is refused as -1, as the C++ reads it).
pub(super) fn set_references_from_file(
    doc: &mut Document,
    trial: usize,
    body: usize,
    path: &Path,
) -> EditResult {
    check_body(doc, trial, body)?;
    let keep_reset = doc.compat().rejected_reference_file_keeps_reset;
    keeping_bodies_current(doc, trial, &[], |doc| {
        body_mut(doc, trial, body).references_set = 0;
        compute_coordinate_system_average(doc, trial, body);
        match set_reference_from_file(doc, trial, body, path) {
            Ok(messages) => Ok(outcome(messages)),
            Err(EditError::Rejected(text)) if keep_reset => {
                if text == super::body_ops::REFERENCE_FILE_NOT_FITTED {
                    body_mut(doc, trial, body).initialised = true;
                }
                Ok(EditOutcome::refused(text))
            }
            // A file that does not open reads as two empty lines: -1 (`:2484-2499`).
            Err(EditError::Io { .. }) if keep_reset => Ok(EditOutcome::refused(
                super::body_ops::REFERENCE_FILE_MISMATCH,
            )),
            Err(e) => Err(e),
        }
    })
}

/// E32 "Set Body from Frame" (0-based frame): the markers' 3D points there become the
/// references (`setReferenceFromFrame`, which recomputes and filters). `Rejected` with the
/// dialog's text when a marker is not valid in 3D at the frame. The dialog's range (the
/// trial's start and end frames) is the interface's; any frame of the trial is accepted.
///
/// Ported from ui/RigidBodyDialog.cpp, RigidBodyDialog::on_pushButton_setFromFrame_clicked
/// (`:323-340`).
pub(super) fn set_references_from_frame(
    doc: &mut Document,
    trial: usize,
    body: usize,
    frame: usize,
) -> EditResult {
    check_body(doc, trial, body)?;
    keeping_bodies_current(doc, trial, &[], |doc| {
        set_reference_from_frame(doc, trial, body, frame)?;
        Ok(EditOutcome::none())
    })
}

/// E33 "Reset To Marker References" (`resetReferences`: references from the markers when
/// every marker has one, else none; recompute and filter).
///
/// Ported from ui/RigidBodyDialog.cpp, RigidBodyDialog::on_pushButton_Reset_clicked
/// (`:511-516`).
pub(super) fn reset_references(doc: &mut Document, trial: usize, body: usize) -> EditResult {
    check_body(doc, trial, body)?;
    keeping_bodies_current(doc, trial, &[], |doc| {
        reset_references_op(doc, trial, body);
        Ok(EditOutcome::none())
    })
}

/// E34 "optimized from tracked" (`setOptimized`, then `recomputeTransformations` without
/// filtering).
///
/// Ported from ui/RigidBodyDialog.cpp, RigidBodyDialog::on_checkBox_Optimized_clicked
/// (`:518-523`).
pub(super) fn set_optimized(
    doc: &mut Document,
    trial: usize,
    body: usize,
    optimized: bool,
) -> EditResult {
    check_body(doc, trial, body)?;
    keeping_bodies_current(doc, trial, &[], |doc| {
        set_optimized_op(doc, trial, body, optimized);
        cpp_recompute(doc, trial, body, false);
        Ok(EditOutcome::none())
    })
}

/// E35 "Update Optimzed": recomputes the body's poses (not filtered in the C++; filtered in
/// the default mode). The optimised points themselves are not recomputed.
///
/// Ported from ui/RigidBodyDialog.cpp, RigidBodyDialog::on_pushButtonUpdate_clicked
/// (`:525-528`).
pub(super) fn update_optimized(doc: &mut Document, trial: usize, body: usize) -> EditResult {
    check_body(doc, trial, body)?;
    keeping_bodies_current(doc, trial, &[body], |doc| {
        cpp_recompute(doc, trial, body, false);
        Ok(EditOutcome::none())
    })
}

/// E36 "Draw Rigid body".
///
/// Ported from ui/RigidBodyDialog.cpp, RigidBodyDialog::on_checkBox_Draw_clicked (`:370-373`).
pub(super) fn set_visible(
    doc: &mut Document,
    trial: usize,
    body: usize,
    visible: bool,
) -> EditResult {
    check_body(doc, trial, body)?;
    if doc.project.trials[trial].rigid_bodies[body].visible != visible {
        body_mut(doc, trial, body).visible = visible;
    }
    Ok(EditOutcome::none())
}

/// E37: the body's colour. Whether a cancelled colour dialog applies (D22) is the
/// interface's.
///
/// Ported from ui/RigidBodyDialog.cpp, RigidBodyDialog::on_toolButton_Color_clicked
/// (`:375-386`).
pub(super) fn set_color(doc: &mut Document, trial: usize, body: usize, color: Rgb) -> EditResult {
    check_body(doc, trial, body)?;
    if doc.project.trials[trial].rigid_bodies[body].color != color {
        body_mut(doc, trial, body).color = color;
    }
    Ok(EditOutcome::none())
}

/// The dialog text when a mesh cannot be loaded.
pub const MESH_NOT_LOADED: &str = "Could not load OBJ-file";

/// E38: the body's mesh (an OBJ file), or with `None` no mesh. A file that does not open is
/// `Rejected` with the dialog's text and the body keeps its mesh (the C++ has already
/// deleted it, `core/RigidBody.cpp:320-339`).
///
/// Ported from ui/RigidBodyDialog.cpp, RigidBodyDialog::on_toolButton_Mesh_clicked
/// (`:535-551`) and core/RigidBody.cpp, RigidBody::addMeshModel. Whether the file loads is
/// tested as the reader tests it ([`mesh_opens`]).
pub(super) fn set_mesh(
    doc: &mut Document,
    trial: usize,
    body: usize,
    path: Option<&Path>,
) -> EditResult {
    check_body(doc, trial, body)?;
    let Some(path) = path else {
        let rb = &doc.project.trials[trial].rigid_bodies[body];
        if rb.mesh_model.is_some() || rb.mesh_readable.is_some() {
            let rb = body_mut(doc, trial, body);
            rb.mesh_model = None;
            rb.mesh_readable = None;
        }
        return Ok(EditOutcome::none());
    };
    let Some(text) = path.to_str() else {
        return Err(EditError::Invalid(format!(
            "{}: the path is not valid Unicode",
            path.display()
        )));
    };
    if !mesh_opens(text) {
        return Err(EditError::Rejected(MESH_NOT_LOADED.to_owned()));
    }
    let rb = body_mut(doc, trial, body);
    rb.mesh_model = Some(text.to_owned());
    rb.mesh_readable = Some(true);
    Ok(EditOutcome::none())
}

/// E39 "Draw bone mesh model".
///
/// Ported from ui/RigidBodyDialog.cpp, RigidBodyDialog::on_checkbox_DrawMesh_clicked
/// (`:530-533`).
pub(super) fn set_draw_mesh(
    doc: &mut Document,
    trial: usize,
    body: usize,
    draw: bool,
) -> EditResult {
    check_body(doc, trial, body)?;
    if doc.project.trials[trial].rigid_bodies[body].draw_mesh_model != draw {
        body_mut(doc, trial, body).draw_mesh_model = draw;
    }
    Ok(EditOutcome::none())
}

/// E40 "mesh scale". `Invalid` for a scale that is negative or not finite (the spin box
/// holds 0 to 10000).
///
/// Ported from ui/RigidBodyDialog.cpp, RigidBodyDialog::on_doubleSpinBoxMeshScale_valueChanged
/// (`:553-556`).
pub(super) fn set_mesh_scale(
    doc: &mut Document,
    trial: usize,
    body: usize,
    scale: f64,
) -> EditResult {
    check_body(doc, trial, body)?;
    if !scale.is_finite() || scale < 0.0 {
        return Err(EditError::Invalid(format!("mesh scale {scale}")));
    }
    if doc.project.trials[trial].rigid_bodies[body].mesh_scale != scale {
        body_mut(doc, trial, body).mesh_scale = scale;
    }
    Ok(EditOutcome::none())
}

/// E41 "Override Trial Cutoff Frequency": the flag and, when on, the body's cutoff. Off
/// leaves the stored cutoff as it is. The C++ does not filter again (D10); the default
/// mode does. `Invalid` for a cutoff that is negative or not finite.
///
/// Ported from ui/RigidBodyDialog.cpp, RigidBodyDialog::on_checkBoxCutoffOverride_clicked and
/// on_doubleSpinBoxCutoff_valueChanged (`:388-410`).
pub(super) fn set_cutoff_override(
    doc: &mut Document,
    trial: usize,
    body: usize,
    enabled: bool,
    cutoff_frequency: f64,
) -> EditResult {
    check_body(doc, trial, body)?;
    if enabled && (!cutoff_frequency.is_finite() || cutoff_frequency < 0.0) {
        return Err(EditError::Invalid(format!(
            "cutoff frequency {cutoff_frequency}"
        )));
    }
    keeping_bodies_current(doc, trial, &[], |doc| {
        let rb = &doc.project.trials[trial].rigid_bodies[body];
        let same = rb.override_cutoff_frequency == enabled
            && (!enabled || rb.cutoff_frequency == cutoff_frequency);
        if !same {
            let rb = body_mut(doc, trial, body);
            rb.override_cutoff_frequency = enabled;
            if enabled {
                rb.cutoff_frequency = cutoff_frequency;
            }
        }
        Ok(EditOutcome::none())
    })
}

/// E42 "Add virtual point": `addDummyPoint` with the dialog's answers, then
/// `recomputeTransformations` (not filtered in the C++). The name is stored as given (the
/// dialog appends " RB<n>" for a point driven by a body). With
/// [`VirtualPointInput::Body`] and no second file, the reference file gives the position in
/// the driving body too.
///
/// A point that would close a cycle of virtual points (this body followed, through other
/// bodies' virtual points, by the driving body; or driven by itself) is `Rejected` in both
/// modes with a sentence naming the bodies (ledger A92: XMALab recurses without end).
/// Which source the reference file's fourth value may choose: see [`add_dummy_point`]
/// (ledger A83).
///
/// Ported from ui/RigidBodyDialog.cpp, RigidBodyDialog::on_pushButton_AddDummy_clicked
/// (`:412-500`) and core/RigidBody.cpp, RigidBody::addDummyPoint (`:1956-2031`).
pub(super) fn add_virtual_point(
    doc: &mut Document,
    trial: usize,
    body: usize,
    name: &str,
    reference_file: &Path,
    input: &VirtualPointInput,
) -> EditResult {
    check_body(doc, trial, body)?;
    if name.is_empty() {
        return Err(EditError::Invalid(
            "a virtual point needs a name".to_owned(),
        ));
    }
    let source = match input {
        VirtualPointInput::Body {
            body: driver,
            driver_file,
        } => {
            check_body(doc, trial, *driver)?;
            DummySource {
                body: Some(*driver),
                reference2: Some(driver_file.as_deref().unwrap_or(reference_file)),
                coordinates: None,
            }
        }
        VirtualPointInput::Coordinates { file } => DummySource {
            body: None,
            reference2: None,
            coordinates: Some(file.as_path()),
        },
    };
    let had_cycle = virtual_point_cycle(&doc.project.trials[trial]).is_some();
    keeping_bodies_current(doc, trial, &[], |doc| {
        let messages = add_dummy_point(doc, trial, body, name, reference_file, &source)?;
        // Ledger A92: refused before anything is computed with the cycle (a trial opened
        // with a cycle is left to `refuse_new_cycles`, which accepts it).
        let t = &doc.project.trials[trial];
        if let Some(cycle) = virtual_point_cycle(t).filter(|_| !had_cycle) {
            return Err(EditError::Rejected(cycle_message(t, &cycle)));
        }
        cpp_recompute(doc, trial, body, false);
        Ok(outcome(messages))
    })
}

/// E43 "Delete all virtual points" (`clearAllDummyPoints`, then `recomputeTransformations`,
/// not filtered in the C++).
///
/// Ported from ui/RigidBodyDialog.cpp, RigidBodyDialog::on_pushButton_DeleteDummy_clicked
/// (`:502-509`) and core/RigidBody.cpp, RigidBody::clearAllDummyPoints.
pub(super) fn clear_virtual_points(doc: &mut Document, trial: usize, body: usize) -> EditResult {
    check_body(doc, trial, body)?;
    keeping_bodies_current(doc, trial, &[], |doc| {
        if !doc.project.trials[trial].rigid_bodies[body]
            .virtual_points
            .is_empty()
        {
            body_mut(doc, trial, body).virtual_points.clear();
        }
        cpp_recompute(doc, trial, body, false);
        Ok(EditOutcome::none())
    })
}

/// E48 "Update": every body of the trial recomputed and filtered. In XMALab 3 one body at
/// a time, without clearing other bodies' pose flags
/// (`Trial::recomputeAndFilterRigidBodyTransformations`, ledger A36); in the default mode
/// as the trial update does (A36 fixed).
///
/// Ported from ui/PlotWindow.cpp, PlotWindow::on_pushButtonUpdate_clicked (`:2813-2828`)
/// and core/Trial.cpp, Trial::recomputeAndFilterRigidBodyTransformations (`:939-947`).
pub(super) fn recompute_bodies(doc: &mut Document, trial: usize) -> EditResult {
    check_trial(doc, trial)?;
    let all: Vec<usize> = (0..doc.project.trials[trial].rigid_bodies.len()).collect();
    keeping_bodies_current(doc, trial, &all, |doc| {
        if cpp_recomputation(doc) {
            doc.recompute_and_filter_rigid_body_transformations(trial);
        }
        Ok(EditOutcome::none())
    })
}
