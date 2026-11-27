//! Rigid-body transformations: File > Export Rigid Body Transformations.
//!
//! Ported from core/Trial.cpp, `Trial::saveRigidBodyTransformations` (`:949-1151`), with
//! the options `ui/MainWindow.cpp`, `MainWindow::saveRigidBodies` (`:1871-1933`) passes, and
//! core/RigidBody.cpp, `RigidBody::getTransformationMatrix` (`:2277-2319`).
//!
//! The C++ first recomputes every exported body (`recomputeTransformations`, then
//! `filterTransformations` when filtered; ledger A36). That is the engine's job: these
//! functions write the poses the model holds.

use xma_model::{CppCompat, PoseTrack, RigidBody, Trial};
use xma_vision::camera::rodrigues_to_matrix;

use super::{
    ExportError, ExportFile, ExportOutput, FrameRange, body_export_name, check_bodies, check_range,
    export_frames, g, padded,
};
use crate::cfmt::format_qt_number;

/// Options of the rigid-body export. The default is what the GUI passes by default: one
/// file, header, **filtered**, no frame column (`core/Settings.cpp:196-203`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransformationsOptions {
    /// One file for all bodies (`ExportTransMulti`, default) or one per body
    /// (`ExportTransSingle`).
    pub one_file: bool,
    /// Header row (`ExportTransHeader`, default on).
    pub header: bool,
    /// Filtered poses (`ExportTransFiltered`, default) or unfiltered (`ExportTransUnfiltered`).
    pub filtered: bool,
    /// A `Frame` column (1-based) and only these frames (`ExportTransOffsetCols`, default
    /// off, which writes every frame and no frame column).
    pub frames: Option<FrameRange>,
    /// Which faults of XMALab 3 to reproduce: [`CppCompat::export_details`] (ledger F25: a
    /// zero translation written `-0`, file names with an empty description, line ends
    /// without bodies). Default: fixed.
    pub compat: CppCompat,
}

impl Default for TransformationsOptions {
    fn default() -> Self {
        Self {
            one_file: true,
            header: true,
            filtered: true,
            frames: None,
            compat: CppCompat::FIXED,
        }
    }
}

/// The 16 values the export writes for a body at a frame, or `None` when the pose (raw or
/// filtered) is absent.
///
/// Ported from core/RigidBody.cpp, RigidBody::getTransformationMatrix (`:2277-2319`):
/// `trans[4y + k] = R(y, k)` with `R = Rodrigues(r)` (world to body), `trans[3] = trans[7] =
/// trans[11] = 0`, `trans[12..15] = Rᵀ(−t)` summed left to right, `trans[15] = 1`. Read
/// column-major this is `[Rᵀ | −Rᵀt]`, body to world (ledger F3).
#[must_use]
pub fn transformation_matrix(track: &PoseTrack, frame: usize) -> Option<[f64; 16]> {
    let pose = track.pose(frame)?;
    let r = rodrigues_to_matrix(&pose.rotation);
    let mut trans = [0.0; 16];
    for y in 0..3 {
        trans[y * 4] = r[(y, 0)];
        trans[y * 4 + 1] = r[(y, 1)];
        trans[y * 4 + 2] = r[(y, 2)];
        trans[y * 4 + 3] = 0.0;
    }
    let t = pose.translation;
    trans[12] = trans[0] * -t[0] + trans[4] * -t[1] + trans[8] * -t[2];
    trans[13] = trans[1] * -t[0] + trans[5] * -t[1] + trans[9] * -t[2];
    trans[14] = trans[2] * -t[0] + trans[6] * -t[1] + trans[10] * -t[2];
    trans[15] = 1.0;
    Some(trans)
}

/// The cutoff a body is filtered at: its own when it overrides the trial's, else the
/// trial's (`core/Trial.cpp:983`, `core/RigidBody.cpp:1702`).
#[must_use]
pub fn body_cutoff(trial: &Trial, body: &RigidBody) -> f64 {
    if body.override_cutoff_frequency {
        body.cutoff_frequency
    } else {
        trial.cutoff_frequency
    }
}

/// Whether the GUI lets the user export filtered transformations of `bodies`: the trial's
/// recording speed must be above 0, and the cutoff too. XMALab 3 checks the **trial's**
/// cutoff even for a body that overrides it (`ui/MainWindow.cpp:1878-1881`; ledger F11,
/// [`CppCompat::filtered_export_trial_cutoff`]); fixed, each exported body's own cutoff
/// ([`body_cutoff`]). Otherwise it shows an error and writes nothing. The unfiltered export
/// has no check. Indices that do not exist are ignored here (the export reports them).
#[must_use]
pub fn can_export_filtered(trial: &Trial, bodies: &[usize], compat: CppCompat) -> bool {
    if trial.recording_speed <= 0.0 {
        return false;
    }
    if compat.filtered_export_trial_cutoff {
        return trial.cutoff_frequency > 0.0;
    }
    bodies
        .iter()
        .filter_map(|&b| trial.rigid_bodies.get(b))
        .all(|b| body_cutoff(trial, b) > 0.0)
}

/// `_<cutoff>Hz` with `QString::number` (6 significant digits), or nothing when unfiltered.
fn filter_suffix(trial: &Trial, body: usize, filtered: bool) -> String {
    if filtered {
        format!(
            "_{}Hz",
            format_qt_number(body_cutoff(trial, &trial.rigid_bodies[body]))
        )
    } else {
        String::new()
    }
}

/// The 16 header names of one body, without a line end.
///
/// Ported from core/Trial.cpp:986-1001 (ledger F3: the names do not match the layout).
fn header_body(out: &mut String, trial: &Trial, body: usize, filtered: bool) {
    let name = body_export_name(&trial.rigid_bodies[body].description, body);
    let suffix = filter_suffix(trial, body, filtered);
    const NAMES: [&str; 16] = [
        "_R11", "_R12", "_R13", "_01", "_R21", "_R22", "_R23", "_02", "_R31", "_R32", "_R33",
        "_03", "_TX", "_TY", "_TZ", "_1",
    ];
    for (k, n) in NAMES.iter().enumerate() {
        if k > 0 {
            out.push(',');
        }
        out.push_str(&name);
        out.push_str(n);
        out.push_str(&suffix);
    }
}

/// The 16 values of one body at one frame, or 16 × `NaN`, without a line end.
///
/// Ported from core/Trial.cpp:1027-1040.
///
/// Fixed (ledger F25), a zero is written `0`, not `-0` (adding `+0` turns `-0` into `0` and
/// changes nothing else).
fn values_body(out: &mut String, body: &RigidBody, frame: usize, filtered: bool, cpp: bool) {
    let track = if filtered { &body.filtered } else { &body.raw };
    match transformation_matrix(track, frame) {
        Some(trans) => {
            for (k, v) in trans.iter().enumerate() {
                if k > 0 {
                    out.push(',');
                }
                out.push_str(&g(if cpp { *v } else { *v + 0.0 }));
            }
        }
        None => out.push_str("NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN"),
    }
}

/// The one-file export of the given bodies, in the order given.
///
/// Ported from core/Trial.cpp, Trial::saveRigidBodyTransformations, the `onefile` branch
/// (`:957-1054`). `opts.one_file` is not read. With no bodies, rows hold only the frame
/// column, if any, with no line end.
///
/// A frame past a body's frames (undefined behaviour in the C++) writes 16 × `NaN`.
///
/// # Errors
/// When a body does not exist or the frame range goes past the trial (undefined behaviour
/// in the C++); nothing is written.
pub fn transformations_file(
    trial: &Trial,
    bodies: &[usize],
    opts: &TransformationsOptions,
) -> Result<String, ExportError> {
    check_bodies(trial, bodies)?;
    check_range(opts.frames, trial.n_frames)?;
    if bodies.is_empty() && !opts.compat.export_details {
        return Ok(super::empty_items_text(
            opts.header,
            opts.frames
                .map(|_| export_frames(opts.frames, trial.n_frames)),
        ));
    }
    let mut out = String::new();
    if opts.header {
        if opts.frames.is_some() {
            out.push_str("Frame,");
        }
        for (k, &b) in bodies.iter().enumerate() {
            header_body(&mut out, trial, b, opts.filtered);
            out.push(if k != bodies.len() - 1 { ',' } else { '\n' });
        }
    }
    for f in export_frames(opts.frames, trial.n_frames) {
        if opts.frames.is_some() {
            out.push_str(&(f + 1).to_string());
            out.push(',');
        }
        for (k, &b) in bodies.iter().enumerate() {
            values_body(
                &mut out,
                &trial.rigid_bodies[b],
                f,
                opts.filtered,
                opts.compat.export_details,
            );
            out.push(if k != bodies.len() - 1 { ',' } else { '\n' });
        }
    }
    Ok(out)
}

/// The export of one body, as written in the one-file-per-body mode.
///
/// Ported from core/Trial.cpp, Trial::saveRigidBodyTransformations, the per-body branch
/// (`:1057-1149`).
///
/// # Panics
/// When the body does not exist (checked by [`transformations`] first).
#[must_use]
pub fn transformations_body_file(
    trial: &Trial,
    body: usize,
    opts: &TransformationsOptions,
) -> String {
    let mut out = String::new();
    if opts.header {
        if opts.frames.is_some() {
            out.push_str("Frame,");
        }
        header_body(&mut out, trial, body, opts.filtered);
        out.push('\n');
    }
    for f in export_frames(opts.frames, trial.n_frames) {
        if opts.frames.is_some() {
            out.push_str(&(f + 1).to_string());
            out.push(',');
        }
        values_body(
            &mut out,
            &trial.rigid_bodies[body],
            f,
            opts.filtered,
            opts.compat.export_details,
        );
        out.push('\n');
    }
    out
}

/// The file name of one body's export:
/// `RigidBody%03d_<description>_transformationFiltered_<cutoff>Hz.csv` or
/// `RigidBody%03d_<description>_transformation.csv`.
///
/// Ported from core/Trial.cpp:1059-1072.
#[must_use]
pub fn transformations_body_file_name(
    trial: &Trial,
    body: usize,
    filtered: bool,
    compat: CppCompat,
) -> String {
    let d = &trial.rigid_bodies[body].description;
    let desc = &super::file_description(d, body_export_name(d, body), compat.export_details);
    if filtered {
        format!(
            "RigidBody{}_{}_transformationFiltered_{}Hz.csv",
            padded(body),
            desc,
            format_qt_number(body_cutoff(trial, &trial.rigid_bodies[body]))
        )
    } else {
        format!("RigidBody{}_{}_transformation.csv", padded(body), desc)
    }
}

/// The rigid-body export as the GUI writes it: one file, or one file per body.
///
/// Ported from core/Trial.cpp, Trial::saveRigidBodyTransformations, without its first step
/// (the recomputation, which the engine runs; ledger A36). The GUI's check
/// [`can_export_filtered`] comes before it.
///
/// # Errors
/// As [`transformations_file`].
pub fn transformations(
    trial: &Trial,
    bodies: &[usize],
    opts: &TransformationsOptions,
) -> Result<ExportOutput, ExportError> {
    if opts.one_file {
        transformations_file(trial, bodies, opts).map(ExportOutput::File)
    } else {
        check_bodies(trial, bodies)?;
        check_range(opts.frames, trial.n_frames)?;
        Ok(ExportOutput::Folder(
            bodies
                .iter()
                .map(|&b| {
                    ExportFile::new(
                        transformations_body_file_name(trial, b, opts.filtered, opts.compat),
                        transformations_body_file(trial, b, opts),
                    )
                })
                .collect(),
        ))
    }
}
