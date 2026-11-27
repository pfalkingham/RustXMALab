//! `xma edit`: apply an edit script (`oracle/FORMAT.md`, "Edit scripts") to a project
//! headlessly, and the runner `xma parity` uses to replay the C++ editing reference.
//!
//! Each script operation is applied as the interface applies it: the edit itself
//! ([`Document::apply`], one undo step per operation), preceded by what XMALab's interface
//! does around it (making a trial active for an action that acts on the active trial), and,
//! in the XMALab 3 mode, with the values passed through XMALab's dialog controls
//! ([`xmalab_dialog`]: spin-box ranges and decimals, the fields a dialog writes whether or
//! not they were changed).

use std::path::{Path, PathBuf};

use xma_engine::edit::script::{ScriptLine, ScriptOp, parse_script};
use xma_engine::edit::{
    Edit, EditError, FrameRange, MarkerProperties, TrialProperties, VirtualPointInput,
};
use xma_engine::tracking::{TrialSources, place_point, refine_points};
use xma_engine::{Document, SaveOptions, SaveTime};

use crate::args::{self, Opt, parse};
use crate::settings::{self, SettingsArgs};
use crate::{CliError, Io};

pub const HELP: &str = "\
xma edit: apply an edit script to a project

Usage: xma edit <project.xma> --script <file> [-o <out.xma>] [options]

The script holds one operation per line, named and laid out as the engine's edits
(oracle/FORMAT.md, \"Edit scripts\"; for example `SetPoint trial=0 marker=3 camera=1
frame=10 x=540.25 y=456 status=60`). The project is opened as XMALab opens it (the active
trial is updated), then each operation is applied in order as one undo step, as the
interface applies it: an action that works on the active trial first makes its trial the
active one (as the trial selector does, which updates that trial). Each operation's result
is printed: ok, rejected (with XMALab's message; nothing changed), invalid (an index out of
range, a value a dialog would not accept), or not applied.

`Dump` lines are only reported. `Save name=<file>` saves the project as XMALab's Save As
into the --out-dir folder (skipped without it). `PlacePoint` (a click in a camera view or
the Detailed View) and `RefinePoints` (the polynomial refinement) read the trial's images
from its video files as the project stores them.

With --xmalab3-compat the values also pass through XMALab 3's dialog controls as the C++
editing reference does: 2-decimal spin boxes (recording speed, cutoffs, mesh scale), their
ranges (penalty 0 to 100, size override -1 to 50, threshold 0 to 100, counts up to 1000),
fields a dialog writes back unchanged, \" RB<n>\" appended to the name of a virtual point
driven by a body.

Options:
      --script <file>      The edit script (required)
  -o, --output <out.xma>   Save the result there at the end, as Save As does
      --out-dir <dir>      Folder for the script's own Save operations
      --save-time <YYYY-MM-DDTHH:MM:SS>
                           Stamp saved files with this time instead of now
      --recompute-when-saving
                           XMALab's RecomputeWhenSaving setting [default: off]
      --xmalab3-compat     Reproduce the XMALab 3 faults this program fixes, and XMALab 3's
                           dialog controls [default: off]
  -h, --help               Print help
";

const OPTS: [Opt; 6] = [
    Opt::value("script"),
    Opt::value("output").short('o'),
    Opt::value("out-dir"),
    Opt::value("save-time"),
    Opt::flag("recompute-when-saving"),
    Opt::flag("xmalab3-compat"),
];

/// The result of one operation, as the C++ reference's `ops_log.csv` names them.
#[derive(Clone, Debug, PartialEq)]
pub enum OpStatus {
    Ok,
    /// XMALab refuses with this message; nothing changed.
    Rejected(String),
    /// An index out of range or a value that does not apply; nothing changed.
    Invalid(String),
    /// Not applied here (an operation not ported).
    NotApplied(String),
}

impl OpStatus {
    /// `ok`, `rejected`, `invalid` or `not applied`.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            OpStatus::Ok => "ok",
            OpStatus::Rejected(_) => "rejected",
            OpStatus::Invalid(_) => "invalid",
            OpStatus::NotApplied(_) => "not applied",
        }
    }

    /// The message, empty for `ok`.
    #[must_use]
    pub fn message(&self) -> &str {
        match self {
            OpStatus::Ok => "",
            OpStatus::Rejected(m) | OpStatus::Invalid(m) | OpStatus::NotApplied(m) => m,
        }
    }
}

/// What applying one operation did besides the edit.
#[derive(Clone, Debug, PartialEq)]
pub struct OpOutcome {
    pub status: OpStatus,
    /// Notes: a change of active trial, the edit's own messages.
    pub notes: Vec<String>,
}

/// Values as XMALab 3's dialog controls give them to the model, as the C++ editing
/// reference runs them (`oracle/xmalab-cpp/src/ui/EditOracleDriver.cpp`, copies of the
/// handlers named there). Not edits of the model: what the interface hands the edit.
///
/// - Marker dialog (`ui/MarkerDialog.cpp:79-87`): every control is written on OK, a field
///   not given is the stored value as the control shows it; penalty 0 to 100, size override
///   -1 to 50, threshold offset 0 to 100 (`ui/ui-files/MarkerDialog.ui`).
/// - Set number of markers or bodies (`ui/PointsDockWidget.cpp:455-503`): 0 to 1000.
/// - Rigid-body dialog: the mesh scale and the cutoff through 2-decimal spin boxes
///   (`ui/RigidBodyDialog.cpp:388-410`); a virtual point driven by a body gets " RB<n>"
///   appended to its name (`:466`).
/// - Plot (`ui/PlotWindow.cpp:3119-3155`): a frame selection for events is put in order
///   and clamped to the trial's start and end frames.
/// - Trial dialog (`ui/TrialDialog.cpp:111-139`): recording speed and cutoff through
///   2-decimal spin boxes, a field not given being the stored value as shown (D14); the
///   reference calibration combo box lists the images calibrated in every camera and shows
///   the first of them when the stored one is not listed.
///
/// # Errors
/// A reference calibration image the Trial dialog does not list.
pub fn xmalab_dialog(doc: &Document, edit: &Edit) -> Result<Edit, EditError> {
    let spin2 = |v: f64, min: f64, max: f64| spin_box(v, min, max, 2);
    Ok(match edit {
        Edit::SetMarkerProperties {
            trial,
            marker,
            properties,
        } => {
            let Some(m) = doc
                .project()
                .trials
                .get(*trial)
                .and_then(|t| t.markers.get(*marker))
            else {
                return Ok(edit.clone());
            };
            let p = properties;
            Edit::SetMarkerProperties {
                trial: *trial,
                marker: *marker,
                properties: MarkerProperties {
                    method: Some(p.method.unwrap_or(m.method)),
                    threshold_offset: Some(
                        p.threshold_offset
                            .unwrap_or(m.threshold_offset)
                            .clamp(0, 100),
                    ),
                    size_override: Some(p.size_override.unwrap_or(m.size_override).clamp(-1, 50)),
                    max_penalty: Some(p.max_penalty.unwrap_or(m.max_penalty).clamp(0, 100)),
                },
            }
        }
        // PlotWindow::setEventOn / setEventOff (ui/PlotWindow.cpp:3119-3155): the
        // selection's ends in order, clamped to the trial's start and end frames.
        Edit::SetEventFrames { trial, frames, on } => {
            let Some(t) = doc.project().trials.get(*trial) else {
                return Ok(edit.clone());
            };
            let (a, b) = (frames.first.min(frames.last), frames.first.max(frames.last));
            let first = a.max(t.start_frame.saturating_sub(1));
            let last = b.min(t.end_frame.saturating_sub(1));
            Edit::SetEventFrames {
                trial: *trial,
                frames: FrameRange { first, last },
                on: *on,
            }
        }
        Edit::SetMarkerCount { trial, count } => Edit::SetMarkerCount {
            trial: *trial,
            count: (*count).min(1000),
        },
        Edit::SetRigidBodyCount { trial, count } => Edit::SetRigidBodyCount {
            trial: *trial,
            count: (*count).min(1000),
        },
        Edit::SetBodyMeshScale { trial, body, scale } => Edit::SetBodyMeshScale {
            trial: *trial,
            body: *body,
            scale: spin2(*scale, 0.0, 10000.0),
        },
        Edit::SetBodyCutoffOverride {
            trial,
            body,
            enabled,
            cutoff_frequency,
        } => Edit::SetBodyCutoffOverride {
            trial: *trial,
            body: *body,
            enabled: *enabled,
            cutoff_frequency: spin2(*cutoff_frequency, 0.0, 10000.0),
        },
        Edit::AddVirtualPoint {
            trial,
            body,
            name,
            reference_file,
            input,
        } => {
            let name = match input {
                VirtualPointInput::Body { body: driver, .. } => {
                    format!("{name} RB{}", driver + 1)
                }
                VirtualPointInput::Coordinates { .. } => name.clone(),
            };
            Edit::AddVirtualPoint {
                trial: *trial,
                body: *body,
                name,
                reference_file: reference_file.clone(),
                input: input.clone(),
            }
        }
        Edit::SetTrialProperties { trial, properties } => {
            let Some(t) = doc.project().trials.get(*trial) else {
                return Ok(edit.clone());
            };
            let p = properties;
            let cams = &doc.project().cameras;
            let listed: Vec<usize> = cams.first().map_or_else(Vec::new, |c0| {
                (0..c0.calibration_images.len())
                    .filter(|&i| {
                        cams.iter().all(|c| {
                            c.calibration_images
                                .get(i)
                                .is_some_and(|im| im.is_calibrated())
                        })
                    })
                    .collect()
            });
            let reference = if listed.is_empty() {
                None
            } else {
                let r = p
                    .reference_calibration
                    .unwrap_or(t.reference_calibration_image);
                if listed.contains(&r) {
                    Some(r)
                } else if p.reference_calibration.is_some() {
                    return Err(EditError::Invalid(format!(
                        "the Trial dialog does not list calibration image {r}"
                    )));
                } else {
                    Some(listed[0])
                }
            };
            Edit::SetTrialProperties {
                trial: *trial,
                properties: TrialProperties {
                    recording_speed: Some(spin2(
                        p.recording_speed.unwrap_or(t.recording_speed),
                        0.0,
                        10000.0,
                    )),
                    cutoff_frequency: Some(spin2(
                        p.cutoff_frequency.unwrap_or(t.cutoff_frequency),
                        0.0,
                        10000.0,
                    )),
                    interpolate_3d: Some(p.interpolate_3d.unwrap_or(t.interpolate_3d)),
                    reference_calibration: reference,
                    force_update: p.force_update,
                },
            }
        }
        other => other.clone(),
    })
}

/// The value a Qt spin box with `decimals` decimals and the range `min..=max` holds after
/// `setValue(v)`: clamped, then rounded to the decimals (`QDoubleSpinBox::round`, which
/// formats with `QString::number(v, 'f', decimals)` and reads it back).
#[must_use]
pub fn spin_box(v: f64, min: f64, max: f64, decimals: usize) -> f64 {
    let v = v.clamp(min, max);
    format!("{v:.decimals$}").parse().unwrap_or(v)
}

/// Applies one operation as the interface does (see the module text). `xmalab3_dialogs`:
/// pass the values through XMALab 3's dialog controls ([`xmalab_dialog`]). `Dump` and
/// `Save` do nothing here (the caller handles them).
pub fn apply_op(doc: &mut Document, op: &ScriptOp, xmalab3_dialogs: bool) -> OpOutcome {
    let mut notes = Vec::new();
    // The handlers act on the active trial: the trial selector makes it active (and updates
    // it) first (`EditOracleDriver::activate`).
    let activate = match op {
        ScriptOp::Edit(Edit::ApplyDefaultToTrials) => {
            doc.project().trials.iter().position(|t| t.is_default)
        }
        _ => op.handler_trial(),
    };
    if let Some(t) = activate
        && t < doc.project().trials.len()
        && doc.project().active_trial != t
    {
        doc.select_trial(t);
        notes.push(format!("active trial changed to {t}"));
    }
    let status = match op {
        ScriptOp::Dump { .. } | ScriptOp::Save { .. } => OpStatus::Ok,
        ScriptOp::PlacePoint {
            trial,
            camera,
            frame,
            x,
            y,
            placement,
            ..
        } => {
            let mut images = match TrialSources::new(doc, *trial) {
                Ok(i) => i,
                Err(e) => return finish(EditError::Invalid(e), notes),
            };
            match place_point(
                doc,
                &mut images,
                *trial,
                *camera,
                *frame,
                *x,
                *y,
                *placement,
            ) {
                Ok(p) => {
                    if p.marker_added {
                        notes.push(format!("marker {} added", p.marker));
                    }
                    notes.push(format!("point placed on marker {}", p.marker));
                    OpStatus::Ok
                }
                Err(e) => return finish(e, notes),
            }
        }
        ScriptOp::RefinePoints {
            trial,
            markers,
            frames,
            which,
        } => {
            let mut images = match TrialSources::new(doc, *trial) {
                Ok(i) => i,
                Err(e) => return finish(EditError::Invalid(e), notes),
            };
            match refine_points(
                doc,
                &mut images,
                *trial,
                markers,
                *frames,
                *which,
                &mut |_| true,
            ) {
                Ok(n) => {
                    notes.push(format!("{n} points refined"));
                    OpStatus::Ok
                }
                Err(e) => return finish(e, notes),
            }
        }
        ScriptOp::Edit(edit) => {
            let edit = if xmalab3_dialogs {
                match xmalab_dialog(doc, edit) {
                    Ok(e) => e,
                    Err(e) => return finish(e, notes),
                }
            } else {
                edit.clone()
            };
            let active_before = doc.project().active_trial;
            match doc.apply(&edit) {
                Ok(outcome) => {
                    notes.extend(outcome.messages);
                    if let Some(text) = outcome.refused {
                        return OpOutcome {
                            status: OpStatus::Rejected(text),
                            notes,
                        };
                    }
                    // MainWindow.cpp:2190-2194: after each trial imported (not after one
                    // declined), re-entering Marker tracking updates the active trial
                    // (WorkspaceNavigationFrame.cpp:166-169).
                    if let Edit::ImportTrials { trials, .. } = &edit {
                        let declined = notes
                            .iter()
                            .filter(|m| m.contains("was not imported"))
                            .count();
                        for _ in 0..trials.len().saturating_sub(declined) {
                            let active = doc.project().active_trial;
                            doc.update_trial(active);
                        }
                    }
                    // MainWindow.cpp:2224-2225: after deleting trials the active index is
                    // only clamped, with the trial update of a new active trial.
                    if let Edit::DeleteTrials { .. } = edit {
                        let n = doc.project().trials.len();
                        if n > 0 && active_before >= n {
                            doc.select_trial(n - 1);
                        }
                    }
                    OpStatus::Ok
                }
                Err(e) => return finish(e, notes),
            }
        }
    };
    OpOutcome { status, notes }
}

fn finish(e: EditError, notes: Vec<String>) -> OpOutcome {
    let status = match e {
        EditError::Rejected(m) => OpStatus::Rejected(m),
        EditError::NotImplemented(what) => OpStatus::NotApplied(format!("{what} is not ported")),
        other => OpStatus::Invalid(other.to_string()),
    };
    OpOutcome { status, notes }
}

/// Reads a script file; relative paths in it are taken from its folder.
///
/// # Errors
/// The file cannot be read, or a line cannot be parsed.
pub fn read_script(path: &Path) -> Result<Vec<ScriptLine>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let base = path
        .parent()
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
    let base = if base.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        base
    };
    parse_script(&text, &base).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn run(args: &[String], io: &mut Io<'_>) -> Result<(), CliError> {
    let opts: Vec<Opt> = OPTS.iter().chain(&settings::OPTIONS).copied().collect();
    let p = parse(args, &opts)?;
    if p.has("help") {
        io.line(format_args!("{HELP}\n{}", settings::HELP));
        return Ok(());
    }
    let path = p.positional(0, "the project file")?;
    p.at_most(1)?;
    let script = p
        .value("script")
        .ok_or_else(|| CliError::usage("--script <file> is required"))?;
    let settings = SettingsArgs::from_parsed(&p)?;
    let save_time = args::save_time(&p, false)?;
    let lines = read_script(Path::new(script)).map_err(CliError::failed)?;
    let compat = crate::compat_mode(&p);
    let mut doc = crate::open_document(Path::new(path), &settings, compat)?;
    crate::with_console_lines(&mut doc, io, |doc, io| {
        io.line(format_args!(
            "{path}: opened (compatibility mode: {}); {} operations in {script}",
            doc.compat().describe(),
            lines.len()
        ));
        let save_options = |time: &Option<SaveTime>| {
            let mut o = SaveOptions {
                recompute_when_saving: p.has("recompute-when-saving"),
                ..SaveOptions::default()
            };
            if let Some(t) = time {
                o.time = t.clone();
            }
            o
        };
        let xmalab3 = p.has("xmalab3-compat");
        let mut not_applied = 0;
        for line in &lines {
            match &line.op {
                ScriptOp::Dump { name } => {
                    io.line(format_args!(
                        "  line {}: Dump {name}: noted (no state dump is written)",
                        line.line
                    ));
                    continue;
                }
                ScriptOp::Save { name } => {
                    match p.value("out-dir") {
                        Some(dir) => {
                            let target = Path::new(dir).join(name);
                            let report =
                                doc.save_as(&target, &save_options(&save_time))
                                    .map_err(|e| {
                                        CliError::failed(format!("{}: {e}", target.display()))
                                    })?;
                            io.line(format_args!(
                                "  line {}: Save: {} ({} entries)",
                                line.line,
                                report.path.display(),
                                report.entries
                            ));
                            crate::console_lines(doc, io);
                        }
                        None => io.line(format_args!(
                            "  line {}: Save {name}: skipped (no --out-dir)",
                            line.line
                        )),
                    }
                    continue;
                }
                _ => {}
            }
            let out = apply_op(doc, &line.op, xmalab3);
            if out.status != OpStatus::Ok {
                not_applied += 1;
            }
            let mut text = format!(
                "  line {}: {}: {}",
                line.line,
                line.op.name(),
                out.status.label()
            );
            if !out.status.message().is_empty() {
                text.push_str(&format!(": {}", out.status.message()));
            }
            for n in &out.notes {
                text.push_str(&format!(" [{n}]"));
            }
            io.line(text);
            // Ledger A210: the operation's cap lines on stderr, after its line.
            crate::console_lines(doc, io);
        }
        io.line(format_args!(
            "{} operations, {not_applied} not applied",
            lines
                .iter()
                .filter(|l| !matches!(l.op, ScriptOp::Dump { .. } | ScriptOp::Save { .. }))
                .count()
        ));
        for w in doc.warnings() {
            io.line(format_args!("  warning: {w}"));
        }
        if let Some(out) = p.value("output") {
            let report = doc
                .save_as(Path::new(out), &save_options(&save_time))
                .map_err(|e| CliError::failed(format!("{out}: {e}")))?;
            io.line(format_args!(
                "saved {} ({} entries)",
                report.path.display(),
                report.entries
            ));
        }
        Ok(())
    })
}
