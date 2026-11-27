//! `xma calibrate`: build or calibrate a project from a script in the calibration-run format
//! (`oracle/FORMAT.md`, "Calibration runs"), headlessly: File > New dataset from raw files,
//! "compute undistortion", the reference clicks and "compute calibration", the outlier and
//! point edits, new trials, and saving, each operation applied as the window applies it
//! (the replay of `xma parity`, [`crate::parity::calibration::apply_op`]).

use std::path::{Path, PathBuf};
use std::time::Instant;

use xma_engine::{SaveOptions, WizardSettings};

use crate::args::{self, Opt, parse};
use crate::parity::calibration::{CalibOp, OpStatus, ReplayState, apply_op, read_calib_script};
use crate::settings::{self, SettingsArgs};
use crate::{CliError, Io};

pub const HELP: &str = "\
xma calibrate: build and calibrate a project from a calibration script

Usage: xma calibrate --script <file> [--project <start.xma>] [-o <out.xma>] [options]

The script holds one operation per line, in the format of the calibration reference runs
(oracle/FORMAT.md, \"Calibration runs\"; paths relative to the script's folder), for example:

  NewProject cameras=2 grid0=grid1.tif images0=cube1.tif grid1=grid2.tif images1=cube2.tif framespec=FrameSpecification.csv refs=Reference_points.ref
  UndistortionButton
  Workspace name=calibration
  ClickReference camera=0 image=0 x=518.2 y=615.7
  ...
  NewTrial cam0=Camera1.avi cam1=Camera2.avi
  Save name=calibrated.xma

Operations: NewProject, NewProjectExternal, NewTrial, NewTrialNoCalibration,
LoadWithDifferentCalibration, Workspace, SelectCalibrationImage, SelectTrial,
UndistortionButton, UndistToggleOutlier, UndistSetCenter, UndistRemoveOutliers, ClickReference
(alt=true snaps with DetectionMethodForCalibration), CalibMovePoint, CalibToggleInlier (a
double click; this program's own), CalibrationButton, ManualMode, ManualCheckbox,
CalibResetFrame, CalibResetCamera, CalibDeleteFrame, CalibDistortion, Optimize, Seed, Setting,
SettingText, Save, Dump (reported only), Wait (ignored). A checkerboard dataset: NewProject
target=checkerboard horizontal= vertical= square_size= (the dialog's three spin boxes) instead
of framespec= and refs=, light<c>=true for a light camera, flip<c>=true to flip; then
ClickReference is the interior upper-left corner (or, with DisableCheckerboardDetection or
after a failed detection, the next of the four corners). NewTrial takes cam<c>dir=<folder>
for an image folder. Comment lines
`# setting: NAME=VALUE` are applied before the first operation. Each operation's result is
printed with its time: ok, rejected (XMALab's message; nothing changed), invalid, or not
implemented (the multi-camera optimisation and the MayaCam files are not built here yet).

Without --project the script must create the project (NewProject, ...). The project is
computed in the default mode (the known faults of XMALab 3 fixed), or as XMALab 3 with
--xmalab3-compat.

Options:
      --script <file>      The calibration script (required)
      --project <file>     Open this project first
  -o, --output <out.xma>   Save the result there at the end, as Save As does
      --out-dir <dir>      Folder for the script's own Save operations [default: the
                           script's folder is not used: Save is skipped without it]
      --save-time <YYYY-MM-DDTHH:MM:SS>
                           Stamp saved files with this time instead of now
      --xmalab3-compat     Reproduce the XMALab 3 faults this program fixes [default: off]
  -h, --help               Print help
";

const OPTS: [Opt; 6] = [
    Opt::value("script"),
    Opt::value("project"),
    Opt::value("output").short('o'),
    Opt::value("out-dir"),
    Opt::value("save-time"),
    Opt::flag("xmalab3-compat"),
];

/// The `# setting: NAME=VALUE` header lines of a script.
fn header_settings(path: &Path) -> Result<Vec<(String, String)>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Vec::new();
    for l in text.lines() {
        if let Some(rest) = l.trim().strip_prefix("# setting:") {
            for kv in rest.split_whitespace() {
                if let Some((k, v)) = kv.split_once('=') {
                    out.push((k.to_owned(), v.to_owned()));
                }
            }
        }
    }
    Ok(out)
}

fn status_text(s: &OpStatus) -> (&'static str, String) {
    match s {
        OpStatus::Ok(m) => ("ok", m.clone()),
        OpStatus::Rejected(m) => ("rejected", m.clone()),
        OpStatus::Invalid(m) => ("invalid", m.clone()),
        OpStatus::NotImplemented => ("not implemented", String::new()),
    }
}

pub fn run(args: &[String], io: &mut Io<'_>) -> Result<(), CliError> {
    let opts: Vec<Opt> = OPTS.iter().chain(&settings::OPTIONS).copied().collect();
    let p = parse(args, &opts)?;
    if p.has("help") {
        io.line(format_args!("{HELP}\n{}", settings::HELP));
        return Ok(());
    }
    p.at_most(0)?;
    let script = p
        .value("script")
        .ok_or_else(|| CliError::usage("--script <file> is required"))?;
    let settings = SettingsArgs::from_parsed(&p)?;
    let save_time = args::save_time(&p, false)?;
    let script_path = Path::new(script);
    let lines = read_calib_script(script_path).map_err(CliError::failed)?;
    let header = header_settings(script_path).map_err(CliError::failed)?;
    let compat = crate::compat_mode(&p);
    let out_dir = p.value("out-dir").map(PathBuf::from);
    let mut state = ReplayState {
        doc: None,
        seed: None,
        options: settings.open_options(compat),
        out_dir: out_dir.clone().unwrap_or_else(|| PathBuf::from(".")),
        wizard: WizardSettings::default(),
        engopt: Default::default(),
    };
    if let Some(project) = p.value("project") {
        let mut doc = crate::open_document(Path::new(project), &settings, compat)?;
        // Ledger A210: the load-time lines first, on stderr.
        crate::console_lines(&mut doc, io);
        doc.session_mut().wizard = state.wizard;
        state.doc = Some(doc);
    }
    io.line(format_args!(
        "{script}: {} operations (compatibility mode: {})",
        lines.len(),
        compat.describe()
    ));
    if !header.is_empty() {
        let s = apply_op(&mut state, &CalibOp::Setting(header));
        let (label, msg) = status_text(&s);
        io.line(format_args!("  header settings: {label} {msg}"));
        if let Some(doc) = state.doc.as_mut() {
            crate::console_lines(doc, io);
        }
    }
    let save_options = || {
        let mut o = SaveOptions::default();
        if let Some(t) = &save_time {
            o.time = t.clone();
        }
        o
    };
    let mut failed = 0;
    for line in &lines {
        match &line.op {
            CalibOp::Dump { name, .. } => {
                io.line(format_args!(
                    "  line {}: Dump {name}: noted (no state dump is written)",
                    line.line
                ));
                continue;
            }
            CalibOp::Save { name } => {
                let Some(doc) = state.doc.as_mut() else {
                    io.line(format_args!(
                        "  line {}: Save {name}: no project is open",
                        line.line
                    ));
                    failed += 1;
                    continue;
                };
                match &out_dir {
                    Some(dir) => {
                        let target = dir.join(name);
                        let saved = doc.save_as(&target, &save_options());
                        // Ledger A210: the save's lines, also when it fails.
                        crate::console_lines(doc, io);
                        let report = saved
                            .map_err(|e| CliError::failed(format!("{}: {e}", target.display())))?;
                        io.line(format_args!(
                            "  line {}: Save: {} ({} entries)",
                            line.line,
                            report.path.display(),
                            report.entries
                        ));
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
        let start = Instant::now();
        let s = apply_op(&mut state, &line.op);
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        let (label, msg) = status_text(&s);
        if !matches!(s, OpStatus::Ok(_)) {
            failed += 1;
        }
        let mut text = format!(
            "  line {}: {}: {label} ({ms:.0} ms)",
            line.line,
            line.op.name()
        );
        if !msg.is_empty() {
            text.push_str(&format!(": {msg}"));
        }
        io.line(text);
        // Ledger A210: the operation's cap lines on stderr, after its line.
        if let Some(doc) = state.doc.as_mut() {
            crate::console_lines(doc, io);
        }
    }
    io.line(format_args!(
        "{} operations, {failed} not ok",
        lines
            .iter()
            .filter(|l| !matches!(l.op, CalibOp::Dump { .. }))
            .count()
    ));
    if let Some(doc) = &state.doc {
        for w in doc.warnings() {
            io.line(format_args!("  warning: {w}"));
        }
    }
    if let Some(out) = p.value("output") {
        let Some(doc) = state.doc.as_mut() else {
            return Err(CliError::failed("no project to save: the script made none"));
        };
        let saved = doc.save_as(Path::new(out), &save_options());
        // Ledger A210: the save's lines, also when it fails.
        crate::console_lines(doc, io);
        let report = saved.map_err(|e| CliError::failed(format!("{out}: {e}")))?;
        io.line(format_args!(
            "saved {} ({} entries)",
            report.path.display(),
            report.entries
        ));
    }
    if let Some(doc) = state.doc.as_mut() {
        crate::console_lines(doc, io);
    }
    Ok(())
}
