//! The replay of the multi-camera optimisation and the MayaCam operations (package
//! engopt): `Optimize`, `ExportMayaCams2`, `ExportMayaCam1`, `ImportMayaCam`.
//!
//! - `Optimize`: [`xma_engine::CalibrationEdit::Optimize`] with the dialog's method,
//!   iterations and factor when `ShowAdvancedCalibration` is on (the run's
//!   `setting_overrides`), else method 0 with the iterations of the script's prompt (10,000
//!   by default) and factor 0.01, as `on_pushButtonOptimize_clicked` does.
//! - `ExportMayaCams2`: [`xma_engine::Export::MayaCams`]; every file is compared with the
//!   C++ file of the same name in `<out>/<dir>`, byte for byte, as an extra comparison
//!   point named after the folder.
//! - `ExportMayaCam1`: MayaCam version 1 is deferred (PLAN.md section 11): skipped, with a
//!   note, and the folder listed as not compared.
//! - `ImportMayaCam`: [`xma_engine::calibration::read_mayacam2`] (a path `@out/...` is read
//!   from the C++ run's output folder, where the C++ export wrote it; the export is
//!   compared with it), the values the frame then shows (in the XMALab 3 mode the spin
//!   boxes' rounding, ledger A148) collected into `mayacam_import.csv` and compared with the
//!   C++ file at the end of the replay, then
//!   [`xma_engine::CalibrationEdit::ApplyExternal`] unless `apply=false`.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use xma_engine::calibration::read_mayacam2;
use xma_engine::{CalibrationEdit, Edit, EditError, Export, ExternalCalibration};
use xma_io::cfmt::format_g17;
use xma_io::export::ExportOutput;
use xma_oracle::{Dataset, Manifest};

use super::super::edits::Point;
use super::super::{Check, Tier, export_file};
use super::{CalibRun, OpStatus, ReplayState, doc_of};

/// The operations this module replays.
pub const OPS: [&str; 4] = [
    "Optimize",
    "ExportMayaCams2",
    "ExportMayaCam1",
    "ImportMayaCam",
];

/// What the replay of these operations keeps between operations.
#[derive(Debug, Default)]
pub struct EngoptState {
    /// `ShowAdvancedCalibration` (a setting of the interface, not held by the model).
    pub show_advanced_calibration: bool,
    /// Comparison points made by an operation (exported folders), taken by the replay
    /// loop after each operation.
    pub points: Vec<Point>,
    /// The rows of `mayacam_import.csv` so far, and the line of the first import.
    pub import_rows: Vec<String>,
    pub first_import_line: Option<usize>,
    /// The script line being replayed.
    pub line: usize,
}

impl EngoptState {
    /// The state at the start of a run: `ShowAdvancedCalibration` from the run's
    /// `setting_overrides` (the runs use the registered defaults otherwise, where it is
    /// off).
    #[must_use]
    pub fn for_run(run: &CalibRun) -> EngoptState {
        let show = Manifest::try_read(run.output.join("manifest.json"))
            .ok()
            .and_then(|m| {
                m.json()
                    .get("setting_overrides")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str())
                            .any(|s| s == "ShowAdvancedCalibration=true")
                    })
            })
            .unwrap_or(false);
        EngoptState {
            show_advanced_calibration: show,
            ..EngoptState::default()
        }
    }
}

fn field<'a>(fields: &'a [(String, String)], k: &str) -> Option<&'a str> {
    fields.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str())
}

fn num<T: std::str::FromStr>(fields: &[(String, String)], k: &str) -> Result<Option<T>, OpStatus> {
    field(fields, k)
        .map(|v| {
            v.parse()
                .map_err(|_| OpStatus::Invalid(format!("field {k} is not a number")))
        })
        .transpose()
}

fn edit_status(r: Result<xma_engine::EditOutcome, EditError>) -> OpStatus {
    match r {
        Ok(o) => OpStatus::Ok(o.messages.join(" | ")),
        Err(EditError::Rejected(m)) => OpStatus::Rejected(m),
        Err(e) => OpStatus::Invalid(e.to_string()),
    }
}

/// Applies one of [`OPS`].
pub fn apply(state: &mut ReplayState, op: &str, fields: &[(String, String)]) -> OpStatus {
    let r = match op {
        "Optimize" => op_optimize(state, fields),
        "ExportMayaCams2" => op_export_mayacams2(state, fields),
        "ExportMayaCam1" => op_export_mayacam1(state, fields),
        "ImportMayaCam" => op_import_mayacam(state, fields),
        _ => Ok(OpStatus::NotImplemented),
    };
    r.unwrap_or_else(|e| e)
}

/// `Optimize`.
fn op_optimize(state: &mut ReplayState, f: &[(String, String)]) -> Result<OpStatus, OpStatus> {
    let iterations = num::<usize>(f, "iterations")?.unwrap_or(10_000);
    // `ShowAdvancedCalibration`: from the run's settings, or a `Setting` line of the script.
    let advanced = state.engopt.show_advanced_calibration || state.wizard.show_advanced_calibration;
    let (method, initial) = if advanced {
        (
            num::<i32>(f, "method")?.unwrap_or(0),
            num::<f64>(f, "initial")?.unwrap_or(0.01),
        )
    } else {
        // QInputDialog::getInt, then MultiCameraCalibration(0, nbIterations, 0.01).
        (0, 0.01)
    };
    let doc = doc_of(state)?;
    Ok(edit_status(doc.apply(&Edit::Calibration(
        CalibrationEdit::Optimize {
            method,
            iterations,
            initial,
        },
    ))))
}

/// `ExportMayaCams2`: the engine's export, compared with the C++ files.
fn op_export_mayacams2(
    state: &mut ReplayState,
    f: &[(String, String)],
) -> Result<OpStatus, OpStatus> {
    let dir = field(f, "dir")
        .ok_or_else(|| OpStatus::Invalid("missing field dir".to_owned()))?
        .to_owned();
    let all = field(f, "all").is_none_or(|v| v == "true" || v == "1");
    let line = state.engopt.line;
    let out = state.out_dir.join(&dir);
    let doc = doc_of(state)?;
    let files = match doc.export(doc.project().active_trial, &Export::MayaCams { all }) {
        Ok(ExportOutput::Folder(files)) => files,
        Ok(ExportOutput::File(_)) => return Err(OpStatus::Invalid("one file".to_owned())),
        Err(e) => return Ok(OpStatus::Rejected(e.to_string())),
    };
    let ds = Dataset::from_paths(&dir, doc.path().unwrap_or(Path::new("")), &out);
    let mut items = Vec::new();
    let mut names = Vec::new();
    for file in &files {
        names.push(file.name.clone());
        if ds.has(&file.name) {
            items.push(export_file(&ds, &file.name, &file.contents, Tier::Read));
        } else {
            let mut c = Check::new(file.name.clone(), Tier::Read);
            c.problem("written here, not by the C++");
            items.push(c.finish());
        }
    }
    for rel in files_in(&out) {
        if !names.contains(&rel) {
            let mut c = Check::new(rel, Tier::Read);
            c.problem("written by the C++, not here");
            items.push(c.finish());
        }
    }
    state.engopt.points.push(Point {
        name: dir,
        line,
        items,
        skipped: None,
    });
    Ok(OpStatus::Ok(format!("{} files", files.len())))
}

fn files_in(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.path().is_file())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

/// `ExportMayaCam1`: deferred.
fn op_export_mayacam1(
    state: &mut ReplayState,
    f: &[(String, String)],
) -> Result<OpStatus, OpStatus> {
    let dir = field(f, "dir").unwrap_or("mayacam1").to_owned();
    let why = "MayaCam version 1 is deferred from version 1 (PLAN.md section 11): not exported, \
               not compared";
    state.engopt.points.push(Point {
        name: dir,
        line: state.engopt.line,
        items: Vec::new(),
        skipped: Some(why.to_owned()),
    });
    Ok(OpStatus::Ok(format!("skipped: {why}")))
}

/// The path of an `ImportMayaCam`: `@out/...` relative to the run's output folder.
fn import_path(state: &ReplayState, p: &str) -> PathBuf {
    match p.strip_prefix("@out/") {
        Some(rest) => state.out_dir.join(rest),
        None => PathBuf::from(p),
    }
}

/// The frame's controls after a load, as `mayacam_import.csv` writes them.
fn import_rows(line: usize, camera: usize, v: &ExternalCalibration) -> Vec<String> {
    let mut rows = Vec::new();
    let mut push = |control: &str, value: String| {
        rows.push(format!("{line},{camera},{control},{value}"));
    };
    push("spinBox_width", v.width.to_string());
    push("spinBox_height", v.height.to_string());
    for (name, m) in [("K", &v.camera_matrix), ("R", &v.rotation)] {
        for (i, row) in m.iter().enumerate() {
            for (j, x) in row.iter().enumerate() {
                push(
                    &format!("doubleSpinBox_{name}{}{}", i + 1, j + 1),
                    format_g17(*x),
                );
            }
        }
    }
    for (n, x) in ["TX", "TY", "TZ"].iter().zip(v.translation) {
        push(&format!("doubleSpinBox_{n}"), format_g17(x));
    }
    push(
        "checkBox_hasDistortion",
        if v.distortion.is_some() { "1" } else { "0" }.to_owned(),
    );
    let d = v.distortion.unwrap_or([0.0; 8]);
    for (n, x) in ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]
        .iter()
        .zip(d)
    {
        push(&format!("doubleSpinBox_{n}"), format_g17(x));
    }
    rows
}

/// `ImportMayaCam`.
fn op_import_mayacam(
    state: &mut ReplayState,
    f: &[(String, String)],
) -> Result<OpStatus, OpStatus> {
    let camera = num::<usize>(f, "camera")?
        .ok_or_else(|| OpStatus::Invalid("missing field camera".to_owned()))?;
    let path = import_path(
        state,
        field(f, "path").ok_or_else(|| OpStatus::Invalid("missing field path".to_owned()))?,
    );
    let apply = field(f, "apply").is_none_or(|v| v == "true" || v == "1");
    let line = state.engopt.line;
    let compat = state.options.compat;
    let read = read_mayacam2(&path).map_err(OpStatus::Rejected)?;
    // What the frame's controls hold: in XMALab 3 the spin boxes round (ledger A148).
    let shown = if compat.external_calibration_spin_box_rounding {
        read.as_spin_boxes()
    } else {
        read
    };
    state
        .engopt
        .import_rows
        .extend(import_rows(line, camera, &shown));
    state.engopt.first_import_line.get_or_insert(line);
    if !apply {
        return Ok(OpStatus::Ok("loaded, not applied".to_owned()));
    }
    let doc = doc_of(state)?;
    Ok(edit_status(doc.apply(&Edit::Calibration(
        CalibrationEdit::ApplyExternal {
            camera,
            calibration: Box::new(shown),
        },
    ))))
}

/// At the end of a replay: `mayacam_import.csv` compared, when the run has one or the
/// replay wrote rows.
pub fn finish(state: &mut ReplayState, run: &CalibRun) -> Option<Point> {
    let has_ref = run.output.join("mayacam_import.csv").is_file();
    if !has_ref && state.engopt.import_rows.is_empty() {
        return None;
    }
    let mut text = String::from("line,camera,control,value\n");
    for r in &state.engopt.import_rows {
        let _ = writeln!(text, "{r}");
    }
    let ds = Dataset::from_paths("mayacam_import", Path::new(""), &run.output);
    let item = if has_ref {
        export_file(&ds, "mayacam_import.csv", &text, Tier::Read)
    } else {
        let mut c = Check::new("mayacam_import.csv", Tier::Read);
        c.problem("written here, not by the C++");
        c.finish()
    };
    Some(Point {
        name: "mayacam_import.csv".to_owned(),
        line: state.engopt.first_import_line.unwrap_or(0),
        items: vec![item],
        skipped: None,
    })
}
