//! The calibration reference: scripts run through XMALab's own undistortion, calibration
//! and project-creation code (`oracle/calibration_runs.json`, `oracle/FORMAT.md`
//! "Calibration runs", format version 5).
//!
//! [`run_calibration_parity`] replays a run's script in the XMALab 3 mode: one function per
//! operation ([`apply_op`]) on a [`ReplayState`] (the document, and the session values the
//! script sets that the document does not hold: the seed, the settings). At every `Dump` the
//! document is compared with the C++ dump: the state dump's tables
//! ([`super::edits::compare_state`]) and the calibration state written by
//! [`super::calib_dump::calibration_state_tables`], each file against the C++ file, byte
//! for byte and then field by field. At every `Save` the project is saved as the driver
//! saves it and compared entry by entry ([`super::edits::compare_save`]).
//!
//! The undistortion and cube-calibration operations are applied by
//! [`super::calib_ops`]; the optimisation and the MayaCam operations by [`engopt`] (the
//! exported files and `mayacam_import.csv` compared as extra points). An operation that is
//! not built (MayaCam version 1, deferred) is skipped with a note.
//!
//! A table the two C++ repeats of a run do not agree on (`calibration_runs.json`,
//! `repeatability`; `scheduler.csv` after a new trial, whose queue depends on timing) is
//! compared with each repeat; the report says which one the engine matches, and the item
//! is "different" only when it matches none.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use xma_engine::create::{CalibrationTarget, NewCameraSpec, NewProjectSpec, NewTrialSpec};
use xma_engine::edit::script::percent_decode;
use xma_engine::{
    Document, Edit, EditError, NoCalibrationSpec, OpenOptions, WizardSettings, Workspace,
};
use xma_model::{CppCompat, SettingKind, SettingValue, SettingsPatch, setting_kind};
use xma_oracle::{Dataset, JsonValue, Manifest};

use super::edits::{OpRow, Point, SizeRules, compare_save, compare_state, loaded_uninit};
use super::{Check, Item, Tier, Verdict, cell, export_file, num};

pub mod engopt;
pub mod images;

/// One run of `oracle/calibration_runs.json`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CalibRun {
    pub name: String,
    /// The project the run starts from, if any.
    pub start_project: Option<PathBuf>,
    pub script: PathBuf,
    /// The run's output folder (repeat `a`).
    pub output: PathBuf,
    /// The other repeats' folders (`b`).
    pub repeats: Vec<PathBuf>,
    /// Files (relative to the output folder) that differ between the repeats.
    pub unstable: Vec<String>,
    /// A checkerboard run (`checker_runs`): compared within the tolerances of
    /// [`super::checker_parity`] instead of ledger A37's.
    pub checker: bool,
}

impl CalibRun {
    /// True when the script, the output folder and the start project (if any) exist.
    #[must_use]
    pub fn is_present(&self) -> bool {
        self.script.is_file()
            && self.output.is_dir()
            && self.start_project.as_ref().is_none_or(|p| p.is_file())
    }
}

/// The runs listed in `oracle/calibration_runs.json`.
///
/// # Errors
/// When the file cannot be read or does not have the expected layout.
pub fn calibration_runs() -> Result<Vec<CalibRun>, String> {
    let root = xma_oracle::oracle_root();
    let path = root.join("calibration_runs.json");
    let manifest = Manifest::try_read(&path).map_err(|e| e.to_string())?;
    let mut runs: Vec<(JsonValue, bool)> = manifest
        .json()
        .get("runs")
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("{}: no 'runs' list", path.display()))?
        .iter()
        .map(|r| (r.clone(), false))
        .collect();
    // The checkerboard runs (format 6), listed under `checker_runs`.
    if let Some(more) = manifest
        .json()
        .get("checker_runs")
        .and_then(|v| v.get("runs"))
        .and_then(|v| v.as_array())
    {
        runs.extend(more.iter().map(|r| (r.clone(), true)));
    }
    let field = |s: &JsonValue, k: &str| -> Result<String, String> {
        s.get(k)
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .ok_or_else(|| format!("{}: a run without '{k}'", path.display()))
    };
    runs.iter()
        .map(|(r, checker)| {
            let output = root.join(field(r, "output")?);
            let parent = output.parent().map(Path::to_path_buf).unwrap_or_default();
            let repeats = r
                .get("repeats")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str())
                        .filter(|x| *x != "a")
                        .map(|x| parent.join(x))
                        .collect()
                })
                .unwrap_or_default();
            let unstable = r
                .get("repeatability")
                .and_then(|v| v.get("differences"))
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|d| d.get("file").and_then(|f| f.as_str()))
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default();
            Ok(CalibRun {
                name: field(r, "name")?,
                start_project: r
                    .get("start_project")
                    .and_then(|v| v.as_str())
                    .map(|p| root.join(p)),
                script: root.join(field(r, "script")?),
                output,
                repeats,
                unstable,
                checker: *checker,
            })
        })
        .collect()
}

/// The run of that name, if `oracle/calibration_runs.json` lists it.
#[must_use]
pub fn find_calibration_run(name: &str) -> Option<CalibRun> {
    calibration_runs()
        .ok()?
        .into_iter()
        .find(|r| r.name == name)
}

// ---------------------------------------------------------------------------------------
// Script
// ---------------------------------------------------------------------------------------

/// One operation of a calibration script (`oracle/FORMAT.md`, "Calibration runs").
#[derive(Clone, Debug, PartialEq)]
pub enum CalibOp {
    NewProject {
        grids: Vec<Option<PathBuf>>,
        images: Vec<Vec<PathBuf>>,
        framespec: PathBuf,
        refs: PathBuf,
    },
    /// `NewProject` with `target=checkerboard` (format 6): `horizontal`, `vertical`,
    /// `square_size` instead of `framespec` and `refs`; per camera `light<c>`, `flip<c>`
    /// (`super::checker_ops::new_project`).
    NewProjectCheckerboard(super::checker_ops::NewBoard),
    NewProjectExternal {
        cameras: usize,
    },
    NewTrial {
        cams: Vec<PathBuf>,
        name: Option<String>,
    },
    NewTrialNoCalibration {
        cams: Vec<PathBuf>,
    },
    LoadWithDifferentCalibration {
        calibration: PathBuf,
        trials: PathBuf,
    },
    Workspace(Workspace),
    SelectCalibrationImage {
        camera: usize,
        image: usize,
    },
    Seed(u32),
    /// `NAME=VALUE` in the order written.
    Setting(Vec<(String, String)>),
    Dump {
        name: String,
        model: bool,
    },
    Save {
        name: String,
    },
    Wait {
        ms: u64,
    },
    /// An operation of the next package, with its fields as written.
    Later {
        op: String,
        fields: Vec<(String, String)>,
    },
}

/// The operations parsed but not yet applied.
pub const LATER_OPS: [&str; 20] = [
    "UndistortionButton",
    "UndistToggleOutlier",
    "UndistSetCenter",
    "UndistRemoveOutliers",
    "ClickReference",
    "CalibMovePoint",
    "CalibToggleInlier",
    "CalibrationButton",
    "ManualMode",
    "ManualCheckbox",
    "CalibResetFrame",
    "CalibResetCamera",
    "CalibDeleteFrame",
    "CalibDistortion",
    "SelectTrial",
    "SettingText",
    "Optimize",
    "ExportMayaCams2",
    "ExportMayaCam1",
    "ImportMayaCam",
];

impl CalibOp {
    /// The operation's name as the script writes it.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            CalibOp::NewProject { .. } | CalibOp::NewProjectCheckerboard { .. } => "NewProject",
            CalibOp::NewProjectExternal { .. } => "NewProjectExternal",
            CalibOp::NewTrial { .. } => "NewTrial",
            CalibOp::NewTrialNoCalibration { .. } => "NewTrialNoCalibration",
            CalibOp::LoadWithDifferentCalibration { .. } => "LoadWithDifferentCalibration",
            CalibOp::Workspace(_) => "Workspace",
            CalibOp::SelectCalibrationImage { .. } => "SelectCalibrationImage",
            CalibOp::Seed(_) => "Seed",
            CalibOp::Setting(_) => "Setting",
            CalibOp::Dump { .. } => "Dump",
            CalibOp::Save { .. } => "Save",
            CalibOp::Wait { .. } => "Wait",
            CalibOp::Later { op, .. } => op,
        }
    }
}

/// One line of a script that holds an operation.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibLine {
    /// The line number, from 1.
    pub line: usize,
    pub op: CalibOp,
}

/// The fields of one line, decoded.
struct Fields {
    list: Vec<(String, String)>,
    base: PathBuf,
}

impl Fields {
    fn get(&self, k: &str) -> Option<&str> {
        self.list
            .iter()
            .find(|(n, _)| n == k)
            .map(|(_, v)| v.as_str())
    }
    fn s(&self, k: &str) -> Result<String, String> {
        self.get(k)
            .map(str::to_owned)
            .ok_or_else(|| format!("missing field {k}"))
    }
    fn n<T: std::str::FromStr>(&self, k: &str) -> Result<T, String> {
        self.s(k)?
            .parse()
            .map_err(|_| format!("field {k} is not a number"))
    }
    fn path_of(&self, p: &str) -> PathBuf {
        let p = PathBuf::from(p);
        if p.is_absolute() {
            p
        } else {
            self.base.join(p)
        }
    }
    fn path(&self, k: &str) -> Result<PathBuf, String> {
        Ok(self.path_of(&self.s(k)?))
    }
    fn bool(&self, k: &str) -> Result<bool, String> {
        match self.s(k)?.as_str() {
            "true" | "1" => Ok(true),
            "false" | "0" => Ok(false),
            _ => Err(format!("field {k} is not true or false")),
        }
    }
}

/// Parses one line (without the comment and blank-line handling).
///
/// # Errors
/// An unknown operation, a token without `=`, a missing or malformed field.
pub fn parse_calib_line(text: &str, base: &Path) -> Result<CalibOp, String> {
    let mut tokens = text.split(' ').filter(|t| !t.is_empty());
    let name = tokens.next().ok_or("empty line")?.to_owned();
    let mut list = Vec::new();
    for t in tokens {
        let (k, v) = t
            .split_once('=')
            .filter(|(k, _)| !k.is_empty())
            .ok_or_else(|| format!("token without '=': {t}"))?;
        list.push((k.to_owned(), percent_decode(v)?));
    }
    let f = Fields {
        list,
        base: base.to_path_buf(),
    };
    let cameras = |f: &Fields| f.n::<usize>("cameras");
    let cams = |f: &Fields, n: usize| -> Result<Vec<PathBuf>, String> {
        (0..n)
            .map(|c| match f.get(&format!("cam{c}dir")) {
                Some(d) => Ok(f.path_of(d)),
                None => f.path(&format!("cam{c}")),
            })
            .collect()
    };
    Ok(match name.as_str() {
        "NewProject" => {
            let n = cameras(&f)?;
            let mut grids = Vec::new();
            let mut images = Vec::new();
            for c in 0..n {
                grids.push(f.get(&format!("grid{c}")).map(|g| f.path_of(g)));
                images.push(
                    f.s(&format!("images{c}"))?
                        .split(',')
                        .map(|i| f.path_of(i))
                        .collect(),
                );
            }
            if f.get("target") == Some("checkerboard") {
                CalibOp::NewProjectCheckerboard(super::checker_ops::NewBoard::parse(
                    grids,
                    images,
                    |k| f.get(k).map(str::to_owned),
                )?)
            } else {
                CalibOp::NewProject {
                    grids,
                    images,
                    framespec: f.path("framespec")?,
                    refs: f.path("refs")?,
                }
            }
        }
        "NewProjectExternal" => CalibOp::NewProjectExternal {
            cameras: cameras(&f)?,
        },
        "NewTrial" => {
            // `cam<c>`, or `cam<c>dir` (a folder, format 6).
            let n = f
                .list
                .iter()
                .filter(|(k, _)| {
                    k.strip_prefix("cam")
                        .map(|r| r.strip_suffix("dir").unwrap_or(r))
                        .is_some_and(|r| r.parse::<usize>().is_ok())
                })
                .count();
            CalibOp::NewTrial {
                cams: cams(&f, n)?,
                name: f.get("name").map(str::to_owned),
            }
        }
        "NewTrialNoCalibration" => CalibOp::NewTrialNoCalibration {
            cams: cams(&f, cameras(&f)?)?,
        },
        "LoadWithDifferentCalibration" => CalibOp::LoadWithDifferentCalibration {
            calibration: f.path("calibration")?,
            trials: f.path("trials")?,
        },
        "Workspace" => CalibOp::Workspace(match f.s("name")?.as_str() {
            "undistortion" => Workspace::Undistortion,
            "calibration" => Workspace::Calibration,
            "tracking" => Workspace::Tracking,
            other => return Err(format!("unknown workspace {other}")),
        }),
        "SelectCalibrationImage" => CalibOp::SelectCalibrationImage {
            camera: f.n("camera")?,
            image: f.n("image")?,
        },
        "Seed" => CalibOp::Seed(f.n("value")?),
        "Setting" => CalibOp::Setting(f.list.clone()),
        "Dump" => CalibOp::Dump {
            name: f.s("name")?,
            model: f.get("model").is_none() || f.bool("model")?,
        },
        "Save" => CalibOp::Save { name: f.s("name")? },
        "Wait" => CalibOp::Wait { ms: f.n("ms")? },
        other if LATER_OPS.contains(&other) => CalibOp::Later {
            op: other.to_owned(),
            fields: f.list,
        },
        other => return Err(format!("unknown operation {other}")),
    })
}

/// Reads a calibration script: one operation per line; empty lines and `#` lines skipped;
/// paths relative to the script's folder.
///
/// # Errors
/// The first line that does not parse, with its number.
pub fn read_calib_script(path: &Path) -> Result<Vec<CalibLine>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let base = path.parent().unwrap_or(Path::new("."));
    let mut out = Vec::new();
    for (i, l) in text.lines().enumerate() {
        let l = l.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        let op = parse_calib_line(l, base)
            .map_err(|e| format!("{}: line {}: {e}", path.display(), i + 1))?;
        out.push(CalibLine { line: i + 1, op });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------------------

/// What the script has set so far that the document does not hold.
#[derive(Debug)]
pub struct ReplayState {
    /// The open project; `None` before the first operation that makes one.
    pub doc: Option<Document>,
    /// `Seed`: the value given to `srand` (the C library generator `CubeCalibration` draws
    /// from); `None` until set.
    pub seed: Option<u32>,
    /// How documents are opened and created: XMALab 3 mode, the run's settings, and the
    /// settings `Setting` operations changed.
    pub options: OpenOptions,
    /// The run's output folder (`@out/` paths of `ImportMayaCam`).
    pub out_dir: PathBuf,
    /// The replay of the optimisation and the MayaCam operations ([`engopt`]).
    pub engopt: engopt::EngoptState,
    /// The interface settings `Setting` lines set (`AutoCalibAfterReference`, ...), given to
    /// every document the script opens or creates.
    pub wizard: WizardSettings,
}

/// The outcome of one operation.
#[derive(Clone, Debug, PartialEq)]
pub enum OpStatus {
    Ok(String),
    Rejected(String),
    Invalid(String),
    NotImplemented,
}

impl OpStatus {
    fn label(&self) -> &'static str {
        match self {
            OpStatus::Ok(_) => "ok",
            OpStatus::Rejected(_) => "rejected",
            OpStatus::Invalid(_) => "invalid",
            OpStatus::NotImplemented => "not implemented",
        }
    }
    fn message(&self) -> &str {
        match self {
            OpStatus::Ok(m) | OpStatus::Rejected(m) | OpStatus::Invalid(m) => m,
            OpStatus::NotImplemented => "",
        }
    }
}

/// A refusal here agrees with the C++ outcome `cpp_status`, `cpp_messages` (`ops_log.csv`).
/// The C++ log must hold the same text as the whole of one of its messages (the messages
/// are joined by `" | "`), anchored as the edit runs anchor theirs: under "rejected"
/// (`rejected: <text>`), or under "ok" for a refusal XMALab shows in a dialog
/// (`dialog: <text>` or `dialog: <dialog class>: <text>`, as `oracle/cpp/dump-mode.patch`
/// notes it, also when logged as "ok: dialog: ..."). An empty text agrees with nothing.
pub(super) fn rejection_agrees(cpp_status: &str, cpp_messages: &str, m: &str) -> bool {
    if !(cpp_status == "rejected" || cpp_status == "ok") || m.is_empty() {
        return false;
    }
    let m = super::dump_text(m);
    cpp_messages.split(" | ").any(|msg| {
        let msg = msg.strip_prefix("ok: ").unwrap_or(msg);
        if msg.strip_prefix("rejected: ") == Some(m.as_str()) {
            return true;
        }
        let Some(d) = msg.strip_prefix("dialog: ") else {
            return false;
        };
        d == m
            || d.split_once(": ").is_some_and(|(class, text)| {
                !class.is_empty() && !class.contains(char::is_whitespace) && text == m
            })
    })
}

pub(super) fn doc_of(state: &mut ReplayState) -> Result<&mut Document, OpStatus> {
    state
        .doc
        .as_mut()
        .ok_or_else(|| OpStatus::Invalid("no project is open".to_owned()))
}

pub(super) fn created(
    state: &mut ReplayState,
    r: Result<Document, xma_engine::CreateError>,
) -> OpStatus {
    match r {
        Ok(mut doc) => {
            let warn = doc.warnings().join(" | ");
            doc.session_mut().wizard = state.wizard;
            state.doc = Some(doc);
            OpStatus::Ok(warn)
        }
        Err(xma_engine::CreateError::Incomplete(m) | xma_engine::CreateError::Refused(m)) => {
            OpStatus::Rejected(m)
        }
        Err(e) => OpStatus::Invalid(e.to_string()),
    }
}

/// `NewProject`: File > New dataset with the script's files, cameras "Camera 1", ...,
/// X-ray cameras, not flipped, cube target.
fn op_new_project(
    state: &mut ReplayState,
    grids: &[Option<PathBuf>],
    images: &[Vec<PathBuf>],
    framespec: &Path,
    refs: &Path,
) -> OpStatus {
    let spec = NewProjectSpec {
        cameras: images
            .iter()
            .zip(grids)
            .enumerate()
            .map(|(c, (im, g))| {
                let mut im = im.clone();
                // CameraBox::addCalibrationImage sorts the list (`QStringList::sort`).
                im.sort_by_key(|p| p.to_string_lossy().into_owned());
                NewCameraSpec {
                    name: format!("Camera {}", c + 1),
                    images: im,
                    light_camera: false,
                    grid: g.clone(),
                    flipped: false,
                }
            })
            .collect(),
        target: CalibrationTarget::Cube {
            frame_specification: framespec.to_path_buf(),
            references: refs.to_path_buf(),
        },
    };
    let r = Document::new_project(&spec, &state.options);
    created(state, r)
}

fn op_new_project_external(state: &mut ReplayState, cameras: usize) -> OpStatus {
    let r = Document::new_external(cameras, &state.options);
    created(state, r)
}

fn op_new_trial(state: &mut ReplayState, cams: &[PathBuf], name: Option<&str>) -> OpStatus {
    let files = super::checker_ops::trial_files(cams, state.options.compat);
    let doc = match doc_of(state) {
        Ok(d) => d,
        Err(e) => return e,
    };
    let spec = NewTrialSpec {
        name: name.map_or_else(
            || format!("Trial {}", doc.project().trials.len() + 1),
            str::to_owned,
        ),
        cameras: files,
    };
    match doc.apply(&Edit::NewTrial(spec)) {
        Ok(_) => OpStatus::Ok(String::new()),
        Err(EditError::Rejected(m)) => OpStatus::Rejected(m),
        Err(e) => OpStatus::Invalid(e.to_string()),
    }
}

fn op_new_trial_no_calibration(state: &mut ReplayState, cams: &[PathBuf]) -> OpStatus {
    let spec = NoCalibrationSpec {
        camera_names: (0..cams.len())
            .map(|c| format!("Camera {}", c + 1))
            .collect(),
        trial: NewTrialSpec {
            name: "Trial 1".to_owned(),
            cameras: super::checker_ops::trial_files(cams, state.options.compat),
        },
    };
    let r = Document::new_without_calibration(&spec, &state.options);
    created(state, r)
}

fn op_load_with_different_calibration(
    state: &mut ReplayState,
    calibration: &Path,
    trials: &Path,
) -> OpStatus {
    let r = Document::open_with_different_calibration(calibration, trials, &state.options);
    created(state, r)
}

fn op_workspace(state: &mut ReplayState, ws: Workspace) -> OpStatus {
    match doc_of(state) {
        Ok(doc) => {
            doc.set_workspace(ws, true);
            OpStatus::Ok(String::new())
        }
        Err(e) => e,
    }
}

fn op_select_calibration_image(state: &mut ReplayState, camera: usize, image: usize) -> OpStatus {
    let doc = match doc_of(state) {
        Ok(d) => d,
        Err(e) => return e,
    };
    let Some(cam) = doc.project().cameras.get(camera) else {
        return OpStatus::Invalid(format!("camera {camera} out of range"));
    };
    if image >= cam.calibration_images.len() {
        return OpStatus::Invalid(format!("calibration image {image} out of range"));
    }
    let manual = doc.settings().disable_checkerboard_detection;
    let s = doc.session_mut();
    // activeCameraChanged / activeFrameCalibrationChanged: the checkerboard's corner mode
    // from the setting.
    if s.active_camera != camera || s.active_frame_calibration != image {
        s.checkerboard_manual = manual;
    }
    s.active_camera = camera;
    s.active_frame_calibration = image;
    OpStatus::Ok(String::new())
}

fn op_seed(state: &mut ReplayState, value: u32) -> OpStatus {
    state.seed = Some(value);
    if let Some(doc) = state.doc.as_mut() {
        doc.session_mut().rng = xma_vision::crand::CRand::new(value);
    }
    OpStatus::Ok(format!("srand({value})"))
}

/// `Setting`: as the driver's `applySetting` (`Settings::set`, nothing recomputed): for a
/// setting the model holds, its value; others are noted.
fn op_setting(state: &mut ReplayState, values: &[(String, String)]) -> OpStatus {
    let mut patch = SettingsPatch::default();
    let mut notes = Vec::new();
    for (name, v) in values {
        let value = match setting_kind(name) {
            Some(SettingKind::Bool) => match v.as_str() {
                "true" => SettingValue::Bool(true),
                "false" => SettingValue::Bool(false),
                _ => return OpStatus::Invalid(format!("{name}: '{v}' is not true or false")),
            },
            Some(SettingKind::Int) => match v.parse() {
                Ok(i) => SettingValue::Int(i),
                Err(_) => return OpStatus::Invalid(format!("{name}: '{v}' is not an integer")),
            },
            Some(SettingKind::Float) => match v.parse::<f32>() {
                // The driver writes floats through `QString::toFloat`.
                Ok(x) => SettingValue::Float(f64::from(x)),
                Err(_) => return OpStatus::Invalid(format!("{name}: '{v}' is not a number")),
            },
            None => {
                let on = match v.as_str() {
                    "true" => Some(true),
                    "false" => Some(false),
                    _ => None,
                };
                match on {
                    Some(on) if state.wizard.set(name, on) => {
                        if let Some(doc) = state.doc.as_mut() {
                            doc.session_mut().wizard.set(name, on);
                        }
                    }
                    _ => notes.push(format!("{name} not held by the model")),
                }
                continue;
            }
        };
        patch.set(name.clone(), value);
    }
    for (n, v) in &patch.values {
        state.options.overrides.set(n.clone(), *v);
    }
    if let Some(doc) = state.doc.as_mut()
        && let Err(e) = patch.apply(doc.settings_mut())
    {
        return OpStatus::Invalid(e);
    }
    OpStatus::Ok(notes.join(" | "))
}

/// Applies one operation that is not a `Dump` or `Save` (which the comparison handles).
pub fn apply_op(state: &mut ReplayState, op: &CalibOp) -> OpStatus {
    match op {
        CalibOp::NewProject {
            grids,
            images,
            framespec,
            refs,
        } => op_new_project(state, grids, images, framespec, refs),
        CalibOp::NewProjectCheckerboard(b) => super::checker_ops::new_project(state, b),
        CalibOp::NewProjectExternal { cameras } => op_new_project_external(state, *cameras),
        CalibOp::NewTrial { cams, name } => op_new_trial(state, cams, name.as_deref()),
        CalibOp::NewTrialNoCalibration { cams } => op_new_trial_no_calibration(state, cams),
        CalibOp::LoadWithDifferentCalibration {
            calibration,
            trials,
        } => op_load_with_different_calibration(state, calibration, trials),
        CalibOp::Workspace(ws) => op_workspace(state, *ws),
        CalibOp::SelectCalibrationImage { camera, image } => {
            op_select_calibration_image(state, *camera, *image)
        }
        CalibOp::Seed(v) => op_seed(state, *v),
        CalibOp::Setting(values) => op_setting(state, values),
        CalibOp::Wait { .. } | CalibOp::Dump { .. } | CalibOp::Save { .. } => {
            OpStatus::Ok(String::new())
        }
        CalibOp::Later { op, fields } if engopt::OPS.contains(&op.as_str()) => {
            engopt::apply(state, op, fields)
        }
        CalibOp::Later { op, fields } => super::checker_ops::apply(state, op, fields)
            .or_else(|| super::calib_ops::apply(state, op, fields))
            .unwrap_or(OpStatus::NotImplemented),
    }
}

// ---------------------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------------------

const RULE_UNSTABLE: &str = "not compared: the two C++ repeats differ here (timing), and this \
     program matches neither";

/// Files of a calibration dump compared by [`compare_state`].
const STATE_FILES: [&str; 4] = [
    "state.json",
    "trials.csv",
    "trial_state.csv",
    "scheduler.csv",
];

/// The tables holding camera matrices and poses, compared within
/// [`super::TOLERANCE_A37_CUBE`].
const CUBE_TABLES: [&str; 2] = ["camera_matrix.csv", "calibration_images.csv"];

/// Every file under `dir`, relative, sorted.
fn files_under(dir: &Path) -> BTreeSet<String> {
    fn walk(base: &Path, d: &Path, out: &mut BTreeSet<String>) {
        if let Ok(rd) = std::fs::read_dir(d) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(base, &p, out);
                } else if let Ok(r) = p.strip_prefix(base) {
                    out.insert(r.to_string_lossy().replace('\\', "/"));
                }
            }
        }
    }
    let mut out = BTreeSet::new();
    walk(dir, dir, &mut out);
    out
}

/// One `Dump`: the state dump's tables and the calibration state (with the grid model's
/// tables unless the dump says `model=false`).
pub(crate) fn compare_dump(run: &CalibRun, name: &str, doc: &Document, model: bool) -> Vec<Item> {
    let dir = run.output.join(name);
    let uninit = loaded_uninit(doc);
    let mut items = compare_state(&dir, name, doc, &uninit, None, &SizeRules::default());
    let ds = Dataset::from_paths(name, doc.path().unwrap_or(Path::new("")), &dir);
    let written = super::calib_dump::calibration_state_tables(doc, model);
    let mut seen: BTreeSet<String> = STATE_FILES.iter().map(|s| (*s).to_owned()).collect();
    for (rel, text) in &written {
        seen.insert(rel.clone());
        if !ds.has(rel) {
            let mut c = Check::new(rel.clone(), Tier::Read);
            c.problem("written here, not in the dump");
            items.push(c.finish());
            continue;
        }
        let tier = if rel.contains("samples") || rel.contains("undistorted") {
            Tier::AUndistortion
        } else {
            Tier::Read
        };
        if run.checker
            && let Some(item) = super::checker_parity::adjusted_item(&ds, rel, text, doc)
        {
            items.push(item);
            continue;
        }
        if CUBE_TABLES.contains(&rel.as_str()) && !run.checker {
            items.push(super::export_file_tolerant(
                &ds,
                rel,
                text,
                tier,
                Some(super::TOLERANCE_A37_CUBE),
            ));
        } else {
            items.push(export_file(&ds, rel, text, tier));
        }
    }
    if run.checker {
        let reference = super::checker_parity::reference_state(&dir);
        super::checker_parity::tolerate(&mut items, reference.as_ref().map_err(String::as_str));
    }
    for rel in files_under(&dir) {
        if seen.contains(&rel) || rel.starts_with("trial_") {
            continue;
        }
        let mut c = Check::new(rel.clone(), Tier::Read);
        c.problem("in the dump, not written here (the next package's table)");
        items.push(c.finish());
    }
    // Files the C++ repeats do not agree on: compared with every repeat.
    for item in &mut items {
        let key = format!("{name}/{}", item.name);
        if item.verdict != Verdict::Different || !run.unstable.contains(&key) {
            continue;
        }
        let mut notes = vec![format!(
            "the C++ repeats differ in this file (calibration_runs.json, repeatability); \
             repeat a: {}",
            item.problems.join("; ")
        )];
        let mut matched = None;
        for rep in &run.repeats {
            let rdir = rep.join(name);
            let rds = Dataset::from_paths(name, doc.path().unwrap_or(Path::new("")), &rdir);
            let alt = if item.name == "scheduler.csv" {
                compare_state(&rdir, name, doc, &uninit, None, &SizeRules::default())
                    .into_iter()
                    .find(|i| i.name == "scheduler.csv")
            } else {
                written
                    .iter()
                    .find(|(r, _)| *r == item.name)
                    .map(|(r, t)| export_file(&rds, r, t, Tier::Read))
            };
            let label = rep
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            match alt {
                Some(a) if a.verdict != Verdict::Different => {
                    notes.push(format!("repeat {label}: identical"));
                    matched = Some(label);
                }
                Some(a) => notes.push(format!("repeat {label}: {}", a.problems.join("; "))),
                None => notes.push(format!("repeat {label}: not compared")),
            }
        }
        item.verdict = Verdict::IdenticalExplained;
        item.problems.clear();
        if let Some(r) = matched {
            item.rules
                .push((format!("identical to the C++ repeat {r}"), 1));
            item.explained += 1;
        } else {
            if item.name == "scheduler.csv" {
                notes.push(format!(
                    "here the queue holds {:?} (one entry per update that started no job, \
                     the jobs here finishing at once)",
                    doc.scheduler().queued
                ));
            }
            item.rules
                .push((RULE_UNSTABLE.to_owned(), item.values.max(1)));
            item.excluded += item.values.max(1);
        }
        item.notes.extend(notes);
    }
    items
}

/// The result of [`run_calibration_parity`].
#[derive(Clone, Debug)]
pub struct CalibParityReport {
    pub run: CalibRun,
    pub ops: Vec<OpRow>,
    pub points: Vec<Point>,
    /// Where the replay stopped: (line, operation), for the first operation of the next
    /// package.
    pub stopped: Option<(usize, String)>,
    pub warnings: Vec<String>,
    pub seconds: f64,
}

impl CalibParityReport {
    /// No table differs, the replay did not stop, and every operation agrees with the C++
    /// outcome. An operation nothing handles ("not implemented") fails the run: everything
    /// after it would go uncompared.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.stopped.is_none()
            && self.points.iter().all(Point::passed)
            && self.ops.iter().all(|o| o.agrees)
    }

    fn count(&self, v: Verdict) -> usize {
        self.points
            .iter()
            .flat_map(|p| &p.items)
            .filter(|i| i.verdict == v)
            .count()
    }

    /// One line.
    #[must_use]
    pub fn summary(&self) -> String {
        let compared = self.points.iter().filter(|p| p.skipped.is_none()).count();
        let items: usize = self.points.iter().map(|p| p.items.len()).sum();
        format!(
            "{}: {} — {} of {} dumps and saves compared, {} tables and files ({} identical, {} \
             identical apart from explained cells, {} empty on both sides, {} within \
             tolerance, {} different), {} values; {} of {} operations agree; {}",
            self.run.name,
            if self.passed() { "PASS" } else { "FAIL" },
            compared,
            self.points.len(),
            items,
            self.count(Verdict::Identical),
            self.count(Verdict::IdenticalExplained),
            self.count(Verdict::BothEmpty),
            self.count(Verdict::WithinTolerance),
            self.count(Verdict::Different),
            self.points
                .iter()
                .flat_map(|p| &p.items)
                .map(|i| i.values)
                .sum::<usize>(),
            self.ops.iter().filter(|o| o.agrees).count(),
            self.ops.len(),
            match &self.stopped {
                Some((l, op)) => format!(
                    "stopped at line {l} (not implemented: {op}); nothing after it compared"
                ),
                None => "script done".to_owned(),
            }
        )
    }

    /// The report as Markdown.
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "# Calibration parity report: {}\n", self.run.name);
        let _ = writeln!(s, "**Result: {}.**\n", self.summary());
        if let Some(p) = &self.run.start_project {
            let _ = writeln!(s, "- Start project: `{}`", p.display());
        }
        let _ = writeln!(s, "- Script: `{}`", self.run.script.display());
        let _ = writeln!(s, "- Reference run: `{}`", self.run.output.display());
        let _ = writeln!(s, "- Compatibility mode: {}", CppCompat::XMALAB3.describe());
        let _ = writeln!(
            s,
            "- Compared by: xma {} (engine {}), in {:.1} s\n",
            crate::VERSION,
            xma_engine::VERSION,
            self.seconds
        );
        let _ = writeln!(s, "## Operations\n");
        let _ = writeln!(
            s,
            "| Line | Operation | C++ | Here | Agree |\n|---:|---|---|---|---|"
        );
        for o in &self.ops {
            let cpp = if o.cpp_messages.is_empty() {
                o.cpp_status.clone()
            } else {
                format!("{}: {}", o.cpp_status, o.cpp_messages)
            };
            let here = if o.message.is_empty() {
                o.status.clone()
            } else {
                format!("{}: {}", o.status, o.message)
            };
            let _ = writeln!(
                s,
                "| {} | {} | {} | {} | {} |",
                o.line,
                o.op,
                cell(&cpp),
                cell(&here),
                if o.agrees {
                    "yes"
                } else if o.status == "not implemented" {
                    "**no** (not implemented or not reached)"
                } else {
                    "**no**"
                }
            );
        }
        let _ = writeln!(s, "\n## Dumps and saves\n");
        let _ = writeln!(
            s,
            "| Line | Dump or save | Tables | Identical | Explained | Empty | Different | Values |\n|---:|---|---:|---:|---:|---:|---:|---:|"
        );
        for p in &self.points {
            if let Some(why) = &p.skipped {
                let _ = writeln!(
                    s,
                    "| {} | `{}` | not reached: {} | | | | | |",
                    p.line,
                    p.name,
                    cell(why)
                );
                continue;
            }
            let n = |v: Verdict| p.items.iter().filter(|i| i.verdict == v).count();
            let _ = writeln!(
                s,
                "| {} | `{}` | {} | {} | {} | {} | {} | {} |",
                p.line,
                p.name,
                p.items.len(),
                n(Verdict::Identical),
                n(Verdict::IdenticalExplained),
                n(Verdict::BothEmpty),
                n(Verdict::Different) + n(Verdict::WithinTolerance),
                p.items.iter().map(|i| i.values).sum::<usize>()
            );
        }
        let flagged: Vec<(&Point, &Item)> = self
            .points
            .iter()
            .flat_map(|p| p.items.iter().map(move |i| (p, i)))
            .filter(|(_, i)| !matches!(i.verdict, Verdict::Identical | Verdict::BothEmpty))
            .collect();
        if !flagged.is_empty() {
            let _ = writeln!(s, "\n## Tables not plainly identical\n");
            let _ = writeln!(
                s,
                "| Dump | Table | Values | Max abs diff | Where | Verdict |\n|---|---|---:|---:|---|---|"
            );
            for (p, i) in &flagged {
                let _ = writeln!(
                    s,
                    "| `{}` | `{}` | {} | {} | {} | {} |",
                    p.name,
                    i.name,
                    i.values,
                    num(i.max_abs),
                    cell(&i.max_abs_at),
                    cell(&i.verdict_label())
                );
            }
            let _ = writeln!(s, "\n### Details\n");
            for (p, i) in &flagged {
                if i.problems.is_empty() && i.notes.is_empty() {
                    continue;
                }
                let _ = writeln!(s, "- `{}/{}`:", p.name, i.name);
                for x in i.problems.iter().chain(&i.notes) {
                    let _ = writeln!(s, "  - {}", x.replace('\n', " "));
                }
            }
        }
        let _ = writeln!(s, "\n## Warnings\n");
        if self.warnings.is_empty() {
            let _ = writeln!(s, "None.");
        } else {
            for w in &self.warnings {
                let _ = writeln!(s, "- {w}");
            }
        }
        s
    }
}

/// `ops_log.csv`: (line, op, status, messages).
pub(crate) fn read_ops_log(path: &Path) -> Vec<(usize, String, String, String)> {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    text.lines()
        .skip(1)
        .filter_map(|l| {
            let mut it = l.splitn(4, ',');
            Some((
                it.next()?.parse().ok()?,
                it.next()?.to_owned(),
                it.next()?.to_owned(),
                it.next().unwrap_or("").to_owned(),
            ))
        })
        .collect()
}

/// The options documents are opened and created with: the XMALab 3 mode and the settings
/// in force at the end of the C++ run (`manifest.json`; the recorded runs use the
/// registered defaults and change none mid-run).
///
/// A run made with `--default-settings` starts from the registered defaults, and its header
/// settings (`setting_overrides`) are applied by [`run_header_settings`]: a script that
/// changes a setting mid-run (`Setting` lines, format 6) must not start with the value it
/// ends with.
pub(crate) fn run_options(run: &CalibRun) -> Result<OpenOptions, String> {
    let manifest =
        Manifest::try_read(run.output.join("manifest.json")).map_err(|e| e.to_string())?;
    let overrides = if run_header_settings(run).is_some() {
        SettingsPatch::default()
    } else {
        crate::settings::manifest_settings(&manifest)?
    };
    Ok(OpenOptions {
        overrides,
        compat: CppCompat::XMALAB3,
        ..OpenOptions::default()
    })
}

/// For a run made with `--default-settings`: the header settings (`manifest.json`, `setting_overrides`,
/// `NAME=VALUE`), applied before the first operation as the driver applies them.
fn run_header_settings(run: &CalibRun) -> Option<Vec<(String, String)>> {
    let m = Manifest::try_read(run.output.join("manifest.json")).ok()?;
    let j = m.json();
    if j.get("default_settings").and_then(|v| v.as_bool()) != Some(true) {
        return None;
    }
    Some(
        j.get("setting_overrides")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str())
                    .filter_map(|x| x.split_once('='))
                    .map(|(k, v)| (k.to_owned(), v.to_owned()))
                    .collect()
            })
            .unwrap_or_default(),
    )
}

/// The interface settings of the run (`manifest.json`, `settings`: those
/// [`WizardSettings`] holds).
pub(crate) fn run_wizard(run: &CalibRun) -> WizardSettings {
    let mut w = WizardSettings::default();
    if let Ok(m) = Manifest::try_read(run.output.join("manifest.json"))
        && let Some(list) = m.json().get("settings").and_then(|v| v.as_array())
    {
        for e in list {
            if let (Some(n), Some(v)) = (
                e.get("name").and_then(|v| v.as_str()),
                e.get("value").and_then(|v| v.as_bool()),
            ) {
                w.set(n, v);
            }
        }
    }
    w
}

/// The replay's state before the first operation: the run's settings, the start project.
fn start_state(run: &CalibRun, compat: CppCompat) -> Result<ReplayState, String> {
    let mut options = run_options(run)?;
    options.compat = compat;
    let mut state = ReplayState {
        doc: None,
        seed: None,
        options,
        out_dir: run.output.clone(),
        engopt: engopt::EngoptState::for_run(run),
        wizard: run_wizard(run),
    };
    if let Some(header) = run_header_settings(run)
        && !header.is_empty()
    {
        let _ = op_setting(&mut state, &header);
    }
    if let Some(p) = &run.start_project {
        let mut doc =
            Document::open_with(p, &state.options).map_err(|e| format!("{}: {e}", p.display()))?;
        doc.session_mut().wizard = state.wizard;
        state.doc = Some(doc);
    }
    Ok(state)
}

/// A run's script replayed in `compat` without comparing anything; `stop_at`: the document
/// as it is when the replay reaches the `Dump` or `Save` of that name (the end of the
/// script when `None` or not found).
///
/// # Errors
/// When the run's script, manifest or start project cannot be read.
pub fn replay_document(
    run: &CalibRun,
    compat: CppCompat,
    stop_at: Option<&str>,
) -> Result<Option<Document>, String> {
    let lines = read_calib_script(&run.script)?;
    let mut state = start_state(run, compat)?;
    for line in &lines {
        if let CalibOp::Dump { name, .. } | CalibOp::Save { name } = &line.op {
            if stop_at == Some(name.as_str()) {
                break;
            }
            continue;
        }
        let _ = apply_op(&mut state, &line.op);
    }
    Ok(state.doc)
}

/// Replays one run and compares every `Dump` and `Save` reached.
///
/// # Errors
/// When the run's script, manifest or start project cannot be read.
pub fn run_calibration_parity(run: &CalibRun) -> Result<CalibParityReport, String> {
    run_calibration_parity_in(run, CppCompat::XMALAB3)
}

/// [`run_calibration_parity`] in another compatibility mode: a measurement of how far that
/// mode is from XMALab (the default mode, which fixes the faults of XMALab 3, is expected to
/// differ where a fix changes numbers).
///
/// # Errors
/// As [`run_calibration_parity`].
pub fn run_calibration_parity_in(
    run: &CalibRun,
    compat: CppCompat,
) -> Result<CalibParityReport, String> {
    let start = Instant::now();
    let lines = read_calib_script(&run.script)?;
    let mut state = start_state(run, compat)?;
    let recompute_when_saving =
        state.options.overrides.get("RecomputeWhenSaving") == Some(SettingValue::Bool(true));
    let cpp_log = read_ops_log(&run.output.join("ops_log.csv"));
    let mut ops = Vec::new();
    let mut points = Vec::new();
    let mut stopped: Option<(usize, String)> = None;
    // The C++ program's state where a saved project or an export is compared (it picks a
    // checkerboard run's tolerance): that of the last dump, as long as only operations
    // that change nothing ran since.
    let mut reference: Result<super::checker_parity::ReferenceState, String> =
        Err("no C++ dump precedes this point".to_owned());
    for line in &lines {
        match &line.op {
            CalibOp::Dump { name, .. } if run.checker => {
                reference = super::checker_parity::reference_state(&run.output.join(name));
            }
            CalibOp::Dump { .. } | CalibOp::Save { .. } | CalibOp::Wait { .. } => {}
            CalibOp::Later { op, .. } if op == "ExportMayaCams2" || op == "ExportMayaCam1" => {}
            other => {
                reference = Err(format!(
                    "line {} ({}) ran after the last C++ dump",
                    line.line,
                    other.name()
                ));
            }
        }
        match &line.op {
            CalibOp::Dump { name, .. } | CalibOp::Save { name } => {
                let is_dump = matches!(line.op, CalibOp::Dump { .. });
                let model = !matches!(line.op, CalibOp::Dump { model: false, .. });
                let skipped = match (&stopped, &state.doc) {
                    (Some((l, op)), _) => Some(format!(
                        "after line {l} ({op}), an operation of the next package"
                    )),
                    (None, None) => Some("no project is open here".to_owned()),
                    _ => None,
                };
                let items = match (&skipped, state.doc.as_mut()) {
                    (None, Some(doc)) if is_dump => compare_dump(run, name, doc, model),
                    (None, Some(doc)) => {
                        let mut items = compare_save(
                            &run.output.join(name),
                            name,
                            doc,
                            recompute_when_saving,
                            &SizeRules {
                                // A checkerboard run: every number compared, then
                                // `checker_parity::tolerate`.
                                calibration_tolerance: Some(if run.checker {
                                    super::checker_parity::NO_TOLERANCE
                                } else {
                                    super::TOLERANCE_A37_CUBE
                                }),
                                ..SizeRules::default()
                            },
                        );
                        if run.checker {
                            super::checker_parity::tolerate(
                                &mut items,
                                reference.as_ref().map_err(String::as_str),
                            );
                        }
                        items
                    }
                    _ => Vec::new(),
                };
                points.push(Point {
                    name: name.clone(),
                    line: line.line,
                    items,
                    skipped,
                });
                continue;
            }
            _ => {}
        }
        let (cpp_status, cpp_messages) = cpp_log
            .iter()
            .find(|(l, ..)| *l == line.line)
            .map(|(_, _, st, m)| (st.clone(), m.clone()))
            .unwrap_or_else(|| ("(not logged)".to_owned(), String::new()));
        if stopped.is_some() {
            ops.push(OpRow {
                line: line.line,
                op: line.op.name().to_owned(),
                cpp_status,
                cpp_messages,
                status: "not implemented".to_owned(),
                message: "not reached".to_owned(),
                agrees: false,
            });
            continue;
        }
        state.engopt.line = line.line;
        let out = apply_op(&mut state, &line.op);
        if run.checker {
            for p in &mut state.engopt.points {
                super::checker_parity::tolerate(
                    &mut p.items,
                    reference.as_ref().map_err(String::as_str),
                );
            }
        }
        points.append(&mut state.engopt.points);
        if out == OpStatus::NotImplemented {
            stopped = Some((line.line, line.op.name().to_owned()));
        }
        let agrees = match &out {
            OpStatus::Ok(_) => cpp_status == "ok",
            OpStatus::Rejected(m) => rejection_agrees(&cpp_status, &cpp_messages, m),
            OpStatus::Invalid(_) => cpp_status == "invalid",
            OpStatus::NotImplemented => false,
        };
        ops.push(OpRow {
            line: line.line,
            op: line.op.name().to_owned(),
            cpp_status,
            cpp_messages,
            status: out.label().to_owned(),
            message: if out == OpStatus::NotImplemented {
                format!("not implemented: {}", line.op.name())
            } else {
                out.message().to_owned()
            },
            agrees,
        });
    }
    points.extend(engopt::finish(&mut state, run));
    if run.checker {
        points.extend(super::checker_parity::hook_points(run));
    }
    let warnings = state
        .doc
        .as_ref()
        .map(|d| {
            let mut w: Vec<String> = d.warnings().to_vec();
            w.extend(d.extras().warnings.iter().cloned());
            w
        })
        .unwrap_or_default();
    Ok(CalibParityReport {
        run: run.clone(),
        ops,
        points,
        stopped,
        warnings,
        seconds: start.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_operation() {
        let base = Path::new("/s");
        let op = parse_calib_line(
            "NewProject cameras=2 grid0=g0.tif images0=a.tif,b.tif images1=c.tif framespec=f.csv refs=/r.ref",
            base,
        )
        .unwrap();
        assert_eq!(
            op,
            CalibOp::NewProject {
                grids: vec![Some(PathBuf::from("/s/g0.tif")), None],
                images: vec![
                    vec![PathBuf::from("/s/a.tif"), PathBuf::from("/s/b.tif")],
                    vec![PathBuf::from("/s/c.tif")]
                ],
                framespec: PathBuf::from("/s/f.csv"),
                refs: PathBuf::from("/r.ref"),
            }
        );
        assert_eq!(
            parse_calib_line("NewTrial cam0=%2Fa%20b.avi cam1=/c.avi", base).unwrap(),
            CalibOp::NewTrial {
                cams: vec![PathBuf::from("/a b.avi"), PathBuf::from("/c.avi")],
                name: None
            }
        );
        assert_eq!(
            parse_calib_line("Dump name=x model=false", base).unwrap(),
            CalibOp::Dump {
                name: "x".into(),
                model: false
            }
        );
        assert_eq!(
            parse_calib_line("Workspace name=calibration", base).unwrap(),
            CalibOp::Workspace(Workspace::Calibration)
        );
        for later in LATER_OPS {
            assert_eq!(
                parse_calib_line(&format!("{later} camera=0"), base)
                    .unwrap()
                    .name(),
                later
            );
        }
        assert!(parse_calib_line("Nonsense a=1", base).is_err());
        assert!(parse_calib_line("Seed 1", base).is_err());
    }

    fn run() -> CalibRun {
        CalibRun {
            name: "t".to_owned(),
            start_project: None,
            script: PathBuf::new(),
            output: PathBuf::new(),
            repeats: Vec::new(),
            unstable: Vec::new(),
            checker: false,
        }
    }

    fn op(status: &str, agrees: bool) -> OpRow {
        OpRow {
            line: 12,
            op: "ManualMode".to_owned(),
            cpp_status: "ok".to_owned(),
            cpp_messages: String::new(),
            status: status.to_owned(),
            message: String::new(),
            agrees,
        }
    }

    /// Review of 20 November 2026, second round, finding 3: a replay that stops at an
    /// operation nothing handles fails, naming the operation and the line.
    #[test]
    fn review12_not_implemented_fails_the_run() {
        let r = CalibParityReport {
            run: run(),
            ops: vec![op("not implemented", false)],
            points: Vec::new(),
            stopped: Some((12, "ManualMode".to_owned())),
            warnings: Vec::new(),
            seconds: 0.0,
        };
        assert!(!r.passed(), "{}", r.summary());
        assert!(
            r.summary().contains("line 12") && r.summary().contains("ManualMode"),
            "{}",
            r.summary()
        );
        // The stop alone fails too.
        let r = CalibParityReport {
            ops: Vec::new(),
            ..r
        };
        assert!(!r.passed(), "{}", r.summary());
    }

    /// Review of 20 November 2026, second round, finding 9: a refusal agrees only with a C++
    /// refusal or an "ok" whose log holds the same text, and never with an empty text.
    #[test]
    fn review12_rejection_agrees_only_with_the_same_message() {
        // XMALab logs a refusal by a dialog as "ok: dialog: ...".
        assert!(rejection_agrees(
            "ok",
            "ok: dialog: Select the four corners",
            "Select the four corners"
        ));
        assert!(rejection_agrees(
            "rejected",
            "rejected: No camera calibrated",
            "No camera calibrated"
        ));
        // A C++ refusal for another reason.
        assert!(!rejection_agrees(
            "rejected",
            "rejected: No camera calibrated",
            "Frame not found"
        ));
        // An empty text agrees with nothing.
        assert!(!rejection_agrees("ok", "ok", ""));
        assert!(!rejection_agrees("rejected", "rejected: x", ""));
        // A C++ status that is neither.
        assert!(!rejection_agrees("invalid", "invalid: x", "x"));
    }

    /// Review of 20 November 2026, third round: the refusal text is the whole text of one
    /// C++ message, not a part of one.
    #[test]
    fn review12c_rejection_agrees_with_a_whole_message_only() {
        // A part of a C++ message.
        assert!(!rejection_agrees(
            "rejected",
            "rejected: No camera calibrated",
            "camera"
        ));
        assert!(!rejection_agrees(
            "rejected",
            "rejected: No camera calibrated",
            "No camera"
        ));
        assert!(!rejection_agrees(
            "ok",
            "ok: dialog: Select the four corners",
            "four corners"
        ));
        assert!(!rejection_agrees(
            "ok",
            "dialog: xma::ErrorDialog: Could not detect chessboard. Please try the manual mode",
            "Please try the manual mode"
        ));
        // A whole message, among others.
        assert!(rejection_agrees(
            "rejected",
            "confirmed: Are you sure? | rejected: No camera calibrated",
            "No camera calibrated"
        ));
        // XMALab's dialog note names the dialog's class (oracle/cpp/dump-mode.patch).
        assert!(rejection_agrees(
            "ok",
            "dialog: xma::ErrorDialog: Could not detect chessboard. Please try the manual mode",
            "Could not detect chessboard. Please try the manual mode"
        ));
        assert!(rejection_agrees(
            "ok",
            "filled: x | dialog: xma::ErrorDialog: Could not detect chessboard. Please try \
             the manual mode | last trial",
            "Could not detect chessboard. Please try the manual mode"
        ));
    }
}
