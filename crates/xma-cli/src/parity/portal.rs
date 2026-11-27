//! The XMA Portal reference: scripts run through XMALab's own Portal import
//! (`oracle/portal_runs.json`, `oracle/FORMAT.md` "Calibration runs, version 6").
//!
//! [`run_portal_parity`] copies each run's staged files into a private working folder (the
//! `@out` of the script), opens the start project there, and replays the script in the
//! XMALab 3 mode: `NewTrialFromPortal`, `OpenProject` (a Portal project), `AddPortalMetadata`,
//! `SettingText`, `SelectTrial` here, the shared operations (`Setting`, `Dump`, `Save`)
//! with the calibration runs' code ([`super::calibration`]). It compares:
//!
//! - every `Dump` with the C++ state dump and calibration state
//!   ([`super::calibration::compare_dump`]);
//! - every `Save` with the C++ file, entry by entry ([`super::edits::compare_save`]), after
//!   the C++ run's own folder is read as this replay's folder in the C++ text entries (the
//!   absolute video paths of `<trial>/<camera>_filenames_absolute.txt`);
//! - `dialogs.csv`: what the New trial and New dataset dialogs hold after loading, and the
//!   item list and tree of Project Metadata, row by row, with the paths of the run's folder
//!   written `<out>/...` and those of the temporary extraction `<private>/tmp/XROMM_tmp/<top
//!   folder>/...`, as the driver writes them;
//! - each question asked, with its text (the confirmations of `ops_log.csv`);
//! - each folder the script lists (`# listing:`): the files there after the run and their
//!   bytes, against the C++ run's folder.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use xma_engine::create::{CalibrationTarget, NewTrialSpec};
use xma_engine::edit::script::percent_decode;
use xma_engine::portal::{
    DELETE_QUESTION, Destination, LOAD_CONFIRM, METADATA_CONFIRM, ProjectImport, assign_trial,
    import_project, import_trial, override_text, read_metadata,
};
use xma_engine::portal_summary::{MetaNode, metadata_tree};
use xma_engine::{Document, Edit, EditError};
use xma_model::CppCompat;
use xma_oracle::{JsonValue, Manifest};

use super::calibration::{
    CalibParityReport, CalibRun, OpStatus, ReplayState, apply_op, compare_dump, engopt,
    parse_calib_line, read_ops_log, run_options, run_wizard,
};
use super::edits::{OpRow, Point, SizeRules, compare_save};
use super::{Check, Item, Tier};

/// One run of `oracle/portal_runs.json`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortalRun {
    pub run: CalibRun,
    /// `src` (under `oracle/`) copied to `dst` (under the run's folder) first.
    pub staged: Vec<(PathBuf, String)>,
}

/// The runs listed in `oracle/portal_runs.json`.
///
/// # Errors
/// When the file cannot be read or does not have the expected layout.
pub fn portal_runs() -> Result<Vec<PortalRun>, String> {
    let root = xma_oracle::oracle_root();
    let path = root.join("portal_runs.json");
    let manifest = Manifest::try_read(&path).map_err(|e| e.to_string())?;
    let runs = manifest
        .json()
        .get("runs")
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("{}: no 'runs' list", path.display()))?;
    let field = |s: &JsonValue, k: &str| -> Result<String, String> {
        s.get(k)
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .ok_or_else(|| format!("{}: a run without '{k}'", path.display()))
    };
    runs.iter()
        .map(|r| {
            let output = root.join(field(r, "output")?);
            let parent = output.parent().map(Path::to_path_buf).unwrap_or_default();
            let list = |k: &str| -> Vec<String> {
                r.get(k)
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str().map(str::to_owned))
                            .collect()
                    })
                    .unwrap_or_default()
            };
            let staged = list("staged")
                .iter()
                .filter_map(|s| {
                    s.split_once(" -> ")
                        .map(|(a, b)| (root.join(a), b.to_owned()))
                })
                .collect();
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
            Ok(PortalRun {
                run: CalibRun {
                    name: field(r, "name")?,
                    start_project: None,
                    script: root.join(field(r, "script")?),
                    output,
                    repeats: list("repeats")
                        .into_iter()
                        .filter(|x| x != "a")
                        .map(|x| parent.join(x))
                        .collect(),
                    unstable,
                    checker: false,
                },
                staged,
            })
        })
        .collect()
}

/// The run of that name, if `oracle/portal_runs.json` lists it.
#[must_use]
pub fn find_portal_run(name: &str) -> Option<PortalRun> {
    portal_runs().ok()?.into_iter().find(|r| r.run.name == name)
}

// ---------------------------------------------------------------------------------------
// Script

/// One line of a Portal script.
#[derive(Clone, Debug, PartialEq)]
enum PortalOp {
    OpenProject {
        path: String,
        board: Option<(u32, u32, f64)>,
    },
    NewTrialFromPortal {
        path: String,
        delete_after: bool,
        name: Option<String>,
    },
    AddPortalMetadata {
        target: String,
        path: String,
        override_expected: bool,
    },
    SettingText(Vec<(String, String)>),
    SelectTrial(String),
    /// A line the calibration runs' code applies (`Setting`, `Dump`, `Save`, ...).
    Shared(super::calibration::CalibOp),
}

struct Line {
    line: usize,
    op: PortalOp,
    name: String,
}

fn fields(text: &str) -> Result<(String, Vec<(String, String)>), String> {
    let mut tokens = text.split(' ').filter(|t| !t.is_empty());
    let name = tokens.next().ok_or("empty line")?.to_owned();
    let mut list = Vec::new();
    for t in tokens {
        let (k, v) = t
            .split_once('=')
            .ok_or_else(|| format!("token without '=': {t}"))?;
        list.push((k.to_owned(), percent_decode(v)?));
    }
    Ok((name, list))
}

fn get<'a>(f: &'a [(String, String)], k: &str) -> Option<&'a str> {
    f.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str())
}

fn read_script(path: &Path) -> Result<(Vec<Line>, Vec<String>), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let base = path.parent().unwrap_or(Path::new("."));
    let mut out = Vec::new();
    let mut listings = Vec::new();
    for (i, l) in text.lines().enumerate() {
        let l = l.trim();
        if let Some(d) = l.strip_prefix("# listing:") {
            listings.push(d.trim().to_owned());
            continue;
        }
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        let at = |e: String| format!("{}: line {}: {e}", path.display(), i + 1);
        let (name, f) = fields(l).map_err(at)?;
        let req = |k: &str| {
            get(&f, k)
                .map(str::to_owned)
                .ok_or_else(|| at(format!("missing field {k}")))
        };
        let op = match name.as_str() {
            "OpenProject" => PortalOp::OpenProject {
                path: req("path")?,
                board: match get(&f, "horizontal") {
                    Some(h) => Some((
                        h.parse().map_err(|_| at("horizontal".into()))?,
                        req("vertical")?
                            .parse()
                            .map_err(|_| at("vertical".into()))?,
                        req("square_size")?
                            .parse()
                            .map_err(|_| at("square_size".into()))?,
                    )),
                    None => None,
                },
            },
            "NewTrialFromPortal" => PortalOp::NewTrialFromPortal {
                path: req("path")?,
                delete_after: get(&f, "delete_after") == Some("true"),
                name: get(&f, "name").map(str::to_owned),
            },
            "AddPortalMetadata" => PortalOp::AddPortalMetadata {
                target: req("target")?,
                path: req("path")?,
                override_expected: get(&f, "override") == Some("true"),
            },
            "SettingText" => PortalOp::SettingText(f.clone()),
            "SelectTrial" => PortalOp::SelectTrial(req("trial")?),
            _ => PortalOp::Shared(parse_calib_line(l, base).map_err(at)?),
        };
        out.push(Line {
            line: i + 1,
            op,
            name,
        });
    }
    Ok((out, listings))
}

// ---------------------------------------------------------------------------------------
// Replay

/// What the replay holds besides the calibration runs' state.
struct Portal {
    state: ReplayState,
    work: PathBuf,
    workspace_path: String,
    custom_workspace: bool,
    /// `dialogs.csv` rows written here: (line, dialog, index, control, value).
    dialogs: Vec<(usize, String, i64, String, String)>,
    /// The confirmations asked, per line.
    questions: Vec<(usize, Vec<String>)>,
    /// Portal projects' temporary folders, kept until the end (`<private>/...`).
    imports: Vec<ProjectImport>,
}

fn at_out(work: &Path, p: &str) -> PathBuf {
    match p.strip_prefix("@out/") {
        Some(r) => work.join(r),
        None => PathBuf::from(p),
    }
}

impl Portal {
    /// A path as the driver writes it in `dialogs.csv`.
    fn shown(&self, p: &str) -> String {
        let mut s = p.to_owned();
        for i in &self.imports {
            let f = i.folder.to_string_lossy().into_owned();
            if s.starts_with(&f) {
                s = format!("<private>/tmp/XROMM_tmp/{}{}", i.top, &s[f.len()..]);
            }
        }
        let w = self.work.to_string_lossy().into_owned();
        s.replace(&w, "<out>")
    }

    fn row(&mut self, line: usize, dialog: &str, index: i64, control: &str, value: &str) {
        let v = self.shown(value);
        self.dialogs
            .push((line, dialog.to_owned(), index, control.to_owned(), v));
    }

    fn doc(&mut self) -> Result<&mut Document, OpStatus> {
        self.state
            .doc
            .as_mut()
            .ok_or_else(|| OpStatus::Invalid("no project is open".to_owned()))
    }
}

/// `CameraBox::commonPrefix` / `CameraBoxTrial::commonPrefix`: the longest start of the
/// first name that every name contains, and one character more (`left(count + 1)`).
///
/// Ported from ui/CameraBox.cpp, `commonPrefix`.
fn common_prefix(names: &[String]) -> String {
    let Some(first) = names.first() else {
        return String::new();
    };
    let chars: Vec<char> = first.chars().collect();
    let mut count = 0;
    while count < chars.len() {
        let prefix: String = chars[..=count].iter().collect();
        if !names.iter().all(|n| n.contains(&prefix)) {
            break;
        }
        count += 1;
    }
    // `left(count + 1)`: one character past the common start, or the whole name.
    chars[..(count + 1).min(chars.len())].iter().collect()
}

fn op_new_trial(
    p: &mut Portal,
    line: usize,
    path: &str,
    delete_after: bool,
    name: Option<&str>,
) -> OpStatus {
    let file = at_out(&p.work, path);
    let compat = p.state.options.compat;
    let ws = (p.custom_workspace && !p.workspace_path.is_empty())
        .then(|| at_out(&p.work, &p.workspace_path));
    let (ids, n_cameras, project_file, default_name, names) = match p.doc() {
        Ok(d) => (
            d.project()
                .cameras
                .iter()
                .map(|c| c.portal_id)
                .collect::<Vec<i32>>(),
            d.project().cameras.len(),
            d.path().map(Path::to_path_buf),
            format!("Trial {}", d.project().trials.len() + 1),
            d.project()
                .cameras
                .iter()
                .map(|c| c.name.clone())
                .collect::<Vec<String>>(),
        ),
        Err(e) => return e,
    };
    p.questions.push((
        line,
        vec![LOAD_CONFIRM.to_owned(), DELETE_QUESTION.to_owned()],
    ));
    let dest = Destination {
        workspace: ws.as_deref(),
        project_file: project_file.as_deref(),
        portal_file: &file,
    };
    let import = match import_trial(&file, &dest, compat, &mut |_| true) {
        Ok(i) => i,
        Err(e) => return OpStatus::Rejected(e.to_string()),
    };
    if delete_after {
        let _ = std::fs::remove_file(&file);
    }
    let a = assign_trial(&import, &ids, n_cameras, false);
    let trial_name = a.trial_name.clone().unwrap_or(default_name);
    let mut boxes: Vec<(Vec<PathBuf>, String)> = vec![(Vec::new(), String::new()); n_cameras];
    for (b, files, path) in &a.cameras {
        let text = xma_engine::portal::nested_zip_text(path).unwrap_or_else(|| {
            common_prefix(
                &files
                    .iter()
                    .map(|f| f.to_string_lossy().into_owned())
                    .collect::<Vec<_>>(),
            )
        });
        boxes[*b] = (files.clone(), text);
    }
    let d = "NewTrialDialog";
    p.row(
        line,
        d,
        -1,
        "window_title",
        "Select videopaths for new trial",
    );
    p.row(line, d, -1, "trial_name", &trial_name);
    let meta = import.metadata_file.to_string_lossy().into_owned();
    p.row(line, d, -1, "xml_metadata", &meta);
    p.row(line, d, -1, "n_cameras", &n_cameras.to_string());
    p.row(
        line,
        d,
        -1,
        "source_exists_after_import",
        if file.exists() { "1" } else { "0" },
    );
    for (i, (files, text)) in boxes.iter().enumerate() {
        let i = i as i64;
        let label = if files.is_empty() && text.is_empty() {
            "Video path".to_owned()
        } else {
            format!("({})", files.len())
        };
        let joined = files
            .iter()
            .map(|f| f.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join("|");
        p.row(line, d, i, "name", &names[i as usize]);
        p.row(line, d, i, "line_edit", text);
        p.row(line, d, i, "label", &label);
        p.row(line, d, i, "n_files", &files.len().to_string());
        p.row(line, d, i, "files", &joined);
    }
    let spec = NewTrialSpec {
        name: name.map_or(trial_name, str::to_owned),
        cameras: boxes.into_iter().map(|(f, _)| f).collect(),
    };
    let doc = match p.doc() {
        Ok(d) => d,
        Err(e) => return e,
    };
    match doc.apply(&Edit::NewTrialFromPortal {
        spec,
        metadata: import.metadata,
    }) {
        Ok(o) => OpStatus::Ok(o.refused.unwrap_or_default()),
        Err(EditError::Rejected(m)) => OpStatus::Rejected(m),
        Err(e) => OpStatus::Invalid(e.to_string()),
    }
}

fn op_open_project(
    p: &mut Portal,
    line: usize,
    path: &str,
    board: Option<(u32, u32, f64)>,
) -> OpStatus {
    let file = at_out(&p.work, path);
    let compat = p.state.options.compat;
    let dest = Destination {
        workspace: None,
        project_file: None,
        portal_file: &file,
    };
    let import = match import_project(&file, &dest, compat, &mut |_| true) {
        Ok(i) => i,
        Err(e) => return OpStatus::Rejected(e.to_string()),
    };
    p.imports.push(import.clone());
    let mut spec = import.spec();
    // The dialog starts with two boxes (`NewProjectDialog::NewProjectDialog`); the second
    // one of a one-camera archive stays empty and is a light camera.
    while spec.cameras.len() < 2 {
        let n = spec.cameras.len() + 1;
        spec.cameras.push(xma_engine::create::NewCameraSpec {
            name: format!("Camera {n}"),
            images: Vec::new(),
            light_camera: true,
            grid: None,
            flipped: false,
        });
    }
    let d = "NewProjectDialog";
    let (cube, csv, refs) = match &spec.target {
        CalibrationTarget::Cube {
            frame_specification,
            references,
        } => (
            true,
            frame_specification.to_string_lossy().into_owned(),
            references.to_string_lossy().into_owned(),
        ),
        CalibrationTarget::Checkerboard { .. } => (false, String::new(), String::new()),
    };
    p.row(line, d, -1, "window_title", "Create a new dataset");
    let n = spec.cameras.len().to_string();
    p.row(line, d, -1, "n_cameras", &n);
    p.row(line, d, -1, "label_nb_cameras", &n);
    p.row(line, d, -1, "cube", if cube { "1" } else { "0" });
    p.row(line, d, -1, "checkerboard", if cube { "0" } else { "1" });
    p.row(line, d, -1, "frame_specifications", &csv);
    p.row(line, d, -1, "reference_points", &refs);
    p.row(line, d, -1, "horizontal_squares", "0");
    p.row(line, d, -1, "vertical_squares", "0");
    p.row(line, d, -1, "square_size", "0");
    let cams = spec.cameras.clone();
    for (i, c) in cams.iter().enumerate() {
        let i = i as i64;
        let images: Vec<String> = c
            .images
            .iter()
            .map(|f| f.to_string_lossy().into_owned())
            .collect();
        let b = |v: bool| if v { "1" } else { "0" };
        p.row(line, d, i, "name", &c.name);
        p.row(line, d, i, "n_images", &images.len().to_string());
        p.row(line, d, i, "images", &images.join("|"));
        p.row(line, d, i, "line_edit_images", &common_prefix(&images));
        let label = if images.is_empty() {
            String::new()
        } else {
            format!("({})", images.len())
        };
        p.row(line, d, i, "label_nb_images", &label);
        p.row(line, d, i, "xray", b(!c.light_camera));
        p.row(line, d, i, "light", b(c.light_camera));
        p.row(line, d, i, "undistortion_grid_checked", "1");
        let grid = c
            .grid
            .as_ref()
            .map(|g| g.to_string_lossy().into_owned())
            .unwrap_or_default();
        p.row(line, d, i, "grid", &grid);
        p.row(line, d, i, "has_undistortion", b(!c.light_camera));
        p.row(line, d, i, "flip", "0");
    }
    if let (Some((h, v, s)), CalibrationTarget::Checkerboard { .. }) = (board, &spec.target) {
        spec.target = CalibrationTarget::Checkerboard {
            horizontal_squares: h,
            vertical_squares: v,
            square_size: s,
        };
    }
    for c in &mut spec.cameras {
        if c.light_camera {
            c.grid = None;
        }
    }
    let r = Document::new_project_from_portal(&spec, &import, &p.state.options);
    match r {
        Ok(mut doc) => {
            doc.session_mut().wizard = p.state.wizard;
            let info = format!(
                "cameras {}; trials {}; calibration type {}",
                doc.project().cameras.len(),
                doc.project().trials.len(),
                doc.project().calibration_type.code()
            );
            p.state.doc = Some(doc);
            OpStatus::Ok(info)
        }
        Err(xma_engine::CreateError::Incomplete(m) | xma_engine::CreateError::Refused(m)) => {
            OpStatus::Rejected(m)
        }
        Err(e) => OpStatus::Invalid(e.to_string()),
    }
}

fn tree_rows(p: &mut Portal, line: usize, nodes: &[MetaNode]) {
    fn walk(p: &mut Portal, line: usize, top: i64, prefix: &str, n: &MetaNode) {
        let path = if prefix.is_empty() {
            n.label.clone()
        } else {
            format!("{prefix}/{}", n.label)
        };
        p.row(line, "MetaDataInfo", top, &path, &n.value);
        for c in &n.children {
            walk(p, line, top, &path, c);
        }
    }
    for (i, n) in nodes.iter().enumerate() {
        walk(p, line, i as i64, "", n);
    }
}

fn op_add_metadata(
    p: &mut Portal,
    line: usize,
    target: &str,
    path: &str,
    override_expected: bool,
) -> OpStatus {
    let file = at_out(&p.work, path);
    let compat = p.state.options.compat;
    let doc = match p.doc() {
        Ok(d) => d,
        Err(e) => return e,
    };
    let pr = doc.project();
    let items = xma_engine::portal::metadata_items(pr);
    let Some(i) = items.iter().position(|x| x == target) else {
        return OpStatus::Invalid(format!("{target} is not in the list"));
    };
    let Some(trial) = xma_engine::portal::metadata_target(pr, i) else {
        return OpStatus::Invalid(format!("{target} is not in the list"));
    };
    let has = match trial {
        None => !pr.portal_metadata.is_empty(),
        Some(t) => !pr.trials[t].portal_metadata.is_empty(),
    };
    let mut q = vec![METADATA_CONFIRM.to_owned()];
    if has {
        q.push(override_text(target));
    }
    let joined = items.join("|");
    p.questions.push((line, q));
    p.row(line, "QInputDialog", -1, "items", &joined);
    if has != override_expected {
        return OpStatus::Invalid(format!(
            "the override question is {}asked here",
            if has { "" } else { "not " }
        ));
    }
    let lines = match read_metadata(&file, compat) {
        Ok(Some(l)) => l,
        Ok(None) => Vec::new(),
        Err(e) => return OpStatus::Rejected(e.to_string()),
    };
    let doc = match p.doc() {
        Ok(d) => d,
        Err(e) => return e,
    };
    if !lines.is_empty()
        && let Err(e) = doc.apply(&Edit::SetPortalMetadata {
            trial,
            metadata: lines,
        })
    {
        return OpStatus::Invalid(e.to_string());
    }
    let tree = metadata_tree(doc.project());
    tree_rows(p, line, &tree);
    OpStatus::Ok(String::new())
}

fn apply(p: &mut Portal, line: usize, op: &PortalOp) -> OpStatus {
    match op {
        PortalOp::OpenProject { path, board } => op_open_project(p, line, path, *board),
        PortalOp::NewTrialFromPortal {
            path,
            delete_after,
            name,
        } => op_new_trial(p, line, path, *delete_after, name.as_deref()),
        PortalOp::AddPortalMetadata {
            target,
            path,
            override_expected,
        } => op_add_metadata(p, line, target, path, *override_expected),
        PortalOp::SettingText(values) => {
            for (k, v) in values {
                if k == "WorkspacePath" {
                    p.workspace_path.clone_from(v);
                }
            }
            OpStatus::Ok(String::new())
        }
        PortalOp::SelectTrial(t) => match p.doc() {
            Ok(doc) => match doc.project().trials.iter().position(|x| x.name == *t) {
                Some(i) => {
                    doc.select_trial(i);
                    OpStatus::Ok(String::new())
                }
                None => OpStatus::Invalid(format!("no trial {t}")),
            },
            Err(e) => e,
        },
        PortalOp::Shared(super::calibration::CalibOp::Setting(values)) => {
            let mut rest = Vec::new();
            for (k, v) in values {
                if k == "CustomWorkspacePath" {
                    p.custom_workspace = v == "true";
                } else {
                    rest.push((k.clone(), v.clone()));
                }
            }
            if rest.is_empty() {
                OpStatus::Ok(String::new())
            } else {
                apply_op(&mut p.state, &super::calibration::CalibOp::Setting(rest))
            }
        }
        PortalOp::Shared(op) => apply_op(&mut p.state, op),
    }
}

// ---------------------------------------------------------------------------------------
// Comparison

/// The C++ run's own folder (`manifest.json`'s script path: `<oracle>/data/portal/scripts/
/// <run>.txt` on the machine that made it).
fn cpp_out(run: &PortalRun) -> Option<String> {
    let m = Manifest::try_read(run.run.output.join("manifest.json")).ok()?;
    let script = PathBuf::from(m.json().get("script")?.as_str()?);
    let root = script.parent()?.parent()?.parent()?.parent()?;
    let rel = run
        .run
        .output
        .strip_prefix(xma_oracle::oracle_root())
        .ok()?
        .to_path_buf();
    Some(root.join(rel).to_string_lossy().into_owned())
}

/// The C++ saved file with its run's folder read as `work` in every text entry.
fn rewritten_save(src: &Path, from: &str, to: &str, dst: &Path) -> Result<(), String> {
    let bytes = std::fs::read(src).map_err(|e| format!("{}: {e}", src.display()))?;
    let a = xma_io::zip::Archive::new(&bytes).map_err(|e| e.to_string())?;
    let mut data = Vec::new();
    let mut names = Vec::new();
    let mut times = Vec::new();
    for i in 0..a.len() {
        let e = a.entry(i).ok_or("entry")?;
        let mut d = a.read(i).map_err(|e| e.to_string())?;
        let n = e.decoded_name();
        if [".txt", ".xml", ".csv"].iter().any(|x| n.ends_with(x)) {
            d = String::from_utf8_lossy(&d).replace(from, to).into_bytes();
        }
        names.push(e.name.clone());
        data.push(d);
        times.push(e.modified);
    }
    let entries: Vec<xma_io::zip::NewEntry<'_>> = (0..names.len())
        .map(|i| xma_io::zip::NewEntry {
            name: &names[i],
            data: &data[i],
            modified: times[i],
        })
        .collect();
    let out = xma_io::zip::write_archive(&entries, a.comment()).map_err(|e| e.to_string())?;
    std::fs::write(dst, out).map_err(|e| format!("{}: {e}", dst.display()))
}

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

/// A listed folder: the same files, with the same bytes (a saved project in it is left out:
/// the start project, unchanged, and the run's own saves).
fn compare_listing(cpp: &Path, ours: &Path, name: &str) -> Item {
    let mut c = Check::new(format!("{name}/ (files and bytes)"), Tier::Read);
    let skip = |f: &String| f.ends_with(".xma");
    let a: BTreeSet<String> = files_under(ours).into_iter().filter(|f| !skip(f)).collect();
    let b: BTreeSet<String> = files_under(cpp).into_iter().filter(|f| !skip(f)).collect();
    c.int("files", a.len() as i64, b.len() as i64);
    let mut same = a == b;
    for f in a.difference(&b) {
        c.problem(format!("{f}: written here, not by the C++"));
    }
    for f in b.difference(&a) {
        c.problem(format!("{f}: written by the C++, not here"));
    }
    for f in a.intersection(&b) {
        let (x, y) = (std::fs::read(ours.join(f)), std::fs::read(cpp.join(f)));
        match (x, y) {
            (Ok(x), Ok(y)) if x == y => {}
            (Ok(x), Ok(y)) => {
                same = false;
                c.problem(format!(
                    "{f}: {} bytes here, {} in the C++",
                    x.len(),
                    y.len()
                ));
            }
            _ => {
                same = false;
                c.problem(format!("{f}: cannot be read"));
            }
        }
    }
    c.bytes_equal = Some(same);
    c.finish()
}

fn compare_dialogs(cpp: &Path, ours: &[(usize, String, i64, String, String)]) -> Item {
    let mut c = Check::new("dialogs.csv", Tier::Read);
    let text = std::fs::read_to_string(cpp).unwrap_or_default();
    let want: Vec<&str> = text.lines().skip(1).collect();
    let got: Vec<String> = ours
        .iter()
        .map(|(l, d, i, k, v)| format!("{l},{d},{i},{k},{v}"))
        .collect();
    c.int("rows", got.len() as i64, want.len() as i64);
    c.bytes_equal = Some(got.len() == want.len() && got.iter().zip(&want).all(|(g, w)| g == w));
    for (k, w) in want.iter().enumerate() {
        match got.get(k) {
            Some(g) => c.text(format_args!("row {}", k + 2), g, w),
            None => c.problem(format!("row {}: missing here: {w}", k + 2)),
        }
    }
    for g in got.iter().skip(want.len()) {
        c.problem(format!("written here, not by the C++: {g}"));
    }
    c.finish()
}

/// The confirmations the C++ log records (`[Confirmation Required: <text>]`), commas
/// written `;` as the driver writes them.
fn cpp_questions(messages: &str) -> Vec<String> {
    let key = "[Confirmation Required: ";
    let mut out = Vec::new();
    let mut rest = messages;
    while let Some(i) = rest.find(key) {
        rest = &rest[i + key.len()..];
        let end = rest.find(']').unwrap_or(rest.len());
        out.push(rest[..end].to_owned());
        rest = &rest[end..];
    }
    out
}

/// Replays one run and compares what it reached.
///
/// # Errors
/// When the run's script, manifest, staged files or start project cannot be read.
pub fn run_portal_parity(run: &PortalRun) -> Result<CalibParityReport, String> {
    let start = Instant::now();
    let (lines, listings) = read_script(&run.run.script)?;
    let work = std::env::temp_dir().join(format!(
        "xma_portal_parity_{}_{}",
        run.run.name,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&work);
    for (src, dst) in &run.staged {
        let to = work.join(dst);
        if let Some(d) = to.parent() {
            std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
        }
        std::fs::copy(src, &to).map_err(|e| format!("{}: {e}", src.display()))?;
    }
    for l in &listings {
        let _ = std::fs::create_dir_all(work.join(l));
    }
    // The trial update queue left after a new trial depends on timing (the C++ repeats of
    // p02 to p04 differ in it, `DATASETS.md`): every dump after a Portal trial import is
    // compared with each repeat, as the calibration runs do (`scheduler.csv`).
    let mut calib_run = run.run.clone();
    let mut after_trial = false;
    for l in &lines {
        match &l.op {
            PortalOp::NewTrialFromPortal { .. } => after_trial = true,
            PortalOp::Shared(super::calibration::CalibOp::Dump { name, .. }) if after_trial => {
                let k = format!("{name}/scheduler.csv");
                if !calib_run.unstable.contains(&k) {
                    calib_run.unstable.push(k);
                }
            }
            _ => {}
        }
    }
    let options = run_options(&run.run)?;
    let mut p = Portal {
        state: ReplayState {
            doc: None,
            seed: None,
            options,
            out_dir: work.clone(),
            engopt: engopt::EngoptState::for_run(&run.run),
            wizard: run_wizard(&run.run),
        },
        work: work.clone(),
        workspace_path: String::new(),
        custom_workspace: false,
        dialogs: Vec::new(),
        questions: Vec::new(),
        imports: Vec::new(),
    };
    // `# project: @out/...`: the start project, opened where the C++ opened it.
    let text = std::fs::read_to_string(&run.run.script).unwrap_or_default();
    if let Some(proj) = text
        .lines()
        .find_map(|l| l.strip_prefix("# project:").map(str::trim))
    {
        let path = at_out(&work, proj);
        let mut doc = Document::open_with(&path, &p.state.options)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        doc.session_mut().wizard = p.state.wizard;
        p.state.doc = Some(doc);
    }
    let cpp_log = read_ops_log(&run.run.output.join("ops_log.csv"));
    let cpp_dir = cpp_out(run);
    let work_s = work.to_string_lossy().into_owned();
    let mut ops = Vec::new();
    let mut points = Vec::new();
    for l in &lines {
        match &l.op {
            PortalOp::Shared(super::calibration::CalibOp::Dump { name, model }) => {
                let items = match &p.state.doc {
                    Some(doc) => compare_dump(&calib_run, name, doc, *model),
                    None => Vec::new(),
                };
                points.push(Point {
                    name: name.clone(),
                    line: l.line,
                    items,
                    skipped: p
                        .state
                        .doc
                        .is_none()
                        .then(|| "no project is open".to_owned()),
                });
                continue;
            }
            PortalOp::Shared(super::calibration::CalibOp::Save { name }) => {
                let cpp_file = run.run.output.join(name);
                let rewritten = work.join(format!("cpp_{name}"));
                let src = match &cpp_dir {
                    Some(d) if rewritten_save(&cpp_file, d, &work_s, &rewritten).is_ok() => {
                        rewritten.clone()
                    }
                    _ => cpp_file,
                };
                let items = match p.state.doc.as_mut() {
                    Some(doc) => compare_save(&src, name, doc, false, &SizeRules::default()),
                    None => Vec::new(),
                };
                points.push(Point {
                    name: name.clone(),
                    line: l.line,
                    items,
                    skipped: p
                        .state
                        .doc
                        .is_none()
                        .then(|| "no project is open".to_owned()),
                });
                continue;
            }
            _ => {}
        }
        let (cpp_status, cpp_messages) = cpp_log
            .iter()
            .find(|(n, ..)| *n == l.line)
            .map(|(_, _, st, m)| (st.clone(), m.clone()))
            .unwrap_or_else(|| ("(not logged)".to_owned(), String::new()));
        p.state.engopt.line = l.line;
        let out = apply(&mut p, l.line, &l.op);
        let (status, message) = match &out {
            OpStatus::Ok(m) => ("ok", m.clone()),
            OpStatus::Rejected(m) => ("rejected", m.clone()),
            OpStatus::Invalid(m) => ("invalid", m.clone()),
            OpStatus::NotImplemented => ("not implemented", String::new()),
        };
        // The questions asked, with their texts.
        let ours: Vec<String> = p
            .questions
            .iter()
            .filter(|(n, _)| *n == l.line)
            .flat_map(|(_, q)| q.iter().map(|t| t.replace(',', ";")))
            .collect();
        let theirs = cpp_questions(&cpp_messages);
        let questions_agree = ours == theirs;
        ops.push(OpRow {
            line: l.line,
            op: l.name.clone(),
            cpp_status: cpp_status.clone(),
            cpp_messages,
            status: status.to_owned(),
            message: if questions_agree {
                message
            } else {
                format!("{message} | questions here {ours:?}, C++ {theirs:?}")
            },
            agrees: status == cpp_status && questions_agree,
        });
    }
    let mut extra = Vec::new();
    extra.push(compare_dialogs(
        &run.run.output.join("dialogs.csv"),
        &p.dialogs,
    ));
    for l in &listings {
        extra.push(compare_listing(&run.run.output.join(l), &work.join(l), l));
    }
    points.push(Point {
        name: "dialogs and files".to_owned(),
        line: 0,
        items: extra,
        skipped: None,
    });
    let warnings = p
        .state
        .doc
        .as_ref()
        .map(|d| d.warnings().to_vec())
        .unwrap_or_default();
    drop(p);
    let _ = std::fs::remove_dir_all(&work);
    Ok(CalibParityReport {
        run: run.run.clone(),
        ops,
        points,
        stopped: None,
        warnings,
        seconds: start.elapsed().as_secs_f64(),
    })
}

/// The compatibility mode the runs are replayed in.
#[must_use]
pub fn mode() -> CppCompat {
    CppCompat::XMALAB3
}

#[cfg(test)]
mod review12_tests {
    use super::common_prefix;

    /// Review of 20 November 2026, second round, finding 4: `CameraBox::commonPrefix`
    /// returns `left(count + 1)`, one character more than the common start.
    #[test]
    fn review12_common_prefix_as_xmalab() {
        let v = |s: &[&str]| s.iter().map(|x| (*x).to_owned()).collect::<Vec<_>>();
        assert_eq!(common_prefix(&v(&["ab1", "ab2"])), "ab1");
        assert_eq!(common_prefix(&v(&["abc"])), "abc");
        assert_eq!(common_prefix(&v(&["abc", "abc"])), "abc");
        assert_eq!(common_prefix(&v(&["x", "y"])), "x");
        assert_eq!(
            common_prefix(&v(&["cam_001.tif", "cam_002.tif"])),
            "cam_001"
        );
        assert_eq!(common_prefix(&[]), "");
    }
}
