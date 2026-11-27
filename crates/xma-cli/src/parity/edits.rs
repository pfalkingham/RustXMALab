//! The editing reference: edit scripts run through XMALab's own code
//! (`oracle/edit_runs.json`, `oracle/FORMAT.md` "State dump" and "Edit scripts").
//!
//! [`run_edit_parity`] opens the run's start project in the XMALab 3 mode with the settings
//! in force at the run's first state dump, applies the script's operations as the C++
//! driver does ([`crate::edit_cmd::apply_op`], with XMALab 3's dialog controls), and at
//! every `Dump` compares the whole document with the C++ state dump, table by table, with
//! the comparisons of [`super`] (the "as held" tables of a dump and the tables of what
//! loading determines) and those of the state dump's own tables (`state.json`,
//! `trial_state.csv`, `scheduler.csv`, `points3d.csv`, `rb_pose_raw.csv`,
//! `rb_error_details.csv`, `rb_state.csv`). At `Save` the project is saved as the driver
//! saves it (`OracleDumpDriver::saveProjectLikeGui`: Save As stamped with the C++ file's
//! own time) and compared with the C++ file entry by entry.
//!
//! Rules that take values out of the comparison, each named in the report:
//!
//! - `markers.csv` `size_range` of a marker created during the run (ledger A75) or loaded
//!   without a `<Marker>` element (ledger A52): uninitialised in the C++;
//! - `rb_state.csv` `center_*`: `RigidBody::center`, a drawing cache refreshed only when
//!   the references are set, which the model does not hold;
//! - `trial_state.csv` `active_frame`, `active_marker`, `active_body`: the interface's
//!   selection, which the model does not hold;
//! - once an operation the C++ applied is not applied here, the later dumps are not
//!   compared (`PlacePoint` and `RefinePoints` are applied, with the trial's images).
//!
//! One named tolerance: once `RefinePoints` has run, the 2D sizes, mean sizes and size
//! ranges (and the saved `Marker<n>size.csv` files) are compared within
//! [`super::TOLERANCE_A37`], the refined radius of ledger A37; every other value exactly.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use xma_engine::edit::script::ScriptOp;
use xma_engine::{Document, OpenOptions, SaveOptions, SaveTime};
use xma_io::zip::Archive;
use xma_model::{CppCompat, SettingValue};
use xma_oracle::{Dataset, JsonValue, Manifest};

use super::race::Race;
use super::{
    Check, Item, NamedTolerance, Tier, Verdict, cell, check_event_details, check_events,
    check_marker_details, check_marker_interpolation, check_markers_tol, check_points2d_with,
    check_points3d, check_points3d_filtered, check_projection, check_rb_details,
    check_rb_dummy_coords, check_rb_dummy_details, check_rb_dummy_points, check_rb_errors,
    check_rb_points, check_rb_poses, check_rb_reference_names, check_reprojection,
    check_rigid_bodies, check_trials, check_undistorted, dump_text, get, num, rf, ri, rs, ru,
};
use crate::edit_cmd::{OpStatus, apply_op, read_script};

const RULE_A75: &str = "not compared: C++ value uninitialised, ledger A75";
const RULE_THROWN: &str = "not compared: XMALab throws an exception before this";
const RULE_CENTRE: &str = "not compared: drawing cache, not held by the model";
const RULE_SELECTION: &str = "not compared: interface selection, not held by the model";

/// One run of `oracle/edit_runs.json`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EditRun {
    pub name: String,
    /// The project the run starts from.
    pub start_project: PathBuf,
    pub script: PathBuf,
    /// The run's output folder (repeat `a`).
    pub output: PathBuf,
}

/// The runs listed in `oracle/edit_runs.json`.
///
/// # Errors
/// When the file cannot be read or does not have the expected layout.
pub fn edit_runs() -> Result<Vec<EditRun>, String> {
    let root = xma_oracle::oracle_root();
    let path = root.join("edit_runs.json");
    let manifest = Manifest::try_read(&path).map_err(|e| e.to_string())?;
    let scripts = manifest
        .json()
        .get("scripts")
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("{}: no 'scripts' list", path.display()))?;
    let field = |s: &JsonValue, k: &str| -> Result<String, String> {
        s.get(k)
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .ok_or_else(|| format!("{}: a script without '{k}'", path.display()))
    };
    scripts
        .iter()
        .map(|s| {
            Ok(EditRun {
                name: field(s, "name")?,
                start_project: root.join(field(s, "start_project")?),
                script: root.join(field(s, "script")?),
                output: root.join(field(s, "output")?),
            })
        })
        .collect()
}

/// The run of that name, if `oracle/edit_runs.json` lists it.
#[must_use]
pub fn find_edit_run(name: &str) -> Option<EditRun> {
    edit_runs().ok()?.into_iter().find(|r| r.name == name)
}

impl EditRun {
    /// True when the start project, the script and the output folder exist.
    #[must_use]
    pub fn is_present(&self) -> bool {
        self.start_project.is_file() && self.script.is_file() && self.output.is_dir()
    }
}

/// One operation: what the C++ logged and what happened here.
#[derive(Clone, Debug)]
pub struct OpRow {
    pub line: usize,
    pub op: String,
    pub cpp_status: String,
    pub cpp_messages: String,
    pub status: String,
    pub message: String,
    /// The C++ and this program agree on the outcome (and on the message of a refusal).
    pub agrees: bool,
}

/// One `Dump` or `Save`, compared.
#[derive(Clone, Debug)]
pub struct Point {
    /// The dump's or saved file's name.
    pub name: String,
    pub line: usize,
    pub items: Vec<Item>,
    /// Why nothing was compared.
    pub skipped: Option<String>,
}

impl Point {
    #[must_use]
    pub fn passed(&self) -> bool {
        self.items.iter().all(|i| i.verdict != Verdict::Different)
    }
}

/// The result of [`run_edit_parity`].
#[derive(Clone, Debug)]
pub struct EditParityReport {
    pub run: EditRun,
    pub ops: Vec<OpRow>,
    pub points: Vec<Point>,
    pub warnings: Vec<String>,
    pub seconds: f64,
}

impl EditParityReport {
    /// No table differs and every operation agrees with the C++ outcome. An operation not
    /// applied here fails the run (the dumps after it go uncompared).
    #[must_use]
    pub fn passed(&self) -> bool {
        self.points.iter().all(Point::passed) && self.ops.iter().all(|o| o.agrees)
    }

    /// The operations not applied here, as "line N (Op)".
    fn not_applied(&self) -> Vec<String> {
        self.ops
            .iter()
            .filter(|o| o.status == "not applied")
            .map(|o| format!("line {} ({})", o.line, o.op))
            .collect()
    }

    /// Tables and files compared.
    #[must_use]
    pub fn items(&self) -> usize {
        self.points.iter().map(|p| p.items.len()).sum()
    }

    fn count(&self, v: Verdict) -> usize {
        self.points
            .iter()
            .flat_map(|p| &p.items)
            .filter(|i| i.verdict == v)
            .count()
    }

    /// The items that differ, as `dump/table`.
    #[must_use]
    pub fn different(&self) -> Vec<String> {
        self.points
            .iter()
            .flat_map(|p| {
                p.items
                    .iter()
                    .filter(|i| i.verdict == Verdict::Different)
                    .map(move |i| format!("{}/{}", p.name, i.name))
            })
            .collect()
    }

    /// One line.
    #[must_use]
    pub fn summary(&self) -> String {
        let compared = self.points.iter().filter(|p| p.skipped.is_none()).count();
        format!(
            "{}: {} — {} of {} dumps and saves compared, {} tables and files ({} identical, {} \
             identical apart from explained cells, {} empty on both sides, {} within \
             tolerance, {} different), {} values; {} of {} operations agree",
            self.run.name,
            if self.passed() { "PASS" } else { "FAIL" },
            compared,
            self.points.len(),
            self.items(),
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
            self.ops.len()
        ) + &match self.not_applied() {
            n if n.is_empty() => String::new(),
            n => format!("; not applied here: {}", n.join(", ")),
        }
    }

    /// The report as Markdown.
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "# Editing parity report: {}\n", self.run.name);
        let _ = writeln!(s, "**Result: {}.**\n", self.summary());
        let _ = writeln!(s, "- Start project: `{}`", self.run.start_project.display());
        let _ = writeln!(s, "- Script: `{}`", self.run.script.display());
        let _ = writeln!(s, "- Reference run: `{}`", self.run.output.display());
        let _ = writeln!(
            s,
            "- Compatibility mode: {} (with XMALab 3's dialog controls)",
            CppCompat::XMALAB3.describe()
        );
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
                if o.agrees { "yes" } else { "**no**" }
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
                    "| {} | `{}` | not compared: {} | | | | | |",
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

/// For each trial and marker, whether the C++ holds an uninitialised size range there, and
/// why (ledger A52 at load, A75 for a marker created during the run).
pub(super) type Uninit = Vec<Vec<Option<(&'static str, &'static str)>>>;

/// How sizes are compared in a state dump and a saved project.
#[derive(Clone, Debug, Default)]
pub(super) struct SizeRules {
    /// (trial, marker, rule, why): markers whose 2D sizes are not compared (and whose
    /// `Marker<n>size.csv` is not compared in a saved project).
    pub skip: Vec<(usize, usize, &'static str, &'static str)>,
    /// Sizes, mean sizes and size ranges compared within this tolerance.
    pub tolerance: Option<NamedTolerance>,
    /// Saved camera matrices, rotation matrices and translation vectors compared within this
    /// tolerance (the calibration runs: [`super::TOLERANCE_A37_CUBE`]).
    pub calibration_tolerance: Option<NamedTolerance>,
}

impl SizeRules {
    fn skip_in(&self, t: usize) -> Vec<(usize, &'static str, &'static str)> {
        self.skip
            .iter()
            .filter(|s| s.0 == t)
            .map(|&(_, m, r, w)| (m, r, w))
            .collect()
    }
}

/// The size ranges the C++ leaves uninitialised after loading (ledger A52).
pub(super) fn loaded_uninit(doc: &Document) -> Uninit {
    doc.project()
        .trials
        .iter()
        .enumerate()
        .map(|(t, tr)| {
            let without: Vec<usize> = doc
                .extras()
                .trial_sources
                .get(t)
                .map_or_else(Vec::new, |s| s.markers_without_element.clone());
            (0..tr.markers.len())
                .map(|i| {
                    without.contains(&i).then_some((
                        super::RULE_A52,
                        "ledger A52: the marker has no <Marker> element, so Marker::load never \
                         sets it",
                    ))
                })
                .collect()
        })
        .collect()
}

/// Runs the comparison for one run.
///
/// # Errors
/// When the run's files cannot be read.
pub fn run_edit_parity(run: &EditRun) -> Result<EditParityReport, String> {
    let start = Instant::now();
    let lines = read_script(&run.script)?;
    let first_dump = lines.iter().find_map(|l| match &l.op {
        ScriptOp::Dump { name } => Some(name.clone()),
        _ => None,
    });
    // The settings in force at the start: the first dump's (the runs override none).
    let settings_file = match &first_dump {
        Some(d) => run.output.join(d).join("state.json"),
        None => run.output.join("manifest.json"),
    };
    let manifest = Manifest::try_read(&settings_file).map_err(|e| e.to_string())?;
    let options = OpenOptions {
        overrides: crate::settings::manifest_settings(&manifest)?,
        compat: CppCompat::XMALAB3,
        ..OpenOptions::default()
    };
    let loaded = xma_engine::read_project(&run.start_project, &options)
        .map_err(|e| format!("{}: {e}", run.start_project.display()))?;
    let mut doc = Document::from_loaded(loaded);
    doc.set_path(&run.start_project);
    doc.update_after_load();
    let recompute_when_saving = manifest.try_bool("RecomputeWhenSaving").unwrap_or(false);

    let mut uninit: Uninit = loaded_uninit(&doc);
    // After a refinement, sizes are compared within the named tolerance of ledger A37.
    let mut sizes = SizeRules::default();

    let cpp_log = read_ops_log(&run.output.join("ops_log.csv"))?;
    let mut ops = Vec::new();
    let mut points = Vec::new();
    let mut not_applied: Option<String> = None;
    // Trials an operation that throws in XMALab acted on: XMALab ends there (an exception
    // out of an interface handler), so what the reference holds afterwards for that trial is
    // the driver's, not the program's; those trials are not compared after it.
    let mut thrown: Option<(usize, String)> = None;
    for line in &lines {
        match &line.op {
            ScriptOp::Dump { name } => {
                let skipped = not_applied.clone();
                let items = if skipped.is_some() {
                    Vec::new()
                } else {
                    compare_state(
                        &run.output.join(name),
                        name,
                        &doc,
                        &uninit,
                        thrown.as_ref(),
                        &sizes,
                    )
                };
                points.push(Point {
                    name: name.clone(),
                    line: line.line,
                    items,
                    skipped,
                });
                continue;
            }
            ScriptOp::Save { name } => {
                let skipped = not_applied.clone().or_else(|| {
                    thrown.as_ref().map(|(t, why)| {
                        format!("the saved project holds trial {t}, not comparable {why}")
                    })
                });
                let items = if skipped.is_some() {
                    Vec::new()
                } else {
                    compare_save(
                        &run.output.join(name),
                        name,
                        &mut doc,
                        recompute_when_saving,
                        &sizes,
                    )
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
        let before = marker_counts(&doc);
        let n_trials_before = doc.project().trials.len();
        let out = apply_op(&mut doc, &line.op, true);
        track_markers(
            &mut uninit,
            &doc,
            &line.op,
            &out.status,
            &before,
            n_trials_before,
        );
        let (cpp_status, cpp_messages) = cpp_log
            .iter()
            .find(|(l, ..)| *l == line.line)
            .map(|(_, _, st, m)| (st.clone(), m.clone()))
            .unwrap_or_else(|| ("(not logged)".to_owned(), String::new()));
        let mut message = out.status.message().to_owned();
        for n in &out.notes {
            if !message.is_empty() {
                message.push_str(" | ");
            }
            message.push_str(n);
        }
        if out.status == OpStatus::Ok && matches!(line.op, ScriptOp::RefinePoints { .. }) {
            sizes.tolerance = Some(super::TOLERANCE_A37);
        }
        let agrees = match &out.status {
            // An exception ends XMALab: nothing to agree with (ledger F29 and the like).
            OpStatus::Ok => cpp_status == "ok" || cpp_status == "exception",
            OpStatus::Rejected(m) => {
                cpp_status == "rejected"
                    && cpp_messages.contains(&format!("rejected: {}", dump_text(m)))
            }
            OpStatus::Invalid(_) => cpp_status == "invalid",
            OpStatus::NotApplied(_) => false,
        };
        if cpp_status == "exception"
            && thrown.is_none()
            && let Some(t) = line.op.handler_trial()
        {
            thrown = Some((
                t,
                format!(
                    "after line {} ({}), which throws an exception in XMALab (ends the \
                     program there; ledger F29)",
                    line.line,
                    line.op.name()
                ),
            ));
        }
        if let OpStatus::NotApplied(why) = &out.status
            && cpp_status == "ok"
            && not_applied.is_none()
        {
            not_applied = Some(format!(
                "after line {} ({}), which the C++ applied and this program does not: {why}",
                line.line,
                line.op.name()
            ));
        }
        ops.push(OpRow {
            line: line.line,
            op: line.op.name().to_owned(),
            cpp_status,
            cpp_messages,
            status: out.status.label().to_owned(),
            message,
            agrees,
        });
    }
    let mut warnings: Vec<String> = doc.warnings().to_vec();
    warnings.extend(doc.extras().warnings.iter().cloned());
    Ok(EditParityReport {
        run: run.clone(),
        ops,
        points,
        warnings,
        seconds: start.elapsed().as_secs_f64(),
    })
}

/// `ops_log.csv`: (line, op, status, messages).
fn read_ops_log(path: &Path) -> Result<Vec<(usize, String, String, String)>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(text
        .lines()
        .skip(1)
        .filter_map(|l| {
            let mut it = l.splitn(4, ',');
            let line = it.next()?.parse().ok()?;
            Some((
                line,
                it.next()?.to_owned(),
                it.next()?.to_owned(),
                it.next().unwrap_or("").to_owned(),
            ))
        })
        .collect())
}

fn marker_counts(doc: &Document) -> Vec<usize> {
    doc.project()
        .trials
        .iter()
        .map(|t| t.markers.len())
        .collect()
}

/// Keeps [`Uninit`] in step with the markers: markers created by an operation (counts that
/// grew) are uninitialised (ledger A75), removed ones dropped, trials deleted or imported
/// followed.
fn track_markers(
    uninit: &mut Uninit,
    doc: &Document,
    op: &ScriptOp,
    status: &OpStatus,
    before: &[usize],
    n_trials_before: usize,
) {
    use xma_engine::Edit;
    const A75: (&str, &str) = (
        RULE_A75,
        "ledger A75: a marker made by Trial::addMarker has no size range until one is computed",
    );
    if *status == OpStatus::Ok {
        match op {
            ScriptOp::Edit(Edit::DeleteTrials { trials }) => {
                let mut list = trials.clone();
                list.sort_unstable();
                list.dedup();
                for &t in list.iter().rev() {
                    if t < uninit.len() {
                        uninit.remove(t);
                    }
                }
            }
            ScriptOp::Edit(Edit::RemoveMarkers { trial, markers }) => {
                let mut list = markers.clone();
                list.sort_unstable();
                list.dedup();
                if let Some(u) = uninit.get_mut(*trial) {
                    for &m in list.iter().rev() {
                        if m < u.len() {
                            u.remove(m);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    let n = doc.project().trials.len();
    // Trials read from a file (imported, or a Default trial replaced): as loading leaves
    // them (ledger A52 for markers without a `<Marker>` element).
    let loaded = |t: usize| -> Vec<Option<(&'static str, &'static str)>> {
        let without: Vec<usize> = doc
            .extras()
            .trial_sources
            .get(t)
            .map_or_else(Vec::new, |s| s.markers_without_element.clone());
        (0..doc.trial(t).markers.len())
            .map(|i| {
                without.contains(&i).then_some((
                    super::RULE_A52,
                    "ledger A52: the marker has no <Marker> element, so Marker::load never \
                     sets it",
                ))
            })
            .collect()
    };
    if *status == OpStatus::Ok
        && let ScriptOp::Edit(Edit::ImportTrials { .. }) = op
    {
        for t in 0..n.min(uninit.len()) {
            if doc.trial(t).is_default {
                uninit[t] = loaded(t);
            }
        }
    }
    while uninit.len() < n {
        let t = uninit.len();
        uninit.push(loaded(t));
    }
    uninit.truncate(n);
    let _ = n_trials_before;
    for (t, u) in uninit.iter_mut().enumerate() {
        let now = doc.trial(t).markers.len();
        let was = before.get(t).copied().unwrap_or(now);
        if u.len() != was && u.len() != now {
            u.resize(was.min(now), None);
        }
        if now > u.len() {
            // A marker made by placing a point gets its size range from that point's
            // detection (Marker::setSize, updateMeanSize): it is set, not uninitialised.
            let made = if *status == OpStatus::Ok && matches!(op, ScriptOp::PlacePoint { .. }) {
                None
            } else {
                Some(A75)
            };
            u.resize(now, made);
        } else {
            u.truncate(now);
        }
    }
}

/// Every table of one state dump.
pub(super) fn compare_state(
    dir: &Path,
    name: &str,
    doc: &Document,
    uninit: &Uninit,
    thrown: Option<&(usize, String)>,
    sizes: &SizeRules,
) -> Vec<Item> {
    let ds = Dataset::from_paths(name, doc.path().unwrap_or(Path::new("")), dir);
    let mut items = Vec::new();
    if !dir.is_dir() {
        let mut c = Check::new(format!("{name}/"), Tier::Read);
        c.problem("the dump has no such folder");
        items.push(c.finish());
        return items;
    }
    items.push(check_state_json(&ds, doc));
    items.push(check_trials(&ds, doc));
    items.push(check_trial_state(&ds, doc));
    items.push(check_scheduler(&ds, doc));
    let n_trials = doc.project().trials.len();
    for t in 0..n_trials {
        let dir = format!("trial_{t}");
        if !ds.path(&dir).is_dir() {
            let mut c = Check::new(format!("{dir}/"), Tier::Read);
            c.problem("the trial has no folder in the dump");
            items.push(c.finish());
            continue;
        }
        if let Some((bad, why)) = thrown
            && *bad == t
        {
            let mut c = Check::new(format!("{dir}/"), Tier::Read);
            c.exclude(RULE_THROWN, 1, format!("{dir}: not compared: {why}"));
            items.push(c.finish());
            continue;
        }
        let rel = |f: &str| format!("{dir}/{f}");
        let race = Race::none();
        type F = fn(&Dataset, &Document, usize, &str) -> Item;
        let skip_sizes = sizes.skip_in(t);
        items.push(check_points2d_with(
            &ds,
            doc,
            t,
            &rel("points2d.csv"),
            &skip_sizes,
            sizes.tolerance,
        ));
        let plain: [(&str, F); 16] = [
            ("projection_matrices.csv", check_projection),
            ("points2d_undistorted.csv", check_undistorted),
            ("reprojection.csv", check_reprojection),
            ("points3d.csv", check_points3d),
            ("points3d_filtered.csv", check_points3d_filtered),
            ("rigid_bodies.csv", check_rigid_bodies),
            ("rb_points.csv", check_rb_points),
            ("rb_dummy_points.csv", check_rb_dummy_points),
            ("events.csv", check_events),
            ("marker_details.csv", check_marker_details),
            ("marker_interpolation.csv", check_marker_interpolation),
            ("rb_details.csv", check_rb_details),
            ("rb_reference_names.csv", check_rb_reference_names),
            ("rb_dummy_details.csv", check_rb_dummy_details),
            ("rb_dummy_coords.csv", check_rb_dummy_coords),
            ("event_details.csv", check_event_details),
        ];
        let markers_rel = rel("markers.csv");
        let list: Vec<(usize, &str, &str)> = uninit
            .get(t)
            .map(|u| {
                u.iter()
                    .enumerate()
                    .filter_map(|(i, r)| r.map(|(rule, why)| (i, rule, why)))
                    .collect()
            })
            .unwrap_or_default();
        items.push(check_markers_tol(
            &ds,
            doc,
            t,
            &markers_rel,
            &list,
            sizes.tolerance,
        ));
        for (file, f) in plain {
            let r = rel(file);
            if ds.has(&r) {
                let t0 = Instant::now();
                items.push(f(&ds, doc, t, &r));
                if std::env::var_os("XMA_PARITY_TIMING").is_some() {
                    eprintln!("{name}/{r}: {:.3} s", t0.elapsed().as_secs_f64());
                }
            }
        }
        for (file, filtered) in [("rb_pose_raw.csv", false), ("rb_pose_filtered.csv", true)] {
            items.push(check_rb_poses(&ds, doc, t, &rel(file), filtered, &race));
        }
        for (file, filtered) in [("rb_errors.csv", false), ("rb_errors_filtered.csv", true)] {
            items.push(check_rb_errors(&ds, doc, t, &rel(file), filtered, &race));
        }
        items.push(check_rb_error_details(
            &ds,
            doc,
            t,
            &rel("rb_error_details.csv"),
        ));
        items.push(check_rb_state(&ds, doc, t, &rel("rb_state.csv")));
    }
    // A trial folder the document does not have.
    let mut t = n_trials;
    while ds.path(&format!("trial_{t}")).is_dir() {
        let mut c = Check::new(format!("trial_{t}/"), Tier::Read);
        c.problem("the dump has this trial, the document does not");
        items.push(c.finish());
        t += 1;
    }
    items
}

/// `state.json`: the active trial and every setting the model holds.
fn check_state_json(ds: &Dataset, doc: &Document) -> Item {
    let mut c = Check::new("state.json", Tier::Read);
    let m = match Manifest::try_read(ds.path("state.json")) {
        Ok(m) => m,
        Err(e) => {
            c.problem(format!("cannot read the reference: {e}"));
            return c.finish();
        }
    };
    match m.json().get("active_trial").and_then(JsonValue::as_i64) {
        // State::activeTrial is -1 while the project has no trial.
        Some(a) => {
            let here = if doc.project().trials.is_empty() {
                -1
            } else {
                doc.project().active_trial as i64
            };
            c.int("active_trial", here, a);
        }
        None => c.problem("no active_trial"),
    }
    let settings = match m.try_settings() {
        Ok(s) => s,
        Err(e) => {
            c.problem(format!("settings: {e}"));
            return c.finish();
        }
    };
    let held: BTreeSet<&str> = xma_model::SETTING_NAMES.iter().map(|(n, _)| *n).collect();
    for s in settings {
        if !held.contains(s.name.as_str()) {
            continue;
        }
        let Some(here) = doc.settings().get(&s.name) else {
            c.problem(format!("setting {}: not held here", s.name));
            continue;
        };
        let want = s
            .value
            .as_f64()
            .or_else(|| s.value.as_bool().map(f64::from));
        let got = match here {
            SettingValue::Bool(b) => f64::from(u8::from(b)),
            SettingValue::Int(i) => i as f64,
            SettingValue::Float(f) => f,
        };
        match want {
            Some(w) if matches!(here, SettingValue::Float(_)) => {
                c.num(format_args!("setting {}", s.name), got, w);
            }
            Some(w) => c.int(format_args!("setting {}", s.name), got as i64, w as i64),
            None => c.problem(format!("setting {}: '{}' in the dump", s.name, s.value)),
        }
    }
    c.finish()
}

/// `trial_state.csv`.
fn check_trial_state(ds: &Dataset, doc: &Document) -> Item {
    let mut c = Check::new("trial_state.csv", Tier::Read);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let trials = &doc.project().trials;
    let active = doc.project().active_trial;
    c.rows(trials.len(), table.len());
    let mut selection = 0;
    c.each_row(&table, |c, r| {
        let i = ru(r, "trial")?;
        let t = get(trials, i, "trial")?;
        c.text(
            format_args!("trial {i} name"),
            &dump_text(&t.name),
            rs(r, "name")?,
        );
        let ints: [(&str, i64); 7] = [
            ("is_default", t.is_default.into()),
            ("is_copy_from_default", t.is_copy_from_default.into()),
            ("requires_recomputation", t.requires_recomputation.into()),
            ("n_markers", t.markers.len() as i64),
            ("n_rigid_bodies", t.rigid_bodies.len() as i64),
            ("n_events", t.events.len() as i64),
            ("is_active_trial", (i == active).into()),
        ];
        for (col, v) in ints {
            c.int(format_args!("trial {i} {col}"), v, ri(r, col)?);
        }
        for col in ["active_frame", "active_marker", "active_body"] {
            ri(r, col)?;
            selection += 1;
        }
        Ok(())
    });
    if selection > 0 {
        c.exclude(
            RULE_SELECTION,
            selection,
            format!(
                "active_frame, active_marker and active_body of every trial ({selection} \
                 cells) not compared: the interface's selection, which the model does not hold"
            ),
        );
    }
    c.finish()
}

/// `scheduler.csv`: the trials waiting for the scheduler's finalize.
fn check_scheduler(ds: &Dataset, doc: &Document) -> Item {
    let mut c = Check::new("scheduler.csv", Tier::Read);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let queued = &doc.scheduler().queued;
    c.rows(queued.len(), table.len());
    c.each_row(&table, |c, r| {
        let i = ru(r, "position")?;
        let t = get(queued, i, "queue position")?;
        c.int(
            format_args!("position {i} trial"),
            *t as i64,
            ri(r, "trial")?,
        );
        // `running` is set by every `updateTrialData` and cleared by finalize, which also
        // empties the queue: it is 1 whenever the queue holds a trial.
        c.int(format_args!("position {i} running"), 1, ri(r, "running")?);
        Ok(())
    });
    c.finish()
}

/// `rb_error_details.csv`: every mean and standard deviation, raw and filtered.
fn check_rb_error_details(ds: &Dataset, doc: &Document, t: usize, rel: &str) -> Item {
    let mut c = Check::new(rel, Tier::B);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let bodies = &doc.trial(t).rigid_bodies;
    c.rows(bodies.iter().map(|b| b.n_frames()).sum(), table.len());
    c.each_row(&table, |c, r| {
        let (b, f) = (ru(r, "body")?, ru(r, "frame")?);
        let body = get(bodies, b, "body")?;
        let raw = get(&body.raw.errors, f, "frame")?;
        let fil = get(&body.filtered.errors, f, "frame")?;
        let cols: [(&str, f64); 8] = [
            ("error3d_mean", raw.mean_3d),
            ("error3d_sd", raw.sd_3d),
            ("error2d_mean", raw.mean_2d),
            ("error2d_sd", raw.sd_2d),
            ("error3d_mean_filtered", fil.mean_3d),
            ("error3d_sd_filtered", fil.sd_3d),
            ("error2d_mean_filtered", fil.mean_2d),
            ("error2d_sd_filtered", fil.sd_2d),
        ];
        for (col, v) in cols {
            c.num(format_args!("body {b} frame {f} {col}"), v, rf(r, col)?);
        }
        Ok(())
    });
    c.finish()
}

/// `rb_state.csv`: `initialised` and the lengths of the arrays (the centre is not held
/// here: see [`RULE_CENTRE`]).
fn check_rb_state(ds: &Dataset, doc: &Document, t: usize, rel: &str) -> Item {
    let mut c = Check::new(rel, Tier::A);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let bodies = &doc.trial(t).rigid_bodies;
    c.rows(bodies.len(), table.len());
    let mut centres = 0;
    c.each_row(&table, |c, r| {
        let b = ru(r, "body")?;
        let body = get(bodies, b, "body")?;
        let ints: [(&str, i64); 5] = [
            ("initialised", body.initialised.into()),
            ("n_points", body.marker_indices.len() as i64),
            ("n_reference_points", body.reference_points.len() as i64),
            (
                "n_reference_points_original",
                body.reference_points_original.len() as i64,
            ),
            ("n_reference_names", body.reference_names.len() as i64),
        ];
        for (col, v) in ints {
            c.int(format_args!("body {b} {col}"), v, ri(r, col)?);
        }
        for axis in ["center_x", "center_y", "center_z"] {
            rf(r, axis)?;
        }
        centres += 1;
        Ok(())
    });
    if centres > 0 {
        c.exclude(
            RULE_CENTRE,
            3 * centres,
            format!(
                "center_x, center_y, center_z of {centres} bodies not compared: \
                 `RigidBody::center` is a drawing cache that only `setReferencesSet` refreshes \
                 (`updateCenter`, core/RigidBody.cpp:2598-2642), so it is stale after most \
                 changes of the reference points; the model does not hold it"
            ),
        );
    }
    c.finish()
}

/// Saves as the driver does and compares every archive entry with the C++ file's.
pub(super) fn compare_save(
    cpp_path: &Path,
    name: &str,
    doc: &mut Document,
    recompute_when_saving: bool,
    sizes: &SizeRules,
) -> Vec<Item> {
    let mut items = Vec::new();
    let mut head = Check::new(format!("{name} (entries)"), Tier::Read);
    let cpp_bytes = match std::fs::read(cpp_path) {
        Ok(b) => b,
        Err(e) => {
            head.problem(format!("cannot read {}: {e}", cpp_path.display()));
            items.push(head.finish());
            return items;
        }
    };
    let cpp = match Archive::new(&cpp_bytes) {
        Ok(a) => a,
        Err(e) => {
            head.problem(format!("{}: {e}", cpp_path.display()));
            items.push(head.finish());
            return items;
        }
    };
    let cpp_names: Vec<String> = cpp.entries().map(|e| e.decoded_name()).collect();
    let xml = cpp_names
        .iter()
        .position(|n| n == "project.xml")
        .and_then(|i| cpp.read(i).ok())
        .map(|d| String::from_utf8_lossy(&d).into_owned())
        .unwrap_or_default();
    let date = xml
        .find("DateCreated=\"")
        .map(|s| s + "DateCreated=\"".len())
        .and_then(|s| xml[s..].find('"').map(|l| xml[s..s + l].to_owned()))
        .unwrap_or_default();
    let Some(first) = cpp.entry(0) else {
        head.problem("the C++ archive is empty");
        items.push(head.finish());
        return items;
    };
    let options = SaveOptions {
        recompute_when_saving,
        time: SaveTime {
            date_created: date,
            modified: first.modified,
            utc_offset_seconds: 0,
        },
        trials: None,
    };
    let entries = match doc.save_entries(true, &options) {
        Ok((e, _)) => e,
        Err(e) => {
            head.problem(format!("the save fails here: {e}"));
            items.push(head.finish());
            return items;
        }
    };
    let ours: Vec<&str> = entries.files.iter().map(|(n, _)| n.as_str()).collect();
    let settings_last = ours.last() == Some(&xma_io::settings_entry::SETTINGS_ENTRY);
    let ours_cpp: Vec<&str> = if settings_last {
        ours[..ours.len() - 1].to_vec()
    } else {
        ours.clone()
    };
    if !settings_last {
        head.problem("the settings entry is not last");
    }
    let theirs: Vec<&str> = cpp_names.iter().map(String::as_str).collect();
    // An entry the C++ regenerates at every save and this program carries over (ledger
    // F24): a project made here has none to carry (`xma_metadata.xml`).
    let missing_regenerated: Vec<&str> = theirs
        .iter()
        .copied()
        .filter(|n| REGENERATED_BY_CPP.contains(n) && !ours_cpp.contains(n))
        .collect();
    let theirs: Vec<&str> = theirs
        .into_iter()
        .filter(|n| !missing_regenerated.contains(n))
        .collect();
    for n in &missing_regenerated {
        head.exclude(
            RULE_REGENERATED,
            1,
            format!(
                "{n}: not written here: the C++ regenerates it at every save, this program \
                 carries it over from the file it read (ledger F24), and a project created \
                 here has none"
            ),
        );
    }
    if ours_cpp != theirs {
        let only_here: Vec<&&str> = ours_cpp.iter().filter(|n| !theirs.contains(n)).collect();
        let only_there: Vec<&&str> = theirs.iter().filter(|n| !ours_cpp.contains(n)).collect();
        head.problem(format!(
            "entry names or order differ: only here {only_here:?}, only in the C++ file \
             {only_there:?}"
        ));
    }
    head.int("entries", ours_cpp.len() as i64, theirs.len() as i64);
    items.push(head.finish());
    for (i, n) in cpp_names.iter().enumerate() {
        let rel = format!("{name}:{n}");
        let mut c = Check::new(rel, Tier::Read);
        if REGENERATED_BY_CPP.contains(&n.as_str()) {
            c.exclude(
                RULE_REGENERATED,
                1,
                format!(
                    "{n}: not compared: rebuilt by the C++ at every save from state the model \
                     does not hold (ledger F24)"
                ),
            );
            items.push(c.finish());
            continue;
        }
        if let Some((_, m, rule, why)) = sizes
            .skip
            .iter()
            .find(|(_, m, _, _)| n.ends_with(&format!("/data/Marker{m:03}size.csv")))
        {
            c.exclude(
                rule,
                1,
                format!("{n}: not compared: the sizes of marker {m}: {why}"),
            );
            items.push(c.finish());
            continue;
        }
        let want = cpp.read(i).unwrap_or_default();
        match entries.files.iter().find(|(m, _)| m == n) {
            None => c.problem("not written here"),
            Some((_, got))
                if *got != want && sizes.tolerance.is_some() && n.ends_with("size.csv") =>
            {
                numbers_within(&mut c, got, &want, sizes.tolerance);
            }
            Some((_, got))
                if *got != want
                    && sizes.calibration_tolerance.is_some()
                    && (n.ends_with("_CameraMatrix.csv")
                        || n.ends_with("_RotationMatrix.csv")
                        || n.ends_with("_TranslationVector.csv")
                        || n.ends_with("_UndistortionParameter.csv")) =>
            {
                numbers_within(&mut c, got, &want, sizes.calibration_tolerance);
            }
            Some((_, got))
                if *got != want
                    && sizes.calibration_tolerance.is_some()
                    && n == "xma_metadata.xml" =>
            {
                // The calibration's errors in the Portal metadata: compared as numbers.
                xml_numbers_within(&mut c, got, &want, sizes.calibration_tolerance);
            }
            Some((_, got)) if *got != want && is_image_entry(n) => {
                same_pixels(&mut c, n, got, &want);
            }
            Some((_, got)) => {
                if *got == want {
                    c.bytes_equal = Some(true);
                } else {
                    c.bytes_equal = Some(false);
                    let (g, w) = (String::from_utf8_lossy(got), String::from_utf8_lossy(&want));
                    let at = g
                        .lines()
                        .zip(w.lines())
                        .position(|(a, b)| a != b)
                        .unwrap_or(g.lines().count().min(w.lines().count()));
                    c.problem(format!(
                        "differs from line {}: here '{}', C++ '{}' ({} and {} bytes)",
                        at + 1,
                        g.lines()
                            .nth(at)
                            .unwrap_or("")
                            .chars()
                            .take(160)
                            .collect::<String>(),
                        w.lines()
                            .nth(at)
                            .unwrap_or("")
                            .chars()
                            .take(160)
                            .collect::<String>(),
                        got.len(),
                        want.len()
                    ));
                }
            }
        }
        items.push(c.finish());
    }
    items
}

const RULE_REENCODED: &str = "image re-encoded by XMALab's OpenCV at save: other bytes, the \
     same pixels (ledger V6)";

/// An archive entry holding a still image (a calibration or grid image).
fn is_image_entry(name: &str) -> bool {
    let l = name.to_ascii_lowercase();
    [".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg"]
        .iter()
        .any(|e| l.ends_with(e))
}

/// An image entry whose bytes differ: compared by its decoded pixels (XMALab writes every
/// still image again with OpenCV, ledger V6; this program keeps an 8-bit file as it is).
fn same_pixels(c: &mut Check, name: &str, got: &[u8], want: &[u8]) {
    let p = Path::new(name);
    match (
        xma_video::image_file::decode_image(got, p),
        xma_video::image_file::decode_image(want, p),
    ) {
        (Ok(a), Ok(b)) if a.width == b.width && a.height == b.height && a.pixels == b.pixels => {
            c.exclude(
                RULE_REENCODED,
                1,
                format!(
                    "{name}: {} x {} pixels identical; the files differ ({} and {} bytes)",
                    a.width,
                    a.height,
                    got.len(),
                    want.len()
                ),
            );
        }
        (Ok(a), Ok(b)) => c.problem(format!(
            "{name}: the pixels differ ({} x {} here, {} x {} in the C++ file)",
            a.width, a.height, b.width, b.height
        )),
        (a, b) => c.problem(format!(
            "{name}: cannot compare the images (here: {}, C++: {})",
            a.err().map_or("decoded".to_owned(), |e| e.to_string()),
            b.err().map_or("decoded".to_owned(), |e| e.to_string())
        )),
    }
}

/// Two text files compared cell by cell (lines split at `\n` only, so that a carriage
/// return or a missing final line end is a difference; cells at commas): text cells
/// exactly, numbers within the named tolerance `tol`. Equal numbers written differently
/// (`0.1` against `0.10000000000000001`, `-0` against `0`) are a problem, and so are white
/// space around a number cell that differs (whether or not the number is tolerated) and
/// bytes that differ where no cell does. Also used for the MayaCam files of a calibration
/// image run (`calibration::images`).
pub(super) fn numbers_within(c: &mut Check, got: &[u8], want: &[u8], tol: Option<NamedTolerance>) {
    let (g, w) = (String::from_utf8_lossy(got), String::from_utf8_lossy(want));
    let (gl, wl): (Vec<&str>, Vec<&str>) = (g.split('\n').collect(), w.split('\n').collect());
    c.rows(gl.len(), wl.len());
    for (i, (a, b)) in gl.iter().zip(&wl).enumerate() {
        let (ac, bc): (Vec<&str>, Vec<&str>) = (a.split(',').collect(), b.split(',').collect());
        if ac.len() != bc.len() {
            c.problem(format!("line {}: here '{a}', C++ '{b}'", i + 1));
            continue;
        }
        for (j, (x, y)) in ac.iter().zip(&bc).enumerate() {
            match (xma_oracle::parse_f64(x), xma_oracle::parse_f64(y)) {
                (Some(p), Some(q)) => {
                    let label = format!("line {} cell {}", i + 1, j + 1);
                    same_spacing(c, &label, x, y);
                    same_text(c, &label, x.trim(), y.trim(), p, q);
                    c.num_named(label, p, q, tol);
                }
                _ => c.text(format_args!("line {} cell {}", i + 1, j + 1), x, y),
            }
        }
    }
    bytes_differ(c, got, want);
}

/// An XML text compared line by line (split at `\n` only), the attribute values (between
/// quotes) that are numbers as numbers within `tol`, everything else as text; equal numbers
/// written differently, white space around a number value that differs (tolerated or not),
/// and bytes that differ where no part does, are a problem.
fn xml_numbers_within(c: &mut Check, got: &[u8], want: &[u8], tol: Option<NamedTolerance>) {
    let (g, w) = (String::from_utf8_lossy(got), String::from_utf8_lossy(want));
    let (gl, wl): (Vec<&str>, Vec<&str>) = (g.split('\n').collect(), w.split('\n').collect());
    c.rows(gl.len(), wl.len());
    for (i, (a, b)) in gl.iter().zip(&wl).enumerate() {
        let (ap, bp): (Vec<&str>, Vec<&str>) = (a.split('"').collect(), b.split('"').collect());
        if ap.len() != bp.len() {
            c.problem(format!("line {}: here '{a}', C++ '{b}'", i + 1));
            continue;
        }
        for (j, (x, y)) in ap.iter().zip(&bp).enumerate() {
            match (xma_oracle::parse_f64(x), xma_oracle::parse_f64(y)) {
                (Some(p), Some(q)) if j % 2 == 1 => {
                    let label = format!("line {} value {}", i + 1, j / 2 + 1);
                    same_spacing(c, &label, x, y);
                    same_text(c, &label, x.trim(), y.trim(), p, q);
                    c.num_named(label, p, q, tol);
                }
                _ => c.text(format_args!("line {} part {}", i + 1, j + 1), x, y),
            }
        }
    }
    bytes_differ(c, got, want);
}

/// The white space around two number cells (which [`xma_oracle::parse_f64`] trims) differs:
/// a problem, as [`super::export_file_impl`] reports it, whether or not the numbers are
/// within a tolerance.
fn same_spacing(c: &mut Check, label: impl std::fmt::Display, x: &str, y: &str) {
    let around = |s: &str| {
        let t = s.trim();
        let start = s.len() - s.trim_start().len();
        (s[..start].to_owned(), s[start + t.len()..].to_owned())
    };
    if around(x) != around(y) {
        c.problem(format!(
            "{label}: '{x}', C++ '{y}': the spaces around the value differ"
        ));
    }
}

/// Two cells that parse to the same number `p`, `q` but are written differently: a problem
/// (as [`super::export_file_impl`] reports it).
fn same_text(c: &mut Check, label: impl std::fmt::Display, x: &str, y: &str, p: f64, q: f64) {
    if super::same(p, q) && x != y {
        c.problem(format!(
            "{label}: '{x}', C++ '{y}': equal numbers, but the bytes differ (number formatting)"
        ));
    }
}

/// The bytes of two entries differ but no problem and no value within the tolerance was
/// recorded: a problem.
fn bytes_differ(c: &mut Check, got: &[u8], want: &[u8]) {
    if got != want && c.n_problems == 0 && c.tolerated.is_none() && c.cmp.max_abs() == 0.0 {
        c.problem(format!(
            "the bytes differ ({} and {} bytes), but no value does",
            got.len(),
            want.len()
        ));
    }
}

/// Entries XMALab rebuilds at every save from state the model does not hold (ledger F24).
const REGENERATED_BY_CPP: [&str; 1] = ["log.html"];
const RULE_REGENERATED: &str = "not compared: rebuilt by the C++ at save, ledger F24";

#[cfg(test)]
mod review12_tests {
    //! Review of 20 November 2026, second round, finding 1 of `parity-xtask.md`: saved
    //! entries compared within a tolerance must not hide formatting and line ends.

    use super::*;
    use crate::parity::TOLERANCE_A37_CUBE;

    fn cells(got: &str, want: &str) -> Item {
        let mut c = Check::new("t", Tier::Read);
        numbers_within(
            &mut c,
            got.as_bytes(),
            want.as_bytes(),
            Some(TOLERANCE_A37_CUBE),
        );
        c.finish()
    }

    fn xml(got: &str, want: &str) -> Item {
        let mut c = Check::new("t", Tier::Read);
        xml_numbers_within(
            &mut c,
            got.as_bytes(),
            want.as_bytes(),
            Some(TOLERANCE_A37_CUBE),
        );
        c.finish()
    }

    #[test]
    fn review12_numbers_written_differently_are_not_identical() {
        for (g, w) in [
            ("0.1\n", "0.10000000000000001\n"),
            ("-0\n", "0\n"),
            ("1e-05\n", "1.0000000000000001e-05\n"),
            ("1, 2\n", "1,2\n"),
        ] {
            let i = cells(g, w);
            assert_eq!(i.verdict, Verdict::Different, "{g:?} against {w:?}: {i:?}");
        }
    }

    #[test]
    fn review12_line_ends_are_not_identical() {
        for (g, w) in [("1,2\r\n", "1,2\n"), ("1,2", "1,2\n"), ("1,2\n\n", "1,2\n")] {
            let i = cells(g, w);
            assert_eq!(i.verdict, Verdict::Different, "{g:?} against {w:?}: {i:?}");
        }
        // A tolerated value does not excuse a carriage return.
        let i = cells("1.0000000001,2\r\n", "1,2\n");
        assert_eq!(i.verdict, Verdict::Different, "{i:?}");
        // A tolerated value alone is still within the tolerance.
        let i = cells("1.0000000001,2\n", "1,2\n");
        assert_eq!(i.verdict, Verdict::WithinTolerance, "{i:?}");
    }

    #[test]
    fn review12_xml_numbers_written_differently_are_not_identical() {
        for (g, w) in [
            ("<a e=\"0.1\"/>\n", "<a e=\"0.10000000000000001\"/>\n"),
            ("<a e=\"1\"/>\r\n", "<a e=\"1\"/>\n"),
            ("<a e=\"1\"/>", "<a e=\"1\"/>\n"),
        ] {
            let i = xml(g, w);
            assert_eq!(i.verdict, Verdict::Different, "{g:?} against {w:?}: {i:?}");
        }
        let i = xml("<a e=\"1.0000000001\"/>\n", "<a e=\"1\"/>\n");
        assert_eq!(i.verdict, Verdict::WithinTolerance, "{i:?}");
    }

    #[test]
    fn review12_not_applied_fails_the_edit_run() {
        let report = EditParityReport {
            run: EditRun {
                name: "t".to_owned(),
                start_project: PathBuf::new(),
                script: PathBuf::new(),
                output: PathBuf::new(),
            },
            ops: vec![OpRow {
                line: 7,
                op: "TrackForward".to_owned(),
                cpp_status: "ok".to_owned(),
                cpp_messages: String::new(),
                status: "not applied".to_owned(),
                message: "tracking is not ported".to_owned(),
                agrees: false,
            }],
            points: Vec::new(),
            warnings: Vec::new(),
            seconds: 0.0,
        };
        assert!(!report.passed(), "{}", report.summary());
        assert!(
            report.summary().contains("line 7") && report.summary().contains("TrackForward"),
            "{}",
            report.summary()
        );
    }
}

#[cfg(test)]
mod review12c_tests {
    //! Review of 20 November 2026, third round: a tolerated cell must not hide the white
    //! space around it.

    use super::*;
    use crate::parity::TOLERANCE_A37_CUBE;

    fn cells(got: &str, want: &str) -> Item {
        let mut c = Check::new("t", Tier::Read);
        numbers_within(
            &mut c,
            got.as_bytes(),
            want.as_bytes(),
            Some(TOLERANCE_A37_CUBE),
        );
        c.finish()
    }

    fn xml(got: &str, want: &str) -> Item {
        let mut c = Check::new("t", Tier::Read);
        xml_numbers_within(
            &mut c,
            got.as_bytes(),
            want.as_bytes(),
            Some(TOLERANCE_A37_CUBE),
        );
        c.finish()
    }

    #[test]
    fn review12c_space_around_a_tolerated_cell_is_a_difference() {
        for (g, w) in [
            ("1.0000000001 ,2\n", "1,2\n"),
            ("\t1.0000000001,2\n", "1,2\n"),
            ("1.0000000001,2 \n", "1,2\n"),
            ("1.0000000001,2\n", "1, 2\n"),
        ] {
            let i = cells(g, w);
            assert_eq!(i.verdict, Verdict::Different, "{g:?} against {w:?}: {i:?}");
            assert!(
                i.problems.iter().any(|p| p.contains("space")),
                "{g:?} against {w:?}: {i:?}"
            );
        }
        // The same spacing on both sides, a tolerated value: still within the tolerance.
        let i = cells("1.0000000001, 2\n", "1, 2\n");
        assert_eq!(i.verdict, Verdict::WithinTolerance, "{i:?}");
    }

    #[test]
    fn review12c_space_around_a_tolerated_xml_value_is_a_difference() {
        for (g, w) in [
            ("<a e=\"1.0000000001 \"/>\n", "<a e=\"1\"/>\n"),
            ("<a e=\"\t1.0000000001\"/>\n", "<a e=\"1\"/>\n"),
        ] {
            let i = xml(g, w);
            assert_eq!(i.verdict, Verdict::Different, "{g:?} against {w:?}: {i:?}");
        }
        let i = xml("<a e=\" 1.0000000001\"/>\n", "<a e=\" 1\"/>\n");
        assert_eq!(i.verdict, Verdict::WithinTolerance, "{i:?}");
    }
}
