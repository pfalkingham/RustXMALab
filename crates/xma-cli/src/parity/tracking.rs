//! The tracking reference: whole runs of XMALab's own tracking loop
//! (`oracle/tracking_runs.json`; `oracle/FORMAT.md` "Tracking runs" and "State dump";
//! `oracle/DATASETS.md` "Tracking references").
//!
//! [`run_tracking_parity`] opens the run's start project in the XMALab 3 mode with the
//! settings the run's `manifest.json` records (the values in force during the run, its
//! overrides included), makes the same request (type, markers, start and end frame,
//! direction, one step or continuous) and runs it step by step. It compares:
//!
//! - after every step, the document with `run_log.csv`: the set of (marker, frame) the step
//!   changed (anything the driver watches: a 2D point, status or size in any camera, a 3D
//!   point or status) together with the step's markers at the frame tracked to, and every
//!   logged value (`status`, `x`, `y`, `size`, `status3d`, `x3d`, `y3d`, `z3d`,
//!   `mean_size`, `size_range`, `changed2d`, `changed3d`), exactly;
//! - `run_steps.csv`: per step the frames from and to, the number of 2D cells changed and
//!   the number of merge messages (none: no kind that is still compared writes one); the
//!   run's number of steps and last frame (`manifest.json`);
//! - the run's Console lines with `merge_log.txt`, line for line (empty on both sides for
//!   every kind still compared);
//! - at the end, the whole document with the state dump (`state/`, the comparisons of
//!   [`super::edits`]) and the project saved as the driver saves it with XMALab's
//!   `project.xma`, entry by entry.
//!
//! One rule takes values out of the comparison: a marker whose detection method is 4 ("no
//! detection") gets its sizes from uninitialised memory in XMALab (ledger A22; this program
//! writes -1). Its 2D sizes and size range, the `changed2d` flags and changed (marker,
//! frame) set of its rows (which follow those sizes), the step's `cells_changed` count where
//! they enter it, and its `Marker<n>size.csv` in the saved project are not compared, and the
//! report says so. Of the reference runs, `r11_methods_t3` has such a marker.
//!
//! A run of type 5 or 6 (merge-aware tracking) is not compared: that kind was removed from
//! this program on 26 November 2026 (parity ledger A226). Its report lists it under "Not
//! compared" with the reason ([`MERGE_REMOVED`]), runs nothing and is not a failure; its
//! reference outputs stay on disk.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use xma_engine::tracking::{
    Direction, StepReport, StopReason, TrackKind, TrackRequest, Tracker, TrialSources,
};
use xma_engine::{Document, OpenOptions};
use xma_model::{CppCompat, DetectionMethod, Trial};
use xma_oracle::{JsonValue, Manifest, Table};

use super::edits::{SizeRules, Uninit, compare_save, compare_state, loaded_uninit};
use super::{Check, Item, NotCompared, Tier, Verdict, cell, num};

/// Why a run of type 5 or 6 is not compared.
pub const MERGE_REMOVED: &str =
    "merge-aware tracking was removed from this program on 26 November 2026 (parity ledger A226)";

/// Rule of ledger A22 (sizes of a method 4 marker).
pub(super) const RULE_A22: &str = "not compared: C++ size uninitialised (method 4), ledger A22";
const WHY_A22: &str = "ledger A22: with detection method 4 XMALab writes the size from \
                       uninitialised memory (MarkerDetection::m_size); this program writes -1";

/// One run of `oracle/tracking_runs.json`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrackingRun {
    pub name: String,
    pub description: String,
    pub start_project: PathBuf,
    /// The folder of repeat `a` (`project.xma`, `run_log.csv`, `state/`, ...).
    pub run_dir: PathBuf,
    /// 1 single point, 2 selected, 3 all, 5 merge-aware selected, 6 merge-aware all (types
    /// 5 and 6 are listed but no longer compared, [`TrackingRun::not_compared`]).
    pub track_type: i64,
    pub markers: Vec<usize>,
    pub start: usize,
    pub forward: bool,
    pub single: bool,
    pub end: Option<usize>,
    /// The C++ run's number of steps, last frame and why it stopped.
    pub steps: usize,
    pub last_frame: usize,
    pub stop_reason: String,
    /// The two repeats of the C++ run were identical.
    pub repeatable: bool,
}

impl TrackingRun {
    /// True when the start project and the run's folder exist.
    #[must_use]
    pub fn is_present(&self) -> bool {
        self.start_project.is_file() && self.run_dir.join("run_log.csv").is_file()
    }

    /// Why the run is not compared, when it is not: types 5 and 6 (merge-aware tracking,
    /// removed on 26 November 2026, parity ledger A226).
    #[must_use]
    pub fn not_compared(&self) -> Option<&'static str> {
        matches!(self.track_type, 5 | 6).then_some(MERGE_REMOVED)
    }

    /// The request the C++ driver made (trial 0, every camera shown).
    ///
    /// # Errors
    /// A type this program does not have (among them 5 and 6, merge-aware tracking, removed
    /// on 26 November 2026), or one without its markers.
    pub fn request(&self) -> Result<TrackRequest, String> {
        let kind = match self.track_type {
            1 => TrackKind::Single(
                *self
                    .markers
                    .first()
                    .ok_or_else(|| format!("{}: type 1 without a marker", self.name))?,
            ),
            2 => TrackKind::Selected(self.markers.clone()),
            3 => TrackKind::All,
            5 | 6 => return Err(format!("{}: {MERGE_REMOVED}", self.name)),
            t => return Err(format!("{}: tracking type {t}", self.name)),
        };
        Ok(TrackRequest {
            trial: 0,
            kind,
            direction: if self.forward {
                Direction::Forward
            } else {
                Direction::Backward
            },
            continuous: !self.single,
            start_frame: self.start,
            end_frame: self.end,
            visible_cameras: Vec::new(),
        })
    }

    /// The markers the request tracks at each step.
    fn step_markers(&self, n_markers: usize) -> Vec<usize> {
        match self.track_type {
            1 => self.markers.iter().take(1).copied().collect(),
            2 | 5 => self.markers.clone(),
            _ => (0..n_markers).collect(),
        }
    }

    /// The request in words.
    #[must_use]
    pub fn request_text(&self) -> String {
        let kind = match self.track_type {
            1 => "single point",
            2 => "selected points",
            3 => "all points",
            5 => "merge-aware selected",
            6 => "merge-aware all",
            _ => "?",
        };
        let markers = if self.markers.is_empty() {
            String::new()
        } else {
            format!(
                ", markers {}",
                self.markers
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        format!(
            "type {} ({kind}{markers}), {} from frame {}{}{}",
            self.track_type,
            if self.forward { "forward" } else { "backward" },
            self.start,
            self.end
                .map(|e| format!(" to frame {e}"))
                .unwrap_or_default(),
            if self.single { ", one step" } else { "" }
        )
    }
}

/// The index file of the tracking runs.
fn index() -> PathBuf {
    xma_oracle::oracle_root().join("tracking_runs.json")
}

/// The runs listed in `oracle/tracking_runs.json`.
///
/// # Errors
/// When the file cannot be read or does not have the expected layout.
pub fn tracking_runs() -> Result<Vec<TrackingRun>, String> {
    let root = xma_oracle::oracle_root();
    let path = index();
    let m = Manifest::try_read(&path).map_err(|e| e.to_string())?;
    let runs = m
        .json()
        .get("runs")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| format!("{}: no 'runs' list", path.display()))?;
    let mut out = Vec::new();
    for r in runs {
        let text = |k: &str| -> Result<String, String> {
            r.get(k)
                .and_then(JsonValue::as_str)
                .map(str::to_owned)
                .ok_or_else(|| format!("{}: a run without '{k}'", path.display()))
        };
        let count = |k: &str| -> Result<usize, String> {
            r.get(k)
                .and_then(JsonValue::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| format!("{}: a run without '{k}'", path.display()))
        };
        let name = text("name")?;
        let opts: Vec<String> = r
            .get("options")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| format!("{name}: no options"))?
            .iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect();
        let opt = |k: &str| -> Option<&str> {
            opts.iter()
                .position(|o| o == k)
                .and_then(|i| opts.get(i + 1))
                .map(String::as_str)
        };
        let number = |k: &str| -> Result<Option<usize>, String> {
            opt(k)
                .map(|v| v.parse().map_err(|_| format!("{name}: {k} {v}")))
                .transpose()
        };
        let result = root.join(text("result_project")?);
        out.push(TrackingRun {
            description: text("description").unwrap_or_default(),
            start_project: root.join(text("start_project")?),
            run_dir: result
                .parent()
                .map_or_else(|| root.clone(), Path::to_path_buf),
            track_type: opt("--type")
                .and_then(|v| v.parse().ok())
                .ok_or_else(|| format!("{name}: no --type"))?,
            markers: match opt("--markers") {
                Some(v) => v
                    .split(',')
                    .map(|x| x.parse().map_err(|_| format!("{name}: --markers {v}")))
                    .collect::<Result<_, _>>()?,
                None => Vec::new(),
            },
            start: number("--start")?.ok_or_else(|| format!("{name}: no --start"))?,
            forward: opt("--direction") != Some("backward"),
            single: opts.iter().any(|o| o == "--single"),
            end: number("--end")?,
            steps: count("steps")?,
            last_frame: count("last_frame")?,
            stop_reason: text("stop_reason").unwrap_or_default(),
            repeatable: r
                .get("repeatability")
                .and_then(|v| v.get("identical"))
                .and_then(JsonValue::as_bool)
                .unwrap_or(false),
            name,
        });
    }
    Ok(out)
}

/// The run of that name, if `oracle/tracking_runs.json` lists it.
#[must_use]
pub fn find_tracking_run(name: &str) -> Option<TrackingRun> {
    tracking_runs().ok()?.into_iter().find(|r| r.name == name)
}

/// A path named in `oracle/tracking_runs.json` (`reference_project`, `full_step_dump`).
#[must_use]
pub fn index_path(key: &str) -> Option<PathBuf> {
    let m = Manifest::try_read(index()).ok()?;
    let rel = m.json().get(key)?.as_str()?.to_owned();
    Some(xma_oracle::oracle_root().join(rel))
}

// ---------------------------------------------------------------------------------------
// Running a request
// ---------------------------------------------------------------------------------------

/// One step as it ran here.
#[derive(Clone, Debug)]
pub struct StepRun {
    /// From 1.
    pub step: usize,
    pub frame_from: usize,
    pub frame_to: usize,
    pub report: StepReport,
}

/// A run made here.
pub struct Tracked {
    pub doc: Document,
    pub steps: usize,
    pub last_frame: usize,
    pub stop: Option<StopReason>,
    /// The run's Console lines, as `merge_log.txt` writes them.
    pub messages: Vec<String>,
}

/// Opens a run's start project as the C++ driver does, with the run's settings, in mode
/// `compat`: the GUI's load (with the active trial's update), then the trial selected as
/// the trial selector does (`TrackOracleDriver::startRun`: `State::changeActiveTrial(trial,
/// true)`, whose handler updates the trial; with nothing to recompute the update starts no
/// job, so the trial stays queued in the scheduler, as the C++ state dump shows).
///
/// # Errors
/// When the manifest or the project cannot be read.
pub fn open_start(run: &TrackingRun, compat: CppCompat) -> Result<Document, String> {
    let manifest =
        Manifest::try_read(run.run_dir.join("manifest.json")).map_err(|e| e.to_string())?;
    let options = OpenOptions {
        overrides: crate::settings::manifest_settings(&manifest)?,
        compat,
        ..OpenOptions::default()
    };
    let loaded = xma_engine::read_project(&run.start_project, &options)
        .map_err(|e| format!("{}: {e}", run.start_project.display()))?;
    let mut doc = Document::from_loaded(loaded);
    doc.set_path(&run.start_project);
    doc.update_after_load();
    doc.update_trial(0);
    Ok(doc)
}

/// Runs the run's request in mode `compat`, one step at a time; `on_step` sees the trial
/// before each step and the document after it.
///
/// # Errors
/// When the project or its images cannot be opened, or the request does not start.
pub fn track_run(
    run: &TrackingRun,
    compat: CppCompat,
    on_step: &mut dyn FnMut(&StepRun, &Trial, &Document),
) -> Result<Tracked, String> {
    let mut doc = open_start(run, compat)?;
    let mut images = TrialSources::new(&doc, 0)?;
    let mut tracker = Tracker::new(&doc, run.request()?).map_err(|e| e.to_string())?;
    doc.begin_step("Track Points");
    let mut messages = Vec::new();
    let mut from = run.start;
    loop {
        let before = doc.trial(0).clone();
        let done = tracker.steps_done();
        let report = tracker.step(&mut doc, &mut images);
        if tracker.steps_done() == done {
            break;
        }
        let step = StepRun {
            step: tracker.steps_done(),
            frame_from: from,
            frame_to: report.frame,
            report,
        };
        for m in &step.report.messages {
            messages.push(format!(
                "step {} (frame {} -> {}): {m}",
                step.step, step.frame_from, step.frame_to
            ));
        }
        on_step(&step, &before, &doc);
        from = step.frame_to;
        if step.report.finished.is_some() {
            break;
        }
    }
    doc.commit_step();
    Ok(Tracked {
        steps: tracker.steps_done(),
        last_frame: tracker.frame(),
        stop: tracker.finished().cloned(),
        doc,
        messages,
    })
}

/// A stop reason in words.
#[must_use]
pub fn stop_text(s: Option<&StopReason>) -> String {
    match s {
        Some(StopReason::SingleStep) => "one step done".to_owned(),
        Some(StopReason::Boundary) => "reached the trial's first or last frame".to_owned(),
        Some(StopReason::EndFrame) => "reached the end frame requested".to_owned(),
        Some(StopReason::InvalidPoint) => "the tracked point is not valid".to_owned(),
        Some(StopReason::Stopped) => "stopped".to_owned(),
        Some(StopReason::Failed(t)) => format!("failed: {t}"),
        None => "not finished".to_owned(),
    }
}

// ---------------------------------------------------------------------------------------
// The comparison
// ---------------------------------------------------------------------------------------

/// The result of [`run_tracking_parity`].
#[derive(Clone, Debug)]
pub struct TrackingParityReport {
    pub run: TrackingRun,
    /// Steps, last frame and why the run stopped here.
    pub steps: usize,
    pub last_frame: usize,
    pub stop: String,
    pub items: Vec<Item>,
    /// The run itself when it is not compared ([`TrackingRun::not_compared`]), with the
    /// reason; nothing is run then and `items` is empty.
    pub not_compared: Vec<NotCompared>,
    pub warnings: Vec<String>,
    pub seconds: f64,
}

impl TrackingParityReport {
    /// No item is different.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.items.iter().all(|i| i.verdict != Verdict::Different)
    }

    #[must_use]
    pub fn count(&self, v: Verdict) -> usize {
        self.items.iter().filter(|i| i.verdict == v).count()
    }

    /// Values compared over every item.
    #[must_use]
    pub fn values(&self) -> usize {
        self.items.iter().map(|i| i.values).sum()
    }

    /// Cells handled by a rule.
    #[must_use]
    pub fn explained_cells(&self) -> usize {
        self.items.iter().map(|i| i.excluded + i.explained).sum()
    }

    /// The names of the items that differ.
    #[must_use]
    pub fn different(&self) -> Vec<String> {
        self.items
            .iter()
            .filter(|i| i.verdict == Verdict::Different)
            .map(|i| i.name.clone())
            .collect()
    }

    /// One line.
    #[must_use]
    pub fn summary(&self) -> String {
        if let Some(n) = self.not_compared.first() {
            return format!("{}: NOT COMPARED — {}", self.run.name, n.reason);
        }
        format!(
            "{}: {} — {} steps to frame {} (XMALab {} to frame {}); {} tables and files \
             ({} identical, {} identical apart from explained cells, {} empty on both sides, \
             {} within tolerance, {} different), {} values; {} cell(s) explained",
            self.run.name,
            if self.passed() { "PASS" } else { "FAIL" },
            self.steps,
            self.last_frame,
            self.run.steps,
            self.run.last_frame,
            self.items.len(),
            self.count(Verdict::Identical),
            self.count(Verdict::IdenticalExplained),
            self.count(Verdict::BothEmpty),
            self.count(Verdict::WithinTolerance),
            self.count(Verdict::Different),
            self.values(),
            self.explained_cells()
        )
    }

    /// The report as Markdown.
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "# Tracking parity report: {}\n", self.run.name);
        if !self.not_compared.is_empty() {
            let _ = writeln!(s, "**Result: not compared.** Nothing was run.\n");
            if !self.run.description.is_empty() {
                let _ = writeln!(s, "{}\n", self.run.description);
            }
            let _ = writeln!(s, "- Request: {} (frames 0-based)", self.run.request_text());
            let _ = writeln!(s, "- Reference run: `{}`", self.run.run_dir.display());
            let _ = writeln!(s, "\n## Not compared\n");
            let _ = writeln!(s, "| File | Why |\n|---|---|");
            for n in &self.not_compared {
                let _ = writeln!(s, "| `{}` | {} |", n.path, cell(&n.reason));
            }
            return s;
        }
        let _ = writeln!(
            s,
            "**Result: {}.** {} tables and files compared: {} identical, {} identical apart \
             from explained cells, {} empty on both sides, {} within tolerance, {} different; \
             {} values in all; {} cell(s) explained by a rule (not compared; see the notes).\n",
            if self.passed() { "PASS" } else { "FAIL" },
            self.items.len(),
            self.count(Verdict::Identical),
            self.count(Verdict::IdenticalExplained),
            self.count(Verdict::BothEmpty),
            self.count(Verdict::WithinTolerance),
            self.count(Verdict::Different),
            self.values(),
            self.explained_cells()
        );
        if !self.run.description.is_empty() {
            let _ = writeln!(s, "{}\n", self.run.description);
        }
        let _ = writeln!(s, "- Request: {} (frames 0-based)", self.run.request_text());
        let _ = writeln!(
            s,
            "- XMALab: {} steps to frame {} ({}); repeats {}",
            self.run.steps,
            self.run.last_frame,
            self.run.stop_reason,
            if self.run.repeatable {
                "identical"
            } else {
                "not identical (values from uninitialised memory)"
            }
        );
        let _ = writeln!(
            s,
            "- Here: {} steps to frame {} ({})",
            self.steps, self.last_frame, self.stop
        );
        let _ = writeln!(s, "- Start project: `{}`", self.run.start_project.display());
        let _ = writeln!(s, "- Reference run: `{}`", self.run.run_dir.display());
        let _ = writeln!(
            s,
            "- Compatibility mode: {} (`xma parity` always compares in the XMALab 3 mode)",
            CppCompat::XMALAB3.describe()
        );
        let _ = writeln!(
            s,
            "- Compared by: xma {} (engine {}), in {:.1} s\n",
            crate::VERSION,
            xma_engine::VERSION,
            self.seconds
        );
        let _ = writeln!(s, "## Compared\n");
        let _ = writeln!(
            s,
            "| Table or file | Values | Max abs diff | Where | Verdict |\n|---|---:|---:|---|---|"
        );
        for i in &self.items {
            let _ = writeln!(
                s,
                "| `{}` | {} | {} | {} | {} |",
                i.name,
                i.values,
                num(i.max_abs),
                cell(&i.max_abs_at),
                cell(&i.verdict_label())
            );
        }
        let flagged: Vec<&Item> = self
            .items
            .iter()
            .filter(|i| !i.problems.is_empty() || !i.notes.is_empty())
            .collect();
        if !flagged.is_empty() {
            let _ = writeln!(s, "\n### Differences and notes\n");
            for i in flagged {
                let _ = writeln!(s, "- `{}`:", i.name);
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

/// Bit-for-bit equality, NaN equal to NaN.
fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// What changed between two states of a trial: the (marker, frame) pairs where anything the
/// run log watches changed, the (marker, camera, frame) whose 2D point, status or size
/// changed, and the (marker, frame) whose 3D point or status changed. The sizes of the
/// markers in `skip` are not looked at.
type Changes = (
    BTreeSet<(usize, usize)>,
    BTreeSet<(usize, usize, usize)>,
    BTreeSet<(usize, usize)>,
);

fn changes(before: &Trial, after: &Trial, skip: &[usize]) -> Changes {
    let (mut any, mut d2, mut d3) = (BTreeSet::new(), BTreeSet::new(), BTreeSet::new());
    for (m, (a, b)) in before.markers.iter().zip(&after.markers).enumerate() {
        for f in 0..a.n_frames().min(b.n_frames()) {
            let p3 = !same(a.points3d[f].x, b.points3d[f].x)
                || !same(a.points3d[f].y, b.points3d[f].y)
                || !same(a.points3d[f].z, b.points3d[f].z)
                || a.status3d[f] != b.status3d[f];
            if p3 {
                d3.insert((m, f));
                any.insert((m, f));
            }
            for (c, (va, vb)) in a.views.iter().zip(&b.views).enumerate() {
                let p2 = !same(va.points[f].x, vb.points[f].x)
                    || !same(va.points[f].y, vb.points[f].y)
                    || va.status[f] != vb.status[f]
                    || (!skip.contains(&m) && !same(va.size[f], vb.size[f]));
                if p2 {
                    d2.insert((m, c, f));
                    any.insert((m, f));
                }
            }
        }
    }
    (any, d2, d3)
}

/// The markers whose sizes XMALab leaves uninitialised (detection method 4, ledger A22).
fn method4_markers(doc: &Document) -> Vec<usize> {
    doc.trial(0)
        .markers
        .iter()
        .enumerate()
        .filter(|(_, m)| m.method == DetectionMethod::NoDetection)
        .map(|(i, _)| i)
        .collect()
}

/// Runs the comparison for one run.
///
/// # Errors
/// When the run's files cannot be read or the request does not start.
#[allow(clippy::too_many_lines)]
pub fn run_tracking_parity(run: &TrackingRun) -> Result<TrackingParityReport, String> {
    let start = Instant::now();
    if let Some(reason) = run.not_compared() {
        return Ok(TrackingParityReport {
            run: run.clone(),
            steps: 0,
            last_frame: run.start,
            stop: "not run".to_owned(),
            items: Vec::new(),
            not_compared: vec![NotCompared {
                path: run.run_dir.display().to_string(),
                reason: reason.to_owned(),
            }],
            warnings: Vec::new(),
            seconds: start.elapsed().as_secs_f64(),
        });
    }
    let probe = open_start(run, CppCompat::XMALAB3)?;
    let skip = method4_markers(&probe);
    let step_markers = run.step_markers(probe.trial(0).markers.len());
    drop(probe);
    let log = Table::try_read(run.run_dir.join("run_log.csv")).map_err(|e| e.to_string())?;
    let steps_table =
        Table::try_read(run.run_dir.join("run_steps.csv")).map_err(|e| e.to_string())?;
    let mut by_step: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for r in log.rows() {
        let s = r.try_f64("step").map_err(|e| e.to_string())? as usize;
        by_step.entry(s).or_default().push(r.index());
    }

    let mut log_check = Check::new("run_log.csv", Tier::Read);
    let mut steps_check = Check::new("run_steps.csv", Tier::Read);
    let mut a22_log = 0usize;
    let mut a22_set = 0usize;
    let mut a22_steps = 0usize;
    let mut seen_steps = 0usize;
    let mut on_step = |st: &StepRun, before: &Trial, doc: &Document| {
        seen_steps = st.step;
        let after = doc.trial(0);
        let (mut got, d2, d3) = changes(before, after, &skip);
        for &m in &step_markers {
            got.insert((m, st.frame_to));
        }
        let rows = by_step.get(&st.step).cloned().unwrap_or_default();
        let mut want: BTreeSet<(usize, usize)> = BTreeSet::new();
        for &i in &rows {
            let r = log.row(i);
            if let (Ok(m), Ok(f)) = (r.try_f64("marker"), r.try_f64("frame")) {
                want.insert((m as usize, f as usize));
            }
        }
        // The rows of a method 4 marker follow its uninitialised sizes (A22).
        let n_skip = want.iter().filter(|(m, _)| skip.contains(m)).count();
        a22_set += n_skip;
        got.retain(|(m, _)| !skip.contains(m));
        want.retain(|(m, _)| !skip.contains(m));
        log_check.int(
            format_args!("step {} rows (marker, frame)", st.step),
            got.len() as i64,
            want.len() as i64,
        );
        if got != want {
            let extra: Vec<_> = got.difference(&want).take(6).collect();
            let missing: Vec<_> = want.difference(&got).take(6).collect();
            log_check.problem(format!(
                "step {} (frame {} -> {}): (marker, frame) changed here only {extra:?}, in \
                 XMALab only {missing:?}",
                st.step, st.frame_from, st.frame_to
            ));
        }
        for &i in &rows {
            let r = log.row(i);
            let res = (|| -> Result<(), String> {
                let g = |c: &str| r.try_f64(c).map_err(|e| e.to_string());
                let (m, c, f) = (
                    g("marker")? as usize,
                    g("camera")? as usize,
                    g("frame")? as usize,
                );
                let mk = after
                    .markers
                    .get(m)
                    .ok_or_else(|| format!("no marker {m} here"))?;
                let v = mk.views.get(c).ok_or_else(|| format!("no camera {c}"))?;
                if f >= v.points.len() {
                    return Err(format!("no frame {f}"));
                }
                let l = format!("step {} marker {m} camera {c} frame {f}", st.step);
                log_check.int(
                    format_args!("{l} status"),
                    v.status[f].code().into(),
                    g("status")? as i64,
                );
                log_check.num(format_args!("{l} x"), v.points[f].x, g("x")?);
                log_check.num(format_args!("{l} y"), v.points[f].y, g("y")?);
                log_check.int(
                    format_args!("{l} status3d"),
                    mk.status3d[f].code().into(),
                    g("status3d")? as i64,
                );
                log_check.num(format_args!("{l} x3d"), mk.points3d[f].x, g("x3d")?);
                log_check.num(format_args!("{l} y3d"), mk.points3d[f].y, g("y3d")?);
                log_check.num(format_args!("{l} z3d"), mk.points3d[f].z, g("z3d")?);
                log_check.num(format_args!("{l} mean_size"), mk.mean_size, g("mean_size")?);
                log_check.int(
                    format_args!("{l} changed3d"),
                    i64::from(d3.contains(&(m, f))),
                    g("changed3d")? as i64,
                );
                if skip.contains(&m) {
                    a22_log += 3;
                } else {
                    log_check.num(format_args!("{l} size"), v.size[f], g("size")?);
                    log_check.num(
                        format_args!("{l} size_range"),
                        mk.size_range,
                        g("size_range")?,
                    );
                    log_check.int(
                        format_args!("{l} changed2d"),
                        i64::from(d2.contains(&(m, c, f))),
                        g("changed2d")? as i64,
                    );
                }
                Ok(())
            })();
            if let Err(e) = res {
                log_check.problem(format!("run_log.csv row {}: {e}", i + 1));
            }
        }
        // run_steps.csv
        match steps_table
            .rows()
            .find(|r| r.try_f64("step").ok() == Some(st.step as f64))
        {
            None => steps_check.problem(format!("step {}: no row in XMALab's run", st.step)),
            Some(r) => {
                let g = |c: &str| r.try_f64(c).unwrap_or(f64::NAN) as i64;
                let l = format!("step {}", st.step);
                steps_check.int(
                    format_args!("{l} frame_from"),
                    st.frame_from as i64,
                    g("frame_from"),
                );
                steps_check.int(
                    format_args!("{l} frame_to"),
                    st.frame_to as i64,
                    g("frame_to"),
                );
                steps_check.int(
                    format_args!("{l} merge_messages"),
                    st.report.messages.len() as i64,
                    g("merge_messages"),
                );
                if skip.is_empty() {
                    steps_check.int(
                        format_args!("{l} cells_changed"),
                        d2.len() as i64,
                        g("cells_changed"),
                    );
                } else {
                    a22_steps += 1;
                }
            }
        }
    };
    let tracked = track_run(run, CppCompat::XMALAB3, &mut on_step)?;
    let mut doc = tracked.doc;
    if seen_steps < steps_table.len() {
        steps_check.problem(format!(
            "XMALab ran {} steps, this program {}",
            steps_table.len(),
            seen_steps
        ));
    }
    if let Some(&last) = by_step.keys().last()
        && last > seen_steps
    {
        log_check.problem(format!(
            "XMALab logged steps up to {last}, this program ran {seen_steps}"
        ));
    }
    if a22_log > 0 {
        log_check.exclude(
            RULE_A22,
            a22_log,
            format!(
                "size, size_range and changed2d of every row of markers {skip:?} ({a22_log} \
                 cells): not compared: {WHY_A22}"
            ),
        );
    }
    if a22_set > 0 {
        log_check.exclude(
            RULE_A22,
            a22_set,
            format!(
                "which (marker, frame) of markers {skip:?} a step changed ({a22_set} rows \
                 logged by XMALab): not compared, as XMALab logs a row when an uninitialised \
                 size changes ({WHY_A22}); the values of those rows are compared"
            ),
        );
    }
    if a22_steps > 0 {
        steps_check.exclude(
            RULE_A22,
            a22_steps,
            format!(
                "cells_changed of {a22_steps} steps: not compared: it counts the cells of \
                 markers {skip:?} whose uninitialised size changed ({WHY_A22})"
            ),
        );
    }
    let mut items = vec![steps_check.finish(), log_check.finish()];

    // manifest.json: steps and last frame.
    let mut m = Check::new("manifest.json (steps, last_frame)", Tier::Read);
    m.int("steps", tracked.steps as i64, run.steps as i64);
    m.int(
        "last_frame",
        tracked.last_frame as i64,
        run.last_frame as i64,
    );
    items.push(m.finish());

    // merge_log.txt
    let mut ml = Check::new("merge_log.txt", Tier::Read);
    match std::fs::read_to_string(run.run_dir.join("merge_log.txt")) {
        Err(e) => ml.problem(format!("cannot read the reference: {e}")),
        Ok(text) => {
            let want: Vec<&str> = text.lines().collect();
            if !(want.is_empty() && tracked.messages.is_empty()) {
                ml.rows(tracked.messages.len(), want.len());
                for (i, (a, b)) in tracked.messages.iter().zip(&want).enumerate() {
                    ml.text(format_args!("line {}", i + 1), a, b);
                }
                ml.bytes_equal = Some(tracked.messages == want);
                ml.equal_values = want.len();
            }
        }
    }
    items.push(ml.finish());

    // The state dump and the saved project.
    let mut uninit: Uninit = loaded_uninit(&doc);
    let mut sizes = SizeRules::default();
    for &k in &skip {
        if let Some(u) = uninit.get_mut(0).and_then(|u| u.get_mut(k)) {
            *u = Some((RULE_A22, WHY_A22));
        }
        sizes.skip.push((0, k, RULE_A22, WHY_A22));
    }
    for mut i in compare_state(
        &run.run_dir.join("state"),
        "state",
        &doc,
        &uninit,
        None,
        &sizes,
    ) {
        i.name = format!("state/{}", i.name);
        items.push(i);
    }
    items.extend(compare_save(
        &run.run_dir.join("project.xma"),
        "project.xma",
        &mut doc,
        false,
        &sizes,
    ));
    let mut warnings: Vec<String> = doc.warnings().to_vec();
    warnings.extend(doc.extras().warnings.iter().cloned());
    Ok(TrackingParityReport {
        run: run.clone(),
        steps: tracked.steps,
        last_frame: tracked.last_frame,
        stop: stop_text(tracked.stop.as_ref()),
        items,
        not_compared: Vec::new(),
        warnings,
        seconds: start.elapsed().as_secs_f64(),
    })
}
