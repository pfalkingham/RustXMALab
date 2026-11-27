//! The tier C report: tracking in the default mode against XMALab's result and against the
//! reference project's own points (PLAN.md section 7, tier C, "reported as a distribution").
//!
//! The default mode corrects XMALab 3's tracking defects (parity ledger, the rows marked
//! "Fix, compat"), so its tracking differs from XMALab's on purpose and no C++ reference
//! exists for it. For every run of `oracle/tracking_runs.json` [`run_tier_c`] makes the
//! run's request from its start project in the default mode and compares the 2D points with:
//!
//! - XMALab's result for the same run: the project XMALab saved after it (`project.xma` of
//!   the run, which the XMALab 3 mode reproduces exactly; `xma parity <run>`);
//! - the reference project's points (`reference_project` of `tracking_runs.json`,
//!   `oracle/data/pig/filled/pigDataset_filled_clean.xma`): the points a person accepted,
//!   every point with a status above 0.
//!
//! The points compared are the (marker, camera, frame) cells that either run changed
//! (position or status different from the start project); the start project's own points
//! are left out, as both runs keep them. Distances are Euclidean, in pixels. Percentiles are
//! nearest-rank (the smallest value with at least p% of the values at or below it).
//!
//! The runs of merge-aware tracking (types 5 and 6) are listed as not compared: that kind
//! was removed from this program on 26 November 2026 (parity ledger A226).

use std::fmt::Write as _;
use std::time::Instant;

use xma_engine::{Document, OpenOptions};
use xma_model::{CppCompat, Trial};

use super::cell;
use super::tracking::{TrackingRun, index_path, open_start, stop_text, track_run, tracking_runs};

/// Worst cases listed per run.
const WORST: usize = 8;

/// A distribution of distances in pixels.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Distribution {
    values: Vec<f64>,
}

impl Distribution {
    fn push(&mut self, v: f64) {
        self.values.push(v);
    }

    fn sorted(&self) -> Vec<f64> {
        let mut v = self.values.clone();
        v.sort_by(f64::total_cmp);
        v
    }

    /// Number of values.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The nearest-rank percentile `p` (0 to 100); `None` when empty.
    #[must_use]
    pub fn percentile(&self, p: f64) -> Option<f64> {
        let v = self.sorted();
        if v.is_empty() {
            return None;
        }
        let rank = ((p / 100.0) * v.len() as f64).ceil().max(1.0) as usize;
        Some(v[rank.min(v.len()) - 1])
    }

    #[must_use]
    pub fn mean(&self) -> Option<f64> {
        (!self.values.is_empty()).then(|| self.values.iter().sum::<f64>() / self.len() as f64)
    }

    #[must_use]
    pub fn max(&self) -> Option<f64> {
        self.values.iter().copied().reduce(f64::max)
    }

    /// "median / 95% / 99% / largest" as text.
    fn text(&self) -> String {
        if self.is_empty() {
            return "none".to_owned();
        }
        let f = |v: Option<f64>| px(v.unwrap_or(f64::NAN));
        format!(
            "{} / {} / {} / {}",
            f(self.percentile(50.0)),
            f(self.percentile(95.0)),
            f(self.percentile(99.0)),
            f(self.max())
        )
    }

    fn extend(&mut self, other: &Distribution) {
        self.values.extend_from_slice(&other.values);
    }
}

/// A distance in pixels as the report writes it.
fn px(v: f64) -> String {
    if v == 0.0 {
        "0".to_owned()
    } else if v < 1e-3 {
        format!("{v:.1e}")
    } else {
        format!("{v:.3}")
    }
}

/// One point the two results place differently.
#[derive(Clone, Debug, PartialEq)]
pub struct Case {
    pub marker: usize,
    pub camera: usize,
    pub frame: usize,
    /// XMALab's position, the default mode's, the reference's (`None`: no point).
    pub xmalab: Option<(f64, f64)>,
    pub default: Option<(f64, f64)>,
    pub reference: Option<(f64, f64)>,
}

impl Case {
    fn moved(&self) -> f64 {
        match (self.xmalab, self.default) {
            (Some(a), Some(b)) => dist(a, b),
            _ => f64::INFINITY,
        }
    }

    /// Default's distance from the reference minus XMALab's (positive: further).
    fn worse(&self) -> f64 {
        match (self.xmalab, self.default, self.reference) {
            (Some(a), Some(b), Some(r)) => dist(b, r) - dist(a, r),
            _ => 0.0,
        }
    }
}

fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

/// What one run gives.
#[derive(Clone, Debug)]
pub struct RunResult {
    pub run: TrackingRun,
    /// Steps and last frame in the default mode, and why it stopped.
    pub steps: usize,
    pub last_frame: usize,
    pub stop: String,
    /// Cells either run changed.
    pub changed: usize,
    /// Same position in both (of them, with a different status).
    pub same: usize,
    pub same_status_differs: usize,
    /// Both have a point at different positions: the distances between them.
    pub moved: Distribution,
    /// A point in one result only.
    pub default_only: usize,
    pub xmalab_only: usize,
    /// No point in either (both runs reset a start point).
    pub neither: usize,
    /// Distance from the reference on the cells where both results and the reference have
    /// a point.
    pub ref_xmalab: Distribution,
    pub ref_default: Distribution,
    /// Of those, cells where the default mode is closer to the reference, further, equal.
    pub closer: usize,
    pub further: usize,
    /// Points of each result at cells where the reference has one (all such cells).
    pub ref_reached_xmalab: usize,
    pub ref_reached_default: usize,
    /// Distance from the reference over every cell where the result and the reference have
    /// a point.
    pub ref_all_xmalab: Distribution,
    pub ref_all_default: Distribution,
    /// Points of each result where the reference has none.
    pub unreferenced_xmalab: usize,
    pub unreferenced_default: usize,
    /// Largest moves, then the cells where the default mode is furthest behind XMALab.
    pub worst_moves: Vec<Case>,
    pub worst_regressions: Vec<Case>,
}

impl RunResult {
    /// Closer, further, the same or mixed, in words.
    #[must_use]
    pub fn verdict(&self) -> String {
        if self.moved.is_empty()
            && self.default_only == 0
            && self.xmalab_only == 0
            && self.same_status_differs == 0
        {
            return "the same (every point identical)".to_owned();
        }
        if self.ref_default.is_empty() {
            return "not decidable (no common point with the reference)".to_owned();
        }
        let (md, mx) = (
            self.ref_default.mean().unwrap_or(0.0),
            self.ref_xmalab.mean().unwrap_or(0.0),
        );
        let (pd, px_) = (
            self.ref_default.percentile(95.0).unwrap_or(0.0),
            self.ref_xmalab.percentile(95.0).unwrap_or(0.0),
        );
        let (c, f) = (self.closer, self.further);
        let fewer = self.ref_reached_default < self.ref_reached_xmalab;
        let base = if c == 0 && f == 0 {
            "the same on the common points"
        } else if md < mx && pd <= px_ && c > f && !fewer {
            "**closer**"
        } else if md > mx && pd >= px_ && f > c {
            "**further**"
        } else {
            "**mixed**"
        };
        let cov = match self.ref_reached_default.cmp(&self.ref_reached_xmalab) {
            std::cmp::Ordering::Equal => String::new(),
            std::cmp::Ordering::Greater => format!(
                "; reaches {} more reference points than XMALab",
                self.ref_reached_default - self.ref_reached_xmalab
            ),
            std::cmp::Ordering::Less => format!(
                "; reaches {} fewer reference points than XMALab",
                self.ref_reached_xmalab - self.ref_reached_default
            ),
        };
        format!(
            "{base} (mean distance {md:.4} px, XMALab {mx:.4} px; {c} points closer, {f} \
             further){cov}"
        )
    }
}

/// The report of [`run_tier_c`].
#[derive(Clone, Debug)]
pub struct TierCReport {
    pub runs: Vec<RunResult>,
    pub reference: std::path::PathBuf,
    pub errors: Vec<String>,
    /// The runs not compared, each with the reason ([`TrackingRun::not_compared`]).
    pub not_compared: Vec<(String, String)>,
    pub seconds: f64,
}

/// The trial 0 of a project as read (no update).
fn read_trial(path: &std::path::Path) -> Result<Trial, String> {
    let options = OpenOptions {
        compat: CppCompat::XMALAB3,
        ..OpenOptions::default()
    };
    let loaded =
        xma_engine::read_project(path, &options).map_err(|e| format!("{}: {e}", path.display()))?;
    let doc = Document::from_loaded(loaded);
    Ok(doc.trial(0).clone())
}

fn point(t: &Trial, m: usize, c: usize, f: usize) -> Option<(f64, f64)> {
    let v = t.markers.get(m)?.views.get(c)?;
    v.status
        .get(f)
        .filter(|s| s.has_data())
        .map(|_| (v.points[f].x, v.points[f].y))
}

/// Compares one run.
///
/// # Errors
/// When a project cannot be read or the request does not start.
#[allow(clippy::too_many_lines)]
pub fn compare_run(run: &TrackingRun, reference: &Trial) -> Result<RunResult, String> {
    let start = open_start(run, CppCompat::default())?.trial(0).clone();
    let xmalab = read_trial(&run.run_dir.join("project.xma"))?;
    let tracked = track_run(run, CppCompat::default(), &mut |_, _, _| {})?;
    let default = tracked.doc.trial(0).clone();
    let mut r = RunResult {
        run: run.clone(),
        steps: tracked.steps,
        last_frame: tracked.last_frame,
        stop: stop_text(tracked.stop.as_ref()),
        changed: 0,
        same: 0,
        same_status_differs: 0,
        moved: Distribution::default(),
        default_only: 0,
        xmalab_only: 0,
        neither: 0,
        ref_xmalab: Distribution::default(),
        ref_default: Distribution::default(),
        closer: 0,
        further: 0,
        ref_reached_xmalab: 0,
        ref_reached_default: 0,
        ref_all_xmalab: Distribution::default(),
        ref_all_default: Distribution::default(),
        unreferenced_xmalab: 0,
        unreferenced_default: 0,
        worst_moves: Vec::new(),
        worst_regressions: Vec::new(),
    };
    let mut cases = Vec::new();
    for (m, sm) in start.markers.iter().enumerate() {
        for (c, sv) in sm.views.iter().enumerate() {
            for f in 0..sv.points.len() {
                let cell_of = |t: &Trial| {
                    t.markers.get(m).and_then(|mk| mk.views.get(c)).map(|v| {
                        (
                            v.status[f],
                            v.points[f].x.to_bits(),
                            v.points[f].y.to_bits(),
                        )
                    })
                };
                let s0 = Some((
                    sv.status[f],
                    sv.points[f].x.to_bits(),
                    sv.points[f].y.to_bits(),
                ));
                let (sx, sd) = (cell_of(&xmalab), cell_of(&default));
                if sx == s0 && sd == s0 {
                    continue;
                }
                r.changed += 1;
                let (px_, pd, pr) = (
                    point(&xmalab, m, c, f),
                    point(&default, m, c, f),
                    point(reference, m, c, f),
                );
                match (px_, pd) {
                    (Some(a), Some(b)) => {
                        if a == b {
                            r.same += 1;
                            if sx.map(|s| s.0) != sd.map(|s| s.0) {
                                r.same_status_differs += 1;
                            }
                        } else {
                            r.moved.push(dist(a, b));
                        }
                    }
                    (Some(_), None) => r.xmalab_only += 1,
                    (None, Some(_)) => r.default_only += 1,
                    (None, None) => r.neither += 1,
                }
                if let Some(rp) = pr {
                    if let Some(a) = px_ {
                        r.ref_reached_xmalab += 1;
                        r.ref_all_xmalab.push(dist(a, rp));
                    }
                    if let Some(b) = pd {
                        r.ref_reached_default += 1;
                        r.ref_all_default.push(dist(b, rp));
                    }
                    if let (Some(a), Some(b)) = (px_, pd) {
                        let (da, db) = (dist(a, rp), dist(b, rp));
                        r.ref_xmalab.push(da);
                        r.ref_default.push(db);
                        if db < da {
                            r.closer += 1;
                        } else if db > da {
                            r.further += 1;
                        }
                    }
                } else {
                    r.unreferenced_xmalab += usize::from(px_.is_some());
                    r.unreferenced_default += usize::from(pd.is_some());
                }
                if px_ != pd {
                    cases.push(Case {
                        marker: m,
                        camera: c,
                        frame: f,
                        xmalab: px_,
                        default: pd,
                        reference: pr,
                    });
                }
            }
        }
    }
    let mut by_move = cases.clone();
    by_move.sort_by(|a, b| b.moved().total_cmp(&a.moved()));
    r.worst_moves = by_move.into_iter().take(WORST).collect();
    let mut by_worse: Vec<Case> = cases.into_iter().filter(|c| c.worse() > 0.0).collect();
    by_worse.sort_by(|a, b| b.worse().total_cmp(&a.worse()));
    r.worst_regressions = by_worse.into_iter().take(WORST).collect();
    Ok(r)
}

/// Runs every tracking run in the default mode and compares (see the module text).
///
/// # Errors
/// When the list of runs or the reference project cannot be read.
pub fn run_tier_c() -> Result<TierCReport, String> {
    let t0 = Instant::now();
    let runs = tracking_runs()?;
    let reference = index_path("reference_project")
        .ok_or("oracle/tracking_runs.json names no reference_project")?;
    let ref_trial = read_trial(&reference)?;
    let mut out = Vec::new();
    let mut errors = Vec::new();
    let mut not_compared = Vec::new();
    for run in &runs {
        if let Some(why) = run.not_compared() {
            not_compared.push((run.name.clone(), why.to_owned()));
            continue;
        }
        if !run.is_present() {
            errors.push(format!("{}: not present", run.name));
            continue;
        }
        match compare_run(run, &ref_trial) {
            Ok(r) => out.push(r),
            Err(e) => errors.push(format!("{}: {e}", run.name)),
        }
    }
    Ok(TierCReport {
        runs: out,
        reference,
        errors,
        not_compared,
        seconds: t0.elapsed().as_secs_f64(),
    })
}

fn pos(p: Option<(f64, f64)>) -> String {
    p.map_or_else(
        || "no point".to_owned(),
        |(x, y)| format!("({x:.3}, {y:.3})"),
    )
}

fn dref(p: Option<(f64, f64)>, r: Option<(f64, f64)>) -> String {
    match (p, r) {
        (Some(a), Some(b)) => px(dist(a, b)),
        _ => String::new(),
    }
}

impl TierCReport {
    /// One line.
    #[must_use]
    pub fn summary(&self) -> String {
        let n = |f: &dyn Fn(&RunResult) -> bool| self.runs.iter().filter(|r| f(r)).count();
        format!(
            "tier C: {} runs; default mode against XMALab: {} identical, {} closer to the \
             reference points, {} further, {} mixed{}",
            self.runs.len(),
            n(&|r| r.verdict().starts_with("the same (every")),
            n(&|r| r.verdict().starts_with("**closer")),
            n(&|r| r.verdict().starts_with("**further")),
            n(&|r| r.verdict().starts_with("**mixed")),
            if self.errors.is_empty() {
                String::new()
            } else {
                format!("; {} runs not made", self.errors.len())
            }
        ) + &if self.not_compared.is_empty() {
            String::new()
        } else {
            format!("; {} runs not compared", self.not_compared.len())
        }
    }

    /// The report as Markdown.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn to_markdown(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "# Tier C report: tracking in the default mode\n");
        let _ = writeln!(s, "**{}.**\n", self.summary());
        let _ = writeln!(
            s,
            "The default mode corrects XMALab 3's tracking defects (parity ledger, the rows \
             marked \"Fix, compat\": A1, A14, A15, A62, A64, A65, A73 and the \
             others), so its tracking differs from XMALab's on purpose, and XMALab has no \
             reference for it. Each run of `oracle/tracking_runs.json` was made here in the \
             default mode from the run's start project and compared with XMALab's result for \
             the same run (the project XMALab saved; the XMALab 3 mode reproduces it exactly, \
             see the run's own report, `tracking/<run>.md` beside this one) and with the points of the reference project, which a \
             person accepted.\n"
        );
        let _ = writeln!(s, "- Default mode: {}", CppCompat::default().describe());
        let _ = writeln!(
            s,
            "- Reference points: `{}` (every point with a status above 0)",
            self.reference.display()
        );
        let _ = writeln!(
            s,
            "- What the reference points are: 2,832 of the reference project's 2,844 points \
             have the status \"tracked\" (12 are \"set\"): they were tracked by XMALab and \
             accepted by a person, not placed independently. Distance from them therefore \
             favours XMALab's own behaviour, defects included, and is not a measure of \
             accuracy against the true marker centre."
        );
        let _ = writeln!(
            s,
            "- Points compared: the (marker, camera, frame) cells either run changed from its \
             start project; the start project's own points are left out. Marker, camera and \
             frame are 0-based."
        );
        let _ = writeln!(
            s,
            "- Distances in pixels; distributions written \"median / 95th / 99th percentile / \
             largest\" (nearest rank)."
        );
        let _ = writeln!(
            s,
            "- Over the points where both results and the reference have a point: \"closer\" \
             when the default mode's mean distance from the reference is smaller than \
             XMALab's, its 95th percentile is not larger, more points are closer than \
             further, and it reaches no fewer reference points; \"further\" when the mean is \
             larger, the 95th percentile not smaller and more points are further; \"mixed\" \
             otherwise; \"the same\" when no point differs. The means and counts are given \
             beside it."
        );
        let _ = writeln!(
            s,
            "- The tier C tolerance of PLAN.md section 7 (median below 0.01 px, 99% below 0.1 \
             px) is for the port against XMALab in the same mode: the XMALab 3 mode meets it \
             with every point identical (the tracking runs' reports). The default mode is not \
             judged by it; its differences are the fixes' effects, shown here."
        );
        let _ = writeln!(
            s,
            "- Made by: xma {} (engine {}), in {:.1} s\n",
            crate::VERSION,
            xma_engine::VERSION,
            self.seconds
        );

        let _ = writeln!(s, "## Against XMALab's result\n");
        let _ = writeln!(
            s,
            "| Run | Steps here (XMALab) | Points changed | Same | Moved | Move (px) | Default only | XMALab only | Neither |\n|---|---|---:|---:|---:|---|---:|---:|---:|"
        );
        let mut all_moved = Distribution::default();
        let mut tot = [0usize; 6];
        for r in &self.runs {
            all_moved.extend(&r.moved);
            for (i, v) in [
                r.changed,
                r.same,
                r.moved.len(),
                r.default_only,
                r.xmalab_only,
                r.neither,
            ]
            .into_iter()
            .enumerate()
            {
                tot[i] += v;
            }
            let _ = writeln!(
                s,
                "| [`{n}`](tracking/{n}.md) | {} to frame {} ({} to frame {}) | {} | {}{} | {} | {} | {} | {} | {} |",
                r.steps,
                r.last_frame,
                r.run.steps,
                r.run.last_frame,
                r.changed,
                r.same,
                if r.same_status_differs > 0 {
                    format!(" ({} with another status)", r.same_status_differs)
                } else {
                    String::new()
                },
                r.moved.len(),
                r.moved.text(),
                r.default_only,
                r.xmalab_only,
                r.neither,
                n = r.run.name
            );
        }
        let _ = writeln!(
            s,
            "| **All runs** | | **{}** | **{}** | **{}** | {} | **{}** | **{}** | **{}** |",
            tot[0],
            tot[1],
            tot[2],
            all_moved.text(),
            tot[3],
            tot[4],
            tot[5]
        );

        let _ = writeln!(s, "\n## Against the reference points\n");
        let _ = writeln!(
            s,
            "Over the points where both results and the reference have a point (\"common\"), \
             then each result over every point where it and the reference have one.\n"
        );
        let _ = writeln!(
            s,
            "| Run | Common points | XMALab from reference (px) | Default from reference (px) | Default mode is | Reference points reached: XMALab, default | Points where the reference has none: XMALab, default |\n|---|---:|---|---|---|---|---|"
        );
        let (mut ax, mut ad) = (Distribution::default(), Distribution::default());
        for r in &self.runs {
            ax.extend(&r.ref_xmalab);
            ad.extend(&r.ref_default);
            let _ = writeln!(
                s,
                "| `{}` | {} | {} | {} | {} | {}, {} | {}, {} |",
                r.run.name,
                r.ref_default.len(),
                r.ref_xmalab.text(),
                r.ref_default.text(),
                cell(&r.verdict()),
                r.ref_reached_xmalab,
                r.ref_reached_default,
                r.unreferenced_xmalab,
                r.unreferenced_default
            );
        }
        let _ = writeln!(
            s,
            "| **All runs** | **{}** | {} | {} | | | |",
            ad.len(),
            ax.text(),
            ad.text()
        );
        let _ = writeln!(
            s,
            "\nEach result over all its points with a reference point (not only the common \
             ones):\n\n| Run | XMALab: points, distance (px) | Default: points, distance (px) |\n|---|---|---|"
        );
        for r in &self.runs {
            let _ = writeln!(
                s,
                "| `{}` | {}: {} | {}: {} |",
                r.run.name,
                r.ref_all_xmalab.len(),
                r.ref_all_xmalab.text(),
                r.ref_all_default.len(),
                r.ref_all_default.text()
            );
        }

        let _ = writeln!(s, "\n## Worst cases, per run\n");
        let _ = writeln!(
            s,
            "For each run, the points the two results place furthest apart (a point in one \
             result only first), then the points where the default mode is furthest behind \
             XMALab in distance from the reference. Up to {WORST} of each.\n"
        );
        for r in &self.runs {
            let _ = writeln!(
                s,
                "### `{}`\n\n{} Default mode: {} steps to frame {} ({}).\n",
                r.run.name, r.run.description, r.steps, r.last_frame, r.stop
            );
            if r.worst_moves.is_empty() {
                let _ = writeln!(s, "Every point identical.\n");
                continue;
            }
            let table = |s: &mut String, title: &str, list: &[Case]| {
                let _ = writeln!(
                    s,
                    "{title}\n\n| Marker | Camera | Frame | XMALab | Default | Moved | Reference | XMALab from ref. | Default from ref. |\n|---:|---:|---:|---|---|---|---|---|---|"
                );
                for c in list {
                    let _ = writeln!(
                        s,
                        "| {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                        c.marker,
                        c.camera,
                        c.frame,
                        pos(c.xmalab),
                        pos(c.default),
                        if c.moved().is_finite() {
                            px(c.moved())
                        } else {
                            "one only".to_owned()
                        },
                        pos(c.reference),
                        dref(c.xmalab, c.reference),
                        dref(c.default, c.reference)
                    );
                }
                let _ = writeln!(s);
            };
            table(&mut s, "Largest differences:", &r.worst_moves);
            if r.worst_regressions.is_empty() {
                let _ = writeln!(
                    s,
                    "No point where the default mode is further from the reference than XMALab.\n"
                );
            } else {
                table(
                    &mut s,
                    "Where the default mode is furthest behind XMALab:",
                    &r.worst_regressions,
                );
            }
        }
        if !self.not_compared.is_empty() {
            let _ = writeln!(s, "## Not compared\n");
            let _ = writeln!(s, "| Run | Why |\n|---|---|");
            for (name, why) in &self.not_compared {
                let _ = writeln!(s, "| `{name}` | {} |", cell(why));
            }
            let _ = writeln!(s);
        }
        if !self.errors.is_empty() {
            let _ = writeln!(s, "## Runs not made\n");
            for e in &self.errors {
                let _ = writeln!(s, "- {e}");
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::Distribution;

    #[test]
    fn nearest_rank_percentiles() {
        let mut d = Distribution::default();
        for v in [5.0, 1.0, 4.0, 2.0, 3.0] {
            d.push(v);
        }
        assert_eq!(d.percentile(50.0), Some(3.0));
        assert_eq!(d.percentile(95.0), Some(5.0));
        assert_eq!(d.percentile(20.0), Some(1.0));
        assert_eq!(d.percentile(21.0), Some(2.0));
        assert_eq!(d.max(), Some(5.0));
        assert_eq!(d.mean(), Some(3.0));
        assert_eq!(Distribution::default().percentile(50.0), None);
    }
}
