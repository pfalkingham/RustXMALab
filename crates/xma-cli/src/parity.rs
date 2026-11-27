//! `xma parity`: this program against the C++ reference dumps, as a Markdown report.
//!
//! [`run_parity`] opens the dump's project with the settings recorded in its
//! `manifest.json`, replays what the C++ dump driver (`OracleDumpDriver.cpp`) does before
//! writing each table, and compares every table and export file of the dump that phase 1
//! covers. The replay, in the driver's order:
//!
//! 1. load as the GUI does (`Document::from_loaded`, `update_after_load`: the active trial
//!    is updated with the user's settings); then the project-level tables and exports
//!    (`writeTopLevel`, `writeUndistortion`, `writeCalibration`, `writeProjectExports`),
//!    with, in a format 2 dump, the top-level tables of what loading determines;
//! 2. for each trial: select it (`State::changeActiveTrial`, a trial update); the "as held"
//!    tables (`writeTrialSnapshot`), with the per-trial format 2 tables; the exports in the driver's order (`writeTrialExports`;
//!    only the rigid-body exports change state, by recomputing the bodies, ledger A36);
//!    then for k = 0 to 4 the triangulation method k, flag, update, `points3d_method<k>`;
//!    then the user's method with `OptimizeRigidBody` off and on, flag, update,
//!    `rb_pose_kabsch` and `rb_pose_optimized`; then the user's settings again and a last
//!    update (`dumpTrial`).
//!
//! Every file of the dump that is not compared is listed with the reason, and so is every
//! table the manifest lists as missing; neither counts as passing. A table that cannot be
//! read, has the wrong number of rows, or whose flags, statuses or names differ is
//! "different" whatever its numbers.
//!
//! Two rules take values out of the plain comparison, each naming its condition and listed
//! in the report with the values on both sides ([`Item::notes`]):
//!
//! - `markers.csv` `size_range` of a marker without a `<Marker>` element: the C++ value is
//!   uninitialised memory (ledger A52);
//! - the C++ data race of ledger A56 ([`race`]): in any trial update the driver runs, the
//!   interface thread can compute, at frame 0, a body that a visible body's virtual point
//!   reaches, so that body ends the update with its pose refined twice. For the updates
//!   whose results the dump holds (the one before the "as held" tables, which also
//!   determines the rigid-body exports, and the one before `rb_pose_optimized.csv`; the
//!   others hold nothing the race can change, and an update with the refinement off cannot
//!   race), a cell of a table the race can change (`rb_errors.csv`, `rb_pose_filtered.csv`,
//!   `rb_errors_filtered.csv`, `rb_pose_optimized.csv`, the two rigid-body export files)
//!   that differs from this program's value is compared with the value each order of the
//!   race gives, replayed through the scheduler's finalize and the exports' recomputation;
//!   an exact match is "explained". A differing cell that no order reproduces exactly is
//!   an ordinary difference, unless an order that the dump shows exactly elsewhere (a raced
//!   body's own frame 0 value) changes it too: then it is "not compared: downstream of a
//!   raced pose", under a rule naming that raced cell.
//!
//! Columns the model does not hold (an event colour's alpha) and values the C++ leaves
//! uninitialised (a calibration frame's `square_size`) are noted the same way.
//!
//! A table with any such cell gets the verdict [`Verdict::IdenticalExplained`] ("identical
//! apart from N explained cells", with the rules in its row), never plain "identical",
//! and has its own count in the headline. A cell not compared is left out of the values
//! count; for a cell compared with an alternative, the row's largest differences are this
//! program's own value against the dump's. Neither makes the run fail (exit status 0).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use xma_engine::{Document, Export, OpenOptions};
use xma_io::export::points2d::Points2dOptions;
use xma_io::export::points3d::{Points3dOptions, filter_marker};
use xma_io::export::rigid_bodies::{TransformationsOptions, can_export_filtered};
use xma_io::export::{ExportFile, ExportOutput};
use xma_model::{CppCompat, MarkerStatus, VirtualPointSource};
use xma_oracle::{Comparison, Dataset, Manifest, Row, Table, Tolerance};

pub mod calib_dump;
pub mod calib_ops;
pub mod calibration;
pub mod checker_ops;
pub mod checker_parity;
pub mod edits;
pub mod portal;
mod race;
#[cfg(test)]
mod race_tests;
#[cfg(test)]
mod review12_tests;
mod steps;
pub mod tier_c;
pub mod tracking;

use race::Race;

use crate::args::{Opt, parse};
use crate::{CliError, Io};

pub const HELP: &str = "\
xma parity: compare this program with the C++ reference dumps

Usage: xma parity <dataset-name | edit-run | calibration-run | tracking-run | tier-c | dump-dir>
                  [--project <project.xma>] [--report <file>]

A tracking run is a run of oracle/tracking_runs.json (for example r01_all_first_fwd_t3): the
start project is opened in the XMALab 3 mode with the run's settings, the same request is
tracked step by step and compared after every step with XMALab's run_log.csv and
run_steps.csv and merge_log.txt, the final state with the state dump and the saved project
with XMALab's (oracle/FORMAT.md, \"Tracking runs\"). The runs of merge-aware tracking
(types 5 and 6: r06_overlap_t5, r07_overlap_t6, r08_overlap_back_t6) are reported as not
compared, with exit status 0: merge-aware tracking was removed from this program on
26 November 2026 (parity ledger A226).

`tier-c` tracks every tracking run in the default mode (the XMALab 3 faults fixed) and
reports how its points differ from XMALab's result and from the reference project's
accepted points, as distributions in pixels with the worst cases; the merge-aware runs are
listed as not compared. It is a measurement: its exit status is 0 unless a run cannot be
made.

A dump directory made with --frame-step 1 (oracle/data/pig/filled/dump_clean_step1) also has
its detection and tracker-step tables compared (detection_methods.csv, tracking_steps.csv).

An edit run is a script of oracle/edit_runs.json (for example e01_points): the start project
is opened in the XMALab 3 mode, the script's operations are applied as the C++ editing
reference applied them, and the document is compared with the C++ state dump at every Dump
and with the C++ saved file at every Save (oracle/FORMAT.md, \"State dump\" and \"Edit
scripts\").

An export or denoising run is a script of oracle/export_runs.json or oracle/denoise_runs.json
(x01_main, d01_default_relink, ...): replayed in the XMALab 3 mode as a calibration run,
every file written compared with XMALab's (paths exactly, TIFF and PNG pixels exactly, JPEG
within 12 grey levels and by the pixels before encoding where the reference has them, text
byte for byte), with the denoising hooks.
An XMA Portal run is a script of oracle/portal_runs.json (p01_trial_keep to p08_metadata):
replayed in the XMALab 3 mode in a private copy of its staged files, with every Dump and Save
compared as for a calibration run, and also dialogs.csv (the New trial and New dataset
dialogs after loading, Project Metadata's list and tree), the questions asked with their
texts, and the files of every folder the script lists.

A calibration run is a script of oracle/calibration_runs.json (for example c11_trial_nocalib):
the script is replayed in the XMALab 3 mode (new datasets, new trials, workspaces, seeds and
settings as the C++ calibration reference ran them), and every Dump is compared with the C++
state dump and calibration state, every Save with the C++ saved file (oracle/FORMAT.md,
\"Calibration runs\"). Undistortion and cube-calibration operations are applied as the
window applies them; the replay stops at the first operation not built yet (the
multi-camera optimisation, the MayaCam files), and later dumps are reported as not reached.
Camera matrices and poses after a cube calibration are compared within the named tolerance
of ledger A37 (9e-10 relative). The checkerboard runs (k01 to k10, `checker_runs`) also have
their hook tables compared (detection, refinement, homography, prediction, sorting, the
calibrations over all frames), and what is computed from a calibration is compared within
the named tolerances of ledger A201.

A dataset name is looked up in oracle/datasets.json (the oracle folder is
$XMA_ORACLE_ROOT, else <workspace>/oracle). A dump directory needs its project: --project,
else the project_path recorded in the dump's manifest.json.

The project is opened with the settings in the dump's manifest.json (the machine's settings
the C++ ran with; they override any settings the project stores, which the C++ ignores),
the C++ dump driver's sequence is replayed (load, select each trial, exports, the five
triangulation methods, rigid-body optimisation off and on), and every table and export file
of the dump that phase 1 covers is compared, including, in a format 2 dump, the tables of
what loading determines. The report (Markdown) gives, for each, the number of values
compared, the largest absolute and relative differences and where, the tier of the plan
and a verdict: identical, within the tier's tolerance, or different. Values not compared
within a compared table are listed with the rule that applies (a value the C++ leaves
uninitialised, ledger A52; another order of a C++ data race, ledger A56, replayed and
matched exactly, or downstream of a raced cell so matched). Files
not compared are listed with the reason, as are the tables the manifest lists as missing.

The comparison always runs in the XMALab 3 compatibility mode (the C++ behaviour of parity
ledger A3, A4, F2 and F18, which this program otherwise fixes), and the report says so.

Options:
      --project <file>  The project the dump was made from
      --report <file>   Write the report there (default: standard output)
  -h, --help            Print help

Exit status: 0 nothing different, 3 something different, 1 error, 2 usage error.";

// ---------------------------------------------------------------------------------------
// Report types
// ---------------------------------------------------------------------------------------

/// The tiers of PLAN.md section 7, with the tolerance each is judged by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    /// Values read from the project file: exact.
    Read,
    /// Tier A, closed-form maths: 1e-9 relative (1e-12 absolute near zero).
    A,
    /// Tier A where undistortion is involved: 1e-7 pixels (PLAN.md section 7).
    AUndistortion,
    /// Tier B, iterative solvers: 1e-6 absolute or relative.
    B,
}

impl Tier {
    /// The tolerance of the tier.
    #[must_use]
    pub fn tolerance(self) -> Tolerance {
        match self {
            Self::Read => Tolerance::Abs(0.0),
            Self::A => Tolerance::Either {
                abs: 1e-12,
                rel: 1e-9,
            },
            Self::AUndistortion => Tolerance::Abs(1e-7),
            Self::B => Tolerance::Either {
                abs: 1e-6,
                rel: 1e-6,
            },
        }
    }

    /// A short name for the report's tables.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::A => "A",
            Self::AUndistortion => "A (undist.)",
            Self::B => "B",
        }
    }

    fn describe(self) -> &'static str {
        match self {
            Self::Read => "values read from the project file",
            Self::A => "closed-form maths: triangulation, projection, filtering, formatting",
            Self::AUndistortion => "closed-form maths through the undistortion grid",
            Self::B => "iterative solvers: rigid-body pose refinement and what reads it",
        }
    }
}

/// The verdict on one table or file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Every value equal (a file: every byte equal).
    Identical,
    /// Every value compared equal, but at least one cell was handled by a named rule: not
    /// compared (a value the C++ leaves uninitialised, a column the model does not hold),
    /// or compared with something other than this program's value (the other order of a
    /// C++ data race). Never counted as plain [`Verdict::Identical`]; see [`Item::rules`].
    IdenticalExplained,
    /// A table with no rows here and none in the dump (for example no virtual points).
    BothEmpty,
    /// Not identical, but every value within the tier's tolerance and nothing else differs.
    WithinTolerance,
    /// A value outside the tolerance, or a difference in layout, rows, flags, statuses or
    /// names, or the reference could not be read.
    Different,
}

impl Verdict {
    fn label(self) -> &'static str {
        match self {
            Self::Identical => "identical",
            Self::IdenticalExplained => "identical apart from explained cells",
            Self::BothEmpty => "identical (no rows either side)",
            Self::WithinTolerance => "within tolerance",
            Self::Different => "**different**",
        }
    }
}

/// One table or export file compared.
#[derive(Clone, Debug)]
pub struct Item {
    /// Path relative to the dump directory.
    pub name: String,
    pub tier: Tier,
    /// Numbers compared (flags and statuses included, as numbers).
    pub values: usize,
    /// Largest absolute difference and where.
    pub max_abs: f64,
    pub max_abs_at: String,
    /// Largest relative difference and where.
    pub max_rel: f64,
    pub max_rel_at: String,
    /// Differences other than numbers within tolerance: the first few, then a count.
    pub problems: Vec<String>,
    /// Values of the table not compared, or compared against something other than this
    /// program's value, each with the rule that applies (for example a value the C++
    /// leaves uninitialised). They do not count as differences.
    pub notes: Vec<String>,
    /// Cells handled by a rule: not compared ([`Item::excluded`]) or compared with the
    /// value of a named alternative ([`Item::explained`]); with the rules, each with its
    /// count.
    pub rules: Vec<(String, usize)>,
    /// Cells not compared (not counted in [`Item::values`]).
    pub excluded: usize,
    /// Cells compared with a named alternative rather than this program's value (counted
    /// in [`Item::values`]; [`Item::max_abs`] shows this program's own difference).
    pub explained: usize,
    pub verdict: Verdict,
    /// Values that differ from the reference within a named tolerance ([`NamedTolerance`]):
    /// the tolerance, the number of such values and the largest relative difference
    /// measured. Counted in [`Item::values`]; they make the verdict
    /// [`Verdict::WithinTolerance`].
    pub tolerated: Option<(NamedTolerance, usize, f64)>,
}

/// A tolerance that applies to named values only, for a difference the parity ledger
/// explains (for example the refined radius of ledger A37). Each is the bound measured on
/// the reference data, never wider.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NamedTolerance {
    /// What it covers and why, with the ledger item.
    pub rule: &'static str,
    /// The largest relative difference accepted.
    pub rel: f64,
    /// A value whose absolute difference is at most this is accepted too (values near 0,
    /// whose relative difference says nothing); 0 for none.
    pub abs: f64,
}

/// Ledger A37: the refined radius (`MarkerDetection::refinePointPolynomialFit`, written as
/// the point's size) comes from a least-squares solve of 30 or more rows, which XMALab's
/// OpenCV hands to LAPACK and this program solves with OpenCV's own QR. The bound is the
/// largest relative difference measured on the editing reference run `e10_place_refine`
/// (6.7e-15), rounded up to one significant digit.
pub const TOLERANCE_A37: NamedTolerance = NamedTolerance {
    rule: "refined radius, ledger A37 (least squares by OpenCV's QR here, LAPACK in XMALab)",
    rel: 7e-15,
    abs: 0.0,
};

/// Ledger A37 in the cube calibration: the six-point search's refits and the refinement
/// solve the DLT of 13 or more points, which XMALab's OpenCV hands to LAPACK and this
/// program solves with OpenCV's own Jacobi SVD (`xma_vision::cube`), so the camera matrix
/// and the poses of a calibrated camera agree to rounding: the calibration runs' camera
/// matrices and poses (`camera_matrix.csv`, `calibration_images.csv` and the saved
/// `_CameraMatrix.csv`, `_RotationMatrix.csv`, `_TranslationVector.csv`). The bound is the
/// largest relative difference measured on the calibration runs (8.3e-10, a translation
/// after the second recalibration of `c07_inlier_change`), rounded up to one significant
/// digit.
pub const TOLERANCE_A37_CUBE: NamedTolerance = NamedTolerance {
    rule: "camera matrix and pose after a cube calibration, ledger A37 (DLT of 13 or more \
           points by OpenCV's Jacobi SVD here, LAPACK in XMALab)",
    rel: 9e-10,
    abs: 0.0,
};

impl Item {
    /// The verdict as the report's row shows it: for
    /// [`Verdict::IdenticalExplained`] (and any verdict with rule cells), the number of
    /// cells and the rules; for values within a named tolerance, the tolerance, how many and
    /// the largest difference.
    #[must_use]
    pub fn verdict_label(&self) -> String {
        let base = self.rules_label();
        match &self.tolerated {
            Some((t, n, worst)) => format!(
                "{base}: {n} value{} within the named tolerance {:.0e} relative ({}); largest \
                 measured {worst:.1e}",
                if *n == 1 { "" } else { "s" },
                t.rel,
                t.rule
            ),
            None => base,
        }
    }

    fn rules_label(&self) -> String {
        let n = self.excluded + self.explained;
        if n == 0 {
            return self.verdict.label().to_owned();
        }
        let rules: Vec<String> = self.rules.iter().map(|(r, k)| format!("{k} {r}")).collect();
        let head = match self.verdict {
            Verdict::IdenticalExplained => format!(
                "identical apart from {n} explained cell{}",
                if n == 1 { "" } else { "s" }
            ),
            v => format!(
                "{}, with {n} explained cell{}",
                v.label(),
                if n == 1 { "" } else { "s" }
            ),
        };
        format!("{head} ({}; see notes)", rules.join("; "))
    }
}

/// A file of the dump that is not compared, or a group of them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NotCompared {
    pub path: String,
    pub reason: String,
}

/// The result of [`run_parity`].
#[derive(Clone, Debug)]
pub struct ParityReport {
    pub dataset: String,
    pub description: String,
    pub project: PathBuf,
    pub dump: PathBuf,
    /// Versions and other facts from `manifest.json`, in the manifest's order.
    pub reference: Vec<(String, String)>,
    /// Every setting in `manifest.json`, and whether the model uses it.
    pub settings: Vec<(String, String)>,
    pub items: Vec<Item>,
    pub not_compared: Vec<NotCompared>,
    /// The manifest's `missing` list.
    pub missing: Vec<NotCompared>,
    /// The manifest's warnings, the reader's and the engine's.
    pub warnings: Vec<String>,
    /// The engine's warnings alone (undefined behaviour in the C++ met while computing).
    pub engine_warnings: Vec<String>,
    pub seconds: f64,
    /// The compatibility mode the comparison ran in: [`CppCompat::XMALAB3`] for
    /// [`run_parity`].
    pub compat: CppCompat,
}

impl ParityReport {
    /// True when no item is different.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.items.iter().all(|i| i.verdict != Verdict::Different)
    }

    /// Number of items with a verdict.
    #[must_use]
    pub fn count(&self, verdict: Verdict) -> usize {
        self.items.iter().filter(|i| i.verdict == verdict).count()
    }

    /// Values compared over every item.
    #[must_use]
    pub fn values(&self) -> usize {
        self.items.iter().map(|i| i.values).sum()
    }

    /// One line: result, items and values.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{}: {} — {} tables and files compared ({} identical, {} identical apart from \
             explained cells, {} empty on both sides, {} within tolerance, {} different), {} \
             values; {} cell(s) explained by a rule; {} files not compared; {} tables missing \
             from the dump",
            self.dataset,
            if self.passed() { "PASS" } else { "FAIL" },
            self.items.len(),
            self.count(Verdict::Identical),
            self.count(Verdict::IdenticalExplained),
            self.count(Verdict::BothEmpty),
            self.count(Verdict::WithinTolerance),
            self.count(Verdict::Different),
            self.values(),
            self.explained_cells(),
            self.not_compared.len(),
            self.missing.len()
        )
    }

    /// Cells handled by a rule (not compared, or compared with a named alternative), over
    /// every item.
    #[must_use]
    pub fn explained_cells(&self) -> usize {
        self.items.iter().map(|i| i.excluded + i.explained).sum()
    }

    /// The report as Markdown.
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "# Parity report: {}\n", self.dataset);
        let _ = writeln!(
            s,
            "**Result: {}.** {} tables and export files compared: {} identical, {} identical \
             apart from explained cells, {} empty on both sides, {} within tolerance, {} \
             different; {} values in all; {} cell(s) explained by a rule (not compared, or \
             compared with a named alternative; see the notes). {} files of the dump not \
             compared; {} tables missing from the dump (both listed below; neither counts as \
             passing).\n",
            if self.passed() { "PASS" } else { "FAIL" },
            self.items.len(),
            self.count(Verdict::Identical),
            self.count(Verdict::IdenticalExplained),
            self.count(Verdict::BothEmpty),
            self.count(Verdict::WithinTolerance),
            self.count(Verdict::Different),
            self.values(),
            self.explained_cells(),
            self.not_compared.len(),
            self.missing.len()
        );
        if !self.description.is_empty() {
            let _ = writeln!(s, "{}\n", self.description);
        }
        let _ = writeln!(s, "- Project: `{}`", self.project.display());
        let _ = writeln!(s, "- Dump: `{}`", self.dump.display());
        let _ = writeln!(
            s,
            "- Compatibility mode: {} (`xma parity` always compares in the XMALab 3 mode)",
            self.compat.describe()
        );
        let _ = writeln!(
            s,
            "- Compared by: xma {} (engine {}), in {:.1} s\n",
            crate::VERSION,
            xma_engine::VERSION,
            self.seconds
        );

        let _ = writeln!(s, "## Reference program (manifest.json)\n");
        let _ = writeln!(s, "| Field | Value |\n|---|---|");
        for (k, v) in &self.reference {
            let _ = writeln!(s, "| {k} | {} |", cell(v));
        }

        let _ = writeln!(s, "\n## Settings (manifest.json, the user's)\n");
        let _ = writeln!(s, "| Setting | Value | Used here |\n|---|---|---|");
        for (k, v) in &self.settings {
            let _ = writeln!(
                s,
                "| {k} | {} | {} |",
                cell(v),
                if crate::settings::USED_SETTINGS.contains(&k.as_str()) {
                    "yes"
                } else {
                    "no"
                }
            );
        }
        let _ = writeln!(
            s,
            "\n\"yes\": read by phase 1 (DefaultMarkerThreshold by loading, ledger F19). \
             \"no\": held and applied, but read by nothing phase 1 computes. Every setting \
             the model holds is taken from here, overriding any settings entry in the \
             project, which the C++ ignores."
        );

        let _ = writeln!(s, "\n## Tiers\n");
        let _ = writeln!(s, "| Tier | What | Tolerance |\n|---|---|---|");
        for t in [Tier::Read, Tier::A, Tier::AUndistortion, Tier::B] {
            let _ = writeln!(
                s,
                "| {} | {} | {} |",
                t.label(),
                t.describe(),
                t.tolerance()
            );
        }

        let _ = writeln!(s, "\n## Compared\n");
        let _ = writeln!(
            s,
            "| Table or file | Tier | Values | Max abs diff | Where | Max rel diff | Where | Verdict |"
        );
        let _ = writeln!(s, "|---|---|---:|---:|---|---:|---|---|");
        for i in &self.items {
            let _ = writeln!(
                s,
                "| `{}` | {} | {} | {} | {} | {} | {} | {} |",
                i.name,
                i.tier.label(),
                i.values,
                num(i.max_abs),
                cell(&i.max_abs_at),
                num(i.max_rel),
                cell(&i.max_rel_at),
                cell(&i.verdict_label())
            );
        }
        let flagged: Vec<&Item> = self
            .items
            .iter()
            .filter(|i| !i.problems.is_empty())
            .collect();
        if !flagged.is_empty() {
            let _ = writeln!(s, "\n### Differences\n");
            for i in flagged {
                let _ = writeln!(s, "- `{}`:", i.name);
                for p in &i.problems {
                    let _ = writeln!(s, "  - {}", p.replace('\n', " "));
                }
            }
        }

        let noted: Vec<&Item> = self.items.iter().filter(|i| !i.notes.is_empty()).collect();
        if !noted.is_empty() {
            let _ = writeln!(s, "\n### Values not compared, or compared by a rule\n");
            for i in noted {
                let _ = writeln!(s, "- `{}`:", i.name);
                for n in &i.notes {
                    let _ = writeln!(s, "  - {}", n.replace('\n', " "));
                }
            }
        }

        let _ = writeln!(s, "\n## Not compared\n");
        if self.not_compared.is_empty() {
            let _ = writeln!(s, "Every file of the dump was compared.");
        } else {
            let _ = writeln!(s, "| File | Why |\n|---|---|");
            for n in &self.not_compared {
                let _ = writeln!(s, "| `{}` | {} |", n.path, cell(&n.reason));
            }
        }

        let _ = writeln!(s, "\n## Missing from the dump (manifest.json)\n");
        if self.missing.is_empty() {
            let _ = writeln!(s, "None.");
        } else {
            let _ = writeln!(s, "| Table | Reason given by the dump |\n|---|---|");
            for n in &self.missing {
                let _ = writeln!(s, "| `{}` | {} |", n.path, cell(&n.reason));
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

fn cell(s: &str) -> String {
    s.replace('|', "\\|").replace('\n', " ")
}

fn num(v: f64) -> String {
    if v == 0.0 {
        "0".to_owned()
    } else {
        format!("{v:.3e}")
    }
}

// ---------------------------------------------------------------------------------------
// One check
// ---------------------------------------------------------------------------------------

const MAX_PROBLEMS: usize = 8;

/// Collects the comparison of one table or file.
struct Check {
    name: String,
    tier: Tier,
    cmp: Comparison,
    problems: Vec<String>,
    n_problems: usize,
    notes: Vec<String>,
    rules: Vec<(String, usize)>,
    excluded: usize,
    explained: usize,
    /// This program's own differences at explained cells: label, absolute, relative.
    explained_diffs: Vec<(String, f64, f64)>,
    /// For export files: the bytes were equal.
    bytes_equal: Option<bool>,
    /// Numbers of an identical file, counted without being added one by one.
    equal_values: usize,
    /// Cells equal to an order of the C++ data race, by order: count and the first few.
    race_explained: BTreeMap<String, (usize, Vec<String>)>,
    /// Cells downstream of a raced pose that no order reproduces, by rule: count and the
    /// first few.
    race_downstream: BTreeMap<String, (usize, Vec<String>)>,
    /// Values within a named tolerance: the tolerance, count, largest relative difference,
    /// largest absolute difference and where.
    tolerated: Option<(NamedTolerance, usize, f64, f64, String)>,
}

/// Cells listed in a race note; the rest are counted.
const RACE_EXAMPLES: usize = 4;

/// Exactly equal, NaN equal to NaN.
fn same(a: f64, b: f64) -> bool {
    a == b || (a.is_nan() && b.is_nan())
}

type R<T> = Result<T, String>;

impl Check {
    fn new(name: impl Into<String>, tier: Tier) -> Self {
        let name = name.into();
        Self {
            cmp: Comparison::new(name.clone()),
            name,
            tier,
            problems: Vec::new(),
            n_problems: 0,
            notes: Vec::new(),
            rules: Vec::new(),
            excluded: 0,
            explained: 0,
            explained_diffs: Vec::new(),
            bytes_equal: None,
            equal_values: 0,
            race_explained: BTreeMap::new(),
            race_downstream: BTreeMap::new(),
            tolerated: None,
        }
    }

    /// A number compared exactly, except that a difference within the named tolerance `tol`
    /// (when given) is counted as within that tolerance rather than as a difference.
    fn num_named(
        &mut self,
        label: impl std::fmt::Display,
        got: f64,
        want: f64,
        tol: Option<NamedTolerance>,
    ) {
        if let Some(t) = tol
            && !same(got, want)
            && got.is_finite()
            && want.is_finite()
        {
            let abs = (got - want).abs();
            let rel = abs / got.abs().max(want.abs());
            if rel <= t.rel || abs <= t.abs {
                let e = self
                    .tolerated
                    .get_or_insert_with(|| (t, 0, 0.0, 0.0, String::new()));
                e.1 += 1;
                e.2 = e.2.max(rel);
                if abs > e.3 {
                    e.3 = abs;
                    e.4 = label.to_string();
                }
                return;
            }
        }
        self.num(label, got, want);
    }

    /// A number the C++ data race of ledger A56 can change ([`race`]). Equal to this
    /// program's value: compared as usual. Otherwise, equal (exactly) to the value of an order
    /// of the race in `race` (`alt` gives it for world `i`): explained, compared with that
    /// value (`alt` gives `None` when that order has no value: the order is skipped).
    /// Otherwise, when an order that a cell of the dump has shown exactly
    /// ([`race::World::established`]) also changes this cell, but not to the dump's value:
    /// not compared, as downstream of that raced pose. Otherwise an ordinary difference.
    fn race_num(
        &mut self,
        race: &Race,
        label: impl std::fmt::Display,
        got: f64,
        want: f64,
        alt: impl Fn(&race::World) -> Option<f64>,
    ) {
        if same(got, want) || !race.possible() {
            self.num(label, got, want);
            return;
        }
        let label = label.to_string();
        let worlds = race.worlds();
        for w in worlds.iter() {
            // A world with no value (its export lacks the line or field) is skipped.
            let Some(v) = alt(w) else {
                continue;
            };
            if same(v, want) {
                let key = format!(
                    "compared with the value of the other order of a C++ data race (ledger \
                     A56) in {}: {}, and identical to it",
                    race.update,
                    w.describe()
                );
                let e = self.race_explained.entry(key).or_default();
                e.0 += 1;
                if e.1.len() < RACE_EXAMPLES {
                    e.1.push(format!("{label}: here {got:e}, dump and race {want:e}"));
                }
                self.explain(RULE_RACE, label, got, want, v, None);
                return;
            }
        }
        for w in worlds.iter() {
            let Some(cell) = &w.established else {
                continue;
            };
            let Some(v) = alt(w) else {
                continue;
            };
            if !same(v, got) {
                let rule = format!("{RULE_DOWNSTREAM} ({cell})");
                let e = self.race_downstream.entry(rule.clone()).or_default();
                e.0 += 1;
                if e.1.len() < RACE_EXAMPLES {
                    e.1.push(format!(
                        "{label}: here {got:e}, dump {want:e}, in that order ({}) {v:e}",
                        w.describe()
                    ));
                }
                self.excluded += 1;
                self.rule(&rule, 1);
                return;
            }
        }
        self.num(label, got, want);
    }

    fn problem(&mut self, p: impl Into<String>) {
        self.n_problems += 1;
        if self.problems.len() < MAX_PROBLEMS {
            self.problems.push(p.into());
        }
    }

    fn num(&mut self, label: impl std::fmt::Display, got: f64, want: f64) {
        self.cmp.add(label, got, want);
    }

    /// An integer, flag or status: compared as a number, and any difference is a problem.
    fn int(&mut self, label: impl std::fmt::Display, got: i64, want: i64) {
        let label = label.to_string();
        if got != want {
            self.problem(format!("{label}: {got}, dump {want}"));
        }
        self.cmp.add(label, got as f64, want as f64);
    }

    fn rule(&mut self, rule: &str, cells: usize) {
        match self.rules.iter_mut().find(|(r, _)| r == rule) {
            Some(r) => r.1 += cells,
            None => self.rules.push((rule.to_owned(), cells)),
        }
    }

    /// `cells` values not compared, under a named rule, with a note saying why.
    fn exclude(&mut self, rule: &str, cells: usize, note: impl Into<String>) {
        self.excluded += cells;
        self.rule(rule, cells);
        self.notes.push(note.into());
    }

    /// A value compared with a named alternative (`alt`) instead of this program's own
    /// (`got`). The comparison counts against `alt`; the row's largest differences show
    /// `got` against `want`.
    #[allow(clippy::too_many_arguments)]
    fn explain(
        &mut self,
        rule: &str,
        label: impl std::fmt::Display,
        got: f64,
        want: f64,
        alt: f64,
        note: Option<String>,
    ) {
        let label = label.to_string();
        let abs = (got - want).abs();
        let rel = if want == 0.0 { abs } else { abs / want.abs() };
        self.explained += 1;
        self.rule(rule, 1);
        self.explained_diffs.push((label.clone(), abs, rel));
        self.cmp.add(label, alt, want);
        if let Some(n) = note {
            self.notes.push(n);
        }
    }

    fn text(&mut self, label: impl std::fmt::Display, got: &str, want: &str) {
        if got != want {
            self.problem(format!("{label}: '{got}', dump '{want}'"));
        }
    }

    fn rows(&mut self, got: usize, want: usize) {
        if got != want {
            self.problem(format!("{got} rows expected, the dump has {want}"));
        }
    }

    /// Reads the dump table; a problem when it cannot be read.
    fn table(&mut self, ds: &Dataset) -> Option<Table> {
        match ds.try_table(&self.name) {
            Ok(t) => Some(t),
            Err(e) => {
                self.problem(format!("cannot read the reference: {e}"));
                None
            }
        }
    }

    /// Runs `f` on every row; an error is a problem naming the row.
    fn each_row(&mut self, table: &Table, mut f: impl FnMut(&mut Self, &Row<'_>) -> R<()>) {
        for row in table.rows() {
            if let Err(e) = f(self, &row) {
                self.problem(format!("row {}: {e}", row.index() + 1));
            }
        }
    }

    fn finish(mut self) -> Item {
        for (what, (n, examples)) in std::mem::take(&mut self.race_explained) {
            let more = if n > examples.len() {
                format!("; {} more", n - examples.len())
            } else {
                String::new()
            };
            self.notes.push(format!(
                "{}: {what} ({n} cell{}: {}{more})",
                examples[0].split(':').next().unwrap_or_default(),
                if n == 1 { "" } else { "s" },
                examples.join("; ")
            ));
        }
        for (rule, (n, examples)) in std::mem::take(&mut self.race_downstream) {
            let more = if n > examples.len() {
                format!("; {} more", n - examples.len())
            } else {
                String::new()
            };
            self.notes.push(format!(
                "{n} cell{} {rule}: a C++ data race (ledger A56) changed this table, and no \
                 order replayed here reproduces these cells exactly: {}{more}",
                if n == 1 { "" } else { "s" },
                examples.join("; ")
            ));
        }
        let tol = self.tier.tolerance();
        let mut problems = self.problems;
        if self.n_problems > problems.len() {
            problems.push(format!("... {} more", self.n_problems - problems.len()));
        }
        let failures = self.cmp.failures(tol);
        if failures > 0
            && let Some(w) = self.cmp.worst(tol)
        {
            problems.push(format!(
                "{failures} of {} values outside {tol}; worst {w}",
                self.cmp.count()
            ));
        }
        let (mut max_abs, mut max_abs_at) = self
            .cmp
            .worst_abs()
            .map_or((0.0, String::new()), |w| (w.abs, w.label));
        let (mut max_rel, mut max_rel_at) = self
            .cmp
            .worst_rel()
            .map_or((0.0, String::new()), |w| (w.rel, w.label));
        let tolerated = self.tolerated.take();
        let n_tolerated = tolerated.as_ref().map_or(0, |t| t.1);
        if let Some((_, _, rel, abs, at)) = &tolerated {
            if *abs > max_abs {
                max_abs = *abs;
                max_abs_at = format!("{at} (named tolerance)");
            }
            if *rel > max_rel {
                max_rel = *rel;
                max_rel_at = format!("{at} (named tolerance)");
            }
        }
        let compared_equal = max_abs == 0.0;
        let ruled = self.excluded + self.explained > 0;
        let verdict = if !problems.is_empty() {
            Verdict::Different
        } else if self.bytes_equal.is_none() && self.cmp.count() == 0 && !ruled && n_tolerated == 0
        {
            Verdict::BothEmpty
        } else if self.bytes_equal == Some(true) || (self.bytes_equal.is_none() && compared_equal) {
            if ruled {
                Verdict::IdenticalExplained
            } else {
                Verdict::Identical
            }
        } else {
            Verdict::WithinTolerance
        };
        // The largest differences shown are this program's own, explained cells included.
        for (label, abs, rel) in &self.explained_diffs {
            if *abs > max_abs {
                max_abs = *abs;
                max_abs_at = format!("{label} (explained)");
            }
            if *rel > max_rel {
                max_rel = *rel;
                max_rel_at = format!("{label} (explained)");
            }
        }
        Item {
            name: self.name,
            tier: self.tier,
            values: self.cmp.count() + self.equal_values + n_tolerated,
            max_abs: if max_abs == 0.0 { 0.0 } else { max_abs },
            max_abs_at: if max_abs == 0.0 {
                String::new()
            } else {
                max_abs_at
            },
            max_rel,
            max_rel_at: if max_rel == 0.0 {
                String::new()
            } else {
                max_rel_at
            },
            problems,
            notes: self.notes,
            rules: self.rules,
            excluded: self.excluded,
            explained: self.explained,
            verdict,
            tolerated: tolerated.map(|(t, n, rel, _, _)| (t, n, rel)),
        }
    }
}

fn e(err: xma_oracle::OracleError) -> String {
    err.to_string()
}
fn ru(r: &Row<'_>, c: &str) -> R<usize> {
    r.try_usize(c).map_err(e)
}
fn ri(r: &Row<'_>, c: &str) -> R<i64> {
    r.try_i64(c).map_err(e)
}
fn rf(r: &Row<'_>, c: &str) -> R<f64> {
    r.try_f64(c).map_err(e)
}
fn rs<'a>(r: &Row<'a>, c: &str) -> R<&'a str> {
    r.try_str(c).map_err(e)
}
fn get<'a, T>(v: &'a [T], i: usize, what: &str) -> R<&'a T> {
    v.get(i)
        .ok_or_else(|| format!("{what} {i} does not exist here"))
}

/// Names as the dump writes them: commas become `;`, line breaks a space
/// (`oracle/cpp/DEVIATIONS.md`, item 1).
fn dump_text(s: &str) -> String {
    s.replace(',', ";").replace(['\n', '\r'], " ")
}

// ---------------------------------------------------------------------------------------
// Export files
// ---------------------------------------------------------------------------------------

/// Splits an export line into fields: `,` separated, surrounding spaces ignored (OpenCV's
/// CSV writes `", "`).
fn fields(line: &str) -> Vec<&str> {
    if line.is_empty() {
        Vec::new()
    } else {
        line.split(',').map(str::trim).collect()
    }
}

/// The text around each field of a line that [`fields`] trims away: (before, after).
fn spacing(line: &str) -> Vec<(&str, &str)> {
    line.split(',')
        .map(|f| {
            let start = f.len() - f.trim_start().len();
            let end = start + f.trim().len();
            (&f[..start], &f[end..])
        })
        .collect()
}

/// Compares an export file with the dump's, byte for byte; when the bytes differ, field by
/// field (numbers within the tier's tolerance, text equal).
fn export_file(ds: &Dataset, rel: &str, got: &str, tier: Tier) -> Item {
    export_file_race(ds, rel, got, tier, &Race::none())
}

/// [`export_file`] for a file the C++ data race can change: a field that differs is compared
/// with the same field of each order's file ([`race::World::exports`], by `rel`), as
/// [`Check::race_num`] does.
fn export_file_race(ds: &Dataset, rel: &str, got: &str, tier: Tier, race: &Race) -> Item {
    export_file_impl(ds, rel, got, tier, race, None)
}

/// [`export_file`] with the numbers compared within the named tolerance `tol`.
fn export_file_tolerant(
    ds: &Dataset,
    rel: &str,
    got: &str,
    tier: Tier,
    tol: Option<NamedTolerance>,
) -> Item {
    export_file_impl(ds, rel, got, tier, &Race::none(), tol)
}

fn export_file_impl(
    ds: &Dataset,
    rel: &str,
    got: &str,
    tier: Tier,
    race: &Race,
    tol: Option<NamedTolerance>,
) -> Item {
    let mut c = Check::new(rel, tier);
    let want = match std::fs::read(ds.path(rel)) {
        Ok(w) => w,
        Err(err) => {
            c.problem(format!("cannot read the reference: {err}"));
            return c.finish();
        }
    };
    let want = String::from_utf8_lossy(&want).into_owned();
    if got == want {
        c.bytes_equal = Some(true);
        // Count the numbers compared; they are all equal.
        c.equal_values = want
            .lines()
            .flat_map(fields)
            .filter(|f| xma_oracle::parse_f64(f).is_some())
            .count();
        return c.finish();
    }
    c.bytes_equal = Some(false);
    let gl: Vec<&str> = got.split('\n').collect();
    let wl: Vec<&str> = want.split('\n').collect();
    if gl.len() != wl.len() {
        c.problem(format!("{} lines, the dump has {}", gl.len(), wl.len()));
    }
    for (i, (g, w)) in gl.iter().zip(&wl).enumerate() {
        let (gf, wf) = (fields(g), fields(w));
        if gf.len() != wf.len() {
            c.problem(format!(
                "line {}: {} fields, the dump has {}",
                i + 1,
                gf.len(),
                wf.len()
            ));
            continue;
        }
        if spacing(g) != spacing(w) {
            // What `fields` trims (spaces around a comma, a carriage return) differs.
            c.problem(format!(
                "line {}: the spaces or line end around the fields differ: here {:?}, dump {:?}",
                i + 1,
                g.chars().take(160).collect::<String>(),
                w.chars().take(160).collect::<String>()
            ));
        }
        for (k, (a, b)) in gf.iter().zip(&wf).enumerate() {
            match (xma_oracle::parse_f64(a), xma_oracle::parse_f64(b)) {
                (Some(x), Some(y)) if same(x, y) && a != b => {
                    // Equal numbers written differently (`-0` against `0`, say): not identical.
                    c.num(format_args!("line {} field {}", i + 1, k + 1), x, y);
                    c.problem(format!(
                        "line {} field {}: '{a}', dump '{b}': equal numbers, but the bytes \
                         differ (number formatting)",
                        i + 1,
                        k + 1
                    ));
                }
                (Some(x), Some(y)) if tol.is_some() => {
                    c.num_named(format_args!("line {} field {}", i + 1, k + 1), x, y, tol);
                }
                (Some(x), Some(y)) => c.race_num(
                    race,
                    format_args!("line {} field {}", i + 1, k + 1),
                    x,
                    y,
                    |w| {
                        w.exports
                            .get(rel)
                            .and_then(|text| text.split('\n').nth(i))
                            .and_then(|l| fields(l).get(k).and_then(|f| xma_oracle::parse_f64(f)))
                    },
                ),
                _ if a != b => {
                    c.problem(format!("line {} field {}: '{a}', dump '{b}'", i + 1, k + 1))
                }
                _ => {}
            }
        }
    }
    for (world, err) in race.export_failures(rel) {
        c.problem(format!(
            "the export failed in the other order of the C++ data race ({world}): {err}"
        ));
    }
    if c.problems.is_empty()
        && c.cmp.count() > 0
        && c.cmp.max_abs() == 0.0
        && c.explained + c.excluded == 0
        && c.tolerated.is_none()
    {
        // The bytes differ, but no field does (spaces, say): not identical.
        c.problem("the fields are equal but the bytes differ");
    }
    c.finish()
}

/// Compares a set of files with a dump folder: one item per file, and one for the set of
/// names when it differs.
fn export_folder(
    items: &mut Vec<Item>,
    seen: &mut BTreeSet<String>,
    ds: &Dataset,
    rel: &str,
    files: &[ExportFile],
    tier: Tier,
    keep: impl Fn(&str) -> bool,
) {
    let want: BTreeSet<String> = std::fs::read_dir(ds.path(rel))
        .map(|d| {
            d.filter_map(Result::ok)
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| keep(n))
                .collect()
        })
        .unwrap_or_default();
    let got: BTreeSet<String> = files.iter().map(|f| f.name.clone()).collect();
    if want != got {
        let mut c = Check::new(format!("{rel}/"), tier);
        c.problem(format!("file names differ: here {got:?}, dump {want:?}"));
        items.push(c.finish());
    }
    for f in files {
        if want.contains(&f.name) {
            let path = format!("{rel}/{}", f.name);
            seen.insert(path.clone());
            items.push(export_file(ds, &path, &f.contents, tier));
        }
    }
}

// ---------------------------------------------------------------------------------------
// Locating the dataset
// ---------------------------------------------------------------------------------------

/// The dataset for a name in `datasets.json`, or a dump directory with its project.
///
/// # Errors
/// An unknown name, absent data, or a dump directory whose project cannot be found.
pub fn locate(arg: &str, project: Option<&Path>) -> Result<Dataset, String> {
    let as_dir = Path::new(arg);
    if as_dir.is_dir() {
        let manifest = Manifest::try_read(as_dir.join("manifest.json"))
            .map_err(|e| format!("{arg} is not a dump directory: {e}"))?;
        let project = match project {
            Some(p) => p.to_path_buf(),
            None => manifest
                .json()
                .get("project_path")
                .and_then(|v| v.as_str())
                .map(PathBuf::from)
                .ok_or_else(|| {
                    format!("{arg}: the manifest records no project_path; give --project")
                })?,
        };
        let name = as_dir
            .canonicalize()
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
            .unwrap_or_else(|| arg.to_owned());
        let ds = Dataset::from_paths(name, project, as_dir);
        if !ds.project.is_file() {
            return Err(format!(
                "project {} not found; give --project",
                ds.project.display()
            ));
        }
        return Ok(ds);
    }
    let entry = xma_oracle::try_dataset_entry(arg).map_err(|e| e.to_string())?;
    let Some(mut ds) = entry else {
        let names = xma_oracle::try_dataset_names()
            .map(|n| n.join(", "))
            .unwrap_or_default();
        return Err(format!(
            "'{arg}' is neither a dump directory nor a dataset in {} (known: {names})",
            xma_oracle::datasets_file().display()
        ));
    };
    if let Some(p) = project {
        ds.project = p.to_path_buf();
    }
    if !ds.is_present() {
        return Err(format!(
            "dataset '{arg}' is not present: {} or {} is missing",
            ds.project.display(),
            ds.dump.display()
        ));
    }
    Ok(ds)
}

// ---------------------------------------------------------------------------------------
// The replay
// ---------------------------------------------------------------------------------------

/// Runs the comparison for one dataset.
///
/// # Errors
/// When the manifest or the project cannot be read. Problems with individual tables are
/// reported in the result, not as errors.
pub fn run_parity(ds: &Dataset) -> Result<ParityReport, String> {
    run_parity_with(ds, CppCompat::XMALAB3)
}

/// [`run_parity`] in another compatibility mode. The reference dumps are the C++ program's,
/// so only [`CppCompat::XMALAB3`] is expected to match them; another mode measures what the
/// fixes change on the reference data.
///
/// # Errors
/// As [`run_parity`].
pub fn run_parity_with(ds: &Dataset, compat: CppCompat) -> Result<ParityReport, String> {
    let start = Instant::now();
    let manifest = ds.try_manifest().map_err(|e| e.to_string())?;
    let options = OpenOptions {
        overrides: crate::settings::manifest_settings(&manifest)?,
        compat,
        ..OpenOptions::default()
    };
    let loaded = xma_engine::read_project(&ds.project, &options)
        .map_err(|e| format!("{}: {e}", ds.project.display()))?;
    let mut doc = Document::from_loaded(loaded);
    doc.set_path(&ds.project);
    // Kept to replay the update after loading in the orders of the C++ data race, when
    // that update determines the "as held" tables of its trial (see the trial loop).
    let before_load = doc.clone();
    let loaded_trial = doc.update_after_load();

    let mut items = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let push = |items: &mut Vec<Item>, seen: &mut BTreeSet<String>, item: Item| {
        seen.insert(item.name.clone());
        items.push(item);
    };

    // Project level.
    for item in [
        check_cameras(ds, &doc),
        check_camera_matrix(ds, &doc),
        check_distortion(ds, &doc),
        check_calibration_images(ds, &doc),
        check_trials(ds, &doc),
    ] {
        push(&mut items, &mut seen, item);
    }
    if ds.has("undistortion/grid_points.csv") {
        push(&mut items, &mut seen, check_grid_points(ds, &doc));
    }
    if ds.has("undistortion/samples.csv") {
        push(&mut items, &mut seen, check_samples(ds, &doc));
    }
    if ds.has("undistortion/camera_undistort_samples.csv") {
        push(&mut items, &mut seen, check_camera_samples(ds, &doc));
    }
    for (rel, all) in [
        ("calibration/cube_points.csv", false),
        ("calibration/cube_points_all.csv", true),
    ] {
        if ds.has(rel) {
            push(&mut items, &mut seen, check_cube_points(ds, &doc, rel, all));
        }
    }
    // Format 2: what loading determines (written after the version 1 top-level tables).
    type P = fn(&Dataset, &Document) -> Item;
    let format2: [(&str, P); 5] = [
        ("camera_details.csv", check_camera_details),
        ("trial_details.csv", check_trial_details),
        ("calibration_object.csv", check_calibration_object),
        (
            "calibration_object_points.csv",
            check_calibration_object_points,
        ),
        (
            "calibration_object_references.csv",
            check_calibration_object_references,
        ),
    ];
    for (rel, f) in format2 {
        if ds.has(rel) {
            push(&mut items, &mut seen, f(ds, &doc));
        }
    }
    project_exports(&mut items, &mut seen, ds, &mut doc);

    // Trials.
    let user = doc.settings().clone();
    let mut selected_before = false;
    for t in 0..doc.project().trials.len() {
        let dir = format!("trial_{t}");
        if !ds.path(&dir).is_dir() {
            let mut c = Check::new(format!("{dir}/"), Tier::Read);
            c.problem("the trial has no folder in the dump");
            items.push(c.finish());
            continue;
        }
        // The orders of the C++ data race (ledger A56) in the update that determines the
        // "as held" tables: selecting the trial when that processes frames; otherwise the
        // update after loading, when it was this trial's and no trial has been selected yet
        // (beyond that the replay would have to repeat the other trials' steps; no dataset
        // needs it).
        let processes = {
            let tr = doc.trial(t);
            tr.requires_recomputation && tr.n_frames > 0
        };
        let race = if processes {
            Race::new(RACE_SNAPSHOT, &doc, t, move |d| d.select_trial(t))
        } else if loaded_trial == Some(t) && !selected_before {
            Race::new(RACE_SNAPSHOT, &before_load, t, move |d| {
                d.update_after_load();
                d.select_trial(t);
            })
        } else {
            Race::none()
        };
        doc.select_trial(t);
        selected_before = true;
        establish(ds, &doc, t, &format!("{dir}/rb_errors.csv"), &race, false);
        for item in snapshot(ds, &doc, t, &race) {
            push(&mut items, &mut seen, item);
        }
        // The detection and tracker-step tables of a `--frame-step 1` dump (phase 3).
        for rel in steps::tables(ds, t) {
            seen.insert(rel);
        }
        items.extend(steps::check_step_tables(ds, t));
        if doc.trial(t).n_frames == 0 {
            continue;
        }
        trial_exports(&mut items, &mut seen, ds, &mut doc, t, &race);
        drop(race);
        for k in 0..5u8 {
            doc.settings_mut().triangulation_method = k;
            doc.set_trial_requires_recomputation(t, true);
            doc.update_trial(t);
            let rel = format!("{dir}/points3d_method{k}.csv");
            push(&mut items, &mut seen, check_points3d(ds, &doc, t, &rel));
        }
        doc.settings_mut().triangulation_method = user.triangulation_method;
        for (optimize, file) in [
            (false, "rb_pose_kabsch.csv"),
            (true, "rb_pose_optimized.csv"),
        ] {
            doc.settings_mut().optimize_rigid_body = optimize;
            doc.set_trial_requires_recomputation(t, true);
            // With the refinement off the race changes nothing (`Race::build` gives none).
            let race = Race::new(RACE_OPTIMIZED, &doc, t, move |d| {
                d.update_trial(t);
            });
            doc.update_trial(t);
            let rel = format!("{dir}/{file}");
            establish(ds, &doc, t, &rel, &race, true);
            push(
                &mut items,
                &mut seen,
                check_rb_poses(ds, &doc, t, &rel, false, &race),
            );
        }
        doc.settings_mut().optimize_rigid_body = user.optimize_rigid_body;
        // `dumpTrial` ends with an update in the user's settings. Nothing compared depends
        // on it unless another trial follows (the scheduler queue is shared).
        if t + 1 < doc.project().trials.len() {
            doc.set_trial_requires_recomputation(t, true);
            doc.update_trial(t);
        }
    }

    // Everything else in the dump.
    let missing: Vec<NotCompared> = manifest
        .try_missing()
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|m| NotCompared {
            path: m.table,
            reason: m.reason,
        })
        .collect();
    let not_compared = not_compared(ds, &seen);

    let mut warnings = manifest.warnings();
    warnings.extend(doc.warnings().iter().cloned());
    warnings.extend(doc.extras().warnings.iter().cloned());
    let engine_warnings = doc.warnings().to_vec();

    let reference = [
        "format_version",
        "xmalab_version",
        "xmalab_source_commit",
        "opencv_version",
        "opencv_use_opencl",
        "qt_version",
        "qt_version_compiled",
        "platform",
        "compiler",
        "project_path",
        "frame_step",
        "calibration_type",
        "n_cameras",
        "n_trials",
    ]
    .iter()
    .filter_map(|k| {
        manifest.json().get(*k).map(|v| {
            let text = v.as_str().map_or_else(|| v.to_string(), str::to_owned);
            ((*k).to_owned(), text)
        })
    })
    .collect();
    let settings = manifest
        .try_settings()
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|s| (s.name, s.value.to_string()))
        .collect();

    Ok(ParityReport {
        dataset: ds.name.clone(),
        description: ds.description.clone(),
        project: ds.project.clone(),
        dump: ds.dump.clone(),
        reference,
        settings,
        items,
        not_compared,
        missing,
        warnings,
        engine_warnings,
        seconds: start.elapsed().as_secs_f64(),
        compat,
    })
}

/// `exports/mayacams2_all` and the undistortion files of `exports/export_all`
/// (`writeProjectExports`, before any trial is selected).
fn project_exports(
    items: &mut Vec<Item>,
    seen: &mut BTreeSet<String>,
    ds: &Dataset,
    doc: &mut Document,
) {
    let t = doc.project().active_trial;
    if doc.project().trials.is_empty() {
        return;
    }
    if ds.path("exports/mayacams2_all").is_dir() {
        match doc.export(t, &Export::MayaCams { all: true }) {
            Ok(out) => export_folder(
                items,
                seen,
                ds,
                "exports/mayacams2_all",
                &folder(out),
                Tier::A,
                |_| true,
            ),
            Err(err) => items.push(refused("exports/mayacams2_all/", &err.to_string())),
        }
    }
    if ds.path("exports/export_all").is_dir() {
        match doc.export(t, &Export::UndistortionLut) {
            Ok(out) => export_folder(
                items,
                seen,
                ds,
                "exports/export_all",
                &folder(out),
                Tier::AUndistortion,
                is_lut_file,
            ),
            Err(err) => items.push(refused("exports/export_all/", &err.to_string())),
        }
    }
}

fn is_lut_file(n: &str) -> bool {
    n.ends_with("_LUT.csv") || n.ends_with("_INPTS.csv") || n.ends_with("_BSPTS.csv")
}

fn folder(out: ExportOutput) -> Vec<ExportFile> {
    match out {
        ExportOutput::Folder(f) => f,
        ExportOutput::File(text) => vec![ExportFile {
            name: String::new(),
            contents: text,
        }],
    }
}

fn file(out: ExportOutput) -> String {
    match out {
        ExportOutput::File(t) => t,
        ExportOutput::Folder(f) => f.into_iter().map(|f| f.contents).collect(),
    }
}

fn refused(name: &str, why: &str) -> Item {
    let mut c = Check::new(name, Tier::A);
    c.problem(format!(
        "the export was refused here, but the dump has it: {why}"
    ));
    c.finish()
}

/// The trial exports, in `writeTrialExports`' order.
fn trial_exports(
    items: &mut Vec<Item>,
    seen: &mut BTreeSet<String>,
    ds: &Dataset,
    doc: &mut Document,
    t: usize,
    race: &Race,
) {
    let ex = format!("trial_{t}/exports");
    let (speed, cutoff, start, end) = {
        let tr = doc.trial(t);
        (
            tr.recording_speed,
            tr.cutoff_frequency,
            tr.start_frame,
            tr.end_frame,
        )
    };
    let one = |items: &mut Vec<Item>,
               seen: &mut BTreeSet<String>,
               doc: &mut Document,
               rel: String,
               export: Export,
               tier: Tier| {
        if !ds.has(&rel) {
            return;
        }
        seen.insert(rel.clone());
        match doc.export(t, &export) {
            Ok(out) => items.push(export_file(ds, &rel, &file(out), tier)),
            Err(err) => items.push(refused(&rel, &err.to_string())),
        }
    };

    one(
        items,
        seen,
        doc,
        format!("{ex}/3DPoints.csv"),
        Export::Points3d {
            markers: None,
            options: Points3dOptions::default(),
        },
        Tier::A,
    );
    if speed > 0.0 && cutoff > 0.0 {
        one(
            items,
            seen,
            doc,
            format!("{ex}/3DPoints_filtered.csv"),
            Export::Points3d {
                markers: None,
                options: Points3dOptions {
                    filter_frequency: cutoff,
                    ..Points3dOptions::default()
                },
            },
            Tier::A,
        );
    }
    one(
        items,
        seen,
        doc,
        format!("{ex}/2DPoints.csv"),
        Export::Points2d(Points2dOptions::default()),
        Tier::Read,
    );
    one(
        items,
        seen,
        doc,
        format!("{ex}/2DPoints_undistorted.csv"),
        Export::Points2d(Points2dOptions {
            distorted: false,
            ..Points2dOptions::default()
        }),
        Tier::AUndistortion,
    );
    // Each recomputes the bodies (ledger A36). The driver always writes the unfiltered one,
    // and the filtered one when the GUI allows it. The recomputation reads other bodies'
    // held poses, so an order of the C++ data race in the update before can change them:
    // each order replays them too.
    let unfiltered = Export::RigidBodies {
        bodies: None,
        options: TransformationsOptions {
            filtered: false,
            ..TransformationsOptions::default()
        },
    };
    let filtered = Export::RigidBodies {
        bodies: None,
        options: TransformationsOptions::default(),
    };
    let mut rb = vec![(format!("{ex}/RigidBodyTransformations.csv"), unfiltered)];
    let all: Vec<usize> = (0..doc.trial(t).rigid_bodies.len()).collect();
    if can_export_filtered(doc.trial(t), &all, doc.compat()) {
        rb.push((
            format!("{ex}/RigidBodyTransformations_filtered.csv"),
            filtered,
        ));
    }
    for (rel, export) in &rb {
        let out = doc.export(t, export);
        // Not in the dump: the driver ran it all the same, and so has the replay.
        if !ds.has(rel) {
            continue;
        }
        seen.insert(rel.clone());
        match out {
            Ok(out) => {
                let text = file(out);
                if std::fs::read(ds.path(rel)).ok().as_deref() != Some(text.as_bytes()) {
                    race.replay_exports(t, &rb);
                }
                items.push(export_file_race(ds, rel, &text, Tier::B, race));
            }
            Err(err) => items.push(refused(rel, &err.to_string())),
        }
    }
    if ds.path(&format!("{ex}/events")).is_dir() {
        match doc.export(t, &Export::Events) {
            Ok(out) => export_folder(
                items,
                seen,
                ds,
                &format!("{ex}/events"),
                &folder(out),
                Tier::Read,
                |_| true,
            ),
            Err(err) => items.push(refused(&format!("{ex}/events/"), &err.to_string())),
        }
    } else if !doc.trial(t).events.is_empty() {
        let mut c = Check::new(format!("{ex}/events/"), Tier::Read);
        c.problem("the trial has events but the dump has no events export");
        items.push(c.finish());
    }
    let (from, to) = (start.saturating_sub(1), end);
    one(
        items,
        seen,
        doc,
        format!("{ex}/MarkerToMarkerDistances.csv"),
        Export::MarkerDistances { from, to },
        Tier::A,
    );
    one(
        items,
        seen,
        doc,
        format!("{ex}/PrecisionInfo.csv"),
        Export::PrecisionInfo { from, to },
        Tier::B,
    );
    for (sub, export, tier) in [
        (
            "reprojection_errors",
            Export::ReprojectionErrors,
            Tier::AUndistortion,
        ),
        (
            "mayacams2_reference",
            Export::MayaCams { all: false },
            Tier::A,
        ),
    ] {
        let rel = format!("{ex}/{sub}");
        if !ds.path(&rel).is_dir() {
            continue;
        }
        match doc.export(t, &export) {
            Ok(out) => export_folder(items, seen, ds, &rel, &folder(out), tier, |_| true),
            Err(err) => items.push(refused(&format!("{rel}/"), &err.to_string())),
        }
    }
}

// ---------------------------------------------------------------------------------------
// Project-level tables
// ---------------------------------------------------------------------------------------

fn check_cameras(ds: &Dataset, doc: &Document) -> Item {
    let mut c = Check::new("cameras.csv", Tier::Read);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let cams = &doc.project().cameras;
    c.rows(cams.len(), table.len());
    c.each_row(&table, |c, r| {
        let i = ru(r, "camera")?;
        let cam = get(cams, i, "camera")?;
        c.text(
            format_args!("camera {i} name"),
            &dump_text(&cam.name),
            rs(r, "name")?,
        );
        c.int(
            format_args!("camera {i} width"),
            cam.width.into(),
            ri(r, "width")?,
        );
        c.int(
            format_args!("camera {i} height"),
            cam.height.into(),
            ri(r, "height")?,
        );
        for (col, v) in [
            ("is_light", cam.is_light_camera),
            ("is_calibrated", cam.calibrated),
            ("is_optimized", cam.optimized),
            ("is_flipped", cam.flipped),
            ("has_undistortion", cam.undistortion.is_some()),
        ] {
            c.int(format_args!("camera {i} {col}"), v.into(), ri(r, col)?);
        }
        Ok(())
    });
    c.finish()
}

fn check_camera_matrix(ds: &Dataset, doc: &Document) -> Item {
    let mut c = Check::new("camera_matrix.csv", Tier::Read);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let cams = &doc.project().cameras;
    c.rows(9 * cams.len(), table.len());
    c.each_row(&table, |c, r| {
        let (i, row, col) = (ru(r, "camera")?, ru(r, "row")?, ru(r, "col")?);
        if row > 2 || col > 2 {
            return Err(format!("({row}, {col}) is outside a 3 x 3 matrix"));
        }
        let cam = get(cams, i, "camera")?;
        c.num(
            format_args!("camera {i} ({row},{col})"),
            cam.camera_matrix[(row, col)],
            rf(r, "value")?,
        );
        Ok(())
    });
    c.finish()
}

fn check_distortion(ds: &Dataset, doc: &Document) -> Item {
    let mut c = Check::new("distortion.csv", Tier::Read);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let cams = &doc.project().cameras;
    c.each_row(&table, |c, r| {
        let (i, k) = (ru(r, "camera")?, ru(r, "index")?);
        let cam = get(cams, i, "camera")?;
        let got = cam.distortion.get(k).copied().unwrap_or(f64::NAN);
        c.num(format_args!("camera {i} k{k}"), got, rf(r, "value")?);
        Ok(())
    });
    c.finish()
}

fn check_calibration_images(ds: &Dataset, doc: &Document) -> Item {
    let mut c = Check::new("calibration_images.csv", Tier::Read);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let cams = &doc.project().cameras;
    let n: usize = cams.iter().map(|c| c.calibration_images.len()).sum();
    c.rows(n, table.len());
    c.each_row(&table, |c, r| {
        let (i, k) = (ru(r, "camera")?, ru(r, "image")?);
        let im = get(&get(cams, i, "camera")?.calibration_images, k, "image")?;
        c.int(
            format_args!("camera {i} image {k} is_calibrated"),
            im.calibrated.into(),
            ri(r, "is_calibrated")?,
        );
        if im.is_calibrated() {
            for (j, axis) in ["x", "y", "z"].iter().enumerate() {
                c.num(
                    format_args!("camera {i} image {k} r{axis}"),
                    im.rotation[j],
                    rf(r, &format!("r{axis}"))?,
                );
                c.num(
                    format_args!("camera {i} image {k} t{axis}"),
                    im.translation[j],
                    rf(r, &format!("t{axis}"))?,
                );
            }
        }
        Ok(())
    });
    c.finish()
}

fn check_trials(ds: &Dataset, doc: &Document) -> Item {
    check_trials_with(ds, doc, None)
}

/// [`check_trials`] with the reference calibrations the dump holds as -1 not compared, under
/// the rule `u62` (ledger U62: XMALab's Trial dialog writes -1 when it opens); every other
/// cell compared as usual.
fn check_trials_u62(ds: &Dataset, doc: &Document, u62: &str) -> Item {
    check_trials_with(ds, doc, Some(u62))
}

fn check_trials_with(ds: &Dataset, doc: &Document, u62: Option<&str>) -> Item {
    let mut c = Check::new("trials.csv", Tier::Read);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let trials = &doc.project().trials;
    c.rows(trials.len(), table.len());
    c.each_row(&table, |c, r| {
        let i = ru(r, "trial")?;
        let t = get(trials, i, "trial")?;
        c.text(
            format_args!("trial {i} name"),
            &dump_text(&t.name),
            rs(r, "name")?,
        );
        let ints: [(&str, i64); 6] = [
            ("is_default", t.is_default.into()),
            ("start_frame", t.start_frame as i64),
            ("end_frame", t.end_frame as i64),
            ("n_frames", t.n_frames as i64),
            ("interpolate_3d", t.interpolate_3d.into()),
            (
                "reference_calibration",
                t.reference_calibration_image as i64,
            ),
        ];
        for (col, v) in ints {
            let want = ri(r, col)?;
            if let Some(rule) = u62
                && col == "reference_calibration"
                && want == -1
                && v != want
            {
                c.exclude(
                    rule,
                    1,
                    format!("trial {i} reference_calibration: {v}, dump -1: not compared"),
                );
                continue;
            }
            c.int(format_args!("trial {i} {col}"), v, want);
        }
        c.num(
            format_args!("trial {i} recording_speed"),
            t.recording_speed,
            rf(r, "recording_speed")?,
        );
        c.num(
            format_args!("trial {i} cutoff_frequency"),
            t.cutoff_frequency,
            rf(r, "cutoff_frequency")?,
        );
        Ok(())
    });
    c.finish()
}

fn check_grid_points(ds: &Dataset, doc: &Document) -> Item {
    let mut c = Check::new("undistortion/grid_points.csv", Tier::Read);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let cams = &doc.project().cameras;
    let n: usize = cams
        .iter()
        .filter_map(|c| c.undistortion.as_ref())
        .map(|g| g.grid_distorted.len())
        .sum();
    c.rows(n, table.len());
    c.each_row(&table, |c, r| {
        let (i, k) = (ru(r, "camera")?, ru(r, "index")?);
        let g = get(cams, i, "camera")?
            .undistortion
            .as_ref()
            .ok_or_else(|| format!("camera {i} has no grid here"))?;
        let d = get(&g.grid_distorted, k, "grid point")?;
        let rf_ = get(&g.grid_references, k, "reference point")?;
        let inl = get(&g.grid_inlier, k, "inlier flag")?;
        c.num(
            format_args!("camera {i} point {k} detected_x"),
            d.x,
            rf(r, "detected_x")?,
        );
        c.num(
            format_args!("camera {i} point {k} detected_y"),
            d.y,
            rf(r, "detected_y")?,
        );
        c.num(
            format_args!("camera {i} point {k} reference_x"),
            rf_.x,
            rf(r, "reference_x")?,
        );
        c.num(
            format_args!("camera {i} point {k} reference_y"),
            rf_.y,
            rf(r, "reference_y")?,
        );
        c.int(
            format_args!("camera {i} point {k} inlier"),
            (*inl).into(),
            ri(r, "inlier")?,
        );
        Ok(())
    });
    c.finish()
}

/// `samples.csv`: the grid transform alone, undistorting then distorting the result
/// (`UndistortionObject::transformPoint`).
fn check_samples(ds: &Dataset, doc: &Document) -> Item {
    let mut c = Check::new("undistortion/samples.csv", Tier::AUndistortion);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let n_cams = doc.project().cameras.len();
    c.each_row(&table, |c, r| {
        let i = ru(r, "camera")?;
        if i >= n_cams {
            return Err(format!("camera {i} does not exist here"));
        }
        let model = &doc.camera(i).model;
        if model.grid.is_none() {
            return Err(format!("camera {i} has no fitted grid here"));
        }
        let p = xma_model::Point2::new(rf(r, "x")?, rf(r, "y")?);
        let u = model.undistort_point(p, true, false, true);
        let d = model.undistort_point(u, false, false, true);
        let l = format!("camera {i} ({}, {})", p.x, p.y);
        c.num(
            format_args!("{l} undistorted_x"),
            u.x,
            rf(r, "undistorted_x")?,
        );
        c.num(
            format_args!("{l} undistorted_y"),
            u.y,
            rf(r, "undistorted_y")?,
        );
        c.num(
            format_args!("{l} redistorted_x"),
            d.x,
            rf(r, "redistorted_x")?,
        );
        c.num(
            format_args!("{l} redistorted_y"),
            d.y,
            rf(r, "redistorted_y")?,
        );
        Ok(())
    });
    c.finish()
}

/// `camera_undistort_samples.csv`: `Camera::undistortPoint(p, true)` (grid, then lens).
fn check_camera_samples(ds: &Dataset, doc: &Document) -> Item {
    let mut c = Check::new(
        "undistortion/camera_undistort_samples.csv",
        Tier::AUndistortion,
    );
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let n_cams = doc.project().cameras.len();
    c.each_row(&table, |c, r| {
        let i = ru(r, "camera")?;
        if i >= n_cams {
            return Err(format!("camera {i} does not exist here"));
        }
        let p = xma_model::Point2::new(rf(r, "x")?, rf(r, "y")?);
        let o = doc.camera(i).model.undistort_point(p, true, true, true);
        let l = format!("camera {i} ({}, {})", p.x, p.y);
        c.num(format_args!("{l} out_x"), o.x, rf(r, "out_x")?);
        c.num(format_args!("{l} out_y"), o.y, rf(r, "out_y")?);
        Ok(())
    });
    c.finish()
}

fn check_cube_points(ds: &Dataset, doc: &Document, rel: &str, all: bool) -> Item {
    let mut c = Check::new(rel, Tier::Read);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let cams = &doc.project().cameras;
    let n: usize = cams
        .iter()
        .flat_map(|c| c.calibration_images.iter())
        .map(|im| {
            if all {
                im.detected_all.len()
            } else {
                im.detected.len()
            }
        })
        .sum();
    c.rows(n, table.len());
    c.each_row(&table, |c, r| {
        let (i, k, p) = (ru(r, "camera")?, ru(r, "image")?, ru(r, "index")?);
        let im = get(&get(cams, i, "camera")?.calibration_images, k, "image")?;
        let l = format!("camera {i} image {k} point {p}");
        if all {
            let pt = get(&im.detected_all, p, "point")?;
            c.num(format_args!("{l} x"), pt.x, rf(r, "x")?);
            c.num(format_args!("{l} y"), pt.y, rf(r, "y")?);
        } else {
            let pt = get(&im.detected, p, "point")?;
            c.num(format_args!("{l} detected_x"), pt.x, rf(r, "detected_x")?);
            c.num(format_args!("{l} detected_y"), pt.y, rf(r, "detected_y")?);
            c.int(
                format_args!("{l} matched_id"),
                p as i64,
                ri(r, "matched_id")?,
            );
            let inl = get(&im.inlier, p, "inlier flag")?;
            c.int(format_args!("{l} inlier"), (*inl).into(), ri(r, "inlier")?);
        }
        Ok(())
    });
    c.finish()
}

// ---------------------------------------------------------------------------------------
// Trial tables
// ---------------------------------------------------------------------------------------

/// The "as held" tables (`writeTrialSnapshot`), present in the dump. `race`: the orders of
/// the C++ data race in the update that determined them.
fn snapshot(ds: &Dataset, doc: &Document, t: usize, race: &Race) -> Vec<Item> {
    let dir = format!("trial_{t}");
    let mut items = Vec::new();
    type F = fn(&Dataset, &Document, usize, &str, &Race) -> Item;
    let checks: [(&str, F); 12] = [
        ("projection_matrices.csv", |ds, doc, t, rel, _| {
            check_projection(ds, doc, t, rel)
        }),
        ("markers.csv", |ds, doc, t, rel, _| {
            check_markers(ds, doc, t, rel)
        }),
        ("points2d.csv", |ds, doc, t, rel, _| {
            check_points2d(ds, doc, t, rel)
        }),
        ("points2d_undistorted.csv", |ds, doc, t, rel, _| {
            check_undistorted(ds, doc, t, rel)
        }),
        ("reprojection.csv", |ds, doc, t, rel, _| {
            check_reprojection(ds, doc, t, rel)
        }),
        ("points3d_filtered.csv", |ds, doc, t, rel, _| {
            check_points3d_filtered(ds, doc, t, rel)
        }),
        ("rigid_bodies.csv", |ds, doc, t, rel, _| {
            check_rigid_bodies(ds, doc, t, rel)
        }),
        ("rb_points.csv", |ds, doc, t, rel, _| {
            check_rb_points(ds, doc, t, rel)
        }),
        ("rb_dummy_points.csv", |ds, doc, t, rel, _| {
            check_rb_dummy_points(ds, doc, t, rel)
        }),
        ("rb_pose_filtered.csv", |ds, doc, t, rel, race| {
            check_rb_poses(ds, doc, t, rel, true, race)
        }),
        ("rb_errors.csv", |ds, doc, t, rel, race| {
            check_rb_errors(ds, doc, t, rel, false, race)
        }),
        ("rb_errors_filtered.csv", |ds, doc, t, rel, race| {
            check_rb_errors(ds, doc, t, rel, true, race)
        }),
    ];
    for (file, f) in checks {
        let rel = format!("{dir}/{file}");
        if ds.has(&rel) {
            items.push(f(ds, doc, t, &rel, race));
        }
    }
    let rel = format!("{dir}/events.csv");
    if ds.has(&rel) {
        items.push(check_events(ds, doc, t, &rel));
    }
    // Format 2, written right after the tables above (`oracle/cpp/DEVIATIONS.md`, item 27).
    type G = fn(&Dataset, &Document, usize, &str) -> Item;
    let format2: [(&str, G); 7] = [
        ("marker_details.csv", check_marker_details),
        ("marker_interpolation.csv", check_marker_interpolation),
        ("rb_details.csv", check_rb_details),
        ("rb_reference_names.csv", check_rb_reference_names),
        ("rb_dummy_details.csv", check_rb_dummy_details),
        ("rb_dummy_coords.csv", check_rb_dummy_coords),
        ("event_details.csv", check_event_details),
    ];
    for (file, f) in format2 {
        let rel = format!("{dir}/{file}");
        if ds.has(&rel) {
            items.push(f(ds, doc, t, &rel));
        }
    }
    items
}

fn check_projection(ds: &Dataset, doc: &Document, t: usize, rel: &str) -> Item {
    let mut c = Check::new(rel, Tier::A);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let p = doc.projection_matrices(t);
    c.rows(12 * p.iter().flatten().count(), table.len());
    c.each_row(&table, |c, r| {
        let (i, row, col) = (ru(r, "camera")?, ru(r, "row")?, ru(r, "col")?);
        if row > 2 || col > 3 {
            return Err(format!("({row}, {col}) is outside a 3 x 4 matrix"));
        }
        let m = get(&p, i, "camera")?
            .as_ref()
            .ok_or_else(|| format!("camera {i} has no reference calibration here"))?;
        c.num(
            format_args!("camera {i} ({row},{col})"),
            m[(row, col)],
            rf(r, "value")?,
        );
        Ok(())
    });
    c.finish()
}

fn check_markers(ds: &Dataset, doc: &Document, t: usize, rel: &str) -> Item {
    let without_element: Vec<usize> = doc
        .extras()
        .trial_sources
        .get(t)
        .map_or_else(Vec::new, |s| s.markers_without_element.clone());
    // Ledger A52: the C++ never runs `Marker::load`, the only caller of `updateMeanSize`
    // while loading, for a marker without a `<Marker>` element, and the constructor does not
    // set `sizeRange` (core/Marker.cpp:44-58).
    let uninitialised: Vec<(usize, &str, &str)> = without_element
        .into_iter()
        .map(|i| {
            (
                i,
                RULE_A52,
                "ledger A52: the marker has no <Marker> element, so Marker::load never sets it",
            )
        })
        .collect();
    check_markers_with(ds, doc, t, rel, &uninitialised)
}

/// `markers.csv`, with `size_range` not compared for the markers listed in
/// `uninitialised` (marker, rule, why), whose C++ value is uninitialised memory.
fn check_markers_with(
    ds: &Dataset,
    doc: &Document,
    t: usize,
    rel: &str,
    uninitialised: &[(usize, &str, &str)],
) -> Item {
    check_markers_tol(ds, doc, t, rel, uninitialised, None)
}

/// [`check_markers_with`], with `mean_size` and `size_range` compared within the named
/// tolerance `sizes` when given (values made from refined radii, ledger A37).
fn check_markers_tol(
    ds: &Dataset,
    doc: &Document,
    t: usize,
    rel: &str,
    uninitialised: &[(usize, &str, &str)],
    sizes: Option<NamedTolerance>,
) -> Item {
    let mut c = Check::new(rel, Tier::A);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let markers = &doc.trial(t).markers;
    c.rows(markers.len(), table.len());
    c.each_row(&table, |c, r| {
        let i = ru(r, "marker")?;
        let m = get(markers, i, "marker")?;
        c.text(
            format_args!("marker {i} description"),
            &dump_text(&m.description),
            rs(r, "description")?,
        );
        let ints: [(&str, i64); 5] = [
            ("detection_method", m.method.code().into()),
            ("tracking_penalty", m.max_penalty.into()),
            ("size_override", m.size_override.into()),
            ("threshold_offset", m.threshold_offset.into()),
            ("requires_recomputation", m.requires_recomputation.into()),
        ];
        for (col, v) in ints {
            c.int(format_args!("marker {i} {col}"), v, ri(r, col)?);
        }
        c.num_named(
            format_args!("marker {i} mean_size"),
            m.mean_size,
            rf(r, "mean_size")?,
            sizes,
        );
        let want = rf(r, "size_range")?;
        if let Some((_, rule, why)) = uninitialised.iter().find(|(k, _, _)| *k == i) {
            c.exclude(
                rule,
                1,
                format!(
                    "marker {i} size_range: not compared: the C++ value is uninitialised \
                     ({why}); here {}, dump {want:e}",
                    m.size_range
                ),
            );
        } else {
            c.num_named(
                format_args!("marker {i} size_range"),
                m.size_range,
                want,
                sizes,
            );
        }
        Ok(())
    });
    c.finish()
}

fn check_points2d(ds: &Dataset, doc: &Document, t: usize, rel: &str) -> Item {
    check_points2d_with(ds, doc, t, rel, &[], None)
}

/// `points2d.csv`, with the sizes of the markers in `uninitialised` (marker, rule, why) not
/// compared, and every size compared within the named tolerance `sizes` when given.
fn check_points2d_with(
    ds: &Dataset,
    doc: &Document,
    t: usize,
    rel: &str,
    uninitialised: &[(usize, &str, &str)],
    sizes: Option<NamedTolerance>,
) -> Item {
    let mut skipped: BTreeMap<usize, usize> = BTreeMap::new();
    let mut c = Check::new(rel, Tier::Read);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let markers = &doc.trial(t).markers;
    let n: usize = markers.iter().map(|m| m.n_cameras() * m.n_frames()).sum();
    c.rows(n, table.len());
    c.each_row(&table, |c, r| {
        let (m, cam, f) = (ru(r, "marker")?, ru(r, "camera")?, ru(r, "frame")?);
        let v = get(&get(markers, m, "marker")?.views, cam, "camera")?;
        let l = format!("marker {m} camera {cam} frame {f}");
        let st = get(&v.status, f, "frame")?;
        c.int(
            format_args!("{l} status"),
            st.code().into(),
            ri(r, "status")?,
        );
        let p = get(&v.points, f, "frame")?;
        c.num(format_args!("{l} x"), p.x, rf(r, "x")?);
        c.num(format_args!("{l} y"), p.y, rf(r, "y")?);
        let size = *get(&v.size, f, "frame")?;
        let want = rf(r, "size")?;
        if uninitialised.iter().any(|(k, _, _)| *k == m) {
            *skipped.entry(m).or_default() += 1;
        } else {
            c.num_named(format_args!("{l} size"), size, want, sizes);
        }
        Ok(())
    });
    for (m, n) in skipped {
        if let Some((_, rule, why)) = uninitialised.iter().find(|(k, _, _)| *k == m) {
            c.exclude(
                rule,
                n,
                format!(
                    "marker {m} size in every camera and frame ({n} cells): not compared: {why}"
                ),
            );
        }
    }
    c.finish()
}

fn check_undistorted(ds: &Dataset, doc: &Document, t: usize, rel: &str) -> Item {
    let mut c = Check::new(rel, Tier::AUndistortion);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let markers = &doc.trial(t).markers;
    let n_cams = doc.project().cameras.len();
    let n: usize = markers
        .iter()
        .flat_map(|m| m.views.iter().take(n_cams))
        .map(|v| v.status.iter().filter(|s| s.has_data()).count())
        .sum();
    c.rows(n, table.len());
    c.each_row(&table, |c, r| {
        let (m, cam, f) = (ru(r, "marker")?, ru(r, "camera")?, ru(r, "frame")?);
        let v = get(&get(markers, m, "marker")?.views, cam, "camera")?;
        if cam >= n_cams {
            return Err(format!("camera {cam} does not exist here"));
        }
        let p = *get(&v.points, f, "frame")?;
        let u = doc.camera(cam).model.undistort_point(p, true, true, true);
        let l = format!("marker {m} camera {cam} frame {f}");
        c.num(format_args!("{l} x"), u.x, rf(r, "x")?);
        c.num(format_args!("{l} y"), u.y, rf(r, "y")?);
        Ok(())
    });
    c.finish()
}

fn check_reprojection(ds: &Dataset, doc: &Document, t: usize, rel: &str) -> Item {
    let mut c = Check::new(rel, Tier::AUndistortion);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let markers = &doc.trial(t).markers;
    let n: usize = markers
        .iter()
        .map(|m| m.views.iter().map(|v| v.points.len()).sum::<usize>())
        .sum();
    c.rows(n, table.len());
    c.each_row(&table, |c, r| {
        let (m, cam, f) = (ru(r, "marker")?, ru(r, "camera")?, ru(r, "frame")?);
        let v = get(&get(markers, m, "marker")?.views, cam, "camera")?;
        let l = format!("marker {m} camera {cam} frame {f}");
        let p = get(&v.projected, f, "frame")?;
        c.num(format_args!("{l} projected_x"), p.x, rf(r, "projected_x")?);
        c.num(format_args!("{l} projected_y"), p.y, rf(r, "projected_y")?);
        c.num(
            format_args!("{l} error2d"),
            *get(&v.error, f, "frame")?,
            rf(r, "error2d")?,
        );
        Ok(())
    });
    c.finish()
}

/// `points3d_filtered.csv`: `Marker::filterMarker` at the trial's cutoff on the 3D points
/// as held; `nan` where the filtered status is 0 or below.
fn check_points3d_filtered(ds: &Dataset, doc: &Document, t: usize, rel: &str) -> Item {
    let mut c = Check::new(rel, Tier::A);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let tr = doc.trial(t);
    let correction = xma_io::export::cutoff_correction(doc.compat());
    let mut filtered = Vec::new();
    for (i, m) in tr.markers.iter().enumerate() {
        if m.status3d.len() < tr.n_frames || m.points3d.len() < tr.n_frames {
            c.problem(format!("marker {i} holds fewer frames than the trial"));
            return c.finish();
        }
        match filter_marker(
            m,
            tr.cutoff_frequency,
            tr.recording_speed,
            tr.n_frames,
            correction,
        ) {
            Some(f) => filtered.push(f),
            None => {
                c.problem(format!(
                    "Marker::filterMarker refuses marker {i} here, but the dump has the table"
                ));
                return c.finish();
            }
        }
    }
    let n: usize = filtered.iter().map(|(_, s)| s.len().min(tr.n_frames)).sum();
    c.rows(n, table.len());
    c.each_row(&table, |c, r| {
        let (m, f) = (ru(r, "marker")?, ru(r, "frame")?);
        let (pts, st) = get(&filtered, m, "marker")?;
        let p = get(pts, f, "frame")?;
        let got = if get(st, f, "frame")?.code() <= 0 {
            [f64::NAN; 3]
        } else {
            [p.x, p.y, p.z]
        };
        for (k, axis) in ["x", "y", "z"].iter().enumerate() {
            c.num(
                format_args!("marker {m} frame {f} {axis}"),
                got[k],
                rf(r, axis)?,
            );
        }
        Ok(())
    });
    c.finish()
}

fn check_rigid_bodies(ds: &Dataset, doc: &Document, t: usize, rel: &str) -> Item {
    let mut c = Check::new(rel, Tier::Read);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let bodies = &doc.trial(t).rigid_bodies;
    c.rows(bodies.len(), table.len());
    c.each_row(&table, |c, r| {
        let i = ru(r, "body")?;
        let b = get(bodies, i, "body")?;
        c.text(
            format_args!("body {i} description"),
            &dump_text(&b.description),
            rs(r, "description")?,
        );
        c.int(
            format_args!("body {i} n_points"),
            b.marker_indices.len() as i64,
            ri(r, "n_points")?,
        );
        c.int(
            format_args!("body {i} override_cutoff"),
            b.override_cutoff_frequency.into(),
            ri(r, "override_cutoff")?,
        );
        c.num(
            format_args!("body {i} cutoff_frequency"),
            b.cutoff_frequency,
            rf(r, "cutoff_frequency")?,
        );
        c.int(
            format_args!("body {i} is_reference_set"),
            b.references_set.into(),
            ri(r, "is_reference_set")?,
        );
        Ok(())
    });
    c.finish()
}

fn check_rb_points(ds: &Dataset, doc: &Document, t: usize, rel: &str) -> Item {
    let mut c = Check::new(rel, Tier::Read);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let bodies = &doc.trial(t).rigid_bodies;
    c.rows(
        bodies.iter().map(|b| b.marker_indices.len()).sum(),
        table.len(),
    );
    c.each_row(&table, |c, r| {
        let (b, i) = (ru(r, "body")?, ru(r, "index")?);
        let body = get(bodies, b, "body")?;
        let m = get(&body.marker_indices, i, "point")?;
        c.int(
            format_args!("body {b} point {i} marker"),
            *m as i64,
            ri(r, "marker")?,
        );
        for (k, axis) in ["x", "y", "z"].iter().enumerate() {
            let orig = body
                .reference_points_original
                .get(i)
                .map_or(f64::NAN, |p| p[k]);
            let work = body.reference_points.get(i).map_or(f64::NAN, |p| p[k]);
            c.num(
                format_args!("body {b} point {i} ref_{axis}"),
                orig,
                rf(r, &format!("ref_{axis}"))?,
            );
            c.num(
                format_args!("body {b} point {i} ref_opt_{axis}"),
                work,
                rf(r, &format!("ref_opt_{axis}"))?,
            );
        }
        Ok(())
    });
    c.finish()
}

fn check_rb_dummy_points(ds: &Dataset, doc: &Document, t: usize, rel: &str) -> Item {
    let mut c = Check::new(rel, Tier::Read);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let bodies = &doc.trial(t).rigid_bodies;
    c.rows(
        bodies.iter().map(|b| b.virtual_points.len()).sum(),
        table.len(),
    );
    c.each_row(&table, |c, r| {
        let (b, i) = (ru(r, "body")?, ru(r, "index")?);
        let vp = get(&get(bodies, b, "body")?.virtual_points, i, "virtual point")?;
        for (k, axis) in ["x", "y", "z"].iter().enumerate() {
            c.num(
                format_args!("body {b} virtual {i} {axis}"),
                vp.reference[k],
                rf(r, axis)?,
            );
        }
        if let VirtualPointSource::OtherBody { body, point } = &vp.source {
            c.int(
                format_args!("body {b} virtual {i} other_body"),
                *body as i64,
                ri(r, "other_body")?,
            );
            for (k, axis) in ["x2", "y2", "z2"].iter().enumerate() {
                c.num(
                    format_args!("body {b} virtual {i} {axis}"),
                    point[k],
                    rf(r, axis)?,
                );
            }
        }
        Ok(())
    });
    c.finish()
}

fn check_events(ds: &Dataset, doc: &Document, t: usize, rel: &str) -> Item {
    let mut c = Check::new(rel, Tier::Read);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let events = &doc.trial(t).events;
    c.rows(events.iter().map(|e| e.data.len()).sum(), table.len());
    c.each_row(&table, |c, r| {
        let (i, f) = (ru(r, "event")?, ru(r, "frame")?);
        let ev = get(events, i, "event")?;
        c.text(
            format_args!("event {i} name"),
            &dump_text(&ev.name),
            rs(r, "name")?,
        );
        c.int(
            format_args!("event {i} frame {f}"),
            (*get(&ev.data, f, "frame")?).into(),
            ri(r, "value")?,
        );
        Ok(())
    });
    c.finish()
}

fn check_points3d(ds: &Dataset, doc: &Document, t: usize, rel: &str) -> Item {
    let mut c = Check::new(rel, Tier::A);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let markers = &doc.trial(t).markers;
    c.rows(markers.iter().map(|m| m.status3d.len()).sum(), table.len());
    c.each_row(&table, |c, r| {
        let (m, f) = (ru(r, "marker")?, ru(r, "frame")?);
        let mk = get(markers, m, "marker")?;
        let st: MarkerStatus = *get(&mk.status3d, f, "frame")?;
        c.int(
            format_args!("marker {m} frame {f} status"),
            st.code().into(),
            ri(r, "status")?,
        );
        let p = get(&mk.points3d, f, "frame")?;
        c.num(format_args!("marker {m} frame {f} x"), p.x, rf(r, "x")?);
        c.num(format_args!("marker {m} frame {f} y"), p.y, rf(r, "y")?);
        c.num(format_args!("marker {m} frame {f} z"), p.z, rf(r, "z")?);
        let e3 = mk.error3d.get(f).copied().unwrap_or(f64::NAN);
        c.num(
            format_args!("marker {m} frame {f} error3d"),
            e3,
            rf(r, "error3d")?,
        );
        Ok(())
    });
    c.finish()
}

/// One component of a body's pose at a frame, raw or filtered (NaN when absent).
fn pose_value(doc: &Document, t: usize, b: usize, f: usize, filtered: bool, k: usize) -> f64 {
    doc.trial(t)
        .rigid_bodies
        .get(b)
        .and_then(|body| {
            if filtered { &body.filtered } else { &body.raw }
                .poses
                .get(f)
        })
        .map_or(f64::NAN, |p| {
            if k < 3 {
                p.rotation[k]
            } else {
                p.translation[k - 3]
            }
        })
}

/// A body's mean 3D (`k` 0) or 2D (`k` 1) error at a frame, raw or filtered (NaN when
/// absent).
fn error_value(doc: &Document, t: usize, b: usize, f: usize, filtered: bool, k: usize) -> f64 {
    doc.trial(t)
        .rigid_bodies
        .get(b)
        .and_then(|body| {
            if filtered { &body.filtered } else { &body.raw }
                .errors
                .get(f)
        })
        .map_or(f64::NAN, |e| if k == 0 { e.mean_3d } else { e.mean_2d })
}

const POSE_COLUMNS: [&str; 6] = ["rx", "ry", "rz", "tx", "ty", "tz"];
const ERROR_COLUMNS: [&str; 2] = ["error3d_mean", "error2d_mean"];

/// Marks the orders of the race that the dump shows exactly: a raced body's own value at
/// the frame shown, in `rel` (`rb_errors.csv`, or a pose table when `poses`), that differs
/// from this program's and equals the order's.
fn establish(ds: &Dataset, doc: &Document, t: usize, rel: &str, race: &Race, poses: bool) {
    if !race.possible() {
        return;
    }
    let Ok(table) = ds.try_table(rel) else {
        return;
    };
    let columns: &[&str] = if poses { &POSE_COLUMNS } else { &ERROR_COLUMNS };
    let value = |d: &Document, b: usize, k: usize| {
        if poses {
            pose_value(d, t, b, race::SHOWN_FRAME, false, k)
        } else {
            error_value(d, t, b, race::SHOWN_FRAME, false, k)
        }
    };
    let raceable = race::raceable(doc.trial(t));
    for row in table.rows() {
        let (Ok(b), Ok(f)) = (row.try_usize("body"), row.try_usize("frame")) else {
            continue;
        };
        if f != race::SHOWN_FRAME || !raceable.contains(&b) {
            continue;
        }
        for (k, col) in columns.iter().enumerate() {
            let Ok(want) = row.try_f64(col) else {
                continue;
            };
            if same(value(doc, b, k), want) {
                continue;
            }
            for w in race.worlds_mut().iter_mut() {
                if w.established.is_none()
                    && w.raced.contains(&b)
                    && same(value(&w.doc, b, k), want)
                {
                    w.established = Some(format!("raced cell: body {b} frame {f} {col} in {rel}"));
                }
            }
        }
    }
}

fn check_rb_poses(
    ds: &Dataset,
    doc: &Document,
    t: usize,
    rel: &str,
    filtered: bool,
    race: &Race,
) -> Item {
    let mut c = Check::new(rel, Tier::B);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let bodies = &doc.trial(t).rigid_bodies;
    c.rows(bodies.iter().map(|b| b.n_frames()).sum(), table.len());
    let flag = if filtered { "filtered" } else { "computed" };
    c.each_row(&table, |c, r| {
        let (b, f) = (ru(r, "body")?, ru(r, "frame")?);
        let body = get(bodies, b, "body")?;
        let track = if filtered { &body.filtered } else { &body.raw };
        let valid = *get(&track.valid, f, "frame")?;
        c.int(
            format_args!("body {b} frame {f} {flag}"),
            valid.into(),
            ri(r, flag)?,
        );
        get(&track.poses, f, "frame")?;
        for (k, col) in POSE_COLUMNS.iter().enumerate() {
            c.race_num(
                race,
                format_args!("body {b} frame {f} {col}"),
                pose_value(doc, t, b, f, filtered, k),
                rf(r, col)?,
                |w| Some(pose_value(&w.doc, t, b, f, filtered, k)),
            );
        }
        Ok(())
    });
    c.finish()
}

fn check_rb_errors(
    ds: &Dataset,
    doc: &Document,
    t: usize,
    rel: &str,
    filtered: bool,
    race: &Race,
) -> Item {
    let mut c = Check::new(rel, Tier::B);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let bodies = &doc.trial(t).rigid_bodies;
    c.rows(bodies.iter().map(|b| b.n_frames()).sum(), table.len());
    c.each_row(&table, |c, r| {
        let (b, f) = (ru(r, "body")?, ru(r, "frame")?);
        let body = get(bodies, b, "body")?;
        let track = if filtered { &body.filtered } else { &body.raw };
        get(&track.errors, f, "frame")?;
        for (k, col) in ERROR_COLUMNS.iter().enumerate() {
            c.race_num(
                race,
                format_args!("body {b} frame {f} {col}"),
                error_value(doc, t, b, f, filtered, k),
                rf(r, col)?,
                |w| Some(error_value(&w.doc, t, b, f, filtered, k)),
            );
        }
        Ok(())
    });
    c.finish()
}

/// Rule names, as the report's rows show them.
const RULE_A52: &str = "not compared: C++ value uninitialised, ledger A52";
const RULE_RACE: &str = "compared with the other order of a C++ data race";
const RULE_DOWNSTREAM: &str = "not compared: downstream of a raced pose";
const RULE_UNINITIALISED_SQUARE: &str =
    "not compared: C++ value uninitialised, DEVIATIONS.md item 29";
const RULE_ALPHA: &str = "not compared: not held by the model";

/// How the report names the updates whose race orders are replayed ([`race`]).
const RACE_SNAPSHOT: &str = "the trial update before the as-held tables";
const RACE_OPTIMIZED: &str = "the update before rb_pose_optimized.csv";

// ---------------------------------------------------------------------------------------
// Format 2: what loading determines
// ---------------------------------------------------------------------------------------

fn check_camera_details(ds: &Dataset, doc: &Document) -> Item {
    let mut c = Check::new("camera_details.csv", Tier::Read);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let cams = &doc.project().cameras;
    c.rows(cams.len(), table.len());
    c.each_row(&table, |c, r| {
        let i = ru(r, "camera")?;
        let cam = get(cams, i, "camera")?;
        c.int(format_args!("camera {i} id"), i as i64, ri(r, "id")?);
        c.int(
            format_args!("camera {i} portal_id"),
            cam.portal_id.into(),
            ri(r, "portal_id")?,
        );
        c.int(
            format_args!("camera {i} has_model_distortion"),
            cam.has_model_distortion.into(),
            ri(r, "has_model_distortion")?,
        );
        for (v, view) in ["calibration", "digitization", "detail"].iter().enumerate() {
            let a = &cam.image_adjustments[v];
            c.num(
                format_args!("camera {i} {view}_scale"),
                a.scale,
                rf(r, &format!("{view}_scale"))?,
            );
            c.num(
                format_args!("camera {i} {view}_bias"),
                a.bias,
                rf(r, &format!("{view}_bias"))?,
            );
        }
        Ok(())
    });
    c.finish()
}

fn check_trial_details(ds: &Dataset, doc: &Document) -> Item {
    let mut c = Check::new("trial_details.csv", Tier::Read);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let trials = &doc.project().trials;
    c.rows(trials.len(), table.len());
    c.each_row(&table, |c, r| {
        let i = ru(r, "trial")?;
        let t = get(trials, i, "trial")?;
        let ints: [(&str, i64); 4] = [
            ("is_copy_from_default", t.is_copy_from_default.into()),
            ("n_markers", t.markers.len() as i64),
            ("n_rigid_bodies", t.rigid_bodies.len() as i64),
            ("n_events", t.events.len() as i64),
        ];
        for (col, v) in ints {
            c.int(format_args!("trial {i} {col}"), v, ri(r, col)?);
        }
        Ok(())
    });
    c.finish()
}

/// `calibration_object.csv`. `planar` is derived on load from the bead positions (some axis
/// zero for every bead); `square_size` is compared only for a checkerboard (the C++ leaves
/// it uninitialised otherwise, `oracle/cpp/DEVIATIONS.md` item 29).
fn check_calibration_object(ds: &Dataset, doc: &Document) -> Item {
    use xma_model::CalibrationObject;
    let mut c = Check::new("calibration_object.csv", Tier::Read);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    c.rows(1, table.len());
    c.each_row(&table, |c, r| {
        match &doc.project().calibration_object {
            None => {
                c.int("initialised", 0, ri(r, "initialised")?);
            }
            Some(CalibrationObject::Checkerboard {
                horizontal_squares,
                vertical_squares,
                square_size,
                ..
            }) => {
                c.int("checkerboard", 1, ri(r, "checkerboard")?);
                c.int(
                    "horizontal_squares",
                    (*horizontal_squares).into(),
                    ri(r, "horizontal_squares")?,
                );
                c.int(
                    "vertical_squares",
                    (*vertical_squares).into(),
                    ri(r, "vertical_squares")?,
                );
                c.num("square_size", *square_size, rf(r, "square_size")?);
            }
            Some(CalibrationObject::Frame(f)) => {
                let planar = (0..3).any(|a| f.points.iter().all(|p| p[a] == 0.0));
                let ints: [(&str, i64); 7] = [
                    ("initialised", 1),
                    ("checkerboard", 0),
                    ("planar", planar.into()),
                    ("white_blobs", f.white_blobs.into()),
                    ("n_points", f.points.len() as i64),
                    ("n_names", f.point_names.len() as i64),
                    ("n_references", f.reference_ids.len() as i64),
                ];
                for (col, v) in ints {
                    c.int(col, v, ri(r, col)?);
                }
                c.text(
                    "frame_specifications_file",
                    &dump_text(&f.frame_specification_filename),
                    rs(r, "frame_specifications_file")?,
                );
                c.text(
                    "references_file",
                    &dump_text(&f.references_filename),
                    rs(r, "references_file")?,
                );
                c.exclude(
                    RULE_UNINITIALISED_SQUARE,
                    1,
                    "square_size not compared: the C++ leaves it uninitialised for a frame \
                     (DEVIATIONS.md item 29); the frame specification's scale factors have no \
                     accessor and are not in the table",
                );
            }
        }
        Ok(())
    });
    c.finish()
}

fn calibration_frame(doc: &Document) -> Option<&xma_model::CalibrationFrame> {
    match &doc.project().calibration_object {
        Some(xma_model::CalibrationObject::Frame(f)) => Some(f),
        _ => None,
    }
}

fn check_calibration_object_points(ds: &Dataset, doc: &Document) -> Item {
    let mut c = Check::new("calibration_object_points.csv", Tier::Read);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let frame = calibration_frame(doc);
    c.rows(
        frame.map_or(0, |f| f.points.len().max(f.point_names.len())),
        table.len(),
    );
    c.each_row(&table, |c, r| {
        let i = ru(r, "index")?;
        let f = frame.ok_or("no calibration frame here")?;
        let p = get(&f.points, i, "bead")?;
        for (a, axis) in ["x", "y", "z"].iter().enumerate() {
            c.num(format_args!("bead {i} {axis}"), p[a], rf(r, axis)?);
        }
        c.text(
            format_args!("bead {i} name"),
            &dump_text(f.point_names.get(i).map_or("", String::as_str)),
            rs(r, "name")?,
        );
        Ok(())
    });
    c.finish()
}

fn check_calibration_object_references(ds: &Dataset, doc: &Document) -> Item {
    let mut c = Check::new("calibration_object_references.csv", Tier::Read);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let frame = calibration_frame(doc);
    c.rows(frame.map_or(0, |f| f.reference_ids.len()), table.len());
    c.each_row(&table, |c, r| {
        let i = ru(r, "index")?;
        let f = frame.ok_or("no calibration frame here")?;
        c.int(
            format_args!("reference {i} point_id"),
            (*get(&f.reference_ids, i, "reference")?).into(),
            ri(r, "point_id")?,
        );
        c.text(
            format_args!("reference {i} name"),
            &dump_text(get(&f.reference_names, i, "reference name")?),
            rs(r, "name")?,
        );
        Ok(())
    });
    c.finish()
}

fn check_marker_details(ds: &Dataset, doc: &Document, t: usize, rel: &str) -> Item {
    let mut c = Check::new(rel, Tier::Read);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let markers = &doc.trial(t).markers;
    c.rows(markers.len(), table.len());
    c.each_row(&table, |c, r| {
        let i = ru(r, "marker")?;
        let m = get(markers, i, "marker")?;
        c.int(
            format_args!("marker {i} reference_set"),
            m.reference_3d.is_some().into(),
            ri(r, "reference_set")?,
        );
        let p = m.reference_3d.map_or([f64::NAN; 3], |p| [p.x, p.y, p.z]);
        for (a, axis) in ["x", "y", "z"].iter().enumerate() {
            c.num(
                format_args!("marker {i} ref_{axis}"),
                p[a],
                rf(r, &format!("ref_{axis}"))?,
            );
        }
        c.int(
            format_args!("marker {i} n_interpolation"),
            m.interpolation.len() as i64,
            ri(r, "n_interpolation")?,
        );
        Ok(())
    });
    c.finish()
}

fn check_marker_interpolation(ds: &Dataset, doc: &Document, t: usize, rel: &str) -> Item {
    let mut c = Check::new(rel, Tier::Read);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let markers = &doc.trial(t).markers;
    c.rows(
        markers.iter().map(|m| m.interpolation.len()).sum(),
        table.len(),
    );
    c.each_row(&table, |c, r| {
        let (i, f) = (ru(r, "marker")?, ru(r, "frame")?);
        let m = get(markers, i, "marker")?;
        c.int(
            format_args!("marker {i} frame {f} method"),
            get(&m.interpolation, f, "frame")?.code().into(),
            ri(r, "method")?,
        );
        Ok(())
    });
    c.finish()
}

fn check_rb_details(ds: &Dataset, doc: &Document, t: usize, rel: &str) -> Item {
    let mut c = Check::new(rel, Tier::Read);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let bodies = &doc.trial(t).rigid_bodies;
    c.rows(bodies.len(), table.len());
    c.each_row(&table, |c, r| {
        let i = ru(r, "body")?;
        let b = get(bodies, i, "body")?;
        // `RigidBody::setColor` keeps red, green and blue only: always valid, alpha 255.
        let ints: [(&str, i64); 12] = [
            ("visible", b.visible.into()),
            ("color_r", b.color.r.into()),
            ("color_g", b.color.g.into()),
            ("color_b", b.color.b.into()),
            ("color_a", 255),
            ("color_valid", 1),
            ("has_mesh", b.mesh_model.is_some().into()),
            ("draw_mesh", b.draw_mesh_model.into()),
            ("references_set", b.references_set.into()),
            ("has_optimized", b.has_optimized_coordinates.into()),
            ("n_reference_names", b.reference_names.len() as i64),
            ("n_dummy", b.virtual_points.len() as i64),
        ];
        for (col, v) in ints {
            c.int(format_args!("body {i} {col}"), v, ri(r, col)?);
        }
        c.text(
            format_args!("body {i} mesh_path"),
            &dump_text(b.mesh_model.as_deref().unwrap_or("")),
            rs(r, "mesh_path")?,
        );
        c.num(
            format_args!("body {i} mesh_scale"),
            b.mesh_scale,
            rf(r, "mesh_scale")?,
        );
        Ok(())
    });
    c.finish()
}

fn check_rb_reference_names(ds: &Dataset, doc: &Document, t: usize, rel: &str) -> Item {
    let mut c = Check::new(rel, Tier::Read);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let bodies = &doc.trial(t).rigid_bodies;
    c.rows(
        bodies.iter().map(|b| b.reference_names.len()).sum(),
        table.len(),
    );
    c.each_row(&table, |c, r| {
        let (b, i) = (ru(r, "body")?, ru(r, "index")?);
        let body = get(bodies, b, "body")?;
        c.text(
            format_args!("body {b} reference name {i}"),
            &dump_text(get(&body.reference_names, i, "reference name")?),
            rs(r, "name")?,
        );
        Ok(())
    });
    c.finish()
}

fn check_rb_dummy_details(ds: &Dataset, doc: &Document, t: usize, rel: &str) -> Item {
    let mut c = Check::new(rel, Tier::Read);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let bodies = &doc.trial(t).rigid_bodies;
    c.rows(
        bodies.iter().map(|b| b.virtual_points.len()).sum(),
        table.len(),
    );
    c.each_row(&table, |c, r| {
        let (b, i) = (ru(r, "body")?, ru(r, "index")?);
        let vp = get(&get(bodies, b, "body")?.virtual_points, i, "virtual point")?;
        c.text(
            format_args!("body {b} virtual {i} name"),
            &dump_text(&vp.name),
            rs(r, "name")?,
        );
        let other = match vp.source {
            VirtualPointSource::OtherBody { body, .. } => body as i64,
            VirtualPointSource::Coordinates(_) => -1,
        };
        c.int(
            format_args!("body {b} virtual {i} other_body"),
            other,
            ri(r, "other_body")?,
        );
        Ok(())
    });
    c.finish()
}

/// `rb_dummy_coords.csv`: every frame held of each virtual point given by coordinates;
/// `set` 0 with `nan` where a frame has none.
fn check_rb_dummy_coords(ds: &Dataset, doc: &Document, t: usize, rel: &str) -> Item {
    let mut c = Check::new(rel, Tier::Read);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let bodies = &doc.trial(t).rigid_bodies;
    let n: usize = bodies
        .iter()
        .flat_map(|b| b.virtual_points.iter())
        .map(|v| match &v.source {
            VirtualPointSource::Coordinates(coords) => coords.len(),
            VirtualPointSource::OtherBody { .. } => 0,
        })
        .sum();
    c.rows(n, table.len());
    c.each_row(&table, |c, r| {
        let (b, i, f) = (ru(r, "body")?, ru(r, "index")?, ru(r, "frame")?);
        let vp = get(&get(bodies, b, "body")?.virtual_points, i, "virtual point")?;
        let VirtualPointSource::Coordinates(coords) = &vp.source else {
            return Err(format!("body {b} virtual {i} is attached to a body here"));
        };
        let p = *get(coords, f, "frame")?;
        c.int(
            format_args!("body {b} virtual {i} frame {f} set"),
            p.is_some().into(),
            ri(r, "set")?,
        );
        let v = p.map_or([f64::NAN; 3], |p| [p.x, p.y, p.z]);
        for (a, axis) in ["x", "y", "z"].iter().enumerate() {
            c.num(
                format_args!("body {b} virtual {i} frame {f} {axis}"),
                v[a],
                rf(r, axis)?,
            );
        }
        Ok(())
    });
    c.finish()
}

/// `event_details.csv`. The model holds an event's colour as red, green and blue; the C++
/// also keeps its alpha (`#aarrggbb`, `transparent`), which only drawing reads, and an
/// invalid colour reads as black with a reader's warning. So `color_a` is not compared,
/// and `color_valid` is compared with whether the reader warned about the colour.
fn check_event_details(ds: &Dataset, doc: &Document, t: usize, rel: &str) -> Item {
    let mut c = Check::new(rel, Tier::Read);
    let Some(table) = c.table(ds) else {
        return c.finish();
    };
    let events = &doc.trial(t).events;
    let warnings = &doc.extras().warnings;
    c.rows(events.len(), table.len());
    let mut alpha_skipped = 0;
    c.each_row(&table, |c, r| {
        let i = ru(r, "event")?;
        let e = get(events, i, "event")?;
        c.text(
            format_args!("event {i} name"),
            &dump_text(&e.name),
            rs(r, "name")?,
        );
        let warned = warnings
            .iter()
            .any(|w| w.contains(&format!("event '{}': colour", e.name)));
        let ints: [(&str, i64); 6] = [
            ("color_r", e.color.r.into()),
            ("color_g", e.color.g.into()),
            ("color_b", e.color.b.into()),
            ("color_valid", (!warned).into()),
            ("draw", e.draw.into()),
            ("n_frames", e.data.len() as i64),
        ];
        for (col, v) in ints {
            c.int(format_args!("event {i} {col}"), v, ri(r, col)?);
        }
        alpha_skipped += 1;
        Ok(())
    });
    if alpha_skipped > 0 {
        c.exclude(
            RULE_ALPHA,
            alpha_skipped,
            format!(
                "color_a of {alpha_skipped} events not compared: the model holds an event's \
             colour without alpha, which only drawing reads"
            ),
        );
    }
    c.finish()
}

// ---------------------------------------------------------------------------------------
// What is not compared
// ---------------------------------------------------------------------------------------

/// Why a file of the dump is not compared, by its path relative to the dump.
fn reason(rel: &str) -> &'static str {
    let file = rel.rsplit('/').next().unwrap_or(rel);
    if rel == "manifest.json" {
        "read for the settings and versions above"
    } else if rel.starts_with("screens/") {
        "window captures: interface work (phase 2 and later)"
    } else if rel.contains("/frames/") || file == "frame_stats.csv" {
        "decoded video frames: phase 2"
    } else if file == "detection.csv" {
        "detection at sampled stored points: compared by the tests of xma-engine \
         (detection_frames.rs), not by xma parity"
    } else if file == "tracking.csv" {
        "forward tracker steps from sampled points: the same calls as the forward rows of \
         tracking_steps.csv, which xma parity compares in the every-frame dump \
         (dump_clean_step1)"
    } else if rel.starts_with("exports/export_all/") {
        "Export All's DLT and MayaCam version 1 files: not ported in phase 1 (its undistortion files are compared)"
    } else if rel.contains("/exports/xromm_vr/") {
        "XROMM VR export: deferred (ledger F12)"
    } else if file == "cube_refit.csv" || file == "bundle.csv" {
        "calibration re-fit and bundle adjustment: phase 4"
    } else {
        "not recognised by this version of xma parity"
    }
}

/// Every file of the dump not in `seen`, grouped by folder and reason where a folder holds
/// more than a few.
fn not_compared(ds: &Dataset, seen: &BTreeSet<String>) -> Vec<NotCompared> {
    let mut files = Vec::new();
    walk(&ds.dump, "", &mut files);
    let mut groups: BTreeMap<(String, &'static str), Vec<String>> = BTreeMap::new();
    for f in files.into_iter().filter(|f| !seen.contains(f)) {
        let dir = f.rsplit_once('/').map_or("", |(d, _)| d).to_owned();
        groups.entry((dir, reason(&f))).or_default().push(f);
    }
    let mut out = Vec::new();
    for ((dir, why), list) in groups {
        if list.len() > 3 {
            out.push(NotCompared {
                path: format!("{dir}/ ({} files)", list.len()),
                reason: why.to_owned(),
            });
        } else {
            out.extend(list.into_iter().map(|path| NotCompared {
                path,
                reason: why.to_owned(),
            }));
        }
    }
    out
}

fn walk(dir: &Path, prefix: &str, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = entries.filter_map(Result::ok).collect();
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for e in entries {
        let name = e.file_name().to_string_lossy().into_owned();
        let rel = if prefix.is_empty() {
            name
        } else {
            format!("{prefix}/{name}")
        };
        if e.path().is_dir() {
            walk(&e.path(), &rel, out);
        } else {
            out.push(rel);
        }
    }
}

// ---------------------------------------------------------------------------------------
// The command
// ---------------------------------------------------------------------------------------

pub fn run(args: &[String], io: &mut Io<'_>) -> Result<(), CliError> {
    let p = parse(args, &[Opt::value("project"), Opt::value("report")])?;
    if p.has("help") {
        io.line(HELP);
        return Ok(());
    }
    let arg = p.positional(0, "a dataset name or dump directory")?;
    p.at_most(1)?;
    if let Some(run) = edits::find_edit_run(arg) {
        if !run.is_present() {
            return Err(CliError::failed(format!(
                "edit run '{arg}' is not present: {} or {} is missing",
                run.start_project.display(),
                run.output.display()
            )));
        }
        let report = edits::run_edit_parity(&run).map_err(CliError::failed)?;
        let md = report.to_markdown();
        match p.value("report") {
            Some(path) => {
                std::fs::write(path, &md).map_err(|e| CliError::failed(format!("{path}: {e}")))?;
                io.line(report.summary());
                io.line(format_args!("report written to {path}"));
            }
            None => io.line(md),
        }
        return if report.passed() {
            Ok(())
        } else {
            Err(CliError {
                message: report.summary(),
                code: 3,
            })
        };
    }
    // The export and denoising runs (package "images", `parity/calibration/images.rs`).
    if let Some(run) = calibration::images::find_image_run(arg) {
        if !run.is_present() {
            return Err(CliError::failed(format!(
                "run '{arg}' is not present: {} or {} is missing",
                run.script.display(),
                run.output.display()
            )));
        }
        let report = calibration::images::run_images_parity(&run).map_err(CliError::failed)?;
        let md = report.to_markdown().replacen(
            "# Calibration parity report",
            "# Export and denoising parity report",
            1,
        );
        write_out(io, p.value("report"), &md, &report.summary())?;
        return if report.passed() {
            Ok(())
        } else {
            Err(CliError {
                message: report.summary(),
                code: 3,
            })
        };
    }
    if let Some(run) = portal::find_portal_run(arg) {
        let report = portal::run_portal_parity(&run).map_err(CliError::failed)?;
        let md = report.to_markdown().replacen(
            "# Calibration parity report",
            "# XMA Portal parity report",
            1,
        );
        write_out(io, p.value("report"), &md, &report.summary())?;
        return if report.passed() {
            Ok(())
        } else {
            Err(CliError {
                message: report.summary(),
                code: 3,
            })
        };
    }
    if let Some(run) = calibration::find_calibration_run(arg) {
        if !run.is_present() {
            return Err(CliError::failed(format!(
                "calibration run '{arg}' is not present: {} or {} is missing",
                run.script.display(),
                run.output.display()
            )));
        }
        let report = calibration::run_calibration_parity(&run).map_err(CliError::failed)?;
        write_out(
            io,
            p.value("report"),
            &report.to_markdown(),
            &report.summary(),
        )?;
        return if report.passed() {
            Ok(())
        } else {
            Err(CliError {
                message: report.summary(),
                code: 3,
            })
        };
    }
    if arg == "tier-c" {
        let report = tier_c::run_tier_c().map_err(CliError::failed)?;
        write_out(
            io,
            p.value("report"),
            &report.to_markdown(),
            &report.summary(),
        )?;
        return if report.errors.is_empty() {
            Ok(())
        } else {
            Err(CliError::failed(report.errors.join("; ")))
        };
    }
    if let Some(run) = tracking::find_tracking_run(arg) {
        // A run of a kind that is no longer compared (types 5 and 6, merge-aware tracking,
        // parity ledger A226) is reported as not compared, whether its files are there or not.
        if run.not_compared().is_none() && !run.is_present() {
            return Err(CliError::failed(format!(
                "tracking run '{arg}' is not present: {} or {} is missing",
                run.start_project.display(),
                run.run_dir.display()
            )));
        }
        let report = tracking::run_tracking_parity(&run).map_err(CliError::failed)?;
        return write_out(
            io,
            p.value("report"),
            &report.to_markdown(),
            &report.summary(),
        )
        .and(if report.passed() {
            Ok(())
        } else {
            Err(CliError {
                message: report.summary(),
                code: 3,
            })
        });
    }
    let ds = locate(arg, p.value("project").map(Path::new)).map_err(CliError::failed)?;
    let report = run_parity(&ds).map_err(CliError::failed)?;
    let md = report.to_markdown();
    match p.value("report") {
        Some(path) => {
            std::fs::write(path, &md).map_err(|e| CliError::failed(format!("{path}: {e}")))?;
            io.line(report.summary());
            io.line(format_args!("report written to {path}"));
        }
        None => io.line(md),
    }
    if report.passed() {
        Ok(())
    } else {
        Err(CliError {
            message: report.summary(),
            code: 3,
        })
    }
}

/// Writes a report to `path` (with the summary line on standard output), or the report to
/// standard output.
fn write_out(io: &mut Io<'_>, path: Option<&str>, md: &str, summary: &str) -> Result<(), CliError> {
    match path {
        Some(path) => {
            std::fs::write(path, md).map_err(|e| CliError::failed(format!("{path}: {e}")))?;
            io.line(summary);
            io.line(format_args!("report written to {path}"));
        }
        None => io.line(md),
    }
    Ok(())
}
