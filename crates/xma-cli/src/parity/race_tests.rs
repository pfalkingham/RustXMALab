//! The race rule (ledger A56) on a dump made to show the race where no real dump has yet:
//! the "as held" tables and the rigid-body exports of `rv_vpchain_b` rewritten as one order
//! of the race in the update before them would leave them (body 0 refined twice at frame 0,
//! bodies 1 and 3, which refer to it through virtual points, computed from it). The
//! rigid-body exports are left as dumped: their recomputation (ledger A36) recomputes body 0
//! before anything reads it, so in this project no order of the race reaches them, which the
//! test checks too. Every other file is a link to the real dump. Skipped
//! when the data is absent.

use std::path::{Path, PathBuf};

use xma_engine::{Document, Export, OpenOptions};
use xma_io::export::rigid_bodies::TransformationsOptions;
use xma_oracle::Dataset;

use super::race::{Race, World};
use super::{
    ERROR_COLUMNS, POSE_COLUMNS, RACE_SNAPSHOT, RULE_DOWNSTREAM, RULE_RACE, Verdict, error_value,
    pose_value, run_parity,
};

const T: usize = 0;

/// The snapshot race of `ds`, replayed as `run_parity` does for its first trial, and the
/// exports that change state.
fn snapshot_race(ds: &Dataset) -> (Race, Vec<(String, Export)>) {
    let manifest = ds.manifest();
    let options = OpenOptions {
        overrides: crate::settings::manifest_settings(&manifest).expect("settings"),
        compat: xma_model::CppCompat::XMALAB3,
        ..OpenOptions::default()
    };
    let loaded = xma_engine::read_project(&ds.project, &options).expect("reads");
    let mut doc = Document::from_loaded(loaded);
    doc.set_path(&ds.project);
    let race = Race::new(RACE_SNAPSHOT, &doc, T, |d| {
        d.update_after_load();
        d.select_trial(T);
    });
    let ex = format!("trial_{T}/exports");
    let exports = vec![
        (
            format!("{ex}/RigidBodyTransformations.csv"),
            Export::RigidBodies {
                bodies: None,
                options: TransformationsOptions {
                    filtered: false,
                    ..TransformationsOptions::default()
                },
            },
        ),
        (
            format!("{ex}/RigidBodyTransformations_filtered.csv"),
            Export::RigidBodies {
                bodies: None,
                options: TransformationsOptions::default(),
            },
        ),
    ];
    (race, exports)
}

/// Rewrites the value columns of a pose or error table with `world`'s values, in Rust's
/// shortest exact form.
fn rewrite(text: &str, world: &World, filtered: bool, poses: bool) -> String {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 {
            out.push(line.to_owned());
            continue;
        }
        let mut cells: Vec<String> = line.split(',').map(str::to_owned).collect();
        let (b, f): (usize, usize) = (
            cells[0].parse().expect("body"),
            cells[1].parse().expect("frame"),
        );
        let n = if poses {
            POSE_COLUMNS.len()
        } else {
            ERROR_COLUMNS.len()
        };
        for k in 0..n {
            let v = if poses {
                pose_value(&world.doc, T, b, f, filtered, k)
            } else {
                error_value(&world.doc, T, b, f, filtered, k)
            };
            // Pose tables: body, frame, flag, then the values; error tables: body, frame,
            // then the values (and more columns, left alone).
            let at = if poses { 3 + k } else { 2 + k };
            cells[at] = format!("{v}");
        }
        out.push(cells.join(","));
    }
    out.join("\n") + "\n"
}

fn link_tree(from: &Path, to: &Path, skip: &[&str], prefix: &str) {
    std::fs::create_dir_all(to).expect("folder");
    for e in std::fs::read_dir(from).expect("dump") {
        let e = e.expect("entry");
        let name = e.file_name().to_string_lossy().into_owned();
        let rel = if prefix.is_empty() {
            name.clone()
        } else {
            format!("{prefix}/{name}")
        };
        if e.path().is_dir() {
            link_tree(&e.path(), &to.join(&name), skip, &rel);
        } else if !skip.contains(&rel.as_str()) {
            std::os::unix::fs::symlink(e.path(), to.join(&name)).expect("link");
        }
    }
}

fn replace_field(text: &str, line: usize, field: usize, f: impl Fn(f64) -> f64) -> String {
    let mut lines: Vec<String> = text.lines().map(str::to_owned).collect();
    let mut cells: Vec<String> = lines[line].split(',').map(str::to_owned).collect();
    let v: f64 = cells[field].parse().expect("number");
    cells[field] = format!("{}", f(v));
    lines[line] = cells.join(",");
    lines.join("\n") + "\n"
}

/// The line of a pose or error table for body `b` at frame `f` (0 is the header).
fn line_of(text: &str, b: usize, f: usize) -> usize {
    text.lines()
        .position(|l| l.starts_with(&format!("{b},{f},")))
        .expect("row")
}

#[test]
fn a_dump_in_one_order_of_the_race_is_explained_exactly() {
    let Some(ds) = xma_oracle::dataset("rv_vpchain_b") else {
        return;
    };
    let (race, exports) = snapshot_race(&ds);
    let worlds = race.worlds();
    let world = worlds
        .iter()
        .find(|w| w.raced == [0] && w.consumers == [1, 3])
        .expect("the order: body 0 raced, bodies 1 and 3 computed from it");

    let dir: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "..",
        "..",
        "target",
        "tmp",
        "race_dump",
    ]
    .iter()
    .collect();
    let _ = std::fs::remove_dir_all(&dir);
    let tables = [
        ("trial_0/rb_errors.csv", false, false),
        ("trial_0/rb_pose_filtered.csv", true, true),
        ("trial_0/rb_errors_filtered.csv", true, false),
    ];
    let skip: Vec<&str> = tables.iter().map(|t| t.0).collect();
    link_tree(&ds.dump, &dir, &skip, "");
    for (rel, filtered, poses) in tables {
        let text = std::fs::read_to_string(ds.path(rel)).expect("table");
        let mut new = rewrite(&text, world, filtered, poses);
        if rel == "trial_0/rb_pose_filtered.csv" {
            // A cell downstream of the raced pose that the order does not reproduce (body 0,
            // frame 5, rx, in the filtered run that starts at frame 0), and one the race
            // cannot reach (body 2, frame 100, rx), each moved off.
            let at = line_of(&new, 0, 5);
            new = replace_field(&new, at, 3, |v| v + 1e-13);
            let at = line_of(&new, 2, 100);
            new = replace_field(&new, at, 3, |v| v + 1e-3);
        }
        assert_ne!(new, text, "{rel}: the order changes the table");
        std::fs::write(dir.join(rel), new).expect("write");
    }
    drop(worlds);
    race.replay_exports(T, &exports);
    for w in race.worlds().iter() {
        for (rel, _) in &exports {
            assert_eq!(
                w.exports.get(rel).expect("replayed").as_bytes(),
                std::fs::read(ds.path(rel)).expect("export").as_slice(),
                "{rel}: {}",
                w.describe()
            );
        }
    }

    let fake = Dataset::from_paths("rv_vpchain_b_race_order", &ds.project, &dir);
    let report = run_parity(&fake).expect("runs");
    let item = |n: &str| report.items.iter().find(|i| i.name == n).expect(n);

    let errors = item("trial_0/rb_errors.csv");
    assert_eq!(errors.verdict, Verdict::IdenticalExplained, "{errors:?}");
    assert_eq!(errors.excluded, 0);
    assert!(errors.explained >= 1, "{}", errors.explained);
    assert!(
        errors.notes[0].contains("body 0 refined twice at frame 0"),
        "{:?}",
        errors.notes
    );

    // The filtered poses: the run around frame 0 explained; the unreproduced cell not
    // compared, under a rule naming the raced cell; the unreachable one different.
    let filtered = item("trial_0/rb_pose_filtered.csv");
    assert_eq!(filtered.verdict, Verdict::Different, "{filtered:?}");
    assert_eq!(filtered.excluded, 1);
    assert!(filtered.explained > 100, "{}", filtered.explained);
    assert_eq!(filtered.problems.len(), 1, "{:?}", filtered.problems);
    assert!(
        filtered.problems[0].contains("body 2 frame 100 rx"),
        "{:?}",
        filtered.problems
    );
    let downstream = filtered
        .rules
        .iter()
        .find(|(r, _)| r.starts_with(RULE_DOWNSTREAM))
        .expect("downstream rule");
    assert_eq!(downstream.1, 1);
    assert!(
        downstream
            .0
            .contains("raced cell: body 0 frame 0 error3d_mean in trial_0/rb_errors.csv"),
        "{}",
        downstream.0
    );
    assert!(filtered.rules.iter().any(|(r, _)| r == RULE_RACE));

    let i = item("trial_0/rb_errors_filtered.csv");
    assert_eq!(i.verdict, Verdict::IdenticalExplained, "{i:?}");
    assert_eq!(i.excluded, 0);
    assert!(i.explained > 0);
    for (rel, _) in &exports {
        assert_eq!(item(rel).verdict, Verdict::Identical, "{rel}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
