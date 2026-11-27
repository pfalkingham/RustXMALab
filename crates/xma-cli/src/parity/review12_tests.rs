//! Review of 20 November 2026, second round: the comparison functions of `parity.rs` on
//! small hand-made inputs that used to pass and must not (findings 1 and 10 of
//! `parity-xtask.md`).

use std::collections::BTreeMap;
use std::path::PathBuf;

use xma_engine::Document;
use xma_model::{CppCompat, Project};
use xma_oracle::Dataset;

use super::race::{Race, World};
use super::{Check, Tier, Verdict, export_file_impl};

/// A scratch dump folder holding `files`.
fn dump(tag: &str, files: &[(&str, &str)]) -> (Dataset, PathBuf) {
    let dir = std::env::temp_dir().join(format!(
        "xma-review12-{tag}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    for (rel, text) in files {
        let p = dir.join(rel);
        std::fs::create_dir_all(p.parent().expect("a parent")).expect("scratch folder");
        std::fs::write(&p, text).expect("scratch file");
    }
    (Dataset::from_paths(tag, &dir, &dir), dir)
}

fn export(tag: &str, got: &str, want: &str, tol: bool) -> Verdict {
    let (ds, dir) = dump(tag, &[("f.csv", want)]);
    let item = export_file_impl(
        &ds,
        "f.csv",
        got,
        Tier::Read,
        &Race::none(),
        tol.then_some(super::TOLERANCE_A37_CUBE),
    );
    let _ = std::fs::remove_dir_all(dir);
    item.verdict
}

#[test]
fn review12_export_tolerated_value_and_carriage_return_differ() {
    // One value within the named tolerance, and a carriage return: not within tolerance.
    assert_eq!(
        export("cr", "1.0000000001,5\r\n", "1,5\n", true),
        Verdict::Different
    );
    // The value alone is still within the tolerance.
    assert_eq!(
        export("tol", "1.0000000001,5\n", "1,5\n", true),
        Verdict::WithinTolerance
    );
}

#[test]
fn review12_export_text_fields_with_other_spaces_differ() {
    assert_eq!(export("sp", "a, b\n", "a,b\n", false), Verdict::Different);
    assert_eq!(export("sp2", "a,b \n", "a,b\n", false), Verdict::Different);
    assert_eq!(
        export("same", "a, b\n", "a, b\n", false),
        Verdict::Identical
    );
}

/// A race with the given worlds already built.
fn race_with(worlds: Vec<World>) -> Race {
    let doc = Document::from_project(Project::default(), CppCompat::XMALAB3);
    Race::with_worlds(doc, worlds)
}

fn world(exports: &[(&str, &str)]) -> World {
    World {
        raced: vec![0],
        consumers: Vec::new(),
        doc: Document::from_project(Project::default(), CppCompat::XMALAB3),
        exports: exports
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect::<BTreeMap<_, _>>(),
        export_errors: BTreeMap::new(),
        established: None,
    }
}

#[test]
fn review12_race_world_without_the_value_is_not_a_match() {
    // The dump has nan (no pose), this program writes a number, and the world's export
    // lacks the line: before, the missing value was NaN and so "the other order".
    let (ds, dir) = dump("race", &[("f.csv", "1,nan\n")]);
    let race = race_with(vec![world(&[("f.csv", "1\n")])]);
    let item = export_file_impl(&ds, "f.csv", "1,2\n", Tier::Read, &race, None);
    assert_eq!(item.verdict, Verdict::Different, "{item:?}");
    // A world that holds the dump's value still explains the cell.
    let race = race_with(vec![world(&[("f.csv", "1,3\n")])]);
    let (ds2, dir2) = dump("race2", &[("f.csv", "1,3\n")]);
    let item = export_file_impl(&ds2, "f.csv", "1,2\n", Tier::Read, &race, None);
    assert_ne!(item.verdict, Verdict::Different, "{item:?}");
    assert_eq!(item.explained, 1, "{item:?}");
    let _ = std::fs::remove_dir_all(dir);
    let _ = std::fs::remove_dir_all(dir2);
}

#[test]
fn review12_race_failed_export_in_a_world_is_a_problem() {
    let (ds, dir) = dump("racefail", &[("f.csv", "1,3\n")]);
    let mut w = world(&[]);
    w.export_errors
        .insert("f.csv".to_owned(), "no rigid body".to_owned());
    let race = race_with(vec![w, world(&[("f.csv", "1,3\n")])]);
    let item = export_file_impl(&ds, "f.csv", "1,2\n", Tier::Read, &race, None);
    assert_eq!(item.verdict, Verdict::Different, "{item:?}");
    assert!(
        item.problems.iter().any(|p| p.contains("no rigid body")),
        "{item:?}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn review12_race_num_skips_a_world_with_no_value() {
    let race = race_with(vec![world(&[])]);
    let mut c = Check::new("t", Tier::Read);
    c.race_num(&race, "cell", 2.0, f64::NAN, |_| None);
    assert_eq!(c.finish().verdict, Verdict::Different);
}
