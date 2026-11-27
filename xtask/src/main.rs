//! Project automation, run as `cargo xtask <command>`.
//!
//! Commands:
//! - `check [--test-jobs N]`: format check, clippy, tests, then `depcheck`; stops at the first
//!   failing step. The test step (`tests_runner.rs`) builds every test program, runs several at
//!   the same time (`--test-jobs N`, else `XTASK_TEST_JOBS`, else a third of the logical cores),
//!   longest first by the previous run's times (`target/xtask-test-times.json`), runs them all
//!   even if one fails, then the doc tests, and ends with a summary.
//! - `test-step [--test-jobs N] [--package P]...` (internal, not in the usage text): the test
//!   step alone, optionally for some packages only; used to check the runner on part of the
//!   workspace.
//! - `depcheck [--metadata FILE]`: dependency policy check over `cargo metadata`.
//! - `dist`: the Linux release archive (`dist.rs`).
//! - `licenses [--out FILE]`: `THIRD-PARTY-LICENSES.txt` for the crates of the two programs
//!   (`licenses.rs`).
//!
//! Written in Rust so it runs the same on Linux, Windows and macOS.

mod depcheck;
mod dist;
mod licenses;
mod tests_runner;

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const USAGE: &str = "Usage: cargo xtask <command>

Commands:
  check [--test-jobs N]     cargo fmt --check, clippy -D warnings, tests, then depcheck.
                            The test programs are built first, then run N at a time
                            (default: XTASK_TEST_JOBS, else a third of the logical cores;
                            --test-jobs 1 runs them one after another), longest first by
                            the times of the previous run (target/xtask-test-times.json);
                            all run even if one fails, then the doc tests, then a summary
                            naming each failed program and the command that reruns it
  depcheck [--metadata F]   check the dependency graph against the pure-Rust policy;
                            with --metadata, read `cargo metadata` JSON from file F
  dist                      build rustxmalab and xma in release mode and write
                            target/dist/rustxmalab-<version>-<arch>-linux.tar.gz
  licenses [--out FILE]     write THIRD-PARTY-LICENSES.txt (the licences of every crate the
                            two programs are built from) to FILE, by default
                            target/dist/THIRD-PARTY-LICENSES.txt";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = workspace_root();
    let result = match args.first().map(String::as_str) {
        Some("check") => tests_runner::parse_test_args(&args[1..], false)
            .map_err(|e| format!("{e}\n\n{USAGE}"))
            .and_then(|opts| check(&root, &opts)),
        Some("test-step") => tests_runner::parse_test_args(&args[1..], true)
            .and_then(|opts| tests_runner::test_step(&root, &opts)),
        Some("depcheck") => match &args[1..] {
            [] => run_depcheck(&root, None),
            [flag, file] if flag == "--metadata" => run_depcheck(&root, Some(Path::new(file))),
            _ => Err(format!("bad arguments to depcheck\n\n{USAGE}")),
        },
        Some("dist") if args.len() == 1 => dist::dist(&root),
        Some("licenses") => match &args[1..] {
            [] => dist::target_dir(&root)
                .and_then(|t| licenses::write(&root, &t.join("dist/THIRD-PARTY-LICENSES.txt"))),
            [flag, file] if flag == "--out" => licenses::write(&root, Path::new(file)),
            _ => Err(format!("bad arguments to licenses\n\n{USAGE}")),
        },
        Some("-h" | "--help" | "help") => {
            println!("{USAGE}");
            Ok(())
        }
        Some(other) => Err(format!("unknown command '{other}'\n\n{USAGE}")),
        None => Err(USAGE.to_string()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("xtask: {msg}");
            ExitCode::FAILURE
        }
    }
}

/// The workspace to check: the one the command is run in (the nearest directory at or above
/// the current one whose `Cargo.toml` declares a workspace), so that a separate copy of the
/// workspace checks itself even if its xtask binary was built in
/// another copy. Falls back to the workspace this binary was built in.
fn workspace_root() -> PathBuf {
    if let Ok(mut dir) = std::env::current_dir() {
        loop {
            let manifest = dir.join("Cargo.toml");
            if std::fs::read_to_string(&manifest).is_ok_and(|text| text.contains("[workspace]")) {
                return dir;
            }
            if !dir.pop() {
                break;
            }
        }
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives one level below the workspace root")
        .to_path_buf()
}

pub(crate) fn cargo() -> Command {
    Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
}

fn check(root: &Path, test_options: &tests_runner::TestOptions) -> Result<(), String> {
    let steps: [(&str, &[&str]); 2] = [
        ("fmt", &["fmt", "--all", "--", "--check"]),
        (
            "clippy",
            &[
                "clippy",
                "--workspace",
                "--all-targets",
                "--",
                "-D",
                "warnings",
            ],
        ),
    ];
    for (name, args) in steps {
        eprintln!("==> [{name}] cargo {}", args.join(" "));
        let status = cargo()
            .args(args)
            .current_dir(root)
            .status()
            .map_err(|e| format!("step '{name}' could not start cargo: {e}"))?;
        if !status.success() {
            return Err(format!("check FAILED at step '{name}' ({status})"));
        }
    }
    tests_runner::test_step(root, test_options)?;
    eprintln!("==> [depcheck]");
    run_depcheck(root, None).map_err(|e| format!("{e}\ncheck FAILED at step 'depcheck'"))?;
    eprintln!("==> all checks passed");
    Ok(())
}

/// The platforms the program is built for (PLAN.md, decision D8). The dependency
/// check covers the packages that are compiled for at least one of them, so a
/// crate used only on Android or on the web, which is never built, needs no entry.
const SHIPPING_TARGETS: &[&str] = &[
    "x86_64-unknown-linux-gnu",
    "x86_64-pc-windows-msvc",
    "aarch64-apple-darwin",
];

fn cargo_metadata(root: &Path, target: &str) -> Result<String, String> {
    let out = cargo()
        .args([
            "metadata",
            "--format-version",
            "1",
            "--filter-platform",
            target,
        ])
        .current_dir(root)
        .output()
        .map_err(|e| format!("could not run cargo metadata: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "cargo metadata failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    String::from_utf8(out.stdout).map_err(|e| format!("cargo metadata output: {e}"))
}

fn run_depcheck(root: &Path, metadata_file: Option<&Path>) -> Result<(), String> {
    let allowlist_path = root.join("deps-allowlist.toml");
    let allowlist_text = std::fs::read_to_string(&allowlist_path)
        .map_err(|e| format!("cannot read {}: {e}", allowlist_path.display()))?;
    let allowlist = depcheck::parse_allowlist(&allowlist_text)?;

    // One graph per shipping platform, or the single file given.
    let graphs: Vec<(String, String)> = match metadata_file {
        Some(path) => vec![(
            path.display().to_string(),
            std::fs::read_to_string(path)
                .map_err(|e| format!("cannot read {}: {e}", path.display()))?,
        )],
        None => SHIPPING_TARGETS
            .iter()
            .map(|t| cargo_metadata(root, t).map(|m| ((*t).to_string(), m)))
            .collect::<Result<_, _>>()?,
    };

    let mut violations: Vec<String> = Vec::new();
    let mut packages_checked = 0;
    for (_, metadata) in &graphs {
        let report = depcheck::check(metadata, &allowlist)?;
        packages_checked = packages_checked.max(report.packages_checked);
        for v in &report.violations {
            let line = v.to_string();
            if !violations.contains(&line) {
                violations.push(line);
            }
        }
    }
    violations.sort();

    if violations.is_empty() {
        eprintln!(
            "depcheck: OK (up to {} packages per platform over {} platform(s), {} allowlisted)",
            packages_checked,
            graphs.len(),
            allowlist.len()
        );
        Ok(())
    } else {
        let mut msg = format!("depcheck FAILED: {} violation(s)\n", violations.len());
        for v in &violations {
            msg.push_str(&format!("  - {v}\n"));
        }
        msg.push_str(
            "Native code is not allowed: remove the dependency, or ask the user to approve \
             an entry in deps-allowlist.toml. Architecture rules (PLAN.md section 4) have no \
             allowlist.",
        );
        Err(msg)
    }
}
