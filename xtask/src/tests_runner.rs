//! The test step of `cargo xtask check`: build every test program once, then run several of them
//! at the same time (the author's decision 19, 21 November 2026, "Run programs side by side").
//!
//! 1. `cargo test --no-run --message-format=json-render-diagnostics` builds every test program
//!    and names each executable in a `compiler-artifact` message (`profile.test == true`).
//! 2. The programs are run [`resolve_test_jobs`] at a time, longest first by the times of the
//!    previous run (`<target>/xtask-test-times.json`, advisory: a missing or damaged file is
//!    ignored). Each runs as `cargo test` runs it: working directory the package root, no
//!    arguments, and the environment cargo gives a test program (see [`program_env`]).
//!    `RUST_TEST_THREADS` is left alone, so each program still runs its own tests on as many
//!    threads as the harness chooses (the machine's available parallelism): a program with one
//!    long test and many short ones then finishes as fast as it does alone, and the operating
//!    system shares the cores among the programs.
//! 3. Each program's output (stdout and stderr through one pipe, in the order written) is printed
//!    whole when the program ends, under cargo's own `Running ...` line, so the log reads as
//!    before; programs never interleave.
//! 4. Every program runs even if one fails. A summary names each failed program with the cargo
//!    command that reruns just it, the totals of tests, the wall-clock time, the sum of the
//!    programs' own times and the ten slowest programs. A program fails if it cannot be started,
//!    exits unsuccessfully (a signal included), prints no `test result:` line, or reports a
//!    failed test.
//! 5. The doc tests then run as `cargo test --doc`.

use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::ffi::{OsStr, OsString};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

/// Environment variable that sets how many test programs run at the same time.
pub(crate) const TEST_JOBS_ENV: &str = "XTASK_TEST_JOBS";

/// File under the target directory holding each program's time in the previous run.
pub(crate) const TIMES_FILE: &str = "xtask-test-times.json";

/// How long to keep reading a program's output after it has exited, in case a process it
/// started still holds the pipe open.
const OUTPUT_GRACE: Duration = Duration::from_secs(5);

/// The default number of test programs run at the same time, from the machine's logical
/// cores: a third of them, at least one. (To be tuned from measurements of the whole suite.)
pub(crate) fn default_test_jobs(logical_cores: usize) -> usize {
    (logical_cores / 3).max(1)
}

/// Options of the test step.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct TestOptions {
    /// `--test-jobs N`.
    pub test_jobs: Option<usize>,
    /// `--package P` (only on the internal `test-step` command): limit the step to packages.
    pub packages: Vec<String>,
}

fn parse_jobs(text: &str, source: &str) -> Result<usize, String> {
    match text.trim().parse::<usize>() {
        Ok(n) if n >= 1 => Ok(n),
        _ => Err(format!(
            "{source} must be a whole number of at least 1, not '{text}'"
        )),
    }
}

/// Parses the arguments after `check` (or after the internal `test-step`, which also takes
/// `--package P`, repeatable).
pub(crate) fn parse_test_args(
    args: &[String],
    allow_packages: bool,
) -> Result<TestOptions, String> {
    let mut opts = TestOptions::default();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        if let Some(value) = arg.strip_prefix("--test-jobs=") {
            opts.test_jobs = Some(parse_jobs(value, "--test-jobs")?);
        } else if arg == "--test-jobs" {
            let value = it.next().ok_or("--test-jobs needs a number")?;
            opts.test_jobs = Some(parse_jobs(value, "--test-jobs")?);
        } else if allow_packages && (arg == "--package" || arg == "-p") {
            let value = it.next().ok_or("--package needs a package name")?;
            opts.packages.push(value.clone());
        } else {
            return Err(format!("unexpected argument '{arg}'"));
        }
    }
    Ok(opts)
}

/// How many programs run at the same time: `--test-jobs`, else `XTASK_TEST_JOBS` (if set and
/// not empty), else [`default_test_jobs`].
pub(crate) fn resolve_test_jobs(
    cli: Option<usize>,
    env: Option<&OsStr>,
    logical_cores: usize,
) -> Result<usize, String> {
    if let Some(n) = cli {
        return Ok(n);
    }
    match env {
        Some(v) if !v.is_empty() => parse_jobs(&v.to_string_lossy(), TEST_JOBS_ENV),
        _ => Ok(default_test_jobs(logical_cores)),
    }
}

/// One test program found in cargo's build messages.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TestProgram {
    pub package_id: String,
    pub package: String,
    pub manifest_path: PathBuf,
    /// `lib`, `bin`, `test`, `bench` or `example`.
    pub kind: String,
    pub name: String,
    pub src_path: PathBuf,
    pub executable: PathBuf,
}

impl TestProgram {
    fn package_root(&self) -> &Path {
        self.manifest_path.parent().unwrap_or(Path::new("."))
    }

    /// The key under which the program's time is kept between runs (no build hash in it).
    pub fn key(&self) -> String {
        format!("{} {} {}", self.package, self.kind, self.name)
    }

    /// The source file, relative to the package root, as cargo prints it.
    fn src_display(&self) -> String {
        self.src_path
            .strip_prefix(self.package_root())
            .unwrap_or(&self.src_path)
            .display()
            .to_string()
    }

    /// Package and source, for the summary: `xma-engine tests/review10_merge_r06.rs`.
    pub fn short_label(&self) -> String {
        format!("{} {}", self.package, self.src_display())
    }

    /// Cargo's own line for the program, for example
    /// `Running tests/foo.rs (target/debug/.../foo-0123)` or `Running unittests src/lib.rs (...)`.
    pub fn running_line(&self, root: &Path) -> String {
        let exe = self
            .executable
            .strip_prefix(root)
            .unwrap_or(&self.executable);
        let prefix = if self.kind == "test" || self.kind == "bench" {
            ""
        } else {
            "unittests "
        };
        format!(
            "     Running {prefix}{} ({})",
            self.src_display(),
            exe.display()
        )
    }

    /// The command that reruns just this program through cargo.
    pub fn rerun_command(&self) -> String {
        let select = match self.kind.as_str() {
            "lib" => "--lib".to_string(),
            other => format!("--{other} {}", self.name),
        };
        format!("cargo test -p {} {select}", self.package)
    }
}

/// What the build messages say: the test programs, each package's build-script `OUT_DIR`, and
/// each package's ordinary binaries (for `CARGO_BIN_EXE_<name>`).
#[derive(Debug, Default)]
pub(crate) struct BuildPlan {
    pub programs: Vec<TestProgram>,
    pub out_dirs: HashMap<String, PathBuf>,
    pub bin_exes: HashMap<String, Vec<(String, PathBuf)>>,
}

/// The package name in a package id: `path+file:///a/xma-engine#0.1.0` gives `xma-engine`,
/// `path+file:///a/dir#name@0.1.0` and `registry+https://...#name@1.0.0` give `name`.
pub(crate) fn package_name_from_id(id: &str) -> String {
    let (path, frag) = id.rsplit_once('#').unwrap_or((id, ""));
    if let Some((name, _)) = frag.split_once('@') {
        return name.to_string();
    }
    path.trim_end_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or(path)
        .to_string()
}

/// Reads cargo's JSON messages (one per line; other lines are ignored).
pub(crate) fn parse_build_messages(text: &str) -> BuildPlan {
    let mut plan = BuildPlan::default();
    for line in text.lines() {
        let Ok(msg) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let package_id = msg["package_id"].as_str().unwrap_or_default().to_string();
        match msg["reason"].as_str() {
            Some("build-script-executed") => {
                if let Some(dir) = msg["out_dir"].as_str() {
                    plan.out_dirs.insert(package_id, PathBuf::from(dir));
                }
            }
            Some("compiler-artifact") => {
                let Some(exe) = msg["executable"].as_str() else {
                    continue;
                };
                let target = &msg["target"];
                let kind = match target["kind"][0].as_str() {
                    Some("bin") => "bin",
                    Some("test") => "test",
                    Some("bench") => "bench",
                    Some("example") => "example",
                    Some(_) => "lib",
                    None => continue,
                };
                let name = target["name"].as_str().unwrap_or_default().to_string();
                if msg["profile"]["test"].as_bool() != Some(true) {
                    if kind == "bin" {
                        plan.bin_exes
                            .entry(package_id)
                            .or_default()
                            .push((name, PathBuf::from(exe)));
                    }
                    continue;
                }
                plan.programs.push(TestProgram {
                    package: package_name_from_id(&package_id),
                    package_id,
                    manifest_path: PathBuf::from(msg["manifest_path"].as_str().unwrap_or_default()),
                    kind: kind.to_string(),
                    name,
                    src_path: PathBuf::from(target["src_path"].as_str().unwrap_or_default()),
                    executable: PathBuf::from(exe),
                });
            }
            _ => {}
        }
    }
    plan
}

/// The fields of a package that cargo passes to its test programs as `CARGO_PKG_*`.
#[derive(Debug, Default, Clone)]
pub(crate) struct PackageInfo {
    pub name: String,
    pub version: String,
    pub authors: Vec<String>,
    pub description: String,
    pub homepage: String,
    pub repository: String,
    pub license: String,
    pub license_file: String,
    pub readme: String,
    pub rust_version: String,
}

/// Reads `cargo metadata --no-deps` output: the target directory and the packages by id.
pub(crate) fn parse_metadata(
    text: &str,
) -> Result<(PathBuf, HashMap<String, PackageInfo>), String> {
    let meta: Value =
        serde_json::from_str(text).map_err(|e| format!("cargo metadata output: {e}"))?;
    let target = meta["target_directory"]
        .as_str()
        .ok_or("cargo metadata gives no target_directory")?;
    let s = |v: &Value| v.as_str().unwrap_or_default().to_string();
    let mut packages = HashMap::new();
    for p in meta["packages"].as_array().into_iter().flatten() {
        packages.insert(
            s(&p["id"]),
            PackageInfo {
                name: s(&p["name"]),
                version: s(&p["version"]),
                authors: p["authors"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|a| a.as_str().map(str::to_string))
                    .collect(),
                description: s(&p["description"]),
                homepage: s(&p["homepage"]),
                repository: s(&p["repository"]),
                license: s(&p["license"]),
                license_file: s(&p["license_file"]),
                readme: s(&p["readme"]),
                rust_version: s(&p["rust_version"]),
            },
        );
    }
    Ok((PathBuf::from(target), packages))
}

/// Changes to the inherited environment for one program: first remove, then set.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct EnvChange {
    pub remove: Vec<OsString>,
    pub set: Vec<(OsString, OsString)>,
}

/// Variables cargo sets for the program being built or run; inherited copies of them (for
/// example those `cargo run` gave the xtask binary itself) are removed before ours are set.
fn is_cargo_per_target_var(key: &str) -> bool {
    key.starts_with("CARGO_PKG_")
        || key.starts_with("CARGO_BIN_EXE_")
        || matches!(
            key,
            "CARGO_MANIFEST_DIR"
                | "CARGO_MANIFEST_PATH"
                | "CARGO_MANIFEST_LINKS"
                | "CARGO_BIN_NAME"
                | "CARGO_CRATE_NAME"
                | "CARGO_PRIMARY_PACKAGE"
                | "CARGO_TARGET_TMPDIR"
                | "CARGO_RUSTC_CURRENT_DIR"
                | "OUT_DIR"
        )
}

/// The name of the variable through which the platform finds shared libraries, as cargo uses it.
pub(crate) fn dylib_path_var() -> &'static str {
    if cfg!(windows) {
        "PATH"
    } else if cfg!(target_os = "macos") {
        "DYLD_FALLBACK_LIBRARY_PATH"
    } else {
        "LD_LIBRARY_PATH"
    }
}

/// The value cargo gives [`dylib_path_var`]: its own directories, then the inherited value (on
/// macOS, when unset, cargo's defaults `$HOME/lib:/usr/local/lib:/usr/lib`).
pub(crate) fn dylib_path_value(dirs: &[PathBuf], inherited: Option<&OsStr>) -> Option<OsString> {
    let mut all: Vec<PathBuf> = dirs.to_vec();
    match inherited {
        Some(v) if !v.is_empty() => all.extend(std::env::split_paths(v)),
        _ if cfg!(target_os = "macos") => {
            if let Some(home) = std::env::var_os("HOME") {
                all.push(PathBuf::from(home).join("lib"));
            }
            all.push("/usr/local/lib".into());
            all.push("/usr/lib".into());
        }
        _ => {}
    }
    std::env::join_paths(all).ok()
}

/// The environment `cargo test` gives a test program at run time (established by dumping the
/// environment of a test run by cargo 1.100): `CARGO`, `CARGO_MANIFEST_DIR`,
/// `CARGO_MANIFEST_PATH`, the `CARGO_PKG_*` fields, `OUT_DIR` if the package has a build script,
/// `CARGO_BIN_EXE_<name>` for each binary of the package (integration tests and benches only),
/// and the shared-library search path; everything else is inherited from the caller.
pub(crate) fn program_env(
    prog: &TestProgram,
    info: Option<&PackageInfo>,
    out_dir: Option<&Path>,
    bin_exes: &[(String, PathBuf)],
    cargo: &OsStr,
    dylib_value: Option<&OsStr>,
    inherited_keys: impl IntoIterator<Item = OsString>,
) -> EnvChange {
    let mut change = EnvChange {
        remove: inherited_keys
            .into_iter()
            .filter(|k| k.to_str().is_some_and(is_cargo_per_target_var))
            .collect(),
        set: Vec::new(),
    };
    let mut set = |k: &str, v: OsString| change.set.push((k.into(), v));
    set("CARGO", cargo.to_os_string());
    set(
        "CARGO_MANIFEST_DIR",
        prog.package_root().as_os_str().to_os_string(),
    );
    set(
        "CARGO_MANIFEST_PATH",
        prog.manifest_path.as_os_str().to_os_string(),
    );
    let fallback = PackageInfo {
        name: prog.package.clone(),
        ..PackageInfo::default()
    };
    let info = info.unwrap_or(&fallback);
    let version = info.version.as_str();
    let (core, pre) = match version
        .split_once('+')
        .map_or(version, |(v, _)| v)
        .split_once('-')
    {
        Some((c, p)) => (c, p),
        None => (version.split_once('+').map_or(version, |(v, _)| v), ""),
    };
    let mut parts = core.splitn(3, '.');
    let fields = [
        ("CARGO_PKG_NAME", info.name.clone()),
        ("CARGO_PKG_VERSION", info.version.clone()),
        (
            "CARGO_PKG_VERSION_MAJOR",
            parts.next().unwrap_or_default().into(),
        ),
        (
            "CARGO_PKG_VERSION_MINOR",
            parts.next().unwrap_or_default().into(),
        ),
        (
            "CARGO_PKG_VERSION_PATCH",
            parts.next().unwrap_or_default().into(),
        ),
        ("CARGO_PKG_VERSION_PRE", pre.into()),
        ("CARGO_PKG_AUTHORS", info.authors.join(":")),
        ("CARGO_PKG_DESCRIPTION", info.description.clone()),
        ("CARGO_PKG_HOMEPAGE", info.homepage.clone()),
        ("CARGO_PKG_REPOSITORY", info.repository.clone()),
        ("CARGO_PKG_LICENSE", info.license.clone()),
        ("CARGO_PKG_LICENSE_FILE", info.license_file.clone()),
        ("CARGO_PKG_README", info.readme.clone()),
        ("CARGO_PKG_RUST_VERSION", info.rust_version.clone()),
    ];
    for (k, v) in fields {
        set(k, v.into());
    }
    if let Some(dir) = out_dir {
        set("OUT_DIR", dir.as_os_str().to_os_string());
    }
    if prog.kind == "test" || prog.kind == "bench" {
        for (name, exe) in bin_exes {
            set(
                &format!("CARGO_BIN_EXE_{name}"),
                exe.as_os_str().to_os_string(),
            );
        }
    }
    if let Some(v) = dylib_value {
        set(dylib_path_var(), v.to_os_string());
    }
    change
}

/// Reads the previous run's times; anything wrong with the file gives an empty map.
pub(crate) fn parse_times(text: &str) -> HashMap<String, f64> {
    let Ok(v) = serde_json::from_str::<Value>(text) else {
        return HashMap::new();
    };
    v["seconds"]
        .as_object()
        .into_iter()
        .flatten()
        .filter_map(|(k, t)| {
            t.as_f64()
                .filter(|t| t.is_finite() && *t >= 0.0)
                .map(|t| (k.clone(), t))
        })
        .collect()
}

/// Writes the times file's content.
pub(crate) fn render_times(times: &HashMap<String, f64>) -> String {
    let mut keys: Vec<&String> = times.keys().collect();
    keys.sort();
    let mut seconds = serde_json::Map::new();
    for k in keys {
        let t = (times[k] * 1000.0).round() / 1000.0;
        seconds.insert(k.clone(), Value::from(t));
    }
    let doc = serde_json::json!({
        "about": "Each test program's time in the last run of `cargo xtask check`; only used to start the longest first.",
        "seconds": seconds,
    });
    serde_json::to_string_pretty(&doc).unwrap_or_default() + "\n"
}

/// Puts the programs in the order they are started: those with no previous time first (in
/// cargo's order), then the rest from the longest previous time to the shortest.
pub(crate) fn order_by_previous_times(programs: &mut [TestProgram], times: &HashMap<String, f64>) {
    programs.sort_by(|a, b| {
        let (ta, tb) = (times.get(&a.key()), times.get(&b.key()));
        match (ta, tb) {
            (None, None) => std::cmp::Ordering::Equal,
            (None, Some(_)) => std::cmp::Ordering::Less,
            (Some(_), None) => std::cmp::Ordering::Greater,
            (Some(x), Some(y)) => y.total_cmp(x),
        }
    });
}

/// Test counts from a program's `test result:` line.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub(crate) struct Counts {
    pub passed: u64,
    pub failed: u64,
    pub ignored: u64,
}

/// Reads the last `test result: ok. 5 passed; 0 failed; 2 ignored; ...` line of the output.
pub(crate) fn parse_result_line(output: &str) -> Option<Counts> {
    let line = output
        .lines()
        .rev()
        .find_map(|l| l.trim_start().strip_prefix("test result:"))?;
    let (mut passed, mut failed, mut ignored) = (None, None, None);
    for part in line.split(';') {
        let words: Vec<&str> = part.split_whitespace().collect();
        let [.., number, label] = words.as_slice() else {
            continue;
        };
        let Ok(n) = number.parse::<u64>() else {
            continue;
        };
        match *label {
            "passed" => passed = Some(n),
            "failed" => failed = Some(n),
            "ignored" => ignored = Some(n),
            _ => {}
        }
    }
    Some(Counts {
        passed: passed?,
        failed: failed?,
        ignored: ignored.unwrap_or(0),
    })
}

/// How a program ended.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Outcome {
    /// It ran; `success` is the exit status's verdict, `status` its description.
    Exited { success: bool, status: String },
    /// It could not be started or waited for.
    StartFailed(String),
}

/// One program's result.
#[derive(Debug, Clone)]
pub(crate) struct ProgramResult {
    pub key: String,
    pub label: String,
    pub rerun: String,
    pub seconds: f64,
    pub outcome: Outcome,
    pub counts: Option<Counts>,
}

impl ProgramResult {
    /// Why the program counts as failed, or `None` if it passed.
    pub fn failure(&self) -> Option<String> {
        let mut reasons = Vec::new();
        match &self.outcome {
            Outcome::StartFailed(e) => return Some(format!("could not be started: {e}")),
            Outcome::Exited {
                success: false,
                status,
            } => reasons.push(format!("ended with {status}")),
            Outcome::Exited { .. } => {}
        }
        match self.counts {
            None => reasons.push("printed no 'test result:' line".to_string()),
            Some(c) if c.failed > 0 => reasons.push(format!("{} test(s) failed", c.failed)),
            Some(_) => {}
        }
        (!reasons.is_empty()).then(|| reasons.join("; "))
    }
}

/// The end-of-step summary.
#[derive(Debug)]
pub(crate) struct Summary {
    pub programs: usize,
    pub totals: Counts,
    /// (program, why it failed, rerun command).
    pub failures: Vec<(String, String, String)>,
    pub wall_seconds: f64,
    pub sum_seconds: f64,
    pub slowest: Vec<(String, f64)>,
}

impl Summary {
    pub fn ok(&self) -> bool {
        self.failures.is_empty()
    }

    pub fn render(&self) -> String {
        let t = &self.totals;
        let mut s = format!(
            "==> test summary: {} program(s), {} failed\n  tests: {} passed, {} failed, {} ignored\n  \
             wall-clock time of the test programs: {:.1} s; sum of their own times: {:.1} s\n",
            self.programs,
            self.failures.len(),
            t.passed,
            t.failed,
            t.ignored,
            self.wall_seconds,
            self.sum_seconds
        );
        if !self.slowest.is_empty() {
            s.push_str(&format!("  the {} slowest programs:\n", self.slowest.len()));
            for (label, secs) in &self.slowest {
                s.push_str(&format!("    {secs:8.1} s  {label}\n"));
            }
        }
        if !self.failures.is_empty() {
            s.push_str("  FAILED programs:\n");
            for (label, why, rerun) in &self.failures {
                s.push_str(&format!("    {label}: {why}\n      rerun: {rerun}\n"));
            }
        }
        s
    }
}

pub(crate) fn summarize(results: &[ProgramResult], wall: Duration) -> Summary {
    let mut totals = Counts::default();
    let mut failures = Vec::new();
    for r in results {
        if let Some(c) = r.counts {
            totals.passed += c.passed;
            totals.failed += c.failed;
            totals.ignored += c.ignored;
        }
        if let Some(why) = r.failure() {
            failures.push((r.label.clone(), why, r.rerun.clone()));
        }
    }
    let mut by_time: Vec<(String, f64)> = results
        .iter()
        .map(|r| (r.label.clone(), r.seconds))
        .collect();
    by_time.sort_by(|a, b| b.1.total_cmp(&a.1));
    by_time.truncate(10);
    Summary {
        programs: results.len(),
        totals,
        failures,
        wall_seconds: wall.as_secs_f64(),
        sum_seconds: results.iter().map(|r| r.seconds).sum(),
        slowest: by_time,
    }
}

/// A program ready to run.
struct Prepared {
    program: TestProgram,
    env: EnvChange,
}

/// Runs one program with its stdout and stderr in one pipe; returns its result and output.
fn run_one(p: &Prepared) -> (Outcome, String, f64) {
    let start = Instant::now();
    let pipe = std::io::pipe().and_then(|(r, w)| {
        let w2 = w.try_clone()?;
        Ok((r, w, w2))
    });
    let (mut reader, w1, w2) = match pipe {
        Ok(x) => x,
        Err(e) => {
            return (
                Outcome::StartFailed(format!("no pipe: {e}")),
                String::new(),
                0.0,
            );
        }
    };
    let mut cmd = Command::new(&p.program.executable);
    cmd.current_dir(p.program.package_root())
        .stdin(Stdio::null())
        .stdout(w1)
        .stderr(w2);
    for k in &p.env.remove {
        cmd.env_remove(k);
    }
    for (k, v) in &p.env.set {
        cmd.env(k, v);
    }
    let spawned = cmd.spawn();
    drop(cmd); // closes this process's copies of the pipe's writing end
    let mut child = match spawned {
        Ok(c) => c,
        Err(e) => {
            return (
                Outcome::StartFailed(e.to_string()),
                String::new(),
                start.elapsed().as_secs_f64(),
            );
        }
    };
    let buf = Arc::new(Mutex::new(Vec::<u8>::new()));
    let (done_tx, done_rx) = mpsc::channel::<()>();
    {
        let buf = Arc::clone(&buf);
        std::thread::spawn(move || {
            let mut chunk = [0u8; 65536];
            loop {
                match reader.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => buf
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .extend_from_slice(&chunk[..n]),
                }
            }
            let _ = done_tx.send(());
        });
    }
    let status = child.wait();
    let seconds = start.elapsed().as_secs_f64();
    let held_open = done_rx.recv_timeout(OUTPUT_GRACE).is_err();
    let mut output =
        String::from_utf8_lossy(&buf.lock().unwrap_or_else(|e| e.into_inner())).into_owned();
    if held_open {
        output.push_str(
            "\n[xtask: stopped reading this program's output: a process it started still holds it open]\n",
        );
    }
    let outcome = match status {
        Ok(s) => Outcome::Exited {
            success: s.success(),
            status: s.to_string(),
        },
        Err(e) => Outcome::StartFailed(format!("could not wait for it: {e}")),
    };
    (outcome, output, seconds)
}

/// Runs the programs `jobs` at a time in the given order, printing each one's output whole as
/// it ends.
fn run_all(programs: Vec<Prepared>, jobs: usize, root: &Path) -> Vec<ProgramResult> {
    let total = programs.len();
    let queue = Mutex::new(programs.into_iter().collect::<VecDeque<_>>());
    let results = Mutex::new(Vec::with_capacity(total));
    std::thread::scope(|scope| {
        for _ in 0..jobs.min(total).max(1) {
            scope.spawn(|| {
                loop {
                    let next = queue.lock().unwrap_or_else(|e| e.into_inner()).pop_front();
                    let Some(p) = next else { break };
                    let (outcome, output, seconds) = run_one(&p);
                    let result = ProgramResult {
                        key: p.program.key(),
                        label: p.program.short_label(),
                        rerun: p.program.rerun_command(),
                        seconds,
                        counts: parse_result_line(&output),
                        outcome,
                    };
                    let mut results = results.lock().unwrap_or_else(|e| e.into_inner());
                    results.push(result.clone());
                    let mut block = format!(
                        "{}  [{}, {:.1} s, {}/{}]\n",
                        p.program.running_line(root),
                        p.program.package,
                        seconds,
                        results.len(),
                        total
                    );
                    block.push_str(&output);
                    if !output.ends_with('\n') {
                        block.push('\n');
                    }
                    if let Some(why) = result.failure() {
                        block.push_str(&format!(
                            "xtask: FAILED: {} {why}\n  rerun: {}\n",
                            result.label, result.rerun
                        ));
                    }
                    let mut out = std::io::stdout().lock();
                    let _ = out.write_all(block.as_bytes());
                    let _ = out.flush();
                }
            });
        }
    });
    results.into_inner().unwrap_or_else(|e| e.into_inner())
}

fn package_args(packages: &[String]) -> Vec<String> {
    if packages.is_empty() {
        vec!["--workspace".to_string()]
    } else {
        packages
            .iter()
            .flat_map(|p| ["-p".to_string(), p.clone()])
            .collect()
    }
}

/// The directory of the shared libraries of the Rust toolchain (`rustc --print target-libdir`),
/// which cargo puts on the library search path; `None` if rustc cannot say.
fn rustc_target_libdir() -> Option<PathBuf> {
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let out = Command::new(rustc)
        .args(["--print", "target-libdir"])
        .output()
        .ok()?;
    let text = String::from_utf8(out.stdout).ok()?;
    let dir = text.trim();
    (out.status.success() && !dir.is_empty()).then(|| PathBuf::from(dir))
}

/// The test step: build, run the programs side by side, summarise, then the doc tests.
pub(crate) fn test_step(root: &Path, opts: &TestOptions) -> Result<(), String> {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let jobs = resolve_test_jobs(
        opts.test_jobs,
        std::env::var_os(TEST_JOBS_ENV).as_deref(),
        cores,
    )?;

    let meta = crate::cargo()
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .current_dir(root)
        .output()
        .map_err(|e| format!("could not run cargo metadata: {e}"))?;
    if !meta.status.success() {
        return Err(format!(
            "cargo metadata failed:\n{}",
            String::from_utf8_lossy(&meta.stderr)
        ));
    }
    let (target_dir, packages) = parse_metadata(&String::from_utf8_lossy(&meta.stdout))?;

    let pkg = package_args(&opts.packages);
    let build_args: Vec<String> = ["test"]
        .iter()
        .map(|s| s.to_string())
        .chain(pkg.iter().cloned())
        .chain(["--no-run", "--message-format=json-render-diagnostics"].map(String::from))
        .collect();
    eprintln!("==> [test] cargo {}", build_args.join(" "));
    let build_start = Instant::now();
    let build = crate::cargo()
        .args(&build_args)
        .current_dir(root)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .output()
        .map_err(|e| format!("step 'test' could not start cargo: {e}"))?;
    if !build.status.success() {
        return Err(format!(
            "check FAILED at step 'test': the test programs did not build ({})",
            build.status
        ));
    }
    let build_seconds = build_start.elapsed().as_secs_f64();
    let plan = parse_build_messages(&String::from_utf8_lossy(&build.stdout));

    let cargo_path = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut dylib_dirs = vec![target_dir.join("debug")];
    dylib_dirs.extend(rustc_target_libdir());
    let dylib = dylib_path_value(&dylib_dirs, std::env::var_os(dylib_path_var()).as_deref());
    let inherited: Vec<OsString> = std::env::vars_os().map(|(k, _)| k).collect();

    let times_path = target_dir.join(TIMES_FILE);
    let mut times = std::fs::read_to_string(&times_path)
        .map(|t| parse_times(&t))
        .unwrap_or_default();
    let mut programs = plan.programs.clone();
    order_by_previous_times(&mut programs, &times);
    let prepared: Vec<Prepared> = programs
        .into_iter()
        .map(|program| {
            let env = program_env(
                &program,
                packages.get(&program.package_id),
                plan.out_dirs.get(&program.package_id).map(PathBuf::as_path),
                plan.bin_exes
                    .get(&program.package_id)
                    .map_or(&[][..], Vec::as_slice),
                &cargo_path,
                dylib.as_deref(),
                inherited.iter().cloned(),
            );
            Prepared { program, env }
        })
        .collect();

    eprintln!(
        "==> [test] built in {build_seconds:.1} s; running {} test programs, {jobs} at a time \
         (--test-jobs N or {TEST_JOBS_ENV} to change; {cores} logical cores)",
        prepared.len()
    );
    let run_start = Instant::now();
    let results = run_all(prepared, jobs, root);
    let summary = summarize(&results, run_start.elapsed());

    for r in &results {
        times.insert(r.key.clone(), r.seconds);
    }
    if let Err(e) = std::fs::write(&times_path, render_times(&times)) {
        eprintln!("xtask: could not write {}: {e}", times_path.display());
    }

    let doc_args: Vec<String> = ["test".to_string()]
        .into_iter()
        .chain(pkg)
        .chain(["--doc".to_string()])
        .collect();
    eprintln!("==> [test] cargo {}", doc_args.join(" "));
    let doc_ok = crate::cargo()
        .args(&doc_args)
        .current_dir(root)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    eprint!("{}", summary.render());
    if !doc_ok {
        eprintln!("  FAILED: the doc tests (cargo {})", doc_args.join(" "));
    }
    if summary.ok() && doc_ok {
        Ok(())
    } else {
        Err("check FAILED at step 'test'".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lines captured from `cargo test -p xma-model -p xma-cli ... --no-run
    /// --message-format=json-render-diagnostics` in this workspace (cargo 1.100), with the
    /// workspace path replaced by `/w`. The workspace has no build script, so the
    /// `build-script-executed` line is a real one (from `serde_core`) given xma-cli's package id.
    const FIXTURE: &str = include_str!("tests_runner_fixture.jsonl");

    fn args(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn build_messages_give_the_test_programs() {
        let plan = parse_build_messages(FIXTURE);
        let keys: Vec<String> = plan.programs.iter().map(TestProgram::key).collect();
        assert_eq!(
            keys,
            [
                "xma-model lib xma_model",
                "xma-cli lib xma_cli",
                "xma-cli bin xma",
                "xma-cli test cli",
            ]
        );
        let cli = &plan.programs[3];
        assert_eq!(cli.package, "xma-cli");
        assert!(cli.executable.ends_with("out/cli-1abed762533731dd"));
        assert_eq!(cli.package_root(), Path::new("/w/crates/xma-cli"));
        assert_eq!(cli.rerun_command(), "cargo test -p xma-cli --test cli");
        assert_eq!(
            plan.programs[0].rerun_command(),
            "cargo test -p xma-model --lib"
        );
        assert_eq!(
            plan.programs[2].rerun_command(),
            "cargo test -p xma-cli --bin xma"
        );
        assert_eq!(
            cli.running_line(Path::new("/w")),
            "     Running tests/cli.rs (target/debug/build/xma-cli/1abed762533731dd/out/cli-1abed762533731dd)"
        );
        assert!(
            plan.programs[0]
                .running_line(Path::new("/w"))
                .starts_with("     Running unittests src/lib.rs (target/")
        );
        // The ordinary `xma` binary (not built as a test) is kept for CARGO_BIN_EXE_xma.
        let bins = &plan.bin_exes[&cli.package_id];
        assert_eq!(bins.len(), 1);
        assert_eq!(bins[0].0, "xma");
        assert_eq!(bins[0].1, Path::new("/w/target/debug/xma"));
        assert_eq!(
            plan.out_dirs[&cli.package_id],
            Path::new("/w/target/debug/build/xma-cli/0a1b2c3d4e5f6071/out")
        );
    }

    #[test]
    fn package_names_from_ids() {
        assert_eq!(
            package_name_from_id("path+file:///w/crates/xma-engine#0.1.0"),
            "xma-engine"
        );
        assert_eq!(
            package_name_from_id("path+file:///w/dir#other@0.1.0"),
            "other"
        );
        assert_eq!(
            package_name_from_id(
                "registry+https://github.com/rust-lang/crates.io-index#serde@1.0.0"
            ),
            "serde"
        );
    }

    #[test]
    fn environment_matches_cargo_test() {
        let plan = parse_build_messages(FIXTURE);
        let cli = &plan.programs[3];
        let info = PackageInfo {
            name: "xma-cli".into(),
            version: "1.2.3-beta.1+build5".into(),
            authors: vec!["A".into(), "B".into()],
            license: "GPL-3.0-only".into(),
            ..PackageInfo::default()
        };
        let inherited = [
            "PATH",
            "CARGO_PKG_NAME",
            "CARGO_BIN_NAME",
            "OUT_DIR",
            "XMA_REQUIRE_ORACLE",
        ]
        .map(OsString::from);
        let env = program_env(
            cli,
            Some(&info),
            None,
            &plan.bin_exes[&cli.package_id],
            OsStr::new("/usr/bin/cargo"),
            Some(OsStr::new("/w/target/debug")),
            inherited,
        );
        assert_eq!(
            env.remove,
            ["CARGO_PKG_NAME", "CARGO_BIN_NAME", "OUT_DIR"].map(OsString::from)
        );
        let get = |k: &str| {
            env.set
                .iter()
                .find(|(key, _)| key == k)
                .map(|(_, v)| v.to_string_lossy().into_owned())
        };
        assert_eq!(get("CARGO").as_deref(), Some("/usr/bin/cargo"));
        assert_eq!(
            get("CARGO_MANIFEST_DIR").as_deref(),
            Some("/w/crates/xma-cli")
        );
        assert_eq!(
            get("CARGO_MANIFEST_PATH").as_deref(),
            Some("/w/crates/xma-cli/Cargo.toml")
        );
        assert_eq!(get("CARGO_PKG_VERSION_MAJOR").as_deref(), Some("1"));
        assert_eq!(get("CARGO_PKG_VERSION_MINOR").as_deref(), Some("2"));
        assert_eq!(get("CARGO_PKG_VERSION_PATCH").as_deref(), Some("3"));
        assert_eq!(get("CARGO_PKG_VERSION_PRE").as_deref(), Some("beta.1"));
        assert_eq!(get("CARGO_PKG_AUTHORS").as_deref(), Some("A:B"));
        assert_eq!(get("CARGO_PKG_DESCRIPTION").as_deref(), Some(""));
        assert_eq!(
            get("CARGO_BIN_EXE_xma").as_deref(),
            Some("/w/target/debug/xma")
        );
        assert_eq!(get(dylib_path_var()).as_deref(), Some("/w/target/debug"));
        assert_eq!(get("OUT_DIR"), None);
        assert_eq!(
            get("XMA_REQUIRE_ORACLE"),
            None,
            "the caller's own variables pass through"
        );
        // Unit tests of the library get no CARGO_BIN_EXE_*, as with cargo.
        let lib = program_env(
            &plan.programs[1],
            Some(&info),
            Some(Path::new("/o")),
            &plan.bin_exes[&cli.package_id],
            OsStr::new("cargo"),
            None,
            [],
        );
        assert!(
            !lib.set
                .iter()
                .any(|(k, _)| k.to_string_lossy().starts_with("CARGO_BIN_EXE"))
        );
        assert!(lib.set.iter().any(|(k, v)| k == "OUT_DIR" && v == "/o"));
    }

    #[test]
    fn dylib_path_keeps_the_inherited_value_after_cargos() {
        let dirs = [PathBuf::from("/t/debug"), PathBuf::from("/sys/lib")];
        let inherited = std::env::join_paths(["/x", "/y"]).unwrap();
        let v = dylib_path_value(&dirs, Some(&inherited)).unwrap();
        let parts: Vec<PathBuf> = std::env::split_paths(&v).collect();
        assert_eq!(
            parts[..4],
            ["/t/debug", "/sys/lib", "/x", "/y"].map(PathBuf::from)
        );
    }

    fn prog(package: &str, kind: &str, name: &str) -> TestProgram {
        TestProgram {
            package_id: format!("path+file:///w/{package}#0.1.0"),
            package: package.into(),
            manifest_path: PathBuf::from(format!("/w/{package}/Cargo.toml")),
            kind: kind.into(),
            name: name.into(),
            src_path: PathBuf::from(format!("/w/{package}/tests/{name}.rs")),
            executable: PathBuf::from(format!("/w/target/{name}")),
        }
    }

    #[test]
    fn longest_previous_time_first_and_new_programs_before_all() {
        let mut programs = vec![
            prog("a", "test", "short"),
            prog("a", "test", "new1"),
            prog("a", "test", "long"),
            prog("b", "lib", "b"),
            prog("a", "test", "new2"),
        ];
        let times = parse_times(
            r#"{"seconds": {"a test short": 1.5, "a test long": 249.0, "b lib b": 30, "gone x y": 9}}"#,
        );
        order_by_previous_times(&mut programs, &times);
        let names: Vec<&str> = programs.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["new1", "new2", "long", "b", "short"]);
    }

    #[test]
    fn a_damaged_times_file_is_ignored() {
        assert!(parse_times("").is_empty());
        assert!(parse_times("{not json").is_empty());
        assert!(parse_times("[1, 2]").is_empty());
        assert!(parse_times(r#"{"seconds": {"a": "x", "b": -1}}"#).is_empty());
        let mut t = HashMap::new();
        t.insert("a test x".to_string(), 12.34567);
        let back = parse_times(&render_times(&t));
        assert_eq!(back["a test x"], 12.346);
    }

    #[test]
    fn result_lines_are_read() {
        let out = "running 3 tests\ntest a ... ok\n\ntest result: ok. 2 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.01s\n\n";
        assert_eq!(
            parse_result_line(out),
            Some(Counts {
                passed: 2,
                failed: 0,
                ignored: 1
            })
        );
        let failed = "test result: FAILED. 10 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out; finished in 249.13s";
        assert_eq!(
            parse_result_line(failed),
            Some(Counts {
                passed: 10,
                failed: 2,
                ignored: 0
            })
        );
        assert_eq!(parse_result_line("running 1 test\nthread panicked\n"), None);
        assert_eq!(parse_result_line(""), None);
    }

    fn result(label: &str, secs: f64, outcome: Outcome, counts: Option<Counts>) -> ProgramResult {
        ProgramResult {
            key: label.into(),
            label: label.into(),
            rerun: format!("cargo test -p x --test {label}"),
            seconds: secs,
            outcome,
            counts,
        }
    }

    fn ok_exit() -> Outcome {
        Outcome::Exited {
            success: true,
            status: "exit status: 0".into(),
        }
    }

    #[test]
    fn summary_totals_and_verdict() {
        let c = |p, f, i| {
            Some(Counts {
                passed: p,
                failed: f,
                ignored: i,
            })
        };
        let mut results = vec![
            result("p1", 2.0, ok_exit(), c(5, 0, 1)),
            result("p2", 249.0, ok_exit(), c(12, 0, 0)),
        ];
        let s = summarize(&results, Duration::from_secs(250));
        assert!(s.ok());
        assert_eq!(
            s.totals,
            Counts {
                passed: 17,
                failed: 0,
                ignored: 1
            }
        );
        assert_eq!(s.sum_seconds, 251.0);
        assert_eq!(s.slowest[0].0, "p2");

        results.push(result(
            "failing",
            3.0,
            Outcome::Exited {
                success: false,
                status: "exit status: 101".into(),
            },
            c(1, 1, 0),
        ));
        results.push(result(
            "killed",
            4.0,
            Outcome::Exited {
                success: false,
                status: "signal: 9 (SIGKILL)".into(),
            },
            None,
        ));
        results.push(result("silent", 1.0, ok_exit(), None));
        results.push(result(
            "missing",
            0.0,
            Outcome::StartFailed("No such file".into()),
            None,
        ));
        for i in 0..10 {
            results.push(result(&format!("q{i}"), 0.1, ok_exit(), c(1, 0, 0)));
        }
        let s = summarize(&results, Duration::from_secs(260));
        assert!(!s.ok());
        assert_eq!(s.programs, 16);
        assert_eq!(
            s.totals,
            Counts {
                passed: 28,
                failed: 1,
                ignored: 1
            }
        );
        let failed: Vec<&str> = s.failures.iter().map(|f| f.0.as_str()).collect();
        assert_eq!(failed, ["failing", "killed", "silent", "missing"]);
        assert!(s.failures[1].1.contains("SIGKILL"));
        assert!(s.failures[1].1.contains("no 'test result:' line"));
        assert_eq!(s.failures[2].1, "printed no 'test result:' line");
        assert!(s.failures[3].1.starts_with("could not be started"));
        assert_eq!(s.slowest.len(), 10);
        let text = s.render();
        assert!(text.contains("rerun: cargo test -p x --test failing"));
        assert!(text.contains("tests: 28 passed, 1 failed, 1 ignored"));
        assert!(text.contains("16 program(s), 4 failed"));
    }

    #[test]
    fn test_jobs_arguments() {
        assert_eq!(parse_test_args(&[], false), Ok(TestOptions::default()));
        assert_eq!(
            parse_test_args(&args(&["--test-jobs", "6"]), false)
                .unwrap()
                .test_jobs,
            Some(6)
        );
        assert_eq!(
            parse_test_args(&args(&["--test-jobs=1"]), false)
                .unwrap()
                .test_jobs,
            Some(1)
        );
        assert!(parse_test_args(&args(&["--test-jobs", "0"]), false).is_err());
        assert!(parse_test_args(&args(&["--test-jobs", "x"]), false).is_err());
        assert!(parse_test_args(&args(&["--test-jobs"]), false).is_err());
        assert!(parse_test_args(&args(&["--package", "xma-io"]), false).is_err());
        assert!(parse_test_args(&args(&["--frobnicate"]), false).is_err());
        let o = parse_test_args(
            &args(&["-p", "xma-io", "--package", "xma-model", "--test-jobs", "2"]),
            true,
        )
        .unwrap();
        assert_eq!(o.packages, ["xma-io", "xma-model"]);
        assert_eq!(o.test_jobs, Some(2));

        assert_eq!(resolve_test_jobs(Some(3), Some(OsStr::new("9")), 24), Ok(3));
        assert_eq!(resolve_test_jobs(None, Some(OsStr::new("9")), 24), Ok(9));
        assert_eq!(resolve_test_jobs(None, Some(OsStr::new("")), 24), Ok(8));
        assert_eq!(resolve_test_jobs(None, None, 24), Ok(8));
        assert_eq!(resolve_test_jobs(None, None, 2), Ok(1));
        assert!(resolve_test_jobs(None, Some(OsStr::new("0")), 24).is_err());
        assert_eq!(default_test_jobs(1), 1);
    }
}
