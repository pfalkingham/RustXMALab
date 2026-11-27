//! Finding the oracle directory and the reference datasets.

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::cells::Cells;
use crate::error::{OracleError, Result, ok};
use crate::manifest::Manifest;
use crate::table::Table;

/// Environment variable that overrides [`oracle_root`].
pub const ORACLE_ROOT_ENV: &str = "XMA_ORACLE_ROOT";

/// Environment variable that, when set to `1`, turns a missing dataset into a panic
/// instead of a skip (see [`dataset`]).
pub const REQUIRE_ORACLE_ENV: &str = "XMA_REQUIRE_ORACLE";

/// The workspace root: two directories above this crate's manifest.
pub fn workspace_root() -> PathBuf {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    crate_dir
        .parent()
        .and_then(Path::parent)
        .expect("xma-oracle lives at <workspace>/crates/xma-oracle")
        .to_path_buf()
}

/// The oracle directory: `$XMA_ORACLE_ROOT` if set and non-empty, otherwise
/// `<workspace root>/oracle`.
pub fn oracle_root() -> PathBuf {
    match std::env::var_os(ORACLE_ROOT_ENV) {
        Some(v) if !v.is_empty() => PathBuf::from(v),
        _ => workspace_root().join("oracle"),
    }
}

/// True if `XMA_REQUIRE_ORACLE` is `1`.
pub fn oracle_required() -> bool {
    std::env::var(REQUIRE_ORACLE_ENV).is_ok_and(|v| v == "1")
}

/// A reference dataset: a project file and the C++ dump made from it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dataset {
    /// Name in `oracle/datasets.json`, for example `pig_clean`.
    pub name: String,
    /// Absolute path of the `.xma` project.
    pub project: PathBuf,
    /// Absolute path of the dump directory.
    pub dump: PathBuf,
    /// Description from `datasets.json`.
    pub description: String,
}

impl Dataset {
    /// A dataset from explicit paths (for `xma parity` on a dump not in `datasets.json`).
    pub fn from_paths(
        name: impl Into<String>,
        project: impl Into<PathBuf>,
        dump: impl Into<PathBuf>,
    ) -> Dataset {
        Dataset {
            name: name.into(),
            project: project.into(),
            dump: dump.into(),
            description: String::new(),
        }
    }

    /// True if both the project file and the dump directory exist.
    pub fn is_present(&self) -> bool {
        self.project.is_file() && self.dump.is_dir()
    }

    /// Absolute path of a file in the dump, `rel` being relative to the dump directory
    /// (for example `trial_0/points2d.csv`).
    pub fn path(&self, rel: &str) -> PathBuf {
        self.dump.join(rel)
    }

    /// True if the file `rel` exists in the dump.
    pub fn has(&self, rel: &str) -> bool {
        self.path(rel).exists()
    }

    /// The directory of trial `t` (0-based) in the dump.
    pub fn trial_dir(&self, t: usize) -> PathBuf {
        self.dump.join(format!("trial_{t}"))
    }

    /// Reads a dump table, `rel` relative to the dump directory.
    pub fn try_table(&self, rel: &str) -> Result<Table> {
        let p = self.path(rel);
        if !p.is_file() {
            return Err(self.absent(&p, rel));
        }
        Table::try_read(p)
    }

    /// Reads a dump table, panicking on error, for example
    /// `ds.table("trial_0/points2d.csv")`.
    #[track_caller]
    pub fn table(&self, rel: &str) -> Table {
        ok(self.try_table(rel))
    }

    /// Reads a file in an export format as plain cells, `rel` relative to the dump
    /// directory (for example `trial_0/exports/3DPoints.csv`).
    pub fn try_cells(&self, rel: &str) -> Result<Cells> {
        let p = self.path(rel);
        if !p.is_file() {
            return Err(self.absent(&p, rel));
        }
        Cells::try_read(p)
    }

    /// Reads a file as plain cells, panicking on error.
    #[track_caller]
    pub fn cells(&self, rel: &str) -> Cells {
        ok(self.try_cells(rel))
    }

    /// Reads the dump's `manifest.json`.
    pub fn try_manifest(&self) -> Result<Manifest> {
        Manifest::try_read(self.path("manifest.json"))
    }

    /// Reads the dump's `manifest.json`, panicking on error.
    #[track_caller]
    pub fn manifest(&self) -> Manifest {
        ok(self.try_manifest())
    }

    fn absent(&self, p: &Path, rel: &str) -> OracleError {
        let mut msg = String::from("file does not exist");
        if let Ok(m) = self.try_manifest()
            && let Ok(missing) = m.try_missing()
            && let Some(entry) = missing.iter().find(|m| m.table == rel)
        {
            msg = format!("listed as missing in the manifest: {}", entry.reason);
        }
        OracleError::file(p, msg)
    }
}

/// Path of `datasets.json`.
pub fn datasets_file() -> PathBuf {
    oracle_root().join("datasets.json")
}

fn read_datasets() -> Result<(PathBuf, Value)> {
    let file = datasets_file();
    let text = std::fs::read_to_string(&file)
        .map_err(|e| OracleError::file(&file, format!("cannot read: {e}")))?;
    let json: Value = serde_json::from_str(&text)
        .map_err(|e| OracleError::file(&file, format!("invalid JSON: {e}")))?;
    Ok((file, json))
}

/// Names of every dataset in `datasets.json`, present or not, sorted by name.
pub fn try_dataset_names() -> Result<Vec<String>> {
    let (file, json) = read_datasets()?;
    let map = json
        .get("datasets")
        .and_then(Value::as_object)
        .ok_or_else(|| OracleError::file(&file, "no 'datasets' object"))?;
    let mut names: Vec<String> = map.keys().cloned().collect();
    names.sort();
    Ok(names)
}

/// Names of every dataset in `datasets.json`, panicking if the file cannot be read.
#[track_caller]
pub fn dataset_names() -> Vec<String> {
    ok(try_dataset_names())
}

/// Looks a dataset up in `datasets.json`, whether or not its files are present.
/// Returns `Ok(None)` if the name is not in the file.
pub fn try_dataset_entry(name: &str) -> Result<Option<Dataset>> {
    let (file, json) = read_datasets()?;
    let map = json
        .get("datasets")
        .and_then(Value::as_object)
        .ok_or_else(|| OracleError::file(&file, "no 'datasets' object"))?;
    let Some(entry) = map.get(name) else {
        return Ok(None);
    };
    let base = file.parent().map(Path::to_path_buf).unwrap_or_default();
    let field = |k: &str| {
        entry
            .get(k)
            .and_then(Value::as_str)
            .ok_or_else(|| OracleError::file(&file, format!("dataset '{name}' has no '{k}'")))
    };
    Ok(Some(Dataset {
        name: name.to_string(),
        project: base.join(field("project")?),
        dump: base.join(field("dump")?),
        description: entry
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
    }))
}

/// Looks up a reference dataset by name, for use in tests:
///
/// ```no_run
/// let Some(ds) = xma_oracle::dataset("pig_clean") else { return };
/// ```
///
/// - Panics if `datasets.json` cannot be read or does not list `name` (a programming error).
/// - If the project file or the dump directory is absent, prints
///   `oracle dataset '<name>' is not present; skipping` to stderr and returns `None`,
///   unless `XMA_REQUIRE_ORACLE=1`, in which case it panics.
#[track_caller]
pub fn dataset(name: &str) -> Option<Dataset> {
    let entry = match try_dataset_entry(name) {
        Ok(Some(d)) => d,
        Ok(None) => panic!(
            "oracle dataset '{name}' is not listed in {} (known: {})",
            datasets_file().display(),
            dataset_names().join(", ")
        ),
        Err(e) => panic!("{e}"),
    };
    if entry.is_present() {
        return Some(entry);
    }
    if oracle_required() {
        panic!(
            "oracle dataset '{name}' is not present ({} or {} missing) and {REQUIRE_ORACLE_ENV}=1",
            entry.project.display(),
            entry.dump.display()
        );
    }
    eprintln!("oracle dataset '{name}' is not present; skipping");
    None
}
