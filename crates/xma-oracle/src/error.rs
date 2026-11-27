//! The error type shared by every reader in this crate.

use std::fmt;
use std::path::{Path, PathBuf};

/// Result type of the `try_*` functions in this crate.
pub type Result<T> = std::result::Result<T, OracleError>;

/// An error reading or interpreting an oracle file.
///
/// It names the file and, where they apply, the 0-based data row (with the 1-based
/// line number in the file) and the column. `Display` gives one line, for example
/// `oracle/data/.../points2d.csv: row 12 (line 14), column 'x': cannot parse "abc" as f64`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OracleError {
    /// The file concerned.
    pub path: PathBuf,
    /// 0-based data row (for tables, the header is not a row), if the error concerns one.
    pub row: Option<usize>,
    /// 1-based line number in the file, if known.
    pub line: Option<usize>,
    /// Column name (tables) or 0-based cell index written as a number (plain cells).
    pub column: Option<String>,
    /// What went wrong.
    pub message: String,
}

impl OracleError {
    /// An error about a whole file.
    pub fn file(path: &Path, message: impl Into<String>) -> Self {
        OracleError {
            path: path.to_path_buf(),
            row: None,
            line: None,
            column: None,
            message: message.into(),
        }
    }

    /// An error about one cell.
    pub fn cell(
        path: &Path,
        row: Option<usize>,
        line: Option<usize>,
        column: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        OracleError {
            path: path.to_path_buf(),
            row,
            line,
            column: Some(column.into()),
            message: message.into(),
        }
    }
}

impl fmt::Display for OracleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.path.display())?;
        match (self.row, self.line) {
            (Some(r), Some(l)) => write!(f, ": row {r} (line {l})")?,
            (Some(r), None) => write!(f, ": row {r}")?,
            (None, Some(l)) => write!(f, ": line {l}")?,
            (None, None) => {}
        }
        if let Some(c) = &self.column {
            write!(f, ", column '{c}'")?;
        }
        write!(f, ": {}", self.message)
    }
}

impl std::error::Error for OracleError {}

/// Unwraps a result, panicking with the error's one-line message.
#[track_caller]
pub(crate) fn ok<T>(r: Result<T>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => panic!("{e}"),
    }
}
