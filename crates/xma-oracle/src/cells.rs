//! Plain comma-separated cells, for files in XMALab's own export formats.

use std::ops::Range;
use std::path::{Path, PathBuf};

use crate::error::{OracleError, Result, ok};
use crate::parse::parse_f64;
use crate::table::split_lines;

/// A text file split into lines and comma-separated cells, with no header assumption.
///
/// This is for the files under a dump's `exports/` directories, which use XMALab's own
/// formats: some have a header row and some do not (`2DPoints.csv`, `xromm_vr/*.csv`),
/// some separate cells with `", "` (`*_LUT.csv`), some contain blank lines and text
/// lines (`PrecisionInfo.csv`, `*_MayaCam1.txt`), and missing values are written `NaN`.
///
/// Each line is split on `,` and each cell has surrounding spaces and tabs removed. A
/// blank line becomes a line with no cells. `\r\n` is tolerated. A trailing comma gives
/// an empty last cell. Line indices are 0-based (line `i` is on line `i + 1` of the file).
///
/// Accessors come as `try_*` (returning [`Result`]) and plain (panicking) forms.
#[derive(Debug, Clone)]
pub struct Cells {
    path: PathBuf,
    lines: Vec<Vec<String>>,
}

impl Cells {
    /// Reads a file.
    pub fn try_read(path: impl AsRef<Path>) -> Result<Cells> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| OracleError::file(path, format!("cannot read: {e}")))?;
        Ok(Cells::parse(&text, path))
    }

    /// Reads a file, panicking on error.
    #[track_caller]
    pub fn read(path: impl AsRef<Path>) -> Cells {
        ok(Cells::try_read(path))
    }

    /// Splits text. `origin` is the path used in error messages. Never fails.
    pub fn parse(text: &str, origin: impl AsRef<Path>) -> Cells {
        let lines = split_lines(text)
            .into_iter()
            .map(|l| {
                if l.trim().is_empty() {
                    Vec::new()
                } else {
                    l.split(',').map(|c| c.trim().to_string()).collect()
                }
            })
            .collect();
        Cells {
            path: origin.as_ref().to_path_buf(),
            lines,
        }
    }

    /// The file the cells were read from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Number of lines (a final newline does not add an empty line).
    pub fn len(&self) -> usize {
        self.lines.len()
    }

    /// True if the file has no lines.
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    /// The cells of line `i`; empty for a blank line. Panics if out of range.
    #[track_caller]
    pub fn line(&self, i: usize) -> &[String] {
        self.check_line(i);
        &self.lines[i]
    }

    /// All lines.
    pub fn lines(&self) -> &[Vec<String>] {
        &self.lines
    }

    /// True if line `i` is blank.
    #[track_caller]
    pub fn is_blank(&self, i: usize) -> bool {
        self.line(i).is_empty()
    }

    /// Ranges of line indices of the runs of non-blank lines, in order. A MayaCam 2 file,
    /// for example, gives five blocks: image size, camera matrix, rotation, translation and
    /// (if present) undistortion, each starting with its title line.
    pub fn blocks(&self) -> Vec<Range<usize>> {
        let mut out = Vec::new();
        let mut start = None;
        for (i, l) in self.lines.iter().enumerate() {
            match (l.is_empty(), start) {
                (false, None) => start = Some(i),
                (true, Some(s)) => {
                    out.push(s..i);
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(s) = start {
            out.push(s..self.lines.len());
        }
        out
    }

    /// Cell `j` of line `i` as text.
    pub fn try_str(&self, i: usize, j: usize) -> Result<&str> {
        self.try_check_line(i)?;
        self.lines[i].get(j).map(String::as_str).ok_or_else(|| {
            OracleError::cell(
                &self.path,
                None,
                Some(i + 1),
                j.to_string(),
                format!("line has only {} cells", self.lines[i].len()),
            )
        })
    }

    /// Cell `j` of line `i` as text, panicking on error.
    #[track_caller]
    pub fn str(&self, i: usize, j: usize) -> &str {
        ok(self.try_str(i, j))
    }

    /// Cell `j` of line `i` as `f64` (`NaN`, `nan`, `inf` and so on accepted).
    pub fn try_f64(&self, i: usize, j: usize) -> Result<f64> {
        let s = self.try_str(i, j)?;
        parse_f64(s).ok_or_else(|| {
            OracleError::cell(
                &self.path,
                None,
                Some(i + 1),
                j.to_string(),
                format!("cannot parse {s:?} as f64"),
            )
        })
    }

    /// Cell `j` of line `i` as `f64`, panicking on error.
    #[track_caller]
    pub fn f64(&self, i: usize, j: usize) -> f64 {
        ok(self.try_f64(i, j))
    }

    /// Every cell of line `i` as `f64`.
    pub fn try_line_f64(&self, i: usize) -> Result<Vec<f64>> {
        self.try_check_line(i)?;
        (0..self.lines[i].len())
            .map(|j| self.try_f64(i, j))
            .collect()
    }

    /// Every cell of line `i` as `f64`, panicking on error.
    #[track_caller]
    pub fn line_f64(&self, i: usize) -> Vec<f64> {
        ok(self.try_line_f64(i))
    }

    /// Every cell of the lines in `range` as `f64`, one vector per line. Blank lines in
    /// the range give empty vectors.
    pub fn try_lines_f64(&self, range: Range<usize>) -> Result<Vec<Vec<f64>>> {
        range.map(|i| self.try_line_f64(i)).collect()
    }

    /// Like [`Cells::try_lines_f64`], panicking on error.
    #[track_caller]
    pub fn lines_f64(&self, range: Range<usize>) -> Vec<Vec<f64>> {
        ok(self.try_lines_f64(range))
    }

    fn try_check_line(&self, i: usize) -> Result<()> {
        if i < self.lines.len() {
            Ok(())
        } else {
            Err(OracleError::file(
                &self.path,
                format!("line index {i} out of range ({} lines)", self.lines.len()),
            ))
        }
    }

    #[track_caller]
    fn check_line(&self, i: usize) {
        ok(self.try_check_line(i));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mayacam_like_blocks() {
        let text = "image size\n1024,1024\n\ncamera matrix\n3207.5,0,457.2\n0,3231.5,614.4\n0,0,1\n\ntranslation\n-4.09\n8.25\n87.4\n";
        let c = Cells::parse(text, "cam.txt");
        assert_eq!(c.len(), 12);
        assert!(c.is_blank(2));
        assert_eq!(c.blocks(), vec![0..2, 3..7, 8..12]);
        assert_eq!(c.str(0, 0), "image size");
        assert_eq!(c.line_f64(1), vec![1024.0, 1024.0]);
        assert_eq!(c.lines_f64(4..7)[2], vec![0.0, 0.0, 1.0]);
        assert_eq!(c.f64(11, 0), 87.4);
    }

    #[test]
    fn lut_like_spaces_nan_and_trailing_comma() {
        let c = Cells::parse("-1, -1, 381.82788\r\n1,NaN,\r\n\r\n", "lut.csv");
        assert_eq!(c.len(), 3);
        assert_eq!(c.line_f64(0), vec![-1.0, -1.0, 381.82788]);
        assert!(c.f64(1, 1).is_nan());
        assert_eq!(c.str(1, 2), "");
        let e = c.try_f64(1, 2).unwrap_err();
        assert_eq!(e.line, Some(2));
        assert_eq!(e.column.as_deref(), Some("2"));
        assert!(c.try_str(1, 7).is_err());
        assert!(c.try_line_f64(9).is_err());
        assert!(c.is_blank(2));
    }
}
