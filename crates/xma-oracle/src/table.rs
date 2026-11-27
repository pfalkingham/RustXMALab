//! Dump tables: CSV with one header row (rules in `oracle/FORMAT.md`).

use std::path::{Path, PathBuf};

use crate::error::{OracleError, Result, ok};
use crate::parse::{parse_f64, parse_i64, parse_usize};

/// A dump table: comma-separated, one header row, no quoting.
///
/// Reading tolerates `\r\n` line ends and a missing final newline. Every row must have
/// as many cells as the header. Cells are kept as text and parsed on access, so a
/// parse error names the file, the row and the column.
///
/// Every accessor comes in two forms: `try_*` returns a [`Result`], and the plain
/// form panics with the same message (convenient in tests).
#[derive(Debug, Clone)]
pub struct Table {
    path: PathBuf,
    columns: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl Table {
    /// Reads a table from a file.
    pub fn try_read(path: impl AsRef<Path>) -> Result<Table> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| OracleError::file(path, format!("cannot read: {e}")))?;
        Table::try_parse(&text, path)
    }

    /// Reads a table from a file, panicking on error.
    #[track_caller]
    pub fn read(path: impl AsRef<Path>) -> Table {
        ok(Table::try_read(path))
    }

    /// Parses table text. `origin` is the path used in error messages.
    pub fn try_parse(text: &str, origin: impl AsRef<Path>) -> Result<Table> {
        let path = origin.as_ref();
        let mut lines = split_lines(text);
        if lines.is_empty() {
            return Err(OracleError::file(path, "empty file, no header row"));
        }
        let header = lines.remove(0);
        let columns: Vec<String> = header.split(',').map(str::to_string).collect();
        for (i, c) in columns.iter().enumerate() {
            if c.is_empty() {
                return Err(OracleError::file(path, format!("header cell {i} is empty")));
            }
            if columns[..i].contains(c) {
                return Err(OracleError::file(path, format!("duplicate column '{c}'")));
            }
        }
        let mut rows = Vec::with_capacity(lines.len());
        for (r, line) in lines.iter().enumerate() {
            let cells: Vec<String> = line.split(',').map(str::to_string).collect();
            if cells.len() != columns.len() {
                return Err(OracleError {
                    path: path.to_path_buf(),
                    row: Some(r),
                    line: Some(r + 2),
                    column: None,
                    message: format!(
                        "has {} cells, the header has {}",
                        cells.len(),
                        columns.len()
                    ),
                });
            }
            rows.push(cells);
        }
        Ok(Table {
            path: path.to_path_buf(),
            columns,
            rows,
        })
    }

    /// Parses table text, panicking on error.
    #[track_caller]
    pub fn parse(text: &str, origin: impl AsRef<Path>) -> Table {
        ok(Table::try_parse(text, origin))
    }

    /// The file the table was read from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Column names, in file order.
    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    /// True if the table has a column of this name.
    pub fn has_column(&self, name: &str) -> bool {
        self.columns.iter().any(|c| c == name)
    }

    /// Number of data rows (the header is not counted).
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// True if the table has no data rows.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Index of a column.
    pub fn try_column_index(&self, name: &str) -> Result<usize> {
        self.columns.iter().position(|c| c == name).ok_or_else(|| {
            OracleError::file(
                &self.path,
                format!("no column '{name}' (columns: {})", self.columns.join(",")),
            )
        })
    }

    /// Row `i` (0-based, header excluded). Panics if out of range.
    #[track_caller]
    pub fn row(&self, i: usize) -> Row<'_> {
        assert!(
            i < self.rows.len(),
            "{}: row {i} out of range ({} rows)",
            self.path.display(),
            self.rows.len()
        );
        Row {
            table: self,
            index: i,
        }
    }

    /// All rows in file order.
    pub fn rows(&self) -> impl ExactSizeIterator<Item = Row<'_>> + '_ {
        (0..self.rows.len()).map(move |index| Row { table: self, index })
    }

    /// The first row whose integer key columns equal the given values, for example
    /// `t.try_find(&[("marker", 3), ("camera", 1), ("frame", 40)])`. Linear search.
    pub fn try_find(&self, keys: &[(&str, i64)]) -> Result<Option<Row<'_>>> {
        let idx: Vec<(usize, i64)> = keys
            .iter()
            .map(|(k, v)| Ok((self.try_column_index(k)?, *v)))
            .collect::<Result<_>>()?;
        for row in self.rows() {
            let mut all = true;
            for &(c, v) in &idx {
                if row.try_i64_at(c)? != v {
                    all = false;
                    break;
                }
            }
            if all {
                return Ok(Some(row));
            }
        }
        Ok(None)
    }

    /// Like [`Table::try_find`], panicking on a parse error or a missing column.
    #[track_caller]
    pub fn find(&self, keys: &[(&str, i64)]) -> Option<Row<'_>> {
        ok(self.try_find(keys))
    }

    /// A whole column as `f64` (see [`crate::parse_f64`] for the accepted spellings).
    pub fn try_column_f64(&self, name: &str) -> Result<Vec<f64>> {
        let c = self.try_column_index(name)?;
        self.rows().map(|r| r.try_f64_at(c)).collect()
    }

    /// A whole column as `f64`, panicking on error.
    #[track_caller]
    pub fn column_f64(&self, name: &str) -> Vec<f64> {
        ok(self.try_column_f64(name))
    }

    /// A whole column as `i64`.
    pub fn try_column_i64(&self, name: &str) -> Result<Vec<i64>> {
        let c = self.try_column_index(name)?;
        self.rows().map(|r| r.try_i64_at(c)).collect()
    }

    /// A whole column as `i64`, panicking on error.
    #[track_caller]
    pub fn column_i64(&self, name: &str) -> Vec<i64> {
        ok(self.try_column_i64(name))
    }

    /// A whole column as `usize`.
    pub fn try_column_usize(&self, name: &str) -> Result<Vec<usize>> {
        let c = self.try_column_index(name)?;
        self.rows().map(|r| r.try_usize_at(c)).collect()
    }

    /// A whole column as `usize`, panicking on error.
    #[track_caller]
    pub fn column_usize(&self, name: &str) -> Vec<usize> {
        ok(self.try_column_usize(name))
    }

    /// A whole column as text.
    pub fn try_column_str(&self, name: &str) -> Result<Vec<&str>> {
        let c = self.try_column_index(name)?;
        Ok(self.rows.iter().map(|r| r[c].as_str()).collect())
    }

    /// A whole column as text, panicking if the column is absent.
    #[track_caller]
    pub fn column_str(&self, name: &str) -> Vec<&str> {
        ok(self.try_column_str(name))
    }

    fn cell_error(&self, row: usize, col: usize, message: String) -> OracleError {
        OracleError::cell(
            &self.path,
            Some(row),
            Some(row + 2),
            self.columns[col].clone(),
            message,
        )
    }
}

/// One data row of a [`Table`].
#[derive(Debug, Clone, Copy)]
pub struct Row<'a> {
    table: &'a Table,
    index: usize,
}

impl<'a> Row<'a> {
    /// 0-based row index (the header is not a row; the row is on line `index + 2`).
    pub fn index(&self) -> usize {
        self.index
    }

    /// The table this row belongs to.
    pub fn table(&self) -> &'a Table {
        self.table
    }

    fn raw(&self, col: usize) -> &'a str {
        &self.table.rows[self.index][col]
    }

    /// A cell as text.
    pub fn try_str(&self, column: &str) -> Result<&'a str> {
        Ok(self.raw(self.table.try_column_index(column)?))
    }

    /// A cell as text, panicking if the column is absent.
    #[track_caller]
    pub fn str(&self, column: &str) -> &'a str {
        ok(self.try_str(column))
    }

    /// A cell as `f64` (see [`crate::parse_f64`] for the accepted spellings).
    pub fn try_f64(&self, column: &str) -> Result<f64> {
        self.try_f64_at(self.table.try_column_index(column)?)
    }

    /// A cell as `f64`, panicking on error.
    #[track_caller]
    pub fn f64(&self, column: &str) -> f64 {
        ok(self.try_f64(column))
    }

    /// A cell as `i64`.
    pub fn try_i64(&self, column: &str) -> Result<i64> {
        self.try_i64_at(self.table.try_column_index(column)?)
    }

    /// A cell as `i64`, panicking on error.
    #[track_caller]
    pub fn i64(&self, column: &str) -> i64 {
        ok(self.try_i64(column))
    }

    /// A cell as `usize`.
    pub fn try_usize(&self, column: &str) -> Result<usize> {
        self.try_usize_at(self.table.try_column_index(column)?)
    }

    /// A cell as `usize`, panicking on error.
    #[track_caller]
    pub fn usize(&self, column: &str) -> usize {
        ok(self.try_usize(column))
    }

    /// A cell as `bool`: `0` is false, `1` is true; anything else is an error.
    pub fn try_bool(&self, column: &str) -> Result<bool> {
        let c = self.table.try_column_index(column)?;
        match self.raw(c).trim() {
            "0" => Ok(false),
            "1" => Ok(true),
            s => Err(self.table.cell_error(
                self.index,
                c,
                format!("cannot parse {s:?} as bool (0 or 1)"),
            )),
        }
    }

    /// A cell as `bool`, panicking on error.
    #[track_caller]
    pub fn bool(&self, column: &str) -> bool {
        ok(self.try_bool(column))
    }

    /// A cell as `f64`, by column index.
    pub fn try_f64_at(&self, col: usize) -> Result<f64> {
        let s = self.raw(col);
        parse_f64(s).ok_or_else(|| {
            self.table
                .cell_error(self.index, col, format!("cannot parse {s:?} as f64"))
        })
    }

    /// A cell as `i64`, by column index.
    pub fn try_i64_at(&self, col: usize) -> Result<i64> {
        let s = self.raw(col);
        parse_i64(s).ok_or_else(|| {
            self.table
                .cell_error(self.index, col, format!("cannot parse {s:?} as i64"))
        })
    }

    /// A cell as `usize`, by column index.
    pub fn try_usize_at(&self, col: usize) -> Result<usize> {
        let s = self.raw(col);
        parse_usize(s).ok_or_else(|| {
            self.table
                .cell_error(self.index, col, format!("cannot parse {s:?} as usize"))
        })
    }
}

/// Splits text into lines on `\n`, removing a trailing `\r` from each, and dropping
/// the empty piece after a final newline.
pub(crate) fn split_lines(text: &str) -> Vec<&str> {
    let mut lines: Vec<&str> = text
        .split('\n')
        .map(|l| l.strip_suffix('\r').unwrap_or(l))
        .collect();
    if lines.last() == Some(&"") {
        lines.pop();
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: &str = "marker,camera,frame,status,x,y,size\r\n\
                     0,0,0,50,566,468,2.2361679077100001\r\n\
                     0,1,0,-1,-2,-2,nan\r\n\
                     1,0,0,40,-nan,inf,-inf";

    #[test]
    fn reads_rows_and_columns() {
        let t = Table::parse(T, "t.csv");
        assert_eq!(t.len(), 3);
        assert_eq!(t.columns()[6], "size");
        assert!(t.has_column("frame") && !t.has_column("z"));
        let r = t.row(0);
        assert_eq!(r.i64("status"), 50);
        assert_eq!(r.usize("marker"), 0);
        assert_eq!(r.f64("size"), 2.23616790771);
        assert_eq!(r.str("x"), "566");
        assert_eq!(t.row(1).i64("status"), -1);
        assert!(t.row(1).f64("size").is_nan());
        assert!(t.row(2).f64("x").is_nan());
        assert_eq!(t.row(2).f64("y"), f64::INFINITY);
        assert_eq!(t.column_i64("status"), vec![50, -1, 40]);
        assert_eq!(t.column_str("camera"), vec!["0", "1", "0"]);
        assert_eq!(t.column_f64("x")[..2], [566.0, -2.0]);
        let found = t.find(&[("marker", 0), ("camera", 1)]).unwrap();
        assert_eq!(found.index(), 1);
        assert!(t.find(&[("marker", 5)]).is_none());
    }

    #[test]
    fn errors_name_file_row_and_column() {
        let t = Table::parse("a,b\n1,x\n", "dir/t.csv");
        let e = t.row(0).try_f64("b").unwrap_err();
        assert_eq!(e.row, Some(0));
        assert_eq!(e.line, Some(2));
        assert_eq!(e.column.as_deref(), Some("b"));
        let msg = e.to_string();
        assert!(msg.contains("dir/t.csv") && msg.contains("row 0") && msg.contains("'b'"));
        assert!(t.row(0).try_usize("a").is_ok());
        assert!(
            t.row(0)
                .try_f64("zz")
                .unwrap_err()
                .to_string()
                .contains("no column 'zz'")
        );
        assert!(t.try_column_usize("b").is_err());
        let e = Table::try_parse("a,b\n1\n", "t.csv").unwrap_err();
        assert_eq!(e.line, Some(2));
        assert!(Table::try_parse("", "t.csv").is_err());
        assert!(Table::try_parse("a,a\n", "t.csv").is_err());
    }

    #[test]
    #[should_panic(expected = "cannot parse \"x\" as i64")]
    fn plain_accessors_panic_with_message() {
        Table::parse("a\nx\n", "t.csv").row(0).i64("a");
    }

    #[test]
    fn header_only_table_is_empty() {
        let t = Table::parse("event,name,frame,value\n", "events.csv");
        assert!(t.is_empty());
        assert_eq!(t.column_f64("value"), Vec::<f64>::new());
    }
}
