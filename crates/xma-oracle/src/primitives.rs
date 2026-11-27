//! The primitive oracle: outputs of single OpenCV routines on fixed inputs, in
//! `oracle/reference/primitives/` (made by `oracle/py/gen_primitives.py`).
//!
//! Each case is a directory named `<routine>-<variant>` holding a `case.json` and the
//! files it lists under `inputs` and `outputs`. A listed item is either a CSV array
//! (`file`, `columns`, `n_rows`) or a raw image (`file`, `dtype` `u8`/`u16`/`f32`,
//! `rows`, `cols`, `channels`; little-endian, row order, channels interleaved).
//! CSV arrays have a header row; `name,value` arrays hold named scalars.

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::error::{OracleError, Result, ok};
use crate::locate::oracle_root;
use crate::table::Table;

/// The directory of the primitive oracle: `<oracle root>/reference/primitives`.
pub fn primitives_root() -> PathBuf {
    oracle_root().join("reference").join("primitives")
}

fn read_json(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| OracleError::file(path, format!("cannot read: {e}")))?;
    serde_json::from_str(&text).map_err(|e| OracleError::file(path, format!("invalid JSON: {e}")))
}

/// Reads `index.json` (fields `format_version`, `generator`, `environment`, `n_cases`,
/// `cases`).
pub fn try_primitive_index() -> Result<Value> {
    read_json(&primitives_root().join("index.json"))
}

/// The routine key of a case name: the part before the first `-`. For example
/// `svd-solvez-triang4_cams2` → `svd`, `solve-svd-overdet_6x3` → `solve`,
/// `solvepnp-epnp-n6-noise0-cube` → `solvepnp`.
pub fn routine_key(case_name: &str) -> &str {
    case_name.split('-').next().unwrap_or(case_name)
}

/// Names of every case in `index.json`, sorted.
pub fn try_case_names() -> Result<Vec<String>> {
    try_case_names_in("index.json")
}

/// Names of every case listed in another index file of the primitives directory (for
/// example `index_contours.json`, written by `oracle/py/gen_primitives_contours.py`),
/// sorted. The file has the layout of `index.json`.
pub fn try_case_names_in(index_file: &str) -> Result<Vec<String>> {
    let path = primitives_root().join(index_file);
    let index = read_json(&path)?;
    let cases = index
        .get("cases")
        .and_then(Value::as_array)
        .ok_or_else(|| OracleError::file(&path, "no 'cases' array"))?;
    let mut names = cases
        .iter()
        .map(|c| {
            c.get("name")
                .and_then(Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| OracleError::file(&path, format!("a case has no 'name': {c}")))
        })
        .collect::<Result<Vec<_>>>()?;
    names.sort();
    Ok(names)
}

/// Every case listed in `index_file` whose [`routine_key`] equals `routine`, sorted by
/// name, panicking on error.
#[track_caller]
pub fn cases_in(index_file: &str, routine: &str) -> Vec<Case> {
    ok(try_case_names_in(index_file).and_then(|names| {
        names
            .iter()
            .filter(|n| routine_key(n) == routine)
            .map(|n| try_case(n))
            .collect()
    }))
}

/// Every case whose [`routine_key`] equals `routine`, sorted by name.
///
/// Keys in use: `rodrigues`, `undistortpoints`, `projectpoints`, `svd` (SVD::solveZ),
/// `invert` (invert DECOMP_SVD), `solve` (solve DECOMP_SVD), `sortidx`, `solvepnp`,
/// `calibratecamera`, `gaussianblur`, `medianblur3`, `threshold`, `findcontours`,
/// `minenclosingcircle`, `detection`, `tracking`, `penalty`, `matchtemplate`,
/// `minmaxloc`, `getrectsubpix`, `remap`, `initundistortrectifymap`, `flip`,
/// `convertto`, `cvtcolor`, `simpleblob`, `gftt`, `subtract`.
pub fn try_cases(routine: &str) -> Result<Vec<Case>> {
    try_case_names()?
        .iter()
        .filter(|n| routine_key(n) == routine)
        .map(|n| try_case(n))
        .collect()
}

/// Every case of a routine, panicking on error. Returns an empty vector if there is
/// none, so callers should usually assert that it is not empty.
#[track_caller]
pub fn cases(routine: &str) -> Vec<Case> {
    ok(try_cases(routine))
}

/// Reads one case by directory name.
pub fn try_case(name: &str) -> Result<Case> {
    let dir = primitives_root().join(name);
    let json = read_json(&dir.join("case.json"))?;
    Ok(Case {
        name: name.to_string(),
        dir,
        json,
    })
}

/// Reads one case by directory name, panicking on error.
#[track_caller]
pub fn case(name: &str) -> Case {
    ok(try_case(name))
}

/// One primitive case.
#[derive(Debug, Clone)]
pub struct Case {
    /// Directory name, for example `rodrigues-generic`.
    pub name: String,
    /// Absolute path of the case directory.
    pub dir: PathBuf,
    /// The parsed `case.json`.
    pub json: Value,
}

impl Case {
    /// `routine` from `case.json`, for example `cv::Rodrigues (vector->matrix->vector)`.
    pub fn routine(&self) -> &str {
        self.json
            .get("routine")
            .and_then(Value::as_str)
            .unwrap_or_default()
    }

    /// The `args` object (may be empty).
    pub fn args(&self) -> &Value {
        self.json.get("args").unwrap_or(&Value::Null)
    }

    /// One argument, or an error if absent.
    pub fn try_arg(&self, name: &str) -> Result<&Value> {
        self.args()
            .get(name)
            .ok_or_else(|| self.err(format!("no argument '{name}' (args: {})", self.args())))
    }

    /// One argument, panicking if absent.
    #[track_caller]
    pub fn arg(&self, name: &str) -> &Value {
        ok(self.try_arg(name))
    }

    /// A numeric argument as `f64`, panicking if absent or not a number.
    #[track_caller]
    pub fn arg_f64(&self, name: &str) -> f64 {
        let v = self.arg(name);
        v.as_f64()
            .unwrap_or_else(|| panic!("{}: argument '{name}' = {v} is not a number", self.name))
    }

    /// An integer argument, panicking if absent or not an integer.
    #[track_caller]
    pub fn arg_i64(&self, name: &str) -> i64 {
        let v = self.arg(name);
        v.as_i64()
            .unwrap_or_else(|| panic!("{}: argument '{name}' = {v} is not an integer", self.name))
    }

    /// `notes` from `case.json`.
    pub fn notes(&self) -> Vec<&str> {
        self.json
            .get("notes")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default()
    }

    /// Names listed under `inputs`, sorted by name.
    pub fn inputs(&self) -> Vec<&str> {
        self.keys("inputs")
    }

    /// Names listed under `outputs`, sorted by name.
    pub fn outputs(&self) -> Vec<&str> {
        self.keys("outputs")
    }

    fn keys(&self, section: &str) -> Vec<&str> {
        self.json
            .get(section)
            .and_then(Value::as_object)
            .map(|o| o.keys().map(String::as_str).collect())
            .unwrap_or_default()
    }

    /// The `case.json` description of an input or output item. Inputs are searched
    /// first, then outputs.
    pub fn try_item(&self, key: &str) -> Result<&Value> {
        for section in ["inputs", "outputs"] {
            if let Some(v) = self.json.get(section).and_then(|s| s.get(key)) {
                return Ok(v);
            }
        }
        Err(self.err(format!(
            "no input or output '{key}' (inputs: {:?}, outputs: {:?})",
            self.inputs(),
            self.outputs()
        )))
    }

    /// Absolute path of a file in the case directory.
    pub fn file(&self, file_name: &str) -> PathBuf {
        self.dir.join(file_name)
    }

    fn item_file(&self, key: &str) -> Result<(PathBuf, &Value)> {
        let item = self.try_item(key)?;
        let f = item
            .get("file")
            .and_then(Value::as_str)
            .ok_or_else(|| self.err(format!("item '{key}' has no 'file'")))?;
        Ok((self.file(f), item))
    }

    /// A CSV item as a [`Table`] (access by column name). Checks `n_rows` and `columns`.
    pub fn try_table(&self, key: &str) -> Result<Table> {
        let (path, item) = self.item_file(key)?;
        let t = Table::try_read(&path)?;
        if let Some(n) = item.get("n_rows").and_then(Value::as_u64)
            && t.len() as u64 != n
        {
            return Err(OracleError::file(
                &path,
                format!("has {} rows, case.json says {n}", t.len()),
            ));
        }
        if let Some(cols) = item.get("columns").and_then(Value::as_array) {
            let want: Vec<&str> = cols.iter().filter_map(Value::as_str).collect();
            if want != t.columns().iter().map(String::as_str).collect::<Vec<_>>() {
                return Err(OracleError::file(
                    &path,
                    format!("columns {:?}, case.json says {want:?}", t.columns()),
                ));
            }
        }
        Ok(t)
    }

    /// A CSV item as a [`Table`], panicking on error.
    #[track_caller]
    pub fn table(&self, key: &str) -> Table {
        ok(self.try_table(key))
    }

    /// A CSV item as a dense row-major matrix of `f64` (every cell must be numeric).
    pub fn try_matrix(&self, key: &str) -> Result<Matrix> {
        let t = self.try_table(key)?;
        let cols = t.columns().len();
        let mut data = Vec::with_capacity(t.len() * cols);
        for r in t.rows() {
            for c in 0..cols {
                data.push(r.try_f64_at(c)?);
            }
        }
        Ok(Matrix {
            rows: t.len(),
            cols,
            columns: t.columns().to_vec(),
            data,
        })
    }

    /// A CSV item as a matrix, panicking on error. For example
    /// `case.matrix("R")` for a Rodrigues case gives a 3×3 matrix.
    #[track_caller]
    pub fn matrix(&self, key: &str) -> Matrix {
        ok(self.try_matrix(key))
    }

    /// A `name,value` CSV item (such as `status`, `retval`, `result`) as pairs in file order.
    pub fn try_scalars(&self, key: &str) -> Result<Vec<(String, f64)>> {
        let t = self.try_table(key)?;
        t.try_column_index("name")?;
        let v = t.try_column_index("value")?;
        t.rows()
            .map(|r| Ok((r.try_str("name")?.to_string(), r.try_f64_at(v)?)))
            .collect()
    }

    /// One named scalar from a `name,value` item, for example
    /// `case.scalar("status", "ok")` or `case.scalar("retval", "retval")`.
    pub fn try_scalar(&self, key: &str, name: &str) -> Result<f64> {
        let (path, _) = self.item_file(key)?;
        self.try_scalars(key)?
            .into_iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v)
            .ok_or_else(|| OracleError::file(&path, format!("no scalar named '{name}'")))
    }

    /// One named scalar, panicking on error.
    #[track_caller]
    pub fn scalar(&self, key: &str, name: &str) -> f64 {
        ok(self.try_scalar(key, name))
    }

    /// A raw image item (`.u8`, `.u16` or `.f32`). Checks the file size against
    /// `rows × cols × channels`.
    pub fn try_image(&self, key: &str) -> Result<Image> {
        let (path, item) = self.item_file(key)?;
        let dim = |k: &str| {
            item.get(k)
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| OracleError::file(&path, format!("case.json item has no '{k}'")))
        };
        let rows = dim("rows")?;
        let cols = dim("cols")?;
        let channels = item
            .get("channels")
            .and_then(Value::as_u64)
            .map_or(1, |v| v as usize);
        let dtype = item
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| OracleError::file(&path, "case.json item has no string 'dtype'"))?;
        let bytes = std::fs::read(&path)
            .map_err(|e| OracleError::file(&path, format!("cannot read: {e}")))?;
        let n = rows * cols * channels;
        let size = match dtype {
            "u8" => 1,
            "u16" => 2,
            "f32" => 4,
            other => {
                return Err(OracleError::file(&path, format!("unknown dtype '{other}'")));
            }
        };
        if bytes.len() != n * size {
            return Err(OracleError::file(
                &path,
                format!(
                    "{} bytes, expected {rows}×{cols}×{channels}×{size} = {}",
                    bytes.len(),
                    n * size
                ),
            ));
        }
        let data = match dtype {
            "u8" => ImageData::U8(bytes),
            "u16" => ImageData::U16(
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| u16::from_le_bytes(*b))
                    .collect(),
            ),
            _ => ImageData::F32(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect(),
            ),
        };
        Ok(Image {
            rows,
            cols,
            channels,
            data,
        })
    }

    /// A raw image item, panicking on error.
    #[track_caller]
    pub fn image(&self, key: &str) -> Image {
        ok(self.try_image(key))
    }

    fn err(&self, message: impl Into<String>) -> OracleError {
        OracleError::file(&self.dir.join("case.json"), message)
    }
}

/// A dense row-major matrix read from a CSV array.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    /// Number of rows (data rows of the CSV).
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Column names from the CSV header (`c0`, `c1`, ... or names such as `x`, `y`).
    pub columns: Vec<String>,
    /// Values in row order: element `(r, c)` is `data[r * cols + c]`.
    pub data: Vec<f64>,
}

impl Matrix {
    /// Element `(r, c)`. Panics if out of range.
    #[track_caller]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        assert!(r < self.rows && c < self.cols, "({r}, {c}) out of range");
        self.data[r * self.cols + c]
    }

    /// Row `r` as a slice.
    #[track_caller]
    pub fn row(&self, r: usize) -> &[f64] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// A column by header name.
    #[track_caller]
    pub fn column(&self, name: &str) -> Vec<f64> {
        let c = self
            .columns
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("no column '{name}' (columns: {:?})", self.columns));
        (0..self.rows).map(|r| self.get(r, c)).collect()
    }

    /// The values in row order as integer indices, panicking if any is not a
    /// non-negative integer (for `sortidx` outputs).
    #[track_caller]
    pub fn to_usize(&self) -> Vec<usize> {
        self.data
            .iter()
            .map(|&v| {
                assert!(
                    v >= 0.0 && v.fract() == 0.0 && v < usize::MAX as f64,
                    "{v} is not an index"
                );
                v as usize
            })
            .collect()
    }
}

/// Pixel data of a raw image.
#[derive(Debug, Clone, PartialEq)]
pub enum ImageData {
    /// 8-bit unsigned (`.u8`).
    U8(Vec<u8>),
    /// 16-bit unsigned, little-endian on disk (`.u16`).
    U16(Vec<u16>),
    /// 32-bit float, little-endian on disk (`.f32`).
    F32(Vec<f32>),
}

/// A raw image from a primitive case: row order, channels interleaved.
#[derive(Debug, Clone, PartialEq)]
pub struct Image {
    /// Height in pixels.
    pub rows: usize,
    /// Width in pixels.
    pub cols: usize,
    /// Channels per pixel.
    pub channels: usize,
    /// The pixels.
    pub data: ImageData,
}

impl Image {
    /// The pixels as `u8`, panicking if the image is another type.
    #[track_caller]
    pub fn u8(&self) -> &[u8] {
        match &self.data {
            ImageData::U8(v) => v,
            other => panic!("image is not u8 ({})", dtype_name(other)),
        }
    }

    /// The pixels as `u16`, panicking if the image is another type.
    #[track_caller]
    pub fn u16(&self) -> &[u16] {
        match &self.data {
            ImageData::U16(v) => v,
            other => panic!("image is not u16 ({})", dtype_name(other)),
        }
    }

    /// The pixels as `f32`, panicking if the image is another type.
    #[track_caller]
    pub fn f32(&self) -> &[f32] {
        match &self.data {
            ImageData::F32(v) => v,
            other => panic!("image is not f32 ({})", dtype_name(other)),
        }
    }
}

fn dtype_name(d: &ImageData) -> &'static str {
    match d {
        ImageData::U8(_) => "u8",
        ImageData::U16(_) => "u16",
        ImageData::F32(_) => "f32",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routine_keys() {
        assert_eq!(routine_key("svd-solvez-triang4_cams2"), "svd");
        assert_eq!(routine_key("solve-svd-overdet_6x3"), "solve");
        assert_eq!(routine_key("solvepnp-epnp-n4-planar"), "solvepnp");
        assert_eq!(routine_key("sortidx"), "sortidx");
    }

    #[test]
    fn matrix_access() {
        let m = Matrix {
            rows: 2,
            cols: 3,
            columns: vec!["c0".into(), "c1".into(), "c2".into()],
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        assert_eq!(m.get(1, 2), 6.0);
        assert_eq!(m.row(1), &[4.0, 5.0, 6.0]);
        assert_eq!(m.column("c1"), vec![2.0, 5.0]);
        assert_eq!(m.to_usize(), vec![1, 2, 3, 4, 5, 6]);
    }
}
