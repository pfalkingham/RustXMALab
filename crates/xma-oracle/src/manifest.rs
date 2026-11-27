//! A dump's `manifest.json` (shape described in `oracle/cpp/DEVIATIONS.md`, item 3).

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::error::{OracleError, Result, ok};

/// One entry of the manifest's `settings` array.
#[derive(Debug, Clone, PartialEq)]
pub struct Setting {
    /// Setting name as registered in `core/Settings.cpp`, for example `TriangulationMethod`.
    pub name: String,
    /// `"bool"`, `"int"` or `"float"`.
    pub kind: String,
    /// Group, for example `"reconstruction"`.
    pub group: String,
    /// The value as written. Float settings are C++ `float`s promoted to `double`, so
    /// 0.9 appears as `0.89999997615814209`.
    pub value: Value,
}

/// One entry of the manifest's `missing` array: a table the dump could not produce.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Missing {
    /// Path of the table relative to the dump directory, for example
    /// `trial_0/points3d_filtered.csv`.
    pub table: String,
    /// Why it is missing.
    pub reason: String,
}

/// A dump's `manifest.json`.
#[derive(Debug, Clone)]
pub struct Manifest {
    path: PathBuf,
    json: Value,
}

impl Manifest {
    /// Reads a manifest file.
    pub fn try_read(path: impl AsRef<Path>) -> Result<Manifest> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| OracleError::file(path, format!("cannot read: {e}")))?;
        Manifest::try_parse(&text, path)
    }

    /// Reads a manifest file, panicking on error.
    #[track_caller]
    pub fn read(path: impl AsRef<Path>) -> Manifest {
        ok(Manifest::try_read(path))
    }

    /// Parses manifest text. `origin` is the path used in error messages.
    pub fn try_parse(text: &str, origin: impl AsRef<Path>) -> Result<Manifest> {
        let path = origin.as_ref();
        let json: Value = serde_json::from_str(text)
            .map_err(|e| OracleError::file(path, format!("invalid JSON: {e}")))?;
        if !json.is_object() {
            return Err(OracleError::file(path, "top level is not a JSON object"));
        }
        Ok(Manifest {
            path: path.to_path_buf(),
            json,
        })
    }

    /// The file the manifest was read from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The whole JSON document, for fields without a helper (`xmalab_version`,
    /// `opencv_version`, `platform`, `n_cameras`, ...).
    pub fn json(&self) -> &Value {
        &self.json
    }

    /// `format_version` (1 for the current format).
    pub fn format_version(&self) -> Option<i64> {
        self.json.get("format_version").and_then(Value::as_i64)
    }

    /// `frame_step`, the `--frame-step` the dump was run with.
    pub fn frame_step(&self) -> Option<i64> {
        self.json.get("frame_step").and_then(Value::as_i64)
    }

    /// Every setting, in file order.
    pub fn try_settings(&self) -> Result<Vec<Setting>> {
        let arr = self
            .json
            .get("settings")
            .and_then(Value::as_array)
            .ok_or_else(|| self.err("no 'settings' array"))?;
        arr.iter()
            .map(|s| {
                let field = |k: &str| {
                    s.get(k)
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .ok_or_else(|| self.err(format!("a setting has no string '{k}': {s}")))
                };
                Ok(Setting {
                    name: field("name")?,
                    kind: field("type")?,
                    group: field("group")?,
                    value: s
                        .get("value")
                        .cloned()
                        .ok_or_else(|| self.err(format!("a setting has no 'value': {s}")))?,
                })
            })
            .collect()
    }

    /// Every setting, panicking on a malformed manifest.
    #[track_caller]
    pub fn settings(&self) -> Vec<Setting> {
        ok(self.try_settings())
    }

    /// The setting of this name, or an error if it is absent.
    pub fn try_setting(&self, name: &str) -> Result<Setting> {
        self.try_settings()?
            .into_iter()
            .find(|s| s.name == name)
            .ok_or_else(|| self.err(format!("no setting '{name}'")))
    }

    /// A boolean setting. JSON `true`/`false` and the integers 0 and 1 are accepted.
    pub fn try_bool(&self, name: &str) -> Result<bool> {
        let s = self.try_setting(name)?;
        match &s.value {
            Value::Bool(b) => Ok(*b),
            v if v.as_i64() == Some(0) => Ok(false),
            v if v.as_i64() == Some(1) => Ok(true),
            v => Err(self.err(format!("setting '{name}' = {v} is not a bool"))),
        }
    }

    /// A boolean setting, panicking on error.
    #[track_caller]
    pub fn bool(&self, name: &str) -> bool {
        ok(self.try_bool(name))
    }

    /// An integer setting. A JSON boolean is not accepted.
    pub fn try_int(&self, name: &str) -> Result<i64> {
        let s = self.try_setting(name)?;
        s.value
            .as_i64()
            .ok_or_else(|| self.err(format!("setting '{name}' = {} is not an integer", s.value)))
    }

    /// An integer setting, panicking on error.
    #[track_caller]
    pub fn int(&self, name: &str) -> i64 {
        ok(self.try_int(name))
    }

    /// A numeric setting as `f64` (integers are converted).
    pub fn try_float(&self, name: &str) -> Result<f64> {
        let s = self.try_setting(name)?;
        s.value
            .as_f64()
            .ok_or_else(|| self.err(format!("setting '{name}' = {} is not a number", s.value)))
    }

    /// A numeric setting as `f64`, panicking on error.
    #[track_caller]
    pub fn float(&self, name: &str) -> f64 {
        ok(self.try_float(name))
    }

    /// The `missing` list: tables the dump could not produce, with reasons.
    pub fn try_missing(&self) -> Result<Vec<Missing>> {
        let Some(arr) = self.json.get("missing") else {
            return Ok(Vec::new());
        };
        let arr = arr
            .as_array()
            .ok_or_else(|| self.err("'missing' is not an array"))?;
        arr.iter()
            .map(|m| {
                let field = |k: &str| {
                    m.get(k)
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .ok_or_else(|| self.err(format!("a 'missing' entry has no '{k}': {m}")))
                };
                Ok(Missing {
                    table: field("table")?,
                    reason: field("reason")?,
                })
            })
            .collect()
    }

    /// The `missing` list, panicking on a malformed manifest.
    #[track_caller]
    pub fn missing(&self) -> Vec<Missing> {
        ok(self.try_missing())
    }

    /// True if `table` (relative to the dump directory) is listed under `missing`.
    #[track_caller]
    pub fn is_missing(&self, table: &str) -> bool {
        self.missing().iter().any(|m| m.table == table)
    }

    /// The `warnings` list (strings), empty if absent.
    pub fn warnings(&self) -> Vec<String> {
        self.json
            .get("warnings")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .map(|w| w.as_str().map_or_else(|| w.to_string(), str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn err(&self, message: impl Into<String>) -> OracleError {
        OracleError::file(&self.path, message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: &str = r#"{
      "format_version": 1, "frame_step": 10,
      "settings": [
        {"name": "TriangulationMethod", "type": "int", "group": "reconstruction", "value": 1},
        {"name": "OptimizeRigidBody", "type": "bool", "group": "rigid_body", "value": true},
        {"name": "BlobDetectorMinConvexity", "type": "float", "group": "blob_detector", "value": 0.89999997615814209}
      ],
      "warnings": ["w1"],
      "missing": [{"table": "calibration/bundle.csv", "reason": "not run"}]
    }"#;

    #[test]
    fn settings_and_missing() {
        let m = Manifest::try_parse(M, "manifest.json").unwrap();
        assert_eq!(m.format_version(), Some(1));
        assert_eq!(m.frame_step(), Some(10));
        assert_eq!(m.int("TriangulationMethod"), 1);
        assert_eq!(m.float("TriangulationMethod"), 1.0);
        assert!(m.bool("OptimizeRigidBody"));
        assert_eq!(m.float("BlobDetectorMinConvexity"), 0.9f32 as f64);
        assert!(m.try_int("OptimizeRigidBody").is_err());
        assert!(m.try_bool("BlobDetectorMinConvexity").is_err());
        let e = m.try_int("Nope").unwrap_err().to_string();
        assert!(e.contains("manifest.json") && e.contains("Nope"));
        assert_eq!(m.settings().len(), 3);
        assert_eq!(m.settings()[1].kind, "bool");
        assert!(m.is_missing("calibration/bundle.csv"));
        assert!(!m.is_missing("cameras.csv"));
        assert_eq!(m.missing()[0].reason, "not run");
        assert_eq!(m.warnings(), vec!["w1".to_string()]);
    }
}
