//! The settings entry of a project file: the settings that change results, stored in the
//! project as an extra archive entry that XMALab 3 ignores (parity ledger A10).
//!
//! XMALab 3 keeps these settings per machine (`core/Settings.cpp`), so the same project can
//! give different numbers on different machines. Here they travel with the project, in the
//! entry [`SETTINGS_ENTRY`] at the root of the archive.
//!
//! Why an archive entry is safe (read in `ui/ProjectFileIO.cpp`):
//!
//! - loading (`ProjectFileIO::loadProject`, `:270-303`) unzips every entry into a temporary
//!   folder, then reads `project.xml` and `log.html` and only the files `project.xml`
//!   names; nothing lists the folder, so an entry it does not name is never opened. The
//!   folder is deleted when loading ends (`removeDir(tmpDir_path)`, `:303`);
//! - saving (`ProjectFileIO::saveProject`, `:114-268`) writes every file afresh into an
//!   empty temporary folder and zips that folder, so the entry is dropped by a save in
//!   XMALab 3, and nothing else changes.
//!
//! An attribute or element in `project.xml` was not used: `readProjectFile` reads elements
//! by name and would skip an unknown one, but the save rewrites `project.xml` and drops it
//! the same way, and a malformed one would affect the C++ reader; an archive entry cannot.
//!
//! Format: UTF-8 text in the style of a Qt INI file, `\n` line ends:
//!
//! ```text
//! ; Settings that change results, written by RustXMALab. XMALab 3 ignores this file.
//! [RustXMALab]
//! version=1
//! TriangulationMethod=1
//! OptimizeRigidBody=true
//! BlobDetectorMinConvexity=0.8999999761581421
//! ...
//! ```
//!
//! One `Name=value` line per setting of [`xma_model::SETTING_NAMES`], with the C++ setting
//! name; booleans `true` or `false`, integers in decimal, floats in Rust's shortest form
//! that reads back to the same `f64`. Lines starting with `;` or `#` and empty lines are
//! ignored. A setting that is absent keeps the value the caller supplies (XMALab's
//! default). Reading tolerates what it does not understand (an unknown name, a bad value,
//! another `version`) with a warning, since the entry is optional.

use xma_model::{AlgorithmSettings, SETTING_NAMES, SettingKind, SettingValue, SettingsPatch};

/// The archive entry's name.
pub const SETTINGS_ENTRY: &str = "RustXMALab_settings.ini";

/// The only format version written and understood.
pub const SETTINGS_VERSION: i64 = 1;

const SECTION: &str = "[RustXMALab]";

/// The entry's contents for a set of settings: every setting, in [`SETTING_NAMES`] order.
#[must_use]
pub fn write_settings(settings: &AlgorithmSettings) -> Vec<u8> {
    let mut s = String::from(
        "; Settings that change results, written by RustXMALab. XMALab 3 ignores this file.\n",
    );
    s.push_str(SECTION);
    s.push('\n');
    s.push_str(&format!("version={SETTINGS_VERSION}\n"));
    for (name, _) in SETTING_NAMES {
        if let Some(v) = settings.get(name) {
            s.push_str(&format!("{name}={v}\n"));
        }
    }
    s.into_bytes()
}

/// Parses the entry. Returns the settings it gives and a warning for each line not used.
#[must_use]
pub fn parse_settings(data: &[u8]) -> (SettingsPatch, Vec<String>) {
    let mut patch = SettingsPatch::default();
    let mut warnings = Vec::new();
    let warn = |w: &mut Vec<String>, line: usize, msg: String| {
        w.push(format!("{SETTINGS_ENTRY}, line {line}: {msg}; ignored"));
    };
    let text = match std::str::from_utf8(data) {
        Ok(t) => t,
        Err(e) => {
            warnings.push(format!(
                "{SETTINGS_ENTRY}: not valid UTF-8 ({e}); the entry is ignored"
            ));
            return (SettingsPatch::default(), warnings);
        }
    };
    let mut version = None;
    let mut in_section = false;
    for (i, raw) in text.lines().enumerate() {
        let n = i + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with(';') || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') {
            in_section = line == SECTION;
            if !in_section {
                warn(
                    &mut warnings,
                    n,
                    format!("section '{line}' is not {SECTION}"),
                );
            }
            continue;
        }
        if !in_section {
            warn(&mut warnings, n, format!("'{line}' is outside {SECTION}"));
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            warn(&mut warnings, n, format!("'{line}' is not name=value"));
            continue;
        };
        let (key, value) = (key.trim(), value.trim());
        if key == "version" {
            version = value.parse::<i64>().ok();
            continue;
        }
        let Some(kind) = xma_model::setting_kind(key) else {
            warn(&mut warnings, n, format!("unknown setting '{key}'"));
            continue;
        };
        let parsed = match kind {
            SettingKind::Bool => match value {
                "true" => Some(SettingValue::Bool(true)),
                "false" => Some(SettingValue::Bool(false)),
                _ => None,
            },
            SettingKind::Int => value.parse::<i64>().ok().map(SettingValue::Int),
            SettingKind::Float => value
                .parse::<f64>()
                .ok()
                .filter(|v| v.is_finite())
                .map(SettingValue::Float),
        };
        let Some(v) = parsed else {
            warn(
                &mut warnings,
                n,
                format!("{key}: '{value}' is not a {kind:?} value"),
            );
            continue;
        };
        // Range checks are those of the model.
        if let Err(e) = AlgorithmSettings::default().set(key, v) {
            warn(&mut warnings, n, e);
            continue;
        }
        patch.set(key, v);
    }
    if version != Some(SETTINGS_VERSION) {
        warnings.push(format!(
            "{SETTINGS_ENTRY}: version {} is not {SETTINGS_VERSION}; the entry is ignored",
            version.map_or_else(|| "absent".to_owned(), |v| v.to_string())
        ));
        return (SettingsPatch::default(), warnings);
    }
    (patch, warnings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut s = AlgorithmSettings {
            triangulation_method: 4,
            optimize_rigid_body: false,
            default_marker_threshold: 5,
            ..AlgorithmSettings::default()
        };
        s.blob_detector.min_area = 0.1 + 0.2;
        let (patch, warnings) = parse_settings(&write_settings(&s));
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(patch.values.len(), SETTING_NAMES.len());
        let mut back = AlgorithmSettings::default();
        patch.apply(&mut back).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn tolerates_what_it_does_not_understand() {
        let text = "; c\n[RustXMALab]\nversion=1\n\nTriangulationMethod = 3\nNoSuch=1\n\
                    OptimizeRigidBody=yes\nTriangulationMethod2\nLocalUndistortionNeighbours=0\n";
        let (patch, warnings) = parse_settings(text.as_bytes());
        assert_eq!(
            patch.values,
            vec![("TriangulationMethod".to_owned(), SettingValue::Int(3))]
        );
        assert_eq!(warnings.len(), 4, "{warnings:?}");
        let (patch, warnings) = parse_settings(b"[RustXMALab]\nversion=2\nFilter3DPoints=true\n");
        assert!(patch.is_empty());
        assert_eq!(warnings.len(), 1);
        let (patch, _) = parse_settings(b"[Other]\nversion=1\n");
        assert!(patch.is_empty());
    }

    /// The five settings of merge-aware tracking (removed on 26 November 2026, parity
    /// ledger A226) are still read without a warning and written back as they were read, in
    /// their place, so that a project file's settings entry is unchanged byte for byte.
    #[test]
    fn retired_merge_settings_read_and_written_as_before() {
        let defaults = String::from_utf8(write_settings(&AlgorithmSettings::default())).unwrap();
        assert!(
            defaults.ends_with(
                "MergeEpipolarThreshold=3.0\nMergeAuditThreshold=4.0\nMergeMaxPendingFrames=10\n\
                 MergeMaxHoldFrames=10\nMergeMaxOverlapFrames=20\n"
            ),
            "{defaults}"
        );
        let text = defaults
            .replace(
                "MergeEpipolarThreshold=3.0\n",
                "MergeEpipolarThreshold=60.5\n",
            )
            .replace("MergeMaxHoldFrames=10\n", "MergeMaxHoldFrames=3\n");
        let (patch, warnings) = parse_settings(text.as_bytes());
        assert!(warnings.is_empty(), "{warnings:?}");
        let mut s = AlgorithmSettings::default();
        patch.apply(&mut s).unwrap();
        assert_eq!(write_settings(&s), text.as_bytes());
        assert_eq!(
            s.get("MergeEpipolarThreshold"),
            Some(SettingValue::Float(60.5))
        );
        // The checks are those of before: an out-of-range value warns and is ignored.
        let (patch, warnings) =
            parse_settings(b"[RustXMALab]\nversion=1\nMergeMaxPendingFrames=-1\n");
        assert!(patch.is_empty());
        assert_eq!(warnings.len(), 1, "{warnings:?}");
    }
}
