//! The settings that change results: command-line options, and the values a reference dump
//! records in its `manifest.json`.

use xma_engine::{Document, OpenOptions};
use xma_model::{AlgorithmSettings, SETTING_NAMES, SettingKind, SettingValue, SettingsPatch};
use xma_oracle::Manifest;

use crate::CliError;
use crate::args::{Opt, Parsed, number, switch};

/// The options shared by `info`, `recompute` and `export`.
pub const OPTIONS: [Opt; 6] = [
    Opt::value("triangulation"),
    Opt::value("optimize-rigid-body"),
    Opt::value("disable-rb-advanced"),
    Opt::value("filter-3d-points"),
    Opt::value("undistortion-neighbours"),
    Opt::value("default-marker-threshold"),
];

/// Help text for [`OPTIONS`].
pub const HELP: &str = "\
Settings that change results. XMALab keeps them per machine; here a project may store them
(an entry XMALab ignores, written on every save). Each is taken from the option if given,
else from the project, else XMALab's default:
      --triangulation <0-4>        Triangulation method (TriangulationMethod) [default: 1]
                                   0 Zisserman, 1 Zisserman incremental, 2 Zisserman (Matlab),
                                   3 Zisserman incremental (Matlab), 4 ray intersection
      --optimize-rigid-body <on|off>
                                   Refine rigid-body poses (OptimizeRigidBody) [default: on]
      --disable-rb-advanced <on|off>
                                   No pose from 2D rays or EPnP when a body has fewer than
                                   three 3D points (DisableRBComputeAdvanced) [default: off]
      --filter-3d-points <on|off>  Filter markers and refit bodies per frame
                                   (Filter3DPoints) [default: off]
      --undistortion-neighbours <n>
                                   Neighbours of the grid fit (LocalUndistortionNeighbours)
                                   [default: 12]
      --default-marker-threshold <n>
                                   Threshold offset of a marker stored without one
                                   (DefaultMarkerThreshold) [default: 8]";

/// The values given on the command line; `None` keeps the project's or XMALab's default.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SettingsArgs {
    pub triangulation: Option<u8>,
    pub optimize_rigid_body: Option<bool>,
    pub disable_rb_advanced: Option<bool>,
    pub filter_3d_points: Option<bool>,
    pub neighbours: Option<u32>,
    pub marker_threshold: Option<i32>,
}

impl SettingsArgs {
    /// Reads the options.
    pub fn from_parsed(p: &Parsed) -> Result<Self, CliError> {
        let triangulation = p
            .value("triangulation")
            .map(|v| number::<u8>(v, "--triangulation"))
            .transpose()?;
        if let Some(t) = triangulation.filter(|&t| t > 4) {
            return Err(CliError::usage(format!(
                "--triangulation: {t} is not a method (0 to 4)"
            )));
        }
        let neighbours = p
            .value("undistortion-neighbours")
            .map(|v| number::<u32>(v, "--undistortion-neighbours"))
            .transpose()?;
        if neighbours == Some(0) {
            return Err(CliError::usage(
                "--undistortion-neighbours: must be at least 1",
            ));
        }
        let marker_threshold = p
            .value("default-marker-threshold")
            .map(|v| number::<i32>(v, "--default-marker-threshold"))
            .transpose()?;
        let sw = |name: &str| {
            p.value(name)
                .map(|v| switch(v, &format!("--{name}")))
                .transpose()
        };
        Ok(Self {
            triangulation,
            optimize_rigid_body: sw("optimize-rigid-body")?,
            disable_rb_advanced: sw("disable-rb-advanced")?,
            filter_3d_points: sw("filter-3d-points")?,
            neighbours,
            marker_threshold,
        })
    }

    /// The values given, as overrides by setting name.
    #[must_use]
    pub fn patch(&self) -> SettingsPatch {
        let mut p = SettingsPatch::default();
        if let Some(v) = self.triangulation {
            p.set("TriangulationMethod", SettingValue::Int(v.into()));
        }
        if let Some(v) = self.optimize_rigid_body {
            p.set("OptimizeRigidBody", SettingValue::Bool(v));
        }
        if let Some(v) = self.disable_rb_advanced {
            p.set("DisableRBComputeAdvanced", SettingValue::Bool(v));
        }
        if let Some(v) = self.filter_3d_points {
            p.set("Filter3DPoints", SettingValue::Bool(v));
        }
        if let Some(v) = self.neighbours {
            p.set("LocalUndistortionNeighbours", SettingValue::Int(v.into()));
        }
        if let Some(v) = self.marker_threshold {
            p.set("DefaultMarkerThreshold", SettingValue::Int(v.into()));
        }
        p
    }

    /// How to open a project with these options, in the compatibility mode `compat`.
    #[must_use]
    pub fn open_options(&self, compat: xma_model::CppCompat) -> OpenOptions {
        OpenOptions {
            overrides: self.patch(),
            compat,
            ..OpenOptions::default()
        }
    }
}

/// The settings phase 1 computations read, with their values and where each came from, for
/// reports.
pub fn describe(doc: &Document) -> Vec<(&'static str, String, &'static str)> {
    let s = doc.settings();
    USED_SETTINGS
        .iter()
        .map(|&name| {
            let source = doc
                .extras()
                .settings_sources
                .iter()
                .find(|(n, _)| *n == name)
                .map_or("default", |(_, src)| src.label());
            (
                name,
                s.get(name).map_or_else(String::new, |v| v.to_string()),
                source,
            )
        })
        .collect()
}

/// The settings that phase 1 reads: the trial update, the rigid bodies, the grid fit and
/// loading. The others are stored and reported but read by nothing yet.
pub const USED_SETTINGS: [&str; 6] = [
    "TriangulationMethod",
    "OptimizeRigidBody",
    "DisableRBComputeAdvanced",
    "Filter3DPoints",
    "LocalUndistortionNeighbours",
    "DefaultMarkerThreshold",
];

/// Every result-affecting setting the model holds, from a dump's `manifest.json`, as the
/// C++ dump had them (the user's settings, captured before the dump changes any), as
/// overrides: the C++ ignores a project's settings entry, so the manifest's values win.
///
/// # Errors
/// When the manifest lacks one of them or holds a value of the wrong type or out of range.
pub fn manifest_settings(m: &Manifest) -> Result<SettingsPatch, String> {
    let mut patch = SettingsPatch::default();
    let mut check = AlgorithmSettings::default();
    for (name, kind) in SETTING_NAMES {
        let value = match kind {
            SettingKind::Bool => SettingValue::Bool(m.try_bool(name).map_err(|e| e.to_string())?),
            SettingKind::Int => SettingValue::Int(m.try_int(name).map_err(|e| e.to_string())?),
            SettingKind::Float => {
                SettingValue::Float(m.try_float(name).map_err(|e| e.to_string())?)
            }
        };
        check
            .set(name, value)
            .map_err(|e| format!("manifest {e}"))?;
        patch.set(name, value);
    }
    Ok(patch)
}
