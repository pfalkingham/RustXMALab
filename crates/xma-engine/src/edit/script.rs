//! Edit scripts: a text file of operations, one per line, named and laid out as the
//! variants of [`Edit`] (`oracle/FORMAT.md`, "Edit scripts (version 4)").
//!
//! Not a port: the format was made for the C++ editing reference (`XMALab --edit-oracle`),
//! which runs the same scripts through XMALab's own code. [`parse_script`] reads a script
//! into [`ScriptLine`]s; `xma edit` applies them, and `xma parity` compares the result with
//! the reference at every `Dump`.
//!
//! The format: one operation per line, its name, then `key=value` fields separated by
//! single spaces; empty lines and lines starting with `#` are skipped. Values are
//! percent-encoded. Indices are 0-based except `frame` of `SetStartFrame` and `SetEndFrame`
//! (1-based, as the model stores them). Lists are comma-separated; `cameras` is `all` or a
//! camera index; booleans `true` or `false`; statuses, methods and interpolation methods
//! their integer codes; colours `#rrggbb`; a frame range `first` and `last`, inclusive. A
//! field that is an `Option` in [`Edit`] is left out for `None`. Paths are relative to the
//! script's directory unless absolute.
//!
//! Besides the edits: `Dump` (a state dump in the reference) and `Save` (the project saved),
//! and the two operations that need images, `PlacePoint` and `RefinePoints`
//! ([`crate::tracking`]), which are parsed here and applied by whoever has the images.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use xma_model::{
    DetectionMethod, Interpolation, MarkerStatus, Rgb, SettingKind, SettingValue, SettingsPatch,
    setting_kind,
};

use super::{
    BodyErrorKind, CameraSel, Edit, ErrorTarget, FrameRange, Import2dOptions, MarkerProperties,
    TrialProperties, VirtualPointInput,
};
use crate::tracking::{Placement, RefineWhich};

/// One operation of a script.
#[derive(Clone, Debug, PartialEq)]
pub enum ScriptOp {
    /// An edit, applied with [`crate::Document::apply`].
    Edit(Edit),
    /// `Dump name=..`: the reference writes its state dump into `<out>/<name>/` here.
    Dump { name: String },
    /// `Save name=..`: the reference saves the project as `<out>/<name>` here.
    Save { name: String },
    /// `PlacePoint`: a click in a camera view ([`crate::tracking::place_point`]). `active` is
    /// the active marker before the click (where the C++ search of "add next" starts).
    PlacePoint {
        trial: usize,
        camera: usize,
        frame: usize,
        x: f64,
        y: f64,
        placement: Placement,
        active: Option<usize>,
    },
    /// `RefinePoints`: the polynomial refinement ([`crate::tracking::refine_points`]).
    RefinePoints {
        trial: usize,
        markers: Vec<usize>,
        frames: FrameRange,
        which: RefineWhich,
    },
}

impl ScriptOp {
    /// The trial XMALab's handler for this operation works on as the active trial: the
    /// editing reference first makes that trial active as the trial selector does
    /// (`oracle/cpp/DEVIATIONS.md`, items 46 and 48; `EditOracleDriver::activate`), for the
    /// handlers that read the active trial. `None` for the others, `Dump` and `Save`; "Apply
    /// to trials" acts from the Default trial (the caller finds it).
    #[must_use]
    pub fn handler_trial(&self) -> Option<usize> {
        match self {
            ScriptOp::Edit(edit) => match edit {
                Edit::ResetAboveError { trial, .. }
                | Edit::SwapMarkerData { trial, .. }
                | Edit::RemoveMarkers { trial, .. }
                | Edit::SetMarkerCount { trial, .. }
                | Edit::SetRigidBodyCount { trial, .. }
                | Edit::CreateRigidBody { trial, .. }
                | Edit::MoveMarkersToBody { trial, .. }
                | Edit::SetEventFrames { trial, .. }
                | Edit::SetTrialProperties { trial, .. }
                | Edit::Import2dPoints { trial, .. }
                | Edit::ImportMarkersCsv { trial, .. }
                | Edit::ImportMarkersFromProject { trial, .. }
                | Edit::CopyMarkersFromTrial { trial, .. } => Some(*trial),
                _ => None,
            },
            ScriptOp::PlacePoint { trial, .. } | ScriptOp::RefinePoints { trial, .. } => {
                Some(*trial)
            }
            ScriptOp::Dump { .. } | ScriptOp::Save { .. } => None,
        }
    }

    /// The operation's name as the script writes it.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            ScriptOp::Edit(edit) => edit_name(edit),
            ScriptOp::Dump { .. } => "Dump",
            ScriptOp::Save { .. } => "Save",
            ScriptOp::PlacePoint { .. } => "PlacePoint",
            ScriptOp::RefinePoints { .. } => "RefinePoints",
        }
    }
}

/// One line of a script that holds an operation.
#[derive(Clone, Debug, PartialEq)]
pub struct ScriptLine {
    /// The line number in the file, from 1.
    pub line: usize,
    /// The line as written.
    pub text: String,
    pub op: ScriptOp,
}

/// A line that cannot be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScriptError {
    /// The line number, from 1.
    pub line: usize,
    pub message: String,
}

impl fmt::Display for ScriptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for ScriptError {}

/// Reads a script. `base` is the directory relative paths are taken from (the script's).
///
/// # Errors
/// The first line that cannot be read: an unknown operation, a missing, unknown or
/// repeated field, or a value that does not parse.
pub fn parse_script(text: &str, base: &Path) -> Result<Vec<ScriptLine>, ScriptError> {
    let mut out = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim_end_matches('\r');
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let op = parse_line(line, base).map_err(|message| ScriptError {
            line: i + 1,
            message,
        })?;
        out.push(ScriptLine {
            line: i + 1,
            text: line.to_owned(),
            op,
        });
    }
    Ok(out)
}

/// Decodes `%xx` escapes.
///
/// # Errors
/// A `%` not followed by two hexadecimal digits, or bytes that are not UTF-8.
pub fn percent_decode(s: &str) -> Result<String, String> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = s
                .get(i + 1..i + 3)
                .and_then(|h| u8::from_str_radix(h, 16).ok())
                .ok_or_else(|| format!("bad escape in '{s}'"))?;
            out.push(hex);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).map_err(|_| format!("'{s}' is not UTF-8 once decoded"))
}

/// The fields of one line, still encoded, with the ones read so far.
struct Fields<'a> {
    op: &'a str,
    map: BTreeMap<&'a str, &'a str>,
    base: &'a Path,
}

impl<'a> Fields<'a> {
    fn raw(&mut self, key: &str) -> Result<&'a str, String> {
        self.map
            .remove(key)
            .ok_or_else(|| format!("{}: field '{key}' missing", self.op))
    }

    fn has(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }

    fn text(&mut self, key: &str) -> Result<String, String> {
        percent_decode(self.raw(key)?)
    }

    fn parse<T: std::str::FromStr>(&mut self, key: &str) -> Result<T, String> {
        let v = self.text(key)?;
        v.parse()
            .map_err(|_| format!("{}: {key}={v} does not parse", self.op))
    }

    fn opt<T: std::str::FromStr>(&mut self, key: &str) -> Result<Option<T>, String> {
        if self.has(key) {
            self.parse(key).map(Some)
        } else {
            Ok(None)
        }
    }

    fn usize(&mut self, key: &str) -> Result<usize, String> {
        self.parse(key)
    }

    fn f64(&mut self, key: &str) -> Result<f64, String> {
        self.parse(key)
    }

    fn bool(&mut self, key: &str) -> Result<bool, String> {
        match self.text(key)?.as_str() {
            "true" => Ok(true),
            "false" => Ok(false),
            v => Err(format!("{}: {key}={v} is not true or false", self.op)),
        }
    }

    fn opt_bool(&mut self, key: &str) -> Result<Option<bool>, String> {
        if self.has(key) {
            self.bool(key).map(Some)
        } else {
            Ok(None)
        }
    }

    /// A comma-separated list, each element decoded.
    fn list(&mut self, key: &str) -> Result<Vec<String>, String> {
        let raw = self.raw(key)?;
        if raw.is_empty() {
            return Ok(Vec::new());
        }
        raw.split(',').map(percent_decode).collect()
    }

    fn indices(&mut self, key: &str) -> Result<Vec<usize>, String> {
        let op = self.op;
        self.list(key)?
            .iter()
            .map(|v| {
                v.parse()
                    .map_err(|_| format!("{op}: {key}: '{v}' is not an index"))
            })
            .collect()
    }

    fn path(&mut self, key: &str) -> Result<PathBuf, String> {
        let p = PathBuf::from(self.text(key)?);
        Ok(self.resolve(p))
    }

    fn resolve(&self, p: PathBuf) -> PathBuf {
        if p.is_absolute() {
            p
        } else {
            self.base.join(p)
        }
    }

    fn cameras(&mut self, key: &str) -> Result<CameraSel, String> {
        let v = self.text(key)?;
        if v == "all" {
            Ok(CameraSel::All)
        } else {
            v.parse()
                .map(CameraSel::One)
                .map_err(|_| format!("{}: {key}={v} is neither 'all' nor a camera", self.op))
        }
    }

    fn frames(&mut self) -> Result<FrameRange, String> {
        Ok(FrameRange {
            first: self.usize("first")?,
            last: self.usize("last")?,
        })
    }

    fn code<T>(&mut self, key: &str, from: fn(i32) -> Option<T>) -> Result<T, String> {
        let v: i32 = self.parse(key)?;
        from(v).ok_or_else(|| format!("{}: {key}={v} is not a known code", self.op))
    }

    fn color(&mut self, key: &str) -> Result<Rgb, String> {
        let v = self.text(key)?;
        Rgb::from_hex(&v).ok_or_else(|| format!("{}: {key}={v} is not #rrggbb", self.op))
    }

    /// No field may be left over.
    fn done(self) -> Result<(), String> {
        match self.map.keys().next() {
            None => Ok(()),
            Some(k) => Err(format!("{}: unknown field '{k}'", self.op)),
        }
    }
}

/// Reads one operation line.
///
/// # Errors
/// As [`parse_script`].
pub fn parse_line(line: &str, base: &Path) -> Result<ScriptOp, String> {
    let mut parts = line.split(' ').filter(|p| !p.is_empty());
    let op = parts.next().ok_or("empty line")?;
    let mut map = BTreeMap::new();
    let mut order = Vec::new();
    for p in parts {
        let (k, v) = p
            .split_once('=')
            .ok_or_else(|| format!("{op}: '{p}' is not key=value"))?;
        if map.insert(k, v).is_some() {
            return Err(format!("{op}: field '{k}' given twice"));
        }
        order.push(k);
    }
    let mut f = Fields { op, map, base };
    let edit = match op {
        "Dump" => {
            let name = f.text("name")?;
            f.done()?;
            return Ok(ScriptOp::Dump { name });
        }
        "Save" => {
            let name = f.text("name")?;
            f.done()?;
            return Ok(ScriptOp::Save { name });
        }
        "PlacePoint" => {
            let trial = f.usize("trial")?;
            let camera = f.usize("camera")?;
            let frame = f.usize("frame")?;
            let x = f.f64("x")?;
            let y = f.f64("y")?;
            let active = f.opt("active")?;
            let placement = match f.text("placement")?.as_str() {
                "add_next" => Placement::AddNext { active },
                "detect" => Placement::Detect(f.usize("marker")?),
                "manual" => Placement::Manual(f.usize("marker")?),
                v => return Err(format!("PlacePoint: placement={v} is not known")),
            };
            f.done()?;
            return Ok(ScriptOp::PlacePoint {
                trial,
                camera,
                frame,
                x,
                y,
                placement,
                active,
            });
        }
        "RefinePoints" => {
            let trial = f.usize("trial")?;
            let markers = f.indices("markers")?;
            let frames = f.frames()?;
            let which = match f.usize("which")? {
                0 => RefineWhich::SetAndTracked,
                1 => RefineWhich::TrackedOnly,
                2 => RefineWhich::All,
                v => return Err(format!("RefinePoints: which={v} is not 0, 1 or 2")),
            };
            f.done()?;
            return Ok(ScriptOp::RefinePoints {
                trial,
                markers,
                frames,
                which,
            });
        }
        "SetPoint" => Edit::SetPoint {
            trial: f.usize("trial")?,
            marker: f.usize("marker")?,
            camera: f.usize("camera")?,
            frame: f.usize("frame")?,
            x: f.f64("x")?,
            y: f.f64("y")?,
            status: f.code("status", MarkerStatus::from_code)?,
        },
        "SetDetectedPoint" => Edit::SetDetectedPoint {
            trial: f.usize("trial")?,
            marker: f.usize("marker")?,
            camera: f.usize("camera")?,
            frame: f.usize("frame")?,
            x: f.f64("x")?,
            y: f.f64("y")?,
            size: f.f64("size")?,
            status: f.code("status", MarkerStatus::from_code)?,
        },
        "ResetFrames" => Edit::ResetFrames {
            trial: f.usize("trial")?,
            markers: f.indices("markers")?,
            cameras: f.cameras("cameras")?,
            frames: f.frames()?,
            untrackable: f.bool("untrackable")?,
        },
        "ResetAboveError" => {
            let trial = f.usize("trial")?;
            let threshold = f.f64("threshold")?;
            let target = match f.text("target")?.as_str() {
                "marker" => ErrorTarget::Marker {
                    marker: f.usize("marker")?,
                    cameras: f.cameras("cameras")?,
                },
                "body" => ErrorTarget::Body {
                    body: f.usize("body")?,
                    kind: f.usize("kind")?,
                    error: match f.opt::<usize>("error")? {
                        None | Some(0) => BodyErrorKind::Error3d,
                        Some(1) => BodyErrorKind::Error2d,
                        Some(v) => return Err(format!("ResetAboveError: error={v} is not 0 or 1")),
                    },
                },
                v => return Err(format!("ResetAboveError: target={v} is not known")),
            };
            Edit::ResetAboveError {
                trial,
                target,
                threshold,
            }
        }
        "SwapMarkerData" => Edit::SwapMarkerData {
            trial: f.usize("trial")?,
            marker: f.usize("marker")?,
            other: f.usize("other")?,
            cameras: f.cameras("cameras")?,
            frames: f.frames()?,
        },
        "Interpolate" => Edit::Interpolate {
            trial: f.usize("trial")?,
            markers: f.indices("markers")?,
        },
        "SetInterpolation" => Edit::SetInterpolation {
            trial: f.usize("trial")?,
            marker: f.usize("marker")?,
            frames: f.frames()?,
            method: f.code("method", Interpolation::from_code)?,
        },
        "SetMarkerDescription" => Edit::SetMarkerDescription {
            trial: f.usize("trial")?,
            marker: f.usize("marker")?,
            description: f.text("description")?,
        },
        "SetMarkerMethod" => Edit::SetMarkerMethod {
            trial: f.usize("trial")?,
            markers: f.indices("markers")?,
            method: f.code("method", DetectionMethod::from_code)?,
        },
        "SetMarkerProperties" => {
            let trial = f.usize("trial")?;
            let marker = f.usize("marker")?;
            let method = if f.has("method") {
                Some(f.code("method", DetectionMethod::from_code)?)
            } else {
                None
            };
            Edit::SetMarkerProperties {
                trial,
                marker,
                properties: MarkerProperties {
                    method,
                    threshold_offset: f.opt("threshold_offset")?,
                    size_override: f.opt("size_override")?,
                    max_penalty: f.opt("max_penalty")?,
                },
            }
        }
        "AddMarker" => Edit::AddMarker {
            trial: f.usize("trial")?,
        },
        "RemoveMarkers" => Edit::RemoveMarkers {
            trial: f.usize("trial")?,
            markers: f.indices("markers")?,
        },
        "SetMarkerCount" => Edit::SetMarkerCount {
            trial: f.usize("trial")?,
            count: f.usize("count")?,
        },
        "SetRigidBodyCount" => Edit::SetRigidBodyCount {
            trial: f.usize("trial")?,
            count: f.usize("count")?,
        },
        "CreateRigidBody" => Edit::CreateRigidBody {
            trial: f.usize("trial")?,
            markers: f.indices("markers")?,
        },
        "MoveMarkersToBody" => Edit::MoveMarkersToBody {
            trial: f.usize("trial")?,
            markers: f.indices("markers")?,
            body: f.opt("body")?,
        },
        "SetBodyDescription" => Edit::SetBodyDescription {
            trial: f.usize("trial")?,
            body: f.usize("body")?,
            description: f.text("description")?,
        },
        "SetBodyCorrespondences" => Edit::SetBodyCorrespondences {
            trial: f.usize("trial")?,
            body: f.usize("body")?,
            markers: f.indices("markers")?,
        },
        "SetBodyReferencesFromFile" => Edit::SetBodyReferencesFromFile {
            trial: f.usize("trial")?,
            body: f.usize("body")?,
            path: f.path("path")?,
        },
        "SetBodyReferencesFromFrame" => Edit::SetBodyReferencesFromFrame {
            trial: f.usize("trial")?,
            body: f.usize("body")?,
            frame: f.usize("frame")?,
        },
        "ResetBodyReferences" => Edit::ResetBodyReferences {
            trial: f.usize("trial")?,
            body: f.usize("body")?,
        },
        "SetBodyOptimized" => Edit::SetBodyOptimized {
            trial: f.usize("trial")?,
            body: f.usize("body")?,
            optimized: f.bool("optimized")?,
        },
        "UpdateBodyOptimized" => Edit::UpdateBodyOptimized {
            trial: f.usize("trial")?,
            body: f.usize("body")?,
        },
        "SetBodyVisible" => Edit::SetBodyVisible {
            trial: f.usize("trial")?,
            body: f.usize("body")?,
            visible: f.bool("visible")?,
        },
        "SetBodyColor" => Edit::SetBodyColor {
            trial: f.usize("trial")?,
            body: f.usize("body")?,
            color: f.color("color")?,
        },
        "SetBodyMesh" => Edit::SetBodyMesh {
            trial: f.usize("trial")?,
            body: f.usize("body")?,
            path: if f.has("path") {
                Some(f.path("path")?)
            } else {
                None
            },
        },
        "SetBodyDrawMesh" => Edit::SetBodyDrawMesh {
            trial: f.usize("trial")?,
            body: f.usize("body")?,
            draw: f.bool("draw")?,
        },
        "SetBodyMeshScale" => Edit::SetBodyMeshScale {
            trial: f.usize("trial")?,
            body: f.usize("body")?,
            scale: f.f64("scale")?,
        },
        "SetBodyCutoffOverride" => Edit::SetBodyCutoffOverride {
            trial: f.usize("trial")?,
            body: f.usize("body")?,
            enabled: f.bool("enabled")?,
            cutoff_frequency: f.f64("cutoff_frequency")?,
        },
        "AddVirtualPoint" => {
            let trial = f.usize("trial")?;
            let body = f.usize("body")?;
            let name = f.text("name")?;
            let reference_file = f.path("reference_file")?;
            let input = match f.text("input")?.as_str() {
                "body" => VirtualPointInput::Body {
                    body: f.usize("input_body")?,
                    driver_file: if f.has("driver_file") {
                        Some(f.path("driver_file")?)
                    } else {
                        None
                    },
                },
                "coordinates" => VirtualPointInput::Coordinates {
                    file: f.path("file")?,
                },
                v => return Err(format!("AddVirtualPoint: input={v} is not known")),
            };
            Edit::AddVirtualPoint {
                trial,
                body,
                name,
                reference_file,
                input,
            }
        }
        "ClearVirtualPoints" => Edit::ClearVirtualPoints {
            trial: f.usize("trial")?,
            body: f.usize("body")?,
        },
        "RecomputeBodies" => Edit::RecomputeBodies {
            trial: f.usize("trial")?,
        },
        "AddEvent" => Edit::AddEvent {
            trial: f.usize("trial")?,
            name: f.text("name")?,
        },
        "RemoveEvent" => Edit::RemoveEvent {
            trial: f.usize("trial")?,
            event: f.usize("event")?,
        },
        "SetEventFrames" => Edit::SetEventFrames {
            trial: f.usize("trial")?,
            frames: f.frames()?,
            on: f.bool("on")?,
        },
        "SetEventColor" => Edit::SetEventColor {
            trial: f.usize("trial")?,
            event: f.usize("event")?,
            color: f.color("color")?,
        },
        "SetEventDraw" => Edit::SetEventDraw {
            trial: f.usize("trial")?,
            event: f.usize("event")?,
            draw: f.bool("draw")?,
        },
        "SetTrialProperties" => Edit::SetTrialProperties {
            trial: f.usize("trial")?,
            properties: TrialProperties {
                recording_speed: f.opt("recording_speed")?,
                cutoff_frequency: f.opt("cutoff_frequency")?,
                interpolate_3d: f.opt_bool("interpolate_3d")?,
                reference_calibration: f.opt("reference_calibration")?,
                force_update: f.opt_bool("force_update")?.unwrap_or(false),
            },
        },
        "SetStartFrame" => Edit::SetStartFrame {
            trial: f.usize("trial")?,
            frame: f.usize("frame")?,
        },
        "SetEndFrame" => Edit::SetEndFrame {
            trial: f.usize("trial")?,
            frame: f.usize("frame")?,
        },
        "DeleteTrials" => Edit::DeleteTrials {
            trials: f.indices("trials")?,
        },
        "UnlinkFromDefault" => Edit::UnlinkFromDefault {
            trial: f.usize("trial")?,
        },
        "ApplyDefaultToTrials" => Edit::ApplyDefaultToTrials,
        "Import2dPoints" => {
            let trial = f.usize("trial")?;
            let files = f
                .list("files")?
                .into_iter()
                .map(|p| f.resolve(PathBuf::from(p)))
                .collect();
            Edit::Import2dPoints {
                trial,
                files,
                options: Import2dOptions {
                    distorted: f.bool("distorted")?,
                    count_from_one: f.bool("count_from_one")?,
                    y_up: f.bool("y_up")?,
                    header: f.bool("header")?,
                    offset_columns: f.bool("offset_columns")?,
                    status_set: f.bool("status_set")?,
                },
            }
        }
        "ImportMarkersCsv" => Edit::ImportMarkersCsv {
            trial: f.usize("trial")?,
            path: f.path("path")?,
            update_existing: f.bool("update_existing")?,
        },
        "ImportMarkersFromProject" => Edit::ImportMarkersFromProject {
            trial: f.usize("trial")?,
            path: f.path("path")?,
            source_trial: f.text("source_trial")?,
        },
        "CopyMarkersFromTrial" => Edit::CopyMarkersFromTrial {
            trial: f.usize("trial")?,
            source: f.usize("source")?,
        },
        "RenameMarkersFromCsv" => Edit::RenameMarkersFromCsv {
            trial: f.usize("trial")?,
            path: f.path("path")?,
        },
        "ImportTrials" => Edit::ImportTrials {
            path: f.path("path")?,
            trials: f.list("trials")?,
            replace_default: f.bool("replace_default")?,
        },
        "SetSettings" => {
            let mut patch = SettingsPatch::default();
            for name in order {
                let v = f.text(name)?;
                let value = match setting_kind(name) {
                    None => return Err(format!("SetSettings: unknown setting '{name}'")),
                    Some(SettingKind::Bool) => SettingValue::Bool(match v.as_str() {
                        "true" => true,
                        "false" => false,
                        _ => return Err(format!("SetSettings: {name}={v} is not true or false")),
                    }),
                    Some(SettingKind::Int) => SettingValue::Int(
                        v.parse()
                            .map_err(|_| format!("SetSettings: {name}={v} is not an integer"))?,
                    ),
                    // XMALab holds these as `float` (QSettings, `Settings::getFloat`).
                    Some(SettingKind::Float) => SettingValue::Float(f64::from(
                        v.parse::<f32>()
                            .map_err(|_| format!("SetSettings: {name}={v} is not a number"))?,
                    )),
                };
                patch.set(name, value);
            }
            Edit::SetSettings(patch)
        }
        other => return Err(format!("unknown operation '{other}'")),
    };
    f.done()?;
    let name = edit_name(&edit);
    debug_assert_eq!(name, op);
    Ok(ScriptOp::Edit(edit))
}

/// The variant's name, as the script writes it.
#[must_use]
pub fn edit_name(edit: &Edit) -> &'static str {
    match edit {
        Edit::SetPoint { .. } => "SetPoint",
        Edit::SetDetectedPoint { .. } => "SetDetectedPoint",
        Edit::ResetFrames { .. } => "ResetFrames",
        Edit::ResetAboveError { .. } => "ResetAboveError",
        Edit::SwapMarkerData { .. } => "SwapMarkerData",
        Edit::Interpolate { .. } => "Interpolate",
        Edit::SetInterpolation { .. } => "SetInterpolation",
        Edit::SetMarkerDescription { .. } => "SetMarkerDescription",
        Edit::SetMarkerMethod { .. } => "SetMarkerMethod",
        Edit::SetMarkerProperties { .. } => "SetMarkerProperties",
        Edit::AddMarker { .. } => "AddMarker",
        Edit::RemoveMarkers { .. } => "RemoveMarkers",
        Edit::SetMarkerCount { .. } => "SetMarkerCount",
        Edit::SetRigidBodyCount { .. } => "SetRigidBodyCount",
        Edit::CreateRigidBody { .. } => "CreateRigidBody",
        Edit::MoveMarkersToBody { .. } => "MoveMarkersToBody",
        Edit::SetBodyDescription { .. } => "SetBodyDescription",
        Edit::SetBodyCorrespondences { .. } => "SetBodyCorrespondences",
        Edit::SetBodyReferencesFromFile { .. } => "SetBodyReferencesFromFile",
        Edit::SetBodyReferencesFromFrame { .. } => "SetBodyReferencesFromFrame",
        Edit::ResetBodyReferences { .. } => "ResetBodyReferences",
        Edit::SetBodyOptimized { .. } => "SetBodyOptimized",
        Edit::UpdateBodyOptimized { .. } => "UpdateBodyOptimized",
        Edit::SetBodyVisible { .. } => "SetBodyVisible",
        Edit::SetBodyColor { .. } => "SetBodyColor",
        Edit::SetBodyMesh { .. } => "SetBodyMesh",
        Edit::SetBodyDrawMesh { .. } => "SetBodyDrawMesh",
        Edit::SetBodyMeshScale { .. } => "SetBodyMeshScale",
        Edit::SetBodyCutoffOverride { .. } => "SetBodyCutoffOverride",
        Edit::AddVirtualPoint { .. } => "AddVirtualPoint",
        Edit::ClearVirtualPoints { .. } => "ClearVirtualPoints",
        Edit::RecomputeBodies { .. } => "RecomputeBodies",
        Edit::AddEvent { .. } => "AddEvent",
        Edit::RemoveEvent { .. } => "RemoveEvent",
        Edit::SetEventFrames { .. } => "SetEventFrames",
        Edit::SetEventColor { .. } => "SetEventColor",
        Edit::SetEventDraw { .. } => "SetEventDraw",
        Edit::SetTrialProperties { .. } => "SetTrialProperties",
        Edit::SetStartFrame { .. } => "SetStartFrame",
        Edit::SetEndFrame { .. } => "SetEndFrame",
        Edit::DeleteTrials { .. } => "DeleteTrials",
        Edit::UnlinkFromDefault { .. } => "UnlinkFromDefault",
        Edit::ApplyDefaultToTrials => "ApplyDefaultToTrials",
        Edit::Import2dPoints { .. } => "Import2dPoints",
        Edit::ImportMarkersCsv { .. } => "ImportMarkersCsv",
        Edit::ImportMarkersFromProject { .. } => "ImportMarkersFromProject",
        Edit::CopyMarkersFromTrial { .. } => "CopyMarkersFromTrial",
        Edit::RenameMarkersFromCsv { .. } => "RenameMarkersFromCsv",
        Edit::ImportTrials { .. } => "ImportTrials",
        Edit::SetSettings(_) => "SetSettings",
        Edit::NewTrial(_) => "NewTrial",
        Edit::AddDefaultTrial => "AddDefaultTrial",
        Edit::ChangeTrialData { .. } => "ChangeTrialData",
        Edit::NewTrialFromPortal { .. } => "NewTrialFromPortal",
        Edit::ChangeTrialDataFromPortal { .. } => "ChangeTrialDataFromPortal",
        Edit::SetPortalMetadata { .. } => "SetPortalMetadata",
        Edit::Calibration(e) => e.name(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(line: &str) -> ScriptOp {
        parse_line(line, Path::new("/base")).unwrap_or_else(|e| panic!("{line}: {e}"))
    }

    #[test]
    fn decodes_and_resolves() {
        assert_eq!(
            percent_decode("Renamed%20marker,%20with%20comma").as_deref(),
            Ok("Renamed marker, with comma")
        );
        assert!(percent_decode("bad%2").is_err());
        assert_eq!(
            one("SetMarkerDescription trial=0 marker=0 description=a%20b"),
            ScriptOp::Edit(Edit::SetMarkerDescription {
                trial: 0,
                marker: 0,
                description: "a b".to_owned()
            })
        );
        assert_eq!(
            one("SetBodyReferencesFromFile trial=0 body=0 path=../files/r.csv"),
            ScriptOp::Edit(Edit::SetBodyReferencesFromFile {
                trial: 0,
                body: 0,
                path: PathBuf::from("/base/../files/r.csv")
            })
        );
        assert_eq!(
            one("ResetAboveError trial=0 target=body body=1 kind=2 error=1 threshold=0.3"),
            ScriptOp::Edit(Edit::ResetAboveError {
                trial: 0,
                target: ErrorTarget::Body {
                    body: 1,
                    kind: 2,
                    error: BodyErrorKind::Error2d
                },
                threshold: 0.3
            })
        );
        assert_eq!(
            one("PlacePoint trial=0 camera=1 frame=31 x=573.2 y=426.1 placement=detect marker=2"),
            ScriptOp::PlacePoint {
                trial: 0,
                camera: 1,
                frame: 31,
                x: 573.2,
                y: 426.1,
                placement: Placement::Detect(2),
                active: None
            }
        );
        match one(
            "SetSettings MaximumReprojectionError=3.5 TriangulationMethod=2 Filter3DPoints=true",
        ) {
            ScriptOp::Edit(Edit::SetSettings(p)) => {
                let names: Vec<&str> = p.values.iter().map(|(n, _)| n.as_str()).collect();
                assert_eq!(
                    names,
                    [
                        "MaximumReprojectionError",
                        "TriangulationMethod",
                        "Filter3DPoints"
                    ]
                );
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn refuses_bad_lines() {
        for bad in [
            "Nonsense trial=0",
            "SetPoint trial=0",
            "AddMarker trial=0 extra=1",
            "AddMarker trial=0 trial=1",
            "SetPoint trial=0 marker=0 camera=0 frame=0 x=1 y=1 status=7",
            "SetSettings NoSuchSetting=1",
            "ResetFrames trial=0 markers=1 cameras=x first=0 last=1 untrackable=false",
        ] {
            assert!(parse_line(bad, Path::new("/")).is_err(), "{bad}");
        }
        let script = "# start: x\n\nDump name=a\nAddMarker trial=0\n";
        let lines = parse_script(script, Path::new("/")).expect("parses");
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[1].line, 4);
        assert_eq!(lines[1].op.name(), "AddMarker");
        assert_eq!(lines[1].op.handler_trial(), None);
    }
}
