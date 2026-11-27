//! Settings that change computed results.
//!
//! XMALab 3 keeps these in a per-machine settings file (`core/Settings.cpp`), so the same
//! project can give different numbers on different machines. Here they are part of the
//! project. Defaults are the C++ defaults at commit `2967a7e`; the C++ setting name is given
//! for each field.

use crate::status::DetectionMethod;

/// Parameters of the blob detector, used for undistortion beads, calibration beads and the
/// blob detection methods. Mirrors OpenCV's `SimpleBlobDetector::Params` as XMALab sets it.
#[derive(Clone, Debug, PartialEq)]
pub struct BlobDetectorSettings {
    /// `BlobDetectorThresholdStep`
    pub threshold_step: f64,
    /// `BlobDetectorMinThreshold`
    pub min_threshold: f64,
    /// `BlobDetectorMaxThreshold`
    pub max_threshold: f64,
    /// `BlobDetectorMinRepeatability`
    pub min_repeatability: u32,
    /// `BlobDetectorMinDistBetweenBlobs`
    pub min_dist_between_blobs: f64,
    /// `BlobDetectorFilterByColor`
    pub filter_by_color: bool,
    /// `BlobDetectorBlobColor`
    pub blob_color: u8,
    /// `BlobDetectorFilterByArea`
    pub filter_by_area: bool,
    /// `BlobDetectorMinArea`
    pub min_area: f64,
    /// `BlobDetectorMaxArea`
    pub max_area: f64,
    /// `BlobDetectorFilterByCircularity`
    pub filter_by_circularity: bool,
    /// `BlobDetectorMinCircularity`
    pub min_circularity: f64,
    /// `BlobDetectorMaxCircularity`
    pub max_circularity: f64,
    /// `BlobDetectorFilterByInertia`
    pub filter_by_inertia: bool,
    /// `BlobDetectorMinInertiaRatio`
    pub min_inertia_ratio: f64,
    /// `BlobDetectorMaxInertiaRatio`
    pub max_inertia_ratio: f64,
    /// `BlobDetectorFilterByConvexity`
    pub filter_by_convexity: bool,
    /// `BlobDetectorMinConvexity`. The C++ default is the `float` literal `0.9f`, so the
    /// value held is the nearest `f32` to 0.9, widened.
    pub min_convexity: f64,
    /// `BlobDetectorMaxConvexity`
    pub max_convexity: f64,
}

impl Default for BlobDetectorSettings {
    fn default() -> Self {
        Self {
            threshold_step: 10.0,
            min_threshold: 10.0,
            max_threshold: 220.0,
            min_repeatability: 2,
            min_dist_between_blobs: 5.0,
            filter_by_color: true,
            blob_color: 255,
            filter_by_area: true,
            min_area: 25.0,
            max_area: 5000.0,
            filter_by_circularity: true,
            min_circularity: 0.5,
            max_circularity: 5_000_000.0,
            filter_by_inertia: true,
            min_inertia_ratio: 0.5,
            max_inertia_ratio: 5_000_000.0,
            filter_by_convexity: true,
            min_convexity: f64::from(0.9_f32),
            max_convexity: 5_000_000.0,
        }
    }
}

/// The five settings of merge-aware tracking (`MergeEpipolarThreshold`,
/// `MergeAuditThreshold`, `MergeMaxPendingFrames`, `MergeMaxHoldFrames`,
/// `MergeMaxOverlapFrames`), which nothing reads since that tracking kind was removed on
/// 26 November 2026 (parity ledger A226). They are held only so that a project file's
/// settings entry is read and written exactly as before (the same names, checks and
/// values); they change no result.
#[derive(Clone, Debug, PartialEq)]
pub struct RetiredSettings {
    values: [SettingValue; 5],
}

/// The names of [`RetiredSettings`], in their order in [`SETTING_NAMES`].
const RETIRED_NAMES: [&str; 5] = [
    "MergeEpipolarThreshold",
    "MergeAuditThreshold",
    "MergeMaxPendingFrames",
    "MergeMaxHoldFrames",
    "MergeMaxOverlapFrames",
];

impl Default for RetiredSettings {
    /// The values XMALab gives them.
    fn default() -> Self {
        Self {
            values: [
                SettingValue::Float(3.0),
                SettingValue::Float(4.0),
                SettingValue::Int(10),
                SettingValue::Int(10),
                SettingValue::Int(20),
            ],
        }
    }
}

impl RetiredSettings {
    fn index(name: &str) -> Option<usize> {
        RETIRED_NAMES.iter().position(|n| *n == name)
    }
}

/// Every setting that changes computed results.
#[derive(Clone, Debug, PartialEq)]
pub struct AlgorithmSettings {
    /// `TriangulationMethod`, 0 to 4. See `Marker::reconstruct3DPoint` for what each selects.
    pub triangulation_method: u8,
    /// `MaximumReprojectionError`, pixels. Points above it are rejected by `Marker::isValid`.
    pub maximum_reprojection_error: f64,
    /// `OptimizeRigidBody`: refine each rigid-body pose on 2D reprojection error.
    pub optimize_rigid_body: bool,
    /// `DisableRBComputeAdvanced`: do not estimate a pose when fewer than three 3D points
    /// are available.
    pub disable_rb_compute_advanced: bool,
    /// `Filter3DPoints`: filter marker positions and refit bodies, instead of filtering the
    /// pose components.
    pub filter_3d_points: bool,
    /// `RetrackOptimizedTrackedPoints`
    pub retrack_optimized_tracked_points: bool,
    /// `TrackInterpolatedPoints`
    pub track_interpolated_points: bool,
    /// `LocalUndistortionNeighbours`: control points used per local polynomial.
    pub local_undistortion_neighbours: u32,
    /// `DetectionMethodForCalibration`: the item of the Settings dialog's box, held by its
    /// number (`DetectionMethod::code`), which is **not** the detection method of that
    /// name: 0 none (the default), 1 default X-ray marker, 2 dark blobs, 3 white marker,
    /// 4 corner detection, 5 white blobs. Alt + click in the Calibration workspace runs
    /// detection method `number - 1`, or method 6 for 5 (`ui/GLCameraView.cpp:265-282`).
    pub detection_method_for_calibration: DetectionMethod,
    /// `IdentificationThresholdCalibration`, pixels.
    pub identification_threshold_calibration: u32,
    /// `OutlierThresholdForCalibration`, pixels.
    pub outlier_threshold_for_calibration: u32,
    /// `FixPrincipal`: keep the principal point fixed during calibration.
    pub fix_principal: bool,
    /// `DisableCheckerboardDetection`
    pub disable_checkerboard_detection: bool,
    /// `DisableCheckerboardRefinement`
    pub disable_checkerboard_refinement: bool,
    /// `CheckerboadInvertedAxis` (the C++ name is misspelt).
    pub checkerboard_inverted_axis: i32,
    /// `DefaultMarkerThreshold`: the threshold offset a new marker gets
    /// (`Marker::Marker`, `core/Marker.cpp:50`), and so, on loading, that of a marker stored
    /// without a `ThresholdOffset` attribute or without a `<Marker>` element (ledger F19).
    pub default_marker_threshold: i32,
    pub blob_detector: BlobDetectorSettings,
    /// Settings no longer read, kept for the project file ([`RetiredSettings`]).
    pub retired: RetiredSettings,
}

impl Default for AlgorithmSettings {
    fn default() -> Self {
        Self {
            triangulation_method: 1,
            maximum_reprojection_error: 5.0,
            optimize_rigid_body: true,
            disable_rb_compute_advanced: false,
            filter_3d_points: false,
            retrack_optimized_tracked_points: true,
            track_interpolated_points: true,
            local_undistortion_neighbours: 12,
            detection_method_for_calibration: DetectionMethod::XrayMarker,
            identification_threshold_calibration: 15,
            outlier_threshold_for_calibration: 10,
            fix_principal: false,
            disable_checkerboard_detection: false,
            disable_checkerboard_refinement: false,
            checkerboard_inverted_axis: 0,
            default_marker_threshold: 8,
            blob_detector: BlobDetectorSettings::default(),
            retired: RetiredSettings::default(),
        }
    }
}

/// The type of a setting, as `core/Settings.cpp` registers it. `Float` settings are C++
/// `float`s, held here widened to `f64`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SettingKind {
    Bool,
    Int,
    Float,
}

/// A setting's value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SettingValue {
    Bool(bool),
    Int(i64),
    Float(f64),
}

impl std::fmt::Display for SettingValue {
    /// `true`/`false`, the integer, or a float with C's `%.17g` digits (Rust's shortest
    /// round-trip form, which reads back to the same value).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bool(b) => write!(f, "{b}"),
            Self::Int(i) => write!(f, "{i}"),
            Self::Float(v) => write!(f, "{v:?}"),
        }
    }
}

/// Where a setting's value came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SettingSource {
    /// XMALab's default (or the machine's settings, for a caller that supplies them).
    Default,
    /// The project file's settings entry.
    Project,
    /// The command line (or another explicit override).
    CommandLine,
}

impl SettingSource {
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Project => "project",
            Self::CommandLine => "command line",
        }
    }
}

/// Every setting [`AlgorithmSettings`] holds, by its C++ name, in the order the settings
/// entry of a project file lists them.
pub const SETTING_NAMES: [(&str, SettingKind); 40] = [
    ("TriangulationMethod", SettingKind::Int),
    ("MaximumReprojectionError", SettingKind::Float),
    ("OptimizeRigidBody", SettingKind::Bool),
    ("DisableRBComputeAdvanced", SettingKind::Bool),
    ("Filter3DPoints", SettingKind::Bool),
    ("RetrackOptimizedTrackedPoints", SettingKind::Bool),
    ("TrackInterpolatedPoints", SettingKind::Bool),
    ("DefaultMarkerThreshold", SettingKind::Int),
    ("LocalUndistortionNeighbours", SettingKind::Int),
    ("DetectionMethodForCalibration", SettingKind::Int),
    ("IdentificationThresholdCalibration", SettingKind::Int),
    ("OutlierThresholdForCalibration", SettingKind::Int),
    ("FixPrincipal", SettingKind::Bool),
    ("DisableCheckerboardDetection", SettingKind::Bool),
    ("DisableCheckerboardRefinement", SettingKind::Bool),
    ("CheckerboadInvertedAxis", SettingKind::Int),
    ("BlobDetectorThresholdStep", SettingKind::Float),
    ("BlobDetectorMinThreshold", SettingKind::Float),
    ("BlobDetectorMaxThreshold", SettingKind::Float),
    ("BlobDetectorMinRepeatability", SettingKind::Int),
    ("BlobDetectorMinDistBetweenBlobs", SettingKind::Float),
    ("BlobDetectorFilterByColor", SettingKind::Bool),
    ("BlobDetectorBlobColor", SettingKind::Int),
    ("BlobDetectorFilterByArea", SettingKind::Bool),
    ("BlobDetectorMinArea", SettingKind::Float),
    ("BlobDetectorMaxArea", SettingKind::Float),
    ("BlobDetectorFilterByCircularity", SettingKind::Bool),
    ("BlobDetectorMinCircularity", SettingKind::Float),
    ("BlobDetectorMaxCircularity", SettingKind::Float),
    ("BlobDetectorFilterByInertia", SettingKind::Bool),
    ("BlobDetectorMinInertiaRatio", SettingKind::Float),
    ("BlobDetectorMaxInertiaRatio", SettingKind::Float),
    ("BlobDetectorFilterByConvexity", SettingKind::Bool),
    ("BlobDetectorMinConvexity", SettingKind::Float),
    ("BlobDetectorMaxConvexity", SettingKind::Float),
    ("MergeEpipolarThreshold", SettingKind::Float),
    ("MergeAuditThreshold", SettingKind::Float),
    ("MergeMaxPendingFrames", SettingKind::Int),
    ("MergeMaxHoldFrames", SettingKind::Int),
    ("MergeMaxOverlapFrames", SettingKind::Int),
];

/// The kind of a setting by its C++ name; `None` for a name [`AlgorithmSettings`] does not
/// hold.
#[must_use]
pub fn setting_kind(name: &str) -> Option<SettingKind> {
    SETTING_NAMES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|&(_, k)| k)
}

impl AlgorithmSettings {
    /// A setting by its C++ name; `None` for a name not held here.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<SettingValue> {
        use SettingValue::{Bool, Float, Int};
        let b = &self.blob_detector;
        if let Some(i) = RetiredSettings::index(name) {
            return Some(self.retired.values[i]);
        }
        Some(match name {
            "TriangulationMethod" => Int(self.triangulation_method.into()),
            "MaximumReprojectionError" => Float(self.maximum_reprojection_error),
            "OptimizeRigidBody" => Bool(self.optimize_rigid_body),
            "DisableRBComputeAdvanced" => Bool(self.disable_rb_compute_advanced),
            "Filter3DPoints" => Bool(self.filter_3d_points),
            "RetrackOptimizedTrackedPoints" => Bool(self.retrack_optimized_tracked_points),
            "TrackInterpolatedPoints" => Bool(self.track_interpolated_points),
            "DefaultMarkerThreshold" => Int(self.default_marker_threshold.into()),
            "LocalUndistortionNeighbours" => Int(self.local_undistortion_neighbours.into()),
            "DetectionMethodForCalibration" => {
                Int(self.detection_method_for_calibration.code().into())
            }
            "IdentificationThresholdCalibration" => {
                Int(self.identification_threshold_calibration.into())
            }
            "OutlierThresholdForCalibration" => Int(self.outlier_threshold_for_calibration.into()),
            "FixPrincipal" => Bool(self.fix_principal),
            "DisableCheckerboardDetection" => Bool(self.disable_checkerboard_detection),
            "DisableCheckerboardRefinement" => Bool(self.disable_checkerboard_refinement),
            "CheckerboadInvertedAxis" => Int(self.checkerboard_inverted_axis.into()),
            "BlobDetectorThresholdStep" => Float(b.threshold_step),
            "BlobDetectorMinThreshold" => Float(b.min_threshold),
            "BlobDetectorMaxThreshold" => Float(b.max_threshold),
            "BlobDetectorMinRepeatability" => Int(b.min_repeatability.into()),
            "BlobDetectorMinDistBetweenBlobs" => Float(b.min_dist_between_blobs),
            "BlobDetectorFilterByColor" => Bool(b.filter_by_color),
            "BlobDetectorBlobColor" => Int(b.blob_color.into()),
            "BlobDetectorFilterByArea" => Bool(b.filter_by_area),
            "BlobDetectorMinArea" => Float(b.min_area),
            "BlobDetectorMaxArea" => Float(b.max_area),
            "BlobDetectorFilterByCircularity" => Bool(b.filter_by_circularity),
            "BlobDetectorMinCircularity" => Float(b.min_circularity),
            "BlobDetectorMaxCircularity" => Float(b.max_circularity),
            "BlobDetectorFilterByInertia" => Bool(b.filter_by_inertia),
            "BlobDetectorMinInertiaRatio" => Float(b.min_inertia_ratio),
            "BlobDetectorMaxInertiaRatio" => Float(b.max_inertia_ratio),
            "BlobDetectorFilterByConvexity" => Bool(b.filter_by_convexity),
            "BlobDetectorMinConvexity" => Float(b.min_convexity),
            "BlobDetectorMaxConvexity" => Float(b.max_convexity),
            _ => return None,
        })
    }

    /// Sets a setting by its C++ name. An integer is accepted for a float setting.
    ///
    /// # Errors
    /// An unknown name, a value of the wrong kind, or one out of the field's range (for
    /// example a triangulation method above 4, or a detection method with no code).
    pub fn set(&mut self, name: &str, value: SettingValue) -> Result<(), String> {
        let kind = setting_kind(name).ok_or_else(|| format!("unknown setting '{name}'"))?;
        let wrong = || format!("setting {name}: {value} is not a {kind:?} value");
        let as_bool = || match value {
            SettingValue::Bool(b) => Ok(b),
            _ => Err(wrong()),
        };
        let as_int = || match value {
            SettingValue::Int(i) => Ok(i),
            _ => Err(wrong()),
        };
        let as_float = || match value {
            // Ledger F51: no setting holds NaN or an infinity.
            SettingValue::Float(v) if !v.is_finite() => {
                Err(format!("setting {name}: {v} is not a finite number"))
            }
            SettingValue::Float(v) => Ok(v),
            SettingValue::Int(i) => Ok(i as f64),
            SettingValue::Bool(_) => Err(wrong()),
        };
        fn conv<T: TryFrom<i64>>(v: i64, name: &str) -> Result<T, String> {
            T::try_from(v).map_err(|_| format!("setting {name}: {v} is out of range"))
        }
        let b = &mut self.blob_detector;
        let r = &mut self.retired.values;
        match name {
            "TriangulationMethod" => {
                let v: u8 = conv(as_int()?, name)?;
                if v > 4 {
                    return Err(format!("setting {name}: {v} is not a method (0 to 4)"));
                }
                self.triangulation_method = v;
            }
            "MaximumReprojectionError" => self.maximum_reprojection_error = as_float()?,
            "OptimizeRigidBody" => self.optimize_rigid_body = as_bool()?,
            "DisableRBComputeAdvanced" => self.disable_rb_compute_advanced = as_bool()?,
            "Filter3DPoints" => self.filter_3d_points = as_bool()?,
            "RetrackOptimizedTrackedPoints" => self.retrack_optimized_tracked_points = as_bool()?,
            "TrackInterpolatedPoints" => self.track_interpolated_points = as_bool()?,
            "DefaultMarkerThreshold" => self.default_marker_threshold = conv(as_int()?, name)?,
            "LocalUndistortionNeighbours" => {
                let v: u32 = conv(as_int()?, name)?;
                if v == 0 {
                    return Err(format!("setting {name}: must be at least 1"));
                }
                self.local_undistortion_neighbours = v;
            }
            "DetectionMethodForCalibration" => {
                let code: i32 = conv(as_int()?, name)?;
                self.detection_method_for_calibration = DetectionMethod::from_code(code)
                    .ok_or_else(|| format!("setting {name}: {code} is not a detection method"))?;
            }
            "IdentificationThresholdCalibration" => {
                self.identification_threshold_calibration = conv(as_int()?, name)?;
            }
            "OutlierThresholdForCalibration" => {
                self.outlier_threshold_for_calibration = conv(as_int()?, name)?;
            }
            "FixPrincipal" => self.fix_principal = as_bool()?,
            "DisableCheckerboardDetection" => self.disable_checkerboard_detection = as_bool()?,
            "DisableCheckerboardRefinement" => self.disable_checkerboard_refinement = as_bool()?,
            "CheckerboadInvertedAxis" => self.checkerboard_inverted_axis = conv(as_int()?, name)?,
            "BlobDetectorThresholdStep" => b.threshold_step = as_float()?,
            "BlobDetectorMinThreshold" => b.min_threshold = as_float()?,
            "BlobDetectorMaxThreshold" => b.max_threshold = as_float()?,
            "BlobDetectorMinRepeatability" => b.min_repeatability = conv(as_int()?, name)?,
            "BlobDetectorMinDistBetweenBlobs" => b.min_dist_between_blobs = as_float()?,
            "BlobDetectorFilterByColor" => b.filter_by_color = as_bool()?,
            "BlobDetectorBlobColor" => b.blob_color = conv(as_int()?, name)?,
            "BlobDetectorFilterByArea" => b.filter_by_area = as_bool()?,
            "BlobDetectorMinArea" => b.min_area = as_float()?,
            "BlobDetectorMaxArea" => b.max_area = as_float()?,
            "BlobDetectorFilterByCircularity" => b.filter_by_circularity = as_bool()?,
            "BlobDetectorMinCircularity" => b.min_circularity = as_float()?,
            "BlobDetectorMaxCircularity" => b.max_circularity = as_float()?,
            "BlobDetectorFilterByInertia" => b.filter_by_inertia = as_bool()?,
            "BlobDetectorMinInertiaRatio" => b.min_inertia_ratio = as_float()?,
            "BlobDetectorMaxInertiaRatio" => b.max_inertia_ratio = as_float()?,
            "BlobDetectorFilterByConvexity" => b.filter_by_convexity = as_bool()?,
            "BlobDetectorMinConvexity" => b.min_convexity = as_float()?,
            "BlobDetectorMaxConvexity" => b.max_convexity = as_float()?,
            // Retired (parity ledger A226): the same kinds and ranges as before (a float; a
            // `u32`), stored as given.
            "MergeEpipolarThreshold" => r[0] = SettingValue::Float(as_float()?),
            "MergeAuditThreshold" => r[1] = SettingValue::Float(as_float()?),
            "MergeMaxPendingFrames" | "MergeMaxHoldFrames" | "MergeMaxOverlapFrames" => {
                let v: u32 = conv(as_int()?, name)?;
                let i = RetiredSettings::index(name).expect("a retired name");
                r[i] = SettingValue::Int(v.into());
            }
            _ => unreachable!("every name of SETTING_NAMES is handled"),
        }
        Ok(())
    }
}

/// Settings given by name, each overriding what a project holds: the command line's, or a
/// project file's settings entry. In the order given; a later entry for the same name wins.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SettingsPatch {
    pub values: Vec<(String, SettingValue)>,
}

impl SettingsPatch {
    /// Adds (or replaces) a value.
    pub fn set(&mut self, name: impl Into<String>, value: SettingValue) {
        let name = name.into();
        self.values.retain(|(n, _)| *n != name);
        self.values.push((name, value));
    }

    /// The value given for a name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<SettingValue> {
        self.values
            .iter()
            .rev()
            .find(|(n, _)| n == name)
            .map(|&(_, v)| v)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Applies every value, in order.
    ///
    /// # Errors
    /// The first value [`AlgorithmSettings::set`] refuses.
    pub fn apply(&self, settings: &mut AlgorithmSettings) -> Result<(), String> {
        for (name, value) in &self.values {
            settings.set(name, *value)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_cpp_settings() {
        let settings = AlgorithmSettings::default();
        assert_eq!(settings.triangulation_method, 1);
        assert_eq!(settings.maximum_reprojection_error, 5.0);
        assert!(settings.optimize_rigid_body);
        assert!(!settings.filter_3d_points);
        assert_eq!(settings.local_undistortion_neighbours, 12);
        assert_eq!(settings.default_marker_threshold, 8);
        assert_eq!(settings.blob_detector.max_threshold, 220.0);
    }

    #[test]
    fn every_setting_reads_and_writes_by_name() {
        let defaults = AlgorithmSettings::default();
        let mut names: Vec<&str> = SETTING_NAMES.iter().map(|(n, _)| *n).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), SETTING_NAMES.len(), "no name twice");
        for (name, kind) in SETTING_NAMES {
            let v = defaults.get(name).expect(name);
            let k = match v {
                SettingValue::Bool(_) => SettingKind::Bool,
                SettingValue::Int(_) => SettingKind::Int,
                SettingValue::Float(_) => SettingKind::Float,
            };
            assert_eq!(k, kind, "{name}");
            let mut s = defaults.clone();
            s.set(name, v).expect(name);
            assert_eq!(s, defaults, "{name}");
        }
        let mut s = defaults.clone();
        s.set("DefaultMarkerThreshold", SettingValue::Int(5))
            .unwrap();
        assert_eq!(s.default_marker_threshold, 5);
        s.set("BlobDetectorMinArea", SettingValue::Int(30)).unwrap();
        assert_eq!(s.blob_detector.min_area, 30.0);
        assert!(s.set("TriangulationMethod", SettingValue::Int(5)).is_err());
        assert!(s.set("OptimizeRigidBody", SettingValue::Int(1)).is_err());
        assert!(s.set("NoSuchSetting", SettingValue::Int(1)).is_err());
        assert!(defaults.get("NoSuchSetting").is_none());
    }

    #[test]
    fn min_convexity_keeps_float_rounding() {
        let value = BlobDetectorSettings::default().min_convexity;
        assert_ne!(value, 0.9);
        assert!((value - 0.9).abs() < 1e-7);
    }
}
