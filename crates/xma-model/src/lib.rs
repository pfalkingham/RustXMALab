//! Data types for an XMALab project: project, camera, trial, marker and
//! rigid body. Data only; no algorithms, I/O or GUI.
//!
//! Dependency rule: depends on no other workspace crate.
//!
//! The types hold what XMALab 3 stores in a project file, with the same placeholder values
//! and integer codes, so files round-trip exactly. Anything that can be recomputed from the
//! stored data (fitted undistortion models, lookup maps, projection matrices) is left to the
//! engine. Interface state (active frame, active marker, expanded tree rows) is not here.
//!
//! Conventions:
//! - Frame, camera, marker and body indices are 0-based. The 1-based numbering of XMALab 3
//!   appears only where a field's documentation says so.
//! - Image coordinates are in pixels, origin top-left, pixel centres at integer values.
//! - [`Project`] holds cameras and trials behind `Arc`, and [`Trial`] holds markers and rigid
//!   bodies behind `Arc`. Cloning is cheap and shares data; mutate with `Arc::make_mut`.

pub mod camera;
pub mod compat;
pub mod marker;
pub mod project;
pub mod rigid_body;
pub mod settings;
pub mod status;
pub mod trial;

pub use camera::{
    AdjustedView, CalibrationImage, CalibrationSequence, Camera, ImageAdjustment, UndistortionGrid,
};
pub use compat::CppCompat;
pub use marker::{Marker, MarkerView};
pub use project::{CalibrationFrame, CalibrationObject, Project};
pub use rigid_body::{FitError, Pose, PoseTrack, RigidBody, VirtualPoint, VirtualPointSource};
pub use settings::{
    AlgorithmSettings, BlobDetectorSettings, RetiredSettings, SETTING_NAMES, SettingKind,
    SettingSource, SettingValue, SettingsPatch, setting_kind,
};
pub use status::{CalibrationType, DetectionMethod, Interpolation, MarkerStatus};
pub use trial::{Event, Trial, VideoSource};

/// Name of this crate. The stub crates use it to check that the workspace links; it goes
/// when they have real contents.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");

/// A point in an image, in pixels.
pub type Point2 = nalgebra::Point2<f64>;
/// A point in 3D, in the units of the calibration object.
pub type Point3 = nalgebra::Point3<f64>;
/// A 3D vector: a translation, or a Rodrigues rotation vector.
pub type Vector3 = nalgebra::Vector3<f64>;
/// A 3x3 matrix: a camera matrix or a rotation.
pub type Matrix3 = nalgebra::Matrix3<f64>;

/// An opaque colour, as XMALab 3 stores it (`#rrggbb`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Rgb {
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Parses `#rrggbb`, the form Qt's `QColor::name()` writes.
    #[must_use]
    pub fn from_hex(text: &str) -> Option<Self> {
        let hex = text.strip_prefix('#')?;
        if hex.len() != 6 || !hex.is_ascii() {
            return None;
        }
        let channel = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).ok();
        Some(Self::new(channel(0)?, channel(2)?, channel(4)?))
    }

    /// Formats as lower-case `#rrggbb`, as Qt's `QColor::name()` does.
    #[must_use]
    pub fn to_hex(self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_hex_round_trip() {
        let colour = Rgb::new(255, 140, 0);
        assert_eq!(colour.to_hex(), "#ff8c00");
        assert_eq!(Rgb::from_hex("#ff8c00"), Some(colour));
        assert_eq!(Rgb::from_hex("#FF8C00"), Some(colour));
    }

    #[test]
    fn rgb_rejects_malformed_text() {
        assert_eq!(Rgb::from_hex("ff8c00"), None);
        assert_eq!(Rgb::from_hex("#ff8c0"), None);
        assert_eq!(Rgb::from_hex("#ff8c0g"), None);
        assert_eq!(Rgb::from_hex("#ffé8c"), None);
    }
}
