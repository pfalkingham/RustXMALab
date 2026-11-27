//! Status, interpolation and detection-method codes.
//!
//! The numeric values are written into project files and must not change.
//! Ported from `core/Marker.h` (`markerStatus`, `interpolationMethod`), `core/Project.h`
//! (`e_calibrationType`) and the method list in `ui/MarkerTreeWidget.cpp`.

/// Defines a C-like enum whose integer codes are part of the file format,
/// with `code` and `from_code` conversions.
macro_rules! coded_enum {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $( $(#[$vmeta:meta])* $variant:ident = $code:literal, )+
        }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[repr(i32)]
        $vis enum $name {
            $( $(#[$vmeta])* $variant = $code, )+
        }

        impl $name {
            /// The integer stored in project files.
            #[must_use]
            pub const fn code(self) -> i32 {
                self as i32
            }

            /// The value for a stored integer, or `None` if the integer is not a known code.
            #[must_use]
            pub const fn from_code(code: i32) -> Option<Self> {
                match code {
                    $( $code => Some(Self::$variant), )+
                    _ => None,
                }
            }
        }
    };
}

coded_enum! {
    /// Tracking status of a marker in one camera at one frame, or of its 3D point at one frame.
    ///
    /// Ordering follows the codes, so `status > MarkerStatus::Undefined` is the C++ test for
    /// "this point has data".
    pub enum MarkerStatus {
        Untrackable = -30,
        Deleted = -20,
        Lost = -10,
        #[default]
        Undefined = 0,
        Predicted = 10,
        Interpolated = 20,
        Tracked = 40,
        TrackedAndOptimized = 45,
        Set = 50,
        SetAndOptimized = 55,
        Manual = 60,
        ManualAndOptimized = 65,
    }
}

impl MarkerStatus {
    /// True when the point carries usable data (C++: `status > UNDEFINED`).
    #[must_use]
    pub const fn has_data(self) -> bool {
        self.code() > 0
    }
}

coded_enum! {
    /// How a missing frame is filled in for a marker.
    pub enum Interpolation {
        #[default]
        None = 0,
        Repeat = 10,
        Linear = 20,
        Cubic = 30,
    }
}

coded_enum! {
    /// Marker detection method. The names shown to the user are in [`DetectionMethod::label`].
    pub enum DetectionMethod {
        #[default]
        XrayMarker = 0,
        Blob = 1,
        WhiteMarker = 2,
        Corner = 3,
        NoDetection = 4,
        XrayMarkerCenterOfMass = 5,
        WhiteBlob = 6,
    }
}

impl DetectionMethod {
    /// The label XMALab 3 shows for this method.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::XrayMarker => "default Xray marker",
            Self::Blob => "Blobdetection",
            Self::WhiteMarker => "white marker",
            Self::Corner => "corner detection",
            Self::NoDetection => "no detection",
            Self::XrayMarkerCenterOfMass => "xray marker (center of mass)",
            Self::WhiteBlob => "white Blobdetection",
        }
    }
}

coded_enum! {
    /// How the project's cameras were calibrated.
    pub enum CalibrationType {
        #[default]
        None = 0,
        Internal = 1,
        External = 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_codes_round_trip() {
        for code in [-30, -20, -10, 0, 10, 20, 40, 45, 50, 55, 60, 65] {
            let status = MarkerStatus::from_code(code).expect("known code");
            assert_eq!(status.code(), code);
        }
        assert_eq!(MarkerStatus::from_code(30), None);
    }

    #[test]
    fn status_order_matches_codes() {
        assert!(MarkerStatus::Tracked > MarkerStatus::Undefined);
        assert!(MarkerStatus::Lost < MarkerStatus::Undefined);
        assert!(MarkerStatus::Predicted.has_data());
        assert!(!MarkerStatus::Undefined.has_data());
        assert!(!MarkerStatus::Untrackable.has_data());
    }

    #[test]
    fn detection_methods_cover_zero_to_six() {
        for code in 0..=6 {
            assert!(DetectionMethod::from_code(code).is_some());
        }
        assert_eq!(DetectionMethod::from_code(7), None);
    }
}
