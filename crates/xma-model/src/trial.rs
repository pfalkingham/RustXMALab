//! Trials: the videos, markers, rigid bodies and events of one recording.
//!
//! Ported from `core/Trial.h`, `core/VideoStream.h` and `core/EventData.h`.
//! Interface state the C++ keeps on the trial (active frame, active marker, active body)
//! is not part of the model.

use std::sync::Arc;

use crate::Rgb;
use crate::marker::Marker;
use crate::rigid_body::RigidBody;

/// The video of one camera in a trial: one video file, or the files of an image sequence.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VideoSource {
    /// Absolute paths, as stored in the project. One entry for a video file, several for
    /// an image sequence.
    pub filenames: Vec<String>,
    /// XMA Portal metadata for the file, kept as name and value pairs so it round-trips.
    pub metadata: Vec<(String, String)>,
}

/// A user-defined event: one flag per frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub name: String,
    pub color: Rgb,
    pub draw: bool,
    pub data: Vec<bool>,
}

/// A trial's reference calibration image of -1: what XMALab's Trial dialog stores when no
/// calibration image is calibrated in every camera (ledger U62, F68), read from and written
/// to the project file as -1. Every use of the index finds no image there.
pub const NO_REFERENCE_CALIBRATION: usize = usize::MAX;

impl Trial {
    /// The reference calibration image as an index, `None` for XMALab's -1
    /// ([`NO_REFERENCE_CALIBRATION`]). Arithmetic on the index goes through this, so that
    /// "no reference" is never moved like an index (ledger F68).
    #[must_use]
    pub fn reference_image(&self) -> Option<usize> {
        (self.reference_calibration_image != NO_REFERENCE_CALIBRATION)
            .then_some(self.reference_calibration_image)
    }
}

/// A trial.
///
/// Markers and rigid bodies are behind `Arc`, so a background job can hold a snapshot
/// cheaply and an undo step can keep the previous version of only what changed.
/// Mutate with `Arc::make_mut`.
#[derive(Clone, Debug, PartialEq)]
pub struct Trial {
    pub name: String,
    /// One per camera, in camera order. Empty for the default trial.
    pub videos: Vec<VideoSource>,
    pub markers: Vec<Arc<Marker>>,
    pub rigid_bodies: Vec<Arc<RigidBody>>,
    pub events: Vec<Event>,
    pub n_frames: usize,
    /// First frame of the range in use, 1-based as in the C++ and the project file.
    pub start_frame: usize,
    /// Last frame of the range in use, 1-based.
    pub end_frame: usize,
    /// Index of the calibration image that defines the camera poses for this trial;
    /// [`NO_REFERENCE_CALIBRATION`] for XMALab's -1 (no image: every lookup of it fails).
    pub reference_calibration_image: usize,
    /// Frames per second.
    pub recording_speed: f64,
    /// Low-pass cutoff frequency in hertz.
    pub cutoff_frequency: f64,
    /// Interpolate missing frames in 3D, then reproject, instead of in 2D per camera.
    pub interpolate_3d: bool,
    /// The placeholder trial that holds markers and bodies before any video is loaded.
    pub is_default: bool,
    pub is_copy_from_default: bool,
    pub requires_recomputation: bool,
    /// XMA Portal metadata for the trial, kept as raw XML lines so it round-trips.
    pub portal_metadata: Vec<String>,
}

impl Trial {
    /// The placeholder "Default" trial, as `Trial::Trial()` creates it.
    #[must_use]
    pub fn default_trial() -> Self {
        Self {
            name: "Default".to_owned(),
            videos: Vec::new(),
            markers: Vec::new(),
            rigid_bodies: Vec::new(),
            events: Vec::new(),
            n_frames: 1,
            start_frame: 1,
            end_frame: 1,
            reference_calibration_image: 0,
            recording_speed: 0.0,
            cutoff_frequency: 0.0,
            interpolate_3d: false,
            is_default: true,
            is_copy_from_default: false,
            requires_recomputation: false,
            portal_metadata: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_trial_matches_cpp() {
        let trial = Trial::default_trial();
        assert_eq!(trial.name, "Default");
        assert!(trial.is_default);
        assert_eq!(
            (trial.n_frames, trial.start_frame, trial.end_frame),
            (1, 1, 1)
        );
    }

    #[test]
    fn cloning_a_trial_shares_markers() {
        let mut trial = Trial::default_trial();
        trial.markers.push(Arc::new(Marker::new(2, 10, 8)));
        let snapshot = trial.clone();
        assert!(Arc::ptr_eq(&trial.markers[0], &snapshot.markers[0]));
        Arc::make_mut(&mut trial.markers[0]).description = "M1".to_owned();
        assert!(!Arc::ptr_eq(&trial.markers[0], &snapshot.markers[0]));
        assert_eq!(snapshot.markers[0].description, "");
    }
}
