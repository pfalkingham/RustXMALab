//! Markers: per-camera 2D tracks and the reconstructed 3D point.
//!
//! Ported from `core/Marker.h` and the constructor, `init` and `reset` in `core/Marker.cpp`.
//!
//! Values are stored exactly as the C++ holds them, including placeholder values for unset
//! points, because project files store them that way. Use the status to decide whether a
//! value is meaningful; the `point2d` and `point3d` accessors do this.

use crate::status::{DetectionMethod, Interpolation, MarkerStatus};
use crate::{Point2, Point3};

/// Coordinate stored for a 2D point that has not been set.
pub const UNSET_2D: f64 = -2.0;
/// Coordinate stored for a 3D point that has not been computed.
pub const UNSET_3D: f64 = -1000.0;
/// Size stored for a 2D point that has not been set.
pub const UNSET_SIZE: f64 = -1.0;
/// Tracking penalty given to a new marker.
pub const DEFAULT_MAX_PENALTY: i32 = 50;

/// One marker's data in one camera, one entry per frame.
#[derive(Clone, Debug, PartialEq)]
pub struct MarkerView {
    /// Position in distorted image pixels.
    pub points: Vec<Point2>,
    /// The 3D point projected back into this camera, in distorted image pixels.
    pub projected: Vec<Point2>,
    pub status: Vec<MarkerStatus>,
    /// Distance in pixels between `points` and `projected`.
    pub error: Vec<f64>,
    /// Detected marker radius in pixels.
    pub size: Vec<f64>,
}

impl MarkerView {
    /// A track of `n_frames` unset frames.
    #[must_use]
    pub fn new(n_frames: usize) -> Self {
        Self {
            points: vec![Point2::new(UNSET_2D, UNSET_2D); n_frames],
            projected: vec![Point2::new(UNSET_2D, UNSET_2D); n_frames],
            status: vec![MarkerStatus::Undefined; n_frames],
            error: vec![0.0; n_frames],
            size: vec![UNSET_SIZE; n_frames],
        }
    }

    /// Returns one frame to the unset state (C++ `Marker::reset`, without the recomputation).
    pub fn reset(&mut self, frame: usize) {
        self.points[frame] = Point2::new(UNSET_2D, UNSET_2D);
        self.projected[frame] = Point2::new(UNSET_2D, UNSET_2D);
        self.status[frame] = MarkerStatus::Undefined;
        self.error[frame] = 0.0;
        self.size[frame] = UNSET_SIZE;
    }
}

/// A marker tracked through a trial.
#[derive(Clone, Debug, PartialEq)]
pub struct Marker {
    pub description: String,
    pub method: DetectionMethod,
    /// Tracking penalty (C++ `maxPenalty`).
    pub max_penalty: i32,
    /// Marker size in pixels to use in place of the detected size; -1 for none.
    pub size_override: i32,
    /// Detection threshold offset, in percent.
    pub threshold_offset: i32,
    /// True when stored 3D points are stale and must be recomputed from the 2D points.
    pub requires_recomputation: bool,
    /// Position of this marker in its rigid body's reference frame, when imported.
    pub reference_3d: Option<Point3>,
    /// One track per camera.
    pub views: Vec<MarkerView>,
    pub points3d: Vec<Point3>,
    pub status3d: Vec<MarkerStatus>,
    pub error3d: Vec<f64>,
    pub interpolation: Vec<Interpolation>,
    /// Mean detected size over all cameras and frames; -1 until computed. Derived.
    pub mean_size: f64,
    /// Size range as the C++ computes it (see the parity ledger, item A1). Derived.
    pub size_range: f64,
}

impl Marker {
    /// A new marker with no data, as `Marker::Marker` creates it.
    ///
    /// `threshold_offset` comes from the `DefaultMarkerThreshold` setting in the C++.
    #[must_use]
    pub fn new(n_cameras: usize, n_frames: usize, threshold_offset: i32) -> Self {
        Self {
            description: String::new(),
            method: DetectionMethod::XrayMarker,
            max_penalty: DEFAULT_MAX_PENALTY,
            size_override: -1,
            threshold_offset,
            requires_recomputation: true,
            reference_3d: None,
            views: (0..n_cameras).map(|_| MarkerView::new(n_frames)).collect(),
            points3d: vec![Point3::new(UNSET_3D, UNSET_3D, UNSET_3D); n_frames],
            status3d: vec![MarkerStatus::Undefined; n_frames],
            error3d: vec![0.0; n_frames],
            interpolation: vec![Interpolation::None; n_frames],
            mean_size: -1.0,
            size_range: 0.0,
        }
    }

    #[must_use]
    pub fn n_frames(&self) -> usize {
        self.status3d.len()
    }

    #[must_use]
    pub fn n_cameras(&self) -> usize {
        self.views.len()
    }

    /// The 2D point in a camera at a frame, if it has data.
    #[must_use]
    pub fn point2d(&self, camera: usize, frame: usize) -> Option<Point2> {
        let view = self.views.get(camera)?;
        view.status
            .get(frame)
            .is_some_and(|s| s.has_data())
            .then(|| view.points[frame])
    }

    /// The 3D point at a frame, if it has data.
    #[must_use]
    pub fn point3d(&self, frame: usize) -> Option<Point3> {
        self.status3d
            .get(frame)
            .is_some_and(|s| s.has_data())
            .then(|| self.points3d[frame])
    }

    /// True when any frame has an interpolation method set (C++ `hasInterpolation`).
    #[must_use]
    pub fn has_interpolation(&self) -> bool {
        self.interpolation.iter().any(|&m| m != Interpolation::None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_marker_matches_cpp_defaults() {
        let marker = Marker::new(2, 5, 8);
        assert_eq!(marker.n_cameras(), 2);
        assert_eq!(marker.n_frames(), 5);
        assert_eq!(marker.max_penalty, 50);
        assert_eq!(marker.size_override, -1);
        assert_eq!(marker.threshold_offset, 8);
        assert!(marker.requires_recomputation);
        assert_eq!(marker.views[1].points[4], Point2::new(-2.0, -2.0));
        assert_eq!(marker.views[1].size[4], -1.0);
        assert_eq!(marker.points3d[0], Point3::new(-1000.0, -1000.0, -1000.0));
        assert!(!marker.has_interpolation());
    }

    #[test]
    fn accessors_respect_status() {
        let mut marker = Marker::new(1, 3, 8);
        assert_eq!(marker.point2d(0, 1), None);
        marker.views[0].points[1] = Point2::new(10.5, 20.25);
        marker.views[0].status[1] = MarkerStatus::Tracked;
        assert_eq!(marker.point2d(0, 1), Some(Point2::new(10.5, 20.25)));
        marker.views[0].status[1] = MarkerStatus::Lost;
        assert_eq!(marker.point2d(0, 1), None);
        assert_eq!(marker.point2d(3, 1), None);
        assert_eq!(marker.point3d(9), None);
    }

    #[test]
    fn reset_restores_placeholders() {
        let mut view = MarkerView::new(2);
        view.points[0] = Point2::new(1.0, 2.0);
        view.status[0] = MarkerStatus::Set;
        view.size[0] = 4.0;
        view.reset(0);
        assert_eq!(view, MarkerView::new(2));
    }
}
