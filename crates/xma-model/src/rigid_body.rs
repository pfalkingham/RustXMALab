//! Rigid bodies: a set of markers with a reference shape and a pose per frame.
//!
//! Ported from `core/RigidBody.h` and the constructor and `init` in `core/RigidBody.cpp`.

use crate::{Point3, Rgb, Vector3};

/// A rigid-body pose as the C++ stores it: a Rodrigues rotation vector and a translation.
///
/// The pose maps points in world (calibration) coordinates onto the body's reference points
/// (`core/RigidBody.cpp`, `computePose`). Exports write the inverse.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub rotation: Vector3,
    pub translation: Vector3,
}

impl Pose {
    /// The value the C++ holds for a frame with no pose: both vectors zero.
    pub const ZERO: Self = Self {
        rotation: Vector3::new(0.0, 0.0, 0.0),
        translation: Vector3::new(0.0, 0.0, 0.0),
    };
}

/// Mean and standard deviation of a rigid body's fit error at one frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FitError {
    pub mean_2d: f64,
    pub sd_2d: f64,
    pub mean_3d: f64,
    pub sd_3d: f64,
}

/// Per-frame pose data, raw or filtered.
#[derive(Clone, Debug, PartialEq)]
pub struct PoseTrack {
    pub poses: Vec<Pose>,
    /// True where `poses` holds a computed (or filtered) pose. The C++ stores this as an int.
    pub valid: Vec<bool>,
    pub errors: Vec<FitError>,
}

impl PoseTrack {
    #[must_use]
    pub fn new(n_frames: usize) -> Self {
        Self {
            poses: vec![Pose::ZERO; n_frames],
            valid: vec![false; n_frames],
            errors: vec![FitError::default(); n_frames],
        }
    }

    /// The pose at a frame, if one has been computed.
    #[must_use]
    pub fn pose(&self, frame: usize) -> Option<Pose> {
        self.valid
            .get(frame)
            .copied()
            .unwrap_or(false)
            .then(|| self.poses[frame])
    }
}

/// Where a virtual point gets its world position at each frame.
#[derive(Clone, Debug, PartialEq)]
pub enum VirtualPointSource {
    /// A point fixed in another rigid body's reference frame (C++ `dummyRBIndex >= 0`),
    /// transformed by that body's pose at each frame.
    OtherBody { body: usize, point: Point3 },
    /// World coordinates imported per frame; `None` where the frame has no value.
    Coordinates(Vec<Option<Point3>>),
}

/// A virtual ("dummy") point: a point of the body that is not a tracked marker.
#[derive(Clone, Debug, PartialEq)]
pub struct VirtualPoint {
    pub name: String,
    /// Position in this body's reference frame (C++ `dummypoints`).
    pub reference: Point3,
    pub source: VirtualPointSource,
}

/// A rigid body.
#[derive(Clone, Debug, PartialEq)]
pub struct RigidBody {
    pub description: String,
    /// Indices into the trial's markers (C++ `pointsIdx`).
    pub marker_indices: Vec<usize>,
    /// Reference position of each marker, in the order of `marker_indices`. May be the
    /// optimised positions when `has_optimized_coordinates` is true (C++ `points3D`).
    pub reference_points: Vec<Point3>,
    /// Reference positions before optimisation (C++ `points3D_original`).
    pub reference_points_original: Vec<Point3>,
    pub reference_names: Vec<String>,
    /// How the reference was set. The C++ stores an int: 0 not set, otherwise set; the exact
    /// values are kept so project files round-trip.
    pub references_set: i32,
    /// C++ `initialised`: the body has a reference shape and `computePose` may compute a
    /// pose. Set by `load` (with `referencesSet = 2`) and `setReferenceMarkerReferences`
    /// (with 1), cleared by `resetReferences` (with 0), and set by the editing operations
    /// that build a reference shape (`computeCoordinateSystem`, `fitAndComputeError`,
    /// `setReferenceFromFrame`). Not stored in the project file: the reader leaves it false
    /// and the engine derives it after loading (`references_set != 0`, which is what the
    /// load-time setters leave).
    pub initialised: bool,
    pub has_optimized_coordinates: bool,
    pub virtual_points: Vec<VirtualPoint>,
    pub override_cutoff_frequency: bool,
    pub cutoff_frequency: f64,
    pub raw: PoseTrack,
    pub filtered: PoseTrack,
    pub visible: bool,
    pub color: Rgb,
    /// Absolute path of an OBJ mesh shown with the body, if any, as the project stores it.
    pub mesh_model: Option<String>,
    pub draw_mesh_model: bool,
    pub mesh_scale: f64,
    /// Whether the mesh file could be read when the project was read: `None` when it was not
    /// checked (no mesh, or a reader without a mesh test). XMALab 3 drops a mesh whose file
    /// cannot be read (ledger F18); the fixed reader keeps the path, draw flag and scale and
    /// records `Some(false)` here.
    pub mesh_readable: Option<bool>,
}

impl RigidBody {
    /// A new, empty rigid body, as `RigidBody::RigidBody` creates it.
    #[must_use]
    pub fn new(n_frames: usize) -> Self {
        Self {
            description: String::new(),
            marker_indices: Vec::new(),
            reference_points: Vec::new(),
            reference_points_original: Vec::new(),
            reference_names: Vec::new(),
            references_set: 0,
            initialised: false,
            has_optimized_coordinates: false,
            virtual_points: Vec::new(),
            override_cutoff_frequency: false,
            cutoff_frequency: 0.0,
            raw: PoseTrack::new(n_frames),
            filtered: PoseTrack::new(n_frames),
            visible: false,
            color: Rgb::new(255, 0, 0),
            mesh_model: None,
            draw_mesh_model: false,
            mesh_scale: 1.0,
            mesh_readable: None,
        }
    }

    #[must_use]
    pub fn n_frames(&self) -> usize {
        self.raw.poses.len()
    }

    /// Clears every pose, flag and error, raw and filtered, and sizes them for `n_frames`.
    ///
    /// Ported from core/RigidBody.cpp, RigidBody::init (`clear()` then `resize`).
    pub fn init(&mut self, n_frames: usize) {
        self.raw = PoseTrack::new(n_frames);
        self.filtered = PoseTrack::new(n_frames);
    }

    /// Appends zeroed frames until there are at least `n_frames`.
    ///
    /// Ported from core/RigidBody.cpp, RigidBody::addFrame, as `computePose` calls it
    /// (`while (Frame >= poseComputed.size()) addFrame();`).
    pub fn grow_to(&mut self, n_frames: usize) {
        for track in [&mut self.raw, &mut self.filtered] {
            if track.poses.len() < n_frames {
                track.poses.resize(n_frames, Pose::ZERO);
            }
            if track.valid.len() < n_frames {
                track.valid.resize(n_frames, false);
            }
            if track.errors.len() < n_frames {
                track.errors.resize(n_frames, FitError::default());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_body_matches_cpp_defaults() {
        let body = RigidBody::new(4);
        assert_eq!(body.n_frames(), 4);
        assert_eq!(body.color, Rgb::new(255, 0, 0));
        assert!(!body.visible);
        assert_eq!(body.mesh_scale, 1.0);
        assert_eq!(body.raw.pose(0), None);
        assert_eq!(body.filtered.pose(7), None);
    }

    #[test]
    fn pose_is_returned_only_when_valid() {
        let mut track = PoseTrack::new(2);
        track.poses[1] = Pose {
            rotation: Vector3::new(0.1, 0.2, 0.3),
            translation: Vector3::new(1.0, 2.0, 3.0),
        };
        assert_eq!(track.pose(1), None);
        track.valid[1] = true;
        assert_eq!(track.pose(1).map(|p| p.translation.z), Some(3.0));
    }

    #[test]
    fn init_clears_and_grow_to_appends_zeros() {
        let mut body = RigidBody::new(2);
        body.raw.valid[1] = true;
        body.filtered.errors[0].mean_2d = 1.0;
        body.grow_to(4);
        assert_eq!(body.n_frames(), 4);
        assert!(body.raw.valid[1]);
        assert_eq!(body.filtered.errors.len(), 4);
        assert_eq!(body.filtered.poses[3], Pose::ZERO);
        body.grow_to(1);
        assert_eq!(body.n_frames(), 4);
        body.init(3);
        assert_eq!(body.raw, PoseTrack::new(3));
        assert_eq!(body.filtered, PoseTrack::new(3));
    }
}
