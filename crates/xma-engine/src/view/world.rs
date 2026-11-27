//! What the 3D world view and the rigid-body mesh overlay of the camera views draw, as
//! plain geometry in world (calibration) coordinates, and the 3D view's orbit camera.
//!
//! Ported from `ui/WorldViewDockGLWidget.cpp` (`paintGL`, `drawCameras`, `drawMarkers`,
//! `drawRigidBodies`, `drawCalibrationCube`, the mouse, wheel and key handlers),
//! `core/RigidBody.cpp` (`RigidBody::draw3D`, `drawMesh`), `core/RigidBodyObj.cpp`
//! (`RigidBodyObj::render`), `core/Trial.cpp` (`Trial::drawRigidBodiesMesh`,
//! `renderMeshes`), `core/Camera.cpp` (`Camera::getGLTransformations`) and the mesh part of
//! `ui/GLCameraView.cpp`, `GLCameraView::paintGL` (lines 720 to 860).
//!
//! Nothing here needs a window: the interface (`xma-app`) builds its GPU buffers from these
//! values, and the tests check them against the calibration and the bodies' poses.
//!
//! # The world view, as XMALab draws it
//!
//! | Item | Colour (OpenGL, before lighting) | Shown |
//! |---|---|---|
//! | World axes, 100 units from the origin, 2.5 px | x red, y green, z blue | always |
//! | Each calibrated camera ([`WorldCamera`]): axes of 10 units | x red, y green, z blue (camera frame flipped, below) | Calibration; Marker tracking with the trial's frame in range |
//! | Each camera's frustum to the focal plane: four edges and the rectangle | white, alpha 0.2 | as the camera |
//! | The camera's image on the focal plane | its texture (Calibration: the calibration image shown; Marker tracking: the trial frame, not with the view's own timeline nor for the Default trial) | as the camera |
//! | Calibration-frame beads, spheres of radius 0.3 | reference beads 1 to 4 red, green, blue, yellow; others white | Calibration, 3D frame (not a checkerboard) |
//! | Markers with a 3D point, spheres of radius 0.3 | active red, others green | Marker tracking |
//! | Rigid bodies: lines from the reference centre to each reference point; to each virtual point stippled | body colour | Marker tracking, body visible with references and a pose |
//! | Rigid-body meshes | body colour | Marker tracking, "Draw bone mesh model" and a pose (not tied to the body's visibility) |
//!
//! Lighting: OpenGL fixed-function lighting is on for everything (`initializeGL`): global
//! ambient 0.2, light 1 ambient 0.3 and diffuse 0.5 at eye position (0, 10, 0), colour
//! material for ambient and diffuse, so a colour `c` with unit normal `n` shows as
//! `c * (0.5 + 0.5 * max(0, n · l))` ([`world_light`]). Lines and the image quads carry
//! no normal of their own (OpenGL uses whatever normal was last set); XMALab's reference
//! screenshot shows its axes at half intensity (128, 0, 0), the ambient part only, which is
//! what is drawn here for lines and images. The background is black (`glClearColor(0, 0,
//! 0, 0)`).
//!
//! # The mesh overlay, as XMALab draws it
//!
//! In Marker tracking, in a camera view (not the Detailed View), when the Display Options'
//! "Meshmodels" (`TrialDrawRigidBodyMeshmodels`) is on, "Hide all" is off, the trial is not
//! the Default trial, the project is calibrated and at least one body has "Draw bone mesh
//! model" on ([`mesh_overlay_shown`]): each such body with a mesh and a pose at the frame
//! ([`meshes_to_draw`]) is drawn through the camera's pose for the trial's reference
//! calibration image, lit by two directional lights along the viewing axis (ambient 0.1
//! each, diffuse 0.7, global ambient 0.2: `c * (0.4 + 0.7 |n_z|)`, [`camera_light`]),
//! distorted into the raw image, and blended over the image with the view's transparency
//! (0.5; Ctrl + wheel). With "Draw transparent bone models" (the default) the triangles
//! multiply onto a white buffer without depth test (so overlapping parts look darker);
//! otherwise the nearest surface wins. The image's scale and bias are applied after the
//! mesh in XMALab, so they change the mesh's colour too; here they do not (proposed ledger
//! row).
//!
//! XMALab renders the mesh undistorted and warps the picture with a per-pixel map
//! (`gl/DistortionShader.cpp`); [`distortion_map`] is the same map turned round (for each
//! undistorted pixel, where it lands in the raw image), which the interface applies per
//! vertex, so every vertex lands where [`project_mesh`] (the exact projection,
//! `Camera::projectPoint`) puts it.
//!
//! # Fixed C++ defects (proposed parity ledger rows, Interface section)
//!
//! - Rigid-body lines to virtual points: `draw3D` starts its loop at 1 and never draws the
//!   line to the first virtual point. All are drawn.
//! - With "draw filtered data" on, a body and its mesh are drawn from the filtered pose
//!   only where that pose is valid (as the camera views do, see `overlays`); the C++ draws
//!   from the filtered arrays wherever the raw pose is valid, at a zero pose where no
//!   filtered one exists.
//! - The cameras in Marker tracking are placed with the trial's reference calibration
//!   image, from which its 3D points are computed. The C++ places them with the
//!   calibration workspace's current image (`getActiveFrameCalibration`), which can be
//!   another one, and reads past the image list when that index is out of range.
//! - The wheel cannot move the eye through the centre of the orbit: the eye distance stays
//!   at least [`MIN_EYE_DISTANCE`]. The C++ adds the wheel steps without a limit, so the
//!   view turns inside out, or is empty at distance 0.
//! - The camera-view mesh is not clipped 1000 units from the camera: its far plane is
//!   [`OVERLAY_FAR`]. The C++ projection (`Camera::getGLTransformations`, far 1000) hides
//!   any part of a mesh farther than 1000 units from the X-ray source, which in a set-up
//!   measured in millimetres is often all of it.

use std::sync::Arc;

use nalgebra::Vector4;
use xma_model::{
    CalibrationObject, CalibrationType, Point2, Point3, Rgb, RigidBody, Trial, Vector3,
};
use xma_vision::Matrix3;
use xma_vision::camera::{CameraModel, RigidTransform, rodrigues_to_matrix};

use crate::Document;

pub use xma_io::obj::{ObjError, ObjMesh, read_obj};

/// The 4x4 matrices of the orbit camera.
pub type Matrix4 = nalgebra::Matrix4<f64>;

/// `WorldViewDockGLWidget` constructor and the reset: eye distance 500.
pub const EYE_DISTANCE: f64 = 500.0;
/// Starting azimuth (degrees).
pub const AZIMUTH: f64 = 45.0;
/// Starting polar angle (degrees).
pub const POLAR: f64 = -45.0;
/// `gluPerspective(25.0, w / h, 1.0, 100000.0)`.
pub const FOVY_DEG: f64 = 25.0;
pub const NEAR: f64 = 1.0;
pub const FAR: f64 = 100_000.0;
/// The smallest eye distance the wheel reaches (the C++ has none; module documentation).
pub const MIN_EYE_DISTANCE: f64 = 1.0;
/// Key P stops at 100 (`keyPressEvent`).
pub const KEY_MIN_EYE_DISTANCE: f64 = 100.0;
/// Starting focal-plane distance (`focal_plane_distance = 200`).
pub const FOCAL_PLANE: f64 = 200.0;
/// Line width of everything the world view draws as lines (`glLineWidth(2.5)`, never
/// reset).
pub const LINE_WIDTH_PX: f32 = 2.5;
/// Radius of the marker and bead spheres (`gluSphere(.., 0.3f, 32, 32)`).
pub const SPHERE_RADIUS: f64 = 0.3;
/// Slices and stacks of `gluSphere`.
pub const SPHERE_DIVISIONS: u32 = 32;
/// Camera-view mesh projection, near plane (`Camera::getGLTransformations`).
pub const OVERLAY_NEAR: f64 = 1.05;
/// Camera-view mesh projection, far plane (XMALab 1000; module documentation).
pub const OVERLAY_FAR: f64 = 100_000.0;
/// Starting mesh transparency of a camera view (`GLCameraView`, 0.5).
pub const MESH_TRANSPARENCY: f64 = 0.5;
/// The red warning while a camera's distortion map is being computed
/// (`GLCameraView.cpp:849`). XMALab 3 also shows it, for good, for a camera without any
/// distortion (its map is never computed); here it is not shown then (proposed ledger
/// row).
/// The text names this program, RustXMALab, where XMALab 3's names XMALab (ledger U173).
pub const MESH_NOT_DISTORTED: &str =
    "Rigid body models are not distorted! RustXMALab is currently computing the distortion!";

/// An OpenGL colour, components 0 to 1.
pub type GlColor = [f32; 4];

pub const RED: GlColor = [1.0, 0.0, 0.0, 1.0];
pub const GREEN: GlColor = [0.0, 1.0, 0.0, 1.0];
pub const BLUE: GlColor = [0.0, 0.0, 1.0, 1.0];
pub const YELLOW: GlColor = [1.0, 1.0, 0.0, 1.0];
pub const WHITE: GlColor = [1.0, 1.0, 1.0, 1.0];
/// The frustum lines: `glColor4f(1, 1, 1, 0.2)`, blended.
pub const FRUSTUM: GlColor = [1.0, 1.0, 1.0, 0.2];

/// A body colour as OpenGL gets it (`glColor3ub`, `glColor4f(redF ...)`).
#[must_use]
pub fn body_color(c: Rgb) -> GlColor {
    [
        f32::from(c.r) / 255.0,
        f32::from(c.g) / 255.0,
        f32::from(c.b) / 255.0,
        1.0,
    ]
}

// ---------------------------------------------------------------------------------------
// Lighting

/// The world view's lighting of a colour at a vertex: global ambient 0.2 plus light 1's
/// ambient 0.3, and its diffuse 0.5 times `max(0, n · l)`, `l` the unit vector from the
/// vertex to the light at eye position (0, 10, 0); alpha is the colour's. `normal_eye` is
/// normalised here (`GL_NORMALIZE` for meshes; the sphere normals are unit already).
/// `None` for the normal gives the ambient part only (lines and images, module
/// documentation).
///
/// Ported from `ui/WorldViewDockGLWidget.cpp`, `initializeGL` (OpenGL's lighting equation
/// with `GL_COLOR_MATERIAL` on ambient and diffuse, no specular, no attenuation).
#[must_use]
pub fn world_light(color: GlColor, normal_eye: Option<Vector3>, pos_eye: Vector3) -> GlColor {
    let mut k = 0.5;
    if let Some(n) = normal_eye {
        let nl = n.norm();
        let l = Vector3::new(0.0, 10.0, 0.0) - pos_eye;
        let ll = l.norm();
        if nl > 0.0 && ll > 0.0 {
            k += 0.5 * (n.dot(&l) / (nl * ll)).max(0.0);
        }
    }
    let k = k as f32;
    [
        (color[0] * k).min(1.0),
        (color[1] * k).min(1.0),
        (color[2] * k).min(1.0),
        color[3],
    ]
}

/// The camera view's lighting of a mesh colour: global ambient 0.2, two lights each with
/// ambient 0.1 and diffuse 0.7, directional along +z and -z of the camera
/// (`LightPosition_front`, `LightPosition_back`), so `c * (0.4 + 0.7 |n_z|)` for the unit
/// normal in camera coordinates; alpha 1.
///
/// Ported from `ui/GLCameraView.cpp`, the constructor's light values and `paintGL`.
#[must_use]
pub fn camera_light(color: GlColor, normal_camera: Vector3) -> GlColor {
    let l = normal_camera.norm();
    let nz = if l > 0.0 {
        (normal_camera.z / l).abs()
    } else {
        0.0
    };
    let k = (0.4 + 0.7 * nz) as f32;
    [
        (color[0] * k).min(1.0),
        (color[1] * k).min(1.0),
        (color[2] * k).min(1.0),
        1.0,
    ]
}

// ---------------------------------------------------------------------------------------
// The orbit camera

/// The world view's viewpoint: an eye on a sphere around a centre.
///
/// Ported from `ui/WorldViewDockGLWidget.cpp` (`eyedistance`, `azimuth`, `polar`,
/// `center_x/y/z`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Orbit {
    pub eye_distance: f64,
    /// Degrees, 0 to 180 once dragged.
    pub azimuth: f64,
    /// Degrees, 0 to 360 once dragged (starts at -45).
    pub polar: f64,
    pub center: Vector3,
}

impl Default for Orbit {
    fn default() -> Self {
        Self {
            eye_distance: EYE_DISTANCE,
            azimuth: AZIMUTH,
            polar: POLAR,
            center: Vector3::zeros(),
        }
    }
}

fn rad(deg: f64) -> f64 {
    deg * std::f64::consts::PI / 180.0
}

impl Orbit {
    /// The eye position relative to the centre (`paintGL`: `e_x`, `e_y`, `e_z`).
    #[must_use]
    pub fn eye_offset(&self) -> Vector3 {
        let (a, p) = (rad(self.azimuth), rad(self.polar));
        Vector3::new(
            self.eye_distance * p.sin() * a.sin(),
            self.eye_distance * a.cos(),
            self.eye_distance * p.cos() * a.sin(),
        )
    }

    /// The eye position.
    #[must_use]
    pub fn eye(&self) -> Vector3 {
        self.eye_offset() + self.center
    }

    /// World to eye coordinates: `gluLookAt(eye, centre, up (0, 1, 0))` then
    /// `glRotated(-90, 1, 0, 0)` and `glRotated(180, 0, 0, 1)` (`paintGL`).
    #[must_use]
    pub fn view_matrix(&self) -> Matrix4 {
        look_at(self.eye(), self.center, Vector3::new(0.0, 1.0, 0.0)) * world_turn()
    }

    /// Eye to clip coordinates (OpenGL's, depth -1 to 1): `gluPerspective(25, aspect, 1,
    /// 100000)`.
    #[must_use]
    pub fn projection(aspect: f64) -> Matrix4 {
        perspective(FOVY_DEG, aspect, NEAR, FAR)
    }

    /// Where a world point appears in a view of `width` by `height` (points, origin top
    /// left), with its depth (0 near, 1 far); `None` behind the eye or outside the depth
    /// range.
    #[must_use]
    pub fn to_view(&self, world: Point3, width: f64, height: f64) -> Option<(f64, f64, f64)> {
        let clip = Self::projection(width / height)
            * self.view_matrix()
            * Vector4::new(world.x, world.y, world.z, 1.0);
        if clip.w <= 0.0 {
            return None;
        }
        let ndc = clip.xyz() / clip.w;
        if !(-1.0..=1.0).contains(&ndc.z) {
            return None;
        }
        Some((
            (ndc.x + 1.0) * 0.5 * width,
            (1.0 - ndc.y) * 0.5 * height,
            ndc.z * 0.5 + 0.5,
        ))
    }

    /// Left drag by `(dx, dy)` points: `azimuth -= prev_y - y`, `polar -= prev_x - x`,
    /// azimuth clamped to 0 to 180, polar wrapped into 0 to 360.
    ///
    /// Ported from `WorldViewDockGLWidget::mouseMoveEvent` (left button).
    pub fn rotate(&mut self, dx: f64, dy: f64) {
        self.azimuth += dy;
        self.polar += dx;
        self.azimuth = self.azimuth.clamp(0.0, 180.0);
        while self.polar > 360.0 {
            self.polar -= 360.0;
        }
        while self.polar < 0.0 {
            self.polar += 360.0;
        }
    }

    /// Middle drag, or Ctrl + right drag, by `(dx, dy)` points: the centre moves in the
    /// view plane, `eyedistance / 500` world units per point.
    ///
    /// Ported from `WorldViewDockGLWidget::mouseMoveEvent` (middle button branch).
    pub fn pan(&mut self, dx: f64, dy: f64) {
        let (a, p) = (rad(self.azimuth), rad(self.polar));
        let e = Vector3::new(p.sin() * a.sin(), a.cos(), p.cos() * a.sin());
        let mut right = Vector3::new(-e.z, 0.0, e.x);
        let r_len = (right.x * right.x + right.z * right.z).sqrt();
        if r_len > 0.0 {
            right.x /= r_len;
            right.z /= r_len;
        }
        let up = Vector3::new(
            right.y * -e.z - right.z * -e.y,
            right.z * -e.x - right.x * -e.z,
            right.x * -e.y - right.y * -e.x,
        );
        let sens = self.eye_distance / 500.0;
        self.center -= (right * dx - up * dy) * sens;
    }

    /// Right click without Ctrl, and key R: back to the centre when panned away
    /// (squared distance above 0.0001), otherwise the starting distance and angles.
    ///
    /// Ported from `WorldViewDockGLWidget::mousePressEvent`, `keyPressEvent`.
    pub fn reset_step(&mut self) {
        if self.center.norm_squared() > 0.0001 {
            self.center = Vector3::zeros();
        } else {
            self.eye_distance = EYE_DISTANCE;
            self.azimuth = AZIMUTH;
            self.polar = POLAR;
        }
    }

    /// The wheel: `eyedistance += angleDelta / 12` (a notch of 120 moves 10 units; turning
    /// away from the user moves the eye away), never below [`MIN_EYE_DISTANCE`].
    ///
    /// Ported from `WorldViewDockGLWidget::wheelEvent` (module documentation for the
    /// limit).
    pub fn wheel(&mut self, angle_delta: f64) {
        self.eye_distance = (self.eye_distance + angle_delta / 12.0).max(MIN_EYE_DISTANCE);
    }

    /// Key O: 10 units farther. Key P: 10 nearer, not below 100.
    ///
    /// Ported from `WorldViewDockGLWidget::keyPressEvent`.
    pub fn key_zoom(&mut self, out: bool) {
        if out {
            self.eye_distance += 10.0;
        } else {
            self.eye_distance -= 10.0;
            if self.eye_distance < KEY_MIN_EYE_DISTANCE {
                self.eye_distance = KEY_MIN_EYE_DISTANCE;
            }
        }
    }
}

/// `gluLookAt`.
fn look_at(eye: Vector3, center: Vector3, up: Vector3) -> Matrix4 {
    let f = (center - eye).normalize();
    let s = f.cross(&up.normalize());
    let s = if s.norm() > 0.0 { s.normalize() } else { s };
    let u = s.cross(&f);
    let m = Matrix4::new(
        s.x, s.y, s.z, 0.0, u.x, u.y, u.z, 0.0, -f.x, -f.y, -f.z, 0.0, 0.0, 0.0, 0.0, 1.0,
    );
    m * Matrix4::new_translation(&-eye)
}

/// `glRotated(-90, 1, 0, 0) * glRotated(180, 0, 0, 1)`: world z up becomes view y up.
fn world_turn() -> Matrix4 {
    let rx = Matrix4::new(
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    );
    let rz = Matrix4::new(
        -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    );
    rx * rz
}

/// `gluPerspective`.
fn perspective(fovy_deg: f64, aspect: f64, near: f64, far: f64) -> Matrix4 {
    let f = 1.0 / (rad(fovy_deg) / 2.0).tan();
    Matrix4::new(
        f / aspect,
        0.0,
        0.0,
        0.0,
        0.0,
        f,
        0.0,
        0.0,
        0.0,
        0.0,
        (far + near) / (near - far),
        2.0 * far * near / (near - far),
        0.0,
        0.0,
        -1.0,
        0.0,
    )
}

// ---------------------------------------------------------------------------------------
// The scene

/// Which workspace the world view draws for (it draws nothing in Undistortion).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorldWorkspace {
    Undistortion,
    Calibration,
    MarkerTracking,
}

/// What the world view is asked to draw.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldQuery {
    pub workspace: WorldWorkspace,
    /// The active trial.
    pub trial: Option<usize>,
    /// The main window's frame of the trial (0-based): the cameras are drawn only while it
    /// is inside the trial's start and end frame.
    pub trial_frame: usize,
    /// The frame the markers and bodies are drawn at (0-based): the main window's, or the
    /// panel's own timeline when "enable" is checked (`WorldViewDockWidget::draw`).
    pub view_frame: usize,
    /// The panel's own timeline is on (`useCustomTimeline`): no video image on the cameras.
    pub custom_timeline: bool,
    pub active_marker: Option<usize>,
    /// The Calibration workspace's current image (`getActiveFrameCalibration`).
    pub calibration_frame: usize,
    /// "Focal plane distance".
    pub focal_plane: f64,
    /// `TrialDrawFiltered`.
    pub filtered: bool,
}

/// What a camera's image quad shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CameraImage {
    /// The image the camera view shows (calibration image or trial frame).
    Shown,
    /// No texture: the quad in its plain colour (custom timeline, Default trial).
    None,
}

/// One camera as the world view draws it.
///
/// Ported from `WorldViewDockGLWidget::drawCameras`: the camera frame used for drawing is
/// OpenCV's with x and z negated (and the principal point's y taken from the bottom row,
/// `(height - 1) - cy`), so its axes point left, down and backwards from the source.
#[derive(Clone, Debug, PartialEq)]
pub struct WorldCamera {
    pub camera: usize,
    /// The calibration image whose pose places the camera.
    pub calibration_image: usize,
    /// World to camera (OpenCV), as stored.
    pub pose: RigidTransform,
    /// The camera centre (the X-ray source) in world coordinates.
    pub position: Point3,
    /// The three axes of 10 units: end points and colours (red, green, blue).
    pub axes: [(Point3, Point3, GlColor); 3],
    /// The focal-plane rectangle, corners `(x_min, y_min)`, `(x_min, y_max)`,
    /// `(x_max, y_max)`, `(x_max, y_min)` of the C++.
    pub corners: [Point3; 4],
    /// Texture coordinates of [`WorldCamera::corners`] (OpenGL's: t = 0 is the image's
    /// first row).
    pub tex: [[f32; 2]; 4],
    pub image: CameraImage,
}

/// A marker or bead sphere.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldSphere {
    /// Marker index (Marker tracking) or bead index (Calibration).
    pub index: usize,
    pub position: Point3,
    pub color: GlColor,
}

/// A rigid body's lines (`RigidBody::draw3D`), in world coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct WorldBody {
    pub body: usize,
    pub color: GlColor,
    pub lines: Vec<[Point3; 2]>,
    /// Lines to the virtual points, drawn stippled (`glLineStipple(10, 0xAAAA)`).
    pub virtual_lines: Vec<[Point3; 2]>,
}

/// A body's mesh to draw: which file, and how its vertices reach world coordinates.
///
/// Ported from `core/RigidBodyObj.cpp`, `RigidBodyObj::render`: the OpenGL matrix is the
/// inverse of the body's pose (rotation transposed, translation `Rᵀ (−t)`), then
/// `glScaled(meshScale)`: a vertex `v` lands at `Rᵀ (s v − t)`.
#[derive(Clone, Debug, PartialEq)]
pub struct BodyMesh {
    pub body: usize,
    /// The OBJ file as the project names it.
    pub path: String,
    pub color: GlColor,
    pub scale: f64,
    /// The body's pose at the frame (world to body), raw or filtered.
    pub pose: RigidTransform,
}

impl BodyMesh {
    /// Body (mesh) to world: rotation and translation, without the scale.
    #[must_use]
    pub fn body_to_world(&self) -> (Matrix3, Vector3) {
        let rt = rodrigues_to_matrix(&self.pose.rotation).transpose();
        (rt, rt * -self.pose.translation)
    }

    /// A mesh vertex in world coordinates.
    #[must_use]
    pub fn to_world(&self, v: [f32; 3]) -> Point3 {
        let (r, t) = self.body_to_world();
        let p = Vector3::new(f64::from(v[0]), f64::from(v[1]), f64::from(v[2])) * self.scale;
        Point3::from(r * p + t)
    }
}

/// Everything the world view draws for one frame, in world coordinates.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WorldScene {
    /// The world axes: (from, to, colour).
    pub axes: Vec<(Point3, Point3, GlColor)>,
    pub cameras: Vec<WorldCamera>,
    /// Calibration-frame beads (Calibration) or markers (Marker tracking).
    pub spheres: Vec<WorldSphere>,
    pub bodies: Vec<WorldBody>,
    pub meshes: Vec<BodyMesh>,
}

/// What the world view draws.
///
/// Ported from `ui/WorldViewDockGLWidget.cpp`, `paintGL` and the functions it calls
/// (module documentation).
#[must_use]
pub fn world_scene(doc: &Document, q: &WorldQuery) -> WorldScene {
    let mut s = WorldScene {
        axes: vec![
            (Point3::origin(), Point3::new(100.0, 0.0, 0.0), RED),
            (Point3::origin(), Point3::new(0.0, 100.0, 0.0), GREEN),
            (Point3::origin(), Point3::new(0.0, 0.0, 100.0), BLUE),
        ],
        ..WorldScene::default()
    };
    let project = doc.project();
    match q.workspace {
        WorldWorkspace::Undistortion => {}
        WorldWorkspace::Calibration => {
            s.spheres = calibration_beads(project.calibration_object.as_ref());
            s.cameras = cameras(doc, q.calibration_frame, CameraImage::Shown, q.focal_plane);
        }
        WorldWorkspace::MarkerTracking => {
            let Some(t) = q.trial else { return s };
            let Some(trial) = project.trials.get(t) else {
                return s;
            };
            let start = trial.start_frame.saturating_sub(1);
            let end = trial.end_frame.saturating_sub(1);
            if q.trial_frame < start || q.trial_frame > end {
                return s;
            }
            let image = if trial.is_default || q.custom_timeline {
                CameraImage::None
            } else {
                CameraImage::Shown
            };
            s.cameras = cameras(doc, trial.reference_calibration_image, image, q.focal_plane);
            s.spheres = markers(trial, q.view_frame, q.active_marker);
            s.bodies = bodies(trial, q.view_frame, q.filtered);
            s.meshes = meshes_to_draw(trial, q.view_frame, q.filtered);
        }
    }
    s
}

/// `drawCalibrationCube`: the beads of a 3D calibration frame.
fn calibration_beads(object: Option<&CalibrationObject>) -> Vec<WorldSphere> {
    let Some(CalibrationObject::Frame(f)) = object else {
        return Vec::new();
    };
    let refs = &f.reference_ids;
    let is_ref = |k: usize, i: usize| refs.get(k).is_some_and(|r| *r >= 0 && *r as usize == i);
    f.points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let color = if is_ref(0, i) {
                RED
            } else if is_ref(1, i) {
                GREEN
            } else if is_ref(2, i) {
                BLUE
            } else if is_ref(3, i) {
                YELLOW
            } else {
                WHITE
            };
            WorldSphere {
                index: i,
                position: *p,
                color,
            }
        })
        .collect()
}

/// `drawCameras`: every calibrated camera whose image `image` is calibrated.
fn cameras(doc: &Document, image: usize, shows: CameraImage, focal: f64) -> Vec<WorldCamera> {
    let mut out = Vec::new();
    for (c, cam) in doc.project().cameras.iter().enumerate() {
        let Some(ci) = cam.calibration_images.get(image) else {
            continue;
        };
        if !cam.calibrated || ci.calibrated <= 0 {
            continue;
        }
        let pose = RigidTransform {
            rotation: ci.rotation,
            translation: ci.translation,
        };
        let k = doc.camera(c).model.intrinsics.camera_matrix;
        out.push(world_camera(
            c,
            image,
            pose,
            &k,
            f64::from(cam.width),
            f64::from(cam.height),
            focal,
            shows,
        ));
    }
    out
}

/// One camera of `drawCameras`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn world_camera(
    camera: usize,
    calibration_image: usize,
    pose: RigidTransform,
    k: &Matrix3,
    width: f64,
    height: f64,
    focal: f64,
    image: CameraImage,
) -> WorldCamera {
    let r = rodrigues_to_matrix(&pose.rotation);
    // The drawing frame: OpenCV's with x and z negated; a point p in it is at
    // Rᵀ (F p − t) in the world, F = diag(-1, 1, -1).
    let to_world = |p: Vector3| -> Point3 {
        let cv = Vector3::new(-p.x, p.y, -p.z);
        Point3::from(r.transpose() * (cv - pose.translation))
    };
    let origin = to_world(Vector3::zeros());
    let axes = [
        (origin, to_world(Vector3::new(10.0, 0.0, 0.0)), RED),
        (origin, to_world(Vector3::new(0.0, 10.0, 0.0)), GREEN),
        (origin, to_world(Vector3::new(0.0, 0.0, 10.0)), BLUE),
    ];
    let (fx, fy, cx) = (k[(0, 0)], k[(1, 1)], k[(0, 2)]);
    let cy = (height - 1.0) - k[(1, 2)];
    let x_min = (0.0 - cx) / fx;
    let x_max = (width - cx) / fx;
    let y_min = (0.0 - cy) / fy;
    let y_max = (height - cy) / fy;
    let z = -focal;
    let at = |x: f64, y: f64| to_world(Vector3::new(z * x, z * y, z));
    WorldCamera {
        camera,
        calibration_image,
        pose,
        position: origin,
        axes,
        corners: [
            at(x_min, y_min),
            at(x_min, y_max),
            at(x_max, y_max),
            at(x_max, y_min),
        ],
        tex: [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        image,
    }
}

/// `drawMarkers`: a sphere at each marker with a 3D point at the frame.
fn markers(trial: &Trial, frame: usize, active: Option<usize>) -> Vec<WorldSphere> {
    trial
        .markers
        .iter()
        .enumerate()
        .filter_map(|(i, m)| {
            m.point3d(frame).map(|p| WorldSphere {
                index: i,
                position: p,
                color: if Some(i) == active { RED } else { GREEN },
            })
        })
        .collect()
}

/// The pose a body is drawn with: raw, or filtered with "draw filtered data" (only where
/// that pose is valid; module documentation).
fn drawn_pose(rb: &RigidBody, frame: usize, filtered: bool) -> Option<RigidTransform> {
    let track = if filtered { &rb.filtered } else { &rb.raw };
    track.pose(frame).map(|p| RigidTransform {
        rotation: p.rotation,
        translation: p.translation,
    })
}

/// `RigidBody::draw3D` for every body.
fn bodies(trial: &Trial, frame: usize, filtered: bool) -> Vec<WorldBody> {
    let mut out = Vec::new();
    for (b, rb) in trial.rigid_bodies.iter().enumerate() {
        if !rb.visible || rb.references_set == 0 {
            continue;
        }
        let Some(pose) = drawn_pose(rb, frame, filtered) else {
            continue;
        };
        let r = rodrigues_to_matrix(&pose.rotation).transpose();
        let w = |p: &Point3| Point3::from(r * (p.coords - pose.translation));
        let refs = &rb.reference_points;
        let center = if refs.is_empty() {
            Point3::origin()
        } else {
            let mut sum = Vector3::zeros();
            for p in refs {
                sum += p.coords;
            }
            Point3::from(sum / refs.len() as f64)
        };
        let c = w(&center);
        out.push(WorldBody {
            body: b,
            color: body_color(rb.color),
            lines: refs.iter().map(|p| [c, w(p)]).collect(),
            // To each virtual point's place in the body, whatever its source, as the C++.
            virtual_lines: rb
                .virtual_points
                .iter()
                .map(|v| [c, w(&v.reference)])
                .collect(),
        });
    }
    out
}

/// The meshes drawn at a frame: every body with "Draw bone mesh model", a mesh file and a
/// pose. Whether the file loads is the caller's to find out (the C++ keeps no mesh for a
/// file that does not load).
///
/// Ported from `core/Trial.cpp`, `Trial::drawRigidBodiesMesh`, and
/// `core/RigidBodyObj.cpp`, `RigidBodyObj::render` (the pose condition, module
/// documentation for the filtered one). The body's visibility is not looked at, as in the
/// C++.
#[must_use]
pub fn meshes_to_draw(trial: &Trial, frame: usize, filtered: bool) -> Vec<BodyMesh> {
    trial
        .rigid_bodies
        .iter()
        .enumerate()
        .filter_map(|(b, rb)| {
            if !rb.draw_mesh_model {
                return None;
            }
            let path = rb.mesh_model.clone()?;
            let pose = drawn_pose(rb, frame, filtered)?;
            Some(BodyMesh {
                body: b,
                path,
                color: body_color(rb.color),
                scale: rb.mesh_scale,
                pose,
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------------------
// The camera views' mesh overlay

/// The settings and view facts [`mesh_overlay_shown`] reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OverlayOptions {
    /// Display Options "Meshmodels" (`TrialDrawRigidBodyMeshmodels`).
    pub meshmodels: bool,
    /// "Hide all" (`TrialDrawHideAll`).
    pub hide_all: bool,
    /// The Detailed View (never draws meshes).
    pub detail: bool,
    /// The workspace is Marker tracking.
    pub marker_tracking: bool,
}

/// Whether a camera view draws meshes at all (`GLCameraView::renderMeshes`).
///
/// Ported from `ui/GLCameraView.cpp`, `paintGL` (lines 724 to 739), with
/// `core/Trial.cpp`, `Trial::renderMeshes`.
#[must_use]
pub fn mesh_overlay_shown(doc: &Document, trial: usize, o: OverlayOptions) -> bool {
    if !o.marker_tracking || o.detail || !o.meshmodels || o.hide_all {
        return false;
    }
    let Some(t) = doc.project().trials.get(trial) else {
        return false;
    };
    t.rigid_bodies.iter().any(|b| b.draw_mesh_model)
        && !t.is_default
        && doc.project().calibration_type != CalibrationType::None
}

/// The camera's pose for the trial (world to camera): its calibration image
/// `trial.reference_calibration_image`, if that image exists.
#[must_use]
pub fn trial_camera_pose(doc: &Document, trial: &Trial, camera: usize) -> Option<RigidTransform> {
    let cam = doc.project().cameras.get(camera)?;
    let ci = cam
        .calibration_images
        .get(trial.reference_calibration_image)?;
    Some(RigidTransform {
        rotation: ci.rotation,
        translation: ci.translation,
    })
}

/// A body's mesh in one camera.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectedMesh {
    pub body: usize,
    /// Each vertex in the raw (distorted) image, pixel centres at integers.
    pub points: Vec<Point2>,
    /// Each vertex's depth along the camera axis (camera z).
    pub depth: Vec<f64>,
    /// The mesh's triangles (indices into `points`).
    pub triangles: Vec<[u32; 3]>,
}

/// Body (mesh) to camera coordinates: rotation and translation (without the mesh scale).
#[must_use]
pub fn body_to_camera(mesh: &BodyMesh, camera_pose: &RigidTransform) -> (Matrix3, Vector3) {
    let rc = rodrigues_to_matrix(&camera_pose.rotation);
    let (rb, tb) = mesh.body_to_world();
    (rc * rb, rc * tb + camera_pose.translation)
}

/// Every vertex of `obj` placed as `mesh` and projected into `camera`'s raw image.
///
/// The projection is `Camera::projectPoint` (pinhole with the trial's reference calibration
/// pose, then the camera's distortion: lens model and undistortion grid), the projection
/// the camera views use for markers. XMALab's own drawing reaches the same place through
/// its OpenGL projection (`Camera::getGLTransformations`) and the distortion shader.
#[must_use]
pub fn project_mesh(
    doc: &Document,
    trial: &Trial,
    camera: usize,
    mesh: &BodyMesh,
    obj: &ObjMesh,
) -> Option<ProjectedMesh> {
    let pose = trial_camera_pose(doc, trial, camera)?;
    let model = &doc.camera(camera).model;
    let (r, t) = body_to_camera(mesh, &pose);
    let (rb, tb) = mesh.body_to_world();
    let mut points = Vec::with_capacity(obj.positions.len());
    let mut depth = Vec::with_capacity(obj.positions.len());
    for v in &obj.positions {
        let p = Vector3::new(f64::from(v[0]), f64::from(v[1]), f64::from(v[2])) * mesh.scale;
        let world = Point3::from(rb * p + tb);
        points.push(model.project_point(&pose, world));
        depth.push((r * p + t).z);
    }
    Some(ProjectedMesh {
        body: mesh.body,
        points,
        depth,
        triangles: obj.triangles.clone(),
    })
}

/// For each pixel of the undistorted (pinhole) image, where it lands in the raw image: the
/// camera's distortion (`Camera::undistortPoint(p, false)`: lens model and grid), sampled
/// at every pixel centre. `None` for a camera without distortion (the identity).
///
/// The inverse of XMALab's distortion shader map (`gl/DistortionShader.cpp`,
/// `setDistortionMap`), which holds for each raw pixel its undistorted position; the
/// interface applies this one to mesh vertices (module documentation). Display only.
#[must_use]
pub fn distortion_map(model: &CameraModel, width: u32, height: u32) -> Option<DistortionMap> {
    if model.grid.is_none() && model.intrinsics.distortion.is_none() {
        return None;
    }
    let (w, h) = (width as usize, height as usize);
    if w == 0 || h == 0 {
        return None;
    }
    let mut data = vec![[0f32; 2]; w * h];
    let threads = std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .clamp(1, 16);
    let rows_per = h.div_ceil(threads);
    std::thread::scope(|scope| {
        for (chunk_i, chunk) in data.chunks_mut(rows_per * w).enumerate() {
            let y0 = chunk_i * rows_per;
            scope.spawn(move || {
                for (i, out) in chunk.iter_mut().enumerate() {
                    let (x, y) = (i % w, y0 + i / w);
                    let d =
                        model.undistort_point(Point2::new(x as f64, y as f64), false, true, true);
                    *out = [d.x as f32, d.y as f32];
                }
            });
        }
    });
    Some(DistortionMap {
        width: w,
        height: h,
        data: Arc::new(data),
    })
}

/// See [`distortion_map`]: `data[y * width + x]` is where undistorted pixel `(x, y)` lands.
#[derive(Clone, Debug, PartialEq)]
pub struct DistortionMap {
    pub width: usize,
    pub height: usize,
    pub data: Arc<Vec<[f32; 2]>>,
}

impl DistortionMap {
    /// The raw-image position of an undistorted point: bilinear between the four pixels
    /// around it; outside the image, the nearest edge value moved by the distance to the
    /// edge. This is what the interface's vertex shader computes.
    #[must_use]
    pub fn apply(&self, p: Point2) -> Point2 {
        let (w, h) = (self.width as f64, self.height as f64);
        let cx = p.x.clamp(0.0, w - 1.0);
        let cy = p.y.clamp(0.0, h - 1.0);
        let x0 = cx.floor().min(w - 2.0).max(0.0);
        let y0 = cy.floor().min(h - 2.0).max(0.0);
        let (fx, fy) = (cx - x0, cy - y0);
        let at = |x: f64, y: f64| {
            let xi = (x as usize).min(self.width - 1);
            let yi = (y as usize).min(self.height - 1);
            let v = self.data[yi * self.width + xi];
            (f64::from(v[0]), f64::from(v[1]))
        };
        let a = at(x0, y0);
        let b = at(x0 + 1.0, y0);
        let c = at(x0, y0 + 1.0);
        let d = at(x0 + 1.0, y0 + 1.0);
        let lerp = |u: f64, v: f64, t: f64| u + (v - u) * t;
        let x = lerp(lerp(a.0, b.0, fx), lerp(c.0, d.0, fx), fy);
        let y = lerp(lerp(a.1, b.1, fx), lerp(c.1, d.1, fx), fy);
        Point2::new(x + (p.x - cx), y + (p.y - cy))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_orbit_is_the_cpp_start() {
        let o = Orbit::default();
        assert_eq!(o.eye_distance, 500.0);
        assert_eq!(o.azimuth, 45.0);
        assert_eq!(o.polar, -45.0);
        let e = o.eye();
        assert!((e.norm() - 500.0).abs() < 1e-9);
    }

    #[test]
    fn light_is_half_without_normal() {
        let c = world_light(RED, None, Vector3::zeros());
        assert_eq!(c, [0.5, 0.0, 0.0, 1.0]);
        let c = camera_light(WHITE, Vector3::new(0.0, 0.0, -2.0));
        assert!((c[0] - 1.0).abs() < 1e-6);
        let c = camera_light(WHITE, Vector3::new(1.0, 0.0, 0.0));
        assert!((c[0] - 0.4).abs() < 1e-6);
    }
}
