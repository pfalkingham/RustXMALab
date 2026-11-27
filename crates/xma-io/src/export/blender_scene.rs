//! The "Blender scene" export (new in RustXMALab; parity ledger F85): a folder that Blender
//! turns into a scene holding the calibrated cameras, each with its image on an image plane.
//!
//! A `.blend` file cannot sensibly be written from outside Blender, so the export writes
//!
//! ```text
//! <output>/<name>/
//!   build_scene.py     the same file for every export (BUILD_SCENE_PY)
//!   scene.json         this export's data ([`BlenderScene::to_json`])
//!   images/camera1/…   one undistorted calibration image, or an undistorted trial sequence
//!   README.txt         how to use it (readme)
//!   <name>.blend       only when the export ran Blender
//! ```
//!
//! This module holds the data (plain structs), its JSON and the two fixed texts; the
//! engine (`xma_engine::blender_scene`) fills the data from a project, writes the folder and
//! runs Blender when asked.
//!
//! # `scene.json`, formats 1 and 2
//!
//! A scene without rigid bodies and 3D points is written as format 1 (what stage 1 wrote);
//! one with them as format 2, which adds `point_size` and fills `bodies` and `points`. The
//! script reads both; a script older than format 2 refuses a format 2 file rather than
//! building it without its bodies and points.
//!
//! ```text
//! {
//!   "format": 1 | 2,
//!   "generator": "RustXMALab x.y.z",
//!   "name": "...",                     the collection's name in Blender
//!   "kind": "cameras" | "trial",
//!   "project": "file.xma",
//!   "calibration_image": n | null,     1-based
//!   "trial": null | {"number": n, "name": "...", "first_frame": a, "last_frame": b},
//!   "scene": {"frame_start": a, "frame_end": b, "fps": f},
//!   "cameras": [{"name", "label", "number", "width", "height", "fx", "fy", "cx", "cy",
//!                "rotation": [[3],[3],[3]], "translation": [3], "flipped", "undistorted",
//!                "plane_distance",
//!                "image": null | {"path", "sequence": null | {"first", "count"}, "mirrored"}}],
//!   "point_size": s,                   format 2: the points' display size (scene units)
//!   "bodies": [{"name", "number", "filtered", "first_frame",
//!               "matrices": [null | [[4],[4],[4],[4]], ...],   one per scene frame
//!               "mesh": null | {"path": "meshes/....obj", "scale": s, "source": "..."}}],
//!   "points": [{"name", "number", "filtered", "first_frame",
//!               "positions": [null | [x, y, z], ...]}]         one per scene frame
//! }
//! ```
//!
//! A body's `matrices[k]` is its pose at scene frame `first_frame + k`: the 4x4 matrix (rows)
//! taking body (reference, CT) coordinates to world, `[Rᵀ | −Rᵀt]` of the stored pose, the
//! matrix File › Export › Rigid body transformations writes (`rigid_bodies::
//! transformation_matrix`, ledger F3); `null` where the body has no pose. Its mesh's vertex v
//! is at `M (s·v)`: the OBJ file in `meshes/` is imported without axis conversion, as a child
//! of the body with scale s. A point's `positions[k]` is its 3D point at frame
//! `first_frame + k` (what File › Export › 3D points writes), `null` where it has none.
//!
//! K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] is the camera matrix of the undistorted image
//! (pixel centres at integers, origin top-left, y down); `rotation` R and `translation` t
//! take world to camera with OpenCV's axes (x right, y down, z forward): x = R X + t. Frames
//! are numbered from 1, as in the interface. `build_scene.py`'s header says how Blender's
//! camera is derived from these numbers. Numbers are written with Rust's shortest
//! round-trip form; a non-finite number is never written ([`BlenderScene::to_json`] fails).

use std::fmt::Write as _;

/// The script written into every export (see its header).
pub const BUILD_SCENE_PY: &str = include_str!("blender_scene/build_scene.py");

/// The format number of a `scene.json` without bodies and points.
pub const FORMAT: u32 = 1;

/// The format number of a `scene.json` with bodies or points.
pub const FORMAT_ANIMATED: u32 = 2;

/// What the scene shows.
#[derive(Clone, Debug, PartialEq)]
pub enum BlenderSceneKind {
    /// The cameras at the poses of one calibration image (1-based).
    Cameras { calibration_image: usize },
    /// A trial: the cameras at its reference calibration's poses.
    Trial(BlenderTrial),
}

/// The trial of a trial scene.
#[derive(Clone, Debug, PartialEq)]
pub struct BlenderTrial {
    /// 1-based.
    pub number: usize,
    pub name: String,
    /// 1-based, inclusive: the frames whose images are written; when the images are left
    /// out, the whole trial if it holds bodies or points, else the trial's start and end
    /// frame.
    pub first_frame: usize,
    pub last_frame: usize,
}

/// A camera's image on its plane.
#[derive(Clone, Debug, PartialEq)]
pub struct BlenderImage {
    /// Relative to the folder, with `/`.
    pub path: String,
    /// A numbered sequence: files for frames `first ..= first + count - 1` (1-based, the
    /// number in each file's name; `path` is the first).
    pub sequence: Option<BlenderSequence>,
    /// The files are mirrored left to right relative to the image the calibration refers
    /// to (a flipped camera whose frames are written as their files hold them).
    pub mirrored: bool,
}

/// See [`BlenderImage::sequence`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlenderSequence {
    pub first: usize,
    pub count: usize,
}

/// One camera.
#[derive(Clone, Debug, PartialEq)]
pub struct BlenderCamera {
    /// The object's name in Blender (`Camera1`).
    pub name: String,
    /// The camera's name in the project.
    pub label: String,
    /// 1-based.
    pub number: usize,
    pub width: u32,
    pub height: u32,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    /// World to camera, rows.
    pub rotation: [[f64; 3]; 3],
    pub translation: [f64; 3],
    pub flipped: bool,
    pub undistorted: bool,
    pub plane_distance: f64,
    pub image: Option<BlenderImage>,
}

/// A rigid body's mesh: an OBJ file copied into the folder.
#[derive(Clone, Debug, PartialEq)]
pub struct BlenderMesh {
    /// Relative to the folder, with `/` (`meshes/<name>.obj`).
    pub path: String,
    /// The body's mesh scale.
    pub scale: f64,
    /// The file's name in the project (for the reader).
    pub source: String,
}

/// A rigid body, animated.
#[derive(Clone, Debug, PartialEq)]
pub struct BlenderBody {
    /// The object's name in Blender.
    pub name: String,
    /// 1-based.
    pub number: usize,
    pub filtered: bool,
    /// The scene frame of `matrices[0]` (1-based).
    pub first_frame: usize,
    /// Body to world, rows, per frame; `None` without a pose.
    pub matrices: Vec<Option<[[f64; 4]; 4]>>,
    pub mesh: Option<BlenderMesh>,
}

/// A marker's 3D point, animated.
#[derive(Clone, Debug, PartialEq)]
pub struct BlenderPoint {
    /// The object's name in Blender.
    pub name: String,
    /// 1-based.
    pub number: usize,
    pub filtered: bool,
    /// The scene frame of `positions[0]` (1-based).
    pub first_frame: usize,
    pub positions: Vec<Option<[f64; 3]>>,
}

/// The data of `scene.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct BlenderScene {
    pub generator: String,
    pub name: String,
    pub project: String,
    pub kind: BlenderSceneKind,
    pub frame_start: usize,
    pub frame_end: usize,
    pub fps: f64,
    pub cameras: Vec<BlenderCamera>,
    /// Format 2: the display size of points (bodies are drawn larger).
    pub point_size: f64,
    pub bodies: Vec<BlenderBody>,
    pub points: Vec<BlenderPoint>,
}

/// A JSON value, written by hand (the crate has no JSON library).
enum J {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(&'static str, J)>),
}

impl J {
    fn n(v: impl Into<f64>) -> J {
        J::Num(v.into())
    }

    fn int(v: usize) -> J {
        J::Num(v as f64)
    }

    fn s(v: &str) -> J {
        J::Str(v.to_owned())
    }

    fn write(&self, out: &mut String, indent: usize, path: &str) -> Result<(), String> {
        let pad = |n: usize| "  ".repeat(n);
        match self {
            J::Null => out.push_str("null"),
            J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            J::Num(v) => {
                if !v.is_finite() {
                    return Err(format!("{path} is not a finite number ({v})"));
                }
                // Shortest text that reads back as the same double; `-0` as `0`.
                let v = if *v == 0.0 { 0.0 } else { *v };
                let _ = write!(out, "{v:?}");
                if out.ends_with(".0") {
                    out.truncate(out.len() - 2);
                }
            }
            J::Str(s) => write_string(out, s),
            J::Arr(items) => {
                let flat = items.iter().all(|i| matches!(i, J::Num(_) | J::Arr(_)));
                out.push('[');
                for (k, i) in items.iter().enumerate() {
                    if k > 0 {
                        out.push_str(if flat { ", " } else { "," });
                    }
                    if !flat {
                        out.push('\n');
                        out.push_str(&pad(indent + 1));
                    }
                    i.write(out, indent + 1, &format!("{path}[{k}]"))?;
                }
                if !flat && !items.is_empty() {
                    out.push('\n');
                    out.push_str(&pad(indent));
                }
                out.push(']');
            }
            J::Obj(fields) => {
                out.push('{');
                for (k, (name, v)) in fields.iter().enumerate() {
                    if k > 0 {
                        out.push(',');
                    }
                    out.push('\n');
                    out.push_str(&pad(indent + 1));
                    write_string(out, name);
                    out.push_str(": ");
                    v.write(out, indent + 1, &format!("{path}.{name}"))?;
                }
                if !fields.is_empty() {
                    out.push('\n');
                    out.push_str(&pad(indent));
                }
                out.push('}');
            }
        }
        Ok(())
    }
}

fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

impl BlenderCamera {
    fn json(&self) -> J {
        let image = match &self.image {
            None => J::Null,
            Some(i) => J::Obj(vec![
                ("path", J::s(&i.path)),
                (
                    "sequence",
                    match i.sequence {
                        None => J::Null,
                        Some(s) => {
                            J::Obj(vec![("first", J::int(s.first)), ("count", J::int(s.count))])
                        }
                    },
                ),
                ("mirrored", J::Bool(i.mirrored)),
            ]),
        };
        J::Obj(vec![
            ("name", J::s(&self.name)),
            ("label", J::s(&self.label)),
            ("number", J::int(self.number)),
            ("width", J::n(self.width)),
            ("height", J::n(self.height)),
            ("fx", J::n(self.fx)),
            ("fy", J::n(self.fy)),
            ("cx", J::n(self.cx)),
            ("cy", J::n(self.cy)),
            (
                "rotation",
                J::Arr(
                    self.rotation
                        .iter()
                        .map(|r| J::Arr(r.iter().map(|&v| J::n(v)).collect()))
                        .collect(),
                ),
            ),
            (
                "translation",
                J::Arr(self.translation.iter().map(|&v| J::n(v)).collect()),
            ),
            ("flipped", J::Bool(self.flipped)),
            ("undistorted", J::Bool(self.undistorted)),
            ("plane_distance", J::n(self.plane_distance)),
            ("image", image),
        ])
    }

    /// Every number of the camera is finite.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        [self.fx, self.fy, self.cx, self.cy, self.plane_distance]
            .iter()
            .chain(self.rotation.iter().flatten())
            .chain(self.translation.iter())
            .all(|v| v.is_finite())
    }
}

fn num_arr(v: &[f64]) -> J {
    J::Arr(v.iter().map(|&x| J::n(x)).collect())
}

impl BlenderBody {
    fn json(&self) -> J {
        J::Obj(vec![
            ("name", J::s(&self.name)),
            ("number", J::int(self.number)),
            ("filtered", J::Bool(self.filtered)),
            ("first_frame", J::int(self.first_frame)),
            (
                "matrices",
                J::Arr(
                    self.matrices
                        .iter()
                        .map(|m| match m {
                            None => J::Null,
                            Some(m) => J::Arr(m.iter().map(|r| num_arr(r)).collect()),
                        })
                        .collect(),
                ),
            ),
            (
                "mesh",
                match &self.mesh {
                    None => J::Null,
                    Some(m) => J::Obj(vec![
                        ("path", J::s(&m.path)),
                        ("scale", J::n(m.scale)),
                        ("source", J::s(&m.source)),
                    ]),
                },
            ),
        ])
    }
}

impl BlenderPoint {
    fn json(&self) -> J {
        J::Obj(vec![
            ("name", J::s(&self.name)),
            ("number", J::int(self.number)),
            ("filtered", J::Bool(self.filtered)),
            ("first_frame", J::int(self.first_frame)),
            (
                "positions",
                J::Arr(
                    self.positions
                        .iter()
                        .map(|p| p.map_or(J::Null, |p| num_arr(&p)))
                        .collect(),
                ),
            ),
        ])
    }
}

impl BlenderScene {
    /// The text of `scene.json`.
    ///
    /// # Errors
    /// A number is not finite (named by its place in the file); nothing is written then.
    pub fn to_json(&self) -> Result<String, String> {
        let (kind, calibration_image, trial) = match &self.kind {
            BlenderSceneKind::Cameras { calibration_image } => {
                ("cameras", J::int(*calibration_image), J::Null)
            }
            BlenderSceneKind::Trial(t) => (
                "trial",
                J::Null,
                J::Obj(vec![
                    ("number", J::int(t.number)),
                    ("name", J::s(&t.name)),
                    ("first_frame", J::int(t.first_frame)),
                    ("last_frame", J::int(t.last_frame)),
                ]),
            ),
        };
        let animated = !self.bodies.is_empty() || !self.points.is_empty();
        let mut root = vec![
            (
                "format",
                J::n(if animated { FORMAT_ANIMATED } else { FORMAT }),
            ),
            ("generator", J::s(&self.generator)),
            ("name", J::s(&self.name)),
            ("kind", J::s(kind)),
            ("project", J::s(&self.project)),
            ("calibration_image", calibration_image),
            ("trial", trial),
            (
                "scene",
                J::Obj(vec![
                    ("frame_start", J::int(self.frame_start)),
                    ("frame_end", J::int(self.frame_end)),
                    ("fps", J::n(self.fps)),
                ]),
            ),
            (
                "cameras",
                J::Arr(self.cameras.iter().map(BlenderCamera::json).collect()),
            ),
        ];
        if animated {
            root.push(("point_size", J::n(self.point_size)));
        }
        root.push((
            "bodies",
            J::Arr(self.bodies.iter().map(BlenderBody::json).collect()),
        ));
        root.push((
            "points",
            J::Arr(self.points.iter().map(BlenderPoint::json).collect()),
        ));
        let root = J::Obj(root);
        let mut out = String::new();
        root.write(&mut out, 0, "scene")?;
        out.push('\n');
        Ok(out)
    }

    /// The text of `README.txt`; `blend`: the name of the `.blend` when one was saved.
    #[must_use]
    pub fn readme(&self, blend: Option<&str>) -> String {
        let what = match &self.kind {
            BlenderSceneKind::Cameras { calibration_image } => format!(
                "the calibrated cameras at their poses for calibration image {calibration_image}, \
                 each with that calibration image (undistorted) on its image plane"
            ),
            BlenderSceneKind::Trial(t) => format!(
                "the cameras of trial {} (\"{}\") at the poses of its reference calibration, \
                 each with the trial's undistorted images of frames {} to {} on its image \
                 plane as an image sequence: Blender frame f shows trial frame f",
                t.number, t.name, t.first_frame, t.last_frame
            ),
        };
        let mut s = format!(
            "Blender scene exported by {} from {}\n\n\
             The scene holds {what}.\n\
             Images that were left out leave the planes grey.\n\n",
            self.generator, self.project
        );
        if !self.bodies.is_empty() || !self.points.is_empty() {
            let filtered =
                self.bodies.iter().any(|b| b.filtered) || self.points.iter().any(|p| p.filtered);
            let meshes = self.bodies.iter().filter(|b| b.mesh.is_some()).count();
            let _ = writeln!(
                s,
                "It also holds {} rigid bod{} and {} 3D point{}{}, in the sub-collections\n\
                 \"Rigid bodies\" and \"Points\".",
                self.bodies.len(),
                if self.bodies.len() == 1 { "y" } else { "ies" },
                self.points.len(),
                if self.points.len() == 1 { "" } else { "s" },
                if filtered { " (filtered)" } else { "" },
            );
            s.push_str(
                "Each rigid body is an Empty named after the body, keyed at every frame where\n\
                 the program has a pose for it with the matrix File > Export > Rigid body\n\
                 transformations writes (body reference coordinates to world), and hidden at\n\
                 the frames where it has none.\n",
            );
            if meshes > 0 {
                s.push_str(
                    "A body's bone mesh (the OBJ file in meshes/, imported without axis\n\
                     conversion) is its child, named after it with _mesh, scaled by the body's\n\
                     mesh scale.\n",
                );
            }
            s.push_str(
                "Each point is an Empty keyed at the frames with a 3D point (the values\n\
                 File > Export > 3D points writes) and hidden at the others.\n\n",
            );
        }
        if let Some(b) = blend {
            let _ = write!(
                s,
                "Open {b} in Blender: the scene is there, with its images referred to by \
                 paths relative to it (keep this folder together).\n\n"
            );
        }
        s.push_str(
            "Three ways to build the scene with Blender 4.2 or later:\n\n\
             1. In Blender: Scripting tab, Open build_scene.py from this folder, Run Script.\n   \
             The scene is added to the file that is open, in a new collection.\n\
             2. From a shell, in this folder: blender --python build_scene.py\n\
             3. Without a window, saving a .blend beside the script:\n   \
             blender --background --factory-startup --python build_scene.py -- --save scene.blend\n   \
             Add --pack to pack the still images into the .blend. Blender cannot pack an\n   \
             image sequence: a trial's images stay files beside the .blend.\n\n\
             Each camera is an object CameraN with its image plane CameraN_plane (material\n\
             CameraN_material). Looking through a camera shows what the X-ray camera saw\n\
             only when the scene's render resolution and pixel aspect are that camera's own:\n\
             the scene's are set from camera 1, and each camera object holds its own as custom\n\
             properties (xma_width, xma_height, xma_pixel_aspect_x, xma_pixel_aspect_y). In\n\
             the Python console, after running build_scene.py in the Text editor:\n   \
             use_camera(\"Camera2\")\n\
             sets them. The plane's custom property \"distance\" slides it along the view; it\n\
             keeps filling the frame. Units are the calibration object's.\n\n\
             build_scene.py's header documents scene.json.\n",
        );
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> BlenderCamera {
        BlenderCamera {
            name: "Camera1".into(),
            label: "Cäm \"1\"\n".into(),
            number: 1,
            width: 800,
            height: 600,
            fx: 1500.0,
            fy: 1200.25,
            cx: 250.0,
            cy: 0.1 + 0.2,
            rotation: [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            translation: [1e-300, -0.0, 5e20],
            flipped: true,
            undistorted: false,
            plane_distance: 650.0,
            image: Some(BlenderImage {
                path: "images/camera1/a b.png".into(),
                sequence: Some(BlenderSequence {
                    first: 10,
                    count: 11,
                }),
                mirrored: true,
            }),
        }
    }

    fn scene() -> BlenderScene {
        BlenderScene {
            generator: "RustXMALab test".into(),
            name: "s".into(),
            project: "p.xma".into(),
            kind: BlenderSceneKind::Cameras {
                calibration_image: 2,
            },
            frame_start: 1,
            frame_end: 1,
            fps: 30.0,
            cameras: vec![camera()],
            point_size: 0.0,
            bodies: Vec::new(),
            points: Vec::new(),
        }
    }

    #[test]
    fn json_numbers_round_trip_and_strings_are_escaped() {
        let j = scene().to_json().unwrap();
        assert!(j.contains("\"cy\": 0.30000000000000004"), "{j}");
        assert!(j.contains("\"translation\": [1e-300, 0, 5e20]"), "{j}");
        assert!(j.contains("\"label\": \"Cäm \\\"1\\\"\\n\""), "{j}");
        assert!(j.contains("\"format\": 1,"));
        assert!(j.contains("\"bodies\": []"));
        assert!(j.contains("\"sequence\": {\n"));
    }

    #[test]
    fn non_finite_numbers_are_refused() {
        let mut s = scene();
        s.cameras[0].rotation[1][2] = f64::NAN;
        assert!(!s.cameras[0].is_finite());
        let e = s.to_json().unwrap_err();
        assert!(e.contains("scene.cameras[0].rotation[1][2]"), "{e}");
        let mut s = scene();
        s.fps = f64::INFINITY;
        assert!(s.to_json().unwrap_err().contains("scene.scene.fps"));
    }

    #[test]
    fn the_script_reads_this_format() {
        assert!(BUILD_SCENE_PY.contains(&format!("FORMAT = {FORMAT}")));
    }

    #[test]
    fn bodies_and_points_make_format_2_with_nulls() {
        let mut s = scene();
        let m = [
            [1.0, 0.0, 0.0, 5.0],
            [0.0, 1.0, 0.0, -0.0],
            [0.0, 0.0, 1.0, 0.25],
            [0.0, 0.0, 0.0, 1.0],
        ];
        s.point_size = 2.5;
        s.bodies.push(BlenderBody {
            name: "Fémur \"a/b\"".into(),
            number: 3,
            filtered: true,
            first_frame: 10,
            matrices: vec![Some(m), None],
            mesh: Some(BlenderMesh {
                path: "meshes/Fémur _a_b_.obj".into(),
                scale: 0.5,
                source: "bone.obj".into(),
            }),
        });
        s.points.push(BlenderPoint {
            name: "Marker 1".into(),
            number: 1,
            filtered: false,
            first_frame: 10,
            positions: vec![None, Some([1.0, 2.0, 3.5])],
        });
        let j = s.to_json().unwrap();
        assert!(j.contains("\"format\": 2,"), "{j}");
        assert!(j.contains("\"point_size\": 2.5,"), "{j}");
        assert!(j.contains("\"name\": \"Fémur \\\"a/b\\\"\""), "{j}");
        assert!(
            j.contains("[[1, 0, 0, 5], [0, 1, 0, 0], [0, 0, 1, 0.25], [0, 0, 0, 1]],\n"),
            "{j}"
        );
        assert!(j.contains("\"matrices\": [\n"), "{j}");
        assert!(j.contains("null\n"), "{j}");
        assert!(j.contains("[1, 2, 3.5]"), "{j}");
        assert!(j.contains("\"path\": \"meshes/Fémur _a_b_.obj\""), "{j}");
        // A non-finite pose is refused, named by its place.
        s.bodies[0].matrices[0].as_mut().unwrap()[2][3] = f64::NAN;
        let e = s.to_json().unwrap_err();
        assert!(e.contains("scene.bodies[0].matrices[0][2][3]"), "{e}");
        assert!(s.readme(None).contains("Rigid bodies"));
    }

    #[test]
    fn the_script_reads_format_2() {
        assert!(BUILD_SCENE_PY.contains(&format!("FORMAT_ANIMATED = {FORMAT_ANIMATED}")));
    }
}
