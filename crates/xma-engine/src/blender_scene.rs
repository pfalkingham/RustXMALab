//! File > Export > Blender scene (new in RustXMALab; parity ledger F85): a folder that
//! Blender turns into a scene with the calibrated cameras and their image planes, and,
//! when asked, Blender run in the background to save it as a `.blend`.
//!
//! The folder (`xma_io::export::blender_scene` describes the files and `scene.json`):
//!
//! ```text
//! <output>/<name>/build_scene.py, scene.json, README.txt, images/cameraN/…, [<name>.blend]
//! ```
//!
//! Two kinds, as the author named them ([`BlenderSceneKind`]):
//! - **Just cameras**: every calibrated camera at its pose for one calibration image (by
//!   default the first that is calibrated in every camera), each with that calibration
//!   image, undistorted as the window shows it ([`crate::undistorted_image`]), as a PNG (or
//!   no image, the planes empty).
//! - **Trial N**: every calibrated camera at the pose of the trial's reference
//!   calibration image (`view::world::trial_camera_pose`), each with the trial's
//!   undistorted images of a range of frames as a JPEG sequence written by
//!   [`TrialImagesJob`] (or no images); Blender frame f shows trial frame f.
//!
//! With a trial, the scene can also hold (stage 2; [`BlenderSceneSpec`]):
//! - **rigid bodies** (`rigid_bodies`): each body with a pose at a covered frame, an Empty
//!   keyed with the matrix File › Export › Rigid body transformations writes (body to world,
//!   ledger F3). The bodies are recomputed first, as that export does (ledger A36), on a copy
//!   of the document (the project is not changed); `filtered` takes the filtered poses,
//!   refused as that export refuses them (no recording speed, or a body's cutoff not above 0).
//! - **bone meshes** (`meshes`): each exported body's OBJ file (`RigidBody::mesh_model`) that
//!   `xma_io::obj::parse_obj` can read, copied into `meshes/` and made the body's child with
//!   its mesh scale; asking for meshes exports the bodies too. A mesh that cannot be read is
//!   left out and the report says why. The files are read in [`BlenderSceneJob::run`] (on the
//!   job's thread, not in [`BlenderSceneJob::new`], which the window calls on its interface
//!   thread), and only a regular file of at most [`MESH_SIZE_LIMIT`] bytes (256 MiB) is
//!   opened, as its metadata says before it is opened: a folder, a pipe (FIFO), a device
//!   such as `/dev/zero` or a larger file is left out with a sentence, so a project's mesh
//!   path cannot freeze the run or fill the memory. 256 MiB is 18 times the pig's largest
//!   mesh (14 MB), several million triangles, more than a bone mesh needs; the file is held
//!   once in memory while it is checked and written. (A file replaced by a pipe between the
//!   check and the opening is not covered.)
//! - **3D points** (`points`): each marker with a 3D point at a covered frame, keyed with the
//!   values File › Export › 3D points writes (filtered at the trial's cutoff when `filtered`,
//!   as that export's filtered variant; refused where that export refuses).
//!
//! The covered frames are the trial's `frames` when its images are written, else the whole
//! trial (frames 1 to its number of frames), and the scene's frame range is set to them.
//! With the cameras kind these options are ignored and the report says so: that scene shows
//! one calibration image, not a trial, so there is no pose or point to show.
//!
//! The work ([`BlenderSceneJob::run`]) writes into a partial folder,
//! `<output>/.<name>.partial-<process>-<number>` (a name new to each job), and renames it to
//! `<name>` only when every file is written: Cancel (or a failure) removes the partial
//! folder, so no half-written folder carries the finished name. The run makes the partial
//! folder itself (`create_dir`, which fails on any existing name) and deletes no folder it
//! did not make: should that name exist, the run stops with a sentence and leaves it. A run
//! stopped without Cancel (the program killed) leaves its partial folder for the user to
//! delete. A finished folder is never overwritten: an output folder that exists and is not
//! empty is refused. Blender, when asked, runs after the rename, on the finished folder: if
//! it is not found, fails, or takes longer than the time limit (10 minutes; it is then
//! stopped), the folder is still complete and the report says what was not done and the
//! command that runs the script by hand ([`by_hand_command`], quoted for the system's shell).
//!
//! Stopping Blender (time limit or Cancel) does not depend on what its launcher does: on
//! Unix Blender starts in a process group of its own and the whole group is killed (`kill -s
//! KILL -- -<group>`, the system's `kill` program, as the standard library cannot signal a
//! group); on Windows the process tree is ended with `taskkill /T /F`, then the process
//! itself. Either way the run waits at most [`READER_GRACE`] for Blender's output after the
//! process ended, so a child of a launcher that still holds the output cannot hold the run.
//! (On Unix, Blender in its own group does not receive the terminal's Ctrl-C sent to `xma`.)
//!
//! The two modes of the program write the same folder but for the rigid bodies' matrices,
//! which come from the bodies recomputed in each mode (ledger A36): on the pig the review
//! measured them differing by up to 2.5e-8 between the default and the XMALab 3 mode.

use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use xma_io::export::blender_scene::{
    BUILD_SCENE_PY, BlenderBody, BlenderCamera, BlenderImage, BlenderMesh, BlenderPoint,
    BlenderScene, BlenderSceneKind as IoKind, BlenderSequence, BlenderTrial,
};
use xma_io::export::points3d::filter_marker;
use xma_io::export::rigid_bodies::{can_export_filtered, transformation_matrix};
use xma_io::export::{cutoff_correction, safe_component};
use xma_vision::camera::rodrigues_to_matrix;
use xma_vision::visual_filter::VisualFilter;

use crate::Document;
use crate::trial_images::{TrialImageFormat, TrialImagesJob, TrialImagesSpec};
use crate::undistorted_image::{CalibrationImageSource, Image8};

/// Environment variable naming the Blender executable.
pub const BLENDER_ENV: &str = "XMALAB_BLENDER";

/// How long Blender may run before it is stopped.
pub const BLENDER_TIME_LIMIT: Duration = Duration::from_secs(600);

/// Most output kept from one Blender run, per stream, in bytes.
const OUTPUT_LIMIT: usize = 16 * 1024;

/// The largest mesh file read and copied, in bytes (256 MiB; the module documentation says
/// why).
pub const MESH_SIZE_LIMIT: u64 = 256 * 1024 * 1024;

/// How long the run waits for Blender's output once the process has ended (a launcher's
/// child may still hold it).
pub const READER_GRACE: Duration = Duration::from_secs(2);

/// Numbers the partial folders of this process.
static PARTIAL_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// What the scene shows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlenderSceneKind {
    /// The cameras at the poses of calibration image `calibration_image` (0-based; `None`:
    /// the first calibrated in every camera); `images`: that image, undistorted, on each
    /// camera's plane (else the planes are empty).
    Cameras {
        calibration_image: Option<usize>,
        images: bool,
    },
    /// Trial `trial` (0-based), its images of `frames` (0-based, inclusive; `None`: the
    /// images are left out, the planes empty).
    Trial {
        trial: usize,
        frames: Option<(usize, usize)>,
    },
}

/// Blender run by the export.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BlendOptions {
    /// The executable; `None`: searched for ([`locate_blender`]).
    pub blender: Option<PathBuf>,
    /// Pack the still images into the `.blend` (Blender cannot pack a sequence).
    pub pack: bool,
}

/// What is exported.
#[derive(Clone, Debug, PartialEq)]
pub struct BlenderSceneSpec {
    pub kind: BlenderSceneKind,
    /// The trial's rigid bodies, animated (trial kind only).
    pub rigid_bodies: bool,
    /// Filtered poses and 3D points instead of raw ones.
    pub filtered: bool,
    /// The bodies' bone meshes (implies the bodies).
    pub meshes: bool,
    /// The markers' 3D points, animated.
    pub points: bool,
    /// The image planes' distance from their cameras; `None`: 1.3 times the camera's
    /// distance from the world origin (as the author's Blender add-on).
    pub plane_distance: Option<f64>,
    /// Run Blender to save `<name>.blend`.
    pub blend: Option<BlendOptions>,
    /// The folder's (and the collection's) name; `None`: `<project>_cameras` or
    /// `<project>_<trial name>`.
    pub name: Option<String>,
}

impl BlenderSceneSpec {
    /// A spec of `kind` with every option at its default.
    #[must_use]
    pub fn new(kind: BlenderSceneKind) -> Self {
        BlenderSceneSpec {
            kind,
            rigid_bodies: false,
            filtered: false,
            meshes: false,
            points: false,
            plane_distance: None,
            blend: None,
            name: None,
        }
    }
}

/// Where a run is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlenderSceneProgress {
    /// Camera `camera` of `cameras` (positions in the run), image `frame` of `frames`.
    pub camera: usize,
    pub cameras: usize,
    pub frame: usize,
    pub frames: usize,
    /// Blender is running.
    pub blender: bool,
}

/// What a run did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BlenderSceneReport {
    /// The finished folder (when it was made).
    pub folder: Option<PathBuf>,
    /// Files written, inside the folder.
    pub written: Vec<PathBuf>,
    /// What was left out and why, one sentence each.
    pub left_out: Vec<String>,
    /// Problems with images (frames not written), one sentence each.
    pub problems: Vec<String>,
    /// The `.blend` Blender saved.
    pub blend: Option<PathBuf>,
    /// What happened about Blender, when it was asked for: what was not done and how to
    /// run the script by hand, or the run's last output.
    pub blender: Option<String>,
    /// The run stopped with this error; nothing was left behind under the folder's name.
    pub error: Option<String>,
    pub cancelled: bool,
    /// Seconds spent writing the folder, and running Blender.
    pub seconds_folder: f64,
    pub seconds_blender: f64,
    /// What was done other than asked, one sentence each (meshes asked for without
    /// bodies: the bodies are exported too).
    pub notes: Vec<String>,
    /// Rigid bodies, meshes and 3D points written into `scene.json`.
    pub bodies: usize,
    pub meshes: usize,
    pub points: usize,
}

/// How one camera's image is made.
#[derive(Clone, Debug)]
enum CameraImages {
    None,
    Calibration(CalibrationImageSource),
    Trial(TrialImagesJob),
}

/// One camera of the scene.
#[derive(Clone, Debug)]
struct CameraPart {
    camera: BlenderCamera,
    images: CameraImages,
    /// Trial files are mirrored relative to the held image.
    mirrored: bool,
}

/// The work of the export, taken from the document ([`BlenderSceneJob::run`] runs on any
/// thread).
#[derive(Clone, Debug)]
pub struct BlenderSceneJob {
    scene: BlenderScene,
    parts: Vec<CameraPart>,
    folder: PathBuf,
    partial: PathBuf,
    left_out: Vec<String>,
    notes: Vec<String>,
    blend: Option<BlendOptions>,
    time_limit: Duration,
    /// Bodies and points (their meshes' paths filled in, `meshes` saying from where).
    bodies: Vec<BlenderBody>,
    points: Vec<BlenderPoint>,
    /// Per body (by position in `bodies`): the OBJ file to read and copy, when it has a mesh.
    mesh_sources: Vec<Option<MeshSource>>,
}

/// A body's mesh file, read in [`BlenderSceneJob::run`].
#[derive(Clone, Debug)]
struct MeshSource {
    path: PathBuf,
    /// The body's number (1-based) and label, for the sentence when it is left out.
    number: usize,
    label: String,
}

/// The mesh file `path` when it is a regular file of at most `limit` bytes that
/// `xma_io::obj::parse_obj` reads: its bytes. Its metadata is checked before it is opened, so
/// a pipe or device is never opened, and at most `limit + 1` bytes are read.
fn read_mesh(path: &Path, limit: u64) -> Result<Vec<u8>, String> {
    let md = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if !md.is_file() {
        return Err(format!(
            "{} is not a regular file (a folder, pipe or device is not read)",
            path.display()
        ));
    }
    let too_big = || {
        format!(
            "{} is larger than the {} MiB a mesh may be",
            path.display(),
            limit / (1024 * 1024)
        )
    };
    if md.len() > limit {
        return Err(too_big());
    }
    let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut bytes = Vec::with_capacity(usize::try_from(md.len()).unwrap_or(0));
    f.take(limit.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() as u64 > limit {
        return Err(too_big());
    }
    xma_io::obj::parse_obj(&String::from_utf8_lossy(&bytes)).map_err(|e| e.to_string())?;
    Ok(bytes)
}

fn image_dir(number: usize) -> String {
    format!("images/camera{number}")
}

impl BlenderSceneJob {
    /// The work for `spec` on `doc`, into `<output>/<name>`; `files` holds each camera's
    /// trial files (as [`TrialImagesJob::new`] takes them; unused for the cameras kind).
    ///
    /// # Errors
    /// A sentence: no calibrated camera; no such trial or calibration image; a calibration
    /// image calibrated in no camera; a trial whose reference calibration image is
    /// calibrated in no camera; a frame range outside the trial; a folder name that cannot
    /// be one; an output folder that exists and is not empty (nothing is overwritten).
    pub fn new(
        doc: &Document,
        spec: BlenderSceneSpec,
        files: &[Vec<PathBuf>],
        output: &Path,
    ) -> Result<BlenderSceneJob, String> {
        let p = doc.project();
        let calibrated: Vec<usize> = (0..p.cameras.len())
            .filter(|&c| p.cameras[c].calibrated)
            .collect();
        if calibrated.is_empty() {
            return Err("no camera is calibrated: a Blender scene needs calibrated cameras".into());
        }
        let mut left_out = Vec::new();
        for (c, cam) in p.cameras.iter().enumerate() {
            if !cam.calibrated {
                left_out.push(format!(
                    "Camera {} (\"{}\"): not calibrated",
                    c + 1,
                    cam.name
                ));
            }
        }
        let mut notes = Vec::new();
        let animated = spec.rigid_bodies || spec.meshes || spec.points;
        if matches!(spec.kind, BlenderSceneKind::Cameras { .. }) && (animated || spec.filtered) {
            left_out.push(
                "rigid bodies, meshes and 3D points: ignored, as a scene of just cameras \
                 shows one calibration image and no trial (they are exported with a trial \
                 scene, stage 2 of the Blender scene)"
                    .into(),
            );
        }
        let stem = project_stem(doc);
        // The calibration image of the poses, the kind for scene.json, the frame range.
        let (image, io_kind, frames, fps, trial_name) = match &spec.kind {
            BlenderSceneKind::Cameras {
                calibration_image, ..
            } => {
                let n_images = calibrated
                    .iter()
                    .map(|&c| p.cameras[c].calibration_images.len())
                    .max()
                    .unwrap_or(0);
                let i = match calibration_image {
                    Some(i) if *i >= n_images => {
                        return Err(format!(
                            "no calibration image {}: the cameras have {n_images}",
                            i + 1
                        ));
                    }
                    Some(i) => *i,
                    None => (0..n_images)
                        .find(|&i| {
                            calibrated.iter().all(|&c| {
                                p.cameras[c]
                                    .calibration_images
                                    .get(i)
                                    .is_some_and(xma_model::CalibrationImage::is_calibrated)
                            })
                        })
                        .ok_or_else(|| {
                            "no calibration image is calibrated in every camera: choose one"
                                .to_owned()
                        })?,
                };
                (
                    i,
                    IoKind::Cameras {
                        calibration_image: i + 1,
                    },
                    (1, 1),
                    0.0,
                    None,
                )
            }
            BlenderSceneKind::Trial { trial, frames } => {
                let t = p
                    .trials
                    .get(*trial)
                    .ok_or_else(|| format!("no trial {}", trial + 1))?;
                if let Some((a, b)) = frames {
                    if t.is_default {
                        return Err("the Default trial has no images: leave them out".into());
                    }
                    if a > b || *b >= t.n_frames {
                        return Err(format!(
                            "frames {} to {}: the trial has frames 1 to {}",
                            a + 1,
                            b + 1,
                            t.n_frames
                        ));
                    }
                }
                let range = match frames {
                    Some((a, b)) => (a + 1, b + 1),
                    None if animated && t.n_frames > 0 => (1, t.n_frames),
                    None => (t.start_frame.max(1), t.end_frame.max(t.start_frame.max(1))),
                };
                (
                    t.reference_calibration_image,
                    IoKind::Trial(BlenderTrial {
                        number: trial + 1,
                        name: t.name.clone(),
                        first_frame: range.0,
                        last_frame: range.1,
                    }),
                    range,
                    if t.recording_speed > 0.0 {
                        t.recording_speed
                    } else {
                        30.0
                    },
                    Some(t.name.as_str()),
                )
            }
        };
        let name = folder_name(&stem, trial_name, spec.name.as_deref())?;
        let folder = output.join(&name);
        if let Ok(mut d) = std::fs::read_dir(&folder)
            && d.next().is_some()
        {
            return Err(format!(
                "{} exists and is not empty: choose another name or folder (nothing is \
                 overwritten)",
                folder.display()
            ));
        }
        if folder.exists() && !folder.is_dir() {
            return Err(format!("{} exists and is not a folder", folder.display()));
        }
        // A name new to this job (the run refuses it, deleting nothing, should it exist).
        let partial = output.join(format!(
            ".{name}.partial-{}-{}",
            std::process::id(),
            PARTIAL_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));

        let mut parts = Vec::new();
        for &c in &calibrated {
            let cam = &p.cameras[c];
            let number = c + 1;
            let Some(ci) = cam
                .calibration_images
                .get(image)
                .filter(|i| i.is_calibrated())
            else {
                left_out.push(format!(
                    "Camera {number} (\"{}\"): calibration image {} is not calibrated in it",
                    cam.name,
                    image + 1
                ));
                continue;
            };
            let r = rodrigues_to_matrix(&ci.rotation);
            let t = ci.translation;
            let k = &cam.camera_matrix;
            let distance = spec
                .plane_distance
                .unwrap_or_else(|| 1.3 * (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt());
            let camera = BlenderCamera {
                name: format!("Camera{number}"),
                label: cam.name.clone(),
                number,
                width: cam.width,
                height: cam.height,
                fx: k[(0, 0)],
                fy: k[(1, 1)],
                cx: k[(0, 2)],
                cy: k[(1, 2)],
                rotation: std::array::from_fn(|i| std::array::from_fn(|j| r[(i, j)])),
                translation: [t[0], t[1], t[2]],
                flipped: cam.flipped,
                undistorted: false,
                plane_distance: distance,
                image: None,
            };
            if !camera.is_finite() || camera.fx <= 0.0 || camera.fy <= 0.0 || distance <= 0.0 {
                left_out.push(format!(
                    "Camera {number} (\"{}\"): its calibration holds a number that is not \
                     finite or a focal length or plane distance not above 0",
                    cam.name
                ));
                continue;
            }
            if cam.width == 0 || cam.height == 0 {
                left_out.push(format!(
                    "Camera {number} (\"{}\"): its size is not known",
                    cam.name
                ));
                continue;
            }
            let (images, undistorted, mirrored) = match &spec.kind {
                BlenderSceneKind::Cameras { images: false, .. } => {
                    (CameraImages::None, false, false)
                }
                BlenderSceneKind::Cameras { images: true, .. } => {
                    match CalibrationImageSource::new(doc, c, image) {
                        Ok(s) => {
                            let u = s.undistorts();
                            (CameraImages::Calibration(s), u, false)
                        }
                        Err(e) => {
                            left_out.push(format!("Camera {number}: its image: {e}"));
                            (CameraImages::None, false, false)
                        }
                    }
                }
                BlenderSceneKind::Trial {
                    trial,
                    frames: Some((a, b)),
                } => {
                    let ts = TrialImagesSpec {
                        trial: *trial,
                        first: *a,
                        last: *b,
                        format: TrialImageFormat::Jpg,
                        filtered: false,
                        filter: VisualFilter::default(),
                        camera: Some(c),
                    };
                    match TrialImagesJob::new(doc, ts, files, &partial.join(image_dir(number))) {
                        Ok(job) => {
                            let o = job.orientation(doc);
                            let (u, m) = o.first().map_or((false, false), |x| (x.1, x.2));
                            (CameraImages::Trial(job), u, m)
                        }
                        Err(e) => {
                            left_out.push(format!("Camera {number}: its images: {e}"));
                            (CameraImages::None, false, false)
                        }
                    }
                }
                BlenderSceneKind::Trial { frames: None, .. } => (CameraImages::None, false, false),
            };
            parts.push(CameraPart {
                camera: BlenderCamera {
                    undistorted,
                    ..camera
                },
                images,
                mirrored,
            });
        }
        if parts.is_empty() {
            return Err(match &spec.kind {
                BlenderSceneKind::Trial { .. } => format!(
                    "the trial's reference calibration image ({}) is calibrated in no camera",
                    image + 1
                ),
                BlenderSceneKind::Cameras { .. } => {
                    format!("calibration image {} is calibrated in no camera", image + 1)
                }
            });
        }
        if matches!(spec.kind, BlenderSceneKind::Trial { frames: None, .. }) {
            left_out.push("the trial's images: left out as asked (the planes are empty)".into());
        }
        if matches!(spec.kind, BlenderSceneKind::Cameras { images: false, .. }) {
            left_out.push("the calibration image: left out as asked (the planes are empty)".into());
        }
        let (mut bodies, mut points, mut mesh_sources) = (Vec::new(), Vec::new(), Vec::new());
        if let BlenderSceneKind::Trial { trial, .. } = spec.kind
            && animated
        {
            let mut names: Vec<String> = parts
                .iter()
                .flat_map(|p| [p.camera.name.clone(), format!("{}_plane", p.camera.name)])
                .collect();
            let range = frames.0 - 1..frames.1;
            if spec.meshes && !spec.rigid_bodies {
                notes.push(
                    "meshes were asked for without rigid bodies: the bodies are exported too \
                     (a mesh moves with its body)"
                        .into(),
                );
            }
            if spec.rigid_bodies || spec.meshes {
                let (b, m) = trial_bodies(
                    doc,
                    trial,
                    range.clone(),
                    spec.filtered,
                    spec.meshes,
                    &mut names,
                    &mut left_out,
                )?;
                bodies = b;
                mesh_sources = m;
            }
            if spec.points {
                points = trial_points(doc, trial, range, spec.filtered, &mut names, &mut left_out)?;
            }
        }
        let distance: Vec<f64> = parts
            .iter()
            .map(|p| {
                let t = p.camera.translation;
                (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt()
            })
            .collect();
        let point_size = POINT_SIZE * distance.iter().sum::<f64>() / distance.len().max(1) as f64;
        let scene = BlenderScene {
            generator: format!("RustXMALab {}", env!("CARGO_PKG_VERSION")),
            name: name.clone(),
            project: doc
                .path()
                .and_then(Path::file_name)
                .map_or_else(String::new, |s| s.to_string_lossy().into_owned()),
            kind: io_kind,
            frame_start: frames.0,
            frame_end: frames.1,
            fps,
            cameras: Vec::new(),
            point_size: if point_size.is_finite() && point_size > 0.0 {
                point_size
            } else {
                1.0
            },
            bodies: Vec::new(),
            points: Vec::new(),
        };
        Ok(BlenderSceneJob {
            scene,
            parts,
            folder,
            partial,
            left_out,
            notes,
            blend: spec.blend,
            time_limit: BLENDER_TIME_LIMIT,
            bodies,
            points,
            mesh_sources,
        })
    }

    /// The finished folder.
    #[must_use]
    pub fn folder(&self) -> &Path {
        &self.folder
    }

    /// Replaces the time Blender may run (tests).
    #[must_use]
    pub fn with_blender_time_limit(mut self, limit: Duration) -> Self {
        self.time_limit = limit;
        self
    }

    /// Writes the folder, then runs Blender when asked, calling `progress` as it goes and
    /// stopping before the next image once `cancel` returns true (Blender is stopped too).
    pub fn run(
        &self,
        progress: &mut dyn FnMut(BlenderSceneProgress),
        cancel: &dyn Fn() -> bool,
    ) -> BlenderSceneReport {
        let started = Instant::now();
        let mut rep = BlenderSceneReport {
            left_out: self.left_out.clone(),
            notes: self.notes.clone(),
            ..BlenderSceneReport::default()
        };
        let fail = |rep: &mut BlenderSceneReport, partial: &Path, e: String| {
            let _ = std::fs::remove_dir_all(partial);
            rep.error = Some(e);
            rep.written.clear();
        };
        // The partial folder is made here, by `create_dir`, which fails on an existing name:
        // a folder this run did not make is never written into or deleted.
        let made = self
            .partial
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|()| std::fs::create_dir(&self.partial));
        if let Err(e) = made {
            rep.error = Some(if e.kind() == std::io::ErrorKind::AlreadyExists {
                format!(
                    "{} exists and was not made by this run: it is left as it is (nothing \
                     was written); export again",
                    self.partial.display()
                )
            } else {
                format!("{}: {e}", self.partial.display())
            });
            return rep;
        }
        let mut scene = self.scene.clone();
        let n = self.parts.len();
        let mut written: Vec<PathBuf> = Vec::new();
        for (k, part) in self.parts.iter().enumerate() {
            if cancel() {
                fail(&mut rep, &self.partial, "cancelled".into());
                rep.error = None;
                rep.cancelled = true;
                return rep;
            }
            let number = part.camera.number;
            let mut camera = part.camera.clone();
            match &part.images {
                CameraImages::None => {}
                CameraImages::Calibration(src) => {
                    progress(BlenderSceneProgress {
                        camera: k,
                        cameras: n,
                        frame: 0,
                        frames: 1,
                        blender: false,
                    });
                    let rel = format!(
                        "{}/camera{number}_calibration_image{}.png",
                        image_dir(number),
                        match &scene.kind {
                            IoKind::Cameras { calibration_image } => *calibration_image,
                            IoKind::Trial(_) => 0,
                        }
                    );
                    let path = self.partial.join(&rel);
                    match src.undistort().and_then(|img| write_png(&path, &img)) {
                        Ok(()) => {
                            written.push(PathBuf::from(&rel));
                            camera.image = Some(BlenderImage {
                                path: rel,
                                sequence: None,
                                mirrored: false,
                            });
                        }
                        Err(e) => rep.left_out.push(format!(
                            "Camera {number}: its image: {e} (the plane is empty)"
                        )),
                    }
                }
                CameraImages::Trial(job) => {
                    let r = job.run(
                        &mut |p| {
                            progress(BlenderSceneProgress {
                                camera: k,
                                cameras: n,
                                frame: p.frame,
                                frames: p.frames,
                                blender: false,
                            });
                        },
                        cancel,
                    );
                    if r.cancelled {
                        fail(&mut rep, &self.partial, "cancelled".into());
                        rep.error = None;
                        rep.cancelled = true;
                        return rep;
                    }
                    rep.problems.extend(r.problems());
                    let files: Vec<PathBuf> =
                        r.cameras.iter().flat_map(|c| c.written.clone()).collect();
                    let rels: Vec<PathBuf> = files
                        .iter()
                        .filter_map(|f| f.strip_prefix(&self.partial).ok().map(Path::to_path_buf))
                        .collect();
                    if let (Some(first), IoKind::Trial(t)) = (rels.first(), &scene.kind) {
                        camera.image = Some(BlenderImage {
                            path: slash_path(first),
                            sequence: Some(BlenderSequence {
                                first: t.first_frame,
                                count: t.last_frame + 1 - t.first_frame,
                            }),
                            mirrored: part.mirrored,
                        });
                    } else {
                        rep.left_out.push(format!(
                            "Camera {number}: no image was written (the plane is empty)"
                        ));
                    }
                    written.extend(rels);
                }
            }
            scene.cameras.push(camera);
        }
        // The bodies' meshes, read here (on the job's thread) and written into meshes/.
        let mut bodies = self.bodies.clone();
        for (body, src) in bodies.iter_mut().zip(&self.mesh_sources) {
            let (Some(src), Some(mesh)) = (src, body.mesh.as_ref()) else {
                continue;
            };
            if cancel() {
                fail(&mut rep, &self.partial, "cancelled".into());
                rep.error = None;
                rep.cancelled = true;
                return rep;
            }
            let bytes = match read_mesh(&src.path, MESH_SIZE_LIMIT) {
                Ok(b) => b,
                Err(e) => {
                    rep.left_out.push(format!(
                        "rigid body {} (\"{}\"): its mesh is left out: {e}",
                        src.number, src.label
                    ));
                    body.mesh = None;
                    continue;
                }
            };
            let dest = self.partial.join(&mesh.path);
            let copied = dest
                .parent()
                .map_or(Ok(()), std::fs::create_dir_all)
                .and_then(|()| std::fs::write(&dest, &bytes));
            match copied {
                Ok(()) => written.push(PathBuf::from(&mesh.path)),
                Err(e) => {
                    rep.left_out.push(format!(
                        "{}: its mesh {}: {e} (the body is exported without it)",
                        body.name,
                        src.path.display()
                    ));
                    body.mesh = None;
                }
            }
        }
        rep.bodies = bodies.len();
        rep.meshes = bodies.iter().filter(|b| b.mesh.is_some()).count();
        rep.points = self.points.len();
        scene.bodies = bodies;
        scene.points.clone_from(&self.points);
        let json = match scene.to_json() {
            Ok(j) => j,
            Err(e) => {
                fail(&mut rep, &self.partial, format!("scene.json: {e}"));
                return rep;
            }
        };
        let readme = scene.readme(None);
        for (name, text) in [
            ("scene.json", json.as_str()),
            ("build_scene.py", BUILD_SCENE_PY),
            ("README.txt", readme.as_str()),
        ] {
            if let Err(e) = std::fs::write(self.partial.join(name), text) {
                fail(&mut rep, &self.partial, format!("{name}: {e}"));
                return rep;
            }
            written.push(PathBuf::from(name));
        }
        if cancel() {
            fail(&mut rep, &self.partial, "cancelled".into());
            rep.error = None;
            rep.cancelled = true;
            return rep;
        }
        // The finished name: an empty folder there (made since) is replaced, a full one not.
        if self.folder.is_dir() {
            let _ = std::fs::remove_dir(&self.folder);
        }
        if let Err(e) = std::fs::rename(&self.partial, &self.folder) {
            fail(
                &mut rep,
                &self.partial,
                format!("{}: {e} (nothing is overwritten)", self.folder.display()),
            );
            return rep;
        }
        rep.folder = Some(self.folder.clone());
        rep.written = written;
        rep.seconds_folder = started.elapsed().as_secs_f64();

        if let Some(opts) = &self.blend {
            let t0 = Instant::now();
            progress(BlenderSceneProgress {
                camera: n,
                cameras: n,
                frame: 0,
                frames: 0,
                blender: true,
            });
            let blend_name = format!("{}.blend", scene.name);
            let by_hand = format!(
                "to make the .blend by hand, run: {}",
                by_hand_command(&self.folder, &blend_name, opts.pack)
            );
            match locate_blender(opts.blender.as_deref()) {
                Err(e) => {
                    rep.blender = Some(format!("No .blend was made: {e}; {by_hand}"));
                }
                Ok(exe) => {
                    match run_blender(
                        &exe,
                        &self.folder,
                        &blend_name,
                        opts.pack,
                        self.time_limit,
                        cancel,
                    ) {
                        Ok(out) => {
                            let blend = self.folder.join(&blend_name);
                            if blend.is_file() {
                                let _ = std::fs::write(
                                    self.folder.join("README.txt"),
                                    scene.readme(Some(&blend_name)),
                                );
                                rep.written.push(PathBuf::from(&blend_name));
                                rep.blend = Some(blend);
                                rep.blender =
                                    Some(format!("{} ran: {}", exe.display(), script_lines(&out)));
                            } else {
                                rep.blender = Some(format!(
                                    "{} ran but saved no .blend: {}; {by_hand}",
                                    exe.display(),
                                    last_lines(&out, 12)
                                ));
                            }
                        }
                        Err(e) => {
                            rep.cancelled = cancel();
                            rep.blender = Some(format!("No .blend was made: {e}; {by_hand}"));
                        }
                    }
                }
            }
            rep.seconds_blender = t0.elapsed().as_secs_f64();
        }
        rep
    }
}

/// The project file's stem, which begins the folder's default name (`project` for a project
/// not saved yet).
#[must_use]
pub fn project_stem(doc: &Document) -> String {
    doc.path().and_then(Path::file_stem).map_or_else(
        || "project".to_owned(),
        |s| s.to_string_lossy().into_owned(),
    )
}

/// The folder's (and the collection's) name the export writes: `name` when given, else
/// `<stem>_cameras` (no `trial`) or `<stem>_<trial name>`, made safe as every export makes a
/// name from the project safe (`xma_io::export::safe_component`, ledger F70: `/`, `"` and the
/// like become `_`).
///
/// # Errors
/// The sentence saying why the name cannot be a folder's (empty, `.` or `..`).
pub fn folder_name(stem: &str, trial: Option<&str>, name: Option<&str>) -> Result<String, String> {
    let default = match trial {
        Some(t) => format!("{stem}_{t}"),
        None => format!("{stem}_cameras"),
    };
    safe_component(name.unwrap_or(&default), false)
}

/// The command that makes `blend` from the finished `folder` by hand, for the shell of the
/// system the program runs on: `cd <folder> && blender --background --factory-startup
/// --python build_scene.py -- --save <blend> [--pack]`, the folder and the name quoted
/// ([`posix_quote`] on macOS and Linux, [`cmd_quote`] on Windows), so that pasting it into
/// that shell runs Blender and nothing else. Where `cmd` cannot quote a part (see
/// [`cmd_quote`]), the command is given with a sentence saying so instead of a quoted form.
#[must_use]
pub fn by_hand_command(folder: &Path, blend: &str, pack: bool) -> String {
    let folder = folder.to_string_lossy();
    let pack = if pack { " --pack" } else { "" };
    if cfg!(windows) {
        match (cmd_quote(&folder), cmd_quote(blend)) {
            (Some(f), Some(b)) => format!(
                "cd /d {f} && blender --background --factory-startup --python build_scene.py \
                 -- --save {b}{pack}"
            ),
            _ => format!(
                "blender --background --factory-startup --python build_scene.py -- --save \
                 <name>.blend{pack}, in the folder {folder}, the name being {blend} (the folder \
                 or the name holds a character, % \" or a line break, that cmd cannot take \
                 in quotes: type them yourself)"
            ),
        }
    } else {
        format!(
            "cd {} && blender --background --factory-startup --python build_scene.py -- \
             --save {}{pack}",
            posix_quote(&folder),
            posix_quote(blend)
        )
    }
}

/// `s` quoted for a POSIX shell (sh, bash, zsh): in single quotes, each `'` written as
/// `'\''`. Inside single quotes the shell expands nothing (`$( )`, backticks, `$`, `\`,
/// `"`, `*` and spaces are taken as they are), so the word is `s` exactly. A line break is
/// kept as it is: the command then shows on two lines, and pasted whole it is still one
/// word (the shell continues the quote on the next line).
#[must_use]
pub fn posix_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

/// `s` quoted for Windows' `cmd`: in double quotes, where `&`, `|`, `<`, `>`, `^`, `(`, `)`
/// and spaces are taken as they are. `None` when `s` holds what `cmd` cannot quote: `"`
/// (ends the quotes; Windows file names cannot hold it), `%` (`cmd` expands `%NAME%` even in
/// quotes and has no escape for it at its prompt), or a control character such as a line
/// break. Not covered: `!` is expanded when delayed expansion is on (off by default), and
/// PowerShell, whose quoting differs.
#[must_use]
pub fn cmd_quote(s: &str) -> Option<String> {
    if s.chars().any(|c| c == '"' || c == '%' || c.is_control()) {
        return None;
    }
    Some(format!("\"{s}\""))
}

/// Why `filtered` is refused for trial `trial` (0-based) with its rigid bodies (`bodies`,
/// also when only meshes are asked for) or its 3D points (`points`), as
/// [`BlenderSceneJob::new`] refuses it: the sentence it returns, or `None` when the filtered
/// variant can be exported. The bodies are filtered as File › Export › Rigid body
/// transformations filters them (no recording speed or a body's cutoff not above 0: refused;
/// a trial without bodies is not refused), the points as the filtered 3D points (no
/// recording speed, or a cutoff the filter cannot take: refused; a trial without markers is
/// not refused). `None` too for a trial that does not exist.
#[must_use]
pub fn filtered_refusal(
    doc: &Document,
    trial: usize,
    bodies: bool,
    points: bool,
) -> Option<String> {
    let t = doc.project().trials.get(trial)?;
    if bodies && !t.rigid_bodies.is_empty() {
        let all: Vec<usize> = (0..t.rigid_bodies.len()).collect();
        if !can_export_filtered(t, &all, doc.compat()) {
            return Some(format!(
                "filtered rigid bodies: {}",
                crate::export::NO_SPEED_OR_CUTOFF
            ));
        }
    }
    if points {
        if t.recording_speed <= 0.0 {
            return Some(format!(
                "filtered 3D points: {}",
                crate::export::NO_RECORDING_SPEED
            ));
        }
        if !t.markers.is_empty()
            && !xma_vision::filter::can_filter(t.cutoff_frequency, t.recording_speed)
        {
            return Some(cannot_filter_points(
                1,
                t.cutoff_frequency,
                t.recording_speed,
            ));
        }
    }
    None
}

fn cannot_filter_points(marker: usize, cutoff: f64, speed: f64) -> String {
    format!(
        "filtered 3D points: cannot filter marker {marker} at {cutoff} Hz with a recording \
         speed of {speed} Hz"
    )
}

/// The points' display size as a fraction of the cameras' mean distance from the world
/// origin (bodies are drawn five times larger).
const POINT_SIZE: f64 = 0.004;

/// The longest object name, in bytes, that leaves room for `_mesh` and `_plane` within
/// Blender 4.2's 63.
const NAME_BYTES: usize = 57;

/// `name`, cut to [`NAME_BYTES`] at a character boundary and made different from every name
/// in `used` (and its `_mesh`, which a body's mesh takes) by ` (2)`, ` (3)`, ...; added to
/// `used`.
fn unique_name(name: &str, used: &mut Vec<String>) -> String {
    let cut = |s: &str, n: usize| -> String {
        let mut end = s.len().min(n);
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        s[..end].to_owned()
    };
    let base = cut(name.trim(), NAME_BYTES);
    let base = if base.is_empty() {
        "object".to_owned()
    } else {
        base
    };
    let taken = |n: &str, used: &[String]| {
        used.iter()
            .any(|u| u == n || *u == format!("{n}_mesh") || format!("{u}_mesh") == n)
    };
    let mut candidate = base.clone();
    let mut k = 2;
    while taken(&candidate, used) {
        let suffix = format!(" ({k})");
        candidate = format!("{}{suffix}", cut(&base, NAME_BYTES - suffix.len()));
        k += 1;
    }
    used.push(candidate.clone());
    candidate
}

/// The trial's rigid bodies over `range` (0-based frames), recomputed as the rigid-body
/// export recomputes them (ledger A36), on a copy of the document; with their meshes when
/// `meshes`. Per body, the mesh file to copy.
#[allow(clippy::type_complexity)]
fn trial_bodies(
    doc: &Document,
    trial: usize,
    range: std::ops::Range<usize>,
    filtered: bool,
    meshes: bool,
    names: &mut Vec<String>,
    left_out: &mut Vec<String>,
) -> Result<(Vec<BlenderBody>, Vec<Option<MeshSource>>), String> {
    let t = doc.trial(trial);
    let all: Vec<usize> = (0..t.rigid_bodies.len()).collect();
    if all.is_empty() {
        left_out.push("rigid bodies: the trial has none".into());
        return Ok((Vec::new(), Vec::new()));
    }
    if filtered && !can_export_filtered(t, &all, doc.compat()) {
        return Err(format!(
            "filtered rigid bodies: {}",
            crate::export::NO_SPEED_OR_CUTOFF
        ));
    }
    let mut copy = doc.clone();
    copy.recompute_rigid_bodies_for_export(trial, &all, filtered);
    let t = copy.trial(trial);
    let dir = doc.path().and_then(Path::parent);
    let mut bodies = Vec::new();
    let mut sources = Vec::new();
    let mut files: Vec<String> = Vec::new();
    for (b, rb) in t.rigid_bodies.iter().enumerate() {
        let label = if rb.description.is_empty() {
            format!("Rigid body {}", b + 1)
        } else {
            rb.description.clone()
        };
        let track = if filtered { &rb.filtered } else { &rb.raw };
        let matrices: Vec<Option<[[f64; 4]; 4]>> = range
            .clone()
            .map(|f| {
                transformation_matrix(track, f).map(|m| {
                    // Column-major as the export writes it; rows here.
                    std::array::from_fn(|i| std::array::from_fn(|j| m[j * 4 + i]))
                })
            })
            .map(|m| m.filter(|m| m.iter().flatten().all(|v| v.is_finite())))
            .collect();
        if matrices.iter().all(Option::is_none) {
            left_out.push(format!(
                "rigid body {} (\"{label}\"): no {}pose at frames {} to {}",
                b + 1,
                if filtered { "filtered " } else { "" },
                range.start + 1,
                range.end
            ));
            continue;
        }
        let name = unique_name(&label, names);
        let mut mesh = None;
        let mut source = None;
        if meshes {
            match rb.mesh_model.as_deref().filter(|m| !m.is_empty()) {
                None => {}
                Some(stored) => {
                    let path = crate::resolve_stored_path(dir, stored);
                    // The file is read and checked in `run`, on the job's thread.
                    let scale_ok = rb.mesh_scale.is_finite() && rb.mesh_scale > 0.0;
                    if scale_ok {
                        let file = unique_file(&name, &mut files);
                        mesh = Some(BlenderMesh {
                            path: format!("meshes/{file}"),
                            scale: rb.mesh_scale,
                            source: path
                                .file_name()
                                .map_or_else(String::new, |s| s.to_string_lossy().into_owned()),
                        });
                        source = Some(MeshSource {
                            path,
                            number: b + 1,
                            label: label.clone(),
                        });
                    } else {
                        left_out.push(format!(
                            "rigid body {} (\"{label}\"): its mesh is left out: its mesh scale \
                             {} is not above 0",
                            b + 1,
                            rb.mesh_scale
                        ));
                    }
                }
            }
        }
        bodies.push(BlenderBody {
            name,
            number: b + 1,
            filtered,
            first_frame: range.start + 1,
            matrices,
            mesh,
        });
        sources.push(source);
    }
    Ok((bodies, sources))
}

/// A file name in `meshes/` for the mesh of body `name`: a safe name, different from the
/// others in `used` without regard to case (folders on Windows and macOS ignore it).
fn unique_file(name: &str, used: &mut Vec<String>) -> String {
    let stem = safe_component(name, false).unwrap_or_else(|_| "mesh".to_owned());
    let mut file = format!("{stem}.obj");
    let mut k = 2;
    while used.iter().any(|u| u.to_lowercase() == file.to_lowercase()) {
        file = format!("{stem} ({k}).obj");
        k += 1;
    }
    used.push(file.clone());
    file
}

/// The trial's markers' 3D points over `range` (0-based frames): what the 3D-point export
/// writes, filtered at the trial's cutoff when `filtered` (refused where that export
/// refuses).
fn trial_points(
    doc: &Document,
    trial: usize,
    range: std::ops::Range<usize>,
    filtered: bool,
    names: &mut Vec<String>,
    left_out: &mut Vec<String>,
) -> Result<Vec<BlenderPoint>, String> {
    let t = doc.trial(trial);
    if filtered && t.recording_speed <= 0.0 {
        return Err(format!(
            "filtered 3D points: {}",
            crate::export::NO_RECORDING_SPEED
        ));
    }
    let mut out = Vec::new();
    for (i, m) in t.markers.iter().enumerate() {
        let label = if m.description.is_empty() {
            format!("Marker {}", i + 1)
        } else {
            m.description.clone()
        };
        let series = if filtered {
            filter_marker(
                m,
                t.cutoff_frequency,
                t.recording_speed,
                t.n_frames.min(m.status3d.len()),
                cutoff_correction(doc.compat()),
            )
            .ok_or_else(|| cannot_filter_points(i + 1, t.cutoff_frequency, t.recording_speed))?
        } else {
            (m.points3d.clone(), m.status3d.clone())
        };
        let positions: Vec<Option<[f64; 3]>> = range
            .clone()
            .map(|f| {
                let ok = series.1.get(f).is_some_and(|s| s.code() > 0);
                let p = series.0.get(f).filter(|_| ok)?;
                let v = [p.x, p.y, p.z];
                v.iter().all(|x| x.is_finite()).then_some(v)
            })
            .collect();
        if positions.iter().all(Option::is_none) {
            left_out.push(format!(
                "marker {} (\"{label}\"): no 3D point at frames {} to {}",
                i + 1,
                range.start + 1,
                range.end
            ));
            continue;
        }
        out.push(BlenderPoint {
            name: unique_name(&label, names),
            number: i + 1,
            filtered,
            first_frame: range.start + 1,
            positions,
        });
    }
    Ok(out)
}

fn slash_path(p: &Path) -> String {
    p.components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

/// The lines `build_scene.py` printed.
fn script_lines(s: &str) -> String {
    s.lines()
        .filter_map(|l| l.strip_prefix("build_scene: "))
        .collect::<Vec<_>>()
        .join(" | ")
}

fn last_lines(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().filter(|l| !l.trim().is_empty()).collect();
    lines[lines.len().saturating_sub(n)..].join(" | ")
}

/// Writes an 8-bit grey or RGB PNG (a temporary name, then renamed).
fn write_png(path: &Path, img: &Image8) -> Result<(), String> {
    use image::ImageEncoder as _;
    let colour = match img.channels {
        1 => image::ExtendedColorType::L8,
        3 => image::ExtendedColorType::Rgb8,
        n => return Err(format!("{n} channels")),
    };
    let mut bytes = Vec::new();
    image::codecs::png::PngEncoder::new(&mut bytes)
        .write_image(&img.data, img.width as u32, img.height as u32, colour)
        .map_err(|e| e.to_string())?;
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    crate::undistort_sequence::write_whole(path, &bytes)
}

/// Finds the Blender executable, in order: `explicit`, the environment variable
/// `XMALAB_BLENDER`, `blender` on `PATH`, then the usual places on macOS
/// (`/Applications/Blender.app`) and Windows (`Program Files\Blender Foundation\Blender
/// *\blender.exe`, the newest). As `xma_video::ffmpeg::FfmpegTools::locate`.
///
/// # Errors
/// A sentence saying where it was looked for.
pub fn locate_blender(explicit: Option<&Path>) -> Result<PathBuf, String> {
    if let Some(p) = explicit {
        return if p.is_file() {
            Ok(p.to_path_buf())
        } else {
            Err(format!(
                "Blender not found at the path given, {}",
                p.display()
            ))
        };
    }
    if let Some(v) = std::env::var_os(BLENDER_ENV).filter(|v| !v.is_empty()) {
        let p = PathBuf::from(v);
        return if p.is_file() {
            Ok(p)
        } else {
            Err(format!(
                "Blender not found at {} (set by {BLENDER_ENV})",
                p.display()
            ))
        };
    }
    let file = format!("blender{}", std::env::consts::EXE_SUFFIX);
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            let c = dir.join(&file);
            if c.is_file() {
                return Ok(c);
            }
        }
    }
    if cfg!(target_os = "macos") {
        let c = PathBuf::from("/Applications/Blender.app/Contents/MacOS/Blender");
        if c.is_file() {
            return Ok(c);
        }
    }
    if cfg!(windows) {
        let mut roots: Vec<PathBuf> = ["ProgramFiles", "ProgramW6432"]
            .iter()
            .filter_map(std::env::var_os)
            .map(PathBuf::from)
            .collect();
        roots.push(PathBuf::from(r"C:\Program Files"));
        let mut found: Vec<(Vec<u32>, PathBuf)> = Vec::new();
        for r in roots {
            let Ok(dir) = std::fs::read_dir(r.join("Blender Foundation")) else {
                continue;
            };
            for e in dir.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                let exe = e.path().join("blender.exe");
                if name.starts_with("Blender") && exe.is_file() {
                    let version = name
                        .split(|c: char| !c.is_ascii_digit())
                        .filter_map(|s| s.parse().ok())
                        .collect();
                    found.push((version, exe));
                }
            }
        }
        found.sort();
        if let Some((_, exe)) = found.pop() {
            return Ok(exe);
        }
    }
    Err(format!(
        "Blender not found: install Blender 4.2 or later, put {file} on PATH, or set \
         {BLENDER_ENV} to its full path"
    ))
}

/// Runs `build_scene.py` in a headless Blender in `folder` to save `blend`; its output
/// (stdout, then stderr; the last part of each) when it ends with status 0.
fn run_blender(
    exe: &Path,
    folder: &Path,
    blend: &str,
    pack: bool,
    limit: Duration,
    cancel: &dyn Fn() -> bool,
) -> Result<String, String> {
    let folder = &std::path::absolute(folder).map_err(|e| format!("{}: {e}", folder.display()))?;
    let mut cmd = Command::new(exe);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd.current_dir(folder)
        .arg("--background")
        .arg("--factory-startup")
        .arg("--python-exit-code")
        .arg("1")
        .arg("--python")
        .arg(folder.join("build_scene.py"))
        .arg("--")
        .arg("--save")
        .arg(blend);
    if pack {
        cmd.arg("--pack");
    }
    // Its own process group, so that stopping it stops whatever its launcher started.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        cmd.process_group(0);
    }
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot start {}: {e}", exe.display()))?;
    // Each reader says when its pipe closed; the run waits for that at most READER_GRACE
    // after the process ended (a launcher's child may hold the pipes much longer).
    let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
    let collect = |pipe: Option<Box<dyn std::io::Read + Send>>| {
        let sink = Arc::new(Mutex::new(Vec::<u8>::new()));
        let s = Arc::clone(&sink);
        let done = done_tx.clone();
        let h = pipe.map(|mut pipe| {
            std::thread::spawn(move || {
                let _done = DoneOnDrop(done);
                let mut buf = [0u8; 4096];
                while let Ok(n) = pipe.read(&mut buf) {
                    if n == 0 {
                        break;
                    }
                    let mut v = s.lock().unwrap_or_else(|e| e.into_inner());
                    v.extend_from_slice(&buf[..n]);
                    if v.len() > 2 * OUTPUT_LIMIT {
                        let cut = v.len() - OUTPUT_LIMIT;
                        v.drain(..cut);
                    }
                }
            })
        });
        (sink, h)
    };
    let (out, h_out) = collect(
        child
            .stdout
            .take()
            .map(|p| Box::new(p) as Box<dyn std::io::Read + Send>),
    );
    let (err, h_err) = collect(
        child
            .stderr
            .take()
            .map(|p| Box::new(p) as Box<dyn std::io::Read + Send>),
    );
    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break Ok(s),
            Ok(None) => {}
            Err(e) => break Err(e.to_string()),
        }
        if cancel() {
            stop(&mut child);
            break Err("cancelled; Blender was stopped".to_owned());
        }
        if started.elapsed() > limit {
            stop(&mut child);
            break Err(format!(
                "Blender did not finish within {} s and was stopped",
                limit.as_secs()
            ));
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    drop(done_tx);
    let handles: Vec<_> = [h_out, h_err].into_iter().flatten().collect();
    let deadline = Instant::now() + READER_GRACE;
    for _ in 0..handles.len() {
        let left = deadline.saturating_duration_since(Instant::now());
        if done_rx.recv_timeout(left).is_err() {
            break;
        }
    }
    for h in handles {
        // A reader still blocked (a pipe held by a child of the launcher) is left to end
        // when the pipe closes; the output read so far is used.
        if h.is_finished() {
            let _ = h.join();
        }
    }
    let text = |s: &Arc<Mutex<Vec<u8>>>| {
        let v = s.lock().unwrap_or_else(|e| e.into_inner());
        let start = v.len().saturating_sub(OUTPUT_LIMIT);
        String::from_utf8_lossy(&v[start..]).into_owned()
    };
    let all = format!("{}\n{}", text(&out), text(&err));
    let status = status?;
    if status.success() {
        Ok(all)
    } else {
        Err(format!(
            "Blender ended with {status}: {}",
            last_lines(&all, 12)
        ))
    }
}

/// Sends on its channel when dropped (a reader thread ended, also by a panic).
struct DoneOnDrop(std::sync::mpsc::Sender<()>);

impl Drop for DoneOnDrop {
    fn drop(&mut self) {
        let _ = self.0.send(());
    }
}

/// Stops Blender and everything its launcher started: on Unix its process group (it was
/// started as the group's leader) with the system's `kill` program, as the standard library
/// cannot signal a group; on Windows its process tree with `taskkill /T /F`; then the
/// process itself, which is waited for. When `kill` or `taskkill` cannot be run, only the
/// process is stopped (a child it started may live on; the run still returns, as it waits
/// for the output a bounded time).
fn stop(child: &mut std::process::Child) {
    let id = child.id().to_string();
    #[cfg(unix)]
    let mut tree = {
        let mut c = Command::new("kill");
        c.arg("-s").arg("KILL").arg("--").arg(format!("-{id}"));
        c
    };
    #[cfg(windows)]
    let mut tree = {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        let mut c = Command::new("taskkill");
        c.args(["/PID", &id, "/T", "/F"]);
        c.creation_flags(CREATE_NO_WINDOW);
        c
    };
    #[cfg(any(unix, windows))]
    {
        let _ = tree
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    #[cfg(not(any(unix, windows)))]
    let _ = id;
    let _ = child.kill();
    let _ = child.wait();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_unique_and_short() {
        let mut used = vec!["Camera1".to_owned(), "Camera1_plane".to_owned()];
        assert_eq!(unique_name("Camera1", &mut used), "Camera1 (2)");
        assert_eq!(unique_name("a", &mut used), "a");
        assert_eq!(unique_name("a", &mut used), "a (2)");
        // A name whose mesh would take another's name.
        assert_eq!(unique_name("a_mesh", &mut used), "a_mesh (2)");
        assert_eq!(unique_name("  ", &mut used), "object");
        let long = "é".repeat(40);
        let n = unique_name(&long, &mut used);
        assert!(n.len() <= NAME_BYTES && n.chars().all(|c| c == 'é'), "{n}");
        let n2 = unique_name(&long, &mut used);
        assert!(n2.len() <= NAME_BYTES && n2.ends_with(" (2)"), "{n2}");
    }

    /// Names a shell would otherwise act on: spaces, `$(x)`, backticks, both quotes and a
    /// line break.
    const NASTY: [&str; 7] = [
        "two words",
        "a$(touch pwned)b",
        "a`touch pwned`b",
        "it's",
        "say \"hi\"",
        "line\nbreak",
        "$HOME * ; & | \\ '' end",
    ];

    #[test]
    fn posix_quote_is_one_word_taken_as_it_is() {
        assert_eq!(posix_quote("it's"), r"'it'\''s'");
        assert_eq!(posix_quote("a$(x)b"), "'a$(x)b'");
        assert_eq!(posix_quote("a`x`b"), "'a`x`b'");
        assert_eq!(posix_quote("say \"hi\""), "'say \"hi\"'");
        // A line break stays inside the quotes: the command shows on two lines.
        assert_eq!(posix_quote("line\nbreak"), "'line\nbreak'");
        #[cfg(unix)]
        for name in NASTY {
            // The shell gives the word back unchanged, as one argument.
            let out = std::process::Command::new("sh")
                .arg("-c")
                .arg(format!("printf '%s|' {}", posix_quote(name)))
                .output()
                .unwrap();
            assert!(out.status.success(), "{name:?}");
            assert_eq!(String::from_utf8(out.stdout).unwrap(), format!("{name}|"));
        }
    }

    #[test]
    fn cmd_quote_refuses_what_cmd_cannot_take() {
        assert_eq!(cmd_quote("two words").as_deref(), Some("\"two words\""));
        assert_eq!(
            cmd_quote("a$(x)b`c`'d'").as_deref(),
            Some("\"a$(x)b`c`'d'\"")
        );
        assert_eq!(cmd_quote("a & b | c").as_deref(), Some("\"a & b | c\""));
        assert_eq!(cmd_quote("say \"hi\""), None);
        assert_eq!(cmd_quote("100%"), None);
        assert_eq!(cmd_quote("line\nbreak"), None);
    }

    #[test]
    fn by_hand_command_quotes_folder_and_name() {
        for name in NASTY {
            let folder = Path::new("/data/my exports").join(name);
            let c = by_hand_command(&folder, &format!("{name}.blend"), true);
            assert!(
                c.contains("--python build_scene.py -- --save") && c.ends_with(" --pack"),
                "{c}"
            );
            if cfg!(windows) {
                continue;
            }
            assert_eq!(
                c,
                format!(
                    "cd {} && blender --background --factory-startup --python build_scene.py \
                     -- --save {} --pack",
                    posix_quote(&folder.to_string_lossy()),
                    posix_quote(&format!("{name}.blend"))
                )
            );
            // Run through sh with `blender` and `cd` stood in for: the two words arrive whole.
            #[cfg(unix)]
            {
                // Words ended by \001, a byte no name here holds.
                let script = c.replacen("cd ", r"printf '%s\001' ", 1).replacen(
                    " && blender ",
                    r" && printf '%s\001' ",
                    1,
                );
                let out = std::process::Command::new("sh")
                    .arg("-c")
                    .arg(&script)
                    .output()
                    .unwrap();
                let got = String::from_utf8(out.stdout).unwrap();
                let words: Vec<&str> = got.split('\u{1}').collect();
                assert_eq!(words[0], folder.to_string_lossy(), "{c}");
                assert_eq!(words[7], format!("{name}.blend"), "{c}");
                assert_eq!(words[8], "--pack", "{c}");
            }
        }
    }

    #[test]
    fn mesh_files_are_safe_and_unique_without_case() {
        let mut used = Vec::new();
        assert_eq!(unique_file("Fémur \"a/b\"", &mut used), "Fémur _a_b_.obj");
        assert_eq!(
            unique_file("fémur \"a/b\"", &mut used),
            "fémur _a_b_ (2).obj"
        );
        assert_eq!(unique_file("..", &mut used), "mesh.obj");
    }
}
