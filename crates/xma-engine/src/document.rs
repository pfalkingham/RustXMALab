//! The document: an open project, what the reader returned besides the model, and the state
//! the C++ derives after loading rather than storing.
//!
//! Ported from what `ui/MainWindow.cpp` runs once `ProjectFileIO::loadProject` has returned
//! (`MainWindow::loadProjectFinished`, `MainWindow::UndistortionAfterloadProjectFinished`)
//! and from `processing/ThreadScheduler.cpp` (the trial update). See `docs/cpp-recompute.md`,
//! sections 1 and 2.

use std::collections::BTreeSet;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use xma_io::project_read::mesh_opens;
use xma_io::{Attachment, FileVersion, LoadedProject, ReadError, ReadOptions, TrialSource};
use xma_model::{
    AlgorithmSettings, CalibrationType, Camera, CppCompat, Project, SettingSource, SettingsPatch,
    Trial, Vector3, VirtualPointSource,
};
use xma_vision::camera::{CameraModel, Intrinsics, ModelCompat, RigidTransform};
use xma_vision::triangulation::TriangulationMethod;
use xma_vision::undistortion::{LookupTable, create_lookup_table, fit_grid_model};

use crate::calibration::HeldPoints;

use xma_vision::rigid::RigidCompat;
use xma_vision::rigid::{PoseCamera, PoseSettings};

use crate::history::History;
use crate::markers::{
    FrameCamera, FrameContext, MarkerFramePatch, update_marker_frame, update_mean_size,
};
use crate::rigid_bodies::{self, FrameStates, PoseContext};
use crate::session::{Session, Workspace};

/// Why a project could not be opened.
#[derive(Debug)]
pub enum OpenError {
    /// The project file could not be read.
    Read(ReadError),
}

impl fmt::Display for OpenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for OpenError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Read(e) => Some(e),
        }
    }
}

impl From<ReadError> for OpenError {
    fn from(e: ReadError) -> Self {
        Self::Read(e)
    }
}

/// What the reader returned besides the model, kept so that a later save can carry it.
#[derive(Clone, Debug)]
pub struct LoadedExtras {
    /// The archive entries the model does not hold (images, `log.html`, ...), shared: a copy
    /// of the document (an undo step, a save or autosave from a worker thread) shares the
    /// file contents instead of copying them. Replace the whole list to change it.
    pub attachments: Arc<Vec<Attachment>>,
    pub file_version: FileVersion,
    pub archive_comment: Vec<u8>,
    pub trial_sources: Vec<TrialSource>,
    /// The reader's warnings.
    pub warnings: Vec<String>,
    /// Where each setting came from (see [`LoadedProject::settings_sources`]).
    pub settings_sources: Vec<(&'static str, SettingSource)>,
    /// The settings the project file's settings entry gave.
    pub stored_settings: SettingsPatch,
}

/// How to open a project: the settings that apply where the project stores none, and
/// those that override it.
///
/// In XMALab 3 every setting that changes results is per machine. Here a project may store
/// them (its settings entry, [`xma_io::settings_entry`], ledger A10); a setting is taken from
/// [`OpenOptions::overrides`] if given there, else from the project, else from
/// [`OpenOptions::base_settings`] (default: XMALab's defaults).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OpenOptions {
    /// The settings where the project stores none: XMALab's defaults, or a machine's.
    pub base_settings: AlgorithmSettings,
    /// Settings that win over the project's (the command line, or the settings a reference
    /// dump was made with).
    pub overrides: SettingsPatch,
    /// Which faults of XMALab 3 to reproduce (ledger A3, A4, F2, F18). Default: none, every
    /// one fixed; [`CppCompat::XMALAB3`] for the C++ behaviour. The document keeps it
    /// ([`Document::compat`]).
    pub compat: CppCompat,
}

/// Reads a project file the one way the engine reads one: with the C++ test of whether a
/// rigid body's mesh loads ([`mesh_opens`], ledger F18) and the settings of `options`, of
/// which loading itself reads `DefaultMarkerThreshold` (ledger F19). Videos are not opened
/// (phase 1), so a trial that stores no frame count gets 0 frames with a warning.
///
/// Every caller that opens a project goes through this function (or [`Document::open`]).
///
/// # Errors
/// When the file cannot be read, or an override is refused.
pub fn read_project(path: &Path, options: &OpenOptions) -> Result<LoadedProject, ReadError> {
    let probe = |p: &str| mesh_opens(p);
    let read = ReadOptions {
        video_info: None,
        mesh_loads: Some(&probe),
        base_settings: options.base_settings.clone(),
        overrides: options.overrides.clone(),
        compat: options.compat,
    };
    let mut loaded = xma_io::read_project_with(path, &read)?;
    crate::load_sizes::set_camera_sizes(&mut loaded, path);
    Ok(loaded)
}

/// State the C++ holds per camera but does not store in the project file.
#[derive(Clone, Debug)]
pub struct DerivedCamera {
    /// The camera matrix, the lens model when the camera has one, and the undistortion-grid
    /// model fitted after loading (`LocalUndistortion::computeUndistortion(true)`) or by the
    /// last undistortion computation.
    pub model: CameraModel,
    /// XMALab 3 mode only (ledger A124, `CppCompat::multicam_lens_maps_with_previous_camera_matrix`):
    /// after a multi-camera optimisation with lens distortion (method 2), the camera matrix
    /// with which XMALab built the lens maps and undistorted the calibration images'
    /// detected points: the matrix from before the optimisation. `None` otherwise (the maps
    /// and points use [`CameraModel::intrinsics`]). Cleared whenever the camera's matrix or
    /// lens model changes again (as the C++'s next `setDistortionCoefficiants` would).
    pub lens_points_camera_matrix: Option<xma_vision::Matrix3>,
    /// The grid's lookup table (`undistortionMapX/Y`) as the last undistortion computation of
    /// this session made it; `None` after loading (the views compute it from the model).
    pub lookup: Option<Arc<LookupTable>>,
    /// Per calibration image, the point lists XMALab holds besides those the project stores,
    /// as the calibration actions of this session left them
    /// ([`crate::calibration::HeldPoints`]); empty, or `None` for an image, where no action
    /// has set them.
    pub images: Vec<Option<Arc<HeldPoints>>>,
    /// What the grid model in [`DerivedCamera::model`] was fitted from: the project's camera
    /// as it was right after the fit (its grid points and inlier flags) and the number of
    /// neighbours (`LocalUndistortionNeighbours`). An undo step that does not hold the
    /// camera's present state keeps the source instead of the model and fits the model again
    /// when it is restored (ledger A166). `None` when no grid model was fitted.
    pub(crate) grid_source: Option<GridSource>,
}

/// The inputs of a grid fit ([`DerivedCamera::grid_source`]): `fit_grid_model` of the
/// camera's stored grid points, references and inlier flags with `neighbours` neighbours.
#[derive(Clone)]
pub(crate) struct GridSource {
    pub(crate) camera: Arc<Camera>,
    pub(crate) neighbours: usize,
}

impl fmt::Debug for GridSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GridSource({} neighbours)", self.neighbours)
    }
}

impl GridSource {
    /// The grid model fitted from the source: the model the fit made, bit for bit (the same
    /// function on the same values; the cell index is built again on first use).
    fn fit(&self) -> Option<xma_vision::undistortion::GridModel> {
        let g = self.camera.undistortion.as_ref()?;
        fit_grid_model(
            &g.grid_distorted,
            &g.grid_references,
            &g.grid_inlier,
            self.neighbours,
        )
    }

    fn same(a: Option<&GridSource>, b: Option<&GridSource>) -> bool {
        match (a, b) {
            (Some(a), Some(b)) => Arc::ptr_eq(&a.camera, &b.camera) && a.neighbours == b.neighbours,
            _ => false,
        }
    }
}

impl DerivedCamera {
    /// A camera's derived state with no held lists.
    #[must_use]
    pub fn new(model: CameraModel) -> Self {
        DerivedCamera {
            model,
            lens_points_camera_matrix: None,
            lookup: None,
            images: Vec::new(),
            grid_source: None,
        }
    }

    /// The camera model with which the calibration images' detected points are undistorted
    /// and the lens maps built: [`DerivedCamera::model`], with the camera matrix replaced by
    /// [`DerivedCamera::lens_points_camera_matrix`] when that is set.
    #[must_use]
    pub fn lens_points_model(&self) -> CameraModel {
        let mut m = self.model.clone();
        if let Some(k) = self.lens_points_camera_matrix {
            m.intrinsics.camera_matrix = k;
        }
        m
    }
}

/// What undo and redo put back besides the project: the cameras' derived state and the
/// session values an edit writes and a later edit reads ([`crate::Session`] lists them: what
/// is computed and what is pending, the calibration-frame count, the active camera and
/// calibration frame, the XMALab 3 mode's random generator). A step is recorded when the project or this state changed
/// ([`Computed::same_as`]).
#[derive(Clone, Debug)]
pub(crate) struct Computed {
    cameras: Vec<Arc<DerivedCamera>>,
    undistorted: bool,
    camera_recalibration: Vec<i32>,
    grid_recalibration: Vec<i32>,
    n_calibration_images: usize,
    active_camera: usize,
    active_frame_calibration: usize,
    rng: xma_vision::crand::CRand,
}

/// Two values equal, or with the same Debug text (so that NaN equals NaN).
fn same_value<T: PartialEq + fmt::Debug>(a: &T, b: &T) -> bool {
    a == b || format!("{a:?}") == format!("{b:?}")
}

/// [`same_value`] for two camera models, with the same answer for every pair: equal by
/// `==`, or with the same Debug text. The Debug text is compared part by part without
/// printing the grid models: two `f64`s print alike exactly when they have the same bits or
/// are both NaN (Rust prints every NaN as `NaN` and `-0.0` apart from `0.0`), and two lists
/// print alike exactly when they have the same length and their elements print alike. So
/// the comparison stops at the first difference instead of printing some 100,000 numbers
/// twice after every refit.
fn same_model(a: &CameraModel, b: &CameraModel) -> bool {
    a == b || same_model_text(a, b)
}

/// `format!("{a:?}") == format!("{b:?}")` for two camera models (see [`same_model`]).
fn same_model_text(a: &CameraModel, b: &CameraModel) -> bool {
    // Every field is named so that a new field cannot be forgotten here.
    let CameraModel {
        intrinsics,
        grid,
        compat,
    } = a;
    // `ModelCompat` holds only `bool`s, which print alike exactly when equal.
    *compat == b.compat
        && format!("{intrinsics:?}") == format!("{:?}", b.intrinsics)
        && match (grid, &b.grid) {
            (None, None) => true,
            (Some(x), Some(y)) => {
                same_lwm_text(&x.undistort, &y.undistort) && same_lwm_text(&x.distort, &y.distort)
            }
            _ => false,
        }
}

/// Whether two transforms print alike (see [`same_model`]); the index always prints as
/// `LwmIndex(..)`.
fn same_lwm_text(
    x: &xma_vision::undistortion::LwmTransform,
    y: &xma_vision::undistortion::LwmTransform,
) -> bool {
    let xma_vision::undistortion::LwmTransform {
        control_points,
        a,
        b,
        radii,
        index: _,
    } = x;
    fn same_f64(x: f64, y: f64) -> bool {
        x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan())
    }
    fn same_list<T>(x: &[T], y: &[T], same: impl Fn(&T, &T) -> bool) -> bool {
        x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same(p, q))
    }
    let same_row = |p: &[f64; 6], q: &[f64; 6]| p.iter().zip(q).all(|(u, v)| same_f64(*u, *v));
    same_list(control_points, &y.control_points, |p, q| {
        same_f64(p.x, q.x) && same_f64(p.y, q.y)
    }) && same_list(a, &y.a, same_row)
        && same_list(b, &y.b, same_row)
        && same_list(radii, &y.radii, |p, q| same_f64(*p, *q))
}

/// True when two cameras' derived states hold the same values. Parts that are the same
/// allocation are not compared.
fn same_derived(a: &Arc<DerivedCamera>, b: &Arc<DerivedCamera>) -> bool {
    Arc::ptr_eq(a, b)
        || (same_derived_but_lookup(a, b)
            && match (&a.lookup, &b.lookup) {
                (None, None) => true,
                (Some(x), Some(y)) => Arc::ptr_eq(x, y) || **x == **y,
                _ => false,
            })
}

/// [`same_derived`] without the lookup tables.
fn same_derived_but_lookup(a: &DerivedCamera, b: &DerivedCamera) -> bool {
    // Every field is named so that a new field cannot be forgotten here.
    // `grid_source` says where the model came from; the model itself is compared.
    let DerivedCamera {
        model,
        lens_points_camera_matrix,
        lookup: _,
        images,
        grid_source: _,
    } = a;
    same_model(model, &b.model)
        && same_value(lens_points_camera_matrix, &b.lens_points_camera_matrix)
        && images.len() == b.images.len()
        && images.iter().zip(&b.images).all(|(x, y)| match (x, y) {
            (None, None) => true,
            (Some(x), Some(y)) => Arc::ptr_eq(x, y) || same_value(&**x, &**y),
            _ => false,
        })
}

/// The lookup table a grid fit of `model` makes for an image of `width` by `height` pixels:
/// `createLookupTable` on the distorting transform, or every entry -1 without a grid model
/// (`xma_vision::grid_match::fit_grid`, the only maker of [`DerivedCamera::lookup`]).
fn rebuild_lookup(model: &CameraModel, width: usize, height: usize) -> LookupTable {
    match &model.grid {
        Some(g) => create_lookup_table(&g.distort, width, height),
        None => LookupTable {
            width,
            height,
            map_x: vec![-1.0; width * height],
            map_y: vec![-1.0; width * height],
        },
    }
}

/// What an undo or redo step does not keep of a camera's derived state
/// ([`Computed::drop_lookup_tables`], ledger A166): the grid lookup table (with its size) and
/// the grid model (fitted again from [`DerivedCamera::grid_source`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Dropped {
    /// The size of the lookup table dropped.
    table: Option<(usize, usize)>,
    /// True when the grid model was dropped.
    grid: bool,
}

/// Per camera, what a step dropped ([`Dropped`]).
pub(crate) type DroppedTables = Vec<Dropped>;

impl Computed {
    /// Ledger A166: drops the caches an undo or redo step holds and the document does not:
    /// a camera of `self` that is not the camera the document holds now (`live`) is replaced
    /// by a copy without its grid lookup table and, when it records what the model was
    /// fitted from ([`DerivedCamera::grid_source`]), without its grid model (and so without
    /// the model's cell index); the same copy for every step that shares the camera (through
    /// `copies`). `dropped` records what [`Computed::restore_lookup_tables`] makes again. The
    /// table is a function of the grid model and the image size (about 180 ms per camera to
    /// make), the model a function of its source (about 110 ms).
    pub(crate) fn drop_lookup_tables(
        &mut self,
        live: &[Arc<DerivedCamera>],
        dropped: &mut DroppedTables,
        copies: &mut std::collections::HashMap<
            usize,
            (Arc<DerivedCamera>, Arc<DerivedCamera>, Dropped),
        >,
    ) {
        if dropped.len() < self.cameras.len() {
            dropped.resize(self.cameras.len(), Dropped::default());
        }
        for (i, cam) in self.cameras.iter_mut().enumerate() {
            let drop_grid = cam.model.grid.is_some() && cam.grid_source.is_some();
            if cam.lookup.is_none() && !drop_grid {
                continue;
            }
            if live.get(i).is_some_and(|l| Arc::ptr_eq(l, cam)) {
                continue;
            }
            // The original is kept in `copies` for the pass, so its address names it.
            let key = Arc::as_ptr(cam) as usize;
            let (_, copy, what) = copies.entry(key).or_insert_with(|| {
                let what = Dropped {
                    table: cam.lookup.as_ref().map(|t| (t.width, t.height)),
                    grid: drop_grid,
                };
                let mut d = (**cam).clone();
                d.lookup = None;
                if drop_grid {
                    d.model.grid = None;
                }
                (cam.clone(), Arc::new(d), what)
            });
            *cam = copy.clone();
            let d = &mut dropped[i];
            d.table = d.table.or(what.table);
            d.grid |= what.grid;
        }
    }

    /// Makes again what [`Computed::drop_lookup_tables`] dropped: the grid model, taken from
    /// the document's camera now (`live`) when that has the same source, else fitted again
    /// from the source; then the lookup table, taken from the live camera when that has the
    /// same model and a table of the same size, else made again from the model
    /// ([`rebuild_lookup`]). Both are what the step held, bit for bit. A camera that comes
    /// out equal to the live one is the live one (shared).
    pub(crate) fn restore_lookup_tables(
        &mut self,
        dropped: &DroppedTables,
        live: &[Arc<DerivedCamera>],
    ) {
        for (i, cam) in self.cameras.iter_mut().enumerate() {
            let Some(what) = dropped.get(i).copied() else {
                continue;
            };
            if what == Dropped::default() {
                continue;
            }
            let l = live.get(i);
            let mut d = (**cam).clone();
            if what.grid && d.model.grid.is_none() {
                d.model.grid = match l {
                    Some(l)
                        if l.model.grid.is_some()
                            && GridSource::same(l.grid_source.as_ref(), d.grid_source.as_ref()) =>
                    {
                        l.model.grid.clone()
                    }
                    _ => d.grid_source.as_ref().and_then(GridSource::fit),
                };
            }
            if let Some((w, h)) = what.table
                && d.lookup.is_none()
            {
                d.lookup = match l {
                    Some(l)
                        if l.lookup
                            .as_ref()
                            .is_some_and(|t| (t.width, t.height) == (w, h))
                            && l.model.grid == d.model.grid =>
                    {
                        l.lookup.clone()
                    }
                    _ => Some(Arc::new(rebuild_lookup(&d.model, w, h))),
                };
            }
            let d = Arc::new(d);
            *cam = match l {
                Some(l) if same_derived(l, &d) => l.clone(),
                _ => d,
            };
        }
    }

    /// True when `self` and `other` hold the same values (the derived cameras compared by
    /// value, NaN equal to NaN).
    pub(crate) fn same_as(&self, other: &Computed) -> bool {
        // Every field is named so that a new field cannot be forgotten here.
        let Computed {
            cameras,
            undistorted,
            camera_recalibration,
            grid_recalibration,
            n_calibration_images,
            active_camera,
            active_frame_calibration,
            rng,
        } = self;
        cameras.len() == other.cameras.len()
            && cameras
                .iter()
                .zip(&other.cameras)
                .all(|(x, y)| same_derived(x, y))
            && *undistorted == other.undistorted
            && *camera_recalibration == other.camera_recalibration
            && *grid_recalibration == other.grid_recalibration
            && *n_calibration_images == other.n_calibration_images
            && *active_camera == other.active_camera
            && *active_frame_calibration == other.active_frame_calibration
            && *rng == other.rng
    }
}

/// Which trials have been updated and are waiting for the scheduler's finalize.
///
/// The C++ `ThreadScheduler::trials`: `updateTrialData` appends to it every time, and
/// `finalize_updateTrialData` empties it, but finalize runs only when a job finishes. An
/// update that starts no job (the trial needs no recomputation, or has no frames) leaves
/// its trial queued until the next update that does (`docs/cpp-recompute.md`, section 8,
/// N10).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Scheduler {
    pub queued: Vec<usize>,
}

/// Takes the place of the rigid-body step of a trial update that processes frames: called
/// with the trial's index, the pose context, the trial (markers already updated) and a
/// function that runs the step (every frame's [`rigid_bodies::update_frame`]), it returns
/// every frame's states, which are then written and the scheduler's finalize runs. A hook
/// may change the states, or return states it computed before for the same inputs.
///
/// Not a step of XMALab. `xma parity` uses it to replay the other orders of a C++ data race
/// (parity ledger, item A56), in which the interface thread computes a body at the frame on
/// screen while the update runs.
pub type UpdateHookFn = dyn Fn(usize, &PoseContext, &Trial, &dyn Fn() -> Vec<FrameStates>) -> Vec<FrameStates>
    + Send
    + Sync;

/// An [`UpdateHookFn`] held by a document ([`Document::set_update_hook`]).
#[derive(Clone)]
pub struct UpdateHook(pub Arc<UpdateHookFn>);

impl fmt::Debug for UpdateHook {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("UpdateHook")
    }
}

/// An open project.
#[derive(Clone, Debug)]
pub struct Document {
    pub(crate) project: Project,
    pub(crate) extras: LoadedExtras,
    pub(crate) cameras: Vec<Arc<DerivedCamera>>,
    scheduler: Scheduler,
    warnings: Vec<String>,
    /// The project's file: where it was opened from or last saved to (C++
    /// `Project::getProjectFilename`). `None` for a document built from a model.
    pub(crate) path: Option<PathBuf>,
    /// Applied by every trial update that processes frames while set; `None` normally.
    update_hook: Option<UpdateHook>,
    /// Which faults of XMALab 3 to reproduce: the mode the project was read in
    /// ([`LoadedProject::compat`]). Not stored in the project file.
    compat: CppCompat,
    /// Undo, redo and the modified state ([`crate::history`]).
    pub(crate) history: History,
    /// Rigid bodies, as (trial, body), whose rotation continuity and filtered poses are
    /// behind their raw poses; [`Document::settle`] brings them up to date.
    stale: BTreeSet<(usize, usize)>,
    /// What the C++ keeps outside the project ([`crate::session`]): not saved, not undone.
    session: Session,
    /// Frames whose pose from 2D rays was capped since the Console was last given them
    /// (ledger A210; [`Document::take_console_lines`]).
    capped: rigid_bodies::CappedLog,
}

impl Document {
    /// Opens a project as the C++ does: read, the after-load steps, then the update of the
    /// active trial when every camera is calibrated.
    ///
    /// The settings are the project's, else XMALab's defaults (parity ledger, item A10).
    /// To open with other settings, use [`Document::open_with`].
    ///
    /// # Errors
    /// When the project file cannot be read.
    pub fn open(path: &Path) -> Result<Document, OpenError> {
        Self::open_with(path, &OpenOptions::default())
    }

    /// [`Document::open`] with settings ([`OpenOptions`]): [`read_project`], then
    /// [`Document::from_loaded`] and [`Document::update_after_load`].
    ///
    /// # Errors
    /// When the project file cannot be read.
    pub fn open_with(path: &Path, options: &OpenOptions) -> Result<Document, OpenError> {
        let loaded = read_project(path, options)?;
        let mut doc = Document::from_loaded(loaded);
        doc.path = Some(path.to_path_buf());
        doc.update_after_load();
        Ok(doc)
    }

    /// The after-load steps that change numbers or state, without the trial update.
    ///
    /// Ported from:
    /// - core/Marker.cpp, Marker::load, which ends with `updateMeanSize` (`:780`; ledger A1,
    ///   fixed unless [`CppCompat::size_range_from_sum`]), for every marker of every trial;
    /// - ui/MainWindow.cpp, MainWindow::loadProjectFinished (`:793-804`): when no camera has
    ///   a calibration image the project becomes `NO_CALIBRATION`;
    /// - the same, `:819-831`: each camera whose undistortion grid is marked computed has its
    ///   grid model refitted from the stored grid points
    ///   (`LocalUndistortion::computeUndistortion(true)`, with `LocalUndistortionNeighbours`);
    /// - ui/MainWindow.cpp, MainWindow::UndistortionAfterloadProjectFinished (`:878-897`): for
    ///   a project that is not `NO_CALIBRATION`, each camera with a calibrated calibration
    ///   image is marked calibrated (`Camera::setCalibrated(true)`). A camera without one
    ///   keeps its stored flag;
    /// - core/RigidBody.cpp, RigidBody::load (`:1553-1554`) and
    ///   RigidBody::setReferenceMarkerReferences (`:557-558`), the only places loading sets
    ///   `initialised`, both with `referencesSet` (2 and 1), and RigidBody::resetReferences
    ///   (`:274-275`), which clears both: each body is `initialised` when its
    ///   `references_set` is not 0;
    /// - the poses a body computes while loading when its references are reset
    ///   ([`reset_references_at_load`], ledger A54).
    #[must_use]
    pub fn from_loaded(loaded: LoadedProject) -> Document {
        let LoadedProject {
            mut project,
            attachments,
            file_version,
            archive_comment,
            trial_sources,
            warnings: read_warnings,
            settings_sources,
            stored_settings,
            compat,
        } = loaded;
        let mut warnings = Vec::new();
        // Ledger F49: a project given NaN or infinite stored values (built in memory; the
        // reader has already replaced those of a file) holds their "not set" values.
        if !compat.nonfinite_values_kept {
            warnings.extend(xma_io::finite::clear_non_finite(&mut project));
        }

        for trial in &mut project.trials {
            let trial = Arc::make_mut(trial);
            for marker in &mut trial.markers {
                update_mean_size(Arc::make_mut(marker), compat.size_range_from_sum);
            }
            for body in &mut trial.rigid_bodies {
                let initialised = body.references_set != 0;
                if body.initialised != initialised {
                    Arc::make_mut(body).initialised = initialised;
                }
            }
        }
        // Ledger A54: XMALab 3 computes poses here; fixed, the trial update computes them.
        if compat.poses_at_load {
            reset_references_at_load(&mut project, compat);
        }

        if !project
            .cameras
            .iter()
            .any(|c| !c.calibration_images.is_empty())
        {
            project.calibration_type = CalibrationType::None;
        }

        let neighbours = project.algorithm_settings.local_undistortion_neighbours as usize;
        let cameras = project
            .cameras
            .iter()
            .enumerate()
            .map(|(i, cam)| {
                let mut d =
                    DerivedCamera::new(camera_model(i, cam, neighbours, compat, &mut warnings));
                if d.model.grid.is_some() {
                    d.grid_source = Some(GridSource {
                        camera: cam.clone(),
                        neighbours,
                    });
                }
                Arc::new(d)
            })
            .collect();

        if project.calibration_type != CalibrationType::None {
            for cam in &mut project.cameras {
                if cam.calibration_images.iter().any(|im| im.is_calibrated()) {
                    Arc::make_mut(cam).calibrated = true;
                }
            }
        }

        let session = Session::after_load(&project);
        Document {
            project,
            extras: LoadedExtras {
                attachments: Arc::new(attachments),
                file_version,
                archive_comment,
                trial_sources,
                warnings: read_warnings,
                settings_sources,
                stored_settings,
            },
            cameras,
            scheduler: Scheduler::default(),
            warnings,
            path: None,
            update_hook: None,
            compat,
            history: History::default(),
            stale: BTreeSet::new(),
            session,
            capped: rigid_bodies::CappedLog::default(),
        }
    }

    /// A document for a project built in memory (File > New dataset, tests): no file, no
    /// carried attachments, the archive comment and version XMALab writes, and the after-load
    /// steps of [`Document::from_loaded`]. Not a port; the C++ builds a new `Project` in its
    /// new-project dialogs.
    #[must_use]
    pub fn from_project(project: Project, compat: CppCompat) -> Document {
        Document::from_loaded(LoadedProject {
            project,
            attachments: Vec::new(),
            file_version: FileVersion {
                text: Some("1.0".to_owned()),
                value: 1.0,
            },
            archive_comment: b"XROMM Project File".to_vec(),
            trial_sources: Vec::new(),
            warnings: Vec::new(),
            settings_sources: Vec::new(),
            stored_settings: SettingsPatch::default(),
            compat,
        })
    }

    /// The derived state undo keeps ([`Computed`]), for [`crate::history`].
    pub(crate) fn computed_state(&self) -> Computed {
        let s = &self.session;
        Computed {
            cameras: self.cameras.clone(),
            undistorted: s.undistorted,
            camera_recalibration: s.camera_recalibration.clone(),
            grid_recalibration: s.grid_recalibration.clone(),
            n_calibration_images: s.n_calibration_images,
            active_camera: s.active_camera,
            active_frame_calibration: s.active_frame_calibration,
            rng: s.rng.clone(),
        }
    }

    /// Puts derived state back, for [`crate::history`].
    pub(crate) fn set_computed_state(&mut self, c: Computed) {
        self.cameras = c.cameras;
        let s = &mut self.session;
        s.undistorted = c.undistorted;
        s.camera_recalibration = c.camera_recalibration;
        s.grid_recalibration = c.grid_recalibration;
        s.n_calibration_images = c.n_calibration_images;
        s.active_camera = c.active_camera;
        s.active_frame_calibration = c.active_frame_calibration;
        s.rng = c.rng;
    }

    /// The scheduler's queue, for [`crate::history`].
    pub(crate) fn scheduler_state(&self) -> &Scheduler {
        &self.scheduler
    }

    /// Puts a scheduler queue back, for [`crate::history`].
    pub(crate) fn set_scheduler_state(&mut self, scheduler: Scheduler) {
        self.scheduler = scheduler;
    }

    /// Records that a rigid body's raw poses changed, so that its rotation continuity and
    /// filtered poses are recomputed by the next [`Document::settle`] (ledger A60). Edits
    /// call this in the default mode; in the XMALab 3 mode they do not, because the C++
    /// leaves filtered data as it was.
    pub fn mark_body_stale(&mut self, trial: usize, body: usize) {
        self.stale.insert((trial, body));
    }

    /// Forgets the bodies marked stale without recomputing them.
    pub(crate) fn clear_stale(&mut self) {
        self.stale.clear();
    }

    /// The bodies marked stale, for [`crate::history`] (a nested step that fails puts them
    /// back as they were when it began).
    pub(crate) fn stale_state(&self) -> &BTreeSet<(usize, usize)> {
        &self.stale
    }

    /// Puts a set of stale bodies back, for [`crate::history`].
    pub(crate) fn set_stale_state(&mut self, stale: BTreeSet<(usize, usize)>) {
        self.stale = stale;
    }

    /// Brings the bodies marked by [`Document::mark_body_stale`] up to date: for each,
    /// `makeRotationsContinous` then `filterTransformations`, as the scheduler's finalize
    /// does for a whole trial. Runs when the outermost edit step is committed
    /// ([`Document::commit_step`]); calling it with nothing marked does nothing.
    ///
    /// The continuity step changes nothing in a series it has already processed, so a body
    /// whose raw poses were recomputed in full, or frame by frame with
    /// [`Document::continue_rotations`], gets the rotations of a full update.
    pub fn settle(&mut self) {
        let stale = std::mem::take(&mut self.stale);
        if !stale.is_empty() {
            self.note_change();
        }
        let mut by_trial: std::collections::BTreeMap<usize, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (trial, body) in stale {
            let Some(t) = self.project.trials.get(trial) else {
                continue;
            };
            if body < t.rigid_bodies.len() {
                by_trial.entry(trial).or_default().push(body);
            }
        }
        for (trial, bodies) in by_trial {
            let cycle = self.continue_and_filter(trial, &bodies);
            self.warn_cycle(trial, cycle);
        }
    }

    /// For the listed bodies of a trial (index order, in range), `makeRotationsContinous`
    /// then `filterTransformations`, as the scheduler's finalize does: body by body with
    /// [`CppCompat::finalize_body_by_body`] (XMALab 3), else every listed body's rotations
    /// first and then every listed body's filter (ledger A91), so that a body filtered from
    /// its virtual points (`Filter3DPoints`) reads every driving body's rotations after the
    /// continuity step, whatever the order of the bodies. Returns whether a cycle of
    /// virtual points was met.
    ///
    /// Ported from processing/ThreadScheduler.cpp, ThreadScheduler::finalize_updateTrialData
    /// (`:101-106`).
    pub(crate) fn continue_and_filter(&mut self, trial: usize, bodies: &[usize]) -> bool {
        let ctx = self.pose_context(trial);
        let by_body = self.compat.finalize_body_by_body;
        let t = Arc::make_mut(&mut self.project.trials[trial]);
        let mut cycle = false;
        if by_body {
            for &body in bodies {
                rigid_bodies::make_rotations_continuous(t, body);
                cycle |= rigid_bodies::filter_transformations(&ctx, t, body);
            }
        } else {
            for &body in bodies {
                rigid_bodies::make_rotations_continuous(t, body);
            }
            for &body in bodies {
                cycle |= rigid_bodies::filter_transformations(&ctx, t, body);
            }
        }
        cycle
    }

    /// After one frame of a body was recomputed ([`Document::recompute_bodies_at_frame`])
    /// in a series whose other frames hold the output of the continuity step, applies that
    /// step from this frame on, so that the series is what the continuity step of a full
    /// update gives. Returns whether a cycle of virtual points was met.
    ///
    /// Not a step of XMALab; it keeps the rule of `makeRotationsContinous`
    /// (core/RigidBody.cpp, `:1594-1613`, through
    /// [`xma_vision::rigid::make_rotations_continuous`]) on a series edited frame by frame.
    /// That step replaces a rotation vector `r` by the same rotation `-r / |r| * (2π - |r|)`
    /// when it turns more than 150 degrees from the frame before, frame after frame, so the
    /// stored rotation of the next frame may be such a replacement made against the old
    /// value of this one. Running the whole step again over the stored series would replace
    /// it a second time, which neither gives back the update's vector bit for bit nor makes
    /// the same decision when the two frames are between 30 and 150 degrees apart. So the
    /// rule is applied at this frame, then each later frame with a pose is recomputed
    /// (giving the update's own rotation) and the rule applied, until a frame comes out as
    /// it is stored, or a frame without a pose (whose successor the rule does not compare);
    /// from there on the stored series is the update's. Usually one later frame is
    /// recomputed, or none.
    pub(crate) fn continue_rotations(&mut self, trial: usize, body: usize, frame: usize) -> bool {
        let mut cycle = false;
        let mut i = frame;
        loop {
            let t = &self.project.trials[trial];
            let rb = &t.rigid_bodies[body];
            let n = t.n_frames.min(rb.n_frames());
            if i >= n {
                break;
            }
            let old = if i == frame {
                None
            } else {
                if !rb.raw.valid[i] {
                    break;
                }
                let old = rb.raw.poses[i].rotation;
                // A body on a cycle of virtual points (a project opened so) is computed
                // with the bodies that follow it, so that the cycle is entered where the
                // update enters it (the lowest of them); others need no company.
                let list = if crate::edit::ops::on_virtual_point_cycle(t, body) {
                    crate::edit::ops::with_dependents(t, &[body])
                } else {
                    vec![body]
                };
                cycle |= self.recompute_bodies_at_frame_writing(trial, i, &list, &[body]);
                Some(old)
            };
            let t = Arc::make_mut(&mut self.project.trials[trial]);
            let rb = Arc::make_mut(&mut t.rigid_bodies[body]);
            if i > 0 {
                let mut pair = [rb.raw.poses[i - 1].rotation, rb.raw.poses[i].rotation];
                let valid = [rb.raw.valid[i - 1], rb.raw.valid[i]];
                xma_vision::rigid::make_rotations_continuous(&mut pair, &valid);
                rb.raw.poses[i].rotation = pair[1];
            }
            if old == Some(rb.raw.poses[i].rotation) {
                break;
            }
            i += 1;
        }
        cycle
    }

    /// Recomputes the raw state (pose flag, pose, unfiltered errors) of `bodies` at one
    /// frame as the trial update's rigid-body step computes it, and writes it for those
    /// bodies only. Returns whether a cycle of virtual points was met.
    ///
    /// Ported from processing/UpdateTrialFrame.cpp, UpdateTrialFrame::process (`:62-70`):
    /// every body's `poseComputed[frame]` is cleared first, then `computePose(frame)` runs
    /// for the listed bodies in index order, so a virtual point driven by another body
    /// uses that body's pose computed afresh at this frame (on demand, through
    /// `transformPoint`), as in the update. The other bodies' stored state is not written.
    /// The markers must hold this frame's 3D points already. Indices out of range are
    /// skipped.
    pub(crate) fn recompute_bodies_at_frame(
        &mut self,
        trial: usize,
        frame: usize,
        bodies: &[usize],
    ) -> bool {
        self.recompute_bodies_at_frame_writing(trial, frame, bodies, bodies)
    }

    /// [`Document::recompute_bodies_at_frame`] computing `bodies` and writing only those of
    /// them listed in `write`.
    pub(crate) fn recompute_bodies_at_frame_writing(
        &mut self,
        trial: usize,
        frame: usize,
        bodies: &[usize],
        write: &[usize],
    ) -> bool {
        let Some(t) = self.project.trials.get(trial) else {
            return false;
        };
        let mut list: Vec<usize> = bodies
            .iter()
            .copied()
            .filter(|&b| b < t.rigid_bodies.len())
            .collect();
        list.sort_unstable();
        list.dedup();
        if list.is_empty() {
            return false;
        }
        let ctx = self.pose_context(trial);
        let t = &self.project.trials[trial];
        let mut states = FrameStates::read(t, frame);
        for s in &mut states.bodies {
            s.computed = false;
        }
        for &b in &list {
            rigid_bodies::compute_pose(&ctx, t, &mut states, b);
        }
        let write: Vec<usize> = list.iter().copied().filter(|b| write.contains(b)).collect();
        let t = Arc::make_mut(&mut self.project.trials[trial]);
        states.write_bodies(t, &write);
        self.warn_cycle(trial, states.cycle);
        states.cycle
    }

    /// The rigid-body step of the trial update and its finalize, without the marker step:
    /// every body's raw pose at every frame as [`rigid_bodies::update_frame`] computes it,
    /// then `makeRotationsContinous` and `filterTransformations` on every body as the
    /// finalize runs them (ledger A91). The markers' 3D points are left as they are (ledger F9), and
    /// the trial's flags and the scheduler are not touched.
    ///
    /// Not a step of XMALab 3, which recomputes nothing when a pose setting changes (ledger
    /// A60); used for those settings in the default mode. Ported from
    /// processing/UpdateTrialFrame.cpp, UpdateTrialFrame::process (`:62-70`), and
    /// processing/ThreadScheduler.cpp, ThreadScheduler::finalize_updateTrialData
    /// (`:101-106`).
    pub(crate) fn recompute_poses_as_update(&mut self, trial: usize) {
        let before = self.project.clone();
        let Some(t) = self.project.trials.get(trial) else {
            return;
        };
        let n_frames = t.n_frames;
        if t.rigid_bodies.is_empty() {
            return;
        }
        init_bodies(&mut self.project.trials[trial], n_frames);
        let ctx = self.pose_context(trial);
        let t = &self.project.trials[trial];
        let states: Vec<FrameStates> = (0..n_frames)
            .map(|frame| rigid_bodies::update_frame(&ctx, t, frame))
            .collect();
        let t = Arc::make_mut(&mut self.project.trials[trial]);
        let mut cycle = write_states(t, &states);
        let all: Vec<usize> = (0..t.rigid_bodies.len()).collect();
        cycle |= self.continue_and_filter(trial, &all);
        self.warn_cycle(trial, cycle);
        self.note_if_changed(&before);
    }

    /// Renumbers the trial indices the document holds besides the project (the scheduler's
    /// queue and the stale bodies) after trials were removed: `map` gives a trial's new
    /// index, or `None` for a removed trial, whose entries are dropped.
    pub(crate) fn remap_trials(&mut self, map: &dyn Fn(usize) -> Option<usize>) {
        self.scheduler.queued = self
            .scheduler
            .queued
            .iter()
            .filter_map(|&t| map(t))
            .collect();
        self.stale = self
            .stale
            .iter()
            .filter_map(|&(t, b)| map(t).map(|t| (t, b)))
            .collect();
    }

    /// Whether a body is marked stale (for tests and edits).
    #[must_use]
    pub fn is_body_stale(&self, trial: usize, body: usize) -> bool {
        self.stale.contains(&(trial, body))
    }

    /// Which faults of XMALab 3 this document reproduces (ledger A3, A4, F2, F18).
    #[must_use]
    pub fn compat(&self) -> CppCompat {
        self.compat
    }

    /// Changes the compatibility mode for later updates, saves and exports. What was
    /// computed or read before is not recomputed; reading (F18) is not repeated.
    pub fn set_compat(&mut self, compat: CppCompat) {
        self.note_change();
        self.compat = compat;
        for c in &mut self.cameras {
            Arc::make_mut(c).model.compat = model_compat(compat);
        }
    }

    /// Sets (or with `None` clears) the [`UpdateHook`] that every later trial update that
    /// processes frames applies. Not a step of XMALab; see [`UpdateHookFn`].
    pub fn set_update_hook(&mut self, hook: Option<UpdateHook>) {
        self.update_hook = hook;
    }

    /// The update that follows loading. Returns the trial updated, if any.
    ///
    /// Ported from ui/MainWindow.cpp, MainWindow::UndistortionAfterloadProjectFinished
    /// (`:860-918`) and ui/WorkspaceNavigationFrame.cpp, WorkspaceNavigationFrame::workspaceChanged
    /// (`:166-169`): nothing for a `NO_CALIBRATION` project; otherwise, only when every
    /// camera has a calibrated calibration image, the workspace changes to Marker tracking,
    /// which updates the **active trial only**. Other trials are updated when they are
    /// selected ([`Document::select_trial`]).
    pub fn update_after_load(&mut self) -> Option<usize> {
        if self.project.calibration_type == CalibrationType::None {
            return None;
        }
        let all_calibrated = self
            .project
            .cameras
            .iter()
            .all(|c| c.calibration_images.iter().any(|im| im.is_calibrated()));
        if !all_calibrated {
            return None;
        }
        let active = self.project.active_trial;
        if active < self.project.trials.len() {
            self.update_trial(active);
            Some(active)
        } else {
            None
        }
    }

    /// Makes a trial the active one and updates it, as the trial selector does once a
    /// project is loaded.
    ///
    /// Ported from ui/WorkspaceNavigationFrame.cpp, WorkspaceNavigationFrame::activeTrialChanged
    /// (`:211-214`).
    pub fn select_trial(&mut self, trial: usize) {
        let before = self.project.clone();
        if trial < self.project.trials.len() {
            self.project.active_trial = trial;
            self.update_trial(trial);
        }
        self.note_if_changed(&before);
    }

    /// Updates a trial: recomputes its markers and rigid bodies when the trial requires
    /// recomputation, then runs the scheduler's finalize.
    ///
    /// Ported from processing/ThreadScheduler.cpp, ThreadScheduler::updateTrialData
    /// (`:65-96`), processing/UpdateTrialFrame.cpp, UpdateTrialFrame::process (`:49-71`), and
    /// ThreadScheduler::finalize_updateTrialData (`:98-111`). Returns whether any frame was
    /// processed.
    ///
    /// - The trial is queued whatever happens.
    /// - If the trial requires recomputation, every body whose pose vectors are not
    ///   `n_frames` long is cleared and resized (`RigidBody::init`), then every frame
    ///   `0..n_frames` is processed (the whole trial; the start and end frames are not used).
    ///   Each frame runs the marker step, then the rigid-body step
    ///   ([`rigid_bodies::update_frame`]: every body's flag cleared, then `computePose` for
    ///   each body in index order). Here the marker step runs for every frame before the
    ///   rigid-body step of any frame; in the C++ the frames run concurrently. Both give the
    ///   same result because a frame's job reads and writes only that frame: the marker step
    ///   at a frame writes only that frame, and `computePose(frame)` reads the markers at
    ///   that frame only, and other bodies only at that frame.
    /// - When at least one frame was processed, the jobs' completion runs finalize for every
    ///   queued trial, in queue order (a trial queued twice is finalized twice; in the
    ///   default mode the repeat is skipped where it provably changes nothing,
    ///   `Document::finalize_repeat_is_noop`):
    ///   `makeRotationsContinous` and `filterTransformations` on every body
    ///   (`continue_and_filter`: body by body in XMALab 3, every body's rotations
    ///   first by default, ledger A91), then
    ///   `Trial::setRequiresRecomputation(false)` (which clears every marker's flag).
    /// - When no frame was processed (the trial needs no recomputation, or has no frames),
    ///   no job finishes, so finalize does not run and the trial stays queued until a later
    ///   update runs one.
    ///
    /// Markers keep their own flags during the update: a marker whose flag is clear is only
    /// reprojected even when the trial is flagged (parity ledger, item F9).
    pub fn update_trial(&mut self, trial: usize) -> bool {
        let before = self.project.clone();
        if trial >= self.project.trials.len() {
            return false;
        }
        self.scheduler.queued.push(trial);
        let t = &self.project.trials[trial];
        if !t.requires_recomputation || t.n_frames == 0 {
            return false;
        }

        // ThreadScheduler.cpp:76-82: `RigidBody::init(nbImages)` for each body whose
        // `poseComputed` is not `nbImages` long.
        let n_frames = t.n_frames;
        init_bodies(&mut self.project.trials[trial], n_frames);

        let ctx = self.frame_context(trial);
        let t = &self.project.trials[trial];
        let patches: Vec<Vec<MarkerFramePatch>> = (0..n_frames)
            .map(|frame| marker_step(&ctx, t, frame))
            .collect();
        apply_marker_patches(Arc::make_mut(&mut self.project.trials[trial]), &patches);

        // UpdateTrialFrame.cpp:62-70, after the marker step of every frame (see above).
        let pose_ctx = self.pose_context(trial);
        let t = &self.project.trials[trial];
        let step = || -> Vec<FrameStates> {
            (0..n_frames)
                .map(|frame| rigid_bodies::update_frame(&pose_ctx, t, frame))
                .collect()
        };
        let states = match &self.update_hook {
            Some(hook) => (hook.0)(trial, &pose_ctx, t, &step),
            None => step(),
        };
        let cycle = write_states(Arc::make_mut(&mut self.project.trials[trial]), &states);
        self.warn_cycle(trial, cycle);

        // Default mode: a trial queued again (by updates that started no job) is finalized
        // once when a repeat provably changes nothing ([`Document::finalize_repeat_is_noop`]).
        // The queue itself is kept as XMALab 3 keeps it, in both modes.
        let skip_repeats = self.compat == CppCompat::FIXED;
        let mut finalized: Vec<usize> = Vec::new();
        for queued in std::mem::take(&mut self.scheduler.queued) {
            if finalized.contains(&queued) {
                continue;
            }
            let repeat_is_noop = skip_repeats && self.finalize_repeat_is_noop(queued);
            self.finalize_trial(queued);
            if repeat_is_noop {
                finalized.push(queued);
            }
        }
        self.clear_non_finite();
        self.note_if_changed(&before);
        true
    }

    /// True when, in the trial's present state, running [`Document::finalize_trial`] a second
    /// time right after the first changes nothing, so that a repeated queue entry may be
    /// skipped. Not a port: an optimisation of the finalize of `ThreadScheduler` (N10).
    ///
    /// The finalize is the continuity step on every body's raw rotations, then the filter
    /// (which writes only the filtered poses and errors, from the raw poses and the
    /// markers), then the trial's flags. The continuity step replaces a rotation `r` that
    /// turns more than 150 degrees from the (already replaced) one before it by
    /// `-r / |r| * (2π - |r|)`: for `|r| < 2π` that turns it to within 30 degrees, so a
    /// second pass replaces nothing (and `|r| = 2π` gives a zero vector, which no test
    /// replaces); for `|r| > 2π` the replacement keeps the direction and a second pass
    /// replaces it again. So no raw rotation of a body may be longer than just under 2π
    /// (or not finite). With `Filter3DPoints`, a virtual point driven by another body
    /// computes that body where its flag is clear and stores the pose, after that body's
    /// continuity step; a second finalize would then differ, so no body may have one.
    fn finalize_repeat_is_noop(&self, trial: usize) -> bool {
        /// Just under 2π, so that rounding in the norm cannot matter.
        const LONGEST: f64 = std::f64::consts::TAU - 1e-4;
        let Some(t) = self.project.trials.get(trial) else {
            return false;
        };
        let virtual_from_body = || {
            t.rigid_bodies.iter().any(|b| {
                b.virtual_points
                    .iter()
                    .any(|v| matches!(v.source, VirtualPointSource::OtherBody { .. }))
            })
        };
        if self.project.algorithm_settings.filter_3d_points && virtual_from_body() {
            return false;
        }
        t.rigid_bodies.iter().all(|b| {
            b.raw.poses.iter().all(|p| {
                let r = p.rotation;
                (r.x * r.x + r.y * r.y + r.z * r.z).sqrt() <= LONGEST
            })
        })
    }

    /// The part of the scheduler's finalize that concerns one trial.
    ///
    /// Ported from processing/ThreadScheduler.cpp, ThreadScheduler::finalize_updateTrialData
    /// (`:101-106`).
    fn finalize_trial(&mut self, trial: usize) {
        let all: Vec<usize> = (0..self.project.trials[trial].rigid_bodies.len()).collect();
        let cycle = self.continue_and_filter(trial, &all);
        self.warn_cycle(trial, cycle);
        self.set_trial_requires_recomputation(trial, false);
    }

    /// Recomputes every frame of one rigid body, then makes its rotations continuous; no
    /// filtering. What the rigid-body dialog runs after adding or deleting a virtual point
    /// or toggling the optimised reference.
    ///
    /// Ported from core/RigidBody.cpp, RigidBody::recomputeTransformations (`:1584-1592`):
    /// other bodies' flags are not cleared first (parity ledger, item A36); see
    /// [`rigid_bodies::recompute_transformations`]. Does nothing for an index out of range.
    pub fn recompute_transformations(&mut self, trial: usize, body: usize) {
        let before = self.project.clone();
        let Some(t) = self.project.trials.get(trial) else {
            return;
        };
        if body >= t.rigid_bodies.len() {
            return;
        }
        let ctx = self.pose_context(trial);
        let t = Arc::make_mut(&mut self.project.trials[trial]);
        let cycle = rigid_bodies::recompute_transformations(&ctx, t, body);
        self.warn_cycle(trial, cycle);
        self.note_if_changed(&before);
    }

    /// Filters one rigid body's poses and updates its filtered errors.
    ///
    /// Ported from core/RigidBody.cpp, RigidBody::filterTransformations; see
    /// [`rigid_bodies::filter_transformations`]. Does nothing for an index out of range.
    pub fn filter_transformations(&mut self, trial: usize, body: usize) {
        let before = self.project.clone();
        let Some(t) = self.project.trials.get(trial) else {
            return;
        };
        if body >= t.rigid_bodies.len() {
            return;
        }
        let ctx = self.pose_context(trial);
        let t = Arc::make_mut(&mut self.project.trials[trial]);
        let cycle = rigid_bodies::filter_transformations(&ctx, t, body);
        self.warn_cycle(trial, cycle);
        self.note_if_changed(&before);
    }

    /// For each body in index order, [`Document::recompute_transformations`] then
    /// [`Document::filter_transformations`]. What the trial dialog runs when only the
    /// recording speed or the cutoff changed (`TRIALDIALOGUPDATEFILTER`); markers are not
    /// touched, and the trial's flags are left as they are.
    ///
    /// Ported from core/Trial.cpp, Trial::recomputeAndFilterRigidBodyTransformations
    /// (`:939-947`).
    pub fn recompute_and_filter_rigid_body_transformations(&mut self, trial: usize) {
        let Some(t) = self.project.trials.get(trial) else {
            return;
        };
        for body in 0..t.rigid_bodies.len() {
            self.recompute_transformations(trial, body);
            self.filter_transformations(trial, body);
        }
    }

    /// The recomputation the rigid-body export runs before writing: for each exported body
    /// in the order given, [`Document::recompute_transformations`], then, for a filtered
    /// export, [`Document::filter_transformations`]. Indices out of range are skipped.
    ///
    /// Ported from core/Trial.cpp, Trial::saveRigidBodyTransformations (`:951-955`).
    pub fn recompute_rigid_bodies_for_export(
        &mut self,
        trial: usize,
        bodies: &[usize],
        filtered: bool,
    ) {
        for &body in bodies {
            self.recompute_transformations(trial, body);
            if filtered {
                self.filter_transformations(trial, body);
            }
        }
    }

    fn warn_cycle(&mut self, trial: usize, cycle: bool) {
        if cycle {
            let text = format!(
                "trial {trial}: virtual points of rigid bodies refer to each other in a cycle; \
                 the C++ recurses without end there, here the inner reference has no pose"
            );
            if !self.warnings.contains(&text) {
                self.warnings.push(text);
            }
        }
    }

    /// The immutable inputs of the rigid-body step for one trial, with the settings as they
    /// are now.
    ///
    /// Each camera's pose is that of its calibration image at the trial's
    /// `referenceCalibrationImage`. A camera without such an image (undefined behaviour in
    /// the C++) gets a zero pose, marked uncalibrated for EPnP.
    ///
    /// Frames whose pose from 2D rays it caps (ledger A210) are noted in the document, for
    /// [`Document::take_console_lines`].
    pub fn pose_context(&self, trial: usize) -> PoseContext {
        let mut ctx = build_pose_context(
            &self.project,
            trial,
            self.compat,
            self.cameras.iter().map(|derived| derived.model.clone()),
        );
        ctx.capped = self.capped.recorder(trial);
        ctx
    }

    /// The Console lines left by the pose computations since the last call, and forgets
    /// them: in the default mode, one line per rigid body whose pose from markers seen by
    /// one camera only had more than 1,000,000 combinations of positions at some frames
    /// (ledger A210), naming the body, the trial and the frames (1-based, runs compressed),
    /// with the trial's and the body's indices. Every computation of poses leaves its
    /// lines here (an edit through [`Document::apply`] or [`Document::apply_all`], a trial
    /// update, a point placed with [`crate::tracking::place_point`], a tracking run,
    /// opening or saving the project): this is the only route by which they reach the
    /// caller, who shows each once. An edit's outcome never carries them.
    pub fn take_capped_lines(&mut self) -> Vec<rigid_bodies::CappedLine> {
        let capped = self.capped.take();
        rigid_bodies::capped_lines(&self.project, &capped)
    }

    /// The text of [`Document::take_capped_lines`].
    pub fn take_console_lines(&mut self) -> Vec<String> {
        self.take_capped_lines()
            .into_iter()
            .map(|l| l.text)
            .collect()
    }

    /// What [`Document::apply`] and [`Document::apply_all`] do after their step: on success
    /// the Console lines of the step stay pending ([`Document::take_capped_lines`]); on an
    /// error the capped frames are put back as they were before the step (`before`), since
    /// the step's computations were taken back.
    pub(crate) fn finish_edit_console(
        &mut self,
        before: rigid_bodies::CappedSnapshot,
        result: crate::edit::EditResult,
    ) -> crate::edit::EditResult {
        if result.is_err() {
            self.capped.restore(before.0);
        }
        result
    }

    /// The capped frames noted now, for [`Document::finish_edit_console`].
    pub(crate) fn capped_snapshot(&self) -> rigid_bodies::CappedSnapshot {
        rigid_bodies::CappedSnapshot(self.capped.snapshot())
    }

    /// Flags or clears a trial for recomputation, and every one of its markers with it.
    ///
    /// Ported from core/Trial.cpp, Trial::setRequiresRecomputation (`:758-766`).
    pub fn set_trial_requires_recomputation(&mut self, trial: usize, value: bool) {
        let Some(t) = self.project.trials.get_mut(trial) else {
            return;
        };
        let mut changed = t.requires_recomputation != value;
        let t = Arc::make_mut(t);
        t.requires_recomputation = value;
        for m in &mut t.markers {
            if m.requires_recomputation != value {
                Arc::make_mut(m).requires_recomputation = value;
                changed = true;
            }
        }
        if changed {
            self.note_change();
        }
    }

    /// Each camera's projection matrix `K [R | t]` for a trial's reference calibration
    /// image, as the trial update uses it (`Camera::getProjectionMatrix`); `None` for a
    /// camera without that image. Empty for a trial out of range.
    #[must_use]
    pub fn projection_matrices(&self, trial: usize) -> Vec<Option<xma_vision::Matrix3x4>> {
        if trial >= self.project.trials.len() {
            return Vec::new();
        }
        self.frame_context(trial)
            .cameras
            .into_iter()
            .map(|c| c.projection)
            .collect()
    }

    /// The immutable inputs of the per-frame step for one trial.
    pub(crate) fn frame_context(&self, trial: usize) -> FrameContext {
        let reference = self.project.trials[trial].reference_calibration_image;
        let cameras = self
            .project
            .cameras
            .iter()
            .zip(&self.cameras)
            .map(|(cam, derived)| FrameCamera {
                calibrated: cam.calibrated,
                ..FrameCamera::new(derived.model.clone(), reference_pose(cam, reference))
            })
            .collect();
        FrameContext {
            cameras,
            no_calibration: self.project.calibration_type == CalibrationType::None,
            method: TriangulationMethod::from_setting(i32::from(
                self.project.algorithm_settings.triangulation_method,
            )),
            triangulation_unchecked: self.compat.triangulation_unchecked,
            project_undefined_points: self.compat.project_undefined_points,
        }
    }

    /// The project's file, where it was opened from or last saved to.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Sets the project's file, for a document built with [`Document::from_loaded`]; the
    /// next [`Document::save`] writes there.
    pub fn set_path(&mut self, path: impl Into<PathBuf>) {
        self.path = Some(path.into());
    }

    /// Ledger F50: in the default mode, every stored number that is NaN or infinite (a
    /// computation's result) is replaced by its field's "not set" value
    /// ([`xma_io::finite::clear_non_finite`]), with a line in [`Document::warnings`] for each.
    /// Runs where the project's stored values are computed: when an edit step is committed,
    /// after a trial update, after loading and before saving. Nothing in the XMALab 3 mode
    /// ([`CppCompat::nonfinite_values_kept`]), nor for a project with finite values only.
    /// Returns whether anything was replaced.
    pub(crate) fn clear_non_finite(&mut self) -> bool {
        if self.compat.nonfinite_values_kept || !xma_io::finite::has_non_finite(&self.project) {
            return false;
        }
        let lines = xma_io::finite::clear_non_finite(&mut self.project);
        for l in lines {
            if !self.warnings.contains(&l) {
                self.warnings.push(l);
            }
        }
        self.note_change();
        true
    }

    /// Adds a line to [`Document::warnings`].
    pub(crate) fn add_warning(&mut self, text: String) {
        self.warnings.push(text);
    }

    /// The session state ([`crate::session`]).
    #[must_use]
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// The session state, to change ([`crate::session`]). Not an undo step.
    pub fn session_mut(&mut self) -> &mut Session {
        &mut self.session
    }

    /// Shows a workspace. Showing Marker tracking (again, when `force`) updates the active
    /// trial, if there is one.
    ///
    /// Ported from ui/State.cpp, State::changeWorkspace (`:73-80`), and
    /// ui/WorkspaceNavigationFrame.cpp, WorkspaceNavigationFrame::workspaceChanged
    /// (`:153-169`, the `updateTrialData` of the active trial).
    pub fn set_workspace(&mut self, workspace: Workspace, force: bool) {
        if !force && self.session.workspace == workspace {
            return;
        }
        self.session.workspace = workspace;
        if workspace == Workspace::Calibration {
            // WizardCalibrationCubeFrame::workspaceChanged: `checkerboardManual` from the
            // setting, then setDialog.
            self.session.checkerboard_manual = self.settings().disable_checkerboard_detection;
            self.calibration_dialog();
        }
        if workspace == Workspace::Tracking {
            let active = self.project.active_trial;
            if active < self.project.trials.len() {
                self.update_trial(active);
            }
        }
    }

    /// The project.
    #[must_use]
    pub fn project(&self) -> &Project {
        &self.project
    }

    /// The settings that change results.
    #[must_use]
    pub fn settings(&self) -> &AlgorithmSettings {
        &self.project.algorithm_settings
    }

    /// The settings, for changing. As in the C++, a change takes effect at the next
    /// computation that reads it: changing the triangulation method does not flag any trial
    /// (the Settings dialog does that separately, `ui/SettingsDialog.cpp:313-325`), and
    /// `local_undistortion_neighbours` is read only when the grid is fitted on load.
    pub fn settings_mut(&mut self) -> &mut AlgorithmSettings {
        self.note_change();
        &mut self.project.algorithm_settings
    }

    /// One trial.
    #[must_use]
    pub fn trial(&self, trial: usize) -> &Trial {
        &self.project.trials[trial]
    }

    /// The derived state of one camera.
    #[must_use]
    pub fn camera(&self, camera: usize) -> &DerivedCamera {
        &self.cameras[camera]
    }

    /// The derived state of one camera, for changing.
    pub(crate) fn camera_mut(&mut self, camera: usize) -> &mut DerivedCamera {
        Arc::make_mut(&mut self.cameras[camera])
    }

    /// Brings each camera's derived camera matrix and lens model up to date with the
    /// project (after a calibration edit, undo or redo). A camera whose values changed loses
    /// its [`DerivedCamera::lens_points_camera_matrix`]. The grid models are not refitted.
    pub(crate) fn sync_camera_intrinsics(&mut self) {
        for (cam, derived) in self.project.cameras.iter().zip(&mut self.cameras) {
            let intrinsics = Intrinsics {
                camera_matrix: cam.camera_matrix,
                distortion: cam.has_model_distortion.then_some(cam.distortion),
            };
            if derived.model.intrinsics != intrinsics {
                let d = Arc::make_mut(derived);
                d.model.intrinsics = intrinsics;
                d.lens_points_camera_matrix = None;
            }
        }
    }

    /// The held point lists of calibration image `image` of `camera`, when an action of this
    /// session has set them ([`crate::calibration::HeldPoints`]).
    #[must_use]
    pub fn camera_held_points(&self, camera: usize, image: usize) -> Option<&Arc<HeldPoints>> {
        self.cameras.get(camera)?.images.get(image)?.as_ref()
    }

    /// True when every camera is calibrated (C++ `Project::isCalibrated`, as
    /// `Project::checkCalibration` computes it).
    #[must_use]
    pub fn is_calibrated(&self) -> bool {
        self.project.cameras.iter().all(|c| c.calibrated)
    }

    /// Trials updated but not yet finalized (see [`Scheduler`]).
    #[must_use]
    pub fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }

    /// What the reader returned besides the model.
    #[must_use]
    pub fn extras(&self) -> &LoadedExtras {
        &self.extras
    }

    /// Things the engine tolerated where the C++ has undefined behaviour or fails silently.
    #[must_use]
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }
}

/// The rigid-body part of a compatibility mode (ledger A2, A3, A4, A20, A23, A26 to A30,
/// A210).
pub(crate) fn rigid_compat(compat: CppCompat) -> RigidCompat {
    RigidCompat {
        virtual_points_per_camera_pair: compat.virtual_points_per_camera_pair,
        virtual_points_at_origin: compat.virtual_points_at_origin,
        lens_distortion: compat.distortion_in_refinement,
        error_sd_nan: compat.error_sd_nan,
        single_precision: compat.single_precision,
        epnp_last_camera: compat.epnp_last_camera,
        rays_misaligned: compat.rays_misaligned,
        refine_failure_ignored: compat.refine_failure_ignored,
        uncalibrated_cameras_in_refinement: compat.uncalibrated_cameras_in_refinement,
        refinement_without_skew: compat.refinement_without_skew,
        pose_from_2d_uncapped: compat.pose_from_2d_uncapped,
    }
}

/// The camera-model part of a compatibility mode (ledger A11, A19).
pub(crate) fn model_compat(compat: CppCompat) -> ModelCompat {
    ModelCompat {
        grid_first_when_distorting: compat.grid_first_when_distorting,
        five_iterations: compat.five_iterations,
    }
}

/// What loading leaves behind in a rigid body whose `<RigidBody>` element names no
/// reference files: the poses it computed while loading (ledger A54).
///
/// Ported from ui/ProjectFileIO.cpp, ProjectFileIO::readProjectFile (`:2144-2160`; the same
/// in the other two readers, `:574` and `:797`): a `<RigidBody>` element without
/// `ReferenceNames` or `ReferencePoints3D` calls `RigidBody::resetReferences`
/// (core/RigidBody.cpp, `:262-278`), which clears the references, calls
/// `setReferenceMarkerReferences` (`:543-560`) and then `recomputeTransformations` and
/// `filterTransformations`, there and then:
///
/// - when one of the body's markers has no reference point, the body stays uninitialised
///   and `computePose` writes only a cleared flag and zero errors, which the trial update
///   writes again: nothing is left;
/// - when every marker has one (the reader then holds `references_set` 1 and a non-empty
///   marker list), the body is initialised and every frame `0..nbImages` is computed, then
///   made continuous and filtered. That is what this function reproduces.
///
/// The state at that moment differs from the trial update's:
///
/// - the markers' 3D points and statuses are those stored in the file (or unset), not
///   recomputed;
/// - the cameras' undistortion grids are not fitted yet (`LocalUndistortion` runs after
///   loading); an unfitted grid's `transformPoint` returns its input
///   (core/UndistortionObject.cpp, `:398-416`, `transformLWM` with no control points), so
///   the grid is the identity, while the lens model (loaded with the camera) applies;
/// - the body's virtual points are added after the reset (`addDummyPoint` follows in the
///   element), so it has none, and no other body is computed on demand;
/// - the working reference points are the markers' (`loadOptimized`, which may replace
///   them, follows the reset);
/// - cameras keep their stored `calibrated` flags and the project its stored calibration
///   type (`NO_CALIBRATION` is decided after loading).
///
/// The trial update recomputes every frame later, clearing each flag first, so what is left
/// is what it does not overwrite: the raw pose of a frame where the update finds none, the
/// filtered pose of a frame outside the update's filtered runs, and the vectors' length
/// (the update's `RigidBody::init` is skipped when they already have `nbImages` frames).
/// The flags and errors are always rewritten.
///
/// Assumes, as in every file XMALab writes, that a trial's `<Marker>` elements come before
/// its `<RigidBody>` elements; a reference point loaded after the reset would not count.
pub(crate) fn reset_references_at_load(project: &mut Project, compat: CppCompat) {
    for t in 0..project.trials.len() {
        let bodies: Vec<usize> = project.trials[t]
            .rigid_bodies
            .iter()
            .enumerate()
            .filter(|(_, b)| b.references_set == 1 && b.initialised && !b.marker_indices.is_empty())
            .map(|(i, _)| i)
            .collect();
        if bodies.is_empty() {
            continue;
        }
        let ctx = build_pose_context(
            project,
            t,
            compat,
            project.cameras.iter().map(|cam| CameraModel {
                intrinsics: Intrinsics {
                    camera_matrix: cam.camera_matrix,
                    distortion: cam.has_model_distortion.then_some(cam.distortion),
                },
                grid: None,
                compat: model_compat(compat),
            }),
        );
        for b in bodies {
            let mut at_load = project.trials[t].as_ref().clone();
            {
                let body = Arc::make_mut(&mut at_load.rigid_bodies[b]);
                body.virtual_points.clear();
                body.reference_points = body.reference_points_original.clone();
            }
            rigid_bodies::recompute_transformations(&ctx, &mut at_load, b);
            rigid_bodies::filter_transformations(&ctx, &mut at_load, b);
            let (raw, filtered) = {
                let done = &at_load.rigid_bodies[b];
                (done.raw.clone(), done.filtered.clone())
            };
            let trial = Arc::make_mut(&mut project.trials[t]);
            let body = Arc::make_mut(&mut trial.rigid_bodies[b]);
            body.raw = raw;
            body.filtered = filtered;
        }
    }
}

/// The pose of a camera's calibration image `reference` (a trial's
/// `referenceCalibrationImage`); `None` when the camera has no such image.
pub(crate) fn reference_pose(cam: &Camera, reference: usize) -> Option<RigidTransform> {
    cam.calibration_images
        .get(reference)
        .map(|im| RigidTransform {
            rotation: im.rotation,
            translation: im.translation,
        })
}

/// The inputs of the rigid-body step for one trial ([`Document::pose_context`]), with the
/// camera models `models` (one per camera, in order; the cameras are cut to the shorter of
/// the two lists). Each camera's pose is that of its calibration image at the trial's
/// `referenceCalibrationImage`; a camera without such an image gets a zero pose, marked
/// uncalibrated for EPnP.
fn build_pose_context(
    project: &Project,
    trial: usize,
    compat: CppCompat,
    models: impl Iterator<Item = CameraModel>,
) -> PoseContext {
    let reference = project.trials[trial].reference_calibration_image;
    let cameras = project
        .cameras
        .iter()
        .zip(models)
        .map(|(cam, model)| PoseCamera {
            model,
            pose: reference_pose(cam, reference).unwrap_or(RigidTransform {
                rotation: Vector3::zeros(),
                translation: Vector3::zeros(),
            }),
            calibrated: cam.calibrated,
            reference_image_calibrated: cam
                .calibration_images
                .get(reference)
                .is_some_and(|im| im.is_calibrated()),
        })
        .collect();
    let s = &project.algorithm_settings;
    PoseContext {
        cameras,
        settings: PoseSettings {
            disable_advanced: s.disable_rb_compute_advanced,
            optimize: s.optimize_rigid_body,
            compat: rigid_compat(compat),
        },
        filter_3d_points: s.filter_3d_points,
        filter_correction: xma_io::export::cutoff_correction(compat),
        no_calibration: project.calibration_type == CalibrationType::None,
        capped: rigid_bodies::CappedFrames::default(),
    }
}

/// `RigidBody::init(n_frames)` on every body of the trial whose pose vectors are not
/// `n_frames` long (ThreadScheduler.cpp:76-82); the trial is not copied when none is.
fn init_bodies(trial: &mut Arc<Trial>, n_frames: usize) {
    if trial
        .rigid_bodies
        .iter()
        .any(|b| b.raw.valid.len() != n_frames)
    {
        let t = Arc::make_mut(trial);
        for body in &mut t.rigid_bodies {
            if body.raw.valid.len() != n_frames {
                Arc::make_mut(body).init(n_frames);
            }
        }
    }
}

/// Writes every frame's states into the trial, in order; true when any of them met a cycle
/// of virtual points.
fn write_states(trial: &mut Trial, states: &[FrameStates]) -> bool {
    let mut cycle = false;
    for s in states {
        s.write(trial);
        cycle |= s.cycle;
    }
    cycle
}

/// Builds a camera's model as the C++ holds it after loading.
///
/// Ported from processing/LocalUndistortion.cpp, LocalUndistortion::computeUndistortion(true)
/// and LocalUndistortion::localUndistortion_thread (the grid fit, through
/// [`fit_grid_model`]), and core/Camera.cpp, Camera::undistortPoint (the grid is used only
/// when `isComputed()`; the lens model only when `hasModelDistortion()`).
pub(crate) fn camera_model(
    index: usize,
    cam: &Camera,
    neighbours: usize,
    compat: CppCompat,
    warnings: &mut Vec<String>,
) -> CameraModel {
    let grid = cam
        .undistortion
        .as_ref()
        .filter(|g| g.computed)
        .and_then(|g| {
            if g.grid_distorted.len() != g.grid_inlier.len()
                || g.grid_references.len() != g.grid_inlier.len()
            {
                warnings.push(format!(
                    "camera {index}: undistortion grid point lists differ in length; grid not used"
                ));
                return None;
            }
            let fit = fit_grid_model(
                &g.grid_distorted,
                &g.grid_references,
                &g.grid_inlier,
                neighbours,
            );
            if fit.is_none() {
                warnings.push(format!(
                    "camera {index}: the undistortion grid could not be fitted (fewer inliers than \
                 {neighbours} neighbours); grid not used"
                ));
            }
            fit
        });
    CameraModel {
        intrinsics: Intrinsics {
            camera_matrix: cam.camera_matrix,
            distortion: cam.has_model_distortion.then_some(cam.distortion),
        },
        grid,
        compat: model_compat(compat),
    }
}

/// The marker step of one frame for every marker of a trial, in marker order.
///
/// Ported from processing/UpdateTrialFrame.cpp, UpdateTrialFrame::process (`:51-60`). A
/// marker shorter than the trial (undefined behaviour in the C++) is skipped.
fn marker_step(ctx: &FrameContext, trial: &Trial, frame: usize) -> Vec<MarkerFramePatch> {
    trial
        .markers
        .iter()
        .map(|m| {
            if frame < m.n_frames() && m.views.iter().all(|v| frame < v.points.len()) {
                update_marker_frame(ctx, m, frame)
            } else {
                MarkerFramePatch {
                    frame,
                    status3d: None,
                    point3d: None,
                    projected: vec![None; m.n_cameras()],
                    error: vec![None; m.n_cameras()],
                }
            }
        })
        .collect()
}

/// Writes the patches of every frame into the trial's markers.
fn apply_marker_patches(trial: &mut Trial, patches: &[Vec<MarkerFramePatch>]) {
    for (i, marker) in trial.markers.iter_mut().enumerate() {
        let marker = Arc::make_mut(marker);
        for frame in patches {
            frame[i].apply(marker);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xma_io::FileVersion;
    use xma_model::{
        CalibrationImage, FitError, Marker, MarkerStatus, Matrix3, Point2, Point3, Pose, Vector3,
    };

    /// Two calibrated cameras, one trial of `n_frames` with one marker seen by both cameras
    /// at every frame.
    fn project(n_frames: usize) -> Project {
        let mut project = Project::new();
        project.calibration_type = CalibrationType::Internal;
        for (i, tx) in [-5.0, 5.0].into_iter().enumerate() {
            let mut cam = Camera::new(format!("Camera {}", i + 1));
            cam.width = 1024;
            cam.height = 1024;
            let mut k = Matrix3::identity();
            k[(0, 0)] = 1000.0;
            k[(1, 1)] = 1000.0;
            k[(0, 2)] = 512.0;
            k[(1, 2)] = 512.0;
            cam.camera_matrix = k;
            let mut image = CalibrationImage::new("cube.tif".to_owned());
            image.calibrated = 1;
            image.rotation = Vector3::new(0.0, 0.1 * tx, 0.0);
            image.translation = Vector3::new(tx, 0.0, 50.0);
            cam.calibration_images.push(image);
            project.cameras.push(Arc::new(cam));
        }
        let mut trial = Trial::default_trial();
        trial.name = "Trial 1".to_owned();
        trial.is_default = false;
        trial.n_frames = n_frames;
        trial.end_frame = n_frames;
        trial.requires_recomputation = true;
        let mut marker = Marker::new(2, n_frames, 8);
        let x = Point3::new(1.0, -2.0, 3.0);
        for c in 0..2 {
            let im = &project.cameras[c].calibration_images[0];
            let model = CameraModel {
                compat: xma_vision::camera::ModelCompat::XMALAB3,
                intrinsics: Intrinsics {
                    camera_matrix: project.cameras[c].camera_matrix,
                    distortion: None,
                },
                grid: None,
            };
            let pose = RigidTransform {
                rotation: im.rotation,
                translation: im.translation,
            };
            for f in 0..n_frames {
                marker.views[c].points[f] = model.project_point(&pose, x);
                marker.views[c].status[f] = MarkerStatus::Tracked;
                marker.views[c].size[f] = 3.0;
            }
        }
        trial.markers.push(Arc::new(marker));
        project.trials.push(Arc::new(trial));
        project
    }

    fn loaded(project: Project) -> LoadedProject {
        LoadedProject {
            project,
            attachments: Vec::new(),
            file_version: FileVersion {
                text: Some("1.0".to_owned()),
                value: 1.0,
            },
            archive_comment: b"XROMM Project File".to_vec(),
            trial_sources: Vec::new(),
            warnings: Vec::new(),
            settings_sources: Vec::new(),
            stored_settings: SettingsPatch::default(),
            compat: CppCompat::XMALAB3,
        }
    }

    /// The document keeps the reader's mode, and its export follows it (ledger F2); the
    /// pose context carries the refinement's part (A3, A4).
    #[test]
    fn compat_mode_reaches_export_and_pose_context() {
        let mut fixed = loaded(project(2));
        fixed.compat = CppCompat::default();
        let mut doc = Document::from_loaded(fixed);
        assert_eq!(doc.compat(), CppCompat::FIXED);
        assert_eq!(doc.pose_context(0).settings.compat, RigidCompat::FIXED);
        let export = crate::Export::Points2d(xma_io::export::points2d::Points2dOptions {
            offset_columns: true,
            // Ignored: the document's mode decides.
            compat: CppCompat::XMALAB3,
            ..Default::default()
        });
        let text = |doc: &mut Document| match doc.export(0, &export).expect("exports") {
            xma_io::export::ExportOutput::File(t) => t,
            other => panic!("{other:?}"),
        };
        assert!(text(&mut doc).starts_with("cam1_offset,cam2_offset,marker001_cam1_X,"));
        doc.set_compat(CppCompat::XMALAB3);
        assert_eq!(doc.pose_context(0).settings.compat, RigidCompat::XMALAB3);
        assert!(text(&mut doc).starts_with("cam0_offsetcam1_offsetmarker001_cam1_X,"));
    }

    #[test]
    fn after_load_computes_sizes_and_updates_the_active_trial() {
        let mut doc = Document::from_loaded(loaded(project(3)));
        let m = &doc.trial(0).markers[0];
        assert_eq!(m.mean_size, 3.0);
        assert_eq!(m.size_range, 15.0); // |3 - 18|, ledger A1
        // Flags as loaded: trial and marker flagged, nothing computed yet.
        assert!(doc.trial(0).requires_recomputation);
        assert!(m.requires_recomputation);
        assert_eq!(m.status3d, vec![MarkerStatus::Undefined; 3]);

        assert_eq!(doc.update_after_load(), Some(0));
        let m = &doc.trial(0).markers[0];
        assert!(!doc.trial(0).requires_recomputation);
        assert!(!m.requires_recomputation);
        assert_eq!(m.status3d, vec![MarkerStatus::Tracked; 3]);
        assert!((m.points3d[2] - Point3::new(1.0, -2.0, 3.0)).norm() < 1e-9);
        assert!(doc.scheduler().queued.is_empty());
    }

    #[test]
    fn cameras_are_marked_calibrated_from_their_images() {
        let mut p = project(1);
        for cam in &mut p.cameras {
            Arc::make_mut(cam).calibrated = false;
        }
        let doc = Document::from_loaded(loaded(p));
        assert!(doc.is_calibrated());
    }

    #[test]
    fn an_uncalibrated_camera_stops_the_update_after_load() {
        let mut p = project(2);
        Arc::make_mut(&mut p.cameras[1]).calibration_images[0].calibrated = 0;
        let mut doc = Document::from_loaded(loaded(p));
        assert_eq!(doc.update_after_load(), None);
        assert!(doc.trial(0).requires_recomputation);
        assert_eq!(doc.trial(0).markers[0].status3d[0], MarkerStatus::Undefined);
    }

    #[test]
    fn no_calibration_images_means_no_calibration() {
        let mut p = project(2);
        for cam in &mut p.cameras {
            Arc::make_mut(cam).calibration_images.clear();
        }
        let mut doc = Document::from_loaded(loaded(p));
        assert_eq!(doc.project().calibration_type, CalibrationType::None);
        assert_eq!(doc.update_after_load(), None);

        // An explicit update triangulates nothing and copies the 2D points as projections.
        let m0 = doc.trial(0).markers[0].clone();
        assert!(doc.update_trial(0));
        let m = &doc.trial(0).markers[0];
        assert_eq!(m.status3d, vec![MarkerStatus::Undefined; 2]);
        assert_eq!(m.points3d, m0.points3d);
        assert_eq!(m.views[1].projected, m0.views[1].projected);
        assert!(!m.requires_recomputation);
    }

    #[test]
    fn a_marker_seen_by_one_camera_keeps_its_old_values() {
        let mut p = project(2);
        {
            let t = Arc::make_mut(&mut p.trials[0]);
            let m = Arc::make_mut(&mut t.markers[0]);
            m.views[1].status[1] = MarkerStatus::Undefined;
            m.points3d[1] = Point3::new(7.0, 8.0, 9.0);
            m.views[0].projected[1] = Point2::new(1.0, 1.0);
            m.views[0].error[1] = 0.25;
        }
        let mut doc = Document::from_loaded(loaded(p));
        doc.update_after_load();
        let m = &doc.trial(0).markers[0];
        assert_eq!(m.status3d[1], MarkerStatus::Undefined);
        assert_eq!(m.points3d[1], Point3::new(7.0, 8.0, 9.0));
        assert_eq!(m.views[0].projected[1], Point2::new(1.0, 1.0));
        assert_eq!(m.views[0].error[1], 0.25);
        assert_eq!(m.status3d[0], MarkerStatus::Tracked);
    }

    #[test]
    fn an_update_with_nothing_to_recompute_waits_for_the_next_finalize() {
        let mut doc = Document::from_loaded(loaded(project(2)));
        doc.update_after_load();
        let before = doc.trial(0).clone();

        // A marker flagged on its own, with the trial clear: nothing runs, the trial is
        // queued, and the marker keeps its flag (no job, so no finalize).
        {
            let t = Arc::make_mut(&mut doc.project.trials[0]);
            Arc::make_mut(&mut t.markers[0]).requires_recomputation = true;
        }
        assert!(!doc.update_trial(0));
        assert_eq!(doc.scheduler().queued, vec![0]);
        assert!(doc.trial(0).markers[0].requires_recomputation);
        assert_eq!(doc.trial(0).markers[0].status3d, before.markers[0].status3d);

        // The next update that runs jobs finalizes every queued trial.
        doc.set_trial_requires_recomputation(0, true);
        assert!(doc.update_trial(0));
        assert!(doc.scheduler().queued.is_empty());
        assert!(!doc.trial(0).markers[0].requires_recomputation);
    }

    #[test]
    fn a_clear_marker_in_a_flagged_trial_is_only_reprojected() {
        let mut doc = Document::from_loaded(loaded(project(1)));
        {
            let t = Arc::make_mut(&mut doc.project.trials[0]);
            let m = Arc::make_mut(&mut t.markers[0]);
            m.requires_recomputation = false;
            m.points3d[0] = Point3::new(0.0, 0.0, 0.0);
            m.status3d[0] = MarkerStatus::Set;
        }
        doc.update_after_load();
        let m = &doc.trial(0).markers[0];
        assert_eq!(m.points3d[0], Point3::new(0.0, 0.0, 0.0));
        assert_eq!(m.status3d[0], MarkerStatus::Set);
        // The origin seen from camera 1 at translation (-5, 0, 50).
        assert_eq!(m.views[0].projected[0], Point2::new(412.0, 512.0));
        assert!(m.views[0].error[0] > 1.0);
    }

    /// `project(n_frames)` with the six markers and the two bodies of the rigid-body tests:
    /// A (markers 0 to 3) and B (markers 4 and 5 and a virtual point in A).
    fn project_with_bodies(n_frames: usize) -> Project {
        use crate::rigid_bodies::tests::{body, camera, marker, virtual_to};
        let mut p = project(n_frames);
        let cams = [camera(-5.0), camera(5.0)];
        let t = Arc::make_mut(&mut p.trials[0]);
        t.markers = (0..6)
            .map(|m| {
                let mut mk = marker(&cams, m, n_frames);
                mk.requires_recomputation = true;
                Arc::new(mk)
            })
            .collect();
        let mut a = body(&[0, 1, 2, 3], n_frames);
        let mut b = body(&[4, 5], n_frames);
        // As the reader leaves them: `initialised` is derived on load.
        a.initialised = false;
        b.initialised = false;
        b.virtual_points.push(virtual_to(0));
        t.rigid_bodies = vec![Arc::new(a), Arc::new(b)];
        p
    }

    #[test]
    fn the_update_computes_poses_and_finalize_filters_them() {
        let mut p = project_with_bodies(20);
        {
            let t = Arc::make_mut(&mut p.trials[0]);
            t.recording_speed = 100.0;
            t.cutoff_frequency = 10.0;
        }
        p.algorithm_settings.optimize_rigid_body = false;
        let mut doc = Document::from_loaded(loaded(p));
        assert!(doc.trial(0).rigid_bodies.iter().all(|b| b.initialised));
        doc.update_after_load();
        for b in &doc.trial(0).rigid_bodies {
            assert!(b.raw.valid.iter().all(|&v| v));
            assert!(b.filtered.valid.iter().all(|&v| v));
            assert!(b.raw.errors[3].mean_2d > 0.0);
            assert!(b.filtered.errors[3].mean_3d > 0.0);
        }

        // The trial dialog's filter-only update with the cutoff removed: poses recomputed
        // (the same), filtering refused, filtered errors zero, filtered poses left.
        let before = doc.trial(0).clone();
        Arc::make_mut(&mut doc.project.trials[0]).cutoff_frequency = 0.0;
        doc.recompute_and_filter_rigid_body_transformations(0);
        for (b, old) in doc.trial(0).rigid_bodies.iter().zip(&before.rigid_bodies) {
            assert_eq!(b.raw, old.raw);
            assert!(b.filtered.valid.iter().all(|&v| !v));
            assert_eq!(b.filtered.poses, old.filtered.poses);
            assert!(b.filtered.errors.iter().all(|e| *e == FitError::default()));
        }

        // The unfiltered export recomputes without filtering.
        Arc::make_mut(&mut doc.project.trials[0]).cutoff_frequency = 10.0;
        doc.recompute_rigid_bodies_for_export(0, &[1], false);
        assert!(
            doc.trial(0).rigid_bodies[1]
                .filtered
                .valid
                .iter()
                .all(|&v| !v)
        );
        doc.recompute_rigid_bodies_for_export(0, &[1], true);
        assert!(
            doc.trial(0).rigid_bodies[1]
                .filtered
                .valid
                .iter()
                .all(|&v| v)
        );
        assert!(
            doc.trial(0).rigid_bodies[0]
                .filtered
                .valid
                .iter()
                .all(|&v| !v)
        );
        assert!(doc.warnings().is_empty());
    }

    #[test]
    fn pose_vectors_of_the_wrong_length_are_reset_before_the_update() {
        let mut p = project_with_bodies(3);
        p.algorithm_settings.optimize_rigid_body = false;
        let stale = Pose {
            rotation: Vector3::new(0.1, 0.2, 0.3),
            translation: Vector3::new(1.0, 2.0, 3.0),
        };
        {
            let t = Arc::make_mut(&mut p.trials[0]);
            // A: too short, with values that `init` clears.
            let a = Arc::make_mut(&mut t.rigid_bodies[0]);
            a.init(1);
            a.filtered.poses[0] = stale;
            // B: the right length; its stale filtered poses survive (no cutoff, nothing
            // filtered).
            let b = Arc::make_mut(&mut t.rigid_bodies[1]);
            b.filtered.poses[2] = stale;
        }
        let mut doc = Document::from_loaded(loaded(p));
        doc.update_after_load();
        let t = doc.trial(0);
        let (a, b) = (&t.rigid_bodies[0], &t.rigid_bodies[1]);
        assert_eq!(a.n_frames(), 3);
        assert_eq!(a.filtered.poses, vec![Pose::ZERO; 3]);
        assert!(a.raw.valid.iter().all(|&v| v));
        assert_eq!(b.filtered.poses[2], stale);
        assert!(b.filtered.valid.iter().all(|&v| !v));
    }

    #[test]
    fn without_calibration_no_pose_is_computed_and_errors_are_kept() {
        let mut p = project_with_bodies(2);
        for cam in &mut p.cameras {
            Arc::make_mut(cam).calibration_images.clear();
        }
        let stale = Pose {
            rotation: Vector3::new(0.1, 0.2, 0.3),
            translation: Vector3::new(1.0, 2.0, 3.0),
        };
        {
            let t = Arc::make_mut(&mut p.trials[0]);
            let a = Arc::make_mut(&mut t.rigid_bodies[0]);
            a.raw.poses[1] = stale;
            a.raw.valid[1] = true;
            a.raw.errors[1].mean_2d = 4.0;
            a.filtered.errors[1].mean_2d = 4.0;
        }
        let mut doc = Document::from_loaded(loaded(p));
        assert_eq!(doc.update_after_load(), None);
        assert!(doc.update_trial(0));
        let a = &doc.trial(0).rigid_bodies[0];
        assert_eq!(a.raw.valid, vec![false, false]);
        assert_eq!(a.raw.poses[1], stale);
        // computePose returns before updateError; the finalize's filtered update still runs.
        assert_eq!(a.raw.errors[1].mean_2d, 4.0);
        assert_eq!(a.filtered.errors[1], FitError::default());
    }
}

/// Round 13 (review engine-document, findings 8 and 9): the comparison of camera models
/// against the expression it replaces, and the finalize's repeat against running it again.
#[cfg(test)]
mod review13_tests {
    use super::*;
    use xma_model::{Matrix3, Point2};
    use xma_vision::undistortion::LwmTransform;

    /// The expression [`same_model`] replaces, kept as the reference.
    fn old_same(a: &CameraModel, b: &CameraModel) -> bool {
        a == b || format!("{a:?}") == format!("{b:?}")
    }

    fn lattice_model() -> CameraModel {
        let mut distorted = Vec::new();
        let mut references = Vec::new();
        for j in 0..8 {
            for i in 0..8 {
                let (x, y) = (f64::from(i) * 40.0, f64::from(j) * 40.0);
                references.push(Point2::new(x, y));
                distorted.push(Point2::new(
                    x + 0.01 * x * y / 300.0,
                    y - 0.02 * x * x / 300.0,
                ));
            }
        }
        let inlier = vec![true; references.len()];
        let mut k = Matrix3::identity();
        k[(0, 0)] = 1000.0;
        k[(1, 1)] = 1000.0;
        CameraModel {
            intrinsics: Intrinsics {
                camera_matrix: k,
                distortion: Some([0.0, -0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
            },
            grid: fit_grid_model(&distorted, &references, &inlier, 12),
            compat: model_compat(CppCompat::FIXED),
        }
    }

    /// Applies `f` to one transform's lists, as `Vec`s.
    fn edit(
        m: &mut CameraModel,
        undistort: bool,
        f: impl FnOnce(&mut Vec<Point2>, &mut Vec<[f64; 6]>, &mut Vec<[f64; 6]>, &mut Vec<f64>),
    ) {
        let g = m.grid.as_mut().unwrap();
        let t: &mut LwmTransform = if undistort {
            &mut g.undistort
        } else {
            &mut g.distort
        };
        let (mut c, mut a, mut b, mut r) = (
            t.control_points.to_vec(),
            t.a.to_vec(),
            t.b.to_vec(),
            t.radii.to_vec(),
        );
        f(&mut c, &mut a, &mut b, &mut r);
        t.control_points = c.into();
        t.a = a.into();
        t.b = b.into();
        t.radii = r.into();
    }

    /// Old against new on pairs that differ nowhere, in one number, in the sign of a zero,
    /// with NaN of several payloads and signs in one or both, in several places at once, in
    /// a list's length, in the grid's presence, in the intrinsics and in the mode.
    #[test]
    fn review13_same_model_answers_as_the_debug_comparison() {
        let base = lattice_model();
        let nan_a = f64::NAN;
        let nan_b = f64::from_bits(f64::NAN.to_bits() ^ 0x5);
        let nan_neg = -f64::NAN;
        type Change = Box<dyn Fn(&mut CameraModel)>;
        let set_a = |u: bool, i: usize, k: usize, v: f64| -> Change {
            Box::new(move |m| edit(m, u, |_, a, _, _| a[i][k] = v))
        };
        let set_r = |u: bool, i: usize, v: f64| -> Change {
            Box::new(move |m| edit(m, u, |_, _, _, r| r[i] = v))
        };
        let set_c = |u: bool, i: usize, x: f64| -> Change {
            Box::new(move |m| edit(m, u, |c, _, _, _| c[i].x = x))
        };
        let mut changes: Vec<(&str, Change)> = vec![
            ("none", Box::new(|_| {})),
            ("a +1 ulp", {
                let v = base.grid.as_ref().unwrap().distort.a[3][2];
                set_a(false, 3, 2, f64::from_bits(v.to_bits() + 1))
            }),
            ("a = 0", set_a(true, 5, 1, 0.0)),
            ("a = -0", set_a(true, 5, 1, -0.0)),
            ("a = NaN", set_a(false, 7, 0, nan_a)),
            ("a = NaN payload", set_a(false, 7, 0, nan_b)),
            ("a = -NaN", set_a(false, 7, 0, nan_neg)),
            ("a = NaN elsewhere", set_a(false, 9, 0, nan_a)),
            ("radius NaN", set_r(true, 0, nan_a)),
            ("radius inf", set_r(true, 0, f64::INFINITY)),
            ("radius -inf", set_r(true, 0, f64::NEG_INFINITY)),
            ("radius 0", set_r(false, 63, 0.0)),
            ("radius -0", set_r(false, 63, -0.0)),
            ("radius subnormal", set_r(false, 63, f64::from_bits(1))),
            ("point x 0", set_c(true, 0, 0.0)),
            ("point x -0", set_c(true, 0, -0.0)),
            ("point x NaN", set_c(true, 0, nan_b)),
            (
                "shorter",
                Box::new(|m| {
                    edit(m, false, |c, a, b, r| {
                        c.pop();
                        a.pop();
                        b.pop();
                        r.pop();
                    })
                }),
            ),
            (
                "radii shorter only",
                Box::new(|m| {
                    edit(m, true, |_, _, _, r| {
                        r.pop();
                    });
                }),
            ),
            ("no grid", Box::new(|m| m.grid = None)),
            (
                "matrix NaN",
                Box::new(move |m| m.intrinsics.camera_matrix[(0, 1)] = nan_a),
            ),
            (
                "matrix -0",
                Box::new(|m| m.intrinsics.camera_matrix[(0, 1)] = -0.0),
            ),
            (
                "distortion -0 to 0",
                Box::new(|m| m.intrinsics.distortion.as_mut().unwrap()[1] = 0.0),
            ),
            (
                "distortion NaN",
                Box::new(move |m| m.intrinsics.distortion.as_mut().unwrap()[4] = nan_neg),
            ),
            (
                "no distortion",
                Box::new(|m| m.intrinsics.distortion = None),
            ),
            (
                "mode",
                Box::new(|m| m.compat = model_compat(CppCompat::XMALAB3)),
            ),
        ];
        // Combinations: a NaN in both with a sign of zero apart, and two NaNs.
        changes.push((
            "NaN and -0",
            Box::new(move |m| {
                edit(m, false, |_, a, _, _| a[7][0] = nan_a);
                edit(m, true, |_, a, _, _| a[5][1] = -0.0);
            }),
        ));
        changes.push((
            "NaN and 0",
            Box::new(move |m| {
                edit(m, false, |_, a, _, _| a[7][0] = nan_b);
                edit(m, true, |_, a, _, _| a[5][1] = 0.0);
            }),
        ));
        changes.push((
            "NaN and NaN matrix",
            Box::new(move |m| {
                edit(m, false, |_, a, _, _| a[7][0] = nan_a);
                m.intrinsics.camera_matrix[(0, 1)] = nan_b;
            }),
        ));
        let models: Vec<(&str, CameraModel)> = changes
            .iter()
            .map(|(name, f)| {
                let mut m = base.clone();
                f(&mut m);
                (*name, m)
            })
            .collect();
        let mut true_seen = 0;
        let mut debug_only = 0;
        for (na, a) in &models {
            for (nb, b) in &models {
                let old = old_same(a, b);
                assert_eq!(same_model(a, b), old, "{na} against {nb}");
                true_seen += usize::from(old);
                debug_only += usize::from(old && a != b);
            }
        }
        // The cases where only the Debug text decides (NaN in both) are reached.
        assert!(debug_only > 5, "{debug_only}");
        assert!(true_seen > models.len());
    }

    /// The finalize run again right after itself changes nothing where
    /// `finalize_repeat_is_noop` says so, on the pig project with a 25 Hz cutoff, filtered
    /// by pose and by 3D points; and a rotation longer than 2π is caught.
    #[test]
    fn review13_finalize_repeat_is_noop_on_pig() {
        for filter_3d in [false, true] {
            let Some(ds) = xma_oracle::dataset("pig_clean_cut25") else {
                return;
            };
            let open = || {
                let mut doc = Document::open(&ds.project).unwrap();
                let mut settings = doc.project.algorithm_settings.clone();
                settings.filter_3d_points = filter_3d;
                doc.project.algorithm_settings = settings;
                doc.update_after_load();
                doc
            };
            let mut once = open();
            let mut thrice = open();
            assert!(once.finalize_repeat_is_noop(0));
            once.finalize_trial(0);
            for _ in 0..3 {
                thrice.finalize_trial(0);
            }
            assert_eq!(
                format!("{:?}", once.project),
                format!("{:?}", thrice.project),
                "filter_3d {filter_3d}"
            );
            assert_eq!(once.warnings, thrice.warnings);

            // A raw rotation longer than 2π after one turned the other way: a second pass
            // replaces it again, and the check refuses.
            let make_long = || {
                let mut doc = open();
                let t = Arc::make_mut(&mut doc.project.trials[0]);
                let b = Arc::make_mut(&mut t.rigid_bodies[0]);
                let f = (1..b.raw.valid.len())
                    .find(|&f| b.raw.valid[f - 1] && b.raw.valid[f])
                    .unwrap();
                let r = b.raw.poses[f - 1].rotation;
                b.raw.poses[f].rotation = -r / r.norm() * 7.0;
                doc
            };
            let mut long = make_long();
            let mut long_twice = make_long();
            assert!(!long.finalize_repeat_is_noop(0));
            long.finalize_trial(0);
            long_twice.finalize_trial(0);
            long_twice.finalize_trial(0);
            assert_ne!(
                format!("{:?}", long.project.trials[0].rigid_bodies[0].raw),
                format!("{:?}", long_twice.project.trials[0].rigid_bodies[0].raw)
            );
        }
    }

    /// Timing of the comparison after a refit (every number of the grid apart), old
    /// expression against [`same_model`], on the pig project's first camera.
    #[test]
    #[ignore = "timing; run with --ignored --nocapture"]
    fn perf13_same_model_after_refit() {
        let Some(ds) = xma_oracle::dataset("pig_clean") else {
            return;
        };
        let doc = Document::open(&ds.project).unwrap();
        let a = doc.camera(0).model.clone();
        assert!(a.grid.is_some());
        let mut b = a.clone();
        edit(&mut b, false, |_, _, _, r| {
            for v in r.iter_mut() {
                *v *= 1.0 + 1e-12;
            }
        });
        edit(&mut b, true, |_, _, _, r| {
            for v in r.iter_mut() {
                *v *= 1.0 + 1e-12;
            }
        });
        let mut nan = a.clone();
        nan.intrinsics.camera_matrix[(0, 1)] = f64::NAN;
        let nan2 = nan.clone();
        let time = |f: &dyn Fn() -> bool| {
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                std::hint::black_box(f());
                best = best.min(t.elapsed().as_secs_f64());
            }
            best * 1e3
        };
        eprintln!(
            "refit: old {:.2} ms, new {:.3} ms; NaN in both: old {:.2} ms, new {:.3} ms",
            time(&|| old_same(&a, &b)),
            time(&|| same_model(&a, &b)),
            time(&|| old_same(&nan, &nan2)),
            time(&|| same_model(&nan, &nan2)),
        );
    }
}
