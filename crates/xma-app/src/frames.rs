//! The frame service: decoded images for the camera views and the Detailed View panel.
//!
//! One [`Frames`] lives in the application and is reached from panels and views as
//! `cx.frames`. Each frame of the interface the application calls [`Frames::sync`], which
//! works out the image every camera shows in the current workspace (as
//! `GLCameraView::drawTexture` chooses it), asks for it without ever blocking, and hands
//! each image that has arrived to the image widget's [`ImageStore`] under the camera's key
//! ([`Frames::image_key`]). The camera view and the detail view of a camera show that key.
//!
//! What each camera shows:
//!
//! - **Marker tracking**: frame `active_frame` of the active trial's video for the camera,
//!   from the files [`Session::video_files`] gives (the project's paths, or the folder or
//!   file found or chosen after loading). The default trial has no video: nothing is shown,
//!   as in the C++.
//! - **Calibration** (internal calibration only): the active calibration image, a still
//!   stored in the project (decoded with [`xma_video::image_file`], found through the
//!   document's attachments, else on disk), or the frame of the camera's calibration
//!   sequence.
//! - **Undistortion**: the undistortion-grid image, a still stored in the project.
//!
//! A camera hidden with the camera selector wants nothing in any workspace: its image is
//! not read, filtered or uploaded, its texture is dropped, and playback does not wait for
//! it (ledger U166).
//!
//! The undistorted and false-colour images of the calibration and undistortion workspaces
//! are not served here; `camera_view::overlay_undistortion` and
//! `camera_view::overlay_calibration` may supply them.
//!
//! Videos are opened on a worker thread with [`open_source`] and read through one shared
//! [`FrameCache`], whose ready callback requests a repaint. While a frame is on its way the
//! view keeps the previous image. The visual filter (Display Options, "filter images") is applied to a
//! decoded frame before the views are drawn, not on a worker thread, and during a tracking
//! run the frame reached is written to its texture before the views are drawn with its
//! points, so that a view never stays on an older frame than the frame bar
//! (`tests/view_follows_tracking.rs`). A frame that cannot be read shows no image and writes one
//! line to the Console (the C++ shows frame 0 there). A source that cannot be opened (for
//! example ffmpeg missing) shows its message in the view and writes it to the Console once.
//!
//! The camera's flip flag mirrors every image horizontally, as `cv::flip(image, image, 1)`
//! does in the C++ readers (`core/Image.cpp`, `AviVideo.cpp`, `CineVideo.cpp`, and so
//! `ImageSequence`, `CalibrationImage` and `UndistortionObject` through `Image`).
//!
//! Ported from `core/VideoStream.cpp` (`bindTexture`), `core/Image.cpp` (the flip),
//! `core/CalibrationSequence.cpp` and `core/UndistortionObject.cpp` (`bindTexture`, the
//! distorted image), as `ui/GLCameraView.cpp`, `GLCameraView::drawTexture`, uses them.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui_wgpu;
use xma_engine::Document;
use xma_model::CalibrationType;
use xma_video::cache::{CacheOptions, FrameCache, SourceId};
use xma_video::image_file::ColourImage;
use xma_video::open::{OpenOptions, open_source};
use xma_video::{Frame, VideoError};
use xma_vision::remap::{DisplacementMap, displacement_image};
use xma_vision::undistortion::{LookupTable, create_lookup_table};
use xma_vision::visual_filter::VisualFilter;

use crate::camera_view::CameraViewState;
use crate::log::AppLog;
use crate::panels::detail_view::DetailViewState;
use crate::session::{Session, VideoLocation};
use crate::settings::UiSettings;
use crate::state::{AppState, Workspace};
use crate::widgets::image_view::{DisplayImage, ImageStore};

/// Memory for decoded frames: 1 GiB (the cache's default).
pub const CACHE_BUDGET_BYTES: usize = 1 << 30;

/// Which image a camera wants.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ImageId {
    /// Frame `frame` (0-based) of the video made of `files`.
    Video { files: Arc<[PathBuf]>, frame: usize },
    /// A still image of camera `camera`: the name it is stored under in the project, or a
    /// path on disk. Two cameras' images may have the same name (each is stored under its
    /// camera's folder), so the camera is part of the key.
    Still { camera: usize, name: String },
}

/// What a camera's view shows now.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CameraImage {
    /// The image wanted for the current state, if any.
    pub want: Option<ImageId>,
    /// The image whose pixels are in the store under the camera's key.
    pub shown: Option<ImageId>,
    /// A message drawn in the view instead of an image (a source that cannot be opened).
    pub message: Option<String>,
    /// The wanted image could not be read; nothing is shown.
    pub failed: bool,
}

/// The kind of frame change a latency sample belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Step {
    Forward,
    Back,
    Jump,
}

/// Time from a frame change to its image being in a texture, per camera and change.
#[derive(Clone, Debug, Default)]
pub struct Timings {
    /// `(step, frame, seconds until the image was handed to the uploader, seconds until
    /// the texture held it)`, newest last, at most 4096.
    pub samples: VecDeque<(Step, usize, f64, f64)>,
    /// Distinct images that reached a texture, per camera, with the time.
    pub shown: Vec<(usize, Instant)>,
}

impl Timings {
    /// Median of the "in a texture" times of one kind of change, in seconds.
    #[must_use]
    pub fn median(&self, step: Step) -> Option<f64> {
        let mut v: Vec<f64> = self
            .samples
            .iter()
            .filter(|s| s.0 == step)
            .map(|s| s.3)
            .collect();
        if v.is_empty() {
            return None;
        }
        v.sort_by(f64::total_cmp);
        Some(v[v.len() / 2])
    }
}

/// Per-view interface state the C++ keeps in its view widgets: one entry per camera.
#[derive(Default)]
pub struct ViewStates {
    /// The camera views (`CameraViewWidget` with its `GLCameraView`).
    pub cameras: Vec<CameraViewState>,
    /// The Detailed View panel's views (`CameraViewDetailWidget`).
    pub details: Vec<DetailViewState>,
    /// Undistortion grids per camera, computed once (`xma_engine::view::calibration`).
    pub grids: HashMap<usize, Arc<xma_engine::view::calibration::UndistortionGridPoints>>,
    /// Calibration image points per (camera, image), computed once.
    pub calibration_points:
        HashMap<(usize, usize), Arc<xma_engine::view::calibration::CalibrationImagePoints>>,
    /// Per camera, what its Calibration Info frame shows and what that was computed from
    /// (`camera_view::info_frames::cached_calibration_info`).
    pub calibration_info: HashMap<
        usize,
        (
            crate::camera_view::info_frames::CalibrationInfoKey,
            Arc<xma_engine::view::calibration::CalibrationInfo>,
        ),
    >,
    /// Per camera, what the external-calibration frame's boxes hold (values loaded or typed
    /// and not yet applied).
    pub external:
        HashMap<usize, crate::camera_view::info_frames::external::ExternalCalibrationFrame>,
    /// Rigid-body meshes and distortion maps of the 3D drawings, the 3D world view's
    /// viewpoint, and what the 3D drawings drew last.
    pub meshes: crate::meshes::Meshes,
}

enum SourceSlot {
    Opening,
    Open(SourceId),
    Failed(String),
}

enum StillSlot {
    Decoding,
    /// The grey image, and the colour image of a colour file.
    Ready(Arc<Frame>, Option<Arc<ColourImage>>),
    Failed(String),
}

/// The files of a video as a map key. Equal keys hold equal files (compared by pointer
/// first: the views' lists are shared, [`Frames::video_files_of`]); the hash reads only
/// their number and the first and last paths, so that looking up a long image sequence
/// does not hash every path on every repaint. Equal files hash alike, so a key is found
/// exactly when the files are equal.
#[derive(Clone, PartialEq, Eq)]
struct Files(Arc<[PathBuf]>);

impl Files {
    fn of(files: &Arc<[PathBuf]>) -> Self {
        Self(files.clone())
    }
}

impl std::hash::Hash for Files {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        self.0.len().hash(h);
        self.0.first().hash(h);
        self.0.last().hash(h);
    }
}

/// A visual filter setting as a hashable key.
type FilterKey = (i32, u32, u32, u32, u32);

/// A filtered frame: which frame, whether mirrored, which filter.
type FilteredKey = (ImageId, bool, FilterKey);

fn filter_key(f: &VisualFilter) -> FilterKey {
    (
        f.krad,
        f.gsigma.to_bits(),
        f.img_wt.to_bits(),
        f.blur_wt.to_bits(),
        f.gamma.to_bits(),
    )
}

/// An image derived from a camera's stored image for display (`Frames::derived`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Derived {
    /// The grid image undistorted with the grid (`UndistortionObject::undistortedImage`).
    UndistortedGrid,
    /// A false-colour displacement map (`UndistortionObject::computeTmpImage`).
    Map(DisplacementMap),
    /// Calibration image `n` undistorted with the grid and the lens model
    /// (`CalibrationImage::undistortedImage`, `CalibrationSequence::undistort`).
    UndistortedCalibration(usize),
}

enum DerivedSlot {
    Computing,
    Ready(egui::Id),
    Failed,
}

/// Number of filtered frames kept.
const FILTERED_KEPT: usize = 64;

enum Done {
    Opened(
        Arc<[PathBuf]>,
        Result<Box<dyn xma_video::FrameSource>, String>,
    ),
    /// Camera, name, the decoded still, and the [`Frames`] still generation it was started
    /// in (a result from a project closed since is dropped).
    Still(
        usize,
        String,
        Result<(Frame, Option<ColourImage>), String>,
        u64,
    ),
    /// Camera, kind, lookup table, image, and the [`Frames`] derived-image generation
    /// it was started in (a result from before [`Frames::calibration_changed`] is dropped).
    Derived(
        usize,
        Derived,
        Option<Arc<LookupTable>>,
        Option<DisplayImage>,
        u64,
    ),
    /// The colour image of one file of an image sequence (`None`: a grey file).
    Colour(Arc<[PathBuf]>, ImageId, Result<Option<ColourImage>, String>),
}

/// Where a derived image comes from.
enum DeriveSource {
    /// A still stored in the project (or on disk).
    Still(Result<(Vec<u8>, PathBuf), String>),
    /// A frame already decoded (a calibration sequence's frame).
    Frame(Arc<Frame>),
}

/// Number of colour frames kept.
const COLOUR_KEPT: usize = 32;

/// A trial camera's files as [`Session::video_files`] made them, and what they were made
/// from.
struct VideoFiles {
    names: Vec<String>,
    location: Option<VideoLocation>,
    files: Arc<[PathBuf]>,
}

struct Pending {
    since: Instant,
    step: Step,
    frame: usize,
    /// Handed to the uploader at; waiting for the texture generation to change from the
    /// value it had then.
    handed: Option<(f64, u64)>,
}

/// The frame service. See the module documentation.
pub struct Frames {
    cache: Arc<FrameCache>,
    repaint: Arc<Mutex<Option<egui::Context>>>,
    store: Option<ImageStore>,
    sources: HashMap<Files, SourceSlot>,
    /// Decoded stills by camera and name.
    stills: HashMap<(usize, String), StillSlot>,
    /// Counts project closes: a still decoded for an older generation is dropped when it
    /// arrives.
    still_generation: u64,
    tx: mpsc::Sender<Done>,
    rx: mpsc::Receiver<Done>,
    /// One per camera of the open project.
    pub cameras: Vec<CameraImage>,
    /// Per camera: whether the shown image is flipped (decided when uploaded).
    flipped: Vec<bool>,
    pending: Vec<Option<Pending>>,
    /// Per camera: when the wanted video frame was last requested (re-requested after 2 s).
    requested: Vec<Option<(ImageId, Instant)>>,
    logged: HashSet<String>,
    /// Interface state of every view.
    pub views: ViewStates,
    /// Scrubbing measurements.
    pub timings: Timings,
    last_frame: Vec<Option<usize>>,
    open_options: OpenOptions,
    /// The visual filter to apply to video frames now, if any.
    filter: Option<VisualFilter>,
    /// Per camera: the filter the shown image was made with.
    shown_filter: Vec<Option<FilterKey>>,
    /// Filtered frames (8-bit, mirrored when the camera is flipped), newest last.
    filtered: VecDeque<(FilteredKey, Arc<Frame>)>,
    /// Derived display images and the lookup tables they come from.
    derived: HashMap<(usize, Derived), DerivedSlot>,
    luts: HashMap<usize, Arc<LookupTable>>,
    /// Derived images made without a texture store (headless tests).
    derived_headless: HashMap<(usize, Derived), Arc<DisplayImage>>,
    /// Counts [`Frames::calibration_changed`] and project changes: a derived image
    /// computed for an older generation is dropped when it arrives.
    derived_generation: u64,
    /// Per image-sequence source: whether its files are colour images (`None` while
    /// the first one is being looked at).
    sequence_colour: HashMap<Files, Option<bool>>,
    /// Colour frames of colour image sequences, newest last. `None` for a file of a colour
    /// sequence that has no colour image (a grey file, or one that cannot be read): it is
    /// shown grey and not decoded again.
    colour_frames: VecDeque<(ImageId, Option<Arc<ColourImage>>)>,
    colour_decoding: HashSet<ImageId>,
    /// Per camera: the shown image is the colour one.
    shown_colour: Vec<bool>,
    /// Per camera: images handed for display (to the uploader, or headless, as shown).
    handed: Vec<u64>,
    /// Per trial and camera, its video's files ([`Frames::video_files_of`]).
    video_files: HashMap<(usize, usize), VideoFiles>,
    /// A tracking run or a refinement is open (`AppState::disable_draw`): each frame the
    /// run reaches is written to its texture on this thread before the views are drawn
    /// with its points ([`Frames::sync`]).
    in_step: bool,
}

impl Default for Frames {
    fn default() -> Self {
        Self::new()
    }
}

impl Frames {
    /// An empty service with no texture store (images are decoded but not uploaded until
    /// [`Frames::set_render_state`]).
    #[must_use]
    pub fn new() -> Self {
        let cache = Arc::new(FrameCache::new(CacheOptions::with_budget(
            CACHE_BUDGET_BYTES,
        )));
        let repaint: Arc<Mutex<Option<egui::Context>>> = Arc::new(Mutex::new(None));
        let r = repaint.clone();
        let _ = cache.set_on_ready(Box::new(move |_, _| {
            if let Some(ctx) = r.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
                ctx.request_repaint_of(egui::ViewportId::ROOT);
            }
        }));
        let (tx, rx) = mpsc::channel();
        Self {
            cache,
            repaint,
            store: None,
            sources: HashMap::new(),
            stills: HashMap::new(),
            still_generation: 0,
            tx,
            rx,
            cameras: Vec::new(),
            flipped: Vec::new(),
            pending: Vec::new(),
            requested: Vec::new(),
            logged: HashSet::new(),
            views: ViewStates::default(),
            timings: Timings::default(),
            last_frame: Vec::new(),
            open_options: OpenOptions::default(),
            filter: None,
            shown_filter: Vec::new(),
            filtered: VecDeque::new(),
            derived: HashMap::new(),
            luts: HashMap::new(),
            derived_headless: HashMap::new(),
            derived_generation: 0,
            sequence_colour: HashMap::new(),
            colour_frames: VecDeque::new(),
            colour_decoding: HashSet::new(),
            shown_colour: Vec::new(),
            handed: Vec::new(),
            video_files: HashMap::new(),
            in_step: false,
        }
    }

    /// The context to wake when a frame arrives.
    pub fn set_repaint_context(&mut self, ctx: &egui::Context) {
        *self.repaint.lock().unwrap_or_else(|e| e.into_inner()) = Some(ctx.clone());
    }

    /// Creates the texture store on eframe's wgpu device. Until then images are decoded
    /// but not shown (headless tests).
    pub fn set_render_state(&mut self, rs: &egui_wgpu::RenderState, ctx: &egui::Context) {
        self.set_repaint_context(ctx);
        self.store = Some(ImageStore::new(rs, Some(ctx.clone())));
        // Upload again what was decoded before the store existed.
        for c in &mut self.cameras {
            c.shown = None;
        }
        // Derived images made headless are made again for the store.
        self.derived.clear();
        self.derived_headless.clear();
    }

    /// The visual filter applied to video frames now (`Image::loadTexture`), if any.
    #[must_use]
    pub fn visual_filter(&self) -> Option<VisualFilter> {
        self.filter
    }

    /// Number of filtered frames kept.
    #[must_use]
    pub fn filtered_len(&self) -> usize {
        self.filtered.len()
    }

    /// The texture store, once there is one.
    #[must_use]
    pub fn store(&self) -> Option<&ImageStore> {
        self.store.as_ref()
    }

    /// The frame cache (statistics, tests).
    #[must_use]
    pub fn cache(&self) -> &FrameCache {
        &self.cache
    }

    /// The store key of a camera's image, shown by its camera view and its detail view.
    #[must_use]
    pub fn image_key(camera: usize) -> egui::Id {
        egui::Id::unique(("xma camera image", camera))
    }

    /// A project was opened: per-camera state for its cameras, scale and bias from the
    /// project (`Camera::getImageAdjustment`).
    pub fn project_opened(&mut self, doc: &Document) {
        self.project_closed();
        let cams = &doc.project().cameras;
        let n = cams.len();
        self.cameras = vec![CameraImage::default(); n];
        self.flipped = vec![false; n];
        self.pending = (0..n).map(|_| None).collect();
        self.requested = vec![None; n];
        self.last_frame = vec![None; n];
        self.shown_filter = vec![None; n];
        self.shown_colour = vec![false; n];
        self.handed = vec![0; n];
        self.views.cameras = cams.iter().map(|c| CameraViewState::new(c)).collect();
        self.views.details = cams.iter().map(|c| DetailViewState::new(c)).collect();
    }

    /// The project was closed: every source and texture goes.
    pub fn project_closed(&mut self) {
        for slot in self.sources.values() {
            if let SourceSlot::Open(id) = slot {
                self.cache.remove_source(*id);
            }
        }
        self.sources.clear();
        self.stills.clear();
        self.still_generation += 1;
        if let Some(store) = &self.store {
            for c in 0..self.cameras.len() {
                store.remove(Self::image_key(c));
            }
        }
        self.cameras.clear();
        self.flipped.clear();
        self.pending.clear();
        self.requested.clear();
        self.last_frame.clear();
        self.shown_filter.clear();
        self.filtered.clear();
        self.clear_derived();
        self.sequence_colour.clear();
        self.colour_frames.clear();
        self.colour_decoding.clear();
        self.shown_colour.clear();
        self.handed.clear();
        self.video_files.clear();
        // The 3D world view keeps its viewpoint, as the C++ widget does.
        let orbit = self.views.meshes.orbit;
        self.views = ViewStates::default();
        self.views.meshes.orbit = orbit;
        self.logged.clear();
    }

    /// Drops every derived image (undistorted grid and calibration images, maps) and the
    /// grids' lookup tables; results still being computed are dropped when they arrive.
    fn clear_derived(&mut self) {
        if let Some(store) = &self.store {
            for slot in self.derived.values() {
                if let DerivedSlot::Ready(k) = slot {
                    store.remove(*k);
                }
            }
        }
        self.derived.clear();
        self.luts.clear();
        self.derived_headless.clear();
        self.derived_generation += 1;
    }

    /// The project's undistortion or calibration changed (a calibration edit, its undo or
    /// redo): the derived images and the views' cached grid and calibration points are
    /// made again from the document when next shown.
    pub fn calibration_changed(&mut self) {
        self.clear_derived();
        self.views.grids.clear();
        self.views.calibration_points.clear();
    }

    /// True when every camera that wants a video frame shows it, its texture written (or
    /// the frame failed): playback moves on only then, as the C++ timer moves on only after
    /// drawing the frame.
    #[must_use]
    pub fn ready_for_playback(&self) -> bool {
        // Without a texture store (headless) nothing is shown, so nothing is waited for.
        if self.store.is_none() {
            return true;
        }
        self.cameras.iter().zip(&self.pending).all(|(c, p)| {
            !matches!(c.want, Some(ImageId::Video { .. }))
                || c.failed
                || c.message.is_some()
                // Handed over, and (with a texture store) in the texture.
                || (c.shown == c.want && p.is_none())
        })
    }

    /// True when every camera shows what it wants (or cannot) and no texture is waiting
    /// to be written.
    #[must_use]
    pub fn settled(&self) -> bool {
        self.pending.iter().all(Option::is_none)
            && self
                .cameras
                .iter()
                .all(|c| c.want.is_none() || c.failed || c.message.is_some() || c.shown == c.want)
    }

    /// The video frame (0-based) whose pixels camera `camera`'s view shows now: the frame
    /// in its texture, or, without a texture store (headless tests), the frame handed over
    /// for display. `None` when it shows no video frame.
    #[must_use]
    pub fn displayed_frame(&self, camera: usize) -> Option<usize> {
        match &self.store {
            Some(store) => store
                .info(Self::image_key(camera))
                .and_then(|i| i.tag.checked_sub(1))
                .map(|t| t as usize),
            None => match self.cameras.get(camera)?.shown.as_ref()? {
                ImageId::Video { frame, .. } => Some(*frame),
                ImageId::Still { .. } => None,
            },
        }
    }

    /// The image `camera` should show in the current state; `None` for a camera hidden
    /// with the camera selector (`AppState::camera_visible`), whose texture is then
    /// dropped ([`Frames::sync`]).
    ///
    /// Ported from `ui/GLCameraView.cpp`, `GLCameraView::drawTexture`.
    #[must_use]
    pub fn wanted(
        doc: &Document,
        state: &AppState,
        session: &Session,
        camera: usize,
    ) -> Option<ImageId> {
        Self::wanted_with(doc, state, camera, |t, c| session.video_files(t, c).into())
    }

    /// [`Frames::wanted`], with the files of a trial camera's video from `video_files`
    /// (trial, camera), which gives what [`Session::video_files`] gives.
    fn wanted_with(
        doc: &Document,
        state: &AppState,
        camera: usize,
        video_files: impl FnOnce(usize, usize) -> Arc<[PathBuf]>,
    ) -> Option<ImageId> {
        // A camera hidden with the camera selector shows nothing in any workspace: its
        // image is not read, filtered or uploaded for display, and playback does not wait
        // for it (the author's decision 14 of 21 November 2026, ledger U166). Shown again,
        // it wants the current image at once. The tracker, the detector and the other
        // computations read frames through `Frames::read_now`, not through this.
        if !state.camera_visible.get(camera).copied().unwrap_or(true) {
            return None;
        }
        let p = doc.project();
        let cam = p.cameras.get(camera)?;
        match state.workspace {
            Workspace::MarkerTracking => {
                let t = state.active_trial?;
                let trial = p.trials.get(t)?;
                if trial.is_default {
                    return None;
                }
                let frame = state.trials.get(t)?.active_frame;
                let files = video_files(t, camera);
                if files.is_empty() {
                    return None;
                }
                Some(ImageId::Video { files, frame })
            }
            Workspace::Calibration => {
                if p.calibration_type != CalibrationType::Internal {
                    return None;
                }
                let i = state.active_frame_calibration.unwrap_or(0);
                if let Some(seq) = &cam.calibration_sequence {
                    let files: Arc<[PathBuf]> = vec![resolve(doc, &seq.filename)].into();
                    return Some(ImageId::Video { files, frame: i });
                }
                let image = cam.calibration_images.get(i)?;
                Some(ImageId::Still {
                    camera,
                    name: image.filename.clone(),
                })
            }
            Workspace::Undistortion => cam.undistortion.as_ref().map(|u| ImageId::Still {
                camera,
                name: u.image_filename.clone(),
            }),
        }
    }

    /// Brings every camera's image up to date with the state. Never blocks.
    pub fn sync(
        &mut self,
        doc: Option<&Document>,
        state: &AppState,
        session: &Session,
        settings: &UiSettings,
        log: &AppLog,
    ) {
        self.collect_done(doc, log);
        let Some(doc) = doc else {
            return;
        };
        // Image::loadTexture: the visual filter in marker tracking, unless "hide all".
        self.filter = (settings.visual_filter_enabled
            && state.workspace == Workspace::MarkerTracking
            && !settings.trial_draw_hide_all)
            .then_some(VisualFilter {
                krad: settings.visual_filter_krad,
                gsigma: settings.visual_filter_gsigma,
                img_wt: settings.visual_filter_img_wt,
                blur_wt: settings.visual_filter_blur_wt,
                gamma: settings.visual_filter_gamma,
            });
        if self.cameras.len() != doc.project().cameras.len() {
            self.project_opened(doc);
        }
        self.in_step = state.disable_draw;
        for camera in 0..self.cameras.len() {
            let want = Self::wanted_with(doc, state, camera, |t, c| {
                self.video_files_of(session, t, c)
            });
            if want != self.cameras[camera].want {
                self.want_changed(camera, want.as_ref());
                self.cameras[camera].want = want;
                self.cameras[camera].failed = false;
                self.cameras[camera].message = None;
            }
        }
        self.filter_now(doc);
        for camera in 0..self.cameras.len() {
            let flip = doc.project().cameras[camera].flipped;
            self.serve(doc, camera, flip, log);
            self.check_uploaded(camera);
        }
    }

    /// [`Session::video_files`] of `trial` and `camera`, made again only when what it is
    /// made from has changed: the trial camera's file names in the session's document and
    /// the session's location for it (`Session::videos`), which are all that
    /// `Session::video_files` reads. The list kept is shared, so that the views' frames and
    /// sources compare it by pointer, not path by path, on every repaint (review of
    /// 20 November 2026, app-views finding 3).
    fn video_files_of(&mut self, session: &Session, trial: usize, camera: usize) -> Arc<[PathBuf]> {
        let names = session
            .doc
            .as_ref()
            .and_then(|d| d.project().trials.get(trial))
            .and_then(|t| t.videos.get(camera))
            .map(|v| &v.filenames);
        let location = session.videos.get(&(trial, camera));
        if let (Some(kept), Some(names)) = (self.video_files.get(&(trial, camera)), names)
            && kept.names == *names
            && kept.location.as_ref() == location
        {
            return kept.files.clone();
        }
        let files: Arc<[PathBuf]> = session.video_files(trial, camera).into();
        match names {
            Some(names) => {
                self.video_files.insert(
                    (trial, camera),
                    VideoFiles {
                        names: names.clone(),
                        location: location.cloned(),
                        files: files.clone(),
                    },
                );
            }
            None => {
                self.video_files.remove(&(trial, camera));
            }
        }
        files
    }

    fn want_changed(&mut self, camera: usize, want: Option<&ImageId>) {
        let frame = match want {
            Some(ImageId::Video { frame, .. }) => Some(*frame),
            _ => None,
        };
        let step = match (self.last_frame[camera], frame) {
            (Some(a), Some(b)) if b == a + 1 => Step::Forward,
            (Some(a), Some(b)) if b + 1 == a => Step::Back,
            _ => Step::Jump,
        };
        self.last_frame[camera] = frame;
        // Back to the image already in the texture: nothing to wait for.
        if want.is_some() && want == self.cameras[camera].shown.as_ref() {
            self.pending[camera] = None;
            return;
        }
        self.pending[camera] = frame.map(|frame| Pending {
            since: Instant::now(),
            step,
            frame,
            handed: None,
        });
    }

    /// Receives opened sources and decoded stills.
    fn collect_done(&mut self, doc: Option<&Document>, log: &AppLog) {
        while let Ok(done) = self.rx.try_recv() {
            match done {
                Done::Opened(files, Ok(source)) => {
                    let id = self.cache.add_source(source);
                    // Closed meanwhile, or opened already by `read_now` (the tracker).
                    if doc.is_none()
                        || matches!(
                            self.sources.get(&Files::of(&files)),
                            Some(SourceSlot::Open(_))
                        )
                    {
                        self.cache.remove_source(id);
                        continue;
                    }
                    self.sources.insert(Files(files), SourceSlot::Open(id));
                }
                Done::Opened(files, Err(_))
                    if matches!(
                        self.sources.get(&Files::of(&files)),
                        Some(SourceSlot::Open(_))
                    ) => {}
                Done::Opened(files, Err(e)) => {
                    let first = files
                        .first()
                        .map_or(String::new(), |f| f.display().to_string());
                    self.log_once(log, format!("Cannot open {first}: {e}"));
                    // The view shows the file name and whether it exists; the Console has
                    // the full path and reason.
                    let name = files
                        .first()
                        .and_then(|f| f.file_name())
                        .map_or(String::new(), |n| n.to_string_lossy().into_owned());
                    let view = if files.first().is_some_and(|f| !f.exists()) {
                        format!("{name} not found")
                    } else if e.contains("ffmpeg") {
                        e
                    } else {
                        format!("Cannot open {name}")
                    };
                    self.sources.insert(Files(files), SourceSlot::Failed(view));
                }
                Done::Still(_, _, _, generation) if generation != self.still_generation => {}
                Done::Still(camera, name, Ok((frame, colour)), _) => {
                    self.stills.insert(
                        (camera, name),
                        StillSlot::Ready(Arc::new(frame), colour.map(Arc::new)),
                    );
                }
                Done::Still(camera, name, Err(e), _) => {
                    let msg = format!("Cannot read image {name}: {e}");
                    self.log_once(log, msg.clone());
                    self.stills.insert((camera, name), StillSlot::Failed(msg));
                }
                Done::Colour(files, id, r) => {
                    self.colour_decoding.remove(&id);
                    // A file that cannot be read is shown grey (the grey decode reports
                    // the error). In a colour sequence, a file with no colour image (grey,
                    // or unreadable) is remembered as such, so that it is shown grey and
                    // not decoded again.
                    let colour = match r {
                        Ok(Some(c)) => {
                            self.sequence_colour.insert(Files(files), Some(true));
                            Some(Arc::new(c))
                        }
                        _ if self.sequence_colour.get(&Files::of(&files)) != Some(&Some(true)) => {
                            self.sequence_colour.insert(Files(files), Some(false));
                            continue;
                        }
                        _ => None,
                    };
                    self.colour_frames.push_back((id, colour));
                    while self.colour_frames.len() > COLOUR_KEPT {
                        self.colour_frames.pop_front();
                    }
                }
                Done::Derived(camera, kind, lut, image, generation) => {
                    if generation != self.derived_generation {
                        continue;
                    }
                    if let Some(lut) = lut {
                        self.luts.insert(camera, lut);
                    }
                    let slot = match (image, &self.store) {
                        (Some(img), Some(store)) => {
                            let key = Self::derived_image_key(camera, kind);
                            store.upload(key, img);
                            DerivedSlot::Ready(key)
                        }
                        (Some(img), None) => {
                            // Headless: nothing to upload to; kept for inspection.
                            self.derived_headless.insert((camera, kind), Arc::new(img));
                            DerivedSlot::Ready(Self::derived_image_key(camera, kind))
                        }
                        (None, _) => DerivedSlot::Failed,
                    };
                    self.derived.insert((camera, kind), slot);
                }
            }
        }
    }

    fn log_once(&mut self, log: &AppLog, msg: String) {
        if self.logged.insert(msg.clone()) {
            log.error(msg);
        }
    }

    fn wake(&self) {
        if let Some(ctx) = self
            .repaint
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            ctx.request_repaint_of(egui::ViewportId::ROOT);
        }
    }

    /// Asks for camera `camera`'s wanted image and uploads it when it is there.
    fn serve(&mut self, doc: &Document, camera: usize, flip: bool, log: &AppLog) {
        let Some(want) = self.cameras[camera].want.clone() else {
            if self.cameras[camera].shown.take().is_some()
                && let Some(store) = &self.store
            {
                store.remove(Self::image_key(camera));
            }
            return;
        };
        let filter = match want {
            ImageId::Video { .. } => self.filter,
            ImageId::Still { .. } => None,
        };
        let fkey = filter.as_ref().map(filter_key);
        // Image::loadTexture: a colour file is shown in colour unless the visual filter
        // applies. Only image sequences carry colour among the video sources.
        let seq_colour = match (&want, filter) {
            (ImageId::Video { files, .. }, None) => self.sequence_is_colour(files),
            _ => Some(false),
        };
        // What would be shown now: for a still, its colour image when it has one (stills are
        // never filtered); for a video frame, the sequence's colour.
        let want_colour = match &want {
            ImageId::Still { camera, name } => matches!(
                self.stills.get(&(*camera, name.clone())),
                Some(StillSlot::Ready(_, Some(_)))
            ),
            // A file of a colour sequence known to have no colour image is shown grey.
            ImageId::Video { .. } => {
                seq_colour == Some(true)
                    && !self
                        .colour_frames
                        .iter()
                        .any(|(k, c)| *k == want && c.is_none())
            }
        };
        if self.cameras[camera].shown.as_ref() == Some(&want)
            && self.flipped[camera] == flip
            && self.shown_filter[camera] == fkey
            && self.shown_colour[camera] == want_colour
        {
            return;
        }
        if self.cameras[camera].failed {
            return;
        }
        let (frame, colour): (Option<Arc<Frame>>, Option<Arc<ColourImage>>) = match &want {
            ImageId::Video { files, frame } => {
                let f = self.video_frame(camera, files, *frame, log);
                let c = match (&f, seq_colour) {
                    (Some(_), None) => {
                        // Not known yet: look at this file in colour; grey meanwhile.
                        self.start_colour(files, &want, *frame);
                        None
                    }
                    (Some(_), Some(true)) => {
                        match self.colour_frames.iter().find(|(k, _)| *k == want) {
                            // `None`: a file with no colour image, shown grey.
                            Some((_, c)) => c.clone(),
                            None => {
                                self.start_colour(files, &want, *frame);
                                return;
                            }
                        }
                    }
                    _ => None,
                };
                (f, c)
            }
            ImageId::Still { name, .. } => match self.still(doc, camera, name) {
                Some((f, c)) => (Some(f), c),
                None => (None, None),
            },
        };
        let Some(frame) = frame else {
            // A source or still that cannot be read: no image, only its message.
            if self.cameras[camera].message.is_some() && self.cameras[camera].shown.take().is_some()
            {
                if let Some(store) = &self.store {
                    store.remove(Self::image_key(camera));
                }
                self.pending[camera] = None;
            }
            return;
        };
        let colour_shown = filter.is_none() && colour.is_some();
        let filtered = if let (Some(f), Some(k)) = (filter, fkey) {
            let key = (want.clone(), flip, k);
            Some(match self.filtered.iter().find(|(kk, _)| *kk == key) {
                Some((_, px)) => px.clone(),
                None => {
                    // Decoded since `filter_now` looked: filtered here.
                    let px = Arc::new(filter_frame(&frame, flip, f));
                    self.keep_filtered(key, px.clone());
                    px
                }
            })
        } else {
            None
        };
        let image = display_image(&frame, colour.as_ref(), filtered, flip);
        // An image the texture store refuses (larger than the graphics device's largest
        // texture): no image, only its message, as for a source that cannot be read.
        if let Some(store) = &self.store
            && let Err(e) = store.check(&image)
        {
            self.log_once(log, format!("Camera {}: {e}", camera + 1));
            self.cameras[camera].message = Some(e);
            self.cameras[camera].failed = true;
            if self.cameras[camera].shown.take().is_some()
                && let Some(store) = &self.store
            {
                store.remove(Self::image_key(camera));
            }
            self.pending[camera] = None;
            return;
        }
        if let Some(store) = &self.store {
            let key = Self::image_key(camera);
            let generation = store.info(key).map_or(0, |i| i.generation);
            if self.in_step {
                // During a run the view is drawn with the points of this frame: its image
                // is in the texture before the frame is drawn, as XMALab draws each frame
                // the run reaches.
                if let Err(e) = store.upload_now_tagged(key, image, image_tag(&want)) {
                    self.log_once(log, format!("Camera {}: {e}", camera + 1));
                }
            } else {
                store.upload_tagged(key, image, image_tag(&want));
            }
            if let Some(p) = &mut self.pending[camera]
                && p.handed.is_none()
            {
                p.handed = Some((p.since.elapsed().as_secs_f64(), generation));
            }
        } else if let Some(p) = self.pending[camera].take() {
            let t = p.since.elapsed().as_secs_f64();
            self.push_sample(camera, p.step, p.frame, t, t);
        }
        self.handed[camera] += 1;
        self.cameras[camera].shown = Some(want);
        self.flipped[camera] = flip;
        self.shown_filter[camera] = fkey;
        self.shown_colour[camera] = colour_shown;
    }

    /// Whether the image-sequence source `files` holds colour images: `Some(false)` for a
    /// video (only sequences carry colour here) or a grey sequence, `None` until known.
    fn sequence_is_colour(&self, files: &Arc<[PathBuf]>) -> Option<bool> {
        let id = match self.sources.get(&Files::of(files)) {
            Some(SourceSlot::Open(id)) => *id,
            _ => return Some(false),
        };
        if self.cache.info(id).map(|i| i.kind) != Some(xma_video::SourceKind::ImageSequence) {
            return Some(false);
        }
        self.sequence_colour
            .get(&Files::of(files))
            .copied()
            .unwrap_or(None)
    }

    /// Decodes one file of an image sequence in colour on a worker thread.
    fn start_colour(&mut self, files: &Arc<[PathBuf]>, id: &ImageId, frame: usize) {
        if !self.colour_decoding.insert(id.clone()) {
            return;
        }
        self.sequence_colour.entry(Files::of(files)).or_insert(None);
        let Some(path) = files.get(frame).or_else(|| files.first()).cloned() else {
            return;
        };
        let (tx, repaint, files, id) = (
            self.tx.clone(),
            self.repaint.clone(),
            files.clone(),
            id.clone(),
        );
        let _ = std::thread::Builder::new()
            .name("xma-decode-colour".into())
            .spawn(move || {
                let r = xma_video::image_file::read_colour_image(&path).map_err(|e| e.to_string());
                let _ = tx.send(Done::Colour(files, id, r));
                if let Some(ctx) = repaint.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
                    ctx.request_repaint_of(egui::ViewportId::ROOT);
                }
            });
    }

    /// A decoded frame of `files` if the cache holds it; asks for it otherwise.
    fn cached_frame(&mut self, files: &Arc<[PathBuf]>, frame: usize) -> Option<Arc<Frame>> {
        match self.sources.get(&Files::of(files)) {
            Some(SourceSlot::Open(id)) => match self.cache.peek(*id, frame) {
                Some(Ok(f)) => Some(f),
                _ => {
                    let _ = self.cache.request(*id, frame);
                    None
                }
            },
            _ => None,
        }
    }

    /// Whether `camera` shows the colour image of its frame (a colour image sequence with
    /// the visual filter off, or a colour still).
    #[must_use]
    pub fn shows_colour(&self, camera: usize) -> bool {
        self.shown_colour.get(camera).copied().unwrap_or(false)
    }

    /// Number of images handed for display for `camera` since the project was opened: to
    /// the uploader, or without a texture store (headless tests), as shown.
    #[must_use]
    pub fn handed_count(&self, camera: usize) -> u64 {
        self.handed.get(camera).copied().unwrap_or(0)
    }

    /// The decoded still `camera` shows now (grey, and colour for a colour file), if it
    /// shows a still.
    #[must_use]
    pub fn shown_still(&self, camera: usize) -> Option<(Arc<Frame>, Option<Arc<ColourImage>>)> {
        let Some(ImageId::Still { camera: c, name }) = self.cameras.get(camera)?.shown.as_ref()
        else {
            return None;
        };
        match self.stills.get(&(*c, name.clone()))? {
            StillSlot::Ready(f, colour) => Some((f.clone(), colour.clone())),
            _ => None,
        }
    }

    /// A derived image made without a texture store (headless), once ready.
    #[must_use]
    pub fn derived_headless(&self, camera: usize, kind: Derived) -> Option<Arc<DisplayImage>> {
        self.derived_headless.get(&(camera, kind)).cloned()
    }

    /// The visual filter of every camera's wanted frame, once the frame is decoded and not
    /// filtered yet, before the views are drawn: one thread per camera, waited for, as the
    /// C++ filters in `Image::loadTexture` before drawing. (During a tracking run the run has
    /// read the frame, so it is decoded.) Filtering on worker threads without waiting left
    /// the views on an old frame for as long as the frame changed faster than the filter
    /// (a tracking run, a dragged slider): each result arrived for a frame no longer wanted
    /// and was not shown.
    fn filter_now(&mut self, doc: &Document) {
        let Some(filter) = self.filter else {
            return;
        };
        let fkey = filter_key(&filter);
        let mut jobs = Vec::new();
        for camera in 0..self.cameras.len() {
            let Some(want) = self.cameras[camera].want.clone() else {
                continue;
            };
            let ImageId::Video { files, frame } = &want else {
                continue;
            };
            let (files, frame) = (files.clone(), *frame);
            let flip = doc.project().cameras[camera].flipped;
            let key = (want.clone(), flip, fkey);
            if self.filtered.iter().any(|(k, _)| *k == key) {
                continue;
            }
            let Some(SourceSlot::Open(id)) = self.sources.get(&Files::of(&files)) else {
                continue;
            };
            if let Some(Ok(f)) = self.cache.peek(*id, frame) {
                jobs.push((key, f, flip));
            }
        }
        let done: Vec<(FilteredKey, Frame)> = std::thread::scope(|s| {
            let handles: Vec<_> = jobs
                .into_iter()
                .map(|(key, f, flip)| s.spawn(move || (key, filter_frame(&f, flip, filter))))
                .collect();
            handles.into_iter().filter_map(|h| h.join().ok()).collect()
        });
        for (key, px) in done {
            self.keep_filtered(key, Arc::new(px));
        }
    }

    fn keep_filtered(&mut self, key: FilteredKey, px: Arc<Frame>) {
        self.filtered.push_back((key, px));
        while self.filtered.len() > FILTERED_KEPT {
            self.filtered.pop_front();
        }
    }

    /// The store key of a derived image.
    #[must_use]
    pub fn derived_image_key(camera: usize, kind: Derived) -> egui::Id {
        egui::Id::unique(("xma derived image", camera, kind))
    }

    /// The texture of a derived image of `camera`, once computed; starts computing it
    /// (off the interface thread, once per camera and kind) otherwise. `None` also when it
    /// cannot be made (no computed grid and no lens model): show the stored image then, as
    /// the C++ shows its unchanged copy.
    pub fn derived(&mut self, doc: &Document, camera: usize, kind: Derived) -> Option<egui::Id> {
        match self.derived.get(&(camera, kind)) {
            Some(DerivedSlot::Ready(k)) => return Some(*k),
            Some(DerivedSlot::Computing | DerivedSlot::Failed) => return None,
            None => {}
        }
        let cam = doc.project().cameras.get(camera)?;
        let grid_computed = doc.camera(camera).model.grid.is_some();
        let lens = cam.has_model_distortion.then(|| {
            let k = &cam.camera_matrix;
            (
                [
                    [k[(0, 0)], k[(0, 1)], k[(0, 2)]],
                    [k[(1, 0)], k[(1, 1)], k[(1, 2)]],
                    [k[(2, 0)], k[(2, 1)], k[(2, 2)]],
                ],
                cam.distortion,
            )
        });
        let source = match kind {
            Derived::UndistortedGrid | Derived::Map(_) => {
                if !grid_computed {
                    return None;
                }
                let name = cam.undistortion.as_ref()?.image_filename.clone();
                DeriveSource::Still(xma_engine::still_bytes(doc, camera, &name))
            }
            Derived::UndistortedCalibration(i) => {
                if !grid_computed && lens.is_none() {
                    return None;
                }
                if let Some(seq) = &cam.calibration_sequence {
                    // CalibrationSequence::undstortSequenceImage: the frame of the
                    // sequence, from the frame cache once it has been read.
                    let files: Arc<[PathBuf]> = vec![resolve(doc, &seq.filename)].into();
                    let frame = self.cached_frame(&files, i)?;
                    DeriveSource::Frame(frame)
                } else {
                    let name = cam.calibration_images.get(i)?.filename.clone();
                    DeriveSource::Still(xma_engine::still_bytes(doc, camera, &name))
                }
            }
        };
        let (w, h) = (cam.width as usize, cam.height as usize);
        let flip = cam.flipped;
        // The engine's table when a computation left one (`DerivedCamera::lookup`), else
        // made from the grid on the worker thread.
        let lut = self
            .luts
            .get(&camera)
            .cloned()
            .or_else(|| doc.camera(camera).lookup.clone());
        let grid = doc.camera(camera).model.grid.clone();
        let generation = self.derived_generation;
        self.derived.insert((camera, kind), DerivedSlot::Computing);
        let tx = self.tx.clone();
        let repaint = self.repaint.clone();
        let _ = std::thread::Builder::new()
            .name("xma-derived-image".into())
            .spawn(move || {
                let lut = match (lut, &grid) {
                    (Some(l), _) => Some(l),
                    (None, Some(g)) => Some(Arc::new(create_lookup_table(&g.distort, w, h))),
                    (None, None) => None,
                };
                let image = derive(kind, lut.as_deref(), lens, source, flip, w, h);
                let _ = tx.send(Done::Derived(camera, kind, lut, image, generation));
                if let Some(ctx) = repaint.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
                    ctx.request_repaint_of(egui::ViewportId::ROOT);
                }
            });
        None
    }

    /// Records when an uploaded image reached its texture.
    fn check_uploaded(&mut self, camera: usize) {
        let Some(store) = &self.store else { return };
        let Some(p) = &self.pending[camera] else {
            return;
        };
        let Some((handed, generation)) = p.handed else {
            return;
        };
        let now = store
            .info(Self::image_key(camera))
            .map_or(0, |i| i.generation);
        if now != generation {
            let (step, frame) = (p.step, p.frame);
            let t = p.since.elapsed().as_secs_f64();
            self.pending[camera] = None;
            self.push_sample(camera, step, frame, handed, t);
        }
    }

    fn push_sample(&mut self, camera: usize, step: Step, frame: usize, handed: f64, textured: f64) {
        let t = &mut self.timings;
        t.samples.push_back((step, frame, handed, textured));
        while t.samples.len() > 4096 {
            t.samples.pop_front();
        }
        t.shown.push((camera, Instant::now()));
        if t.shown.len() > 100_000 {
            t.shown.drain(..50_000);
        }
    }

    /// The decoded video frame, if it has arrived; opens the source and queues the frame
    /// otherwise.
    fn video_frame(
        &mut self,
        camera: usize,
        files: &Arc<[PathBuf]>,
        frame: usize,
        log: &AppLog,
    ) -> Option<Arc<Frame>> {
        let id = match self.sources.get(&Files::of(files)) {
            None => {
                self.sources.insert(Files::of(files), SourceSlot::Opening);
                let tx = self.tx.clone();
                let f = files.clone();
                let opts = self.open_options.clone();
                let repaint = self.repaint.clone();
                let spawned = std::thread::Builder::new()
                    .name("xma-open-video".into())
                    .spawn(move || {
                        let r = open_source(&f, &opts).map_err(|e| describe_open_error(&e));
                        let _ = tx.send(Done::Opened(f, r));
                        if let Some(ctx) =
                            repaint.lock().unwrap_or_else(|e| e.into_inner()).as_ref()
                        {
                            ctx.request_repaint_of(egui::ViewportId::ROOT);
                        }
                    });
                if spawned.is_err() {
                    self.sources.insert(
                        Files::of(files),
                        SourceSlot::Failed("Cannot start a thread to open the video".into()),
                    );
                }
                return None;
            }
            Some(SourceSlot::Opening) => return None,
            Some(SourceSlot::Failed(msg)) => {
                self.cameras[camera].message = Some(msg.clone());
                return None;
            }
            Some(SourceSlot::Open(id)) => *id,
        };
        let want = ImageId::Video {
            files: files.clone(),
            frame,
        };
        let fresh = match &self.requested[camera] {
            Some((w, at)) => *w != want || at.elapsed() > Duration::from_secs(2),
            None => true,
        };
        let got = if fresh {
            self.requested[camera] = Some((want, Instant::now()));
            self.cache.request(id, frame)
        } else {
            self.cache.peek(id, frame)
        };
        match got {
            Some(Ok(f)) => Some(f),
            Some(Err(e)) => {
                let name = files
                    .first()
                    .map_or(String::new(), |f| f.display().to_string());
                self.log_once(
                    log,
                    format!(
                        "Camera {}: frame {} of {name} cannot be read: {e}",
                        camera + 1,
                        frame + 1
                    ),
                );
                self.cameras[camera].failed = true;
                if self.cameras[camera].shown.take().is_some()
                    && let Some(store) = &self.store
                {
                    store.remove(Self::image_key(camera));
                }
                self.pending[camera] = None;
                self.wake();
                None
            }
            None => None,
        }
    }

    /// The decoded still, if ready; starts decoding it otherwise.
    fn still(
        &mut self,
        doc: &Document,
        camera: usize,
        name: &str,
    ) -> Option<(Arc<Frame>, Option<Arc<ColourImage>>)> {
        let key = (camera, name.to_owned());
        match self.stills.get(&key) {
            Some(StillSlot::Ready(f, c)) => Some((f.clone(), c.clone())),
            Some(StillSlot::Decoding) => None,
            Some(StillSlot::Failed(msg)) => {
                self.cameras[camera].message = Some(msg.clone());
                None
            }
            None => {
                self.stills.insert(key, StillSlot::Decoding);
                // Found as the engine's undistortion and calibration actions find it.
                let bytes = xma_engine::still_bytes(doc, camera, name);
                let tx = self.tx.clone();
                let (key, generation) = (name.to_owned(), self.still_generation);
                let repaint = self.repaint.clone();
                let _ = std::thread::Builder::new()
                    .name("xma-decode-still".into())
                    .spawn(move || {
                        let r = match bytes {
                            Ok((data, path)) => decode_still(&data, &path),
                            Err(e) => Err(e),
                        };
                        let _ = tx.send(Done::Still(camera, key, r, generation));
                        if let Some(ctx) =
                            repaint.lock().unwrap_or_else(|e| e.into_inner()).as_ref()
                        {
                            ctx.request_repaint_of(egui::ViewportId::ROOT);
                        }
                    });
                None
            }
        }
    }

    /// Frame `frame` (0-based) of the video made of `files`, as the source holds it (full
    /// depth, not mirrored), waiting for the decode: what the tracker and the detector read
    /// (`crate::trial_images`). The source is opened here, on this thread, when the views
    /// have not opened it yet; the frame goes through the shared [`FrameCache`], so the views
    /// and the tracker decode each frame once.
    ///
    /// # Errors
    /// The message of a source that cannot be opened or a frame that cannot be read.
    pub fn read_now(&mut self, files: &[PathBuf], frame: usize) -> Result<Arc<Frame>, String> {
        let key = Files(files.into());
        let id = match self.sources.get(&key) {
            Some(SourceSlot::Open(id)) => *id,
            Some(SourceSlot::Failed(msg)) => return Err(msg.clone()),
            Some(SourceSlot::Opening) | None => {
                let source =
                    open_source(files, &self.open_options).map_err(|e| describe_open_error(&e))?;
                let id = self.cache.add_source(source);
                self.sources.insert(key, SourceSlot::Open(id));
                id
            }
        };
        self.cache.get(id, frame).map_err(|e| e.to_string())
    }
}

/// The visual filter of a frame (`Image::loadTexture` with `FilterImage::run`): the 8-bit
/// image the C++ holds, mirrored first when the camera is flipped. The result has the
/// frame's width and height.
fn filter_frame(frame: &Frame, flip: bool, filter: VisualFilter) -> Frame {
    let gray = frame.to_gray8();
    let (w, h) = (frame.width as usize, frame.height as usize);
    let px = if flip {
        let mut gray = gray.into_owned();
        mirror_rows(&mut gray, w, 1);
        filter.apply(&gray, w, h)
    } else {
        filter.apply(&gray, w, h)
    };
    Frame::gray8(frame.width, frame.height, px)
}

/// Mirrors each row of an image left to right in place: `data` holds rows of `width`
/// pixels of `channels` values each (`cv::flip(image, image, 1)`). A last row shorter than
/// the others (data that does not fill whole rows) is mirrored by its whole pixels, and
/// values after its last whole pixel stay; with `width * channels` 0 nothing changes.
///
/// Ported from `core/Image.cpp`, `Image::Image` and `Image::setImage` (`if (flip)`), and the
/// same step in `core/AviVideo.cpp` and `core/CineVideo.cpp`.
pub(crate) fn mirror_rows<T>(data: &mut [T], width: usize, channels: usize) {
    let row = width * channels;
    if row == 0 {
        return;
    }
    for r in data.chunks_mut(row) {
        let n = r.len() / channels;
        if channels == 1 {
            r[..n].reverse();
            continue;
        }
        for i in 0..n / 2 {
            for k in 0..channels {
                r.swap(i * channels + k, (n - 1 - i) * channels + k);
            }
        }
    }
}

/// The image handed to the texture store for a camera's frame: the filtered frame when the
/// visual filter applies (`filtered`, mirrored already), else the colour image when there
/// is one, else the frame; the last two mirrored when `flip` is set. Nothing is copied
/// here: the store mirrors a shared image when it writes it.
fn display_image(
    frame: &Arc<Frame>,
    colour: Option<&Arc<ColourImage>>,
    filtered: Option<Arc<Frame>>,
    flip: bool,
) -> DisplayImage {
    if let Some(px) = filtered {
        DisplayImage::gray_shared(px, false)
    } else if let Some(c) = colour {
        DisplayImage::RgbaShared {
            image: c.clone(),
            mirror: flip,
        }
    } else {
        DisplayImage::gray_shared(frame.clone(), flip)
    }
}

/// The store tag of an image ([`Frames::displayed_frame`]): a video frame's number plus
/// one, 0 for a still.
fn image_tag(id: &ImageId) -> u64 {
    match id {
        ImageId::Video { frame, .. } => *frame as u64 + 1,
        ImageId::Still { .. } => 0,
    }
}

/// A message for a source that cannot be opened. A missing ffmpeg says how to fix it.
fn describe_open_error(e: &VideoError) -> String {
    match e {
        VideoError::Ffmpeg(reason) => format!(
            "this video needs ffmpeg, which could not be used ({reason}). Install ffmpeg and \
             ffprobe, or set XMALAB_FFMPEG and XMALAB_FFPROBE to their paths."
        ),
        other => other.to_string(),
    }
}

/// A path stored in the project: separators adjusted, relative to the project's folder
/// ([`xma_engine::resolve_stored_path`]).
fn resolve(doc: &Document, stored: &str) -> PathBuf {
    xma_engine::resolve_stored_path(doc.path().and_then(Path::parent), stored)
}

/// A still as the C++ `Image` holds it: grey, and the colour image of a colour file.
///
/// Ported from `core/Image.cpp`, `Image::Image` (`image`, `image_color`).
fn decode_still(data: &[u8], path: &Path) -> Result<(Frame, Option<ColourImage>), String> {
    let grey = xma_video::image_file::decode_image(data, path).map_err(|e| e.to_string())?;
    let colour =
        xma_video::image_file::decode_colour_image(data, path).map_err(|e| e.to_string())?;
    Ok((grey, colour))
}

/// Computes a derived display image (see [`Derived`]).
///
/// Ported from `core/UndistortionObject.cpp` (`undistort`, `computeTmpImage`) and
/// `core/CalibrationSequence.cpp` (`undistort`: the grid remap, then the lens-model remap
/// when the camera has one).
fn derive(
    kind: Derived,
    lut: Option<&LookupTable>,
    lens: Option<([[f64; 3]; 3], [f64; 8])>,
    source: DeriveSource,
    flip: bool,
    w: usize,
    h: usize,
) -> Option<DisplayImage> {
    if let Derived::Map(m) = kind {
        let lut = lut?;
        let rgba = displacement_image(&lut.map_x, &lut.map_y, lut.width, lut.height, m);
        return Some(DisplayImage::Rgba8 {
            width: lut.width as u32,
            height: lut.height as u32,
            data: rgba,
        });
    }
    let (grey, colour) = match source {
        DeriveSource::Still(s) => {
            let (data, path) = s.ok()?;
            let (g, c) = decode_still(&data, &path).ok()?;
            (Arc::new(g), c)
        }
        // The frame cache holds frames as read: mirrored below like a still.
        DeriveSource::Frame(f) => (f, None),
    };
    // The image as the C++ holds it: 8 bits, colour when the file is, flipped.
    let (px, cn, sw, sh) = match colour {
        Some(c) => {
            let rgb: Vec<u8> = c.rgba.chunks(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
            (rgb, 3, c.width as usize, c.height as usize)
        }
        None => (
            grey.to_gray8().into_owned(),
            1,
            grey.width as usize,
            grey.height as usize,
        ),
    };
    // The engine's computation, which the Blender-scene export also uses.
    let lens = match kind {
        Derived::UndistortedCalibration(_) => lens,
        _ => None,
    };
    let out = xma_engine::undistorted_image::undistort_held_image(
        px,
        sw,
        sh,
        cn,
        flip,
        lut,
        lens.as_ref(),
        w,
        h,
    );
    let (px, pw, ph) = (out.data, out.width, out.height);
    Some(if cn == 3 {
        DisplayImage::Rgba8 {
            width: pw as u32,
            height: ph as u32,
            data: px.chunks(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect(),
        }
    } else {
        DisplayImage::from(Frame::gray8(pw as u32, ph as u32, px))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use xma_video::Pixels;

    // ----- Review 13 (performance): the hand-over without copies -------------------------
    //
    // The code as it was before the third review round, kept as the reference: what
    // `Frames::serve` handed to the store, the visual filter's input, and the mirror loops.

    /// The old `filter_frame`.
    fn old_filter_frame(frame: &Frame, flip: bool, filter: VisualFilter) -> Vec<u8> {
        let mut gray = frame.to_gray8().into_owned();
        let w = frame.width as usize;
        if flip {
            gray.chunks_mut(w.max(1)).for_each(<[u8]>::reverse);
        }
        filter.apply(&gray, w, frame.height as usize)
    }

    /// The image the old `serve` handed over (`px`: the old filtered pixels).
    fn old_display_image(
        frame: &Arc<Frame>,
        colour: Option<&Arc<ColourImage>>,
        px: Option<Arc<Vec<u8>>>,
        flip: bool,
    ) -> DisplayImage {
        if let Some(px) = px {
            DisplayImage::from(Frame::gray8(frame.width, frame.height, px.to_vec()))
        } else if let Some(c) = colour.cloned() {
            let mut rgba = c.rgba.clone();
            if flip {
                rgba.chunks_mut(c.width as usize * 4).for_each(|row| {
                    let n = row.len() / 4;
                    for i in 0..n / 2 {
                        for k in 0..4 {
                            row.swap(i * 4 + k, (n - 1 - i) * 4 + k);
                        }
                    }
                });
            }
            DisplayImage::Rgba8 {
                width: c.width,
                height: c.height,
                data: rgba,
            }
        } else {
            let mut owned = (**frame).clone();
            if flip {
                owned.flip_horizontal();
            }
            DisplayImage::from(owned)
        }
    }

    /// The old per-channel mirror loop of `derive` (and of the colour image in `serve`).
    fn old_mirror_channels(px: &mut [u8], sw: usize, cn: usize) {
        px.chunks_mut(sw * cn).for_each(|row| {
            let n = row.len() / cn;
            for i in 0..n / 2 {
                for k in 0..cn {
                    row.swap(i * cn + k, (n - 1 - i) * cn + k);
                }
            }
        });
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    fn grey_frame(rng: &mut Rng, w: u32, h: u32, depth: u8) -> Frame {
        let n = (w * h) as usize;
        if depth == 8 {
            Frame::gray8(w, h, (0..n).map(|_| rng.next() as u8).collect())
        } else {
            // Values up to 16 bits whatever the depth: a `.cine` can hold values above it.
            Frame::gray16(w, h, (0..n).map(|_| rng.next() as u16).collect(), depth)
        }
    }

    fn colour_image(rng: &mut Rng, w: u32, h: u32) -> ColourImage {
        ColourImage {
            width: w,
            height: h,
            rgba: (0..(w * h * 4) as usize)
                .map(|_| rng.next() as u8)
                .collect(),
        }
    }

    fn assert_same_upload(old: &DisplayImage, new: &DisplayImage, what: &str) {
        assert_eq!(
            old.upload_meta(),
            new.upload_meta(),
            "{what}: format and size"
        );
        assert_eq!(old.upload_bytes(), new.upload_bytes(), "{what}: bytes");
    }

    #[test]
    fn review13_handed_over_pixels_are_unchanged() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let filter = VisualFilter::default();
        let sizes = [
            (1, 1),
            (1, 7),
            (7, 1),
            (2, 2),
            (3, 5),
            (64, 48),
            (65, 33),
            (257, 3),
        ];
        for &(w, h) in &sizes {
            for depth in [8u8, 9, 10, 12, 14, 16] {
                let frame = Arc::new(grey_frame(&mut rng, w, h, depth));
                for flip in [false, true] {
                    let what = format!("{w}x{h} depth {depth} flip {flip}");
                    // Grey, not filtered.
                    let old = old_display_image(&frame, None, None, flip);
                    let new = display_image(&frame, None, None, flip);
                    assert_same_upload(&old, &new, &format!("grey {what}"));
                    // Filtered.
                    let old_px = old_filter_frame(&frame, flip, filter);
                    let new_px = filter_frame(&frame, flip, filter);
                    assert_eq!((new_px.width, new_px.height), (w, h));
                    assert_eq!(new_px.bit_depth, 8);
                    assert_eq!(
                        new_px.pixels,
                        Pixels::Gray8(old_px.clone()),
                        "filter {what}"
                    );
                    let old = old_display_image(&frame, None, Some(Arc::new(old_px)), flip);
                    let new = display_image(&frame, None, Some(Arc::new(new_px)), flip);
                    assert_same_upload(&old, &new, &format!("filtered {what}"));
                    // Colour (a colour file: the grey frame is 8-bit).
                    if depth == 8 {
                        let c = Arc::new(colour_image(&mut rng, w, h));
                        let old = old_display_image(&frame, Some(&c), None, flip);
                        let new = display_image(&frame, Some(&c), None, flip);
                        assert_same_upload(&old, &new, &format!("colour {what}"));
                    }
                }
            }
        }
        // A 2048 x 2048 16-bit frame and colour image, as the timing harness hands over.
        let frame = Arc::new(grey_frame(&mut rng, 2048, 2048, 16));
        let c = Arc::new(colour_image(&mut rng, 2048, 2048));
        for flip in [false, true] {
            let old = old_display_image(&frame, None, None, flip);
            let new = display_image(&frame, None, None, flip);
            assert_same_upload(&old, &new, "2048 grey");
            let old = old_display_image(&frame, Some(&c), None, flip);
            let new = display_image(&frame, Some(&c), None, flip);
            assert_same_upload(&old, &new, "2048 colour");
        }
    }

    #[test]
    fn review13_mirror_rows_matches_the_old_loops() {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        for _ in 0..2000 {
            let len = rng.below(200) as usize;
            let data: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            let w = 1 + rng.below(40) as usize;
            for cn in [1usize, 3, 4] {
                // Data that does not fill whole rows too: the old loops mirrored the last,
                // shorter row by its whole pixels.
                let mut old = data.clone();
                old_mirror_channels(&mut old, w, cn);
                let mut new = data.clone();
                mirror_rows(&mut new, w, cn);
                assert_eq!(old, new, "len {len} width {w} channels {cn}");
            }
            // The old filter's mirror: whole chunks of `w.max(1)` reversed.
            for w in [0usize, w] {
                let mut old = data.clone();
                old.chunks_mut(w.max(1)).for_each(<[u8]>::reverse);
                let mut new = data.clone();
                mirror_rows(&mut new, w, 1);
                assert_eq!(old, new, "filter mirror, len {len} width {w}");
            }
        }
        // Empty data, and a frame mirrored like `Frame::flip_horizontal`.
        let mut e: Vec<u8> = Vec::new();
        mirror_rows(&mut e, 3, 4);
        assert!(e.is_empty());
        let mut f = grey_frame(&mut rng, 13, 5, 16);
        let Pixels::Gray16(mut d) = f.pixels.clone() else {
            unreachable!()
        };
        f.flip_horizontal();
        mirror_rows(&mut d, 13, 1);
        assert_eq!(f.pixels, Pixels::Gray16(d));
    }

    #[test]
    fn colour_still_is_kept_in_colour() {
        let mut png = Vec::new();
        let img = image::RgbImage::from_fn(4, 2, |x, _| image::Rgb([200, 10 * x as u8, 30]));
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let (grey, colour) = decode_still(&png, Path::new("c.png")).unwrap();
        assert_eq!((grey.width, grey.height), (4, 2));
        let c = colour.expect("colour image");
        assert_eq!(&c.rgba[4..8], &[200, 10, 30, 255]);
        let tiny = DisplayImage::from(grey);
        assert_eq!(tiny.width(), 4);
    }
}
