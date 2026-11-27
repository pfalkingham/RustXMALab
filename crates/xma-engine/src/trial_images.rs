//! File > Export > Undistorted Trial images, and File > Export > All Trials for External:
//! a trial's frames written as image files (or a video), undistorted where the camera has
//! an undistortion grid or a lens model, optionally through the visual filter; and, for
//! every trial and camera, the MayaCams 2.0, the images, the 2D points and the
//! reprojection errors in the folder layout XMALab makes.
//!
//! Ported from `core/Trial.cpp`, `Trial::saveTrialImages` (`:1154-1314`),
//! `ui/MainWindow.cpp`, `on_actionExport_Undistorted_Trial_images_for_Maya_triggered`
//! (`:2000-2020`) and `on_actionAll_Trials_for_External_triggered` (`:2076-2108`),
//! `core/UndistortionObject.cpp`, `UndistortionObject::undistort(Image*, QString, bool)`,
//! and `core/Image.cpp`, `Image::save`.
//!
//! # What XMALab writes ([`TrialImagesJob`])
//!
//! For each camera (all of them, or the one `id` names), from frame `from` to frame `to`
//! (1-based, the "Select Range" dialog; the trial's start and end frame by default), in the
//! format chosen (`jpg`, `tif` or `avi`), unfiltered or filtered (`FilterImage::run`, the
//! visual filter with the settings `VisualFilter_*`), into `<folder>` = `<output>/<base>UND`
//! for every camera, or `<output>` itself for one camera, `<base>` being the
//! `completeBaseName` of the longest start all the camera's file names share
//! (`VideoStream::getFileBasename`, [`crate::undistort_sequence::file_basename`]):
//!
//! - a camera with an undistortion grid (`hasUndistortion`): each frame as XMALab holds it
//!   (8 bits; mirrored for a flipped camera, as loaded), remapped with the grid's lookup
//!   table and then, with a lens model, with the lens maps (`INTER_LANCZOS4`, border 0;
//!   [`xma_vision::remap`]), filtered when asked, as `<folder>/<base>_UND.<n>.<format>`
//!   with `n` the frame number, 1-based, four digits at least. Only when the grid is
//!   computed and the frame has the grid's size (`UndistortionObject::undistort` returns
//!   false otherwise, and nothing is written);
//! - a camera with a lens model only: the lens remap, the same names;
//! - any other camera: the frame as its file holds it (a flipped camera's frame mirrored
//!   back, `Image::save(name, isFlipped, filter)`), filtered when asked, as
//!   `<folder>/<base>.<n>.<format>`;
//! - `avi`: one video `<folder>/<base>.avi` at the trial's recording speed (30 when not
//!   set), of the camera's size, the same frames in colour (grey as R = G = B).
//!
//! Files are written with `cv::imwrite`: JPEG of quality 95, LZW TIFF; one channel for a
//! grey frame, three for a colour image and for every frame of an `AviVideo` (OpenCV's
//! FFmpeg gives B, G, R frames). The Default trial has no images: nothing is written.
//!
//! # Differences (parity ledger V26 to V29, F65 to F67, U161, U162)
//!
//! In the default mode (each as XMALab 3 in its mode, `CppCompat::XMALAB3`):
//!
//! - V26 (`trial_images_uncomputed_grid_writes_nothing`): a camera with a grid image whose
//!   grid is not computed is treated as a camera without a grid (its lens model, else the
//!   plain frames); XMALab writes nothing for it. Frames of another size than the grid are
//!   skipped and reported in both modes (XMALab skips them silently).
//! - V27 (`trial_images_avi_not_written`): "avi" writes a Motion JPEG AVI
//!   ([`xma_video::avi_writer`]); XMALab 3 on Linux writes nothing (its FFmpeg back end
//!   refuses the codec -1), reported here as an error per camera.
//! - V28 (`trial_images_eight_bit`): a source deeper than 8 bits is written as a TIFF at its
//!   depth (16-bit samples, the remap saturated at the depth's largest value, ledger V20);
//!   JPEG, AVI and filtered images are 8-bit.
//!
//! - F70 (`export_names_as_given`): every folder and file name built from a name the
//!   project holds is one safe part ([`xma_io::export::safe_component`]): `.`, `..` and an
//!   empty name are refused in both modes, other characters no system can store replaced.
//! - F71 (`trial_images_same_folder_for_same_name`): cameras whose videos share a base name
//!   get `<base>UND_cam<n>` instead of one shared folder.
//!
//! In both modes: V29, the orientation of a flipped camera's files is XMALab's (undistorted
//! files mirrored as XMALab holds the frames, the others as the files hold them), a choice
//! left to the author; the readers are this program's (ledger V2, V7); a colour video is
//! written grey (as File > Undistort sequence, V24); files are written whole (a temporary
//! name, then renamed); and the work runs as a job with progress and Cancel (U161).
//!
//! # All Trials for External ([`AllTrialsJob`])
//!
//! For every trial and camera `i` (0-based): `<output>/<trial name>/Camera <i>/` holding
//! `calib/` (the camera's MayaCams 2.0 of every calibrated calibration image,
//! `exportMayaCamVersion2(folder, -1, i)`), `images/` (`saveTrialImages(folder, 1,
//! frames of camera i, "tif", i)`), `markers/` (the 2D points of camera `i`, one file per
//! marker, undistorted, no header, `save2dPoints(folder, false, false, false, false, false,
//! false, i)`) and `reprojection_errors/` (`saveReprojectionErrors`, every camera).
//!
//! F65 (`export_all_trials_images_misplaced`): the images call passes the camera index `i`
//! where `saveTrialImages` takes `filter` and leaves its `id` at -1, so XMALab writes the
//! images of **every** camera into each camera's `images/` folder, each in a
//! `<base>UND` subfolder, unfiltered in `Camera 0` and through the visual filter in every
//! other camera's folder. What was meant (the argument list before `filter` was added):
//! camera `i`'s images, unfiltered, in its `images/` folder. That is the default; the
//! XMALab 3 mode writes what XMALab writes.
//!
//! F67 (`export_all_trials_default_trial_fails`): a Default trial has no videos and XMALab
//! reads `getVideoStreams()[i]` of it (out of range, a crash in practice) after writing its
//! `calib/` folder. Default: its `calib/`, `markers/` and `reprojection_errors/` are
//! written, without images. XMALab 3 mode: the export stops there with an error.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use xma_io::export::ExportFile;
use xma_io::export::points2d::{Points2dOptions, points_2d};
use xma_io::export::{mayacam, reprojection};
use xma_model::CppCompat;
use xma_video::avi_writer::{MjpegAviWriter, Samples, encode_jpeg};
use xma_video::open::{FileSignature, OpenOptions as VideoOptions, file_signature, open_video};
use xma_video::{FrameSource, SourceKind, VideoCompat};
use xma_vision::remap::lens_undistortion_maps;
use xma_vision::undistortion::{LookupTable, create_lookup_table};
use xma_vision::visual_filter::VisualFilter;

use crate::Document;
use crate::undistort_sequence::{
    UndistortImage, UndistortPixels, complete_base_name, file_basename, read_still_image,
    write_whole,
};
use xma_io::export::safe_component;

/// The format of `FromToDialog` (`getFormat`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TrialImageFormat {
    #[default]
    Jpg,
    Tif,
    Avi,
}

impl TrialImageFormat {
    /// The text `getFormat` returns, which is also the files' extension.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            TrialImageFormat::Jpg => "jpg",
            TrialImageFormat::Tif => "tif",
            TrialImageFormat::Avi => "avi",
        }
    }

    /// `jpg`, `tif` or `avi`.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "jpg" => Some(TrialImageFormat::Jpg),
            "tif" => Some(TrialImageFormat::Tif),
            "avi" => Some(TrialImageFormat::Avi),
            _ => None,
        }
    }
}

/// What `saveTrialImages` is asked for.
#[derive(Clone, Debug, PartialEq)]
pub struct TrialImagesSpec {
    /// The trial (0-based).
    pub trial: usize,
    /// The first and last frame, 0-based and inclusive (`from - 1 ..= to - 1` of the
    /// dialog); nothing when `first > last`.
    pub first: usize,
    pub last: usize,
    pub format: TrialImageFormat,
    /// "filtered": the visual filter is applied to each image.
    pub filtered: bool,
    /// The visual filter settings (`VisualFilter_*`, per machine).
    pub filter: VisualFilter,
    /// `id`: one camera, written into the output folder itself; `None`: every camera, each
    /// into `<output>/<base>UND`.
    pub camera: Option<usize>,
}

/// How a camera's frames are made.
#[derive(Clone, Debug)]
enum Branch {
    /// Grid (and lens) remap, `_UND` names.
    Grid {
        grid: Arc<LookupTable>,
        lens: Option<Arc<LensMaps>>,
    },
    /// Lens remap only, `_UND` names.
    Lens(Arc<LensMaps>),
    /// The frame as its file holds it.
    Plain,
    /// Nothing is written (XMALab 3: a grid that is not computed).
    Nothing(String),
}

/// `Camera::undistortionMapX/Y` at the camera's size.
#[derive(Clone, Debug)]
struct LensMaps {
    x: Vec<f32>,
    y: Vec<f32>,
    width: usize,
    height: usize,
}

/// Each camera's grid lookup table and lens maps, made once (`None` until then).
#[derive(Default)]
struct MapCache(Vec<Option<CameraMapsPair>>);

/// A camera's grid lookup table and lens maps, when it has them.
type CameraMapsPair = (Option<Arc<LookupTable>>, Option<Arc<LensMaps>>);

/// One camera's part of the work.
#[derive(Clone, Debug)]
struct CameraPlan {
    camera: usize,
    files: Vec<PathBuf>,
    folder: PathBuf,
    base: String,
    branch: Branch,
    /// The frames are mirrored before the remap (a flipped camera, as XMALab holds them).
    mirrored: bool,
}

/// What happened to one camera.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CameraImagesReport {
    /// 0-based.
    pub camera: usize,
    /// The folder written into.
    pub folder: PathBuf,
    /// Files written, in frame order (one video for `avi`).
    pub written: Vec<PathBuf>,
    /// Frames not written (0-based) and why.
    pub skipped: Vec<(usize, String)>,
    /// Why the camera wrote nothing (or stopped).
    pub error: Option<String>,
}

/// What a run of [`TrialImagesJob`] did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TrialImagesReport {
    pub cameras: Vec<CameraImagesReport>,
    pub cancelled: bool,
    /// Cameras whose files replace each other (XMALab 3 mode, ledger F71).
    pub warnings: Vec<String>,
}

impl TrialImagesReport {
    /// Files written in all.
    #[must_use]
    pub fn written(&self) -> usize {
        self.cameras.iter().map(|c| c.written.len()).sum()
    }

    /// One line per camera error and skipped frame.
    #[must_use]
    pub fn problems(&self) -> Vec<String> {
        let mut out = self.warnings.clone();
        for c in &self.cameras {
            if let Some(e) = &c.error {
                out.push(format!("Camera {}: {e}", c.camera + 1));
            }
            for (f, why) in &c.skipped {
                out.push(format!("Camera {} frame {}: {why}", c.camera + 1, f + 1));
            }
        }
        out
    }
}

/// Where a run is: camera `camera` of `cameras` (positions in the run), frame `frame` of
/// `frames` of that camera.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImagesProgress {
    pub camera: usize,
    pub cameras: usize,
    pub frame: usize,
    pub frames: usize,
}

/// The work of "Undistorted Trial images" for one trial, taken from the document so that it
/// can run on another thread ([`TrialImagesJob::run`]).
#[derive(Clone, Debug)]
pub struct TrialImagesJob {
    spec: TrialImagesSpec,
    cameras: Vec<CameraPlan>,
    output: PathBuf,
    fps: f64,
    full_depth: bool,
    avi_refused: bool,
    video: VideoOptions,
    /// Cameras whose files replace each other (XMALab 3 mode, ledger F71).
    warnings: Vec<String>,
}

/// The video readers' options for a document's mode.
fn video_options(doc: &Document) -> VideoOptions {
    VideoOptions {
        compat: if doc.compat() == CppCompat::XMALAB3 {
            VideoCompat::XMALAB3
        } else {
            VideoCompat::FIXED
        },
        ..VideoOptions::default()
    }
}

impl TrialImagesJob {
    /// The work for `spec` on `doc`, writing into `output`; `files` holds each camera's
    /// files (the trial's videos as the caller resolves them: the window's session, or the
    /// project's own paths, [`crate::tracking::TrialSources`]).
    ///
    /// # Errors
    /// The trial or the camera does not exist, the trial is the Default trial (XMALab
    /// writes nothing), or `files` has no entry for a camera.
    pub fn new(
        doc: &Document,
        spec: TrialImagesSpec,
        files: &[Vec<PathBuf>],
        output: &Path,
    ) -> Result<TrialImagesJob, String> {
        Self::with_maps(doc, spec, files, output, &mut MapCache::default())
    }

    /// [`TrialImagesJob::new`] taking each camera's maps from `cache` (made once there):
    /// All Trials for External shares them between its trials.
    fn with_maps(
        doc: &Document,
        spec: TrialImagesSpec,
        files: &[Vec<PathBuf>],
        output: &Path,
        cache: &mut MapCache,
    ) -> Result<TrialImagesJob, String> {
        let p = doc.project();
        let t = p
            .trials
            .get(spec.trial)
            .ok_or_else(|| format!("no trial {}", spec.trial + 1))?;
        if t.is_default {
            return Err("the Default trial has no images".to_owned());
        }
        let compat = doc.compat();
        let wanted: Vec<usize> = match spec.camera {
            Some(c) if c >= p.cameras.len() => return Err(format!("no camera {}", c + 1)),
            Some(c) => vec![c],
            None => (0..p.cameras.len()).collect(),
        };
        let mut cameras = Vec::new();
        for c in wanted {
            let cam = &p.cameras[c];
            let derived = doc.camera(c);
            let f = files
                .get(c)
                .filter(|f| !f.is_empty())
                .ok_or_else(|| format!("camera {} has no video files", c + 1))?
                .clone();
            let names: Vec<String> = f.iter().map(|x| x.to_string_lossy().into_owned()).collect();
            // Names built from the video's name: one safe part each (ledger F70).
            let keep = compat.export_names_as_given;
            let base = complete_base_name(&file_basename(&names));
            let base = if base.is_empty() {
                base
            } else {
                safe_component(&base, keep).map_err(|e| format!("camera {}: {e}", c + 1))?
            };
            let folder = if spec.camera.is_none() {
                output.join(
                    safe_component(&format!("{base}UND"), keep)
                        .map_err(|e| format!("camera {}: {e}", c + 1))?,
                )
            } else {
                output.to_path_buf()
            };
            let (w, h) = (cam.width as usize, cam.height as usize);
            if cache.0.len() < p.cameras.len() {
                cache.0.resize(p.cameras.len(), None);
            }
            let (grid, lens) = cache.0[c]
                .get_or_insert_with(|| {
                    let lens = cam.has_model_distortion.then(|| {
                        // XMALab 3 mode, ledger A124: the matrix the C++ built its lens maps with.
                        let k = match (
                            &derived.lens_points_camera_matrix,
                            compat == CppCompat::XMALAB3,
                        ) {
                            (Some(m), true) => *m,
                            _ => cam.camera_matrix,
                        };
                        let k: [[f64; 3]; 3] =
                            std::array::from_fn(|r| std::array::from_fn(|q| k[(r, q)]));
                        let (x, y) = lens_undistortion_maps(&k, &cam.distortion, w, h);
                        Arc::new(LensMaps {
                            x,
                            y,
                            width: w,
                            height: h,
                        })
                    });
                    let grid = match (&derived.lookup, &derived.model.grid) {
                        (Some(l), _) => Some(Arc::clone(l)),
                        (None, Some(g)) => Some(Arc::new(create_lookup_table(&g.distort, w, h))),
                        (None, None) => None,
                    };
                    (grid, lens)
                })
                .clone();
            // `hasUndistortion()` is the grid image being there; `isComputed()` the grid.
            let branch = match (cam.undistortion.is_some(), grid, lens) {
                (true, Some(grid), lens) => Branch::Grid { grid, lens },
                (true, None, _) if compat.trial_images_uncomputed_grid_writes_nothing => {
                    Branch::Nothing(
                        "its undistortion grid is not computed: XMALab 3 writes nothing for it"
                            .to_owned(),
                    )
                }
                (_, _, Some(lens)) => Branch::Lens(lens),
                (_, _, None) => Branch::Plain,
            };
            let mirrored = cam.flipped && !matches!(branch, Branch::Plain);
            cameras.push(CameraPlan {
                camera: c,
                files: f,
                folder,
                base,
                branch,
                mirrored,
            });
        }
        // Ledger F71: cameras whose videos share a base name would share `<base>UND`.
        let mut warnings = Vec::new();
        if spec.camera.is_none() {
            let mut groups: Vec<(PathBuf, Vec<usize>)> = Vec::new();
            for (k, c) in cameras.iter().enumerate() {
                match groups.iter_mut().find(|(f, _)| *f == c.folder) {
                    Some((_, v)) => v.push(k),
                    None => groups.push((c.folder.clone(), vec![k])),
                }
            }
            for (folder, members) in groups.into_iter().filter(|(_, m)| m.len() > 1) {
                let names: Vec<String> = members
                    .iter()
                    .map(|&k| (cameras[k].camera + 1).to_string())
                    .collect();
                if compat.trial_images_same_folder_for_same_name {
                    warnings.push(format!(
                        "cameras {} write into the same folder {} (their videos have the same \
                         name): each camera's files replace the one's before, as in XMALab 3",
                        names.join(", "),
                        folder.display()
                    ));
                } else {
                    let name = folder
                        .file_name()
                        .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
                    for &k in &members {
                        cameras[k].folder =
                            output.join(format!("{name}_cam{}", cameras[k].camera + 1));
                    }
                }
            }
        }
        Ok(TrialImagesJob {
            spec,
            cameras,
            output: output.to_path_buf(),
            fps: if t.recording_speed > 0.0 {
                t.recording_speed
            } else {
                30.0
            },
            full_depth: !compat.trial_images_eight_bit,
            avi_refused: compat.trial_images_avi_not_written,
            video: video_options(doc),
            warnings,
        })
    }

    /// Replaces the video readers' options (the ffmpeg executables); their `compat` is kept.
    #[must_use]
    pub fn with_video_options(mut self, options: VideoOptions) -> TrialImagesJob {
        self.video = VideoOptions {
            compat: self.video.compat,
            ..options
        };
        self
    }

    /// The output folder.
    #[must_use]
    pub fn output(&self) -> &Path {
        &self.output
    }

    /// The folder each camera writes into, in order.
    #[must_use]
    pub fn folders(&self) -> Vec<(usize, PathBuf)> {
        self.cameras
            .iter()
            .map(|c| (c.camera, c.folder.clone()))
            .collect()
    }

    /// Per camera, in order: its files are undistorted (a grid or a lens model; `_UND`
    /// names), and they are mirrored left to right relative to the image as XMALab holds
    /// it (a flipped camera without undistortion, whose frames are written as their files
    /// hold them; ledger V29). The Blender-scene export (ledger F85) reads it.
    #[must_use]
    pub fn orientation(&self, doc: &Document) -> Vec<(usize, bool, bool)> {
        self.cameras
            .iter()
            .map(|c| {
                let flipped = doc
                    .project()
                    .cameras
                    .get(c.camera)
                    .is_some_and(|k| k.flipped);
                let undistorted = matches!(c.branch, Branch::Grid { .. } | Branch::Lens(_));
                (c.camera, undistorted, flipped && !c.mirrored)
            })
            .collect()
    }

    /// The name of frame `frame` (0-based) of camera plan `c`.
    fn file_name(&self, c: &CameraPlan, frame: usize) -> PathBuf {
        let und = if matches!(c.branch, Branch::Plain) {
            ""
        } else {
            "_UND"
        };
        c.folder.join(format!(
            "{}{und}.{:04}.{}",
            c.base,
            frame + 1,
            self.spec.format.as_str()
        ))
    }

    /// Writes every camera's frames, calling `progress` before each frame, stopping before
    /// the next frame once `cancel` returns true.
    ///
    /// Ported from core/Trial.cpp, Trial::saveTrialImages.
    pub fn run(
        &self,
        progress: &mut dyn FnMut(ImagesProgress),
        cancel: &dyn Fn() -> bool,
    ) -> TrialImagesReport {
        let mut report = TrialImagesReport {
            warnings: self.warnings.clone(),
            ..TrialImagesReport::default()
        };
        let n = self.cameras.len();
        for (k, c) in self.cameras.iter().enumerate() {
            if cancel() {
                report.cancelled = true;
                break;
            }
            let mut r = CameraImagesReport {
                camera: c.camera,
                folder: c.folder.clone(),
                ..CameraImagesReport::default()
            };
            // QDir().mkpath(foldername); a failure ends the whole export in XMALab.
            if let Err(e) = std::fs::create_dir_all(&c.folder) {
                r.error = Some(format!("{}: {e}", c.folder.display()));
                report.cameras.push(r);
                break;
            }
            if let Branch::Nothing(why) = &c.branch {
                r.error = Some(why.clone());
                report.cameras.push(r);
                continue;
            }
            if self.spec.format == TrialImageFormat::Avi && self.avi_refused {
                r.error = Some(
                    "no video written: XMALab 3 opens cv::VideoWriter with the codec -1, which \
                     it refuses on Linux"
                        .to_owned(),
                );
                report.cameras.push(r);
                continue;
            }
            let stopped = self.run_camera(c, k, n, &mut r, progress, cancel);
            report.cameras.push(r);
            if stopped {
                report.cancelled = true;
                break;
            }
        }
        report
    }

    /// One camera; true when cancelled.
    fn run_camera(
        &self,
        c: &CameraPlan,
        k: usize,
        n: usize,
        r: &mut CameraImagesReport,
        progress: &mut dyn FnMut(ImagesProgress),
        cancel: &dyn Fn() -> bool,
    ) -> bool {
        let mut reader = match Reader::open(&c.files, &self.video, self.full_depth) {
            Ok(rd) => rd,
            Err(e) => {
                r.error = Some(e);
                return false;
            }
        };
        let (first, last) = (self.spec.first, self.spec.last);
        let frames = if first <= last { last - first + 1 } else { 0 };
        // The video is opened at the first frame's size (the camera's, `Camera::getWidth`,
        // which a project without calibration images does not store here).
        let avi_path = c.folder.join(format!("{}.avi", c.base));
        let mut avi: Option<MjpegAviWriter> = None;
        for (i, frame) in (first..=last).enumerate().take(frames) {
            if cancel() {
                return true;
            }
            progress(ImagesProgress {
                camera: k,
                cameras: n,
                frame: i,
                frames,
            });
            let (image, rgb) = match reader.read(frame) {
                Ok(v) => v,
                Err(e) => {
                    r.skipped.push((frame, format!("could not be read: {e}")));
                    continue;
                }
            };
            let Some(out) = self.make_image(c, image) else {
                r.skipped.push((
                    frame,
                    "its size differs from the undistortion grid's".to_owned(),
                ));
                continue;
            };
            if self.spec.format == TrialImageFormat::Avi {
                let (w, h) = (out.width as u32, out.height as u32);
                if avi.is_none() {
                    match MjpegAviWriter::create(&avi_path, w, h, self.fps) {
                        Ok(wr) => avi = Some(wr),
                        Err(e) => {
                            r.error = Some(e.to_string());
                            return false;
                        }
                    }
                }
                if let Some(wr) = &mut avi
                    && let Err(e) = wr.push(w, h, Samples::Rgb(&out.rgb8()))
                {
                    r.skipped.push((frame, e.to_string()));
                }
                continue;
            }
            let path = self.file_name(c, frame);
            match encode(&out, self.spec.format, rgb).and_then(|b| write_whole(&path, &b)) {
                Ok(()) => r.written.push(path),
                Err(e) => r.skipped.push((frame, format!("not written: {e}"))),
            }
        }
        progress(ImagesProgress {
            camera: k,
            cameras: n,
            frame: frames,
            frames,
        });
        match avi {
            Some(w) if w.frames() > 0 => match w.finish() {
                Ok(p) => r.written.push(p),
                Err(e) => r.error = Some(e.to_string()),
            },
            _ if self.spec.format == TrialImageFormat::Avi => {
                r.error = Some("no frame to write".to_owned());
            }
            _ => {}
        }
        false
    }

    /// The image written for a frame as read (in its file's orientation): mirrored as
    /// XMALab holds it and remapped (undistorted cameras), then filtered. `None` when its
    /// size differs from the grid's.
    fn make_image(&self, c: &CameraPlan, mut image: UndistortImage) -> Option<UndistortImage> {
        if c.mirrored {
            image.flip();
        }
        let out = match &c.branch {
            Branch::Grid { grid, lens } => {
                if image.width != grid.width || image.height != grid.height {
                    return None;
                }
                let mut o = image.remap(&grid.map_x, &grid.map_y, grid.width, grid.height);
                if let Some(l) = lens {
                    o = o.remap(&l.x, &l.y, l.width, l.height);
                }
                o
            }
            Branch::Lens(l) => image.remap(&l.x, &l.y, l.width, l.height),
            Branch::Plain | Branch::Nothing(_) => image,
        };
        Some(if self.spec.filtered {
            filtered(&out, &self.spec.filter)
        } else {
            out
        })
    }
}

impl UndistortImage {
    /// 8-bit samples: the pixels of an 8-bit image, a deeper one scaled to 8 bits
    /// (`convertTo(CV_8U, 2^-(depth - 8))`, rounded half to even).
    fn to_u8(&self) -> (Vec<u8>, usize) {
        match &self.pixels {
            UndistortPixels::U8 { data, channels } => (data.clone(), *channels),
            UndistortPixels::U16 { data, bit_depth } => (
                xma_video::convert::scale16_to_gray8(data, u32::from(bit_depth.saturating_sub(8))),
                1,
            ),
        }
    }

    /// R, G, B samples (grey as three equal channels).
    fn rgb8(&self) -> Vec<u8> {
        let (d, cn) = self.to_u8();
        if cn == 3 {
            d
        } else {
            d.iter().flat_map(|&v| [v, v, v]).collect()
        }
    }
}

/// `FilterImage::run` on an image: each channel of an 8-bit image (the routines it calls
/// work per channel), a deeper image reduced to 8 bits first.
fn filtered(image: &UndistortImage, f: &VisualFilter) -> UndistortImage {
    let (data, cn) = image.to_u8();
    let (w, h) = (image.width, image.height);
    let out = if cn == 1 {
        f.apply(&data, w, h)
    } else {
        let planes: Vec<Vec<u8>> = (0..cn)
            .map(|k| {
                let plane: Vec<u8> = data.iter().skip(k).step_by(cn).copied().collect();
                f.apply(&plane, w, h)
            })
            .collect();
        (0..w * h)
            .flat_map(|i| planes.iter().map(move |p| p[i]))
            .collect()
    };
    UndistortImage {
        width: w,
        height: h,
        pixels: UndistortPixels::U8 {
            data: out,
            channels: cn,
        },
    }
}

/// The file's bytes: `cv::imwrite` of a JPEG (quality 95) or a TIFF; an 8-bit grey image
/// of a source XMALab holds in colour (a video) as three equal channels.
fn encode(image: &UndistortImage, format: TrialImageFormat, rgb: bool) -> Result<Vec<u8>, String> {
    match format {
        TrialImageFormat::Tif => image.encode(rgb),
        TrialImageFormat::Jpg | TrialImageFormat::Avi => {
            let (d, cn) = image.to_u8();
            let (w, h) = (image.width as u32, image.height as u32);
            let r = if cn == 3 || rgb {
                encode_jpeg(w, h, Samples::Rgb(&image_rgb(&d, cn)))
            } else {
                encode_jpeg(w, h, Samples::Gray(&d))
            };
            r.map_err(|e| e.to_string())
        }
    }
}

fn image_rgb(d: &[u8], cn: usize) -> Vec<u8> {
    if cn == 3 {
        d.to_vec()
    } else {
        d.iter().flat_map(|&v| [v, v, v]).collect()
    }
}

/// A camera's frames as `core/Image.cpp` holds them (before any mirroring).
enum Reader {
    /// Still images, one per frame (colour kept).
    Stills {
        files: Vec<PathBuf>,
        compat: VideoCompat,
        full_depth: bool,
    },
    /// A video; `rgb`: XMALab holds its frames in colour (every `AviVideo`).
    Video {
        source: Box<dyn FrameSource>,
        rgb: bool,
        full_depth: bool,
    },
}

impl Reader {
    fn open(files: &[PathBuf], video: &VideoOptions, full_depth: bool) -> Result<Reader, String> {
        let still = match files {
            [] => return Err("no video files".to_owned()),
            [one] => file_signature(one).map_err(|e| e.to_string())? == FileSignature::StillImage,
            _ => true,
        };
        if still {
            return Ok(Reader::Stills {
                files: files.to_vec(),
                compat: video.compat,
                full_depth,
            });
        }
        let source = open_video(&files[0], video).map_err(|e| e.to_string())?;
        let rgb = source.info().kind != SourceKind::Cine;
        Ok(Reader::Video {
            source,
            rgb,
            full_depth,
        })
    }

    /// Frame `frame` and whether it is written in three channels.
    fn read(&mut self, frame: usize) -> Result<(UndistortImage, bool), String> {
        match self {
            Reader::Stills {
                files,
                compat,
                full_depth,
            } => {
                let f = files
                    .get(frame)
                    .ok_or_else(|| format!("the sequence has {} images", files.len()))?;
                Ok((read_still_image(f, compat, *full_depth)?, false))
            }
            Reader::Video {
                source,
                rgb,
                full_depth,
            } => {
                let f = source.read_frame(frame).map_err(|e| e.to_string())?;
                Ok((UndistortImage::from_frame(&f, *full_depth), *rgb))
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// All Trials for External.

/// One trial's part of "All Trials for External".
#[derive(Clone, Debug)]
struct TrialPart {
    trial: usize,
    name: String,
    /// Per camera: the CSV files of `calib/`, `markers/` and `reprojection_errors/`, or why
    /// they could not be made.
    calib: Vec<Vec<ExportFile>>,
    markers: Vec<Result<Vec<ExportFile>, String>>,
    reprojection: Result<Vec<ExportFile>, String>,
    /// Per camera, the images job (`None` for the Default trial).
    images: Vec<Option<Result<TrialImagesJob, String>>>,
    is_default: bool,
    /// The trial's folder name, or why the name cannot be one (ledger F70).
    folder: Result<String, String>,
}

/// What a run of [`AllTrialsJob`] did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AllTrialsReport {
    /// Files written (CSV files, images).
    pub written: usize,
    /// Problems, one sentence each.
    pub problems: Vec<String>,
    /// The run stopped with this error (XMALab 3 mode, a Default trial).
    pub error: Option<String>,
    pub cancelled: bool,
}

/// Where an "All Trials for External" run is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllTrialsProgress {
    /// 0-based trial and camera, of `trials` and `cameras`.
    pub trial: usize,
    pub trials: usize,
    pub camera: usize,
    pub cameras: usize,
    /// Frames of the images being written.
    pub frame: usize,
    pub frames: usize,
}

/// The work of File > Export > All Trials for External, taken from the document
/// ([`AllTrialsJob::run`] runs on any thread).
#[derive(Clone, Debug)]
pub struct AllTrialsJob {
    output: PathBuf,
    trials: Vec<TrialPart>,
    n_cameras: usize,
    default_trial_fails: bool,
    compat: CppCompat,
}

impl AllTrialsJob {
    /// The work for every trial of `doc`, into `output`; `files(t)` gives trial `t`'s files
    /// per camera, `filter` the visual filter settings (used by the XMALab 3 mode, F65).
    ///
    /// Ported from ui/MainWindow.cpp, MainWindow::on_actionAll_Trials_for_External_triggered.
    #[must_use]
    pub fn new(
        doc: &Document,
        output: &Path,
        files: &dyn Fn(usize) -> Vec<Vec<PathBuf>>,
        filter: VisualFilter,
    ) -> AllTrialsJob {
        let p = doc.project();
        let compat = doc.compat();
        let n_cameras = p.cameras.len();
        let models: Vec<_> = (0..n_cameras)
            .map(|c| doc.camera(c).model.clone())
            .collect();
        let mut trials = Vec::new();
        let mut cache = MapCache::default();
        for (ti, t) in p.trials.iter().enumerate() {
            let calib = (0..n_cameras)
                .map(|c| {
                    let id = format!("_MayaCam{}.txt", c + 1);
                    mayacam::maya_cams_v2(&p.cameras, None)
                        .into_iter()
                        .filter(|f| f.name.ends_with(&id))
                        .collect()
                })
                .collect();
            let markers = (0..n_cameras)
                .map(|c| {
                    points_2d(
                        t,
                        &p.cameras,
                        &models,
                        &Points2dOptions {
                            one_file: false,
                            distorted: false,
                            offset1: false,
                            y_invert: false,
                            header: false,
                            offset_columns: false,
                            compat,
                            camera: Some(c),
                        },
                    )
                    .map(xma_io::export::ExportOutput::into_folder)
                    .map_err(|e| e.to_string())
                })
                .collect();
            let reprojection =
                reprojection::reprojection_errors(t, n_cameras, compat).map_err(|e| e.to_string());
            let trial_files = if t.is_default { Vec::new() } else { files(ti) };
            let folder = safe_component(&t.name, compat.export_names_as_given);
            let folder_name = folder.clone().unwrap_or_default();
            let images = (0..n_cameras)
                .map(|c| {
                    if t.is_default {
                        return None;
                    }
                    let dir = output
                        .join(&folder_name)
                        .join(format!("Camera {c}"))
                        .join("images");
                    // Frames 1 to the camera's frame count (`getNbImages()`).
                    let n = t.n_frames;
                    let spec = TrialImagesSpec {
                        trial: ti,
                        first: if n == 0 { 1 } else { 0 },
                        last: n.saturating_sub(1),
                        format: TrialImageFormat::Tif,
                        filtered: compat.export_all_trials_images_misplaced && c != 0,
                        filter,
                        camera: (!compat.export_all_trials_images_misplaced).then_some(c),
                    };
                    Some(TrialImagesJob::with_maps(
                        doc,
                        spec,
                        &trial_files,
                        &dir,
                        &mut cache,
                    ))
                })
                .collect();
            trials.push(TrialPart {
                trial: ti,
                name: t.name.clone(),
                folder,
                calib,
                markers,
                reprojection,
                images,
                is_default: t.is_default,
            });
        }
        AllTrialsJob {
            output: output.to_path_buf(),
            trials,
            n_cameras,
            default_trial_fails: compat.export_all_trials_default_trial_fails,
            compat,
        }
    }

    /// The output folder.
    #[must_use]
    pub fn output(&self) -> &Path {
        &self.output
    }

    /// Writes everything in XMALab's order (per trial, per camera: `calib`, `images`,
    /// `markers`, `reprojection_errors`), calling `progress` as it goes and stopping before
    /// the next file or frame once `cancel` returns true.
    pub fn run(
        &self,
        progress: &mut dyn FnMut(AllTrialsProgress),
        cancel: &dyn Fn() -> bool,
    ) -> AllTrialsReport {
        let mut rep = AllTrialsReport::default();
        let nt = self.trials.len();
        for (k, t) in self.trials.iter().enumerate() {
            // Ledger F70: never a folder outside the one chosen.
            let trial_dir = match &t.folder {
                Ok(f) => self.output.join(f),
                Err(e) => {
                    rep.problems
                        .push(format!("Trial {} (\"{}\"): {e}", t.trial + 1, t.name));
                    continue;
                }
            };
            for c in 0..self.n_cameras {
                if cancel() {
                    rep.cancelled = true;
                    return rep;
                }
                let mut tick = |frame, frames| {
                    progress(AllTrialsProgress {
                        trial: k,
                        trials: nt,
                        camera: c,
                        cameras: self.n_cameras,
                        frame,
                        frames,
                    });
                };
                tick(0, 0);
                let cam_dir = trial_dir.join(format!("Camera {c}"));
                let what = format!("{} Camera {c}", t.name);
                write_files(
                    &cam_dir.join("calib"),
                    &t.calib[c],
                    &what,
                    self.compat,
                    &mut rep,
                );
                // QDir().mkdir(<camera>/images), also for a Default trial.
                if let Err(e) = std::fs::create_dir_all(cam_dir.join("images")) {
                    rep.problems.push(format!("{what}: images: {e}"));
                }
                match &t.images[c] {
                    None if t.is_default && self.default_trial_fails => {
                        rep.error = Some(format!(
                            "{}: the Default trial has no videos (XMALab 3 reads past its \
                             list of videos here)",
                            t.name
                        ));
                        return rep;
                    }
                    None => {}
                    Some(Err(e)) => rep.problems.push(format!("{what}: images: {e}")),
                    Some(Ok(job)) => {
                        if let Err(e) = std::fs::create_dir_all(job.output()) {
                            rep.problems
                                .push(format!("{}: {e}", job.output().display()));
                        }
                        let r = job.run(&mut |p| tick(p.frame, p.frames), cancel);
                        rep.written += r.written();
                        rep.problems
                            .extend(r.problems().into_iter().map(|p| format!("{what}: {p}")));
                        if r.cancelled {
                            rep.cancelled = true;
                            return rep;
                        }
                    }
                }
                match &t.markers[c] {
                    Ok(f) => write_files(&cam_dir.join("markers"), f, &what, self.compat, &mut rep),
                    Err(e) => rep.problems.push(format!("{what}: markers: {e}")),
                }
                match &t.reprojection {
                    Ok(f) => {
                        write_files(
                            &cam_dir.join("reprojection_errors"),
                            f,
                            &what,
                            self.compat,
                            &mut rep,
                        );
                    }
                    Err(e) => rep
                        .problems
                        .push(format!("{what}: reprojection errors: {e}")),
                }
            }
        }
        rep
    }
}

/// Creates `dir` and writes `files` into it (`QDir().mkdir`, then the writers).
fn write_files(
    dir: &Path,
    files: &[ExportFile],
    what: &str,
    compat: CppCompat,
    rep: &mut AllTrialsReport,
) {
    if let Err(e) = std::fs::create_dir_all(dir) {
        rep.problems.push(format!("{what}: {}: {e}", dir.display()));
        return;
    }
    let out = xma_io::export::ExportOutput::Folder(files.to_vec());
    match crate::export::write_output_with(dir, &out, "", compat) {
        Ok(w) => rep.written += w.len(),
        Err(e) => rep.problems.push(format!("{what}: {}: {e}", dir.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats() {
        for f in [
            TrialImageFormat::Jpg,
            TrialImageFormat::Tif,
            TrialImageFormat::Avi,
        ] {
            assert_eq!(TrialImageFormat::parse(f.as_str()), Some(f));
        }
        assert_eq!(TrialImageFormat::parse("png"), None);
    }

    #[test]
    fn filter_works_per_channel() {
        let w = 6;
        let h = 4;
        let grey: Vec<u8> = (0..w * h).map(|i| (i * 9) as u8).collect();
        let rgb: Vec<u8> = grey.iter().flat_map(|&v| [v, v, v]).collect();
        let f = VisualFilter::default();
        let g = filtered(
            &UndistortImage {
                width: w,
                height: h,
                pixels: UndistortPixels::U8 {
                    data: grey.clone(),
                    channels: 1,
                },
            },
            &f,
        );
        let c = filtered(
            &UndistortImage {
                width: w,
                height: h,
                pixels: UndistortPixels::U8 {
                    data: rgb,
                    channels: 3,
                },
            },
            &f,
        );
        let (gd, _) = g.to_u8();
        let (cd, _) = c.to_u8();
        assert_eq!(cd, gd.iter().flat_map(|&v| [v, v, v]).collect::<Vec<_>>());
        assert_eq!(gd, f.apply(&grey, w, h));
    }
}
