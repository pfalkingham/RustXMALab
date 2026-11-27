//! [`TrialImages`] over a trial's own video files, read with xma-video: what `xma track`
//! and the tests use (the window reads through its frame cache instead).
//!
//! Each camera's files are those the project stores for the trial (`Trial::videos`), with
//! separators adjusted to the system and a relative path taken from the project's folder;
//! more than one file is an image sequence ([`xma_video::open::open_source`]). A flipped
//! camera's frames are mirrored left to right, as the C++ readers mirror them on load. The
//! last few frames of each camera are kept, so a tracking step reads each image once.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use xma_video::open::{OpenOptions, open_source};
use xma_video::{Frame, FrameSource, VideoCompat};

use super::TrialImages;
use crate::document::Document;

/// Frames kept per camera (a step reads the start and end frames).
const KEEP_PER_CAMERA: usize = 4;

/// One camera's source, opened on first use.
struct CameraSource {
    files: Vec<PathBuf>,
    flip: bool,
    source: Option<Box<dyn FrameSource>>,
    failed: Option<String>,
}

/// The images of one trial, read from its video files.
pub struct TrialSources {
    cameras: Vec<CameraSource>,
    options: OpenOptions,
    kept: VecDeque<((usize, usize), Arc<Frame>)>,
}

impl std::fmt::Debug for TrialSources {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrialSources")
            .field(
                "files",
                &self.cameras.iter().map(|c| &c.files).collect::<Vec<_>>(),
            )
            .finish_non_exhaustive()
    }
}

impl TrialSources {
    /// The sources of `trial` of `doc`. The readers follow the document's mode: XMALab 3's
    /// colour weights, MJPEG decoding, packed `.cine` truncated to 8 bits and 16-bit stills
    /// at 16 bits whatever they declare ([`VideoCompat::XMALAB3`]) when the document is in
    /// the XMALab 3 mode, the corrected readers otherwise (ledger V2, V15, V18).
    ///
    /// # Errors
    /// A message when the trial does not exist or is the Default trial.
    pub fn new(doc: &Document, trial: usize) -> Result<TrialSources, String> {
        let compat = if doc.compat() == xma_model::CppCompat::XMALAB3 {
            VideoCompat::XMALAB3
        } else {
            VideoCompat::FIXED
        };
        Self::with_options(
            doc,
            trial,
            OpenOptions {
                compat,
                ..OpenOptions::default()
            },
        )
    }

    /// [`TrialSources::new`] with explicit reader options.
    ///
    /// # Errors
    /// A message when the trial does not exist or is the Default trial.
    pub fn with_options(
        doc: &Document,
        trial: usize,
        options: OpenOptions,
    ) -> Result<TrialSources, String> {
        let t = doc
            .project()
            .trials
            .get(trial)
            .ok_or_else(|| format!("no trial {}", trial + 1))?;
        if t.is_default {
            return Err("the Default trial has no images".to_owned());
        }
        let dir = doc.path().and_then(Path::parent).map(Path::to_path_buf);
        let files = (0..doc.project().cameras.len())
            .map(|c| {
                t.videos.get(c).map_or_else(Vec::new, |v| {
                    v.filenames
                        .iter()
                        .map(|f| resolve_stored_path(dir.as_deref(), f))
                        .collect()
                })
            })
            .collect();
        let flips = doc.project().cameras.iter().map(|c| c.flipped).collect();
        Ok(Self::from_files(files, flips, options))
    }

    /// Sources from explicit files per camera (one entry per camera) and flip flags.
    #[must_use]
    pub fn from_files(files: Vec<Vec<PathBuf>>, flips: Vec<bool>, options: OpenOptions) -> Self {
        let cameras = files
            .into_iter()
            .enumerate()
            .map(|(c, files)| CameraSource {
                files,
                flip: flips.get(c).copied().unwrap_or(false),
                source: None,
                failed: None,
            })
            .collect();
        TrialSources {
            cameras,
            options,
            kept: VecDeque::new(),
        }
    }

    /// The files of each camera.
    #[must_use]
    pub fn files(&self) -> Vec<&[PathBuf]> {
        self.cameras.iter().map(|c| c.files.as_slice()).collect()
    }
}

impl TrialImages for TrialSources {
    fn image(&mut self, camera: usize, frame: usize) -> Result<Arc<Frame>, String> {
        if let Some((_, f)) = self.kept.iter().find(|(k, _)| *k == (camera, frame)) {
            return Ok(f.clone());
        }
        let options = self.options.clone();
        let cam = self
            .cameras
            .get_mut(camera)
            .ok_or_else(|| format!("no camera {}", camera + 1))?;
        if let Some(msg) = &cam.failed {
            return Err(msg.clone());
        }
        if cam.source.is_none() {
            if cam.files.is_empty() {
                let msg = format!("camera {} has no video in this trial", camera + 1);
                cam.failed = Some(msg.clone());
                return Err(msg);
            }
            match open_source(&cam.files, &options) {
                Ok(s) => cam.source = Some(s),
                Err(e) => {
                    let msg = format!(
                        "camera {}: {} cannot be opened: {e}",
                        camera + 1,
                        cam.files[0].display()
                    );
                    cam.failed = Some(msg.clone());
                    return Err(msg);
                }
            }
        }
        let source = cam.source.as_mut().expect("opened");
        let mut f = source.read_frame(frame).map_err(|e| {
            format!(
                "camera {}: frame {} cannot be read: {e}",
                camera + 1,
                frame + 1
            )
        })?;
        if cam.flip {
            f.flip_horizontal();
        }
        let f = Arc::new(f);
        let n_kept = self.kept.iter().filter(|(k, _)| k.0 == camera).count();
        if n_kept >= KEEP_PER_CAMERA
            && let Some(i) = self.kept.iter().position(|(k, _)| k.0 == camera)
        {
            self.kept.remove(i);
        }
        self.kept.push_back(((camera, frame), f.clone()));
        Ok(f)
    }
}

/// A path as stored in the project with both separators, `\` and `/`, made this system's.
///
/// Ported from core/HelperFunctions.cpp, littleHelper::adjustPathToOS.
#[must_use]
pub fn adjust_path_to_os(stored: &str) -> String {
    let sep = std::path::MAIN_SEPARATOR;
    stored
        .chars()
        .map(|c| if c == '\\' || c == '/' { sep } else { c })
        .collect()
}

/// A path stored in the project, with separators adjusted to the system
/// ([`adjust_path_to_os`]) and a relative one taken from the project's folder `dir` (as
/// is when `dir` is `None`).
#[must_use]
pub fn resolve_stored_path(dir: Option<&Path>, stored: &str) -> PathBuf {
    let p = PathBuf::from(adjust_path_to_os(stored));
    if p.is_absolute() {
        return p;
    }
    dir.map_or(p.clone(), |d| d.join(&p))
}

/// The bytes of a still image stored in the project (a calibration image or an
/// undistortion grid of camera `camera`) and the path they were found under: the archive
/// entry of that name (with either separator), else the entry `<camera name>/<file name>`,
/// else an entry ending in `/<camera name>/<file name>` (one folder deeper), else the file
/// on disk at the stored path ([`resolve_stored_path`], relative to the project's folder).
/// The camera views and the undistortion and calibration actions all find stills here.
///
/// After `CalibrationImage` and `UndistortionObject::getImage` (core/CalibrationImage.cpp,
/// core/UndistortionObject.cpp), which read `<project folder>/<stored path>` with the
/// separators adjusted (ui/ProjectFileIO.cpp, `CalibrationImage`).
///
/// # Errors
/// A sentence with the path tried when the image is neither in the project nor readable.
pub fn still_bytes(
    doc: &Document,
    camera: usize,
    name: &str,
) -> Result<(Vec<u8>, PathBuf), String> {
    let norm = |s: &str| s.replace('\\', "/");
    let wanted = norm(name);
    let atts = &doc.extras().attachments;
    if let Some(a) = atts.iter().find(|a| norm(&a.name) == wanted) {
        return Ok((a.data.clone(), PathBuf::from(&a.name)));
    }
    let base = wanted.rsplit('/').next().unwrap_or(&wanted).to_owned();
    if let Some(cam) = doc.project().cameras.get(camera) {
        let in_folder = format!("{}/{base}", cam.name);
        if let Some(a) = atts.iter().find(|a| norm(&a.name) == in_folder) {
            return Ok((a.data.clone(), PathBuf::from(&a.name)));
        }
        if let Some(a) = atts
            .iter()
            .find(|a| norm(&a.name).ends_with(&format!("/{in_folder}")))
        {
            return Ok((a.data.clone(), PathBuf::from(&a.name)));
        }
    }
    let path = resolve_stored_path(doc.path().and_then(Path::parent), name);
    std::fs::read(&path)
        .map(|d| (d, path.clone()))
        .map_err(|e| {
            format!(
                "not in the project and not readable at {}: {e}",
                path.display()
            )
        })
}
