//! The Trial dialog's "Denoise Images": every frame of each camera denoised with OpenCV's
//! non-local means over a temporal window ([`xma_vision::denoise`]) and written as JPEG
//! files, then, optionally, the trial relinked to those files.
//!
//! Ported from `processing/DenoiseTrial.cpp` (`DenoiseTrial::process`),
//! `ui/DenoiseDialog.cpp` and `ui/WorkspaceNavigationFrame.cpp`
//! (`on_toolButtonTrialSettings_clicked`, `:335-362`, and
//! `changeDenoiseTrialDataAfterDenoise`, `:412-430`; `docs/cpp-calibration.md`, C65).
//!
//! # What XMALab does
//!
//! - The folder question ("Save to Directory "), then the dialog: search window 35,
//!   template window 11, temporal window 5 (each bumped to the next odd number when even),
//!   filter strength 5.0, "relink Trial to denoised images" (checked).
//! - One "Denoise Trial" job per camera `i` (0-based), all at once: every frame of the trial
//!   read into memory as XMALab holds it (8-bit grey, mirrored for a flipped camera); with
//!   `w = (temporal - 1) / 2`, frame `f` is copied when `f < w` or `f > N - w` and otherwise
//!   `fastNlMeansDenoisingMulti(frames, out, f, temporal, strength, template, search)`;
//!   each written as `<folder>/Cam<i>/<base>.<f + 1, four digits>.jpg`
//!   (`cv::imwrite`, quality 95), `<base>` the `completeBaseName` of the longest start the
//!   camera's file names share.
//! - With "relink", when the last job has finished: the trial's data changed to the
//!   `*.jpg` files of each `Cam<i>` folder, in natural order (`Trial::changeTrialData`,
//!   [`crate::Edit::ChangeTrialData`]).
//!
//! # Differences (parity ledger A190 to A192, V30, V31, F66, U163)
//!
//! Default, each kept as XMALab 3 has it in its mode (`CppCompat::XMALAB3`):
//!
//! - A190 (`denoise_strength_truncated`): the strength as entered (XMALab: `int`).
//! - A191 (`denoise_window_past_end`): the last `w` frames are copied, as the first `w`
//!   are; XMALab denoises frame `N - w` with a window reaching frame `N`, which OpenCV
//!   refuses: in the XMALab 3 mode the camera's job stops there with that error, the frames
//!   before it written, and the relink (as in XMALab, whose job catches the exception and
//!   finishes) links the trial to what the folders hold, `N - w` frames ([`relink_due`]).
//! - A192 (`denoise_neighbours_already_denoised`): every frame is denoised from the original
//!   frames; XMALab writes each result into the frame it holds (`out_image = images[i]`
//!   shares the pixels), so later frames see denoised neighbours.
//! - V30 (`denoise_writes_mirrored`): a flipped camera's files are written as its source
//!   files hold the frames; XMALab writes them mirrored, so a relinked trial shows them
//!   unmirrored.
//! - F66 (`denoise_relinks_folder_listing`): the trial is relinked to the files this run
//!   wrote; XMALab lists every `*.jpg` of the folders, earlier runs' files included.
//!
//! In both modes (V31, U163): frames are read as they are needed (a window of `temporal`
//! frames is kept), not all at once; they are processed at 8 bits as XMALab does (a source
//! deeper than 8 bits is reduced first); the JPEG files are this program's encoder's, so
//! their bytes are not OpenCV's (the pixels before encoding are); each camera's job shows
//! its frame in the Progress panel and can be stopped; a camera that cannot be read or
//! written ends with an error and the trial is relinked only when every camera finished.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use xma_model::CppCompat;
use xma_video::avi_writer::{Samples, encode_jpeg};
use xma_video::open::{OpenOptions as VideoOptions, open_source};
use xma_video::{FrameSource, VideoCompat};
use xma_vision::denoise::{DenoiseParams, fast_nl_means_denoising_multi};

use crate::Document;
use crate::create::NewTrialSpec;
use crate::undistort_sequence::{complete_base_name, file_basename, folder_images, write_whole};

/// Where [`DenoiseJob::run_with`] hands each frame: frame (0-based), width, height, the
/// pixels before encoding; returns the file written.
pub type FrameSink<'a> = dyn FnMut(usize, u32, u32, &[u8]) -> Result<PathBuf, String> + 'a;

/// The dialog's values (`DenoiseDialog`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenoiseSettings {
    /// `spinBox_searchWindowSize`.
    pub search: i32,
    /// `spinBox_templateWindowSize`.
    pub template: i32,
    /// `spinBox_temporalWindowSize`.
    pub temporal: i32,
    /// `doubleSpinBox_FilterStrength`.
    pub strength: f64,
    /// `checkBox_RelinkTrial`.
    pub relink: bool,
}

impl Default for DenoiseSettings {
    /// `DenoiseDialog.ui`.
    fn default() -> Self {
        DenoiseSettings {
            search: 35,
            template: 11,
            temporal: 5,
            strength: 5.0,
            relink: true,
        }
    }
}

impl DenoiseSettings {
    /// The parameters passed to `fastNlMeansDenoisingMulti` in a mode: the strength as
    /// entered, or truncated to an integer (A190).
    #[must_use]
    pub fn params(&self, compat: CppCompat) -> DenoiseParams {
        let h = if compat.denoise_strength_truncated {
            // `int getFilterStrength() { return value(); }`, then `float` again.
            self.strength.trunc() as f32
        } else {
            self.strength as f32
        };
        DenoiseParams {
            temporal: self.temporal,
            h,
            template: self.template,
            search: self.search,
        }
    }
}

/// `DenoiseDialog::on_spinBox_*_textChanged`: an even value becomes the next odd one.
#[must_use]
pub fn odd(v: i32) -> i32 {
    if v % 2 == 0 { v.saturating_add(1) } else { v }
}

/// `<folder>/Cam<camera>`.
#[must_use]
pub fn camera_folder(folder: &Path, camera: usize) -> PathBuf {
    folder.join(format!("Cam{camera}"))
}

/// `<folder>/Cam<camera>/<base>.<frame + 1, four digits at least>.jpg`.
#[must_use]
pub fn output_path(folder: &Path, camera: usize, base: &str, frame: usize) -> PathBuf {
    camera_folder(folder, camera).join(format!("{base}.{:04}.jpg", frame + 1))
}

/// What a camera's job did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DenoiseReport {
    /// 0-based.
    pub camera: usize,
    /// Files written, in frame order.
    pub written: Vec<PathBuf>,
    /// Frames denoised and frames copied.
    pub denoised: usize,
    pub copied: usize,
    /// Why the job ended early.
    pub error: Option<String>,
    pub cancelled: bool,
}

impl DenoiseReport {
    /// Every frame was written.
    #[must_use]
    pub fn complete(&self, frames: usize) -> bool {
        self.error.is_none() && !self.cancelled && self.written.len() == frames
    }
}

/// Where a job is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenoiseProgress {
    pub frame: usize,
    pub frames: usize,
}

/// One camera's "Denoise Trial" job, taken from the document so that it can run on another
/// thread ([`DenoiseJob::run`]).
#[derive(Clone, Debug)]
pub struct DenoiseJob {
    camera: usize,
    files: Vec<PathBuf>,
    folder: PathBuf,
    base: String,
    frames: usize,
    params: DenoiseParams,
    mirror: bool,
    window_past_end: bool,
    reuse_denoised: bool,
    video: VideoOptions,
}

impl DenoiseJob {
    /// The job of camera `camera` of `trial`, reading `files` (the camera's video as the
    /// caller resolves it), writing under `folder`.
    ///
    /// # Errors
    /// The trial or camera does not exist, the trial is the Default trial, or the camera has
    /// no files.
    pub fn new(
        doc: &Document,
        trial: usize,
        camera: usize,
        files: &[PathBuf],
        folder: &Path,
        settings: &DenoiseSettings,
    ) -> Result<DenoiseJob, String> {
        let p = doc.project();
        let t = p
            .trials
            .get(trial)
            .ok_or_else(|| format!("no trial {}", trial + 1))?;
        if t.is_default {
            return Err("the Default trial has no images".to_owned());
        }
        let cam = p
            .cameras
            .get(camera)
            .ok_or_else(|| format!("no camera {}", camera + 1))?;
        if files.is_empty() {
            return Err(format!("camera {} has no video files", camera + 1));
        }
        let compat = doc.compat();
        let names: Vec<String> = files
            .iter()
            .map(|f| f.to_string_lossy().into_owned())
            .collect();
        Ok(DenoiseJob {
            camera,
            files: files.to_vec(),
            folder: folder.to_path_buf(),
            // One safe part of the file names (ledger F70); an empty base stays empty.
            base: {
                let b = complete_base_name(&file_basename(&names));
                if b.is_empty() {
                    b
                } else {
                    xma_io::export::safe_component(&b, compat.export_names_as_given)?
                }
            },
            frames: t.n_frames,
            params: settings.params(compat),
            mirror: cam.flipped && compat.denoise_writes_mirrored,
            window_past_end: compat.denoise_window_past_end,
            reuse_denoised: compat.denoise_neighbours_already_denoised,
            video: VideoOptions {
                compat: if compat == CppCompat::XMALAB3 {
                    VideoCompat::XMALAB3
                } else {
                    VideoCompat::FIXED
                },
                ..VideoOptions::default()
            },
        })
    }

    /// Replaces the video readers' options (the ffmpeg executables); their `compat` is kept.
    #[must_use]
    pub fn with_video_options(mut self, options: VideoOptions) -> DenoiseJob {
        self.video = VideoOptions {
            compat: self.video.compat,
            ..options
        };
        self
    }

    /// The camera (0-based).
    #[must_use]
    pub fn camera(&self) -> usize {
        self.camera
    }

    /// The trial's frame count, which every camera writes.
    #[must_use]
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Whether frame `f` is copied rather than denoised.
    ///
    /// Ported from processing/DenoiseTrial.cpp:69 (`i < window_size || i > N -
    /// window_size`; fixed, A191: `i + w >= N`).
    #[must_use]
    pub fn copied(&self, f: usize) -> bool {
        let w = ((self.params.temporal.max(1) - 1) / 2) as usize;
        let n = self.frames;
        if self.window_past_end {
            f < w || f as i64 > n as i64 - w as i64
        } else {
            f < w || f + w >= n
        }
    }

    /// Denoises and writes every frame, calling `progress` before each and stopping before
    /// the next once `cancel` returns true.
    ///
    /// Ported from processing/DenoiseTrial.cpp, DenoiseTrial::process.
    pub fn run(
        &self,
        progress: &mut dyn FnMut(DenoiseProgress),
        cancel: &dyn Fn() -> bool,
    ) -> DenoiseReport {
        let dir = camera_folder(&self.folder, self.camera);
        if let Err(e) = std::fs::create_dir_all(&dir) {
            return DenoiseReport {
                camera: self.camera,
                error: Some(format!("{}: {e}", dir.display())),
                ..DenoiseReport::default()
            };
        }
        let mut sink = |f: usize, w: u32, h: u32, px: &[u8]| -> Result<PathBuf, String> {
            let path = output_path(&self.folder, self.camera, &self.base, f);
            encode_jpeg(w, h, Samples::Gray(px))
                .map_err(|e| e.to_string())
                .and_then(|b| write_whole(&path, &b))
                .map(|()| path.clone())
                .map_err(|e| format!("{}: not written: {e}", path.display()))
        };
        self.run_with(progress, cancel, &mut sink)
    }

    /// [`DenoiseJob::run`] with the frames handed to `sink` (frame, width, height, the
    /// pixels before encoding) instead of written; `sink` returns the file it wrote.
    pub fn run_with(
        &self,
        progress: &mut dyn FnMut(DenoiseProgress),
        cancel: &dyn Fn() -> bool,
        sink: &mut FrameSink<'_>,
    ) -> DenoiseReport {
        let mut rep = DenoiseReport {
            camera: self.camera,
            ..DenoiseReport::default()
        };
        let mut source = match open_source(&self.files, &self.video) {
            Ok(s) => s,
            Err(e) => {
                rep.error = Some(e.to_string());
                return rep;
            }
        };
        let w = ((self.params.temporal.max(1) - 1) / 2) as usize;
        let n = self.frames;
        // The frames `f - w ..= f + w` around the frame being processed, as (index, width,
        // height, pixels).
        let mut window: VecDeque<(usize, u32, u32, Vec<u8>)> = VecDeque::new();
        for f in 0..n {
            if cancel() {
                rep.cancelled = true;
                return rep;
            }
            progress(DenoiseProgress {
                frame: f,
                frames: n,
            });
            while window.front().is_some_and(|(i, ..)| i + w < f) {
                window.pop_front();
            }
            let need_last = (f + w).min(n.saturating_sub(1));
            let mut next = window.back().map_or(f.saturating_sub(w), |(i, ..)| i + 1);
            while next <= need_last {
                match read8(&mut *source, next, self.mirror) {
                    Ok((fw, fh, px)) => window.push_back((next, fw, fh, px)),
                    Err(e) => {
                        rep.error = Some(format!("frame {}: {e}", next + 1));
                        return rep;
                    }
                }
                next += 1;
            }
            let Some(pos) = window.iter().position(|(i, ..)| *i == f) else {
                rep.error = Some(format!("frame {} was not read", f + 1));
                return rep;
            };
            let (_, fw, fh, _) = window[pos];
            let out = if self.copied(f) {
                rep.copied += 1;
                window[pos].3.clone()
            } else {
                // The temporal window as OpenCV indexes it: frames `f - w ..= f + w` of the
                // sequence (only those are read), the frame denoised in the middle; past
                // the end (A191, XMALab 3 mode) OpenCV's refusal.
                let first = f.saturating_sub(w);
                let frames: Vec<&[u8]> = window
                    .iter()
                    .filter(|(i, ..)| *i >= first)
                    .map(|(_, _, _, px)| px.as_slice())
                    .collect();
                let index = f - first;
                // With a window past the end the slice is short, so OpenCV's own check
                // (`index + T/2 >= size`) refuses it, as it refuses the whole sequence.
                if window.iter().any(|(_, w2, h2, _)| (*w2, *h2) != (fw, fh)) {
                    rep.error = Some(format!(
                        "frame {}: Input images should have the same size and type!",
                        f + 1
                    ));
                    return rep;
                }
                match fast_nl_means_denoising_multi(
                    &frames,
                    fw as usize,
                    fh as usize,
                    index,
                    &self.params,
                ) {
                    Ok(o) => {
                        rep.denoised += 1;
                        o
                    }
                    Err(e) => {
                        rep.error = Some(format!("frame {}: OpenCV refuses: {e}", f + 1));
                        return rep;
                    }
                }
            };
            if self.reuse_denoised {
                // A192, XMALab 3 mode: the result replaces the frame held in memory.
                window[pos].3.clone_from(&out);
            }
            match sink(f, fw, fh, &out) {
                Ok(path) => rep.written.push(path),
                Err(e) => {
                    rep.error = Some(e);
                    return rep;
                }
            }
        }
        progress(DenoiseProgress {
            frame: n,
            frames: n,
        });
        rep
    }
}

/// Frame `f` as XMALab holds it for denoising: 8-bit grey (`getImage(image, false)`),
/// mirrored when `mirror`.
fn read8(
    source: &mut dyn FrameSource,
    f: usize,
    mirror: bool,
) -> Result<(u32, u32, Vec<u8>), String> {
    let mut frame = source.read_frame(f).map_err(|e| e.to_string())?;
    if mirror {
        frame.flip_horizontal();
    }
    Ok((frame.width, frame.height, frame.to_gray8().into_owned()))
}

/// Whether "relink Trial to denoised images" relinks once every camera's job has ended.
/// Default: only when every camera wrote every frame. XMALab 3 mode (A191): also after a
/// job ended with OpenCV's refusal, as XMALab's `QtConcurrent` job catches the exception,
/// finishes, and the last one relinks to what the folders hold (fewer frames than the
/// trial); not after a Cancel (XMALab has none).
#[must_use]
pub fn relink_due(doc: &Document, reports: &[DenoiseReport], frames: usize) -> bool {
    if reports.is_empty() || reports.iter().any(|r| r.cancelled) {
        return false;
    }
    if doc.compat().denoise_window_past_end {
        return true;
    }
    reports.iter().all(|r| r.complete(frames))
}

/// What "relink Trial to denoised images" changes the trial to: per camera, the files
/// written (default), or every `*.jpg` of the camera's folder in natural order (F66,
/// XMALab 3 mode).
///
/// Ported from ui/WorkspaceNavigationFrame.cpp,
/// WorkspaceNavigationFrame::changeDenoiseTrialDataAfterDenoise.
#[must_use]
pub fn relink_spec(
    doc: &Document,
    trial: usize,
    folder: &Path,
    reports: &[DenoiseReport],
) -> Option<NewTrialSpec> {
    let t = doc.project().trials.get(trial)?;
    let n = doc.project().cameras.len();
    let cameras = (0..n)
        .map(|c| {
            if doc.compat().denoise_relinks_folder_listing {
                jpg_files(&camera_folder(folder, c))
            } else {
                reports
                    .iter()
                    .find(|r| r.camera == c)
                    .map(|r| r.written.clone())
                    .unwrap_or_default()
            }
        })
        .collect();
    Some(NewTrialSpec {
        name: t.name.clone(),
        cameras,
    })
}

/// `QDir::entryList({"*.jpg"}, QDir::Files | QDir::NoSymLinks)` sorted with
/// `littleHelper::compareNames`, as absolute paths.
fn jpg_files(dir: &Path) -> Vec<PathBuf> {
    folder_images(dir, true)
        .into_iter()
        .filter(|p| {
            p.extension()
                .is_some_and(|e| e.to_string_lossy().eq_ignore_ascii_case("jpg"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn even_sizes_become_odd() {
        assert_eq!(odd(4), 5);
        assert_eq!(odd(0), 1);
        assert_eq!(odd(35), 35);
        assert_eq!(odd(98), 99);
        assert_eq!(odd(i32::MAX - 1), i32::MAX);
    }

    #[test]
    fn strength_in_each_mode() {
        let s = DenoiseSettings {
            strength: 5.5,
            ..DenoiseSettings::default()
        };
        assert_eq!(s.params(CppCompat::FIXED).h, 5.5);
        assert_eq!(s.params(CppCompat::XMALAB3).h, 5.0);
    }

    #[test]
    fn names() {
        assert_eq!(
            output_path(Path::new("/o"), 1, "Camera2", 0),
            PathBuf::from("/o/Cam1/Camera2.0001.jpg")
        );
    }
}
