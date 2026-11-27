//! The camera sizes of a project without calibration images, set when it is opened.
//!
//! Ported from ui/MainWindow.cpp, MainWindow::loadProjectFinished (`:793-804`): when no
//! camera has a calibration image, the project becomes `NO_CALIBRATION` and
//! `Trial::setCameraSizes` (core/Trial.cpp, `:2424-2430`) gives camera `i` the size of the
//! image the first trial's video `i` holds (its first frame; an image that cannot be read
//! is 0 x 0). XMALab does this after `checkTrialImagePaths`, so a video found elsewhere
//! counts. The project file stores no camera size for such a project, so without this step
//! every camera is 0 x 0, and "Change Trial Data" (and the denoising relink, which uses it)
//! refuses any video ("Width and Height of the videos does is not equal to the calibration
//! images").
//!
//! Ledger F69 (`CppCompat::camera_sizes_from_first_trial_only`): XMALab reads only the first
//! trial: a camera whose video in that trial is not found stays 0 x 0 although another
//! trial's video gives its size, and a project without trials reads past the trial list
//! (a crash). Default: each camera takes the size of its video in the first trial where it
//! can be read; a camera none gives stays 0 x 0 with a warning. XMALab 3 mode: the first
//! trial only, and a project without trials keeps 0 x 0 cameras with a warning in place of
//! the crash.

use std::path::Path;
use std::sync::Arc;

use xma_io::LoadedProject;
use xma_model::CppCompat;
use xma_video::VideoCompat;
use xma_video::open::{OpenOptions as VideoOptions, open_source};

/// The size of the first image of `files` (a video, or an image sequence), if it opens.
fn video_size(files: &[std::path::PathBuf], compat: CppCompat) -> Option<(u32, u32)> {
    let options = VideoOptions {
        compat: if compat == CppCompat::XMALAB3 {
            VideoCompat::XMALAB3
        } else {
            VideoCompat::FIXED
        },
        ..VideoOptions::default()
    };
    let source = open_source(files, &options).ok()?;
    let i = source.info();
    (i.width > 0 && i.height > 0).then_some((i.width, i.height))
}

/// The files of camera `c` of a trial, as stored (`\` and `/` both separators; a relative
/// name taken from the project's folder).
fn files_of(t: &xma_model::Trial, c: usize, dir: Option<&Path>) -> Vec<std::path::PathBuf> {
    t.videos.get(c).map_or_else(Vec::new, |v| {
        v.filenames
            .iter()
            .map(|f| crate::tracking::images::resolve_stored_path(dir, f))
            .collect()
    })
}

/// Sets the camera sizes of a project without calibration images from its trials (see the
/// module documentation); does nothing for a project with calibration images.
pub(crate) fn set_camera_sizes(loaded: &mut LoadedProject, path: &Path) {
    let project = &mut loaded.project;
    if project
        .cameras
        .iter()
        .any(|c| !c.calibration_images.is_empty())
    {
        return;
    }
    let compat = loaded.compat;
    let dir = path.parent();
    let trials: Vec<Arc<xma_model::Trial>> = project
        .trials
        .iter()
        .filter(|t| !t.is_default)
        .cloned()
        .collect();
    if project.trials.is_empty() {
        if compat.camera_sizes_from_first_trial_only && !project.cameras.is_empty() {
            loaded.warnings.push(
                "XMALab 3 reads the first trial of a project without trials here (it stops); \
                 the cameras have no size"
                    .to_owned(),
            );
        }
        return;
    }
    for c in 0..project.cameras.len() {
        let size = if compat.camera_sizes_from_first_trial_only {
            project
                .trials
                .first()
                .and_then(|t| video_size(&files_of(t, c, dir), compat))
        } else {
            trials
                .iter()
                .find_map(|t| video_size(&files_of(t, c, dir), compat))
        };
        match size {
            Some((w, h)) => {
                let cam = Arc::make_mut(&mut project.cameras[c]);
                (cam.width, cam.height) = (w, h);
            }
            None => loaded.warnings.push(format!(
                "{}: no video of a trial can be read, so the camera has no size",
                project.cameras[c].name
            )),
        }
    }
}
