//! The images the undistortion and calibration actions read: a camera's undistortion grid
//! and its calibration images (still images or the frames of a calibration video), as
//! XMALab holds them (`core/Image.cpp`: mirrored when the camera is flipped), and the bead
//! detection on them.
//!
//! Still images are the project's archive entries (what File > New dataset copied in, or
//! what the project file holds), else the file next to the project; they are 8-bit (ledger
//! V6). A frame of a calibration video is read from the video; a frame deeper than 8 bits is
//! read at its own depth in the default mode and reduced to 8 bits in the XMALab 3 mode
//! (ledger V1, [`crate::tracking::jobs::algorithm_image_for`]).

use std::path::{Path, PathBuf};

use xma_model::Point2;
use xma_video::{Frame, FrameSource};
use xma_vision::beads::{BeadImage, bead_blob_params, detect_beads};
use xma_vision::depth::to_f32_8bit_scale;
use xma_vision::imgproc::blob::detect_blobs_f32;

use crate::document::Document;
use crate::tracking::jobs::{AlgorithmImage, algorithm_image_for, detect_options};

/// The bytes of a still image stored in the project ([`crate::tracking::images::still_bytes`],
/// which the camera views use too), an error naming the image.
fn still_bytes(doc: &Document, camera: usize, name: &str) -> Result<(Vec<u8>, PathBuf), String> {
    crate::tracking::images::still_bytes(doc, camera, name).map_err(|e| format!("{name}: {e}"))
}

fn decode(doc: &Document, data: &[u8], path: &Path) -> Result<Frame, String> {
    let compat = crate::create::video_options(doc.compat()).compat;
    xma_video::image_file::decode_image_with(data, path, &compat).map_err(|e| e.to_string())
}

fn mirrored(doc: &Document, camera: usize, mut f: Frame) -> Frame {
    if doc.project().cameras[camera].flipped {
        f.flip_horizontal();
    }
    f
}

/// The undistortion grid image of `camera` (`UndistortionObject::getImage`).
///
/// # Errors
/// A sentence naming the image when the camera has no grid or the image cannot be read.
pub(crate) fn grid_frame(doc: &Document, camera: usize) -> Result<Frame, String> {
    let cam = &doc.project().cameras[camera];
    let g = cam
        .undistortion
        .as_ref()
        .ok_or_else(|| format!("{} has no undistortion grid", cam.name))?;
    let (data, path) = still_bytes(doc, camera, &g.image_filename)?;
    Ok(mirrored(doc, camera, decode(doc, &data, &path)?))
}

/// The calibration videos one edit has opened, kept open for its later reads of the same
/// camera's video: opening a video costs far more than reading a frame from an open one
/// (calibration.md finding 11 of the review of 20 November 2026). A frame read from a kept
/// source is the frame a newly opened source reads (`review13_calibration_frames`).
#[derive(Default)]
pub(crate) struct CalibrationVideos {
    open: Vec<(usize, PathBuf, Box<dyn FrameSource>)>,
}

/// Calibration image `image` of `camera`, distorted (`CalibrationSequence::getImage(image,
/// true)`).
///
/// # Errors
/// A sentence naming the image or video when it cannot be read.
pub(crate) fn calibration_frame(
    doc: &Document,
    camera: usize,
    image: usize,
) -> Result<Frame, String> {
    calibration_frame_in(doc, camera, image, &mut CalibrationVideos::default())
}

/// [`calibration_frame`], with a calibration video taken from `videos` when this edit has
/// opened it already (and kept there when it is opened now).
///
/// # Errors
/// As [`calibration_frame`].
pub(crate) fn calibration_frame_in(
    doc: &Document,
    camera: usize,
    image: usize,
    videos: &mut CalibrationVideos,
) -> Result<Frame, String> {
    let cam = &doc.project().cameras[camera];
    if let Some(seq) = &cam.calibration_sequence {
        // The stored path with its separators adjusted, as ui/ProjectFileIO.cpp reads it
        // (`CalibrationSequence`), and a relative one taken from the project's folder, as
        // the camera views take it.
        let p = crate::tracking::images::resolve_stored_path(
            doc.path().and_then(Path::parent),
            &seq.filename,
        );
        let at = videos
            .open
            .iter()
            .position(|(c, q, _)| *c == camera && *q == p);
        let at = match at {
            Some(i) => i,
            None => {
                let src =
                    xma_video::open::open_video(&p, &crate::create::video_options(doc.compat()))
                        .map_err(|e| format!("{}: {e}", p.display()))?;
                videos.open.push((camera, p.clone(), src));
                videos.open.len() - 1
            }
        };
        let f = videos.open[at]
            .2
            .read_frame(image)
            .map_err(|e| format!("{} frame {}: {e}", p.display(), image + 1))?;
        return Ok(mirrored(doc, camera, f));
    }
    let im = cam
        .calibration_images
        .get(image)
        .ok_or_else(|| format!("{} has no calibration image {}", cam.name, image + 1))?;
    let (data, path) = still_bytes(doc, camera, &im.filename)?;
    Ok(mirrored(doc, camera, decode(doc, &data, &path)?))
}

/// The beads of an image: `BlobDetection::process` with the `BlobDetector*` settings, the
/// keypoint centres in detection order.
///
/// Ported from processing/BlobDetection.cpp, BlobDetection::process (and `process_finished`,
/// whose storing the caller does). A frame deeper than 8 bits in the default mode goes
/// through the full-depth detector (`detect_blobs_f32`, ledger V1).
///
/// # Errors
/// OpenCV's parameter check for these settings (XMALab does not catch it).
pub(crate) fn detect(
    doc: &Document,
    frame: &Frame,
    kind: BeadImage,
) -> Result<Vec<Point2>, String> {
    let settings = detect_options(doc).blob;
    match algorithm_image_for(frame, doc.compat()) {
        AlgorithmImage::U8(im) => {
            detect_beads(&im, &settings, kind).map_err(|e| format!("bead detection: {e}"))
        }
        AlgorithmImage::Deep { image, bit_depth } => {
            let params = bead_blob_params(&settings, kind);
            let f = to_f32_8bit_scale(&image, bit_depth);
            detect_blobs_f32(&f, &params)
                .map(|k| {
                    k.iter()
                        .map(|k| Point2::new(f64::from(k.pt.0), f64::from(k.pt.1)))
                        .collect()
                })
                .map_err(|e| format!("bead detection: {e}"))
        }
    }
}

/// Whether the calibration object has white beads (`CalibrationObject::hasWhiteBlobs`).
pub(crate) fn white_blobs(doc: &Document) -> bool {
    matches!(
        &doc.project().calibration_object,
        Some(xma_model::CalibrationObject::Frame(f)) if f.white_blobs
    )
}

#[cfg(test)]
mod tests {
    //! Review 13 (performance round, calibration.md finding 11): frames read from a
    //! calibration video kept open by one edit against frames read from a newly opened
    //! video, as `calibration_frame` read every frame before.

    use std::sync::Arc;

    use super::*;

    /// `pig_clean` with camera 0's calibration images replaced by `Camera1.avi` as a
    /// calibration sequence (camera 0 flipped when `flipped`), in `compat`.
    fn doc_with_video(compat: xma_model::CppCompat, flipped: bool) -> Option<Document> {
        let p = xma_oracle::dataset("pig_clean")?.project;
        let options = crate::OpenOptions {
            compat,
            ..Default::default()
        };
        let mut loaded = crate::read_project(&p, &options).expect("reads");
        let avi = xma_oracle::workspace_root().join("ExampleDatasets/1 - Pig dataset/Camera1.avi");
        if !avi.is_file() {
            assert!(!xma_oracle::oracle_required(), "{} absent", avi.display());
            return None;
        }
        let cam = Arc::make_mut(&mut loaded.project.cameras[0]);
        cam.flipped = flipped;
        cam.calibration_sequence = Some(xma_model::CalibrationSequence {
            filename: avi.to_string_lossy().into_owned(),
            n_frames: 100,
            width: cam.width,
            height: cam.height,
        });
        let mut doc = Document::from_loaded(loaded);
        doc.update_after_load();
        Some(doc)
    }

    #[test]
    fn review13_calibration_frames() {
        // Forward, backward, repeated and scattered reads, and frames past the end.
        let order = [0usize, 1, 2, 50, 49, 10, 10, 99, 0, 73, 5, 100, 98, 1000, 3];
        for compat in [xma_model::CppCompat::FIXED, xma_model::CppCompat::XMALAB3] {
            for flipped in [false, true] {
                let Some(doc) = doc_with_video(compat, flipped) else {
                    return;
                };
                let mut videos = CalibrationVideos::default();
                for &f in &order {
                    let kept = calibration_frame_in(&doc, 0, f, &mut videos);
                    let fresh = calibration_frame(&doc, 0, f);
                    assert_eq!(kept, fresh, "{compat:?} flipped {flipped} frame {f}");
                    // A still-image camera through the same reader.
                    assert_eq!(
                        calibration_frame_in(&doc, 1, 0, &mut videos),
                        calibration_frame(&doc, 1, 0)
                    );
                }
                assert_eq!(videos.open.len(), 1);
            }
        }
    }

    /// A video that cannot be opened: the same error on every read, nothing kept.
    #[test]
    fn review13_calibration_frames_missing_video() {
        let Some(mut doc) = doc_with_video(xma_model::CppCompat::FIXED, false) else {
            return;
        };
        let cam = Arc::make_mut(&mut doc.project.cameras[0]);
        cam.calibration_sequence.as_mut().unwrap().filename = "/nonexistent/x.avi".into();
        let mut videos = CalibrationVideos::default();
        for f in [0, 1, 0] {
            let kept = calibration_frame_in(&doc, 0, f, &mut videos);
            assert!(kept.is_err());
            assert_eq!(kept, calibration_frame(&doc, 0, f));
        }
        assert!(videos.open.is_empty());
    }
}
