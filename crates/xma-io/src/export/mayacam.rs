//! MayaCam 2.0: File > Export > MayaCams 2.0.
//!
//! Ported from core/Camera.cpp, `Camera::saveMayaCamVersion2` (`:514-559`), and
//! core/Project.cpp, `Project::exportMayaCamVersion2` (`:552-570`), with the choice
//! `ui/MainWindow.cpp`, `MainWindow::on_actionMayaCams_2_0_triggered` (`:2047-2067`) offers:
//! every calibration image ("Yes") or the active trial's reference calibration image ("No").

use std::sync::Arc;

use xma_model::Camera;
use xma_vision::camera::rodrigues_to_matrix;

use super::{ExportFile, complete_base_name, g};

/// The MayaCam 2.0 text of one camera for one calibration image.
///
/// Ported from core/Camera.cpp, Camera::saveMayaCamVersion2: `image size`, `width,height`;
/// `camera matrix` and its 3 rows; `rotation` and the 3 rows of `Rodrigues` of the image's
/// rotation vector (world to camera); `translation` and its 3 values one per line; blank
/// lines between the blocks; then, only when the camera uses its lens model
/// (`hasModelDistortion()`), a blank line, `undistortion` and the 8 coefficients one per line.
/// Every line ends with `std::endl`; there is no trailing blank line.
#[must_use]
pub fn maya_cam_v2(camera: &Camera, image: usize) -> String {
    let mut out = String::new();
    out.push_str("image size\n");
    out.push_str(&format!("{},{}\n", camera.width, camera.height));
    out.push('\n');

    let row3 = |out: &mut String, a: f64, b: f64, c: f64| {
        out.push_str(&g(a));
        out.push(',');
        out.push_str(&g(b));
        out.push(',');
        out.push_str(&g(c));
        out.push('\n');
    };

    out.push_str("camera matrix\n");
    let k = &camera.camera_matrix;
    for i in 0..3 {
        row3(&mut out, k[(i, 0)], k[(i, 1)], k[(i, 2)]);
    }
    out.push('\n');

    let im = &camera.calibration_images[image];
    out.push_str("rotation\n");
    let r = rodrigues_to_matrix(&im.rotation);
    for i in 0..3 {
        row3(&mut out, r[(i, 0)], r[(i, 1)], r[(i, 2)]);
    }
    out.push('\n');

    out.push_str("translation\n");
    for i in 0..3 {
        out.push_str(&g(im.translation[i]));
        out.push('\n');
    }

    if camera.has_model_distortion {
        out.push('\n');
        out.push_str("undistortion\n");
        for d in camera.distortion {
            out.push_str(&g(d));
            out.push('\n');
        }
    }
    out
}

/// `CalibrationImage::getFilenameBase()`: the complete base name of the image's file name.
///
/// For a frame of a calibration video the C++ names the image `<video>/Frame%06d` (0-based,
/// core/CalibrationSequence.cpp, `CalibrationSequence::setCalibrationSequence`), so the base
/// is `Frame%06d`; the model stores an empty file name for these, and this rebuilds it.
#[must_use]
pub fn calibration_image_base_name(camera: &Camera, image: usize) -> String {
    let filename = &camera.calibration_images[image].filename;
    if filename.is_empty() && camera.calibration_sequence.is_some() {
        format!("Frame{image:06}")
    } else {
        complete_base_name(filename)
    }
}

/// The MayaCam 2.0 files of every camera, for every calibration image (`frame = None`, the
/// GUI's "Yes") or for one (`Some(reference)`, the GUI's "No"), each only if that image is
/// calibrated. Named `<image base name>_MayaCam<camera + 1>.txt`, cameras in order, then
/// images in order.
///
/// Ported from core/Project.cpp, Project::exportMayaCamVersion2 with `id = -1`. The GUI
/// requires a calibrated project first. An image index past a camera's images is skipped
/// (undefined behaviour in the C++).
#[must_use]
pub fn maya_cams_v2(cameras: &[Arc<Camera>], frame: Option<usize>) -> Vec<ExportFile> {
    let mut files = Vec::new();
    for (id, cam) in cameras.iter().enumerate() {
        let images = match frame {
            None => 0..cam.calibration_images.len(),
            Some(f) if f < cam.calibration_images.len() => f..f + 1,
            Some(_) => 0..0,
        };
        for f in images {
            if cam.calibration_images[f].calibrated > 0 {
                files.push(ExportFile::new(
                    format!(
                        "{}_MayaCam{}.txt",
                        calibration_image_base_name(cam, f),
                        id + 1
                    ),
                    maya_cam_v2(cam, f),
                ));
            }
        }
    }
    files
}
