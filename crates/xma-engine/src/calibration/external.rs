//! The external calibration (C43 to C46): reading a MayaCam 2.0 file, what the
//! external-calibration frame shows, and its "APPLY CHANGES".
//!
//! Ported from `ui/ExternalCalibrationFrame.cpp` (`update` `:54-113`,
//! `on_pushButton_Apply_clicked` `:115-182`, `on_pushButton_MayaCam_clicked` `:184-273`).
//!
//! # Fixed C++ defects
//!
//! - Ledger F40 (CD6): the C++ neither checks that the file opened nor what `fscanf` read,
//!   so a missing file crashes and a malformed one fills the frame with indeterminate
//!   values. [`read_mayacam2`] gives a sentence for the user instead, in both modes.
//! - Ledger F1: the C++ reads the eight coefficients as one comma-separated line, but every
//!   MayaCam 2.0 file XMALab writes has one per line, so it reads k1 and leaves seven
//!   values indeterminate. [`read_mayacam2`] reads both layouts, in both modes.
//! - Ledger A148 (CD7): the frame's spin boxes hold 12 decimals, so Apply rounds every
//!   value. The default applies the values as given; the XMALab 3 mode
//!   (`CppCompat::external_calibration_spin_box_rounding`) rounds them as the spin boxes do
//!   ([`ExternalCalibration::as_spin_boxes`]).
//! - Ledger A147 (CD22): Apply flags no trial for recomputation. The default flags every
//!   trial; the XMALab 3 mode (`CppCompat::external_calibration_trials_not_flagged`) does
//!   not.
//! - Ledger A149 (CD24): the frame shows a camera matrix or a pose the C++ holds without
//!   values (a camera reset, the image of a new external project);
//!   [`Document::external_calibration`] gives the values the model holds (identity or the
//!   last values set; a zero pose).
//! - Ledger A150: Apply on a camera without a calibration image reads past the image list
//!   in the C++; here it is refused, in both modes.
//! - Ledger A162: Apply accepts any rotation matrix (`cv::Rodrigues` stores the rotation
//!   nearest to it, or one far from it for a reflection or a singular matrix), a width or
//!   height of 0 and a camera matrix with a focal length of 0 or less. The default refuses
//!   them with a sentence; the XMALab 3 mode
//!   (`CppCompat::external_calibration_accepts_degenerate`) applies them as XMALab does.
//! - Ledger A165: `cv::Rodrigues` stores a rotation within about 1e-5 rad of 0 or of 180
//!   degrees inexactly (up to 2.1e-5 per matrix entry; a rotation by 1e-5 rad becomes none).
//!   The default stores the rotation vector of the nearest rotation, exact to rounding
//!   (`xma_vision::camera::nearest_rotation_to_rodrigues`); the XMALab 3 mode
//!   (`CppCompat::external_rotation_cv_rodrigues`) stores `cv::Rodrigues`'.

use std::path::Path;
use std::sync::Arc;

use xma_model::{CalibrationType, Matrix3, Vector3};
use xma_vision::camera::{matrix_to_rodrigues, nearest_rotation_to_rodrigues, rodrigues_to_matrix};

use super::ExternalCalibration;
use crate::document::Document;
use crate::edit::{EditError, EditOutcome, EditResult};

/// The range of the frame's double spin boxes (`ExternalCalibrationFrame.ui`).
const SPIN_LIMIT: f64 = 999_999_999.0;
/// The maximum of the width and height spin boxes.
const SIZE_LIMIT: u32 = 100_000;

/// `QDoubleSpinBox::setValue` with 12 decimals and the range +-999999999: the value rounded
/// to 12 decimal places (`QString::number(value, 'f', 12).toDouble()`, Qt's
/// `QDoubleSpinBoxPrivate::round`), then bounded to the range.
#[must_use]
pub fn spin_box_value(v: f64) -> f64 {
    if !v.is_finite() {
        return v;
    }
    // Qt rounds an exact tie away from zero (`format_qt_fixed`), where Rust's `{:.12}`
    // rounds it to even (review 12). Zero keeps its sign, as before.
    if v == 0.0 {
        return v;
    }
    let r = xma_io::cparse::qt_to_double(&xma_io::cfmt::format_qt_fixed(v, 12));
    r.clamp(-SPIN_LIMIT, SPIN_LIMIT)
}

impl ExternalCalibration {
    /// The values as the external-calibration frame's spin boxes hold them in XMALab 3:
    /// every number rounded to 12 decimals and bounded to +-999999999, the width and height
    /// bounded to 100000 (ledger A148).
    #[must_use]
    pub fn as_spin_boxes(&self) -> ExternalCalibration {
        let m = |a: &[[f64; 3]; 3]| a.map(|row| row.map(spin_box_value));
        ExternalCalibration {
            width: self.width.min(SIZE_LIMIT),
            height: self.height.min(SIZE_LIMIT),
            camera_matrix: m(&self.camera_matrix),
            rotation: m(&self.rotation),
            translation: self.translation.map(spin_box_value),
            distortion: self.distortion.map(|d| d.map(spin_box_value)),
        }
    }

    /// True when every number is finite.
    fn is_finite(&self) -> bool {
        self.camera_matrix
            .iter()
            .chain(&self.rotation)
            .flatten()
            .chain(&self.translation)
            .chain(self.distortion.iter().flatten())
            .all(|v| v.is_finite())
    }
}

/// A reader with `fscanf`'s rules: a white-space character of the format matches any amount
/// of white space (none included); numbers skip leading white space.
struct Scanner<'a> {
    text: &'a str,
    pos: usize,
}

impl Scanner<'_> {
    fn line(&self) -> usize {
        self.text[..self.pos].matches('\n').count() + 1
    }
    fn skip_ws(&mut self) {
        let rest = &self.text[self.pos..];
        self.pos += rest.len() - rest.trim_start().len();
    }
    fn at_end(&mut self) -> bool {
        self.skip_ws();
        self.pos >= self.text.len()
    }
    /// A literal of the format (`image size`, `,`): white space in it matches any white
    /// space.
    fn literal(&mut self, lit: &str) -> Result<(), String> {
        for ch in lit.chars() {
            if ch.is_whitespace() {
                self.skip_ws();
            } else if self.text[self.pos..].starts_with(ch) {
                self.pos += ch.len_utf8();
            } else {
                return Err(format!("'{}' expected on line {}", lit.trim(), self.line()));
            }
        }
        Ok(())
    }
    /// The longest prefix (after white space) made of number characters that parses.
    fn token(&mut self) -> &str {
        self.skip_ws();
        let rest = &self.text[self.pos..];
        let n = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.')))
            .unwrap_or(rest.len());
        &rest[..n]
    }
    fn number(&mut self, what: &str) -> Result<f64, String> {
        // The line the value is on: after the white space before it.
        self.skip_ws();
        let line = self.line();
        let tok = self.token().to_owned();
        match tok.parse::<f64>() {
            Ok(v) if v.is_finite() => {
                self.pos += tok.len();
                Ok(v)
            }
            _ => Err(format!("{what} on line {line} is not a number")),
        }
    }
    fn size(&mut self, what: &str) -> Result<u32, String> {
        // The line the value is on: after the white space before it.
        self.skip_ws();
        let line = self.line();
        let tok = self.token().to_owned();
        match tok.parse::<u32>() {
            Ok(v) => {
                self.pos += tok.len();
                Ok(v)
            }
            _ => Err(format!(
                "{what} on line {line} is not a whole number of pixels"
            )),
        }
    }
    fn row(&mut self, what: &str) -> Result<[f64; 3], String> {
        let a = self.number(what)?;
        self.literal(",")?;
        let b = self.number(what)?;
        self.literal(",")?;
        let c = self.number(what)?;
        Ok([a, b, c])
    }
}

/// Reads a MayaCam 2.0 file: `image size` / `w,h` / `camera matrix` and three rows of three
/// comma-separated values / `rotation` and three rows / `translation` and three values one
/// per line / optionally `undistortion` and eight coefficients, one per line (as
/// `Camera::saveMayaCamVersion2` and this program's export write them) or comma-separated
/// on one line (as `ExternalCalibrationFrame` reads them). Text after the last value read is
/// ignored, as by the C++.
///
/// Ported from `ui/ExternalCalibrationFrame.cpp:184-273`, `on_pushButton_MayaCam_clicked`,
/// with ledger F1 and F40 fixed (both modes).
///
/// # Errors
/// A sentence for the user when the file cannot be read or is not a MayaCam 2.0 file.
pub fn read_mayacam2(path: &Path) -> Result<ExternalCalibration, String> {
    let bytes = std::fs::read(path).map_err(|e| {
        format!(
            "The MayaCam 2.0 file {} cannot be read: {e}.",
            path.display()
        )
    })?;
    let text = String::from_utf8_lossy(&bytes);
    parse_mayacam2(&text).map_err(|e| format!("{} is not a MayaCam 2.0 file: {e}.", path.display()))
}

/// [`read_mayacam2`] on the file's text.
///
/// # Errors
/// What is missing or malformed, with its line.
pub fn parse_mayacam2(text: &str) -> Result<ExternalCalibration, String> {
    let mut s = Scanner { text, pos: 0 };
    s.literal("image size\n")?;
    let width = s.size("the width")?;
    s.literal(",")?;
    let height = s.size("the height")?;
    s.literal("\n\ncamera matrix\n")?;
    let mut camera_matrix = [[0.0; 3]; 3];
    for row in &mut camera_matrix {
        *row = s.row("a camera matrix value")?;
    }
    s.literal("\nrotation\n")?;
    let mut rotation = [[0.0; 3]; 3];
    for row in &mut rotation {
        *row = s.row("a rotation value")?;
    }
    s.literal("\ntranslation\n")?;
    let mut translation = [0.0; 3];
    for v in &mut translation {
        *v = s.number("a translation value")?;
    }
    // `fscanf(pfile, "\nundistortion\n") < 0`: only white space is left.
    let distortion = if s.at_end() {
        None
    } else {
        s.literal("undistortion\n")?;
        let mut d = [0.0; 8];
        for (i, v) in d.iter_mut().enumerate() {
            if i > 0 {
                // One per line (the export) or comma-separated (the C++ reader).
                s.skip_ws();
                if s.text[s.pos..].starts_with(',') {
                    s.pos += 1;
                }
            }
            *v = s.number("a distortion coefficient")?;
        }
        Some(d)
    };
    Ok(ExternalCalibration {
        width,
        height,
        camera_matrix,
        rotation,
        translation,
        distortion,
    })
}

impl Document {
    /// What the external-calibration frame of `camera` shows (`ExternalCalibrationFrame::
    /// update`): the camera's size and matrix, the rotation matrix and translation of its
    /// calibration image 0, and its coefficients when the lens model is on. `None` for a
    /// camera out of range or without a calibration image (the C++ frame then keeps what it
    /// showed). Ledger A149: where the C++ holds no values (a camera reset, the image of a
    /// new project) this gives the model's: the camera matrix as last set (identity for a
    /// new camera) and a zero pose (identity rotation).
    ///
    /// Ported from `ui/ExternalCalibrationFrame.cpp:54-113`.
    #[must_use]
    pub fn external_calibration(&self, camera: usize) -> Option<ExternalCalibration> {
        let cam = self.project.cameras.get(camera)?;
        let im = cam.calibration_images.first()?;
        let k = cam.camera_matrix;
        let r = rodrigues_to_matrix(&im.rotation);
        Some(ExternalCalibration {
            width: cam.width,
            height: cam.height,
            camera_matrix: [0, 1, 2].map(|i| [k[(i, 0)], k[(i, 1)], k[(i, 2)]]),
            rotation: [0, 1, 2].map(|i| [r[(i, 0)], r[(i, 1)], r[(i, 2)]]),
            translation: [im.translation.x, im.translation.y, im.translation.z],
            distortion: cam.has_model_distortion.then_some(cam.distortion),
        })
    }
}

/// The largest difference, entry by entry, between the rotation matrix of an external
/// calibration and the nearest rotation, for it to be accepted (ledger A162). A matrix
/// written with 12 significant digits, or computed in single precision, is within it; so is
/// one typed with 4 or more decimals (measured on 20,000 random rotations rounded to 4
/// decimals: at most 7.5e-5 from the nearest rotation).
pub const ROTATION_TOLERANCE: f64 = 1e-4;

/// The largest entry of `|m - r|`, where `r` is the rotation nearest to `m` (`U Vt` of its
/// SVD, the same orthogonalisation `cv::Rodrigues` applies). `None` when the determinant of
/// `m` is not positive (a reflection, or a singular matrix), which has no nearest rotation
/// that `cv::Rodrigues` would give.
#[must_use]
pub fn distance_from_rotation(m: &[[f64; 3]; 3]) -> Option<f64> {
    let a = Matrix3::from_fn(|i, j| m[i][j]);
    let det = xma_vision::linalg::determinant(&a);
    if det.is_nan() || det <= 0.0 {
        return None;
    }
    let usv = xma_vision::linalg::svd(&a, xma_vision::linalg::SvdMode::Thin);
    let mut worst: f64 = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let mut r = 0.0;
            for k in 0..3 {
                r += usv.u[(i, k)] * usv.vt[(k, j)];
            }
            worst = worst.max((m[i][j] - r).abs());
        }
    }
    Some(worst)
}

/// Ledger A162: what the default mode refuses in "APPLY CHANGES" besides non-finite values:
/// a width or height of 0, a camera matrix whose focal length (`fx`, `fy`) is not positive,
/// and a rotation matrix whose determinant is not positive or that differs from the
/// nearest rotation by more than [`ROTATION_TOLERANCE`] in an entry.
fn check_values(v: &ExternalCalibration) -> Result<(), String> {
    if v.width == 0 || v.height == 0 {
        return Err(format!(
            "The image size of the external calibration must be at least 1 x 1 pixels (it is \
             {} x {}).",
            v.width, v.height
        ));
    }
    let (fx, fy) = (v.camera_matrix[0][0], v.camera_matrix[1][1]);
    if fx <= 0.0 || fy <= 0.0 {
        return Err(format!(
            "The focal lengths of the camera matrix must be positive (fx is {fx}, fy is {fy})."
        ));
    }
    match distance_from_rotation(&v.rotation) {
        None => Err(
            "The rotation matrix of the external calibration is not a rotation: its \
             determinant is not positive."
                .to_owned(),
        ),
        Some(d) if d > ROTATION_TOLERANCE => Err(format!(
            "The rotation matrix of the external calibration is not a rotation: it differs \
             from the nearest rotation by up to {d:.3e} (at most {ROTATION_TOLERANCE:e} is \
             accepted)."
        )),
        Some(_) => Ok(()),
    }
}

/// C44, "APPLY CHANGES": the frame's values become camera `camera`'s calibration.
///
/// Ported from `ui/ExternalCalibrationFrame.cpp:115-182`, `on_pushButton_Apply_clicked`:
/// `setResolution`; `setCameraMatrix` (all nine entries); image 0 `setMatrices` with
/// `cv::Rodrigues` of the rotation matrix (OpenCV orthogonalises a matrix that is not a
/// rotation through its SVD) and the translation, which makes the image calibrated (1);
/// `setDistortionCoefficiants` when the lens model is on, else `resetDistortion`;
/// `setRecalibrationRequired(0)`. With ledger A147 and A148 fixed unless the document's
/// mode says otherwise.
pub(crate) fn apply_external(
    doc: &mut Document,
    camera: usize,
    calibration: &ExternalCalibration,
) -> EditResult {
    if doc.project.calibration_type != CalibrationType::External {
        return Err(EditError::Invalid(
            "the external calibration applies only to a project with external calibration"
                .to_owned(),
        ));
    }
    let Some(cam) = doc.project.cameras.get(camera) else {
        return Err(EditError::Invalid(format!("camera {camera} out of range")));
    };
    // Ledger A150.
    if cam.calibration_images.is_empty() {
        return Err(EditError::Rejected(format!(
            "{} has no calibration image to hold the external calibration.",
            cam.name
        )));
    }
    if !calibration.is_finite() {
        return Err(EditError::Rejected(
            "Every value of the external calibration must be a finite number.".to_owned(),
        ));
    }
    let compat = doc.compat();
    // Ledger A162.
    if !compat.external_calibration_accepts_degenerate {
        check_values(calibration).map_err(EditError::Rejected)?;
    }
    let values = if compat.external_calibration_spin_box_rounding {
        calibration.as_spin_boxes()
    } else {
        calibration.clone()
    };
    let m = |a: &[[f64; 3]; 3]| Matrix3::from_fn(|i, j| a[i][j]);
    // Ledger A165: `cv::Rodrigues` in the XMALab 3 mode; the rotation vector of the nearest
    // rotation, exact for every axis and angle, in the default (where `check_values` has
    // refused a matrix without a nearest rotation).
    let r = m(&values.rotation);
    let rotation = if compat.external_rotation_cv_rodrigues {
        matrix_to_rodrigues(&r)
    } else {
        nearest_rotation_to_rodrigues(&r).unwrap_or_else(|| matrix_to_rodrigues(&r))
    };
    let cam = Arc::make_mut(&mut doc.project.cameras[camera]);
    cam.width = values.width;
    cam.height = values.height;
    cam.camera_matrix = m(&values.camera_matrix);
    cam.calibrated = true;
    let im = &mut cam.calibration_images[0];
    im.rotation = rotation;
    im.translation = Vector3::from(values.translation);
    im.calibrated = 1;
    match values.distortion {
        Some(d) => {
            cam.distortion = d;
            cam.has_model_distortion = true;
        }
        None => {
            if cam.has_model_distortion {
                cam.distortion = [0.0; 8];
                cam.has_model_distortion = false;
            }
        }
    }
    doc.sync_camera_intrinsics();
    // The pose and camera matrix changed: held point lists no longer belong to them.
    doc.invalidate_held_points(camera);
    if let Some(r) = doc.session_mut().camera_recalibration.get_mut(camera) {
        *r = 0;
    }
    // Ledger A147 (CD22).
    if !compat.external_calibration_trials_not_flagged {
        for t in 0..doc.project.trials.len() {
            doc.set_trial_requires_recomputation(t, true);
        }
    }
    Ok(EditOutcome::none())
}
