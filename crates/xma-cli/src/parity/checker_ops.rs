//! The operations of calibration scripts in format 6 (`oracle/FORMAT.md`, "Calibration runs,
//! version 6") that the checkerboard runs use, applied as the C++ driver applies them
//! (`oracle/xmalab-cpp/src/ui/CalibOracleDriver.cpp`, `runOp`): `NewProject` with
//! `target=checkerboard`, `CalibResetFrame`, `CalibResetCamera`, `CalibDeleteFrame`,
//! `CalibDistortion`, `SelectTrial`, `SettingText`, the folders of `NewTrial` (`cam<c>dir`),
//! and `CalibToggleInlier` (a double click on a calibrated image: this program's own
//! operation, the reference has none for that action).
//!
//! [`apply`] returns `None` for an operation it does not handle.

use std::path::PathBuf;

use xma_engine::calibration::CalibrationEdit;
use xma_engine::create::{CalibrationTarget, NewCameraSpec, NewProjectSpec};
use xma_engine::{Document, Workspace};
use xma_model::CppCompat;

use super::calib_ops::{F, apply_edit};
use super::calibration::{OpStatus, ReplayState, created};

/// `NewProject target=checkerboard`: the camera boxes and the board's three spin boxes.
#[derive(Clone, Debug, PartialEq)]
pub struct NewBoard {
    pub grids: Vec<Option<PathBuf>>,
    pub images: Vec<Vec<PathBuf>>,
    /// `light<c>`: "Light Camera".
    pub light: Vec<bool>,
    /// `flip<c>`: "Flip images horizontally".
    pub flip: Vec<bool>,
    pub horizontal: u32,
    pub vertical: u32,
    pub square_size: f64,
}

impl NewBoard {
    /// The checkerboard fields of a `NewProject` line (`get` reads a field).
    ///
    /// # Errors
    /// A missing or malformed field.
    pub fn parse(
        grids: Vec<Option<PathBuf>>,
        images: Vec<Vec<PathBuf>>,
        get: impl Fn(&str) -> Option<String>,
    ) -> Result<NewBoard, String> {
        let flag = |k: String| -> Result<bool, String> {
            match get(&k).as_deref() {
                None | Some("false" | "0") => Ok(false),
                Some("true" | "1") => Ok(true),
                Some(_) => Err(format!("field {k} is not true or false")),
            }
        };
        let num = |k: &str| -> Result<f64, String> {
            get(k)
                .ok_or_else(|| format!("missing field {k}"))?
                .parse::<f64>()
                .map_err(|_| format!("field {k} is not a number"))
        };
        let n = images.len();
        let light = (0..n)
            .map(|c| flag(format!("light{c}")))
            .collect::<Result<_, _>>()?;
        let flip = (0..n)
            .map(|c| flag(format!("flip{c}")))
            .collect::<Result<_, _>>()?;
        // `QSpinBox::setValue` (0 to 99) and `QDoubleSpinBox::setValue` (0 to 99.99, 2
        // decimals), as the driver types them.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let spin = |v: f64| v.round().clamp(0.0, 99.0) as u32;
        Ok(NewBoard {
            grids,
            images,
            light,
            flip,
            horizontal: spin(num("horizontal")?),
            vertical: spin(num("vertical")?),
            square_size: ((num("square_size")? * 100.0).round() / 100.0).clamp(0.0, 99.99),
        })
    }
}

/// File > New dataset with the Checkerboard target: cameras "Camera 1", ...; a light camera
/// has no grid, an X-ray camera without `grid<c>` its "Undistortion grid" box unchecked.
pub fn new_project(state: &mut ReplayState, b: &NewBoard) -> OpStatus {
    let spec = NewProjectSpec {
        cameras: b
            .images
            .iter()
            .enumerate()
            .map(|(c, im)| {
                let mut im = im.clone();
                // CameraBox::addCalibrationImage sorts the list (`QStringList::sort`).
                im.sort_by_key(|p| p.to_string_lossy().into_owned());
                let light = b.light.get(c).copied().unwrap_or(false);
                NewCameraSpec {
                    name: format!("Camera {}", c + 1),
                    images: im,
                    light_camera: light,
                    grid: if light {
                        None
                    } else {
                        b.grids.get(c).cloned().flatten()
                    },
                    flipped: b.flip.get(c).copied().unwrap_or(false),
                }
            })
            .collect(),
        target: CalibrationTarget::Checkerboard {
            horizontal_squares: b.horizontal,
            vertical_squares: b.vertical,
            square_size: b.square_size,
        },
    };
    let r = Document::new_project(&spec, &state.options);
    created(state, r)
}

/// The files of each camera of a `NewTrial`: a folder (`cam<c>dir`) gives its images in
/// natural order (`CameraBoxTrial::on_toolButtonImage_clicked`), a file itself.
#[must_use]
pub fn trial_files(cams: &[PathBuf], compat: CppCompat) -> Vec<Vec<PathBuf>> {
    cams.iter()
        .map(|c| {
            if c.is_dir() {
                xma_engine::undistort_sequence::folder_images(c, compat == CppCompat::XMALAB3)
            } else {
                vec![c.clone()]
            }
        })
        .collect()
}

const OPS: [&str; 7] = [
    "CalibToggleInlier",
    "CalibResetFrame",
    "CalibResetCamera",
    "CalibDeleteFrame",
    "CalibDistortion",
    "SelectTrial",
    "SettingText",
];

/// Applies an operation of this module; `None` for another one.
pub fn apply(state: &mut ReplayState, op: &str, fields: &[(String, String)]) -> Option<OpStatus> {
    if !OPS.contains(&op) {
        return None;
    }
    if op == "SettingText" {
        // Text settings (`WorkspacePath`, ...) are the interface's, not the model's.
        let names: Vec<&str> = fields.iter().map(|(k, _)| k.as_str()).collect();
        return Some(OpStatus::Ok(format!(
            "{} not held by the model",
            names.join(", ")
        )));
    }
    let Some(doc) = state.doc.as_mut() else {
        return Some(OpStatus::Invalid("no project is open".to_owned()));
    };
    Some(apply_to(doc, op, &F(fields)).unwrap_or_else(|e| e))
}

/// `State::changeActiveCamera` and `changeActiveFrameCalibration` when the line gives them.
fn select(doc: &mut Document, f: &F<'_>) -> Result<(), OpStatus> {
    if !f.has("camera") {
        return Ok(());
    }
    let camera: usize = f.n("camera")?;
    let Some(cam) = doc.project().cameras.get(camera) else {
        return Err(OpStatus::Invalid(format!("camera {camera} out of range")));
    };
    let image = if f.has("image") {
        f.n("image")?
    } else {
        doc.session().active_frame_calibration
    };
    if image >= cam.calibration_images.len() {
        return Err(OpStatus::Invalid(format!(
            "calibration image {image} out of range"
        )));
    }
    doc.set_active_calibration_image(camera, image);
    Ok(())
}

fn apply_to(doc: &mut Document, op: &str, f: &F<'_>) -> Result<OpStatus, OpStatus> {
    let mut notes = Vec::new();
    Ok(match op {
        "CalibToggleInlier" => {
            // GLCameraView::mouseDoubleClickEvent: the camera made active; a calibrated image
            // in the Calibration workspace.
            select(doc, f)?;
            let s = doc.session();
            let (camera, image) = (s.active_camera, s.active_frame_calibration);
            if s.workspace != Workspace::Calibration {
                return Err(OpStatus::Invalid(
                    "not in the Calibration workspace".to_owned(),
                ));
            }
            apply_edit(
                doc,
                CalibrationEdit::ToggleInlier {
                    camera,
                    image,
                    x: f.n("x")?,
                    y: f.n("y")?,
                    undistorted: f.has("undistorted") && f.b("undistorted")?,
                },
                &mut notes,
            )
        }
        "CalibResetFrame" | "CalibResetCamera" | "CalibDeleteFrame" => {
            // The page's buttons on the active camera and calibration image; the question
            // answered OK.
            select(doc, f)?;
            let s = doc.session();
            let (camera, image) = (s.active_camera, s.active_frame_calibration);
            let e = match op {
                "CalibResetFrame" => CalibrationEdit::ResetFrame { camera, image },
                "CalibResetCamera" => CalibrationEdit::ResetCamera { camera },
                _ => {
                    if s.n_calibration_images <= 1 {
                        // on_pushButtonDeleteFrame_clicked does nothing with one frame.
                        return Ok(OpStatus::Ok("one calibration frame: nothing".to_owned()));
                    }
                    CalibrationEdit::DeleteFrame { image }
                }
            };
            apply_edit(doc, e, &mut notes)
        }
        "CalibDistortion" => {
            // on_checkBoxDistortion_clicked; the box is shown only with
            // ShowAdvancedCalibration.
            select(doc, f)?;
            if !doc.session().wizard.show_advanced_calibration {
                return Err(OpStatus::Invalid(
                    "the model distortion check box is not shown (ShowAdvancedCalibration is off)"
                        .to_owned(),
                ));
            }
            let on = f.b("on")?;
            let camera = doc.session().active_camera;
            if doc.project().cameras[camera].has_model_distortion == on {
                return Ok(OpStatus::Ok(
                    "already in that state: nothing clicked".to_owned(),
                ));
            }
            apply_edit(
                doc,
                CalibrationEdit::SetLensModel { camera, on },
                &mut notes,
            )
        }
        "SelectTrial" => {
            let t: usize = f.n("trial")?;
            if t >= doc.project().trials.len() {
                return Err(OpStatus::Invalid("trial out of range".to_owned()));
            }
            doc.select_trial(t);
            OpStatus::Ok(String::new())
        }
        _ => unreachable!("checked in apply"),
    })
}
