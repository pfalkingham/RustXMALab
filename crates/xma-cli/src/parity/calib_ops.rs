//! The undistortion and cube-calibration operations of a calibration script (`oracle/FORMAT.md`,
//! "Calibration runs"), applied as the C++ driver applies them
//! (`oracle/xmalab-cpp/src/ui/CalibOracleDriver.cpp`, `runOp`): `UndistortionButton`,
//! `UndistToggleOutlier`, `UndistSetCenter`, `UndistRemoveOutliers`, `ClickReference`,
//! `CalibMovePoint`, `CalibrationButton`, `ManualMode`, `ManualCheckbox` (the cube's and the
//! checkerboard's; the operations of format 6 are in [`super::checker_ops`]). They arrive as
//! [`super::calibration::CalibOp::Later`]; [`apply`] returns `None` for an operation it does
//! not handle.
//!
//! Each operation is what the driver does around the handler (the active camera and
//! calibration image, the click's snapping) and the engine's edit or session function for the
//! handler. The edits are applied as one undo step each, as the window applies them.

use xma_engine::calibration::{CalibrationEdit, GridPointList};
use xma_engine::{Document, Edit, EditError, Workspace};

use super::calibration::{OpStatus, ReplayState};

/// The fields of one operation, as written.
pub(super) struct F<'a>(pub(super) &'a [(String, String)]);

impl F<'_> {
    pub(super) fn s(&self, k: &str) -> Result<&str, OpStatus> {
        self.0
            .iter()
            .find(|(n, _)| n == k)
            .map(|(_, v)| v.as_str())
            .ok_or_else(|| OpStatus::Invalid(format!("missing field {k}")))
    }
    pub(super) fn has(&self, k: &str) -> bool {
        self.0.iter().any(|(n, _)| n == k)
    }
    pub(super) fn n<T: std::str::FromStr>(&self, k: &str) -> Result<T, OpStatus> {
        self.s(k)?
            .parse()
            .map_err(|_| OpStatus::Invalid(format!("field {k} is not a number")))
    }
    pub(super) fn b(&self, k: &str) -> Result<bool, OpStatus> {
        match self.s(k)? {
            "true" | "1" => Ok(true),
            "false" | "0" => Ok(false),
            _ => Err(OpStatus::Invalid(format!("field {k} is not true or false"))),
        }
    }
}

pub(super) fn edit_status(
    r: Result<xma_engine::EditOutcome, EditError>,
    notes: &mut Vec<String>,
) -> OpStatus {
    match r {
        Ok(o) => {
            notes.extend(o.messages);
            match o.refused {
                Some(m) => OpStatus::Rejected(m),
                None => OpStatus::Ok(notes.join(" | ")),
            }
        }
        Err(EditError::Rejected(m)) => OpStatus::Rejected(m),
        Err(EditError::NotImplemented(w)) => OpStatus::Invalid(format!("{w} is not ported")),
        Err(e) => OpStatus::Invalid(e.to_string()),
    }
}

pub(super) fn apply_edit(
    doc: &mut Document,
    e: CalibrationEdit,
    notes: &mut Vec<String>,
) -> OpStatus {
    let r = doc.apply(&Edit::Calibration(e));
    edit_status(r, notes)
}

/// Applies an operation of this package; `None` for another one.
pub fn apply(state: &mut ReplayState, op: &str, fields: &[(String, String)]) -> Option<OpStatus> {
    let known = [
        "UndistortionButton",
        "UndistToggleOutlier",
        "UndistSetCenter",
        "UndistRemoveOutliers",
        "ClickReference",
        "CalibMovePoint",
        "CalibrationButton",
        "ManualMode",
        "ManualCheckbox",
    ];
    if !known.contains(&op) {
        return None;
    }
    let Some(doc) = state.doc.as_mut() else {
        return Some(OpStatus::Invalid("no project is open".to_owned()));
    };
    Some(apply_to(doc, op, &F(fields)).unwrap_or_else(|e| e))
}

fn apply_to(doc: &mut Document, op: &str, f: &F<'_>) -> Result<OpStatus, OpStatus> {
    let mut notes = Vec::new();
    let camera_ok = |doc: &Document, c: usize| {
        if c < doc.project().cameras.len() {
            Ok(())
        } else {
            Err(OpStatus::Invalid(format!("camera {c} out of range")))
        }
    };
    Ok(match op {
        "UndistortionButton" => apply_edit(doc, CalibrationEdit::ComputeUndistortion, &mut notes),
        "UndistToggleOutlier" => {
            let camera: usize = f.n("camera")?;
            camera_ok(doc, camera)?;
            let list = match f.n::<i32>("points")? {
                2 => GridPointList::Distorted,
                3 => GridPointList::Undistorted,
                4 => GridPointList::References,
                _ => {
                    return Ok(OpStatus::Rejected(
                        "You either have to display the grid points (distorted or undistorted) or the reference points to toggle outlier".to_owned(),
                    ));
                }
            };
            // GLCameraView::mousePressEvent: State::changeActiveCamera.
            doc.set_active_calibration_image(camera, doc.session().active_frame_calibration);
            apply_edit(
                doc,
                CalibrationEdit::ToggleGridOutlier {
                    camera,
                    list,
                    x: f.n("x")?,
                    y: f.n("y")?,
                },
                &mut notes,
            )
        }
        "UndistSetCenter" => {
            let camera: usize = f.n("camera")?;
            camera_ok(doc, camera)?;
            doc.set_active_calibration_image(camera, doc.session().active_frame_calibration);
            apply_edit(
                doc,
                CalibrationEdit::SetGridCenter {
                    camera,
                    x: f.n("x")?,
                    y: f.n("y")?,
                },
                &mut notes,
            )
        }
        "UndistRemoveOutliers" => apply_edit(
            doc,
            CalibrationEdit::RemoveGridOutliers {
                border: f.n("border")?,
                circle: f.n("circle")?,
            },
            &mut notes,
        ),
        "ClickReference" | "CalibMovePoint" => {
            let camera: usize = f.n("camera")?;
            let image: usize = f.n("image")?;
            camera_ok(doc, camera)?;
            if image >= doc.project().cameras[camera].calibration_images.len() {
                return Err(OpStatus::Invalid(format!(
                    "calibration image {image} out of range"
                )));
            }
            doc.set_active_calibration_image(camera, image);
            if doc.session().workspace != Workspace::Calibration {
                return Err(OpStatus::Invalid(
                    "not in the Calibration workspace".to_owned(),
                ));
            }
            let (mut x, mut y): (f64, f64) = (f.n("x")?, f.n("y")?);
            if f.has("alt") && f.b("alt")? {
                let (sx, sy) = doc.snap_calibration_click(camera, image, x, y);
                if (sx, sy) != (x, y) || doc.settings().detection_method_for_calibration.code() > 0
                {
                    notes.push(format!("snapped ({x}; {y}) to ({sx}; {sy})"));
                }
                x = sx;
                y = sy;
            }
            let ctrl = op == "CalibMovePoint";
            if doc.session().manual_mode {
                let point = doc.session().manual_point;
                apply_edit(
                    doc,
                    CalibrationEdit::SetManualPoint {
                        camera,
                        image,
                        point,
                        x,
                        y,
                        undistorted: false,
                    },
                    &mut notes,
                )
            } else if doc.project().cameras[camera].calibration_images[image].calibrated != 1 {
                // WizardCalibrationCubeFrame::addCalibrationReference: only once undistorted.
                if !doc.session().undistorted {
                    return Ok(OpStatus::Ok(
                        "not undistorted: the click is ignored".to_owned(),
                    ));
                }
                // A cube's next reference, or a checkerboard's interior upper-left corner
                // (automatic) or next corner (manual).
                if doc.click_calibration_reference(x, y) {
                    apply_edit(
                        doc,
                        CalibrationEdit::ComputeCalibration { camera, image },
                        &mut notes,
                    )
                } else {
                    OpStatus::Ok(notes.join(" | "))
                }
            } else if ctrl {
                apply_edit(
                    doc,
                    CalibrationEdit::MovePoint {
                        camera,
                        image,
                        x,
                        y,
                        undistorted: false,
                    },
                    &mut notes,
                )
            } else {
                OpStatus::Ok("click on a calibrated image without Ctrl: nothing happens".to_owned())
            }
        }
        "CalibrationButton" => {
            if f.has("camera") {
                let camera: usize = f.n("camera")?;
                let image: usize = f.n("image")?;
                camera_ok(doc, camera)?;
                doc.set_active_calibration_image(camera, image);
            }
            let s = doc.session();
            let (camera, image) = (s.active_camera, s.active_frame_calibration);
            apply_edit(
                doc,
                CalibrationEdit::ComputeCalibration { camera, image },
                &mut notes,
            )
        }
        "ManualMode" => {
            let on = f.b("on")?;
            if doc.session().manual_mode != on {
                doc.set_manual_mode(on);
            }
            OpStatus::Ok(String::new())
        }
        "ManualCheckbox" => {
            if !doc.session().manual_mode {
                return Err(OpStatus::Invalid("manual mode is off".to_owned()));
            }
            let index: usize = f.n("index")?;
            let s = doc.session();
            let (camera, image) = (s.active_camera, s.active_frame_calibration);
            let im = &doc.project().cameras[camera].calibration_images[image];
            // The check boxes show the inliers as reloadManualPoints set them.
            let mut boxes: Vec<bool> = im.inlier.iter().map(|&v| v > 0).collect();
            let Some(b) = boxes.get_mut(index) else {
                return Err(OpStatus::Invalid("no check box for that point".to_owned()));
            };
            *b = !*b;
            apply_edit(
                doc,
                CalibrationEdit::SetManualInliers {
                    camera,
                    image,
                    inliers: boxes,
                },
                &mut notes,
            )
        }
        _ => unreachable!("checked in apply"),
    })
}
