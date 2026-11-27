//! The Calibration workspace's image choice, overlays and input (internal calibration).
//!
//! Ported from `ui/GLCameraView.cpp` (`drawTexture`, `paintGL`, `renderPointText(true)` for
//! `CALIBRATION`), `core/CalibrationImage.cpp` (`bindTexture`, `draw`, `drawPoints`,
//! `getDrawTextData`) and `ui/WizardCalibrationCubeFrame.cpp` (`draw`, in the active camera
//! only through `WizardDockWidget::draw`).
//!
//! Image (`calibrationVisImage`): the calibration image; "undistorted" shows it remapped
//! with the grid and the lens model when the image is calibrated (and the camera has a
//! computed grid or a lens model), as `CalibrationImage::bindTexture`; a frame of a
//! calibration sequence the same way (`CalibrationSequence::bindTexture`).
//!
//! | `calibrationVisPoints` | Drawn |
//! |---|---|
//! | None | nothing |
//! | Detected all | every detected point, red ±2 px cross |
//! | Cube detected (distorted / undistorted) | the matched detection of each frame point, ±5 px cross: green inlier, red outlier; undetected points not drawn |
//! | Cube projected (distorted / undistorted) | each frame point projected with the image's pose, ±5 px: green inlier, red outlier, blue (0, 0, 255) undetected |
//!
//! | `calibrationVisText` | Text at (x + 3, y), 15 pt, at the drawn crosses (projected ones in a projected mode, else detected), raw or undistorted as the image mode |
//! |---|---|
//! | None | nothing |
//! | Id | the point's number from 1 |
//! | Error | its error with 2 decimals (empty for undetected points) |
//! | Error colour | the error coloured from green (0) to red (largest) |
//! | Marker name | the frame's bead name |
//!
//! The text colour (except "error colour") is green inlier, (80, 140, 255) undetected, red
//! outlier. The cube calibration's four reference points are red ±5 px × in the active
//! camera (the engine's session, `Session::references`).
//!
//! Input (`GLCameraView::mousePressEvent`, `mouseDoubleClickEvent`): a left click raises
//! `CalibAction::CalibrationClick` with Ctrl (Command on macOS) and Alt (C15, C16, C19,
//! C24, C27); a double click `CalibAction::DoubleClick` (C23), placed with the press
//! formula (ledger U3). What each does is in `app/calibration.rs`.

use std::sync::Arc;

use egui::{Align2, Color32, Painter};
use xma_engine::view::calibration::{CalibrationImagePoints, calibration_image};
use xma_model::Point2;

use crate::actions::Action;
use crate::calibration_ui::CalibAction;
use crate::frames::Derived;
use crate::panels::Cx;
use crate::state::{CalibrationVisImage, CalibrationVisPoints, CalibrationVisText};
use crate::widgets::image_view::{ImageViewEvent, ViewTransform};

const GREEN: Color32 = Color32::from_rgb(0, 255, 0);
const RED: Color32 = Color32::from_rgb(255, 0, 0);
const BLUE: Color32 = Color32::from_rgb(0, 0, 255);
const LIGHT_BLUE: Color32 = Color32::from_rgb(80, 140, 255);

/// The texture to show in place of the distorted calibration image, if any.
///
/// Ported from `core/CalibrationImage.cpp`, `CalibrationImage::bindTexture`.
#[must_use]
pub fn image_key(cx: &mut Cx<'_>, camera: usize) -> Option<egui::Id> {
    let doc = cx.doc?;
    if cx.state.calibration_vis_image != CalibrationVisImage::Undistorted {
        return None;
    }
    let i = cx.state.active_frame_calibration.unwrap_or(0);
    let cam = doc.project().cameras.get(camera)?;
    // A frame of a calibration sequence is undistorted like a still
    // (`CalibrationSequence::bindTexture`).
    if cam.calibration_images.get(i)?.calibrated <= 0 {
        return None;
    }
    cx.frames
        .derived(doc, camera, Derived::UndistortedCalibration(i))
}

/// The points of the active calibration image of `camera`, computed once per project.
pub fn points(cx: &mut Cx<'_>, camera: usize) -> Option<Arc<CalibrationImagePoints>> {
    let i = cx.state.active_frame_calibration.unwrap_or(0);
    if let Some(p) = cx.frames.views.calibration_points.get(&(camera, i)) {
        return Some(p.clone());
    }
    let p = Arc::new(calibration_image(cx.doc?, camera, i)?);
    cx.frames
        .views
        .calibration_points
        .insert((camera, i), p.clone());
    Some(p)
}

/// Draws the calibration overlays of `camera` over its image.
pub fn draw(painter: &Painter, t: &ViewTransform, cx: &mut Cx<'_>, camera: usize) {
    let Some(p) = points(cx, camera) else { return };
    let pen = super::Pen::new(painter, t);
    let cross = |q: &Point2, r: f64, c: Color32| pen.cross((q.x, q.y), r, c);
    // CalibrationImage::drawPoints.
    let cube = |pts: &[Point2], draw_all: bool| {
        if pts.len() != p.inlier.len() {
            return;
        }
        for (q, &inl) in pts.iter().zip(&p.inlier) {
            let c = match inl {
                1 => GREEN,
                0 => RED,
                -1 => {
                    if !draw_all {
                        continue;
                    }
                    BLUE
                }
                // Other stored values keep the previous colour in the C++; drawn red here.
                _ => RED,
            };
            cross(q, 5.0, c);
        }
    };
    let mode = cx.state.calibration_vis_points;
    match mode {
        CalibrationVisPoints::None => {}
        CalibrationVisPoints::DetectedAll => {
            for q in &p.detected_all {
                cross(q, 2.0, RED);
            }
        }
        CalibrationVisPoints::DetectedCubeDistorted => cube(&p.detected, false),
        CalibrationVisPoints::ProjectedCubeDistorted => cube(&p.projected, true),
        CalibrationVisPoints::DetectedCubeUndistorted => cube(&p.detected_undistorted, false),
        CalibrationVisPoints::ProjectedCubeUndistorted => cube(&p.projected_undistorted, true),
    }
    let text = cx.state.calibration_vis_text;
    if text != CalibrationVisText::None {
        let distorted = cx.state.calibration_vis_image == CalibrationVisImage::Distorted;
        let at_projected = matches!(
            mode,
            CalibrationVisPoints::ProjectedCubeDistorted
                | CalibrationVisPoints::ProjectedCubeUndistorted
        );
        // The cube's bead names or the checkerboard's "x , y" (`getMarkerNames`).
        let names = cx
            .doc
            .and_then(|d| d.project().calibration_object.as_ref())
            .map(xma_model::CalibrationObject::point_names)
            .unwrap_or_default();
        let labels = text_data(&p, text, distorted, at_projected, &names);
        let font = super::qt_font(15.0);
        if text == CalibrationVisText::ErrorColor {
            let max = labels
                .iter()
                .filter_map(|l| l.2.parse::<f64>().ok())
                .fold(0.0, f64::max);
            let max = if max <= 0.0 { 1.0 } else { max };
            for (q, _, s) in &labels {
                if s.is_empty() {
                    continue;
                }
                let v = s.parse::<f64>().unwrap_or(0.0);
                let c =
                    Color32::from_rgb((255.0 * v / max) as u8, (255.0 * (1.0 - v / max)) as u8, 0);
                painter.text(
                    t.image_to_screen((q.x + 3.0, q.y)),
                    Align2::LEFT_BOTTOM,
                    s,
                    font.clone(),
                    c,
                );
            }
        } else {
            for (q, inl, s) in &labels {
                let c = match inl {
                    1 => GREEN,
                    i if *i < 0 => LIGHT_BLUE,
                    _ => RED,
                };
                painter.text(
                    t.image_to_screen((q.x + 3.0, q.y)),
                    Align2::LEFT_BOTTOM,
                    s,
                    font.clone(),
                    c,
                );
            }
        }
    }
    // WizardCalibrationCubeFrame::draw, active camera only.
    if cx.state.active_camera == Some(camera)
        && let Some(doc) = cx.doc
    {
        for r in doc.session().references.iter().filter_map(|r| r.position) {
            if r.0 > -1.0 && r.1 > -1.0 {
                pen.ex((r.0, r.1), RED);
            }
        }
    }
}

/// `(position, inlier flag, text)` per label.
///
/// Ported from `core/CalibrationImage.cpp`, `CalibrationImage::getDrawTextData`.
#[must_use]
pub fn text_data(
    p: &CalibrationImagePoints,
    text: CalibrationVisText,
    distorted: bool,
    at_projected: bool,
    names: &[String],
) -> Vec<(Point2, i32, String)> {
    let (projected, detected, err) = if distorted {
        (&p.projected, &p.detected, &p.error)
    } else {
        (
            &p.projected_undistorted,
            &p.detected_undistorted,
            &p.error_undistorted,
        )
    };
    let n = p.inlier.len();
    if detected.len() != n || (!projected.is_empty() && projected.len() != n) {
        return Vec::new();
    }
    // An image that is not calibrated has no projection: every projected point is unset,
    // as the C++ initialises them.
    let unset = Point2::new(-1.0, -1.0);
    let mut out = Vec::new();
    for i in 0..n {
        let inl = p.inlier[i];
        if !at_projected && inl < 0 {
            continue;
        }
        let mut q = if at_projected {
            projected.get(i).copied().unwrap_or(unset)
        } else {
            detected[i]
        };
        if q == unset && at_projected {
            q = detected[i];
        }
        if q == unset {
            continue;
        }
        let s = match text {
            CalibrationVisText::Error | CalibrationVisText::ErrorColor => {
                if inl >= 0 && i < err.len() {
                    format!("{:.2}", err[i])
                } else {
                    String::new()
                }
            }
            CalibrationVisText::MarkerName if i < names.len() => names[i].clone(),
            _ => (i + 1).to_string(),
        };
        out.push((q, inl, s));
    }
    out
}

/// Handles one input event of `camera`'s view in the Calibration workspace: a left click
/// raises [`CalibAction::CalibrationClick`], a double click [`CalibAction::DoubleClick`].
///
/// Ported from ui/GLCameraView.cpp, GLCameraView::mousePressEvent and
/// mouseDoubleClickEvent (`CALIBRATION`).
pub fn on_event(cx: &mut Cx<'_>, camera: usize, event: &ImageViewEvent) {
    match *event {
        ImageViewEvent::Pressed {
            button: egui::PointerButton::Primary,
            modifiers,
            pos,
        } => {
            if cx.state.disable_draw {
                return;
            }
            cx.raise(Action::Calib(CalibAction::CalibrationClick {
                camera,
                x: pos.x,
                y: pos.y,
                ctrl: modifiers.command,
                alt: modifiers.alt,
            }));
        }
        // U3 fixed: the press formula (XMALab leaves out its -0.5 here).
        ImageViewEvent::DoubleClicked {
            button: egui::PointerButton::Primary,
            pos,
            ..
        } => {
            cx.raise(Action::Calib(CalibAction::DoubleClick {
                camera,
                x: pos.x,
                y: pos.y,
            }));
        }
        _ => {}
    }
}
