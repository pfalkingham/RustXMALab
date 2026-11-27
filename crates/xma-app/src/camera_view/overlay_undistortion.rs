//! The Undistortion workspace's image choice, overlays and input.
//!
//! Ported from `ui/GLCameraView.cpp` (`drawTexture`, `paintGL` and `mousePressEvent` for
//! `UNDISTORTION`) and `core/UndistortionObject.cpp` (`bindTexture`, `drawData`,
//! `drawPoints`):
//!
//! | `undistortionVisImage` | Image |
//! |---|---|
//! | Distorted | the grid image |
//! | Undistorted | the grid image remapped with the fitted grid |
//! | Angle scaled, Angle, Distance, Horizontal, Vertical | the false-colour maps of `computeTmpImage` (HSV, HSV, JET, JET, JET) |
//!
//! Without a computed grid the undistorted image and the maps fall back to the grid image
//! (the C++ shows its unchanged copy).
//!
//! | `undistortionVisPoints` | Drawn |
//! |---|---|
//! | None | nothing |
//! | Detected | every detected bead, red ±2 px cross |
//! | Grid distorted, Grid undistorted, Reference | each matched bead, ±2 px cross, green (0, 204, 0) inlier, red (204, 0, 0) outlier; the first bead also ±5 px |
//!
//! The centre, when set, is a blue ±5 px ×, in every points mode.
//!
//! A left click (not while drawing is disabled) raises `CalibAction::UndistortionClick`
//! with the press position (`GLCameraView::mousePressEvent`): in mouse mode "toggle
//! outlier" it toggles the nearest point of the list shown (C6), in "set center" it sets
//! the centre on the distorted image (C7), each with XMALab's error for the other displays
//! (`app/calibration.rs`).

use std::sync::Arc;

use egui::{Color32, Painter};
use xma_engine::view::calibration::{UndistortionGridPoints, undistortion_grid};
use xma_model::Point2;
use xma_vision::remap::DisplacementMap;

use crate::actions::Action;
use crate::calibration_ui::CalibAction;
use crate::frames::Derived;
use crate::panels::Cx;
use crate::state::{UndistortionVisImage, UndistortionVisPoints};
use crate::widgets::image_view::{ImageViewEvent, ViewTransform};

const GREEN: Color32 = Color32::from_rgb(0, 204, 0);
const RED_D: Color32 = Color32::from_rgb(204, 0, 0);
const RED: Color32 = Color32::from_rgb(255, 0, 0);
const BLUE: Color32 = Color32::from_rgb(0, 0, 255);

/// The texture to show in place of the grid image, if any.
///
/// Ported from `core/UndistortionObject.cpp`, `UndistortionObject::bindTexture`.
#[must_use]
pub fn image_key(cx: &mut Cx<'_>, camera: usize) -> Option<egui::Id> {
    let doc = cx.doc?;
    let kind = match cx.state.undistortion_vis_image {
        UndistortionVisImage::Distorted => return None,
        UndistortionVisImage::Undistorted => Derived::UndistortedGrid,
        UndistortionVisImage::MapAngleScaled => Derived::Map(DisplacementMap::ScaledAngle),
        UndistortionVisImage::MapAngles => Derived::Map(DisplacementMap::Angle),
        UndistortionVisImage::MapDistance => Derived::Map(DisplacementMap::Length),
        UndistortionVisImage::MapHorizontal => Derived::Map(DisplacementMap::Horizontal),
        UndistortionVisImage::MapVertical => Derived::Map(DisplacementMap::Vertical),
    };
    cx.frames.derived(doc, camera, kind)
}

/// The grid of `camera`, computed once per project.
pub fn grid(cx: &mut Cx<'_>, camera: usize) -> Option<Arc<UndistortionGridPoints>> {
    if let Some(g) = cx.frames.views.grids.get(&camera) {
        return Some(g.clone());
    }
    let g = Arc::new(undistortion_grid(cx.doc?, camera)?);
    cx.frames.views.grids.insert(camera, g.clone());
    Some(g)
}

/// Draws the undistortion overlays of `camera` over its image.
pub fn draw(painter: &Painter, t: &ViewTransform, cx: &mut Cx<'_>, camera: usize) {
    let mode = cx.state.undistortion_vis_points;
    let Some(g) = grid(cx, camera) else { return };
    let pen = super::Pen::new(painter, t);
    let cross = |p: &Point2, r: f64, c: Color32| pen.cross((p.x, p.y), r, c);
    // UndistortionObject::drawPoints.
    let points = |pts: &[Point2]| {
        for (p, inl) in pts.iter().zip(&g.inlier) {
            cross(p, 2.0, if *inl { GREEN } else { RED_D });
        }
        if let (Some(p), Some(inl)) = (pts.first(), g.inlier.first()) {
            cross(p, 5.0, if *inl { GREEN } else { RED_D });
        }
    };
    match mode {
        UndistortionVisPoints::None => {}
        UndistortionVisPoints::Detected => {
            for p in &g.detected {
                cross(p, 2.0, RED);
            }
        }
        UndistortionVisPoints::GridDistorted => points(&g.grid_distorted),
        UndistortionVisPoints::GridUndistorted => points(&g.grid_undistorted),
        UndistortionVisPoints::Reference => points(&g.references),
    }
    if let Some(c) = g.center {
        pen.ex((c.x, c.y), BLUE);
    }
}

/// Handles one input event of `camera`'s view in the Undistortion workspace (a camera with
/// a grid): a left click raises [`CalibAction::UndistortionClick`] (C6, C7).
///
/// Ported from ui/GLCameraView.cpp, GLCameraView::mousePressEvent (`UNDISTORTION`).
pub fn on_event(cx: &mut Cx<'_>, camera: usize, event: &ImageViewEvent) {
    if let ImageViewEvent::Pressed {
        button: egui::PointerButton::Primary,
        pos,
        ..
    } = event
        && !cx.state.disable_draw
    {
        cx.raise(Action::Calib(CalibAction::UndistortionClick {
            camera,
            x: pos.x,
            y: pos.y,
        }));
    }
}
