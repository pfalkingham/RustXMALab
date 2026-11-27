//! Drawing the marker-tracking overlays of a camera view or a detail view.
//!
//! The geometry comes from [`xma_engine::view::overlays::tracking_overlay`]; this module
//! paints it with egui's painter in image coordinates through the view transform, in the
//! C++ drawing order and colours (`core/Trial.cpp`, `Trial::drawPoints`;
//! `core/RigidBody.cpp`, `RigidBody::draw2D`; `ui/GLCameraView.cpp`, `renderPointText`):
//!
//! | Item | Colour | Shown when |
//! |---|---|---|
//! | Marker cross, ±5 px | red active, green others; status colour with `ShowColoredMarkerCross` | `TrialDrawMarkers`, main view |
//! | Crosshair, ±12 px (+ ticks, size box with `AdvancedCrosshairDetailView`) | red; box red at 30 % | `TrialDrawMarkers`, detail view |
//! | Projected 3D point, × ±5 px | cyan | `TrialDrawMarkers`; active marker (detail: `Show3dPointDetailView`); all markers with `DrawProjected2DpositionsForAllPoints` (main view) |
//! | Epipolar lines of the active marker | blue | `TrialDrawEpipolar` (detail: and `ShowEpiLineDetailView`) |
//! | Rigid-body constellations | body colour; virtual points stippled | `TrialDrawRigidBodyConstellation`, main view; filtered poses with `TrialDrawFiltered` |
//! | Marker numbers, 15 pt | red active, green others; status colour with `ShowColoredMarkerIDs` | `TrialDrawMarkerIds` (detail: and `ShowIDsInDetail`) |
//! | "Rendering of filtered data is enabled, ..." | red, 12 pt | `TrialDrawFiltered` and the trial's frame rate or cutoff not set |
//!
//! `TrialDrawHideAll` hides all of it except the filtered-data warning. Lines are one
//! device pixel wide, as OpenGL draws them.

use egui::{Color32, Painter, Pos2, Shape, Stroke};
use xma_engine::view::overlays::{
    FILTERED_WARNING, MarkColor, TrackingOptions, ViewTarget, tracking_overlay,
};
use xma_model::{MarkerStatus, Point2};

use crate::panels::Cx;
use crate::settings::{UiSettings, parse_color};
use crate::widgets::image_view::ViewTransform;

const RED: Color32 = Color32::from_rgb(255, 0, 0);
const GREEN: Color32 = Color32::from_rgb(0, 255, 0);
const CYAN: Color32 = Color32::from_rgb(0, 255, 255);
const BLUE: Color32 = Color32::from_rgb(0, 0, 255);

/// The overlay switches from the settings.
#[must_use]
pub fn options(s: &UiSettings) -> TrackingOptions {
    TrackingOptions {
        hide_all: s.trial_draw_hide_all,
        markers: s.trial_draw_markers,
        marker_ids: s.trial_draw_marker_ids,
        epipolar: s.trial_draw_epipolar,
        constellations: s.trial_draw_rigid_body_constellation,
        filtered: s.trial_draw_filtered,
        colored_cross: s.show_colored_marker_cross,
        colored_ids: s.show_colored_marker_ids,
        projected_all: s.draw_projected_2d_positions_for_all_points,
        detail_3d_point: s.show_3d_point_detail_view,
        detail_epipolar: s.show_epi_line_detail_view,
        detail_ids: s.show_ids_in_detail,
        detail_advanced_crosshair: s.advanced_crosshair_detail_view,
        epipolar_precision: s.epipolar_line_precision,
        xmalab3: false,
    }
}

/// The colour setting of a status.
///
/// Ported from `core/Marker.cpp`, `Marker::getStatusColor`.
#[must_use]
pub fn status_color(s: &UiSettings, status: MarkerStatus) -> Color32 {
    let c = &s.colors;
    parse_color(match status {
        MarkerStatus::Untrackable => &c.untrackable,
        MarkerStatus::Deleted | MarkerStatus::Lost | MarkerStatus::Undefined => &c.undefined,
        MarkerStatus::Predicted => &c.predicted,
        MarkerStatus::Interpolated => &c.interpolated,
        MarkerStatus::Tracked => &c.tracked,
        MarkerStatus::TrackedAndOptimized => &c.tracked_and_opt,
        MarkerStatus::Set => &c.set,
        MarkerStatus::SetAndOptimized => &c.set_and_opt,
        MarkerStatus::Manual => &c.manual,
        MarkerStatus::ManualAndOptimized => &c.manual_and_opt,
    })
}

fn mark_color(s: &UiSettings, c: MarkColor) -> Color32 {
    match c {
        MarkColor::Active => RED,
        MarkColor::Other => GREEN,
        MarkColor::Status(st) => status_color(s, st),
    }
}

/// Draws the overlays of `camera` (a detail view when `detail`).
pub fn draw(painter: &Painter, t: &ViewTransform, cx: &Cx<'_>, camera: usize, detail: bool) {
    let Some(doc) = cx.doc else { return };
    let (Some(trial), Some(view)) = (cx.state.active_trial, cx.state.trial_view()) else {
        return;
    };
    let target = ViewTarget {
        trial,
        camera,
        frame: view.active_frame,
        active_marker: view.active_marker,
        detail,
    };
    let o = tracking_overlay(doc, target, &options(cx.settings));
    let ppp = t.pixels_per_point;
    let px = 1.0 / ppp;
    let at = |p: Point2| t.image_to_screen((p.x, p.y));
    let pen = super::Pen::new(painter, t);
    let seg = |a: (f64, f64), b: (f64, f64), color: Color32| pen.seg(a, b, color);
    let cross = |p: Point2, r: f64, color: Color32| pen.cross((p.x, p.y), r, color);
    let ex = |p: Point2, color: Color32| pen.ex((p.x, p.y), color);

    for c in &o.crosses {
        cross(c.at, 5.0, mark_color(cx.settings, c.color));
    }
    if let Some(h) = o.crosshair {
        let (x, y) = (h.at.x, h.at.y);
        cross(h.at, 12.0, RED);
        if h.advanced {
            for i in 0..6 {
                let d = f64::from(i) * 2.0;
                seg((x - 1.0, y + d), (x + 1.0, y + d), RED);
                seg((x - 1.0, y - d), (x + 1.0, y - d), RED);
                seg((x + d, y - 1.0), (x + d, y + 1.0), RED);
                seg((x - d, y - 1.0), (x - d, y + 1.0), RED);
            }
            if let Some(s) = h.size {
                let c = Color32::from_rgba_unmultiplied(255, 0, 0, 77);
                seg((x - s, y - s), (x - s, y + s), c);
                seg((x + s, y - s), (x + s, y + s), c);
                seg((x - s, y - s), (x + s, y - s), c);
                seg((x - s, y + s), (x + s, y + s), c);
            }
        }
    }
    for (_, p) in &o.projected {
        ex(*p, CYAN);
    }
    for line in &o.epipolar {
        if line.points.len() >= 2 {
            let pts: Vec<Pos2> = line.points.iter().map(|p| at(*p)).collect();
            painter.add(Shape::line(pts, Stroke::new(px, BLUE)));
        }
    }
    for c in &o.constellations {
        let color = Color32::from_rgb(c.color.r, c.color.g, c.color.b);
        for [a, b] in &c.lines {
            painter.line_segment([at(*a), at(*b)], Stroke::new(px, color));
        }
        // glLineStipple(10, 0xAAAA): 10 device pixels on, 10 off.
        for [a, b] in &c.virtual_lines {
            painter.extend(Shape::dashed_line(
                &[at(*a), at(*b)],
                Stroke::new(px, color),
                10.0 * px,
                10.0 * px,
            ));
        }
        for p in &c.virtual_crosses {
            cross(*p, 5.0, color);
        }
    }
    let font = super::qt_font(15.0);
    for l in &o.labels {
        // renderText(x + 3, y): the text's baseline at the point.
        painter.text(
            at(l.at),
            egui::Align2::LEFT_BOTTOM,
            &l.text,
            font.clone(),
            mark_color(cx.settings, l.color),
        );
    }
    if o.filtered_warning {
        let r = t.view_rect();
        painter.text(
            egui::pos2(r.center().x, r.max.y - 4.0),
            egui::Align2::CENTER_BOTTOM,
            FILTERED_WARNING,
            super::qt_font(12.0),
            crate::theme::VIEW_MESSAGE,
        );
    }
}
