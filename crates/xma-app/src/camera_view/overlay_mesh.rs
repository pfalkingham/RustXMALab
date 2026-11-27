//! The rigid-body mesh overlay of a camera view in Marker tracking.
//!
//! Ported from `ui/GLCameraView.cpp`, `paintGL` (lines 720 to 860: `renderMeshes`, the
//! off-screen buffer, `Trial::drawRigidBodiesMesh`, the distortion and blend shaders). The
//! rule ([`xma_engine::view::world::mesh_overlay_shown`]): Marker tracking, a camera view
//! (not the Detailed View), Display Options "Meshmodels" on, "Hide all" off, a calibrated
//! project, not the Default trial, and a body with "Draw bone mesh model" on. Each such
//! body with a mesh that loads and a pose at the frame
//! ([`xma_engine::view::world::meshes_to_draw`]) is drawn in its colour through the
//! camera's pose for the trial's reference calibration image, distorted into the raw image
//! ([`crate::widgets::scene3d`]), blended over the image with the view's transparency
//! (info frame slider, Ctrl + wheel) and, with "Draw transparent bone models", multiplied
//! together without depth test. It is drawn after the image and before the markers, as in
//! XMALab.
//!
//! While a camera's distortion map is being computed the mesh is drawn undistorted with
//! XMALab's red warning. Differences from XMALab (parity ledger U134 to U136): the image's
//! scale and bias do not change the mesh's colours (XMALab applies them after the mesh;
//! U136, kept by the author's decision of 16 November 2026); the warning is not shown for a
//! camera without distortion; the far clipping plane is not at 1000 units.

use xma_engine::view::world::{
    MESH_NOT_DISTORTED, OVERLAY_FAR, OVERLAY_NEAR, OverlayOptions, body_to_camera,
    mesh_overlay_shown, meshes_to_draw, trial_camera_pose,
};

use crate::meshes::{MapState, MeshState, OverlayDrawn};
use crate::panels::Cx;
use crate::widgets::image_view::ViewTransform;
use crate::widgets::scene3d::{self, GpuTarget, MeshInstance, OverlayDraw, mat4};

/// Draws the meshes of `camera` over its image (main camera view, Marker tracking).
pub fn draw(painter: &egui::Painter, t: &ViewTransform, cx: &mut Cx<'_>, camera: usize) {
    cx.frames.views.meshes.overlay.remove(&camera);
    let Some(doc) = cx.doc else { return };
    let (Some(trial_i), Some(view)) = (cx.state.active_trial, cx.state.trial_view()) else {
        return;
    };
    let frame = view.active_frame;
    let options = OverlayOptions {
        meshmodels: cx.settings.trial_draw_rigid_body_meshmodels,
        hide_all: cx.settings.trial_draw_hide_all,
        detail: false,
        marker_tracking: true,
    };
    if !mesh_overlay_shown(doc, trial_i, options) {
        return;
    }
    let trial = &doc.project().trials[trial_i];
    let Some(pose) = trial_camera_pose(doc, trial, camera) else {
        return;
    };
    let Some(cam) = doc.project().cameras.get(camera) else {
        return;
    };
    cx.frames.views.meshes.poll(painter.ctx(), cx.log);
    let model = &doc.camera(camera).model;
    let k = model.intrinsics.camera_matrix;
    let mut meshes = Vec::new();
    let mut drawn = OverlayDrawn {
        frame,
        ..OverlayDrawn::default()
    };
    for m in meshes_to_draw(trial, frame, cx.settings.trial_draw_filtered) {
        let MeshState::Ready(obj) = cx.frames.views.meshes.mesh(&m.path) else {
            continue;
        };
        let (r, tr) = body_to_camera(&m, &pose);
        drawn.meshes += 1;
        drawn.vertices += obj.n_vertices();
        drawn.triangles += obj.n_triangles();
        meshes.push(MeshInstance {
            mesh: obj,
            model: body_model_matrix(&r, &tr, m.scale),
            color: m.color,
        });
    }
    let shader_k = scene3d::overlay_camera_matrix(&k);
    drawn.camera_matrix = shader_k;
    let state = &cx.frames.views.cameras[camera];
    drawn.transparency = state.transparency;
    drawn.transparent_models = state.transparent_models;
    let size = [cam.width.max(1), cam.height.max(1)];
    let map = match cx.frames.views.meshes.map(camera, model, size) {
        MapState::Identity => None,
        MapState::Computing => {
            drawn.warning = true;
            None
        }
        MapState::Ready(m) => {
            drawn.distorted = true;
            Some(m)
        }
    };
    if drawn.warning {
        // renderText at the bottom centre of the view, red, 12 points (drawn one line above
        // the filtered-data warning, which XMALab writes at the same place).
        let r = t.view_rect();
        painter.text(
            egui::pos2(r.center().x, r.max.y - 24.0),
            egui::Align2::CENTER_BOTTOM,
            MESH_NOT_DISTORTED,
            super::qt_font(12.0),
            crate::theme::VIEW_MESSAGE,
        );
    }
    if !meshes.is_empty()
        && let Some(store) = cx.frames.store()
    {
        let target = GpuTarget::from_store(store);
        let image_rect = egui::Rect::from_two_pos(
            t.image_to_screen((-0.5, -0.5)),
            t.image_to_screen((f64::from(size[0]) - 0.5, f64::from(size[1]) - 0.5)),
        );
        scene3d::paint_overlay(
            painter,
            image_rect,
            egui::Id::unique(("xma mesh overlay", camera)),
            &target,
            OverlayDraw {
                size,
                k: shader_k,
                near: OVERLAY_NEAR as f32,
                far: OVERLAY_FAR as f32,
                meshes,
                map,
                transparent_models: state.transparent_models,
                transparency: state.transparency as f32,
            },
        );
    }
    cx.frames.views.meshes.overlay.insert(camera, drawn);
}

/// The model matrix of a body's mesh: the rotation `r` times the mesh's scale `s`, and the
/// translation `t` (where `RigidBodyObj::render` places a mesh), for the camera views'
/// overlay and the 3D world view.
pub(crate) fn body_model_matrix(
    r: &xma_vision::Matrix3,
    t: &xma_model::Vector3,
    s: f64,
) -> scene3d::Mat4 {
    let mut model = xma_engine::view::world::Matrix4::identity();
    for i in 0..3 {
        for j in 0..3 {
            model[(i, j)] = r[(i, j)] * s;
        }
        model[(i, 3)] = t[i];
    }
    mat4(&model)
}
