//! The 3D view of the 3D world view panel (`WorldViewDockGLWidget`): its mouse, wheel and
//! keys, and its drawing.
//!
//! The geometry comes from [`xma_engine::view::world::world_scene`] (what XMALab draws,
//! listed in that module's documentation); the viewpoint is
//! [`xma_engine::view::world::Orbit`], kept in `Frames::views.meshes.orbit`; the drawing is
//! [`crate::widgets::scene3d`]. Without a GPU (headless tests) the area is painted black
//! and the geometry is still computed and recorded (`Meshes::world`).
//!
//! A camera hidden with the camera selector keeps its frustum lines but its image plane is
//! empty while it is hidden, in every workspace (ledger U166). XMALab 3 instead binds the
//! hidden camera's last texture: the frame it showed when hidden in Marker tracking, and
//! the current calibration image in Calibration.
//!
//! Input, as `WorldViewDockGLWidget` handles it (positions in points, as Qt's logical
//! pixels):
//!
//! | Input | Effect |
//! |---|---|
//! | Left drag | rotate: azimuth follows vertical motion (0 to 180), polar horizontal (wraps) |
//! | Middle drag, or Ctrl (Command on macOS) + right drag | pan the centre, eye distance / 500 per point |
//! | Right press without Ctrl | back to the centre; if already there, the starting view |
//! | Wheel | eye distance + delta / 12 (10 per notch; away from the user zooms out), at least 1 |
//! | R, P with the pointer over the view | as right press; 10 nearer (not below 100) |
//!
//! Key O (10 farther) is not taken: it is the application's "Event on" shortcut, which the
//! C++ also gives it first (`docs/ui-map.md`); in XMALab 3 the view never has the keyboard
//! focus, so none of the three keys reach it there.

use egui::{Color32, Event, MouseWheelUnit, PointerButton, Pos2, Rect, Sense};
use xma_engine::view::world::{
    CameraImage, Orbit, SPHERE_RADIUS, WorldQuery, WorldScene, WorldWorkspace, world_scene,
};

use crate::frames::Frames;
use crate::meshes::{MeshState, WorldDrawn};
use crate::panels::Cx;
use crate::state::Workspace;
use crate::widgets::image_view::ANGLE_DELTA_PER_NOTCH;
use crate::widgets::scene3d::{
    self, GpuTarget, ImageQuad, MeshInstance, WorldDraw, WorldLine, WorldSphere, mat4,
};

/// The panel's settings the view reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewSettings {
    /// The panel's own frame (0-based) while "enable" is checked.
    pub own_frame: Option<usize>,
    /// "Focal plane distance".
    pub focal_plane: f64,
}

/// The accessible name of the 3D view.
pub const VIEW_NAME: &str = "3D world view";

/// The drag in progress: the buttons pressed on the view, the last position and the
/// modifiers (kept between frames).
#[derive(Clone, Copy, Debug, Default)]
struct Drag {
    prev: Option<Pos2>,
    primary: bool,
    middle: bool,
    secondary: bool,
    modifiers: egui::Modifiers,
}

/// Draws the 3D view into `rect` and handles its input.
pub fn show(ui: &mut egui::Ui, cx: &mut Cx<'_>, rect: Rect, settings: ViewSettings) {
    let id = egui::Id::unique("xma 3d world view");
    let response = ui
        .interact(rect, id, Sense::click_and_drag())
        .accessible_name(VIEW_NAME);
    input(ui, &response, &mut cx.frames.views.meshes.orbit, id);
    cx.frames.views.meshes.poll(ui.ctx(), cx.log);

    let painter = ui.painter_at(rect);
    // glClearColor(0, 0, 0, 0): black (module documentation of `xma_engine::view::world`).
    painter.rect_filled(rect, 0.0, Color32::BLACK);
    let Some(doc) = cx.doc else {
        cx.frames.views.meshes.world = None;
        return;
    };
    let workspace = match cx.state.workspace {
        Workspace::Undistortion => WorldWorkspace::Undistortion,
        Workspace::Calibration => WorldWorkspace::Calibration,
        Workspace::MarkerTracking => WorldWorkspace::MarkerTracking,
    };
    let view = cx.state.trial_view();
    let trial_frame = view.map_or(0, |v| v.active_frame);
    let q = WorldQuery {
        workspace,
        trial: cx.state.active_trial,
        trial_frame,
        view_frame: settings.own_frame.unwrap_or(trial_frame),
        custom_timeline: settings.own_frame.is_some(),
        active_marker: view.and_then(|v| v.active_marker),
        calibration_frame: cx.state.active_frame_calibration.unwrap_or(0),
        focal_plane: settings.focal_plane,
        filtered: cx.settings.trial_draw_filtered,
    };
    let scene = world_scene(doc, &q);
    let orbit = cx.frames.views.meshes.orbit;
    let draw = build(cx, &scene, &orbit, rect, workspace);
    let mut drawn = WorldDrawn {
        frame: q.view_frame,
        cameras: scene.cameras.len(),
        textured: draw.quads.iter().filter(|q| q.texture.is_some()).count(),
        spheres: scene.spheres.len(),
        bodies: scene.bodies.len(),
        meshes: draw.meshes.len(),
        rect,
        vertices: 0,
        triangles: 0,
    };
    for m in &draw.meshes {
        drawn.vertices += m.mesh.n_vertices();
        drawn.triangles += m.mesh.n_triangles();
    }
    cx.frames.views.meshes.world = Some(drawn);
    if let Some(store) = cx.frames.store() {
        let target = GpuTarget::from_store(store);
        scene3d::paint_world(&painter, rect, id, &target, draw);
    }
}

/// The mouse, wheel and keys (module documentation).
fn input(ui: &egui::Ui, response: &egui::Response, orbit: &mut Orbit, id: egui::Id) {
    let rect = response.rect;
    let over = response.contains_pointer();
    let line_speed = ui.ctx().options(|o| o.input_options.line_scroll_speed);
    let mut drag: Drag = ui.data(|d| d.get_temp(id)).unwrap_or_default();
    let (events, any_down) = ui.input(|i| (i.events.clone(), i.pointer.any_down()));
    for e in &events {
        match *e {
            Event::ModifiersChanged(m) => drag.modifiers = m,
            Event::PointerButton {
                pos,
                button,
                pressed: true,
                modifiers,
            } => {
                drag.modifiers = modifiers;
                if !(over && rect.contains(pos)) {
                    continue;
                }
                // mousePressEvent: the right button without Ctrl resets; every press
                // starts a drag from here.
                if button == PointerButton::Secondary && !modifiers.command {
                    orbit.reset_step();
                }
                match button {
                    PointerButton::Primary => drag.primary = true,
                    PointerButton::Middle => drag.middle = true,
                    PointerButton::Secondary => drag.secondary = true,
                    _ => {}
                }
                drag.prev = Some(pos);
            }
            Event::PointerButton {
                button,
                pressed: false,
                modifiers,
                ..
            } => {
                drag.modifiers = modifiers;
                match button {
                    PointerButton::Primary => drag.primary = false,
                    PointerButton::Middle => drag.middle = false,
                    PointerButton::Secondary => drag.secondary = false,
                    _ => {}
                }
                if !(drag.primary || drag.middle || drag.secondary) {
                    drag.prev = None;
                }
            }
            Event::PointerMoved(pos) => {
                let Some(prev) = drag.prev else { continue };
                let (dx, dy) = (f64::from(pos.x - prev.x), f64::from(pos.y - prev.y));
                // mouseMoveEvent: the left button first; then the middle button, or the
                // right one with Ctrl.
                if drag.primary {
                    orbit.rotate(dx, dy);
                } else if drag.middle || (drag.secondary && drag.modifiers.command) {
                    orbit.pan(dx, dy);
                }
                drag.prev = Some(pos);
            }
            Event::MouseWheel { unit, delta, .. } => {
                if !over {
                    continue;
                }
                let notches = match unit {
                    MouseWheelUnit::Line => f64::from(delta.y),
                    MouseWheelUnit::Point => f64::from(delta.y / line_speed),
                    MouseWheelUnit::Page => f64::from(delta.y * rect.height() / line_speed),
                };
                if notches != 0.0 {
                    orbit.wheel(notches * ANGLE_DELTA_PER_NOTCH);
                }
            }
            Event::Key {
                key,
                pressed: true,
                modifiers,
                ..
            } if over && modifiers.is_none() => match key {
                egui::Key::R => orbit.reset_step(),
                egui::Key::P => orbit.key_zoom(false),
                _ => {}
            },
            _ => {}
        }
    }
    // A release outside the window is not always reported.
    if !any_down && events.is_empty() {
        drag = Drag {
            modifiers: drag.modifiers,
            ..Drag::default()
        };
    }
    ui.data_mut(|d| d.insert_temp(id, drag));
}

fn f3(p: &xma_model::Point3) -> [f32; 3] {
    [p.x as f32, p.y as f32, p.z as f32]
}

/// What to draw, from the scene: the lines, spheres, image quads (with the textures the
/// camera views show) and meshes (those already read).
fn build(
    cx: &mut Cx<'_>,
    scene: &WorldScene,
    orbit: &Orbit,
    rect: Rect,
    workspace: WorldWorkspace,
) -> WorldDraw {
    let aspect = f64::from(rect.width()) / f64::from(rect.height().max(1.0));
    let mut d = WorldDraw {
        view: mat4(&orbit.view_matrix()),
        proj: mat4(&Orbit::projection(aspect)),
        ..WorldDraw::default()
    };
    let line =
        |a: &xma_model::Point3, b: &xma_model::Point3, color: [f32; 4], dashed: bool| WorldLine {
            a: f3(a),
            b: f3(b),
            color,
            dashed,
        };
    for (a, b, c) in &scene.axes {
        d.lines.push(line(a, b, *c, false));
    }
    let store = cx.frames.store().cloned();
    for cam in &scene.cameras {
        for (a, b, c) in &cam.axes {
            d.lines.push(line(a, b, *c, false));
        }
        let fr = xma_engine::view::world::FRUSTUM;
        for k in 0..4 {
            d.lines
                .push(line(&cam.position, &cam.corners[k], fr, false));
            d.lines
                .push(line(&cam.corners[k], &cam.corners[(k + 1) % 4], fr, false));
        }
        // A camera hidden with the camera selector has an empty plane while it is hidden
        // (ledger U166): its trial frame is not uploaded, and no other image (the
        // calibration workspace's undistorted image) is shown or made for it.
        let visible = cx
            .state
            .camera_visible
            .get(cam.camera)
            .copied()
            .unwrap_or(true);
        let texture = match (&store, cam.image) {
            (Some(store), CameraImage::Shown) if visible => {
                let key = image_key(cx, cam.camera, workspace);
                store.texture_binding(key)
            }
            _ => None,
        };
        d.quads.push(ImageQuad {
            corners: [
                f3(&cam.corners[0]),
                f3(&cam.corners[1]),
                f3(&cam.corners[2]),
                f3(&cam.corners[3]),
            ],
            tex: cam.tex,
            texture,
        });
    }
    for s in &scene.spheres {
        d.spheres.push(WorldSphere {
            center: f3(&s.position),
            radius: SPHERE_RADIUS as f32,
            color: s.color,
        });
    }
    for b in &scene.bodies {
        for [p, q] in &b.lines {
            d.lines.push(line(p, q, b.color, false));
        }
        for [p, q] in &b.virtual_lines {
            d.lines.push(line(p, q, b.color, true));
        }
    }
    for m in &scene.meshes {
        if let MeshState::Ready(obj) = cx.frames.views.meshes.mesh(&m.path) {
            let (r, t) = m.body_to_world();
            d.meshes.push(MeshInstance {
                mesh: obj,
                model: crate::camera_view::overlay_mesh::body_model_matrix(&r, &t, m.scale),
                color: m.color,
            });
        }
    }
    d
}

/// The texture a camera's image quad shows: what its camera view shows (the calibration
/// image, undistorted when the view shows it so, or the trial frame).
fn image_key(cx: &mut Cx<'_>, camera: usize, workspace: WorldWorkspace) -> egui::Id {
    match workspace {
        WorldWorkspace::Calibration => {
            crate::camera_view::overlay_calibration::image_key(cx, camera)
                .unwrap_or_else(|| Frames::image_key(camera))
        }
        _ => Frames::image_key(camera),
    }
}
