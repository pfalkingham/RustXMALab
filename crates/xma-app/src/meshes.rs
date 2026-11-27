//! The rigid-body meshes and camera distortion maps the 3D drawings use, loaded or computed
//! on worker threads, and what the 3D world view and the mesh overlays drew last (for the
//! tests and the frame-time measurements).
//!
//! XMALab reads a body's OBJ file when the project is loaded or the file is chosen
//! (`RigidBody::addMeshModel`), and computes each camera's distortion map on a thread the
//! first time a mesh is drawn (`DistortionShader::draw`), drawing the mesh undistorted with
//! a red warning meanwhile. Here a mesh is read the first time it is drawn, on a thread;
//! until it arrives nothing is drawn for it. A file that cannot be read writes one Console
//! line and is not drawn (`RigidBodyObj::vboSet` false). Held in
//! `Frames::views.meshes`.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::mpsc;

use xma_engine::view::world::{DistortionMap, ObjMesh, Orbit, distortion_map, read_obj};
use xma_vision::camera::CameraModel;

use crate::log::AppLog;

/// A mesh file's state.
#[derive(Clone, Debug)]
pub enum MeshState {
    Loading,
    Ready(Arc<ObjMesh>),
    Failed(String),
}

/// A camera's distortion for the overlay.
#[derive(Clone, Debug)]
pub enum MapState {
    /// The camera has no distortion: draw the pinhole projection.
    Identity,
    /// Being computed: XMALab draws undistorted with a warning meanwhile.
    Computing,
    Ready(DistortionMap),
}

enum Done {
    Mesh(String, Result<Arc<ObjMesh>, String>),
    Map(usize, Box<CameraModel>, [u32; 2], Option<DistortionMap>),
}

struct MapSlot {
    model: CameraModel,
    size: [u32; 2],
    state: MapState,
}

/// What the world view drew in its last frame.
#[derive(Clone, Debug, PartialEq)]
pub struct WorldDrawn {
    /// The frame its markers and bodies were drawn at (0-based).
    pub frame: usize,
    pub cameras: usize,
    /// Cameras whose image quad showed a texture.
    pub textured: usize,
    pub spheres: usize,
    pub bodies: usize,
    pub meshes: usize,
    /// Vertices and triangles of the meshes drawn.
    pub vertices: usize,
    pub triangles: usize,
    /// The view's rectangle in points (it may be partly hidden by its window's edges).
    pub rect: egui::Rect,
}

/// What a camera view's mesh overlay drew in its last frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OverlayDrawn {
    pub frame: usize,
    pub meshes: usize,
    pub vertices: usize,
    pub triangles: usize,
    /// Drawn through the distortion map (false: identity, or the map not ready).
    pub distorted: bool,
    /// The "not distorted" warning was shown (the map is being computed).
    pub warning: bool,
    pub transparency: f64,
    pub transparent_models: bool,
    /// The camera matrix handed to the shader, by rows ([`crate::widgets::scene3d::overlay_camera_matrix`]).
    pub camera_matrix: [[f32; 3]; 3],
}

/// The meshes, maps, the world view's viewpoint and what was drawn.
pub struct Meshes {
    meshes: HashMap<String, MeshState>,
    maps: HashMap<usize, MapSlot>,
    tx: mpsc::Sender<Done>,
    rx: mpsc::Receiver<Done>,
    logged: HashSet<String>,
    repaint: Option<egui::Context>,
    /// The 3D world view's viewpoint (`WorldViewDockGLWidget`'s eye distance, angles and
    /// centre). Kept while the program runs, as the C++ widget keeps it.
    pub orbit: Orbit,
    /// What the world view drew last, `None` before it drew or while it draws nothing.
    pub world: Option<WorldDrawn>,
    /// Per camera, what its mesh overlay drew last (absent: no overlay that frame).
    pub overlay: HashMap<usize, OverlayDrawn>,
}

impl Default for Meshes {
    fn default() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            meshes: HashMap::new(),
            maps: HashMap::new(),
            tx,
            rx,
            logged: HashSet::new(),
            repaint: None,
            orbit: Orbit::default(),
            world: None,
            overlay: HashMap::new(),
        }
    }
}

impl Meshes {
    /// Takes what the worker threads finished; writes a Console line for a mesh that could
    /// not be read (once per file).
    pub fn poll(&mut self, ctx: &egui::Context, log: &AppLog) {
        self.repaint = Some(ctx.clone());
        while let Ok(done) = self.rx.try_recv() {
            self.finish(done);
        }
        for (path, state) in &self.meshes {
            if let MeshState::Failed(e) = state
                && self.logged.insert(path.clone())
            {
                log.error(format!("Could not load OBJ-file {path}: {e}"));
            }
        }
    }

    fn finish(&mut self, done: Done) {
        match done {
            Done::Mesh(path, r) => {
                let state = match r {
                    Ok(m) => MeshState::Ready(m),
                    Err(e) => MeshState::Failed(e),
                };
                self.meshes.insert(path, state);
            }
            Done::Map(camera, model, size, map) => {
                if let Some(slot) = self.maps.get_mut(&camera)
                    && slot.model == *model
                    && slot.size == size
                {
                    slot.state = map.map_or(MapState::Identity, MapState::Ready);
                }
            }
        }
    }

    /// The mesh of `path` (as the project names it), read on a thread the first time.
    pub fn mesh(&mut self, path: &str) -> MeshState {
        if let Some(s) = self.meshes.get(path) {
            return s.clone();
        }
        self.meshes.insert(path.to_owned(), MeshState::Loading);
        let tx = self.tx.clone();
        let repaint = self.repaint.clone();
        let p = path.to_owned();
        std::thread::spawn(move || {
            let r = read_obj(std::path::Path::new(&p))
                .map(Arc::new)
                .map_err(|e| e.to_string());
            let _ = tx.send(Done::Mesh(p, r));
            if let Some(c) = repaint {
                c.request_repaint();
            }
        });
        MeshState::Loading
    }

    /// The distortion map of `camera` for `model` at `size`, computed on a thread when the
    /// model or size changed.
    pub fn map(&mut self, camera: usize, model: &CameraModel, size: [u32; 2]) -> MapState {
        if model.grid.is_none() && model.intrinsics.distortion.is_none() {
            return MapState::Identity;
        }
        if let Some(slot) = self.maps.get(&camera)
            && slot.model == *model
            && slot.size == size
        {
            return slot.state.clone();
        }
        self.maps.insert(
            camera,
            MapSlot {
                model: model.clone(),
                size,
                state: MapState::Computing,
            },
        );
        let tx = self.tx.clone();
        let repaint = self.repaint.clone();
        let model = model.clone();
        std::thread::spawn(move || {
            let map = distortion_map(&model, size[0], size[1]);
            let _ = tx.send(Done::Map(camera, Box::new(model), size, map));
            if let Some(c) = repaint {
                c.request_repaint();
            }
        });
        MapState::Computing
    }

    /// Waits until every mesh read and map computation started so far has finished (tests,
    /// screenshots, measurements).
    pub fn wait_idle(&mut self) {
        loop {
            let busy = self
                .meshes
                .values()
                .any(|m| matches!(m, MeshState::Loading))
                || self
                    .maps
                    .values()
                    .any(|m| matches!(m.state, MapState::Computing));
            if !busy {
                return;
            }
            match self.rx.recv() {
                Ok(done) => self.finish(done),
                Err(_) => return,
            }
        }
    }
}
