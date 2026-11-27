//! 3D drawing with wgpu inside egui: the 3D world view and the rigid-body mesh overlay of
//! the camera views. Knows nothing about projects: it takes world-space lines, spheres,
//! meshes and image quads (or, for the overlay, meshes in camera coordinates with the
//! camera's intrinsics and distortion map) and draws them.
//!
//! How: an egui paint callback ([`egui_wgpu::CallbackTrait`]) records, in `prepare`, a
//! render pass of its own into an off-screen colour and depth texture (so depth testing
//! works, which egui's pass has no buffer for), and in `paint` draws that texture into
//! egui's pass over the widget's rectangle (`blit.wgsl`):
//!
//! - **World view** (`world.wgsl`): 4x multisampled colour (`Rgba8Unorm`) and depth
//!   (`Depth32Float`) at the view's size in pixels, resolved, then drawn opaque. Triangles
//!   (spheres and meshes) are lit per vertex as OpenGL's fixed-function lighting does
//!   ([`xma_engine::view::world::world_light`]); lines are screen-space quads 2.5 pixels
//!   wide (`glLineWidth(2.5)`, which wgpu has no equivalent of), stippled lines dashed
//!   10 pixels on and off; the camera images are sampled from the image store's integer
//!   textures (the frames the camera views show).
//! - **Mesh overlay** (`overlay.wgsl`): one off-screen texture per camera at the camera's
//!   image size, as XMALab's `rigidbodyBufferUndistorted`. Each vertex is projected with
//!   the pinhole model and moved by the camera's distortion map
//!   ([`xma_engine::view::world::distortion_map`]) in the vertex shader, so the mesh is
//!   drawn distorted directly (XMALab draws undistorted and warps per pixel). Transparent
//!   models: white clear, colours multiplied (`glBlendFunc(GL_ZERO, GL_SRC_COLOR)`), no
//!   depth test; otherwise the nearest surface. Alpha marks covered pixels. The result is
//!   drawn over the image with the view's transparency (`BlendShader`).
//!
//! Without a GPU (headless tests) nothing is drawn: the callbacks never run.

use std::collections::HashMap;
use std::sync::Arc;

use eframe::egui_wgpu::{self, wgpu};
use wgpu::util::DeviceExt;
use xma_engine::view::world::DistortionMap;
use xma_engine::view::world::ObjMesh;

const WORLD_SAMPLES: u32 = 4;
const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// A 4x4 matrix as WGSL wants it: columns.
pub type Mat4 = [[f32; 4]; 4];

/// A column-major `f32` matrix from nalgebra's `f64` one.
#[must_use]
pub fn mat4(m: &xma_engine::view::world::Matrix4) -> Mat4 {
    let mut out = [[0f32; 4]; 4];
    for (c, col) in out.iter_mut().enumerate() {
        for (r, v) in col.iter_mut().enumerate() {
            *v = m[(r, c)] as f32;
        }
    }
    out
}

/// A mesh drawn with a model matrix (object to world for the world view, object to camera
/// for the overlay; the mesh scale included) and a colour.
#[derive(Clone)]
pub struct MeshInstance {
    pub mesh: Arc<ObjMesh>,
    pub model: Mat4,
    pub color: [f32; 4],
}

/// A line segment in world coordinates.
#[derive(Clone, Copy, Debug)]
pub struct WorldLine {
    pub a: [f32; 3],
    pub b: [f32; 3],
    pub color: [f32; 4],
    pub dashed: bool,
}

/// A sphere (`gluSphere`, 32 slices and stacks).
#[derive(Clone, Copy, Debug)]
pub struct WorldSphere {
    pub center: [f32; 3],
    pub radius: f32,
    pub color: [f32; 4],
}

/// A textured quad in world coordinates: a camera's image on its focal plane.
#[derive(Clone)]
pub struct ImageQuad {
    /// Corners in drawing order (a quad: 0, 1, 2, 3 around).
    pub corners: [[f32; 3]; 4],
    /// Texture coordinates of the corners (0 to 1, t = 0 the image's first row).
    pub tex: [[f32; 2]; 4],
    /// The store texture: its bind group, RGBA or grey, bit depth. `None` draws the quad
    /// in its plain colour.
    pub texture: Option<(wgpu::BindGroup, bool, u8)>,
}

/// Everything the world view draws in one frame.
#[derive(Clone, Default)]
pub struct WorldDraw {
    pub view: Mat4,
    pub proj: Mat4,
    pub lines: Vec<WorldLine>,
    pub spheres: Vec<WorldSphere>,
    pub meshes: Vec<MeshInstance>,
    pub quads: Vec<ImageQuad>,
}

/// The camera matrix as the overlay's shader takes it: rows, every entry (the skew
/// `K[0][1]` and the third row included), so that a vertex lands where
/// `xma_engine::view::world::project_mesh` (`Camera::projectPoint`: `K * c` divided by its
/// third value) puts it.
#[must_use]
pub fn overlay_camera_matrix(k: &xma_model::Matrix3) -> [[f32; 3]; 3] {
    std::array::from_fn(|r| std::array::from_fn(|c| k[(r, c)] as f32))
}

/// The shader's projection of a point in camera coordinates (`vs_overlay` before the
/// distortion map), for tests: `None` behind the camera.
#[must_use]
pub fn overlay_project(k: &[[f32; 3]; 3], c: [f32; 3]) -> Option<[f32; 2]> {
    let dot = |r: &[f32; 3]| r[0] * c[0] + r[1] * c[1] + r[2] * c[2];
    let h = [dot(&k[0]), dot(&k[1]), dot(&k[2])];
    (c[2] > 0.0 && h[2] > 0.0).then(|| [h[0] / h[2], h[1] / h[2]])
}

/// One camera view's meshes for the overlay.
#[derive(Clone)]
pub struct OverlayDraw {
    /// The camera image's size in pixels.
    pub size: [u32; 2],
    /// The camera matrix by rows, skew included ([`overlay_camera_matrix`]).
    pub k: [[f32; 3]; 3],
    pub near: f32,
    pub far: f32,
    /// Meshes with their body-to-camera matrices.
    pub meshes: Vec<MeshInstance>,
    /// The distortion map, `None` to draw undistorted.
    pub map: Option<DistortionMap>,
    /// "Draw transparent bone models".
    pub transparent_models: bool,
    /// The view's mesh transparency (0 to 1).
    pub transparency: f32,
}

/// The facts the callbacks need from the window's renderer.
#[derive(Clone)]
pub struct GpuTarget {
    pub format: wgpu::TextureFormat,
    pub texture_layout: wgpu::BindGroupLayout,
}

impl GpuTarget {
    /// From the image store (which lives on eframe's device).
    #[must_use]
    pub fn from_store(store: &crate::widgets::image_view::ImageStore) -> Self {
        Self {
            format: store.target_format(),
            texture_layout: store.texture_layout(),
        }
    }
}

/// How many meshes and distortion maps the 3D drawing keeps on the GPU of `render_state`
/// (`None` before anything was drawn there). For tests and diagnostics.
#[must_use]
pub fn gpu_cache_counts(render_state: &egui_wgpu::RenderState) -> Option<(usize, usize)> {
    let renderer = render_state.renderer.read();
    renderer
        .callback_resources
        .get::<Gpu>()
        .map(|g| (g.meshes.len(), g.maps.len()))
}

/// Adds the world view's drawing over `rect` (points) to `painter`. `key` names its
/// off-screen texture (one per view).
pub fn paint_world(
    painter: &egui::Painter,
    rect: egui::Rect,
    key: egui::Id,
    target: &GpuTarget,
    draw: WorldDraw,
) {
    let ppp = painter.pixels_per_point();
    let size = [
        ((rect.width() * ppp).round() as u32).max(1),
        ((rect.height() * ppp).round() as u32).max(1),
    ];
    painter.add(egui_wgpu::Callback::new_paint_callback(
        rect,
        WorldCallback {
            draw: Arc::new(draw),
            key,
            size,
            rect,
            target: target.clone(),
        },
    ));
}

/// Adds a camera view's mesh overlay to `painter`: the overlay texture over `image_rect`,
/// the screen rectangle (points) of the image's edges, `(-0.5, -0.5)` to
/// `(width - 0.5, height - 0.5)` in image coordinates.
pub fn paint_overlay(
    painter: &egui::Painter,
    image_rect: egui::Rect,
    key: egui::Id,
    target: &GpuTarget,
    draw: OverlayDraw,
) {
    painter.add(egui_wgpu::Callback::new_paint_callback(
        painter.clip_rect().intersect(image_rect),
        OverlayCallback {
            draw: Arc::new(draw),
            key,
            rect: image_rect,
            target: target.clone(),
        },
    ));
}

// ---------------------------------------------------------------------------------------
// GPU resources (kept in egui_wgpu's callback resources)

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct WorldUniform {
    view: Mat4,
    proj: Mat4,
    viewport: [f32; 2],
    line_width: f32,
    pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CamUniform {
    k0: [f32; 4],
    k1: [f32; 4],
    k2: [f32; 4],
    size: [f32; 4],
    flags: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BlitUniform {
    rect: [f32; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 3],
    normal: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Instance {
    model: Mat4,
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LineInstance {
    a: [f32; 3],
    b: [f32; 3],
    color: [f32; 4],
    style: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct QuadVertex {
    pos: [f32; 3],
    uv: [f32; 2],
    info: [f32; 4],
}

/// An off-screen target and the blit's bindings for it.
struct Target {
    size: [u32; 2],
    samples: u32,
    msaa: Option<wgpu::TextureView>,
    resolve: wgpu::TextureView,
    depth: wgpu::TextureView,
    blit_buffer: wgpu::Buffer,
    blit_uniform: wgpu::BindGroup,
    blit_texture: wgpu::BindGroup,
}

/// A mesh on the GPU: one vertex per triangle corner (position, normal).
struct GpuMesh {
    _keep: Arc<ObjMesh>,
    buffer: wgpu::Buffer,
    count: u32,
}

struct GpuMap {
    _keep: Arc<Vec<[f32; 2]>>,
    bind_group: wgpu::BindGroup,
}

struct Gpu {
    format: wgpu::TextureFormat,
    world_layout: wgpu::BindGroupLayout,
    world_tri: wgpu::RenderPipeline,
    world_line: wgpu::RenderPipeline,
    world_quad: wgpu::RenderPipeline,
    cam_layout: wgpu::BindGroupLayout,
    map_layout: wgpu::BindGroupLayout,
    overlay_transparent: wgpu::RenderPipeline,
    overlay_opaque: wgpu::RenderPipeline,
    blit_uniform_layout: wgpu::BindGroupLayout,
    blit_texture_layout: wgpu::BindGroupLayout,
    blit_opaque: wgpu::RenderPipeline,
    blit_over: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    sphere: wgpu::Buffer,
    sphere_count: u32,
    dummy_image: wgpu::BindGroup,
    dummy_map: wgpu::BindGroup,
    targets: HashMap<egui::Id, Target>,
    meshes: HashMap<usize, GpuMesh>,
    maps: HashMap<usize, GpuMap>,
}

fn uniform_layout(
    device: &wgpu::Device,
    label: &str,
    vis: wgpu::ShaderStages,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

const TRI_ATTRS: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
const INST_ATTRS: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
    2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Float32x4
];
const LINE_ATTRS: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
    0 => Float32x3, 1 => Float32x3, 2 => Float32x4, 3 => Float32x4
];
const QUAD_ATTRS: [wgpu::VertexAttribute; 3] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x4];

fn tri_buffers() -> [Option<wgpu::VertexBufferLayout<'static>>; 2] {
    [
        Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &TRI_ATTRS,
        }),
        Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Instance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &INST_ATTRS,
        }),
    ]
}

#[allow(clippy::too_many_arguments)]
fn pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    layouts: &[&wgpu::BindGroupLayout],
    vs: &str,
    fs: &str,
    buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
    format: wgpu::TextureFormat,
    blend: Option<wgpu::BlendState>,
    depth: Option<(wgpu::CompareFunction, bool)>,
    samples: u32,
) -> wgpu::RenderPipeline {
    let layouts: Vec<Option<&wgpu::BindGroupLayout>> = layouts.iter().map(|l| Some(*l)).collect();
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &layouts,
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(vs),
            compilation_options: Default::default(),
            buffers,
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: depth.map(|(compare, write)| wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(write),
            depth_compare: Some(compare),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: samples,
            ..Default::default()
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(fs),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// A unit sphere as `gluSphere(q, 1, 32, 32)` with `GLU_SMOOTH` normals.
fn sphere_vertices() -> Vec<Vertex> {
    let n = xma_engine::view::world::SPHERE_DIVISIONS;
    let pt = |i: u32, j: u32| -> [f32; 3] {
        let theta = std::f32::consts::PI * i as f32 / n as f32; // from +z
        let phi = 2.0 * std::f32::consts::PI * j as f32 / n as f32;
        [
            theta.sin() * phi.sin(),
            theta.sin() * phi.cos(),
            theta.cos(),
        ]
    };
    let mut out = Vec::with_capacity((n * n * 6) as usize);
    for i in 0..n {
        for j in 0..n {
            let (a, b, c, d) = (pt(i, j), pt(i + 1, j), pt(i + 1, j + 1), pt(i, j + 1));
            for p in [a, b, c, a, c, d] {
                out.push(Vertex { pos: p, normal: p });
            }
        }
    }
    out
}

impl Gpu {
    fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        texture_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let world_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene3d world"),
            source: wgpu::ShaderSource::Wgsl(include_str!("world.wgsl").into()),
        });
        let overlay_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene3d overlay"),
            source: wgpu::ShaderSource::Wgsl(include_str!("overlay.wgsl").into()),
        });
        let blit_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene3d blit"),
            source: wgpu::ShaderSource::Wgsl(include_str!("blit.wgsl").into()),
        });
        let vs_fs = wgpu::ShaderStages::VERTEX_FRAGMENT;
        let world_layout = uniform_layout(device, "scene3d world uniforms", vs_fs);
        let cam_layout = uniform_layout(device, "scene3d camera uniforms", vs_fs);
        let blit_uniform_layout = uniform_layout(device, "scene3d blit uniforms", vs_fs);
        let map_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene3d distortion map"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let blit_texture_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scene3d blit texture"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let alpha_blend = Some(wgpu::BlendState::ALPHA_BLENDING);
        let less = Some((wgpu::CompareFunction::LessEqual, true));
        let tri = tri_buffers();
        let world_tri = pipeline(
            device,
            "scene3d world triangles",
            &world_module,
            &[&world_layout],
            "vs_world_tri",
            "fs_color",
            &tri,
            COLOR_FORMAT,
            None,
            less,
            WORLD_SAMPLES,
        );
        let world_line = pipeline(
            device,
            "scene3d world lines",
            &world_module,
            &[&world_layout],
            "vs_world_line",
            "fs_world_line",
            &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<LineInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &LINE_ATTRS,
            })],
            COLOR_FORMAT,
            alpha_blend,
            less,
            WORLD_SAMPLES,
        );
        let world_quad = pipeline(
            device,
            "scene3d world images",
            &world_module,
            &[&world_layout, texture_layout],
            "vs_world_quad",
            "fs_world_quad",
            &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<QuadVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &QUAD_ATTRS,
            })],
            COLOR_FORMAT,
            None,
            less,
            WORLD_SAMPLES,
        );
        // glBlendFunc(GL_ZERO, GL_SRC_COLOR) on the colour; alpha marks coverage.
        let multiply = Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::Src,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Max,
            },
        });
        let overlay_transparent = pipeline(
            device,
            "scene3d overlay transparent",
            &overlay_module,
            &[&cam_layout, &map_layout],
            "vs_overlay",
            "fs_color",
            &tri,
            COLOR_FORMAT,
            multiply,
            Some((wgpu::CompareFunction::Always, true)),
            1,
        );
        let overlay_opaque = pipeline(
            device,
            "scene3d overlay opaque",
            &overlay_module,
            &[&cam_layout, &map_layout],
            "vs_overlay",
            "fs_color",
            &tri,
            COLOR_FORMAT,
            None,
            Some((wgpu::CompareFunction::LessEqual, true)),
            1,
        );
        let blit_opaque = pipeline(
            device,
            "scene3d blit",
            &blit_module,
            &[&blit_uniform_layout, &blit_texture_layout],
            "vs_blit",
            "fs_blit",
            &[],
            format,
            None,
            None,
            1,
        );
        let blit_over = pipeline(
            device,
            "scene3d blit over",
            &blit_module,
            &[&blit_uniform_layout, &blit_texture_layout],
            "vs_blit",
            "fs_blit",
            &[],
            format,
            Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
            None,
            1,
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("scene3d blit"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let sv = sphere_vertices();
        let sphere = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scene3d sphere"),
            contents: bytemuck::cast_slice(&sv),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let tex1 = |format: wgpu::TextureFormat| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("scene3d dummy"),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let dummy_image = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene3d dummy image"),
            layout: texture_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&tex1(wgpu::TextureFormat::R8Uint)),
            }],
        });
        let dummy_map = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene3d dummy map"),
            layout: &map_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&tex1(wgpu::TextureFormat::Rg32Float)),
            }],
        });
        Self {
            format,
            world_layout,
            world_tri,
            world_line,
            world_quad,
            cam_layout,
            map_layout,
            overlay_transparent,
            overlay_opaque,
            blit_uniform_layout,
            blit_texture_layout,
            blit_opaque,
            blit_over,
            sampler,
            sphere,
            sphere_count: sv.len() as u32,
            dummy_image,
            dummy_map,
            targets: HashMap::new(),
            meshes: HashMap::new(),
            maps: HashMap::new(),
        }
    }

    /// The resources, created on first use (or again if egui's target format changed).
    fn get<'a>(
        resources: &'a mut egui_wgpu::CallbackResources,
        device: &wgpu::Device,
        target: &GpuTarget,
    ) -> &'a mut Gpu {
        let stale = resources
            .get::<Gpu>()
            .is_none_or(|g| g.format != target.format);
        if stale {
            resources.insert(Gpu::new(device, target.format, &target.texture_layout));
        }
        resources.get_mut::<Gpu>().expect("inserted")
    }

    fn ensure_target(
        &mut self,
        device: &wgpu::Device,
        key: egui::Id,
        size: [u32; 2],
        samples: u32,
    ) {
        let ok = self
            .targets
            .get(&key)
            .is_some_and(|t| t.size == size && t.samples == samples);
        if !ok {
            let tex = |label: &str, format, samples, usage| {
                device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some(label),
                        size: wgpu::Extent3d {
                            width: size[0],
                            height: size[1],
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: samples,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage,
                        view_formats: &[],
                    })
                    .create_view(&Default::default())
            };
            let att = wgpu::TextureUsages::RENDER_ATTACHMENT;
            let msaa = (samples > 1).then(|| tex("scene3d msaa", COLOR_FORMAT, samples, att));
            let resolve = tex(
                "scene3d colour",
                COLOR_FORMAT,
                1,
                att | wgpu::TextureUsages::TEXTURE_BINDING,
            );
            let depth = tex("scene3d depth", DEPTH_FORMAT, samples, att);
            let blit_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scene3d blit uniforms"),
                size: std::mem::size_of::<BlitUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let blit_uniform = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scene3d blit uniforms"),
                layout: &self.blit_uniform_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: blit_buffer.as_entire_binding(),
                }],
            });
            let blit_texture = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scene3d blit texture"),
                layout: &self.blit_texture_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&resolve),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            self.targets.insert(
                key,
                Target {
                    size,
                    samples,
                    msaa,
                    resolve,
                    depth,
                    blit_buffer,
                    blit_uniform,
                    blit_texture,
                },
            );
        }
    }

    fn mesh(&mut self, device: &wgpu::Device, mesh: &Arc<ObjMesh>) -> (wgpu::Buffer, u32) {
        let id = Arc::as_ptr(mesh) as usize;
        let e = self.meshes.entry(id).or_insert_with(|| {
            let mut v = Vec::with_capacity(mesh.triangles.len() * 3);
            for (t, n) in mesh.triangles.iter().zip(&mesh.normals) {
                for k in 0..3 {
                    v.push(Vertex {
                        pos: mesh.positions[t[k] as usize],
                        normal: n[k],
                    });
                }
            }
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scene3d mesh"),
                contents: bytemuck::cast_slice(&v),
                usage: wgpu::BufferUsages::VERTEX,
            });
            GpuMesh {
                _keep: mesh.clone(),
                buffer,
                count: v.len() as u32,
            }
        });
        (e.buffer.clone(), e.count)
    }

    /// Drops meshes and maps nobody else holds any more. Called from the prepare step of
    /// both callbacks (world view and mesh overlay).
    fn prune(&mut self) {
        self.meshes.retain(|_, m| Arc::strong_count(&m._keep) > 1);
        self.maps.retain(|_, m| Arc::strong_count(&m._keep) > 1);
    }

    fn map(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        map: &DistortionMap,
    ) -> wgpu::BindGroup {
        let id = Arc::as_ptr(&map.data) as usize;
        if let Some(m) = self.maps.get(&id) {
            return m.bind_group.clone();
        }
        let size = wgpu::Extent3d {
            width: map.width as u32,
            height: map.height as u32,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene3d distortion map"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(map.data.as_slice()),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(map.width as u32 * 8),
                rows_per_image: Some(map.height as u32),
            },
            size,
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene3d distortion map"),
            layout: &self.map_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &texture.create_view(&Default::default()),
                ),
            }],
        });
        self.maps.insert(
            id,
            GpuMap {
                _keep: map.data.clone(),
                bind_group: bind_group.clone(),
            },
        );
        bind_group
    }
}

fn uniform_bind_group<T: bytemuck::Pod>(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    value: &T,
) -> wgpu::BindGroup {
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("scene3d uniforms"),
        contents: bytemuck::bytes_of(value),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scene3d uniforms"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    })
}

/// The rectangle in normalised device coordinates of the screen.
fn ndc_rect(rect: egui::Rect, screen: &egui_wgpu::ScreenDescriptor) -> [f32; 4] {
    let ppp = screen.pixels_per_point;
    let (w, h) = (
        screen.size_in_pixels[0] as f32,
        screen.size_in_pixels[1] as f32,
    );
    let x = |p: f32| p * ppp / w * 2.0 - 1.0;
    let y = |p: f32| 1.0 - p * ppp / h * 2.0;
    [x(rect.min.x), y(rect.min.y), x(rect.max.x), y(rect.max.y)]
}

fn blit_paint(
    gpu: &Gpu,
    target: &Target,
    over: bool,
    info: &egui::PaintCallbackInfo,
    pass: &mut wgpu::RenderPass<'static>,
) {
    pass.set_viewport(
        0.0,
        0.0,
        info.screen_size_px[0] as f32,
        info.screen_size_px[1] as f32,
        0.0,
        1.0,
    );
    pass.set_pipeline(if over {
        &gpu.blit_over
    } else {
        &gpu.blit_opaque
    });
    pass.set_bind_group(0, &target.blit_uniform, &[]);
    pass.set_bind_group(1, &target.blit_texture, &[]);
    pass.draw(0..6, 0..1);
}

// ---------------------------------------------------------------------------------------
// The world view callback

struct WorldCallback {
    draw: Arc<WorldDraw>,
    key: egui::Id,
    size: [u32; 2],
    rect: egui::Rect,
    target: GpuTarget,
}

impl egui_wgpu::CallbackTrait for WorldCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen: &egui_wgpu::ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let gpu = Gpu::get(resources, device, &self.target);
        gpu.prune();
        let d = &*self.draw;
        let uniform = uniform_bind_group(
            device,
            &gpu.world_layout,
            &WorldUniform {
                view: d.view,
                proj: d.proj,
                viewport: [self.size[0] as f32, self.size[1] as f32],
                line_width: xma_engine::view::world::LINE_WIDTH_PX,
                pad: 0.0,
            },
        );
        // Instances.
        let sphere_inst: Vec<Instance> = d
            .spheres
            .iter()
            .map(|s| {
                let r = s.radius;
                Instance {
                    model: [
                        [r, 0.0, 0.0, 0.0],
                        [0.0, r, 0.0, 0.0],
                        [0.0, 0.0, r, 0.0],
                        [s.center[0], s.center[1], s.center[2], 1.0],
                    ],
                    color: s.color,
                }
            })
            .collect();
        let sphere_buf = (!sphere_inst.is_empty()).then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scene3d spheres"),
                contents: bytemuck::cast_slice(&sphere_inst),
                usage: wgpu::BufferUsages::VERTEX,
            })
        });
        let meshes: Vec<(wgpu::Buffer, u32, wgpu::Buffer)> = d
            .meshes
            .iter()
            .map(|m| {
                let (buf, n) = gpu.mesh(device, &m.mesh);
                let inst = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("scene3d mesh instance"),
                    contents: bytemuck::bytes_of(&Instance {
                        model: m.model,
                        color: m.color,
                    }),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                (buf, n, inst)
            })
            .collect();
        let lines: Vec<LineInstance> = d
            .lines
            .iter()
            .map(|l| LineInstance {
                a: l.a,
                b: l.b,
                color: l.color,
                style: [if l.dashed { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
            })
            .collect();
        let line_buf = (!lines.is_empty()).then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scene3d lines"),
                contents: bytemuck::cast_slice(&lines),
                usage: wgpu::BufferUsages::VERTEX,
            })
        });
        let quads: Vec<(wgpu::Buffer, wgpu::BindGroup)> = d
            .quads
            .iter()
            .map(|q| {
                let info = match &q.texture {
                    Some((_, rgba, depth)) => {
                        let shift = u32::from((*depth).clamp(8, 16) - 8);
                        [
                            1.0,
                            if *rgba { 1.0 } else { 0.0 },
                            (1.0 / (f64::from(1u32 << shift) * 255.0)) as f32,
                            0.0,
                        ]
                    }
                    None => [0.0; 4],
                };
                let v: Vec<QuadVertex> = [0, 1, 2, 0, 2, 3]
                    .iter()
                    .map(|&i| QuadVertex {
                        pos: q.corners[i],
                        uv: q.tex[i],
                        info,
                    })
                    .collect();
                let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("scene3d image quad"),
                    contents: bytemuck::cast_slice(&v),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let bg = q
                    .texture
                    .as_ref()
                    .map_or_else(|| gpu.dummy_image.clone(), |t| t.0.clone());
                (buf, bg)
            })
            .collect();

        gpu.ensure_target(device, self.key, self.size, WORLD_SAMPLES);
        let gpu: &Gpu = gpu;
        let target = &gpu.targets[&self.key];
        queue.write_buffer(
            &target.blit_buffer,
            0,
            bytemuck::bytes_of(&BlitUniform {
                rect: ndc_rect(self.rect, screen),
                params: [
                    1.0,
                    if self.target.format.is_srgb() {
                        1.0
                    } else {
                        0.0
                    },
                    0.0,
                    0.0,
                ],
            }),
        );
        let (color_view, resolve) = match &target.msaa {
            Some(m) => (m, Some(&target.resolve)),
            None => (&target.resolve, None),
        };
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene3d world"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    depth_slice: None,
                    resolve_target: resolve,
                    ops: wgpu::Operations {
                        // glClearColor(0, 0, 0, 0): black.
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &uniform, &[]);
            pass.set_pipeline(&gpu.world_quad);
            for (buf, bg) in &quads {
                pass.set_bind_group(1, bg, &[]);
                pass.set_vertex_buffer(0, buf.slice(..));
                pass.draw(0..6, 0..1);
            }
            pass.set_pipeline(&gpu.world_tri);
            if let Some(sb) = &sphere_buf {
                pass.set_vertex_buffer(0, gpu.sphere.slice(..));
                pass.set_vertex_buffer(1, sb.slice(..));
                pass.draw(0..gpu.sphere_count, 0..sphere_inst.len() as u32);
            }
            for (buf, n, inst) in &meshes {
                pass.set_vertex_buffer(0, buf.slice(..));
                pass.set_vertex_buffer(1, inst.slice(..));
                pass.draw(0..*n, 0..1);
            }
            if let Some(lb) = &line_buf {
                pass.set_pipeline(&gpu.world_line);
                pass.set_vertex_buffer(0, lb.slice(..));
                pass.draw(0..6, 0..lines.len() as u32);
            }
        }
        Vec::new()
    }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(gpu) = resources.get::<Gpu>() else {
            return;
        };
        let Some(target) = gpu.targets.get(&self.key) else {
            return;
        };
        blit_paint(gpu, target, false, &info, pass);
    }
}

// ---------------------------------------------------------------------------------------
// The overlay callback

struct OverlayCallback {
    draw: Arc<OverlayDraw>,
    key: egui::Id,
    rect: egui::Rect,
    target: GpuTarget,
}

impl egui_wgpu::CallbackTrait for OverlayCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen: &egui_wgpu::ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let gpu = Gpu::get(resources, device, &self.target);
        // Free what neither view holds any more here too: with the 3D world view closed, a
        // camera model change would otherwise keep every replaced map (review 12). Whatever
        // a callback of this frame draws is held by that callback, so it is kept.
        gpu.prune();
        let d = &*self.draw;
        let uniform = uniform_bind_group(
            device,
            &gpu.cam_layout,
            &CamUniform {
                k0: [d.k[0][0], d.k[0][1], d.k[0][2], 0.0],
                k1: [d.k[1][0], d.k[1][1], d.k[1][2], 0.0],
                k2: [d.k[2][0], d.k[2][1], d.k[2][2], 0.0],
                size: [d.size[0] as f32, d.size[1] as f32, d.near, d.far],
                flags: [if d.map.is_some() { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
            },
        );
        let map = match &d.map {
            Some(m) => gpu.map(device, queue, m),
            None => gpu.dummy_map.clone(),
        };
        let meshes: Vec<(wgpu::Buffer, u32, wgpu::Buffer)> = d
            .meshes
            .iter()
            .map(|m| {
                let (buf, n) = gpu.mesh(device, &m.mesh);
                let inst = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("scene3d overlay instance"),
                    contents: bytemuck::bytes_of(&Instance {
                        model: m.model,
                        color: m.color,
                    }),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                (buf, n, inst)
            })
            .collect();
        gpu.ensure_target(device, self.key, d.size, 1);
        let gpu: &Gpu = gpu;
        let target = &gpu.targets[&self.key];
        queue.write_buffer(
            &target.blit_buffer,
            0,
            bytemuck::bytes_of(&BlitUniform {
                rect: ndc_rect(self.rect, screen),
                params: [
                    d.transparency,
                    if self.target.format.is_srgb() {
                        1.0
                    } else {
                        0.0
                    },
                    1.0,
                    0.0,
                ],
            }),
        );
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene3d overlay"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.resolve,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // glClearColor(1, 1, 1, 0).
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 1.0,
                            g: 1.0,
                            b: 1.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(if d.transparent_models {
                &gpu.overlay_transparent
            } else {
                &gpu.overlay_opaque
            });
            pass.set_bind_group(0, &uniform, &[]);
            pass.set_bind_group(1, &map, &[]);
            for (buf, n, inst) in &meshes {
                pass.set_vertex_buffer(0, buf.slice(..));
                pass.set_vertex_buffer(1, inst.slice(..));
                pass.draw(0..*n, 0..1);
            }
        }
        Vec::new()
    }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(gpu) = resources.get::<Gpu>() else {
            return;
        };
        let Some(target) = gpu.targets.get(&self.key) else {
            return;
        };
        blit_paint(gpu, target, true, &info, pass);
    }
}
