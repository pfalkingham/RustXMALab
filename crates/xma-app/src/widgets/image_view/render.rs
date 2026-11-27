//! GPU side: the pipeline, the per-view uniform buffer and the egui paint
//! callback that draws the image into egui's own render pass.

use eframe::egui_wgpu::{self, wgpu};

use super::transform::ViewTransform;

/// Scale and bias of the displayed grey levels, as the Scale and Bias controls
/// set them (`GLCameraView::setScale`, `setBias`). Identity is scale 1, bias 0.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageAdjust {
    pub scale: f64,
    pub bias: f64,
}

impl Default for ImageAdjust {
    fn default() -> Self {
        Self {
            scale: 1.0,
            bias: 0.0,
        }
    }
}

impl ImageAdjust {
    /// Display level in [0, 1] of a grey level `v` in [0, 1] (the image as the
    /// texture holds it, after filtering).
    ///
    /// Ported from `ui/GLCameraView.cpp`, `GLCameraView::paintGL` (the blend
    /// passes): for scale > 1 the framebuffer is multiplied by the scale
    /// (clamped to [0, 1]) and then the bias is added or subtracted (clamped);
    /// for scale <= 1 one pass gives `v * scale + bias` (clamped). `glColor`
    /// clamps the scale used as alpha to [0, 1], so a negative scale acts as 0.
    /// The 8-bit rounding of the framebuffer between passes is not reproduced.
    pub fn apply(&self, v: f64) -> f64 {
        if self.scale > 1.0 {
            ((v * self.scale).clamp(0.0, 1.0) + self.bias).clamp(0.0, 1.0)
        } else {
            (v * self.scale.max(0.0) + self.bias).clamp(0.0, 1.0)
        }
    }
}

/// How image pixels are interpolated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Filtering {
    /// Nearest pixel when magnified (zoom ratio <= 1, one image pixel covers at
    /// least one device pixel), bilinear when reduced. Shows the real pixels when
    /// placing markers.
    NearestWhenMagnified,
    /// Bilinear at every zoom, as XMALab 3 draws (`GL_LINEAR` for both
    /// minification and magnification, no mipmaps; `core/Image.cpp`,
    /// `Image::loadTexture`). The default.
    #[default]
    Linear,
    /// Nearest pixel at every zoom.
    Nearest,
}

impl Filtering {
    /// True when this filter reads the nearest pixel at the given zoom ratio.
    pub fn is_nearest(self, zoom_ratio: f64) -> bool {
        match self {
            Filtering::NearestWhenMagnified => zoom_ratio <= 1.0,
            Filtering::Linear => false,
            Filtering::Nearest => true,
        }
    }
}

/// Grey level in [0, 1] that a raw value of a `bit_depth`-bit image is shown at,
/// before scale and bias.
///
/// Full depth (the default, parity ledger V1): `raw / (2^(bit_depth - 8) * 255)`,
/// clamped to 1. This is the C++ display level without its 8-bit rounding.
/// With `reduce_to_8bit` the value is first reduced as `convertTo(CV_8U,
/// 1 / 2^(bit_depth - 8))` does (round half to even, saturate at 255), as in
/// `core/Image.cpp` and `core/CineVideo.cpp`.
pub fn grey_level(raw: u32, bit_depth: u8, reduce_to_8bit: bool) -> f64 {
    let shift = u32::from(bit_depth.clamp(8, 16) - 8);
    if reduce_to_8bit {
        if shift == 0 {
            return f64::from(raw.min(255)) / 255.0;
        }
        let mut q = raw >> shift;
        let r = raw & ((1 << shift) - 1);
        let h = 1 << (shift - 1);
        if r > h || (r == h && q & 1 == 1) {
            q += 1;
        }
        f64::from(q.min(255)) / 255.0
    } else {
        (f64::from(raw) / (f64::from(1u32 << shift) * 255.0)).min(1.0)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Uniforms {
    origin: [f32; 2],
    screen: [f32; 2],
    zoom: f32,
    scale: f32,
    bias: f32,
    inv_white: f32,
    filter_mode: u32,
    flags: u32,
    shift: u32,
    pad: u32,
    background: [f32; 4],
}

/// What the shader needs to know about the texture.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TextureMeta {
    pub rgba: bool,
    pub bit_depth: u8,
}

pub(crate) struct DrawParams {
    pub transform: ViewTransform,
    pub adjust: ImageAdjust,
    pub filtering: Filtering,
    pub reduce_to_8bit: bool,
    pub background: Option<egui::Color32>,
}

impl Uniforms {
    pub fn new(p: &DrawParams, meta: TextureMeta, srgb_target: bool) -> Self {
        let t = &p.transform;
        let z = t.zoom_ratio;
        let ppp = f64::from(t.pixels_per_point);
        // Image coordinate at framebuffer pixel position (0, 0):
        // view_px = fb - view_min * ppp, image = z * view_px - (0.5 z W + xo) - 0.5.
        let left = f64::from(t.view_min.x) * ppp;
        let top = f64::from(t.view_min.y) * ppp;
        let origin = [
            -z * left - (0.5 * z * t.view_size_px[0] + t.offset[0]) - 0.5,
            -z * top - (0.5 * z * t.view_size_px[1] + t.offset[1]) - 0.5,
        ];
        let shift = u32::from(meta.bit_depth.clamp(8, 16) - 8);
        let mut flags = 0;
        if srgb_target {
            flags |= 1;
        }
        if meta.rgba {
            flags |= 2;
        }
        if p.reduce_to_8bit {
            flags |= 4;
        }
        let mut background = [0.0; 4];
        if let Some(c) = p.background {
            flags |= 8;
            let rgba = if srgb_target {
                egui::Rgba::from(c).to_array()
            } else {
                c.to_normalized_gamma_f32()
            };
            background = rgba;
        }
        Self {
            origin: [origin[0] as f32, origin[1] as f32],
            screen: [0.0, 0.0],
            zoom: z as f32,
            scale: p.adjust.scale as f32,
            bias: p.adjust.bias as f32,
            inv_white: (1.0 / (f64::from(1u32 << shift) * 255.0)) as f32,
            filter_mode: u32::from(!p.filtering.is_nearest(z)),
            flags,
            shift,
            pad: 0,
            background,
        }
    }
}

/// Pipeline and layouts, shared by every view drawn through one store.
pub(crate) struct Pipelines {
    pub pipeline: wgpu::RenderPipeline,
    pub uniform_layout: wgpu::BindGroupLayout,
    pub texture_layout: wgpu::BindGroupLayout,
    pub srgb_target: bool,
    /// egui's target format (also used by the 3D drawings, `widgets::scene3d`).
    pub target_format: wgpu::TextureFormat,
}

impl Pipelines {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("image view shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("image view uniforms"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("image view texture"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("image view layout"),
            bind_group_layouts: &[Some(&uniform_layout), Some(&texture_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("image view pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        Self {
            pipeline,
            uniform_layout,
            texture_layout,
            srgb_target: target_format.is_srgb(),
            target_format,
        }
    }
}

/// The uniform buffer of one view, kept in its [`super::ImageViewState`].
pub(crate) struct ViewGpu {
    pub buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    /// Identity of the store whose device made it.
    pub store_id: usize,
}

impl ViewGpu {
    pub fn new(device: &wgpu::Device, pipelines: &Pipelines, store_id: usize) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("image view uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("image view uniforms"),
            layout: &pipelines.uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Self {
            buffer,
            bind_group,
            store_id,
        }
    }
}

/// One image draw, added to egui's paint list.
pub(crate) struct ImageCallback {
    pub pipeline: wgpu::RenderPipeline,
    pub buffer: wgpu::Buffer,
    pub view_bind_group: wgpu::BindGroup,
    pub texture_bind_group: wgpu::BindGroup,
    pub uniforms: Uniforms,
}

impl egui_wgpu::CallbackTrait for ImageCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        _resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let mut u = self.uniforms;
        u.screen = [
            screen.size_in_pixels[0] as f32,
            screen.size_in_pixels[1] as f32,
        ];
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&u));
        Vec::new()
    }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        _resources: &egui_wgpu::CallbackResources,
    ) {
        // The whole framebuffer as viewport, so that the shader's mapping does not
        // depend on egui rounding or clamping the callback rectangle; egui's scissor
        // (the clip rectangle) still limits drawing to the view.
        pass.set_viewport(
            0.0,
            0.0,
            info.screen_size_px[0] as f32,
            info.screen_size_px[1] as f32,
            0.0,
            1.0,
        );
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.view_bind_group, &[]);
        pass.set_bind_group(1, &self.texture_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
