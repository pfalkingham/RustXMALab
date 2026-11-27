// 3D drawing for the world view and the camera views' mesh overlay
// (`widgets/scene3d/mod.rs`). Colours are OpenGL's: values written to an 8-bit
// framebuffer without gamma conversion; the blit converts when egui's target is sRGB.
//
// Depth: the OpenGL projections give clip z in [-w, w]; wgpu wants [0, w], so every
// vertex stage maps z to (z + w) / 2.

// ---------------------------------------------------------------------------------------
// Blit into egui's pass

struct Blit {
    // The rectangle in normalised device coordinates: x0, y0 (top left), x1, y1.
    rect: vec4<f32>,
    // x: alpha factor (overlay transparency); y: 1 to convert to linear (sRGB target);
    // z: 1 for the overlay (premultiplied, alpha from the texture), 0 opaque.
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> blit: Blit;
@group(1) @binding(0) var src: texture_2d<f32>;
@group(1) @binding(1) var smp: sampler;

struct BlitOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_blit(@builtin(vertex_index) vi: u32) -> BlitOut {
    var xs = array<f32, 6>(0.0, 1.0, 0.0, 0.0, 1.0, 1.0);
    var ys = array<f32, 6>(0.0, 0.0, 1.0, 1.0, 0.0, 1.0);
    let uv = vec2<f32>(xs[vi], ys[vi]);
    var o: BlitOut;
    o.pos = vec4<f32>(mix(blit.rect.x, blit.rect.z, uv.x), mix(blit.rect.y, blit.rect.w, uv.y), 0.0, 1.0);
    o.uv = uv;
    return o;
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        return c / 12.92;
    }
    return pow((c + 0.055) / 1.055, 2.4);
}

@fragment
fn fs_blit(i: BlitOut) -> @location(0) vec4<f32> {
    let t = textureSample(src, smp, i.uv);
    var c = t.rgb;
    if blit.params.y > 0.5 {
        c = vec3<f32>(srgb_to_linear(c.r), srgb_to_linear(c.g), srgb_to_linear(c.b));
    }
    if blit.params.z > 0.5 {
        let a = t.a * blit.params.x;
        return vec4<f32>(c * a, a);
    }
    return vec4<f32>(c, 1.0);
}
