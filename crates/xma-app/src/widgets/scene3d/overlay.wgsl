// 3D drawing for the world view and the camera views' mesh overlay
// (`widgets/scene3d/mod.rs`). Colours are OpenGL's: values written to an 8-bit
// framebuffer without gamma conversion; the blit converts when egui's target is sRGB.
//
// Depth: the OpenGL projections give clip z in [-w, w]; wgpu wants [0, w], so every
// vertex stage maps z to (z + w) / 2.

struct TriIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // Model (object to world) columns.
    @location(2) m0: vec4<f32>,
    @location(3) m1: vec4<f32>,
    @location(4) m2: vec4<f32>,
    @location(5) m3: vec4<f32>,
    @location(6) color: vec4<f32>,
};

struct ColorOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@fragment
fn fs_color(i: ColorOut) -> @location(0) vec4<f32> {
    return i.color;
}

// ---------------------------------------------------------------------------------------
// Camera view mesh overlay

struct Cam {
    // The camera matrix by rows (w unused), skew included, as
    // `xma_engine::view::world::project_mesh` applies it (pixel centres at integers).
    k0: vec4<f32>,
    k1: vec4<f32>,
    k2: vec4<f32>,
    // width, height, near, far.
    size: vec4<f32>,
    // x: 1 when the distortion map is used.
    flags: vec4<f32>,
};

@group(0) @binding(0) var<uniform> cam: Cam;
@group(1) @binding(0) var dmap: texture_2d<f32>;

fn map_at(p: vec2<i32>) -> vec2<f32> {
    let dims = vec2<i32>(textureDimensions(dmap));
    return textureLoad(dmap, clamp(p, vec2<i32>(0, 0), dims - vec2<i32>(1, 1)), 0).xy;
}

// `DistortionMap::apply`: bilinear inside, edge value moved by the distance outside.
fn distort(u: vec2<f32>) -> vec2<f32> {
    let dims = vec2<f32>(textureDimensions(dmap));
    let c = clamp(u, vec2<f32>(0.0), dims - vec2<f32>(1.0));
    let b = clamp(floor(c), vec2<f32>(0.0), max(dims - vec2<f32>(2.0), vec2<f32>(0.0)));
    let f = c - b;
    let bi = vec2<i32>(b);
    let a00 = map_at(bi);
    let a10 = map_at(bi + vec2<i32>(1, 0));
    let a01 = map_at(bi + vec2<i32>(0, 1));
    let a11 = map_at(bi + vec2<i32>(1, 1));
    return mix(mix(a00, a10, f.x), mix(a01, a11, f.x), f.y) + (u - c);
}

@vertex
fn vs_overlay(v: TriIn) -> ColorOut {
    let model = mat4x4<f32>(v.m0, v.m1, v.m2, v.m3);
    let c = (model * vec4<f32>(v.pos, 1.0)).xyz;
    let n = (model * vec4<f32>(v.normal, 0.0)).xyz;
    var o: ColorOut;
    // `xma_engine::view::world::camera_light`.
    var k = 0.4;
    let nl = length(n);
    if nl > 0.0 {
        k = k + 0.7 * abs(n.z / nl);
    }
    o.color = vec4<f32>(min(v.color.rgb * k, vec3<f32>(1.0)), 1.0);
    let z = c.z;
    // `Camera::projectPoint`: K * c, divided by its third value.
    let h = vec3<f32>(dot(cam.k0.xyz, c), dot(cam.k1.xyz, c), dot(cam.k2.xyz, c));
    if z <= 0.0 || h.z <= 0.0 {
        o.pos = vec4<f32>(0.0, 0.0, -1.0, 1.0);
        return o;
    }
    var p = h.xy / h.z;
    if cam.flags.x > 0.5 {
        p = distort(p);
    }
    let ndc = vec2<f32>((p.x + 0.5) / cam.size.x * 2.0 - 1.0, 1.0 - (p.y + 0.5) / cam.size.y * 2.0);
    let n_ = cam.size.z;
    let f_ = cam.size.w;
    let zgl = (f_ + n_) / (f_ - n_) - 2.0 * f_ * n_ / ((f_ - n_) * z);
    o.pos = vec4<f32>(ndc * z, (zgl + 1.0) * 0.5 * z, z);
    return o;
}

