// 3D drawing for the world view and the camera views' mesh overlay
// (`widgets/scene3d/mod.rs`). Colours are OpenGL's: values written to an 8-bit
// framebuffer without gamma conversion; the blit converts when egui's target is sRGB.
//
// Depth: the OpenGL projections give clip z in [-w, w]; wgpu wants [0, w], so every
// vertex stage maps z to (z + w) / 2.

// ---------------------------------------------------------------------------------------
// World view

struct World {
    // World to eye (gluLookAt and the two glRotated).
    view: mat4x4<f32>,
    // Eye to clip, OpenGL convention (gluPerspective).
    proj: mat4x4<f32>,
    // Target size in pixels.
    viewport: vec2<f32>,
    // Line width in pixels (glLineWidth 2.5).
    line_width: f32,
    pad: f32,
};

@group(0) @binding(0) var<uniform> w: World;

fn gl_to_wgpu(c: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(c.x, c.y, (c.z + c.w) * 0.5, c.w);
}

// The world view's lighting (`xma_engine::view::world::world_light`): global ambient 0.2,
// light 1 ambient 0.3 and diffuse 0.5 at eye position (0, 10, 0).
fn world_light(color: vec4<f32>, n_eye: vec3<f32>, p_eye: vec3<f32>) -> vec4<f32> {
    var k = 0.5;
    let nl = length(n_eye);
    let l = vec3<f32>(0.0, 10.0, 0.0) - p_eye;
    let ll = length(l);
    if nl > 0.0 && ll > 0.0 {
        k = k + 0.5 * max(dot(n_eye / nl, l / ll), 0.0);
    }
    return vec4<f32>(min(color.rgb * k, vec3<f32>(1.0)), color.a);
}

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

@vertex
fn vs_world_tri(v: TriIn) -> ColorOut {
    let model = mat4x4<f32>(v.m0, v.m1, v.m2, v.m3);
    let eye = w.view * model * vec4<f32>(v.pos, 1.0);
    let n = (w.view * model * vec4<f32>(v.normal, 0.0)).xyz;
    var o: ColorOut;
    o.pos = gl_to_wgpu(w.proj * eye);
    o.color = world_light(v.color, n, eye.xyz);
    return o;
}

@fragment
fn fs_color(i: ColorOut) -> @location(0) vec4<f32> {
    return i.color;
}

// Lines as screen-space quads: two triangles per segment, `line_width` pixels wide.
struct LineIn {
    @builtin(vertex_index) vi: u32,
    @location(0) a: vec3<f32>,
    @location(1) b: vec3<f32>,
    @location(2) color: vec4<f32>,
    // x: 1 for a stippled line (glLineStipple(10, 0xAAAA)).
    @location(3) style: vec4<f32>,
};

struct LineOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    // Pixels from the start of the segment.
    @location(1) along: f32,
    @location(2) dashed: f32,
};

@vertex
fn vs_world_line(v: LineIn) -> LineOut {
    var ca = w.proj * w.view * vec4<f32>(v.a, 1.0);
    var cb = w.proj * w.view * vec4<f32>(v.b, 1.0);
    var o: LineOut;
    o.color = vec4<f32>(min(v.color.rgb * 0.5, vec3<f32>(1.0)), v.color.a);
    o.dashed = v.style.x;
    // Clip against the near plane (z = -w).
    let da = ca.z + ca.w;
    let db = cb.z + cb.w;
    if da < 0.0 && db < 0.0 {
        o.pos = vec4<f32>(0.0, 0.0, -1.0, 1.0);
        o.along = 0.0;
        return o;
    }
    if da < 0.0 {
        ca = ca + (cb - ca) * (da / (da - db));
    }
    if db < 0.0 {
        cb = cb + (ca - cb) * (db / (db - da));
    }
    let sa = ca.xy / ca.w * 0.5 * w.viewport;
    let sb = cb.xy / cb.w * 0.5 * w.viewport;
    var d = sb - sa;
    let len = length(d);
    if len > 0.0 {
        d = d / len;
    } else {
        d = vec2<f32>(1.0, 0.0);
    }
    let n = vec2<f32>(-d.y, d.x) * (0.5 * w.line_width);
    // Corners: 0 a-, 1 a+, 2 b-, 3 b+; triangles (0, 1, 2), (2, 1, 3).
    var corner = array<u32, 6>(0u, 1u, 2u, 2u, 1u, 3u);
    let c = corner[v.vi];
    var clip = ca;
    var off = -n;
    o.along = 0.0;
    if c == 1u || c == 3u {
        off = n;
    }
    if c >= 2u {
        clip = cb;
        o.along = len;
    }
    let pix = off / (0.5 * w.viewport) * clip.w;
    o.pos = gl_to_wgpu(vec4<f32>(clip.xy + pix, clip.z, clip.w));
    return o;
}

@fragment
fn fs_world_line(i: LineOut) -> @location(0) vec4<f32> {
    if i.dashed > 0.5 {
        // 0xAAAA with factor 10: bit 0 (off) first, 10 pixels per bit.
        if fract(i.along / 20.0) < 0.5 {
            discard;
        }
    }
    return i.color;
}

// Camera images on the focal plane: the store's integer texture, grey scaled as the
// image view does at full depth, modulated by the ambient-only lighting (0.5).
@group(1) @binding(0) var img: texture_2d<u32>;

struct QuadIn {
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    // x: 1 with a texture; y: 1 for an RGBA texture; z: 1 / (2^shift * 255).
    @location(2) info: vec4<f32>,
};

struct QuadOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) info: vec4<f32>,
};

@vertex
fn vs_world_quad(v: QuadIn) -> QuadOut {
    var o: QuadOut;
    o.pos = gl_to_wgpu(w.proj * w.view * vec4<f32>(v.pos, 1.0));
    o.uv = v.uv;
    o.info = v.info;
    return o;
}

@fragment
fn fs_world_quad(i: QuadOut) -> @location(0) vec4<f32> {
    if i.info.x < 0.5 {
        return vec4<f32>(0.5, 0.5, 0.5, 1.0);
    }
    let dims = vec2<i32>(textureDimensions(img));
    let p = clamp(vec2<i32>(floor(i.uv * vec2<f32>(dims))), vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
    let t = textureLoad(img, p, 0);
    var c: vec3<f32>;
    if i.info.y > 0.5 {
        c = vec3<f32>(f32(t.r), f32(t.g), f32(t.b)) / 255.0;
    } else {
        let g = min(f32(t.r) * i.info.z, 1.0);
        c = vec3<f32>(g, g, g);
    }
    return vec4<f32>(c * 0.5, 1.0);
}

