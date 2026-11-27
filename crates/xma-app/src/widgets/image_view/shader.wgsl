// Image view shader: one image held as an unsigned-integer texture
// (R8Uint, R16Uint or Rgba8Uint) and read with textureLoad, so raw values stay
// exact. Filtering is done here, because integer textures cannot be sampled
// with a filtering sampler.
//
// Coordinates: the paint callback sets the viewport to the whole framebuffer,
// so in.uv * u.screen is the framebuffer pixel position (pixel centres at +0.5),
// and image = u.origin + framebuffer_px * u.zoom. Image pixel i covers
// [i - 0.5, i + 0.5] (centres at integers), as in GLCameraView.cpp.

struct Uniforms {
    // Image coordinate at the top-left corner of the framebuffer.
    origin: vec2<f32>,
    // Framebuffer size in pixels.
    screen: vec2<f32>,
    // Image pixels per device pixel (GLCameraView zoomRatio).
    zoom: f32,
    scale: f32,
    bias: f32,
    // 1 / (2^shift * 255): raw grey value to [0, 1] (full depth display).
    inv_white: f32,
    // 0 = nearest, 1 = bilinear.
    filter_mode: u32,
    // bit 0: sRGB target; bit 1: RGBA texture; bit 2: reduce to 8 bits first;
    // bit 3: paint the background outside the image (else leave it untouched).
    flags: u32,
    // bit depth - 8.
    shift: u32,
    pad: u32,
    background: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(1) @binding(0) var img: texture_2d<u32>;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VOut {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    let xy = p[i];
    var o: VOut;
    o.pos = vec4<f32>(xy, 0.0, 1.0);
    // uv (0,0) = top-left of the viewport, (1,1) = bottom-right.
    o.uv = vec2<f32>(xy.x * 0.5 + 0.5, 0.5 - xy.y * 0.5);
    return o;
}

// OpenCV convertTo(CV_8U, 1 / 2^shift): round half to even, saturate at 255.
fn to8(x: u32) -> u32 {
    let s = u.shift;
    if s == 0u {
        return min(x, 255u);
    }
    var q = x >> s;
    let r = x & ((1u << s) - 1u);
    let h = 1u << (s - 1u);
    if r > h || (r == h && (q & 1u) == 1u) {
        q = q + 1u;
    }
    return min(q, 255u);
}

fn grey(x: u32) -> f32 {
    if (u.flags & 4u) != 0u {
        return f32(to8(x)) / 255.0;
    }
    return min(f32(x) * u.inv_white, 1.0);
}

fn load(c: vec2<i32>, maxc: vec2<i32>) -> vec3<f32> {
    let v = textureLoad(img, clamp(c, vec2<i32>(0, 0), maxc), 0);
    if (u.flags & 2u) != 0u {
        return vec3<f32>(f32(v.r), f32(v.g), f32(v.b)) / 255.0;
    }
    let g = grey(v.r);
    return vec3<f32>(g, g, g);
}

// The result of GLCameraView's blend passes (paintGL): for scale > 1 the image
// is multiplied (clamped to [0, 1] in the framebuffer) and then the bias added;
// otherwise image * scale + bias in one pass. glColor clamps scale to [0, 1].
fn adjust(v: vec3<f32>) -> vec3<f32> {
    if u.scale > 1.0 {
        return clamp(clamp(v * u.scale, vec3<f32>(0.0), vec3<f32>(1.0)) + vec3<f32>(u.bias), vec3<f32>(0.0), vec3<f32>(1.0));
    }
    return clamp(v * max(u.scale, 0.0) + vec3<f32>(u.bias), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        return c / 12.92;
    }
    return pow((c + 0.055) / 1.055, 2.4);
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let fb = in.uv * u.screen;
    let p = u.origin + fb * u.zoom;
    let dims = vec2<f32>(textureDimensions(img));
    if p.x < -0.5 || p.y < -0.5 || p.x >= dims.x - 0.5 || p.y >= dims.y - 0.5 {
        if (u.flags & 8u) == 0u {
            discard;
        }
        return u.background;
    }
    let maxc = vec2<i32>(dims) - vec2<i32>(1, 1);
    var v: vec3<f32>;
    if u.filter_mode == 0u {
        v = load(vec2<i32>(floor(p + vec2<f32>(0.5, 0.5))), maxc);
    } else {
        let b = floor(p);
        let f = p - b;
        let bi = vec2<i32>(b);
        let a00 = load(bi, maxc);
        let a10 = load(bi + vec2<i32>(1, 0), maxc);
        let a01 = load(bi + vec2<i32>(0, 1), maxc);
        let a11 = load(bi + vec2<i32>(1, 1), maxc);
        v = mix(mix(a00, a10, f.x), mix(a01, a11, f.x), f.y);
    }
    var g = adjust(v);
    if (u.flags & 1u) != 0u {
        g = vec3<f32>(srgb_to_linear(g.r), srgb_to_linear(g.g), srgb_to_linear(g.b));
    }
    return vec4<f32>(g, 1.0);
}
