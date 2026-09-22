// Per-vertex position and UV, with per-instance color and transform
struct VertexInput {
    @location(0) position: vec2<f32>,
    // Per-instance solid color
    @location(1) color: vec4<f32>,
    // Optional texture coordinates for shape texturing
    @location(2) tex_coords: vec2<f32>,
    // Per-instance transform in column-major order
    @location(3) t_col0: vec4<f32>,
    @location(4) t_col1: vec4<f32>,
    @location(5) t_col2: vec4<f32>,
    @location(6) t_col3: vec4<f32>,
    // Per-instance draw order for Z-fighting resolution
    @location(7) draw_order: f32,
    // Outward model-space normal for the AA fringe
    @location(8) normal: vec2<f32>,
    // AA coverage is 1.0 at the interior and 0.0 at the outer fringe
    @location(9) coverage: f32,
    // Bits 0 and 1 activate texture layers 0 and 1, respectively
    // Zero skips texture sampling and uses only the solid fill
    @location(10) texture_flags: f32,
    // Per-layer UV transform. XY is scale and ZW is offset.
    @location(11) texture_uv_transform_layer0: vec4<f32>,
    @location(12) texture_uv_transform_layer1: vec4<f32>,
};

struct VertexOutput {
    @invariant @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) layer0_tex_coords: vec2<f32>,
    @location(2) layer1_tex_coords: vec2<f32>,
    @location(3) coverage: f32,
    @location(4) @interpolate(flat) texture_flags: f32,
    @location(5) shape_tex_coords: vec2<f32>,
};

struct GradientVertexOutput {
    @invariant @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) layer0_tex_coords: vec2<f32>,
    @location(2) layer1_tex_coords: vec2<f32>,
    @location(3) coverage: f32,
    @location(4) @interpolate(flat) texture_flags: f32,
    // Model-space position before the transform, used to evaluate gradients
    @location(5) model_pos: vec2<f32>,
    // Screen position in pixels after the transform
    @location(6) screen_pos: vec2<f32>,
    @location(7) shape_tex_coords: vec2<f32>,
};

// Viewport dimensions and antialiasing settings from the renderer.
struct Uniforms {
    canvas_size: vec2<f32>,
    scale_factor: f32,
    // Outward AA fringe width in physical pixels. Zero disables the fringe.
    fringe_width: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
// Texture/sampler for optional shape texturing. A default white texture can be bound when unused.
// Background texture at layer 0
@group(1) @binding(0) var t_shape_layer0: texture_2d<f32>;
@group(1) @binding(1) var s_shape_layer0: sampler;
// Foreground texture at layer 1
@group(2) @binding(0) var t_shape_layer1: texture_2d<f32>;
@group(2) @binding(1) var s_shape_layer1: sampler;
// Shared material resources at group 3
struct GradientColorParams {
    gradient_type: u32,   // 0=none, 1=linear, 2=radial, 3=conic
    spread_mode: u32,     // 0=pad, 1=repeat
    units: u32,           // 0=model space, 1=screen space
    is_constant: u32,
    constant_color: vec4<f32>,
    linear_start: vec2<f32>,
    linear_end: vec2<f32>,
    radial_center: vec2<f32>,
    radial_radius: vec2<f32>,
    conic_center: vec2<f32>,
    conic_start_angle: f32,
    period_start: f32,
    period_len: f32,
    ramp_start: f32,
    ramp_end: f32,
    _padding: f32,
};

struct TextureSamplingParams {
    origin: vec2<f32>,
    inverse_size: vec2<f32>,
    uses_target_coordinates: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
};

struct MaterialParams {
    gradient: GradientColorParams,
    texture_sampling: TextureSamplingParams,
};

@group(3) @binding(0) var<uniform> material_params: MaterialParams;
@group(3) @binding(1) var t_gradient_ramp: texture_1d<f32>;
@group(3) @binding(2) var s_gradient_ramp: sampler;
// Optional material texture composited below the fill.
@group(3) @binding(3) var t_under_fill: texture_2d<f32>;
@group(3) @binding(4) var s_under_fill: sampler;

const BAYER_4X4_THRESHOLDS: array<f32, 16> = array<f32, 16>(
    0.0, 8.0, 2.0, 10.0,
    12.0, 4.0, 14.0, 6.0,
    3.0, 11.0, 1.0, 9.0,
    15.0, 7.0, 13.0, 5.0,
);

fn bayer_4x4_threshold(pixel_pos: vec2<f32>) -> f32 {
    let x = u32(pixel_pos.x) & 3u;
    let y = u32(pixel_pos.y) & 3u;
    return (BAYER_4X4_THRESHOLDS[y * 4u + x] + 0.5) * (1.0 / 16.0);
}

fn apply_gradient_bayer_dither(color_pma: vec4<f32>, pixel_pos: vec2<f32>) -> vec4<f32> {
    if color_pma.a <= 1e-6 {
        return color_pma;
    }

    let alpha = color_pma.a;
    let rgb = color_pma.rgb / alpha;
    let offset = (bayer_4x4_threshold(pixel_pos) - 0.5) * (1.0 / 255.0);
    let dithered_rgb = clamp(rgb + vec3<f32>(offset), vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(dithered_rgb * alpha, alpha);
}

// Gradient evaluation

/// Computes the raw gradient parameter t for the given position.
fn gradient_raw_t(pos: vec2<f32>) -> f32 {
    let gtype = material_params.gradient.gradient_type;
    if gtype == 1u {
        // Linear
        let d = material_params.gradient.linear_end - material_params.gradient.linear_start;
        let len_sq = dot(d, d);
        if len_sq < 1e-12 {
            return 0.0;
        }
        return dot(pos - material_params.gradient.linear_start, d) / len_sq;
    } else if gtype == 2u {
        // Elliptical radial gradient
        let diff = pos - material_params.gradient.radial_center;
        let rx = material_params.gradient.radial_radius.x;
        let ry = material_params.gradient.radial_radius.y;
        if rx < 1e-6 || ry < 1e-6 {
            return 0.0;
        }
        let nx = diff.x / rx;
        let ny = diff.y / ry;
        return length(vec2<f32>(nx, ny));
    } else if gtype == 3u {
        // Conic
        let diff = pos - material_params.gradient.conic_center;
        var angle = atan2(diff.y, diff.x); // [-pi, pi]
        angle = angle - material_params.gradient.conic_start_angle;
        // Normalize to [0, 1)
        let tau = 6.283185307179586;
        angle = angle - floor(angle / tau) * tau;
        return angle / tau;
    }
    return 0.0;
}

/// Pads or repeats t, then maps it to the ramp UV
fn gradient_apply_spread(raw_t: f32) -> f32 {
    let period_start = material_params.gradient.period_start;
    let period_len = material_params.gradient.period_len;
    let ramp_start = material_params.gradient.ramp_start;
    let ramp_end = material_params.gradient.ramp_end;

    if period_len <= 0.0 {
        // Clamp non-repeating gradients to the ramp domain
        let t_clamped = clamp(raw_t, ramp_start, ramp_end);
        if ramp_end <= ramp_start {
            return 0.5;
        }
        return (t_clamped - ramp_start) / (ramp_end - ramp_start);
    }

    // Repeating gradient: wrap into the period
    let spread = material_params.gradient.spread_mode;
    var t = raw_t;

    if spread == 1u {
        // Repeat
        t = period_start + ((t - period_start) - floor((t - period_start) / period_len) * period_len);
    } else {
        // Pad (clamp)
        t = clamp(t, ramp_start, ramp_end);
    }

    if ramp_end <= ramp_start {
        return 0.5;
    }
    return (t - ramp_start) / (ramp_end - ramp_start);
}

/// Evaluates the gradient at the given model and screen positions.
/// Returns a premultiplied linear RGBA color from the pre-baked ramp.
fn evaluate_gradient(model_pos: vec2<f32>, screen_pos: vec2<f32>) -> vec4<f32> {
    let gtype = material_params.gradient.gradient_type;
    if gtype == 0u {
        // Transparent if no gradient
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    if material_params.gradient.is_constant != 0u {
        return material_params.gradient.constant_color;
    }

    let pos = select(screen_pos, model_pos, material_params.gradient.units == 0u);
    let raw_t = gradient_raw_t(pos);
    let uv = gradient_apply_spread(raw_t);

    // Sample the 1D ramp texture. The ramp is pre-baked in linear premultiplied space.
    return textureSampleLevel(t_gradient_ramp, s_gradient_ramp, uv, 0.0);
}

struct VertexPosition {
    clip_position: vec4<f32>,
    screen_position: vec2<f32>,
};

fn compute_vertex_position(input: VertexInput) -> VertexPosition {
    // CPU transform columns map directly to WGSL matrix columns.
    let model: mat4x4<f32> = mat4x4<f32>(input.t_col0, input.t_col1, input.t_col2, input.t_col3);
    let p = model * vec4<f32>(input.position, 0.0, 1.0);

    let w = p.w;
    let invw = 1.0 / max(abs(w), 1e-6);
    let px = p.x * invw;
    let py = p.y * invw;
    let pz = p.z * invw;

    // Offset after projection so the AA fringe keeps its physical-pixel width.
    var final_px = px;
    var final_py = py;

    if (input.coverage < 1.0) {
        let epsilon = 0.01;
        let p2 = model * vec4<f32>(input.position + input.normal * epsilon, 0.0, 1.0);
        let invw2 = 1.0 / max(abs(p2.w), 1e-6);
        let px2 = p2.x * invw2;
        let py2 = p2.y * invw2;

        let screen_dir = vec2<f32>(px2 - px, py2 - py);
        let screen_len = length(screen_dir);

        if (screen_len > 1e-8) {
            let unit_dir = screen_dir / screen_len;
            // Convert the physical-pixel width to logical screen coordinates.
            let fringe_width = uniforms.fringe_width / uniforms.scale_factor;
            final_px = px + unit_dir.x * fringe_width;
            final_py = py + unit_dir.y * fringe_width;
        }
    }

    // Map screen coordinates to [-1, 1], with Y increasing upward.
    let ndc_x = 2.0 * final_px / uniforms.canvas_size.x - 1.0;
    let ndc_y = 1.0 - 2.0 * final_py / uniforms.canvas_size.y;
    // Map Z from [-scale / 2, scale / 2] to [1, 0], clamping values outside that range.
    let scale = 1000.0;
    var depth = clamp(0.5 - pz / scale, 0.0, 1.0);

    // TODO: a bit of a hacky hack to avoid intersection between shapes that do and shapes that doesn't use perspective.
    //  The basic idea is that shapes with pz=0 are the shapes that are likely don't use perspective, so we push them to
    //  the far plane. This is not a very good solution, and likely will cause some confusion in certain cases, for
    //  example when the user explicitly wants a shape to intersect another shape at z=0. I'm a bit too lazy to fix
    //  this properly right now, so leaving a TODO here.
    if pz == 0.0 {
        depth = 1.0; // Place at far plane if Z is exactly zero
    }

    // Apply a tiny depth bias based on draw order to resolve Z-fighting for coplanar shapes.
    // Higher draw_order moves the depth closer to the camera.
    let bias = input.draw_order * 0.00001;
    let biased_depth = clamp(depth - bias, 0.0, 1.0);

    // Biased depth here is a remnant of old code that used to actually do z sorting. I needed to add some transparency
    //  effects later on, and I figured that the easiest way would be just to disable depth compare function in the
    //  pipeline, and just use fs_main to do color compositing. That has one downside: as the compositing does not rely
    //  on the Z buffer, but rather on the draw order, if two shapes intersect, the one drawn later will
    //  always appear on top, even though part of it should be behind the other shape. A proper solution would be to implement
    //  some other algoritm to handle that, like depth peeling or weighted blended order-independent transparency, but
    //  I don't have a particular use case for it right now, so I'm leaving it as is.
    //  If you want to enable intersection without transparency, change the pipeline to enable depth test/write with
    //  less-equal function. (set depth_compare: wgpu::CompareFunction::LessEqual on the stencil/depth state)
    return VertexPosition(
        vec4<f32>(ndc_x, ndc_y, biased_depth, 1.0),
        vec2<f32>(final_px, final_py),
    );
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = compute_vertex_position(input).clip_position;
    output.color = input.color;
    output.layer0_tex_coords = input.tex_coords * input.texture_uv_transform_layer0.xy
        + input.texture_uv_transform_layer0.zw;
    output.layer1_tex_coords = input.tex_coords * input.texture_uv_transform_layer1.xy
        + input.texture_uv_transform_layer1.zw;
    output.coverage = input.coverage;
    output.texture_flags = input.texture_flags;
    output.shape_tex_coords = input.tex_coords;
    return output;
}

@vertex
fn vs_main_gradient(input: VertexInput) -> GradientVertexOutput {
    let position = compute_vertex_position(input);
    var output: GradientVertexOutput;
    output.position = position.clip_position;
    output.color = input.color;
    output.layer0_tex_coords = input.tex_coords * input.texture_uv_transform_layer0.xy
        + input.texture_uv_transform_layer0.zw;
    output.layer1_tex_coords = input.tex_coords * input.texture_uv_transform_layer1.xy
        + input.texture_uv_transform_layer1.zw;
    output.coverage = input.coverage;
    output.texture_flags = input.texture_flags;
    output.shape_tex_coords = input.tex_coords;
    output.model_pos = input.position;

    output.screen_pos = position.screen_position;
    return output;
}

fn texture_footprint_coverage(texture_coordinates: vec2<f32>) -> f32 {
    let minimum = vec2<f32>(0.0);
    let maximum = vec2<f32>(1.0);
    let is_inside = all(texture_coordinates >= minimum) && all(texture_coordinates <= maximum);
    return select(0.0, 1.0, is_inside);
}

fn composite_texture_layers(
    color_pma: vec4<f32>,
    layer0_tex_coords: vec2<f32>,
    layer1_tex_coords: vec2<f32>,
    coverage: f32,
    texture_flags: f32,
) -> vec4<f32> {
    let flags = u32(texture_flags);
    if (flags == 0u) {
        return color_pma * coverage;
    }

    // Explicit LOD permits sampling in non-uniform control flow. Shape textures
    // have one mip level and contain premultiplied colors.
    var base_pma = color_pma;
    if ((flags & 1u) != 0u) {
        let layer0_pma = textureSampleLevel(t_shape_layer0, s_shape_layer0, layer0_tex_coords, 0.0)
            * texture_footprint_coverage(layer0_tex_coords);
        base_pma = layer0_pma + color_pma * (1.0 - layer0_pma.a);
    }

    var final_pma = base_pma;
    if ((flags & 2u) != 0u) {
        let layer1_pma = textureSampleLevel(t_shape_layer1, s_shape_layer1, layer1_tex_coords, 0.0)
            * texture_footprint_coverage(layer1_tex_coords);
        final_pma = layer1_pma + base_pma * (1.0 - layer1_pma.a);
    }

    // Scale alpha and RGB together to preserve premultiplication at AA edges.
    return final_pma * coverage;
}

fn compute_fragment_color(
    color: vec4<f32>,
    layer0_tex_coords: vec2<f32>,
    layer1_tex_coords: vec2<f32>,
    coverage: f32,
    texture_flags: f32,
) -> vec4<f32> {
    // The CPU converts the fill to linear RGB; premultiply it before compositing.
    let fill_pma = vec4<f32>(color.rgb * color.a, color.a);
    return composite_texture_layers(
        fill_pma, layer0_tex_coords, layer1_tex_coords, coverage, texture_flags,
    );
}

fn compute_gradient_fragment_color(
    layer0_tex_coords: vec2<f32>,
    layer1_tex_coords: vec2<f32>,
    coverage: f32,
    texture_flags: f32,
    model_pos: vec2<f32>,
    screen_pos: vec2<f32>,
    dither_coords: vec2<f32>,
) -> vec4<f32> {
    let fill_pma = apply_gradient_bayer_dither(
        evaluate_gradient(model_pos, screen_pos),
        dither_coords,
    );

    return composite_texture_layers(
        fill_pma, layer0_tex_coords, layer1_tex_coords, coverage, texture_flags,
    );
}

fn composite_under_fill_texture(
    fill_pma: vec4<f32>,
    fragment_position: vec4<f32>,
    shape_tex_coords: vec2<f32>,
) -> vec4<f32> {
    let mapping = material_params.texture_sampling;
    let uses_target_coordinates = mapping.uses_target_coordinates != 0u;
    let position = select(shape_tex_coords, fragment_position.xy, uses_target_coordinates);
    let texture_coordinates = (position - mapping.origin) * mapping.inverse_size;
    let footprint = select(texture_footprint_coverage(texture_coordinates), 1.0, uses_target_coordinates);
    let texture_pma = textureSampleLevel(t_under_fill, s_under_fill, texture_coordinates, 0.0) * footprint;
    return fill_pma + texture_pma * (1.0 - fill_pma.a);
}

@fragment
fn fs_main(
    @location(0) color: vec4<f32>,
    @location(1) layer0_tex_coords: vec2<f32>,
    @location(2) layer1_tex_coords: vec2<f32>,
    @location(3) coverage: f32,
    @location(4) @interpolate(flat) texture_flags: f32,
) -> @location(0) vec4<f32> {
    return compute_fragment_color(
        color,
        layer0_tex_coords,
        layer1_tex_coords,
        coverage,
        texture_flags,
    );
}

@fragment
fn fs_main_gradient(
    @builtin(position) fragment_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) layer0_tex_coords: vec2<f32>,
    @location(2) layer1_tex_coords: vec2<f32>,
    @location(3) coverage: f32,
    @location(4) @interpolate(flat) texture_flags: f32,
    @location(5) model_pos: vec2<f32>,
    @location(6) screen_pos: vec2<f32>,
) -> @location(0) vec4<f32> {
    return compute_gradient_fragment_color(
        layer0_tex_coords,
        layer1_tex_coords,
        coverage,
        texture_flags,
        model_pos,
        screen_pos,
        fragment_position.xy,
    );
}

// Used by stencil-only passes that write no color. Color work is skipped entirely;
// only the fixed-function stencil operation matters for these draws.
// NOTE: do not add discard here. That would kill the stencil write.
@fragment
fn fs_stencil_only() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

// Used by stencil-mutating passes that still produce visible color output.
// Keep this separate from fs_main so adding discard there cannot suppress
// stencil writes here.
@fragment
fn fs_passthrough(
    @location(0) color: vec4<f32>,
    @location(1) layer0_tex_coords: vec2<f32>,
    @location(2) layer1_tex_coords: vec2<f32>,
    @location(3) coverage: f32,
    @location(4) @interpolate(flat) texture_flags: f32,
) -> @location(0) vec4<f32> {
    return compute_fragment_color(
        color,
        layer0_tex_coords,
        layer1_tex_coords,
        coverage,
        texture_flags,
    );
}

@fragment
fn fs_passthrough_gradient(
    @builtin(position) fragment_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) layer0_tex_coords: vec2<f32>,
    @location(2) layer1_tex_coords: vec2<f32>,
    @location(3) coverage: f32,
    @location(4) @interpolate(flat) texture_flags: f32,
    @location(5) model_pos: vec2<f32>,
    @location(6) screen_pos: vec2<f32>,
) -> @location(0) vec4<f32> {
    return compute_gradient_fragment_color(
        layer0_tex_coords,
        layer1_tex_coords,
        coverage,
        texture_flags,
        model_pos,
        screen_pos,
        fragment_position.xy,
    );
}

@fragment
fn fs_texture_material(
    @builtin(position) fragment_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) layer0_tex_coords: vec2<f32>,
    @location(2) layer1_tex_coords: vec2<f32>,
    @location(3) coverage: f32,
    @location(4) @interpolate(flat) texture_flags: f32,
    @location(5) shape_tex_coords: vec2<f32>,
) -> @location(0) vec4<f32> {
    let fill_pma = vec4<f32>(color.rgb * color.a, color.a);
    let base_pma = composite_under_fill_texture(fill_pma, fragment_position, shape_tex_coords);
    return composite_texture_layers(base_pma, layer0_tex_coords, layer1_tex_coords, coverage, texture_flags);
}

@fragment
fn fs_texture_material_gradient(
    @builtin(position) fragment_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) layer0_tex_coords: vec2<f32>,
    @location(2) layer1_tex_coords: vec2<f32>,
    @location(3) coverage: f32,
    @location(4) @interpolate(flat) texture_flags: f32,
    @location(5) model_pos: vec2<f32>,
    @location(6) screen_pos: vec2<f32>,
    @location(7) shape_tex_coords: vec2<f32>,
) -> @location(0) vec4<f32> {
    let fill_pma = apply_gradient_bayer_dither(evaluate_gradient(model_pos, screen_pos), fragment_position.xy);
    let base_pma = composite_under_fill_texture(fill_pma, fragment_position, shape_tex_coords);
    return composite_texture_layers(base_pma, layer0_tex_coords, layer1_tex_coords, coverage, texture_flags);
}
