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
    @location(7) paint_col0: vec3<f32>,
    @location(8) paint_col1: vec3<f32>,
    @location(12) paint_col2: vec3<f32>,
    // Bits 0 and 1 activate texture layers 0 and 1, respectively
    // Zero skips texture sampling and uses only the solid fill
    @location(9) texture_flags: f32,
    // Per-layer UV transform. XY is scale and ZW is offset.
    @location(10) texture_uv_transform_layer0: vec4<f32>,
    @location(11) texture_uv_transform_layer1: vec4<f32>,
};

struct VertexOutput {
    @invariant @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(flat) layer0_uv_transform: vec4<f32>,
    @location(2) @interpolate(flat) layer1_uv_transform: vec4<f32>,
    @location(4) @interpolate(flat) texture_flags: f32,
    @location(5) shape_tex_coords: vec3<f32>,
    @location(8) @interpolate(flat) coverage_start: u32,
    @location(9) @interpolate(flat) coverage_origin: vec2<f32>,
};

struct GradientVertexOutput {
    @invariant @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(flat) layer0_uv_transform: vec4<f32>,
    @location(2) @interpolate(flat) layer1_uv_transform: vec4<f32>,
    @location(4) @interpolate(flat) texture_flags: f32,
    // Homogeneous position in the fill owner's paint coordinates
    @location(5) paint_position: vec3<f32>,
    // Screen position in pixels after the transform
    @location(6) screen_pos: vec2<f32>,
    @location(7) shape_tex_coords: vec3<f32>,
    @location(8) @interpolate(flat) coverage_start: u32,
    @location(9) @interpolate(flat) coverage_origin: vec2<f32>,
};

// Target dimensions in logical pixels, its physical origin, and the scene scale.
struct Uniforms {
    canvas_size: vec2<f32>,
    target_origin: vec2<f32>,
    scale_factor: f32,
    _padding: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
// Set per pipeline as clip-space z. compute_vertex_position fixes clip w at 1.
// Targets clear depth to 1. Direct overlapping stroke draws that keep stencil use
// DepthAction::RejectOverlap to write 0 with Less, so the first hit passes 0 < 1
// and later hits fail 0 < 0.
// The immediate Reset draw overrides this to 1 with Greater and color writes disabled,
// restoring marked samples for the next shape without changing stencil values.
// Other draws leave depth untouched. Shapes composite in command order.
// See DepthAction in execution/shapes/pipelines.rs and DrawPass::reset_shape_depth.
override tessellated_depth: f32 = 0.0;
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
    units: u32,           // 0=paint coordinates, 1=screen coordinates
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

/// Evaluates the gradient at the given paint and screen positions.
/// Returns a premultiplied linear RGBA color from the pre-baked ramp.
fn evaluate_gradient(paint_position: vec2<f32>, screen_pos: vec2<f32>) -> vec4<f32> {
    let gtype = material_params.gradient.gradient_type;
    if gtype == 0u {
        // Transparent if no gradient
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    if material_params.gradient.is_constant != 0u {
        return material_params.gradient.constant_color;
    }

    let pos = select(screen_pos, paint_position, material_params.gradient.units == 0u);
    let raw_t = gradient_raw_t(pos);
    let uv = gradient_apply_spread(raw_t);

    // Sample the 1D ramp texture. The ramp is pre-baked in linear premultiplied space.
    return textureSampleLevel(t_gradient_ramp, s_gradient_ramp, uv, 0.0);
}

struct VertexPosition {
    clip_position: vec4<f32>,
    screen_position: vec2<f32>,
};

fn compute_vertex_position(position: vec2<f32>, model: mat4x4<f32>) -> VertexPosition {
    let p = model * vec4<f32>(position, 0.0, 1.0);

    let w = p.w;
    let invw = 1.0 / max(abs(w), 1e-6);
    let px = p.x * invw;
    let py = p.y * invw;

    // Map screen coordinates to [-1, 1], with Y increasing upward.
    let target_position = vec2<f32>(px, py) - uniforms.target_origin / uniforms.scale_factor;
    let ndc_x = 2.0 * target_position.x / uniforms.canvas_size.x - 1.0;
    let ndc_y = 1.0 - 2.0 * target_position.y / uniforms.canvas_size.y;

    return VertexPosition(
        vec4<f32>(ndc_x, ndc_y, tessellated_depth, 1.0),
        vec2<f32>(px, py),
    );
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = compute_vertex_position(input.position, mat4x4<f32>(input.t_col0, input.t_col1, input.t_col2, input.t_col3)).clip_position;
    output.color = input.color;
    output.layer0_uv_transform = input.texture_uv_transform_layer0;
    output.layer1_uv_transform = input.texture_uv_transform_layer1;
    output.texture_flags = input.texture_flags;
    output.shape_tex_coords = vec3<f32>(input.tex_coords, 1.0);
    output.coverage_start = 0xffffffffu;
    output.coverage_origin = vec2<f32>(0.0);
    return output;
}

@vertex
fn vs_main_gradient(input: VertexInput) -> GradientVertexOutput {
    let position = compute_vertex_position(input.position, mat4x4<f32>(input.t_col0, input.t_col1, input.t_col2, input.t_col3));
    var output: GradientVertexOutput;
    output.position = position.clip_position;
    output.color = input.color;
    output.layer0_uv_transform = input.texture_uv_transform_layer0;
    output.layer1_uv_transform = input.texture_uv_transform_layer1;
    output.texture_flags = input.texture_flags;
    output.shape_tex_coords = vec3<f32>(input.tex_coords, 1.0);
    output.coverage_start = 0xffffffffu;
    output.coverage_origin = vec2<f32>(0.0);
    output.paint_position = mat3x3<f32>(input.paint_col0, input.paint_col1, input.paint_col2) * vec3<f32>(input.position, 1.0);

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
    shape_tex_coords: vec3<f32>,
    layer0_uv_transform: vec4<f32>,
    layer1_uv_transform: vec4<f32>,
    coverage: f32,
    texture_flags: f32,
) -> vec4<f32> {
    let flags = u32(texture_flags);
    if (flags == 0u) {
        return color_pma * coverage;
    }

    // Explicit LOD permits sampling in non-uniform control flow. Shape textures
    // have one mip level and contain premultiplied colors.
    let shape_uv = clamp(shape_tex_coords.xy / shape_tex_coords.z, vec2<f32>(0.0), vec2<f32>(1.0));
    let layer0_tex_coords = shape_uv * layer0_uv_transform.xy + layer0_uv_transform.zw;
    let layer1_tex_coords = shape_uv * layer1_uv_transform.xy + layer1_uv_transform.zw;
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
    shape_tex_coords: vec3<f32>,
    layer0_uv_transform: vec4<f32>,
    layer1_uv_transform: vec4<f32>,
    coverage: f32,
    texture_flags: f32,
) -> vec4<f32> {
    // The CPU converts the fill to linear RGB; premultiply it before compositing.
    let fill_pma = vec4<f32>(color.rgb * color.a, color.a);
    return composite_texture_layers(
        fill_pma, shape_tex_coords, layer0_uv_transform, layer1_uv_transform, coverage, texture_flags,
    );
}

fn compute_gradient_fragment_color(
    shape_tex_coords: vec3<f32>,
    layer0_uv_transform: vec4<f32>,
    layer1_uv_transform: vec4<f32>,
    coverage: f32,
    texture_flags: f32,
    paint_position: vec3<f32>,
    screen_pos: vec2<f32>,
    dither_coords: vec2<f32>,
) -> vec4<f32> {
    let fill_pma = apply_gradient_bayer_dither(
        evaluate_gradient(paint_position.xy / paint_position.z, screen_pos),
        dither_coords,
    );

    return composite_texture_layers(
        fill_pma, shape_tex_coords, layer0_uv_transform, layer1_uv_transform, coverage, texture_flags,
    );
}

fn composite_under_fill_texture(
    fill_pma: vec4<f32>,
    fragment_position: vec4<f32>,
    shape_tex_coords: vec3<f32>,
) -> vec4<f32> {
    let mapping = material_params.texture_sampling;
    let uses_target_coordinates = mapping.uses_target_coordinates != 0u;
    let position = select(shape_tex_coords.xy / shape_tex_coords.z, fragment_position.xy + uniforms.target_origin, uses_target_coordinates);
    let texture_coordinates = (position - mapping.origin) * mapping.inverse_size;
    let footprint = select(texture_footprint_coverage(texture_coordinates), 1.0, uses_target_coordinates);
    let texture_pma = textureSampleLevel(t_under_fill, s_under_fill, texture_coordinates, 0.0) * footprint;
    return fill_pma + texture_pma * (1.0 - fill_pma.a);
}

@fragment
fn fs_main(
    @builtin(position) fragment_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(flat) layer0_uv_transform: vec4<f32>,
    @location(2) @interpolate(flat) layer1_uv_transform: vec4<f32>,
    @location(4) @interpolate(flat) texture_flags: f32,
    @location(5) shape_tex_coords: vec3<f32>,
    @location(8) @interpolate(flat) coverage_start: u32,
    @location(9) @interpolate(flat) coverage_origin: vec2<f32>,
) -> @location(0) vec4<f32> {
    let pixel_coverage = resolve_path_coverage(coverage_start, coverage_origin, fragment_position);
    return compute_fragment_color(
        color,
        shape_tex_coords,
        layer0_uv_transform,
        layer1_uv_transform,
        pixel_coverage,
        texture_flags,
    );
}

@fragment
fn fs_main_gradient(
    @builtin(position) fragment_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(flat) layer0_uv_transform: vec4<f32>,
    @location(2) @interpolate(flat) layer1_uv_transform: vec4<f32>,
    @location(4) @interpolate(flat) texture_flags: f32,
    @location(5) paint_position: vec3<f32>,
    @location(6) screen_pos: vec2<f32>,
    @location(7) shape_tex_coords: vec3<f32>,
    @location(8) @interpolate(flat) coverage_start: u32,
    @location(9) @interpolate(flat) coverage_origin: vec2<f32>,
) -> @location(0) vec4<f32> {
    let pixel_coverage = resolve_path_coverage(coverage_start, coverage_origin, fragment_position);
    return compute_gradient_fragment_color(
        shape_tex_coords,
        layer0_uv_transform,
        layer1_uv_transform,
        pixel_coverage,
        texture_flags,
        paint_position,
        screen_pos,
        fragment_position.xy + uniforms.target_origin,
    );
}

// Coverage rejects empty pixels before stencil or depth updates.
@fragment
fn fs_no_color(
    @builtin(position) fragment_position: vec4<f32>,
    @location(8) @interpolate(flat) coverage_start: u32,
    @location(9) @interpolate(flat) coverage_origin: vec2<f32>,
) -> @location(0) vec4<f32> {
    let pixel_coverage = resolve_path_coverage(coverage_start, coverage_origin, fragment_position);
    return vec4<f32>(0.0);
}

@fragment
fn fs_texture_material(
    @builtin(position) fragment_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(flat) layer0_uv_transform: vec4<f32>,
    @location(2) @interpolate(flat) layer1_uv_transform: vec4<f32>,
    @location(4) @interpolate(flat) texture_flags: f32,
    @location(5) shape_tex_coords: vec3<f32>,
    @location(8) @interpolate(flat) coverage_start: u32,
    @location(9) @interpolate(flat) coverage_origin: vec2<f32>,
) -> @location(0) vec4<f32> {
    let pixel_coverage = resolve_path_coverage(coverage_start, coverage_origin, fragment_position);
    let fill_pma = vec4<f32>(color.rgb * color.a, color.a);
    let base_pma = composite_under_fill_texture(fill_pma, fragment_position, shape_tex_coords);
    return composite_texture_layers(base_pma, shape_tex_coords, layer0_uv_transform, layer1_uv_transform, pixel_coverage, texture_flags);
}

@fragment
fn fs_texture_material_gradient(
    @builtin(position) fragment_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(flat) layer0_uv_transform: vec4<f32>,
    @location(2) @interpolate(flat) layer1_uv_transform: vec4<f32>,
    @location(4) @interpolate(flat) texture_flags: f32,
    @location(5) paint_position: vec3<f32>,
    @location(6) screen_pos: vec2<f32>,
    @location(7) shape_tex_coords: vec3<f32>,
    @location(8) @interpolate(flat) coverage_start: u32,
    @location(9) @interpolate(flat) coverage_origin: vec2<f32>,
) -> @location(0) vec4<f32> {
    let pixel_coverage = resolve_path_coverage(coverage_start, coverage_origin, fragment_position);
    let fill_pma = apply_gradient_bayer_dither(evaluate_gradient(paint_position.xy / paint_position.z, screen_pos), fragment_position.xy + uniforms.target_origin);
    let base_pma = composite_under_fill_texture(fill_pma, fragment_position, shape_tex_coords);
    return composite_texture_layers(base_pma, shape_tex_coords, layer0_uv_transform, layer1_uv_transform, pixel_coverage, texture_flags);
}
