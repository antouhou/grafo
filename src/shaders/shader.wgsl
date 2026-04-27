// Vertex input: per-vertex position/uv and per-instance color+transform
struct VertexInput {
    @location(0) position: vec2<f32>,
    // Per-instance solid color
    @location(1) color: vec4<f32>,
    // Optional texture coordinates for shape texturing
    @location(2) tex_coords: vec2<f32>,
    // Per-instance transform matrix columns (column-major layout)
    @location(3) t_col0: vec4<f32>,
    @location(4) t_col1: vec4<f32>,
    @location(5) t_col2: vec4<f32>,
    @location(6) t_col3: vec4<f32>,
    // Per-instance draw order for Z-fighting resolution
    @location(7) draw_order: f32,
    // AA: outward boundary normal in model space
    @location(8) normal: vec2<f32>,
    // AA: coverage factor (1.0 = interior, 0.0 = outer fringe)
    @location(9) coverage: f32,
    // Per-instance bitmask: bit 0 = layer 0 active, bit 1 = layer 1 active.
    // 0 = solid fill only (skip all texture samples).
    @location(10) texture_flags: f32,
    // Per-layer UV scale computed on the CPU from fit mode and texture dimensions.
    @location(11) texture_uv_scale_layer0: vec2<f32>,
    @location(12) texture_uv_scale_layer1: vec2<f32>,
};

struct VertexOutput {
    @invariant @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) layer0_tex_coords: vec2<f32>,
    @location(2) layer1_tex_coords: vec2<f32>,
    @location(3) coverage: f32,
    @location(4) @interpolate(flat) texture_flags: f32,
};

struct GradientVertexOutput {
    @invariant @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) layer0_tex_coords: vec2<f32>,
    @location(2) layer1_tex_coords: vec2<f32>,
    @location(3) coverage: f32,
    @location(4) @interpolate(flat) texture_flags: f32,
    // Model-space position for gradient evaluation (before transform)
    @location(5) model_pos: vec2<f32>,
    // Screen-space position (pixel coordinates, after transform)
    @location(6) screen_pos: vec2<f32>,
};

// This is a struct that will be used for position normalization
struct Uniforms {
    canvas_size: vec2<f32>,
    scale_factor: f32,
    /// AA fringe offset in physical pixels (default 0.5). Set to 0 to disable fringe.
    fringe_width: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
// Texture/sampler for optional shape texturing. A default white texture can be bound when unused.
// Layer 0 (background)
@group(1) @binding(0) var t_shape_layer0: texture_2d<f32>;
@group(1) @binding(1) var s_shape_layer0: sampler;
// Layer 1 (foreground/overlay)
@group(2) @binding(0) var t_shape_layer1: texture_2d<f32>;
@group(2) @binding(1) var s_shape_layer1: sampler;

// Gradient resources (group 3)
struct GradientParams {
    gradient_type: u32,   // 0=none, 1=linear, 2=radial, 3=conic
    spread_mode: u32,     // 0=pad, 1=repeat
    units: u32,           // 0=local (model space), 1=canvas (screen space)
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

@group(3) @binding(0) var<uniform> gradient_params: GradientParams;
@group(3) @binding(1) var t_gradient_ramp: texture_1d<f32>;
@group(3) @binding(2) var s_gradient_ramp: sampler;

fn to_linear(color: vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(0.04045);
    let higher = pow((color + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    let lower = color / vec3<f32>(12.92);
    return select(higher, lower, color <= cutoff);
}

fn to_srgb(color: vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(0.0031308);
    let higher = vec3<f32>(1.055) * pow(color, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    let lower = color * vec3<f32>(12.92);
    return select(higher, lower, color <= cutoff);
}

// ── Gradient evaluation ─────────────────────────────────────────────

/// Computes the raw gradient parameter t for the given position.
fn gradient_raw_t(pos: vec2<f32>) -> f32 {
    let gtype = gradient_params.gradient_type;
    if gtype == 1u {
        // Linear
        let d = gradient_params.linear_end - gradient_params.linear_start;
        let len_sq = dot(d, d);
        if len_sq < 1e-12 {
            return 0.0;
        }
        return dot(pos - gradient_params.linear_start, d) / len_sq;
    } else if gtype == 2u {
        // Radial (elliptical)
        let diff = pos - gradient_params.radial_center;
        let rx = gradient_params.radial_radius.x;
        let ry = gradient_params.radial_radius.y;
        if rx < 1e-6 || ry < 1e-6 {
            return 0.0;
        }
        let nx = diff.x / rx;
        let ny = diff.y / ry;
        return length(vec2<f32>(nx, ny));
    } else if gtype == 3u {
        // Conic
        let diff = pos - gradient_params.conic_center;
        var angle = atan2(diff.y, diff.x); // [-pi, pi]
        angle = angle - gradient_params.conic_start_angle;
        // Normalize to [0, 1)
        let tau = 6.283185307179586;
        angle = angle - floor(angle / tau) * tau;
        return angle / tau;
    }
    return 0.0;
}

/// Applies the spread mode (pad or repeat) and maps t to the ramp UV.
fn gradient_apply_spread(raw_t: f32) -> f32 {
    let period_start = gradient_params.period_start;
    let period_len = gradient_params.period_len;
    let ramp_start = gradient_params.ramp_start;
    let ramp_end = gradient_params.ramp_end;

    if period_len <= 0.0 {
        // Non-repeating: clamp to ramp domain
        let t_clamped = clamp(raw_t, ramp_start, ramp_end);
        if ramp_end <= ramp_start {
            return 0.5;
        }
        return (t_clamped - ramp_start) / (ramp_end - ramp_start);
    }

    // Repeating gradient: wrap into the period
    let spread = gradient_params.spread_mode;
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
    let gtype = gradient_params.gradient_type;
    if gtype == 0u {
        // No gradient — return transparent (caller uses solid fill)
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    if gradient_params.is_constant != 0u {
        return gradient_params.constant_color;
    }

    let pos = select(screen_pos, model_pos, gradient_params.units == 0u);
    let raw_t = gradient_raw_t(pos);
    let uv = gradient_apply_spread(raw_t);

    // Sample the 1D ramp texture. The ramp is pre-baked in linear premultiplied space.
    return textureSampleLevel(t_gradient_ramp, s_gradient_ramp, uv, 0.0);
}

fn compute_vertex_position(input: VertexInput) -> vec4<f32> {
    // Build the transform matrix from column-major CPU data.
    // Each vec4 (t_col0..t_col3) is one column of the matrix. WGSL's mat4x4
    // constructor treats each argument as a column, so this is a direct mapping.
    let model: mat4x4<f32> = mat4x4<f32>(input.t_col0, input.t_col1, input.t_col2, input.t_col3);

    // Apply the per-instance transform in pixel space
    let p = model * vec4<f32>(input.position, 0.0, 1.0);

    // Perform homogeneous divide to handle perspective projection
    let w = p.w;
    let invw = 1.0 / max(abs(w), 1e-6);
    let px = p.x * invw;
    let py = p.y * invw;
    let pz = p.z * invw;

    // AA fringe offset: push outer fringe vertices outward by 1 logical pixel in screen space.
    // Only applied to fringe vertices (coverage < 1.0). This ensures the fringe width is
    // uniform regardless of perspective transforms.
    var final_px = px;
    var final_py = py;

    if (input.coverage < 1.0) {
        // Transform a point slightly offset along the normal through the same model matrix
        let epsilon = 0.01;
        let p2 = model * vec4<f32>(input.position + input.normal * epsilon, 0.0, 1.0);
        let invw2 = 1.0 / max(abs(p2.w), 1e-6);
        let px2 = p2.x * invw2;
        let py2 = p2.y * invw2;

        // Screen-space direction of the normal
        let screen_dir = vec2<f32>(px2 - px, py2 - py);
        let screen_len = length(screen_dir);

        if (screen_len > 1e-8) {
            let unit_dir = screen_dir / screen_len;
            // Offset outward by the configured fringe width (in physical pixels).
            // This centers the AA band on the shape boundary, avoiding bloating thin features
            // (a 1px line stays ~2px instead of 3px with a full-pixel fringe).
            let fringe_width = uniforms.fringe_width / uniforms.scale_factor;
            final_px = px + unit_dir.x * fringe_width;
            final_py = py + unit_dir.y * fringe_width;
        }
    }

    // Then convert to NDC (Normalized Device Coordinates)
    // NDC is a cube with corners (-1, -1, -1) and (1, 1, 1).
    let ndc_x = 2.0 * final_px / uniforms.canvas_size.x - 1.0;
    let ndc_y = 1.0 - 2.0 * final_py / uniforms.canvas_size.y;
    // Map pz to [0, 1] depth range. Scale determines the Z range that maps to full depth.
    // Clamp to ensure we stay within valid depth bounds.
    // Larger Z -> smaller depth (closer to camera)
    let scale = 1000.0;  // Z range of [-scale, +scale] maps to [1, 0]
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
    // Later shapes (higher draw_order) get a smaller depth value (closer to camera).
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
    return vec4<f32>(ndc_x, ndc_y, biased_depth, 1.0);
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = compute_vertex_position(input);
    output.color = input.color;
    output.layer0_tex_coords = input.tex_coords * input.texture_uv_scale_layer0;
    output.layer1_tex_coords = input.tex_coords * input.texture_uv_scale_layer1;
    output.coverage = input.coverage;
    output.texture_flags = input.texture_flags;
    return output;
}

@vertex
fn vs_main_gradient(input: VertexInput) -> GradientVertexOutput {
    var output: GradientVertexOutput;
    output.position = compute_vertex_position(input);
    output.color = input.color;
    output.layer0_tex_coords = input.tex_coords * input.texture_uv_scale_layer0;
    output.layer1_tex_coords = input.tex_coords * input.texture_uv_scale_layer1;
    output.coverage = input.coverage;
    output.texture_flags = input.texture_flags;
    output.model_pos = input.position;

    let model: mat4x4<f32> = mat4x4<f32>(input.t_col0, input.t_col1, input.t_col2, input.t_col3);
    let p = model * vec4<f32>(input.position, 0.0, 1.0);
    let invw = 1.0 / max(abs(p.w), 1e-6);
    let px = p.x * invw;
    let py = p.y * invw;

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
            let fringe_width = uniforms.fringe_width / uniforms.scale_factor;
            final_px = px + unit_dir.x * fringe_width;
            final_py = py + unit_dir.y * fringe_width;
        }
    }

    output.screen_pos = vec2<f32>(final_px, final_py);
    return output;
}

fn texture_coords_are_in_bounds(tex_coords: vec2<f32>) -> bool {
    return all(tex_coords >= vec2<f32>(0.0, 0.0)) && all(tex_coords <= vec2<f32>(1.0, 1.0));
}

// Computes the final premultiplied color for a fragment given fill color, texture
// coordinates, and AA coverage.
fn compute_fragment_color(
    color: vec4<f32>,
    layer0_tex_coords: vec2<f32>,
    layer1_tex_coords: vec2<f32>,
    coverage: f32,
    texture_flags: f32,
) -> vec4<f32> {
    // Shape fill color arrives already in linear space (sRGB->linear conversion
    // is performed on the CPU in normalize_rgba_color).
    // Convert fill to premultiplied
    let fill_pma = vec4<f32>(color.rgb * color.a, color.a);

    // Fast path: no textures bound — solid fill only. Skip both texture samples.
    let flags = u32(texture_flags);
    if (flags == 0u) {
        return fill_pma * coverage;
    }

    // At least one texture layer is active.
    // Use textureSampleLevel (explicit LOD 0) instead of textureSample so that
    // sampling is valid inside non-uniform control flow. Our textures are created
    // without mipmaps (mip_level_count = 1), so LOD 0 is always correct.
    // Data is premultiplied (Rgba8UnormSrgb -> linear automatically).

    // Compose: base = texture layer 0 over shape fill, then layer 1 over result.
    var base_pma = fill_pma;
    if ((flags & 1u) != 0u) {
            let layer0_pma = textureSampleLevel(t_shape_layer0, s_shape_layer0, layer0_tex_coords, 0.0);
            base_pma = layer0_pma + fill_pma * (1.0 - layer0_pma.a);
    }

    var final_pma = base_pma;
    if ((flags & 2u) != 0u) {
            let layer1_pma = textureSampleLevel(t_shape_layer1, s_shape_layer1, layer1_tex_coords, 0.0);
            final_pma = layer1_pma + base_pma * (1.0 - layer1_pma.a);
    }

    // Apply AA coverage: scale premultiplied color by coverage factor.
    // With premultiplied alpha blending (src: One, dst: OneMinusSrcAlpha),
    // multiplying all four channels by coverage correctly fades the fringe to transparent.
    return final_pma * coverage;
}

fn compute_gradient_fragment_color(
    layer0_tex_coords: vec2<f32>,
    layer1_tex_coords: vec2<f32>,
    coverage: f32,
    texture_flags: f32,
    model_pos: vec2<f32>,
    screen_pos: vec2<f32>,
) -> vec4<f32> {
    let fill_pma = evaluate_gradient(model_pos, screen_pos);

    let flags = u32(texture_flags);
    if (flags == 0u) {
        return fill_pma * coverage;
    }

    var base_pma = fill_pma;
    if ((flags & 1u) != 0u) {
            let layer0_pma = textureSampleLevel(t_shape_layer0, s_shape_layer0, layer0_tex_coords, 0.0);
            base_pma = layer0_pma + fill_pma * (1.0 - layer0_pma.a);
    }

    var final_pma = base_pma;
    if ((flags & 2u) != 0u) {
            let layer1_pma = textureSampleLevel(t_shape_layer1, s_shape_layer1, layer1_tex_coords, 0.0);
            final_pma = layer1_pma + base_pma * (1.0 - layer1_pma.a);
    }

    return final_pma * coverage;
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
    );
}

// Used by stencil-only passes that write no color. Color work is skipped entirely;
// only the fixed-function stencil operation matters for these draws.
// NOTE: do not add discard here — that would also kill the stencil write.
@fragment
fn fs_stencil_only() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

// Used by stencil-mutating passes that still produce visible color output.
// Intentionally separate from fs_main: any future discard-based optimizations
// in fs_main must not suppress stencil writes on these passes. Do not merge with fs_main.
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
    );
}
