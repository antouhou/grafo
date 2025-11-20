// Vertex input: per-vertex position/order/uv and per-instance color+transform
struct VertexInput {
    @location(0) position: vec2<f32>,
    // Per-vertex render order value (not true geometric depth)
    @location(2) order: f32,
    // Per-instance solid color (was per-vertex before)
    @location(1) color: vec4<f32>,
    // Per-instance transform matrix columns (column-major)
    @location(3) t_col0: vec4<f32>,
    @location(4) t_col1: vec4<f32>,
    @location(5) t_col2: vec4<f32>,
    @location(6) t_col3: vec4<f32>,
    // Optional texture coordinates for shape texturing
    @location(7) tex_coords: vec2<f32>,
    // Per-instance camera perspective
    @location(8) camera_perspective: f32,
    // Per-instance viewport position
    @location(9) viewport_position: vec2<f32>,
};

struct VertexOutput {
    @invariant @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) tex_coords: vec2<f32>,
};

// This is a struct that will be used for position normalization
struct Uniforms {
    canvas_size: vec2<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
// Texture/sampler for optional shape texturing. A default white texture can be bound when unused.
// Layer 0 (background)
@group(1) @binding(0) var t_shape_layer0: texture_2d<f32>;
@group(1) @binding(1) var s_shape_layer0: sampler;
// Layer 1 (foreground/overlay)
@group(2) @binding(0) var t_shape_layer1: texture_2d<f32>;
@group(2) @binding(1) var s_shape_layer1: sampler;

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

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    // Build the transform matrix
    let model: mat4x4<f32> = mat4x4<f32>(input.t_col0, input.t_col1, input.t_col2, input.t_col3);

    // Apply the per-instance transform in pixel space first
    let p = model * vec4<f32>(input.position, 0.0, 1.0);
    
    // Apply perspective if specified (acting on the z-coordinate from transform)
    var w: f32;
    var px: f32 = p.x;
    var py: f32 = p.y;
    if input.camera_perspective > 0.0 {
        // CSS-style perspective: w' = 1 + z/d
        // w = 1.0 + p.z / input.camera_perspective;

        let cx = input.viewport_position.x; // canvas-space origin
        let cy = input.viewport_position.y;

        let x_rel = p.x - cx;
        let y_rel = p.y - cy;

        w = 1.0 + p.z / input.camera_perspective;
        let invw = 1.0 / max(abs(w), 1e-6);

        px = x_rel * invw + cx;
        py = y_rel * invw + cy;
    } else {
        // No perspective, use w from transform
        w = p.w;
    }
    
//    // Homogeneous divide to account for perspective. Clamp w to avoid infinities.
//    let invw = 1.0 / max(abs(w), 1e-6);
//    var px = p.x * invw; // pixel-space x after perspective
//    var py = p.y * invw; // pixel-space y after perspective
    
    // Apply viewport position in pixel space
//    px += input.viewport_position.x;
//    py += input.viewport_position.y;

    // Then convert to NDC (Normalized Device Coordinates)
    // NDC is a cube with corners (-1, -1, -1) and (1, 1, 1).
    let ndc_x = 2.0 * px / uniforms.canvas_size.x - 1.0;
    let ndc_y = 1.0 - 2.0 * py / uniforms.canvas_size.y;
    // Keep ordering controlled by the provided per-vertex order attribute.
    output.position = vec4<f32>(ndc_x, ndc_y, input.order, 1.0);
    output.color = input.color;
    output.tex_coords = input.tex_coords;
    return output;
}

@fragment
fn fs_main(@location(0) color: vec4<f32>, @location(1) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    // Convert shape fill color from sRGB to linear
    let fill_lin = vec4<f32>(to_linear(color.rgb), color.a);
    // Sample layer0 and layer1 (Rgba8UnormSrgb -> linear automatically). Data is premultiplied.
    let layer0_pma = textureSample(t_shape_layer0, s_shape_layer0, tex_coords);
    let layer1_pma = textureSample(t_shape_layer1, s_shape_layer1, tex_coords);
    // Convert fill to premultiplied
    let fill_pma = vec4<f32>(fill_lin.rgb * fill_lin.a, fill_lin.a);
    // Compose: base = texture layer 0 over shape fill, then layer 1 over result.
    let base_pma = layer0_pma + fill_pma * (1.0 - layer0_pma.a);
    let final_pma = layer1_pma + base_pma * (1.0 - layer1_pma.a);
    return final_pma;
}
