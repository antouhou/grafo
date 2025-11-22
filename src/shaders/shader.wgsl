// Vertex input: per-vertex position/uv and per-instance color+transform
struct VertexInput {
    @location(0) position: vec2<f32>,
    // Per-instance solid color
    @location(1) color: vec4<f32>,
    // Optional texture coordinates for shape texturing
    @location(2) tex_coords: vec2<f32>,
    // Per-instance transform matrix rows (row-major from euclid)
    @location(3) t_row0: vec4<f32>,
    @location(4) t_row1: vec4<f32>,
    @location(5) t_row2: vec4<f32>,
    @location(6) t_row3: vec4<f32>,
    // Per-instance draw order for Z-fighting resolution
    @location(7) draw_order: f32,
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
    // Build the transform matrix from row-major CPU data.
    // Euclid's to_arrays() gives us rows, and WGSL mat4x4 constructor treats each vec4 as a column.
    // So we pass rows as if they were columns, which gives us the transpose.
    // Then matrix * vector will work correctly for row-major semantics.
    let model: mat4x4<f32> = mat4x4<f32>(input.t_row0, input.t_row1, input.t_row2, input.t_row3);

    // Apply the per-instance transform in pixel space.
    let p = model * vec4<f32>(input.position, 0.0, 1.0);
    
    // Perform homogeneous divide to handle perspective projection
    let w = p.w;
    let invw = 1.0 / max(abs(w), 1e-6);
    let px = p.x * invw;
    let py = p.y * invw;
    let pz = p.z * invw;

    // Then convert to NDC (Normalized Device Coordinates)
    // NDC is a cube with corners (-1, -1, -1) and (1, 1, 1).
    let ndc_x = 2.0 * px / uniforms.canvas_size.x - 1.0;
    let ndc_y = 1.0 - 2.0 * py / uniforms.canvas_size.y;
    // Map pz to [0, 1] depth range. Scale determines the Z range that maps to full depth.
    // Clamp to ensure we stay within valid depth bounds.
    // Larger Z -> smaller depth (closer to camera)
    let scale = 1000.0;  // Z range of [-scale, +scale] maps to [1, 0]
    let depth = clamp(0.5 - pz / scale, 0.0, 1.0);
    
    // Apply a tiny depth bias based on draw order to resolve Z-fighting for coplanar shapes.
    // Later shapes (higher draw_order) get a smaller depth value (closer to camera).
    let bias = input.draw_order * 0.00001;
    let biased_depth = clamp(depth - bias, 0.0, 1.0);
    
    output.position = vec4<f32>(ndc_x, ndc_y, biased_depth, 1.0);
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
