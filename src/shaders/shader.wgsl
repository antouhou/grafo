struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) depth: f32,
    // Per-instance transform matrix columns (column-major)
    @location(3) t_col0: vec4<f32>,
    @location(4) t_col1: vec4<f32>,
    @location(5) t_col2: vec4<f32>,
    @location(6) t_col3: vec4<f32>,
};

struct VertexOutput {
    @invariant @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

// This is a struct that will be used for position normalization
struct Uniforms {
    canvas_size: vec2<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

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
    // NDC stands for Normalized Device Coordinates. It's a coordinate system where the visible area of the screen
    // is a cube with corners (-1, -1, -1) and (1, 1, 1).
    let ndc_x = 2.0 * input.position.x / uniforms.canvas_size.x - 1.0;
    let ndc_y = 1.0 - 2.0 * input.position.y / uniforms.canvas_size.y;
    // Build the transform matrix and apply it AFTER normalization so shapes keep their size
    let model: mat4x4<f32> = mat4x4<f32>(input.t_col0, input.t_col1, input.t_col2, input.t_col3);
    let clip_pos = vec4<f32>(ndc_x, ndc_y, input.depth, 1.0);
    output.position = model * clip_pos;
    output.color = input.color;
    return output;
}

@fragment
fn fs_main(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
    // Convert input color from sRGB to linear space that shader expects
    let linear_color = vec4<f32>(to_linear(color.rgb), color.a);

    return linear_color;
}
