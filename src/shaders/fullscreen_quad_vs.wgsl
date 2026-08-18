// Built-in vertex shader for drawing a fullscreen triangle (3 vertices, no vertex buffer).
// Used both by effect apply passes and the composite pass.

struct QuadOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_quad(@builtin(vertex_index) vi: u32) -> QuadOutput {
    // Fullscreen triangle trick: 3 vertices cover the entire screen
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    var out: QuadOutput;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}
