struct Uniforms {
    canvas_size: vec2<f32>,
    target_origin: vec2<f32>,
    scale_factor: f32,
    _padding: f32,
};

struct VertexPosition {
    clip_position: vec4<f32>,
    screen_position: vec2<f32>,
};

fn compute_vertex_position(
    position: vec2<f32>,
    model: mat4x4<f32>,
    target_uniforms: Uniforms,
) -> VertexPosition {
    let transformed = model * vec4<f32>(position, 0.0, 1.0);
    let inverse_w = 1.0 / max(abs(transformed.w), 1e-6);
    let screen_position = transformed.xy * inverse_w;
    let target_position = screen_position - target_uniforms.target_origin / target_uniforms.scale_factor;
    let normalized_x = 2.0 * target_position.x / target_uniforms.canvas_size.x - 1.0;
    let normalized_y = 1.0 - 2.0 * target_position.y / target_uniforms.canvas_size.y;
    return VertexPosition(
        vec4<f32>(normalized_x, normalized_y, 0.0, 1.0),
        screen_position,
    );
}
