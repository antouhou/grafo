struct StrokeRasterInput {
    @location(0) position: vec2<f32>,
    @location(3) transform_col0: vec4<f32>,
    @location(4) transform_col1: vec4<f32>,
    @location(5) transform_col3: vec4<f32>,
    @location(6) screen_to_clip: vec4<f32>,
    @location(7) scale: f32,
    @location(8) vertices_are_projected: u32,
};

fn stroke_screen_position(input: StrokeRasterInput) -> vec2<f32> {
    if input.vertices_are_projected != 0u {
        return input.position;
    }
    let point = input.transform_col0 * input.position.x
        + input.transform_col1 * input.position.y + input.transform_col3;
    return point.xy * (input.scale / max(abs(point.w), 1e-6));
}

fn stroke_clip_position(physical: vec2<f32>, screen_to_clip: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        (physical - screen_to_clip.zw) * screen_to_clip.xy + vec2<f32>(-1.0, 1.0),
        0.0,
        1.0,
    );
}

@vertex
fn vs_stroke_raster(input: StrokeRasterInput) -> @builtin(position) vec4<f32> {
    return stroke_clip_position(stroke_screen_position(input), input.screen_to_clip);
}

@fragment
fn fs_stroke_raster() -> @location(0) f32 {
    return 1.0;
}
