struct StrokeGradientVertex {
    @builtin(position) position: vec4<f32>,
    @location(0) local_position: vec2<f32>,
    @location(1) physical_position: vec2<f32>,
    @location(2) @interpolate(flat) scale: f32,
};

@vertex
fn vs_stroke_raster_gradient(
    input: StrokeRasterInput,
    @location(2) original_position: vec2<f32>,
) -> StrokeGradientVertex {
    let physical = stroke_screen_position(input);
    return StrokeGradientVertex(
        stroke_clip_position(physical, input.screen_to_clip),
        select(input.position, original_position, input.vertices_are_projected != 0u),
        physical,
        input.scale,
    );
}

@fragment
fn fs_stroke_raster_gradient(input: StrokeGradientVertex) -> @location(0) vec4<f32> {
    return apply_gradient_bayer_dither(
        evaluate_gradient(input.local_position, input.physical_position / input.scale),
        input.physical_position,
    );
}
