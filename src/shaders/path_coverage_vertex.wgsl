struct CoverageVertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) texture_position: vec3<f32>,
    @location(7) model_position: vec3<f32>,
    @location(8) row_start: u32,
    @location(9) texture_flags: f32,
    @location(10) texture_uv_transform_layer0: vec4<f32>,
    @location(11) texture_uv_transform_layer1: vec4<f32>,
    @location(12) tile_y: u32,
};

fn coverage_vertex(input: CoverageVertexInput) -> VertexOutput {
    var output: VertexOutput;
    let size = uniforms.canvas_size * uniforms.scale_factor;
    output.position = vec4<f32>(2.0 * input.position.x / size.x - 1.0, 1.0 - 2.0 * input.position.y / size.y, 0.0, 1.0);
    output.color = input.color;
    let weight = input.texture_position.z;
    output.layer0_tex_coords = vec3<f32>(input.texture_position.xy * input.texture_uv_transform_layer0.xy + input.texture_uv_transform_layer0.zw * weight, weight);
    output.layer1_tex_coords = vec3<f32>(input.texture_position.xy * input.texture_uv_transform_layer1.xy + input.texture_uv_transform_layer1.zw * weight, weight);
    output.coverage = 1.0;
    output.texture_flags = input.texture_flags;
    output.shape_tex_coords = input.texture_position;
    output.coverage_row_start = input.row_start;
    output.coverage_tile_y = input.tile_y;
    return output;
}

@vertex
fn vs_coverage_gradient(input: CoverageVertexInput) -> GradientVertexOutput {
    let base = coverage_vertex(input);
    var output: GradientVertexOutput;
    output.position = base.position;
    output.color = base.color;
    output.layer0_tex_coords = base.layer0_tex_coords;
    output.layer1_tex_coords = base.layer1_tex_coords;
    output.coverage = base.coverage;
    output.texture_flags = base.texture_flags;
    output.model_pos = input.model_position;
    output.screen_pos = input.position / uniforms.scale_factor;
    output.shape_tex_coords = base.shape_tex_coords;
    output.coverage_row_start = base.coverage_row_start;
    output.coverage_tile_y = base.coverage_tile_y;
    return output;
}

@vertex
fn vs_coverage(input: CoverageVertexInput) -> VertexOutput {
    return coverage_vertex(input);
}
