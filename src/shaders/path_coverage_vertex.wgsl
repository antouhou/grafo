struct CoverageInstanceInput {
    @location(1) color: vec4<f32>,
    @location(9) texture_flags: f32,
    @location(10) texture_uv_transform_layer0: vec4<f32>,
    @location(11) texture_uv_transform_layer1: vec4<f32>,
    @location(13) coverage_translation: vec2<f32>,
};

struct CoverageCoordinates {
    model_position: vec3<f32>,
    texture_position: vec3<f32>,
};

fn coverage_coordinates(vertex: CoverageGeometryVertex) -> CoverageCoordinates {
    let mapping = coverage_projections[vertex.projection_index];
    let delta = vertex.position - vec2<f32>(mapping.screen_origin_x, mapping.screen_origin_y);
    let model = mapping.model_origin + mapping.model_dx * delta.x + mapping.model_dy * delta.y;
    var output: CoverageCoordinates;
    output.model_position = model;
    output.texture_position = vec3<f32>((model.xy - mapping.local_origin * model.z) / mapping.local_size, model.z);
    return output;
}

fn coverage_clip_position(position: vec2<f32>) -> vec4<f32> {
    let size = uniforms.canvas_size * uniforms.scale_factor;
    return vec4<f32>(2.0 * position.x / size.x - 1.0, 1.0 - 2.0 * position.y / size.y, 0.0, 1.0);
}

fn coverage_vertex(input: CoverageInstanceInput, vertex: CoverageGeometryVertex, coordinates: CoverageCoordinates) -> VertexOutput {
    var output: VertexOutput;
    output.position = coverage_clip_position(vertex.position + input.coverage_translation);
    output.color = input.color;
    let texture_position = coordinates.texture_position;
    let weight = texture_position.z;
    output.layer0_tex_coords = vec3<f32>(texture_position.xy * input.texture_uv_transform_layer0.xy + input.texture_uv_transform_layer0.zw * weight, weight);
    output.layer1_tex_coords = vec3<f32>(texture_position.xy * input.texture_uv_transform_layer1.xy + input.texture_uv_transform_layer1.zw * weight, weight);
    output.coverage = 1.0;
    output.texture_flags = input.texture_flags;
    output.shape_tex_coords = texture_position;
    output.coverage_start = vertex.coverage_start;
    output.coverage_origin = input.coverage_translation + vec2<f32>(vertex.tile_origin);
    return output;
}

@vertex
fn vs_coverage_gradient(input: CoverageInstanceInput, @builtin(vertex_index) vertex_index: u32) -> GradientVertexOutput {
    let vertex = coverage_geometry_vertex(vertex_index);
    let coordinates = coverage_coordinates(vertex);
    let base = coverage_vertex(input, vertex, coordinates);
    var output: GradientVertexOutput;
    output.position = base.position;
    output.color = base.color;
    output.layer0_tex_coords = base.layer0_tex_coords;
    output.layer1_tex_coords = base.layer1_tex_coords;
    output.coverage = base.coverage;
    output.texture_flags = base.texture_flags;
    output.model_pos = coordinates.model_position;
    output.screen_pos = (vertex.position + input.coverage_translation) / uniforms.scale_factor;
    output.shape_tex_coords = base.shape_tex_coords;
    output.coverage_start = base.coverage_start;
    output.coverage_origin = base.coverage_origin;
    return output;
}

@vertex
fn vs_coverage(input: CoverageInstanceInput, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let vertex = coverage_geometry_vertex(vertex_index);
    return coverage_vertex(input, vertex, coverage_coordinates(vertex));
}
