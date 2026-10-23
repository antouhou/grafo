struct SolidVertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(3) t_col0: vec4<f32>,
    @location(4) t_col1: vec4<f32>,
    @location(5) t_col2: vec4<f32>,
    @location(6) t_col3: vec4<f32>,
};

struct SolidCoverageVertexInput {
    @location(1) color: vec4<f32>,
    @location(13) coverage_translation: vec2<f32>,
};

struct SolidVertexOutput {
    @invariant @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(8) @interpolate(flat) coverage_start: u32,
    @location(9) @interpolate(flat) coverage_origin: vec2<f32>,
};

@vertex
fn vs_solid(input: SolidVertexInput) -> SolidVertexOutput {
    var output: SolidVertexOutput;
    let model = mat4x4<f32>(input.t_col0, input.t_col1, input.t_col2, input.t_col3);
    output.position = compute_vertex_position(input.position, model).clip_position;
    output.color = input.color;
    output.coverage_start = 0xffffffffu;
    output.coverage_origin = vec2<f32>(0.0);
    return output;
}

@vertex
fn vs_coverage_solid(input: SolidCoverageVertexInput, @builtin(vertex_index) vertex_index: u32) -> SolidVertexOutput {
    var output: SolidVertexOutput;
    let vertex = coverage_geometry_vertex(vertex_index);
    output.position = coverage_clip_position(vertex.position + input.coverage_translation);
    output.color = input.color;
    output.coverage_start = vertex.coverage_start;
    output.coverage_origin = input.coverage_translation + vec2<f32>(vertex.tile_origin) - uniforms.target_origin;
    return output;
}

@fragment
fn fs_solid(
    @builtin(position) fragment_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(8) @interpolate(flat) coverage_start: u32,
    @location(9) @interpolate(flat) coverage_origin: vec2<f32>,
) -> @location(0) vec4<f32> {
    let pixel_coverage = resolve_path_coverage(coverage_start, coverage_origin, fragment_position);
    return vec4<f32>(color.rgb * color.a, color.a) * pixel_coverage;
}
