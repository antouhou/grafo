// Rasterizes a shape's coverage mask into an offscreen texture for shape mask effects.
// Expands boundary vertices along their normals before interpolation,
// converting the fringe width from physical pixels to logical units.

struct MaskUniforms {
    local_origin: vec2<f32>,
    logical_size: vec2<f32>,
    scale_factor: f32,
    fringe_width: f32,
    _padding: vec2<f32>,
};

struct MaskVertexInput {
    @location(0) position: vec2<f32>,
    @location(7) normal: vec2<f32>,
    @location(8) coverage: f32,
};

struct MaskVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) coverage: f32,
};

@group(0) @binding(0) var<uniform> uniforms: MaskUniforms;

@vertex
fn mask_vertex(input: MaskVertexInput) -> MaskVertexOutput {
    var local_position = input.position;
    if input.coverage < 1.0 {
        local_position += input.normal * (uniforms.fringe_width / uniforms.scale_factor);
    }

    let target_position = local_position - uniforms.local_origin;
    var output: MaskVertexOutput;
    output.position = vec4<f32>(
        2.0 * target_position.x / uniforms.logical_size.x - 1.0,
        1.0 - 2.0 * target_position.y / uniforms.logical_size.y,
        0.0,
        1.0,
    );
    output.coverage = input.coverage;
    return output;
}

@fragment
fn mask_fragment(input: MaskVertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.coverage);
}

struct CoverageMaskOutput {
    @builtin(position) position: vec4<f32>,
    @location(8) @interpolate(flat) coverage_start: u32,
    @location(9) @interpolate(flat) tile_origin: vec2<u32>,
};
@vertex
fn mask_coverage_vertex(@builtin(vertex_index) vertex_index: u32) -> CoverageMaskOutput {
    var output: CoverageMaskOutput;
    let vertex = coverage_geometry_vertex(vertex_index);
    output.position = vec4<f32>(2.0 * vertex.position.x / uniforms.logical_size.x - 1.0, 1.0 - 2.0 * vertex.position.y / uniforms.logical_size.y, 0.0, 1.0);
    output.coverage_start = vertex.coverage_start;
    output.tile_origin = vertex.tile_origin;
    return output;
}
@fragment
fn mask_coverage_fragment(input: CoverageMaskOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(resolve_path_coverage(1.0, input.coverage_start, vec2<f32>(input.tile_origin), input.position));
}
