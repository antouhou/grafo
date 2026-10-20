struct CoverageBatchShape {
    color: vec4<f32>,
    texture_uv_transform_layer0: vec4<f32>,
    texture_uv_transform_layer1: vec4<f32>,
    translation: vec2<f32>,
    texture_flags: f32,
    padding: f32,
};

@group(0) @binding(4) var<storage, read> coverage_batch_shapes: array<CoverageBatchShape>;

@vertex
fn vs_coverage_batch_solid(
    @location(0) tile_index: u32,
    @location(2) shape_index: u32,
    @builtin(vertex_index) corner: u32,
) -> SolidVertexOutput {
    let shape = coverage_batch_shapes[shape_index];
    let input = SolidCoverageVertexInput(shape.color, shape.translation);
    return coverage_solid_vertex(input, coverage_geometry_vertex(tile_index * 4u + corner));
}

@vertex
fn vs_coverage_batch(
    @location(0) tile_index: u32,
    @location(2) shape_index: u32,
    @builtin(vertex_index) corner: u32,
) -> VertexOutput {
    let shape = coverage_batch_shapes[shape_index];
    let input = CoverageInstanceInput(
        shape.color,
        shape.texture_flags,
        shape.texture_uv_transform_layer0,
        shape.texture_uv_transform_layer1,
        shape.translation,
    );
    let vertex = coverage_geometry_vertex(tile_index * 4u + corner);
    return coverage_vertex(input, vertex, coverage_coordinates(vertex));
}
