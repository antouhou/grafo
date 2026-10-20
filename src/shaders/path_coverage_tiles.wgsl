struct CoverageTile {
    min: vec2<u32>,
    max: vec2<u32>,
    coverage_start: u32,
    projection_index: u32,
};

struct CoverageProjection {
    model_origin: vec3<f32>,
    screen_origin_x: f32,
    model_dx: vec3<f32>,
    screen_origin_y: f32,
    model_dy: vec3<f32>,
    padding: f32,
    local_origin: vec2<f32>,
    local_size: vec2<f32>,
};

@group(0) @binding(2) var<storage, read> coverage_tiles: array<CoverageTile>;
@group(0) @binding(3) var<storage, read> coverage_projections: array<CoverageProjection>;

struct CoverageGeometryVertex {
    position: vec2<f32>,
    coverage_start: u32,
    tile_origin: vec2<u32>,
    projection_index: u32,
};

fn coverage_geometry_vertex(vertex_index: u32) -> CoverageGeometryVertex {
    let tile = coverage_tiles[vertex_index / 4u];
    let corner = vertex_index % 4u;
    var output: CoverageGeometryVertex;
    output.position = vec2<f32>(
        f32(select(tile.min.x, tile.max.x, corner == 1u || corner == 2u)),
        f32(select(tile.min.y, tile.max.y, corner >= 2u)),
    );
    output.coverage_start = tile.coverage_start;
    output.tile_origin = tile.min;
    output.projection_index = tile.projection_index;
    return output;
}
