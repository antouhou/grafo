struct PathCoverageRow {
    baseline: f32,
    edge_start: u32,
    edge_count: u32,
    tile_x: u32,
};

struct PathCoverageEdge {
    x0: f32,
    x1: f32,
    height: f32,
    inverse_width: f32,
};

@group(0) @binding(1) var<storage, read> path_coverage_rows: array<PathCoverageRow>;
@group(0) @binding(2) var<storage, read> path_coverage_edges: array<PathCoverageEdge>;

fn integrated_clamp(x: f32) -> f32 {
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return x - 0.5; }
    return 0.5 * x * x;
}

fn resolve_path_coverage(coverage: f32, row_start: u32, tile_y: u32, position: vec4<f32>) -> f32 {
    if row_start == 0xffffffffu { return coverage; }
    let row = path_coverage_rows[row_start + u32(position.y) - tile_y];
    let pixel_x = floor(position.x) - f32(row.tile_x);
    var area = row.baseline;
    for (var i = 0u; i < row.edge_count; i++) {
        let edge = path_coverage_edges[row.edge_start + i];
        let a = edge.x0 - pixel_x;
        let b = edge.x1 - pixel_x;
        if min(a, b) >= 1.0 {
            area += edge.height;
        } else if min(a, b) >= 0.0 && max(a, b) <= 1.0 {
            area += edge.height * (a + b) * 0.5;
        } else if max(a, b) > 0.0 {
            area += edge.height * (integrated_clamp(b) - integrated_clamp(a)) * edge.inverse_width;
        }
    }
    if area <= 0.0 { discard; }
    return clamp(area, 0.0, 1.0);
}
