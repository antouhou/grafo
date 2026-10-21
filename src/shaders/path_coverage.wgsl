override evaluates_path_coverage: bool = false;

@group(0) @binding(1) var<storage, read> path_coverage: array<f32>;

fn resolve_path_coverage(start: u32, origin: vec2<f32>, position: vec4<f32>) -> f32 {
    if !evaluates_path_coverage { return 1.0; }
    let pixel = vec2<u32>(floor(position.xy) - origin);
    let area = path_coverage[start + pixel.y * 8u + pixel.x];
    if area <= 0.0 { discard; }
    return area;
}
