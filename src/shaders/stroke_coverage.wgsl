override evaluates_path_coverage: bool = false;

@group(0) @binding(1) var stroke_coverage: texture_2d<f32>;

fn load_stroke_pixel(start: u32, origin: vec2<f32>, position: vec4<f32>) -> vec4<f32> {
    let atlas_origin = vec2<u32>(start & 0xffffu, start >> 16u);
    let pixel = vec2<u32>(floor(position.xy) - origin);
    return textureLoad(stroke_coverage, vec2<i32>(atlas_origin + pixel), 0);
}

fn resolve_path_coverage(start: u32, origin: vec2<f32>, position: vec4<f32>) -> f32 {
    if !evaluates_path_coverage { return 1.0; }
    let coverage = load_stroke_pixel(start, origin, position).r;
    if coverage == 0.0 { discard; }
    return coverage;
}

@fragment
fn fs_stroke_color(
    @builtin(position) fragment_position: vec4<f32>,
    @location(8) @interpolate(flat) coverage_start: u32,
    @location(9) @interpolate(flat) coverage_origin: vec2<f32>,
) -> @location(0) vec4<f32> {
    if !evaluates_path_coverage { return vec4<f32>(0.0); }
    return load_stroke_pixel(coverage_start, coverage_origin, fragment_position);
}
