struct Params {
    radius: f32,
    sigma: f32,
}
@group(1) @binding(0) var<uniform> params: Params;

@fragment
fn effect_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let pixel = DIRECTION / vec2<f32>(textureDimensions(t_input));
    let radius = i32(ceil(params.radius));
    let sigma = max(params.sigma, 0.001);

    var color = vec4<f32>(0.0);
    var total_weight = 0.0;
    for (var sample_offset = -radius; sample_offset <= radius; sample_offset++) {
        let offset = f32(sample_offset);
        let weight = exp(-(offset * offset) / (2.0 * sigma * sigma));
        color += textureSample(t_input, s_input, uv + pixel * offset) * weight;
        total_weight += weight;
    }
    return color / total_weight;
}
