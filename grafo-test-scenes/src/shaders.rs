/// Horizontal separable Gaussian blur pass (direction = (1,0)).
pub const HORIZONTAL_BLUR_WGSL: &str = r#"
const DIRECTION: vec2<f32> = vec2<f32>(1.0, 0.0);

struct Params {
    radius: f32,
    _pad: f32,
    tex_size: vec2<f32>,
}
@group(1) @binding(0) var<uniform> params: Params;

@fragment
fn effect_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let pixel = DIRECTION / params.tex_size;
    let sigma = max(params.radius / 3.0, 0.001);
    var color = vec4<f32>(0.0);
    var total_weight = 0.0;
    let r = i32(ceil(params.radius));
    for (var i = -r; i <= r; i++) {
        let offset = f32(i);
        let weight = exp(-(offset * offset) / (2.0 * sigma * sigma));
        color += textureSample(t_input, s_input, uv + pixel * offset) * weight;
        total_weight += weight;
    }
    return color / total_weight;
}
"#;

/// Vertical separable Gaussian blur pass (direction = (0,1)).
pub const VERTICAL_BLUR_WGSL: &str = r#"
const DIRECTION: vec2<f32> = vec2<f32>(0.0, 1.0);

struct Params {
    radius: f32,
    _pad: f32,
    tex_size: vec2<f32>,
}
@group(1) @binding(0) var<uniform> params: Params;

@fragment
fn effect_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let pixel = DIRECTION / params.tex_size;
    let sigma = max(params.radius / 3.0, 0.001);
    var color = vec4<f32>(0.0);
    var total_weight = 0.0;
    let r = i32(ceil(params.radius));
    for (var i = -r; i <= r; i++) {
        let offset = f32(i);
        let weight = exp(-(offset * offset) / (2.0 * sigma * sigma));
        color += textureSample(t_input, s_input, uv + pixel * offset) * weight;
        total_weight += weight;
    }
    return color / total_weight;
}
"#;

/// Parameters for the Gaussian blur effect.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(dead_code)]
pub struct BlurParams {
    pub radius: f32,
    #[allow(dead_code)]
    pub _pad: f32,
    pub tex_size: [f32; 2],
}

/// Single-pass no-op effect used to validate backdrop capture placement without blur math.
pub const PASSTHROUGH_WGSL: &str = r#"
@fragment
fn effect_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(t_input, s_input, uv);
}
"#;

/// Opaque blue eight-pixel drop used by cached shape-effect regression tiles.
pub const SHAPE_DROP_WGSL: &str = r#"
@fragment
fn effect_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let dimensions = vec2<f32>(textureDimensions(t_input));
    let coverage = textureSample(t_input, s_input, uv - vec2<f32>(8.0) / dimensions).a;
    return vec4<f32>(0.0, 0.0, coverage, coverage);
}
"#;

/// Fixed-radius horizontal Gaussian blur used by the visual drop-shadow tile.
pub const DROP_SHADOW_HORIZONTAL_BLUR_WGSL: &str = r#"
@fragment
fn effect_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let dimensions = vec2<f32>(textureDimensions(t_input));
    let pixel = vec2<f32>(1.0 / dimensions.x, 0.0);
    let sigma = 2.0;
    var color = vec4<f32>(0.0);
    var total_weight = 0.0;

    for (var sample_offset = -5; sample_offset <= 5; sample_offset++) {
        let distance = f32(sample_offset);
        let weight = exp(-(distance * distance) / (2.0 * sigma * sigma));
        color += textureSample(t_input, s_input, uv + pixel * distance) * weight;
        total_weight += weight;
    }

    return color / total_weight;
}
"#;

/// Vertical Gaussian blur, offset, and premultiplied black tint for a real drop shadow.
pub const DROP_SHADOW_VERTICAL_TINT_WGSL: &str = r#"
@fragment
fn effect_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let dimensions = vec2<f32>(textureDimensions(t_input));
    let pixel = vec2<f32>(0.0, 1.0 / dimensions.y);
    let shadow_offset = vec2<f32>(7.0, 8.0) / dimensions;
    let sigma = 2.0;
    var coverage = 0.0;
    var total_weight = 0.0;

    for (var sample_offset = -5; sample_offset <= 5; sample_offset++) {
        let distance = f32(sample_offset);
        let weight = exp(-(distance * distance) / (2.0 * sigma * sigma));
        coverage += textureSample(t_input, s_input, uv - shadow_offset + pixel * distance).a
            * weight;
        total_weight += weight;
    }

    let alpha = 0.65 * coverage / total_weight;
    return vec4<f32>(0.0, 0.0, 0.0, alpha);
}
"#;
