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
