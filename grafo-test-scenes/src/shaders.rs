/// Horizontal Gaussian blur pass with direction `(1,0)`
pub const HORIZONTAL_BLUR_WGSL: &str = concat!(
    "const DIRECTION: vec2<f32> = vec2<f32>(1.0, 0.0);\n",
    include_str!("shaders/gaussian_blur.wgsl"),
);

/// Vertical Gaussian blur pass with direction `(0,1)`
pub const VERTICAL_BLUR_WGSL: &str = concat!(
    "const DIRECTION: vec2<f32> = vec2<f32>(0.0, 1.0);\n",
    include_str!("shaders/gaussian_blur.wgsl"),
);

/// Parameters for the Gaussian blur effect.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlurParams {
    pub radius: f32,
    pub _pad: f32,
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

/// Vertical Gaussian blur, offset, and premultiplied black tint for a drop shadow.
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

/// Horizontal blur tuned for half-resolution shape-effect masks: half the sigma of
/// `DROP_SHADOW_HORIZONTAL_BLUR_WGSL`, so the bilinearly upscaled result matches it.
pub const DOWNSAMPLED_DROP_SHADOW_HORIZONTAL_BLUR_WGSL: &str = r#"
@fragment
fn effect_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let dimensions = vec2<f32>(textureDimensions(t_input));
    let pixel = vec2<f32>(1.0 / dimensions.x, 0.0);
    let sigma = 1.0;
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

/// Vertical blur, offset, and tint tuned for half-resolution shape-effect masks:
/// half the sigma and offset of `DROP_SHADOW_VERTICAL_TINT_WGSL`, so the bilinearly
/// upscaled result lands at the same screen position.
pub const DOWNSAMPLED_DROP_SHADOW_VERTICAL_TINT_WGSL: &str = r#"
@fragment
fn effect_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let dimensions = vec2<f32>(textureDimensions(t_input));
    let pixel = vec2<f32>(0.0, 1.0 / dimensions.y);
    let shadow_offset = vec2<f32>(3.5, 4.0) / dimensions;
    let sigma = 1.0;
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
