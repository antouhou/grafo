/// Gaussian blur radius and sigma in input texture pixels.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlurParams {
    pub radius: f32,
    pub sigma: f32,
}

impl BlurParams {
    pub fn new(radius: f32) -> Self {
        Self {
            radius,
            sigma: radius / 3.0,
        }
    }
}

/// All passes share one uniform buffer, so radius and sigma must match `BlurParams`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DropShadowParams {
    pub radius: f32,
    pub sigma: f32,
    /// Shadow displacement in input texture pixels.
    pub offset: [f32; 2],
    /// Premultiplied RGBA.
    pub color: [f32; 4],
}

/// Horizontal Gaussian blur pass with direction `(1,0)`
pub const HORIZONTAL_BLUR_WGSL: &str = concat!(
    "const DIRECTION: vec2<f32> = vec2<f32>(1.0, 0.0);\n",
    include_str!("gaussian_blur.wgsl"),
);

/// Vertical Gaussian blur pass with direction `(0,1)`
pub const VERTICAL_BLUR_WGSL: &str = concat!(
    "const DIRECTION: vec2<f32> = vec2<f32>(0.0, 1.0);\n",
    include_str!("gaussian_blur.wgsl"),
);

/// Offsets and colors the alpha mask produced by the blur passes.
pub const SHADOW_TINT_WGSL: &str = r#"
struct DropShadowParams {
    radius: f32,
    sigma: f32,
    offset: vec2<f32>,
    color: vec4<f32>,
}
@group(1) @binding(0) var<uniform> params: DropShadowParams;

@fragment
fn effect_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let dimensions = vec2<f32>(textureDimensions(t_input));
    let coverage = textureSample(t_input, s_input, uv - params.offset / dimensions).a;
    return params.color * coverage;
}
"#;

/// Single-pass no-op effect used to validate backdrop capture placement without blur math.
pub const PASSTHROUGH_WGSL: &str = r#"
@fragment
fn effect_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(t_input, s_input, uv);
}
"#;

/// Samples padded capture edges and makes transparent offscreen pixels visible as black.
pub const PADDED_BACKDROP_SAMPLING_WGSL: &str = r#"
@fragment
fn effect_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let width = f32(textureDimensions(t_input).x);
    let offset = select(7.0, -7.0, uv.x < 0.25) / width;
    let sampled = textureSample(t_input, s_input, uv + vec2<f32>(offset, 0.0));
    return vec4<f32>(sampled.rgb, 1.0);
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
