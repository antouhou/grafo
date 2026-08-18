// Simple passthrough fragment shader for compositing effect results back into the parent target.

@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var s_input: sampler;

@fragment
fn fs_composite(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(t_input, s_input, uv);
}
