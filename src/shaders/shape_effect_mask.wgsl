// Rasterizes a shape's coverage mask into an offscreen texture for shape effects.
// Anti-aliased boundary vertices are expanded along their normal by the fringe width
// (converted from physical pixels to logical units) before interpolation.

struct MaskUniforms {
    local_origin: vec2<f32>,
    logical_size: vec2<f32>,
    scale_factor: f32,
    fringe_width: f32,
    _padding: vec2<f32>,
};

struct MaskVertexInput {
    @location(0) position: vec2<f32>,
    @location(8) normal: vec2<f32>,
    @location(9) coverage: f32,
};

struct MaskVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) coverage: f32,
};

@group(0) @binding(0) var<uniform> uniforms: MaskUniforms;

@vertex
fn mask_vertex(input: MaskVertexInput) -> MaskVertexOutput {
    var local_position = input.position;
    if input.coverage < 1.0 {
        local_position += input.normal * (uniforms.fringe_width / uniforms.scale_factor);
    }

    let target_position = local_position - uniforms.local_origin;
    var output: MaskVertexOutput;
    output.position = vec4<f32>(
        2.0 * target_position.x / uniforms.logical_size.x - 1.0,
        1.0 - 2.0 * target_position.y / uniforms.logical_size.y,
        0.0,
        1.0,
    );
    output.coverage = input.coverage;
    return output;
}

@fragment
fn mask_fragment(input: MaskVertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.coverage);
}
