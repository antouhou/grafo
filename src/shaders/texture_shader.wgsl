struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
//    let x = 2.0 * model.position.x - 1.0;
//    let y = 1.0 - 2.0 * model.position.y;
//    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.clip_position = vec4<f32>(model.position, 0.0, 1.0);
    return out;
}

// Fragment shader

//    return vec4<f32>(1.0, 0.0, 1.0, 1.0); // <--- This is the color

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
//      return vec4<f32>(in.tex_coords, 0.0, 1.0);
    return textureSample(t_diffuse, s_diffuse, in.tex_coords);
}
