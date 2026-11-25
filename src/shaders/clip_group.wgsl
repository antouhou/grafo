struct ClipParameters {
    destination: vec4<f32>,
    source: vec4<f32>,
};
struct ClipTile {
    minimum: vec2<u32>,
    maximum: vec2<u32>,
    coverage_start: u32,
    projection_index: u32,
};
@group(0) @binding(0) var<uniform> parameters: ClipParameters;
@group(0) @binding(1) var<storage, read> coverage: array<f32>;
@group(0) @binding(2) var<storage, read> tiles: array<ClipTile>;
@group(1) @binding(0) var background: CLIP_TEXTURE;
@group(1) @binding(1) var foreground: CLIP_TEXTURE;
@group(1) @binding(2) var backdrop_background: CLIP_TEXTURE;

fn load_color(image: CLIP_TEXTURE, pixel: vec2<i32>, sample: u32) -> vec4<f32> {
    return textureLoad(image,pixel,CLIP_SAMPLE);
}

struct ClipVertex {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) minimum: vec2<u32>,
    @location(1) @interpolate(flat) coverage_start: u32,
};
@vertex
fn clip_vertex(@builtin(vertex_index) index: u32) -> ClipVertex {
    let tile_index = index / 4u;
    let tile = tiles[tile_index];
    let corners = array<vec2<f32>,4>(vec2<f32>(0.0,0.0),vec2<f32>(1.0,0.0),vec2<f32>(1.0,1.0),vec2<f32>(0.0,1.0));
    let point = mix(vec2<f32>(tile.minimum),vec2<f32>(tile.maximum),corners[index % 4u]);
    let normalized = (point-parameters.destination.xy)/parameters.destination.zw;
    return ClipVertex(vec4<f32>(normalized.x*2.0-1.0,1.0-normalized.y*2.0,0.0,1.0),tile.minimum,tile.coverage_start);
}
@fragment
fn clip_fragment(input: ClipVertex, @builtin(sample_index) sample: u32) -> @location(0) vec4<f32> {
    let world = vec2<i32>(floor(input.position.xy+parameters.destination.xy));
    let pixel = vec2<u32>(world)-input.minimum;
    var amount = 1.0;
    if input.coverage_start != 0xffffffffu {
        amount = coverage[input.coverage_start+pixel.y*8u+pixel.x];
    }
    let source_pixel = world-vec2<i32>(parameters.source.xy);
    return mix(load_color(background,source_pixel,sample),load_color(foreground,source_pixel,sample),amount);
}

@fragment
fn isolated_fragment(input: ClipVertex, @builtin(sample_index) sample: u32) -> @location(0) vec4<f32> {
    let world = vec2<i32>(floor(input.position.xy+parameters.destination.xy));
    let pixel = vec2<u32>(world)-input.minimum;
    var amount = 1.0;
    if input.coverage_start != 0xffffffffu {
        amount = coverage[input.coverage_start+pixel.y*8u+pixel.x];
    }
    return amount*load_color(foreground,world-vec2<i32>(parameters.source.xy),sample);
}
@fragment
fn capture_fragment(input: ClipVertex, @builtin(sample_index) sample: u32) -> @location(0) vec4<f32> {
    let world = vec2<i32>(floor(input.position.xy+parameters.destination.xy));
    let pixel = vec2<u32>(world)-input.minimum;
    var amount = 1.0;
    if input.coverage_start != 0xffffffffu {
        amount = coverage[input.coverage_start+pixel.y*8u+pixel.x];
    }
    let source_pixel = world-vec2<i32>(parameters.source.xy);
    return load_color(backdrop_background,source_pixel,sample)+amount*(load_color(foreground,source_pixel,sample)-load_color(background,source_pixel,sample));
}
@vertex
fn initialize_vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let points = array<vec2<f32>,3>(vec2<f32>(-1.0,-1.0),vec2<f32>(3.0,-1.0),vec2<f32>(-1.0,3.0));
    return vec4<f32>(points[index],0.0,1.0);
}
@fragment
fn initialize_fragment(@builtin(position) position: vec4<f32>, @builtin(sample_index) sample: u32) -> @location(0) vec4<f32> {
    return load_color(background,vec2<i32>(position.xy+parameters.source.zw),sample);
}

@fragment
fn export_stencil() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}

@fragment
fn import_stencil(@builtin(position) position: vec4<f32>, @builtin(sample_index) sample: u32) -> @location(0) vec4<f32> {
    if load_color(background,vec2<i32>(position.xy+parameters.source.zw),sample).r == 0.0 {
        discard;
    }
    return vec4<f32>(0.0);
}

