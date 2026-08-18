// Composites a captured backdrop layer into the parent target, translating
// from destination pixel coordinates back into the capture's source region.

struct BackdropLayerParams {
    capture_origin: vec2<i32>,
    source_size: vec2<i32>,
};

@group(0) @binding(0) var foreground_texture: texture_2d<f32>;
@group(0) @binding(1) var<uniform> layer_params: BackdropLayerParams;

@fragment
fn fs_backdrop_layer(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let destination_pixel = vec2<i32>(position.xy);
    let source_pixel = layer_params.capture_origin + destination_pixel;
    if source_pixel.x < 0
        || source_pixel.y < 0
        || source_pixel.x >= layer_params.source_size.x
        || source_pixel.y >= layer_params.source_size.y
    {
        return vec4<f32>(0.0);
    }
    return textureLoad(foreground_texture, source_pixel, 0);
}
