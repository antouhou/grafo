@group(2) @binding(0) var filter_sampler: sampler;

fn transfer_filter_change(working: vec4<f32>, original: vec4<f32>, filtered: vec4<f32>) -> vec4<f32> {
    if all(filtered==vec4<f32>(0.0)) && any(filtered!=original) {
        return filtered;
    }
    // Transfer the change toward black or white without extrapolating past either endpoint.
    let darker = working*filtered/max(original,vec4<f32>(0.000001));
    let lighter = vec4<f32>(1.0)-(vec4<f32>(1.0)-working)*(vec4<f32>(1.0)-filtered)/max(vec4<f32>(1.0)-original,vec4<f32>(0.000001));
    let changed = select(darker,lighter,filtered>original);
    return select(changed,working,filtered==original);
}

@fragment
fn filter_fragment(input: ClipVertex) -> @location(0) vec4<f32> {
    let uv = input.position.xy/parameters.destination.zw;
    let filtered = textureSampleLevel(foreground,filter_sampler,uv,0.0);
    let world = vec2<i32>(floor(input.position.xy+parameters.destination.xy));
    let pixel = vec2<u32>(world)-input.minimum;
    var amount = 1.0;
    if input.coverage_start != 0xffffffffu {
        amount = coverage[input.coverage_start+pixel.y*8u+pixel.x];
    }
    if amount == 1.0 || amount == 0.0 {
        return filtered;
    }
    let original = textureSampleLevel(backdrop_background,filter_sampler,uv,0.0);
    let working = load_color(background,world-vec2<i32>(parameters.source.xy),0u);
    return transfer_filter_change(working,original,filtered);
}

@fragment
fn inherited_fragment(input: ClipVertex) -> @location(0) vec4<f32> {
    let world = vec2<i32>(floor(input.position.xy+parameters.destination.xy));
    let pixel = vec2<u32>(world)-input.minimum;
    var amount = 1.0;
    if input.coverage_start != 0xffffffffu {
        amount = coverage[input.coverage_start+pixel.y*8u+pixel.x];
    }
    let source_pixel = world-vec2<i32>(parameters.source.xy);
    let conditional = load_color(foreground,vec2<i32>(input.position.xy),0u);
    let prior = load_color(background,source_pixel,0u);
    return load_color(backdrop_background,source_pixel,0u)+amount*(conditional-prior);
}
