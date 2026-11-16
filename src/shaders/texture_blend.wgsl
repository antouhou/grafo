override blend_mode: u32 = 0u;

struct Uniforms {
    canvas_size: vec2<f32>,
    target_origin: vec2<f32>,
    scale_factor: f32,
    _padding: f32,
};

struct LocalVertexInput {
    @location(0) position: vec2<f32>,
    @location(2) texture_coordinates: vec2<f32>,
    @location(3) transform_column0: vec4<f32>,
    @location(4) transform_column1: vec4<f32>,
    @location(5) transform_column2: vec4<f32>,
    @location(6) transform_column3: vec4<f32>,
    @location(10) source_uv_transform: vec4<f32>,
};

struct VertexOutput {
    @invariant @builtin(position) position: vec4<f32>,
    @location(0) raw_uv: vec2<f32>,
    @location(1) @interpolate(flat) source_uv_transform: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var source_texture: texture_2d<f32>;
@group(1) @binding(1) var source_sampler: sampler;
@group(2) @binding(0) var destination_texture: texture_2d<f32>;
@group(2) @binding(1) var destination_sampler: sampler;

fn compute_vertex_position(position: vec2<f32>, model: mat4x4<f32>) -> vec4<f32> {
    let transformed = model * vec4<f32>(position, 0.0, 1.0);
    let inverse_w = 1.0 / max(abs(transformed.w), 1e-6);
    let screen_position = transformed.xy * inverse_w;
    let target_position = screen_position - uniforms.target_origin / uniforms.scale_factor;
    let normalized_x = 2.0 * target_position.x / uniforms.canvas_size.x - 1.0;
    let normalized_y = 1.0 - 2.0 * target_position.y / uniforms.canvas_size.y;
    return vec4<f32>(normalized_x, normalized_y, 0.0, 1.0);
}

@vertex
fn vs_local(input: LocalVertexInput) -> VertexOutput {
    let transform = mat4x4<f32>(
        input.transform_column0,
        input.transform_column1,
        input.transform_column2,
        input.transform_column3,
    );
    var output: VertexOutput;
    output.position = compute_vertex_position(input.position, transform);
    output.raw_uv = input.texture_coordinates;
    output.source_uv_transform = input.source_uv_transform;
    return output;
}

@vertex
fn vs_target(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    var output: VertexOutput;
    output.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    output.raw_uv = vec2<f32>(uv.x, 1.0 - uv.y);
    output.source_uv_transform = vec4<f32>(1.0, 1.0, 0.0, 0.0);
    return output;
}

fn sample_source(input: VertexOutput) -> vec4<f32> {
    let raw_uv = clamp(input.raw_uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let source_uv = raw_uv * input.source_uv_transform.xy + input.source_uv_transform.zw;
    if any(source_uv < vec2<f32>(0.0)) || any(source_uv > vec2<f32>(1.0)) {
        return vec4<f32>(0.0);
    }
    return textureSampleLevel(source_texture, source_sampler, source_uv, 0.0);
}

fn unpremultiply(color: vec4<f32>) -> vec3<f32> {
    if color.a > 0.0 {
        return clamp(color.rgb / color.a, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    return vec3<f32>(0.0);
}

fn luminosity(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.3, 0.59, 0.11));
}

fn saturation(color: vec3<f32>) -> f32 {
    return max(max(color.r, color.g), color.b) - min(min(color.r, color.g), color.b);
}

fn clip_color(color: vec3<f32>) -> vec3<f32> {
    let brightness = luminosity(color);
    let minimum = min(min(color.r, color.g), color.b);
    let maximum = max(max(color.r, color.g), color.b);
    var clipped = color;
    if minimum < 0.0 {
        clipped = vec3<f32>(brightness)
            + (clipped - vec3<f32>(brightness)) * brightness / (brightness - minimum);
    }
    if maximum > 1.0 {
        clipped = vec3<f32>(brightness)
            + (clipped - vec3<f32>(brightness)) * (1.0 - brightness) / (maximum - brightness);
    }
    return clipped;
}

fn set_luminosity(color: vec3<f32>, brightness: f32) -> vec3<f32> {
    return clip_color(color + vec3<f32>(brightness - luminosity(color)));
}

fn set_saturation(color: vec3<f32>, amount: f32) -> vec3<f32> {
    let minimum = min(min(color.r, color.g), color.b);
    let maximum = max(max(color.r, color.g), color.b);
    if maximum > minimum {
        return (color - vec3<f32>(minimum)) * amount / (maximum - minimum);
    }
    return vec3<f32>(0.0);
}

fn color_dodge(destination: f32, source: f32) -> f32 {
    if destination == 0.0 {
        return 0.0;
    }
    if source == 1.0 {
        return 1.0;
    }
    return min(1.0, destination / (1.0 - source));
}

fn color_burn(destination: f32, source: f32) -> f32 {
    if destination == 1.0 {
        return 1.0;
    }
    if source == 0.0 {
        return 0.0;
    }
    return 1.0 - min(1.0, (1.0 - destination) / source);
}

fn hard_light_overlap(destination: vec4<f32>, source: vec4<f32>) -> vec3<f32> {
    return select(
        2.0 * destination.rgb * source.rgb,
        vec3<f32>(source.a * destination.a)
            - 2.0 * (vec3<f32>(destination.a) - destination.rgb) * (vec3<f32>(source.a) - source.rgb),
        2.0 * source.rgb > vec3<f32>(source.a),
    );
}

fn soft_light(destination: f32, source: f32) -> f32 {
    if source <= 0.5 {
        return destination - (1.0 - 2.0 * source) * destination * (1.0 - destination);
    }
    var curve = sqrt(destination);
    if destination <= 0.25 {
        curve = ((16.0 * destination - 12.0) * destination + 4.0) * destination;
    }
    return destination + (2.0 * source - 1.0) * (curve - destination);
}

fn blend_colors(destination: vec3<f32>, source: vec3<f32>) -> vec3<f32> {
    switch blend_mode {
        case 17u: {
            return vec3<f32>(
                color_dodge(destination.r, source.r),
                color_dodge(destination.g, source.g),
                color_dodge(destination.b, source.b),
            );
        }
        case 18u: {
            return vec3<f32>(
                color_burn(destination.r, source.r),
                color_burn(destination.g, source.g),
                color_burn(destination.b, source.b),
            );
        }
        case 20u: {
            return vec3<f32>(
                soft_light(destination.r, source.r),
                soft_light(destination.g, source.g),
                soft_light(destination.b, source.b),
            );
        }
        case 24u: {
            return set_luminosity(set_saturation(source, saturation(destination)), luminosity(destination));
        }
        case 25u: {
            return set_luminosity(set_saturation(destination, saturation(source)), luminosity(destination));
        }
        case 26u: {
            return set_luminosity(source, luminosity(destination));
        }
        case 27u: {
            return set_luminosity(destination, luminosity(source));
        }
        default: {
            return source;
        }
    }
}

fn blend_overlap(destination: vec4<f32>, source: vec4<f32>) -> vec3<f32> {
    switch blend_mode {
        case 14u: {
            return hard_light_overlap(source, destination);
        }
        case 15u: {
            return min(destination.rgb * source.a, source.rgb * destination.a);
        }
        case 16u: {
            return max(destination.rgb * source.a, source.rgb * destination.a);
        }
        case 19u: {
            return hard_light_overlap(destination, source);
        }
        case 21u: {
            return abs(destination.rgb * source.a - source.rgb * destination.a);
        }
        case 23u: {
            return destination.rgb * source.rgb;
        }
        default: {
            let blended = clamp(
                blend_colors(unpremultiply(destination), unpremultiply(source)),
                vec3<f32>(0.0),
                vec3<f32>(1.0),
            );
            return source.a * destination.a * blended;
        }
    }
}

@fragment
fn fs_source(input: VertexOutput) -> @location(0) vec4<f32> {
    return sample_source(input);
}

@fragment
fn fs_blend(input: VertexOutput) -> @location(0) vec4<f32> {
    let source = sample_source(input);
    if source.a == 0.0 {
        return vec4<f32>(0.0);
    }
    let dimensions = textureDimensions(destination_texture, 0);
    let coordinates = clamp(
        vec2<i32>(floor(input.raw_uv * vec2<f32>(dimensions))),
        vec2<i32>(0),
        vec2<i32>(dimensions) - vec2<i32>(1),
    );
    let destination = textureLoad(destination_texture, coordinates, 0);
    // The attachment supplies the destination contribution.
    return vec4<f32>(
        (1.0 - destination.a) * source.rgb + blend_overlap(destination, source),
        source.a,
    );
}
