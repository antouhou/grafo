@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0)
    );
    return vec4<f32>(positions[index], 0.0, 1.0);
}

@fragment
fn fs_clear() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}

@group(0) @binding(0) var retained_image: texture_2d<f32>;

override linear_output: bool = false;
override clamp_to_sdr: bool = true;
override attachment_is_srgb: bool = false;
override straight_alpha: bool = false;
override premultiplied_alpha: bool = false;

fn linear_to_srgb(linear: vec3<f32>) -> vec3<f32> {
    let magnitude = abs(linear);
    return select(
        sign(linear) * (1.055 * pow(magnitude, vec3<f32>(1.0 / 2.4)) - 0.055),
        12.92 * linear,
        magnitude <= vec3<f32>(0.0031308)
    );
}

fn srgb_to_linear(encoded: vec3<f32>) -> vec3<f32> {
    let magnitude = abs(encoded);
    return select(
        sign(encoded) * pow((magnitude + 0.055) / 1.055, vec3<f32>(2.4)),
        encoded / 12.92,
        magnitude <= vec3<f32>(0.04045)
    );
}

fn surface_color(color: vec4<f32>) -> vec4<f32> {
    var rgb = color.rgb;
    let premultiply_in_srgb = premultiplied_alpha && !linear_output && color.a < 1.0;
    if (straight_alpha && color.a < 1.0) || premultiply_in_srgb {
        if color.a == 0.0 {
            return vec4<f32>(0.0);
        }
        rgb /= color.a;
    }
    if linear_output {
        return vec4<f32>(rgb, color.a);
    }
    if clamp_to_sdr {
        rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    if attachment_is_srgb && !premultiply_in_srgb {
        return vec4<f32>(rgb, color.a);
    }
    var encoded = linear_to_srgb(rgb);
    if premultiply_in_srgb {
        encoded *= color.a;
    }
    if attachment_is_srgb {
        // The attachment re-encodes this premultiplied value on write.
        return vec4<f32>(srgb_to_linear(encoded), color.a);
    }
    return vec4<f32>(encoded, color.a);
}

@fragment
fn fs_present(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return surface_color(textureLoad(retained_image, vec2<i32>(position.xy), 0));
}

@fragment
fn fs_dirty_region_overlay(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    // Premultiplied red at 25% opacity.
    let scene = textureLoad(retained_image, vec2<i32>(position.xy), 0);
    return surface_color(vec4<f32>(0.25, 0.0, 0.0, 0.25) + 0.75 * scene);
}
