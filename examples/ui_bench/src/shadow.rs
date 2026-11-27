use crate::BenchmarkError;
use grafo::{Scene, ShapeMaskEffectConfig};

const EFFECT: u64 = 1;
const OFFSET: [f32; 2] = [6.0, 8.0];
const OPACITY: f32 = 0.4;

const SHADER: &str = r#"
@group(1) @binding(0) var<uniform> shadow: vec4<f32>;

@fragment
fn effect_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(t_input));
    let offset = shadow.xy * vec2<f32>(1.0, size.x / size.y);
    let coverage = textureSample(t_input, s_input, uv - offset).a;
    return vec4<f32>(0.0, 0.0, 0.0, coverage * shadow.z);
}
"#;

pub fn attach(scene: &mut Scene, node: usize, width: f32) -> Result<(), BenchmarkError> {
    scene.load_effect(EFFECT, &[SHADER])?;
    // Normalize both axes by width so resizing the mask needs no uniform update.
    let parameters = [
        OFFSET[0] / (width + OFFSET[0]),
        OFFSET[1] / (width + OFFSET[0]),
        OPACITY,
        0.0,
    ];
    scene.set_shape_mask_effect(
        node,
        EFFECT,
        bytemuck::bytes_of(&parameters),
        ShapeMaskEffectConfig::new().outsets(0.0, 0.0, OFFSET[0], OFFSET[1]),
    )?;
    Ok(())
}
