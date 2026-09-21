use super::{build_effect_wgsl, validate_effect_shader};
use crate::effect::EffectShaderError;
use naga::front::wgsl;
use naga::valid::{Capabilities, ValidationFlags, Validator};

const PASSTHROUGH: &str = r#"
@fragment
fn effect_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(t_input, s_input, uv);
}
"#;

fn validate_fragment(source: &str) -> Result<bool, EffectShaderError> {
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::default());
    validate_effect_shader(&build_effect_wgsl(source), &mut validator)
}

#[test]
fn parameter_detection_ignores_nested_comments() {
    let source = format!("/* outer /* inner */ @group(1) */\n{PASSTHROUGH}");
    assert!(!validate_fragment(&source).unwrap());
}

#[test]
fn parameter_detection_preserves_bindings_between_line_comments() {
    let source = format!(
        "// /*\n@group(1) @binding(0) var<uniform> params: vec4<f32>;\n// */\n{PASSTHROUGH}"
    );
    assert!(validate_fragment(&source).unwrap());
}

#[test]
fn parameter_detection_accepts_constant_expressions() {
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::default());
    for group in ["1u", "0x1", "1i", "PARAMS_GROUP", "0 + 1"] {
        let source = format!(
            "const PARAMS_GROUP = 1u;\n@group({group}) @binding(0) var<uniform> params: vec4<f32>;\n{PASSTHROUGH}"
        );
        assert!(
            validate_effect_shader(&build_effect_wgsl(&source), &mut validator).unwrap(),
            "missed group {group}"
        );
    }
}

#[test]
fn invalid_wgsl_reports_the_parse_error() {
    let error =
        validate_fragment("@fragment fn effect_main(").expect_err("invalid WGSL must be rejected");
    assert!(matches!(error, EffectShaderError::Parse(error) if error.labels().len() > 0));
}

#[test]
fn semantic_errors_are_rejected_after_parsing() {
    let source = r#"
@fragment
fn effect_main() -> @location(0) vec4<f32> {
    return vec3<f32>(1.0);
}
"#;
    assert!(wgsl::parse_str(&build_effect_wgsl(source)).is_ok());
    let error = validate_fragment(source).expect_err("invalid return type must be rejected");
    assert!(matches!(error, EffectShaderError::Validation(_)));
}

#[test]
fn effect_main_must_be_a_fragment_entry_point() {
    for source in [
        "",
        "fn effect_main() {}",
        "@compute @workgroup_size(1) fn effect_main() {}",
    ] {
        assert!(matches!(
            validate_fragment(source),
            Err(EffectShaderError::MissingFragmentEntryPoint)
        ));
    }
}

#[test]
fn unsupported_resource_bindings_are_rejected() {
    for (expected_group, expected_binding, declaration) in [
        (
            1,
            1,
            "@group(1) @binding(1) var<uniform> params: vec4<f32>;",
        ),
        (
            2,
            0,
            "@group(2) @binding(0) var<uniform> params: vec4<f32>;",
        ),
        (
            0,
            2,
            "@group(0) @binding(2) var<uniform> params: vec4<f32>;",
        ),
        (
            0,
            0,
            "@group(0) @binding(0) var<uniform> params: vec4<f32>;",
        ),
        (
            1,
            0,
            "@group(1) @binding(0) var<storage, read> params: vec4<f32>;",
        ),
        (1, 0, "@group(1) @binding(0) var params: texture_2d<f32>;"),
    ] {
        let source = format!("{declaration}\n{PASSTHROUGH}");
        assert!(
            matches!(validate_fragment(&source),
                Err(EffectShaderError::UnsupportedBinding { group, binding })
                if group == expected_group && binding == expected_binding),
            "incorrect rejection reason for {declaration}"
        );
    }
    let source = format!(
        "@group(1) @binding(0) var<uniform> first: vec4<f32>;\n@group(1) @binding(0) var<uniform> second: vec4<f32>;\n{PASSTHROUGH}"
    );
    assert!(matches!(
        validate_fragment(&source),
        Err(EffectShaderError::DuplicateParameterBinding)
    ));
}
