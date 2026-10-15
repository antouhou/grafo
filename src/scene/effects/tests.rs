use crate::core::effect::{BackdropCaptureArea, BackdropEffectConfig, ShapeMaskEffectConfig};
use crate::scene::effects::{self, backdrops};

#[test]
fn validate_backdrop_config_rejects_non_positive_downsample() {
    let result = backdrops::validate_backdrop_config(&BackdropEffectConfig::new().downsample(0.0));
    assert!(result.is_err());
}

#[test]
fn validate_backdrop_config_rejects_negative_padding() {
    let result = backdrops::validate_backdrop_config(&BackdropEffectConfig::new().padding(-1.0));
    assert!(result.is_err());
}

#[test]
fn validate_backdrop_config_rejects_inverted_screen_rect() {
    let result = backdrops::validate_backdrop_config(
        &BackdropEffectConfig::new()
            .capture_area(BackdropCaptureArea::ScreenRect([(10.0, 10.0), (5.0, 15.0)])),
    );
    assert!(result.is_err());
}

#[test]
fn validate_backdrop_config_rejects_non_finite_screen_rect() {
    let result = backdrops::validate_backdrop_config(&BackdropEffectConfig::new().capture_area(
        BackdropCaptureArea::ScreenRect([(0.0, 0.0), (f32::INFINITY, 15.0)]),
    ));
    assert!(result.is_err());
}

#[test]
fn validate_shape_mask_effect_config_rejects_negative_or_non_finite_outsets() {
    assert!(
        effects::validate_shape_mask_effect_config(&ShapeMaskEffectConfig::new().outset(-1.0))
            .is_err()
    );
    assert!(
        effects::validate_shape_mask_effect_config(&ShapeMaskEffectConfig::new().outsets(
            0.0,
            f32::INFINITY,
            0.0,
            0.0
        ))
        .is_err()
    );
}

#[test]
fn validate_shape_mask_effect_config_rejects_out_of_range_downsample() {
    for downsample in [0.0, -0.5, f32::NAN, 1.5] {
        assert!(effects::validate_shape_mask_effect_config(
            &ShapeMaskEffectConfig::new().downsample(downsample)
        )
        .is_err());
    }
    for downsample in [1.0, 0.5, f32::EPSILON] {
        assert!(effects::validate_shape_mask_effect_config(
            &ShapeMaskEffectConfig::new().downsample(downsample)
        )
        .is_ok());
    }
}
