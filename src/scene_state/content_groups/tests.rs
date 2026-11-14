use super::ContentGroup;
use crate::core::effect::ShapeMaskEffectBounds;
use crate::core::shape::{CachedShapeHandle, ShapeResources};
use crate::core::{
    ContentEffectConfig, PhysicalRect, Shape, ShapeDrawOptions, ShapeMaskEffectConfig,
    UnsignedPhysicalRect, Viewport,
};
use crate::scene_state::effects::parameters::EffectParameterStorage;
use crate::scene_state::effects::{ContentEffectInstance, ShapeMaskEffectInstance};
use crate::scene_state::types::{CachedShapeDrawData, DrawTreeNode};

fn shape_node(bounds: [(f32, f32); 2]) -> DrawTreeNode {
    let shape =
        CachedShapeHandle::new(&Shape::rect(bounds), &mut ShapeResources::new(), None).unwrap();
    DrawTreeNode::CachedShape(CachedShapeDrawData::new(shape, ShapeDrawOptions::new()))
}

#[test]
fn effect_free_groups_keep_full_resolution_mask_bounds_and_clip_only_the_input() {
    let viewport = Viewport {
        physical_size: (64, 48),
        scale_factor: 2.0,
    };
    let bounds = [(-2.0, 4.0), (12.0, 14.0)];
    let node = shape_node(bounds);
    let config = ShapeMaskEffectConfig::new()
        .outsets(1.0, 2.0, 3.0, 4.0)
        .downsample(0.5);
    let mask = ShapeMaskEffectInstance {
        effect_id: 7,
        parameters: EffectParameterStorage::default().store(&[]),
        config,
        bounds: ShapeMaskEffectBounds::new(bounds, config, None, viewport.scale_factor).unwrap(),
    };

    let group = ContentGroup::new(None, &node, Some(&mask), viewport, 4096);

    assert_eq!(
        group.input_bounds,
        Some(UnsignedPhysicalRect::new((0, 3).into(), (31, 37).into()))
    );
    assert_eq!(
        group.output_bounds,
        Some(PhysicalRect::new((-7, 3).into(), (31, 37).into()))
    );
}

#[test]
fn effect_padding_can_exceed_output_limits_without_losing_the_input() {
    let viewport = Viewport {
        physical_size: (64, 64),
        scale_factor: 1.0,
    };
    let node = shape_node([(20.0, 20.0), (40.0, 40.0)]);
    let effect = ContentEffectInstance {
        effect_id: 8,
        parameters: EffectParameterStorage::default().store(&[]),
        config: ContentEffectConfig::new().outset(64.0),
    };

    let oversized = ContentGroup::new(Some(effect), &node, None, viewport, 128);
    let without_effect = ContentGroup::new(None, &node, None, viewport, 128);

    let expected_input = Some(UnsignedPhysicalRect::new((20, 20).into(), (40, 40).into()));
    assert_eq!(oversized.input_bounds, expected_input);
    assert!(oversized.output_bounds.is_none());
    assert_eq!(without_effect.input_bounds, expected_input);
    assert_eq!(
        without_effect.output_bounds,
        Some(PhysicalRect::new((20, 20).into(), (40, 40).into()))
    );
}
