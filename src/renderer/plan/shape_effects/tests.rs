use super::{compute_shape_effect_raster_rect, shape_effect_quad_transform, ShapeEffectPlan};
use crate::commands::{DrawSegment, IntermediateTextureId, ShapeDrawId, Target, TexturePlacement};
use crate::core::effect::{ShapeEffectConfig, ShapeEffectInstance};
use crate::core::shape::CachedShapeHandle;
use crate::core::util::ShapeResources;
use crate::core::vertex::InstanceTransform;
use crate::renderer::types::CachedShapeDrawData;
use crate::renderer::types::DrawTreeNode;
use crate::{Shape, ShapeDrawCommandOptions, Size, Stroke};
use ahash::{HashMap, HashMapExt};
use easy_tree::Tree;
use lyon::tessellation::FillTessellator;
use std::sync::Arc;

#[test]
fn raster_rect_rounds_outward_and_adds_fringe_guard() {
    let raster_rect = compute_shape_effect_raster_rect(
        [(1.25, 2.75), (10.1, 20.2)],
        ShapeEffectConfig::new().outsets(1.0, 2.0, 3.0, 4.0),
        2.0,
        0.75,
    )
    .unwrap();

    assert_eq!(raster_rect.local_physical_origin, [-1, 0]);
    assert_eq!(raster_rect.texture_size, [29, 50]);
    assert_eq!(raster_rect.local_bounds, [(-0.5, 0.0), (14.0, 25.0)]);
}

#[test]
fn raster_rect_downsample_shrinks_texture_but_not_coverage() {
    let full_resolution = compute_shape_effect_raster_rect(
        [(1.25, 2.75), (10.1, 20.2)],
        ShapeEffectConfig::new().outsets(1.0, 2.0, 3.0, 4.0),
        2.0,
        0.75,
    )
    .unwrap();
    let downsampled = compute_shape_effect_raster_rect(
        [(1.25, 2.75), (10.1, 20.2)],
        ShapeEffectConfig::new()
            .outsets(1.0, 2.0, 3.0, 4.0)
            .downsample(0.5),
        2.0,
        0.75,
    )
    .unwrap();

    assert_eq!(downsampled.texture_size, [15, 25]);
    assert_eq!(
        downsampled.local_physical_origin,
        full_resolution.local_physical_origin
    );
    assert_eq!(downsampled.local_bounds, full_resolution.local_bounds);
}

#[test]
fn raster_rect_downsample_keeps_at_least_one_texel() {
    let raster_rect = compute_shape_effect_raster_rect(
        [(0.0, 0.0), (1.0, 1.0)],
        ShapeEffectConfig::new().downsample(0.1),
        1.0,
        0.75,
    )
    .unwrap();

    assert!(raster_rect.texture_size[0] >= 1);
    assert!(raster_rect.texture_size[1] >= 1);
}

#[test]
fn raster_rect_rejects_out_of_range_downsample() {
    for downsample in [0.0, -0.5, f32::NAN, 1.5] {
        assert!(compute_shape_effect_raster_rect(
            [(0.0, 0.0), (10.0, 10.0)],
            ShapeEffectConfig::new().downsample(downsample),
            1.0,
            0.75,
        )
        .is_none());
    }
}

#[test]
fn raster_rect_rejects_non_finite_inputs() {
    assert!(compute_shape_effect_raster_rect(
        [(0.0, 0.0), (f32::NAN, 10.0)],
        ShapeEffectConfig::default(),
        1.0,
        0.75,
    )
    .is_none());
}

#[test]
fn shape_effect_quad_transform_maps_unit_quad_before_source_transform() {
    let transform = shape_effect_quad_transform(
        [(-3.0, -4.0), (11.0, 15.0)],
        Some(InstanceTransform::translation(5.0, 7.0)),
    );

    assert_eq!(transform.col0, [14.0, 0.0, 0.0, 0.0]);
    assert_eq!(transform.col1, [0.0, 19.0, 0.0, 0.0]);
    assert_eq!(transform.col3, [2.0, 3.0, 0.0, 1.0]);
}

fn scene_with_effects() -> (Tree<DrawTreeNode>, HashMap<usize, ShapeEffectInstance>) {
    let mut tree = Tree::new();
    let mut effects = HashMap::new();
    let mut resources = ShapeResources::new();
    let handle = CachedShapeHandle::new(
        &Shape::rect([(0.0, 0.0), (20.0, 10.0)], Stroke::default()),
        &mut FillTessellator::new(),
        &mut resources,
        Some(17),
    );
    for translation in [5.0, 45.0] {
        let node = tree.add_node(DrawTreeNode::CachedShape(CachedShapeDrawData::new(
            handle.clone(),
            &ShapeDrawCommandOptions::new()
                .transform(InstanceTransform::translation(translation, 8.0)),
        )));
        effects.insert(
            node,
            ShapeEffectInstance {
                effect_id: 9,
                params: Arc::from([1u8, 2, 3, 4]),
                config: ShapeEffectConfig::new().outset(3.0).downsample(0.5),
            },
        );
    }
    (tree, effects)
}

#[test]
fn mask_scopes_and_effect_outputs_are_complete_without_the_scene() {
    let mut plan = ShapeEffectPlan::new();
    {
        let (tree, effects) = scene_with_effects();
        plan.plan(&tree, &effects, 2.0, 0.75, Size::new(200, 100), 1024);
    }
    assert_eq!(plan.commands.segments.len(), 8);
    for (index, commands) in plan.commands.segments.as_chunks::<4>().0.iter().enumerate() {
        let [DrawSegment::BeginTarget(Target::Mask(target)), DrawSegment::DrawShapeMask(mask), DrawSegment::EndTarget, DrawSegment::ApplyEffect(effect)] =
            commands
        else {
            panic!("expected completed mask scope before the effect")
        };
        assert_eq!(target.texture, IntermediateTextureId::Planned(index));
        assert_eq!(target.size, [27, 17]);
        assert_eq!(mask.local_physical_origin, [-7, -7]);
        assert_eq!(mask.local_bounds, [(-3.5, -3.5), (23.5, 13.5)]);
        assert_eq!(effect.input, target.texture);
        assert_eq!(effect.output, IntermediateTextureId::ShapeEffect(index));
        assert_eq!(effect.effect_id, 9);
        assert_eq!(
            effect.parameters.bytes(&plan.commands.effect_parameters),
            [1, 2, 3, 4]
        );
        let ShapeDrawId(node) = mask.shape;
        let composite = plan.composites[&node];
        assert_eq!(composite.texture, effect.output);
        let TexturePlacement::Local {
            transform,
            sampling,
        } = composite.placement
        else {
            panic!("expected source-local placement")
        };
        assert_eq!(transform.col0, [27.0, 0.0, 0.0, 0.0]);
        assert_eq!(transform.col1, [0.0, 17.0, 0.0, 0.0]);
        assert_eq!(
            transform.col3,
            [if node == 0 { 1.5 } else { 41.5 }, 4.5, 0.0, 1.0]
        );
        assert_eq!(sampling.scale, [1.0, 1.0]);
    }
}

#[test]
fn rebuilding_shape_effect_commands_reuses_storage_and_drops_removed_outputs() {
    let (mut tree, mut effects) = scene_with_effects();
    let mut plan = ShapeEffectPlan::new();
    plan.plan(&tree, &effects, 1.0, 0.75, Size::new(100, 100), 1024);
    let commands_pointer = plan.commands.segments.as_ptr();
    let composite_capacity = plan.composites.capacity();
    tree.clear();
    effects.clear();
    let (rebuilt_tree, rebuilt_effects) = scene_with_effects();
    for viewport in [Size::new(100, 100), Size::new(1, 1), Size::new(100, 100)] {
        plan.plan(&rebuilt_tree, &rebuilt_effects, 1.0, 0.75, viewport, 1024);
        assert_eq!(plan.commands.segments.as_ptr(), commands_pointer);
        assert_eq!(plan.composites.capacity(), composite_capacity);
        assert_eq!(plan.composites.is_empty(), viewport.width == 1);
    }
    plan.plan(&tree, &effects, 1.0, 0.75, Size::new(100, 100), 1024);
    assert!(plan.commands.segments.is_empty());
    assert!(plan.composites.is_empty());
}

#[test]
fn invalid_or_empty_masks_never_produce_texture_references() {
    let (mut tree, mut effects) = scene_with_effects();
    let empty = tree.add_node(DrawTreeNode::CachedShape(CachedShapeDrawData::new(
        CachedShapeHandle::new(
            &Shape::builder().build(),
            &mut FillTessellator::new(),
            &mut ShapeResources::new(),
            None,
        ),
        &ShapeDrawCommandOptions::new(),
    )));
    effects.insert(
        empty,
        ShapeEffectInstance {
            effect_id: 9,
            params: Arc::from([]),
            config: ShapeEffectConfig::default(),
        },
    );
    let mut plan = ShapeEffectPlan::new();
    for (scale, max_dimension) in [(f64::NAN, 1024), (1.0, 1)] {
        plan.plan(
            &tree,
            &effects,
            scale,
            0.75,
            Size::new(100, 100),
            max_dimension,
        );
        assert!(plan.commands.segments.is_empty());
        assert!(plan.composites.is_empty());
    }
    plan.plan(&tree, &effects, 1.0, 0.75, Size::new(100, 100), 1024);
    assert_eq!(plan.composites.len(), 2);
    assert!(!plan.composites.contains_key(&empty));
}
