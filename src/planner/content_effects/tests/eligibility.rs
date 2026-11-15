use crate::commands::{RenderOperation, ShapeDrawId, Target};
use crate::core::{
    BlendMode, Color, ContentEffectConfig, Shape, ShapeDrawOptions, ShapeMaskEffectConfig,
    UnsignedPhysicalRect, Viewport,
};
use crate::planner::Planner;
use crate::scene_state::types::{CachedShapeDrawData, DrawTreeNode};
use crate::scene_state::{SceneState, SceneStateContext};

fn add_shape(
    scene: &mut SceneState,
    parent: usize,
    bounds: [(f32, f32); 2],
    blend_mode: BlendMode,
) -> usize {
    let shape = scene.tessellate(&Shape::rect(bounds), None).unwrap();
    scene.insert_node(
        DrawTreeNode::CachedShape(CachedShapeDrawData::new(
            shape,
            ShapeDrawOptions::new()
                .color(Color::WHITE)
                .clips_children(false)
                .blend_mode(blend_mode),
        )),
        Some(parent),
        4096,
    )
}

#[test]
fn nested_captures_forward_each_shape_mode_without_changing_mask_composites() {
    for (outer_has_effect, inner_has_effect) in
        [(false, false), (false, true), (true, false), (true, true)]
    {
        let mut scene = SceneState::new(
            SceneStateContext::default(),
            Viewport {
                physical_size: (32, 32),
                scale_factor: 1.0,
            },
        );
        let root = scene.insert_node(DrawTreeNode::Group(Default::default()), None, 4096);
        let outer = add_shape(
            &mut scene,
            root,
            [(2.0, 2.0), (28.0, 28.0)],
            BlendMode::Multiply,
        );
        let inner = add_shape(
            &mut scene,
            outer,
            [(6.0, 6.0), (20.0, 20.0)],
            BlendMode::Screen,
        );
        let child = add_shape(
            &mut scene,
            inner,
            [(8.0, 8.0), (12.0, 12.0)],
            BlendMode::SourceOver,
        );
        for (node, effect_id, has_effect) in
            [(outer, 31, outer_has_effect), (inner, 37, inner_has_effect)]
        {
            if has_effect {
                scene
                    .set_content_effect(node, effect_id, &[], ContentEffectConfig::default(), 4096)
                    .unwrap();
            }
        }
        scene
            .set_shape_mask_effect(
                child,
                17,
                &[1, 2, 3, 4],
                ShapeMaskEffectConfig::default(),
                4096,
            )
            .unwrap();
        let mut planner = Planner::default();
        let damage = UnsignedPhysicalRect::from_size((32, 32).into());
        let commands: Vec<_> = planner.plan(&scene, 4096, Some(damage)).collect();
        let capture_targets: Vec<_> = commands
            .iter()
            .filter_map(|command| match command.operation {
                RenderOperation::BeginTarget(Target::Texture { texture, .. }) => Some(texture),
                _ => None,
            })
            .collect();
        assert_eq!(capture_targets.len(), 2);
        let effects: Vec<_> = commands
            .iter()
            .filter_map(|command| match command.operation {
                RenderOperation::ApplyEffect(effect) => Some(effect),
                _ => None,
            })
            .collect();
        assert_eq!(
            effects.len(),
            1 + usize::from(outer_has_effect) + usize::from(inner_has_effect)
        );
        let content_sources: Vec<_> = [(37, inner_has_effect), (31, outer_has_effect)]
            .into_iter()
            .zip(capture_targets.iter().rev())
            .map(|((effect_id, has_effect), &capture_texture)| {
                let effect = effects.iter().find(|effect| effect.effect_id == effect_id);
                assert_eq!(effect.is_some(), has_effect);
                effect.map_or(capture_texture, |effect| {
                    assert_eq!(effect.input, capture_texture);
                    effect.output
                })
            })
            .collect();
        let mask_effect = effects
            .iter()
            .find(|effect| effect.effect_id == 17)
            .unwrap();
        let composites: Vec<_> = commands
            .iter()
            .filter_map(|command| match command.operation {
                RenderOperation::CompositeTexture(composite) => {
                    Some((composite.blend_mode, composite.source))
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            composites,
            [
                (BlendMode::SourceOver, mask_effect.output),
                (BlendMode::Screen, content_sources[0]),
                (BlendMode::Multiply, content_sources[1]),
            ]
        );
        assert_eq!(
            commands
                .iter()
                .filter(|command| matches!(command.operation, RenderOperation::BeginTarget(_)))
                .count(),
            commands
                .iter()
                .filter(|command| matches!(command.operation, RenderOperation::EndTarget))
                .count()
        );
    }
}

#[test]
fn captures_without_effects_prepare_child_masks_outside_surface_damage() {
    for blend_mode in [BlendMode::SourceOver, BlendMode::Multiply] {
        let mut scene = SceneState::new(
            SceneStateContext::default(),
            Viewport {
                physical_size: (32, 32),
                scale_factor: 1.0,
            },
        );
        let root = scene.insert_node(DrawTreeNode::Group(Default::default()), None, 4096);
        let parent = add_shape(&mut scene, root, [(4.0, 4.0), (12.0, 12.0)], blend_mode);
        let child = add_shape(
            &mut scene,
            parent,
            [(8.0, 8.0), (10.0, 10.0)],
            BlendMode::SourceOver,
        );
        scene
            .set_shape_mask_effect(
                child,
                17,
                &[1, 2, 3, 4],
                ShapeMaskEffectConfig::default(),
                4096,
            )
            .unwrap();
        let damage = UnsignedPhysicalRect::new((4, 4).into(), (5, 5).into());
        assert!(!scene
            .shape_mask_effect(child)
            .unwrap()
            .bounds
            .logical_screen_bounds
            .intersects(&damage.to_f32()));
        let mut planner = Planner::default();
        let commands: Vec<_> = planner.plan(&scene, 4096, Some(damage)).collect();
        let has_capture = blend_mode != BlendMode::SourceOver;
        assert_eq!(
            commands.iter().any(|command| matches!(
                command.operation,
                RenderOperation::DrawShapeMask(mask) if mask.shape == ShapeDrawId(child)
            )),
            has_capture
        );
        assert_eq!(
            commands.iter().any(|command| matches!(
                command.operation,
                RenderOperation::DrawShape(draw) if draw.id == ShapeDrawId(child)
            )),
            has_capture
        );
        let effects: Vec<_> = commands
            .iter()
            .filter_map(|command| match command.operation {
                RenderOperation::ApplyEffect(effect) => Some(effect),
                _ => None,
            })
            .collect();
        assert_eq!(effects.len(), usize::from(has_capture));
        let capture_target = commands.iter().find_map(|command| match command.operation {
            RenderOperation::BeginTarget(Target::Texture {
                texture, bounds, ..
            }) => {
                assert_eq!(bounds.min.to_tuple(), (4, 4));
                assert_eq!(bounds.max.to_tuple(), (12, 12));
                Some(texture)
            }
            _ => None,
        });
        assert_eq!(capture_target.is_some(), has_capture);
        let composites: Vec<_> = commands
            .iter()
            .filter_map(|command| match command.operation {
                RenderOperation::CompositeTexture(composite) => Some((composite, command.clip)),
                _ => None,
            })
            .collect();
        if let Some(texture) = capture_target {
            assert_eq!(effects[0].effect_id, 17);
            assert_eq!(effects[0].parameters.bytes, &[1, 2, 3, 4]);
            assert_eq!(composites.len(), 2);
            assert_eq!(composites[0].0.source, effects[0].output);
            assert_eq!(composites[1].0.source, texture);
            assert_eq!(
                composites[1].1.scissor,
                UnsignedPhysicalRect::new((0, 0).into(), (32, 32).into())
            );
        } else {
            assert!(composites.is_empty());
        }
    }
}
