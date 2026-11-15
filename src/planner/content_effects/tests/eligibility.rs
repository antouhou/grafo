use crate::commands::{RenderOperation, ShapeDrawId, Target};
use crate::core::{
    BlendMode, Color, Shape, ShapeDrawOptions, ShapeMaskEffectConfig, UnsignedPhysicalRect,
    Viewport,
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
