use super::{glyph_ids, options, scene, surface, Glyphs};
use crate::commands::RenderOperation;
use crate::core::{
    BackdropEffectConfig, Color, ContentEffectConfig, DrawMode, Shape, ShapeDrawOptions,
    ShapeMaskEffectConfig, StrokeGeometryOptions, TransformInstance, UnsignedPhysicalRect,
};
use crate::scene::tests::path_with_nan_positions;
use crate::scene::DrawCommandError;
use crate::scene_state::effects::EffectConfig;
use crate::scene_state::types::DrawTreeNode;
use crate::scene_state::SceneError;

#[derive(Clone, Copy)]
enum ReplacedNodeKind {
    Shape,
    ClipRect,
    Group,
}

#[test]
fn stroke_to_text_replacement_commits_glyphs_or_preserves_the_stroke() {
    for should_fail in [false, true] {
        let mut scene = scene();
        let stroke = scene
            .add_shape(
                Shape::rect([(2.0, 2.0), (20.0, 20.0)])
                    .with_stroke(StrokeGeometryOptions::new(4.0)),
                None,
                None,
                ShapeDrawOptions::new()
                    .draw_mode(DrawMode::Stroke)
                    .color(Color::BLACK),
            )
            .unwrap();
        scene
            .set_content_effect(stroke, 8, &[1, 2, 3, 4], ContentEffectConfig::default())
            .unwrap();
        let mut surface = surface();
        scene.render(&mut surface).unwrap();
        let bounds = scene
            .state
            .draw_tree
            .get(stroke)
            .unwrap()
            .logical_screen_bounds();
        let effect_bounds = scene.state.content_effects[&stroke]
            .bounds
            .unwrap()
            .capture_region
            .bounds;
        if should_fail {
            scene.backend.registration_failure = Some(scene.state.next_node_id() + 1);
        }

        let result = scene.replace_with_text(stroke, Glyphs::new(&[10, 11]), options());

        if should_fail {
            assert!(matches!(result, Err(DrawCommandError::Backend(_))));
            assert_eq!(
                scene.state.shape(stroke).unwrap().draw_mode,
                DrawMode::Stroke
            );
            assert!(scene.state.draw_tree.children(stroke).is_empty());
            assert_eq!(scene.backend.registered_shapes, [stroke]);
            assert_eq!(
                scene
                    .state
                    .draw_tree
                    .get(stroke)
                    .unwrap()
                    .logical_screen_bounds(),
                bounds
            );
            assert_eq!(
                scene.state.content_effects[&stroke]
                    .bounds
                    .unwrap()
                    .capture_region
                    .bounds,
                effect_bounds,
            );
            assert!(scene.dirty_bounds.is_none());
        } else {
            assert_eq!(result.unwrap(), stroke);
            assert!(matches!(
                scene.state.draw_tree.get(stroke),
                Some(DrawTreeNode::Group(_))
            ));
            let glyphs = glyph_ids(&scene, stroke);
            assert_eq!(glyphs.len(), 2);
            scene.render(&mut surface).unwrap();
            assert_eq!(surface.resource().draws, glyphs);
            assert!(!scene.backend.registered_shapes.contains(&stroke));
            assert!(scene
                .state
                .node_effects(stroke)
                .iter()
                .any(|effect| effect.instance.effect_id == 8));
        }
    }
}

#[test]
fn glyph_tessellation_error_preserves_the_previous_text() {
    let mut scene = scene();
    let text = scene.add_text(Glyphs::new(&[10]), None, options()).unwrap();
    let previous_glyphs = glyph_ids(&scene, text);
    scene.render(&mut surface()).unwrap();
    let mut replacement = Glyphs::new(&[11, 12]);
    replacement.outline_override = Some((12, path_with_nan_positions().iter().collect()));

    let result = scene.replace_with_text(text, &mut replacement, options());

    assert!(matches!(
        result,
        Err(DrawCommandError::Scene(SceneError::Tessellation(_)))
    ));
    assert_eq!(replacement.extractions, [11, 12]);
    assert_eq!(glyph_ids(&scene, text), previous_glyphs);
    assert_eq!(scene.backend.registered_shapes, previous_glyphs);
    assert!(scene.dirty_bounds.is_none());
}

#[test]
fn clip_replacement_validates_only_surviving_text_glyphs() {
    for should_preserve_glyphs in [false, true] {
        let mut scene = scene();
        let ancestor = scene.add_group(None, Default::default()).unwrap();
        let node = if should_preserve_glyphs {
            scene.add_group(Some(ancestor), Default::default()).unwrap()
        } else {
            scene
                .add_text(Glyphs::new(&[10, 11]), Some(ancestor), options())
                .unwrap()
        };
        scene
            .set_shape_mask_effect(node, 7, &[1, 2, 3, 4], ShapeMaskEffectConfig::default())
            .unwrap();
        let text = if should_preserve_glyphs {
            scene
                .add_text(Glyphs::new(&[10, 11]), Some(node), options())
                .unwrap()
        } else {
            node
        };
        let glyphs = glyph_ids(&scene, text);
        scene
            .set_shape_mask_effect(
                ancestor,
                8,
                &[1, 2, 3, 4],
                ShapeMaskEffectConfig::new().outset(f32::MAX),
            )
            .unwrap();
        let mut surface = surface();
        scene.render(&mut surface).unwrap();

        let result = scene.replace_with_clipping_rect(
            node,
            [(0.0, 0.0), (24.0, 24.0)],
            None::<TransformInstance>,
            true,
        );

        if should_preserve_glyphs {
            assert!(
                matches!(
                    &result,
                    Err(DrawCommandError::Scene(SceneError::InvalidShapeMaskEffectBounds(id))) if *id == glyphs[0]
                ),
                "{result:?}"
            );
            assert!(matches!(
                scene.state.draw_tree.get(node),
                Some(DrawTreeNode::Group(_))
            ));
            assert_eq!(glyph_ids(&scene, text), glyphs);
            for glyph_id in glyphs {
                assert_eq!(scene.state.shape_mask_effects[&glyph_id].effect_id, 7);
            }
            assert!(scene.dirty_bounds.is_none());
        } else {
            assert_eq!(result.unwrap(), node);
            assert!(matches!(
                scene.state.draw_tree.get(node),
                Some(DrawTreeNode::ClipRect(_))
            ));
            assert!(scene.state.draw_tree.children(node).is_empty());
            assert!(!scene.state.has_text_glyphs());
            assert!(scene.backend.registered_shapes.is_empty());
        }
        scene.render(&mut surface).unwrap();
        assert_eq!(surface.resource().shape_masks, 0);
        let damage = Some(UnsignedPhysicalRect::from_size(
            scene.state.viewport().physical_size.into(),
        ));
        let planned_masks = scene
            .planner
            .plan(&scene.state, 4096, damage)
            .filter(|command| matches!(command.operation, RenderOperation::DrawShapeMask(_)))
            .count();
        assert_eq!(planned_masks, if should_preserve_glyphs { 2 } else { 0 });
    }
}

#[test]
fn text_replacement_removes_previous_subtrees_and_preserves_node_effects() {
    for config in [
        EffectConfig::Content(ContentEffectConfig::default()),
        EffectConfig::Backdrop(BackdropEffectConfig::default()),
        EffectConfig::ShapeMask(ShapeMaskEffectConfig::default()),
    ] {
        for kind in [
            ReplacedNodeKind::Shape,
            ReplacedNodeKind::ClipRect,
            ReplacedNodeKind::Group,
        ] {
            let mut scene = scene();
            let ancestor = scene.add_group(None, Default::default()).unwrap();
            scene
                .set_effect(ancestor, 8, &[1, 2, 3, 4], config)
                .unwrap();
            let bounds = [(0.0, 0.0), (24.0, 24.0)];
            let node = match kind {
                ReplacedNodeKind::Shape => scene
                    .add_shape(Shape::rect(bounds), Some(ancestor), None, options())
                    .unwrap(),
                ReplacedNodeKind::ClipRect => scene
                    .add_clipping_rect(bounds, Some(ancestor), None::<TransformInstance>, true)
                    .unwrap(),
                ReplacedNodeKind::Group => {
                    scene.add_group(Some(ancestor), Default::default()).unwrap()
                }
            };
            let has_own_effect = !matches!(kind, ReplacedNodeKind::ClipRect);
            if has_own_effect {
                scene.set_effect(node, 9, &[1, 2, 3, 4], config).unwrap();
            }
            let group = scene.add_group(Some(node), Default::default()).unwrap();
            scene.set_effect(group, 10, &[1, 2, 3, 4], config).unwrap();
            let child = scene
                .add_shape(Shape::rect(bounds), Some(group), None, options())
                .unwrap();
            scene.set_effect(child, 11, &[1, 2, 3, 4], config).unwrap();
            let nested_text = scene
                .add_text(Glyphs::new(&[10]), Some(group), options())
                .unwrap();
            let mut removed_ids = vec![group, child, nested_text];
            removed_ids.extend(glyph_ids(&scene, nested_text));
            let previous_children = scene.state.draw_tree.children(node).to_vec();
            let registered_shapes = scene.backend.registered_shapes.clone();
            let mut surface = surface();
            scene.render(&mut surface).unwrap();

            scene.backend.registration_failure = Some(scene.state.next_node_id() + 1);
            assert!(scene
                .replace_with_text(node, Glyphs::new(&[20, 21]), options())
                .is_err());
            assert_eq!(scene.state.draw_tree.children(node), previous_children);
            assert_eq!(scene.backend.registered_shapes, registered_shapes);
            assert!(removed_ids
                .iter()
                .all(|&id| scene.state.draw_tree.get(id).is_some()));
            assert!(scene.dirty_bounds.is_none());
            scene.backend.registration_failure = None;
            assert_eq!(
                scene
                    .replace_with_text(node, Glyphs::new(&[20, 21]), options())
                    .unwrap(),
                node
            );

            let glyphs = glyph_ids(&scene, node);
            assert_eq!(
                scene.state.draw_tree.parent_index_unchecked(node),
                Some(ancestor)
            );
            assert_eq!(scene.state.draw_tree.children(node), glyphs);
            assert_eq!(glyphs.len(), 2);
            for id in removed_ids {
                assert!(scene.state.draw_tree.get(id).is_none());
                assert!(!scene.backend.registered_shapes.contains(&id));
                assert!(!scene.state.is_text_glyph(id));
                assert!(!scene.state.content_effects.contains_key(&id));
                assert!(!scene.state.backdrop_effects.contains_key(&id));
                assert!(!scene.state.shape_mask_effects.contains_key(&id));
            }
            let expected = if has_own_effect {
                (9, Some(node))
            } else {
                (8, Some(ancestor))
            };
            for &id in &glyphs {
                let effect = scene.state.node_effects(id).get(config.kind()).unwrap();
                assert_eq!(
                    (effect.instance.effect_id, effect.source_group_id),
                    expected
                );
            }
            scene.render(&mut surface).unwrap();
            assert_eq!(surface.resource().draws, glyphs);
            scene.set_effect(node, 12, &[1, 2, 3, 4], config).unwrap();
            scene.render(&mut surface).unwrap();
            assert_eq!(surface.resource().effects, [12, 12]);

            scene.backend.registration_failure = Some(scene.state.next_node_id());
            assert!(scene
                .replace_with_text(node, Glyphs::new(&[22]), options())
                .is_err());
            assert_eq!(glyph_ids(&scene, node), glyphs);
            assert!(scene.dirty_bounds.is_none());
            scene.backend.registration_failure = None;
            scene
                .replace_with_text(node, Glyphs::new(&[22]), options())
                .unwrap();
            assert!(glyphs
                .iter()
                .all(|&id| scene.state.draw_tree.get(id).is_none()));
            scene.render(&mut surface).unwrap();
            assert_eq!(surface.resource().effects, [12]);
            scene
                .replace_with_text(node, Glyphs::new(&[]), options())
                .unwrap();
            assert!(scene.state.draw_tree.children(node).is_empty());
            assert!(scene.backend.registered_shapes.is_empty());
            assert!(!scene.state.has_text_glyphs());
            scene.render(&mut surface).unwrap();
            assert!(surface.resource().draws.is_empty());
            assert!(surface.resource().effects.is_empty());
        }
    }
}

#[test]
fn discarded_children_do_not_restrict_text_effect_bounds() {
    for is_text_empty in [false, true] {
        let mut scene = scene();
        let ancestor = scene.add_group(None, Default::default()).unwrap();
        scene
            .set_shape_mask_effect(ancestor, 7, &[1, 2, 3, 4], ShapeMaskEffectConfig::default())
            .unwrap();
        let node = scene
            .add_shape(
                Shape::rect([(0.0, 0.0), (16.0, 16.0)]),
                Some(ancestor),
                None,
                options(),
            )
            .unwrap();
        let child = scene
            .add_shape(
                Shape::rect([(0.0, 0.0), (f32::MAX, f32::MAX)]),
                Some(node),
                None,
                options(),
            )
            .unwrap();
        let keys: &[u64] = if is_text_empty { &[] } else { &[10] };

        scene
            .replace_with_text(node, Glyphs::new(keys), options())
            .unwrap();

        assert!(scene.state.draw_tree.get(child).is_none());
        let glyphs = glyph_ids(&scene, node);
        assert_eq!(scene.state.draw_tree.children(node), glyphs);
        assert_eq!(glyphs.len(), keys.len());
        for id in glyphs {
            assert_eq!(scene.state.shape_mask_effects[&id].effect_id, 7);
        }
        let mut surface = surface();
        scene.render(&mut surface).unwrap();
        assert_eq!(surface.resource().effects, vec![7; keys.len()]);
    }
}
