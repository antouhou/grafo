use super::{glyph_ids, options, scene, surface, Glyphs};
use crate::core::{
    BackdropEffectConfig, ContentEffectConfig, Shape, ShapeMaskEffectConfig, TransformInstance,
};
use crate::scene_state::effects::EffectConfig;

#[derive(Clone, Copy)]
enum ReplacedNodeKind {
    Shape,
    ClipRect,
    Group,
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
            let ancestor = scene.add_group(None).unwrap();
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
                ReplacedNodeKind::Group => scene.add_group(Some(ancestor)).unwrap(),
            };
            let has_own_effect = !matches!(kind, ReplacedNodeKind::ClipRect);
            if has_own_effect {
                scene.set_effect(node, 9, &[1, 2, 3, 4], config).unwrap();
            }
            let group = scene.add_group(Some(node)).unwrap();
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
                assert!(!scene.state.text_glyphs.contains(&id));
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
            assert!(scene.state.text_glyphs.is_empty());
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
        let ancestor = scene.add_group(None).unwrap();
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
