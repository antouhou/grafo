use super::{queue_shape, scene, surface, Scene, TestBackend};
use crate::commands::RenderOperation;
use crate::core::{
    BackdropEffectConfig, Color, ContentEffectConfig, GlyphContent, MathRect, PositionedGlyph,
    Shape, ShapeDrawCommandOptions, ShapeMaskEffectConfig, TextSource, TransformInstance,
};
use crate::scene::draw_commands::{DrawCommand, NodeTarget};
use crate::scene_state::effects::EffectConfig;
use crate::scene_state::types::DrawTreeNode;
use crate::scene_state::SceneStateContext;
use lyon::path::PathEvent;
use std::sync::Arc;

mod replacement;

struct Glyphs {
    glyphs: Vec<PositionedGlyph<u64>>,
    events: Vec<PathEvent>,
    cursor: usize,
    extractions: Vec<u64>,
    visible_bounds: Option<MathRect>,
}

impl Glyphs {
    fn new(keys: &[u64]) -> Self {
        let Shape::Path(path) = Shape::builder()
            .begin((0.0, 0.0))
            .line_to((4.0, 0.0))
            .line_to((4.0, 6.0))
            .line_to((0.0, 6.0))
            .close()
            .build()
        else {
            unreachable!()
        };
        Self {
            glyphs: keys
                .iter()
                .enumerate()
                .map(|(index, &key)| PositionedGlyph {
                    key,
                    position: (index as f32 * 6.0, 2.0),
                })
                .collect(),
            events: path.path.iter().collect(),
            cursor: 0,
            extractions: Vec::new(),
            visible_bounds: None,
        }
    }
}

impl TextSource for Glyphs {
    type GlyphKey = u64;

    fn set_visible_bounds(&mut self, bounds: Option<MathRect>) {
        self.visible_bounds = bounds;
    }

    fn next_glyph(&mut self) -> Option<PositionedGlyph<u64>> {
        let glyph = self.glyphs.get(self.cursor).copied()?;
        self.cursor += 1;
        Some(glyph)
    }

    fn glyph_content(&mut self, key: &u64) -> Option<GlyphContent<'_>> {
        self.extractions.push(*key);
        (*key != 0).then_some(GlyphContent::Outline(&self.events))
    }
}

fn options() -> ShapeDrawCommandOptions {
    ShapeDrawCommandOptions::new().color(Color::BLACK)
}

fn glyph_ids(scene: &Scene<TestBackend>, group: usize) -> Vec<usize> {
    scene
        .state
        .draw_tree
        .children(group)
        .iter()
        .copied()
        .filter(|id| scene.state.text_glyphs.contains(id))
        .collect()
}

#[test]
fn combined_text_effects_draw_each_glyph_once() {
    let glyph_counts = [16, 32, 64, 128];
    let mut command_counts = Vec::new();
    let mut draw_counts = Vec::new();
    for glyph_count in glyph_counts {
        let mut scene = scene();
        scene.resize((4096, 32));
        let keys = vec![10; glyph_count];
        let text = scene.add_text(Glyphs::new(&keys), None, options()).unwrap();
        scene
            .set_content_effect(text, 8, &[1, 2, 3, 4], ContentEffectConfig::default())
            .unwrap();
        scene
            .set_backdrop_effect(text, 9, &[1, 2, 3, 4], BackdropEffectConfig::default())
            .unwrap();
        let commands: Vec<_> = scene.planner.plan(&scene.state, 4096).collect();
        command_counts.push(commands.len());
        draw_counts.push(
            commands
                .iter()
                .filter(|command| matches!(command.operation, RenderOperation::DrawShape(_)))
                .count(),
        );
    }
    assert_eq!(draw_counts, glyph_counts, "commands {command_counts:?}");
    assert!(
        command_counts
            .windows(2)
            .all(|counts| counts[1] <= counts[0] * 2 + 2),
        "commands {command_counts:?}"
    );
}

#[test]
fn text_replacement_preserves_inherited_and_explicit_effects() {
    for config in [
        EffectConfig::Content(ContentEffectConfig::default()),
        EffectConfig::Backdrop(BackdropEffectConfig::default()),
        EffectConfig::ShapeMask(ShapeMaskEffectConfig::default()),
    ] {
        for (has_explicit_effect, is_initially_empty) in
            [(false, false), (false, true), (true, false), (true, true)]
        {
            let mut scene = scene();
            let parent = scene.add_group(None).unwrap();
            scene.set_effect(parent, 8, &[1, 2, 3, 4], config).unwrap();
            let child = scene
                .add_shape(
                    Shape::rect([(0.0, 0.0), (16.0, 16.0)]),
                    Some(parent),
                    None,
                    options(),
                )
                .unwrap();
            if has_explicit_effect {
                scene.set_effect(child, 9, &[1, 2, 3, 4], config).unwrap();
            }
            let glyph_keys: &[u64] = if is_initially_empty { &[] } else { &[10, 11] };
            scene
                .replace_with_text(child, Glyphs::new(glyph_keys), options())
                .unwrap();
            let mut surface = surface();
            scene.render(&mut surface).unwrap();

            scene.set_effect(parent, 10, &[1, 2, 3, 4], config).unwrap();
            if is_initially_empty {
                scene
                    .replace_with_text(child, Glyphs::new(&[10, 11]), options())
                    .unwrap();
            }
            scene.render(&mut surface).unwrap();
            let expected_effect_id = if has_explicit_effect { 9 } else { 10 };
            assert_eq!(
                surface.resource().effects,
                [expected_effect_id; 2],
                "{config:?}, explicit effect: {has_explicit_effect}, initially empty: {is_initially_empty}"
            );

            scene
                .replace_with_text(child, Glyphs::new(&[10, 11]), options())
                .unwrap();
            scene.render(&mut surface).unwrap();
            assert_eq!(surface.resource().effects, [expected_effect_id; 2]);
            match config {
                EffectConfig::Content(_) => scene.remove_content_effect(parent),
                EffectConfig::Backdrop(_) => scene.remove_backdrop_effect(parent),
                EffectConfig::ShapeMask(_) => scene.remove_shape_mask_effect(parent),
            }
            scene.render(&mut surface).unwrap();
            let expected_effects: &[u64] = if has_explicit_effect { &[9, 9] } else { &[] };
            assert_eq!(surface.resource().effects, expected_effects);
        }
    }
}

#[test]
fn glyph_geometry_is_reused_across_positions_replacements_clears_and_scenes() {
    let context = SceneStateContext::default();
    let mut scene = Scene::from_parts(TestBackend::default(), context.clone());
    let mut source = Glyphs::new(&[10, 10, 11, 0, 0]);
    let transform =
        TransformInstance::scale(2.0, 3.0).then(&TransformInstance::translation(1.0, 4.0));
    let text = scene
        .add_text(&mut source, None, options().transform(transform))
        .unwrap();
    assert!(matches!(
        scene.state.draw_tree.get(text),
        Some(DrawTreeNode::Group(_))
    ));
    assert_eq!(source.extractions, [10, 11, 0, 0]);
    let glyphs = glyph_ids(&scene, text);
    assert_eq!(glyphs.len(), 3);
    let first = &scene.state.shape(glyphs[0]).unwrap().cached_shape;
    let geometry = Arc::clone(&first.tessellation);
    let geometry_id = first.geometry_id.unwrap();
    let second = scene.state.shape(glyphs[1]).unwrap();
    assert_eq!(second.cached_shape.geometry_id, Some(geometry_id));
    assert!(Arc::ptr_eq(&geometry, &second.cached_shape.tessellation));
    assert_eq!(second.transform.unwrap().col3, [13.0, 10.0, 0.0, 1.0]);
    assert_ne!(
        scene
            .state
            .shape(glyphs[2])
            .unwrap()
            .cached_shape
            .geometry_id,
        Some(geometry_id)
    );
    let mut surface = surface();
    scene.render(&mut surface).unwrap();
    scene.render(&mut surface).unwrap();
    scene.render(&mut surface).unwrap();
    let mut replacement = Glyphs::new(&[10, 11, 10]);
    scene
        .replace_with_text(text, &mut replacement, options())
        .unwrap();
    assert!(replacement.extractions.is_empty());
    assert!(Arc::ptr_eq(
        &geometry,
        &scene
            .state
            .shape(glyph_ids(&scene, text)[0])
            .unwrap()
            .cached_shape
            .tessellation
    ));
    scene.clear();
    let mut reused = Glyphs::new(&[10]);
    scene.add_text(&mut reused, None, options()).unwrap();
    assert!(reused.extractions.is_empty());
    let mut shared = Scene::from_parts(TestBackend::default(), context);
    let mut reused = Glyphs::new(&[10]);
    let text = shared.add_text(&mut reused, None, options()).unwrap();
    assert!(reused.extractions.is_empty());
    assert!(Arc::ptr_eq(
        &geometry,
        &shared
            .state
            .shape(glyph_ids(&shared, text)[0])
            .unwrap()
            .cached_shape
            .tessellation
    ));
}

#[test]
fn ordinary_replacement_removes_only_text_owned_glyphs() {
    let shape = Shape::rect([(0.0, 0.0), (24.0, 24.0)]);
    for command in [
        DrawCommand::Shape {
            shape: &shape,
            geometry_id: None,
            options: options().clips_children(false),
        },
        DrawCommand::CachedShape {
            cache_key: 1,
            options: options().clips_children(false),
        },
        DrawCommand::ClippingRect {
            rect_bounds: [(0.0, 0.0), (24.0, 24.0)],
            transform: None,
            clips_children: false,
        },
    ] {
        let mut scene = scene();
        scene.load_shape(&shape, 1, None);
        let root = scene.add_group(None).unwrap();
        let text = scene
            .add_text(Glyphs::new(&[10, 11]), Some(root), options())
            .unwrap();
        let glyphs = glyph_ids(&scene, text);
        let child = scene
            .add_shape(&shape, Some(text), None, options())
            .unwrap();
        let nested_text = scene
            .add_text(Glyphs::new(&[12]), Some(text), options())
            .unwrap();
        let nested_glyphs = glyph_ids(&scene, nested_text);
        scene
            .set_content_effect(text, 8, &[1, 2, 3, 4], ContentEffectConfig::default())
            .unwrap();
        scene
            .set_shape_mask_effect(text, 7, &[1, 2, 3, 4], ShapeMaskEffectConfig::default())
            .unwrap();
        scene
            .set_backdrop_effect(text, 9, &[1, 2, 3, 4], BackdropEffectConfig::default())
            .unwrap();
        let mut surface = surface();
        scene.render(&mut surface).unwrap();

        assert_eq!(
            scene
                .queue_draw_command(NodeTarget::Replace { node_id: text }, command)
                .unwrap(),
            text
        );

        assert_eq!(scene.state.draw_tree.children(root), &[text]);
        assert_eq!(scene.state.draw_tree.children(text), &[child, nested_text]);
        assert_eq!(glyph_ids(&scene, nested_text), nested_glyphs);
        for &glyph in &glyphs {
            assert!(scene.state.draw_tree.get(glyph).is_none());
            assert!(!scene.state.text_glyphs.contains(&glyph));
            assert!(!scene.state.content_effects.contains_key(&glyph));
            assert!(!scene.state.shape_mask_effects.contains_key(&glyph));
            assert!(!scene.state.backdrop_effects.contains_key(&glyph));
            assert!(!scene.backend.registered_shapes.contains(&glyph));
        }
        scene.render(&mut surface).unwrap();
        assert!(surface.resource().draws.contains(&child));
        assert!(nested_glyphs
            .iter()
            .all(|id| surface.resource().draws.contains(id)));
        assert!(glyphs
            .iter()
            .all(|id| !surface.resource().draws.contains(id)));
    }
}

#[test]
fn failed_shape_replacement_preserves_text_glyphs_and_other_children() {
    let mut scene = scene();
    let text = scene
        .add_text(Glyphs::new(&[10, 11]), None, options())
        .unwrap();
    let shape = Shape::rect([(0.0, 0.0), (24.0, 24.0)]);
    let child = scene
        .add_shape(&shape, Some(text), None, options())
        .unwrap();
    let glyphs = glyph_ids(&scene, text);
    let mut surface = surface();
    scene.render(&mut surface).unwrap();
    let registered_shapes = scene.backend.registered_shapes.clone();
    scene.backend.registration_failure = Some(text);

    assert!(scene
        .replace_with_shape(text, shape, None, options())
        .is_err());

    assert!(matches!(
        scene.state.draw_tree.get(text),
        Some(DrawTreeNode::Group(_))
    ));
    assert_eq!(glyph_ids(&scene, text), glyphs);
    assert_eq!(scene.state.draw_tree.children(text).last(), Some(&child));
    assert_eq!(scene.backend.registered_shapes, registered_shapes);
    assert!(scene.dirty_bounds.is_none());
}

#[test]
fn text_effects_apply_to_each_glyph_and_survive_empty_replacements() {
    let mut scene = scene();
    let text = scene.add_text(Glyphs::new(&[]), None, options()).unwrap();
    scene
        .set_content_effect(text, 8, &[1, 2, 3, 4], ContentEffectConfig::default())
        .unwrap();
    scene
        .set_shape_mask_effect(text, 7, &[1, 2, 3, 4], ShapeMaskEffectConfig::default())
        .unwrap();
    scene
        .set_backdrop_effect(text, 9, &[1, 2, 3, 4], BackdropEffectConfig::default())
        .unwrap();
    let mut surface = surface();
    scene.render(&mut surface).unwrap();
    scene
        .replace_with_text(text, Glyphs::new(&[10, 10]), options())
        .unwrap();
    scene.render(&mut surface).unwrap();
    assert_eq!(surface.resource().shape_masks, 2);
    for effect in [7, 8, 9] {
        assert_eq!(
            surface
                .resource()
                .effects
                .iter()
                .filter(|&&id| id == effect)
                .count(),
            2
        );
    }
    let glyphs = glyph_ids(&scene, text);
    scene
        .set_content_effect(text, 8, &[1, 2, 3, 4], ContentEffectConfig::default())
        .unwrap();
    scene
        .set_backdrop_effect(text, 9, &[1, 2, 3, 4], BackdropEffectConfig::default())
        .unwrap();
    scene.remove_content_effect(text);
    scene.remove_backdrop_effect(text);
    scene.remove_shape_mask_effect(text);
    for id in glyphs {
        assert!(!scene.state.content_effects.contains_key(&id));
        assert!(!scene.state.backdrop_effects.contains_key(&id));
        assert!(!scene.state.shape_mask_effects.contains_key(&id));
    }
    scene
        .set_shape_mask_effect(text, 7, &[1, 2, 3, 4], ShapeMaskEffectConfig::default())
        .unwrap();
    scene.unload_effect(7);
    scene
        .replace_with_text(text, Glyphs::new(&[10]), options())
        .unwrap();
    scene.render(&mut surface).unwrap();
    assert!(surface.resource().effects.is_empty());
}

#[test]
fn failed_text_insertion_removes_partial_glyphs_and_the_group() {
    let mut scene = scene();
    let root = queue_shape(&mut scene, false);
    scene.backend.registration_failure = Some(scene.state.next_node_id() + 2);
    assert!(scene
        .add_text(Glyphs::new(&[10, 11]), Some(root), options())
        .is_err());
    assert_eq!(scene.backend.registered_shapes, [root]);
    assert!(scene.state.draw_tree.children(root).is_empty());
}
