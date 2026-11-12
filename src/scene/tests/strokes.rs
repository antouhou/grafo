use super::{scene, surface, DrawCommandError};
use crate::commands::{RenderOperation, ShapeDrawId};
use crate::core::{
    BackdropEffectConfig, Color, ContentEffectConfig, DrawMode, GlyphContent, PositionedGlyph,
    Shape, ShapeDrawOptions, ShapeMaskEffectConfig, StrokeGeometryOptions, TextSource,
    TransformInstance,
};
use crate::scene_state::SceneError;
use std::sync::Arc;

struct EmptyText;

impl TextSource for EmptyText {
    type GlyphKey = u64;

    fn next_glyph(&mut self) -> Option<PositionedGlyph<u64>> {
        None
    }

    fn glyph_content(&mut self, _: &u64) -> Option<GlyphContent<'_>> {
        None
    }
}

fn stroke_options() -> ShapeDrawOptions {
    ShapeDrawOptions::new()
        .draw_mode(DrawMode::Stroke)
        .color(Color::BLACK)
}

#[test]
fn cached_fill_and_stroke_draws_share_meshes_and_keep_effects_independent() {
    let mut scene = scene();
    let group = scene.add_group(None, Default::default()).unwrap();
    scene
        .load_shape(
            Shape::rect([(4.0, 4.0), (28.0, 28.0)]).with_stroke(StrokeGeometryOptions::new(4.0)),
            1,
            Some(1),
        )
        .unwrap();
    let fill = scene
        .add_cached_shape(1, Some(group), ShapeDrawOptions::new().color(Color::WHITE))
        .unwrap();
    let stroke = scene
        .add_cached_shape(1, Some(group), stroke_options())
        .unwrap();
    let fill_instance = scene.state.shape(fill).unwrap();
    let stroke_instance = scene.state.shape(stroke).unwrap();
    assert_eq!(fill_instance.draw_mode, DrawMode::Fill);
    assert_eq!(stroke_instance.local_bounds(), [(2.0, 2.0), (30.0, 30.0)]);
    assert!(Arc::ptr_eq(
        &fill_instance.cached_shape.fill_tessellation,
        &stroke_instance.cached_shape.fill_tessellation,
    ));
    assert!(Arc::ptr_eq(
        stroke_instance.tessellation().unwrap(),
        fill_instance
            .cached_shape
            .stroke_tessellation
            .as_ref()
            .unwrap(),
    ));
    scene
        .set_shape_mask_effect(stroke, 7, &[1, 2, 3, 4], ShapeMaskEffectConfig::default())
        .unwrap();
    scene
        .set_content_effect(stroke, 8, &[1, 2, 3, 4], ContentEffectConfig::default())
        .unwrap();
    scene
        .set_backdrop_effect(stroke, 9, &[1, 2, 3, 4], BackdropEffectConfig::default())
        .unwrap();
    assert_eq!(
        scene
            .state
            .shape_mask_effect(stroke)
            .unwrap()
            .bounds
            .local_bounds,
        [(1.0, 1.0), (31.0, 31.0)],
    );
    assert!(scene.state.node_effects(fill).iter().next().is_none());

    let mut surface = surface();
    scene.render(&mut surface).unwrap();
    assert_eq!(surface.resource().draws, [fill, stroke]);
    assert_eq!(surface.resource().shape_masks, 1);
    for effect in [7, 8, 9] {
        assert!(surface.resource().effects.contains(&effect));
    }
}

#[test]
fn open_strokes_are_scheduled_when_only_their_width_intersects_damage() {
    let mut scene = scene();
    let stroke = scene
        .add_shape(
            Shape::builder()
                .begin((-2.0, 4.0))
                .line_to((-2.0, 28.0))
                .end()
                .build()
                .with_stroke(StrokeGeometryOptions::new(8.0)),
            None,
            None,
            stroke_options(),
        )
        .unwrap();
    assert!(scene
        .state
        .shape(stroke)
        .unwrap()
        .cached_shape
        .fill_vertex_buffers()
        .indices
        .is_empty());
    let commands: Vec<_> = scene
        .planner
        .plan(&scene.state, 4096, scene.dirty_bounds)
        .collect();
    assert!(commands.iter().any(|command| {
        matches!(command.operation, RenderOperation::DrawShape(draw) if draw.id == ShapeDrawId(stroke))
    }));
}

#[test]
fn stroke_parents_are_rejected_by_every_insertion_api() {
    let mut scene = scene();
    let shape =
        Shape::rect([(4.0, 4.0), (28.0, 28.0)]).with_stroke(StrokeGeometryOptions::new(4.0));
    scene.load_shape(&shape, 1, None).unwrap();
    let stroke = scene.add_cached_shape(1, None, stroke_options()).unwrap();
    let next_id = scene.state.next_node_id();
    let dirty = scene.dirty_bounds;
    for parent in [Some(stroke), None] {
        for result in [
            scene.add_shape(&shape, parent, None, ShapeDrawOptions::new()),
            scene.add_cached_shape(1, parent, ShapeDrawOptions::new()),
            scene.add_clipping_rect(
                [(0.0, 0.0), (16.0, 16.0)],
                parent,
                None::<TransformInstance>,
                true,
            ),
            scene.add_text(EmptyText, parent, ShapeDrawOptions::new()),
        ] {
            assert!(
                matches!(result, Err(DrawCommandError::Scene(SceneError::StrokeDrawMustBeLeaf(id))) if id == stroke)
            );
        }
        assert!(matches!(
            scene.add_group(parent, Default::default()),
            Err(SceneError::StrokeDrawMustBeLeaf(id)) if id == stroke
        ));
    }
    assert_eq!(scene.state.next_node_id(), next_id);
    assert_eq!(scene.backend.registered_shapes, [stroke]);
    assert!(scene.state.draw_tree.children(stroke).is_empty());
    assert_eq!(scene.dirty_bounds, dirty);
}

#[test]
fn stroke_replacement_preserves_non_leaf_nodes_on_rejection() {
    for is_group in [false, true] {
        for is_cached in [false, true] {
            let mut scene = scene();
            let shape = Shape::rect([(4.0, 4.0), (28.0, 28.0)])
                .with_stroke(StrokeGeometryOptions::new(4.0));
            scene.load_shape(&shape, 1, None).unwrap();
            let parent = if is_group {
                scene.add_group(None, Default::default()).unwrap()
            } else {
                scene
                    .add_cached_shape(1, None, ShapeDrawOptions::new())
                    .unwrap()
            };
            let child = scene.add_group(Some(parent), Default::default()).unwrap();
            let registered = scene.backend.registered_shapes.clone();
            let dirty = scene.dirty_bounds;
            scene.backend.should_fail = true;
            let result = if is_cached {
                scene.replace_with_cached_shape(parent, 1, stroke_options())
            } else {
                scene.replace_with_shape(parent, &shape, None, stroke_options())
            };
            assert!(
                matches!(result, Err(DrawCommandError::Scene(SceneError::StrokeDrawMustBeLeaf(id))) if id == parent)
            );
            assert!(!scene.state.draw_tree.get(parent).unwrap().is_stroke_draw());
            assert_eq!(scene.state.draw_tree.children(parent), &[child]);
            assert_eq!(scene.backend.registered_shapes, registered);
            assert_eq!(scene.dirty_bounds, dirty);
        }
    }
}

#[test]
fn missing_and_zero_width_strokes_do_not_schedule_fill_draws() {
    for shape in [
        Shape::rect([(4.0, 4.0), (28.0, 28.0)]),
        Shape::rect([(4.0, 4.0), (28.0, 28.0)]).with_stroke(StrokeGeometryOptions::new(0.0)),
    ] {
        let mut scene = scene();
        scene
            .add_shape(shape, None, None, stroke_options())
            .unwrap();
        let mut surface = surface();
        scene.render(&mut surface).unwrap();
        assert!(surface.resource().draws.is_empty());
    }
}
