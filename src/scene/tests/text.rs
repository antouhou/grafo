use super::{queue_shape, scene, surface};
use crate::core::{Color, Shape, ShapeDrawCommandOptions};
use crate::CosmicText;
use cosmic_text::{Attrs, Buffer, Family, FontSystem, Metrics, Shaping};
use std::sync::Arc;

#[test]
fn text_replacement_preserves_the_node_and_failed_uploads_preserve_geometry() {
    let mut fonts = FontSystem::new();
    let mut buffer = Buffer::new(&mut fonts, Metrics::new(12.0, 16.0));
    let attributes = Attrs::new().family(Family::SansSerif);
    buffer.set_text("O", &attributes, Shaping::Advanced, None);
    buffer.shape_until_scroll(&mut fonts, false);

    let mut scene = scene();
    let root = queue_shape(&mut scene, false);
    let text = queue_shape(&mut scene, true);
    let options = ShapeDrawCommandOptions::new()
        .color(Color::BLACK)
        .clips_children(false);
    let child = scene
        .add_shape(
            Shape::rect([(0.0, 0.0), (1.0, 1.0)]),
            Some(text),
            None,
            options.clone(),
        )
        .unwrap();
    assert_eq!(
        scene
            .replace_with_text(text, CosmicText::new(&buffer, &mut fonts), options.clone())
            .unwrap(),
        text
    );
    assert_eq!(scene.state.draw_tree.children(text), &[child]);
    assert_eq!(
        scene.state.draw_tree.parent_index_unchecked(text),
        Some(root)
    );
    assert!(scene.state.content_effects.contains_key(&text));
    assert!(scene.state.shape_mask_effects.contains_key(&text));
    let geometry = Arc::clone(&scene.state.shape(text).unwrap().cached_shape.tessellation);
    let mut surface = surface();
    scene.render(&mut surface).unwrap();
    scene.render(&mut surface).unwrap();
    assert_eq!(scene.backend.root_scissor, None);

    buffer.set_text("O\nO", &attributes, Shaping::Advanced, None);
    buffer.shape_until_scroll(&mut fonts, false);
    scene.backend.registration_failure = Some(text);
    assert!(scene
        .replace_with_text(text, CosmicText::new(&buffer, &mut fonts), options.clone())
        .is_err());
    assert!(Arc::ptr_eq(
        &geometry,
        &scene.state.shape(text).unwrap().cached_shape.tessellation
    ));
    scene.backend.registration_failure = None;
    scene
        .replace_with_text(text, CosmicText::new(&buffer, &mut fonts), options.clone())
        .unwrap();
    let updated = &scene.state.shape(text).unwrap().cached_shape.tessellation;
    assert!((updated.local_bounds[1].1 - geometry.local_bounds[1].1 - 16.0).abs() < 0.001);
    scene.render(&mut surface).unwrap();
    assert!(scene.backend.root_scissor.is_some());

    buffer.set_text("", &attributes, Shaping::Advanced, None);
    buffer.shape_until_scroll(&mut fonts, false);
    scene
        .replace_with_text(text, CosmicText::new(&buffer, &mut fonts), options.clone())
        .unwrap();
    assert!(scene
        .state
        .shape(text)
        .unwrap()
        .cached_shape
        .vertex_buffers()
        .vertices
        .is_empty());
    scene.render(&mut surface).unwrap();
    assert!(scene.backend.root_scissor.is_some());
    let empty = scene
        .add_text(CosmicText::new(&buffer, &mut fonts), Some(root), options)
        .unwrap();
    let mut removed = Vec::new();
    scene.remove_subtrees([text, empty], |id| removed.push(id));
    assert_eq!(removed.len(), 3);
    assert!([text, child, empty].iter().all(|id| removed.contains(id)));
    assert_eq!(scene.backend.registered_shapes, [root]);
    assert!(scene.state.content_effects.is_empty());
    assert!(scene.state.shape_mask_effects.is_empty());
}
