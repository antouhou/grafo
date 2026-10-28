use super::{glyph_ids, options, scene, surface, Glyphs};
use crate::core::cache::CachedTessellation;
use crate::core::Shape;
use crate::scene::tests::TestBackend;
use crate::scene::Scene;
use crate::scene_state::SceneStateContext;
use std::sync::{Arc, Weak};

fn geometry(scene: &Scene<TestBackend>, text: usize) -> Vec<(u64, Weak<CachedTessellation>)> {
    glyph_ids(scene, text)
        .into_iter()
        .map(|id| {
            let shape = &scene.state.shape(id).unwrap().cached_shape;
            (
                shape.geometry_id.unwrap(),
                Arc::downgrade(&shape.tessellation),
            )
        })
        .collect()
}

#[test]
fn discarded_glyphs_unload_at_frame_end_without_unloading_explicit_shapes() {
    let mut scene = scene();
    let mut surface = surface();
    scene.load_shape(Shape::rect([(0.0, 0.0), (8.0, 8.0)]), 1, None);

    for batch in 0..3 {
        let keys: Vec<_> = (1..=64).map(|key| key + batch * 64).collect();
        let text = scene.add_text(Glyphs::new(&keys), None, options()).unwrap();
        let discarded = geometry(&scene, text);
        assert_eq!(discarded.len(), 64);
        scene.render(&mut surface).unwrap();
        scene.clear();
        assert!(discarded
            .iter()
            .all(|(key, _)| scene.state.loaded_shape(*key).is_ok()));

        scene.render(&mut surface).unwrap();
        assert!(discarded
            .iter()
            .all(|(key, _)| scene.state.loaded_shape(*key).is_err()));
        scene.render(&mut surface).unwrap();
        assert!(discarded.iter().all(|(_, glyph)| glyph.upgrade().is_none()));
    }

    scene.add_cached_shape(1, None, options()).unwrap();
}

#[test]
fn shared_glyph_unloads_only_after_the_last_node_is_removed_or_dropped() {
    let context = SceneStateContext::default();
    let mut idle = Scene::from_parts(TestBackend::default(), context.clone());
    let mut active = Scene::from_parts(TestBackend::default(), context);
    let text = idle
        .add_text(Glyphs::new(&[10, 10]), None, options())
        .unwrap();
    let retained = geometry(&idle, text);
    let mut surface = surface();
    let mut source = Glyphs::new(&[10]);
    let shared = active.add_text(&mut source, None, options()).unwrap();
    assert!(source.extractions.is_empty());
    assert!(Weak::ptr_eq(
        &retained[0].1,
        &geometry(&active, shared)[0].1
    ));
    idle.remove_subtrees([glyph_ids(&idle, text)[0]], |_| {});
    active.clear();

    for _ in 0..3 {
        active.render(&mut surface).unwrap();
        assert!(active.state.loaded_shape(retained[0].0).is_ok());
    }

    drop(idle);
    active.render(&mut surface).unwrap();
    assert!(active.state.loaded_shape(retained[0].0).is_err());
    assert!(retained.iter().all(|(_, glyph)| glyph.upgrade().is_none()));
}

#[test]
fn reinserting_a_glyph_before_frame_end_cancels_its_unload() {
    let mut scene = scene();
    let text = scene.add_text(Glyphs::new(&[10]), None, options()).unwrap();
    let original = geometry(&scene, text);
    scene.clear();
    let mut source = Glyphs::new(&[10]);
    let text = scene.add_text(&mut source, None, options()).unwrap();
    assert!(source.extractions.is_empty());
    assert!(Weak::ptr_eq(&original[0].1, &geometry(&scene, text)[0].1));
    let mut surface = surface();
    scene.render(&mut surface).unwrap();
    assert!(scene.state.loaded_shape(original[0].0).is_ok());

    let mut source = Glyphs::new(&[11]);
    scene
        .replace_with_text(text, &mut source, options())
        .unwrap();
    let replacement = geometry(&scene, text);
    scene.render(&mut surface).unwrap();
    assert!(scene.state.loaded_shape(original[0].0).is_err());
    scene
        .replace_with_shape(text, Shape::rect([(0.0, 0.0), (8.0, 8.0)]), None, options())
        .unwrap();
    scene.render(&mut surface).unwrap();
    assert!(scene.state.loaded_shape(replacement[0].0).is_err());
}

#[test]
fn failed_text_insertion_releases_partial_glyphs() {
    for failed_glyph in 0..2 {
        let mut scene = scene();
        let root = scene.add_group(None).unwrap();
        scene.backend.registration_failure = Some(scene.state.next_node_id() + 1 + failed_glyph);
        assert!(scene
            .add_text(Glyphs::new(&[10, 11]), Some(root), options())
            .is_err());
        scene.backend.registration_failure = None;
        let text = scene
            .add_text(Glyphs::new(&[10, 11]), Some(root), options())
            .unwrap();
        let discarded = geometry(&scene, text);
        scene.clear();
        let mut surface = surface();
        scene.render(&mut surface).unwrap();
        assert!(discarded
            .iter()
            .all(|(key, _)| scene.state.loaded_shape(*key).is_err()));
        scene.render(&mut surface).unwrap();
        assert!(discarded.iter().all(|(_, glyph)| glyph.upgrade().is_none()));
    }
}

#[test]
fn empty_glyphs_unload_at_frame_end() {
    let mut scene = scene();
    let mut source = Glyphs::new(&[10, 10]);
    source.events.clear();
    scene.add_text(&mut source, None, options()).unwrap();
    assert_eq!(source.extractions, [10]);
    let mut surface = surface();
    scene.render(&mut surface).unwrap();

    let mut source = Glyphs::new(&[10]);
    source.events.clear();
    scene.add_text(&mut source, None, options()).unwrap();
    assert_eq!(source.extractions, [10]);
}
