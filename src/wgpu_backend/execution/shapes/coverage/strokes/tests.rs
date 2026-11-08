use super::{CoverageKey, StrokeAtlas, StrokeKey};
use crate::core::util::ShapeResources;
use crate::core::{Shape, StrokeOptions, TransformInstance, Viewport};

#[test]
fn offscreen_stroke_replacements_do_not_queue_or_retain_atlas_work() {
    let tessellation = Shape::rect([(0.0, 0.0), (8.0, 8.0)])
        .tessellate_stroke(StrokeOptions::new(2.0), &mut ShapeResources::new())
        .unwrap();
    let viewport = Viewport {
        physical_size: (64, 64),
        scale_factor: 1.0,
    };
    let mut atlas = StrokeAtlas::default();

    // Offscreen replacements produce no damage, so the renderer need not run the atlas.
    for offset in 100..200 {
        let transform = Some(TransformInstance::translation(offset as f32, 0.0));
        let coverage = atlas.prepare(&tessellation, transform, viewport, None);
        assert!(coverage.regions().is_empty());
        let key = StrokeKey {
            coverage: CoverageKey::new(&tessellation, transform),
            gradient_material: None,
        };
        assert!(atlas.entries.get(&key).is_none());
    }
    assert!(atlas.pending_masks.is_empty());
    assert!(atlas.pending_gradients.is_empty());
}
