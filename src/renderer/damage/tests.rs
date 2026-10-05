use super::PendingClipDamage;
use crate::core::{Shape, ShapeDrawCommandOptions, UnsignedPhysicalRect, Viewport};
use crate::scene::Scene;

#[test]
fn overlapping_clip_changes_visit_each_descendant_once() {
    let mut scene = Scene::default();
    let shape = scene.tessellate(&Shape::rect([(40.0, 40.0), (48.0, 48.0)]), None);
    let mut pending = PendingClipDamage::default();
    let mut parent = None;
    for _ in 0..256 {
        let node_id = scene
            .add_shape(shape.clone(), parent, ShapeDrawCommandOptions::new())
            .unwrap();
        pending.insert(node_id);
        pending.insert(node_id);
        parent = Some(node_id);
    }
    assert!(pending.visited.is_empty());
    let mut bounds = None;
    pending.apply(
        &scene,
        &mut bounds,
        Viewport {
            physical_size: (1024, 512),
            scale_factor: 1.0,
        },
        0.75,
    );
    assert_eq!(pending.visited.len(), 256);
    assert!(pending.roots.is_empty());
    assert!(pending.stack.is_empty());
    assert_eq!(
        bounds,
        Some(UnsignedPhysicalRect::new((39, 39).into(), (49, 49).into()))
    );
}
