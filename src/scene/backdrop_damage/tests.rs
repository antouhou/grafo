use super::{BackdropDamageEntry, BackdropDamageIndex};
use crate::core::effect::backdrops;
use crate::core::{
    BackdropCaptureArea, BackdropCaptureRegion, BackdropEffectConfig, MathRect, Shape,
    ShapeDrawCommandOptions, UnsignedPhysicalRect, Viewport,
};
use crate::scene::types::{CachedShapeDrawData, DrawTreeNode};
use crate::scene::Scene;

fn rect(min: (u32, u32), max: (u32, u32)) -> UnsignedPhysicalRect {
    UnsignedPhysicalRect::new(min.into(), max.into())
}

fn entry(
    node_id: usize,
    capture_region: UnsignedPhysicalRect,
    target_region: UnsignedPhysicalRect,
) -> BackdropDamageEntry {
    BackdropDamageEntry {
        node_id,
        capture_region: Some(capture_region),
        target_region: Some(target_region),
    }
}

fn assert_matching_nodes(
    index: &BackdropDamageIndex,
    query: UnsignedPhysicalRect,
    expected: &[usize],
) {
    let mut nodes = [0; 4];
    let mut count = 0;
    index.for_each_intersecting(query, |entry| {
        nodes[count] = entry.node_id;
        count += 1;
    });
    let nodes = &mut nodes[..count];
    nodes.sort_unstable();
    assert_eq!(nodes, expected);
}

fn shape_node(bounds: [(f32, f32); 2]) -> DrawTreeNode {
    let shape = Scene::default().tessellate(&Shape::rect(bounds), None);
    DrawTreeNode::CachedShape(CachedShapeDrawData::new(
        shape,
        ShapeDrawCommandOptions::new(),
    ))
}

fn capture(bounds: [(f32, f32); 2], viewport: Viewport) -> Option<BackdropCaptureRegion> {
    backdrops::compute_backdrop_capture_region(
        MathRect::zero(),
        BackdropEffectConfig::new().capture_area(BackdropCaptureArea::ScreenRect(bounds)),
        viewport.scale_factor,
        viewport.physical_size.into(),
        1024,
    )
}

#[test]
fn replacement_removal_and_rebuild_keep_spatial_queries_and_storage_consistent() {
    let mut index = BackdropDamageIndex::default();
    let original = entry(3, rect((10, 10), (20, 20)), rect((500, 10), (520, 30)));
    let unrelated = entry(
        4,
        rect((800, 300), (820, 320)),
        rect((900, 300), (920, 320)),
    );
    index.replace(3, Some(original));
    index.replace(4, Some(unrelated));
    let capacity = index.entries.capacity();
    index.replace(3, Some(original));
    assert_matching_nodes(&index, original.capture_region.unwrap(), &[3]);
    assert_matching_nodes(&index, original.target_region.unwrap(), &[3]);

    let replacement = entry(
        3,
        rect((600, 200), (620, 220)),
        rect((700, 200), (720, 220)),
    );
    index.replace(3, Some(replacement));
    assert_matching_nodes(&index, original.capture_region.unwrap(), &[]);
    assert_matching_nodes(&index, original.target_region.unwrap(), &[]);
    index.remove(3);
    assert_matching_nodes(&index, replacement.capture_region.unwrap(), &[]);
    assert_matching_nodes(&index, unrelated.target_region.unwrap(), &[4]);

    index.replace(3, Some(original));
    index.rebuild([replacement]);
    assert_matching_nodes(&index, original.capture_region.unwrap(), &[]);
    assert_matching_nodes(&index, unrelated.capture_region.unwrap(), &[]);
    assert_matching_nodes(&index, replacement.capture_region.unwrap(), &[3]);
    index.replace(3, None);
    assert!(index.is_empty());
    assert_matching_nodes(&index, replacement.target_region.unwrap(), &[]);

    index.replace(3, Some(original));
    index.clear();
    assert!(index.is_empty());
    assert_matching_nodes(&index, original.capture_region.unwrap(), &[]);
    assert_eq!(index.entries.capacity(), capacity);
}

#[test]
fn capture_and_target_regions_keep_independent_viewport_clipping_and_fringe_coverage() {
    let viewport = Viewport {
        physical_size: (1024, 512),
        scale_factor: 2.0,
    };
    let visible = shape_node([(-4.0, 8.0), (12.0, 24.0)]);
    let offscreen = shape_node([(600.0, 8.0), (616.0, 24.0)]);
    let visible_capture = capture([(256.0, 8.0), (272.0, 24.0)], viewport);
    let offscreen_capture = capture([(600.0, 8.0), (616.0, 24.0)], viewport);
    let target_region = rect((0, 15), (25, 49));
    let capture_region = rect((512, 16), (544, 48));

    let both = BackdropDamageEntry::new(1, &visible, visible_capture, viewport, 1.0).unwrap();
    assert_eq!(both.capture_region, Some(capture_region));
    assert_eq!(both.target_region, Some(target_region));
    let target_only =
        BackdropDamageEntry::new(2, &visible, offscreen_capture, viewport, 1.0).unwrap();
    assert_eq!(target_only.capture_region, None);
    assert_eq!(target_only.target_region, Some(target_region));
    let capture_only =
        BackdropDamageEntry::new(3, &offscreen, visible_capture, viewport, 1.0).unwrap();
    assert_eq!(capture_only.capture_region, Some(capture_region));
    assert_eq!(capture_only.target_region, None);
    assert!(BackdropDamageEntry::new(4, &offscreen, offscreen_capture, viewport, 1.0).is_none());
    assert!(BackdropDamageEntry::new(5, &visible, None, viewport, 1.0).is_none());

    let mut index = BackdropDamageIndex::default();
    index.rebuild([both, target_only, capture_only]);
    assert_matching_nodes(&index, capture_region, &[1, 3]);
    assert_matching_nodes(&index, target_region, &[1, 2]);
    index.remove(3);
    assert_matching_nodes(&index, capture_region, &[1]);
}
