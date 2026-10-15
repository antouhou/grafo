use super::{envelope, EffectDamage, EffectDamageEntry, EffectDamageKind};
use crate::core::effect::backdrops;
use crate::core::effect::GroupEffectBounds;
use crate::core::{
    BackdropCaptureArea, BackdropEffectConfig, CaptureRegion, GroupEffectConfig, MathRect, Shape,
    ShapeDrawCommandOptions, Size, UnsignedPhysicalRect, Viewport,
};
use crate::scene::types::{CachedShapeDrawData, DrawTreeNode};
use crate::scene::{Scene, SceneContext};

fn rect(min: (u32, u32), max: (u32, u32)) -> UnsignedPhysicalRect {
    UnsignedPhysicalRect::new(min.into(), max.into())
}

fn entry(
    node_id: usize,
    capture_region: UnsignedPhysicalRect,
    target_region: UnsignedPhysicalRect,
) -> EffectDamageEntry {
    EffectDamageEntry {
        node_id,
        kind: EffectDamageKind::Backdrop,
        capture_region: Some(capture_region),
        target_region: Some(target_region),
    }
}

fn assert_matching_nodes(damage: &EffectDamage, query: UnsignedPhysicalRect, expected: &[usize]) {
    let mut nodes = [0; 4];
    let mut count = 0;
    for entry in damage.tree.locate_in_envelope_intersecting(envelope(query)) {
        nodes[count] = entry.node_id;
        count += 1;
    }
    let nodes = &mut nodes[..count];
    nodes.sort_unstable();
    assert_eq!(nodes, expected);
}

fn shape_node(bounds: [(f32, f32); 2]) -> DrawTreeNode {
    let shape = Scene::new(
        SceneContext::default(),
        Viewport {
            physical_size: (32, 32),
            scale_factor: 1.0,
        },
        0.75,
    )
    .tessellate(&Shape::rect(bounds), None);
    DrawTreeNode::CachedShape(CachedShapeDrawData::new(
        shape,
        ShapeDrawCommandOptions::new(),
    ))
}

fn capture(bounds: [(f32, f32); 2], viewport: Viewport) -> Option<CaptureRegion> {
    backdrops::compute_backdrop_capture_region(
        MathRect::zero(),
        BackdropEffectConfig::new().capture_area(BackdropCaptureArea::ScreenRect(bounds)),
        viewport.scale_factor,
        viewport.physical_size.into(),
        1024,
    )
}

fn assert_disjoint_regions(regions: &[UnsignedPhysicalRect]) {
    for (index, region) in regions.iter().enumerate() {
        assert!(!region.is_empty());
        for previous in &regions[..index] {
            assert!(!region.intersects(previous));
        }
    }
}

fn storage_capacities(damage: &EffectDamage) -> [usize; 3] {
    [
        damage.entries.capacity(),
        damage.query_regions.capacity(),
        damage.processed_effects.capacity(),
    ]
}

#[test]
fn replacement_removal_and_rebuild_keep_spatial_queries_and_storage_consistent() {
    let mut damage = EffectDamage::default();
    let original = entry(3, rect((10, 10), (20, 20)), rect((500, 10), (520, 30)));
    let unrelated = entry(
        4,
        rect((800, 300), (820, 320)),
        rect((900, 300), (920, 320)),
    );
    damage.replace(EffectDamageKind::Backdrop, 3, Some(original));
    damage.replace(EffectDamageKind::Backdrop, 4, Some(unrelated));
    let capacity = damage.entries.capacity();
    damage.replace(EffectDamageKind::Backdrop, 3, Some(original));
    assert_matching_nodes(&damage, original.capture_region.unwrap(), &[3]);
    assert_matching_nodes(&damage, original.target_region.unwrap(), &[3]);

    let replacement = entry(
        3,
        rect((600, 200), (620, 220)),
        rect((700, 200), (720, 220)),
    );
    damage.replace(EffectDamageKind::Backdrop, 3, Some(replacement));
    assert_matching_nodes(&damage, original.capture_region.unwrap(), &[]);
    assert_matching_nodes(&damage, original.target_region.unwrap(), &[]);
    damage.remove(EffectDamageKind::Backdrop, 3);
    assert_matching_nodes(&damage, replacement.capture_region.unwrap(), &[]);
    assert_matching_nodes(&damage, unrelated.target_region.unwrap(), &[4]);

    damage.replace(EffectDamageKind::Backdrop, 3, Some(original));
    damage.rebuild([replacement]);
    assert_matching_nodes(&damage, original.capture_region.unwrap(), &[]);
    assert_matching_nodes(&damage, unrelated.capture_region.unwrap(), &[]);
    assert_matching_nodes(&damage, replacement.capture_region.unwrap(), &[3]);
    damage.replace(EffectDamageKind::Backdrop, 3, None);
    assert!(damage.is_empty());
    assert_matching_nodes(&damage, replacement.target_region.unwrap(), &[]);

    damage.replace(EffectDamageKind::Backdrop, 3, Some(original));
    damage.clear();
    assert!(damage.is_empty());
    assert_matching_nodes(&damage, original.capture_region.unwrap(), &[]);
    assert_eq!(damage.entries.capacity(), capacity);
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

    let both = EffectDamageEntry::backdrop(1, &visible, visible_capture, viewport, 1.0).unwrap();
    assert_eq!(both.capture_region, Some(capture_region));
    assert_eq!(both.target_region, Some(target_region));
    let target_only =
        EffectDamageEntry::backdrop(2, &visible, offscreen_capture, viewport, 1.0).unwrap();
    assert_eq!(target_only.capture_region, None);
    assert_eq!(target_only.target_region, Some(target_region));
    let capture_only =
        EffectDamageEntry::backdrop(3, &offscreen, visible_capture, viewport, 1.0).unwrap();
    assert_eq!(capture_only.capture_region, Some(capture_region));
    assert_eq!(capture_only.target_region, None);
    assert!(EffectDamageEntry::backdrop(4, &offscreen, offscreen_capture, viewport, 1.0).is_none());
    assert!(EffectDamageEntry::backdrop(5, &visible, None, viewport, 1.0).is_none());

    let mut damage = EffectDamage::default();
    damage.rebuild([both, target_only, capture_only]);
    assert_matching_nodes(&damage, capture_region, &[1, 3]);
    assert_matching_nodes(&damage, target_region, &[1, 2]);
    for initial in [capture_region, target_region] {
        assert_eq!(
            damage.expand(viewport.physical_size.into(), Some(initial)),
            Some(capture_region.union(&target_region))
        );
        assert_eq!(damage.processed_effects.len(), 3);
    }
    damage.remove(EffectDamageKind::Backdrop, 3);
    assert_matching_nodes(&damage, capture_region, &[1]);

    for (entry, region) in [(target_only, target_region), (capture_only, capture_region)] {
        damage.rebuild([entry]);
        assert_eq!(
            damage.expand(viewport.physical_size.into(), Some(region)),
            Some(region)
        );
        assert_eq!(damage.processed_effects.len(), 1);
    }
}

#[test]
fn damage_in_the_gap_between_capture_and_target_does_not_expand() {
    let mut damage = EffectDamage::default();
    damage.replace(
        EffectDamageKind::Backdrop,
        1,
        Some(entry(
            1,
            rect((0, 0), (16, 128)),
            rect((1008, 0), (1024, 128)),
        )),
    );
    let initial = Some(rect((500, 40), (501, 41)));
    assert_eq!(damage.expand(Size::new(1024, 128), initial), initial);
    assert!(damage.processed_effects.is_empty());
}

#[test]
fn unrelated_damage_in_the_same_cell_does_not_expand() {
    let mut damage = EffectDamage::default();
    damage.replace(
        EffectDamageKind::Backdrop,
        1,
        Some(entry(
            1,
            rect((100, 100), (110, 110)),
            rect((900, 100), (910, 110)),
        )),
    );
    let initial = Some(rect((1, 1), (2, 2)));
    assert_eq!(damage.expand(Size::new(1024, 128), initial), initial);
    assert!(damage.processed_effects.is_empty());
}

#[test]
fn touching_capture_edges_do_not_expand_damage() {
    let mut damage = EffectDamage::default();
    damage.replace(
        EffectDamageKind::Backdrop,
        1,
        Some(entry(1, rect((0, 1), (1, 2)), rect((100, 1), (101, 2)))),
    );
    let initial = Some(rect((1, 1), (2, 2)));
    assert_eq!(damage.expand(Size::new(256, 128), initial), initial);
    assert!(damage.processed_effects.is_empty());
}

#[test]
fn same_cell_growth_reaches_dependencies_for_both_insertion_orders() {
    let trigger = entry(1, rect((1, 1), (2, 2)), rect((50, 1), (51, 2)));
    let dependent = entry(2, rect((50, 1), (51, 2)), rect((100, 1), (101, 2)));
    for entries in [[trigger, dependent], [dependent, trigger]] {
        let mut damage = EffectDamage::default();
        for entry in entries {
            damage.replace(EffectDamageKind::Backdrop, entry.node_id, Some(entry));
        }
        assert_eq!(
            damage.expand(Size::new(256, 128), Some(rect((1, 1), (2, 2)))),
            Some(rect((1, 1), (101, 2)))
        );
        assert_eq!(damage.processed_effects.len(), 2);
    }
}

#[test]
fn growth_reaches_dependencies_inside_the_new_bounding_rectangle_gap() {
    let mut damage = EffectDamage::default();
    damage.rebuild([
        entry(1, rect((10, 10), (11, 11)), rect((50, 50), (51, 51))),
        entry(2, rect((25, 25), (26, 26)), rect((100, 20), (101, 21))),
    ]);
    assert_eq!(
        damage.expand(Size::new(256, 128), Some(rect((10, 10), (11, 11)))),
        Some(rect((10, 10), (101, 51)))
    );
    assert_eq!(damage.processed_effects.len(), 2);
}

#[test]
fn long_chains_query_disjoint_regions_and_reuse_worklist_storage() {
    let mut damage = EffectDamage::default();
    damage.rebuild((0..64).rev().map(|node| {
        let bounds = rect((node * 64, 140), (((node + 1) * 64 + 1).min(4096), 180));
        entry(node as usize, bounds, bounds)
    }));
    let initial = Some(rect((0, 150), (1, 151)));
    let expected = Some(rect((0, 140), (4096, 180)));
    assert_eq!(damage.expand(Size::new(8192, 512), initial), expected);
    assert_eq!(damage.processed_effects.len(), 64);
    assert_disjoint_regions(&damage.query_regions);
    let capacities = storage_capacities(&damage);
    for _ in 0..3 {
        assert_eq!(damage.expand(Size::new(8192, 512), None), None);
        assert_eq!(damage.expand(Size::new(8192, 512), initial), expected);
        assert_eq!(storage_capacities(&damage), capacities);
        assert_eq!(damage.processed_effects.len(), 64);
        assert_disjoint_regions(&damage.query_regions);
    }
}

#[test]
fn expansion_in_all_directions_queries_newly_covered_regions_without_overlap() {
    let mut damage = EffectDamage::default();
    damage.rebuild(
        [
            (1, rect((128, 128), (600, 600))),
            (2, rect((0, 350), (160, 400))),
            (3, rect((200, 0), (260, 160))),
            (4, rect((599, 100), (900, 129))),
        ]
        .map(|(node_id, bounds)| entry(node_id, bounds, bounds)),
    );
    assert_eq!(
        damage.expand(Size::new(1024, 1024), Some(rect((150, 150), (151, 151)))),
        Some(rect((0, 0), (900, 600)))
    );
    assert_eq!(damage.processed_effects.len(), 4);
    assert_disjoint_regions(&damage.query_regions);
}

#[test]
fn expansion_propagates_through_separate_capture_and_target_regions() {
    let mut damage = EffectDamage::default();
    damage.rebuild([
        entry(1, rect((0, 0), (64, 64)), rect((256, 8), (288, 32))),
        entry(2, rect((256, 0), (320, 64)), rect((512, 8), (544, 32))),
    ]);
    assert_eq!(
        damage.expand(Size::new(1024, 512), Some(rect((8, 8), (16, 16)))),
        Some(rect((0, 0), (544, 64)))
    );
    assert_eq!(damage.processed_effects.len(), 2);
    assert_eq!(damage.expand(Size::new(1024, 512), None), None);
}

#[test]
fn dependency_changes_update_expansion_and_preserve_reusable_storage() {
    let mut damage = EffectDamage::default();
    let size = Size::new(1024, 512);
    let initial = Some(rect((8, 8), (16, 16)));
    let capture = rect((0, 0), (64, 64));
    let original = entry(1, capture, rect((768, 8), (800, 32)));
    damage.replace(EffectDamageKind::Backdrop, 1, Some(original));
    assert_eq!(damage.expand(size, initial), Some(rect((0, 0), (800, 64))));
    let capacities = storage_capacities(&damage);

    let replacement = entry(1, capture, rect((256, 8), (288, 32)));
    damage.replace(EffectDamageKind::Backdrop, 1, Some(replacement));
    assert_eq!(damage.expand(size, initial), Some(rect((0, 0), (288, 64))));
    damage.remove(EffectDamageKind::Backdrop, 1);
    assert_eq!(damage.expand(size, initial), initial);

    damage.rebuild([replacement]);
    assert_eq!(damage.expand(size, initial), Some(rect((0, 0), (288, 64))));
    damage.clear();
    assert_eq!(damage.expand(size, initial), initial);
    assert_eq!(storage_capacities(&damage), capacities);

    damage.replace(EffectDamageKind::Backdrop, 1, Some(original));
    assert_eq!(damage.expand(size, initial), Some(rect((0, 0), (800, 64))));
    assert_eq!(storage_capacities(&damage), capacities);
    assert_eq!(damage.processed_effects.len(), 1);
}

#[test]
fn group_and_backdrop_entries_on_the_same_node_expand_in_both_insertion_orders() {
    let mut first = entry(1, rect((20, 20), (40, 40)), rect((10, 10), (50, 50)));
    first.kind = EffectDamageKind::Group;
    let backdrop = entry(2, rect((45, 20), (50, 30)), rect((89, 19), (101, 31)));
    let mut second = entry(2, rect((90, 20), (100, 30)), rect((85, 15), (105, 35)));
    second.kind = EffectDamageKind::Group;
    for entries in [[first, backdrop, second], [second, backdrop, first]] {
        let mut damage = EffectDamage::default();
        damage.rebuild(entries);
        assert_eq!(
            damage.expand(Size::new(128, 128), Some(rect((26, 26), (28, 28)))),
            Some(rect((10, 10), (105, 50)))
        );
        assert_eq!(damage.processed_effects.len(), 3);
        assert_disjoint_regions(&damage.query_regions);
        let capacities = storage_capacities(&damage);
        damage.expand(Size::new(128, 128), Some(rect((26, 26), (28, 28))));
        assert_eq!(storage_capacities(&damage), capacities);
        damage.remove(EffectDamageKind::Group, 2);
        assert_eq!(
            damage.expand(Size::new(128, 128), Some(rect((26, 26), (28, 28)))),
            Some(rect((10, 10), (101, 50)))
        );
    }
}

#[test]
fn group_rectangles_clip_full_resolution_input_and_output_independently() {
    let viewport = Viewport {
        physical_size: (64, 32),
        scale_factor: 2.0,
    };
    let mut damage = EffectDamage::default();
    for downsample in [1.0, 0.25] {
        let bounds = GroupEffectBounds::new(
            MathRect::new((-2.0, 4.0).into(), (10.0, 14.0).into()),
            None,
            GroupEffectConfig::new().outset(3.0).downsample(downsample),
            viewport,
            4096,
        );
        let group = EffectDamageEntry::group(1, bounds, viewport.physical_size.into()).unwrap();
        assert_eq!(group.capture_region, Some(rect((0, 8), (20, 28))));
        assert_eq!(group.target_region, Some(rect((0, 2), (26, 32))));
        damage.replace(EffectDamageKind::Group, 1, Some(group));
        assert_eq!(
            damage.expand(viewport.physical_size.into(), Some(rect((1, 10), (2, 11)))),
            group.target_region
        );
    }
    let bounds = GroupEffectBounds::new(
        MathRect::new((-12.0, 4.0).into(), (-8.0, 14.0).into()),
        None,
        GroupEffectConfig::new().outset(10.0),
        viewport,
        4096,
    );
    let group = EffectDamageEntry::group(1, bounds, viewport.physical_size.into()).unwrap();
    assert_eq!(group.capture_region, None);
    assert_eq!(group.target_region, Some(rect((0, 0), (4, 32))));
    damage.replace(EffectDamageKind::Group, 1, Some(group));
    let initial = Some(rect((1, 10), (2, 11)));
    assert_eq!(
        damage.expand(viewport.physical_size.into(), initial),
        initial
    );
    assert!(EffectDamageEntry::group(1, None, viewport.physical_size.into()).is_none());
}
