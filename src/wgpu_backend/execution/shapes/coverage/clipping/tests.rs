use super::cache::ClipCoverageCache;
use super::ClipCoverage;
use crate::core::shape::{CachedShapeHandle, ShapeResources};
use crate::core::{UnsignedPhysicalRect, Viewport};
use crate::wgpu_backend::execution::shapes::coverage::tiles::CoverageTile;
use crate::wgpu_backend::execution::shapes::coverage::{
    CoverageBuilder, CoverageCache, CoverageGeometry,
};
use crate::{BorderRadii, Shape};
use std::sync::Arc;

fn rectangle(bounds: [f64; 4], viewport: UnsignedPhysicalRect) -> Arc<ClipCoverage> {
    ClipCoverage::rectangle(bounds, viewport, &mut Vec::new())
}

fn intersection(first: &Arc<ClipCoverage>, second: &Arc<ClipCoverage>) -> Arc<ClipCoverage> {
    ClipCoverage::intersection(first, second, &mut Vec::new())
}

fn path_coverage(shape: &Shape) -> Arc<ClipCoverage> {
    let handle = CachedShapeHandle::new(shape, &mut ShapeResources::new(), None).unwrap();
    let mut geometry = CoverageGeometry::default();
    CoverageBuilder::default().build(
        &handle.fill_tessellation,
        None,
        1.0,
        (32, 32),
        &mut geometry,
    );
    ClipCoverage::from_raster(
        &geometry.tiles,
        &geometry.coverage,
        UnsignedPhysicalRect::from_size((32, 32).into()),
        [0.0; 2],
        &mut Vec::new(),
        &mut Vec::new(),
    )
}

#[test]
fn translated_solid_tiles_keep_only_boundary_storage() {
    let tiles: Vec<_> = (5..261)
        .step_by(8)
        .map(|top| CoverageTile {
            min: [3, top],
            max: [259, top + 8],
            coverage_start: u32::MAX,
            projection_index: 0,
        })
        .collect();
    let actual = ClipCoverage::from_raster(
        &tiles,
        &[],
        UnsignedPhysicalRect::new([3, 5].into(), [259, 261].into()),
        [2.0, 2.0],
        &mut Vec::new(),
        &mut Vec::new(),
    );
    for y in 0..270 {
        for x in 0..270 {
            let expected = f32::from((5..261).contains(&x) && (7..263).contains(&y));
            assert_eq!(actual.value(x, y), expected);
        }
    }
    assert!(actual.borders.capacity() < actual.cells.len() / 2);
}

#[test]
fn clips_read_only_their_shape_from_shared_uploads() {
    let first = Shape::builder()
        .begin((0.0, 0.0))
        .line_to((32.0, 32.0))
        .line_to((0.0, 32.0))
        .close()
        .build();
    let second = Shape::builder()
        .begin((0.0, 0.0))
        .line_to((32.0, 0.0))
        .line_to((32.0, 32.0))
        .close()
        .build();
    let mut resources = ShapeResources::new();
    let first = CachedShapeHandle::new(&first, &mut resources, None).unwrap();
    let expected = path_coverage(&second);
    let second = CachedShapeHandle::new(&second, &mut resources, None).unwrap();
    let viewport = Viewport {
        physical_size: (32, 32),
        scale_factor: 1.0,
    };
    let mut paths = CoverageCache::new();
    let first = paths.prepare(&first.fill_tessellation, None, viewport);
    let second = paths.prepare(&second.fill_tessellation, None, viewport);
    assert!(Arc::ptr_eq(
        &first.allocation.storage,
        &second.allocation.storage
    ));
    assert!(second.allocation.range.tile_start > 0);
    paths.seal_pending();
    let actual = ClipCoverageCache::new().shape(&second, viewport);
    for y in 0..32 {
        for x in 0..32 {
            assert_eq!(actual.value(x, y), expected.value(x, y));
        }
    }
}

#[test]
fn coincident_and_opposite_diagonal_edges_preserve_coverage() {
    let shape = Shape::builder()
        .begin((0.0, 0.0))
        .line_to((32.0, 32.0))
        .line_to((0.0, 32.0))
        .close()
        .build();
    let opposite = Shape::builder()
        .begin((0.0, 0.0))
        .line_to((32.0, 0.0))
        .line_to((32.0, 32.0))
        .close()
        .build();
    let first = path_coverage(&shape);
    let matching = intersection(&first, &path_coverage(&shape));
    let disjoint = intersection(&first, &path_coverage(&opposite));
    for pixel in 1..31 {
        assert!((matching.value(pixel, pixel) - 0.5).abs() < 1.0e-6);
        assert!(disjoint.value(pixel, pixel) < 1.0e-6);
    }
}

#[test]
fn matching_subpixel_paths_do_not_fade_under_repeated_clips() {
    let shape = Shape::builder()
        .begin((8.25, 0.0))
        .line_to((8.75, 0.0))
        .line_to((8.75, 32.0))
        .line_to((8.25, 32.0))
        .close()
        .build();
    let mut combined = path_coverage(&shape);
    for _ in 0..8 {
        combined = intersection(&combined, &path_coverage(&shape));
        assert_eq!(combined.value(8, 12), 0.5);
    }
    let mut cache = ClipCoverageCache::new();
    let context =
        cache.for_offscreen_target(&combined, UnsignedPhysicalRect::from_size((32, 32).into()));
    let captured = cache.intersection(&context, &path_coverage(&shape));
    assert_eq!(captured.value(8, 12), 0.5);
}

#[test]
fn live_paths_keep_coverage_and_expired_masks_reuse_their_storage() {
    let shape = CachedShapeHandle::new(
        &Shape::rounded_rect([(0.25, 0.5), (31.75, 31.5)], BorderRadii::new(4.0)),
        &mut ShapeResources::new(),
        None,
    )
    .unwrap();
    let viewport = Viewport {
        physical_size: (32, 32),
        scale_factor: 1.0,
    };
    let mut paths = CoverageCache::new();
    let mut path = paths.prepare(&shape.fill_tessellation, None, viewport);
    paths.seal_pending();
    let mut cache = ClipCoverageCache::new();
    let original = cache.shape(&path, viewport);
    cache.finish_render();
    cache.finish_render();
    assert!(Arc::ptr_eq(&original, &cache.shape(&path, viewport)));
    path.clear_clipping();
    cache.clear();
    assert!(!Arc::ptr_eq(&original, &cache.shape(&path, viewport)));
    let viewport = UnsignedPhysicalRect::from_size((32, 32).into());
    let rectangle = cache.rectangle([0.5, 0.5, 31.5, 31.5], viewport);
    let storage = (rectangle.cells.as_ptr(), rectangle.borders.as_ptr());
    drop(rectangle);
    cache.finish_render();
    cache.finish_render();
    let reused = cache.rectangle([0.75, 0.75, 31.25, 31.25], viewport);
    assert_eq!(storage, (reused.cells.as_ptr(), reused.borders.as_ptr()));
    assert_eq!(reused.value(0, 0), 0.0625);
}

#[test]
fn perpendicular_path_edges_keep_a_quarter_pixel_at_the_crossing() {
    let vertical = path_coverage(
        &Shape::builder()
            .begin((8.5, 0.0))
            .line_to((32.0, 0.0))
            .line_to((32.0, 32.0))
            .line_to((8.5, 32.0))
            .close()
            .build(),
    );
    let horizontal = path_coverage(
        &Shape::builder()
            .begin((0.0, 8.5))
            .line_to((32.0, 8.5))
            .line_to((32.0, 32.0))
            .line_to((0.0, 32.0))
            .close()
            .build(),
    );
    let combined = intersection(&vertical, &horizontal);
    assert_eq!(combined.value(8, 8), 0.25);
    assert_eq!(combined.value(8, 9), 0.5);
    assert_eq!(combined.value(9, 8), 0.5);
    assert_eq!(combined.value(9, 9), 1.0);
}

#[test]
fn rectangle_sources_and_nested_intersections_reuse_cached_storage() {
    let mut cache = ClipCoverageCache::new();
    let viewport = UnsignedPhysicalRect::from_size((32, 32).into());
    let bounds = [0.001, 0.0, 0.006, 1.0];
    let first = cache.rectangle(bounds, viewport);
    let second = cache.rectangle(bounds, viewport);
    assert!(Arc::ptr_eq(&first, &second));
    let combined = cache.intersection(&first, &second);
    assert_eq!(combined.value(0, 0), 0.005);
    let parent = cache.rectangle([0.0, 0.0, 32.0, 32.0], viewport);
    let first_pass = cache.intersection(&parent, &combined);
    cache.finish_render();
    let second_pass = cache.intersection(&parent, &combined);
    assert!(Arc::ptr_eq(&first_pass, &second_pass));
    cache.clear();
    let replaced = cache.rectangle(bounds, UnsignedPhysicalRect::zero());
    assert_eq!(replaced.value(0, 0), 0.0);
}

#[test]
fn offscreen_children_keep_shared_edges_and_content_outside_the_clip() {
    let mut cache = ClipCoverageCache::new();
    let viewport = UnsignedPhysicalRect::from_size((32, 32).into());
    let parent = cache.rectangle([1.25, 1.0, 10.75, 12.0], viewport);
    let context = cache.for_offscreen_target(&parent, viewport);
    assert_eq!(context.value(0, 5), 1.0);
    assert_eq!(context.value(1, 5), 0.75);
    let combined = cache.intersection(&context, &parent);
    let mut geometry = CoverageGeometry::default();
    combined.compile(Some(&context), [0, 0], None, &mut geometry);
    assert!(geometry.coverage.is_empty());
    let outside = cache.rectangle([20.5, 1.0, 24.0, 12.0], viewport);
    assert_eq!(cache.intersection(&context, &outside).value(20, 5), 0.5);
    let mut inherited = Arc::clone(&context);
    for _ in 0..10_000 {
        inherited = cache.for_offscreen_target(&inherited, viewport);
    }
    assert!(Arc::ptr_eq(&context, &inherited));
}

#[test]
fn nested_rectangles_keep_shared_and_disjoint_parts_of_edge_pixels() {
    let viewport = UnsignedPhysicalRect::from_size((32, 32).into());
    let left = rectangle([1.25, 1.0, 10.75, 12.0], viewport);
    let same = intersection(&left, &left);
    assert!(Arc::ptr_eq(&left, &same));
    assert_eq!(same.value(1, 4), 0.75);
    let right = rectangle([1.5, 1.0, 11.0, 12.0], viewport);
    let combined = intersection(&left, &right);
    assert_eq!(combined.value(1, 4), 0.5);
    assert_eq!(combined.value(10, 4), 0.75);
    let separate = rectangle([10.8, 1.0, 12.0, 12.0], viewport);
    let empty = intersection(&combined, &separate);
    assert_eq!(empty.value(10, 4), 0.0);
    let mut geometry = CoverageGeometry::default();
    combined.compile(Some(&left), [0, 0], None, &mut geometry);
    let mut selected = None;
    for tile in &geometry.tiles {
        if tile.min[0] <= 1 && tile.max[0] > 1 && tile.min[1] <= 4 && tile.max[1] > 4 {
            selected = Some(tile);
            break;
        }
    }
    let tile = selected.unwrap();
    let value =
        geometry.coverage[(tile.coverage_start + (4 - tile.min[1]) * 8 + 1 - tile.min[0]) as usize];
    assert!((value - 2.0 / 3.0).abs() < 1.0e-6);
}

#[test]
fn older_ancestor_edges_survive_fully_covering_parents() {
    let viewport = UnsignedPhysicalRect::from_size((32, 32).into());
    let ancestor = rectangle([0.5, 0.5, 31.5, 31.5], viewport);
    let parent = rectangle([0.0, 0.0, 32.0, 32.0], viewport);
    let child = rectangle([0.25, 0.25, 31.75, 31.75], viewport);
    let inherited = intersection(&ancestor, &parent);
    let combined = intersection(&inherited, &child);
    assert_eq!(combined.value(0, 0), 0.25);
    assert_eq!(combined.value(31, 31), 0.25);
    assert_eq!(combined.value(12, 12), 1.0);
}

#[test]
fn contained_intersections_preserve_the_existing_coverage() {
    let viewport = UnsignedPhysicalRect::from_size((128, 128).into());
    let parent = rectangle([0.0, 0.0, 128.0, 128.0], viewport);
    let child = rectangle([12.25, 15.5, 113.75, 110.0], viewport);
    for combined in [intersection(&parent, &child), intersection(&child, &parent)] {
        assert!(Arc::ptr_eq(&combined, &child));
    }
    let overlapping = rectangle([12.5, 15.0, 113.75, 110.0], viewport);
    let combined = intersection(&overlapping, &child);
    assert!(!Arc::ptr_eq(&combined, &child));
    assert_eq!(combined.value(12, 16), 0.5);
}
