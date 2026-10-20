use super::{CoverageUploads, SHARED_BUFFER_BYTES};
use crate::core::{BorderRadii, Shape, TransformInstance, Viewport};
use crate::scene::{Scene, SceneContext};
use crate::wgpu_backend::execution::shapes::coverage::{CoverageBuilder, CoverageGeometry};
use std::sync::Arc;

fn striped_path_with_solid_span(stripe_count: u32, height: f32) -> Shape {
    let mut shape = Shape::builder();
    for stripe in 0..stripe_count {
        let left = stripe as f32 * 8.0 + 0.25;
        shape = shape
            .begin((left, 0.25))
            .line_to((left + 0.5, 0.25))
            .line_to((left + 0.5, height))
            .line_to((left, height))
            .close();
    }
    let left = stripe_count as f32 * 8.0 + 8.25;
    shape
        .begin((left, 0.25))
        .line_to((left + 24.0, 0.25))
        .line_to((left + 24.0, height))
        .line_to((left, height))
        .close()
        .build()
}

#[test]
fn retained_small_path_does_not_keep_oversized_upload_alive() {
    let viewport = Viewport {
        physical_size: (1024, 2048),
        scale_factor: 1.0,
    };
    let mut scene = Scene::new(SceneContext::default(), viewport, 0.75);
    let small = scene.tessellate(
        &Shape::rounded_rect([(0.375, 0.625), (12.875, 12.25)], BorderRadii::new(3.0)),
        None,
    );
    let large = scene.tessellate(&striped_path_with_solid_span(64, 2047.75), None);
    let mut builder = CoverageBuilder::default();
    for transform in [None, Some(TransformInstance::translation(4096.0, 0.0))] {
        let mut uploads = CoverageUploads::default();
        let retained = uploads.prepare(&mut builder, &small.tessellation, transform, viewport);
        let temporary = uploads.prepare(&mut builder, &large.tessellation, None, viewport);
        assert!(!Arc::ptr_eq(&retained.buffers, &temporary.buffers));
        assert!(uploads.pending[0].geometry.byte_len() < SHARED_BUFFER_BYTES);
        assert!(uploads.pending[1].geometry.byte_len() > 4 * 1024 * 1024);
        let following = uploads.prepare(&mut builder, &small.tessellation, None, viewport);
        assert!(!Arc::ptr_eq(&following.buffers, &temporary.buffers));
        let temporary_buffers = Arc::downgrade(&temporary.buffers);
        uploads.clear();
        drop(temporary);
        assert!(temporary_buffers.upgrade().is_none());
        assert_eq!(Arc::strong_count(&retained.buffers), 1);
    }
}

#[test]
fn incoming_path_size_limits_groups_and_preserves_coverage_offsets() {
    let viewport = Viewport {
        physical_size: (512, 128),
        scale_factor: 1.0,
    };
    let mut scene = Scene::new(SceneContext::default(), viewport, 0.75);
    let small = scene.tessellate(
        &Shape::rounded_rect([(0.375, 0.625), (12.875, 12.25)], BorderRadii::new(3.0)),
        None,
    );
    let medium = scene.tessellate(&striped_path_with_solid_span(32, 127.75), None);
    let mut builder = CoverageBuilder::default();
    let mut expected = CoverageGeometry::default();
    let medium_coverage_draw = builder.build(
        &medium.tessellation,
        None,
        1.0,
        viewport.physical_size,
        &mut expected,
    );
    assert!(expected.byte_len() < SHARED_BUFFER_BYTES);
    assert!(expected.byte_len() * 2 > SHARED_BUFFER_BYTES);
    assert!(medium_coverage_draw.range.solid_tile_count > 0);
    let mut uploads = CoverageUploads::default();
    let first_small = uploads.prepare(&mut builder, &small.tessellation, None, viewport);
    let first_medium = uploads.prepare(&mut builder, &medium.tessellation, None, viewport);
    let second_medium = uploads.prepare(&mut builder, &medium.tessellation, None, viewport);
    let second_small = uploads.prepare(&mut builder, &small.tessellation, None, viewport);
    assert!(Arc::ptr_eq(&first_small.buffers, &first_medium.buffers));
    assert!(!Arc::ptr_eq(&first_medium.buffers, &second_medium.buffers));
    assert!(Arc::ptr_eq(&second_medium.buffers, &second_small.buffers));
    assert!(uploads
        .pending
        .iter()
        .all(|pending| pending.geometry.byte_len() <= SHARED_BUFFER_BYTES));
    assert_eq!(second_medium.range.tile_start, 0);
    assert_eq!(
        second_medium.range.tile_count,
        medium_coverage_draw.range.tile_count
    );
    assert_eq!(
        second_medium.range.solid_tile_count,
        medium_coverage_draw.range.solid_tile_count
    );
    let actual = &uploads.pending[1].geometry;
    assert_eq!(
        &actual.coverage[..expected.coverage.len()],
        expected.coverage
    );
    assert_eq!(
        bytemuck::cast_slice::<_, u8>(&actual.projections[..1]),
        bytemuck::cast_slice::<_, u8>(&expected.projections),
    );
    for (actual_tile, expected_tile) in actual.tiles.iter().zip(&expected.tiles) {
        assert_eq!(actual_tile.min, expected_tile.min);
        assert_eq!(actual_tile.max, expected_tile.max);
        assert_eq!(actual_tile.projection_index, expected_tile.projection_index);
        assert_eq!(actual_tile.coverage_start, expected_tile.coverage_start);
    }
}
