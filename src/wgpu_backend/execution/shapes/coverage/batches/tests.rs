use super::CoverageBatch;
use crate::core::UnsignedPhysicalRect;
use crate::wgpu_backend::execution::shapes::coverage::tiles::CoverageDrawRange;
use crate::wgpu_backend::execution::shapes::coverage::{CoverageAllocation, CoverageBuffers};
use std::sync::{Arc, OnceLock};

fn allocation(
    buffers: &Arc<OnceLock<CoverageBuffers>>,
    tile_start: u32,
) -> Arc<CoverageAllocation> {
    Arc::new(CoverageAllocation {
        buffers: Arc::clone(buffers),
        range: CoverageDrawRange {
            tile_start,
            solid_tile_count: 2,
            tile_count: 7,
        },
        bounds: UnsignedPhysicalRect::from_size((16, 16).into()),
    })
}

fn batch(allocation: &Arc<CoverageAllocation>, x: u32, y: u32, instance: u32) -> CoverageBatch {
    CoverageBatch::new(
        allocation,
        UnsignedPhysicalRect::new((x, y).into(), (x + 16, y + 16).into()),
        instance,
    )
}

#[test]
fn distinct_paths_use_conservative_bounds_and_preserve_instance_locations() {
    let buffers = Arc::new(OnceLock::new());
    let first = allocation(&buffers, 0);
    let second = allocation(&buffers, 7);
    let mut pending = batch(&first, 4, 4, 2);
    assert!(pending.try_append(&batch(&second, 24, 4, 8)));
    assert!(!pending.uses_contiguous_shape_instances());
    assert!(pending.try_append(&batch(&first, 4, 24, 3)));

    // This shape misses all three shapes, but intersects their accumulated rectangle.
    assert!(!pending.try_append(&batch(&second, 24, 24, 9)));
    assert!(!pending.try_append(&batch(&second, 8, 4, 9)));
    let separate = allocation(&Arc::new(OnceLock::new()), 14);
    assert!(!pending.try_append(&batch(&separate, 44, 24, 9)));
    assert!(pending.try_append(&batch(&second, 44, 24, 9)));
    assert_eq!(
        pending
            .draws
            .iter()
            .map(|draw| (draw.instance_index, draw.allocation.range.tile_start))
            .collect::<Vec<_>>(),
        [(2, 0), (8, 7), (3, 0), (9, 7)],
    );
    assert_eq!(pending.tile_count, 28);
}

#[test]
fn repeated_outline_keeps_instancing_without_building_a_draw_list() {
    let buffers = Arc::new(OnceLock::new());
    let first = allocation(&buffers, 0);
    let second = allocation(&buffers, 7);
    let mut pending = batch(&first, 4, 4, 2);
    assert!(pending.try_append(&batch(&first, 24, 4, 3)));
    assert!(pending.try_append(&batch(&first, 44, 4, 4)));
    assert!(pending.uses_contiguous_shape_instances());
    assert_eq!(pending.instance_count, 3);
    assert!(pending.draws.is_empty());

    // End an instanced run instead of expanding its earlier draws.
    assert!(!pending.try_append(&batch(&second, 64, 4, 5)));
    assert!(!pending.try_append(&batch(&first, 64, 4, 8)));
    assert!(pending.try_append(&batch(&first, 64, 4, 5)));
    assert_eq!(pending.instance_count, 4);
}
