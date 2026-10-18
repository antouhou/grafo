use super::{extend_disjoint_bounds, LeafBatch};
use crate::commands::{
    DrawClip, ShapeDraw, ShapeDrawId, ShapeDrawMaterial, ShapeTextureBinding, ShapeTextureLayer,
    TextureSampling,
};
use crate::core::UnsignedPhysicalRect;
use crate::wgpu_backend::execution::shapes::ShapeBufferLocation;
use crate::wgpu_backend::vertex::GeometryBufferRange;

fn draw() -> ShapeDraw {
    ShapeDraw {
        id: ShapeDrawId(1),
        material: ShapeDrawMaterial {
            has_gradient_fill: false,
            texture_bindings: [ShapeTextureBinding::Managed(1), ShapeTextureBinding::None],
            under_fill_texture: None,
        },
    }
}

fn location(instance_index: usize) -> ShapeBufferLocation {
    ShapeBufferLocation {
        geometry_range: GeometryBufferRange {
            index_start: 3,
            index_count: 6,
            vertex_start: 4,
            vertex_count: 4,
        },
        instance_index,
    }
}

fn clip() -> DrawClip {
    DrawClip {
        scissor: UnsignedPhysicalRect::new((2, 3).into(), (20, 30).into()),
        stencil_reference: 2,
    }
}

fn batch(instance_index: usize) -> LeafBatch {
    LeafBatch::new(draw(), clip(), location(instance_index)).unwrap()
}

#[test]
fn consecutive_instances_extend_the_pending_batch_in_order() {
    let mut pending = batch(7);
    for instance_index in 8..12 {
        assert!(pending.try_append(&batch(instance_index)));
    }
    assert_eq!(pending.first_instance_index, 7);
    assert_eq!(pending.instance_count, 5);
    assert_eq!(pending.geometry_range, location(7).geometry_range);
    assert_eq!(pending.texture_bindings, draw().material.texture_bindings);
    assert_eq!(pending.clip, clip());
}

#[test]
fn incompatible_draws_leave_the_pending_batch_unchanged() {
    let mut different_geometry = batch(8);
    different_geometry.geometry_range.index_start += 6;
    let mut different_background = batch(8);
    different_background.texture_bindings[0] = ShapeTextureBinding::Managed(2);
    let mut different_foreground = batch(8);
    different_foreground.texture_bindings[1] = ShapeTextureBinding::Managed(3);
    let mut different_scissor = batch(8);
    different_scissor.clip.scissor = UnsignedPhysicalRect::new((4, 5).into(), (20, 30).into());
    let mut different_stencil = batch(8);
    different_stencil.clip.stencil_reference += 1;

    for next in [
        different_geometry,
        different_background,
        different_foreground,
        different_scissor,
        different_stencil,
        batch(7),
        batch(9),
    ] {
        let mut pending = batch(7);
        assert!(!pending.try_append(&next));
        assert_eq!(pending.first_instance_index, 7);
        assert_eq!(pending.instance_count, 1);
        assert!(pending.try_append(&batch(8)));
    }
}

#[test]
fn gradient_and_under_fill_materials_require_individual_draws() {
    let mut gradient = draw();
    gradient.material.has_gradient_fill = true;
    let mut under_fill = draw();
    under_fill.material.under_fill_texture = Some(ShapeTextureLayer {
        texture: ShapeTextureBinding::Managed(4),
        sampling: TextureSampling::ShapeUv,
    });
    for draw in [gradient, under_fill] {
        assert!(LeafBatch::new(draw, clip(), location(7)).is_none());
    }
}

#[test]
fn coverage_batches_reject_overlap_with_any_prior_instance() {
    let mut bounds = UnsignedPhysicalRect::new((4, 4).into(), (20, 20).into());
    assert!(extend_disjoint_bounds(
        &mut bounds,
        UnsignedPhysicalRect::new((24, 4).into(), (40, 20).into())
    ));
    let previous = bounds;
    assert!(!extend_disjoint_bounds(
        &mut bounds,
        UnsignedPhysicalRect::new((12, 4).into(), (28, 20).into())
    ));
    assert_eq!(bounds, previous);
    assert!(extend_disjoint_bounds(
        &mut bounds,
        UnsignedPhysicalRect::new((4, 24).into(), (20, 40).into())
    ));
}
