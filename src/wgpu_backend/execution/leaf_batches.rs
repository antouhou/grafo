use super::draws::{self, DrawPass};
use super::shapes::coverage::CoverageAllocation;
use super::shapes::{ShapeBufferLocation, ShapeExecutionResources};
use crate::commands::{DrawClip, ShapeDraw, ShapeDrawMaterial, ShapeTextureBinding};
use crate::core::UnsignedPhysicalRect;
use crate::wgpu_backend::types::Pipeline;
use crate::wgpu_backend::vertex::GeometryBufferRange;
use std::sync::Arc;

struct CoverageBatch {
    allocation: Arc<CoverageAllocation>,
    bounds: UnsignedPhysicalRect,
}

fn extend_disjoint_bounds(bounds: &mut UnsignedPhysicalRect, next: UnsignedPhysicalRect) -> bool {
    // Interior and boundary draws may only be regrouped when their pixels do not overlap.
    if bounds.intersects(&next) {
        return false;
    }
    *bounds = bounds.union(&next);
    true
}

pub(super) struct LeafBatch {
    coverage: Option<CoverageBatch>,
    clip: DrawClip,
    geometry_range: GeometryBufferRange,
    texture_bindings: [ShapeTextureBinding; 2],
    first_instance_index: u32,
    instance_count: u32,
}

impl LeafBatch {
    fn new(draw: ShapeDraw, clip: DrawClip, location: ShapeBufferLocation) -> Option<Self> {
        if draw.material.has_gradient_fill || draw.material.under_fill_texture.is_some() {
            return None;
        }
        Some(Self {
            coverage: None,
            clip,
            geometry_range: location.geometry_range,
            texture_bindings: draw.material.texture_bindings,
            first_instance_index: location.instance_index as u32,
            instance_count: 1,
        })
    }

    fn try_append(&mut self, next: &Self) -> bool {
        if next.clip != self.clip
            || next.texture_bindings != self.texture_bindings
            || next.geometry_range != self.geometry_range
            || next.first_instance_index != self.first_instance_index + self.instance_count
        {
            return false;
        }
        match (&mut self.coverage, &next.coverage) {
            (Some(current), Some(next)) => {
                if !Arc::ptr_eq(&current.allocation, &next.allocation)
                    || !extend_disjoint_bounds(&mut current.bounds, next.bounds)
                {
                    return false;
                }
            }
            (None, None) => {}
            _ => return false,
        }
        self.instance_count += next.instance_count;
        true
    }
}

impl DrawPass<'_, '_> {
    fn draw_batch(&mut self, batch: &LeafBatch) {
        self.set_scissor(batch.clip.scissor);
        let pipelines = &self.pipelines.shapes;
        if batch.coverage.is_some() || self.pipeline_tracker.current != Pipeline::LeafDraw {
            if batch.coverage.is_none() {
                self.render_pass.set_pipeline(&pipelines.leaf_draw_pipeline);
                self.render_pass
                    .set_bind_group(0, &pipelines.and_bind_group, &[]);
                self.pipeline_tracker.switch_to(Pipeline::LeafDraw);
            }
            self.render_pass.set_bind_group(
                1,
                &*pipelines.default_shape_texture_bind_groups[0],
                &[],
            );
            self.render_pass.set_bind_group(
                2,
                &*pipelines.default_shape_texture_bind_groups[1],
                &[],
            );
            self.bound_textures.mark_bound(0, ShapeTextureBinding::None);
            self.bound_textures.mark_bound(1, ShapeTextureBinding::None);
        }
        if let Some(coverage) = &batch.coverage {
            let buffers = coverage.allocation.buffers();
            buffers.bind(self.render_pass);
            self.render_pass.set_bind_group(0, &buffers.bind_group, &[]);
        } else {
            draws::bind_aggregated_geometry_buffers(self.render_pass, self.buffers);
        }
        self.textures.bind_shape_texture_layers(
            self.render_pass,
            &batch.texture_bindings,
            &pipelines.texture_manager,
            &pipelines.shape_texture_bind_group_layout_background,
            &pipelines.shape_texture_bind_group_layout_foreground,
            &pipelines.default_shape_texture_bind_groups,
            self.bound_textures,
        );
        self.render_pass
            .set_vertex_buffer(1, self.buffers.instance_transform_buffer().slice(..));
        self.render_pass
            .set_vertex_buffer(2, self.buffers.instance_color_buffer().slice(..));
        self.render_pass
            .set_vertex_buffer(3, self.buffers.instance_metadata_buffer().slice(..));
        self.render_pass
            .set_stencil_reference(batch.clip.stencil_reference);
        let instances =
            batch.first_instance_index..batch.first_instance_index + batch.instance_count;
        if let Some(coverage) = &batch.coverage {
            let material = ShapeDrawMaterial {
                has_gradient_fill: false,
                under_fill_texture: None,
                texture_bindings: batch.texture_bindings,
            };
            let _draw_count = coverage.allocation.draw(
                self.render_pass,
                [
                    pipelines.coverage.solid.material(material, false),
                    pipelines.coverage.boundary.material(material, false),
                ],
                instances,
            );
            self.pipeline_tracker.current = Pipeline::None;
            #[cfg(feature = "render_metrics")]
            self.pipeline_tracker
                .record_coverage_draws(Pipeline::LeafDraw, _draw_count, false);
            self.bound_textures.invalidate();
        } else {
            self.buffers
                .draw_indexed(self.render_pass, batch.geometry_range, instances);
        }
    }

    pub(super) fn flush_leaf_batch(&mut self, pending_batch: &mut Option<LeafBatch>) {
        if let Some(batch) = pending_batch.take() {
            self.draw_batch(&batch);
        }
    }

    /// Defers compatible draws until a different draw or pass boundary flushes the batch.
    pub(super) fn queue_leaf_draw(
        &mut self,
        draw: ShapeDraw,
        clip: DrawClip,
        shapes: &ShapeExecutionResources,
        pending_batch: &mut Option<LeafBatch>,
    ) {
        let resources = shapes.draw_resources(draw.id);
        let Some(location) = resources.geometry_buffer_location else {
            self.flush_leaf_batch(pending_batch);
            return;
        };
        let Some(mut batch) = LeafBatch::new(draw, clip, location) else {
            self.flush_leaf_batch(pending_batch);
            self.set_scissor(clip.scissor);
            self.draw_shape(clip.stencil_reference, draw.material, resources);
            return;
        };
        if let Some(coverage) = &resources.path_coverage {
            let bounds = coverage.allocation.bounds;
            let [x, y] = coverage.translation.map(|value| value as u32);
            batch.coverage = Some(CoverageBatch {
                allocation: Arc::clone(&coverage.allocation),
                bounds: UnsignedPhysicalRect::new(
                    (bounds.min.x + x, bounds.min.y + y).into(),
                    (bounds.max.x + x, bounds.max.y + y).into(),
                ),
            });
        }
        if pending_batch
            .as_mut()
            .is_some_and(|pending| pending.try_append(&batch))
        {
            return;
        }
        self.flush_leaf_batch(pending_batch);
        *pending_batch = Some(batch);
    }
}

#[cfg(test)]
mod tests;
