use super::draws::{self, DrawPass};
use super::shapes::coverage::CoverageBatch;
use super::shapes::{
    ShapeBufferLocation, ShapeExecutionResources, ShapeGeometryType, ShapeShaderType, StencilAction,
};
use crate::commands::{DrawClip, ShapeDraw, ShapeDrawMaterial, ShapeTextureBinding};
use crate::core::UnsignedPhysicalRect;
use crate::wgpu_backend::vertex::GeometryBufferRange;
use wgpu::Device;

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
        if next.clip != self.clip || next.texture_bindings != self.texture_bindings {
            return false;
        }
        match (&mut self.coverage, &next.coverage) {
            (Some(current), Some(next)) => {
                if !current.try_append(next) {
                    return false;
                }
            }
            (None, None) => {
                if next.geometry_range != self.geometry_range
                    || next.first_instance_index != self.first_instance_index + self.instance_count
                {
                    return false;
                }
            }
            _ => return false,
        }
        self.instance_count += next.instance_count;
        true
    }
}

impl DrawPass<'_, '_> {
    fn draw_batch(
        &mut self,
        batch: &LeafBatch,
        shapes: &mut ShapeExecutionResources,
        device: &Device,
    ) {
        self.set_scissor(batch.clip.scissor);
        let pipelines = &self.pipelines.shapes;
        let material = ShapeDrawMaterial {
            has_gradient_fill: false,
            under_fill_texture: None,
            texture_bindings: batch.texture_bindings,
        };
        let shader = ShapeShaderType::from(material);
        self.bind_material_textures(material);
        self.render_pass
            .set_stencil_reference(batch.clip.stencil_reference);
        if let Some(coverage) = batch
            .coverage
            .as_ref()
            .filter(|coverage| !coverage.can_draw_instanced())
        {
            shapes.coverage_batches.draw(
                self.render_pass,
                coverage,
                (&shapes.instance_colors, &shapes.instance_metadata),
                device,
                pipelines,
                shader,
            );
            return;
        }
        if let Some(coverage) = &batch.coverage {
            let buffers = coverage.allocation.buffers();
            buffers.bind(self.render_pass);
            self.render_pass.set_bind_group(0, &buffers.bind_group, &[]);
        } else {
            self.render_pass.set_pipeline(pipelines.get(
                ShapeGeometryType::Tessellated,
                shader,
                StencilAction::Keep,
            ));
            self.render_pass
                .set_bind_group(0, &pipelines.and_bind_group, &[]);
            draws::bind_aggregated_geometry_buffers(self.render_pass, self.buffers);
        }
        if batch.coverage.is_none() {
            self.render_pass
                .set_vertex_buffer(1, self.buffers.instance_transform_buffer().slice(..));
        }
        let color_buffer_slot = if batch.coverage.is_some() { 0 } else { 2 };
        self.render_pass.set_vertex_buffer(
            color_buffer_slot,
            self.buffers.instance_color_buffer().slice(..),
        );
        if batch.coverage.is_some() || shader.uses_texture_bindings() {
            let metadata_buffer_slot = if batch.coverage.is_some() { 1 } else { 3 };
            self.render_pass.set_vertex_buffer(
                metadata_buffer_slot,
                self.buffers.instance_metadata_buffer().slice(..),
            );
        }
        let instances =
            batch.first_instance_index..batch.first_instance_index + batch.instance_count;
        if let Some(coverage) = &batch.coverage {
            coverage.allocation.draw(
                self.render_pass,
                |boundary| {
                    pipelines.get(
                        ShapeGeometryType::coverage(boundary),
                        shader,
                        StencilAction::Keep,
                    )
                },
                instances,
            );
        } else {
            self.buffers
                .draw_indexed(self.render_pass, batch.geometry_range, instances);
        }
    }

    pub(super) fn flush_leaf_batch(
        &mut self,
        pending_batch: &mut Option<LeafBatch>,
        shapes: &mut ShapeExecutionResources,
        device: &Device,
    ) {
        if let Some(batch) = pending_batch.take() {
            self.draw_batch(&batch, shapes, device);
            if let Some(coverage) = batch.coverage {
                shapes.coverage_batches.recycle_draw_storage(coverage);
            }
        }
    }

    /// Defers compatible draws until a different draw or pass boundary flushes the batch.
    pub(super) fn queue_leaf_draw(
        &mut self,
        draw: ShapeDraw,
        clip: DrawClip,
        shapes: &mut ShapeExecutionResources,
        device: &Device,
        pending_batch: &mut Option<LeafBatch>,
    ) {
        let resources = shapes.draw_resources(draw.id);
        let Some(location) = resources.geometry_buffer_location else {
            self.flush_leaf_batch(pending_batch, shapes, device);
            return;
        };
        let Some(mut batch) = LeafBatch::new(draw, clip, location) else {
            self.flush_leaf_batch(pending_batch, shapes, device);
            self.set_scissor(clip.scissor);
            self.draw_shape(
                clip.stencil_reference,
                draw.material,
                shapes.draw_resources(draw.id),
            );
            return;
        };
        if let Some(coverage) = &resources.path_coverage {
            let bounds = coverage.allocation.bounds;
            let [x, y] = coverage.translation.map(|value| value as u32);
            batch.coverage = Some(CoverageBatch::new(
                &coverage.allocation,
                UnsignedPhysicalRect::new(
                    (bounds.min.x + x, bounds.min.y + y).into(),
                    (bounds.max.x + x, bounds.max.y + y).into(),
                ),
                location.instance_index as u32,
            ));
        }
        if pending_batch
            .as_mut()
            .is_some_and(|pending| pending.try_append(&batch))
        {
            return;
        }
        self.flush_leaf_batch(pending_batch, shapes, device);
        if let Some(coverage) = &mut batch.coverage {
            shapes.coverage_batches.reuse_draw_storage(coverage);
        }
        *pending_batch = Some(batch);
    }
}

#[cfg(test)]
mod tests;
