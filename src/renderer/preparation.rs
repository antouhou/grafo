use super::execution::shapes::ShapeDrawResources;
use super::types::GeometryBufferError;
use super::Renderer;
use crate::shape::CachedShapeDrawData;

impl Renderer<'_> {
    pub(super) fn append_shape_resources(
        &mut self,
        shape: &mut CachedShapeDrawData,
    ) -> Result<ShapeDrawResources, GeometryBufferError> {
        self.refresh_geometry_cache(shape);
        self.state.shape_execution.prepare_draw(
            shape,
            &self.device,
            &self.queue,
            &self.pipeline_resources.shapes,
            self.state.scale_factor,
        )
    }

    pub(super) fn prepare_render(&mut self) -> Result<(), GeometryBufferError> {
        self.begin_frame_scratch();
        // Include prepared effect leaves in this upload without making them part
        // of the durable user draw queue.
        let base_vertex_count = self.state.shape_execution.vertices.len();
        let base_index_count = self.state.shape_execution.indices.len();
        let base_instance_count = self.state.shape_execution.instance_transforms.len();
        self.prepare_shape_effect_leaves()?;
        self.state
            .shape_execution
            .upload(&self.device, &self.queue, &mut self.state.buffers);
        let execution = &mut self.state.shape_execution;
        execution.vertices.truncate(base_vertex_count);
        execution.indices.truncate(base_index_count);
        execution.instance_transforms.truncate(base_instance_count);
        execution.instance_colors.truncate(base_instance_count);
        execution.instance_metadata.truncate(base_instance_count);
        Ok(())
    }
}
