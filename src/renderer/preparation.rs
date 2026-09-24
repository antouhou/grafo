use super::backend::execution::shapes::ShapeDrawResources;
use super::types::GeometryBufferError;
use super::Renderer;
use crate::renderer::types::CachedShapeDrawData;

impl Renderer<'_> {
    pub(super) fn append_shape_resources(
        &mut self,
        shape: &mut CachedShapeDrawData,
    ) -> Result<ShapeDrawResources, GeometryBufferError> {
        self.planner.refresh_geometry_cache(shape);
        self.backend.resources.shape_execution.prepare_draw(
            shape,
            &self.backend.device,
            &self.backend.queue,
            &self.backend.pipeline_resources.shapes,
            self.viewport.scale_factor,
        )
    }
}
