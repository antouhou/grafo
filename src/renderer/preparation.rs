use super::execution::shapes::ShapeDrawResources;
use super::plan::groups::GroupPlanningInput;
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

    pub(super) fn prepare_render(&mut self) {
        self.begin_frame_scratch();
        self.state.scratch.shape_effect_plan.plan(
            &self.state.draw_tree,
            &self.state.shape_effects,
            self.state.scale_factor,
            self.fringe_width,
            self.state.physical_size.into(),
            self.device.limits().max_texture_dimension_2d,
        );
        self.state.scratch.group_planner.plan(
            GroupPlanningInput {
                tree: &self.state.draw_tree,
                group_effects: &self.state.group_effects,
                backdrop_effects: &self.state.backdrop_effects,
                shape_effects: &self.state.scratch.shape_effect_plan.composites,
                scale_factor: self.state.scale_factor,
                physical_size: self.state.physical_size.into(),
                max_capture_dimension: self.device.limits().max_texture_dimension_2d,
            },
            &mut self.state.scratch.draw_plan,
        );
        self.state
            .shape_execution
            .upload(&self.device, &self.queue, &mut self.state.buffers);
    }
}
