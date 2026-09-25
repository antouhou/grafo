use super::execution::effects::{EffectExecutionResources, EffectRegistry};
use super::execution::shape_effects::ShapeEffectRendererResources;
use super::execution::shapes::ShapeExecutionResources;
use super::execution::textures::IntermediateTextureResources;
use super::resources::{BackendResources, Buffers, RendererPipelineResources, ShapePipelines};
use super::{WgpuBackend, WgpuContext};
use crate::core::Viewport;
use std::sync::Arc;
use wgpu::SurfaceConfiguration;

const DEFAULT_FRINGE_WIDTH: f32 = 0.75;

impl WgpuBackend {
    pub(in crate::backend) fn new(
        context: Arc<WgpuContext>,
        config: SurfaceConfiguration,
        physical_size: (u32, u32),
        scale_factor: f64,
        msaa_sample_count: u32,
    ) -> Self {
        let device = context.device.clone();
        let resources = ShapePipelines::new(
            &context,
            &config,
            physical_size,
            scale_factor,
            DEFAULT_FRINGE_WIDTH,
            msaa_sample_count,
            None,
        );
        let queue = context.queue.clone();
        let shape_effect_resources = ShapeEffectRendererResources::new(&device, config.format);
        let effect_registry = EffectRegistry::new(&device);

        let supports_base_vertex = context.supports_base_vertex;
        let mut backend = Self {
            context,
            device,
            queue,
            config,
            fringe_width: DEFAULT_FRINGE_WIDTH,
            readback_bytes: Vec::new(),
            viewport: Viewport {
                physical_size,
                scale_factor,
            },
            pipeline_resources: RendererPipelineResources {
                shapes: resources,
                shape_effects: shape_effect_resources,
                effect_sampler: None,
                composite_resources: None,
                backdrops: None,
            },
            argb_readback: None,
            bgra_readback: None,
            msaa_sample_count,
            msaa_color_texture: None,
            msaa_color_texture_view: None,
            depth_stencil_texture: None,
            depth_stencil_view: None,
            effect_registry,
            #[cfg(feature = "render_metrics")]
            last_phase_timings: Default::default(),
            last_render_to_texture_view_cpu_time: Default::default(),
            resources: BackendResources {
                shape_execution: ShapeExecutionResources::new(),
                effect_execution: EffectExecutionResources::default(),
                textures: IntermediateTextureResources::new(),
                #[cfg(feature = "render_metrics")]
                pipeline_switch_counts: Default::default(),
                #[cfg(feature = "render_metrics")]
                shape_effect_cache_metrics: Default::default(),
                buffers: Buffers {
                    supports_base_vertex,
                    aggregated_vertex_buffer: None,
                    aggregated_index_buffer: None,
                    aggregated_instance_transform_buffer: None,
                    aggregated_instance_color_buffer: None,
                    aggregated_instance_metadata_buffer: None,
                },
            },
        };

        backend.recreate_msaa_texture();
        backend.recreate_depth_stencil_texture();
        backend
    }

    pub(in crate::backend) fn recreate_pipelines(&mut self) {
        let resources = ShapePipelines::new(
            &self.context,
            &self.config,
            self.viewport.physical_size,
            self.viewport.scale_factor,
            self.fringe_width,
            self.msaa_sample_count,
            Some(
                self.pipeline_resources
                    .shapes
                    .gradient_bind_group_layout
                    .clone(),
            ),
        );
        self.pipeline_resources.shapes = resources;
        self.resources
            .shape_execution
            .texture_materials
            .invalidate_bindings();

        self.resources.textures.clear_shape_effects();
        self.pipeline_resources.composite_resources = None;
        self.pipeline_resources
            .shape_effects
            .recreate_pipeline(&self.device, self.config.format);

        // Reset lazily-created pipelines so they pick up the new layout
        self.pipeline_resources.backdrops = None;

        // Gradient layouts and their uploaded resources remain valid across MSAA changes.
        for resources in self.resources.shape_execution.draws.values_mut() {
            resources.clear_under_fill_binding();
        }

        self.resources
            .effect_execution
            .backdrop_composites
            .invalidate_bindings();
    }
}
