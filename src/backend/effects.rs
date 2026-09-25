use super::execution::effects::{compile_composite_pipeline, CompositePipelineResources};
use super::execution::shapes::TextureMaterialPipelines;
use super::resources::BackdropPipelineResources;
use super::WgpuBackend;

impl WgpuBackend {
    pub(in crate::backend) fn ensure_composite_pipeline(&mut self) -> &CompositePipelineResources {
        self.pipeline_resources
            .composite_resources
            .get_or_insert_with(|| {
                compile_composite_pipeline(&self.device, self.config.format, self.msaa_sample_count)
            })
    }

    pub(in crate::backend) fn ensure_backdrop_pipelines(&mut self) {
        if self.pipeline_resources.backdrops.is_some() {
            return;
        }

        self.ensure_composite_pipeline();
        if self
            .pipeline_resources
            .shapes
            .under_fill_pipelines
            .is_none()
        {
            let under_fill_pipelines = TextureMaterialPipelines::new(
                &self.device,
                self.config.format,
                self.msaa_sample_count,
                &self.pipeline_resources.shapes,
            );
            self.pipeline_resources.shapes.under_fill_pipelines = Some(under_fill_pipelines);
        }
        let resources = &self.pipeline_resources;
        let composite = resources
            .composite_resources
            .as_ref()
            .expect("composite resources were initialized above");
        self.pipeline_resources.backdrops = Some(BackdropPipelineResources::new(
            &self.device,
            self.config.format,
            &composite.bind_group_layout,
        ));
    }

    pub(in crate::backend) fn ensure_effect_sampler(&mut self) {
        if self.pipeline_resources.effect_sampler.is_none() {
            self.pipeline_resources.effect_sampler =
                Some(self.device.create_sampler(&wgpu::SamplerDescriptor {
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::FilterMode::Linear,
                    ..Default::default()
                }));
        }
    }
}
