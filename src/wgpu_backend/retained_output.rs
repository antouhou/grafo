use super::execution::targets::{self, RegionClear};
use super::BackendCreationError;
use crate::core::UnsignedPhysicalRect;
use wgpu::{
    Backend, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, Color,
    ColorTargetState, ColorWrites, CommandEncoder, CompareFunction, CompositeAlphaMode,
    DepthStencilState, Device, Extent3d, FragmentState, LoadOp, MultisampleState, Operations,
    PipelineCompilationOptions, PipelineLayout, PipelineLayoutDescriptor,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor,
    ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages, StoreOp, SurfaceColorSpace,
    SurfaceConfiguration, Texture, TextureDescriptor, TextureDimension, TextureFormat,
    TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension,
    VertexState,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct OutputColorFormat {
    pub(super) texture_format: TextureFormat,
    color_space: SurfaceColorSpace,
    alpha_mode: CompositeAlphaMode,
}

impl OutputColorFormat {
    pub fn for_surface(
        configuration: &SurfaceConfiguration,
        backend: Backend,
    ) -> Result<Self, BackendCreationError> {
        let supported = match configuration.color_space {
            SurfaceColorSpace::Srgb => true,
            // Extended sRGB needs to store negative and above-white RGB values.
            SurfaceColorSpace::ExtendedSrgbLinear | SurfaceColorSpace::ExtendedSrgb => matches!(
                configuration.format,
                TextureFormat::Rgba16Float | TextureFormat::Rgba32Float
            ),
            _ => false,
        };
        if !supported {
            return Err(BackendCreationError::UnsupportedSurface);
        }
        let alpha_mode = match (backend, configuration.alpha_mode) {
            // wgpu 30.0.2 advertises PostMultiplied for nonopaque CAMetalLayers.
            (Backend::Metal, CompositeAlphaMode::PostMultiplied) => {
                CompositeAlphaMode::PreMultiplied
            }
            (
                _,
                mode @ (CompositeAlphaMode::PreMultiplied | CompositeAlphaMode::PostMultiplied),
            ) => mode,
            _ => CompositeAlphaMode::Opaque,
        };
        Ok(Self {
            texture_format: configuration.format,
            color_space: configuration.color_space,
            alpha_mode,
        })
    }

    pub fn opaque(texture_format: TextureFormat) -> Self {
        Self {
            texture_format,
            color_space: SurfaceColorSpace::Srgb,
            alpha_mode: CompositeAlphaMode::Opaque,
        }
    }

    fn needs_sdr_clamp(self) -> bool {
        // UNORM clips on write. Premultiplied sRGB needs to clip before multiplying by alpha.
        let clips_on_write = matches!(
            self.texture_format.remove_srgb_suffix(),
            TextureFormat::Rgba8Unorm | TextureFormat::Bgra8Unorm | TextureFormat::Rgb10a2Unorm
        );
        self.color_space == SurfaceColorSpace::Srgb
            && (!clips_on_write || self.alpha_mode == CompositeAlphaMode::PreMultiplied)
    }
}

fn create_output_pipeline(
    device: &Device,
    shader: &ShaderModule,
    output_format: OutputColorFormat,
    fragment_entry_point: &str,
    sample_count: u32,
    depth_stencil: Option<DepthStencilState>,
    layout: Option<&PipelineLayout>,
) -> RenderPipeline {
    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some(fragment_entry_point),
        layout,
        vertex: VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(FragmentState {
            module: shader,
            entry_point: Some(fragment_entry_point),
            compilation_options: PipelineCompilationOptions {
                constants: &[
                    (
                        "linear_output",
                        f64::from(
                            output_format.color_space == SurfaceColorSpace::ExtendedSrgbLinear,
                        ),
                    ),
                    ("clamp_to_sdr", f64::from(output_format.needs_sdr_clamp())),
                    (
                        "attachment_is_srgb",
                        f64::from(output_format.texture_format.is_srgb()),
                    ),
                    (
                        "straight_alpha",
                        f64::from(output_format.alpha_mode == CompositeAlphaMode::PostMultiplied),
                    ),
                    (
                        "premultiplied_alpha",
                        f64::from(output_format.alpha_mode == CompositeAlphaMode::PreMultiplied),
                    ),
                ],
                ..Default::default()
            },
            targets: &[Some(ColorTargetState {
                format: output_format.texture_format,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        primitive: Default::default(),
        depth_stencil,
        multisample: MultisampleState {
            count: sample_count,
            ..Default::default()
        },
        multiview_mask: None,
        cache: None,
    })
}

pub(super) struct RetainedOutputResources {
    shader: ShaderModule,
    texture_layout: BindGroupLayout,
    presentation_layout: PipelineLayout,
}

impl RetainedOutputResources {
    pub fn new(device: &Device) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("retained_output"),
            source: ShaderSource::Wgsl(include_str!("../shaders/retained_output.wgsl").into()),
        });
        let texture_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("retained_output_texture"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let presentation_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("present_retained_output"),
            bind_group_layouts: &[Some(&texture_layout)],
            immediate_size: 0,
        });
        Self {
            shader,
            texture_layout,
            presentation_layout,
        }
    }
}

/// One persistent image shared by surface presentation and pixmap readback.
pub(super) struct RetainedOutput {
    pub texture: Texture,
    pub view: TextureView,
    clear_pipeline: RenderPipeline,
    present_binding: BindGroup,
}

impl RetainedOutput {
    pub fn new(
        device: &Device,
        resources: &RetainedOutputResources,
        size: (u32, u32),
        render_format: TextureFormat,
        samples: u32,
    ) -> Self {
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("retained_output"),
            size: Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: render_format,
            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        let clear_pipeline = create_output_pipeline(
            device,
            &resources.shader,
            OutputColorFormat::opaque(render_format),
            "fs_clear",
            samples,
            Some(DepthStencilState {
                format: TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: Some(false),
                depth_compare: Some(CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            None,
        );
        let present_binding = device.create_bind_group(&BindGroupDescriptor {
            label: Some("retained_output"),
            layout: &resources.texture_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&view),
            }],
        });
        Self {
            texture,
            view,
            clear_pipeline,
            present_binding,
        }
    }

    /// Clears changed pixels at the start of the scene's first render pass.
    pub fn region_clear(&self, scissor: UnsignedPhysicalRect) -> RegionClear<'_> {
        RegionClear {
            pipeline: &self.clear_pipeline,
            scissor,
        }
    }
}

/// Presentation pipelines for one window surface configuration.
pub(super) struct SurfacePresentation {
    present_pipeline: RenderPipeline,
    dirty_region_overlay_pipeline: RenderPipeline,
}

impl SurfacePresentation {
    pub fn new(
        device: &Device,
        resources: &RetainedOutputResources,
        output_format: OutputColorFormat,
    ) -> Self {
        let present_pipeline = create_output_pipeline(
            device,
            &resources.shader,
            output_format,
            "fs_present",
            1,
            None,
            Some(&resources.presentation_layout),
        );
        let dirty_region_overlay_pipeline = create_output_pipeline(
            device,
            &resources.shader,
            output_format,
            "fs_dirty_region_overlay",
            1,
            None,
            Some(&resources.presentation_layout),
        );
        Self {
            present_pipeline,
            dirty_region_overlay_pipeline,
        }
    }

    /// Copies the clean scene to the disposable output.
    pub fn present(
        &self,
        encoder: &mut CommandEncoder,
        retained: &RetainedOutput,
        output: &TextureView,
        overlay: Option<UnsignedPhysicalRect>,
    ) {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("present_retained_output"),
            color_attachments: &[Some(RenderPassColorAttachment {
                depth_slice: None,
                view: output,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::TRANSPARENT),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.present_pipeline);
        pass.set_bind_group(0, &retained.present_binding, &[]);
        pass.draw(0..3, 0..1);
        if let Some(scissor) = overlay {
            targets::set_scissor(&mut pass, scissor);
            pass.set_pipeline(&self.dirty_region_overlay_pipeline);
            pass.draw(0..3, 0..1);
        }
    }
}
