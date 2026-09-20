//! Custom shader effects for groups, backdrops, and cacheable shape-local masks.
//!
//! Each attachment supplies different input pixels:
//!
//! - Group effects process a rendered subtree captured into an offscreen texture.
//! - Backdrop effects process previously rendered scene pixels behind the node.
//! - Shape effects process a padded white coverage mask of one shape. The result
//!   texture is cached while the shape and effect inputs are unchanged.
//!
//! `load_effect()` compiles WGSL shaders into GPU pipelines cached by `effect_id`.
//! `set_group_effect()`, `set_shape_backdrop_effect()`, and `set_shape_effect()`
//! attach a loaded effect to a draw tree node. Nodes share the compiled pipelines
//! and can supply different parameters.

use crate::gradient::gpu::GpuMaterialParams;
use crate::pipeline::BackdropSamplingUniform;
use naga::front::wgsl::{self, ParseError};
use naga::valid::{ValidationError, Validator};
use naga::{AddressSpace, ShaderStage, WithSpan};
use std::sync::Arc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

#[cfg(test)]
mod tests;

/// Why an effect shader was rejected.
#[derive(Debug, Clone, thiserror::Error)]
pub enum EffectShaderError {
    #[error("WGSL parsing failed: {0}")]
    Parse(#[source] Box<ParseError>),
    #[error("WGSL validation failed: {0}")]
    Validation(#[source] Box<WithSpan<ValidationError>>),
    #[error("Missing @fragment entry point effect_main")]
    MissingFragmentEntryPoint,
    #[error(
        "Unsupported resource at @group({group}) @binding({binding}); effect parameters must be a uniform at @group(1) @binding(0)"
    )]
    UnsupportedBinding { group: u32, binding: u32 },
    #[error("Only one effect parameter uniform may be declared")]
    DuplicateParameterBinding,
}

/// Errors from loading or attaching effects and updating their parameters.
#[derive(Debug, Clone, thiserror::Error)]
pub enum EffectError {
    /// WGSL or the effect interface is invalid for the zero-based pass index.
    #[error("Invalid shader in effect pass {pass_index}: {reason}")]
    InvalidShader {
        pass_index: usize,
        #[source]
        reason: EffectShaderError,
    },
    /// The referenced effect_id has not been loaded.
    #[error("Effect {0} has not been loaded")]
    EffectNotLoaded(u64),
    /// The referenced node_id does not exist in the draw tree.
    #[error("Node {0} not found in draw tree")]
    NodeNotFound(usize),
    /// Parameter data does not match the existing uniform buffer size.
    #[error(
        "Effect {effect_id} expects {expected_size} parameter bytes but {actual_size} were provided"
    )]
    ParameterSizeMismatch {
        effect_id: u64,
        expected_size: u64,
        actual_size: u64,
    },
    /// Invalid effect parameters or configuration.
    #[error("Invalid effect parameters: {0}")]
    InvalidParams(String),
}

/// The rendered region to capture as input to a backdrop effect.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub enum BackdropCaptureArea {
    /// Capture the node's transformed local bounds.
    #[default]
    NodeBounds,
    /// Capture the entire viewport.
    FullScene,
    /// Capture an explicit logical screen-space rectangle.
    ScreenRect([(f32, f32); 2]),
}

/// Per-node configuration for backdrop capture before the effect shader runs.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BackdropEffectConfig {
    /// The rendered region to capture, in logical screen coordinates.
    pub capture_area: BackdropCaptureArea,
    /// Additional logical screen-space padding applied around the requested capture area.
    ///
    /// Blur effects need pixels outside the node bounds to avoid clipped edges.
    pub padding: f32,
    /// Scale factor applied to the captured region before running the effect.
    /// `1.0` keeps full resolution, `0.5` halves each axis, and so on.
    pub downsample: f32,
}

impl Default for BackdropEffectConfig {
    fn default() -> Self {
        Self {
            capture_area: BackdropCaptureArea::NodeBounds,
            padding: 0.0,
            downsample: 1.0,
        }
    }
}

impl BackdropEffectConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn capture_area(mut self, capture_area: BackdropCaptureArea) -> Self {
        self.capture_area = capture_area;
        self
    }

    pub fn padding(mut self, padding: f32) -> Self {
        self.padding = padding;
        self
    }

    pub fn downsample(mut self, downsample: f32) -> Self {
        self.downsample = downsample;
        self
    }
}

/// Padding around a shape's local bounds for a cached shape effect.
/// Outsets are measured in logical pixels.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ShapeEffectConfig {
    pub left_outset: f32,
    pub top_outset: f32,
    pub right_outset: f32,
    pub bottom_outset: f32,
    /// Scale factor applied to the mask and effect textures before running the
    /// effect. `1.0` keeps full resolution, `0.5` halves each axis, and so on.
    /// The smaller result texture is bilinearly upscaled to the shape's full
    /// bounds when drawn. Effect shaders operate in texels of the downsampled
    /// texture, so texel-based radii and offsets scale up visually.
    pub downsample: f32,
}

impl Default for ShapeEffectConfig {
    fn default() -> Self {
        Self {
            left_outset: 0.0,
            top_outset: 0.0,
            right_outset: 0.0,
            bottom_outset: 0.0,
            downsample: 1.0,
        }
    }
}

impl ShapeEffectConfig {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets all four outsets to the same logical-space distance.
    pub fn outset(mut self, outset: f32) -> Self {
        self.left_outset = outset;
        self.top_outset = outset;
        self.right_outset = outset;
        self.bottom_outset = outset;
        self
    }

    /// Sets the left, top, right, and bottom logical-space outsets.
    pub fn outsets(mut self, left: f32, top: f32, right: f32, bottom: f32) -> Self {
        self.left_outset = left;
        self.top_outset = top;
        self.right_outset = right;
        self.bottom_outset = bottom;
        self
    }

    /// Sets the rasterization scale for the mask and effect textures.
    /// Must be in the range `(0.0, 1.0]`; values below `1.0` render the effect
    /// at reduced resolution and bilinearly upscale it when drawing.
    pub fn downsample(mut self, downsample: f32) -> Self {
        self.downsample = downsample;
        self
    }
}

/// Draws a fullscreen triangle from three vertex indices, without a vertex buffer.
/// Effect and composite passes share this shader.
pub(crate) const FULLSCREEN_TRIANGLE_VS: &str = include_str!("shaders/fullscreen_triangle_vs.wgsl");

/// Input texture bindings prepended to the user's effect fragment shader.
pub(crate) const EFFECT_FS_PREAMBLE: &str = include_str!("shaders/effect_fs_preamble.wgsl");

/// Samples effect results for compositing into the parent target.
pub(crate) const COMPOSITE_FS: &str = include_str!("shaders/composite_fs.wgsl");

const BACKDROP_LAYER_COMPOSITE_FS: &str = include_str!("shaders/backdrop_layer_composite_fs.wgsl");

/// A single compiled pass within a multi-pass effect.
pub(crate) struct LoadedEffectPass {
    /// The compiled render pipeline for this pass's fullscreen triangle.
    pub pipeline: wgpu::RenderPipeline,
    /// Whether this pass references user parameters at `@group(1)`.
    pub has_params: bool,
}

/// Compiled effect passes cached by `effect_id` and shared across nodes.
pub(crate) struct LoadedEffect {
    /// Exact pass sources used to compile this effect.
    pub pass_sources: Box<[Box<str>]>,
    /// Compiled passes, executed sequentially with ping-pong textures.
    pub passes: Vec<LoadedEffectPass>,
    /// The input texture and sampler layout at group 0.
    pub input_bind_group_layout: wgpu::BindGroupLayout,
    /// The user's parameter uniform layout at group 1.
    /// None if no pass uses user params. Shared across all passes that reference it.
    pub params_bind_group_layout: Option<wgpu::BindGroupLayout>,
}

/// A cached shape effect attachment. GPU parameter resources are created only on cache misses.
#[derive(Clone)]
pub(crate) struct ShapeEffectInstance {
    pub effect_id: u64,
    pub params: Arc<[u8]>,
    pub config: ShapeEffectConfig,
}

/// Uniform buffer and its group 1 binding.
pub(crate) struct EffectParameterResources {
    pub buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

pub(crate) struct CompositePipelineResources {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

/// Parameters shared by group and backdrop effect attachments.
pub(crate) struct EffectInstance {
    /// The loaded effect's ID.
    pub effect_id: u64,
    /// Raw bytes for the effect's uniform parameters.
    /// The byte layout must match the shader's uniform declaration.
    pub params: Vec<u8>,
    /// Created when attaching a parameterized effect; updated when its parameters change.
    pub parameter_resources: Option<EffectParameterResources>,
}

/// A backdrop attachment and its cached capture bindings.
pub(crate) struct BackdropEffectInstance {
    pub effect: EffectInstance,
    pub config: BackdropEffectConfig,
    /// Persistent uniform buffer for backdrop material params bound at group 3 binding 0.
    pub backdrop_material_params_buffer: Option<wgpu::Buffer>,
    /// Persistent uniform buffer for compositing a group subtree into its backdrop capture.
    pub backdrop_layer_params_buffer: Option<wgpu::Buffer>,
    /// Cached backdrop bind group reused while the captured output texture identity is stable.
    pub backdrop_texture_bind_group: Option<wgpu::BindGroup>,
    /// Stable id of the pooled texture currently referenced by `backdrop_texture_bind_group`.
    pub backdrop_texture_id: Option<u64>,
}

impl BackdropEffectInstance {
    pub(crate) fn new(effect: EffectInstance, config: BackdropEffectConfig) -> Self {
        Self {
            effect,
            config,
            backdrop_material_params_buffer: None,
            backdrop_layer_params_buffer: None,
            backdrop_texture_bind_group: None,
            backdrop_texture_id: None,
        }
    }
}

pub(crate) fn backdrop_layer_params(
    capture_origin: (i32, i32),
    source_size: (u32, u32),
) -> [i32; 4] {
    [
        capture_origin.0,
        capture_origin.1,
        i32::try_from(source_size.0).unwrap_or(i32::MAX),
        i32::try_from(source_size.1).unwrap_or(i32::MAX),
    ]
}

/// A pooled offscreen texture with color, optional depth/stencil, and optional MSAA resolve
/// resources.
pub(crate) struct PooledTexture {
    pub texture_id: u64,
    pub color_texture: wgpu::Texture,
    pub color_view: wgpu::TextureView,
    pub depth_stencil_view: Option<wgpu::TextureView>,
    pub resolve_texture: Option<wgpu::Texture>,
    pub resolve_view: Option<wgpu::TextureView>,
    pub width: u32,
    pub height: u32,
    pub sample_count: u32,
}

/// Pool of reusable offscreen textures for effect compositing.
/// Textures return to the pool after render submission.
pub(crate) struct OffscreenTexturePool {
    available: Vec<PooledTexture>,
    next_texture_id: u64,
}

/// Maximum number of textures to keep in the pool.
const MAX_POOL_SIZE: usize = 8;

impl OffscreenTexturePool {
    pub fn new() -> Self {
        Self {
            available: Vec::new(),
            next_texture_id: 1,
        }
    }

    /// Return textures to the pool and discard entries beyond `MAX_POOL_SIZE`.
    pub fn recycle(&mut self, textures: &mut Vec<PooledTexture>) {
        self.available.append(textures);
        self.available.truncate(MAX_POOL_SIZE);
    }

    /// Retain textures matching the dimensions and sample count, capped at `MAX_POOL_SIZE`.
    pub fn trim(&mut self, width: u32, height: u32, sample_count: u32) {
        self.available
            .retain(|t| t.width == width && t.height == height && t.sample_count == sample_count);
        if self.available.len() > MAX_POOL_SIZE {
            self.available.truncate(MAX_POOL_SIZE);
        }
    }

    /// Acquire a texture matching the given dimensions and sample count, plus a depth/stencil
    /// attachment for render passes that write depth or stencil.
    pub fn acquire_with_depth(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> PooledTexture {
        self.acquire(device, width, height, format, sample_count, true)
    }

    /// Acquire a color-only texture matching the given dimensions and sample count.
    pub fn acquire_color_only(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> PooledTexture {
        self.acquire(device, width, height, format, sample_count, false)
    }

    fn acquire(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        sample_count: u32,
        with_depth: bool,
    ) -> PooledTexture {
        let found = self.available.iter().position(|texture| {
            texture.width == width
                && texture.height == height
                && texture.sample_count == sample_count
                && texture.depth_stencil_view.is_some() == with_depth
        });

        if let Some(idx) = found {
            self.available.swap_remove(idx)
        } else {
            self.create_pooled_texture(device, width, height, format, sample_count, with_depth)
        }
    }

    fn create_pooled_texture(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        sample_count: u32,
        with_depth: bool,
    ) -> PooledTexture {
        let texture_id = self.next_texture_id;
        self.next_texture_id += 1;

        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("effect_offscreen_color"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_stencil_view = with_depth.then(|| {
            let depth_stencil_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("effect_offscreen_depth_stencil"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            depth_stencil_texture.create_view(&wgpu::TextureViewDescriptor::default())
        });

        // Effect shaders sample the resolved texture when MSAA is enabled.
        let (resolve_texture, resolve_view) = if sample_count > 1 {
            let resolve_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("effect_offscreen_resolve"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let resolve_v = resolve_tex.create_view(&wgpu::TextureViewDescriptor::default());
            (Some(resolve_tex), Some(resolve_v))
        } else {
            (None, None)
        };

        PooledTexture {
            texture_id,
            color_texture,
            color_view,
            depth_stencil_view,
            resolve_texture,
            resolve_view,
            width,
            height,
            sample_count,
        }
    }
}

/// Creates the layout for the input texture and sampler at group 0.
pub(crate) fn create_effect_input_bind_group_layout(
    device: &wgpu::Device,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("effect_input_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

/// Creates the user parameter uniform layout at group 1, binding 0.
pub(crate) fn create_effect_params_bind_group_layout(
    device: &wgpu::Device,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("effect_params_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

/// Combines the built-in vertex shader, input bindings, and user fragment shader into one module.
pub(crate) fn build_effect_wgsl(user_fragment_source: &str) -> String {
    format!("{FULLSCREEN_TRIANGLE_VS}\n{EFFECT_FS_PREAMBLE}\n{user_fragment_source}")
}

/// Combines the fullscreen vertex shader and passthrough fragment shader.
pub(crate) fn build_composite_wgsl() -> String {
    format!("{FULLSCREEN_TRIANGLE_VS}\n{COMPOSITE_FS}")
}

/// Validates the complete effect module and detects its parameter uniform.
fn validate_effect_shader(
    source: &str,
    validator: &mut Validator,
) -> Result<bool, EffectShaderError> {
    let module =
        wgsl::parse_str(source).map_err(|error| EffectShaderError::Parse(Box::new(error)))?;
    validator
        .validate(&module)
        .map_err(|error| EffectShaderError::Validation(Box::new(error)))?;
    if !module.entry_points.iter().any(|entry_point| {
        entry_point.name == "effect_main" && entry_point.stage == ShaderStage::Fragment
    }) {
        return Err(EffectShaderError::MissingFragmentEntryPoint);
    }

    let mut has_params = false;
    for (_, variable) in module.global_variables.iter() {
        let Some(binding) = &variable.binding else {
            continue;
        };
        if binding.group == 0 && matches!(variable.name.as_deref(), Some("t_input" | "s_input")) {
            continue;
        }
        if binding.group != 1 || binding.binding != 0 || variable.space != AddressSpace::Uniform {
            return Err(EffectShaderError::UnsupportedBinding {
                group: binding.group,
                binding: binding.binding,
            });
        }
        if has_params {
            return Err(EffectShaderError::DuplicateParameterBinding);
        }
        has_params = true;
    }
    Ok(has_params)
}

/// Compiles one or more WGSL passes into an effect.
///
/// Each entry in `pass_sources` is a WGSL fragment shader for one pass.
/// Passes execute sequentially; each reads the previous pass's output via `t_input`.
/// Passes that use user parameters share the uniform layout at group 1.
///
/// For single-pass effects, pass a one-element slice.
pub(crate) fn compile_effect_pipeline(
    device: &wgpu::Device,
    pass_sources: &[&str],
    format: wgpu::TextureFormat,
    validator: &mut Validator,
) -> Result<LoadedEffect, EffectError> {
    if pass_sources.is_empty() {
        return Err(EffectError::InvalidParams(
            "At least one effect pass is required".into(),
        ));
    }

    let validated_passes = pass_sources
        .iter()
        .enumerate()
        .map(|(pass_index, fragment_source)| {
            let source = build_effect_wgsl(fragment_source);
            let has_params = validate_effect_shader(&source, validator)
                .map_err(|reason| EffectError::InvalidShader { pass_index, reason })?;
            Ok((source, has_params))
        })
        .collect::<Result<Vec<_>, EffectError>>()?;

    let input_bgl = create_effect_input_bind_group_layout(device);

    // Passes using group 1 share the parameter layout.
    let any_has_params = validated_passes.iter().any(|(_, has_params)| *has_params);
    let params_bgl = if any_has_params {
        Some(create_effect_params_bind_group_layout(device))
    } else {
        None
    };

    let mut passes = Vec::with_capacity(pass_sources.len());

    for (i, (full_wgsl, pass_has_params)) in validated_passes.into_iter().enumerate() {
        let shader_label = format!("effect_pass{i}_shader");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&shader_label),
            source: wgpu::ShaderSource::Wgsl(full_wgsl.into()),
        });

        // Include group 1 only for passes that reference it.
        let layout_label = format!("effect_pass{i}_layout");
        let pipeline_layout = if pass_has_params {
            let bind_group_layouts = [&input_bgl, params_bgl.as_ref().unwrap()];
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&layout_label),
                bind_group_layouts: &bind_group_layouts,
                push_constant_ranges: &[],
            })
        } else {
            let bind_group_layouts = [&input_bgl];
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&layout_label),
                bind_group_layouts: &bind_group_layouts,
                push_constant_ranges: &[],
            })
        };

        let pipeline_label = format!("effect_pass{i}_pipeline");
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&pipeline_label),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_triangle"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("effect_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        passes.push(LoadedEffectPass {
            pipeline,
            has_params: pass_has_params,
        });
    }

    Ok(LoadedEffect {
        pass_sources: pass_sources
            .iter()
            .map(|source| Box::<str>::from(*source))
            .collect(),
        passes,
        input_bind_group_layout: input_bgl,
        params_bind_group_layout: params_bgl,
    })
}

/// Compiles the composite pipeline, which samples the effect result and respects the parent clip.
pub(crate) fn compile_composite_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> CompositePipelineResources {
    let wgsl = build_composite_wgsl();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("composite_shader"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });

    let input_bgl = create_effect_input_bind_group_layout(device);

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("composite_pipeline_layout"),
        bind_group_layouts: &[&input_bgl],
        push_constant_ranges: &[],
    });

    // Respect the parent's clip without changing stencil values.
    let stencil_face = wgpu::StencilFaceState {
        compare: wgpu::CompareFunction::Equal,
        fail_op: wgpu::StencilOperation::Keep,
        depth_fail_op: wgpu::StencilOperation::Keep,
        pass_op: wgpu::StencilOperation::Keep,
    };

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("composite_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_triangle"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_composite"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Always,
            stencil: wgpu::StencilState {
                front: stencil_face,
                back: stencil_face,
                read_mask: 0xff,
                write_mask: 0x00,
            },
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    CompositePipelineResources {
        pipeline,
        bind_group_layout: input_bgl,
    }
}

/// Compile a fullscreen texture-sampling pipeline without stencil/depth usage.
/// Used for capture downsampling before running the user effect shader.
pub(crate) fn compile_texture_blit_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    input_bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let wgsl = build_composite_wgsl();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("texture_blit_shader"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("texture_blit_pipeline_layout"),
        bind_group_layouts: &[input_bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("texture_blit_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_triangle"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_composite"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

/// Compile a fullscreen pipeline that overlays an already-rendered transparent group prefix
/// onto a backdrop capture using premultiplied-alpha blending.
pub(crate) fn compile_backdrop_layer_composite_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> CompositePipelineResources {
    let shader_source = format!("{FULLSCREEN_TRIANGLE_VS}\n{BACKDROP_LAYER_COMPOSITE_FS}");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("backdrop_layer_composite_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("backdrop_layer_composite_bind_group_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("backdrop_layer_composite_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("backdrop_layer_composite_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_triangle"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_backdrop_layer"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    CompositePipelineResources {
        pipeline,
        bind_group_layout,
    }
}

pub(crate) fn create_backdrop_layer_composite_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    foreground_view: &wgpu::TextureView,
    params_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("backdrop_layer_composite_bind_group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(foreground_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    })
}

/// Creates a texture binding for effect input or compositing.
pub(crate) fn create_texture_sample_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    texture_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    label: Option<&str>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label,
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

pub(crate) fn create_backdrop_texture_sample_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    material_params_buffer: &wgpu::Buffer,
    texture_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    label: Option<&str>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label,
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: material_params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

pub(crate) fn prepare_solid_backdrop_material_params_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    backdrop_material_params_buffer: &mut Option<wgpu::Buffer>,
    sampling_uniform: BackdropSamplingUniform,
) -> wgpu::Buffer {
    let material_params = GpuMaterialParams::for_backdrop_sampling(sampling_uniform);

    if let Some(existing_buffer) = backdrop_material_params_buffer.as_ref() {
        queue.write_buffer(existing_buffer, 0, bytemuck::bytes_of(&material_params));
    } else {
        *backdrop_material_params_buffer = Some(device.create_buffer_init(&BufferInitDescriptor {
            label: Some("solid_backdrop_material_params_buffer"),
            contents: bytemuck::bytes_of(&material_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        }));
    }

    backdrop_material_params_buffer
        .as_ref()
        .expect("backdrop material params buffer should be initialized")
        .clone()
}

pub(crate) fn prepare_backdrop_layer_params_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    backdrop_layer_params_buffer: &mut Option<wgpu::Buffer>,
    layer_params: [i32; 4],
) -> wgpu::Buffer {
    if let Some(existing_buffer) = backdrop_layer_params_buffer.as_ref() {
        queue.write_buffer(existing_buffer, 0, bytemuck::bytes_of(&layer_params));
    } else {
        *backdrop_layer_params_buffer = Some(device.create_buffer_init(&BufferInitDescriptor {
            label: Some("backdrop_layer_params_buffer"),
            contents: bytemuck::bytes_of(&layer_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        }));
    }

    backdrop_layer_params_buffer
        .as_ref()
        .expect("backdrop layer params buffer should be initialized")
        .clone()
}

/// Create a bind group for effect parameter uniforms.
pub(crate) fn create_params_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("effect_params_bg"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    })
}
