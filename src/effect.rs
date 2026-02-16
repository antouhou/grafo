//! Effect system for group compositing with custom shaders.
//!
//! This module provides the infrastructure for rendering subtrees to offscreen textures,
//! applying user-defined WGSL post-processing effects, and compositing the results back
//! into the parent render target.
//!
//! The system separates **loading** (compile once) from **attaching** (use per node, cheap):
//! - `load_effect()` compiles a WGSL effect shader into a GPU pipeline, cached by `effect_id`.
//! - `set_group_effect()` attaches a loaded effect to a specific draw tree node with per-instance parameters.
//!
//! Multiple nodes can share the same loaded effect (same compiled pipeline), each with different parameters.

use std::collections::hash_map::DefaultHasher;
use std::hash::Hash;

// ── Error type ───────────────────────────────────────────────────────────────

/// Errors that can occur when working with the effect system.
#[derive(Debug, Clone)]
pub enum EffectError {
    /// The WGSL source failed to compile. Contains the error message from wgpu/naga.
    CompilationFailed(String),
    /// The referenced effect_id has not been loaded.
    EffectNotLoaded(u64),
    /// The referenced node_id does not exist in the draw tree.
    NodeNotFound(usize),
    /// Invalid parameter data (e.g. wrong size for uniform buffer).
    InvalidParams(String),
}

impl std::fmt::Display for EffectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EffectError::CompilationFailed(msg) => {
                write!(f, "Effect WGSL compilation failed: {msg}")
            }
            EffectError::EffectNotLoaded(id) => write!(f, "Effect {id} has not been loaded"),
            EffectError::NodeNotFound(id) => write!(f, "Node {id} not found in draw tree"),
            EffectError::InvalidParams(msg) => write!(f, "Invalid effect parameters: {msg}"),
        }
    }
}

impl std::error::Error for EffectError {}

// ── Built-in shaders ─────────────────────────────────────────────────────────

/// Built-in vertex shader for drawing a fullscreen triangle (3 vertices, no vertex buffer).
/// Used both by effect apply passes and the composite pass.
pub(crate) const FULLSCREEN_QUAD_VS: &str = r#"
struct QuadOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_quad(@builtin(vertex_index) vi: u32) -> QuadOutput {
    // Fullscreen triangle trick: 3 vertices cover the entire screen
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    var out: QuadOutput;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}
"#;

/// Built-in fragment shader preamble providing the input texture bindings.
/// This is prepended to the user's effect fragment shader.
pub(crate) const EFFECT_FS_PREAMBLE: &str = r#"
// -- Provided by the engine (group 0) --
@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var s_input: sampler;
"#;

/// Simple passthrough fragment shader for compositing effect results back into the parent target.
pub(crate) const COMPOSITE_FS: &str = r#"
@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var s_input: sampler;

@fragment
fn fs_composite(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(t_input, s_input, uv);
}
"#;

// ── Loaded effect (compiled pipeline) ────────────────────────────────────────

/// A single compiled pass within a multi-pass effect.
pub(crate) struct LoadedEffectPass {
    /// The compiled render pipeline for this pass's fullscreen quad.
    pub pipeline: wgpu::RenderPipeline,
    /// Whether this pass references @group(1) (user params).
    pub has_params: bool,
}

/// A loaded (compiled) effect. Stored in a cache on the Renderer, keyed by effect_id.
/// Multiple nodes can reference the same LoadedEffect.
/// Supports single-pass and multi-pass effects (e.g., separable Gaussian blur).
pub(crate) struct LoadedEffect {
    /// Compiled passes, executed sequentially with ping-pong textures.
    pub passes: Vec<LoadedEffectPass>,
    /// A bind group layout for the input texture (group 0): texture and sampler.
    pub input_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group layout for the user's parameter uniform (group 1).
    /// None if no pass uses user params. Shared across all passes that reference it.
    pub params_bind_group_layout: Option<wgpu::BindGroupLayout>,
}

// ── Per-node effect instance ─────────────────────────────────────────────────

/// A per-node effect instance. Stored in a HashMap<usize, EffectInstance> on the Renderer,
/// keyed by node_id.
pub(crate) struct EffectInstance {
    /// Reference to the loaded effect (by effect_id key).
    pub effect_id: u64,
    /// Raw bytes for the effect's uniform parameters.
    /// The user is responsible for ensuring the layout matches the shader.
    pub params: Vec<u8>,
    /// GPU buffer for the parameters (created/updated lazily).
    pub params_buffer: Option<wgpu::Buffer>,
    /// Bind group for the parameters (group 1).
    pub params_bind_group: Option<wgpu::BindGroup>,
}

// ── Offscreen texture pool ───────────────────────────────────────────────────

/// A pooled offscreen texture with color, depth/stencil, and optional MSAA resolve resources.
pub(crate) struct PooledTexture {
    pub color_view: wgpu::TextureView,
    pub depth_stencil_view: wgpu::TextureView,
    pub resolve_view: Option<wgpu::TextureView>,
    pub width: u32,
    pub height: u32,
    pub sample_count: u32,
}

/// Pool of reusable offscreen textures for effect compositing.
/// At frame start, all textures move back to `available`.
pub(crate) struct OffscreenTexturePool {
    available: Vec<PooledTexture>,
}

/// Maximum number of textures to keep in the pool.
const MAX_POOL_SIZE: usize = 8;

impl OffscreenTexturePool {
    pub fn new() -> Self {
        Self {
            available: Vec::new(),
        }
    }

    /// Return textures for reuse in future frames.
    /// Textures that don't match the given active configuration are dropped
    /// immediately, and the pool is capped at `MAX_POOL_SIZE`.
    pub fn recycle(&mut self, textures: Vec<PooledTexture>) {
        self.available.extend(textures);
    }

    /// Drop all pooled textures whose dimensions, or sample count don't match
    /// the current active configuration, and enforce the maximum pool size.
    /// Call this when size, format, or MSAA settings change (e.g. on resize).
    pub fn trim(&mut self, width: u32, height: u32, sample_count: u32) {
        self.available
            .retain(|t| t.width == width && t.height == height && t.sample_count == sample_count);
        // Enforce max pool size — drop oldest excess textures
        if self.available.len() > MAX_POOL_SIZE {
            self.available.truncate(MAX_POOL_SIZE);
        }
    }

    /// Acquire a texture matching the given dimensions and sample count.
    /// Reuses an existing one if possible, otherwise creates a new one.
    /// Returns an owned PooledTexture (moved out of the pool).
    pub fn acquire(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> PooledTexture {
        let found = self
            .available
            .iter()
            .position(|t| t.width == width && t.height == height && t.sample_count == sample_count);

        if let Some(idx) = found {
            self.available.swap_remove(idx)
        } else {
            Self::create_pooled_texture(device, width, height, format, sample_count)
        }
    }

    fn create_pooled_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> PooledTexture {
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

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
        let depth_stencil_view =
            depth_stencil_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // When MSAA is enabled, create a resolve target (non-MSAA) for effects to read from
        let resolve_view = if sample_count > 1 {
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
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let resolve_v = resolve_tex.create_view(&wgpu::TextureViewDescriptor::default());
            Some(resolve_v)
        } else {
            None
        };

        PooledTexture {
            // color_texture,
            color_view,
            // depth_stencil_texture,
            depth_stencil_view,
            // resolve_texture,
            resolve_view,
            width,
            height,
            sample_count,
        }
    }
}

// ── Pipeline creation helpers ────────────────────────────────────────────────

/// Creates the bind group layout for effect input: texture_2d + sampler at group(0).
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

/// Creates a bind group layout for user effect parameters: uniform buffer at group(1) binding(0).
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

/// Concatenate built-in vertex shader + preamble + user fragment shader into a single WGSL module.
pub(crate) fn build_effect_wgsl(user_fragment_source: &str) -> String {
    format!("{FULLSCREEN_QUAD_VS}\n{EFFECT_FS_PREAMBLE}\n{user_fragment_source}")
}

/// Build the composite WGSL module (fullscreen VS + passthrough FS).
pub(crate) fn build_composite_wgsl() -> String {
    format!("{FULLSCREEN_QUAD_VS}\n{COMPOSITE_FS}")
}

/// Checks if user WGSL source declares a `@group(1)` binding (params uniform).
///
/// Strips WGSL line (`//`) and block (`/* … */`) comments first, then matches
/// `@group(1)` with optional whitespace so that `@group( 1 )` and similar
/// variants are detected while occurrences inside comments are ignored.
pub(crate) fn has_user_params(user_fragment_source: &str) -> bool {
    // Strip block comments (/* ... */), then line comments (// ... \n).
    let no_block = regex::Regex::new(r"(?s)/\*.*?\*/")
        .unwrap()
        .replace_all(user_fragment_source, "");
    let stripped = regex::Regex::new(r"//[^\n]*")
        .unwrap()
        .replace_all(&no_block, "");

    regex::Regex::new(r"@group\s*\(\s*1\s*\)")
        .unwrap()
        .is_match(&stripped)
}

/// Compile a (possibly multi-pass) effect from WGSL source(s).
///
/// Each entry in `pass_sources` is a WGSL fragment shader for one pass.
/// Passes execute sequentially; each reads the previous pass's output via `t_input`.
/// All passes share the same user-params uniform layout (group 1) when present.
///
/// For single-pass effects, pass a one-element slice.
pub(crate) fn compile_effect_pipeline(
    device: &wgpu::Device,
    pass_sources: &[&str],
    format: wgpu::TextureFormat,
) -> Result<LoadedEffect, EffectError> {
    if pass_sources.is_empty() {
        return Err(EffectError::InvalidParams(
            "At least one effect pass is required".into(),
        ));
    }

    let input_bgl = create_effect_input_bind_group_layout(device);

    // Create the params BGL once if ANY pass uses @group(1)
    let any_has_params = pass_sources.iter().any(|s| has_user_params(s));
    let params_bgl = if any_has_params {
        Some(create_effect_params_bind_group_layout(device))
    } else {
        None
    };

    let mut hasher = DefaultHasher::new();
    let mut passes = Vec::with_capacity(pass_sources.len());

    for (i, &source) in pass_sources.iter().enumerate() {
        source.hash(&mut hasher);
        let full_wgsl = build_effect_wgsl(source);
        let pass_has_params = has_user_params(source);

        let shader_label = format!("effect_pass{i}_shader");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&shader_label),
            source: wgpu::ShaderSource::Wgsl(full_wgsl.into()),
        });

        // Each pass gets its own pipeline layout — only include group(1)
        // if this particular pass references it.
        let bind_group_layouts: Vec<&wgpu::BindGroupLayout> = if pass_has_params {
            vec![&input_bgl, params_bgl.as_ref().unwrap()]
        } else {
            vec![&input_bgl]
        };

        let layout_label = format!("effect_pass{i}_layout");
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&layout_label),
            bind_group_layouts: &bind_group_layouts,
            push_constant_ranges: &[],
        });

        let pipeline_label = format!("effect_pass{i}_pipeline");
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&pipeline_label),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_quad"),
                compilation_options: Default::default(),
                buffers: &[], // Fullscreen triangle — no vertex buffers
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("effect_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None, // Effect apply pass has no stencil
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
        passes,
        input_bind_group_layout: input_bgl,
        params_bind_group_layout: params_bgl,
    })
}

/// Compile the shared composite pipeline (passthrough FS, stencil-aware for parent clipping).
pub(crate) fn compile_composite_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
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

    // Stencil: compare Equal, pass_op Keep (respects parent clipping, no stencil modification)
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
            entry_point: Some("vs_quad"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_composite"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
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
                write_mask: 0x00, // No stencil writes
            },
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    (pipeline, input_bgl)
}

/// Create a bind group to sample a texture (for effect input or composite input).
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
