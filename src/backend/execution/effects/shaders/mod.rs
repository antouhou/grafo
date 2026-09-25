use super::bindings::create_effect_params_bind_group_layout;
use crate::backend::errors::{EffectResourceError, EffectShaderError};
use naga::front::wgsl;
use naga::valid::Validator;
use naga::{AddressSpace, ShaderStage};
use wgpu::{
    BindGroupLayout, BlendState, ColorTargetState, ColorWrites, Device, FragmentState,
    MultisampleState, PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, RenderPipeline,
    RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderSource, TextureFormat, VertexState,
};
/// Draws a fullscreen triangle from three vertex indices, without a vertex buffer.
/// Effect and composite passes share this shader.
pub(crate) const FULLSCREEN_TRIANGLE_VS: &str =
    include_str!("../../../../shaders/fullscreen_triangle_vs.wgsl");

/// Input texture bindings prepended to the user's effect fragment shader.
pub(crate) const EFFECT_FS_PREAMBLE: &str =
    include_str!("../../../../shaders/effect_fs_preamble.wgsl");

/// A single compiled pass within a multi-pass effect.
pub(crate) struct LoadedEffectPass {
    /// The compiled render pipeline for this pass's fullscreen triangle.
    pub pipeline: RenderPipeline,
    /// Whether this pass references user parameters at `@group(1)`.
    pub has_params: bool,
}

/// Compiled effect passes cached by `effect_id` and shared across nodes.
pub(crate) struct LoadedEffect {
    /// Exact pass sources used to compile this effect.
    pub pass_sources: Box<[Box<str>]>,
    /// Compiled passes, executed sequentially with ping-pong textures.
    pub passes: Vec<LoadedEffectPass>,
    /// The user's parameter uniform layout at group 1.
    /// None if no pass uses user params. Shared across all passes that reference it.
    pub params_bind_group_layout: Option<BindGroupLayout>,
}

/// Combines the built-in vertex shader, input bindings, and user fragment shader into one module.
pub(crate) fn build_effect_wgsl(user_fragment_source: &str) -> String {
    format!("{FULLSCREEN_TRIANGLE_VS}\n{EFFECT_FS_PREAMBLE}\n{user_fragment_source}")
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
/// All registered effects use the same input layout at group 0.
/// Passes that use user parameters share the uniform layout at group 1.
///
/// For single-pass effects, pass a one-element slice.
pub(crate) fn compile_effect_pipeline(
    device: &Device,
    pass_sources: &[&str],
    format: TextureFormat,
    input_bind_group_layout: &BindGroupLayout,
    validator: &mut Validator,
) -> Result<LoadedEffect, EffectResourceError> {
    if pass_sources.is_empty() {
        return Err(EffectResourceError::InvalidParams(
            "At least one effect pass is required".into(),
        ));
    }

    let validated_passes = pass_sources
        .iter()
        .enumerate()
        .map(|(pass_index, fragment_source)| {
            let source = build_effect_wgsl(fragment_source);
            let has_params = validate_effect_shader(&source, validator)
                .map_err(|reason| EffectResourceError::InvalidShader { pass_index, reason })?;
            Ok((source, has_params))
        })
        .collect::<Result<Vec<_>, EffectResourceError>>()?;

    // Passes using group 1 share the parameter layout.
    let any_has_params = validated_passes.iter().any(|(_, has_params)| *has_params);
    let params_bind_group_layout = if any_has_params {
        Some(create_effect_params_bind_group_layout(device))
    } else {
        None
    };

    let mut passes = Vec::with_capacity(pass_sources.len());

    for (pass_index, (full_wgsl, pass_has_params)) in validated_passes.into_iter().enumerate() {
        let shader_label = format!("effect_pass{pass_index}_shader");
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some(&shader_label),
            source: ShaderSource::Wgsl(full_wgsl.into()),
        });

        // Include group 1 only for passes that reference it.
        let layout_label = format!("effect_pass{pass_index}_layout");
        let pipeline_layout = if pass_has_params {
            let bind_group_layouts = [
                input_bind_group_layout,
                params_bind_group_layout.as_ref().unwrap(),
            ];
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some(&layout_label),
                bind_group_layouts: &bind_group_layouts,
                push_constant_ranges: &[],
            })
        } else {
            let bind_group_layouts = [input_bind_group_layout];
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some(&layout_label),
                bind_group_layouts: &bind_group_layouts,
                push_constant_ranges: &[],
            })
        };

        let pipeline_label = format!("effect_pass{pass_index}_pipeline");
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some(&pipeline_label),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_triangle"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("effect_main"),
                compilation_options: Default::default(),
                targets: &[Some(ColorTargetState {
                    format,
                    blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
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
        params_bind_group_layout,
    })
}

#[cfg(test)]
mod tests;
