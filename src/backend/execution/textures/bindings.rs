use super::IntermediateTextureResources;
use crate::backend::texture_manager::WgpuTextureManager;
use crate::backend::types::BoundTextureState;
use crate::commands::ShapeTextureBinding;
use crate::render_backend::TextureManager;
use std::sync::Arc;
use wgpu::{BindGroup, BindGroupLayout, RenderPass};

impl IntermediateTextureResources {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn bind_shape_texture_layers(
        &self,
        render_pass: &mut RenderPass<'_>,
        texture_bindings: &[ShapeTextureBinding; 2],
        texture_manager: &WgpuTextureManager,
        shape_texture_bind_group_layout_background: &BindGroupLayout,
        shape_texture_bind_group_layout_foreground: &BindGroupLayout,
        default_shape_texture_bind_groups: &[Arc<BindGroup>; 2],
        bound_texture_state: &mut BoundTextureState,
    ) {
        for (layer, texture_binding) in texture_bindings.iter().enumerate() {
            let effective_binding = match texture_binding {
                ShapeTextureBinding::Managed(texture_id)
                    if !texture_manager.is_texture_loaded(*texture_id) =>
                {
                    ShapeTextureBinding::None
                }
                texture_binding => *texture_binding,
            };
            if !bound_texture_state.needs_rebind(layer, &effective_binding) {
                continue;
            }
            match &effective_binding {
                ShapeTextureBinding::Managed(texture_id) => {
                    match texture_manager.get_or_create_shape_bind_group(
                        if layer == 0 {
                            shape_texture_bind_group_layout_background
                        } else {
                            shape_texture_bind_group_layout_foreground
                        },
                        *texture_id,
                    ) {
                        Ok(bind_group) => {
                            render_pass.set_bind_group(1 + layer as u32, &*bind_group, &[]);
                        }
                        Err(_) => {
                            render_pass.set_bind_group(
                                1 + layer as u32,
                                &*default_shape_texture_bind_groups[layer],
                                &[],
                            );
                            bound_texture_state.mark_bound(layer, ShapeTextureBinding::None);
                            continue;
                        }
                    }
                }
                ShapeTextureBinding::Intermediate(texture_id) => {
                    render_pass.set_bind_group(1 + layer as u32, self.bind_group(*texture_id), &[]);
                }
                ShapeTextureBinding::None => {
                    render_pass.set_bind_group(
                        1 + layer as u32,
                        &*default_shape_texture_bind_groups[layer],
                        &[],
                    );
                }
            }
            bound_texture_state.mark_bound(layer, effective_binding);
        }
    }
}
