use super::IntermediateTextureResources;
use crate::commands::ShapeTextureBinding;
use crate::render_backend::texture_manager::TextureManager;
use crate::wgpu_backend::execution::render_pass::RenderPass;
use crate::wgpu_backend::texture_manager::WgpuTextureManager;
use std::sync::Arc;
use wgpu::{BindGroup, BindGroupLayout};

impl IntermediateTextureResources {
    pub(crate) fn bind_shape_texture_layers(
        &self,
        render_pass: &mut RenderPass<'_>,
        texture_bindings: &[ShapeTextureBinding; 2],
        texture_manager: &WgpuTextureManager,
        shape_texture_bind_group_layout_background: &BindGroupLayout,
        shape_texture_bind_group_layout_foreground: &BindGroupLayout,
        default_shape_texture_bind_groups: &[Arc<BindGroup>; 2],
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
            if render_pass.texture_is_bound(layer, effective_binding) {
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
                            render_pass.set_texture_bind_group(
                                layer,
                                effective_binding,
                                &bind_group,
                            );
                        }
                        Err(_) => {
                            render_pass.set_texture_bind_group(
                                layer,
                                ShapeTextureBinding::None,
                                &default_shape_texture_bind_groups[layer],
                            );
                            continue;
                        }
                    }
                }
                ShapeTextureBinding::Intermediate(texture_id) => {
                    render_pass.set_texture_bind_group(
                        layer,
                        effective_binding,
                        self.bind_group(*texture_id),
                    );
                }
                ShapeTextureBinding::None => {
                    render_pass.set_texture_bind_group(
                        layer,
                        effective_binding,
                        &default_shape_texture_bind_groups[layer],
                    );
                }
            }
        }
    }
}
