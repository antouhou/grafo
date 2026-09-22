use super::{ShapeDrawResources, TextureSamplingUniform};
use crate::gradient::gpu::GpuMaterialParams;
use crate::gradient::types::Fill;
use crate::renderer::execution::textures::IntermediateTextureResources;
use crate::renderer::execution::uniforms;
use crate::renderer::state::ShapePipelines;
use crate::shape::{ShapeDrawMaterial, ShapeTextureBinding, ShapeTextureLayer};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindingResource, Buffer,
    Device, Queue, Sampler, Texture, TextureView,
};

struct TextureMaterialBinding {
    texture: Texture,
    gradient_view: Option<TextureView>,
    bind_group: BindGroup,
}

#[derive(Default)]
struct TextureMaterialSlot {
    params_buffer: Option<Buffer>,
    binding: Option<TextureMaterialBinding>,
}

/// GPU material slots survive queue clearing. Each use before submission gets its own buffer.
#[derive(Default)]
pub(in crate::renderer) struct TextureMaterialPool {
    slots: Vec<TextureMaterialSlot>,
    used: usize,
}

impl TextureMaterialPool {
    pub(in crate::renderer) fn begin_render(&mut self) {
        self.used = 0;
    }

    fn next_slot(&mut self) -> &mut TextureMaterialSlot {
        if self.used == self.slots.len() {
            self.slots.push(TextureMaterialSlot::default());
        }
        let slot = &mut self.slots[self.used];
        self.used += 1;
        slot
    }

    /// Release surplus resources after submission while retaining the current working set.
    pub(in crate::renderer) fn finish_render(&mut self) {
        self.slots.truncate(self.used);
    }

    pub(in crate::renderer) fn invalidate_bindings(&mut self) {
        for slot in &mut self.slots {
            slot.binding = None;
        }
    }
}

fn create_material_binding(
    device: &Device,
    layout: &BindGroupLayout,
    params_buffer: &Buffer,
    sampler: &Sampler,
    texture_view: &TextureView,
    gradient_view: Option<&TextureView>,
) -> BindGroup {
    let shared_entries = [
        BindGroupEntry {
            binding: 0,
            resource: params_buffer.as_entire_binding(),
        },
        BindGroupEntry {
            binding: 3,
            resource: BindingResource::TextureView(texture_view),
        },
        BindGroupEntry {
            binding: 4,
            resource: BindingResource::Sampler(sampler),
        },
    ];
    let gradient_entries;
    let entries = if let Some(gradient_view) = gradient_view {
        gradient_entries = [
            shared_entries[0].clone(),
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(gradient_view),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Sampler(sampler),
            },
            shared_entries[1].clone(),
            shared_entries[2].clone(),
        ];
        &gradient_entries[..]
    } else {
        &shared_entries[..]
    };
    device.create_bind_group(&BindGroupDescriptor {
        label: Some("shape_texture_material"),
        layout,
        entries,
    })
}

impl ShapeDrawResources {
    /// Draws reference reusable bindings whose allocation lifetime survives queue clearing.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::renderer) fn prepare_texture_material(
        &mut self,
        fill: Option<&Fill>,
        layer: ShapeTextureLayer,
        materials: &mut TextureMaterialPool,
        device: &Device,
        queue: &Queue,
        pipelines: &ShapePipelines,
        textures: &IntermediateTextureResources,
    ) -> Option<ShapeTextureLayer> {
        let managed_texture;
        let texture = match layer.texture {
            ShapeTextureBinding::None => return None,
            ShapeTextureBinding::Managed(texture_id) => {
                managed_texture = pipelines.texture_manager.texture(texture_id)?;
                &managed_texture
            }
            ShapeTextureBinding::Intermediate(texture_id) => textures.texture(texture_id),
        };
        let sampling = TextureSamplingUniform::from(layer.sampling);
        let (params, gradient_view) = match fill {
            Some(Fill::Gradient(gradient)) => (
                GpuMaterialParams::from_gradient_data(&gradient.data)
                    .with_texture_sampling(sampling),
                Some(
                    self.gradient_material
                        .as_ref()
                        .expect("gradient fills are prepared while queuing shapes")
                        .ramp_view
                        .as_ref()
                        .clone(),
                ),
            ),
            _ => (GpuMaterialParams::for_texture_sampling(sampling), None),
        };
        let slot = materials.next_slot();
        let buffer = uniforms::prepare_buffer(
            &mut slot.params_buffer,
            device,
            queue,
            &params,
            "shape_texture_material_params",
        );
        if let Some(binding) = slot
            .binding
            .as_ref()
            .filter(|binding| binding.texture == *texture && binding.gradient_view == gradient_view)
        {
            self.texture_material_bind_group = Some(binding.bind_group.clone());
            return Some(layer);
        }

        let layouts = pipelines
            .under_fill_pipelines
            .as_ref()
            .expect("texture material pipelines are initialized before preparation");
        let layout = if gradient_view.is_some() {
            &layouts.gradient_layout
        } else {
            &layouts.solid_layout
        };
        let view = texture.create_view(&Default::default());
        let binding = create_material_binding(
            device,
            layout,
            buffer,
            &pipelines.linear_clamp_sampler,
            &view,
            gradient_view.as_ref(),
        );
        self.texture_material_bind_group = Some(binding.clone());
        slot.binding = Some(TextureMaterialBinding {
            texture: texture.clone(),
            gradient_view,
            bind_group: binding,
        });
        Some(layer)
    }

    pub(in crate::renderer) fn material_bind_group(
        &self,
        material: ShapeDrawMaterial<'_>,
    ) -> Option<&BindGroup> {
        if material.under_fill_texture.is_some() {
            Some(
                self.texture_material_bind_group
                    .as_ref()
                    .expect("under-fill textures have a prepared material binding"),
            )
        } else if material.has_gradient_fill() {
            Some(
                &self
                    .gradient_material
                    .as_deref()
                    .expect("gradient fills have a prepared material binding")
                    .bind_group,
            )
        } else {
            None
        }
    }
}
