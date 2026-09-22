use super::{ShapeDrawResources, ShapeExecutionResources, TextureSamplingUniform};
use crate::gradient::gpu::{GpuGradientColorParams, GpuMaterialParams};
use crate::renderer::commands::{DrawOperation, DrawPlan, ShapeDrawId};
use crate::renderer::execution::textures::IntermediateTextureResources;
use crate::renderer::execution::uniforms;
use crate::renderer::state::ShapePipelines;
use crate::shape::{ShapeDrawMaterial, ShapeTextureBinding, ShapeTextureLayer};
use std::mem;
use std::ops::Range;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindingResource, Buffer,
    CommandEncoder, Device, Queue, Sampler, Texture, TextureView,
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
    fn prepare_texture_material(
        &mut self,
        has_gradient_fill: bool,
        encoder: &mut CommandEncoder,
        layer: ShapeTextureLayer,
        materials: &mut TextureMaterialPool,
        device: &Device,
        queue: &Queue,
        pipelines: &ShapePipelines,
        textures: &IntermediateTextureResources,
    ) {
        let managed_texture;
        let texture = match layer.texture {
            ShapeTextureBinding::None => unreachable!("under-fill materials have a texture"),
            ShapeTextureBinding::Managed(texture_id) => {
                managed_texture = pipelines
                    .texture_manager
                    .texture(texture_id)
                    .expect("material commands reference registered textures");
                &managed_texture
            }
            ShapeTextureBinding::Intermediate(texture_id) => textures.texture(texture_id),
        };
        let sampling = TextureSamplingUniform::from(layer.sampling);
        let gradient = has_gradient_fill.then(|| {
            self.gradient_material
                .as_ref()
                .expect("gradient material was uploaded")
        });
        let params = GpuMaterialParams::for_texture_sampling(sampling);
        let gradient_view = gradient.map(|material| material.ramp_view.as_ref().clone());
        let slot = materials.next_slot();
        let buffer = uniforms::prepare_buffer(
            &mut slot.params_buffer,
            device,
            queue,
            &params,
            "shape_texture_material_params",
        );
        if let Some(gradient) = gradient {
            encoder.copy_buffer_to_buffer(
                &gradient.params_buffer,
                0,
                buffer,
                0,
                mem::size_of::<GpuGradientColorParams>() as u64,
            );
        }
        if let Some(binding) = slot
            .binding
            .as_ref()
            .filter(|binding| binding.texture == *texture && binding.gradient_view == gradient_view)
        {
            self.texture_material_bind_group = Some(binding.bind_group.clone());
            return;
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
    }

    pub(in crate::renderer) fn material_bind_group(
        &self,
        material: ShapeDrawMaterial,
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

impl ShapeExecutionResources {
    /// Prepares only indexed material draws, without scanning ordinary instructions.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::renderer) fn prepare_texture_materials(
        &mut self,
        encoder: &mut CommandEncoder,
        commands: &DrawPlan,
        material_range: Range<usize>,
        device: &Device,
        queue: &Queue,
        pipelines: &ShapePipelines,
        textures: &IntermediateTextureResources,
    ) {
        for &index in &commands.texture_material_draws[material_range] {
            let (DrawOperation::DrawShape(draw)
            | DrawOperation::DrawShapeAndIncrementStencil(draw)) =
                commands.instructions[index].operation
            else {
                unreachable!("material preparation references shape draws");
            };
            let resources = match draw.id {
                ShapeDrawId::Shape(id) => self.draws.get_mut(&id),
                ShapeDrawId::EffectLeaf(id) => self.effect_leaves.get_mut(&id),
            }
            .expect("material draw was uploaded");
            resources.prepare_texture_material(
                draw.material.has_gradient_fill(),
                encoder,
                draw.material
                    .under_fill_texture
                    .expect("material draw has an under-fill texture"),
                &mut self.texture_materials,
                device,
                queue,
                pipelines,
                textures,
            );
        }
    }
}
