use std::sync::Arc;

use ahash::HashMap;

use crate::effect::{EffectInstance, LoadedEffect};
use crate::shape::{CachedShapeDrawData, DrawShapeCommand, ShapeDrawData};
use crate::texture_manager::TextureManager;
use crate::vertex::InstanceTransform;

#[derive(Debug)]
pub(super) enum DrawCommand {
    Shape(ShapeDrawData),
    CachedShape(CachedShapeDrawData),
}

impl DrawCommand {
    pub(super) fn set_transform(&mut self, transform: InstanceTransform) {
        match self {
            DrawCommand::Shape(shape) => shape.set_transform(transform),
            DrawCommand::CachedShape(cached_shape) => cached_shape.set_transform(transform),
        }
    }

    pub(super) fn set_texture_id(&mut self, layer: usize, texture_id: Option<u64>) {
        match self {
            DrawCommand::Shape(shape) => shape.set_texture_id(layer, texture_id),
            DrawCommand::CachedShape(cached_shape) => {
                cached_shape.set_texture_id(layer, texture_id)
            }
        }
    }

    pub(super) fn set_instance_color_override(&mut self, color: Option<[f32; 4]>) {
        match self {
            DrawCommand::Shape(shape) => shape.set_instance_color_override(color),
            DrawCommand::CachedShape(cached_shape) => {
                cached_shape.set_instance_color_override(color)
            }
        }
    }

    pub(super) fn clear_frame_state(&mut self) {
        match self {
            DrawCommand::Shape(shape) => {
                shape.index_buffer_range = None;
                shape.stencil_ref = None;
            }
            DrawCommand::CachedShape(cached_shape) => {
                cached_shape.index_buffer_range = None;
                cached_shape.stencil_ref = None;
            }
        }
    }
}

#[derive(Debug)]
pub(super) enum TraversalEvent {
    Pre(usize),
    Post(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Pipeline {
    None,
    StencilIncrement,
    StencilDecrement,
}

pub(super) struct Buffers<'a> {
    pub(super) aggregated_vertex_buffer: &'a wgpu::Buffer,
    pub(super) aggregated_index_buffer: &'a wgpu::Buffer,
    pub(super) identity_instance_transform_buffer: &'a wgpu::Buffer,
    pub(super) identity_instance_color_buffer: &'a wgpu::Buffer,
    pub(super) identity_instance_metadata_buffer: &'a wgpu::Buffer,
    pub(super) aggregated_instance_transform_buffer: Option<&'a wgpu::Buffer>,
    pub(super) aggregated_instance_color_buffer: Option<&'a wgpu::Buffer>,
    pub(super) aggregated_instance_metadata_buffer: Option<&'a wgpu::Buffer>,
}

pub(super) struct Pipelines<'a> {
    pub(super) and_pipeline: &'a wgpu::RenderPipeline,
    pub(super) and_bind_group: &'a wgpu::BindGroup,
    pub(super) decrementing_pipeline: &'a wgpu::RenderPipeline,
    pub(super) decrementing_bind_group: &'a wgpu::BindGroup,
    pub(super) shape_texture_bind_group_layout_background: &'a wgpu::BindGroupLayout,
    pub(super) shape_texture_bind_group_layout_foreground: &'a wgpu::BindGroupLayout,
    pub(super) default_shape_texture_bind_groups: &'a [Arc<wgpu::BindGroup>; 2],
    pub(super) shape_texture_layout_epoch: u64,
    pub(super) texture_manager: &'a TextureManager,
}

pub(super) struct BackdropContext<'a> {
    pub(super) backdrop_effects: &'a HashMap<usize, EffectInstance>,
    pub(super) loaded_effects: &'a HashMap<u64, LoadedEffect>,
    pub(super) composite_pipeline: &'a wgpu::RenderPipeline,
    pub(super) composite_bgl: &'a wgpu::BindGroupLayout,
    pub(super) effect_sampler: &'a wgpu::Sampler,
    pub(super) stencil_only_pipeline: &'a wgpu::RenderPipeline,
    pub(super) backdrop_color_pipeline: &'a wgpu::RenderPipeline,
    pub(super) and_bind_group: &'a wgpu::BindGroup,
    pub(super) default_shape_texture_bind_groups: &'a [Arc<wgpu::BindGroup>; 2],
    pub(super) device: &'a wgpu::Device,
    pub(super) physical_size: (u32, u32),
    pub(super) config_format: wgpu::TextureFormat,
    pub(super) backdrop_snapshot_texture: &'a wgpu::Texture,
    pub(super) backdrop_snapshot_view: &'a wgpu::TextureView,
    pub(super) texture_manager: &'a TextureManager,
    pub(super) shape_texture_bind_group_layout_background: &'a wgpu::BindGroupLayout,
    pub(super) shape_texture_bind_group_layout_foreground: &'a wgpu::BindGroupLayout,
    pub(super) shape_texture_layout_epoch: u64,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct BufferSizingDecision {
    pub(super) should_reallocate: bool,
}

pub(super) fn decide_buffer_sizing(
    existing_size: Option<u64>,
    required_size: usize,
) -> BufferSizingDecision {
    let required_size = required_size as u64;
    let should_reallocate = existing_size
        .map(|size| size < required_size)
        .unwrap_or(true);

    BufferSizingDecision { should_reallocate }
}

#[cfg(test)]
mod tests {
    use super::decide_buffer_sizing;

    #[test]
    fn decide_buffer_sizing_reallocates_when_missing() {
        let decision = decide_buffer_sizing(None, 128);
        assert!(decision.should_reallocate);
    }

    #[test]
    fn decide_buffer_sizing_reallocates_when_too_small() {
        let decision = decide_buffer_sizing(Some(64), 128);
        assert!(decision.should_reallocate);
    }

    #[test]
    fn decide_buffer_sizing_keeps_buffer_when_large_enough() {
        let decision = decide_buffer_sizing(Some(512), 128);
        assert!(!decision.should_reallocate);
    }
}
