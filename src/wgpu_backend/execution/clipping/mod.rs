use super::effects::PooledTexture;
use super::instructions::{ExecutionContext, ExecutionResources};
use super::shapes::coverage::clipping::ClipCoverage;
use super::shapes::coverage::CoverageAllocation;
use super::shapes::ShapeExecutionResources;
use crate::commands::{IntermediateTextureId, ShapeDrawId, Target};
use crate::core::{PhysicalRect, UnsignedPhysicalRect};
pub(super) use backdrop_bases::ClipBackdropSources;
pub(super) use filtering::ClipBackdropComposition;
pub(super) use instructions::{is_empty_clip_paint, prepare_clip};
use pipelines::ClipPipelines;
use std::{cell::Cell, sync::Arc};
use wgpu::{BindGroup, Texture, TextureView};

mod backdrop_bases;
pub(in crate::wgpu_backend) mod bindings;
mod captures;
mod copies;
mod filtering;
mod instructions;
pub(in crate::wgpu_backend) mod pipelines;
mod stencils;

impl ExecutionContext<'_> {
    fn clip_pipelines(&self) -> &ClipPipelines {
        self.pipelines
            .clipping(self.device, self.format, self.sample_count)
    }
}

pub(super) struct ClipCapture {
    pub texture: Texture,
    pub view: TextureView,
    pub samples: TextureView,
    pub bounds: PhysicalRect,
}

pub(super) struct ClipBackdropFilter {
    pub input: IntermediateTextureId,
    pub bounds: PhysicalRect,
    pub coverage: Arc<CoverageAllocation>,
    pub working: TextureView,
}

#[derive(Clone)]
pub(super) struct ClipBackdropBase {
    pub conditional_background: usize,
    pub bounds: PhysicalRect,
    pub coverage: Arc<ClipCoverage>,
}

#[derive(Clone)]
pub(super) struct ClipStencilMask {
    pub view: TextureView,
    pub bounds: PhysicalRect,
}

pub(super) struct ClipLayerBackdrop {
    pub backdrop_background: Option<PooledTexture>,
    pub capture_coverage: Arc<CoverageAllocation>,
    pub capture_binding: BindGroup,
    pub capture_images: BindGroup,
    pub is_capture_dirty: Cell<bool>,
    pub background: PooledTexture,
}

pub(super) struct ClipLayer {
    pub stencil_mask: Option<ClipStencilMask>,
    pub backdrop: Option<ClipLayerBackdrop>,
    pub needs_clear: bool,
    pub working: PooledTexture,
    pub bounds: PhysicalRect,
    pub uniform_index: usize,
    pub closing_coverage: Arc<CoverageAllocation>,
    pub closing_binding: BindGroup,
    pub images: BindGroup,
}

pub(super) struct ClipScope {
    pub uses_coverage: bool,
    pub coverage: Option<Arc<ClipCoverage>>,
    pub layer: Option<ClipLayer>,
    pub previous_stencil: u32,
    pub previous_layer: Option<usize>,
}

pub(super) fn prepare_draw(
    shapes: &mut ShapeExecutionResources,
    context: &ExecutionContext<'_>,
    id: ShapeDrawId,
    parent: &Arc<ClipCoverage>,
    scissor: UnsignedPhysicalRect,
    clips_directly: bool,
) -> Option<Arc<CoverageAllocation>> {
    let source = shapes.draws[&id.0].path_coverage.as_ref()?;
    let bounds = source
        .allocation
        .bounds
        .to_i32()
        .translate([source.translation[0] as i32, source.translation[1] as i32].into())
        .intersection(&scissor.to_i32())
        .unwrap_or_else(PhysicalRect::zero);
    if parent.covers_rect(bounds) {
        return None;
    }
    let cache = shapes
        .clip_coverage
        .as_deref_mut()
        .expect("clip coverage is initialized before drawing children");
    if !parent.bounds.intersects(&bounds) {
        let empty = cache.rectangle([0.0; 4], UnsignedPhysicalRect::zero());
        return Some(cache.prepare(
            &empty,
            None,
            None,
            context.device,
            context.queue,
            &context.pipelines.shapes,
        ));
    }
    let coverage = cache.shape(source, context.viewport);
    let combined = cache.intersection(parent, &coverage);
    if clips_directly && Arc::ptr_eq(&combined, &coverage) {
        return None;
    }
    Some(cache.prepare(
        &combined,
        (!clips_directly).then_some(parent),
        Some(source),
        context.device,
        context.queue,
        &context.pipelines.shapes,
    ))
}

impl ExecutionResources<'_> {
    pub(in crate::wgpu_backend::execution) fn inherited_clip_coverage(
        &mut self,
        target: Target,
    ) -> Option<Arc<ClipCoverage>> {
        if matches!(
            target,
            Target::Mask(_)
                | Target::Texture {
                    is_content_effect_input: true,
                    ..
                }
        ) {
            return None;
        }
        let parent = self
            .textures
            .active_targets
            .last()?
            .current_coverage()?
            .clone();
        Some(self.shapes.clip_coverage().for_offscreen_target(
            &parent,
            UnsignedPhysicalRect::from_size(self.context.viewport.physical_size.into()),
        ))
    }
}
