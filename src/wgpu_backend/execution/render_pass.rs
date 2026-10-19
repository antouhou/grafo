use crate::commands::ShapeTextureBinding;
use crate::core::UnsignedPhysicalRect;
#[cfg(feature = "render_metrics")]
use crate::wgpu_backend::metrics::PipelineSwitchCounts;
use smallvec::SmallVec;
use std::ops::Range;
use wgpu::{
    BindGroup, Buffer, BufferBinding, BufferSize, BufferSlice, IndexFormat,
    RenderPass as WgpuRenderPass, RenderPipeline,
};

struct BufferBindingState {
    buffer: Buffer,
    offset: u64,
    size: Option<BufferSize>,
}

impl BufferBindingState {
    fn new(binding: &BufferBinding<'_>) -> Self {
        Self {
            buffer: binding.buffer.clone(),
            offset: binding.offset,
            size: binding.size,
        }
    }

    fn matches(&self, binding: &BufferBinding<'_>) -> bool {
        self.buffer == *binding.buffer && self.offset == binding.offset && self.size == binding.size
    }
}

struct BindGroupState {
    group: BindGroup,
    offsets: SmallVec<[u32; 4]>,
}

/// Owns bindings until the underlying render pass ends.
pub(in crate::wgpu_backend) struct RenderPass<'encoder> {
    pass: WgpuRenderPass<'encoder>,
    pipeline: Option<RenderPipeline>,
    bind_groups: [Option<BindGroupState>; 4],
    vertex_buffers: [Option<BufferBindingState>; 4],
    index_buffer: Option<(BufferBindingState, IndexFormat)>,
    texture_sources: [Option<ShapeTextureBinding>; 2],
    scissor: Option<UnsignedPhysicalRect>,
    stencil_reference: Option<u32>,
    #[cfg(feature = "render_metrics")]
    switches: PipelineSwitchCounts,
}

impl<'encoder> RenderPass<'encoder> {
    pub fn new(pass: WgpuRenderPass<'encoder>) -> Self {
        Self {
            pass,
            pipeline: None,
            bind_groups: Default::default(),
            vertex_buffers: Default::default(),
            index_buffer: None,
            texture_sources: [None; 2],
            scissor: None,
            stencil_reference: None,
            #[cfg(feature = "render_metrics")]
            switches: PipelineSwitchCounts::default(),
        }
    }

    pub fn set_pipeline(&mut self, pipeline: &RenderPipeline) {
        if self.pipeline.as_ref() == Some(pipeline) {
            return;
        }
        self.pass.set_pipeline(pipeline);
        self.pipeline = Some(pipeline.clone());
        #[cfg(feature = "render_metrics")]
        {
            self.switches.total_switches += 1;
        }
    }

    pub fn set_bind_group(&mut self, index: u32, group: &BindGroup, offsets: &[u32]) {
        let binding = &mut self.bind_groups[index as usize];
        if binding
            .as_ref()
            .is_some_and(|binding| binding.group == *group && binding.offsets.as_slice() == offsets)
        {
            return;
        }
        self.pass.set_bind_group(index, group, offsets);
        match binding {
            Some(binding) => {
                binding.group = group.clone();
                binding.offsets.clear();
                binding.offsets.extend_from_slice(offsets);
            }
            None => {
                *binding = Some(BindGroupState {
                    group: group.clone(),
                    offsets: SmallVec::from_slice(offsets),
                });
            }
        }
        if matches!(index, 1 | 2) {
            self.texture_sources[index as usize - 1] = None;
        }
    }

    pub fn texture_is_bound(&self, layer: usize, source: ShapeTextureBinding) -> bool {
        self.texture_sources[layer] == Some(source)
    }

    pub fn set_texture_bind_group(
        &mut self,
        layer: usize,
        source: ShapeTextureBinding,
        group: &BindGroup,
    ) {
        self.set_bind_group(layer as u32 + 1, group, &[]);
        self.texture_sources[layer] = Some(source);
    }

    pub fn set_vertex_buffer(&mut self, slot: u32, slice: BufferSlice<'_>) {
        let binding = BufferBinding::from(slice);
        let current = &mut self.vertex_buffers[slot as usize];
        if current
            .as_ref()
            .is_some_and(|current| current.matches(&binding))
        {
            return;
        }
        self.pass.set_vertex_buffer(slot, slice);
        *current = Some(BufferBindingState::new(&binding));
    }

    pub fn set_index_buffer(&mut self, slice: BufferSlice<'_>, format: IndexFormat) {
        let binding = BufferBinding::from(slice);
        if self
            .index_buffer
            .as_ref()
            .is_some_and(|(current, current_format)| {
                current.matches(&binding) && *current_format == format
            })
        {
            return;
        }
        self.pass.set_index_buffer(slice, format);
        self.index_buffer = Some((BufferBindingState::new(&binding), format));
    }

    pub fn set_scissor(&mut self, scissor: UnsignedPhysicalRect) {
        if self.scissor == Some(scissor) {
            return;
        }
        self.pass.set_scissor_rect(
            scissor.min.x,
            scissor.min.y,
            scissor.width(),
            scissor.height(),
        );
        self.scissor = Some(scissor);
    }

    pub fn set_stencil_reference(&mut self, reference: u32) {
        if self.stencil_reference == Some(reference) {
            return;
        }
        self.pass.set_stencil_reference(reference);
        self.stencil_reference = Some(reference);
    }

    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        self.pass.draw(vertices, instances);
    }

    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        self.pass.draw_indexed(indices, base_vertex, instances);
    }

    #[cfg(feature = "render_metrics")]
    pub fn pipeline_switch_counts(&self) -> PipelineSwitchCounts {
        self.switches
    }
}
