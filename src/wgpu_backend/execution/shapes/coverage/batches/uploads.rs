use super::{CoverageAllocation, CoverageBatch, CoverageBatchDraw};
use crate::core::cache::FrameCache;
use crate::wgpu_backend::execution::render_pass::RenderPass;
use crate::wgpu_backend::execution::shapes::{ShapeGeometryType, ShapeShaderType, StencilAction};
use crate::wgpu_backend::resources::ShapePipelines;
use crate::wgpu_backend::vertex::{InstanceColor, InstanceMetadata};
use bytemuck::{Pod, Zeroable};
use smallvec::SmallVec;
use std::mem;
use wgpu::{
    vertex_attr_array, BindGroup, BindGroupDescriptor, BindGroupEntry, Buffer, BufferDescriptor,
    BufferUsages, Device, Queue, VertexAttribute, VertexBufferLayout, VertexStepMode,
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(in crate::wgpu_backend) struct CoverageTileInstance {
    tile_index: u32,
    shape_index: u32,
}

impl CoverageTileInstance {
    pub(in crate::wgpu_backend) fn desc() -> VertexBufferLayout<'static> {
        const ATTRIBUTES: [VertexAttribute; 2] = vertex_attr_array![0 => Uint32, 2 => Uint32];
        VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as u64,
            step_mode: VertexStepMode::Instance,
            attributes: &ATTRIBUTES,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CoverageBatchShape {
    color: [f32; 4],
    texture_uv_transform_layer0: [f32; 4],
    texture_uv_transform_layer1: [f32; 4],
    translation: [f32; 2],
    texture_flags: f32,
    padding: f32,
}

fn create_buffer(device: &Device, label: &'static str, size: usize, usage: BufferUsages) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: size as u64,
        usage: usage | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

struct BatchChunk {
    tiles: Vec<CoverageTileInstance>,
    shapes: Vec<CoverageBatchShape>,
    tile_buffer: Buffer,
    shape_buffer: Buffer,
    bindings: FrameCache<BindGroup, BindGroup>,
}

impl BatchChunk {
    fn new(device: &Device, tile_count: usize, shape_count: usize) -> Self {
        let tile_capacity = tile_count.max(4096).next_power_of_two();
        let shape_capacity = shape_count.max(256).next_power_of_two();
        Self {
            tiles: Vec::with_capacity(tile_capacity),
            shapes: Vec::with_capacity(shape_capacity),
            tile_buffer: create_buffer(
                device,
                "coverage_batch_tiles",
                tile_capacity * mem::size_of::<CoverageTileInstance>(),
                BufferUsages::VERTEX,
            ),
            shape_buffer: create_buffer(
                device,
                "coverage_batch_shapes",
                shape_capacity * mem::size_of::<CoverageBatchShape>(),
                BufferUsages::STORAGE,
            ),
            bindings: FrameCache::new(),
        }
    }

    fn fits(&self, tile_count: usize, shape_count: usize) -> bool {
        (self.tiles.len() + tile_count) * mem::size_of::<CoverageTileInstance>()
            <= self.tile_buffer.size() as usize
            && (self.shapes.len() + shape_count) * mem::size_of::<CoverageBatchShape>()
                <= self.shape_buffer.size() as usize
    }

    fn append(
        &mut self,
        draws: &[CoverageBatchDraw],
        colors: &[InstanceColor],
        metadata: &[InstanceMetadata],
    ) -> [u32; 3] {
        let first_shape = self.shapes.len() as u32;
        for draw in draws {
            let instance = draw.instance_index as usize;
            let metadata = metadata[instance];
            self.shapes.push(CoverageBatchShape {
                color: colors[instance].color,
                texture_uv_transform_layer0: bytemuck::cast(metadata.texture_uv_transform_layer0),
                texture_uv_transform_layer1: bytemuck::cast(metadata.texture_uv_transform_layer1),
                translation: metadata.coverage_translation,
                texture_flags: metadata.texture_flags,
                padding: 0.0,
            });
        }
        let start = self.tiles.len() as u32;
        let mut boundary_start = start;
        for boundary in [false, true] {
            for (index, draw) in draws.iter().enumerate() {
                let range = draw.allocation.range;
                let split = range.tile_start + range.solid_tile_count;
                let tiles = if boundary {
                    split..range.tile_start + range.tile_count
                } else {
                    range.tile_start..split
                };
                self.tiles
                    .extend(tiles.map(|tile_index| CoverageTileInstance {
                        tile_index,
                        shape_index: first_shape + index as u32,
                    }));
            }
            if !boundary {
                boundary_start = self.tiles.len() as u32;
            }
        }
        [start, boundary_start, self.tiles.len() as u32]
    }

    fn binding(
        &mut self,
        allocation: &CoverageAllocation,
        device: &Device,
        pipelines: &ShapePipelines,
    ) -> &BindGroup {
        let buffers = allocation.buffers();
        let (binding, _) = self
            .bindings
            .get_or_insert_with(buffers.bind_group.clone(), || {
                let [coverage, tiles, projections] = buffers.storage();
                device.create_bind_group(&BindGroupDescriptor {
                    label: Some("coverage_batch"),
                    layout: &pipelines.layouts.coverage_batch,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: pipelines.and_uniform_buffer.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: coverage.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: tiles.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: projections.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 4,
                            resource: self.shape_buffer.as_entire_binding(),
                        },
                    ],
                })
            });
        binding
    }
}

#[derive(Default)]
pub(in crate::wgpu_backend) struct CoverageBatchUploads {
    chunks: Vec<BatchChunk>,
    current_chunk: usize,
    draw_storage: SmallVec<[CoverageBatchDraw; 4]>,
}

impl CoverageBatchUploads {
    pub(in crate::wgpu_backend) fn reuse_draw_storage(&mut self, batch: &mut CoverageBatch) {
        if self.draw_storage.capacity() > batch.draws.capacity() {
            mem::swap(&mut batch.draws, &mut self.draw_storage);
        }
    }

    pub(in crate::wgpu_backend) fn recycle_draw_storage(&mut self, mut batch: CoverageBatch) {
        batch.draws.clear();
        if batch.draws.capacity() > self.draw_storage.capacity() {
            self.draw_storage = batch.draws;
        }
    }

    pub(in crate::wgpu_backend) fn begin_render(&mut self) {
        self.current_chunk = 0;
        for chunk in &mut self.chunks {
            chunk.tiles.clear();
            chunk.shapes.clear();
        }
    }

    pub(in crate::wgpu_backend) fn draw(
        &mut self,
        pass: &mut RenderPass<'_>,
        batch: &CoverageBatch,
        (colors, metadata): (&[InstanceColor], &[InstanceMetadata]),
        device: &Device,
        pipelines: &ShapePipelines,
        shader: ShapeShaderType,
    ) {
        let draws = &batch.draws;
        let tile_count = batch.tile_count;
        if self
            .chunks
            .get(self.current_chunk)
            .is_some_and(|chunk| !chunk.fits(tile_count, draws.len()))
        {
            self.current_chunk += 1;
        }
        if self.current_chunk == self.chunks.len() {
            self.chunks
                .push(BatchChunk::new(device, tile_count, draws.len()));
        } else if !self.chunks[self.current_chunk].fits(tile_count, draws.len()) {
            self.chunks[self.current_chunk] = BatchChunk::new(device, tile_count, draws.len());
        }
        let chunk = &mut self.chunks[self.current_chunk];
        let [start, boundary_start, end] = chunk.append(draws, colors, metadata);
        let allocation = &draws[0].allocation;
        pass.set_bind_group(0, chunk.binding(allocation, device, pipelines), &[]);
        pass.set_vertex_buffer(0, chunk.tile_buffer.slice(..));
        allocation.buffers().bind(pass);
        for (boundary, instances) in [(false, start..boundary_start), (true, boundary_start..end)] {
            if !instances.is_empty() {
                pass.set_pipeline(pipelines.get(
                    ShapeGeometryType::coverage_batch(boundary),
                    shader,
                    StencilAction::Keep,
                ));
                pass.draw_indexed(0..6, 0, instances);
            }
        }
    }

    pub(in crate::wgpu_backend) fn upload(&mut self, queue: &Queue) {
        for chunk in &mut self.chunks {
            if !chunk.tiles.is_empty() {
                queue.write_buffer(&chunk.tile_buffer, 0, bytemuck::cast_slice(&chunk.tiles));
                queue.write_buffer(&chunk.shape_buffer, 0, bytemuck::cast_slice(&chunk.shapes));
            }
            drop(chunk.bindings.end_frame());
        }
    }
}
