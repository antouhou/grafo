use crate::renderer::MathRect;
use crate::texture_manager::TextureManager;
use crate::util::PoolManager;
use wgpu::{BindGroup, BufferSlice};

pub(crate) struct ImageDrawData {
    pub(crate) texture_id: u64,
    pub(crate) logical_rect: MathRect,
    pub(crate) clip_to_shape: Option<usize>,
    pub(crate) bind_group: Option<BindGroup>,
    pub(crate) vertex_buffer: Option<wgpu::Buffer>,
    pub(crate) index_buffer: Option<wgpu::Buffer>,
    pub(crate) num_indices: Option<u32>,
}

impl ImageDrawData {
    pub fn new(
        texture_id: u64,
        rect: [(f32, f32); 2],
        clip_to_shape: Option<usize>,
    ) -> Self {
        Self {
            texture_id,
            logical_rect: MathRect::new(
                (rect[0].0, rect[0].1).into(),
                (rect[1].0, rect[1].1).into(),
            ),
            clip_to_shape,
            bind_group: None,
            vertex_buffer: None,
            index_buffer: None,
            num_indices: None,
        }
    }

    pub(crate) fn prepare(
        &mut self,
        texture_manager: &TextureManager,
        // device: &wgpu::Device,
        // queue: &wgpu::Queue,
        // bind_group_layout: &BindGroupLayout,
        canvas_physical_size: (u32, u32),
        scale_factor: f32,
        buffers_pool: &mut PoolManager,
    ) {
        let (vertex_buffer, index_buffer, bind_group) = texture_manager
            .create_everything_to_render_texture(
                self.texture_id,
                canvas_physical_size,
                &self.logical_rect,
                scale_factor,
                buffers_pool,
            )
            .unwrap();

        self.bind_group = Some(bind_group);
        self.vertex_buffer = Some(vertex_buffer);
        self.index_buffer = Some(index_buffer);
        self.num_indices = Some(6);
    }

    pub fn return_buffers_to_pool(&mut self, buffers_pool: &mut PoolManager) {
        if let Some(vertex_buffer) = self.vertex_buffer.take() {
            buffers_pool
                .image_buffers_pool
                .return_vertex_buffer(vertex_buffer);
        }

        if let Some(index_buffer) = self.index_buffer.take() {
            buffers_pool
                .image_buffers_pool
                .return_index_buffer(index_buffer);
        }
    }

    pub(crate) fn vertex_buffer(&self) -> BufferSlice<'_> {
        self.vertex_buffer
            .as_ref()
            .expect("Image buffers to be prepared")
            .slice(..)
    }

    pub(crate) fn index_buffer(&self) -> BufferSlice<'_> {
        self.index_buffer
            .as_ref()
            .expect("Image buffers to be prepared")
            .slice(..)
    }

    pub(crate) fn bind_group(&self) -> &BindGroup {
        self.bind_group
            .as_ref()
            .expect("Image buffers to be prepared")
    }

    pub(crate) fn num_indices(&self) -> u32 {
        self.num_indices.expect("Image buffers to be prepared")
    }
}
