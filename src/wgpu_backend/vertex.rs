use crate::core::vertex::{CustomVertex, InstanceTransform, TextureUvTransform};
use bytemuck::{Pod, Zeroable};
use std::{mem, ops::Range};
use wgpu::{BufferAddress, VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode};

/// Locations of one geometry's vertices and local indices in the shared buffers.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct GeometryBufferRange {
    pub(crate) index_start: u32,
    pub(crate) index_count: u32,
    pub(crate) vertex_start: i32,
}

impl GeometryBufferRange {
    pub(crate) fn indices(self) -> Range<u32> {
        self.index_start..self.index_start + self.index_count
    }
}

impl CustomVertex {
    pub(crate) const STRIDE: BufferAddress = mem::size_of::<Self>() as BufferAddress;

    const ATTRIBUTES: [VertexAttribute; 4] = [
        // Position
        VertexAttribute {
            format: VertexFormat::Float32x2,
            offset: 0,
            shader_location: 0,
        },
        // Tex Coords
        VertexAttribute {
            format: VertexFormat::Float32x2,
            offset: mem::size_of::<[f32; 2]>() as BufferAddress,
            shader_location: 2,
        },
        // Outward model-space normal for the AA fringe.
        VertexAttribute {
            format: VertexFormat::Float32x2,
            offset: (mem::size_of::<[f32; 2]>() * 2) as BufferAddress,
            shader_location: 8,
        },
        // AA coverage, from 1.0 at the interior to 0.0 at the outer fringe.
        VertexAttribute {
            format: VertexFormat::Float32,
            offset: (mem::size_of::<[f32; 2]>() * 3) as BufferAddress,
            shader_location: 9,
        },
    ];

    pub fn desc<'a>() -> VertexBufferLayout<'a> {
        VertexBufferLayout {
            array_stride: Self::STRIDE,
            step_mode: VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

/// Per-instance color payload
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct InstanceColor {
    pub color: [f32; 4],
}

impl InstanceColor {
    pub(crate) const STRIDE: BufferAddress = mem::size_of::<Self>() as BufferAddress;

    const ATTRIBUTES: [VertexAttribute; 1] = [VertexAttribute {
        format: VertexFormat::Float32x4,
        offset: 0,
        shader_location: 1,
    }];

    pub fn desc<'a>() -> VertexBufferLayout<'a> {
        VertexBufferLayout {
            array_stride: Self::STRIDE,
            step_mode: VertexStepMode::Instance,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

impl InstanceTransform {
    pub(crate) const STRIDE: BufferAddress = mem::size_of::<Self>() as BufferAddress;

    const ATTRIBUTES: [VertexAttribute; 4] = [
        VertexAttribute {
            format: VertexFormat::Float32x4,
            offset: 0,
            shader_location: 3,
        },
        VertexAttribute {
            format: VertexFormat::Float32x4,
            offset: 16,
            shader_location: 4,
        },
        VertexAttribute {
            format: VertexFormat::Float32x4,
            offset: 32,
            shader_location: 5,
        },
        VertexAttribute {
            format: VertexFormat::Float32x4,
            offset: 48,
            shader_location: 6,
        },
    ];

    pub fn desc<'a>() -> VertexBufferLayout<'a> {
        VertexBufferLayout {
            array_stride: Self::STRIDE,
            step_mode: VertexStepMode::Instance,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct InstanceMetadata {
    pub draw_order: f32,
    pub texture_flags: f32,
    pub texture_uv_transform_layer0: TextureUvTransform,
    pub texture_uv_transform_layer1: TextureUvTransform,
}

impl Default for InstanceMetadata {
    fn default() -> Self {
        Self {
            draw_order: 0.0,
            texture_flags: 0.0,
            texture_uv_transform_layer0: TextureUvTransform::IDENTITY,
            texture_uv_transform_layer1: TextureUvTransform::IDENTITY,
        }
    }
}

impl InstanceMetadata {
    pub(crate) const STRIDE: BufferAddress = mem::size_of::<Self>() as BufferAddress;

    const ATTRIBUTES: [VertexAttribute; 4] = [
        VertexAttribute {
            format: VertexFormat::Float32,
            offset: 0,
            shader_location: 7,
        },
        VertexAttribute {
            format: VertexFormat::Float32,
            offset: mem::size_of::<f32>() as BufferAddress,
            shader_location: 10,
        },
        VertexAttribute {
            format: VertexFormat::Float32x4,
            offset: mem::size_of::<[f32; 2]>() as BufferAddress,
            shader_location: 11,
        },
        VertexAttribute {
            format: VertexFormat::Float32x4,
            offset: (mem::size_of::<[f32; 2]>() + mem::size_of::<TextureUvTransform>())
                as BufferAddress,
            shader_location: 12,
        },
    ];

    pub fn desc<'a>() -> VertexBufferLayout<'a> {
        VertexBufferLayout {
            array_stride: Self::STRIDE,
            step_mode: VertexStepMode::Instance,
            attributes: &Self::ATTRIBUTES,
        }
    }
}
