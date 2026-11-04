use crate::core::vertex::{CustomVertex, InstanceTransform, TextureUvTransform};
use bytemuck::{Pod, Zeroable};
use lyon::geom::euclid::default::Transform3D;
use std::{mem, ops::Range};
use wgpu::{BufferAddress, VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode};

pub(crate) fn encode_paint_transform(transform: Transform3D<f32>) -> [[f32; 3]; 3] {
    let columns = transform.to_arrays();
    [0, 1, 3].map(|column| [columns[column][0], columns[column][1], columns[column][3]])
}

/// Locations of one geometry's vertices and local indices in the shared buffers.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct GeometryBufferRange {
    pub(crate) index_start: u32,
    pub(crate) index_count: u32,
    pub(crate) vertex_start: i32,
    pub(crate) vertex_count: usize,
}

impl GeometryBufferRange {
    pub(crate) fn indices(self) -> Range<u32> {
        self.index_start..self.index_start + self.index_count
    }
}

impl CustomVertex {
    pub(crate) const STRIDE: BufferAddress = mem::size_of::<Self>() as BufferAddress;

    const ATTRIBUTES: [VertexAttribute; 2] = [
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
    ];

    pub fn desc<'a>() -> VertexBufferLayout<'a> {
        VertexBufferLayout {
            array_stride: Self::STRIDE,
            step_mode: VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }

    pub(crate) fn solid_desc<'a>() -> VertexBufferLayout<'a> {
        VertexBufferLayout {
            attributes: &Self::ATTRIBUTES[..1],
            ..Self::desc()
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
    pub coverage_translation: [f32; 2],
    pub texture_flags: f32,
    pub texture_uv_transform_layer0: TextureUvTransform,
    pub texture_uv_transform_layer1: TextureUvTransform,
    pub paint_transform: [[f32; 3]; 3],
}

impl Default for InstanceMetadata {
    fn default() -> Self {
        Self {
            coverage_translation: [0.0; 2],
            texture_flags: 0.0,
            texture_uv_transform_layer0: TextureUvTransform::IDENTITY,
            texture_uv_transform_layer1: TextureUvTransform::IDENTITY,
            paint_transform: encode_paint_transform(Transform3D::identity()),
        }
    }
}

impl InstanceMetadata {
    pub(crate) const STRIDE: BufferAddress = mem::size_of::<Self>() as BufferAddress;

    const ATTRIBUTES: [VertexAttribute; 7] = [
        VertexAttribute {
            format: VertexFormat::Float32x2,
            offset: mem::offset_of!(Self, coverage_translation) as BufferAddress,
            shader_location: 13,
        },
        VertexAttribute {
            format: VertexFormat::Float32,
            offset: mem::offset_of!(Self, texture_flags) as BufferAddress,
            shader_location: 9,
        },
        VertexAttribute {
            format: VertexFormat::Float32x4,
            offset: mem::offset_of!(Self, texture_uv_transform_layer0) as BufferAddress,
            shader_location: 10,
        },
        VertexAttribute {
            format: VertexFormat::Float32x4,
            offset: mem::offset_of!(Self, texture_uv_transform_layer1) as BufferAddress,
            shader_location: 11,
        },
        VertexAttribute {
            format: VertexFormat::Float32x3,
            offset: mem::offset_of!(Self, paint_transform) as BufferAddress,
            shader_location: 7,
        },
        VertexAttribute {
            format: VertexFormat::Float32x3,
            offset: mem::offset_of!(Self, paint_transform) as BufferAddress + 12,
            shader_location: 8,
        },
        VertexAttribute {
            format: VertexFormat::Float32x3,
            offset: mem::offset_of!(Self, paint_transform) as BufferAddress + 24,
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

    pub(crate) fn coverage_desc<'a>() -> VertexBufferLayout<'a> {
        VertexBufferLayout {
            attributes: &Self::ATTRIBUTES[..1],
            ..Self::desc()
        }
    }
}
