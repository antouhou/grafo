use super::CoverageVertex;
use std::mem;
use wgpu::{VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode};

impl CoverageVertex {
    const ATTRIBUTES: [VertexAttribute; 5] = [
        VertexAttribute {
            format: VertexFormat::Float32x2,
            offset: 0,
            shader_location: 0,
        },
        VertexAttribute {
            format: VertexFormat::Float32x3,
            offset: 8,
            shader_location: 2,
        },
        VertexAttribute {
            format: VertexFormat::Float32x3,
            offset: 20,
            shader_location: 7,
        },
        VertexAttribute {
            format: VertexFormat::Uint32,
            offset: mem::offset_of!(CoverageVertex, coverage_start) as u64,
            shader_location: 8,
        },
        VertexAttribute {
            format: VertexFormat::Uint32x2,
            offset: mem::offset_of!(CoverageVertex, tile_origin) as u64,
            shader_location: 12,
        },
    ];

    pub(in crate::wgpu_backend) fn desc<'a>() -> VertexBufferLayout<'a> {
        VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as u64,
            step_mode: VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }

    pub(in crate::wgpu_backend) fn solid_desc<'a>() -> VertexBufferLayout<'a> {
        const ATTRIBUTES: [VertexAttribute; 3] = [
            CoverageVertex::ATTRIBUTES[0],
            CoverageVertex::ATTRIBUTES[3],
            CoverageVertex::ATTRIBUTES[4],
        ];
        VertexBufferLayout {
            attributes: &ATTRIBUTES,
            ..Self::desc()
        }
    }
}
