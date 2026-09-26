use super::TextureSamplingUniform;
use crate::wgpu_backend::gradient::GpuMaterialParams;
use naga::front::wgsl;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga::{Module, TypeInner};
use std::mem;

fn assert_uniform_layout(module: &Module, name: &str, size: usize, offsets: &[(&str, usize)]) {
    let (_, declaration) = module
        .types
        .iter()
        .find(|(_, declaration)| declaration.name.as_deref() == Some(name))
        .unwrap();
    let TypeInner::Struct { members, span } = &declaration.inner else {
        panic!("{name} must be a uniform structure");
    };
    assert_eq!(*span as usize, size, "{name} uniform size");
    for (name, offset) in offsets {
        let member = members
            .iter()
            .find(|member| member.name.as_deref() == Some(name))
            .unwrap();
        assert_eq!(member.offset as usize, *offset, "{name} uniform offset");
    }
}

#[test]
fn shape_material_uniforms_match_shader_layout() {
    let shader = wgsl::parse_str(include_str!("../../../../shaders/shader.wgsl")).unwrap();
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&shader)
        .unwrap();
    assert_uniform_layout(
        &shader,
        "TextureSamplingParams",
        mem::size_of::<TextureSamplingUniform>(),
        &[
            ("origin", mem::offset_of!(TextureSamplingUniform, origin)),
            (
                "inverse_size",
                mem::offset_of!(TextureSamplingUniform, inverse_size),
            ),
            (
                "uses_target_coordinates",
                mem::offset_of!(TextureSamplingUniform, uses_target_coordinates),
            ),
            (
                "_padding0",
                mem::offset_of!(TextureSamplingUniform, padding),
            ),
        ],
    );
    assert_uniform_layout(
        &shader,
        "MaterialParams",
        mem::size_of::<GpuMaterialParams>(),
        &[
            ("gradient", mem::offset_of!(GpuMaterialParams, gradient)),
            (
                "texture_sampling",
                mem::offset_of!(GpuMaterialParams, texture_sampling),
            ),
        ],
    );
}
