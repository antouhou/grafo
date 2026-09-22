use crate::shape::TextureSampling;

/// Matches the shader's TextureSamplingParams uniform.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct TextureSamplingUniform {
    origin: [f32; 2],
    inverse_size: [f32; 2],
    uses_target_coordinates: u32,
    padding: [u32; 3],
}

impl From<TextureSampling> for TextureSamplingUniform {
    fn from(sampling: TextureSampling) -> Self {
        match sampling {
            TextureSampling::ShapeUv => Self {
                origin: [0.0; 2],
                inverse_size: [1.0; 2],
                uses_target_coordinates: 0,
                padding: [0; 3],
            },
            TextureSampling::TargetPixels(bounds) => Self {
                origin: [bounds.min.x as f32, bounds.min.y as f32],
                inverse_size: [
                    1.0 / bounds.width().max(1) as f32,
                    1.0 / bounds.height().max(1) as f32,
                ],
                uses_target_coordinates: 1,
                padding: [0; 3],
            },
        }
    }
}

impl Default for TextureSamplingUniform {
    fn default() -> Self {
        TextureSampling::default().into()
    }
}

#[cfg(test)]
mod tests;
