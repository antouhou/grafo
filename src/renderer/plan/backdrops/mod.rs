use crate::effect::{BackdropCaptureArea, BackdropEffectConfig};
use crate::pipeline::BackdropSamplingUniform;
use crate::renderer::rect_utils::{
    logical_rect_to_physical_rect, transformed_bounds_to_logical_screen_rect,
};
use crate::renderer::types::DrawCommand;
use crate::{MathRect, PhysicalRect, Size, UnsignedPhysicalPoint, UnsignedPhysicalRect};
use tracing::warn;

const MAX_BACKDROP_CAPTURE_VIEWPORT_TEXEL_MULTIPLIER: u64 = 4;

fn does_capture_size_exceeds_limits(capture_size: Size, max_capture_dimension: u32) -> bool {
    capture_size.width > max_capture_dimension || capture_size.height > max_capture_dimension
}

fn max_backdrop_capture_texels(physical_size: Size) -> u64 {
    u64::from(physical_size.width)
        .saturating_mul(u64::from(physical_size.height))
        .saturating_mul(MAX_BACKDROP_CAPTURE_VIEWPORT_TEXEL_MULTIPLIER)
}

fn does_capture_size_exceeds_budget(capture_size: Size, physical_size: Size) -> bool {
    let capture_texels =
        u64::from(capture_size.width).saturating_mul(u64::from(capture_size.height));
    capture_texels > max_backdrop_capture_texels(physical_size)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(in crate::renderer) struct BackdropCaptureRegion {
    /// Requested bounds in full-resolution physical pixels, including offscreen padding.
    pub(in crate::renderer) bounds: PhysicalRect,
    /// Viewport overlap to copy, or None when the requested bounds are fully offscreen.
    pub(in crate::renderer) source_rect: Option<UnsignedPhysicalRect>,
    pub(in crate::renderer) copy_destination_origin: UnsignedPhysicalPoint,
}

impl BackdropCaptureRegion {
    pub(in crate::renderer) fn sample_uniform(self) -> BackdropSamplingUniform {
        BackdropSamplingUniform::new(
            self.bounds.min.to_tuple(),
            self.bounds.size().to_u32().to_tuple(),
        )
    }
}

fn resolve_capture_region_to_viewport(
    requested_rect: PhysicalRect,
    physical_size: Size,
) -> BackdropCaptureRegion {
    let viewport = UnsignedPhysicalRect::from_size(physical_size).to_i64();
    let source_rect = requested_rect
        .to_i64()
        .intersection(&viewport)
        .map(|overlap| overlap.to_u32());
    let copy_destination_origin = source_rect
        .map(|source_rect| {
            (source_rect.min.to_i32() - requested_rect.min)
                .to_u32()
                .to_point()
        })
        .unwrap_or_else(UnsignedPhysicalPoint::zero);
    BackdropCaptureRegion {
        bounds: requested_rect,
        source_rect,
        copy_destination_origin,
    }
}

/// Resolves the requested bounds and viewport overlap before allocating capture textures.
pub(in crate::renderer) fn compute_backdrop_capture_region(
    draw_command: &DrawCommand,
    backdrop_config: BackdropEffectConfig,
    scale_factor: f64,
    physical_size: Size,
    max_capture_dimension: u32,
) -> Option<BackdropCaptureRegion> {
    let logical_rect = match backdrop_config.capture_area {
        BackdropCaptureArea::NodeBounds => {
            let bounds = draw_command.local_bounds();
            transformed_bounds_to_logical_screen_rect(
                MathRect::new(bounds[0].into(), bounds[1].into()),
                draw_command.transform(),
            )
        }
        BackdropCaptureArea::FullScene => {
            // Match capture rounding in f32; to_logical's f64 division can add a pixel.
            MathRect::from_size(physical_size.to_f32() / scale_factor as f32)
        }
        BackdropCaptureArea::ScreenRect(rect) => MathRect::new(rect[0].into(), rect[1].into()),
    };

    if !logical_rect.is_finite() {
        return None;
    }
    let logical_rect = MathRect::from_points([logical_rect.min, logical_rect.max])
        .inflate(backdrop_config.padding, backdrop_config.padding);

    logical_rect_to_physical_rect(logical_rect, scale_factor).and_then(|requested_rect| {
        let capture_size = requested_rect.size().to_u32();
        if does_capture_size_exceeds_limits(capture_size, max_capture_dimension) {
            warn!(
                requested_width = capture_size.width,
                requested_height = capture_size.height,
                max_capture_dimension,
                "Skipping backdrop capture that exceeds supported texture dimensions"
            );
            return None;
        }

        if does_capture_size_exceeds_budget(capture_size, physical_size) {
            warn!(
                requested_width = capture_size.width,
                requested_height = capture_size.height,
                requested_texels =
                    u64::from(capture_size.width).saturating_mul(u64::from(capture_size.height)),
                max_capture_texels = max_backdrop_capture_texels(physical_size),
                viewport_width = physical_size.width,
                viewport_height = physical_size.height,
                "Skipping backdrop capture that exceeds the per-viewport texel budget"
            );
            return None;
        }

        Some(resolve_capture_region_to_viewport(
            requested_rect,
            physical_size,
        ))
    })
}

#[cfg(test)]
mod tests;
