use crate::core::effect::{self, BackdropCaptureArea, BackdropCaptureRegion, BackdropEffectConfig};

use crate::core::geometry;
use crate::core::vertex::InstanceTransform;
use crate::core::{MathRect, Size};
use tracing::warn;

const MAX_BACKDROP_CAPTURE_VIEWPORT_TEXEL_MULTIPLIER: u64 = 4;

fn does_capture_size_exceeds_limits(capture_size: Size, max_capture_dimension: u32) -> bool {
    capture_size.width > max_capture_dimension || capture_size.height > max_capture_dimension
}

fn max_backdrop_capture_texels(physical_size: Size) -> u64 {
    let viewport_texels = u64::from(physical_size.width) * u64::from(physical_size.height);
    viewport_texels.saturating_mul(MAX_BACKDROP_CAPTURE_VIEWPORT_TEXEL_MULTIPLIER)
}

fn does_capture_size_exceeds_budget(capture_size: Size, physical_size: Size) -> bool {
    let capture_texels = u64::from(capture_size.width) * u64::from(capture_size.height);
    capture_texels > max_backdrop_capture_texels(physical_size)
}

/// Resolves the requested bounds and viewport overlap before allocating capture textures.
pub(crate) fn compute_backdrop_capture_region(
    local_bounds: MathRect,
    transform: Option<InstanceTransform>,
    backdrop_config: BackdropEffectConfig,
    scale_factor: f64,
    physical_size: Size,
    max_capture_dimension: u32,
) -> Option<BackdropCaptureRegion> {
    let logical_rect = match backdrop_config.capture_area {
        BackdropCaptureArea::NodeBounds => {
            geometry::transformed_bounds_to_logical_screen_rect(local_bounds, transform)
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

    geometry::logical_rect_to_physical_rect(logical_rect, scale_factor).and_then(|requested_rect| {
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
                requested_texels = u64::from(capture_size.width) * u64::from(capture_size.height),
                max_capture_texels = max_backdrop_capture_texels(physical_size),
                viewport_width = physical_size.width,
                viewport_height = physical_size.height,
                "Skipping backdrop capture that exceeds the per-viewport texel budget"
            );
            return None;
        }

        Some(effect::resolve_capture_region_to_viewport(
            requested_rect,
            physical_size,
        ))
    })
}

#[cfg(test)]
mod tests;
