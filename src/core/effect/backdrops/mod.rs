use crate::core::effect::{
    self, BackdropCaptureArea, BackdropEffectConfig, CaptureRegion, TextureSizeError,
};
use crate::core::geometry;
use crate::core::{MathRect, Size};
use tracing::warn;

/// Resolves capture bounds and viewport overlap
pub(crate) fn compute_backdrop_capture_region(
    logical_screen_bounds: MathRect,
    backdrop_config: BackdropEffectConfig,
    scale_factor: f64,
    physical_size: Size,
    max_capture_dimension: u32,
) -> Option<CaptureRegion> {
    let logical_rect = match backdrop_config.capture_area {
        BackdropCaptureArea::NodeBounds => logical_screen_bounds,
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
        match effect::validate_texture_size(capture_size, physical_size, max_capture_dimension) {
            Ok(()) => {}
            Err(TextureSizeError::DimensionExceeded) => {
                warn!(
                    requested_width = capture_size.width,
                    requested_height = capture_size.height,
                    max_capture_dimension,
                    "Skipping backdrop capture that exceeds supported texture dimensions"
                );
                return None;
            }
            Err(TextureSizeError::TexelBudgetExceeded { requested, maximum }) => {
                warn!(
                    requested_width = capture_size.width,
                    requested_height = capture_size.height,
                    requested_texels = requested,
                    max_capture_texels = maximum,
                    viewport_width = physical_size.width,
                    viewport_height = physical_size.height,
                    "Skipping backdrop capture that exceeds the per-viewport texel budget"
                );
                return None;
            }
        }

        Some(effect::resolve_capture_region_to_viewport(
            requested_rect,
            requested_rect,
            physical_size,
        ))
    })
}

#[cfg(test)]
mod tests;
