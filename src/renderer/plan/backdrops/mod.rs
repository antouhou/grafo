use crate::effect::{BackdropCaptureArea, BackdropEffectConfig};
use crate::pipeline::BackdropSamplingUniform;
use crate::renderer::rect_utils::{
    inflate_logical_rect, logical_rect_to_physical_rect, transformed_bounds_to_logical_screen_rect,
};
use crate::renderer::types::DrawCommand;
use tracing::warn;

const MAX_BACKDROP_CAPTURE_VIEWPORT_TEXEL_MULTIPLIER: u64 = 4;

fn does_capture_size_exceeds_limits(capture_size: (u32, u32), max_capture_dimension: u32) -> bool {
    capture_size.0 > max_capture_dimension || capture_size.1 > max_capture_dimension
}

fn max_backdrop_capture_texels(physical_size: (u32, u32)) -> u64 {
    u64::from(physical_size.0)
        .saturating_mul(u64::from(physical_size.1))
        .saturating_mul(MAX_BACKDROP_CAPTURE_VIEWPORT_TEXEL_MULTIPLIER)
}

fn does_capture_size_exceeds_budget(capture_size: (u32, u32), physical_size: (u32, u32)) -> bool {
    let capture_texels = u64::from(capture_size.0).saturating_mul(u64::from(capture_size.1));
    capture_texels > max_backdrop_capture_texels(physical_size)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(in crate::renderer) struct BackdropCaptureRegion {
    /// Requested bounds in full-resolution physical pixels, including offscreen padding.
    pub(in crate::renderer) capture_origin: (i32, i32),
    pub(in crate::renderer) capture_size: (u32, u32),
    /// No source copy is needed when the requested bounds are fully offscreen.
    pub(in crate::renderer) copy_source_origin: Option<(u32, u32)>,
    pub(in crate::renderer) copy_destination_origin: (u32, u32),
    pub(in crate::renderer) copy_size: (u32, u32),
}

impl BackdropCaptureRegion {
    pub(in crate::renderer) fn sample_uniform(self) -> BackdropSamplingUniform {
        BackdropSamplingUniform::new(self.capture_origin, self.capture_size)
    }
}

fn resolve_capture_region_to_viewport(
    requested_rect: (i32, i32, u32, u32),
    physical_size: (u32, u32),
) -> BackdropCaptureRegion {
    let (capture_x, capture_y, capture_width, capture_height) = requested_rect;
    let capture_right = capture_x.saturating_add(capture_width as i32);
    let capture_bottom = capture_y.saturating_add(capture_height as i32);

    let overlap_left = capture_x.max(0);
    let overlap_top = capture_y.max(0);
    let overlap_right = capture_right.min(physical_size.0 as i32);
    let overlap_bottom = capture_bottom.min(physical_size.1 as i32);

    let overlap_width = overlap_right.saturating_sub(overlap_left).max(0) as u32;
    let overlap_height = overlap_bottom.saturating_sub(overlap_top).max(0) as u32;
    let has_overlap = overlap_width > 0 && overlap_height > 0;

    BackdropCaptureRegion {
        capture_origin: (capture_x, capture_y),
        capture_size: (capture_width, capture_height),
        copy_source_origin: has_overlap.then_some((overlap_left as u32, overlap_top as u32)),
        copy_destination_origin: (
            overlap_left.saturating_sub(capture_x) as u32,
            overlap_top.saturating_sub(capture_y) as u32,
        ),
        copy_size: (overlap_width, overlap_height),
    }
}

/// Resolves the requested bounds and viewport overlap before allocating capture textures.
pub(in crate::renderer) fn compute_backdrop_capture_region(
    draw_command: &DrawCommand,
    backdrop_config: BackdropEffectConfig,
    scale_factor: f64,
    physical_size: (u32, u32),
    max_capture_dimension: u32,
) -> Option<BackdropCaptureRegion> {
    let logical_rect = match backdrop_config.capture_area {
        BackdropCaptureArea::NodeBounds => transformed_bounds_to_logical_screen_rect(
            draw_command.local_bounds(),
            draw_command.transform(),
        ),
        BackdropCaptureArea::FullScene => {
            // Match capture rounding in f32; to_logical's f64 division can add a pixel.
            let logical_width = physical_size.0 as f32 / scale_factor as f32;
            let logical_height = physical_size.1 as f32 / scale_factor as f32;
            [(0.0, 0.0), (logical_width, logical_height)]
        }
        BackdropCaptureArea::ScreenRect(rect) => rect,
    };

    let logical_rect = inflate_logical_rect(logical_rect, backdrop_config.padding);

    logical_rect_to_physical_rect(logical_rect, scale_factor).and_then(|requested_rect| {
        let capture_size = (requested_rect.2, requested_rect.3);
        if does_capture_size_exceeds_limits(capture_size, max_capture_dimension) {
            warn!(
                requested_width = capture_size.0,
                requested_height = capture_size.1,
                max_capture_dimension,
                "Skipping backdrop capture that exceeds supported texture dimensions"
            );
            return None;
        }

        if does_capture_size_exceeds_budget(capture_size, physical_size) {
            warn!(
                requested_width = capture_size.0,
                requested_height = capture_size.1,
                requested_texels =
                    u64::from(capture_size.0).saturating_mul(u64::from(capture_size.1)),
                max_capture_texels = max_backdrop_capture_texels(physical_size),
                viewport_width = physical_size.0,
                viewport_height = physical_size.1,
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
