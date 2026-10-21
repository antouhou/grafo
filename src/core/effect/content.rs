use crate::core::effect::{self, CaptureRegion, ShapeMaskEffectBounds};
use crate::core::{geometry, MathRect, Size, Viewport};
use tracing::warn;

/// Output padding around a content effect's input, in logical screen pixels.
///
/// The input contains the node and its descendants within the node's bounds. Overflowing
/// content does not expand the bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContentEffectConfig {
    pub left_outset: f32,
    pub top_outset: f32,
    pub right_outset: f32,
    pub bottom_outset: f32,
    /// Scale applied to the padded capture before filtering, in the range `(0.0, 1.0]`.
    /// Shaders operate in texels of the reduced texture. The result is upscaled
    /// to the same padded screen bounds when composited.
    pub downsample: f32,
}

impl Default for ContentEffectConfig {
    fn default() -> Self {
        Self {
            left_outset: 0.0,
            top_outset: 0.0,
            right_outset: 0.0,
            bottom_outset: 0.0,
            downsample: 1.0,
        }
    }
}

impl ContentEffectConfig {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets all output outsets to the same logical screen-space distance.
    pub fn outset(self, outset: f32) -> Self {
        self.outsets(outset, outset, outset, outset)
    }

    /// Sets the left, top, right, and bottom output outsets.
    pub fn outsets(mut self, left: f32, top: f32, right: f32, bottom: f32) -> Self {
        self.left_outset = left;
        self.top_outset = top;
        self.right_outset = right;
        self.bottom_outset = bottom;
        self
    }

    pub fn downsample(mut self, downsample: f32) -> Self {
        self.downsample = downsample;
        self
    }

    pub(crate) fn is_valid(self) -> bool {
        self.downsample > 0.0
            && self.downsample <= 1.0
            && [
                self.left_outset,
                self.top_outset,
                self.right_outset,
                self.bottom_outset,
            ]
            .iter()
            .all(|outset| outset.is_finite() && *outset >= 0.0)
    }
}

fn padded_output_bounds(input_bounds: MathRect, config: ContentEffectConfig) -> MathRect {
    MathRect::new(
        (
            input_bounds.min.x - config.left_outset,
            input_bounds.min.y - config.top_outset,
        )
            .into(),
        (
            input_bounds.max.x + config.right_outset,
            input_bounds.max.y + config.bottom_outset,
        )
            .into(),
    )
}

fn input_bounds(
    shape_bounds: MathRect,
    shape_mask_effect_bounds: Option<ShapeMaskEffectBounds>,
    viewport_size: Size,
    maximum_texture_dimension: u32,
) -> MathRect {
    let Some(bounds) = shape_mask_effect_bounds.filter(|bounds| {
        effect::validate_texture_size(
            Size::new(bounds.texture_size[0], bounds.texture_size[1]),
            viewport_size,
            maximum_texture_dimension,
        )
        .is_ok()
    }) else {
        return shape_bounds;
    };
    shape_bounds.union(&bounds.logical_screen_bounds)
}

/// Input and padded output bounds for a content effect.
#[derive(Clone, Copy)]
pub(crate) struct ContentEffectBounds {
    pub capture_region: CaptureRegion,
    pub sampling_size: Size,
    pub logical_output_bounds: [(f32, f32); 2],
}

impl ContentEffectBounds {
    pub(crate) fn new(
        shape_bounds: MathRect,
        shape_mask_effect_bounds: Option<ShapeMaskEffectBounds>,
        config: ContentEffectConfig,
        viewport: Viewport,
        maximum_texture_dimension: u32,
    ) -> Option<Self> {
        if !config.is_valid() {
            return None;
        }
        let viewport_size = viewport.physical_size.into();
        let logical_input_bounds = input_bounds(
            shape_bounds,
            shape_mask_effect_bounds,
            viewport_size,
            maximum_texture_dimension,
        );
        let input_bounds =
            geometry::logical_rect_to_physical_rect(logical_input_bounds, viewport.scale_factor)?;
        let padded_bounds = padded_output_bounds(logical_input_bounds, config);
        let output_bounds =
            geometry::logical_rect_to_physical_rect(padded_bounds, viewport.scale_factor)?;
        let capture_size = output_bounds.size().to_u32();
        if effect::validate_texture_size(capture_size, viewport_size, maximum_texture_dimension)
            .is_err()
        {
            warn!("skipping oversized content effect texture");
            return None;
        }
        let logical_output_bounds = output_bounds.to_f32() / viewport.scale_factor as f32;
        Some(Self {
            capture_region: effect::resolve_capture_region_to_viewport(
                input_bounds,
                output_bounds,
                viewport_size,
            ),
            sampling_size: geometry::compute_downsampled_dimensions(
                capture_size,
                config.downsample,
            ),
            logical_output_bounds: [
                logical_output_bounds.min.to_tuple(),
                logical_output_bounds.max.to_tuple(),
            ],
        })
    }
}
