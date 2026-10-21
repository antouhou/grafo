//! Custom shader effects on shape masks, rendered content, and backdrops.
//!
//! Each attachment supplies different input pixels:
//!
//! - Shape mask effects process one shape's padded white geometric coverage mask,
//!   without its fill, textures, or children. The result is drawn behind the shape.
//! - Content effects process the node and its descendants. Input includes shape's own bound,
//!   mask effect if any, and all content withing the shape's bounds. Output outsets reserve
//!   transparent space around the input. The result replaces the captured content.
//! - Backdrop effects process previously rendered scene pixels behind the node.
//!   The result is drawn through the shape beneath its fill.
//!
//! These descriptions contain effect IDs, configuration, and parameter bytes.
//! `Scene::load_effect()` delegates WGSL compilation to execution, which owns
//! the pipelines and uploaded parameters.
//! `set_content_effect()`, `set_backdrop_effect()`, and `set_shape_mask_effect()`
//! attach a loaded effect to a draw tree node. Nodes share the compiled pipelines
//! and can supply different parameters.

use crate::core::geometry;
use crate::core::vertex::InstanceTransform;
use crate::core::{MathRect, PhysicalRect, Size, UnsignedPhysicalPoint, UnsignedPhysicalRect};
pub(crate) use content::ContentEffectBounds;
pub use content::ContentEffectConfig;
use thiserror::Error;

pub(crate) mod backdrops;
mod content;

#[derive(Debug, Error, PartialEq, Eq)]
pub(crate) enum TextureSizeError {
    #[error("Texture dimensions exceed the supported maximum")]
    DimensionExceeded,
    #[error("Texture requires {requested} texels, exceeding the budget of {maximum}")]
    TexelBudgetExceeded { requested: u64, maximum: u64 },
}

pub(crate) fn validate_texture_size(
    texture_size: Size,
    viewport_size: Size,
    maximum_texture_dimension: u32,
) -> Result<(), TextureSizeError> {
    if texture_size.width > maximum_texture_dimension
        || texture_size.height > maximum_texture_dimension
    {
        return Err(TextureSizeError::DimensionExceeded);
    }
    let viewport_texels = u64::from(viewport_size.width) * u64::from(viewport_size.height);
    let maximum_texels = viewport_texels.saturating_mul(4);
    let texture_texels = u64::from(texture_size.width) * u64::from(texture_size.height);
    if texture_texels > maximum_texels {
        return Err(TextureSizeError::TexelBudgetExceeded {
            requested: texture_texels,
            maximum: maximum_texels,
        });
    }
    Ok(())
}

/// The rendered region to capture as input to a backdrop effect.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub enum BackdropCaptureArea {
    /// Capture the node's transformed local bounds.
    #[default]
    NodeBounds,
    /// Capture the entire viewport.
    FullScene,
    /// Capture an explicit logical screen-space rectangle.
    ScreenRect([(f32, f32); 2]),
}

/// Per-node configuration for backdrop capture before the effect shader runs.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BackdropEffectConfig {
    /// The rendered region to capture, in logical screen coordinates.
    pub capture_area: BackdropCaptureArea,
    /// Additional logical screen-space padding applied around the requested capture area.
    ///
    /// Blur effects need pixels outside the node bounds to avoid clipped edges.
    pub padding: f32,
    /// Scale factor applied to the captured region before running the effect.
    /// `1.0` keeps full resolution, `0.5` halves each axis, and so on.
    pub downsample: f32,
}

impl Default for BackdropEffectConfig {
    fn default() -> Self {
        Self {
            capture_area: BackdropCaptureArea::NodeBounds,
            padding: 0.0,
            downsample: 1.0,
        }
    }
}

impl BackdropEffectConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn capture_area(mut self, capture_area: BackdropCaptureArea) -> Self {
        self.capture_area = capture_area;
        self
    }

    pub fn padding(mut self, padding: f32) -> Self {
        self.padding = padding;
        self
    }

    pub fn downsample(mut self, downsample: f32) -> Self {
        self.downsample = downsample;
        self
    }
}

/// Padding around a shape's local bounds and a scale for a mask effect.
/// Outsets are measured in logical pixels.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ShapeMaskEffectConfig {
    pub left_outset: f32,
    pub top_outset: f32,
    pub right_outset: f32,
    pub bottom_outset: f32,
    /// Scale factor applied to the mask and effect textures before running the
    /// effect. `1.0` keeps full resolution, `0.5` halves each axis, and so on.
    /// The smaller result texture is bilinearly upscaled to the shape's full
    /// bounds when drawn. Effect shaders operate in texels of the downsampled
    /// texture, so texel-based radii and offsets scale up visually.
    pub downsample: f32,
}

impl Default for ShapeMaskEffectConfig {
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

impl ShapeMaskEffectConfig {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets all four outsets to the same logical-space distance.
    pub fn outset(mut self, outset: f32) -> Self {
        self.left_outset = outset;
        self.top_outset = outset;
        self.right_outset = outset;
        self.bottom_outset = outset;
        self
    }

    /// Sets the left, top, right, and bottom logical-space outsets.
    pub fn outsets(mut self, left: f32, top: f32, right: f32, bottom: f32) -> Self {
        self.left_outset = left;
        self.top_outset = top;
        self.right_outset = right;
        self.bottom_outset = bottom;
        self
    }

    /// Sets the rasterization scale for the mask and effect textures.
    /// Must be in the range `(0.0, 1.0]`; values below `1.0` render the effect
    /// at reduced resolution and bilinearly upscale it when drawing.
    pub fn downsample(mut self, downsample: f32) -> Self {
        self.downsample = downsample;
        self
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CaptureRegion {
    /// Destination bounds in full-resolution physical pixels, including padding.
    pub bounds: PhysicalRect,
    /// Source pixels to copy, or None when the input is fully offscreen.
    pub source_rect: Option<UnsignedPhysicalRect>,
    pub copy_destination_origin: UnsignedPhysicalPoint,
}

pub(crate) fn resolve_capture_region_to_viewport(
    input_bounds: PhysicalRect,
    output_bounds: PhysicalRect,
    physical_size: Size,
) -> CaptureRegion {
    let viewport = UnsignedPhysicalRect::from_size(physical_size).to_i64();
    let source_rect = input_bounds
        .to_i64()
        .intersection(&viewport)
        .map(|overlap| overlap.to_u32());
    let copy_destination_origin = source_rect
        .map(|source_rect| {
            (source_rect.min.to_i32() - output_bounds.min)
                .to_u32()
                .to_point()
        })
        .unwrap_or_else(UnsignedPhysicalPoint::zero);
    CaptureRegion {
        bounds: output_bounds,
        source_rect,
        copy_destination_origin,
    }
}

/// Local raster bounds, texture dimensions, and the unclipped screen footprint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ShapeMaskEffectBounds {
    pub local_physical_origin: [i32; 2],
    pub texture_size: [u32; 2],
    pub local_bounds: [(f32, f32); 2],
    pub logical_screen_bounds: MathRect,
}

impl ShapeMaskEffectBounds {
    pub(crate) fn new(
        local_bounds: [(f32, f32); 2],
        config: ShapeMaskEffectConfig,
        transform: Option<InstanceTransform>,
        scale_factor: f64,
    ) -> Option<Self> {
        let bounds_and_outsets = [
            local_bounds[0].0,
            local_bounds[0].1,
            local_bounds[1].0,
            local_bounds[1].1,
            config.left_outset,
            config.top_outset,
            config.right_outset,
            config.bottom_outset,
        ];
        if !scale_factor.is_finite()
            || scale_factor <= 0.0
            || !config.downsample.is_finite()
            || config.downsample <= 0.0
            || config.downsample > 1.0
            || !bounds_and_outsets.iter().all(|value| value.is_finite())
        {
            return None;
        }

        let minimum_x = local_bounds[0].0.min(local_bounds[1].0) - config.left_outset;
        let minimum_y = local_bounds[0].1.min(local_bounds[1].1) - config.top_outset;
        let maximum_x = local_bounds[0].0.max(local_bounds[1].0) + config.right_outset;
        let maximum_y = local_bounds[0].1.max(local_bounds[1].1) + config.bottom_outset;
        if ![minimum_x, minimum_y, maximum_x, maximum_y]
            .iter()
            .all(|value| value.is_finite())
        {
            return None;
        }

        // Keep a transparent border for filtering the mask texture.
        let guard = 1.0;
        let physical_minimum_x = (f64::from(minimum_x) * scale_factor).floor() - guard;
        let physical_minimum_y = (f64::from(minimum_y) * scale_factor).floor() - guard;
        let physical_maximum_x = (f64::from(maximum_x) * scale_factor).ceil() + guard;
        let physical_maximum_y = (f64::from(maximum_y) * scale_factor).ceil() + guard;

        let coordinates = [
            physical_minimum_x,
            physical_minimum_y,
            physical_maximum_x,
            physical_maximum_y,
        ];
        if !coordinates.iter().all(|value| {
            value.is_finite() && *value >= f64::from(i32::MIN) && *value <= f64::from(i32::MAX)
        }) {
            return None;
        }

        let local_physical_origin = [physical_minimum_x as i32, physical_minimum_y as i32];
        let physical_width = physical_maximum_x - physical_minimum_x;
        let physical_height = physical_maximum_y - physical_minimum_y;
        if physical_width <= 0.0
            || physical_height <= 0.0
            || physical_width > f64::from(u32::MAX)
            || physical_height > f64::from(u32::MAX)
        {
            return None;
        }

        let full_resolution_size = Size::new(physical_width as u32, physical_height as u32);
        let texture_size =
            geometry::compute_downsampled_dimensions(full_resolution_size, config.downsample);
        let local_bounds = [
            (
                physical_minimum_x as f32 / scale_factor as f32,
                physical_minimum_y as f32 / scale_factor as f32,
            ),
            (
                physical_maximum_x as f32 / scale_factor as f32,
                physical_maximum_y as f32 / scale_factor as f32,
            ),
        ];
        let logical_screen_bounds = geometry::transformed_bounds_to_logical_screen_rect(
            MathRect::new(local_bounds[0].into(), local_bounds[1].into()),
            transform,
        );
        Some(Self {
            local_physical_origin,
            texture_size: texture_size.to_array(),
            local_bounds,
            logical_screen_bounds,
        })
    }
}

#[cfg(test)]
mod tests;
