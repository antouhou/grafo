//! Custom shader effects for groups, backdrops, and cacheable shape-local masks.
//!
//! Each attachment supplies different input pixels:
//!
//! - Group effects process a rendered subtree captured into an offscreen texture.
//! - Backdrop effects process previously rendered scene pixels behind the node.
//! - Shape effects process a padded white coverage mask of one shape. The result
//!   texture is cached while the shape and effect inputs are unchanged.
//!
//! These descriptions contain effect IDs, configuration, and parameter bytes.
//! `Renderer::load_effect()` delegates WGSL compilation to execution, which owns
//! the pipelines and uploaded parameters.
//! `set_group_effect()`, `set_shape_backdrop_effect()`, and `set_shape_effect()`
//! attach a loaded effect to a draw tree node. Nodes share the compiled pipelines
//! and can supply different parameters.

use crate::core::geometry;
use crate::core::{PhysicalRect, Size, UnsignedPhysicalPoint, UnsignedPhysicalRect};

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

/// Padding around a shape's local bounds for a cached shape effect.
/// Outsets are measured in logical pixels.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ShapeEffectConfig {
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

impl Default for ShapeEffectConfig {
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

impl ShapeEffectConfig {
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
pub struct BackdropCaptureRegion {
    /// Requested bounds in full-resolution physical pixels, including offscreen padding.
    pub bounds: PhysicalRect,
    /// Viewport overlap to copy, or None when the requested bounds are fully offscreen.
    pub source_rect: Option<UnsignedPhysicalRect>,
    pub copy_destination_origin: UnsignedPhysicalPoint,
}

pub(crate) fn resolve_capture_region_to_viewport(
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

#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct ShapeEffectRasterRect {
    pub(crate) local_physical_origin: [i32; 2],
    pub(crate) texture_size: [u32; 2],
    pub(crate) local_bounds: [(f32, f32); 2],
}

pub(crate) fn compute_shape_effect_raster_rect(
    local_bounds: [(f32, f32); 2],
    config: ShapeEffectConfig,
    scale_factor: f64,
    fringe_width: f32,
) -> Option<ShapeEffectRasterRect> {
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
        || !fringe_width.is_finite()
        || fringe_width < 0.0
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

    let guard = f64::from(fringe_width).ceil();
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
    Some(ShapeEffectRasterRect {
        local_physical_origin,
        texture_size: texture_size.to_array(),
        local_bounds: [
            (
                physical_minimum_x as f32 / scale_factor as f32,
                physical_minimum_y as f32 / scale_factor as f32,
            ),
            (
                physical_maximum_x as f32 / scale_factor as f32,
                physical_maximum_y as f32 / scale_factor as f32,
            ),
        ],
    })
}

#[cfg(test)]
mod tests;
