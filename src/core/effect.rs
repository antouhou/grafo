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

use std::sync::Arc;

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

/// A cached shape effect attachment. GPU parameter resources are created only on cache misses.
#[derive(Clone)]
pub(crate) struct ShapeEffectInstance {
    pub effect_id: u64,
    pub params: Arc<[u8]>,
    pub config: ShapeEffectConfig,
}

/// Parameters shared by group and backdrop effect attachments.
pub(crate) struct EffectInstance {
    /// The loaded effect's ID.
    pub effect_id: u64,
    /// Raw bytes for the effect's uniform parameters.
    /// The byte layout must match the shader's uniform declaration.
    pub params: Vec<u8>,
}

/// A backdrop effect attachment and its capture configuration.
pub(crate) struct BackdropEffectInstance {
    pub effect: EffectInstance,
    pub config: BackdropEffectConfig,
}

impl BackdropEffectInstance {
    pub(crate) fn new(effect: EffectInstance, config: BackdropEffectConfig) -> Self {
        Self { effect, config }
    }
}
