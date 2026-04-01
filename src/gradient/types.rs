use smallvec::SmallVec;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use crate::Color;

use super::errors::GradientError;
use super::normalize::NormalizedGradient;

// ── Gradient kind discriminant ───────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GradientKind {
    Linear,
    Radial,
    Conic,
}

// ── Public descriptor types ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum GradientDesc {
    Linear(LinearGradientDesc),
    Radial(RadialGradientDesc),
    Conic(ConicGradientDesc),
}

#[derive(Debug, Clone)]
pub struct GradientCommonDesc {
    pub units: GradientUnits,
    pub spread: SpreadMode,
    pub interpolation: ColorInterpolation,
    pub stops: GradientStops,
}

impl GradientCommonDesc {
    pub fn new(stops: impl Into<GradientStops>) -> Self {
        Self {
            units: GradientUnits::Local,
            spread: SpreadMode::Pad,
            interpolation: ColorInterpolation::Srgb,
            stops: stops.into(),
        }
    }

    pub fn with_units(mut self, units: GradientUnits) -> Self {
        self.units = units;
        self
    }

    pub fn with_spread(mut self, spread: SpreadMode) -> Self {
        self.spread = spread;
        self
    }

    pub fn with_interpolation(mut self, interpolation: ColorInterpolation) -> Self {
        self.interpolation = interpolation;
        self
    }
}

#[derive(Debug, Clone, Default)]
pub struct GradientStops {
    stops: SmallVec<[GradientStop; 8]>,
}

impl GradientStops {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.stops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.stops.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, GradientStop> {
        self.stops.iter()
    }

    pub fn as_slice(&self) -> &[GradientStop] {
        self.stops.as_slice()
    }
}

impl Deref for GradientStops {
    type Target = [GradientStop];

    fn deref(&self) -> &Self::Target {
        self.stops.as_slice()
    }
}

impl DerefMut for GradientStops {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.stops.as_mut_slice()
    }
}

impl AsRef<[GradientStop]> for GradientStops {
    fn as_ref(&self) -> &[GradientStop] {
        self.as_slice()
    }
}

impl<const N: usize> From<[GradientStop; N]> for GradientStops {
    fn from(stops: [GradientStop; N]) -> Self {
        Self {
            stops: stops.into_iter().collect(),
        }
    }
}

impl From<Vec<GradientStop>> for GradientStops {
    fn from(stops: Vec<GradientStop>) -> Self {
        Self {
            stops: SmallVec::from_vec(stops),
        }
    }
}

impl FromIterator<GradientStop> for GradientStops {
    fn from_iter<T: IntoIterator<Item = GradientStop>>(iter: T) -> Self {
        Self {
            stops: iter.into_iter().collect(),
        }
    }
}

impl IntoIterator for GradientStops {
    type Item = GradientStop;
    type IntoIter = <SmallVec<[GradientStop; 8]> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.stops.into_iter()
    }
}

impl<'a> IntoIterator for &'a GradientStops {
    type Item = &'a GradientStop;
    type IntoIter = std::slice::Iter<'a, GradientStop>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &'a mut GradientStops {
    type Item = &'a mut GradientStop;
    type IntoIter = std::slice::IterMut<'a, GradientStop>;

    fn into_iter(self) -> Self::IntoIter {
        self.stops.iter_mut()
    }
}

type GradientRampKeyStops = SmallVec<[GradientRampStopKey; 8]>;

#[derive(Debug, Clone)]
pub struct LinearGradientDesc {
    pub common: GradientCommonDesc,
    pub line: LinearGradientLine,
}

impl LinearGradientDesc {
    pub fn new(line: LinearGradientLine, stops: impl Into<GradientStops>) -> Self {
        Self {
            common: GradientCommonDesc::new(stops),
            line,
        }
    }

    pub fn with_units(mut self, units: GradientUnits) -> Self {
        self.common = self.common.with_units(units);
        self
    }

    pub fn with_spread(mut self, spread: SpreadMode) -> Self {
        self.common = self.common.with_spread(spread);
        self
    }

    pub fn with_interpolation(mut self, interpolation: ColorInterpolation) -> Self {
        self.common = self.common.with_interpolation(interpolation);
        self
    }
}

#[derive(Debug, Clone)]
pub struct RadialGradientDesc {
    pub common: GradientCommonDesc,
    pub center: [f32; 2],
    pub shape: RadialGradientShape,
    pub size: RadialGradientSize,
}

impl RadialGradientDesc {
    pub fn new(
        center: [f32; 2],
        shape: RadialGradientShape,
        size: RadialGradientSize,
        stops: impl Into<GradientStops>,
    ) -> Self {
        Self {
            common: GradientCommonDesc::new(stops),
            center,
            shape,
            size,
        }
    }

    pub fn with_units(mut self, units: GradientUnits) -> Self {
        self.common = self.common.with_units(units);
        self
    }

    pub fn with_spread(mut self, spread: SpreadMode) -> Self {
        self.common = self.common.with_spread(spread);
        self
    }

    pub fn with_interpolation(mut self, interpolation: ColorInterpolation) -> Self {
        self.common = self.common.with_interpolation(interpolation);
        self
    }
}

#[derive(Debug, Clone)]
pub struct ConicGradientDesc {
    pub common: GradientCommonDesc,
    pub center: [f32; 2],
    pub start_angle_radians: f32,
}

impl ConicGradientDesc {
    pub fn new(
        center: [f32; 2],
        start_angle_radians: f32,
        stops: impl Into<GradientStops>,
    ) -> Self {
        Self {
            common: GradientCommonDesc::new(stops),
            center,
            start_angle_radians,
        }
    }

    pub fn with_units(mut self, units: GradientUnits) -> Self {
        self.common = self.common.with_units(units);
        self
    }

    pub fn with_spread(mut self, spread: SpreadMode) -> Self {
        self.common = self.common.with_spread(spread);
        self
    }

    pub fn with_interpolation(mut self, interpolation: ColorInterpolation) -> Self {
        self.common = self.common.with_interpolation(interpolation);
        self
    }
}

#[derive(Debug, Clone)]
pub struct GradientStop {
    pub positions: GradientStopPositions,
    pub color: GradientColor,
    pub hint_to_next_segment: Option<GradientStopOffset>,
}

impl GradientStop {
    pub fn auto(color: impl Into<GradientColor>) -> Self {
        Self {
            positions: GradientStopPositions::Auto,
            color: color.into(),
            hint_to_next_segment: None,
        }
    }

    pub fn at_position(position: GradientStopOffset, color: impl Into<GradientColor>) -> Self {
        Self {
            positions: GradientStopPositions::Single(position),
            color: color.into(),
            hint_to_next_segment: None,
        }
    }

    pub fn between_positions(
        start_position: GradientStopOffset,
        end_position: GradientStopOffset,
        color: impl Into<GradientColor>,
    ) -> Self {
        Self {
            positions: GradientStopPositions::Double(start_position, end_position),
            color: color.into(),
            hint_to_next_segment: None,
        }
    }

    pub fn with_hint_to_next_segment(mut self, hint_to_next_segment: GradientStopOffset) -> Self {
        self.hint_to_next_segment = Some(hint_to_next_segment);
        self
    }
}

// ── Supporting enums ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientUnits {
    Local,
    Canvas,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RadialGradientSize {
    ExplicitCircleRadius(f32),
    ExplicitEllipseRadii { radius_x: f32, radius_y: f32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorInterpolation {
    Oklab,
    Srgb,
    SrgbLinear,
    Hsl { hue: HueInterpolationMethod },
    Hwb { hue: HueInterpolationMethod },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HueInterpolationMethod {
    Shorter,
    Longer,
    Increasing,
    Decreasing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpreadMode {
    Pad,
    Repeat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadialGradientShape {
    Circle,
    Ellipse,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GradientStopPositions {
    Auto,
    Single(GradientStopOffset),
    Double(GradientStopOffset, GradientStopOffset),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GradientStopOffset {
    LinearRadial(f32),
    ConicRadians(f32),
}

impl GradientStopOffset {
    pub fn linear_radial(value: f32) -> Self {
        Self::LinearRadial(value)
    }

    pub fn conic_radians(value: f32) -> Self {
        Self::ConicRadians(value)
    }

    pub(crate) fn value(&self) -> f32 {
        match self {
            GradientStopOffset::LinearRadial(v) => *v,
            GradientStopOffset::ConicRadians(v) => *v,
        }
    }

    pub(crate) fn is_linear_radial(&self) -> bool {
        matches!(self, GradientStopOffset::LinearRadial(_))
    }

    pub(crate) fn is_conic(&self) -> bool {
        matches!(self, GradientStopOffset::ConicRadians(_))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HueComponent {
    Degrees(f32),
    Missing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum HueComponentKey {
    Degrees(u32),
    Missing,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GradientColor {
    Srgb {
        red: f32,
        green: f32,
        blue: f32,
        alpha: f32,
    },
    SrgbLinear {
        red: f32,
        green: f32,
        blue: f32,
        alpha: f32,
    },
    Oklab {
        l: f32,
        a: f32,
        b: f32,
        alpha: f32,
    },
    Hsl {
        hue: HueComponent,
        saturation: f32,
        lightness: f32,
        alpha: f32,
    },
    Hwb {
        hue: HueComponent,
        whiteness: f32,
        blackness: f32,
        alpha: f32,
    },
}

impl From<Color> for GradientColor {
    fn from(color: Color) -> Self {
        let [red, green, blue, alpha] = color.to_array();
        Self::Srgb {
            red: red as f32 / 255.0,
            green: green as f32 / 255.0,
            blue: blue as f32 / 255.0,
            alpha: alpha as f32 / 255.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum GradientColorKey {
    Srgb {
        red_bits: u32,
        green_bits: u32,
        blue_bits: u32,
        alpha_bits: u32,
    },
    SrgbLinear {
        red_bits: u32,
        green_bits: u32,
        blue_bits: u32,
        alpha_bits: u32,
    },
    Oklab {
        l_bits: u32,
        a_bits: u32,
        b_bits: u32,
        alpha_bits: u32,
    },
    Hsl {
        hue: HueComponentKey,
        saturation_bits: u32,
        lightness_bits: u32,
        alpha_bits: u32,
    },
    Hwb {
        hue: HueComponentKey,
        whiteness_bits: u32,
        blackness_bits: u32,
        alpha_bits: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ColorInterpolationKey {
    Oklab,
    Srgb,
    SrgbLinear,
    Hsl { hue: HueInterpolationMethod },
    Hwb { hue: HueInterpolationMethod },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GradientRampStopKey {
    pub(crate) position_bits: u32,
    pub(crate) color: GradientColorKey,
    pub(crate) hint_bits: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct GradientRampCacheKey {
    pub(crate) interpolation: ColorInterpolationKey,
    pub(crate) stops: GradientRampKeyStops,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearGradientLine {
    pub start: [f32; 2],
    pub end: [f32; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientSupport {
    Unsupported,
    Supported,
}

// ── Fill enum ────────────────────────────────────────────────────────────────

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum Fill {
    Solid(Color),
    Gradient(Gradient),
}

impl From<Color> for Fill {
    fn from(color: Color) -> Self {
        Self::Solid(color)
    }
}

impl From<Gradient> for Fill {
    fn from(gradient: Gradient) -> Self {
        Self::Gradient(gradient)
    }
}

// ── Validated opaque Gradient ────────────────────────────────────────────────

/// The number of texels in a baked gradient ramp texture.
pub(crate) const RAMP_RESOLUTION: usize = 256;

#[derive(Debug, Clone)]
pub struct Gradient {
    pub(crate) data: GradientData,
}

#[derive(Debug, Clone)]
pub(crate) enum GradientRamp {
    Constant([f32; 4]),
    /// Ramps are not initialized right away, since we use cache as a performance optimization.
    /// We still need to create an instance of a ramp right away to make the gradient struct
    /// complete, hence there's a variant that signifies that we need to create/get actual ramp
    /// before actually using the gradient.
    Pending(Box<GradientRampSource>),
    Sampled(Arc<[[f32; 4]; RAMP_RESOLUTION]>),
}

impl GradientRamp {
    pub(crate) fn as_slice(&self) -> &[[f32; 4]] {
        match self {
            GradientRamp::Constant(color) => std::slice::from_ref(color),
            GradientRamp::Pending(_) => {
                panic!("gradient ramp must be materialized before accessing ramp texels")
            }
            GradientRamp::Sampled(ramp) => &ramp[..],
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct GradientRampSource {
    pub(crate) interpolation: ColorInterpolation,
    pub(crate) normalized: NormalizedGradient,
}

#[derive(Debug, Clone)]
pub(crate) struct GradientData {
    pub(crate) kind: GradientKind,
    pub(crate) units: GradientUnits,
    pub(crate) spread: SpreadMode,
    pub(crate) ramp_cache_key: GradientRampCacheKey,
    /// Baked linear premultiplied RGBA ramp, RAMP_RESOLUTION entries.
    pub(crate) ramp: GradientRamp,
    // Geometry params:
    pub(crate) linear_line: Option<LinearGradientLine>,
    pub(crate) radial_center: Option<[f32; 2]>,
    pub(crate) radial_radius: Option<[f32; 2]>, // (rx, ry)
    pub(crate) conic_center: Option<[f32; 2]>,
    pub(crate) conic_start_angle: Option<f32>,
    /// For repeating: period_start and period_len in the t/theta domain
    pub(crate) period_start: f32,
    pub(crate) period_len: f32,
    /// True when the gradient is a constant fill (degenerate cases, single stop)
    pub(crate) is_constant: bool,
    pub(crate) constant_color: [f32; 4],
}

impl Gradient {
    pub fn new(desc: GradientDesc) -> Result<Self, GradientError> {
        match desc {
            GradientDesc::Linear(d) => Self::linear(d),
            GradientDesc::Radial(d) => Self::radial(d),
            GradientDesc::Conic(d) => Self::conic(d),
        }
    }

    pub fn linear(desc: LinearGradientDesc) -> Result<Self, GradientError> {
        validate_common(&desc.common, GradientKind::Linear)?;
        validate_finite_f32(desc.line.start[0], "line.start[0]")?;
        validate_finite_f32(desc.line.start[1], "line.start[1]")?;
        validate_finite_f32(desc.line.end[0], "line.end[0]")?;
        validate_finite_f32(desc.line.end[1], "line.end[1]")?;

        let normalized = NormalizedGradient::from_common(&desc.common, GradientKind::Linear);
        let ramp_cache_key =
            GradientRampCacheKey::from_normalized(&desc.common.interpolation, &normalized);

        let dx = desc.line.end[0] - desc.line.start[0];
        let dy = desc.line.end[1] - desc.line.start[1];
        let axis_len_sq = dx * dx + dy * dy;

        if axis_len_sq <= RESOLVED_DEGENERATE_EPSILON * RESOLVED_DEGENERATE_EPSILON {
            let constant_color = normalized.degenerate_constant_color();
            return Ok(Gradient {
                data: GradientData {
                    kind: GradientKind::Linear,
                    units: desc.common.units,
                    spread: desc.common.spread,
                    ramp_cache_key,
                    ramp: GradientRamp::Constant(constant_color),
                    linear_line: Some(desc.line),
                    radial_center: None,
                    radial_radius: None,
                    conic_center: None,
                    conic_start_angle: None,
                    period_start: normalized.period_start,
                    period_len: normalized.period_len,
                    is_constant: true,
                    constant_color,
                },
            });
        }

        let period_start = normalized.period_start;
        let period_len = normalized.period_len;
        let ramp = GradientRamp::Pending(Box::new(GradientRampSource {
            interpolation: desc.common.interpolation,
            normalized,
        }));

        Ok(Gradient {
            data: GradientData {
                kind: GradientKind::Linear,
                units: desc.common.units,
                spread: desc.common.spread,
                ramp_cache_key,
                ramp,
                linear_line: Some(desc.line),
                radial_center: None,
                radial_radius: None,
                conic_center: None,
                conic_start_angle: None,
                period_start,
                period_len,
                is_constant: false,
                constant_color: [0.0; 4],
            },
        })
    }

    pub fn radial(desc: RadialGradientDesc) -> Result<Self, GradientError> {
        validate_common(&desc.common, GradientKind::Radial)?;
        validate_finite_f32(desc.center[0], "center[0]")?;
        validate_finite_f32(desc.center[1], "center[1]")?;

        let (radius_x, radius_y) = match (&desc.shape, &desc.size) {
            (RadialGradientShape::Circle, RadialGradientSize::ExplicitCircleRadius(r)) => {
                validate_finite_f32(*r, "radius")?;
                if *r < 0.0 {
                    return Err(GradientError::InvalidRadialDefinition);
                }
                (*r, *r)
            }
            (
                RadialGradientShape::Ellipse,
                RadialGradientSize::ExplicitEllipseRadii { radius_x, radius_y },
            ) => {
                validate_finite_f32(*radius_x, "radius_x")?;
                validate_finite_f32(*radius_y, "radius_y")?;
                if *radius_x < 0.0 || *radius_y < 0.0 {
                    return Err(GradientError::InvalidRadialDefinition);
                }
                (*radius_x, *radius_y)
            }
            _ => return Err(GradientError::InvalidRadialDefinition),
        };

        let normalized = NormalizedGradient::from_common(&desc.common, GradientKind::Radial);
        let ramp_cache_key =
            GradientRampCacheKey::from_normalized(&desc.common.interpolation, &normalized);

        let is_degenerate = radius_x.abs() <= RESOLVED_DEGENERATE_EPSILON
            || radius_y.abs() <= RESOLVED_DEGENERATE_EPSILON;

        if is_degenerate {
            let constant_color = normalized.degenerate_constant_color();
            return Ok(Gradient {
                data: GradientData {
                    kind: GradientKind::Radial,
                    units: desc.common.units,
                    spread: desc.common.spread,
                    ramp_cache_key,
                    ramp: GradientRamp::Constant(constant_color),
                    linear_line: None,
                    radial_center: Some(desc.center),
                    radial_radius: Some([radius_x, radius_y]),
                    conic_center: None,
                    conic_start_angle: None,
                    period_start: normalized.period_start,
                    period_len: normalized.period_len,
                    is_constant: true,
                    constant_color,
                },
            });
        }

        let period_start = normalized.period_start;
        let period_len = normalized.period_len;
        let ramp = GradientRamp::Pending(Box::new(GradientRampSource {
            interpolation: desc.common.interpolation,
            normalized,
        }));

        Ok(Gradient {
            data: GradientData {
                kind: GradientKind::Radial,
                units: desc.common.units,
                spread: desc.common.spread,
                ramp_cache_key,
                ramp,
                linear_line: None,
                radial_center: Some(desc.center),
                radial_radius: Some([radius_x, radius_y]),
                conic_center: None,
                conic_start_angle: None,
                period_start,
                period_len,
                is_constant: false,
                constant_color: [0.0; 4],
            },
        })
    }

    pub fn conic(desc: ConicGradientDesc) -> Result<Self, GradientError> {
        validate_common(&desc.common, GradientKind::Conic)?;
        validate_finite_f32(desc.center[0], "center[0]")?;
        validate_finite_f32(desc.center[1], "center[1]")?;

        if !desc.start_angle_radians.is_finite() {
            return Err(GradientError::NonFiniteAngle {
                field: "start_angle_radians",
            });
        }

        let normalized = NormalizedGradient::from_common(&desc.common, GradientKind::Conic);
        let ramp_cache_key =
            GradientRampCacheKey::from_normalized(&desc.common.interpolation, &normalized);
        // For repeating conic with zero period, degenerate
        let is_degenerate = desc.common.spread == SpreadMode::Repeat
            && normalized.period_len <= RESOLVED_DEGENERATE_EPSILON;
        let constant_color = if is_degenerate {
            normalized.degenerate_constant_color()
        } else {
            [0.0; 4]
        };
        let period_start = normalized.period_start;
        let period_len = normalized.period_len;
        let ramp = if is_degenerate {
            GradientRamp::Constant(constant_color)
        } else {
            GradientRamp::Pending(Box::new(GradientRampSource {
                interpolation: desc.common.interpolation,
                normalized,
            }))
        };

        Ok(Gradient {
            data: GradientData {
                kind: GradientKind::Conic,
                units: desc.common.units,
                spread: desc.common.spread,
                ramp_cache_key,
                ramp,
                linear_line: None,
                radial_center: None,
                radial_radius: None,
                conic_center: Some(desc.center),
                conic_start_angle: Some(desc.start_angle_radians),
                period_start,
                period_len,
                is_constant: is_degenerate,
                constant_color,
            },
        })
    }
}

impl HueComponentKey {
    fn from_hue_component(hue_component: HueComponent) -> Self {
        match hue_component {
            HueComponent::Degrees(value) => Self::Degrees(value.to_bits()),
            HueComponent::Missing => Self::Missing,
        }
    }
}

impl GradientColorKey {
    fn from_gradient_color(color: GradientColor) -> Self {
        match color {
            GradientColor::Srgb {
                red,
                green,
                blue,
                alpha,
            } => Self::Srgb {
                red_bits: red.to_bits(),
                green_bits: green.to_bits(),
                blue_bits: blue.to_bits(),
                alpha_bits: alpha.to_bits(),
            },
            GradientColor::SrgbLinear {
                red,
                green,
                blue,
                alpha,
            } => Self::SrgbLinear {
                red_bits: red.to_bits(),
                green_bits: green.to_bits(),
                blue_bits: blue.to_bits(),
                alpha_bits: alpha.to_bits(),
            },
            GradientColor::Oklab { l, a, b, alpha } => Self::Oklab {
                l_bits: l.to_bits(),
                a_bits: a.to_bits(),
                b_bits: b.to_bits(),
                alpha_bits: alpha.to_bits(),
            },
            GradientColor::Hsl {
                hue,
                saturation,
                lightness,
                alpha,
            } => Self::Hsl {
                hue: HueComponentKey::from_hue_component(hue),
                saturation_bits: saturation.to_bits(),
                lightness_bits: lightness.to_bits(),
                alpha_bits: alpha.to_bits(),
            },
            GradientColor::Hwb {
                hue,
                whiteness,
                blackness,
                alpha,
            } => Self::Hwb {
                hue: HueComponentKey::from_hue_component(hue),
                whiteness_bits: whiteness.to_bits(),
                blackness_bits: blackness.to_bits(),
                alpha_bits: alpha.to_bits(),
            },
        }
    }
}

impl ColorInterpolationKey {
    fn from_interpolation(interpolation: ColorInterpolation) -> Self {
        match interpolation {
            ColorInterpolation::Oklab => Self::Oklab,
            ColorInterpolation::Srgb => Self::Srgb,
            ColorInterpolation::SrgbLinear => Self::SrgbLinear,
            ColorInterpolation::Hsl { hue } => Self::Hsl { hue },
            ColorInterpolation::Hwb { hue } => Self::Hwb { hue },
        }
    }
}

impl GradientRampCacheKey {
    pub(crate) fn from_normalized(
        interpolation: &ColorInterpolation,
        normalized: &NormalizedGradient,
    ) -> Self {
        let mut stops = GradientRampKeyStops::with_capacity(normalized.stops.len());
        for stop in &normalized.stops {
            stops.push(GradientRampStopKey {
                position_bits: stop.position.to_bits(),
                color: GradientColorKey::from_gradient_color(stop.color),
                hint_bits: stop.hint.map(f32::to_bits),
            });
        }

        Self {
            interpolation: ColorInterpolationKey::from_interpolation(*interpolation),
            stops,
        }
    }
}

// ── Degenerate threshold ─────────────────────────────────────────────────────

pub(crate) const RESOLVED_DEGENERATE_EPSILON: f32 = 1e-6;

// ── Validation helpers ───────────────────────────────────────────────────────

fn validate_finite_f32(value: f32, field: &'static str) -> Result<(), GradientError> {
    if !value.is_finite() {
        return Err(GradientError::NonFiniteGeometryParameter { field });
    }
    Ok(())
}

fn validate_common(common: &GradientCommonDesc, kind: GradientKind) -> Result<(), GradientError> {
    if common.stops.is_empty() {
        return Err(GradientError::EmptyStops);
    }

    let is_conic = kind == GradientKind::Conic;

    for (stop_index, stop) in common.stops.iter().enumerate() {
        // Validate stop color components are finite
        validate_gradient_color_finite(stop_index, &stop.color)?;

        // Validate stop positions
        match &stop.positions {
            GradientStopPositions::Auto => {}
            GradientStopPositions::Single(offset) => {
                validate_stop_offset_kind(stop_index, offset, is_conic)?;
                validate_stop_offset_finite(stop_index, offset)?;
            }
            GradientStopPositions::Double(a, b) => {
                validate_stop_offset_kind(stop_index, a, is_conic)?;
                validate_stop_offset_kind(stop_index, b, is_conic)?;
                validate_stop_offset_finite(stop_index, a)?;
                validate_stop_offset_finite(stop_index, b)?;
                if a.value() > b.value() {
                    return Err(GradientError::ReversedDoublePositionStop {
                        stop_index,
                        first: a.value(),
                        second: b.value(),
                    });
                }
            }
        }

        // Validate hint
        if let Some(hint) = &stop.hint_to_next_segment {
            if is_conic && !hint.is_conic() {
                return Err(GradientError::InvalidHintOffsetKind { stop_index });
            }
            if !is_conic && !hint.is_linear_radial() {
                return Err(GradientError::InvalidHintOffsetKind { stop_index });
            }
            if !hint.value().is_finite() {
                return Err(GradientError::NonFiniteHint { stop_index });
            }
        }
    }

    Ok(())
}

fn validate_stop_offset_kind(
    stop_index: usize,
    offset: &GradientStopOffset,
    is_conic: bool,
) -> Result<(), GradientError> {
    if is_conic && !offset.is_conic() {
        return Err(GradientError::InvalidStopOffsetKind { stop_index });
    }
    if !is_conic && !offset.is_linear_radial() {
        return Err(GradientError::InvalidStopOffsetKind { stop_index });
    }
    Ok(())
}

fn validate_stop_offset_finite(
    stop_index: usize,
    offset: &GradientStopOffset,
) -> Result<(), GradientError> {
    if !offset.value().is_finite() {
        return Err(GradientError::NonFiniteStopOffset { stop_index });
    }
    Ok(())
}

fn validate_gradient_color_finite(
    stop_index: usize,
    color: &GradientColor,
) -> Result<(), GradientError> {
    match color {
        GradientColor::Srgb {
            red,
            green,
            blue,
            alpha,
        }
        | GradientColor::SrgbLinear {
            red,
            green,
            blue,
            alpha,
        } => {
            check_finite(stop_index, *red, "red")?;
            check_finite(stop_index, *green, "green")?;
            check_finite(stop_index, *blue, "blue")?;
            check_finite(stop_index, *alpha, "alpha")?;
        }
        GradientColor::Oklab { l, a, b, alpha } => {
            check_finite(stop_index, *l, "l")?;
            check_finite(stop_index, *a, "a")?;
            check_finite(stop_index, *b, "b")?;
            check_finite(stop_index, *alpha, "alpha")?;
        }
        GradientColor::Hsl {
            hue,
            saturation,
            lightness,
            alpha,
        } => {
            if let HueComponent::Degrees(deg) = hue {
                check_finite(stop_index, *deg, "hue")?;
            }
            check_finite(stop_index, *saturation, "saturation")?;
            check_finite(stop_index, *lightness, "lightness")?;
            check_finite(stop_index, *alpha, "alpha")?;
        }
        GradientColor::Hwb {
            hue,
            whiteness,
            blackness,
            alpha,
        } => {
            if let HueComponent::Degrees(deg) = hue {
                check_finite(stop_index, *deg, "hue")?;
            }
            check_finite(stop_index, *whiteness, "whiteness")?;
            check_finite(stop_index, *blackness, "blackness")?;
            check_finite(stop_index, *alpha, "alpha")?;
        }
    }
    Ok(())
}

fn check_finite(
    stop_index: usize,
    value: f32,
    component: &'static str,
) -> Result<(), GradientError> {
    if !value.is_finite() {
        return Err(GradientError::NonFiniteColorComponent {
            stop_index,
            component,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_stop_common() -> GradientCommonDesc {
        GradientCommonDesc::new([GradientStop::auto(Color::rgb(255, 0, 0))])
            .with_interpolation(ColorInterpolation::SrgbLinear)
    }

    #[test]
    fn degenerate_linear_has_nonempty_ramp() {
        let g = Gradient::linear(LinearGradientDesc {
            common: single_stop_common(),
            line: LinearGradientLine {
                start: [0.0, 0.0],
                end: [0.0, 0.0], // zero-length → degenerate
            },
        })
        .unwrap();
        assert!(g.data.is_constant);
        assert!(matches!(g.data.ramp, GradientRamp::Constant(_)));
    }

    #[test]
    fn degenerate_radial_has_nonempty_ramp() {
        let g = Gradient::radial(RadialGradientDesc {
            common: single_stop_common(),
            center: [50.0, 50.0],
            shape: RadialGradientShape::Circle,
            size: RadialGradientSize::ExplicitCircleRadius(0.0), // zero radius → degenerate
        })
        .unwrap();
        assert!(g.data.is_constant);
        assert!(matches!(g.data.ramp, GradientRamp::Constant(_)));
    }

    #[test]
    fn radial_rejects_negative_circle_radius() {
        let gradient = Gradient::radial(RadialGradientDesc {
            common: single_stop_common(),
            center: [50.0, 50.0],
            shape: RadialGradientShape::Circle,
            size: RadialGradientSize::ExplicitCircleRadius(-1.0),
        });

        assert!(matches!(
            gradient,
            Err(GradientError::InvalidRadialDefinition)
        ));
    }

    #[test]
    fn radial_rejects_negative_ellipse_radius() {
        let gradient = Gradient::radial(RadialGradientDesc {
            common: single_stop_common(),
            center: [50.0, 50.0],
            shape: RadialGradientShape::Ellipse,
            size: RadialGradientSize::ExplicitEllipseRadii {
                radius_x: 20.0,
                radius_y: -1.0,
            },
        });

        assert!(matches!(
            gradient,
            Err(GradientError::InvalidRadialDefinition)
        ));
    }

    #[test]
    fn gradient_stops_collect_without_exposing_smallvec() {
        let stops = [
            GradientStop::at_position(
                GradientStopOffset::linear_radial(0.0),
                Color::rgb(255, 0, 0),
            ),
            GradientStop::at_position(
                GradientStopOffset::linear_radial(1.0),
                Color::rgb(0, 0, 255),
            ),
        ];
        let first_color = stops[0].color;
        let second_color = stops[1].color;
        let gradient_stops = GradientStops::from(stops);

        assert_eq!(gradient_stops.len(), 2);
        assert_eq!(gradient_stops[0].color, first_color);
        assert_eq!(gradient_stops[1].color, second_color);
    }

    #[test]
    fn fill_converts_from_color_and_gradient() {
        let solid_fill = Fill::from(Color::rgb(10, 20, 30));
        assert!(matches!(solid_fill, Fill::Solid(_)));

        let gradient = Gradient::linear(
            LinearGradientDesc::new(
                LinearGradientLine {
                    start: [0.0, 0.0],
                    end: [10.0, 0.0],
                },
                [
                    GradientStop::at_position(
                        GradientStopOffset::linear_radial(0.0),
                        Color::rgb(255, 0, 0),
                    ),
                    GradientStop::at_position(
                        GradientStopOffset::linear_radial(1.0),
                        Color::rgb(0, 0, 255),
                    ),
                ],
            )
            .with_interpolation(ColorInterpolation::SrgbLinear),
        )
        .unwrap();

        let gradient_fill = Fill::from(gradient);
        assert!(matches!(gradient_fill, Fill::Gradient(_)));
    }

    #[test]
    fn descriptor_builder_methods_apply_defaults_and_overrides() {
        let gradient = Gradient::linear(
            LinearGradientDesc::new(
                LinearGradientLine {
                    start: [0.0, 0.0],
                    end: [10.0, 0.0],
                },
                [
                    GradientStop::at_position(
                        GradientStopOffset::linear_radial(0.0),
                        Color::rgb(255, 0, 0),
                    )
                    .with_hint_to_next_segment(GradientStopOffset::linear_radial(0.25)),
                    GradientStop::between_positions(
                        GradientStopOffset::linear_radial(0.5),
                        GradientStopOffset::linear_radial(0.75),
                        Color::rgb(0, 0, 255),
                    ),
                ],
            )
            .with_units(GradientUnits::Canvas)
            .with_spread(SpreadMode::Repeat)
            .with_interpolation(ColorInterpolation::SrgbLinear),
        )
        .unwrap();

        assert_eq!(gradient.data.units, GradientUnits::Canvas);
        assert_eq!(gradient.data.spread, SpreadMode::Repeat);
        assert!(!gradient.data.is_constant);
    }

    #[test]
    fn radial_and_conic_descriptor_builders_set_common_configuration() {
        let radial_gradient = Gradient::radial(
            RadialGradientDesc::new(
                [50.0, 50.0],
                RadialGradientShape::Circle,
                RadialGradientSize::ExplicitCircleRadius(20.0),
                [
                    GradientStop::at_position(
                        GradientStopOffset::linear_radial(0.0),
                        Color::rgb(255, 255, 0),
                    ),
                    GradientStop::at_position(
                        GradientStopOffset::linear_radial(1.0),
                        Color::rgb(0, 255, 0),
                    ),
                ],
            )
            .with_units(GradientUnits::Canvas)
            .with_interpolation(ColorInterpolation::SrgbLinear),
        )
        .unwrap();

        let conic_gradient = Gradient::conic(
            ConicGradientDesc::new(
                [10.0, 20.0],
                0.5,
                [
                    GradientStop::at_position(
                        GradientStopOffset::conic_radians(0.0),
                        Color::rgb(255, 0, 0),
                    ),
                    GradientStop::at_position(
                        GradientStopOffset::conic_radians(std::f32::consts::TAU),
                        Color::rgb(255, 0, 0),
                    ),
                ],
            )
            .with_spread(SpreadMode::Repeat),
        )
        .unwrap();

        assert_eq!(radial_gradient.data.units, GradientUnits::Canvas);
        assert_eq!(radial_gradient.data.kind, GradientKind::Radial);
        assert_eq!(conic_gradient.data.spread, SpreadMode::Repeat);
        assert_eq!(conic_gradient.data.kind, GradientKind::Conic);
    }

    #[test]
    fn nonconstant_gradients_start_with_pending_ramp() {
        let gradient = Gradient::linear(
            LinearGradientDesc::new(
                LinearGradientLine {
                    start: [0.0, 0.0],
                    end: [10.0, 0.0],
                },
                [
                    GradientStop::at_position(
                        GradientStopOffset::linear_radial(0.0),
                        Color::rgb(255, 0, 0),
                    ),
                    GradientStop::at_position(
                        GradientStopOffset::linear_radial(1.0),
                        Color::rgb(0, 0, 255),
                    ),
                ],
            )
            .with_interpolation(ColorInterpolation::SrgbLinear),
        )
        .unwrap();

        assert!(matches!(gradient.data.ramp, GradientRamp::Pending(_)));
    }
}
