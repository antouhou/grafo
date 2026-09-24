use super::errors::GradientError;
use super::normalize::NormalizedGradient;
use crate::core::Color;
use smallvec::SmallVec;
use std::ops::{Deref, DerefMut};
use std::slice::{Iter, IterMut};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GradientKind {
    Linear,
    Radial,
    Conic,
}

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
    type IntoIter = Iter<'a, GradientStop>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &'a mut GradientStops {
    type Item = &'a mut GradientStop;
    type IntoIter = IterMut<'a, GradientStop>;

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
    pub size: RadialGradientSize,
}

impl RadialGradientDesc {
    pub fn new(
        center: [f32; 2],
        size: RadialGradientSize,
        stops: impl Into<GradientStops>,
    ) -> Self {
        Self {
            common: GradientCommonDesc::new(stops),
            center,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
pub(crate) struct GradientRampStopKey {
    pub(crate) position_bits: u32,
    pub(crate) color: GradientColorKey,
    pub(crate) hint_bits: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct GradientRampCacheKey {
    pub(crate) interpolation: ColorInterpolation,
    pub(crate) stops: GradientRampKeyStops,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearGradientLine {
    pub start: [f32; 2],
    pub end: [f32; 2],
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum Fill {
    Solid(Color),
    Gradient(Gradient),
}

impl Fill {
    #[inline]
    pub fn to_normalized_solid(&self) -> Option<[f32; 4]> {
        match self {
            Fill::Solid(color) => Some(color.normalize()),
            _ => None,
        }
    }
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

/// The number of texels in a baked gradient ramp texture.
pub(crate) const RAMP_RESOLUTION: usize = 1024;

#[derive(Debug, Clone)]
pub struct Gradient {
    pub(crate) data: GradientData,
}

#[derive(Debug, Clone)]
pub(crate) enum GradientRamp {
    Constant([f32; 4]),
    /// Resolved from the ramp cache or baked before upload.
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

#[derive(Debug, Clone, Copy)]
pub(crate) enum GradientGeometry {
    Linear(LinearGradientLine),
    Radial { center: [f32; 2], radius: [f32; 2] },
    Conic { center: [f32; 2], start_angle: f32 },
}

#[derive(Debug, Clone)]
pub(crate) struct GradientData {
    pub(crate) geometry: GradientGeometry,
    pub(crate) units: GradientUnits,
    pub(crate) spread: SpreadMode,
    pub(crate) ramp_cache_key: GradientRampCacheKey,
    /// Pending source, a constant color, or a sampled linear premultiplied RGBA ramp.
    pub(crate) ramp: GradientRamp,
    /// For repeating: period_start and period_len in the t/theta domain
    pub(crate) period_start: f32,
    pub(crate) period_len: f32,
}

impl GradientData {
    fn new(
        common: &GradientCommonDesc,
        geometry: GradientGeometry,
        normalized: NormalizedGradient,
        is_degenerate: bool,
    ) -> Self {
        let ramp_cache_key =
            GradientRampCacheKey::from_normalized(&common.interpolation, &normalized);
        let period_start = normalized.period_start;
        let period_len = normalized.period_len;
        let constant_color = if is_degenerate {
            Some(normalized.degenerate_constant_color())
        } else {
            normalized.constant_color()
        };
        let ramp = match constant_color {
            Some(color) => GradientRamp::Constant(color),
            None => GradientRamp::Pending(Box::new(GradientRampSource {
                interpolation: common.interpolation,
                normalized,
            })),
        };

        Self {
            geometry,
            units: common.units,
            spread: common.spread,
            ramp_cache_key,
            ramp,
            period_start,
            period_len,
        }
    }
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
        let dx = desc.line.end[0] - desc.line.start[0];
        let dy = desc.line.end[1] - desc.line.start[1];
        let axis_len_sq = dx * dx + dy * dy;
        let is_degenerate =
            axis_len_sq <= RESOLVED_DEGENERATE_EPSILON * RESOLVED_DEGENERATE_EPSILON;

        Ok(Gradient {
            data: GradientData::new(
                &desc.common,
                GradientGeometry::Linear(desc.line),
                normalized,
                is_degenerate,
            ),
        })
    }

    pub fn radial(desc: RadialGradientDesc) -> Result<Self, GradientError> {
        validate_common(&desc.common, GradientKind::Radial)?;
        validate_finite_f32(desc.center[0], "center[0]")?;
        validate_finite_f32(desc.center[1], "center[1]")?;

        let (radius_x, radius_y) = match desc.size {
            RadialGradientSize::ExplicitCircleRadius(radius) => {
                validate_finite_f32(radius, "radius")?;
                if radius < 0.0 {
                    return Err(GradientError::InvalidRadialDefinition);
                }
                (radius, radius)
            }
            RadialGradientSize::ExplicitEllipseRadii { radius_x, radius_y } => {
                validate_finite_f32(radius_x, "radius_x")?;
                validate_finite_f32(radius_y, "radius_y")?;
                if radius_x < 0.0 || radius_y < 0.0 {
                    return Err(GradientError::InvalidRadialDefinition);
                }
                (radius_x, radius_y)
            }
        };

        let normalized = NormalizedGradient::from_common(&desc.common, GradientKind::Radial);
        let is_degenerate = radius_x.abs() <= RESOLVED_DEGENERATE_EPSILON
            || radius_y.abs() <= RESOLVED_DEGENERATE_EPSILON;

        Ok(Gradient {
            data: GradientData::new(
                &desc.common,
                GradientGeometry::Radial {
                    center: desc.center,
                    radius: [radius_x, radius_y],
                },
                normalized,
                is_degenerate,
            ),
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
        let is_degenerate = desc.common.spread == SpreadMode::Repeat
            && normalized.period_len <= RESOLVED_DEGENERATE_EPSILON;

        Ok(Gradient {
            data: GradientData::new(
                &desc.common,
                GradientGeometry::Conic {
                    center: desc.center,
                    start_angle: desc.start_angle_radians,
                },
                normalized,
                is_degenerate,
            ),
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
            interpolation: *interpolation,
            stops,
        }
    }
}

pub(crate) const RESOLVED_DEGENERATE_EPSILON: f32 = 1e-6;

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
        validate_gradient_color_finite(stop_index, &stop.color)?;

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
