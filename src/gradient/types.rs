use crate::Color;

use super::errors::GradientError;
use super::normalize::NormalizedGradient;
use super::sampling::bake_gradient_ramp;

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
    pub stops: Vec<GradientStop>,
}

#[derive(Debug, Clone)]
pub struct LinearGradientDesc {
    pub common: GradientCommonDesc,
    pub line: LinearGradientLine,
}

#[derive(Debug, Clone)]
pub struct RadialGradientDesc {
    pub common: GradientCommonDesc,
    pub center: [f32; 2],
    pub shape: RadialGradientShape,
    pub size: RadialGradientSize,
}

#[derive(Debug, Clone)]
pub struct ConicGradientDesc {
    pub common: GradientCommonDesc,
    pub center: [f32; 2],
    pub start_angle_radians: f32,
}

#[derive(Debug, Clone)]
pub struct GradientStop {
    pub positions: GradientStopPositions,
    pub color: GradientColor,
    pub hint_to_next_segment: Option<GradientStopOffset>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone)]
pub enum Fill {
    Solid(Color),
    Gradient(Gradient),
}

// ── Validated opaque Gradient ────────────────────────────────────────────────

/// The number of texels in a baked gradient ramp texture.
pub(crate) const RAMP_RESOLUTION: usize = 256;

#[derive(Debug, Clone)]
pub struct Gradient {
    pub(crate) data: GradientData,
}

#[derive(Debug, Clone)]
pub(crate) struct GradientData {
    pub(crate) kind: GradientKind,
    pub(crate) units: GradientUnits,
    pub(crate) spread: SpreadMode,
    /// Baked linear premultiplied RGBA ramp, RAMP_RESOLUTION entries.
    pub(crate) ramp: Vec<[f32; 4]>,
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
                    ramp: vec![constant_color],
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

        let ramp = bake_gradient_ramp(&normalized, &desc.common.interpolation);

        Ok(Gradient {
            data: GradientData {
                kind: GradientKind::Linear,
                units: desc.common.units,
                spread: desc.common.spread,
                ramp,
                linear_line: Some(desc.line),
                radial_center: None,
                radial_radius: None,
                conic_center: None,
                conic_start_angle: None,
                period_start: normalized.period_start,
                period_len: normalized.period_len,
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
                (*r, *r)
            }
            (
                RadialGradientShape::Ellipse,
                RadialGradientSize::ExplicitEllipseRadii { radius_x, radius_y },
            ) => {
                validate_finite_f32(*radius_x, "radius_x")?;
                validate_finite_f32(*radius_y, "radius_y")?;
                (*radius_x, *radius_y)
            }
            _ => return Err(GradientError::InvalidRadialDefinition),
        };

        let normalized = NormalizedGradient::from_common(&desc.common, GradientKind::Radial);

        let is_degenerate = radius_x.abs() <= RESOLVED_DEGENERATE_EPSILON
            || radius_y.abs() <= RESOLVED_DEGENERATE_EPSILON;

        if is_degenerate {
            let constant_color = normalized.degenerate_constant_color();
            return Ok(Gradient {
                data: GradientData {
                    kind: GradientKind::Radial,
                    units: desc.common.units,
                    spread: desc.common.spread,
                    ramp: vec![constant_color],
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

        let ramp = bake_gradient_ramp(&normalized, &desc.common.interpolation);

        Ok(Gradient {
            data: GradientData {
                kind: GradientKind::Radial,
                units: desc.common.units,
                spread: desc.common.spread,
                ramp,
                linear_line: None,
                radial_center: Some(desc.center),
                radial_radius: Some([radius_x, radius_y]),
                conic_center: None,
                conic_start_angle: None,
                period_start: normalized.period_start,
                period_len: normalized.period_len,
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
        let ramp = bake_gradient_ramp(&normalized, &desc.common.interpolation);

        // For repeating conic with zero period, degenerate
        let is_degenerate =
            desc.common.spread == SpreadMode::Repeat && normalized.period_len <= RESOLVED_DEGENERATE_EPSILON;
        let constant_color = if is_degenerate {
            normalized.degenerate_constant_color()
        } else {
            [0.0; 4]
        };

        Ok(Gradient {
            data: GradientData {
                kind: GradientKind::Conic,
                units: desc.common.units,
                spread: desc.common.spread,
                ramp: if is_degenerate { vec![constant_color] } else { ramp },
                linear_line: None,
                radial_center: None,
                radial_radius: None,
                conic_center: Some(desc.center),
                conic_start_angle: Some(desc.start_angle_radians),
                period_start: normalized.period_start,
                period_len: normalized.period_len,
                is_constant: is_degenerate,
                constant_color,
            },
        })
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

fn check_finite(stop_index: usize, value: f32, component: &'static str) -> Result<(), GradientError> {
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
        GradientCommonDesc {
            stops: vec![GradientStop {
                color: GradientColor::Srgb {
                    red: 1.0,
                    green: 0.0,
                    blue: 0.0,
                    alpha: 1.0,
                },
                positions: GradientStopPositions::Auto,
                hint_to_next_segment: None,
            }],
            spread: SpreadMode::Pad,
            units: GradientUnits::Local,
            interpolation: ColorInterpolation::SrgbLinear,
        }
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
        assert!(!g.data.ramp.is_empty(), "degenerate linear ramp must not be empty");
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
        assert!(!g.data.ramp.is_empty(), "degenerate radial ramp must not be empty");
    }
}
