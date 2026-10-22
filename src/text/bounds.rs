use crate::core::{geometry, MathRect, TransformInstance};
use cosmic_text::{CacheKeyFlags, Font, LayoutGlyph};
use lyon::math::point;

pub(super) fn glyph_bounds(
    font: &Font,
    glyph: &LayoutGlyph,
    position: (f32, f32),
) -> Option<MathRect> {
    // The font-wide box does not guarantee coverage of variable outlines.
    if font.as_swash().variations().next().is_some() {
        return None;
    }
    let metrics = font.metrics();
    let bounds = metrics.bounds?;
    let scale = glyph.font_size / f32::from(metrics.units_per_em);
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let local_bounds = MathRect::new(
        point(bounds.x_min, -bounds.y_max),
        point(bounds.x_max, -bounds.y_min),
    );
    if local_bounds.is_empty() {
        return None;
    }
    let skew = if glyph.cache_key_flags.contains(CacheKeyFlags::FAKE_ITALIC) {
        -14.0_f32.to_radians().tan()
    } else {
        0.0
    };
    let transform =
        TransformInstance::affine_2d(scale, 0.0, skew * scale, scale, position.0, position.1);
    let bounds = geometry::transformed_bounds_to_logical_screen_rect(local_bounds, Some(transform));
    bounds.is_finite().then_some(bounds)
}
