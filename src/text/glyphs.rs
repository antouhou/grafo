use super::outline;
use crate::core::GlyphContent;
use cosmic_text::{CacheKey, CacheKeyFlags, Font};
use lyon::path::PathEvent;
use swash::{
    scale::{outline::Outline, ScaleContext},
    zeno::{Angle, Transform},
    Tag,
};

/// Reusable storage for extracting glyphs from resolved cosmic-text fonts.
/// Can be owned by a rendering thread independently of the shaping font system.
#[derive(Default)]
pub struct CosmicGlyphExtractor {
    scale_context: ScaleContext,
    outline: Outline,
    events: Vec<PathEvent>,
}

impl CosmicGlyphExtractor {
    /// Extracts a glyph using the font and weight that produced its cache key.
    /// The returned content is borrowed until the next extraction.
    pub fn glyph_content(&mut self, font: &Font, key: &CacheKey) -> Option<GlyphContent<'_>> {
        let font = font.as_swash();
        let mut scaler = self
            .scale_context
            .builder(font)
            .size(f32::from_bits(key.font_size_bits))
            .hint(false);
        let weight_tag = Tag::from_be_bytes(*b"wght");
        if let Some(variation) = font.variations().find_by_tag(weight_tag) {
            let weight =
                f32::from(key.font_weight.0).clamp(variation.min_value(), variation.max_value());
            scaler = scaler
                .normalized_coords(font.variations().normalized_coords([(weight_tag, weight)]));
        }
        if !scaler
            .build()
            .scale_outline_into(key.glyph_id, &mut self.outline)
        {
            return None;
        }
        if key.flags.contains(CacheKeyFlags::FAKE_ITALIC) {
            self.outline.transform(&Transform::skew(
                Angle::from_degrees(14.0),
                Angle::from_degrees(0.0),
            ));
        }
        outline::collect_events(&self.outline, &mut self.events);
        Some(GlyphContent::Outline(&self.events))
    }
}
