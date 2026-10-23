use crate::core::{GlyphContent, MathRect, PositionedGlyph, TextSource};
use cosmic_text::{Buffer, CacheKey, CacheKeyFlags, FontSystem, LayoutGlyph, LayoutRunIter};
use lyon::path::PathEvent;
use std::slice;
use swash::{
    scale::{outline::Outline, ScaleContext},
    zeno::{Angle, Transform},
    Tag,
};

mod bounds;
mod outline;

#[derive(Default)]
struct GlyphExtraction {
    scale_context: ScaleContext,
    outline: Outline,
    events: Vec<PathEvent>,
}

/// A cursor over an already shaped cosmic-text buffer. Borrowed only during add or replace.
/// Span colors, decorations, and glyphs without ordinary outlines are ignored.
pub struct CosmicText<'a> {
    runs: LayoutRunIter<'a>,
    glyphs: slice::Iter<'a, LayoutGlyph>,
    baseline: f32,
    font_system: &'a mut FontSystem,
    extraction: GlyphExtraction,
    visible_bounds: Option<MathRect>,
}

impl<'a> CosmicText<'a> {
    /// Use the font system that shaped `buffer`. This does not perform layout.
    pub fn new(buffer: &'a Buffer, font_system: &'a mut FontSystem) -> Self {
        Self {
            runs: buffer.layout_runs(),
            glyphs: [].iter(),
            baseline: 0.0,
            font_system,
            extraction: GlyphExtraction::default(),
            visible_bounds: None,
        }
    }

    fn is_glyph_outside_visible_bounds(
        &mut self,
        glyph: &LayoutGlyph,
        position: (f32, f32),
    ) -> bool {
        let Some(visible_bounds) = self.visible_bounds else {
            return false;
        };
        let Some(font) = self.font_system.get_font(glyph.font_id, glyph.font_weight) else {
            return false;
        };
        bounds::glyph_bounds(&font, glyph, position)
            .is_some_and(|bounds| !visible_bounds.intersects(&bounds))
    }
}

impl TextSource for CosmicText<'_> {
    type GlyphKey = CacheKey;

    fn set_visible_bounds(&mut self, bounds: Option<MathRect>) {
        self.visible_bounds = bounds;
    }

    fn next_glyph(&mut self) -> Option<PositionedGlyph<CacheKey>> {
        if self.visible_bounds.is_some_and(|bounds| bounds.is_empty()) {
            return None;
        }
        loop {
            let Some(glyph) = self.glyphs.next() else {
                let run = self.runs.next()?;
                self.glyphs = run.glyphs.iter();
                self.baseline = run.line_y;
                continue;
            };
            let position = (
                glyph.x + glyph.font_size * glyph.x_offset,
                self.baseline + glyph.y - glyph.font_size * glyph.y_offset,
            );
            if self.is_glyph_outside_visible_bounds(glyph, position) {
                continue;
            }
            let (key, _, _) = CacheKey::new(
                glyph.font_id,
                glyph.glyph_id,
                glyph.font_size,
                (0.0, 0.0),
                glyph.font_weight,
                glyph.cache_key_flags | CacheKeyFlags::DISABLE_HINTING,
            );
            return Some(PositionedGlyph { key, position });
        }
    }

    fn glyph_content(&mut self, key: &CacheKey) -> Option<GlyphContent<'_>> {
        let font = self.font_system.get_font(key.font_id, key.font_weight)?;
        let font = font.as_swash();
        let mut scaler = self
            .extraction
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
            .scale_outline_into(key.glyph_id, &mut self.extraction.outline)
        {
            return None;
        }
        if key.flags.contains(CacheKeyFlags::FAKE_ITALIC) {
            self.extraction.outline.transform(&Transform::skew(
                Angle::from_degrees(14.0),
                Angle::from_degrees(0.0),
            ));
        }
        outline::collect_events(&self.extraction.outline, &mut self.extraction.events);
        Some(GlyphContent::Outline(&self.extraction.events))
    }
}
