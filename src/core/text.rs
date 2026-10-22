use super::{Fill, MathRect, TransformInstance};
use lyon::path::{FillRule, PathEvent};
use std::{hash::Hash, ops::Range};

/// A glyph placed in logical, y-down text coordinates, including shaping offsets
#[derive(Debug, Clone, Copy)]
pub struct PositionedGlyph<Key> {
    pub key: Key,
    pub position: (f32, f32),
}

/// Borrowed glyph data in logical, y-down coordinates relative to the glyph origin.
/// Font size and variations have already been applied
/// TODO: implement bitmap and color-vector drawing
pub enum GlyphContent<'a> {
    /// Complete contours
    Outline(&'a [PathEvent]),
    ColorVector(ColorGlyph<'a>),
    Bitmap(GlyphBitmap<'a>),
}

pub struct ColorGlyph<'a> {
    /// Contiguous paths referenced by ranges in `commands`.
    pub path_events: &'a [PathEvent],
    /// Ordered drawing operations. Transform, clip, and layer pushes must be balanced
    pub commands: &'a [GlyphPaintCommand],
}

pub struct GlyphBitmap<'a> {
    /// Tightly packed RGBA8, premultiplied in linear space and then encoded as sRGB.
    pub pixels: &'a [u8],
    pub pixel_size: (u32, u32),
    /// Logical bounds relative to the glyph origin, independent of pixel dimensions.
    pub bounds: [(f32, f32); 2],
}

// Keep paints inline to avoid a heap allocation per drawing operation.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum GlyphPaint {
    TextFill { opacity: f32 },
    Intrinsic(Fill),
}

// Keep paints inline to avoid a heap allocation per drawing operation.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum GlyphPaintCommand {
    PushTransform(TransformInstance),
    PopTransform,
    PushClip {
        /// A range in `ColorGlyph::path_events`, containing complete Begin/End contours.
        /// Several commands can reference the same range.
        path_events: Range<usize>,
        fill_rule: FillRule,
    },
    PopClip,
    /// Paints the current clip using the current transform.
    Fill(GlyphPaint),
    /// Starts a transparent layer, composited onto its parent on `PopLayer`.
    PushLayer(CompositeMode),
    PopLayer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeMode {
    Clear,
    Source,
    Destination,
    SourceOver,
    DestinationOver,
    SourceIn,
    DestinationIn,
    SourceOut,
    DestinationOut,
    SourceAtop,
    DestinationAtop,
    Xor,
    Plus,
    Screen,
    Overlay,
    Darken,
    Lighten,
    ColorDodge,
    ColorBurn,
    HardLight,
    SoftLight,
    Difference,
    Exclusion,
    Multiply,
    HslHue,
    HslSaturation,
    HslColor,
    HslLuminosity,
}

/// A cursor over already positioned glyphs. Grafo never does anything related to layout; That's
/// responsibility of the caller.
pub trait TextSource {
    /// Identifies glyph content, including font, size, variations, and palette.
    /// Placement is excluded. Keys must not alias different content within the source.
    type GlyphKey: Eq + Hash;

    /// Called before iteration with bounds of a parent shape, if any. `None` means no bounds - all
    /// glyphs. An empty rectangle has no visible glyphs.
    /// Sources may omit glyphs wholly outside the bounds, or ignore this hint. This is purely for
    /// optimization purposes, so we can skip processing for glyphs that are not visible anyway.
    fn set_visible_bounds(&mut self, _bounds: Option<MathRect>) {}

    fn next_glyph(&mut self) -> Option<PositionedGlyph<Self::GlyphKey>>;

    /// Should return the actual draw data for the glyph. `None` draws nothing.
    /// Callers must not depend on this being called for every positioned glyph - once the glyph
    /// is cached, this method won't be called for the same glyph again
    fn glyph_content(&mut self, key: &Self::GlyphKey) -> Option<GlyphContent<'_>>;
}

impl<Source: TextSource + ?Sized> TextSource for &mut Source {
    type GlyphKey = Source::GlyphKey;

    fn set_visible_bounds(&mut self, bounds: Option<MathRect>) {
        (**self).set_visible_bounds(bounds);
    }

    fn next_glyph(&mut self) -> Option<PositionedGlyph<Self::GlyphKey>> {
        (**self).next_glyph()
    }

    fn glyph_content(&mut self, key: &Self::GlyphKey) -> Option<GlyphContent<'_>> {
        (**self).glyph_content(key)
    }
}
