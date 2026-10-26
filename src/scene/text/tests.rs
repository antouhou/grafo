use crate::core::{CachedShapeHandle, MathRect, TextSource, Viewport};
use crate::scene_state::SceneState;
use crate::CosmicText;
use cosmic_text::{Attrs, Buffer, Family, FontSystem, Metrics, Scroll, Shaping};
use std::sync::Arc;

fn buffer_glyphs(buffer: &Buffer, fonts: &mut FontSystem) -> Vec<(CachedShapeHandle, (f32, f32))> {
    let mut state = SceneState::new(
        Default::default(),
        Viewport {
            physical_size: (256, 256),
            scale_factor: 1.0,
        },
    );
    let mut source = CosmicText::new(buffer, fonts);
    let mut glyphs = Vec::new();
    while let Some(glyph) = source.next_glyph() {
        if let Some(key) = state.load_glyph(&mut source, &glyph.key) {
            glyphs.push((state.loaded_shape(key).unwrap(), glyph.position));
        }
    }
    glyphs
}

fn bounds(glyphs: &[(CachedShapeHandle, (f32, f32))]) -> MathRect {
    glyphs
        .iter()
        .map(|(shape, position)| {
            let [min, max] = shape.local_bounds();
            MathRect::new(
                (min.0 + position.0, min.1 + position.1).into(),
                (max.0 + position.0, max.1 + position.1).into(),
            )
        })
        .reduce(|bounds, glyph| bounds.union(&glyph))
        .unwrap()
}

fn text(text: &str) -> (FontSystem, Buffer) {
    let mut fonts = FontSystem::new();
    let mut buffer = Buffer::new(&mut fonts, Metrics::new(24.0, 32.0));
    buffer.set_text(
        text,
        &Attrs::new().family(Family::SansSerif),
        Shaping::Advanced,
        None,
    );
    buffer.shape_until_scroll(&mut fonts, false);
    (fonts, buffer)
}

#[test]
fn outlines_preserve_multiline_baselines_and_fractional_scroll() {
    let (mut fonts, mut buffer) = text("O\nO");
    let original = buffer_glyphs(&buffer, &mut fonts);
    assert_eq!(original.len(), 2);
    assert!(Arc::ptr_eq(
        &original[0].0.tessellation,
        &original[1].0.tessellation
    ));
    buffer.set_scroll(Scroll {
        vertical: 0.375,
        ..Scroll::default()
    });
    buffer.shape_until_scroll(&mut fonts, false);
    let scrolled = buffer_glyphs(&buffer, &mut fonts);
    assert!((bounds(&original).min.y - 0.375 - bounds(&scrolled).min.y).abs() < 0.001);
    assert!((bounds(&original).max.y - 0.375 - bounds(&scrolled).max.y).abs() < 0.001);
    let (mut fonts, buffer) = text("O");
    let single = buffer_glyphs(&buffer, &mut fonts);
    assert_eq!(bounds(&original).min, bounds(&single).min);
    assert!((bounds(&original).max.y - bounds(&single).max.y - 32.0).abs() < 0.001);
    let single = &single[0].0;
    let [(left, top), (right, bottom)] = single.local_bounds();
    let center = [(left + right) / 2.0, (top + bottom) / 2.0];
    for triangle in single.vertex_buffers().indices.as_chunks::<3>().0 {
        let points =
            triangle.map(|index| single.vertex_buffers().vertices[index as usize].position);
        let crosses = [0, 1, 2].map(|index| {
            let a = points[index];
            let b = points[(index + 1) % 3];
            (b[0] - a[0]) * (center[1] - a[1]) - (b[1] - a[1]) * (center[0] - a[0])
        });
        assert!(
            crosses.iter().any(|&value| value < 0.0) && crosses.iter().any(|&value| value > 0.0),
            "the counter of O must remain empty"
        );
    }
}

#[test]
fn empty_and_whitespace_buffers_have_no_geometry() {
    for value in ["", " \t\n "] {
        let (mut fonts, buffer) = text(value);
        assert!(buffer_glyphs(&buffer, &mut fonts).is_empty());
    }
}
