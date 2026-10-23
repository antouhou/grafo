use super::build_text_shape;
use crate::core::util::ShapeResources;
use crate::{CosmicText, Shape};
use cosmic_text::{Attrs, Buffer, Family, FontSystem, Metrics, Scroll, Shaping};
use lyon::tessellation::FillTessellator;

fn buffer_shape(buffer: &Buffer, fonts: &mut FontSystem) -> Shape {
    build_text_shape(&mut CosmicText::new(buffer, fonts))
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
    let mut tessellator = FillTessellator::new();
    let mut resources = ShapeResources::new();
    let original =
        buffer_shape(&buffer, &mut fonts).tessellate(&mut tessellator, &mut resources, None);
    assert!(!original.vertex_buffers.vertices.is_empty());
    buffer.set_scroll(Scroll {
        vertical: 0.375,
        ..Scroll::default()
    });
    buffer.shape_until_scroll(&mut fonts, false);
    let scrolled =
        buffer_shape(&buffer, &mut fonts).tessellate(&mut tessellator, &mut resources, None);
    for (before, after) in original.local_bounds.into_iter().zip(scrolled.local_bounds) {
        assert!((before.0 - after.0).abs() < 0.001);
        assert!((before.1 - 0.375 - after.1).abs() < 0.001);
    }
    let (mut fonts, buffer) = text("O");
    let single =
        buffer_shape(&buffer, &mut fonts).tessellate(&mut tessellator, &mut resources, None);
    assert_eq!(original.local_bounds[0], single.local_bounds[0]);
    assert!((original.local_bounds[1].1 - single.local_bounds[1].1 - 32.0).abs() < 0.001);
    let [(left, top), (right, bottom)] = single.local_bounds;
    let center = [(left + right) / 2.0, (top + bottom) / 2.0];
    for triangle in single.vertex_buffers.indices.as_chunks::<3>().0 {
        let points = triangle.map(|index| single.vertex_buffers.vertices[index as usize].position);
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
        let geometry = buffer_shape(&buffer, &mut fonts).tessellate(
            &mut FillTessellator::new(),
            &mut ShapeResources::new(),
            None,
        );
        assert!(geometry.vertex_buffers.vertices.is_empty());
        assert!(geometry.vertex_buffers.indices.is_empty());
    }
}
