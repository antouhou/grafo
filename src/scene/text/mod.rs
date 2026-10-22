use crate::core::{Shape, ShapeBuilder};
use cosmic_text::{Buffer, CacheKey, CacheKeyFlags, Command, FontSystem, SwashCache};

fn append_outline(
    mut builder: ShapeBuilder,
    commands: &[Command],
    origin: (f32, f32),
) -> ShapeBuilder {
    let position = |x, y| (origin.0 + x, origin.1 - y);
    for command in commands {
        builder = match *command {
            Command::MoveTo(point) => builder.begin(position(point.x, point.y)),
            Command::LineTo(point) => builder.line_to(position(point.x, point.y)),
            Command::QuadTo(control, point) => builder
                .quadratic_bezier_to(position(control.x, control.y), position(point.x, point.y)),
            Command::CurveTo(first, second, point) => builder.cubic_bezier_to(
                position(first.x, first.y),
                position(second.x, second.y),
                position(point.x, point.y),
            ),
            Command::Close => builder.close(),
        };
    }
    builder
}

pub(super) fn buffer_shape(buffer: &Buffer, font_system: &mut FontSystem) -> Shape {
    let mut cache = SwashCache::new();
    let mut builder = Shape::builder();
    for run in buffer.layout_runs() {
        for glyph in run.glyphs {
            let (key, _, _) = CacheKey::new(
                glyph.font_id,
                glyph.glyph_id,
                glyph.font_size,
                (0.0, 0.0),
                glyph.font_weight,
                glyph.cache_key_flags | CacheKeyFlags::DISABLE_HINTING,
            );
            let Some(commands) = cache.get_outline_commands(font_system, key) else {
                continue;
            };
            let origin = (
                glyph.x + glyph.font_size * glyph.x_offset,
                run.line_y + glyph.y - glyph.font_size * glyph.y_offset,
            );
            builder = append_outline(builder, commands, origin);
        }
    }
    builder.build()
}

#[cfg(test)]
mod tests;
