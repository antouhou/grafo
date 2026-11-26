use grafo::cosmic_text::{fontdb, Align, Attrs, Buffer, Family, FontSystem, Metrics, Shaping};
use std::sync::Arc;

pub fn load() -> FontSystem {
    let mut database = fontdb::Database::new();
    database.load_font_source(fontdb::Source::Binary(Arc::new(
        include_bytes!("../assets/Inter.ttf").as_slice(),
    )));
    FontSystem::new_with_locale_and_db("en-US".to_owned(), database)
}

pub fn set_text(fonts: &mut FontSystem, buffer: &mut Buffer, text: &str, alignment: Option<Align>) {
    buffer.set_text(
        text,
        &Attrs::new().family(Family::Name("Inter")),
        Shaping::Advanced,
        alignment,
    );
    buffer.shape_until_scroll(fonts, false);
}

pub fn shape(fonts: &mut FontSystem, text: &str, size: f32, line_height: f32) -> Buffer {
    let mut buffer = Buffer::new(fonts, Metrics::new(size, line_height));
    set_text(fonts, &mut buffer, text, None);
    buffer
}
