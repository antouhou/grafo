use glyphon::cosmic_text::{Attrs, Buffer, Family, FontSystem, Metrics, Scroll, Shaping, Wrap};

fn main() {
    let mut font_system = FontSystem::new();
    let metrics = Metrics::new(14.0, 20.0);

    let mut buffer = Buffer::new(&mut font_system, metrics);

    let mut buffer = buffer.borrow_with(&mut font_system);

    buffer.set_wrap(Wrap::None);
    buffer.set_size(Some(100.0), Some(50.0));

    buffer.set_text(
        "Lorem ipsum dolor sit amet, qui minim labore adipisicing minim sint cillum sint consectetur cupidatat.",
        Attrs::new().family(Family::SansSerif),
        Shaping::Advanced
    );

    let scroll = Scroll::new(0, 15.0, 200.0);
    buffer.set_scroll(scroll);

    // Perform shaping as desired
    buffer.shape_until_scroll(false);

    for run in buffer.layout_runs() {
        println!("Layout run: {:?}", run);
    }
}
