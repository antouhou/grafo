use super::Camera;

fn artwork_point(camera: &Camera, screen: (f64, f64)) -> (f64, f64) {
    (
        (screen.0 - camera.offset.0) / camera.scale,
        (screen.1 - camera.offset.1) / camera.scale,
    )
}

fn assert_same_point(actual: (f64, f64), expected: (f64, f64)) {
    assert!((actual.0 - expected.0).abs() < 1e-8);
    assert!((actual.1 - expected.1).abs() < 1e-8);
}

#[test]
fn zoom_after_pan_preserves_the_cursor_anchor_at_both_limits() {
    let mut camera = Camera::new((-120.0, 40.0, 550.0, 570.0), (960.0, 800.0));
    camera.pan((83.0, -39.0));
    let cursor = (173.0, 317.0);
    let anchor = artwork_point(&camera, cursor);
    for scroll in [2.0, 1000.0, -1000.0, 3.0] {
        camera.zoom(cursor, scroll);
        assert_same_point(artwork_point(&camera, cursor), anchor);
    }
}

#[test]
fn resizing_preserves_the_artwork_at_the_viewport_center() {
    let mut camera = Camera::new((30.0, -200.0, 550.0, 570.0), (960.0, 800.0));
    camera.pan((-112.0, 49.0));
    camera.zoom((245.0, 344.0), 4.0);
    let anchor = artwork_point(&camera, (480.0, 400.0));
    camera.resize((720.0, 600.0));
    assert_same_point(artwork_point(&camera, (360.0, 300.0)), anchor);
}
