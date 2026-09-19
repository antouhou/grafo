use super::{
    capture_size_exceeds_budget, capture_size_exceeds_limits, inflate_logical_rect,
    logical_rect_to_physical_capture_rect, resolve_capture_region_to_viewport,
    transform_point_to_logical_screen,
};
use crate::vertex::InstanceTransform;

#[test]
fn physical_capture_rect_preserves_requested_size_outside_viewport() {
    let requested_rect = logical_rect_to_physical_capture_rect([(-10.0, 5.0), (30.0, 25.0)], 1.0)
        .expect("capture rect should be non-empty");

    assert_eq!(requested_rect, (-10, 5, 40, 20));
}

#[test]
fn transform_point_to_logical_screen_preserves_negative_w_sign() {
    let transform = InstanceTransform {
        col0: [2.0, 0.0, 0.0, 0.0],
        col1: [0.0, 3.0, 0.0, 0.0],
        col2: [0.0, 0.0, 1.0, 0.0],
        col3: [0.0, 0.0, 0.0, -2.0],
    };

    let point = transform_point_to_logical_screen((1.0, 1.0), Some(transform));

    assert_eq!(point, (-1.0, -1.5));
}

#[test]
fn physical_capture_rect_rejects_non_finite_coordinates() {
    let requested_rect =
        logical_rect_to_physical_capture_rect([(0.0, 0.0), (f32::INFINITY, 25.0)], 1.0);

    assert!(requested_rect.is_none());
}

#[test]
fn capture_size_exceeds_limits_rejects_oversized_regions() {
    let max_capture_dimension = 4_096u32;

    assert!(capture_size_exceeds_limits(
        (max_capture_dimension + 1, 64),
        max_capture_dimension,
    ));
    assert!(!capture_size_exceeds_limits(
        (max_capture_dimension, max_capture_dimension),
        max_capture_dimension,
    ));
}

#[test]
fn capture_size_exceeds_budget_rejects_large_dimension_valid_regions() {
    assert!(capture_size_exceeds_budget((1_500, 1_500), (480, 800)));
    assert!(!capture_size_exceeds_budget((480, 800), (480, 800)));
}

#[test]
fn capture_region_offsets_visible_copy_into_transparent_texture() {
    let region = resolve_capture_region_to_viewport((-10, 5, 40, 20), (100, 100));

    assert_eq!(region.capture_origin, (-10, 5));
    assert_eq!(region.capture_size, (40, 20));
    assert_eq!(region.copy_source_origin, Some((0, 5)));
    assert_eq!(region.copy_destination_origin, (10, 0));
    assert_eq!(region.copy_size, (30, 20));
}

#[test]
fn capture_region_skips_copy_when_fully_offscreen() {
    let offscreen_right = resolve_capture_region_to_viewport((110, 5, 20, 20), (100, 100));
    let offscreen_bottom = resolve_capture_region_to_viewport((5, 110, 20, 20), (100, 100));

    assert_eq!(offscreen_right.copy_source_origin, None);
    assert_eq!(offscreen_right.copy_size, (0, 20));
    assert_eq!(offscreen_bottom.copy_source_origin, None);
    assert_eq!(offscreen_bottom.copy_size, (20, 0));
}

#[test]
fn padded_capture_sets_sampling_origin_and_size() {
    let padded_rect = inflate_logical_rect([(100.0, 100.0), (200.0, 200.0)], 20.0);
    let requested_rect = logical_rect_to_physical_capture_rect(padded_rect, 1.0)
        .expect("capture rect should be non-empty");
    let capture_region = resolve_capture_region_to_viewport(requested_rect, (1_000, 1_000));
    let sample_transform = capture_region.sample_uniform();

    assert_eq!(capture_region.capture_size, (140, 140));
    assert_eq!(sample_transform.capture_origin, [80.0, 80.0]);
    assert_eq!(sample_transform.inverse_capture_size, [1.0 / 140.0; 2]);
}
