use super::{
    compute_backdrop_capture_region, does_capture_size_exceeds_budget,
    does_capture_size_exceeds_limits, resolve_capture_region_to_viewport, BackdropCaptureRegion,
};
use crate::effect::{BackdropCaptureArea, BackdropEffectConfig};
use crate::renderer::rect_utils::{compute_downsampled_dimensions, logical_rect_to_physical_rect};
use crate::renderer::types::{ClipRectDrawData, DrawCommand};
use crate::vertex::InstanceTransform;
use crate::{MathRect, PhysicalRect, Size, UnsignedPhysicalPoint, UnsignedPhysicalRect};
use lyon::geom::Point;

#[test]
fn capture_dimension_limit_rejects_oversized_regions() {
    let max_capture_dimension = 4_096u32;

    assert!(does_capture_size_exceeds_limits(
        Size::new(max_capture_dimension + 1, 64),
        max_capture_dimension,
    ));
    assert!(!does_capture_size_exceeds_limits(
        Size::new(max_capture_dimension, max_capture_dimension),
        max_capture_dimension,
    ));
}

#[test]
fn capture_texel_budget_rejects_large_dimension_valid_regions() {
    assert!(does_capture_size_exceeds_budget(
        Size::new(1_500, 1_500),
        Size::new(480, 800)
    ));
    assert!(!does_capture_size_exceeds_budget(
        Size::new(480, 800),
        Size::new(480, 800)
    ));
}

#[test]
fn capture_region_offsets_visible_copy_into_transparent_texture() {
    let region = resolve_capture_region_to_viewport(
        PhysicalRect::new(Point::new(-10, 5), Point::new(30, 25)),
        Size::new(100, 100),
    );

    assert_eq!(region.bounds.min, Point::new(-10, 5));
    assert_eq!(
        region.copy_destination_origin,
        UnsignedPhysicalPoint::new(10, 0)
    );
    assert_eq!(region.bounds.size(), Size::new(40, 20).to_i32());
    assert_eq!(
        region.source_rect,
        Some(UnsignedPhysicalRect::new(
            Point::new(0, 5),
            Point::new(30, 25)
        ))
    );
}

#[test]
fn capture_region_skips_copy_when_fully_offscreen() {
    let offscreen_right = resolve_capture_region_to_viewport(
        PhysicalRect::new(Point::new(110, 5), Point::new(130, 25)),
        Size::new(100, 100),
    );
    let offscreen_bottom = resolve_capture_region_to_viewport(
        PhysicalRect::new(Point::new(5, 110), Point::new(25, 130)),
        Size::new(100, 100),
    );

    assert_eq!(offscreen_right.source_rect, None);
    assert_eq!(offscreen_bottom.source_rect, None);
}

#[test]
fn padded_capture_sets_sampling_origin_and_size() {
    let padded_rect =
        MathRect::new(Point::new(100.0, 100.0), Point::new(200.0, 200.0)).inflate(20.0, 20.0);
    let requested_rect =
        logical_rect_to_physical_rect(padded_rect, 1.0).expect("capture rect should be non-empty");
    let capture_region =
        resolve_capture_region_to_viewport(requested_rect, Size::new(1_000, 1_000));
    let sample_transform = capture_region.sample_uniform();

    assert_eq!(capture_region.bounds.size(), Size::new(140, 140).to_i32());
    assert_eq!(sample_transform.capture_origin, [80.0, 80.0]);
    assert_eq!(sample_transform.inverse_capture_size, [1.0 / 140.0; 2]);
}

#[test]
fn node_capture_applies_transform_padding_and_scale_before_viewport_overlap() {
    let draw_command = DrawCommand::ClipRect(ClipRectDrawData::new(
        [(1.0, 2.0), (11.0, 7.0)],
        Some(InstanceTransform::affine_2d(
            -2.0, 0.0, 0.0, 3.0, 20.0, -10.0,
        )),
        true,
    ));

    let capture_region = compute_backdrop_capture_region(
        &draw_command,
        BackdropEffectConfig::new().padding(1.25),
        1.5,
        Size::new(24, 16),
        64,
    )
    .unwrap();

    assert_eq!(
        capture_region,
        BackdropCaptureRegion {
            bounds: PhysicalRect::new(Point::new(-5, -8), Point::new(29, 19)),
            source_rect: Some(UnsignedPhysicalRect::new(
                Point::new(0, 0),
                Point::new(24, 16)
            )),
            copy_destination_origin: UnsignedPhysicalPoint::new(5, 8),
        }
    );
}

#[test]
fn screen_capture_keeps_physical_mapping_when_downsampled() {
    let draw_command = DrawCommand::ClipRect(ClipRectDrawData::new(
        [(10.0, 20.0), (30.0, 40.0)],
        Some(InstanceTransform::translation(50.0, 60.0)),
        true,
    ));
    let config = BackdropEffectConfig::new()
        .capture_area(BackdropCaptureArea::ScreenRect([
            (3.75, 3.25),
            (-2.25, -1.75),
        ]))
        .padding(0.5);
    let full_resolution =
        compute_backdrop_capture_region(&draw_command, config, 2.0, Size::new(100, 80), 128)
            .unwrap();

    assert_eq!(
        full_resolution,
        BackdropCaptureRegion {
            bounds: PhysicalRect::new(Point::new(-6, -5), Point::new(9, 8)),
            source_rect: Some(UnsignedPhysicalRect::new(
                Point::new(0, 0),
                Point::new(9, 8)
            )),
            copy_destination_origin: UnsignedPhysicalPoint::new(6, 5),
        }
    );
    for (downsample, expected_texture_size) in [(0.5, Size::new(8, 7)), (0.01, Size::new(1, 1))] {
        let downsampled_config = config.downsample(downsample);
        let capture_region = compute_backdrop_capture_region(
            &draw_command,
            downsampled_config,
            2.0,
            Size::new(100, 80),
            128,
        )
        .unwrap();

        assert_eq!(capture_region, full_resolution);
        assert_eq!(
            compute_downsampled_dimensions(capture_region.bounds.size().to_u32(), downsample),
            expected_texture_size
        );
        let sampling = capture_region.sample_uniform();
        assert_eq!(sampling.capture_origin, [-6.0, -5.0]);
        assert_eq!(sampling.inverse_capture_size, [1.0 / 15.0, 1.0 / 13.0]);
    }
}

#[test]
fn full_scene_capture_preserves_viewport_mapping_across_scales_and_padding() {
    let draw_command = DrawCommand::ClipRect(ClipRectDrawData::new(
        [(10.0, 20.0), (30.0, 40.0)],
        Some(InstanceTransform::translation(50.0, 60.0)),
        true,
    ));
    for (scale_factor, padding, expected_region) in [
        (
            2.0,
            1.25,
            BackdropCaptureRegion {
                bounds: PhysicalRect::new(Point::new(-3, -3), Point::new(104, 82)),
                source_rect: Some(UnsignedPhysicalRect::new(
                    Point::new(0, 0),
                    Point::new(101, 79),
                )),
                copy_destination_origin: UnsignedPhysicalPoint::new(3, 3),
            },
        ),
        (
            1.1,
            0.0,
            BackdropCaptureRegion {
                bounds: PhysicalRect::new(Point::new(0, 0), Point::new(101, 80)),
                source_rect: Some(UnsignedPhysicalRect::new(
                    Point::new(0, 0),
                    Point::new(101, 79),
                )),
                copy_destination_origin: UnsignedPhysicalPoint::new(0, 0),
            },
        ),
    ] {
        let capture_region = compute_backdrop_capture_region(
            &draw_command,
            BackdropEffectConfig::new()
                .capture_area(BackdropCaptureArea::FullScene)
                .padding(padding),
            scale_factor,
            Size::new(101, 79),
            128,
        )
        .unwrap();

        assert_eq!(capture_region, expected_region);
    }
}

#[test]
fn capture_rejects_invalid_bounds_and_full_resolution_allocation_excesses() {
    let draw_command = DrawCommand::ClipRect(ClipRectDrawData::new(
        [(0.0, 0.0), (10.0, 10.0)],
        None,
        true,
    ));
    let config = BackdropEffectConfig::new().downsample(0.01);
    let accepted_config =
        config.capture_area(BackdropCaptureArea::ScreenRect([(0.0, 0.0), (64.0, 64.0)]));
    assert!(compute_backdrop_capture_region(
        &draw_command,
        accepted_config,
        1.0,
        Size::new(32, 32),
        128
    )
    .is_some());

    for rejected_bounds in [
        [(0.0, 0.0), (129.0, 1.0)],
        [(0.0, 0.0), (65.0, 64.0)],
        [(0.0, 0.0), (0.0, 10.0)],
        [(0.0, 0.0), (f32::INFINITY, 10.0)],
        [(0.0, f32::NAN), (10.0, 10.0)],
        [(0.0, 0.0), (f32::MAX, 10.0)],
    ] {
        let rejected_config = config.capture_area(BackdropCaptureArea::ScreenRect(rejected_bounds));
        assert!(
            compute_backdrop_capture_region(
                &draw_command,
                rejected_config,
                1.0,
                Size::new(32, 32),
                128
            )
            .is_none(),
            "capture must reject {rejected_bounds:?} even when downsampled"
        );
    }
}
