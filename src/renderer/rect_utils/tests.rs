use super::{
    compute_scissor_rect, logical_rect_to_physical_rect, should_skip_visible_rect_draw,
    transform_point_to_logical_screen, transformed_bounds_to_logical_screen_rect,
    try_scissor_for_rect,
};
use crate::effect::EffectInstance;
use crate::gradient::types::{
    ColorInterpolation, Fill, Gradient, GradientStop, GradientStopOffset, LinearGradientDesc,
    LinearGradientLine,
};
use crate::renderer::types::DrawCommand;
use crate::shape::CachedShapeDrawData;
use crate::util::ShapeResources;
use crate::{CachedShapeHandle, Color, Shape, ShapeDrawCommandOptions, Stroke, TransformInstance};
use ahash::{HashMap, HashMapExt};
use lyon::tessellation::FillTessellator;

fn create_test_gradient() -> Gradient {
    Gradient::linear(
        LinearGradientDesc::new(
            LinearGradientLine {
                start: [0.0, 5.0],
                end: [10.0, 5.0],
            },
            [
                GradientStop::at_position(
                    GradientStopOffset::linear_radial(0.0),
                    Color::rgb(255, 0, 0),
                ),
                GradientStop::at_position(
                    GradientStopOffset::linear_radial(1.0),
                    Color::rgb(0, 0, 255),
                ),
            ],
        )
        .with_interpolation(ColorInterpolation::Srgb),
    )
    .expect("valid test gradient")
}

fn rect_draw_command_with_options(options: ShapeDrawCommandOptions) -> DrawCommand {
    let mut tessellator = FillTessellator::new();
    let mut shape_resources = ShapeResources::new();
    let shape_handle = CachedShapeHandle::new(
        &Shape::rect([(0.0, 0.0), (10.0, 10.0)], Stroke::default()),
        &mut tessellator,
        &mut shape_resources,
        None,
    );
    DrawCommand::CachedShape(CachedShapeDrawData::new(shape_handle, &options))
}

fn rect_draw_command() -> DrawCommand {
    rect_draw_command_with_options(ShapeDrawCommandOptions::new())
}

#[test]
fn axis_aligned_rect_transform_accepts_translation_and_scale() {
    let transform = TransformInstance::affine_2d(2.0, 0.0, 0.0, -3.0, 10.0, 20.0);

    let scissor = compute_scissor_rect([(0.0, 0.0), (10.0, 5.0)], Some(transform), 1.0, (100, 100));

    assert_eq!(scissor, Some((10, 5, 20, 15)));
}

#[test]
fn scissor_rejects_non_axis_aligned_transform() {
    let draw_command = rect_draw_command_with_options(
        ShapeDrawCommandOptions::new()
            .transform(TransformInstance::affine_2d(1.0, 0.0, 0.5, 1.0, 5.0, 5.0)),
    );

    assert!(try_scissor_for_rect(&draw_command, 1.0, (100, 100)).is_none());
}

#[test]
fn skip_visible_rect_draw_rejects_effect_nodes() {
    let draw_command = rect_draw_command();
    let node_id = 7usize;

    let mut group_effects = HashMap::new();
    group_effects.insert(
        node_id,
        EffectInstance {
            effect_id: 1,
            params: Vec::new(),
            parameter_resources: None,
            backdrop_config: None,
            backdrop_material_params_buffer: None,
            backdrop_layer_params_buffer: None,
            backdrop_texture_bind_group: None,
            backdrop_texture_id: None,
        },
    );

    assert!(!should_skip_visible_rect_draw(
        node_id,
        &draw_command,
        &group_effects,
        &HashMap::new(),
    ));

    let mut backdrop_effects = HashMap::new();
    backdrop_effects.insert(
        node_id,
        EffectInstance {
            effect_id: 2,
            params: Vec::new(),
            parameter_resources: None,
            backdrop_config: None,
            backdrop_material_params_buffer: None,
            backdrop_layer_params_buffer: None,
            backdrop_texture_bind_group: None,
            backdrop_texture_id: None,
        },
    );

    assert!(!should_skip_visible_rect_draw(
        node_id,
        &draw_command,
        &HashMap::new(),
        &backdrop_effects,
    ));
}

#[test]
fn skip_visible_rect_draw_accepts_untextured_none_color_rect() {
    let draw_command = rect_draw_command();

    assert!(should_skip_visible_rect_draw(
        1,
        &draw_command,
        &HashMap::new(),
        &HashMap::new(),
    ));
}

#[test]
fn skip_visible_rect_draw_rejects_opaque_color_and_textures() {
    let opaque_draw_command =
        rect_draw_command_with_options(ShapeDrawCommandOptions::new().color(Color::WHITE));

    assert!(!should_skip_visible_rect_draw(
        1,
        &opaque_draw_command,
        &HashMap::new(),
        &HashMap::new(),
    ));

    let textured_draw_command =
        rect_draw_command_with_options(ShapeDrawCommandOptions::new().background_texture_id(9));

    assert!(!should_skip_visible_rect_draw(
        2,
        &textured_draw_command,
        &HashMap::new(),
        &HashMap::new(),
    ));
}

#[test]
fn skip_visible_rect_draw_rejects_gradient_rects() {
    let mut draw_command = rect_draw_command();

    match &mut draw_command {
        DrawCommand::CachedShape(shape) => {
            shape.fill = Some(Fill::Gradient(create_test_gradient()));
        }
        _ => unreachable!(),
    }

    assert!(!should_skip_visible_rect_draw(
        3,
        &draw_command,
        &HashMap::new(),
        &HashMap::new(),
    ));
}

#[test]
fn physical_capture_rect_preserves_requested_size_outside_viewport() {
    let requested_rect = logical_rect_to_physical_rect([(-10.0, 5.0), (30.0, 25.0)], 1.0)
        .expect("capture rect should be non-empty");

    assert_eq!(requested_rect, (-10, 5, 40, 20));
}

#[test]
fn transform_point_to_logical_screen_preserves_negative_w_sign() {
    let transform = TransformInstance {
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
    let requested_rect = logical_rect_to_physical_rect([(0.0, 0.0), (f32::INFINITY, 25.0)], 1.0);

    assert!(requested_rect.is_none());
}

#[test]
fn transformed_capture_bounds_include_all_four_corners() {
    let transform = TransformInstance::affine_2d(1.0, 0.5, -0.25, 1.0, 12.0, -4.0);

    assert_eq!(
        transformed_bounds_to_logical_screen_rect([(0.0, 0.0), (20.0, 10.0)], Some(transform)),
        [(9.5, -4.0), (32.0, 16.0)]
    );
}
