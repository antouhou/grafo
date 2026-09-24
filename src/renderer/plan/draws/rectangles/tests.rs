use super::{should_skip_visible_rect_draw, try_scissor_for_rect};
use crate::core::effect::{BackdropEffectConfig, BackdropEffectInstance, EffectInstance};
use crate::core::gradient::types::{
    ColorInterpolation, Fill, Gradient, GradientStop, GradientStopOffset, LinearGradientDesc,
    LinearGradientLine,
};
use crate::core::util::ShapeResources;
use crate::renderer::types::CachedShapeDrawData;
use crate::renderer::types::DrawTreeNode;
use crate::{
    CachedShapeHandle, Color, Shape, ShapeDrawCommandOptions, Size, Stroke, TransformInstance,
};
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

fn rect_draw_tree_node_with_options(options: ShapeDrawCommandOptions) -> DrawTreeNode {
    let mut tessellator = FillTessellator::new();
    let mut shape_resources = ShapeResources::new();
    let shape_handle = CachedShapeHandle::new(
        &Shape::rect([(0.0, 0.0), (10.0, 10.0)], Stroke::default()),
        &mut tessellator,
        &mut shape_resources,
        None,
    );
    DrawTreeNode::CachedShape(CachedShapeDrawData::new(shape_handle, &options))
}

fn rect_draw_tree_node() -> DrawTreeNode {
    rect_draw_tree_node_with_options(ShapeDrawCommandOptions::new())
}

#[test]
fn scissor_rejects_non_axis_aligned_transform() {
    let draw_tree_node = rect_draw_tree_node_with_options(
        ShapeDrawCommandOptions::new()
            .transform(TransformInstance::affine_2d(1.0, 0.0, 0.5, 1.0, 5.0, 5.0)),
    );

    assert!(try_scissor_for_rect(&draw_tree_node, 1.0, Size::new(100, 100)).is_none());
}

#[test]
fn skip_visible_rect_draw_rejects_effect_nodes() {
    let draw_tree_node = rect_draw_tree_node();
    let node_id = 7usize;

    let mut group_effects = HashMap::new();
    group_effects.insert(
        node_id,
        EffectInstance {
            effect_id: 1,
            params: Vec::new(),
        },
    );

    assert!(!should_skip_visible_rect_draw(
        node_id,
        &draw_tree_node,
        &group_effects,
        &HashMap::new(),
    ));

    let mut backdrop_effects = HashMap::new();
    backdrop_effects.insert(
        node_id,
        BackdropEffectInstance::new(
            EffectInstance {
                effect_id: 2,
                params: Vec::new(),
            },
            BackdropEffectConfig::default(),
        ),
    );

    assert!(!should_skip_visible_rect_draw(
        node_id,
        &draw_tree_node,
        &HashMap::new(),
        &backdrop_effects,
    ));
}

#[test]
fn skip_visible_rect_draw_accepts_untextured_none_color_rect() {
    let draw_tree_node = rect_draw_tree_node();

    assert!(should_skip_visible_rect_draw(
        1,
        &draw_tree_node,
        &HashMap::new(),
        &HashMap::new(),
    ));
}

#[test]
fn skip_visible_rect_draw_rejects_opaque_color_and_textures() {
    let opaque_draw_tree_node =
        rect_draw_tree_node_with_options(ShapeDrawCommandOptions::new().color(Color::WHITE));

    assert!(!should_skip_visible_rect_draw(
        1,
        &opaque_draw_tree_node,
        &HashMap::new(),
        &HashMap::new(),
    ));

    let textured_draw_tree_node =
        rect_draw_tree_node_with_options(ShapeDrawCommandOptions::new().background_texture_id(9));

    assert!(!should_skip_visible_rect_draw(
        2,
        &textured_draw_tree_node,
        &HashMap::new(),
        &HashMap::new(),
    ));
}

#[test]
fn skip_visible_rect_draw_rejects_gradient_rects() {
    let mut draw_tree_node = rect_draw_tree_node();

    match &mut draw_tree_node {
        DrawTreeNode::CachedShape(shape) => {
            shape.fill = Some(Fill::Gradient(create_test_gradient()));
        }
        _ => unreachable!(),
    }

    assert!(!should_skip_visible_rect_draw(
        3,
        &draw_tree_node,
        &HashMap::new(),
        &HashMap::new(),
    ));
}
