use super::{should_skip_visible_draw, try_scissor_for_rect};
use crate::commands::{
    IntermediateTextureId, ShapeTextureBinding, ShapeTextureLayer, TextureSampling,
};
use crate::core::effect::ContentEffectConfig;
use crate::core::gradient::types::{
    ColorInterpolation, Fill, Gradient, GradientStop, GradientStopOffset, LinearGradientDesc,
    LinearGradientLine,
};
use crate::core::shape::ShapeResources;
use crate::planner::draws::shape_material;
use crate::scene_state::content_groups::ContentGroup;
use crate::scene_state::effects::parameters::{EffectParameterRange, EffectParametersHandle};
use crate::scene_state::effects::ContentEffectInstance;
use crate::scene_state::types::CachedShapeDrawData;
use crate::scene_state::types::DrawTreeNode;
use crate::{CachedShapeHandle, Color, Shape, ShapeDrawOptions, Size, TransformInstance, Viewport};
use ahash::{HashMap, HashMapExt};

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

fn rect_draw_tree_node_with_options(options: ShapeDrawOptions) -> DrawTreeNode {
    let mut shape_resources = ShapeResources::new();
    let shape_handle = CachedShapeHandle::new(
        &Shape::rect([(0.0, 0.0), (10.0, 10.0)]),
        &mut shape_resources,
        None,
    )
    .unwrap();
    DrawTreeNode::CachedShape(CachedShapeDrawData::new(shape_handle, options))
}

fn rect_draw_tree_node() -> DrawTreeNode {
    rect_draw_tree_node_with_options(ShapeDrawOptions::new())
}

fn should_skip(node_id: usize, node: &DrawTreeNode, groups: &HashMap<usize, ContentGroup>) -> bool {
    let material = match node {
        DrawTreeNode::CachedShape(description) => Some(shape_material(&description.instance)),
        _ => None,
    };
    should_skip_visible_draw(node_id, node, material, groups)
}

#[test]
fn scissor_rejects_non_axis_aligned_transform() {
    let draw_tree_node = rect_draw_tree_node_with_options(
        ShapeDrawOptions::new()
            .transform(TransformInstance::affine_2d(1.0, 0.0, 0.5, 1.0, 5.0, 5.0)),
    );

    assert!(try_scissor_for_rect(&draw_tree_node, None, 1.0, Size::new(100, 100)).is_none());
}

#[test]
fn skip_visible_draw_rejects_content_effect_nodes() {
    let draw_tree_node = rect_draw_tree_node();
    let node_id = 7usize;

    let mut content_groups = HashMap::new();
    content_groups.insert(
        node_id,
        ContentGroup::new(
            Some(ContentEffectInstance {
                effect_id: 1,
                parameters: EffectParametersHandle {
                    range: EffectParameterRange { start: 0, end: 0 },
                    hash: 0,
                },
                config: ContentEffectConfig::default(),
            }),
            &draw_tree_node,
            None,
            Viewport {
                physical_size: (100, 100),
                scale_factor: 1.0,
            },
            4096,
        ),
    );

    assert!(!should_skip(node_id, &draw_tree_node, &content_groups,));
}

#[test]
fn skip_visible_draw_accepts_untextured_none_color_rect() {
    let draw_tree_node = rect_draw_tree_node();

    assert!(should_skip(1, &draw_tree_node, &HashMap::new(),));
}

#[test]
fn skip_visible_draw_respects_solid_fill_opacity_and_textures() {
    for alpha in [0, 1, 128, 255] {
        let draw_tree_node = rect_draw_tree_node_with_options(
            ShapeDrawOptions::new().color(Color::rgba(101, 81, 226, alpha)),
        );
        assert_eq!(should_skip(1, &draw_tree_node, &HashMap::new()), alpha == 0);
    }

    let textured_draw_tree_node =
        rect_draw_tree_node_with_options(ShapeDrawOptions::new().background_texture_id(9));

    assert!(!should_skip(2, &textured_draw_tree_node, &HashMap::new(),));
}

#[test]
fn positive_float_alpha_keeps_the_rectangle_draw() {
    let draw_tree_node = rect_draw_tree_node_with_options(
        ShapeDrawOptions::new().color(Color::from_srgba_f32(1.0, 0.0, 0.0, 0.001)),
    );

    assert!(!should_skip(1, &draw_tree_node, &HashMap::new()));
}

#[test]
fn skip_visible_draw_rejects_gradient_rects() {
    let mut draw_tree_node = rect_draw_tree_node();

    match &mut draw_tree_node {
        DrawTreeNode::CachedShape(shape) => {
            shape.instance.fill = Some(Fill::Gradient(create_test_gradient()));
        }
        _ => unreachable!(),
    }

    assert!(!should_skip(3, &draw_tree_node, &HashMap::new(),));
}

#[test]
fn texture_below_transparent_fill_keeps_the_rectangle_draw() {
    let node = rect_draw_tree_node();
    let DrawTreeNode::CachedShape(description) = &node else {
        unreachable!()
    };
    let mut material = shape_material(&description.instance);
    material.under_fill_texture = Some(ShapeTextureLayer {
        texture: ShapeTextureBinding::Intermediate(IntermediateTextureId::Planned(0)),
        sampling: TextureSampling::ShapeUv,
    });
    assert!(!should_skip_visible_draw(
        1,
        &node,
        Some(material),
        &HashMap::new()
    ));
}
