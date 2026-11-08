use super::{
    clip, rect, shape_description, snapshot, DrawPlanner, Operation, RecordedCommands, Scene,
};
use crate::commands::ShapeDrawId;
use crate::core::util::ShapeResources;
use crate::scene_state::types::DrawTreeNode;
use crate::{BorderRadii, Color, LineCap, LineJoin, Shape, StrokeInstance, StrokeOptions};

fn stroked_shape(shape: Shape, options: StrokeOptions) -> DrawTreeNode {
    let tessellation = shape
        .tessellate_stroke(options, &mut ShapeResources::new())
        .unwrap();
    let mut draw = shape_description(shape, true);
    draw.set_stroke(StrokeInstance {
        tessellation,
        paint: Color::WHITE.into(),
        cache_key: None,
    });
    DrawTreeNode::CachedShape(draw)
}

#[test]
fn offscreen_lines_keep_visible_caps_and_miter_tips() {
    let line = Shape::builder()
        .begin((-20.0, 20.0))
        .line_to((-2.0, 20.0))
        .end()
        .build();
    let corner = Shape::builder()
        .begin((-8.0, -2.0))
        .line_to((-2.0, -2.0))
        .line_to((-2.0, -8.0))
        .end()
        .build();
    for (shape, options, should_draw) in [
        (line.clone(), StrokeOptions::new(8.0), false),
        (
            line.clone(),
            StrokeOptions::new(8.0).line_cap(LineCap::Square),
            true,
        ),
        (line, StrokeOptions::new(8.0).line_cap(LineCap::Round), true),
        (
            corner,
            StrokeOptions::new(8.0).line_join(LineJoin::Miter),
            true,
        ),
    ] {
        let mut scene = Scene::new();
        let root = scene.add(None, clip((0.0, 0.0), (100.0, 100.0)));
        let node = scene.add(Some(root), stroked_shape(shape, options));
        let mut output = RecordedCommands::default();
        scene.plan(&mut DrawPlanner::default(), &mut output);
        if should_draw {
            assert!(snapshot(&output).contains(&(
                Operation::Draw(ShapeDrawId(node).stroke()),
                0,
                rect((0, 0), (100, 100))
            )),);
        } else {
            assert!(output.instructions.is_empty());
        }
    }
}

#[test]
fn stencil_clip_is_installed_after_fill_and_stroke() {
    let mut scene = Scene::new();
    let root = scene.add(None, clip((10.0, 10.0), (90.0, 90.0)));
    let parent = scene.add(
        Some(root),
        stroked_shape(
            Shape::rounded_rect([(20.0, 20.0), (80.0, 80.0)], BorderRadii::new(8.0)),
            StrokeOptions::new(8.0),
        ),
    );
    let child = scene.add(
        Some(parent),
        DrawTreeNode::CachedShape(shape_description(
            Shape::rect([(10.0, 10.0), (90.0, 90.0)]),
            true,
        )),
    );
    let sibling = scene.add(
        Some(root),
        DrawTreeNode::CachedShape(shape_description(
            Shape::rect([(30.0, 30.0), (70.0, 70.0)]),
            true,
        )),
    );
    let mut output = RecordedCommands::default();
    scene.plan(&mut DrawPlanner::default(), &mut output);
    let scissor = rect((10, 10), (90, 90));
    assert_eq!(
        snapshot(&output),
        [
            (Operation::Draw(ShapeDrawId(parent)), 0, scissor),
            (Operation::Draw(ShapeDrawId(parent).stroke()), 0, scissor),
            (Operation::Increment(ShapeDrawId(parent)), 0, scissor),
            (Operation::Draw(ShapeDrawId(child)), 1, scissor),
            (Operation::Decrement(ShapeDrawId(parent)), 1, scissor),
            (Operation::Draw(ShapeDrawId(sibling)), 0, scissor),
        ],
    );
}

#[test]
fn rectangle_stroke_uses_inherited_scissor_and_children_use_fill_bounds() {
    let mut scene = Scene::new();
    let root = scene.add(None, clip((10.0, 10.0), (90.0, 90.0)));
    let parent = scene.add(
        Some(root),
        stroked_shape(
            Shape::rect([(20.0, 20.0), (80.0, 80.0)]),
            StrokeOptions::new(8.0),
        ),
    );
    let child = scene.add(
        Some(parent),
        DrawTreeNode::CachedShape(shape_description(
            Shape::rect([(10.0, 10.0), (90.0, 90.0)]),
            true,
        )),
    );
    let mut output = RecordedCommands::default();
    scene.plan(&mut DrawPlanner::default(), &mut output);
    let inherited = rect((10, 10), (90, 90));
    let child_clip = rect((20, 20), (80, 80));
    assert_eq!(
        snapshot(&output),
        [
            (Operation::Draw(ShapeDrawId(parent)), 0, child_clip),
            (Operation::Draw(ShapeDrawId(parent).stroke()), 0, inherited),
            (Operation::Draw(ShapeDrawId(child)), 0, child_clip),
        ],
    );
}
