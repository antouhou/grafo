use super::{
    clip, rect, shape_description, snapshot, DrawPlanner, Operation, RecordedCommands, Scene,
};
use crate::commands::ShapeDrawId;
use crate::scene_state::types::DrawTreeNode;
use crate::{LineCap, LineJoin, Shape, StrokeOptions};

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
    for (stroke, should_draw) in [
        (line.clone().stroke(StrokeOptions::new(8.0)), false),
        (
            line.clone()
                .stroke(StrokeOptions::new(8.0).line_cap(LineCap::Square)),
            true,
        ),
        (
            line.stroke(StrokeOptions::new(8.0).line_cap(LineCap::Round)),
            true,
        ),
        (
            corner.stroke(StrokeOptions::new(8.0).line_join(LineJoin::Miter)),
            true,
        ),
    ] {
        let mut scene = Scene::new();
        let root = scene.add(None, clip((0.0, 0.0), (100.0, 100.0)));
        let node = scene.add(
            Some(root),
            DrawTreeNode::CachedShape(shape_description(stroke, true)),
        );
        let mut output = RecordedCommands::default();
        scene.plan(&mut DrawPlanner::default(), &mut output);
        if should_draw {
            assert_eq!(
                snapshot(&output),
                [(
                    Operation::Draw(ShapeDrawId(node)),
                    0,
                    rect((0, 0), (100, 100))
                )],
            );
        } else {
            assert!(output.instructions.is_empty());
        }
    }
}

#[test]
fn stroked_rectangle_parents_clip_children_with_their_outline() {
    let mut scene = Scene::new();
    let root = scene.add(None, clip((10.0, 10.0), (90.0, 90.0)));
    let outline = scene.add(
        Some(root),
        DrawTreeNode::CachedShape(shape_description(
            Shape::rect([(20.0, 20.0), (80.0, 80.0)]).stroke(StrokeOptions::new(8.0)),
            true,
        )),
    );
    let child = scene.add(
        Some(outline),
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
            (
                Operation::DrawAndIncrement(ShapeDrawId(outline)),
                0,
                scissor
            ),
            (Operation::Draw(ShapeDrawId(child)), 1, scissor),
            (Operation::Decrement(ShapeDrawId(outline)), 1, scissor),
            (Operation::Draw(ShapeDrawId(sibling)), 0, scissor),
        ],
    );
}
