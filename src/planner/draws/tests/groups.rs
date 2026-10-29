use super::{clip, rect, shape, snapshot, DrawPlanner, Operation, RecordedCommands, Scene};
use crate::commands::ShapeDrawId;
use crate::scene_state::types::DrawTreeNode;

#[test]
fn structural_groups_preserve_draw_order_and_inherited_clips() {
    let mut scene = Scene::new();
    let root = scene.add(None, DrawTreeNode::Group(Default::default()));
    let scissor = scene.add(Some(root), clip((10.0, 10.0), (90.0, 80.0)));
    let stencil = scene.add(Some(scissor), shape(true));
    let group = scene.add(Some(stencil), DrawTreeNode::Group(Default::default()));
    scene.add(Some(group), DrawTreeNode::Group(Default::default()));
    let inner = scene.add(Some(group), clip((0.0, 20.0), (40.0, 90.0)));
    let nested = scene.add(Some(inner), DrawTreeNode::Group(Default::default()));
    let leaf = scene.add(Some(nested), shape(true));
    let group_sibling = scene.add(Some(group), shape(true));
    let root_sibling = scene.add(Some(root), shape(true));
    let mut planner = DrawPlanner::default();
    let mut output = RecordedCommands::default();
    scene.plan(&mut planner, &mut output);
    let outer_clip = rect((10, 10), (90, 80));
    assert_eq!(
        snapshot(&output),
        [
            (
                Operation::DrawAndIncrement(ShapeDrawId(stencil)),
                0,
                outer_clip
            ),
            (
                Operation::Draw(ShapeDrawId(leaf)),
                1,
                rect((10, 20), (40, 80))
            ),
            (Operation::Draw(ShapeDrawId(group_sibling)), 1, outer_clip),
            (Operation::Decrement(ShapeDrawId(stencil)), 1, outer_clip),
            (
                Operation::Draw(ShapeDrawId(root_sibling)),
                0,
                rect((0, 0), (100, 100))
            ),
        ]
    );
    assert_eq!(output.instructions.len(), 5);
    assert!(planner.parents.is_empty());
}
