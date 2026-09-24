use super::*;

#[test]
fn nested_targets_reset_clips_and_restore_each_parent_before_compositing() {
    let mut scene = Scene::new();
    let root = scene.add(None, shape(true));
    let scissor = scene.add(Some(root), clip((10.0, 10.0), (60.0, 60.0)));
    let outer = scene.add(Some(scissor), shape(true));
    let inner = scene.add(Some(outer), shape(true));
    let inner_sibling = scene.add(Some(outer), shape(true));
    let outer_sibling = scene.add(Some(scissor), shape(true));
    let unclipped_sibling = scene.add(Some(root), shape(true));
    for node in [outer, inner] {
        scene.groups.insert(
            node,
            EffectInstance {
                effect_id: 1,
                params: Vec::new(),
            },
        );
    }
    let viewport = rect((0, 0), (100, 100));
    let parent_scissor = rect((10, 10), (60, 60));
    let mut output = RenderPlan::default();
    let mut planner = DrawPlanner::default();
    scene.plan(&mut planner, &mut output);
    assert_eq!(
        snapshot(&output),
        [
            (Operation::DrawAndIncrement(ShapeDrawId(root)), 0, viewport),
            (Operation::DrawAndIncrement(ShapeDrawId(outer)), 0, viewport),
            (Operation::Draw(ShapeDrawId(inner)), 0, viewport),
            (
                Operation::Composite(IntermediateTextureId::Planned(2)),
                1,
                viewport
            ),
            (Operation::Draw(ShapeDrawId(inner_sibling)), 1, viewport),
            (Operation::Decrement(ShapeDrawId(outer)), 1, viewport),
            (
                Operation::Composite(IntermediateTextureId::Planned(3)),
                1,
                parent_scissor
            ),
            (
                Operation::Draw(ShapeDrawId(outer_sibling)),
                1,
                parent_scissor
            ),
            (Operation::Draw(ShapeDrawId(unclipped_sibling)), 1, viewport),
            (Operation::Decrement(ShapeDrawId(root)), 1, viewport),
        ]
    );
    let storage = (planner.parents.as_ptr(), output.instructions.as_ptr());
    scene.plan(&mut planner, &mut output);
    assert_eq!(
        storage,
        (planner.parents.as_ptr(), output.instructions.as_ptr())
    );
}
