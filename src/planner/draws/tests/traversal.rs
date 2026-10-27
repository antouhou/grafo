use super::*;

#[test]
fn deep_and_wide_rebuilt_trees_reuse_parent_storage() {
    let mut scene = Scene::new();
    let mut parent = scene.add(None, clip((10.0, 10.0), (80.0, 80.0)));
    for _ in 0..4096 {
        parent = scene.add(Some(parent), clip((0.0, 0.0), (90.0, 90.0)));
    }
    let leaf = scene.add(Some(parent), shape(true));
    let mut planner = DrawPlanner::default();
    let mut output = RecordedCommands::default();
    scene.plan(&mut planner, &mut output);
    assert_eq!(
        snapshot(&output),
        [(
            Operation::Draw(ShapeDrawId(leaf)),
            0,
            rect((10, 10), (80, 80))
        )]
    );
    let storage = planner.parents.as_ptr();
    drop(output);
    scene.tree.clear();
    let root = scene.add(None, clip((20.0, 20.0), (70.0, 70.0)));
    for _ in 0..4096 {
        scene.add(Some(root), clip((30.0, 30.0), (40.0, 40.0)));
    }
    let leaf = scene.add(Some(root), shape(true));
    let mut output = RecordedCommands::default();
    scene.plan(&mut planner, &mut output);
    assert_eq!(
        snapshot(&output),
        [(
            Operation::Draw(ShapeDrawId(leaf)),
            0,
            rect((20, 20), (70, 70))
        )]
    );
    assert_eq!(storage, planner.parents.as_ptr());
    assert!(planner.parents.is_empty());
}
