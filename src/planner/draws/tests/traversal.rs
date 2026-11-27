use super::*;

#[test]
fn nested_clips_use_direct_coverage_unless_descendants_need_perspective_rasterization() {
    for crosses_projection_plane in [false, true] {
        let mut scene = Scene::new();
        let root = scene.add(None, shape(true));
        let wrapper = scene.add(Some(root), shape(true));
        let mut description = shape_description(Shape::rect([(0.0, 0.0), (80.0, 80.0)]), false);
        if crosses_projection_plane {
            let mut transform = InstanceTransform::identity();
            transform.col0[3] = -0.1;
            description = CachedShapeDrawData::new(
                description.instance.cached_shape.clone(),
                ShapeDrawOptions::new()
                    .color(Color::WHITE)
                    .transform(transform),
            );
            assert!(!description.logical_screen_bounds.is_finite());
        }
        scene.add(Some(wrapper), DrawTreeNode::CachedShape(description));
        let mut output = RecordedCommands::default();
        scene.plan(&mut DrawPlanner::default(), &mut output);
        let mut clips = 0;
        for command in &output.instructions {
            if let RenderOperation::DrawShapeAndIncrementStencil(draw) = command.operation {
                assert!(!draw.requires_destination);
                assert_eq!(draw.requires_layer, crosses_projection_plane);
                clips += 1;
            }
        }
        assert_eq!(clips, 2);
    }
}

#[test]
fn fractional_scroll_clips_only_need_coverage_when_children_reach_fractional_edges() {
    let mut scene = Scene::new();
    let root = scene.add(None, clip((10.25, 10.0), (89.75, 80.0)));
    let child = scene.add(
        Some(root),
        DrawTreeNode::CachedShape(shape_description(
            Shape::rect([(20.0, 20.0), (70.0, 95.0)]),
            false,
        )),
    );
    let mut output = RecordedCommands::default();
    scene.plan(&mut DrawPlanner::default(), &mut output);
    assert_eq!(
        snapshot(&output),
        [(
            Operation::Draw(ShapeDrawId(child)),
            0,
            rect((10, 10), (90, 80))
        )]
    );
    assert!(!output.instructions.iter().any(|command| matches!(
        command.operation,
        RenderOperation::DrawShapeAndIncrementStencil(_)
    )));
    drop(output);
    *scene.tree.get_mut(child).unwrap() = DrawTreeNode::CachedShape(shape_description(
        Shape::rect([(10.5, 20.0), (70.0, 95.0)]),
        false,
    ));
    let mut output = RecordedCommands::default();
    scene.plan(&mut DrawPlanner::default(), &mut output);
    assert!(output.instructions.iter().any(|command| matches!(
        command.operation,
        RenderOperation::DrawShapeAndIncrementStencil(_)
    )));
}

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

#[test]
fn rounded_parent_does_not_clip_children_whose_pixels_are_fully_inside() {
    let mut scene = Scene::new();
    let root = scene.add(None, shape(true));
    let child = scene.add(
        Some(root),
        DrawTreeNode::CachedShape(shape_description(
            Shape::rect([(10.25, 20.0), (69.75, 60.0)]),
            false,
        )),
    );
    let mut output = RecordedCommands::default();
    scene.plan(&mut DrawPlanner::default(), &mut output);
    let viewport = rect((0, 0), (100, 100));
    assert_eq!(
        snapshot(&output),
        [
            (Operation::Draw(ShapeDrawId(root)), 0, viewport),
            (Operation::Draw(ShapeDrawId(child)), 0, viewport),
        ]
    );
    drop(output);
    scene.add(
        Some(child),
        DrawTreeNode::CachedShape(shape_description(
            Shape::rect([(0.0, 0.0), (8.0, 8.0)]),
            false,
        )),
    );
    let mut output = RecordedCommands::default();
    scene.plan(&mut DrawPlanner::default(), &mut output);
    assert!(matches!(
        snapshot(&output)[0].0,
        Operation::DrawAndIncrement(_)
    ));
}

#[test]
fn generic_path_clips_skip_contained_children_but_keep_holes_clipped() {
    let mut scene = Scene::new();
    let parent = Shape::builder()
        .begin((0.0, 10.0))
        .line_to((10.0, 0.0))
        .line_to((70.0, 0.0))
        .line_to((80.0, 10.0))
        .line_to((80.0, 70.0))
        .line_to((70.0, 80.0))
        .line_to((10.0, 80.0))
        .line_to((0.0, 70.0))
        .close();
    let root = scene.add(
        None,
        DrawTreeNode::CachedShape(shape_description(parent.clone().build(), true)),
    );
    scene.add(
        Some(root),
        DrawTreeNode::CachedShape(shape_description(
            Shape::rect([(15.0, 20.0), (65.0, 60.0)]),
            false,
        )),
    );
    let mut output = RecordedCommands::default();
    scene.plan(&mut DrawPlanner::default(), &mut output);
    assert!(!output.instructions.iter().any(|command| matches!(
        command.operation,
        RenderOperation::DrawShapeAndIncrementStencil(_)
    )));
    drop(output);
    let hole = parent
        .begin((30.0, 30.0))
        .line_to((30.0, 50.0))
        .line_to((50.0, 50.0))
        .line_to((50.0, 30.0))
        .close()
        .build();
    *scene.tree.get_mut(root).unwrap() = DrawTreeNode::CachedShape(shape_description(hole, true));
    let mut output = RecordedCommands::default();
    scene.plan(&mut DrawPlanner::default(), &mut output);
    assert!(output.instructions.iter().any(|command| matches!(
        command.operation,
        RenderOperation::DrawShapeAndIncrementStencil(_)
    )));
}

#[test]
fn transparent_wrappers_do_not_expand_visible_children_but_backdrops_do() {
    let mut scene = Scene::new();
    let root = scene.add(None, shape(true));
    let mut wrapper = shape_description(
        Shape::rounded_rect([(0.0, 0.0), (80.0, 80.0)], BorderRadii::new(5.0)),
        false,
    );
    wrapper.instance.fill = None;
    let wrapper = scene.add(Some(root), DrawTreeNode::CachedShape(wrapper));
    scene.add(
        Some(wrapper),
        DrawTreeNode::CachedShape(shape_description(
            Shape::rect([(10.0, 20.0), (70.0, 60.0)]),
            false,
        )),
    );
    let mut output = RecordedCommands::default();
    scene.plan(&mut DrawPlanner::default(), &mut output);
    assert!(!output.instructions.iter().any(|command| matches!(
        command.operation,
        RenderOperation::DrawShape(draw) if draw.id == ShapeDrawId(wrapper)
    )));
    assert!(!output.instructions.iter().any(|command| matches!(
        command.operation,
        RenderOperation::DrawShapeAndIncrementStencil(_)
    )));
    drop(output);
    scene.attach_backdrop(wrapper);
    let mut output = RecordedCommands::default();
    scene.plan(&mut DrawPlanner::default(), &mut output);
    assert!(output.instructions.iter().any(|command| matches!(
        command.operation,
        RenderOperation::DrawShapeAndIncrementStencil(ClipDraw {
            requires_destination: true,
            ..
        })
    )));
}
