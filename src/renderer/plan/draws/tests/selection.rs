use super::*;

#[test]
fn selected_subtree_starts_at_the_viewport_and_substitutes_nested_effects() {
    let mut scene = Scene::new();
    let root = scene.add(None, shape(true));
    let ancestor = scene.add(Some(root), clip((20.0, 20.0), (40.0, 40.0)));
    let selected = scene.add(Some(ancestor), shape(true));
    let scissor = scene.add(Some(selected), clip((10.0, 10.0), (80.0, 80.0)));
    let nested_group = scene.add(Some(scissor), shape(true));
    scene.add(Some(nested_group), shape(true));
    let sibling = scene.add(Some(selected), shape(true));
    scene.add(Some(ancestor), shape(true));
    let texture = IntermediateTextureId::Registered(4);
    scene.results.insert(nested_group, texture);
    let mut planner = DrawPlanner::default();
    let mut output = DrawPlan::default();
    scene.plan_selection(
        DrawTreeSelection {
            subtree_root: Some(selected),
            ..Default::default()
        },
        &mut planner,
        &mut output,
    );

    let viewport = rect((0, 0), (100, 100));
    assert_eq!(
        snapshot(&output),
        [
            (
                Operation::DrawAndIncrement(ShapeDrawId(selected)),
                0,
                viewport
            ),
            (Operation::Composite(texture), 1, rect((10, 10), (80, 80))),
            (Operation::Draw(ShapeDrawId(sibling)), 1, viewport),
            (Operation::Decrement(ShapeDrawId(selected)), 1, viewport),
        ]
    );
    assert!(planner.parents.is_empty());
}

#[test]
fn exclusion_precedes_effect_substitution_and_restores_clips_for_siblings() {
    let mut scene = Scene::new();
    let root = scene.add(None, shape(true));
    let scissor = scene.add(Some(root), clip((10.0, 10.0), (80.0, 80.0)));
    let excluded = scene.add(Some(scissor), shape(true));
    let descendant = scene.add(Some(excluded), shape(true));
    let sibling = scene.add(Some(scissor), shape(true));
    let outside = scene.add(Some(root), shape(true));
    scene.attach_backdrop(excluded);
    scene.attach_backdrop(descendant);
    scene
        .results
        .insert(excluded, IntermediateTextureId::Registered(0));
    scene
        .results
        .insert(descendant, IntermediateTextureId::Registered(1));
    scene.shape_effects.insert(
        excluded,
        TextureComposite {
            texture: IntermediateTextureId::ShapeEffect(0),
            placement: TexturePlacement::Target,
        },
    );
    let mut planner = DrawPlanner::default();
    let mut output = DrawPlan::default();
    scene.plan_selection(
        DrawTreeSelection {
            excluded_subtree: Some(excluded),
            ..Default::default()
        },
        &mut planner,
        &mut output,
    );
    let viewport = rect((0, 0), (100, 100));
    assert_eq!(
        snapshot(&output),
        [
            (Operation::DrawAndIncrement(ShapeDrawId(root)), 0, viewport),
            (
                Operation::Draw(ShapeDrawId(sibling)),
                1,
                rect((10, 10), (80, 80))
            ),
            (Operation::Draw(ShapeDrawId(outside)), 1, viewport),
            (Operation::Decrement(ShapeDrawId(root)), 1, viewport),
        ]
    );
    assert!(matches!(
        output.segments.as_slice(),
        [DrawSegment::Draws { .. }]
    ));
    assert!(output.composites.is_empty());
    assert!(output.effect_parameters.is_empty());
    assert_eq!(output.texture_count, 0);

    scene.plan_selection(
        DrawTreeSelection {
            subtree_root: Some(scissor),
            excluded_subtree: Some(excluded),
        },
        &mut planner,
        &mut output,
    );
    assert_eq!(
        snapshot(&output),
        [(
            Operation::Draw(ShapeDrawId(sibling)),
            0,
            rect((10, 10), (80, 80))
        )]
    );
}

#[test]
fn substituted_group_omits_its_clips_effects_and_deep_descendants() {
    let mut scene = Scene::new();
    let root = scene.add(None, shape(true));
    let scissor = scene.add(Some(root), clip((10.0, 10.0), (80.0, 80.0)));
    let group = scene.add(Some(scissor), shape(true));
    let nested = scene.add(Some(group), shape(true));
    let mut descendant = nested;
    for _ in 0..4096 {
        descendant = scene.add(Some(descendant), clip((30.0, 30.0), (40.0, 40.0)));
    }
    scene.attach_backdrop(group);
    scene.attach_backdrop(nested);
    let texture = IntermediateTextureId::Registered(1);
    scene.results.insert(group, texture);
    scene
        .results
        .insert(nested, IntermediateTextureId::Registered(2));
    scene.shape_effects.insert(
        group,
        TextureComposite {
            texture: IntermediateTextureId::ShapeEffect(0),
            placement: TexturePlacement::Target,
        },
    );
    let sibling = scene.add(Some(root), shape(true));
    let mut planner = DrawPlanner::default();
    planner.parents.reserve(4);
    let parent_capacity = planner.parents.capacity();
    let mut output = DrawPlan::default();

    // Excluding a descendant does not undo substitution of its ancestor.
    scene.plan_selection(
        DrawTreeSelection {
            excluded_subtree: Some(nested),
            ..Default::default()
        },
        &mut planner,
        &mut output,
    );
    let viewport = rect((0, 0), (100, 100));
    assert_eq!(
        snapshot(&output),
        [
            (Operation::DrawAndIncrement(ShapeDrawId(root)), 0, viewport),
            (Operation::Composite(texture), 1, rect((10, 10), (80, 80))),
            (Operation::Draw(ShapeDrawId(sibling)), 1, viewport),
            (Operation::Decrement(ShapeDrawId(root)), 1, viewport),
        ]
    );
    assert_eq!(planner.parents.capacity(), parent_capacity);
    assert_eq!(output.composites.len(), 1);
    assert!(matches!(
        output.segments.as_slice(),
        [DrawSegment::Draws { .. }]
    ));
    assert!(output.effect_parameters.is_empty());
    assert_eq!(output.texture_count, 0);

    scene.plan_selection(
        DrawTreeSelection {
            subtree_root: Some(group),
            ..Default::default()
        },
        &mut planner,
        &mut output,
    );
    assert_eq!(
        snapshot(&output),
        [(Operation::Composite(texture), 0, viewport)]
    );

    for selection in [
        DrawTreeSelection {
            subtree_root: Some(group),
            excluded_subtree: Some(group),
        },
        DrawTreeSelection {
            excluded_subtree: Some(root),
            ..Default::default()
        },
        DrawTreeSelection {
            subtree_root: Some(usize::MAX),
            ..Default::default()
        },
    ] {
        scene.plan_selection(selection, &mut planner, &mut output);
        assert!(output.instructions.is_empty());
        assert!(output.segments.is_empty());
        assert!(output.composites.is_empty());
        assert!(planner.parents.is_empty());
    }
}

#[test]
fn deep_and_wide_rebuilt_trees_reuse_parent_and_command_storage() {
    let mut scene = Scene::new();
    let mut parent = scene.add(None, clip((10.0, 10.0), (80.0, 80.0)));
    for _ in 0..4096 {
        parent = scene.add(Some(parent), clip((0.0, 0.0), (90.0, 90.0)));
    }
    let leaf = scene.add(Some(parent), shape(true));
    let mut planner = DrawPlanner::default();
    let mut output = DrawPlan::default();
    scene.plan(&mut planner, &mut output);
    assert_eq!(
        snapshot(&output),
        [(
            Operation::Draw(ShapeDrawId(leaf)),
            0,
            rect((10, 10), (80, 80))
        )]
    );
    let storage = (
        planner.parents.as_ptr(),
        output.instructions.as_ptr(),
        output.segments.as_ptr(),
    );
    scene.tree.clear();
    let root = scene.add(None, clip((20.0, 20.0), (70.0, 70.0)));
    for _ in 0..4096 {
        scene.add(Some(root), clip((30.0, 30.0), (40.0, 40.0)));
    }
    let leaf = scene.add(Some(root), shape(true));
    scene.plan(&mut planner, &mut output);
    assert_eq!(
        snapshot(&output),
        [(
            Operation::Draw(ShapeDrawId(leaf)),
            0,
            rect((20, 20), (70, 70))
        )]
    );
    assert_eq!(
        storage,
        (
            planner.parents.as_ptr(),
            output.instructions.as_ptr(),
            output.segments.as_ptr()
        )
    );
    assert!(planner.parents.is_empty());
    #[cfg(feature = "render_metrics")]
    assert_eq!(output.scissor_clip_count, 1);
}
