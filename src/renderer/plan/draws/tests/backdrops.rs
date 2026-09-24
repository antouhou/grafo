use super::*;
use crate::commands::{ShapeTextureBinding, TextureSampling};
use crate::core::effect::BackdropCaptureArea;
use crate::{
    Fill, Gradient, GradientCommonDesc, GradientStop, GradientStopOffset, LinearGradientDesc,
    LinearGradientLine,
};

#[test]
fn capture_and_effect_precede_ordinary_backdrop_draws_and_restore_clips() {
    for (has_child, clips_children) in [(false, true), (true, false), (true, true)] {
        let mut scene = Scene::new();
        let root = scene.add(None, shape(true));
        let scissor = scene.add(Some(root), clip((10.0, 10.0), (60.0, 60.0)));
        let backdrop = scene.add(Some(scissor), shape(clips_children));
        let child = has_child.then(|| scene.add(Some(backdrop), shape(true)));
        let sibling = scene.add(Some(root), shape(true));
        scene.attach_backdrop(backdrop);
        let mut output = DrawPlan::default();
        scene.plan(&mut DrawPlanner::default(), &mut output);
        let [DrawSegment::Draws {
            instructions: prefix,
            ..
        }, DrawSegment::CaptureBackdrop(capture), DrawSegment::ApplyEffect(effect), DrawSegment::Draws {
            instructions: suffix,
            texture_materials,
            ..
        }] = output.segments.as_slice()
        else {
            panic!("capture and effect split the draw stream")
        };
        assert_eq!(prefix, &(0..1));
        assert_eq!(suffix, &(1..output.instructions.len()));
        assert_eq!(capture.source, BackdropCaptureSource::Target);
        assert_eq!(capture.output, IntermediateTextureId::Planned(0));
        assert_eq!(effect.input, capture.output);
        assert_eq!(effect.output, IntermediateTextureId::Planned(1));
        assert_eq!(effect.effect_id, 42);
        assert_eq!(
            effect.parameters.bytes(&output.effect_parameters),
            &[1, 2, 3, 4]
        );
        assert_eq!(
            &output.texture_material_draws[texture_materials.clone()],
            &[2]
        );
        let DrawOperation::DrawShape(draw) = output.instructions[2].operation else {
            panic!("ordinary shape draw")
        };
        let layer = draw.material.under_fill_texture.unwrap();
        assert_eq!(
            layer.texture,
            ShapeTextureBinding::Intermediate(effect.output)
        );
        let TextureSampling::TargetPixels(bounds) = layer.sampling else {
            panic!("physical sampling")
        };
        assert_eq!(bounds, capture.region.bounds);
        let viewport = rect((0, 0), (100, 100));
        let inherited = rect((10, 10), (60, 60));
        let mut expected = vec![
            (Operation::DrawAndIncrement(ShapeDrawId(root)), 0, viewport),
            (Operation::Increment(ShapeDrawId(backdrop)), 1, inherited),
            (Operation::Draw(ShapeDrawId(backdrop)), 2, inherited),
        ];
        let decrement = (Operation::Decrement(ShapeDrawId(backdrop)), 2, inherited);
        if let Some(child) = child.filter(|_| clips_children) {
            expected.push((Operation::Draw(ShapeDrawId(child)), 2, inherited));
        }
        expected.push(decrement);
        if let Some(child) = child.filter(|_| !clips_children) {
            expected.push((Operation::Draw(ShapeDrawId(child)), 1, inherited));
        }
        expected.extend([
            (Operation::Draw(ShapeDrawId(sibling)), 1, viewport),
            (Operation::Decrement(ShapeDrawId(root)), 1, viewport),
        ]);
        assert_eq!(snapshot(&output), expected);
    }
}

#[test]
fn consecutive_captures_follow_preceding_draws_and_keep_separate_outputs() {
    let mut scene = Scene::new();
    let root = scene.add(None, clip((5.0, 5.0), (90.0, 90.0)));
    let first = scene.add(Some(root), shape(true));
    let second = scene.add(Some(root), shape(true));
    scene.attach_backdrop(first);
    scene.attach_backdrop(second);
    scene.backdrops.get_mut(&first).unwrap().config =
        BackdropEffectConfig::new().padding(4.0).downsample(0.5);
    scene.backdrops.get_mut(&second).unwrap().effect.params = vec![9, 8];
    let mut output = DrawPlan::default();
    scene.plan(&mut DrawPlanner::default(), &mut output);
    let [DrawSegment::CaptureBackdrop(first_capture), DrawSegment::ApplyEffect(first_effect), DrawSegment::Draws {
        instructions: first_draws,
        ..
    }, DrawSegment::CaptureBackdrop(second_capture), DrawSegment::ApplyEffect(second_effect), DrawSegment::Draws {
        instructions: second_draws,
        ..
    }] = output.segments.as_slice()
    else {
        panic!("each capture follows all earlier draws")
    };
    assert_eq!(first_draws, &(0..3));
    assert_eq!(second_draws, &(3..6));
    assert_eq!(first_capture.sampling_size, Size::new(44, 44));
    assert_eq!(
        first_capture.region.bounds.size().to_u32(),
        Size::new(88, 88)
    );
    assert_eq!(first_effect.input, first_capture.output);
    assert_eq!(second_effect.input, second_capture.output);
    assert_eq!(output.texture_count, 4);
    assert_ne!(first_effect.output, second_effect.output);
    assert_eq!(
        first_effect.parameters.bytes(&output.effect_parameters),
        &[1, 2, 3, 4]
    );
    assert_eq!(
        second_effect.parameters.bytes(&output.effect_parameters),
        &[9, 8]
    );
    assert_eq!(output.texture_material_draws, [1, 4]);
}

#[test]
fn layered_capture_references_registered_resources_and_disabling_capture_removes_all_work() {
    let mut scene = Scene::new();
    let panel = scene.add(None, shape(true));
    scene.attach_backdrop(panel);
    let base = IntermediateTextureId::Registered(0);
    scene.backdrop_source = Some(BackdropCaptureSource::Layered { base });
    let mut output = DrawPlan::default();
    scene.plan(&mut DrawPlanner::default(), &mut output);
    let DrawSegment::CaptureBackdrop(capture) = output.segments[0] else {
        panic!("capture precedes the first draw")
    };
    assert_eq!(capture.source, BackdropCaptureSource::Layered { base });
    assert_ne!(capture.output, base);

    scene.backdrop_source = None;
    scene.plan(&mut DrawPlanner::default(), &mut output);
    assert!(matches!(
        output.segments.as_slice(),
        [DrawSegment::Draws { .. }]
    ));
    assert_eq!(
        snapshot(&output),
        [(
            Operation::Draw(ShapeDrawId(panel)),
            0,
            rect((0, 0), (100, 100)),
        )]
    );
    assert!(output.effect_parameters.is_empty());
    assert!(output.texture_material_draws.is_empty());
    assert_eq!(output.texture_count, 0);
}

#[test]
fn rejected_capture_preserves_gradient_and_stencil_without_texture_work() {
    let mut scene = Scene::new();
    let panel = scene.add(None, shape(true));
    scene.attach_backdrop(panel);
    let DrawTreeNode::CachedShape(description) = scene.tree.get_mut(panel).unwrap() else {
        unreachable!()
    };
    description.fill = Some(Fill::Gradient(
        Gradient::linear(LinearGradientDesc {
            common: GradientCommonDesc::new([
                GradientStop::at_position(
                    GradientStopOffset::linear_radial(0.0),
                    Color::rgb(255, 0, 0),
                ),
                GradientStop::at_position(
                    GradientStopOffset::linear_radial(1.0),
                    Color::rgb(0, 0, 255),
                ),
            ]),
            line: LinearGradientLine {
                start: [0.0, 0.0],
                end: [80.0, 0.0],
            },
        })
        .unwrap(),
    ));
    scene.backdrops.get_mut(&panel).unwrap().config =
        BackdropEffectConfig::new().capture_area(BackdropCaptureArea::ScreenRect([
            (0.0, 0.0),
            (20_000.0, 20_000.0),
        ]));
    let mut output = DrawPlan::default();
    scene.plan(&mut DrawPlanner::default(), &mut output);
    assert!(matches!(
        output.segments.as_slice(),
        [DrawSegment::Draws { .. }]
    ));
    let DrawOperation::DrawShape(draw) = output.instructions[1].operation else {
        panic!("ordinary fallback draw")
    };
    assert!(draw.material.has_gradient_fill());
    assert!(draw.material.under_fill_texture.is_none());
    assert!(output.texture_material_draws.is_empty());
    assert!(output.effect_parameters.is_empty());
    assert_eq!(output.texture_count, 0);
    let viewport = rect((0, 0), (100, 100));
    assert_eq!(
        snapshot(&output),
        [
            (Operation::Increment(ShapeDrawId(panel)), 0, viewport),
            (Operation::Draw(ShapeDrawId(panel)), 1, viewport),
            (Operation::Decrement(ShapeDrawId(panel)), 1, viewport),
        ]
    );
}

#[test]
fn commands_remain_complete_after_planner_and_scene_are_dropped() {
    let mut output = DrawPlan::default();
    {
        let mut scene = Scene::new();
        let root = scene.add(None, shape(true));
        let panel = scene.add(Some(root), shape(true));
        scene.attach_backdrop(panel);
        scene.plan(&mut DrawPlanner::default(), &mut output);
    }
    assert_eq!(output.instructions.len(), 5);
    let DrawSegment::ApplyEffect(command) = &output.segments[2] else {
        panic!("effect command")
    };
    assert_eq!(command.effect_id, 42);
    assert_eq!(
        command.parameters.bytes(&output.effect_parameters),
        &[1, 2, 3, 4]
    );
    let DrawOperation::DrawShape(draw) = output.instructions[2].operation else {
        panic!("backdrop draw")
    };
    assert_eq!(draw.id, ShapeDrawId(1));
    assert_eq!(
        draw.material.texture_bindings,
        [ShapeTextureBinding::None; 2]
    );
    assert_eq!(
        draw.material.under_fill_texture.unwrap().texture,
        ShapeTextureBinding::Intermediate(command.output)
    );
    let _: DrawInstruction = output.instructions[0];
}

#[test]
fn rebuilt_queues_reuse_storage_and_replace_all_commands_and_parameters() {
    let mut scene = Scene::new();
    let mut planner = DrawPlanner::default();
    let mut output = DrawPlan::default();
    planner.parents.reserve(8);
    output.instructions.reserve(16);
    output.segments.reserve(8);
    output.effect_parameters.reserve(32);
    output.texture_material_draws.reserve(4);
    let allocations = (
        planner.parents.as_ptr(),
        output.instructions.as_ptr(),
        output.segments.as_ptr(),
        output.effect_parameters.as_ptr(),
        output.texture_material_draws.as_ptr(),
    );
    for has_backdrop in [true, false, true] {
        scene.tree.clear();
        scene.backdrops.clear();
        let root = scene.add(None, shape(true));
        let leaf = scene.add(Some(root), shape(true));
        if has_backdrop {
            scene.attach_backdrop(leaf);
        }
        scene.plan(&mut planner, &mut output);
        assert_eq!(
            allocations,
            (
                planner.parents.as_ptr(),
                output.instructions.as_ptr(),
                output.segments.as_ptr(),
                output.effect_parameters.as_ptr(),
                output.texture_material_draws.as_ptr(),
            )
        );
        assert_eq!(output.effect_parameters.is_empty(), !has_backdrop);
        assert_eq!(output.texture_material_draws.is_empty(), !has_backdrop);
        assert_eq!(output.texture_count, if has_backdrop { 2 } else { 0 });
        assert_eq!(output.segments.len(), if has_backdrop { 4 } else { 1 });
    }
    scene.tree.clear();
    scene.backdrops.clear();
    scene.plan(&mut planner, &mut output);
    assert!(output.instructions.is_empty());
    assert!(output.segments.is_empty());
    assert!(output.effect_parameters.is_empty());
    assert!(output.texture_material_draws.is_empty());
    assert_eq!(output.texture_count, 0);
}

#[test]
fn shape_effect_composite_precedes_its_source_backdrop_capture_under_inherited_clips() {
    let mut scene = Scene::new();
    let parent = scene.add(None, shape(true));
    let scissor = scene.add(Some(parent), clip((10.0, 10.0), (60.0, 60.0)));
    let source = scene.add(Some(scissor), shape(true));
    scene.attach_backdrop(source);
    scene.shape_effects.insert(
        source,
        TextureComposite {
            texture: IntermediateTextureId::ShapeEffect(0),
            placement: TexturePlacement::Local {
                transform: InstanceTransform::translation(12.0, 18.0),
                sampling: TextureUvTransform::IDENTITY,
            },
        },
    );
    let mut output = DrawPlan::default();
    scene.plan(&mut DrawPlanner::default(), &mut output);
    let [DrawSegment::Draws {
        instructions,
        composites,
        ..
    }, DrawSegment::CaptureBackdrop(_), DrawSegment::ApplyEffect(_), DrawSegment::Draws { .. }] =
        output.segments.as_slice()
    else {
        panic!("shape effect must be in the draw prefix captured by the backdrop")
    };
    assert_eq!(instructions, &(0..2));
    assert_eq!(&output.composite_draws[composites.clone()], &[1]);
    assert_eq!(
        snapshot(&output)[1],
        (
            Operation::Composite(IntermediateTextureId::ShapeEffect(0)),
            1,
            rect((10, 10), (60, 60))
        )
    );
    assert_eq!(
        snapshot(&output)[2],
        (
            Operation::Increment(ShapeDrawId(source)),
            1,
            rect((10, 10), (60, 60))
        )
    );
}
