use super::*;
use crate::commands::{ShapeTextureBinding, TextureSampling};
use crate::core::effect::BackdropCaptureArea;
use crate::planner::tests::RecordedCommands;
use crate::{
    Fill, Gradient, GradientCommonDesc, GradientStop, GradientStopOffset, LinearGradientDesc,
    LinearGradientLine,
};

#[test]
fn backdrop_texture_preserves_ordinary_draw_and_clip_commands() {
    for is_rectangle in [false, true] {
        for (has_child, clips_children) in [(false, true), (true, false), (true, true)] {
            let mut scene = Scene::new();
            let root = scene.add(None, shape(true));
            let panel = scene.add(
                Some(root),
                if is_rectangle {
                    DrawTreeNode::CachedShape(shape_description(
                        Shape::rect([(10.0, 10.0), (60.0, 60.0)]),
                        clips_children,
                    ))
                } else {
                    shape(clips_children)
                },
            );
            if has_child {
                scene.add(Some(panel), shape(true));
            }
            scene.add(Some(root), shape(true));
            let mut ordinary = RecordedCommands::default();
            scene.plan(&mut DrawPlanner::default(), &mut ordinary);
            let ordinary_snapshot = snapshot(&ordinary);
            drop(ordinary);
            let mut backdrop = RecordedCommands::default();
            scene.attach_backdrop(panel);
            scene.plan(&mut DrawPlanner::default(), &mut backdrop);
            assert_eq!(
                snapshot(&backdrop),
                ordinary_snapshot,
                "rectangle={is_rectangle}, has_child={has_child}, clips_children={clips_children}",
            );
            assert!(backdrop.instructions.iter().any(|command| {
                matches!(command.operation,
                    RenderOperation::DrawShape(draw)
                    | RenderOperation::DrawShapeAndIncrementStencil(draw)
                    if draw.id == ShapeDrawId(panel) && draw.material.under_fill_texture.is_some())
            }));
        }
    }
}

#[test]
fn texture_only_rectangle_draws_only_when_capture_produces_a_texture() {
    let mut scene = Scene::new();
    let panel = scene.add(
        None,
        DrawTreeNode::CachedShape(shape_description(
            Shape::rect([(0.0, 0.0), (80.0, 80.0)]),
            true,
        )),
    );
    let DrawTreeNode::CachedShape(description) = scene.tree.get_mut(panel).unwrap() else {
        unreachable!()
    };
    description.instance.fill = None;
    description.instance.color_override = None;
    let mut output = RecordedCommands::default();
    scene.attach_backdrop(panel);
    scene.plan(&mut DrawPlanner::default(), &mut output);
    assert_eq!(
        snapshot(&output),
        [(
            Operation::Draw(ShapeDrawId(panel)),
            0,
            rect((0, 0), (100, 100))
        )],
    );
    drop(output);
    let mut output = RecordedCommands::default();
    scene.update_backdrop_config(
        panel,
        BackdropEffectConfig::new().capture_area(BackdropCaptureArea::ScreenRect([
            (0.0, 0.0),
            (20_000.0, 20_000.0),
        ])),
    );
    scene.plan(&mut DrawPlanner::default(), &mut output);
    assert!(output.instructions.is_empty());
}

#[test]
fn capture_and_effect_precede_backdrop_draws_and_restore_clips() {
    for (has_child, clips_children) in [(false, true), (true, false), (true, true)] {
        let mut output = RecordedCommands::default();
        let mut scene = Scene::new();
        let root = scene.add(None, shape(true));
        let scissor = scene.add(Some(root), clip((10.0, 10.0), (60.0, 60.0)));
        let backdrop = scene.add(Some(scissor), shape(clips_children));
        let child = has_child.then(|| scene.add(Some(backdrop), shape(true)));
        let sibling = scene.add(Some(root), shape(true));
        scene.attach_backdrop(backdrop);
        scene.plan(&mut DrawPlanner::default(), &mut output);
        let ops = operations(&output);
        let [RenderOperation::DrawShapeAndIncrementStencil(_), RenderOperation::Capture(capture), RenderOperation::ApplyEffect(effect), draw_command, ..] =
            ops.as_slice()
        else {
            panic!("capture and effect precede the backdrop draw");
        };
        let draw = match draw_command {
            RenderOperation::DrawShape(draw)
            | RenderOperation::DrawShapeAndIncrementStencil(draw) => draw,
            _ => panic!("filtered texture is used by a shape draw"),
        };
        assert_eq!(capture.source, CaptureSource::Target);
        assert_eq!(capture.output, IntermediateTextureId::Planned(0));
        assert_eq!(effect.input, capture.output);
        assert_eq!(effect.output, IntermediateTextureId::Planned(1));
        assert_eq!(effect.effect_id, 42);
        assert_eq!(effect.parameters.bytes, &[1, 2, 3, 4]);
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
            (
                if has_child && clips_children {
                    Operation::DrawAndIncrement(ShapeDrawId(backdrop))
                } else {
                    Operation::Draw(ShapeDrawId(backdrop))
                },
                1,
                inherited,
            ),
        ];
        let decrement = (Operation::Decrement(ShapeDrawId(backdrop)), 2, inherited);
        if let Some(child) = child.filter(|_| clips_children) {
            expected.push((Operation::Draw(ShapeDrawId(child)), 2, inherited));
        }
        if has_child && clips_children {
            expected.push(decrement);
        }
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
    let mut output = RecordedCommands::default();
    let mut scene = Scene::new();
    let root = scene.add(None, clip((5.0, 5.0), (90.0, 90.0)));
    let first = scene.add(Some(root), shape(true));
    let second = scene.add(Some(root), shape(true));
    scene.attach_backdrop(first);
    scene.attach_backdrop(second);
    scene.update_backdrop_config(
        first,
        BackdropEffectConfig::new().padding(4.0).downsample(0.5),
    );
    scene.backdrops.get_mut(&second).unwrap().effect.parameters = scene.parameters.store(&[9, 8]);
    scene.plan(&mut DrawPlanner::default(), &mut output);
    let ops = operations(&output);
    let [RenderOperation::Capture(first_capture), RenderOperation::ApplyEffect(first_effect), RenderOperation::DrawShape(_), RenderOperation::Capture(second_capture), RenderOperation::ApplyEffect(second_effect), RenderOperation::DrawShape(_)] =
        ops.as_slice()
    else {
        panic!("each capture follows all earlier draws");
    };
    assert_eq!(first_capture.sampling_size, Size::new(44, 44));
    assert_eq!(
        first_capture.region.bounds.size().to_u32(),
        Size::new(88, 88)
    );
    assert_eq!(first_effect.input, first_capture.output);
    assert_eq!(second_effect.input, second_capture.output);
    assert_eq!(first_effect.output, IntermediateTextureId::Planned(1));
    assert_eq!(second_effect.output, IntermediateTextureId::Planned(3));
    assert_ne!(first_effect.output, second_effect.output);
    assert_eq!(first_effect.parameters.bytes, &[1, 2, 3, 4]);
    assert_eq!(second_effect.parameters.bytes, &[9, 8]);
}

#[test]
fn layered_capture_references_registered_resources_and_disabling_capture_removes_all_work() {
    let mut output = RecordedCommands::default();
    let mut scene = Scene::new();
    let panel = scene.add(None, shape(true));
    scene.attach_backdrop(panel);
    let base = IntermediateTextureId::Registered(0);
    scene.backdrop_source = Some(CaptureSource::Layered {
        base,
        origin: [12, 24],
    });
    scene.plan(&mut DrawPlanner::default(), &mut output);
    let RenderOperation::Capture(capture) = output.instructions[0].operation else {
        panic!("capture precedes the first draw")
    };
    assert_eq!(
        capture.source,
        CaptureSource::Layered {
            base,
            origin: [12, 24]
        }
    );
    assert_ne!(capture.output, base);

    drop(output);
    let mut output = RecordedCommands::default();
    scene.backdrop_source = None;
    scene.plan(&mut DrawPlanner::default(), &mut output);
    assert_eq!(
        snapshot(&output),
        [(
            Operation::Draw(ShapeDrawId(panel)),
            0,
            rect((0, 0), (100, 100)),
        )]
    );
}

#[test]
fn rejected_capture_preserves_ordinary_gradient_draw_without_texture_work() {
    let mut output = RecordedCommands::default();
    let mut scene = Scene::new();
    let panel = scene.add(None, shape(true));
    scene.attach_backdrop(panel);
    let DrawTreeNode::CachedShape(description) = scene.tree.get_mut(panel).unwrap() else {
        unreachable!()
    };
    description.instance.fill = Some(Fill::Gradient(
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
    scene.update_backdrop_config(
        panel,
        BackdropEffectConfig::new().capture_area(BackdropCaptureArea::ScreenRect([
            (0.0, 0.0),
            (20_000.0, 20_000.0),
        ])),
    );
    scene.plan(&mut DrawPlanner::default(), &mut output);
    let RenderOperation::DrawShape(draw) = output.instructions[0].operation else {
        panic!("fallback uses the ordinary shape draw")
    };
    assert!(draw.material.has_gradient_fill);
    assert!(draw.material.under_fill_texture.is_none());
    let viewport = rect((0, 0), (100, 100));
    assert_eq!(
        snapshot(&output),
        [(Operation::Draw(ShapeDrawId(panel)), 0, viewport)]
    );
}

#[test]
fn commands_borrow_parameters_after_planner_is_dropped() {
    let mut output = RecordedCommands::default();
    let mut scene = Scene::new();
    let root = scene.add(None, shape(true));
    let panel = scene.add(Some(root), shape(true));
    scene.attach_backdrop(panel);
    scene.plan(&mut DrawPlanner::default(), &mut output);
    assert_eq!(output.instructions.len(), 5);
    let RenderOperation::ApplyEffect(command) = &output.instructions[2].operation else {
        panic!("effect command")
    };
    assert_eq!(command.effect_id, 42);
    assert_eq!(command.parameters.bytes, &[1, 2, 3, 4]);
    let RenderOperation::DrawShape(draw) = output.instructions[3].operation else {
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
    let _: RenderCommand = output.instructions[0];
}

#[test]
fn rebuilt_queues_reuse_traversal_and_parameter_storage() {
    let mut scene = Scene::new();
    let mut planner = DrawPlanner::default();
    planner.parents.reserve(8);
    scene.parameters.bytes.reserve(32);
    let allocations = (planner.parents.as_ptr(), scene.parameters.bytes.as_ptr());
    for has_backdrop in [true, false, true] {
        scene.tree.clear();
        scene.backdrops.clear();
        scene.parameters.clear();
        let root = scene.add(None, shape(true));
        let leaf = scene.add(Some(root), shape(true));
        if has_backdrop {
            scene.attach_backdrop(leaf);
        }
        let mut output = RecordedCommands::default();
        scene.plan(&mut planner, &mut output);
        assert_eq!(
            allocations,
            (planner.parents.as_ptr(), scene.parameters.bytes.as_ptr())
        );
        assert_eq!(scene.parameters.bytes.is_empty(), !has_backdrop);
    }
    scene.tree.clear();
    scene.backdrops.clear();
    scene.parameters.clear();
    let mut output = RecordedCommands::default();
    scene.plan(&mut planner, &mut output);
    assert!(output.instructions.is_empty());
    assert!(scene.parameters.bytes.is_empty());
}

#[test]
fn shape_mask_effect_composite_precedes_its_source_backdrop_capture_under_inherited_clips() {
    let mut output = RecordedCommands::default();
    let mut scene = Scene::new();
    let parent = scene.add(None, shape(true));
    let scissor = scene.add(Some(parent), clip((10.0, 10.0), (60.0, 60.0)));
    let source = scene.add(Some(scissor), shape(true));
    scene.attach_backdrop(source);
    scene.shape_mask_effects.insert(
        source,
        TextureComposite {
            texture: IntermediateTextureId::Registered(100),
            placement: TexturePlacement::Local {
                transform: InstanceTransform::translation(12.0, 18.0),
                sampling: TextureUvTransform::IDENTITY,
            },
        },
    );
    scene.plan(&mut DrawPlanner::default(), &mut output);
    let ops = operations(&output);
    assert!(matches!(
        ops.as_slice(),
        [
            RenderOperation::DrawShapeAndIncrementStencil(_),
            RenderOperation::CompositeTexture(_),
            RenderOperation::Capture(_),
            RenderOperation::ApplyEffect(_),
            ..
        ]
    ));
    assert_eq!(
        snapshot(&output)[1],
        (
            Operation::Composite(IntermediateTextureId::Registered(100)),
            1,
            rect((10, 10), (60, 60))
        )
    );
    assert_eq!(
        snapshot(&output)[2],
        (
            Operation::Draw(ShapeDrawId(source)),
            1,
            rect((10, 10), (60, 60))
        )
    );
}
