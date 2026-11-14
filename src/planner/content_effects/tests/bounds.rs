use super::*;
use crate::commands::TexturePlacement;
use crate::core::effect::BackdropCaptureArea;

#[test]
fn content_targets_use_the_padded_effect_bounds() {
    let mut scene = Scene::new();
    let root = scene.add(None);
    let node = scene.tree.add_child(
        root,
        DrawTreeNode::ClipRect(ClipRectDrawData::new(
            [(30.0, 20.0), (50.0, 40.0)],
            None,
            false,
        )),
    );
    scene.attach_content(
        node,
        1,
        ContentEffectConfig::new().outsets(4.0, 2.0, 6.0, 8.0),
    );
    let mut output = RecordedCommands::default();
    scene.plan(&mut DrawPlanner::default(), &mut output);
    let targets: Vec<_> = output
        .instructions
        .iter()
        .filter_map(|command| match command.operation {
            RenderOperation::BeginTarget(Target::Texture { bounds, .. }) => {
                Some(bounds.size().to_u32())
            }
            _ => None,
        })
        .collect();
    assert_eq!(targets, [Size::new(30, 30)]);
    assert!(!output
        .instructions
        .iter()
        .any(|command| matches!(command.operation, RenderOperation::Capture(_))));
    validate_dependencies(&output);
}

#[test]
fn downsampling_keeps_full_resolution_rendering_and_padded_bounds() {
    let mut scene = Scene::new();
    let node = scene
        .tree
        .add_node(DrawTreeNode::ClipRect(ClipRectDrawData::new(
            [(-3.0, 20.0), (21.0, 44.0)],
            None,
            false,
        )));
    scene.attach_content(
        node,
        1,
        ContentEffectConfig::new().outset(4.0).downsample(0.5),
    );
    let mut output = RecordedCommands::default();
    scene.plan(&mut DrawPlanner::default(), &mut output);
    validate_dependencies(&output);
    let RenderOperation::BeginTarget(Target::Texture {
        bounds,
        draw_scissor,
        ..
    }) = output.instructions[1].operation
    else {
        panic!("content starts a bounded target");
    };
    assert_eq!(bounds.min.to_tuple(), (-7, 16));
    assert_eq!(bounds.size().to_tuple(), (32, 32));
    assert_eq!(draw_scissor.min.to_tuple(), (0, 20));
    let RenderOperation::Capture(capture) = output.instructions[2].operation else {
        panic!("downsampling happens after rendering");
    };
    assert_eq!(capture.destination_bounds, bounds);
    assert_eq!(capture.source_rect, Some(draw_scissor));
    assert_eq!(capture.sampling_size, Size::new(16, 16));
}

#[test]
fn fractional_inputs_keep_draw_limits_separate_from_padded_outputs() {
    for (scale, input, output, sampling_size) in [
        (1.0, [(0, 2), (11, 12)], [(0, 1), (11, 13)], [6, 6]),
        (1.25, [(1, 2), (13, 15)], [(0, 2), (14, 16)], [7, 7]),
        (2.0, [(1, 4), (21, 24)], [(1, 3), (21, 25)], [10, 11]),
    ] {
        let mut scene = Scene::new();
        scene.scale_factor = scale;
        let node = scene
            .tree
            .add_node(DrawTreeNode::ClipRect(ClipRectDrawData::new(
                [(0.875, 2.25), (10.125, 11.75)],
                None,
                false,
            )));
        scene.attach_content(
            node,
            1,
            ContentEffectConfig::new().outset(0.375).downsample(0.5),
        );
        let mut commands = RecordedCommands::default();
        scene.plan(&mut DrawPlanner::default(), &mut commands);
        validate_dependencies(&commands);
        let RenderOperation::BeginTarget(Target::Texture {
            bounds,
            draw_scissor,
            ..
        }) = commands.instructions[1].operation
        else {
            panic!("content starts a bounded target");
        };
        assert_eq!([bounds.min.to_tuple(), bounds.max.to_tuple()], output);
        assert_eq!(
            [draw_scissor.min.to_tuple(), draw_scissor.max.to_tuple()],
            input
        );
        let RenderOperation::Capture(capture) = commands.instructions[2].operation else {
            panic!("downsampling preserves padded output and input restrictions");
        };
        assert_eq!(capture.destination_bounds, bounds);
        assert_eq!(capture.source_rect, Some(draw_scissor));
        assert_eq!(capture.sampling_size.to_array(), sampling_size);
        let RenderOperation::CompositeTexture(composite) = commands.instructions[5].operation
        else {
            panic!("the effect result is composited after closing its target");
        };
        let TexturePlacement::Local { transform, .. } = composite.placement else {
            panic!("effect output has screen-space placement");
        };
        assert_eq!(transform.col3[0], output[0].0 as f32 / scale as f32);
        assert_eq!(transform.col3[1], output[0].1 as f32 / scale as f32);
    }
}

#[test]
fn internal_backdrops_expand_rendering_then_crop_to_the_content_input() {
    let mut scene = Scene::new();
    let root = scene.add(None);
    let node = scene.tree.add_child(
        root,
        DrawTreeNode::ClipRect(ClipRectDrawData::new(
            [(30.0, 20.0), (50.0, 40.0)],
            None,
            false,
        )),
    );
    scene.attach_content(
        node,
        1,
        ContentEffectConfig::new().outsets(4.0, 2.0, 6.0, 8.0),
    );
    scene.backdrop(node);
    let mut output = RecordedCommands::default();
    scene.plan(&mut DrawPlanner::default(), &mut output);
    validate_dependencies(&output);
    let regions: Vec<_> = output
        .instructions
        .iter()
        .filter_map(|command| match command.operation {
            RenderOperation::BeginTarget(Target::Texture { bounds, .. }) => Some(bounds),
            _ => None,
        })
        .collect();
    assert_eq!(regions.len(), 1);
    assert_eq!(regions[0].min.to_tuple(), (6, 6));
    assert_eq!(regions[0].max.to_tuple(), (50, 44));
    let capture = output
        .instructions
        .iter()
        .find_map(|command| match command.operation {
            RenderOperation::Capture(capture)
                if capture.destination_bounds.min.to_tuple() == (26, 18) =>
            {
                Some(capture)
            }
            _ => None,
        })
        .unwrap();
    assert_eq!(capture.destination_bounds.min.to_tuple(), (26, 18));
    assert_eq!(capture.destination_bounds.max.to_tuple(), (56, 48));
    assert_eq!(capture.source_rect.unwrap().min.to_tuple(), (30, 20));
    assert_eq!(capture.source_rect.unwrap().max.to_tuple(), (50, 40));
}

#[test]
fn fully_offscreen_backdrop_inputs_still_run_the_effect() {
    let mut scene = Scene::new();
    let group = scene.group(None, 1);
    let node = scene.backdrop(group);
    scene.backdrops.insert(
        node,
        BackdropEffectInstance::new(
            scene.backdrops[&node].effect,
            BackdropEffectConfig::new().capture_area(BackdropCaptureArea::ScreenRect([
                (120.0, 10.0),
                (150.0, 40.0),
            ])),
            scene.tree.get(node).unwrap().logical_screen_bounds(),
            Viewport {
                physical_size: (100, 100),
                scale_factor: 1.0,
            },
            4096,
        ),
    );
    let mut output = RecordedCommands::default();
    scene.plan(&mut DrawPlanner::default(), &mut output);
    validate_dependencies(&output);
    let capture = output
        .instructions
        .iter()
        .find_map(|command| match command.operation {
            RenderOperation::Capture(capture) => Some(capture),
            _ => None,
        })
        .unwrap();
    assert_eq!(capture.source, CaptureSource::Target);
    assert!(capture.source_rect.is_none());
    assert_eq!(capture.sampling_size, Size::new(30, 30));
}

#[test]
fn nested_backdrop_captures_expand_parent_rendering_before_crop() {
    let mut scene = Scene::new();
    let root = scene.add(None);
    let parent = scene.tree.add_child(
        root,
        DrawTreeNode::ClipRect(ClipRectDrawData::new(
            [(30.0, 20.0), (50.0, 40.0)],
            None,
            false,
        )),
    );
    scene.attach_content(parent, 1, ContentEffectConfig::default());
    let child = scene.group(Some(parent), 2);
    let backdrop = scene.backdrop(child);
    scene.backdrops.insert(
        backdrop,
        BackdropEffectInstance::new(
            scene.backdrops[&backdrop].effect,
            BackdropEffectConfig::new().capture_area(BackdropCaptureArea::FullScene),
            scene.tree.get(backdrop).unwrap().logical_screen_bounds(),
            Viewport {
                physical_size: (100, 100),
                scale_factor: 1.0,
            },
            4096,
        ),
    );
    let mut output = RecordedCommands::default();
    scene.plan(&mut DrawPlanner::default(), &mut output);
    validate_dependencies(&output);
    let sizes: Vec<_> = output
        .instructions
        .iter()
        .filter_map(|command| match command.operation {
            RenderOperation::BeginTarget(Target::Texture { bounds, .. }) => {
                Some(bounds.size().to_u32())
            }
            _ => None,
        })
        .collect();
    assert_eq!(sizes, [Size::new(100, 100), Size::new(100, 100)]);
    let parent_crop = output
        .instructions
        .iter()
        .find_map(|command| match command.operation {
            RenderOperation::Capture(capture)
                if capture.destination_bounds.min.to_tuple() == (30, 20)
                    && capture.destination_bounds.max.to_tuple() == (50, 40) =>
            {
                Some(capture)
            }
            _ => None,
        })
        .expect("expanded parent rendering is cropped to the effect input");
    assert_eq!(parent_crop.source, CaptureSource::Target);
    assert_eq!(parent_crop.sampling_size, Size::new(20, 20));
}
