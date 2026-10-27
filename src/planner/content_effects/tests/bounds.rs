use super::*;
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
    scene.plan(&mut SceneTraversal::default(), &mut output);
    let targets: Vec<_> = output
        .instructions
        .iter()
        .filter_map(|command| match command.operation {
            RenderOperation::BeginTarget(Target::Texture { region, .. }) => {
                Some(region.bounds.size().to_u32())
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
    scene.plan(&mut SceneTraversal::default(), &mut output);
    validate_dependencies(&output);
    let RenderOperation::BeginTarget(Target::Texture { region, .. }) =
        output.instructions[1].operation
    else {
        panic!("content starts a bounded target");
    };
    assert_eq!(region.bounds.min.to_tuple(), (-7, 16));
    assert_eq!(region.bounds.size().to_tuple(), (32, 32));
    assert_eq!(region.source_rect.unwrap().min.to_tuple(), (0, 20));
    let RenderOperation::Capture(capture) = output.instructions[2].operation else {
        panic!("downsampling happens after rendering");
    };
    assert_eq!(capture.region, region);
    assert_eq!(capture.sampling_size, Size::new(16, 16));
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
    scene.plan(&mut SceneTraversal::default(), &mut output);
    validate_dependencies(&output);
    let regions: Vec<_> = output
        .instructions
        .iter()
        .filter_map(|command| match command.operation {
            RenderOperation::BeginTarget(Target::Texture { region, .. }) => Some(region),
            _ => None,
        })
        .collect();
    assert_eq!(regions.len(), 1);
    assert_eq!(regions[0].bounds.min.to_tuple(), (6, 6));
    assert_eq!(regions[0].bounds.max.to_tuple(), (50, 44));
    let capture = output
        .instructions
        .iter()
        .find_map(|command| match command.operation {
            RenderOperation::Capture(capture)
                if capture.region.bounds.min.to_tuple() == (26, 18) =>
            {
                Some(capture)
            }
            _ => None,
        })
        .unwrap();
    assert_eq!(capture.region.bounds.min.to_tuple(), (26, 18));
    assert_eq!(capture.region.bounds.max.to_tuple(), (56, 48));
    assert_eq!(capture.region.source_rect.unwrap().min.to_tuple(), (30, 20));
    assert_eq!(capture.region.source_rect.unwrap().max.to_tuple(), (50, 40));
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
    scene.plan(&mut SceneTraversal::default(), &mut output);
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
    assert!(capture.region.source_rect.is_none());
    assert_eq!(capture.sampling_size, Size::new(30, 30));
}

#[test]
fn completed_child_captures_do_not_expand_the_parent_target() {
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
    scene.plan(&mut SceneTraversal::default(), &mut output);
    validate_dependencies(&output);
    let sizes: Vec<_> = output
        .instructions
        .iter()
        .filter_map(|command| match command.operation {
            RenderOperation::BeginTarget(Target::Texture { region, .. }) => {
                Some(region.bounds.size().to_u32())
            }
            _ => None,
        })
        .collect();
    assert_eq!(sizes, [Size::new(20, 20), Size::new(100, 100)]);
}
