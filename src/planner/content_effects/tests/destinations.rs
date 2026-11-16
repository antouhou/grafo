use super::*;
use crate::commands::BlendDestination;
use crate::core::PhysicalRect;
use crate::planner::content_effects::ContentCapture;

#[test]
fn content_composites_capture_destination_pixels_only_when_needed() {
    let node = DrawTreeNode::ClipRect(ClipRectDrawData::new(
        [(10.0, 20.0), (50.0, 60.0)],
        None,
        false,
    ));
    let viewport = Viewport {
        physical_size: (100, 100),
        scale_factor: 1.0,
    };
    let bounds = UnsignedPhysicalRect::new((10, 20).into(), (50, 60).into());
    let mut parameters = EffectParameterStorage::default();
    let effect = ContentEffectInstance {
        effect_id: 7,
        parameters: parameters.store(&[1, 2, 3, 4]),
        config: ContentEffectConfig::default(),
    };
    for (blend_mode, captures_destination) in [
        (BlendMode::Clear, false),
        (BlendMode::Source, false),
        (BlendMode::Destination, false),
        (BlendMode::SourceOver, false),
        (BlendMode::DestinationOver, false),
        (BlendMode::SourceIn, false),
        (BlendMode::DestinationIn, false),
        (BlendMode::SourceOut, false),
        (BlendMode::DestinationOut, false),
        (BlendMode::SourceAtop, false),
        (BlendMode::DestinationAtop, false),
        (BlendMode::Xor, false),
        (BlendMode::Plus, false),
        (BlendMode::Screen, false),
        (BlendMode::Overlay, true),
        (BlendMode::Darken, true),
        (BlendMode::Lighten, true),
        (BlendMode::ColorDodge, true),
        (BlendMode::ColorBurn, true),
        (BlendMode::HardLight, true),
        (BlendMode::SoftLight, true),
        (BlendMode::Difference, true),
        (BlendMode::Exclusion, false),
        (BlendMode::Multiply, true),
        (BlendMode::Hue, true),
        (BlendMode::Saturation, true),
        (BlendMode::Color, true),
        (BlendMode::Luminosity, true),
    ] {
        for has_effect in [false, true] {
            let group =
                ContentGroup::new(has_effect.then_some(effect), &node, None, viewport, 4096);
            let mut texture_ids = TextureIdAllocator::default();
            let capture = ContentCapture::new(group, 1.0, None, &mut texture_ids).unwrap();
            let mut commands = TraversalCommands::default();
            commands.push(RenderOperation::BeginTarget(Target::Surface));
            capture.append_begin(&mut commands);
            let texture = capture.append_end(&mut commands, &mut texture_ids);
            let composite = capture.prepare_composite(
                texture,
                blend_mode,
                viewport.physical_size.into(),
                &parameters,
                &mut commands,
                &mut texture_ids,
            );
            commands.push_composite(composite, Default::default());
            commands.push(RenderOperation::EndTarget);
            let output = RecordedCommands {
                instructions: commands.into_iter().collect(),
            };
            validate_dependencies(&output);

            let source_index = usize::from(has_effect);
            let destination_index = source_index + 1;
            let mut expected = vec![Command::Begin(None), Command::Begin(Some(0)), Command::End];
            if has_effect {
                expected.push(Command::Effect(7, 0, source_index));
            }
            if captures_destination {
                expected.push(Command::Capture(destination_index));
            }
            expected.extend([Command::Composite(source_index), Command::End]);
            assert_eq!(snapshot(&output), expected, "{blend_mode:?}, {has_effect}");
            assert_eq!(composite.blend_mode, blend_mode);
            assert_eq!(
                composite.source,
                IntermediateTextureId::Planned(source_index)
            );
            assert_eq!(
                composite.destination,
                if captures_destination {
                    BlendDestination::Texture(IntermediateTextureId::Planned(destination_index))
                } else {
                    BlendDestination::CurrentTarget
                }
            );
            assert_eq!(
                texture_ids.allocate(),
                IntermediateTextureId::Planned(
                    destination_index + usize::from(captures_destination)
                )
            );
            for command in &output.instructions {
                match command.operation {
                    RenderOperation::Capture(destination) => {
                        assert_eq!(destination.source, CaptureSource::Target);
                        assert_eq!(destination.source_rect, Some(bounds));
                        assert_eq!(destination.destination_bounds, bounds.to_i32());
                        assert_eq!(destination.sampling_size, Size::new(40, 40));
                    }
                    RenderOperation::ApplyEffect(effect) => {
                        assert_eq!(effect.parameters.bytes, &[1, 2, 3, 4]);
                    }
                    _ => {}
                }
            }
        }
    }
}

#[test]
fn destination_captures_keep_full_resolution_padded_output_bounds() {
    let viewport = Viewport {
        physical_size: (100, 100),
        scale_factor: 1.0,
    };
    let mut parameters = EffectParameterStorage::default();
    let effect = ContentEffectInstance {
        effect_id: 9,
        parameters: parameters.store(&[1, 2, 3, 4]),
        config: ContentEffectConfig::new()
            .outsets(4.0, 2.0, 6.0, 8.0)
            .downsample(0.5),
    };
    for (input_bounds, output_bounds, source_rect, destination_rect) in [
        (
            [(-3.0, 20.0), (21.0, 44.0)],
            PhysicalRect::new((-7, 18).into(), (27, 52).into()),
            Some(UnsignedPhysicalRect::new((0, 20).into(), (21, 44).into())),
            Some(UnsignedPhysicalRect::new((0, 18).into(), (27, 52).into())),
        ),
        (
            [(120.0, 20.0), (144.0, 44.0)],
            PhysicalRect::new((116, 18).into(), (150, 52).into()),
            None,
            None,
        ),
    ] {
        let node = DrawTreeNode::ClipRect(ClipRectDrawData::new(input_bounds, None, false));
        let group = ContentGroup::new(Some(effect), &node, None, viewport, 4096);
        assert_eq!(group.output_bounds, Some(output_bounds));
        assert_eq!(group.input_bounds, source_rect);
        let mut texture_ids = TextureIdAllocator::default();
        let capture = ContentCapture::new(group, 1.0, None, &mut texture_ids).unwrap();
        let mut commands = TraversalCommands::default();
        commands.push(RenderOperation::BeginTarget(Target::Surface));
        capture.append_begin(&mut commands);
        let texture = capture.append_end(&mut commands, &mut texture_ids);
        let composite = capture.prepare_composite(
            texture,
            BlendMode::Multiply,
            viewport.physical_size.into(),
            &parameters,
            &mut commands,
            &mut texture_ids,
        );
        commands.push_composite(composite, Default::default());
        commands.push(RenderOperation::EndTarget);
        let output = RecordedCommands {
            instructions: commands.into_iter().collect(),
        };
        validate_dependencies(&output);
        assert_eq!(
            snapshot(&output),
            [
                Command::Begin(None),
                Command::Begin(Some(0)),
                Command::Capture(1),
                Command::End,
                Command::Effect(9, 1, 2),
                Command::Capture(3),
                Command::Composite(2),
                Command::End,
            ]
        );
        let RenderOperation::Capture(source) = output.instructions[2].operation else {
            panic!("source content is downsampled before its target closes");
        };
        assert_eq!(source.source, CaptureSource::Target);
        assert_eq!(source.source_rect, source_rect);
        assert_eq!(source.destination_bounds, output_bounds);
        assert_eq!(source.sampling_size, Size::new(17, 17));
        let RenderOperation::Capture(destination) = output.instructions[5].operation else {
            panic!("destination is captured after the effect and parent restoration");
        };
        assert_eq!(destination.source, CaptureSource::Target);
        assert_eq!(destination.source_rect, destination_rect);
        assert_eq!(destination.destination_bounds, output_bounds);
        assert_eq!(destination.sampling_size, Size::new(34, 34));
        assert_eq!(
            composite.destination,
            BlendDestination::Texture(destination.output)
        );
        assert_eq!(composite.source, IntermediateTextureId::Planned(2));
        assert_eq!(texture_ids.allocate(), IntermediateTextureId::Planned(4));
    }
}
