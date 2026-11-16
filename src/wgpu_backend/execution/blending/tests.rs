use super::fixed_blend_state;
use crate::core::BlendMode;
use std::array;
use wgpu::{BlendComponent, BlendFactor, BlendOperation};

fn factor_value(
    factor: BlendFactor,
    channel: usize,
    source: [f32; 4],
    destination: [f32; 4],
) -> f32 {
    match factor {
        BlendFactor::Zero => 0.0,
        BlendFactor::One => 1.0,
        BlendFactor::Src => source[channel],
        BlendFactor::OneMinusSrc => 1.0 - source[channel],
        BlendFactor::SrcAlpha => source[3],
        BlendFactor::OneMinusSrcAlpha => 1.0 - source[3],
        BlendFactor::Dst => destination[channel],
        BlendFactor::OneMinusDst => 1.0 - destination[channel],
        BlendFactor::DstAlpha => destination[3],
        BlendFactor::OneMinusDstAlpha => 1.0 - destination[3],
        _ => panic!("unexpected blend factor {factor:?}"),
    }
}

fn blend_channel(
    component: BlendComponent,
    channel: usize,
    source: [f32; 4],
    destination: [f32; 4],
) -> f32 {
    assert_eq!(component.operation, BlendOperation::Add);
    (source[channel] * factor_value(component.src_factor, channel, source, destination)
        + destination[channel] * factor_value(component.dst_factor, channel, source, destination))
    .clamp(0.0, 1.0)
}

fn reference_composite(mode: BlendMode, source: [f32; 4], destination: [f32; 4]) -> [f32; 4] {
    let source_alpha = source[3];
    let destination_alpha = destination[3];
    array::from_fn(|channel| {
        let source_channel = source[channel];
        let destination_channel = destination[channel];
        match mode {
            BlendMode::Clear => 0.0,
            BlendMode::Source => source_channel,
            BlendMode::Destination => destination_channel,
            BlendMode::SourceOver => source_channel + destination_channel * (1.0 - source_alpha),
            BlendMode::DestinationOver => {
                destination_channel + source_channel * (1.0 - destination_alpha)
            }
            BlendMode::SourceIn => source_channel * destination_alpha,
            BlendMode::DestinationIn => destination_channel * source_alpha,
            BlendMode::SourceOut => source_channel * (1.0 - destination_alpha),
            BlendMode::DestinationOut => destination_channel * (1.0 - source_alpha),
            BlendMode::SourceAtop => {
                source_channel * destination_alpha + destination_channel * (1.0 - source_alpha)
            }
            BlendMode::DestinationAtop => {
                destination_channel * source_alpha + source_channel * (1.0 - destination_alpha)
            }
            BlendMode::Xor => {
                source_channel * (1.0 - destination_alpha)
                    + destination_channel * (1.0 - source_alpha)
            }
            BlendMode::Plus => (source_channel + destination_channel).min(1.0),
            BlendMode::Screen | BlendMode::Exclusion if channel == 3 => {
                source_alpha + destination_alpha - source_alpha * destination_alpha
            }
            BlendMode::Screen | BlendMode::Exclusion => {
                let source_color = if source_alpha == 0.0 {
                    0.0
                } else {
                    source_channel / source_alpha
                };
                let destination_color = if destination_alpha == 0.0 {
                    0.0
                } else {
                    destination_channel / destination_alpha
                };
                let overlap_color = if mode == BlendMode::Screen {
                    source_color + destination_color - source_color * destination_color
                } else {
                    source_color + destination_color - 2.0 * source_color * destination_color
                };
                source_channel * (1.0 - destination_alpha)
                    + destination_channel * (1.0 - source_alpha)
                    + source_alpha * destination_alpha * overlap_color
            }
            _ => panic!("reference only covers fixed-function modes"),
        }
    })
}

#[test]
fn fixed_function_factors_match_compositing_equations_with_partial_alpha() {
    for mode in [
        BlendMode::Clear,
        BlendMode::Source,
        BlendMode::Destination,
        BlendMode::SourceOver,
        BlendMode::DestinationOver,
        BlendMode::SourceIn,
        BlendMode::DestinationIn,
        BlendMode::SourceOut,
        BlendMode::DestinationOut,
        BlendMode::SourceAtop,
        BlendMode::DestinationAtop,
        BlendMode::Xor,
        BlendMode::Plus,
        BlendMode::Screen,
        BlendMode::Exclusion,
    ] {
        let state = fixed_blend_state(mode).unwrap();
        for source_alpha in [0.0, 0.15, 0.65, 1.0] {
            for destination_alpha in [0.0, 0.25, 0.8, 1.0] {
                for (source_color, destination_color) in [
                    ([0.2, 0.7, 0.4], [0.6, 0.3, 0.9]),
                    ([1.0, 0.0, 0.5], [0.0, 1.0, 0.5]),
                ] {
                    let source = [
                        source_color[0] * source_alpha,
                        source_color[1] * source_alpha,
                        source_color[2] * source_alpha,
                        source_alpha,
                    ];
                    let destination = [
                        destination_color[0] * destination_alpha,
                        destination_color[1] * destination_alpha,
                        destination_color[2] * destination_alpha,
                        destination_alpha,
                    ];
                    let actual: [f32; 4] = array::from_fn(|channel| {
                        blend_channel(
                            if channel == 3 {
                                state.alpha
                            } else {
                                state.color
                            },
                            channel,
                            source,
                            destination,
                        )
                    });
                    let expected = reference_composite(mode, source, destination);
                    for (actual, expected) in actual.into_iter().zip(expected) {
                        assert!((actual - expected).abs() < 1e-6, "{mode:?}, source {source:?}, destination {destination:?}: {actual} != {expected}");
                    }
                }
            }
        }
    }
}
