use crate::caption_font;
use crate::expectations::PixelExpectation;
use crate::shaders::{PASSTHROUGH_WGSL, SHAPE_DROP_WGSL};
use grafo::{
    BackdropEffectConfig, BlendMode, BorderRadii, Color, ContentEffectConfig, Scene, Shape,
    ShapeDrawOptions, ShapeMaskEffectConfig, TextureManager, TransformInstance,
};
use reference::{EXPECTED, MODES, SAMPLES};

mod reference;

pub const BLEND_CANVAS_WIDTH: u32 = 728;
pub const BLEND_CANVAS_HEIGHT: u32 = 504;

const PASSTHROUGH_EFFECT: u64 = 98_001;
const MASK_EFFECT: u64 = 98_002;
const FOREGROUND_TEXTURE: u64 = 98_003;
const RETAINED_SOURCE_BOUNDS: [(f32, f32); 2] = [(616.0, 428.0), (664.0, 468.0)];
const RETAINED_DESTINATION_BOUNDS: [(f32, f32); 2] = [(612.0, 408.0), (724.0, 492.0)];

pub struct BlendModesScene {
    pub expectations: Vec<PixelExpectation>,
    source_node: usize,
    destination_node: usize,
}

impl BlendModesScene {
    pub fn replace_source(&self, scene: &mut Scene, fill: Color, mode: BlendMode) {
        scene
            .replace_with_shape(
                self.source_node,
                Shape::rect(RETAINED_SOURCE_BOUNDS),
                None,
                ShapeDrawOptions::new().color(fill).blend_mode(mode),
            )
            .unwrap();
    }

    pub fn replace_destination(&self, scene: &mut Scene, fill: Color) {
        scene
            .replace_with_shape(
                self.destination_node,
                Shape::rect(RETAINED_DESTINATION_BOUNDS),
                None,
                ShapeDrawOptions::new().color(fill),
            )
            .unwrap();
    }
}

fn color([red, green, blue, alpha]: [u8; 4]) -> Color {
    Color::rgba(red, green, blue, alpha)
}

fn rectangle(
    scene: &mut Scene,
    parent: usize,
    bounds: [(f32, f32); 2],
    fill: Color,
    mode: BlendMode,
) -> usize {
    scene
        .add_shape(
            Shape::rect(bounds),
            Some(parent),
            None,
            ShapeDrawOptions::new().color(fill).blend_mode(mode),
        )
        .unwrap()
}

fn caption(scene: &mut Scene, parent: usize, text: &str, x: f32, y: f32) {
    scene
        .add_shape(
            caption_font::shape(text),
            Some(parent),
            None,
            ShapeDrawOptions::new()
                .color(Color::rgb(180, 180, 180))
                .transform(TransformInstance::translation(x, y)),
        )
        .unwrap();
}

fn matrix(scene: &mut Scene, root: usize, expectations: &mut Vec<PixelExpectation>) {
    for (mode_index, (mode, name)) in MODES.into_iter().enumerate() {
        let origin_x = (mode_index % 7) as f32 * 104.0;
        let origin_y = (mode_index / 7) as f32 * 96.0;
        caption(scene, root, name, origin_x + 4.0, origin_y + 4.0);
        for (sample_index, (source, destination)) in SAMPLES.into_iter().enumerate() {
            let x = origin_x + 4.0 + (sample_index % 3) as f32 * 32.0;
            let y = origin_y + 16.0 + (sample_index / 3) as f32 * 24.0;
            let bounds = [(x, y), (x + 28.0, y + 20.0)];
            rectangle(
                scene,
                root,
                bounds,
                color(destination),
                BlendMode::SourceOver,
            );
            let source = rectangle(scene, root, bounds, color(source), mode);
            if sample_index == 1 {
                scene
                    .set_content_effect(
                        source,
                        PASSTHROUGH_EFFECT,
                        &[],
                        ContentEffectConfig::new().downsample(0.5),
                    )
                    .unwrap();
            }
            let [red, green, blue, alpha] = EXPECTED[mode_index][sample_index];
            expectations.push(
                PixelExpectation::new(x as u32 + 14, y as u32 + 10, red, green, blue, alpha, name)
                    .with_tolerance(3),
            );
        }
    }
}

fn captured_rectangle(scene: &mut Scene, root: usize) -> Vec<PixelExpectation> {
    caption(scene, root, "CAPTURE RECTANGLE", 4.0, 392.0);
    rectangle(
        scene,
        root,
        [(4.0, 408.0), (140.0, 492.0)],
        Color::rgb(0, 0, 255),
        BlendMode::SourceOver,
    );
    for (mode, top) in [(BlendMode::Source, 416.0), (BlendMode::Clear, 456.0)] {
        let source = scene
            .add_shape(
                Shape::rounded_rect([(12.0, top), (132.0, top + 28.0)], BorderRadii::new(12.0)),
                Some(root),
                None,
                ShapeDrawOptions::new()
                    .color(Color::TRANSPARENT)
                    .blend_mode(mode),
            )
            .unwrap();
        for left in [24.0, 96.0] {
            rectangle(
                scene,
                source,
                [(left, top + 6.0), (left + 24.0, top + 22.0)],
                Color::rgb(0, 255, 0),
                BlendMode::SourceOver,
            );
        }
    }
    vec![
        PixelExpectation::opaque(6, 440, 0, 0, 255, "source_outside_capture"),
        PixelExpectation::opaque(36, 430, 0, 255, 0, "source_child"),
        PixelExpectation::transparent(72, 430, "source_transparent_hole"),
        PixelExpectation::transparent(13, 417, "source_rounded_corner"),
        PixelExpectation::transparent(36, 470, "clear_opaque_child"),
        PixelExpectation::transparent(72, 470, "clear_transparent_hole"),
        PixelExpectation::transparent(13, 457, "clear_rounded_corner"),
    ]
}

fn padded_source(scene: &mut Scene, root: usize) -> Vec<PixelExpectation> {
    caption(scene, root, "PADDED SOURCE", 148.0, 392.0);
    rectangle(
        scene,
        root,
        [(148.0, 408.0), (284.0, 492.0)],
        Color::rgb(255, 0, 0),
        BlendMode::SourceOver,
    );
    let source = scene
        .add_shape(
            Shape::rounded_rect([(172.0, 424.0), (260.0, 464.0)], BorderRadii::new(12.0)),
            Some(root),
            None,
            ShapeDrawOptions::new()
                .color(Color::rgb(0, 0, 255))
                .blend_mode(BlendMode::Source),
        )
        .unwrap();
    scene
        .set_content_effect(
            source,
            PASSTHROUGH_EFFECT,
            &[],
            ContentEffectConfig::new().outset(8.0).downsample(0.5),
        )
        .unwrap();
    vec![
        PixelExpectation::opaque(154, 444, 255, 0, 0, "source_outside_padding"),
        PixelExpectation::opaque(212, 444, 0, 0, 255, "source_effect_center"),
        PixelExpectation::transparent(166, 444, "source_effect_padding"),
        PixelExpectation::transparent(172, 424, "source_effect_rounded_corner"),
        PixelExpectation::transparent(212, 470, "source_effect_bottom_padding"),
    ]
}

fn nested_targets(scene: &mut Scene, root: usize) -> Vec<PixelExpectation> {
    caption(scene, root, "NESTED TARGET", 296.0, 392.0);
    let outer = scene
        .add_shape(
            Shape::rounded_rect([(296.0, 408.0), (472.0, 492.0)], BorderRadii::new(12.0)),
            Some(root),
            None,
            ShapeDrawOptions::new().color(color(SAMPLES[0].1)),
        )
        .unwrap();
    let clip = scene
        .add_clipping_rect(
            [(308.0, 416.0), (460.0, 480.0)],
            Some(outer),
            None::<TransformInstance>,
            true,
        )
        .unwrap();
    let inner = rectangle(
        scene,
        clip,
        [(288.0, 432.0), (480.0, 464.0)],
        color(SAMPLES[7].0),
        BlendMode::Multiply,
    );
    scene
        .add_shape(
            Shape::rect([(324.0, 436.0), (352.0, 460.0)]),
            Some(inner),
            None,
            ShapeDrawOptions::new().foreground_texture_id(FOREGROUND_TEXTURE),
        )
        .unwrap();
    for (node, config) in [
        (outer, ContentEffectConfig::new().downsample(0.5)),
        (
            inner,
            ContentEffectConfig::new().outset(6.0).downsample(0.5),
        ),
    ] {
        scene
            .set_content_effect(node, PASSTHROUGH_EFFECT, &[], config)
            .unwrap();
    }
    vec![
        PixelExpectation::opaque(380, 448, 44, 93, 179, "nested_blend_immediate_destination"),
        PixelExpectation::opaque(338, 448, 0, 137, 0, "nested_blend_textured_source"),
        PixelExpectation::opaque(302, 448, 48, 137, 224, "nested_blend_scissor_left"),
        PixelExpectation::opaque(466, 448, 48, 137, 224, "nested_blend_scissor_right"),
        PixelExpectation::opaque(380, 420, 48, 137, 224, "nested_blend_target_restored"),
        PixelExpectation::transparent(296, 408, "nested_blend_parent_stencil"),
    ]
}

fn masked_backdrop(scene: &mut Scene, root: usize) -> Vec<PixelExpectation> {
    caption(scene, root, "MASK BACKDROP", 488.0, 392.0);
    rectangle(
        scene,
        root,
        [(488.0, 408.0), (600.0, 492.0)],
        Color::WHITE,
        BlendMode::SourceOver,
    );
    let source = rectangle(
        scene,
        root,
        [(512.0, 424.0), (552.0, 464.0)],
        Color::TRANSPARENT,
        BlendMode::Difference,
    );
    scene
        .set_shape_mask_effect(
            source,
            MASK_EFFECT,
            &[],
            ShapeMaskEffectConfig::new().outset(12.0),
        )
        .unwrap();
    scene
        .set_backdrop_effect(
            source,
            PASSTHROUGH_EFFECT,
            &[],
            BackdropEffectConfig::default(),
        )
        .unwrap();
    vec![
        PixelExpectation::opaque(536, 448, 255, 255, 0, "blend_shape_mask_with_backdrop"),
        PixelExpectation::opaque(556, 448, 255, 255, 0, "blend_shape_mask_outset"),
        PixelExpectation::opaque(592, 448, 255, 255, 255, "blend_outside_mask_bounds"),
    ]
}

fn restored_foreground(scene: &mut Scene, root: usize) -> (Vec<PixelExpectation>, usize, usize) {
    caption(scene, root, "BINDINGS", 612.0, 392.0);
    let destination = rectangle(
        scene,
        root,
        RETAINED_DESTINATION_BOUNDS,
        color(SAMPLES[0].1),
        BlendMode::SourceOver,
    );
    let foreground = |scene: &mut Scene, bounds| {
        scene
            .add_shape(
                Shape::rect(bounds),
                Some(root),
                None,
                ShapeDrawOptions::new().foreground_texture_id(FOREGROUND_TEXTURE),
            )
            .unwrap();
    };
    foreground(scene, [(616.0, 412.0), (640.0, 420.0)]);
    let source = rectangle(
        scene,
        root,
        RETAINED_SOURCE_BOUNDS,
        color(SAMPLES[0].0),
        BlendMode::Overlay,
    );
    foreground(scene, [(680.0, 428.0), (720.0, 468.0)]);
    (
        vec![
            PixelExpectation::opaque(628, 416, 0, 255, 0, "foreground_before_blend"),
            PixelExpectation::opaque(640, 448, 59, 50, 213, "sampled_blend_before_foreground"),
            PixelExpectation::opaque(700, 448, 0, 255, 0, "foreground_binding_after_blend"),
        ],
        source,
        destination,
    )
}

pub fn build_blend_modes_scene(scene: &mut Scene) -> BlendModesScene {
    scene
        .load_effect(PASSTHROUGH_EFFECT, &[PASSTHROUGH_WGSL])
        .unwrap();
    scene.load_effect(MASK_EFFECT, &[SHAPE_DROP_WGSL]).unwrap();
    scene.texture_manager().allocate_texture_with_data(
        FOREGROUND_TEXTURE,
        (1, 1),
        &[0, 255, 0, 255],
    );
    let root = scene.add_group(None, Default::default()).unwrap();
    let mut expectations = Vec::new();
    matrix(scene, root, &mut expectations);
    expectations.extend(captured_rectangle(scene, root));
    expectations.extend(padded_source(scene, root));
    expectations.extend(nested_targets(scene, root));
    expectations.extend(masked_backdrop(scene, root));
    let (foreground_expectations, source_node, destination_node) = restored_foreground(scene, root);
    expectations.extend(foreground_expectations);
    BlendModesScene {
        expectations,
        source_node,
        destination_node,
    }
}
