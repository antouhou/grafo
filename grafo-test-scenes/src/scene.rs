use grafo::{
    premultiply_rgba8_srgb_inplace, BackdropCaptureArea, BackdropEffectConfig, BorderRadii, Color,
    ColorInterpolation, ConicGradientDesc, Fill, Gradient, GradientColor, GradientCommonDesc,
    GradientStop, GradientStopOffset, GradientStopPositions, GradientUnits, LinearGradientDesc,
    LinearGradientLine, RadialGradientDesc, RadialGradientShape, RadialGradientSize, Renderer,
    Shape, ShapeDrawCommandOptions, ShapeEffectConfig, ShapeTextureFitMode, ShapeTextureOptions,
    SpreadMode, Stroke, TransformInstance,
};

use crate::expectations::PixelExpectation;
use crate::shaders::{
    BlurParams, DOWNSAMPLED_DROP_SHADOW_HORIZONTAL_BLUR_WGSL,
    DOWNSAMPLED_DROP_SHADOW_VERTICAL_TINT_WGSL, DROP_SHADOW_HORIZONTAL_BLUR_WGSL,
    DROP_SHADOW_VERTICAL_TINT_WGSL, HORIZONTAL_BLUR_WGSL, PASSTHROUGH_WGSL, SHAPE_DROP_WGSL,
    VERTICAL_BLUR_WGSL,
};

// Grid layout

const TILE_SIZE: u32 = 80;
const COLUMNS: u32 = 6;
const ROWS: u32 = 12;

pub const CANVAS_WIDTH: u32 = TILE_SIZE * COLUMNS;
pub const CANVAS_HEIGHT: u32 = TILE_SIZE * ROWS;

const BLUR_EFFECT_ID: u64 = 1;
const PASSTHROUGH_EFFECT_ID: u64 = 2;
const SHAPE_DROP_EFFECT_ID: u64 = 3;
const DROP_SHADOW_EFFECT_ID: u64 = 4;
const DOWNSAMPLED_DROP_SHADOW_EFFECT_ID: u64 = 5;
const CHECKERBOARD_TEXTURE_ID: u64 = 100;
const SOLID_GREEN_TEXTURE_ID: u64 = 101;
const SOLID_GREEN_20X20_TEXTURE_ID: u64 = 102;
const SOLID_RED_TEXTURE_ID: u64 = 103;
const TRANSLUCENT_CHECKERBOARD_TEXTURE_ID: u64 = 104;

/// Returns the top-left pixel of the tile. Tile numbers start at one.
fn tile_origin(tile_number: u32) -> (f32, f32) {
    let index = tile_number - 1;
    let column = index % COLUMNS;
    let row = index / COLUMNS;
    ((column * TILE_SIZE) as f32, (row * TILE_SIZE) as f32)
}

/// Queues the tile grid and returns its expected pixel colors.
/// The headless regression test and visual_test_grid example share this scene.
pub fn build_main_scene(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let mut expectations: Vec<PixelExpectation> = Vec::new();

    // The draw tree requires a single root node. All tile shapes are added as
    // children of this full-canvas root so they are not clipped to each other.
    let canvas_root = Shape::rect(
        [(0.0, 0.0), (CANVAS_WIDTH as f32, CANVAS_HEIGHT as f32)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            canvas_root,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::WHITE),
        )
        .unwrap();

    load_shared_resources(renderer);

    expectations.extend(tile_01_rect_solid(renderer));
    expectations.extend(tile_02_rounded_rect_solid(renderer));
    expectations.extend(tile_03_path_triangle(renderer));
    expectations.extend(tile_04_path_bezier(renderer));
    expectations.extend(tile_05_rect_parent_child_inside(renderer));
    expectations.extend(tile_06_rect_parent_child_overflow(renderer));
    expectations.extend(tile_07_rect_parent_multi_children(renderer));
    expectations.extend(tile_08_rect_nested_3_levels(renderer));
    expectations.extend(tile_09_rect_siblings_overlap(renderer));
    expectations.extend(tile_10_rounded_rect_clip(renderer));
    expectations.extend(tile_11_path_parent_clip(renderer));
    expectations.extend(tile_12_stencil_nested_3_levels(renderer));
    expectations.extend(tile_13_rotated_rect_clip(renderer));
    expectations.extend(tile_14_scissor_then_stencil(renderer));
    expectations.extend(tile_15_stencil_then_scissor(renderer));
    expectations.extend(tile_16_deep_mixed_5_levels(renderer));
    expectations.extend(tile_17_translated_rect(renderer));
    expectations.extend(tile_18_scaled_rect(renderer));
    expectations.extend(tile_19_rotated_rect_leaf(renderer));
    expectations.extend(tile_20_transform_parent_child(renderer));
    expectations.extend(tile_21_alpha_overlap(renderer));
    expectations.extend(tile_22_no_color_default(renderer));
    expectations.extend(tile_23_fully_transparent(renderer));
    expectations.extend(tile_24_textured_rect(renderer));
    expectations.extend(tile_25_textured_with_color(renderer));
    expectations.extend(tile_26_textured_parent_child(renderer));
    expectations.extend(tile_27_group_blur_leaf(renderer));
    expectations.extend(tile_28_group_blur_with_children(renderer));
    expectations.extend(tile_29_backdrop_blur_leaf(renderer));
    expectations.extend(tile_30_backdrop_blur_nonleaf(renderer));
    expectations.extend(tile_31_backdrop_under_scissor(renderer));
    expectations.extend(tile_32_tiny_1px_shape(renderer));
    expectations.extend(tile_33_shape_at_canvas_edge(renderer));
    expectations.extend(tile_34_cached_shape(renderer));
    expectations.extend(tile_35_trivial_transform_transparent_leaf(renderer));
    expectations.extend(tile_36_trivial_transform_transparent_parent(renderer));
    expectations.extend(tile_37_textured_transparent_rects(renderer));
    expectations.extend(tile_38_sheared_transparent_parent(renderer));

    // Gradient tiles
    expectations.extend(tile_39_linear_gradient(renderer));
    expectations.extend(tile_40_radial_gradient(renderer));
    expectations.extend(tile_41_conic_gradient(renderer));
    expectations.extend(tile_42_repeating_linear_gradient(renderer));
    expectations.extend(tile_43_gradient_hard_stops(renderer));
    expectations.extend(tile_44_gradient_clipped(renderer));
    expectations.extend(tile_45_gradient_group_blur(renderer));
    expectations.extend(tile_46_gradient_backdrop_blur(renderer));

    // Gradient regression tiles
    expectations.extend(tile_47_gradient_nonleaf_stencil(renderer));
    expectations.extend(tile_48_gradient_state_leak(renderer));
    expectations.extend(tile_49_conic_quadrant_colors(renderer));
    expectations.extend(tile_50_overflow_visible_delegates_to_ancestor(renderer));
    expectations.extend(tile_51_clip_rect_overflow_visible_container(renderer));
    expectations.extend(tile_52_backdrop_overflow_visible_children(renderer));
    expectations.extend(tile_53_texture_original_size(renderer));
    expectations.extend(tile_54_texture_original_size_scaled(renderer));
    expectations.extend(tile_55_backdrop_capture_screen_rect(renderer));
    expectations.extend(tile_56_backdrop_capture_downsampled(renderer));
    expectations.extend(tile_57_gradient_backdrop_oversized_capture_falls_back(
        renderer,
    ));
    expectations.extend(tile_58_backdrop_budgeted_capture_falls_back(renderer));
    expectations.extend(tile_59_backdrop_node_bounds_offscreen_preserves_size(
        renderer,
    ));
    expectations.extend(tile_60_cached_shape_effect_rect(renderer));
    expectations.extend(tile_61_cached_shape_effect_path_clipped(renderer));
    expectations.extend(tile_62_cached_shape_effect_inside_group_effect(renderer));
    expectations.extend(tile_63_cached_shape_effect_with_backdrop(renderer));
    expectations.extend(tile_64_drop_shadow_with_backdrop_blur(renderer));
    expectations.extend(tile_65_grouped_shape_effect_in_backdrop(renderer));
    expectations.extend(tile_66_same_node_shape_backdrop_and_group_effects(renderer));
    expectations.extend(tile_67_downsampled_drop_shadow_with_backdrop_blur(renderer));
    expectations.extend(tile_68_gradient_transition_hints(renderer));
    expectations.extend(tile_69_gradient_automatic_stop_after_decreasing_stop(
        renderer,
    ));

    expectations
}

fn tile_68_gradient_transition_hints(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (origin_x, origin_y) = tile_origin(68);
    for (top, bottom, hint) in [(8.0, 36.0, 0.25), (44.0, 72.0, 0.75)] {
        let gradient = Gradient::linear(LinearGradientDesc::new(
            LinearGradientLine {
                start: [origin_x + 8.5, origin_y],
                end: [origin_x + 72.5, origin_y],
            },
            [
                GradientStop::at_position(GradientStopOffset::linear_radial(0.0), Color::BLACK)
                    .with_hint_to_next_segment(GradientStopOffset::linear_radial(hint)),
                GradientStop::at_position(GradientStopOffset::linear_radial(1.0), Color::WHITE),
            ],
        ))
        .unwrap();
        renderer
            .add_shape(
                Shape::rect(
                    [
                        (origin_x + 8.0, origin_y + top),
                        (origin_x + 72.0, origin_y + bottom),
                    ],
                    Stroke::default(),
                ),
                None,
                None,
                ShapeDrawCommandOptions::new().fill(Fill::Gradient(gradient)),
            )
            .unwrap();
    }

    // 255 * P^log_H(0.5), with P measured at the pixel center.
    [
        (12, 20, 64, "t68_quarter_hint_before"),
        (48, 20, 202, "t68_quarter_hint_after"),
        (40, 56, 48, "t68_three_quarter_hint_before"),
        (64, 56, 185, "t68_three_quarter_hint_after"),
    ]
    .into_iter()
    .map(|(x, y, channel, label)| {
        PixelExpectation::opaque(
            origin_x as u32 + x,
            origin_y as u32 + y,
            channel,
            channel,
            channel,
            label,
        )
        .with_tolerance(2)
    })
    .collect()
}

fn tile_69_gradient_automatic_stop_after_decreasing_stop(
    renderer: &mut Renderer,
) -> Vec<PixelExpectation> {
    let (origin_x, origin_y) = tile_origin(69);
    let gradient = Gradient::linear(LinearGradientDesc::new(
        LinearGradientLine {
            start: [origin_x + 8.5, origin_y],
            end: [origin_x + 72.5, origin_y],
        },
        [
            GradientStop::at_position(GradientStopOffset::linear_radial(0.5), Color::BLACK),
            GradientStop::at_position(
                GradientStopOffset::linear_radial(0.25),
                Color::rgb(255, 0, 0),
            ),
            GradientStop::auto(Color::rgb(0, 255, 0)),
            GradientStop::at_position(
                GradientStopOffset::linear_radial(1.0),
                Color::rgb(0, 0, 255),
            ),
        ],
    ))
    .unwrap();
    renderer
        .add_shape(
            Shape::rect(
                [
                    (origin_x + 8.0, origin_y + 8.0),
                    (origin_x + 72.0, origin_y + 72.0),
                ],
                Stroke::default(),
            ),
            None,
            None,
            ShapeDrawCommandOptions::new().fill(Fill::Gradient(gradient)),
        )
        .unwrap();

    // Corrected stop positions are 0.5, 0.5, 0.75, 1.0.
    vec![
        PixelExpectation::opaque(
            origin_x as u32 + 48,
            origin_y as u32 + 40,
            128,
            128,
            0,
            "t69_red_to_automatic_green",
        )
        .with_tolerance(2),
        PixelExpectation::opaque(
            origin_x as u32 + 60,
            origin_y as u32 + 40,
            0,
            191,
            64,
            "t69_automatic_green_to_blue",
        )
        .with_tolerance(2),
    ]
}

// Shared resource setup

fn load_shared_resources(renderer: &mut Renderer) {
    renderer
        .load_effect(BLUR_EFFECT_ID, &[HORIZONTAL_BLUR_WGSL, VERTICAL_BLUR_WGSL])
        .expect("Failed to compile blur effect");
    renderer
        .load_effect(PASSTHROUGH_EFFECT_ID, &[PASSTHROUGH_WGSL])
        .expect("Failed to compile passthrough effect");
    renderer
        .load_effect(SHAPE_DROP_EFFECT_ID, &[SHAPE_DROP_WGSL])
        .expect("Failed to compile cached shape effect");
    renderer
        .load_effect(
            DROP_SHADOW_EFFECT_ID,
            &[
                DROP_SHADOW_HORIZONTAL_BLUR_WGSL,
                DROP_SHADOW_VERTICAL_TINT_WGSL,
            ],
        )
        .expect("Failed to compile visual drop shadow effect");
    renderer
        .load_effect(
            DOWNSAMPLED_DROP_SHADOW_EFFECT_ID,
            &[
                DOWNSAMPLED_DROP_SHADOW_HORIZONTAL_BLUR_WGSL,
                DOWNSAMPLED_DROP_SHADOW_VERTICAL_TINT_WGSL,
            ],
        )
        .expect("Failed to compile downsampled drop shadow effect");

    // 4×4 checkerboard: alternating white and black pixels, RGBA
    let mut checkerboard = [0u8; 4 * 4 * 4];
    for row in 0..4u32 {
        for col in 0..4u32 {
            let is_white = (row + col) % 2 == 0;
            let offset = ((row * 4 + col) * 4) as usize;
            let value = if is_white { 255 } else { 0 };
            checkerboard[offset] = value;
            checkerboard[offset + 1] = value;
            checkerboard[offset + 2] = value;
            checkerboard[offset + 3] = 255;
        }
    }
    renderer.texture_manager().allocate_texture_with_data(
        CHECKERBOARD_TEXTURE_ID,
        (4, 4),
        &checkerboard,
    );
    for pixel in checkerboard.as_chunks_mut::<4>().0 {
        pixel[3] = 128;
    }
    premultiply_rgba8_srgb_inplace(&mut checkerboard);
    renderer.texture_manager().allocate_texture_with_data(
        TRANSLUCENT_CHECKERBOARD_TEXTURE_ID,
        (4, 4),
        &checkerboard,
    );
    renderer.texture_manager().allocate_texture_with_data(
        SOLID_GREEN_TEXTURE_ID,
        (1, 1),
        &[0, 255, 0, 255],
    );
    renderer.texture_manager().allocate_texture_with_data(
        SOLID_RED_TEXTURE_ID,
        (1, 1),
        &[255, 0, 0, 255],
    );

    let solid_green_20x20 = (0..20u32)
        .flat_map(|y| {
            (0..20u32).flat_map(move |x| {
                if x == 0 || x == 19 || y == 0 || y == 19 {
                    [0u8, 0u8, 0u8, 0u8]
                } else {
                    [0u8, 255u8, 0u8, 255u8]
                }
            })
        })
        .collect::<Vec<_>>();
    renderer.texture_manager().allocate_texture_with_data(
        SOLID_GREEN_20X20_TEXTURE_ID,
        (20, 20),
        &solid_green_20x20,
    );
}

// Basic shapes

fn tile_01_rect_solid(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(1);
    let shape = Shape::rect(
        [(ox + 10.0, oy + 10.0), (ox + 70.0, oy + 70.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            shape,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(220, 50, 50)),
        )
        .unwrap();

    vec![
        PixelExpectation::opaque(ox as u32 + 40, oy as u32 + 40, 220, 50, 50, "t01_interior"),
        PixelExpectation::opaque(
            ox as u32 + 5,
            oy as u32 + 5,
            255,
            255,
            255,
            "t01_outside_is_canvas_bg",
        ),
    ]
}

fn tile_02_rounded_rect_solid(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(2);
    let shape = Shape::rounded_rect(
        [(ox + 10.0, oy + 10.0), (ox + 70.0, oy + 70.0)],
        BorderRadii::new(15.0),
        Stroke::default(),
    );
    renderer
        .add_shape(
            shape,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(50, 180, 50)),
        )
        .unwrap();

    vec![
        PixelExpectation::opaque(ox as u32 + 40, oy as u32 + 40, 50, 180, 50, "t02_interior"),
        // The rounded corner leaves this pixel outside the shape, showing the canvas.
        PixelExpectation::opaque(
            ox as u32 + 11,
            oy as u32 + 11,
            255,
            255,
            255,
            "t02_corner_is_bg",
        ),
    ]
}

fn tile_03_path_triangle(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(3);
    let shape = Shape::builder()
        .begin((ox + 40.0, oy + 10.0))
        .line_to((ox + 70.0, oy + 70.0))
        .line_to((ox + 10.0, oy + 70.0))
        .close()
        .build();
    renderer
        .add_shape(
            shape,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(50, 50, 220)),
        )
        .unwrap();

    vec![
        PixelExpectation::opaque(ox as u32 + 40, oy as u32 + 50, 50, 50, 220, "t03_interior"),
        PixelExpectation::opaque(
            ox as u32 + 15,
            oy as u32 + 15,
            255,
            255,
            255,
            "t03_outside_is_bg",
        ),
    ]
}

fn tile_04_path_bezier(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(4);
    let shape = Shape::builder()
        .begin((ox + 10.0, oy + 40.0))
        .cubic_bezier_to(
            (ox + 25.0, oy + 5.0),
            (ox + 55.0, oy + 5.0),
            (ox + 70.0, oy + 40.0),
        )
        .cubic_bezier_to(
            (ox + 55.0, oy + 75.0),
            (ox + 25.0, oy + 75.0),
            (ox + 10.0, oy + 40.0),
        )
        .close()
        .build();
    renderer
        .add_shape(
            shape,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(220, 200, 50)),
        )
        .unwrap();

    vec![
        PixelExpectation::opaque(ox as u32 + 40, oy as u32 + 40, 220, 200, 50, "t04_interior"),
        PixelExpectation::opaque(
            ox as u32 + 5,
            oy as u32 + 5,
            255,
            255,
            255,
            "t04_outside_is_bg",
        ),
    ]
}

// Hierarchy and scissor clipping

fn tile_05_rect_parent_child_inside(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(5);
    let parent = Shape::rect(
        [(ox + 5.0, oy + 5.0), (ox + 75.0, oy + 75.0)],
        Stroke::default(),
    );
    let parent_id = renderer
        .add_shape(
            parent,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(180, 180, 220)),
        )
        .unwrap();

    let child = Shape::rect(
        [(ox + 20.0, oy + 20.0), (ox + 60.0, oy + 60.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            child,
            Some(parent_id),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(220, 100, 50)),
        )
        .unwrap();

    vec![
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 40,
            220,
            100,
            50,
            "t05_child_center",
        ),
        PixelExpectation::opaque(
            ox as u32 + 10,
            oy as u32 + 10,
            180,
            180,
            220,
            "t05_parent_visible",
        ),
    ]
}

fn tile_06_rect_parent_child_overflow(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(6);
    let parent = Shape::rect(
        [(ox + 10.0, oy + 10.0), (ox + 55.0, oy + 70.0)],
        Stroke::default(),
    );
    let parent_id = renderer
        .add_shape(
            parent,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(150, 200, 150)),
        )
        .unwrap();

    // Child extends past parent's right edge
    let child = Shape::rect(
        [(ox + 30.0, oy + 20.0), (ox + 75.0, oy + 60.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            child,
            Some(parent_id),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(50, 50, 200)),
        )
        .unwrap();

    vec![
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 40,
            50,
            50,
            200,
            "t06_child_visible",
        ),
        // Parent-only area (left of child, inside parent)
        PixelExpectation::opaque(
            ox as u32 + 15,
            oy as u32 + 40,
            150,
            200,
            150,
            "t06_parent_visible",
        ),
        // The parent clips the child here, exposing the canvas.
        PixelExpectation::opaque(
            ox as u32 + 60,
            oy as u32 + 40,
            255,
            255,
            255,
            "t06_clipped_shows_bg",
        ),
    ]
}

fn tile_07_rect_parent_multi_children(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(7);
    let parent = Shape::rect(
        [(ox + 5.0, oy + 5.0), (ox + 75.0, oy + 75.0)],
        Stroke::default(),
    );
    let parent_id = renderer
        .add_shape(
            parent,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(200, 200, 200)),
        )
        .unwrap();

    let child_colors = [
        Color::rgb(220, 50, 50),
        Color::rgb(50, 180, 50),
        Color::rgb(50, 50, 220),
    ];
    let child_y_offsets: [(f32, f32); 3] = [(10.0, 25.0), (30.0, 45.0), (50.0, 65.0)];

    for (idx, &(y_start, y_end)) in child_y_offsets.iter().enumerate() {
        let child = Shape::rect(
            [(ox + 15.0, oy + y_start), (ox + 65.0, oy + y_end)],
            Stroke::default(),
        );
        renderer
            .add_shape(
                child,
                Some(parent_id),
                None,
                ShapeDrawCommandOptions::new().color(child_colors[idx]),
            )
            .unwrap();
    }

    vec![
        PixelExpectation::opaque(ox as u32 + 40, oy as u32 + 17, 220, 50, 50, "t07_child_red"),
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 37,
            50,
            180,
            50,
            "t07_child_green",
        ),
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 57,
            50,
            50,
            220,
            "t07_child_blue",
        ),
    ]
}

fn tile_08_rect_nested_3_levels(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(8);
    let level0 = Shape::rect(
        [(ox + 5.0, oy + 5.0), (ox + 75.0, oy + 75.0)],
        Stroke::default(),
    );
    let id0 = renderer
        .add_shape(
            level0,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(200, 180, 180)),
        )
        .unwrap();

    let level1 = Shape::rect(
        [(ox + 15.0, oy + 15.0), (ox + 65.0, oy + 65.0)],
        Stroke::default(),
    );
    let id1 = renderer
        .add_shape(
            level1,
            Some(id0),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(180, 200, 180)),
        )
        .unwrap();

    let level2 = Shape::rect(
        [(ox + 25.0, oy + 25.0), (ox + 55.0, oy + 55.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            level2,
            Some(id1),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(100, 100, 220)),
        )
        .unwrap();

    vec![
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 40,
            100,
            100,
            220,
            "t08_innermost",
        ),
        PixelExpectation::opaque(ox as u32 + 20, oy as u32 + 20, 180, 200, 180, "t08_middle"),
        PixelExpectation::opaque(ox as u32 + 10, oy as u32 + 10, 200, 180, 180, "t08_outer"),
    ]
}

fn tile_09_rect_siblings_overlap(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(9);
    let parent = Shape::rect(
        [(ox + 5.0, oy + 5.0), (ox + 75.0, oy + 75.0)],
        Stroke::default(),
    );
    let parent_id = renderer
        .add_shape(
            parent,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(200, 200, 200)),
        )
        .unwrap();

    // First child (drawn first)
    let child1 = Shape::rect(
        [(ox + 15.0, oy + 20.0), (ox + 50.0, oy + 55.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            child1,
            Some(parent_id),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(220, 50, 50)),
        )
        .unwrap();

    // Second child overlaps first (drawn on top)
    let child2 = Shape::rect(
        [(ox + 30.0, oy + 30.0), (ox + 65.0, oy + 65.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            child2,
            Some(parent_id),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(50, 50, 220)),
        )
        .unwrap();

    vec![
        // The second child covers the first in the overlap.
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 40,
            50,
            50,
            220,
            "t09_overlap_top",
        ),
        // Area only in first child (may have anti-aliasing near overlap edge)
        PixelExpectation::new(
            ox as u32 + 20,
            oy as u32 + 25,
            220,
            50,
            50,
            255,
            "t09_first_only",
        )
        .with_tolerance(35),
    ]
}

// Stencil clipping

fn tile_10_rounded_rect_clip(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(10);
    let parent = Shape::rounded_rect(
        [(ox + 5.0, oy + 5.0), (ox + 75.0, oy + 75.0)],
        BorderRadii::new(20.0),
        Stroke::default(),
    );
    let parent_id = renderer
        .add_shape(
            parent,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(200, 200, 230)),
        )
        .unwrap();

    // Child is smaller than parent, leaving a visible parent ring
    let child = Shape::rect(
        [(ox + 15.0, oy + 15.0), (ox + 65.0, oy + 65.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            child,
            Some(parent_id),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(220, 100, 50)),
        )
        .unwrap();

    vec![
        // Child interior
        PixelExpectation::opaque(ox as u32 + 40, oy as u32 + 40, 220, 100, 50, "t10_child"),
        // Parent ring visible between child edge and rounded border
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 8,
            200,
            200,
            230,
            "t10_parent_ring",
        ),
        // The canvas shows outside the rounded clip.
        PixelExpectation::opaque(
            ox as u32 + 7,
            oy as u32 + 7,
            255,
            255,
            255,
            "t10_corner_shows_bg",
        ),
    ]
}

fn tile_11_path_parent_clip(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(11);
    // Triangle parent
    let parent = Shape::builder()
        .begin((ox + 40.0, oy + 5.0))
        .line_to((ox + 75.0, oy + 70.0))
        .line_to((ox + 5.0, oy + 70.0))
        .close()
        .build();
    let parent_id = renderer
        .add_shape(
            parent,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(180, 180, 220)),
        )
        .unwrap();

    // Child rect offset to the right so left part of triangle shows parent color
    let child = Shape::rect(
        [(ox + 35.0, oy + 15.0), (ox + 75.0, oy + 75.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            child,
            Some(parent_id),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(220, 150, 50)),
        )
        .unwrap();

    vec![
        // The orange child remains visible inside the triangle.
        PixelExpectation::opaque(
            ox as u32 + 55,
            oy as u32 + 55,
            220,
            150,
            50,
            "t11_child_in_tri",
        ),
        // The child leaves this part of the blue parent uncovered.
        PixelExpectation::opaque(
            ox as u32 + 25,
            oy as u32 + 55,
            180,
            180,
            220,
            "t11_parent_in_tri",
        ),
        // The canvas shows outside the triangle.
        PixelExpectation::opaque(
            ox as u32 + 10,
            oy as u32 + 10,
            255,
            255,
            255,
            "t11_outside_is_bg",
        ),
    ]
}

fn tile_12_stencil_nested_3_levels(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(12);

    // L0 uses stencil clipping in the upper-left quadrant.
    let level0 = Shape::rounded_rect(
        [(ox + 5.0, oy + 5.0), (ox + 60.0, oy + 60.0)],
        BorderRadii::new(10.0),
        Stroke::default(),
    );
    let id0 = renderer
        .add_shape(
            level0,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(200, 180, 200)),
        )
        .unwrap(); // lavender

    // L1 extends past the right and bottom of L0.
    // Visible (L0∩L1) ≈ (20,20)→(60,60).
    let level1 = Shape::rounded_rect(
        [(ox + 20.0, oy + 20.0), (ox + 75.0, oy + 75.0)],
        BorderRadii::new(8.0),
        Stroke::default(),
    );
    let id1 = renderer
        .add_shape(
            level1,
            Some(id0),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(100, 200, 100)),
        )
        .unwrap(); // green

    // L2 extends left of L1 and below L0.
    // Visible (L0∩L1∩L2) ≈ (20,35)→(50,60).
    let level2 = Shape::rounded_rect(
        [(ox + 10.0, oy + 35.0), (ox + 50.0, oy + 75.0)],
        BorderRadii::new(8.0),
        Stroke::default(),
    );
    renderer
        .add_shape(
            level2,
            Some(id1),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(100, 100, 220)),
        )
        .unwrap(); // blue

    vec![
        // Inside all three → L2 blue
        PixelExpectation::opaque(
            ox as u32 + 35,
            oy as u32 + 47,
            100,
            100,
            220,
            "t12_l2_visible",
        ),
        // Inside L0 only (NW, outside L1 and L2) → L0 lavender
        PixelExpectation::opaque(ox as u32 + 15, oy as u32 + 15, 200, 180, 200, "t12_l0_only"),
        // Inside L0∩L1, outside L2 (y < 35) → L1 green
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 28,
            100,
            200,
            100,
            "t12_l1_visible",
        ),
        // L1 extends past L0 at x=60, so the canvas shows at (65,40).
        PixelExpectation::opaque(
            ox as u32 + 65,
            oy as u32 + 40,
            255,
            255,
            255,
            "t12_l0_clips_l1",
        ),
        // L1 clips L2 left of x=20, exposing L0 at (15,50).
        PixelExpectation::opaque(
            ox as u32 + 15,
            oy as u32 + 50,
            200,
            180,
            200,
            "t12_l1_clips_l2",
        ),
        // L0 clips both descendants below y=60, exposing the canvas at (30,70).
        PixelExpectation::opaque(
            ox as u32 + 30,
            oy as u32 + 70,
            255,
            255,
            255,
            "t12_l0_clips_chain",
        ),
    ]
}

fn tile_13_rotated_rect_clip(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(13);
    // Parent rect defined centered at origin, then rotated+translated
    let parent = Shape::rect([(-20.0, -20.0), (20.0, 20.0)], Stroke::default());
    let rotation = TransformInstance::rotation_z_deg(45.0);
    let translation = TransformInstance::translation(ox + 40.0, oy + 40.0);
    let parent_id = renderer
        .add_shape(
            parent,
            None,
            None,
            ShapeDrawCommandOptions::new()
                .color(Color::rgb(200, 200, 180))
                .transform(rotation.multiply(&translation)),
        )
        .unwrap();

    // The smaller child leaves a visible ring of the parent.
    let child = Shape::rect([(-12.0, -12.0), (12.0, 12.0)], Stroke::default());
    renderer
        .add_shape(
            child,
            Some(parent_id),
            None,
            ShapeDrawCommandOptions::new()
                .color(Color::rgb(220, 80, 80))
                .transform(rotation.multiply(&translation)),
        )
        .unwrap();

    vec![
        // The child covers the center of the rotated diamond.
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 40,
            220,
            80,
            80,
            "t13_child_center",
        ),
        // The parent remains visible between its edge and the child edge.
        // Parent diamond tip is at ~(40, 40±28), child diamond tip at ~(40, 40±17)
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 17,
            200,
            200,
            180,
            "t13_parent_ring",
        ),
        // The canvas shows outside the diamond.
        PixelExpectation::opaque(
            ox as u32 + 10,
            oy as u32 + 10,
            255,
            255,
            255,
            "t13_outside_is_bg",
        ),
    ]
}

// Mixed scissor and stencil clipping

fn tile_14_scissor_then_stencil(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(14);

    // L0: rect in NW quadrant → scissor clip.
    let level0 = Shape::rect(
        [(ox + 5.0, oy + 5.0), (ox + 60.0, oy + 60.0)],
        Stroke::default(),
    );
    let id0 = renderer
        .add_shape(
            level0,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(200, 200, 200)),
        )
        .unwrap(); // gray

    // L1 uses stencil clipping and extends past the right and bottom of L0.
    // Visible (L0∩L1) ≈ (20,20)→(60,60).
    let level1 = Shape::rounded_rect(
        [(ox + 20.0, oy + 20.0), (ox + 75.0, oy + 75.0)],
        BorderRadii::new(10.0),
        Stroke::default(),
    );
    let id1 = renderer
        .add_shape(
            level1,
            Some(id0),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(180, 220, 180)),
        )
        .unwrap(); // green

    // The leaf extends left of L1 and below L0.
    // Visible (L0∩L1∩leaf) ≈ (20,35)→(50,60).
    let leaf = Shape::rect(
        [(ox + 10.0, oy + 35.0), (ox + 50.0, oy + 75.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            leaf,
            Some(id1),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(50, 50, 200)),
        )
        .unwrap(); // blue

    vec![
        // Inside all three → leaf blue
        PixelExpectation::opaque(
            ox as u32 + 35,
            oy as u32 + 47,
            50,
            50,
            200,
            "t14_leaf_visible",
        ),
        // Inside L0∩L1, outside leaf (y < 35) → L1 green
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 28,
            180,
            220,
            180,
            "t14_l1_visible",
        ),
        // Inside L0 only (NW, outside L1 and leaf) → L0 gray
        PixelExpectation::opaque(ox as u32 + 15, oy as u32 + 15, 200, 200, 200, "t14_l0_only"),
        // L1 extends past L0 at x=60, so the canvas shows at (65,40).
        PixelExpectation::opaque(
            ox as u32 + 65,
            oy as u32 + 40,
            255,
            255,
            255,
            "t14_l0_clips_l1",
        ),
        // L1 clips the leaf left of x=20, exposing L0 at (15,50).
        PixelExpectation::opaque(
            ox as u32 + 15,
            oy as u32 + 50,
            200,
            200,
            200,
            "t14_l1_clips_leaf",
        ),
        // L0 clips the leaf below y=60, exposing the canvas at (30,65).
        PixelExpectation::opaque(
            ox as u32 + 30,
            oy as u32 + 65,
            255,
            255,
            255,
            "t14_l0_clips_chain",
        ),
    ]
}

fn tile_15_stencil_then_scissor(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(15);

    // L0: rounded-rect in NW quadrant → stencil clip.
    let level0 = Shape::rounded_rect(
        [(ox + 5.0, oy + 5.0), (ox + 60.0, oy + 60.0)],
        BorderRadii::new(12.0),
        Stroke::default(),
    );
    let id0 = renderer
        .add_shape(
            level0,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(220, 200, 200)),
        )
        .unwrap(); // pink

    // L1 uses scissor clipping and extends past the right and bottom of L0.
    // Visible (L0∩L1) ≈ (20,20)→(60,60).
    let level1 = Shape::rect(
        [(ox + 20.0, oy + 20.0), (ox + 75.0, oy + 75.0)],
        Stroke::default(),
    );
    let id1 = renderer
        .add_shape(
            level1,
            Some(id0),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(200, 220, 200)),
        )
        .unwrap(); // green

    // The leaf extends left of L1 and below L0.
    // Visible (L0∩L1∩leaf) ≈ (20,35)→(50,60).
    let leaf = Shape::rect(
        [(ox + 10.0, oy + 35.0), (ox + 50.0, oy + 75.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            leaf,
            Some(id1),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(50, 200, 50)),
        )
        .unwrap(); // bright green

    vec![
        // Inside all three → leaf green
        PixelExpectation::opaque(
            ox as u32 + 35,
            oy as u32 + 47,
            50,
            200,
            50,
            "t15_leaf_visible",
        ),
        // Inside L0∩L1, outside leaf (y < 35) → L1 green
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 28,
            200,
            220,
            200,
            "t15_l1_visible",
        ),
        // Inside L0 only (NW, outside L1 and leaf) → L0 pink
        PixelExpectation::opaque(ox as u32 + 15, oy as u32 + 15, 220, 200, 200, "t15_l0_only"),
        // L1 extends past L0 at x=60, so the canvas shows at (65,40).
        PixelExpectation::opaque(
            ox as u32 + 65,
            oy as u32 + 40,
            255,
            255,
            255,
            "t15_l0_clips_l1",
        ),
        // L1 clips the leaf left of x=20, exposing L0 at (15,50).
        PixelExpectation::opaque(
            ox as u32 + 15,
            oy as u32 + 50,
            220,
            200,
            200,
            "t15_l1_clips_leaf",
        ),
        // L0 stencil corner: inside L0 bbox but outside rounded corner → bg.
        // TL corner center (17,17) r=12, dist((7,7),(17,17)) ≈ 14.1 > 12.
        PixelExpectation::opaque(
            ox as u32 + 7,
            oy as u32 + 7,
            255,
            255,
            255,
            "t15_l0_corner_clips",
        ),
    ]
}

fn tile_16_deep_mixed_5_levels(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(16);

    // L0: rect, tall left portion → scissor.
    let l0 = Shape::rect(
        [(ox + 5.0, oy + 5.0), (ox + 55.0, oy + 75.0)],
        Stroke::default(),
    );
    let id0 = renderer
        .add_shape(
            l0,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(220, 220, 220)),
        )
        .unwrap(); // light gray

    // L1 uses scissor clipping and extends right of L0.
    // Visible (L0∩L1) = (20,5)→(55,55).
    let l1 = Shape::rect(
        [(ox + 20.0, oy + 5.0), (ox + 75.0, oy + 55.0)],
        Stroke::default(),
    );
    let id1 = renderer
        .add_shape(
            l1,
            Some(id0),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(200, 200, 220)),
        )
        .unwrap(); // blue-gray

    // L2 uses stencil clipping and extends below L1.
    // Visible (L0∩L1∩L2) ≈ (20,20)→(50,55).
    let l2 = Shape::rounded_rect(
        [(ox + 15.0, oy + 20.0), (ox + 50.0, oy + 70.0)],
        BorderRadii::new(10.0),
        Stroke::default(),
    );
    let id2 = renderer
        .add_shape(
            l2,
            Some(id1),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(180, 220, 180)),
        )
        .unwrap(); // green

    // L3 uses scissor clipping and extends right of and above L2.
    // Visible (L0∩L1∩L2∩L3) ≈ (25,20)→(50,50).
    let l3 = Shape::rect(
        [(ox + 25.0, oy + 10.0), (ox + 70.0, oy + 50.0)],
        Stroke::default(),
    );
    let id3 = renderer
        .add_shape(
            l3,
            Some(id2),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(200, 180, 220)),
        )
        .unwrap(); // purple

    // L4 extends left of and below L3.
    // Visible (all 5) ≈ (25,30)→(45,50).
    let l4 = Shape::rect(
        [(ox + 10.0, oy + 30.0), (ox + 45.0, oy + 65.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            l4,
            Some(id3),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(50, 50, 200)),
        )
        .unwrap(); // blue

    vec![
        // Center of 5-way intersection → L4 blue
        PixelExpectation::opaque(ox as u32 + 35, oy as u32 + 40, 50, 50, 200, "t16_leaf"),
        // Inside L0∩L1∩L2∩L3, outside L4 (x=47 > L4.right=45) → L3 purple
        PixelExpectation::opaque(
            ox as u32 + 47,
            oy as u32 + 35,
            200,
            180,
            220,
            "t16_l3_visible",
        ),
        // L3 clips L4 left of x=25, exposing L2 at (22,40).
        PixelExpectation::opaque(
            ox as u32 + 22,
            oy as u32 + 40,
            180,
            220,
            180,
            "t16_l3_clips_l4",
        ),
        // L2 clips L3 above y=20, exposing L1 at (35,12).
        PixelExpectation::opaque(
            ox as u32 + 35,
            oy as u32 + 12,
            200,
            200,
            220,
            "t16_l2_clips_l3",
        ),
        // L1 clips L2 left of x=20, exposing L0 at (17,40).
        PixelExpectation::opaque(
            ox as u32 + 17,
            oy as u32 + 40,
            220,
            220,
            220,
            "t16_l1_clips_l2",
        ),
        // L0 clips L1 right of x=55, exposing the canvas at (60,30).
        PixelExpectation::opaque(
            ox as u32 + 60,
            oy as u32 + 30,
            255,
            255,
            255,
            "t16_l0_clips_l1",
        ),
    ]
}

// Transforms

fn tile_17_translated_rect(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(17);
    // Rect defined at origin, translated to bottom-right of tile
    let shape = Shape::rect([(0.0, 0.0), (40.0, 25.0)], Stroke::default());
    renderer
        .add_shape(
            shape,
            None,
            None,
            ShapeDrawCommandOptions::new()
                .color(Color::rgb(220, 150, 50))
                .transform(TransformInstance::translation(ox + 30.0, oy + 45.0)),
        )
        .unwrap();

    vec![
        // Inside the translated rect (bottom-right area)
        PixelExpectation::opaque(
            ox as u32 + 50,
            oy as u32 + 57,
            220,
            150,
            50,
            "t17_translated",
        ),
        // Original position at origin is empty (translation moved it)
        PixelExpectation::opaque(
            ox as u32 + 10,
            oy as u32 + 10,
            255,
            255,
            255,
            "t17_origin_is_bg",
        ),
        // Top-left of tile is bg (rect is bottom-right)
        PixelExpectation::opaque(
            ox as u32 + 15,
            oy as u32 + 40,
            255,
            255,
            255,
            "t17_above_rect",
        ),
    ]
}

fn tile_18_scaled_rect(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(18);
    // 60×60 rect, scaled to 0.5× horizontally and 1.0× vertically around tile center.
    // Result: 30×60 rect, horizontally centered in tile.
    let shape = Shape::rect(
        [(ox + 10.0, oy + 10.0), (ox + 70.0, oy + 70.0)],
        Stroke::default(),
    );
    let to_origin = TransformInstance::translation(-(ox + 40.0), -(oy + 40.0));
    let scale = TransformInstance::scale(0.5, 1.0);
    let back = TransformInstance::translation(ox + 40.0, oy + 40.0);
    renderer
        .add_shape(
            shape,
            None,
            None,
            ShapeDrawCommandOptions::new()
                .color(Color::rgb(150, 50, 200))
                .transform(to_origin.multiply(&scale.multiply(&back))),
        )
        .unwrap();

    vec![
        // Center should still have color
        PixelExpectation::opaque(ox as u32 + 40, oy as u32 + 40, 150, 50, 200, "t18_center"),
        // Left edge of original rect (x=15) now scaled inward → bg
        PixelExpectation::opaque(
            ox as u32 + 15,
            oy as u32 + 40,
            255,
            255,
            255,
            "t18_left_scaled_away",
        ),
        // Right edge of original rect (x=65) now scaled inward → bg
        PixelExpectation::opaque(
            ox as u32 + 65,
            oy as u32 + 40,
            255,
            255,
            255,
            "t18_right_scaled_away",
        ),
        // Vertically still full → colored at y=15
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 15,
            150,
            50,
            200,
            "t18_vert_intact",
        ),
    ]
}

fn tile_19_rotated_rect_leaf(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(19);
    let shape = Shape::rect([(-15.0, -15.0), (15.0, 15.0)], Stroke::default());
    let rotation = TransformInstance::rotation_z_deg(45.0);
    let translation = TransformInstance::translation(ox + 40.0, oy + 40.0);
    renderer
        .add_shape(
            shape,
            None,
            None,
            ShapeDrawCommandOptions::new()
                .color(Color::rgb(220, 100, 100))
                .transform(rotation.multiply(&translation)),
        )
        .unwrap();

    vec![
        PixelExpectation::opaque(ox as u32 + 40, oy as u32 + 40, 220, 100, 100, "t19_center"),
        PixelExpectation::opaque(
            ox as u32 + 22,
            oy as u32 + 22,
            255,
            255,
            255,
            "t19_corner_is_bg",
        ),
    ]
}

fn tile_20_transform_parent_child(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(20);
    // Parent translated
    let parent = Shape::rect([(0.0, 0.0), (50.0, 50.0)], Stroke::default());
    let parent_id = renderer
        .add_shape(
            parent,
            None,
            None,
            ShapeDrawCommandOptions::new()
                .color(Color::rgb(180, 180, 220))
                .transform(TransformInstance::translation(ox + 15.0, oy + 15.0)),
        )
        .unwrap();

    // The transform maps the child bounds from local (10,10)-(40,40) to tile (25,25)-(55,55).
    let child = Shape::rect([(10.0, 10.0), (40.0, 40.0)], Stroke::default());
    renderer
        .add_shape(
            child,
            Some(parent_id),
            None,
            ShapeDrawCommandOptions::new()
                .color(Color::rgb(50, 200, 50))
                .transform(TransformInstance::translation(ox + 15.0, oy + 15.0)),
        )
        .unwrap();

    vec![
        PixelExpectation::opaque(ox as u32 + 40, oy as u32 + 40, 50, 200, 50, "t20_child"),
        PixelExpectation::opaque(ox as u32 + 20, oy as u32 + 20, 180, 180, 220, "t20_parent"),
    ]
}

// Colors and alpha

fn tile_21_alpha_overlap(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(21);
    // Opaque blue background
    let bg = Shape::rect(
        [(ox + 10.0, oy + 10.0), (ox + 50.0, oy + 60.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            bg,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(50, 50, 220)),
        )
        .unwrap();

    // Semi-transparent red on top
    let fg = Shape::rect(
        [(ox + 25.0, oy + 20.0), (ox + 65.0, oy + 55.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            fg,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgba(220, 50, 50, 128)),
        )
        .unwrap();

    vec![
        // Blue region only
        PixelExpectation::opaque(ox as u32 + 15, oy as u32 + 15, 50, 50, 220, "t21_blue_only"),
        // Blend in linear RGB with alpha 128/255, then encode the result as sRGB.
        PixelExpectation::new(
            ox as u32 + 35,
            oy as u32 + 35,
            165,
            50,
            164,
            255,
            "t21_red_over_blue",
        ),
        // The same red over the white canvas.
        PixelExpectation::new(
            ox as u32 + 55,
            oy as u32 + 40,
            238,
            190,
            190,
            255,
            "t21_red_over_white",
        ),
    ]
}

fn tile_22_no_color_default(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(22);
    // Colored background so we can verify that an unset color stays transparent.
    let bg = Shape::rect(
        [(ox + 5.0, oy + 5.0), (ox + 75.0, oy + 75.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            bg,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(100, 100, 200)),
        )
        .unwrap();

    let shape = Shape::rect(
        [(ox + 20.0, oy + 20.0), (ox + 60.0, oy + 60.0)],
        Stroke::default(),
    );
    // An unset fill is transparent.
    renderer
        .add_shape(shape, None, None, ShapeDrawCommandOptions::new())
        .unwrap();

    vec![
        // The background shows through the transparent shape.
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 40,
            100,
            100,
            200,
            "t22_default_transparent",
        ),
        // Background visible around the shape
        PixelExpectation::opaque(
            ox as u32 + 10,
            oy as u32 + 10,
            100,
            100,
            200,
            "t22_bg_visible",
        ),
    ]
}

fn tile_23_fully_transparent(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(23);
    let shape = Shape::rect(
        [(ox + 15.0, oy + 15.0), (ox + 65.0, oy + 65.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            shape,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::TRANSPARENT),
        )
        .unwrap();

    // Transparent shape over white canvas root → shows white
    vec![PixelExpectation::opaque(
        ox as u32 + 40,
        oy as u32 + 40,
        255,
        255,
        255,
        "t23_transparent_shows_bg",
    )]
}

// Textures

fn tile_24_textured_rect(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(24);
    let shape = Shape::rect(
        [(ox + 10.0, oy + 10.0), (ox + 70.0, oy + 70.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            shape,
            None,
            None,
            ShapeDrawCommandOptions::new()
                .background_texture_id(CHECKERBOARD_TEXTURE_ID)
                .color(Color::WHITE),
        )
        .unwrap();

    vec![
        // Adjacent texel centers avoid interpolation between black and white.
        PixelExpectation::opaque(
            ox as u32 + 17,
            oy as u32 + 17,
            255,
            255,
            255,
            "t24_white_texel",
        ),
        PixelExpectation::opaque(ox as u32 + 32, oy as u32 + 17, 0, 0, 0, "t24_black_texel"),
        // The canvas shows outside the textured rectangle.
        PixelExpectation::opaque(
            ox as u32 + 5,
            oy as u32 + 5,
            255,
            255,
            255,
            "t24_outside_is_bg",
        ),
    ]
}

fn tile_25_textured_with_color(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(25);
    let shape = Shape::rect(
        [(ox + 10.0, oy + 10.0), (ox + 70.0, oy + 70.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            shape,
            None,
            None,
            ShapeDrawCommandOptions::new()
                .background_texture_id(TRANSLUCENT_CHECKERBOARD_TEXTURE_ID)
                .color(Color::rgb(255, 100, 100)),
        )
        .unwrap();

    vec![
        // Sample texel centers. In linear RGB, texture * alpha + fill * (1 - alpha)
        // gives these sRGB colors for white and black texels at alpha 128/255.
        PixelExpectation::opaque(
            ox as u32 + 17,
            oy as u32 + 17,
            255,
            198,
            198,
            "t25_translucent_white_over_fill",
        ),
        PixelExpectation::opaque(
            ox as u32 + 32,
            oy as u32 + 17,
            187,
            71,
            71,
            "t25_translucent_black_over_fill",
        ),
        // Outside the textured rect
        PixelExpectation::opaque(
            ox as u32 + 5,
            oy as u32 + 5,
            255,
            255,
            255,
            "t25_outside_is_bg",
        ),
    ]
}

fn tile_26_textured_parent_child(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(26);
    let parent = Shape::rect(
        [(ox + 5.0, oy + 5.0), (ox + 75.0, oy + 75.0)],
        Stroke::default(),
    );
    let parent_id = renderer
        .add_shape(
            parent,
            None,
            None,
            ShapeDrawCommandOptions::new()
                .background_texture_id(CHECKERBOARD_TEXTURE_ID)
                .color(Color::WHITE),
        )
        .unwrap();

    let child = Shape::rect(
        [(ox + 25.0, oy + 25.0), (ox + 55.0, oy + 55.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            child,
            Some(parent_id),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(220, 50, 50)),
        )
        .unwrap();

    vec![
        // Child interior on top of textured parent
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 40,
            220,
            50,
            50,
            "t26_child_over_texture",
        ),
        // The corner texels stay unmixed under linear sampling and outside the child.
        PixelExpectation::opaque(
            ox as u32 + 10,
            oy as u32 + 10,
            255,
            255,
            255,
            "t26_parent_white_texel",
        ),
        PixelExpectation::opaque(
            ox as u32 + 70,
            oy as u32 + 10,
            0,
            0,
            0,
            "t26_parent_black_texel",
        ),
    ]
}

// Group effects

fn tile_27_group_blur_leaf(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(27);

    // Only the shape is blurred; the background stripe must stay sharp.
    let bg_stripe = Shape::rect(
        [(ox + 5.0, oy + 30.0), (ox + 75.0, oy + 50.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            bg_stripe,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(50, 180, 50)),
        )
        .unwrap(); // green stripe

    // Blurred shape (slightly transparent so stripe shows through)
    let shape = Shape::rect(
        [(ox + 15.0, oy + 10.0), (ox + 65.0, oy + 70.0)],
        Stroke::default(),
    );
    let id = renderer
        .add_shape(
            shape,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgba(220, 50, 50, 200)),
        )
        .unwrap(); // semi-transparent red

    let blur_params = BlurParams {
        radius: 8.0,
        _pad: 0.0,
    };
    renderer
        .set_group_effect(id, BLUR_EFFECT_ID, bytemuck::bytes_of(&blur_params))
        .expect("Failed to set group effect");

    vec![
        // Blurred shape center: red-ish (blurred, high tolerance)
        PixelExpectation::new(
            ox as u32 + 40,
            oy as u32 + 40,
            180,
            40,
            40,
            200,
            "t27_blurred_center",
        )
        .with_tolerance(60),
        // The uncovered stripe stays sharp and green.
        PixelExpectation::opaque(
            ox as u32 + 10,
            oy as u32 + 40,
            50,
            180,
            50,
            "t27_bg_stripe_sharp",
        )
        .with_tolerance(20),
    ]
}

fn tile_28_group_blur_with_children(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(28);

    // Only the group is blurred; the background stripe must stay sharp.
    let bg_stripe = Shape::rect(
        [(ox + 5.0, oy + 30.0), (ox + 75.0, oy + 50.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            bg_stripe,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(220, 180, 50)),
        )
        .unwrap(); // yellow stripe

    // Blurred parent (group effect applies to parent+child as a unit)
    let parent = Shape::rect(
        [(ox + 10.0, oy + 5.0), (ox + 70.0, oy + 75.0)],
        Stroke::default(),
    );
    let parent_id = renderer
        .add_shape(
            parent,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgba(200, 200, 200, 200)),
        )
        .unwrap(); // semi-transparent gray

    let child = Shape::rect(
        [(ox + 20.0, oy + 20.0), (ox + 60.0, oy + 60.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            child,
            Some(parent_id),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(50, 50, 220)),
        )
        .unwrap();

    let blur_params = BlurParams {
        radius: 6.0,
        _pad: 0.0,
    };
    renderer
        .set_group_effect(parent_id, BLUR_EFFECT_ID, bytemuck::bytes_of(&blur_params))
        .expect("Failed to set group effect");

    vec![
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 40,
            50,
            50,
            220,
            "t28_group_blur_center",
        ),
        // Each blur pass must mix the blue child with the parent at its edge.
        PixelExpectation::opaque(
            ox as u32 + 20,
            oy as u32 + 40,
            140,
            134,
            205,
            "t28_group_blur_child_left_edge",
        ),
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 20,
            146,
            146,
            217,
            "t28_group_blur_child_top_edge",
        ),
        // The parent's blur reaches the stripe outside the original bounds.
        PixelExpectation::opaque(
            ox as u32 + 8,
            oy as u32 + 40,
            217,
            184,
            100,
            "t28_group_blur_outer_edge",
        ),
    ]
}

// Backdrop effects

fn tile_29_backdrop_blur_leaf(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(29);
    // Red background
    let bg = Shape::rect(
        [(ox + 5.0, oy + 5.0), (ox + 75.0, oy + 75.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            bg,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(220, 50, 50)),
        )
        .unwrap();

    // Sharp-edged stripe that partially overlaps the backdrop panel.
    // Outside the panel it should be crisp; under the panel it should get blurred.
    let stripe = Shape::rect(
        [(ox + 10.0, oy + 32.0), (ox + 70.0, oy + 48.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            stripe,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(50, 50, 220)),
        )
        .unwrap(); // blue stripe

    // Backdrop blur panel on top (leaf, no children)
    let panel = Shape::rect(
        [(ox + 20.0, oy + 15.0), (ox + 60.0, oy + 65.0)],
        Stroke::default(),
    );
    let panel_id = renderer
        .add_shape(
            panel,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgba(255, 255, 255, 80)),
        )
        .unwrap();

    let blur_params = BlurParams {
        radius: 10.0,
        _pad: 0.0,
    };
    renderer
        .set_shape_backdrop_effect(
            panel_id,
            BLUR_EFFECT_ID,
            bytemuck::bytes_of(&blur_params),
            BackdropEffectConfig::default(),
        )
        .expect("Failed to set backdrop effect");

    vec![
        // Panel interior: blurred mix of red bg + blue stripe under the panel's white fill
        PixelExpectation::new(
            ox as u32 + 40,
            oy as u32 + 40,
            158,
            157,
            231,
            255,
            "t29_backdrop_interior",
        )
        .with_tolerance(20),
        // Blue stripe outside panel stays sharp and fully blue
        PixelExpectation::opaque(
            ox as u32 + 12,
            oy as u32 + 40,
            50,
            50,
            220,
            "t29_stripe_sharp_outside",
        ),
        // Red background outside panel and stripe
        PixelExpectation::opaque(ox as u32 + 10, oy as u32 + 10, 220, 50, 50, "t29_bg_intact"),
    ]
}

fn tile_30_backdrop_blur_nonleaf(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(30);
    // Green background
    let bg = Shape::rect(
        [(ox + 5.0, oy + 5.0), (ox + 75.0, oy + 75.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            bg,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(50, 180, 50)),
        )
        .unwrap();

    // Sharp-edged stripe partially behind the backdrop panel
    let stripe = Shape::rect(
        [(ox + 8.0, oy + 30.0), (ox + 72.0, oy + 50.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            stripe,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(220, 50, 50)),
        )
        .unwrap(); // red stripe

    // Backdrop panel with a child
    let panel = Shape::rect(
        [(ox + 15.0, oy + 10.0), (ox + 65.0, oy + 70.0)],
        Stroke::default(),
    );
    let panel_id = renderer
        .add_shape(
            panel,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgba(255, 255, 255, 80)),
        )
        .unwrap();

    let child = Shape::rect(
        [(ox + 25.0, oy + 50.0), (ox + 55.0, oy + 65.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            child,
            Some(panel_id),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(50, 50, 220)),
        )
        .unwrap();

    let blur_params = BlurParams {
        radius: 8.0,
        _pad: 0.0,
    };
    renderer
        .set_shape_backdrop_effect(
            panel_id,
            BLUR_EFFECT_ID,
            bytemuck::bytes_of(&blur_params),
            BackdropEffectConfig::default(),
        )
        .expect("Failed to set backdrop effect");

    vec![
        // At the stripe edge, 42.5% of the blur kernel samples the other side.
        // Composite the white panel in linear color, then encode sRGB.
        PixelExpectation::opaque_approx(
            ox as u32 + 40,
            oy as u32 + 29,
            193,
            188,
            157,
            5,
            "t30_blur_above_stripe_edge",
        ),
        PixelExpectation::opaque_approx(
            ox as u32 + 40,
            oy as u32 + 30,
            204,
            181,
            157,
            5,
            "t30_blur_inside_stripe_edge",
        ),
        // Child visible on top of blurred background
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 57,
            50,
            50,
            220,
            "t30_child_visible",
        ),
        // Red stripe outside panel stays sharp
        PixelExpectation::opaque(
            ox as u32 + 10,
            oy as u32 + 40,
            220,
            50,
            50,
            "t30_stripe_sharp_outside",
        ),
        // Green background intact outside panel and stripe
        PixelExpectation::opaque(ox as u32 + 10, oy as u32 + 10, 50, 180, 50, "t30_bg_intact"),
    ]
}

fn tile_31_backdrop_under_scissor(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(31);
    // Yellow background
    let bg = Shape::rect(
        [(ox + 5.0, oy + 5.0), (ox + 75.0, oy + 75.0)],
        Stroke::default(),
    );
    let bg_id = renderer
        .add_shape(
            bg,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(220, 180, 50)),
        )
        .unwrap();

    // The panel should blur only the covered part of the blue stripe.
    let stripe = Shape::rect(
        [(ox + 8.0, oy + 32.0), (ox + 72.0, oy + 48.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            stripe,
            Some(bg_id),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(50, 50, 200)),
        )
        .unwrap(); // blue stripe

    // Draw the scissor-clipping parent after its sibling stripe.
    let clip_parent = Shape::rect(
        [(ox + 10.0, oy + 10.0), (ox + 60.0, oy + 70.0)],
        Stroke::default(),
    );
    let clip_id = renderer
        .add_shape(
            clip_parent,
            Some(bg_id),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgba(0, 0, 0, 0)),
        )
        .unwrap(); // Transparent fill leaves only the clip.

    // Backdrop panel inside scissor-clipped parent
    let panel = Shape::rect(
        [(ox + 15.0, oy + 15.0), (ox + 55.0, oy + 65.0)],
        Stroke::default(),
    );
    let panel_id = renderer
        .add_shape(
            panel,
            Some(clip_id),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgba(255, 255, 255, 80)),
        )
        .unwrap();

    let blur_params = BlurParams {
        radius: 6.0,
        _pad: 0.0,
    };
    renderer
        .set_shape_backdrop_effect(
            panel_id,
            BLUR_EFFECT_ID,
            bytemuck::bytes_of(&blur_params),
            BackdropEffectConfig::default(),
        )
        .expect("Failed to set backdrop effect");

    vec![
        // Panel interior: blurred mix of yellow bg + blue stripe under the panel's white fill
        PixelExpectation::new(
            ox as u32 + 35,
            oy as u32 + 40,
            157,
            157,
            219,
            255,
            "t31_backdrop_in_scissor",
        )
        .with_tolerance(20),
        // Blue stripe outside panel stays sharp
        PixelExpectation::opaque(
            ox as u32 + 65,
            oy as u32 + 40,
            50,
            50,
            200,
            "t31_stripe_sharp_outside",
        ),
        // Yellow background outside panel and stripe
        PixelExpectation::opaque(
            ox as u32 + 65,
            oy as u32 + 15,
            220,
            180,
            50,
            "t31_bg_intact",
        ),
    ]
}

// Edge cases

fn tile_32_tiny_1px_shape(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(32);
    let shape = Shape::rect(
        [(ox + 40.0, oy + 40.0), (ox + 41.0, oy + 41.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            shape,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(255, 0, 255)),
        )
        .unwrap();

    // The antialiasing fringe can spread coverage beyond the one-pixel shape.
    vec![PixelExpectation::new(
        ox as u32 + 40,
        oy as u32 + 40,
        200,
        0,
        200,
        200,
        "t32_tiny_pixel",
    )
    .with_tolerance(80)]
}

fn tile_33_shape_at_canvas_edge(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    // Reserve the bottom-right grid slot for a shape crossing both canvas edges.
    let origin_x = (CANVAS_WIDTH - TILE_SIZE) as f32;
    let origin_y = (CANVAS_HEIGHT - TILE_SIZE) as f32;
    let shape = Shape::rect(
        [
            (origin_x + 50.0, origin_y + 50.0),
            (origin_x + 120.0, origin_y + 120.0),
        ],
        Stroke::default(),
    );
    renderer
        .add_shape(
            shape,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(180, 50, 180)),
        )
        .unwrap();

    vec![
        PixelExpectation::opaque(
            CANVAS_WIDTH - 1,
            CANVAS_HEIGHT - 15,
            180,
            50,
            180,
            "t33_right_edge",
        ),
        PixelExpectation::opaque(
            CANVAS_WIDTH - 15,
            CANVAS_HEIGHT - 1,
            180,
            50,
            180,
            "t33_bottom_edge",
        ),
        PixelExpectation::opaque(
            CANVAS_WIDTH - 1,
            CANVAS_HEIGHT - 1,
            180,
            50,
            180,
            "t33_bottom_right_corner",
        ),
        PixelExpectation::opaque(
            CANVAS_WIDTH - 40,
            CANVAS_HEIGHT - 15,
            255,
            255,
            255,
            "t33_left_of_shape",
        ),
        PixelExpectation::opaque(
            CANVAS_WIDTH - 15,
            CANVAS_HEIGHT - 40,
            255,
            255,
            255,
            "t33_above_shape",
        ),
    ]
}

fn tile_34_cached_shape(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(34);
    let shape = Shape::rect(
        [(ox + 10.0, oy + 10.0), (ox + 70.0, oy + 70.0)],
        Stroke::default(),
    );

    let cache_key = 9999;
    renderer.load_shape(shape, cache_key, None);
    renderer
        .add_cached_shape_to_the_render_queue(
            cache_key,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(50, 180, 220)),
        )
        .unwrap();

    vec![
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 40,
            50,
            180,
            220,
            "t34_cached_interior",
        ),
        // The canvas shows outside the cached shape.
        PixelExpectation::opaque(
            ox as u32 + 5,
            oy as u32 + 5,
            255,
            255,
            255,
            "t34_outside_is_bg",
        ),
    ]
}

fn tile_35_trivial_transform_transparent_leaf(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(35);

    let bg = Shape::rect(
        [(ox + 5.0, oy + 5.0), (ox + 75.0, oy + 75.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            bg,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(40, 90, 200)),
        )
        .unwrap();

    let leaf = Shape::rect([(0.0, 0.0), (20.0, 20.0)], Stroke::default());
    renderer
        .add_shape(
            leaf,
            None,
            None,
            ShapeDrawCommandOptions::new().transform(TransformInstance::affine_2d(
                2.0,
                0.0,
                0.0,
                2.0,
                ox + 20.0,
                oy + 20.0,
            )),
        )
        .unwrap();

    vec![PixelExpectation::opaque(
        ox as u32 + 40,
        oy as u32 + 40,
        40,
        90,
        200,
        "t35_transparent_leaf_shows_bg",
    )]
}

fn tile_36_trivial_transform_transparent_parent(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(36);

    let parent = Shape::rect([(0.0, 0.0), (20.0, 20.0)], Stroke::default());
    let parent_id = renderer
        .add_shape(
            parent,
            None,
            None,
            ShapeDrawCommandOptions::new().transform(TransformInstance::affine_2d(
                2.0,
                0.0,
                0.0,
                2.0,
                ox + 20.0,
                oy + 20.0,
            )),
        )
        .unwrap();

    let child = Shape::rect(
        [(ox + 10.0, oy + 10.0), (ox + 70.0, oy + 40.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            child,
            Some(parent_id),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(30, 110, 220)),
        )
        .unwrap();

    vec![
        PixelExpectation::opaque(
            ox as u32 + 30,
            oy as u32 + 30,
            30,
            110,
            220,
            "t36_child_inside_clip",
        ),
        PixelExpectation::opaque(
            ox as u32 + 65,
            oy as u32 + 30,
            255,
            255,
            255,
            "t36_child_outside_clip",
        ),
    ]
}

fn tile_37_textured_transparent_rects(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(37);

    let explicit_alpha_zero = Shape::rect(
        [(ox + 10.0, oy + 10.0), (ox + 30.0, oy + 30.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            explicit_alpha_zero,
            None,
            None,
            ShapeDrawCommandOptions::new()
                .color(Color::rgba(255, 0, 0, 0))
                .background_texture_id(SOLID_GREEN_TEXTURE_ID),
        )
        .unwrap();

    let none_color = Shape::rect(
        [(ox + 40.0, oy + 10.0), (ox + 60.0, oy + 30.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            none_color,
            None,
            None,
            ShapeDrawCommandOptions::new().background_texture_id(SOLID_GREEN_TEXTURE_ID),
        )
        .unwrap();

    vec![
        PixelExpectation::opaque(
            ox as u32 + 20,
            oy as u32 + 20,
            0,
            255,
            0,
            "t37_explicit_alpha_zero_texture",
        ),
        PixelExpectation::opaque(
            ox as u32 + 50,
            oy as u32 + 20,
            0,
            255,
            0,
            "t37_none_color_texture",
        ),
    ]
}

fn tile_38_sheared_transparent_parent(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(38);

    let parent = Shape::rect([(0.0, 0.0), (20.0, 20.0)], Stroke::default());
    let parent_id = renderer
        .add_shape(
            parent,
            None,
            None,
            ShapeDrawCommandOptions::new().transform(TransformInstance::affine_2d(
                1.0,
                0.0,
                0.5,
                1.0,
                ox + 20.0,
                oy + 20.0,
            )),
        )
        .unwrap();

    let child = Shape::rect(
        [(ox + 15.0, oy + 15.0), (ox + 55.0, oy + 45.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            child,
            Some(parent_id),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(20, 120, 230)),
        )
        .unwrap();

    vec![
        PixelExpectation::opaque(
            ox as u32 + 35,
            oy as u32 + 30,
            20,
            120,
            230,
            "t38_inside_sheared_clip",
        ),
        PixelExpectation::opaque(
            ox as u32 + 22,
            oy as u32 + 38,
            255,
            255,
            255,
            "t38_outside_sheared_clip",
        ),
    ]
}

// Gradient fills

fn two_stop_common(c1: (u8, u8, u8), c2: (u8, u8, u8), spread: SpreadMode) -> GradientCommonDesc {
    two_stop_common_with_units(c1, c2, spread, GradientUnits::Local)
}

fn two_stop_common_canvas(
    c1: (u8, u8, u8),
    c2: (u8, u8, u8),
    spread: SpreadMode,
) -> GradientCommonDesc {
    two_stop_common_with_units(c1, c2, spread, GradientUnits::Canvas)
}

fn two_stop_common_with_units(
    c1: (u8, u8, u8),
    c2: (u8, u8, u8),
    spread: SpreadMode,
    units: GradientUnits,
) -> GradientCommonDesc {
    GradientCommonDesc::new([
        GradientStop::at_position(
            GradientStopOffset::linear_radial(0.0),
            Color::rgb(c1.0, c1.1, c1.2),
        ),
        GradientStop::at_position(
            GradientStopOffset::linear_radial(1.0),
            Color::rgb(c2.0, c2.1, c2.2),
        ),
    ])
    .with_units(units)
    .with_spread(spread)
    .with_interpolation(ColorInterpolation::Srgb)
}

fn tile_39_linear_gradient(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(39);
    let shape = Shape::rect([(0.0, 0.0), (60.0, 60.0)], Stroke::default());
    let gradient = Gradient::linear(LinearGradientDesc {
        common: two_stop_common_canvas((220, 30, 30), (30, 30, 220), SpreadMode::Pad),
        line: LinearGradientLine {
            start: [ox + 10.0, oy + 40.0],
            end: [ox + 70.0, oy + 40.0],
        },
    })
    .expect("valid gradient");

    renderer
        .add_shape(
            shape,
            None,
            None,
            ShapeDrawCommandOptions::new()
                .transform(TransformInstance::translation(ox + 10.0, oy + 10.0))
                .fill(Fill::Gradient(gradient)),
        )
        .unwrap();

    vec![
        // Canvas units are evaluated in screen space, so the transformed shape still
        // follows the tile-space line rather than restarting in local space.
        PixelExpectation::opaque_approx(
            ox as u32 + 15,
            oy as u32 + 40,
            200,
            30,
            50,
            45,
            "t39_left_red",
        ),
        // Right edge should be bluish
        PixelExpectation::opaque_approx(
            ox as u32 + 65,
            oy as u32 + 40,
            50,
            30,
            200,
            45,
            "t39_right_blue",
        ),
    ]
}

fn tile_40_radial_gradient(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(40);
    let shape = Shape::rect(
        [(ox + 10.0, oy + 10.0), (ox + 70.0, oy + 70.0)],
        Stroke::default(),
    );
    let gradient = Gradient::radial(RadialGradientDesc {
        common: two_stop_common((240, 240, 30), (30, 180, 30), SpreadMode::Pad),
        center: [ox + 40.0, oy + 40.0],
        shape: RadialGradientShape::Circle,
        size: RadialGradientSize::ExplicitCircleRadius(30.0),
    })
    .expect("valid gradient");

    renderer
        .add_shape(
            shape,
            None,
            None,
            ShapeDrawCommandOptions::new().fill(Fill::Gradient(gradient)),
        )
        .unwrap();

    vec![
        // Center should be yellowish
        PixelExpectation::opaque_approx(
            ox as u32 + 40,
            oy as u32 + 40,
            240,
            240,
            30,
            20,
            "t40_center_yellow",
        ),
        // Edge (~28px from center) should be greenish
        PixelExpectation::opaque_approx(
            ox as u32 + 65,
            oy as u32 + 40,
            30,
            180,
            30,
            50,
            "t40_edge_green",
        ),
    ]
}

fn tile_41_conic_gradient(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(41);
    let shape = Shape::rect(
        [(ox + 10.0, oy + 10.0), (ox + 70.0, oy + 70.0)],
        Stroke::default(),
    );
    let tau = std::f32::consts::TAU;
    let gradient = Gradient::conic(ConicGradientDesc {
        common: GradientCommonDesc {
            units: GradientUnits::Local,
            spread: SpreadMode::Pad,
            interpolation: ColorInterpolation::Srgb,
            stops: vec![
                GradientStop {
                    positions: GradientStopPositions::Single(GradientStopOffset::ConicRadians(0.0)),
                    color: GradientColor::Srgb {
                        red: 1.0,
                        green: 0.0,
                        blue: 0.0,
                        alpha: 1.0,
                    },
                    hint_to_next_segment: None,
                },
                GradientStop {
                    positions: GradientStopPositions::Single(GradientStopOffset::ConicRadians(
                        tau / 3.0,
                    )),
                    color: GradientColor::Srgb {
                        red: 0.0,
                        green: 1.0,
                        blue: 0.0,
                        alpha: 1.0,
                    },
                    hint_to_next_segment: None,
                },
                GradientStop {
                    positions: GradientStopPositions::Single(GradientStopOffset::ConicRadians(
                        2.0 * tau / 3.0,
                    )),
                    color: GradientColor::Srgb {
                        red: 0.0,
                        green: 0.0,
                        blue: 1.0,
                        alpha: 1.0,
                    },
                    hint_to_next_segment: None,
                },
                GradientStop {
                    positions: GradientStopPositions::Single(GradientStopOffset::ConicRadians(tau)),
                    color: GradientColor::Srgb {
                        red: 1.0,
                        green: 0.0,
                        blue: 0.0,
                        alpha: 1.0,
                    },
                    hint_to_next_segment: None,
                },
            ]
            .into(),
        },
        center: [ox + 40.0, oy + 40.0],
        start_angle_radians: 0.0,
    })
    .expect("valid gradient");

    renderer
        .add_shape(
            shape,
            None,
            None,
            ShapeDrawCommandOptions::new().fill(Fill::Gradient(gradient)),
        )
        .unwrap();

    vec![
        // Right of center (0°) should be reddish
        PixelExpectation::opaque_approx(
            ox as u32 + 65,
            oy as u32 + 40,
            230,
            0,
            25,
            60,
            "t41_right_red",
        ),
    ]
}

fn tile_42_repeating_linear_gradient(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(42);
    let shape = Shape::rect(
        [(ox + 10.0, oy + 10.0), (ox + 70.0, oy + 70.0)],
        Stroke::default(),
    );
    let gradient = Gradient::linear(LinearGradientDesc {
        common: GradientCommonDesc {
            units: GradientUnits::Local,
            spread: SpreadMode::Repeat,
            interpolation: ColorInterpolation::Srgb,
            stops: vec![
                GradientStop {
                    positions: GradientStopPositions::Single(GradientStopOffset::LinearRadial(0.0)),
                    color: GradientColor::Srgb {
                        red: 0.9,
                        green: 0.1,
                        blue: 0.1,
                        alpha: 1.0,
                    },
                    hint_to_next_segment: None,
                },
                GradientStop {
                    positions: GradientStopPositions::Single(GradientStopOffset::LinearRadial(0.5)),
                    color: GradientColor::Srgb {
                        red: 0.1,
                        green: 0.1,
                        blue: 0.9,
                        alpha: 1.0,
                    },
                    hint_to_next_segment: None,
                },
                GradientStop {
                    positions: GradientStopPositions::Single(GradientStopOffset::LinearRadial(1.0)),
                    color: GradientColor::Srgb {
                        red: 0.9,
                        green: 0.1,
                        blue: 0.1,
                        alpha: 1.0,
                    },
                    hint_to_next_segment: None,
                },
            ]
            .into(),
        },
        // Short axis so it repeats ~3 times across the 60px rect.
        line: LinearGradientLine {
            start: [ox + 10.0, oy + 40.0],
            end: [ox + 30.0, oy + 40.0],
        },
    })
    .expect("valid gradient");

    renderer
        .add_shape(
            shape,
            None,
            None,
            ShapeDrawCommandOptions::new().fill(Fill::Gradient(gradient)),
        )
        .unwrap();

    vec![
        // Midpoint of first period (10px in 20px period) should be bluish
        PixelExpectation::opaque_approx(
            ox as u32 + 20,
            oy as u32 + 40,
            25,
            25,
            200,
            70,
            "t42_repeat_mid_blue",
        ),
        // Midpoint of a later repeated period should still be bluish.
        PixelExpectation::opaque_approx(
            ox as u32 + 60,
            oy as u32 + 40,
            25,
            25,
            200,
            70,
            "t42_repeat_far_blue",
        ),
    ]
}

fn tile_43_gradient_hard_stops(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(43);
    let shape = Shape::rect(
        [(ox + 10.0, oy + 10.0), (ox + 70.0, oy + 70.0)],
        Stroke::default(),
    );
    let gradient = Gradient::linear(LinearGradientDesc {
        common: GradientCommonDesc {
            units: GradientUnits::Local,
            spread: SpreadMode::Pad,
            interpolation: ColorInterpolation::Srgb,
            stops: vec![
                GradientStop {
                    positions: GradientStopPositions::Single(GradientStopOffset::LinearRadial(0.0)),
                    color: GradientColor::Srgb {
                        red: 0.86,
                        green: 0.2,
                        blue: 0.2,
                        alpha: 1.0,
                    },
                    hint_to_next_segment: None,
                },
                GradientStop {
                    positions: GradientStopPositions::Single(GradientStopOffset::LinearRadial(0.5)),
                    color: GradientColor::Srgb {
                        red: 0.86,
                        green: 0.2,
                        blue: 0.2,
                        alpha: 1.0,
                    },
                    hint_to_next_segment: None,
                },
                GradientStop {
                    positions: GradientStopPositions::Single(GradientStopOffset::LinearRadial(0.5)),
                    color: GradientColor::Srgb {
                        red: 0.12,
                        green: 0.12,
                        blue: 0.86,
                        alpha: 1.0,
                    },
                    hint_to_next_segment: None,
                },
                GradientStop {
                    positions: GradientStopPositions::Single(GradientStopOffset::LinearRadial(1.0)),
                    color: GradientColor::Srgb {
                        red: 0.12,
                        green: 0.12,
                        blue: 0.86,
                        alpha: 1.0,
                    },
                    hint_to_next_segment: None,
                },
            ]
            .into(),
        },
        line: LinearGradientLine {
            start: [ox + 10.0, oy + 40.0],
            end: [ox + 70.0, oy + 40.0],
        },
    })
    .expect("valid gradient");

    renderer
        .add_shape(
            shape,
            None,
            None,
            ShapeDrawCommandOptions::new().fill(Fill::Gradient(gradient)),
        )
        .unwrap();

    vec![
        // Just left of the 50% boundary should still be red.
        PixelExpectation::opaque_approx(
            ox as u32 + 39,
            oy as u32 + 40,
            200,
            50,
            60,
            45,
            "t43_left_of_hard_stop_red",
        ),
        // Just right of the 50% boundary should flip immediately to blue.
        PixelExpectation::opaque_approx(
            ox as u32 + 41,
            oy as u32 + 40,
            30,
            30,
            200,
            45,
            "t43_right_of_hard_stop_blue",
        ),
    ]
}

fn tile_44_gradient_clipped(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(44);

    // Rounded parent creates the clip mask.
    let parent = Shape::rounded_rect(
        [(ox + 10.0, oy + 10.0), (ox + 70.0, oy + 70.0)],
        BorderRadii::new(15.0),
        Stroke::default(),
    );
    // Child rect filled with a gradient, clipped by rounded parent.
    let child = Shape::rect(
        [(ox + 10.0, oy + 10.0), (ox + 70.0, oy + 70.0)],
        Stroke::default(),
    );

    let gradient = Gradient::linear(LinearGradientDesc {
        common: two_stop_common((30, 220, 30), (220, 30, 220), SpreadMode::Pad),
        line: LinearGradientLine {
            start: [ox + 10.0, oy + 10.0],
            end: [ox + 70.0, oy + 70.0],
        },
    })
    .expect("valid gradient");

    let parent_id = renderer
        .add_shape(parent, None, None, ShapeDrawCommandOptions::new())
        .unwrap();
    renderer
        .add_shape(
            child,
            Some(parent_id),
            None,
            ShapeDrawCommandOptions::new().fill(Fill::Gradient(gradient)),
        )
        .unwrap();

    vec![
        // For tile-local pixels, t = (x + y + 1 - 20) / 120 at the pixel center.
        // Interpolate sRGB bytes and allow a tolerance of 2 for GPU rounding.
        PixelExpectation::opaque_approx(
            ox as u32 + 25,
            oy as u32 + 25,
            79,
            171,
            79,
            2,
            "t44_gradient_start",
        ),
        PixelExpectation::opaque_approx(
            ox as u32 + 40,
            oy as u32 + 40,
            127,
            123,
            127,
            2,
            "t44_center_gradient_mix",
        ),
        PixelExpectation::opaque_approx(
            ox as u32 + 55,
            oy as u32 + 55,
            174,
            76,
            174,
            2,
            "t44_gradient_end",
        ),
        // The white canvas shows outside the rounded clip.
        PixelExpectation::opaque(
            ox as u32 + 11,
            oy as u32 + 11,
            255,
            255,
            255,
            "t44_outside_rounded_clip",
        ),
    ]
}

fn tile_45_gradient_group_blur(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(45);

    // Only the shape is blurred; the background stripe must stay sharp.
    let bg_stripe = Shape::rect(
        [(ox + 5.0, oy + 30.0), (ox + 75.0, oy + 50.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            bg_stripe,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(50, 180, 50)),
        )
        .unwrap(); // green stripe

    // Gradient-filled shape with group blur
    let shape = Shape::rect(
        [(ox + 15.0, oy + 10.0), (ox + 65.0, oy + 70.0)],
        Stroke::default(),
    );
    let gradient = Gradient::linear(LinearGradientDesc {
        common: two_stop_common((220, 50, 50), (50, 50, 220), SpreadMode::Pad),
        line: LinearGradientLine {
            start: [ox + 15.0, oy + 10.0],
            end: [ox + 65.0, oy + 70.0],
        },
    })
    .expect("valid gradient");

    let id = renderer
        .add_shape(
            shape,
            None,
            None,
            ShapeDrawCommandOptions::new().fill(Fill::Gradient(gradient)),
        )
        .unwrap();

    let blur_params = BlurParams {
        radius: 8.0,
        _pad: 0.0,
    };
    renderer
        .set_group_effect(id, BLUR_EFFECT_ID, bytemuck::bytes_of(&blur_params))
        .expect("Failed to set group effect");

    vec![
        // Pixels outside the left and top edges require both blur passes to spread coverage.
        PixelExpectation::opaque(
            ox as u32 + 14,
            oy as u32 + 40,
            118,
            143,
            78,
            "t45_horizontal_blur_spread",
        ),
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 9,
            227,
            202,
            206,
            "t45_vertical_blur_spread",
        ),
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 40,
            134,
            50,
            137,
            "t45_blurred_gradient_center",
        ),
        // This stripe pixel is beyond the eight-pixel blur radius.
        PixelExpectation::opaque(
            ox as u32 + 5,
            oy as u32 + 40,
            50,
            180,
            50,
            "t45_bg_stripe_sharp",
        ),
    ]
}

fn tile_46_gradient_backdrop_blur(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(46);

    // Red background
    let bg = Shape::rect(
        [(ox + 5.0, oy + 5.0), (ox + 75.0, oy + 75.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            bg,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(220, 50, 50)),
        )
        .unwrap();

    // Gradient stripe that partially overlaps the backdrop panel.
    // Outside the panel it should be crisp; under the panel it should get blurred.
    let stripe = Shape::rect(
        [(ox + 10.0, oy + 32.0), (ox + 70.0, oy + 48.0)],
        Stroke::default(),
    );
    let gradient = Gradient::linear(LinearGradientDesc {
        common: two_stop_common((50, 50, 220), (50, 220, 50), SpreadMode::Pad),
        line: LinearGradientLine {
            start: [ox + 10.0, oy + 32.0],
            end: [ox + 70.0, oy + 48.0],
        },
    })
    .expect("valid gradient");

    renderer
        .add_shape(
            stripe,
            None,
            None,
            ShapeDrawCommandOptions::new().fill(Fill::Gradient(gradient)),
        )
        .unwrap();

    // Backdrop blur panel on top (leaf, no children)
    let panel = Shape::rect(
        [(ox + 20.0, oy + 15.0), (ox + 60.0, oy + 65.0)],
        Stroke::default(),
    );
    let panel_id = renderer
        .add_shape(
            panel,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgba(255, 255, 255, 80)),
        )
        .unwrap();

    let blur_params = BlurParams {
        radius: 10.0,
        _pad: 0.0,
    };
    renderer
        .set_shape_backdrop_effect(
            panel_id,
            BLUR_EFFECT_ID,
            bytemuck::bytes_of(&blur_params),
            BackdropEffectConfig::default(),
        )
        .expect("Failed to set backdrop effect");

    vec![
        // Panel interior: blurred mix of red bg + gradient stripe under the panel's white fill
        PixelExpectation::new(
            ox as u32 + 40,
            oy as u32 + 40,
            158,
            185,
            183,
            255,
            "t46_backdrop_interior",
        )
        .with_tolerance(25),
        // The uncovered left side of the gradient stays sharp and blue.
        PixelExpectation::opaque(
            ox as u32 + 12,
            oy as u32 + 40,
            50,
            60,
            210,
            "t46_gradient_stripe_outside",
        )
        .with_tolerance(30),
        // Red background outside panel and stripe
        PixelExpectation::opaque(ox as u32 + 10, oy as u32 + 10, 220, 50, 50, "t46_bg_intact"),
    ]
}

// Gradient regression tiles

/// Stencil increment must draw the parent gradient before its child.
fn tile_47_gradient_nonleaf_stencil(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(47);

    // Rounded-rect parent with gradient fill (forces stencil path, not scissor).
    let parent = Shape::rounded_rect(
        [(ox + 10.0, oy + 10.0), (ox + 70.0, oy + 70.0)],
        BorderRadii::new(12.0),
        Stroke::default(),
    );
    let gradient = Gradient::linear(LinearGradientDesc {
        common: two_stop_common((220, 30, 30), (30, 30, 220), SpreadMode::Pad),
        line: LinearGradientLine {
            start: [ox + 10.0, oy + 40.0],
            end: [ox + 70.0, oy + 40.0],
        },
    })
    .expect("valid gradient");

    let parent_id = renderer
        .add_shape(
            parent,
            None,
            None,
            ShapeDrawCommandOptions::new().fill(Fill::Gradient(gradient)),
        )
        .unwrap();

    // Small opaque child in the center.
    let child = Shape::rect(
        [(ox + 30.0, oy + 30.0), (ox + 50.0, oy + 50.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            child,
            Some(parent_id),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(255, 255, 0)),
        )
        .unwrap(); // yellow

    vec![
        // Left side of parent (gradient should be red-ish, not white/background).
        PixelExpectation::opaque_approx(
            ox as u32 + 15,
            oy as u32 + 40,
            200,
            30,
            50,
            60,
            "t47_parent_gradient_left_red",
        ),
        // Right side of parent (gradient should be blue-ish, not white/background).
        PixelExpectation::opaque_approx(
            ox as u32 + 65,
            oy as u32 + 40,
            50,
            30,
            200,
            60,
            "t47_parent_gradient_right_blue",
        ),
        // Child center should be yellow (drawn on top of gradient parent).
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 40,
            255,
            255,
            0,
            "t47_child_yellow",
        ),
    ]
}

/// A solid draw following a gradient must not inherit the gradient binding.
fn tile_48_gradient_state_leak(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(48);

    // First: gradient-filled rect (left half).
    let grad_shape = Shape::rect(
        [(ox + 5.0, oy + 10.0), (ox + 37.0, oy + 70.0)],
        Stroke::default(),
    );
    let gradient = Gradient::linear(LinearGradientDesc {
        common: two_stop_common((220, 30, 30), (30, 220, 30), SpreadMode::Pad),
        line: LinearGradientLine {
            start: [ox + 5.0, oy + 10.0],
            end: [ox + 37.0, oy + 70.0],
        },
    })
    .expect("valid gradient");

    renderer
        .add_shape(
            grad_shape,
            None,
            None,
            ShapeDrawCommandOptions::new().fill(Fill::Gradient(gradient)),
        )
        .unwrap();

    // Second: solid cyan rect (right half), drawn immediately after the gradient.
    let solid_shape = Shape::rect(
        [(ox + 43.0, oy + 10.0), (ox + 75.0, oy + 70.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            solid_shape,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(0, 220, 220)),
        )
        .unwrap(); // cyan

    vec![
        // For tile-local pixels, t = (32*(x + 0.5 - 5) + 60*(y + 0.5 - 10)) / 4624.
        // Interpolate sRGB bytes and allow a tolerance of 2 for GPU rounding.
        PixelExpectation::opaque_approx(
            ox as u32 + 13,
            oy as u32 + 25,
            171,
            79,
            30,
            2,
            "t48_gradient_start",
        ),
        PixelExpectation::opaque_approx(
            ox as u32 + 21,
            oy as u32 + 40,
            123,
            127,
            30,
            2,
            "t48_gradient_center",
        ),
        PixelExpectation::opaque_approx(
            ox as u32 + 29,
            oy as u32 + 55,
            76,
            174,
            30,
            2,
            "t48_gradient_end",
        ),
        // The cyan rectangle must not inherit the preceding shape's gradient.
        PixelExpectation::opaque(
            ox as u32 + 59,
            oy as u32 + 40,
            0,
            220,
            220,
            "t48_solid_cyan",
        ),
    ]
}

/// Four quadrant colors expose unit mismatches between CPU angles and shader ramp lookups.
fn tile_49_conic_quadrant_colors(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(49);
    let cx = ox + 40.0;
    let cy = oy + 40.0;

    let shape = Shape::rect(
        [(ox + 10.0, oy + 10.0), (ox + 70.0, oy + 70.0)],
        Stroke::default(),
    );
    use std::f32::consts::{FRAC_PI_2, PI, TAU};

    let gradient = Gradient::conic(ConicGradientDesc {
        common: GradientCommonDesc {
            units: GradientUnits::Local,
            spread: SpreadMode::Pad,
            interpolation: ColorInterpolation::Srgb,
            stops: vec![
                GradientStop {
                    positions: GradientStopPositions::Single(GradientStopOffset::ConicRadians(0.0)),
                    color: GradientColor::Srgb {
                        red: 1.0,
                        green: 0.0,
                        blue: 0.0,
                        alpha: 1.0,
                    },
                    hint_to_next_segment: None,
                },
                GradientStop {
                    positions: GradientStopPositions::Single(GradientStopOffset::ConicRadians(
                        FRAC_PI_2,
                    )),
                    color: GradientColor::Srgb {
                        red: 0.0,
                        green: 1.0,
                        blue: 0.0,
                        alpha: 1.0,
                    },
                    hint_to_next_segment: None,
                },
                GradientStop {
                    positions: GradientStopPositions::Single(GradientStopOffset::ConicRadians(PI)),
                    color: GradientColor::Srgb {
                        red: 0.0,
                        green: 0.0,
                        blue: 1.0,
                        alpha: 1.0,
                    },
                    hint_to_next_segment: None,
                },
                GradientStop {
                    positions: GradientStopPositions::Single(GradientStopOffset::ConicRadians(
                        3.0 * FRAC_PI_2,
                    )),
                    color: GradientColor::Srgb {
                        red: 1.0,
                        green: 1.0,
                        blue: 0.0,
                        alpha: 1.0,
                    },
                    hint_to_next_segment: None,
                },
                GradientStop {
                    positions: GradientStopPositions::Single(GradientStopOffset::ConicRadians(TAU)),
                    color: GradientColor::Srgb {
                        red: 1.0,
                        green: 0.0,
                        blue: 0.0,
                        alpha: 1.0,
                    },
                    hint_to_next_segment: None,
                },
            ]
            .into(),
        },
        center: [cx, cy],
        start_angle_radians: 0.0,
    })
    .expect("valid gradient");

    renderer
        .add_shape(
            shape,
            None,
            None,
            ShapeDrawCommandOptions::new().fill(Fill::Gradient(gradient)),
        )
        .unwrap();

    vec![
        // At 90 degrees, below the center, the gradient is green.
        PixelExpectation::opaque_approx(
            ox as u32 + 40,
            oy as u32 + 65,
            0,
            255,
            0,
            40,
            "t49_bottom_green_90deg",
        ),
        // At 180 degrees, left of the center, the gradient is blue.
        PixelExpectation::opaque_approx(
            ox as u32 + 15,
            oy as u32 + 40,
            0,
            0,
            255,
            40,
            "t49_left_blue_180deg",
        ),
        // At 270 degrees, above the center, the gradient is yellow.
        PixelExpectation::opaque_approx(
            ox as u32 + 40,
            oy as u32 + 15,
            255,
            255,
            0,
            40,
            "t49_top_yellow_270deg",
        ),
    ]
}

/// Visible overflow skips the parent clip but still respects the ancestor clip.
fn tile_50_overflow_visible_delegates_to_ancestor(
    renderer: &mut Renderer,
) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(50);

    let outer = Shape::rounded_rect(
        [(ox + 5.0, oy + 5.0), (ox + 75.0, oy + 75.0)],
        BorderRadii::new(12.0),
        Stroke::default(),
    );
    let outer_id = renderer
        .add_shape(
            outer,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(180, 180, 220)),
        )
        .unwrap();

    let middle = Shape::rounded_rect(
        [(ox + 25.0, oy + 25.0), (ox + 55.0, oy + 55.0)],
        BorderRadii::new(6.0),
        Stroke::default(),
    );
    let middle_id = renderer
        .add_shape(
            middle,
            Some(outer_id),
            None,
            ShapeDrawCommandOptions::new()
                .color(Color::rgb(100, 200, 120))
                .clips_children(false),
        )
        .unwrap();

    let child = Shape::rect(
        [(ox + 0.0, oy + 35.0), (ox + 80.0, oy + 50.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            child,
            Some(middle_id),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(220, 60, 60)),
        )
        .unwrap();

    vec![
        PixelExpectation::opaque(
            ox as u32 + 15,
            oy as u32 + 42,
            220,
            60,
            60,
            "t50_child_visible_outside_middle",
        ),
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 42,
            220,
            60,
            60,
            "t50_child_over_middle",
        ),
        PixelExpectation::opaque(
            ox as u32 + 2,
            oy as u32 + 42,
            255,
            255,
            255,
            "t50_outer_still_clips_child",
        ),
    ]
}

/// A non-clipping rectangle must preserve the ancestor clip for its descendants.
fn tile_51_clip_rect_overflow_visible_container(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(51);

    let outer = Shape::rounded_rect(
        [(ox + 5.0, oy + 5.0), (ox + 75.0, oy + 75.0)],
        BorderRadii::new(12.0),
        Stroke::default(),
    );
    let outer_id = renderer
        .add_shape(
            outer,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(180, 180, 220)),
        )
        .unwrap();

    let container_id = renderer
        .add_clipping_rect(
            [(ox + 25.0, oy + 25.0), (ox + 55.0, oy + 55.0)],
            Some(outer_id),
            None::<TransformInstance>,
            false,
        )
        .unwrap();

    let child = Shape::rect(
        [(ox + 0.0, oy + 35.0), (ox + 80.0, oy + 50.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            child,
            Some(container_id),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(70, 80, 220)),
        )
        .unwrap();

    vec![
        PixelExpectation::opaque(
            ox as u32 + 15,
            oy as u32 + 42,
            70,
            80,
            220,
            "t51_child_visible_outside_clip_rect",
        ),
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 42,
            70,
            80,
            220,
            "t51_child_inside_clip_rect",
        ),
        PixelExpectation::opaque(
            ox as u32 + 2,
            oy as u32 + 42,
            255,
            255,
            255,
            "t51_outer_still_clips_child",
        ),
    ]
}

/// Visible overflow clips the backdrop to its node and descendants to the ancestor.
fn tile_52_backdrop_overflow_visible_children(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(52);

    let outer = Shape::rounded_rect(
        [(ox + 5.0, oy + 5.0), (ox + 75.0, oy + 75.0)],
        BorderRadii::new(12.0),
        Stroke::default(),
    );
    let outer_id = renderer
        .add_shape(
            outer,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(185, 185, 220)),
        )
        .unwrap();

    let backdrop_panel = Shape::rounded_rect(
        [(ox + 25.0, oy + 25.0), (ox + 55.0, oy + 55.0)],
        BorderRadii::new(6.0),
        Stroke::default(),
    );
    let panel_id = renderer
        .add_shape(
            backdrop_panel,
            Some(outer_id),
            None,
            ShapeDrawCommandOptions::new()
                .color(Color::rgba(255, 255, 255, 80))
                .clips_children(false),
        )
        .unwrap();

    let blur_params = BlurParams {
        radius: 5.0,
        _pad: 0.0,
    };
    renderer
        .set_shape_backdrop_effect(
            panel_id,
            BLUR_EFFECT_ID,
            bytemuck::bytes_of(&blur_params),
            BackdropEffectConfig::default(),
        )
        .expect("Failed to set backdrop effect");

    let child = Shape::rect(
        [(ox + 0.0, oy + 35.0), (ox + 80.0, oy + 50.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            child,
            Some(panel_id),
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(230, 70, 70)),
        )
        .unwrap();

    vec![
        PixelExpectation::opaque(
            ox as u32 + 15,
            oy as u32 + 42,
            230,
            70,
            70,
            "t52_child_visible_outside_backdrop",
        ),
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 42,
            230,
            70,
            70,
            "t52_child_inside_backdrop",
        ),
        PixelExpectation::opaque_approx(
            ox as u32 + 40,
            oy as u32 + 30,
            207,
            207,
            231,
            25,
            "t52_panel_pass_inside_outside_child",
        ),
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 22,
            185,
            185,
            220,
            "t52_outside_panel_shows_outer",
        ),
        PixelExpectation::opaque(
            ox as u32 + 2,
            oy as u32 + 42,
            255,
            255,
            255,
            "t52_outer_still_clips_child",
        ),
    ]
}

fn tile_53_texture_original_size(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(53);
    let shape = Shape::rect(
        [(ox + 10.0, oy + 10.0), (ox + 70.0, oy + 70.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            shape,
            None,
            None,
            ShapeDrawCommandOptions::new()
                .background_texture(ShapeTextureOptions::new(SOLID_RED_TEXTURE_ID))
                .foreground_texture(
                    ShapeTextureOptions::new(SOLID_GREEN_20X20_TEXTURE_ID)
                        .fit_mode(ShapeTextureFitMode::OriginalSize),
                )
                .color(Color::WHITE),
        )
        .unwrap();

    vec![
        PixelExpectation::opaque(
            ox as u32 + 20,
            oy as u32 + 20,
            0,
            255,
            0,
            "t53_original_size_texture_visible",
        ),
        PixelExpectation::opaque(
            ox as u32 + 45,
            oy as u32 + 20,
            255,
            0,
            0,
            "t53_foreground_original_size_does_not_stretch",
        ),
    ]
}

fn tile_54_texture_original_size_scaled(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(54);
    let shape = Shape::rect([(0.0, 0.0), (20.0, 20.0)], Stroke::default());
    renderer
        .add_shape(
            shape,
            None,
            None,
            ShapeDrawCommandOptions::new()
                .background_texture(
                    ShapeTextureOptions::new(SOLID_GREEN_20X20_TEXTURE_ID)
                        .fit_mode(ShapeTextureFitMode::OriginalSize),
                )
                .color(Color::WHITE)
                .transform(TransformInstance::affine_2d(
                    2.0,
                    0.0,
                    0.0,
                    2.0,
                    ox + 10.0,
                    oy + 10.0,
                )),
        )
        .unwrap();

    vec![
        PixelExpectation::opaque(
            ox as u32 + 30,
            oy as u32 + 30,
            0,
            255,
            0,
            "t54_scaled_original_size_texture_scales_with_shape_center",
        ),
        PixelExpectation::opaque(
            ox as u32 + 45,
            oy as u32 + 20,
            0,
            255,
            0,
            "t54_scaled_original_size_texture_scales_with_shape_edge",
        ),
        PixelExpectation::opaque(
            ox as u32 + 55,
            oy as u32 + 20,
            255,
            255,
            255,
            "t54_scaled_original_size_texture_clipped_outside_scaled_shape",
        ),
    ]
}

fn tile_55_backdrop_capture_screen_rect(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(55);

    let red_source = Shape::rect(
        [(ox + 5.0, oy + 10.0), (ox + 35.0, oy + 40.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            red_source,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(220, 50, 50)),
        )
        .unwrap();

    let blue_behind_panel = Shape::rect(
        [(ox + 45.0, oy + 20.0), (ox + 75.0, oy + 50.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            blue_behind_panel,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(50, 50, 220)),
        )
        .unwrap();

    let panel = Shape::rect(
        [(ox + 45.0, oy + 20.0), (ox + 75.0, oy + 50.0)],
        Stroke::default(),
    );
    let panel_id = renderer
        .add_shape(panel, None, None, ShapeDrawCommandOptions::new())
        .unwrap();

    renderer
        .set_shape_backdrop_effect(
            panel_id,
            PASSTHROUGH_EFFECT_ID,
            &[],
            BackdropEffectConfig::new()
                .capture_area(BackdropCaptureArea::ScreenRect([
                    (ox + 5.0, oy + 10.0),
                    (ox + 35.0, oy + 40.0),
                ]))
                .downsample(1.0),
        )
        .expect("Failed to set backdrop screen-rect effect");

    vec![
        PixelExpectation::opaque(
            ox as u32 + 60,
            oy as u32 + 35,
            220,
            50,
            50,
            "t55_panel_uses_captured_rect",
        ),
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 35,
            255,
            255,
            255,
            "t55_outside_panel_stays_canvas_bg",
        ),
        PixelExpectation::opaque(
            ox as u32 + 20,
            oy as u32 + 25,
            220,
            50,
            50,
            "t55_source_rect_stays_red",
        ),
    ]
}

fn tile_56_backdrop_capture_downsampled(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(56);

    let green_source = Shape::rect(
        [(ox + 5.0, oy + 10.0), (ox + 45.0, oy + 50.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            green_source,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(50, 180, 50)),
        )
        .unwrap();

    let red_behind_panel = Shape::rect(
        [(ox + 50.0, oy + 15.0), (ox + 75.0, oy + 55.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            red_behind_panel,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(220, 50, 50)),
        )
        .unwrap();

    let panel = Shape::rect(
        [(ox + 50.0, oy + 15.0), (ox + 75.0, oy + 55.0)],
        Stroke::default(),
    );
    let panel_id = renderer
        .add_shape(panel, None, None, ShapeDrawCommandOptions::new())
        .unwrap();

    renderer
        .set_shape_backdrop_effect(
            panel_id,
            PASSTHROUGH_EFFECT_ID,
            &[],
            BackdropEffectConfig::new()
                .capture_area(BackdropCaptureArea::ScreenRect([
                    (ox + 5.0, oy + 10.0),
                    (ox + 45.0, oy + 50.0),
                ]))
                .downsample(0.5),
        )
        .expect("Failed to set downsampled backdrop effect");

    vec![
        PixelExpectation::opaque(
            ox as u32 + 62,
            oy as u32 + 35,
            50,
            180,
            50,
            "t56_downsampled_panel_stays_green",
        ),
        PixelExpectation::opaque(
            ox as u32 + 47,
            oy as u32 + 35,
            255,
            255,
            255,
            "t56_gap_between_source_and_panel_is_canvas_bg",
        ),
        PixelExpectation::opaque(
            ox as u32 + 20,
            oy as u32 + 25,
            50,
            180,
            50,
            "t56_source_rect_stays_green",
        ),
    ]
}

fn tile_57_gradient_backdrop_oversized_capture_falls_back(
    renderer: &mut Renderer,
) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(57);

    let panel = Shape::rect(
        [(ox + 10.0, oy + 10.0), (ox + 70.0, oy + 70.0)],
        Stroke::default(),
    );
    let gradient = Gradient::linear(LinearGradientDesc {
        common: two_stop_common((220, 30, 30), (30, 30, 220), SpreadMode::Pad),
        line: LinearGradientLine {
            start: [ox + 10.0, oy + 40.0],
            end: [ox + 70.0, oy + 40.0],
        },
    })
    .expect("valid gradient");

    let panel_id = renderer
        .add_shape(
            panel,
            None,
            None,
            ShapeDrawCommandOptions::new().fill(Fill::Gradient(gradient)),
        )
        .unwrap();

    renderer
        .set_shape_backdrop_effect(
            panel_id,
            PASSTHROUGH_EFFECT_ID,
            &[],
            BackdropEffectConfig::new().capture_area(BackdropCaptureArea::ScreenRect([
                (0.0, 0.0),
                (20_000.0, 20_000.0),
            ])),
        )
        .expect("Failed to set oversized gradient backdrop effect");

    vec![
        PixelExpectation::opaque_approx(
            ox as u32 + 20,
            oy as u32 + 40,
            190,
            30,
            60,
            60,
            "t57_left_side_remains_gradient_red",
        ),
        PixelExpectation::opaque_approx(
            ox as u32 + 60,
            oy as u32 + 40,
            60,
            30,
            190,
            60,
            "t57_right_side_remains_gradient_blue",
        ),
        PixelExpectation::opaque(
            ox as u32 + 5,
            oy as u32 + 5,
            255,
            255,
            255,
            "t57_outside_panel_stays_canvas_bg",
        ),
    ]
}

fn tile_58_backdrop_budgeted_capture_falls_back(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(58);

    let red_source = Shape::rect(
        [(ox + 5.0, oy + 10.0), (ox + 75.0, oy + 45.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            red_source,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(220, 50, 50)),
        )
        .unwrap();

    let blue_behind_panel = Shape::rect(
        [(ox + 45.0, oy + 20.0), (ox + 75.0, oy + 50.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            blue_behind_panel,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(50, 50, 220)),
        )
        .unwrap();

    let panel = Shape::rect(
        [(ox + 45.0, oy + 20.0), (ox + 75.0, oy + 50.0)],
        Stroke::default(),
    );
    let panel_id = renderer
        .add_shape(panel, None, None, ShapeDrawCommandOptions::new())
        .unwrap();

    renderer
        .set_shape_backdrop_effect(
            panel_id,
            PASSTHROUGH_EFFECT_ID,
            &[],
            BackdropEffectConfig::new().capture_area(BackdropCaptureArea::ScreenRect([
                (ox + 5.0, oy + 10.0),
                (ox + 1_505.0, oy + 1_510.0),
            ])),
        )
        .expect("Failed to set budgeted backdrop effect");

    vec![
        PixelExpectation::opaque(
            ox as u32 + 60,
            oy as u32 + 35,
            50,
            50,
            220,
            "t58_panel_stays_underlying_blue_when_capture_budget_skips",
        ),
        PixelExpectation::opaque(
            ox as u32 + 20,
            oy as u32 + 25,
            220,
            50,
            50,
            "t58_source_rect_stays_red",
        ),
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 60,
            255,
            255,
            255,
            "t58_outside_panel_stays_canvas_bg",
        ),
    ]
}

fn tile_59_backdrop_node_bounds_offscreen_preserves_size(
    renderer: &mut Renderer,
) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(59);

    let red_band = Shape::rect(
        [(ox + 60.0, oy + 15.0), (ox + 70.0, oy + 55.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            red_band,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(220, 50, 50)),
        )
        .unwrap();

    let green_band = Shape::rect(
        [(ox + 70.0, oy + 15.0), (ox + 80.0, oy + 55.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            green_band,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(50, 180, 50)),
        )
        .unwrap();

    let blue_offscreen_band = Shape::rect(
        [(ox + 80.0, oy + 15.0), (ox + 170.0, oy + 55.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            blue_offscreen_band,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(50, 50, 220)),
        )
        .unwrap();

    let yellow_offscreen_band = Shape::rect(
        [(ox + 170.0, oy + 15.0), (ox + 190.0, oy + 55.0)],
        Stroke::default(),
    );
    renderer
        .add_shape(
            yellow_offscreen_band,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(220, 200, 50)),
        )
        .unwrap();

    let panel = Shape::rect(
        [(ox + 60.0, oy + 15.0), (ox + 190.0, oy + 55.0)],
        Stroke::default(),
    );
    let panel_id = renderer
        .add_shape(panel, None, None, ShapeDrawCommandOptions::new())
        .unwrap();

    renderer
        .set_shape_backdrop_effect(
            panel_id,
            PASSTHROUGH_EFFECT_ID,
            &[],
            BackdropEffectConfig::default(),
        )
        .expect("Failed to set offscreen node-bounds backdrop effect");

    vec![
        PixelExpectation::opaque(
            ox as u32 + 65,
            oy as u32 + 35,
            220,
            50,
            50,
            "t59_visible_left_half_stays_red",
        ),
        PixelExpectation::opaque(
            ox as u32 + 75,
            oy as u32 + 35,
            50,
            180,
            50,
            "t59_visible_right_half_stays_green",
        ),
        PixelExpectation::opaque(
            ox as u32 + 85,
            oy as u32 + 35,
            50,
            50,
            220,
            "t59_preserves_size_middle_is_blue",
        ),
        PixelExpectation::opaque(
            ox as u32 + 55,
            oy as u32 + 35,
            255,
            255,
            255,
            "t59_outside_panel_stays_canvas_bg",
        ),
    ]
}

/// The cached effect must draw beyond the source bounds and behind its fill.
fn tile_60_cached_shape_effect_rect(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (origin_x, origin_y) = tile_origin(60);
    let shape = Shape::rect(
        [
            (origin_x + 20.0, origin_y + 20.0),
            (origin_x + 50.0, origin_y + 50.0),
        ],
        Stroke::default(),
    );
    let shape_id = renderer
        .add_shape(
            shape,
            None,
            Some(60_060),
            ShapeDrawCommandOptions::new().color(Color::rgb(220, 50, 50)),
        )
        .unwrap();
    let transparent_child = Shape::rect(
        [
            (origin_x + 30.0, origin_y + 30.0),
            (origin_x + 40.0, origin_y + 40.0),
        ],
        Stroke::default(),
    );
    renderer
        .add_shape(
            transparent_child,
            Some(shape_id),
            None,
            ShapeDrawCommandOptions::new().color(Color::TRANSPARENT),
        )
        .unwrap();
    renderer
        .set_shape_effect(
            shape_id,
            SHAPE_DROP_EFFECT_ID,
            &[],
            ShapeEffectConfig::new().outset(12.0),
        )
        .expect("Failed to attach rectangular cached shape effect");

    vec![
        PixelExpectation::opaque(
            origin_x as u32 + 35,
            origin_y as u32 + 35,
            220,
            50,
            50,
            "t60_source_draws_over_effect",
        ),
        PixelExpectation::opaque(
            origin_x as u32 + 55,
            origin_y as u32 + 35,
            0,
            0,
            255,
            "t60_effect_extends_outside_source",
        ),
        PixelExpectation::opaque(
            origin_x as u32 + 15,
            origin_y as u32 + 5,
            255,
            255,
            255,
            "t60_padding_stays_transparent",
        ),
    ]
}

fn tile_61_cached_shape_effect_path_clipped(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (origin_x, origin_y) = tile_origin(61);
    let clip_parent = Shape::rounded_rect(
        [
            (origin_x + 10.0, origin_y + 8.0),
            (origin_x + 60.0, origin_y + 60.0),
        ],
        BorderRadii::new(8.0),
        Stroke::default(),
    );
    let clip_parent_id = renderer
        .add_shape(clip_parent, None, None, ShapeDrawCommandOptions::new())
        .unwrap();

    let path = Shape::builder()
        .begin((origin_x + 20.0, origin_y + 15.0))
        .line_to((origin_x + 50.0, origin_y + 15.0))
        .line_to((origin_x + 35.0, origin_y + 45.0))
        .close()
        .build();
    let path_id = renderer
        .add_shape(
            path,
            Some(clip_parent_id),
            Some(61_061),
            ShapeDrawCommandOptions::new().color(Color::rgb(50, 180, 50)),
        )
        .unwrap();
    renderer
        .set_shape_effect(
            path_id,
            SHAPE_DROP_EFFECT_ID,
            &[],
            ShapeEffectConfig::new().outset(12.0),
        )
        .expect("Failed to attach path cached shape effect");

    vec![
        PixelExpectation::opaque(
            origin_x as u32 + 35,
            origin_y as u32 + 25,
            50,
            180,
            50,
            "t61_path_draws_over_effect",
        ),
        PixelExpectation::opaque(
            origin_x as u32 + 53,
            origin_y as u32 + 28,
            0,
            0,
            255,
            "t61_path_effect_outside_geometry",
        ),
        PixelExpectation::opaque(
            origin_x as u32 + 65,
            origin_y as u32 + 28,
            255,
            255,
            255,
            "t61_ancestor_clips_effect",
        ),
    ]
}

/// Group preprocessing must include the node's shape effect exactly once.
fn tile_62_cached_shape_effect_inside_group_effect(
    renderer: &mut Renderer,
) -> Vec<PixelExpectation> {
    let (origin_x, origin_y) = tile_origin(62);
    let shape = Shape::rect(
        [
            (origin_x + 20.0, origin_y + 20.0),
            (origin_x + 50.0, origin_y + 50.0),
        ],
        Stroke::default(),
    );
    let shape_id = renderer
        .add_shape(
            shape,
            None,
            Some(62_062),
            ShapeDrawCommandOptions::new().color(Color::rgb(220, 50, 50)),
        )
        .unwrap();
    renderer
        .set_shape_effect(
            shape_id,
            SHAPE_DROP_EFFECT_ID,
            &[],
            ShapeEffectConfig::new().outset(12.0),
        )
        .unwrap();
    renderer
        .set_group_effect(shape_id, PASSTHROUGH_EFFECT_ID, &[])
        .unwrap();

    vec![
        PixelExpectation::opaque(
            origin_x as u32 + 35,
            origin_y as u32 + 35,
            220,
            50,
            50,
            "t62_group_keeps_source_over_effect",
        ),
        PixelExpectation::opaque(
            origin_x as u32 + 55,
            origin_y as u32 + 35,
            0,
            0,
            255,
            "t62_group_contains_shape_effect_once",
        ),
    ]
}

/// Backdrop capture must include the target node's shape effect.
fn tile_63_cached_shape_effect_with_backdrop(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (origin_x, origin_y) = tile_origin(63);
    let backdrop_source = Shape::rect(
        [
            (origin_x + 5.0, origin_y + 5.0),
            (origin_x + 75.0, origin_y + 70.0),
        ],
        Stroke::default(),
    );
    renderer
        .add_shape(
            backdrop_source,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(220, 200, 50)),
        )
        .unwrap();

    let panel = Shape::rect(
        [
            (origin_x + 20.0, origin_y + 20.0),
            (origin_x + 50.0, origin_y + 50.0),
        ],
        Stroke::default(),
    );
    let panel_id = renderer
        .add_shape(panel, None, Some(63_063), ShapeDrawCommandOptions::new())
        .unwrap();
    renderer
        .set_shape_effect(
            panel_id,
            SHAPE_DROP_EFFECT_ID,
            &[],
            ShapeEffectConfig::new().outset(12.0),
        )
        .unwrap();
    renderer
        .set_shape_backdrop_effect(
            panel_id,
            PASSTHROUGH_EFFECT_ID,
            &[],
            BackdropEffectConfig::default(),
        )
        .unwrap();

    vec![
        PixelExpectation::opaque(
            origin_x as u32 + 35,
            origin_y as u32 + 35,
            0,
            0,
            255,
            "t63_backdrop_includes_shape_effect",
        ),
        PixelExpectation::opaque(
            origin_x as u32 + 55,
            origin_y as u32 + 35,
            0,
            0,
            255,
            "t63_shape_effect_draws_behind_backdrop_node",
        ),
    ]
}

fn tile_64_drop_shadow_with_backdrop_blur(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (origin_x, origin_y) = tile_origin(64);
    let backing_shape = Shape::rect(
        [
            (origin_x + 6.0, origin_y + 6.0),
            (origin_x + 72.0, origin_y + 42.0),
        ],
        Stroke::default(),
    );
    renderer
        .add_shape(
            backing_shape,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(245, 190, 70)),
        )
        .unwrap();

    let card = Shape::rounded_rect(
        [
            (origin_x + 18.0, origin_y + 14.0),
            (origin_x + 56.0, origin_y + 52.0),
        ],
        BorderRadii::new(8.0),
        Stroke::default(),
    );
    let card_id = renderer
        .add_shape(
            card,
            None,
            Some(64_064),
            ShapeDrawCommandOptions::new().color(Color::rgba(75, 125, 235, 150)),
        )
        .unwrap();
    renderer
        .set_shape_effect(
            card_id,
            DROP_SHADOW_EFFECT_ID,
            &[],
            ShapeEffectConfig::new().outsets(8.0, 8.0, 20.0, 22.0),
        )
        .expect("Failed to attach visual drop shadow effect");
    let backdrop_blur_params = BlurParams {
        radius: 5.0,
        _pad: 0.0,
    };
    renderer
        .set_shape_backdrop_effect(
            card_id,
            BLUR_EFFECT_ID,
            bytemuck::bytes_of(&backdrop_blur_params),
            BackdropEffectConfig::default(),
        )
        .expect("Failed to attach backdrop blur to shadowed card");

    vec![
        PixelExpectation::opaque(
            origin_x as u32 + 10,
            origin_y as u32 + 20,
            245,
            190,
            70,
            "t64_backing_shape_is_visible",
        ),
        PixelExpectation::opaque_approx(
            origin_x as u32 + 62,
            origin_y as u32 + 35,
            191,
            148,
            53,
            8,
            "t64_backing_shape_visible_through_shadow",
        ),
        PixelExpectation::opaque_approx(
            origin_x as u32 + 35,
            origin_y as u32 + 30,
            115,
            122,
            187,
            12,
            "t64_translucent_card_tints_blurred_backdrop",
        ),
        PixelExpectation::opaque_approx(
            origin_x as u32 + 35,
            origin_y as u32 + 41,
            117,
            129,
            195,
            12,
            "t64_backdrop_edge_blurs_inside_card",
        ),
        PixelExpectation::opaque(
            origin_x as u32 + 10,
            origin_y as u32 + 41,
            245,
            190,
            70,
            "t64_backdrop_edge_stays_sharp_outside_card",
        ),
        PixelExpectation::opaque(
            origin_x as u32 + 10,
            origin_y as u32 + 44,
            255,
            255,
            255,
            "t64_below_backdrop_edge_stays_white_outside_card",
        ),
        PixelExpectation::opaque_approx(
            origin_x as u32 + 62,
            origin_y as u32 + 57,
            234,
            234,
            234,
            10,
            "t64_shadow_has_soft_diagonal_falloff",
        ),
        PixelExpectation::opaque_approx(
            origin_x as u32 + 42,
            origin_y as u32 + 57,
            170,
            170,
            170,
            10,
            "t64_shadow_body_visible_below_card",
        ),
        PixelExpectation::opaque(
            origin_x as u32 + 75,
            origin_y as u32 + 10,
            255,
            255,
            255,
            "t64_shadow_outsets_remain_transparent",
        ),
    ]
}

/// A grouped backdrop must capture the shape effect already drawn in its subtree.
fn tile_65_grouped_shape_effect_in_backdrop(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (origin_x, origin_y) = tile_origin(65);
    let backing_shape = Shape::rect(
        [
            (origin_x + 5.0, origin_y + 5.0),
            (origin_x + 75.0, origin_y + 70.0),
        ],
        Stroke::default(),
    );
    renderer
        .add_shape(
            backing_shape,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(220, 200, 50)),
        )
        .unwrap();

    let group_root = Shape::rect(
        [
            (origin_x + 8.0, origin_y + 8.0),
            (origin_x + 72.0, origin_y + 65.0),
        ],
        Stroke::default(),
    );
    let group_root_id = renderer
        .add_shape(group_root, None, None, ShapeDrawCommandOptions::new())
        .unwrap();

    let panel = Shape::rect(
        [
            (origin_x + 20.0, origin_y + 20.0),
            (origin_x + 50.0, origin_y + 50.0),
        ],
        Stroke::default(),
    );
    let panel_id = renderer
        .add_shape(
            panel,
            Some(group_root_id),
            Some(65_065),
            ShapeDrawCommandOptions::new(),
        )
        .unwrap();
    renderer
        .set_shape_effect(
            panel_id,
            SHAPE_DROP_EFFECT_ID,
            &[],
            ShapeEffectConfig::new().outset(12.0),
        )
        .expect("Failed to attach grouped cached shape effect");
    renderer
        .set_shape_backdrop_effect(
            panel_id,
            PASSTHROUGH_EFFECT_ID,
            &[],
            BackdropEffectConfig::default(),
        )
        .expect("Failed to attach grouped backdrop effect");
    renderer
        .set_group_effect(group_root_id, PASSTHROUGH_EFFECT_ID, &[])
        .expect("Failed to attach enclosing group effect");

    vec![
        PixelExpectation::opaque(
            origin_x as u32 + 35,
            origin_y as u32 + 35,
            0,
            0,
            255,
            "t65_grouped_backdrop_includes_shape_effect",
        ),
        PixelExpectation::opaque(
            origin_x as u32 + 55,
            origin_y as u32 + 35,
            0,
            0,
            255,
            "t65_grouped_shape_effect_draws_outside_panel",
        ),
        PixelExpectation::opaque(
            origin_x as u32 + 10,
            origin_y as u32 + 10,
            220,
            200,
            50,
            "t65_scene_behind_group_remains_visible",
        ),
    ]
}

/// Shape, backdrop, and group effects on one node must preserve painter order.
fn tile_66_same_node_shape_backdrop_and_group_effects(
    renderer: &mut Renderer,
) -> Vec<PixelExpectation> {
    let (origin_x, origin_y) = tile_origin(66);
    let backing_shape = Shape::rect(
        [
            (origin_x + 5.0, origin_y + 5.0),
            (origin_x + 75.0, origin_y + 70.0),
        ],
        Stroke::default(),
    );
    renderer
        .add_shape(
            backing_shape,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(50, 180, 80)),
        )
        .unwrap();

    let panel = Shape::rect(
        [
            (origin_x + 20.0, origin_y + 20.0),
            (origin_x + 50.0, origin_y + 50.0),
        ],
        Stroke::default(),
    );
    let panel_id = renderer
        .add_shape(panel, None, Some(66_066), ShapeDrawCommandOptions::new())
        .unwrap();
    renderer
        .set_shape_effect(
            panel_id,
            SHAPE_DROP_EFFECT_ID,
            &[],
            ShapeEffectConfig::new().outset(12.0),
        )
        .expect("Failed to attach same-node cached shape effect");
    renderer
        .set_shape_backdrop_effect(
            panel_id,
            PASSTHROUGH_EFFECT_ID,
            &[],
            BackdropEffectConfig::default(),
        )
        .expect("Failed to attach same-node backdrop effect");
    renderer
        .set_group_effect(panel_id, PASSTHROUGH_EFFECT_ID, &[])
        .expect("Failed to attach same-node group effect");

    vec![
        PixelExpectation::opaque(
            origin_x as u32 + 35,
            origin_y as u32 + 35,
            0,
            0,
            255,
            "t66_grouped_backdrop_includes_same_node_shape_effect",
        ),
        PixelExpectation::opaque(
            origin_x as u32 + 55,
            origin_y as u32 + 35,
            0,
            0,
            255,
            "t66_group_result_contains_shape_effect_outside_panel",
        ),
        PixelExpectation::opaque(
            origin_x as u32 + 10,
            origin_y as u32 + 10,
            50,
            180,
            80,
            "t66_scene_behind_group_remains_unprocessed",
        ),
    ]
}

/// Halving the mask resolution, shader sigma, and offset must preserve the shadow size.
/// Bilinear upscaling slightly softens its edges.
fn tile_67_downsampled_drop_shadow_with_backdrop_blur(
    renderer: &mut Renderer,
) -> Vec<PixelExpectation> {
    let (origin_x, origin_y) = tile_origin(67);
    let backing_shape = Shape::rect(
        [
            (origin_x + 6.0, origin_y + 6.0),
            (origin_x + 72.0, origin_y + 42.0),
        ],
        Stroke::default(),
    );
    renderer
        .add_shape(
            backing_shape,
            None,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(245, 190, 70)),
        )
        .unwrap();

    let card = Shape::rounded_rect(
        [
            (origin_x + 18.0, origin_y + 14.0),
            (origin_x + 56.0, origin_y + 52.0),
        ],
        BorderRadii::new(8.0),
        Stroke::default(),
    );
    let card_id = renderer
        .add_shape(
            card,
            None,
            Some(67_067),
            ShapeDrawCommandOptions::new().color(Color::rgba(75, 125, 235, 150)),
        )
        .unwrap();
    renderer
        .set_shape_effect(
            card_id,
            DOWNSAMPLED_DROP_SHADOW_EFFECT_ID,
            &[],
            ShapeEffectConfig::new()
                .outsets(8.0, 8.0, 20.0, 22.0)
                .downsample(0.5),
        )
        .expect("Failed to attach downsampled drop shadow effect");
    let backdrop_blur_params = BlurParams {
        radius: 5.0,
        _pad: 0.0,
    };
    renderer
        .set_shape_backdrop_effect(
            card_id,
            BLUR_EFFECT_ID,
            bytemuck::bytes_of(&backdrop_blur_params),
            BackdropEffectConfig::default(),
        )
        .expect("Failed to attach backdrop blur to downsampled shadow card");

    vec![
        PixelExpectation::opaque(
            origin_x as u32 + 10,
            origin_y as u32 + 20,
            245,
            190,
            70,
            "t67_backing_shape_is_visible",
        ),
        PixelExpectation::opaque_approx(
            origin_x as u32 + 62,
            origin_y as u32 + 35,
            191,
            148,
            53,
            12,
            "t67_backing_shape_visible_through_shadow",
        ),
        PixelExpectation::opaque_approx(
            origin_x as u32 + 35,
            origin_y as u32 + 30,
            115,
            122,
            187,
            12,
            "t67_translucent_card_tints_blurred_backdrop",
        ),
        PixelExpectation::opaque_approx(
            origin_x as u32 + 35,
            origin_y as u32 + 41,
            117,
            129,
            195,
            12,
            "t67_backdrop_edge_blurs_inside_card",
        ),
        PixelExpectation::opaque(
            origin_x as u32 + 10,
            origin_y as u32 + 41,
            245,
            190,
            70,
            "t67_backdrop_edge_stays_sharp_outside_card",
        ),
        PixelExpectation::opaque(
            origin_x as u32 + 10,
            origin_y as u32 + 44,
            255,
            255,
            255,
            "t67_below_backdrop_edge_stays_white_outside_card",
        ),
        PixelExpectation::opaque_approx(
            origin_x as u32 + 62,
            origin_y as u32 + 57,
            234,
            234,
            234,
            12,
            "t67_shadow_has_soft_diagonal_falloff",
        ),
        PixelExpectation::opaque_approx(
            origin_x as u32 + 42,
            origin_y as u32 + 57,
            170,
            170,
            170,
            12,
            "t67_shadow_body_visible_below_card",
        ),
        PixelExpectation::opaque(
            origin_x as u32 + 75,
            origin_y as u32 + 10,
            255,
            255,
            255,
            "t67_shadow_outsets_remain_transparent",
        ),
    ]
}
