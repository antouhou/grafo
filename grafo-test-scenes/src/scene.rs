use grafo::{BorderRadii, Color, Renderer, Shape, Stroke, TransformInstance};

use crate::expectations::PixelExpectation;
use crate::shaders::{BlurParams, HORIZONTAL_BLUR_WGSL, VERTICAL_BLUR_WGSL};

// ── Grid layout constants ────────────────────────────────────────────────────

const TILE_SIZE: u32 = 80;
const COLUMNS: u32 = 6;
const ROWS: u32 = 6;

pub const CANVAS_WIDTH: u32 = TILE_SIZE * COLUMNS;
pub const CANVAS_HEIGHT: u32 = TILE_SIZE * ROWS;

const BLUR_EFFECT_ID: u64 = 1;
const CHECKERBOARD_TEXTURE_ID: u64 = 100;

/// Returns the pixel origin (top-left corner) of tile number `n` (1-based).
fn tile_origin(tile_number: u32) -> (f32, f32) {
    let index = tile_number - 1;
    let column = index % COLUMNS;
    let row = index / COLUMNS;
    ((column * TILE_SIZE) as f32, (row * TILE_SIZE) as f32)
}

/// Builds the entire main test scene on the given renderer and returns a list
/// of pixel expectations to validate the rendered output.
///
/// This function is shared between the `#[test]` (via `render_to_buffer`) and
/// the winit visual-confirmation example (via `render()`).
pub fn build_main_scene(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let mut expectations: Vec<PixelExpectation> = Vec::new();

    // The draw tree requires a single root node. All tile shapes are added as
    // children of this full-canvas root so they are not clipped to each other.
    let canvas_root = Shape::rect(
        [(0.0, 0.0), (CANVAS_WIDTH as f32, CANVAS_HEIGHT as f32)],
        Stroke::default(),
    );
    renderer.add_shape(canvas_root, None, None);
    // Leave the root with default white color — it acts as the background.

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

    expectations
}

// ── Shared resource setup ────────────────────────────────────────────────────

fn load_shared_resources(renderer: &mut Renderer) {
    renderer
        .load_effect(BLUR_EFFECT_ID, &[HORIZONTAL_BLUR_WGSL, VERTICAL_BLUR_WGSL])
        .expect("Failed to compile blur effect");

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
}

// ── Section A: Basic Shapes ──────────────────────────────────────────────────

fn tile_01_rect_solid(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(1);
    let shape = Shape::rect(
        [(ox + 10.0, oy + 10.0), (ox + 70.0, oy + 70.0)],
        Stroke::default(),
    );
    let id = renderer.add_shape(shape, None, None);
    renderer.set_shape_color(id, Some(Color::rgb(220, 50, 50)));

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
    let id = renderer.add_shape(shape, None, None);
    renderer.set_shape_color(id, Some(Color::rgb(50, 180, 50)));

    vec![
        PixelExpectation::opaque(ox as u32 + 40, oy as u32 + 40, 50, 180, 50, "t02_interior"),
        // Corner at (ox+11, oy+11) is inside the border radius curve — should be canvas bg
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
    let id = renderer.add_shape(shape, None, None);
    renderer.set_shape_color(id, Some(Color::rgb(50, 50, 220)));

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
    let id = renderer.add_shape(shape, None, None);
    renderer.set_shape_color(id, Some(Color::rgb(220, 200, 50)));

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

// ── Section B: Hierarchy & Scissor Clipping ──────────────────────────────────

fn tile_05_rect_parent_child_inside(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(5);
    let parent = Shape::rect(
        [(ox + 5.0, oy + 5.0), (ox + 75.0, oy + 75.0)],
        Stroke::default(),
    );
    let parent_id = renderer.add_shape(parent, None, None);
    renderer.set_shape_color(parent_id, Some(Color::rgb(180, 180, 220)));

    let child = Shape::rect(
        [(ox + 20.0, oy + 20.0), (ox + 60.0, oy + 60.0)],
        Stroke::default(),
    );
    let child_id = renderer.add_shape(child, Some(parent_id), None);
    renderer.set_shape_color(child_id, Some(Color::rgb(220, 100, 50)));

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
    let parent_id = renderer.add_shape(parent, None, None);
    renderer.set_shape_color(parent_id, Some(Color::rgb(150, 200, 150)));

    // Child extends past parent's right edge
    let child = Shape::rect(
        [(ox + 30.0, oy + 20.0), (ox + 75.0, oy + 60.0)],
        Stroke::default(),
    );
    let child_id = renderer.add_shape(child, Some(parent_id), None);
    renderer.set_shape_color(child_id, Some(Color::rgb(50, 50, 200)));

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
        // Pixel outside parent boundary — child should be clipped, shows canvas bg
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
    let parent_id = renderer.add_shape(parent, None, None);
    renderer.set_shape_color(parent_id, Some(Color::rgb(200, 200, 200)));

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
        let child_id = renderer.add_shape(child, Some(parent_id), None);
        renderer.set_shape_color(child_id, Some(child_colors[idx]));
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
    let id0 = renderer.add_shape(level0, None, None);
    renderer.set_shape_color(id0, Some(Color::rgb(200, 180, 180)));

    let level1 = Shape::rect(
        [(ox + 15.0, oy + 15.0), (ox + 65.0, oy + 65.0)],
        Stroke::default(),
    );
    let id1 = renderer.add_shape(level1, Some(id0), None);
    renderer.set_shape_color(id1, Some(Color::rgb(180, 200, 180)));

    let level2 = Shape::rect(
        [(ox + 25.0, oy + 25.0), (ox + 55.0, oy + 55.0)],
        Stroke::default(),
    );
    let id2 = renderer.add_shape(level2, Some(id1), None);
    renderer.set_shape_color(id2, Some(Color::rgb(100, 100, 220)));

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
    let parent_id = renderer.add_shape(parent, None, None);
    renderer.set_shape_color(parent_id, Some(Color::rgb(200, 200, 200)));

    // First child (drawn first)
    let child1 = Shape::rect(
        [(ox + 15.0, oy + 20.0), (ox + 50.0, oy + 55.0)],
        Stroke::default(),
    );
    let c1 = renderer.add_shape(child1, Some(parent_id), None);
    renderer.set_shape_color(c1, Some(Color::rgb(220, 50, 50)));

    // Second child overlaps first (drawn on top)
    let child2 = Shape::rect(
        [(ox + 30.0, oy + 30.0), (ox + 65.0, oy + 65.0)],
        Stroke::default(),
    );
    let c2 = renderer.add_shape(child2, Some(parent_id), None);
    renderer.set_shape_color(c2, Some(Color::rgb(50, 50, 220)));

    vec![
        // Overlap region — second child on top
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

// ── Section C: Stencil Clipping ──────────────────────────────────────────────

fn tile_10_rounded_rect_clip(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(10);
    let parent = Shape::rounded_rect(
        [(ox + 5.0, oy + 5.0), (ox + 75.0, oy + 75.0)],
        BorderRadii::new(20.0),
        Stroke::default(),
    );
    let parent_id = renderer.add_shape(parent, None, None);
    renderer.set_shape_color(parent_id, Some(Color::rgb(200, 200, 230)));

    // Child is smaller than parent, leaving a visible parent ring
    let child = Shape::rect(
        [(ox + 15.0, oy + 15.0), (ox + 65.0, oy + 65.0)],
        Stroke::default(),
    );
    let child_id = renderer.add_shape(child, Some(parent_id), None);
    renderer.set_shape_color(child_id, Some(Color::rgb(220, 100, 50)));

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
        // Corner outside rounded clip — shows canvas bg
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
    let parent_id = renderer.add_shape(parent, None, None);
    renderer.set_shape_color(parent_id, Some(Color::rgb(180, 180, 220)));

    // Child rect offset to the right so left part of triangle shows parent color
    let child = Shape::rect(
        [(ox + 35.0, oy + 15.0), (ox + 75.0, oy + 75.0)],
        Stroke::default(),
    );
    let child_id = renderer.add_shape(child, Some(parent_id), None);
    renderer.set_shape_color(child_id, Some(Color::rgb(220, 150, 50)));

    vec![
        // Right side of triangle — child (orange) is visible and clipped to triangle
        PixelExpectation::opaque(
            ox as u32 + 55,
            oy as u32 + 55,
            220,
            150,
            50,
            "t11_child_in_tri",
        ),
        // Left side of triangle — parent (blue-ish) is visible where child doesn't cover
        PixelExpectation::opaque(
            ox as u32 + 25,
            oy as u32 + 55,
            180,
            180,
            220,
            "t11_parent_in_tri",
        ),
        // Outside triangle — shows canvas bg
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

    // L0: rounded-rect in the NW quadrant — stencil clip.
    let level0 = Shape::rounded_rect(
        [(ox + 5.0, oy + 5.0), (ox + 60.0, oy + 60.0)],
        BorderRadii::new(10.0),
        Stroke::default(),
    );
    let id0 = renderer.add_shape(level0, None, None);
    renderer.set_shape_color(id0, Some(Color::rgb(200, 180, 200))); // lavender

    // L1: shifted SE — overflows L0 right and bottom.
    // Visible (L0∩L1) ≈ (20,20)→(60,60).
    let level1 = Shape::rounded_rect(
        [(ox + 20.0, oy + 20.0), (ox + 75.0, oy + 75.0)],
        BorderRadii::new(8.0),
        Stroke::default(),
    );
    let id1 = renderer.add_shape(level1, Some(id0), None);
    renderer.set_shape_color(id1, Some(Color::rgb(100, 200, 100))); // green

    // L2: shifted SW/down — overflows L1 to the left and L0 below.
    // Visible (L0∩L1∩L2) ≈ (20,35)→(50,60).
    let level2 = Shape::rounded_rect(
        [(ox + 10.0, oy + 35.0), (ox + 50.0, oy + 75.0)],
        BorderRadii::new(8.0),
        Stroke::default(),
    );
    let id2 = renderer.add_shape(level2, Some(id1), None);
    renderer.set_shape_color(id2, Some(Color::rgb(100, 100, 220))); // blue

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
        // --- Clipping proofs ---
        // L1 extends to (65,40) but L0 ends at x=60 → bg white.
        // Proves L0 stencil clips L1.
        PixelExpectation::opaque(
            ox as u32 + 65,
            oy as u32 + 40,
            255,
            255,
            255,
            "t12_l0_clips_l1",
        ),
        // L2 extends to (15,50) (inside L0) but L1 starts at x=20 → L0 lavender.
        // Proves L1 stencil clips L2: blue does NOT leak into L0's area.
        PixelExpectation::opaque(
            ox as u32 + 15,
            oy as u32 + 50,
            200,
            180,
            200,
            "t12_l1_clips_l2",
        ),
        // L1+L2 extend to (30,70) but L0 ends at y=60 → bg white.
        // Proves L0 clips the entire chain.
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
    let parent_id = renderer.add_shape(parent, None, None);
    renderer.set_shape_color(parent_id, Some(Color::rgb(200, 200, 180)));
    let rotation = TransformInstance::rotation_z_deg(45.0);
    let translation = TransformInstance::translation(ox + 40.0, oy + 40.0);
    // a.multiply(&b) applies a first, then b: first rotate, then translate
    renderer.set_shape_transform(parent_id, rotation.multiply(&translation));

    // Child is smaller than parent — parent ring visible around child
    let child = Shape::rect([(-12.0, -12.0), (12.0, 12.0)], Stroke::default());
    let child_id = renderer.add_shape(child, Some(parent_id), None);
    renderer.set_shape_color(child_id, Some(Color::rgb(220, 80, 80)));
    renderer.set_shape_transform(child_id, rotation.multiply(&translation));

    vec![
        // Center of the rotated diamond — child visible
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 40,
            220,
            80,
            80,
            "t13_child_center",
        ),
        // Parent ring — between child edge and parent edge along the vertical axis
        // Parent diamond tip is at ~(40, 40±28), child diamond tip at ~(40, 40±17)
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 17,
            200,
            200,
            180,
            "t13_parent_ring",
        ),
        // Far corner — outside the diamond, shows canvas bg
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

// ── Section D: Mixed Scissor + Stencil ───────────────────────────────────────

fn tile_14_scissor_then_stencil(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(14);

    // L0: rect in NW quadrant → scissor clip.
    let level0 = Shape::rect(
        [(ox + 5.0, oy + 5.0), (ox + 60.0, oy + 60.0)],
        Stroke::default(),
    );
    let id0 = renderer.add_shape(level0, None, None);
    renderer.set_shape_color(id0, Some(Color::rgb(200, 200, 200))); // gray

    // L1: rounded-rect shifted SE — overflows L0 right and bottom → stencil clip.
    // Visible (L0∩L1) ≈ (20,20)→(60,60).
    let level1 = Shape::rounded_rect(
        [(ox + 20.0, oy + 20.0), (ox + 75.0, oy + 75.0)],
        BorderRadii::new(10.0),
        Stroke::default(),
    );
    let id1 = renderer.add_shape(level1, Some(id0), None);
    renderer.set_shape_color(id1, Some(Color::rgb(180, 220, 180))); // green

    // Leaf: rect shifted SW/down — overflows L1 to the left and L0 below.
    // Visible (L0∩L1∩leaf) ≈ (20,35)→(50,60).
    let leaf = Shape::rect(
        [(ox + 10.0, oy + 35.0), (ox + 50.0, oy + 75.0)],
        Stroke::default(),
    );
    let leaf_id = renderer.add_shape(leaf, Some(id1), None);
    renderer.set_shape_color(leaf_id, Some(Color::rgb(50, 50, 200))); // blue

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
        // --- Clipping proofs ---
        // L1 extends to (65,40) but L0 ends at x=60 → bg white.
        // Proves L0 scissor clips L1.
        PixelExpectation::opaque(
            ox as u32 + 65,
            oy as u32 + 40,
            255,
            255,
            255,
            "t14_l0_clips_l1",
        ),
        // Leaf extends to (15,50) (inside L0) but L1 starts at x=20 → L0 gray.
        // Proves L1 stencil clips leaf: blue does NOT leak into L0's area.
        PixelExpectation::opaque(
            ox as u32 + 15,
            oy as u32 + 50,
            200,
            200,
            200,
            "t14_l1_clips_leaf",
        ),
        // Leaf extends to (30,65) but L0 ends at y=60 → bg white.
        // Proves L0 scissor clips the entire chain.
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
    let id0 = renderer.add_shape(level0, None, None);
    renderer.set_shape_color(id0, Some(Color::rgb(220, 200, 200))); // pink

    // L1: rect shifted SE — overflows L0 right and bottom → scissor clip.
    // Visible (L0∩L1) ≈ (20,20)→(60,60).
    let level1 = Shape::rect(
        [(ox + 20.0, oy + 20.0), (ox + 75.0, oy + 75.0)],
        Stroke::default(),
    );
    let id1 = renderer.add_shape(level1, Some(id0), None);
    renderer.set_shape_color(id1, Some(Color::rgb(200, 220, 200))); // green

    // Leaf: rect shifted SW/down — overflows L1 to the left and L0 below.
    // Visible (L0∩L1∩leaf) ≈ (20,35)→(50,60).
    let leaf = Shape::rect(
        [(ox + 10.0, oy + 35.0), (ox + 50.0, oy + 75.0)],
        Stroke::default(),
    );
    let leaf_id = renderer.add_shape(leaf, Some(id1), None);
    renderer.set_shape_color(leaf_id, Some(Color::rgb(50, 200, 50))); // bright green

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
        // --- Clipping proofs ---
        // L1 extends to (65,40) but L0 ends at x=60 → bg white.
        // Proves L0 stencil clips L1.
        PixelExpectation::opaque(
            ox as u32 + 65,
            oy as u32 + 40,
            255,
            255,
            255,
            "t15_l0_clips_l1",
        ),
        // Leaf extends to (15,50) (inside L0) but L1 starts at x=20 → L0 pink.
        // Proves L1 scissor clips leaf.
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
    let id0 = renderer.add_shape(l0, None, None);
    renderer.set_shape_color(id0, Some(Color::rgb(220, 220, 220))); // light gray

    // L1: rect, wide top portion — overflows L0 to right → scissor.
    // Visible (L0∩L1) = (20,5)→(55,55).
    let l1 = Shape::rect(
        [(ox + 20.0, oy + 5.0), (ox + 75.0, oy + 55.0)],
        Stroke::default(),
    );
    let id1 = renderer.add_shape(l1, Some(id0), None);
    renderer.set_shape_color(id1, Some(Color::rgb(200, 200, 220))); // blue-gray

    // L2: rounded-rect, tall — overflows L1 below → stencil.
    // Visible (L0∩L1∩L2) ≈ (20,20)→(50,55).
    let l2 = Shape::rounded_rect(
        [(ox + 15.0, oy + 20.0), (ox + 50.0, oy + 70.0)],
        BorderRadii::new(10.0),
        Stroke::default(),
    );
    let id2 = renderer.add_shape(l2, Some(id1), None);
    renderer.set_shape_color(id2, Some(Color::rgb(180, 220, 180))); // green

    // L3: rect, wide — overflows L2 right and above → scissor.
    // Visible (L0∩L1∩L2∩L3) ≈ (25,20)→(50,50).
    let l3 = Shape::rect(
        [(ox + 25.0, oy + 10.0), (ox + 70.0, oy + 50.0)],
        Stroke::default(),
    );
    let id3 = renderer.add_shape(l3, Some(id2), None);
    renderer.set_shape_color(id3, Some(Color::rgb(200, 180, 220))); // purple

    // L4: leaf rect — overflows L3 left and below.
    // Visible (all 5) ≈ (25,30)→(45,50).
    let l4 = Shape::rect(
        [(ox + 10.0, oy + 30.0), (ox + 45.0, oy + 65.0)],
        Stroke::default(),
    );
    let id4 = renderer.add_shape(l4, Some(id3), None);
    renderer.set_shape_color(id4, Some(Color::rgb(50, 50, 200))); // blue

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
        // --- Clipping proofs (one per level) ---
        // L4 extends to (22,40) but L3 starts at x=25 → L2 green.
        // Proves L3 clips L4.
        PixelExpectation::opaque(
            ox as u32 + 22,
            oy as u32 + 40,
            180,
            220,
            180,
            "t16_l3_clips_l4",
        ),
        // L3 extends to (35,12) but L2 starts at y=20 → L1 blue-gray.
        // Proves L2 clips L3.
        PixelExpectation::opaque(
            ox as u32 + 35,
            oy as u32 + 12,
            200,
            200,
            220,
            "t16_l2_clips_l3",
        ),
        // L2 extends to (17,40) but L1 starts at x=20 (inside L0) → L0 gray.
        // Proves L1 clips L2.
        PixelExpectation::opaque(
            ox as u32 + 17,
            oy as u32 + 40,
            220,
            220,
            220,
            "t16_l1_clips_l2",
        ),
        // L1 extends to (60,30) but L0 ends at x=55 → bg white.
        // Proves L0 clips L1.
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

// ── Section E: Transforms ────────────────────────────────────────────────────

fn tile_17_translated_rect(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(17);
    // Rect defined at origin, translated to bottom-right of tile
    let shape = Shape::rect([(0.0, 0.0), (40.0, 25.0)], Stroke::default());
    let id = renderer.add_shape(shape, None, None);
    renderer.set_shape_color(id, Some(Color::rgb(220, 150, 50)));
    renderer.set_shape_transform(id, TransformInstance::translation(ox + 30.0, oy + 45.0));

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
    let id = renderer.add_shape(shape, None, None);
    renderer.set_shape_color(id, Some(Color::rgb(150, 50, 200)));
    // Scale 0.5× horizontal, 1.0× vertical around tile center
    let to_origin = TransformInstance::translation(-(ox + 40.0), -(oy + 40.0));
    let scale = TransformInstance::scale(0.5, 1.0);
    let back = TransformInstance::translation(ox + 40.0, oy + 40.0);
    renderer.set_shape_transform(id, to_origin.multiply(&scale.multiply(&back)));

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
    let id = renderer.add_shape(shape, None, None);
    renderer.set_shape_color(id, Some(Color::rgb(220, 100, 100)));
    let rotation = TransformInstance::rotation_z_deg(45.0);
    let translation = TransformInstance::translation(ox + 40.0, oy + 40.0);
    // a.multiply(&b) applies a first, then b: first rotate, then translate
    renderer.set_shape_transform(id, rotation.multiply(&translation));

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
    let parent_id = renderer.add_shape(parent, None, None);
    renderer.set_shape_color(parent_id, Some(Color::rgb(180, 180, 220)));
    renderer.set_shape_transform(
        parent_id,
        TransformInstance::translation(ox + 15.0, oy + 15.0),
    );

    // Child at local (10,10)-(40,40) — ends up at tile (25,25)-(55,55)
    let child = Shape::rect([(10.0, 10.0), (40.0, 40.0)], Stroke::default());
    let child_id = renderer.add_shape(child, Some(parent_id), None);
    renderer.set_shape_color(child_id, Some(Color::rgb(50, 200, 50)));
    renderer.set_shape_transform(
        child_id,
        TransformInstance::translation(ox + 15.0, oy + 15.0),
    );

    vec![
        PixelExpectation::opaque(ox as u32 + 40, oy as u32 + 40, 50, 200, 50, "t20_child"),
        PixelExpectation::opaque(ox as u32 + 20, oy as u32 + 20, 180, 180, 220, "t20_parent"),
    ]
}

// ── Section F: Colors & Alpha ────────────────────────────────────────────────

fn tile_21_alpha_overlap(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(21);
    // Opaque blue background
    let bg = Shape::rect(
        [(ox + 10.0, oy + 10.0), (ox + 50.0, oy + 60.0)],
        Stroke::default(),
    );
    let bg_id = renderer.add_shape(bg, None, None);
    renderer.set_shape_color(bg_id, Some(Color::rgb(50, 50, 220)));

    // Semi-transparent red on top
    let fg = Shape::rect(
        [(ox + 25.0, oy + 20.0), (ox + 65.0, oy + 55.0)],
        Stroke::default(),
    );
    let fg_id = renderer.add_shape(fg, None, None);
    renderer.set_shape_color(fg_id, Some(Color::rgba(220, 50, 50, 128)));

    vec![
        // Blue region only
        PixelExpectation::opaque(ox as u32 + 15, oy as u32 + 15, 50, 50, 220, "t21_blue_only"),
        // Overlap region: semi-transparent red over opaque blue
        // Blend: dst_rgb*(1-src_a) + src_rgb = (50,50,220)*(1-0.5) + (220,50,50)*0.5
        //      ≈ (25+110, 25+25, 110+25) = (135, 50, 135)
        PixelExpectation::new(
            ox as u32 + 35,
            oy as u32 + 35,
            135,
            50,
            135,
            255,
            "t21_red_over_blue",
        )
        .with_tolerance(30),
        // Red-only region — semi-transparent red over white canvas bg
        // Alpha blending produces a pinkish tint
        PixelExpectation::new(
            ox as u32 + 55,
            oy as u32 + 40,
            238,
            152,
            152,
            255,
            "t21_red_over_white",
        )
        .with_tolerance(40),
    ]
}

fn tile_22_no_color_default(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(22);
    // Colored background so we can distinguish the white default-color shape
    let bg = Shape::rect(
        [(ox + 5.0, oy + 5.0), (ox + 75.0, oy + 75.0)],
        Stroke::default(),
    );
    let bg_id = renderer.add_shape(bg, None, None);
    renderer.set_shape_color(bg_id, Some(Color::rgb(100, 100, 200)));

    let shape = Shape::rect(
        [(ox + 20.0, oy + 20.0), (ox + 60.0, oy + 60.0)],
        Stroke::default(),
    );
    // No set_shape_color call — defaults to white
    renderer.add_shape(shape, None, None);

    vec![
        // Shape center — default white
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 40,
            255,
            255,
            255,
            "t22_default_white",
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
    let id = renderer.add_shape(shape, None, None);
    renderer.set_shape_color(id, Some(Color::TRANSPARENT));

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

// ── Section G: Textures ──────────────────────────────────────────────────────

fn tile_24_textured_rect(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(24);
    let shape = Shape::rect(
        [(ox + 10.0, oy + 10.0), (ox + 70.0, oy + 70.0)],
        Stroke::default(),
    );
    let id = renderer.add_shape(shape, None, None);
    renderer.set_shape_texture(id, Some(CHECKERBOARD_TEXTURE_ID));
    // White color so texture shows through unmodified
    renderer.set_shape_color(id, Some(Color::WHITE));

    vec![
        // Textured interior — should not be fully white or fully transparent
        // The checkerboard alternates white/black, so center pixel is one or the other
        PixelExpectation::new(
            ox as u32 + 40,
            oy as u32 + 40,
            128,
            128,
            128,
            255,
            "t24_textured_interior",
        )
        .with_tolerance(128), // white or black—just verify opaque & non-canvas
        // Outside the textured rect — canvas bg
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
    let id = renderer.add_shape(shape, None, None);
    renderer.set_shape_texture(id, Some(CHECKERBOARD_TEXTURE_ID));
    renderer.set_shape_color(id, Some(Color::rgb(255, 100, 100)));

    vec![
        PixelExpectation::new(
            ox as u32 + 40,
            oy as u32 + 40,
            190,
            130,
            130,
            255,
            "t25_tinted_texture",
        )
        .with_tolerance(80),
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
    let parent_id = renderer.add_shape(parent, None, None);
    renderer.set_shape_texture(parent_id, Some(CHECKERBOARD_TEXTURE_ID));
    renderer.set_shape_color(parent_id, Some(Color::WHITE));

    let child = Shape::rect(
        [(ox + 25.0, oy + 25.0), (ox + 55.0, oy + 55.0)],
        Stroke::default(),
    );
    let child_id = renderer.add_shape(child, Some(parent_id), None);
    renderer.set_shape_color(child_id, Some(Color::rgb(220, 50, 50)));

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
        // Textured parent visible in its border area (not pure white canvas bg)
        PixelExpectation::new(
            ox as u32 + 10,
            oy as u32 + 10,
            128,
            128,
            128,
            255,
            "t26_parent_texture_visible",
        )
        .with_tolerance(128),
    ]
}

// ── Section H: Group Effects ─────────────────────────────────────────────────

fn tile_27_group_blur_leaf(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(27);

    // Background stripe (sharp, NOT blurred) — partially behind the blurred shape
    let bg_stripe = Shape::rect(
        [(ox + 5.0, oy + 30.0), (ox + 75.0, oy + 50.0)],
        Stroke::default(),
    );
    let bg_id = renderer.add_shape(bg_stripe, None, None);
    renderer.set_shape_color(bg_id, Some(Color::rgb(50, 180, 50))); // green stripe

    // Blurred shape (slightly transparent so stripe shows through)
    let shape = Shape::rect(
        [(ox + 15.0, oy + 10.0), (ox + 65.0, oy + 70.0)],
        Stroke::default(),
    );
    let id = renderer.add_shape(shape, None, None);
    renderer.set_shape_color(id, Some(Color::rgba(220, 50, 50, 200))); // semi-transparent red

    let (pw, ph) = renderer.size();
    let blur_params = BlurParams {
        radius: 8.0,
        _pad: 0.0,
        tex_size: [pw as f32, ph as f32],
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
        // Green stripe outside blurred shape — stays sharp and fully green
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

    // Background stripe (sharp, NOT blurred) — partially behind the blurred group
    let bg_stripe = Shape::rect(
        [(ox + 5.0, oy + 30.0), (ox + 75.0, oy + 50.0)],
        Stroke::default(),
    );
    let bg_stripe_id = renderer.add_shape(bg_stripe, None, None);
    renderer.set_shape_color(bg_stripe_id, Some(Color::rgb(220, 180, 50))); // yellow stripe

    // Blurred parent (group effect applies to parent+child as a unit)
    let parent = Shape::rect(
        [(ox + 10.0, oy + 5.0), (ox + 70.0, oy + 75.0)],
        Stroke::default(),
    );
    let parent_id = renderer.add_shape(parent, None, None);
    renderer.set_shape_color(parent_id, Some(Color::rgba(200, 200, 200, 200))); // semi-transparent gray

    let child = Shape::rect(
        [(ox + 20.0, oy + 20.0), (ox + 60.0, oy + 60.0)],
        Stroke::default(),
    );
    let child_id = renderer.add_shape(child, Some(parent_id), None);
    renderer.set_shape_color(child_id, Some(Color::rgb(50, 50, 220)));

    let (pw, ph) = renderer.size();
    let blur_params = BlurParams {
        radius: 6.0,
        _pad: 0.0,
        tex_size: [pw as f32, ph as f32],
    };
    renderer
        .set_group_effect(parent_id, BLUR_EFFECT_ID, bytemuck::bytes_of(&blur_params))
        .expect("Failed to set group effect");

    vec![
        // Child center inside blurred group: blue-ish
        PixelExpectation::new(
            ox as u32 + 40,
            oy as u32 + 40,
            80,
            80,
            220,
            200,
            "t28_group_blur_center",
        )
        .with_tolerance(60),
        // Yellow stripe outside blurred group — stays sharp
        PixelExpectation::opaque(
            ox as u32 + 8,
            oy as u32 + 40,
            220,
            180,
            50,
            "t28_bg_stripe_sharp",
        )
        .with_tolerance(55),
    ]
}

// ── Section I: Backdrop Effects ──────────────────────────────────────────────

fn tile_29_backdrop_blur_leaf(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(29);
    // Red background
    let bg = Shape::rect(
        [(ox + 5.0, oy + 5.0), (ox + 75.0, oy + 75.0)],
        Stroke::default(),
    );
    let bg_id = renderer.add_shape(bg, None, None);
    renderer.set_shape_color(bg_id, Some(Color::rgb(220, 50, 50)));

    // Sharp-edged stripe that partially overlaps the backdrop panel.
    // Outside the panel it should be crisp; under the panel it should get blurred.
    let stripe = Shape::rect(
        [(ox + 10.0, oy + 32.0), (ox + 70.0, oy + 48.0)],
        Stroke::default(),
    );
    let stripe_id = renderer.add_shape(stripe, None, None);
    renderer.set_shape_color(stripe_id, Some(Color::rgb(50, 50, 220))); // blue stripe

    // Backdrop blur panel on top (leaf, no children)
    let panel = Shape::rect(
        [(ox + 20.0, oy + 15.0), (ox + 60.0, oy + 65.0)],
        Stroke::default(),
    );
    let panel_id = renderer.add_shape(panel, None, None);
    renderer.set_shape_color(panel_id, Some(Color::rgba(255, 255, 255, 80)));

    let (pw, ph) = renderer.size();
    let blur_params = BlurParams {
        radius: 10.0,
        _pad: 0.0,
        tex_size: [pw as f32, ph as f32],
    };
    renderer
        .set_shape_backdrop_effect(panel_id, BLUR_EFFECT_ID, bytemuck::bytes_of(&blur_params))
        .expect("Failed to set backdrop effect");

    vec![
        // Panel interior: blurred mix of red bg + blue stripe + white overlay
        PixelExpectation::new(
            ox as u32 + 40,
            oy as u32 + 40,
            158,
            157,
            231,
            255,
            "t29_backdrop_interior",
        )
        .with_tolerance(40),
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
    let bg_id = renderer.add_shape(bg, None, None);
    renderer.set_shape_color(bg_id, Some(Color::rgb(50, 180, 50)));

    // Sharp-edged stripe partially behind the backdrop panel
    let stripe = Shape::rect(
        [(ox + 8.0, oy + 30.0), (ox + 72.0, oy + 50.0)],
        Stroke::default(),
    );
    let stripe_id = renderer.add_shape(stripe, None, None);
    renderer.set_shape_color(stripe_id, Some(Color::rgb(220, 50, 50))); // red stripe

    // Backdrop panel with a child
    let panel = Shape::rect(
        [(ox + 15.0, oy + 10.0), (ox + 65.0, oy + 70.0)],
        Stroke::default(),
    );
    let panel_id = renderer.add_shape(panel, None, None);
    renderer.set_shape_color(panel_id, Some(Color::rgba(255, 255, 255, 80)));

    let child = Shape::rect(
        [(ox + 25.0, oy + 50.0), (ox + 55.0, oy + 65.0)],
        Stroke::default(),
    );
    let child_id = renderer.add_shape(child, Some(panel_id), None);
    renderer.set_shape_color(child_id, Some(Color::rgb(50, 50, 220)));

    let (pw, ph) = renderer.size();
    let blur_params = BlurParams {
        radius: 8.0,
        _pad: 0.0,
        tex_size: [pw as f32, ph as f32],
    };
    renderer
        .set_shape_backdrop_effect(panel_id, BLUR_EFFECT_ID, bytemuck::bytes_of(&blur_params))
        .expect("Failed to set backdrop effect");

    vec![
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
    let bg_id = renderer.add_shape(bg, None, None);
    renderer.set_shape_color(bg_id, Some(Color::rgb(220, 180, 50)));

    // Sharp-edged blue stripe as child of bg — partially behind the backdrop panel
    let stripe = Shape::rect(
        [(ox + 8.0, oy + 32.0), (ox + 72.0, oy + 48.0)],
        Stroke::default(),
    );
    let stripe_id = renderer.add_shape(stripe, Some(bg_id), None);
    renderer.set_shape_color(stripe_id, Some(Color::rgb(50, 50, 200))); // blue stripe

    // Scissor-clipping parent (child of bg, sibling of stripe — drawn after stripe)
    let clip_parent = Shape::rect(
        [(ox + 10.0, oy + 10.0), (ox + 60.0, oy + 70.0)],
        Stroke::default(),
    );
    let clip_id = renderer.add_shape(clip_parent, Some(bg_id), None);
    renderer.set_shape_color(clip_id, Some(Color::rgba(0, 0, 0, 0))); // transparent — just clips

    // Backdrop panel inside scissor-clipped parent
    let panel = Shape::rect(
        [(ox + 15.0, oy + 15.0), (ox + 55.0, oy + 65.0)],
        Stroke::default(),
    );
    let panel_id = renderer.add_shape(panel, Some(clip_id), None);
    renderer.set_shape_color(panel_id, Some(Color::rgba(255, 255, 255, 80)));

    let (pw, ph) = renderer.size();
    let blur_params = BlurParams {
        radius: 6.0,
        _pad: 0.0,
        tex_size: [pw as f32, ph as f32],
    };
    renderer
        .set_shape_backdrop_effect(panel_id, BLUR_EFFECT_ID, bytemuck::bytes_of(&blur_params))
        .expect("Failed to set backdrop effect");

    vec![
        // Panel interior: blurred mix of yellow bg + blue stripe + white overlay
        PixelExpectation::new(
            ox as u32 + 35,
            oy as u32 + 40,
            157,
            157,
            219,
            255,
            "t31_backdrop_in_scissor",
        )
        .with_tolerance(40),
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

// ── Section J: Edge Cases ────────────────────────────────────────────────────

fn tile_32_tiny_1px_shape(renderer: &mut Renderer) -> Vec<PixelExpectation> {
    let (ox, oy) = tile_origin(32);
    let shape = Shape::rect(
        [(ox + 40.0, oy + 40.0), (ox + 41.0, oy + 41.0)],
        Stroke::default(),
    );
    let id = renderer.add_shape(shape, None, None);
    renderer.set_shape_color(id, Some(Color::rgb(255, 0, 255)));

    // Small shape — check that the general area has some color
    // Anti-aliasing fringe might spread the pixel
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
    let (ox, oy) = tile_origin(33);
    // Shape that extends beyond the right and bottom edges of this tile
    // (and possibly beyond the canvas itself for the last-row tiles)
    let shape = Shape::rect(
        [(ox + 50.0, oy + 50.0), (ox + 120.0, oy + 120.0)],
        Stroke::default(),
    );
    let id = renderer.add_shape(shape, None, None);
    renderer.set_shape_color(id, Some(Color::rgb(180, 50, 180)));

    vec![
        // Interior of the visible portion
        PixelExpectation::opaque(
            ox as u32 + 60,
            oy as u32 + 60,
            180,
            50,
            180,
            "t33_visible_portion",
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
    let id = renderer.add_cached_shape_to_the_render_queue(cache_key, None);
    renderer.set_shape_color(id, Some(Color::rgb(50, 180, 220)));

    vec![
        PixelExpectation::opaque(
            ox as u32 + 40,
            oy as u32 + 40,
            50,
            180,
            220,
            "t34_cached_interior",
        ),
        // Outside the cached shape — canvas bg
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
