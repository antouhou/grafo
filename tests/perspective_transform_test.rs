use std::f32::consts::PI;
use euclid::{Transform3D, Angle};
use grafo::{TransformInstance, InstanceRenderParams};

/// Converts euclid's Transform3D to grafo's InstanceTransform format.
/// Euclid stores translation in the last ROW (m41, m42, m43) with column-major notation.
/// Our WGSL uses column-major matrices multiplied by column vectors: model * vec4(x, y, z, 1).
/// Therefore, translation must be in the LAST COLUMN (col3.xyz). Map as follows:
///   col0 = [m11, m21, m31, m14]
///   col1 = [m12, m22, m32, m24]
///   col2 = [m13, m23, m33, m34]
///   col3 = [m41, m42, m43, m44]
/// For typical affine transforms, m14/m24/m34 are 0.
fn transform_instance_from_euclid(m: Transform3D<f32, (), ()>) -> TransformInstance {
    TransformInstance {
        col0: [m.m11, m.m21, m.m31, m.m14],
        col1: [m.m12, m.m22, m.m32, m.m24],
        col2: [m.m13, m.m23, m.m33, m.m34],
        col3: [m.m41, m.m42, m.m43, m.m44],
    }
}

/// Multiplies a TransformInstance with a 4D vector
fn mul_vec4(t: &TransformInstance, v: [f32; 4]) -> [f32; 4] {
    [
        t.col0[0] * v[0] + t.col1[0] * v[1] + t.col2[0] * v[2] + t.col3[0] * v[3],
        t.col0[1] * v[0] + t.col1[1] * v[1] + t.col2[1] * v[2] + t.col3[1] * v[3],
        t.col0[2] * v[0] + t.col1[2] * v[1] + t.col2[2] * v[2] + t.col3[2] * v[3],
        t.col0[3] * v[0] + t.col1[3] * v[1] + t.col2[3] * v[2] + t.col3[3] * v[3],
    ]
}

/// Creates a perspective matrix
/// d is the distance from the viewer to the projection plane
/// In euclid's row-major notation: element mRC is row R, column C
/// For CSS-style perspective where closer objects (positive z) appear larger:
/// w' = w - z/d means m34 = -1/d, but CSS uses the opposite convention
/// CSS perspective: w' = 1 - z/d, so we want w' = w + z/d, meaning m34 = +1/d
fn perspective_matrix(d: f32) -> Transform3D<f32, (), ()> {
    // Row-major order: m11,m12,m13,m14, m21,m22,m23,m24, m31,m32,m33,m34, m41,m42,m43,m44
    Transform3D::new(
        1.0, 0.0, 0.0, 0.0,      // row 1
        0.0, 1.0, 0.0, 0.0,      // row 2
        0.0, 0.0, 1.0, 1.0 / d,  // row 3: m34 = +1/d for CSS-style perspective
        0.0, 0.0, 0.0, 1.0,      // row 4
    )
}

/// Applies perspective and transform to a 2D point, then applies a 2D offset.
/// This mimics the shader logic but with separate perspective parameter.
///
/// Note: Unlike a traditional graphics pipeline where transforms can be combined into a single matrix,
/// perspective and geometric transforms cannot be meaningfully combined when starting from 2D coordinates.
/// The perspective must act on the Z-coordinate that results from the 3D transform, so we apply them
/// in two separate steps: transform first (2D -> 3D with rotation), then perspective.
/// Finally, a 2D offset is applied for positioning without affecting perspective ratios.
///
/// # Arguments
/// * `position` - The input position in pixel space (x, y)
/// * `transform` - The transformation matrix (e.g., rotation, scale, translation)
/// * `perspective_distance` - The perspective distance (like CSS perspective value)
/// * `offset` - Optional 2D offset to apply after perspective transformation for positioning
///
/// # Returns
/// The transformed position in pixel space after perspective divide and offset
fn apply_transform_with_perspective(
    position: [f32; 2],
    transform: Transform3D<f32, (), ()>,
    perspective_distance: f32,
    offset: Option<[f32; 2]>,
) -> [f32; 2] {
    // Create perspective matrix
    let perspective = perspective_matrix(perspective_distance);
    
    // Apply transform first (converts 2D to 3D with rotation)
    let shader_transform = transform_instance_from_euclid(transform);
    let p_after_transform = mul_vec4(&shader_transform, [position[0], position[1], 0.0, 1.0]);
    
    // Then apply perspective to the 3D result
    let shader_perspective = transform_instance_from_euclid(perspective);
    let p = mul_vec4(&shader_perspective, p_after_transform);
    
    // Homogeneous divide (perspective divide by w). Mirroring shader logic.
    let invw = 1.0 / f32::max(p[3].abs(), 1e-6);
    let px = p[0] * invw;
    let py = p[1] * invw;
    
    // Apply 2D offset if provided (for positioning without affecting perspective)
    if let Some([ox, oy]) = offset {
        [px + ox, py + oy]
    } else {
        [px, py]
    }
}

/// Mimics the shader vertex transformation logic with added perspective capability.
/// This prototypes future shader changes where perspective is a separate parameter.
///
/// # Arguments
/// * `position` - The input position in pixel space (x, y)
/// * `transform` - The per-instance transformation matrix (rotation, scale, etc.)
/// * `render_params` - The render parameters including perspective_distance and viewport_position
/// * `canvas_size` - The canvas size in pixels [width, height]
///
/// # Returns
/// NDC coordinates [x, y] in range [-1, 1]
fn shader_vertex_transform(
    position: [f32; 2],
    transform: &TransformInstance,
    render_params: &InstanceRenderParams,
    canvas_size: [f32; 2],
) -> [f32; 2] {
    // Apply the per-instance transform in pixel space first
    let p = mul_vec4(transform, [position[0], position[1], 0.0, 1.0]);
    
    // Apply perspective if specified (acting on the z-coordinate from transform)
    let w = if render_params.camera_perspective > 0.0 {
        // CSS-style perspective: w' = 1 + z/d
        1.0 + p[2] / render_params.camera_perspective
    } else {
        // No perspective, use w from transform
        p[3]
    };
    
    // Homogeneous divide to account for perspective. Clamp w to avoid infinities.
    let invw = 1.0 / f32::max(w.abs(), 1e-6);
    let mut px = p[0] * invw; // pixel-space x after perspective
    let mut py = p[1] * invw; // pixel-space y after perspective
    
    // Apply viewport position in pixel space
    px += render_params.viewport_position[0];
    py += render_params.viewport_position[1];
    
    // Then convert to NDC (Normalized Device Coordinates)
    // NDC is a cube with corners (-1, -1, -1) and (1, 1, 1).
    let ndc_x = 2.0 * px / canvas_size[0] - 1.0;
    let ndc_y = 1.0 - 2.0 * py / canvas_size[1];
    
    [ndc_x, ndc_y]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_star_wars_tilt_transform() {
        // Setup: 100x100 rectangle with rotateX(45deg) and perspective(500px)
        // The rectangle is centered at (0, 0) for simplicity, so corners are at:
        // top-left: (-50, -50), top-right: (50, -50)
        // bottom-left: (-50, 50), bottom-right: (50, 50)
        
        let perspective_distance = 500.0;
        let angle_deg = 45.0;
        let angle_rad = Angle::degrees(angle_deg);
        
        // Create rotation transform using euclid
        let transform = Transform3D::rotation(1.0, 0.0, 0.0, angle_rad);
        
        // Define the four corners of a 100x100 rectangle centered at origin
        let top_left = [-50.0, -50.0];
        let top_right = [50.0, -50.0];
        let bottom_left = [-50.0, 50.0];
        let bottom_right = [50.0, 50.0];
        
        // Apply transform with perspective
        let tl_transformed = apply_transform_with_perspective(top_left, transform, perspective_distance, None);
        let tr_transformed = apply_transform_with_perspective(top_right, transform, perspective_distance, None);
        let bl_transformed = apply_transform_with_perspective(bottom_left, transform, perspective_distance, None);
        let br_transformed = apply_transform_with_perspective(bottom_right, transform, perspective_distance, None);
        
        // Calculate expected values manually:
        // For rotateX(45deg):
        // x' = x
        // y' = y * cos(45°) - z * sin(45°) = y * 0.707... (since z=0 initially)
        // z' = y * sin(45°) + z * cos(45°) = y * 0.707...
        // Then perspective: w = 1 - z'/d = 1 - y*sin(45°)/500
        // After divide: x'' = x/w, y'' = y'/w
        
        let angle_rad_f32 = angle_deg * PI / 180.0;
        let cos45 = angle_rad_f32.cos();
        let sin45 = angle_rad_f32.sin();
        
        // Top-left corner: (-50, -50)
        // After rotation: x=-50, y'=-50*cos45=-35.355, z'=-50*sin45=-35.355
        // After perspective: w=1-(-35.355)/500=1.0707, x''=-50/1.0707=-46.698, y''=-35.355/1.0707=-33.018
        let expected_tl_x = -50.0 / (1.0 - (-50.0 * sin45) / perspective_distance);
        let expected_tl_y = (-50.0 * cos45) / (1.0 - (-50.0 * sin45) / perspective_distance);
        
        // Top-right corner: (50, -50)
        let expected_tr_x = 50.0 / (1.0 - (-50.0 * sin45) / perspective_distance);
        let expected_tr_y = (-50.0 * cos45) / (1.0 - (-50.0 * sin45) / perspective_distance);
        
        // Bottom-left corner: (-50, 50)
        let expected_bl_x = -50.0 / (1.0 - (50.0 * sin45) / perspective_distance);
        let expected_bl_y = (50.0 * cos45) / (1.0 - (50.0 * sin45) / perspective_distance);
        
        // Bottom-right corner: (50, 50)
        let expected_br_x = 50.0 / (1.0 - (50.0 * sin45) / perspective_distance);
        let expected_br_y = (50.0 * cos45) / (1.0 - (50.0 * sin45) / perspective_distance);
        
        // Assert with small epsilon for floating point comparison
        let epsilon = 0.001;
        
        // Top-left
        assert!(
            (tl_transformed[0] - expected_tl_x).abs() < epsilon,
            "Top-left X: expected {}, got {}",
            expected_tl_x,
            tl_transformed[0]
        );
        assert!(
            (tl_transformed[1] - expected_tl_y).abs() < epsilon,
            "Top-left Y: expected {}, got {}",
            expected_tl_y,
            tl_transformed[1]
        );
        
        // Top-right
        assert!(
            (tr_transformed[0] - expected_tr_x).abs() < epsilon,
            "Top-right X: expected {}, got {}",
            expected_tr_x,
            tr_transformed[0]
        );
        assert!(
            (tr_transformed[1] - expected_tr_y).abs() < epsilon,
            "Top-right Y: expected {}, got {}",
            expected_tr_y,
            tr_transformed[1]
        );
        
        // Bottom-left
        assert!(
            (bl_transformed[0] - expected_bl_x).abs() < epsilon,
            "Bottom-left X: expected {}, got {}",
            expected_bl_x,
            bl_transformed[0]
        );
        assert!(
            (bl_transformed[1] - expected_bl_y).abs() < epsilon,
            "Bottom-left Y: expected {}, got {}",
            expected_bl_y,
            bl_transformed[1]
        );
        
        // Bottom-right
        assert!(
            (br_transformed[0] - expected_br_x).abs() < epsilon,
            "Bottom-right X: expected {}, got {}",
            expected_br_x,
            br_transformed[0]
        );
        assert!(
            (br_transformed[1] - expected_br_y).abs() < epsilon,
            "Bottom-right Y: expected {}, got {}",
            expected_br_y,
            br_transformed[1]
        );
        
        // Print the actual values for verification
        println!("Top-left: ({:.3}, {:.3})", tl_transformed[0], tl_transformed[1]);
        println!("Top-right: ({:.3}, {:.3})", tr_transformed[0], tr_transformed[1]);
        println!("Bottom-left: ({:.3}, {:.3})", bl_transformed[0], bl_transformed[1]);
        println!("Bottom-right: ({:.3}, {:.3})", br_transformed[0], br_transformed[1]);
    }

    #[test]
    fn test_translation_preserves_perspective_ratio() {
        // This test verifies that when we have identical rectangles (same size, same rotation)
        // the perspective distortion creates the same width/height ratios regardless of
        // where the rectangle's center is positioned in 2D space before the 3D transform.
        // 
        // Key insight: We're testing that the intrinsic perspective effect on the rectangle
        // (top narrower than bottom for rotateX) is preserved, even though the absolute
        // transformed positions will differ.
        
        let perspective_distance = 500.0;
        let angle_deg = 45.0;
        let angle_rad = Angle::degrees(angle_deg);
        
        // Create a rotation transform
        let rotation = Transform3D::rotation(1.0, 0.0, 0.0, angle_rad);
        
        // Rectangle 1: corners relative to center at origin
        let rect1_corners = [
            [-50.0, -50.0],
            [50.0, -50.0],
            [-50.0, 50.0],
            [50.0, 50.0],
        ];
        
        // Apply rotation and perspective to rectangle 1 (no offset)
        let transformed1: Vec<[f32; 2]> = rect1_corners
            .iter()
            .map(|&corner| apply_transform_with_perspective(corner, rotation, perspective_distance, None))
            .collect();
        
        // Calculate dimensions for rectangle 1
        let width_top_1 = transformed1[1][0] - transformed1[0][0];
        let width_bottom_1 = transformed1[3][0] - transformed1[2][0];
        let height_left_1 = transformed1[2][1] - transformed1[0][1];
        
        // Now apply with 2D offset (positioning after perspective)
        let translation_x = 200.0;
        let translation_y = 150.0;
        
        let transformed1_translated: Vec<[f32; 2]> = rect1_corners
            .iter()
            .map(|&corner| apply_transform_with_perspective(
                corner, 
                rotation, 
                perspective_distance, 
                Some([translation_x, translation_y])
            ))
            .collect();
        
        // Calculate dimensions for translated rectangle
        let width_top_1_trans = transformed1_translated[1][0] - transformed1_translated[0][0];
        let width_bottom_1_trans = transformed1_translated[3][0] - transformed1_translated[2][0];
        let height_left_1_trans = transformed1_translated[2][1] - transformed1_translated[0][1];
        
        // The dimensions should be identical (translation after perspective doesn't change sizes)
        let epsilon = 0.001;
        
        assert!(
            (width_top_1 - width_top_1_trans).abs() < epsilon,
            "Top width should be identical after translation: before={}, after={}",
            width_top_1,
            width_top_1_trans
        );
        
        assert!(
            (width_bottom_1 - width_bottom_1_trans).abs() < epsilon,
            "Bottom width should be identical after translation: before={}, after={}",
            width_bottom_1,
            width_bottom_1_trans
        );
        
        assert!(
            (height_left_1 - height_left_1_trans).abs() < epsilon,
            "Left height should be identical after translation: before={}, after={}",
            height_left_1,
            height_left_1_trans
        );
        
        // Verify translation actually moved the rectangle
        assert!(
            (transformed1_translated[0][0] - transformed1[0][0] - translation_x).abs() < epsilon,
            "X translation should be {}", translation_x
        );
        assert!(
            (transformed1_translated[0][1] - transformed1[0][1] - translation_y).abs() < epsilon,
            "Y translation should be {}", translation_y
        );
        
        println!("\nOriginal perspective-transformed rectangle:");
        println!("  Top width: {:.3}, Bottom width: {:.3}", width_top_1, width_bottom_1);
        println!("  Left height: {:.3}", height_left_1);
        println!("  Top-left corner: ({:.3}, {:.3})", transformed1[0][0], transformed1[0][1]);
        
        println!("\nAfter 2D translation (+{}, +{}):", translation_x, translation_y);
        println!("  Top width: {:.3}, Bottom width: {:.3}", width_top_1_trans, width_bottom_1_trans);
        println!("  Left height: {:.3}", height_left_1_trans);
        println!("  Top-left corner: ({:.3}, {:.3})", transformed1_translated[0][0], transformed1_translated[0][1]);
    }

    #[test]
    fn test_shader_vertex_transform_basic() {
        // Test that shader_vertex_transform produces correct NDC coordinates
        let canvas_size = [800.0, 600.0];
        
        // Identity transform
        let identity = TransformInstance::identity();
        let no_effects = InstanceRenderParams::default();
        
        // Test center of canvas should map to (0, 0) in NDC
        let center = [400.0, 300.0];
        let ndc = shader_vertex_transform(center, &identity, &no_effects, canvas_size);
        
        let epsilon = 0.001;
        assert!(
            ndc[0].abs() < epsilon && ndc[1].abs() < epsilon,
            "Center of canvas should map to NDC (0, 0), got ({}, {})",
            ndc[0], ndc[1]
        );
        
        // Test top-left corner should map to (-1, 1) in NDC
        let top_left = [0.0, 0.0];
        let ndc_tl = shader_vertex_transform(top_left, &identity, &no_effects, canvas_size);
        assert!(
            (ndc_tl[0] + 1.0).abs() < epsilon && (ndc_tl[1] - 1.0).abs() < epsilon,
            "Top-left should map to NDC (-1, 1), got ({}, {})",
            ndc_tl[0], ndc_tl[1]
        );
        
        // Test bottom-right corner should map to (1, -1) in NDC
        let bottom_right = [800.0, 600.0];
        let ndc_br = shader_vertex_transform(bottom_right, &identity, &no_effects, canvas_size);
        assert!(
            (ndc_br[0] - 1.0).abs() < epsilon && (ndc_br[1] + 1.0).abs() < epsilon,
            "Bottom-right should map to NDC (1, -1), got ({}, {})",
            ndc_br[0], ndc_br[1]
        );
        
        println!("\nBasic NDC mapping:");
        println!("  Center (400, 300) -> NDC ({:.3}, {:.3})", ndc[0], ndc[1]);
        println!("  Top-left (0, 0) -> NDC ({:.3}, {:.3})", ndc_tl[0], ndc_tl[1]);
        println!("  Bottom-right (800, 600) -> NDC ({:.3}, {:.3})", ndc_br[0], ndc_br[1]);
    }

    #[test]
    fn test_shader_vertex_transform_with_offset() {
        // Test that viewport_position is applied in pixel space before NDC conversion
        let canvas_size = [800.0, 600.0];
        let identity = TransformInstance::identity();
        
        // Start at origin, apply 400px right and 300px down viewport position
        let position = [0.0, 0.0];
        let render_params = InstanceRenderParams {
            camera_perspective: 0.0,
            viewport_position: [400.0, 300.0],
            _padding: 0.0,
        };
        
        // Should end up at canvas center, which is NDC (0, 0)
        let ndc = shader_vertex_transform(position, &identity, &render_params, canvas_size);
        
        let epsilon = 0.001;
        assert!(
            ndc[0].abs() < epsilon && ndc[1].abs() < epsilon,
            "Origin + viewport_position (400, 300) should map to NDC (0, 0), got ({}, {})",
            ndc[0], ndc[1]
        );
        
        println!("\nViewport position test:");
        println!("  Position (0, 0) + viewport_position (400, 300) -> NDC ({:.3}, {:.3})", ndc[0], ndc[1]);
    }

    #[test]
    fn test_shader_vertex_transform_with_perspective() {
        // Test shader_vertex_transform with separate perspective parameter
        // This should produce the same Star Wars tilt effect as our manual tests
        let canvas_size = [800.0, 600.0];
        let angle_rad = Angle::degrees(45.0);
        let perspective_dist = 500.0;
        
        // Create ONLY rotation transform (no perspective baked in)
        let rotation = Transform3D::rotation(1.0, 0.0, 0.0, angle_rad);
        let rotation_transform = transform_instance_from_euclid(rotation);
        
        // Rectangle corners centered at origin
        let corners = [
            [-50.0, -50.0],
            [50.0, -50.0],
            [-50.0, 50.0],
            [50.0, 50.0],
        ];
        
        // Apply using shader_vertex_transform with separate perspective
        let render_params = InstanceRenderParams {
            camera_perspective: perspective_dist,
            viewport_position: [400.0, 300.0],
            _padding: 0.0,
        };
        let ndc_corners: Vec<[f32; 2]> = corners
            .iter()
            .map(|&corner| shader_vertex_transform(
                corner,
                &rotation_transform,
                &render_params,
                canvas_size
            ))
            .collect();
        
        // Calculate widths in NDC space
        let width_top_ndc = ndc_corners[1][0] - ndc_corners[0][0];
        let width_bottom_ndc = ndc_corners[3][0] - ndc_corners[2][0];
        
        // Bottom should be wider than top (Star Wars effect)
        assert!(
            width_bottom_ndc > width_top_ndc,
            "Bottom width should be greater than top width (perspective effect). Top: {}, Bottom: {}",
            width_top_ndc, width_bottom_ndc
        );
        
        // Check the ratio matches our expected Star Wars tilt
        // From the first test, we know the pixel-space widths are ~93.4 (top) and ~107.6 (bottom)
        let ratio = width_bottom_ndc / width_top_ndc;
        let expected_ratio = 107.609 / 93.396; // ~1.152
        let ratio_epsilon = 0.01;
        
        assert!(
            (ratio - expected_ratio).abs() < ratio_epsilon,
            "Width ratio should match expected perspective ratio. Got: {}, Expected: {}",
            ratio, expected_ratio
        );
        
        // All corners should be within NDC range [-1, 1]
        for (i, ndc) in ndc_corners.iter().enumerate() {
            assert!(
                ndc[0] >= -1.0 && ndc[0] <= 1.0,
                "Corner {} X should be in NDC range [-1, 1], got {}",
                i, ndc[0]
            );
            assert!(
                ndc[1] >= -1.0 && ndc[1] <= 1.0,
                "Corner {} Y should be in NDC range [-1, 1], got {}",
                i, ndc[1]
            );
        }
        
        println!("\nShader perspective transform to NDC:");
        println!("  Top-left: ({:.3}, {:.3})", ndc_corners[0][0], ndc_corners[0][1]);
        println!("  Top-right: ({:.3}, {:.3})", ndc_corners[1][0], ndc_corners[1][1]);
        println!("  Bottom-left: ({:.3}, {:.3})", ndc_corners[2][0], ndc_corners[2][1]);
        println!("  Bottom-right: ({:.3}, {:.3})", ndc_corners[3][0], ndc_corners[3][1]);
        println!("  Top width (NDC): {:.3}, Bottom width (NDC): {:.3}", width_top_ndc, width_bottom_ndc);
        println!("  Width ratio: {:.3} (expected: {:.3})", ratio, expected_ratio);
    }

    #[test]
    fn test_shader_vertex_transform_matches_manual() {
        // Verify shader_vertex_transform produces identical results to manual calculation
        let canvas_size = [800.0, 600.0];
        let angle_rad = Angle::degrees(45.0);
        let perspective_dist = 500.0;
        
        let rotation = Transform3D::rotation(1.0, 0.0, 0.0, angle_rad);
        let transform = transform_instance_from_euclid(rotation);
        
        let corners = [
            [-50.0, -50.0],
            [50.0, -50.0],
            [-50.0, 50.0],
            [50.0, 50.0],
        ];
        
        let viewport_position = [400.0, 300.0];
        
        for &corner in &corners {
            // Method 1: Use shader_vertex_transform
            let render_params = InstanceRenderParams {
                camera_perspective: perspective_dist,
                viewport_position,
                _padding: 0.0,
            };
            let ndc_shader = shader_vertex_transform(
                corner,
                &transform,
                &render_params,
                canvas_size
            );
            
            // Method 2: Use apply_transform_with_perspective (our original manual approach)
            let rotation_euclid = Transform3D::rotation(1.0, 0.0, 0.0, angle_rad);
            let pixel_pos = apply_transform_with_perspective(
                corner,
                rotation_euclid,
                perspective_dist,
                Some(viewport_position)
            );
            
            // Convert pixel position to NDC manually
            let ndc_x_manual = 2.0 * pixel_pos[0] / canvas_size[0] - 1.0;
            let ndc_y_manual = 1.0 - 2.0 * pixel_pos[1] / canvas_size[1];
            
            let epsilon = 0.001;
            assert!(
                (ndc_shader[0] - ndc_x_manual).abs() < epsilon,
                "Corner {:?}: NDC X should match. Shader: {}, Manual: {}",
                corner, ndc_shader[0], ndc_x_manual
            );
            assert!(
                (ndc_shader[1] - ndc_y_manual).abs() < epsilon,
                "Corner {:?}: NDC Y should match. Shader: {}, Manual: {}",
                corner, ndc_shader[1], ndc_y_manual
            );
        }
        
        println!("\n✓ shader_vertex_transform matches manual calculation for all corners");
    }

    #[test]
    fn test_shader_vertex_transform_offset_preserves_ratio() {
        // Test that viewport_position doesn't affect perspective ratios
        let canvas_size = [800.0, 600.0];
        let angle_rad = Angle::degrees(45.0);
        let perspective_dist = 500.0;
        
        let rotation = Transform3D::rotation(1.0, 0.0, 0.0, angle_rad);
        let transform = transform_instance_from_euclid(rotation);
        
        let corners = [
            [-50.0, -50.0],
            [50.0, -50.0],
            [-50.0, 50.0],
            [50.0, 50.0],
        ];
        
        // Test with no viewport_position
        let render_params_no_offset = InstanceRenderParams {
            camera_perspective: perspective_dist,
            viewport_position: [0.0, 0.0],
            _padding: 0.0,
        };
        let ndc_no_offset: Vec<[f32; 2]> = corners
            .iter()
            .map(|&corner| shader_vertex_transform(corner, &transform, &render_params_no_offset, canvas_size))
            .collect();
        
        // Test with viewport_position
        let render_params_with_offset = InstanceRenderParams {
            camera_perspective: perspective_dist,
            viewport_position: [200.0, 150.0],
            _padding: 0.0,
        };
        let ndc_with_offset: Vec<[f32; 2]> = corners
            .iter()
            .map(|&corner| shader_vertex_transform(corner, &transform, &render_params_with_offset, canvas_size))
            .collect();
        
        // Calculate widths and heights for both
        let width_top_no_offset = ndc_no_offset[1][0] - ndc_no_offset[0][0];
        let width_bottom_no_offset = ndc_no_offset[3][0] - ndc_no_offset[2][0];
        let height_left_no_offset = ndc_no_offset[2][1] - ndc_no_offset[0][1];
        
        let width_top_with_offset = ndc_with_offset[1][0] - ndc_with_offset[0][0];
        let width_bottom_with_offset = ndc_with_offset[3][0] - ndc_with_offset[2][0];
        let height_left_with_offset = ndc_with_offset[2][1] - ndc_with_offset[0][1];
        
        // Widths and heights should be identical (viewport_position doesn't change shape)
        let epsilon = 0.001;
        
        assert!(
            (width_top_no_offset - width_top_with_offset).abs() < epsilon,
            "Top width should be identical with/without viewport_position"
        );
        assert!(
            (width_bottom_no_offset - width_bottom_with_offset).abs() < epsilon,
            "Bottom width should be identical with/without viewport_position"
        );
        assert!(
            (height_left_no_offset - height_left_with_offset).abs() < epsilon,
            "Left height should be identical with/without viewport_position"
        );
        
        println!("\n✓ Viewport position preserves perspective ratios in NDC space");
        println!("  Top width: {:.6} (both)", width_top_no_offset);
        println!("  Bottom width: {:.6} (both)", width_bottom_no_offset);
        println!("  Height: {:.6} (both)", height_left_no_offset);
    }

    #[test]
    fn test_shader_transform_matches_manual_calculation() {
        // Verify that shader_vertex_transform with a pre-baked perspective transform
        // gives the same results as manual step-by-step calculation
        let canvas_size = [800.0, 600.0];
        let angle_rad = Angle::degrees(45.0);
        let perspective_dist = 500.0;
        
        let rotation = Transform3D::rotation(1.0, 0.0, 0.0, angle_rad);
        let perspective = perspective_matrix(perspective_dist);
        let combined = perspective.then(&rotation);
        let transform = transform_instance_from_euclid(combined);
        
        let position = [-50.0, -50.0];
        let viewport_position = [400.0, 300.0];
        
        // Method 1: Use shader_vertex_transform with pre-baked perspective (no separate perspective param)
        let render_params = InstanceRenderParams {
            camera_perspective: 0.0, // No additional perspective since it's pre-baked
            viewport_position,
            _padding: 0.0,
        };
        let ndc_shader = shader_vertex_transform(position, &transform, &render_params, canvas_size);
        
        // Method 2: Manual calculation with pre-baked perspective
        let p = mul_vec4(&transform, [position[0], position[1], 0.0, 1.0]);
        let invw = 1.0 / f32::max(p[3].abs(), 1e-6);
        let px = p[0] * invw + viewport_position[0];
        let py = p[1] * invw + viewport_position[1];
        let ndc_x_manual = 2.0 * px / canvas_size[0] - 1.0;
        let ndc_y_manual = 1.0 - 2.0 * py / canvas_size[1];
        
        let epsilon = 0.001;
        assert!(
            (ndc_shader[0] - ndc_x_manual).abs() < epsilon,
            "NDC X should match: shader={}, manual={}",
            ndc_shader[0], ndc_x_manual
        );
        assert!(
            (ndc_shader[1] - ndc_y_manual).abs() < epsilon,
            "NDC Y should match: shader={}, manual={}",
            ndc_shader[1], ndc_y_manual
        );
        
        println!("\nShader vs Manual (pre-baked perspective):");
        println!("  Shader: ({:.6}, {:.6})", ndc_shader[0], ndc_shader[1]);
        println!("  Manual: ({:.6}, {:.6})", ndc_x_manual, ndc_y_manual);
    }
}
