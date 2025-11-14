use std::f32::consts::PI;
use euclid::{Transform3D, Angle};

/// Represents the shader's transform instance format (column-major, 4 columns of vec4)
#[derive(Debug, Clone, Copy)]
struct ShaderTransform {
    col0: [f32; 4],
    col1: [f32; 4],
    col2: [f32; 4],
    col3: [f32; 4],
}

impl ShaderTransform {
    /// Converts euclid's Transform3D to shader format
    /// Following the conversion pattern from the user:
    ///   col0 = [m11, m21, m31, m14]
    ///   col1 = [m12, m22, m32, m24]
    ///   col2 = [m13, m23, m33, m34]
    ///   col3 = [m41, m42, m43, m44]
    fn from_euclid(m: Transform3D<f32, (), ()>) -> Self {
        Self {
            col0: [m.m11, m.m21, m.m31, m.m14],
            col1: [m.m12, m.m22, m.m32, m.m24],
            col2: [m.m13, m.m23, m.m33, m.m34],
            col3: [m.m41, m.m42, m.m43, m.m44],
        }
    }

    /// Multiplies this transform with a 4D vector
    fn mul_vec4(&self, v: [f32; 4]) -> [f32; 4] {
        [
            self.col0[0] * v[0] + self.col1[0] * v[1] + self.col2[0] * v[2] + self.col3[0] * v[3],
            self.col0[1] * v[0] + self.col1[1] * v[1] + self.col2[1] * v[2] + self.col3[1] * v[3],
            self.col0[2] * v[0] + self.col1[2] * v[1] + self.col2[2] * v[2] + self.col3[2] * v[3],
            self.col0[3] * v[0] + self.col1[3] * v[1] + self.col2[3] * v[2] + self.col3[3] * v[3],
        ]
    }
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

/// Applies perspective and transform to a 2D point.
/// This mimics the shader logic but with separate perspective parameter.
///
/// Note: Unlike a traditional graphics pipeline where transforms can be combined into a single matrix,
/// perspective and geometric transforms cannot be meaningfully combined when starting from 2D coordinates.
/// The perspective must act on the Z-coordinate that results from the 3D transform, so we apply them
/// in two separate steps: transform first (2D -> 3D with rotation), then perspective.
///
/// # Arguments
/// * `position` - The input position in pixel space (x, y)
/// * `transform` - The transformation matrix (e.g., rotation, scale, translation)
/// * `perspective_distance` - The perspective distance (like CSS perspective value)
///
/// # Returns
/// The transformed position in pixel space after perspective divide
fn apply_transform_with_perspective(
    position: [f32; 2],
    transform: Transform3D<f32, (), ()>,
    perspective_distance: f32,
) -> [f32; 2] {
    // Create perspective matrix
    let perspective = perspective_matrix(perspective_distance);
    
    // Apply transform first (converts 2D to 3D with rotation)
    let shader_transform = ShaderTransform::from_euclid(transform);
    let p_after_transform = shader_transform.mul_vec4([position[0], position[1], 0.0, 1.0]);
    
    // Then apply perspective to the 3D result
    let shader_perspective = ShaderTransform::from_euclid(perspective);
    let p = shader_perspective.mul_vec4(p_after_transform);
    
    // Homogeneous divide (perspective divide by w). Mirroring shader logic.
    let invw = 1.0 / f32::max(p[3].abs(), 1e-6);
    let px = p[0] * invw;
    let py = p[1] * invw;
    
    [px, py]
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
        let tl_transformed = apply_transform_with_perspective(top_left, transform, perspective_distance);
        let tr_transformed = apply_transform_with_perspective(top_right, transform, perspective_distance);
        let bl_transformed = apply_transform_with_perspective(bottom_left, transform, perspective_distance);
        let br_transformed = apply_transform_with_perspective(bottom_right, transform, perspective_distance);
        
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
}
