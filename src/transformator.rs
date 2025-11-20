use crate::{transformator, TransformInstance};
use euclid::{Point2D, Transform3D, UnknownUnit};

#[derive(Clone, Debug)]
pub struct Transform {
    /// Local transform relative to parent
    pub local_transform: Transform3D<f32, UnknownUnit, UnknownUnit>,
    /// Fully composed world transform including all parent transforms (may include perspective)
    pub world_transform: Transform3D<f32, UnknownUnit, UnknownUnit>,
    /// Origin relative to the shape (pivot)
    pub origin: (f32, f32),
    /// Layout position relative to the parent
    pub position_relative_to_parent: (f32, f32),
    /// Optional perspective matrix of the current element's parent
    pub parent_container_camera_perspective: Option<Transform3D<f32, UnknownUnit, UnknownUnit>>,
}

impl Transform {
    pub fn new() -> Self {
        Self {
            local_transform: Transform3D::identity(),
            world_transform: Transform3D::identity(),
            origin: (0.0, 0.0),
            position_relative_to_parent: (0.0, 0.0),
            parent_container_camera_perspective: None,
        }
    }

    /// Composes local transform with parent's world transform, and stores the result as this
    /// transform's world transform. Prent should be composed before calling this method.
    /// You can set up an empty transform for the root element.
    pub fn compose(&mut self, parent: &Transform) {
        // Should compose current local transform, perspective and parent and store it in the
        // world_transform field.
    }

    pub fn compose_2(mut self, parent: &Transform) -> Self {
        self.compose(parent);
        self
    }

    pub fn set_origin(&mut self, ox: f32, oy: f32) {
        self.origin = (ox, oy);
    }

    pub fn with_origin(mut self, ox: f32, oy: f32) -> Self {
        self.set_origin(ox, oy);
        self
    }

    pub fn set_position_relative_to_parent(&mut self, x: f32, y: f32) {
        self.position_relative_to_parent.0 = x;
        self.position_relative_to_parent.1 = y;
    }

    pub fn with_position_relative_to_parent(mut self, x: f32, y: f32) -> Self {
        self.set_position_relative_to_parent(x, y);
        self
    }

    /// Sets the parent's perspective parameters. In CSS this would be done on the parent element,
    /// but here we set it on the child for convenience.
    pub fn set_parent_container_perspective(&mut self, distance: f32, origin_x: f32, origin_y: f32) {
        // Calculate and set the perspective matrix based on distance and origin.
    }

    /// Sets the parent's perspective parameters. In CSS this would be done on the parent element,
    /// but here we set it on the child for convenience.
    pub fn with_parent_container_perspective(mut self, distance: f32, origin_x: f32, origin_y: f32) -> Self {
        self.set_parent_container_perspective(distance, origin_x, origin_y);
        self
    }

    // ===== Translations =====

    pub fn translate(&mut self, tx: f32, ty: f32, tz: f32) {
        self.local_transform = self
            .local_transform
            .then(&euclid::Transform3D::translation(tx, ty, tz));
    }

    pub fn then_translate(mut self, tx: f32, ty: f32, tz: f32) -> Self {
        self.translate(tx, ty, tz);
        self
    }

    pub fn translate_x(&mut self, tx: f32) {
        self.local_transform = self
            .local_transform
            .then(&euclid::Transform3D::translation(tx, 0.0, 0.0));
    }

    pub fn then_translate_x(mut self, tx: f32) -> Self {
        self.translate_x(tx);
        self
    }

    pub fn translate_y(&mut self, ty: f32) {
        self.local_transform = self
            .local_transform
            .then(&euclid::Transform3D::translation(0.0, ty, 0.0));
    }

    pub fn then_translate_y(mut self, ty: f32) -> Self {
        self.translate_y(ty);
        self
    }

    pub fn translate_z(&mut self, tz: f32) {
        self.local_transform = self
            .local_transform
            .then(&euclid::Transform3D::translation(0.0, 0.0, tz));
    }

    pub fn then_translate_z(mut self, tz: f32) -> Self {
        self.translate_z(tz);
        self
    }

    pub fn translate_2d(&mut self, tx: f32, ty: f32) {
        self.local_transform = self
            .local_transform
            .then(&euclid::Transform3D::translation(tx, ty, 0.0));
    }

    pub fn then_translate_2d(mut self, tx: f32, ty: f32) -> Self {
        self.translate_2d(tx, ty);
        self
    }

    // ===== Rotations =====

    pub fn rotate_x(&mut self, angle_degrees: f32) {
        self.local_transform = self.local_transform.then(&euclid::Transform3D::rotation(
            1.0,
            0.0,
            0.0,
            euclid::Angle::degrees(angle_degrees),
        ));
    }

    pub fn then_rotate_x(mut self, angle_degrees: f32) -> Self {
        self.rotate_x(angle_degrees);
        self
    }

    pub fn rotate_y(&mut self, angle_degrees: f32) {
        self.local_transform = self.local_transform.then(&euclid::Transform3D::rotation(
            0.0,
            1.0,
            0.0,
            euclid::Angle::degrees(angle_degrees),
        ));
    }

    pub fn then_rotate_y(mut self, angle_degrees: f32) -> Self {
        self.rotate_y(angle_degrees);
        self
    }

    pub fn rotate_z(&mut self, angle_degrees: f32) {
        self.local_transform = self.local_transform.then(&euclid::Transform3D::rotation(
            0.0,
            0.0,
            1.0,
            euclid::Angle::degrees(angle_degrees),
        ));
    }

    pub fn then_rotate_z(mut self, angle_degrees: f32) -> Self {
        self.rotate_z(angle_degrees);
        self
    }

    pub fn rotate(&mut self, axis_x: f32, axis_y: f32, axis_z: f32, angle_degrees: f32) {
        self.local_transform = self.local_transform.then(&euclid::Transform3D::rotation(
            axis_x,
            axis_y,
            axis_z,
            euclid::Angle::degrees(angle_degrees),
        ));
    }

    pub fn then_rotate(
        mut self,
        axis_x: f32,
        axis_y: f32,
        axis_z: f32,
        angle_degrees: f32,
    ) -> Self {
        self.rotate(axis_x, axis_y, axis_z, angle_degrees);
        self
    }

    pub fn transform_point2d_world(&self, x: f32, y: f32) -> (f32, f32) {
        // implement
        (0.0, 0.0)
    }

    // TODO: change the shader to actually work with euclid matrices instead of doing this conversion
    pub fn cols_local(&self) -> [[f32; 4]; 4] {
        let m = &self.local_transform;
        [
            [m.m11, m.m21, m.m31, m.m14],
            [m.m12, m.m22, m.m32, m.m24],
            [m.m13, m.m23, m.m33, m.m34],
            [m.m41, m.m42, m.m43, m.m44],
        ]
    }

    pub fn cols_world(&self) -> [[f32; 4]; 4] {
        let m = &self.world_transform;
        [
            [m.m11, m.m21, m.m31, m.m14],
            [m.m12, m.m22, m.m32, m.m24],
            [m.m13, m.m23, m.m33, m.m34],
            [m.m41, m.m42, m.m43, m.m44],
        ]
    }
}

pub mod tests {
    use crate::transformator::Transform;

    #[test]
    pub fn test_a() {
        // This test rotates the main rectangle around both X and Y axes and checks that inner rects
        //  are correctly transformed as well.
        let viewport_center = (400.0, 300.0);
        let rect_size = (100.0, 100.0);
        let inner_rect_size = (35.0, 80.0);

        let parent = Transform::new()
            .with_position_relative_to_parent(
                viewport_center.0 - 50.0,
                viewport_center.1 - 50.0,
            )
            .with_parent_container_perspective(500.0, viewport_center.0, viewport_center.1)
            .with_origin(50.0, 50.0)
            .then_rotate_x(45.0)
            .compose_2(&Transform::new());

        // Inner rectangles inherit parent transform and sit inside with 10px padding.
        // Layout: padding(10) + rect(35) + gap(10) + rect(35) + padding(10) = 100 total width.
        // Vertical: padding(10) + height(80) + padding(10) = 100 total height.

        let child1 = Transform::new()
            .with_position_relative_to_parent(10.0, 10.0)
            .compose_2(&parent);

        let child2 = Transform::new()
            .with_position_relative_to_parent(55.0, 10.0) // 10 + 35 + 10
            .compose_2(&parent);

        // VERY Rough (+- 5 pixels) estimations measured by hovering the mouse in Chrome for the
        // equivalent CSS-transformed elements. Points are clockwise.
        let rect_corners_after_transform_expected = [
            // Top left
            (346.0, 264.0),
            (455.0, 264.0),
            (465.0, 348.0),
            (336.0, 348.0),
        ];

        let inner_rect_after_transform_expected = [
            // Child 1 top-left
            (355.0, 270.0),
            (395.0, 270.0),
            (394.0, 338.0),
            (348.0, 338.0),
        ];

        let inner_rect2_after_transform_expected = [
            // Child 2 top-left
            (406.0, 270.0),
            (446.0, 270.0),
            (453.0, 338.0),
            (405.0, 338.0),
        ];

        let actual_rect_corners = [
            parent.transform_point2d_world(0.0, 0.0),
            parent.transform_point2d_world(rect_size.0, 0.0),
            parent.transform_point2d_world(rect_size.0, rect_size.1),
            parent.transform_point2d_world(0.0, rect_size.1),
        ];
        println!("Actual rect corners: {:?}", actual_rect_corners);

        for (actual, expected) in actual_rect_corners.iter().zip(rect_corners_after_transform_expected.iter()) {
            let dx = (actual.0 - expected.0).abs();
            let dy = (actual.1 - expected.1).abs();
            assert!(dx < 5.0 && dy < 5.0, "Parent rect corner deviated: got {:?}, expected {:?}, delta=({},{})", actual, expected, dx, dy);
        }

        let inner_rect1_corners = [
            child1.transform_point2d_world(0.0, 0.0),
            child1.transform_point2d_world(inner_rect_size.0, 0.0),
            child1.transform_point2d_world(inner_rect_size.0, inner_rect_size.1),
            child1.transform_point2d_world(0.0, inner_rect_size.1),
        ];
        println!("Child 1 rect corners: {:?}", inner_rect1_corners);

        for (actual, expected) in inner_rect1_corners.iter().zip(inner_rect_after_transform_expected.iter()) {
            let dx = (actual.0 - expected.0).abs();
            let dy = (actual.1 - expected.1).abs();
            assert!(dx < 5.0 && dy < 5.0, "Child 1 rect corner deviated: got {:?}, expected {:?}, delta=({},{})", actual, expected, dx, dy);
        }

        let inner_rect2_corners = [
            child2.transform_point2d_world(0.0, 0.0),
            child2.transform_point2d_world(inner_rect_size.0, 0.0),
            child2.transform_point2d_world(inner_rect_size.0, inner_rect_size.1),
            child2.transform_point2d_world(0.0, inner_rect_size.1),
        ];
        println!("Child 2 rect corners: {:?}", inner_rect2_corners);

        for (actual, expected) in inner_rect2_corners.iter().zip(inner_rect2_after_transform_expected.iter()) {
            let dx = (actual.0 - expected.0).abs();
            let dy = (actual.1 - expected.1).abs();
            assert!(dx < 5.0 && dy < 5.0, "Child 2 rect corner deviated: got {:?}, expected {:?}, delta=({},{})", actual, expected, dx, dy);
        }
    }

    #[test]
    pub fn test_b() {
        // This test rotates the main rectangle around both X and Y axes and checks that inner rects
        //  are correctly transformed as well.
        let viewport_center = (400.0, 300.0);
        let rect_size = (100.0, 100.0);
        let inner_rect_size = (35.0, 80.0);

        let parent = Transform::new()
            .with_position_relative_to_parent(
                viewport_center.0 - 50.0,
                viewport_center.1 - 50.0,
            )
            .with_parent_container_perspective(500.0, viewport_center.0, viewport_center.1)
            .with_origin(50.0, 50.0)
            .then_rotate_x(45.0)
            .then_rotate_y(30.0)
            .compose_2(&Transform::new());

        // Inner rectangles inherit parent transform and sit inside with 10px padding.
        // Layout: padding(10) + rect(35) + gap(10) + rect(35) + padding(10) = 100 total width.
        // Vertical: padding(10) + height(80) + padding(10) = 100 total height.

        let child1 = Transform::new()
            .with_position_relative_to_parent(10.0, 10.0)
            .compose_2(&parent);

        let child2 = Transform::new()
            .with_position_relative_to_parent(55.0, 10.0) // 10 + 35 + 10
            .compose_2(&parent);

        // VERY Rough (+- 5 pixels) estimations measured by hovering the mouse in Chrome for the
        // equivalent CSS-transformed elements. Points are clockwise.
        let rect_corners_after_transform_expected = [
            // Top left
            (352.0, 242.0),
            (446.0, 285.0),
            (455.0, 369.0),
            (342.0, 327.0),
        ];

        let inner_rect_after_transform_expected = [
            // Child 1 top-left
            (360.0, 253.0),
            (395.0, 268.0),
            (395.0, 338.0),
            (353.0, 321.0),
        ];

        let inner_rect2_after_transform_expected = [
            // Child 2 top-left
            (405.0, 272.0),
            (439.0, 287.0),
            (446.0, 356.0),
            (405.0, 341.0),
        ];

        let actual_rect_corners = [
            parent.transform_point2d_world(0.0, 0.0),
            parent.transform_point2d_world(rect_size.0, 0.0),
            parent.transform_point2d_world(rect_size.0, rect_size.1),
            parent.transform_point2d_world(0.0, rect_size.1),
        ];
        println!("Actual rect corners: {:?}", actual_rect_corners);

        for (actual, expected) in actual_rect_corners.iter().zip(rect_corners_after_transform_expected.iter()) {
            let dx = (actual.0 - expected.0).abs();
            let dy = (actual.1 - expected.1).abs();
            assert!(dx < 5.0 && dy < 5.0, "Parent rect corner deviated: got {:?}, expected {:?}, delta=({},{})", actual, expected, dx, dy);
        }

        let inner_rect1_corners = [
            child1.transform_point2d_world(0.0, 0.0),
            child1.transform_point2d_world(inner_rect_size.0, 0.0),
            child1.transform_point2d_world(inner_rect_size.0, inner_rect_size.1),
            child1.transform_point2d_world(0.0, inner_rect_size.1),
        ];
        println!("Child 1 rect corners: {:?}", inner_rect1_corners);

        for (actual, expected) in inner_rect1_corners.iter().zip(inner_rect_after_transform_expected.iter()) {
            let dx = (actual.0 - expected.0).abs();
            let dy = (actual.1 - expected.1).abs();
            assert!(dx < 5.0 && dy < 5.0, "Child 1 rect corner deviated: got {:?}, expected {:?}, delta=({},{})", actual, expected, dx, dy);
        }

        let inner_rect2_corners = [
            child2.transform_point2d_world(0.0, 0.0),
            child2.transform_point2d_world(inner_rect_size.0, 0.0),
            child2.transform_point2d_world(inner_rect_size.0, inner_rect_size.1),
            child2.transform_point2d_world(0.0, inner_rect_size.1),
        ];
        println!("Child 2 rect corners: {:?}", inner_rect2_corners);

        for (actual, expected) in inner_rect2_corners.iter().zip(inner_rect2_after_transform_expected.iter()) {
            let dx = (actual.0 - expected.0).abs();
            let dy = (actual.1 - expected.1).abs();
            assert!(dx < 5.0 && dy < 5.0, "Child 2 rect corner deviated: got {:?}, expected {:?}, delta=({},{})", actual, expected, dx, dy);
        }
    }

    #[test]
    pub fn test_c() {
        // This test rotates the main rectangle around both X and Y. It then rotates the inner rectangles around
        // Y axes to check that all rotations compose correctly.
        let viewport_center = (400.0, 300.0);
        let rect_size = (100.0, 100.0);
        let inner_rect_size = (35.0, 80.0);

        let parent = Transform::new()
            .with_position_relative_to_parent(
                viewport_center.0 - 50.0,
                viewport_center.1 - 50.0,
            )
            .with_parent_container_perspective(500.0, viewport_center.0, viewport_center.1)
            .with_origin(50.0, 50.0)
            .then_rotate_x(45.0)
            .then_rotate_y(30.0)
            .compose_2(&Transform::new());

        // Inner rectangles inherit parent transform and sit inside with 10px padding.
        // Layout: padding(10) + rect(35) + gap(10) + rect(35) + padding(10) = 100 total width.
        // Vertical: padding(10) + height(80) + padding(10) = 100 total height.

        let child1 = Transform::new()
            .with_position_relative_to_parent(10.0, 10.0)
            .then_rotate_y(20.0)
            .compose_2(&parent);

        let child2 = Transform::new()
            .with_position_relative_to_parent(55.0, 10.0) // 10 + 35 + 10
            .then_rotate_y(20.0)
            .compose_2(&parent);

        // VERY Rough (+- 5 pixels) estimations measured by hovering the mouse in Chrome for the
        // equivalent CSS-transformed elements. Points are clockwise.
        let rect_corners_after_transform_expected = [
            // Top left
            (352.0, 242.0),
            (446.0, 285.0),
            (455.0, 369.0),
            (342.0, 327.0),
        ];

        let inner_rect_after_transform_expected = [
            // Child 1 top-left
            (364.0, 248.0),
            (391.0, 272.0),
            (390.0, 343.0),
            (358.0, 317.0),
        ];

        let inner_rect2_after_transform_expected = [
            // Child 2 top-left
            (410.0, 269.0),
            (436.0, 292.0),
            (441.0, 363.0),
            (410.0, 339.0),
        ];

        let actual_rect_corners = [
            parent.transform_point2d_world(0.0, 0.0),
            parent.transform_point2d_world(rect_size.0, 0.0),
            parent.transform_point2d_world(rect_size.0, rect_size.1),
            parent.transform_point2d_world(0.0, rect_size.1),
        ];
        println!("Actual rect corners: {:?}", actual_rect_corners);

        for (actual, expected) in actual_rect_corners.iter().zip(rect_corners_after_transform_expected.iter()) {
            let dx = (actual.0 - expected.0).abs();
            let dy = (actual.1 - expected.1).abs();
            assert!(dx < 5.0 && dy < 5.0, "Parent rect corner deviated: got {:?}, expected {:?}, delta=({},{})", actual, expected, dx, dy);
        }

        let inner_rect1_corners = [
            child1.transform_point2d_world(0.0, 0.0),
            child1.transform_point2d_world(inner_rect_size.0, 0.0),
            child1.transform_point2d_world(inner_rect_size.0, inner_rect_size.1),
            child1.transform_point2d_world(0.0, inner_rect_size.1),
        ];
        println!("Child 1 rect corners: {:?}", inner_rect1_corners);

        for (actual, expected) in inner_rect1_corners.iter().zip(inner_rect_after_transform_expected.iter()) {
            let dx = (actual.0 - expected.0).abs();
            let dy = (actual.1 - expected.1).abs();
            assert!(dx < 5.0 && dy < 5.0, "Child 1 rect corner deviated: got {:?}, expected {:?}, delta=({},{})", actual, expected, dx, dy);
        }

        let inner_rect2_corners = [
            child2.transform_point2d_world(0.0, 0.0),
            child2.transform_point2d_world(inner_rect_size.0, 0.0),
            child2.transform_point2d_world(inner_rect_size.0, inner_rect_size.1),
            child2.transform_point2d_world(0.0, inner_rect_size.1),
        ];
        println!("Child 2 rect corners: {:?}", inner_rect2_corners);

        for (actual, expected) in inner_rect2_corners.iter().zip(inner_rect2_after_transform_expected.iter()) {
            let dx = (actual.0 - expected.0).abs();
            let dy = (actual.1 - expected.1).abs();
            assert!(dx < 5.0 && dy < 5.0, "Child 2 rect corner deviated: got {:?}, expected {:?}, delta=({},{})", actual, expected, dx, dy);
        }
    }
}
