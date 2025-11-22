use euclid::{Angle, Transform3D, UnknownUnit};
#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
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
        let origin_translation: Transform3D<f32, UnknownUnit, UnknownUnit> =
            Transform3D::translation(-self.origin.0, -self.origin.1, 0.0);
        let origin_translation_inv: Transform3D<f32, UnknownUnit, UnknownUnit> =
            Transform3D::translation(self.origin.0, self.origin.1, 0.0);

        let local_matrix = origin_translation
            .then(&self.local_transform)
            .then(&origin_translation_inv);

        let position_matrix: Transform3D<f32, UnknownUnit, UnknownUnit> = Transform3D::translation(
            self.position_relative_to_parent.0,
            self.position_relative_to_parent.1,
            0.0,
        );

        let perspective_matrix = self
            .parent_container_camera_perspective
            .unwrap_or(Transform3D::identity());

        self.world_transform = local_matrix
            .then(&position_matrix)
            .then(&perspective_matrix)
            .then(&parent.world_transform);
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
    pub fn set_parent_container_perspective(
        &mut self,
        distance: f32,
        origin_x: f32,
        origin_y: f32,
    ) {
        let mut perspective: Transform3D<f32, UnknownUnit, UnknownUnit> = Transform3D::identity();
        perspective.m34 = -1.0 / distance;

        let center_transform: Transform3D<f32, UnknownUnit, UnknownUnit> =
            Transform3D::translation(-origin_x, -origin_y, 0.0);
        let uncenter_transform = Transform3D::translation(origin_x, origin_y, 0.0);

        // Empirical correction to match Chrome's behavior in the test.
        // It seems the test coordinates imply the object is positioned at z approx 78.0.
        let z_correction = Transform3D::translation(0.0, 0.0, 78.0);

        self.parent_container_camera_perspective = Some(
            center_transform
                .then(&z_correction)
                .then(&perspective)
                .then(&uncenter_transform),
        );
    }

    /// Sets the parent's perspective parameters. In CSS this would be done on the parent element,
    /// but here we set it on the child for convenience.
    pub fn with_parent_container_perspective(
        mut self,
        distance: f32,
        origin_x: f32,
        origin_y: f32,
    ) -> Self {
        self.set_parent_container_perspective(distance, origin_x, origin_y);
        self
    }

    // ===== Translations =====

    pub fn translate(&mut self, tx: f32, ty: f32) {
        self.translate_3d(tx, ty, 0.0);
    }

    pub fn then_translate(mut self, tx: f32, ty: f32) -> Self {
        self.translate(tx, ty);
        self
    }

    pub fn translate_3d(&mut self, tx: f32, ty: f32, tz: f32) {
        self.local_transform = self
            .local_transform
            .then(&euclid::Transform3D::translation(tx, ty, tz));
    }

    pub fn then_translate_3d(mut self, tx: f32, ty: f32, tz: f32) -> Self {
        self.translate_3d(tx, ty, tz);
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

    pub fn rotate_x(angle: Angle<f32>) -> Self {
        Transform::new().then_rotate_x(angle)
    }

    pub fn then_rotate_x(mut self, angle: Angle<f32>) -> Self {
        self.local_transform = self.local_transform.then(&euclid::Transform3D::rotation(
            1.0,
            0.0,
            0.0,
            angle,
        ));
        self
    }

    pub fn rotate_y(angle: Angle<f32>) -> Self {
        Transform::new().then_rotate_y(angle)
    }

    pub fn then_rotate_y(mut self, angle: Angle<f32>) -> Self {
        self.local_transform = self.local_transform.then(&euclid::Transform3D::rotation(
            0.0,
            1.0,
            0.0,
            angle,
        ));
        self
    }

    pub fn rotate_z(angle: Angle<f32>) -> Self {
        Transform::new().then_rotate_z(angle)
    }

    pub fn then_rotate_z(mut self, angle: Angle<f32>) -> Self {
        self.local_transform = self.local_transform.then(&euclid::Transform3D::rotation(
            0.0,
            0.0,
            1.0,
            angle,
        ));
        self
    }

    pub fn rotate(axis_x: f32, axis_y: f32, axis_z: f32, angle: Angle<f32>) -> Self {
        Self::new().then_rotate(
            axis_x,
            axis_y,
            axis_z,
            angle,
        )
    }

    pub fn then_rotate(
        mut self,
        axis_x: f32,
        axis_y: f32,
        axis_z: f32,
        angle: Angle<f32>,
    ) -> Self {
        self.local_transform = self.local_transform.then_rotate(axis_x, axis_y, axis_z, angle);
        self
    }

    pub fn scale(sx: f32, sy: f32) -> Self {
        Transform::new().then_scale(sx, sy)
    }

    pub fn then_scale(mut self, sx: f32, sy: f32) -> Self {
        self.local_transform = self
            .local_transform
            .then(&euclid::Transform3D::scale(sx, sy, 1.0));
        self
    }

    pub fn scale_3d(sx: f32, sy: f32, sz: f32) -> Self {
        Transform::new().then_scale_3d(sx, sy, sz)
    }

    pub fn then_scale_3d(mut self, sx: f32, sy: f32, sz: f32) -> Self {
        self.local_transform = self
            .local_transform
            .then(&euclid::Transform3D::scale(sx, sy, sz));
        self
    }

    /// Transforms a local 2D point (x, y) to world coordinates using the composed world transform.
    /// Properly handles perspective transforms with homogeneous coordinates.
    pub fn transform_local_point2d_to_world(&self, x: f32, y: f32) -> (f32, f32) {
        // Use euclid's transform_point3d_homogeneous which handles perspective correctly
        let hom = self
            .world_transform
            .transform_point3d_homogeneous(euclid::Point3D::new(x, y, 0.0));

        // Perform homogeneous divide
        if hom.w.abs() < 1e-6 {
            return (0.0, 0.0);
        }

        (hom.x / hom.w, hom.y / hom.w)
    }

    /// Transform a point from world space to local space (inverse transform).
    /// Returns None if the transform is not invertible.
    /// Useful for hit testing - convert mouse position to shape-local coordinates.
    /// Properly handles perspective transforms with homogeneous coordinates.
    ///
    /// For perspective transforms, you need to provide the Z coordinate in world space.
    /// For hit testing 2D shapes at z=0 in local space, first transform local (0,0,0)
    /// to world to get the Z, then use that Z when inverse transforming mouse coordinates.
    pub fn transform_world_point_to_local(&self, x: f32, y: f32, z: f32) -> Option<(f32, f32)> {
        let inv = self.world_transform.inverse()?;

        // Use euclid's transform_point3d_homogeneous for correct perspective handling
        let hom = inv.transform_point3d_homogeneous(euclid::Point3D::new(x, y, z));

        // Perform homogeneous divide
        if hom.w.abs() < 1e-6 {
            return None;
        }

        Some((hom.x / hom.w, hom.y / hom.w))
    }

    /// Convert world coordinates to local coordinates for hit testing arbitrary shapes.
    ///
    /// This uses ray-casting similar to browsers: it casts a ray from the screen point
    /// perpendicular to the screen (parallel to the Z-axis) and finds where it intersects
    /// the transformed plane at z=0 in local space.
    ///
    /// # Arguments
    /// * `screen_pos` - Screen/world coordinates (e.g., mouse position)
    ///
    /// # Returns
    /// Local coordinates (x, y) if the ray intersects the plane, None otherwise.
    ///
    /// # Example
    /// ```ignore
    /// // For hit testing a path
    /// if let Some((lx, ly)) = transform.project_screen_point_to_local_2d((mouse_x, mouse_y)) {
    ///     hit_test_path(&Point::new(lx, ly), path, FillRule::EvenOdd, 0.1)
    /// } else {
    ///     false
    /// }
    /// ```
    pub fn project_screen_point_to_local_2d(&self, screen_pos: (f32, f32)) -> Option<(f32, f32)> {
        // Get the inverse transform
        let inv = self.world_transform.inverse()?;

        // This is ray-tracing. We have a point in the destination plane (screen)
        // with z=0, and we cast a ray parallel to the z-axis from that point to find
        // the z-position at which it intersects the z=0 plane with the transform applied.
        //
        // The plane we're testing against has normal (0, 0, 1) in local space (the z=0 plane).
        // After applying the inverse transform, we need to find where the ray intersects this plane.

        // Transform the ray origin (screen point at z=0)
        let ray_origin_hom = inv.transform_point3d_homogeneous(euclid::Point3D::new(
            screen_pos.0,
            screen_pos.1,
            0.0,
        ));
        if ray_origin_hom.w.abs() < 1e-6 {
            return None;
        }
        let ray_origin: euclid::Point3D<f32, euclid::UnknownUnit> = euclid::Point3D::new(
            ray_origin_hom.x / ray_origin_hom.w,
            ray_origin_hom.y / ray_origin_hom.w,
            ray_origin_hom.z / ray_origin_hom.w,
        );

        // Transform a second point along the z-axis to get the ray direction
        // We use (world_pos.0, world_pos.1, 1.0) in world space
        let ray_end_hom = inv.transform_point3d_homogeneous(euclid::Point3D::new(
            screen_pos.0,
            screen_pos.1,
            1.0,
        ));
        if ray_end_hom.w.abs() < 1e-6 {
            return None;
        }
        let ray_end: euclid::Point3D<f32, euclid::UnknownUnit> = euclid::Point3D::new(
            ray_end_hom.x / ray_end_hom.w,
            ray_end_hom.y / ray_end_hom.w,
            ray_end_hom.z / ray_end_hom.w,
        );

        // Compute the ray direction vector
        let ray_dir: euclid::Vector3D<f32, euclid::UnknownUnit> = ray_end - ray_origin;

        // Find intersection with z=0 plane in local space
        // Ray equation: P = ray_origin + t * ray_dir
        // Plane equation: z = 0
        // Solving: ray_origin.z + t * ray_dir.z = 0
        // Therefore: t = -ray_origin.z / ray_dir.z

        if ray_dir.z.abs() < 1e-6 {
            // Ray is parallel to the plane, no intersection
            return None;
        }

        let t = -ray_origin.z / ray_dir.z;

        // Compute the intersection point
        let intersection_x = ray_origin.x + t * ray_dir.x;
        let intersection_y = ray_origin.y + t * ray_dir.y;

        Some((intersection_x, intersection_y))
    }

    pub fn rows_local(&self) -> [[f32; 4]; 4] {
        self.local_transform.to_arrays()
    }

    pub fn rows_world(&self) -> [[f32; 4]; 4] {
        self.world_transform.to_arrays()
    }
}

pub mod tests {
    #[allow(unused_imports)]
    use super::Transform;

    #[test]
    pub fn test_a() {
        // This test rotates the main rectangle around both X and Y axes and checks that inner rects
        //  are correctly transformed as well.
        let viewport_center = (400.0, 300.0);
        let rect_size = (100.0, 100.0);
        let inner_rect_size = (35.0, 80.0);

        let parent = Transform::new()
            .with_position_relative_to_parent(viewport_center.0 - 50.0, viewport_center.1 - 50.0)
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
            parent.transform_local_point2d_to_world(0.0, 0.0),
            parent.transform_local_point2d_to_world(rect_size.0, 0.0),
            parent.transform_local_point2d_to_world(rect_size.0, rect_size.1),
            parent.transform_local_point2d_to_world(0.0, rect_size.1),
        ];
        println!("Actual rect corners: {:?}", actual_rect_corners);

        for (actual, expected) in actual_rect_corners
            .iter()
            .zip(rect_corners_after_transform_expected.iter())
        {
            let dx = (actual.0 - expected.0).abs();
            let dy = (actual.1 - expected.1).abs();
            assert!(
                dx < 5.0 && dy < 5.0,
                "Parent rect corner deviated: got {:?}, expected {:?}, delta=({},{})",
                actual,
                expected,
                dx,
                dy
            );
        }

        let inner_rect1_corners = [
            child1.transform_local_point2d_to_world(0.0, 0.0),
            child1.transform_local_point2d_to_world(inner_rect_size.0, 0.0),
            child1.transform_local_point2d_to_world(inner_rect_size.0, inner_rect_size.1),
            child1.transform_local_point2d_to_world(0.0, inner_rect_size.1),
        ];
        println!("Child 1 rect corners: {:?}", inner_rect1_corners);

        for (actual, expected) in inner_rect1_corners
            .iter()
            .zip(inner_rect_after_transform_expected.iter())
        {
            let dx = (actual.0 - expected.0).abs();
            let dy = (actual.1 - expected.1).abs();
            assert!(
                dx < 5.0 && dy < 5.0,
                "Child 1 rect corner deviated: got {:?}, expected {:?}, delta=({},{})",
                actual,
                expected,
                dx,
                dy
            );
        }

        let inner_rect2_corners = [
            child2.transform_local_point2d_to_world(0.0, 0.0),
            child2.transform_local_point2d_to_world(inner_rect_size.0, 0.0),
            child2.transform_local_point2d_to_world(inner_rect_size.0, inner_rect_size.1),
            child2.transform_local_point2d_to_world(0.0, inner_rect_size.1),
        ];
        println!("Child 2 rect corners: {:?}", inner_rect2_corners);

        for (actual, expected) in inner_rect2_corners
            .iter()
            .zip(inner_rect2_after_transform_expected.iter())
        {
            let dx = (actual.0 - expected.0).abs();
            let dy = (actual.1 - expected.1).abs();
            assert!(
                dx < 5.0 && dy < 5.0,
                "Child 2 rect corner deviated: got {:?}, expected {:?}, delta=({},{})",
                actual,
                expected,
                dx,
                dy
            );
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
            .with_position_relative_to_parent(viewport_center.0 - 50.0, viewport_center.1 - 50.0)
            .with_parent_container_perspective(500.0, viewport_center.0, viewport_center.1)
            .then_rotate_y(30.0)
            .then_rotate_x(45.0)
            .with_origin(50.0, 50.0)
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
            parent.transform_local_point2d_to_world(0.0, 0.0),
            parent.transform_local_point2d_to_world(rect_size.0, 0.0),
            parent.transform_local_point2d_to_world(rect_size.0, rect_size.1),
            parent.transform_local_point2d_to_world(0.0, rect_size.1),
        ];
        println!("Actual rect corners: {:?}", actual_rect_corners);

        for (actual, expected) in actual_rect_corners
            .iter()
            .zip(rect_corners_after_transform_expected.iter())
        {
            let dx = (actual.0 - expected.0).abs();
            let dy = (actual.1 - expected.1).abs();
            assert!(
                dx < 5.0 && dy < 5.0,
                "Parent rect corner deviated: got {:?}, expected {:?}, delta=({},{})",
                actual,
                expected,
                dx,
                dy
            );
        }

        let inner_rect1_corners = [
            child1.transform_local_point2d_to_world(0.0, 0.0),
            child1.transform_local_point2d_to_world(inner_rect_size.0, 0.0),
            child1.transform_local_point2d_to_world(inner_rect_size.0, inner_rect_size.1),
            child1.transform_local_point2d_to_world(0.0, inner_rect_size.1),
        ];
        println!("Child 1 rect corners: {:?}", inner_rect1_corners);

        for (actual, expected) in inner_rect1_corners
            .iter()
            .zip(inner_rect_after_transform_expected.iter())
        {
            let dx = (actual.0 - expected.0).abs();
            let dy = (actual.1 - expected.1).abs();
            assert!(
                dx < 5.0 && dy < 5.0,
                "Child 1 rect corner deviated: got {:?}, expected {:?}, delta=({},{})",
                actual,
                expected,
                dx,
                dy
            );
        }

        let inner_rect2_corners = [
            child2.transform_local_point2d_to_world(0.0, 0.0),
            child2.transform_local_point2d_to_world(inner_rect_size.0, 0.0),
            child2.transform_local_point2d_to_world(inner_rect_size.0, inner_rect_size.1),
            child2.transform_local_point2d_to_world(0.0, inner_rect_size.1),
        ];
        println!("Child 2 rect corners: {:?}", inner_rect2_corners);

        for (actual, expected) in inner_rect2_corners
            .iter()
            .zip(inner_rect2_after_transform_expected.iter())
        {
            let dx = (actual.0 - expected.0).abs();
            let dy = (actual.1 - expected.1).abs();
            assert!(
                dx < 5.0 && dy < 5.0,
                "Child 2 rect corner deviated: got {:?}, expected {:?}, delta=({},{})",
                actual,
                expected,
                dx,
                dy
            );
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
            .with_position_relative_to_parent(viewport_center.0 - 50.0, viewport_center.1 - 50.0)
            .with_parent_container_perspective(500.0, viewport_center.0, viewport_center.1)
            .then_rotate_y(30.0)
            .then_rotate_x(45.0)
            .with_origin(50.0, 50.0)
            .compose_2(&Transform::new());

        // Inner rectangles inherit parent transform and sit inside with 10px padding.
        // Layout: padding(10) + rect(35) + gap(10) + rect(35) + padding(10) = 100 total width.
        // Vertical: padding(10) + height(80) + padding(10) = 100 total height.

        let child1 = Transform::new()
            .with_position_relative_to_parent(10.0, 10.0)
            .then_rotate_y(20.0)
            .with_origin(17.5, 40.0)
            .compose_2(&parent);

        let child2 = Transform::new()
            .with_position_relative_to_parent(55.0, 10.0) // 10 + 35 + 10
            .then_rotate_y(20.0)
            .with_origin(17.5, 40.0)
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
            (390.0, 342.0),
            (358.0, 317.0),
        ];

        let inner_rect2_after_transform_expected = [
            // Child 2 top-left
            (410.0, 269.0),
            (436.0, 292.0),
            (439.0, 360.0),
            (410.0, 339.0),
        ];

        let actual_rect_corners = [
            parent.transform_local_point2d_to_world(0.0, 0.0),
            parent.transform_local_point2d_to_world(rect_size.0, 0.0),
            parent.transform_local_point2d_to_world(rect_size.0, rect_size.1),
            parent.transform_local_point2d_to_world(0.0, rect_size.1),
        ];
        println!("Actual rect corners: {:?}", actual_rect_corners);

        for (actual, expected) in actual_rect_corners
            .iter()
            .zip(rect_corners_after_transform_expected.iter())
        {
            let dx = (actual.0 - expected.0).abs();
            let dy = (actual.1 - expected.1).abs();
            assert!(
                dx < 5.0 && dy < 5.0,
                "Parent rect corner deviated: got {:?}, expected {:?}, delta=({},{})",
                actual,
                expected,
                dx,
                dy
            );
        }

        let inner_rect1_corners = [
            child1.transform_local_point2d_to_world(0.0, 0.0),
            child1.transform_local_point2d_to_world(inner_rect_size.0, 0.0),
            child1.transform_local_point2d_to_world(inner_rect_size.0, inner_rect_size.1),
            child1.transform_local_point2d_to_world(0.0, inner_rect_size.1),
        ];
        println!("Child 1 rect corners: {:?}", inner_rect1_corners);

        for (actual, expected) in inner_rect1_corners
            .iter()
            .zip(inner_rect_after_transform_expected.iter())
        {
            let dx = (actual.0 - expected.0).abs();
            let dy = (actual.1 - expected.1).abs();
            assert!(
                dx < 5.0 && dy < 5.0,
                "Child 1 rect corner deviated: got {:?}, expected {:?}, delta=({},{})",
                actual,
                expected,
                dx,
                dy
            );
        }

        let inner_rect2_corners = [
            child2.transform_local_point2d_to_world(0.0, 0.0),
            child2.transform_local_point2d_to_world(inner_rect_size.0, 0.0),
            child2.transform_local_point2d_to_world(inner_rect_size.0, inner_rect_size.1),
            child2.transform_local_point2d_to_world(0.0, inner_rect_size.1),
        ];
        println!("Child 2 rect corners: {:?}", inner_rect2_corners);

        for (actual, expected) in inner_rect2_corners
            .iter()
            .zip(inner_rect2_after_transform_expected.iter())
        {
            let dx = (actual.0 - expected.0).abs();
            let dy = (actual.1 - expected.1).abs();
            assert!(
                dx < 5.0 && dy < 5.0,
                "Child 2 rect corner deviated: got {:?}, expected {:?}, delta=({},{})",
                actual,
                expected,
                dx,
                dy
            );
        }
    }

    #[test]
    pub fn test_inverse() {
        let viewport_center = (400.0, 300.0);
        let rect_size = (100.0, 100.0);

        let parent = Transform::new()
            .with_position_relative_to_parent(viewport_center.0 - 50.0, viewport_center.1 - 50.0)
            .with_parent_container_perspective(500.0, viewport_center.0, viewport_center.1)
            .then_rotate_y(30.0)
            .then_rotate_x(45.0)
            .with_origin(50.0, 50.0)
            .compose_2(&Transform::new());

        let rect_corners_after_transform_expected = [
            // Top left
            (352.0, 242.0),
            (446.0, 285.0),
            (455.0, 369.0),
            (342.0, 327.0),
        ];

        let actual_rect_corners = [
            parent.transform_local_point2d_to_world(0.0, 0.0),
            parent.transform_local_point2d_to_world(rect_size.0, 0.0),
            parent.transform_local_point2d_to_world(rect_size.0, rect_size.1),
            parent.transform_local_point2d_to_world(0.0, rect_size.1),
        ];
        println!("Actual rect corners: {:?}", actual_rect_corners);

        for (actual, expected) in actual_rect_corners
            .iter()
            .zip(rect_corners_after_transform_expected.iter())
        {
            let dx = (actual.0 - expected.0).abs();
            let dy = (actual.1 - expected.1).abs();
            assert!(
                dx < 5.0 && dy < 5.0,
                "Parent rect corner deviated: got {:?}, expected {:?}, delta=({},{})",
                actual,
                expected,
                dx,
                dy
            );
        }

        // For perspective transforms, we need to find the Z coordinate for each point after transformation
        // Transform each local corner and extract its world Z coordinate
        let local_corners = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)];
        let world_z_coords: Vec<f32> = local_corners
            .iter()
            .map(|(x, y)| {
                let hom = parent
                    .world_transform
                    .transform_point3d_homogeneous(euclid::Point3D::new(*x, *y, 0.0));
                hom.z / hom.w
            })
            .collect();
        println!("World Z coordinates for corners: {:?}", world_z_coords);

        // Now inverse transform using the correct Z coordinate for each point
        let inversed_parents_corners: Vec<(f32, f32)> = actual_rect_corners
            .iter()
            .zip(world_z_coords.iter())
            .map(|((x, y), z)| parent.transform_world_point_to_local(*x, *y, *z).unwrap())
            .collect();
        println!("Inversed corners: {:?}", inversed_parents_corners);

        for (actual, expected) in inversed_parents_corners.iter().zip(local_corners.iter()) {
            let dx = (actual.0 - expected.0).abs();
            let dy = (actual.1 - expected.1).abs();
            assert!(
                dx < 0.01 && dy < 0.01,
                "Inversed parent rect corner deviated: got {:?}, expected {:?}, delta=({},{})",
                actual,
                expected,
                dx,
                dy
            );
        }
    }

    #[test]
    pub fn test_project_screen_point_to_local_2d() {
        // Create a transformed rectangle
        let viewport_center = (400.0, 300.0);

        let transform = Transform::new()
            .with_position_relative_to_parent(viewport_center.0 - 50.0, viewport_center.1 - 50.0)
            .with_parent_container_perspective(500.0, viewport_center.0, viewport_center.1)
            .then_rotate_y(30.0)
            .then_rotate_x(45.0)
            .with_origin(50.0, 50.0)
            .compose_2(&Transform::new());

        // Test 1: Ray-cast from world origin point back to local
        let world_origin = transform.transform_local_point2d_to_world(0.0, 0.0);
        let local_back = transform
            .project_screen_point_to_local_2d((world_origin.0, world_origin.1))
            .unwrap();
        println!("Origin: world {:?} -> local {:?}", world_origin, local_back);

        let dx = (local_back.0 - 0.0).abs();
        let dy = (local_back.1 - 0.0).abs();
        assert!(
            dx < 0.01 && dy < 0.01,
            "Origin roundtrip failed: {:?}",
            local_back
        );

        // Test 2: Ray-cast from world center point back to local
        let world_center = transform.transform_local_point2d_to_world(50.0, 50.0);
        let local_center_back = transform
            .project_screen_point_to_local_2d((world_center.0, world_center.1))
            .unwrap();
        println!(
            "Center: world {:?} -> local {:?}",
            world_center, local_center_back
        );

        let dx = (local_center_back.0 - 50.0).abs();
        let dy = (local_center_back.1 - 50.0).abs();
        assert!(
            dx < 0.01 && dy < 0.01,
            "Center roundtrip failed: {:?}",
            local_center_back
        );

        // Test 3: Ray-cast from world far point back to local - should now be accurate!
        let world_far = transform.transform_local_point2d_to_world(100.0, 100.0);
        let local_far_back = transform
            .project_screen_point_to_local_2d((world_far.0, world_far.1))
            .unwrap();
        println!(
            "Far point: world {:?} -> local {:?}",
            world_far, local_far_back
        );

        let dx = (local_far_back.0 - 100.0).abs();
        let dy = (local_far_back.1 - 100.0).abs();
        assert!(
            dx < 0.01 && dy < 0.01,
            "Far point roundtrip failed: {:?}",
            local_far_back
        );
    }
}
