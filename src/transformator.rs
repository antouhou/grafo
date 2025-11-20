use crate::{transformator, TransformInstance};
use euclid::{Point2D, Transform3D, UnknownUnit};
use crate::vertex::InstanceTransform;

#[derive(Clone, Debug)]
pub struct Transform {
    /// Local transform relative to parent
    pub local_transform: Transform3D<f32, UnknownUnit, UnknownUnit>,
    /// Fully composed world transform including all parent transforms (may include perspective)
    pub world_transform: Transform3D<f32, UnknownUnit, UnknownUnit>,
    /// Legacy CSS-like perspective distance (kept temporarily for compatibility)
    pub parent_container_camera_perspective_distance: f32,
    /// Origin relative to the shape (pivot)
    pub origin: (f32, f32),
    /// Legacy perspective origin (canvas-space)
    pub parent_container_camera_perspective_origin: (f32, f32),
    pub position_relative_to_parent: (f32, f32),
    /// Optional perspective matrix (with origin translations) replacing manual parameters.
    pub perspective_matrix: Option<Transform3D<f32, UnknownUnit, UnknownUnit>>,
}

impl Transform {
    pub fn new() -> Self {
        Self {
            local_transform: Transform3D::identity(),
            world_transform: Transform3D::identity(),
            parent_container_camera_perspective_distance: 0.0,
            origin: (0.0, 0.0),
            parent_container_camera_perspective_origin: (0.0, 0.0),
            position_relative_to_parent: (0.0, 0.0),
            perspective_matrix: None,
        }
    }

    /// Composes local transform with parent's world transform, and stores the result as this
    /// transform's world transform. Prent should be composed before calling this method.
    /// You can set up an empty transform for the root element.
    pub fn compose(&mut self, parent: &Transform) {
        let (px, py) = self.position_relative_to_parent;
        let (ox, oy) = self.origin;

        let t_from_origin: Transform3D<f32, UnknownUnit, UnknownUnit> =
            Transform3D::translation(-ox, -oy, 0.0);
        // Preserve depth: multiply parent affine (excluding perspective) by 3D point (px,py,0,1)
        let pm = &parent.world_transform; // parent affine world (no perspective baked)
        let p4 = [
            pm.m11 * px + pm.m12 * py + pm.m13 * 0.0 + pm.m14,
            pm.m21 * px + pm.m22 * py + pm.m23 * 0.0 + pm.m24,
            pm.m31 * px + pm.m32 * py + pm.m33 * 0.0 + pm.m34,
            pm.m41 * px + pm.m42 * py + pm.m43 * 0.0 + pm.m44,
        ];
        let px = p4[0];
        let py = p4[1];
        let pz = p4[2];
        let t_to_final: Transform3D<f32, UnknownUnit, UnknownUnit> =
            Transform3D::translation(px + ox, py + oy, pz);

        let base_world = parent
            .world_transform
            .then(&t_from_origin)
            .then(&self.local_transform)
            .then(&t_to_final);

        if self.parent_container_camera_perspective_distance <= 0.0 {
            self.parent_container_camera_perspective_distance = parent.parent_container_camera_perspective_distance;
            self.parent_container_camera_perspective_origin = parent.parent_container_camera_perspective_origin;
        }

        self.world_transform = base_world;

        println!("origin: ({}, {}), position: ({}, {})", ox, oy, px, py);

        // Perspective inheritance (CSS-like): if this transform has no perspective (==0)
        // inherit the nearest ancestor's perspective & origin. Do not override if we already have one.
        // Legacy path: parent has a perspective distance but no perspective_matrix (examples using with_perspective_distance)
        if self.parent_container_camera_perspective_distance <= 0.0 && parent.parent_container_camera_perspective_distance > 0.0 {
            // Only inherit if we didn't already set perspective (matrix or distance)
            if self.perspective_matrix.is_none() && parent.perspective_matrix.is_none() {
                self.parent_container_camera_perspective_distance = parent.parent_container_camera_perspective_distance;
                self.parent_container_camera_perspective_origin = parent.parent_container_camera_perspective_origin;
                println!("Inherited legacy perspective distance/origin from parent");
            } else if self.perspective_matrix.is_none() && parent.perspective_matrix.is_some() {
                println!("Perspective matrix inherited (will be applied last) ");
            }
        }
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

    pub fn set_parent_container_perspective_distance(&mut self, distance: f32) {
        self.parent_container_camera_perspective_distance = distance;
    }

    pub fn with_perspective_distance(mut self, distance: f32) -> Self {
        self.set_parent_container_perspective_distance(distance);
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

    pub fn set_parent_container_camera_perspective_origin(&mut self, x: f32, y: f32) {
        self.parent_container_camera_perspective_origin.0 = x;
        self.parent_container_camera_perspective_origin.1 = y;
    }

    pub fn with_camera_perspective_origin(mut self, x: f32, y: f32) -> Self {
        self.set_parent_container_camera_perspective_origin(x, y);
        self
    }

    /// Sets the parent's perspective parameters. In CSS this would be done on the parent element,
    /// but here we set it on the child for convenience.
    pub fn set_parent_container_perspective(&mut self, distance: f32, origin_x: f32, origin_y: f32) {
        self.set_parent_container_perspective_distance(distance);
        self.set_parent_container_camera_perspective_origin(origin_x, origin_y);
    }

    /// Sets the parent's perspective parameters. In CSS this would be done on the parent element,
    /// but here we set it on the child for convenience.
    pub fn with_parent_container_perspective(mut self, distance: f32, origin_x: f32, origin_y: f32) -> Self {
        self.set_parent_container_perspective(distance, origin_x, origin_y);
        self
    }

    /// Set perspective via matrix (distance > 0 enables). Stores legacy fields for now.
    // pub fn set_perspective(&mut self, distance: f32, origin_x: f32, origin_y: f32) {
    //     self.camera_perspective_distance = distance;
    //     self.camera_perspective_origin = (origin_x, origin_y);
    //     if distance <= 0.0 {
    //         self.perspective_matrix = None;
    //         return;
    //     }
    //     let d_inv = 1.0 / distance;
    //     // Perspective matrix producing w' = 1 + z/d
    //     let persp: Transform3D<f32, UnknownUnit, UnknownUnit> = Transform3D::new(
    //         1.0, 0.0, 0.0, 0.0,
    //         0.0, 1.0, 0.0, 0.0,
    //         0.0, 0.0, 1.0, 0.0,
    //         0.0, 0.0, d_inv, 1.0,
    //     );
    //     let to_origin = Transform3D::translation(-origin_x, -origin_y, 0.0);
    //     let back = Transform3D::translation(origin_x, origin_y, 0.0);
    //     let full = back.then(&persp).then(&to_origin);
    //     self.perspective_matrix = Some(full);
    // }

    // pub fn with_perspective(mut self, distance: f32, origin_x: f32, origin_y: f32) -> Self {
    //     self.set_perspective(distance, origin_x, origin_y);
    //     self
    // }

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

    pub fn transform_point_world(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        let m = &self.world_transform;
        let p4 = [
            m.m11 * x + m.m12 * y + m.m13 * z + m.m14,
            m.m21 * x + m.m22 * y + m.m23 * z + m.m24,
            m.m31 * x + m.m32 * y + m.m33 * z + m.m34,
            m.m41 * x + m.m42 * y + m.m43 * z + m.m44,
        ];
        let w = if p4[3].abs() < 1e-6 { 1.0 } else { p4[3] };
        (p4[0] / w, p4[1] / w, p4[2] / w)
    }

    pub fn transform_point2d_world(&self, x: f32, y: f32) -> (f32, f32) {
        let model = self.cols_world();

        // Apply the per-instance transform in pixel space first
        let p = mul_vec4(&InstanceTransform::from_columns(model), [x, y, 0.0, 1.0]);

        // p = (x', y', z', w') after affine
        let mut px = p[0];
        let mut py = p[1];
        if self.parent_container_camera_perspective_distance > 0.0 {
            let cx = self.parent_container_camera_perspective_origin.0;
            let cy = self.parent_container_camera_perspective_origin.1;
            let x_rel = px - cx;
            let y_rel = py - cy;
            // w' = 1 + z / d (z is p[2])
            let w_persp = 1.0 + p[2] / self.parent_container_camera_perspective_distance;
            let invw = 1.0 / f32::max(w_persp.abs(), 1e-6);
            px = x_rel * invw + cx;
            py = y_rel * invw + cy;
        }
        (px, py)
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

    pub fn cols_world_with_perspective(&self) -> [[f32; 4]; 4] {
        if let Some(p) = &self.perspective_matrix {
            // Apply perspective LAST: composite = world_affine * perspective
            let m = self.world_transform.then(p);
            // Standard column-major: col0=(m11,m21,m31,m41), etc.
            [
                [m.m11, m.m21, m.m31, m.m41],
                [m.m12, m.m22, m.m32, m.m42],
                [m.m13, m.m23, m.m33, m.m43],
                [m.m14, m.m24, m.m34, m.m44],
            ]
        } else {
            let m = &self.world_transform;
            [
                [m.m11, m.m21, m.m31, m.m41],
                [m.m12, m.m22, m.m32, m.m42],
                [m.m13, m.m23, m.m33, m.m43],
                [m.m14, m.m24, m.m34, m.m44],
            ]
        }
    }
}

/// Multiply a TransformInstance with a 4D vector (public for tests).
pub fn mul_vec4(t: &TransformInstance, v: [f32; 4]) -> [f32; 4] {
    [
        t.col0[0] * v[0] + t.col1[0] * v[1] + t.col2[0] * v[2] + t.col3[0] * v[3],
        t.col0[1] * v[0] + t.col1[1] * v[1] + t.col2[1] * v[2] + t.col3[1] * v[3],
        t.col0[2] * v[0] + t.col1[2] * v[1] + t.col2[2] * v[2] + t.col3[2] * v[3],
        t.col0[3] * v[0] + t.col1[3] * v[1] + t.col2[3] * v[2] + t.col3[3] * v[3],
    ]
}

#[test]
pub fn test_a() {
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
        parent.transform_point2d_world(100.0, 0.0),
        parent.transform_point2d_world(100.0, 100.0),
        parent.transform_point2d_world(0.0, 100.0),
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

