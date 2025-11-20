use euclid::{Transform3D, UnknownUnit};
use crate::TransformInstance;

#[derive(Clone, Debug)]
pub struct Transform {
    /// Local transform relative to parent
    pub local_transform: Transform3D<f32, UnknownUnit, UnknownUnit>,
    /// Fully composed world transform including all parent transforms (may include perspective)
    pub world_transform: Transform3D<f32, UnknownUnit, UnknownUnit>,
    /// Legacy CSS-like perspective distance (kept temporarily for compatibility)
    pub camera_perspective_distance: f32,
    /// Origin relative to the shape (pivot)
    pub origin: (f32, f32),
    /// Legacy perspective origin (canvas-space)
    pub camera_perspective_origin: (f32, f32),
    pub position_relative_to_parent: (f32, f32),
    /// Optional perspective matrix (with origin translations) replacing manual parameters.
    pub perspective_matrix: Option<Transform3D<f32, UnknownUnit, UnknownUnit>>,
}

impl Transform {
    pub fn new() -> Self {
        Self {
            local_transform: Transform3D::identity(),
            world_transform: Transform3D::identity(),
            camera_perspective_distance: 0.0,
            origin: (0.0, 0.0),
            camera_perspective_origin: (0.0, 0.0),
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

        // Attempt pivot-first ordering given current matrix multiplication expectations:
        // world = parent * translate(-origin) * local * translate(position + origin)

        // let newo = parent.local_transform.transform_point2d(Point2D::new(px, py)).unwrap();
        // let px = newp.x;
        // let py = newp.y;

        let t_from_origin: Transform3D<f32, UnknownUnit, UnknownUnit> = Transform3D::translation(-ox, -oy, 0.0);
        // let newp = apply_transform(
        //     [px, py],
        //     &TransformInstance::from_columns(parent.cols_world()),
        //     &InstanceRenderParams {
        //         camera_perspective: parent.camera_perspective_distance,
        //         camera_perspective_origin: [parent.camera_perspective_origin.0, parent.camera_perspective_origin.1],
        //         _padding: 0.0,
        //     },
        // );
        // let px = newp[0];
        // let py = newp[1];
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
        let t_to_final: Transform3D<f32, UnknownUnit, UnknownUnit> = Transform3D::translation(px + ox, py + oy, pz);

        let base_world = parent.world_transform
            .then(&t_from_origin)
            .then(&self.local_transform)
            .then(&t_to_final);

        // Perspective inheritance: adopt parent's matrix if none locally and legacy distance <= 0
        let effective_persp = if let Some(ref local) = self.perspective_matrix {
            Some(local.clone())
        } else if let Some(ref parent_p) = parent.perspective_matrix {
            if self.camera_perspective_distance <= 0.0 {
                self.camera_perspective_distance = parent.camera_perspective_distance;
                self.camera_perspective_origin = parent.camera_perspective_origin;
                Some(parent_p.clone())
            } else {
                None
            }
        } else {
            None
        };

        // Do NOT bake perspective here; keep world_transform as pure affine chain.
        // Store inherited perspective matrix if any for later final projection.
        if self.perspective_matrix.is_none() {
            self.perspective_matrix = effective_persp;
        }
        self.world_transform = base_world;

        println!("origin: ({}, {}), position: ({}, {})", ox, oy, px, py);

        // Perspective inheritance (CSS-like): if this transform has no perspective (==0)
        // inherit the nearest ancestor's perspective & origin. Do not override if we already have one.
        // Legacy path: parent has a perspective distance but no perspective_matrix (examples using with_perspective_distance)
        if self.camera_perspective_distance <= 0.0 && parent.camera_perspective_distance > 0.0 {
            // Only inherit if we didn't already set perspective (matrix or distance)
            if self.perspective_matrix.is_none() && parent.perspective_matrix.is_none() {
                self.camera_perspective_distance = parent.camera_perspective_distance;
                self.camera_perspective_origin = parent.camera_perspective_origin;
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

    pub fn set_perspective_distance(&mut self, distance: f32) {
        self.camera_perspective_distance = distance;
    }

    pub fn with_perspective_distance(mut self, distance: f32) -> Self {
        self.set_perspective_distance(distance);
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

    pub fn set_camera_perspective_origin(&mut self, x: f32, y: f32) {
        self.camera_perspective_origin.0 = x;
        self.camera_perspective_origin.1 = y;
    }

    pub fn with_camera_perspective_origin(mut self, x: f32, y: f32) -> Self {
        self.set_camera_perspective_origin(x, y);
        self
    }

    /// Set perspective via matrix (distance > 0 enables). Stores legacy fields for now.
    pub fn set_perspective(&mut self, distance: f32, origin_x: f32, origin_y: f32) {
        self.camera_perspective_distance = distance;
        self.camera_perspective_origin = (origin_x, origin_y);
        if distance <= 0.0 {
            self.perspective_matrix = None;
            return;
        }
        let d_inv = 1.0 / distance;
        // Perspective matrix producing w' = 1 + z/d
        let persp: Transform3D<f32, UnknownUnit, UnknownUnit> = Transform3D::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, d_inv, 1.0,
        );
        let to_origin = Transform3D::translation(-origin_x, -origin_y, 0.0);
        let back = Transform3D::translation(origin_x, origin_y, 0.0);
        let full = back.then(&persp).then(&to_origin);
        self.perspective_matrix = Some(full);
    }

    pub fn with_perspective(mut self, distance: f32, origin_x: f32, origin_y: f32) -> Self {
        self.set_perspective(distance, origin_x, origin_y);
        self
    }

    // ===== Translations =====

    pub fn translate(&mut self, tx: f32, ty: f32, tz: f32) {
        self.local_transform = self.local_transform.then(&euclid::Transform3D::translation(tx, ty, tz));
    }

    pub fn then_translate(mut self, tx: f32, ty: f32, tz: f32) -> Self {
        self.translate(tx, ty, tz);
        self
    }

    pub fn translate_x(&mut self, tx: f32) {
        self.local_transform = self.local_transform.then(&euclid::Transform3D::translation(tx, 0.0, 0.0));
    }

    pub fn then_translate_x(mut self, tx: f32) -> Self {
        self.translate_x(tx);
        self
    }

    pub fn translate_y(&mut self, ty: f32) {
        self.local_transform = self.local_transform.then(&euclid::Transform3D::translation(0.0, ty, 0.0));
    }

    pub fn then_translate_y(mut self, ty: f32) -> Self {
        self.translate_y(ty);
        self
    }

    pub fn translate_z(&mut self, tz: f32) {
        self.local_transform = self.local_transform.then(&euclid::Transform3D::translation(0.0, 0.0, tz));
    }

    pub fn then_translate_z(mut self, tz: f32) -> Self {
        self.translate_z(tz);
        self
    }

    pub fn translate_2d(&mut self, tx: f32, ty: f32) {
        self.local_transform = self.local_transform.then(&euclid::Transform3D::translation(tx, ty, 0.0));
    }

    pub fn then_translate_2d(mut self, tx: f32, ty: f32) -> Self {
        self.translate_2d(tx, ty);
        self
    }

    // ===== Rotations =====

    pub fn rotate_x(&mut self, angle_degrees: f32) {
        self.local_transform = self.local_transform.then(&euclid::Transform3D::rotation(1.0, 0.0, 0.0, euclid::Angle::degrees(angle_degrees)));
    }

    pub fn then_rotate_x(mut self, angle_degrees: f32) -> Self {
        self.rotate_x(angle_degrees);
        self
    }

    pub fn rotate_y(&mut self, angle_degrees: f32) {
        self.local_transform = self.local_transform.then(&euclid::Transform3D::rotation(0.0, 1.0, 0.0, euclid::Angle::degrees(angle_degrees)));
    }

    pub fn then_rotate_y(mut self, angle_degrees: f32) -> Self {
        self.rotate_y(angle_degrees);
        self
    }

    pub fn rotate_z(&mut self, angle_degrees: f32) {
        self.local_transform = self.local_transform.then(&euclid::Transform3D::rotation(0.0, 0.0, 1.0, euclid::Angle::degrees(angle_degrees)));
    }

    pub fn then_rotate_z(mut self, angle_degrees: f32) -> Self {
        self.rotate_z(angle_degrees);
        self
    }

    pub fn rotate(&mut self, axis_x: f32, axis_y: f32, axis_z: f32, angle_degrees: f32) {
        self.local_transform = self.local_transform.then(&euclid::Transform3D::rotation(axis_x, axis_y, axis_z, euclid::Angle::degrees(angle_degrees)));
    }

    pub fn then_rotate(mut self, axis_x: f32, axis_y: f32, axis_z: f32, angle_degrees: f32) -> Self {
        self.rotate(axis_x, axis_y, axis_z, angle_degrees);
        self
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

    pub fn cols_world_with_perspective(&self) -> [[f32;4];4] {
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

#[test]
pub fn test() {
    let viewport_center = (400.0, 300.0);
    let transform = Transform::new()
        .with_position_relative_to_parent(viewport_center.0 - 50.0, viewport_center.1 - 50.0)
        .with_perspective(500.0, viewport_center.0, viewport_center.1)
        .with_origin(50.0, 50.0)
        .then_rotate_x(45.0)
        // Use viewport position to position the shape at screen center after perspective is applied.
        .compose_2(&Transform::new());
    // Apply perspective last (like CSS): use helper.
    let world_cols = transform.cols_world_with_perspective();
    let tinst = TransformInstance::from_columns(world_cols);
    let center_local = [50.0, 50.0];
    let p4 = mul_vec4(&tinst, [center_local[0], center_local[1], 0.0, 1.0]);
    let w = if p4[3].abs() < 1e-6 { 1.0 } else { p4[3] };
    let projected = [p4[0] / w, p4[1] / w];
    println!("projected center = {:?} (w={})", projected, w);

    // Expect near viewport center (allow small epsilon due to perspective numeric error)
    let eps = 0.01;
    assert!((projected[0] - viewport_center.0).abs() < eps && (projected[1] - viewport_center.1).abs() < eps,
        "Projected center deviated: got {:?} expected {:?}", projected, viewport_center);
}

#[test]
pub fn test_inherit_perspective() {
    // Parent with perspective
    let parent = Transform::new()
        .with_position_relative_to_parent(0.0, 0.0)
        .with_perspective(600.0, 400.0, 300.0)
        .with_origin(0.0, 0.0)
        .compose_2(&Transform::new());

    // Child without perspective should inherit
    let child = Transform::new()
        .with_position_relative_to_parent(10.0, 20.0)
        .with_origin(0.0, 0.0)
        .compose_2(&parent);

    assert_eq!(child.camera_perspective_distance, parent.camera_perspective_distance, "Child did not inherit perspective distance");
    assert_eq!(child.camera_perspective_origin, parent.camera_perspective_origin, "Child did not inherit perspective origin");

    // Grandchild with explicit perspective should NOT inherit
    let grandchild = Transform::new()
        .with_perspective(300.0, 200.0, 150.0)
        .compose_2(&child);

    assert_eq!(grandchild.camera_perspective_distance, 300.0, "Grandchild perspective overridden unexpectedly");
    assert_eq!(grandchild.camera_perspective_origin, (200.0, 150.0), "Grandchild origin overridden unexpectedly");
}

#[test]
pub fn test_child_perspective_effect() {
    // Parent sets perspective; child rotates causing z variation which should be projected.
    let parent = Transform::new()
        .with_perspective(800.0, 400.0, 300.0)
        .compose_2(&Transform::new());

    let child = Transform::new()
        .with_position_relative_to_parent(100.0, 100.0)
        .with_origin(50.0, 50.0)
        .then_rotate_x(60.0) // large rotation for visible perspective skew
        .compose_2(&parent);

    let cols_no_persp = child.cols_world(); // affine only
    let cols_with_persp = child.cols_world_with_perspective(); // with perspective
    let t_no = TransformInstance::from_columns(cols_no_persp);
    let t_p = TransformInstance::from_columns(cols_with_persp);
    let local_pt = [50.0, 10.0]; // point with some y to produce z after rotation
    let p4_no = mul_vec4(&t_no, [local_pt[0], local_pt[1], 0.0, 1.0]);
    let p4_p = mul_vec4(&t_p, [local_pt[0], local_pt[1], 0.0, 1.0]);
    let w_no = if p4_no[3].abs() < 1e-6 { 1.0 } else { p4_no[3] };
    let w_p = if p4_p[3].abs() < 1e-6 { 1.0 } else { p4_p[3] };
    let screen_no = [p4_no[0]/w_no, p4_no[1]/w_no];
    let screen_p = [p4_p[0]/w_p, p4_p[1]/w_p];
    // Expect vertical position to shift due to perspective (y move toward origin center)
    assert!((screen_no[1] - screen_p[1]).abs() > 0.01, "Perspective had no visible effect on child rotation");
}

#[test]
pub fn test_sibling_depth_difference() {
    let parent = Transform::new()
        .with_position_relative_to_parent(0.0, 0.0)
        .with_perspective(700.0, 400.0, 300.0)
        .then_rotate_x(45.0)
        .then_rotate_y(35.0)
        .compose_2(&Transform::new());

    let sibling1 = Transform::new()
        .with_position_relative_to_parent(10.0, 10.0)
        .with_origin(50.0, 50.0)
        .compose_2(&parent);
    let sibling2 = Transform::new()
        .with_position_relative_to_parent(60.0, 10.0) // farther along X
        .with_origin(50.0, 50.0)
        .compose_2(&parent);

    let c1_cols = sibling1.cols_world_with_perspective();
    let c2_cols = sibling2.cols_world_with_perspective();
    let t1 = TransformInstance::from_columns(c1_cols);
    let t2 = TransformInstance::from_columns(c2_cols);
    let local_pt = [50.0, 20.0];
    let p1 = mul_vec4(&t1, [local_pt[0], local_pt[1], 0.0, 1.0]);
    let p2 = mul_vec4(&t2, [local_pt[0], local_pt[1], 0.0, 1.0]);
    let w1 = if p1[3].abs() < 1e-6 {1.0} else {p1[3]};
    let w2 = if p2[3].abs() < 1e-6 {1.0} else {p2[3]};
    let s1 = [p1[0]/w1, p1[1]/w1];
    let s2 = [p2[0]/w2, p2[1]/w2];
    // Expect horizontal projected positions to differ not only by translation but also slight vertical/skew difference.
    assert!((s1[1] - s2[1]).abs() > 0.001, "Siblings show no vertical divergence under perspective");
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