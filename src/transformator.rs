use euclid::{Point2D, Transform3D, UnknownUnit};
use crate::{InstanceRenderParams, TransformInstance};

#[derive(Clone, Debug)]
pub struct Transform {
    /// Local transform relative to parent
    pub local_transform: Transform3D<f32, UnknownUnit, UnknownUnit>,
    /// Fully composed world transform including all parent transforms
    pub world_transform: Transform3D<f32, UnknownUnit, UnknownUnit>,
    /// In CSS that's usually a parent's perspective property
    pub camera_perspective_distance: f32,
    /// Origin relative to the shape
    pub origin: (f32, f32),
    /// TODO: figure that one out
    pub camera_perspective_origin: (f32, f32),
    pub position_relative_to_parent: (f32, f32),
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
        let t_from_origin: Transform3D<f32, UnknownUnit, UnknownUnit> = Transform3D::translation(-ox, -oy, 0.0);
        let t_to_final: Transform3D<f32, UnknownUnit, UnknownUnit> = Transform3D::translation(px + ox, py + oy, 0.0);

        self.world_transform = parent.world_transform
            .then(&t_from_origin)
            .then(&self.local_transform)
            .then(&t_to_final);
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
}

#[test]
pub fn test() {
    let viewport_center = (400.0, 300.0);
    let transform = Transform::new()
        .with_position_relative_to_parent(viewport_center.0 - 50.0, viewport_center.1 - 50.0)
        .with_camera_perspective_origin(viewport_center.0, viewport_center.1)
        .with_perspective_distance(500.0)
        .with_origin(50.0, 50.0)
        .then_rotate_x(45.0)
        // Use viewport position to position the shape at screen center after perspective is applied.
        .compose_2(&Transform::new());


    let p = apply_transform(
        [50.0, 50.0], // local center == origin
        &TransformInstance::from_columns(transform.cols_world()),
        &InstanceRenderParams {
            camera_perspective: 0.0,         // <— IMPORTANT
            camera_perspective_origin: [0.0, 0.0],   // unused when no perspective
            _padding: 0.0,
        },
    );
    println!("p = {:?}", p);

    let point = apply_transform(
        [50.0, 50.0],
        &TransformInstance::from_columns(transform.cols_world()),
        &InstanceRenderParams {
            camera_perspective: transform.camera_perspective_distance,
            camera_perspective_origin: [transform.camera_perspective_origin.0, transform.camera_perspective_origin.1],
            _padding: 0.0,
        }
    );
    println!("{:?}", point);

    // Assert center stability with and without perspective.
    let eps = 0.0001;
    assert!((p[0] - viewport_center.0).abs() < eps && (p[1] - viewport_center.1).abs() < eps,
        "Center moved without perspective: got {:?} expected {:?}", p, viewport_center);
    assert!((point[0] - viewport_center.0).abs() < eps && (point[1] - viewport_center.1).abs() < eps,
        "Center moved with perspective: got {:?} expected {:?}", point, viewport_center);
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

fn apply_transform(
    position: [f32; 2],
    transform: &TransformInstance,
    render_params: &InstanceRenderParams,
) -> [f32; 2] {
    // 1) Apply world/model transform
    let p = mul_vec4(transform, [position[0], position[1], 0.0, 1.0]);

    // 2) No perspective: just return canvas-space coords
    if render_params.camera_perspective <= 0.0 {
        return [p[0], p[1]];
    }

    // 3) Perspective case
    let cx = render_params.camera_perspective_origin[0]; // canvas-space origin
    let cy = render_params.camera_perspective_origin[1];

    let x_rel = p[0] - cx;
    let y_rel = p[1] - cy;

    let w = 1.0 + p[2] / render_params.camera_perspective;
    let invw = 1.0 / f32::max(w.abs(), 1e-6);

    let px = x_rel * invw + cx;
    let py = y_rel * invw + cy;

    [px, py]
}