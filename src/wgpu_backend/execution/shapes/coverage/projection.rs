use super::tiles::CoverageProjection;
use crate::core::vertex::InstanceTransform;

pub(super) struct Projection {
    matrix: [[f64; 3]; 3],
    inverse: [[f64; 3]; 3],
    scale: f64,
    homogeneous_w_sign: f64,
    pub(super) boundary_orientation: Option<i32>,
}

impl Projection {
    pub(super) fn new(
        transform: Option<InstanceTransform>,
        scale: f64,
        bounds: [(f32, f32); 2],
    ) -> Option<Self> {
        let transform = transform.unwrap_or_else(InstanceTransform::identity);
        let mut matrix = [
            [
                transform.col0[0] as f64,
                transform.col1[0] as f64,
                transform.col3[0] as f64,
            ],
            [
                transform.col0[1] as f64,
                transform.col1[1] as f64,
                transform.col3[1] as f64,
            ],
            [
                transform.col0[3] as f64,
                transform.col1[3] as f64,
                transform.col3[3] as f64,
            ],
        ];
        let corners = [
            bounds[0],
            (bounds[1].0, bounds[0].1),
            bounds[1],
            (bounds[0].0, bounds[1].1),
        ];
        if corners.iter().all(|&(x, y)| {
            (matrix[2][0] * x as f64 + matrix[2][1] * y as f64 + matrix[2][2]).abs() <= 1e-6
        }) {
            matrix[2] = [0.0, 0.0, 1e-6];
        }
        let [a, b, c] = matrix;
        let cofactors = [
            [
                b[1] * c[2] - b[2] * c[1],
                a[2] * c[1] - a[1] * c[2],
                a[1] * b[2] - a[2] * b[1],
            ],
            [
                b[2] * c[0] - b[0] * c[2],
                a[0] * c[2] - a[2] * c[0],
                a[2] * b[0] - a[0] * b[2],
            ],
            [
                b[0] * c[1] - b[1] * c[0],
                a[1] * c[0] - a[0] * c[1],
                a[0] * b[1] - a[1] * b[0],
            ],
        ];
        let determinant = a[0] * cofactors[0][0] + a[1] * cofactors[1][0] + a[2] * cofactors[2][0];
        if !determinant.is_finite() || determinant == 0.0 {
            return None;
        }
        let center = [
            (bounds[0].0 as f64 + bounds[1].0 as f64) * 0.5,
            (bounds[0].1 as f64 + bounds[1].1 as f64) * 0.5,
        ];
        let homogeneous_w_sign = (c[0] * center[0] + c[1] * center[1] + c[2]).signum();
        let boundary_orientation = corners
            .iter()
            .all(|&(x, y)| (c[0] * x as f64 + c[1] * y as f64 + c[2]) * homogeneous_w_sign >= 1e-6)
            .then_some((determinant * homogeneous_w_sign).signum() as i32);
        Some(Self {
            matrix,
            inverse: cofactors.map(|row| row.map(|value| value / determinant)),
            scale,
            homogeneous_w_sign,
            boundary_orientation,
        })
    }

    pub(super) fn project(&self, position: [f32; 2]) -> [f32; 2] {
        let point = [position[0] as f64, position[1] as f64, 1.0];
        let projected = self
            .matrix
            .map(|row| row.into_iter().zip(point).map(|(a, b)| a * b).sum::<f64>());
        let factor = self.scale / projected[2].abs().max(1e-6);
        [
            (projected[0] * factor) as f32,
            (projected[1] * factor) as f32,
        ]
    }

    pub(super) fn mapping(
        &self,
        origin: [u32; 2],
        local_bounds: [(f32, f32); 2],
    ) -> CoverageProjection {
        let point = [
            origin[0] as f64 / self.scale * self.homogeneous_w_sign,
            origin[1] as f64 / self.scale * self.homogeneous_w_sign,
            1.0,
        ];
        let model_origin = self
            .inverse
            .map(|row| row.into_iter().zip(point).map(|(a, b)| a * b).sum::<f64>() as f32);
        let factor = self.homogeneous_w_sign / self.scale;
        let dx = self.inverse.map(|row| (row[0] * factor) as f32);
        let dy = self.inverse.map(|row| (row[1] * factor) as f32);
        CoverageProjection {
            model_origin,
            screen_origin_x: origin[0] as f32,
            model_dx: dx,
            screen_origin_y: origin[1] as f32,
            model_dy: dy,
            padding: 0.0,
            local_origin: [local_bounds[0].0, local_bounds[0].1],
            local_size: [
                (local_bounds[1].0 - local_bounds[0].0).max(1e-6),
                (local_bounds[1].1 - local_bounds[0].1).max(1e-6),
            ],
        }
    }
}
