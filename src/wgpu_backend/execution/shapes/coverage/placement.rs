use crate::core::vertex::InstanceTransform;
use crate::core::Viewport;

pub(super) struct CoveragePlacement {
    pub transform: Option<InstanceTransform>,
    pub translation: [f32; 2],
}

impl CoveragePlacement {
    pub(super) fn new(
        transform: Option<InstanceTransform>,
        bounds: [(f32, f32); 2],
        viewport: Viewport,
    ) -> Self {
        let original = Self {
            transform,
            translation: [0.0; 2],
        };
        let mut normalized = transform.unwrap_or_else(InstanceTransform::identity);
        if normalized.col0[3] != 0.0 || normalized.col1[3] != 0.0 || normalized.col3[3] != 1.0 {
            return original;
        }
        let mut minimum = [f32::INFINITY; 2];
        for point in [
            bounds[0],
            (bounds[1].0, bounds[0].1),
            bounds[1],
            (bounds[0].0, bounds[1].1),
        ] {
            let point = [0, 1].map(|axis| {
                ((normalized.col0[axis] as f64 * point.0 as f64
                    + normalized.col1[axis] as f64 * point.1 as f64
                    + normalized.col3[axis] as f64)
                    * viewport.scale_factor) as f32
            });
            if !point.iter().all(|value| value.is_finite())
                || point[0] < 0.0
                || point[1] < 0.0
                || point[0] > viewport.physical_size.0 as f32
                || point[1] > viewport.physical_size.1 as f32
            {
                return original;
            }
            minimum[0] = minimum[0].min(point[0]);
            minimum[1] = minimum[1].min(point[1]);
        }
        let translation = minimum.map(f32::floor);
        for (axis, offset) in translation.iter().enumerate() {
            normalized.col3[axis] =
                (normalized.col3[axis] as f64 - *offset as f64 / viewport.scale_factor) as f32;
        }
        Self {
            transform: Some(normalized),
            translation,
        }
    }
}
