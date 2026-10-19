use std::ops::Range;

#[derive(Clone, Copy)]
pub(super) struct RowEdge {
    pub x0: f32,
    pub x1: f32,
    pub signed_winding_height: f64,
}

impl RowEdge {
    fn coverage(&self, pixel: f64, inverse_width: f64) -> f64 {
        let a = self.x0.min(self.x1) as f64 - pixel;
        let b = self.x0.max(self.x1) as f64 - pixel;
        let area = if a >= 1.0 {
            1.0
        } else if b <= 0.0 {
            0.0
        } else if a >= 0.0 && b <= 1.0 {
            (a + b) * 0.5
        } else if a < 0.0 {
            if b <= 1.0 {
                b * b * 0.5 * inverse_width
            } else {
                (b - 0.5) * inverse_width
            }
        } else {
            1.0 - (1.0 - a) * (1.0 - a) * 0.5 * inverse_width
        };
        area * self.signed_winding_height
    }
}

#[derive(Default)]
pub(super) struct Scanline {
    changes: Vec<f64>,
    area: f64,
}

impl Scanline {
    pub(super) fn prepare(
        &mut self,
        edges: &[RowEdge],
        left: u32,
        right: u32,
        intervals: &[Range<u32>],
    ) {
        self.changes.resize(right as usize, 0.0);
        for interval in intervals {
            self.changes[interval.start as usize..interval.end as usize].fill(0.0);
        }
        self.area = 0.0;
        for edge in edges {
            let minimum = edge.x0.min(edge.x1);
            let maximum = edge.x0.max(edge.x1);
            if maximum <= left as f32 {
                continue;
            }
            if minimum >= right as f32 {
                self.area += edge.signed_winding_height;
                continue;
            }
            let inverse_width = if edge.x0 == edge.x1 {
                0.0
            } else {
                1.0 / (edge.x1 as f64 - edge.x0 as f64).abs()
            };
            let first = minimum.floor().clamp(left as f32, right as f32) as u32;
            let end = maximum.ceil().clamp(left as f32, right as f32) as u32;
            let mut previous = edge.coverage(left as f64 - 1.0, inverse_width);
            self.area += previous;
            // Store changes once per crossed pixel; a prefix sum combines all edges.
            for pixel in first..end {
                let next = edge.coverage(pixel as f64, inverse_width);
                self.changes[pixel as usize] += next - previous;
                previous = next;
            }
            if end < right {
                self.changes[end as usize] -= previous;
            }
        }
    }

    pub(super) fn fill_coverage(&mut self, left: u32, values: &mut [f32]) {
        let changes = &self.changes[left as usize..left as usize + values.len()];
        let mut area = self.area;
        for (&change, value) in changes.iter().zip(values) {
            area += change;
            *value = Self::clamp_coverage(area);
        }
        self.area = area;
    }

    pub(super) fn coverage(&self) -> f32 {
        Self::clamp_coverage(self.area)
    }

    fn clamp_coverage(area: f64) -> f32 {
        // Roundoff must not give empty pixels stencil coverage or allocate empty tiles.
        if area <= 1e-12 {
            0.0
        } else if area >= 1.0 - 1e-12 {
            1.0
        } else {
            area as f32
        }
    }
}
