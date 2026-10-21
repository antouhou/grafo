use super::geometry::{SOLID_COVERAGE, TILE_SIZE};
use super::projection::Projection;
use super::tiles::{CoverageDrawMetadata, CoverageDrawRange, CoverageGeometry, CoverageTile};
use crate::core::UnsignedPhysicalRect;

#[derive(Clone, Copy, Default)]
struct AxisSpan {
    start: u32,
    end: u32,
    coverage: f32,
}

fn axis_spans(minimum: f32, maximum: f32, limit: u32) -> [AxisSpan; 3] {
    let minimum = minimum.clamp(0.0, limit as f32);
    let maximum = maximum.clamp(0.0, limit as f32);
    let mut spans = [AxisSpan::default(); 3];
    if minimum >= maximum {
        return spans;
    }
    let first = minimum.floor() as u32;
    let end = maximum.ceil() as u32;
    if end - first == 1 {
        spans[0] = AxisSpan {
            start: first,
            end,
            coverage: maximum - minimum,
        };
        return spans;
    }
    let full_start = minimum.ceil() as u32;
    let full_end = maximum.floor() as u32;
    spans[0] = AxisSpan {
        start: first,
        end: full_start,
        coverage: full_start as f32 - minimum,
    };
    spans[1] = AxisSpan {
        start: full_start,
        end: full_end,
        coverage: 1.0,
    };
    spans[2] = AxisSpan {
        start: full_end,
        end,
        coverage: maximum - full_end as f32,
    };
    spans
}

fn append_patch(horizontal: AxisSpan, vertical: AxisSpan, geometry: &mut CoverageGeometry) {
    let width = horizontal.end - horizontal.start;
    let height = vertical.end - vertical.start;
    let value = horizontal.coverage * vertical.coverage;
    if value <= 1e-12 {
        return;
    }
    let projection_index = geometry.projections.len() as u32 - 1;
    if value == 1.0 {
        geometry.tiles.push(CoverageTile {
            min: [horizontal.start, vertical.start],
            max: [horizontal.end, vertical.end],
            coverage_start: SOLID_COVERAGE,
            projection_index,
        });
        return;
    }
    let coverage_start = geometry.coverage.len() as u32;
    let sample_count = (height.min(TILE_SIZE) - 1) * TILE_SIZE + width.min(TILE_SIZE);
    geometry
        .coverage
        .resize(geometry.coverage.len() + sample_count as usize, value);
    for top in (vertical.start..vertical.end).step_by(TILE_SIZE as usize) {
        for left in (horizontal.start..horizontal.end).step_by(TILE_SIZE as usize) {
            geometry.tiles.push(CoverageTile {
                min: [left, top],
                max: [
                    (left + TILE_SIZE).min(horizontal.end),
                    (top + TILE_SIZE).min(vertical.end),
                ],
                coverage_start,
                projection_index,
            });
        }
    }
}

pub(super) fn build(
    local_bounds: [(f32, f32); 2],
    projection: &Projection,
    size: (u32, u32),
    geometry: &mut CoverageGeometry,
) -> Option<CoverageDrawMetadata> {
    let mut result = CoverageDrawMetadata {
        range: CoverageDrawRange {
            tile_start: geometry.tiles.len() as u32,
            ..Default::default()
        },
        bounds: UnsignedPhysicalRect::zero(),
    };
    if !projection.preserves_axis_alignment() {
        return None;
    }
    let corners = local_bounds.map(|(x, y)| projection.project([x, y]));
    if !corners.into_iter().flatten().all(f32::is_finite) {
        return None;
    }
    let horizontal = axis_spans(
        corners[0][0].min(corners[1][0]),
        corners[0][0].max(corners[1][0]),
        size.0,
    );
    let vertical = axis_spans(
        corners[0][1].min(corners[1][1]),
        corners[0][1].max(corners[1][1]),
        size.1,
    );
    let mut horizontal_bounds = horizontal.iter().filter(|span| span.start < span.end);
    let mut vertical_bounds = vertical.iter().filter(|span| span.start < span.end);
    let (Some(first_x), Some(first_y)) = (
        horizontal_bounds.clone().next(),
        vertical_bounds.clone().next(),
    ) else {
        return Some(result);
    };
    let end_x = horizontal_bounds
        .next_back()
        .expect("horizontal span exists")
        .end;
    let end_y = vertical_bounds
        .next_back()
        .expect("vertical span exists")
        .end;
    result.bounds =
        UnsignedPhysicalRect::new((first_x.start, first_y.start).into(), (end_x, end_y).into());
    geometry
        .projections
        .push(projection.mapping([first_x.start, first_y.start], local_bounds));
    for is_solid in [true, false] {
        for horizontal in horizontal
            .iter()
            .copied()
            .filter(|span| span.start < span.end)
        {
            for vertical in vertical
                .iter()
                .copied()
                .filter(|span| span.start < span.end)
            {
                if (horizontal.coverage * vertical.coverage == 1.0) == is_solid {
                    append_patch(horizontal, vertical, geometry);
                }
            }
        }
        if is_solid {
            result.range.solid_tile_count = geometry.tiles.len() as u32 - result.range.tile_start;
        }
    }
    result.range.tile_count = geometry.tiles.len() as u32 - result.range.tile_start;
    Some(result)
}
