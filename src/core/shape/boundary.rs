use crate::core::vertex::CustomVertex;
use crate::core::MathRect;
use ahash::HashMap;
use std::{mem, ops::Range};

#[derive(Clone, Copy, Debug)]
pub(crate) struct PathBoundaryEdge {
    pub start: [f32; 2],
    pub end: [f32; 2],
    pub winding: i32,
}

/// Conservatively accepts rectangles inside a filled path without crossing any boundary.
pub(crate) fn path_contains_rect(edges: &[PathBoundaryEdge], bounds: MathRect) -> bool {
    let mut winding = 0_i64;
    for edge in edges {
        if !edge.start.into_iter().chain(edge.end).all(f32::is_finite) {
            return false;
        }
        if edge.start[0].min(edge.end[0]) <= bounds.max.x
            && edge.start[0].max(edge.end[0]) >= bounds.min.x
            && edge.start[1].min(edge.end[1]) <= bounds.max.y
            && edge.start[1].max(edge.end[1]) >= bounds.min.y
        {
            return false;
        }
        let crosses_up = edge.start[1] <= bounds.min.y && edge.end[1] > bounds.min.y;
        let crosses_down = edge.end[1] <= bounds.min.y && edge.start[1] > bounds.min.y;
        if !crosses_up && !crosses_down {
            continue;
        }
        let fraction = (f64::from(bounds.min.y) - f64::from(edge.start[1]))
            / (f64::from(edge.end[1]) - f64::from(edge.start[1]));
        let crossing = f64::from(edge.start[0])
            + fraction * (f64::from(edge.end[0]) - f64::from(edge.start[0]));
        if crossing > f64::from(bounds.min.x) {
            winding += if crosses_up {
                i64::from(edge.winding)
            } else {
                -i64::from(edge.winding)
            };
        }
    }
    winding != 0
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum BoundaryEdgeKind {
    LeftChain,
    RightChain,
    Horizontal,
}

fn classify_boundary_edge(edge: &PathBoundaryEdge) -> BoundaryEdgeKind {
    let direction = (edge.end[1] - edge.start[1]) * edge.winding as f32;
    if direction < 0.0 {
        BoundaryEdgeKind::LeftChain
    } else if direction > 0.0 {
        BoundaryEdgeKind::RightChain
    } else {
        BoundaryEdgeKind::Horizontal
    }
}

fn chain_limits(edges: &[PathBoundaryEdge]) -> Option<[f32; 2]> {
    let first = edges.first()?;
    let top = first.start[1].min(first.end[1]);
    let mut bottom = top;
    for edge in edges {
        if edge.winding.abs() != 1 || edge.start[1].min(edge.end[1]) != bottom {
            return None;
        }
        bottom = edge.start[1].max(edge.end[1]);
    }
    Some([top, bottom])
}

#[derive(Debug)]
pub(crate) struct RoundedRectBoundary {
    pub left: Range<usize>,
    pub right: Range<usize>,
}

impl RoundedRectBoundary {
    pub(crate) fn contains_rect(&self, edges: &[PathBoundaryEdge], bounds: MathRect) -> bool {
        for (chain, x, is_left) in [
            (&edges[self.left.clone()], bounds.min.x, true),
            (&edges[self.right.clone()], bounds.max.x, false),
        ] {
            for y in [bounds.min.y, bounds.max.y] {
                let index = chain.partition_point(|edge| edge.start[1].max(edge.end[1]) < y);
                let Some(edge) = chain.get(index) else {
                    return false;
                };
                if y < edge.start[1].min(edge.end[1]) {
                    return false;
                }
                let fraction = (f64::from(y) - f64::from(edge.start[1]))
                    / (f64::from(edge.end[1]) - f64::from(edge.start[1]));
                let boundary_x = f64::from(edge.start[0])
                    + fraction * (f64::from(edge.end[0]) - f64::from(edge.start[0]));
                if (is_left && f64::from(x) < boundary_x) || (!is_left && f64::from(x) > boundary_x)
                {
                    return false;
                }
            }
        }
        true
    }

    pub(crate) fn new(edges: &mut [PathBoundaryEdge]) -> Option<Self> {
        if !edges
            .iter()
            .all(|edge| edge.start.into_iter().chain(edge.end).all(f32::is_finite))
        {
            return None;
        }
        edges.sort_unstable_by(|a, b| {
            classify_boundary_edge(a)
                .cmp(&classify_boundary_edge(b))
                .then_with(|| {
                    a.start[1]
                        .min(a.end[1])
                        .total_cmp(&b.start[1].min(b.end[1]))
                })
        });
        let left_end = edges
            .partition_point(|edge| classify_boundary_edge(edge) == BoundaryEdgeKind::LeftChain);
        let right_end = edges
            .partition_point(|edge| classify_boundary_edge(edge) != BoundaryEdgeKind::Horizontal);
        let left = 0..left_end;
        let right = left_end..right_end;
        if chain_limits(&edges[left.clone()])? != chain_limits(&edges[right.clone()])? {
            return None;
        }
        Some(Self { left, right })
    }
}

type EdgeKey = ([u32; 2], [u32; 2]);

fn position_key(position: [f32; 2]) -> [u32; 2] {
    position.map(|value| if value == 0.0 { 0 } else { value.to_bits() })
}

#[derive(Default)]
pub(crate) struct PathBoundaryBuilder {
    edges: HashMap<EdgeKey, PathBoundaryEdge>,
    positions: Vec<[f32; 2]>,
}

impl PathBoundaryBuilder {
    pub(crate) fn build(
        &mut self,
        vertices: &[CustomVertex],
        indices: &[u32],
        project: impl Fn([f32; 2]) -> [f32; 2],
    ) -> impl Iterator<Item = PathBoundaryEdge> + '_ {
        self.edges.clear();
        self.positions.clear();
        self.positions
            .extend(vertices.iter().map(|vertex| project(vertex.position)));
        for triangle in indices.as_chunks::<3>().0 {
            let [a, mut b, mut c] = triangle.map(|index| self.positions[index as usize]);
            let area = (b[0] as f64 - a[0] as f64) * (c[1] as f64 - a[1] as f64)
                - (b[1] as f64 - a[1] as f64) * (c[0] as f64 - a[0] as f64);
            if !area.is_finite() || area == 0.0 {
                continue;
            }
            if area < 0.0 {
                mem::swap(&mut b, &mut c);
            }
            for (start, end) in [(a, b), (b, c), (c, a)] {
                let start_key = position_key(start);
                let end_key = position_key(end);
                let (key, edge) = if start_key < end_key {
                    (
                        (start_key, end_key),
                        PathBoundaryEdge {
                            start,
                            end,
                            winding: 1,
                        },
                    )
                } else {
                    (
                        (end_key, start_key),
                        PathBoundaryEdge {
                            start: end,
                            end: start,
                            winding: -1,
                        },
                    )
                };
                self.edges
                    .entry(key)
                    .and_modify(|entry| entry.winding += edge.winding)
                    .or_insert(edge);
            }
        }
        self.edges
            .values()
            .copied()
            .filter(|edge| edge.winding != 0)
    }
}
