use super::scanline::RowEdge;
use super::winding::RowSegment;
use rand_xoshiro::{
    rand_core::{Rng, SeedableRng},
    SplitMix64,
};

mod crossings;

#[derive(Default)]
struct ActiveEdge {
    children: [Option<usize>; 2],
    parent: Option<usize>,
    previous: Option<usize>,
    next: Option<usize>,
    priority: u64,
    segment: usize,
    winding_sum: i64,
    minimum_winding: i64,
    maximum_winding: i64,
    visible_count: usize,
    visible_sign: i8,
    visible_since: f64,
    crossing_y: f64,
    heap_position: Option<usize>,
}

/// The tree keeps edges in left-to-right order. Each edge gets a priority
/// number that is lower than the numbers of the edges directly below it.
/// Random-looking priorities help avoid a long chain of edges, where finding
/// or updating one edge could mean walking through most of the tree.
/// Seeding SplitMix64 with the index gives the same priority every time.
fn priority(index: usize) -> u64 {
    SplitMix64::seed_from_u64(index as u64).next_u64()
}

/// Retains active x order and one directly removable event per neighboring pair.
#[derive(Default)]
pub(super) struct WindingOrder {
    edges: Vec<ActiveEdge>,
    segment_positions: Vec<usize>,
    root: Option<usize>,
    crossings: Vec<usize>,
}

impl WindingOrder {
    pub(super) fn clear(&mut self, segment_count: usize) {
        self.edges.clear();
        self.edges.resize_with(segment_count, ActiveEdge::default);
        self.segment_positions.resize(segment_count, 0);
        self.root = None;
        self.crossings.clear();
    }

    fn sum(&self, node: Option<usize>) -> i64 {
        node.map_or(0, |index| self.edges[index].winding_sum)
    }

    fn update(&mut self, index: usize, segments: &[RowSegment]) {
        let [left, right] = self.edges[index].children;
        let middle = self.sum(left) + segments[self.edges[index].segment].winding;
        self.edges[index].winding_sum = middle + self.sum(right);
        self.edges[index].minimum_winding = left
            .map_or(0, |child| self.edges[child].minimum_winding)
            .min(middle + right.map_or(0, |child| self.edges[child].minimum_winding));
        self.edges[index].maximum_winding = left
            .map_or(0, |child| self.edges[child].maximum_winding)
            .max(middle + right.map_or(0, |child| self.edges[child].maximum_winding));
        self.edges[index].visible_count = usize::from(self.edges[index].visible_sign != 0)
            + left.map_or(0, |child| self.edges[child].visible_count)
            + right.map_or(0, |child| self.edges[child].visible_count);
    }

    fn update_ancestors(&mut self, mut index: Option<usize>, segments: &[RowSegment]) {
        while let Some(current) = index {
            self.update(current, segments);
            index = self.edges[current].parent;
        }
    }

    fn rotate_up(&mut self, child: usize, segments: &[RowSegment]) {
        let Some(parent) = self.edges[child].parent else {
            return;
        };
        let grandparent = self.edges[parent].parent;
        let direction = usize::from(self.edges[parent].children[1] == Some(child));
        let middle = self.edges[child].children[1 - direction];
        self.edges[parent].children[direction] = middle;
        if let Some(middle) = middle {
            self.edges[middle].parent = Some(parent);
        }
        self.edges[child].children[1 - direction] = Some(parent);
        self.edges[parent].parent = Some(child);
        self.edges[child].parent = grandparent;
        if let Some(grandparent) = grandparent {
            let direction = usize::from(self.edges[grandparent].children[1] == Some(parent));
            self.edges[grandparent].children[direction] = Some(child);
        } else {
            self.root = Some(child);
        }
        self.update(parent, segments);
        self.update(child, segments);
    }

    fn winding_before(&self, index: usize, segments: &[RowSegment]) -> i64 {
        let mut winding = self.sum(self.edges[index].children[0]);
        let mut child = index;
        while let Some(parent) = self.edges[child].parent {
            if self.edges[parent].children[1] == Some(child) {
                winding += self.sum(self.edges[parent].children[0])
                    + segments[self.edges[parent].segment].winding;
            }
            child = parent;
        }
        winding
    }

    fn set_visible(
        &mut self,
        index: usize,
        sign: i8,
        y: f64,
        segments: &[RowSegment],
        output: &mut Vec<RowEdge>,
    ) {
        let edge = &mut self.edges[index];
        if edge.visible_sign == sign {
            return;
        }
        if edge.visible_sign != 0 && y > edge.visible_since {
            let segment = segments[edge.segment];
            output.push(RowEdge {
                x0: segment.x_at(edge.visible_since) as f32,
                x1: segment.x_at(y) as f32,
                signed_winding_height: (y - edge.visible_since) * f64::from(edge.visible_sign),
            });
        }
        edge.visible_sign = sign;
        edge.visible_since = y;
    }

    fn classify(
        &mut self,
        index: usize,
        winding: i64,
        y: f64,
        segments: &[RowSegment],
        output: &mut Vec<RowEdge>,
    ) {
        let after = winding + segments[self.edges[index].segment].winding;
        let sign = if winding == 0 && after != 0 {
            -1
        } else if winding != 0 && after == 0 {
            1
        } else {
            0
        };
        self.set_visible(index, sign, y, segments, output);
    }

    pub(super) fn next_crossing(&self) -> Option<f64> {
        self.crossings
            .first()
            .map(|&index| self.edges[index].crossing_y)
    }

    pub(super) fn initialize(
        &mut self,
        ordered_segments: impl Iterator<Item = usize>,
        y: f64,
        segments: &[RowSegment],
        output: &mut Vec<RowEdge>,
    ) {
        let mut previous: Option<usize> = None;
        let mut first = None;
        let mut winding = 0;
        for index in ordered_segments {
            let priority = priority(index);
            let mut parent = previous;
            let mut left = None;
            while let Some(ancestor) = parent {
                if self.edges[ancestor].priority <= priority {
                    break;
                }
                left = parent;
                parent = self.edges[ancestor].parent;
            }
            self.edges[index] = ActiveEdge {
                children: [left, None],
                parent,
                previous,
                priority,
                segment: index,
                ..ActiveEdge::default()
            };
            self.segment_positions[index] = index;
            if let Some(left) = left {
                self.edges[left].parent = Some(index);
            }
            if let Some(parent) = parent {
                self.edges[parent].children[1] = Some(index);
            } else {
                self.root = Some(index);
            }
            if let Some(previous) = previous {
                self.edges[previous].next = Some(index);
            } else {
                first = Some(index);
            }
            self.classify(index, winding, y, segments, output);
            winding += segments[index].winding;
            previous = Some(index);
        }
        if let Some(root) = self.root {
            self.update_subtree(root, segments);
        }
        while let Some(index) = first {
            self.schedule_crossing(index, y, segments);
            first = self.edges[index].next;
        }
    }

    fn update_subtree(&mut self, index: usize, segments: &[RowSegment]) {
        for child in self.edges[index].children.into_iter().flatten() {
            self.update_subtree(child, segments);
        }
        self.update(index, segments);
    }

    pub(super) fn finish(&mut self, y: f64, segments: &[RowSegment], output: &mut Vec<RowEdge>) {
        if let Some(root) = self.root {
            self.finish_subtree(root, y, segments, output);
        }
    }

    fn finish_subtree(
        &mut self,
        index: usize,
        y: f64,
        segments: &[RowSegment],
        output: &mut Vec<RowEdge>,
    ) {
        if self.edges[index].visible_count == 0 {
            return;
        }
        for child in self.edges[index].children.into_iter().flatten() {
            self.finish_subtree(child, y, segments, output);
        }
        self.set_visible(index, 0, y, segments, output);
    }

    pub(super) fn insert(
        &mut self,
        segment: usize,
        y: f64,
        segments: &[RowSegment],
        output: &mut Vec<RowEdge>,
    ) -> [f64; 2] {
        let mut parent = None;
        let mut current = self.root;
        let mut direction = 0;
        let mut previous = None;
        let mut next = None;
        while let Some(index) = current {
            parent = current;
            let other = self.edges[index].segment;
            let order = segments[segment]
                .x_at(y)
                .total_cmp(&segments[other].x_at(y))
                .then_with(|| {
                    segments[segment]
                        .slope()
                        .total_cmp(&segments[other].slope())
                })
                .then_with(|| segment.cmp(&other));
            direction = usize::from(order.is_gt());
            if direction == 0 {
                next = current;
            } else {
                previous = current;
            }
            current = self.edges[index].children[direction];
        }
        let index = segment;
        self.edges[index] = ActiveEdge {
            parent,
            previous,
            next,
            segment,
            priority: priority(index),
            ..ActiveEdge::default()
        };
        self.segment_positions[segment] = index;
        if let Some(parent) = parent {
            self.edges[parent].children[direction] = Some(index);
        } else {
            self.root = Some(index);
        }
        if let Some(previous) = previous {
            self.edges[previous].next = Some(index);
        }
        if let Some(next) = next {
            self.edges[next].previous = Some(index);
        }
        self.update_ancestors(Some(index), segments);
        while let Some(parent) = self.edges[index].parent {
            if self.edges[parent].priority <= self.edges[index].priority {
                break;
            }
            self.rotate_up(index, segments);
        }
        let winding = self.winding_before(index, segments);
        self.classify(index, winding, y, segments, output);
        self.update_ancestors(Some(index), segments);
        if let Some(previous) = previous {
            self.schedule_crossing(previous, y, segments);
        }
        self.schedule_crossing(index, y, segments);
        self.neighbor_bounds(index, y, segments)
    }

    pub(super) fn remove(
        &mut self,
        segment: usize,
        y: f64,
        segments: &[RowSegment],
        output: &mut Vec<RowEdge>,
    ) -> [f64; 2] {
        let index = self.segment_positions[segment];
        let bounds = self.neighbor_bounds(index, y, segments);
        self.set_visible(index, 0, y, segments, output);
        self.cancel_crossing(index);
        let previous = self.edges[index].previous;
        let next = self.edges[index].next;
        if let Some(previous) = previous {
            self.edges[previous].next = next;
        }
        if let Some(next) = next {
            self.edges[next].previous = previous;
        }
        while self.edges[index].children != [None, None] {
            let child = match self.edges[index].children {
                [Some(left), Some(right)] => {
                    if self.edges[left].priority < self.edges[right].priority {
                        left
                    } else {
                        right
                    }
                }
                [Some(child), None] | [None, Some(child)] => child,
                [None, None] => unreachable!(),
            };
            self.rotate_up(child, segments);
        }
        let parent = self.edges[index].parent;
        if let Some(parent) = parent {
            let direction = usize::from(self.edges[parent].children[1] == Some(index));
            self.edges[parent].children[direction] = None;
        } else {
            self.root = None;
        }
        self.update_ancestors(parent, segments);
        if let Some(previous) = previous {
            self.schedule_crossing(previous, y, segments);
        }
        bounds
    }

    fn neighbor_bounds(&self, index: usize, y: f64, segments: &[RowSegment]) -> [f64; 2] {
        let edge = &self.edges[index];
        let x = segments[edge.segment].x_at(y);
        let mut bounds = [x, x];
        for neighbor in [edge.previous, edge.next].into_iter().flatten() {
            let neighbor_x = segments[self.edges[neighbor].segment].x_at(y);
            bounds[0] = bounds[0].min(neighbor_x);
            bounds[1] = bounds[1].max(neighbor_x);
        }
        bounds
    }

    pub(super) fn cross(&mut self, y: f64, segments: &[RowSegment], output: &mut Vec<RowEdge>) {
        let Some(&left) = self.crossings.first() else {
            return;
        };
        let Some(right) = self.edges[left].next else {
            return;
        };
        let previous = self.edges[left].previous;
        // The swap only changes the winding between these two segments.
        let winding = self.winding_before(left, segments);
        let payload = |edge: &ActiveEdge| (edge.segment, edge.visible_since, edge.visible_sign);
        let first = payload(&self.edges[left]);
        let second = payload(&self.edges[right]);
        (
            self.edges[left].segment,
            self.edges[left].visible_since,
            self.edges[left].visible_sign,
        ) = second;
        (
            self.edges[right].segment,
            self.edges[right].visible_since,
            self.edges[right].visible_sign,
        ) = first;
        self.segment_positions[first.0] = right;
        self.segment_positions[second.0] = left;
        self.classify(left, winding, y, segments, output);
        self.classify(
            right,
            winding + segments[second.0].winding,
            y,
            segments,
            output,
        );
        self.update_ancestors(Some(left), segments);
        self.update_ancestors(Some(right), segments);
        if let Some(previous) = previous {
            self.schedule_crossing(previous, y, segments);
        }
        self.schedule_crossing(left, y, segments);
        self.schedule_crossing(right, y, segments);
    }

    pub(super) fn refresh(
        &mut self,
        range: [f64; 2],
        y: f64,
        segments: &[RowSegment],
        output: &mut Vec<RowEdge>,
    ) {
        if let Some(root) = self.root {
            self.refresh_subtree(
                root,
                0,
                [f64::NEG_INFINITY, f64::INFINITY],
                range,
                y,
                segments,
                output,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn refresh_subtree(
        &mut self,
        index: usize,
        winding: i64,
        bounds: [f64; 2],
        range: [f64; 2],
        y: f64,
        segments: &[RowSegment],
        output: &mut Vec<RowEdge>,
    ) {
        if bounds[1] < range[0] || bounds[0] > range[1] {
            return;
        }
        let edge = &self.edges[index];
        let cannot_cross_zero =
            winding + edge.minimum_winding > 0 || winding + edge.maximum_winding < 0;
        if cannot_cross_zero && edge.visible_count == 0 {
            return;
        }
        let [left, right] = edge.children;
        let x = segments[edge.segment].x_at(y);
        let middle = winding + self.sum(left);
        let after = middle + segments[edge.segment].winding;
        if let Some(left) = left {
            self.refresh_subtree(left, winding, [bounds[0], x], range, y, segments, output);
        }
        if x >= range[0] && x <= range[1] {
            self.classify(index, middle, y, segments, output);
        }
        if let Some(right) = right {
            self.refresh_subtree(right, after, [x, bounds[1]], range, y, segments, output);
        }
        self.update(index, segments);
    }
}
