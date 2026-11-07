use super::{RowSegment, WindingOrder};
use std::cmp::Ordering;

impl WindingOrder {
    fn event_order(&self, first: usize, second: usize) -> Ordering {
        self.edges[first]
            .crossing_y
            .total_cmp(&self.edges[second].crossing_y)
            .then_with(|| first.cmp(&second))
    }

    fn heap_swap(&mut self, first: usize, second: usize) {
        self.crossings.swap(first, second);
        self.edges[self.crossings[first]].heap_position = Some(first);
        self.edges[self.crossings[second]].heap_position = Some(second);
    }

    fn sift_up(&mut self, mut position: usize) -> usize {
        while position > 0 {
            let parent = (position - 1) / 2;
            if self
                .event_order(self.crossings[parent], self.crossings[position])
                .is_le()
            {
                break;
            }
            self.heap_swap(parent, position);
            position = parent;
        }
        position
    }

    fn sift_down(&mut self, mut position: usize) {
        loop {
            let left = position * 2 + 1;
            if left >= self.crossings.len() {
                break;
            }
            let right = left + 1;
            let child = if right < self.crossings.len()
                && self
                    .event_order(self.crossings[right], self.crossings[left])
                    .is_lt()
            {
                right
            } else {
                left
            };
            if self
                .event_order(self.crossings[position], self.crossings[child])
                .is_le()
            {
                break;
            }
            self.heap_swap(position, child);
            position = child;
        }
    }

    pub(super) fn cancel_crossing(&mut self, index: usize) {
        let Some(position) = self.edges[index].heap_position.take() else {
            return;
        };
        let last = self
            .crossings
            .pop()
            .expect("an indexed crossing is present");
        if position < self.crossings.len() {
            self.crossings[position] = last;
            self.edges[last].heap_position = Some(position);
            let position = self.sift_up(position);
            self.sift_down(position);
        }
    }

    pub(super) fn schedule_crossing(&mut self, index: usize, y: f64, segments: &[RowSegment]) {
        self.cancel_crossing(index);
        let Some(next) = self.edges[index].next else {
            return;
        };
        let left = segments[self.edges[index].segment];
        let right = segments[self.edges[next].segment];
        let relative_slope = left.slope() - right.slope();
        if relative_slope <= 0.0 {
            return;
        }
        let crossing_y = (y + (right.x_at(y) - left.x_at(y)) / relative_slope).max(y);
        if crossing_y >= left.end_y.min(right.end_y) {
            return;
        }
        self.edges[index].crossing_y = crossing_y;
        self.edges[index].heap_position = Some(self.crossings.len());
        self.crossings.push(index);
        self.sift_up(self.crossings.len() - 1);
    }
}
