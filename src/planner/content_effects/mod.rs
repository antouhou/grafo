use super::draws::{DrawPlanner, DrawPlanningInput};
pub(crate) use capture::ContentCapture;

mod capture;

impl DrawPlanner {
    pub(super) fn resolve_backdrop_regions(&mut self, input: &DrawPlanningInput<'_, '_>) {
        self.subtree_backdrop_regions.clear();
        self.node_order.clear();
        if input.content_effects.is_empty()
            || input.backdrop_effects.is_empty()
            || input.tree.get(0).is_none()
        {
            return;
        }
        self.node_order.push(0);
        let mut next_node = 0;
        while let Some(&node_id) = self.node_order.get(next_node) {
            self.node_order
                .extend_from_slice(input.tree.children(node_id));
            next_node += 1;
        }
        for &node_id in self.node_order.iter().rev() {
            let mut region = input
                .backdrop_effects
                .get(&node_id)
                .and_then(|effect| effect.capture_region)
                .and_then(|capture| capture.source_rect);
            for child_id in input.tree.children(node_id) {
                if let Some(child_region) = self.subtree_backdrop_regions.get(child_id) {
                    region =
                        Some(region.map_or(*child_region, |region| region.union(child_region)));
                }
            }
            if let Some(region) = region {
                self.subtree_backdrop_regions.insert(node_id, region);
            }
        }
    }
}

#[cfg(test)]
mod tests;
