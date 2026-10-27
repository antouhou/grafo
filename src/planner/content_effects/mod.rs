use super::draws::{DrawPlanner, DrawPlanningInput};
use crate::commands::{CaptureSource, TextureComposite};
use crate::core::UnsignedPhysicalRect;
use crate::planner::commands::TraversalCommands;
use crate::planner::TextureIdAllocator;
use crate::scene_state::effects::parameters::EffectParameterStorage;
use crate::scene_state::effects::{BackdropEffectInstance, ContentEffectInstance};
use crate::scene_state::types::DrawTreeNode;
use crate::Size;
use ahash::HashMap;
pub(crate) use capture::ContentCapture;
use easy_tree::Tree;

mod capture;

pub(crate) struct ContentPlanningInput<'scene, 'planning> {
    pub tree: &'scene Tree<DrawTreeNode>,
    pub parameters: &'scene EffectParameterStorage,
    pub content_effects: &'scene HashMap<usize, ContentEffectInstance>,
    pub backdrop_effects: &'scene HashMap<usize, BackdropEffectInstance>,
    pub shape_mask_effects: &'planning HashMap<usize, TextureComposite>,
    pub scale_factor: f64,
    pub physical_size: Size,
}

/// Reuses backdrop bounds and traversal storage across command iterators.
#[derive(Default)]
pub(crate) struct SceneTraversal {
    subtree_backdrop_regions: HashMap<usize, UnsignedPhysicalRect>,
    node_order: Vec<usize>,
    draws: DrawPlanner,
}

impl SceneTraversal {
    pub(crate) fn start(&mut self, input: &ContentPlanningInput<'_, '_>) {
        self.resolve_backdrop_regions(input);
        self.draws.start(input.physical_size, CaptureSource::Target);
    }

    pub(crate) fn advance<'scene>(
        &mut self,
        input: &ContentPlanningInput<'scene, '_>,
        output: &mut TraversalCommands<'scene>,
        texture_ids: &mut TextureIdAllocator,
    ) -> bool {
        self.draws.advance(
            &DrawPlanningInput {
                tree: input.tree,
                parameters: input.parameters,
                shape_mask_effects: input.shape_mask_effects,
                content_effects: input.content_effects,
                backdrop_effects: input.backdrop_effects,
                subtree_backdrop_regions: &self.subtree_backdrop_regions,
                scale_factor: input.scale_factor,
                physical_size: input.physical_size,
            },
            output,
            texture_ids,
        )
    }

    fn resolve_backdrop_regions(&mut self, input: &ContentPlanningInput<'_, '_>) {
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
