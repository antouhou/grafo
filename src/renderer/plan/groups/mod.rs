use super::draws::{DrawPlanner, DrawPlanningInput, DrawTreeSelection};
use crate::commands::{
    BackdropCaptureSource, EffectApplication, IntermediateTextureId, RenderOperation, RenderPlan,
    Target, TextureComposite,
};
use crate::core::effect::{BackdropEffectInstance, EffectInstance};
use crate::renderer::types::DrawTreeNode;
use crate::Size;
use ahash::{HashMap, HashSet};
use easy_tree::Tree;
use std::cmp::Reverse;

fn node_depth(tree: &Tree<DrawTreeNode>, mut node: usize) -> usize {
    let mut depth = 0;
    while let Some(parent) = tree.parent_index_unchecked(node) {
        depth += 1;
        node = parent;
    }
    depth
}

pub(in crate::renderer) struct GroupPlanningInput<'a> {
    pub tree: &'a Tree<DrawTreeNode>,
    pub group_effects: &'a HashMap<usize, EffectInstance>,
    pub backdrop_effects: &'a HashMap<usize, BackdropEffectInstance>,
    pub shape_effects: &'a HashMap<usize, TextureComposite>,
    pub scale_factor: f64,
    pub physical_size: Size,
    pub max_capture_dimension: u32,
}

/// Appends dependency targets and scene draws to the planner's shared stream.
#[derive(Default)]
pub(in crate::renderer) struct SceneTraversal {
    groups: Vec<(usize, usize)>,
    results: HashMap<usize, IntermediateTextureId>,
    backdrop_ancestors: HashSet<usize>,
    draws: DrawPlanner,
}

impl SceneTraversal {
    pub fn plan(&mut self, input: GroupPlanningInput<'_>, output: &mut RenderPlan) {
        self.results.clear();
        self.schedule(&input);
        for index in 0..self.groups.len() {
            let node = self.groups[index].0;
            let backdrop_source = self.plan_backdrop_source(node, &input, output);
            let source = output.allocate_texture();
            self.append_target(
                Target::Texture {
                    texture: source,
                    size: input.physical_size,
                },
                DrawTreeSelection {
                    subtree_root: Some(node),
                    ..Default::default()
                },
                backdrop_source,
                &input,
                output,
            );
            let result = output.allocate_texture();
            let effect = &input.group_effects[&node];
            let parameters = output.store_parameters(&effect.params);
            output.push(RenderOperation::ApplyEffect(EffectApplication {
                effect_id: effect.effect_id,
                parameters,
                input: source,
                output: result,
            }));
            self.results.insert(node, result);
        }
        self.draws.append(
            DrawPlanningInput {
                tree: input.tree,
                selection: DrawTreeSelection::default(),
                effect_results: &self.results,
                shape_effects: input.shape_effects,
                group_effects: input.group_effects,
                backdrop_effects: input.backdrop_effects,
                scale_factor: input.scale_factor,
                physical_size: input.physical_size,
                backdrop_source: Some(BackdropCaptureSource::Target),
                max_capture_dimension: Some(input.max_capture_dimension),
            },
            output,
        );
    }

    fn schedule(&mut self, input: &GroupPlanningInput<'_>) {
        self.groups.clear();
        self.backdrop_ancestors.clear();
        // Layered captures need completed group dependencies outside the captured subtree.
        // Without captures, the tree walk opens group targets directly inside their parents.
        if input.group_effects.is_empty() || input.backdrop_effects.is_empty() {
            return;
        }
        for &node in input.group_effects.keys() {
            if input.tree.get(node).is_some() {
                self.groups.push((node, node_depth(input.tree, node)));
            }
        }
        // Descendant results must exist before ancestor scopes substitute them.
        self.groups
            .sort_unstable_by_key(|&(node, depth)| (Reverse(depth), node));
        for &node in input.backdrop_effects.keys() {
            if input.tree.get(node).is_none() {
                continue;
            }
            let mut ancestor = Some(node);
            while let Some(node) = ancestor {
                if !self.backdrop_ancestors.insert(node) {
                    break;
                }
                ancestor = input.tree.parent_index_unchecked(node);
            }
        }
    }

    fn plan_backdrop_source(
        &mut self,
        node: usize,
        input: &GroupPlanningInput<'_>,
        output: &mut RenderPlan,
    ) -> Option<BackdropCaptureSource> {
        if !self.backdrop_ancestors.contains(&node) {
            return None;
        }
        let base = output.allocate_texture();
        self.append_target(
            Target::Texture {
                texture: base,
                size: input.physical_size,
            },
            DrawTreeSelection {
                excluded_subtree: Some(node),
                ..Default::default()
            },
            None,
            input,
            output,
        );
        Some(BackdropCaptureSource::Layered { base })
    }

    fn append_target(
        &mut self,
        target: Target,
        selection: DrawTreeSelection,
        backdrop_source: Option<BackdropCaptureSource>,
        input: &GroupPlanningInput<'_>,
        output: &mut RenderPlan,
    ) {
        output.push(RenderOperation::BeginTarget(target));
        self.draws.append(
            DrawPlanningInput {
                tree: input.tree,
                selection,
                effect_results: &self.results,
                shape_effects: input.shape_effects,
                group_effects: input.group_effects,
                backdrop_effects: input.backdrop_effects,
                backdrop_source,
                scale_factor: input.scale_factor,
                physical_size: input.physical_size,
                max_capture_dimension: Some(input.max_capture_dimension),
            },
            output,
        );
        output.push(RenderOperation::EndTarget);
    }
}

#[cfg(test)]
mod tests;
