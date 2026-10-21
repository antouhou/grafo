use super::draws::{DrawPlanner, DrawPlanningInput, DrawTreeSelection};
use crate::commands::{CaptureSource, RenderOperation, Target, TextureComposite};
use crate::planner::commands::TraversalCommands;
use crate::planner::TextureIdAllocator;
use crate::scene_state::effects::parameters::EffectParameterStorage;
use crate::scene_state::effects::{BackdropEffectInstance, ContentEffectInstance};
use crate::scene_state::types::DrawTreeNode;
use crate::Size;
use ahash::{HashMap, HashSet};
pub(crate) use capture::ContentCapture;
use easy_tree::Tree;
use std::cmp::Reverse;

mod capture;

fn node_depth(tree: &Tree<DrawTreeNode>, mut node: usize) -> usize {
    let mut depth = 0;
    while let Some(parent) = tree.parent_index_unchecked(node) {
        depth += 1;
        node = parent;
    }
    depth
}

pub(crate) struct ContentPlanningInput<'scene, 'planning> {
    pub tree: &'scene Tree<DrawTreeNode>,
    pub parameters: &'scene EffectParameterStorage,
    pub content_effects: &'scene HashMap<usize, ContentEffectInstance>,
    pub backdrop_effects: &'scene HashMap<usize, BackdropEffectInstance>,
    pub shape_mask_effects: &'planning HashMap<usize, TextureComposite>,
    pub scale_factor: f64,
    pub physical_size: Size,
}

enum ContentTask {
    PlanContent(usize),
    BeginContent(usize, Option<CaptureSource>),
    FinishContent(usize, ContentCapture),
    BeginTarget(Target),
    EndTarget,
    DrawTree(DrawTreeSelection, Option<CaptureSource>),
}

/// Reuses dependency scheduling and traversal storage across command iterators.
#[derive(Default)]
pub(crate) struct SceneTraversal {
    groups: Vec<(usize, usize)>,
    results: HashMap<usize, TextureComposite>,
    backdrop_ancestors: HashSet<usize>,
    draws: DrawPlanner,
    tasks: Vec<ContentTask>,
    is_drawing: bool,
    selection: DrawTreeSelection,
    backdrop_source: Option<CaptureSource>,
}

impl SceneTraversal {
    pub(crate) fn start(&mut self, input: &ContentPlanningInput<'_, '_>) {
        self.results.clear();
        self.tasks.clear();
        self.is_drawing = false;
        self.schedule(input);
        self.tasks.push(ContentTask::DrawTree(
            DrawTreeSelection::default(),
            Some(CaptureSource::Target),
        ));
        for &(node_id, _) in self.groups.iter().rev() {
            self.tasks.push(ContentTask::PlanContent(node_id));
        }
    }

    pub(crate) fn advance<'scene>(
        &mut self,
        input: &ContentPlanningInput<'scene, '_>,
        output: &mut TraversalCommands<'scene>,
        texture_ids: &mut TextureIdAllocator,
    ) -> bool {
        if self.is_drawing {
            let progressed = self.draws.advance(
                &DrawPlanningInput {
                    tree: input.tree,
                    parameters: input.parameters,
                    selection: self.selection,
                    effect_results: &self.results,
                    shape_mask_effects: input.shape_mask_effects,
                    content_effects: input.content_effects,
                    backdrop_effects: input.backdrop_effects,
                    backdrop_source: self.backdrop_source,
                    scale_factor: input.scale_factor,
                    physical_size: input.physical_size,
                },
                output,
                texture_ids,
            );
            self.is_drawing = progressed;
            return true;
        }
        let Some(task) = self.tasks.pop() else {
            return false;
        };
        match task {
            ContentTask::PlanContent(node_id) => {
                if input.content_effects[&node_id].bounds.is_none() {
                    return true;
                }
                if self.backdrop_ancestors.contains(&node_id) {
                    let base = texture_ids.allocate();
                    self.tasks.push(ContentTask::BeginContent(
                        node_id,
                        Some(CaptureSource::Layered { base }),
                    ));
                    self.tasks.push(ContentTask::EndTarget);
                    self.tasks.push(ContentTask::DrawTree(
                        DrawTreeSelection {
                            excluded_subtree: Some(node_id),
                            ..Default::default()
                        },
                        None,
                    ));
                    self.tasks.push(ContentTask::BeginTarget(Target::Texture {
                        texture: base,
                        size: input.physical_size,
                    }));
                } else {
                    self.tasks.push(ContentTask::BeginContent(node_id, None));
                }
            }
            ContentTask::BeginContent(node_id, backdrop_source) => {
                let capture = ContentCapture::new(input.content_effects[&node_id])
                    .expect("scheduled content has capture bounds");
                let texture = texture_ids.allocate();
                output.push(RenderOperation::BeginTarget(Target::Texture {
                    texture,
                    size: input.physical_size,
                }));
                self.tasks
                    .push(ContentTask::FinishContent(node_id, capture));
                self.tasks.push(ContentTask::DrawTree(
                    DrawTreeSelection {
                        subtree_root: Some(node_id),
                        ..Default::default()
                    },
                    backdrop_source,
                ));
            }
            ContentTask::FinishContent(node_id, capture) => {
                let texture = capture.append_capture(output, texture_ids);
                output.push(RenderOperation::EndTarget);
                let composite =
                    capture.append_effect(texture, input.parameters, output, texture_ids);
                self.results.insert(node_id, composite);
            }
            ContentTask::BeginTarget(target) => output.push(RenderOperation::BeginTarget(target)),
            ContentTask::EndTarget => output.push(RenderOperation::EndTarget),
            ContentTask::DrawTree(selection, backdrop_source) => {
                self.selection = selection;
                self.backdrop_source = backdrop_source;
                self.draws.start(selection, input.physical_size);
                self.is_drawing = true;
            }
        }
        true
    }

    fn schedule(&mut self, input: &ContentPlanningInput<'_, '_>) {
        self.groups.clear();
        self.backdrop_ancestors.clear();
        // Layered captures need completed group dependencies outside the captured subtree.
        if input.content_effects.is_empty() || input.backdrop_effects.is_empty() {
            return;
        }
        for &node in input.content_effects.keys() {
            if input.tree.get(node).is_some() {
                self.groups.push((node, node_depth(input.tree, node)));
            }
        }
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
}

#[cfg(test)]
mod tests;
