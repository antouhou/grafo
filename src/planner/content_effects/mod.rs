use super::draws::{DrawPlanner, DrawPlanningInput, DrawTreeSelection};
use crate::commands::{CaptureSource, RenderOperation, Target, TextureComposite};
use crate::core::effect::CaptureRegion;
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
use std::cmp::Reverse;

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
    backdrop_regions: HashMap<usize, Option<UnsignedPhysicalRect>>,
    node_order: Vec<(usize, usize)>,
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
                if let Some(&Some(region)) = self.backdrop_regions.get(&node_id) {
                    let base = texture_ids.allocate();
                    self.tasks.push(ContentTask::BeginContent(
                        node_id,
                        Some(CaptureSource::Layered {
                            base,
                            origin: [region.min.x as i32, region.min.y as i32],
                        }),
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
                        region: CaptureRegion {
                            bounds: region.to_i32(),
                            source_rect: Some(region),
                        },
                    }));
                } else {
                    let source = self
                        .backdrop_regions
                        .contains_key(&node_id)
                        .then_some(CaptureSource::Target);
                    self.tasks.push(ContentTask::BeginContent(node_id, source));
                }
            }
            ContentTask::BeginContent(node_id, backdrop_source) => {
                let capture = ContentCapture::new(
                    input.content_effects[&node_id],
                    self.backdrop_regions.get(&node_id).copied().flatten(),
                    texture_ids,
                )
                .expect("scheduled content has capture bounds");
                capture.append_begin(output);
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
                let texture = capture.append_end(output, texture_ids);
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
        self.backdrop_regions.clear();
        self.node_order.clear();
        // Layered captures need completed group dependencies outside the captured subtree.
        if input.content_effects.is_empty() || input.backdrop_effects.is_empty() {
            return;
        }
        if input.tree.get(0).is_none() {
            return;
        }
        self.node_order.push((0, 0));
        let mut next_node = 0;
        while let Some(&(node, depth)) = self.node_order.get(next_node) {
            self.node_order.extend(
                input
                    .tree
                    .children(node)
                    .iter()
                    .map(|&child| (child, depth + 1)),
            );
            next_node += 1;
        }
        for &(node, depth) in self.node_order.iter().rev() {
            if input.content_effects.contains_key(&node) {
                self.groups.push((node, depth));
            }
            let mut has_backdrop = input.backdrop_effects.contains_key(&node);
            let mut region = input
                .backdrop_effects
                .get(&node)
                .and_then(|effect| effect.capture_region)
                .and_then(|capture| capture.source_rect);
            for child in input.tree.children(node) {
                // Completed child effects supply pixels, not their capture dependencies.
                if input
                    .content_effects
                    .get(child)
                    .is_some_and(|effect| effect.bounds.is_some())
                {
                    continue;
                }
                if let Some(child_region) = self.backdrop_regions.get(child) {
                    has_backdrop = true;
                    if let Some(child_region) = *child_region {
                        region =
                            Some(region.map_or(child_region, |region| region.union(&child_region)));
                    }
                }
            }
            if has_backdrop {
                self.backdrop_regions.insert(node, region);
            }
        }
        self.groups
            .sort_unstable_by_key(|&(node, depth)| (Reverse(depth), node));
    }
}

#[cfg(test)]
mod tests;
