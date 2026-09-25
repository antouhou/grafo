use super::{GroupPlanningInput, SceneTraversal};
use crate::commands::{
    BackdropCaptureSource, IntermediateTextureId, RenderOperation, RenderPlan, ShapeTextureBinding,
    Target, TextureComposite,
};
use crate::core::effect::BackdropEffectConfig;
use crate::core::shape::CachedShapeHandle;
use crate::core::util::ShapeResources;
use crate::scene::effects::{BackdropEffectInstance, EffectInstance};
use crate::scene::types::CachedShapeDrawData;
use crate::scene::types::{ClipRectDrawData, DrawTreeNode};
use crate::{Shape, ShapeDrawCommandOptions, Size, Stroke};
use ahash::{HashMap, HashMapExt};
use easy_tree::Tree;
use lyon::tessellation::FillTessellator;

#[derive(Debug, PartialEq, Eq)]
enum Command {
    Begin(Option<usize>),
    End,
    Effect(u64, usize, usize),
    Composite(usize),
}

fn planned_index(texture: IntermediateTextureId) -> usize {
    let IntermediateTextureId::Planned(index) = texture else {
        panic!("group resources must use planned texture IDs");
    };
    index
}

/// Consumes only completed commands, including after dropping their scene and planner.
fn validate_dependencies(plan: &RenderPlan) {
    let mut produced = vec![false; plan.texture_count];
    let mut targets = Vec::new();
    let mut surface_count = 0;
    for command in &plan.instructions {
        match &command.operation {
            RenderOperation::BeginTarget(target) => {
                if matches!(target, Target::Surface) {
                    surface_count += 1;
                }
                targets.push(*target);
            }
            RenderOperation::EndTarget => {
                if let Target::Texture { texture, .. } = targets.pop().unwrap() {
                    produced[planned_index(texture)] = true;
                }
            }
            RenderOperation::ApplyEffect(effect) => {
                assert!(produced[planned_index(effect.input)]);
                produced[planned_index(effect.output)] = true;
            }
            RenderOperation::CaptureBackdrop(capture) => {
                assert!(!targets.is_empty());
                if let BackdropCaptureSource::Layered { base } = capture.source {
                    assert!(produced[planned_index(base)]);
                }
                produced[planned_index(capture.output)] = true;
            }
            operation => {
                assert!(!targets.is_empty());
                let texture = match operation {
                    RenderOperation::CompositeTexture(composite) => Some(composite.texture),
                    RenderOperation::DrawShape(draw)
                    | RenderOperation::DrawShapeAndIncrementStencil(draw) => draw
                        .material
                        .under_fill_texture
                        .and_then(|layer| match layer.texture {
                            ShapeTextureBinding::Intermediate(texture) => Some(texture),
                            _ => None,
                        }),
                    _ => None,
                };
                if let Some(IntermediateTextureId::Planned(index)) = texture {
                    assert!(produced[index]);
                }
            }
        }
    }
    assert!(targets.is_empty());
    assert_eq!(surface_count, 1);
    assert!(produced.into_iter().all(|is_produced| is_produced));
}

fn snapshot(plan: &RenderPlan) -> Vec<Command> {
    plan.instructions
        .iter()
        .map(|command| match &command.operation {
            RenderOperation::BeginTarget(Target::Texture { texture, .. }) => {
                Command::Begin(Some(planned_index(*texture)))
            }
            RenderOperation::BeginTarget(Target::Surface) => Command::Begin(None),
            RenderOperation::EndTarget => Command::End,
            RenderOperation::ApplyEffect(effect) => Command::Effect(
                effect.effect_id,
                planned_index(effect.input),
                planned_index(effect.output),
            ),
            RenderOperation::CompositeTexture(composite) => {
                Command::Composite(planned_index(composite.texture))
            }
            _ => panic!("unexpected command in clip-only scene"),
        })
        .collect()
}

struct Scene {
    tree: Tree<DrawTreeNode>,
    groups: HashMap<usize, EffectInstance>,
    backdrops: HashMap<usize, BackdropEffectInstance>,
    shapes: HashMap<usize, TextureComposite>,
}

impl Scene {
    fn new() -> Self {
        Self {
            tree: Tree::new(),
            groups: HashMap::new(),
            backdrops: HashMap::new(),
            shapes: HashMap::new(),
        }
    }

    fn add(&mut self, parent: Option<usize>) -> usize {
        let node = DrawTreeNode::ClipRect(ClipRectDrawData::new(
            [(0.0, 0.0), (100.0, 100.0)],
            None,
            true,
        ));
        if let Some(parent) = parent {
            self.tree.get_mut(parent).unwrap().set_not_leaf();
            self.tree.add_child(parent, node)
        } else {
            self.tree.add_node(node)
        }
    }

    fn group(&mut self, parent: Option<usize>, effect_id: u64) -> usize {
        let node = self.add(parent);
        self.groups.insert(
            node,
            EffectInstance {
                effect_id,
                params: effect_id.to_le_bytes().to_vec(),
            },
        );
        node
    }

    fn backdrop(&mut self, parent: usize) {
        let shape = CachedShapeHandle::new(
            &Shape::rect([(10.0, 10.0), (40.0, 40.0)], Stroke::default()),
            &mut FillTessellator::new(),
            &mut ShapeResources::new(),
            None,
        );
        self.tree.get_mut(parent).unwrap().set_not_leaf();
        let node = self.tree.add_child(
            parent,
            DrawTreeNode::CachedShape(CachedShapeDrawData::new(
                shape,
                ShapeDrawCommandOptions::new(),
            )),
        );
        self.backdrops.insert(
            node,
            BackdropEffectInstance::new(
                EffectInstance {
                    effect_id: 99,
                    params: vec![9; 4],
                },
                BackdropEffectConfig::new().padding(4.0).downsample(0.5),
            ),
        );
    }

    fn plan(&self, planner: &mut SceneTraversal, output: &mut RenderPlan) {
        output.clear();
        output.push(RenderOperation::BeginTarget(Target::Surface));
        planner.plan(
            GroupPlanningInput {
                tree: &self.tree,
                group_effects: &self.groups,
                backdrop_effects: &self.backdrops,
                shape_effects: &self.shapes,
                scale_factor: 1.0,
                physical_size: Size::new(100, 100),
                max_capture_dimension: 1024,
            },
            output,
        );
        output.push(RenderOperation::EndTarget);
    }
}

#[test]
fn nested_groups_close_into_their_parent_targets() {
    let mut scene = Scene::new();
    let outer = scene.group(None, 1);
    let inner = scene.group(Some(outer), 2);
    scene.group(Some(inner), 3);
    let mut output = RenderPlan::default();
    scene.plan(&mut SceneTraversal::default(), &mut output);
    drop(scene);
    validate_dependencies(&output);
    assert_eq!(
        snapshot(&output),
        [
            Command::Begin(None),
            Command::Begin(Some(0)),
            Command::Begin(Some(1)),
            Command::Begin(Some(2)),
            Command::End,
            Command::Effect(3, 2, 3),
            Command::Composite(3),
            Command::End,
            Command::Effect(2, 1, 4),
            Command::Composite(4),
            Command::End,
            Command::Effect(1, 0, 5),
            Command::Composite(5),
            Command::End,
        ]
    );
    let effects = output
        .instructions
        .iter()
        .map(|command| &command.operation)
        .filter_map(|operation| {
            if let RenderOperation::ApplyEffect(effect) = operation {
                Some(effect)
            } else {
                None
            }
        });
    for effect in effects {
        assert_eq!(
            output.parameters(effect.parameters),
            effect.effect_id.to_le_bytes()
        );
    }
}

#[test]
fn uneven_groups_resume_the_parent_between_siblings() {
    let mut scene = Scene::new();
    let root = scene.add(None);
    let first = scene.group(Some(root), 1);
    let inner = scene.group(Some(first), 2);
    scene.group(Some(inner), 3);
    scene.group(Some(root), 4);
    let mut output = RenderPlan::default();
    scene.plan(&mut SceneTraversal::default(), &mut output);
    validate_dependencies(&output);
    assert_eq!(
        snapshot(&output),
        [
            Command::Begin(None),
            Command::Begin(Some(0)),
            Command::Begin(Some(1)),
            Command::Begin(Some(2)),
            Command::End,
            Command::Effect(3, 2, 3),
            Command::Composite(3),
            Command::End,
            Command::Effect(2, 1, 4),
            Command::Composite(4),
            Command::End,
            Command::Effect(1, 0, 5),
            Command::Composite(5),
            Command::Begin(Some(6)),
            Command::End,
            Command::Effect(4, 6, 7),
            Command::Composite(7),
            Command::End,
        ]
    );
}

#[test]
fn layered_backdrop_sources_finish_before_captures_and_skip_their_group() {
    let mut scene = Scene::new();
    let root = scene.add(None);
    scene.backdrop(root);
    let outer = scene.group(Some(root), 1);
    scene.backdrop(outer);
    let inner = scene.group(Some(outer), 2);
    scene.backdrop(inner);
    let mut output = RenderPlan::default();
    scene.plan(&mut SceneTraversal::default(), &mut output);
    validate_dependencies(&output);
    let captures: Vec<_> = output
        .instructions
        .iter()
        .map(|command| &command.operation)
        .filter_map(|operation| {
            if let RenderOperation::CaptureBackdrop(capture) = operation {
                Some(capture)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(captures.len(), 3);
    assert_eq!(
        captures[0].source,
        BackdropCaptureSource::Layered {
            base: IntermediateTextureId::Planned(0)
        }
    );
    assert_eq!(
        captures[1].source,
        BackdropCaptureSource::Layered {
            base: IntermediateTextureId::Planned(5)
        }
    );
    assert_eq!(captures[2].source, BackdropCaptureSource::Target);
    for capture in captures {
        assert_eq!(capture.sampling_size, Size::new(19, 19));
    }
    // The outer group's behind scene excludes the completed inner result as well.
    let mut target = None;
    let mut composites = Vec::new();
    for command in &output.instructions {
        match command.operation {
            RenderOperation::BeginTarget(Target::Texture { texture, .. }) => target = Some(texture),
            RenderOperation::EndTarget => target = None,
            RenderOperation::CompositeTexture(composite)
                if target == Some(IntermediateTextureId::Planned(5)) =>
            {
                composites.push(composite)
            }
            _ => {}
        }
    }
    assert!(composites.is_empty());
}

#[test]
fn queue_rebuilds_reuse_storage_and_empty_scenes_clear_the_surface() {
    let mut planner = SceneTraversal::default();
    let mut output = RenderPlan::default();
    let mut scene = Scene::new();
    let group = scene.group(None, 1);
    scene.backdrop(group);
    scene.plan(&mut planner, &mut output);
    let capacities = (
        output.instructions.capacity(),
        output.effect_parameters.capacity(),
        planner.groups.capacity(),
        planner.results.capacity(),
        planner.backdrop_ancestors.capacity(),
    );
    for _ in 0..3 {
        let mut rebuilt = Scene::new();
        let group = rebuilt.group(None, 7);
        rebuilt.backdrop(group);
        rebuilt.plan(&mut planner, &mut output);
        validate_dependencies(&output);
        assert_eq!(
            capacities,
            (
                output.instructions.capacity(),
                output.effect_parameters.capacity(),
                planner.groups.capacity(),
                planner.results.capacity(),
                planner.backdrop_ancestors.capacity()
            )
        );
    }
    Scene::new().plan(&mut planner, &mut output);
    validate_dependencies(&output);
    assert_eq!(snapshot(&output), [Command::Begin(None), Command::End]);
    assert!(output.effect_parameters.is_empty());
    assert!(planner.results.is_empty());
    assert!(planner.backdrop_ancestors.is_empty());
    scene.plan(&mut planner, &mut output);
    validate_dependencies(&output);
}
