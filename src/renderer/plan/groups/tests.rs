use super::{GroupPlanner, GroupPlanningInput};
use crate::commands::{
    BackdropCaptureSource, DrawOperation, DrawPlan, DrawSegment, IntermediateTextureId,
    ShapeTextureBinding, Target, TextureComposite,
};
use crate::core::effect::{BackdropEffectConfig, BackdropEffectInstance, EffectInstance};
use crate::core::shape::CachedShapeHandle;
use crate::core::util::ShapeResources;
use crate::renderer::types::CachedShapeDrawData;
use crate::renderer::types::{ClipRectDrawData, DrawTreeNode};
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
fn validate_dependencies(plan: &DrawPlan) {
    let mut produced = vec![false; plan.texture_count];
    let mut target = None;
    let mut instruction_end = 0;
    let mut surface_count = 0;
    for segment in &plan.segments {
        match segment {
            DrawSegment::BeginTarget(next) => {
                assert!(target.replace(*next).is_none(), "targets cannot nest");
                match next {
                    Target::Surface => surface_count += 1,
                    Target::Texture { texture, .. } => assert!(!produced[planned_index(*texture)]),
                    Target::Mask(_) => panic!("masks are prepared before group targets"),
                }
            }
            DrawSegment::EndTarget => {
                if let Target::Texture { texture, .. } = target.take().unwrap() {
                    produced[planned_index(texture)] = true;
                }
            }
            DrawSegment::ApplyEffect(effect) => {
                assert!(
                    produced[planned_index(effect.input)],
                    "effect input must be completed"
                );
                let output = planned_index(effect.output);
                assert!(!produced[output]);
                produced[output] = true;
                effect.parameters.bytes(&plan.effect_parameters);
            }
            DrawSegment::CaptureBackdrop(capture) => {
                assert!(target.is_some());
                if let BackdropCaptureSource::Layered { base } = capture.source {
                    assert!(
                        produced[planned_index(base)],
                        "layered base must be completed"
                    );
                }
                let output = planned_index(capture.output);
                assert!(!produced[output]);
                produced[output] = true;
            }
            DrawSegment::Draws { instructions, .. } => {
                assert!(target.is_some());
                assert_eq!(instructions.start, instruction_end);
                instruction_end = instructions.end;
                for instruction in &plan.instructions[instructions.clone()] {
                    let texture = match instruction.operation {
                        DrawOperation::CompositeTexture(index) => {
                            Some(plan.composites[index].texture)
                        }
                        DrawOperation::DrawShape(draw)
                        | DrawOperation::DrawShapeAndIncrementStencil(draw) => {
                            draw.material.under_fill_texture.and_then(|layer| {
                                if let ShapeTextureBinding::Intermediate(texture) = layer.texture {
                                    Some(texture)
                                } else {
                                    None
                                }
                            })
                        }
                        _ => None,
                    };
                    if let Some(IntermediateTextureId::Planned(index)) = texture {
                        assert!(produced[index], "draw input must be completed");
                    }
                }
            }
            DrawSegment::DrawShapeMask(_) => panic!("unexpected mask draw"),
        }
    }
    assert!(target.is_none());
    assert_eq!(surface_count, 1);
    assert_eq!(instruction_end, plan.instructions.len());
    assert!(produced.into_iter().all(|is_produced| is_produced));
}

fn snapshot(plan: &DrawPlan) -> Vec<Command> {
    let mut commands = Vec::new();
    for segment in &plan.segments {
        match segment {
            DrawSegment::BeginTarget(Target::Texture { texture, .. }) => {
                commands.push(Command::Begin(Some(planned_index(*texture))));
            }
            DrawSegment::BeginTarget(Target::Surface) => commands.push(Command::Begin(None)),
            DrawSegment::EndTarget => commands.push(Command::End),
            DrawSegment::ApplyEffect(effect) => commands.push(Command::Effect(
                effect.effect_id,
                planned_index(effect.input),
                planned_index(effect.output),
            )),
            DrawSegment::Draws { instructions, .. } => {
                for instruction in &plan.instructions[instructions.clone()] {
                    let DrawOperation::CompositeTexture(index) = instruction.operation else {
                        panic!("clip-only scenes contain only composites");
                    };
                    commands.push(Command::Composite(planned_index(
                        plan.composites[index].texture,
                    )));
                }
            }
            _ => panic!("unexpected command in clip-only scene"),
        }
    }
    commands
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
                &ShapeDrawCommandOptions::new(),
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

    fn plan(&self, planner: &mut GroupPlanner, output: &mut DrawPlan) {
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
    }
}

#[test]
fn nested_groups_finish_before_ancestor_effects_and_surface_composites() {
    let mut scene = Scene::new();
    let outer = scene.group(None, 1);
    let inner = scene.group(Some(outer), 2);
    scene.group(Some(inner), 3);
    let mut output = DrawPlan::default();
    scene.plan(&mut GroupPlanner::default(), &mut output);
    drop(scene);
    validate_dependencies(&output);
    assert_eq!(
        snapshot(&output),
        [
            Command::Begin(Some(0)),
            Command::End,
            Command::Effect(3, 0, 1),
            Command::Begin(Some(2)),
            Command::Composite(1),
            Command::End,
            Command::Effect(2, 2, 3),
            Command::Begin(Some(4)),
            Command::Composite(3),
            Command::End,
            Command::Effect(1, 4, 5),
            Command::Begin(None),
            Command::Composite(5),
            Command::End,
        ]
    );
    let effects = output.segments.iter().filter_map(|segment| {
        if let DrawSegment::ApplyEffect(effect) = segment {
            Some(effect)
        } else {
            None
        }
    });
    for effect in effects {
        assert_eq!(
            effect.parameters.bytes(&output.effect_parameters),
            effect.effect_id.to_le_bytes()
        );
    }
}

#[test]
fn uneven_groups_use_descendant_results_and_order_siblings_deterministically() {
    let mut scene = Scene::new();
    let root = scene.add(None);
    let first = scene.group(Some(root), 1);
    let inner = scene.group(Some(first), 2);
    scene.group(Some(inner), 3);
    scene.group(Some(root), 4);
    let mut output = DrawPlan::default();
    scene.plan(&mut GroupPlanner::default(), &mut output);
    validate_dependencies(&output);
    assert_eq!(
        snapshot(&output),
        [
            Command::Begin(Some(0)),
            Command::End,
            Command::Effect(3, 0, 1),
            Command::Begin(Some(2)),
            Command::Composite(1),
            Command::End,
            Command::Effect(2, 2, 3),
            Command::Begin(Some(4)),
            Command::Composite(3),
            Command::End,
            Command::Effect(1, 4, 5),
            Command::Begin(Some(6)),
            Command::End,
            Command::Effect(4, 6, 7),
            Command::Begin(None),
            Command::Composite(5),
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
    let mut output = DrawPlan::default();
    scene.plan(&mut GroupPlanner::default(), &mut output);
    validate_dependencies(&output);
    let captures: Vec<_> = output
        .segments
        .iter()
        .filter_map(|segment| {
            if let DrawSegment::CaptureBackdrop(capture) = segment {
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
    for segment in &output.segments {
        match segment {
            DrawSegment::BeginTarget(Target::Texture { texture, .. }) => target = Some(*texture),
            DrawSegment::EndTarget => target = None,
            DrawSegment::Draws { instructions, .. }
                if target == Some(IntermediateTextureId::Planned(5)) =>
            {
                composites.extend(output.instructions[instructions.clone()].iter().filter_map(
                    |draw| {
                        if let DrawOperation::CompositeTexture(index) = draw.operation {
                            Some(index)
                        } else {
                            None
                        }
                    },
                ));
            }
            _ => {}
        }
    }
    assert!(composites.is_empty());
}

#[test]
fn queue_rebuilds_reuse_storage_and_empty_scenes_clear_the_surface() {
    let mut planner = GroupPlanner::default();
    let mut output = DrawPlan::default();
    let mut scene = Scene::new();
    let group = scene.group(None, 1);
    scene.backdrop(group);
    scene.plan(&mut planner, &mut output);
    let capacities = (
        output.segments.capacity(),
        output.instructions.capacity(),
        output.effect_parameters.capacity(),
        output.composites.capacity(),
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
                output.segments.capacity(),
                output.instructions.capacity(),
                output.effect_parameters.capacity(),
                output.composites.capacity(),
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
