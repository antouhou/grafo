use super::{ContentPlanningInput, SceneTraversal};
use crate::commands::{
    CaptureSource, IntermediateTextureId, RenderOperation, ShapeTextureBinding, Target,
    TextureComposite,
};
use crate::core::effect::{BackdropEffectConfig, ContentEffectConfig};
use crate::core::shape::CachedShapeHandle;
use crate::core::util::ShapeResources;
use crate::core::Viewport;
use crate::planner::commands::TraversalCommands;
use crate::planner::tests::RecordedCommands;
use crate::planner::TextureIdAllocator;
use crate::scene_state::effects::parameters::EffectParameterStorage;
use crate::scene_state::effects::{BackdropEffectInstance, ContentEffectInstance, EffectInstance};
use crate::scene_state::types::CachedShapeDrawData;
use crate::scene_state::types::{ClipRectDrawData, DrawTreeNode};
use crate::{Shape, ShapeDrawCommandOptions, Size};
use ahash::{HashMap, HashMapExt};
use easy_tree::Tree;
use lyon::tessellation::FillTessellator;
use std::mem;

#[derive(Debug, PartialEq, Eq)]
enum Command {
    Begin(Option<usize>),
    End,
    Capture(usize),
    Effect(u64, usize, usize),
    Composite(usize),
}

fn planned_index(texture: IntermediateTextureId) -> usize {
    let IntermediateTextureId::Planned(index) = texture else {
        panic!("group resources must use planned texture IDs");
    };
    index
}

/// Checks resource dependencies in the yielded command order.
fn validate_dependencies(plan: &RecordedCommands) {
    let mut produced = Vec::new();
    let mut targets = Vec::new();
    let mut surface_count = 0;
    for command in &plan.instructions {
        match &command.operation {
            RenderOperation::BeginTarget(target) => {
                if matches!(target, Target::Surface) {
                    surface_count += 1;
                }
                if let Target::Texture { texture, .. } = target {
                    assert_eq!(planned_index(*texture), produced.len());
                    produced.push(false);
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
                assert_eq!(planned_index(effect.output), produced.len());
                produced.push(true);
            }
            RenderOperation::Capture(capture) => {
                assert!(!targets.is_empty());
                if let CaptureSource::Layered { base, .. } = capture.source {
                    assert!(produced[planned_index(base)]);
                }
                assert_eq!(planned_index(capture.output), produced.len());
                produced.push(true);
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

fn snapshot(plan: &RecordedCommands) -> Vec<Command> {
    plan.instructions
        .iter()
        .map(|command| match &command.operation {
            RenderOperation::BeginTarget(Target::Texture { texture, .. }) => {
                Command::Begin(Some(planned_index(*texture)))
            }
            RenderOperation::BeginTarget(Target::Surface) => Command::Begin(None),
            RenderOperation::EndTarget => Command::End,
            RenderOperation::Capture(capture) => Command::Capture(planned_index(capture.output)),
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
    parameters: EffectParameterStorage,
    tree: Tree<DrawTreeNode>,
    groups: HashMap<usize, ContentEffectInstance>,
    backdrops: HashMap<usize, BackdropEffectInstance>,
    shapes: HashMap<usize, TextureComposite>,
}

impl Scene {
    fn new() -> Self {
        Self {
            parameters: EffectParameterStorage::default(),
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
            self.tree.add_child(parent, node)
        } else {
            self.tree.add_node(node)
        }
    }

    fn group(&mut self, parent: Option<usize>, effect_id: u64) -> usize {
        let node = self.add(parent);
        self.attach_content(node, effect_id, ContentEffectConfig::default());
        node
    }

    fn attach_content(&mut self, node: usize, effect_id: u64, config: ContentEffectConfig) {
        self.groups.insert(
            node,
            ContentEffectInstance::new(
                EffectInstance {
                    effect_id,
                    parameters: self.parameters.store(&effect_id.to_le_bytes()),
                },
                config,
                self.tree.get(node).unwrap(),
                None,
                Viewport {
                    physical_size: (100, 100),
                    scale_factor: 1.0,
                },
                4096,
            ),
        );
    }

    fn backdrop(&mut self, parent: usize) -> usize {
        let shape = CachedShapeHandle::new(
            &Shape::rect([(10.0, 10.0), (40.0, 40.0)]),
            &mut FillTessellator::new(),
            &mut ShapeResources::new(),
            None,
        );
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
                    parameters: self.parameters.store(&[9; 4]),
                },
                BackdropEffectConfig::new().padding(4.0).downsample(0.5),
                self.tree.get(node).unwrap().logical_screen_bounds(),
                Viewport {
                    physical_size: (100, 100),
                    scale_factor: 1.0,
                },
                1024,
            ),
        );
        node
    }

    fn plan<'scene>(
        &'scene self,
        planner: &mut SceneTraversal,
        output: &mut RecordedCommands<'scene>,
    ) {
        output.clear_commands();
        output.push(RenderOperation::BeginTarget(Target::Surface));
        let input = ContentPlanningInput {
            tree: &self.tree,
            parameters: &self.parameters,
            content_effects: &self.groups,
            backdrop_effects: &self.backdrops,
            shape_mask_effects: &self.shapes,
            scale_factor: 1.0,
            physical_size: Size::new(100, 100),
        };
        planner.start(&input);
        let mut commands = TraversalCommands::default();
        let mut texture_ids = TextureIdAllocator::default();
        while planner.advance(&input, &mut commands, &mut texture_ids) {
            output.instructions.extend(mem::take(&mut commands));
        }
        output.push(RenderOperation::EndTarget);
    }
}

#[test]
fn nested_groups_close_into_their_parent_targets() {
    let mut output = RecordedCommands::default();
    let mut scene = Scene::new();
    let outer = scene.group(None, 1);
    let inner = scene.group(Some(outer), 2);
    scene.group(Some(inner), 3);
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
        assert_eq!(effect.parameters.bytes, effect.effect_id.to_le_bytes());
    }
}

#[test]
fn uneven_groups_resume_the_parent_between_siblings() {
    let mut output = RecordedCommands::default();
    let mut scene = Scene::new();
    let root = scene.add(None);
    let first = scene.group(Some(root), 1);
    let inner = scene.group(Some(first), 2);
    scene.group(Some(inner), 3);
    scene.group(Some(root), 4);
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
    let mut output = RecordedCommands::default();
    let mut scene = Scene::new();
    let root = scene.add(None);
    scene.backdrop(root);
    let outer = scene.group(Some(root), 1);
    scene.backdrop(outer);
    let inner = scene.group(Some(outer), 2);
    scene.backdrop(inner);
    scene.plan(&mut SceneTraversal::default(), &mut output);
    validate_dependencies(&output);
    let captures: Vec<_> = output
        .instructions
        .windows(2)
        .filter_map(
            |commands| match (&commands[0].operation, &commands[1].operation) {
                (RenderOperation::Capture(capture), RenderOperation::ApplyEffect(effect))
                    if effect.effect_id == 99 =>
                {
                    Some(capture)
                }
                _ => None,
            },
        )
        .collect();
    assert_eq!(captures.len(), 3);
    assert_eq!(
        captures[0].source,
        CaptureSource::Layered {
            base: IntermediateTextureId::Planned(0),
            origin: [6, 6],
        }
    );
    assert_eq!(
        captures[1].source,
        CaptureSource::Layered {
            base: IntermediateTextureId::Planned(5),
            origin: [6, 6],
        }
    );
    assert_eq!(captures[2].source, CaptureSource::Target);
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
fn queue_rebuilds_reuse_scheduling_storage_and_empty_scenes_clear_the_surface() {
    let mut planner = SceneTraversal::default();
    let mut scene = Scene::new();
    let group = scene.group(None, 1);
    scene.backdrop(group);
    let mut output = RecordedCommands::default();
    scene.plan(&mut planner, &mut output);
    let capacities = (
        planner.groups.capacity(),
        planner.results.capacity(),
        planner.backdrop_regions.capacity(),
        planner.tasks.capacity(),
    );
    for _ in 0..3 {
        let mut rebuilt = Scene::new();
        let group = rebuilt.group(None, 7);
        rebuilt.backdrop(group);
        let mut output = RecordedCommands::default();
        rebuilt.plan(&mut planner, &mut output);
        validate_dependencies(&output);
        assert_eq!(
            capacities,
            (
                planner.groups.capacity(),
                planner.results.capacity(),
                planner.backdrop_regions.capacity(),
                planner.tasks.capacity()
            )
        );
    }
    let empty = Scene::new();
    let mut output = RecordedCommands::default();
    empty.plan(&mut planner, &mut output);
    validate_dependencies(&output);
    assert_eq!(snapshot(&output), [Command::Begin(None), Command::End]);
    assert!(planner.results.is_empty());
    assert!(planner.backdrop_regions.is_empty());
}

mod bounds;
