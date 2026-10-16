use super::{DrawPlanner, DrawPlanningInput, DrawTreeSelection};
use crate::commands::{
    CaptureSource, IntermediateTextureId, RenderCommand, RenderOperation, ShapeDrawId,
    TextureComposite, TexturePlacement,
};
use crate::core::effect::BackdropEffectConfig;
use crate::core::shape::CachedShapeHandle;
use crate::core::util::ShapeResources;
use crate::core::vertex::{InstanceTransform, TextureUvTransform};
use crate::core::Viewport;
use crate::planner::commands::TraversalCommands;
use crate::planner::tests::RecordedCommands;
use crate::planner::TextureIdAllocator;
use crate::scene::effects::parameters::EffectParameterStorage;
use crate::scene::effects::{BackdropEffectInstance, ContentEffectInstance, EffectInstance};
use crate::scene::types::CachedShapeDrawData;
use crate::scene::types::{ClipRectDrawData, DrawTreeNode};
use crate::{BorderRadii, Color, Shape, ShapeDrawCommandOptions, Size, UnsignedPhysicalRect};
use ahash::{HashMap, HashMapExt};
use easy_tree::Tree;
use lyon::tessellation::FillTessellator;
use std::mem;

#[derive(Debug, PartialEq)]
enum Operation {
    Draw(ShapeDrawId),
    DrawAndIncrement(ShapeDrawId),
    Decrement(ShapeDrawId),
    Composite(IntermediateTextureId),
}

fn snapshot(plan: &RecordedCommands) -> Vec<(Operation, u32, UnsignedPhysicalRect)> {
    plan.instructions
        .iter()
        .filter_map(|instruction| {
            let operation = match instruction.operation {
                RenderOperation::DrawShape(draw) => Operation::Draw(draw.id),
                RenderOperation::DrawShapeAndIncrementStencil(draw) => {
                    Operation::DrawAndIncrement(draw.id)
                }
                RenderOperation::DecrementStencil(draw) => Operation::Decrement(draw.id),
                RenderOperation::CompositeTexture(texture) => Operation::Composite(texture.texture),
                _ => return None,
            };
            Some((
                operation,
                instruction.clip.stencil_reference,
                instruction.clip.scissor,
            ))
        })
        .collect()
}

fn rect(min: (u32, u32), max: (u32, u32)) -> UnsignedPhysicalRect {
    UnsignedPhysicalRect::new(min.into(), max.into())
}

fn shape_description(shape: Shape, clips_children: bool) -> CachedShapeDrawData {
    let handle = CachedShapeHandle::new(
        &shape,
        &mut FillTessellator::new(),
        &mut ShapeResources::new(),
        None,
    );
    CachedShapeDrawData::new(
        handle,
        ShapeDrawCommandOptions::new()
            .color(Color::WHITE)
            .clips_children(clips_children),
    )
}

fn shape(clips_children: bool) -> DrawTreeNode {
    DrawTreeNode::CachedShape(shape_description(
        Shape::rounded_rect([(0.0, 0.0), (80.0, 80.0)], BorderRadii::new(5.0)),
        clips_children,
    ))
}

fn empty_shape() -> DrawTreeNode {
    DrawTreeNode::CachedShape(shape_description(Shape::builder().build(), true))
}

fn clip(min: (f32, f32), max: (f32, f32)) -> DrawTreeNode {
    DrawTreeNode::ClipRect(ClipRectDrawData::new([min, max], None, true))
}

struct Scene {
    tree: Tree<DrawTreeNode>,
    results: HashMap<usize, TextureComposite>,
    shape_mask_effects: HashMap<usize, TextureComposite>,
    groups: HashMap<usize, ContentEffectInstance>,
    backdrops: HashMap<usize, BackdropEffectInstance>,
    backdrop_source: Option<CaptureSource>,
    parameters: EffectParameterStorage,
}

impl Scene {
    fn new() -> Self {
        Self {
            parameters: EffectParameterStorage::default(),
            tree: Tree::new(),
            results: HashMap::new(),
            shape_mask_effects: HashMap::new(),
            groups: HashMap::new(),
            backdrops: HashMap::new(),
            backdrop_source: Some(CaptureSource::Target),
        }
    }

    fn add(&mut self, parent: Option<usize>, node: DrawTreeNode) -> usize {
        if let Some(parent) = parent {
            self.tree.add_child(parent, node)
        } else {
            self.tree.add_node(node)
        }
    }

    fn plan<'scene>(
        &'scene self,
        planner: &mut DrawPlanner,
        output: &mut RecordedCommands<'scene>,
    ) {
        self.plan_selection(DrawTreeSelection::default(), planner, output);
    }

    fn plan_selection<'scene>(
        &'scene self,
        selection: DrawTreeSelection,
        planner: &mut DrawPlanner,
        output: &mut RecordedCommands<'scene>,
    ) {
        output.clear_commands();
        let input = DrawPlanningInput {
            tree: &self.tree,
            parameters: &self.parameters,
            selection,
            effect_results: &self.results,
            shape_mask_effects: &self.shape_mask_effects,
            content_effects: &self.groups,
            backdrop_effects: &self.backdrops,
            scale_factor: 1.0,
            physical_size: Size::new(100, 100),
            backdrop_source: self.backdrop_source,
        };
        let mut commands = TraversalCommands::default();
        let mut texture_ids = TextureIdAllocator::default();
        planner.start(selection, input.physical_size);
        while planner.advance(&input, &mut commands, &mut texture_ids) {
            output.instructions.extend(mem::take(&mut commands));
        }
    }

    fn attach_backdrop(&mut self, node: usize) {
        self.backdrops.insert(
            node,
            BackdropEffectInstance::new(
                EffectInstance {
                    effect_id: 42,
                    parameters: self.parameters.store(&[1, 2, 3, 4]),
                },
                BackdropEffectConfig::default(),
                self.tree.get(node).unwrap().logical_screen_bounds(),
                Viewport {
                    physical_size: (100, 100),
                    scale_factor: 1.0,
                },
                1024,
            ),
        );
    }

    fn update_backdrop_config(&mut self, node: usize, config: BackdropEffectConfig) {
        let instance = self.backdrops.get_mut(&node).unwrap();
        *instance = BackdropEffectInstance::new(
            instance.effect,
            config,
            self.tree.get(node).unwrap().logical_screen_bounds(),
            Viewport {
                physical_size: (100, 100),
                scale_factor: 1.0,
            },
            1024,
        );
    }
}

#[test]
fn mixed_clips_resolve_each_draw_and_restore_before_siblings() {
    let mut scene = Scene::new();
    let root = scene.add(None, clip((10.0, 10.0), (90.0, 80.0)));
    let stencil = scene.add(Some(root), shape(true));
    let inner = scene.add(Some(stencil), clip((0.0, 20.0), (40.0, 90.0)));
    let leaf = scene.add(Some(inner), shape(true));
    let sibling = scene.add(Some(root), shape(true));
    let mut planner = DrawPlanner::default();
    let mut output = RecordedCommands::default();
    scene.plan(&mut planner, &mut output);
    let outer_clip = rect((10, 10), (90, 80));
    assert_eq!(
        snapshot(&output),
        [
            (
                Operation::DrawAndIncrement(ShapeDrawId(stencil)),
                0,
                outer_clip
            ),
            (
                Operation::Draw(ShapeDrawId(leaf)),
                1,
                rect((10, 20), (40, 80))
            ),
            (Operation::Decrement(ShapeDrawId(stencil)), 1, outer_clip),
            (Operation::Draw(ShapeDrawId(sibling)), 0, outer_clip),
        ]
    );
    assert!(planner.parents.is_empty());
}

#[test]
fn empty_geometry_and_visible_overflow_keep_the_ancestor_stencil() {
    let mut scene = Scene::new();
    let root = scene.add(None, shape(true));
    let empty = scene.add(Some(root), empty_shape());
    let nested = scene.add(Some(empty), shape(true));
    let leaf = scene.add(Some(nested), shape(true));
    let overflow = scene.add(Some(root), shape(false));
    let overflow_leaf = scene.add(Some(overflow), shape(true));
    let sibling = scene.add(Some(root), shape(true));
    let mut output = RecordedCommands::default();
    scene.plan(&mut DrawPlanner::default(), &mut output);
    let viewport = rect((0, 0), (100, 100));
    assert_eq!(
        snapshot(&output),
        [
            (Operation::DrawAndIncrement(ShapeDrawId(root)), 0, viewport),
            (
                Operation::DrawAndIncrement(ShapeDrawId(nested)),
                1,
                viewport
            ),
            (Operation::Draw(ShapeDrawId(leaf)), 2, viewport),
            (Operation::Decrement(ShapeDrawId(nested)), 2, viewport),
            (Operation::Draw(ShapeDrawId(overflow)), 1, viewport),
            (Operation::Draw(ShapeDrawId(overflow_leaf)), 1, viewport),
            (Operation::Draw(ShapeDrawId(sibling)), 1, viewport),
            (Operation::Decrement(ShapeDrawId(root)), 1, viewport),
        ]
    );
}

#[test]
fn effect_composites_carry_only_ids_placements_and_resolved_clips() {
    let mut scene = Scene::new();
    let root = scene.add(None, shape(true));
    let scissor = scene.add(Some(root), clip((10.0, 10.0), (60.0, 60.0)));
    let group = scene.add(Some(scissor), shape(true));
    scene.add(Some(group), shape(true));
    let source = scene.add(Some(scissor), shape(true));
    let child = scene.add(Some(source), shape(true));
    let group_texture = IntermediateTextureId::Registered(5);
    let leaf_texture = IntermediateTextureId::Registered(6);
    scene.results.insert(
        group,
        TextureComposite {
            texture: group_texture,
            placement: TexturePlacement::Target,
        },
    );
    scene.shape_mask_effects.insert(
        source,
        TextureComposite {
            texture: leaf_texture,
            placement: TexturePlacement::Local {
                transform: InstanceTransform::translation(10.0, 20.0),
                sampling: TextureUvTransform::IDENTITY,
            },
        },
    );
    let mut output = RecordedCommands::default();
    scene.plan(&mut DrawPlanner::default(), &mut output);
    let viewport = rect((0, 0), (100, 100));
    let inherited = rect((10, 10), (60, 60));
    assert_eq!(
        snapshot(&output),
        [
            (Operation::DrawAndIncrement(ShapeDrawId(root)), 0, viewport),
            (Operation::Composite(group_texture), 1, inherited),
            (Operation::Composite(leaf_texture), 1, inherited),
            (
                Operation::DrawAndIncrement(ShapeDrawId(source)),
                1,
                inherited
            ),
            (Operation::Draw(ShapeDrawId(child)), 2, inherited),
            (Operation::Decrement(ShapeDrawId(source)), 2, inherited),
            (Operation::Decrement(ShapeDrawId(root)), 1, viewport),
        ]
    );
    let RenderOperation::CompositeTexture(composite) = output.instructions[2].operation else {
        panic!("expected effect composite")
    };
    assert_eq!(composite.texture, leaf_texture);
    let TexturePlacement::Local {
        transform,
        sampling,
    } = composite.placement
    else {
        panic!("expected local placement");
    };
    assert_eq!(transform.col3, [10.0, 20.0, 0.0, 1.0]);
    assert_eq!(sampling.scale, [1.0; 2]);
}

#[test]
fn offscreen_scissor_and_transparent_parent_restore_the_visible_sibling() {
    let mut scene = Scene::new();
    let root = scene.add(None, clip((10.0, 10.0), (60.0, 60.0)));
    let offscreen = scene.add(Some(root), clip((150.0, 150.0), (200.0, 200.0)));
    let clipped = scene.add(Some(offscreen), shape(true));
    let sibling = scene.add(Some(root), shape(true));
    let mut output = RecordedCommands::default();
    scene.plan(&mut DrawPlanner::default(), &mut output);
    assert_eq!(
        snapshot(&output),
        [
            (
                Operation::Draw(ShapeDrawId(clipped)),
                0,
                UnsignedPhysicalRect::zero()
            ),
            (
                Operation::Draw(ShapeDrawId(sibling)),
                0,
                rect((10, 10), (60, 60))
            ),
        ]
    );
}

#[test]
fn empty_backdrop_parent_preserves_ancestor_clips_without_capture() {
    let mut output = RecordedCommands::default();
    let mut scene = Scene::new();
    let root = scene.add(None, shape(true));
    let scissor = scene.add(Some(root), clip((10.0, 10.0), (60.0, 60.0)));
    let empty = scene.add(Some(scissor), empty_shape());
    scene.attach_backdrop(empty);
    let child = scene.add(Some(empty), shape(true));
    let sibling = scene.add(Some(root), shape(true));
    scene.plan(&mut DrawPlanner::default(), &mut output);
    let viewport = rect((0, 0), (100, 100));
    assert_eq!(
        snapshot(&output),
        [
            (Operation::DrawAndIncrement(ShapeDrawId(root)), 0, viewport),
            (
                Operation::Draw(ShapeDrawId(child)),
                1,
                rect((10, 10), (60, 60))
            ),
            (Operation::Draw(ShapeDrawId(sibling)), 1, viewport),
            (Operation::Decrement(ShapeDrawId(root)), 1, viewport),
        ]
    );
}

mod backdrops;
mod selection;
mod targets;

fn operations<'commands, 'scene>(
    plan: &'commands RecordedCommands<'scene>,
) -> Vec<&'commands RenderOperation<'scene>> {
    plan.instructions
        .iter()
        .map(|command| &command.operation)
        .collect()
}
