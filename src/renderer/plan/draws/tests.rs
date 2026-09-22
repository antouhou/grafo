use super::{DrawPlanner, DrawPlanningInput};
use crate::effect::{BackdropEffectConfig, BackdropEffectInstance, EffectInstance};
use crate::renderer::commands::{
    DrawInstruction, DrawOperation, DrawPlan, DrawSegment, IntermediateTextureId, ShapeDrawId,
};
use crate::renderer::plan::shape_effects::{PreparedShapeEffectLeaf, ShapeEffectRasterRect};
use crate::renderer::traversal::{plan_traversal_in_place, TraversalScratch};
use crate::renderer::types::{ClipRectDrawData, DrawTreeNode};
use crate::shape::{CachedShapeDrawData, CachedShapeHandle, ShapeTextureBinding};
use crate::util::ShapeResources;
use crate::{
    BorderRadii, Color, Shape, ShapeDrawCommandOptions, Size, Stroke, UnsignedPhysicalRect,
};
use ahash::{HashMap, HashMapExt};
use easy_tree::Tree;
use lyon::tessellation::FillTessellator;

#[derive(Debug, PartialEq)]
enum Operation {
    Draw(ShapeDrawId),
    DrawAndIncrement(ShapeDrawId),
    Decrement(ShapeDrawId),
    Composite(IntermediateTextureId),
}

fn snapshot(plan: &DrawPlan) -> Vec<(Operation, u32, UnsignedPhysicalRect)> {
    plan.instructions
        .iter()
        .map(|instruction| {
            let operation = match instruction.operation {
                DrawOperation::DrawShape(draw) => Operation::Draw(draw.id),
                DrawOperation::DrawShapeAndIncrementStencil(draw) => {
                    Operation::DrawAndIncrement(draw.id)
                }
                DrawOperation::DecrementStencil(draw) => Operation::Decrement(draw.id),
                DrawOperation::CompositeTexture(texture) => Operation::Composite(texture),
            };
            (
                operation,
                instruction.clip.stencil_reference,
                instruction.clip.scissor,
            )
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
        &ShapeDrawCommandOptions::new()
            .color(Color::WHITE)
            .clips_children(clips_children),
    )
}

fn shape(clips_children: bool) -> DrawTreeNode {
    DrawTreeNode::CachedShape(shape_description(
        Shape::rounded_rect(
            [(0.0, 0.0), (80.0, 80.0)],
            BorderRadii::new(5.0),
            Stroke::default(),
        ),
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
    results: HashMap<usize, IntermediateTextureId>,
    leaves: HashMap<usize, PreparedShapeEffectLeaf>,
    groups: HashMap<usize, EffectInstance>,
    backdrops: HashMap<usize, BackdropEffectInstance>,
    traversal: TraversalScratch,
}

impl Scene {
    fn new() -> Self {
        Self {
            tree: Tree::new(),
            results: HashMap::new(),
            leaves: HashMap::new(),
            groups: HashMap::new(),
            backdrops: HashMap::new(),
            traversal: TraversalScratch::new(),
        }
    }

    fn add(&mut self, parent: Option<usize>, node: DrawTreeNode) -> usize {
        if let Some(parent) = parent {
            self.tree.get_mut(parent).unwrap().set_not_leaf();
            self.tree.add_child(parent, node)
        } else {
            self.tree.add_node(node)
        }
    }

    fn plan(&mut self, planner: &mut DrawPlanner, output: &mut DrawPlan) {
        plan_traversal_in_place(
            &mut self.tree,
            &self.results,
            &self.leaves,
            None,
            None,
            &mut self.traversal,
        );
        planner.plan(
            self.traversal.events(),
            DrawPlanningInput {
                tree: &self.tree,
                effect_results: &self.results,
                effect_leaves: &self.leaves,
                group_effects: &self.groups,
                backdrop_effects: &self.backdrops,
                scale_factor: 1.0,
                physical_size: Size::new(100, 100),
                max_capture_dimension: Some(1024),
            },
            output,
        );
    }

    fn attach_backdrop(&mut self, node: usize) {
        self.backdrops.insert(
            node,
            BackdropEffectInstance::new(
                EffectInstance {
                    effect_id: 42,
                    params: vec![1, 2, 3, 4],
                },
                BackdropEffectConfig::default(),
            ),
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
    let mut output = DrawPlan::default();
    scene.plan(&mut planner, &mut output);
    let outer_clip = rect((10, 10), (90, 80));
    assert_eq!(
        snapshot(&output),
        [
            (
                Operation::DrawAndIncrement(ShapeDrawId::Shape(stencil)),
                0,
                outer_clip
            ),
            (
                Operation::Draw(ShapeDrawId::Shape(leaf)),
                1,
                rect((10, 20), (40, 80))
            ),
            (
                Operation::Decrement(ShapeDrawId::Shape(stencil)),
                1,
                outer_clip
            ),
            (Operation::Draw(ShapeDrawId::Shape(sibling)), 0, outer_clip),
        ]
    );
    assert!(planner.parents.is_empty());
    #[cfg(feature = "render_metrics")]
    assert_eq!(output.scissor_clip_count, 2);
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
    let mut output = DrawPlan::default();
    scene.plan(&mut DrawPlanner::default(), &mut output);
    let viewport = rect((0, 0), (100, 100));
    assert_eq!(
        snapshot(&output),
        [
            (
                Operation::DrawAndIncrement(ShapeDrawId::Shape(root)),
                0,
                viewport
            ),
            (
                Operation::DrawAndIncrement(ShapeDrawId::Shape(nested)),
                1,
                viewport
            ),
            (Operation::Draw(ShapeDrawId::Shape(leaf)), 2, viewport),
            (
                Operation::Decrement(ShapeDrawId::Shape(nested)),
                2,
                viewport
            ),
            (Operation::Draw(ShapeDrawId::Shape(overflow)), 1, viewport),
            (
                Operation::Draw(ShapeDrawId::Shape(overflow_leaf)),
                1,
                viewport
            ),
            (Operation::Draw(ShapeDrawId::Shape(sibling)), 1, viewport),
            (Operation::Decrement(ShapeDrawId::Shape(root)), 1, viewport),
        ]
    );
}

#[test]
fn effect_composites_and_prepared_leaves_carry_only_ids_and_resolved_clips() {
    let mut scene = Scene::new();
    let root = scene.add(None, shape(true));
    let scissor = scene.add(Some(root), clip((10.0, 10.0), (60.0, 60.0)));
    let group = scene.add(Some(scissor), shape(true));
    scene.add(Some(group), shape(true));
    let source = scene.add(Some(scissor), shape(true));
    let child = scene.add(Some(source), shape(true));
    let group_texture = IntermediateTextureId(5);
    let leaf_texture = IntermediateTextureId(6);
    scene.results.insert(group, group_texture);
    let mut description = shape_description(
        Shape::rect([(0.0, 0.0), (20.0, 20.0)], Stroke::default()),
        true,
    );
    description.texture_bindings[0] = ShapeTextureBinding::Intermediate(leaf_texture);
    scene.leaves.insert(
        source,
        PreparedShapeEffectLeaf {
            draw_data: description,
            raster_rect: ShapeEffectRasterRect {
                local_physical_origin: [0, 0],
                texture_size: [20, 20],
                local_bounds: [(0.0, 0.0), (20.0, 20.0)],
            },
        },
    );
    let mut output = DrawPlan::default();
    scene.plan(&mut DrawPlanner::default(), &mut output);
    let viewport = rect((0, 0), (100, 100));
    let inherited = rect((10, 10), (60, 60));
    assert_eq!(
        snapshot(&output),
        [
            (
                Operation::DrawAndIncrement(ShapeDrawId::Shape(root)),
                0,
                viewport
            ),
            (Operation::Composite(group_texture), 1, inherited),
            (
                Operation::Draw(ShapeDrawId::EffectLeaf(source)),
                1,
                inherited
            ),
            (
                Operation::DrawAndIncrement(ShapeDrawId::Shape(source)),
                1,
                inherited
            ),
            (Operation::Draw(ShapeDrawId::Shape(child)), 2, inherited),
            (
                Operation::Decrement(ShapeDrawId::Shape(source)),
                2,
                inherited
            ),
            (Operation::Decrement(ShapeDrawId::Shape(root)), 1, viewport),
        ]
    );
    let DrawOperation::DrawShape(draw) = output.instructions[2].operation else {
        panic!("expected effect draw")
    };
    assert_eq!(
        draw.material.texture_bindings[0],
        ShapeTextureBinding::Intermediate(leaf_texture)
    );
}

#[test]
fn offscreen_scissor_and_transparent_parent_restore_the_visible_sibling() {
    let mut scene = Scene::new();
    let root = scene.add(None, clip((10.0, 10.0), (60.0, 60.0)));
    let offscreen = scene.add(Some(root), clip((150.0, 150.0), (200.0, 200.0)));
    let clipped = scene.add(Some(offscreen), shape(true));
    let sibling = scene.add(Some(root), shape(true));
    let mut output = DrawPlan::default();
    scene.plan(&mut DrawPlanner::default(), &mut output);
    assert_eq!(
        snapshot(&output),
        [
            (
                Operation::Draw(ShapeDrawId::Shape(clipped)),
                0,
                UnsignedPhysicalRect::zero()
            ),
            (
                Operation::Draw(ShapeDrawId::Shape(sibling)),
                0,
                rect((10, 10), (60, 60))
            ),
        ]
    );
}

#[test]
fn backdrop_boundaries_are_complete_before_execution_and_preserve_clips() {
    for clips_children in [false, true] {
        let mut scene = Scene::new();
        let root = scene.add(None, shape(true));
        let scissor = scene.add(Some(root), clip((10.0, 10.0), (60.0, 60.0)));
        let backdrop = scene.add(Some(scissor), shape(clips_children));
        let child = scene.add(Some(backdrop), shape(true));
        let sibling = scene.add(Some(root), shape(true));
        scene.attach_backdrop(backdrop);
        let mut output = DrawPlan::default();
        scene.plan(&mut DrawPlanner::default(), &mut output);
        assert_eq!(output.segments.len(), 3);
        let DrawSegment::Draws(prefix) = &output.segments[0] else {
            panic!("draw prefix")
        };
        assert_eq!(prefix, &(0..1));
        let DrawSegment::Backdrop(command) = &output.segments[1] else {
            panic!("backdrop boundary")
        };
        assert_eq!(command.parent_clip.stencil_reference, 1);
        assert_eq!(command.shape_clip.stencil_reference, 2);
        assert_eq!(command.shape_clip.scissor, rect((10, 10), (60, 60)));
        assert_eq!(command.decrements_stencil, !clips_children);
        assert_eq!(command.effect_id, 42);
        assert_eq!(
            &output.effect_parameters[command.parameter_start..command.parameter_end],
            &[1, 2, 3, 4]
        );
        let viewport = rect((0, 0), (100, 100));
        let mut expected = vec![
            (
                Operation::DrawAndIncrement(ShapeDrawId::Shape(root)),
                0,
                viewport,
            ),
            (
                Operation::Draw(ShapeDrawId::Shape(child)),
                if clips_children { 2 } else { 1 },
                rect((10, 10), (60, 60)),
            ),
        ];
        if clips_children {
            expected.push((
                Operation::Decrement(ShapeDrawId::Shape(backdrop)),
                2,
                rect((10, 10), (60, 60)),
            ));
        }
        expected.extend([
            (Operation::Draw(ShapeDrawId::Shape(sibling)), 1, viewport),
            (Operation::Decrement(ShapeDrawId::Shape(root)), 1, viewport),
        ]);
        assert_eq!(snapshot(&output), expected);
    }
}

#[test]
fn empty_backdrop_parent_preserves_ancestor_clips_without_capture() {
    let mut scene = Scene::new();
    let root = scene.add(None, shape(true));
    let scissor = scene.add(Some(root), clip((10.0, 10.0), (60.0, 60.0)));
    let empty = scene.add(Some(scissor), empty_shape());
    scene.attach_backdrop(empty);
    let child = scene.add(Some(empty), shape(true));
    let sibling = scene.add(Some(root), shape(true));
    let mut output = DrawPlan::default();
    scene.plan(&mut DrawPlanner::default(), &mut output);
    let viewport = rect((0, 0), (100, 100));
    assert_eq!(
        snapshot(&output),
        [
            (
                Operation::DrawAndIncrement(ShapeDrawId::Shape(root)),
                0,
                viewport
            ),
            (
                Operation::Draw(ShapeDrawId::Shape(child)),
                1,
                rect((10, 10), (60, 60))
            ),
            (Operation::Draw(ShapeDrawId::Shape(sibling)), 1, viewport),
            (Operation::Decrement(ShapeDrawId::Shape(root)), 1, viewport),
        ]
    );
    assert_eq!(output.segments.len(), 1);
    assert!(output.effect_parameters.is_empty());
}

#[test]
fn commands_remain_complete_after_planner_and_scene_are_dropped() {
    let mut output = DrawPlan::default();
    {
        let mut scene = Scene::new();
        let root = scene.add(None, shape(true));
        let panel = scene.add(Some(root), shape(true));
        scene.attach_backdrop(panel);
        scene.plan(&mut DrawPlanner::default(), &mut output);
    }
    assert_eq!(output.instructions.len(), 2);
    assert_eq!(output.segments.len(), 3);
    let DrawSegment::Backdrop(command) = &output.segments[1] else {
        panic!("backdrop command")
    };
    assert_eq!(command.effect_id, 42);
    assert_eq!(
        &output.effect_parameters[command.parameter_start..command.parameter_end],
        &[1, 2, 3, 4]
    );
    assert_eq!(command.draw.id, ShapeDrawId::Shape(1));
    assert_eq!(
        command.draw.material.texture_bindings,
        [ShapeTextureBinding::None; 2]
    );
    let _: DrawInstruction = output.instructions[0];
}

#[test]
fn rebuilt_queues_reuse_storage_and_replace_all_commands_and_parameters() {
    let mut scene = Scene::new();
    let mut planner = DrawPlanner::default();
    let mut output = DrawPlan::default();
    planner.parents.reserve(8);
    output.instructions.reserve(16);
    output.segments.reserve(8);
    output.effect_parameters.reserve(32);
    let allocations = (
        planner.parents.as_ptr(),
        output.instructions.as_ptr(),
        output.segments.as_ptr(),
        output.effect_parameters.as_ptr(),
    );
    for has_backdrop in [true, false, true] {
        scene.tree.clear();
        scene.backdrops.clear();
        let root = scene.add(None, shape(true));
        let leaf = scene.add(Some(root), shape(true));
        if has_backdrop {
            scene.attach_backdrop(leaf);
        }
        scene.plan(&mut planner, &mut output);
        assert_eq!(
            allocations,
            (
                planner.parents.as_ptr(),
                output.instructions.as_ptr(),
                output.segments.as_ptr(),
                output.effect_parameters.as_ptr()
            )
        );
        assert_eq!(output.effect_parameters.is_empty(), !has_backdrop);
        assert_eq!(output.segments.len(), if has_backdrop { 3 } else { 1 });
    }
    scene.tree.clear();
    scene.backdrops.clear();
    scene.plan(&mut planner, &mut output);
    assert!(output.instructions.is_empty());
    assert!(output.segments.is_empty());
    assert!(output.effect_parameters.is_empty());
}
