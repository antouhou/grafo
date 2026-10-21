use super::append_shape_mask_effect;
use crate::commands::{
    IntermediateTextureId, RenderOperation, ShapeDrawId, Target, TextureComposite, TexturePlacement,
};
use crate::core::effect::{ShapeMaskEffectBounds, ShapeMaskEffectConfig};
use crate::core::shape::CachedShapeHandle;
use crate::core::util::ShapeResources;
use crate::core::vertex::InstanceTransform;
use crate::core::Viewport;
use crate::planner::commands::TraversalCommands;
use crate::planner::tests::RecordedCommands;
use crate::planner::TextureIdAllocator;
use crate::scene_state::effects::parameters::EffectParameterStorage;
use crate::scene_state::effects::parameters::{EffectParameterRange, EffectParametersHandle};
use crate::scene_state::effects::ShapeMaskEffectInstance;
use crate::scene_state::types::CachedShapeDrawData;
use crate::scene_state::types::DrawTreeNode;
use crate::{Shape, ShapeDrawCommandOptions, Size};
use ahash::{HashMap, HashMapExt};
use easy_tree::Tree;
use lyon::tessellation::FillTessellator;
use std::mem;

#[derive(Default)]
struct MaskCommands<'scene> {
    commands: RecordedCommands<'scene>,
    composites: HashMap<usize, TextureComposite>,
}

impl<'scene> MaskCommands<'scene> {
    fn new() -> Self {
        Self::default()
    }

    fn plan(
        &mut self,
        tree: &Tree<DrawTreeNode>,
        effects: &HashMap<usize, ShapeMaskEffectInstance>,
        parameters: &'scene EffectParameterStorage,
        scale: f64,
        size: Size,
        limit: u32,
    ) {
        self.commands.clear_commands();
        self.composites.clear();
        let mut texture_ids = TextureIdAllocator::default();
        let mut commands = TraversalCommands::default();
        for (&node_id, effect) in effects {
            append_shape_mask_effect(
                &mut commands,
                &mut texture_ids,
                &mut self.composites,
                tree,
                node_id,
                effect,
                parameters,
                Viewport {
                    physical_size: (size.width, size.height),
                    scale_factor: scale,
                },
                limit,
            );
            self.commands.instructions.extend(mem::take(&mut commands));
        }
    }
}

fn scene_with_effects(
    parameters: &mut EffectParameterStorage,
    scale: f64,
) -> (Tree<DrawTreeNode>, HashMap<usize, ShapeMaskEffectInstance>) {
    let mut tree = Tree::new();
    let mut effects = HashMap::new();
    let mut resources = ShapeResources::new();
    let handle = CachedShapeHandle::new(
        &Shape::rect([(0.0, 0.0), (20.0, 10.0)]),
        &mut FillTessellator::new(),
        &mut resources,
        Some(17),
    );
    for translation in [5.0, 45.0] {
        let transform = InstanceTransform::translation(translation, 8.0);
        let config = ShapeMaskEffectConfig::new().outset(3.0).downsample(0.5);
        let node = tree.add_node(DrawTreeNode::CachedShape(CachedShapeDrawData::new(
            handle.clone(),
            ShapeDrawCommandOptions::new().transform(transform),
        )));
        effects.insert(
            node,
            ShapeMaskEffectInstance {
                effect_id: 9,
                parameters: parameters.store(&[node as u8, 2, 3, 4]),
                config,
                bounds: ShapeMaskEffectBounds::new(
                    handle.tessellation.local_bounds,
                    config,
                    Some(transform),
                    scale,
                )
                .unwrap(),
            },
        );
    }
    (tree, effects)
}

#[test]
fn mask_scopes_resolve_geometry_and_borrow_the_parameter_storage() {
    let mut parameters = EffectParameterStorage::default();
    let mut plan = MaskCommands::new();
    {
        let (tree, effects) = scene_with_effects(&mut parameters, 2.0);
        plan.plan(&tree, &effects, &parameters, 2.0, Size::new(200, 100), 1024);
    }
    assert_eq!(plan.commands.instructions.len(), 8);
    for (index, commands) in plan
        .commands
        .instructions
        .as_chunks::<4>()
        .0
        .iter()
        .enumerate()
    {
        let [RenderOperation::BeginTarget(Target::Mask(target)), RenderOperation::DrawShapeMask(mask), RenderOperation::EndTarget, RenderOperation::ApplyEffect(effect)] =
            commands.each_ref().map(|command| &command.operation)
        else {
            panic!("expected completed mask scope before the effect")
        };
        assert_eq!(target.texture, IntermediateTextureId::Planned(index * 2));
        assert_eq!(target.size, [27, 17]);
        assert_eq!(mask.local_physical_origin, [-7, -7]);
        assert_eq!(mask.local_bounds, [(-3.5, -3.5), (23.5, 13.5)]);
        assert_eq!(effect.input, target.texture);
        assert_eq!(effect.output, IntermediateTextureId::Planned(index * 2 + 1));
        assert_eq!(effect.effect_id, 9);
        let ShapeDrawId(node) = mask.shape;
        assert_eq!(effect.parameters.bytes, [node as u8, 2, 3, 4]);
        let composite = plan.composites[&node];
        assert_eq!(composite.texture, effect.output);
        let TexturePlacement::Local {
            transform,
            sampling,
        } = composite.placement
        else {
            panic!("expected source-local placement")
        };
        assert_eq!(transform.col0, [27.0, 0.0, 0.0, 0.0]);
        assert_eq!(transform.col1, [0.0, 17.0, 0.0, 0.0]);
        assert_eq!(
            transform.col3,
            [if node == 0 { 1.5 } else { 41.5 }, 4.5, 0.0, 1.0]
        );
        assert_eq!(sampling.scale, [1.0, 1.0]);
    }
}

#[test]
fn replanning_masks_reuses_composite_storage_and_drops_removed_outputs() {
    let mut parameters = EffectParameterStorage::default();
    let (mut tree, mut effects) = scene_with_effects(&mut parameters, 1.0);
    let mut plan = MaskCommands::new();
    plan.plan(&tree, &effects, &parameters, 1.0, Size::new(100, 100), 1024);
    let composite_capacity = plan.composites.capacity();
    for viewport in [Size::new(100, 100), Size::new(1, 1), Size::new(100, 100)] {
        plan.plan(&tree, &effects, &parameters, 1.0, viewport, 1024);
        assert_eq!(plan.composites.capacity(), composite_capacity);
        assert_eq!(plan.composites.is_empty(), viewport.width == 1);
    }
    tree.clear();
    effects.clear();
    plan.plan(&tree, &effects, &parameters, 1.0, Size::new(100, 100), 1024);
    assert!(plan.commands.instructions.is_empty());
    assert!(plan.composites.is_empty());
}

#[test]
fn empty_or_oversized_masks_never_produce_texture_references() {
    let mut parameters = EffectParameterStorage::default();
    let mut plan = MaskCommands::new();
    let (mut tree, mut effects) = scene_with_effects(&mut parameters, 1.0);
    let empty_shape = CachedShapeHandle::new(
        &Shape::builder().build(),
        &mut FillTessellator::new(),
        &mut ShapeResources::new(),
        None,
    );
    let config = ShapeMaskEffectConfig::default();
    let bounds =
        ShapeMaskEffectBounds::new(empty_shape.tessellation.local_bounds, config, None, 1.0)
            .unwrap();
    let empty = tree.add_node(DrawTreeNode::CachedShape(CachedShapeDrawData::new(
        empty_shape,
        ShapeDrawCommandOptions::new(),
    )));
    effects.insert(
        empty,
        ShapeMaskEffectInstance {
            effect_id: 9,
            parameters: EffectParametersHandle {
                range: EffectParameterRange { start: 0, end: 0 },
                hash: 0,
            },
            config,
            bounds,
        },
    );
    plan.plan(&tree, &effects, &parameters, 1.0, Size::new(100, 100), 1);
    assert!(plan.commands.instructions.is_empty());
    assert!(plan.composites.is_empty());
    plan.plan(&tree, &effects, &parameters, 1.0, Size::new(100, 100), 1024);
    assert_eq!(plan.composites.len(), 2);
    assert!(!plan.composites.contains_key(&empty));
}
