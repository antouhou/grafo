use super::append_shape_effects;
use crate::commands::{
    EffectParameterRange, EffectParameters, IntermediateTextureId, RenderOperation, RenderPlan,
    ShapeDrawId, Target, TextureComposite, TexturePlacement,
};
use crate::core::effect::{ShapeEffectBounds, ShapeEffectConfig};
use crate::core::shape::CachedShapeHandle;
use crate::core::util::ShapeResources;
use crate::core::vertex::InstanceTransform;
use crate::core::Viewport;
use crate::planner::TextureIdAllocator;
use crate::scene::effects::ShapeEffectInstance;
use crate::scene::types::CachedShapeDrawData;
use crate::scene::types::DrawTreeNode;
use crate::{Shape, ShapeDrawCommandOptions, Size};
use ahash::{HashMap, HashMapExt};
use easy_tree::Tree;
use lyon::tessellation::FillTessellator;

#[derive(Default)]
struct MaskCommands {
    commands: RenderPlan,
    composites: HashMap<usize, TextureComposite>,
}

impl MaskCommands {
    fn new() -> Self {
        Self::default()
    }

    fn plan(
        &mut self,
        tree: &Tree<DrawTreeNode>,
        effects: &HashMap<usize, ShapeEffectInstance>,
        scale: f64,
        fringe: f32,
        size: Size,
        limit: u32,
    ) {
        self.commands.clear_commands();
        append_shape_effects(
            &mut self.commands,
            &mut TextureIdAllocator::default(),
            &mut self.composites,
            tree,
            effects,
            Viewport {
                physical_size: (size.width, size.height),
                scale_factor: scale,
            },
            fringe,
            limit,
        );
    }
}

fn scene_with_effects(
    commands: &mut RenderPlan,
    scale: f64,
    fringe: f32,
) -> (Tree<DrawTreeNode>, HashMap<usize, ShapeEffectInstance>) {
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
        let config = ShapeEffectConfig::new().outset(3.0).downsample(0.5);
        let node = tree.add_node(DrawTreeNode::CachedShape(CachedShapeDrawData::new(
            handle.clone(),
            ShapeDrawCommandOptions::new().transform(transform),
        )));
        effects.insert(
            node,
            ShapeEffectInstance {
                effect_id: 9,
                parameters: commands.store_parameters(&[node as u8, 2, 3, 4]),
                config,
                bounds: ShapeEffectBounds::new(
                    handle.tessellation.local_bounds,
                    config,
                    Some(transform),
                    scale,
                    fringe,
                )
                .unwrap(),
            },
        );
    }
    (tree, effects)
}

#[test]
fn mask_scopes_and_effect_outputs_are_complete_without_the_scene() {
    let mut plan = MaskCommands::new();
    {
        let (tree, effects) = scene_with_effects(&mut plan.commands, 2.0, 0.75);
        plan.plan(&tree, &effects, 2.0, 0.75, Size::new(200, 100), 1024);
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
        assert_eq!(
            plan.commands.parameters(effect.parameters),
            [node as u8, 2, 3, 4]
        );
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
fn rebuilding_shape_effect_commands_reuses_storage_and_drops_removed_outputs() {
    let mut plan = MaskCommands::new();
    let (mut tree, mut effects) = scene_with_effects(&mut plan.commands, 1.0, 0.75);
    plan.plan(&tree, &effects, 1.0, 0.75, Size::new(100, 100), 1024);
    let commands_pointer = plan.commands.instructions.as_ptr();
    let parameters_pointer = plan.commands.effect_parameters.as_ptr();
    let composite_capacity = plan.composites.capacity();
    tree.clear();
    effects.clear();
    plan.commands.clear();
    let (rebuilt_tree, rebuilt_effects) = scene_with_effects(&mut plan.commands, 1.0, 0.75);
    for viewport in [Size::new(100, 100), Size::new(1, 1), Size::new(100, 100)] {
        plan.plan(&rebuilt_tree, &rebuilt_effects, 1.0, 0.75, viewport, 1024);
        assert_eq!(plan.commands.instructions.as_ptr(), commands_pointer);
        assert_eq!(plan.commands.effect_parameters.as_ptr(), parameters_pointer);
        assert_eq!(plan.composites.capacity(), composite_capacity);
        assert_eq!(plan.composites.is_empty(), viewport.width == 1);
    }
    plan.commands.clear();
    plan.plan(&tree, &effects, 1.0, 0.75, Size::new(100, 100), 1024);
    assert!(plan.commands.instructions.is_empty());
    assert!(plan.commands.effect_parameters.is_empty());
    assert!(plan.composites.is_empty());
}

#[test]
fn empty_or_oversized_masks_never_produce_texture_references() {
    let mut plan = MaskCommands::new();
    let (mut tree, mut effects) = scene_with_effects(&mut plan.commands, 1.0, 0.75);
    let empty_shape = CachedShapeHandle::new(
        &Shape::builder().build(),
        &mut FillTessellator::new(),
        &mut ShapeResources::new(),
        None,
    );
    let config = ShapeEffectConfig::default();
    let bounds = ShapeEffectBounds::new(
        empty_shape.tessellation.local_bounds,
        config,
        None,
        1.0,
        0.75,
    )
    .unwrap();
    let empty = tree.add_node(DrawTreeNode::CachedShape(CachedShapeDrawData::new(
        empty_shape,
        ShapeDrawCommandOptions::new(),
    )));
    effects.insert(
        empty,
        ShapeEffectInstance {
            effect_id: 9,
            parameters: EffectParameters {
                range: EffectParameterRange { start: 0, end: 0 },
                hash: 0,
            },
            config,
            bounds,
        },
    );
    plan.plan(&tree, &effects, 1.0, 0.75, Size::new(100, 100), 1);
    assert!(plan.commands.instructions.is_empty());
    assert!(plan.composites.is_empty());
    plan.plan(&tree, &effects, 1.0, 0.75, Size::new(100, 100), 1024);
    assert_eq!(plan.composites.len(), 2);
    assert!(!plan.composites.contains_key(&empty));
}
