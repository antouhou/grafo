use super::append_shape_effects;
use crate::commands::{
    IntermediateTextureId, RenderOperation, RenderPlan, ShapeDrawId, Target, TextureComposite,
    TexturePlacement,
};
use crate::core::effect::{ShapeEffectConfig, ShapeEffectInstance};
use crate::core::shape::CachedShapeHandle;
use crate::core::util::ShapeResources;
use crate::core::vertex::InstanceTransform;
use crate::core::Viewport;
use crate::renderer::types::CachedShapeDrawData;
use crate::renderer::types::DrawTreeNode;
use crate::{Shape, ShapeDrawCommandOptions, Size, Stroke};
use ahash::{HashMap, HashMapExt};
use easy_tree::Tree;
use lyon::tessellation::FillTessellator;
use std::sync::Arc;

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
        self.commands.clear();
        append_shape_effects(
            &mut self.commands,
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

fn scene_with_effects() -> (Tree<DrawTreeNode>, HashMap<usize, ShapeEffectInstance>) {
    let mut tree = Tree::new();
    let mut effects = HashMap::new();
    let mut resources = ShapeResources::new();
    let handle = CachedShapeHandle::new(
        &Shape::rect([(0.0, 0.0), (20.0, 10.0)], Stroke::default()),
        &mut FillTessellator::new(),
        &mut resources,
        Some(17),
    );
    for translation in [5.0, 45.0] {
        let node = tree.add_node(DrawTreeNode::CachedShape(CachedShapeDrawData::new(
            handle.clone(),
            &ShapeDrawCommandOptions::new()
                .transform(InstanceTransform::translation(translation, 8.0)),
        )));
        effects.insert(
            node,
            ShapeEffectInstance {
                effect_id: 9,
                params: Arc::from([node as u8, 2, 3, 4]),
                config: ShapeEffectConfig::new().outset(3.0).downsample(0.5),
            },
        );
    }
    (tree, effects)
}

#[test]
fn mask_scopes_and_effect_outputs_are_complete_without_the_scene() {
    let mut plan = MaskCommands::new();
    {
        let (tree, effects) = scene_with_effects();
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
    let (mut tree, mut effects) = scene_with_effects();
    let mut plan = MaskCommands::new();
    plan.plan(&tree, &effects, 1.0, 0.75, Size::new(100, 100), 1024);
    let commands_pointer = plan.commands.instructions.as_ptr();
    let parameters_pointer = plan.commands.shared_effect_parameters.as_ptr();
    let composite_capacity = plan.composites.capacity();
    tree.clear();
    effects.clear();
    let (rebuilt_tree, rebuilt_effects) = scene_with_effects();
    for viewport in [Size::new(100, 100), Size::new(1, 1), Size::new(100, 100)] {
        plan.plan(&rebuilt_tree, &rebuilt_effects, 1.0, 0.75, viewport, 1024);
        assert_eq!(plan.commands.instructions.as_ptr(), commands_pointer);
        assert_eq!(
            plan.commands.shared_effect_parameters.as_ptr(),
            parameters_pointer
        );
        assert_eq!(plan.composites.capacity(), composite_capacity);
        assert_eq!(plan.composites.is_empty(), viewport.width == 1);
    }
    plan.plan(&tree, &effects, 1.0, 0.75, Size::new(100, 100), 1024);
    assert!(plan.commands.instructions.is_empty());
    assert!(plan.commands.shared_effect_parameters.is_empty());
    assert!(plan.composites.is_empty());
}

#[test]
fn invalid_or_empty_masks_never_produce_texture_references() {
    let (mut tree, mut effects) = scene_with_effects();
    let empty = tree.add_node(DrawTreeNode::CachedShape(CachedShapeDrawData::new(
        CachedShapeHandle::new(
            &Shape::builder().build(),
            &mut FillTessellator::new(),
            &mut ShapeResources::new(),
            None,
        ),
        &ShapeDrawCommandOptions::new(),
    )));
    effects.insert(
        empty,
        ShapeEffectInstance {
            effect_id: 9,
            params: Arc::from([]),
            config: ShapeEffectConfig::default(),
        },
    );
    let mut plan = MaskCommands::new();
    for (scale, max_dimension) in [(f64::NAN, 1024), (1.0, 1)] {
        plan.plan(
            &tree,
            &effects,
            scale,
            0.75,
            Size::new(100, 100),
            max_dimension,
        );
        assert!(plan.commands.instructions.is_empty());
        assert!(plan.composites.is_empty());
    }
    plan.plan(&tree, &effects, 1.0, 0.75, Size::new(100, 100), 1024);
    assert_eq!(plan.composites.len(), 2);
    assert!(!plan.composites.contains_key(&empty));
}
