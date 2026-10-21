use crate::commands::{
    DrawClip, EffectApplication, MaskTarget, RenderOperation, ShapeDrawId, ShapeMaskDraw, Target,
    TextureComposite, TexturePlacement,
};
use crate::core::vertex::TextureUvTransform;
use crate::core::{effect, geometry};
use crate::core::{Size, UnsignedPhysicalRect, Viewport};
use crate::planner::commands::TraversalCommands;
use crate::planner::TextureIdAllocator;
use crate::scene_state::effects::parameters::EffectParameterStorage;
use crate::scene_state::effects::ShapeMaskEffectInstance;
use crate::scene_state::types::DrawTreeNode;
use ahash::HashMap;
use easy_tree::Tree;

#[allow(clippy::too_many_arguments)]
pub(super) fn append_shape_mask_effect<'scene>(
    commands: &mut TraversalCommands<'scene>,
    texture_ids: &mut TextureIdAllocator,
    composites: &mut HashMap<usize, TextureComposite>,
    tree: &Tree<DrawTreeNode>,
    node_id: usize,
    effect: &ShapeMaskEffectInstance,
    parameters: &'scene EffectParameterStorage,
    viewport: Viewport,
    maximum_texture_dimension: u32,
) {
    let Some(DrawTreeNode::CachedShape(shape)) = tree.get(node_id) else {
        return;
    };
    if !shape.has_geometry() {
        return;
    }
    let bounds = effect.bounds;
    let [width, height] = bounds.texture_size;
    if effect::validate_texture_size(
        Size::new(width, height),
        viewport.physical_size.into(),
        maximum_texture_dimension,
    )
    .is_err()
    {
        tracing::warn!(node_id, "skipping oversized shape mask effect texture");
        return;
    }
    let mask = texture_ids.allocate();
    let output = texture_ids.allocate();
    commands.push(RenderOperation::BeginTarget(Target::Mask(MaskTarget {
        texture: mask,
        size: bounds.texture_size,
    })));
    commands.push(RenderOperation::DrawShapeMask(ShapeMaskDraw {
        shape: ShapeDrawId(node_id),
        clip: DrawClip {
            scissor: UnsignedPhysicalRect::from_size(Size::new(width, height)),
            stencil_reference: 0,
        },
        local_physical_origin: bounds.local_physical_origin,
        local_bounds: bounds.local_bounds,
        scale_factor: viewport.scale_factor,
        downsample: effect.config.downsample,
    }));
    commands.push(RenderOperation::EndTarget);
    let parameters = parameters.get(effect.parameters);
    commands.push(RenderOperation::ApplyEffect(EffectApplication {
        effect_id: effect.effect_id,
        parameters,
        input: mask,
        output,
    }));
    composites.insert(
        node_id,
        TextureComposite {
            texture: output,
            placement: TexturePlacement::Local {
                transform: geometry::unit_quad_transform(
                    bounds.local_bounds,
                    shape.instance.transform,
                ),
                sampling: TextureUvTransform::IDENTITY,
            },
        },
    );
}

#[cfg(test)]
mod tests;
