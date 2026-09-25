use crate::commands::{
    DrawClip, EffectApplication, MaskTarget, RenderOperation, RenderPlan, ShapeDrawId,
    ShapeMaskDraw, Target, TextureComposite, TexturePlacement,
};
use crate::core::effect;
use crate::core::geometry;
use crate::core::vertex::TextureUvTransform;
use crate::core::{Size, UnsignedPhysicalRect, Viewport};
use crate::scene::effects::ShapeEffectInstance;
use crate::scene::types::DrawTreeNode;
use ahash::HashMap;
use easy_tree::Tree;

pub(super) fn append_shape_effects(
    commands: &mut RenderPlan,
    composites: &mut HashMap<usize, TextureComposite>,
    tree: &Tree<DrawTreeNode>,
    effects: &HashMap<usize, ShapeEffectInstance>,
    viewport: Viewport,
    fringe_width: f32,
    maximum_texture_dimension: u32,
) {
    composites.clear();
    let (width, height) = viewport.physical_size;
    let maximum_texel_count = (u64::from(width) * u64::from(height)).saturating_mul(4);
    for (&node_id, effect) in effects {
        let Some(DrawTreeNode::CachedShape(shape)) = tree.get(node_id) else {
            continue;
        };
        let geometry = shape.instance.cached_shape.vertex_buffers();
        if geometry.vertices.is_empty() || geometry.indices.is_empty() {
            continue;
        }
        let Some(raster_rect) = effect::compute_shape_effect_raster_rect(
            shape.instance.cached_shape.tessellation.local_bounds,
            effect.config,
            viewport.scale_factor,
            fringe_width,
        ) else {
            continue;
        };
        let [width, height] = raster_rect.texture_size;
        if width > maximum_texture_dimension
            || height > maximum_texture_dimension
            || u64::from(width) * u64::from(height) > maximum_texel_count
        {
            tracing::warn!(node_id, "skipping oversized shape effect texture");
            continue;
        }
        let mask = commands.allocate_texture();
        let output = commands.allocate_texture();
        commands.push(RenderOperation::BeginTarget(Target::Mask(MaskTarget {
            texture: mask,
            size: raster_rect.texture_size,
        })));
        commands.push(RenderOperation::DrawShapeMask(ShapeMaskDraw {
            shape: ShapeDrawId(node_id),
            clip: DrawClip {
                scissor: UnsignedPhysicalRect::from_size(Size::new(width, height)),
                stencil_reference: 0,
            },
            local_physical_origin: raster_rect.local_physical_origin,
            local_bounds: raster_rect.local_bounds,
            scale_factor: viewport.scale_factor,
            fringe_width,
            downsample: effect.config.downsample,
        }));
        commands.push(RenderOperation::EndTarget);
        let parameters = effect.parameters;
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
                        raster_rect.local_bounds,
                        shape.instance.transform,
                    ),
                    sampling: TextureUvTransform::IDENTITY,
                },
            },
        );
    }
}

#[cfg(test)]
mod tests;
