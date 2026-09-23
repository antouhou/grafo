use crate::effect::{ShapeEffectConfig, ShapeEffectInstance};
use crate::renderer::commands::{
    DrawClip, DrawPlan, DrawSegment, EffectApplication, EffectParameters, IntermediateTextureId,
    MaskTarget, ShapeDrawId, ShapeMaskDraw, Target, TextureComposite, TexturePlacement,
};
use crate::renderer::rect_utils::compute_downsampled_dimensions;
use crate::renderer::types::DrawTreeNode;
use crate::vertex::{InstanceTransform, TextureUvTransform};
use crate::{Size, UnsignedPhysicalRect};
use ahash::{HashMap, HashMapExt};
use easy_tree::Tree;
use std::sync::Arc;

#[derive(Copy, Clone, Debug, PartialEq)]
struct ShapeEffectRasterRect {
    pub local_physical_origin: [i32; 2],
    pub texture_size: [u32; 2],
    pub local_bounds: [(f32, f32); 2],
}

fn compute_shape_effect_raster_rect(
    local_bounds: [(f32, f32); 2],
    config: ShapeEffectConfig,
    scale_factor: f64,
    fringe_width: f32,
) -> Option<ShapeEffectRasterRect> {
    let bounds_and_outsets = [
        local_bounds[0].0,
        local_bounds[0].1,
        local_bounds[1].0,
        local_bounds[1].1,
        config.left_outset,
        config.top_outset,
        config.right_outset,
        config.bottom_outset,
    ];
    if !scale_factor.is_finite()
        || scale_factor <= 0.0
        || !fringe_width.is_finite()
        || fringe_width < 0.0
        || !config.downsample.is_finite()
        || config.downsample <= 0.0
        || config.downsample > 1.0
        || !bounds_and_outsets.iter().all(|value| value.is_finite())
    {
        return None;
    }

    let minimum_x = local_bounds[0].0.min(local_bounds[1].0) - config.left_outset;
    let minimum_y = local_bounds[0].1.min(local_bounds[1].1) - config.top_outset;
    let maximum_x = local_bounds[0].0.max(local_bounds[1].0) + config.right_outset;
    let maximum_y = local_bounds[0].1.max(local_bounds[1].1) + config.bottom_outset;
    if ![minimum_x, minimum_y, maximum_x, maximum_y]
        .iter()
        .all(|value| value.is_finite())
    {
        return None;
    }

    let guard = f64::from(fringe_width).ceil();
    let physical_minimum_x = (f64::from(minimum_x) * scale_factor).floor() - guard;
    let physical_minimum_y = (f64::from(minimum_y) * scale_factor).floor() - guard;
    let physical_maximum_x = (f64::from(maximum_x) * scale_factor).ceil() + guard;
    let physical_maximum_y = (f64::from(maximum_y) * scale_factor).ceil() + guard;

    let coordinates = [
        physical_minimum_x,
        physical_minimum_y,
        physical_maximum_x,
        physical_maximum_y,
    ];
    if !coordinates.iter().all(|value| {
        value.is_finite() && *value >= f64::from(i32::MIN) && *value <= f64::from(i32::MAX)
    }) {
        return None;
    }

    let local_physical_origin = [physical_minimum_x as i32, physical_minimum_y as i32];
    let physical_width = physical_maximum_x - physical_minimum_x;
    let physical_height = physical_maximum_y - physical_minimum_y;
    if physical_width <= 0.0
        || physical_height <= 0.0
        || physical_width > f64::from(u32::MAX)
        || physical_height > f64::from(u32::MAX)
    {
        return None;
    }

    let full_resolution_size = Size::new(physical_width as u32, physical_height as u32);
    let texture_size = compute_downsampled_dimensions(full_resolution_size, config.downsample);
    Some(ShapeEffectRasterRect {
        local_physical_origin,
        texture_size: texture_size.to_array(),
        local_bounds: [
            (
                physical_minimum_x as f32 / scale_factor as f32,
                physical_minimum_y as f32 / scale_factor as f32,
            ),
            (
                physical_maximum_x as f32 / scale_factor as f32,
                physical_maximum_y as f32 / scale_factor as f32,
            ),
        ],
    })
}

fn shape_effect_quad_transform(
    local_bounds: [(f32, f32); 2],
    source_transform: Option<InstanceTransform>,
) -> InstanceTransform {
    let [(minimum_x, minimum_y), (maximum_x, maximum_y)] = local_bounds;
    let bounds_transform = InstanceTransform::affine_2d(
        maximum_x - minimum_x,
        0.0,
        0.0,
        maximum_y - minimum_y,
        minimum_x,
        minimum_y,
    );
    source_transform.map_or(bounds_transform, |transform| {
        bounds_transform.then(&transform)
    })
}

/// Completed mask commands and placements, independent of execution caches.
pub(in crate::renderer) struct ShapeEffectPlan {
    pub commands: DrawPlan,
    pub composites: HashMap<usize, TextureComposite>,
}

impl ShapeEffectPlan {
    pub fn new() -> Self {
        Self {
            commands: DrawPlan::default(),
            composites: HashMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.commands.clear();
        self.composites.clear();
    }

    pub fn plan(
        &mut self,
        tree: &Tree<DrawTreeNode>,
        effects: &HashMap<usize, ShapeEffectInstance>,
        scale_factor: f64,
        fringe_width: f32,
        physical_size: Size,
        maximum_texture_dimension: u32,
    ) {
        self.clear();
        let maximum_texel_count =
            (u64::from(physical_size.width) * u64::from(physical_size.height)).saturating_mul(4);
        for (&node_id, effect) in effects {
            let Some(DrawTreeNode::CachedShape(shape)) = tree.get(node_id) else {
                continue;
            };
            let geometry = shape.cached_shape.vertex_buffers();
            if geometry.vertices.is_empty() || geometry.indices.is_empty() {
                continue;
            }
            let Some(raster_rect) = compute_shape_effect_raster_rect(
                shape.cached_shape.tessellation.local_bounds,
                effect.config,
                scale_factor,
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
            let mask = self.commands.allocate_texture();
            let output = IntermediateTextureId::ShapeEffect(self.composites.len());
            self.commands
                .segments
                .push(DrawSegment::BeginTarget(Target::Mask(MaskTarget {
                    texture: mask,
                    size: raster_rect.texture_size,
                })));
            self.commands
                .segments
                .push(DrawSegment::DrawShapeMask(ShapeMaskDraw {
                    shape: ShapeDrawId(node_id),
                    clip: DrawClip {
                        scissor: UnsignedPhysicalRect::from_size(Size::new(width, height)),
                        stencil_reference: 0,
                    },
                    local_physical_origin: raster_rect.local_physical_origin,
                    local_bounds: raster_rect.local_bounds,
                    scale_factor,
                    fringe_width,
                    downsample: effect.config.downsample,
                }));
            self.commands.segments.push(DrawSegment::EndTarget);
            self.commands
                .segments
                .push(DrawSegment::ApplyEffect(EffectApplication {
                    effect_id: effect.effect_id,
                    parameters: EffectParameters::Shared(Arc::clone(&effect.params)),
                    input: mask,
                    output,
                }));
            self.composites.insert(
                node_id,
                TextureComposite {
                    texture: output,
                    placement: TexturePlacement::Local {
                        transform: shape_effect_quad_transform(
                            raster_rect.local_bounds,
                            shape.transform,
                        ),
                        sampling: TextureUvTransform::IDENTITY,
                    },
                },
            );
        }
    }
}

#[cfg(test)]
mod tests;
