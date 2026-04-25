use ahash::HashMap;

use super::types::DrawCommand;
use crate::effect::EffectInstance;
use crate::vertex::InstanceTransform;

#[derive(Clone, Copy)]
pub(super) struct AxisAlignedRectTransform {
    pub(super) scale_x: f32,
    pub(super) scale_y: f32,
    pub(super) translate_x: f32,
    pub(super) translate_y: f32,
}

pub(super) fn extract_axis_aligned_rect_transform(
    transform: Option<InstanceTransform>,
) -> Option<AxisAlignedRectTransform> {
    let transform = transform.unwrap_or_else(InstanceTransform::identity);

    if transform.col0[3] != 0.0 || transform.col1[3] != 0.0 || transform.col3[3] != 1.0 {
        return None;
    }

    if transform.col0[1] != 0.0 || transform.col1[0] != 0.0 {
        return None;
    }

    Some(AxisAlignedRectTransform {
        scale_x: transform.col0[0],
        scale_y: transform.col1[1],
        translate_x: transform.col3[0],
        translate_y: transform.col3[1],
    })
}

pub(super) fn should_skip_visible_rect_draw(
    node_id: usize,
    draw_command: &DrawCommand,
    group_effects: &HashMap<usize, EffectInstance>,
    backdrop_effects: &HashMap<usize, EffectInstance>,
) -> bool {
    if !draw_command.is_rect() {
        return false;
    }

    if group_effects.contains_key(&node_id) || backdrop_effects.contains_key(&node_id) {
        return false;
    }

    if draw_command.texture_id(0).is_some() || draw_command.texture_id(1).is_some() {
        return false;
    }

    // Gradient-filled shapes are visually active even before GPU prep creates bind groups.
    if draw_command.has_gradient_fill() {
        return false;
    }

    if draw_command
        .instance_color_override()
        .is_some_and(|color| color[3] != 0.0)
    {
        return false;
    }

    extract_axis_aligned_rect_transform(draw_command.transform()).is_some()
}

/// Compute a screen-space scissor rect from a local-space axis-aligned rect and its transform.
///
/// Returns `Some((x, y, width, height))` in physical pixels if the transform preserves
/// axis-alignment (identity, translation, and/or scale — no rotation, skew, or perspective).
/// Returns `None` if scissor clipping cannot be used (the caller should fall back to stencil).
pub(super) fn compute_scissor_rect(
    rect: [(f32, f32); 2],
    transform: Option<InstanceTransform>,
    scale_factor: f64,
    physical_size: (u32, u32),
) -> Option<(u32, u32, u32, u32)> {
    let axis_aligned_transform = extract_axis_aligned_rect_transform(transform)?;

    let x0 = rect[0].0 * axis_aligned_transform.scale_x + axis_aligned_transform.translate_x;
    let y0 = rect[0].1 * axis_aligned_transform.scale_y + axis_aligned_transform.translate_y;
    let x1 = rect[1].0 * axis_aligned_transform.scale_x + axis_aligned_transform.translate_x;
    let y1 = rect[1].1 * axis_aligned_transform.scale_y + axis_aligned_transform.translate_y;

    let min_x = x0.min(x1);
    let min_y = y0.min(y1);
    let max_x = x0.max(x1);
    let max_y = y0.max(y1);

    let scale_factor = scale_factor as f32;
    let px_min_x = ((min_x * scale_factor).floor().max(0.0) as u32).min(physical_size.0);
    let px_min_y = ((min_y * scale_factor).floor().max(0.0) as u32).min(physical_size.1);
    let px_max_x = (max_x * scale_factor).ceil().min(physical_size.0 as f32) as u32;
    let px_max_y = (max_y * scale_factor).ceil().min(physical_size.1 as f32) as u32;

    let width = px_max_x.saturating_sub(px_min_x);
    let height = px_max_y.saturating_sub(px_min_y);

    Some((px_min_x, px_min_y, width, height))
}

/// Intersect two scissor rects, returning the overlapping region.
/// If the rects don't overlap, returns a zero-size rect.
pub(super) fn intersect_scissor(
    a: (u32, u32, u32, u32),
    b: (u32, u32, u32, u32),
) -> (u32, u32, u32, u32) {
    let a_right = a.0 + a.2;
    let a_bottom = a.1 + a.3;
    let b_right = b.0 + b.2;
    let b_bottom = b.1 + b.3;

    let left = a.0.max(b.0);
    let top = a.1.max(b.1);
    let right = a_right.min(b_right);
    let bottom = a_bottom.min(b_bottom);

    let width = right.saturating_sub(left);
    let height = bottom.saturating_sub(top);

    (left, top, width, height)
}

/// Check whether a non-leaf draw command is eligible for scissor clipping,
/// and if so, compute the scissor rect. This centralizes the eligibility logic
/// so pre-visit and post-visit make the same deterministic decision.
pub(super) fn try_scissor_for_rect(
    draw_command: &DrawCommand,
    scale_factor: f64,
    physical_size: (u32, u32),
) -> Option<(u32, u32, u32, u32)> {
    if !draw_command.is_rect() {
        return None;
    }
    let rect_bounds = draw_command.rect_bounds()?;
    let transform = draw_command.transform();
    compute_scissor_rect(rect_bounds, transform, scale_factor, physical_size)
}

#[cfg(test)]
mod tests {
    use super::{compute_scissor_rect, should_skip_visible_rect_draw, try_scissor_for_rect};
    use crate::effect::EffectInstance;
    use crate::gradient::types::{
        ColorInterpolation, Fill, Gradient, GradientStop, GradientStopOffset, LinearGradientDesc,
        LinearGradientLine,
    };
    use crate::renderer::types::DrawCommand;
    use crate::shape::CachedShapeDrawData;
    use crate::util::PoolManager;
    use crate::{CachedShapeHandle, Color, Shape, Stroke, TransformInstance};
    use ahash::{HashMap, HashMapExt};
    use lyon::tessellation::FillTessellator;
    use std::num::NonZeroUsize;

    fn create_test_gradient() -> Gradient {
        Gradient::linear(
            LinearGradientDesc::new(
                LinearGradientLine {
                    start: [0.0, 5.0],
                    end: [10.0, 5.0],
                },
                [
                    GradientStop::at_position(
                        GradientStopOffset::linear_radial(0.0),
                        Color::rgb(255, 0, 0),
                    ),
                    GradientStop::at_position(
                        GradientStopOffset::linear_radial(1.0),
                        Color::rgb(0, 0, 255),
                    ),
                ],
            )
            .with_interpolation(ColorInterpolation::Srgb),
        )
        .expect("valid test gradient")
    }

    fn rect_draw_command() -> DrawCommand {
        let mut tessellator = FillTessellator::new();
        let mut pool = PoolManager::new(NonZeroUsize::new(4).unwrap());
        let shape_handle = CachedShapeHandle::new(
            &Shape::rect([(0.0, 0.0), (10.0, 10.0)], Stroke::default()),
            &mut tessellator,
            &mut pool,
            None,
        );
        DrawCommand::CachedShape(CachedShapeDrawData::new(shape_handle))
    }

    #[test]
    fn axis_aligned_rect_transform_accepts_translation_and_scale() {
        let transform = TransformInstance::affine_2d(2.0, 0.0, 0.0, -3.0, 10.0, 20.0);

        let scissor =
            compute_scissor_rect([(0.0, 0.0), (10.0, 5.0)], Some(transform), 1.0, (100, 100));

        assert_eq!(scissor, Some((10, 5, 20, 15)));
    }

    #[test]
    fn scissor_rejects_non_axis_aligned_transform() {
        let mut draw_command = rect_draw_command();
        draw_command.set_transform(TransformInstance::affine_2d(1.0, 0.0, 0.5, 1.0, 5.0, 5.0));

        assert!(try_scissor_for_rect(&draw_command, 1.0, (100, 100)).is_none());
    }

    #[test]
    fn skip_visible_rect_draw_rejects_effect_nodes() {
        let draw_command = rect_draw_command();
        let node_id = 7usize;

        let mut group_effects = HashMap::new();
        group_effects.insert(
            node_id,
            EffectInstance {
                effect_id: 1,
                params: Vec::new(),
                params_buffer: None,
                params_bind_group: None,
            },
        );

        assert!(!should_skip_visible_rect_draw(
            node_id,
            &draw_command,
            &group_effects,
            &HashMap::new(),
        ));

        let mut backdrop_effects = HashMap::new();
        backdrop_effects.insert(
            node_id,
            EffectInstance {
                effect_id: 2,
                params: Vec::new(),
                params_buffer: None,
                params_bind_group: None,
            },
        );

        assert!(!should_skip_visible_rect_draw(
            node_id,
            &draw_command,
            &HashMap::new(),
            &backdrop_effects,
        ));
    }

    #[test]
    fn skip_visible_rect_draw_accepts_untextured_none_color_rect() {
        let draw_command = rect_draw_command();

        assert!(should_skip_visible_rect_draw(
            1,
            &draw_command,
            &HashMap::new(),
            &HashMap::new(),
        ));
    }

    #[test]
    fn skip_visible_rect_draw_rejects_opaque_color_and_textures() {
        let mut opaque_draw_command = rect_draw_command();
        opaque_draw_command.set_instance_color_override(Some(Color::WHITE.normalize()));

        assert!(!should_skip_visible_rect_draw(
            1,
            &opaque_draw_command,
            &HashMap::new(),
            &HashMap::new(),
        ));

        let mut textured_draw_command = rect_draw_command();
        textured_draw_command.set_texture_id(0, Some(9));

        assert!(!should_skip_visible_rect_draw(
            2,
            &textured_draw_command,
            &HashMap::new(),
            &HashMap::new(),
        ));
    }

    #[test]
    fn skip_visible_rect_draw_rejects_gradient_rects() {
        let mut draw_command = rect_draw_command();

        match &mut draw_command {
            DrawCommand::CachedShape(shape) => {
                shape.fill = Some(Fill::Gradient(create_test_gradient()));
            }
            _ => unreachable!(),
        }

        assert!(!should_skip_visible_rect_draw(
            3,
            &draw_command,
            &HashMap::new(),
            &HashMap::new(),
        ));
    }
}
