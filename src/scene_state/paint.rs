use super::types::{CachedShapeDrawData, DrawTreeNode};
use super::{SceneError, SceneState};
use crate::core::{geometry, Fill, GradientUnits, GroupDrawCommandOptions, TransformInstance};
use easy_tree::Tree;
use lyon::geom::euclid::default::Transform3D;
use std::ptr;

fn fills_match(left: Option<&Fill>, right: Option<&Fill>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(Fill::Solid(left)), Some(Fill::Solid(right))) => left == right,
        (Some(Fill::Gradient(left)), Some(Fill::Gradient(right))) => {
            ptr::eq(left.data(), right.data())
        }
        _ => false,
    }
}

#[derive(Debug)]
pub(crate) struct GroupFill {
    fill: Fill,
    scene_to_paint_transform: Option<Transform3D<f64>>,
}

impl GroupFill {
    pub(crate) fn new(options: GroupDrawCommandOptions) -> Result<Option<Self>, SceneError> {
        let Some(fill) = options.fill else {
            return Ok(None);
        };
        let scene_to_paint_transform = match &fill {
            Fill::Gradient(gradient) if gradient.data().units == GradientUnits::Local => {
                let projection = geometry::projected_transform(
                    options
                        .paint_to_scene_transform
                        .unwrap_or_else(TransformInstance::identity),
                );
                let inverse = projection
                    .inverse()
                    .ok_or(SceneError::InvalidPaintTransform)?;
                if !inverse
                    .to_arrays()
                    .iter()
                    .flatten()
                    .all(|value| value.is_finite())
                {
                    return Err(SceneError::InvalidPaintTransform);
                }
                Some(inverse)
            }
            _ => None,
        };
        Ok(Some(Self {
            fill,
            scene_to_paint_transform,
        }))
    }

    fn shape_transform(&self, transform: Option<TransformInstance>) -> Transform3D<f32> {
        self.scene_to_paint_transform
            .map_or_else(Transform3D::identity, |inverse| {
                geometry::projected_transform(transform.unwrap_or_else(TransformInstance::identity))
                    .then(&inverse)
                    .cast()
            })
    }
}

fn resolve_inherited_fill(
    tree: &mut Tree<DrawTreeNode>,
    id: usize,
    source: Option<usize>,
) -> Option<&CachedShapeDrawData> {
    let DrawTreeNode::CachedShape(shape) = tree.get_unchecked(id) else {
        return None;
    };
    if !shape.inherits_fill {
        return None;
    }
    let group_fill = source.and_then(|source| match tree.get_unchecked(source) {
        DrawTreeNode::Group(group) => group.fill.as_ref(),
        _ => None,
    });
    let fill = group_fill.map(|paint| &paint.fill);
    let transform = group_fill.map_or_else(Transform3D::identity, |paint| {
        paint.shape_transform(shape.instance.transform)
    });
    if fills_match(shape.instance.fill.as_ref(), fill)
        && shape.instance.paint_transform == transform
    {
        return None;
    }
    let fill = fill.cloned();
    let DrawTreeNode::CachedShape(shape) = tree.get_mut(id)? else {
        return None;
    };
    shape.instance.color_override = fill.as_ref().and_then(Fill::to_normalized_solid);
    shape.instance.fill = fill;
    shape.instance.paint_transform = transform;
    Some(shape)
}

impl SceneState {
    pub(crate) fn prepare_inherited_fills(
        &mut self,
        mut updated: impl FnMut(usize, &CachedShapeDrawData),
    ) {
        if !self.has_pending_fill_changes {
            return;
        }
        self.fill_source_scratch.clear();
        if !self.draw_tree.is_empty() {
            self.fill_source_scratch.push((0, None));
        }
        while let Some((id, mut source)) = self.fill_source_scratch.pop() {
            if let DrawTreeNode::Group(group) = self.draw_tree.get_unchecked(id) {
                if group.fill.is_some() {
                    source = Some(id);
                }
            }
            if let Some(shape) = resolve_inherited_fill(&mut self.draw_tree, id, source) {
                updated(id, shape);
            }
            self.fill_source_scratch.extend(
                self.draw_tree
                    .children(id)
                    .iter()
                    .map(|&child| (child, source)),
            );
        }
        self.has_pending_fill_changes = false;
    }
}
