use crate::core::shape::{CachedShapeHandle, ShapeDrawCommandOptions, ShapeInstance};
use crate::core::vertex::InstanceTransform;

#[derive(Debug)]
pub(crate) struct CachedShapeDrawData {
    pub(crate) instance: ShapeInstance,
    pub(crate) is_leaf: bool,
    pub(crate) clips_children: bool,
}

impl CachedShapeDrawData {
    pub(crate) fn new(cached_shape: CachedShapeHandle, options: ShapeDrawCommandOptions) -> Self {
        let clips_children = options.clips_children;
        Self {
            instance: ShapeInstance::new(cached_shape, options),
            is_leaf: true,
            clips_children,
        }
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub(crate) enum DrawTreeNode {
    CachedShape(CachedShapeDrawData),
    ClipRect(ClipRectDrawData),
}

#[derive(Debug)]
pub(crate) struct ClipRectDrawData {
    pub(crate) rect_bounds: [(f32, f32); 2],
    pub(crate) transform: Option<InstanceTransform>,
    pub(crate) is_leaf: bool,
    pub(crate) clips_children: bool,
}

impl ClipRectDrawData {
    pub(crate) fn new(
        rect_bounds: [(f32, f32); 2],
        transform: Option<InstanceTransform>,
        clips_children: bool,
    ) -> Self {
        Self {
            rect_bounds,
            transform,
            clips_children,
            is_leaf: true,
        }
    }
}

impl DrawTreeNode {
    /// Whether this node has no children in the draw tree
    /// Starts as `true`; set to `false` when a child is added.
    pub(crate) fn is_leaf(&self) -> bool {
        match self {
            DrawTreeNode::CachedShape(s) => s.is_leaf,
            DrawTreeNode::ClipRect(clip_rect) => clip_rect.is_leaf,
        }
    }

    pub(crate) fn set_not_leaf(&mut self) {
        match self {
            DrawTreeNode::CachedShape(s) => s.is_leaf = false,
            DrawTreeNode::ClipRect(clip_rect) => clip_rect.is_leaf = false,
        }
    }
}

impl DrawTreeNode {
    pub(crate) fn transform(&self) -> Option<InstanceTransform> {
        match self {
            DrawTreeNode::CachedShape(cached_shape) => cached_shape.instance.transform,
            DrawTreeNode::ClipRect(clip_rect) => clip_rect.transform,
        }
    }

    pub(crate) fn texture_id(&self, layer: usize) -> Option<u64> {
        match self {
            DrawTreeNode::CachedShape(cached_shape) => cached_shape
                .instance
                .textures
                .get(layer)
                .and_then(|texture| texture.texture_id),
            DrawTreeNode::ClipRect(_) => None,
        }
    }

    pub(crate) fn local_bounds(&self) -> [(f32, f32); 2] {
        match self {
            DrawTreeNode::CachedShape(cached_shape) => {
                cached_shape.instance.cached_shape.local_bounds()
            }
            DrawTreeNode::ClipRect(clip_rect) => clip_rect.rect_bounds,
        }
    }

    pub(crate) fn instance_color_override(&self) -> Option<[f32; 4]> {
        match self {
            DrawTreeNode::CachedShape(cached_shape) => cached_shape.instance.color_override,
            DrawTreeNode::ClipRect(_) => None,
        }
    }

    pub(crate) fn has_gradient_fill(&self) -> bool {
        match self {
            DrawTreeNode::CachedShape(cached_shape) => cached_shape.instance.has_gradient_fill(),
            DrawTreeNode::ClipRect(_) => false,
        }
    }

    pub(crate) fn clips_children(&self) -> bool {
        match self {
            DrawTreeNode::CachedShape(cached_shape) => cached_shape.clips_children,
            DrawTreeNode::ClipRect(clip_rect) => clip_rect.clips_children,
        }
    }

    pub(crate) fn is_rect(&self) -> bool {
        match self {
            DrawTreeNode::CachedShape(cached_shape) => cached_shape.instance.cached_shape.is_rect,
            DrawTreeNode::ClipRect(_) => true,
        }
    }

    pub(crate) fn rect_bounds(&self) -> Option<[(f32, f32); 2]> {
        match self {
            DrawTreeNode::CachedShape(cached_shape) => {
                cached_shape.instance.cached_shape.rect_bounds
            }
            DrawTreeNode::ClipRect(clip_rect) => Some(clip_rect.rect_bounds),
        }
    }
}
