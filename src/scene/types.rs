use crate::core::shape::{CachedShapeHandle, ShapeDrawCommandOptions, ShapeInstance};
use crate::core::vertex::InstanceTransform;
use crate::core::{geometry, MathRect};

#[derive(Debug)]
pub(crate) struct CachedShapeDrawData {
    pub(crate) instance: ShapeInstance,
    pub(crate) clips_children: bool,
    pub(crate) logical_screen_bounds: MathRect,
}

impl CachedShapeDrawData {
    pub(crate) fn new(cached_shape: CachedShapeHandle, options: ShapeDrawCommandOptions) -> Self {
        let clips_children = options.clips_children;
        let instance = ShapeInstance::new(cached_shape, options);
        let bounds = instance.cached_shape.local_bounds();
        let logical_screen_bounds = geometry::transformed_bounds_to_logical_screen_rect(
            MathRect::new(bounds[0].into(), bounds[1].into()),
            instance.transform,
        );
        Self {
            instance,
            clips_children,
            logical_screen_bounds,
        }
    }

    pub(crate) fn has_geometry(&self) -> bool {
        let geometry = self.instance.cached_shape.vertex_buffers();
        !geometry.vertices.is_empty() && !geometry.indices.is_empty()
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
    pub(crate) clips_children: bool,
    pub(crate) logical_screen_bounds: MathRect,
}

impl ClipRectDrawData {
    pub(crate) fn new(
        rect_bounds: [(f32, f32); 2],
        transform: Option<InstanceTransform>,
        clips_children: bool,
    ) -> Self {
        Self {
            clips_children,
            logical_screen_bounds: geometry::transformed_bounds_to_logical_screen_rect(
                MathRect::new(rect_bounds[0].into(), rect_bounds[1].into()),
                transform,
            ),
        }
    }
}

impl DrawTreeNode {
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

    /// Unpadded bounds cached for the node's fixed geometry and transform.
    pub(crate) fn logical_screen_bounds(&self) -> MathRect {
        match self {
            DrawTreeNode::CachedShape(cached_shape) => cached_shape.logical_screen_bounds,
            DrawTreeNode::ClipRect(clip_rect) => clip_rect.logical_screen_bounds,
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

    pub(crate) fn is_axis_aligned_rect(&self) -> bool {
        match self {
            DrawTreeNode::CachedShape(cached_shape) => {
                cached_shape.instance.cached_shape.is_rect
                    && geometry::is_axis_aligned_rect_transform(cached_shape.instance.transform)
            }
            DrawTreeNode::ClipRect(_) => true,
        }
    }
}
