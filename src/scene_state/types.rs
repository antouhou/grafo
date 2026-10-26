use super::effects::EffectAttachments;
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GroupEffectSources {
    pub(crate) content: Option<usize>,
    pub(crate) backdrop: Option<usize>,
    pub(crate) shape_mask: Option<usize>,
}

impl GroupEffectSources {
    pub(crate) fn iter(self) -> impl Iterator<Item = usize> {
        self.content
            .into_iter()
            .chain(self.backdrop)
            .chain(self.shape_mask)
    }
}

#[derive(Debug, Default)]
pub(crate) struct GroupDrawData {
    pub(crate) effects: EffectAttachments,
    pub(crate) effect_sources: GroupEffectSources,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub(crate) enum DrawTreeNode {
    Group(GroupDrawData),
    CachedShape(CachedShapeDrawData),
    ClipRect(ClipRectDrawData),
}

#[derive(Debug)]
pub(crate) struct ClipRectDrawData {
    pub(crate) clips_children: bool,
    pub(crate) logical_screen_bounds: MathRect,
    pub(crate) effect_sources: GroupEffectSources,
}

impl ClipRectDrawData {
    pub(crate) fn new(
        rect_bounds: [(f32, f32); 2],
        transform: Option<InstanceTransform>,
        clips_children: bool,
    ) -> Self {
        Self {
            clips_children,
            effect_sources: GroupEffectSources::default(),
            logical_screen_bounds: geometry::transformed_bounds_to_logical_screen_rect(
                MathRect::new(rect_bounds[0].into(), rect_bounds[1].into()),
                transform,
            ),
        }
    }
}

impl DrawTreeNode {
    pub(crate) fn group_effects(&self) -> EffectAttachments {
        match self {
            Self::Group(group) => group.effects,
            Self::CachedShape(_) | Self::ClipRect(_) => EffectAttachments::default(),
        }
    }

    pub(crate) fn group_effect_sources(&self) -> GroupEffectSources {
        match self {
            Self::Group(group) => group.effect_sources,
            Self::ClipRect(clip) => clip.effect_sources,
            Self::CachedShape(_) => GroupEffectSources::default(),
        }
    }

    /// Unpadded bounds of the node itself, empty for structural groups.
    pub(crate) fn logical_screen_bounds(&self) -> MathRect {
        match self {
            DrawTreeNode::Group(_) => MathRect::zero(),
            DrawTreeNode::CachedShape(cached_shape) => cached_shape.logical_screen_bounds,
            DrawTreeNode::ClipRect(clip_rect) => clip_rect.logical_screen_bounds,
        }
    }

    pub(crate) fn instance_color_override(&self) -> Option<[f32; 4]> {
        match self {
            DrawTreeNode::CachedShape(cached_shape) => cached_shape.instance.color_override,
            DrawTreeNode::Group(_) | DrawTreeNode::ClipRect(_) => None,
        }
    }

    pub(crate) fn clips_children(&self) -> bool {
        match self {
            DrawTreeNode::Group(_) => false,
            DrawTreeNode::CachedShape(cached_shape) => cached_shape.clips_children,
            DrawTreeNode::ClipRect(clip_rect) => clip_rect.clips_children,
        }
    }

    pub(crate) fn is_axis_aligned_rect(&self) -> bool {
        match self {
            DrawTreeNode::Group(_) => false,
            DrawTreeNode::CachedShape(cached_shape) => {
                cached_shape.instance.cached_shape.tessellation.is_rect
                    && geometry::is_axis_aligned_rect_transform(cached_shape.instance.transform)
            }
            DrawTreeNode::ClipRect(_) => true,
        }
    }
}
