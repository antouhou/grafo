use super::{
    draw_commands::{has_child_clip, DrawCommand, NodeTarget},
    DrawCommandError, Scene,
};
use crate::core::{
    geometry, GlyphContent, MathRect, PathShape, Shape, ShapeDrawCommandOptions, Size, TextSource,
    TransformInstance,
};
use crate::render_backend::RenderBackend;
use crate::scene_state::SceneError;
use lyon::{math::Transform, path::Path};

pub(crate) fn build_text_shape(source: &mut impl TextSource) -> Shape {
    let mut builder = Path::builder();
    while let Some(glyph) = source.next_glyph() {
        let Some(GlyphContent::Outline(events)) = source.glyph_content(&glyph.key) else {
            continue;
        };
        let offset = Transform::translation(glyph.position.0, glyph.position.1);
        for event in events {
            builder.path_event(event.transformed(&offset));
        }
    }
    Shape::Path(PathShape::new(builder.build()))
}

impl<B: RenderBackend> Scene<B> {
    fn text_visible_bounds(
        &self,
        target: &NodeTarget,
        transform: Option<TransformInstance>,
    ) -> Option<MathRect> {
        let viewport = self.state.viewport();
        let scale = viewport.scale_factor as f32;
        if !scale.is_finite() || scale <= 0.0 {
            return None;
        }
        let mut bounds = MathRect::from_size(Size::from(viewport.physical_size).to_f32()) / scale;
        let replaced = match *target {
            NodeTarget::Insert { .. } => None,
            NodeTarget::Replace { node_id } => Some(node_id),
        };
        let mut ancestor = match *target {
            NodeTarget::Insert { parent } => parent,
            NodeTarget::Replace { node_id } => Some(node_id),
        };
        // TODO: cache that, this is not good
        while let Some(node_id) = ancestor {
            let node = self.state.draw_tree.get(node_id)?;
            if Some(node_id) != replaced
                && has_child_clip(node)
                && node.logical_screen_bounds().is_finite()
            {
                bounds = bounds
                    .intersection(&node.logical_screen_bounds())
                    .unwrap_or_else(MathRect::zero);
            }
            ancestor = self.state.draw_tree.parent_index_unchecked(node_id);
        }
        if bounds.is_empty() {
            return Some(MathRect::zero());
        }
        // Keep geometry contributing antialiasing coverage along the boundary.
        let guard = (1.0 / viewport.scale_factor) as f32;
        geometry::logical_screen_rect_to_local_bounds(bounds.inflate(guard, guard), transform)
    }

    pub(super) fn queue_text(
        &mut self,
        target: NodeTarget,
        mut source: impl TextSource,
        options: ShapeDrawCommandOptions,
    ) -> Result<usize, DrawCommandError<B::Error>> {
        match target {
            NodeTarget::Insert { parent } => self.state.validate_parent(parent)?,
            NodeTarget::Replace { node_id } => {
                self.state
                    .draw_tree
                    .get(node_id)
                    .ok_or(SceneError::NodeNotFound(node_id))?;
            }
        }
        let bounds = self.text_visible_bounds(&target, options.transform);
        source.set_visible_bounds(bounds);
        let shape = build_text_shape(&mut source);
        self.queue_draw_command(
            target,
            DrawCommand::Shape {
                shape: &shape,
                geometry_id: None,
                options,
            },
        )
    }
}

#[cfg(all(test, feature = "cosmic-text"))]
mod tests;
