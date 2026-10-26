//! Public scene construction, configuration, and rendering API.
use crate::commands::ShapeDrawId;
use crate::core::effect::{
    BackdropEffectConfig, ContentEffectConfig, ShapeMaskEffectBounds, ShapeMaskEffectConfig,
};
use crate::core::shape::{Shape, ShapeDrawCommandOptions};
use crate::core::vertex::InstanceTransform;
use crate::core::{TextSource, UnsignedPhysicalRect};
use crate::planner::Planner;
use crate::render_backend::render_target::RenderTarget;
use crate::render_backend::RenderBackend;
use crate::scene_state::effects::{
    EffectAttachment, EffectAttachmentKind, EffectAttachments, EffectConfig,
};
use crate::scene_state::types::DrawTreeNode;
use crate::scene_state::{SceneError, SceneState, SceneStateContext};
use damage::{mark_dirty, mark_physical_dirty, mark_shape_mask_effect_dirty, PendingDamage};
use draw_commands::{DrawCommand, NodeTarget};
use groups::GroupEffectScratch;
#[cfg(feature = "render_metrics")]
use metrics::RenderLoopMetricsTracker;
use std::mem;
#[cfg(feature = "render_metrics")]
use std::time::Instant;
use text::TextScratch;
use thiserror::Error;

mod damage;
mod draw_commands;
mod groups;
#[cfg(feature = "render_metrics")]
mod metrics;
mod text;

/// Scene mutation or backend resource preparation failed.
#[derive(Debug, Error)]
pub enum DrawCommandError<E> {
    #[error(transparent)]
    Scene(#[from] SceneError),
    #[error("Backend upload failed: {0}")]
    Backend(#[source] E),
}

/// Scene attachment or backend effect validation failed.
#[derive(Debug, Error)]
pub enum EffectError<E> {
    #[error(transparent)]
    Scene(#[from] SceneError),
    #[error("Backend effect failed: {0}")]
    Backend(#[source] E),
}

/// CPU shape storage and backend context shared between scenes.
#[derive(Clone)]
pub struct SceneContext<B> {
    pub(crate) backend: B,
    pub(crate) state: SceneStateContext,
}

/// A retained scene of groups, shapes, clipping rectangles, and effects.
pub struct Scene<B: RenderBackend> {
    state: SceneState,
    planner: Planner,
    backend: B,
    removed_shape_ids: Vec<ShapeDrawId>,
    text_scratch: TextScratch,
    group_effect_scratch: GroupEffectScratch,
    dirty_bounds: Option<UnsignedPhysicalRect>,
    pending_damage: PendingDamage,
    #[cfg(feature = "render_metrics")]
    render_loop_metrics_tracker: RenderLoopMetricsTracker,
}

impl<B: RenderBackend> Scene<B> {
    /// Creates an empty scene using the supplied backend.
    /// Loaded shapes are shared through `context`.
    pub(crate) fn from_parts(backend: B, context: SceneStateContext) -> Self {
        let viewport = backend.viewport();
        Self {
            state: SceneState::new(context, viewport),
            planner: Planner::default(),
            removed_shape_ids: Vec::new(),
            text_scratch: TextScratch::default(),
            group_effect_scratch: GroupEffectScratch::default(),
            pending_damage: PendingDamage::default(),
            dirty_bounds: Some(UnsignedPhysicalRect::from_size(
                viewport.physical_size.into(),
            )),
            backend,
            #[cfg(feature = "render_metrics")]
            render_loop_metrics_tracker: RenderLoopMetricsTracker::default(),
        }
    }

    /// Provides read-only access to backend-specific resources and diagnostics.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn texture_manager(&self) -> &B::TextureManager {
        self.backend.texture_manager()
    }

    pub fn size(&self) -> (u32, u32) {
        self.state.viewport().physical_size
    }

    pub fn scale_factor(&self) -> f64 {
        self.state.viewport().scale_factor
    }

    /// Returns an error and preserves settings if attached shape mask effects cannot use the new scale.
    pub fn change_scale_factor(&mut self, scale_factor: f64) -> Result<(), SceneError> {
        self.state
            .change_scale_factor(scale_factor, self.backend.maximum_texture_dimension())?;
        self.update_backend_viewport();
        Ok(())
    }

    pub fn resize(&mut self, new_physical_size: (u32, u32)) {
        self.state
            .resize(new_physical_size, self.backend.maximum_texture_dimension());
        self.update_backend_viewport();
    }

    pub fn set_msaa_samples(&mut self, samples: u32) {
        self.backend.set_msaa_samples(samples);
        self.dirty_bounds = Some(UnsignedPhysicalRect::from_size(
            self.state.viewport().physical_size.into(),
        ));
    }

    /// Tessellates into the shared CPU cache. Geometry IDs let identical shapes share uploads.
    pub fn load_shape(
        &mut self,
        shape: impl AsRef<Shape>,
        cache_key: u64,
        geometry_id: Option<u64>,
    ) {
        self.state.load_shape(shape, cache_key, geometry_id);
    }

    pub fn remove_shape(&mut self, cache_key: u64) {
        self.state.remove_shape(cache_key);
    }

    /// Adds a structural group. Doesn't draw or clip anything by itself, but can carry effects.
    /// Use [`Scene::set_backdrop_effect`]/[`Scene::set_shape_mask_effect`]/[`Scene::set_content_effect`]
    /// The effects will be applied to a whole group; Individual group members can override the
    /// effect set that way by calling the same methods on individual nodes.
    pub fn add_group(&mut self, parent_node_id: Option<usize>) -> Result<usize, SceneError> {
        self.state.validate_parent(parent_node_id)?;
        Ok(self
            .state
            .insert_node(DrawTreeNode::Group(Default::default()), parent_node_id))
    }

    /// Queues a shape without retaining it in the loaded-shape cache.
    /// Children inherit clipping unless their parent uses `clips_children(false)`.
    pub fn add_shape(
        &mut self,
        shape: impl AsRef<Shape>,
        parent_shape_id: Option<usize>,
        geometry_id: Option<u64>,
        options: ShapeDrawCommandOptions,
    ) -> Result<usize, DrawCommandError<B::Error>> {
        self.queue_draw_command(
            NodeTarget::Insert {
                parent: parent_shape_id,
            },
            DrawCommand::Shape {
                shape: shape.as_ref(),
                geometry_id,
                options,
            },
        )
    }

    /// Adds a text from a source to the scene. Text node can do everything a shape or a rect can -
    /// you apply to it a gradient fill, backdrop filters, mask effects, etc. This method supplies
    /// the current visible bounds to the text source. If you want to change the size of the parent
    /// node, call `Scene::replace_with_text` to update the bounds.
    pub fn add_text(
        &mut self,
        source: impl TextSource,
        parent_shape_id: Option<usize>,
        options: ShapeDrawCommandOptions,
    ) -> Result<usize, DrawCommandError<B::Error>> {
        self.queue_text(
            NodeTarget::Insert {
                parent: parent_shape_id,
            },
            source,
            options,
        )
    }

    /// Queues a loaded shape and prepares backend resources for this instance.
    pub fn add_cached_shape(
        &mut self,
        cache_key: u64,
        parent_shape_id: Option<usize>,
        options: ShapeDrawCommandOptions,
    ) -> Result<usize, DrawCommandError<B::Error>> {
        self.queue_draw_command(
            NodeTarget::Insert {
                parent: parent_shape_id,
            },
            DrawCommand::CachedShape { cache_key, options },
        )
    }

    /// Adds a scissor clip without geometry. Rotation, skew and perspective are rejected.
    pub fn add_clipping_rect(
        &mut self,
        rect_bounds: [(f32, f32); 2],
        parent_shape_id: Option<usize>,
        transform: Option<impl Into<InstanceTransform>>,
        clips_children: bool,
    ) -> Result<usize, DrawCommandError<B::Error>> {
        self.queue_draw_command(
            NodeTarget::Insert {
                parent: parent_shape_id,
            },
            DrawCommand::ClippingRect {
                rect_bounds,
                transform: transform.map(Into::into),
                clips_children,
            },
        )
    }

    /// Replaces one command with a shape and returns its unchanged ID.
    /// Parent, children and attached effects are preserved.
    pub fn replace_with_shape(
        &mut self,
        node_id: usize,
        shape: impl AsRef<Shape>,
        geometry_id: Option<u64>,
        options: ShapeDrawCommandOptions,
    ) -> Result<usize, DrawCommandError<B::Error>> {
        self.queue_draw_command(
            NodeTarget::Replace { node_id },
            DrawCommand::Shape {
                shape: shape.as_ref(),
                geometry_id,
                options,
            },
        )
    }

    /// Replaces a node with text, preserving its ID, parent, children, and effects. If the node is
    /// a text node already, updates the text and visible bounds. Supplies the current visible
    /// bounds. Call after changing the source or after resizing a parent node.
    /// Resize the scene first when the target dimensions change.
    pub fn replace_with_text(
        &mut self,
        node_id: usize,
        source: impl TextSource,
        options: ShapeDrawCommandOptions,
    ) -> Result<usize, DrawCommandError<B::Error>> {
        self.queue_text(NodeTarget::Replace { node_id }, source, options)
    }

    /// Replaces one command with a loaded shape and returns its unchanged ID.
    /// Children and attached effects are retained.
    pub fn replace_with_cached_shape(
        &mut self,
        node_id: usize,
        cache_key: u64,
        options: ShapeDrawCommandOptions,
    ) -> Result<usize, DrawCommandError<B::Error>> {
        self.queue_draw_command(
            NodeTarget::Replace { node_id },
            DrawCommand::CachedShape { cache_key, options },
        )
    }

    /// Replaces one command with a clip rectangle and returns its unchanged ID.
    /// Children are retained.
    /// Effects attached to this node are removed because they require a shape.
    pub fn replace_with_clipping_rect(
        &mut self,
        node_id: usize,
        rect_bounds: [(f32, f32); 2],
        transform: Option<impl Into<InstanceTransform>>,
        clips_children: bool,
    ) -> Result<usize, DrawCommandError<B::Error>> {
        self.queue_draw_command(
            NodeTarget::Replace { node_id },
            DrawCommand::ClippingRect {
                rect_bounds,
                transform: transform.map(Into::into),
                clips_children,
            },
        )
    }

    /// Removes queued nodes and their descendants.
    ///
    /// Calls `removed` once per removed ID.
    /// Remaining nodes keep their IDs and effects. New nodes may reuse removed IDs.
    /// Loaded shapes and textures remain available.
    pub fn remove_subtrees(
        &mut self,
        node_ids: impl IntoIterator<Item = usize>,
        mut removed: impl FnMut(usize),
    ) {
        self.removed_shape_ids.clear();
        let mut has_removed_nodes = false;
        let viewport = self.state.viewport();
        self.state
            .remove_subtrees_with(node_ids, |id, node, effect, group_output| {
                if matches!(node, DrawTreeNode::CachedShape(_)) {
                    mark_dirty(
                        &mut self.dirty_bounds,
                        node.logical_screen_bounds(),
                        viewport,
                    );
                }
                if let Some(effect) = effect {
                    mark_shape_mask_effect_dirty(&mut self.dirty_bounds, effect.bounds, viewport);
                }
                mark_physical_dirty(&mut self.dirty_bounds, group_output);
                has_removed_nodes = true;
                if matches!(node, DrawTreeNode::CachedShape(_)) {
                    self.removed_shape_ids.push(ShapeDrawId(id));
                }
                removed(id);
            });
        if !has_removed_nodes {
            return;
        }
        if self.state.draw_tree.is_empty() {
            self.planner.clear();
            self.backend.clear();
        } else if !self.removed_shape_ids.is_empty() {
            self.backend.unregister_shapes(&self.removed_shape_ids);
        }
        self.removed_shape_ids.clear();
    }

    /// Removes all groups, shapes, clipping rectangles, and attached effects from the scene.
    /// Loaded shapes, textures, and effect definitions remain available.
    pub fn clear(&mut self) {
        self.remove_subtrees([0], |_| {});
    }

    /// Loads shader passes. Changed sources detach old instances and invalidate cached results.
    pub fn load_effect(
        &mut self,
        effect_id: u64,
        pass_sources: &[&str],
    ) -> Result<(), EffectError<B::Error>> {
        if self
            .backend
            .load_effect(effect_id, pass_sources)
            .map_err(EffectError::Backend)?
        {
            self.remove_effect_attachments(effect_id);
        }
        Ok(())
    }

    pub fn unload_effect(&mut self, effect_id: u64) {
        self.remove_effect_attachments(effect_id);
        self.backend.unload_effect(effect_id);
    }

    /// Sets an effect applied to the node and its descendants.
    ///
    /// Captures all descendants rendered within the shape's bounds and uses that as an input for
    /// the effect. Overflowing children are not included into the capture. The output is drawn in
    /// a place of the shape. You can reserve additional space around the shape for the output via
    /// the effect config, if the output needs to expand original shape.
    pub fn set_content_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        params: &[u8],
        config: ContentEffectConfig,
    ) -> Result<(), EffectError<B::Error>> {
        if matches!(
            self.state.draw_tree.get(node_id),
            Some(DrawTreeNode::Group(_))
        ) {
            return self.set_group_effect(
                node_id,
                effect_id,
                params,
                EffectConfig::Content(config),
            );
        }
        self.state.shape(node_id)?;
        self.backend
            .validate_effect_params(effect_id, params)
            .map_err(EffectError::Backend)?;
        let previous_output = self.state.content_effect_output(node_id);
        self.state.set_content_effect(
            node_id,
            effect_id,
            params,
            config,
            self.backend.maximum_texture_dimension(),
        )?;
        mark_physical_dirty(&mut self.dirty_bounds, previous_output);
        mark_physical_dirty(
            &mut self.dirty_bounds,
            self.state.content_effect_output(node_id),
        );
        self.mark_shape_dirty(node_id);
        Ok(())
    }

    /// Removes the content effect, restoring the node's ordinary rendering.
    pub fn remove_content_effect(&mut self, node_id: usize) {
        if matches!(
            self.state.draw_tree.get(node_id),
            Some(DrawTreeNode::Group(_))
        ) {
            self.remove_group_effect(node_id, EffectAttachmentKind::Content);
            return;
        }
        let previous_output = self.state.content_effect_output(node_id);
        if self.state.remove_content_effect(node_id) {
            mark_physical_dirty(&mut self.dirty_bounds, previous_output);
            self.mark_shape_dirty(node_id);
        }
    }

    /// A backdrop filter: Filters previously rendered scene pixels behind the node. Perfect fit
    /// for frosted glass background and similar effects.
    ///
    /// Effect config can be used to control capture area and padding around the input, independently
    /// of the node's own config
    pub fn set_backdrop_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        params: &[u8],
        config: BackdropEffectConfig,
    ) -> Result<(), EffectError<B::Error>> {
        if matches!(
            self.state.draw_tree.get(node_id),
            Some(DrawTreeNode::Group(_))
        ) {
            return self.set_group_effect(
                node_id,
                effect_id,
                params,
                EffectConfig::Backdrop(config),
            );
        }
        self.state.shape(node_id)?;
        self.backend
            .validate_effect_params(effect_id, params)
            .map_err(EffectError::Backend)?;
        self.state.set_backdrop_effect(
            node_id,
            effect_id,
            params,
            config,
            self.backend.maximum_texture_dimension(),
        )?;
        // Damage expansion performed before planning will include the capture region
        self.mark_shape_dirty(node_id);
        Ok(())
    }

    /// Removes the node's backdrop effect.
    pub fn remove_backdrop_effect(&mut self, node_id: usize) {
        if matches!(
            self.state.draw_tree.get(node_id),
            Some(DrawTreeNode::Group(_))
        ) {
            self.remove_group_effect(node_id, EffectAttachmentKind::Backdrop);
            return;
        }
        if self.state.remove_backdrop_effect(node_id) {
            self.mark_shape_dirty(node_id);
        }
        self.backend.remove_backdrop_effect(ShapeDrawId(node_id));
    }

    /// Applies an effect to the shape's white geometric coverage alpha mask. Perfect for things
    /// like box shadows.
    ///
    /// The input is an alpha mask of the shape it is attached to. It excludes the shape's fill,
    /// textures, and children. The result is drawn behind the shape.
    pub fn set_shape_mask_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        params: &[u8],
        config: ShapeMaskEffectConfig,
    ) -> Result<(), EffectError<B::Error>> {
        if matches!(
            self.state.draw_tree.get(node_id),
            Some(DrawTreeNode::Group(_))
        ) {
            return self.set_group_effect(
                node_id,
                effect_id,
                params,
                EffectConfig::ShapeMask(config),
            );
        }
        self.state.shape(node_id)?;
        self.backend
            .validate_effect_params(effect_id, params)
            .map_err(EffectError::Backend)?;
        let old_bounds = self
            .state
            .shape_mask_effects
            .get(&node_id)
            .map(|effect| effect.bounds);
        let previous_content_output = self.state.content_effect_output(node_id);
        self.state.set_shape_mask_effect(
            node_id,
            effect_id,
            params,
            config,
            self.backend.maximum_texture_dimension(),
        )?;
        self.backend.set_shape_mask_effect_geometry(
            ShapeDrawId(node_id),
            &self.state.shape(node_id)?.cached_shape,
        );
        if let Some(bounds) = old_bounds {
            self.mark_shape_mask_effect_dirty(bounds);
        }
        self.mark_shape_mask_effect_dirty(self.state.shape_mask_effect(node_id)?.bounds);
        mark_physical_dirty(&mut self.dirty_bounds, previous_content_output);
        mark_physical_dirty(
            &mut self.dirty_bounds,
            self.state.content_effect_output(node_id),
        );
        Ok(())
    }

    /// Removes the shape's mask effect while keeping its fill and children.
    pub fn remove_shape_mask_effect(&mut self, node_id: usize) {
        if matches!(
            self.state.draw_tree.get(node_id),
            Some(DrawTreeNode::Group(_))
        ) {
            self.remove_group_effect(node_id, EffectAttachmentKind::ShapeMask);
            return;
        }
        let previous_content_output = self.state.content_effect_output(node_id);
        let bounds = self
            .state
            .shape_mask_effects
            .get(&node_id)
            .map(|effect| effect.bounds);
        self.state
            .remove_shape_mask_effect(node_id, self.backend.maximum_texture_dimension());
        if let Some(bounds) = bounds {
            self.mark_shape_mask_effect_dirty(bounds);
        }
        self.backend.remove_shape_mask_effect(ShapeDrawId(node_id));
        if bounds.is_some() {
            mark_physical_dirty(&mut self.dirty_bounds, previous_content_output);
            mark_physical_dirty(
                &mut self.dirty_bounds,
                self.state.content_effect_output(node_id),
            );
        }
    }

    /// Enables a red overlay at 25% opacity over the current surface redraw bounds.
    /// Disabled by default. Pixmap output and the retained scene stay unchanged.
    /// Toggling the overlay does not request a scene redraw.
    pub fn set_dirty_region_overlay_enabled(&mut self, enabled: bool) {
        self.backend.set_dirty_region_overlay_enabled(enabled);
    }

    /// Returns whether surface redraw bounds are highlighted.
    pub fn is_dirty_region_overlay_enabled(&self) -> bool {
        self.backend.is_dirty_region_overlay_enabled()
    }

    /// Renders the scene to `target`, resizing it to match the target's physical size.
    ///
    /// Pixmap pixels are ready on success and unchanged on error. Surface rendering
    /// submits and presents the frame, though GPU work may still be pending.
    /// Retains the scene and output image for later renders. Changes to shapes, effects, or
    /// textures redraw only the affected areas. An unchanged scene preserves the image.
    pub fn render<'target>(
        &mut self,
        target: impl Into<RenderTarget<'target, B::Surface>>,
    ) -> Result<(), B::Error>
    where
        B::Surface: 'target,
    {
        let target = target.into();
        let maximum = self.backend.maximum_texture_dimension();
        let size = target.validate_size(maximum)?;
        if self.size() != size {
            self.resize(size);
        }
        #[cfg(feature = "render_metrics")]
        let started_at = Instant::now();
        self.state.compact_effect_parameters();
        self.pending_damage
            .apply(&self.state, &mut self.dirty_bounds);
        let viewport = self.state.viewport();
        for (id, shape) in self.state.shapes_with_updated_textures() {
            self.backend.refresh_uv(ShapeDrawId(id), &shape.instance);
            mark_dirty(
                &mut self.dirty_bounds,
                shape.logical_screen_bounds,
                viewport,
            );
        }
        self.dirty_bounds = self.state.expand_effect_damage(self.dirty_bounds);
        self.state.finish_preparation();
        let commands = self
            .planner
            .plan(&self.state, self.backend.maximum_texture_dimension());
        self.backend.render(self.dirty_bounds, commands, target)?;
        self.dirty_bounds = None;
        #[cfg(feature = "render_metrics")]
        self.render_loop_metrics_tracker
            .record_presented_frame(started_at, Instant::now());
        Ok(())
    }
}

impl<B: RenderBackend> Scene<B> {
    fn update_backend_viewport(&mut self) {
        let viewport = self.state.viewport();
        self.dirty_bounds = Some(UnsignedPhysicalRect::from_size(
            viewport.physical_size.into(),
        ));
        self.backend.resize(viewport);
    }

    fn set_effect(
        &mut self,
        node_id: usize,
        effect_id: u64,
        params: &[u8],
        config: EffectConfig,
    ) -> Result<(), EffectError<B::Error>> {
        match config {
            EffectConfig::Content(config) => {
                self.set_content_effect(node_id, effect_id, params, config)
            }
            EffectConfig::Backdrop(config) => {
                self.set_backdrop_effect(node_id, effect_id, params, config)
            }
            EffectConfig::ShapeMask(config) => {
                self.set_shape_mask_effect(node_id, effect_id, params, config)
            }
        }
    }

    fn apply_stored_effect(
        &mut self,
        node_id: usize,
        attachment: EffectAttachment,
        parameters: &mut Vec<u8>,
    ) -> Result<(), EffectError<B::Error>> {
        parameters.clear();
        parameters.extend_from_slice(
            self.state
                .effect_parameters
                .get(attachment.instance.parameters)
                .bytes,
        );
        self.set_effect(
            node_id,
            attachment.instance.effect_id,
            parameters,
            attachment.config,
        )
    }

    fn apply_effects(
        &mut self,
        node_id: usize,
        effects: EffectAttachments,
    ) -> Result<(), DrawCommandError<B::Error>> {
        if effects.iter().next().is_none() {
            return Ok(());
        }
        let mut scratch = mem::take(&mut self.group_effect_scratch);
        let result = effects.iter().copied().try_for_each(|effect| {
            self.apply_stored_effect(node_id, effect, &mut scratch.parameters)
        });
        scratch.parameters.clear();
        self.group_effect_scratch = scratch;
        result.map_err(|error| match error {
            EffectError::Scene(error) => DrawCommandError::Scene(error),
            EffectError::Backend(error) => DrawCommandError::Backend(error),
        })
    }

    fn remove_effect_attachments(&mut self, effect_id: u64) {
        self.state.remove_effect_attachments(
            effect_id,
            self.backend.maximum_texture_dimension(),
            |attachment| {
                self.group_effect_scratch
                    .removed_effects
                    .push((attachment.node_id, attachment.kind));
                mark_physical_dirty(&mut self.dirty_bounds, attachment.damage_bounds);
                match attachment.kind {
                    EffectAttachmentKind::Content => {}
                    EffectAttachmentKind::Backdrop => {
                        self.backend
                            .remove_backdrop_effect(ShapeDrawId(attachment.node_id));
                    }
                    EffectAttachmentKind::ShapeMask => {
                        self.backend
                            .remove_shape_mask_effect(ShapeDrawId(attachment.node_id));
                    }
                }
            },
        );
        self.restore_removed_group_effects();
        self.backend.invalidate_effect(effect_id);
    }

    fn mark_shape_dirty(&mut self, node_id: usize) {
        if let Some(DrawTreeNode::CachedShape(shape)) = self.state.draw_tree.get(node_id) {
            mark_dirty(
                &mut self.dirty_bounds,
                shape.logical_screen_bounds,
                self.state.viewport(),
            );
        }
    }

    fn mark_shape_mask_effect_dirty(&mut self, bounds: ShapeMaskEffectBounds) {
        mark_shape_mask_effect_dirty(&mut self.dirty_bounds, bounds, self.state.viewport());
    }
}

#[cfg(test)]
mod tests;
