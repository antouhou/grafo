use super::*;
use crate::renderer::traversal::subtree_has_backdrop_effects;
use crate::vertex::InstanceOcclusion;

impl<'a> Renderer<'a> {
    /// Computes per-instance occlusion rects for all non-leaf parents with
    /// opaque axis-aligned children. Must be called after `prepare_render()`
    /// and before `render_segments()`.
    ///
    /// Populates `self.temp_instance_occlusions` with (offset, count) pairs and
    /// `self.temp_occlusion_rects` with the packed rect data. Uploads the storage
    /// buffer and creates the bind group for `@group(3)`.
    pub(super) fn compute_occlusion_rects(&mut self) {
        debug_assert!(
            self.aggregated_vertex_buffer.is_some(),
            "compute_occlusion_rects() called before prepare_render()"
        );

        self.temp_occlusion_rects.clear();
        self.temp_parent_node_ids.clear();

        // Collect non-leaf parent node IDs into a reusable scratch buffer
        // (we cannot mutate temp_instance_occlusions while iterating the tree).
        for (id, cmd) in self.draw_tree.iter() {
            if !cmd.is_leaf() {
                self.temp_parent_node_ids.push(id);
            }
        }

        for i in 0..self.temp_parent_node_ids.len() {
            let parent_id = self.temp_parent_node_ids[i];
            let parent_cmd = match self.draw_tree.get(parent_id) {
                Some(cmd) => cmd,
                None => continue,
            };

            // Skip empty parents or those without an instance index.
            if parent_cmd.is_empty() {
                continue;
            }
            let parent_instance_idx = match parent_cmd.instance_index() {
                Some(idx) => idx,
                None => continue,
            };

            // Rule 6 (clip-scope): Only compute occlusion rects for parents that
            // clip their children. A non-clipping parent's children inherit the
            // clip region from an ancestor — their visible area may be smaller than
            // their projected bounds, so using those bounds would discard parent
            // pixels the child never actually covers.
            if !parent_cmd.clips_children() {
                continue;
            }

            // Rule 7: Skip if this node is a group-effect node.
            if self.group_effects.contains_key(&parent_id) {
                continue;
            }

            // Rule 7: Skip if any ancestor is a group-effect node.
            if has_group_effect_ancestor(&self.draw_tree, &self.group_effects, parent_id) {
                continue;
            }

            // Rule 8: Skip if any descendant has a backdrop effect.
            if subtree_has_backdrop_effects(&self.draw_tree, &self.backdrop_effects, parent_id) {
                continue;
            }

            // Compute parent's screen-space bounds for optional clamping (rect parents only).
            let parent_screen_bounds = parent_cmd.rect_bounds().and_then(|rb| {
                transform_rect_to_screen(rb, parent_cmd.transform(), self.scale_factor)
            });

            let rect_offset = self.temp_occlusion_rects.len();

            // Iterate direct children without allocating (children() returns &[usize]).
            let num_children = self.draw_tree.children(parent_id).len();
            for ci in 0..num_children {
                let child_id = self.draw_tree.children(parent_id)[ci];
                let child_cmd = match self.draw_tree.get(child_id) {
                    Some(cmd) => cmd,
                    None => continue,
                };

                // Skip empty children or those without instance data.
                if child_cmd.is_empty() || child_cmd.instance_index().is_none() {
                    continue;
                }

                // Skip children with group or backdrop effects.
                if self.group_effects.contains_key(&child_id)
                    || self.backdrop_effects.contains_key(&child_id)
                {
                    continue;
                }

                // Opacity check: solid color with alpha >= 1.0, no textures.
                if child_cmd.texture_id(0).is_some() || child_cmd.texture_id(1).is_some() {
                    continue;
                }
                let color = child_cmd
                    .instance_color_override()
                    .unwrap_or([1.0, 1.0, 1.0, 1.0]);
                if color[3] < 1.0 {
                    continue;
                }

                // Need inner bounds for the child.
                let inner = match child_cmd.inner_bounds() {
                    Some(b) => b,
                    None => continue,
                };

                // Rule 5: Reject non-axis-aligned transforms.
                let transform = child_cmd.transform();
                let screen_rect =
                    match transform_rect_to_screen(inner, transform, self.scale_factor) {
                        Some(r) => r,
                        None => continue,
                    };

                // MSAA inset: shrink by 1 physical pixel on each edge when MSAA is active.
                let (mut min_x, mut min_y, mut max_x, mut max_y) = screen_rect;
                if self.msaa_sample_count > 1 {
                    min_x += 1.0;
                    min_y += 1.0;
                    max_x -= 1.0;
                    max_y -= 1.0;
                }

                // Optional: clamp to parent's screen-space bounds (pure optimization).
                if let Some((p_min_x, p_min_y, p_max_x, p_max_y)) = parent_screen_bounds {
                    min_x = min_x.max(p_min_x);
                    min_y = min_y.max(p_min_y);
                    max_x = max_x.min(p_max_x);
                    max_y = max_y.min(p_max_y);
                }

                // Rule: discard rects with negligible area (< 1px²).
                if (max_x - min_x) < 1.0 || (max_y - min_y) < 1.0 {
                    continue;
                }

                self.temp_occlusion_rects.push([min_x, min_y, max_x, max_y]);
            }

            let rect_count = self.temp_occlusion_rects.len() - rect_offset;

            // Sort rects by descending area for early-out in the shader.
            if rect_count > 1 {
                self.temp_occlusion_rects[rect_offset..].sort_by(|a, b| {
                    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
                    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
                    area_b
                        .partial_cmp(&area_a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }

            if rect_count > 0 {
                eprintln!("parent {} -> {} occlusion rects", parent_id, rect_count);
                self.temp_instance_occlusions[parent_instance_idx] = InstanceOcclusion {
                    occlusion_rects_buffer_offset: rect_offset as u32,
                    occlusion_rects_count: rect_count as u32,
                };
            }
        }

        eprintln!(
            "occlusion: rects={}, parents_with_occlusion={}",
            self.temp_occlusion_rects.len(),
            self.temp_instance_occlusions
                .iter()
                .filter(|occ| occ.occlusion_rects_count > 0)
                .count()
        );

        // Re-upload the updated occlusion instance data.
        if !self.temp_instance_occlusions.is_empty() {
            super::preparation::upsert_gpu_buffer(
                &self.device,
                &self.queue,
                &mut self.aggregated_instance_occlusion_buffer,
                "Aggregated Instance Occlusion Buffer",
                bytemuck::cast_slice(&self.temp_instance_occlusions),
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
            );
        }

        // Upload occlusion rects to the storage buffer.
        let rects_bytes: &[u8] = if self.temp_occlusion_rects.is_empty() {
            // Bind a minimal 16-byte buffer to satisfy validation (vec4<f32>).
            bytemuck::cast_slice(&[[0.0f32; 4]])
        } else {
            bytemuck::cast_slice(&self.temp_occlusion_rects)
        };

        super::preparation::upsert_gpu_buffer(
            &self.device,
            &self.queue,
            &mut self.occlusion_rects_storage_buffer,
            "Occlusion Rects Storage Buffer",
            rects_bytes,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );

        // Create/recreate the bind group referencing the storage buffer.
        self.occlusion_rects_bind_group = Some(
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("occlusion_rects_bind_group"),
                layout: &self.occlusion_rects_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self
                        .occlusion_rects_storage_buffer
                        .as_ref()
                        .unwrap()
                        .as_entire_binding(),
                }],
            }),
        );
    }
}

/// Check if any ancestor of `node_id` is in `group_effects`.
fn has_group_effect_ancestor(
    tree: &easy_tree::Tree<DrawCommand>,
    group_effects: &HashMap<usize, EffectInstance>,
    node_id: usize,
) -> bool {
    let mut current = node_id;
    while let Some(parent) = tree.parent_index_unchecked(current) {
        if group_effects.contains_key(&parent) {
            return true;
        }
        current = parent;
    }
    false
}

/// Transform a local-space rect through an axis-aligned transform and scale factor,
/// returning `(min_x, min_y, max_x, max_y)` in physical screen-space pixels.
/// Returns `None` if the transform has rotation, skew, or perspective.
fn transform_rect_to_screen(
    rect: [(f32, f32); 2],
    transform: Option<InstanceTransform>,
    scale_factor: f64,
) -> Option<(f32, f32, f32, f32)> {
    let t = transform.unwrap_or_else(InstanceTransform::identity);

    // Reject perspective transforms.
    if t.col0[3] != 0.0 || t.col1[3] != 0.0 || t.col3[3] != 1.0 {
        return None;
    }

    // Reject rotation/skew.
    if t.col0[1] != 0.0 || t.col1[0] != 0.0 {
        return None;
    }

    let sx = t.col0[0];
    let sy = t.col1[1];
    let tx = t.col3[0];
    let ty = t.col3[1];

    let x0 = rect[0].0 * sx + tx;
    let y0 = rect[0].1 * sy + ty;
    let x1 = rect[1].0 * sx + tx;
    let y1 = rect[1].1 * sy + ty;

    let sf = scale_factor as f32;
    let min_x = x0.min(x1) * sf;
    let min_y = y0.min(y1) * sf;
    let max_x = x0.max(x1) * sf;
    let max_y = y0.max(y1) * sf;

    Some((min_x, min_y, max_x, max_y))
}
