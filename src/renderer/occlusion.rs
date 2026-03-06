use super::*;
use crate::vertex::InstanceOcclusion;

/// Maximum number of occlusion rects stored per node (descendant or subtree summary).
/// Keeps GPU shader loop cost and CPU scratch memory bounded.
const MAX_OCCLUSION_RECTS_PER_NODE: usize = 8;

impl<'a> Renderer<'a> {
    /// Computes per-instance occlusion rects using a hierarchical bottom-up pass.
    ///
    /// For each node, builds two summaries in screen-space physical pixels:
    /// - `descendant_occlusion_summary`: opaque coverage from the node's descendants,
    ///   written to `temp_instance_occlusions` to cull the node itself.
    /// - `subtree_occlusion_summary`: the node's own opaque contribution plus
    ///   the propagated descendant coverage, used by the node's parent.
    ///
    /// Must be called after `prepare_render()` and before `render_segments()`.
    pub(super) fn compute_occlusion_rects(&mut self) {
        debug_assert!(
            self.aggregated_vertex_buffer.is_some(),
            "compute_occlusion_rects() called before prepare_render()"
        );

        let num_nodes = self.draw_tree.len();

        // Clear all per-frame scratch buffers.
        self.temp_occlusion_rects.clear();
        self.temp_node_subtree_rects.clear();
        self.temp_parent_node_ids.clear();

        // Resize and zero per-node indexed scratch buffers.
        self.temp_node_subtree_ranges.clear();
        self.temp_node_subtree_ranges.resize(num_nodes, None);
        self.temp_has_group_effect_ancestor.clear();
        self.temp_has_group_effect_ancestor.resize(num_nodes, false);
        self.temp_subtree_has_backdrop_effect.clear();
        self.temp_subtree_has_backdrop_effect.resize(num_nodes, false);

        // ── Step 1: Precompute has_group_effect_ancestor (pre-order) ──────────
        //
        // In easy_tree, parent index < child index (children are always added after
        // their parent). Iterating 0..num_nodes naturally visits parents before
        // children, giving us a correct pre-order pass.
        for id in 0..num_nodes {
            let parent_has_group = self
                .draw_tree
                .parent_index_unchecked(id)
                .map_or(false, |p| {
                    self.group_effects.contains_key(&p)
                        || self.temp_has_group_effect_ancestor[p]
                });
            self.temp_has_group_effect_ancestor[id] = parent_has_group;
        }

        // ── Step 2: Precompute subtree_has_backdrop_effect (post-order) ───────
        //
        // Children have higher indices than their parent, so iterating in reverse
        // order gives a correct post-order pass (children computed before parent).
        for id in (0..num_nodes).rev() {
            let num_ch = self.draw_tree.children(id).len();
            let mut has_backdrop = false;
            for ci in 0..num_ch {
                let c = self.draw_tree.children(id)[ci];
                if self.backdrop_effects.contains_key(&c)
                    || self.temp_subtree_has_backdrop_effect[c]
                {
                    has_backdrop = true;
                    break;
                }
            }
            self.temp_subtree_has_backdrop_effect[id] = has_backdrop;
        }

        // ── Step 3: Build post-order traversal list ────────────────────────────
        //
        // Pushes children left-to-right onto a DFS stack, collecting nodes in
        // reverse pre-order, then reverses to produce left-to-right post-order.
        // Stored in temp_parent_node_ids (reused scratch vec).
        {
            // Seed the stack with root nodes in reverse order so the first root
            // is processed first (LIFO).
            let mut stack: Vec<usize> = (0..num_nodes)
                .filter(|&id| self.draw_tree.parent_index_unchecked(id).is_none())
                .collect();
            stack.reverse();

            while let Some(id) = stack.pop() {
                self.temp_parent_node_ids.push(id);
                let num_ch = self.draw_tree.children(id).len();
                for ci in 0..num_ch {
                    stack.push(self.draw_tree.children(id)[ci]);
                }
            }
            // Reverse pre-order → post-order (children appear before their parent).
            self.temp_parent_node_ids.reverse();
        }

        // ── Step 4: Process each node in post-order ───────────────────────────
        //
        // Working rect buffer reused per node to avoid per-loop allocations.
        let mut working_rects: Vec<[f32; 4]> =
            Vec::with_capacity(MAX_OCCLUSION_RECTS_PER_NODE * 2);

        for i in 0..self.temp_parent_node_ids.len() {
            let node_id = self.temp_parent_node_ids[i];

            // Extract all data we need from the draw command in a single borrow
            // scope, so subsequent mutable accesses to other fields are unblocked.
            let node_data = {
                let Some(cmd) = self.draw_tree.get(node_id) else {
                    continue;
                };
                if cmd.is_empty() {
                    continue;
                }
                let clips = cmd.clips_children();
                let prop_clip: Option<[f32; 4]> = if clips {
                    propagation_clip_rect(cmd, self.msaa_sample_count, self.scale_factor)
                } else {
                    None
                };
                let own_rect =
                    opaque_contribution_rect(cmd, self.msaa_sample_count, self.scale_factor);
                let instance_idx = cmd.instance_index();
                (clips, prop_clip, own_rect, instance_idx)
            };

            let (clips_children, prop_clip, own_rect, instance_idx) = node_data;

            // Effect-boundary exclusions: skip this node entirely if it or its
            // ancestors are group-effect nodes, or if its subtree contains a
            // backdrop-effect node.
            if self.group_effects.contains_key(&node_id)
                || self.temp_has_group_effect_ancestor[node_id]
                || self.temp_subtree_has_backdrop_effect[node_id]
            {
                // Leave temp_node_subtree_ranges[node_id] = None (already initialised).
                continue;
            }

            working_rects.clear();

            // Whether descendant coverage can be used safely:
            // - non-clipping parent → always yes (no clip boundary to worry about)
            // - clipping parent with a known conservative clip rect → yes, intersect
            // - clipping parent without a conservative clip rect → no (can't guarantee
            //   that descendant rects lie within the actual clip scope)
            let can_use_descendants = !clips_children || prop_clip.is_some();

            // ── Step A: Collect descendant summary ────────────────────────────
            if can_use_descendants {
                let num_ch = self.draw_tree.children(node_id).len();
                for ci in 0..num_ch {
                    let child_id = self.draw_tree.children(node_id)[ci];

                    let Some((offset, count)) = self.temp_node_subtree_ranges[child_id] else {
                        continue;
                    };
                    let start = offset as usize;
                    let end = start + count as usize;

                    if let Some(clip) = prop_clip {
                        // Clipping parent: intersect each child subtree rect with
                        // the node's conservative clip rect before propagating.
                        for ri in start..end {
                            let rect = self.temp_node_subtree_rects[ri];
                            if let Some(clipped) = intersect_rects(rect, clip) {
                                if is_rect_significant(clipped) {
                                    working_rects.push(clipped);
                                }
                            }
                        }
                    } else {
                        // Non-clipping parent: propagate child subtree rects unchanged.
                        for ri in start..end {
                            working_rects.push(self.temp_node_subtree_rects[ri]);
                        }
                    }
                }

                // Keep the working set bounded before writing to the instance.
                merge_occlusion_rects_bounded(&mut working_rects, MAX_OCCLUSION_RECTS_PER_NODE);
            }

            // Write descendant summary → temp_instance_occlusions so this node
            // can be culled by its own descendants in the GPU fragment shader.
            if !working_rects.is_empty() {
                if let Some(idx) = instance_idx {
                    let rect_offset = self.temp_occlusion_rects.len();
                    self.temp_occlusion_rects.extend_from_slice(&working_rects);
                    self.temp_instance_occlusions[idx] = InstanceOcclusion {
                        occlusion_rects_buffer_offset: rect_offset as u32,
                        occlusion_rects_count: working_rects.len() as u32,
                    };
                }
            }

            // ── Step B: Build subtree summary for the parent ──────────────────
            //
            // Extend the working set with this node's own opaque contribution.
            // The result (descendant coverage + own coverage) is what the parent
            // will read when computing its own descendant summary.
            if let Some(own) = own_rect {
                working_rects.push(own);
                merge_occlusion_rects_bounded(&mut working_rects, MAX_OCCLUSION_RECTS_PER_NODE);
            }

            if !working_rects.is_empty() {
                let subtree_offset = self.temp_node_subtree_rects.len() as u32;
                let subtree_count = working_rects.len() as u32;
                self.temp_node_subtree_rects
                    .extend_from_slice(&working_rects);
                self.temp_node_subtree_ranges[node_id] =
                    Some((subtree_offset, subtree_count));
            }
        }

        // ── Step 5: Upload GPU buffers ─────────────────────────────────────────

        let nodes_with_occlusion = self
            .temp_instance_occlusions
            .iter()
            .filter(|occ| occ.occlusion_rects_count > 0)
            .count();
        let total_instances = self.temp_instance_occlusions.len();
        eprintln!(
            "occlusion: nodes_culled={}/{}, total_rects={}, subtree_rects={}",
            nodes_with_occlusion,
            total_instances,
            self.temp_occlusion_rects.len(),
            self.temp_node_subtree_rects.len(),
        );

        // Re-upload the updated per-instance occlusion vertex data.
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

// ── Helper functions ──────────────────────────────────────────────────────────

/// Returns the MSAA-normalised screen-space opaque rect for a node's own
/// contribution, or `None` if the node is ineligible (textured, non-opaque,
/// no inner bounds, or non-axis-aligned transform).
fn opaque_contribution_rect(
    cmd: &DrawCommand,
    msaa_sample_count: u32,
    scale_factor: f64,
) -> Option<[f32; 4]> {
    if cmd.texture_id(0).is_some() || cmd.texture_id(1).is_some() {
        return None;
    }
    let color = cmd.instance_color_override().unwrap_or([1.0, 1.0, 1.0, 1.0]);
    if color[3] < 1.0 {
        return None;
    }
    let inner = cmd.inner_bounds()?;
    let sr = transform_rect_to_screen(inner, cmd.transform(), scale_factor)?;
    normalize_occlusion_rect_for_msaa(sr, msaa_sample_count)
}

/// Returns a conservative clip rect (MSAA-normalised) for propagating descendant
/// occlusion rects through a clipping node:
/// - rect nodes: use `rect_bounds()` (exact clip scope)
/// - non-rect nodes: use `inner_bounds()` if available (conservative)
/// - otherwise: `None` (propagation must stop here)
fn propagation_clip_rect(
    cmd: &DrawCommand,
    msaa_sample_count: u32,
    scale_factor: f64,
) -> Option<[f32; 4]> {
    let bounds = if cmd.is_rect() {
        cmd.rect_bounds()?
    } else {
        cmd.inner_bounds()?
    };
    let sr = transform_rect_to_screen(bounds, cmd.transform(), scale_factor)?;
    normalize_occlusion_rect_for_msaa(sr, msaa_sample_count)
}

/// Applies MSAA normalisation to a raw screen-space rect by insetting each edge
/// by 1 physical pixel when MSAA is active.  Returns `None` if the rect becomes
/// degenerate (< 1 px in either dimension) after the inset.
///
/// All coverage rects must pass through this function exactly once at creation.
/// Propagated rects must NOT be re-normalised as they move up the tree.
fn normalize_occlusion_rect_for_msaa(
    rect: (f32, f32, f32, f32),
    msaa_sample_count: u32,
) -> Option<[f32; 4]> {
    let (mut min_x, mut min_y, mut max_x, mut max_y) = rect;
    if msaa_sample_count > 1 {
        min_x += 1.0;
        min_y += 1.0;
        max_x -= 1.0;
        max_y -= 1.0;
    }
    if (max_x - min_x) < 1.0 || (max_y - min_y) < 1.0 {
        return None;
    }
    Some([min_x, min_y, max_x, max_y])
}

/// Returns the intersection of two screen-space rects, or `None` if they do not
/// overlap.
fn intersect_rects(a: [f32; 4], b: [f32; 4]) -> Option<[f32; 4]> {
    let min_x = a[0].max(b[0]);
    let min_y = a[1].max(b[1]);
    let max_x = a[2].min(b[2]);
    let max_y = a[3].min(b[3]);
    if max_x > min_x && max_y > min_y {
        Some([min_x, min_y, max_x, max_y])
    } else {
        None
    }
}

/// Returns `true` if the rect has area ≥ 1 px² (both dimensions ≥ 1 px).
#[inline]
fn is_rect_significant(r: [f32; 4]) -> bool {
    (r[2] - r[0]) >= 1.0 && (r[3] - r[1]) >= 1.0
}

/// Sorts a working set of rects by descending area and truncates to `max_count`.
///
/// Sorting largest-first lets the GPU shader exit early once a fragment is fully
/// covered.  Truncating is conservative: we only lose optimisation opportunities,
/// never introduce incorrect discards.
fn merge_occlusion_rects_bounded(rects: &mut Vec<[f32; 4]>, max_count: usize) {
    if rects.is_empty() {
        return;
    }
    rects.sort_unstable_by(|a, b| {
        let area_a = (a[2] - a[0]) * (a[3] - a[1]);
        let area_b = (b[2] - b[0]) * (b[3] - b[1]);
        area_b
            .partial_cmp(&area_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rects.truncate(max_count);
}

/// Transforms a local-space rect through an axis-aligned transform and scale factor,
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
