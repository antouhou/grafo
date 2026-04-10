use std::ops::Range;

use bytemuck::Pod;
use bytemuck::Zeroable;
use tracing::warn;

use super::prepared_scene::{PreparedGeometryKey, PreparedGeometryUpload};
use super::types::decide_buffer_sizing;
use super::*;
use crate::renderer::rect_utils::{
    should_use_discard_rect_clip, try_logical_clip_rect_for_draw_command, LogicalClipRect,
};

fn create_gpu_buffer(
    device: &wgpu::Device,
    label: &'static str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(4),
        usage,
        mapped_at_creation: false,
    })
}

fn ensure_gpu_buffer_capacity(
    device: &wgpu::Device,
    buffer: &mut Option<wgpu::Buffer>,
    label: &'static str,
    usage: wgpu::BufferUsages,
    required_size_in_bytes: usize,
) -> bool {
    let decision = decide_buffer_sizing(
        buffer
            .as_ref()
            .map(|existing_buffer| existing_buffer.size()),
        required_size_in_bytes,
    );

    if decision.should_reallocate {
        *buffer = Some(create_gpu_buffer(
            device,
            label,
            decision.target_size,
            usage,
        ));
        true
    } else {
        false
    }
}

fn write_buffer_ranges<T: Pod>(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    source_data: &[T],
    dirty_ranges: &[Range<usize>],
) {
    for dirty_range in dirty_ranges {
        if dirty_range.is_empty() {
            continue;
        }

        let byte_offset = (dirty_range.start * std::mem::size_of::<T>()) as u64;
        queue.write_buffer(
            buffer,
            byte_offset,
            bytemuck::cast_slice(&source_data[dirty_range.clone()]),
        );
    }
}

fn write_full_buffer<T: Pod>(queue: &wgpu::Queue, buffer: &wgpu::Buffer, source_data: &[T]) {
    queue.write_buffer(buffer, 0, bytemuck::cast_slice(source_data));
}

fn coalesce_ranges(dirty_ranges: &mut Vec<Range<usize>>) {
    if dirty_ranges.len() <= 1 {
        return;
    }

    dirty_ranges.sort_unstable_by_key(|dirty_range| dirty_range.start);

    let mut merged_ranges: Vec<Range<usize>> = Vec::with_capacity(dirty_ranges.len());
    for dirty_range in dirty_ranges.drain(..) {
        if let Some(last_merged_range) = merged_ranges.last_mut() {
            if dirty_range.start <= last_merged_range.end {
                last_merged_range.end = last_merged_range.end.max(dirty_range.end);
                continue;
            }
        }

        merged_ranges.push(dirty_range);
    }

    *dirty_ranges = merged_ranges;
}

fn find_first_mismatched_slot(previous_order: &[usize], current_order: &[usize]) -> Option<usize> {
    let shared_prefix_len = previous_order.len().min(current_order.len());
    for index in 0..shared_prefix_len {
        if previous_order[index] != current_order[index] {
            return Some(index);
        }
    }

    (previous_order.len() != current_order.len()).then_some(shared_prefix_len)
}

fn min_dirty_slot(current_first_dirty_slot: Option<usize>, candidate_slot: usize) -> Option<usize> {
    Some(match current_first_dirty_slot {
        Some(existing_slot) => existing_slot.min(candidate_slot),
        None => candidate_slot,
    })
}

fn collect_depth_first_node_ids(
    draw_tree: &easy_tree::Tree<DrawCommand>,
    depth_first_nodes: &mut Vec<usize>,
    traversal_stack: &mut Vec<usize>,
) {
    depth_first_nodes.clear();
    traversal_stack.clear();
    if draw_tree.is_empty() {
        return;
    }

    traversal_stack.push(0);
    while let Some(node_id) = traversal_stack.pop() {
        depth_first_nodes.push(node_id);
        for &child_node_id in draw_tree.children(node_id).iter().rev() {
            traversal_stack.push(child_node_id);
        }
    }
}

fn geometry_key_for_draw_command(
    node_id: usize,
    draw_command: &DrawCommand,
) -> PreparedGeometryKey {
    match draw_command {
        DrawCommand::Shape(shape) => match shape.cache_key {
            Some(cache_key) => PreparedGeometryKey::SharedShape(cache_key),
            None => PreparedGeometryKey::NodeLocal(node_id),
        },
        DrawCommand::CachedShape(cached_shape) => PreparedGeometryKey::CachedShape(cached_shape.id),
    }
}

fn update_prepared_geometry_from_draw_command(
    prepared_scene: &mut super::prepared_scene::PreparedScene,
    geometry_key: PreparedGeometryKey,
    draw_command: &mut DrawCommand,
    shape_cache: &HashMap<u64, CachedShape>,
    tessellator: &mut FillTessellator,
    buffers_pool_manager: &mut PoolManager,
) -> PreparedGeometryUpload {
    match draw_command {
        DrawCommand::Shape(shape) => {
            let tessellated_geometry = shape.tessellate(tessellator, buffers_pool_manager);
            let geometry_upload = prepared_scene.update_geometry(
                geometry_key,
                tessellated_geometry.vertices(),
                tessellated_geometry.indices(),
            );

            if let Some(owned_vertex_buffers) = tessellated_geometry.into_owned() {
                buffers_pool_manager
                    .lyon_vertex_buffers_pool
                    .return_vertex_buffers(owned_vertex_buffers);
            }

            geometry_upload
        }
        DrawCommand::CachedShape(cached_shape) => match shape_cache.get(&cached_shape.id) {
            Some(loaded_cached_shape) => prepared_scene.update_geometry(
                geometry_key,
                &loaded_cached_shape.vertex_buffers.vertices,
                &loaded_cached_shape.vertex_buffers.indices,
            ),
            None => {
                warn!("Cached shape not found in cache");
                prepared_scene.update_geometry(geometry_key, &[], &[])
            }
        },
    }
}

fn build_instance_transform(draw_command: &DrawCommand) -> InstanceTransform {
    draw_command
        .transform()
        .unwrap_or_else(InstanceTransform::identity)
}

fn build_instance_color(draw_command: &DrawCommand) -> InstanceColor {
    InstanceColor {
        color: draw_command
            .instance_color_override()
            .unwrap_or([0.0, 0.0, 0.0, 0.0]),
    }
}

fn build_instance_metadata(
    draw_command: &DrawCommand,
    draw_order: usize,
    inherited_clip_rect: Option<LogicalClipRect>,
) -> InstanceMetadata {
    let texture_flags = (draw_command.texture_id(0).is_some() as u32)
        | ((draw_command.texture_id(1).is_some() as u32) << 1);

    let (clip_rect_min, clip_rect_max) = inherited_clip_rect
        .map(|clip_rect| {
            (
                [clip_rect.min_x, clip_rect.min_y],
                [clip_rect.max_x, clip_rect.max_y],
            )
        })
        .unwrap_or(([1.0, 1.0], [0.0, 0.0]));

    InstanceMetadata {
        draw_order: draw_order as f32,
        texture_flags: texture_flags as f32,
        clip_rect_min,
        clip_rect_max,
    }
}

impl<'a> Renderer<'a> {
    fn ensure_placeholder_geometry_buffers(&mut self) {
        if self.aggregated_vertex_buffer.is_none() {
            self.aggregated_vertex_buffer = Some(crate::pipeline::create_buffer_init(
                &self.device,
                Some("Placeholder Aggregated Vertex Buffer"),
                bytemuck::cast_slice(&[crate::vertex::CustomVertex::zeroed()]),
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
            ));
        }

        if self.aggregated_index_buffer.is_none() {
            self.aggregated_index_buffer = Some(crate::pipeline::create_buffer_init(
                &self.device,
                Some("Placeholder Aggregated Index Buffer"),
                bytemuck::cast_slice(&[0u16]),
                BufferUsages::INDEX | BufferUsages::COPY_DST,
            ));
        }
    }

    fn ensure_identity_instance_buffers(&mut self) {
        if self.identity_instance_transform_buffer.is_none() {
            let identity = InstanceTransform::identity();
            self.identity_instance_transform_buffer = Some(crate::pipeline::create_buffer_init(
                &self.device,
                Some("Identity Instance Transform Buffer"),
                bytemuck::cast_slice(&[identity]),
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
            ));
        }

        if self.identity_instance_color_buffer.is_none() {
            let transparent = InstanceColor::transparent();
            self.identity_instance_color_buffer = Some(crate::pipeline::create_buffer_init(
                &self.device,
                Some("Identity Instance Color Buffer"),
                bytemuck::cast_slice(&[transparent]),
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
            ));
        }

        if self.identity_instance_metadata_buffer.is_none() {
            let metadata = InstanceMetadata::default();
            self.identity_instance_metadata_buffer = Some(crate::pipeline::create_buffer_init(
                &self.device,
                Some("Identity Instance Metadata Buffer"),
                bytemuck::cast_slice(&[metadata]),
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
            ));
        }
    }

    pub(super) fn prepare_render(&mut self) {
        let prepare_started_at = std::time::Instant::now();

        self.ensure_identity_instance_buffers();

        if self.draw_tree.is_empty() {
            self.prepared_scene.drawable_order.clear();
            self.prepared_scene.instance_transforms.clear();
            self.prepared_scene.instance_colors.clear();
            self.prepared_scene.instance_metadata.clear();
            self.prepared_scene.finish_prepare();
            self.last_prepare_cpu_time = prepare_started_at.elapsed();
            return;
        }

        let mut depth_first_nodes =
            std::mem::take(&mut self.prepared_scene.depth_first_nodes_scratch);
        let mut traversal_stack = std::mem::take(&mut self.prepared_scene.traversal_stack_scratch);
        collect_depth_first_node_ids(
            &self.draw_tree,
            &mut depth_first_nodes,
            &mut traversal_stack,
        );

        let mut drawable_order = std::mem::take(&mut self.prepared_scene.drawable_order_scratch);
        drawable_order.clear();
        let mut node_instance_clip_rects =
            std::mem::take(&mut self.prepared_scene.node_instance_clip_rects_scratch);
        let mut node_children_clip_rects =
            std::mem::take(&mut self.prepared_scene.node_children_clip_rects_scratch);

        let mut dirty_vertex_ranges =
            std::mem::take(&mut self.prepared_scene.geometry_vertex_upload_ranges);
        dirty_vertex_ranges.clear();
        let mut dirty_index_ranges =
            std::mem::take(&mut self.prepared_scene.geometry_index_upload_ranges);
        dirty_index_ranges.clear();
        let mut requires_full_geometry_upload = false;
        let canvas_logical_size = to_logical(self.physical_size, self.scale_factor);

        if let Some(max_node_id) = depth_first_nodes.iter().copied().max() {
            if node_instance_clip_rects.len() <= max_node_id {
                node_instance_clip_rects.resize(max_node_id + 1, None);
            }
            if node_children_clip_rects.len() <= max_node_id {
                node_children_clip_rects.resize(max_node_id + 1, None);
            }

            for &node_id in &depth_first_nodes {
                node_instance_clip_rects[node_id] = None;
                node_children_clip_rects[node_id] = None;
            }
        }

        {
            let draw_tree = &mut self.draw_tree;
            let prepared_scene = &mut self.prepared_scene;
            let shape_cache = &self.shape_cache;
            let tessellator = &mut self.tessellator;
            let buffers_pool_manager = &mut self.buffers_pool_manager;

            for &node_id in &depth_first_nodes {
                let inherited_clip_rect =
                    draw_tree
                        .parent_index_unchecked(node_id)
                        .and_then(|parent_node_id| {
                            node_children_clip_rects
                                .get(parent_node_id)
                                .copied()
                                .flatten()
                        });
                node_instance_clip_rects[node_id] = inherited_clip_rect;

                let draw_command = draw_tree
                    .get_mut(node_id)
                    .expect("depth-first node list must only contain live node ids");
                let children_clip_rect = if draw_command.clips_children() {
                    match try_logical_clip_rect_for_draw_command(draw_command) {
                        Some(node_clip_rect)
                            if should_use_discard_rect_clip(
                                node_clip_rect,
                                canvas_logical_size,
                            ) =>
                        {
                            Some(
                                inherited_clip_rect
                                    .map(|clip_rect| clip_rect.intersect(node_clip_rect))
                                    .unwrap_or(node_clip_rect),
                            )
                        }
                        None => inherited_clip_rect,
                        Some(_) => inherited_clip_rect,
                    }
                } else {
                    inherited_clip_rect
                };
                node_children_clip_rects[node_id] = children_clip_rect;

                let geometry_key = geometry_key_for_draw_command(node_id, draw_command);

                prepared_scene.ensure_node_geometry_key(node_id, geometry_key);

                if prepared_scene.dirty_geometry_keys.contains(&geometry_key) {
                    match update_prepared_geometry_from_draw_command(
                        prepared_scene,
                        geometry_key,
                        draw_command,
                        shape_cache,
                        tessellator,
                        buffers_pool_manager,
                    ) {
                        PreparedGeometryUpload::None => {}
                        PreparedGeometryUpload::Partial {
                            vertex_range,
                            index_range,
                        } => {
                            dirty_vertex_ranges.push(vertex_range);
                            dirty_index_ranges.push(index_range);
                        }
                        PreparedGeometryUpload::Full => {
                            requires_full_geometry_upload = true;
                        }
                    }
                }

                let geometry_entry = prepared_scene.geometry_entry(geometry_key);
                let index_buffer_range = geometry_entry.and_then(|entry| entry.index_range());
                let is_empty = geometry_entry
                    .map(|entry| entry.vertices.is_empty() || entry.indices.is_empty())
                    .unwrap_or(true);

                draw_command.set_prepared_geometry(index_buffer_range, is_empty);
                if is_empty {
                    draw_command.set_instance_index(None);
                } else {
                    drawable_order.push(node_id);
                }
            }
        }

        let mut first_dirty_instance_slot = if self.prepared_scene.topology_dirty {
            find_first_mismatched_slot(&self.prepared_scene.drawable_order, &drawable_order)
        } else {
            None
        };

        for (slot, &node_id) in drawable_order.iter().enumerate() {
            let draw_command = self
                .draw_tree
                .get_mut(node_id)
                .expect("drawable order must only contain live node ids");

            if draw_command.instance_index() != Some(slot) {
                first_dirty_instance_slot = min_dirty_slot(first_dirty_instance_slot, slot);
            }
            if self.prepared_scene.dirty_instance_nodes.contains(&node_id) {
                first_dirty_instance_slot = min_dirty_slot(first_dirty_instance_slot, slot);
            }

            draw_command.set_instance_index(Some(slot));
        }

        self.prepared_scene
            .instance_transforms
            .resize(drawable_order.len(), InstanceTransform::identity());
        self.prepared_scene
            .instance_colors
            .resize(drawable_order.len(), InstanceColor::transparent());
        self.prepared_scene
            .instance_metadata
            .resize(drawable_order.len(), InstanceMetadata::default());

        if let Some(first_dirty_slot) = first_dirty_instance_slot {
            for (slot, &node_id) in drawable_order.iter().enumerate().skip(first_dirty_slot) {
                let draw_command = self
                    .draw_tree
                    .get(node_id)
                    .expect("drawable order must only contain live node ids");

                self.prepared_scene.instance_transforms[slot] =
                    build_instance_transform(draw_command);
                self.prepared_scene.instance_colors[slot] = build_instance_color(draw_command);
                self.prepared_scene.instance_metadata[slot] =
                    build_instance_metadata(draw_command, slot, node_instance_clip_rects[node_id]);
            }
        }

        let live_vertex_count = self.prepared_scene.geometry_vertices_len();
        if live_vertex_count > 0 {
            let geometry_vertices = &self.prepared_scene.geometry_vertices[..live_vertex_count];
            let reallocated = ensure_gpu_buffer_capacity(
                &self.device,
                &mut self.aggregated_vertex_buffer,
                "Aggregated Vertex Buffer",
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
                std::mem::size_of_val(geometry_vertices),
            );

            let aggregated_vertex_buffer = self.aggregated_vertex_buffer.as_ref().unwrap();
            if requires_full_geometry_upload || reallocated {
                write_full_buffer(&self.queue, aggregated_vertex_buffer, geometry_vertices);
            } else if !dirty_vertex_ranges.is_empty() {
                coalesce_ranges(&mut dirty_vertex_ranges);
                write_buffer_ranges(
                    &self.queue,
                    aggregated_vertex_buffer,
                    geometry_vertices,
                    &dirty_vertex_ranges,
                );
            }
        } else {
            self.ensure_placeholder_geometry_buffers();
        }

        let live_index_count = self.prepared_scene.geometry_indices_len();
        if live_index_count > 0 {
            let geometry_indices = &self.prepared_scene.geometry_indices[..live_index_count];
            let reallocated = ensure_gpu_buffer_capacity(
                &self.device,
                &mut self.aggregated_index_buffer,
                "Aggregated Index Buffer",
                BufferUsages::INDEX | BufferUsages::COPY_DST,
                std::mem::size_of_val(geometry_indices),
            );

            let aggregated_index_buffer = self.aggregated_index_buffer.as_ref().unwrap();
            if requires_full_geometry_upload || reallocated {
                write_full_buffer(&self.queue, aggregated_index_buffer, geometry_indices);
            } else if !dirty_index_ranges.is_empty() {
                coalesce_ranges(&mut dirty_index_ranges);
                write_buffer_ranges(
                    &self.queue,
                    aggregated_index_buffer,
                    geometry_indices,
                    &dirty_index_ranges,
                );
            }
        } else {
            self.ensure_placeholder_geometry_buffers();
        }

        if !self.prepared_scene.instance_transforms.is_empty() {
            let instance_update_range = first_dirty_instance_slot.map(|first_dirty_slot| {
                first_dirty_slot..self.prepared_scene.instance_transforms.len()
            });

            let reallocated = ensure_gpu_buffer_capacity(
                &self.device,
                &mut self.aggregated_instance_transform_buffer,
                "Aggregated Instance Transform Buffer",
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
                std::mem::size_of_val(self.prepared_scene.instance_transforms.as_slice()),
            );

            let aggregated_instance_transform_buffer =
                self.aggregated_instance_transform_buffer.as_ref().unwrap();
            match instance_update_range.as_ref() {
                Some(instance_update_range) if !reallocated => {
                    write_buffer_ranges(
                        &self.queue,
                        aggregated_instance_transform_buffer,
                        &self.prepared_scene.instance_transforms,
                        std::slice::from_ref(instance_update_range),
                    );
                }
                _ => {
                    write_full_buffer(
                        &self.queue,
                        aggregated_instance_transform_buffer,
                        &self.prepared_scene.instance_transforms,
                    );
                }
            }

            let reallocated = ensure_gpu_buffer_capacity(
                &self.device,
                &mut self.aggregated_instance_color_buffer,
                "Aggregated Instance Color Buffer",
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
                std::mem::size_of_val(self.prepared_scene.instance_colors.as_slice()),
            );

            let aggregated_instance_color_buffer =
                self.aggregated_instance_color_buffer.as_ref().unwrap();
            match instance_update_range.as_ref() {
                Some(instance_update_range) if !reallocated => {
                    write_buffer_ranges(
                        &self.queue,
                        aggregated_instance_color_buffer,
                        &self.prepared_scene.instance_colors,
                        std::slice::from_ref(instance_update_range),
                    );
                }
                _ => {
                    write_full_buffer(
                        &self.queue,
                        aggregated_instance_color_buffer,
                        &self.prepared_scene.instance_colors,
                    );
                }
            }

            let reallocated = ensure_gpu_buffer_capacity(
                &self.device,
                &mut self.aggregated_instance_metadata_buffer,
                "Aggregated Instance Metadata Buffer",
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
                std::mem::size_of_val(self.prepared_scene.instance_metadata.as_slice()),
            );

            let aggregated_instance_metadata_buffer =
                self.aggregated_instance_metadata_buffer.as_ref().unwrap();
            match instance_update_range.as_ref() {
                Some(instance_update_range) if !reallocated => {
                    write_buffer_ranges(
                        &self.queue,
                        aggregated_instance_metadata_buffer,
                        &self.prepared_scene.instance_metadata,
                        std::slice::from_ref(instance_update_range),
                    );
                }
                _ => {
                    write_full_buffer(
                        &self.queue,
                        aggregated_instance_metadata_buffer,
                        &self.prepared_scene.instance_metadata,
                    );
                }
            }
        }

        std::mem::swap(&mut self.prepared_scene.drawable_order, &mut drawable_order);
        drawable_order.clear();
        self.prepared_scene.drawable_order_scratch = drawable_order;

        depth_first_nodes.clear();
        self.prepared_scene.depth_first_nodes_scratch = depth_first_nodes;
        traversal_stack.clear();
        self.prepared_scene.traversal_stack_scratch = traversal_stack;
        self.prepared_scene.node_instance_clip_rects_scratch = node_instance_clip_rects;
        self.prepared_scene.node_children_clip_rects_scratch = node_children_clip_rects;

        dirty_vertex_ranges.clear();
        self.prepared_scene.geometry_vertex_upload_ranges = dirty_vertex_ranges;
        dirty_index_ranges.clear();
        self.prepared_scene.geometry_index_upload_ranges = dirty_index_ranges;

        self.prepared_scene.finish_prepare();
        self.last_prepare_cpu_time = prepare_started_at.elapsed();
    }
}

#[cfg(test)]
mod tests {
    use super::{coalesce_ranges, find_first_mismatched_slot};

    #[test]
    fn coalesce_ranges_merges_overlapping_and_adjacent_ranges() {
        let mut dirty_ranges = vec![6..8, 0..2, 2..4, 5..6];
        coalesce_ranges(&mut dirty_ranges);

        assert_eq!(dirty_ranges, vec![0..4, 5..8]);
    }

    #[test]
    fn find_first_mismatched_slot_detects_length_changes_after_shared_prefix() {
        assert_eq!(find_first_mismatched_slot(&[1, 2, 3], &[1, 2]), Some(2));
        assert_eq!(find_first_mismatched_slot(&[1, 2], &[1, 2, 3]), Some(2));
        assert_eq!(find_first_mismatched_slot(&[1, 2, 3], &[1, 4, 3]), Some(1));
        assert_eq!(find_first_mismatched_slot(&[1, 2, 3], &[1, 2, 3]), None);
    }
}
