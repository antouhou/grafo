use super::ShapeExecutionResources;
use crate::wgpu_backend::vertex::GeometryBufferRange;
use ahash::HashMap;

#[derive(Default)]
pub(super) struct DrawBufferCompactionStorage {
    /// Maps old instance indices to new ones. Removed instances have no new index.
    instance_relocations: Vec<Option<usize>>,
    retained_geometry_ranges: Vec<GeometryBufferRange>,
    geometry_relocations: HashMap<u32, GeometryBufferRange>,
}

impl DrawBufferCompactionStorage {
    fn clear(&mut self) {
        self.instance_relocations.clear();
        self.retained_geometry_ranges.clear();
        self.geometry_relocations.clear();
    }
}

impl ShapeExecutionResources {
    fn compact_instance_buffers(&mut self) {
        let instance_relocations = &mut self.compaction.instance_relocations;
        instance_relocations.resize(self.instance_transforms.len(), None);
        for location in self
            .draws
            .values()
            .filter_map(|draw| draw.geometry_buffer_location)
        {
            instance_relocations[location.instance_index] = Some(0);
        }
        let mut retained_count = 0;
        for (previous_index, relocated_index) in instance_relocations.iter_mut().enumerate() {
            if relocated_index.is_none() {
                continue;
            }
            self.instance_transforms[retained_count] = self.instance_transforms[previous_index];
            self.instance_colors[retained_count] = self.instance_colors[previous_index];
            self.instance_metadata[retained_count] = self.instance_metadata[previous_index];
            *relocated_index = Some(retained_count);
            retained_count += 1;
        }
        self.instance_transforms.truncate(retained_count);
        self.instance_colors.truncate(retained_count);
        self.instance_metadata.truncate(retained_count);
        for location in self
            .draws
            .values_mut()
            .filter_map(|draw| draw.geometry_buffer_location.as_mut())
        {
            location.instance_index = instance_relocations[location.instance_index]
                .expect("surviving draws have retained instances");
        }
    }

    fn collect_retained_geometry(&mut self) {
        for location in self
            .draws
            .values()
            .filter_map(|draw| draw.geometry_buffer_location)
        {
            let range = location.geometry_range;
            self.compaction
                .geometry_relocations
                .insert(range.index_start, range);
        }
        self.compaction
            .retained_geometry_ranges
            .extend(self.compaction.geometry_relocations.values().copied());
        // Preserve source order so copying down cannot overwrite unread geometry.
        self.compaction
            .retained_geometry_ranges
            .sort_unstable_by_key(|range| range.index_start);
    }

    fn compact_geometry_buffers(&mut self) {
        self.collect_retained_geometry();
        let mut vertex_count = 0;
        let mut index_count = 0;
        for &previous in &self.compaction.retained_geometry_ranges {
            let vertex_start = previous.vertex_start as usize;
            self.vertices.copy_within(
                vertex_start..vertex_start + previous.vertex_count,
                vertex_count,
            );
            self.indices.copy_within(
                previous.index_start as usize..previous.indices().end as usize,
                index_count as usize,
            );
            let relocated = GeometryBufferRange {
                vertex_start: vertex_count as i32,
                index_start: index_count,
                ..previous
            };
            self.compaction
                .geometry_relocations
                .insert(previous.index_start, relocated);
            vertex_count += previous.vertex_count;
            index_count += previous.index_count;
        }
        self.vertices.truncate(vertex_count);
        self.indices.truncate(index_count as usize);
        self.geometry_ranges.retain(|_, range| {
            let Some(relocated) = self.compaction.geometry_relocations.get(&range.index_start)
            else {
                return false;
            };
            *range = *relocated;
            true
        });
        for location in self
            .draws
            .values_mut()
            .filter_map(|draw| draw.geometry_buffer_location.as_mut())
        {
            location.geometry_range =
                self.compaction.geometry_relocations[&location.geometry_range.index_start];
        }
    }

    pub(in crate::wgpu_backend) fn compact_draw_buffers(&mut self) {
        if !self.has_unused_draw_buffers {
            return;
        }
        self.compaction.clear();
        self.compact_instance_buffers();
        self.compact_geometry_buffers();
        self.compaction.clear();
        self.has_unused_draw_buffers = false;
    }
}
