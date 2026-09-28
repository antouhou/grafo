use super::ShapeExecutionResources;
use crate::commands::ShapeDrawId;
use crate::wgpu_backend::vertex::GeometryBufferRange;

fn shift_geometry_range(range: &mut GeometryBufferRange, removed: GeometryBufferRange) {
    if range.index_start > removed.index_start {
        range.index_start -= removed.index_count;
        range.vertex_start = (range.vertex_start as usize - removed.vertex_count) as i32;
    }
}

impl ShapeExecutionResources {
    fn remove_geometry(&mut self, removed: GeometryBufferRange) {
        let vertex_start = removed.vertex_start as usize;
        self.vertices
            .drain(vertex_start..vertex_start + removed.vertex_count);
        self.indices
            .drain(removed.index_start as usize..removed.indices().end as usize);
        self.geometry_ranges.retain(|_, range| {
            if *range == removed {
                return false;
            }
            shift_geometry_range(range, removed);
            true
        });
        for location in self
            .draws
            .values_mut()
            .filter_map(|draw| draw.location.as_mut())
        {
            shift_geometry_range(&mut location.geometry_range, removed);
        }
    }

    /// Compacts queued instance data and drops geometry after its last draw is removed.
    pub(in crate::wgpu_backend) fn remove_draw(&mut self, id: ShapeDrawId) {
        let Some(location) = self.draws.remove(&id.0).and_then(|draw| draw.location) else {
            return;
        };
        self.instance_transforms.remove(location.instance_index);
        self.instance_colors.remove(location.instance_index);
        self.instance_metadata.remove(location.instance_index);
        let mut is_geometry_shared = false;
        for remaining in self
            .draws
            .values_mut()
            .filter_map(|draw| draw.location.as_mut())
        {
            if remaining.instance_index > location.instance_index {
                remaining.instance_index -= 1;
            }
            is_geometry_shared |= remaining.geometry_range == location.geometry_range;
        }
        if !is_geometry_shared {
            self.remove_geometry(location.geometry_range);
        }
    }
}
