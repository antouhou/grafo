use super::types::DrawTreeNode;
use ahash::{HashMap, HashSet};
use smallvec::SmallVec;

fn texture_ids(node: &DrawTreeNode) -> SmallVec<[u64; 2]> {
    let mut texture_ids = SmallVec::new();
    if let DrawTreeNode::CachedShape(shape) = node {
        for texture_id in shape
            .instance
            .textures
            .iter()
            .filter_map(|texture| texture.texture_id)
        {
            if !texture_ids.contains(&texture_id) {
                texture_ids.push(texture_id);
            }
        }
    }
    texture_ids
}

#[derive(Default)]
pub(super) struct TextureIndex {
    shapes_by_texture: HashMap<u64, HashSet<usize>>,
    // A SmallVec since having a fixed array of Option<u64> seems a bit of a hassle
    textures_by_shape: HashMap<usize, SmallVec<[u64; 2]>>,
}

impl TextureIndex {
    pub(super) fn shapes_using(&self, texture_id: u64) -> impl Iterator<Item = usize> + '_ {
        self.shapes_by_texture
            .get(&texture_id)
            .into_iter()
            .flat_map(|shapes| shapes.iter().copied())
    }

    pub(super) fn replace(&mut self, shape_id: usize, node: &DrawTreeNode) {
        let texture_ids = texture_ids(node);
        if self.textures_by_shape.get(&shape_id) == Some(&texture_ids) {
            return;
        }
        self.remove(shape_id);
        if texture_ids.is_empty() {
            return;
        }
        for &texture_id in &texture_ids {
            self.shapes_by_texture
                .entry(texture_id)
                .or_default()
                .insert(shape_id);
        }
        self.textures_by_shape.insert(shape_id, texture_ids);
    }

    pub(super) fn remove(&mut self, shape_id: usize) {
        let Some(texture_ids) = self.textures_by_shape.remove(&shape_id) else {
            return;
        };
        for texture_id in texture_ids {
            if let Some(shapes) = self.shapes_by_texture.get_mut(&texture_id) {
                shapes.remove(&shape_id);
                if shapes.is_empty() {
                    self.shapes_by_texture.remove(&texture_id);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
