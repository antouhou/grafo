use std::fmt;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureId(pub u64);

impl fmt::Display for TextureId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl TextureId {
    pub const NULL: Self = TextureId(0);

    pub fn new(id: impl Hash) -> Self {
        Self(create_texture_id(id))
    }

    pub fn with(&self, id: impl Hash) -> Self {
        Self::new((self.0, id))
    }
}

fn create_texture_id(id: impl Hash) -> u64 {
    let mut hasher = ahash::AHasher::default();
    id.hash(&mut hasher);
    hasher.finish()
}
