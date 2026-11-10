use super::boundary::PathBoundaryBuilder;
use crate::core::cache::Cache;
use lyon::tessellation::{FillTessellator, StrokeTessellator};

pub(crate) struct ShapeResources {
    pub path_boundary_builder: PathBoundaryBuilder,
    pub fill_tessellator: FillTessellator,
    pub stroke_tessellator: StrokeTessellator,
    pub tessellation_cache: Cache,
}

impl ShapeResources {
    pub(crate) fn new() -> Self {
        Self {
            path_boundary_builder: PathBoundaryBuilder::default(),
            fill_tessellator: FillTessellator::new(),
            stroke_tessellator: StrokeTessellator::new(),
            tessellation_cache: Cache::new(),
        }
    }
}
