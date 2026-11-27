use super::ClipCoverage;
use crate::core::cache::FrameCache;
use crate::core::{UnsignedPhysicalRect, Viewport};
use crate::wgpu_backend::execution::shapes::coverage::cache::CoverageKey;
use crate::wgpu_backend::execution::shapes::coverage::placement::CoveragePlacement;
use crate::wgpu_backend::execution::shapes::coverage::projection::Projection;
use crate::wgpu_backend::execution::shapes::coverage::tiles::{
    CoverageDrawRange, CoverageProjection,
};
use crate::wgpu_backend::execution::shapes::coverage::uploads::CoverageStorage;
use crate::wgpu_backend::execution::shapes::coverage::{
    CoverageAllocation, CoverageBuffers, CoverageGeometry, CoverageUpload, PathCoverage,
};
use crate::wgpu_backend::resources::ShapePipelines;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};
use wgpu::{Device, Queue};

#[derive(Clone, Debug)]
pub(in crate::wgpu_backend::execution::shapes::coverage) struct CachedClipCoverage {
    coverage: Arc<ClipCoverage>,
    projection: Option<CoverageProjection>,
}

#[derive(Clone, Debug)]
struct CoverageIdentity(Arc<ClipCoverage>);

impl PartialEq for CoverageIdentity {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for CoverageIdentity {}
impl Hash for CoverageIdentity {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.0).hash(state);
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum SourceKey {
    Shape(CoverageKey, [u32; 2]),
    Rectangle([u64; 4]),
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct PreparedKey {
    coverage: CoverageIdentity,
    denominator: Option<CoverageIdentity>,
    source: Option<CoverageKey>,
    translation: [i32; 2],
}

pub(in crate::wgpu_backend) struct ClipCoverageCache {
    storage: Vec<Arc<ClipCoverage>>,
    offscreen: FrameCache<CoverageIdentity, Arc<ClipCoverage>>,
    sources: FrameCache<SourceKey, Arc<ClipCoverage>>,
    intersections: FrameCache<(CoverageIdentity, CoverageIdentity), Arc<ClipCoverage>>,
    prepared: FrameCache<PreparedKey, Arc<CoverageAllocation>>,
    geometry: CoverageGeometry,
    solid_samples: Vec<u64>,
    indices: Vec<u32>,
    allocation_storage: Vec<Arc<CoverageAllocation>>,
}

impl ClipCoverageCache {
    pub fn new() -> Self {
        Self {
            storage: Vec::new(),
            offscreen: FrameCache::new(),
            sources: FrameCache::new(),
            intersections: FrameCache::new(),
            prepared: FrameCache::new(),
            geometry: CoverageGeometry::default(),
            solid_samples: Vec::new(),
            indices: Vec::new(),
            allocation_storage: Vec::new(),
        }
    }

    fn shape_rectangle(
        &mut self,
        coverage: &PathCoverage,
        viewport: Viewport,
    ) -> Option<CachedClipCoverage> {
        let bounds = coverage.tessellation.local_bounds;
        if !coverage.tessellation.is_rect {
            return None;
        }
        let placement = CoveragePlacement::new(coverage.transform, bounds, viewport);
        let projection = Projection::new(placement.transform, viewport.scale_factor, bounds)?;
        if !projection.preserves_axis_alignment() {
            return None;
        }
        let corners = bounds.map(|(x, y)| projection.project([x, y]));
        if !corners.into_iter().flatten().all(f32::is_finite) {
            return None;
        }
        let translation = coverage.translation.map(f64::from);
        let source = self.rectangle(
            [
                f64::from(corners[0][0].min(corners[1][0])) + translation[0],
                f64::from(corners[0][1].min(corners[1][1])) + translation[1],
                f64::from(corners[0][0].max(corners[1][0])) + translation[0],
                f64::from(corners[0][1].max(corners[1][1])) + translation[1],
            ],
            UnsignedPhysicalRect::from_size(viewport.physical_size.into()),
        );
        Some(CachedClipCoverage {
            coverage: source,
            projection: Some(projection.mapping(coverage.allocation.bounds.min.to_array(), bounds)),
        })
    }

    fn prepare_shape(&mut self, coverage: &PathCoverage, viewport: Viewport) -> CachedClipCoverage {
        if let Some(rectangle) = self.shape_rectangle(coverage, viewport) {
            return rectangle;
        }
        let key = SourceKey::Shape(
            coverage.key.clone(),
            [
                coverage.translation[0].to_bits(),
                coverage.translation[1].to_bits(),
            ],
        );
        let allocation = &coverage.allocation;
        let geometry = allocation
            .storage
            .geometry
            .get()
            .expect("coverage uploads were sealed");
        let start = allocation.range.tile_start as usize;
        let tiles = &geometry.tiles[start..start + allocation.range.tile_count as usize];
        let projection = tiles
            .first()
            .map(|tile| geometry.projections[tile.projection_index as usize]);
        if let Some(source) = self.sources.get(&key) {
            return CachedClipCoverage {
                coverage: source,
                projection,
            };
        }
        let mut source = ClipCoverage::from_raster(
            tiles,
            &geometry.coverage,
            coverage.allocation.bounds,
            coverage.translation,
            &mut self.storage,
            &mut self.solid_samples,
        );
        Arc::get_mut(&mut source)
            .expect("new coverage has one owner")
            .is_convex = (coverage.tessellation.is_rect
            || coverage.tessellation.rounded_rect_boundary.is_some())
            && coverage
                .transform
                .is_none_or(|transform| transform.col0[3] == 0.0 && transform.col1[3] == 0.0);
        self.sources.insert(key, Arc::clone(&source));
        CachedClipCoverage {
            coverage: source,
            projection,
        }
    }

    pub fn shape(&mut self, coverage: &PathCoverage, viewport: Viewport) -> Arc<ClipCoverage> {
        let cached = coverage
            .clipping
            .get_or_init(|| self.prepare_shape(coverage, viewport));
        Arc::clone(&cached.coverage)
    }

    pub fn rectangle(
        &mut self,
        bounds: [f64; 4],
        viewport: UnsignedPhysicalRect,
    ) -> Arc<ClipCoverage> {
        let key = SourceKey::Rectangle([
            bounds[0].to_bits(),
            bounds[1].to_bits(),
            bounds[2].to_bits(),
            bounds[3].to_bits(),
        ]);
        if let Some(source) = self.sources.get(&key) {
            return source;
        }
        let source = ClipCoverage::rectangle(bounds, viewport, &mut self.storage);
        self.sources.insert(key, Arc::clone(&source));
        source
    }

    pub fn intersection(
        &mut self,
        first: &Arc<ClipCoverage>,
        second: &Arc<ClipCoverage>,
    ) -> Arc<ClipCoverage> {
        if Arc::ptr_eq(first, second) {
            return Arc::clone(first);
        }
        let key = (
            CoverageIdentity(Arc::clone(first)),
            CoverageIdentity(Arc::clone(second)),
        );
        if let Some(combined) = self.intersections.get(&key) {
            return combined;
        }
        let combined = ClipCoverage::intersection(first, second, &mut self.storage);
        self.intersections.insert(key, Arc::clone(&combined));
        combined
    }

    pub fn for_offscreen_target(
        &mut self,
        parent: &Arc<ClipCoverage>,
        viewport: UnsignedPhysicalRect,
    ) -> Arc<ClipCoverage> {
        let key = CoverageIdentity(Arc::clone(parent));
        if let Some(coverage) = self.offscreen.get(&key) {
            return coverage;
        }
        let coverage = ClipCoverage::for_offscreen_target(parent, viewport, &mut self.storage);
        self.offscreen.insert(key, Arc::clone(&coverage));
        coverage
    }

    pub fn prepare(
        &mut self,
        coverage: &Arc<ClipCoverage>,
        denominator: Option<&Arc<ClipCoverage>>,
        source: Option<&PathCoverage>,
        device: &Device,
        queue: &Queue,
        pipelines: &ShapePipelines,
    ) -> Arc<CoverageAllocation> {
        let translation = match source {
            Some(source) => [source.translation[0] as i32, source.translation[1] as i32],
            None => [0; 2],
        };
        let key = PreparedKey {
            coverage: CoverageIdentity(Arc::clone(coverage)),
            denominator: denominator.map(|value| CoverageIdentity(Arc::clone(value))),
            source: source.map(|source| source.key.clone()),
            translation,
        };
        if let Some(prepared) = self.prepared.get(&key) {
            return prepared;
        }
        let projection = source
            .and_then(|source| source.clipping.get())
            .and_then(|source| source.projection);
        let parent = denominator.map(Arc::as_ref);
        let solid_tile_count =
            coverage.compile(parent, translation, projection, &mut self.geometry);
        let upload = CoverageUpload {
            geometry: &self.geometry,
            solid_tile_count,
        };
        let bounds = if coverage.bounds.is_empty() {
            UnsignedPhysicalRect::zero()
        } else {
            coverage
                .bounds
                .translate((-translation[0], -translation[1]).into())
                .to_u32()
        };
        let range = CoverageDrawRange {
            tile_start: 0,
            tile_count: self.geometry.tiles.len() as u32,
            solid_tile_count,
        };
        let allocation = if let Some(mut allocation) = self.allocation_storage.pop() {
            let recycled = Arc::get_mut(&mut allocation).expect("recycled coverage has one owner");
            Arc::get_mut(&mut recycled.storage)
                .expect("recycled coverage owns its buffers")
                .buffers
                .get_mut()
                .expect("recycled coverage was uploaded")
                .update(device, queue, upload, pipelines, &mut self.indices);
            recycled.range = range;
            recycled.bounds = bounds;
            allocation
        } else {
            Arc::new(CoverageAllocation {
                storage: Arc::new(CoverageStorage {
                    buffers: OnceLock::from(CoverageBuffers::new(
                        device,
                        queue,
                        upload,
                        pipelines,
                        &mut self.indices,
                    )),
                    geometry: OnceLock::new(),
                }),
                range,
                bounds,
            })
        };
        self.prepared.insert(key, Arc::clone(&allocation));
        allocation
    }

    pub fn finish_render(&mut self) {
        for (_, allocation) in self.prepared.end_frame() {
            if Arc::strong_count(&allocation) == 1 && Arc::strong_count(&allocation.storage) == 1 {
                self.allocation_storage.push(allocation);
            }
        }
        self.storage
            .extend(self.intersections.end_frame().map(|(_, coverage)| coverage));
        self.storage
            .extend(self.offscreen.end_frame().map(|(_, coverage)| coverage));
        self.storage
            .extend(self.sources.end_frame().map(|(_, coverage)| coverage));
        self.storage
            .retain(|coverage| Arc::strong_count(coverage) == 1);
    }

    pub fn clear(&mut self) {
        self.offscreen.clear();
        self.sources.clear();
        self.intersections.clear();
        self.prepared.clear();
        self.allocation_storage.clear();
        self.storage.clear();
    }
}
