use super::placement::CoveragePlacement;
use super::uploads::CoverageUploads;
use super::{CoverageAllocation, CoverageBuffers, CoverageBuilder, PathCoverage};
use crate::core::cache::{CachedTessellation, FrameCache};
use crate::core::vertex::InstanceTransform;
use crate::core::Viewport;
use crate::wgpu_backend::resources::ShapePipelines;
use ahash::{HashMap, HashMapExt};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};
use wgpu::{Device, Queue};

#[derive(Clone, Debug)]
pub(super) struct CoverageKey {
    tessellation: Arc<CachedTessellation>,
    transform: [u32; 16],
}

impl CoverageKey {
    pub(super) fn new(
        tessellation: &Arc<CachedTessellation>,
        transform: Option<InstanceTransform>,
    ) -> Self {
        let transform = transform.unwrap_or_else(InstanceTransform::identity);
        Self {
            tessellation: Arc::clone(tessellation),
            transform: bytemuck::cast::<_, [f32; 16]>(transform).map(|value| {
                if value == 0.0 {
                    0
                } else {
                    value.to_bits()
                }
            }),
        }
    }
}

impl PartialEq for CoverageKey {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.tessellation, &other.tessellation) && self.transform == other.transform
    }
}

impl Eq for CoverageKey {}

impl Hash for CoverageKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.tessellation).hash(state);
        self.transform.hash(state);
    }
}

pub(in crate::wgpu_backend) struct CoverageCache {
    entries: FrameCache<CoverageKey, Arc<CoverageAllocation>>,
    builder: CoverageBuilder,
    uploads: CoverageUploads,
    buffers: Vec<CoverageBuffers>,
    retained: Vec<(CoverageKey, Arc<CoverageAllocation>)>,
}

impl CoverageCache {
    pub(in crate::wgpu_backend) fn new() -> Self {
        Self {
            entries: FrameCache::new(),
            builder: CoverageBuilder::default(),
            uploads: CoverageUploads::default(),
            buffers: Vec::new(),
            retained: Vec::new(),
        }
    }

    pub(in crate::wgpu_backend) fn prepare(
        &mut self,
        tessellation: &Arc<CachedTessellation>,
        transform: Option<InstanceTransform>,
        viewport: Viewport,
    ) -> PathCoverage {
        let placement = CoveragePlacement::new(transform, tessellation.local_bounds, viewport);
        let key = CoverageKey::new(tessellation, placement.transform);
        let (allocation, _) = self.entries.get_or_insert_with(key.clone(), || {
            self.uploads.prepare(
                &mut self.builder,
                tessellation,
                placement.transform,
                viewport,
            )
        });
        PathCoverage {
            key,
            allocation: Arc::clone(allocation),
            tessellation: Arc::clone(tessellation),
            transform,
            translation: placement.translation,
        }
    }

    pub(in crate::wgpu_backend) fn retain(&mut self, coverage: PathCoverage) {
        self.entries.insert(coverage.key, coverage.allocation);
    }

    pub(in crate::wgpu_backend) fn clear(&mut self) {
        self.entries.clear();
        self.uploads.clear();
        self.buffers.clear();
    }

    pub(in crate::wgpu_backend) fn upload(
        &mut self,
        device: &Device,
        queue: &Queue,
        pipelines: &ShapePipelines,
    ) {
        self.uploads
            .upload(device, queue, pipelines, &mut self.buffers);
    }

    pub(in crate::wgpu_backend) fn rebind<'a>(
        &mut self,
        coverage: impl Iterator<Item = &'a mut PathCoverage>,
        device: &Device,
        pipelines: &ShapePipelines,
    ) {
        self.entries.clear();
        self.buffers.clear();
        let mut rebound = HashMap::new();
        for coverage in coverage {
            let previous = &coverage.allocation;
            let buffers = rebound
                .entry(Arc::as_ptr(&previous.buffers) as usize)
                .or_insert_with(|| {
                    if let Some(buffers) = previous.buffers.get() {
                        let mut buffers = buffers.clone();
                        buffers.rebind(device, pipelines);
                        Arc::new(OnceLock::from(buffers))
                    } else {
                        Arc::clone(&previous.buffers)
                    }
                });
            let (allocation, _) = self.entries.get_or_insert_with(coverage.key.clone(), || {
                Arc::new(CoverageAllocation {
                    buffers: Arc::clone(buffers),
                    range: previous.range,
                    bounds: previous.bounds,
                })
            });
            coverage.allocation = Arc::clone(allocation);
        }
    }

    pub(in crate::wgpu_backend) fn finish_render(&mut self) {
        self.buffers.clear();
        for (key, coverage) in self.entries.end_frame() {
            match Arc::try_unwrap(coverage) {
                Ok(allocation) => {
                    if let Ok(shared) = Arc::try_unwrap(allocation.buffers) {
                        if let Some(buffers) = shared.into_inner() {
                            self.buffers.push(buffers);
                        }
                    }
                }
                Err(coverage) => self.retained.push((key, coverage)),
            }
        }
        for (key, coverage) in self.retained.drain(..) {
            self.entries.insert(key, coverage);
        }
    }
}
