use super::placement::CoveragePlacement;
use super::{take_reusable_buffers, CoverageBuffers, CoverageBuilder, PathCoverage};
use crate::core::cache::{CachedTessellation, FrameCache};
use crate::core::vertex::InstanceTransform;
use crate::core::Viewport;
use crate::wgpu_backend::resources::ShapePipelines;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
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
    entries: FrameCache<CoverageKey, Arc<CoverageBuffers>>,
    builder: CoverageBuilder,
    buffers: Vec<CoverageBuffers>,
    retained: Vec<(CoverageKey, Arc<CoverageBuffers>)>,
}

impl CoverageCache {
    pub(in crate::wgpu_backend) fn new() -> Self {
        Self {
            entries: FrameCache::new(),
            builder: CoverageBuilder::default(),
            buffers: Vec::new(),
            retained: Vec::new(),
        }
    }

    pub(in crate::wgpu_backend) fn prepare(
        &mut self,
        tessellation: &Arc<CachedTessellation>,
        transform: Option<InstanceTransform>,
        viewport: Viewport,
        device: &Device,
        queue: &Queue,
        pipelines: &ShapePipelines,
    ) -> PathCoverage {
        let placement = CoveragePlacement::new(transform, tessellation.local_bounds, viewport);
        let key = CoverageKey::new(tessellation, placement.transform);
        let (buffers, _) = self.entries.get_or_insert_with(key.clone(), || {
            self.builder.build(
                tessellation,
                placement.transform,
                viewport.scale_factor,
                viewport.physical_size,
            );
            let buffers = match take_reusable_buffers(&mut self.buffers, &self.builder) {
                Some(mut buffers) => {
                    buffers.update(device, queue, &self.builder, pipelines);
                    buffers
                }
                None => CoverageBuffers::new(device, queue, &self.builder, pipelines),
            };
            Arc::new(buffers)
        });
        PathCoverage {
            key,
            buffers: Arc::clone(buffers),
            tessellation: Arc::clone(tessellation),
            transform,
            translation: placement.translation,
        }
    }

    pub(in crate::wgpu_backend) fn retain(&mut self, coverage: PathCoverage) {
        self.entries.insert(coverage.key, coverage.buffers);
    }

    pub(in crate::wgpu_backend) fn clear(&mut self) {
        self.entries.clear();
        self.buffers.clear();
    }

    pub(in crate::wgpu_backend) fn rebind(
        &mut self,
        coverage: &PathCoverage,
        device: &Device,
        pipelines: &ShapePipelines,
    ) -> Arc<CoverageBuffers> {
        let key = coverage.key.clone();
        let (coverage, _) = self.entries.get_or_insert_with(key, || {
            let mut rebound = coverage.buffers.as_ref().clone();
            rebound.rebind(device, pipelines);
            Arc::new(rebound)
        });
        Arc::clone(coverage)
    }

    pub(in crate::wgpu_backend) fn finish_render(&mut self) {
        self.buffers.clear();
        for (key, coverage) in self.entries.end_frame() {
            match Arc::try_unwrap(coverage) {
                Ok(buffers) => self.buffers.push(buffers),
                Err(coverage) => self.retained.push((key, coverage)),
            }
        }
        for (key, coverage) in self.retained.drain(..) {
            self.entries.insert(key, coverage);
        }
    }
}
