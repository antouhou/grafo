use super::cache::coverage_transform_key;
use super::placement::CoveragePlacement;
use super::projection::Projection;
use super::tiles::{CoverageProjection, CoverageTile};
use crate::core::cache::{CachedTessellation, FrameCache};
use crate::core::vertex::{CustomVertex, InstanceTransform};
use crate::core::{geometry, MathRect, UnsignedPhysicalRect, Viewport};
use crate::wgpu_backend::gradient::GradientMaterial;
use crate::wgpu_backend::resources::ShapePipelines;
use ahash::HashMap;
use bins::{requires_precise_projection, TriangleBins};
use bytemuck::Zeroable;
use raster::{RasterInstance, RasterResources};
use smallvec::SmallVec;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock, Weak};
use std::{mem, ops::Range};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer, BufferUsages,
    CommandEncoder, Device, Queue,
};

mod bins;
mod raster;

const ATLAS_SIDE: u32 = 1024;

#[derive(Debug)]
pub(in crate::wgpu_backend) struct StrokeMaskRegion {
    pub(in crate::wgpu_backend) binding: BindGroup,
    pub(in crate::wgpu_backend) indices: Buffer,
    pub(in crate::wgpu_backend) index_range: Range<u32>,
}

#[derive(Debug)]
struct StrokeMask {
    transform: Option<InstanceTransform>,
    bounds: UnsignedPhysicalRect,
    needs_triangle_bins: bool,
    scale: f64,
    regions: OnceLock<SmallVec<[StrokeMaskRegion; 1]>>,
    rasterized_gradient: Option<Arc<GradientMaterial>>,
}

#[derive(Debug)]
pub(in crate::wgpu_backend) struct StrokeCoverage {
    mask: Arc<StrokeMask>,
    tessellation: Arc<CachedTessellation>,
    pub(in crate::wgpu_backend) transform: Option<InstanceTransform>,
    pub(in crate::wgpu_backend) translation: [f32; 2],
}

impl StrokeCoverage {
    pub(in crate::wgpu_backend) fn tessellation(&self) -> &Arc<CachedTessellation> {
        &self.tessellation
    }

    pub(in crate::wgpu_backend) fn rasterized_gradient(&self) -> Option<&Arc<GradientMaterial>> {
        self.mask.rasterized_gradient.as_ref()
    }

    pub(in crate::wgpu_backend) fn regions(&self) -> &[StrokeMaskRegion] {
        self.mask
            .regions
            .get()
            .expect("stroke coverage was prepared")
    }
}

#[derive(Default)]
struct Shelf {
    x: u32,
    y: u32,
    height: u32,
    size: [u32; 2],
    rasterizes_color: bool,
}

impl Shelf {
    fn allocate(&mut self, [width, height]: [u32; 2]) -> Option<[u32; 2]> {
        let [x, y] = if self.x + width <= ATLAS_SIDE && self.y + height <= ATLAS_SIDE {
            [self.x, self.y]
        } else {
            [0, self.y + self.height]
        };
        if y + height > ATLAS_SIDE {
            return None;
        }
        if y != self.y {
            self.height = 0;
        }
        self.x = x + width;
        self.y = y;
        self.height = self.height.max(height);
        self.size = [self.size[0].max(self.x), self.size[1].max(y + height)];
        Some([x, y])
    }
}

#[derive(Clone)]
struct MeshRange {
    vertices: Range<u64>,
    indices: Range<u32>,
}

struct RasterDraw {
    mesh: MeshRange,
    atlas_origin: [u32; 2],
    size: [u32; 2],
    mask_index: usize,
}

struct AtlasPage {
    size: [u32; 2],
    rasterizes_color: bool,
    draws: Range<usize>,
}

struct PendingMask {
    mask: Arc<StrokeMask>,
    regions: SmallVec<[StrokeMaskRegion; 1]>,
}

#[derive(Default)]
struct AtlasBatch {
    masks: Vec<PendingMask>,
    pages: Vec<AtlasPage>,
    draws: Vec<RasterDraw>,
    tiles: Vec<CoverageTile>,
    projections: Vec<CoverageProjection>,
    instances: Vec<RasterInstance>,
    meshes: HashMap<usize, MeshRange>,
    triangle_bins: TriangleBins,
    vertices: Vec<CustomVertex>,
    indices: Vec<u32>,
}

fn append_vertices(storage: &mut Vec<CustomVertex>, vertices: &[CustomVertex]) -> Range<u64> {
    let start = storage.len() * size_of::<CustomVertex>();
    storage.extend_from_slice(vertices);
    start as u64..(storage.len() * size_of::<CustomVertex>()) as u64
}

fn append_indices(storage: &mut Vec<u32>, indices: &[u32]) -> Range<u32> {
    let start = storage.len() as u32;
    storage.extend_from_slice(indices);
    start..storage.len() as u32
}

impl AtlasBatch {
    fn push_mask(
        &mut self,
        mask: Arc<StrokeMask>,
        tessellation: &Arc<CachedTessellation>,
        shelf: &mut Shelf,
    ) {
        let projection = Projection::new(mask.transform, mask.scale, tessellation.local_bounds);
        let projection_index = self.projections.len() as u32;
        self.projections.push(
            projection.map_or_else(CoverageProjection::zeroed, |projection| {
                projection.mapping(
                    [mask.bounds.min.x, mask.bounds.min.y],
                    tessellation.local_bounds,
                )
            }),
        );
        if mask.needs_triangle_bins {
            self.push_binned_mask(&mask, tessellation, shelf, projection_index);
        } else {
            let mesh = self
                .meshes
                .entry(Arc::as_ptr(tessellation) as usize)
                .or_insert_with(|| {
                    let geometry = &tessellation.vertex_buffers;
                    MeshRange {
                        vertices: append_vertices(&mut self.vertices, &geometry.vertices),
                        indices: append_indices(&mut self.indices, &geometry.indices),
                    }
                })
                .clone();
            self.push_region(
                mesh,
                mask.bounds,
                RasterInstance::new(
                    mask.transform.unwrap_or_else(InstanceTransform::identity),
                    mask.scale,
                    false,
                ),
                projection_index,
                shelf,
            );
        }
        self.masks.push(PendingMask {
            mask,
            regions: SmallVec::new(),
        });
    }

    fn push_binned_mask(
        &mut self,
        mask: &StrokeMask,
        tessellation: &CachedTessellation,
        shelf: &mut Shelf,
        projection_index: u32,
    ) {
        let geometry = &tessellation.vertex_buffers;
        let mut bins = mem::take(&mut self.triangle_bins);
        bins.prepare(
            &geometry.vertices,
            &geometry.indices,
            mask.transform.unwrap_or_else(InstanceTransform::identity),
            mask.scale,
            mask.bounds,
        );
        if bins.occupied().next().is_some() {
            let vertices = append_vertices(&mut self.vertices, &bins.projected_vertices);
            for (bounds, indices) in bins.occupied() {
                let indices = if mask.rasterized_gradient.is_some() {
                    // Replacement writes make the last triangle win. Reverse submission
                    // order so overlapping projected triangles retain their original paint.
                    let start = self.indices.len() as u32;
                    self.indices
                        .extend(indices.as_chunks::<3>().0.iter().rev().flatten().copied());
                    start..self.indices.len() as u32
                } else {
                    append_indices(&mut self.indices, indices)
                };
                let mesh = MeshRange {
                    vertices: vertices.clone(),
                    indices,
                };
                self.push_region(
                    mesh,
                    bounds,
                    RasterInstance::new(InstanceTransform::identity(), mask.scale, true),
                    projection_index,
                    shelf,
                );
            }
        }
        self.triangle_bins = bins;
    }

    fn push_region(
        &mut self,
        mesh: MeshRange,
        bounds: UnsignedPhysicalRect,
        mut instance: RasterInstance,
        projection_index: u32,
        shelf: &mut Shelf,
    ) {
        let size = [bounds.width(), bounds.height()];
        let atlas_origin = shelf.allocate(size).unwrap_or_else(|| {
            self.finish_page(shelf);
            shelf
                .allocate(size)
                .expect("a region fits on an empty stroke atlas page")
        });
        self.draws.push(RasterDraw {
            mesh,
            atlas_origin,
            size,
            mask_index: self.masks.len(),
        });
        self.tiles.push(CoverageTile {
            min: bounds.min.to_array(),
            max: bounds.max.to_array(),
            coverage_start: atlas_origin[0] | (atlas_origin[1] << 16),
            projection_index,
        });
        instance.screen_to_clip = [
            2.0 / size[0] as f32,
            -2.0 / size[1] as f32,
            bounds.min.x as f32,
            bounds.min.y as f32,
        ];
        self.instances.push(instance);
    }

    fn finish_page(&mut self, shelf: &mut Shelf) {
        let start = self.pages.last().map_or(0, |page| page.draws.end);
        if start != self.draws.len() {
            self.pages.push(AtlasPage {
                size: shelf.size.map(u32::next_power_of_two),
                rasterizes_color: shelf.rasterizes_color,
                draws: start..self.draws.len(),
            });
        }
        *shelf = Shelf {
            rasterizes_color: shelf.rasterizes_color,
            ..Default::default()
        };
    }

    fn finish(&mut self) {
        for pending in self.masks.drain(..) {
            pending
                .mask
                .regions
                .set(pending.regions)
                .expect("a stroke mask is resolved once");
        }
        self.pages.clear();
        self.draws.clear();
        self.tiles.clear();
        self.projections.clear();
        self.instances.clear();
        self.meshes.clear();
        self.vertices.clear();
        self.indices.clear();
    }
}

#[derive(Clone)]
struct StrokeMaskKey {
    tessellation: Weak<CachedTessellation>,
    transform: [u32; 16],
    // Only baked paint participates in the key. The mask retains the material.
    gradient_material_identity: Option<usize>,
}

impl StrokeMaskKey {
    fn new(
        tessellation: &Arc<CachedTessellation>,
        transform: Option<InstanceTransform>,
        rasterized_gradient: Option<&Arc<GradientMaterial>>,
    ) -> Self {
        Self {
            tessellation: Arc::downgrade(tessellation),
            transform: coverage_transform_key(transform),
            gradient_material_identity: rasterized_gradient
                .map(|material| Arc::as_ptr(material) as usize),
        }
    }
}

impl PartialEq for StrokeMaskKey {
    fn eq(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.tessellation, &other.tessellation)
            && self.transform == other.transform
            && self.gradient_material_identity == other.gradient_material_identity
    }
}

impl Eq for StrokeMaskKey {}

impl Hash for StrokeMaskKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.tessellation.as_ptr().hash(state);
        self.transform.hash(state);
        self.gradient_material_identity.hash(state);
    }
}

pub(in crate::wgpu_backend) struct StrokeAtlas {
    entries: FrameCache<StrokeMaskKey, Arc<StrokeMask>>,
    pending_masks: Vec<(StrokeMaskKey, Weak<StrokeMask>)>,
    pending_gradients: Vec<(StrokeMaskKey, Weak<StrokeMask>)>,
    retained: Vec<(StrokeMaskKey, Arc<StrokeMask>)>,
    batch: AtlasBatch,
    raster: RasterResources,
}

impl Default for StrokeAtlas {
    fn default() -> Self {
        Self {
            entries: FrameCache::new(),
            pending_masks: Vec::new(),
            pending_gradients: Vec::new(),
            retained: Vec::new(),
            batch: AtlasBatch::default(),
            raster: RasterResources::default(),
        }
    }
}

impl StrokeAtlas {
    pub(in crate::wgpu_backend) fn prepare(
        &mut self,
        tessellation: &Arc<CachedTessellation>,
        transform: Option<InstanceTransform>,
        viewport: Viewport,
        rasterized_gradient: Option<&Arc<GradientMaterial>>,
    ) -> StrokeCoverage {
        let placement = CoveragePlacement::new(transform, tessellation.local_bounds, viewport);
        let key = StrokeMaskKey::new(tessellation, placement.transform, rasterized_gradient);
        let mask = self.entries.get(&key).unwrap_or_else(|| {
            let bounds = tessellation.local_bounds;
            let logical = geometry::transformed_bounds_to_logical_screen_rect(
                MathRect::new(bounds[0].into(), bounds[1].into()),
                placement.transform,
            );
            let bounds = geometry::logical_rect_to_scissor_rect(
                logical,
                viewport.scale_factor,
                viewport.physical_size.into(),
            )
            .unwrap_or_else(UnsignedPhysicalRect::zero);
            let visible = !bounds.is_empty();
            let mask = Arc::new(StrokeMask {
                transform: placement.transform,
                bounds,
                needs_triangle_bins: visible
                    && (rasterized_gradient.is_some()
                        || !logical.is_finite()
                        || bounds.width() > ATLAS_SIDE
                        || bounds.height() > ATLAS_SIDE
                        || requires_precise_projection(
                            placement.transform,
                            tessellation.local_bounds,
                            viewport.scale_factor,
                        )),
                scale: viewport.scale_factor,
                regions: if visible {
                    OnceLock::new()
                } else {
                    OnceLock::from(SmallVec::new())
                },
                rasterized_gradient: rasterized_gradient.cloned(),
            });
            if visible {
                self.entries.insert(key.clone(), Arc::clone(&mask));
                let pending = if rasterized_gradient.is_some() {
                    &mut self.pending_gradients
                } else {
                    &mut self.pending_masks
                };
                pending.push((key, Arc::downgrade(&mask)));
            }
            mask
        });
        StrokeCoverage {
            mask,
            tessellation: Arc::clone(tessellation),
            transform,
            translation: placement.translation,
        }
    }

    pub(in crate::wgpu_backend) fn clear(&mut self) {
        self.entries.clear();
        self.pending_masks.clear();
        self.pending_gradients.clear();
    }

    pub(in crate::wgpu_backend) fn finish_render(&mut self) {
        self.retained.extend(
            self.entries
                .end_frame()
                .filter(|(_, mask)| Arc::strong_count(mask) > 1),
        );
        for (key, mask) in self.retained.drain(..) {
            self.entries.insert(key, mask);
        }
    }

    pub(in crate::wgpu_backend) fn rasterize(
        &mut self,
        device: &Device,
        queue: &Queue,
        pipelines: &ShapePipelines,
        encoder: &mut CommandEncoder,
    ) {
        if self.pending_masks.is_empty() && self.pending_gradients.is_empty() {
            return;
        }
        let batch = &mut self.batch;
        for (pending, rasterizes_color) in [
            (&mut self.pending_masks, false),
            (&mut self.pending_gradients, true),
        ] {
            let mut shelf = Shelf {
                rasterizes_color,
                ..Default::default()
            };
            for (key, pending) in pending.drain(..) {
                let Some(mask) = pending.upgrade() else {
                    continue;
                };
                if Arc::strong_count(&mask) == 2 {
                    self.entries.remove(&key);
                } else if let Some(tessellation) = key.tessellation.upgrade() {
                    batch.push_mask(mask, &tessellation, &mut shelf);
                }
            }
            batch.finish_page(&mut shelf);
        }
        if !batch.pages.is_empty() {
            batch
                .pages
                .sort_unstable_by_key(|page| (page.rasterizes_color, page.size));
            self.raster.upload(batch, device, queue);
            let projections = device.create_buffer_init(&BufferInitDescriptor {
                label: Some("stroke_atlas_projections"),
                contents: bytemuck::cast_slice(&batch.projections),
                usage: BufferUsages::STORAGE,
            });
            for page in &batch.pages {
                let resolved = self
                    .raster
                    .rasterize_page(page, batch, device, pipelines, encoder);
                let tiles = device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("stroke_atlas_tiles"),
                    contents: bytemuck::cast_slice(&batch.tiles[page.draws.clone()]),
                    usage: BufferUsages::STORAGE,
                });
                let binding = device.create_bind_group(&BindGroupDescriptor {
                    label: Some("stroke_atlas"),
                    layout: pipelines.stroke_layout(),
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: pipelines.and_uniform_buffer.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: BindingResource::TextureView(&resolved),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: tiles.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: projections.as_entire_binding(),
                        },
                    ],
                });
                let indices = self
                    .raster
                    .quad_indices
                    .as_ref()
                    .expect("stroke quad indices");
                for (slot, draw) in batch.draws[page.draws.clone()].iter().enumerate() {
                    let start = slot as u32 * 6;
                    batch.masks[draw.mask_index].regions.push(StrokeMaskRegion {
                        binding: binding.clone(),
                        indices: indices.clone(),
                        index_range: start..start + 6,
                    });
                }
            }
        }
        batch.finish();
    }
}
