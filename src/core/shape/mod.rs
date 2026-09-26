//! Rectangles and paths. Set each instance's fill through the renderer.
//!
//! See [`Shape::rect`], [`Shape::rounded_rect`], and [`ShapeBuilder`] for examples.

use crate::core::cache::CachedTessellation;
use crate::core::gradient::types::Fill;
use crate::core::util::ShapeResources;
use crate::core::vertex::{CustomVertex, InstanceTransform, TextureUvTransform};
use crate::core::{Color, Stroke};
use ahash::AHashMap;
use lyon::lyon_tessellation::{
    BuffersBuilder, FillOptions, FillTessellator, FillVertex, VertexBuffers,
};
use lyon::path::Winding;
use lyon::tessellation::FillVertexConstructor;
use smallvec::SmallVec;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct CachedShapeHandle {
    pub tessellation: Arc<CachedTessellation>,
    /// Whether the original shape was an axis-aligned rectangle. Used to enable scissor-based
    /// clipping instead of stencil for rect parents.
    pub(crate) is_rect: bool,
    /// The local-space bounding rect when `is_rect` is true, for scissor computation.
    pub(crate) rect_bounds: Option<[(f32, f32); 2]>,
    pub geometry_id: Option<u64>,
}

impl CachedShapeHandle {
    /// Caches tessellation under `geometry_id` and reuses it during buffer aggregation.
    /// Equal IDs must identify identical geometry. Use a content-derived key, or `None`
    /// to disable reuse when no reliable key is available.
    pub(crate) fn new(
        shape: &Shape,
        tessellator: &mut FillTessellator,
        shape_resources: &mut ShapeResources,
        geometry_id: Option<u64>,
    ) -> Self {
        let (is_rect, rect_bounds) = match shape {
            Shape::Rect(r) => (true, Some(r.rect)),
            _ => (false, None),
        };
        let tessellation = shape.tessellate(tessellator, shape_resources, geometry_id);
        Self {
            tessellation,
            is_rect,
            rect_bounds,
            geometry_id,
        }
    }

    #[inline]
    pub fn vertex_buffers(&self) -> &Arc<VertexBuffers<CustomVertex, u16>> {
        &self.tessellation.vertex_buffers
    }

    #[inline]
    pub fn local_bounds(&self) -> [(f32, f32); 2] {
        self.tessellation.local_bounds
    }

    #[inline]
    pub fn texture_mapping_size(&self) -> [f32; 2] {
        self.tessellation.texture_mapping_size
    }
}

#[derive(Debug)]
pub struct ShapeInstance {
    pub cached_shape: CachedShapeHandle,
    /// Optional per-shape transform applied in pixel space before clip-space normalization.
    pub transform: Option<InstanceTransform>,
    /// Texture sources associated with this cached shape.
    pub textures: [ShapeTextureOptions; 2],
    /// Linear RGBA color for a solid fill. Other fills leave this unset.
    pub color_override: Option<[f32; 4]>,
    /// A solid or gradient fill. `None` leaves the shape transparent
    pub fill: Option<Fill>,
}

impl ShapeInstance {
    pub fn new(cached_shape: CachedShapeHandle, options: ShapeDrawCommandOptions) -> Self {
        Self {
            cached_shape,
            transform: options.transform,
            textures: [options.background_texture, options.foreground_texture],
            color_override: match options.fill.as_ref() {
                Some(Fill::Solid(color)) => Some(color.normalize()),
                _ => None,
            },
            fill: options.fill,
        }
    }

    pub fn has_gradient_fill(&self) -> bool {
        matches!(&self.fill, Some(Fill::Gradient(_)))
    }
}

fn rect_size(rect_bounds: [(f32, f32); 2]) -> [f32; 2] {
    [
        (rect_bounds[1].0 - rect_bounds[0].0).abs().max(1e-6),
        (rect_bounds[1].1 - rect_bounds[0].1).abs().max(1e-6),
    ]
}

fn compute_vertex_bounds(vertices: &[CustomVertex]) -> [(f32, f32); 2] {
    if vertices.is_empty() {
        return [(0.0, 0.0), (1.0, 1.0)];
    }

    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    for vertex in vertices {
        let x = vertex.position[0];
        let y = vertex.position[1];
        if x < min_x {
            min_x = x;
        }
        if y < min_y {
            min_y = y;
        }
        if x > max_x {
            max_x = x;
        }
        if y > max_y {
            max_y = y;
        }
    }

    [(min_x, min_y), (max_x, max_y)]
}

/// A rectangle or a path made of lines and Bezier curves.
///
/// Use [`Self::rect`] or [`Self::rounded_rect`] for rectangles, or [`Self::builder`]
/// for a custom path. Set the fill when queueing the shape with [`ShapeDrawCommandOptions`].
#[derive(Debug, Clone)]
pub enum Shape {
    /// A custom path shape defined using Bezier curves and lines.
    Path(PathShape),
    /// An axis-aligned rectangle with square corners.
    Rect(RectShape),
}

impl Shape {
    /// Starts a path with the default black stroke. See [`ShapeBuilder`] for an example.
    pub fn builder() -> ShapeBuilder {
        ShapeBuilder::new()
    }

    /// Creates a rectangle from its top-left and bottom-right coordinates.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::{Color, Shape, Stroke};
    ///
    /// let rect = Shape::rect(
    ///     [(0.0, 0.0), (100.0, 50.0)],
    ///     Stroke::new(2.0_f32, Color::BLACK),
    /// );
    /// ```
    pub fn rect(rect: [(f32, f32); 2], stroke: Stroke) -> Shape {
        let rect_shape = RectShape::new(rect, stroke);
        Shape::Rect(rect_shape)
    }

    /// Creates a rounded rectangle from its top-left and bottom-right coordinates.
    /// Each corner radius is specified by [`BorderRadii`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::{BorderRadii, Color, Shape, Stroke};
    ///
    /// let rounded_rect = Shape::rounded_rect(
    ///     [(0.0, 0.0), (100.0, 50.0)],
    ///     BorderRadii::new(10.0),
    ///     Stroke::new(1.5_f32, Color::BLACK),
    /// );
    /// ```
    pub fn rounded_rect(rect: [(f32, f32); 2], border_radii: BorderRadii, stroke: Stroke) -> Shape {
        let mut path_builder = lyon::path::Path::builder();
        let box2d = lyon::math::Box2D::new(rect[0].into(), rect[1].into());

        path_builder.add_rounded_rectangle(&box2d, &border_radii.into(), Winding::Positive);
        let path = path_builder.build();

        let path_shape = PathShape { path, stroke };
        Shape::Path(path_shape)
    }

    pub(crate) fn tessellate(
        &self,
        tessellator: &mut FillTessellator,
        shape_resources: &mut ShapeResources,
        tesselation_cache_key: Option<u64>,
    ) -> Arc<CachedTessellation> {
        match &self {
            Shape::Path(path_shape) => {
                path_shape.tessellate(tessellator, shape_resources, tesselation_cache_key)
            }
            Shape::Rect(rect_shape) => {
                if let Some(cache_key) = tesselation_cache_key {
                    if let Some(cached_tessellation) = shape_resources
                        .tessellation_cache
                        .get_tessellation(&cache_key)
                    {
                        return cached_tessellation;
                    }
                }

                let min_x = rect_shape.rect[0].0;
                let min_y = rect_shape.rect[0].1;
                let max_x = rect_shape.rect[1].0;
                let max_y = rect_shape.rect[1].1;

                // Compute UVs mapping the rectangle to [0,1] in local space
                let w = (max_x - min_x).max(1e-6);
                let h = (max_y - min_y).max(1e-6);
                let uv = |x: f32, y: f32| -> [f32; 2] { [(x - min_x) / w, (y - min_y) / h] };

                let quad = [
                    CustomVertex {
                        position: [min_x, min_y],
                        tex_coords: uv(min_x, min_y),
                        normal: [0.0, 0.0],
                        coverage: 1.0,
                    },
                    CustomVertex {
                        position: [max_x, min_y],
                        tex_coords: uv(max_x, min_y),
                        normal: [0.0, 0.0],
                        coverage: 1.0,
                    },
                    CustomVertex {
                        position: [max_x, max_y],
                        tex_coords: uv(max_x, max_y),
                        normal: [0.0, 0.0],
                        coverage: 1.0,
                    },
                    CustomVertex {
                        position: [min_x, max_y],
                        tex_coords: uv(min_x, max_y),
                        normal: [0.0, 0.0],
                        coverage: 1.0,
                    },
                ];
                let indices = [0u16, 1, 2, 0, 2, 3];
                let local_bounds = rect_shape.rect;

                let mut vertex_buffers = VertexBuffers::new();

                vertex_buffers.vertices.extend(quad);
                vertex_buffers.indices.extend(indices);

                generate_aa_fringe(
                    &mut vertex_buffers.vertices,
                    &mut vertex_buffers.indices,
                    &mut shape_resources.aa_fringe_scratch,
                );

                let tessellation = Arc::new(CachedTessellation {
                    vertex_buffers: Arc::new(vertex_buffers),
                    local_bounds,
                    texture_mapping_size: rect_size(local_bounds),
                });

                if let Some(tesselation_cache_key) = tesselation_cache_key {
                    shape_resources
                        .tessellation_cache
                        .insert_tessellation(tesselation_cache_key, Arc::clone(&tessellation));
                }

                tessellation
            }
        }
    }
}

impl From<PathShape> for Shape {
    fn from(value: PathShape) -> Self {
        Shape::Path(value)
    }
}

impl From<RectShape> for Shape {
    fn from(value: RectShape) -> Self {
        Shape::Rect(value)
    }
}

impl AsRef<Shape> for Shape {
    fn as_ref(&self) -> &Shape {
        self
    }
}

/// A rectangle's coordinates and stroke. Set its fill with [`ShapeDrawCommandOptions`].
///
/// [`Shape::rect`] constructs this and wraps it in [`Shape::Rect`].
#[derive(Debug, Clone)]
pub struct RectShape {
    /// Top-left and bottom-right coordinates.
    pub(crate) rect: [(f32, f32); 2],
    #[allow(unused)]
    pub(crate) stroke: Stroke,
}

impl RectShape {
    /// Creates a rectangle from its top-left and bottom-right coordinates.
    pub fn new(rect: [(f32, f32); 2], stroke: Stroke) -> Self {
        Self { rect, stroke }
    }
}

/// A custom path with stroke settings.
///
/// [`Shape::builder`] constructs this and wraps it in [`Shape::Path`].
#[derive(Clone, Debug)]
pub struct PathShape {
    pub(crate) path: lyon::path::Path,
    #[allow(unused)]
    pub(crate) stroke: Stroke,
}

struct VertexConverter;

impl FillVertexConstructor<CustomVertex> for VertexConverter {
    fn new_vertex(&mut self, vertex: FillVertex) -> CustomVertex {
        CustomVertex {
            position: vertex.position().to_array(),
            tex_coords: [0.0, 0.0],
            normal: [0.0, 0.0],
            coverage: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
struct BoundaryVertexKey {
    x_bits: u32,
    y_bits: u32,
}

impl BoundaryVertexKey {
    fn from_position(position: [f32; 2]) -> Self {
        Self {
            x_bits: normalized_float_bits(position[0]),
            y_bits: normalized_float_bits(position[1]),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct BoundaryEdgeKey {
    start: BoundaryVertexKey,
    end: BoundaryVertexKey,
}

impl BoundaryEdgeKey {
    fn new(start_position: [f32; 2], end_position: [f32; 2]) -> Self {
        let start = BoundaryVertexKey::from_position(start_position);
        let end = BoundaryVertexKey::from_position(end_position);
        if start <= end {
            Self { start, end }
        } else {
            Self {
                start: end,
                end: start,
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct BoundaryEdge {
    start_vertex_index: u16,
    end_vertex_index: u16,
    opposite_vertex_index: u16,
    triangle_index: usize,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct BoundaryCornerKey {
    vertex_key: BoundaryVertexKey,
    component_index: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct BoundaryCornerNormalData {
    accumulated_normal: [f32; 2],
    source_vertex_index: u16,
}

pub(crate) struct AaFringeScratch {
    edge_use_counts: AHashMap<BoundaryEdgeKey, (usize, BoundaryEdge)>,
    edge_owners: AHashMap<BoundaryEdgeKey, SmallVec<[usize; 2]>>,
    incident_triangles_by_vertex: AHashMap<BoundaryVertexKey, SmallVec<[usize; 4]>>,
    triangle_adjacency: AHashMap<(BoundaryVertexKey, usize), SmallVec<[usize; 4]>>,
    visited_triangles: AHashMap<usize, usize>,
    triangle_component_map: AHashMap<(usize, BoundaryVertexKey), usize>,
    boundary_corner_normals: AHashMap<BoundaryCornerKey, BoundaryCornerNormalData>,
    outer_vertex_indices: AHashMap<BoundaryCornerKey, u16>,
    boundary_edges: Vec<BoundaryEdge>,
    triangle_stack: Vec<usize>,
}

impl AaFringeScratch {
    pub(crate) fn new() -> Self {
        Self {
            edge_use_counts: AHashMap::new(),
            edge_owners: AHashMap::new(),
            incident_triangles_by_vertex: AHashMap::new(),
            triangle_adjacency: AHashMap::new(),
            visited_triangles: AHashMap::new(),
            triangle_component_map: AHashMap::new(),
            boundary_corner_normals: AHashMap::new(),
            outer_vertex_indices: AHashMap::new(),
            boundary_edges: Vec::new(),
            triangle_stack: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.edge_use_counts.clear();
        self.edge_owners.clear();
        self.incident_triangles_by_vertex.clear();
        self.triangle_adjacency.clear();
        self.visited_triangles.clear();
        self.triangle_component_map.clear();
        self.boundary_corner_normals.clear();
        self.outer_vertex_indices.clear();
        self.boundary_edges.clear();
        self.triangle_stack.clear();
    }
}

fn normalized_float_bits(value: f32) -> u32 {
    if value == 0.0 {
        0.0f32.to_bits()
    } else {
        value.to_bits()
    }
}

/// Clears and fills scratch storage with edge owners and incident triangles keyed by position.
///
/// Edges used by one triangle become boundary edges. Each records the triangle and its
/// opposite vertex so fringe generation can determine the outward direction.
fn build_boundary_data(vertices: &[CustomVertex], indices: &[u16], scratch: &mut AaFringeScratch) {
    scratch.clear();

    for (triangle_index, tri) in indices.as_chunks::<3>().0.iter().enumerate() {
        let a = tri[0];
        let b = tri[1];
        let c = tri[2];
        let vertex_keys = [a, b, c].map(|vertex_index| {
            BoundaryVertexKey::from_position(vertices[vertex_index as usize].position)
        });

        for &vertex_key in &vertex_keys {
            scratch
                .incident_triangles_by_vertex
                .entry(vertex_key)
                .or_default()
                .push(triangle_index);
        }

        for &(i, j, opp) in &[(a, b, c), (b, c, a), (c, a, b)] {
            let key =
                BoundaryEdgeKey::new(vertices[i as usize].position, vertices[j as usize].position);
            scratch
                .edge_use_counts
                .entry(key)
                .and_modify(|(count, _)| *count += 1)
                .or_insert((
                    1,
                    BoundaryEdge {
                        start_vertex_index: i,
                        end_vertex_index: j,
                        opposite_vertex_index: opp,
                        triangle_index,
                    },
                ));
            scratch
                .edge_owners
                .entry(key)
                .or_default()
                .push(triangle_index);
        }
    }

    scratch.boundary_edges.extend(
        scratch
            .edge_use_counts
            .values()
            .filter_map(|(count, boundary_edge)| (*count == 1).then_some(*boundary_edge)),
    );
}

fn build_triangle_component_map(scratch: &mut AaFringeScratch) {
    for (edge_key, owners) in &scratch.edge_owners {
        if owners.len() < 2 {
            continue;
        }

        for &vertex_key in &[edge_key.start, edge_key.end] {
            for (owner_index, &owner_triangle_index) in owners.iter().enumerate() {
                for &other_triangle_index in &owners[owner_index + 1..] {
                    scratch
                        .triangle_adjacency
                        .entry((vertex_key, owner_triangle_index))
                        .or_default()
                        .push(other_triangle_index);
                    scratch
                        .triangle_adjacency
                        .entry((vertex_key, other_triangle_index))
                        .or_default()
                        .push(owner_triangle_index);
                }
            }
        }
    }

    for (&vertex_key, incident_triangles) in &scratch.incident_triangles_by_vertex {
        scratch.visited_triangles.clear();
        let mut component_index = 0usize;

        for &triangle_index in incident_triangles {
            if scratch.visited_triangles.contains_key(&triangle_index) {
                continue;
            }

            scratch.triangle_stack.push(triangle_index);
            while let Some(current_triangle_index) = scratch.triangle_stack.pop() {
                if scratch
                    .visited_triangles
                    .insert(current_triangle_index, component_index)
                    .is_some()
                {
                    continue;
                }

                if let Some(neighbors) = scratch
                    .triangle_adjacency
                    .get(&(vertex_key, current_triangle_index))
                {
                    for &neighbor_triangle_index in neighbors {
                        if !scratch
                            .visited_triangles
                            .contains_key(&neighbor_triangle_index)
                        {
                            scratch.triangle_stack.push(neighbor_triangle_index);
                        }
                    }
                }
            }

            component_index += 1;
        }

        for (&triangle_index, &component_index) in &scratch.visited_triangles {
            scratch
                .triangle_component_map
                .insert((triangle_index, vertex_key), component_index);
        }
    }
}

/// Adds two antialiasing triangles per boundary edge, fading coverage from 1 to 0.
/// Outer vertices keep the boundary positions. The vertex shader uses their normals
/// to apply the screen-space offset.
fn generate_aa_fringe(
    vertices: &mut Vec<CustomVertex>,
    indices: &mut Vec<u16>,
    scratch: &mut AaFringeScratch,
) {
    build_boundary_data(vertices, indices, scratch);

    if scratch.boundary_edges.is_empty() {
        return;
    }

    build_triangle_component_map(scratch);

    // Average outward normals at each boundary corner.

    for boundary_edge in &scratch.boundary_edges {
        let pa = vertices[boundary_edge.start_vertex_index as usize].position;
        let pb = vertices[boundary_edge.end_vertex_index as usize].position;
        let po = vertices[boundary_edge.opposite_vertex_index as usize].position;

        let dx = pb[0] - pa[0];
        let dy = pb[1] - pa[1];
        let edge_len = (dx * dx + dy * dy).sqrt();
        if edge_len < 1e-10 {
            continue;
        }

        let n1 = [-dy / edge_len, dx / edge_len];
        let n2 = [dy / edge_len, -dx / edge_len];

        let to_opp = [po[0] - pa[0], po[1] - pa[1]];
        let dot1 = n1[0] * to_opp[0] + n1[1] * to_opp[1];
        let outward = if dot1 < 0.0 { n1 } else { n2 };

        for &vertex_index in &[
            boundary_edge.start_vertex_index,
            boundary_edge.end_vertex_index,
        ] {
            let vertex_key =
                BoundaryVertexKey::from_position(vertices[vertex_index as usize].position);
            let component_index = *scratch
                .triangle_component_map
                .get(&(boundary_edge.triangle_index, vertex_key))
                .unwrap_or_else(|| {
                    panic!(
                    "missing triangle component mapping for triangle {} and boundary vertex {:?}",
                    boundary_edge.triangle_index,
                    vertex_key
                )
                });
            let entry = scratch
                .boundary_corner_normals
                .entry(BoundaryCornerKey {
                    vertex_key,
                    component_index,
                })
                .or_insert(BoundaryCornerNormalData {
                    accumulated_normal: [0.0, 0.0],
                    source_vertex_index: vertex_index,
                });
            entry.accumulated_normal[0] += outward[0];
            entry.accumulated_normal[1] += outward[1];
        }
    }

    for boundary_corner_normal in scratch.boundary_corner_normals.values_mut() {
        let len = (boundary_corner_normal.accumulated_normal[0]
            * boundary_corner_normal.accumulated_normal[0]
            + boundary_corner_normal.accumulated_normal[1]
                * boundary_corner_normal.accumulated_normal[1])
            .sqrt();
        if len > 1e-10 {
            boundary_corner_normal.accumulated_normal[0] /= len;
            boundary_corner_normal.accumulated_normal[1] /= len;
        }
    }

    // Duplicate boundary vertices with zero coverage for the outer fringe.

    vertices.reserve(scratch.boundary_corner_normals.len());
    indices.reserve(scratch.boundary_edges.len() * 6);

    for (&boundary_corner_key, boundary_corner_normal) in &scratch.boundary_corner_normals {
        let source_vertex = &vertices[boundary_corner_normal.source_vertex_index as usize];
        let outer_vertex = CustomVertex {
            position: source_vertex.position,
            tex_coords: source_vertex.tex_coords,
            normal: boundary_corner_normal.accumulated_normal,
            coverage: 0.0,
        };
        let new_idx = vertices.len() as u16;
        vertices.push(outer_vertex);
        scratch
            .outer_vertex_indices
            .insert(boundary_corner_key, new_idx);
    }

    // Join each boundary edge to its outer vertices with two triangles.

    for boundary_edge in &scratch.boundary_edges {
        let start_vertex_key = BoundaryVertexKey::from_position(
            vertices[boundary_edge.start_vertex_index as usize].position,
        );
        let end_vertex_key = BoundaryVertexKey::from_position(
            vertices[boundary_edge.end_vertex_index as usize].position,
        );
        let pa = vertices[boundary_edge.start_vertex_index as usize].position;
        let pb = vertices[boundary_edge.end_vertex_index as usize].position;
        let po = vertices[boundary_edge.opposite_vertex_index as usize].position;

        let dx = pb[0] - pa[0];
        let dy = pb[1] - pa[1];
        let edge_len = (dx * dx + dy * dy).sqrt();
        if edge_len < 1e-10 {
            continue;
        }

        let start_component_index = *scratch
            .triangle_component_map
            .get(&(boundary_edge.triangle_index, start_vertex_key))
            .unwrap_or_else(|| {
                panic!(
                    "missing triangle component mapping for triangle {} and boundary vertex {:?}",
                    boundary_edge.triangle_index, start_vertex_key
                )
            });
        let end_component_index = *scratch
            .triangle_component_map
            .get(&(boundary_edge.triangle_index, end_vertex_key))
            .unwrap_or_else(|| {
                panic!(
                    "missing triangle component mapping for triangle {} and boundary vertex {:?}",
                    boundary_edge.triangle_index, end_vertex_key
                )
            });

        let start_boundary_corner_key = BoundaryCornerKey {
            vertex_key: start_vertex_key,
            component_index: start_component_index,
        };
        let start_outer_vertex_index = match scratch
            .outer_vertex_indices
            .get(&start_boundary_corner_key)
        {
            Some(&idx) => idx,
            None => {
                debug_assert!(
                    false,
                    "missing outer vertex index for {:?} (start_vertex_key: {:?}, start_component_index: {})",
                    start_boundary_corner_key,
                    start_vertex_key,
                    start_component_index
                );
                continue;
            }
        };
        let end_boundary_corner_key = BoundaryCornerKey {
            vertex_key: end_vertex_key,
            component_index: end_component_index,
        };
        let end_outer_vertex_index = match scratch
            .outer_vertex_indices
            .get(&end_boundary_corner_key)
        {
            Some(&idx) => idx,
            None => {
                debug_assert!(
                    false,
                    "missing outer vertex index for {:?} (end_vertex_key: {:?}, end_component_index: {})",
                    end_boundary_corner_key,
                    end_vertex_key,
                    end_component_index
                );
                continue;
            }
        };

        let cross = (pb[0] - pa[0]) * (po[1] - pa[1]) - (pb[1] - pa[1]) * (po[0] - pa[0]);

        if cross >= 0.0 {
            indices.push(boundary_edge.start_vertex_index);
            indices.push(boundary_edge.end_vertex_index);
            indices.push(start_outer_vertex_index);

            indices.push(boundary_edge.end_vertex_index);
            indices.push(end_outer_vertex_index);
            indices.push(start_outer_vertex_index);
        } else {
            indices.push(boundary_edge.start_vertex_index);
            indices.push(start_outer_vertex_index);
            indices.push(boundary_edge.end_vertex_index);

            indices.push(boundary_edge.end_vertex_index);
            indices.push(start_outer_vertex_index);
            indices.push(end_outer_vertex_index);
        }
    }
}

impl PathShape {
    /// Uses an existing Lyon path as the shape's geometry.
    pub fn new(path: lyon::path::Path, stroke: Stroke) -> Self {
        Self { path, stroke }
    }

    /// Returns shared geometry and bounds, reusing the tessellation cache when a key is given.
    pub(crate) fn tessellate(
        &self,
        tessellator: &mut FillTessellator,
        shape_resources: &mut ShapeResources,
        tesselation_cache_key: Option<u64>,
    ) -> Arc<CachedTessellation> {
        if let Some(cache_key) = tesselation_cache_key {
            if let Some(cached_tessellation) = shape_resources
                .tessellation_cache
                .get_tessellation(&cache_key)
            {
                return cached_tessellation;
            }
        }

        let mut buffers = VertexBuffers::new();
        let local_bounds = self.tessellate_into_buffers(
            &mut buffers,
            tessellator,
            &mut shape_resources.aa_fringe_scratch,
        );

        #[allow(clippy::manual_is_multiple_of)]
        let needs_index_padding = buffers.indices.len() % 2 != 0;
        if needs_index_padding {
            buffers.indices.push(0);
        }

        let tessellation = Arc::new(CachedTessellation {
            vertex_buffers: Arc::new(buffers),
            local_bounds,
            texture_mapping_size: rect_size(local_bounds),
        });

        if let Some(cache_key) = tesselation_cache_key {
            shape_resources
                .tessellation_cache
                .insert_tessellation(cache_key, Arc::clone(&tessellation));
        }

        tessellation
    }

    fn tessellate_into_buffers(
        &self,
        buffers: &mut VertexBuffers<CustomVertex, u16>,
        tessellator: &mut FillTessellator,
        aa_fringe_scratch: &mut AaFringeScratch,
    ) -> [(f32, f32); 2] {
        let options = FillOptions::default();

        tessellator
            .tessellate_path(
                &self.path,
                &options,
                &mut BuffersBuilder::new(buffers, VertexConverter),
            )
            .unwrap();

        let local_bounds = compute_vertex_bounds(&buffers.vertices);

        // Generate UVs for the tessellated path using its axis-aligned bounding box in local space
        if !buffers.vertices.is_empty() {
            let w = (local_bounds[1].0 - local_bounds[0].0).max(1e-6);
            let h = (local_bounds[1].1 - local_bounds[0].1).max(1e-6);
            for v in buffers.vertices.iter_mut() {
                let u = (v.position[0] - local_bounds[0].0) / w;
                let vcoord = (v.position[1] - local_bounds[0].1) / h;
                v.tex_coords = [u, vcoord];
            }
        }

        // Generate AA fringe geometry after UVs are computed so fringe vertices
        // inherit the correct tex_coords from their source boundary vertices.
        generate_aa_fringe(
            &mut buffers.vertices,
            &mut buffers.indices,
            aa_fringe_scratch,
        );

        local_bounds
    }
}

/// Controls how bound textures are fit into a shape's local texture-coordinate space.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum ShapeTextureFitMode {
    /// Normalize the shape bounds to `[0, 1]` and stretch the texture to cover them fully.
    #[default]
    Stretch,
    /// Preserve the texture's aspect ratio, center it, and fill the shape bounds by cropping.
    Cover,
    /// Preserve the texture's aspect ratio and center it inside the shape bounds.
    Contain,
    /// Treat one texture texel as one physical pixel before the shape transform is applied.
    /// The texture moves and transforms with the shape.
    /// Outside the texture footprint, this layer contributes no color.
    /// The fill and other texture layer remain visible.
    OriginalSize,
}

fn compute_texture_uv_scale_from_dimensions(
    texture_mapping_size: [f32; 2],
    texture_dimensions: (u32, u32),
    scale_factor: f64,
) -> [f32; 2] {
    [
        texture_mapping_size[0] * scale_factor as f32 / texture_dimensions.0.max(1) as f32,
        texture_mapping_size[1] * scale_factor as f32 / texture_dimensions.1.max(1) as f32,
    ]
}

impl ShapeTextureFitMode {
    pub(crate) fn compute_uv_transform(
        self,
        texture_mapping_size: [f32; 2],
        texture_dimensions: (u32, u32),
        scale_factor: f64,
    ) -> TextureUvTransform {
        let original_size_uv_scale = compute_texture_uv_scale_from_dimensions(
            texture_mapping_size,
            texture_dimensions,
            scale_factor,
        );

        let normalization_factor = match self {
            ShapeTextureFitMode::Stretch => return TextureUvTransform::IDENTITY,
            ShapeTextureFitMode::Cover => {
                f32::max(original_size_uv_scale[0], original_size_uv_scale[1])
            }
            ShapeTextureFitMode::Contain => {
                f32::min(original_size_uv_scale[0], original_size_uv_scale[1])
            }
            ShapeTextureFitMode::OriginalSize => {
                return TextureUvTransform {
                    scale: original_size_uv_scale,
                    offset: [0.0, 0.0],
                };
            }
        };

        if !normalization_factor.is_finite() || normalization_factor <= f32::EPSILON {
            return TextureUvTransform::IDENTITY;
        }

        let scale = [
            original_size_uv_scale[0] / normalization_factor,
            original_size_uv_scale[1] / normalization_factor,
        ];

        TextureUvTransform {
            scale,
            offset: [(1.0 - scale[0]) / 2.0, (1.0 - scale[1]) / 2.0],
        }
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct ShapeTextureOptions {
    pub texture_id: Option<u64>,
    pub fit_mode: ShapeTextureFitMode,
}

impl ShapeTextureOptions {
    pub fn new(texture_id: u64) -> Self {
        Self {
            texture_id: Some(texture_id),
            fit_mode: ShapeTextureFitMode::Stretch,
        }
    }

    pub fn fit_mode(mut self, fit_mode: ShapeTextureFitMode) -> Self {
        self.fit_mode = fit_mode;
        self
    }
}

#[derive(Clone, Debug)]
pub struct ShapeDrawCommandOptions {
    pub transform: Option<InstanceTransform>,
    pub clips_children: bool,
    pub background_texture: ShapeTextureOptions,
    pub foreground_texture: ShapeTextureOptions,
    pub fill: Option<Fill>,
}

impl Default for ShapeDrawCommandOptions {
    fn default() -> Self {
        Self {
            transform: None,
            clips_children: true,
            background_texture: ShapeTextureOptions::default(),
            foreground_texture: ShapeTextureOptions::default(),
            fill: None,
        }
    }
}

impl ShapeDrawCommandOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn transform(mut self, transform: InstanceTransform) -> Self {
        self.transform = Some(transform);
        self
    }

    pub fn clips_children(mut self, clips_children: bool) -> Self {
        self.clips_children = clips_children;
        self
    }

    pub fn background_texture(mut self, background_texture: ShapeTextureOptions) -> Self {
        self.background_texture = background_texture;
        self
    }

    pub fn foreground_texture(mut self, foreground_texture: ShapeTextureOptions) -> Self {
        self.foreground_texture = foreground_texture;
        self
    }

    pub fn background_texture_id(mut self, background_texture_id: u64) -> Self {
        self.background_texture.texture_id = Some(background_texture_id);
        self
    }

    pub fn foreground_texture_id(mut self, foreground_texture_id: u64) -> Self {
        self.foreground_texture.texture_id = Some(foreground_texture_id);
        self
    }

    pub fn texture_fit_mode(mut self, texture_fit_mode: ShapeTextureFitMode) -> Self {
        self.background_texture.fit_mode = texture_fit_mode;
        self.foreground_texture.fit_mode = texture_fit_mode;
        self
    }

    pub fn background_texture_fit_mode(
        mut self,
        background_texture_fit_mode: ShapeTextureFitMode,
    ) -> Self {
        self.background_texture.fit_mode = background_texture_fit_mode;
        self
    }

    pub fn foreground_texture_fit_mode(
        mut self,
        foreground_texture_fit_mode: ShapeTextureFitMode,
    ) -> Self {
        self.foreground_texture.fit_mode = foreground_texture_fit_mode;
        self
    }

    pub fn fill(mut self, fill: Fill) -> Self {
        self.fill = Some(fill);
        self
    }

    pub fn color(mut self, color: Color) -> Self {
        self.fill = Some(Fill::Solid(color));
        self
    }
}

/// Builds a shape's path and stroke through method chaining.
///
/// Assign a fill through [`ShapeDrawCommandOptions`] when queueing the shape.
/// An unset fill renders as transparent. [`Shape::builder`] also creates this builder.
///
/// # Examples
///
/// ```rust
/// use grafo::{Color, ShapeBuilder, ShapeDrawCommandOptions, Stroke};
///
/// # fn example(renderer: &mut grafo::Renderer<'_>) {
/// let custom_shape = ShapeBuilder::new()
///     .stroke(Stroke::new(3.0_f32, Color::BLACK))
///     .begin((0.0, 0.0))
///     .line_to((50.0, 10.0))
///     .line_to((50.0, 50.0))
///     .close()
///     .build();
/// renderer.add_shape(
///     custom_shape,
///     None,
///     None,
///     ShapeDrawCommandOptions::new().color(Color::rgb(0, 128, 255)),
/// ).unwrap();
/// # }
/// ```
#[derive(Clone)]
pub struct ShapeBuilder {
    stroke: Stroke,
    path_builder: lyon::path::Builder,
}

impl Default for ShapeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ShapeBuilder {
    /// Starts an empty path with a black stroke of width 1.0.
    pub fn new() -> Self {
        Self {
            stroke: Stroke::new(1.0_f32, Color::rgb(0, 0, 0)),
            path_builder: lyon::path::Path::builder(),
        }
    }

    /// Sets the stroke properties of the shape.
    pub fn stroke(mut self, stroke: Stroke) -> Self {
        self.stroke = stroke;
        self
    }

    /// Starts a new subpath at `point`.
    pub fn begin(mut self, point: (f32, f32)) -> Self {
        self.path_builder.begin(point.into());
        self
    }

    /// Draws a line from the current point to `point`.
    pub fn line_to(mut self, point: (f32, f32)) -> Self {
        self.path_builder.line_to(point.into());
        self
    }

    /// Draws a cubic Bezier curve to `to`, using `ctrl` and `ctrl2` as the first
    /// and second control points.
    pub fn cubic_bezier_to(mut self, ctrl: (f32, f32), ctrl2: (f32, f32), to: (f32, f32)) -> Self {
        self.path_builder
            .cubic_bezier_to(ctrl.into(), ctrl2.into(), to.into());
        self
    }

    /// Draws a quadratic Bezier curve to `to`, using `ctrl` as the control point.
    pub fn quadratic_bezier_to(mut self, ctrl: (f32, f32), to: (f32, f32)) -> Self {
        self.path_builder
            .quadratic_bezier_to(ctrl.into(), to.into());
        self
    }

    /// Closes the current subpath with a line back to its starting point.
    pub fn close(mut self) -> Self {
        self.path_builder.close();
        self
    }

    /// Builds the [`Shape`] from the accumulated path and stroke.
    pub fn build(self) -> Shape {
        let path = self.path_builder.build();
        Shape::Path(PathShape {
            path,
            stroke: self.stroke,
        })
    }
}

impl From<ShapeBuilder> for Shape {
    fn from(value: ShapeBuilder) -> Self {
        value.build()
    }
}

/// The radius of each corner of a rounded rectangle.
///
/// # Examples
///
/// ```rust
/// use grafo::BorderRadii;
///
/// let uniform_radii = BorderRadii::new(10.0);
///
/// let custom_radii = BorderRadii {
///     top_left: 5.0,
///     top_right: 10.0,
///     bottom_left: 15.0,
///     bottom_right: 20.0,
/// };
/// ```
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug, Default)]
pub struct BorderRadii {
    pub top_left: f32,
    pub top_right: f32,
    pub bottom_left: f32,
    pub bottom_right: f32,
}

impl BorderRadii {
    /// Sets every corner to the absolute value of `radius`.
    pub fn new(radius: f32) -> Self {
        let r = radius.abs();
        BorderRadii {
            top_left: r,
            top_right: r,
            bottom_left: r,
            bottom_right: r,
        }
    }
}

impl core::fmt::Display for BorderRadii {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "BorderRadii({}, {}, {}, {})",
            self.top_left, self.top_right, self.bottom_left, self.bottom_right
        )
    }
}

impl From<BorderRadii> for lyon::path::builder::BorderRadii {
    fn from(val: BorderRadii) -> Self {
        lyon::path::builder::BorderRadii {
            top_left: val.top_left,
            top_right: val.top_right,
            bottom_left: val.bottom_left,
            bottom_right: val.bottom_right,
        }
    }
}

#[cfg(test)]
mod tests;
