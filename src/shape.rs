//! The `shape` module provides structures and methods for creating and managing graphical shapes
//! within the Grafo library. It supports both simple and complex shapes, including rectangles and
//! custom paths with stroke properties. Fill color is per-instance and set via the renderer.
//!
//! # Examples
//!
//! Creating and using different shapes:
//!
//! ```rust
//! use grafo::Stroke;
//! use grafo::{Shape, ShapeBuilder, BorderRadii};
//! use grafo::Color;
//!
//! // Create a simple rectangle
//! let rect = Shape::rect(
//!     [(0.0, 0.0), (100.0, 50.0)],
//!     Stroke::new(2.0, Color::BLACK), // Black stroke with width 2.0
//! );
//!
//! // Create a rounded rectangle
//! let rounded_rect = Shape::rounded_rect(
//!     [(0.0, 0.0), (100.0, 50.0)],
//!     BorderRadii::new(10.0),
//!     Stroke::new(1.5, Color::BLACK), // Black stroke with width 1.5
//! );
//!
//! // Build a custom shape using ShapeBuilder
//! let custom_shape = Shape::builder()
//!     .stroke(Stroke::new(3.0, Color::BLACK)) // Black stroke with width 3.0
//!     .begin((0.0, 0.0))
//!     .line_to((50.0, 10.0))
//!     .line_to((50.0, 50.0))
//!     .close()
//!     .build();
//! ```

use crate::util::PoolManager;
use crate::vertex::{CustomVertex, InstanceTransform};
use crate::{Color, Stroke};
use ahash::AHashMap;
use lyon::lyon_tessellation::{
    BuffersBuilder, FillOptions, FillTessellator, FillVertex, VertexBuffers,
};
use lyon::path::Winding;
use lyon::tessellation::FillVertexConstructor;
use smallvec::SmallVec;
use std::sync::Arc;

pub(crate) struct CachedShape {
    pub vertex_buffers: Arc<VertexBuffers<CustomVertex, u16>>,
    /// Whether the original shape was an axis-aligned rectangle.
    pub(crate) is_rect: bool,
    /// The local-space bounding rect when `is_rect` is true.
    pub(crate) rect_bounds: Option<[(f32, f32); 2]>,
}

pub(crate) enum TessellatedGeometry {
    Owned(VertexBuffers<CustomVertex, u16>),
    Shared(Arc<VertexBuffers<CustomVertex, u16>>),
}

impl TessellatedGeometry {
    pub(crate) fn vertices(&self) -> &[CustomVertex] {
        match self {
            Self::Owned(vertex_buffers) => &vertex_buffers.vertices,
            Self::Shared(vertex_buffers) => &vertex_buffers.vertices,
        }
    }

    pub(crate) fn indices(&self) -> &[u16] {
        match self {
            Self::Owned(vertex_buffers) => &vertex_buffers.indices,
            Self::Shared(vertex_buffers) => &vertex_buffers.indices,
        }
    }

    /// Returns the vertex buffers if they are owned, or `None` if they are shared (cached).
    pub(crate) fn into_owned(self) -> Option<VertexBuffers<CustomVertex, u16>> {
        match self {
            Self::Owned(vertex_buffers) => Some(vertex_buffers),
            Self::Shared(_) => None,
        }
    }
}

impl CachedShape {
    /// Creates a new `CachedShape` with the specified shape and depth.
    /// Note that tessellator_cache_key is different from the shape cache key; a Shape cache key is
    /// the shape identifier, while tesselator_cache_key is used to cache the tessellation of the
    /// shape and should be based on the shape properties, and not the shape identifier
    pub fn new(
        shape: &Shape,
        tessellator: &mut FillTessellator,
        pool: &mut PoolManager,
        tessellator_cache_key: Option<u64>,
    ) -> Self {
        let (is_rect, rect_bounds) = match shape {
            Shape::Rect(r) => (true, Some(r.rect)),
            _ => (false, None),
        };
        let vertices = match shape.tessellate(tessellator, pool, tessellator_cache_key) {
            TessellatedGeometry::Owned(vertex_buffers) => Arc::new(vertex_buffers),
            TessellatedGeometry::Shared(vertex_buffers) => vertex_buffers,
        };
        Self {
            vertex_buffers: vertices,
            is_rect,
            rect_bounds,
        }
    }
}
/// Represents a graphical shape, which can be either a custom path or a simple rectangle.
///
/// # Variants
///
/// - `Path(PathShape)`: A custom path shape defined using Bézier curves and lines.
/// - `Rect(RectShape)`: A simple rectangular shape with optional rounded corners.
///
/// # Examples
///
/// ```rust
/// use grafo::Stroke;
/// use grafo::{Shape, BorderRadii};
/// use grafo::Color;
///
/// // Create a simple rectangle
/// let rect = Shape::rect(
///     [(0.0, 0.0), (100.0, 50.0)],
///     Stroke::new(2.0, Color::BLACK), // Black stroke with width 2.0
/// );
///
/// // Create a custom path shape
/// let custom_path = Shape::builder()
///     .stroke(Stroke::new(1.0, Color::BLACK))
///     .begin((0.0, 0.0))
///     .line_to((50.0, 10.0))
///     .line_to((50.0, 50.0))
///     .close()
///     .build();
/// ```
#[derive(Debug, Clone)]
pub enum Shape {
    /// A custom path shape defined using Bézier curves and lines.
    Path(PathShape),
    /// A simple rectangular shape.
    Rect(RectShape),
}

impl Shape {
    /// Creates a new [`ShapeBuilder`] for constructing complex shapes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::Shape;
    ///
    /// let builder = Shape::builder();
    /// ```
    pub fn builder() -> ShapeBuilder {
        ShapeBuilder::new()
    }

    /// Creates a simple rectangle shape with the specified coordinates and stroke.
    ///
    /// # Parameters
    ///
    /// - `rect`: An array containing two tuples representing the top-left and bottom-right
    ///   coordinates of the rectangle.
    /// - `stroke`: The stroke properties of the rectangle.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::Color;
    /// use grafo::Stroke;
    /// use grafo::Shape;
    ///
    /// let rect = Shape::rect(
    ///     [(0.0, 0.0), (100.0, 50.0)],
    ///     Stroke::new(2.0, Color::BLACK), // Black stroke with width 2.0
    /// );
    /// ```
    pub fn rect(rect: [(f32, f32); 2], stroke: Stroke) -> Shape {
        let rect_shape = RectShape::new(rect, stroke);
        Shape::Rect(rect_shape)
    }

    /// Creates a rectangle shape with rounded corners.
    ///
    /// # Parameters
    ///
    /// - `rect`: An array containing two tuples representing the top-left and bottom-right
    ///   coordinates of the rectangle.
    /// - `border_radii`: The radii for each corner of the rectangle.
    /// - `stroke`: The stroke properties of the rectangle.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::Color;
    /// use grafo::Stroke;
    /// use grafo::{Shape, BorderRadii};
    ///
    /// let rounded_rect = Shape::rounded_rect(
    ///     [(0.0, 0.0), (100.0, 50.0)],
    ///     BorderRadii::new(10.0),
    ///     Stroke::new(1.5, Color::BLACK), // Black stroke with width 1.5
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
        buffers_pool: &mut PoolManager,
        tesselation_cache_key: Option<u64>,
    ) -> TessellatedGeometry {
        match &self {
            Shape::Path(path_shape) => {
                path_shape.tessellate(tessellator, buffers_pool, tesselation_cache_key)
            }
            Shape::Rect(rect_shape) => {
                let min_width = rect_shape.rect[0].0;
                let min_height = rect_shape.rect[0].1;
                let max_width = rect_shape.rect[1].0;
                let max_height = rect_shape.rect[1].1;

                // Compute UVs mapping the rectangle to [0,1] in local space
                let w = (max_width - min_width).max(1e-6);
                let h = (max_height - min_height).max(1e-6);
                let uv =
                    |x: f32, y: f32| -> [f32; 2] { [(x - min_width) / w, (y - min_height) / h] };

                let quad = [
                    CustomVertex {
                        position: [min_width, min_height],
                        tex_coords: uv(min_width, min_height),
                        normal: [0.0, 0.0],
                        coverage: 1.0,
                    },
                    CustomVertex {
                        position: [max_width, min_height],
                        tex_coords: uv(max_width, min_height),
                        normal: [0.0, 0.0],
                        coverage: 1.0,
                    },
                    CustomVertex {
                        position: [max_width, max_height],
                        tex_coords: uv(max_width, max_height),
                        normal: [0.0, 0.0],
                        coverage: 1.0,
                    },
                    CustomVertex {
                        position: [min_width, max_height],
                        tex_coords: uv(min_width, max_height),
                        normal: [0.0, 0.0],
                        coverage: 1.0,
                    },
                ];
                let indices = [0u16, 1, 2, 0, 2, 3];

                let mut vertex_buffers = buffers_pool.lyon_vertex_buffers_pool.get_vertex_buffers();

                vertex_buffers.vertices.extend(quad);
                vertex_buffers.indices.extend(indices);

                // Generate AA fringe geometry for the rect
                generate_aa_fringe(
                    &mut vertex_buffers.vertices,
                    &mut vertex_buffers.indices,
                    &mut buffers_pool.aa_fringe_scratch,
                );

                TessellatedGeometry::Owned(vertex_buffers)
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

/// Represents a simple rectangular shape with a fill color and stroke.
///
/// You typically do not need to use `RectShape` directly; instead, use the [`Shape::rect`] method.
///
/// # Fields
///
/// - `rect`: An array containing two tuples representing the top-left and bottom-right
///   coordinates of the rectangle.
/// - `stroke`: The stroke properties of the rectangle.
///
/// # Examples
///
/// ```rust
/// use grafo::RectShape;
/// use grafo::Stroke;
/// use grafo::Color;
///
/// let rect_shape = RectShape::new(
///     [(0.0, 0.0), (100.0, 50.0)],
///     Stroke::new(2.0, Color::BLACK), // Black stroke with width 2.0
/// );
/// ```
#[derive(Debug, Clone)]
pub struct RectShape {
    /// An array containing two tuples representing the top-left and bottom-right coordinates
    /// of the rectangle.
    pub(crate) rect: [(f32, f32); 2],
    /// The stroke properties of the rectangle.
    #[allow(unused)]
    pub(crate) stroke: Stroke,
}

impl RectShape {
    /// Creates a new `RectShape` with the specified coordinates and stroke.
    ///
    /// # Parameters
    ///
    /// - `rect`: An array containing two tuples representing the top-left and bottom-right
    ///   coordinates of the rectangle.
    /// - `stroke`: The stroke properties of the rectangle.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::RectShape;
    /// use grafo::Stroke;
    /// use grafo::Color;
    ///
    /// let rect_shape = RectShape::new(
    ///     [(0.0, 0.0), (100.0, 50.0)],
    ///     Stroke::new(2.0, Color::BLACK), // Black stroke with width 2.0
    /// );
    /// ```
    pub fn new(rect: [(f32, f32); 2], stroke: Stroke) -> Self {
        Self { rect, stroke }
    }
}

/// Represents a custom path shape with stroke.
///
/// You typically do not need to use `PathShape` directly; instead, use the [`Shape::builder`]
/// method to construct complex shapes.
///
/// # Fields
///
/// - `path`: The geometric path defining the shape.
/// - `stroke`: The stroke properties of the shape.
///
/// # Examples
///
/// ```rust
/// use grafo::{Shape, PathShape};
/// use grafo::Stroke;
/// use grafo::Color;
///
/// // Replace this with your own path
/// let path = lyon::path::Path::builder().build();
///
/// let path_shape = PathShape::new(
///     path,
///     Stroke::new(1.0, Color::BLACK), // Black stroke with width 1.0
/// );
///
/// let shape = Shape::Path(path_shape);
/// ```
#[derive(Clone, Debug)]
pub struct PathShape {
    /// The geometric path defining the shape.
    pub(crate) path: lyon::path::Path,
    /// The stroke properties of the shape.
    #[allow(unused)]
    pub(crate) stroke: Stroke,
}

struct VertexConverter {}

impl VertexConverter {
    fn new() -> Self {
        Self {}
    }
}

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

#[derive(Clone, Copy, Debug, PartialEq)]
struct TriangleData {
    vertex_keys: [BoundaryVertexKey; 3],
}

pub(crate) struct AaFringeScratch {
    edge_use_counts: AHashMap<BoundaryEdgeKey, (usize, BoundaryEdge)>,
    edge_owners: AHashMap<BoundaryEdgeKey, SmallVec<[usize; 2]>>,
    incident_triangles_by_vertex: AHashMap<BoundaryVertexKey, SmallVec<[usize; 4]>>,
    triangle_adjacency: AHashMap<(BoundaryVertexKey, usize), SmallVec<[usize; 4]>>,
    triangle_component_map: AHashMap<(usize, BoundaryVertexKey), usize>,
    boundary_corner_normals: AHashMap<BoundaryCornerKey, BoundaryCornerNormalData>,
    outer_vertex_indices: AHashMap<BoundaryCornerKey, u16>,
    triangles: Vec<TriangleData>,
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
            triangle_component_map: AHashMap::new(),
            boundary_corner_normals: AHashMap::new(),
            outer_vertex_indices: AHashMap::new(),
            triangles: Vec::new(),
            boundary_edges: Vec::new(),
            triangle_stack: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.edge_use_counts.clear();
        self.edge_owners.clear();
        self.incident_triangles_by_vertex.clear();
        self.triangle_adjacency.clear();
        self.triangle_component_map.clear();
        self.boundary_corner_normals.clear();
        self.outer_vertex_indices.clear();
        self.triangles.clear();
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

// ---------------------------------------------------------------------------
// Anti-Aliasing: Inflated-Geometry Fringe Generation
// ---------------------------------------------------------------------------

/// Identifies boundary edges (edges belonging to only one triangle) from a triangle index buffer.
///
/// Returns a list of `(vertex_a, vertex_b, opposite_vertex)` tuples. The `opposite_vertex` is
/// the third vertex of the triangle that owns the edge — it is used to determine which side
/// of the edge faces outward (away from the triangle interior).
fn build_boundary_data(vertices: &[CustomVertex], indices: &[u16], scratch: &mut AaFringeScratch) {
    scratch.clear();

    for (triangle_index, tri) in indices.chunks_exact(3).enumerate() {
        let a = tri[0];
        let b = tri[1];
        let c = tri[2];
        let vertex_keys = [a, b, c].map(|vertex_index| {
            BoundaryVertexKey::from_position(vertices[vertex_index as usize].position)
        });

        scratch.triangles.push(TriangleData { vertex_keys });

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
        let mut visited_triangles = AHashMap::new();
        let mut component_index = 0usize;

        for &triangle_index in incident_triangles {
            if visited_triangles.contains_key(&triangle_index) {
                continue;
            }

            scratch.triangle_stack.push(triangle_index);
            while let Some(current_triangle_index) = scratch.triangle_stack.pop() {
                if visited_triangles
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
                        if !visited_triangles.contains_key(&neighbor_triangle_index) {
                            scratch.triangle_stack.push(neighbor_triangle_index);
                        }
                    }
                }
            }

            component_index += 1;
        }

        for (triangle_index, component_index) in visited_triangles {
            scratch
                .triangle_component_map
                .insert((triangle_index, vertex_key), component_index);
        }
    }
}

/// Generates a thin fringe of antialiasing triangles around shape boundaries.
///
/// For each boundary edge, two triangles are added, forming a quad that fades from
/// `coverage = 1.0` (at the original boundary) to `coverage = 0.0` (at the outer fringe).
/// The actual screen-space offset is computed in the vertex shader, so the fringe positions
/// in the buffer are identical to the source boundary vertices — only the `normal` and
/// `coverage` fields differ.
#[cfg(test)]
fn find_boundary_edges<'a>(
    vertices: &[CustomVertex],
    indices: &[u16],
    scratch: &'a mut AaFringeScratch,
) -> &'a [BoundaryEdge] {
    build_boundary_data(vertices, indices, scratch);
    &scratch.boundary_edges
}

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

    // --- Step 1: Compute per-boundary-vertex averaged outward (miter) normals ---

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
                .unwrap_or(&0);
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

    // --- Step 2: Create outer fringe (duplicate) vertices ---

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

    // --- Step 3: Emit fringe quads (two triangles per boundary edge) ---

    for boundary_edge in &scratch.boundary_edges {
        let start_vertex_key = BoundaryVertexKey::from_position(
            vertices[boundary_edge.start_vertex_index as usize].position,
        );
        let end_vertex_key = BoundaryVertexKey::from_position(
            vertices[boundary_edge.end_vertex_index as usize].position,
        );
        let start_component_index = *scratch
            .triangle_component_map
            .get(&(boundary_edge.triangle_index, start_vertex_key))
            .unwrap_or(&0);
        let end_component_index = *scratch
            .triangle_component_map
            .get(&(boundary_edge.triangle_index, end_vertex_key))
            .unwrap_or(&0);

        let start_outer_vertex_index = match scratch.outer_vertex_indices.get(&BoundaryCornerKey {
            vertex_key: start_vertex_key,
            component_index: start_component_index,
        }) {
            Some(&idx) => idx,
            None => continue,
        };
        let end_outer_vertex_index = match scratch.outer_vertex_indices.get(&BoundaryCornerKey {
            vertex_key: end_vertex_key,
            component_index: end_component_index,
        }) {
            Some(&idx) => idx,
            None => continue,
        };

        let pa = vertices[boundary_edge.start_vertex_index as usize].position;
        let pb = vertices[boundary_edge.end_vertex_index as usize].position;
        let po = vertices[boundary_edge.opposite_vertex_index as usize].position;

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
    /// Creates a new `PathShape` with the specified path and stroke.
    ///
    /// # Parameters
    ///
    /// - `path`: The geometric path defining the shape.
    /// - `stroke`: The stroke properties of the shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::PathShape;
    /// use grafo::Stroke;
    /// use lyon::path::Path;
    ///
    /// let path = Path::builder().build();
    /// let path_shape = PathShape::new(path, Stroke::default());
    /// ```
    pub fn new(path: lyon::path::Path, stroke: Stroke) -> Self {
        Self { path, stroke }
    }

    /// Tessellates the path shape into vertex and index buffers for rendering.
    ///
    /// # Parameters
    ///
    /// - `depth`: The depth value used for rendering order.
    ///
    /// # Returns
    ///
    /// A `VertexBuffers` structure containing the tessellated vertices and indices.
    /// ```
    pub(crate) fn tessellate(
        &self,
        tessellator: &mut FillTessellator,
        buffers_pool: &mut PoolManager,
        tesselation_cache_key: Option<u64>,
    ) -> TessellatedGeometry {
        if let Some(cache_key) = tesselation_cache_key {
            if let Some(cached_vertex_buffers) = buffers_pool
                .tessellation_cache
                .get_vertex_buffers(&cache_key)
            {
                return TessellatedGeometry::Shared(cached_vertex_buffers);
            }
        }

        let mut buffers: VertexBuffers<CustomVertex, u16> =
            buffers_pool.lyon_vertex_buffers_pool.get_vertex_buffers();
        self.tesselate_into_buffers(
            &mut buffers,
            tessellator,
            &mut buffers_pool.aa_fringe_scratch,
        );

        #[allow(clippy::manual_is_multiple_of)]
        let needs_index_padding = buffers.indices.len() % 2 != 0;
        if needs_index_padding {
            buffers.indices.push(0);
        }

        if let Some(cache_key) = tesselation_cache_key {
            let shared_vertex_buffers = Arc::new(buffers);
            buffers_pool
                .tessellation_cache
                .insert_vertex_buffers(cache_key, shared_vertex_buffers.clone());
            TessellatedGeometry::Shared(shared_vertex_buffers)
        } else {
            TessellatedGeometry::Owned(buffers)
        }
    }

    fn tesselate_into_buffers(
        &self,
        buffers: &mut VertexBuffers<CustomVertex, u16>,
        tessellator: &mut FillTessellator,
        aa_fringe_scratch: &mut AaFringeScratch,
    ) {
        let options = FillOptions::default();

        let vertex_converter = VertexConverter::new();

        tessellator
            .tessellate_path(
                &self.path,
                &options,
                &mut BuffersBuilder::new(buffers, vertex_converter),
            )
            .unwrap();

        // Generate UVs for the tessellated path using its axis-aligned bounding box in local space
        if !buffers.vertices.is_empty() {
            // Compute AABB of positions before offset translation
            let mut min_x = f32::INFINITY;
            let mut min_y = f32::INFINITY;
            let mut max_x = f32::NEG_INFINITY;
            let mut max_y = f32::NEG_INFINITY;
            for v in buffers.vertices.iter() {
                let x = v.position[0];
                let y = v.position[1];
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
            let w = (max_x - min_x).max(1e-6);
            let h = (max_y - min_y).max(1e-6);
            for v in buffers.vertices.iter_mut() {
                let u = (v.position[0] - min_x) / w;
                let vcoord = (v.position[1] - min_y) / h;
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
    }
}

/// Contains the data required to draw a shape, including vertex and index buffers.
///
/// This struct is used internally by the renderer and typically does not need to be used
/// directly by library users.
#[derive(Debug)]
pub(crate) struct ShapeDrawData {
    /// The shape associated with this draw data.
    pub(crate) shape: Shape,
    /// Optional cache key for the shape, used for caching tessellated buffers.
    pub(crate) cache_key: Option<u64>,
    /// Range in the aggregated index buffer (start_index, count)  
    pub(crate) index_buffer_range: Option<(usize, usize)>,
    /// Indicates whether the shape is empty (no vertices or indices).
    pub(crate) is_empty: bool,
    /// Stencil reference assigned during render traversal (parent + 1). Cleared after frame.
    pub(crate) stencil_ref: Option<u32>,
    /// Index into the per-frame instance transform buffer
    pub(crate) instance_index: Option<usize>,
    /// Optional per-shape transform applied in clip-space (post-normalization)
    pub(crate) transform: Option<InstanceTransform>,
    /// Optional texture ids associated with this shape for multi-texturing layers.
    /// Layer 0: background/base
    /// Layer 1: foreground/overlay (e.g. text or decals) blended on top
    pub(crate) texture_ids: [Option<u64>; 2],
    /// Optional per-instance color override (normalized [0,1]). If None, use shape's fill.
    pub(crate) color_override: Option<[f32; 4]>,
    /// Whether this node is a leaf in the draw tree (no children).
    pub(crate) is_leaf: bool,
    /// When `false`, skip stencil increment/decrement for this parent
    /// (children render without being clipped to this shape).
    pub(crate) clips_children: bool,
    /// Whether the underlying shape is an axis-aligned rectangle (`Shape::Rect`).
    /// Used to enable scissor-based clipping instead of stencil for rect parents.
    pub(crate) is_rect: bool,
}

impl ShapeDrawData {
    pub fn new(shape: impl Into<Shape>, cache_key: Option<u64>) -> Self {
        let shape = shape.into();
        let is_rect = matches!(shape, Shape::Rect(_));

        ShapeDrawData {
            shape,
            cache_key,
            index_buffer_range: None,
            is_empty: false,
            stencil_ref: None,
            instance_index: None,
            transform: None,
            texture_ids: [None, None],
            color_override: None,
            is_leaf: true,
            clips_children: true,
            is_rect,
        }
    }

    /// Tessellates complex shapes and stores the resulting buffers.
    #[inline(always)]
    pub(crate) fn tessellate(
        &mut self,
        tessellator: &mut FillTessellator,
        buffers_pool: &mut PoolManager,
    ) -> TessellatedGeometry {
        self.shape
            .tessellate(tessellator, buffers_pool, self.cache_key)
    }
}

#[derive(Debug)]
pub(crate) struct CachedShapeDrawData {
    pub(crate) id: u64,
    pub(crate) index_buffer_range: Option<(usize, usize)>,
    pub(crate) is_empty: bool,
    /// Stencil reference assigned during render traversal (parent + 1). Cleared after frame.
    pub(crate) stencil_ref: Option<u32>,
    /// Index into the per-frame instance transform buffer
    pub(crate) instance_index: Option<usize>,
    /// Optional per-shape transform applied in clip-space (post-normalization)
    pub(crate) transform: Option<InstanceTransform>,
    /// Optional texture ids associated with this cached shape
    pub(crate) texture_ids: [Option<u64>; 2],
    /// Optional per-instance color override (normalized [0,1]). If None, use cached shape default.
    pub(crate) color_override: Option<[f32; 4]>,
    /// Whether this node is a leaf in the draw tree (no children).
    pub(crate) is_leaf: bool,
    /// When `false`, skip stencil increment/decrement for this parent
    /// (children render without being clipped to this shape).
    pub(crate) clips_children: bool,
    /// Whether the underlying shape is an axis-aligned rectangle.
    /// Used to enable scissor-based clipping instead of stencil for rect parents.
    pub(crate) is_rect: bool,
    /// The local-space bounding rect when `is_rect` is true, for scissor computation.
    pub(crate) rect_bounds: Option<[(f32, f32); 2]>,
}

impl CachedShapeDrawData {
    pub fn new(id: u64) -> Self {
        Self {
            id,
            index_buffer_range: None,
            is_empty: false,
            stencil_ref: None,
            instance_index: None,
            transform: None,
            texture_ids: [None, None],
            color_override: None,
            is_leaf: true,
            clips_children: true,
            is_rect: false,
            rect_bounds: None,
        }
    }

    pub fn new_rect(id: u64, rect_bounds: [(f32, f32); 2]) -> Self {
        Self {
            is_rect: true,
            rect_bounds: Some(rect_bounds),
            ..Self::new(id)
        }
    }
}

/// A builder for creating complex shapes using a fluent interface.
///
/// The `ShapeBuilder` allows you to define the stroke and path of a shape using
/// method chaining. Fill is assigned per instance through the renderer, and an
/// unset fill renders as transparent. You also can get it from the [`Shape::builder`] method.
///
/// # Examples
///
/// ```rust
/// use grafo::Color;
/// use grafo::Stroke;
/// use grafo::ShapeBuilder;
///
/// let custom_shape = ShapeBuilder::new()
///     // Fill is set per-instance via the renderer (renderer.set_shape_color)
///     .stroke(Stroke::new(3.0, Color::BLACK)) // Black stroke with width 3.0
///     .begin((0.0, 0.0))
///     .line_to((50.0, 10.0))
///     .line_to((50.0, 50.0))
///     .close()
///     .build();
/// ```
#[derive(Clone)]
pub struct ShapeBuilder {
    /// The stroke properties of the shape.
    stroke: Stroke,
    /// The path builder used to construct the shape's geometric path.
    path_builder: lyon::path::Builder,
}

impl Default for ShapeBuilder {
    /// Creates a default `ShapeBuilder` with a black stroke.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::ShapeBuilder;    ///
    /// let builder = ShapeBuilder::default();
    /// ```
    fn default() -> Self {
        Self::new()
    }
}

impl ShapeBuilder {
    /// Creates a new `ShapeBuilder` with a default black stroke.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::ShapeBuilder;
    ///
    /// let builder = ShapeBuilder::new();
    /// ```
    pub fn new() -> Self {
        Self {
            stroke: Stroke::new(1.0, Color::rgb(0, 0, 0)),
            path_builder: lyon::path::Path::builder(),
        }
    }

    /// Sets the stroke properties of the shape.
    ///
    /// # Parameters
    ///
    /// - `stroke`: The desired stroke properties.
    ///
    /// # Returns
    ///
    /// The updated `ShapeBuilder` instance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::Stroke;
    /// use grafo::Color;
    /// use grafo::ShapeBuilder;
    ///
    /// let builder = ShapeBuilder::new().stroke(Stroke::new(2.0, Color::BLACK)); // Black stroke with width 2.0
    /// ```
    pub fn stroke(mut self, stroke: Stroke) -> Self {
        self.stroke = stroke;
        self
    }

    /// Begin path at point
    ///
    /// # Parameters
    ///
    /// - `point`: The start point of the shape.
    ///
    /// # Returns
    ///
    /// The updated `ShapeBuilder` instance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::ShapeBuilder;
    ///
    /// let builder = ShapeBuilder::new().begin((0.0, 0.0));
    /// ```
    pub fn begin(mut self, point: (f32, f32)) -> Self {
        self.path_builder.begin(point.into());
        self
    }

    /// Draws a line from the current point to the specified point.
    ///
    /// # Parameters
    ///
    /// - `point`: The end point of the line.
    ///
    /// # Returns
    ///
    /// The updated `ShapeBuilder` instance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::ShapeBuilder;
    ///
    /// let builder = ShapeBuilder::new().begin((0.0, 0.0)).line_to((50.0, 10.0));
    /// ```
    pub fn line_to(mut self, point: (f32, f32)) -> Self {
        self.path_builder.line_to(point.into());
        self
    }

    /// Draws a cubic Bézier curve from the current point to the specified end point.
    ///
    /// # Parameters
    ///
    /// - `ctrl`: The first control point.
    /// - `ctrl2`: The second control point.
    /// - `to`: The end point of the curve.
    ///
    /// # Returns
    ///
    /// The updated `ShapeBuilder` instance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::ShapeBuilder;
    ///
    /// let builder = ShapeBuilder::new()
    ///     .begin((0.0, 0.0))
    ///     .cubic_bezier_to((20.0, 30.0), (40.0, 30.0), (50.0, 10.0));
    /// ```
    pub fn cubic_bezier_to(mut self, ctrl: (f32, f32), ctrl2: (f32, f32), to: (f32, f32)) -> Self {
        self.path_builder
            .cubic_bezier_to(ctrl.into(), ctrl2.into(), to.into());
        self
    }

    /// Draws a quadratic Bézier curve from the current point to the specified end point.
    ///
    /// # Parameters
    ///
    /// - `ctrl`: The control point.
    /// - `to`: The end point of the curve.
    ///
    /// # Returns
    ///
    /// The updated `ShapeBuilder` instance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::ShapeBuilder;
    ///
    /// let builder = ShapeBuilder::new()
    ///     .begin((0.0, 0.0))
    ///     .quadratic_bezier_to((25.0, 40.0), (50.0, 10.0));
    /// ```
    pub fn quadratic_bezier_to(mut self, ctrl: (f32, f32), to: (f32, f32)) -> Self {
        self.path_builder
            .quadratic_bezier_to(ctrl.into(), to.into());
        self
    }

    /// Closes the current sub-path by drawing a line back to the starting point.
    ///
    /// # Returns
    ///
    /// The updated `ShapeBuilder` instance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::ShapeBuilder;
    ///
    /// let builder = ShapeBuilder::new().begin((0.0, 0.0)).close();
    /// ```
    pub fn close(mut self) -> Self {
        self.path_builder.close();
        self
    }

    /// Builds the [`Shape`] from the accumulated path, fill color, and stroke.
    ///
    /// # Returns
    ///
    /// A `Shape` instance representing the constructed shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::ShapeBuilder;
    ///
    /// let shape = ShapeBuilder::new()
    ///     .begin((0.0, 0.0))
    ///     .line_to((50.0, 10.0))
    ///     .line_to((50.0, 50.0))
    ///     .close()
    ///     .build();
    /// ```
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

/// A set of border radii for a rounded rectangle
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug, Default)]
pub struct BorderRadii {
    pub top_left: f32,
    pub top_right: f32,
    pub bottom_left: f32,
    pub bottom_right: f32,
}

/// Represents the radii of each corner for a rounded rectangle.
///
/// # Fields
///
/// - `top_left`: Radius of the top-left corner.
/// - `top_right`: Radius of the top-right corner.
/// - `bottom_left`: Radius of the bottom-left corner.
/// - `bottom_right`: Radius of the bottom-right corner.
///
/// # Examples
///
/// Creating uniform and non-uniform border radii:
///
/// ```rust
/// use grafo::BorderRadii;
///
/// // Uniform border radii
/// let uniform_radii = BorderRadii::new(10.0);
///
/// // Custom border radii
/// let custom_radii = BorderRadii {
///     top_left: 5.0,
///     top_right: 10.0,
///     bottom_left: 15.0,
///     bottom_right: 20.0,
/// };
/// ```
impl BorderRadii {
    /// Creates a new `BorderRadii` with the same radius for all corners.
    ///
    /// # Parameters
    ///
    /// - `radius`: The radius to apply to all corners.
    ///
    /// # Returns
    ///
    /// A `BorderRadii` instance with uniform corner radii.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::BorderRadii;
    ///
    /// let radii = BorderRadii::new(10.0);
    /// ```
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
        // In the order of a well known convention (CSS) clockwise from top left
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

pub(crate) trait DrawShapeCommand {
    fn index_buffer_range(&self) -> Option<(usize, usize)>; // (start_index, index_count)
    fn is_empty(&self) -> bool;
    fn stencil_ref_mut(&mut self) -> &mut Option<u32>;
    fn instance_index_mut(&mut self) -> &mut Option<usize>;
    fn instance_index(&self) -> Option<usize>;
    fn transform(&self) -> Option<InstanceTransform>;
    fn set_transform(&mut self, t: InstanceTransform);
    fn texture_id(&self, layer: usize) -> Option<u64>;
    fn set_texture_id(&mut self, layer: usize, id: Option<u64>);
    fn instance_color_override(&self) -> Option<[f32; 4]>;
    fn set_instance_color_override(&mut self, color: Option<[f32; 4]>);
    fn clips_children(&self) -> bool;
    fn is_rect(&self) -> bool;
    fn rect_bounds(&self) -> Option<[(f32, f32); 2]>;
}

impl DrawShapeCommand for ShapeDrawData {
    #[inline]
    fn index_buffer_range(&self) -> Option<(usize, usize)> {
        self.index_buffer_range
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.is_empty
    }

    #[inline]
    fn stencil_ref_mut(&mut self) -> &mut Option<u32> {
        &mut self.stencil_ref
    }

    #[inline]
    fn instance_index_mut(&mut self) -> &mut Option<usize> {
        &mut self.instance_index
    }

    #[inline]
    fn instance_index(&self) -> Option<usize> {
        self.instance_index
    }

    #[inline]
    fn transform(&self) -> Option<InstanceTransform> {
        self.transform
    }

    #[inline]
    fn set_transform(&mut self, t: InstanceTransform) {
        self.transform = Some(t);
    }

    #[inline]
    fn texture_id(&self, layer: usize) -> Option<u64> {
        self.texture_ids.get(layer).copied().unwrap_or(None)
    }

    #[inline]
    fn set_texture_id(&mut self, layer: usize, id: Option<u64>) {
        if let Some(slot) = self.texture_ids.get_mut(layer) {
            *slot = id;
        }
    }

    #[inline]
    fn instance_color_override(&self) -> Option<[f32; 4]> {
        self.color_override
    }

    #[inline]
    fn set_instance_color_override(&mut self, color: Option<[f32; 4]>) {
        self.color_override = color;
    }

    #[inline]
    fn clips_children(&self) -> bool {
        self.clips_children
    }

    #[inline]
    fn is_rect(&self) -> bool {
        self.is_rect
    }

    #[inline]
    fn rect_bounds(&self) -> Option<[(f32, f32); 2]> {
        match &self.shape {
            Shape::Rect(r) => Some(r.rect),
            _ => None,
        }
    }
}

impl DrawShapeCommand for CachedShapeDrawData {
    #[inline]
    fn index_buffer_range(&self) -> Option<(usize, usize)> {
        self.index_buffer_range
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.is_empty
    }

    #[inline]
    fn stencil_ref_mut(&mut self) -> &mut Option<u32> {
        &mut self.stencil_ref
    }

    #[inline]
    fn instance_index_mut(&mut self) -> &mut Option<usize> {
        &mut self.instance_index
    }

    #[inline]
    fn instance_index(&self) -> Option<usize> {
        self.instance_index
    }

    #[inline]
    fn transform(&self) -> Option<InstanceTransform> {
        self.transform
    }

    #[inline]
    fn set_transform(&mut self, t: InstanceTransform) {
        self.transform = Some(t);
    }

    #[inline]
    fn texture_id(&self, layer: usize) -> Option<u64> {
        self.texture_ids.get(layer).copied().unwrap_or(None)
    }

    #[inline]
    fn set_texture_id(&mut self, layer: usize, id: Option<u64>) {
        if let Some(slot) = self.texture_ids.get_mut(layer) {
            *slot = id;
        }
    }

    #[inline]
    fn instance_color_override(&self) -> Option<[f32; 4]> {
        self.color_override
    }

    #[inline]
    fn set_instance_color_override(&mut self, color: Option<[f32; 4]>) {
        self.color_override = color;
    }

    #[inline]
    fn clips_children(&self) -> bool {
        self.clips_children
    }

    #[inline]
    fn is_rect(&self) -> bool {
        self.is_rect
    }

    #[inline]
    fn rect_bounds(&self) -> Option<[(f32, f32); 2]> {
        self.rect_bounds
    }
}

#[cfg(test)]
mod tests {
    use super::{
        find_boundary_edges, generate_aa_fringe, AaFringeScratch, BoundaryVertexKey, CustomVertex,
        RectShape, Shape,
    };
    use crate::{util::PoolManager, Stroke};
    use lyon::lyon_tessellation::FillTessellator;
    use std::num::NonZeroUsize;

    fn test_vertex(position: [f32; 2]) -> CustomVertex {
        CustomVertex {
            position,
            tex_coords: [0.0, 0.0],
            normal: [0.0, 0.0],
            coverage: 1.0,
        }
    }

    #[test]
    fn aa_fringe_ignores_internal_seams_with_duplicate_vertices() {
        let mut vertices = vec![
            test_vertex([0.0, 0.0]),
            test_vertex([1.0, 0.0]),
            test_vertex([1.0, 1.0]),
            test_vertex([0.0, 0.0]),
            test_vertex([1.0, 1.0]),
            test_vertex([0.0, 1.0]),
        ];
        let mut indices = vec![0, 1, 2, 3, 4, 5];
        let mut aa_fringe_scratch = AaFringeScratch::new();

        let boundary_edges = find_boundary_edges(&vertices, &indices, &mut aa_fringe_scratch);
        assert_eq!(boundary_edges.len(), 4);

        generate_aa_fringe(&mut vertices, &mut indices, &mut aa_fringe_scratch);

        assert_eq!(vertices.len(), 10);
        assert_eq!(indices.len(), 30);

        let outer_vertex_count = vertices
            .iter()
            .filter(|vertex| vertex.coverage == 0.0)
            .count();
        assert_eq!(outer_vertex_count, 4);

        let unique_outer_vertex_positions = vertices
            .iter()
            .filter(|vertex| vertex.coverage == 0.0)
            .map(|vertex| BoundaryVertexKey::from_position(vertex.position))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(unique_outer_vertex_positions.len(), 4);
    }

    #[test]
    fn rect_tessellation_uses_shared_quad_corners() {
        let rect_shape = RectShape::new([(10.0, 20.0), (30.0, 50.0)], Stroke::default());
        let mut tessellator = FillTessellator::new();
        let mut pool_manager = PoolManager::new(NonZeroUsize::new(1).unwrap());

        let tessellated_geometry =
            Shape::Rect(rect_shape).tessellate(&mut tessellator, &mut pool_manager, None);

        assert_eq!(tessellated_geometry.vertices().len(), 8);
        assert_eq!(tessellated_geometry.indices().len(), 30);

        let fill_vertex_count = tessellated_geometry
            .vertices()
            .iter()
            .filter(|vertex| vertex.coverage == 1.0)
            .count();
        assert_eq!(fill_vertex_count, 4);
    }

    #[test]
    fn aa_fringe_keeps_distinct_corners_for_point_touching_triangles() {
        let mut vertices = vec![
            test_vertex([0.0, 0.0]),
            test_vertex([1.0, 0.0]),
            test_vertex([0.0, 1.0]),
            test_vertex([0.0, 0.0]),
            test_vertex([-1.0, 0.0]),
            test_vertex([0.0, -1.0]),
        ];
        let mut indices = vec![0, 1, 2, 3, 4, 5];
        let mut aa_fringe_scratch = AaFringeScratch::new();

        generate_aa_fringe(&mut vertices, &mut indices, &mut aa_fringe_scratch);

        let outer_vertices = vertices
            .iter()
            .filter(|vertex| vertex.coverage == 0.0)
            .collect::<Vec<_>>();
        assert_eq!(outer_vertices.len(), 6);

        let shared_point_outer_vertices = outer_vertices
            .iter()
            .filter(|vertex| {
                BoundaryVertexKey::from_position(vertex.position)
                    == BoundaryVertexKey::from_position([0.0, 0.0])
            })
            .count();
        assert_eq!(shared_point_outer_vertices, 2);
    }
}
