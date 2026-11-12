//! Rectangles and paths. Set each instance's fill through the scene.
//!
//! See [`Shape::rect`], [`Shape::rounded_rect`], [`Shape::path`], and [`ShapeBuilder`].

use crate::core::cache::CachedTessellation;
use crate::core::gradient::types::Fill;
use crate::core::shape::boundary::RoundedRectBoundary;
use crate::core::vertex::{CustomVertex, InstanceTransform, TextureUvTransform};
use crate::core::Color;
use lyon::geom::euclid::default::Transform3D;
use lyon::path::{Builder, Path, PathEvent, Winding};
pub use lyon::path::{FillRule, LineCap};
use lyon::tessellation::{
    BuffersBuilder, FillOptions, FillVertex, FillVertexConstructor, TessellationError,
    VertexBuffers,
};
pub(crate) use resources::ShapeResources;
use std::sync::Arc;
pub use stroke::{LineJoin, StrokeGeometryError};
use thiserror::Error;

pub(crate) mod boundary;
mod resources;
mod stroke;

#[derive(Debug, Error)]
pub(crate) enum ShapeTessellationError {
    #[error(transparent)]
    Tessellation(#[from] TessellationError),
    #[error(transparent)]
    StrokeGeometry(#[from] StrokeGeometryError),
}

/// Geometry selected for a shape draw.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DrawMode {
    /// Draws the shape's filled interior.
    #[default]
    Fill,
    /// Draws the shape's stroke. Stroke draws must be leaf nodes.
    Stroke,
}

/// Shared tessellated geometry for a shape, independent of its paint and transform.
#[derive(Debug, Clone)]
pub struct CachedShapeHandle {
    /// Fill geometry.
    pub fill_tessellation: Arc<CachedTessellation>,
    /// Geometry for a nonempty stroke.
    pub stroke_tessellation: Option<Arc<CachedTessellation>>,
    /// Content key for the fill and stroke geometry.
    pub geometry_id: Option<u64>,
}

impl CachedShapeHandle {
    /// Caches tessellation under `geometry_id` and reuses it during buffer aggregation.
    /// Equal IDs must identify identical fill and stroke geometry.
    pub(crate) fn new(
        shape: &Shape,
        shape_resources: &mut ShapeResources,
        geometry_id: Option<u64>,
    ) -> Result<Self, ShapeTessellationError> {
        if let Some(cached_shape) =
            geometry_id.and_then(|id| shape_resources.tessellation_cache.get_shape(&id))
        {
            return Ok(cached_shape);
        }

        let cached_shape = Self {
            fill_tessellation: shape.tessellate_fill(shape_resources)?,
            stroke_tessellation: shape.tessellate_stroke(shape_resources)?,
            geometry_id,
        };
        shape_resources
            .tessellation_cache
            .insert_shape(&cached_shape);
        Ok(cached_shape)
    }

    pub(crate) fn from_fill_events(
        events: impl IntoIterator<Item = PathEvent>,
        shape_resources: &mut ShapeResources,
        geometry_id: u64,
    ) -> Result<Self, TessellationError> {
        if let Some(cached_shape) = shape_resources.tessellation_cache.get_shape(&geometry_id) {
            return Ok(cached_shape);
        }

        let cached_shape = Self {
            fill_tessellation: PathShape::tessellate_fill_events(
                events,
                FillRule::NonZero,
                false,
                shape_resources,
            )?,
            stroke_tessellation: None,
            geometry_id: Some(geometry_id),
        };
        shape_resources
            .tessellation_cache
            .insert_shape(&cached_shape);
        Ok(cached_shape)
    }

    #[inline]
    pub fn fill_vertex_buffers(&self) -> &Arc<VertexBuffers<CustomVertex, u32>> {
        &self.fill_tessellation.vertex_buffers
    }

    #[inline]
    pub fn fill_local_bounds(&self) -> [(f32, f32); 2] {
        self.fill_tessellation.local_bounds
    }

    #[inline]
    pub fn fill_texture_mapping_size(&self) -> [f32; 2] {
        self.fill_tessellation.texture_mapping_size
    }
}

#[derive(Debug)]
pub struct ShapeInstance {
    pub cached_shape: CachedShapeHandle,
    pub draw_mode: DrawMode,
    /// Optional per-shape transform applied in pixel space before clip-space normalization.
    pub transform: Option<InstanceTransform>,
    /// Texture sources associated with this cached shape.
    pub textures: [ShapeTextureOptions; 2],
    /// Linear RGBA color for a solid fill. Other fills leave this unset.
    pub color_override: Option<[f32; 4]>,
    /// Resolved solid or gradient fill. `None` leaves the shape transparent.
    pub fill: Option<Fill>,
    /// Maps shape-local positions to the local gradient's paint coordinates.
    pub paint_transform: Transform3D<f32>,
}

impl ShapeInstance {
    pub fn new(cached_shape: CachedShapeHandle, options: ShapeDrawOptions) -> Self {
        Self {
            cached_shape,
            draw_mode: options.draw_mode,
            transform: options.transform,
            textures: [options.background_texture, options.foreground_texture],
            color_override: match options.fill.as_ref() {
                Some(Fill::Solid(color)) => Some(color.normalize()),
                _ => None,
            },
            fill: options.fill,
            paint_transform: Transform3D::identity(),
        }
    }

    pub fn tessellation(&self) -> Option<&Arc<CachedTessellation>> {
        match self.draw_mode {
            DrawMode::Fill => Some(&self.cached_shape.fill_tessellation),
            DrawMode::Stroke => self.cached_shape.stroke_tessellation.as_ref(),
        }
    }

    /// Returns the vertices and triangle indices for the selected draw mode.
    pub fn vertex_buffers(&self) -> Option<&Arc<VertexBuffers<CustomVertex, u32>>> {
        self.tessellation().map(|geometry| &geometry.vertex_buffers)
    }

    pub fn local_bounds(&self) -> [(f32, f32); 2] {
        self.tessellation()
            .map_or([(0.0, 0.0); 2], |geometry| geometry.local_bounds)
    }

    pub fn texture_mapping_size(&self) -> [f32; 2] {
        self.tessellation()
            .map_or([0.0; 2], |geometry| geometry.texture_mapping_size)
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

fn finalize_path_tessellation(
    mut buffers: VertexBuffers<CustomVertex, u32>,
    is_rounded_rect: bool,
    shape_resources: &mut ShapeResources,
) -> Arc<CachedTessellation> {
    let local_bounds = compute_vertex_bounds(&buffers.vertices);
    let texture_mapping_size = rect_size(local_bounds);
    for vertex in &mut buffers.vertices {
        vertex.tex_coords = [
            (vertex.position[0] - local_bounds[0].0) / texture_mapping_size[0],
            (vertex.position[1] - local_bounds[0].1) / texture_mapping_size[1],
        ];
    }

    let mut boundary_edges: Vec<_> = shape_resources
        .path_boundary_builder
        .build(&buffers.vertices, &buffers.indices, |position| position)
        .collect();
    let rounded_rect_boundary = is_rounded_rect
        .then(|| RoundedRectBoundary::new(&mut boundary_edges))
        .flatten();
    Arc::new(CachedTessellation {
        is_rect: false,
        boundary_edges,
        rounded_rect_boundary,
        vertex_buffers: Arc::new(buffers),
        local_bounds,
        texture_mapping_size,
    })
}

#[derive(Debug, Clone)]
enum ShapeGeometry {
    Path(PathShape),
    Rect(RectShape),
}

/// A rectangle or a path made of lines and Bezier curves, with optional stroke geometry.
///
/// Use [`Self::rect`] or [`Self::rounded_rect`] for rectangles, [`Self::path`] for an existing
/// Lyon path, or [`Self::builder`] to build a custom path. Set the paint when queueing the shape
/// with [`ShapeDrawOptions`].
#[derive(Debug, Clone)]
pub struct Shape {
    geometry: ShapeGeometry,
    stroke: Option<StrokeGeometryOptions>,
}

impl Shape {
    /// Starts an empty path. See [`ShapeBuilder`] for an example.
    pub fn builder() -> ShapeBuilder {
        ShapeBuilder::new()
    }

    /// Creates a shape from a Lyon path using [`FillRule::NonZero`].
    pub fn path(path: Path) -> Self {
        PathShape::new(path).into()
    }

    /// Stores the stroke geometry.
    ///
    /// Draw it with [`DrawMode::Stroke`] and set its paint with [`ShapeDrawOptions::fill`].
    pub fn with_stroke(mut self, stroke: StrokeGeometryOptions) -> Self {
        self.stroke = Some(stroke);
        self
    }

    /// Returns the stroke geometry attached to the shape, if present.
    pub fn stroke(&self) -> Option<&StrokeGeometryOptions> {
        self.stroke.as_ref()
    }

    /// Creates a rectangle from its top-left and bottom-right coordinates.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::Shape;
    ///
    /// let rect = Shape::rect([(0.0, 0.0), (100.0, 50.0)]);
    /// ```
    pub fn rect(rect: [(f32, f32); 2]) -> Shape {
        RectShape::new(rect).into()
    }

    /// Creates a rounded rectangle from its top-left and bottom-right coordinates.
    /// Each corner radius is specified by [`BorderRadii`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use grafo::{BorderRadii, Shape};
    ///
    /// let rounded_rect = Shape::rounded_rect(
    ///     [(0.0, 0.0), (100.0, 50.0)],
    ///     BorderRadii::new(10.0),
    /// );
    /// ```
    pub fn rounded_rect(rect: [(f32, f32); 2], border_radii: BorderRadii) -> Shape {
        let mut path_builder = Path::builder();
        let box2d = lyon::math::Box2D::new(rect[0].into(), rect[1].into());

        path_builder.add_rounded_rectangle(&box2d, &border_radii.into(), Winding::Positive);
        let path = path_builder.build();

        let mut path_shape = PathShape::new(path);
        path_shape.is_rounded_rect = true;
        path_shape.into()
    }

    fn tessellate_fill(
        &self,
        shape_resources: &mut ShapeResources,
    ) -> Result<Arc<CachedTessellation>, TessellationError> {
        match &self.geometry {
            ShapeGeometry::Path(path_shape) => PathShape::tessellate_fill_events(
                path_shape.path.iter(),
                path_shape.fill_rule,
                path_shape.is_rounded_rect,
                shape_resources,
            ),
            ShapeGeometry::Rect(rect_shape) => {
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
                    },
                    CustomVertex {
                        position: [max_x, min_y],
                        tex_coords: uv(max_x, min_y),
                    },
                    CustomVertex {
                        position: [max_x, max_y],
                        tex_coords: uv(max_x, max_y),
                    },
                    CustomVertex {
                        position: [min_x, max_y],
                        tex_coords: uv(min_x, max_y),
                    },
                ];
                let indices = [0u32, 1, 2, 0, 2, 3];
                let local_bounds = rect_shape.rect;

                let mut vertex_buffers = VertexBuffers::new();

                vertex_buffers.vertices.extend(quad);
                vertex_buffers.indices.extend(indices);

                let boundary_edges = shape_resources
                    .path_boundary_builder
                    .build(
                        &vertex_buffers.vertices,
                        &vertex_buffers.indices,
                        |position| position,
                    )
                    .collect();

                Ok(Arc::new(CachedTessellation {
                    is_rect: true,
                    boundary_edges,
                    rounded_rect_boundary: None,
                    vertex_buffers: Arc::new(vertex_buffers),
                    local_bounds,
                    texture_mapping_size: rect_size(local_bounds),
                }))
            }
        }
    }
}

impl From<PathShape> for Shape {
    fn from(value: PathShape) -> Self {
        Self {
            geometry: ShapeGeometry::Path(value),
            stroke: None,
        }
    }
}

impl From<RectShape> for Shape {
    fn from(value: RectShape) -> Self {
        Self {
            geometry: ShapeGeometry::Rect(value),
            stroke: None,
        }
    }
}

impl AsRef<Shape> for Shape {
    fn as_ref(&self) -> &Shape {
        self
    }
}

/// A rectangle's coordinates. Set its fill with [`ShapeDrawOptions`].
///
/// Use [`Shape::rect`] to construct a shape directly from rectangle coordinates.
#[derive(Debug, Clone)]
pub struct RectShape {
    /// Top-left and bottom-right coordinates.
    pub(crate) rect: [(f32, f32); 2],
}

impl RectShape {
    /// Creates a rectangle from its top-left and bottom-right coordinates.
    pub fn new(rect: [(f32, f32); 2]) -> Self {
        Self { rect }
    }
}

/// A custom path. Set its fill with [`ShapeDrawOptions`].
///
/// Use [`Shape::path`] to construct a shape directly from a Lyon path with the default fill rule.
#[derive(Clone, Debug)]
pub struct PathShape {
    pub(crate) path: Path,
    fill_rule: FillRule,
    is_rounded_rect: bool,
}

struct FillVertexConverter;

impl FillVertexConstructor<CustomVertex> for FillVertexConverter {
    fn new_vertex(&mut self, vertex: FillVertex) -> CustomVertex {
        CustomVertex {
            position: vertex.position().to_array(),
            tex_coords: [0.0, 0.0],
        }
    }
}

impl PathShape {
    /// Uses an existing Lyon path as the shape's geometry, filled with [`FillRule::NonZero`].
    pub fn new(path: Path) -> Self {
        Self {
            path,
            fill_rule: FillRule::NonZero,
            is_rounded_rect: false,
        }
    }

    /// Selects which regions inside the path are filled. Defaults to [`FillRule::NonZero`].
    /// Use [`FillRule::EvenOdd`] for holes that do not depend on contour direction.
    pub fn fill_rule(mut self, fill_rule: FillRule) -> Self {
        self.fill_rule = fill_rule;
        self
    }

    fn tessellate_fill_events(
        events: impl IntoIterator<Item = PathEvent>,
        fill_rule: FillRule,
        is_rounded_rect: bool,
        shape_resources: &mut ShapeResources,
    ) -> Result<Arc<CachedTessellation>, TessellationError> {
        let mut buffers = VertexBuffers::new();
        let options = FillOptions::default().with_fill_rule(fill_rule);
        shape_resources.fill_tessellator.tessellate(
            events,
            &options,
            &mut BuffersBuilder::new(&mut buffers, FillVertexConverter),
        )?;
        Ok(finalize_path_tessellation(
            buffers,
            is_rounded_rect,
            shape_resources,
        ))
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

/// Geometry options for a shape's stroke.
///
/// [`Self::new`] uses butt caps, miter joins, and a miter limit of `4.0`.
#[derive(Clone, Debug)]
pub struct StrokeGeometryOptions {
    /// Width in shape-local units, before the shape transform and viewport scaling.
    /// Must be finite and nonnegative. Zero produces no stroke mesh.
    pub width: f32,
    /// Cap at the start of each open subpath.
    pub start_cap: LineCap,
    /// Cap at the end of each open subpath.
    pub end_cap: LineCap,
    /// Join between consecutive path segments.
    pub line_join: LineJoin,
    /// Maximum miter length as a multiple of the stroke width. Must be finite and at least `1.0`.
    /// Applies to [`LineJoin::Miter`].
    pub miter_limit: f32,
}

impl StrokeGeometryOptions {
    pub fn new(width: f32) -> Self {
        Self {
            width,
            start_cap: LineCap::Butt,
            end_cap: LineCap::Butt,
            line_join: LineJoin::Miter,
            miter_limit: 4.0,
        }
    }

    /// Sets both the start and end caps.
    pub fn line_cap(mut self, line_cap: LineCap) -> Self {
        self.start_cap = line_cap;
        self.end_cap = line_cap;
        self
    }

    pub fn start_cap(mut self, start_cap: LineCap) -> Self {
        self.start_cap = start_cap;
        self
    }

    pub fn end_cap(mut self, end_cap: LineCap) -> Self {
        self.end_cap = end_cap;
        self
    }

    pub fn line_join(mut self, line_join: LineJoin) -> Self {
        self.line_join = line_join;
        self
    }

    pub fn miter_limit(mut self, miter_limit: f32) -> Self {
        self.miter_limit = miter_limit;
        self
    }
}

#[derive(Clone, Debug)]
pub struct ShapeDrawOptions {
    /// Selects fill or stroke geometry. Defaults to [`DrawMode::Fill`].
    /// Stroke mode draws nothing when the shape has no stroke geometry.
    pub draw_mode: DrawMode,
    pub transform: Option<InstanceTransform>,
    pub clips_children: bool,
    pub background_texture: ShapeTextureOptions,
    pub foreground_texture: ShapeTextureOptions,
    /// Paint for the selected geometry. `None` inherits the nearest group fill,
    /// or is transparent if no group supplies one.
    /// An explicit fill, including transparent, overrides the group and uses shape coordinates.
    pub fill: Option<Fill>,
}

impl Default for ShapeDrawOptions {
    fn default() -> Self {
        Self {
            draw_mode: DrawMode::default(),
            transform: None,
            clips_children: true,
            background_texture: ShapeTextureOptions::default(),
            foreground_texture: ShapeTextureOptions::default(),
            fill: None,
        }
    }
}

impl ShapeDrawOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn draw_mode(mut self, draw_mode: DrawMode) -> Self {
        self.draw_mode = draw_mode;
        self
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

/// Builds a shape's path through method chaining.
///
/// Assign a fill through [`ShapeDrawOptions`] when queueing the shape.
/// An unset fill inherits the group fill or draws transparent. [`Shape::builder`] also creates this
/// builder.
///
/// # Examples
///
/// ```rust
/// use grafo::{Color, Scene, ShapeBuilder, ShapeDrawOptions};
///
/// # fn example(scene: &mut Scene) {
/// let custom_shape = ShapeBuilder::new()
///     .begin((0.0, 0.0))
///     .line_to((50.0, 10.0))
///     .line_to((50.0, 50.0))
///     .close()
///     .build();
/// scene.add_shape(
///     custom_shape,
///     None,
///     None,
///     ShapeDrawOptions::new().color(Color::rgb(0, 128, 255)),
/// ).unwrap();
/// # }
/// ```
#[derive(Clone)]
pub struct ShapeBuilder {
    path_builder: Builder,
    fill_rule: FillRule,
}

impl Default for ShapeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ShapeBuilder {
    /// Starts an empty path using [`FillRule::NonZero`].
    pub fn new() -> Self {
        Self {
            path_builder: Path::builder(),
            fill_rule: FillRule::NonZero,
        }
    }

    /// Selects which regions inside the path are filled. Defaults to [`FillRule::NonZero`].
    /// Use [`FillRule::EvenOdd`] for holes that do not depend on contour direction.
    pub fn fill_rule(mut self, fill_rule: FillRule) -> Self {
        self.fill_rule = fill_rule;
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

    /// Ends the current subpath and leaves it open.
    /// Filling the shape treats the subpath as closed.
    pub fn end(mut self) -> Self {
        self.path_builder.end(false);
        self
    }

    /// Closes the current subpath with a line back to its starting point.
    pub fn close(mut self) -> Self {
        self.path_builder.close();
        self
    }

    /// Builds the [`Shape`] from the accumulated path.
    pub fn build(self) -> Shape {
        PathShape::new(self.path_builder.build())
            .fill_rule(self.fill_rule)
            .into()
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
