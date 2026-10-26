//! Rectangles and paths. Set each instance's fill through the scene.
//!
//! See [`Shape::rect`], [`Shape::rounded_rect`], and [`ShapeBuilder`] for examples.

use crate::core::cache::CachedTessellation;
use crate::core::gradient::types::Fill;
use crate::core::shape::boundary::RoundedRectBoundary;
use crate::core::util::ShapeResources;
use crate::core::vertex::{CustomVertex, InstanceTransform, TextureUvTransform};
use crate::core::Color;
use lyon::lyon_tessellation::{
    BuffersBuilder, FillOptions, FillTessellator, FillVertex, VertexBuffers,
};
pub use lyon::path::FillRule;
use lyon::path::{Builder, Path, PathEvent, Winding};
use lyon::tessellation::FillVertexConstructor;
use std::sync::Arc;

pub(crate) mod boundary;

#[derive(Debug, Clone)]
pub struct CachedShapeHandle {
    pub tessellation: Arc<CachedTessellation>,
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
        let tessellation = shape.tessellate(tessellator, shape_resources, geometry_id);
        Self {
            tessellation,
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
    /// Starts an empty path. See [`ShapeBuilder`] for an example.
    pub fn builder() -> ShapeBuilder {
        ShapeBuilder::new()
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
        let rect_shape = RectShape::new(rect);
        Shape::Rect(rect_shape)
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
                let indices = [0u16, 1, 2, 0, 2, 3];
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

                let tessellation = Arc::new(CachedTessellation {
                    is_rect: true,
                    boundary_edges,
                    rounded_rect_boundary: None,
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

/// A rectangle's coordinates. Set its fill with [`ShapeDrawCommandOptions`].
///
/// [`Shape::rect`] constructs this and wraps it in [`Shape::Rect`].
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

/// A custom path. Set its fill with [`ShapeDrawCommandOptions`].
///
/// [`Shape::builder`] constructs this and wraps it in [`Shape::Path`].
#[derive(Clone, Debug)]
pub struct PathShape {
    pub(crate) path: Path,
    fill_rule: FillRule,
    is_rounded_rect: bool,
}

struct VertexConverter;

impl FillVertexConstructor<CustomVertex> for VertexConverter {
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

    /// Returns shared geometry and bounds, reusing the tessellation cache when a key is given.
    pub(crate) fn tessellate(
        &self,
        tessellator: &mut FillTessellator,
        shape_resources: &mut ShapeResources,
        tesselation_cache_key: Option<u64>,
    ) -> Arc<CachedTessellation> {
        Self::tessellate_events(
            self.path.iter(),
            self.fill_rule,
            self.is_rounded_rect,
            tessellator,
            shape_resources,
            tesselation_cache_key,
        )
    }

    pub(crate) fn tessellate_events(
        events: impl IntoIterator<Item = PathEvent>,
        fill_rule: FillRule,
        is_rounded_rect: bool,
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
        let local_bounds =
            Self::tessellate_into_buffers(events, fill_rule, &mut buffers, tessellator);

        #[allow(clippy::manual_is_multiple_of)]
        let needs_index_padding = buffers.indices.len() % 2 != 0;
        if needs_index_padding {
            buffers.indices.push(0);
        }

        let mut boundary_edges: Vec<_> = shape_resources
            .path_boundary_builder
            .build(&buffers.vertices, &buffers.indices, |position| position)
            .collect();
        let rounded_rect_boundary = is_rounded_rect
            .then(|| RoundedRectBoundary::new(&mut boundary_edges))
            .flatten();
        let tessellation = Arc::new(CachedTessellation {
            is_rect: false,
            boundary_edges,
            rounded_rect_boundary,
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
        events: impl IntoIterator<Item = PathEvent>,
        fill_rule: FillRule,
        buffers: &mut VertexBuffers<CustomVertex, u16>,
        tessellator: &mut FillTessellator,
    ) -> [(f32, f32); 2] {
        let options = FillOptions::default().with_fill_rule(fill_rule);

        tessellator
            .tessellate(
                events,
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

/// Builds a shape's path through method chaining.
///
/// Assign a fill through [`ShapeDrawCommandOptions`] when queueing the shape.
/// An unset fill renders as transparent. [`Shape::builder`] also creates this builder.
///
/// # Examples
///
/// ```rust
/// use grafo::{Color, Scene, ShapeBuilder, ShapeDrawCommandOptions};
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
///     ShapeDrawCommandOptions::new().color(Color::rgb(0, 128, 255)),
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

    /// Closes the current subpath with a line back to its starting point.
    pub fn close(mut self) -> Self {
        self.path_builder.close();
        self
    }

    /// Builds the [`Shape`] from the accumulated path.
    pub fn build(self) -> Shape {
        let path = self.path_builder.build();
        Shape::Path(PathShape {
            path,
            fill_rule: self.fill_rule,
            is_rounded_rect: false,
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
