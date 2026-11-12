use super::camera::Camera;
use super::ExampleError;
use grafo::{
    Color, DrawMode, FillRule, LineCap, LineJoin, Scene, Shape, ShapeDrawOptions,
    StrokeGeometryOptions, TransformInstance,
};
use usvg::tiny_skia_path::PathSegment;
use usvg::{
    FillRule as SvgFillRule, Group, LineCap as SvgLineCap, LineJoin as SvgLineJoin, Node, Paint,
    PaintOrder, Path, Stroke, Tree,
};

fn solid_color(paint: &Paint, opacity: f32) -> Result<Color, ExampleError> {
    match paint {
        Paint::Color(color) => Ok(Color::rgba(
            color.red,
            color.green,
            color.blue,
            (opacity * 255.0).round() as u8,
        )),
        _ => Err(ExampleError::UnsupportedSvg),
    }
}

fn stroke_geometry(stroke: &Stroke) -> Result<StrokeGeometryOptions, ExampleError> {
    if stroke.dasharray().is_some() {
        return Err(ExampleError::UnsupportedSvg);
    }
    let cap = match stroke.linecap() {
        SvgLineCap::Butt => LineCap::Butt,
        SvgLineCap::Round => LineCap::Round,
        SvgLineCap::Square => LineCap::Square,
    };
    let join = match stroke.linejoin() {
        SvgLineJoin::Miter => LineJoin::Miter,
        SvgLineJoin::Round => LineJoin::Round,
        SvgLineJoin::Bevel => LineJoin::Bevel,
        SvgLineJoin::MiterClip => return Err(ExampleError::UnsupportedSvg),
    };
    Ok(StrokeGeometryOptions::new(stroke.width().get())
        .line_cap(cap)
        .line_join(join)
        .miter_limit(stroke.miterlimit().get()))
}

fn import_shape(path: &Path) -> Result<Shape, ExampleError> {
    let fill_rule = match path.fill().map(|fill| fill.rule()) {
        Some(SvgFillRule::EvenOdd) => FillRule::EvenOdd,
        _ => FillRule::NonZero,
    };
    let mut builder = Shape::builder().fill_rule(fill_rule);
    let mut is_subpath_open = false;
    for segment in path.data().segments() {
        match segment {
            PathSegment::MoveTo(point) => {
                if is_subpath_open {
                    builder = builder.end();
                }
                builder = builder.begin((point.x, point.y));
                is_subpath_open = true;
            }
            PathSegment::LineTo(point) => builder = builder.line_to((point.x, point.y)),
            PathSegment::QuadTo(control, point) => {
                builder = builder.quadratic_bezier_to((control.x, control.y), (point.x, point.y));
            }
            PathSegment::CubicTo(first_control, second_control, point) => {
                builder = builder.cubic_bezier_to(
                    (first_control.x, first_control.y),
                    (second_control.x, second_control.y),
                    (point.x, point.y),
                );
            }
            PathSegment::Close => {
                builder = builder.close();
                is_subpath_open = false;
            }
        }
    }
    if is_subpath_open {
        builder = builder.end();
    }
    let mut shape = builder.build();
    if let Some(stroke) = path.stroke() {
        shape = shape.with_stroke(stroke_geometry(stroke)?);
    }
    Ok(shape)
}

struct TigerDraw {
    cache_key: u64,
    mode: DrawMode,
    color: Color,
    document_transform: TransformInstance,
}

impl TigerDraw {
    fn options(&self, camera: &Camera) -> ShapeDrawOptions {
        ShapeDrawOptions::new()
            .draw_mode(self.mode)
            .color(self.color)
            .transform(self.document_transform.then(&camera.transform()))
    }
}

fn load_path(
    path: &Path,
    scene: &mut Scene,
    draws: &mut Vec<TigerDraw>,
) -> Result<(), ExampleError> {
    if !path.is_visible() {
        return Ok(());
    }
    let shape = import_shape(path)?;
    let cache_key = draws.len() as u64;
    scene.load_shape(shape, cache_key, None)?;
    let transform = path.abs_transform();
    let document_transform = TransformInstance::affine_2d(
        transform.sx,
        transform.ky,
        transform.kx,
        transform.sy,
        transform.tx,
        transform.ty,
    );
    let fill = path
        .fill()
        .map(|fill| solid_color(fill.paint(), fill.opacity().get()))
        .transpose()?;
    let stroke = path
        .stroke()
        .map(|stroke| solid_color(stroke.paint(), stroke.opacity().get()))
        .transpose()?;
    let mut paints = [(DrawMode::Fill, fill), (DrawMode::Stroke, stroke)];
    if path.paint_order() == PaintOrder::StrokeAndFill {
        paints.reverse();
    }
    for (mode, color) in paints {
        if let Some(color) = color {
            draws.push(TigerDraw {
                cache_key,
                mode,
                color,
                document_transform,
            });
        }
    }
    Ok(())
}

fn load_group(
    group: &Group,
    scene: &mut Scene,
    draws: &mut Vec<TigerDraw>,
) -> Result<(), ExampleError> {
    if group.opacity().get() != 1.0
        || group.clip_path().is_some()
        || group.mask().is_some()
        || !group.filters().is_empty()
    {
        return Err(ExampleError::UnsupportedSvg);
    }
    for node in group.children() {
        match node {
            Node::Group(group) => load_group(group, scene, draws)?,
            Node::Path(path) => load_path(path, scene, draws)?,
            _ => return Err(ExampleError::UnsupportedSvg),
        }
    }
    Ok(())
}

/// Imports the bundled tiger's paths and keeps their fill and stroke draws in paint order.
pub struct TigerArtwork {
    draws: Vec<TigerDraw>,
    pub bounds: (f64, f64, f64, f64),
}

impl TigerArtwork {
    pub fn load(scene: &mut Scene) -> Result<Self, ExampleError> {
        let tree = Tree::from_str(include_str!("../assets/tiger.svg"), &Default::default())?;
        let bounds = tree.root().abs_stroke_bounding_box();
        let mut draws = Vec::new();
        load_group(tree.root(), scene, &mut draws)?;
        Ok(Self {
            draws,
            bounds: (
                bounds.x() as f64,
                bounds.y() as f64,
                bounds.width() as f64,
                bounds.height() as f64,
            ),
        })
    }

    pub fn queue(
        &self,
        scene: &mut Scene,
        camera: &Camera,
        draw_ids: &mut Vec<usize>,
    ) -> Result<(), ExampleError> {
        draw_ids.clear();
        for draw in &self.draws {
            draw_ids.push(scene.add_cached_shape(draw.cache_key, None, draw.options(camera))?);
        }
        Ok(())
    }

    pub fn update_camera(
        &self,
        scene: &mut Scene,
        camera: &Camera,
        draw_ids: &[usize],
    ) -> Result<(), ExampleError> {
        for (draw, &node_id) in self.draws.iter().zip(draw_ids) {
            scene.replace_with_cached_shape(node_id, draw.cache_key, draw.options(camera))?;
        }
        Ok(())
    }
}
