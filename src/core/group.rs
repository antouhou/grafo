use super::{Color, Fill, MathRect, TransformInstance};

/// Default fill and shared paint coordinates for a group's descendants.
/// Explicit child fills override the group fill and use their own coordinates.
///
/// ```rust
/// use grafo::{Fill, Gradient, GroupDrawCommandOptions, MathRect, Scene, SceneError, TransformInstance};
///
/// # fn example(scene: &mut Scene, gradient: Gradient) -> Result<(), SceneError> {
/// let bounds = MathRect::new((0.0, 0.0).into(), (200.0, 40.0).into());
/// scene.add_group(
///     None,
///     GroupDrawCommandOptions::new()
///         .fill(Fill::Gradient(gradient))
///         .texture_bounds(bounds)
///         .paint_transform(TransformInstance::translation(20.0, 30.0)),
/// )?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Default)]
pub struct GroupDrawCommandOptions {
    /// Fill inherited by descendants without an explicit fill.
    pub fill: Option<Fill>,
    /// Texture fitting rectangle in group paint coordinates. `None` leaves it unspecified.
    pub texture_bounds: Option<MathRect>,
    /// Maps group paint coordinates to logical scene coordinates without moving descendants.
    /// Applies to this group's local gradient. `None` uses the identity transform.
    pub paint_to_scene_transform: Option<TransformInstance>,
}

impl GroupDrawCommandOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fill(mut self, fill: Fill) -> Self {
        self.fill = Some(fill);
        self
    }

    pub fn color(self, color: Color) -> Self {
        self.fill(Fill::Solid(color))
    }

    pub fn texture_bounds(mut self, bounds: MathRect) -> Self {
        self.texture_bounds = Some(bounds);
        self
    }

    pub fn paint_transform(mut self, transform: TransformInstance) -> Self {
        self.paint_to_scene_transform = Some(transform);
        self
    }
}
