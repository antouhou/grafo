use super::{MathRect, TransformInstance};

/// Shared paint coordinates for a group's descendants.
///
/// ```rust
/// use grafo::{GroupDrawCommandOptions, MathRect, Scene, SceneError, TransformInstance};
///
/// # fn example(scene: &mut Scene) -> Result<(), SceneError> {
/// let bounds = MathRect::new((0.0, 0.0).into(), (200.0, 40.0).into());
/// scene.add_group(
///     None,
///     GroupDrawCommandOptions::new()
///         .texture_bounds(bounds)
///         .paint_transform(TransformInstance::translation(20.0, 30.0)),
/// )?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct GroupDrawCommandOptions {
    /// Texture fitting rectangle in group paint coordinates. `None` leaves it unspecified.
    pub texture_bounds: Option<MathRect>,
    /// Maps group paint coordinates to logical scene coordinates without moving descendants.
    /// `None` leaves the paint transform unspecified.
    pub paint_to_scene_transform: Option<TransformInstance>,
}

impl GroupDrawCommandOptions {
    pub fn new() -> Self {
        Self::default()
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
