/// Identifies an execution-owned texture, separately from registered source texture IDs.
/// Planned IDs index outputs in one command stream; registered IDs address existing resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum IntermediateTextureId {
    Registered(u64),
    Planned(usize),
    /// Shape-effect output retained across the render's target scopes.
    ShapeEffect(usize),
}
