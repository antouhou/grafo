use crate::commands::RenderPlan;

/// Executes completed commands on the supplied surface.
///
/// Resource registration and uploads are separate from command interpretation.
/// The surface lifetime permits window surfaces borrowed from their owner.
pub trait RenderBackend<'surface> {
    type Surface;
    type Error;

    fn render(
        &mut self,
        commands: &RenderPlan,
        surface: &mut Self::Surface,
    ) -> Result<(), Self::Error>;
}
