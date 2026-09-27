use grafo::wgpu::SurfaceError;
use grafo::{Renderer, Surface, WgpuBackendError};
use tracing::{error, warn};
use winit::event_loop::ActiveEventLoop;

/// Keeps the queued scene intact so reconfiguring a lost surface can retry the same draw.
pub fn render(
    renderer: &mut Renderer,
    surface: &mut Surface,
    event_loop: &ActiveEventLoop,
) -> bool {
    if surface.size().0 == 0 || surface.size().1 == 0 {
        return false;
    }
    let result = match renderer.render(&mut *surface) {
        Err(WgpuBackendError::Surface(SurfaceError::Lost | SurfaceError::Outdated)) => {
            surface.invalidate();
            renderer.render(&mut *surface)
        }
        result => result,
    };

    match result {
        Ok(()) => true,
        Err(WgpuBackendError::Surface(
            error @ (SurfaceError::Timeout | SurfaceError::Lost | SurfaceError::Outdated),
        )) => {
            // Surface acquisition has no readiness event. Wait for the next window redraw.
            warn!("Skipping redraw: {error}");
            false
        }
        Err(error) => {
            error!("Rendering failed: {error}");
            event_loop.exit();
            false
        }
    }
}
