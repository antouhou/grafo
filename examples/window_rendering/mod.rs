use grafo::wgpu::SurfaceError;
use grafo::{Renderer, WgpuBackendError};
use tracing::{error, warn};
use winit::event_loop::ActiveEventLoop;

/// Keeps the queued scene intact so reconfiguring a lost surface can retry the same draw.
pub fn render(renderer: &mut Renderer<'_>, event_loop: &ActiveEventLoop) -> bool {
    let result = match renderer.render() {
        Err(WgpuBackendError::Surface(SurfaceError::Lost | SurfaceError::Outdated)) => {
            renderer.resize(renderer.size());
            renderer.render()
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
