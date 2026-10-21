use grafo::wgpu::SurfaceError;
use grafo::{Scene, Surface, WgpuBackendError};
use tracing::{error, warn};
use winit::event_loop::ActiveEventLoop;

/// Keeps the queued scene intact so reconfiguring a lost surface can retry the same draw.
pub fn render(scene: &mut Scene, surface: &mut Surface, event_loop: &ActiveEventLoop) -> bool {
    if surface.size().0 == 0 || surface.size().1 == 0 {
        return false;
    }
    let result = match scene.render(&mut *surface) {
        Err(WgpuBackendError::Surface(SurfaceError::Lost | SurfaceError::Outdated)) => {
            surface.invalidate();
            scene.render(&mut *surface)
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
