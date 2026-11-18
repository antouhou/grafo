use grafo::{Scene, Surface, SurfaceError, WgpuBackendError};
use std::sync::Arc;
use tracing::{error, warn};
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RenderResult {
    Presented,
    Occluded,
    Skipped,
}

/// Keeps the queued scene intact so reconfiguring an outdated surface can retry the same draw.
pub fn render(
    scene: &mut Scene,
    surface: &mut Surface,
    window: &Arc<Window>,
    event_loop: &ActiveEventLoop,
) -> RenderResult {
    if surface.size().0 == 0 || surface.size().1 == 0 {
        return RenderResult::Skipped;
    }
    let result = match scene.render(&mut *surface) {
        Err(WgpuBackendError::Surface(SurfaceError::Outdated)) => scene.render(&mut *surface),
        Err(WgpuBackendError::Surface(SurfaceError::Lost)) => {
            if let Err(error) = surface.recreate(window.clone()) {
                error!("Recreating the surface failed: {error}");
                event_loop.exit();
                return RenderResult::Skipped;
            }
            scene.render(&mut *surface)
        }
        result => result,
    };

    match result {
        Ok(()) => RenderResult::Presented,
        Err(WgpuBackendError::Surface(SurfaceError::Occluded)) => RenderResult::Occluded,
        Err(WgpuBackendError::Surface(
            error @ (SurfaceError::Timeout | SurfaceError::Outdated),
        )) => {
            // Surface acquisition has no readiness event. Wait for the next window redraw.
            warn!("Skipping redraw: {error}");
            RenderResult::Skipped
        }
        Err(error) => {
            error!("Rendering failed: {error}");
            event_loop.exit();
            RenderResult::Skipped
        }
    }
}
