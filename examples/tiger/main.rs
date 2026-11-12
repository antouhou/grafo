//! Scroll to zoom at the pointer, drag with the left button to pan, and press Space to toggle
//! clearing and requeueing the scene on every redraw. Geometry stays loaded in either mode.

use artwork::TigerArtwork;
use camera::Camera;
use futures::executor::block_on;
use grafo::wgpu::SurfaceError;
use grafo::{
    Color, DrawCommandError, Scene, SceneContext, SceneCreationError, SceneError, Shape,
    ShapeDrawOptions, Surface, TransformInstance, WgpuBackendError,
};
use stats::{FrameStatistics, StatsOverlay};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tracing::{error, warn};
use usvg::Error as SvgError;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::error::{EventLoopError, OsError};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

mod artwork;
mod camera;
mod stats;

const BACKGROUND_CACHE_KEY: u64 = u64::MAX;

#[derive(Debug, Error)]
enum ExampleError {
    #[error(transparent)]
    EventLoop(#[from] EventLoopError),
    #[error(transparent)]
    Window(#[from] OsError),
    #[error(transparent)]
    Context(#[from] SceneCreationError),
    #[error(transparent)]
    Scene(#[from] SceneError),
    #[error(transparent)]
    Draw(#[from] DrawCommandError<WgpuBackendError>),
    #[error(transparent)]
    Render(#[from] WgpuBackendError),
    #[error(transparent)]
    Svg(#[from] SvgError),
    #[error("the tiger importer supports only solid fills and undashed strokes")]
    UnsupportedSvg,
}

fn logical_size(size: PhysicalSize<u32>, scale_factor: f64) -> (f64, f64) {
    let size = size.to_logical::<f64>(scale_factor);
    (size.width, size.height)
}

fn background_options(viewport: (f64, f64)) -> ShapeDrawOptions {
    ShapeDrawOptions::new()
        .color(Color::rgb(204, 204, 204))
        .transform(TransformInstance::scale(
            viewport.0 as f32,
            viewport.1 as f32,
        ))
}

struct TigerWindow {
    window: Arc<Window>,
    scene: Scene,
    surface: Surface,
    artwork: TigerArtwork,
    camera: Camera,
    draw_ids: Vec<usize>,
    background_id: usize,
    stats: FrameStatistics,
    overlay: StatsOverlay,
    cursor: Option<(f64, f64)>,
    is_dragging: bool,
    is_occluded: bool,
    should_rebuild_each_redraw: bool,
}

impl TigerWindow {
    fn viewport(&self) -> (f64, f64) {
        logical_size(self.window.inner_size(), self.window.scale_factor())
    }

    fn refresh_overlay(&mut self) -> Result<(), ExampleError> {
        self.overlay.update_text(
            &self.stats,
            self.should_rebuild_each_redraw,
            self.camera.scale(),
        );
        let viewport = self.viewport();
        self.overlay.refresh(&mut self.scene, viewport)
    }

    fn update_camera(&mut self) -> Result<(), ExampleError> {
        self.artwork
            .update_camera(&mut self.scene, &self.camera, &self.draw_ids)
    }

    fn resize(&mut self, size: PhysicalSize<u32>) -> Result<(), ExampleError> {
        self.surface.resize((size.width, size.height));
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }
        self.camera
            .resize(logical_size(size, self.window.scale_factor()));
        self.scene.replace_with_cached_shape(
            self.background_id,
            BACKGROUND_CACHE_KEY,
            background_options(self.viewport()),
        )?;
        self.update_camera()?;
        self.refresh_overlay()?;
        self.stats = FrameStatistics::default();
        self.window.request_redraw();
        Ok(())
    }

    fn redraw(&mut self) -> Result<(), ExampleError> {
        if self.is_occluded || self.surface.size().0 == 0 || self.surface.size().1 == 0 {
            return Ok(());
        }
        let rebuild_elapsed = if self.should_rebuild_each_redraw {
            let started = Instant::now();
            self.scene.clear();
            let viewport = self.viewport();
            self.background_id = self.scene.add_cached_shape(
                BACKGROUND_CACHE_KEY,
                None,
                background_options(viewport),
            )?;
            self.artwork
                .queue(&mut self.scene, &self.camera, &mut self.draw_ids)?;
            self.overlay.queue(&mut self.scene, viewport)?;
            started.elapsed()
        } else {
            Duration::ZERO
        };
        let started = Instant::now();
        let result = match self.scene.render(&mut self.surface) {
            Err(WgpuBackendError::Surface(SurfaceError::Lost | SurfaceError::Outdated)) => {
                self.surface.invalidate();
                self.scene.render(&mut self.surface)
            }
            result => result,
        };
        match result {
            Ok(()) => {
                if self.stats.record(started.elapsed(), rebuild_elapsed) {
                    self.refresh_overlay()?;
                }
            }
            Err(WgpuBackendError::Surface(
                error @ (SurfaceError::Timeout | SurfaceError::Lost | SurfaceError::Outdated),
            )) => warn!("Skipping redraw: {error}"),
            Err(error) => return Err(error.into()),
        }
        self.window.request_redraw();
        Ok(())
    }

    fn new(event_loop: &ActiveEventLoop) -> Result<Self, ExampleError> {
        let window = Arc::new(
            event_loop.create_window(
                Window::default_attributes()
                    .with_title("Grafo - Ghostscript tiger")
                    .with_inner_size(LogicalSize::new(960.0, 800.0))
                    .with_min_inner_size(LogicalSize::new(480.0, 320.0)),
            )?,
        );
        let size = window.inner_size();
        let scale_factor = window.scale_factor();
        let physical_size = (size.width, size.height);
        let viewport = logical_size(size, scale_factor);
        let context = block_on(SceneContext::try_new())?;
        let surface = Surface::new(&context, window.clone(), physical_size, false, false)?;
        let mut scene = Scene::try_new_with_context(context, physical_size, scale_factor, 1)?;
        scene.load_shape(
            Shape::rect([(0.0, 0.0), (1.0, 1.0)]),
            BACKGROUND_CACHE_KEY,
            None,
        )?;
        let artwork = TigerArtwork::load(&mut scene)?;
        let camera = Camera::new(artwork.bounds, viewport);
        let stats = FrameStatistics::default();
        let mut overlay = StatsOverlay::new(&mut scene)?;
        overlay.update_text(&stats, false, camera.scale());
        let mut draw_ids = Vec::new();
        let background_id =
            scene.add_cached_shape(BACKGROUND_CACHE_KEY, None, background_options(viewport))?;
        artwork.queue(&mut scene, &camera, &mut draw_ids)?;
        overlay.queue(&mut scene, viewport)?;
        window.request_redraw();
        Ok(Self {
            window,
            scene,
            surface,
            artwork,
            camera,
            draw_ids,
            background_id,
            stats,
            overlay,
            cursor: None,
            is_dragging: false,
            is_occluded: false,
            should_rebuild_each_redraw: false,
        })
    }

    fn handle_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        event: WindowEvent,
    ) -> Result<(), ExampleError> {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => self.redraw()?,
            WindowEvent::Resized(size) => self.resize(size)?,
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.scene.change_scale_factor(scale_factor)?;
                self.cursor = None;
                self.resize(self.window.inner_size())?;
            }
            WindowEvent::Occluded(is_occluded) => {
                self.is_occluded = is_occluded;
                self.stats = FrameStatistics::default();
                if !is_occluded {
                    self.window.request_redraw();
                }
            }
            WindowEvent::Focused(false) | WindowEvent::CursorLeft { .. } => {
                self.is_dragging = false;
                self.cursor = None;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let position = position.to_logical::<f64>(self.window.scale_factor());
                let cursor = (position.x, position.y);
                if let Some(previous) = self.cursor.filter(|_| self.is_dragging) {
                    self.camera
                        .pan((cursor.0 - previous.0, cursor.1 - previous.1));
                    self.update_camera()?;
                }
                self.cursor = Some(cursor);
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => self.is_dragging = state == ElementState::Pressed,
            WindowEvent::MouseWheel { delta, .. } => {
                if let Some(cursor) = self.cursor {
                    let lines = match delta {
                        MouseScrollDelta::LineDelta(_, lines) => f64::from(lines),
                        MouseScrollDelta::PixelDelta(pixels) => {
                            pixels.y / self.window.scale_factor() / 40.0
                        }
                    };
                    self.camera.zoom(cursor, lines);
                    self.update_camera()?;
                    self.refresh_overlay()?;
                }
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.physical_key == PhysicalKey::Code(KeyCode::Space)
                    && event.state == ElementState::Pressed
                    && !event.repeat =>
            {
                self.should_rebuild_each_redraw = !self.should_rebuild_each_redraw;
                self.stats = FrameStatistics::default();
                self.refresh_overlay()?;
            }
            _ => {}
        }
        Ok(())
    }
}

#[derive(Default)]
struct TigerExample {
    window: Option<TigerWindow>,
    error: Option<ExampleError>,
}

impl ApplicationHandler for TigerExample {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        match TigerWindow::new(event_loop) {
            Ok(window) => self.window = Some(window),
            Err(error) => {
                error!("Could not open tiger example: {error}");
                self.error = Some(error);
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.window.as_mut() else {
            return;
        };
        if window.window.id() != window_id {
            return;
        }
        if let Err(error) = window.handle_event(event_loop, event) {
            error!("Tiger example failed: {error}");
            self.error = Some(error);
            event_loop.exit();
        }
    }
}

fn main() -> Result<(), ExampleError> {
    env_logger::init();
    let event_loop = EventLoop::new()?;
    let mut example = TigerExample::default();
    event_loop.run_app(&mut example)?;
    match example.error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}
