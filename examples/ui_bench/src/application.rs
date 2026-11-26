use crate::animation::{Animation, RunStatus, Workload};
use crate::controls::{self, Action};
use crate::interface::{self, Interface};
use crate::overlay::{LiveTimings, Overlay};
use crate::statistics::Statistics;
use crate::BenchmarkError;
use grafo::{Scene, SceneContext, Surface, SurfaceError, WgpuBackendError};
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, MouseButton, StartCause, TouchPhase, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};
use winit::window::{Window, WindowId};

fn scale_factor(size: PhysicalSize<u32>) -> f32 {
    (size.width as f32 / interface::WIDTH)
        .min(size.height as f32 / interface::HEIGHT)
        .max(0.01)
}

fn report(statistics: &Statistics, vsync_requested: bool, size: (u32, u32)) {
    for (workload, sample) in Workload::ALL.into_iter().zip(&statistics.workloads) {
        if sample.samples != 0 {
            tracing::info!(
                workload = workload.label(),
                vsync_requested,
                width = size.0,
                height = size.1,
                apps_per_page = interface::APPS_PER_PAGE,
                samples = sample.samples,
                fps = sample.fps(),
                mean_ms = sample.mean_ms(),
                p95_ms = %sample.p95_ms(),
                max_ms = sample.max_ms(),
                "ui_bench"
            );
        }
    }
}

struct BenchmarkWindow {
    window: Arc<Window>,
    surface: Surface,
    scene: Scene,
    interface: Interface,
    overlay: Overlay,
    animation: Animation,
    statistics: Statistics,
    live: LiveTimings,
    report_started: Instant,
    cursor: PhysicalPosition<f32>,
    is_occluded: bool,
}

impl BenchmarkWindow {
    fn new(event_loop: &ActiveEventLoop) -> Result<Self, BenchmarkError> {
        let window = Arc::new(
            event_loop.create_window(
                Window::default_attributes()
                    .with_title("Grafo - UI bench")
                    .with_inner_size(PhysicalSize::new(1280, 720))
                    .with_min_inner_size(PhysicalSize::new(640, 360)),
            )?,
        );
        let size = window.inner_size();
        let dimensions = (size.width.max(1), size.height.max(1));
        let (context, surface) = pollster::block_on(SceneContext::try_new_with_surface(
            event_loop.owned_display_handle(),
            window.clone(),
            dimensions,
            false,
            false,
        ))?;
        let mut scene = Scene::try_new(context, dimensions, scale_factor(size).into(), 1)?;
        let (mut interface, mut artwork) = Interface::new(&mut scene)?;
        let mut overlay = Overlay::new(&mut scene, &mut interface.fonts, &mut artwork)?;
        let now = Instant::now();
        let mut animation = Animation::new(now);
        let statistics = Statistics::default();
        let live = LiveTimings::default();
        overlay.refresh(
            &mut scene,
            &statistics,
            &live,
            &animation.sample(now),
            &mut interface,
            false,
        )?;
        window.request_redraw();
        Ok(Self {
            window,
            surface,
            scene,
            interface,
            overlay,
            animation,
            statistics,
            live,
            report_started: now,
            cursor: PhysicalPosition::new(0.0, 0.0),
            is_occluded: false,
        })
    }

    fn start(&mut self) {
        let now = Instant::now();
        self.statistics = Statistics::default();
        self.live = LiveTimings::default();
        self.animation = Animation::new(now);
        self.report_started = now;
    }

    fn resize(&mut self, size: PhysicalSize<u32>) -> Result<(), BenchmarkError> {
        self.surface.resize((size.width, size.height));
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }
        self.scene.resize((size.width, size.height));
        self.scene.change_scale_factor(scale_factor(size).into())?;
        if self.animation.is_running() {
            self.start();
        }
        self.window.request_redraw();
        Ok(())
    }

    fn redraw(&mut self, event_loop: &ActiveEventLoop) -> Result<(), BenchmarkError> {
        if self.is_occluded || self.surface.size().0 == 0 || self.surface.size().1 == 0 {
            event_loop.set_control_flow(ControlFlow::Wait);
            return Ok(());
        }
        let started = Instant::now();
        let was_running = self.animation.is_running();
        let sample = self.animation.sample(started);
        let has_completed = was_running && sample.status == RunStatus::Complete;
        let measurement = self
            .statistics
            .measure_motion(sample.workload, sample.is_warming_up);
        let is_settling = sample.workload.is_none() && measurement.is_some();
        self.interface.update(&mut self.scene, sample.pose)?;
        if measurement.is_none() {
            self.overlay.refresh(
                &mut self.scene,
                &self.statistics,
                &self.live,
                &sample,
                &mut self.interface,
                self.surface.vsync(),
            )?;
        }
        let render_started = Instant::now();
        self.window.pre_present_notify();
        let result = match self.scene.render(&mut self.surface) {
            Err(WgpuBackendError::Surface(SurfaceError::Outdated)) => {
                self.scene.render(&mut self.surface)
            }
            Err(WgpuBackendError::Surface(SurfaceError::Lost)) => {
                self.surface.recreate(self.window.clone())?;
                self.scene.render(&mut self.surface)
            }
            result => result,
        };
        let completed = Instant::now();
        let was_rendered = match result {
            Ok(()) => {
                if measurement.is_some() {
                    self.live.update = render_started.duration_since(started);
                    self.live.render = completed.duration_since(render_started);
                }
                true
            }
            Err(WgpuBackendError::Surface(SurfaceError::Timeout | SurfaceError::Outdated)) => false,
            Err(WgpuBackendError::Surface(SurfaceError::Occluded)) => {
                self.statistics.break_sequence();
                tracing::warn!("Surface unavailable; waiting for the next window event");
                event_loop.set_control_flow(ControlFlow::Wait);
                return Ok(());
            }
            Err(error) => return Err(error.into()),
        };
        let (workload, is_warming_up) = measurement
            .map_or((None, sample.is_warming_up), |(workload, warming)| {
                (Some(workload), warming)
            });
        self.statistics
            .record(workload, is_warming_up, started, completed, was_rendered);
        if has_completed
            || (self.animation.is_running()
                && completed.duration_since(self.report_started) >= Duration::from_secs(2))
        {
            report(&self.statistics, self.surface.vsync(), self.scene.size());
            self.report_started = completed;
        }
        if sample.workload.is_some() || is_settling {
            event_loop.set_control_flow(ControlFlow::Wait);
            self.window.request_redraw();
        } else if let Some(duration) = sample.next_transition {
            event_loop.set_control_flow(ControlFlow::WaitUntil(started + duration));
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
        }
        Ok(())
    }

    fn apply_action(
        &mut self,
        action: Action,
        event_loop: &ActiveEventLoop,
    ) -> Result<(), BenchmarkError> {
        match action {
            Action::Start => self.start(),
            Action::Stop => {
                let was_running = self.animation.is_running();
                self.animation.stop(Instant::now());
                self.statistics.break_sequence();
                if was_running {
                    report(&self.statistics, self.surface.vsync(), self.scene.size());
                }
            }
            Action::Exit => event_loop.exit(),
        }
        self.window.request_redraw();
        Ok(())
    }

    fn pointer_released(
        &mut self,
        event_loop: &ActiveEventLoop,
        position: PhysicalPosition<f32>,
    ) -> Result<(), BenchmarkError> {
        let scale = scale_factor(self.window.inner_size());
        let x = position.x / scale;
        let y = position.y / scale;
        if let Some(action) = controls::pointer_action([x, y]) {
            return self.apply_action(action, event_loop);
        }
        self.window.request_redraw();
        Ok(())
    }

    fn event(
        &mut self,
        event_loop: &ActiveEventLoop,
        event: WindowEvent,
    ) -> Result<(), BenchmarkError> {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => self.redraw(event_loop)?,
            WindowEvent::Resized(size) => self.resize(size)?,
            WindowEvent::ScaleFactorChanged { .. } => self.resize(self.window.inner_size())?,
            WindowEvent::Occluded(occluded) => {
                self.is_occluded = occluded;
                self.statistics.break_sequence();
                if occluded {
                    event_loop.set_control_flow(ControlFlow::Wait);
                } else {
                    if self.animation.is_running() {
                        self.start();
                    }
                    self.window.request_redraw();
                }
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed && !event.repeat =>
            {
                let code = match (event.physical_key, event.logical_key) {
                    (PhysicalKey::Code(code), _) => Some(code),
                    (_, Key::Named(NamedKey::BrowserBack)) => Some(KeyCode::BrowserBack),
                    _ => None,
                };
                if let Some(action) = code.and_then(controls::key_action) {
                    self.apply_action(action, event_loop)?;
                }
            }
            WindowEvent::CursorMoved { position, .. } => self.cursor = position.cast(),
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => self.pointer_released(event_loop, self.cursor)?,
            WindowEvent::Touch(touch) if touch.phase == TouchPhase::Ended => {
                self.pointer_released(event_loop, touch.location.cast())?;
            }
            _ => {}
        }
        Ok(())
    }
}

#[derive(Default)]
pub struct BenchmarkApplication {
    window: Option<BenchmarkWindow>,
    error: Option<BenchmarkError>,
}

impl BenchmarkApplication {
    pub fn finish(self) -> Result<(), BenchmarkError> {
        if let Some(window) = self.window {
            report(
                &window.statistics,
                window.surface.vsync(),
                window.scene.size(),
            );
        }
        match self.error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn fail(&mut self, event_loop: &ActiveEventLoop, error: BenchmarkError) {
        tracing::error!("UI bench failed: {error}");
        self.error = Some(error);
        event_loop.exit();
    }
}

impl ApplicationHandler for BenchmarkApplication {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            match BenchmarkWindow::new(event_loop) {
                Ok(window) => self.window = Some(window),
                Err(error) => self.fail(event_loop, error),
            }
        }
    }

    fn suspended(&mut self, event_loop: &ActiveEventLoop) {
        // Android requires every surface reference to be dropped before suspension returns.
        event_loop.set_control_flow(ControlFlow::Wait);
        self.window = None;
    }

    fn new_events(&mut self, event_loop: &ActiveEventLoop, cause: StartCause) {
        if matches!(cause, StartCause::ResumeTimeReached { .. }) {
            event_loop.set_control_flow(ControlFlow::Wait);
            if let Some(window) = &self.window {
                window.window.request_redraw();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let Some(window) = &mut self.window else {
            return;
        };
        if window.window.id() != id {
            return;
        }
        if let Err(error) = window.event(event_loop, event) {
            self.fail(event_loop, error);
        }
    }
}
