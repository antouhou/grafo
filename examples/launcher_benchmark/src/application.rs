use crate::animation::{Animation, Scenario, Workload};
use crate::interface::{self, Interface};
use crate::overlay::{LiveTimings, Overlay};
use crate::settings;
use crate::statistics::Statistics;
use crate::BenchmarkError;
use grafo::{Scene, SceneContext, Surface, SurfaceError, WgpuBackendError};
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{
    ElementState, MouseButton, MouseScrollDelta, StartCause, TouchPhase, WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};
use winit::window::{Fullscreen, Window, WindowId};

fn scale_factor(size: PhysicalSize<u32>) -> f64 {
    (f64::from(size.width) / f64::from(interface::WIDTH))
        .min(f64::from(size.height) / f64::from(interface::HEIGHT))
        .max(0.01)
}

fn report(statistics: &Statistics, vsync: bool, size: (u32, u32), apps_per_page: usize) {
    for (workload, sample) in Workload::ALL.into_iter().zip(&statistics.workloads) {
        if sample.samples != 0 {
            tracing::info!(
                workload = workload.label(),
                vsync,
                width = size.0,
                height = size.1,
                apps_per_page,
                samples = sample.samples,
                fps = sample.fps(),
                mean_ms = sample.mean_ms(),
                p95_ms = sample.p95_ms(),
                max_ms = sample.max_ms(),
                "launcher_benchmark"
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
    cursor: PhysicalPosition<f64>,
    pointer_start: Option<PhysicalPosition<f64>>,
    is_dense: bool,
    is_occluded: bool,
}

impl BenchmarkWindow {
    fn new(event_loop: &ActiveEventLoop) -> Result<Self, BenchmarkError> {
        let window = Arc::new(
            event_loop.create_window(
                Window::default_attributes()
                    .with_title("Grafo - Launcher benchmark")
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
        let mut scene = Scene::try_new(context, dimensions, scale_factor(size), 1)?;
        let (mut interface, mut artwork) = Interface::new(&mut scene, false)?;
        let mut overlay = Overlay::new(&mut scene, &mut interface.fonts, &mut artwork)?;
        let now = Instant::now();
        let animation = Animation::new(now);
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
            pointer_start: None,
            is_dense: false,
            is_occluded: false,
        })
    }

    fn reset(&mut self) {
        let now = Instant::now();
        self.statistics = Statistics::default();
        self.animation.select(self.animation.scenario(), now);
        self.report_started = now;
    }

    fn select(&mut self, scenario: Scenario) {
        self.animation.select(scenario, Instant::now());
        self.reset();
    }

    fn resize(&mut self, size: PhysicalSize<u32>) -> Result<(), BenchmarkError> {
        self.surface.resize((size.width, size.height));
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }
        self.scene.resize((size.width, size.height));
        self.scene.change_scale_factor(scale_factor(size))?;
        self.reset();
        self.window.request_redraw();
        Ok(())
    }

    fn change_density(&mut self) -> Result<(), BenchmarkError> {
        self.is_dense = !self.is_dense;
        self.scene.clear();
        let (mut interface, mut artwork) = Interface::new(&mut self.scene, self.is_dense)?;
        self.overlay = Overlay::new(&mut self.scene, &mut interface.fonts, &mut artwork)?;
        self.interface = interface;
        self.reset();
        Ok(())
    }

    fn toggle_vsync(&mut self) {
        self.surface.set_vsync(!self.surface.vsync());
        self.reset();
    }

    fn redraw(&mut self, event_loop: &ActiveEventLoop) -> Result<(), BenchmarkError> {
        if self.is_occluded || self.surface.size().0 == 0 || self.surface.size().1 == 0 {
            event_loop.set_control_flow(ControlFlow::Wait);
            return Ok(());
        }
        let started = Instant::now();
        let sample = self.animation.sample(started);
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
        if completed.duration_since(self.report_started) >= Duration::from_secs(2) {
            report(
                &self.statistics,
                self.surface.vsync(),
                self.scene.size(),
                self.interface.apps_per_page,
            );
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

    fn key(&mut self, code: KeyCode, event_loop: &ActiveEventLoop) -> Result<(), BenchmarkError> {
        match code {
            KeyCode::KeyA => self.select(Scenario::Mixed),
            KeyCode::KeyP => self.select(Scenario::Pages),
            KeyCode::KeyS => self.select(Scenario::Settings),
            KeyCode::KeyC => self.select(Scenario::Scroll),
            KeyCode::KeyO => self.select(Scenario::Power),
            KeyCode::KeyR => self.reset(),
            KeyCode::KeyV => self.toggle_vsync(),
            KeyCode::KeyD => self.change_density()?,
            KeyCode::Space => {
                self.animation.toggle_pause(Instant::now());
                self.statistics.break_sequence();
            }
            KeyCode::ArrowDown => self.animation.scroll_by(0.5, Instant::now()),
            KeyCode::ArrowUp => self.animation.scroll_by(-0.5, Instant::now()),
            KeyCode::ArrowLeft => self.animation.change_page(-1.0, Instant::now()),
            KeyCode::ArrowRight => self.animation.change_page(1.0, Instant::now()),
            KeyCode::Enter => self.animation.toggle_settings(Instant::now()),
            KeyCode::KeyF => self
                .window
                .set_fullscreen(if self.window.fullscreen().is_some() {
                    None
                } else {
                    Some(Fullscreen::Borderless(None))
                }),
            KeyCode::Escape | KeyCode::BrowserBack => {
                let pose = self.animation.sample(Instant::now()).pose;
                if pose.power > 0.0 {
                    self.animation.close_power(Instant::now());
                } else if pose.settings > 0.0 {
                    self.animation.close_settings(Instant::now());
                } else {
                    event_loop.exit();
                }
            }
            _ => return Ok(()),
        }
        self.window.request_redraw();
        Ok(())
    }

    fn pointer_released(&mut self, position: PhysicalPosition<f64>) -> Result<(), BenchmarkError> {
        let scale = scale_factor(self.window.inner_size());
        let x = position.x / scale;
        let y = position.y / scale;
        if let Some(start) = self.pointer_start.take() {
            let distance = (position.x - start.x) / scale;
            let vertical = (position.y - start.y) / scale;
            let pose = self.animation.sample(Instant::now()).pose;
            let is_inside_scroll_view = (f64::from(settings::X)
                ..=f64::from(settings::X + settings::WIDTH))
                .contains(&(start.x / scale))
                && (f64::from(settings::SCROLL_TOP)..f64::from(settings::Y + settings::HEIGHT))
                    .contains(&(start.y / scale));
            if pose.settings == 1.0
                && pose.power == 0.0
                && is_inside_scroll_view
                && vertical.abs() > 25.0
            {
                self.animation
                    .scroll_by(-vertical as f32 / settings::SCROLL_DISTANCE, Instant::now());
                self.window.request_redraw();
                return Ok(());
            }
            if distance.abs() > 50.0 {
                self.animation
                    .change_page(-distance.signum() as f32, Instant::now());
                self.window.request_redraw();
                return Ok(());
            }
        }
        if (34.0..=82.0).contains(&y) {
            match x as u32 {
                408..=487 => self.select(Scenario::Mixed),
                496..=575 => self.select(Scenario::Pages),
                584..=663 => self.select(Scenario::Settings),
                672..=751 => self.select(Scenario::Scroll),
                760..=839 => self.select(Scenario::Power),
                848..=927 => {
                    self.animation.toggle_pause(Instant::now());
                    self.statistics.break_sequence();
                }
                936..=1015 => self.change_density()?,
                1068..=1228 => self.animation.toggle_settings(Instant::now()),
                _ => {}
            }
        } else if (114.0..=158.0).contains(&y) && (882.0..=1204.0).contains(&x) {
            let pose = self.animation.sample(Instant::now()).pose;
            if pose.settings == 1.0 && pose.power > 0.0 {
                self.animation.close_power(Instant::now());
            } else if pose.settings == 1.0 && x >= 1160.0 {
                self.animation.toggle_power(Instant::now());
            }
        } else if y >= 580.0 && x >= 645.0 {
            self.toggle_vsync();
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
                    self.reset();
                    self.window.request_redraw();
                }
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed && !event.repeat =>
            {
                let code = match (event.physical_key, event.logical_key) {
                    (PhysicalKey::Code(code), _) => Some(code),
                    (_, Key::Named(NamedKey::Enter)) => Some(KeyCode::Enter),
                    (_, Key::Named(NamedKey::BrowserBack)) => Some(KeyCode::BrowserBack),
                    _ => None,
                };
                if let Some(code) = code {
                    self.key(code, event_loop)?;
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scale = scale_factor(self.window.inner_size());
                let pose = self.animation.sample(Instant::now()).pose;
                if pose.settings == 1.0
                    && pose.power == 0.0
                    && (f64::from(settings::X)..=f64::from(settings::X + settings::WIDTH))
                        .contains(&(self.cursor.x / scale))
                    && self.cursor.y / scale >= f64::from(settings::SCROLL_TOP)
                    && self.cursor.y / scale < f64::from(settings::Y + settings::HEIGHT)
                {
                    let distance = match delta {
                        MouseScrollDelta::LineDelta(_, y) => -y * 0.25,
                        MouseScrollDelta::PixelDelta(position) => {
                            -position.y as f32 / scale as f32 / settings::SCROLL_DISTANCE
                        }
                    };
                    self.animation.scroll_by(distance, Instant::now());
                    self.window.request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } => self.cursor = position,
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                if state == ElementState::Pressed {
                    self.pointer_start = Some(self.cursor);
                } else {
                    self.pointer_released(self.cursor)?;
                }
            }
            WindowEvent::Touch(touch) => match touch.phase {
                TouchPhase::Started => self.pointer_start = Some(touch.location),
                TouchPhase::Ended => self.pointer_released(touch.location)?,
                TouchPhase::Cancelled => self.pointer_start = None,
                TouchPhase::Moved => {}
            },
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
                window.interface.apps_per_page,
            );
        }
        match self.error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn fail(&mut self, event_loop: &ActiveEventLoop, error: BenchmarkError) {
        tracing::error!("Launcher benchmark failed: {error}");
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
