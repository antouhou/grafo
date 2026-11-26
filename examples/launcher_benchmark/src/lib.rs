use application::BenchmarkApplication;
use grafo::{DrawCommandError, EffectError, SceneCreationError, SceneError, WgpuBackendError};
use thiserror::Error;
use tracing_subscriber::EnvFilter;
#[cfg(target_os = "android")]
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use winit::error::{EventLoopError, OsError};
use winit::event_loop::EventLoop;
#[cfg(target_os = "android")]
use winit::platform::android::{activity::AndroidApp, EventLoopBuilderExtAndroid};

mod animation;
mod application;
mod artwork;
mod controls;
mod font;
mod interface;
mod overlay;
mod settings;
mod shadow;
mod statistics;

#[derive(Debug, Error)]
pub enum BenchmarkError {
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
    Effect(#[from] EffectError<WgpuBackendError>),
    #[error(transparent)]
    Render(#[from] WgpuBackendError),
}

fn run_event_loop(event_loop: EventLoop<()>) -> Result<(), BenchmarkError> {
    let mut application = BenchmarkApplication::default();
    event_loop.run_app(&mut application)?;
    application.finish()
}

pub fn run() -> Result<(), BenchmarkError> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("warn,grafo_launcher_benchmark=info")),
        )
        .try_init();
    run_event_loop(EventLoop::new()?)
}

#[cfg(target_os = "android")]
#[no_mangle]
pub fn android_main(android_app: AndroidApp) {
    if let Ok(layer) = tracing_android::layer("grafo-benchmark") {
        let _ = tracing_subscriber::registry()
            .with(EnvFilter::new("warn,grafo_launcher_benchmark=info"))
            .with(layer)
            .try_init();
    }
    let result = EventLoop::builder()
        .with_android_app(android_app)
        .build()
        .map_err(BenchmarkError::from)
        .and_then(run_event_loop);
    if let Err(error) = result {
        tracing::error!("Launcher benchmark failed: {error}");
    }
}
