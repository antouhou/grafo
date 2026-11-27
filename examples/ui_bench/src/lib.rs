use application::BenchmarkApplication;
use grafo::{DrawCommandError, EffectError, SceneCreationError, SceneError, WgpuBackendError};
use serde_json::Error as JsonError;
use std::path::PathBuf;
use std::{env, fs, io};
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
    #[error("Benchmark results I/O failed: {0}")]
    ResultsIo(#[from] io::Error),
    #[error("Could not serialize benchmark results: {0}")]
    ResultsJson(#[from] JsonError),
    #[cfg(target_os = "android")]
    #[error("Android did not provide an external data directory for benchmark results")]
    ResultsDirectoryUnavailable,
}

fn run_event_loop(
    event_loop: EventLoop<()>,
    results_directory: PathBuf,
) -> Result<(), BenchmarkError> {
    fs::create_dir_all(&results_directory)?;
    let mut application = BenchmarkApplication::new(event_loop.create_proxy(), results_directory);
    event_loop.run_app(&mut application)?;
    application.finish()
}

pub fn run() -> Result<(), BenchmarkError> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("warn,ui_bench=info")),
        )
        .try_init();
    run_event_loop(
        EventLoop::new()?,
        env::current_dir()?.join("ui-bench-results"),
    )
}

#[cfg(target_os = "android")]
#[no_mangle]
pub fn android_main(android_app: AndroidApp) {
    if let Ok(layer) = tracing_android::layer("ui-bench") {
        let _ = tracing_subscriber::registry()
            .with(EnvFilter::new("warn,ui_bench=info"))
            .with(layer)
            .try_init();
    }
    let result = android_app
        .external_data_path()
        .ok_or(BenchmarkError::ResultsDirectoryUnavailable)
        .and_then(|directory| {
            let event_loop = EventLoop::builder().with_android_app(android_app).build()?;
            run_event_loop(event_loop, directory.join("ui-bench-results"))
        });
    match result {
        Ok(()) => tracing::info!("UI_BENCH_EXITED"),
        Err(error) => tracing::error!("UI_BENCH_FAILED {error}"),
    }
}
