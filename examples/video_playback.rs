use std::fs;
use std::path::Path;
use futures::executor::block_on;
use grafo::{fontdb, Color, FontFamily, Stroke};
use grafo::{MathRect, Shape, TextAlignment, TextLayout};
use std::sync::Arc;
use std::time::Instant;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;
use dextreamer;
use dextreamer::VideoStreamEvent;

const TEST_VIDEO_FILE_PATH: &str = "examples/assets/sintel_trailer-480p.mkv";
pub fn main() {
    let test_video_relative_path = Path::new(TEST_VIDEO_FILE_PATH);
    let test_video_absolute_path =
        fs::canonicalize(test_video_relative_path).expect("Test video file to exist");
    let test_video_uri = format!(
        "file://{}",
        test_video_absolute_path
            .to_string_lossy()
            .to_string()
            .replace('\\', "/")
    );

    env_logger::init();
    let event_loop = EventLoop::new().expect("To create the event loop");
    let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());

    let window_size = window.inner_size();
    let scale_factor = window.scale_factor();
    let physical_size = (window_size.width, window_size.height);

    let mut renderer = block_on(grafo::Renderer::new(
        window.clone(),
        physical_size,
        scale_factor,
    ));

    let video_size = (848, 360);
    /// Test id - you should use something else to uniquely identify video texture with
    let video_texture_id = 123;
    renderer.texture_manager().allocate_texture(123, video_size);

    let texture_manager_for_video = renderer.texture_manager().clone();

    let window_clone = window.clone();
    let (video_sender, video_receiver) = dextreamer::open_video(test_video_uri);

    // Start video receiver thread handler
    std::thread::spawn(move || {
        let mut video_receiver = video_receiver;
        loop {
            for video_event in video_receiver.iter() {
                match video_event {
                    VideoStreamEvent::VideoLoaded(_) => {}
                    VideoStreamEvent::NewFrame(data) => {
                        texture_manager_for_video.load_data_into_texture(video_texture_id, &data.data, video_size).expect("To load video frame into texture");
                        window_clone.request_redraw();
                    }
                    VideoStreamEvent::Error(_) => {}
                    VideoStreamEvent::CurrentAudioTrackChanged(_) => {}
                    VideoStreamEvent::CurrentSubtitleTrackChanged(_) => {}
                    VideoStreamEvent::VolumeChanged(_) => {}
                    VideoStreamEvent::PlayingStateChanged(_) => {}
                    VideoStreamEvent::PositionChanged(_) => {}
                    VideoStreamEvent::Closed => {}
                }
            }
        }
    });

    let _ = event_loop.run(move |event, event_loop_window_target| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => match event {
            WindowEvent::CloseRequested => event_loop_window_target.exit(),
            WindowEvent::Resized(physical_size) => {
                let new_size = (physical_size.width, physical_size.height);
                renderer.resize(new_size);

                window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                let timer = Instant::now();

                renderer.add_texture_draw_to_queue(
                    video_texture_id,
                    (848, 360),
                    [(0.0, 0.0), (848.0, 360.0)],
                    None
                );

                match renderer.render() {
                    Ok(_) => {
                        renderer.clear_draw_queue();
                    }
                    Err(wgpu::SurfaceError::Lost) => renderer.resize(renderer.size()),
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop_window_target.exit(),
                    Err(e) => eprintln!("{:?}", e),
                }
                println!("Render time: {:?}", timer.elapsed());
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                renderer.change_scale_factor(*scale_factor);
            }
            _ => {}
        },
        _ => {}
    });
}
