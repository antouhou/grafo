// use dextreamer::VideoStreamEvent;
// use futures::executor::block_on;
// use grafo::TextureManager;
// use std::fs;
// use std::path::Path;
// use std::sync::Arc;
// use winit::event::{Event, WindowEvent};
// use winit::event_loop::EventLoop;
// use winit::window::WindowBuilder;
//
// struct VideoFrameLoader(TextureManager);
//
// impl dextreamer::FrameHandler for VideoFrameLoader {
//     fn handle_new_frame(&self, frame_data: &[u8], frame_size: (u32, u32)) {
//         let texture_manager = &self.0;
//         let texture_id = 123;
//         texture_manager
//             .load_data_into_texture(texture_id, frame_data, frame_size)
//             .expect("To load frame data into texture");
//     }
// }
//
// const TEST_VIDEO_FILE_PATH: &str = "examples/assets/sintel_trailer-480p.mkv";
pub fn main() {
    //     let test_video_relative_path = Path::new(TEST_VIDEO_FILE_PATH);
    //     let test_video_absolute_path =
    //         fs::canonicalize(test_video_relative_path).expect("Test video file to exist");
    //     let test_video_uri = format!(
    //         "file://{}",
    //         test_video_absolute_path
    //             .to_string_lossy()
    //             .to_string()
    //             .replace('\\', "/")
    //     );
    //
    //     env_logger::init();
    //     let event_loop = EventLoop::new().expect("To create the event loop");
    //     let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
    //
    //     let window_size = window.inner_size();
    //     let scale_factor = window.scale_factor();
    //     let physical_size = (window_size.width, window_size.height);
    //
    //     let mut renderer = block_on(grafo::Renderer::new(
    //         window.clone(),
    //         physical_size,
    //         scale_factor,
    //     ));
    //
    //     let video_size = (848, 360);
    //     // Test id - you should use something else to uniquely identify video texture with
    //     let video_texture_id = 123;
    //     renderer.texture_manager().allocate_texture(123, video_size);
    //
    //     let texture_manager_for_video = renderer.texture_manager().clone();
    //     let frame_loader = VideoFrameLoader(texture_manager_for_video);
    //
    //     let window_clone = window.clone();
    //     let (video_sender, video_receiver) = dextreamer::open_video(test_video_uri, frame_loader);
    //
    //     // Start video receiver thread handler
    //     let _ = std::thread::spawn(move || {
    //         let video_receiver = video_receiver;
    //         loop {
    //             for video_event in video_receiver.iter() {
    //                 match video_event {
    //                     VideoStreamEvent::VideoLoaded(_) => {}
    //                     VideoStreamEvent::NewFrame => {
    //                         window_clone.request_redraw();
    //                     }
    //                     VideoStreamEvent::Error(_) => {}
    //                     VideoStreamEvent::CurrentAudioTrackChanged(_) => {}
    //                     VideoStreamEvent::CurrentSubtitleTrackChanged(_) => {}
    //                     VideoStreamEvent::VolumeChanged(_) => {}
    //                     VideoStreamEvent::PlayingStateChanged(_) => {}
    //                     VideoStreamEvent::PositionChanged(_) => {}
    //                     VideoStreamEvent::Closed => {
    //                         println!("Video closed");
    //                     }
    //                 }
    //             }
    //         }
    //     });
    //
    //     let _ = event_loop.run(move |event, event_loop_window_target| match event {
    //         Event::WindowEvent {
    //             ref event,
    //             window_id,
    //         } if window_id == window.id() => match event {
    //             WindowEvent::CloseRequested => {
    //                 video_sender
    //                     .send(dextreamer::VideoStreamAction::Close)
    //                     .expect("To send close command to video stream");
    //                 event_loop_window_target.exit();
    //             }
    //             WindowEvent::Resized(physical_size) => {
    //                 let new_size = (physical_size.width, physical_size.height);
    //                 renderer.resize(new_size);
    //
    //                 window.request_redraw();
    //             }
    //             WindowEvent::RedrawRequested => {
    //                 renderer.add_texture_draw_to_queue(
    //                     video_texture_id,
    //                     [(0.0, 0.0), (video_size.0 as f32, video_size.1 as f32)],
    //                     None,
    //                 );
    //
    //                 match renderer.render() {
    //                     Ok(_) => {
    //                         renderer.clear_draw_queue();
    //                     }
    //                     Err(wgpu::SurfaceError::Lost) => renderer.resize(renderer.size()),
    //                     Err(wgpu::SurfaceError::OutOfMemory) => event_loop_window_target.exit(),
    //                     Err(e) => eprintln!("{:?}", e),
    //                 }
    //             }
    //             WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
    //                 renderer.change_scale_factor(*scale_factor);
    //             }
    //             _ => {}
    //         },
    //         _ => {}
    //     });
}
