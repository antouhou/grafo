use futures::executor::block_on;
use std::sync::Arc;
use std::time::Instant;
use lyon::geom::Size;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

pub fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("To create the event loop");
    let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
    let inner_logical_size = window.inner_size().to_logical::<f32>(window.scale_factor());

    let window_size = window.inner_size();
    let scale_factor = window.scale_factor();
    let physical_size = (window_size.width, window_size.height);

    let mut state = block_on(grafo::Renderer::new(
        window.clone(),
        physical_size,
        scale_factor,
    ));

    let _ = event_loop.run(move |event, event_loop_window_target| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => {
                match event {
                    WindowEvent::CloseRequested => event_loop_window_target.exit(),
                    WindowEvent::KeyboardInput {
                        device_id,
                        event,
                        is_synthetic,
                    } => {

                    }
                    WindowEvent::MouseInput {
                        device_id,
                        state,
                        button,
                    } => {

                    }
                    WindowEvent::CursorMoved {
                        device_id,
                        position,
                    } => {

                    }
                    WindowEvent::MouseWheel {
                        device_id,
                        delta,
                        phase,
                        ..
                    } => {

                    }
                    WindowEvent::Resized(physical_size) => {
                        let new_size = (physical_size.width, physical_size.height);
                        state.resize(new_size);

                        let logical_size = physical_size.to_logical::<f32>(window.scale_factor());

                        window.request_redraw();
                    }
                    WindowEvent::RedrawRequested => {
                        // {
                        //     let timer = Instant::now();
                        //     let mut_ref = &mut state;
                        //     app_runner.run_frame(mut_ref);
                        //     println!("Frame time: {:?}", timer.elapsed());
                        // }

                        // let red = Shape::rounded_rect(
                        //     Rect::new((0.0, 0.0).into(), (200.0, 200.0).into()),
                        //     BorderRadii::new(0.0),
                        //     Color32::from_rgb(255, 0, 0),
                        //     Stroke::new(1.0, Color32::from_rgb(0, 0, 0)),
                        // );
                        //
                        // let green = Shape::rounded_rect(
                        //     Rect::new((100.0, 100.0).into(), (300.0, 300.0).into()),
                        //     BorderRadii::new(0.0),
                        //     Color32::from_rgb(0, 255, 0),
                        //     Stroke::new(1.0, Color32::from_rgb(0, 0, 0)),
                        // );
                        //
                        // let blue = Shape::rounded_rect(
                        //     Rect::new((150.0, 150.0).into(), (350.0, 350.0).into()),
                        //     BorderRadii::new(10.0),
                        //     Color32::from_rgb(0, 0, 255),
                        //     Stroke::new(1.0, Color32::from_rgb(0, 0, 0)),
                        // );
                        //
                        // let yellow = Shape::rounded_rect(
                        //     Rect::new((0.0, 0.0).into(), (150.0, 150.0).into()),
                        //     BorderRadii::new(0.0),
                        //     Color32::from_rgb(255, 255, 0),
                        //     Stroke::new(1.0, Color32::from_rgb(0, 0, 0)),
                        // );
                        //
                        // let white = Shape::rounded_rect(
                        //     Rect::new((0.0, 0.0).into(), (20.0, 20.0).into()),
                        //     BorderRadii::new(0.0),
                        //     Color32::from_rgb(255, 255, 255),
                        //     Stroke::new(1.0, Color32::from_rgb(0, 0, 0)),
                        // );
                        //
                        // let shape_that_doesnt_fit = Shape::rounded_rect(
                        //     Rect::new((0.0, 0.0).into(), (20.0, 20.0).into()),
                        //     BorderRadii::new(0.0),
                        //     Color32::from_rgb(255, 255, 255),
                        //     Stroke::new(1.0, Color32::from_rgb(0, 0, 0)),
                        // );
                        //
                        // let red_id = state.add_shape(&red, None);
                        // let green_id = state.add_shape(&green, Some(red_id));
                        // let blue_id = state.add_shape(&blue, Some(green_id));
                        // state.add_shape(&yellow, Some(green_id));
                        // state.add_shape(&white, Some(red_id));
                        // state.add_shape(&shape_that_doesnt_fit, Some(blue_id));

                        // state.update();
                        let timer = Instant::now();
                        match state.render() {
                            Ok(_) => {
                                state.clear_draw_queue();
                            }
                            Err(wgpu::SurfaceError::Lost) => state.resize(state.size()),
                            Err(wgpu::SurfaceError::OutOfMemory) => event_loop_window_target.exit(),
                            Err(e) => eprintln!("{:?}", e),
                        }
                        println!("Render time: {:?}", timer.elapsed());
                    }
                    WindowEvent::ScaleFactorChanged {
                        scale_factor,
                        inner_size_writer,
                    } => {
                        // state.resize(PhysicalSize::new());
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    });
}
