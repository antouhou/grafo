use futures::executor::block_on;
use grafo::math::Box2D;
use grafo::path::builder::BorderRadii;
use grafo::{Color, Shape, Stroke};
use std::sync::Arc;
use std::time::Instant;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

use image::io::Reader as ImageReader;

pub fn main() {
    let rust_logo_png_bytes = include_bytes!("./rust-logo-256x256-blk.png");
    let rust_logo_png = ImageReader::new(std::io::Cursor::new(rust_logo_png_bytes))
        .with_guessed_format()
        .unwrap()
        .decode()
        .unwrap();
    let rust_logo_rgba = rust_logo_png.as_rgba8().unwrap();
    let rust_logo_png_dimensions = rust_logo_rgba.dimensions();
    let rust_logo_png_dimensions_f32 = (
        rust_logo_png_dimensions.0 as f32,
        rust_logo_png_dimensions.1 as f32,
    );
    let rust_logo_png_bytes = rust_logo_rgba.to_vec();

    env_logger::init();
    let event_loop = EventLoop::new().expect("To create the event loop");
    let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
    let inner_logical_size = window.inner_size().to_logical::<f32>(window.scale_factor());

    let window_size = window.inner_size();
    let scale_factor = window.scale_factor();
    let physical_size = (window_size.width, window_size.height);

    // let rust_png_logo__dimensions_adjusted_for_scale_factor = (
    //     (rust_logo_png_dimensions.0 as f64 / scale_factor) as f32,
    //     (rust_logo_png_dimensions.1 as f64 / scale_factor) as f32,
    // );

    let mut renderer = block_on(grafo::Renderer::new(
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
                    } => {}
                    WindowEvent::MouseInput {
                        device_id,
                        state,
                        button,
                    } => {}
                    WindowEvent::CursorMoved {
                        device_id,
                        position,
                    } => {}
                    WindowEvent::MouseWheel {
                        device_id,
                        delta,
                        phase,
                        ..
                    } => {}
                    WindowEvent::Resized(physical_size) => {
                        let new_size = (physical_size.width, physical_size.height);
                        renderer.resize(new_size);

                        let logical_size = physical_size.to_logical::<f32>(window.scale_factor());

                        window.request_redraw();
                    }
                    WindowEvent::RedrawRequested => {
                        let background = Shape::rect(
                            [
                                (0.0, 0.0),
                                (window_size.width as f32, window_size.height as f32),
                            ]
                            .into(),
                            Color::rgb(255, 255, 255),
                            Stroke::new(1.0, Color::rgb(0, 0, 0)),
                        );

                        let red = Shape::rounded_rect(
                            &Box2D::new((0.0, 0.0).into(), (200.0, 200.0).into()),
                            &BorderRadii::new(0.0),
                            Color::rgb(255, 0, 0),
                            Stroke::new(1.0, Color::rgb(0, 0, 0)),
                        );

                        let green = Shape::rounded_rect(
                            &Box2D::new((100.0, 100.0).into(), (300.0, 300.0).into()),
                            &BorderRadii::new(0.0),
                            Color::rgb(0, 255, 0),
                            Stroke::new(1.0, Color::rgb(0, 0, 0)),
                        );

                        let blue = Shape::rounded_rect(
                            &Box2D::new((150.0, 150.0).into(), (350.0, 350.0).into()),
                            &BorderRadii::new(10.0),
                            Color::rgb(0, 0, 255),
                            Stroke::new(1.0, Color::rgb(0, 0, 0)),
                        );

                        let yellow = Shape::rounded_rect(
                            &Box2D::new((0.0, 0.0).into(), (150.0, 150.0).into()),
                            &BorderRadii::new(0.0),
                            Color::rgb(255, 255, 0),
                            Stroke::new(1.0, Color::rgb(0, 0, 0)),
                        );

                        let white = Shape::rounded_rect(
                            &Box2D::new((0.0, 0.0).into(), (20.0, 20.0).into()),
                            &BorderRadii::new(0.0),
                            Color::rgb(255, 255, 255),
                            Stroke::new(1.0, Color::rgb(0, 0, 0)),
                        );

                        let shape_that_doesnt_fit = Shape::rounded_rect(
                            &Box2D::new((0.0, 0.0).into(), (20.0, 20.0).into()),
                            &BorderRadii::new(0.0),
                            Color::rgb(255, 255, 255),
                            Stroke::new(1.0, Color::rgb(0, 0, 0)),
                        );

                        let background_id = renderer.add_shape(background, None);
                        let red_id = renderer.add_shape(red, Some(background_id));
                        let green_id = renderer.add_shape(green, Some(red_id));
                        let blue_id = renderer.add_shape(blue, Some(green_id));
                        renderer.add_shape(yellow, Some(green_id));
                        renderer.add_shape(white, Some(red_id));
                        renderer.add_shape(shape_that_doesnt_fit, Some(blue_id));

                        renderer.add_image(
                            &rust_logo_png_bytes,
                            rust_logo_png_dimensions,
                            Box2D::from_origin_and_size(
                                (100.0, 100.0).into(),
                                rust_logo_png_dimensions_f32.into(),
                            ),
                            Some(red_id),
                        );

                        renderer.add_image(
                            &rust_logo_png_bytes,
                            rust_logo_png_dimensions,
                            Box2D::from_origin_and_size(
                                (200.0, 200.0).into(),
                                rust_logo_png_dimensions_f32.into(),
                            ),
                            Some(background_id),
                        );

                        let timer = Instant::now();
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
