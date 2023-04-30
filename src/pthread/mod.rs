use rand::Rng;
use sdl2::event::Event;
use sdl2::pixels::Color;

use crate::geometry::Body;
use crate::global;
use crate::global::{MASS_RANGE, RADIUS};
use crate::pthread::pool::*;
use crate::quad_tree::node::QuadNode;
use std::f64::EPSILON;
use std::sync::Arc;

pub mod pool;

fn generate_body_wrappers() -> (Vec<BodyWrapper>, Arc<QuadNode>) {
    let mut body_wrappers = Vec::new();
    let mut rng = rand::thread_rng();
    // let mut root = pool::new_root();
    let root = global::new_root();

    for _ in 0..*global::SIZE {
        let body = Body::new(
            rng.gen_range(RADIUS + EPSILON, *global::REAL_WIDTH),
            rng.gen_range(RADIUS + EPSILON, *global::REAL_HEIGHT),
            rng.gen_range(0.0, MASS_RANGE),
            root.clone(),
        );
        body_wrappers.push(BodyWrapper::from(body));
    }

    (body_wrappers, root)
}

fn benchmark_threading_algorithm(
    with_rayon: bool,
    body_wrappers: &Vec<BodyWrapper>,
    root: Arc<QuadNode>,
) {
    let start = std::time::SystemTime::now();
    if with_rayon {
        thread_rayon(body_wrappers, root);
    } else {
        thread_go(body_wrappers, root);
    }
    let end = std::time::SystemTime::now();
    println!(
        "Duration: {} ms",
        end.duration_since(start).unwrap().as_millis()
    );
}

/// generates and animates a tree-like structure of nodes (bodies) using threads. The function takes a boolean parameter with_rayon which determines whether to use Rayon or PThread library for parallelism.
pub fn start_thread_tree(with_rayon: bool) {
    let (body_wrappers, mut root) = generate_body_wrappers();

    if *global::BENCHMARK {
        benchmark_threading_algorithm(with_rayon, &body_wrappers, root);
    } else {
        let (mut event_pump, mut canvas) =
            crate::global::init_sdl(if with_rayon { "RayonTree" } else { "PThread" });

        canvas.set_draw_color(Color::RGB(0, 255, 255));
        canvas.clear();
        canvas.present();

        let mut i = 0;
        let mut n = 0;
        let mut start = std::time::SystemTime::now();

        'running: loop {
            n += 1;
            canvas
                .set_scale(*global::SCALE_FACTOR as f32, *global::SCALE_FACTOR as f32)
                .unwrap();
            canvas.set_draw_color(Color::RGB(255, 255, 255));
            canvas.clear();

            i = (i + 1) % 255;
            canvas.set_draw_color(Color::RGB(i, 64, 255 - i));
            let points = body_wrappers.iter().map(|x| x.to_sdl()).collect::<Vec<_>>();
            canvas
                .draw_points(points.as_slice())
                .expect("unable to draw points");

            if with_rayon {
                root = thread_rayon(&body_wrappers, root);
            } else {
                root = thread_go(&body_wrappers, root);
            }

            for event in event_pump.poll_iter() {
                match event {
                    Event::Quit { .. } => break 'running,
                    _ => {}
                }
            }
            canvas.present();
            crate::global::show_fps(&mut n, &mut start);
        }
    }
}
