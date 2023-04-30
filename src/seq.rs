use std::sync::Arc;


use rand::Rng;
use sdl2::event::Event;
use sdl2::pixels::Color;

// use crate::geometry;
use crate::geometry::{Body, Square};
use crate::global;
// use crate::quad_tree;
use crate::quad_tree::node::QuadNode;
// use std::f64::EPSILON;

fn refresh(pool: &mut Vec<Body>, root: &mut Arc<QuadNode>, boundary: &Square) {
    {
        let mut a = global::VMAP.write();
        a.clear();
        for i in &*pool {
            a.insert(i.position.clone(), i.velocity);
        }
        for i in &*pool {
            assert!(a.contains_key(&i.position))
        }
    }
    for i in &mut *pool {
        i.make_ready();
        i.collision_detect();
        i.update_velocity();
        i.update_position();
        i.gravity_impact(root.clone());
        i.check_boundary();
    }
    *root = Arc::new(QuadNode::new(boundary.clone()));
    for i in &mut *pool {
        i.reinsert(root.clone());
    }
}

fn init_tree() -> (Arc<QuadNode>, Vec<Body>, Square) {
    let root = global::new_root();
    let boundary = global::new_boundary();
    let mut rng = rand::thread_rng();
    let mut pool = Vec::new();
    for _ in 0..*global::SIZE {
        pool.push(Body::new(
            rng.gen_range(global::RADIUS + f64::EPSILON, *global::REAL_WIDTH - global::RADIUS),
            rng.gen_range(global::RADIUS + f64::EPSILON, *global::REAL_HEIGHT - global::RADIUS),
            rng.gen_range(0.0, global::MASS_RANGE),
            root.clone(),
        ));
    }
    
    (root, pool, boundary)
}

pub fn start_tree() {
    let (mut root, mut pool, boundary) = init_tree();

    if *global::BENCHMARK {
        let start = std::time::SystemTime::now();
        refresh(&mut pool, &mut root, &boundary);
        let end = std::time::SystemTime::now();
        println!("Duration: {} ms", end.duration_since(start).unwrap().as_millis());
    } else {
        let (mut event_pump, mut canvas) = crate::global::init_sdl("Sequential");

        canvas.set_draw_color(Color::RGB(0, 255, 255));
        canvas.clear();
        canvas.present();
        let mut i = 0;
        let mut n = 0;
        let mut start = std::time::SystemTime::now();
        'running: loop {
            n += 1;
            //println!("{:?}", pool);
            canvas.set_scale(*global::SCALE_FACTOR as f32, *global::SCALE_FACTOR as f32).unwrap();
            canvas.set_draw_color(Color::RGB(255, 255, 255));
            canvas.clear();
            i = (i + 1) % 255;
            canvas.set_draw_color(Color::RGB(i, 64, 255 - i));
            let points = pool.iter().map(|x| x.geometric()).collect::<Vec<_>>();
            canvas.draw_points(points.as_slice()).expect("unable to draw points");

            for event in event_pump.poll_iter() {
                match event {
                    Event::Quit { .. } => {
                        break 'running;
                    }
                    _ => {}
                }
            }

            canvas.present();
            refresh(&mut pool, &mut root, &boundary);
            global::show_fps(&mut n, &mut start);
        }
    }
}