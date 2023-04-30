use std::cell::RefCell;
// use std::sync::atomic::AtomicUsize;
// use std::sync::atomic::Ordering::SeqCst;
use std::sync::{Arc, Barrier};

// use nalgebra::Vector2;
use rayon::prelude::*;
use sdl2::rect::Point;

use crate::geometry::Body;
use crate::global::{SIZE, THREAD};
use crate::quad_tree::node::QuadNode;
use crate::{geometry, global};

/// struct SharedData contains two fields:
///
/// root which is an Arc (atomic reference-counted pointer) to a QuadNode struct, and
///
/// finished which is an AtomicUsize (an atomic unsigned integer) used to keep track of how many threads have finished their work.
struct SharedData {
    root: Arc<QuadNode>,
    // finished: AtomicUsize,
}

impl SharedData {
    fn new() -> Self {
        Self {
            root: global::new_root(),
            // finished: AtomicUsize::new(0),
        }
    }
}


/// a helper function that calculates the chunk size for each thread in a parallel computation. It takes three arguments:
///
/// total: total number of items to be processed,
///
/// group: the number of threads, and
///
/// kth:  the index of the current thread.
fn chunk_size(total: usize, group: usize, kth: usize) -> usize {
    let a = total - kth;
    if a % group > 0 {
        a / group + 1
    } else {
        a / group
    }
}

/// wraps a Body struct inside an Arc<RefCell<>> so that it can be shared between threads. It also provides methods to convert it to an SDL point and to clone itself.
#[derive(Clone)]
pub struct BodyWrapper {
    ptr: Arc<RefCell<geometry::Body>>,
}

unsafe impl Sync for BodyWrapper {}

unsafe impl Send for BodyWrapper {}

// impl Clone for BodyWrapper {
//     fn clone(&self) -> Self {
//         BodyWrapper {
//             ptr: self.ptr.clone()
//         }
//     }
// }

impl From<Body> for BodyWrapper {
    fn from(body: Body) -> Self {
        BodyWrapper {
            ptr: Arc::new(RefCell::new(body)),
        }
    }
}

impl BodyWrapper {
    pub(crate) fn to_sdl(&self) -> sdl2::rect::Point {
        let body = self.ptr.borrow();
        Point::new(body.position.x as i32, body.position.y as i32)
    }
}

fn insert_bodies(points: &Vec<BodyWrapper>) {
    let mut lock = crate::global::VMAP.write();
    for i in points {
        let mut inst = i.ptr.borrow_mut();
        inst.make_ready();
        lock.insert(inst.position.clone(), inst.velocity.clone());
    }
}

fn update_body(instance: &mut Body, last_root: Arc<QuadNode>, shared: Arc<SharedData>) {
    instance.collision_detect();
    instance.update_velocity();
    instance.update_position();
    instance.check_boundary();
    instance.gravity_impact(last_root.clone());
    instance.reinsert(shared.root.clone());
}

/// performs the n-body simulation using the thread::spawn function to run each chunk of work in a separate thread.
///
/// It first creates a new SharedData struct with a new root QuadNode, and then populates the global VMAP with the positions and velocities of the bodies. It then divides the work into chunks and spawns threads to perform the work on each chunk. Each thread performs collision detection, updates the body's velocity and position, checks for boundary conditions, applies gravity, and re-inserts the body into the quadtree. Finally, it waits for all threads to finish and returns the updated root QuadNode.
pub fn thread_go(points: &Vec<BodyWrapper>, last_root: Arc<QuadNode>) -> Arc<QuadNode> {
    crate::global::VMAP.write().clear();
    let shared = Arc::new(SharedData::new());
    let mut counter = 0;
    insert_bodies(points);
    let barrier = Arc::new(Barrier::new(*THREAD));
    let mut handlers = Vec::new();
    for i in 0..*THREAD {
        let work_size = chunk_size(*SIZE, *THREAD, i);
        let points = (&points[counter..counter + work_size])
            .iter()
            .map(|x| x.clone())
            .collect::<Vec<_>>();
        let last_root = last_root.clone();
        let shared = shared.clone();
        let barrier = barrier.clone();
        handlers.push(std::thread::spawn(move || {
            for i in &points {
                let mut instance = i.ptr.borrow_mut();
                update_body(&mut instance, last_root.clone(), shared.clone());
            }
            barrier.wait();
            // println!("Thread {} finished", i)
        }));
        counter += work_size
    }
    for handler in handlers {
        handler.join().unwrap();
    }
    shared.root.clone()
}

/// similar to thread_go, but instead of using thread::spawn, it uses the rayon::par_iter function to parallelize the work across multiple threads.
pub fn thread_rayon(points: &Vec<BodyWrapper>, last_root: Arc<QuadNode>) -> Arc<QuadNode> {
    crate::global::VMAP.write().clear();
    let shared = Arc::new(SharedData::new());

    insert_bodies(points);

    points.par_iter().for_each(|i| {
        let mut inst = i.ptr.borrow_mut();
        inst.make_ready();
    });

    points.par_iter().for_each(|i| {
        let mut instance = i.ptr.borrow_mut();
        update_body(&mut instance, last_root.clone(), shared.clone());
    });

    shared.root.clone()
}
