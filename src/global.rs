use std::sync::Arc;
use std::time::SystemTime;

use clap::*;
use sdl2::render::Canvas;
// use clap::Arg;
// use clap::ArgMatches;
// use clap::Command;
use crate::geometry::{self, Point};
use crate::quad_tree::node::QuadNode;
use hashbrown::HashMap;
use lazy_static;
use mpi::environment::*;
use mpi::topology::{Process, SystemCommunicator};
use mpi::traits::Communicator;
use nalgebra::Vector2;
use parking_lot::RwLock;

// lazy_static crate to lazily initialize them when they are first used.
lazy_static! {
    /// A thread-safe, read-write hash map from a custom Point type to a two-dimensional Vector2 type from the nalgebra crate.
    pub static ref VMAP : RwLock<HashMap<Point, Vector2<f64>>> = RwLock::new(HashMap::new());

    /// An MPI universe, initialized by calling a initialize() function defined elsewhere. If initialization fails, this variable will be set to None.
    pub static ref UNIVERSE : Universe = initialize().unwrap();

    ///  An MPI communicator for the entire universe.
    pub static ref WORLD : SystemCommunicator = UNIVERSE.world();

    static ref ENGINES : Vec<&'static str> =
        vec!["tree", "openmp", "pthread", "mpi_normal", "mpi_openmp", "brute_force", "rayon", "rayon_tree"];

    static ref MODES : Vec<&'static str> =
        vec!["benchmark", "display"];

    // pub static ref MATCHES : Option<ArgMatches> = {
    //     let result = Command::new("MyApp")
    //     .arg(Arg::new("engine")
    //         .short('e').long("engine").value_name("ENGINE").help("render engine").required(true)
    //     //     set ENGINE as the possible value
    //         .value_parser(|x: &str| {
    //             if ENGINES.contains(&x) {
    //                 Ok(x)
    //             } else {
    //                 Err(format!("{} is not a valid engine", x))
    //             }
    //         }))
    //
    //     .arg(Arg::new("width")
    //         .short('w').long("width").value_name("WIDTH").help("canvas width").default_value("1000"))
    //     .get_matches();
    //     Some(result)
    // };

    pub static ref MATCHES : Option<ArgMatches<'static>> = {
        let result = App::new("MyApp")
        .arg(Arg::with_name("engine")
            .short("e").value_name("ENGINE").help("render engine").required(true)
            .possible_values(ENGINES.as_slice()))
        .arg(Arg::with_name("width")
            .short("w").value_name("WIDTH").help("canvas width").default_value("800"))
        .arg(Arg::with_name("height")
            .short("h").value_name("HEIGHT").help("canvas height").default_value("600"))
        .arg(Arg::with_name("scale")
            .short("s").value_name("SCALE").help("scale factor").default_value("4.0"))
        .arg(Arg::with_name("number")
            .short("n").value_name("NUM").help("number of bodies").default_value("2000"))
        .arg(Arg::with_name("thread").help("thread number (for openmp/pthread), must be greater than 0, otherwise reset to 6")
            .short("t").default_value("6"))
        .arg(Arg::with_name("mode").value_name("MODE")
            .short("m").help("running mode").possible_values(MODES.as_slice()).default_value("benchmark"))
        .arg(Arg::with_name("fps").value_name("FPS_FLAG")
            .short("f").help("whether to show fps").possible_values(&["yes", "no"]).default_value("yes"))
        .get_matches_safe();
        match result {
            Ok(x) => Some(x),
            Err(m) => {
                if WORLD.rank() == ROOT {
                    m.exit();
                }
                None
            }
        }
    };


    // pub static ref WIDTH : f64 = match MATCHES.as_ref().and_then(|m| m.get_one("width").and_then(|x: &str|x.parse::<usize>().ok())) {
    //     Some(w) if w > 0 => w as f64,
    //     _ => 800.0
    // };

    // pubstatic ref WIDTH:f64 = MATCHES.as_ref().and_then(|m| m.value_of("width").and_then(|x|x.parse::<usize>().ok())).map(|w| w as f64).unwrap_or(800.0);



    /// WIDTH, HEIGHT: f64 values representing the width and height of the canvas, respectively. These are obtained from the MATCHES variable if present and parseable, otherwise they default to 800.0 and 600.0.
    pub static ref WIDTH : f64 = match MATCHES.as_ref().and_then(|m| m.value_of("width").and_then(|x|x.parse::<usize>().ok())) {
        Some(w) if w > 0 => w as f64,
        _ => 800.0
    };

    /// WIDTH, HEIGHT: f64 values representing the width and height of the canvas, respectively. These are obtained from the MATCHES variable if present and parseable, otherwise they default to 800.0 and 600.0.
    pub static ref HEIGHT : f64 = match MATCHES.as_ref().and_then(|m| m.value_of("height").and_then(|x|x.parse::<usize>().ok())) {
        Some(w) if w > 0 => w as f64,
        _ => 600.0
    };

    /// SCALE_FACTOR: A f64 value representing the scaling factor to apply to the canvas. This is obtained from the MATCHES variable if present and parseable, otherwise it defaults to 1.0.
    pub static ref SCALE_FACTOR : f64 = match MATCHES.as_ref().and_then(|m| m.value_of("scale").and_then(|x|x.parse::<f64>().ok())) {
        Some(w) if w > 0.0 => w,
        _ => 1.0
    };

    /// A usize value representing the number of bodies to generate for the simulation. This is obtained from the MATCHES variable if present and parseable, otherwise it defaults to 50.
    pub static ref SIZE : usize = match MATCHES.as_ref().and_then(|m| m.value_of("number").and_then(|x|x.parse::<usize>().ok())) {
        Some(w)  => w,
        _ => 50
    };

    pub static ref BENCHMARK : bool = match MATCHES.as_ref().and_then(|m| m.value_of("mode")) {
        Some("benchmark") => true,
        _ => false
    };

    pub static ref FPS_FLAG : bool = match MATCHES.as_ref().and_then(|m| m.value_of("fps")) {
        Some("yes") => true,
        _ => false
    };

    pub static ref THREAD : usize = match MATCHES.as_ref().and_then(|m| m.value_of("thread").and_then(|x|x.parse::<usize>().ok())) {
        Some(w) if w > 0 => w,
        _ => 6
    };

    pub static ref REAL_WIDTH: f64 = *WIDTH / *SCALE_FACTOR;

    pub static ref REAL_HEIGHT : f64 = *HEIGHT / *SCALE_FACTOR;

    pub static ref ROOT_PROC : Process<'static, SystemCommunicator> =  WORLD.process_at_rank(ROOT);
}

pub const MIN_SIZE: f64 = 10.0;
pub const DIST_SCALE_LIMIT: f64 = 0.75;
pub const RADIUS: f64 = 0.5;
pub const G: f64 = 5.0;
pub const ALPHA: f64 = 0.001;
pub const ROOT: i32 = 0;
pub const MASS_RANGE: f64 = 50.0;

pub fn show_fps(frame_count: &mut usize, start_time: &mut SystemTime) {
    const FPS_THRESHOLD_MILLIS: u128 = 1000;
    const MILLIS_PER_SECOND: f64 = 1000.0;

    if *FPS_FLAG {
        let current_time = SystemTime::now();
        let elapsed_millis = current_time
            .duration_since(*start_time)
            .unwrap_or_else(|_| std::time::Duration::default())
            .as_millis();

        if elapsed_millis >= FPS_THRESHOLD_MILLIS {
            let fps = (*frame_count as f64 / elapsed_millis as f64) * MILLIS_PER_SECOND;
            println!("FPS: {:.2}", fps);

            *start_time = current_time;
            *frame_count = 0;
        }
    }
}

use sdl2::video::Window;

/// Initializes SDL and returns an EventPump and Canvas<Window> tuple.
pub fn init_sdl(title: &str) -> (sdl2::EventPump, Canvas<Window>) {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    let window = video_subsystem
        .window(
            &format!("nbody {}", title),
            *crate::global::WIDTH as u32,
            *crate::global::HEIGHT as u32,
        )
        .position_centered()
        .build()
        .unwrap();

    let canvas = window.into_canvas().build().unwrap();
    let event_pump = sdl_context.event_pump().unwrap();

    (event_pump, canvas)
}

pub fn new_boundary() -> geometry::Square {
    geometry::Square(
        Vector2::new(*REAL_WIDTH, *REAL_HEIGHT),
        Vector2::new(0.0, 0.0),
    )
}

pub fn new_root() -> Arc<QuadNode> {
    Arc::new(QuadNode::new(new_boundary()))
}
