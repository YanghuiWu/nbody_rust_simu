use mpi::topology::Communicator;
use mpi::traits::Root;
use sdl2::event::Event;
use sdl2::pixels::Color;

use mpi_module::*;

use crate::global;
use crate::openmp::cpp_module::setup;

mod mpi_module;

fn normal_procedure(s: usize, t: usize, flag: bool, g_data: &mut GlobalData, with_openmp: bool) {
    g_data.broadcast();
    if with_openmp {
        g_data.update_all_openmp(s, t);
    } else {
        g_data.update_all(s, t);
    }
    if flag {
        g_data.gather(s, t + 1);
        //println!("{} will send {} data", global::WORLD.rank(), t - s + 1)
    } else {
        g_data.gather(s, t);
        //println!("{} will send {} data", global::WORLD.rank(), t - s)
    }
}

fn block_distribution(
    global_size: usize,
    world_size: usize,
) -> (Vec<usize>, Vec<usize>, Vec<bool>) {
    let block_size = if global_size % world_size > 0 {
        global_size / world_size + 1
    } else {
        global_size / world_size
    };
    let mut counter = 0;
    let mut i = 0;
    let mut starts = Vec::new();
    let mut ends = Vec::new();
    let mut flags = Vec::new();
    while counter < block_size * world_size {
        starts.push(counter);
        let (length, flag) = chunk_size(global_size, world_size, i);
        ends.push(counter + length);
        flags.push(flag);
        counter += block_size;
        i += 1;
    }
    (starts, ends, flags)
}

fn benchmark_mode(with_openmp: bool) {
    let mut g_data = GlobalData::new();
    let world_size = global::WORLD.size() as usize;
    let (starts, ends, flags) = block_distribution(*global::SIZE, world_size);

    let mut s = 0;
    let mut t = 0;
    let mut flag = false;

    global::ROOT_PROC.scatter_into_root(starts.as_slice(), &mut s);
    global::ROOT_PROC.scatter_into_root(ends.as_slice(), &mut t);
    global::ROOT_PROC.scatter_into_root(flags.as_slice(), &mut flag);
    let mut finished = true;
    let start = std::time::SystemTime::now();
    normal_procedure(s, t, flag, &mut g_data, with_openmp);
    global::ROOT_PROC.broadcast_into(&mut finished);
    let end = std::time::SystemTime::now();
    println!(
        "Duration: {} ms",
        end.duration_since(start).unwrap().as_millis()
    );
}

pub fn start_mpi_root(with_openmp: bool) {
    if with_openmp {
        setup();
    }
    if *global::BENCHMARK {
        return benchmark_mode(with_openmp);
    }
    
    let (mut event_pump, mut canvas) = crate::global::init_sdl("MPI");

    let mut g_data = GlobalData::new();
    let world_size = global::WORLD.size() as usize;
    let (starts, ends, flags) = block_distribution(*global::SIZE, world_size);

    let mut s = 0;
    let mut t = 0;
    let mut flag = false;

    global::ROOT_PROC.scatter_into_root(starts.as_slice(), &mut s);
    global::ROOT_PROC.scatter_into_root(ends.as_slice(), &mut t);
    global::ROOT_PROC.scatter_into_root(flags.as_slice(), &mut flag);
    canvas.set_draw_color(Color::RGB(0, 255, 255));
    canvas.clear();
    canvas.present();
    let mut i = 0;
    let mut n = 0;
    let mut start = std::time::SystemTime::now();
    let mut finished = false;
    'running: loop {
        n += 1;
        canvas
            .set_scale(*global::SCALE_FACTOR as f32, *global::SCALE_FACTOR as f32)
            .unwrap();
        canvas.set_draw_color(Color::RGB(255, 255, 255));
        canvas.clear();
        i = (i + 1) % 255;
        canvas.set_draw_color(Color::RGB(i, 64, 255 - i));
        let points = g_data.to_sdl(&starts, &ends);
        canvas
            .draw_points(points.as_slice())
            .expect("unable to draw points");
        normal_procedure(s, t, flag, &mut g_data, with_openmp);
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => {
                    finished = true;
                    global::ROOT_PROC.broadcast_into(&mut finished);
                    break 'running;
                }
                _ => {}
            }
        }
        canvas.present();
        global::show_fps(&mut n, &mut start);
        global::ROOT_PROC.broadcast_into(&mut finished);
    }
}

pub fn start_mpi_child(with_openmp: bool) {
    let mut g_data = GlobalData::new();
    let mut s = 0;
    let mut t = 0;
    let mut flag = false;
    let mut finished = false;
    global::ROOT_PROC.scatter_into(&mut s);
    global::ROOT_PROC.scatter_into(&mut t);
    global::ROOT_PROC.scatter_into(&mut flag);
    while !finished {
        normal_procedure(s, t, flag, &mut g_data, with_openmp);
        global::ROOT_PROC.broadcast_into(&mut finished);
    }
}
