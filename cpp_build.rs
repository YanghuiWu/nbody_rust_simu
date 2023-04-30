fn main() {
    cpp_build::Config::new().compiler("g++-12").flag("-fopenmp").flag("-std=c++17").opt_level(3).build("src/openmp/cpp_module.rs");
}