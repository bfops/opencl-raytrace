#[macro_use]
extern crate glium;
extern crate glutin;
#[macro_use]
extern crate log;
extern crate opencl;
extern crate time;

mod main;
mod scene;

fn main() {
  main::main();
}
