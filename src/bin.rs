#![feature(repr_simd)]

extern crate cgmath;
#[macro_use]
extern crate glium;
extern crate glutin;
#[macro_use]
extern crate log;
extern crate ocl;
extern crate rand;
extern crate time;

mod main;
mod scene;

fn main() {
  main::main();
}
