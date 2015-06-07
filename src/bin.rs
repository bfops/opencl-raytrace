#![feature(convert)]

extern crate gl;
#[macro_use]
extern crate log;
extern crate opencl;
extern crate sdl2;
extern crate stopwatch;
extern crate time;
extern crate yaglw;

mod main;
mod scene;

pub fn main() {
  main::main();
}
