use opencl;
use opencl::mem::CLBuffer;
use std;
use std::borrow::Borrow;
use std::io::Read;

use main::{WINDOW_WIDTH, WINDOW_HEIGHT, RGB};

pub struct T {
  pub obj1_center: [f32; 3],
  pub obj1_radius: f32,
  pub obj2_center: [f32; 3],
  pub obj2_radius: f32,
  pub camera: [f32; 3],
}

impl T {
  pub fn render(&self) -> Vec<RGB> {
    let (device, ctx, queue) = opencl::util::create_compute_context().unwrap();

    let len = WINDOW_WIDTH as usize * WINDOW_HEIGHT as usize;

    let output_buffer: CLBuffer<RGB> = ctx.create_buffer(len, opencl::cl::CL_MEM_WRITE_ONLY);

    let program = {
      let mut file = std::fs::File::open("src/kernel.cl").unwrap();
      let mut ker = String::new();
      file.read_to_string(&mut ker).unwrap();
      ctx.create_program_from_source(ker.borrow())
    };
    program.build(&device).unwrap();

    let kernel = program.create_kernel("render");
    let w = WINDOW_WIDTH as u32;
    let h = WINDOW_HEIGHT as u32;
    kernel.set_arg(0, &w);
    kernel.set_arg(1, &h);
    kernel.set_arg(2, &self.obj1_center);
    kernel.set_arg(3, &self.obj1_radius);
    kernel.set_arg(4, &self.obj2_center);
    kernel.set_arg(5, &self.obj2_radius);
    kernel.set_arg(6, &self.camera);

    // This is sketchy; we "implicitly cast" output_buffer from a CLBuffer<RGB> to a CLBuffer<f32>.
    kernel.set_arg(7, &output_buffer);

    let event = queue.enqueue_async_kernel(&kernel, len, None, ());

    queue.get(&output_buffer, &event)
  }
}
