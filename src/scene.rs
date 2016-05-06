use opencl;
use opencl::mem::CLBuffer;
use std;
use std::borrow::Borrow;
use std::io::Read;

use main::RGB;

#[repr(C)]
pub struct Object {
  pub center      : [f32; 3],
  pub radius      : f32,
  pub color       : [f32; 3],
  pub scattering  : f32,
  pub emittance   : f32,
  pub reflectance : f32,
}

pub struct T {
  pub objects       : Vec<Object>,
  pub fovy          : f32,
  pub eye           : [f32; 3],
  pub look          : [f32; 3],
  pub up            : [f32; 3],
  pub ambient_light : [f32; 3],
}

impl T {
  pub fn render(&self, width: u32, height: u32) -> Vec<RGB> {
    let (device, ctx, queue) = opencl::util::create_compute_context().unwrap();

    let num_pixels = width as usize * height as usize;

    let program = {
      let mut file = std::fs::File::open("cl/main.cl").unwrap();
      let mut ker = String::new();
      file.read_to_string(&mut ker).unwrap();
      ctx.create_program_from_source(ker.borrow())
    };
    program.build(&device).unwrap();

    let kernel = program.create_kernel("render");

    let mut arg = 0;
    kernel.set_arg(arg, &width)              ; arg = arg + 1;
    kernel.set_arg(arg, &height)             ; arg = arg + 1;
    kernel.set_arg(arg, &self.fovy)          ; arg = arg + 1;
    kernel.set_arg(arg, &self.eye)           ; arg = arg + 1;
    kernel.set_arg(arg, &self.look)          ; arg = arg + 1;
    kernel.set_arg(arg, &self.up)            ; arg = arg + 1;

    let random_seed: u64 = 0x123456789abcdef0;
    kernel.set_arg(arg, &random_seed)        ; arg = arg + 1;

    kernel.set_arg(arg, &self.ambient_light) ; arg = arg + 1;

    let objects: &[Object] = &self.objects;
    let object_buffer: CLBuffer<Object> = ctx.create_buffer(objects.len(), opencl::cl::CL_MEM_READ_ONLY);
    queue.write(&object_buffer, &&objects[..], ());
    kernel.set_arg(arg, &object_buffer)      ; arg = arg + 1;

    let num_objects = objects.len() as u32;
    kernel.set_arg(arg, &num_objects)        ; arg = arg + 1;

    let output_buffer: CLBuffer<RGB> = ctx.create_buffer(num_pixels, opencl::cl::CL_MEM_WRITE_ONLY);
    kernel.set_arg(arg, &output_buffer)      ; arg = arg + 1;

    let event = queue.enqueue_async_kernel(&kernel, num_pixels, None, ());

    queue.get(&output_buffer, &event)
  }
}
