use opencl;
use opencl::cl::cl_float;
use opencl::mem::CLBuffer;
use std;
use std::borrow::Borrow;
use std::io::Read;

use main::RGB;

#[repr(simd)]
struct SixteenBytes(u64, u64);

#[allow(non_camel_case_types)]
#[repr(C)]
pub struct cl_float3 {
  data: [cl_float; 4],
  align: [SixteenBytes; 0],
}

impl cl_float3 {
  pub fn new(xyz: [cl_float; 3]) -> Self {
    cl_float3 {
      data: [xyz[0], xyz[1], xyz[2], 0.0],
      align: [],
    }
  }
}

pub mod texture {
  use std;

  use opencl::cl::{cl_uchar, cl_float};
  use super::{cl_float3};

  #[repr(C)]
  pub struct T {
    data: [cl_float; 4],
    tag: cl_uchar,
    align: [cl_float3; 0],
  }

  fn of<X>(tag: cl_uchar, mut x: X) -> T {
    assert!(std::mem::size_of::<T>() >= std::mem::size_of::<X>());

    let mut t =
      T {
        data: [0.0; 4],
        tag: tag,
        align: [],
      };

    let p = unsafe {
      std::mem::transmute(&mut t.data[0])
    };
    std::mem::swap(p, &mut x);

    t
  }

  #[repr(C)]
  pub struct Sky;

  impl Sky {
    pub fn to_texture() -> T {
      of(1, [0; 0])
    }
  }

  #[repr(C)]
  pub struct Grass;

  impl Grass {
    pub fn to_texture() -> T {
      of(2, [0; 0])
    }
  }

  #[repr(C)]
  pub struct Wood;

  impl Wood {
    pub fn to_texture() -> T {
      of(3, [0; 0])
    }
  }

  #[repr(C)]
  pub struct SolidColor(pub cl_float3);

  impl SolidColor {
    pub fn to_texture(self) -> T {
      of(0, self)
    }
  }
}

#[repr(C)]
pub struct Object {
  pub center        : cl_float3,
  pub radius        : cl_float,
  pub diffuseness   : cl_float,
  pub emittance     : cl_float,
  pub reflectance   : cl_float,
  pub transmittance : cl_float,
  pub texture       : texture::T,
}

pub struct T {
  pub objects       : Vec<Object>,
  pub fovy          : f32,
  pub eye           : [f32; 3],
  pub look          : [f32; 3],
  pub up            : [f32; 3],
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
    if let Err(e) = program.build(&device) {
      panic!("Error building program: {}", e);
    }

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
