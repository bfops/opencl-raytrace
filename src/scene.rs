use cgmath;
use ocl;
use std;
use std::io::Read;

use main::RGB;

#[allow(non_camel_case_types)]
pub type cl_uchar = ocl::cl_h::cl_uchar;
#[allow(non_camel_case_types)]
pub type cl_float = ocl::cl_h::cl_float;
#[allow(non_camel_case_types)]
pub type cl_float3 = ocl::aliases::ClFloat3;
#[allow(non_camel_case_types)]
pub type cl_float4 = ocl::aliases::ClFloat4;

pub mod texture {
  use std;

  use super::{cl_uchar, cl_float, cl_float3};

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
  pub eye           : cgmath::Vector3<f32>,
  pub look          : cgmath::Vector3<f32>,
  pub up            : cgmath::Vector3<f32>,
}

impl T {
  pub fn move_camera(&mut self, v: &cgmath::Vector3<f32>) {
    self.eye = self.eye + v;
  }

  pub fn x(&self) -> cgmath::Vector3<f32> {
    self.look.cross(self.up)
  }

  pub fn y(&self) -> cgmath::Vector3<f32> {
    self.up
  }

  pub fn z(&self) -> cgmath::Vector3<f32> {
    self.look
  }

  pub fn render(&self, width: u32, height: u32, random_seed: u64) -> Vec<RGB> {
    let num_pixels = width as usize * height as usize;

    let pq = {
      let mut file = std::fs::File::open("cl/main.cl").unwrap();
      let mut ker = String::new();
      file.read_to_string(&mut ker).unwrap();
      ocl::ProQue::builder()
        .src(ker)
        .dims([num_pixels])
        .build()
        .unwrap()
    };

    let objects: &[f32] =
      unsafe {
        std::slice::from_raw_parts(
          self.objects.as_ptr() as *const f32,
          self.objects.len() * std::mem::size_of::<Object>() / std::mem::size_of::<f32>()
        )
      };
    let object_buffer =
      ocl::Buffer::new(
        pq.queue(),
        Some(ocl::core::MEM_READ_ONLY | ocl::core::MEM_USE_HOST_PTR),
        [objects.len()],
        Some(objects),
      ).unwrap();

    let output_buffer: ocl::Buffer<f32> = pq.create_buffer().unwrap();

    pq
      .create_kernel("render")
      .unwrap()
      .arg_scl(width)
      .arg_scl(height)
      .arg_scl(self.fovy)
      .arg_vec(cl_float3::new(self.eye.x, self.eye.y, self.eye.z))
      .arg_vec(cl_float3::new(self.look.x, self.look.y, self.look.z))
      .arg_vec(cl_float3::new(self.up.x, self.up.y, self.up.z))
      .arg_scl(random_seed)
      .arg_buf(&object_buffer)
      .arg_scl(self.objects.len())
      .arg_buf(&output_buffer)
      .enq()
      .unwrap();

    let mut output = Vec::new();
    output_buffer.read(&mut output).enq().unwrap();
    output.shrink_to_fit();

    unsafe {
      let p: *mut RGB = output.as_ptr() as *mut RGB;
      Vec::from_raw_parts(p, output.len() / 3, output.capacity() / 3)
    }
  }
}
