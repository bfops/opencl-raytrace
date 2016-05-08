use glium;
use glutin;
use std;
use time;

use scene;

fn cl_float3(x: f32, y: f32, z: f32) -> scene::cl_float3 {
  scene::cl_float3::new([x, y, z])
}

fn solid_color(r: f32, g: f32, b: f32) -> scene::texture::T {
  scene::texture::SolidColor(cl_float3(r, g, b)).to_texture()
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RGB {
  pub r: f32,
  pub g: f32,
  pub b: f32,
}

unsafe impl Send for RGB {}

unsafe impl glium::texture::PixelValue for RGB {
  fn get_format() -> glium::texture::ClientFormat {
    glium::texture::ClientFormat::F32F32F32
  }
}

pub const WINDOW_WIDTH: u32 = 800;
pub const WINDOW_HEIGHT: u32 = 600;

pub fn main() {
  use glium::DisplayBuild;

  let window =
    glutin::WindowBuilder::new()
    .with_dimensions(WINDOW_WIDTH, WINDOW_HEIGHT)
    .build_glium()
    .unwrap();

  let mut image_data = vec!();
  for y in 0 .. WINDOW_HEIGHT {
    let mut row = vec!();
    for x in 0 .. WINDOW_WIDTH {
      row.push(
        RGB {
          r: x as f32 / WINDOW_WIDTH as f32,
          g: y as f32 / WINDOW_HEIGHT as f32,
          b: 0.0,
        }
      );
    }
    image_data.push(row);
  }

  let scene =
    scene::T {
      objects       :
        vec!(
          scene::Object { center: cl_float3(-1.0,    0.0,  -4.0), radius:   1.0, emittance:  0.0, reflectance: 1.0, transmittance: 0.0, diffuseness: 1.0 , texture: solid_color(1.0, 0.0, 0.0) },
          scene::Object { center: cl_float3( 0.0,   -1.0,  -5.0), radius:   1.0, emittance:  0.0, reflectance: 0.1, transmittance: 0.9, diffuseness: 0.01, texture: solid_color(0.0, 0.6, 1.0) },
          scene::Object { center: cl_float3(-1.0,   -1.0,  -2.0), radius:   0.5, emittance:  0.0, reflectance: 1.0, transmittance: 0.0, diffuseness: 0.2 , texture: solid_color(1.0, 0.4, 0.0) },
          scene::Object { center: cl_float3( 6.0,    1.5, -10.0), radius:   4.0, emittance:  0.0, reflectance: 1.0, transmittance: 0.0, diffuseness: 0.01, texture: solid_color(1.0, 1.0, 1.0) },
          scene::Object { center: cl_float3(30.0,   40.0,  10.0), radius:  10.0, emittance:  1.0, reflectance: 0.0, transmittance: 0.4, diffuseness: 1.0 , texture: solid_color(1.0, 1.0, 1.0) },
          scene::Object { center: cl_float3(-1.0,    0.2,  -2.0), radius:   0.4, emittance:  0.0, reflectance: 0.1, transmittance: 0.8, diffuseness: 0.01, texture: solid_color(0.9, 0.9, 1.0) },
          scene::Object { center: cl_float3( 0.0,    0.0,   0.0), radius: 100.0, emittance:  0.2, reflectance: 0.0, transmittance: 0.0, diffuseness: 0.0 , texture: scene::texture::Sky::to_texture() },
          scene::Object { center: cl_float3( 0.0, -102.0,   0.0), radius: 100.0, emittance:  0.0, reflectance: 1.0, transmittance: 0.0, diffuseness: 1.0 , texture: scene::texture::Grass::to_texture() },
        ),
      fovy          : std::f32::consts::FRAC_PI_2,
      eye           : [0.0, 0.0,  0.0],
      look          : [0.0, 0.0, -1.0],
      up            : [0.0, 1.0,  0.0],
    };

  let mut rendered_2d = vec!();
  {
    let scale = 1 << 0;
    let w = scale * WINDOW_WIDTH;
    let h = scale * WINDOW_HEIGHT;

    let before = time::precise_time_ns();
    let rendered = scene.render(w, h);
    let after = time::precise_time_ns();
    println!("Render took {:?}ms", (after - before) as f32 / 1_000_000.0);

    for y in 0 .. h as usize {
      let mut row = vec!();
      for x in 0 .. w as usize {
        row.push(rendered[y * w as usize + x]);
      }
      rendered_2d.push(row);
    }
  }

  let texture = glium::texture::Texture2d::new(&window, rendered_2d).unwrap();

  // building the vertex buffer, which contains all the vertices that we will draw
  let vertex_buffer = {
    #[derive(Copy, Clone)]
    struct Vertex {
      position: [f32; 2],
      tex_coords: [f32; 2],
    }

    implement_vertex!(Vertex, position, tex_coords);

    glium::VertexBuffer::new(&window,
      &[
        Vertex { position: [-1.0, -1.0], tex_coords: [0.0, 0.0] },
        Vertex { position: [-1.0,  1.0], tex_coords: [0.0, 1.0] },
        Vertex { position: [ 1.0,  1.0], tex_coords: [1.0, 1.0] },
        Vertex { position: [ 1.0, -1.0], tex_coords: [1.0, 0.0] }
      ]
    ).unwrap()
  };

  // building the index buffer
  let index_buffer =
    glium::IndexBuffer::new(
      &window,
      glium::index::PrimitiveType::TriangleStrip,
      &[1 as u16, 2, 0, 3],
    ).unwrap();

  // compiling shaders and linking them together
  let program = program!(&window,
    330 => {
      vertex: "
        #version 330

        uniform mat4 matrix;

        in vec2 position;
        in vec2 tex_coords;

        out vec2 v_tex_coords;

        void main() {
          gl_Position = matrix * vec4(position, 0.0, 1.0);
          v_tex_coords = tex_coords;
        }
      ",

      fragment: "
        #version 330
        uniform sampler2D tex;
        in vec2 v_tex_coords;
        out vec4 f_color;

        void main() {
          f_color = texture(tex, v_tex_coords);
        }
      "
    },
  ).unwrap();

  loop {
    use glium::Surface;

    // building the uniforms
    let uniforms = uniform! {
      matrix: [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0f32]
      ],
      tex:
        glium::uniforms::Sampler::new(&texture)
        .magnify_filter(glium::uniforms::MagnifySamplerFilter::Linear)
        .minify_filter(glium::uniforms::MinifySamplerFilter::LinearMipmapLinear)
    };

    let mut target = window.draw();
    target.clear_color(0.0, 0.0, 0.0, 0.0);
    target.draw(&vertex_buffer, &index_buffer, &program, &uniforms, &Default::default()).unwrap();
    target.finish().unwrap();

    for event in window.poll_events() {
      if let glutin::Event::Closed = event {
        return
      }
    }
  }
}
