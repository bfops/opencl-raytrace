use cgmath;
use glium;
use glutin;
use std;
use rand;
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

  let mut scene =
    scene::T {
      objects       :
        vec!(
          // red ball
          scene::Object { center: cl_float3(-4.0,   -1.0,  -5.0), radius:   1.0, emittance:  0.0, reflectance: 1.0, transmittance: 0.0, diffuseness: 1.0 , texture: solid_color(1.0, 0.0, 0.0) },
          // blue ball
          scene::Object { center: cl_float3(-0.5,   -1.0,  -5.0), radius:   1.0, emittance:  0.0, reflectance: 0.1, transmittance: 0.9, diffuseness: 0.01, texture: solid_color(0.0, 0.6, 1.0) },
          // frosted glass ball
          scene::Object { center: cl_float3(-0.7,   -0.5,  -1.5), radius:   0.5, emittance:  0.0, reflectance: 0.1, transmittance: 0.8, diffuseness: 0.04, texture: solid_color(0.9, 0.9, 1.0) },
          // glass ball
          scene::Object { center: cl_float3( 0.2,   -0.5,  -1.0), radius:   0.5, emittance:  0.0, reflectance: 0.1, transmittance: 0.9, diffuseness: 0.0 , texture: solid_color(0.9, 0.9, 1.0) },
          // brass ball
          scene::Object { center: cl_float3( 3.0,    1.5, -10.0), radius:   4.0, emittance:  0.0, reflectance: 1.0, transmittance: 0.0, diffuseness: 0.1 , texture: solid_color(1.0, 0.4, 0.1) },
          // small mirror ball
          scene::Object { center: cl_float3( 3.0,   -1.0,  -3.5), radius:   1.0, emittance:  0.0, reflectance: 0.9, transmittance: 0.0, diffuseness: 0.0 , texture: solid_color(1.0, 1.0, 1.0) },
          // light
          scene::Object { center: cl_float3(-9.0,   10.0,   0.0), radius:   1.0, emittance:  1.0, reflectance: 0.0, transmittance: 1.0, diffuseness: 0.0 , texture: solid_color(0.9, 0.9, 1.0) },
          // walls
          scene::Object { center: cl_float3( 0.0,    0.0,   0.0), radius:  20.0, emittance:  0.2, reflectance: 0.0, transmittance: 0.0, diffuseness: 1.0 , texture: solid_color(1.0, 1.0, 1.0) },
          // floor
          scene::Object { center: cl_float3( 0.0, -102.0,   0.0), radius: 100.0, emittance:  0.0, reflectance: 1.0, transmittance: 0.0, diffuseness: 0.02, texture: scene::texture::Wood::to_texture() },
        ),
      fovy          : std::f32::consts::FRAC_PI_2,
      eye           : cgmath::Vector3::new(0.0, 0.0,  0.0),
      look          : cgmath::Vector3::new(0.0, 0.0, -1.0),
      up            : cgmath::Vector3::new(0.0, 1.0,  0.0),
    };

  let scale = 1 << 0;
  let w = scale * WINDOW_WIDTH;
  let h = scale * WINDOW_HEIGHT;

  let mut make_random_seed: rand::XorShiftRng =
    rand::SeedableRng::from_seed([0x12345678, 0x9abcdef0, 0x89765432, 0x12324121]);

  let framebuffer_texture =
    glium::texture::Texture2d::new(
      &window,
      glium::texture::RawImage2d {
        width: w,
        height: h,
        format: glium::texture::ClientFormat::F32F32F32,
        data: std::borrow::Cow::Owned(std::iter::repeat((0.0, 0.0, 0.0)).take((w*h) as usize).collect()),
      },
    ).unwrap();

  let mut framebuffer = glium::framebuffer::SimpleFrameBuffer::new(&window, &framebuffer_texture).unwrap();

  loop {
    let before = time::precise_time_ns();
    let rendered = scene.render(w, h, rand::Rng::next_u64(&mut make_random_seed));
    let after = time::precise_time_ns();
    println!("Render took {:?}ms", (after - before) as f32 / 1_000_000.0);

    let rendered =
      glium::texture::Texture2d::with_format(
        &window,
        glium::texture::RawImage2d {
          data: std::borrow::Cow::Owned(rendered),
          width: w,
          height: h,
          format: glium::texture::ClientFormat::F32F32F32,
        },
        glium::texture::UncompressedFloatFormat::F32F32F32,
        glium::texture::MipmapsOption::NoMipmap,
      )
      .unwrap();

    let draw_parameters =
      glium::DrawParameters {
        depth:
          glium::Depth {
            test: glium::DepthTest::Overwrite,
            write: false,
            .. Default::default()
          },
        blend:
          glium::Blend {
            color:
              glium::BlendingFunction::Addition {
                source: glium::LinearBlendingFactor::ConstantColor,
                destination: glium::LinearBlendingFactor::OneMinusConstantColor,
              },
            alpha: glium::BlendingFunction::AlwaysReplace,
            constant_value: (0.1, 0.1, 0.1, 0.0),
          },
        .. Default::default()
      };

    let source =
      glium::uniforms::Sampler::new(&rendered)
      .magnify_filter(glium::uniforms::MagnifySamplerFilter::Linear)
      .minify_filter(glium::uniforms::MinifySamplerFilter::LinearMipmapLinear);

    draw(&window, source, &mut framebuffer, &draw_parameters);

    let draw_parameters =
      glium::DrawParameters {
        depth:
          glium::Depth {
            test: glium::DepthTest::Overwrite,
            write: false,
            .. Default::default()
          },
        blend:
          glium::Blend {
            color: glium::BlendingFunction::AlwaysReplace,
            alpha: glium::BlendingFunction::AlwaysReplace,
            constant_value: (0.0, 0.0, 0.0, 0.0),
          },
        .. Default::default()
      };

    let source =
      glium::uniforms::Sampler::new(&framebuffer_texture)
      .magnify_filter(glium::uniforms::MagnifySamplerFilter::Linear)
      .minify_filter(glium::uniforms::MinifySamplerFilter::Linear);

    let mut target = window.draw();
    glium::Surface::clear(&mut target, None, Some((0.0, 0.0, 0.0, 1.0)), false, None, None);
    draw(&window, source, &mut target, &draw_parameters);
    target.finish();

    for event in window.poll_events() {
      match event {
        glutin::Event::Closed => return,
        glutin::Event::KeyboardInput(_, _, Some(key)) => {
          match key {
            glutin::VirtualKeyCode::W => scene.eye = scene.eye + scene.look,
            glutin::VirtualKeyCode::S => scene.eye = scene.eye - scene.look,
            _ => {},
          }
        },
        _ => {},
      }
    }
  }
}

fn draw<'a, Dest: glium::Surface>(
  window: &glium::backend::glutin_backend::GlutinFacade,
  source: glium::uniforms::Sampler<'a, glium::Texture2d>,
  destination: &mut Dest,
  draw_parameters: &glium::DrawParameters,
) {
  // building the vertex buffer, which contains all the vertices that we will draw
  let vertex_buffer = {
    #[derive(Copy, Clone)]
    struct Vertex {
      position: [f32; 2],
      tex_coords: [f32; 2],
    }

    implement_vertex!(Vertex, position, tex_coords);

    glium::VertexBuffer::new(
      window,
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
      window,
      glium::index::PrimitiveType::TriangleStrip,
      &[1 as u16, 2, 0, 3],
    ).unwrap();

  // compiling shaders and linking them together
  let program =
    program!(
      window,
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

  // building the uniforms
  let uniforms = uniform! {
    matrix: [
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0f32]
    ],
    tex: source,
  };

  destination.draw(
    &vertex_buffer,
    &index_buffer,
    &program,
    &uniforms,
    &draw_parameters,
  ).unwrap();
}
