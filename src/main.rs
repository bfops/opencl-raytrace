use gl;
use scene::Scene;
use sdl2;
use sdl2::event::Event;
use std::io::timer;
use std::mem;
use std::time::duration::Duration;
use stopwatch::TimerSet;
use yaglw::gl_context::{GLContext, GLContextExistence};
use yaglw::shader::Shader;
use yaglw::vertex_buffer::{GLArray, GLBuffer, GLType, VertexAttribData, DrawMode};

pub const WINDOW_WIDTH: u32 = 800;
pub const WINDOW_HEIGHT: u32 = 800;

#[repr(C)]
pub struct RGB {
  pub r: f32,
  pub g: f32,
  pub b: f32,
}

pub fn main() {
  let timers = TimerSet::new();

  let window = make_window();

  let _sdl_gl_context = window.gl_create_context().unwrap();

  // Load the OpenGL function pointers.
  gl::load_with(|s| unsafe {
    mem::transmute(sdl2::video::gl_get_proc_address(s))
  });

  let (gl, mut gl_context) = unsafe {
    GLContext::new()
  };

  match gl_context.get_error() {
    gl::NO_ERROR => {},
    err => {
      println!("OpenGL error 0x{:x} in setup", err);
      return;
    },
  }

  let shader = make_shader(&gl);
  shader.use_shader(&mut gl_context);

  let mut vao = make_vao(&gl, &mut gl_context, &shader);
  vao.bind(&mut gl_context);

  let scene =
    Scene {
      center: [0.0, 0.0, -0.5],
      radius: 1.0,
      camera: [0.0, 0.0, 0.0],
    };

  timers.time("update", || {
    vao.push(&mut gl_context, scene.render().as_slice());
  });

  while process_events() {
    timers.time("draw", || {
      gl_context.clear_buffer();
      vao.draw(&mut gl_context);
      // swap buffers
      window.gl_swap_window();
    });

    timer::sleep(Duration::milliseconds(10));
  }

  timers.print();
}

fn make_shader<'a>(
  gl: &'a GLContextExistence,
) -> Shader<'a> {
  let vertex_shader: String = format!("
    #version 330 core

    const int W = {};
    const int H = {};

    in vec3 color;
    out vec4 v_color;

    void main() {{
      v_color = vec4(color, 1);
      gl_Position =
        vec4(
          float(gl_VertexID % W) / W * 2 - 1,
          float(gl_VertexID / W) / H * 2 - 1,
          0, 1
        );
    }}
  ", WINDOW_WIDTH, WINDOW_HEIGHT);

  let fragment_shader: String = "
    #version 330 core

    in vec4 v_color;

    void main() {
      gl_FragColor = v_color;
    }
  ".to_string();

  let components = vec!(
    (gl::VERTEX_SHADER, vertex_shader),
    (gl::FRAGMENT_SHADER, fragment_shader),
  );

  Shader::new(gl, components.into_iter())
}

fn make_window() -> sdl2::video::Window {
  sdl2::init(sdl2::INIT_EVERYTHING);

  sdl2::video::gl_set_attribute(sdl2::video::GLAttr::GLContextMajorVersion, 3);
  sdl2::video::gl_set_attribute(sdl2::video::GLAttr::GLContextMinorVersion, 3);
  sdl2::video::gl_set_attribute(
    sdl2::video::GLAttr::GLContextProfileMask,
    sdl2::video::GLProfile::GLCoreProfile as isize
  );

  let window = sdl2::video::Window::new(
    "OpenCL",
    sdl2::video::WindowPos::PosCentered,
    sdl2::video::WindowPos::PosCentered,
    WINDOW_WIDTH as isize,
    WINDOW_HEIGHT as isize,
    sdl2::video::OPENGL,
  ).unwrap();

  window
}

fn make_vao<'a>(
  gl: &'a GLContextExistence,
  gl_context: &mut GLContext,
  shader: &Shader<'a>,
) -> GLArray<'a, RGB> {
  let attribs = [
    VertexAttribData {
      name: "color",
      size: 3,
      unit: GLType::Float,
    },
  ];

  let capacity = WINDOW_WIDTH as usize * WINDOW_HEIGHT as usize;
  let vbo = GLBuffer::new(gl, gl_context, capacity);

  GLArray::new(
    gl,
    gl_context,
    shader,
    &attribs,
    DrawMode::Points,
    vbo,
  )
}

fn process_events<'a>() -> bool {
  loop {
    match sdl2::event::poll_event() {
      Event::None => {
        return true;
      },
      Event::Quit(_) => {
        return false;
      },
      Event::AppTerminating(_) => {
        return false;
      },
      _ => {},
    }
  }
}
