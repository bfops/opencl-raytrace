use gl;
use scene::Scene;
use sdl2;
use sdl2::event::Event;
use std;
use std::mem;
use stopwatch::TimerSet;
use yaglw::gl_context::GLContext;
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

  let mut sdl = sdl2::init().everything().unwrap();
  let window = make_window(&sdl);

  let _sdl_gl = window.gl_create_context().unwrap();

  let mut event_pump = sdl.event_pump();

  // Load the OpenGL function pointers.
  gl::load_with(|s| unsafe {
    mem::transmute(sdl2::video::gl_get_proc_address(s))
  });

  let mut gl = unsafe {
    GLContext::new()
  };

  match gl.get_error() {
    gl::NO_ERROR => {},
    err => {
      println!("OpenGL error 0x{:x} in setup", err);
      return;
    },
  }

  let shader = make_shader(&gl);
  shader.use_shader(&mut gl);

  let mut vao = make_vao(&mut gl, &shader);
  vao.bind(&mut gl);

  let scene = {
    Scene {
      objects: vec!(
        0.1, 0.7, -1.0,    16.0,  1.0, 1.0, 1.0,
      ),
      lights: vec!(
        0.0,  0.0, 2.0,   1.0,    0.0, 0.0, 1.0,
      ),
      camera: [-0.2, 0.3, 0.4],
    }
  };

  timers.time("update", || {
    vao.push(&mut gl, scene.render().as_slice());
  });

  while process_events(&mut event_pump) {
    timers.time("draw", || {
      gl.clear_buffer();
      vao.draw(&mut gl);
      // swap buffers
      window.gl_swap_window();
    });

    std::thread::sleep_ms(10);
  }

  timers.print();
}

fn make_shader<'a, 'b:'a>(
  gl: &'a GLContext,
) -> Shader<'b> {
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

fn make_window(sdl: &sdl2::Sdl) -> sdl2::video::Window {
  sdl2::video::gl_attr::set_context_profile(sdl2::video::GLProfile::Core);
  sdl2::video::gl_attr::set_context_version(3, 3);

  // Open the window as fullscreen at the current resolution.
  let mut window =
    sdl2::video::WindowBuilder::new(
      &sdl,
      "Raytrace",
      WINDOW_WIDTH, WINDOW_HEIGHT,
    );

  let window = window.position_centered();
  window.opengl();

  window.build().unwrap()
}

fn make_vao<'a, 'b:'a>(
  gl: &'a mut GLContext,
  shader: &Shader<'b>,
) -> GLArray<'b, RGB> {
  let attribs = [
    VertexAttribData {
      name: "color",
      size: 3,
      unit: GLType::Float,
    },
  ];

  let capacity = WINDOW_WIDTH as usize * WINDOW_HEIGHT as usize;
  let vbo = GLBuffer::new(gl, capacity);

  GLArray::new(
    gl,
    shader,
    &attribs,
    DrawMode::Points,
    vbo,
  )
}

fn process_events(event_pump: &mut sdl2::event::EventPump) -> bool {
  loop {
    match event_pump.poll_event() {
      None => {
        return true;
      },
      Some(Event::Quit {..}) => {
        return false;
      },
      Some(Event::AppTerminating {..}) => {
        return false;
      },
      _ => {},
    }
  }
}
