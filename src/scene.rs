use main::{WINDOW_WIDTH, WINDOW_HEIGHT, RGB};
use opencl;
use opencl::mem::CLBuffer;

pub struct Scene {
  pub center: [f32; 3],
  pub radius: f32,
  pub camera: [f32; 3],
}

impl Scene {
  pub fn render(&self) -> Vec<RGB> {
    let (device, ctx, queue) = opencl::util::create_compute_context().unwrap();

    let len = WINDOW_WIDTH as usize * WINDOW_HEIGHT as usize;

    let output_buffer: CLBuffer<RGB> = ctx.create_buffer(len, opencl::cl::CL_MEM_WRITE_ONLY);

    let program = {
      let ker = format!("
        __kernel void render(
          // sphere
          const float3 center,
          const float radius,

          const float3 eye,

          __global float * output)
        {{
          int W = {};
          int H = {};

          float fov_x = 3.14 / 2;
          float fov_y = 3.14 / 2;

          int i = get_global_id(0);

          float x_pix = i % W;
          float y_pix = i / W;

          float t_x = fov_x * (x_pix / W - 0.5);
          float t_y = fov_y * (y_pix / H - 0.5);

          float c = -cos(t_y);
          float3 ray = {{c*sin(t_x), sin(t_y), c*cos(t_x)}};
          float a = dot(ray, center - eye) / dot(ray, ray);
          float3 d = eye + a*ray - center;

          i = i * 3;
          if (dot(d, d) <= radius * radius) {{
            output[i+0] = 1;
            output[i+1] = 0;
            output[i+2] = 0;
          }} else {{
            output[i+0] = 0;
            output[i+1] = 0;
            output[i+2] = 0;
          }}
        }}",
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
      );
      ctx.create_program_from_source(ker.as_slice())
    };
    program.build(&device).unwrap();

    let kernel = program.create_kernel("render");
    kernel.set_arg(0, &self.center);
    kernel.set_arg(1, &self.radius);
    kernel.set_arg(2, &self.camera);

    // This is sketchy; we "implicitly cast" output_buffer from a CLBuffer<RGB> to a CLBuffer<f32>.
    kernel.set_arg(3, &output_buffer);

    let event = queue.enqueue_async_kernel(&kernel, len, None, ());

    queue.get(&output_buffer, &event)
  }
}
