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
        float toi(
          const float3 eye,
          const float3 look,

          const float3 sphere,
          const float radius)
        {{
          float a = dot(radius, radius);
          float b = 2 * dot(eye - sphere, look);
          float c = dot(sphere, sphere) + dot(eye, eye) - dot(eye, sphere) - radius * radius;

          float d = b*b - 4*a*c;

          if (d < 0) {{
            return HUGE_VALF;
          }}

          return (-sqrt(d) - b) / (2 * a);
        }}

        __kernel void render(
          const float3 obj_center,
          const float obj_radius,

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

          float c = cos(t_y);
          float3 ray = {{c*sin(t_x), sin(t_y), -c*cos(t_x)}};

          float t = toi(eye, ray, obj_center, obj_radius);

          i = i * 3;
          if (t == HUGE_VALF) {{
            output[i+0] = 0;
            output[i+1] = 0;
            output[i+2] = 0;
            return;
          }}

          output[i+0] = 1;
          output[i+1] = 0;
          output[i+2] = 0;
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
