use main::{WINDOW_WIDTH, WINDOW_HEIGHT, RGB};
use opencl;
use opencl::mem::CLBuffer;

pub struct Scene {
  pub obj1_center: [f32; 3],
  pub obj1_radius: f32,
  pub obj2_center: [f32; 3],
  pub obj2_radius: f32,
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
          const float3 obj1_center,
          const float obj1_radius,
          const float3 obj2_center,
          const float obj2_radius,

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

          float toi1 = toi(eye, ray, obj1_center, obj1_radius);
          float toi2 = toi(eye, ray, obj2_center, obj2_radius);

          i = i * 3;
          if (toi1 == HUGE_VALF && toi2 == HUGE_VALF) {{
            output[i+0] = 0;
            output[i+1] = 0;
            output[i+2] = 0;
            return;
          }}

          if (toi1 < toi2) {{
            output[i+0] = 1;
            output[i+1] = 0;
            output[i+2] = 0;
          }} else {{
            output[i+0] = 0;
            output[i+1] = 1;
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
    kernel.set_arg(0, &self.obj1_center);
    kernel.set_arg(1, &self.obj1_radius);
    kernel.set_arg(2, &self.obj2_center);
    kernel.set_arg(3, &self.obj2_radius);
    kernel.set_arg(4, &self.camera);

    // This is sketchy; we "implicitly cast" output_buffer from a CLBuffer<RGB> to a CLBuffer<f32>.
    kernel.set_arg(5, &output_buffer);

    let event = queue.enqueue_async_kernel(&kernel, len, None, ());

    queue.get(&output_buffer, &event)
  }
}
