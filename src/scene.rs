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
          const float center_x,
          const float center_y,
          const float center_z,
          const float radius,

          const float camera_x,
          const float camera_y,
          const float camera_z,
          __global float * output)
        {{
          int W = {};
          int H = {};

          int i = get_global_id(0);

          float x = i % W;
          float y = i / W;

          x = camera_x + 2 * (x / W) - 1;
          y = camera_y + 2 * (y / H) - 1;
          float z = camera_z;

          float dx = x - center_x;
          float dy = y - center_y;
          float dz = z - center_z;

          float distance = dx*dx + dy*dy + dz*dz;
          i = i * 3;
          if (distance <= radius * radius) {{
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
    kernel.set_arg(0, &self.center[0]);
    kernel.set_arg(1, &self.center[1]);
    kernel.set_arg(2, &self.center[2]);
    kernel.set_arg(3, &self.radius);

    kernel.set_arg(4, &self.camera[0]);
    kernel.set_arg(5, &self.camera[1]);
    kernel.set_arg(6, &self.camera[2]);

    // This is sketchy; we "implicitly cast" output_buffer from a CLBuffer<RGB> to a CLBuffer<f32>.
    kernel.set_arg(7, &output_buffer);

    let event = queue.enqueue_async_kernel(&kernel, len, None, ());

    queue.get(&output_buffer, &event)
  }
}
