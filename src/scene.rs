use opencl;
use opencl::mem::CLBuffer;
use std::borrow::Borrow;

use main::{WINDOW_WIDTH, WINDOW_HEIGHT, RGB};

pub struct Scene {
  pub objects: Vec<f64>,
  pub lights: Vec<f64>,
  pub camera: [f64; 3],
}

impl Scene {
  pub fn render(&self) -> Vec<RGB> {
    let (device, ctx, queue) = opencl::util::create_compute_context().unwrap();

    let program = {
      let ker = format!("
        // Doesn't return 0 so that rays can bounce without bumping.
        double toi(
          const double3 eye,
          const double3 look,

          const double3 sphere,
          const double radius)
        {{
          // quadratic coefficients
          double a = dot(look, look);
          double b = 2 * dot(eye - sphere, look);
          double c = dot(sphere, sphere) + dot(eye, eye) - dot(eye, sphere) - radius * radius;

          // discriminant
          double d = b*b - 4*a*c;

          if (d < 0) {{
            return HUGE_VALF;
          }}

          double s1 = (sqrt(d) - b) / (2 * a);
          if (s1 <= 0) {{
            s1 = HUGE_VALF;
          }}

          double s2 = (-sqrt(d) - b) / (2 * a);
          if (s1 < s2 || s2 <= 0) {{
            return s1;
          }}

          return s2;
        }}

        double3 rotate_vec(
          const double3 vec,
          const double3 axis)
        {{
          double3 r1 = {{2*axis[0]*axis[0] - 1, 2*axis[0]*axis[1], 2*axis[0]*axis[2]}};
          double3 r2 = {{2*axis[0]*axis[1], 2*axis[1]*axis[1] - 1, 2*axis[1]*axis[2]}};
          double3 r3 = {{2*axis[0]*axis[2], 2*axis[1]*axis[2], 2*axis[2]*axis[2] - 1}};

          double3 r = {{dot(r1, vec), dot(r2, vec), dot(r3, vec)}};
          return r;
        }}

        __kernel void render(
          __global const double* objects,
          const int n_objects,
          __global const double* lights,
          const int n_lights,

          double3 eye,

          __global float * output)
        {{
          int W = {};
          int H = {};

          double fov_x = 3.14 / 2;
          double fov_y = 3.14 / 2;

          int i = get_global_id(0);

          double x_pix = i % W;
          double y_pix = i / W;

          double t_x = fov_x * (x_pix / W - 0.5);
          double t_y = fov_y * (y_pix / H - 0.5);

          double c = cos(t_y);
          double3 look = {{c*sin(t_x), sin(t_y), -c*cos(t_x)}};

          i = i * 3;
          double3 color = {{1, 1, 1}};

          int max_bounces = 128;
          // The number of casts is the number of bounces - 1.
          for (int cast = 0; cast <= max_bounces; ++cast) {{
            double min_toi = HUGE_VALF;
            double3 min_object = {{0, 0, 0}};
            double3 min_color = {{0, 0, 0}};

            for (int j = 0; j < n_objects; ++j) {{
              int idx = j * 7;
              double3 center = {{objects[idx], objects[idx+1], objects[idx+2]}};
              double radius = objects[idx+3];
              double3 obj_color = {{objects[idx+4], objects[idx+5], objects[idx+6]}};
              double cur_toi = toi(eye, look, center, radius);

              if (cur_toi < min_toi) {{
                min_toi = cur_toi;
                min_object = center;
                min_color = obj_color;
              }}
            }}

            bool min_light = false;

            for (int j = 0; j < n_lights; ++j) {{
              int idx = j * 7;
              double3 center = {{lights[idx], lights[idx+1], lights[idx+2]}};
              double radius = lights[idx+3];
              double3 light_color = {{lights[idx+4], lights[idx+5], lights[idx+6]}};
              double cur_toi = toi(eye, look, center, radius);

              if (cur_toi < min_toi) {{
                min_toi = cur_toi;
                min_light = true;
                min_color = light_color;
              }}
            }}

            if (min_toi == HUGE_VALF) {{
              output[i+0] = 0;
              output[i+1] = 0;
              output[i+2] = 0;
              return;
            }}

            color *= min_color;

            if (min_light) {{
              // We hit a light.

              output[i+0] = color[0];
              output[i+1] = color[1];
              output[i+2] = color[2];
              return;
            }}

            double3 intersection = eye + min_toi * look;
            double3 normal = normalize(intersection - min_object);
            double directness = dot(normal, look);
            if (directness < 0) {{
              directness = -directness;
            }} else {{
              normal = -normal;
            }}
            directness = fmin(directness, 1);
            color *= directness;

            eye = intersection;
            look = rotate_vec(-look, normal);
          }}

          output[i+0] = 1;
          output[i+1] = 0;
          output[i+2] = 1;
        }}",
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
      );
      ctx.create_program_from_source(ker.borrow())
    };
    program.build(&device).unwrap();

    let len = WINDOW_WIDTH as usize * WINDOW_HEIGHT as usize;

    let objects: CLBuffer<f64> =
      ctx.create_buffer(self.objects.len(), opencl::cl::CL_MEM_READ_ONLY);
    queue.write(&objects, &self.objects.as_slice(), ());
    let n_objects = self.objects.len() as i32 / 7;

    let lights: CLBuffer<f64> =
      ctx.create_buffer(self.lights.len(), opencl::cl::CL_MEM_READ_ONLY);
    queue.write(&lights, &self.lights.as_slice(), ());
    let n_lights = self.lights.len() as i32 / 4;

    let output_buffer: CLBuffer<RGB> = ctx.create_buffer(len, opencl::cl::CL_MEM_READ_WRITE);

    let kernel = program.create_kernel("render");
    kernel.set_arg(0, &objects);
    kernel.set_arg(1, &n_objects);
    kernel.set_arg(2, &lights);
    kernel.set_arg(3, &n_lights);
    kernel.set_arg(4, &self.camera);

    // This is sketchy; we "implicitly cast" output_buffer from a CLBuffer<RGB> to a CLBuffer<f32>.
    kernel.set_arg(5, &output_buffer);

    let event = queue.enqueue_async_kernel(&kernel, len, None, ());

    queue.get(&output_buffer, &event)
  }
}
