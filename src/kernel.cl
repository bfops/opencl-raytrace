// Doesn't return 0 so that rays can bounce without bumping.
float toi(
  const float3 eye,
  const float3 look,

  const float3 sphere,
  const float radius)
{
  // quadratic coefficients
  float a = dot(look, look);
  float b = 2 * dot(eye - sphere, look);
  float c = dot(sphere, sphere) + dot(eye, eye) - dot(eye, sphere) - radius * radius;

  // discriminant
  float d = b*b - 4*a*c;

  if (d < 0) {
    return HUGE_VALF;
  }

  float s1 = (sqrt(d) - b) / (2 * a);
  if (s1 <= 0) {
    s1 = HUGE_VALF;
  }

  float s2 = (-sqrt(d) - b) / (2 * a);
  if (s1 < s2 || s2 <= 0) {
    return s1;
  }

  return s2;
}

float3 rotate_vec(
  const float3 vec,
  const float3 axis)
{
  float3 r1 = {2*axis[0]*axis[0] - 1, 2*axis[0]*axis[1], 2*axis[0]*axis[2]};
  float3 r2 = {2*axis[0]*axis[1], 2*axis[1]*axis[1] - 1, 2*axis[1]*axis[2]};
  float3 r3 = {2*axis[0]*axis[2], 2*axis[1]*axis[2], 2*axis[2]*axis[2] - 1};

  float3 r = {dot(r1, vec), dot(r2, vec), dot(r3, vec)};
  return r;
}

__kernel void render(
  const unsigned int window_width,
  const unsigned int window_height,

  const float3 obj1_center,
  const float obj1_radius,
  const float3 obj2_center,
  const float obj2_radius,

  float3 eye,

  __global float * output)
{
  float fov_x = 3.14 / 2;
  float fov_y = 3.14 / 2;

  int i = get_global_id(0);

  float x_pix = i % window_width;
  float y_pix = i / window_width;

  float t_x = fov_x * (x_pix / window_width - 0.5);
  float t_y = fov_y * (y_pix / window_height - 0.5);

  float c = cos(t_y);
  float3 look = {c*sin(t_x), sin(t_y), -c*cos(t_x)};

  i = i * 3;
  float3 color = {1, 1, 1};

  int max_bounces = 1;
  // The number of casts is the number of bounces - 1.
  for (int cast = 0; cast <= max_bounces; ++cast) {
    float toi1 = toi(eye, look, obj1_center, obj1_radius);
    float toi2 = toi(eye, look, obj2_center, obj2_radius);

    if (toi1 == HUGE_VALF && toi2 == HUGE_VALF) {
      output[i+0] = 0;
      output[i+1] = 0;
      output[i+2] = 0;
      return;
    }

    if (toi1 < toi2) {
      float3 obj_color = {1, 0, 0};
      color *= obj_color;

      float3 intersection = eye + toi1 * look;
      float3 normal = normalize(intersection - obj1_center);
      float directness = -dot(normal, look) / length(look);

      if (directness < 0) {
        directness = 0;
      }

      color *= directness;

      eye = intersection;
      look = rotate_vec(-look, normal);
    } else {
      // We hit the light.

      output[i+0] = color[0];
      output[i+1] = color[1];
      output[i+2] = color[2];
      return;
    }
  }

  output[i+0] = 1;
  output[i+1] = 0;
  output[i+2] = 1;
}
