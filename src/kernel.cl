float4 vec4(float x, float y, float z, float w) {
  float4 r;
  r.x = x;
  r.y = y;
  r.z = z;
  r.w = w;
  return r;
}

float4 vec31(float3 xyz, float w) {
  float4 r;
  r.xyz = xyz;
  r.w = w;
  return r;
}

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

typedef struct {
  float4 row1;
  float4 row2;
  float4 row3;
  float4 row4;
} mat4;

float4 vmult(mat4 m, float4 v) {
  return
    vec4(
      dot(m.row1, v),
      dot(m.row2, v),
      dot(m.row3, v),
      dot(m.row4, v)
    );
}

mat4 screen_to_view(unsigned int w, unsigned int h, float fovy) {
  float aspect = (float)w / (float)h;

  float b = tan(fovy / 2);
  float a = 2 * b / h;
  b = -b;

  mat4 r;
  // We divide both x and y by h, to take aspect ratio into account.
  // An increase in image width will cause wider rays to be shot.
  r.row1 = vec4(a, 0, 0, b * aspect);
  r.row2 = vec4(0, a, 0, b);
  r.row3 = vec4(0, 0, 1, 0);
  r.row4 = vec4(0, 0, 0, 1);

  return r;
}

mat4 view_to_world(float3 eye, float3 look, float3 up) {
  float4 x = vec31(cross(look, up), 0);
  float4 y = vec31(up, 0);
  float4 z = vec31(look, 0);
  float4 w = vec31(eye, 1);

  // Change of basis

  mat4 r;
  r.row1.x = x.x;
  r.row2.x = x.y;
  r.row3.x = x.z;
  r.row4.x = x.w;

  r.row1.y = y.x;
  r.row2.y = y.y;
  r.row3.y = y.z;
  r.row4.y = y.w;

  r.row1.z = z.x;
  r.row2.z = z.y;
  r.row3.z = z.z;
  r.row4.z = z.w;

  r.row1.w = w.x;
  r.row2.w = w.y;
  r.row3.w = w.z;
  r.row4.w = w.w;

  return r;
}

typedef struct {
  float r;
  float g;
  float b;
} RGB;

__kernel void render(
  const unsigned int window_width,
  const unsigned int window_height,

  const float3 obj1_center,
  const float obj1_radius,
  const float3 obj2_center,
  const float obj2_radius,

  const float fovy,
  float3 eye,
  const float3 look,
  const float3 up,

  __global RGB * output)
{
  int i = get_global_id(0);

  const int x_pix = i % window_width;
  const int y_pix = i / window_width;

  float4 world_pos =
    vmult(view_to_world(eye, look, up),
    vmult(screen_to_view(window_width, window_height, fovy),
    vec4(x_pix, y_pix, 1, 1)
  ));

  float3 ray = normalize((world_pos / world_pos.w).xyz - eye);

  float3 color = {1, 1, 1};

  int max_bounces = 1;
  // The number of casts is the number of bounces - 1.
  for (int cast = 0; cast <= max_bounces; ++cast) {
    float toi1 = toi(eye, ray, obj1_center, obj1_radius);
    float toi2 = toi(eye, ray, obj2_center, obj2_radius);

    if (toi1 == HUGE_VALF && toi2 == HUGE_VALF) {
      output[i].r = 0;
      output[i].g = 0;
      output[i].b = 0;
      return;
    }

    if (toi1 < toi2) {
      float3 obj_color = {1, 0, 0};
      color *= obj_color;

      float3 intersection = eye + toi1 * ray;
      float3 normal = normalize(intersection - obj1_center);
      float directness = -dot(normal, ray) / length(ray);

      if (directness < 0) {
        directness = 0;
      }

      color *= directness;

      eye = intersection;
      ray = rotate_vec(-look, normal);
    } else {
      // We hit the light.

      output[i].r = color[0];
      output[i].g = color[1];
      output[i].b = color[2];
      return;
    }
  }

  output[i].r = 1;
  output[i].g = 0;
  output[i].b = 1;
}
