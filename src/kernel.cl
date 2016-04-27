// Doesn't return 0 so that rays can bounce without bumping.
float sphere_toi(
  const float3 eye,
  const float3 look,

  const float3 center,
  const float radius)
{
  // quadratic coefficients
  float a = dot(look, look);
  float b = 2 * dot(eye - center, look);
  float c = dot(center, center) + dot(eye, eye) - dot(eye, center) - radius * radius;

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
    (float4)(
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
  r.row1 = (float4)(a, 0, 0, b * aspect);
  r.row2 = (float4)(0, a, 0, b);
  r.row3 = (float4)(0, 0, 1, 0);
  r.row4 = (float4)(0, 0, 0, 1);

  return r;
}

mat4 view_to_world(float3 eye, float3 look, float3 up) {
  float4 x = (float4)(cross(look, up), 0);
  float4 y = (float4)(up, 0);
  float4 z = (float4)(look, 0);
  float4 w = (float4)(eye, 1);

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

RGB rgb(float3 xyz) {
  RGB r;
  r.r = xyz.x;
  r.g = xyz.y;
  r.b = xyz.z;
  return r;
}

typedef struct { float x, y, z } float3_parse;

float3 pack_float3(float3_parse f) {
  return (float3)(f.x, f.y, f.z);
}

typedef struct {
  float3_parse center;
  float radius;
  float3_parse color;
} Object;

float parse_float(__global const float** data) {
  float r = **data;
  ++*data;
  return r;
}

float3_parse parse_float3(__global const float** data) {
  float3_parse r;
  r.x = parse_float(data);
  r.y = parse_float(data);
  r.z = parse_float(data);
  return r;
}

Object parse_object(__global const float* data) {
  Object r;
  r.center = parse_float3(&data);
  r.radius = parse_float(&data);
  r.color = parse_float3(&data);
  return r;
}

typedef struct {
  float3 origin;
  float3 direction;
} Ray;

void raycast(
  Ray ray,
  
  __global const float* objects,
  const unsigned int num_objects,

  float* toi,
  Object* collision
) {
  *toi = HUGE_VALF;

  for (unsigned int i = 0; i < num_objects; ++i) {
    Object object = parse_object((Object*)objects + i);
    float this_toi = sphere_toi(ray.origin, ray.direction, pack_float3(object.center), object.radius);

    if (this_toi >= *toi) {
      continue;
    }

    *toi = this_toi;
    *collision = object;
  }
}

float3 raytrace(
  Ray ray,

  __global const float* objects,
  const unsigned int num_objects
) {
  float toi;
  Object collision;

  raycast(ray, objects, num_objects, &toi, &collision);

  if (toi == HUGE_VALF) {
    return (float3)(0, 0, 0);
  }

  return pack_float3(collision.color);
}

__kernel void render(
  const unsigned int window_width,
  const unsigned int window_height,

  const float fovy,
  float3 eye,
  const float3 look,
  const float3 up,

  __global const float* objects,
  const unsigned int num_objects,

  __global RGB * output)
{
  int id = get_global_id(0);

  const int x_pix = id % window_width;
  const int y_pix = id / window_width;

  float4 world_pos =
    vmult(view_to_world(eye, look, up),
    vmult(screen_to_view(window_width, window_height, fovy),
    (float4)(x_pix, y_pix, 1, 1)
  ));

  Ray ray;
  ray.origin = eye;
  ray.direction = normalize((world_pos / world_pos.w).xyz - eye);

  output[id] = rgb(raytrace(ray, objects, num_objects));
}
