#include "cl/mwc64x/cl/mwc64x.cl"

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
  float3 to_center = eye - center;
  float c = dot(to_center, to_center) - radius*radius;

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

mat4 screen_to_view(uint w, uint h, float fovy) {
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
  float diffuseness;
  float emittance;
  float reflectance;
  float transmittance;
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
  r.center        = parse_float3(&data);
  r.radius        = parse_float(&data);
  r.color         = parse_float3(&data);
  r.diffuseness   = parse_float(&data);
  r.emittance     = parse_float(&data);
  r.reflectance   = parse_float(&data);
  r.transmittance = parse_float(&data);
  return r;
}

typedef struct {
  float3 origin;
  float3 direction;
} Ray;

void raycast(
  Ray ray,

  __global const float* objects,
  const uint num_objects,

  float* toi,
  Object* collision
) {
  *toi = HUGE_VALF;

  for (uint i = 0; i < num_objects; ++i) {
    Object object = parse_object((Object*)objects + i);
    float this_toi = sphere_toi(ray.origin, ray.direction, pack_float3(object.center), object.radius);

    if (this_toi >= *toi) {
      continue;
    }

    *toi = this_toi;
    *collision = object;
  }
}

float rand(mwc64x_state_t* rand_state) {
  return (float)MWC64X_NextUint(rand_state) / (float)UINT_MAX;
}

float3 perturb_frame(mwc64x_state_t* rand_state, const float max_angle, float3 x, float3 y, float3 z) {
  float3 coeffs;
  const float c = cos(max_angle);
  coeffs.y = rand(rand_state) * (1 - c) + c;
  const float xz = sqrt(1 - coeffs.y * coeffs.y);
  const float azimuth = rand(rand_state) * 2 * 3.14;
  coeffs.x = cos(azimuth);
  coeffs.z = sin(azimuth);
  coeffs.x *= xz;
  coeffs.z *= xz;
  return coeffs.x*x + coeffs.y*y + coeffs.z*z;
}

float3 perturb(mwc64x_state_t* rand_state, const float3 unperturbed, const float3 normal, const float max_angle) {
  const float3 y = unperturbed;
  // TODO: find z/x better when normal ~= unperturbed
  const float3 z = normalize(cross(normal, y));
  const float3 x = normalize(cross(z, y));

  for (int i = 0; i < 4; ++i) {
    const float3 r = perturb_frame(rand_state, max_angle, x, y, z);
    if (dot(r, normal) >= 0) {
      return r;
    }
  }

  // If we failed several times, we're probably almost perpendicular to the normal.
  // I think that's a pretty small area, so we can just forget the perturb here.
  return unperturbed;
}

float3 pathtrace(
  Ray ray,
  uint max_depth,
  float3 ambient,
  mwc64x_state_t* rand_state,
  __global const float* objects,
  const uint num_objects
) {
  if (max_depth == 0) {
    return ambient;
  }

  float toi;
  Object collided_object;

  raycast(ray, objects, num_objects, &toi, &collided_object);

  if (toi == HUGE_VALF) {
    return ambient;
  }

  float3 light = ambient;

  light += (float3)(collided_object.emittance);

  const float3 collision_point = ray.origin + toi*ray.direction;
  const float3 normal = (collision_point - pack_float3(collided_object.center)) / collided_object.radius;
  const float max_scatter_angle = 3.14 * collided_object.diffuseness;

  float r = rand(rand_state);

  // TODO: loop + accumulators instead of recursion
  // TODO: consider removing ambient light and adding a large light "around" the whole world.

  r -= collided_object.transmittance;
  if (r <= 0) {
    // TODO: cos_theta < 0?
    Ray transmitted_ray;
    const float3 unperturbed = ray.direction;
    transmitted_ray.direction = perturb(rand_state, unperturbed, -normal, max_scatter_angle);
    transmitted_ray.origin = collision_point + 0.1f * transmitted_ray.direction;

    light +=
      pathtrace(
        transmitted_ray,
        max_depth - 1,
        ambient,
        rand_state,
        objects,
        num_objects
      );
  } else {
    r -= collided_object.reflectance;
    if (r <= 0) {
      // TODO: cos_theta < 0?
      Ray reflected_ray;
      const float cos_theta = dot(ray.direction, normal);
      const float3 unperturbed = ray.direction - 2*cos_theta*normal;
      reflected_ray.direction = perturb(rand_state, unperturbed, normal, max_scatter_angle);
      reflected_ray.origin = collision_point + 0.1f * reflected_ray.direction;

      light +=
        pathtrace(
          reflected_ray,
          max_depth - 1,
          ambient,
          rand_state,
          objects,
          num_objects
        );
    }
  }

  return pack_float3(collided_object.color) * light;
}

mwc64x_state_t init_rand_state(ulong random_seed) {
  mwc64x_state_t rand_state;
  rand_state.x = (uint)(random_seed & 0xFFFFFFFF);
  rand_state.c = (random_seed & 0xFFFFFFFF00000000) >> 32;
  return rand_state;
}

__kernel void render(
  const uint image_width,
  const uint image_height,

  const float fovy,
  float3 eye,
  const float3 look,
  const float3 up,

  ulong random_seed,
  float3 ambient_light,

  __global const float* objects,
  const uint num_objects,

  __global RGB * output)
{
  int id = get_global_id(0);

  const int x_pix = id % image_width;
  const int y_pix = id / image_width;

  float4 world_pos =
    vmult(view_to_world(eye, look, up),
    vmult(screen_to_view(image_width, image_height, fovy),
    (float4)(x_pix, y_pix, 1, 1)
  ));

  Ray ray;
  ray.origin = eye;
  ray.direction = normalize((world_pos / world_pos.w).xyz - eye);

  random_seed = random_seed * id * id;
  mwc64x_state_t rand_state = init_rand_state(random_seed);
  MWC64X_Skip(&rand_state, 20);

  output[id] = rgb(pathtrace(ray, 8, ambient_light, &rand_state, objects, num_objects));
}
