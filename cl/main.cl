#include "cl/mwc64x/cl/mwc64x.cl"

float sphere_toi(
  const float3* eye,
  const float3* look,

  const __global float3* center,
  const float radius)
{
  // quadratic coefficients
  float a = dot(*look, *look);
  float3 to_center = *eye - *center;
  float b = 2 * dot(to_center, *look);
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

typedef struct {
  float3 center;
  float radius;
  float3 color;
  float diffuseness;
  float emittance;
  float reflectance;
  float transmittance;
} Object;

typedef struct {
  float3 origin;
  float3 direction;
} Ray;

void raycast(
  const Ray* ray,

  __global const Object* objects,
  const uint num_objects,

  float* const toi,
  __global const Object** const collision
) {
  *toi = HUGE_VALF;

  for (uint i = 0; i < num_objects; ++i) {
    __global const Object* const object = &objects[i];
    float this_toi = sphere_toi(&ray->origin, &ray->direction, &object->center, object->radius);

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

float3 perturb_frame(mwc64x_state_t* rand_state, const float max_angle, const float3* x, const float3* y, const float3* z) {
  float3 coeffs;
  const float c = cos(max_angle);
  coeffs.y = rand(rand_state) * (1 - c) + c;
  const float xz = sqrt(1 - coeffs.y * coeffs.y);
  const float azimuth = rand(rand_state) * 2 * 3.14;
  coeffs.x = cos(azimuth);
  coeffs.z = sin(azimuth);
  coeffs.x *= xz;
  coeffs.z *= xz;
  return coeffs.x * *x + coeffs.y * *y + coeffs.z * *z;
}

float3 perturb(mwc64x_state_t* rand_state, const float3* unperturbed, const float3* normal, const float max_angle) {
  const float3* y = unperturbed;
  // TODO: find z/x better when normal ~= unperturbed
  const float3 z = normalize(cross(*normal, *y));
  const float3 x = normalize(cross(z, *y));

  for (int i = 0; i < 8; ++i) {
    const float3 r = perturb_frame(rand_state, max_angle, &x, y, &z);
    if (dot(r, *normal) >= 0) {
      return r;
    }
  }

  // If we failed several times, we're probably almost perpendicular to the normal.
  // I think that's a pretty small area, so we can just forget the perturb here.
  return *unperturbed;
}

// If the ray is absorbed (i.e. there is no next path), this returns false.
bool pick_next_path(
  mwc64x_state_t* const rand_state,
  const Ray* const ray,
  const float3* const collision_point,
  const float3* const normal,
  const __global Object* const collided_object,
  float3* const new_direction,
  float3* const new_normal
) {
  float r = rand(rand_state);

  r -= collided_object->transmittance;
  if (r <= 0) {
    *new_direction = ray->direction;
    *new_normal = -*normal;
    return true;
  }

  r -= collided_object->reflectance;
  if (r <= 0) {
    const float cos_theta = dot(ray->direction, *normal);
    *new_direction = ray->direction - 2 * cos_theta * *normal;
    *new_normal = *normal;
    return true;
  }

  return false;
}

float3 pathtrace(
  Ray ray,
  const uint max_depth,
  const float min_attenuation,
  mwc64x_state_t* rand_state,
  __global const Object* objects,
  const uint num_objects
) {
  float3 pixel_color = (float3)(0, 0, 0);
  float3 attenuation = (float3)(1, 1, 1);

  for (unsigned int i = 0; i < max_depth; ++i) {
    float toi;
    const __global Object* collided_object;

    raycast(&ray, objects, num_objects, &toi, &collided_object);

    if (toi == HUGE_VALF) {
      break;
    }

    attenuation *= collided_object->color;

    if (attenuation.x < min_attenuation && attenuation.y < min_attenuation && attenuation.z < min_attenuation) {
      break;
    }

    pixel_color += attenuation * (float3)(collided_object->emittance);

    const float3 collision_point = ray.origin + toi*ray.direction;
    const float3 normal = (collision_point - collided_object->center) / collided_object->radius;

    float3 new_direction;
    float3 new_normal;
    if (!pick_next_path(rand_state, &ray, &collision_point, &normal, collided_object, &new_direction, &new_normal)) {
      // Ray is absorbed.
      break;
    }

    const float max_scatter_angle = 3.14 * collided_object->diffuseness;

    // TODO: maybe instead of a max_scatter_angle, we could describe a probability distribution over the scatter angle.
    ray.direction = perturb(rand_state, &new_direction, &new_normal, max_scatter_angle);
    ray.origin = collision_point + 0.1f * ray.direction;
  }

  return pixel_color;
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

  __global const Object* objects,
  const uint num_objects,

  __global RGB * output)
{
  const int id = get_global_id(0);

  const int x_pix = id % image_width;
  const int y_pix = id / image_width;

  const float4 world_pos =
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

  output[id] = rgb(pathtrace(ray, 20, 0.01, &rand_state, objects, num_objects));
}
