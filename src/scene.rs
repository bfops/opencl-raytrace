use crate::renderer::{ClFloat3, Object, Texture};

pub struct Camera {
    pub eye: ClFloat3,
    pub look: ClFloat3,
    pub up: ClFloat3,
    pub fovy: f32,
}

pub struct Scene {
    objects: Vec<Object>,
    camera: Camera,
}

impl Scene {
    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    pub fn objects(&self) -> &[Object] {
        &self.objects
    }

    pub fn forward(&self) -> [f32; 3] {
        self.camera.look.xyz()
    }

    pub fn right(&self) -> [f32; 3] {
        cross(self.camera.look.xyz(), self.camera.up.xyz())
    }

    pub fn move_camera(&mut self, delta: [f32; 3]) {
        self.camera.eye.add(delta);
    }
}

impl Default for Scene {
    fn default() -> Self {
        let solid = |r, g, b| Texture::solid_color(r, g, b);
        let object =
            |center, radius, emittance, reflectance, transmittance, diffuseness, texture| {
                Object::new(
                    center,
                    radius,
                    diffuseness,
                    emittance,
                    reflectance,
                    transmittance,
                    texture,
                )
            };

        Self {
            objects: vec![
                object(
                    [-4.0, -1.0, -5.0],
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    solid(1.0, 0.0, 0.0),
                ),
                object(
                    [-0.5, -1.0, -5.0],
                    1.0,
                    0.0,
                    0.1,
                    0.9,
                    0.01,
                    solid(0.0, 0.6, 1.0),
                ),
                object(
                    [-0.7, -0.5, -1.5],
                    0.5,
                    0.0,
                    0.1,
                    0.8,
                    0.02,
                    solid(0.9, 0.9, 1.0),
                ),
                object(
                    [0.2, -0.5, -1.0],
                    0.5,
                    0.0,
                    0.1,
                    0.9,
                    0.0,
                    solid(0.9, 0.9, 1.0),
                ),
                object(
                    [3.0, 1.5, -10.0],
                    4.0,
                    0.0,
                    1.0,
                    0.0,
                    0.1,
                    solid(1.0, 0.4, 0.1),
                ),
                object(
                    [3.0, -1.0, -3.5],
                    1.0,
                    0.0,
                    0.9,
                    0.0,
                    0.0,
                    solid(1.0, 1.0, 1.0),
                ),
                object(
                    [-9.0, 10.0, 0.0],
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    solid(0.9, 0.9, 1.0),
                ),
                object(
                    [0.0, 0.0, 0.0],
                    20.0,
                    0.2,
                    0.0,
                    0.0,
                    1.0,
                    solid(1.0, 1.0, 1.0),
                ),
                object(
                    [0.0, -102.0, 0.0],
                    100.0,
                    0.0,
                    1.0,
                    0.0,
                    0.02,
                    Texture::wood(),
                ),
            ],
            camera: Camera {
                eye: ClFloat3::new(0.0, 0.0, 0.0),
                look: ClFloat3::new(0.0, 0.0, -1.0),
                up: ClFloat3::new(0.0, 1.0, 0.0),
                fovy: std::f32::consts::FRAC_PI_2,
            },
        }
    }
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_camera_axes_and_movement_match_controls() {
        let mut scene = Scene::default();
        assert_eq!(scene.forward(), [0.0, 0.0, -1.0]);
        assert_eq!(scene.right(), [1.0, -0.0, 0.0]);

        scene.move_camera([1.0, 0.0, -1.0]);
        assert_eq!(scene.camera().eye.xyz(), [1.0, 0.0, -1.0]);
    }
}
