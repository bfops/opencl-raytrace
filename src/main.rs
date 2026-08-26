mod renderer;
mod scene;

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use minifb::{Key, Window, WindowOptions};

use renderer::{OpenClRenderer, Rgb};
use scene::Scene;

const WIDTH: usize = 800;
const HEIGHT: usize = 600;
const MAX_MOVEMENT_SPEED: f32 = 2.0;
const MOVEMENT_ACCELERATION: f32 = 8.0;
const MOVEMENT_DECELERATION: f32 = 10.0;
const MAX_MOVEMENT_DELTA_SECONDS: f32 = 0.1;

fn main() -> Result<()> {
    let mut scene = Scene::default();
    let mut renderer = OpenClRenderer::new(WIDTH, HEIGHT, &scene)
        .context("failed to initialize the OpenCL renderer")?;
    let mut accumulation = Accumulation::new(WIDTH * HEIGHT);

    println!("OpenCL device: {}", renderer.device_name());
    println!("Controls: W/A/S/D move, Escape exits");

    let mut window = Window::new(
        "OpenCL Ray Tracer",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: false,
            ..WindowOptions::default()
        },
    )
    .context("failed to create the display window")?;
    window.set_target_fps(60);

    let mut movement = MovementController::default();
    let mut last_movement_update = Instant::now();
    let mut last_title_update = Instant::now();
    while window.is_open() && !window.is_key_down(Key::Escape) {
        let now = Instant::now();
        let elapsed = now.duration_since(last_movement_update).as_secs_f32();
        last_movement_update = now;

        if move_camera_from_input(&window, &mut scene, &mut movement, elapsed) {
            accumulation.reset();
        }

        let started = Instant::now();
        let seed = sample_seed(accumulation.samples());
        let sample = renderer
            .render(scene.camera(), seed)
            .context("OpenCL frame rendering failed")?;
        accumulation.add_sample(sample);
        let render_time = started.elapsed();

        window
            .update_with_buffer(accumulation.display_pixels(), WIDTH, HEIGHT)
            .context("failed to update the display window")?;

        if last_title_update.elapsed() >= Duration::from_millis(250) {
            window.set_title(&format!(
                "OpenCL Ray Tracer — {} — {} samples — {:.1} ms",
                renderer.device_name(),
                accumulation.samples(),
                render_time.as_secs_f64() * 1_000.0,
            ));
            last_title_update = Instant::now();
        }
    }

    Ok(())
}

fn move_camera_from_input(
    window: &Window,
    scene: &mut Scene,
    movement: &mut MovementController,
    elapsed: f32,
) -> bool {
    let mut direction = [0.0; 3];

    if window.is_key_down(Key::W) {
        add_scaled(&mut direction, scene.forward(), 1.0);
    }
    if window.is_key_down(Key::S) {
        add_scaled(&mut direction, scene.forward(), -1.0);
    }
    if window.is_key_down(Key::D) {
        add_scaled(&mut direction, scene.right(), 1.0);
    }
    if window.is_key_down(Key::A) {
        add_scaled(&mut direction, scene.right(), -1.0);
    }

    let displacement = movement.update(direction, elapsed);
    if displacement == [0.0; 3] {
        false
    } else {
        scene.move_camera(displacement);
        true
    }
}

#[derive(Default)]
struct MovementController {
    velocity: [f32; 3],
}

impl MovementController {
    fn update(&mut self, direction: [f32; 3], elapsed: f32) -> [f32; 3] {
        let elapsed = elapsed.clamp(0.0, MAX_MOVEMENT_DELTA_SECONDS);
        let direction = normalized_or_zero(direction);
        let target_velocity = scaled(direction, MAX_MOVEMENT_SPEED);
        let acceleration = if direction == [0.0; 3] {
            MOVEMENT_DECELERATION
        } else {
            MOVEMENT_ACCELERATION
        };

        self.velocity = move_towards(self.velocity, target_velocity, acceleration * elapsed);
        scaled(self.velocity, elapsed)
    }
}

fn add_scaled(destination: &mut [f32; 3], vector: [f32; 3], scale: f32) {
    for axis in 0..3 {
        destination[axis] += vector[axis] * scale;
    }
}

fn scaled(vector: [f32; 3], scale: f32) -> [f32; 3] {
    [vector[0] * scale, vector[1] * scale, vector[2] * scale]
}

fn normalized_or_zero(vector: [f32; 3]) -> [f32; 3] {
    let length = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if length == 0.0 {
        [0.0; 3]
    } else {
        scaled(vector, 1.0 / length)
    }
}

fn move_towards(current: [f32; 3], target: [f32; 3], maximum_delta: f32) -> [f32; 3] {
    let difference = [
        target[0] - current[0],
        target[1] - current[1],
        target[2] - current[2],
    ];
    let distance = difference
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();

    if distance <= maximum_delta || distance == 0.0 {
        target
    } else {
        let step = maximum_delta / distance;
        [
            current[0] + difference[0] * step,
            current[1] + difference[1] * step,
            current[2] + difference[2] * step,
        ]
    }
}

fn sample_seed(sample_index: u64) -> u64 {
    let mut value = sample_index.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

struct Accumulation {
    linear: Vec<[f32; 3]>,
    display: Vec<u32>,
    samples: u64,
}

impl Accumulation {
    fn new(pixel_count: usize) -> Self {
        Self {
            linear: vec![[0.0; 3]; pixel_count],
            display: vec![0; pixel_count],
            samples: 0,
        }
    }

    fn reset(&mut self) {
        self.linear.fill([0.0; 3]);
        self.display.fill(0);
        self.samples = 0;
    }

    fn add_sample(&mut self, sample: &[Rgb]) {
        assert_eq!(sample.len(), self.linear.len());
        self.samples += 1;
        let weight = 1.0 / self.samples as f32;

        for ((average, display), pixel) in self.linear.iter_mut().zip(&mut self.display).zip(sample)
        {
            average[0] += (pixel.r - average[0]) * weight;
            average[1] += (pixel.g - average[1]) * weight;
            average[2] += (pixel.b - average[2]) * weight;
            *display = pack_rgb(*average);
        }
    }

    fn display_pixels(&self) -> &[u32] {
        &self.display
    }

    fn samples(&self) -> u64 {
        self.samples
    }
}

fn pack_rgb(rgb: [f32; 3]) -> u32 {
    fn channel(value: f32) -> u32 {
        let linear = if value.is_nan() {
            0.0
        } else {
            value.clamp(0.0, 1.0)
        };
        let srgb = if linear <= 0.003_130_8 {
            12.92 * linear
        } else {
            1.055 * linear.powf(1.0 / 2.4) - 0.055
        };
        (srgb * 255.0).round() as u32
    }

    (channel(rgb[0]) << 16) | (channel(rgb[1]) << 8) | channel(rgb[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulation_averages_and_resets() {
        let mut accumulation = Accumulation::new(1);
        accumulation.add_sample(&[Rgb::new(1.0, 0.0, 0.5)]);
        accumulation.add_sample(&[Rgb::new(0.0, 1.0, 0.5)]);

        assert_eq!(accumulation.samples(), 2);
        assert_eq!(accumulation.display_pixels(), &[0x00bc_bcbc]);

        accumulation.reset();
        assert_eq!(accumulation.samples(), 0);
        assert_eq!(accumulation.display_pixels(), &[0]);
    }

    #[test]
    fn rgb_packing_clamps_out_of_range_values() {
        assert_eq!(pack_rgb([-1.0, 0.5, 2.0]), 0x0000_bcff);
        assert_eq!(pack_rgb([0.003_130_8, 0.0, 0.0]), 0x000a_0000);
        assert_eq!(pack_rgb([f32::NAN, 0.0, 0.0]), 0);
    }

    #[test]
    fn sample_seeds_are_nonzero_and_distinct() {
        assert_ne!(sample_seed(0), 0);
        assert_ne!(sample_seed(0), sample_seed(1));
    }

    #[test]
    fn movement_accelerates_to_maximum_speed_and_brakes_to_a_stop() {
        let mut movement = MovementController::default();

        let first = movement.update([0.0, 0.0, -1.0], 0.1);
        assert_vector_near(first, [0.0, 0.0, -0.08]);

        movement.update([0.0, 0.0, -1.0], 0.1);
        let at_maximum = movement.update([0.0, 0.0, -1.0], 0.1);
        assert_vector_near(at_maximum, [0.0, 0.0, -0.2]);

        let braking = movement.update([0.0; 3], 0.1);
        assert_vector_near(braking, [0.0, 0.0, -0.1]);
        assert_eq!(movement.update([0.0; 3], 0.1), [0.0; 3]);
    }

    #[test]
    fn diagonal_movement_is_normalized() {
        let mut movement = MovementController::default();
        movement.update([1.0, 0.0, -1.0], 0.1);
        movement.update([1.0, 0.0, -1.0], 0.1);
        let displacement = movement.update([1.0, 0.0, -1.0], 0.1);

        assert_near(vector_length(displacement), 0.2);
        assert_near(displacement[0], -displacement[2]);
    }

    #[test]
    fn reversing_direction_transitions_through_zero_velocity() {
        let mut movement = MovementController::default();
        for _ in 0..3 {
            movement.update([1.0, 0.0, 0.0], 0.1);
        }

        let first_reverse_frame = movement.update([-1.0, 0.0, 0.0], 0.1);
        assert!(first_reverse_frame[0] > 0.0);
        for _ in 0..4 {
            movement.update([-1.0, 0.0, 0.0], 0.1);
        }

        assert_vector_near(movement.velocity, [-2.0, 0.0, 0.0]);
    }

    #[test]
    fn movement_frame_delta_is_capped() {
        let mut normal = MovementController::default();
        let mut stalled = MovementController::default();

        assert_vector_near(
            normal.update([1.0, 0.0, 0.0], 0.1),
            stalled.update([1.0, 0.0, 0.0], 10.0),
        );
    }

    fn assert_vector_near(actual: [f32; 3], expected: [f32; 3]) {
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert_near(actual, expected);
        }
    }

    fn assert_near(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 1.0e-6, "{actual} != {expected}");
    }

    fn vector_length(vector: [f32; 3]) -> f32 {
        vector.iter().map(|value| value * value).sum::<f32>().sqrt()
    }
}
