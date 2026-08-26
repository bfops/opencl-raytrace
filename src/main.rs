mod renderer;
mod scene;

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use minifb::{Key, KeyRepeat, Window, WindowOptions};

use renderer::{OpenClRenderer, Rgb};
use scene::Scene;

const WIDTH: usize = 800;
const HEIGHT: usize = 600;
const CAMERA_STEP: f32 = 1.0;

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

    let mut last_title_update = Instant::now();
    while window.is_open() && !window.is_key_down(Key::Escape) {
        if move_camera_from_input(&window, &mut scene) {
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

fn move_camera_from_input(window: &Window, scene: &mut Scene) -> bool {
    let mut delta = [0.0; 3];

    if window.is_key_pressed(Key::W, KeyRepeat::Yes) {
        add_scaled(&mut delta, scene.forward(), CAMERA_STEP);
    }
    if window.is_key_pressed(Key::S, KeyRepeat::Yes) {
        add_scaled(&mut delta, scene.forward(), -CAMERA_STEP);
    }
    if window.is_key_pressed(Key::D, KeyRepeat::Yes) {
        add_scaled(&mut delta, scene.right(), CAMERA_STEP);
    }
    if window.is_key_pressed(Key::A, KeyRepeat::Yes) {
        add_scaled(&mut delta, scene.right(), -CAMERA_STEP);
    }

    if delta == [0.0; 3] {
        false
    } else {
        scene.move_camera(delta);
        true
    }
}

fn add_scaled(destination: &mut [f32; 3], vector: [f32; 3], scale: f32) {
    for axis in 0..3 {
        destination[axis] += vector[axis] * scale;
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
}
