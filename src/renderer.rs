use std::mem::{align_of, size_of};
use std::ptr;

use anyhow::{Context as _, Result, anyhow, bail};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_GPU, Device};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
use opencl3::platform::get_platforms;
use opencl3::program::Program;
use opencl3::types::{CL_BLOCKING, cl_event};

use crate::scene::{Camera, Scene};

const KERNEL_NAME: &str = "render";
const SKIP_MWC_SOURCE: &str = include_str!("../cl/mwc64x/cl/mwc64x/skip_mwc.cl");
const MWC64X_SOURCE: &str = include_str!("../cl/mwc64x/cl/mwc64x/mwc64x_rng.cl");
const NOISE_SOURCE: &str = include_str!("../cl/Noise/Noise/Noise.cl");
const RAYTRACE_SOURCE: &str = include_str!("../cl/main.cl");

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ClFloat3 {
    data: [f32; 4],
}

impl ClFloat3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            data: [x, y, z, 0.0],
        }
    }

    pub const fn xyz(self) -> [f32; 3] {
        [self.data[0], self.data[1], self.data[2]]
    }

    pub fn add(&mut self, delta: [f32; 3]) {
        for (value, change) in self.data[..3].iter_mut().zip(delta) {
            *value += change;
        }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct Texture {
    data: [f32; 4],
    tag: u8,
    padding: [u8; 15],
}

impl Texture {
    pub const fn solid_color(r: f32, g: f32, b: f32) -> Self {
        Self {
            data: [r, g, b, 0.0],
            tag: 0,
            padding: [0; 15],
        }
    }

    pub const fn wood() -> Self {
        Self {
            data: [0.0; 4],
            tag: 3,
            padding: [0; 15],
        }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct Object {
    center: ClFloat3,
    radius: f32,
    diffuseness: f32,
    emittance: f32,
    reflectance: f32,
    transmittance: f32,
    padding: [u8; 12],
    texture: Texture,
}

impl Object {
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        center: [f32; 3],
        radius: f32,
        diffuseness: f32,
        emittance: f32,
        reflectance: f32,
        transmittance: f32,
        texture: Texture,
    ) -> Self {
        Self {
            center: ClFloat3::new(center[0], center[1], center[2]),
            radius,
            diffuseness,
            emittance,
            reflectance,
            transmittance,
            padding: [0; 12],
            texture,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rgb {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Rgb {
    #[cfg(test)]
    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }
}

pub struct OpenClRenderer {
    _context: Context,
    queue: CommandQueue,
    kernel: Kernel,
    object_buffer: Buffer<Object>,
    output_buffer: Buffer<Rgb>,
    output: Vec<Rgb>,
    width: u32,
    height: u32,
    object_count: u32,
    device_name: String,
}

impl OpenClRenderer {
    pub fn new(width: usize, height: usize, scene: &Scene) -> Result<Self> {
        verify_abi()?;
        let (device, platform_name) = select_device()?;
        let device_name = device
            .name()
            .context("could not query OpenCL device name")?;
        let context = Context::from_device(&device).context("could not create OpenCL context")?;
        let queue = CommandQueue::create_default(&context, 0)
            .context("could not create OpenCL command queue")?;

        let source = compose_kernel_source();
        let program = Program::create_and_build_from_source(&context, &source, "")
            .map_err(|log| anyhow!("OpenCL kernel build failed:\n{log}"))?;
        let kernel = Kernel::create(&program, KERNEL_NAME)
            .context("could not create the OpenCL render kernel")?;

        let objects = scene.objects();
        if objects.is_empty() {
            bail!("the scene must contain at least one object");
        }

        // SAFETY: Object has a verified C layout, and the buffer is sized for this slice.
        let mut object_buffer = unsafe {
            Buffer::<Object>::create(&context, CL_MEM_READ_ONLY, objects.len(), ptr::null_mut())
        }
        .context("could not allocate the OpenCL scene buffer")?;
        // SAFETY: object_buffer and objects have identical element counts and layouts.
        unsafe { queue.enqueue_write_buffer(&mut object_buffer, CL_BLOCKING, 0, objects, &[]) }
            .context("could not upload the scene to OpenCL")?;

        let pixel_count = width
            .checked_mul(height)
            .context("render dimensions overflowed")?;
        // SAFETY: Rgb has a verified C layout and output remains alive with this context.
        let output_buffer = unsafe {
            Buffer::<Rgb>::create(&context, CL_MEM_WRITE_ONLY, pixel_count, ptr::null_mut())
        }
        .context("could not allocate the OpenCL output buffer")?;

        println!("OpenCL platform: {platform_name}");

        Ok(Self {
            _context: context,
            queue,
            kernel,
            object_buffer,
            output_buffer,
            output: vec![Rgb::default(); pixel_count],
            width: width
                .try_into()
                .context("render width exceeds OpenCL uint")?,
            height: height
                .try_into()
                .context("render height exceeds OpenCL uint")?,
            object_count: objects
                .len()
                .try_into()
                .context("scene object count exceeds OpenCL uint")?,
            device_name,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn render(&mut self, camera: &Camera, random_seed: u64) -> Result<&[Rgb]> {
        let global_work_size = self.output.len();

        // SAFETY: Every argument matches the render kernel's order, size, and ABI. Buffers
        // and scalar arguments remain alive until the enqueued command completes.
        let kernel_event = unsafe {
            ExecuteKernel::new(&self.kernel)
                .set_arg(&self.width)
                .set_arg(&self.height)
                .set_arg(&camera.fovy)
                .set_arg(&camera.eye)
                .set_arg(&camera.look)
                .set_arg(&camera.up)
                .set_arg(&random_seed)
                .set_arg(&self.object_buffer)
                .set_arg(&self.object_count)
                .set_arg(&self.output_buffer)
                .set_global_work_size(global_work_size)
                .enqueue_nd_range(&self.queue)
        }
        .context("could not enqueue the OpenCL render kernel")?;

        let wait_events: [cl_event; 1] = [kernel_event.get()];
        // SAFETY: output is exactly the size and layout of output_buffer. The blocking read
        // waits for kernel_event, so both the event and destination remain valid throughout.
        unsafe {
            self.queue.enqueue_read_buffer(
                &self.output_buffer,
                CL_BLOCKING,
                0,
                &mut self.output,
                &wait_events,
            )
        }
        .context("could not read the OpenCL render output")?;

        Ok(&self.output)
    }
}

fn select_device() -> Result<(Device, String)> {
    let platforms = get_platforms().map_err(|error| {
        anyhow!(
            "could not enumerate OpenCL platforms; install an OpenCL ICD and vendor driver ({error})"
        )
    })?;
    if platforms.is_empty() {
        bail!("no OpenCL platforms found; install an OpenCL ICD and vendor driver");
    }

    for device_type in [CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ALL] {
        for platform in &platforms {
            let Ok(device_ids) = platform.get_devices(device_type) else {
                continue;
            };
            if let Some(&device_id) = device_ids.first() {
                let platform_name = platform
                    .name()
                    .unwrap_or_else(|_| "Unknown OpenCL platform".to_owned());
                return Ok((Device::new(device_id), platform_name));
            }
        }
    }

    bail!("OpenCL platforms were found, but none exposed a usable device")
}

fn compose_kernel_source() -> String {
    let sources = [
        SKIP_MWC_SOURCE,
        MWC64X_SOURCE,
        NOISE_SOURCE,
        RAYTRACE_SOURCE,
    ];
    let capacity = sources.iter().map(|source| source.len()).sum();
    let mut combined = String::with_capacity(capacity);

    for source in sources {
        for line in source.lines() {
            if !line.trim_start().starts_with("#include") {
                combined.push_str(line);
                combined.push('\n');
            }
        }
    }

    combined
}

fn verify_abi() -> Result<()> {
    let layouts = [
        (
            "ClFloat3",
            size_of::<ClFloat3>(),
            align_of::<ClFloat3>(),
            16,
            16,
        ),
        (
            "Texture",
            size_of::<Texture>(),
            align_of::<Texture>(),
            32,
            16,
        ),
        ("Object", size_of::<Object>(), align_of::<Object>(), 80, 16),
        ("Rgb", size_of::<Rgb>(), align_of::<Rgb>(), 12, 4),
    ];

    for (name, size, alignment, expected_size, expected_alignment) in layouts {
        if size != expected_size || alignment != expected_alignment {
            bail!(
                "{name} has incompatible OpenCL layout: size {size}, alignment {alignment}; expected size {expected_size}, alignment {expected_alignment}"
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opencl_abi_layouts_are_exact() {
        verify_abi().unwrap();
    }

    #[test]
    fn embedded_kernel_is_self_contained() {
        let source = compose_kernel_source();
        assert!(!source.contains("#include"));
        assert!(source.contains("__kernel void render"));
        assert!(source.contains("MWC64X_NextUint"));
        assert!(source.contains("Noise_3d"));
    }

    #[test]
    #[ignore = "requires an OpenCL device"]
    fn opencl_render_smoke_test() {
        let scene = Scene::default();
        let mut renderer = OpenClRenderer::new(16, 16, &scene).unwrap();
        let pixels = renderer.render(scene.camera(), 1).unwrap();

        assert_eq!(pixels.len(), 16 * 16);
        assert!(
            pixels
                .iter()
                .all(|pixel| { pixel.r.is_finite() && pixel.g.is_finite() && pixel.b.is_finite() })
        );
        assert!(
            pixels
                .iter()
                .any(|pixel| pixel.r > 0.0 || pixel.g > 0.0 || pixel.b > 0.0)
        );
    }
}
