# OpenCL Ray Tracer

An interactive OpenCL path tracer written in Rust. It renders the original
scene continuously, averages successive samples to reduce noise, and resets
the accumulated image whenever the camera moves.

<img width="828" height="666" alt="image" src="https://github.com/user-attachments/assets/3e98987b-c506-4d5a-a757-01f1a1634743" />

## Requirements

- A current stable Rust toolchain
- A Linux X11 or Wayland desktop
- An OpenCL ICD loader and a vendor OpenCL driver for at least one device
- The native window-development packages required by `minifb`

On Ubuntu or Debian, the window dependencies can be installed with:

```sh
sudo apt install libx11-dev libxkbcommon-dev libwayland-dev libwayland-cursor0
```

Install the OpenCL package supplied by the GPU or CPU vendor. For example,
NVIDIA's display driver normally provides its OpenCL implementation. `clinfo`
is useful for checking that the ICD can see a device before launching the ray
tracer.

## Build and run

```sh
cargo build --release --locked
cargo run --release --locked
```

The program prefers the first GPU exposed by any OpenCL platform and falls
back to the first other OpenCL device if no GPU is available. The selected
platform and device are printed at startup.

Controls:

- `W` / `S`: move forward / backward
- `A` / `D`: move left / right
- `Escape`: exit

## Tests

Run the device-independent tests with:

```sh
cargo test --locked
```

An additional smoke test compiles and executes the kernel on a real OpenCL
device:

```sh
cargo test --locked opencl_render_smoke_test -- --ignored --show-output
```

## Troubleshooting

- `no OpenCL platforms found`: install an ICD loader and a vendor driver, then
  verify the installation with `clinfo`.
- `OpenCL kernel build failed`: the complete device compiler log is included in
  the error. Confirm that the selected device supports OpenCL C 1.2 or newer.
- Window creation errors: make sure `DISPLAY` or `WAYLAND_DISPLAY` points to the
  active desktop session and that the native window libraries above are
  installed.

The OpenCL source is embedded in the executable at compile time, so the binary
does not need to be launched from the repository directory.
