//! The GPU pass: scene buffers in, pixels out.
//!
//! The tracing itself is not written yet — `shader.wgsl` fills the frame with a
//! background gradient and a readout of what it was handed. Everything on this
//! side is the real thing, so landing the tracer is a change to the shader and
//! to the sample loop, not to the plumbing underneath it.

mod camera;

pub use camera::GpuCamera;

use crate::config::Config;
use crate::scene::Scene;
use bytemuck::cast_slice;
use std::error::Error;
use std::path::Path;
use std::sync::mpsc;
use wgpu::util::BufferInitDescriptor;
use wgpu::util::DeviceExt;

/// A finished frame, 8-bit RGBA and ready to write.
pub struct Image {
    pub width: u32,
    pub height: u32,
    pixels: Vec<u8>,
}

impl Image {
    pub fn save(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        image::save_buffer(
            path,
            &self.pixels,
            self.width,
            self.height,
            image::ColorType::Rgba8,
        )?;
        Ok(())
    }
}

/// Renders `scene` through `config`'s camera, blocking until the GPU is done.
///
/// The adapter is whatever the platform offers — there is no surface to be
/// compatible with, so any device that can run a compute pass will do. Its name
/// is returned along with the image because "which GPU did this" is the first
/// thing worth knowing when a render is slow or wrong.
pub fn render(config: &Config, scene: &Scene) -> Result<(Image, String), Box<dyn Error>> {
    pollster::block_on(run(config, scene))
}

async fn run(config: &Config, scene: &Scene) -> Result<(Image, String), Box<dyn Error>> {
    // An empty buffer cannot be bound, and a scene with nothing in it is a
    // mistake worth naming rather than a black frame worth writing.
    if scene.triangles.is_empty() {
        return Err("scene has no geometry to render".into());
    }

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("device"),
            ..Default::default()
        })
        .await?;

    let info = adapter.get_info();
    let renderer = format!("{} ({:?})", info.name, info.backend);

    let camera = GpuCamera::from(&config.camera);
    let (width, height) = (camera.width, camera.height);

    let camera_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("camera"),
        contents: bytemuck::bytes_of(&camera),
        // COPY_DST so the sample index can be rewritten between passes.
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let material_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("materials"),
        contents: cast_slice(&scene.materials),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let triangle_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("triangles"),
        contents: cast_slice(&scene.triangles),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // One `vec4<f32>` of summed radiance per pixel, started at zero.
    let accumulator_size = (width as u64) * (height as u64) * size_of::<[f32; 4]>() as u64;
    let accumulator = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("accumulator"),
        size: accumulator_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    // A storage buffer cannot be mapped, so the sums come back through a second
    // buffer that exists only to be read.
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: accumulator_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("path tracer"),
        // Reflected from the shader: the two groups it declares are the two
        // built below, so a binding that changes there fails here and not in
        // some later frame.
        layout: None,
        module: &shader,
        entry_point: Some("trace"),
        compilation_options: Default::default(),
        cache: None,
    });

    let frame = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("frame"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: accumulator.as_entire_binding(),
            },
        ],
    });
    let geometry = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scene"),
        layout: &pipeline.get_bind_group_layout(1),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: material_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: triangle_buffer.as_entire_binding(),
            },
        ],
    });

    // One pass, whatever the config asked for: with no RNG in the shader yet,
    // every sample of a pixel would take the identical path and average to
    // exactly what one pass already produced. Once that stops being true this
    // becomes a loop that rewrites `sample` and dispatches again — the buffers
    // and bind groups above are already shaped for it.
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("sample"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("trace"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &frame, &[]);
        pass.set_bind_group(1, &geometry, &[]);
        // Rounded up to whole workgroups; the shader drops the threads that
        // land outside the image.
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }
    encoder.copy_buffer_to_buffer(&accumulator, 0, &readback, 0, accumulator_size);
    queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    // The map only resolves once the queued work ahead of it has run, so this
    // is also where the render is waited on.
    device.poll(wgpu::PollType::wait_indefinitely())?;
    receiver.recv()??;

    let mapped = slice.get_mapped_range()?;
    let pixels = resolve(cast_slice(&mapped));
    drop(mapped);
    readback.unmap();

    Ok((
        Image {
            width,
            height,
            pixels,
        },
        renderer,
    ))
}

/// Averages each pixel's accumulated radiance and gamma-encodes it to 8-bit
/// sRGB.
fn resolve(sums: &[[f32; 4]]) -> Vec<u8> {
    sums.iter()
        .flat_map(|sum| {
            let scale = match sum[3] > 0.0 {
                true => 1.0 / sum[3],
                false => 0.0,
            };
            let encode = |channel: f32| {
                let corrected = (channel * scale).max(0.0).powf(1.0 / 2.2);
                (corrected.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
            };

            [encode(sum[0]), encode(sum[1]), encode(sum[2]), 255]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::GpuMaterial;
    use crate::scene::GpuTriangle;
    use naga::proc::Layouter;
    use naga::valid::Capabilities;
    use naga::valid::ValidationFlags;
    use naga::valid::Validator;

    /// Compiles the shader the way wgpu will and reports what each of its
    /// structs measures, so the tests below can hold the two sides of the
    /// binding against each other without a GPU in the room.
    fn shader_struct_sizes() -> Vec<(String, u32)> {
        let source = include_str!("shader.wgsl");
        let module = naga::front::wgsl::parse_str(source).expect("the shader should parse");
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .expect("the shader should validate");

        let mut layouter = Layouter::default();
        layouter.update(module.to_ctx()).unwrap();

        module
            .types
            .iter()
            .filter_map(|(handle, ty)| Some((ty.name.clone()?, layouter[handle].size)))
            .collect()
    }

    #[test]
    fn the_shader_agrees_with_the_structs_it_is_handed() {
        let sizes = shader_struct_sizes();
        let wgsl = |name: &str| {
            sizes
                .iter()
                .find(|(found, _)| found == name)
                .unwrap_or_else(|| panic!("the shader should declare {name}, found {sizes:?}"))
                .1 as usize
        };

        assert_eq!(wgsl("Camera"), size_of::<GpuCamera>());
        assert_eq!(wgsl("Material"), size_of::<GpuMaterial>());
        assert_eq!(wgsl("Triangle"), size_of::<GpuTriangle>());
    }

    #[test]
    fn averages_by_the_sample_count_it_was_given() {
        // Four passes of mid-grey average back to mid-grey, not to four times it.
        let one = resolve(&[[0.25, 0.5, 0.75, 1.0]]);
        let four = resolve(&[[1.0, 2.0, 3.0, 4.0]]);

        assert_eq!(one, four);
        assert_eq!(one[3], 255, "the frame is opaque");
    }

    #[test]
    fn an_untouched_pixel_is_black_and_not_a_nan() {
        assert_eq!(resolve(&[[0.0; 4]]), [0, 0, 0, 255]);
    }

    #[test]
    fn gamma_encodes_and_clamps() {
        let pixels = resolve(&[[0.0, 0.5, 1.0, 1.0], [-1.0, 2.0, 1.0, 1.0]]);

        assert_eq!(pixels[0], 0);
        assert_eq!(pixels[2], 255);
        assert!(pixels[1] > 128, "0.5 should brighten to {}", pixels[1]);
        assert_eq!(&pixels[4..7], &[0, 255, 255], "out of range should clamp");
    }
}
