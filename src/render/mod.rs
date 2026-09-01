mod camera;
mod progress;
mod timing;

#[cfg(test)]
mod golden;

pub use camera::GpuCamera;
pub use timing::Timings;

use crate::config::Config;
use crate::scene::GpuLight;
use crate::scene::Scene;
use bytemuck::Zeroable;
use bytemuck::cast_slice;
use half::f16;
use progress::Progress;
use std::collections::VecDeque;
use std::error::Error;
use std::path::Path;
use std::sync::mpsc;
use timing::Timer;
use wgpu::util::BufferInitDescriptor;
use wgpu::util::DeviceExt;

/// Samples allowed in flight before the loop waits on the oldest one. Enough to
/// keep the GPU fed while it works through the queue, few enough that the
/// progress readout reflects samples the GPU has actually finished.
const IN_FLIGHT_SAMPLES: usize = 4;

/// A finished render: the frame, the GPU that drew it, and what that cost.
pub struct Render {
    pub image: Image,
    /// The adapter's name and backend — "which GPU did this" is the first thing
    /// worth knowing when a render is slow or wrong.
    pub renderer: String,
    /// `None` on an adapter that cannot write timestamps.
    pub timings: Option<Timings>,
}

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
/// compatible with, so any device that can run a compute pass will do.
pub fn render(config: &Config, scene: &Scene) -> Result<Render, Box<dyn Error>> {
    pollster::block_on(run(config, scene))
}

async fn run(config: &Config, scene: &Scene) -> Result<Render, Box<dyn Error>> {
    // An empty buffer cannot be bound, and a scene with nothing in it is a
    // mistake worth naming rather than a black frame worth writing.
    if scene.triangles.is_empty() {
        return Err("scene has no geometry to render".into());
    }

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await?;
    // Timestamps are optional on every backend — Metal wants stage-boundary
    // counter sampling, Vulkan a queue with enough valid timestamp bits — and
    // asking for a feature the adapter lacks fails the request outright, so the
    // ask is whatever it turns out to have.
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("device"),
            required_features: adapter.features() & wgpu::Features::TIMESTAMP_QUERY,
            ..Default::default()
        })
        .await?;

    let info = adapter.get_info();
    let renderer = format!("{} ({:?})", info.name, info.backend);

    let mut camera = GpuCamera::from(&config.camera);
    camera.light_count = scene.lights.len() as u32;
    camera.light_power = scene.light_power;
    camera.environment_rotation = config.environment.rotation.to_radians();
    camera.environment_intensity = config.environment.intensity;
    if let Some(sky) = &scene.sky {
        camera.sky_width = sky.width;
        camera.sky_height = sky.height;
    }
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
    let bvh_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bvh"),
        contents: cast_slice(&scene.nodes),
        usage: wgpu::BufferUsages::STORAGE,
    });
    // A scene may legitimately have no emitters — every example but melee is lit
    // by its background alone — but an empty buffer cannot be bound, so the
    // binding falls back to this one entry. Nothing reads it: `light_count` is
    // zero alongside it, and that is what the shader checks.
    let unused = [GpuLight::zeroed()];
    let light_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("lights"),
        contents: match scene.lights.is_empty() {
            true => cast_slice(&unused),
            false => cast_slice(&scene.lights),
        },
        usage: wgpu::BufferUsages::STORAGE,
    });

    let (environment_texture, environment_sampler) = environment(&device, &queue, config, scene);

    // The sky's sampling distribution, and the same fallback the light table
    // takes: a scene with no map to aim at still has to bind something, so it
    // binds one entry nothing reads. `sky_width` is zero alongside it, and that
    // is what the shader checks.
    let unaimed = [0.0f32];
    let sky_marginal = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("sky marginal"),
        contents: match &scene.sky {
            Some(sky) => cast_slice(&sky.marginal),
            None => cast_slice(&unaimed),
        },
        usage: wgpu::BufferUsages::STORAGE,
    });
    let sky_conditional = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("sky conditional"),
        contents: match &scene.sky {
            Some(sky) => cast_slice(&sky.conditional),
            None => cast_slice(&unaimed),
        },
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
            wgpu::BindGroupEntry {
                binding: 2,
                resource: bvh_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: light_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&environment_texture),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(&environment_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: sky_marginal.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: sky_conditional.as_entire_binding(),
            },
        ],
    });

    // One dispatch per sample, each reseeding the shader's RNG from the sample
    // index it is handed.
    let sample_offset = std::mem::offset_of!(GpuCamera, sample) as wgpu::BufferAddress;
    let mut in_flight: VecDeque<wgpu::SubmissionIndex> = VecDeque::new();
    let mut progress = Progress::new(config.camera.samples);
    let timer = Timer::new(&device, &queue, config.camera.samples);

    for sample in 1..=config.camera.samples {
        // A queued write lands before the submission that follows it.
        queue.write_buffer(&camera_buffer, sample_offset, bytemuck::bytes_of(&sample));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sample"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("trace"),
                timestamp_writes: timer.as_ref().and_then(|timer| timer.writes(sample)),
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &frame, &[]);
            pass.set_bind_group(1, &geometry, &[]);
            // Rounded up to whole workgroups; the shader drops the threads that
            // land outside the image.
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        in_flight.push_back(queue.submit([encoder.finish()]));

        // Let a few samples queue up, then block on the oldest. Waiting keeps
        // the count honest — without it every sample would "finish" instantly
        // and the GPU would still be tracing long after the loop ended.
        if in_flight.len() > IN_FLIGHT_SAMPLES {
            wait_for(&device, &mut in_flight, &mut progress)?;
        }
    }
    while !in_flight.is_empty() {
        wait_for(&device, &mut in_flight, &mut progress)?;
    }
    progress.finish();

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback"),
    });
    encoder.copy_buffer_to_buffer(&accumulator, 0, &readback, 0, accumulator_size);
    if let Some(timer) = &timer {
        timer.resolve(&mut encoder);
    }
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

    let timings = match &timer {
        Some(timer) => timer.timings(&device)?,
        None => None,
    };

    Ok(Render {
        image: Image {
            width,
            height,
            pixels,
        },
        renderer,
        timings,
    })
}

/// Uploads the sky as a texture, with the sampler that reads it.
///
/// A scene that named no HDRI gets a **one texel** map holding its flat color.
/// Sampling a one-texel texture anywhere returns exactly that texel, so the flat
/// background is not a second path through the shader — it is the same lookup
/// over a smaller image. This is the trick the light buffer already plays one
/// level down, where an empty table becomes one zeroed entry because an empty
/// buffer cannot be bound.
///
/// `Rgba16Float` rather than `Rgba32Float`: sixteen-bit float is filterable
/// everywhere, while a filtering sampler over a thirty-two-bit one needs
/// `FLOAT32_FILTERABLE`, which the device request above does not ask for and not
/// every adapter offers. Half is also what an EXR very likely stores already, so
/// the narrowing usually costs nothing. `Environment::read` has clamped the
/// texels into the range, so nothing here can overflow to an infinity.
fn environment(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: &Config,
    scene: &Scene,
) -> (wgpu::TextureView, wgpu::Sampler) {
    let flat = [[
        config.environment.color[0],
        config.environment.color[1],
        config.environment.color[2],
        1.0,
    ]];
    let (width, height, texels) = match &scene.environment {
        Some(map) => (map.width, map.height, map.texels.as_slice()),
        None => (1, 1, flat.as_slice()),
    };

    let halves: Vec<f16> = texels
        .iter()
        .flatten()
        .map(|&channel| f16::from_f32(channel))
        .collect();

    let texture = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("environment"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        cast_slice(&halves),
    );

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("environment"),
        // Longitude wraps and latitude does not: the left and right edges of an
        // equirectangular map are the same meridian, so repeating there hides
        // the seam, while repeating at the poles would fold the sky back on
        // itself.
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    (texture.create_view(&Default::default()), sampler)
}

/// Blocks until the oldest queued sample has run, then counts it.
fn wait_for(
    device: &wgpu::Device,
    in_flight: &mut VecDeque<wgpu::SubmissionIndex>,
    progress: &mut Progress,
) -> Result<(), wgpu::PollError> {
    let Some(submission) = in_flight.pop_front() else {
        return Ok(());
    };

    device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission),
        timeout: None,
    })?;
    progress.advance();

    Ok(())
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
    use crate::scene::GpuBvhNode;
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
        assert_eq!(wgsl("Light"), size_of::<GpuLight>());
        assert_eq!(wgsl("Material"), size_of::<GpuMaterial>());
        assert_eq!(wgsl("Triangle"), size_of::<GpuTriangle>());
        assert_eq!(wgsl("BvhNode"), size_of::<GpuBvhNode>());
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

    /// The shader rejects a non-finite sample before it reaches the accumulator,
    /// so this should never fire in practice. It is asserted anyway because it
    /// is the other half of that guarantee: `f32::max` returns its non-NaN
    /// operand, which is the only reason a poisoned sum resolves to black rather
    /// than to whatever `as u8` makes of a NaN.
    #[test]
    fn a_poisoned_sum_still_resolves_to_a_pixel() {
        let nan = f32::NAN;
        assert_eq!(resolve(&[[nan, nan, nan, 1.0]]), [0, 0, 0, 255]);
        assert_eq!(resolve(&[[f32::INFINITY, 0.0, 0.0, 1.0]]), [255, 0, 0, 255]);
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
