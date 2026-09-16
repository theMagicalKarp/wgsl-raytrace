mod aov;
mod camera;
mod denoise;
mod kitty;
mod preview;
mod progress;
mod timing;
mod tonemap;
mod variance;

#[cfg(test)]
mod golden;
#[cfg(test)]
mod shader_tests;

pub use aov::Aovs;
pub use camera::GpuCamera;
pub use denoise::Denoised;
pub use denoise::GpuDenoise;
pub use preview::preview;
pub use timing::Timings;
pub use variance::Statistics;

use crate::config::Config;
use crate::scene::GpuLight;
use crate::scene::Scene;
use bytemuck::Pod;
use bytemuck::Zeroable;
use bytemuck::cast_slice;
use half::f16;
use progress::Progress;
use std::collections::VecDeque;
use std::error::Error;
use std::path::Path;
use std::sync::mpsc;
use std::time::Instant;
use timing::Timer;
use wgpu::util::BufferInitDescriptor;
use wgpu::util::DeviceExt;

/// Samples allowed in flight before the loop waits on the oldest one. Enough to
/// keep the GPU fed while it works through the queue, few enough that the
/// progress readout reflects samples the GPU has actually finished.
const IN_FLIGHT_SAMPLES: usize = 4;

/// Storage buffers the shader binds across its two groups: the accumulator, the
/// moments and the two feature buffers in the frame's, and the materials,
/// triangles, BVH, light table and the sky's two distributions in the scene's.
///
/// wgpu defaults this limit to eight, which is a floor every backend can meet
/// including WebGL. This renderer is native, offline and will never see one of
/// those, so it asks the adapter for what it actually needs — and a test below
/// counts the shader's own bindings against this number, so adding one and
/// forgetting to raise it fails without a GPU in the room.
const STORAGE_BUFFERS: u32 = 10;

/// Materials the object-id vote can hold: sixteen bits, less the zero that
/// stands for a miss.
const OBJECT_IDS: usize = 0xffff;

/// What a run wants out of a render beyond the frame itself.
///
/// Each costs something a run that only wants a picture should not pay — two
/// full-frame copies over the bus for the features, a shader compile and a
/// handful of dispatches for the filter, a frame's worth of allocation for the
/// heatmap — so each is asked for rather than assumed.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Outputs {
    /// Resolve the feature buffers into images.
    pub aovs: bool,
    /// Run the a-trous filter over the finished frame.
    pub denoise: bool,
    /// Keep the per-pixel variance field as a heatmap. The summary beside it is
    /// built either way; this is only about the picture of it.
    pub variance: bool,
}

/// A finished render: the frame, the GPU that drew it, and what that cost.
pub struct Render {
    pub image: Image,
    /// The adapter's name and backend — "which GPU did this" is the first thing
    /// worth knowing when a render is slow or wrong.
    pub renderer: String,
    /// `None` on an adapter that cannot write timestamps.
    pub timings: Option<Timings>,
    /// What the frame's own samples say about how converged it is, recovered
    /// from the moments the shader accumulated beside the radiance.
    pub statistics: Statistics,
    /// The feature buffers, resolved — `None` unless the render was asked for
    /// them. Reading them back costs two full-frame copies over the bus and
    /// three images' worth of allocation, which is not worth spending on a run
    /// that is only after a picture.
    pub aovs: Option<Aovs>,
    /// The same frame with the filter run over it — `None` unless the render was
    /// asked for it. Beside [`Render::image`] rather than in place of it: tuning
    /// the filter's sigmas means comparing the two, so the unfiltered frame has
    /// to survive the filter running. Which of them a caller writes where is the
    /// caller's business — `--denoise` gives this one `--output` and leaves the
    /// other for `--debug`.
    pub denoised: Option<Denoised>,
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
/// Every sample, start to finish. [`preview`] drives the same [`Renderer`] one
/// sample at a time instead, so that it can look at the accumulator on the way
/// through.
pub fn render(config: &Config, scene: &Scene, outputs: Outputs) -> Result<Render, Box<dyn Error>> {
    let mut renderer = Renderer::new(config, scene)?;
    renderer.wants(outputs);
    while renderer.remaining() > 0 {
        renderer.sample()?;
    }
    renderer.finish()
}

/// One render's GPU state, driveable a sample at a time.
///
/// The split exists for [`preview`], which has to interleave the sample loop
/// with reading the accumulator and drawing it. Nothing here is preview-specific
/// — [`render`] runs the same three calls in a row.
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    frame: wgpu::BindGroup,
    geometry: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
    accumulator: wgpu::Buffer,
    readback: wgpu::Buffer,
    accumulator_size: u64,
    /// The luminance moments, and the buffer they come home through. Kept off
    /// [`Renderer::request`]'s path entirely — the preview never wants them, and
    /// the one readback buffer that path shares is exactly what makes its
    /// unblocked loop work.
    moments: wgpu::Buffer,
    moments_readback: wgpu::Buffer,
    moments_size: u64,
    /// The feature buffers. Bound and written every sample whether or not
    /// anything will read them — they are two more stores in a shader that is
    /// already storing — and read back only when [`Renderer::wants_aovs`] says
    /// somebody asked.
    normals: wgpu::Buffer,
    albedos: wgpu::Buffer,
    /// The filter's settings, resolved out of the scene at construction. Its
    /// `samples` is filled in at the end, from the count the render actually
    /// reached rather than the one it was budgeted.
    denoise: GpuDenoise,
    iterations: u32,
    /// Linear gain into the tone curve, off the scene's camera. Read at resolve
    /// time and nowhere else.
    exposure: f32,
    outputs: Outputs,
    timer: Option<Timer>,
    in_flight: VecDeque<wgpu::SubmissionIndex>,
    /// Set while a copy into `readback` is queued and its map has not resolved.
    /// There is one readback buffer, so there is at most one of these.
    pending: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    progress: Progress,
    width: u32,
    height: u32,
    samples: u32,
    /// Samples handed to the queue, which runs ahead of `completed`.
    submitted: u32,
    /// Samples the GPU has finished, which is what the accumulator holds.
    completed: u32,
    /// The adapter's name and backend.
    renderer: String,
    /// Set once [`Renderer::finish`] has resolved the query set, which may only
    /// happen once.
    finished: bool,
}

impl Renderer {
    /// Uploads `scene` and builds the pipeline that will trace it.
    ///
    /// The adapter is whatever the platform offers — there is no surface to be
    /// compatible with, so any device that can run a compute pass will do. That
    /// stays true of the preview, which draws through the terminal rather than
    /// through a swapchain.
    pub fn new(config: &Config, scene: &Scene) -> Result<Renderer, Box<dyn Error>> {
        pollster::block_on(Renderer::build(config, scene))
    }

    async fn build(config: &Config, scene: &Scene) -> Result<Renderer, Box<dyn Error>> {
        // An empty buffer cannot be bound, and a scene with nothing in it is a
        // mistake worth naming rather than a black frame worth writing.
        if scene.triangles.is_empty() {
            return Err("scene has no geometry to render".into());
        }

        // The shader's object-id vote packs `material + 1` into sixteen bits
        // (`shader.wgsl`'s `Albedo`), and an id past that would silently alias
        // another object's and let the filter blur across their shared edge.
        if scene.materials.len() > OBJECT_IDS {
            return Err(format!(
                "scene has {} materials and the denoiser can tell at most {OBJECT_IDS} apart",
                scene.materials.len(),
            )
            .into());
        }

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await?;
        // Timestamps are optional on every backend — Metal wants stage-boundary
        // counter sampling, Vulkan a queue with enough valid timestamp bits — and
        // asking for a feature the adapter lacks fails the request outright, so the
        // ask is whatever it turns out to have.
        let available = adapter.limits().max_storage_buffers_per_shader_stage;
        if available < STORAGE_BUFFERS {
            return Err(format!(
                "adapter binds at most {available} storage buffers per stage \
                 and the tracer needs {STORAGE_BUFFERS}",
            )
            .into());
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("device"),
                required_features: adapter.features() & wgpu::Features::TIMESTAMP_QUERY,
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: STORAGE_BUFFERS,
                    ..Default::default()
                },
                ..Default::default()
            })
            .await?;

        let info = adapter.get_info();
        let renderer = format!("{} ({:?})", info.name, info.backend);

        let mut camera = GpuCamera::from(&config.camera);
        camera.outlier_k = config.outliers.k;
        camera.outlier_warmup = config.outliers.warmup;
        camera.seed = config.seed;
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

        let (environment_texture, environment_sampler) =
            environment(&device, &queue, config, scene);

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

        // And two `vec2<f32>`s of summed luminance and summed squared luminance,
        // which is what a variance is recovered from: one over every sample as
        // drawn, for the firefly rejection, and one over what the accumulator
        // kept, for everything else (`shader.wgsl`'s `Moments`). The width of
        // the accumulator and read exactly once, at the end.
        let moments_size = (width as u64) * (height as u64) * size_of::<[f32; 4]>() as u64;
        let moments = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("moments"),
            size: moments_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let moments_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("moments readback"),
            size: moments_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // The features, at the accumulator's width: `vec4f` each, one holding
        // the normal with the depth in its fourth channel and one the albedo
        // with the object-id vote packed into its.
        let feature = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: accumulator_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let normals = feature("normals");
        let albedos = feature("albedos");

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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: moments.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: normals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: albedos.as_entire_binding(),
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

        let timer = Timer::new(&device, &queue, config.camera.samples);

        Ok(Renderer {
            device,
            queue,
            pipeline,
            frame,
            geometry,
            camera_buffer,
            accumulator,
            readback,
            accumulator_size,
            moments,
            moments_readback,
            moments_size,
            normals,
            albedos,
            denoise: GpuDenoise::new(&config.denoise, (width, height), config.camera.samples),
            iterations: config.denoise.iterations,
            exposure: config.camera.exposure,
            outputs: Outputs::default(),
            timer,
            in_flight: VecDeque::new(),
            pending: None,
            progress: Progress::new(config.camera.samples),
            width,
            height,
            samples: config.camera.samples,
            submitted: 0,
            completed: 0,
            renderer,
            finished: false,
        })
    }

    /// Asks for whatever a run wants beyond the frame.
    ///
    /// Nothing here changes the sample loop. The shader fills the feature
    /// buffers either way — leaving them unbound would be a second pipeline —
    /// and the filter does not exist until the last sample has landed, so what
    /// this decides is only what happens after.
    pub fn wants(&mut self, outputs: Outputs) {
        self.outputs = outputs;
    }

    /// The gain the preview has to paint through, so that what it draws is what
    /// the PNG will hold.
    pub fn exposure(&self) -> f32 {
        self.exposure
    }

    /// The frame's dimensions, which the preview fits its block to.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Samples the budget has left to hand out.
    pub fn remaining(&self) -> u32 {
        self.samples - self.submitted
    }

    /// Queues the next sample, blocking on an older one if too many are in
    /// flight. A no-op once the budget is spent.
    ///
    /// One dispatch per sample, each reseeding the shader's RNG from the sample
    /// index it is handed.
    pub fn sample(&mut self) -> Result<(), wgpu::PollError> {
        if self.remaining() == 0 {
            return Ok(());
        }

        self.submitted += 1;
        let sample = self.submitted;
        let offset = std::mem::offset_of!(GpuCamera, sample) as wgpu::BufferAddress;
        // A queued write lands before the submission that follows it.
        self.queue
            .write_buffer(&self.camera_buffer, offset, bytemuck::bytes_of(&sample));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sample"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("trace"),
                timestamp_writes: self.timer.as_ref().and_then(|timer| timer.writes(sample)),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.frame, &[]);
            pass.set_bind_group(1, &self.geometry, &[]);
            // Rounded up to whole workgroups; the shader drops the threads that
            // land outside the image.
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }
        self.in_flight
            .push_back(self.queue.submit([encoder.finish()]));

        // Let a few samples queue up, then block on the oldest. Waiting keeps
        // the count honest — without it every sample would "finish" instantly
        // and the GPU would still be tracing long after the loop ended.
        if self.in_flight.len() > IN_FLIGHT_SAMPLES {
            self.retire()?;
        }

        Ok(())
    }

    /// Blocks until the oldest queued sample has run, then counts it.
    fn retire(&mut self) -> Result<(), wgpu::PollError> {
        let Some(submission) = self.in_flight.pop_front() else {
            return Ok(());
        };

        self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })?;
        self.completed += 1;
        self.progress.advance();

        Ok(())
    }

    /// Queues a copy of the accumulator and asks for it to be mapped, waiting
    /// for neither. [`Renderer::snapshot`] collects the result.
    ///
    /// Safe mid-render: the copy is queued on the same queue as the samples, so
    /// it is ordered behind every one already submitted and cannot catch a
    /// dispatch half-written. Samples submitted *after* it land in the
    /// accumulator after it has been read, which is what makes a snapshot a
    /// consistent frame rather than a smear.
    ///
    /// A no-op while a request is already outstanding. There is one readback
    /// buffer and it cannot be written to while it is mapped.
    ///
    /// It deliberately does **not** resolve the query set. [`Timer::resolve`]
    /// documents why that has to wait for the end.
    pub fn request(&mut self) {
        if self.pending.is_some() {
            return;
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        encoder.copy_buffer_to_buffer(
            &self.accumulator,
            0,
            &self.readback,
            0,
            self.accumulator_size,
        );
        self.queue.submit([encoder.finish()]);

        let (sender, receiver) = mpsc::channel();
        self.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        self.pending = Some(receiver);
    }

    /// The sums from an outstanding [`Renderer::request`], or `None` while the
    /// GPU has not finished the copy yet.
    ///
    /// Returning `None` rather than waiting is the whole point: the sample loop
    /// asks every iteration, keeps tracing while the answer is no, and picks the
    /// frame up whenever it happens to be ready.
    pub fn snapshot(&mut self) -> Result<Option<Vec<[f32; 4]>>, Box<dyn Error>> {
        let Some(receiver) = self.pending.take() else {
            return Ok(None);
        };

        // A map callback only fires from inside a poll. `retire` does a blocking
        // one every few samples and would deliver this on its own, but a
        // non-blocking check here is what keeps that an optimisation rather than
        // a dependency.
        self.device.poll(wgpu::PollType::Poll)?;
        match receiver.try_recv() {
            Err(mpsc::TryRecvError::Empty) => {
                self.pending = Some(receiver);
                return Ok(None);
            }
            Err(error) => return Err(error.into()),
            Ok(result) => result?,
        }

        let slice = self.readback.slice(..);
        let mapped = slice.get_mapped_range()?;
        let sums: Vec<[f32; 4]> = cast_slice(&mapped).to_vec();
        drop(mapped);
        self.readback.unmap();

        Ok(Some(sums))
    }

    /// [`Renderer::request`] and [`Renderer::snapshot`], waited on.
    pub fn sums(&mut self) -> Result<Vec<[f32; 4]>, Box<dyn Error>> {
        self.read(<[[f32; 4]]>::to_vec)
    }

    /// A fresh copy of the accumulator, handed to `read` as the mapped range
    /// rather than as a `Vec`.
    ///
    /// The closure is what keeps [`Renderer::finish`] from allocating a
    /// full-frame copy of the sums — sixteen bytes a pixel, 130MB at 1080p — on
    /// the way to pixels that are a quarter of the size. [`Renderer::sums`] is
    /// the same call with that copy asked for explicitly, because the preview
    /// hands the sums to another thread and cannot hold the buffer mapped while
    /// it draws.
    ///
    /// It is [`read_back`], so it costs a pipeline drain — which is what the
    /// preview's [`Renderer::request`] and [`Renderer::snapshot`] exist to
    /// avoid mid-render.
    fn read<T>(&mut self, read: impl FnOnce(&[[f32; 4]]) -> T) -> Result<T, Box<dyn Error>> {
        // The preview may have left a snapshot outstanding, and there is one
        // readback buffer: it cannot be copied into again until that map has
        // resolved and been thrown away.
        self.discard()?;

        read_back(
            &self.device,
            &self.queue,
            &self.accumulator,
            &self.readback,
            self.accumulator_size,
            read,
        )
    }

    /// What the moments say about this frame's noise.
    ///
    /// Its own buffer and its own readback, so none of this touches the one the
    /// preview's [`Renderer::request`]/[`Renderer::snapshot`] pair shares — the
    /// moments are wanted once, at the end, and never mid-render.
    ///
    /// The variance is built inside the closure for the same reason
    /// [`Renderer::read`] takes one: sixteen bytes a pixel of moments never has to
    /// exist as a `Vec` on the way to the summary. Nor does the variance field
    /// itself, unless [`Outputs::variance`] asked for a picture of it.
    pub fn statistics(&mut self) -> Result<Statistics, Box<dyn Error>> {
        let dimensions = (self.width, self.height);
        // Every pixel is in every dispatch, so the frame's sample count is the
        // pixel's. `completed` rather than `samples` because an interrupted
        // preview stops somewhere in between, and the moments only count the
        // dispatches that actually ran.
        let samples = self.completed;

        read_back(
            &self.device,
            &self.queue,
            &self.moments,
            &self.moments_readback,
            self.moments_size,
            |moments| Statistics::new(moments, dimensions, samples, self.outputs.variance),
        )
    }

    /// Waits out a request the preview left outstanding and throws it away,
    /// which is what frees the one readback buffer for the next one.
    ///
    /// Thrown away rather than returned because those sums read the accumulator
    /// as it stood when that request was *queued*, which is not what a caller
    /// asking now is asking for.
    fn discard(&mut self) -> Result<(), Box<dyn Error>> {
        let Some(receiver) = self.pending.take() else {
            return Ok(());
        };

        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        receiver.recv()??;
        self.readback.unmap();

        Ok(())
    }

    /// Waits until every sample handed to the queue has landed.
    ///
    /// [`Renderer::finish`] does this on the way past, so [`render`] never calls
    /// it. The preview does, to draw one last frame that is level with the sums
    /// it is about to resolve — and to draw it *before* [`Renderer::resolved`]
    /// closes off the progress bar's line, which would otherwise leave the
    /// cursor a row below the block the frame is placed against.
    pub fn drain(&mut self) -> Result<(), wgpu::PollError> {
        while !self.in_flight.is_empty() {
            self.retire()?;
        }

        Ok(())
    }

    /// The feature buffers, averaged and encoded, or `None` if nobody asked.
    ///
    /// Both come home through the accumulator's own readback buffer, which is
    /// exactly the right size for them and is free by the time this runs. The
    /// normals have to become a `Vec` because the two are needed together and
    /// only one buffer can be mapped at a time; the albedos never do.
    fn features(&mut self) -> Result<Option<Aovs>, Box<dyn Error>> {
        if !self.outputs.aovs {
            return Ok(None);
        }

        // The preview may still have a snapshot outstanding against the buffer
        // both of these are about to use.
        self.discard()?;

        let dimensions = (self.width, self.height);
        let samples = self.completed;
        let normals: Vec<[f32; 4]> = read_back(
            &self.device,
            &self.queue,
            &self.normals,
            &self.readback,
            self.accumulator_size,
            <[[f32; 4]]>::to_vec,
        )?;

        let aovs = read_back(
            &self.device,
            &self.queue,
            &self.albedos,
            &self.readback,
            self.accumulator_size,
            |albedos| Aovs::new(&normals, albedos, dimensions, samples),
        )?;

        Ok(Some(aovs))
    }

    /// Runs the a-trous filter over the finished frame, or `None` if nobody
    /// asked.
    ///
    /// Everything here is built on the spot and dropped on the way out: a shader
    /// compile, two full-frame scratch buffers and a handful of dispatches, none
    /// of which a render that only wants a picture should be paying for. It is
    /// also the only reason any of it can be this simple — the sample loop is
    /// over, the queue is empty, and the tracer's four buffers are read-only from
    /// here.
    fn filtered(&mut self) -> Result<Option<Denoised>, Box<dyn Error>> {
        if !self.outputs.denoise {
            return Ok(None);
        }

        let started = Instant::now();

        // The frame comes home through the accumulator's readback buffer at the
        // end of this, and the preview may still have a snapshot outstanding
        // against it. Both callers happen to have read the accumulator already,
        // which clears it — this is what keeps that a coincidence rather than a
        // requirement.
        self.discard()?;

        let module = self
            .device
            .create_shader_module(wgpu::include_wgsl!("denoise.wgsl"));
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("denoise"),
                // Reflected out of the shader, as the tracer's is. The one entry
                // point reaches every binding through the three passes it
                // branches between, so both groups come back whole and the
                // ping-pong pair stays interchangeable.
                layout: None,
                module: &module,
                entry_point: Some("denoise_pass"),
                compilation_options: Default::default(),
                cache: None,
            });

        let mut uniform = self.denoise;
        // The count the render reached, which an interrupted preview leaves
        // short of the one it was budgeted.
        uniform.samples = self.completed.max(1) as f32;

        let settings = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("denoise settings"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let scratch = |label| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: self.accumulator_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let ping = scratch("denoise ping");
        let pong = scratch("denoise pong");

        let inputs = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("denoise inputs"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: settings.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.accumulator.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.moments.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.normals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.albedos.as_entire_binding(),
                },
            ],
        });

        // The two directions the pair can be read in. A pass picks whichever one
        // writes into the buffer it is that pass's turn to fill.
        let layout = pipeline.get_bind_group_layout(1);
        let pair = |label, source: &wgpu::Buffer, destination: &wgpu::Buffer| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: source.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: destination.as_entire_binding(),
                    },
                ],
            })
        };
        let into_pong = pair("denoise ping to pong", &ping, &pong);
        let into_ping = pair("denoise pong to ping", &pong, &ping);

        // One submission per pass, because each rewrites the uniform and a
        // queued write only lands before the submission that follows it. Seven
        // submissions for a default schedule, once, at the end of a render that
        // has already made several hundred.
        // `prepare` has to land in `ping`, and every pass after it reads what
        // the one before wrote, so the pair alternates from there.
        let mut into_ping_now = true;
        for (pass, stride) in denoise::schedule(self.iterations) {
            uniform.stage = pass;
            uniform.stride = stride;
            self.queue
                .write_buffer(&settings, 0, bytemuck::bytes_of(&uniform));

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("denoise"),
                });
            {
                let mut compute = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("denoise"),
                    timestamp_writes: None,
                });
                compute.set_pipeline(&pipeline);
                compute.set_bind_group(0, &inputs, &[]);
                compute.set_bind_group(
                    1,
                    match into_ping_now {
                        true => &into_ping,
                        false => &into_pong,
                    },
                    &[],
                );
                compute.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
            }
            self.queue.submit([encoder.finish()]);

            // The buffer this pass just filled is the next one's source, so the
            // next one writes into the other.
            into_ping_now = !into_ping_now;
        }

        // The flag has moved on past the buffer the last pass wrote.
        let filled = match into_ping_now {
            true => &pong,
            false => &ping,
        };
        // `remodulate` writes a sample count of one alongside the colour, so the
        // frame resolves through exactly the code an unfiltered one does.
        let exposure = self.exposure;
        let pixels = read_back(
            &self.device,
            &self.queue,
            filled,
            &self.readback,
            self.accumulator_size,
            |sums| resolve(sums, exposure),
        )?;

        Ok(Some(Denoised {
            image: Image {
                width: self.width,
                height: self.height,
                pixels,
            },
            elapsed: started.elapsed(),
        }))
    }

    /// Waits on everything still in flight and resolves the frame.
    ///
    /// Takes `&mut self` rather than consuming, so that an interrupted preview
    /// can finish without unwinding the state it is still drawing from.
    pub fn finish(&mut self) -> Result<Render, Box<dyn Error>> {
        self.drain()?;
        // Resolved straight off the mapped range: the pixels are all this path
        // wants, so there is no reason for a copy of the sums to exist on the
        // way to them.
        let exposure = self.exposure;
        let pixels = self.read(|sums| resolve(sums, exposure))?;

        self.assembled(pixels)
    }

    /// [`Renderer::finish`] from sums the caller has already read back.
    ///
    /// The preview reads the accumulator itself to paint the last frame, and
    /// the renderer is drained by then, so nothing can have moved between that
    /// read and this one. Resolving what it read rather than asking the GPU for
    /// the same buffer again saves a second drain, a second full-frame copy
    /// over the bus, and a second full-frame allocation.
    pub fn resolved(&mut self, sums: &[[f32; 4]]) -> Result<Render, Box<dyn Error>> {
        self.assembled(resolve(sums, self.exposure))
    }

    /// The half of a finish that is not the pixels: the progress bar closed off,
    /// the query set read, and a [`Render`] built out of the three.
    fn assembled(&mut self, pixels: Vec<u8>) -> Result<Render, Box<dyn Error>> {
        // Before the progress bar closes off its line: this is a copy and a map
        // like any other, and both callers have already drained, so it costs a
        // round trip and no waiting on the sample loop.
        let statistics = self.statistics()?;
        let aovs = self.features()?;
        let denoised = self.filtered()?;

        if !self.finished {
            self.progress.finish();
        }

        // A query set may only be resolved once, and only after the passes that
        // wrote it have completed — so this is the one place it can happen, and
        // only on the first call.
        let timings = match (&self.timer, self.finished) {
            (Some(timer), false) => {
                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("timestamps"),
                        });
                timer.resolve(&mut encoder, self.completed);
                self.queue.submit([encoder.finish()]);
                timer.timings(&self.device, self.completed)?
            }
            _ => None,
        };
        self.finished = true;

        Ok(Render {
            image: Image {
                width: self.width,
                height: self.height,
                pixels,
            },
            renderer: self.renderer.clone(),
            timings,
            statistics,
            aovs,
            denoised,
        })
    }
}

/// Copies `source` into `readback`, waits for the map, and hands the mapped
/// range to `read`.
///
/// A free function rather than a method because it is called with two different
/// pairs of buffers and a method taking `&mut self` could not be handed one of
/// the renderer's own fields.
///
/// This is the call that costs a pipeline drain: the map only resolves once
/// every submission ahead of it has run, so the queue empties and has to refill.
/// The preview avoids it mid-render with [`Renderer::request`] and
/// [`Renderer::snapshot`], which ask the same question without waiting for the
/// answer.
fn read_back<T: Pod, R>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    readback: &wgpu::Buffer,
    size: u64,
    read: impl FnOnce(&[T]) -> R,
) -> Result<R, Box<dyn Error>> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback"),
    });
    encoder.copy_buffer_to_buffer(source, 0, readback, 0, size);
    queue.submit([encoder.finish()]);

    let (sender, receiver) = mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

    loop {
        device.poll(wgpu::PollType::wait_indefinitely())?;
        match receiver.try_recv() {
            Err(mpsc::TryRecvError::Empty) => continue,
            Err(error) => return Err(error.into()),
            Ok(result) => break result?,
        }
    }

    let slice = readback.slice(..);
    let mapped = slice.get_mapped_range()?;
    let value = read(cast_slice(&mapped));
    drop(mapped);
    readback.unmap();

    Ok(value)
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

/// Box-averages the accumulated sums down to `to`, for a preview that has to fit
/// in a terminal.
///
/// Averaging the sums rather than the resolved pixels is the whole point: the
/// radiance in the accumulator is linear, and averaging linear light before
/// gamma-encoding it is the correct order. Reversing the two darkens every edge
/// — a black and a white pixel average to mid-grey, which is 186 out of 255
/// after a gamma of 2.2 and not 128.
///
/// The sample count in `w` is summed alongside the radiance, so [`resolve`]'s
/// divide still lands on the right number with nothing special-cased.
///
/// A `to` at or above `from` in either axis is returned unscaled: this exists to
/// shrink a render into a terminal, and there is nothing to be gained by
/// magnifying one first.
fn downsample(sums: &[[f32; 4]], from: (u32, u32), to: (u32, u32)) -> Vec<[f32; 4]> {
    let (width, height) = from;
    let (target_width, target_height) = to;
    if target_width == 0 || target_height == 0 {
        return Vec::new();
    }
    if target_width >= width && target_height >= height {
        return sums.to_vec();
    }

    let mut out = Vec::with_capacity((target_width * target_height) as usize);
    for y in 0..target_height {
        // The source rows this output row covers. Computed from the edges rather
        // than from a stride so that every source pixel lands in exactly one
        // box, whatever the ratio between the two sizes.
        let top = (y as u64 * height as u64 / target_height as u64) as u32;
        let bottom = (((y + 1) as u64 * height as u64 / target_height as u64) as u32).max(top + 1);

        for x in 0..target_width {
            let left = (x as u64 * width as u64 / target_width as u64) as u32;
            let right =
                (((x + 1) as u64 * width as u64 / target_width as u64) as u32).max(left + 1);

            let mut total = [0.0f64; 4];
            for row in top..bottom {
                for column in left..right {
                    let sum = sums[(row * width + column) as usize];
                    for (channel, value) in total.iter_mut().zip(sum) {
                        *channel += value as f64;
                    }
                }
            }

            let count = ((bottom - top) * (right - left)) as f64;
            out.push(total.map(|channel| (channel / count) as f32));
        }
    }

    out
}

/// Averages each pixel's accumulated radiance, exposes it, tone maps it, and
/// gamma-encodes the result to 8-bit sRGB.
///
/// Three steps in that order, and the order is the whole of it. The average is
/// taken on linear radiance, because that is the only space in which averaging
/// samples means anything. The curve then decides what happens to the light that
/// will not fit in a display — see [`tonemap`], which is where the reasoning
/// lives. Only then does the gamma encode run, which is a statement about how a
/// monitor reads a byte and not about the image.
///
/// Everything upstream of this function works in linear light: the accumulator,
/// the downsample the preview draws through, the a-trous filter's edge-stopping
/// weights, and the standard error the noise figure reports. This is the one
/// place any of it stops being linear — and the only place `exposure` is
/// applied, so no amount of it can flatter a render that has not converged.
fn resolve(sums: &[[f32; 4]], exposure: f32) -> Vec<u8> {
    sums.iter()
        .flat_map(|sum| {
            let scale = match sum[3] > 0.0 {
                true => 1.0 / sum[3],
                false => 0.0,
            };

            // Exposure folds into the averaging divide, so it is free: one gain
            // on the way into the curve, and the only step between the scene's
            // radiance and the display's.
            let scale = scale * exposure;
            let display = tonemap::aces([sum[0] * scale, sum[1] * scale, sum[2] * scale]);
            // `aces` has already bounded this to [0, 1] and taken the NaN with
            // it, so the encode is only the curve a monitor expects.
            let encode = |channel: f32| (channel.powf(1.0 / 2.2) * 255.0 + 0.5) as u8;

            [
                encode(display[0]),
                encode(display[1]),
                encode(display[2]),
                255,
            ]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unity gain: what every scene gets unless it says otherwise, and what the
    /// tests below that are not about exposure resolve at.
    const NEUTRAL: f32 = 1.0;
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
        struct_sizes(include_str!("shader.wgsl"))
    }

    fn struct_sizes(source: &str) -> Vec<(String, u32)> {
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

    /// The device asks for exactly the limit the shader turns out to need, so
    /// a binding added without raising [`STORAGE_BUFFERS`] fails here rather
    /// than inside `create_compute_pipeline` on whichever machine runs it next.
    #[test]
    fn the_device_asks_for_the_storage_buffers_the_shader_binds() {
        let storage_bindings = |source: &str| {
            let module = naga::front::wgsl::parse_str(source).expect("the shader should parse");
            module
                .global_variables
                .iter()
                .filter(|(_, global)| matches!(global.space, naga::AddressSpace::Storage { .. }))
                .count() as u32
        };

        // The limit is per stage, so it is the hungrier of the two pipelines
        // that sets it — today the tracer, with the filter reading four of its
        // buffers and ping-ponging between two more.
        let tracer = storage_bindings(include_str!("shader.wgsl"));
        let filter = storage_bindings(include_str!("denoise.wgsl"));

        assert_eq!(tracer.max(filter), STORAGE_BUFFERS);
    }

    /// The filter is a second module with its own pipeline, and it has to
    /// compile and lay out the same way the tracer does.
    #[test]
    fn the_filter_agrees_with_the_struct_it_is_handed() {
        let sizes = struct_sizes(include_str!("denoise.wgsl"));
        let denoise = sizes
            .iter()
            .find(|(name, _)| name == "Denoise")
            .unwrap_or_else(|| panic!("the filter should declare Denoise, found {sizes:?}"));

        assert_eq!(denoise.1 as usize, size_of::<GpuDenoise>());
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
        // Never uploaded, but read back: the host walks them as `[f32; 4]`.
        assert_eq!(wgsl("Moments"), size_of::<[f32; 4]>());
        assert_eq!(wgsl("Albedo"), size_of::<[f32; 4]>());
    }

    #[test]
    fn averages_by_the_sample_count_it_was_given() {
        // Four passes of mid-grey average back to mid-grey, not to four times it.
        let one = resolve(&[[0.25, 0.5, 0.75, 1.0]], NEUTRAL);
        let four = resolve(&[[1.0, 2.0, 3.0, 4.0]], NEUTRAL);

        assert_eq!(one, four);
        assert_eq!(one[3], 255, "the frame is opaque");
    }

    #[test]
    fn an_untouched_pixel_is_black_and_not_a_nan() {
        assert_eq!(resolve(&[[0.0; 4]], NEUTRAL), [0, 0, 0, 255]);
    }

    /// The shader rejects a non-finite sample before it reaches the accumulator,
    /// so this should never fire in practice. It is asserted anyway because it
    /// is the other half of that guarantee: `f32::max` returns its non-NaN
    /// operand, which is the only reason a poisoned sum resolves to black rather
    /// than to whatever `as u8` makes of a NaN.
    ///
    /// An infinity comes back white in all three channels rather than in the one
    /// it arrived in. That is the tone curve rather than a loss of information —
    /// a colour bleaches toward neutral as it gets brighter, the way film does,
    /// and nothing is brighter than this.
    #[test]
    fn a_poisoned_sum_still_resolves_to_a_pixel() {
        let nan = f32::NAN;
        assert_eq!(resolve(&[[nan, nan, nan, 1.0]], NEUTRAL), [0, 0, 0, 255]);
        assert_eq!(
            resolve(&[[f32::INFINITY, 0.0, 0.0, 1.0]], NEUTRAL),
            [255, 255, 255, 255],
        );
    }

    #[test]
    fn downsampling_box_averages() {
        // A 4x2 whose left half is one sample of 1.0 and right half two of 0.5,
        // halved in width: each output pixel is the mean of its two inputs.
        let sums = vec![
            [1.0, 1.0, 1.0, 1.0],
            [3.0, 3.0, 3.0, 1.0],
            [0.5, 0.5, 0.5, 2.0],
            [1.5, 1.5, 1.5, 2.0],
            [1.0, 1.0, 1.0, 1.0],
            [3.0, 3.0, 3.0, 1.0],
            [0.5, 0.5, 0.5, 2.0],
            [1.5, 1.5, 1.5, 2.0],
        ];
        let scaled = downsample(&sums, (4, 2), (2, 1));

        assert_eq!(scaled.len(), 2);
        assert_eq!(scaled[0], [2.0, 2.0, 2.0, 1.0]);
        assert_eq!(scaled[1], [1.0, 1.0, 1.0, 2.0]);
    }

    /// The whole reason the averaging happens on the sums rather than on the
    /// resolved pixels. A black pixel beside a white one is mid-grey in *linear*
    /// light, and mid-grey resolves to 163. Averaging the other way round —
    /// resolve each, then average the bytes — would land halfway between 0 and
    /// whatever white resolves to, and every edge in the preview would come out
    /// darker than the render it is previewing.
    #[test]
    fn downsampling_averages_before_the_tone_curve() {
        let sums = [[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]];
        let pixels = resolve(&downsample(&sums, (2, 1), (1, 1)), NEUTRAL);
        let white = resolve(&[[1.0, 1.0, 1.0, 1.0]], NEUTRAL)[0];

        assert_eq!(pixels[0], 163, "0.5 linear should resolve to 163");
        assert_ne!(
            pixels[0],
            (white as u16).div_ceil(2) as u8,
            "that is the wrong order, not this one",
        );
    }

    /// A ratio that does not divide evenly still has to put every source pixel
    /// in exactly one box, and never read past the end of the buffer.
    #[test]
    fn downsampling_covers_an_uneven_ratio() {
        let sums: Vec<[f32; 4]> = (0..7 * 5).map(|n| [n as f32, 0.0, 0.0, 1.0]).collect();
        let scaled = downsample(&sums, (7, 5), (3, 2));

        assert_eq!(scaled.len(), 6);
        assert!(scaled.iter().all(|sum| sum[3] == 1.0), "{scaled:?}");
        // The first box is the top-left 2x2 of a 7x5: indices 0, 1, 7 and 8.
        assert_eq!(scaled[0][0], (0.0 + 1.0 + 7.0 + 8.0) / 4.0);
    }

    /// The preview only ever shrinks a frame. Asking for more than there is
    /// returns what there is rather than inventing pixels.
    #[test]
    fn downsampling_never_magnifies() {
        let sums = [[1.0, 2.0, 3.0, 1.0], [4.0, 5.0, 6.0, 1.0]];

        assert_eq!(downsample(&sums, (2, 1), (2, 1)), sums);
        assert_eq!(downsample(&sums, (2, 1), (9, 9)), sums);
        assert!(downsample(&sums, (2, 1), (0, 1)).is_empty());
    }

    /// One linear unit is no longer the top of the range: the curve keeps
    /// headroom above it so that a light at 15 and a surface at 1 are still
    /// different pixels, which is the entire reason the curve is here.
    #[test]
    fn resolve_leaves_headroom_above_one() {
        let pixels = resolve(
            &[[1.0; 4], [4.0, 4.0, 4.0, 1.0], [15.0, 15.0, 15.0, 1.0]],
            NEUTRAL,
        );

        assert!(pixels[0] < 255, "one should not clip: {}", pixels[0]);
        assert!(pixels[0] < pixels[4], "four is brighter than one");
        assert!(pixels[4] < pixels[8], "and fifteen brighter than four");
    }

    /// Exposure is a gain on the way into the curve, so it moves a frame up and
    /// down the same way pointing the camera at more light would.
    #[test]
    fn exposure_scales_the_frame_before_the_curve() {
        let sums = [[0.18, 0.18, 0.18, 1.0]];

        let dim = resolve(&sums, 0.5)[0];
        let neutral = resolve(&sums, NEUTRAL)[0];
        let bright = resolve(&sums, 2.0)[0];

        assert!(dim < neutral, "{dim} vs {neutral}");
        assert!(neutral < bright, "{neutral} vs {bright}");

        // And it is exactly a gain, not a curve of its own: twice the exposure
        // on half the light is the same pixel.
        assert_eq!(resolve(&[[0.09, 0.09, 0.09, 1.0]], 2.0)[0], neutral);
    }

    /// A gain of zero is a legal thing to ask for and is not a special case: it
    /// is a black frame, the way pointing a camera at nothing is.
    #[test]
    fn an_exposure_of_zero_is_a_black_frame_rather_than_a_divide() {
        let pixels = resolve(&[[1.0, 2.0, 3.0, 1.0]], 0.0);

        assert_eq!(pixels, [0, 0, 0, 255]);
    }

    /// A negative channel floors and a bright one rolls off rather than
    /// clipping. The small red in the first pixel is the tone curve's doing and
    /// not a bug: the curve runs in a wider set of primaries than sRGB, and a
    /// colour on the edge of the display's gamut picks up a little of its
    /// neighbours coming back out. That is what stops a saturated highlight from
    /// swinging its hue on the way to white.
    #[test]
    fn resolve_bounds_what_the_curve_is_given() {
        let pixels = resolve(&[[0.0, 0.5, 1.0, 1.0], [-1.0, 2.0, 1.0, 1.0]], NEUTRAL);

        assert!(pixels[0] < 32, "black-ish, not black: {}", pixels[0]);
        assert!(pixels[1] > 128, "0.5 should brighten to {}", pixels[1]);
        assert!(pixels[2] > pixels[1], "and 1.0 further still");
        assert!(
            pixels[4] < pixels[5],
            "a negative floors below its neighbour"
        );
    }
}
