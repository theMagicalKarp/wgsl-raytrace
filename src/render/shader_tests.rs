//! Unit tests for `shader.wgsl`'s functions, run on the GPU against the shader
//! that actually ships rather than against a Rust copy of its math.
//!
//! [`run`] appends a small compute entry point to the tracer's source. Each test
//! supplies an `Input` struct, an `Output` struct and a `test` function mapping
//! one to the other; the harness maps it over a storage buffer of inputs and
//! reads the outputs back. The pipeline's layout is reflected from the entry
//! point, and only bindings it reaches end up in it, so the render's own groups
//! stay out of the way as long as the function under test does not touch them.

use bytemuck::Pod;
use bytemuck::Zeroable;
use bytemuck::cast_slice;
use naga::proc::Layouter;
use naga::valid::Capabilities;
use naga::valid::ValidationFlags;
use naga::valid::Validator;
use std::env;
use std::sync::mpsc;
use wgpu::util::BufferInitDescriptor;
use wgpu::util::DeviceExt;

/// Threads per workgroup. A single axis, since inputs are a flat list.
const WORKGROUP: u32 = 64;

/// The entry point every test module gets. Group 0 is shared with the render's
/// frame bindings, which stop at 4; these sit well clear of them so a test can
/// call a function that reads `camera` without a collision.
const ENTRY: &str = r#"
@group(0) @binding(16) var<storage, read> test_inputs: array<Input>;
@group(0) @binding(17) var<storage, read_write> test_outputs: array<Output>;

@compute @workgroup_size(64, 1, 1)
fn test_main(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= arrayLength(&test_inputs) {
        return;
    }
    test_outputs[id.x] = test(test_inputs[id.x], id.x);
}
"#;

/// Maps the WGSL function `test(input: Input, index: u32) -> Output`, declared
/// in `source` along with both structs, over `inputs`.
///
/// `None` when `WGSL_RAYTRACE_SKIP_GPU_TESTS` is set, the same switch the golden
/// renders honour. `I` and `O` have to lay out exactly as `Input` and `Output`
/// do, which is checked before anything is dispatched.
fn run<I: Pod, O: Pod>(source: &str, inputs: &[I]) -> Option<Vec<O>> {
    if env::var_os("WGSL_RAYTRACE_SKIP_GPU_TESTS").is_some() {
        return None;
    }
    assert!(!inputs.is_empty(), "an empty buffer cannot be bound");

    let module = format!("{}\n{source}\n{ENTRY}", include_str!("shader.wgsl"));
    check_layout::<I, O>(&module);

    Some(pollster::block_on(dispatch(&module, inputs)).expect(
        "the test shader should run — set WGSL_RAYTRACE_SKIP_GPU_TESTS=1 \
         on a machine with no working adapter",
    ))
}

/// Validates the module with naga, which reports errors against the combined
/// source far more readably than a failed pipeline does, and holds the host
/// types against the WGSL structs.
fn check_layout<I, O>(module: &str) {
    let parsed = naga::front::wgsl::parse_str(module)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(module)));
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&parsed)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(module)));

    let mut layouter = Layouter::default();
    layouter.update(parsed.to_ctx()).unwrap();
    let size = |name: &str| {
        parsed
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some(name))
            .map(|(handle, _)| layouter[handle].size as usize)
            .unwrap_or_else(|| panic!("the test should declare {name}"))
    };

    assert_eq!(size("Input"), size_of::<I>(), "Input's host layout");
    assert_eq!(size("Output"), size_of::<O>(), "Output's host layout");
}

async fn dispatch<I: Pod, O: Pod>(
    module: &str,
    inputs: &[I],
) -> Result<Vec<O>, Box<dyn std::error::Error>> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("shader tests"),
            ..Default::default()
        })
        .await?;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("shader tests"),
        source: wgpu::ShaderSource::Wgsl(module.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("shader tests"),
        layout: None,
        module: &shader,
        entry_point: Some("test_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let count = inputs.len();
    let size = (count * size_of::<O>()) as u64;
    let input = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("test inputs"),
        contents: cast_slice(inputs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test outputs"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test bindings"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 16,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 17,
                resource: output.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("shader tests"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("shader tests"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups((count as u32).div_ceil(WORKGROUP), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size);
    queue.submit([encoder.finish()]);

    let (sender, receiver) = mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    receiver.recv()??;

    let mapped = readback.slice(..).get_mapped_range()?;
    let outputs = cast_slice(&mapped).to_vec();
    drop(mapped);
    readback.unmap();

    Ok(outputs)
}

/// A `vec3f` as it sits in a WGSL struct: twelve bytes, padded to sixteen.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
    _pad: f32,
}

impl Vec3 {
    fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 { x, y, z, _pad: 0.0 }
    }

    fn normalized(self) -> Vec3 {
        let length = self.dot(self).sqrt();
        Vec3::new(self.x / length, self.y / length, self.z / length)
    }

    fn dot(self, other: Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn cross(self, other: Vec3) -> Vec3 {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    fn distance(self, other: Vec3) -> f32 {
        let d = Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z);
        d.dot(d).sqrt()
    }
}

/// A deterministic spread of unit vectors: a Fibonacci sphere, which covers the
/// whole sphere evenly including both poles' neighbourhoods.
fn sphere(count: usize) -> Vec<Vec3> {
    let golden = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());
    (0..count)
        .map(|i| {
            let z = 1.0 - 2.0 * (i as f32 + 0.5) / count as f32;
            let radius = (1.0 - z * z).sqrt();
            let phi = golden * i as f32;
            Vec3::new(radius * phi.cos(), radius * phi.sin(), z)
        })
        .collect()
}

/// The normals a basis is most likely to get wrong: the poles, the equator a
/// sign flip happens across, and a hair either side of each pole, where the
/// naive construction divides by something near zero.
fn awkward_normals() -> Vec<Vec3> {
    let mut normals = vec![
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.0, 0.0, -0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
    ];
    for epsilon in [1e-2, 1e-4, 1e-6, 1e-8] {
        for z in [1.0, -1.0] {
            normals.push(Vec3::new(epsilon, epsilon, z).normalized());
            normals.push(Vec3::new(-epsilon, epsilon, z).normalized());
            normals.push(Vec3::new(epsilon, 0.0, z).normalized());
        }
    }
    normals.extend(sphere(512));
    normals
}

const BASIS: &str = r#"
struct Input {
    n: vec3f,
    v: vec3f,
}

struct Output {
    t: vec3f,
    b: vec3f,
    n: vec3f,
    local_n: vec3f,
    round_trip: vec3f,
}

fn test(input: Input, index: u32) -> Output {
    let frame = orthonormal_basis(input.n);
    return Output(
        frame.t,
        frame.b,
        frame.n,
        to_local(frame, input.n),
        to_world(frame, to_local(frame, input.v)),
    );
}
"#;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct BasisInput {
    n: Vec3,
    v: Vec3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct BasisOutput {
    t: Vec3,
    b: Vec3,
    n: Vec3,
    local_n: Vec3,
    round_trip: Vec3,
}

fn basis(inputs: &[BasisInput]) -> Option<Vec<BasisOutput>> {
    run(BASIS, inputs)
}

fn basis_inputs() -> Vec<BasisInput> {
    let normals = awkward_normals();
    let vectors = sphere(normals.len());
    normals
        .into_iter()
        .zip(vectors.into_iter().rev())
        .map(|(n, v)| BasisInput { n, v })
        .collect()
}

const TOLERANCE: f32 = 1e-5;

#[test]
fn the_basis_is_orthonormal_and_right_handed() {
    let inputs = basis_inputs();
    let Some(outputs) = basis(&inputs) else {
        return;
    };

    for (input, frame) in inputs.iter().zip(&outputs) {
        let context = format!("n = {:?}, frame = {frame:?}", input.n);
        assert!(frame.n.distance(input.n) < TOLERANCE, "n: {context}");
        for (name, axis) in [("t", frame.t), ("b", frame.b)] {
            assert!(
                (axis.dot(axis) - 1.0).abs() < TOLERANCE,
                "|{name}|: {context}"
            );
            assert!(axis.dot(frame.n).abs() < TOLERANCE, "{name}·n: {context}");
        }
        assert!(frame.t.dot(frame.b).abs() < TOLERANCE, "t·b: {context}");
        assert!(
            frame.t.cross(frame.b).distance(frame.n) < TOLERANCE,
            "t × b should be n: {context}",
        );
    }
}

#[test]
fn a_normal_is_up_in_its_own_frame() {
    let inputs = basis_inputs();
    let Some(outputs) = basis(&inputs) else {
        return;
    };

    let up = Vec3::new(0.0, 0.0, 1.0);
    for (input, frame) in inputs.iter().zip(&outputs) {
        assert!(
            frame.local_n.distance(up) < TOLERANCE,
            "n = {:?} is {:?} locally",
            input.n,
            frame.local_n,
        );
    }
}

#[test]
fn to_world_undoes_to_local() {
    let inputs = basis_inputs();
    let Some(outputs) = basis(&inputs) else {
        return;
    };

    for (input, frame) in inputs.iter().zip(&outputs) {
        assert!(
            frame.round_trip.distance(input.v) < TOLERANCE,
            "n = {:?}: {:?} came back as {:?}",
            input.n,
            input.v,
            frame.round_trip,
        );
    }
}

/// The frame has no discontinuity near either pole. Neighbouring normals a hair
/// apart should get neighbouring tangents, which is what the Frisvad basis this
/// construction replaces fails at `n = -z`.
#[test]
fn the_basis_is_continuous_near_the_poles() {
    let mut inputs = Vec::new();
    for z in [1.0f32, -1.0] {
        for epsilon in [0.0f32, 1e-7, 1e-5, 1e-3] {
            let n = Vec3::new(epsilon, epsilon * 0.5, z).normalized();
            inputs.push(BasisInput { n, v: n });
        }
    }
    let Some(outputs) = basis(&inputs) else {
        return;
    };

    for (pole, frames) in outputs.chunks_exact(4).enumerate() {
        for pair in frames.windows(2) {
            // A thousandth of a radian of normal moves the tangent by about as
            // much; a flipped or spun frame moves it by order one.
            assert!(
                pair[0].t.distance(pair[1].t) < 1e-2,
                "pole {pole}: {:?} jumped to {:?}",
                pair[0].t,
                pair[1].t,
            );
        }
    }
}

const LAMBERTIAN: &str = r#"
struct Input {
    normal: vec3f,
    wo: vec3f,
    color: vec3f,
}

struct Output {
    wi: vec3f,
    weight: vec3f,
    value: vec3f,
    // x: sample pdf, y: eval pdf, z: delta, w: valid.
    pdf: vec4f,
}

fn test(input: Input, index: u32) -> Output {
    rng_state = jenkins_hash(index + 1u);

    // The lambertian preset, as `GpuMaterial::from` writes it.
    let material = Material(input.color, PRINCIPLED, 1.0, 0.0, 1.0, 0.0, 1.0);
    let hit = Intersection(input.normal, 1.0, 0u, true, 0u);
    let sample = bsdf_sample(material, hit, input.wo);
    let eval = bsdf_eval(material, hit, input.wo, sample.wi);

    return Output(
        sample.wi,
        sample.weight,
        eval.value,
        vec4f(sample.pdf, eval.pdf, f32(sample.delta), f32(sample.valid)),
    );
}
"#;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct LambertianInput {
    normal: Vec3,
    wo: Vec3,
    color: Vec3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct LambertianOutput {
    wi: Vec3,
    weight: Vec3,
    value: Vec3,
    pdf: [f32; 4],
}

/// What `bsdf_sample` reports about a direction it drew is what `bsdf_eval`
/// says about the same direction, which is the contract light sampling and the
/// heuristic both lean on. And the lambertian preset is still exactly the
/// cosine-weighted diffuse it was before it became a principled surface: no
/// specular lobe, so every draw is diffuse and its density is `cos / PI`.
#[test]
fn a_lambertian_sample_agrees_with_its_evaluation() {
    let normals = sphere(1024);
    let inputs: Vec<LambertianInput> = normals
        .iter()
        .zip(sphere(1024).iter().rev())
        .map(|(&normal, &wo)| LambertianInput {
            normal,
            // Facing the normal, the way an intersection always hands it over.
            wo: if wo.dot(normal) < 0.0 {
                Vec3::new(-wo.x, -wo.y, -wo.z)
            } else {
                wo
            },
            color: Vec3::new(0.8, 0.5, 0.2),
        })
        .collect();
    let Some(outputs) = run::<_, LambertianOutput>(LAMBERTIAN, &inputs) else {
        return;
    };

    for (input, output) in inputs.iter().zip(&outputs) {
        let context = format!("{input:?} -> {output:?}");
        let [sample_pdf, eval_pdf, delta, valid] = output.pdf;
        assert_eq!((delta, valid), (0.0, 1.0), "{context}");
        assert!(
            (output.wi.dot(output.wi) - 1.0).abs() < TOLERANCE,
            "{context}"
        );

        let cosine = output.wi.dot(input.normal);
        assert!(cosine >= 0.0, "drawn below the surface: {context}");
        assert!((sample_pdf - eval_pdf).abs() < 1e-4, "pdf: {context}");
        assert!(
            (sample_pdf - cosine / std::f32::consts::PI).abs() < 1e-4,
            "{context}"
        );

        // `weight` is `value / pdf`, channel by channel.
        if eval_pdf > 1e-3 {
            for (weight, value) in [
                (output.weight.x, output.value.x),
                (output.weight.y, output.value.y),
                (output.weight.z, output.value.z),
            ] {
                assert!((weight - value / eval_pdf).abs() < 1e-3, "{context}");
            }
        }
    }
}

/// A small deterministic generator for test inputs, so a failure reproduces.
/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32
    }
}

impl Vec3 {
    fn scale(self, s: f32) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }

    fn add(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    /// `local`, read in a frame whose `z` is `self`. Any frame will do: these
    /// only have to land at the right angle to the normal.
    fn frame(self, local: Vec3) -> Vec3 {
        let helper = match self.x.abs() < 0.9 {
            true => Vec3::new(1.0, 0.0, 0.0),
            false => Vec3::new(0.0, 1.0, 0.0),
        };
        let t = helper.cross(self).normalized();
        let b = self.cross(t);
        t.scale(local.x)
            .add(b.scale(local.y))
            .add(self.scale(local.z))
            .normalized()
    }
}

const PRINCIPLED: &str = r#"
struct Input {
    normal: vec3f,
    wo: vec3f,
    wi: vec3f,
    color: vec3f,
    // Whether the ray struck the side the mesh's normal points out of.
    front_face: u32,
    // roughness, metallic, ior, transmission.
    params: vec4f,
}

struct Output {
    // What `bsdf_sample` drew.
    wi: vec3f,
    weight: vec3f,
    // `bsdf_eval` at the drawn direction, at the given one, and at the given
    // one with the two directions swapped.
    sampled: vec3f,
    given: vec3f,
    swapped: vec3f,
    // x: sample pdf, y: eval pdf at the drawn direction, z: delta, w: valid.
    sample: vec4f,
    // x: eval pdf at the given direction, y: the same swapped.
    pdfs: vec4f,
}

fn test(input: Input, index: u32) -> Output {
    rng_state = jenkins_hash(index + 1u);

    let material = Material(
        input.color,
        PRINCIPLED,
        input.params.x,
        input.params.y,
        input.params.z,
        input.params.w,
        1.0,
    );
    let hit = Intersection(input.normal, 1.0, 0u, input.front_face != 0u, 0u);
    let sample = bsdf_sample(material, hit, input.wo);
    let sampled = bsdf_eval(material, hit, input.wo, sample.wi);
    let given = bsdf_eval(material, hit, input.wo, input.wi);
    let swapped = bsdf_eval(material, hit, input.wi, input.wo);

    return Output(
        sample.wi,
        sample.weight,
        sampled.value,
        given.value,
        swapped.value,
        vec4f(sample.pdf, sampled.pdf, f32(sample.delta), f32(sample.valid)),
        vec4f(given.pdf, swapped.pdf, 0.0, 0.0),
    );
}
"#;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct PrincipledInput {
    normal: Vec3,
    wo: Vec3,
    wi: Vec3,
    color: [f32; 3],
    front_face: u32,
    params: [f32; 4],
}

impl PrincipledInput {
    fn new(
        normal: Vec3,
        wo: Vec3,
        wi: Vec3,
        color: [f32; 3],
        roughness: f32,
        metallic: f32,
        ior: f32,
    ) -> Self {
        PrincipledInput {
            normal,
            wo,
            wi,
            color,
            front_face: 1,
            params: [roughness, metallic, ior, 0.0],
        }
    }

    fn transmission(self, transmission: f32) -> Self {
        let [roughness, metallic, ior, _] = self.params;
        PrincipledInput {
            params: [roughness, metallic, ior, transmission],
            ..self
        }
    }

    /// Struck from inside the object, so the glass sees the index inverted.
    fn back_face(self) -> Self {
        PrincipledInput {
            front_face: 0,
            ..self
        }
    }

    fn roughness(&self) -> f32 {
        self.params[0]
    }

    fn ior(&self) -> f32 {
        self.params[2]
    }

    fn transmissive(&self) -> bool {
        self.params[3] > 0.0
    }

    /// The index across the surface along the path, as `relative_ior`.
    fn eta(&self) -> f32 {
        match self.front_face {
            0 => 1.0 / self.ior(),
            _ => self.ior(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct PrincipledOutput {
    wi: Vec3,
    weight: Vec3,
    sampled: Vec3,
    given: Vec3,
    swapped: Vec3,
    sample: [f32; 4],
    pdfs: [f32; 4],
}

impl PrincipledOutput {
    fn valid(&self) -> bool {
        self.sample[3] == 1.0
    }

    fn delta(&self) -> bool {
        self.sample[2] == 1.0
    }
}

fn principled(inputs: &[PrincipledInput]) -> Option<Vec<PrincipledOutput>> {
    run(PRINCIPLED, inputs)
}

/// Where the GGX alpha stops being a distribution and the shader samples a
/// mirror, as `DELTA_ALPHA` there.
const DELTA_ALPHA: f64 = 1e-3;

/// The directional albedo of a white GGX conductor, `∫ D G2 / (4 cos_o) dω_i`
/// over the hemisphere, by quadrature on the CPU in doubles.
///
/// Integrated over microfacet normals rather than over `wi`, with `θm =
/// atan(α tan t)` so the grid in `t` follows the lobe however narrow alpha
/// makes it: `dω_i = 4 (wo·m) dω_m`, and a reflection that lands below the
/// horizon is light the single-scattering model loses.
fn reference_albedo(cos_o: f64, alpha: f64) -> f64 {
    use std::f64::consts::FRAC_PI_2;
    use std::f64::consts::PI;

    if alpha < DELTA_ALPHA {
        return 1.0;
    }

    let a2 = alpha * alpha;
    let lambda = |v: [f64; 3]| {
        let tan2 = (v[0] * v[0] + v[1] * v[1]) / (v[2] * v[2]);
        0.5 * ((1.0 + a2 * tan2).sqrt() - 1.0)
    };
    let wo = [(1.0 - cos_o * cos_o).max(0.0).sqrt(), 0.0, cos_o];
    let lambda_o = lambda(wo);

    let (steps_t, steps_phi) = (1024, 512);
    let dt = FRAC_PI_2 / steps_t as f64;
    // `wo` lies in the xz plane, so the integrand is symmetric in phi and half
    // of the circle counts twice.
    let dphi = PI / steps_phi as f64;

    let mut sum = 0.0;
    for i in 0..steps_t {
        let t = (i as f64 + 0.5) * dt;
        let tan_t = t.tan();
        let theta = (alpha * tan_t).atan();
        let dtheta = alpha / (t.cos().powi(2) * (1.0 + a2 * tan_t * tan_t)) * dt;
        let (sin_m, cos_m) = (theta.sin(), theta.cos());
        let d = a2 / (PI * (sin_m * sin_m + a2 * cos_m * cos_m).powi(2));

        for j in 0..steps_phi {
            let phi = (j as f64 + 0.5) * dphi;
            let m = [sin_m * phi.cos(), sin_m * phi.sin(), cos_m];
            let dot = wo[0] * m[0] + wo[2] * m[2];
            if dot <= 0.0 {
                continue;
            }
            let wi = [
                2.0 * dot * m[0] - wo[0],
                2.0 * dot * m[1],
                2.0 * dot * m[2] - wo[2],
            ];
            if wi[2] <= 0.0 {
                continue;
            }

            let g2 = 1.0 / (1.0 + lambda_o + lambda(wi));
            sum += 2.0 * d * g2 * dot / cos_o * sin_m * dtheta * dphi;
        }
    }
    sum
}

fn ggx_alpha(roughness: f32) -> f64 {
    (roughness as f64 * roughness as f64).max(1e-4)
}

/// A white conductor's directional albedo, estimated from `bsdf_sample`'s
/// weights, against the quadrature above.
///
/// Nothing reflects more than arrives, a smooth mirror returns everything, and
/// rougher is never brighter. Rough is not expected to return everything:
/// single-scattering GGX loses what bounces between microfacets, and
/// compensating for that is deferred.
#[test]
fn a_white_conductor_never_gains_energy_and_loses_it_with_roughness() {
    const SAMPLES: usize = 8192;
    let cosines = [1.0f32, 0.7, 0.4, 0.15];
    let roughnesses = [0.0f32, 0.2, 0.4, 0.6, 0.8, 1.0];

    let normal = Vec3::new(0.0, 0.0, 1.0);
    let mut inputs = Vec::new();
    for &cos_o in &cosines {
        let wo = Vec3::new((1.0 - cos_o * cos_o).sqrt(), 0.0, cos_o);
        for &roughness in &roughnesses {
            let input = PrincipledInput::new(normal, wo, normal, [1.0; 3], roughness, 1.0, 1.5);
            inputs.extend(std::iter::repeat_n(input, SAMPLES));
        }
    }
    let Some(outputs) = principled(&inputs) else {
        return;
    };

    for (row, &cos_o) in cosines.iter().enumerate() {
        let mut previous = f64::INFINITY;
        for (column, &roughness) in roughnesses.iter().enumerate() {
            let start = (row * roughnesses.len() + column) * SAMPLES;
            let weights: Vec<f64> = outputs[start..start + SAMPLES]
                .iter()
                .map(|output| match output.valid() {
                    true => output.weight.y as f64,
                    false => 0.0,
                })
                .collect();

            let mean = weights.iter().sum::<f64>() / SAMPLES as f64;
            let variance = weights.iter().map(|w| (w - mean).powi(2)).sum::<f64>() / SAMPLES as f64;
            let tolerance = 4.0 * (variance / SAMPLES as f64).sqrt() + 2e-3;
            let expected = reference_albedo(cos_o as f64, ggx_alpha(roughness));

            let context = format!(
                "cos_o {cos_o}, roughness {roughness}: E = {mean:.4} ± {tolerance:.4}, \
                 quadrature says {expected:.4}"
            );
            assert!(mean <= 1.0 + tolerance, "gained energy: {context}");
            assert!(
                mean <= previous + tolerance,
                "brighter than smoother: {context}"
            );
            assert!((mean - expected).abs() <= tolerance, "{context}");
            previous = mean;
        }
    }

    // And the loss is real, or the quadrature and the shader could agree on
    // nothing happening. At alpha one GGX's normals are uniform, so head on
    // every facet past 45 degrees reflects below the horizon and at most half
    // the light comes back before masking takes its share.
    assert!(reference_albedo(1.0, 1.0) < 0.5);
}

/// Every non-delta lobe's density, mixed by its chance of being picked,
/// integrates over the sphere to the chance a draw is valid at all — one, less
/// the microfacet reflections that land below the horizon and the refractions
/// that land above it.
///
/// Priced with `bsdf_eval` over a stratified uniform grid of directions and
/// held against how often the same material's `bsdf_sample` came back valid.
#[test]
fn the_density_integrates_to_the_chance_of_a_valid_draw() {
    const SIDE: usize = 128;
    let materials = [
        // roughness, metallic, ior, transmission, front face, color. Rough
        // enough throughout that no lobe, even at a grazing `wo`, slips between
        // the grid's directions.
        (0.5f32, 1.0f32, 1.5f32, 0.0f32, true, [0.9f32, 0.6, 0.3]),
        (0.7, 0.0, 1.5, 0.0, true, [0.5, 0.5, 0.5]),
        (1.0, 0.5, 1.8, 0.0, true, [0.2, 0.8, 0.4]),
        (0.6, 0.0, 1.0, 0.0, true, [0.7, 0.7, 0.7]),
        (0.5, 0.0, 1.5, 1.0, true, [1.0, 1.0, 1.0]),
        (0.5, 0.0, 1.5, 1.0, false, [1.0, 1.0, 1.0]),
        (0.8, 0.2, 1.33, 0.6, true, [0.3, 0.6, 0.9]),
        (0.5, 0.0, 2.4, 1.0, false, [0.8, 0.8, 0.8]),
    ];
    let cosines = [0.95f32, 0.5, 0.15];
    let normal = Vec3::new(0.3, -0.8, 0.52).normalized();

    let mut inputs = Vec::new();
    for &(roughness, metallic, ior, transmission, front, color) in &materials {
        for &cos_o in &cosines {
            let wo = normal.frame(Vec3::new((1.0 - cos_o * cos_o).sqrt(), 0.0, cos_o));
            for i in 0..SIDE {
                for j in 0..SIDE {
                    // Uniform in `cos` is uniform in solid angle.
                    let z = -1.0 + 2.0 * (i as f32 + 0.5) / SIDE as f32;
                    let phi = std::f32::consts::TAU * (j as f32 + 0.5) / SIDE as f32;
                    let r = (1.0 - z * z).sqrt();
                    let wi = normal.frame(Vec3::new(r * phi.cos(), r * phi.sin(), z));
                    let input =
                        PrincipledInput::new(normal, wo, wi, color, roughness, metallic, ior)
                            .transmission(transmission);
                    inputs.push(match front {
                        true => input,
                        false => input.back_face(),
                    });
                }
            }
        }
    }
    let Some(outputs) = principled(&inputs) else {
        return;
    };

    let count = SIDE * SIDE;
    for (case, chunk) in outputs.chunks_exact(count).enumerate() {
        let material = materials[case / cosines.len()];
        let cos_o = cosines[case % cosines.len()];

        let integral = chunk.iter().map(|o| o.pdfs[0] as f64).sum::<f64>()
            * (2.0 * std::f64::consts::TAU)
            / count as f64;
        let valid = chunk.iter().filter(|o| o.valid()).count() as f64 / count as f64;

        let context =
            format!("{material:?}, cos_o {cos_o}: ∫pdf = {integral:.4}, valid {valid:.4}");
        assert!(integral <= 1.0 + 0.02, "{context}");
        assert!((integral - valid).abs() < 0.02, "{context}");
    }
}

/// What the smooth glass lobe can send `wo` off along about `normal`: the
/// mirror reflection, and the refraction unless it is totally internally
/// reflected.
fn smooth_directions(input: &PrincipledInput) -> (Vec3, Option<Vec3>) {
    let (n, wo) = (input.normal, input.wo);
    let cos_o = wo.dot(n);
    let reflected = n.scale(2.0 * cos_o).add(wo.scale(-1.0));

    let eta = input.eta();
    let sin2_t = (1.0 - cos_o * cos_o) / (eta * eta);
    let refracted = (sin2_t < 1.0).then(|| {
        wo.scale(-1.0 / eta)
            .add(n.scale(cos_o / eta - (1.0 - sin2_t).sqrt()))
    });
    (reflected, refracted)
}

/// For a direction `bsdf_sample` drew from a non-delta lobe, its density and
/// weight are what `bsdf_eval` says about that direction — across roughness,
/// metallic, index, transmission, orientation and which face was struck — and
/// a delta lobe is only ever reported where the surface has one.
#[test]
fn a_principled_sample_agrees_with_its_evaluation() {
    let mut rng = Rng(7);
    let normals = sphere(8192);
    let inputs: Vec<PrincipledInput> = normals
        .iter()
        .enumerate()
        .map(|(i, &normal)| {
            let cos_o = rng.next().max(1e-3);
            let phi = std::f32::consts::TAU * rng.next();
            let r = (1.0 - cos_o * cos_o).sqrt();
            let wo = normal.frame(Vec3::new(r * phi.cos(), r * phi.sin(), cos_o));
            let roughness = match i % 8 {
                0 => 0.0,
                1 => 0.02,
                _ => 0.05 + 0.95 * rng.next(),
            };
            let metallic = match i % 3 {
                0 => 0.0,
                1 => 1.0,
                _ => rng.next(),
            };
            let ior = [1.0, 1.33, 1.5, 2.4][i % 4];
            let transmission = match i % 5 {
                0 | 1 => 0.0,
                2 | 3 => 1.0,
                _ => rng.next(),
            };
            let color = [rng.next(), rng.next(), rng.next()];
            let input = PrincipledInput::new(normal, wo, normal, color, roughness, metallic, ior)
                .transmission(transmission);
            match rng.next() < 0.5 {
                true => input,
                false => input.back_face(),
            }
        })
        .collect();
    let Some(outputs) = principled(&inputs) else {
        return;
    };

    let mut smooth = 0;
    let mut mirrors = 0;
    let mut transmitted = 0;
    for (input, output) in inputs.iter().zip(&outputs) {
        let context = format!("{input:?} -> {output:?}");
        if !output.valid() {
            continue;
        }

        assert!((output.wi.dot(output.wi) - 1.0).abs() < 1e-4, "{context}");
        let below = output.wi.dot(input.normal) < 0.0;
        assert!(
            !below || input.transmissive(),
            "below an opaque surface: {context}"
        );
        transmitted += below as usize;
        for w in [output.weight.x, output.weight.y, output.weight.z] {
            assert!(w.is_finite() && w >= 0.0, "weight: {context}");
        }

        if output.delta() {
            mirrors += 1;
            let smooth_glass = input.transmissive() && input.ior() == 1.0;
            assert!(
                ggx_alpha(input.roughness()) < DELTA_ALPHA || smooth_glass,
                "a delta lobe on a rough surface: {context}"
            );
            // Reflected about the normal, or refracted through it by glass.
            let (reflected, refracted) = smooth_directions(input);
            let hits = |target: Vec3| output.wi.distance(target) < 1e-3;
            assert!(
                hits(reflected) || (input.transmissive() && refracted.is_some_and(hits)),
                "{context}"
            );
            continue;
        }

        smooth += 1;
        let [sample_pdf, eval_pdf, ..] = output.sample;
        assert!(sample_pdf > 0.0, "{context}");
        assert!(
            (sample_pdf - eval_pdf).abs() <= 1e-3 * sample_pdf.max(1.0),
            "pdf: {context}"
        );
        for (weight, value) in [
            (output.weight.x, output.sampled.x),
            (output.weight.y, output.sampled.y),
            (output.weight.z, output.sampled.z),
        ] {
            let expected = value / eval_pdf;
            assert!(
                (weight - expected).abs() <= 1e-3 * expected.max(1.0),
                "weight: {context}"
            );
        }
    }

    // Every branch was actually exercised.
    assert!(smooth > 5000, "{smooth} smooth samples");
    assert!(mirrors > 500, "{mirrors} delta samples");
    assert!(transmitted > 1000, "{transmitted} transmitted samples");
}

/// The microfacet lobes are reciprocal: `f(wo, wi) == f(wi, wo)`. Only those —
/// the diffuse base is scaled by `1 - F(wo)` and is not — so each case isolates
/// one of them: a pure conductor, and a black dielectric whose only lobe is
/// its specular coat.
#[test]
fn the_specular_lobes_are_reciprocal() {
    let mut rng = Rng(11);
    let normals = sphere(2048);
    let direction = |rng: &mut Rng, normal: Vec3| {
        let z = rng.next().max(1e-2);
        let phi = std::f32::consts::TAU * rng.next();
        let r = (1.0 - z * z).sqrt();
        normal.frame(Vec3::new(r * phi.cos(), r * phi.sin(), z))
    };
    let inputs: Vec<PrincipledInput> = normals
        .iter()
        .enumerate()
        .map(|(i, &normal)| {
            let wo = direction(&mut rng, normal);
            let wi = direction(&mut rng, normal);
            let roughness = 0.1 + 0.9 * rng.next();
            match i % 2 {
                0 => PrincipledInput::new(normal, wo, wi, [0.9, 0.6, 0.3], roughness, 1.0, 1.5),
                _ => PrincipledInput::new(normal, wo, wi, [0.0; 3], roughness, 0.0, 1.5),
            }
        })
        .collect();
    let Some(outputs) = principled(&inputs) else {
        return;
    };

    for (input, output) in inputs.iter().zip(&outputs) {
        let context = format!("{input:?} -> {output:?}");
        // `value` carries the cosine of the direction light arrives from.
        let forward = output.given.scale(1.0 / input.wi.dot(input.normal));
        let backward = output.swapped.scale(1.0 / input.wo.dot(input.normal));

        for (f, b) in [
            (forward.x, backward.x),
            (forward.y, backward.y),
            (forward.z, backward.z),
        ] {
            assert!(
                (f - b).abs() <= 1e-3 * f.abs().max(b.abs()).max(1e-3),
                "{context}"
            );
        }
    }
}

/// The directional albedo of white glass, reflection and transmission
/// together, estimated from `bsdf_sample`'s weights from both faces.
///
/// Smooth glass loses nothing: every draw either reflects or refracts, and
/// neither is tinted. Rough glass loses what single scattering does, as the
/// conductor above does, so it is only held to never gaining energy and never
/// brightening with roughness.
#[test]
fn white_glass_never_gains_energy_and_loses_it_with_roughness() {
    const SAMPLES: usize = 8192;
    let cosines = [1.0f32, 0.7, 0.4, 0.15];
    let roughnesses = [0.0f32, 0.2, 0.4, 0.6, 0.8, 1.0];
    let faces = [true, false];

    let normal = Vec3::new(0.0, 0.0, 1.0);
    let mut inputs = Vec::new();
    for &front in &faces {
        for &cos_o in &cosines {
            let wo = Vec3::new((1.0 - cos_o * cos_o).sqrt(), 0.0, cos_o);
            for &roughness in &roughnesses {
                let input = PrincipledInput::new(normal, wo, normal, [1.0; 3], roughness, 0.0, 1.5)
                    .transmission(1.0);
                let input = match front {
                    true => input,
                    false => input.back_face(),
                };
                inputs.extend(std::iter::repeat_n(input, SAMPLES));
            }
        }
    }
    let Some(outputs) = principled(&inputs) else {
        return;
    };

    let cases = cosines.len() * roughnesses.len();
    for (index, chunk) in outputs.chunks_exact(cases * SAMPLES).enumerate() {
        let front = faces[index];
        for (row, &cos_o) in cosines.iter().enumerate() {
            let mut previous = f64::INFINITY;
            for (column, &roughness) in roughnesses.iter().enumerate() {
                let start = (row * roughnesses.len() + column) * SAMPLES;
                let weights: Vec<f64> = chunk[start..start + SAMPLES]
                    .iter()
                    .map(|output| match output.valid() {
                        true => output.weight.y as f64,
                        false => 0.0,
                    })
                    .collect();

                let mean = weights.iter().sum::<f64>() / SAMPLES as f64;
                let variance =
                    weights.iter().map(|w| (w - mean).powi(2)).sum::<f64>() / SAMPLES as f64;
                let tolerance = 4.0 * (variance / SAMPLES as f64).sqrt() + 2e-3;

                let context = format!(
                    "front {front}, cos_o {cos_o}, roughness {roughness}: E = {mean:.4} ± {tolerance:.4}"
                );
                assert!(mean <= 1.0 + tolerance, "gained energy: {context}");
                assert!(
                    mean <= previous + tolerance,
                    "brighter than smoother: {context}"
                );
                if roughness == 0.0 {
                    assert!(
                        (mean - 1.0).abs() < 1e-4,
                        "smooth glass lost energy: {context}"
                    );
                }
                previous = mean;
            }
        }
    }
}

/// The radiance of every pixel of a render of `source`, a scene file whose
/// meshes resolve against `tests/golden`, as `(width, pixels)`. `edit` gets the
/// loaded scene before it is uploaded.
fn render_scene(
    source: &str,
    edit: impl FnOnce(&mut crate::scene::Scene),
) -> Option<(usize, Vec<[f32; 3]>)> {
    use crate::config::Config;
    use crate::scene::Scene;
    use std::path::Path;

    if env::var_os("WGSL_RAYTRACE_SKIP_GPU_TESTS").is_some() {
        return None;
    }

    let mut config: Config = toml::from_str(source).expect("the test scene should parse");
    config
        .validate(Path::new("tests/golden"))
        .expect("the test scene's mesh should exist");
    let mut scene = Scene::load(&config).expect("the test scene should load");
    edit(&mut scene);

    let mut renderer = super::Renderer::new(&config, &scene).expect(
        "the test scene should render — set WGSL_RAYTRACE_SKIP_GPU_TESTS=1 \
         on a machine with no working adapter",
    );
    while renderer.remaining() > 0 {
        renderer.sample().expect("a sample should dispatch");
    }
    let sums = renderer.sums().expect("the accumulator should read back");

    let (width, _) = renderer.dimensions();
    let pixels = sums
        .iter()
        .map(|[r, g, b, n]| [r / n, g / n, b / n])
        .collect();
    Some((width as usize, pixels))
}

/// The radiance of every pixel of a render of a sphere under a uniform white
/// sky, as `(width, pixels)`. `material` is the sphere's material, as the
/// lines of a scene file.
fn furnace(material: &str) -> Option<(usize, Vec<[f32; 3]>)> {
    // The smooth sphere from the normals scene: radius one about (-2, 2, 0),
    // filling most of the frame. Nothing else, so every path either escapes
    // straight to the sky or scatters off the sphere and then escapes — once
    // for a conductor, and as many times as it takes to get back out of glass.
    let source = format!(
        r#"
[camera]
aspect_ratio = "square"
image_width = 32
samples = 256
max_bounces = 32
fov = 20
look_from = [-2.0, 2.0, 8.0]
look_at = [-2.0, 2.0, 0.0]

[environment]
color = [1.0, 1.0, 1.0]

[[objects]]
shape = "wavefront"
file = "normals.obj"
group = "Smooth"
{material}
"#
    );
    render_scene(&source, |_| {})
}

/// The pixels within `radius` of the middle of the frame, which all land on
/// the sphere nearly head on.
fn middle(width: usize, pixels: &[[f32; 3]], radius: usize) -> Vec<[f32; 3]> {
    let centre = width / 2;
    (centre - radius..centre + radius)
        .flat_map(|y| (centre - radius..centre + radius).map(move |x| (x, y)))
        .map(|(x, y)| pixels[y * width + x])
        .collect()
}

fn conductor(roughness: f32) -> String {
    format!(
        "material = \"principled\"\nbase_color = [1.0, 1.0, 1.0]\nmetallic = 1.0\n\
         roughness = {roughness}"
    )
}

fn glass(roughness: f32) -> String {
    format!(
        "material = \"principled\"\nbase_color = [1.0, 1.0, 1.0]\ntransmission = 1.0\n\
         roughness = {roughness}"
    )
}

/// A white furnace: a smooth white conductor under a uniform white sky
/// reflects the sky back exactly, and cannot be told apart from it.
#[test]
fn a_smooth_white_conductor_disappears_in_a_white_furnace() {
    let Some((_, pixels)) = furnace(&conductor(0.0)) else {
        return;
    };

    let channels = || pixels.iter().flatten();
    let mean = channels().sum::<f32>() / (pixels.len() * 3) as f32;
    let brightest = channels().fold(0.0f32, |a, &b| a.max(b));

    assert!(brightest <= 1.0 + 1e-4, "gained energy: {brightest}");
    // Silhouette pixels can lose a path to the bounce limit, where an
    // interpolated normal sends a reflection back into the mesh; nothing else
    // may lose anything.
    assert!(mean > 0.99, "mean {mean}");
}

/// The same furnace, rough. It is darker, by the directional albedo the
/// sampler test above measures, and never brighter than the sky.
#[test]
fn a_rough_white_conductor_darkens_by_its_albedo_in_a_white_furnace() {
    let Some((width, pixels)) = furnace(&conductor(1.0)) else {
        return;
    };

    let centre = middle(width, &pixels, 2);
    let mean = centre.iter().flatten().sum::<f32>() / (centre.len() * 3) as f32;

    // Head on, give or take the few degrees the middle pixels span.
    let expected = reference_albedo(1.0, 1.0) as f32;
    assert!(
        (mean - expected).abs() < 0.03,
        "middle {mean}, albedo {expected}"
    );

    for pixel in &pixels {
        for &channel in pixel {
            assert!(channel <= 1.0 + 0.1, "gained energy: {pixel:?}");
        }
    }
}

/// Smooth clear glass under a uniform white sky scatters every path it
/// refracts or reflects back out to that same sky, and disappears.
#[test]
fn smooth_white_glass_disappears_in_a_white_furnace() {
    let Some((_, pixels)) = furnace(&glass(0.0)) else {
        return;
    };

    let channels = || pixels.iter().flatten();
    let mean = channels().sum::<f32>() / (pixels.len() * 3) as f32;
    let brightest = channels().fold(0.0f32, |a, &b| a.max(b));

    assert!(brightest <= 1.0 + 1e-4, "gained energy: {brightest}");
    assert!(mean > 0.99, "mean {mean}");
}

/// Frosted, it loses what single scattering loses on the way in and on the
/// way out, and still never outshines the sky.
#[test]
fn rough_white_glass_darkens_but_never_brightens_in_a_white_furnace() {
    let Some((width, pixels)) = furnace(&glass(0.5)) else {
        return;
    };

    let centre = middle(width, &pixels, 2);
    let mean = centre.iter().flatten().sum::<f32>() / (centre.len() * 3) as f32;
    assert!(
        mean < 0.99,
        "middle {mean}: rough glass should lose some light"
    );
    assert!(mean > 0.5, "middle {mean}: rough glass lost too much light");

    for pixel in &pixels {
        for &channel in pixel {
            assert!(channel <= 1.0 + 0.1, "gained energy: {pixel:?}");
        }
    }
}

/// A principled surface that absorbs everything it stops — black, and at an
/// index of one so there is no specular coat to reflect the sky — with the
/// given coverage, as the lines of a scene file.
fn cutout(alpha: f32) -> String {
    format!("material = \"principled\"\nbase_color = [0.0, 0.0, 0.0]\nior = 1.0\nalpha = {alpha}")
}

/// Looking straight down through `layers` stacked black cutouts at a uniform
/// white sky, which is all there is below them.
fn coverage(alpha: f32, layers: usize) -> Option<f32> {
    let planes: String = (0..layers)
        .map(|layer| {
            format!(
                r#"
[[objects]]
shape = "wavefront"
file = "normals.obj"
group = "Plane"
{material}

[[objects.transform]]
type = "translate"
offset = [0.0, {height}, 0.0]
"#,
                material = cutout(alpha),
                height = layer as f32 * 0.5,
            )
        })
        .collect();
    let source = format!(
        r#"
[camera]
aspect_ratio = "square"
image_width = 32
samples = 256
max_bounces = 8
fov = 30
look_from = [0.0, 50.0, 0.0]
look_at = [0.0, 0.0, 0.0]
vup = [0.0, 0.0, -1.0]

[environment]
color = [1.0, 1.0, 1.0]
{planes}
"#
    );

    let (_, pixels) = render_scene(&source, |_| {})?;
    Some(pixels.iter().flatten().sum::<f32>() / (pixels.len() * 3) as f32)
}

/// Alpha is coverage: a black surface at one half lets half of the sky through,
/// two of them a quarter, and at the two ends of the range it is either not
/// there at all or entirely opaque.
#[test]
fn a_cutout_lets_through_the_share_it_does_not_cover() {
    for (alpha, layers, expected) in [
        (0.5, 1, 0.5),
        (0.5, 2, 0.25),
        (0.2, 1, 0.8),
        (0.0, 3, 1.0),
        (1.0, 1, 0.0),
    ] {
        let Some(mean) = coverage(alpha, layers) else {
            return;
        };
        // 256 samples of 1024 pixels puts one standard deviation of a coin
        // flip at a thousandth.
        assert!(
            (mean - expected).abs() < 0.01,
            "{layers} at alpha {alpha}: {mean}, expected {expected}"
        );
    }
}

/// So many cutouts in a row that a path gives up on them, rather than walking
/// through every one: `MAX_TRANSPARENT` is 32, so a stack of 40 invisible
/// surfaces is darker than the sky behind it, and a stack of 30 is not.
#[test]
fn a_path_gives_up_after_too_many_cutouts() {
    let Some(below) = coverage(0.0, 30) else {
        return;
    };
    let Some(above) = coverage(0.0, 40) else {
        return;
    };

    assert!((below - 1.0).abs() < 1e-4, "30 layers: {below}");
    assert!(above < 1e-4, "40 layers: {above}");
}

/// A floor under a small emitter, with `layers` black cutouts of `alpha` hung
/// between them wide enough to cover the whole emitter from the middle of the
/// floor — or no cutout at all, for `None`. Seen from below the cutouts, so the
/// camera never looks through them. `lights` false renders with next event
/// estimation off, leaving scattering alone to find the emitter.
///
/// The mean radiance of the middle of the frame.
fn shadow(alpha: Option<f32>, lights: bool, layers: usize) -> Option<f32> {
    let blocker = match alpha {
        Some(alpha) => (0..layers)
            .map(|layer| {
                format!(
                    r#"
# Blocker: half-width two, a unit below the emitter. Stacked ones are spaced
# so that each is a surface of its own rather than a coincident face.
[[objects]]
shape = "wavefront"
file = "normals.obj"
group = "Plane"
{material}

[[objects.transform]]
type = "scale"
scalar = [0.04, 1.0, 0.04]

[[objects.transform]]
type = "translate"
offset = [0.0, {height}, 0.0]
"#,
                    material = cutout(alpha),
                    height = 2.5 + 0.01 * layer as f32,
                )
            })
            .collect(),
        None => String::new(),
    };
    let source = format!(
        r#"
[camera]
aspect_ratio = "square"
image_width = 32
samples = 2048
max_bounces = 4
fov = 30
look_from = [0.0, 1.5, 5.0]
look_at = [0.0, 0.0, 0.0]

[[objects]]
shape = "wavefront"
file = "normals.obj"
group = "Plane"
material = "lambertian"
albedo = [0.5, 0.5, 0.5]

# Emitter: half-width one, turned to face the floor four units below.
[[objects]]
shape = "wavefront"
file = "normals.obj"
group = "Plane"
material = "light"
emit = [40.0, 40.0, 40.0]

[[objects.transform]]
type = "scale"
scalar = [0.02, 1.0, 0.02]

[[objects.transform]]
type = "rotate"
axis = "x"
degrees = 180.0

[[objects.transform]]
type = "translate"
offset = [0.0, 4.0, 0.0]
{blocker}
"#
    );

    let (width, pixels) = render_scene(&source, |scene| {
        if !lights {
            scene.lights.clear();
            scene.light_power = 0.0;
        }
    })?;
    let centre = middle(width, &pixels, 4);
    Some(centre.iter().flatten().sum::<f32>() / (centre.len() * 3) as f32)
}

/// A shadow ray is stopped by a cutout exactly as often as a scattered ray
/// is, so the floor under a half-covering blocker gets half its light whether
/// next event estimation finds the emitter or scattering does — and a blocker
/// at alpha zero is no blocker at all.
#[test]
fn light_sampling_and_scattering_agree_through_a_cutout() {
    let Some(unshadowed) = shadow(None, true, 1) else {
        return;
    };
    let Some(transparent) = shadow(Some(0.0), true, 1) else {
        return;
    };
    let Some(sampled) = shadow(Some(0.5), true, 1) else {
        return;
    };
    let Some(scattered) = shadow(Some(0.5), false, 1) else {
        return;
    };
    let Some(opaque) = shadow(Some(1.0), true, 1) else {
        return;
    };

    let context = format!(
        "unshadowed {unshadowed}, alpha 0 {transparent}, alpha 0.5 with light sampling \
         {sampled} and without {scattered}, alpha 1 {opaque}"
    );
    assert!(unshadowed > 0.1, "the floor should be lit: {context}");
    assert!(opaque < 1e-3 * unshadowed, "{context}");
    assert!(
        (transparent - unshadowed).abs() < 0.02 * unshadowed,
        "alpha 0 should be no blocker: {context}"
    );
    assert!(
        (sampled - 0.5 * unshadowed).abs() < 0.03 * unshadowed,
        "light sampling: {context}"
    );
    // Scattering alone finds the emitter a few percent of the time, so it is
    // the noisier of the two by far.
    assert!(
        (scattered - sampled).abs() < 0.06 * sampled,
        "scattering: {context}"
    );
}

/// The two strategies agree at `MAX_TRANSPARENT` as well as under it. A path
/// that has crossed 32 cutouts is given up on and brings nothing back, so a
/// shadow ray through a stack of 40 invisible blockers has to count as blocked
/// too — otherwise next event estimation lights a floor that scattering finds
/// dark. Thirty of them are under the bound and block nothing either way.
#[test]
fn a_shadow_ray_gives_up_after_too_many_cutouts() {
    let Some(unshadowed) = shadow(None, true, 1) else {
        return;
    };
    let Some(under) = shadow(Some(0.0), true, 30) else {
        return;
    };
    let Some(over_sampled) = shadow(Some(0.0), true, 40) else {
        return;
    };
    let Some(over_scattered) = shadow(Some(0.0), false, 40) else {
        return;
    };

    let context = format!(
        "unshadowed {unshadowed}, 30 layers {under}, 40 layers with light sampling \
         {over_sampled} and without {over_scattered}"
    );
    assert!(
        (under - unshadowed).abs() < 0.02 * unshadowed,
        "30 invisible layers should block nothing: {context}"
    );
    assert!(
        over_scattered < 1e-3 * unshadowed,
        "scattering gives up past 32: {context}"
    );
    assert!(
        over_sampled < 1e-3 * unshadowed,
        "so does the shadow ray: {context}"
    );
}
