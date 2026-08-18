// Camera and frame settings, packed by `GpuCamera` on the host.
struct Camera {
    origin: vec3f,
    // Vertical field of view, in radians.
    fov: f32,
    // Right, up, forward. Orthonormal, resolved on the host.
    u: vec3f,
    defocus_angle: f32,
    v: vec3f,
    focus_distance: f32,
    w: vec3f,
    max_bounces: u32,
    background: vec3f,
    // 1-based index of the sample being traced; will also reseed the RNG.
    sample: u32,
    width: u32,
    height: u32,
}

// One surface, packed by `GpuMaterial`. `kind` selects what the other two mean:
// 0 lambertian, 1 metal, 2 dielectric, 3 light.
struct Material {
    color: vec3f,
    kind: u32,
    parameter: f32,
}

// One triangle in world space, packed by `GpuTriangle`. A `vec3f` is 12 bytes
// but 16-aligned, so a scalar placed right after one costs nothing — that is
// where the material index rides.
struct Triangle {
    v0: vec3f,
    material: u32,
    v1: vec3f,
    _pad1: f32,
    v2: vec3f,
    _pad2: f32,
    n0: vec3f,
    _pad3: f32,
    n1: vec3f,
    _pad4: f32,
    n2: vec3f,
    _pad5: f32,
}

// Group 0 changes between passes, group 1 is the scene and never changes. They
// are split so a future per-sample loop can rebind the frame without touching
// the megabytes of geometry behind it.
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read_write> accum: array<vec4f>;

@group(1) @binding(0) var<storage> materials: array<Material>;
@group(1) @binding(1) var<storage> triangles: array<Triangle>;

struct Ray {
    origin: vec3f,
    direction: vec3f,
}

// The ray through the center of `pixel`, from the camera's basis and field of
// view. Sub-pixel jitter belongs here once there is an RNG to draw it from;
// until then every sample of a pixel would trace the same path, which is why
// the host only runs one.
fn primary_ray(pixel: vec2u) -> Ray {
    let resolution = vec2f(f32(camera.width), f32(camera.height));
    let uv = (vec2f(pixel) + 0.5) / resolution;

    // To [-1, 1], with +y up rather than down the image.
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let half_height = tan(camera.fov * 0.5);
    let aspect = resolution.x / resolution.y;

    let direction = camera.w
        + camera.u * ndc.x * half_height * aspect
        + camera.v * ndc.y * half_height;

    return Ray(camera.origin, normalize(direction));
}

// Stands in for the tracer: no triangle is tested, so every ray reaches the
// background. The gradient runs from the scene's configured background color at
// the bottom of the view to a pale blue overhead, which is enough to tell that
// the camera is pointed where the config says it is — turn the camera and the
// horizon turns with it.
fn shade(ray: Ray) -> vec3f {
    let height = clamp(0.5 * (ray.direction.y + 1.0), 0.0, 1.0);
    let sky = mix(vec3f(1.0), vec3f(0.5, 0.7, 1.0), height);

    return mix(camera.background, sky, height);
}

// A readout across the bottom of the image, drawn straight from the scene
// buffers: one swatch per material in config order, and above them a bar whose
// length grows with the triangle count. It exists to prove the geometry and the
// materials made it to the GPU intact while there is nothing yet that draws
// them, and it goes away with the tracer.
fn readout(pixel: vec2u) -> vec3f {
    let resolution = vec2f(f32(camera.width), f32(camera.height));
    let position = vec2f(pixel) / resolution;

    let count = arrayLength(&materials);
    if position.y > 0.94 && count > 0u {
        let swatch = min(u32(position.x * f32(count)), count - 1u);
        return materials[swatch].color;
    }

    // Logarithmic, because a scene is as likely to hold a dozen triangles as a
    // million and both should be visible as something other than empty or full.
    let filled = log2(f32(arrayLength(&triangles)) + 1.0) / 24.0;
    if position.y > 0.92 && position.x < filled {
        return vec3f(0.85);
    }

    return vec3f(-1.0);
}

@compute @workgroup_size(8, 8, 1)
fn trace(@builtin(global_invocation_id) id: vec3u) {
    // The dispatch is rounded up to whole workgroups, so the last row and
    // column of threads can fall outside the image.
    if id.x >= camera.width || id.y >= camera.height {
        return;
    }

    let pixel = id.xy;
    var color = readout(pixel);
    if color.r < 0.0 {
        color = shade(primary_ray(pixel));
    }

    // Radiance sums in `rgb`, samples in `w`, so the host can average without
    // being told how many passes ran.
    let index = pixel.y * camera.width + pixel.x;
    accum[index] = accum[index] + vec4f(color, 1.0);
}
