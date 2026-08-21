const FLT_MAX: f32 = 3.40282346638528859812e+38;

// How far off a surface a scattered ray starts, and the nearest hit a ray will
// accept. Both are the same number for a reason: rays carry unit directions, so
// `t` is in world units at every bounce and one epsilon covers self-intersection
// everywhere.
const EPSILON: f32 = 1e-3;
const TWO_PI: f32 = 6.2831853;

// WGSL has no dynamic allocation, so BVH traversal carries a fixed-size stack.
// A root-to-leaf descent pushes at most one node per level, and the builder caps
// depth below this, so it is a real bound rather than a hope. Keeping it small
// matters: this is per-thread storage, and 64 threads share a workgroup.
const MAX_BVH_STACK: u32 = 32u;

// Bounces a path is allowed before Russian roulette starts deciding whether it
// lives. The first few carry most of the image and killing them there would only
// trade traversal for noise; past that a path is usually deep in an interreflection
// and worth very little, and the survivors it is folded into cost nothing extra.
const MIN_ROULETTE_BOUNCE: u32 = 4u;

// `Material.kind`, matching the constants in `scene/material.rs`.
const LAMBERTIAN: u32 = 0u;
const METAL: u32 = 1u;
const DIELECTRIC: u32 = 2u;
const LIGHT: u32 = 3u;

// Camera and frame settings, packed by `GpuCamera` on the host.
struct Camera {
    origin: vec3f,
    // Vertical field of view, in radians.
    fov: f32,
    // Right, up, forward. Orthonormal, resolved on the host.
    u: vec3f,
    // Radius of the lens disk rays start from; zero is a pinhole.
    defocus_radius: f32,
    v: vec3f,
    focus_distance: f32,
    w: vec3f,
    max_bounces: u32,
    background: vec3f,
    // 1-based index of the sample being traced; also reseeds the RNG.
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

// One node of the bounding volume hierarchy, built on the host in `scene/bvh.rs`.
// Same packing trick as `Triangle`. Children are stored adjacently, so the left
// index locates both.
struct BvhNode {
    min: vec3f,
    // Interior: index of the left child. Leaf: index of its first triangle.
    left_or_first: u32,
    max: vec3f,
    // Zero marks an interior node.
    primitive_count: u32,
}

// Group 0 changes between passes, group 1 is the scene and never changes. They
// are split so the per-sample loop can rewrite the frame's uniform without
// touching the megabytes of geometry behind it.
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read_write> accum: array<vec4f>;

@group(1) @binding(0) var<storage> materials: array<Material>;
@group(1) @binding(1) var<storage> triangles: array<Triangle>;
@group(1) @binding(2) var<storage> bvh: array<BvhNode>;

// Every thread draws from its own stream, seeded from where and when it is, so
// two pixels never walk the same path and a pixel never repeats one across
// samples.
var<private> rng_state: u32;

fn init_rng(pixel: vec2u) {
    let seed = (pixel.x + pixel.y * camera.width) ^ jenkins_hash(camera.sample);
    rng_state = jenkins_hash(seed);
}

// A slightly modified version of the "One-at-a-Time Hash" function by Bob
// Jenkins. See https://www.burtleburtle.net/bob/hash/doobs.html
fn jenkins_hash(i: u32) -> u32 {
    var x = i;
    x += x << 10u;
    x ^= x >> 6u;
    x += x << 3u;
    x ^= x >> 11u;
    x += x << 15u;
    return x;
}

// The 32-bit "xor" function from Marsaglia G., "Xorshift RNGs", Section 3.
fn xorshift32() -> u32 {
    var x = rng_state;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    rng_state = x;
    return x;
}

// A random float in [0, 1). This sets the floating point exponent to zero and
// the most significant 23 bits of a random 32-bit integer as the mantissa,
// generating a number in [1, 2) that subtraction maps down.
// See Ray Tracing Gems II, Section 14.3.4.
fn rand_f32() -> f32 {
    return bitcast<f32>(0x3f800000u | (xorshift32() >> 9u)) - 1.0;
}

// Uniformly sample the surface of a unit sphere centered at the origin.
fn sample_sphere() -> vec3f {
    // Map to [-1, 1], then take the radius of that slice from Pythagoras.
    let y = 1.0 - 2.0 * rand_f32();
    let radius = sqrt(max(1.0 - y * y, 0.0));
    let phi = TWO_PI * rand_f32();

    return vec3f(radius * cos(phi), y, radius * sin(phi));
}

// Uniformly sample a unit disk. Rejection sampling is the usual way, but every
// thread in a workgroup would then wait on the unluckiest one; taking the square
// root of the radius spreads the samples evenly with no branch at all.
fn sample_disk() -> vec2f {
    let radius = sqrt(rand_f32());
    let theta = TWO_PI * rand_f32();

    return vec2f(radius * cos(theta), radius * sin(theta));
}

struct Ray {
    origin: vec3f,
    direction: vec3f,
}

fn point_on_ray(ray: Ray, t: f32) -> vec3f {
    return ray.origin + t * ray.direction;
}

struct Intersection {
    // Interpolated, and always facing the ray it was found with.
    normal: vec3f,
    // Negative when nothing was hit.
    t: f32,
    material: u32,
    front_face: bool,
}

fn no_intersection() -> Intersection {
    return Intersection(vec3f(0.0), -1.0, 0u, true);
}

// Möller-Trumbore. Deliberately two-sided: a mesh here is a surface rather than
// a solid — the example teapot is open at the lid and the base — so a back face
// is a real surface and not the inside of something.
fn intersect_triangle(ray: Ray, tri: Triangle) -> Intersection {
    let e1 = tri.v1 - tri.v0;
    let e2 = tri.v2 - tri.v0;

    let p = cross(ray.direction, e2);
    let det = dot(e1, p);
    // The ray is parallel to the triangle's plane.
    if abs(det) < 1e-8 {
        return no_intersection();
    }
    let inv_det = 1.0 / det;

    let s = ray.origin - tri.v0;
    let u = dot(s, p) * inv_det;
    if u < 0.0 || u > 1.0 {
        return no_intersection();
    }

    let q = cross(s, e1);
    let v = dot(ray.direction, q) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return no_intersection();
    }

    let t = dot(e2, q) * inv_det;
    if t < EPSILON {
        return no_intersection();
    }

    // Interpolate the vertex normals across the barycentric coordinates, so a
    // faceted mesh shades smoothly. The host guaranteed all three exist.
    var normal = normalize(tri.n0 * (1.0 - u - v) + tri.n1 * u + tri.n2 * v);

    // Turn the normal to meet the ray. Which side was struck is worth keeping:
    // a dielectric needs it to know whether it is entering or leaving.
    let front_face = dot(ray.direction, normal) < 0.0;
    normal = select(-normal, normal, front_face);

    return Intersection(normal, t, tri.material, front_face);
}

// Slab test, returning the distance at which the ray enters the node, or FLT_MAX
// if it never enters within (EPSILON, closest).
//
// `inv_direction` is infinite on any axis the ray does not travel along. The
// min/max form handles that: both slab distances go to the same infinity and
// drop out of the comparison. The one ill-defined case is an origin sitting
// exactly on a slab plane of such an axis, where 0 * infinity is a NaN and the
// comparison goes whichever way the hardware takes it. That can only add or drop
// a node visit, never change a hit, because every triangle a leaf offers is
// still tested exactly.
fn hit_aabb(origin: vec3f, inv_direction: vec3f, node: BvhNode, closest: f32) -> f32 {
    let t0 = (node.min - origin) * inv_direction;
    let t1 = (node.max - origin) * inv_direction;
    let near = min(t0, t1);
    let far = max(t0, t1);
    let enter = max(max(near.x, near.y), max(near.z, EPSILON));
    let exit = min(min(far.x, far.y), min(far.z, closest));
    return select(FLT_MAX, enter, enter <= exit);
}

// Walks the hierarchy instead of the triangle list. A scan costs one triangle
// test per triangle in the scene; this costs a couple of dozen box tests and a
// handful of triangle tests, whatever the scene size.
fn intersect_scene(ray: Ray) -> Intersection {
    var closest = no_intersection();
    closest.t = FLT_MAX;

    let inv_direction = 1.0 / ray.direction;
    var stack: array<u32, MAX_BVH_STACK>;
    var stack_depth = 0u;
    var index = 0u;

    loop {
        let node = bvh[index];
        if node.primitive_count > 0u {
            // A leaf. Its triangles were permuted into a contiguous run on the
            // host, so the count and the offset are all it takes to find them.
            for (var i = 0u; i < node.primitive_count; i += 1u) {
                let hit = intersect_triangle(ray, triangles[node.left_or_first + i]);
                if hit.t > 0.0 && hit.t < closest.t {
                    closest = hit;
                }
            }
        } else {
            // Descend into whichever child the ray reaches first, keeping the
            // other for later. Front to back is most of why this is fast: it
            // shrinks `closest.t` early, and by the time the far subtree comes
            // back off the stack it is usually already behind that bound —
            // `hit_aabb` will have rejected it outright.
            var near = node.left_or_first;
            var far = node.left_or_first + 1u;
            var near_t = hit_aabb(ray.origin, inv_direction, bvh[near], closest.t);
            var far_t = hit_aabb(ray.origin, inv_direction, bvh[far], closest.t);
            if far_t < near_t {
                let closer = far;
                far = near;
                near = closer;

                let closer_t = far_t;
                far_t = near_t;
                near_t = closer_t;
            }
            if near_t < FLT_MAX {
                // The builder caps tree depth below MAX_BVH_STACK, and a
                // root-to-leaf path pushes at most one node per level, so the
                // bound check is unreachable — it is here so that a future
                // change to the cap corrupts memory nowhere.
                if far_t < FLT_MAX && stack_depth < MAX_BVH_STACK {
                    stack[stack_depth] = far;
                    stack_depth += 1u;
                }
                index = near;
                continue;
            }
        }

        // This subtree is finished; take the nearest thing still owed. Note that
        // it is not retested against the now-smaller `closest.t`, which costs an
        // occasional wasted visit and saves a box test on every pop.
        if stack_depth == 0u {
            break;
        }
        stack_depth -= 1u;
        index = stack[stack_depth];
    }

    if closest.t < FLT_MAX {
        return closest;
    }
    return no_intersection();
}

struct Scatter {
    attenuation: vec3f,
    ray: Ray,
    // False ends the path here: the surface swallowed the ray rather than
    // sending it somewhere.
    bounced: bool,
}

// The probability a dielectric reflects rather than refracts, by Schlick's
// approximation. Glass at a glancing angle turns into a mirror, and this is what
// makes it do that.
fn schlick_reflectance(cosine: f32, ratio: f32) -> f32 {
    var r0 = (1.0 - ratio) / (1.0 + ratio);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(1.0 - cosine, 5.0);
}

// Where the ray goes next, and what the surface takes out of it on the way.
//
// A light never reaches here — the path terminates on one — so this is the three
// scattering models. All of them are sampled proportionally to their own lobe,
// which is why the attenuation is the plain albedo rather than a BRDF over a
// pdf.
fn scatter(ray: Ray, hit: Intersection, material: Material) -> Scatter {
    let normal = hit.normal;

    if material.kind == METAL {
        // A perfect mirror, roughened by nudging the reflected direction around
        // inside a sphere whose radius is the roughness.
        let reflected = reflect(ray.direction, normal)
            + material.parameter * sample_sphere();
        let direction = normalize(reflected);

        // Enough roughness can push the bounce below the surface, where it would
        // otherwise travel through the object it just left. Those rays are
        // absorbed.
        return Scatter(
            material.color,
            Ray(point_on_ray(ray, hit.t), direction),
            dot(direction, normal) > 0.0,
        );
    }

    if material.kind == DIELECTRIC {
        // `parameter` is the index of refraction, quoted against the air the
        // scene is otherwise made of; leaving the surface inverts it.
        let ratio = select(material.parameter, 1.0 / material.parameter, hit.front_face);
        let cosine = min(dot(-ray.direction, normal), 1.0);
        let sine = sqrt(max(1.0 - cosine * cosine, 0.0));

        // Past the critical angle there is no refracted direction at all, and
        // below it Schlick decides how often light reflects anyway.
        let must_reflect = ratio * sine > 1.0;
        let direction = select(
            refract(ray.direction, normal, ratio),
            reflect(ray.direction, normal),
            must_reflect || schlick_reflectance(cosine, ratio) > rand_f32(),
        );

        return Scatter(
            material.color,
            Ray(point_on_ray(ray, hit.t), normalize(direction)),
            true,
        );
    }

    // Lambertian: a cosine-weighted direction about the normal, drawn as the
    // normal plus a point on the unit sphere. Shrinking that sphere a hair keeps
    // the sum away from zero, so the result is always a direction.
    let direction = normal + sample_sphere() * (1.0 - EPSILON);

    return Scatter(
        material.color,
        Ray(point_on_ray(ray, hit.t), normalize(direction)),
        true,
    );
}

// The ray through `pixel` for this sample: jittered inside the pixel so that
// samples across passes anti-alias the edges, and started from a point on the
// lens so that everything off the focus plane blurs.
fn primary_ray(pixel: vec2u) -> Ray {
    let resolution = vec2f(f32(camera.width), f32(camera.height));
    let uv = (vec2f(pixel) + vec2f(rand_f32(), rand_f32())) / resolution;

    // To [-1, 1], with +y up rather than down the image.
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    // The viewport sits on the plane of perfect focus, so a ray aimed at a point
    // on it lands there no matter where on the lens it started.
    let half_height = tan(camera.fov * 0.5) * camera.focus_distance;
    let half_width = half_height * resolution.x / resolution.y;
    let focus_point = camera.origin
        + camera.w * camera.focus_distance
        + camera.u * ndc.x * half_width
        + camera.v * ndc.y * half_height;

    var origin = camera.origin;
    if camera.defocus_radius > 0.0 {
        let lens = sample_disk() * camera.defocus_radius;
        origin += camera.u * lens.x + camera.v * lens.y;
    }

    return Ray(origin, normalize(focus_point - origin));
}

// Follows one path from the camera until it reaches a light, escapes the scene,
// is absorbed, or runs out of bounces, and returns the radiance it carried back.
fn trace_path(primary: Ray) -> vec3f {
    var ray = primary;
    var radiance = vec3f(0.0);
    var throughput = vec3f(1.0);

    for (var bounce = 0u; bounce <= camera.max_bounces; bounce += 1u) {
        let hit = intersect_scene(ray);
        if hit.t < 0.0 {
            // Nothing was hit, so the path escapes and the background lights it.
            return radiance + throughput * camera.background;
        }

        let material = materials[hit.material];
        if material.kind == LIGHT {
            // An emitter is where a path ends: it is the only thing in the scene
            // that adds rather than attenuates.
            return radiance + throughput * material.color;
        }

        let scattered = scatter(ray, hit, material);
        if !scattered.bounced {
            break;
        }

        throughput *= scattered.attenuation;
        ray = scattered.ray;

        // Russian roulette. Past the first few bounces a path is killed with
        // probability `1 - survival` and the survivors are divided by
        // `survival`, so what the path carries is unchanged in expectation —
        // the estimator stays unbiased, it just spends its bounces on the paths
        // that still matter.
        //
        // The brightest channel is the survival probability: a throughput of
        // 0.73 across the board survives 73% of the time, which turns melee's
        // 64-bounce paths into about eight. Anything at or above one always
        // survives and is never scaled, which is what keeps a dielectric — the
        // one material that attenuates by nothing — walking the full path it
        // needs to.
        let survival = min(max(throughput.r, max(throughput.g, throughput.b)), 1.0);
        if bounce >= MIN_ROULETTE_BOUNCE {
            // A survival of zero terminates here: `rand_f32` returns [0, 1), so
            // nothing is ever below it, and the division is never reached.
            if rand_f32() >= survival {
                break;
            }
            throughput /= survival;
        } else if survival <= 0.0 {
            // Before roulette takes over, a path that can no longer contribute
            // still is not worth bouncing further.
            break;
        }
    }

    // Out of bounces, or absorbed: whatever it had gathered is all it gets.
    return radiance;
}

@compute @workgroup_size(8, 8, 1)
fn trace(@builtin(global_invocation_id) id: vec3u) {
    // The dispatch is rounded up to whole workgroups, so the last row and
    // column of threads can fall outside the image.
    if id.x >= camera.width || id.y >= camera.height {
        return;
    }

    let pixel = id.xy;
    init_rng(pixel);
    let radiance = trace_path(primary_ray(pixel));

    // Radiance sums in `rgb`, samples in `w`, so the host can average without
    // being told how many passes ran.
    let index = pixel.y * camera.width + pixel.x;
    accum[index] = accum[index] + vec4f(radiance, 1.0);
}
