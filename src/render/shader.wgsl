const FLT_MAX: f32 = 3.40282346638528859812e+38;

// How far off a surface a scattered ray starts, and the nearest hit a ray will
// accept. Both are the same number for a reason: rays carry unit directions, so
// `t` is in world units at every bounce and one epsilon covers self-intersection
// everywhere.
const EPSILON: f32 = 1e-3;
const PI: f32 = 3.1415927;
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
    // Entries in `lights`. Zero means the scene has no emitters, and the whole
    // of next event estimation is skipped.
    light_count: u32,
    // What those entries add up to, and what turns one entry's share of the
    // table back into a probability.
    light_power: f32,
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
// One emissive triangle, as an entry in the distribution the host built over
// them in `scene/light.rs`.
struct Light {
    // Index into `triangles`.
    triangle: u32,
    // The chance of drawing this entry or any before it. Entries ascend and the
    // last is exactly one.
    cdf: f32,
}

// The emitters, in proportion to the light they put out. Sampling searches this
// rather than the scene: an emitter that is large or bright is worth more shadow
// rays than one that is small or dim, and the table is what says by how much.
@group(1) @binding(3) var<storage> lights: array<Light>;

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
    // Index into `triangles`. Only [`traverse`] can fill this in — a triangle
    // does not know where it is stored — and light sampling needs it to recover
    // the emitter's area from a hit.
    triangle: u32,
}

fn no_intersection() -> Intersection {
    return Intersection(vec3f(0.0), -1.0, 0u, true, 0u);
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

    return Intersection(normal, t, tri.material, front_face, 0u);
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
//
// Nothing beyond `limit` is considered, which is what lets a shadow ray stop at
// the light it is aimed at. `any_hit` returns the first thing found rather than
// the nearest: a shadow ray only asks whether the path is blocked, so it can
// abandon the tree the moment anything says yes.
fn traverse(ray: Ray, limit: f32, any_hit: bool) -> Intersection {
    var closest = no_intersection();
    closest.t = limit;

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
                let primitive = node.left_or_first + i;
                let hit = intersect_triangle(ray, triangles[primitive]);
                if hit.t > 0.0 && hit.t < closest.t {
                    closest = hit;
                    closest.triangle = primitive;
                    if any_hit {
                        return closest;
                    }
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

    if closest.t < limit {
        return closest;
    }
    return no_intersection();
}

// The nearest thing the ray meets anywhere in the scene.
fn intersect_scene(ray: Ray) -> Intersection {
    return traverse(ray, FLT_MAX, false);
}

// Whether anything at all stands between `origin` and a point `distance` away
// along `direction`. The limit stops an epsilon short so that the emitter being
// aimed at does not shadow itself.
fn occluded(origin: vec3f, direction: vec3f, distance: f32) -> bool {
    return traverse(Ray(origin, direction), distance - EPSILON, true).t > 0.0;
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

// Veach's power heuristic with an exponent of two, weighing a sample drawn at
// density `chosen` against the density `other` some rival strategy would have
// drawn it at. The two weights sum to one for any pair, so combining the
// strategies neither loses light nor counts it twice; squaring is what makes the
// strategy that was confident about a sample keep most of it.
fn power_heuristic(chosen: f32, other: f32) -> f32 {
    let a = chosen * chosen;
    let b = other * other;
    return a / max(a + b, 1e-8);
}

// Rec. 709 luma, matching what `scene/light.rs` weighed the table by. The two
// have to be the same function: one decides how often a triangle is drawn and
// the other divides by how often it was.
fn luminance(color: vec3f) -> f32 {
    return dot(color, vec3f(0.2126, 0.7152, 0.0722));
}

// The density with which [`sample_light`] draws a direction, per unit solid
// angle, given the emitter it lands on and how that emitter is turned.
//
// The triangle's own area is missing, and not by omission. An entry is drawn in
// proportion to its area times its brightness, and a point is then drawn per
// unit of that same area, so the two cancel and what is left is one emitter's
// brightness against everything the scene emits. That cancellation is why
// nothing here has to find which entry of the table a triangle belongs to: a
// direction can be priced from the triangle alone, which is what makes weighing
// a scattered ray that happened to land on an emitter cost a lookup rather than
// a search.
fn light_density(emit: vec3f, distance: f32, cosine: f32) -> f32 {
    if cosine <= 0.0 || camera.light_power <= 0.0 {
        return 0.0;
    }

    return (distance * distance) * luminance(emit) / (cosine * camera.light_power);
}

// The density with which [`sample_light`] would have produced a direction, given
// that following it landed on `hit`.
//
// Only the emitter a ray reaches first matters: anything behind it is occluded,
// so a shadow ray aimed there returns nothing and that draw contributes zero.
fn light_pdf(hit: Intersection, direction: vec3f) -> f32 {
    let tri = triangles[hit.triangle];
    let normal = cross(tri.v1 - tri.v0, tri.v2 - tri.v0);
    // Emitters are two-sided, so a face turned away still emits toward whatever
    // is looking at its back.
    let cosine = abs(dot(normalize(normal), direction));

    // Directions are unit, so `hit.t` is the distance to the emitter.
    return light_density(materials[hit.material].color, hit.t, cosine);
}

// The entry whose slice of the table a uniform draw lands in: the first whose
// cumulative chance has overtaken it.
//
// A dozen divergent iterations for melee's sphere, against the couple of hundred
// a traversal costs — the search is not what this pays for. Every entry is worth
// drawing, which is the point of leaving the ones that emit nothing out of the
// table rather than giving them a slice of zero to be searched past.
fn select_light(draw: f32) -> Light {
    var low = 0u;
    var high = camera.light_count - 1u;

    loop {
        if low >= high {
            break;
        }

        let middle = (low + high) / 2u;
        if lights[middle].cdf > draw {
            high = middle;
        } else {
            low = middle + 1u;
        }
    }

    return lights[low];
}

// One point on one emitter, and what the estimator needs to weigh it.
struct LightSample {
    // From the shading point toward the light, unit length.
    direction: vec3f,
    // How far the light is, so the shadow ray knows where to stop.
    distance: f32,
    // What the emitter sends back along `direction`.
    radiance: vec3f,
    // Solid angle density of having drawn this direction. Zero is unusable.
    pdf: f32,
}

// Draws a point on an emitter: a triangle out of the table, then a point
// uniformly on that triangle.
fn sample_light(origin: vec3f) -> LightSample {
    let tri = triangles[select_light(rand_f32()).triangle];

    let e1 = tri.v1 - tri.v0;
    let e2 = tri.v2 - tri.v0;

    // Uniform over the triangle. The square folds along its diagonal rather than
    // rejecting the far half, for the same reason `sample_disk` does not reject:
    // a retry would stall every other thread in the workgroup.
    var u = rand_f32();
    var v = rand_f32();
    if u + v > 1.0 {
        u = 1.0 - u;
        v = 1.0 - v;
    }
    let point = tri.v0 + e1 * u + e2 * v;

    let offset = point - origin;
    let distance = length(offset);
    let direction = offset / max(distance, 1e-8);

    // The geometric normal, not the interpolated one: the density is written
    // against the surface that was actually sampled, and a shading normal is not
    // that surface. Emitters are two-sided like everything else, so only how
    // foreshortened the face is matters.
    let normal = cross(e1, e2);
    let cosine = abs(dot(normalize(normal), direction));

    let emit = materials[tri.material].color;
    var pdf = 0.0;
    if distance > 0.0 {
        pdf = light_density(emit, distance, cosine);
    }

    return LightSample(direction, distance, emit, pdf);
}

// Next event estimation: the light this surface receives directly, found by
// aiming at an emitter instead of waiting for a scattered ray to wander into
// one.
//
// This is what removes fireflies. A scattered ray finds a small bright light by
// luck, so one sample in a few hundred comes back carrying all of it and the
// pixel keeps that dot until the average catches up. A shadow ray finds it every
// time, and what it brings back varies only with visibility.
//
// Lambertian only. The other two models are sampled from lobes narrow enough
// that a random direction already lands near the light — and neither has a
// density this could weigh a light sample against, since both are sampled by
// construction rather than from a distribution anyone wrote down.
//
// The caller has to have checked that there is a light to sample: with none,
// `sample_light` has nothing to pick from.
fn direct_light(point: vec3f, normal: vec3f, albedo: vec3f) -> vec3f {
    let light = sample_light(point);

    // Behind the surface, or a degenerate emitter.
    let cosine = dot(normal, light.direction);
    if light.pdf <= 0.0 || cosine <= 0.0 {
        return vec3f(0.0);
    }

    if occluded(point, light.direction, light.distance) {
        return vec3f(0.0);
    }

    // Scattering would have found this same direction with this density, and
    // the heuristic splits the contribution between the two strategies. Near a
    // large emitter scattering is the better one and keeps most of it; for the
    // small bright light that made the fireflies, this sample keeps nearly all.
    let weight = power_heuristic(light.pdf, cosine / PI);

    // The lambertian brdf is `albedo / PI`, and the estimator is `f * cos / pdf`.
    return light.radiance * albedo * cosine * weight / (PI * light.pdf);
}

// Follows one path from the camera until it reaches a light, escapes the scene,
// is absorbed, or runs out of bounces, and returns the radiance it carried back.
fn trace_path(primary: Ray) -> vec3f {
    var ray = primary;
    var radiance = vec3f(0.0);
    var throughput = vec3f(1.0);

    // How the current direction was chosen, which is what decides how much of an
    // emitter this ray is allowed to keep. A camera ray counts as specular:
    // nothing sampled it, so no other strategy could have found it and there is
    // nothing to share the light with.
    var scatter_pdf = 0.0;
    var specular = true;

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
            //
            // The previous vertex already sampled this emitter directly, so
            // taking all of what the scattered ray found would count it twice.
            // The heuristic hands over only the share this strategy earned, and
            // `direct_light` took the rest. A specular bounce, or the camera,
            // had no direct sample to double, and keeps everything.
            var weight = 1.0;
            if !specular {
                weight = power_heuristic(scatter_pdf, light_pdf(hit, ray.direction));
            }
            return radiance + throughput * material.color * weight;
        }

        // Light arriving straight from an emitter, gathered before the path
        // wanders off to find whatever else this surface can see.
        if material.kind == LAMBERTIAN && camera.light_count > 0u {
            let point = point_on_ray(ray, hit.t);
            radiance += throughput * direct_light(point, hit.normal, material.color);
        }

        let scattered = scatter(ray, hit, material);
        if !scattered.bounced {
            break;
        }

        // A cosine-weighted direction was drawn at `cos / PI`; the other two
        // models did not draw from anything an emitter can be weighed against.
        specular = material.kind != LAMBERTIAN;
        scatter_pdf = dot(scattered.ray.direction, hit.normal) / PI;

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
