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

// Surfaces a path may pass through by their alpha before it is given up on, as
// Cycles' `transparent_max_bounces`. Kept apart from `max_bounces` because a
// pass-through scatters nothing: a stack of cutout leaves should not eat the
// bounces the light behind it needs.
const MAX_TRANSPARENT: u32 = 32u;

// `Material.kind`, matching the constants in `scene/material.rs`.
const PRINCIPLED: u32 = 0u;

// GGX alpha below which a microfacet lobe is treated as a perfect mirror. Past
// here the distribution is a spike whose density no longer fits in a float, and
// the heuristic would be dividing one infinity by another.
const DELTA_ALPHA: f32 = 1e-3;

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
    // Yaw applied to an escaped ray before the environment lookup, in radians,
    // and the scale on what that lookup returns.
    environment_rotation: f32,
    environment_intensity: f32,
    // Standard deviations above a pixel's running mean a sample may reach before
    // it is scaled back. Zero turns outlier rejection off.
    outlier_k: f32,
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
    // The side of the square grid a pixel's samples are stratified over. One is
    // an unstratified jitter.
    strata: u32,
    // Size of the sky's sampling distribution, in cells. Zero width means the
    // scene has no map worth aiming at, and the whole of sky sampling is
    // skipped — the same signal `light_count` gives for the triangle half.
    sky_width: u32,
    sky_height: u32,
    // Samples a pixel needs before `outlier_k` is allowed to act on it.
    outlier_warmup: u32,
    // Decorrelates one render of a scene from another. Zero is an identity —
    // `jenkins_hash` maps zero to zero — and is what every render draws unless
    // something asked for a second, independent one to measure against.
    seed: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// One surface, packed by `GpuMaterial`. `kind` selects which fields are read,
// and a principled surface — the only kind there is — reads all of them. A
// light is one too: black, at an index of one, and emissive.
struct Material {
    color: vec3f,
    kind: u32,
    roughness: f32,
    metallic: f32,
    ior: f32,
    transmission: f32,
    // Coverage: the chance a ray is stopped by the surface at all.
    alpha: f32,
    // How much of the diffuse base is a random walk through the inside of the
    // surface instead. Zero for a surface light does not go into.
    subsurface_weight: f32,
    // Henyey-Greenstein asymmetry for that walk: straight back at -1, every
    // direction alike at zero, straight on at 1.
    subsurface_anisotropy: f32,
    // Radiance given off the front face, decided by the geometric normal. Zero
    // for a surface that does not emit.
    emission: vec3f,
    // Mean free path inside the surface per channel, in world units. The host
    // has already multiplied the scene's radius by its scale, and has zeroed
    // `subsurface_weight` when every channel came out at nothing.
    subsurface_radius: vec3f,
}

// One triangle in world space, packed by `GpuTriangle`. A `vec3f` is 12 bytes
// but 16-aligned, so a scalar placed right after one costs nothing — that is
// where the material index rides, and the material's alpha after it.
struct Triangle {
    v0: vec3f,
    material: u32,
    v1: vec3f,
    // A copy of the material's, so that a shadow ray can tell whether a
    // triangle stops it without looking the material up.
    alpha: f32,
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
// The first two moments of each pixel's sample luminances, `vec2f(sum, sum of
// squares)`, which is everything needed to recover a variance. A second buffer
// rather than two more channels on `accum`: this is read and written once per
// sample per pixel just as the radiance is, and luminance is what every
// consumer of it wants — a per-channel version would double that traffic to
// answer a question nobody asked.
//
// Two pairs, because two consumers want two different distributions. The
// firefly rejection has to measure every sample as it was drawn, or its window
// collapses onto whatever it already capped (see `trace`). Everything else —
// the filter's noise estimate and the host's variance figures — has to measure
// what the accumulator actually kept, or it reports noise the image no longer
// has. With the rejection off the two pairs are identical.
struct Moments {
    drawn: vec2f,
    kept: vec2f,
}
@group(0) @binding(2) var<storage, read_write> moments: array<Moments>;
// The feature buffers, summed the same way the radiance is so that a pixel
// straddling an edge averages the surfaces its samples actually found rather
// than keeping whichever one landed last. `accum`'s `w` counts for all three.
//
// Packed two to a `vec4f` because a `vec3f` in a storage array is padded to
// sixteen bytes anyway: the fourth channel is free, and putting the scalar
// there costs nothing over storing the vector alone.
@group(0) @binding(3) var<storage, read_write> normals: array<vec4f>;
// The albedo's fourth channel is the object id, and it is the one feature that
// is *not* summed: an average of two ids is a third id belonging to neither
// surface. It is a majority vote instead — Boyer-Moore, which finds whichever id
// more than half of a pixel's samples saw in one candidate and one counter:
//
//   - high 16 bits: the candidate id plus one, so a miss (-1) is zero;
//   - low 16 bits: the candidate's lead, saturating rather than wrapping.
//
// Zeroed, that reads as no candidate and no lead, so the first sample to
// arrive takes the seat. The host guards the material count that has to fit.
struct Albedo {
    color: vec3f,
    votes: u32,
}
@group(0) @binding(4) var<storage, read_write> albedos: array<Albedo>;

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

// The sky, as an equirectangular image. A scene that named no HDRI is handed a
// one-texel map holding its flat color, so this is bound either way and there is
// no second way for a path to find the background.
@group(1) @binding(4) var environment: texture_2d<f32>;
@group(1) @binding(5) var environment_sampler: sampler;

// The sky's sampling distribution, built on the host in `scene/sky.rs`: a
// marginal over the rows of the map, and one conditional over the columns of
// each row, laid out row-major behind it. Both are cumulative, so drawing from
// either is a binary search and the chance of a single cell is the gap between
// two neighbouring entries — which is why the densities are not uploaded as
// well.
@group(1) @binding(6) var<storage> sky_marginal: array<f32>;
@group(1) @binding(7) var<storage> sky_conditional: array<f32>;

// Yaw about +Y, so the map can be turned to aim its sun at the scene without
// moving the camera. `environment_direction` asks for the negative of the same
// angle to undo it.
fn rotate_y(direction: vec3f, angle: f32) -> vec3f {
    let s = sin(angle);
    let c = cos(angle);
    return vec3f(
        direction.x * c - direction.z * s,
        direction.y,
        direction.x * s + direction.z * c,
    );
}

// Where a direction lands on the environment map.
//
// Longitude around +Y goes into u and latitude from the +Y pole down into v,
// which is the layout every equirectangular image ships in. `direction` is
// already normalized — primary rays and every scattered direction both are — so
// the latitude is an `acos` and nothing more.
fn environment_uv(direction: vec3f) -> vec2f {
    let turned = rotate_y(direction, camera.environment_rotation);

    return vec2f(
        0.5 + atan2(turned.x, -turned.z) / TWO_PI,
        acos(clamp(turned.y, -1.0, 1.0)) / PI,
    );
}

// The inverse: the direction `environment_uv` maps back to this place on the
// map.
//
// The two have to be exact inverses of each other, because `sample_sky` draws a
// place on the map, comes here for a direction to aim a shadow ray along, and
// then prices that direction by going back through `environment_uv`. A round
// trip that drifted would leave the two halves of the heuristic weighing
// different directions.
fn environment_direction(uv: vec2f) -> vec3f {
    let theta = uv.y * PI;
    let phi = (uv.x - 0.5) * TWO_PI;
    let sin_theta = sin(theta);

    let turned = vec3f(sin_theta * sin(phi), cos(theta), -sin_theta * cos(phi));

    return rotate_y(turned, -camera.environment_rotation);
}

// What the map holds at a place on it.
//
// Split from [`environment_radiance`] because half the callers already know the
// uv: [`sample_sky`] draws one out of the distribution and only converts it to a
// direction to aim a shadow ray along. Handing that uv straight to the sampler
// saves the round trip back through [`environment_uv`], which is an `atan2`, an
// `acos` and a rotation.
fn environment_radiance_uv(uv: vec2f) -> vec3f {
    // `textureSampleLevel` and not `textureSample`: a compute shader has no
    // implicit derivatives to pick a mip from, and there is only the one level
    // to pick anyway.
    let texel = textureSampleLevel(environment, environment_sampler, uv, 0.0);
    return texel.rgb * camera.environment_intensity;
}

// What a ray aimed at the sky brings home.
fn environment_radiance(direction: vec3f) -> vec3f {
    return environment_radiance_uv(environment_uv(direction));
}

// Every thread draws from its own stream, seeded from where and when it is, so
// two pixels never walk the same path and a pixel never repeats one across
// samples.
var<private> rng_state: u32;

fn init_rng(pixel: vec2u) {
    // `jenkins_hash(0)` is zero — every operation in it leaves zero alone — so a
    // seed of zero XORs nothing in and this is bit for bit the stream the
    // renderer drew before the seed existed. That is what keeps a defaulted
    // render reproducible while a seeded one shares no noise with it.
    let stream = jenkins_hash(camera.sample) ^ jenkins_hash(camera.seed);
    let seed = (pixel.x + pixel.y * camera.width) ^ stream;
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

// What a path saw of the surface it landed on, for a denoiser to steer by.
// None of it is radiance: these are the noise-free facts about the geometry
// that say which neighbouring pixels are looking at the same thing.
struct Features {
    // Zero for a path that hit nothing, which is a value no real hit can take —
    // an interpolated normal is unit length — and so is the signal a consumer
    // uses to leave the background alone.
    normal: vec3f,
    // Distance from the camera in world units, rays being unit length. Summed
    // along the path rather than read off the hit: `trace_path` re-origins the
    // ray at every bounce, so a `t` from bounce one on measures that segment
    // alone, and a surface captured behind glass would otherwise file the
    // distance from the refraction rather than from the camera. Both consumers
    // — the filter's depth gradient and `depth.png` — need one common origin.
    //
    // Zero for a miss rather than the infinity that would be the natural
    // sentinel: these are summed across every sample before they are averaged,
    // and no number near the top of the range survives being added to itself
    // five hundred times. `normal` is the miss flag; this is only a distance.
    depth: f32,
    // The surface's own colour, with the lighting divided out of it. One for a
    // miss, so that dividing by it is the identity.
    albedo: vec3f,
    // Which material the surface came from — one per config object, so it
    // doubles as an object id. Negative for a miss.
    object: i32,
}

// What one traced path came home with: the light, and the surface that carried
// it. Two things rather than one because the second is not a colour and must
// never be filtered as though it were.
struct Path {
    radiance: vec3f,
    features: Features,
}

// The colour a surface is filed under. A surface that scatters nothing has no
// reflectance to speak of, and dividing its radiance by that would divide by the
// filter's floor; its emission's hue, brightest channel at one, is the colour it
// is seen as instead.
fn feature_albedo(material: Material) -> vec3f {
    let brightest = max(material.emission.r, max(material.emission.g, material.emission.b));
    if scatters(material) || brightest <= 0.0 {
        return material.color;
    }
    return material.emission / brightest;
}

fn no_features() -> Features {
    return Features(vec3f(0.0), 0.0, vec3f(1.0), -1);
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
    // glass needs it to know whether it is entering or leaving.
    let front_face = dot(ray.direction, normal) < 0.0;
    normal = select(-normal, normal, front_face);

    return Intersection(normal, t, tri.material, front_face, 0u);
}

// The same test as [`intersect_triangle`], answering only whether the triangle
// was met inside `limit`.
//
// A shadow ray does not shade anything: `occluded` wants a yes or a no, and
// every attribute the full test builds on the way to one — the interpolated
// normal, its `normalize`, the side that was struck, the material — is computed
// and thrown away. Dropping them takes the work out, and it takes the live
// state out too, which on this kernel is the half that pays.
fn hits_triangle(ray: Ray, tri: Triangle, limit: f32) -> bool {
    let e1 = tri.v1 - tri.v0;
    let e2 = tri.v2 - tri.v0;

    let p = cross(ray.direction, e2);
    let det = dot(e1, p);
    if abs(det) < 1e-8 {
        return false;
    }
    let inv_det = 1.0 / det;

    let s = ray.origin - tri.v0;
    let u = dot(s, p) * inv_det;
    if u < 0.0 || u > 1.0 {
        return false;
    }

    let q = cross(s, e1);
    let v = dot(ray.direction, q) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return false;
    }

    let t = dot(e2, q) * inv_det;
    return t >= EPSILON && t < limit;
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
// Nothing beyond `limit` is considered. Shadow rays do not come through here at
// all — [`occluded`] walks the tree itself, for the first thing rather than the
// nearest and without an intersection to fill in.
fn traverse(ray: Ray, limit: f32) -> Intersection {
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
    return traverse(ray, FLT_MAX);
}

// Whether anything at all stands between `origin` and a point `distance` away
// along `direction`. The limit stops an epsilon short so that the emitter being
// aimed at does not shadow itself.
//
// A surface with an alpha below one blocks with that chance, rolled once for
// each such triangle the ray actually crosses — the same chance a scattered ray
// in `trace_path` has of stopping on it, which is what keeps the two strategies
// agreeing about how much light gets through.
//
// Its own walk rather than a flag on [`traverse`]. The two look alike, and the
// difference is the point: this one carries a bound that never moves and no
// candidate at all, where the other carries an `Intersection` it narrows as it
// goes. Next event estimation aims a shadow ray from every diffuse bounce — two
// where a scene has both an emitter and a sky — so this is at least half the
// traversals in the render, and the cheapest possible version of it is worth
// the duplicated loop.
fn occluded(origin: vec3f, direction: vec3f, distance: f32) -> bool {
    let ray = Ray(origin, direction);
    // Nothing found here shrinks anything, so the bound handed to every box test
    // is the same one from the first node to the last.
    let limit = distance - EPSILON;

    let inv_direction = 1.0 / ray.direction;
    var stack: array<u32, MAX_BVH_STACK>;
    var stack_depth = 0u;
    var index = 0u;

    // Cutouts survived, against the same `MAX_TRANSPARENT` a scattered ray is
    // held to.
    var passes = 0u;

    loop {
        let node = bvh[index];
        if node.primitive_count > 0u {
            for (var i = 0u; i < node.primitive_count; i += 1u) {
                let tri = triangles[node.left_or_first + i];
                // The roll comes after the geometric test, so the RNG is only
                // touched when a cutout is actually in the way and an opaque
                // scene draws exactly the stream it always did. Each triangle
                // lives in one leaf and a leaf is visited at most once, so
                // every blocker gets one independent roll.
                if hits_triangle(ray, tri, limit) {
                    if tri.alpha >= 1.0 || rand_f32() < tri.alpha {
                        return true;
                    }
                    // `trace_path` gives up on a path that has come through
                    // `MAX_TRANSPARENT` cutouts, and a path given up on brings
                    // nothing back — so this gives up as blocked, and the two
                    // strategies go on agreeing at the bound as well as under
                    // it. It also caps what one shadow ray can draw.
                    passes += 1u;
                    if passes > MAX_TRANSPARENT {
                        return true;
                    }
                }
            }
        } else {
            var near = node.left_or_first;
            var far = node.left_or_first + 1u;
            var near_t = hit_aabb(ray.origin, inv_direction, bvh[near], limit);
            var far_t = hit_aabb(ray.origin, inv_direction, bvh[far], limit);
            if far_t < near_t {
                let closer = far;
                far = near;
                near = closer;

                let closer_t = far_t;
                far_t = near_t;
                near_t = closer_t;
            }
            if near_t < FLT_MAX {
                if far_t < FLT_MAX && stack_depth < MAX_BVH_STACK {
                    stack[stack_depth] = far;
                    stack_depth += 1u;
                }
                index = near;
                continue;
            }
        }

        if stack_depth == 0u {
            break;
        }
        stack_depth -= 1u;
        index = stack[stack_depth];
    }

    return false;
}

// A right-handed orthonormal frame around a unit normal, for working on a
// surface as though it lay flat with its normal along +z.
struct Frame {
    t: vec3f,
    b: vec3f,
    n: vec3f,
}

// Duff et al., "Building an Orthonormal Basis, Revisited" (JCGT 2017).
//
// Branchless and continuous everywhere except across the z = 0 plane, where the
// sign flips — which is exactly where the older Frisvad construction divides by
// zero at n = -z instead. `select` rather than `sign`, because `sign(0.0)` is
// zero and would put that same division right back.
fn orthonormal_basis(n: vec3f) -> Frame {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;

    return Frame(
        vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x),
        vec3f(b, s + n.y * n.y * a, -n.y),
        n,
    );
}

fn to_local(frame: Frame, v: vec3f) -> vec3f {
    return vec3f(dot(v, frame.t), dot(v, frame.b), dot(v, frame.n));
}

fn to_world(frame: Frame, v: vec3f) -> vec3f {
    return frame.t * v.x + frame.b * v.y + frame.n * v.z;
}

// One direction drawn from a surface's BSDF, and what the path needs to carry
// on along it.
struct BsdfSample {
    // Unit direction the path continues along.
    wi: vec3f,
    // `f(wo, wi) * |n·wi| / pdf`: what throughput is multiplied by.
    weight: vec3f,
    // Solid-angle density of `wi`. Meaningless when `delta`.
    pdf: f32,
    // True when `wi` came from a lobe with no density (mirror, smooth glass,
    // legacy metal). The next emitter hit keeps its full contribution, and no
    // light sample was taken against this lobe.
    delta: bool,
    // True when `wi` points *into* the surface because the subsurface lobe was
    // picked. `wi` is then not where the path carries on but where it dives in:
    // `subsurface_walk` takes it from here, and the path comes back out
    // somewhere else on the same object. `pdf` and `delta` say nothing in that
    // case — the exit vertex draws its own direction and sets both.
    subsurface: bool,
    // False when the surface absorbed the ray.
    valid: bool,
}

// A BSDF priced at a direction something other than the BSDF chose — a light
// sample, most often.
struct BsdfEval {
    // `f(wo, wi) * |n·wi|`, i.e. already cosine-weighted.
    value: vec3f,
    // Solid-angle density `bsdf_sample` would have drawn `wi` with.
    pdf: f32,
}

// GGX's alpha, from the roughness a scene is written in. Squared, as Blender
// does, so that a sweep of roughness reads as an even sweep of blur; floored so
// that a roughness of zero still names a distribution rather than a division by
// zero. Anything under `DELTA_ALPHA` is sampled as a mirror anyway.
fn ggx_alpha(roughness: f32) -> f32 {
    return max(roughness * roughness, 1e-4);
}

// Whether any of the surface's lobes has weight at any angle, which is whether a
// path that reaches it can go anywhere afterwards. The angle-free form of
// `lobe_probabilities`' weights: a metal's Fresnel color is never zero at a
// glancing angle, and the dielectric reflectance is zero at every angle exactly
// when the index is one.
//
// A light is the surface that fails this — black, at an index of one, and
// neither metal nor glass — and so is any other perfect absorber.
// Subsurface needs no term of its own anywhere below: it takes its share out of
// the diffuse slot, so `opaque` and the base colour already speak for both, and
// a surface that scatters under its skin is one that scatters.
fn scatters(material: Material) -> bool {
    if material.kind != PRINCIPLED {
        return false;
    }

    let dielectric = 1.0 - material.metallic;
    let opaque = dielectric * (1.0 - material.transmission);
    return material.metallic > 0.0
        || dielectric * material.transmission > 0.0
        || (opaque > 0.0 && (material.ior != 1.0 || luminance(material.color) > 0.0));
}

// Whether the surface has any lobe a light sample could be weighed against.
// Next event estimation is wasted on a surface without one: `bsdf_eval` would
// price every light direction at zero.
//
// A principled surface always has one unless its microfacet lobes are mirrors
// and there is no diffuse underneath them to speak of — which is clear glass,
// and polished metal — or it has no lobes worth the name at all. The roughness
// of a surface that scatters nothing says nothing: a light is as rough as the
// default, and a shadow ray from one would be wasted on every hit.
fn has_smooth_lobe(material: Material) -> bool {
    if !scatters(material) {
        return false;
    }

    let rough = ggx_alpha(material.roughness) >= DELTA_ALPHA;
    // What the coat lets through, less the share of it that dives under the
    // surface: subsurface is not a lobe a light sample can be weighed against
    // here. `principled_eval` prices it at nothing and gives it no density —
    // the walk's exit aims its own shadow ray, and that vertex is where the
    // light it finds is weighed. Counting it here would send a shadow ray from
    // a surface with a mirror coat and nothing left to reflect with, and the
    // answer would be zero every time.
    //
    // The factor is exactly one where there is no subsurface, so a scene
    // without any keeps the shadow rays — and the random stream — it had.
    let diffuse = (1.0 - material.metallic) * (1.0 - material.transmission)
        * (1.0 - material.subsurface_weight) * luminance(material.color);
    return rough || diffuse > 0.0;
}

// GGX alpha above which a microfacet lobe blurs enough to count as a surface
// for the denoiser rather than a reflection of one — Cycles' threshold, at a
// roughness of about 0.27.
const ROUGH_ALPHA: f32 = 0.075;

// Whether a denoiser should file a path under this surface, or look through it
// to whatever the path finds next. Cycles' rule: a lobe is non-specular if it is
// diffuse or rough, and a surface is captured once those make up a quarter of
// its weight.
//
// Metal, specular and glass share one roughness, so they are all rough or all
// sharp together, and the weights of every lobe add to one.
//
// Subsurface counts as diffuse, and it is already counted: `diffuse` below is
// the whole of what the coat lets through, which is the two lobes together. A
// skin or a marble is a surface a denoiser should file a path under, not a
// window to look through.
fn is_opaque(material: Material) -> bool {
    // A surface that scatters nothing is where every path to reach it ends, so
    // there is nothing behind it to look through to. A light is one.
    if !scatters(material) {
        return true;
    }

    let rough = ggx_alpha(material.roughness) > ROUGH_ALPHA;
    let diffuse = (1.0 - material.metallic) * (1.0 - material.transmission);
    // Diffuse is always non-specular. When rough, metal, specular and glass are
    // too, so every lobe counts.
    let non_specular = select(diffuse, 1.0, rough);
    return non_specular >= 0.25;
}

// The fraction of light a smooth dielectric boundary reflects, unpolarized and
// exact rather than Schlick's fit. `eta` is the index on the far side over the
// index on the near one, and `cosine` is measured on the near side.
//
// Glass at a glancing angle turns into a mirror, and this is what makes it do
// that; past the critical angle on the way out it is one outright.
//
// An index of one is answered before any arithmetic. It is zero exactly, and it
// has to be: the lambertian preset is a principled surface at that index, and a
// reflectance that rounded to a hair above zero would hand its specular lobe a
// share of the lobe selection — and a draw from the RNG — that it never had.
fn fresnel_dielectric(cosine: f32, eta: f32) -> f32 {
    if eta == 1.0 {
        return 0.0;
    }

    let cos_i = clamp(cosine, 0.0, 1.0);
    let sin2_t = (1.0 - cos_i * cos_i) / (eta * eta);
    // Total internal reflection.
    if sin2_t >= 1.0 {
        return 1.0;
    }
    let cos_t = sqrt(1.0 - sin2_t);

    let parallel = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    let perpendicular = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    return 0.5 * (parallel * parallel + perpendicular * perpendicular);
}

// A conductor's reflectance by Schlick's approximation, from its color at normal
// incidence. Blender fits an F82 tint on top of this; it is close enough without.
fn fresnel_schlick(f0: vec3f, cosine: f32) -> vec3f {
    let m = clamp(1.0 - cosine, 0.0, 1.0);
    let m2 = m * m;
    return f0 + (vec3f(1.0) - f0) * (m2 * m2 * m);
}

// The GGX (Trowbridge-Reitz) distribution of microfacet normals, for a normal
// `m` in the local frame above the surface.
//
// Written with `sin²` from the tangent components rather than as `1 - cos²`. At
// the small alphas just above the mirror cutoff the whole distribution sits
// within a milliradian of the pole, where `1 - cos²` has already rounded away
// every digit that says where in it `m` is.
fn ggx_d(m: vec3f, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let sin2 = m.x * m.x + m.y * m.y;
    let t = sin2 + a2 * m.z * m.z;
    return a2 / (PI * t * t);
}

// Smith's auxiliary function for GGX: how much of the microsurface facing `v`
// is hidden behind other microfacets. `G1 = 1 / (1 + Λ)`.
fn ggx_lambda(v: vec3f, alpha: f32) -> f32 {
    let cos2 = max(v.z * v.z, 1e-12);
    let tan2 = (v.x * v.x + v.y * v.y) / cos2;
    return 0.5 * (sqrt(1.0 + alpha * alpha * tan2) - 1.0);
}

// A microfacet normal drawn in proportion to how much of it `wo` can see: Dupuy
// and Benyoub's spherical-cap form of Heitz's visible normal sampling.
//
// Stretching the configuration by alpha turns the GGX ellipsoid into a unit
// hemisphere, whose visible normals are a uniform cap of the sphere offset by
// the view direction. Unstretching the result is the transform for a normal,
// which is why alpha scales the tangent components rather than dividing them.
fn sample_ggx_vndf(wo: vec3f, alpha: f32, u: vec2f) -> vec3f {
    let stretched = normalize(vec3f(wo.xy * alpha, wo.z));

    let phi = TWO_PI * u.x;
    let z = (1.0 - u.y) * (1.0 + stretched.z) - stretched.z;
    let sin_theta = sqrt(clamp(1.0 - z * z, 0.0, 1.0));
    let h = vec3f(sin_theta * cos(phi), sin_theta * sin(phi), z) + stretched;

    return normalize(vec3f(h.xy * alpha, max(h.z, 0.0)));
}

// How often `bsdf_sample` picks each of a principled surface's lobes, decided
// from `wo` alone so that `bsdf_eval` can reproduce it for any `wi`.
struct Lobes {
    metal: f32,
    specular: f32,
    glass: f32,
    diffuse: f32,
    subsurface: f32,
}

// The least share of the selection a lobe with any weight at all is given, as a
// fraction of the total. A dim lobe picked in proportion to its weight is found
// so rarely that every find is a firefly.
const MIN_LOBE_SHARE: f32 = 0.05;

// Each lobe's chance is roughly the light it sends back toward `wo`: the metal by
// its Fresnel color, the specular coat by the dielectric reflectance, the glass
// by its whole share of the surface — it reflects or transmits nearly all of
// what reaches it — and the diffuse by what the coat lets through. All zero for
// a surface that scatters nothing at all.
//
// The metal is weighed by its Fresnel term rather than its base color alone. A
// black conductor still turns into a mirror at a glancing angle, and a lobe with
// no chance of being sampled would lose that light outright.
//
// Subsurface takes its share out of the diffuse slot rather than beside it:
// both sit under the same coat, and `subsurface_weight` says how much of what
// the coat lets through goes *into* the surface instead of bouncing off its
// base. At a weight of zero every term below is the one it was before
// subsurface existed, down to the last bit — `w * 1.0` and `+ 0.0` do not
// round — which is what keeps every scene that has none on the random stream it
// already drew.
fn lobe_probabilities(material: Material, cos_o: f32) -> Lobes {
    let dielectric = 1.0 - material.metallic;
    let opaque = dielectric * (1.0 - material.transmission);
    let fresnel = fresnel_dielectric(cos_o, material.ior);
    let base = opaque * (1.0 - fresnel) * luminance(material.color);
    let weights = vec4f(
        material.metallic * luminance(fresnel_schlick(material.color, cos_o)),
        opaque * fresnel,
        dielectric * material.transmission,
        base * (1.0 - material.subsurface_weight),
    );
    let subsurface = base * material.subsurface_weight;

    let total = weights.x + weights.y + weights.z + weights.w + subsurface;
    if total <= 0.0 {
        return Lobes(0.0, 0.0, 0.0, 0.0, 0.0);
    }

    // A lobe alone keeps a chance of exactly one: `w / w` does not round.
    let floored = select(vec4f(0.0), max(weights, vec4f(MIN_LOBE_SHARE * total)), weights > vec4f(0.0));
    let floored_subsurface = select(0.0, max(subsurface, MIN_LOBE_SHARE * total), subsurface > 0.0);
    let sum = floored.x + floored.y + floored.z + floored.w + floored_subsurface;
    let p = floored / sum;
    return Lobes(p.x, p.y, p.z, p.w, floored_subsurface / sum);
}

// The index of refraction across the surface in the direction the path is
// travelling: what lies beyond it over what lies in front. Quoted against the
// air the scene is otherwise made of, so leaving an object inverts it.
fn relative_ior(material: Material, hit: Intersection) -> f32 {
    return select(1.0 / material.ior, material.ior, hit.front_face);
}

// Whether the glass lobe has no density: a mirror, or an index of one, where
// every refraction goes straight through undeviated whatever microfacet it
// found.
fn smooth_glass(alpha: f32, eta: f32) -> bool {
    return alpha < DELTA_ALPHA || eta == 1.0;
}

// The value and density of a principled surface's non-delta lobes, for two
// directions in its local frame. `value` is `f * |cos(wi)|`, and `eta` is
// `relative_ior`.
//
// Blender's layering reduced to four lobes: a GGX conductor weighed by
// `metallic`, and under `1 - metallic` a blend by `transmission` of GGX glass
// against a GGX dielectric coat over a diffuse base that gets whatever the coat
// lets through. Scaling the diffuse by `1 - F(wo)` is not reciprocal, which a
// path traced only from the camera never notices.
//
// The opaque lobes always see air on both faces — a surface that does not
// transmit has no inside to be in — so only the glass reads `eta`, and the rest
// read the index as written.
//
// The density is the one-sample mixture: every non-delta lobe's density at `wi`,
// weighed by how often that lobe is picked. That is what makes `value / pdf` the
// right weight whichever lobe actually drew the direction.
fn principled_eval(material: Material, lobes: Lobes, eta: f32, wo: vec3f, wi: vec3f) -> BsdfEval {
    let cos_o = wo.z;
    let cos_i = wi.z;
    // `wo` is on the side the normal faces by construction. Exactly tangent is
    // neither side, and has no density to give anything.
    if cos_o <= 0.0 || cos_i == 0.0 {
        return BsdfEval(vec3f(0.0), 0.0);
    }

    let dielectric = 1.0 - material.metallic;
    let glass = dielectric * material.transmission;
    let alpha = ggx_alpha(material.roughness);
    let lambda_o = ggx_lambda(wo, alpha);

    if cos_i < 0.0 {
        // Through the surface, which only rough glass can price.
        if smooth_glass(alpha, eta) || lobes.glass <= 0.0 {
            return BsdfEval(vec3f(0.0), 0.0);
        }
        return rough_transmission(material, alpha, eta, lambda_o, wo, wi, glass, lobes.glass);
    }

    let opaque = dielectric * (1.0 - material.transmission);
    let coat = 1.0 - fresnel_dielectric(cos_o, material.ior);
    // What the coat lets through, less the share of it that goes into the
    // surface rather than off its base. The subsurface lobe itself is priced at
    // nothing here and has no density: what it sends back depends on where the
    // walk inside comes out, which is not a function of `wi` at this point at
    // all. Next event estimation at the entry therefore only prices the lobes
    // that reflect, and the walk's exit vertex aims its own shadow ray.
    let diffuse = opaque * coat * (1.0 - material.subsurface_weight);
    var value = material.color * (diffuse * cos_i / PI);
    var pdf = lobes.diffuse * cos_i / PI;

    if alpha >= DELTA_ALPHA {
        // Both directions are above the surface, so the half vector is too.
        let m = normalize(wo + wi);
        let cos_m = dot(wo, m);

        let d = ggx_d(m, alpha);
        // Height-correlated masking and shadowing.
        let g2 = 1.0 / (1.0 + lambda_o + ggx_lambda(wi, alpha));

        // `D G2 F / (4 cos_o cos_i)` times `cos_i`, less the Fresnel term the
        // three reflecting lobes differ by. Glass at an index of one reflects
        // nothing, so it needs no guard of its own here.
        let microfacet = d * g2 / (4.0 * cos_o);
        let reflectance = fresnel_dielectric(cos_m, eta);
        let fresnel = material.metallic * fresnel_schlick(material.color, cos_m)
            + vec3f(opaque * fresnel_dielectric(cos_m, material.ior) + glass * reflectance);
        value += fresnel * microfacet;

        // The visible normal density `G1(wo) max(0, wo·m) D / cos_o`, through
        // the reflection's Jacobian `1 / (4 wo·m)`. Glass only reflects the
        // `F` of the time; the rest of its draws refract.
        let reflected = d / ((1.0 + lambda_o) * 4.0 * cos_o);
        pdf += (lobes.metal + lobes.specular + lobes.glass * reflectance) * reflected;
    }

    return BsdfEval(value, pdf);
}

// Rough glass's refraction lobe, after Walter et al. 2007 and pbrt-v4's
// `DielectricBxDF`, for a `wi` below the surface. `glass` is the lobe's share
// of the surface and `chance` how often it is picked.
//
// The microfacet that refracts `wo` into `wi` is the generalized half vector
// `wo + eta wi`, turned to face the outside. Radiance crossing the boundary is
// left unscaled by `1 / eta²`: every glass object here is closed, so what a path
// gains going in it gives back coming out.
fn rough_transmission(
    material: Material,
    alpha: f32,
    eta: f32,
    lambda_o: f32,
    wo: vec3f,
    wi: vec3f,
    glass: f32,
    chance: f32,
) -> BsdfEval {
    let half = normalize(wo + wi * eta);
    let m = select(half, -half, half.z < 0.0);

    // A facet `wo` cannot see, or one `wi` would have to reach from its front.
    let cos_om = dot(wo, m);
    let cos_im = dot(wi, m);
    if cos_om <= 0.0 || cos_im >= 0.0 {
        return BsdfEval(vec3f(0.0), 0.0);
    }

    let d = ggx_d(m, alpha);
    let g2 = 1.0 / (1.0 + lambda_o + ggx_lambda(wi, alpha));
    let transmitted = 1.0 - fresnel_dielectric(cos_om, eta);

    // The refraction's Jacobian `|wi·m| / (wi·m + wo·m / eta)²`. The sum is
    // `(2 wo·wi + eta + 1 / eta) / |wo + eta wi|`, which is only zero at an
    // index of one — ruled out before this is reached.
    let s = cos_im + cos_om / eta;
    let jacobian = -cos_im / (s * s);

    // `f |cos_i| = D G2 (1 - F) |wi·m| |wo·m| / (cos_o denom)`.
    let value = material.color * (glass * d * g2 * transmitted * cos_om * jacobian / wo.z);
    let visible = d * cos_om / ((1.0 + lambda_o) * wo.z);
    return BsdfEval(value, chance * transmitted * visible * jacobian);
}

fn absorbed() -> BsdfSample {
    return BsdfSample(vec3f(0.0), vec3f(0.0), 0.0, false, false, false);
}

fn delta_sample(wi: vec3f, weight: vec3f) -> BsdfSample {
    return BsdfSample(normalize(wi), weight, 0.0, true, false, true);
}

// Where the path goes next, and what the surface takes out of it on the way.
//
// `wo` points away from the surface, back along the way the path came. A light
// never reaches here — the path terminates on one.
//
// The subsurface lobe is the one that does not answer the question: it hands
// back a direction pointing into the surface and leaves it to
// [`subsurface_walk`] to find where the path comes back out.
fn bsdf_sample(material: Material, hit: Intersection, wo: vec3f) -> BsdfSample {
    let normal = hit.normal;
    let frame = orthonormal_basis(normal);
    let local_o = to_local(frame, wo);
    let lobes = lobe_probabilities(material, local_o.z);
    if local_o.z <= 0.0
        || lobes.metal + lobes.specular + lobes.glass + lobes.diffuse + lobes.subsurface <= 0.0 {
        return absorbed();
    }

    // Which lobe draws the direction. A surface with only one possible lobe does
    // not roll for it, which keeps the lambertian preset on the exact random
    // stream it drew before it became a principled surface.
    let possible = u32(lobes.metal > 0.0) + u32(lobes.specular > 0.0) + u32(lobes.glass > 0.0)
        + u32(lobes.diffuse > 0.0) + u32(lobes.subsurface > 0.0);
    var pick = 0.0;
    if possible > 1u {
        pick = rand_f32();
    }

    let alpha = ggx_alpha(material.roughness);
    let eta = relative_ior(material, hit);
    let cos_o = local_o.z;

    // Whether the direction drawn is meant to go through the surface rather
    // than back off it.
    var direction: vec3f;
    var through = false;
    // The thresholds are summed in floats, and only the single-lobe case is
    // exact — three shares that came from one division can total a hair under
    // one. So each branch also takes the sliver past its end when no later
    // lobe exists to claim it, rather than letting a draw in it fall through
    // to a lobe this surface does not have and be priced as one it does.
    let smooth_end = lobes.metal + lobes.specular;
    let glass_end = smooth_end + lobes.glass;
    let diffuse_end = glass_end + lobes.diffuse;
    if pick < smooth_end
        || (lobes.glass <= 0.0 && lobes.diffuse <= 0.0 && lobes.subsurface <= 0.0) {
        if alpha < DELTA_ALPHA {
            // A mirror. It has no density to weigh against the other lobes, so
            // the weight is this lobe's own `f * cos / pdf` — its reflectance —
            // over the chance it was picked.
            var weight: vec3f;
            if pick < lobes.metal {
                weight = material.metallic * fresnel_schlick(material.color, cos_o) / lobes.metal;
            } else {
                let opaque = (1.0 - material.metallic) * (1.0 - material.transmission);
                weight = vec3f(opaque * fresnel_dielectric(cos_o, material.ior) / lobes.specular);
            }
            return delta_sample(reflect(-wo, normal), weight);
        }

        let m = sample_ggx_vndf(local_o, alpha, vec2f(rand_f32(), rand_f32()));
        direction = normalize(to_world(frame, reflect(-local_o, m)));
    } else if pick < glass_end || (lobes.diffuse <= 0.0 && lobes.subsurface <= 0.0) {
        // Glass reflects the Fresnel fraction of the time and refracts the rest,
        // so choosing between the two in that proportion leaves a weight of one
        // for either, less the tint on what goes through.
        let share = (1.0 - material.metallic) * material.transmission / lobes.glass;

        if smooth_glass(alpha, eta) {
            // Past the critical angle the reflectance is one and there is no
            // refracted direction at all. The draw comes first so that it is
            // taken either way, and the zero `refract` hands back at the edge of
            // total internal reflection is caught even where rounding has left
            // the reflectance a hair below one.
            let refracted = refract(-wo, normal, 1.0 / eta);
            if rand_f32() < fresnel_dielectric(cos_o, eta) || all(refracted == vec3f(0.0)) {
                return delta_sample(reflect(-wo, normal), vec3f(share));
            }
            return delta_sample(refracted, material.color * share);
        }

        let m = sample_ggx_vndf(local_o, alpha, vec2f(rand_f32(), rand_f32()));
        if rand_f32() < fresnel_dielectric(dot(local_o, m), eta) {
            direction = normalize(to_world(frame, reflect(-local_o, m)));
        } else {
            direction = normalize(to_world(frame, refract(-local_o, m, 1.0 / eta)));
            through = true;
        }
    } else if pick < diffuse_end || lobes.subsurface <= 0.0 {
        // Diffuse: a cosine-weighted direction about the normal, drawn as the
        // normal plus a point on the unit sphere. Shrinking that sphere a hair
        // keeps the sum away from zero, so the result is always a direction.
        direction = normalize(normal + sample_sphere() * (1.0 - EPSILON));
    } else {
        // Subsurface: the same cosine-weighted draw, mirrored to point into the
        // surface. This is not where the path continues — it is where it enters
        // the medium, and `trace_path` walks it from here.
        //
        // The entry is a diffuse transmission through the boundary, whose
        // `f cos / pdf` is one at every angle for a cosine draw, so the weight
        // is the lobe's share of the surface over the chance it was picked and
        // nothing else. The colour is not in it: the walk inside carries that,
        // as the albedo it scatters with.
        let inward = -normalize(normal + sample_sphere() * (1.0 - EPSILON));
        let opaque = (1.0 - material.metallic) * (1.0 - material.transmission);
        let coat = 1.0 - fresnel_dielectric(cos_o, material.ior);
        let share = opaque * coat * material.subsurface_weight / lobes.subsurface;
        return BsdfSample(inward, vec3f(share), 0.0, false, true, true);
    }

    // A microfacet can reflect below the horizon, where the ray would travel
    // into the object it just left, and a refraction off a grazing enough facet
    // can come out on the side it went in from. Both are absorbed, and have to
    // be here rather than left to the density: `principled_eval` prices a
    // direction by the side it is on, and would price either one as the other
    // lobe. Anything that came out of the arithmetic as a NaN fails the test
    // on the density below, and is absorbed with them.
    let local_i = to_local(frame, direction);
    if (local_i.z < 0.0) != through {
        return absorbed();
    }
    let eval = principled_eval(material, lobes, eta, local_o, local_i);
    if !(eval.pdf > 0.0) {
        return absorbed();
    }

    return BsdfSample(direction, eval.value / eval.pdf, eval.pdf, false, false, true);
}

// The value and density of the surface's non-delta lobes at `wi`, both pointing
// away from the surface. A delta lobe has no density to evaluate: the chance a
// direction chosen by anything else lands exactly on it is zero, so it prices
// every direction at nothing.
fn bsdf_eval(material: Material, hit: Intersection, wo: vec3f, wi: vec3f) -> BsdfEval {
    if material.kind != PRINCIPLED {
        return BsdfEval(vec3f(0.0), 0.0);
    }

    // A zero pdf is what tells a light sample in a direction the surface does
    // not scatter into — behind an opaque one, most often — not to bother with
    // a shadow ray.
    let frame = orthonormal_basis(hit.normal);
    let local_o = to_local(frame, wo);
    let lobes = lobe_probabilities(material, local_o.z);
    return principled_eval(material, lobes, relative_ior(material, hit), local_o, to_local(frame, wi));
}

// Steps one walk inside a surface is allowed before it is given up on, as
// Cycles' random walk. Roulette below kills nearly every walk long before this,
// and a walk still going after two hundred and fifty-six scattering events is
// deep inside something opaque and carrying almost nothing.
const MAX_SUBSURFACE_STEPS: u32 = 256u;

// The shortest mean free path a walk is run with, in world units. The host has
// already zeroed the weight when every channel came out at nothing; this is for
// the channel that is at nothing on its own, where `1 / radius` would be an
// infinity and every transmittance in the walk after it a NaN.
const MIN_SUBSURFACE_RADIUS: f32 = 1e-6;

// How far off either end the asymmetry is held. At exactly one the phase
// function is a spike with no density to draw from, and its inverse cdf divides
// zero by zero at one end of the draw.
const MAX_ANISOTROPY: f32 = 0.999;

// The single-scattering albedo a random walk has to scatter with for the
// surface to come out the colour it was written as.
//
// Those two are nowhere near the same number. Light inside a medium scatters
// dozens of times before it finds its way back out, and every one of them takes
// its bite: a medium of 0.8 presents a diffuse albedo well under half that, and
// a surface written at 0.8 needs a medium of about 0.99 to match it. This is
// Chiang, Kutz and Burley's fit to that relation, inverted, so that a scene
// writes the colour it wants to see and not the one the physics needs.
//
// Clamped on the way in: `base_color` is only held to being non-negative, and
// the fit turns back on itself above one.
fn subsurface_albedo(base: vec3f) -> vec3f {
    let a = clamp(base, vec3f(0.0), vec3f(1.0));
    let x = 4.09712 + 4.20863 * a - sqrt(9.59217 + 41.6808 * a + 17.7126 * a * a);
    return 1.0 - x * x;
}

// Henyey-Greenstein's phase function as a density over the sphere, for the
// cosine between the way light was travelling and the way it goes next. `g`
// leans it: back the way it came at -1, every direction alike at zero, straight
// on at 1.
//
// The walk itself never calls this. It draws from the phase function and is
// weighed by it, and the two cancel to one — which is the whole reason
// [`sample_henyey_greenstein`] needs no weight. What this is for is saying so:
// the sampler is checked against this density, and this density against the
// analytic one, so a sampler that drew the right shape leaning the wrong way
// would have nothing to hide behind.
fn henyey_greenstein(cosine: f32, g: f32) -> f32 {
    // `1 + g² - 2g cos`, written so that neither term can cancel the other. At
    // a strong lean and a small angle the two halves of that expression agree to
    // four digits, and the handful left is what the density is about to be
    // raised to the power of -3/2 of. Both rearrangements below are exact, and
    // each one is a sum of two non-negative terms over the half of the range it
    // is used on.
    let denominator = max(select(
        (1.0 + g) * (1.0 + g) - 2.0 * g * (1.0 + cosine),
        (1.0 - g) * (1.0 - g) + 2.0 * g * (1.0 - cosine),
        g >= 0.0,
    ), 1e-12);
    return (1.0 - g * g) / (2.0 * TWO_PI * denominator * sqrt(denominator));
}

// A direction drawn in proportion to that density, for a walk currently
// travelling along `direction`.
fn sample_henyey_greenstein(direction: vec3f, g: f32) -> vec3f {
    let u = rand_f32();

    // The isotropic case is not a limit of the formula below — it is zero over
    // zero — so it is written out. Near enough to zero the two agree to well
    // inside a float anyway.
    var cosine = 1.0 - 2.0 * u;
    if abs(g) > 1e-3 {
        // The phase function's cdf, inverted. Runs from -1 at u = 0 to 1 at
        // u = 1 whichever way `g` leans, so the draw is monotone either way and
        // the clamp only catches rounding at the two ends.
        let s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u);
        cosine = clamp((1.0 + g * g - s * s) / (2.0 * g), -1.0, 1.0);
    }

    let sine = sqrt(max(1.0 - cosine * cosine, 0.0));
    let phi = TWO_PI * rand_f32();
    let frame = orthonormal_basis(direction);
    return normalize(to_world(frame, vec3f(sine * cos(phi), sine * sin(phi), cosine)));
}

// The density a free flight of `t` was drawn with, per unit distance: every
// channel's exponential, weighed by how often that channel is the one the
// distance is drawn against. Pricing every draw by the whole mixture rather
// than by the one channel that produced it is what keeps the three channels
// converging on a single colour instead of on three fields of noise.
fn subsurface_density(sigma_t: vec3f, chance: vec3f, t: f32) -> f32 {
    return dot(chance, sigma_t * exp(-sigma_t * t));
}

// The chance a flight drawn that way reaches `t` without scattering at all,
// which is the same mixture's tail and what a boundary inside the flight is
// priced by.
fn subsurface_survival(sigma_t: vec3f, chance: vec3f, t: f32) -> f32 {
    return dot(chance, exp(-sigma_t * t));
}

// Where a walk through the inside of a surface came back out, and what the
// medium took out of the path on the way.
struct Walk {
    // The exit point, on the object the path went in through.
    point: vec3f,
    // The surface normal there, pointing out of the object.
    normal: vec3f,
    // The triangle it came out through, so the exit vertex is as complete an
    // `Intersection` as any other.
    triangle: u32,
    // What the path is multiplied by for having made the trip.
    weight: vec3f,
    // False when the path never came back out: absorbed, rouletted away, or
    // still inside after `MAX_SUBSURFACE_STEPS`.
    valid: bool,
}

fn no_walk() -> Walk {
    return Walk(vec3f(0.0), vec3f(0.0), 0u, vec3f(0.0), false);
}

// A volumetric random walk through the inside of `object`, entered at `entry`
// travelling along `direction`, which points into the surface.
//
// This is the whole of subsurface scattering here, and it is deliberately not a
// diffusion profile: the path really does walk around inside the object, step
// by step, until it finds a boundary. Nothing about it is approximate except
// the medium it walks through, and it costs nothing the renderer did not
// already have — `traverse` is the same one the camera rays use, and the
// geometry it finds is the real geometry, so a thin ear glows and a thick cheek
// does not without either being written down anywhere.
//
// A step draws a free flight against one channel's extinction and prices it
// against all three, so the colours converge together rather than into three
// separate fields of noise. A boundary inside the flight ends the walk; nothing
// in the way means a scattering event, and the walk turns by the phase function
// and goes again.
//
// The medium belongs to one object, which is what `object` is for. Another
// object's surface crossing the same space is not a boundary of this one, and
// the walk slides past it rather than coming out through it. Alpha is ignored
// throughout: a cutout is a property of a surface seen from outside, and there
// is no outside in here.
//
// `object` is a material index, and it names an object only because the host
// writes one material per object and never shares one between two — `Scene::load`
// pushes an entry per `[[objects]]` block and hands that same index to the
// triangles. Deduplicating identical materials would quietly make two objects
// one medium, and a walk would come out through the wrong geometry with nothing
// to say it had. `every_object_has_its_own_material` holds the host to it.
fn subsurface_walk(material: Material, entry: vec3f, direction: vec3f, object: u32) -> Walk {
    let radius = max(material.subsurface_radius, vec3f(MIN_SUBSURFACE_RADIUS));
    let sigma_t = 1.0 / radius;
    let sigma_s = subsurface_albedo(material.color) * sigma_t;
    let g = clamp(material.subsurface_anisotropy, -MAX_ANISOTROPY, MAX_ANISOTROPY);

    var position = entry;
    var heading = direction;
    var weight = vec3f(1.0);

    for (var step = 0u; step < MAX_SUBSURFACE_STEPS; step += 1u) {
        // Which channel's mean free path this step is drawn against, in
        // proportion to what the path is still carrying: red travels furthest
        // through skin, and by the time a walk is deep inside one it is the
        // only channel left worth spending a step on. The densities below are
        // the whole mixture rather than the one channel, so every channel is
        // priced by every draw whichever one produced it.
        var chance = vec3f(1.0 / 3.0);
        let carried = weight.r + weight.g + weight.b;
        if carried > 0.0 {
            chance = weight / carried;
        }

        let draw = rand_f32();
        var extinction = sigma_t.b;
        if draw < chance.r {
            extinction = sigma_t.r;
        } else if draw < chance.r + chance.g {
            extinction = sigma_t.g;
        }

        // Free flight: an exponential in that channel's extinction.
        let distance = -log(1.0 - rand_f32()) / extinction;
        let ray = Ray(position, heading);
        let hit = traverse(ray, distance);

        if hit.t > 0.0 {
            // A boundary inside the flight. The path is priced by the chance it
            // got this far without scattering, which is the mixture's tail.
            let survived = subsurface_survival(sigma_t, chance, hit.t);
            if !(survived > 0.0) {
                return no_walk();
            }
            weight *= exp(-sigma_t * hit.t) / survived;

            if hit.material != object {
                // Something else's surface, crossing the medium. Not a boundary
                // of this object, so the path slides up to it and carries on
                // the same way. The flight is redrawn from there, which an
                // exponential is free to do: it has no memory of how far the
                // path had already come.
                position = point_on_ray(ray, hit.t + EPSILON);
                continue;
            }

            // `traverse` turns the normal to meet the ray, and this ray is on
            // its way out, so what comes back points into the medium.
            return Walk(point_on_ray(ray, hit.t), -hit.normal, hit.triangle, weight, true);
        }

        // Nothing in the way, so the flight ends in a scattering event.
        let density = subsurface_density(sigma_t, chance, distance);
        if !(density > 0.0) {
            return no_walk();
        }
        weight *= sigma_s * exp(-sigma_t * distance) / density;

        position = point_on_ray(ray, distance);
        heading = sample_henyey_greenstein(heading, g);

        // Roulette, on the rule `trace_path` uses on its own bounces: what the
        // walk carries is unchanged in expectation, and a walk the medium has
        // absorbed down to nothing stops spending traversals on itself.
        let survival = min(max(weight.r, max(weight.g, weight.b)), 1.0);
        if !(survival > 0.0) || rand_f32() >= survival {
            return no_walk();
        }
        weight /= survival;
    }

    return no_walk();
}

// The surface a walk comes back out through: white, perfectly diffuse, and at
// an index of one so it has no specular coat of its own. The coat is priced
// once, on the way in, and what the medium took out of the path is already in
// the walk's weight — there is nothing left here to tint or to reflect.
//
// A literal `Material` and not a special case, so that `direct_light`,
// `direct_sky` and `bsdf_sample` all work on the exit exactly as they do on any
// other diffuse surface.
fn exit_surface() -> Material {
    return Material(
        vec3f(1.0),
        PRINCIPLED,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        vec3f(0.0),
        vec3f(0.0),
    );
}

// The ray through `pixel` for this sample: jittered inside the pixel so that
// samples across passes anti-alias the edges, and started from a point on the
// lens so that everything off the focus plane blurs.
//
// The jitter is stratified rather than drawn twice from [0, 1). Independent
// draws clump and gap exactly as often as chance allows; one draw per cell of a
// square grid cannot, and it costs nothing. The book nests two loops per pixel
// and reads the cell off the counters, but a dispatch here carries one sample,
// so the cell has to come from `camera.sample` — which the host rewrites every
// pass anyway.
//
// Only the pixel is stratified. The lens draw below, the scatter directions and
// the light draws stay independent: padding those dimensions needs machinery
// this does not have, and the edge noise is what the pixel grid buys.
fn primary_ray(pixel: vec2u) -> Ray {
    let resolution = vec2f(f32(camera.width), f32(camera.height));

    // Which cell of the s-by-s grid this sample owns. With `samples` an exact
    // multiple of `s * s` every cell takes the same count; otherwise the wrap
    // leaves them differing by one, which is what unstratified sampling manages
    // only on average — so it is never worse than what this replaces.
    let s = max(camera.strata, 1u);
    let cell = (camera.sample - 1u) % (s * s);

    // Every pixel would otherwise walk the grid in the same order, which leaves
    // the residual error correlated across the image and reads as banding rather
    // than as noise. Rotating the cell per pixel is a bijection on [0, s), so a
    // pixel still visits every cell exactly once per `s * s` samples. Both
    // offsets are reduced before the addition so the sum cannot wrap a u32,
    // which would cost the map its bijectivity for the pixels it happened to.
    //
    // The seed goes in here as well as into `rng_state`, and this is the only
    // dimension that needs telling about it: everything else a sample draws is
    // already decorrelated by the stream. Without it two seeds would still
    // assign sample `k` to the same cell of the grid for every pixel in the
    // frame, which is the one thing a reference "sharing no noise" cannot do.
    // `jenkins_hash(0)` is zero, so the default render is unchanged.
    let shuffle = jenkins_hash(pixel.x + pixel.y * camera.width) ^ jenkins_hash(camera.seed);
    let x = (cell % s + shuffle % s) % s;
    let y = (cell / s + (shuffle >> 16u) % s) % s;

    let jitter = (vec2f(f32(x), f32(y)) + vec2f(rand_f32(), rand_f32())) / f32(s);
    let uv = (vec2f(pixel) + jitter) / resolution;

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

// The normal of the triangle as it was built, not as it is shaded. Emission
// sidedness is a question about the surface a point was drawn on, and an
// interpolated normal is not that surface — the two disagree on a smooth-shaded
// mesh near a silhouette, which is the hardest possible place to notice that one
// strategy is claiming a direction the other prices at zero.
fn geometric_normal(tri: Triangle) -> vec3f {
    return normalize(cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
}

// How foreshortened an emitter is along `direction`, which points from the
// shading point toward it — and zero when the emitter has its back turned, so
// that [`light_density`] prices the draw at nothing. Emission is one-sided,
// always: a light gives off nothing through the face its winding points away
// from.
//
// Both [`sample_light`] and [`light_pdf`] go through here for the same reason
// the cosine floor lives in one place: the two have to agree exactly on which
// directions the light sampler can produce, or the heuristic loses light on one
// side and doubles it on the other.
fn emitted_cosine(tri: Triangle, direction: vec3f) -> f32 {
    // Positive when the winding points back along the direction of travel, which
    // is the face a shading point at the other end of it can see.
    let facing = -dot(geometric_normal(tri), direction);

    return max(facing, 0.0);
}

// What an emitter gives off on average, which is what light sampling has to
// price and return. A surface with an alpha below one is only there that
// fraction of the time: a scattered ray rolls for it and collects the full
// emission when it stays, `alpha` of it on average, and a shadow ray stops short
// of the surface it aims at without rolling. Scaling here is what makes the two
// strategies estimate the same thing, and it matches `power` in light.rs, which
// the table and `camera.light_power` are built from.
fn emitted(material: Material) -> vec3f {
    return material.emission * material.alpha;
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
//
// The floor under `cosine` is not the same thing as rejecting a back face. A
// direction can be arbitrarily close to tangent to an emitter without being
// behind it, and dividing by that cosine sends the density to infinity — which
// [`power_heuristic`] then squares into `inf / inf`, a NaN that reaches the
// accumulator and stays there. Rejecting those directions outright is the fix,
// and it belongs here rather than in either caller: this one function is what
// both [`sample_light`] and [`light_pdf`] price a direction with, so a floor
// raised in one place keeps the two strategies agreeing about which directions
// the light sampler can produce at all.
const MIN_LIGHT_COSINE: f32 = 1e-6;

fn light_density(emit: vec3f, distance: f32, cosine: f32) -> f32 {
    if cosine <= MIN_LIGHT_COSINE || camera.light_power <= 0.0 {
        return 0.0;
    }

    return (distance * distance) * luminance(emit) / (cosine * camera.light_power);
}

// The density with which [`sample_light`] would have produced a direction, given
// that following it landed on `hit`, `distance` from the vertex that chose it.
//
// Only the emitter a ray reaches first matters: anything behind it is occluded,
// so a shadow ray aimed there returns nothing and that draw contributes zero.
//
// The distance is not `hit.t`. It is whenever the ray came straight from the
// vertex, but a ray that passed through a cutout on the way was re-cast from
// there, and its `t` measures only the last stretch.
fn light_pdf(hit: Intersection, direction: vec3f, distance: f32) -> f32 {
    let tri = triangles[hit.triangle];
    let material = materials[hit.material];
    let cosine = emitted_cosine(tri, direction);

    // An emitter arrived at from behind prices at zero here, which is the half
    // of sidedness that is easy to forget and the half that keeps the heuristic
    // consistent.
    return light_density(emitted(material), distance, cosine);
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
    // that surface.
    let material = materials[tri.material];
    let cosine = emitted_cosine(tri, direction);

    var pdf = 0.0;
    if distance > 0.0 {
        pdf = light_density(emitted(material), distance, cosine);
    }

    // An emitter with its back turned leaves `pdf` at zero, and
    // `direct_light` ends the draw on that before the shadow ray is cast — so
    // this is a traversal saving as much as it is a knob.
    return LightSample(direction, distance, emitted(material), pdf);
}

// The row of the sky's distribution a uniform draw lands in: the first whose
// cumulative chance has overtaken it. The same search `select_light` runs, over
// the marginal rather than the emitter table.
fn sky_row(draw: f32) -> u32 {
    var low = 0u;
    var high = camera.sky_height - 1u;

    loop {
        if low >= high {
            break;
        }

        let middle = (low + high) / 2u;
        if sky_marginal[middle] > draw {
            high = middle;
        } else {
            low = middle + 1u;
        }
    }

    return low;
}

// And the column within that row, out of the conditional distribution stored
// for it.
fn sky_column(row: u32, draw: f32) -> u32 {
    let base = row * camera.sky_width;
    var low = 0u;
    var high = camera.sky_width - 1u;

    loop {
        if low >= high {
            break;
        }

        let middle = (low + high) / 2u;
        if sky_conditional[base + middle] > draw {
            high = middle;
        } else {
            low = middle + 1u;
        }
    }

    return low;
}

// The density with which [`sample_sky`] draws `direction`, per unit solid angle,
// and zero for a scene with no distribution to aim with — which is what leaves
// the sky to scattering alone and gives an escaped ray its full weight in
// [`trace_path`].
//
// The two cumulative tables are differenced back into the chance of the single
// cell the direction lands in, which is why the host uploads no densities
// beside them. Multiplied by the cell count that chance is a density over the
// unit square, and the divisor converts it to one over solid angle: with
// `theta = PI * v` and `phi = TWO_PI * u`, `sin(theta) dtheta dphi` is
// `2 PI^2 sin(theta) du dv`.
//
// That `sin(theta)` is the real one, taken from the direction being priced. It
// is not the `sin(theta)` the host folded into each cell's weight, and confusing
// the two is the easy mistake here. The host's decides *where* samples go; this
// one converts between measures, and it has to be evaluated pointwise or what
// comes back is not a density at all.
fn sky_pdf(direction: vec3f) -> f32 {
    return sky_pdf_uv(environment_uv(direction));
}

// The same density, given the place on the map rather than the direction that
// reaches it.
//
// [`sample_sky`] holds a uv and wants its density; an escaped ray holds a
// direction and wants the same number for it. Both go through this one function
// so the two strategies cannot disagree about what a cell is worth — which was
// the reason the pdf was taken from the direction in the first place, and is
// still the reason, just enforced a level lower down where the sampler does not
// have to make a round trip to get it.
fn sky_pdf_uv(uv: vec2f) -> f32 {
    if camera.sky_width == 0u {
        return 0.0;
    }

    let sin_theta = sin(uv.y * PI);
    if sin_theta <= 0.0 {
        // Straight up or straight down: no solid angle to spread a density over.
        return 0.0;
    }

    // `atan2` returns PI on the seam, which puts u at exactly one and the index
    // one past the end of the row.
    let x = min(u32(uv.x * f32(camera.sky_width)), camera.sky_width - 1u);
    let y = min(u32(uv.y * f32(camera.sky_height)), camera.sky_height - 1u);

    var row = sky_marginal[y];
    if y > 0u {
        row -= sky_marginal[y - 1u];
    }

    let base = y * camera.sky_width;
    var column = sky_conditional[base + x];
    if x > 0u {
        column -= sky_conditional[base + x - 1u];
    }

    let cells = f32(camera.sky_width * camera.sky_height);
    return row * column * cells / (TWO_PI * PI * sin_theta);
}

// One direction drawn out of the sky, and what the estimator needs to weigh it.
// The same shape as [`LightSample`] without the distance: nothing stands behind
// the sky, so the shadow ray that checks it runs to infinity.
struct SkySample {
    // From the shading point toward the sky, unit length.
    direction: vec3f,
    // What the map holds there.
    radiance: vec3f,
    // Solid angle density of having drawn this direction. Zero is unusable.
    pdf: f32,
}

// Draws a direction in proportion to how bright the map is along it: a cell out
// of the distribution, then a point uniformly inside that cell.
//
// Uniformly inside is exactly right rather than an approximation — the density
// is piecewise constant over a cell by construction. It is also why the host is
// free to build the distribution coarser than the map it describes: a coarse
// cell aims less precisely and the estimator divides by a correspondingly
// blunter density, and neither of those is a bias.
//
// The pdf comes back through [`sky_pdf`] rather than out of the cell that was
// drawn, so that the density this sample is divided by and the density a
// scattered ray arriving the same way is priced at are the same function of the
// same direction. Two evaluations that disagreed would leave the heuristic's
// weights summing to something other than one, which loses light on one side and
// doubles it on the other.
fn sample_sky() -> SkySample {
    let row = sky_row(rand_f32());
    let column = sky_column(row, rand_f32());

    let cell = vec2f(f32(column), f32(row)) + vec2f(rand_f32(), rand_f32());
    let uv = cell / vec2f(f32(camera.sky_width), f32(camera.sky_height));

    // The uv is what was drawn, so it is what the map is read at and what the
    // density is taken from. Only the shadow ray needs a direction, and that
    // conversion happens once rather than being undone and redone twice.
    return SkySample(
        environment_direction(uv),
        environment_radiance_uv(uv),
        sky_pdf_uv(uv),
    );
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
// Only worth calling on a surface with a smooth lobe (`has_smooth_lobe`). A
// delta lobe is priced at zero by `bsdf_eval`, and scattering already finds the
// light along it every time.
//
// The caller has to have checked that there is a light to sample: with none,
// `sample_light` has nothing to pick from.
fn direct_light(point: vec3f, hit: Intersection, material: Material, wo: vec3f) -> vec3f {
    let light = sample_light(point);
    if !(light.pdf > 0.0) {
        return vec3f(0.0);
    }

    // A zero pdf covers a light behind the surface as well as a lobe that could
    // never have scattered this way, and either one saves the shadow ray.
    //
    // Written as the negation of the test that has to pass, the way
    // `bsdf_sample` writes it, so that a NaN density is caught here and not
    // carried into the heuristic — `pdf <= 0.0` is false for one.
    let bsdf = bsdf_eval(material, hit, wo, light.direction);
    if !(bsdf.pdf > 0.0) {
        return vec3f(0.0);
    }

    if occluded(point, light.direction, light.distance) {
        return vec3f(0.0);
    }

    // Scattering would have found this same direction with this density, and
    // the heuristic splits the contribution between the two strategies. Near a
    // large emitter scattering is the better one and keeps most of it; for the
    // small bright light that made the fireflies, this sample keeps nearly all.
    let weight = power_heuristic(light.pdf, bsdf.pdf);

    // The estimator is `f * cos / pdf`, and `bsdf.value` is already `f * cos`.
    return light.radiance * bsdf.value * weight / light.pdf;
}

// The same thing for the sky, which is an emitter here as much as a triangle is.
//
// An escaped ray finds the environment map on its own and always has, so this is
// not what makes an HDRI light the scene — it is what makes it light the scene
// *quietly*. A photographed sky puts a fifth of its light into a twentieth of a
// percent of its texels, so a cosine-weighted bounce finds the sun about one
// draw in two thousand and comes back carrying thousands of times the average.
// That is a firefly, and it is why a map like that renders soft ambient light
// beautifully and a sharp sun shadow not at all. Aiming at the sun finds it
// every draw, and the shadow ray is what turns it into a shadow.
//
// Kept beside [`direct_light`] rather than folded into it because the two are
// independent estimators and not two strategies for one: a direction either
// reaches an emissive triangle or escapes to the sky, never both, so each is its
// own two-way contest against scattering and neither one's weight has anything
// to say about the other's. A surface under both pays for two shadow rays, which
// is the honest price of sampling two things.
fn direct_sky(point: vec3f, hit: Intersection, material: Material, wo: vec3f) -> vec3f {
    let sky = sample_sky();

    // A direction the distribution gives no weight to.
    if !(sky.pdf > 0.0) {
        return vec3f(0.0);
    }

    // Or one the surface does not scatter into, behind it included. Negated
    // like `direct_light`'s, and for the same reason: a NaN fails it.
    let bsdf = bsdf_eval(material, hit, wo, sky.direction);
    if !(bsdf.pdf > 0.0) {
        return vec3f(0.0);
    }

    // To infinity, because that is where the sky is: `traverse` reads the limit
    // as the farthest hit it will accept, so anything at all in the way blocks
    // this.
    if occluded(point, sky.direction, FLT_MAX) {
        return vec3f(0.0);
    }

    let weight = power_heuristic(sky.pdf, bsdf.pdf);

    return sky.radiance * bsdf.value * weight / sky.pdf;
}

// Follows one path from the camera until it reaches a light, escapes the scene,
// is absorbed, or runs out of bounces, and returns the radiance it carried back.
fn trace_path(primary: Ray) -> Path {
    var ray = primary;
    var radiance = vec3f(0.0);
    var throughput = vec3f(1.0);

    // The surface this path will be filed under, and whether it is the one we
    // actually wanted. See the capture rule below.
    var features = no_features();
    var captured = false;

    // How far the path has already walked when it reaches the current vertex.
    // Every scattered ray starts at the surface it left, so `hit.t` is the
    // length of one segment; a depth that means anything to a consumer is the
    // sum of the segments before it plus that.
    var traveled = 0.0;

    // How far the ray has come since the vertex that chose its direction.
    // Usually that is just `hit.t`; a pass-through re-casts the ray from the
    // cutout, and the stretch before it has to be remembered to price an
    // emitter from where the direction was actually drawn.
    var segment = 0.0;

    // Surfaces passed through by their alpha, against `MAX_TRANSPARENT`.
    var passes = 0u;

    // How the current direction was chosen, which is what decides how much of an
    // emitter this ray is allowed to keep. A camera ray counts as specular:
    // nothing sampled it, so no other strategy could have found it and there is
    // nothing to share the light with.
    var scatter_pdf = 0.0;
    var specular = true;

    // A `while` and not a `for`, so that a pass-through can `continue` without
    // spending a bounce.
    var bounce = 0u;
    while bounce <= camera.max_bounces {
        let hit = intersect_scene(ray);
        if hit.t < 0.0 {
            // Nothing was hit, so the path escapes and the sky lights it.
            //
            // The previous vertex may already have aimed a shadow ray at this
            // same part of the sky, in which case keeping all of what this ray
            // found would count it twice — the identical bookkeeping an emissive
            // triangle gets below, and the half of it that is easy to forget.
            // `sky_pdf` returns zero when the scene has no distribution to aim
            // with, so a scene that never sampled the sky hands over everything
            // without needing a second path through here.
            var weight = 1.0;
            if !specular {
                weight = power_heuristic(scatter_pdf, sky_pdf(ray.direction));
            }
            let escaped = environment_radiance(ray.direction) * weight;
            return Path(radiance + throughput * escaped, features);
        }

        let material = materials[hit.material];

        // Coverage. With probability `1 - alpha` the surface is not there: the
        // ray carries on from it undeviated, and nothing about the path
        // changes — not the throughput, and not how its direction was chosen,
        // so whatever it lands on next is weighed against the vertex that
        // actually sampled it. It is not a feature either: the denoiser should
        // see what the path stopped on.
        //
        // The roll is only taken below one, so an opaque scene keeps exactly
        // the random stream it drew before alpha existed.
        if material.alpha < 1.0 && rand_f32() >= material.alpha {
            ray = Ray(point_on_ray(ray, hit.t), ray.direction);
            segment += hit.t;
            traveled += hit.t;
            passes += 1u;
            if passes > MAX_TRANSPARENT {
                break;
            }
            continue;
        }

        // The surface a denoiser gets to steer by, and it is deliberately not
        // the first one the ray met. Following melee's glass torus through to
        // the diffuse floor behind it hands over an edge that is really there;
        // capturing the glass itself hands over the torus's own silhouette and
        // smears the whole refraction behind it.
        //
        // So: the first opaque hit, and the first hit of any kind as a fallback
        // for a path that never finds one — a ray that leaves through the far
        // side of a dielectric and escapes still has to be filed under
        // something. "Opaque" is `is_opaque`: diffuse or rough enough that what
        // it reflects is a blur rather than a picture, so a polished metal is
        // looked through the way glass is. A light counts as a surface too and
        // ends the search the same way a diffuse one does: a light seen through
        // glass is as much a surface behind that glass as a floor is.
        let opaque = is_opaque(material);
        if !captured && (bounce == 0u || opaque) {
            features = Features(hit.normal, traveled + hit.t, feature_albedo(material), i32(hit.material));
        }
        captured = captured || opaque;

        // Emission, from any surface that has some, and only off its front.
        //
        // The previous vertex already sampled this emitter directly, so taking
        // all of what the scattered ray found would count it twice. The
        // heuristic hands over only the share this strategy earned, and
        // `direct_light` took the rest. A specular bounce, or the camera, had no
        // direct sample to double, and keeps everything.
        //
        // An emitter struck on its back emits nothing. Decided by the geometric
        // normal and never by `hit.front_face`, which is derived from the
        // interpolated vertex normal: on a smooth-shaded mesh the two part
        // company near a silhouette, and `sample_light` draws its points on the
        // geometric surface.
        if any(material.emission > vec3f(0.0)) {
            let front = dot(geometric_normal(triangles[hit.triangle]), ray.direction) < 0.0;
            if front {
                var weight = 1.0;
                if !specular {
                    weight = power_heuristic(scatter_pdf, light_pdf(hit, ray.direction, segment + hit.t));
                }
                radiance += throughput * material.emission * weight;
            }
        }

        // A surface with no lobe to scatter by is where the path ends — a light,
        // most often — and before any shadow ray is spent on it.
        if !scatters(material) {
            break;
        }

        // Where the path gathers light and chooses where to go next. Usually
        // that is the surface just hit; a subsurface entry moves it to the far
        // end of a walk through the inside of the object, where a white
        // Lambertian facing out takes over. The vertex is replaced rather than
        // added to, so everything below reads the same four names either way.
        var surface = material;
        var vertex = hit;
        var origin = point_on_ray(ray, hit.t);
        var outgoing = -ray.direction;

        var scattered = absorbed();
        // Two vertices at most: the one that was hit, and the exit of the one
        // walk it may dive into. The exit is a white Lambertian and cannot pick
        // a subsurface lobe, so the second pass always ends the loop on its
        // own; the bound is here so that it is a bound and not a hope.
        for (var vertices = 0u; vertices < 2u; vertices += 1u) {
            // Light arriving straight from an emitter, gathered before the path
            // wanders off to find whatever else this surface can see. Both kinds
            // of emitter, where a scene has both: the triangles in the light
            // table, and the sky.
            if has_smooth_lobe(surface) {
                if camera.light_count > 0u {
                    radiance += throughput * direct_light(origin, vertex, surface, outgoing);
                }
                if camera.sky_width > 0u {
                    radiance += throughput * direct_sky(origin, vertex, surface, outgoing);
                }
            }

            scattered = bsdf_sample(surface, vertex, outgoing);
            if !scattered.valid || !scattered.subsurface {
                break;
            }

            // Into the surface. The entry's own weight is collected here rather
            // than below, because what is below belongs to the exit's sample
            // and this one is about to be replaced by it.
            let walk = subsurface_walk(surface, origin, scattered.wi, vertex.material);
            if !walk.valid {
                scattered = absorbed();
                break;
            }
            throughput *= scattered.weight * walk.weight;

            surface = exit_surface();
            // `t` is never read off a vertex, and the material index is kept so
            // that the exit is still filed under the object it came out of.
            vertex = Intersection(walk.normal, 0.0, vertex.material, true, walk.triangle);
            origin = walk.point;
            // The exit is a diffuse transmission: what leaves does not depend on
            // which way the walk happened to arrive, and a surface at an index
            // of one has no coat for the direction to matter to. Straight out is
            // the one choice that is above the hemisphere at every exit.
            outgoing = walk.normal;
        }
        if !scattered.valid {
            break;
        }

        // A delta lobe drew from nothing an emitter can be weighed against, so
        // whatever it finds next keeps everything.
        specular = scattered.delta;
        scatter_pdf = scattered.pdf;

        throughput *= scattered.weight;
        ray = Ray(origin, scattered.wi);
        traveled += hit.t;
        segment = 0.0;

        // Russian roulette. Past the first few bounces a path is killed with
        // probability `1 - survival` and the survivors are divided by
        // `survival`, so what the path carries is unchanged in expectation —
        // the estimator stays unbiased, it just spends its bounces on the paths
        // that still matter.
        //
        // The brightest channel is the survival probability: a throughput of
        // 0.73 across the board survives 73% of the time, which turns melee's
        // 64-bounce paths into about eight. Anything at or above one always
        // survives and is never scaled, which is what keeps clear white glass —
        // smooth enough to attenuate by nothing — walking the full path it
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

        bounce += 1u;
    }

    // Out of bounces, or absorbed: whatever it had gathered is all it gets.
    return Path(radiance, features);
}

fn index_of(pixel: vec2u) -> u32 {
    return pixel.y * camera.width + pixel.x;
}

// A sample scaled back to the top of this pixel's acceptance window, or handed
// straight back if it is inside it.
//
// The window is `mean + k * deviation * n^(1/4)` over the samples gathered so
// far, and three things about it are load-bearing:
//
//   - **The deviation is floored at the mean.** A pixel whose samples have all
//     agreed has a variance near zero and would reject the next one to differ at
//     any `k` at all; the floor means a sample has to be several times the
//     pixel's own brightness before it counts as an outlier, however tightly the
//     samples so far happened to agree.
//   - **A pixel with no light yet accepts anything.** Its mean and its variance
//     are both zero, so the window is zero and it would swallow the first real
//     sample to arrive — which in a dim corner is the one carrying the whole
//     pixel.
//   - **The scale keeps the hue.** Discarding a sample outright loses its energy
//     unconditionally; scaling all three channels by the same factor caps the
//     magnitude and leaves the colour it arrived with.
fn reject_outlier(index: u32, radiance: vec3f) -> vec3f {
    let n = f32(camera.sample - 1u);
    if camera.outlier_k <= 0.0 || camera.sample <= camera.outlier_warmup || n <= 0.0 {
        return radiance;
    }

    let moment = moments[index].drawn / n;
    let mean = moment.x;
    let variance = max(0.0, moment.y - mean * mean);
    let deviation = max(sqrt(variance), mean);

    let threshold = mean + camera.outlier_k * deviation * pow(n, 0.25);
    let luma = luminance(radiance);
    if threshold <= 0.0 || luma <= threshold {
        return radiance;
    }

    return radiance * (threshold / luma);
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
    let path = trace_path(primary_ray(pixel));
    let radiance = path.radiance;

    // A backstop for whatever still manages to produce one. The accumulator sums
    // across every dispatch, so a single non-finite sample would otherwise turn
    // that pixel's running total into a NaN no later sample could pull back —
    // one bad path costing the pixel the entire render. Dropping the sample and
    // still counting it in `w` costs one sample's worth of light instead.
    //
    // NaN fails every comparison it is given, so this rejects one as well as an
    // infinity. A finite value large enough to trip it is already past anything
    // the accumulator could average back down.
    let finite = abs(radiance) < vec3f(FLT_MAX);
    let clean = select(vec3f(0.0), radiance, finite);

    // Firefly rejection: a sample far enough above what this pixel has been
    // averaging is scaled back to the edge of the window rather than kept whole.
    //
    // The window widens as `n^(1/4)`, which is the entire reason this converges
    // rather than biasing the image permanently — a sample capped at a hundred
    // samples is accepted at ten thousand, so the estimator still reaches the
    // right answer, it just gets there without the variance a single lucky path
    // would otherwise inject. Off unless a scene asks for it.
    let capped = reject_outlier(index_of(pixel), clean);

    // Radiance sums in `rgb`, samples in `w`, so the host can average without
    // being told how many passes ran.
    let index = index_of(pixel);
    accum[index] = accum[index] + vec4f(capped, 1.0);

    // `drawn` takes `clean` and not `capped`, which is the trap this whole stage
    // turns on. Those moments are what the threshold above is computed from;
    // feed them the value the threshold already shrank and the variance
    // collapses toward it, the window tightens in response, and the bias locks
    // in at a level no sample count undoes. They have to keep measuring the
    // distribution the tracer actually draws from, not the one the rejection
    // leaves behind.
    //
    // `kept` takes `capped`, for the opposite reason: the filter and the host
    // ask how noisy the *image* is, and a firefly the rejection already scaled
    // down is noise the image does not have. Measured off `clean`, one capped
    // firefly would widen the filter's luminance threshold around exactly the
    // pixel it was removed from, and the filter would blur across edges there.
    //
    // The non-finite guard still applies: a NaN would poison the variance of the
    // pixel for the rest of the render exactly as it would have poisoned its
    // colour.
    //
    // This draws nothing from the RNG and feeds nothing back into the path, so
    // with the rejection off the image is bit-for-bit what it was without any of
    // it.
    let drawn = luminance(clean);
    let kept = luminance(capped);
    let moment = moments[index];
    moments[index] = Moments(
        moment.drawn + vec2f(drawn, drawn * drawn),
        moment.kept + vec2f(kept, kept * kept),
    );

    // The features, summed the same way. Unguarded, and deliberately: a normal,
    // a distance and a surface colour are finite whatever the path did with
    // them, and a sample whose radiance was thrown away still saw a real
    // surface worth averaging in.
    normals[index] = normals[index] + vec4f(path.features.normal, path.features.depth);
    let albedo = albedos[index];
    albedos[index] = Albedo(albedo.color + path.features.albedo, vote(albedo.votes, path.features.object));
}

// One sample's ballot in a pixel's object-id vote, packed as `Albedo` describes.
//
// Boyer-Moore: a matching id extends the candidate's lead, any other id cuts it,
// and a lead cut to nothing hands the seat to the next id to arrive. Whatever
// holds more than half the samples is guaranteed to be sitting there at the end,
// which is exactly "whichever surface most of its samples saw" — a silhouette
// pixel that is nine-tenths teapot is filed under the teapot, and a pane of
// glass over a floor under the floor, rather than under the mean of the two.
fn vote(votes: u32, object: i32) -> u32 {
    let ballot = u32(object + 1);
    let candidate = votes >> 16u;
    let lead = votes & 0xffffu;

    if lead == 0u {
        return (ballot << 16u) | 1u;
    }
    if candidate == ballot {
        return (candidate << 16u) | min(lead + 1u, 0xffffu);
    }
    return (candidate << 16u) | (lead - 1u);
}
