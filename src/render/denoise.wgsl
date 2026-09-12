// The spatial half of SVGF: an edge-avoiding a-trous wavelet filter, run once
// over the finished accumulator.
//
// The trick the whole thing rests on is in the tap stride. Each pass reads the
// same 5x5 neighbourhood but spaces its taps twice as far apart as the one
// before it, so five passes of 25 taps reach across 125 pixels — cumulatively,
// no single pass spanning more than 65 — for the cost of 125 taps rather than
// the 15625 a 125x125 box would need. What keeps that from being a blur is the
// weight on each tap: a tap is only allowed to contribute in proportion to how
// much it looks like it is sitting on the same surface, judged by the noise-free
// feature buffers the tracer wrote beside the radiance.
//
// Three passes drive it, all through one entry point so that wgpu derives one
// bind group layout for the lot and the ping-pong buffers stay interchangeable:
// `PREPARE` builds the filter's input, `FILTER` runs one a-trous iteration, and
// `REMODULATE` puts the colour back.

const PREPARE: u32 = 0u;
const FILTER: u32 = 1u;
const REMODULATE: u32 = 2u;

// Divisors are guarded rather than branched around, and this is what they are
// guarded with. Small enough to change no answer that was not already degenerate.
const EPSILON: f32 = 1e-8;

// The darkest albedo the filter will divide radiance by. A surface darker than
// this turns a demodulated irradiance into a number large enough to swamp its
// neighbours, so it is clamped and those pixels filter as radiance instead —
// which is the behaviour the filter had before demodulation was added at all.
const ALBEDO_FLOOR: f32 = 1e-2;

// B3-spline, the kernel SVGF uses. Separable, so the 5x5 is this outer-producted
// with itself and never stored.
var<private> KERNEL: array<f32, 5> = array(0.0625, 0.25, 0.375, 0.25, 0.0625);

struct Denoise {
    width: u32,
    height: u32,
    // Pixels between adjacent taps: 1, 2, 4, 8, 16 across the five passes.
    stride: u32,
    // Which of the three passes above this dispatch is. Named `stage` because
    // `pass` is a WGSL reserved word.
    stage: u32,
    // Samples behind every pixel, as a divisor. One number rather than a read of
    // `accum[i].w` per tap: the dispatch covers the whole image every sample, so
    // every pixel carries exactly the same count.
    samples: f32,
    sigma_normal: f32,
    sigma_depth: f32,
    sigma_luminance: f32,
}

@group(0) @binding(0) var<uniform> denoise: Denoise;
// The tracer's output, read-only from here: summed radiance with its sample
// count, the luminance moments, and the two feature buffers.
@group(0) @binding(1) var<storage> accum: array<vec4f>;
//
// Both structs are `shader.wgsl`'s, and are documented there. Only `kept` is read
// here: the noise the filter has to explain is the noise the image kept.
struct Moments {
    drawn: vec2f,
    kept: vec2f,
}
struct Albedo {
    color: vec3f,
    votes: u32,
}
@group(0) @binding(2) var<storage> moments: array<Moments>;
@group(0) @binding(3) var<storage> normals: array<vec4f>;
@group(0) @binding(4) var<storage> albedos: array<Albedo>;

// The ping-pong pair, swapped between every dispatch. `xyz` is irradiance — the
// radiance with the surface's own colour divided out — and `w` is the variance
// of that irradiance, filtered alongside it so each pass knows how much noise
// the one before it left behind.
@group(1) @binding(0) var<storage> source: array<vec4f>;
@group(1) @binding(1) var<storage, read_write> destination: array<vec4f>;

// Rec. 709 luma, the same weights `shader.wgsl` accumulated the moments with.
fn luminance(color: vec3f) -> f32 {
    return dot(color, vec3f(0.2126, 0.7152, 0.0722));
}

fn index_of(x: u32, y: u32) -> u32 {
    return y * denoise.width + x;
}

// The surface a pixel is filed under, averaged back out of the sums the tracer
// left. Two loads, and every weight below is built from them.
struct Surface {
    // Unit length, or exactly zero for a pixel whose path hit nothing.
    normal: vec3f,
    depth: f32,
    albedo: vec3f,
    // The id most of the pixel's samples saw, plus one — the vote's candidate,
    // compared as it stands rather than decoded, since only equality is asked.
    object: u32,
}

fn surface_at(index: u32) -> Surface {
    let inverse = 1.0 / denoise.samples;
    let geometry = normals[index];
    let material = albedos[index];

    // Renormalised, unlike the AOV image the same buffer writes out. The weight
    // below asks whether two pixels face the same way, and an averaged normal is
    // the best estimate of that direction — but it is shorter than one at an
    // anti-aliased edge, and `pow(dot, 128)` on a pair of them would reject the
    // two halves of every silhouette in the frame from each other.
    //
    // A zero-length one is a pixel that hit nothing, and stays zero: it is the
    // signal the filter uses to leave the background alone.
    var normal = vec3f(0.0);
    let length2 = dot(geometry.xyz, geometry.xyz);
    if length2 > EPSILON {
        normal = geometry.xyz * inverseSqrt(length2);
    }

    return Surface(normal, geometry.w * inverse, material.color * inverse, material.votes >> 16u);
}

// The albedo the radiance is divided by on the way in and multiplied by on the
// way out. Clamped, and the *same* clamp both times, so a surface too dark to
// demodulate still round-trips to exactly what it started as.
fn demodulator(albedo: vec3f) -> vec3f {
    return max(albedo, vec3f(ALBEDO_FLOOR));
}

// The variance of a pixel's *estimate*, demodulated to match the irradiance the
// filter compares.
//
// Two divisions by the sample count, and both are load-bearing. The first is the
// variance of one sample, `E[x^2] - E[x]^2`; the second turns it into the
// variance of the mean of `n` of them, which is what the pixel actually holds.
// Skipping it would hand the luminance weight a threshold `sqrt(n)` too generous
// — at 500 samples, twenty-two times — and the filter would blur straight
// through every edge it was built to keep.
//
// The demodulation is by the *squared* albedo luminance, because the irradiance
// it has to be comparable against was divided by the albedo once and variance
// scales with the square. Getting this backwards has a diagnostic symptom: the
// white surfaces filter cleanly and every coloured one stays noisy in proportion
// to how dark it is.
fn variance_at(index: u32) -> f32 {
    let inverse = 1.0 / denoise.samples;
    let moment = moments[index].kept * inverse;
    let mean = moment.x;

    // Floating point returns a small negative here on a converged pixel, where
    // the two terms are nearly equal. The clamp is not optional.
    let variance = max(0.0, moment.y - mean * mean) * inverse;

    let albedo = luminance(demodulator(albedos[index].color * inverse));

    return variance / max(albedo * albedo, EPSILON);
}

// Builds the filter's input: the demodulated irradiance, and a variance that has
// been smoothed before anything is allowed to trust it.
//
// The 3x3 prefilter is the correction that matters most here. Plenty of pixels
// come back with a variance of exactly zero — at 500 samples a flat, well-lit
// surface easily agrees with itself — and a zero variance makes the luminance
// weight reject every neighbour, so the pixel keeps whatever value it happened
// to land on and the frame comes out speckled with survivors. Nine taps once,
// before the first iteration, and they are gone.
fn prepared(x: u32, y: u32, index: u32) -> vec4f {
    let inverse = 1.0 / denoise.samples;
    let radiance = accum[index].xyz * inverse;
    let albedo = demodulator(albedos[index].color * inverse);

    var variance = 0.0;
    var total = 0.0;
    for (var dy = -1; dy <= 1; dy += 1) {
        for (var dx = -1; dx <= 1; dx += 1) {
            let tap = tap_at(x, y, dx, dy);
            if !tap.inside {
                continue;
            }

            // 1 2 1 / 2 4 2 / 1 2 1, carried as the separable pair it factors
            // into, so the corners, edges and centre fall out of one product.
            let weight = select(1.0, 2.0, dx == 0) * select(1.0, 2.0, dy == 0);
            variance += weight * variance_at(tap.index);
            total += weight;
        }
    }

    return vec4f(radiance / albedo, variance / max(total, EPSILON));
}

struct Tap {
    index: u32,
    inside: bool,
}

// A neighbour `dx, dy` *pixels* away. The offset is raw and never scaled here:
// `filtered` multiplies by the pass's stride before it calls this, while the 3x3
// prefilter and the one-pixel depth gradient deliberately do not, because
// neither is supposed to widen as the passes do.
//
// Taps that fall off the frame are dropped and the weights renormalised around
// them, rather than clamped to the edge — clamping would let a border pixel vote
// several times and pull the frame's edge toward itself.
fn tap_at(x: u32, y: u32, dx: i32, dy: i32) -> Tap {
    let tx = i32(x) + dx;
    let ty = i32(y) + dy;
    if tx < 0 || ty < 0 || tx >= i32(denoise.width) || ty >= i32(denoise.height) {
        return Tap(0u, false);
    }

    return Tap(index_of(u32(tx), u32(ty)), true);
}

// How fast depth is changing across the surface under this pixel, as a central
// difference over the depth AOV.
//
// The article this follows reads `ddx`/`ddy` off a rasterised G-buffer, which
// this renderer does not have and cannot get: there is no fragment shader and no
// derivatives over a storage buffer. So it is a finite difference, computed once
// per pixel per pass rather than once per tap.
//
// It is also a gradient of a distance from the camera rather than of a view-space
// z, so it carries a little of the ray-direction spread as well as the surface
// slope. At these fields of view that is small and concentrated at the frame
// edges; if it ever shows, the fix is to store the projected depth in stage 2
// instead, which changes nothing here. What it is *not* any more is a gradient of
// one segment: `Features.depth` is summed along the path, so it stays continuous
// across a refraction and the difference below stays meaningful there.
fn depth_gradient(x: u32, y: u32) -> vec2f {
    let inverse = 1.0 / denoise.samples;
    let depth = normals[index_of(x, y)].w * inverse;

    let left = tap_at(x, y, -1, 0);
    let right = tap_at(x, y, 1, 0);
    let up = tap_at(x, y, 0, -1);
    let down = tap_at(x, y, 0, 1);

    // One-sided at the border, where the other neighbour does not exist.
    let west = select(depth, normals[left.index].w * inverse, left.inside);
    let east = select(depth, normals[right.index].w * inverse, right.inside);
    let north = select(depth, normals[up.index].w * inverse, up.inside);
    let south = select(depth, normals[down.index].w * inverse, down.inside);

    return vec2f(abs(east - west) * 0.5, abs(south - north) * 0.5);
}

// One a-trous iteration.
fn filtered(x: u32, y: u32, index: u32) -> vec4f {
    let centre = source[index];
    let here = surface_at(index);
    let gradient = depth_gradient(x, y);

    // The luminance weight's threshold, fixed for the whole neighbourhood: how
    // much of a difference this pixel's own noise could account for.
    let deviation = denoise.sigma_luminance * sqrt(max(centre.w, 0.0)) + EPSILON;
    let luma = luminance(centre.xyz);

    // The centre tap, which every edge-stopping term would score at one by
    // definition — and which the normal term would score at *zero* on a
    // background pixel, whose normal is zero and cannot be dotted with itself to
    // anything. Seeding the sums with it is what leaves those pixels alone
    // rather than dividing them by nothing.
    var colour = KERNEL[2] * KERNEL[2] * centre.xyz;
    var variance = KERNEL[2] * KERNEL[2] * KERNEL[2] * KERNEL[2] * centre.w;
    var total = KERNEL[2] * KERNEL[2];

    for (var dy = -2; dy <= 2; dy += 1) {
        for (var dx = -2; dx <= 2; dx += 1) {
            if dx == 0 && dy == 0 {
                continue;
            }

            let offset = vec2i(dx, dy) * i32(denoise.stride);
            let tap = tap_at(x, y, offset.x, offset.y);
            if !tap.inside {
                continue;
            }

            let there = surface_at(tap.index);
            let neighbour = source[tap.index];

            // Do the two pixels face the same way? An exponent rather than a
            // falloff because the interesting range of a dot product between two
            // normals is the last hundredth of it.
            let facing = pow(max(0.0, dot(here.normal, there.normal)), denoise.sigma_normal);

            // Are they the same distance away, allowing for the fact that a
            // sloped surface is *supposed* to change depth across the gap? The
            // gradient is scaled by how far this tap reaches, which is what stops
            // the wider passes from rejecting everything they can see.
            let expected = abs(gradient.x * f32(offset.x)) + abs(gradient.y * f32(offset.y));
            let near = exp(-abs(here.depth - there.depth) / (denoise.sigma_depth * expected + EPSILON));

            // Is the difference in brightness bigger than this pixel's noise can
            // explain? This is the term that does the denoising; the other three
            // only decide where it is allowed to act.
            let similar = exp(-abs(luma - luminance(neighbour.xyz)) / deviation);

            // And are they even the same object? Voted rather than averaged, so
            // a pixel straddling an edge is filed under whichever surface most
            // of its samples saw rather than an id between the two. This is what
            // keeps an emitter from bleeding: its variance is near zero, so the
            // luminance weight would happily pull its neighbours into it, and
            // only the identity stops that.
            let same = select(0.0, 1.0, here.object == there.object);

            let weight = KERNEL[dx + 2] * KERNEL[dy + 2] * facing * near * similar * same;
            colour += weight * neighbour.xyz;
            // Squared, because variance is a second moment: a weighted sum of
            // independent estimates has the *squares* of those weights on its
            // variance, and getting this right is what tells the next, wider
            // pass how much noise this one actually removed.
            variance += weight * weight * neighbour.w;
            total += weight;
        }
    }

    let normalise = max(total, EPSILON);

    return vec4f(colour / normalise, variance / (normalise * normalise));
}

// Puts the surface colour back, and shapes the result like the accumulator so
// the host resolves it with exactly the code that resolves an unfiltered frame:
// a sample count of one, and the averaging divide that follows lands on it.
fn remodulated(index: u32) -> vec4f {
    // `* inverse` rather than `/ denoise.samples`, matching `prepared` and
    // `variance_at` exactly. The two differ in the last ulp for most sample
    // counts — `1.0 / samples` is itself rounded — and this is the half of the
    // round trip's error that costs nothing to remove.
    let inverse = 1.0 / denoise.samples;
    let albedo = demodulator(albedos[index].color * inverse);

    return vec4f(source[index].xyz * albedo, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn denoise_pass(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= denoise.width || id.y >= denoise.height {
        return;
    }

    let index = index_of(id.x, id.y);
    switch denoise.stage {
        case PREPARE: {
            destination[index] = prepared(id.x, id.y, index);
        }
        case REMODULATE: {
            destination[index] = remodulated(index);
        }
        case FILTER, default: {
            destination[index] = filtered(id.x, id.y, index);
        }
    }
}
