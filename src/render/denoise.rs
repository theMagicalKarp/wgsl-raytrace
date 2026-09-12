use crate::config::DenoiseOptions;
use bytemuck::Pod;
use bytemuck::Zeroable;

/// Which of the shader's three passes a dispatch is. Matches the constants at
/// the top of `denoise.wgsl`.
const PREPARE: u32 = 0;
const FILTER: u32 = 1;
const REMODULATE: u32 = 2;

/// What one dispatch of the filter needs to know, as one uniform block.
///
/// Rewritten between dispatches the way `GpuCamera`'s sample index is, and for
/// the same reason: the passes differ only in `pass` and `stride`, so rebuilding
/// the block is cheaper than binding a second one.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuDenoise {
    pub width: u32,
    pub height: u32,
    /// Pixels between adjacent taps, doubling every iteration.
    pub stride: u32,
    /// Which of the three passes above this dispatch is. Named to match the
    /// shader, where `pass` is a reserved word.
    pub stage: u32,

    /// Samples behind every pixel, as a divisor for the sums the tracer left.
    pub samples: f32,
    pub sigma_normal: f32,
    pub sigma_depth: f32,
    pub sigma_luminance: f32,
}

const _: () = assert!(size_of::<GpuDenoise>() == 32);

impl GpuDenoise {
    pub fn new(options: &DenoiseOptions, (width, height): (u32, u32), samples: u32) -> GpuDenoise {
        GpuDenoise {
            width,
            height,
            stride: 1,
            stage: PREPARE,
            // A render interrupted before its first sample would divide by zero
            // here, and the frame it is filtering is black either way.
            samples: samples.max(1) as f32,
            sigma_normal: options.sigma_normal,
            sigma_depth: options.sigma_depth,
            sigma_luminance: options.sigma_luminance,
        }
    }
}

/// A filtered frame, and what it cost.
///
/// The filter is a fixed cost after the sample loop rather than a per-dispatch
/// one, so it is reported as wall time — the `gpu:` line's per-dispatch figure
/// would divide it by a sample count it has nothing to do with.
pub struct Denoised {
    pub image: super::Image,
    pub elapsed: std::time::Duration,
}

/// The passes to run, in order, as `(pass, stride)` pairs.
///
/// The stride doubling is the whole reason five passes of a 5x5 kernel are worth
/// running: pass `i` spaces its taps `2^i` pixels apart, so it spreads
/// information `2 * 2^i` in each direction and the five together reach
/// `2 * 2 * (1 + 2 + 4 + 8 + 16) + 1` — 125 pixels — for the cost of 125 taps.
/// The last pass alone spans 65 of those; the reach is cumulative and belongs to
/// the schedule rather than to any one dispatch in it.
///
/// `iterations` of zero is a legal thing to ask for and produces a prepare and a
/// remodulate with nothing in between — which is the unfiltered frame, routed
/// through the demodulation and back. That it comes out unchanged is worth being
/// able to test.
///
/// Anything past [`crate::config::MAX_ITERATIONS`] is refused by
/// [`crate::config::Config::validate`] rather than clamped here, so the shift
/// guard below is a second line and not the only one.
pub fn schedule(iterations: u32) -> Vec<(u32, u32)> {
    let mut passes = vec![(PREPARE, 1)];
    passes.extend((0..iterations).map(|iteration| (FILTER, 1u32 << iteration.min(16))));
    passes.push((REMODULATE, 1));

    passes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stride_doubles_between_iterations() {
        let strides: Vec<u32> = schedule(5)
            .iter()
            .filter(|(pass, _)| *pass == FILTER)
            .map(|(_, stride)| *stride)
            .collect();

        assert_eq!(strides, [1, 2, 4, 8, 16]);
    }

    /// A prepare and a remodulate always run, so the demodulation round-trip is
    /// exercised even when nothing is being filtered between them.
    #[test]
    fn a_schedule_is_bracketed_by_its_two_end_passes() {
        assert_eq!(schedule(0), [(PREPARE, 1), (REMODULATE, 1)]);

        let passes = schedule(3);
        assert_eq!(passes.first(), Some(&(PREPARE, 1)));
        assert_eq!(passes.last(), Some(&(REMODULATE, 1)));
        assert_eq!(passes.len(), 5);
    }

    /// The shift that builds the stride must not run off the end of a `u32`. A
    /// config cannot ask for this many — `validate` stops it well short — but
    /// the arithmetic holds anyway.
    #[test]
    fn an_absurd_iteration_count_still_produces_a_stride() {
        let passes = schedule(64);

        assert_eq!(passes.len(), 66);
        assert!(passes.iter().all(|(_, stride)| *stride > 0), "{passes:?}");
    }

    #[test]
    fn the_uniform_carries_the_scenes_sigmas() {
        let options = DenoiseOptions::default();
        let uniform = GpuDenoise::new(&options, (800, 600), 500);

        assert_eq!((uniform.width, uniform.height), (800, 600));
        assert_eq!(uniform.samples, 500.0);
        assert_eq!(uniform.sigma_normal, 128.0);
        assert_eq!(uniform.stage, PREPARE, "a schedule starts by demodulating");
    }

    /// An interrupted preview can resolve with nothing accumulated, and the
    /// shader divides by this.
    #[test]
    fn a_render_with_no_samples_still_divides_by_something() {
        let uniform = GpuDenoise::new(&DenoiseOptions::default(), (8, 8), 0);

        assert_eq!(uniform.samples, 1.0);
    }
}
