//! Per-pixel variance, recovered from the moments the shader accumulates.
//!
//! The shader sums each pixel's sample luminances and their squares
//! (`shader.wgsl`'s `moments`), which is all it takes to get a variance back on
//! the host without keeping a single sample around: `E[x^2] - E[x]^2`. Two
//! things want that number. The a-trous filter wants it per pixel, to know how
//! much of a luminance difference between two neighbours is signal and how much
//! is noise. A person at a terminal wants it summarised, to answer "is this
//! render noisier than the last one?" without rendering a reference and
//! diffing against it.
//!
//! The filter reads its copy on the GPU and never through here, so what this
//! module hands back is the summary — and, when a run asked for one,
//! [`Statistics::heatmap`], which is the per-pixel field made something you can
//! look at.

use super::Image;
use std::fmt;

/// What one render's moments say about its noise.
///
/// Built once at the end of a render, from a buffer the shader has been
/// accumulating alongside the radiance all the way through.
pub struct Statistics {
    /// Samples behind every pixel, which is what turns a variance into a
    /// standard error. Equal for every pixel: the dispatch is one pass over the
    /// whole image and every pixel is in it.
    samples: u32,
    /// The variance field as an image — `None` unless the run asked for one.
    ///
    /// Built here rather than kept as a `Vec<f32>` for a caller to render later,
    /// and for the same reason `Render::aovs` is an `Option`: the field is four
    /// bytes a pixel, which is ~33MB on a 4K frame, and every figure above is
    /// already out of it by the time this is decided. A run that only wants a
    /// picture should not be holding a second frame it will never look at.
    pub heatmap: Option<Image>,
    /// Mean per-pixel standard error, `sqrt(variance / n)` averaged over the
    /// frame. **This is the figure a filter is judged by** — it falls as the
    /// render converges and it is comparable between two runs of the same scene
    /// at the same sample count.
    ///
    /// It is not the figure a *sampler* is judged by, and the distinction is
    /// worth keeping straight: a stratified or low-discrepancy sequence draws
    /// from the same distribution an independent one does, so it converges
    /// faster without this number moving at all. Anything touching the sampler
    /// still has to be measured as error against a reference.
    pub standard_error: f32,
    /// Mean scene luminance, in linear units.
    ///
    /// The bias probe, and the reason it is quoted in linear rather than read
    /// off the PNG: gamma encoding is concave, so a noisier render is
    /// systematically *brighter* once encoded and a signed difference taken on
    /// two images measures that on top of whatever it was actually looking for.
    pub luminance: f32,
    /// The largest per-pixel variance in the frame, which is both what
    /// [`Statistics::heatmap`] normalises against and the first sign of a
    /// firefly — an isolated pixel orders of magnitude above the mean.
    pub peak: f32,
}

impl Statistics {
    /// Recovers the variance field from the moments the shader accumulated.
    ///
    /// `moments` is `shader.wgsl`'s `Moments` per pixel: `[sum of luminances,
    /// sum of their squares]` over every sample as drawn, then the same pair
    /// over what the accumulator kept. Only the second is read — the figures
    /// here describe the image, and a firefly the rejection already capped is
    /// not in it. `samples` is how many went into each, so both means are a
    /// divide away.
    /// The `max` on the variance is not optional: `E[x^2] - E[x]^2` is a
    /// difference of two nearly equal numbers on a converged pixel, and floating
    /// point hands back a small negative often enough that a `sqrt` of it would
    /// be a routine source of NaNs rather than a rare one.
    ///
    /// `heatmap` says whether the per-pixel field is wanted as an image. The
    /// summary below never needs it kept — it is accumulated as the field is
    /// recovered — so a run without `--variance` allocates nothing for it.
    pub fn new(
        moments: &[[f32; 4]],
        (width, height): (u32, u32),
        samples: u32,
        heatmap: bool,
    ) -> Statistics {
        let count = 1.0 / samples.max(1) as f32;
        let mut variance = Vec::with_capacity(match heatmap {
            true => moments.len(),
            false => 0,
        });
        let mut total_error = 0.0f64;
        let mut total_luminance = 0.0f64;
        let mut peak = 0.0f32;

        for moment in moments {
            let mean = moment[2] * count;
            let pixel = (moment[3] * count - mean * mean).max(0.0);

            total_error += (pixel * count).sqrt() as f64;
            total_luminance += mean as f64;
            peak = peak.max(pixel);
            if heatmap {
                variance.push(pixel);
            }
        }

        let pixels = moments.len().max(1) as f64;

        Statistics {
            samples,
            // After the loop, because the ramp is normalised against a peak that
            // is only known once every pixel has been seen.
            heatmap: heatmap.then(|| heat(&variance, peak, (width, height))),
            standard_error: (total_error / pixels) as f32,
            luminance: (total_luminance / pixels) as f32,
            peak,
        }
    }
}

/// The variance field as an image, for looking at rather than for measuring.
///
/// Normalised against `peak` and square-rooted, which is the only way one frame
/// can show both a firefly and the ordinary noise underneath it: a firefly is
/// orders of magnitude above the mean, and a linear ramp against it leaves the
/// entire rest of the image black. The scale is therefore relative, and two
/// heatmaps are only comparable if the `peak` quoted beside them is — which is
/// why [`fmt::Display`] prints it.
fn heat(variance: &[f32], peak: f32, (width, height): (u32, u32)) -> Image {
    let scale = match peak > 0.0 {
        true => 1.0 / peak,
        false => 0.0,
    };

    let pixels = variance
        .iter()
        .flat_map(|&pixel| {
            let [red, green, blue] = ramp((pixel * scale).sqrt());
            [red, green, blue, 255]
        })
        .collect();

    Image {
        width,
        height,
        pixels,
    }
}

/// Black through purple and red to a pale yellow, lerped between five stops.
///
/// A dark-to-bright ramp rather than a hue wheel, so that "how noisy" reads off
/// the image without a legend and the quiet parts of the frame stay out of the
/// way of the loud ones.
fn ramp(t: f32) -> [u8; 3] {
    const STOPS: [[f32; 3]; 5] = [
        [0.0, 0.0, 0.0],
        [60.0, 15.0, 110.0],
        [165.0, 45.0, 95.0],
        [235.0, 120.0, 30.0],
        [250.0, 250.0, 190.0],
    ];

    let position = t.clamp(0.0, 1.0) * (STOPS.len() - 1) as f32;
    let index = (position as usize).min(STOPS.len() - 2);
    let fraction = position - index as f32;

    let (low, high) = (STOPS[index], STOPS[index + 1]);
    [0, 1, 2].map(|channel| {
        let value = low[channel] + (high[channel] - low[channel]) * fraction;
        (value + 0.5) as u8
    })
}

impl fmt::Display for Statistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.5} standard error/pixel  ·  {:.4} mean luminance (linear)  ·  \
             peak variance {:.3} over {} samples",
            self.standard_error, self.luminance, self.peak, self.samples,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One pixel whose samples never disagreed. The variance is exactly zero and
    /// so is the standard error, and the mean is the sample value.
    ///
    /// On a one-pixel frame `peak` *is* that pixel's variance, which is how these
    /// tests read a field the struct no longer keeps.
    #[test]
    fn a_converged_pixel_has_no_variance() {
        let stats = Statistics::new(&kept(&[[4.0, 4.0]]), (1, 1), 4, false);

        assert_eq!(stats.luminance, 1.0, "four samples of one");
        assert_eq!(stats.standard_error, 0.0);
        assert_eq!(stats.peak, 0.0);
    }

    /// Two samples, 0.0 and 2.0: a mean of 1.0 and a variance of 1.0, so the
    /// standard error is `sqrt(1 / 2)`.
    #[test]
    fn variance_is_the_second_moment_less_the_first_squared() {
        let stats = Statistics::new(&kept(&[[2.0, 4.0]]), (1, 1), 2, false);

        assert_eq!(stats.luminance, 1.0);
        assert_eq!(stats.peak, 1.0);
        assert!(
            (stats.standard_error - 0.5f32.sqrt()).abs() < 1e-6,
            "{stats}"
        );
    }

    /// The clamp that keeps a `sqrt` off a negative. `E[x^2] - E[x]^2` is a
    /// difference of two nearly equal numbers on a converged pixel, and this is
    /// what that rounds to when it goes the wrong way.
    #[test]
    fn a_negative_variance_is_floored_rather_than_propagated() {
        let stats = Statistics::new(&kept(&[[2.0, 0.999999]]), (1, 1), 2, false);

        assert_eq!(stats.peak, 0.0);
        assert!(stats.standard_error.is_finite(), "{stats}");
    }

    /// The heatmap costs a frame's worth of allocation, so a run that did not
    /// ask for one gets nothing — the summary above is built either way.
    #[test]
    fn the_field_is_only_kept_when_something_wants_to_look_at_it() {
        let moments = kept(&[[2.0, 4.0], [20.0, 400.0]]);

        assert!(
            Statistics::new(&moments, (2, 1), 2, false)
                .heatmap
                .is_none()
        );
        assert!(Statistics::new(&moments, (2, 1), 2, true).heatmap.is_some());
    }

    /// A render that was interrupted before its first sample landed. Every
    /// figure is zero and none of them is a NaN.
    #[test]
    fn a_render_with_no_samples_divides_by_one_instead() {
        let stats = Statistics::new(&kept(&[[0.0, 0.0]]), (1, 1), 0, true);

        assert_eq!(stats.luminance, 0.0);
        assert_eq!(stats.standard_error, 0.0);
        assert_eq!(heat_of(&stats).pixels, [0, 0, 0, 255]);
    }

    /// Moments whose kept pair is `pairs` and whose drawn pair is a NaN, so
    /// that a figure read off the wrong one fails every test it reaches.
    fn kept(pairs: &[[f32; 2]]) -> Vec<[f32; 4]> {
        pairs
            .iter()
            .map(|&[sum, squares]| [f32::NAN, f32::NAN, sum, squares])
            .collect()
    }

    /// The firefly rejection's scaled-back samples are not noise the image
    /// has, so a pixel whose drawn samples disagree and whose kept ones do not
    /// reports no noise at all.
    #[test]
    fn the_noise_is_what_the_accumulator_kept() {
        let stats = Statistics::new(&[[1000.5, 1_000_000.25, 1.0, 0.5]], (1, 1), 2, false);

        assert_eq!(stats.luminance, 0.5);
        assert_eq!(stats.peak, 0.0);
        assert_eq!(stats.standard_error, 0.0);
    }

    /// The heatmap of a `Statistics` that was built with one.
    fn heat_of(stats: &Statistics) -> &Image {
        stats
            .heatmap
            .as_ref()
            .expect("statistics built with a heatmap should carry one")
    }

    /// The whole point of the square root in the heatmap: a firefly two orders
    /// of magnitude above its neighbours must not black out the rest of the
    /// frame. Linearly the quiet pixel would be 1/100th of the way up the ramp;
    /// rooted it is a tenth, which is a visible colour.
    #[test]
    fn the_heatmap_ramps_from_the_frames_own_peak() {
        let stats = Statistics::new(&kept(&[[2.0, 4.0], [20.0, 400.0]]), (2, 1), 2, true);
        let heat = heat_of(&stats);

        assert_eq!(heat.pixels.len(), 8);
        assert_eq!(
            &heat.pixels[4..8],
            &[250, 250, 190, 255],
            "the peak is the top"
        );
        assert_ne!(
            &heat.pixels[0..3],
            &[0, 0, 0],
            "and the quiet pixel is not black"
        );
    }

    #[test]
    fn the_ramp_spans_its_stops() {
        assert_eq!(ramp(0.0), [0, 0, 0]);
        assert_eq!(ramp(1.0), [250, 250, 190]);
        assert_eq!(
            ramp(2.0),
            ramp(1.0),
            "out of range clamps rather than wraps"
        );
        assert_eq!(ramp(-1.0), ramp(0.0));
        assert_eq!(ramp(0.25), [60, 15, 110], "a stop lands exactly on itself");
    }
}
