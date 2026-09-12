//! The curve between the radiance the tracer computed and the eight bits a PNG
//! can hold.
//!
//! There is more light in a scene than a display can show — melee's emitter is
//! at 15.0, where white is 1.0 — so something has to decide what happens to the
//! rest of it. Clamping is the decision the renderer used to make by default,
//! and it is the worst one available: everything within a few units of the light
//! lands on 255 and stops carrying information, so a filtered frame and a noisy
//! one look identical across the whole bright half of the image, and a firefly
//! forty times too bright is the same white pixel as one four times too bright.
//!
//! A filmic curve makes the other decision. It compresses the highlights into
//! the top of the range instead of discarding them, which costs some contrast in
//! the midtones and buys back every distinction that used to be clipped away.
//!
//! **Order matters.** This runs on averaged linear radiance and nothing but the
//! gamma encode follows it — the denoiser's edge-stopping weights compare
//! luminances, and a concave curve compresses the bright end, so a filter fed
//! tone-mapped values would under-weight precisely the high-contrast neighbours
//! it exists to reject. Everything upstream of here is linear and stays linear.

/// The ceiling a channel is held under before the curve sees it.
///
/// The curve is a ratio of two quadratics, so an infinity divides an infinity by
/// an infinity and comes back a NaN — which would resolve to black, making the
/// brightest possible pixel indistinguishable from the darkest. Anything at all
/// past the curve's saturation point maps to white regardless, so pinning the
/// input somewhere far above it costs no real value and turns the one case that
/// breaks into the one that is obviously right.
const CEILING: f32 = 1e6;

/// sRGB into ACEScg (AP1), where the curve is applied.
///
/// Applying a tone curve per channel in the display's own primaries pulls hues
/// around as one channel saturates before its neighbours — a bright orange goes
/// yellow on the way to white. AP1 is wide enough that the same per-channel
/// curve keeps the hue, which is the whole reason for the round trip. Every
/// entry is positive and every row sums to one, so a neutral grey stays neutral
/// and nothing here can hand the curve a negative.
const INTO_AP1: [[f32; 3]; 3] = [
    [0.59719, 0.35458, 0.04823],
    [0.07600, 0.90834, 0.01566],
    [0.02840, 0.13383, 0.83777],
];

/// AP1 back to sRGB. The inverse of [`INTO_AP1`], so the pair cancels wherever
/// the curve between them does nothing.
const FROM_AP1: [[f32; 3]; 3] = [
    [1.60475, -0.53108, -0.07367],
    [-0.10208, 1.10813, -0.00605],
    [-0.00327, -0.07276, 1.07602],
];

/// Stephen Hill's fit to the ACES reference rendering and output transforms,
/// applied per channel in AP1.
///
/// A ratio of two quadratics chosen to follow the real transform closely enough
/// for a renderer, at a cost that is four multiplies rather than a lookup table
/// and a colour science library. It rises steeply out of black, rolls off toward
/// a horizontal asymptote a little above one, and is monotonic throughout.
fn curve(value: f32) -> f32 {
    let numerator = value * (value + 0.0245786) - 0.000090537;
    let denominator = value * (0.983729 * value + 0.432951) + 0.238081;

    numerator / denominator
}

/// `value` held inside `[low, high]`, with a NaN coming out as `low`.
///
/// Deliberately not [`f32::clamp`], which is specified to return a NaN it is
/// given — and a NaN reaching the encode would become whatever `as u8` makes of
/// one rather than a pixel. [`f32::max`] and [`f32::min`] hand back whichever
/// operand is *not* a NaN, which is the behaviour wanted here and the reason
/// clippy's suggestion to collapse this into a `clamp` is refused.
#[allow(clippy::manual_clamp)]
fn bound(value: f32, low: f32, high: f32) -> f32 {
    value.max(low).min(high)
}

fn transform(matrix: &[[f32; 3]; 3], color: [f32; 3]) -> [f32; 3] {
    matrix.map(|row| row[0] * color[0] + row[1] * color[1] + row[2] * color[2])
}

/// Linear scene radiance to a display-referred colour in `[0, 1]`.
///
/// Still linear on the way out — the gamma encode is [`resolve`](super::resolve)'s
/// job and happens after this, not instead of it.
pub fn aces(color: [f32; 3]) -> [f32; 3] {
    // The floor is also what turns a poisoned channel into black rather than
    // propagating it — see [`bound`]. The shader's own guard should mean one
    // never arrives; this is the other half of that guarantee.
    let bounded = color.map(|channel| bound(channel, 0.0, CEILING));

    let curved = transform(&INTO_AP1, bounded).map(curve);

    // The curve dips a ten-thousandth below zero at an input of zero, and the
    // matrix above can carry that through, so black needs the floor as much as
    // white needs the ceiling.
    transform(&FROM_AP1, curved).map(|channel| bound(channel, 0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn black_stays_black_and_is_not_lifted() {
        assert_eq!(aces([0.0; 3]), [0.0; 3]);
    }

    /// The two matrices are inverses and every row of each sums to one, so a
    /// colour with no hue cannot acquire one on the way through.
    #[test]
    fn a_neutral_grey_comes_back_neutral() {
        for level in [0.05, 0.18, 0.5, 1.0, 4.0] {
            let [red, green, blue] = aces([level; 3]);

            assert!((red - green).abs() < 1e-5, "{level}: {red} vs {green}");
            assert!((green - blue).abs() < 1e-5, "{level}: {green} vs {blue}");
        }
    }

    /// The property the whole stage rests on: brighter in is brighter out, all
    /// the way up. A curve that flattened would clip exactly the way the clamp
    /// it replaces did.
    #[test]
    fn the_curve_is_monotonic_across_the_range_that_matters() {
        let mut previous = -1.0;
        for step in 0..2000 {
            let level = step as f32 * 0.05;
            let mapped = aces([level; 3])[0];

            assert!(
                mapped >= previous,
                "{level} mapped to {mapped} < {previous}"
            );
            previous = mapped;
        }
    }

    /// The point of the exercise. Under the old clamp everything at or above one
    /// was the same white; here a light at 15 is still distinguishable from a
    /// surface at 1.
    #[test]
    fn values_above_one_keep_their_differences() {
        let one = aces([1.0; 3])[0];
        let four = aces([4.0; 3])[0];
        let fifteen = aces([15.0; 3])[0];

        assert!(one < 1.0, "one is not the top of the range any more: {one}");
        assert!(one < four, "{one} vs {four}");
        assert!(four < fifteen, "{four} vs {fifteen}");
    }

    /// Not something the shader's non-finite guard should ever let through, and
    /// asserted anyway because the two cases go opposite ways: a NaN is not a
    /// brightness at all and reads as black, while an infinity is the brightest
    /// thing there is and reads as white.
    #[test]
    fn a_non_finite_channel_resolves_rather_than_propagating() {
        assert_eq!(aces([f32::NAN; 3]), [0.0; 3]);
        assert_eq!(aces([f32::INFINITY; 3]), [1.0; 3]);
        assert_eq!(aces([-1.0; 3]), [0.0; 3], "and so does a negative");
    }

    /// A saturated colour bleaching toward white as it gets brighter is what
    /// film does and what the AP1 round trip is there to reproduce. A per-channel
    /// curve in sRGB would swing the hue instead.
    #[test]
    fn a_bright_saturated_colour_desaturates_rather_than_shifting_hue() {
        let dim = aces([1.0, 0.2, 0.2]);
        let bright = aces([60.0, 12.0, 12.0]);

        let spread = |color: [f32; 3]| color[0] - color[1];
        assert!(
            spread(bright) < spread(dim),
            "{bright:?} should be closer to neutral than {dim:?}",
        );
    }
}
