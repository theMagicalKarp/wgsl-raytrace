use image::DynamicImage;
use std::error::Error;
use std::path::Path;

/// The largest 2D texture a default `wgpu::Limits` device promises. An 8K HDRI
/// is exactly this wide; a 16K one is a clear error rather than a panic from
/// somewhere inside the driver.
const MAX_DIMENSION: u32 = 8192;

/// The largest finite value `f16` can hold, which is what the texture is
/// uploaded as. An HDRI's sun can exceed it, and a texel that overflows to
/// infinity poisons every sample that draws it.
const MAX_HALF: f32 = 65504.0;

/// An equirectangular sky, decoded to linear RGB and ready to upload. Rows run
/// from the +Y pole down and columns wrap the full turn — the convention every
/// HDRI ships in, and what the shader's `environment_radiance` assumes.
pub struct Environment {
    pub width: u32,
    pub height: u32,
    /// Row-major RGBA. Alpha is carried to fill the format's fourth channel and
    /// never read.
    pub texels: Vec<[f32; 4]>,
}

/// Its shape and nothing else: `Scene` derives `Debug`, and a megapixel of
/// texels in an error message helps nobody.
impl std::fmt::Debug for Environment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Environment({}x{})", self.width, self.height)
    }
}

/// The sRGB electro-optical transfer function.
///
/// `image` converts an 8-bit channel to `f32` by dividing by 255 — it normalizes
/// a range, it does not decode a colorspace. Skipping this leaves a JPEG sky
/// washed out and, since the sky is the ambient light, everything lit by it too
/// bright in the midtones.
fn srgb_to_linear(channel: f32) -> f32 {
    match channel <= 0.04045 {
        true => channel / 12.92,
        false => ((channel + 0.055) / 1.055).powf(2.4),
    }
}

/// Reads an equirectangular image as linear radiance. Float sources — OpenEXR
/// and Radiance — are already linear and pass through; integer formats hold sRGB
/// by convention and are decoded.
pub(super) fn read(path: &Path) -> Result<Environment, Box<dyn Error>> {
    let image = image::open(path).map_err(|error| format!("{}: {}", path.display(), error))?;

    let linear = matches!(
        image,
        DynamicImage::ImageRgb32F(_) | DynamicImage::ImageRgba32F(_)
    );
    let (width, height) = (image.width(), image.height());

    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(format!(
            "{}: environment map is {}x{}, larger than the {}x{} this device can hold",
            path.display(),
            width,
            height,
            MAX_DIMENSION,
            MAX_DIMENSION,
        )
        .into());
    }

    Ok(Environment {
        width,
        height,
        texels: texels(&image.to_rgba32f(), linear),
    })
}

/// Decodes and sanitizes every channel. Split out from [`read`] so the tests can
/// reach it without a file on disk.
fn texels(pixels: &image::Rgba32FImage, linear: bool) -> Vec<[f32; 4]> {
    pixels
        .pixels()
        .map(|pixel| {
            let mut texel = pixel.0;
            for channel in &mut texel[..3] {
                if !linear {
                    *channel = srgb_to_linear(*channel);
                }
                // `trace` throws away any sample whose radiance is not finite,
                // so one bad texel would silently blacken every pixel that sees
                // it. The clamp doubles as what the `f16` upload needs.
                *channel = match channel.is_finite() {
                    true => channel.clamp(0.0, MAX_HALF),
                    false => 0.0,
                };
            }
            texel
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;
    use image::Rgba32FImage;

    fn one(pixel: [f32; 4], linear: bool) -> [f32; 4] {
        let image = Rgba32FImage::from_pixel(1, 1, Rgba(pixel));
        texels(&image, linear)[0]
    }

    #[test]
    fn an_integer_source_is_decoded_out_of_srgb() {
        // Mid-grey shows it: 0.5 encoded is a little under a quarter of the
        // light, not half of it.
        let decoded = one([0.5, 0.5, 0.5, 1.0], false);

        assert!(
            (decoded[0] - 0.2140).abs() < 1e-3,
            "0.5 sRGB should be about 0.214 linear, got {}",
            decoded[0]
        );
        // Linear at the bottom and pinned at the top, which keeps black black
        // and white white.
        assert_eq!(one([0.0, 0.0, 0.0, 1.0], false)[0], 0.0);
        assert!((one([1.0, 1.0, 1.0, 1.0], false)[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_float_source_is_already_linear() {
        // And carries values well past one, which is the whole point of an HDRI.
        assert_eq!(one([0.5, 2.0, 900.0, 1.0], true)[..3], [0.5, 2.0, 900.0]);
    }

    #[test]
    fn a_non_finite_texel_becomes_black_rather_than_a_ruined_sample() {
        assert_eq!(
            one([f32::NAN, f32::INFINITY, -1.0, 1.0], true)[..3],
            [0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn a_value_past_the_half_range_is_clamped_into_it() {
        // Otherwise the `f16` upload turns it into the infinity the check above
        // exists to keep out.
        assert_eq!(one([1e30, 0.0, 0.0, 1.0], true)[0], MAX_HALF);
    }

    #[test]
    fn alpha_is_carried_through_untouched() {
        // The format has a fourth channel to fill; nothing reads it.
        assert_eq!(one([0.0, 0.0, 0.0, 0.25], false)[3], 0.25);
    }
}
