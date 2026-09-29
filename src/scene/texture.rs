//! The images a scene's patterns read, gathered into the two texture arrays the
//! shader samples.
//!
//! Two arrays rather than a texture per image, because a binding is a scarce
//! thing and a layer is not: every image a scene names is one layer of one of
//! them, whatever the scene's size. Which one is decided by how its texels are
//! decoded — `color` is `Rgba8UnormSrgb`, so the sampler hands back linear
//! light from a gamma-encoded file and filters it in linear, and `data` is
//! `Rgba8Unorm`, so a roughness or a normal comes back as the number it was
//! stored as.
//!
//! OpenEXR and Radiance files hold linear floats rather than gamma-encoded
//! bytes. One headed for `color` is encoded to sRGB before it is quantized, so
//! the sampler's decode lands back on the file's own values instead of
//! decoding them a second time; one headed for `data` is quantized as it is.
//! Either way it is clamped to [0, 1] — a layer is eight bits a channel.
//!
//! Every layer of an array is one size, which is what an array is. Each image
//! is resized to the largest one in its array, clamped to [`MAX_LAYER_SIZE`] —
//! texture coordinates are fractions of the image, so a stretch to a common
//! size is invisible to everything that samples it.

use crate::config::ColorSpace;
use image::DynamicImage;
use image::RgbaImage;
use image::imageops::FilterType;
use std::error::Error;
use std::fmt;
use std::path::Path;
use std::path::PathBuf;

/// The largest a layer is allowed to be, on either side. A 4096² layer is 64
/// MB, and a scene with a handful of them is already most of a laptop's GPU; an
/// 8K map is scaled down to it rather than refused.
pub const MAX_LAYER_SIZE: u32 = 4096;

/// Which array an image lives in, matching the shader's `TEXTURES_*`.
pub const COLOR_TEXTURES: u32 = 0;
pub const DATA_TEXTURES: u32 = 1;

/// Every image the scene's programs name, in the order they were first named,
/// once each per colour space.
///
/// The same file read two ways is two layers, because it decodes to two
/// different sets of numbers. The same file read the same way twice — a
/// colour map driving both `base_color` and a mask — is one.
#[derive(Debug, Default)]
pub struct Images {
    color: Vec<PathBuf>,
    data: Vec<PathBuf>,
}

impl Images {
    /// The array and layer `file` will be sampled from, adding it if this is
    /// the first time it has been asked for.
    pub fn add(&mut self, file: &Path, space: ColorSpace) -> (u32, u32) {
        let (array, files) = match space {
            ColorSpace::Srgb => (COLOR_TEXTURES, &mut self.color),
            ColorSpace::Linear => (DATA_TEXTURES, &mut self.data),
        };

        let layer = match files.iter().position(|known| known == file) {
            Some(layer) => layer,
            None => {
                files.push(file.to_path_buf());
                files.len() - 1
            }
        };
        (array, layer as u32)
    }

    /// Reads, decodes and resizes every image, ready to upload.
    pub fn load(&self) -> Result<Textures, Box<dyn Error>> {
        Ok(Textures {
            color: Layers::read(&self.color, ColorSpace::Srgb)?,
            data: Layers::read(&self.data, ColorSpace::Linear)?,
        })
    }
}

/// Both arrays, as the renderer uploads them.
#[derive(Debug)]
pub struct Textures {
    pub color: Layers,
    pub data: Layers,
}

/// One texture array's worth of texels: `count` layers of `width` by `height`
/// RGBA8, layer-major, rows from the top of the image down.
pub struct Layers {
    pub width: u32,
    pub height: u32,
    pub count: u32,
    pub texels: Vec<u8>,
}

/// Its shape and nothing else, as `Environment`'s is: a megabyte of texels in
/// an error message helps nobody.
impl fmt::Debug for Layers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Layers({}x{}x{})", self.width, self.height, self.count)
    }
}

impl Layers {
    /// A single white texel. An array nothing samples still has to be bound,
    /// the same fallback the light table and the sky take; the program buffer
    /// never names a layer of it.
    fn unused() -> Layers {
        Layers {
            width: 1,
            height: 1,
            count: 1,
            texels: vec![255; 4],
        }
    }

    fn read(files: &[PathBuf], space: ColorSpace) -> Result<Layers, Box<dyn Error>> {
        if files.is_empty() {
            return Ok(Layers::unused());
        }

        let images = files
            .iter()
            .map(|file| {
                image::open(file)
                    .map(|image| rgba8(image, space))
                    .map_err(|error| format!("{}: {error}", file.display()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let images: Vec<(u32, u32, Vec<u8>)> = images
            .into_iter()
            .map(|image| (image.width(), image.height(), image.into_raw()))
            .collect();
        Ok(Layers::build(images))
    }

    /// The layers of `images`, each `(width, height, rgba8)`, resized to the
    /// largest of them. Split from [`read`](Self::read) so it can be tested
    /// without files on disk.
    ///
    /// Resized in the encoded space, sRGB or not. A filter over gamma-encoded
    /// values darkens a high-contrast edge a little; it only happens to an
    /// image that is not already the layer size, and a scene built from one
    /// pack of maps has every image the same size and resizes none of them.
    fn build(images: Vec<(u32, u32, Vec<u8>)>) -> Layers {
        let width = images.iter().map(|image| image.0).max().unwrap_or(1);
        let height = images.iter().map(|image| image.1).max().unwrap_or(1);
        let width = width.clamp(1, MAX_LAYER_SIZE);
        let height = height.clamp(1, MAX_LAYER_SIZE);

        let mut texels = Vec::with_capacity((width * height * 4) as usize * images.len());
        for (w, h, rgba) in &images {
            match (*w, *h) == (width, height) {
                true => texels.extend_from_slice(rgba),
                false => {
                    let image = image::RgbaImage::from_raw(*w, *h, rgba.clone())
                        .expect("a decoded image is as long as its shape");
                    let resized =
                        image::imageops::resize(&image, width, height, FilterType::Triangle);
                    texels.extend_from_slice(resized.as_raw());
                }
            }
        }

        Layers {
            width,
            height,
            count: images.len() as u32,
            texels,
        }
    }
}

/// `image`'s texels as the array `space` names stores them.
///
/// `to_rgba8` scales a float channel to a byte without touching its curve, which
/// is right for `data` and wrong for `color`: the array decodes every byte as
/// sRGB, so a linear 0.5 stored as 128 would come back as 0.21.
fn rgba8(image: DynamicImage, space: ColorSpace) -> RgbaImage {
    let float = matches!(
        image,
        DynamicImage::ImageRgb32F(_) | DynamicImage::ImageRgba32F(_)
    );
    if !float || space == ColorSpace::Linear {
        return image.to_rgba8();
    }

    let mut pixels = image.to_rgba32f();
    for pixel in pixels.pixels_mut() {
        for channel in &mut pixel.0[..3] {
            *channel = linear_to_srgb(*channel);
        }
    }
    DynamicImage::ImageRgba32F(pixels).to_rgba8()
}

/// The inverse of the sRGB decode the colour array's sampler does. Clamped
/// first: a byte cannot hold more than white, and a NaN becomes black.
fn linear_to_srgb(channel: f32) -> f32 {
    let channel = match channel.is_nan() {
        true => 0.0,
        false => channel.clamp(0.0, 1.0),
    };
    match channel <= 0.0031308 {
        true => channel * 12.92,
        false => 1.055 * channel.powf(1.0 / 2.4) - 0.055,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_image_asked_for_twice_is_one_layer() {
        let mut images = Images::default();

        assert_eq!(
            images.add(Path::new("a.png"), ColorSpace::Srgb),
            (COLOR_TEXTURES, 0)
        );
        assert_eq!(
            images.add(Path::new("b.png"), ColorSpace::Srgb),
            (COLOR_TEXTURES, 1)
        );
        assert_eq!(
            images.add(Path::new("a.png"), ColorSpace::Srgb),
            (COLOR_TEXTURES, 0)
        );
    }

    /// Read two ways, a file decodes to two sets of numbers, so it is two
    /// layers — one in each array.
    #[test]
    fn an_image_read_two_ways_is_one_layer_in_each_array() {
        let mut images = Images::default();

        assert_eq!(
            images.add(Path::new("a.png"), ColorSpace::Srgb),
            (COLOR_TEXTURES, 0)
        );
        assert_eq!(
            images.add(Path::new("a.png"), ColorSpace::Linear),
            (DATA_TEXTURES, 0)
        );
    }

    #[test]
    fn an_array_nothing_samples_is_one_texel() {
        let textures = Images::default().load().expect("nothing to read");

        for layers in [textures.color, textures.data] {
            assert_eq!((layers.width, layers.height, layers.count), (1, 1, 1));
            assert_eq!(layers.texels.len(), 4);
        }
    }

    /// Every layer is the largest image's size, each side separately, and an
    /// image that is already that size is copied through untouched.
    #[test]
    fn every_layer_is_the_largest_size() {
        let wide = (4, 2, vec![10; 4 * 2 * 4]);
        let tall = (2, 3, vec![200; 2 * 3 * 4]);
        let layers = Layers::build(vec![wide, tall]);

        assert_eq!((layers.width, layers.height, layers.count), (4, 3, 2));
        assert_eq!(layers.texels.len(), 4 * 3 * 4 * 2);
        // A flat image stays flat however it is stretched.
        assert!(layers.texels[..48].iter().all(|&t| t == 10));
        assert!(layers.texels[48..].iter().all(|&t| t == 200));
    }

    #[test]
    fn a_layer_never_grows_past_the_cap() {
        let huge = (
            MAX_LAYER_SIZE * 2,
            1,
            vec![0; (MAX_LAYER_SIZE * 2 * 4) as usize],
        );
        let layers = Layers::build(vec![huge]);

        assert_eq!((layers.width, layers.height), (MAX_LAYER_SIZE, 1));
    }

    /// A float colour map is encoded, so the sampler's decode gives back the
    /// linear value the file holds: 0.5 is stored as 188, not 128.
    #[test]
    fn a_float_colour_map_is_encoded_to_srgb() {
        let pixel = image::Rgba([0.5, 0.0, 1.0, 0.5]);
        let float = || DynamicImage::ImageRgba32F(image::Rgba32FImage::from_pixel(1, 1, pixel));

        assert_eq!(
            rgba8(float(), ColorSpace::Srgb).as_raw(),
            &[188, 0, 255, 128]
        );
        assert_eq!(
            rgba8(float(), ColorSpace::Linear).as_raw(),
            &[128, 0, 255, 128]
        );
    }

    /// An 8-bit file is already sRGB-encoded, and passes through untouched.
    #[test]
    fn a_byte_colour_map_is_left_alone() {
        let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(1, 1, image::Rgba([128; 4])));

        assert_eq!(rgba8(image, ColorSpace::Srgb).as_raw(), &[128; 4]);
    }

    #[test]
    fn a_missing_image_is_named() {
        let mut images = Images::default();
        images.add(Path::new("does/not/exist.png"), ColorSpace::Linear);

        let error = images.load().expect_err("there is no file").to_string();
        assert!(error.contains("exist.png"), "{error}");
    }
}
