use super::Image;

/// The three feature images, resolved and ready to write.
pub struct Aovs {
    pub normal: Image,
    pub albedo: Image,
    /// Distance from the camera, scaled against [`Aovs::far`] because depth has
    /// no natural range to encode into eight bits, and inverted so that near
    /// reads bright. A pixel that hit nothing encodes as far rather than as the
    /// zero the shader filed for it — see the encode below.
    pub depth: Image,
    /// The farthest surface in the frame, in world units, and so the divisor
    /// `depth` was normalised by. Quoted rather than assumed: without it the
    /// depth image is a picture of a scene at an unknown scale.
    pub far: f32,
}

impl Aovs {
    /// Averages the summed features and encodes each as an 8-bit image.
    ///
    /// `normals` is `[normal.xyz, depth]` per pixel and `albedos` is
    /// `[albedo.rgb, object-id vote]`, the colours summed over `samples` passes
    /// exactly as the radiance was. The fourth channel is not a float at all —
    /// it is `shader.wgsl`'s packed `u32` vote — and it is not encoded: it
    /// exists to be compared for equality by a filter, not looked at.
    pub fn new(
        normals: &[[f32; 4]],
        albedos: &[[f32; 4]],
        (width, height): (u32, u32),
        samples: u32,
    ) -> Aovs {
        let count = 1.0 / samples.max(1) as f32;

        // One pass to find the range before the one that encodes into it. A
        // scene-independent constant would be the alternative and there isn't
        // one: these are world units, and a teapot and a landscape do not share
        // a scale.
        let far = normals
            .iter()
            .map(|normal| normal[3] * count)
            .filter(|depth| depth.is_finite())
            .fold(0.0f32, f32::max);
        let range = match far > 0.0 {
            true => 1.0 / far,
            false => 0.0,
        };

        // Signed and centred, the way every consumer of a normal map expects.
        // The average of several unit normals is shorter than one — that is the
        // point, an edge pixel should not claim a confidence it does not have —
        // so this is not renormalised on the way out.
        let normal = encode(width, height, normals, |normal| {
            [0, 1, 2].map(|channel| normal[channel] * count * 0.5 + 0.5)
        });
        let albedo = encode(width, height, albedos, |albedo| {
            [0, 1, 2].map(|channel| albedo[channel] * count)
        });
        // Near is white and far is black, which is the way round that keeps the
        // subject readable and the background out of the way — and which is also
        // why a miss cannot simply be encoded. The shader files zero depth for a
        // path that hit nothing, and zero inverts to white, so the background
        // would come out as the nearest thing in the frame to anything reading
        // this file. It is pushed to `far` instead: nothing is behind the sky.
        //
        // A summed depth of exactly zero is the test, and it is exact rather
        // than approximate: every real hit is at a positive `t`, so only a pixel
        // whose every sample missed can sum to nothing.
        let depth = encode(width, height, normals, |normal| {
            let depth = normal[3] * count;
            match depth > 0.0 {
                true => [1.0 - (depth * range).clamp(0.0, 1.0); 3],
                false => [0.0; 3],
            }
        });

        Aovs {
            normal,
            albedo,
            depth,
            far,
        }
    }
}

/// Turns one summed buffer into an opaque 8-bit image through `channel`.
///
/// No gamma anywhere: none of these is a colour being shown to an eye. A normal
/// and a depth are geometry, and the albedo is the linear reflectance a denoiser
/// will divide the radiance by — encoding any of them through a display curve
/// would mean every consumer had to undo it first.
fn encode(
    width: u32,
    height: u32,
    sums: &[[f32; 4]],
    channel: impl Fn(&[f32; 4]) -> [f32; 3],
) -> Image {
    let pixels = sums
        .iter()
        .flat_map(|sum| {
            let [red, green, blue] = channel(sum).map(|value| {
                let scaled = value.clamp(0.0, 1.0) * 255.0 + 0.5;
                // NaN fails the comparison inside `clamp` rather than being
                // corrected by it, so it reaches here intact and `as u8` would
                // make a zero of it silently. Nothing should produce one; this
                // is what keeps that a fact rather than an assumption.
                match scaled.is_finite() {
                    true => scaled as u8,
                    false => 0,
                }
            });
            [red, green, blue, 255]
        })
        .collect();

    Image {
        width,
        height,
        pixels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Four samples of one surface: the average is that surface, and a normal of
    /// `+x` lands mid-grey in the two channels it is zero in.
    #[test]
    fn features_average_over_the_samples_that_saw_them() {
        let aovs = Aovs::new(&[[4.0, 0.0, 0.0, 8.0]], &[[2.0, 1.0, 0.0, 12.0]], (1, 1), 4);

        assert_eq!(aovs.normal.pixels, [255, 128, 128, 255], "+x, encoded");
        assert_eq!(aovs.albedo.pixels, [128, 64, 0, 255]);
        assert_eq!(aovs.far, 2.0, "eight over four samples");
    }

    /// A pixel that hit nothing. The shader leaves a zero normal, a zero depth
    /// and a white albedo, and none of it may come out as a NaN.
    #[test]
    fn a_background_pixel_encodes_without_a_range_to_scale_by() {
        let aovs = Aovs::new(&[[0.0; 4]], &[[1.0, 1.0, 1.0, -1.0]], (1, 1), 1);

        assert_eq!(aovs.far, 0.0);
        assert_eq!(aovs.normal.pixels, [128, 128, 128, 255], "a zero normal");
        assert_eq!(aovs.albedo.pixels, [255, 255, 255, 255]);
        assert_eq!(
            aovs.depth.pixels,
            [0, 0, 0, 255],
            "and a miss is far, not near"
        );
    }

    /// The background against a frame that does have a range: the sky must
    /// encode as the farthest thing in it and not, through the inversion, as the
    /// nearest. A near surface, a far one, and a miss.
    #[test]
    fn a_miss_encodes_as_far_rather_than_inverting_to_near() {
        let surface = [1.0, 1.0, 1.0, 0.0];
        let aovs = Aovs::new(
            &[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 2.0], [0.0; 4]],
            &[surface, surface, [1.0, 1.0, 1.0, -1.0]],
            (3, 1),
            1,
        );

        assert_eq!(aovs.far, 2.0);
        assert_eq!(aovs.depth.pixels[0], 128, "halfway out and half bright");
        assert_eq!(aovs.depth.pixels[4], 0, "the far surface is black");
        assert_eq!(aovs.depth.pixels[8], 0, "and the sky is no nearer than it");
    }

    /// Depth is normalised against the frame's own farthest surface, and
    /// inverted so that near reads bright.
    #[test]
    fn depth_is_scaled_against_the_farthest_surface_in_the_frame() {
        let aovs = Aovs::new(
            &[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 4.0]],
            &[[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]],
            (2, 1),
            1,
        );

        assert_eq!(aovs.far, 4.0);
        assert_eq!(aovs.depth.pixels[0], 191, "a quarter of the way out");
        assert_eq!(aovs.depth.pixels[4], 0, "and the far surface is black");
    }

    /// An averaged normal is shorter than a unit one, and stays that way: an
    /// edge pixel whose samples disagreed should read as less certain, not be
    /// renormalised into a direction none of them saw.
    #[test]
    fn an_edge_normal_keeps_the_length_averaging_gave_it() {
        // Two samples, `+x` and `-x`, which average to nothing at all.
        let aovs = Aovs::new(&[[0.0, 0.0, 0.0, 2.0]], &[[2.0, 2.0, 2.0, 0.0]], (1, 1), 2);

        assert_eq!(aovs.normal.pixels, [128, 128, 128, 255]);
    }
}
