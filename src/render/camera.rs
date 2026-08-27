use crate::config::CameraOptions;
use crate::math;
use bytemuck::Pod;
use bytemuck::Zeroable;

/// The camera and frame settings, as one uniform block.
///
/// The scene's `look_from`/`look_at`/`vup` are resolved into an orthonormal
/// basis here rather than in the shader: it is the same answer for every pixel,
/// and it keeps a `sin`, a `cos` and two cross products out of a function that
/// runs a few hundred thousand times a pass. The field order pairs each
/// `vec3<f32>` with a scalar, so every row is exactly the 16 bytes WGSL aligns
/// them to and no padding has to be invented.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuCamera {
    pub origin: [f32; 3],
    /// Vertical field of view, in radians — the shader wants a `tan`, not degrees.
    pub fov: f32,

    /// Right.
    pub u: [f32; 3],
    /// Radius of the lens disk primary rays start from, already resolved out of
    /// the scene's `defocus_angle` and `focus_dist`. Zero is a pinhole.
    pub defocus_radius: f32,

    /// Up.
    pub v: [f32; 3],
    pub focus_distance: f32,

    /// Forward, toward `look_at`.
    pub w: [f32; 3],
    pub max_bounces: u32,

    pub background: [f32; 3],
    /// 1-based index of the sample being traced, which also reseeds the RNG.
    pub sample: u32,

    pub width: u32,
    pub height: u32,
    /// How many entries the shader's light table has, and the total power they
    /// share out. Neither is a camera setting — they belong to the scene — but
    /// they ride here because the shader needs them as scalars and this is the
    /// block it already reads scalars from.
    /// [`render`](crate::render::render) fills them in once the scene has been
    /// loaded, which is why [`From`] leaves them zero.
    pub light_count: u32,
    pub light_power: f32,

    /// The side of the square grid a pixel's samples are stratified over, so
    /// that `strata * strata` samples cover the pixel one cell each instead of
    /// clumping the way independent draws do. One reduces exactly to an
    /// unstratified jitter, which is what keeps a single-sample render from
    /// being a special case.
    ///
    /// Unlike `light_count` and `light_power` this is a camera setting — it
    /// falls out of `samples` — so [`From`] fills it in rather than
    /// [`render`](crate::render::render).
    pub strata: u32,

    /// Every row above is a `vec3<f32>` paired with a scalar and so is exactly
    /// the 16 bytes WGSL aligns one to. `strata` has no vector to ride beside,
    /// which is what costs this block a seventh row.
    _pad: [u32; 3],
}

const _: () = assert!(size_of::<GpuCamera>() == 112);

impl From<&CameraOptions> for GpuCamera {
    fn from(camera: &CameraOptions) -> Self {
        let (width, height) = camera.get_dimensions();
        let (u, v, w) = math::camera_basis(camera.look_from, camera.look_at, camera.vup);

        GpuCamera {
            origin: camera.look_from,
            fov: camera.fov.to_radians(),
            u,
            defocus_radius: camera.focus_dist * (camera.defocus_angle.to_radians() / 2.0).tan(),
            v,
            focus_distance: camera.focus_dist,
            w,
            max_bounces: camera.max_bounces,
            background: camera.background,
            sample: 1,
            width,
            height,
            light_count: 0,
            light_power: 0.0,
            strata: (camera.samples as f32).sqrt() as u32,
            _pad: [0; 3],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn camera(source: &str) -> GpuCamera {
        let config: Config = toml::from_str(source).unwrap();
        GpuCamera::from(&config.camera)
    }

    #[test]
    fn resolves_the_scene_camera_into_a_basis() {
        let gpu = camera(
            r#"
[camera]
aspect_ratio = "square"
image_width = 64
samples = 8
max_bounces = 4
fov = 90
look_from = [0.0, 0.0, 5.0]
look_at = [0.0, 0.0, 0.0]
"#,
        );

        assert_eq!((gpu.width, gpu.height), (64, 64));
        assert_eq!(gpu.fov, std::f32::consts::FRAC_PI_2);
        assert_eq!(gpu.w, [0.0, 0.0, -1.0], "the camera should look inward");
        assert_eq!(gpu.v, [0.0, 1.0, 0.0]);
        assert_eq!(gpu.sample, 1, "samples are counted from one");
        assert_eq!(gpu.light_count, 0, "the scene is what knows about lights");
        assert_eq!(gpu.light_power, 0.0);
        assert_eq!(gpu.defocus_radius, 0.0, "a scene without one is a pinhole");
        assert_eq!(
            gpu.strata, 2,
            "eight samples stratify over a two by two grid"
        );
    }

    #[test]
    fn strata_is_the_largest_square_that_fits_the_sample_count() {
        let with = |samples: u32| {
            camera(&format!(
                r#"
[camera]
aspect_ratio = "square"
image_width = 64
samples = {samples}
max_bounces = 4
fov = 90
look_from = [0.0, 0.0, 5.0]
look_at = [0.0, 0.0, 0.0]
"#
            ))
            .strata
        };

        // A single sample has no grid to walk, and the shader's `max(strata, 1)`
        // turns this back into the jitter it had before stratification.
        assert_eq!(with(1), 1);
        assert_eq!(with(64), 8, "an exact square covers every cell equally");

        // Not a square, so the shader wraps: with 70 samples the first six cells
        // of the 8x8 grid take two and the rest take one, which is a spread
        // unstratified sampling only manages on average.
        assert_eq!(with(70), 8);
        assert_eq!(with(63), 7, "and it never claims a grid it cannot fill");
    }

    #[test]
    fn a_defocus_angle_becomes_a_lens_radius() {
        // The angle spans the whole cone, so a 90 degree one over a focus
        // distance of two opens a lens of exactly that radius.
        let gpu = camera(
            r#"
[camera]
aspect_ratio = "square"
image_width = 64
samples = 8
max_bounces = 4
fov = 90
look_from = [0.0, 0.0, 5.0]
look_at = [0.0, 0.0, 0.0]
defocus_angle = 90.0
focus_dist = 2.0
"#,
        );

        assert!((gpu.defocus_radius - 2.0).abs() < 1e-5, "{gpu:?}");
    }
}
