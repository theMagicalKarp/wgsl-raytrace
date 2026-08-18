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
    pub defocus_angle: f32,

    /// Up.
    pub v: [f32; 3],
    pub focus_distance: f32,

    /// Forward, toward `look_at`.
    pub w: [f32; 3],
    pub max_bounces: u32,

    pub background: [f32; 3],
    /// 1-based index of the sample being traced, which will also reseed the RNG.
    pub sample: u32,

    pub width: u32,
    pub height: u32,
    _pad: [u32; 2],
}

const _: () = assert!(size_of::<GpuCamera>() == 96);

impl From<&CameraOptions> for GpuCamera {
    fn from(camera: &CameraOptions) -> Self {
        let (width, height) = camera.get_dimensions();
        let (u, v, w) = math::camera_basis(camera.look_from, camera.look_at, camera.vup);

        GpuCamera {
            origin: camera.look_from,
            fov: camera.fov.to_radians(),
            u,
            defocus_angle: camera.defocus_angle,
            v,
            focus_distance: camera.focus_dist,
            w,
            max_bounces: camera.max_bounces,
            background: camera.background,
            sample: 1,
            width,
            height,
            _pad: [0; 2],
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
    }
}
