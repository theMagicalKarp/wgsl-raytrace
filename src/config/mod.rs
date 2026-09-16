//! The TOML scene description and the CLI that points at it.
//!
//! Everything here is plain data. Deserializing a [`Config`] never touches the
//! filesystem beyond the config file itself, so a scene can be parsed and
//! checked without loading a single triangle — [`Config::validate`] is the step
//! that resolves object paths against the config's directory.

use clap::Parser;
use colored::Colorize;
use serde::Deserialize;
use serde_inline_default::serde_inline_default;
use std::error::Error;
use std::fmt;
use std::ops::Range;
use std::path::Path;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// Path of toml configuration file
    #[arg(short, long, value_parser=file_exists)]
    pub config: PathBuf,

    /// Path of file to save the render to
    #[arg(short, long, default_value = "render.png", value_parser=parent_exists)]
    pub output: PathBuf,

    /// Directly override the sample count listed in the configuration file
    #[arg(short, long)]
    pub samples: Option<u32>,

    /// Draw the frame in the terminal as it converges, and stop early on Ctrl-C
    #[arg(short, long)]
    pub preview: bool,

    /// Also write a heatmap of the render's per-pixel variance here
    #[arg(long, value_parser=parent_exists)]
    pub variance: Option<PathBuf>,

    /// Also write the frame the tracer drew before any filtering into this
    /// directory, beside the render's normal, albedo and depth buffers, for an
    /// external denoiser to read and for the filter to be judged against
    #[arg(long, value_parser=directory_exists)]
    pub debug: Option<PathBuf>,

    /// Run the a-trous filter over the finished frame. The filtered frame is
    /// the render, so it is what --output gets; --debug is where the unfiltered
    /// one it was built from stays reachable
    #[arg(long)]
    pub denoise: bool,

    /// Directly override the firefly rejection strength listed in the
    /// configuration file. Zero is off
    #[arg(long)]
    pub outlier_k: Option<f32>,

    /// Draw a different sample sequence for the same scene. Zero, the default,
    /// is the sequence every render draws; any other value shares no noise with
    /// it, which is what a reference for measuring bias has to do
    #[arg(long, default_value_t = 0)]
    pub seed: u32,
}

/// A file a finished render will be written to. The file itself will not exist
/// yet; the directory holding it has to.
///
/// Checked at parse time for the same reason `--debug`'s directory is: a typo in
/// an output path is only discovered when the write fails, which is after the
/// render, and on a scene like `examples/stairs` that is the difference between
/// a typo costing a second and costing the whole thing.
fn parent_exists(path: &str) -> Result<PathBuf, String> {
    let path_buf = PathBuf::from(path);
    // A bare filename has an empty parent, which is the working directory and
    // always exists.
    match path_buf.parent() {
        Some(parent) if !parent.as_os_str().is_empty() && !parent.is_dir() => {
            Err(format!("Directory does not exist: {}", parent.display()))
        }
        _ => Ok(path_buf),
    }
}

fn directory_exists(path: &str) -> Result<PathBuf, String> {
    let path_buf = PathBuf::from(path);
    if path_buf.is_dir() {
        Ok(path_buf)
    } else {
        Err(format!("Directory does not exist: {}", path))
    }
}

fn file_exists(path: &str) -> Result<PathBuf, String> {
    let path_buf = PathBuf::from(path);
    if path_buf.is_file() {
        Ok(path_buf)
    } else {
        Err(format!("File does not exist: {}", path))
    }
}

#[derive(Deserialize, Debug, PartialEq)]
pub enum AspectRatios {
    #[serde(alias = "widescreen")]
    Widescreen,

    #[serde(alias = "square")]
    Square,

    #[serde(alias = "smartphone")]
    Smartphone,

    #[serde(alias = "standard")]
    Standard,

    #[serde(alias = "cinema")]
    Cinema,
}

impl AspectRatios {
    pub fn get_ratio(&self) -> (f32, f32) {
        match self {
            AspectRatios::Widescreen => (16.0, 9.0),
            AspectRatios::Square => (1.0, 1.0),
            AspectRatios::Smartphone => (9.0, 16.0),
            AspectRatios::Standard => (4.0, 3.0),
            AspectRatios::Cinema => (1.85, 1.0),
        }
    }

    pub fn get_height(&self, width: u32) -> u32 {
        let (ratio_x, ratio_y) = self.get_ratio();
        let ratio = ratio_x / ratio_y;
        std::cmp::max(1, (width as f32 / ratio) as u32)
    }
}

impl fmt::Display for AspectRatios {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (ratio_x, ratio_y) = self.get_ratio();
        write!(f, "{}:{}", ratio_x, ratio_y)
    }
}

/// Everything the camera needs. Values are `f32` throughout because they end up
/// in a uniform buffer, and WGSL has no `f64`.
#[serde_inline_default]
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct CameraOptions {
    pub aspect_ratio: AspectRatios,
    pub image_width: u32,
    pub samples: u32,
    pub max_bounces: u32,
    pub fov: f32,

    pub look_from: [f32; 3],
    pub look_at: [f32; 3],
    #[serde_inline_default([0.0, 1.0, 0.0])]
    pub vup: [f32; 3],

    #[serde_inline_default(0.0)]
    pub defocus_angle: f32,
    #[serde_inline_default(1.0)]
    pub focus_dist: f32,

    /// Linear gain on the scene's radiance, applied on the way into the tone
    /// curve and nowhere else.
    ///
    /// A camera setting rather than a scene one, which is why it lives here
    /// beside the lens: it changes how much of the light reaches the film and
    /// not how much of it there is. Nothing upstream of the tone curve sees it —
    /// the accumulator, the denoiser and the noise figure all work in the
    /// radiance the scene actually carries, so turning this up cannot make a
    /// render converge faster or a filter look better than it is.
    ///
    /// Multiplicative, matching `environment.intensity` rather than counting
    /// stops: `2.0` is one stop up, `0.5` one stop down.
    #[serde_inline_default(1.0)]
    pub exposure: f32,
}

/// The sky, and with it the scene's ambient light.
///
/// A path that escapes the geometry brings home whatever is here, so a scene
/// holding no `light` material and left at the default black renders black.
/// `color` and `file` are the same setting at two fidelities — a constant sky
/// and a photographed one — and `file` wins where both are given.
#[serde_inline_default]
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentOptions {
    /// The sky when there is no `file`. Uploaded as a one-texel map, which is
    /// what keeps a flat background from being a second path through the shader.
    #[serde(default)]
    pub color: [f32; 3],

    /// An equirectangular image to read the sky out of, relative to the config
    /// file. [`Config::validate`] rewrites this the way it does an object's mesh.
    #[serde(default)]
    pub file: Option<PathBuf>,

    /// Scales whichever of the two is in play. The place to pull an HDRI's
    /// exposure down without re-authoring it.
    #[serde_inline_default(1.0)]
    pub intensity: f32,

    /// Degrees of yaw about the +Y axis, for aiming a map's sun at the scene.
    #[serde_inline_default(0.0)]
    pub rotation: f32,
}

/// Black, unlit, unrotated — what a config that omits the table entirely gets,
/// and what `background` defaulted to before it moved here.
impl Default for EnvironmentOptions {
    fn default() -> Self {
        EnvironmentOptions {
            color: [0.0, 0.0, 0.0],
            file: None,
            intensity: 1.0,
            rotation: 0.0,
        }
    }
}

/// The most a-trous passes a scene may ask for.
///
/// Past this the stride is 65536 and every further pass is a no-op that still
/// costs a dispatch, so an unbounded count buys nothing and a mistyped one would
/// build a `Vec` of dispatches the size of whatever number was typed.
pub const MAX_ITERATIONS: u32 = 16;

/// How hard the a-trous filter is allowed to blur, and how far.
///
/// Every field has an inline default, because a scene written before the filter
/// existed must still parse — the whole table is optional and the numbers below
/// are what a scene that never mentions it gets.
///
/// The three sigmas are all thresholds on a difference between two pixels, and
/// they run in opposite directions: `normal` is an exponent, so **larger is
/// stricter**, while `depth` and `luminance` divide, so larger is *looser*.
#[serde_inline_default]
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct DenoiseOptions {
    /// A-trous passes, each doubling the tap stride of the one before it. Five
    /// passes of a 5x5 kernel reach 125 pixels for the cost of 125 taps, which
    /// is the whole trick — and the reason this is not worth turning up much
    /// further, since a sixth pass reaches a quarter of a 1080p frame.
    ///
    /// Bounded by [`MAX_ITERATIONS`], which is where the strides stop meaning
    /// anything rather than where they stop being useful.
    #[serde_inline_default(5)]
    pub iterations: u32,

    /// Exponent on `dot(n_p, n_q)`. High by design: two surfaces eight degrees
    /// apart still share 0.99 of a dot product, and only an exponent this steep
    /// turns that into a rejection.
    #[serde_inline_default(128.0)]
    pub sigma_normal: f32,

    /// Scales the depth difference a tap is allowed, in multiples of how fast
    /// depth is already changing across the surface under the pixel. Relative
    /// rather than absolute, so one number works on a teapot and a landscape.
    #[serde_inline_default(1.0)]
    pub sigma_depth: f32,

    /// Scales the luminance difference a tap is allowed, in multiples of the
    /// centre pixel's own standard error. This is the term that does the actual
    /// denoising: a difference the pixel's noise can explain gets blurred away,
    /// and one it cannot is an edge and survives.
    #[serde_inline_default(4.0)]
    pub sigma_luminance: f32,
}

impl Default for DenoiseOptions {
    fn default() -> Self {
        DenoiseOptions {
            iterations: 5,
            sigma_normal: 128.0,
            sigma_depth: 1.0,
            sigma_luminance: 4.0,
        }
    }
}

impl fmt::Display for DenoiseOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} iterations · normal {} · depth {} · luminance {}",
            self.iterations, self.sigma_normal, self.sigma_depth, self.sigma_luminance,
        )
    }
}

/// Firefly rejection: what a sample has to look like before the tracer decides
/// it is a lucky path rather than light that is really there.
///
/// Its own table rather than a field of `[denoise]`, because it is not a filter.
/// Everything under `[denoise]` runs once after the last sample has landed and
/// cannot touch the render it was given; this acts *during* tracing, on the way
/// into the accumulator, and it is the only setting in the tree that can bias
/// the image. `k` of zero is off, and off is the default.
#[serde_inline_default]
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct OutlierOptions {
    /// Standard deviations above a pixel's running mean a sample may reach
    /// before it is scaled back to that edge.
    ///
    /// The window widens as the fourth root of the sample count, so a sample
    /// capped early is accepted later and the estimator still converges on the
    /// unbiased answer — it simply gets there without the variance one lucky
    /// path would have injected. Zero disables the whole mechanism, which is the
    /// only setting under which the render is unbiased at *every* sample count
    /// rather than in the limit.
    #[serde_inline_default(0.0)]
    pub k: f32,

    /// Samples a pixel must have gathered before `k` is allowed to act on it.
    ///
    /// Over the first handful of samples the variance estimate is noise and the
    /// running mean is far below where it will settle, so a threshold built from
    /// them rejects the light source itself — melee's emitter is at 15.0, and no
    /// mean over three samples is anywhere near that.
    #[serde_inline_default(32)]
    pub warmup: u32,
}

impl Default for OutlierOptions {
    fn default() -> Self {
        OutlierOptions { k: 0.0, warmup: 32 }
    }
}

impl fmt::Display for OutlierOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.k > 0.0 {
            true => write!(f, "k {} after {} samples", self.k, self.warmup),
            false => write!(f, "off"),
        }
    }
}

impl CameraOptions {
    pub fn get_dimensions(&self) -> (u32, u32) {
        (
            self.image_width,
            self.aspect_ratio.get_height(self.image_width),
        )
    }
}

/// A physically based surface in the spirit of Blender's Principled BSDF: a
/// diffuse base under a dielectric specular coat, blended toward glass by
/// `transmission` and toward a conductor by `metallic`.
///
/// Every field defaults to what Blender's node does, so `material =
/// "principled"` alone is a grey, half-rough plastic.
#[serde_inline_default]
#[derive(Deserialize, Debug, Clone, Copy, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Principled {
    /// Diffuse albedo, and the reflectance at normal incidence once metallic.
    #[serde_inline_default([0.8, 0.8, 0.8])]
    pub base_color: [f32; 3],

    /// Microfacet roughness in `[0, 1]`, squared into GGX's alpha.
    #[serde_inline_default(0.5)]
    pub roughness: f32,

    /// Dielectric at zero, conductor at one, and a blend of the two between.
    #[serde_inline_default(0.0)]
    pub metallic: f32,

    /// Index of refraction, which sets how strongly the dielectric part
    /// reflects. One reflects nothing at any angle.
    #[serde_inline_default(1.5)]
    pub ior: f32,

    /// Opaque at zero, glass at one: the share of the dielectric part that
    /// refracts through the surface rather than scattering off its base.
    #[serde_inline_default(0.0)]
    pub transmission: f32,

    /// Coverage: the chance a ray is stopped by the surface at all. The rest
    /// pass straight through as though it were not there, which is a cutout
    /// rather than a refraction.
    #[serde_inline_default(1.0)]
    pub alpha: f32,
}

impl Default for Principled {
    fn default() -> Self {
        Principled {
            base_color: [0.8, 0.8, 0.8],
            roughness: 0.5,
            metallic: 0.0,
            ior: 1.5,
            transmission: 0.0,
            alpha: 1.0,
        }
    }
}

/// The surface models the tracer knows how to scatter off of.
///
/// Everything but a light is a [`Principled`] surface on the GPU. The other names are shorthand for one, kept because they read better in
/// a scene than the handful of numbers they stand for.
#[derive(Deserialize, Debug, PartialEq)]
#[serde(tag = "material", deny_unknown_fields)]
pub enum Material {
    #[serde(rename = "principled")]
    Principled(Principled),

    #[serde(rename = "lambertian")]
    Lambertian { albedo: [f32; 3] },

    #[serde(rename = "metal")]
    Metal { albedo: [f32; 3], roughness: f32 },

    #[serde(rename = "dielectric")]
    Dielectric { refraction_index: f32 },

    #[serde(rename = "glass")]
    Glass {},

    #[serde(rename = "water")]
    Water {},

    #[serde(rename = "light")]
    Light { emit: [f32; 3] },
}

impl fmt::Display for Material {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Material::Principled(p) => write!(
                f,
                "principled{:?} roughness {} metallic {} ior {} transmission {} alpha {}",
                p.base_color, p.roughness, p.metallic, p.ior, p.transmission, p.alpha,
            ),
            Material::Lambertian { albedo } => write!(f, "lambertian{:?}", albedo),
            Material::Metal { albedo, roughness } => {
                write!(f, "metal{:?} roughness {}", albedo, roughness)
            }
            Material::Dielectric { refraction_index } => {
                write!(f, "dielectric ior {}", refraction_index)
            }
            Material::Glass {} => write!(f, "glass"),
            Material::Water {} => write!(f, "water"),
            Material::Light { emit } => write!(f, "light{:?}", emit),
        }
    }
}

impl Material {
    /// Rejects the values a scene can spell but the BSDF cannot price: a
    /// roughness or metallic outside `[0, 1]` has no microfacet meaning, an
    /// index of refraction under one turns the Fresnel term inside out, and a
    /// negative color is light taken out of nowhere. NaN fails every range.
    fn validate(&self) -> Result<(), String> {
        let unit = |name: &str, value: f32| match (0.0..=1.0).contains(&value) {
            true => Ok(()),
            false => Err(format!("{name} must be between 0 and 1, not {value}")),
        };
        let color = |name: &str, value: [f32; 3]| match value.iter().all(|c| *c >= 0.0) {
            true => Ok(()),
            false => Err(format!("{name} must be non-negative, not {value:?}")),
        };
        let ior = |name: &str, value: f32| match value >= 1.0 {
            true => Ok(()),
            false => Err(format!("{name} must be at least 1, not {value}")),
        };

        match self {
            Material::Principled(p) => {
                color("base_color", p.base_color)?;
                unit("roughness", p.roughness)?;
                unit("metallic", p.metallic)?;
                unit("transmission", p.transmission)?;
                unit("alpha", p.alpha)?;
                ior("ior", p.ior)
            }
            Material::Lambertian { albedo } => color("albedo", *albedo),
            Material::Metal { albedo, roughness } => {
                color("albedo", *albedo)?;
                unit("roughness", *roughness)
            }
            Material::Light { emit } => color("emit", *emit),
            Material::Dielectric { refraction_index } => ior("refraction_index", *refraction_index),
            Material::Glass {} | Material::Water {} => Ok(()),
        }
    }
}

#[derive(Deserialize, Debug, Clone, Copy, PartialEq)]
pub enum Axis {
    #[serde(alias = "x")]
    X,
    #[serde(alias = "y")]
    Y,
    #[serde(alias = "z")]
    Z,
}

/// A single step of an object's model transform. The list is applied in the
/// order it appears in the config, so the last entry is the outermost one.
#[derive(Deserialize, Debug, PartialEq)]
#[serde(tag = "type", deny_unknown_fields)]
pub enum Transform {
    #[serde(rename = "translate")]
    Translate { offset: [f32; 3] },

    #[serde(rename = "rotate")]
    Rotate { axis: Axis, degrees: f32 },

    #[serde(rename = "scale")]
    Scale { scalar: [f32; 3] },
}

/// A Wavefront `.obj` mesh, the only primitive this tracer supports.
///
/// No `deny_unknown_fields` here: serde cannot combine it with the flattened
/// material, which is what makes `material = "metal"` and its parameters sit
/// beside `file` in the same table.
#[derive(Deserialize, Debug)]
pub struct Wavefront {
    /// Path to the `.obj` file, relative to the config file.
    pub file: PathBuf,

    /// Optional named group within the file. When absent the whole file is used.
    pub group: Option<String>,

    #[serde(flatten)]
    pub material: Material,

    #[serde(default)]
    pub transform: Vec<Transform>,
}

impl fmt::Display for Wavefront {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.file.display())?;
        if let Some(group) = &self.group {
            write!(f, " [{}]", group)?;
        }
        write!(f, " · {}", self.material)?;
        if !self.transform.is_empty() {
            write!(f, " · {} transforms", self.transform.len())?;
        }
        Ok(())
    }
}

#[derive(Deserialize, Debug)]
#[serde(tag = "shape")]
pub enum Object {
    #[serde(rename = "wavefront")]
    Wavefront(Wavefront),
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub camera: CameraOptions,

    #[serde(default)]
    pub environment: EnvironmentOptions,

    #[serde(default)]
    pub denoise: DenoiseOptions,

    #[serde(default)]
    pub outliers: OutlierOptions,

    /// Which sample sequence to draw. Never read from the TOML — it comes off
    /// `--seed` and exists so a second render of the same scene can be drawn
    /// with no noise in common with the first, which is what measuring bias
    /// needs. Zero is what every render draws.
    #[serde(skip)]
    pub seed: u32,

    #[serde(default)]
    pub objects: Vec<Object>,
}

impl Config {
    /// Rewrites every object path so it is relative to the process, not the
    /// config file, and fails on the first one that does not exist.
    ///
    /// Doing this as its own pass — rather than during deserialization — keeps
    /// the parse pure and puts a clear error in front of the user before any
    /// GPU work starts.
    pub fn validate(&mut self, config_dir: &Path) -> Result<(), Box<dyn Error>> {
        for (index, object) in self.objects.iter_mut().enumerate() {
            let Object::Wavefront(wavefront) = object;

            wavefront.file = config_dir.join(&wavefront.file);
            if !wavefront.file.is_file() {
                return Err(
                    format!("Object file does not exist: {}", wavefront.file.display()).into(),
                );
            }

            wavefront
                .material
                .validate()
                .map_err(|error| format!("Object {index} material: {error}"))?;
        }

        if let Some(file) = &mut self.environment.file {
            *file = config_dir.join(&file);
            if !file.is_file() {
                return Err(format!("Environment file does not exist: {}", file.display()).into());
            }
        }

        // A negative `k` reads as "off" to `reject_outlier`, which tests it
        // against zero, so a typo asking for rejection silently gets none of it.
        // A NaN passes that test and reaches the threshold, where it makes a NaN
        // of the sample and of every pixel it lands on. Both are typos rather
        // than intentions, and neither says anything about itself.
        if !self.outliers.k.is_finite() || self.outliers.k < 0.0 {
            return Err(format!(
                "Outlier k must be a non-negative number, not {}",
                self.outliers.k,
            )
            .into());
        }

        // A negative gain floors to black in the tone curve and a NaN resolves
        // the same way, so either would render a black frame with nothing said
        // about why.
        if !self.camera.exposure.is_finite() || self.camera.exposure < 0.0 {
            return Err(format!(
                "Camera exposure must be a non-negative number, not {}",
                self.camera.exposure,
            )
            .into());
        }

        // The filter's three sigmas, which reach the same end by three different
        // routes. `luminance` and `depth` are divided by, so a negative turns
        // `exp(-x / -k)` into an overflow to infinity; `normal` is an exponent,
        // and `pow(0.0, -k)` is infinity directly. Every one of them ends as
        // `inf / inf`, the tone curve floors the NaN to zero, and `--denoise`
        // writes an all-black PNG with nothing raised — which is exactly what
        // the two guards above exist to prevent.
        //
        // `normal` has to be strictly positive besides. WGSL leaves `pow(0, 0)`
        // undefined, and a dot product of zero is routine — beside every
        // background pixel and across every right-angled edge — so a zero there
        // is a NaN on any GPU that computes it as `exp2(0 * log2(0))`, spread by
        // each wider pass into black blotches around it.
        for (name, sigma, zero) in [
            ("normal", self.denoise.sigma_normal, false),
            ("depth", self.denoise.sigma_depth, true),
            ("luminance", self.denoise.sigma_luminance, true),
        ] {
            if !sigma.is_finite() || sigma < 0.0 || (sigma == 0.0 && !zero) {
                let least = match zero {
                    true => "a non-negative",
                    false => "a positive",
                };
                return Err(
                    format!("Denoise sigma_{name} must be {least} number, not {sigma}").into(),
                );
            }
        }

        if self.denoise.iterations > MAX_ITERATIONS {
            return Err(format!(
                "Denoise iterations must be at most {MAX_ITERATIONS}, not {}",
                self.denoise.iterations,
            )
            .into());
        }

        Ok(())
    }
}

impl fmt::Display for EnvironmentOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.file {
            Some(file) => write!(f, "{}", file.display())?,
            None => write!(
                f,
                "[{}, {}, {}]",
                self.color[0], self.color[1], self.color[2]
            )?,
        }
        if self.intensity != 1.0 {
            write!(f, " x{}", self.intensity)?;
        }
        if self.rotation != 0.0 {
            write!(f, " @ {}deg", self.rotation)?;
        }
        Ok(())
    }
}

/// One line of the settings box: a right-aligned cyan label and its value, cut
/// to length so a long path can never push the border off the edge.
fn row(label: &str, value: &str) -> String {
    const VALUE_WIDTH: usize = 64;

    let value = match value.chars().count() > VALUE_WIDTH {
        true => value.chars().take(VALUE_WIDTH - 1).chain(['…']).collect(),
        false => value.to_string(),
    };

    format!("│{:>14}: {:VALUE_WIDTH$}│", label.cyan().bold(), value)
}

impl fmt::Display for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (width, height) = self.camera.get_dimensions();
        let vector = |v: [f32; 3]| {
            format!(
                "[{:3}, {:3}, {:3}]",
                v[0].to_string(),
                v[1].to_string(),
                v[2].to_string(),
            )
        };

        let msg = [
            ("Dimensions", format!("{}x{}", width, height)),
            ("Aspect Ratio", format!("{}", self.camera.aspect_ratio)),
            ("Samples", format!("{}", self.camera.samples)),
            ("Max Bounces", format!("{}", self.camera.max_bounces)),
            ("Field of View", format!("{}", self.camera.fov)),
            ("Look From", vector(self.camera.look_from)),
            ("Look At", vector(self.camera.look_at)),
            ("Vup", vector(self.camera.vup)),
            ("Defocus Angle", format!("{}", self.camera.defocus_angle)),
            ("Focus Distance", format!("{}", self.camera.focus_dist)),
            ("Exposure", format!("{}", self.camera.exposure)),
            ("Fireflies", self.outliers.to_string()),
            ("Environment", self.environment.to_string()),
            ("Objects", format!("{}", self.objects.len())),
        ]
        .map(|(k, v)| row(k, &v))
        .join("\n");

        writeln!(
            f,
            "┌───{}{}┐",
            " Render Settings ".blue().bold(),
            "─".repeat(60)
        )?;
        writeln!(f, "{}", msg)?;
        for (index, Object::Wavefront(wavefront)) in self.objects.iter().enumerate() {
            writeln!(f, "{}", row(&index.to_string(), &wavefront.to_string()))?;
        }
        write!(f, "└{}┘", "─".repeat(80))
    }
}

/// Renders a TOML parse failure with the offending lines quoted underneath, the
/// way rustc does it. The caller supplies the `error:` prefix, so this starts
/// with the message itself.
pub fn format_error(config_content: &str, path: &Path, error: &toml::de::Error) -> String {
    let Some(span) = error.span() else {
        return error.to_string();
    };

    format!(
        "{}\n  {} {}:{}:{}\n{}",
        error.message().bold(),
        "-->".blue(),
        path.display(),
        span.start,
        span.end,
        span_dump(config_content, span.clone()),
    )
}

pub fn span_dump(config_content: &str, span: Range<usize>) -> String {
    let start = config_content[..span.start].lines().count();
    let end = config_content[..span.end].lines().count();

    config_content
        .lines()
        .enumerate()
        .skip_while(|(i, _)| *i < start.saturating_sub(1))
        .take_while(|(i, _)| *i < end)
        .map(|(i, line)| format!("{:4} | {}", (i + 1).to_string().blue(), line))
        .collect::<Vec<String>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"
[camera]
aspect_ratio = "widescreen"
image_width = 800
samples = 500
max_bounces = 64
fov = 45
look_from = [1.5, 0.0, 1.9]
look_at = [0.0, -0.5, 0.0]

[[objects]]
shape = "wavefront"
file = "teapot.obj"
material = "lambertian"
albedo = [0.42, 0.2, 0.7]

[[objects.transform]]
type = "rotate"
axis = "y"
degrees = 31.5
"#;

    #[test]
    fn parses_a_minimal_scene() {
        let config: Config = toml::from_str(MINIMAL).unwrap();

        assert_eq!(config.camera.aspect_ratio, AspectRatios::Widescreen);
        assert_eq!(config.camera.get_dimensions(), (800, 450));
        assert_eq!(config.objects.len(), 1);

        let Object::Wavefront(wavefront) = &config.objects[0];
        assert_eq!(wavefront.file, PathBuf::from("teapot.obj"));
        assert_eq!(wavefront.group, None);
        assert_eq!(
            wavefront.material,
            Material::Lambertian {
                albedo: [0.42, 0.2, 0.7]
            }
        );
        assert_eq!(
            wavefront.transform,
            vec![Transform::Rotate {
                axis: Axis::Y,
                degrees: 31.5
            }]
        );
    }

    #[test]
    fn applies_camera_defaults() {
        let config: Config = toml::from_str(MINIMAL).unwrap();

        assert_eq!(config.camera.vup, [0.0, 1.0, 0.0]);
        assert_eq!(config.camera.defocus_angle, 0.0);
        assert_eq!(config.camera.focus_dist, 1.0);
        assert_eq!(
            config.camera.exposure, 1.0,
            "unity gain, so scenes written \
             before the tone curve resolve exactly as they always did"
        );
        assert_eq!(config.environment.color, [0.0, 0.0, 0.0]);
        assert_eq!(config.outliers.k, 0.0, "firefly rejection is opt-in");
        assert_eq!(config.outliers.warmup, 32);
        assert_eq!(config.seed, 0, "and the seed never comes from the scene");
    }

    /// The one setting in the tree that can bias the image, so it is worth
    /// asserting that a scene has to ask for it by name.
    #[test]
    fn a_scene_can_ask_for_firefly_rejection() {
        let source = format!("{MINIMAL}\n[outliers]\nk = 3.0\nwarmup = 64\n");
        let config: Config = toml::from_str(&source).unwrap();

        assert_eq!(config.outliers.k, 3.0);
        assert_eq!(config.outliers.warmup, 64);
        assert_eq!(config.outliers.to_string(), "k 3 after 64 samples");
    }

    #[test]
    fn a_negative_k_is_named_rather_than_rendered() {
        let source = format!("{MINIMAL}\n[outliers]\nk = -1.0\n");
        let mut config: Config = toml::from_str(&source).unwrap();

        let error = config
            .validate(Path::new("tests/golden"))
            .expect_err("a negative k should not be accepted");

        assert!(error.to_string().contains("Outlier k"), "{error}");
    }

    /// A scene may set a gain, and it is the one camera field the tone curve
    /// reads.
    #[test]
    fn a_scene_can_open_the_aperture() {
        let source = MINIMAL.replace("fov = 45", "fov = 45\nexposure = 2.5");
        let config: Config = toml::from_str(&source).unwrap();

        assert_eq!(config.camera.exposure, 2.5);
    }

    /// Both of these would render a black frame and say nothing about why, so
    /// they are caught before any GPU work starts rather than after it.
    #[test]
    fn a_nonsense_exposure_is_named_rather_than_rendered() {
        for bad in ["-1.0", "nan"] {
            let source = MINIMAL.replace("fov = 45", &format!("fov = 45\nexposure = {bad}"));
            let mut config: Config = toml::from_str(&source).unwrap();

            let error = config
                .validate(Path::new("tests/golden"))
                .expect_err(&format!("{bad} should not be accepted"));

            assert!(error.to_string().contains("exposure"), "{bad}: {error}",);
        }
    }

    /// Every one of the three sigmas ends as a NaN weight, which the tone curve
    /// floors to black — the same silent failure the two guards above it exist
    /// to prevent, so they are caught the same way.
    #[test]
    fn a_nonsense_sigma_is_named_rather_than_filtered_with() {
        for field in ["sigma_normal", "sigma_depth", "sigma_luminance"] {
            for bad in ["-1.0", "nan"] {
                let source = format!("{MINIMAL}\n[denoise]\n{field} = {bad}\n");
                let mut config: Config = toml::from_str(&source).unwrap();

                let error = config
                    .validate(Path::new("tests/golden"))
                    .expect_err(&format!("{field} = {bad} should not be accepted"));

                assert!(error.to_string().contains(field), "{field}: {error}");
            }
        }
    }

    /// Zero is the one value the three sigmas disagree about. It is a legal,
    /// if blunt, setting for the two that are divided by, and a `pow(0, 0)` for
    /// the exponent — undefined in WGSL, and a NaN on the GPUs that take it
    /// literally.
    #[test]
    fn a_zero_sigma_is_only_refused_where_it_is_an_exponent() {
        let validate = |field: &str| {
            let source = format!("{MINIMAL}\n[denoise]\n{field} = 0.0\n");
            let mut config: Config = toml::from_str(&source).unwrap();
            config.validate(Path::new("tests/golden"))
        };

        let error = validate("sigma_normal").expect_err("a zero exponent should not be accepted");
        assert!(error.to_string().contains("sigma_normal"), "{error}");

        for field in ["sigma_depth", "sigma_luminance"] {
            validate(field).unwrap_or_else(|error| panic!("{field} = 0 is legal: {error}"));
        }
    }

    /// Past the ceiling every pass is a no-op that still costs a dispatch, and
    /// an unbounded count builds a `Vec` the size of whatever was typed.
    #[test]
    fn an_unbounded_iteration_count_is_named_rather_than_dispatched() {
        let source = format!("{MINIMAL}\n[denoise]\niterations = 4000000000\n");
        let mut config: Config = toml::from_str(&source).unwrap();

        let error = config
            .validate(Path::new("tests/golden"))
            .expect_err("four billion passes should not be accepted");

        assert!(error.to_string().contains("iterations"), "{error}");

        let source = format!("{MINIMAL}\n[denoise]\niterations = {MAX_ITERATIONS}\n");
        let mut config: Config = toml::from_str(&source).unwrap();
        config
            .validate(Path::new("tests/golden"))
            .expect("the ceiling itself is a legal thing to ask for");
    }

    /// An output path is checked where `--debug`'s directory is: before the
    /// render, not after it.
    #[test]
    fn an_output_lands_in_a_directory_that_exists() {
        assert_eq!(parent_exists("render.png"), Ok(PathBuf::from("render.png")));
        assert_eq!(
            parent_exists("tests/golden/heat.png"),
            Ok(PathBuf::from("tests/golden/heat.png")),
        );
        assert!(parent_exists("tests/goldne/heat.png").is_err(), "a typo");
    }

    /// `--output` is the path every render writes, so it is the one a typo
    /// costs the most on — and the default has to get past the same check.
    #[test]
    fn the_render_is_refused_an_output_directory_that_does_not_exist() {
        let parse = |extra: &[&str]| {
            let args = ["wgsl-raytrace", "--config", "tests/golden/sky.toml"];
            Args::try_parse_from(args.iter().chain(extra))
        };

        assert!(parse(&["--output", "tests/goldne/render.png"]).is_err());
        assert_eq!(
            parse(&[]).expect("the default output is legal").output,
            PathBuf::from("render.png"),
        );
    }

    /// A light is its emitted color and nothing else: emission is always from
    /// the one face the triangle's winding points at, so there is no flag to
    /// misspell into the config.
    #[test]
    fn a_light_takes_no_field_beyond_what_it_emits() {
        let source = MINIMAL.replace(
            r#"material = "lambertian"
albedo = [0.42, 0.2, 0.7]"#,
            r#"material = "light"
emit = [3.0, 3.0, 3.0]"#,
        );

        let config: Config = toml::from_str(&source).unwrap();
        let Object::Wavefront(wavefront) = &config.objects[0];

        assert_eq!(
            wavefront.material,
            Material::Light {
                emit: [3.0, 3.0, 3.0]
            }
        );
        assert_eq!(wavefront.material.to_string(), "light[3.0, 3.0, 3.0]");

        assert!(
            toml::from_str::<Config>(&source.replace(
                "emit = [3.0, 3.0, 3.0]",
                "emit = [3.0, 3.0, 3.0]\ntwo_sided = true"
            ))
            .is_err(),
            "sidedness is not a thing a scene gets to ask about"
        );
    }

    fn with_material(material: &str) -> String {
        MINIMAL.replace(
            "material = \"lambertian\"\nalbedo = [0.42, 0.2, 0.7]",
            material,
        )
    }

    /// Every field left out takes Blender's default, so the bare name is a
    /// complete material.
    #[test]
    fn a_principled_material_takes_blenders_defaults() {
        let config: Config = toml::from_str(&with_material("material = \"principled\"")).unwrap();
        let Object::Wavefront(wavefront) = &config.objects[0];

        assert_eq!(
            wavefront.material,
            Material::Principled(Principled {
                base_color: [0.8, 0.8, 0.8],
                roughness: 0.5,
                metallic: 0.0,
                ior: 1.5,
                transmission: 0.0,
                alpha: 1.0,
            })
        );
        assert_eq!(
            wavefront.material,
            Material::Principled(Principled::default())
        );
    }

    #[test]
    fn a_principled_material_reads_every_field() {
        let source = with_material(
            "material = \"principled\"\nbase_color = [0.1, 0.2, 0.3]\n\
             roughness = 0.25\nmetallic = 1.0\nior = 1.33\ntransmission = 0.75\nalpha = 0.4",
        );
        let config: Config = toml::from_str(&source).unwrap();
        let Object::Wavefront(wavefront) = &config.objects[0];

        assert_eq!(
            wavefront.material,
            Material::Principled(Principled {
                base_color: [0.1, 0.2, 0.3],
                roughness: 0.25,
                metallic: 1.0,
                ior: 1.33,
                transmission: 0.75,
                alpha: 0.4,
            })
        );
        assert!(
            toml::from_str::<Config>(&format!(
                "{}\nsheen = 1.0",
                with_material("material = \"principled\"")
                    .split("[[objects.transform]]")
                    .next()
                    .unwrap()
            ))
            .is_err(),
            "a field the model does not have yet is named, not ignored"
        );
    }

    #[test]
    fn an_out_of_range_material_is_named_rather_than_rendered() {
        for (material, field) in [
            ("material = \"principled\"\nroughness = 1.5", "roughness"),
            ("material = \"principled\"\nroughness = -0.1", "roughness"),
            ("material = \"principled\"\nroughness = nan", "roughness"),
            ("material = \"principled\"\nmetallic = 2.0", "metallic"),
            (
                "material = \"principled\"\ntransmission = 1.1",
                "transmission",
            ),
            (
                "material = \"principled\"\ntransmission = -0.5",
                "transmission",
            ),
            ("material = \"principled\"\nalpha = 1.5", "alpha"),
            ("material = \"principled\"\nalpha = -0.25", "alpha"),
            ("material = \"principled\"\nalpha = nan", "alpha"),
            ("material = \"principled\"\nior = 0.9", "ior"),
            ("material = \"principled\"\nior = nan", "ior"),
            (
                "material = \"principled\"\nbase_color = [0.5, -0.1, 0.5]",
                "base_color",
            ),
            (
                "material = \"lambertian\"\nalbedo = [-1.0, 0.0, 0.0]",
                "albedo",
            ),
            (
                "material = \"metal\"\nalbedo = [0.5, 0.5, 0.5]\nroughness = 1.2",
                "roughness",
            ),
            ("material = \"light\"\nemit = [1.0, -1.0, 1.0]", "emit"),
            (
                "material = \"dielectric\"\nrefraction_index = 0.0",
                "refraction_index",
            ),
            (
                "material = \"dielectric\"\nrefraction_index = -1.5",
                "refraction_index",
            ),
            (
                "material = \"dielectric\"\nrefraction_index = nan",
                "refraction_index",
            ),
        ] {
            let mut config: Config = toml::from_str(&with_material(material)).unwrap();
            let error = config
                .validate(Path::new("tests/golden"))
                .expect_err(&format!("{material:?} should not be accepted"));

            assert!(error.to_string().contains(field), "{material:?}: {error}");
        }

        for material in [
            "material = \"principled\"",
            "material = \"principled\"\nroughness = 0.0\nmetallic = 1.0\nior = 1.0",
            "material = \"principled\"\nroughness = 1.0\nbase_color = [0.0, 0.0, 0.0]",
            "material = \"principled\"\ntransmission = 1.0\nroughness = 0.0",
            "material = \"principled\"\nalpha = 0.0",
            "material = \"principled\"\nalpha = 1.0",
            "material = \"dielectric\"\nrefraction_index = 1.0",
        ] {
            let mut config: Config = toml::from_str(&with_material(material)).unwrap();
            config
                .validate(Path::new("tests/golden"))
                .unwrap_or_else(|error| panic!("{material:?} is legal: {error}"));
        }
    }

    #[test]
    fn rejects_unsupported_shapes() {
        let scene = MINIMAL.replace(
            r#"shape = "wavefront""#,
            r#"shape = "sphere"
radius = 1.0"#,
        );

        assert!(toml::from_str::<Config>(&scene).is_err());
    }

    #[test]
    fn rejects_unknown_camera_fields() {
        let scene = MINIMAL.replace("[camera]", "[camera]\nthreads = 8");

        assert!(toml::from_str::<Config>(&scene).is_err());
    }

    #[test]
    fn missing_object_files_fail_validation() {
        let mut config: Config = toml::from_str(MINIMAL).unwrap();

        let error = config
            .validate(Path::new("does/not/exist"))
            .expect_err("missing file should not validate");
        assert!(error.to_string().contains("teapot.obj"));
    }

    /// The whole table is optional, and its absence is the black sky a config
    /// got from omitting `background` before the setting moved here.
    #[test]
    fn a_scene_without_an_environment_table_gets_a_black_sky() {
        let config: Config = toml::from_str(MINIMAL).unwrap();

        assert_eq!(config.environment.color, [0.0, 0.0, 0.0]);
        assert_eq!(config.environment.file, None);
        assert_eq!(config.environment.intensity, 1.0);
        assert_eq!(config.environment.rotation, 0.0);
    }

    #[test]
    fn an_environment_table_reads_a_map_and_how_to_aim_it() {
        let scene = format!(
            "{MINIMAL}\n[environment]\nfile = \"sky.exr\"\nintensity = 0.5\nrotation = 90.0\n"
        );
        let config: Config = toml::from_str(&scene).unwrap();

        assert_eq!(config.environment.file, Some(PathBuf::from("sky.exr")));
        assert_eq!(config.environment.intensity, 0.5);
        assert_eq!(config.environment.rotation, 90.0);
        // Still black, and still unread while there is a file to read instead.
        assert_eq!(config.environment.color, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn rejects_unknown_environment_fields() {
        let scene = format!("{MINIMAL}\n[environment]\nbackground = [1.0, 1.0, 1.0]\n");

        assert!(toml::from_str::<Config>(&scene).is_err());
    }

    /// `background` used to live on `[camera]`. A scene that still sets it there
    /// should say so rather than silently render the black it now defaults to.
    #[test]
    fn rejects_the_camera_background_that_moved_here() {
        // Spliced into `[camera]`, not appended: the end of MINIMAL is inside
        // an object's transform table.
        let (camera, rest) = MINIMAL.split_once("[[objects]]").unwrap();
        let scene = format!("{camera}background = [1.0, 1.0, 1.0]\n\n[[objects]]{rest}");

        assert!(toml::from_str::<Config>(&scene).is_err());
    }

    /// The map is resolved against the config's directory the way a mesh is, and
    /// a path that is not there is named before any GPU work starts.
    #[test]
    fn a_missing_environment_file_fails_validation() {
        // No objects, so the mesh check ahead of it cannot be what fails.
        let scene = format!(
            "{}\n[environment]\nfile = \"sky.exr\"\n",
            MINIMAL.split("[[objects]]").next().unwrap()
        );
        let mut config: Config = toml::from_str(&scene).unwrap();

        let error = config
            .validate(Path::new("does/not/exist"))
            .expect_err("a missing map should not validate");
        assert!(error.to_string().contains("sky.exr"), "{error}");
    }

    #[test]
    fn every_aspect_ratio_keeps_a_positive_height() {
        for ratio in [
            AspectRatios::Widescreen,
            AspectRatios::Square,
            AspectRatios::Smartphone,
            AspectRatios::Standard,
            AspectRatios::Cinema,
        ] {
            assert!(ratio.get_height(1) >= 1, "{} collapsed to zero", ratio);
        }
    }

    #[test]
    fn span_dump_quotes_the_offending_lines() {
        let content = "alpha\nbeta\ngamma\n";
        let span = content.find("beta").unwrap();

        let dump = span_dump(content, span..span + 4);
        assert!(dump.contains("beta"), "{dump}");
        assert!(!dump.contains("gamma"), "{dump}");
    }
}
