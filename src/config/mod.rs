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
    #[arg(short, long, default_value = "render.png")]
    pub output: PathBuf,

    /// Directly override the sample count listed in the configuration file
    #[arg(short, long)]
    pub samples: Option<u32>,
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

impl CameraOptions {
    pub fn get_dimensions(&self) -> (u32, u32) {
        (
            self.image_width,
            self.aspect_ratio.get_height(self.image_width),
        )
    }
}

/// The surface models the tracer knows how to scatter off of.
///
/// This is deliberately smaller than the CPU tracer's set: the shader carries a
/// single scalar per material, so anything needing a texture lookup (checkered,
/// image, noise) is left out until there is somewhere to put it.
#[derive(Deserialize, Debug, PartialEq)]
#[serde(tag = "material", deny_unknown_fields)]
pub enum Material {
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
        for object in &mut self.objects {
            let Object::Wavefront(wavefront) = object;

            wavefront.file = config_dir.join(&wavefront.file);
            if !wavefront.file.is_file() {
                return Err(
                    format!("Object file does not exist: {}", wavefront.file.display()).into(),
                );
            }
        }

        if let Some(file) = &mut self.environment.file {
            *file = config_dir.join(&file);
            if !file.is_file() {
                return Err(format!("Environment file does not exist: {}", file.display()).into());
            }
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
        assert_eq!(config.environment.color, [0.0, 0.0, 0.0]);
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
