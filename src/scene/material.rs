use crate::config::Material;
use crate::config::Operand;
use crate::config::Principled;
use crate::scene::program::NO_PROGRAM;
use crate::scene::program::Programs;
use crate::scene::program::constant;
use crate::scene::program::emission_constant;
use bytemuck::Pod;
use bytemuck::Zeroable;

/// The surface models, as the shader's branches will see them.
///
/// Lambertian, metal, the dielectrics and lights are absent on purpose: they
/// are principled surfaces with a few fields pinned. All of them are flattened
/// on the way in, so the shader never learns that the scene format has names
/// for them.
pub(crate) const PRINCIPLED: u32 = 0;

/// A material as the shader reads it.
///
/// The same fields for every kind, so the buffer stays a flat array.
///
/// WGSL aligns a `vec3f` to sixteen bytes, so `emission` starts on the
/// boundary after `alpha`, and the struct as a whole rounds up to 80. Every
/// field keeps the offset it had before the one after it existed: the two
/// subsurface scalars went into the padding beside `alpha` rather than past
/// `emission`, which is what keeps the struct at 80 bytes instead of 96.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuMaterial {
    /// Base color.
    pub color: [f32; 3],

    /// One of the constants above.
    pub kind: u32,

    /// Microfacet roughness in `[0, 1]`.
    pub roughness: f32,

    /// Dielectric at zero, conductor at one.
    pub metallic: f32,

    /// Index of refraction.
    pub ior: f32,

    /// Opaque at zero, glass at one.
    pub transmission: f32,

    /// Coverage: the chance a ray is stopped by the surface at all. Also
    /// copied onto every triangle, so a shadow ray can answer it without a
    /// material lookup.
    pub alpha: f32,

    /// How much of the diffuse base is a random walk through the inside of the
    /// surface instead. Zeroed here when the walk would have nowhere to go.
    pub subsurface_weight: f32,

    /// Henyey-Greenstein asymmetry for that walk, in `[-1, 1]`.
    pub subsurface_anisotropy: f32,

    /// Where this surface's field table starts in the program buffer, or
    /// [`NO_PROGRAM`] when nothing about it is patterned.
    ///
    /// Sits in what was padding, so a material with inputs and one without are
    /// the same eighty bytes and the shader's single test against
    /// [`NO_PROGRAM`] is the whole cost of the feature on a scene that does not
    /// use it.
    pub inputs: u32,

    /// Radiance given off the front face: the scene's color times its
    /// strength, multiplied out here so the shader reads one number.
    ///
    /// The *upper bound* of that product when a pattern drives either half.
    /// The light table is built from this and `light_pdf` prices a direction by
    /// it, so it has to be a number known before the render starts; what the
    /// surface actually emits at a point is resolved per hit and is never
    /// larger. A bound only makes light sampling noisier, never wrong.
    pub emission: [f32; 3],

    /// The index the opaque part's specular coat reflects with: `ior` moved by
    /// `specular_ior_level`, worked out here so the shader reads one number.
    /// Exactly `ior` at the default level. Glass keeps `ior`.
    ///
    /// In what was padding after `emission`, as `specular_ior_level` is after
    /// the radius, so the struct is still eighty bytes.
    pub specular_ior: f32,

    /// Mean free path inside the surface per channel, in world units: the
    /// scene's radius times its scale, multiplied out here the way emission is.
    pub subsurface_radius: [f32; 3],

    /// The level `specular_ior` was worked out from, kept so that a hit whose
    /// `ior` is patterned can work it out again with the level it was given.
    pub specular_ior_level: f32,
}

const _: () = assert!(size_of::<GpuMaterial>() == 80);

/// The index whose reflectance at normal incidence is `ior`'s, scaled by
/// `level` the way Blender 4's Specular IOR Level scales it: twice the level,
/// so 0.5 leaves it alone.
///
/// Converting back to an index rather than scaling the Fresnel term keeps the
/// Fresnel term exact at every angle. The shader's `specular_ior` is the same
/// arithmetic, for a hit whose `ior` or level is patterned.
pub fn specular_ior(ior: f32, level: f32) -> f32 {
    // The default, answered before any arithmetic so that every scene written
    // before this existed reflects with the index it always did, to the bit.
    if level == 0.5 {
        return ior;
    }

    let r = (ior - 1.0) / (ior + 1.0);
    // A reflectance of one is an infinite index, and the Fresnel term turns
    // into a NaN on the way there.
    let root = (r * r * 2.0 * level).sqrt().min(MAX_SPECULAR_ROOT);
    (1.0 + root) / (1.0 - root)
}

/// The largest `sqrt(F0)` [`specular_ior`] converts, matching the shader's: an
/// index of about 400, a reflectance of 0.99, and a long way from the pole.
const MAX_SPECULAR_ROOT: f32 = 0.995;

/// Whether the subsurface walk has anywhere to go: some channel of the radius,
/// scaled, is positive. Asked of the scene's own numbers rather than of
/// [`GpuMaterial::subsurface_weight`], because a patterned weight has no
/// constant to ask and would read as zero.
fn walks(principled: &Principled) -> bool {
    principled
        .subsurface_radius
        .iter()
        .any(|channel| channel * principled.subsurface_scale > 0.0)
}

impl GpuMaterial {
    /// The surface as the shader reads it, with every patterned field compiled
    /// into `programs` and named by `inputs`.
    pub(super) fn build(
        material: &Material,
        programs: &mut Programs,
    ) -> Result<GpuMaterial, String> {
        let mut gpu = GpuMaterial::from(material);

        match material {
            Material::Principled(principled) => {
                gpu.inputs = programs.compile(principled, walks(principled))?;
            }
            Material::Light { emit } => {
                let principled = Material::light(emit);
                gpu.inputs = programs.compile(&principled, walks(&principled))?;
            }
            _ => {}
        }

        Ok(gpu)
    }

    fn principled(principled: &Principled) -> GpuMaterial {
        let radius = principled
            .subsurface_radius
            .map(|channel| channel * principled.subsurface_scale);

        // A walk with no distance to walk is not a walk. Every channel is at
        // zero exactly when the scale is, or when the radius is black, and the
        // limit as the radius shrinks is a diffuse surface of the base color —
        // which is what the lobe underneath already is. So the weight goes to
        // zero rather than the shader dividing by a mean free path of nothing.
        // A patterned weight is read the same way: what matters is whether the
        // walk has anywhere to go, and the radius is what says so. `compile`
        // drops the pattern alongside this.
        let weight = constant(&principled.subsurface_weight, [0.0; 3])[0];
        let subsurface_weight = match walks(principled) {
            true => weight,
            false => 0.0,
        };

        let ior = constant(&principled.ior, [1.5; 3])[0];
        let level = constant(&principled.specular_ior_level, [0.5; 3])[0];

        GpuMaterial {
            color: constant(&principled.base_color, [0.8; 3]),
            kind: PRINCIPLED,
            roughness: constant(&principled.roughness, [0.5; 3])[0],
            metallic: constant(&principled.metallic, [0.0; 3])[0],
            ior,
            transmission: constant(&principled.transmission, [0.0; 3])[0],
            alpha: principled.alpha,
            subsurface_weight,
            subsurface_anisotropy: principled.subsurface_anisotropy,
            inputs: NO_PROGRAM,
            emission: emission_constant(principled),
            specular_ior: specular_ior(ior, level),
            subsurface_radius: radius,
            specular_ior_level: level,
        }
    }

    /// Clear, colorless glass: nothing but the transmission lobe, perfectly
    /// smooth.
    fn dielectric(ior: f32) -> GpuMaterial {
        GpuMaterial::principled(&Principled {
            base_color: Operand::Color([1.0; 3]),
            roughness: Operand::Scalar(0.0),
            metallic: Operand::Scalar(0.0),
            ior: Operand::Scalar(ior),
            transmission: Operand::Scalar(1.0),
            ..Principled::default()
        })
    }
}

/// The surface with every field taken at its constant, which is all of them on
/// a scene that writes no patterns. [`GpuMaterial::build`] is what adds the
/// programs; this is what the loader uses when it only wants the alpha, and
/// what the tests compare against.
impl From<&Material> for GpuMaterial {
    fn from(material: &Material) -> Self {
        match material {
            Material::Principled(principled) => GpuMaterial::principled(principled),

            // An index of one makes the dielectric Fresnel term zero at every
            // angle, so there is no specular lobe left and this is the plain
            // diffuse it always was.
            Material::Lambertian { albedo } => GpuMaterial::principled(&Principled {
                base_color: Operand::Color(*albedo),
                roughness: Operand::Scalar(1.0),
                metallic: Operand::Scalar(0.0),
                ior: Operand::Scalar(1.0),
                ..Principled::default()
            }),

            Material::Metal { albedo, roughness } => GpuMaterial::principled(&Principled {
                base_color: Operand::Color(*albedo),
                roughness: Operand::Scalar(*roughness),
                metallic: Operand::Scalar(1.0),
                ..Principled::default()
            }),

            Material::Dielectric { refraction_index } => GpuMaterial::dielectric(*refraction_index),
            Material::Glass {} => GpuMaterial::dielectric(1.5),
            Material::Water {} => GpuMaterial::dielectric(1.33),

            Material::Light { emit } => GpuMaterial::principled(&Material::light(emit)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Input;
    use crate::config::Space;

    #[test]
    fn a_principled_material_is_uploaded_as_written() {
        let material = GpuMaterial::from(&Material::Principled(Box::new(Principled {
            base_color: Operand::Color([0.1, 0.2, 0.3]),
            roughness: Operand::Scalar(0.4),
            metallic: Operand::Scalar(0.5),
            ior: Operand::Scalar(1.6),
            transmission: Operand::Scalar(0.7),
            specular_ior_level: Operand::Scalar(0.5),
            normal: None,
            alpha: 0.8,
            emission_color: Operand::Color([0.5, 1.0, 2.0]),
            emission_strength: Operand::Scalar(3.0),
            subsurface_weight: Operand::Scalar(0.9),
            subsurface_radius: [1.0, 0.5, 0.25],
            subsurface_scale: 2.0,
            subsurface_anisotropy: -0.3,
        })));

        assert_eq!(
            material,
            GpuMaterial {
                color: [0.1, 0.2, 0.3],
                kind: PRINCIPLED,
                roughness: 0.4,
                metallic: 0.5,
                ior: 1.6,
                transmission: 0.7,
                alpha: 0.8,
                subsurface_weight: 0.9,
                subsurface_anisotropy: -0.3,
                inputs: NO_PROGRAM,
                emission: [1.5, 3.0, 6.0],
                specular_ior: 1.6,
                subsurface_radius: [2.0, 1.0, 0.5],
                specular_ior_level: 0.5,
            }
        );
    }

    /// The radius a walk is measured in is the scene's radius times its scale,
    /// multiplied out on the way in the way emission is.
    #[test]
    fn the_subsurface_radius_is_scaled_on_the_host() {
        let material = GpuMaterial::from(&Material::Principled(Box::new(Principled {
            subsurface_weight: Operand::Scalar(1.0),
            subsurface_radius: [1.0, 0.2, 0.1],
            subsurface_scale: 0.05,
            ..Principled::default()
        })));

        assert_eq!(
            material.subsurface_radius,
            [0.05, 0.010000001, 0.0050000004]
        );
        assert_eq!(material.subsurface_weight, 1.0);
    }

    /// A walk with nowhere to walk is not one. At a scale of zero — or a radius
    /// of black — the mean free path is nothing in every channel, the limit of
    /// the walk is the diffuse surface already underneath it, and the shader
    /// would otherwise be dividing by zero to find out.
    #[test]
    fn a_subsurface_with_no_radius_is_left_to_the_diffuse_lobe() {
        for radius in [[1.0, 0.2, 0.1], [0.0; 3]] {
            let scale = match radius[0] {
                0.0 => 0.05,
                _ => 0.0,
            };
            let material = GpuMaterial::from(&Material::Principled(Box::new(Principled {
                subsurface_weight: Operand::Scalar(1.0),
                subsurface_radius: radius,
                subsurface_scale: scale,
                ..Principled::default()
            })));

            assert_eq!(material.subsurface_weight, 0.0, "{radius:?} at {scale}");
            assert_eq!(material.subsurface_radius, [0.0; 3]);
        }
    }

    /// A pattern has no constant for the radius test to read, so asking it for
    /// one answers zero and would drop the field on every patterned weight —
    /// walk or no walk. What decides is the radius, which is a number either
    /// way.
    #[test]
    fn a_patterned_subsurface_weight_survives_a_walk_with_somewhere_to_go() {
        let mut programs = Programs::default();
        let material = Material::Principled(Box::new(Principled {
            subsurface_weight: Operand::Input(Input::Coordinates {
                space: Space::Uv,
                channel: None,
            }),
            subsurface_radius: [1.0, 0.2, 0.1],
            subsurface_scale: 0.05,
            ..Principled::default()
        }));

        let gpu = GpuMaterial::build(&material, &mut programs).expect("should compile");
        assert_ne!(gpu.inputs, NO_PROGRAM, "the weight is patterned");
        assert_ne!(
            crate::scene::program::table(
                &programs,
                gpu.inputs,
                crate::scene::program::FIELD_SUBSURFACE_WEIGHT
            ),
            NO_PROGRAM,
            "and its program is the one the shader resolves"
        );
    }

    /// Every preset that is not written as a principled surface is opaque all
    /// the way through.
    #[test]
    fn the_presets_do_not_scatter_under_the_surface() {
        for material in [
            Material::Lambertian { albedo: [0.5; 3] },
            Material::Metal {
                albedo: [0.5; 3],
                roughness: 0.2,
            },
            Material::Glass {},
            Material::Water {},
            Material::Light {
                emit: Operand::Color([1.0; 3]),
            },
        ] {
            assert_eq!(
                GpuMaterial::from(&material).subsurface_weight,
                0.0,
                "{material}"
            );
        }
    }

    #[test]
    fn a_lambertian_is_a_principled_surface_with_no_specular() {
        let material = GpuMaterial::from(&Material::Lambertian {
            albedo: [0.1, 0.2, 0.3],
        });

        assert_eq!(material.kind, PRINCIPLED);
        assert_eq!(material.color, [0.1, 0.2, 0.3]);
        assert_eq!(material.roughness, 1.0);
        assert_eq!(material.metallic, 0.0);
        assert_eq!(material.ior, 1.0, "no Fresnel reflection at any angle");
        assert_eq!((material.transmission, material.alpha), (0.0, 1.0));
        assert_eq!(material.emission, [0.0; 3]);
    }

    #[test]
    fn a_metal_is_a_fully_metallic_principled_surface() {
        let material = GpuMaterial::from(&Material::Metal {
            albedo: [0.1, 0.2, 0.3],
            roughness: 0.4,
        });

        assert_eq!(material.kind, PRINCIPLED);
        assert_eq!(material.color, [0.1, 0.2, 0.3]);
        assert_eq!(material.roughness, 0.4);
        assert_eq!(material.metallic, 1.0);
        assert_eq!((material.transmission, material.alpha), (0.0, 1.0));
    }

    #[test]
    fn dielectrics_are_smooth_clear_principled_glass() {
        for (material, ior) in [
            (Material::Glass {}, 1.5),
            (Material::Water {}, 1.33),
            (
                Material::Dielectric {
                    refraction_index: 2.4,
                },
                2.4,
            ),
        ] {
            assert_eq!(
                GpuMaterial::from(&material),
                GpuMaterial {
                    color: [1.0; 3],
                    kind: PRINCIPLED,
                    roughness: 0.0,
                    metallic: 0.0,
                    ior,
                    transmission: 1.0,
                    alpha: 1.0,
                    subsurface_weight: 0.0,
                    subsurface_anisotropy: 0.0,
                    inputs: NO_PROGRAM,
                    emission: [0.0; 3],
                    specular_ior: ior,
                    subsurface_radius: [0.05, 0.010000001, 0.0050000004],
                    specular_ior_level: 0.5,
                },
                "{material}"
            );
        }
    }

    /// A light is a principled surface that scatters nothing: black, with no
    /// metal, glass or specular coat to reflect with, and its emitted radiance
    /// carried through unchanged. Which face it emits from is not a
    /// per-material question — the shader answers it the one way, from the
    /// triangle's winding — so nothing about sidedness has to survive the trip
    /// into the buffer.
    #[test]
    fn a_light_is_a_principled_surface_that_only_emits() {
        let light = GpuMaterial::from(&Material::Light {
            emit: Operand::Color([1.0, 2.0, 3.0]),
        });

        assert_eq!(light.kind, PRINCIPLED);
        assert_eq!(light.emission, [1.0, 2.0, 3.0]);
        assert_eq!(light.color, [0.0; 3]);
        assert_eq!(light.metallic, 0.0);
        assert_eq!(light.transmission, 0.0);
        assert_eq!(light.ior, 1.0, "no Fresnel reflection at any angle");
        assert_eq!(light.alpha, 1.0);
    }

    /// The default level is the index as written, to the bit, and the two
    /// ends are no coat at all and twice the reflectance.
    #[test]
    fn the_specular_level_moves_the_coat_and_only_the_coat() {
        let reflectance = |ior: f32| ((ior - 1.0) / (ior + 1.0)).powi(2);

        assert_eq!(specular_ior(1.5, 0.5), 1.5);
        assert_eq!(
            specular_ior(1.5, 0.0),
            1.0,
            "no reflectance is an index of one"
        );
        assert!((reflectance(specular_ior(1.5, 1.0)) - 2.0 * reflectance(1.5)).abs() < 1e-6);
        assert!((reflectance(specular_ior(1.5, 0.25)) - 0.5 * reflectance(1.5)).abs() < 1e-6);
        assert!(specular_ior(1000.0, 1.0).is_finite(), "held off the pole");

        let material = GpuMaterial::from(&Material::Principled(Box::new(Principled {
            ior: Operand::Scalar(1.45),
            specular_ior_level: Operand::Scalar(0.0),
            ..Principled::default()
        })));
        assert_eq!(material.ior, 1.45, "glass keeps the index as written");
        assert_eq!(material.specular_ior, 1.0);
        assert_eq!(material.specular_ior_level, 0.0);
    }

    #[test]
    fn a_principled_surface_emits_nothing_by_default() {
        let material = GpuMaterial::from(&Material::Principled(Box::default()));

        assert_eq!(material.emission, [0.0; 3]);
    }
}
