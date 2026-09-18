use crate::config::Material;
use crate::config::Principled;
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
/// boundary after `alpha`, and the struct as a whole rounds up to 64. The fields
/// ahead of it keep the offsets they had before it existed.
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

    _pad0: [f32; 3],

    /// Radiance given off the front face: the scene's color times its
    /// strength, multiplied out here so the shader reads one number.
    pub emission: [f32; 3],

    _pad1: f32,
}

const _: () = assert!(size_of::<GpuMaterial>() == 64);

impl GpuMaterial {
    fn principled(principled: Principled) -> GpuMaterial {
        GpuMaterial {
            color: principled.base_color,
            kind: PRINCIPLED,
            roughness: principled.roughness,
            metallic: principled.metallic,
            ior: principled.ior,
            transmission: principled.transmission,
            alpha: principled.alpha,
            _pad0: [0.0; 3],
            emission: principled
                .emission_color
                .map(|channel| channel * principled.emission_strength),
            _pad1: 0.0,
        }
    }

    /// Clear, colorless glass: nothing but the transmission lobe, perfectly
    /// smooth.
    fn dielectric(ior: f32) -> GpuMaterial {
        GpuMaterial::principled(Principled {
            base_color: [1.0; 3],
            roughness: 0.0,
            metallic: 0.0,
            ior,
            transmission: 1.0,
            ..Principled::default()
        })
    }
}

impl From<&Material> for GpuMaterial {
    fn from(material: &Material) -> Self {
        match *material {
            Material::Principled(principled) => GpuMaterial::principled(principled),

            // An index of one makes the dielectric Fresnel term zero at every
            // angle, so there is no specular lobe left and this is the plain
            // diffuse it always was.
            Material::Lambertian { albedo } => GpuMaterial::principled(Principled {
                base_color: albedo,
                roughness: 1.0,
                metallic: 0.0,
                ior: 1.0,
                ..Principled::default()
            }),

            Material::Metal { albedo, roughness } => GpuMaterial::principled(Principled {
                base_color: albedo,
                roughness,
                metallic: 1.0,
                ..Principled::default()
            }),

            Material::Dielectric { refraction_index } => GpuMaterial::dielectric(refraction_index),
            Material::Glass {} => GpuMaterial::dielectric(1.5),
            Material::Water {} => GpuMaterial::dielectric(1.33),

            // Black, and at an index of one so that there is no specular coat
            // either: a surface that scatters nothing, and gives off `emit`.
            // Written as a strength of one on the color rather than the other
            // way around, so the radiance is `emit` exactly and not a rounding
            // of it.
            Material::Light { emit } => GpuMaterial::principled(Principled {
                base_color: [0.0; 3],
                ior: 1.0,
                emission_color: emit,
                emission_strength: 1.0,
                ..Principled::default()
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_principled_material_is_uploaded_as_written() {
        let material = GpuMaterial::from(&Material::Principled(Principled {
            base_color: [0.1, 0.2, 0.3],
            roughness: 0.4,
            metallic: 0.5,
            ior: 1.6,
            transmission: 0.7,
            alpha: 0.8,
            emission_color: [0.5, 1.0, 2.0],
            emission_strength: 3.0,
        }));

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
                _pad0: [0.0; 3],
                emission: [1.5, 3.0, 6.0],
                _pad1: 0.0,
            }
        );
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
                    _pad0: [0.0; 3],
                    emission: [0.0; 3],
                    _pad1: 0.0,
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
            emit: [1.0, 2.0, 3.0],
        });

        assert_eq!(light.kind, PRINCIPLED);
        assert_eq!(light.emission, [1.0, 2.0, 3.0]);
        assert_eq!(light.color, [0.0; 3]);
        assert_eq!(light.metallic, 0.0);
        assert_eq!(light.transmission, 0.0);
        assert_eq!(light.ior, 1.0, "no Fresnel reflection at any angle");
        assert_eq!(light.alpha, 1.0);
    }

    #[test]
    fn a_principled_surface_emits_nothing_by_default() {
        let material = GpuMaterial::from(&Material::Principled(Principled::default()));

        assert_eq!(material.emission, [0.0; 3]);
    }
}
