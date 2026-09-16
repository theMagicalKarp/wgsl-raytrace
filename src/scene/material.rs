use crate::config::Material;
use crate::config::Principled;
use bytemuck::Pod;
use bytemuck::Zeroable;

/// The surface models, as the shader's branches will see them.
///
/// Lambertian, metal and the dielectrics are absent on purpose: they are
/// principled surfaces with a few fields pinned. All of them are flattened on
/// the way in, so the shader never learns that the scene format has names for
/// them.
pub(crate) const PRINCIPLED: u32 = 0;
pub(crate) const LIGHT: u32 = 1;

/// A material as the shader reads it.
///
/// The same fields for every kind, so the buffer stays a flat array. A light
/// reads only `color`.
///
/// WGSL pads a struct to its largest alignment, which is the `vec3f`'s sixteen,
/// so the five scalars after `kind` round up to a trailing twelve bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuMaterial {
    /// Base color, or emitted radiance for a light.
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

    _pad: [f32; 3],
}

const _: () = assert!(size_of::<GpuMaterial>() == 48);

impl GpuMaterial {
    fn new(kind: u32, color: [f32; 3], principled: Principled) -> GpuMaterial {
        GpuMaterial {
            color,
            kind,
            roughness: principled.roughness,
            metallic: principled.metallic,
            ior: principled.ior,
            transmission: principled.transmission,
            alpha: principled.alpha,
            _pad: [0.0; 3],
        }
    }

    fn principled(principled: Principled) -> GpuMaterial {
        GpuMaterial::new(PRINCIPLED, principled.base_color, principled)
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
            alpha: 1.0,
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
                transmission: 0.0,
                alpha: 1.0,
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

            Material::Light { emit } => GpuMaterial::new(LIGHT, emit, Principled::default()),
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
                _pad: [0.0; 3],
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
                    _pad: [0.0; 3],
                },
                "{material}"
            );
        }
    }

    /// A light is its emitted radiance and nothing more. Which face it emits
    /// from is not a per-material question any more — the shader answers it the
    /// one way, from the triangle's winding — so nothing about sidedness has to
    /// survive the trip into the buffer.
    #[test]
    fn a_light_carries_its_emission_and_nothing_else() {
        let light = GpuMaterial::from(&Material::Light {
            emit: [1.0, 2.0, 3.0],
        });

        assert_eq!(light.kind, LIGHT);
        assert_eq!(light.color, [1.0, 2.0, 3.0]);
        assert_eq!(light.alpha, 1.0);
    }
}
