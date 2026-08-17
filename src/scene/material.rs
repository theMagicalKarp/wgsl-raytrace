use crate::config::Material;
use bytemuck::Pod;
use bytemuck::Zeroable;

/// The surface models, as the shader's `switch` will see them.
///
/// Glass and water are absent on purpose: they are dielectrics with a fixed
/// index of refraction, flattened on the way in, so the shader never learns
/// that the scene format has names for them.
pub(crate) const LAMBERTIAN: u32 = 0;
pub(crate) const METAL: u32 = 1;
pub(crate) const DIELECTRIC: u32 = 2;
pub(crate) const LIGHT: u32 = 3;

/// A material as the shader reads it.
///
/// One `vec3<f32>` and two scalars cover every model in the scene format: the
/// vector is an albedo or an emitted radiance, and the scalar is a roughness or
/// an index of refraction. Which of those a field means is decided by `kind`,
/// and nothing else in the struct varies, so the buffer stays a flat array.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuMaterial {
    /// Albedo, or emitted radiance for a light. Dielectrics are colorless and
    /// carry white, so the shader can attenuate by this unconditionally.
    pub color: [f32; 3],

    /// One of the constants above.
    pub kind: u32,

    /// Metal roughness, or dielectric index of refraction. Unused otherwise.
    pub parameter: f32,

    _pad: [f32; 3],
}

const _: () = assert!(size_of::<GpuMaterial>() == 32);

impl From<&Material> for GpuMaterial {
    fn from(material: &Material) -> Self {
        const CLEAR: [f32; 3] = [1.0; 3];

        let (kind, color, parameter) = match material {
            Material::Lambertian { albedo } => (LAMBERTIAN, *albedo, 0.0),
            Material::Metal { albedo, roughness } => (METAL, *albedo, *roughness),
            Material::Dielectric { refraction_index } => (DIELECTRIC, CLEAR, *refraction_index),
            Material::Glass {} => (DIELECTRIC, CLEAR, 1.5),
            Material::Water {} => (DIELECTRIC, CLEAR, 1.33),
            Material::Light { emit } => (LIGHT, *emit, 0.0),
        };

        GpuMaterial {
            color,
            kind,
            parameter,
            _pad: [0.0; 3],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn materials_flatten_to_what_the_shader_understands() {
        let glass = GpuMaterial::from(&Material::Glass {});
        let water = GpuMaterial::from(&Material::Water {});

        assert_eq!(glass.kind, DIELECTRIC);
        assert_eq!(glass.parameter, 1.5);
        assert_eq!(water.kind, DIELECTRIC);
        assert_eq!(water.parameter, 1.33);
        assert_eq!(glass.color, [1.0; 3], "a dielectric should not tint");

        assert_eq!(
            GpuMaterial::from(&Material::Metal {
                albedo: [0.1, 0.2, 0.3],
                roughness: 0.4,
            }),
            GpuMaterial {
                color: [0.1, 0.2, 0.3],
                kind: METAL,
                parameter: 0.4,
                _pad: [0.0; 3],
            }
        );
    }
}
