use crate::math;
use crate::scene::GpuMaterial;
use crate::scene::GpuTriangle;
use crate::scene::material::LIGHT;
use bytemuck::Pod;
use bytemuck::Zeroable;

/// One emissive triangle, as an entry in the shader's cumulative distribution.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuLight {
    /// Index into [`Scene::triangles`](crate::scene::Scene::triangles).
    pub triangle: u32,

    /// The chance of drawing this entry or any before it. Entries ascend, and
    /// the last is exactly one, so a uniform draw in [0, 1) always lands on
    /// somebody.
    pub cdf: f32,
}

const _: () = assert!(size_of::<GpuLight>() == 8);

/// Rec. 709 luma. Which emitter matters most is a question about what the eye
/// will see, so a green light is weighted well above a blue one of the same
/// radiance.
fn luminance(color: [f32; 3]) -> f32 {
    0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
}

/// The triangle's area — half the length of the cross product of its edges,
/// which is how the shader recovers it too, so the table and the density the
/// shader divides by are talking about the same number.
fn area(triangle: &GpuTriangle) -> f32 {
    let e1 = math::sub(triangle.v1, triangle.v0);
    let e2 = math::sub(triangle.v2, triangle.v0);
    let normal = math::cross(e1, e2);
    math::dot(normal, normal).sqrt() * 0.5
}

/// The power an emissive triangle puts into the scene, and what the table is
/// built in proportion to. Anything that does not emit weighs nothing.
fn power(triangle: &GpuTriangle, materials: &[GpuMaterial]) -> f32 {
    let material = materials[triangle.material as usize];
    match material.kind == LIGHT {
        true => area(triangle) * luminance(material.color),
        false => 0.0,
    }
}

/// Builds the sampling table over `triangles`, and returns it with the total
/// power it distributes.
///
/// Triangles that emit nothing measurable are left out rather than given a
/// vanishing slice: an entry that is never drawn is an entry the shader still
/// has to search past. That includes a light whose color is black and a
/// degenerate triangle with no area, both of which would otherwise sit in the
/// table forever.
///
/// A scene with nothing left after that gets an empty table and a total of zero,
/// which is the shader's signal to skip direct lighting entirely.
///
/// What this deliberately does not do is account for where the light is being
/// sampled *from*. The table is built once, and a triangle's share of it cannot
/// depend on which way it faces relative to a shading point that does not exist
/// yet — so a closed emitter like melee's sphere still spends about half its
/// draws on its own far side, where the shadow ray is blocked by the near side.
/// Fixing that needs a structure queried per shading point rather than a flat
/// distribution.
pub(super) fn build(triangles: &[GpuTriangle], materials: &[GpuMaterial]) -> (Vec<GpuLight>, f32) {
    let emitters: Vec<(u32, f32)> = triangles
        .iter()
        .enumerate()
        .map(|(index, triangle)| (index as u32, power(triangle, materials)))
        .filter(|&(_, power)| power > 0.0)
        .collect();

    let total: f32 = emitters.iter().map(|&(_, power)| power).sum();
    if total <= 0.0 {
        return (Vec::new(), 0.0);
    }

    let mut running = 0.0;
    let mut table: Vec<GpuLight> = emitters
        .iter()
        .map(|&(triangle, power)| {
            running += power;
            GpuLight {
                triangle,
                cdf: running / total,
            }
        })
        .collect();

    // The running sum lands a rounding error either side of the total, and the
    // shader's search needs somewhere for a draw of 0.999… to go. Pinning the
    // last entry is what guarantees there is one.
    if let Some(last) = table.last_mut() {
        last.cdf = 1.0;
    }

    (table, total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Material;
    use crate::scene::testing::QUAD;
    use crate::scene::testing::triangles as load_triangles;
    use crate::scene::testing::wavefront;

    /// The unit quad from `testing`, fanned into two triangles of half a unit
    /// of area each, with `material` on both.
    fn quad(material: Material) -> (Vec<GpuTriangle>, Vec<GpuMaterial>) {
        (
            load_triangles(QUAD, &wavefront(None, vec![])),
            vec![GpuMaterial::from(&material)],
        )
    }

    #[test]
    fn a_scene_with_nothing_emissive_has_no_table() {
        let (triangles, materials) = quad(Material::Lambertian { albedo: [0.5; 3] });

        assert_eq!(build(&triangles, &materials), (Vec::new(), 0.0));
    }

    #[test]
    fn a_black_light_is_not_worth_an_entry() {
        // It would never be drawn, and every search would still walk past it.
        let (triangles, materials) = quad(Material::Light { emit: [0.0; 3] });

        assert_eq!(build(&triangles, &materials), (Vec::new(), 0.0));
    }

    #[test]
    fn equal_triangles_split_the_distribution_evenly() {
        let (triangles, materials) = quad(Material::Light { emit: [1.0; 3] });

        let (table, total) = build(&triangles, &materials);

        assert_eq!(table.len(), 2);
        assert_eq!(table[0].triangle, 0);
        assert_eq!(table[1].triangle, 1);
        assert!((table[0].cdf - 0.5).abs() < 1e-6, "{table:?}");
        assert_eq!(table[1].cdf, 1.0, "the last entry has to be exactly one");
        // Two triangles of half a unit each, at a luminance of one.
        assert!((total - 1.0).abs() < 1e-6, "{total}");
    }

    #[test]
    fn a_bigger_triangle_takes_a_bigger_share() {
        let (mut triangles, materials) = quad(Material::Light { emit: [1.0; 3] });

        // Three times the area of the half-unit triangle it replaces, so it
        // should be worth three draws in four.
        triangles[1].v0 = [0.0, 0.0, 0.0];
        triangles[1].v1 = [3.0, 0.0, 0.0];
        triangles[1].v2 = [0.0, 1.0, 0.0];

        let (table, _) = build(&triangles, &materials);

        assert!((table[0].cdf - 0.25).abs() < 1e-5, "{table:?}");
        assert_eq!(table[1].cdf, 1.0);
    }

    #[test]
    fn brightness_counts_as_much_as_size() {
        // Rec. 709 puts green well above red, so of two emitters the same size
        // the green one should take that much more of the distribution.
        let (mut triangles, _) = quad(Material::Light { emit: [1.0; 3] });
        let materials = vec![
            GpuMaterial::from(&Material::Light {
                emit: [1.0, 0.0, 0.0],
            }),
            GpuMaterial::from(&Material::Light {
                emit: [0.0, 1.0, 0.0],
            }),
        ];
        triangles[1].material = 1;

        let (table, _) = build(&triangles, &materials);

        let red = 0.2126 / (0.2126 + 0.7152);
        assert!((table[0].cdf - red).abs() < 1e-5, "{table:?}");
    }

    #[test]
    fn the_table_only_holds_what_emits() {
        let (mut triangles, _) = quad(Material::Light { emit: [3.0; 3] });
        let materials = vec![
            GpuMaterial::from(&Material::Lambertian { albedo: [0.5; 3] }),
            GpuMaterial::from(&Material::Light { emit: [3.0; 3] }),
        ];
        triangles[1].material = 1;

        let (table, _) = build(&triangles, &materials);

        assert_eq!(table.len(), 1);
        assert_eq!(table[0].triangle, 1, "the lambertian is not an emitter");
        assert_eq!(table[0].cdf, 1.0);
    }
}
