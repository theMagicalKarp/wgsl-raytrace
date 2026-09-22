use crate::config::Wavefront;
use crate::math;
use crate::scene::material::GpuMaterial;
use crate::scene::transform::Model;
use crate::scene::wavefront::Corner;
use crate::scene::wavefront::corners;
use crate::scene::wavefront::selected;
use bytemuck::Pod;
use bytemuck::Zeroable;
use obj::raw::object::RawObj;
use std::error::Error;

/// A triangle as the shader reads it, with the model transform already applied.
///
/// A `vec3f` is 16-aligned, so each one leaves a scalar's worth of room behind
/// it. Two of those carry something: the material index, and the material's
/// alpha, which a shadow ray needs on every triangle it meets and would
/// otherwise have to fetch the material for.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuTriangle {
    pub v0: [f32; 3],
    /// Index into [`Scene::materials`](crate::scene::Scene::materials).
    pub material: u32,
    pub v1: [f32; 3],
    /// The material's coverage, as [`GpuMaterial::alpha`].
    pub alpha: f32,
    pub v2: [f32; 3],
    _pad2: f32,
    pub n0: [f32; 3],
    _pad3: f32,
    pub n1: [f32; 3],
    _pad4: f32,
    pub n2: [f32; 3],
    _pad5: f32,
}

const _: () = assert!(size_of::<GpuTriangle>() == 96);

/// The shading attributes of one triangle, in the same order as
/// [`GpuTriangle`], as the shader reads them.
///
/// A second buffer rather than three more fields on the triangle, because
/// traversal is the hot loop and it never wants these: a ray tests a few dozen
/// triangles to find one hit, and only the hit is shaded. Keeping the tested
/// bytes and the shaded bytes apart is what stops a texture coordinate from
/// costing a cache line on every miss.
///
/// A `vec2f` is 8-aligned, so three of them pack with no padding at all.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuAttributes {
    /// Texture coordinates as the `.obj` spelled them, one per corner, matching
    /// `v0`, `v1` and `v2`. Zero on a face whose file supplied none, which is
    /// every mesh here but melee's.
    ///
    /// Not flipped, wrapped or clamped: these are the file's numbers, and what
    /// a coordinate outside the unit square means is the sampler's question,
    /// not the loader's.
    pub uv0: [f32; 2],
    pub uv1: [f32; 2],
    pub uv2: [f32; 2],
}

const _: () = assert!(size_of::<GpuAttributes>() == 24);

/// Appends `wavefront`'s triangles, in world space and tagged with `material`
/// and its alpha.
///
/// Faces with more than three corners are fanned from their first corner, which
/// is right for the convex faces an exporter emits. A corner without a normal
/// falls back to the face's geometric normal, so the shader can interpolate
/// unconditionally and never has to ask whether a triangle is smooth-shaded.
///
/// An object that yields nothing is an error rather than a silent no-op: it
/// means a mesh, a group name or a whole file is not what the scene thought it
/// was, and that is worth hearing before a GPU spends minutes on the rest.
pub(super) fn append(
    object: &RawObj,
    wavefront: &Wavefront,
    material: u32,
    out: &mut Vec<GpuTriangle>,
    attributes: &mut Vec<GpuAttributes>,
) -> Result<(), Box<dyn Error>> {
    let model = Model::new(&wavefront.transform);
    let alpha = GpuMaterial::from(&wavefront.material).alpha;
    let before = out.len();

    // Indices are bounds-checked while parsing, so these lookups cannot fail.
    let position = |corner: Corner| {
        let (x, y, z, _) = object.positions[corner.0];
        math::transform_point(model.points, [x, y, z])
    };
    let texture = |corner: Corner| {
        corner.1.map_or([0.0, 0.0], |index| {
            let (u, v, _) = object.tex_coords[index];
            [u, v]
        })
    };
    let normal = |corner: Corner| {
        corner.2.map(|index| {
            let (x, y, z) = object.normals[index];
            math::transform_direction(model.normals, [x, y, z])
        })
    };

    for polygon in selected(object, wavefront.group.as_deref())? {
        let corners = corners(polygon);

        for corner in 1..corners.len().saturating_sub(1) {
            let fan = [corners[0], corners[corner], corners[corner + 1]];
            let points = fan.map(position);

            let face = math::normalize(math::cross(
                math::sub(points[1], points[0]),
                math::sub(points[2], points[0]),
            ));
            let normals = fan.map(|corner| normal(corner).unwrap_or(face));

            out.push(GpuTriangle {
                v0: points[0],
                material,
                v1: points[1],
                alpha,
                v2: points[2],
                _pad2: 0.0,
                n0: normals[0],
                _pad3: 0.0,
                n1: normals[1],
                _pad4: 0.0,
                n2: normals[2],
                _pad5: 0.0,
            });
            let uv = fan.map(texture);
            attributes.push(GpuAttributes {
                uv0: uv[0],
                uv1: uv[1],
                uv2: uv[2],
            });
        }
    }

    match out.len() > before {
        true => Ok(()),
        false => Err(format!("{wavefront} contributed no triangles").into()),
    }
}

#[cfg(test)]
mod tests {
    use crate::config::Material;
    use crate::config::Principled;
    use crate::config::Transform;
    use crate::scene::testing::BLOCKS;
    use crate::scene::testing::QUAD;
    use crate::scene::testing::attributes;
    use crate::scene::testing::close;
    use crate::scene::testing::load;
    use crate::scene::testing::triangles;
    use crate::scene::testing::wavefront;

    #[test]
    fn fans_a_quad_into_two_triangles() {
        let out = triangles(QUAD, &wavefront(None, vec![]));

        assert_eq!(out.len(), 2);
        assert_eq!(out[0].v0, [0.0, 0.0, 0.0]);
        assert_eq!(out[0].v2, out[1].v1, "the fan should share an edge");
    }

    #[test]
    fn falls_back_to_the_face_normal() {
        let out = triangles(QUAD, &wavefront(None, vec![]));

        for triangle in out {
            close(triangle.n0, [0.0, 0.0, 1.0]);
            assert_eq!([triangle.n0, triangle.n1], [triangle.n2; 2]);
        }
    }

    #[test]
    fn an_object_that_contributes_nothing_is_an_error() {
        let error = load("v 0.0 0.0 0.0\n", &wavefront(None, vec![]))
            .expect_err("a file with no faces has nothing to render");

        assert!(error.to_string().contains("test.obj"), "{error}");
    }

    /// Shadow rays read coverage off the triangle rather than the material, so
    /// every triangle has to carry its object's.
    #[test]
    fn every_triangle_carries_its_materials_alpha() {
        let mut object = wavefront(None, vec![]);
        object.material = Material::Principled(Box::new(Principled {
            alpha: 0.3,
            ..Principled::default()
        }));
        let cutout = triangles(QUAD, &object);

        object.material = Material::Lambertian { albedo: [0.5; 3] };
        let opaque = triangles(QUAD, &object);

        assert_eq!(cutout.len(), 2);
        assert!(cutout.iter().all(|triangle| triangle.alpha == 0.3));
        assert!(opaque.iter().all(|triangle| triangle.alpha == 1.0));
    }

    /// A fan splits one face into triangles, and each one has to take the
    /// texture coordinates of the corners it was actually built from — the same
    /// three corners its positions came from, or the coordinates slide across
    /// the quad's diagonal.
    #[test]
    fn keeps_the_texture_coordinates_a_file_supplies() {
        let source = "
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.0 1.0
f 1/1 2/2 3/3 4/4
";

        let out = attributes(source, &wavefront(None, vec![]));

        assert_eq!(out.len(), 2);
        assert_eq!(out[0].uv0, [0.0, 0.0]);
        assert_eq!(out[0].uv1, [1.0, 0.0]);
        assert_eq!(out[0].uv2, [1.0, 1.0]);
        assert_eq!(out[1].uv0, [0.0, 0.0]);
        assert_eq!(out[1].uv1, [1.0, 1.0]);
        assert_eq!(out[1].uv2, [0.0, 1.0]);
    }

    /// Nearly every mesh here has no `vt` lines at all, and one that does not
    /// is not an error — it is a surface no input can be mapped onto.
    #[test]
    fn a_face_without_texture_coordinates_gets_zeros() {
        let out = attributes(QUAD, &wavefront(None, vec![]));

        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|face| *face == Default::default()));
    }

    /// Texture coordinates belong to the surface, not to where it is standing.
    #[test]
    fn texture_coordinates_ignore_the_model_transform() {
        let source = "
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
vt 0.25 0.75
f 1/1 2/1 3/1
";
        let moved = Transform::Translate {
            offset: [5.0, 0.0, 0.0],
        };

        let out = attributes(source, &wavefront(None, vec![moved]));

        assert_eq!(out[0].uv0, [0.25, 0.75]);
    }

    #[test]
    fn keeps_the_normals_a_file_supplies() {
        let source = "
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
vn 1.0 0.0 0.0
vn 0.0 1.0 0.0
f 1//1 2//2 3//1
";

        let out = triangles(source, &wavefront(None, vec![]));

        assert_eq!(out.len(), 1);
        close(out[0].n0, [1.0, 0.0, 0.0]);
        close(out[0].n1, [0.0, 1.0, 0.0]);
        close(out[0].n2, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn selects_a_block_named_either_way() {
        let left = triangles(BLOCKS, &wavefront(Some("Left"), vec![]));
        let right = triangles(BLOCKS, &wavefront(Some("Right"), vec![]));

        assert_eq!(left.len(), 1);
        assert_eq!(right.len(), 1);
        close(left[0].n0, [0.0, 0.0, 1.0]);
        close(right[0].n0, [0.0, 0.0, -1.0]);
    }

    #[test]
    fn transforms_apply_in_the_order_they_are_listed() {
        let scale = || Transform::Scale { scalar: [2.0; 3] };
        let translate = || Transform::Translate {
            offset: [1.0, 0.0, 0.0],
        };

        let scaled_first = triangles(QUAD, &wavefront(None, vec![scale(), translate()]));
        let moved_first = triangles(QUAD, &wavefront(None, vec![translate(), scale()]));

        close(scaled_first[0].v0, [1.0, 0.0, 0.0]);
        close(moved_first[0].v0, [2.0, 0.0, 0.0]);
    }

    #[test]
    fn normals_survive_a_non_uniform_scale() {
        // A 45° ramp: flattening it to a tenth of its height should leave the
        // surface nearly flat, so its normal turns *up*, not down toward the
        // slope the model matrix would give it.
        let source = "
v 0.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 1.0
f 1 2 3
";
        let squash = Transform::Scale {
            scalar: [1.0, 0.1, 1.0],
        };

        let out = triangles(source, &wavefront(None, vec![squash]));
        let normal = out[0].n0;

        let flat = triangles(source, &wavefront(None, vec![]));
        assert!(
            normal[1].abs() > flat[0].n0[1].abs(),
            "squashing should tip the normal toward vertical, got {normal:?}",
        );
    }
}
