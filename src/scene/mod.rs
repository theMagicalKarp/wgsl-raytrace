mod bvh;
mod geometry;
mod light;
mod material;
mod transform;
mod wavefront;

#[cfg(test)]
mod testing;

pub use bvh::GpuBvhNode;
pub use geometry::GpuTriangle;
pub use light::GpuLight;
pub use material::GpuMaterial;

use crate::config::Config;
use crate::config::Object;
use std::error::Error;

/// Every mesh in the scene, flattened into world space and indexed by a BVH.
#[derive(Debug)]
pub struct Scene {
    /// In the hierarchy's leaf order, not the order the config listed them:
    /// that is what lets a leaf name its triangles with an offset and a count,
    /// and it keeps the triangles one leaf tests next to each other in memory.
    pub triangles: Vec<GpuTriangle>,
    pub materials: Vec<GpuMaterial>,
    /// The distribution the shader draws emitters from, built over `triangles`
    /// after the permutation above so its indices address the list the shader is
    /// handed. Empty when nothing in the scene emits.
    pub lights: Vec<GpuLight>,
    /// Total power the entries in `lights` distribute between them, which is
    /// what turns one of their shares back into a probability. Zero alongside an
    /// empty `lights`.
    pub light_power: f32,
    /// The hierarchy over `triangles`, flattened. Node 0 is the root.
    pub nodes: Vec<GpuBvhNode>,
    /// Depth of the hierarchy's deepest leaf, with the root at zero.
    pub depth: u32,
}

impl Scene {
    /// Reads every object's mesh and bakes it into one triangle list.
    ///
    /// Expects the paths [`Config::validate`] resolved; a missing file is
    /// reported here rather than assumed away.
    pub fn load(config: &Config) -> Result<Scene, Box<dyn Error>> {
        let mut triangles = Vec::new();
        let mut materials = Vec::with_capacity(config.objects.len());

        for (index, Object::Wavefront(wavefront)) in config.objects.iter().enumerate() {
            materials.push(GpuMaterial::from(&wavefront.material));

            let object = wavefront::read(&wavefront.file)?;
            geometry::append(&object, wavefront, index as u32, &mut triangles)?;
        }

        // The shader walks the tree rather than the triangle list, so the
        // triangles are permuted into leaf order before they are handed over.
        let bounds: Vec<bvh::Aabb> = triangles
            .iter()
            .map(|triangle| bvh::Aabb::of_points([triangle.v0, triangle.v1, triangle.v2]))
            .collect();
        let hierarchy = bvh::build(&bounds);
        let triangles: Vec<GpuTriangle> = hierarchy
            .order
            .iter()
            .map(|&index| triangles[index as usize])
            .collect();

        let (lights, light_power) = light::build(&triangles, &materials);

        Ok(Scene {
            triangles,
            materials,
            lights,
            light_power,
            nodes: hierarchy.nodes,
            depth: hierarchy.max_depth,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::material::LAMBERTIAN;
    use crate::scene::material::LIGHT;
    use crate::scene::testing::triangles;
    use crate::scene::testing::wavefront;
    use std::fs;
    use std::path::Path;

    #[test]
    fn the_example_scene_loads() {
        let source = fs::read_to_string("examples/teapot/render.toml").unwrap();
        let mut config: Config = toml::from_str(&source).unwrap();
        config.validate(Path::new("examples/teapot")).unwrap();

        let scene = Scene::load(&config).unwrap();

        // Both blocks are lambertian; they stay separate materials because
        // they ask for different albedos, and they keep the order the file
        // lists them in.
        assert_eq!(scene.materials.len(), 2);
        assert_eq!(scene.materials[0].kind, LAMBERTIAN);
        assert_eq!(scene.materials[1].kind, LAMBERTIAN);
        assert_eq!(scene.materials[0].color, [0.72, 0.72, 0.75]);
        assert_eq!(scene.materials[1].color, [0.3, 0.72, 0.3]);

        // Every triangle belongs to the object that asked for it, and each
        // block holds exactly the group it named.
        let obj = fs::read_to_string("examples/teapot/teapot.obj").unwrap();
        let teapot = scene.triangles.iter().filter(|t| t.material == 0).count();
        let plane = scene.triangles.iter().filter(|t| t.material == 1).count();
        assert!(teapot > 0 && plane > 0);
        assert_eq!(teapot + plane, scene.triangles.len());
        assert_eq!(
            teapot,
            triangles(&obj, &wavefront(Some("Teapot"), vec![])).len()
        );
        assert_eq!(
            plane,
            triangles(&obj, &wavefront(Some("Plane"), vec![])).len()
        );

        // The file also holds a group the scene never asks for, so selecting by
        // group has to leave something out rather than take the whole file.
        let whole_file = triangles(&obj, &wavefront(None, vec![]));
        assert!(scene.triangles.len() < whole_file.len());
    }

    #[test]
    fn the_hierarchy_accounts_for_every_triangle() {
        // The shader only ever reaches a triangle through a leaf, so one left
        // out of the tree is one the render silently drops.
        let source = fs::read_to_string("examples/teapot/render.toml").unwrap();
        let mut config: Config = toml::from_str(&source).unwrap();
        config.validate(Path::new("examples/teapot")).unwrap();

        let scene = Scene::load(&config).unwrap();

        let mut covered = vec![0u32; scene.triangles.len()];
        for node in &scene.nodes {
            for offset in 0..node.primitive_count {
                covered[(node.left_or_first + offset) as usize] += 1;
            }
        }

        assert!(
            covered.iter().all(|&times| times == 1),
            "every triangle should sit in exactly one leaf",
        );
        assert!(scene.depth > 0, "1576 triangles should not be one leaf");
    }

    #[test]
    fn the_light_table_covers_every_emissive_triangle() {
        // Melee is the example with a light in it: a mesh, so the table should
        // hold all of its triangles and nothing else.
        let source = fs::read_to_string("examples/melee/render.toml").unwrap();
        let mut config: Config = toml::from_str(&source).unwrap();
        config.validate(Path::new("examples/melee")).unwrap();

        let scene = Scene::load(&config).unwrap();

        let emissive = scene
            .triangles
            .iter()
            .filter(|t| scene.materials[t.material as usize].kind == LIGHT)
            .count();
        assert!(emissive > 0, "melee's light should have survived the load");
        assert_eq!(scene.lights.len(), emissive);
        assert!(scene.light_power > 0.0);

        // The entries address the permuted list the shader reads, not the order
        // the config listed the objects in, and they ascend to exactly one.
        for entry in &scene.lights {
            let triangle = scene.triangles[entry.triangle as usize];
            assert_eq!(scene.materials[triangle.material as usize].kind, LIGHT);
        }
        assert!(scene.lights.windows(2).all(|p| p[0].cdf <= p[1].cdf));
        assert_eq!(scene.lights.last().unwrap().cdf, 1.0);
    }

    #[test]
    fn a_scene_without_an_emitter_has_no_lights() {
        let source = fs::read_to_string("examples/teapot/render.toml").unwrap();
        let mut config: Config = toml::from_str(&source).unwrap();
        config.validate(Path::new("examples/teapot")).unwrap();

        let scene = Scene::load(&config).unwrap();

        assert!(scene.lights.is_empty());
        assert_eq!(scene.light_power, 0.0);
    }

    #[test]
    fn a_missing_group_fails_the_whole_scene() {
        let source = fs::read_to_string("examples/teapot/render.toml")
            .unwrap()
            .replace(r#"group = "Teapot""#, r#"group = "Nothing""#);
        let mut config: Config = toml::from_str(&source).unwrap();
        config.validate(Path::new("examples/teapot")).unwrap();

        let error = Scene::load(&config).expect_err("the group does not exist");
        assert!(error.to_string().contains("Nothing"), "{error}");
    }
}
