mod bvh;
mod geometry;
mod material;
mod transform;
mod wavefront;

#[cfg(test)]
mod testing;

pub use bvh::GpuBvhNode;
pub use geometry::GpuTriangle;
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
        let triangles = hierarchy
            .order
            .iter()
            .map(|&index| triangles[index as usize])
            .collect();

        Ok(Scene {
            triangles,
            materials,
            nodes: hierarchy.nodes,
            depth: hierarchy.max_depth,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::material::LAMBERTIAN;
    use crate::scene::material::METAL;
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

        assert_eq!(scene.materials.len(), 2);
        assert_eq!(scene.materials[0].kind, METAL);
        assert_eq!(scene.materials[1].kind, LAMBERTIAN);

        // Every triangle belongs to the object that asked for it, and the two
        // blocks together account for the whole file.
        let teapot = scene.triangles.iter().filter(|t| t.material == 0).count();
        let plane = scene.triangles.iter().filter(|t| t.material == 1).count();
        assert!(teapot > 0 && plane > 0);
        assert_eq!(teapot + plane, scene.triangles.len());

        let whole_file = triangles(
            &fs::read_to_string("examples/teapot/teapot.obj").unwrap(),
            &wavefront(None, vec![]),
        );
        assert_eq!(teapot + plane, whole_file.len());
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
