mod geometry;
mod material;
mod transform;
mod wavefront;

#[cfg(test)]
mod testing;

pub use geometry::GpuTriangle;
pub use material::GpuMaterial;

use crate::config::Config;
use crate::config::Object;
use std::error::Error;

/// Every mesh in the scene, flattened into world space.
#[derive(Debug)]
pub struct Scene {
    pub triangles: Vec<GpuTriangle>,
    pub materials: Vec<GpuMaterial>,
}

impl Scene {
    /// Reads every object's mesh and bakes it into one triangle list.
    ///
    /// Expects the paths [`Config::validate`] resolved; a missing file is
    /// reported here rather than assumed away.
    pub fn load(config: &Config) -> Result<Scene, Box<dyn Error>> {
        let mut scene = Scene {
            triangles: Vec::new(),
            materials: Vec::with_capacity(config.objects.len()),
        };

        for (index, Object::Wavefront(wavefront)) in config.objects.iter().enumerate() {
            scene.materials.push(GpuMaterial::from(&wavefront.material));

            let object = wavefront::read(&wavefront.file)?;
            geometry::append(&object, wavefront, index as u32, &mut scene.triangles)?;
        }

        Ok(scene)
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
