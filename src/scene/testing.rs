use crate::config::Material;
use crate::config::Transform;
use crate::config::Wavefront;
use crate::math;
use crate::scene::geometry;
use crate::scene::geometry::GpuTriangle;
use crate::scene::wavefront::parse;
use std::error::Error;
use std::path::PathBuf;

/// A unit square in the xy plane as a single quad, with no normals.
pub(super) const QUAD: &str = "
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
f 1 2 3 4
";

/// Two named blocks, one spelled `o` and one spelled `g`, a triangle each.
pub(super) const BLOCKS: &str = "
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
o Left
f 1 2 3
g Right
f 3 2 1
";

pub(super) fn wavefront(group: Option<&str>, transform: Vec<Transform>) -> Wavefront {
    Wavefront {
        file: PathBuf::from("test.obj"),
        group: group.map(String::from),
        material: Material::Glass {},
        transform,
    }
}

pub(super) fn load(
    source: &str,
    wavefront: &Wavefront,
) -> Result<Vec<GpuTriangle>, Box<dyn Error>> {
    let object = parse(source)?;

    let mut out = Vec::new();
    geometry::append(&object, wavefront, 0, &mut out)?;
    Ok(out)
}

pub(super) fn triangles(source: &str, wavefront: &Wavefront) -> Vec<GpuTriangle> {
    load(source, wavefront).expect("test mesh should load")
}

pub(super) fn close(actual: [f32; 3], expected: [f32; 3]) {
    let error = math::sub(actual, expected)
        .map(f32::abs)
        .into_iter()
        .fold(0.0, f32::max);
    assert!(error < 1e-5, "expected {expected:?}, got {actual:?}");
}
