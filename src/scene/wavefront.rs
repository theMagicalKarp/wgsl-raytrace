use obj::raw::object::Polygon;
use obj::raw::object::RawObj;
use obj::raw::object::parse_obj;
use std::error::Error;
use std::fs;
use std::io::Cursor;
use std::path::Path;

/// Parses a `.obj` file, naming it in any error so the message points at a
/// scene, not at a line number in a file the reader has to go find.
pub(super) fn read(path: &Path) -> Result<RawObj, Box<dyn Error>> {
    let source =
        fs::read_to_string(path).map_err(|error| format!("{}: {}", path.display(), error))?;

    parse(&source).map_err(|error| format!("{}: {}", path.display(), error).into())
}

pub(super) fn parse(source: &str) -> Result<RawObj, Box<dyn Error>> {
    Ok(parse_obj(Cursor::new(as_groups(source).as_bytes()))?)
}

/// Rewrites single-word `o` lines as `g` lines.
fn as_groups(source: &str) -> String {
    source
        .lines()
        .map(|line| match line.trim_start().strip_prefix("o ") {
            Some(name) if !name.trim().is_empty() && !name.trim().contains(' ') => {
                format!("g {}", name.trim())
            }
            _ => line.to_string(),
        })
        .collect::<Vec<String>>()
        .join("\n")
}

/// One corner of a polygon: an index into the file's positions, and into its
/// normals when the file supplied one.
pub(super) type Corner = (usize, Option<usize>);

/// Flattens the four ways a `.obj` face can spell its corners into the two
/// pieces this tracer uses. Texture coordinates are dropped — no material here
/// samples one.
pub(super) fn corners(polygon: &Polygon) -> Vec<Corner> {
    match polygon {
        Polygon::P(corners) => corners.iter().map(|&p| (p, None)).collect(),
        Polygon::PT(corners) => corners.iter().map(|&(p, _)| (p, None)).collect(),
        Polygon::PN(corners) => corners.iter().map(|&(p, n)| (p, Some(n))).collect(),
        Polygon::PTN(corners) => corners.iter().map(|&(p, _, n)| (p, Some(n))).collect(),
    }
}

/// The polygons an object selects: the whole file, or one named block of it.
pub(super) fn selected<'a>(
    object: &'a RawObj,
    group: Option<&str>,
) -> Result<Vec<&'a Polygon>, Box<dyn Error>> {
    let Some(name) = group else {
        return Ok(object.polygons.iter().collect());
    };

    let Some(group) = object.groups.get(name) else {
        let mut available: Vec<&String> = object.groups.keys().collect();
        available.sort();

        return Err(format!("No group named {name:?}, the file has {available:?}").into());
    };

    Ok(group
        .polygons
        .iter()
        .flat_map(|range| object.polygons[range.start..range.end].iter())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::testing::BLOCKS;

    #[test]
    fn a_block_name_with_spaces_stays_an_object_name() {
        let source = BLOCKS.replace("o Left", "o Far Left");
        let object = parse(&source).expect("test mesh should parse");

        assert_eq!(object.name.as_deref(), Some("Far Left"));
        assert!(!object.groups.contains_key("Far Left"));
    }

    #[test]
    fn an_unknown_group_lists_what_the_file_has() {
        let object = parse(BLOCKS).expect("test mesh should parse");

        let error = selected(&object, Some("Middle")).expect_err("group should be missing");
        let message = error.to_string();
        assert!(message.contains("Middle"), "{message}");
        assert!(
            message.contains("Left") && message.contains("Right"),
            "{message}"
        );
    }
}
