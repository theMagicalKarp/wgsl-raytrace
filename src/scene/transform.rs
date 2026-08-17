use crate::config::Axis;
use crate::config::Transform;
use crate::math;
use crate::math::Mat4;

/// The model transform, and the matrix that carries its normals.
pub(super) struct Model {
    pub(super) points: Mat4,
    pub(super) normals: Mat4,
}

impl Model {
    pub(super) fn new(transforms: &[Transform]) -> Model {
        transforms.iter().fold(
            Model {
                points: math::IDENTITY,
                normals: math::IDENTITY,
            },
            |model, transform| {
                let (points, normals) = match transform {
                    Transform::Translate { offset } => (math::translation(*offset), math::IDENTITY),
                    Transform::Rotate { axis, degrees } => {
                        let axis = match axis {
                            Axis::X => 0,
                            Axis::Y => 1,
                            Axis::Z => 2,
                        };
                        let rotation = math::rotation(axis, degrees.to_radians());
                        (rotation, rotation)
                    }
                    Transform::Scale { scalar } => (
                        math::scale(*scalar),
                        math::scale(scalar.map(|factor| 1.0 / factor)),
                    ),
                };

                Model {
                    points: math::multiply(points, model.points),
                    normals: math::multiply(normals, model.normals),
                }
            },
        )
    }
}
