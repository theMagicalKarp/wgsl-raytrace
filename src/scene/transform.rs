use crate::config::Axis;
use crate::config::Transform;
use crate::math;
use crate::math::Mat4;

/// The model transform, the matrix that carries its normals, and the way back.
pub(super) struct Model {
    pub(super) points: Mat4,
    pub(super) normals: Mat4,
    /// World space back to the space the `.obj` was authored in.
    ///
    /// Everything is baked into world space at load, so an object's own
    /// coordinates are gone by the time the shader sees a triangle — and a
    /// pattern written in them has to survive the object being moved, or it
    /// swims across the surface whenever the scene is nudged. This is what
    /// brings a hit back.
    ///
    /// Built by inverting each step and composing them the other way round
    /// rather than by inverting the product: the inverse of a translate, a
    /// rotate or a scale is another one of the same, exactly, and a general
    /// 4x4 inversion of a matrix built from those would only be a less accurate
    /// way of arriving at the same place.
    pub(super) inverse: Mat4,
}

impl Model {
    pub(super) fn new(transforms: &[Transform]) -> Model {
        transforms.iter().fold(
            Model {
                points: math::IDENTITY,
                normals: math::IDENTITY,
                inverse: math::IDENTITY,
            },
            |model, transform| {
                let reciprocal = |scalar: &[f32; 3]| scalar.map(|factor| 1.0 / factor);
                let (points, normals, inverse) = match transform {
                    Transform::Translate { offset } => (
                        math::translation(*offset),
                        math::IDENTITY,
                        math::translation(offset.map(|axis| -axis)),
                    ),
                    Transform::Rotate { axis, degrees } => {
                        let axis = match axis {
                            Axis::X => 0,
                            Axis::Y => 1,
                            Axis::Z => 2,
                        };
                        let rotation = math::rotation(axis, degrees.to_radians());
                        (
                            rotation,
                            rotation,
                            math::rotation(axis, -degrees.to_radians()),
                        )
                    }
                    Transform::Scale { scalar } => (
                        math::scale(*scalar),
                        math::scale(reciprocal(scalar)),
                        math::scale(reciprocal(scalar)),
                    ),
                };

                Model {
                    points: math::multiply(points, model.points),
                    normals: math::multiply(normals, model.normals),
                    // The steps apply left to right, so their inverses undo
                    // right to left: this one comes off first and therefore
                    // multiplies on the right of everything already collected.
                    inverse: math::multiply(model.inverse, inverse),
                }
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Axis;

    fn close(actual: [f32; 3], expected: [f32; 3]) {
        let error = math::sub(actual, expected)
            .map(f32::abs)
            .into_iter()
            .fold(0.0, f32::max);
        assert!(error < 1e-5, "expected {expected:?}, got {actual:?}");
    }

    /// The whole point of the inverse: a point taken into world space and
    /// brought back is the point the `.obj` listed, whatever the object has
    /// been through on the way.
    #[test]
    fn the_inverse_undoes_the_transform() {
        let model = Model::new(&[
            Transform::Scale {
                scalar: [2.0, 3.0, 0.5],
            },
            Transform::Rotate {
                axis: Axis::Y,
                degrees: 37.0,
            },
            Transform::Translate {
                offset: [4.0, -1.0, 9.0],
            },
        ]);

        for point in [[0.0; 3], [1.0, 2.0, 3.0], [-5.0, 0.25, 7.5]] {
            let world = math::transform_point(model.points, point);
            close(math::transform_point(model.inverse, world), point);
        }
    }

    #[test]
    fn an_untransformed_object_is_its_own_inverse() {
        assert_eq!(Model::new(&[]).inverse, math::IDENTITY);
    }
}
