pub type Mat4 = [[f32; 4]; 4];

pub const IDENTITY: Mat4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

pub fn multiply(a: Mat4, b: Mat4) -> Mat4 {
    let mut out = [[0.0; 4]; 4];
    for (column, source) in out.iter_mut().zip(b) {
        for (row, slot) in column.iter_mut().enumerate() {
            *slot = (0..4).map(|k| a[k][row] * source[k]).sum();
        }
    }
    out
}

pub fn translation(offset: [f32; 3]) -> Mat4 {
    let mut m = IDENTITY;
    m[3] = [offset[0], offset[1], offset[2], 1.0];
    m
}

pub fn scale(scalar: [f32; 3]) -> Mat4 {
    let mut m = IDENTITY;
    for (axis, factor) in scalar.into_iter().enumerate() {
        m[axis][axis] = factor;
    }
    m
}

pub fn rotation(axis: usize, radians: f32) -> Mat4 {
    let (sin, cos) = radians.sin_cos();
    let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);

    let mut m = IDENTITY;
    m[u][u] = cos;
    m[v][u] = -sin;
    m[u][v] = sin;
    m[v][v] = cos;
    m
}

pub fn transform_point(m: Mat4, point: [f32; 3]) -> [f32; 3] {
    let mut out = [0.0; 3];
    for (row, slot) in out.iter_mut().enumerate() {
        *slot = m[0][row] * point[0] + m[1][row] * point[1] + m[2][row] * point[2] + m[3][row];
    }
    out
}

pub fn transform_direction(m: Mat4, direction: [f32; 3]) -> [f32; 3] {
    let mut out = [0.0; 3];
    for (row, slot) in out.iter_mut().enumerate() {
        *slot = m[0][row] * direction[0] + m[1][row] * direction[1] + m[2][row] * direction[2];
    }
    normalize(out)
}

pub fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn normalize(v: [f32; 3]) -> [f32; 3] {
    let length = dot(v, v).sqrt();
    match length > 0.0 {
        true => [v[0] / length, v[1] / length, v[2] / length],
        false => [0.0; 3],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn close(actual: [f32; 3], expected: [f32; 3]) {
        let error = sub(actual, expected)
            .map(f32::abs)
            .into_iter()
            .fold(0.0, f32::max);
        assert!(error < 1e-5, "expected {expected:?}, got {actual:?}");
    }

    #[test]
    fn multiply_applies_the_right_hand_side_first() {
        let scale_then_move = multiply(translation([10.0, 0.0, 0.0]), scale([2.0; 3]));
        let move_then_scale = multiply(scale([2.0; 3]), translation([10.0, 0.0, 0.0]));

        close(
            transform_point(scale_then_move, [1.0, 0.0, 0.0]),
            [12.0, 0.0, 0.0],
        );
        close(
            transform_point(move_then_scale, [1.0, 0.0, 0.0]),
            [22.0, 0.0, 0.0],
        );
    }

    #[test]
    fn each_rotation_turns_the_other_two_axes() {
        let quarter = std::f32::consts::FRAC_PI_2;

        close(
            transform_point(rotation(0, quarter), [0.0, 1.0, 0.0]),
            [0.0, 0.0, 1.0],
        );
        close(
            transform_point(rotation(1, quarter), [0.0, 0.0, 1.0]),
            [1.0, 0.0, 0.0],
        );
        close(
            transform_point(rotation(2, quarter), [1.0, 0.0, 0.0]),
            [0.0, 1.0, 0.0],
        );
    }

    #[test]
    fn rotation_leaves_its_own_axis_alone() {
        for axis in 0..3 {
            let mut point = [0.0; 3];
            point[axis] = 1.0;

            close(transform_point(rotation(axis, 0.7), point), point);
        }
    }

    #[test]
    fn directions_ignore_translation() {
        let m = translation([10.0, 20.0, 30.0]);

        close(transform_direction(m, [1.0, 0.0, 0.0]), [1.0, 0.0, 0.0]);
    }

    #[test]
    fn degenerate_directions_normalize_to_zero() {
        assert_eq!(normalize([0.0; 3]), [0.0; 3]);
        assert_eq!(transform_direction(IDENTITY, [0.0; 3]), [0.0; 3]);
    }
}
