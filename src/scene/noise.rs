//! The procedural patterns of `shader.wgsl`, written again in Rust.
//!
//! Only the tests read these: `scene::program::evaluate` is the reference the
//! GPU's evaluator is held against, and a noise it could not evaluate would be
//! the one op with nothing to be wrong against. Every function here is the
//! shader's of the same name, line for line, so a difference between the two is
//! a difference in the encoding rather than in the arithmetic.

/// See `pcg3d` in the shader.
fn pcg3d(seed: [u32; 3]) -> [u32; 3] {
    let mut v = seed.map(|c| c.wrapping_mul(1_664_525).wrapping_add(1_013_904_223));
    v[0] = v[0].wrapping_add(v[1].wrapping_mul(v[2]));
    v[1] = v[1].wrapping_add(v[2].wrapping_mul(v[0]));
    v[2] = v[2].wrapping_add(v[0].wrapping_mul(v[1]));
    v = v.map(|c| c ^ (c >> 16));
    v[0] = v[0].wrapping_add(v[1].wrapping_mul(v[2]));
    v[1] = v[1].wrapping_add(v[2].wrapping_mul(v[0]));
    v[2] = v[2].wrapping_add(v[0].wrapping_mul(v[1]));
    v
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn lattice_gradient(corner: [i32; 3], offset: [f32; 3]) -> f32 {
    let h = pcg3d(corner.map(|c| c as u32))[0] >> 28;
    let [x, y, z] = offset;
    let u = if h < 8 { x } else { y };
    let v = if h < 4 {
        y
    } else if h == 12 || h == 14 {
        x
    } else {
        z
    };
    (if h & 1 != 0 { -u } else { u }) + (if h & 2 != 0 { -v } else { v })
}

const PERLIN_SCALE: f32 = 0.982;

pub fn perlin(p: [f32; 3]) -> f32 {
    let floor = p.map(f32::floor);
    let cell = floor.map(|c| c as i32);
    let f = [p[0] - floor[0], p[1] - floor[1], p[2] - floor[2]];
    let u = f.map(fade);

    let corner = |dx: i32, dy: i32, dz: i32| {
        lattice_gradient(
            [cell[0] + dx, cell[1] + dy, cell[2] + dz],
            [f[0] - dx as f32, f[1] - dy as f32, f[2] - dz as f32],
        )
    };
    let near = lerp(
        lerp(corner(0, 0, 0), corner(1, 0, 0), u[0]),
        lerp(corner(0, 1, 0), corner(1, 1, 0), u[0]),
        u[1],
    );
    let far = lerp(
        lerp(corner(0, 0, 1), corner(1, 0, 1), u[0]),
        lerp(corner(0, 1, 1), corner(1, 1, 1), u[0]),
        u[1],
    );
    PERLIN_SCALE * lerp(near, far, u[2])
}

fn fbm(p: [f32; 3], detail: f32, roughness: f32, lacunarity: f32) -> f32 {
    let mut frequency = 1.0;
    let mut amplitude = 1.0;
    let mut total = 0.0;
    let mut sum = 0.0;
    let octaves = detail as u32;
    for _ in 0..=octaves {
        sum += perlin(p.map(|c| c * frequency)) * amplitude;
        total += amplitude;
        amplitude *= roughness;
        frequency *= lacunarity;
    }

    let coarse = 0.5 * sum / total + 0.5;
    let rest = detail - octaves as f32;
    if rest == 0.0 {
        return coarse;
    }
    let fine = sum + perlin(p.map(|c| c * frequency)) * amplitude;
    lerp(coarse, 0.5 * fine / (total + amplitude) + 0.5, rest)
}

const DISTORT: [[f32; 3]; 3] = [
    [17.13, 3.71, 91.37],
    [-43.7, 61.9, 7.3],
    [29.3, -83.1, 51.7],
];

pub fn noise(
    input: [f32; 3],
    scale: f32,
    detail: f32,
    roughness: f32,
    lacunarity: f32,
    distortion: f32,
) -> f32 {
    let mut p = input.map(|c| c * scale);
    if distortion != 0.0 {
        let shifted = DISTORT.map(|offset| {
            perlin([p[0] + offset[0], p[1] + offset[1], p[2] + offset[2]]) * distortion
        });
        p = [p[0] + shifted[0], p[1] + shifted[1], p[2] + shifted[2]];
    }
    fbm(p, detail, roughness, lacunarity).clamp(0.0, 1.0)
}

fn cell_random(cell: [i32; 3], salt: u32) -> [f32; 3] {
    pcg3d(cell.map(|c| c as u32 ^ salt)).map(|h| (h >> 8) as f32 * (1.0 / 16_777_216.0))
}

const CELL_COLOR_SALT: u32 = 0x9e37_79b9;

/// The distance to the nearest feature point and the colour of its cell.
pub fn voronoi(input: [f32; 3], scale: f32, randomness: f32) -> (f32, [f32; 3]) {
    let p = input.map(|c| c * scale);
    let floor = p.map(f32::floor);
    let cell = floor.map(|c| c as i32);
    let local = [p[0] - floor[0], p[1] - floor[1], p[2] - floor[2]];

    let mut nearest = f32::MAX;
    let mut owner = [0; 3];
    for z in -1..=1 {
        for y in -1..=1 {
            for x in -1..=1 {
                let offset = [x, y, z];
                let jitter = cell_random([cell[0] + x, cell[1] + y, cell[2] + z], 0);
                let d = [0, 1, 2].map(|c| offset[c] as f32 + jitter[c] * randomness - local[c]);
                let squared = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if squared < nearest {
                    nearest = squared;
                    owner = offset;
                }
            }
        }
    }

    let color = cell_random(
        [cell[0] + owner[0], cell[1] + owner[1], cell[2] + owner[2]],
        CELL_COLOR_SALT,
    );
    (nearest.sqrt(), color)
}

/// The distance to the nearest edge of the cell the point is in.
pub fn voronoi_edge(input: [f32; 3], scale: f32, randomness: f32) -> f32 {
    let p = input.map(|c| c * scale);
    let floor = p.map(f32::floor);
    let cell = floor.map(|c| c as i32);
    let local = [p[0] - floor[0], p[1] - floor[1], p[2] - floor[2]];
    let to_point = |x: i32, y: i32, z: i32| {
        let offset = [x, y, z];
        let jitter = cell_random([cell[0] + x, cell[1] + y, cell[2] + z], 0);
        [0, 1, 2].map(|c| offset[c] as f32 + jitter[c] * randomness - local[c])
    };
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

    let mut nearest = f32::MAX;
    let mut closest = [0.0; 3];
    for z in -1..=1 {
        for y in -1..=1 {
            for x in -1..=1 {
                let d = to_point(x, y, z);
                let squared = dot(d, d);
                if squared < nearest {
                    nearest = squared;
                    closest = d;
                }
            }
        }
    }

    let mut edge = f32::MAX;
    for z in -1..=1 {
        for y in -1..=1 {
            for x in -1..=1 {
                let d = to_point(x, y, z);
                let across = [0, 1, 2].map(|c| d[c] - closest[c]);
                let squared = dot(across, across);
                if squared > 1e-4 {
                    let middle = [0, 1, 2].map(|c| (closest[c] + d[c]) * 0.5);
                    edge = edge.min(dot(middle, across) / squared.sqrt());
                }
            }
        }
    }
    edge
}

pub fn ramp(t: f32, stops: &[(f32, [f32; 3])]) -> [f32; 3] {
    let mut color = stops[0].1;
    for pair in stops.windows(2) {
        let ((before, previous), (after, next)) = (pair[0], pair[1]);
        if t <= before {
            break;
        }
        if t >= after {
            color = next;
            continue;
        }
        let f = (t - before) / (after - before);
        color = [0, 1, 2].map(|c| lerp(previous[c], next[c], f));
        break;
    }
    color
}
