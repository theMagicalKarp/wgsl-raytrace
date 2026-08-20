use bytemuck::Pod;
use bytemuck::Zeroable;

/// Candidate split planes considered per axis. More bins find slightly better
/// splits for a linearly higher build cost; 12 is the usual place to stop.
const BINS: usize = 12;

/// Subtrees at or below this size become leaves without evaluating a split. A
/// couple of triangle tests are cheaper than the node fetch and box tests needed
/// to avoid them.
const MAX_LEAF_SIZE: usize = 2;

/// Hard cap on tree depth. The shader traverses with a fixed-size stack, so the
/// builder — not the shader — is what has to guarantee the bound. Well-formed
/// input never comes close to this: a balanced tree over a million primitives is
/// twenty deep.
pub const MAX_DEPTH: u32 = 30;

/// An axis-aligned bounding box. An empty box is inverted, so joining it with
/// anything yields that thing.
#[derive(Copy, Clone, Debug)]
pub struct Aabb {
    min: [f32; 3],
    max: [f32; 3],
}

impl Aabb {
    pub const EMPTY: Self = Self {
        min: [f32::INFINITY; 3],
        max: [f32::NEG_INFINITY; 3],
    };

    /// The smallest box containing every given point.
    pub fn of_points(points: impl IntoIterator<Item = [f32; 3]>) -> Self {
        points.into_iter().fold(Self::EMPTY, |mut bounds, point| {
            for (axis, coordinate) in point.into_iter().enumerate() {
                bounds.min[axis] = bounds.min[axis].min(coordinate);
                bounds.max[axis] = bounds.max[axis].max(coordinate);
            }
            bounds
        })
    }

    fn join(self, other: Self) -> Self {
        let mut out = self;
        for axis in 0..3 {
            out.min[axis] = self.min[axis].min(other.min[axis]);
            out.max[axis] = self.max[axis].max(other.max[axis]);
        }
        out
    }

    fn centroid(self) -> [f32; 3] {
        let mut out = [0.0; 3];
        for (axis, slot) in out.iter_mut().enumerate() {
            *slot = (self.min[axis] + self.max[axis]) / 2.0;
        }
        out
    }

    /// Half the surface area, which is all the heuristic needs: every cost it
    /// compares carries the same factor of two.
    ///
    /// Clamping the extent keeps the empty box at zero rather than infinity.
    fn half_area(self) -> f32 {
        let extent = |axis: usize| (self.max[axis] - self.min[axis]).max(0.0);
        let (x, y, z) = (extent(0), extent(1), extent(2));
        x * y + y * z + z * x
    }
}

/// One node as the shader reads it.
///
/// Same packing trick as [`GpuTriangle`](crate::scene::GpuTriangle): a `vec3f`
/// is 12 bytes but 16-aligned in WGSL, so the scalar after each one costs
/// nothing. Children are always emitted as a pair, so storing the left index
/// locates both.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuBvhNode {
    pub min: [f32; 3],
    /// Interior: the index of the left child, whose sibling follows it.
    /// Leaf: the index of its first primitive in [`Bvh::order`].
    pub left_or_first: u32,
    pub max: [f32; 3],
    /// Zero marks an interior node, so leaves are never empty.
    pub primitive_count: u32,
}

const _: () = assert!(size_of::<GpuBvhNode>() == 32);

impl GpuBvhNode {
    fn new(bounds: Aabb, left_or_first: usize, primitive_count: usize) -> Self {
        Self {
            min: bounds.min,
            left_or_first: left_or_first as u32,
            max: bounds.max,
            primitive_count: primitive_count as u32,
        }
    }
}

pub struct Bvh {
    /// The tree, flattened. Node 0 is the root.
    pub nodes: Vec<GpuBvhNode>,
    /// The primitives, permuted so that each leaf's are contiguous. Callers are
    /// expected to reorder their own data to match, which is what lets a leaf
    /// name its primitives with an offset and a count.
    pub order: Vec<u32>,
    /// Depth of the deepest leaf, with the root at zero. Reported so a tree that
    /// degenerates into a list is obvious rather than merely slow.
    pub max_depth: u32,
}

/// The plane a subtree is split on, kept as the binning parameters that produced
/// it so partitioning can recompute bin indices without storing one per
/// primitive.
struct Split {
    axis: usize,
    /// Primitives in bins at or below this index go left.
    last_left_bin: usize,
    origin: f32,
    scale: f32,
}

/// Builds a tree over one box per primitive.
pub fn build(bounds: &[Aabb]) -> Bvh {
    let centroids: Vec<[f32; 3]> = bounds.iter().map(|b| b.centroid()).collect();
    let mut order: Vec<u32> = (0..bounds.len() as u32).collect();

    // Nodes are appended as they are discovered, but a node's own slot is
    // reserved by its parent before its contents are known — that reservation is
    // what makes the left child's index enough to find both children. The root
    // reserves itself.
    let mut nodes = vec![GpuBvhNode::default()];
    let mut max_depth = 0;

    let mut pending = vec![(0usize, 0..order.len(), 0u32)];
    while let Some((node, range, depth)) = pending.pop() {
        let node_bounds = order[range.clone()]
            .iter()
            .fold(Aabb::EMPTY, |acc, &p| acc.join(bounds[p as usize]));

        let split = (range.len() > MAX_LEAF_SIZE && depth < MAX_DEPTH)
            .then(|| find_split(bounds, &centroids, &order[range.clone()], node_bounds))
            .flatten();

        let Some(split) = split else {
            nodes[node] = GpuBvhNode::new(node_bounds, range.start, range.len());
            max_depth = max_depth.max(depth);
            continue;
        };

        let mid = range.start + partition(&centroids, &mut order[range.clone()], &split);
        debug_assert!(range.start < mid && mid < range.end);

        let left = nodes.len();
        nodes.extend([GpuBvhNode::default(); 2]);
        nodes[node] = GpuBvhNode::new(node_bounds, left, 0);
        pending.push((left, range.start..mid, depth + 1));
        pending.push((left + 1, mid..range.end, depth + 1));
    }

    Bvh {
        nodes,
        order,
        max_depth,
    }
}

/// Which bin a coordinate falls in. A negative offset saturates to zero on the
/// cast, and the maximum coordinate would otherwise land one past the end.
fn bin_of(coordinate: f32, origin: f32, scale: f32) -> usize {
    (((coordinate - origin) * scale) as usize).min(BINS - 1)
}

/// Picks the cheapest split across all three axes, or `None` if leaving the
/// subtree as a leaf is cheaper than every candidate.
fn find_split(
    bounds: &[Aabb],
    centroids: &[[f32; 3]],
    indices: &[u32],
    node_bounds: Aabb,
) -> Option<Split> {
    // Binning spans the centroids rather than the node, so the bins stay
    // populated even when the primitives themselves are much larger than their
    // spread.
    let centroid_bounds = Aabb::of_points(indices.iter().map(|&p| centroids[p as usize]));

    let mut best: Option<(f32, Split)> = None;
    for axis in [0, 1, 2] {
        let origin = centroid_bounds.min[axis];
        let extent = centroid_bounds.max[axis] - origin;
        // Every centroid shares this coordinate, so no plane on this axis
        // separates anything.
        if extent < f32::EPSILON {
            continue;
        }
        let scale = BINS as f32 / extent;

        let mut bin_bounds = [Aabb::EMPTY; BINS];
        let mut bin_counts = [0usize; BINS];
        for &p in indices {
            let bin = bin_of(centroids[p as usize][axis], origin, scale);
            bin_bounds[bin] = bin_bounds[bin].join(bounds[p as usize]);
            bin_counts[bin] += 1;
        }

        // Sweep left to right accumulating what each plane would put on its
        // left, then right to left to pair it with the other side. Two linear
        // passes price all BINS - 1 planes.
        let mut left_area = [0.0; BINS - 1];
        let mut left_count = [0usize; BINS - 1];
        let mut accumulated = Aabb::EMPTY;
        let mut counted = 0;
        for bin in 0..BINS - 1 {
            accumulated = accumulated.join(bin_bounds[bin]);
            counted += bin_counts[bin];
            left_area[bin] = accumulated.half_area();
            left_count[bin] = counted;
        }

        let mut accumulated = Aabb::EMPTY;
        let mut counted = 0;
        for bin in (0..BINS - 1).rev() {
            accumulated = accumulated.join(bin_bounds[bin + 1]);
            counted += bin_counts[bin + 1];
            // A plane with an empty side splits nothing and would loop forever.
            if left_count[bin] == 0 || counted == 0 {
                continue;
            }
            let cost =
                left_area[bin] * left_count[bin] as f32 + accumulated.half_area() * counted as f32;
            if best.as_ref().is_none_or(|&(best_cost, _)| cost < best_cost) {
                best = Some((
                    cost,
                    Split {
                        axis,
                        last_left_bin: bin,
                        origin,
                        scale,
                    },
                ));
            }
        }
    }

    // The textbook comparison is `traversal + (A_l*N_l + A_r*N_r) / A_node`
    // against `N`, scaled through by `A_node` to keep it in the same units as
    // the costs above and to stay well-defined for a flat node.
    let (cost, split) = best?;
    let parent_area = node_bounds.half_area();
    let traversal_cost = parent_area;
    let leaf_cost = indices.len() as f32 * parent_area;
    (traversal_cost + cost < leaf_cost).then_some(split)
}

/// Shuffles `indices` so everything left of the split comes first, returning
/// where the right side starts. Only splits with both sides populated reach
/// here, so the result is always strictly inside the slice.
fn partition(centroids: &[[f32; 3]], indices: &mut [u32], split: &Split) -> usize {
    let mut boundary = 0;
    for i in 0..indices.len() {
        let coordinate = centroids[indices[i] as usize][split.axis];
        if bin_of(coordinate, split.origin, split.scale) <= split.last_left_bin {
            indices.swap(boundary, i);
            boundary += 1;
        }
    }
    boundary
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math;

    /// The shader's xorshift32, so the test scenes are deterministic without
    /// pulling in a dependency.
    struct Rng(u32);

    impl Rng {
        fn next_f32(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            f32::from_bits(0x3f80_0000 | (self.0 >> 9)) - 1.0
        }

        /// A coordinate in [-range, range].
        fn coordinate(&mut self, range: f32) -> f32 {
            (self.next_f32() * 2.0 - 1.0) * range
        }
    }

    type Triangle = [[f32; 3]; 3];

    /// Möller-Trumbore, matching `intersect_triangle` in the shader.
    fn intersect_triangle(origin: [f32; 3], direction: [f32; 3], tri: &Triangle) -> Option<f32> {
        let e1 = math::sub(tri[1], tri[0]);
        let e2 = math::sub(tri[2], tri[0]);
        let p = math::cross(direction, e2);
        let det = math::dot(e1, p);
        if det.abs() < 1e-8 {
            return None;
        }
        let inv_det = 1.0 / det;
        let s = math::sub(origin, tri[0]);
        let u = math::dot(s, p) * inv_det;
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let q = math::cross(s, e1);
        let v = math::dot(direction, q) * inv_det;
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let t = math::dot(e2, q) * inv_det;
        (t > 1e-3).then_some(t)
    }

    /// The slab test, matching `hit_aabb` in the shader.
    fn hit_aabb(origin: [f32; 3], inv_direction: [f32; 3], node: &GpuBvhNode, closest: f32) -> f32 {
        let mut enter = 1e-3f32;
        let mut exit = closest;
        for axis in 0..3 {
            let t0 = (node.min[axis] - origin[axis]) * inv_direction[axis];
            let t1 = (node.max[axis] - origin[axis]) * inv_direction[axis];
            enter = enter.max(t0.min(t1));
            exit = exit.min(t0.max(t1));
        }
        if enter <= exit { enter } else { f32::MAX }
    }

    /// The traversal the shader performs, kept structurally identical so a bug
    /// in the tree shows up here.
    fn traverse(bvh: &Bvh, triangles: &[Triangle], origin: [f32; 3], direction: [f32; 3]) -> f32 {
        let inv_direction = [1.0 / direction[0], 1.0 / direction[1], 1.0 / direction[2]];
        let mut closest = f32::MAX;
        let mut stack = Vec::new();
        let mut node = 0usize;
        loop {
            let current = &bvh.nodes[node];
            if current.primitive_count > 0 {
                for i in 0..current.primitive_count as usize {
                    let primitive = bvh.order[current.left_or_first as usize + i] as usize;
                    if let Some(t) = intersect_triangle(origin, direction, &triangles[primitive])
                        && t < closest
                    {
                        closest = t;
                    }
                }
            } else {
                let left = current.left_or_first as usize;
                let mut near = left;
                let mut far = left + 1;
                let mut near_t = hit_aabb(origin, inv_direction, &bvh.nodes[near], closest);
                let mut far_t = hit_aabb(origin, inv_direction, &bvh.nodes[far], closest);
                if far_t < near_t {
                    std::mem::swap(&mut near, &mut far);
                    std::mem::swap(&mut near_t, &mut far_t);
                }
                if near_t < f32::MAX {
                    if far_t < f32::MAX {
                        stack.push(far);
                    }
                    node = near;
                    continue;
                }
            }
            match stack.pop() {
                Some(next) => node = next,
                None => return closest,
            }
        }
    }

    fn brute_force(triangles: &[Triangle], origin: [f32; 3], direction: [f32; 3]) -> f32 {
        triangles
            .iter()
            .filter_map(|tri| intersect_triangle(origin, direction, tri))
            .fold(f32::MAX, f32::min)
    }

    /// Small triangles scattered through a cube, plus two huge ground-plane
    /// triangles — the mix that makes a midpoint split misbehave, and the same
    /// shape as the example scene.
    fn scene(rng: &mut Rng, count: usize) -> Vec<Triangle> {
        let mut triangles: Vec<Triangle> = (0..count)
            .map(|_| {
                let base = [
                    rng.coordinate(1.0),
                    rng.coordinate(1.0),
                    rng.coordinate(1.0),
                ];
                std::array::from_fn(|_| {
                    std::array::from_fn(|axis| base[axis] + rng.coordinate(0.1))
                })
            })
            .collect();
        triangles.push([
            [-40.0, -1.0, -40.0],
            [40.0, -1.0, -40.0],
            [40.0, -1.0, 40.0],
        ]);
        triangles.push([
            [-40.0, -1.0, -40.0],
            [40.0, -1.0, 40.0],
            [-40.0, -1.0, 40.0],
        ]);
        triangles
    }

    fn bounds_of(triangles: &[Triangle]) -> Vec<Aabb> {
        triangles
            .iter()
            .map(|tri| Aabb::of_points(tri.iter().copied()))
            .collect()
    }

    #[test]
    fn traversal_agrees_with_brute_force() {
        let mut rng = Rng(0x9e37_79b9);
        let triangles = scene(&mut rng, 600);
        let bvh = build(&bounds_of(&triangles));

        let mut tested_hits = 0;
        for _ in 0..2000 {
            let origin = [
                rng.coordinate(3.0),
                rng.coordinate(3.0),
                rng.coordinate(3.0),
            ];
            let direction = math::normalize([
                rng.coordinate(1.0),
                rng.coordinate(1.0),
                rng.coordinate(1.0),
            ]);
            let expected = brute_force(&triangles, origin, direction);
            let actual = traverse(&bvh, &triangles, origin, direction);
            assert_eq!(
                expected, actual,
                "ray from {origin:?} toward {direction:?} disagreed"
            );
            if expected < f32::MAX {
                tested_hits += 1;
            }
        }
        // A tree that returned "miss" for everything would otherwise pass.
        assert!(tested_hits > 500, "only {tested_hits} rays hit anything");
    }

    #[test]
    fn axis_aligned_rays_agree_with_brute_force() {
        // Rays parallel to an axis make an inverse direction infinite, and rays
        // grazing a slab plane make it a NaN. Both go down paths the random
        // directions above almost never reach.
        let mut rng = Rng(0x1234_5678);
        let triangles = scene(&mut rng, 300);
        let bvh = build(&bounds_of(&triangles));

        for axis in 0..3 {
            for sign in [-1.0, 1.0] {
                for _ in 0..200 {
                    let mut origin = [rng.coordinate(2.0); 3];
                    origin[axis] = sign * -3.0;
                    // Deliberately place some origins exactly on a node boundary.
                    origin[(axis + 1) % 3] = -1.0;
                    let mut direction = [0.0; 3];
                    direction[axis] = sign;

                    assert_eq!(
                        brute_force(&triangles, origin, direction),
                        traverse(&bvh, &triangles, origin, direction),
                        "axis-aligned ray from {origin:?} toward {direction:?} disagreed"
                    );
                }
            }
        }
    }

    #[test]
    fn every_primitive_lands_in_exactly_one_leaf() {
        let mut rng = Rng(0xdead_beef);
        let triangles = scene(&mut rng, 500);
        let bvh = build(&bounds_of(&triangles));

        let mut seen = vec![0u32; triangles.len()];
        for node in &bvh.nodes {
            if node.primitive_count == 0 {
                continue;
            }
            for i in 0..node.primitive_count as usize {
                seen[bvh.order[node.left_or_first as usize + i] as usize] += 1;
            }
        }
        assert!(
            seen.iter().all(|&count| count == 1),
            "primitives covered {:?} times",
            seen.iter().collect::<std::collections::BTreeSet<_>>()
        );
    }

    #[test]
    fn depth_stays_within_the_shader_stack() {
        let mut rng = Rng(0x0bad_f00d);
        let triangles = scene(&mut rng, 2000);
        let bvh = build(&bounds_of(&triangles));
        assert!(bvh.max_depth <= MAX_DEPTH, "depth {}", bvh.max_depth);
        // Sanity: 2002 primitives in leaves of at most 2 cannot be a shallow tree.
        assert!(bvh.max_depth >= 10, "depth {}", bvh.max_depth);
    }

    #[test]
    fn degenerate_input_still_builds() {
        // Every centroid identical, so no axis has a usable split plane.
        let triangle: Triangle = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let triangles = vec![triangle; 64];
        let bvh = build(&bounds_of(&triangles));

        let leaves: usize = bvh.nodes.iter().map(|n| n.primitive_count as usize).sum();
        assert_eq!(leaves, 64);
        assert_eq!(bvh.max_depth, 0, "should collapse to a single leaf");
    }

    #[test]
    fn single_primitive_builds_a_lone_leaf() {
        let triangles = vec![[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]];
        let bvh = build(&bounds_of(&triangles));
        assert_eq!(bvh.nodes.len(), 1);
        assert_eq!(bvh.nodes[0].primitive_count, 1);
        assert_eq!(bvh.order, vec![0]);
    }
}
