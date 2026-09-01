use crate::scene::Environment;

/// The largest distribution built over a map, per side. A cell is a cone of
/// directions drawn uniformly within it, so 1024x512 is the finest the sampler
/// can aim: about a third of a degree, which resolves a half-degree sun disc
/// into a handful of cells.
///
/// The cap exists because the conditional is a float per cell — over an 8K map
/// that would be 134MB of buffer to sharpen a sampler already aiming inside the
/// sun. Coarser costs variance and never correctness: the density is piecewise
/// constant either way, and the shader divides by the one it drew from.
const MAX_WIDTH: u32 = 1024;
const MAX_HEIGHT: u32 = 512;

/// A 2D distribution over an equirectangular map, as the shader searches it: a
/// marginal over rows, and one conditional over columns per row.
///
/// Cumulative rather than densities, because drawing from one is a binary
/// search over exactly this layout. A single cell's density — which the shader
/// needs to price a direction for MIS — is the gap between neighbouring
/// entries, so it is not stored twice.
pub struct Sky {
    pub width: u32,
    pub height: u32,
    /// Row-major, `width` entries per row. Each row ascends to exactly one.
    pub conditional: Vec<f32>,
    /// `height` entries, ascending to exactly one.
    pub marginal: Vec<f32>,
}

/// Its shape and nothing else: a megapixel of cumulative probabilities in an
/// error message helps nobody.
impl std::fmt::Debug for Sky {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sky({}x{})", self.width, self.height)
    }
}

/// Rec. 709 luma, matching `scene/light.rs` and the shader's `luminance`. All
/// three have to agree: one decides how often a direction is drawn, the others
/// divide by how often it was.
fn luminance(texel: &[f32; 4]) -> f32 {
    0.2126 * texel[0] + 0.7152 * texel[1] + 0.0722 * texel[2]
}

/// How many map texels go into one cell of the distribution, per side. One
/// leaves the map alone, which is what every HDRI at 2K or below gets.
fn stride(map: &Environment) -> u32 {
    map.width
        .div_ceil(MAX_WIDTH)
        .max(map.height.div_ceil(MAX_HEIGHT))
        .max(1)
}

/// What each cell is worth: the mean radiance of the texels it covers, times how
/// much sky one of those texels stands for.
///
/// The `sin(theta)` is why a row at the pole does not dominate. It has as many
/// texels as the equator and covers almost no sky, so weighing by radiance alone
/// would send a third of the samples into a pinprick; folding the Jacobian in
/// here makes the solid-angle density proportional to radiance instead.
///
/// Applied per map row, not per cell row: a cell can span 64 rows of an 8K map,
/// where `sin(theta)` changes eightfold, and averaging the product is right
/// where multiplying the averages is not.
fn weights(map: &Environment, stride: u32, width: u32, height: u32) -> Vec<f32> {
    let mut sums = vec![0.0f32; (width * height) as usize];
    let mut counts = vec![0u32; (width * height) as usize];

    for y in 0..map.height {
        // Row centres from the +Y pole down, the latitude
        // `environment_radiance` reads the map with.
        let theta = std::f32::consts::PI * (y as f32 + 0.5) / map.height as f32;
        let sin_theta = theta.sin();
        let row = (y / stride) * width;

        for x in 0..map.width {
            let texel = &map.texels[(y * map.width + x) as usize];
            let cell = (row + x / stride) as usize;
            sums[cell] += luminance(texel) * sin_theta;
            counts[cell] += 1;
        }
    }

    // A mean and not a sum: when the stride does not divide the map the last
    // cell of a row or column covers fewer texels, and a sum would make it less
    // likely to be drawn for no reason but where it sits.
    let means: Vec<f32> = sums
        .iter()
        .zip(&counts)
        .map(|(&sum, &count)| match count {
            0 => 0.0,
            _ => sum / count as f32,
        })
        .collect();

    dilate(&means, width, height)
}

/// Raises every cell to the largest of it and its eight neighbours.
///
/// The one place the distribution stops being proportional to the map on
/// purpose. The shader reads the environment through a **linear** sampler, so a
/// direction returns the blend of the four texels around it, not the one the
/// distribution weighed. A cell one texel out from a 300-unit sun is weighed as
/// dim sky and still returns a good fraction of the sun — a large radiance over
/// a tiny density, which is a firefly. Leaving this out cost 2.5x in noise on
/// the golden sun scene, more than everything else in this file buys.
///
/// A blend reaches at most one texel out, so one cell in each direction whatever
/// the stride, and the neighbourhood maximum is an upper bound on what the
/// sampler can return inside the cell. An upper bound is still strictly positive
/// wherever the map is, which is all an importance density has to be; it costs a
/// little efficiency near a bright spot and bounds the ratio being divided.
///
/// Edges wrap in longitude and clamp in latitude, matching the sampler's address
/// modes.
fn dilate(cells: &[f32], width: u32, height: u32) -> Vec<f32> {
    let at = |x: u32, y: u32| cells[(y * width + x) as usize];

    (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .map(|(x, y)| {
            let mut peak = 0.0f32;
            for dy in -1i32..=1 {
                let row = (y as i32 + dy).clamp(0, height as i32 - 1) as u32;
                for dx in -1i32..=1 {
                    let column = (x as i32 + dx).rem_euclid(width as i32) as u32;
                    peak = peak.max(at(column, row));
                }
            }
            peak
        })
        .collect()
}

/// Turns `values` into a cumulative distribution in place, returning the total.
///
/// The last entry is pinned rather than divided: the running sum lands a
/// rounding error either side of the total, and the shader's binary search needs
/// somewhere for a draw of 0.999… to go. `light.rs` gives its table the same
/// guarantee.
///
/// A slice summing to nothing gets a uniform distribution rather than a division
/// by zero. The marginal never draws such a row, so its contents cannot be
/// observed; filling it only keeps "every conditional ends at one" universal.
fn cumulate(values: &mut [f32]) -> f32 {
    let total: f32 = values.iter().sum();
    if total <= 0.0 {
        let count = values.len() as f32;
        for (index, value) in values.iter_mut().enumerate() {
            *value = (index + 1) as f32 / count;
        }
        return 0.0;
    }

    let mut running = 0.0;
    for value in values.iter_mut() {
        running += *value;
        *value = running / total;
    }
    if let Some(last) = values.last_mut() {
        *last = 1.0;
    }

    total
}

/// Builds the distribution the shader aims sky samples with.
///
/// `None` for a map that emits nothing, which tells the shader to skip sky
/// sampling and let escaped rays find the background on their own. Not an error:
/// a black map is a black background, and renders correctly unsampled.
pub(super) fn build(map: &Environment) -> Option<Sky> {
    let stride = stride(map);
    let width = map.width.div_ceil(stride);
    let height = map.height.div_ceil(stride);
    if width == 0 || height == 0 {
        return None;
    }

    let mut conditional = weights(map, stride, width, height);

    // Each row's total becomes its weight in the marginal: a row is drawn as
    // often as it is bright, and a column within it likewise, which multiplies
    // out to a cell drawn in proportion to its share of the sky's light.
    let mut marginal: Vec<f32> = conditional
        .chunks_mut(width as usize)
        .map(cumulate)
        .collect();

    match cumulate(&mut marginal) > 0.0 {
        true => Some(Sky {
            width,
            height,
            conditional,
            marginal,
        }),
        false => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A map of the given size, with `bright` texels set to `value` and the rest
    /// to `floor`. Coordinates are (x, y) with y counted from the +Y pole.
    fn map(width: u32, height: u32, floor: f32, bright: &[(u32, u32, f32)]) -> Environment {
        let mut texels = vec![[floor, floor, floor, 1.0]; (width * height) as usize];
        for &(x, y, value) in bright {
            texels[(y * width + x) as usize] = [value, value, value, 1.0];
        }
        Environment {
            width,
            height,
            texels,
        }
    }

    /// The chance of cell `(x, y)`, recovered the way the shader recovers it:
    /// neighbouring cumulative entries differenced, then multiplied.
    fn chance(sky: &Sky, x: u32, y: u32) -> f32 {
        let delta = |values: &[f32], index: u32| match index {
            0 => values[0],
            _ => values[index as usize] - values[index as usize - 1],
        };
        let row = delta(&sky.marginal, y);
        let column = delta(
            &sky.conditional[(y * sky.width) as usize..][..sky.width as usize],
            x,
        );

        row * column
    }

    #[test]
    fn the_chances_of_every_cell_add_to_one() {
        // What the shader's binary search depends on, and what makes dividing
        // by the density an estimator rather than a guess.
        let sky = build(&map(8, 4, 0.2, &[(3, 1, 40.0), (7, 3, 5.0)])).unwrap();

        let total: f32 = (0..sky.height)
            .flat_map(|y| (0..sky.width).map(move |x| (x, y)))
            .map(|(x, y)| chance(&sky, x, y))
            .sum();

        assert!((total - 1.0).abs() < 1e-5, "{total}");
        assert_eq!(*sky.marginal.last().unwrap(), 1.0);
        assert!(sky.marginal.windows(2).all(|pair| pair[0] <= pair[1]));
        for row in sky.conditional.chunks(sky.width as usize) {
            assert_eq!(*row.last().unwrap(), 1.0);
            assert!(row.windows(2).all(|pair| pair[0] <= pair[1]));
        }
    }

    #[test]
    fn a_sun_takes_the_share_of_the_draws_it_takes_of_the_light() {
        // The point of the whole file. One texel in 64 at 100x the radiance of
        // the rest holds most of the light, and should be drawn most of the time
        // rather than one time in 64. Measured over the sun and the ring
        // `dilate` spread it into, since the sampler returns sun light there too.
        let sky = build(&map(8, 8, 1.0, &[(2, 4, 100.0)])).unwrap();

        let sun: f32 = (1..=3)
            .flat_map(|x| (3..=5).map(move |y| (x, y)))
            .map(|(x, y)| chance(&sky, x, y))
            .sum();

        assert!(sun > 0.5, "the sun should dominate the draws, got {sun}");
        // And an unstratified sampler would have given those nine cells this.
        assert!(sun > 9.0 / 64.0);
    }

    #[test]
    fn a_cell_beside_a_bright_one_is_drawn_as_though_it_were_bright() {
        // What `dilate` is for. A direction landing in the dark cell next door
        // still comes back carrying a quarter of the sun through the linear
        // sampler, and dividing that by the density of dark sky is a firefly.
        let sky = build(&map(8, 8, 1.0, &[(4, 4, 500.0)])).unwrap();

        let neighbour = chance(&sky, 5, 4);
        let sun = chance(&sky, 4, 4);
        assert!(
            (neighbour - sun).abs() < 1e-6,
            "the neighbour should be weighed with the sun: {neighbour} against {sun}",
        );

        // Two cells out is past what the filter can reach, and stays dark.
        assert!(chance(&sky, 6, 4) < sun / 100.0);
    }

    #[test]
    fn a_row_at_the_pole_is_weighed_down_by_the_sky_it_covers() {
        // Equal radiance everywhere. The top row has as many texels as the
        // middle one and covers almost none of the sky, so weighing by radiance
        // alone would aim the sampler at the pole. Tall enough that `dilate`
        // lifting the pole row to its neighbour is a nudge rather than the
        // answer, as it would be over 16 rows.
        let sky = build(&map(16, 64, 1.0, &[])).unwrap();

        let pole: f32 = (0..sky.width).map(|x| chance(&sky, x, 0)).sum();
        let equator: f32 = (0..sky.width).map(|x| chance(&sky, x, 32)).sum();

        assert!(
            pole < equator * 0.2,
            "the pole should be far less likely than the equator, got {pole} against {equator}",
        );

        // Nothing distinguishes one column from another, so the conditional
        // should be flat — and stay flat through the dilation, which wraps in
        // longitude rather than clamping.
        let first = chance(&sky, 0, 32);
        for x in 0..sky.width {
            assert!((chance(&sky, x, 32) - first).abs() < 1e-6);
        }
    }

    #[test]
    fn a_black_map_has_nothing_to_aim_at() {
        // Not an error: a black background renders correctly unsampled, and
        // `None` is what tells the shader to skip it.
        assert!(build(&map(4, 4, 0.0, &[])).is_none());
    }

    #[test]
    fn a_large_map_is_capped_and_keeps_every_bright_texel_reachable() {
        // A cell is a box mean of non-negative values, so it is zero only when
        // every texel in it is. That is why the cap costs variance and not
        // correctness: the coarse distribution is nonzero everywhere the map is.
        let mut bright = map(4096, 2048, 0.0, &[(4000, 1000, 12.0)]);
        bright.texels[(1024 * 4096 + 8) as usize] = [0.0, 3.0, 0.0, 1.0];

        let sky = build(&bright).unwrap();

        assert_eq!((sky.width, sky.height), (1024, 512));
        assert!(chance(&sky, 1000, 250) > 0.0, "the sun should be reachable");
        assert!(chance(&sky, 2, 256) > 0.0, "and so should the other texel");
        assert!(chance(&sky, 500, 100) == 0.0, "black stays black");
    }

    #[test]
    fn a_map_that_the_stride_does_not_divide_still_covers_every_texel() {
        // 1501 is not a multiple of the stride of 2 it forces, so the last cell
        // of each row covers one texel where a full one covers two. Averaging
        // rather than summing is what keeps it as likely as its neighbours.
        let sky = build(&map(1501, 751, 1.0, &[])).unwrap();

        assert_eq!((sky.width, sky.height), (751, 376));

        let row = 188;
        let full = chance(&sky, 0, row);
        let partial = chance(&sky, sky.width - 1, row);
        assert!(
            (full - partial).abs() < 1e-6,
            "the ragged edge should not be less likely: {full} against {partial}",
        );
    }
}
