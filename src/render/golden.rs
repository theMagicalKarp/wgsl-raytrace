use super::Image;
use crate::config::Config;
use crate::scene::Scene;
use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

const SCENE: &str = "tests/golden/render.toml";
const REFERENCE: &str = "tests/golden/reference.png";

/// How far one channel may drift before its pixel is called an outlier rather
/// than noise, out of 255.
const OUTLIER: i32 = 24;

/// Mean absolute channel error the render may carry, out of 255.
///
/// So this sits an order of magnitude above the noise and a factor of two below
/// the weakest break worth catching. Re-measure before moving it.
const MAX_MAE: f32 = 0.25;

/// Fraction of pixels allowed past [`OUTLIER`].
const MAX_OUTLIERS: f32 = 0.01;

#[derive(Debug, Clone, Copy, PartialEq)]
struct Difference {
    mae: f32,
    outliers: f32,
}

/// Both frames are 8-bit RGBA of the same length. Alpha is skipped: `resolve`
/// writes 255 into it unconditionally, so it carries no signal.
fn compare(actual: &[u8], expected: &[u8]) -> Difference {
    let pixels = actual.len() / 4;
    let mut error = 0u64;
    let mut outliers = 0u32;

    for (actual, expected) in actual.chunks_exact(4).zip(expected.chunks_exact(4)) {
        let mut worst = 0;
        for channel in 0..3 {
            let difference = (actual[channel] as i32 - expected[channel] as i32).abs();
            error += difference as u64;
            worst = worst.max(difference);
        }

        if worst > OUTLIER {
            outliers += 1;
        }
    }

    Difference {
        mae: error as f32 / (pixels * 3) as f32,
        outliers: outliers as f32 / pixels as f32,
    }
}

/// Renders the golden scene, from the config on disk so the test exercises the
/// same path the binary does.
fn render() -> Image {
    let source = fs::read_to_string(SCENE).expect("the golden scene should be readable");
    let mut config: Config = toml::from_str(&source).expect("the golden scene should parse");
    config
        .validate(Path::new(SCENE).parent().unwrap())
        .expect("the golden scene's mesh should exist");

    let scene = Scene::load(&config).expect("the golden scene should load");
    super::render(&config, &scene)
        .expect(
            "the golden scene should render — set WGSL_RAYTRACE_SKIP_GPU_TESTS=1 \
             on a machine with no working adapter",
        )
        .image
}

/// Writes a frame somewhere a failing run can point at.
fn dump(name: &str, image: &Image) -> PathBuf {
    let path = env::temp_dir().join(name);
    image.save(&path).expect("the dump should be writable");
    path
}

/// Every channel's error as a frame of its own, amplified so a difference of
/// one or two is visible rather than a black square.
fn amplified(actual: &Image, expected: &[u8]) -> Image {
    let pixels = actual
        .pixels
        .chunks_exact(4)
        .zip(expected.chunks_exact(4))
        .flat_map(|(actual, expected)| {
            let channel = |i: usize| {
                let difference = (actual[i] as i32 - expected[i] as i32).abs();
                (difference * 8).min(255) as u8
            };
            [channel(0), channel(1), channel(2), 255]
        })
        .collect();

    Image {
        width: actual.width,
        height: actual.height,
        pixels,
    }
}

#[test]
fn the_render_matches_the_reference() {
    if env::var_os("WGSL_RAYTRACE_SKIP_GPU_TESTS").is_some() {
        return;
    }

    let image = render();

    if env::var_os("UPDATE_GOLDEN").is_some() {
        image
            .save(Path::new(REFERENCE))
            .expect("the reference should be writable");
        return;
    }

    let reference = image::open(REFERENCE)
        .expect("the reference should decode — regenerate it with UPDATE_GOLDEN=1")
        .to_rgba8();
    assert_eq!(
        (image.width, image.height),
        reference.dimensions(),
        "the scene's dimensions changed; regenerate the reference with UPDATE_GOLDEN=1",
    );

    let difference = compare(&image.pixels, &reference);
    if difference.mae <= MAX_MAE && difference.outliers <= MAX_OUTLIERS {
        return;
    }

    // Everything a run needs to judge whether the tolerance is wrong or the
    // shader is: the two measured numbers, and somewhere to look at the frames.
    let rendered = dump("wgsl-raytrace-golden-actual.png", &image);
    let diff = dump(
        "wgsl-raytrace-golden-diff.png",
        &amplified(&image, &reference),
    );

    panic!(
        "the render drifted from {REFERENCE}\n  \
           mae      {:.3} / {MAX_MAE} per channel\n  \
           outliers {:.4} / {MAX_OUTLIERS} of pixels past {OUTLIER}\n  \
           rendered {}\n  \
           diff     {} (amplified 8x)",
        difference.mae,
        difference.outliers,
        rendered.display(),
        diff.display(),
    );
}

#[test]
fn a_frame_matches_itself() {
    let frame = [12, 34, 56, 255, 200, 100, 0, 255];

    assert_eq!(
        compare(&frame, &frame),
        Difference {
            mae: 0.0,
            outliers: 0.0
        },
    );
}

#[test]
fn a_shift_too_small_to_flag_still_shows_in_the_mean() {
    let actual = [10, 10, 10, 255, 10, 10, 10, 255];
    let expected = [9, 9, 9, 255, 9, 9, 9, 255];

    let difference = compare(&actual, &expected);
    assert_eq!(difference.mae, 1.0);
    assert_eq!(difference.outliers, 0.0, "one out of 255 is not a break");
}

#[test]
fn one_wrecked_pixel_is_one_outlier() {
    // A frame the size of the golden one, with a single pixel blown out.
    let pixels = 32 * 32;
    let expected = vec![0u8; pixels * 4];
    let mut actual = expected.clone();
    actual[1] = 255;

    let difference = compare(&actual, &expected);
    assert_eq!(difference.outliers, 1.0 / pixels as f32);
    // One blown pixel in a thousand hardly moves the mean, which is why the
    // outlier budget is bounded separately rather than folded into it.
    assert!(difference.mae < MAX_MAE, "{difference:?}");
}

#[test]
fn alpha_is_not_compared() {
    // `resolve` writes 255 unconditionally, so a difference there would only
    // ever be noise from the encoder.
    let difference = compare(&[7, 7, 7, 0], &[7, 7, 7, 255]);

    assert_eq!(
        difference,
        Difference {
            mae: 0.0,
            outliers: 0.0
        }
    );
}
