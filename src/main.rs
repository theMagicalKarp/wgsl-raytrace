#![forbid(unsafe_code)]

mod config;
mod math;
mod render;
mod scene;

use clap::Parser;
use colored::Colorize;
use config::Args;
use config::Config;
use config::format_error;
use scene::Scene;
use std::error::Error;
use std::fs;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{}{} {}", "error".bold().red(), ":".bold(), error);
            ExitCode::FAILURE
        }
    }
}

/// What the unfiltered frame is called inside `--debug`'s directory.
const RAW: &str = "raw.png";

fn run() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let config_content = fs::read_to_string(&args.config)?;
    let mut config: Config =
        toml::from_str(&config_content).map_err(|error| -> Box<dyn Error> {
            format_error(&config_content, &args.config, &error).into()
        })?;

    if let Some(samples) = args.samples {
        config.camera.samples = samples;
    }
    if let Some(k) = args.outlier_k {
        config.outliers.k = k;
    }
    // Not a scene setting at all — a second render of the same scene, drawn from
    // a different stream so its noise does not cancel against the first one's.
    config.seed = args.seed;

    // Object paths are written relative to the config, not the shell.
    let config_dir = args.config.parent().unwrap_or(Path::new("."));
    config.validate(config_dir)?;

    println!("{}", config);

    let scene = Scene::load(&config)?;
    println!(
        "{}{} {} triangles across {} materials, indexed by {} bvh nodes {} deep",
        "scene".bold().green(),
        ":".bold(),
        scene.triangles.len(),
        scene.materials.len(),
        scene.nodes.len(),
        scene.depth,
    );

    let started = Instant::now();
    let outputs = render::Outputs {
        aovs: args.debug.is_some(),
        denoise: args.denoise,
        variance: args.variance.is_some(),
    };
    let render = match args.preview {
        true => render::preview(&config, &scene, outputs)?,
        false => render::render(&config, &scene, outputs)?,
    };

    // A filtered frame is the render rather than a variant of it, so it is what
    // `--output` gets. The unfiltered frame it was built from is still the only
    // thing the sigmas can be judged against, which is what `--debug` is for.
    let image = match &render.denoised {
        Some(denoised) => &denoised.image,
        None => &render.image,
    };
    image.save(&args.output)?;

    println!(
        "{}{} {}x{} written to {} in {:.1}s on {}",
        "render".bold().green(),
        ":".bold(),
        image.width,
        image.height,
        args.output.display(),
        started.elapsed().as_secs_f32(),
        render.renderer,
    );

    // The a-trous filter, which is a fixed cost after the sample loop rather
    // than part of it — so it is quoted as wall time rather than folded into the
    // `gpu:` line's per-dispatch figure.
    if let Some(denoised) = &render.denoised {
        let unfiltered = match &args.debug {
            Some(directory) => format!("compare against {}", directory.join(RAW).display()),
            None => "--debug <dir> keeps the unfiltered frame".to_string(),
        };
        println!(
            "{}{} {} in {:.2}s · {}",
            "denoise".bold().green(),
            ":".bold(),
            config.denoise,
            denoised.elapsed.as_secs_f32(),
            unfiltered.dimmed(),
        );
    }

    // What the frame's own samples say about how converged it is, out of the
    // moments the shader accumulated beside the radiance. The standard error is
    // the figure a denoising filter is judged by; the mean luminance is the one
    // that says whether a change that was not supposed to move the exposure
    // moved it, and it is quoted in linear units because a gamma-encoded frame
    // cannot answer that question — the curve is concave, so a noisier render
    // encodes brighter regardless of any bias it actually carries.
    println!(
        "{}{}  {}",
        "noise".bold().green(),
        ":".bold(),
        render.statistics,
    );

    // Where the fireflies are, when something wants looking at rather than
    // measuring.
    if let Some(path) = &args.variance {
        let heatmap = render
            .statistics
            .heatmap
            .as_ref()
            .expect("a render asked for the variance field should carry it");
        heatmap.save(path)?;
        println!("        heatmap written to {}", path.display());
    }

    // What the tracer saw rather than what it computed: the frame before any
    // filtering, and the surfaces behind it, for a denoiser that was not written
    // here to steer by. The raw frame lands here whether or not the filter ran,
    // so the directory holds the same set either way.
    if let Some(directory) = &args.debug {
        render.image.save(&directory.join(RAW))?;

        let aovs = render
            .aovs
            .as_ref()
            .expect("a render asked for the feature buffers should carry them");
        aovs.normal.save(&directory.join("normal.png"))?;
        aovs.albedo.save(&directory.join("albedo.png"))?;
        aovs.depth.save(&directory.join("depth.png"))?;

        // The depth image has no scale of its own — it is normalised against
        // whatever the farthest surface in this frame turned out to be — so the
        // divisor is quoted rather than left for somebody to guess at.
        println!(
            "{}{}  {RAW}, normal, albedo and depth written to {} · depth over {:.2} world units",
            "debug".bold().green(),
            ":".bold(),
            directory.display(),
            aovs.far,
        );
    }

    // Wall time above covers the scene load, the uploads and every host stall
    // in the sample loop. This is the dispatches alone, and is the number a
    // change to the shader should be judged by. Absent on an adapter without
    // timestamp queries.
    if let Some(timings) = render.timings {
        println!("{}{}    {}", "gpu".bold().green(), ":".bold(), timings);
    }

    Ok(())
}
