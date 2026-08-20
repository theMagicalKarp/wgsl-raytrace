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
    let render = render::render(&config, &scene)?;
    render.image.save(&args.output)?;

    println!(
        "{}{} {}x{} written to {} in {:.1}s on {}",
        "render".bold().green(),
        ":".bold(),
        render.image.width,
        render.image.height,
        args.output.display(),
        started.elapsed().as_secs_f32(),
        render.renderer,
    );

    // Wall time above covers the scene load, the uploads and every host stall
    // in the sample loop. This is the dispatches alone, and is the number a
    // change to the shader should be judged by. Absent on an adapter without
    // timestamp queries.
    if let Some(timings) = render.timings {
        println!("{}{}    {}", "gpu".bold().green(), ":".bold(), timings);
    }

    Ok(())
}
