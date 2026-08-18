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
        "{}{} {} triangles across {} materials",
        "scene".bold().green(),
        ":".bold(),
        scene.triangles.len(),
        scene.materials.len(),
    );

    let started = Instant::now();
    let (image, renderer) = render::render(&config, &scene)?;
    image.save(&args.output)?;

    println!(
        "{}{} {}x{} written to {} in {:.1}s on {}",
        "render".bold().green(),
        ":".bold(),
        image.width,
        image.height,
        args.output.display(),
        started.elapsed().as_secs_f32(),
        renderer,
    );
    println!(
        "{}{} the tracer is a stand-in: every ray reaches the background, so \
         samples and bounces are not honored yet",
        "note".bold().yellow(),
        ":".bold(),
    );

    Ok(())
}
