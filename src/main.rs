//! Headless WebGPU (wgpu + WGSL) path tracer over Wavefront meshes.
//!
//! Today this is the front half only: it reads a scene, checks it, and reports
//! what it would render. The tracer itself lands behind this same CLI.

mod config;

use clap::Parser;
use colored::Colorize;
use config::Args;
use config::Config;
use config::format_error;
use std::error::Error;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

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
    println!(
        "{}{} rendering is not implemented yet, nothing was written to {}",
        "note".bold().yellow(),
        ":".bold(),
        args.output.display(),
    );

    Ok(())
}
