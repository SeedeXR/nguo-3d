// SPDX-License-Identifier: GPL-2.0-or-later
//! `nguo inspect` for a `.nguo`, until `nguo-cli` grows the verb:
//!
//! ```text
//! cargo run -p nguo-doc --example inspect -- path/to/project.nguo
//! ```
//!
//! Prints the manifest verbatim, the archive members with their sizes, and the
//! load time — so a project file can be understood without opening the app.
//! Wiring this into `nguo-cli` is three lines; it lives here because this crate
//! owns the format and not the CLI.

use std::path::PathBuf;

fn main() -> std::process::ExitCode {
    let Some(path) = std::env::args_os().nth(1).map(PathBuf::from) else {
        eprintln!("usage: inspect <project.nguo>");
        return std::process::ExitCode::FAILURE;
    };

    let t0 = std::time::Instant::now();
    let project = match nguo_doc::bundle::load(&path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("cannot open: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;

    println!(
        "{}",
        serde_json::to_string_pretty(&project.manifest).expect("a manifest always serializes")
    );
    match std::fs::read(&path).map(|raw| nguo_doc::bundle::members(&raw)) {
        Ok(Ok(members)) => {
            let on_disk = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            println!("\n{on_disk} bytes on disk, loaded in {load_ms:.2} ms");
            for (name, body) in &members {
                println!("  {name}: {} bytes uncompressed", body.len());
            }
        }
        _ => println!("\nloaded in {load_ms:.2} ms"),
    }
    std::process::ExitCode::SUCCESS
}
