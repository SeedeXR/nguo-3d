// SPDX-License-Identifier: GPL-2.0-or-later
//! `nguo` — headless, scriptable driver for the nguo-3d cloth suite.
//!
//! Process contract, identical to the `mguso` engine binary
//! (`core-engine/docs/PROCESS_CONTRACT.md`):
//!   * stdout — exactly one pretty JSON value per invocation, nothing else.
//!   * stderr — logs (`tracing`); `--log-format json` for one record per line.
//!   * exit 0 success, 1 failure, 2 the input was understood and is invalid.

use clap::Parser;
use nguo_cli::{
    cli::{init_logging, Cli},
    ops::{dispatch, Env, OpError},
};

#[tokio::main]
async fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    init_logging(cli.log_format, cli.verbose);
    let env = match cli.engine {
        Some(engine) => Env { engine },
        None => Env::default(),
    };
    let (op, args) = cli.command.to_op();
    match dispatch(op, &args, &env).await {
        Ok(value) => {
            print_json(&value);
            std::process::ExitCode::SUCCESS
        }
        // The input is invalid: the report IS the result, so it still goes to
        // stdout — but the exit code says it failed.
        Err(OpError::Invalid(report)) => {
            print_json(&report);
            tracing::error!(op, "validation failed");
            std::process::ExitCode::from(2)
        }
        Err(e) => {
            tracing::error!(op, error = %e, "command failed");
            std::process::ExitCode::FAILURE
        }
    }
}

fn print_json(value: &serde_json::Value) {
    println!(
        "{}",
        serde_json::to_string_pretty(value).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
    );
}
