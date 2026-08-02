// SPDX-License-Identifier: GPL-2.0-or-later
//! The process transport: drive the `mguso` binary over its documented process
//! contract (`core-engine/docs/PROCESS_CONTRACT.md`).
//!
//! Contract as observed in `core-engine/crates/engine-cli/src/main.rs`:
//! `mguso run <scene.json> --frames N --out DIR --solver xpbd|implicit` writes
//! `DIR/frame_%04d.obj` as it goes, one pretty JSON summary to stdout at the
//! end, `tracing` logs to stderr, exit 0 on success and 1 on failure.

use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::transport::{BridgeError, Flow, FrameSink, Submission, Summary, Transport};

/// Locate the `mguso` binary.
///
/// 1. `$NGUO_MGUSO` if set (the escape hatch for CI and odd layouts),
/// 2. the sibling `core-engine` checkout's release then debug target dir,
/// 3. plain `mguso`, resolved by the OS from `PATH`.
pub fn resolve_engine_binary() -> PathBuf {
    if let Some(p) = std::env::var_os("NGUO_MGUSO") {
        return PathBuf::from(p);
    }
    // `ui/crates/nguo-engine-bridge` -> `nguo-3d`
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .map(Path::to_path_buf);
    if let Some(repo) = repo {
        for profile in ["release", "debug"] {
            let candidate = repo
                .join("core-engine")
                .join("target")
                .join(profile)
                .join("mguso");
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    PathBuf::from("mguso")
}

/// Runs drapes by spawning `mguso`. See the crate docs for why this, and not a
/// path dependency on `engine-core`, is the implementation that exists today.
pub struct ProcessTransport {
    binary: PathBuf,
    /// How often to look for the next finished frame file.
    // ponytail: filesystem polling, because `mguso run` streams frames as OBJ
    // files and nothing else. Upgrade path: have the engine emit NDJSON frames
    // on a pipe, or drop in the in-process transport once wgpu unifies —
    // either way only this file changes.
    poll: std::time::Duration,
}

impl Default for ProcessTransport {
    fn default() -> Self {
        Self::new(resolve_engine_binary())
    }
}

impl ProcessTransport {
    pub fn new(binary: PathBuf) -> Self {
        Self {
            binary,
            poll: std::time::Duration::from_millis(2),
        }
    }

    pub fn binary(&self) -> &Path {
        &self.binary
    }

    /// Run a non-`run` subcommand (e.g. `gpu-info`, `verify`) and return its
    /// stdout JSON. Exit code 2 is `verify`'s "invariants failed", which is a
    /// result, not an error — it comes back as `Ok` with the report.
    pub fn run_json(
        &self,
        args: &[&str],
        ok_codes: &[i32],
    ) -> Result<serde_json::Value, BridgeError> {
        let out = Command::new(&self.binary)
            .args(args)
            .output()
            .map_err(|e| BridgeError::Binary(format!("{}: {e}", self.binary.display())))?;
        let code = out.status.code().unwrap_or(-1);
        if !ok_codes.contains(&code) {
            return Err(BridgeError::Engine {
                code,
                context: format!("mguso {}", args.join(" ")),
                stderr_tail: tail_of(&String::from_utf8_lossy(&out.stderr)),
            });
        }
        let stdout = String::from_utf8_lossy(&out.stdout);
        Ok(serde_json::from_str(&stdout)?)
    }
}

impl Transport for ProcessTransport {
    fn solve(&self, sub: &Submission, sink: &mut dyn FrameSink) -> Result<Summary, BridgeError> {
        let scratch = tempfile::Builder::new().prefix("nguo-drape-").tempdir()?;
        let scene_path = scratch.path().join("scene.json");
        std::fs::write(&scene_path, serde_json::to_vec_pretty(&sub.scene)?)?;

        let frame_dir = match &sub.out_dir {
            Some(dir) => {
                std::fs::create_dir_all(dir)?;
                dir.clone()
            }
            None => scratch.path().join("frames"),
        };
        let log_path = scratch.path().join("engine.log");
        let log = std::fs::File::create(&log_path)?;

        let mut cmd = Command::new(&self.binary);
        cmd.arg("run")
            .arg(&scene_path)
            .arg("--out")
            .arg(&frame_dir)
            .arg("--solver")
            .arg(sub.tier.solver_flag())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(log));
        if let Some(frames) = sub.frames {
            cmd.arg("--frames").arg(frames.to_string());
        }
        let mut child = cmd
            .spawn()
            .map_err(|e| BridgeError::Binary(format!("{}: {e}", self.binary.display())))?;

        let mut next = 0u32;
        let status = loop {
            let exited = child.try_wait()?;
            // Frame N is only safe to read once N+1 exists (the engine writes
            // them in order) or the engine has exited and flushed everything —
            // otherwise we could parse a half-written file.
            while frame_complete(&frame_dir, next, exited.is_some()) {
                let positions = read_obj_positions(&frame_path(&frame_dir, next))?;
                if sink.push(next, positions) == Flow::Cancel {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(BridgeError::Cancelled);
                }
                next += 1;
            }
            if let Some(status) = exited {
                break status;
            }
            if sink.is_cancelled() {
                let _ = child.kill();
                let _ = child.wait();
                return Err(BridgeError::Cancelled);
            }
            std::thread::sleep(self.poll);
        };

        // Safe to read only now: `mguso run` writes its single summary object
        // at the very end, so the pipe cannot fill and deadlock the child.
        let mut stdout = String::new();
        if let Some(mut pipe) = child.stdout.take() {
            use std::io::Read;
            pipe.read_to_string(&mut stdout)?;
        }
        let engine_log = std::fs::read_to_string(&log_path).unwrap_or_default();

        if !status.success() {
            // Surface the engine's own words, never a paraphrase, and mirror
            // them onto our stderr so a CLI user sees them too.
            tracing::error!(engine_log = %engine_log.trim(), "mguso run failed");
            return Err(BridgeError::Engine {
                code: status.code().unwrap_or(-1),
                context: format!("mguso run {}", scene_path.display()),
                stderr_tail: tail_of(&engine_log),
            });
        }
        tracing::debug!(engine_log = %engine_log.trim(), "mguso run ok");

        Ok(Summary {
            tier: sub.tier,
            frames_emitted: next,
            report: serde_json::from_str(&stdout)?,
        })
    }
}

fn frame_path(dir: &Path, index: u32) -> PathBuf {
    dir.join(format!("frame_{index:04}.obj"))
}

fn frame_complete(dir: &Path, index: u32, engine_exited: bool) -> bool {
    frame_path(dir, index).is_file() && (engine_exited || frame_path(dir, index + 1).is_file())
}

/// Read the `v x y z` lines of a Wavefront OBJ. That is all the engine writes
/// per frame, and all a drape frame is.
fn read_obj_positions(path: &Path) -> Result<Vec<[f32; 3]>, BridgeError> {
    let file = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut out = Vec::new();
    for line in file.lines() {
        let line = line?;
        let Some(rest) = line.strip_prefix("v ") else {
            continue;
        };
        let mut it = rest
            .split_whitespace()
            .filter_map(|t| t.parse::<f32>().ok());
        match (it.next(), it.next(), it.next()) {
            (Some(x), Some(y), Some(z)) => out.push([x, y, z]),
            _ => {
                return Err(BridgeError::Binary(format!(
                    "malformed vertex line in {}: {line:?}",
                    path.display()
                )))
            }
        }
    }
    Ok(out)
}

/// Last 40 lines of the engine log — enough to carry the `ERROR` line and its
/// context without pasting a whole debug session into an error message.
fn tail_of(log: &str) -> String {
    let lines: Vec<&str> = log.lines().collect();
    lines[lines.len().saturating_sub(40)..].join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obj_positions_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("frame_0000.obj");
        std::fs::write(
            &path,
            "# comment\nv 1 2 3\nvn 0 1 0\nv -0.5 0.25 1e2\nf 1 2 3\n",
        )
        .unwrap();
        assert_eq!(
            read_obj_positions(&path).unwrap(),
            vec![[1.0, 2.0, 3.0], [-0.5, 0.25, 100.0]]
        );
    }

    #[test]
    fn a_frame_is_only_complete_once_its_successor_exists_or_the_engine_exited() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(frame_path(dir.path(), 0), "v 0 0 0\n").unwrap();
        // Still being written as far as we know: no successor, engine alive.
        assert!(!frame_complete(dir.path(), 0, false));
        assert!(frame_complete(dir.path(), 0, true));
        std::fs::write(frame_path(dir.path(), 1), "v 0 0 0\n").unwrap();
        assert!(frame_complete(dir.path(), 0, false));
        assert!(!frame_complete(dir.path(), 1, false));
    }
}
