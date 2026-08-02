// SPDX-License-Identifier: GPL-2.0-or-later
//! `nguo-bridge-dcc` — drive Blender and Unity headlessly from nguo-3d.
//!
//! This crate is the Rust half of the interop layer: it finds the DCC
//! executables ([`discover`]), builds the exact argv for a headless run
//! ([`blender_argv`] / [`unity_argv`]), spawns it while streaming its output,
//! maps exit codes to typed errors ([`BridgeError`]), and lifts the single
//! machine-readable line the DCC-side scripts print ([`RESULT_PREFIX`]).
//!
//! It also owns the coordinate-system maths ([`axis`]) — the part that, when
//! wrong, silently mirrors garments.
//!
//! No GUI dependency: the app, the CLI and the MCP server all call the same
//! functions. See `ui/docs/INTEROP.md` for the full data flow.

pub mod axis;
pub mod discover;

use std::ffi::OsString;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub use discover::find;

/// Which DCC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Dcc {
    Blender,
    Unity,
}

impl std::fmt::Display for Dcc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Blender => "Blender",
            Self::Unity => "Unity",
        })
    }
}

/// Exit code our Blender scripts are told to use for an uncaught Python
/// exception (`blender --python-exit-code`), so a *script* failure is
/// distinguishable from Blender itself dying. 47 is arbitrary but outside
/// Blender's own range.
pub const BLENDER_PYTHON_EXIT_CODE: i32 = 47;

/// Both DCC-side scripts print exactly one line of the form
/// `NGUO-BRIDGE-RESULT: {json}`. Everything else on the stream is log noise
/// (Unity in particular is extremely chatty), so a sentinel is the only robust
/// way to get a structured answer back out of a batch run.
pub const RESULT_PREFIX: &str = "NGUO-BRIDGE-RESULT:";

/// Exit code Unity uses for "No valid Unity Editor license found. Please
/// activate your license." — observed directly, Unity 6000.4.8f1 on macOS.
pub const UNITY_NO_LICENCE: i32 = 198;

/// How many trailing log lines are quoted back in an error.
const LOG_TAIL_LINES: usize = 40;

#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    #[error("{dcc} executable not found (tried {tried:?}); set {env} to point at it")]
    NotFound {
        dcc: Dcc,
        tried: Vec<PathBuf>,
        env: &'static str,
    },
    #[error("could not spawn {dcc}")]
    Spawn {
        dcc: Dcc,
        #[source]
        source: std::io::Error,
    },
    /// The DCC started fine but our script inside it failed.
    #[error("the nguo script inside {dcc} failed (exit {code})\n{log}")]
    Script { dcc: Dcc, code: i32, log: String },
    /// The DCC itself failed — bad project, no licence, crash.
    #[error("{dcc} failed (exit {code})\n{log}")]
    Dcc { dcc: Dcc, code: i32, log: String },
    #[error("{dcc} was terminated by a signal")]
    Killed { dcc: Dcc },
    /// Unity exits **198** with "No valid Unity Editor license found" — observed
    /// on this machine, Unity 6000.4.8f1. It is by far the most common way a
    /// headless Unity run dies in CI, and the generic "exit 198" tells nobody
    /// anything, so it gets its own variant.
    #[error("{dcc} has no activated licence (exit {UNITY_NO_LICENCE}); activate it in Unity Hub or with `-serial`/`UNITY_LICENSE`")]
    NoLicence { dcc: Dcc },
    #[error("{dcc} exited 0 but printed no `{RESULT_PREFIX}` line\n{log}")]
    NoResult { dcc: Dcc, log: String },
}

/// A completed headless run.
#[derive(Debug)]
pub struct DccRun {
    pub dcc: Dcc,
    /// Everything the process wrote, in arrival order, stdout and stderr merged.
    pub log: String,
    /// The payload of the last [`RESULT_PREFIX`] line, if any.
    pub result: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// argv
// ---------------------------------------------------------------------------

/// argv (excluding argv[0]) for `blender --background`.
///
/// Blender processes arguments **in order**, so `--python-exit-code` has to
/// precede `--python`, and everything after a bare `--` is left for the script
/// to read out of `sys.argv`. Flags verified against `blender --help` on
/// Blender 5.2.0 LTS.
///
/// `--factory-startup` is deliberate: without it a user's installed add-ons and
/// preferences change the result, which is exactly what a CI run must not
/// depend on.
#[must_use]
pub fn blender_argv(script: &Path, script_args: &[String]) -> Vec<OsString> {
    let mut argv: Vec<OsString> = vec![
        "--background".into(),
        "--factory-startup".into(),
        "--python-exit-code".into(),
        BLENDER_PYTHON_EXIT_CODE.to_string().into(),
        "--python".into(),
        script.into(),
    ];
    if !script_args.is_empty() {
        argv.push("--".into());
        argv.extend(script_args.iter().map(Into::into));
    }
    argv
}

/// argv (excluding argv[0]) for a Unity Editor batch run.
///
/// Flags and their semantics are from Unity's *Editor command line arguments*
/// reference (<https://docs.unity3d.com/Manual/EditorCommandLineArguments.html>):
///
/// * `-batchmode` — no human interaction; Unity exits with **1** on any
///   uncaught exception.
/// * `-nographics` — do not initialise the graphics device. **"Output logs are
///   turned off in this mode. To enable the creation of output logs, specify a
///   file location using `-logFile`."** — which is why `-logFile -` is not
///   optional here: it is the only way we see the run at all.
/// * `-logFile -` — write the Editor log to stdout.
/// * `-quit` — quit once the other commands finish.
/// * `-projectPath <p>` — the project to open.
/// * `-executeMethod <Class.Method>` — a **static** method run after the
///   project opens. "To return an error from the command line process, either
///   throw an exception ... or call `EditorApplication.Exit` with a non-zero
///   return code."
///
/// Unity has no first-class way to pass parameters to that method, so extra
/// arguments are appended verbatim and the C# side reads them back with
/// `System.Environment.GetCommandLineArgs()` — again, per the same reference.
#[must_use]
pub fn unity_argv(project: &Path, method: &str, script_args: &[String]) -> Vec<OsString> {
    let mut argv: Vec<OsString> = vec![
        "-batchmode".into(),
        "-nographics".into(),
        "-quit".into(),
        "-logFile".into(),
        "-".into(),
        "-projectPath".into(),
        project.into(),
        "-executeMethod".into(),
        method.into(),
    ];
    argv.extend(script_args.iter().map(Into::into));
    argv
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

/// Spawn `exe` with `argv`, stream its output to `tracing` as it arrives, and
/// classify the exit code.
///
/// # Errors
/// See [`BridgeError`]; every non-zero exit becomes a typed variant carrying
/// the tail of the log.
pub fn run(dcc: Dcc, exe: &Path, argv: &[OsString]) -> Result<DccRun, BridgeError> {
    tracing::info!(%dcc, exe = %exe.display(), ?argv, "spawning DCC");
    let mut child = Command::new(exe)
        .args(argv)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|source| BridgeError::Spawn { dcc, source })?;

    // stderr on a helper thread so a chatty stream cannot deadlock the pipe
    // while we are reading the other one.
    let err_pipe = child.stderr.take().expect("stderr piped");
    let err_thread = std::thread::spawn(move || {
        let mut buf = String::new();
        for line in BufReader::new(err_pipe).lines().map_while(Result::ok) {
            tracing::debug!(target: "nguo_bridge_dcc::stderr", "{line}");
            buf.push_str(&line);
            buf.push('\n');
        }
        buf
    });

    let out_pipe = child.stdout.take().expect("stdout piped");
    let mut log = String::new();
    for line in BufReader::new(out_pipe).lines().map_while(Result::ok) {
        tracing::debug!(target: "nguo_bridge_dcc::stdout", "{line}");
        log.push_str(&line);
        log.push('\n');
    }
    log.push_str(&err_thread.join().unwrap_or_default());

    let status = child
        .wait()
        .map_err(|source| BridgeError::Spawn { dcc, source })?;
    let result = parse_result(&log);
    match classify(dcc, status.code(), &log) {
        Some(e) => Err(e),
        None if result.is_none() => Err(BridgeError::NoResult {
            dcc,
            log: tail(&log),
        }),
        None => Ok(DccRun { dcc, log, result }),
    }
}

/// Locate the DCC and run it in one call.
///
/// # Errors
/// See [`BridgeError`].
pub fn run_blender(script: &Path, script_args: &[String]) -> Result<DccRun, BridgeError> {
    let exe = find(Dcc::Blender)?;
    run(Dcc::Blender, &exe, &blender_argv(script, script_args))
}

/// Locate Unity and run a static method in a project.
///
/// # Errors
/// See [`BridgeError`].
pub fn run_unity(
    project: &Path,
    method: &str,
    script_args: &[String],
) -> Result<DccRun, BridgeError> {
    let exe = find(Dcc::Unity)?;
    run(Dcc::Unity, &exe, &unity_argv(project, method, script_args))
}

/// Map an exit code to a typed error. `None` = success.
#[must_use]
pub fn classify(dcc: Dcc, code: Option<i32>, log: &str) -> Option<BridgeError> {
    let log = tail(log);
    match (dcc, code) {
        (_, Some(0)) => None,
        (_, None) => Some(BridgeError::Killed { dcc }),
        (Dcc::Unity, Some(UNITY_NO_LICENCE)) => Some(BridgeError::NoLicence { dcc }),
        // Blender: we asked for BLENDER_PYTHON_EXIT_CODE on a Python exception,
        // so that code and only that code means "our script broke".
        (Dcc::Blender, Some(c)) if c == BLENDER_PYTHON_EXIT_CODE => {
            Some(BridgeError::Script { dcc, code: c, log })
        }
        // Unity: 1 is Unity's own documented "an exception occurred". Anything
        // in 2..=125 is a code our C# chose via EditorApplication.Exit.
        (Dcc::Unity, Some(c)) if (2..=125).contains(&c) => {
            Some(BridgeError::Script { dcc, code: c, log })
        }
        (_, Some(code)) => Some(BridgeError::Dcc { dcc, code, log }),
    }
}

/// Payload of the last [`RESULT_PREFIX`] line in `log`.
#[must_use]
pub fn parse_result(log: &str) -> Option<serde_json::Value> {
    log.lines()
        .rev()
        .find_map(|l| l.trim_start().strip_prefix(RESULT_PREFIX))
        .and_then(|json| serde_json::from_str(json.trim()).ok())
}

fn tail(log: &str) -> String {
    let lines: Vec<&str> = log.lines().collect();
    let start = lines.len().saturating_sub(LOG_TAIL_LINES);
    lines[start..].join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[OsString]) -> Vec<String> {
        v.iter().map(|o| o.to_string_lossy().into_owned()).collect()
    }

    #[test]
    fn blender_argv_is_exact() {
        assert_eq!(
            s(&blender_argv(Path::new("/tmp/h.py"), &[])),
            [
                "--background",
                "--factory-startup",
                "--python-exit-code",
                "47",
                "--python",
                "/tmp/h.py",
            ]
        );
    }

    #[test]
    fn blender_argv_puts_script_args_after_a_bare_double_dash() {
        let argv = s(&blender_argv(
            Path::new("/tmp/h.py"),
            &["import-sequence".into(), "/out".into()],
        ));
        assert_eq!(
            argv,
            [
                "--background",
                "--factory-startup",
                "--python-exit-code",
                "47",
                "--python",
                "/tmp/h.py",
                "--",
                "import-sequence",
                "/out",
            ]
        );
        // The `--` must come after --python, or Blender consumes the args.
        let dd = argv.iter().position(|a| a == "--").unwrap();
        let py = argv.iter().position(|a| a == "--python").unwrap();
        assert!(py < dd);
    }

    #[test]
    fn unity_argv_is_exact() {
        assert_eq!(
            s(&unity_argv(
                Path::new("/proj"),
                "Nguo.Bridge.Headless.Import",
                &["--glb".into(), "/tmp/a.glb".into()]
            )),
            [
                "-batchmode",
                "-nographics",
                "-quit",
                "-logFile",
                "-",
                "-projectPath",
                "/proj",
                "-executeMethod",
                "Nguo.Bridge.Headless.Import",
                "--glb",
                "/tmp/a.glb",
            ]
        );
    }

    /// -nographics suppresses logs unless -logFile is given, so losing the
    /// -logFile pair would silently blind every Unity run.
    #[test]
    fn unity_argv_always_carries_logfile_because_nographics_mutes_output() {
        let argv = s(&unity_argv(Path::new("/p"), "C.M", &[]));
        let i = argv.iter().position(|a| a == "-logFile").unwrap();
        assert_eq!(argv[i + 1], "-");
        assert!(argv.contains(&"-nographics".to_string()));
    }

    #[test]
    fn exit_codes_map_to_typed_errors() {
        assert!(classify(Dcc::Blender, Some(0), "").is_none());
        assert!(classify(Dcc::Unity, Some(0), "").is_none());
        assert!(matches!(
            classify(Dcc::Blender, Some(BLENDER_PYTHON_EXIT_CODE), ""),
            Some(BridgeError::Script { code: 47, .. })
        ));
        assert!(matches!(
            classify(Dcc::Blender, Some(1), ""),
            Some(BridgeError::Dcc { code: 1, .. })
        ));
        // Unity's own generic failure is 1 -> Dcc, our chosen codes -> Script.
        assert!(matches!(
            classify(Dcc::Unity, Some(1), ""),
            Some(BridgeError::Dcc { code: 1, .. })
        ));
        assert!(matches!(
            classify(Dcc::Unity, Some(3), ""),
            Some(BridgeError::Script { code: 3, .. })
        ));
        assert!(matches!(
            classify(Dcc::Unity, None, ""),
            Some(BridgeError::Killed { .. })
        ));
    }

    /// Observed on this machine: an unlicensed Unity 6000.4.8f1 exits 198 with
    /// "No valid Unity Editor license found". That must not read as a generic
    /// failure — it is the single most common headless-Unity CI stall.
    #[test]
    fn unity_missing_licence_is_called_by_its_name() {
        let e = classify(
            Dcc::Unity,
            Some(198),
            "No valid Unity Editor license found.",
        )
        .unwrap();
        assert!(matches!(e, BridgeError::NoLicence { .. }));
        assert!(e.to_string().contains("licence"), "{e}");
        // Blender has no such code; 198 there is just a failure.
        assert!(matches!(
            classify(Dcc::Blender, Some(198), ""),
            Some(BridgeError::Dcc { code: 198, .. })
        ));
    }

    #[test]
    fn result_line_is_lifted_out_of_the_noise() {
        let log = "\
Blender 5.2.0 LTS
Read prefs: /whatever
NGUO-BRIDGE-RESULT: {\"objects\": 1, \"frames\": 24}
Blender quit
";
        let v = parse_result(log).expect("result parsed");
        assert_eq!(v["objects"], 1);
        assert_eq!(v["frames"], 24);
    }

    #[test]
    fn the_last_result_line_wins_and_indentation_is_tolerated() {
        let log = "NGUO-BRIDGE-RESULT: {\"n\":1}\n  NGUO-BRIDGE-RESULT: {\"n\":2}\n";
        assert_eq!(parse_result(log).unwrap()["n"], 2);
    }

    #[test]
    fn no_result_line_is_not_a_result() {
        assert!(parse_result("just logs\nmore logs\n").is_none());
        // ...and a malformed payload is not silently accepted as one.
        assert!(parse_result("NGUO-BRIDGE-RESULT: not json\n").is_none());
    }

    #[test]
    fn error_tail_is_bounded() {
        let big: String = (0..500).map(|i| format!("line {i}\n")).collect();
        let e = classify(Dcc::Unity, Some(1), &big).unwrap();
        let msg = e.to_string();
        assert!(
            msg.lines().count() <= LOG_TAIL_LINES + 2,
            "{}",
            msg.lines().count()
        );
        assert!(msg.contains("line 499"));
        assert!(!msg.contains("line 0\n"));
    }

    /// Exercise the real spawn/stream/classify path without needing a DCC.
    #[cfg(unix)]
    #[test]
    fn run_streams_output_and_lifts_the_result() {
        let argv: Vec<OsString> = vec![
            "-c".into(),
            format!("echo hello; echo '{RESULT_PREFIX} {{\"ok\":true}}'; echo oops 1>&2").into(),
        ];
        let out = run(Dcc::Blender, Path::new("/bin/sh"), &argv).unwrap();
        assert_eq!(out.result.unwrap()["ok"], true);
        assert!(out.log.contains("hello"));
        assert!(out.log.contains("oops"), "stderr must be captured too");
    }

    #[cfg(unix)]
    #[test]
    fn run_maps_a_failing_child_to_a_typed_error() {
        let argv: Vec<OsString> = vec!["-c".into(), "echo boom 1>&2; exit 47".into()];
        let e = run(Dcc::Blender, Path::new("/bin/sh"), &argv).unwrap_err();
        assert!(matches!(e, BridgeError::Script { code: 47, .. }));
        assert!(e.to_string().contains("boom"));
    }

    #[cfg(unix)]
    #[test]
    fn run_rejects_a_silent_success() {
        let argv: Vec<OsString> = vec!["-c".into(), "echo nothing useful".into()];
        let e = run(Dcc::Blender, Path::new("/bin/sh"), &argv).unwrap_err();
        assert!(matches!(e, BridgeError::NoResult { .. }));
    }

    #[test]
    fn spawn_failure_is_typed() {
        let e = run(Dcc::Unity, Path::new("/definitely/not/here"), &[]).unwrap_err();
        assert!(matches!(e, BridgeError::Spawn { .. }));
    }
}
