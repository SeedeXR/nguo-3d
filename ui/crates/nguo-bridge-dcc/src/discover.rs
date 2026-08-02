// SPDX-License-Identifier: GPL-2.0-or-later
//! Locate the Blender / Unity executables on macOS, Linux and Windows.
//!
//! The environment override (`NGUO_BLENDER` / `NGUO_UNITY`) always wins, so CI
//! and users can pin an exact build. After that:
//!
//! * **Blender** — `PATH`, then the platform install location.
//! * **Unity** — the Hub install directory (one sub-directory per version),
//!   then `PATH`. `PATH` is *last* here on purpose; see
//!   [`path_is_last_resort`].

use crate::{BridgeError, Dcc};
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

/// Environment variable that pins this DCC's executable.
#[must_use]
pub const fn env_var(dcc: Dcc) -> &'static str {
    match dcc {
        Dcc::Blender => "NGUO_BLENDER",
        Dcc::Unity => "NGUO_UNITY",
    }
}

/// Bare names to look for on `PATH`.
const fn path_names(dcc: Dcc) -> &'static [&'static str] {
    match dcc {
        Dcc::Blender => &["blender"],
        Dcc::Unity => &["Unity", "unity"],
    }
}

/// Whether `PATH` is searched **last** rather than first, for this DCC.
///
/// Unity: yes. Observed on this machine — `~/.unity/bin/unity` is on `PATH`
/// and is the **Unity CLI** (`unity --version` -> `1.0.0-beta.3`), a different
/// program from the Editor; handing it `-batchmode` gets
/// `error: unknown option '-batchmode'`. macOS's case-insensitive filesystem
/// means looking for `Unity` finds it too. The Hub directory is the canonical
/// Editor location, so it wins and `PATH` is the fallback. `NGUO_UNITY` still
/// overrides everything.
///
/// Blender: no. `blender` on `PATH` is always the real thing.
const fn path_is_last_resort(dcc: Dcc) -> bool {
    matches!(dcc, Dcc::Unity)
}

/// Fixed install locations for Blender.
fn blender_fixed() -> Vec<PathBuf> {
    if cfg!(target_os = "macos") {
        vec![PathBuf::from(
            "/Applications/Blender.app/Contents/MacOS/Blender",
        )]
    } else if cfg!(windows) {
        // Blender installs one directory per minor version; the Hub-style scan
        // below handles that, so only the stable shim is listed here.
        vec![PathBuf::from(
            r"C:\Program Files\Blender Foundation\Blender\blender.exe",
        )]
    } else {
        vec![
            PathBuf::from("/usr/bin/blender"),
            PathBuf::from("/usr/local/bin/blender"),
            PathBuf::from("/snap/bin/blender"),
        ]
    }
}

/// Roots that hold one sub-directory per installed version, plus the path from
/// such a sub-directory down to the executable.
///
/// Verified on this machine: `/Applications/Unity/Hub/Editor/6000.4.8f1/Unity.app/Contents/MacOS/Unity`.
fn version_roots(dcc: Dcc) -> Vec<(PathBuf, &'static str)> {
    match dcc {
        Dcc::Unity => {
            if cfg!(target_os = "macos") {
                vec![(
                    PathBuf::from("/Applications/Unity/Hub/Editor"),
                    "Unity.app/Contents/MacOS/Unity",
                )]
            } else if cfg!(windows) {
                vec![(
                    PathBuf::from(r"C:\Program Files\Unity\Hub\Editor"),
                    r"Editor\Unity.exe",
                )]
            } else {
                let home =
                    std::env::var_os("HOME").map_or_else(|| PathBuf::from("/root"), PathBuf::from);
                vec![(home.join("Unity/Hub/Editor"), "Editor/Unity")]
            }
        }
        Dcc::Blender => {
            if cfg!(windows) {
                vec![(
                    PathBuf::from(r"C:\Program Files\Blender Foundation"),
                    r"blender.exe",
                )]
            } else {
                Vec::new()
            }
        }
    }
}

/// Newest install under `root/<version>/<suffix>`, by descending directory
/// name.
///
/// ponytail: lexicographic ordering, not semver. It gets Unity right
/// (`6000.4.8f1` > `2022.3.1f1` because `'6' > '2'`) and Blender right
/// (`Blender 4.5` < `Blender 5.2`). Swap in a real version parse if a scheme
/// ever appears where it doesn't.
fn newest_under(root: &Path, suffix: &str) -> Option<PathBuf> {
    let mut names: Vec<OsString> = std::fs::read_dir(root)
        .ok()?
        .filter_map(|e| {
            let e = e.ok()?;
            e.file_type().ok()?.is_dir().then(|| e.file_name())
        })
        .collect();
    names.sort();
    names
        .into_iter()
        .rev()
        .map(|n| root.join(n).join(suffix))
        .find(|p| p.is_file())
}

/// First `name` on `path_env` that is a file.
fn on_path(path_env: Option<&OsStr>, names: &[&str]) -> Option<PathBuf> {
    let path_env = path_env?;
    std::env::split_paths(path_env)
        .find_map(|dir| names.iter().map(|n| dir.join(n)).find(|p| p.is_file()))
}

/// Locate `dcc`, honouring the real process environment.
///
/// # Errors
/// [`BridgeError::NotFound`] listing every candidate that was tried.
pub fn find(dcc: Dcc) -> Result<PathBuf, BridgeError> {
    find_with(
        dcc,
        std::env::var_os(env_var(dcc)).as_deref(),
        std::env::var_os("PATH").as_deref(),
    )
}

/// [`find`] with the two environment inputs injected, so the search order is
/// testable without touching the process environment.
///
/// # Errors
/// [`BridgeError::NotFound`] listing every candidate that was tried.
pub fn find_with(
    dcc: Dcc,
    env_override: Option<&OsStr>,
    path_env: Option<&OsStr>,
) -> Result<PathBuf, BridgeError> {
    let fixed = match dcc {
        Dcc::Blender => blender_fixed(),
        Dcc::Unity => Vec::new(),
    };
    search(dcc, env_override, path_env, &fixed, &version_roots(dcc))
}

/// The whole search with every location injected — the unit-testable core.
fn search(
    dcc: Dcc,
    env_override: Option<&OsStr>,
    path_env: Option<&OsStr>,
    fixed: &[PathBuf],
    roots: &[(PathBuf, &str)],
) -> Result<PathBuf, BridgeError> {
    let mut tried = Vec::new();

    if let Some(p) = env_override {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Ok(p);
        }
        tried.push(p);
    }
    let path_last = path_is_last_resort(dcc);
    if !path_last {
        if let Some(p) = on_path(path_env, path_names(dcc)) {
            return Ok(p);
        }
    }
    for p in fixed {
        if p.is_file() {
            return Ok(p.clone());
        }
        tried.push(p.clone());
    }
    for (root, suffix) in roots {
        if let Some(p) = newest_under(root, suffix) {
            return Ok(p);
        }
        tried.push(root.clone());
    }
    if path_last {
        if let Some(p) = on_path(path_env, path_names(dcc)) {
            return Ok(p);
        }
    }
    Err(BridgeError::NotFound {
        dcc,
        tried,
        env: env_var(dcc),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch dir that cleans itself up. `std::env::temp_dir` + the test
    /// name is enough; no tempfile dependency for four tests.
    struct Scratch(PathBuf);
    impl Scratch {
        fn new(tag: &str) -> Self {
            let p = std::env::temp_dir().join(format!("nguo-bridge-dcc-{tag}"));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
        fn touch(&self, rel: &str) -> PathBuf {
            let p = self.0.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, b"#!/bin/sh\n").unwrap();
            p
        }
        fn dir(&self, rel: &str) -> PathBuf {
            let p = self.0.join(rel);
            std::fs::create_dir_all(&p).unwrap();
            p
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn joined(dirs: &[&Path]) -> OsString {
        std::env::join_paths(dirs.iter().map(|d| d.to_path_buf())).unwrap()
    }

    #[test]
    fn env_override_wins_over_path() {
        let s = Scratch::new("override");
        let pinned = s.touch("pinned/blender");
        let on_path = s.touch("bin/blender");
        let bin = on_path.parent().unwrap();
        let got = find_with(
            Dcc::Blender,
            Some(pinned.as_os_str()),
            Some(&joined(&[bin])),
        )
        .unwrap();
        assert_eq!(got, pinned);
    }

    #[test]
    fn falls_through_to_path_when_override_is_missing() {
        let s = Scratch::new("fallthrough");
        let on_path = s.touch("bin/blender");
        let bin = on_path.parent().unwrap().to_path_buf();
        let ghost = s.0.join("does-not-exist");
        let got = find_with(
            Dcc::Blender,
            Some(ghost.as_os_str()),
            Some(&joined(&[&bin])),
        )
        .unwrap();
        assert_eq!(got, on_path);
    }

    #[test]
    fn path_is_scanned_in_order_and_directories_are_not_executables() {
        let s = Scratch::new("order");
        let decoy = s.dir("first/blender"); // a *directory* named blender
        let real = s.touch("second/blender");
        let got = find_with(
            Dcc::Blender,
            None,
            Some(&joined(&[decoy.parent().unwrap(), real.parent().unwrap()])),
        )
        .unwrap();
        assert_eq!(got, real);
    }

    #[test]
    fn unity_falls_back_to_path_when_no_hub_install_exists() {
        let s = Scratch::new("unity-path");
        let exe = s.touch("bin/Unity");
        let got = search(
            Dcc::Unity,
            None,
            Some(&joined(&[exe.parent().unwrap()])),
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(got, exe);
    }

    /// `~/.unity/bin/unity` on this machine is the Unity **CLI**, not the
    /// Editor, and it rejects `-batchmode`. The Hub install must win over it.
    #[test]
    fn unity_hub_install_beats_a_unity_on_path() {
        let s = Scratch::new("unity-hub-wins");
        let decoy = s.touch("bin/unity");
        let root = s.dir("Hub/Editor");
        let hub = s.touch("Hub/Editor/6000.4.8f1/Unity.app/Contents/MacOS/Unity");
        let got = search(
            Dcc::Unity,
            None,
            Some(&joined(&[decoy.parent().unwrap()])),
            &[],
            &[(root, "Unity.app/Contents/MacOS/Unity")],
        )
        .unwrap();
        assert_eq!(got, hub);
    }

    /// ...but an explicit override still beats the Hub.
    #[test]
    fn unity_env_override_beats_the_hub() {
        let s = Scratch::new("unity-override");
        let pinned = s.touch("pinned/Unity");
        let root = s.dir("Hub/Editor");
        s.touch("Hub/Editor/6000.4.8f1/Unity.app/Contents/MacOS/Unity");
        let got = search(
            Dcc::Unity,
            Some(pinned.as_os_str()),
            None,
            &[],
            &[(root, "Unity.app/Contents/MacOS/Unity")],
        )
        .unwrap();
        assert_eq!(got, pinned);
    }

    #[test]
    fn newest_version_dir_wins() {
        let s = Scratch::new("versions");
        let root = s.dir("Hub/Editor");
        s.touch("Hub/Editor/2022.3.1f1/Unity.app/Contents/MacOS/Unity");
        s.touch("Hub/Editor/6000.4.8f1/Unity.app/Contents/MacOS/Unity");
        let got = newest_under(&root, "Unity.app/Contents/MacOS/Unity").unwrap();
        assert!(
            got.to_string_lossy().contains("6000.4.8f1"),
            "picked {got:?}"
        );
    }

    #[test]
    fn not_found_reports_what_it_tried_and_how_to_override() {
        let s = Scratch::new("notfound");
        let empty = s.dir("empty");
        let ghost = s.0.join("nowhere");
        // `search`, not `find_with`: this machine really does have Unity in
        // /Applications, and the point of the test is the not-found path.
        let err = search(
            Dcc::Unity,
            None,
            Some(&joined(&[&empty])),
            std::slice::from_ref(&ghost),
            &[(ghost.clone(), "Editor/Unity")],
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("NGUO_UNITY"), "{msg}");
        assert!(matches!(
            err,
            BridgeError::NotFound {
                dcc: Dcc::Unity,
                ..
            }
        ));
    }
}
