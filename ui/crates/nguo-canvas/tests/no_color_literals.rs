// SPDX-License-Identifier: GPL-2.0-or-later
//! `nguo-ui-theme` is the only crate allowed to hold a colour literal
//! (its module docs say so). This asserts the canvas keeps that rule, because
//! "restyle later without breaking functionality" is only true while it holds.
//!
//! Written as a source scan rather than a review checklist: a reviewer misses
//! one `Color::from_rgba8` in a 400-line diff, and this does not.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Ways a colour gets written in Rust without going through a token.
const BANNED: &[&str] = &[
    "from_rgba8",
    "from_rgb8",
    "Color::rgb",
    "Color::rgba",
    "Rgba::hex",
    "Rgba::new",
    "parse_color",
];

fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("the crate has a src directory") {
        let path = entry.expect("readable dir entry").path();
        if path.is_dir() {
            sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn no_colour_literal_escapes_the_theme_crate() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    sources(&root.join("src"), &mut files);
    sources(&root.join("tests"), &mut files);

    // A scan that scanned nothing is a test that passes by doing nothing —
    // exactly the failure this codebase has produced before.
    assert!(
        files.len() >= 8,
        "expected the crate's modules, found {} files",
        files.len()
    );

    let mut report = String::new();
    let mut token_uses = 0usize;
    for path in &files {
        let text = std::fs::read_to_string(path).expect("source is UTF-8");
        let name = path
            .strip_prefix(root)
            .unwrap_or(path)
            .display()
            .to_string();
        token_uses += text.matches("tokens.color.").count();
        for (n, line) in text.lines().enumerate() {
            // This file names the patterns it bans, and the doc comments in
            // `scene.rs` name the conversion function. Neither is a colour.
            if name.ends_with("no_color_literals.rs") {
                continue;
            }
            for pattern in BANNED {
                if line.contains(pattern) {
                    let _ = writeln!(report, "  {name}:{}: {}", n + 1, line.trim());
                }
            }
            // A bare hex colour: `0xRRGGBB` or `0xRRGGBBAA`, which is how one
            // sneaks in without any of the constructors above.
            if let Some(rest) = line.split("0x").nth(1) {
                let hex: String = rest.chars().take_while(|c| c.is_ascii_hexdigit()).collect();
                if (hex.len() == 6 || hex.len() == 8) && !line.contains("SPDX") {
                    let _ = writeln!(report, "  {name}:{}: {}", n + 1, line.trim());
                }
            }
        }
    }

    assert!(
        report.is_empty(),
        "colour literals must live in nguo-ui-theme:\n{report}"
    );

    // `Color::new` takes raw components, so it is the one call that *could*
    // smuggle a literal past the patterns above. There is exactly one: the
    // `Rgba -> peniko::Color` bridge in `scene::ink`.
    let constructions: usize = files
        .iter()
        .filter(|p| !p.ends_with("no_color_literals.rs"))
        .map(|p| {
            std::fs::read_to_string(p)
                .expect("source is UTF-8")
                .matches("Color::new(")
                .count()
        })
        .sum();
    assert_eq!(
        constructions, 1,
        "the only raw colour construction may be scene::ink"
    );

    // ...and the crate must actually be reading tokens, or it passes the scan by
    // drawing nothing.
    assert!(
        token_uses >= 10,
        "only {token_uses} token reads: is the canvas really themed?"
    );
}
