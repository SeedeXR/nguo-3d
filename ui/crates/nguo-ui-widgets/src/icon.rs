// SPDX-License-Identifier: GPL-2.0-or-later
//! Compile-time-embedded icon set.
//!
//! Lucide (ISC) is the icon set; Tabler (MIT) fills the two gaps Lucide has for
//! a pattern tool. Provenance and licences: `ui/assets/icons/LICENSE.md`.
//!
//! Every variant maps to a vendored file via `include_bytes!`, so a deleted or
//! renamed SVG is a **compile error**, not a blank square at runtime. There is
//! no icon-font crate and nothing is fetched at run time.

use iced::widget::svg;
use iced::{Element, Length};
use nguo_ui_theme::Rgba;

use crate::color;

/// Every icon the app may draw, named for what it *means* in a cloth tool,
/// not for the glyph it happens to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Icon {
    // --- 2D tools (guide §6.1) -------------------------------------------
    Select,
    DirectSelect,
    Pen,
    Curve,
    Line,
    Dart,
    Notch,
    InternalLine,
    Seam,
    Measure,
    Pan,
    ZoomIn,
    ZoomOut,
    Mirror,
    Point,
    Snap,
    Grain,
    Annotate,

    // --- 3D tools (guide §13) --------------------------------------------
    Box3d,
    Move3d,
    Rotate3d,
    Garment,

    // --- document / history ----------------------------------------------
    New,
    Open,
    Save,
    Import,
    Export,
    Undo,
    Redo,
    Copy,
    Paste,
    Trash,

    // --- simulation --------------------------------------------------------
    Play,
    Pause,
    Refresh,

    // --- view / chrome -----------------------------------------------------
    Layers,
    Grid,
    Eye,
    EyeOff,
    Lock,
    Unlock,
    Settings,
    Sliders,
    Palette,
    Sun,
    Moon,
    PanelLeft,
    PanelRight,
    PanelBottom,
    Maximize,
    Search,
    Scissors,

    // --- affordances -------------------------------------------------------
    Plus,
    Minus,
    ChevronUp,
    ChevronDown,
    ChevronLeft,
    ChevronRight,
    Check,
    Close,
    Info,
    Warning,
    Crosshair,
}

macro_rules! icons {
    ($($variant:ident => $file:literal),* $(,)?) => {
        impl Icon {
            /// The vendored SVG source. Embedded at build time.
            pub const fn bytes(self) -> &'static [u8] {
                match self {
                    $(Icon::$variant => include_bytes!(
                        concat!("../../../assets/icons/", $file, ".svg")
                    ),)*
                }
            }

            /// The vendored file's stem, for the licence table and tests.
            pub const fn file_stem(self) -> &'static str {
                match self {
                    $(Icon::$variant => $file,)*
                }
            }

            /// Every variant, so tests and icon pickers stay exhaustive.
            pub const ALL: &'static [Icon] = &[$(Icon::$variant),*];
        }
    };
}

icons! {
    Select => "mouse-pointer-2",
    DirectSelect => "mouse-pointer-click",
    Pen => "pen-tool",
    Curve => "spline",
    Line => "tabler-line",
    Dart => "triangle",
    Notch => "git-commit-vertical",
    InternalLine => "square-dashed",
    Seam => "tabler-needle-thread",
    Measure => "ruler",
    Pan => "hand",
    ZoomIn => "zoom-in",
    ZoomOut => "zoom-out",
    Mirror => "flip-horizontal-2",
    Point => "circle-dot",
    Snap => "magnet",
    Grain => "arrow-up-down",
    Annotate => "type",

    Box3d => "box",
    Move3d => "move-3d",
    Rotate3d => "rotate-3d",
    Garment => "shirt",

    New => "file-plus",
    Open => "folder-open",
    Save => "save",
    Import => "upload",
    Export => "download",
    Undo => "undo-2",
    Redo => "redo-2",
    Copy => "copy",
    Paste => "clipboard",
    Trash => "trash-2",

    Play => "play",
    Pause => "pause",
    Refresh => "refresh-cw",

    Layers => "layers",
    Grid => "grid-3x3",
    Eye => "eye",
    EyeOff => "eye-off",
    Lock => "lock",
    Unlock => "lock-open",
    Settings => "settings",
    Sliders => "sliders-horizontal",
    Palette => "palette",
    Sun => "sun",
    Moon => "moon",
    PanelLeft => "panel-left",
    PanelRight => "panel-right",
    PanelBottom => "panel-bottom",
    Maximize => "maximize-2",
    Search => "search",
    Scissors => "scissors",

    Plus => "plus",
    Minus => "minus",
    ChevronUp => "chevron-up",
    ChevronDown => "chevron-down",
    ChevronLeft => "chevron-left",
    ChevronRight => "chevron-right",
    Check => "check",
    Close => "x",
    Info => "info",
    Warning => "triangle-alert",
    Crosshair => "crosshair",
}

impl Icon {
    /// An iced SVG handle for this icon.
    ///
    /// `svg::Handle::from_memory(impl Into<Cow<'static, [u8]>>)` — verified in
    /// `iced_core-0.14.0/src/svg.rs:89`.
    pub fn handle(self) -> svg::Handle {
        svg::Handle::from_memory(self.bytes())
    }
}

/// Render an icon at `size` logical px in `tint`.
///
/// Lucide/Tabler glyphs are `stroke="currentColor"`, and iced's
/// `svg::Style { color: Option<Color> }` recolours the whole rendering, so one
/// vendored file serves every semantic colour.
pub fn icon<'a, Message: 'a>(which: Icon, size: f32, tint: Rgba) -> Element<'a, Message> {
    let tint = color(tint);
    svg(which.handle())
        .width(Length::Fixed(size))
        .height(Length::Fixed(size))
        .style(move |_theme, _status| svg::Style { color: Some(tint) })
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ponytail: a real XML parse would mean a new dependency for one test.
    /// This scanner catches what actually goes wrong with vendored assets —
    /// a truncated download or a mismatched tag. Swap in `roxmltree` only if
    /// something slips past it.
    fn parses_as_xml_with_svg_root(src: &str) -> Result<(), String> {
        let body = src.trim();
        let body = match body.find("<svg") {
            // Only a comment, an XML declaration or whitespace may precede the
            // root element; anything else means this is not an SVG document.
            Some(i) => {
                let prefix = body[..i].trim();
                let prefix_ok = prefix.is_empty()
                    || prefix
                        .split_inclusive("-->")
                        .all(|c| c.trim().is_empty() || c.trim_start().starts_with("<!"));
                if !prefix_ok {
                    return Err(format!("junk before <svg root: {prefix:?}"));
                }
                &body[i..]
            }
            None => return Err("no <svg root element".into()),
        };

        if !body.ends_with("</svg>") {
            return Err("document does not end with </svg> (truncated?)".into());
        }

        let mut depth = 0i32;
        let mut rest = body;
        while let Some(open) = rest.find('<') {
            let tail = &rest[open..];
            let close = tail.find('>').ok_or("unterminated tag")?;
            let tag = &tail[1..close];
            if tag.starts_with('!') || tag.starts_with('?') {
                // comment / declaration
            } else if tag.starts_with('/') {
                depth -= 1;
                if depth < 0 {
                    return Err(format!("closing tag with no opener: </{tag}>"));
                }
            } else if !tag.ends_with('/') {
                depth += 1;
            }
            rest = &tail[close + 1..];
        }

        if depth != 0 {
            return Err(format!("{depth} unclosed element(s)"));
        }
        Ok(())
    }

    #[test]
    fn every_icon_variant_resolves_to_a_real_svg() {
        let mut failures = Vec::new();
        for &i in Icon::ALL {
            let bytes = i.bytes();
            let Ok(src) = std::str::from_utf8(bytes) else {
                failures.push(format!("{i:?} ({}) is not UTF-8", i.file_stem()));
                continue;
            };
            if bytes.len() < 64 {
                failures.push(format!(
                    "{i:?} ({}) is only {} bytes — truncated?",
                    i.file_stem(),
                    bytes.len()
                ));
                continue;
            }
            if let Err(e) = parses_as_xml_with_svg_root(src) {
                failures.push(format!("{i:?} ({}): {e}", i.file_stem()));
            }
        }
        assert!(
            failures.is_empty(),
            "bad icons:\n  {}",
            failures.join("\n  ")
        );
    }

    #[test]
    fn the_xml_check_actually_rejects_broken_input() {
        assert!(parses_as_xml_with_svg_root("<svg><path/></svg>").is_ok());
        // truncated mid-download
        assert!(parses_as_xml_with_svg_root("<svg><path d=\"M0").is_err());
        // right root, unbalanced children
        assert!(parses_as_xml_with_svg_root("<svg><g><path/></svg>").is_err());
        // an HTML 404 page saved as .svg
        assert!(parses_as_xml_with_svg_root("<html><body>Not Found</body></html>").is_err());
    }

    #[test]
    fn no_two_variants_share_a_file_and_no_file_is_orphaned() {
        let mut stems: Vec<_> = Icon::ALL.iter().map(|i| i.file_stem()).collect();
        let used = stems.len();
        stems.sort_unstable();
        stems.dedup();
        assert_eq!(used, stems.len(), "two Icon variants map to the same SVG");

        // Every vendored file is referenced: an unused SVG is a licence entry
        // we are carrying for nothing.
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/icons");
        let mut on_disk: Vec<String> = std::fs::read_dir(dir)
            .expect("icons directory")
            .filter_map(|e| {
                let p = e.ok()?.path();
                (p.extension()? == "svg").then(|| p.file_stem()?.to_str().map(String::from))?
            })
            .collect();
        on_disk.sort();
        assert_eq!(
            on_disk,
            stems.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            "vendored SVGs and Icon variants have drifted apart"
        );
    }
}
