// SPDX-License-Identifier: GPL-2.0-or-later
//! Every shipped WGSL file parses and validates, on a machine with no GPU.
//!
//! `create_shader_module` fails at *runtime*, inside a pipeline the app builds
//! during its first frame — which is the failure mode "the 3D pane is black
//! and the log has one line in it". naga is the same front end wgpu uses, so
//! this catches it in CI instead.

use naga::valid::{Capabilities, ValidationFlags, Validator};

const SHADERS: [(&str, &str); 3] = [
    ("cloth.wgsl", include_str!("../src/cloth.wgsl")),
    ("avatar.wgsl", include_str!("../src/avatar.wgsl")),
    ("grid.wgsl", include_str!("../src/grid.wgsl")),
];

fn validate(name: &str, source: &str) -> Result<naga::valid::ModuleInfo, String> {
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|e| format!("{name}: {}", e.emit_to_string(source)))?;
    Validator::new(ValidationFlags::all(), Capabilities::default())
        .validate(&module)
        .map_err(|e| format!("{name}: {e:?}"))
}

#[test]
fn every_shader_parses_and_validates() {
    let mut failures = Vec::new();
    for (name, source) in SHADERS {
        if let Err(e) = validate(name, source) {
            failures.push(e);
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

/// The entry points the pipelines in `renderer.rs` name by string. A rename in
/// the shader is otherwise a runtime panic with a good message and a bad time.
#[test]
fn every_shader_has_the_vs_and_fs_the_pipeline_asks_for() {
    for (name, source) in SHADERS {
        let module = naga::front::wgsl::parse_str(source).expect("validated above");
        let names: Vec<&str> = module
            .entry_points
            .iter()
            .map(|e| e.name.as_str())
            .collect();
        assert!(names.contains(&"vs"), "{name} has no `vs`: {names:?}");
        assert!(names.contains(&"fs"), "{name} has no `fs`: {names:?}");
    }
}

/// The check has to be able to fail. Without this, a naga that silently
/// accepted everything would leave the two tests above passing by doing
/// nothing.
#[test]
fn the_validator_rejects_a_broken_shader() {
    assert!(
        validate(
            "bad",
            "@fragment fn fs() -> @location(0) vec4<f32> { return 1; }"
        )
        .is_err(),
        "a type error validated clean"
    );
    assert!(validate("bad", "this is not wgsl").is_err());
}
