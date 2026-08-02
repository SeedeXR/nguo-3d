// SPDX-License-Identifier: GPL-2.0-or-later
//! Reader for the reference capture — the recorded inputs and per-frame vertex
//! checkpoints of ZOZO's CUDA solver on an A100, used to check this port for
//! correctness against the reference.
//!
//! The capture itself is an internal benchmark corpus and is **not distributed
//! with this source** (see `docs/GAUGE_FINDINGS.md`); [`gauge_dir`] returns
//! `None` in a clean clone and every caller skips loudly. This module and its
//! parsing tests are still built and still run.
//!
//! Binary layout — **little-endian, empirically verified against the capture**
//! (`goldens/<scene>/`):
//! - `input_bin/vert.bin`      : `f64` xyz per vertex — the REST/template positions
//!   (e.g. drape's five stacked sheets are all flat at y=0 here). NOT the initial state
//!   on their own: each vertex's displacement-group offset must be added on top.
//!   `vert + displacement[vert_dmap[i]]` reproduces `checkpoints/vert_0.bin` to f32
//!   precision for ALL 22 gauge scenes (the sheets separate in y), while raw `vert.bin`
//!   is off by up to 13.0 world units (ribbon) in 18 of them. [`load_golden_input`]
//!   applies it; `loaded_positions_match_the_reference_frame_zero` proves it.
//! - `input_bin/vert_dmap.bin` : `u32` per vertex — displacement group id.
//! - `input_bin/displacement.bin` : `f64` xyz per group — the offset table.
//! - `input_bin/tri.bin`       : `u64` xyz per triangle — vertex indices.
//! - `input_bin/vel.bin`       : `f32` xyz per vertex — initial velocities.
//! - `checkpoints/vert_N.bin`  : `f32` xyz per vertex — reference positions at
//!   output frame `N`; frame 0 is the solver's initial state.
//!
//! This is the format reader only — the run-and-compare driver (drive our solver
//! with the input + params, measure divergence from the checkpoints) builds on it.
//! Note the fidelity caveat: our solver is a mass-spring/membrane tier, not the
//! reference's full IPC solver, so a driver measures divergence, not bit-parity.

use anyhow::{bail, Context, Result};
use std::path::Path;

/// A reference scene's raw inputs, decoded into engine-native `f32` types.
#[derive(Clone, Debug)]
pub struct GoldenInput {
    /// Initial positions (from `vert.bin`, `f64`→`f32`).
    pub positions: Vec<[f32; 3]>,
    /// Triangles (from `tri.bin`, `u64`→`u32`).
    pub tris: Vec<[u32; 3]>,
    /// Initial velocities (from `vel.bin`), parallel to `positions`.
    pub velocities: Vec<[f32; 3]>,
    /// Tetrahedra (from `tet.bin`, `u64`→`u32`). Empty for pure cloth/shell scenes;
    /// the reference calls these "Solids" in `param_summary.txt`.
    pub tets: Vec<[u32; 4]>,
    /// Rod segments (from `rod.bin`, `u64`→`u32`) — the reference's "Rods". 1-D
    /// codimensional strands: yarn, noodles, woven threads. Empty for most scenes.
    pub rods: Vec<[u32; 2]>,
}

/// Decode a flat little-endian `f64` xyz buffer (24 bytes/vertex) to `[f32; 3]`.
pub fn decode_vec3_f64(bytes: &[u8]) -> Result<Vec<[f32; 3]>> {
    if !bytes.len().is_multiple_of(24) {
        bail!(
            "f64 vec3 buffer length {} not a multiple of 24",
            bytes.len()
        );
    }
    Ok(bytes
        .chunks_exact(24)
        .map(|c| {
            let g = |o: usize| f64::from_le_bytes(c[o..o + 8].try_into().unwrap()) as f32;
            [g(0), g(8), g(16)]
        })
        .collect())
}

/// Decode a flat little-endian `f32` xyz buffer (12 bytes/vertex) to `[f32; 3]`.
pub fn decode_vec3_f32(bytes: &[u8]) -> Result<Vec<[f32; 3]>> {
    if !bytes.len().is_multiple_of(12) {
        bail!(
            "f32 vec3 buffer length {} not a multiple of 12",
            bytes.len()
        );
    }
    Ok(bytes
        .chunks_exact(12)
        .map(|c| {
            let g = |o: usize| f32::from_le_bytes(c[o..o + 4].try_into().unwrap());
            [g(0), g(4), g(8)]
        })
        .collect())
}

/// Decode a flat little-endian `u64` xyz triangle buffer (24 bytes/triangle) to
/// `[u32; 3]`. Indices exceeding `u32::MAX` are an error (they never occur in the
/// capture, but this guards a silent truncation).
pub fn decode_tris_u64(bytes: &[u8]) -> Result<Vec<[u32; 3]>> {
    if !bytes.len().is_multiple_of(24) {
        bail!("u64 tri buffer length {} not a multiple of 24", bytes.len());
    }
    bytes
        .chunks_exact(24)
        .map(|c| {
            let mut out = [0u32; 3];
            for (k, slot) in out.iter_mut().enumerate() {
                let v = u64::from_le_bytes(c[k * 8..k * 8 + 8].try_into().unwrap());
                if v > u32::MAX as u64 {
                    bail!("triangle index {v} exceeds u32::MAX");
                }
                *slot = v as u32;
            }
            Ok(out)
        })
        .collect()
}

/// Load `input_bin/{vert,tri,vel}.bin` for a golden scene directory.
/// Decode a flat little-endian `u64` tet-index buffer (32 bytes/tet) to `[u32; 4]`.
/// Verified against the gauge: `plastic`'s 405,440-byte `tet.bin` yields 12,670
/// tets, matching its `param_summary.txt` "Solids (12670 elements)".
pub fn decode_tets_u64(bytes: &[u8]) -> Result<Vec<[u32; 4]>> {
    if !bytes.len().is_multiple_of(32) {
        bail!("u64 tet buffer length {} not a multiple of 32", bytes.len());
    }
    bytes
        .chunks_exact(32)
        .map(|c| {
            let mut out = [0u32; 4];
            for (k, slot) in out.iter_mut().enumerate() {
                *slot = narrow_index(c, k, "tet")?;
            }
            Ok(out)
        })
        .collect()
}

/// Read index `k` of a chunk as `u64` and narrow to `u32`, refusing to truncate.
///
/// A bare `as u32` here is a silent-corruption bug, not a lint: the loader's
/// `i >= nv` bounds check runs AFTER narrowing, so a hostile or corrupt word like
/// `0x1_0000_0005` truncates to vertex `5`, passes the bounds check, and the solver
/// then builds elements over the wrong vertices with no error. `decode_tris_u64`
/// already guarded this; tets and rods did not, and this is the shared guard.
fn narrow_index(chunk: &[u8], k: usize, what: &str) -> Result<u32> {
    let v = u64::from_le_bytes(chunk[k * 8..k * 8 + 8].try_into().unwrap());
    if v > u32::MAX as u64 {
        bail!("{what} index {v} exceeds u32::MAX");
    }
    Ok(v as u32)
}

/// Decode a flat little-endian `u64` rod-index buffer (16 bytes/rod) to `[u32; 2]`.
/// Verified against the gauge: `codim`'s 21,504-byte `rod.bin` yields 1344 segments,
/// matching its `param_summary.txt` "Rods (1344 elements)"; likewise `noodle` 116160,
/// `woven` 115440, `yarn` 28507.
pub fn decode_rods_u64(bytes: &[u8]) -> Result<Vec<[u32; 2]>> {
    if !bytes.len().is_multiple_of(16) {
        bail!("u64 rod buffer length {} not a multiple of 16", bytes.len());
    }
    bytes
        .chunks_exact(16)
        .map(|c| Ok([narrow_index(c, 0, "rod")?, narrow_index(c, 1, "rod")?]))
        .collect()
}

/// The reference's per-element constitutive model, from `data.hpp`:
/// `enum class Model { ARAP, StVK, BaraffWitkin, SNHk, Pdrd }` — so the on-disk byte
/// is a plain discriminant. Confirmed two ways: the enum declaration in the reference
/// source, AND a byte histogram of `plastic`'s `tri-model.bin` (5120 zeros + 3844
/// twos) matching its `param_summary.txt` line
/// `model: (arap: 5120 elements, baraffwitkin: 3844 elements)` exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefModel {
    Arap,
    StVk,
    BaraffWitkin,
    SnHk,
    Pdrd,
    /// A discriminant this build does not know about, preserved rather than guessed.
    Unknown(u8),
}

impl From<u8> for RefModel {
    fn from(b: u8) -> Self {
        match b {
            0 => RefModel::Arap,
            1 => RefModel::StVk,
            2 => RefModel::BaraffWitkin,
            3 => RefModel::SnHk,
            4 => RefModel::Pdrd,
            other => RefModel::Unknown(other),
        }
    }
}

/// Per-element parameter arrays a scene assigns individually (the reference's
/// `param_arrays`). Each `Vec` is parallel to its element list — one entry per
/// triangle or per tet. Missing files yield empty vectors rather than an error, since
/// which params a scene ships varies.
#[derive(Clone, Debug, Default)]
pub struct ElementParams {
    /// Constitutive model per element (`*-model.bin`, 1 byte each).
    pub model: Vec<RefModel>,
    /// Creep rate and dead-zone threshold (`*-plasticity{,-threshold}.bin`, f32).
    pub plasticity: Vec<f32>,
    pub plasticity_threshold: Vec<f32>,
    /// Young's modulus and Poisson ratio (f32).
    pub young_mod: Vec<f32>,
    pub poiss_rat: Vec<f32>,
    /// Coulomb friction coefficient per element (`*-friction.bin`).
    pub friction: Vec<f32>,
    /// Allowed extra stretch fraction per element (`*-strain-limit.bin`). The
    /// reference's value is a FRACTION: `0.05` means a 1.05 stretch ceiling, which is
    /// how its `INVARIANTS` reports it ("limit 0.05 (ceiling 1.0500)"). 0 = disabled.
    pub strain_limit: Vec<f32>,
    /// Contact activation distance per element (`*-contact-gap.bin`).
    pub contact_gap: Vec<f32>,
    /// Mass density per element (`*-density.bin`). The reference turns this into element
    /// mass as `density * area` for faces and `density * volume` for tets
    /// (`ppf-cts-solver/src/scene.rs:1665,1759`), then lumps it to vertices.
    pub density: Vec<f32>,
    /// Bending stiffness per face (`*-bend.bin`); absent for the `tet` prefix.
    pub bend: Vec<f32>,
    /// Bend rest angle taken from the initial geometry rather than the straight default π
    /// (`rod-bend-rest-from-geometry.bin`); rods only. 0 = use π. No gauge scene sets it.
    pub bend_rest_from_geometry: Vec<f32>,
    /// Lagged bending damping per rod segment (`rod-bending-damping.bin`); rods only.
    /// Every gauge scene ships 0.
    pub bending_damping: Vec<f32>,
    /// Rest-length scale per rod segment (`rod-length-factor.bin`); rods only. The
    /// reference computes rod mass from the GEOMETRIC length and only then scales the
    /// rest length (`ppf-cts-solver/src/scene.rs:1498-1499`), so a factor below 1
    /// pre-tensions the strand: `yarn` ships 0.85, `woven` 0.8.
    pub length_factor: Vec<f32>,
    /// Features we do not model; decoded ONLY so the driver can tell whether a scene
    /// actually activates them (shrink is identity at 1.0, the others at 0.0).
    pub contact_offset: Vec<f32>,
    pub shrink_x: Vec<f32>,
    pub shrink_y: Vec<f32>,
    pub pressure: Vec<f32>,
    pub deformation_damping: Vec<f32>,
}

fn read_f32_array(path: &Path) -> Result<Vec<f32>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let raw = read(path)?;
    if !raw.len().is_multiple_of(4) {
        bail!(
            "f32 param {} length {} not a multiple of 4",
            path.display(),
            raw.len()
        );
    }
    Ok(raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

/// Read the per-element parameter arrays for one prefix (`"tri"` or `"tet"`) from a
/// scene's `input_bin/param/` directory. `bend` selects the bend-specific plasticity
/// files (`tri-bend-plasticity*`), which the reference keeps separate from the
/// in-plane ones because hinge creep acts on the rest ANGLE.
pub fn load_element_params(
    scene_dir: impl AsRef<Path>,
    prefix: &str,
    bend: bool,
) -> Result<ElementParams> {
    let dir = scene_dir.as_ref().join("input_bin/param");
    let model_path = dir.join(format!("{prefix}-model.bin"));
    let model = if model_path.exists() {
        read(&model_path)?.into_iter().map(RefModel::from).collect()
    } else {
        Vec::new()
    };
    let (pl, plt) = if bend {
        ("bend-plasticity", "bend-plasticity-threshold")
    } else {
        ("plasticity", "plasticity-threshold")
    };
    Ok(ElementParams {
        model,
        plasticity: read_f32_array(&dir.join(format!("{prefix}-{pl}.bin")))?,
        plasticity_threshold: read_f32_array(&dir.join(format!("{prefix}-{plt}.bin")))?,
        young_mod: read_f32_array(&dir.join(format!("{prefix}-young-mod.bin")))?,
        poiss_rat: read_f32_array(&dir.join(format!("{prefix}-poiss-rat.bin")))?,
        friction: read_f32_array(&dir.join(format!("{prefix}-friction.bin")))?,
        strain_limit: read_f32_array(&dir.join(format!("{prefix}-strain-limit.bin")))?,
        contact_gap: read_f32_array(&dir.join(format!("{prefix}-contact-gap.bin")))?,
        density: read_f32_array(&dir.join(format!("{prefix}-density.bin")))?,
        bend: read_f32_array(&dir.join(format!("{prefix}-bend.bin")))?,
        bend_rest_from_geometry: read_f32_array(
            &dir.join(format!("{prefix}-bend-rest-from-geometry.bin")),
        )?,
        bending_damping: read_f32_array(&dir.join(format!("{prefix}-bending-damping.bin")))?,
        length_factor: read_f32_array(&dir.join(format!("{prefix}-length-factor.bin")))?,
        contact_offset: read_f32_array(&dir.join(format!("{prefix}-contact-offset.bin")))?,
        shrink_x: read_f32_array(&dir.join(format!("{prefix}-shrink-x.bin")))?,
        shrink_y: read_f32_array(&dir.join(format!("{prefix}-shrink-y.bin")))?,
        pressure: read_f32_array(&dir.join(format!("{prefix}-pressure.bin")))?,
        deformation_damping: read_f32_array(
            &dir.join(format!("{prefix}-deformation-damping.bin")),
        )?,
    })
}

/// Split element indices by constitutive model — the query the solver needs in order
/// to drive disjoint face sets with different models (the gauge's shell scenes mix
/// `arap` and `baraffwitkin` per face).
pub fn indices_with_model(models: &[RefModel], want: RefModel) -> Vec<u32> {
    models
        .iter()
        .enumerate()
        .filter(|(_, m)| **m == want)
        .map(|(i, _)| i as u32)
        .collect()
}

pub fn load_golden_input(scene_dir: impl AsRef<Path>) -> Result<GoldenInput> {
    let ib = scene_dir.as_ref().join("input_bin");
    // `vert.bin` holds the REST/TEMPLATE positions, which are NOT the initial state: the
    // scene's per-vertex displacement group must be added on top. Verified against every
    // gauge scene — `vert + displacement[vert_dmap[i]]` reproduces `checkpoints/vert_0.bin`
    // to f32 precision for all 22, while raw `vert.bin` is off by up to 13.0 world units
    // (ribbon) in 18 of them. This is what the module doc's "drape's five stacked sheets
    // are all flat at y=0 here" note was describing.
    let mut positions = decode_vec3_f64(&read(&ib.join("vert.bin"))?)?;
    apply_displacement(&ib, &mut positions)?;
    // Rod-only scenes (noodle, woven, yarn) ship no `tri.bin` at all.
    let tri_path = ib.join("tri.bin");
    let tris = if tri_path.exists() {
        decode_tris_u64(&read(&tri_path)?)?
    } else {
        Vec::new()
    };
    let velocities = decode_vec3_f32(&read(&ib.join("vel.bin"))?)?;
    if velocities.len() != positions.len() {
        bail!(
            "vel count {} != vert count {}",
            velocities.len(),
            positions.len()
        );
    }
    let nv = positions.len() as u32;
    for t in &tris {
        if t[0] >= nv || t[1] >= nv || t[2] >= nv {
            bail!("triangle {t:?} references a vertex >= {nv}");
        }
    }
    // Volumetric scenes carry tet.bin; cloth-only scenes may omit it or ship it empty.
    let tet_path = ib.join("tet.bin");
    let tets = if tet_path.exists() {
        let raw = read(&tet_path)?;
        if raw.is_empty() {
            Vec::new()
        } else {
            decode_tets_u64(&raw)?
        }
    } else {
        Vec::new()
    };
    for t in &tets {
        if t.iter().any(|&i| i >= nv) {
            bail!("tet {t:?} references a vertex >= {nv}");
        }
    }
    let rod_path = ib.join("rod.bin");
    let rods = if rod_path.exists() {
        decode_rods_u64(&read(&rod_path)?)?
    } else {
        Vec::new()
    };
    for r in &rods {
        if r.iter().any(|&i| i >= nv) {
            bail!("rod {r:?} references a vertex >= {nv}");
        }
    }
    Ok(GoldenInput {
        positions,
        tris,
        velocities,
        tets,
        rods,
    })
}

/// `param.toml: friction_mode` — how two participants' friction coefficients combine at a
/// contact. Mirrors the reference's `FrictionMode` (`contact/contact.cu:37-47`). Declared here
/// rather than reusing `engine_core`'s so `engine-io` keeps no engine-core dependency; the
/// driver maps between them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GoldenFrictionMode {
    #[default]
    Min,
    Max,
    Mean,
}

/// The subset of `param.toml` a run driver needs to schedule steps. Fields our
/// solver doesn't model (barrier type, ACCD/CCD, friction modes, plasticity, …) are
/// intentionally ignored here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GoldenParams {
    pub dt: f32,
    pub fps: f32,
    pub frames: u32,
    pub gravity: [f32; 3],
    pub world_scaling: f32,
    /// Whether a SOLID's surface faces contribute vertex mass (`builder.rs:485`).
    /// Default `false`, matching the reference (`params.rs:278`).
    pub include_face_mass: bool,
    pub air_density: f32,
    pub air_friction: f32,
    /// Direction-independent velocity damping (`isotropic_air_friction`).
    pub isotropic_air_friction: f32,
    /// Constant wind velocity (`wind`). Zero in every gauge scene.
    pub wind: [f32; 3],
    /// Degenerate-eigenvalue guard for the isotropic eigensystems (`eiganalysis_eps`).
    pub eiganalysis_eps: f32,
    /// Linear-solve tolerance and cap (`cg_tol`, `cg_max_iter`).
    pub cg_tol: f32,
    pub cg_max_iter: u32,
    /// Conservative-advancement iteration cap (`ccd_max_iter`).
    pub ccd_max_iter: u32,
    /// Minimum Newton iterations per step (`min_newton_steps`).
    pub min_newton_steps: u32,
    /// Newton termination target (`target_toi`).
    pub target_toi: f32,
    /// Search-direction magnitude cap in metres (`max_dx`).
    pub max_dx: f32,
    /// Friction regularization floor (`friction_eps`).
    pub friction_eps: f32,
    /// Contact barrier form (`barrier`): `"cubic"` in every gauge scene.
    pub cubic_barrier: bool,
    /// Whether contact is switched off entirely (`disable_contact`); false everywhere.
    pub disable_contact: bool,
    /// How two participants' friction coefficients combine (`param.toml: friction_mode`).
    /// **14 of the 22 gauge scenes ship `Max`** and only 8 ship `Min`.
    pub friction_mode: GoldenFrictionMode,
    /// Multiplier on a stitch's rest length before its pull saturates
    /// (`param.toml: stitch_length_factor`). Mandatory in the reference's args (no serde
    /// default); every gauge scene ships 10.0.
    pub stitch_length_factor: f32,
    /// Pin confinement radius (`param.toml: constraint_ghat`, reference default 1e-3).
    pub constraint_ghat: f32,
    /// Gap floor as a fraction of `constraint_ghat` (`param.toml: constraint_tol`).
    pub constraint_tol: f32,
}

/// Parse the fields of interest from a scene's `param.toml`. This is a minimal
/// flat `key = value` reader for the reference's `[param]` table (no `toml` dep):
/// scalars and the `[x, y, z]` gravity vector. Missing fields take sensible
/// defaults (dt 1e-3, fps 60, gravity −Y, world_scaling 1).
pub fn read_param_toml(scene_dir: impl AsRef<Path>) -> Result<GoldenParams> {
    let text = std::fs::read_to_string(scene_dir.as_ref().join("param.toml"))
        .with_context(|| "reading param.toml")?;
    let mut p = GoldenParams {
        dt: 1.0e-3,
        fps: 60.0,
        frames: 0,
        gravity: [0.0, -9.8, 0.0],
        world_scaling: 1.0,
        include_face_mass: false,
        air_density: 1.0e-3,
        air_friction: 0.0,
        isotropic_air_friction: 0.0,
        wind: [0.0; 3],
        eiganalysis_eps: 1.0e-2,
        cg_tol: 1.0e-3,
        cg_max_iter: 10_000,
        ccd_max_iter: 4096,
        min_newton_steps: 0,
        target_toi: 0.25,
        max_dx: 1.0,
        friction_eps: 1.0e-5,
        cubic_barrier: false,
        disable_contact: false,
        friction_mode: GoldenFrictionMode::Min,
        stitch_length_factor: 10.0,
        constraint_ghat: 1.0e-3,
        constraint_tol: 1.0e-2,
    };
    for line in text.lines() {
        let line = line.trim();
        let Some((key, val)) = line.split_once('=') else {
            continue;
        };
        let (key, val) = (key.trim(), val.trim());
        match key {
            "dt" => p.dt = val.parse().unwrap_or(p.dt),
            "fps" => p.fps = val.parse().unwrap_or(p.fps),
            "frames" => p.frames = val.parse().unwrap_or(p.frames),
            "world_scaling" => p.world_scaling = val.parse().unwrap_or(p.world_scaling),
            "include_face_mass" => p.include_face_mass = val.trim() == "true",
            "air_density" => p.air_density = val.parse().unwrap_or(p.air_density),
            "air_friction" => p.air_friction = val.parse().unwrap_or(p.air_friction),
            "eiganalysis_eps" => p.eiganalysis_eps = val.parse().unwrap_or(p.eiganalysis_eps),
            "target_toi" => p.target_toi = val.parse().unwrap_or(p.target_toi),
            "max_dx" => p.max_dx = val.parse().unwrap_or(p.max_dx),
            "friction_eps" => p.friction_eps = val.parse().unwrap_or(p.friction_eps),
            "cg_tol" => p.cg_tol = val.parse().unwrap_or(p.cg_tol),
            "cg_max_iter" => p.cg_max_iter = val.parse().unwrap_or(p.cg_max_iter),
            "ccd_max_iter" => p.ccd_max_iter = val.parse().unwrap_or(p.ccd_max_iter),
            "min_newton_steps" => p.min_newton_steps = val.parse().unwrap_or(p.min_newton_steps),
            "barrier" => p.cubic_barrier = val.trim().trim_matches('"') == "cubic",
            "disable_contact" => p.disable_contact = val.trim() == "true",
            "isotropic_air_friction" => {
                p.isotropic_air_friction = val.parse().unwrap_or(p.isotropic_air_friction)
            }
            "wind" => {
                let nums: Vec<f32> = val
                    .trim_matches(['[', ']'])
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
                if nums.len() == 3 {
                    p.wind = [nums[0], nums[1], nums[2]];
                }
            }
            "friction_mode" => {
                p.friction_mode = match val.trim().trim_matches('"') {
                    "max" => GoldenFrictionMode::Max,
                    "mean" => GoldenFrictionMode::Mean,
                    // The reference's `combine_friction` falls through to Min by default
                    // (`contact/contact.cu:43-45`), so an unknown spelling does too.
                    _ => GoldenFrictionMode::Min,
                }
            }
            "stitch_length_factor" => {
                p.stitch_length_factor = val.parse().unwrap_or(p.stitch_length_factor)
            }
            "constraint_ghat" => p.constraint_ghat = val.parse().unwrap_or(p.constraint_ghat),
            "constraint_tol" => p.constraint_tol = val.parse().unwrap_or(p.constraint_tol),
            "gravity" => {
                let nums: Vec<f32> = val
                    .trim_matches(['[', ']'])
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
                if nums.len() == 3 {
                    p.gravity = [nums[0], nums[1], nums[2]];
                }
            }
            _ => {}
        }
    }
    Ok(p)
}

/// Read the constrained-vertex indices from `input_bin/pin-ind-*.bin`. Each file is a
/// flat little-endian `u64` LIST of the vertex indices in pin block `i`
/// (`PORTING_data_layout.md:657`: `u64`, shape `[P_i]`, length asserted equal to the
/// scene's own `count["pin-<i>"]["pin"]`); this returns the concatenation of all blocks.
///
/// An earlier version of this reader read the two `u64` in a 16-byte file as an
/// inclusive RANGE `[a, b]`. That was wrong, and wrong in a way that inflated results:
/// `hang` ships `(16256, 16383)`, which the range reading expands to 128 vertices while
/// the scene's `info.toml` declares `[pin-0] pin = 2`. Verified against the declared
/// counts: hang 16 B = 2, plastic 8192 B = 1024, belt 11440 B = 1430.
///
/// IMPORTANT — semantics caveat: these indices are the vertices the reference
/// *constrains*, but the reference's `FixPair` (PORTING_data_layout.md) is a SOFT
/// constraint with a `stiffness` and an optional `kinematic` (animated) target
/// `position` — NOT a hard fixed pin. Empirically (hang), the "pinned" vertices
/// still move ~0.05 across the run. A faithful run driver must model that, not
/// treat these as immovable — hence the run step is staged behind understanding the
/// FixPair semantics. This reader only reports *which* vertices are constrained.
pub fn read_pin_indices(scene_dir: impl AsRef<Path>) -> Result<Vec<u32>> {
    let ib = scene_dir.as_ref().join("input_bin");
    let mut out = Vec::new();
    let mut n = 0;
    while let Ok(bytes) = std::fs::read(ib.join(format!("pin-ind-{n}.bin"))) {
        if !bytes.len().is_multiple_of(8) {
            bail!(
                "pin-ind-{n}.bin is {} bytes, not a whole number of u64",
                bytes.len()
            );
        }
        for c in bytes.chunks_exact(8) {
            let i = u64::from_le_bytes(c.try_into().unwrap());
            if i > u32::MAX as u64 {
                bail!("pin index {i} exceeds u32::MAX");
            }
            out.push(i as u32);
        }
        n += 1;
    }
    Ok(out)
}

/// Add each vertex's displacement-group offset in place (`displacement.bin` indexed by
/// `vert_dmap.bin`, `PORTING_data_layout.md:649-650`). A scene without both files is left
/// untouched — for those, `vert.bin` already IS the initial state (verified: hang, twist,
/// woven, yarn match `checkpoints/vert_0.bin` exactly with no displacement applied).
///
/// The reference scales displacement by `world_scaling` (`scene.rs:475-477`); every gauge
/// scene ships `world_scaling = 1.0`, so that factor is unobservable here and is NOT
/// applied — a scene with `ws != 1` would need it threaded in from `param.toml`.
///
/// Why this is safe to fold into the positions used as BOTH the initial and the rest
/// state: no element straddles two displacement groups in any gauge scene (checked for
/// every triangle and tet — zero straddling elements), so per-group displacement acts as a
/// rigid translation on each element and leaves every rest shape (`Dm`, edge lengths, rest
/// angles) bit-identical. Were an element to span two groups, the template and the
/// displaced state would disagree and the rest shape would have to come from the template.
fn apply_displacement(input_bin: &Path, positions: &mut [[f32; 3]]) -> Result<()> {
    let (dmap_p, disp_p) = (
        input_bin.join("vert_dmap.bin"),
        input_bin.join("displacement.bin"),
    );
    if !dmap_p.exists() || !disp_p.exists() {
        return Ok(());
    }
    let disp = decode_vec3_f64(&std::fs::read(&disp_p)?)?;
    let raw = std::fs::read(&dmap_p)?;
    if !raw.len().is_multiple_of(4) {
        bail!(
            "vert_dmap.bin is {} bytes, not a whole number of u32",
            raw.len()
        );
    }
    if raw.len() / 4 != positions.len() {
        bail!(
            "vert_dmap has {} entries but there are {} vertices",
            raw.len() / 4,
            positions.len()
        );
    }
    for (p, c) in positions.iter_mut().zip(raw.chunks_exact(4)) {
        let g = u32::from_le_bytes(c.try_into().unwrap()) as usize;
        let d = disp
            .get(g)
            .with_context(|| format!("vert_dmap group {g} exceeds the displacement table"))?;
        for k in 0..3 {
            p[k] += d[k];
        }
    }
    Ok(())
}

/// A scene's analytic collider, decoded from `info.toml` + its `*-pos-/-radius-/-timing-`
/// binaries. Geometry-only and engine-agnostic (engine-io does not depend on engine-core),
/// so the caller converts to its own collider type.
///
/// Reference semantics (`data.hpp:355-373`, applied `contact.cu:1080-1200`):
/// * A **wall** is the reference's `Floor`: a half-space through `position` with outward
///   normal `normal` (its `up`). Contact activates within `ghat` of that plane.
/// * A **sphere** has `position` as its centre and an animatable `radius`.
/// * `thickness` is a PASS-THROUGH depth, not a shell: the reference SKIPS the contact once
///   penetration exceeds it (`contact.cu:1171`). Every gauge scene ships `1.0`, far beyond
///   any real penetration, so it never fires — decoded and reported rather than modelled.
/// * `active_duration < 0` means always active (every gauge scene).
#[derive(Clone, Debug, PartialEq)]
pub struct SceneCollider {
    /// `None` ⇒ sphere; `Some(n)` ⇒ wall/half-space with outward normal `n`.
    pub normal: Option<[f32; 3]>,
    /// Keyframe times (`*-timing-<i>.bin`, f64). Length matches `positions`.
    pub times: Vec<f32>,
    /// Keyframe positions (`*-pos-<i>.bin`, f64 xyz): plane point, or sphere centre.
    pub positions: Vec<[f32; 3]>,
    /// Keyframe radii (`sphere-radius-<i>.bin`, f32). Empty for a wall.
    pub radii: Vec<f32>,
    /// Contact activation distance for THIS collider (`contact-gap`).
    pub ghat: f32,
    /// Per-collider Coulomb friction, combined with the vertex's by the reference.
    pub friction: f32,
    /// Pass-through penetration depth (see the type docs).
    pub thickness: f32,
    /// Negative ⇒ always active.
    pub active_duration: f32,
}

impl SceneCollider {
    /// Is this a wall (half-space) rather than a sphere?
    pub fn is_wall(&self) -> bool {
        self.normal.is_some()
    }

    /// Does this collider move or resize over the run?
    pub fn is_animated(&self) -> bool {
        self.positions.len() > 1 || self.radii.len() > 1
    }

    /// Position and radius at `time`, linearly interpolated between keyframes and clamped
    /// at both ends (`transition = "linear"` in every gauge scene). A single keyframe is
    /// constant, which is how all 6 wall scenes and 2 of the 3 sphere scenes are authored.
    pub fn sample(&self, time: f32) -> ([f32; 3], f32) {
        let lerp = |a: f32, b: f32, u: f32| a + (b - a) * u;
        let pos = |i: usize| self.positions[i.min(self.positions.len().saturating_sub(1))];
        let rad = |i: usize| {
            self.radii
                .get(i.min(self.radii.len().saturating_sub(1)))
                .copied()
                .unwrap_or(0.0)
        };
        if self.positions.is_empty() {
            return ([0.0; 3], 0.0);
        }
        if self.times.len() < 2 {
            return (pos(0), rad(0));
        }
        // Clamp outside the keyframe range rather than extrapolating.
        if time <= self.times[0] {
            return (pos(0), rad(0));
        }
        let last = self.times.len() - 1;
        if time >= self.times[last] {
            return (pos(last), rad(last));
        }
        let k = self
            .times
            .windows(2)
            .position(|w| time >= w[0] && time <= w[1])
            .unwrap_or(0);
        let (t0, t1) = (self.times[k], self.times[k + 1]);
        let u = if (t1 - t0).abs() > 0.0 {
            (time - t0) / (t1 - t0)
        } else {
            0.0
        };
        let (p0, p1) = (pos(k), pos(k + 1));
        (
            [
                lerp(p0[0], p1[0], u),
                lerp(p0[1], p1[1], u),
                lerp(p0[2], p1[2], u),
            ],
            lerp(rad(k), rad(k + 1), u),
        )
    }
}

/// Read every analytic collider a scene ships. Walls first (`[wall-<i>]`), then spheres
/// (`[sphere-<i>]`), each in declaration order.
pub fn read_colliders(scene_dir: impl AsRef<Path>) -> Result<Vec<SceneCollider>> {
    let dir = scene_dir.as_ref();
    let ib = dir.join("input_bin");
    let info = std::fs::read_to_string(dir.join("info.toml")).unwrap_or_default();
    let mut out = Vec::new();
    for kind in ["wall", "sphere"] {
        for i in 0..64usize {
            let pos_p = ib.join(format!("{kind}-pos-{i}.bin"));
            if !pos_p.exists() {
                break;
            }
            let sec = toml_section(&info, &format!("{kind}-{i}"));
            let num = |k: &str, d: f32| sec.get(k).and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
            let positions = decode_vec3_f64(&std::fs::read(&pos_p)?)?;
            let times: Vec<f32> = {
                let p = ib.join(format!("{kind}-timing-{i}.bin"));
                if p.exists() {
                    let b = std::fs::read(&p)?;
                    if !b.len().is_multiple_of(8) {
                        bail!("{kind}-timing-{i}.bin is not a whole number of f64");
                    }
                    b.chunks_exact(8)
                        .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                        .collect()
                } else {
                    Vec::new()
                }
            };
            let radii: Vec<f32> = {
                let p = ib.join(format!("{kind}-radius-{i}.bin"));
                if p.exists() {
                    let b = std::fs::read(&p)?;
                    if !b.len().is_multiple_of(4) {
                        bail!("{kind}-radius-{i}.bin is not a whole number of f32");
                    }
                    b.chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect()
                } else {
                    Vec::new()
                }
            };
            let normal = if kind == "wall" {
                Some([num("nx", 0.0), num("ny", 1.0), num("nz", 0.0)])
            } else {
                None
            };
            out.push(SceneCollider {
                normal,
                times,
                positions,
                radii,
                ghat: num("contact-gap", 1.0e-3),
                friction: num("friction", 0.0),
                thickness: num("thickness", 1.0),
                active_duration: num("active-duration", -1.0),
            });
        }
    }
    Ok(out)
}

/// Count the scene's ANALYTIC COLLIDER inputs: `(walls, spheres)`.
///
/// A wall ships `wall-pos-<i>.bin` (+ `wall-timing-<i>.bin` when keyframed) and a sphere
/// `sphere-pos-<i>.bin` / `-radius-` / `-timing-`. These are half-space and sphere
/// constraints the cloth must not penetrate, and they are frequently the DRIVING geometry
/// of a scene, not scenery — `domino` is a solid pushed onto a wall.
///
/// Exists so a caller can report them: a driver that ignores a scene's colliders while
/// reporting no gaps is claiming a fidelity it does not have. Nine of the 22 gauge scenes
/// ship them (cards, codim, domino, noodle, ribbon, roller, stack, trampoline, trapped).
pub fn count_collider_inputs(scene_dir: impl AsRef<Path>) -> (usize, usize) {
    let ib = scene_dir.as_ref().join("input_bin");
    let count = |prefix: &str| {
        (0..64)
            .take_while(|i| ib.join(format!("{prefix}-pos-{i}.bin")).exists())
            .count()
    };
    (count("wall"), count("sphere"))
}

/// One stitch (the reference's `Stitch`, `constraint.stitch`): a weighted barycentric
/// point on one piece of cloth sewn to a weighted barycentric point on another.
///
/// `index[0..3]` / `weight[0..3]` is the SOURCE barycentric, `index[3..6]` / `weight[3..6]`
/// the TARGET; each side's weights sum to 1. A single-vertex source degenerates to
/// `index = {s, s, s}`, `weight = {1, 0, 0}` — which is what all 851 of `fitting`'s
/// stitches do (`energy.cu:704-708`).
///
/// On disk: `stitch_ind.bin` is `u64[S][6]` (48 B/stitch — width 6 is an ABI contract
/// asserted producer-side at `_scene_.py:989-995`), `stitch_w.bin` is `f32[S][6]` (24 B),
/// and `stitch_stiffness.bin` is an OPTIONAL `f32[S]` defaulting to 1.0
/// (`scene.rs:616-617`). All three agree on `S` — verified on `fitting`: 851 each.
#[derive(Clone, Copy, Debug)]
pub struct SceneStitch {
    pub index: [u32; 6],
    pub weight: [f32; 6],
    pub stiffness: f32,
}

impl SceneStitch {
    /// True when the source is a single vertex (`index[0..3]` all equal).
    pub fn source_is_single_vertex(&self) -> bool {
        self.index[0] == self.index[1] && self.index[1] == self.index[2]
    }
}

/// Read `input_bin/stitch_{ind,w,stiffness}.bin`. Returns an empty vector when the scene
/// ships no stitches (all but `fitting`).
pub fn read_stitches(scene_dir: impl AsRef<Path>) -> Result<Vec<SceneStitch>> {
    let ib = scene_dir.as_ref().join("input_bin");
    let ind_path = ib.join("stitch_ind.bin");
    if !ind_path.exists() {
        return Ok(Vec::new());
    }
    let ind = read(&ind_path)?;
    if !ind.len().is_multiple_of(48) {
        bail!("stitch_ind length {} not a multiple of 48", ind.len());
    }
    let n = ind.len() / 48;
    let w = read(&ib.join("stitch_w.bin"))?;
    if w.len() / 24 != n {
        bail!(
            "stitch_w implies {} stitches but stitch_ind implies {n}",
            w.len() / 24
        );
    }
    // Optional; missing ⇒ 1.0 per stitch.
    let stiff = read_f32_array(&ib.join("stitch_stiffness.bin"))?;
    if !stiff.is_empty() && stiff.len() != n {
        bail!(
            "stitch_stiffness has {} entries but there are {n} stitches",
            stiff.len()
        );
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let index: [u32; 6] = std::array::from_fn(|k| {
            u64::from_le_bytes(ind[i * 48 + k * 8..i * 48 + k * 8 + 8].try_into().unwrap()) as u32
        });
        let weight: [f32; 6] = std::array::from_fn(|k| {
            f32::from_le_bytes(w[i * 24 + k * 4..i * 24 + k * 4 + 4].try_into().unwrap())
        });
        out.push(SceneStitch {
            index,
            weight,
            stiffness: stiff.get(i).copied().unwrap_or(1.0),
        });
    }
    Ok(out)
}

/// One pin block: the vertices it constrains, and whether the scene ships any
/// `pin-<i>-op-<j>.bin` for it (which makes the block's target ANIMATED).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PinBlock {
    pub indices: Vec<u32>,
}

/// Read the pin blocks separately (see [`read_pin_indices`] for the file format), so a
/// caller can tell a STATIC block from an ANIMATED one instead of pooling them.
pub fn read_pin_blocks(scene_dir: impl AsRef<Path>) -> Result<Vec<PinBlock>> {
    let ib = scene_dir.as_ref().join("input_bin");
    let mut out = Vec::new();
    let mut n = 0;
    while let Ok(bytes) = std::fs::read(ib.join(format!("pin-ind-{n}.bin"))) {
        if !bytes.len().is_multiple_of(8) {
            bail!(
                "pin-ind-{n}.bin is {} bytes, not a whole number of u64",
                bytes.len()
            );
        }
        let mut indices = Vec::with_capacity(bytes.len() / 8);
        for c in bytes.chunks_exact(8) {
            let i = u64::from_le_bytes(c.try_into().unwrap());
            if i > u32::MAX as u64 {
                bail!("pin index {i} exceeds u32::MAX");
            }
            indices.push(i as u32);
        }
        out.push(PinBlock { indices });
        n += 1;
    }
    Ok(out)
}

/// How a pin operation eases from its start time to its end time
/// (`info.toml: transition`, evaluated by `pin_apply::progress_at`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PinTransition {
    /// `raw` — the linear ramp.
    Linear,
    /// `raw²(3 − 2·raw)` — smoothstep.
    Smooth,
    /// Bezier easing via `-handles.bin`; not implemented, treated as a gap.
    Bezier,
}

/// A pin operation's kind, with the per-kind parameters the reference reads from
/// `info.toml` (`scene.rs:675-760`). These four are every kind the gauge uses
/// (`move_by` 423, `spin` 325, `move_to` 120, `scale` 12 across all 22 scenes); anything
/// else is preserved by NAME so a caller can report exactly which one it skipped rather
/// than silently applying the wrong motion.
#[derive(Clone, Debug, PartialEq)]
pub enum PinOpKind {
    /// `position + delta·progress` (`pin_apply.rs:53`).
    MoveBy,
    /// `lerp(position, target, progress)` (`pin_apply.rs:63`).
    MoveTo,
    /// Rigid rotation about `axis` through `center`, sweeping
    /// `radians(angular_velocity)·(min(t, t_end) − t_start)` (`pin_apply.rs:76-117`).
    /// `center_mode` is NOT read: the reference ignores it and uses `center_*` directly
    /// (and every gauge scene ships `center_mode = "absolute"` anyway).
    Spin {
        center: [f32; 3],
        axis: [f32; 3],
        /// Degrees per second.
        angular_velocity: f32,
    },
    /// Uniform scaling about `center`, blending 1.0 → `factor` across the window
    /// (`pin_apply.rs:119-150`).
    Scale { center: [f32; 3], factor: f32 },
    /// `transform_keyframes`, `torque`, … — none of which appear in the gauge.
    Unsupported(String),
}

/// One scheduled pin operation: its kind and window from `info.toml`, plus the per-vertex
/// `f64[P_i][3]` payload from `pin-<i>-op-<j>.bin` (a delta for `move_by`, an absolute
/// target for `move_to`).
#[derive(Clone, Debug)]
pub struct PinOp {
    pub kind: PinOpKind,
    pub t_start: f64,
    pub t_end: f64,
    pub transition: PinTransition,
    /// One row per vertex of the owning block, parallel to its `indices`.
    pub data: Vec<[f32; 3]>,
}

impl PinOp {
    /// The eased ramp at `time`, clamped to `[0, 1]`. Mirrors the reference's
    /// `apply_op`: below `t_start` the op has not begun (`None`), at or past `t_end` the
    /// progress is exactly 1, and in between it is `progress_at`
    /// (`scene.rs:1925-1943`, `pin_apply.rs:27-48`).
    pub fn progress_at(&self, time: f64) -> Option<f64> {
        if time < self.t_start {
            return None;
        }
        if time >= self.t_end {
            return Some(1.0);
        }
        let denom = self.t_end - self.t_start;
        let raw = if denom != 0.0 {
            (time - self.t_start) / denom
        } else {
            0.0
        };
        Some(match self.transition {
            PinTransition::Smooth => raw * raw * (3.0 - 2.0 * raw),
            // Bezier is reported as a gap by the caller; fall back to the linear ramp
            // rather than inventing handle values.
            PinTransition::Linear | PinTransition::Bezier => raw,
        })
    }
}

/// One pin block's full schedule: which vertices it holds, its `FixPair` scalars, and its
/// ordered operations.
#[derive(Clone, Debug)]
pub struct PinBlockSchedule {
    pub indices: Vec<u32>,
    /// `info.toml: stiffness` — the per-pin force scale, applied only to kinematic pins.
    pub stiffness: f32,
    /// `info.toml: pull` — when > 0 the reference routes the vertex to a `PullPair`
    /// (a weighted soft pull) instead of a `FixPair`. Not modeled.
    pub pull: f32,
    pub ops: Vec<PinOp>,
}

impl PinBlockSchedule {
    /// Target position for the block's `slot`-th vertex at `time`, starting from `base`
    /// (the vertex's initial position) and composing every operation IN ORDER, exactly as
    /// the reference's `apply_op` loop does (`scene.rs:2068-2081`).
    ///
    /// The second return value is the reference's `kinematic` flag: true once at least one
    /// operation has begun. A block whose ops all lie in the future is still static, and a
    /// static pin is not stiffness-scaled (`contact.cu:1052-1056`).
    ///
    /// `base` must be the DISPLACED initial position. The reference composes from the
    /// template `vert[ind]` and adds `displacement[vert_dmap[ind]]` at the end; since every
    /// modeled op is a translation, starting from `vert + displacement` is the same result
    /// with the displacement folded in (and [`load_golden_input`] already folds it).
    pub fn target_at(&self, slot: usize, base: [f32; 3], time: f64) -> ([f32; 3], bool) {
        let mut p = base;
        let mut kinematic = false;
        for op in &self.ops {
            match &op.kind {
                PinOpKind::MoveBy | PinOpKind::MoveTo => {
                    // These carry a per-vertex payload and use the eased ramp.
                    let (Some(progress), Some(row)) = (op.progress_at(time), op.data.get(slot))
                    else {
                        continue;
                    };
                    for k in 0..3 {
                        if op.kind == PinOpKind::MoveBy {
                            p[k] += row[k] * progress as f32;
                        } else {
                            p[k] += (row[k] - p[k]) * progress as f32;
                        }
                    }
                    kinematic = true;
                }
                PinOpKind::Spin {
                    center,
                    axis,
                    angular_velocity,
                } => {
                    // Spin does NOT use `progress_at`: the swept angle is angular velocity
                    // times elapsed time, clamped at `t_end` (`pin_apply.rs:109-117`).
                    let angle =
                        (*angular_velocity as f64).to_radians() * (time.min(op.t_end) - op.t_start);
                    // The reference gates on `angle <= 0`, which also covers `time <
                    // t_start` (negative elapsed). Note the consequence, faithfully kept: a
                    // NEGATIVE angular_velocity is a no-op in the reference, not a reverse
                    // spin (`scene.rs:1985-1989`).
                    if angle <= 0.0 {
                        continue;
                    }
                    p = rodrigues(p, *center, *axis, angle);
                    kinematic = true;
                }
                PinOpKind::Scale { center, factor } => {
                    if time < op.t_start {
                        continue;
                    }
                    let f = if time >= op.t_end {
                        *factor as f64
                    } else {
                        // 1.0 → factor across the window, along the eased ramp.
                        let progress = op.progress_at(time).unwrap_or(0.0);
                        1.0 + (*factor as f64 - 1.0) * progress
                    } as f32;
                    for k in 0..3 {
                        p[k] = (p[k] - center[k]) * f + center[k];
                    }
                    kinematic = true;
                }
                // An unmodeled op contributes no motion. The caller reports it.
                PinOpKind::Unsupported(_) => {}
            }
        }
        (p, kinematic)
    }
}

/// Rotate `p` by `angle_rad` about the (normalized) `axis` through `center` — Rodrigues,
/// matching `pin_apply::spin_step` (`pin_apply.rs:76-102`). A zero-length axis yields no
/// rotation, as there it does.
fn rodrigues(p: [f32; 3], center: [f32; 3], axis: [f32; 3], angle_rad: f64) -> [f32; 3] {
    let n = ((axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]) as f64).sqrt();
    let inv = if n > 0.0 { 1.0 / n } else { 0.0 };
    let a = [
        axis[0] as f64 * inv,
        axis[1] as f64 * inv,
        axis[2] as f64 * inv,
    ];
    let (cos_t, sin_t) = (angle_rad.cos(), angle_rad.sin());
    let omc = 1.0 - cos_t;
    let d = [
        (p[0] - center[0]) as f64,
        (p[1] - center[1]) as f64,
        (p[2] - center[2]) as f64,
    ];
    let cross = [
        a[1] * d[2] - a[2] * d[1],
        a[2] * d[0] - a[0] * d[2],
        a[0] * d[1] - a[1] * d[0],
    ];
    let dot = d[0] * a[0] + d[1] * a[1] + d[2] * a[2];
    let mut out = [0.0f32; 3];
    for k in 0..3 {
        out[k] = (d[k] * cos_t + cross[k] * sin_t + a[k] * dot * omc) as f32 + center[k];
    }
    out
}

/// Read every pin block's schedule: `info.toml`'s `[pin-<i>]` / `[pin-<i>-op-<j>]` tables
/// plus each op's `f64[P_i][3]` payload.
///
/// The op TYPE and window live only in `info.toml` (plain text: `type`, `t_start`,
/// `t_end`, `transition`), so they are read from there rather than guessed from which
/// binaries exist.
pub fn read_pin_schedule(scene_dir: impl AsRef<Path>) -> Result<Vec<PinBlockSchedule>> {
    let dir = scene_dir.as_ref();
    let blocks = read_pin_blocks(dir)?;
    let info = std::fs::read_to_string(dir.join("info.toml")).unwrap_or_default();
    let ib = dir.join("input_bin");
    let mut out = Vec::with_capacity(blocks.len());

    for (i, block) in blocks.iter().enumerate() {
        let head = toml_section(&info, &format!("pin-{i}"));
        let mut ops = Vec::new();
        let n_ops = head
            .get("operation_count")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        for j in 0..n_ops {
            let sec = toml_section(&info, &format!("pin-{i}-op-{j}"));
            let ty = sec.get("type").map(String::as_str).unwrap_or("");
            let f32_of = |k: &str| {
                sec.get(k)
                    .and_then(|v| v.parse::<f32>().ok())
                    .unwrap_or(0.0)
            };
            let kind = match ty {
                "move_by" => PinOpKind::MoveBy,
                "move_to" => PinOpKind::MoveTo,
                "spin" => PinOpKind::Spin {
                    center: [f32_of("center_x"), f32_of("center_y"), f32_of("center_z")],
                    axis: [f32_of("axis_x"), f32_of("axis_y"), f32_of("axis_z")],
                    angular_velocity: f32_of("angular_velocity"),
                },
                "scale" => PinOpKind::Scale {
                    center: [f32_of("center_x"), f32_of("center_y"), f32_of("center_z")],
                    factor: sec
                        .get("factor")
                        .and_then(|v| v.parse::<f32>().ok())
                        .unwrap_or(1.0),
                },
                other => PinOpKind::Unsupported(other.to_string()),
            };
            let transition = match sec.get("transition").map(String::as_str) {
                Some("smooth") => PinTransition::Smooth,
                Some("bezier") => PinTransition::Bezier,
                _ => PinTransition::Linear,
            };
            let num = |k: &str, d: f64| sec.get(k).and_then(|v| v.parse::<f64>().ok()).unwrap_or(d);
            // The payload is absent for ops that carry their motion in `info.toml`
            // (spin/scale) rather than per-vertex; those are unsupported anyway.
            let path = ib.join(format!("pin-{i}-op-{j}.bin"));
            let data = if path.exists() {
                let rows = decode_vec3_f64(&std::fs::read(&path)?)?;
                if rows.len() != block.indices.len() {
                    bail!(
                        "pin-{i}-op-{j}.bin has {} rows but block {i} holds {} vertices",
                        rows.len(),
                        block.indices.len()
                    );
                }
                rows
            } else {
                Vec::new()
            };
            ops.push(PinOp {
                kind,
                t_start: num("t_start", 0.0),
                t_end: num("t_end", 0.0),
                transition,
                data,
            });
        }
        let scalar = |k: &str, d: f32| head.get(k).and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
        out.push(PinBlockSchedule {
            indices: block.indices.clone(),
            stiffness: scalar("stiffness", 1.0),
            pull: scalar("pull", 0.0),
            ops,
        });
    }
    Ok(out)
}

/// Collect `key = value` pairs of one `[section]` from a flat TOML document. Values keep
/// their raw text with surrounding quotes stripped; this is the same minimal reader
/// approach as [`read_param_toml`] (no `toml` dependency).
fn toml_section(text: &str, want: &str) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    let mut inside = false;
    for line in text.lines() {
        let t = line.trim();
        if let Some(name) = t.strip_prefix('[').and_then(|r| r.strip_suffix(']')) {
            inside = name == want;
            continue;
        }
        if !inside {
            continue;
        }
        if let Some((k, v)) = t.split_once('=') {
            map.insert(k.trim().to_string(), v.trim().trim_matches('"').to_string());
        }
    }
    map
}

/// The reference per-triangle material for a scene, averaged to scalars (the gauge
/// scenes are materially uniform, so the mean == the per-element value).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GoldenMaterial {
    pub young_mod: f32,
    pub bend: f32,
    pub density: f32,
    pub strain_limit: f32,
}

/// Mean of a flat little-endian `f32` buffer, or `None` if absent/empty/misaligned.
fn read_f32_mean(path: &Path) -> Option<f32> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return None;
    }
    let mut sum = 0.0f64;
    let mut n = 0u64;
    for c in bytes.chunks_exact(4) {
        sum += f32::from_le_bytes(c.try_into().unwrap()) as f64;
        n += 1;
    }
    Some((sum / n as f64) as f32)
}

/// Read the reference material from `input_bin/param/tri-*.bin` (`f32` per triangle):
/// `tri-young-mod`, `tri-bend`, `tri-density`, `tri-strain-limit`. Returns `None` if
/// `tri-young-mod.bin` is missing (no per-object material captured); the others fall
/// back to sensible defaults. Lets the golden driver use the reference's own
/// stiffness instead of our default, which is the dominant lever for drape parity.
pub fn read_material(scene_dir: impl AsRef<Path>) -> Option<GoldenMaterial> {
    let p = scene_dir.as_ref().join("input_bin/param");
    Some(GoldenMaterial {
        young_mod: read_f32_mean(&p.join("tri-young-mod.bin"))?,
        bend: read_f32_mean(&p.join("tri-bend.bin")).unwrap_or(0.0),
        density: read_f32_mean(&p.join("tri-density.bin")).unwrap_or(1.0),
        strain_limit: read_f32_mean(&p.join("tri-strain-limit.bin")).unwrap_or(0.0),
    })
}

/// Load a reference checkpoint `checkpoints/vert_<frame>.bin` (`f32` xyz positions).
pub fn load_checkpoint(scene_dir: impl AsRef<Path>, frame: u32) -> Result<Vec<[f32; 3]>> {
    let path = scene_dir
        .as_ref()
        .join("checkpoints")
        .join(format!("vert_{frame}.bin"));
    decode_vec3_f32(&read(&path)?)
}

fn read(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).with_context(|| format!("reading golden binary {}", path.display()))
}

/// Divergence of our positions from a reference checkpoint, normalized by the
/// reference bounding-box diagonal so the thresholds are scale-free. The three
/// relative gates (`docs/GAUGE_FINDINGS.md` §1):
/// - `max_rel` — worst per-vertex distance / diagonal (gate: `< 8e-2`)
/// - `rms` — RMS per-vertex distance / diagonal (gate: `< 2e-2`)
/// - `centroid_rel` — centroid displacement / diagonal (gate: `< 6e-3`)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GateReport {
    pub max_rel: f32,
    pub rms: f32,
    pub centroid_rel: f32,
    /// Reference bbox diagonal used to normalize (scene scale).
    pub diagonal: f32,
}

impl GateReport {
    /// The three gate thresholds (max_rel, rms, centroid_rel).
    pub const GATES: (f32, f32, f32) = (8.0e-2, 2.0e-2, 6.0e-3);

    /// Whether all three gates pass.
    pub fn passes(&self) -> bool {
        self.max_rel < Self::GATES.0
            && self.rms < Self::GATES.1
            && self.centroid_rel < Self::GATES.2
    }
}

/// Compare our positions to a `reference` checkpoint, normalized by the reference
/// bbox diagonal (a degenerate/zero diagonal falls back to 1.0 so metrics stay
/// finite). Panics only via the length assert — both must have the same vertex
/// count.
pub fn compare_frames(ours: &[[f32; 3]], reference: &[[f32; 3]]) -> Result<GateReport> {
    if ours.len() != reference.len() {
        bail!(
            "frame vertex count mismatch: ours {} vs reference {}",
            ours.len(),
            reference.len()
        );
    }
    let n = ours.len().max(1);
    // Reference bbox diagonal.
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    let mut ref_centroid = [0.0f64; 3];
    let mut our_centroid = [0.0f64; 3];
    for k in 0..3 {
        for r in reference {
            lo[k] = lo[k].min(r[k]);
            hi[k] = hi[k].max(r[k]);
            ref_centroid[k] += r[k] as f64;
        }
        for o in ours {
            our_centroid[k] += o[k] as f64;
        }
    }
    let diagonal = {
        let d = (0..3).map(|k| (hi[k] - lo[k]).powi(2)).sum::<f32>().sqrt();
        if d > 1.0e-12 {
            d
        } else {
            1.0
        }
    };
    let mut max_d = 0.0f32;
    let mut sum_sq = 0.0f64;
    for (o, r) in ours.iter().zip(reference) {
        let d2: f32 = (0..3).map(|k| (o[k] - r[k]).powi(2)).sum();
        let d = d2.sqrt();
        max_d = max_d.max(d);
        sum_sq += d2 as f64;
    }
    let rms = ((sum_sq / n as f64).sqrt()) as f32;
    let centroid_d = (0..3)
        .map(|k| ((our_centroid[k] - ref_centroid[k]) / n as f64).powi(2))
        .sum::<f64>()
        .sqrt() as f32;
    Ok(GateReport {
        max_rel: max_d / diagonal,
        rms: rms / diagonal,
        centroid_rel: centroid_d / diagonal,
        diagonal,
    })
}

/// Locate the reference capture, or `None` if it is not present — which is the
/// normal case, since the capture is not distributed with this source.
///
/// Test support, deliberately **public and not `#[cfg(test)]`**: an integration test in
/// `tests/` cannot see a `#[cfg(test)]` item, which is precisely why this function was
/// copied into four places and then diverged. Two of those copies guarded the env var
/// with `exists()` instead of `is_dir()`, which is a real bug — see (2) below.
///
/// Resolution order:
/// 1. `PPF_GAUGE_DIR`, **if it names a real directory**.
/// 2. the nearest `ppf-cts-gauge` in any ancestor of this crate.
///
/// Two failures this shape is defending against, both of which actually happened:
///
/// 1. **Dormant guards.** The env-var-only form meant these checks were skipped in a
///    plain `cargo test` — nothing sets the variable, so systemic checks returned early
///    and still reported `ok`. Five of them were silently vacuous for multiple sessions.
/// 2. **A stale or mistyped path sailing past the guard.** `exists()` is true for a
///    *file* as well as a directory, so a bad `PPF_GAUGE_DIR` got past the check and
///    then failed deep inside the assertions (`colliders_decode_as_the_scenes_declare_them`
///    panicked `0 != 1`) instead of skipping cleanly. `is_dir()` answers both.
///
/// The ancestor walk rather than a fixed `../../../` is the third lesson: the tree moved
/// once (`ppf-contact-solver/ppf-mguso-solver` → `nguo-3d/core-engine`, which also pulled
/// the gauge inside `core-engine`) and the hardcoded depth then resolved to nothing,
/// re-creating failure (1). If this crate is ever extracted to its own repository the
/// walk finds nothing and `PPF_GAUGE_DIR` becomes the only route — hence step 1.
#[must_use]
pub fn gauge_dir() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("PPF_GAUGE_DIR") {
        let p = std::path::PathBuf::from(p);
        return p.is_dir().then_some(p);
    }
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .map(|a| a.join("ppf-cts-gauge"))
        .find(|p| p.is_dir())
        .map(|p| p.canonicalize().unwrap_or(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    // One implementation, at module scope. See `super::gauge_dir` for why it is public
    // and what the two guard failures it defends against were.
    use super::gauge_dir;

    /// Every `u64`-index decoder must REFUSE a word that does not fit in `u32`, not
    /// truncate it.
    ///
    /// Regression test for a real defect: `decode_tets_u64`/`decode_rods_u64` used a bare
    /// `as u32` while `decode_tris_u64` guarded. The bite is subtle — the loader's
    /// `i >= nv` bounds check runs AFTER narrowing, so `0x1_0000_0005` became vertex `5`,
    /// sailed through the bounds check, and built elements over the wrong vertices with
    /// no error at all. The index chosen here is deliberately SMALL after truncation, so
    /// a regression cannot be caught by a bounds check downstream.
    #[test]
    fn u64_index_decoders_refuse_to_truncate() {
        let poison: u64 = 0x1_0000_0005; // truncates to 5 — a plausible, in-range vertex
        assert_eq!(poison as u32, 5, "the test's own premise");

        let tri: Vec<u8> = [0u64, 1, poison]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let tet: Vec<u8> = [0u64, 1, 2, poison]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let rod: Vec<u8> = [0u64, poison]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        for (what, err) in [
            ("triangle", super::decode_tris_u64(&tri).err()),
            ("tet", super::decode_tets_u64(&tet).err()),
            ("rod", super::decode_rods_u64(&rod).err()),
        ] {
            let e = err.unwrap_or_else(|| panic!("{what} decoder silently truncated {poison}"));
            assert!(
                e.to_string().contains("exceeds u32::MAX"),
                "{what}: unhelpful error {e}"
            );
        }

        // And the in-range path still decodes, so the guard is not simply refusing work.
        let ok: Vec<u8> = [7u64, 8, 9, 10]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(super::decode_tets_u64(&ok).unwrap(), vec![[7, 8, 9, 10]]);
        assert_eq!(super::decode_rods_u64(&ok[..16]).unwrap(), vec![[7, 8]]);
    }

    /// The loaded initial positions must equal the reference's own frame-0 checkpoint for
    /// EVERY gauge scene. This is what proves the displacement map is applied correctly:
    /// `vert.bin` alone is off by up to 13 world units (ribbon) in 18 of the 22 scenes, so
    /// this test fails loudly if `apply_displacement` regresses or is skipped.
    #[test]
    fn loaded_positions_match_the_reference_frame_zero() {
        let Some(gauge) = gauge_dir() else {
            eprintln!("no gauge capture (set PPF_GAUGE_DIR); skipping");
            return;
        };
        let goldens = std::path::Path::new(&gauge).join("goldens");
        let Ok(entries) = std::fs::read_dir(&goldens) else {
            eprintln!("no goldens dir; skipping");
            return;
        };
        let mut checked = 0;
        let mut worst = 0.0f32;
        // Records what the `continue` below actually skipped. This vector used to be
        // built empty and never pushed to, so the report at the end always read
        // "skipped 0 scene(s): []" no matter how many scenes were passed over — the
        // exact silent-rot this test exists to prevent, hiding inside the test itself.
        let mut skipped: Vec<String> = Vec::new();
        for e in entries.filter_map(Result::ok) {
            let scene = e.path();
            if !scene.join("input_bin/vert.bin").exists()
                || !scene.join("checkpoints/vert_0.bin").exists()
            {
                skipped.push(scene.file_name().map_or_else(
                    || scene.display().to_string(),
                    |n| n.to_string_lossy().into(),
                ));
                continue;
            }
            // Rod-only scenes (noodle, woven, yarn) ship `rod.bin` and no `tri.bin`; the
            // loader now reads them, so they are checked like everything else. `skipped`
            // is the report of anything a future scene shape excludes.
            let input = match load_golden_input(&scene) {
                Ok(i) => i,
                Err(err) => panic!("{:?} failed to load: {err}", scene.file_name().unwrap()),
            };
            let frame0 = load_checkpoint(&scene, 0).expect("frame 0");
            assert_eq!(
                frame0.len(),
                input.positions.len(),
                "{:?}: checkpoint/vert count mismatch",
                scene.file_name().unwrap()
            );
            let mut max = 0.0f32;
            for (a, b) in input.positions.iter().zip(&frame0) {
                for k in 0..3 {
                    max = max.max((a[k] - b[k]).abs());
                }
            }
            // f32 checkpoints against f64 inputs: a few ULPs at these magnitudes.
            assert!(
                max < 1.0e-4,
                "{:?}: initial positions differ from frame 0 by {max}",
                scene.file_name().unwrap()
            );
            worst = worst.max(max);
            checked += 1;
        }
        assert!(checked > 0, "no gauge scene was checked");
        eprintln!(
            "frame-0 match verified for {checked} scenes, worst error {worst:.2e}; \
             skipped {} scene(s): {skipped:?}",
            skipped.len()
        );
        // The rod-only scenes are the ones this used to skip; they must now be covered.
        for name in ["noodle", "woven", "yarn"] {
            let scene = goldens.join(name);
            if scene.join("input_bin/vert.bin").exists() {
                let input = load_golden_input(&scene).expect("rod-only scene must load");
                assert!(
                    input.tris.is_empty() && !input.rods.is_empty(),
                    "{name} must load as a rod-only scene, got {} tris / {} rods",
                    input.tris.len(),
                    input.rods.len()
                );
            }
        }
    }

    /// `spin` must be a RIGID rotation with the reference's angle law
    /// (`radians(ω)·(min(t,t_end) − t_start)`, clamped at `t_end`), and `scale` must blend
    /// 1.0 → factor about its center. Checked against geometry known independently of the
    /// implementation: a 90° turn about +Z maps (1,0,0) → (0,1,0).
    #[test]
    fn spin_and_scale_ops_match_the_reference_laws() {
        let mk = |kind: PinOpKind, t_start: f64, t_end: f64| PinBlockSchedule {
            indices: vec![0],
            stiffness: 1.0,
            pull: 0.0,
            ops: vec![PinOp {
                kind,
                t_start,
                t_end,
                transition: PinTransition::Linear,
                data: Vec::new(),
            }],
        };
        let near = |a: [f32; 3], b: [f32; 3]| (0..3).all(|k| (a[k] - b[k]).abs() < 1e-5);

        // 90 deg/s about +Z through the origin: after 1 s, (1,0,0) -> (0,1,0).
        let spin = mk(
            PinOpKind::Spin {
                center: [0.0; 3],
                axis: [0.0, 0.0, 1.0],
                angular_velocity: 90.0,
            },
            0.0,
            2.0,
        );
        let x = [1.0f32, 0.0, 0.0];
        let (q, kin) = spin.target_at(0, x, 1.0);
        assert!(near(q, [0.0, 1.0, 0.0]), "90 deg about +Z: {q:?}");
        assert!(kin, "an active spin is kinematic");
        // Half the time, half the angle (45 deg).
        let (h, _) = spin.target_at(0, x, 0.5);
        let r = std::f32::consts::FRAC_1_SQRT_2;
        assert!(near(h, [r, r, 0.0]), "45 deg about +Z: {h:?}");
        // Rigid: the radius about the axis is preserved.
        let rad = (h[0] * h[0] + h[1] * h[1]).sqrt();
        assert!((rad - 1.0).abs() < 1e-5, "spin must be rigid, radius {rad}");
        // Clamped at t_end: 2 s of 90 deg/s = 180 deg, and it stays there.
        let (e1, _) = spin.target_at(0, x, 2.0);
        let (e2, _) = spin.target_at(0, x, 50.0);
        assert!(near(e1, [-1.0, 0.0, 0.0]), "180 deg: {e1:?}");
        assert!(near(e1, e2), "spin must clamp at t_end: {e1:?} vs {e2:?}");
        // Before t_start the elapsed time is negative, so the angle is <= 0 and the
        // reference treats the op as a no-op.
        let later = mk(
            PinOpKind::Spin {
                center: [0.0; 3],
                axis: [0.0, 0.0, 1.0],
                angular_velocity: 90.0,
            },
            5.0,
            6.0,
        );
        let (pre, pre_kin) = later.target_at(0, x, 0.0);
        assert!(near(pre, x), "a spin before t_start must not move: {pre:?}");
        assert!(!pre_kin);

        // Scale: 1.0 -> 3.0 about (1,1,1) over t in [0,1].
        let sc = mk(
            PinOpKind::Scale {
                center: [1.0, 1.0, 1.0],
                factor: 3.0,
            },
            0.0,
            1.0,
        );
        let p = [2.0f32, 1.0, 1.0];
        assert!(
            near(sc.target_at(0, p, 0.0).0, [2.0, 1.0, 1.0]),
            "t=0 is factor 1"
        );
        // Halfway the factor is 2.0, so the offset from the center doubles.
        assert!(
            near(sc.target_at(0, p, 0.5).0, [3.0, 1.0, 1.0]),
            "t=0.5 is factor 2"
        );
        assert!(
            near(sc.target_at(0, p, 1.0).0, [4.0, 1.0, 1.0]),
            "t=1 is factor 3"
        );
        assert!(
            near(sc.target_at(0, p, 9.0).0, [4.0, 1.0, 1.0]),
            "clamped past t_end"
        );
        // The center itself is a fixed point of a scale.
        assert!(near(
            sc.target_at(0, [1.0, 1.0, 1.0], 1.0).0,
            [1.0, 1.0, 1.0]
        ));
    }

    /// Every pin operation the gauge actually uses must decode to a MODELED kind — if a
    /// scene ever introduces `transform_keyframes` or `torque`, this fails and says which.
    #[test]
    fn all_gauge_pin_ops_are_modeled() {
        let Some(gauge) = gauge_dir() else {
            eprintln!("no gauge capture (set PPF_GAUGE_DIR); skipping");
            return;
        };
        let goldens = std::path::Path::new(&gauge).join("goldens");
        let Ok(entries) = std::fs::read_dir(&goldens) else {
            eprintln!("no goldens dir; skipping");
            return;
        };
        let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
        let mut unmodeled: Vec<String> = Vec::new();
        for e in entries.filter_map(Result::ok) {
            let scene = e.path();
            if !scene.join("info.toml").exists() {
                continue;
            }
            for b in read_pin_schedule(&scene).expect("schedule") {
                for op in &b.ops {
                    let name = match &op.kind {
                        PinOpKind::MoveBy => "move_by",
                        PinOpKind::MoveTo => "move_to",
                        PinOpKind::Spin { .. } => "spin",
                        PinOpKind::Scale { .. } => "scale",
                        PinOpKind::Unsupported(n) => {
                            unmodeled.push(format!(
                                "{:?}: {n}",
                                scene.file_name().unwrap().to_string_lossy()
                            ));
                            "unsupported"
                        }
                    };
                    *counts.entry(name.to_string()).or_default() += 1;
                }
            }
        }
        eprintln!("gauge pin op kinds: {counts:?}");
        assert!(
            unmodeled.is_empty(),
            "unmodeled pin operations in the gauge: {unmodeled:?}"
        );
        assert!(counts.values().sum::<usize>() > 0, "no pin ops were read");
    }

    /// The pin schedule must decode `plastic` exactly as its `info.toml` and op binaries
    /// declare it, and `target_at` must compose the ops over time the way the reference's
    /// `apply_op` loop does.
    #[test]
    fn plastic_pin_schedule_decodes_and_composes() {
        let Some(gauge) = gauge_dir() else {
            eprintln!("no gauge capture (set PPF_GAUGE_DIR); skipping");
            return;
        };
        let scene = std::path::Path::new(&gauge).join("goldens/plastic");
        if !scene.exists() {
            eprintln!("plastic absent; skipping");
            return;
        }
        let sched = read_pin_schedule(&scene).expect("schedule");
        assert_eq!(sched.len(), 2, "plastic declares two pin blocks");
        for b in &sched {
            assert_eq!(b.indices.len(), 1024);
            assert_eq!(b.stiffness, 1.0);
            assert_eq!(b.pull, 0.0, "pull = 0 routes to FixPair, not PullPair");
        }
        // Block 0: move_by (0,−0.74,0) over t∈[0,1], then move_by (0,+1,0) over t∈[1.5,2].
        let b0 = &sched[0];
        assert_eq!(b0.ops.len(), 2);
        assert_eq!(b0.ops[0].kind, PinOpKind::MoveBy);
        assert_eq!(b0.ops[0].transition, PinTransition::Linear);
        assert_eq!((b0.ops[0].t_start, b0.ops[0].t_end), (0.0, 1.0));
        assert_eq!((b0.ops[1].t_start, b0.ops[1].t_end), (1.5, 2.0));
        let near = |a: f32, b: f32| (a - b).abs() < 1e-5;
        assert!(near(b0.ops[0].data[0][1], -0.74), "op0 delta y");
        assert!(near(b0.ops[1].data[0][1], 1.0), "op1 delta y");
        // Block 1: a single move_by with a ZERO delta — animated in form, static in effect.
        assert_eq!(sched[1].ops.len(), 1);
        assert!(near(sched[1].ops[0].data[0][1], 0.0));

        // Compose over time. Base is arbitrary; only the offset matters.
        let base = [0.0f32, 10.0, 0.0];
        let y_at = |t: f64| b0.target_at(0, base, t).0[1] - base[1];
        assert!(near(y_at(0.0), 0.0), "at t=0 progress is 0: {}", y_at(0.0));
        assert!(near(y_at(0.5), -0.37), "half of op0: {}", y_at(0.5));
        assert!(near(y_at(1.0), -0.74), "op0 complete: {}", y_at(1.0));
        // Between the ops, op1 has not begun.
        assert!(near(y_at(1.25), -0.74), "gap between ops: {}", y_at(1.25));
        assert!(near(y_at(1.75), -0.24), "op0 + half op1: {}", y_at(1.75));
        assert!(near(y_at(2.0), 0.26), "both complete: {}", y_at(2.0));
        assert!(near(y_at(99.0), 0.26), "clamped past t_end: {}", y_at(99.0));

        // The kinematic flag: block 0's first op starts at t=0, so it is kinematic from
        // the outset; a time before any op would not be.
        assert!(
            b0.target_at(0, base, 0.0).1,
            "op starting at t=0 is kinematic"
        );
        let future = PinBlockSchedule {
            indices: vec![0],
            stiffness: 1.0,
            pull: 0.0,
            ops: vec![PinOp {
                kind: PinOpKind::MoveBy,
                t_start: 5.0,
                t_end: 6.0,
                transition: PinTransition::Linear,
                data: vec![[0.0, 1.0, 0.0]],
            }],
        };
        let (p, kin) = future.target_at(0, base, 0.0);
        assert_eq!(
            p, base,
            "an op that has not started must not move the target"
        );
        assert!(
            !kin,
            "a block whose ops are all in the future is still static"
        );
    }

    /// Colliders must decode exactly as the scenes declare them, INCLUDING the one animated
    /// case. Checked against `info.toml`/binary facts established by hand: all 6 wall scenes
    /// are static axis-aligned half-spaces; `ribbon`/`noodle` are static spheres; `trapped`
    /// is a sphere with **4 keyframes animating position AND radius**.
    #[test]
    fn colliders_decode_as_the_scenes_declare_them() {
        let Some(gauge) = gauge_dir() else {
            eprintln!("no gauge capture (set PPF_GAUGE_DIR); skipping");
            return;
        };
        let g = std::path::Path::new(&gauge).join("goldens");

        // domino: one static floor at y = -0.1255, normal +Y.
        let c = read_colliders(g.join("domino")).expect("domino");
        assert_eq!(c.len(), 1);
        assert!(c[0].is_wall() && !c[0].is_animated());
        assert_eq!(c[0].normal, Some([0.0, 1.0, 0.0]));
        let (p, _r) = c[0].sample(0.0);
        assert!((p[1] + 0.1255).abs() < 1e-6, "floor height, got {p:?}");
        // A static collider must read the same at ANY time.
        assert_eq!(c[0].sample(0.0), c[0].sample(123.0));
        assert!((c[0].ghat - 1.0e-3).abs() < 1e-9);

        // codim: four walls, opposing pairs on x and z.
        let c = read_colliders(g.join("codim")).expect("codim");
        assert_eq!(c.len(), 4, "codim ships 4 walls");
        assert!(c.iter().all(|w| w.is_wall() && !w.is_animated()));
        let normals: Vec<[f32; 3]> = c.iter().map(|w| w.normal.unwrap()).collect();
        assert!(normals.contains(&[-1.0, 0.0, 0.0]) && normals.contains(&[1.0, 0.0, 0.0]));
        assert!(normals.contains(&[0.0, 0.0, -1.0]) && normals.contains(&[0.0, 0.0, 1.0]));

        // cards: the only collider in the gauge with nonzero friction.
        let c = read_colliders(g.join("cards")).expect("cards");
        assert!(
            (c[0].friction - 0.5).abs() < 1e-6,
            "cards wall friction 0.5"
        );

        // ribbon: a static sphere with a radius.
        let c = read_colliders(g.join("ribbon")).expect("ribbon");
        assert_eq!(c.len(), 1);
        assert!(!c[0].is_wall(), "sphere has no normal");
        assert!(!c[0].is_animated());
        let (_p, r) = c[0].sample(0.0);
        assert!(r > 0.0, "sphere radius must be positive, got {r}");

        // trapped: the ANIMATED case — 4 keyframes, position and radius both interpolated.
        let c = read_colliders(g.join("trapped")).expect("trapped");
        assert_eq!(c.len(), 1);
        assert!(c[0].is_animated(), "trapped's sphere is animated");
        assert_eq!(c[0].positions.len(), 4);
        assert_eq!(c[0].radii.len(), 4);
        assert_eq!(c[0].times.len(), 4);
        // Endpoint clamping: before the first and after the last keyframe.
        let t0 = c[0].times[0];
        let t3 = c[0].times[3];
        assert_eq!(
            c[0].sample(t0 - 100.0),
            c[0].sample(t0),
            "must clamp, not extrapolate"
        );
        assert_eq!(
            c[0].sample(t3 + 100.0),
            c[0].sample(t3),
            "must clamp, not extrapolate"
        );
        // Midway between two keyframes must be strictly between them, per component.
        if (t3 - t0).abs() > 1e-6 {
            let (ta, tb) = (c[0].times[0], c[0].times[1]);
            if (tb - ta).abs() > 1e-6 {
                let (pm, rm) = c[0].sample(0.5 * (ta + tb));
                let (pa, ra) = (c[0].positions[0], c[0].radii[0]);
                let (pb, rb) = (c[0].positions[1], c[0].radii[1]);
                for k in 0..3 {
                    let (lo, hi) = (pa[k].min(pb[k]), pa[k].max(pb[k]));
                    assert!(
                        pm[k] >= lo - 1e-6 && pm[k] <= hi + 1e-6,
                        "interpolated pos[{k}] {} outside [{lo}, {hi}]",
                        pm[k]
                    );
                }
                let (lo, hi) = (ra.min(rb), ra.max(rb));
                assert!(
                    rm >= lo - 1e-6 && rm <= hi + 1e-6,
                    "interpolated radius {rm} outside range"
                );
            }
        }
        // Every gauge collider is always-active with a pass-through thickness of 1.
        for sc in [
            "domino",
            "codim",
            "cards",
            "ribbon",
            "trapped",
            "roller",
            "stack",
            "trampoline",
        ] {
            for col in read_colliders(g.join(sc)).expect(sc) {
                assert!(col.active_duration < 0.0, "{sc}: expected always-active");
                assert!(
                    (col.thickness - 1.0).abs() < 1e-6,
                    "{sc}: expected thickness 1"
                );
            }
        }
    }

    /// `pin-ind-*.bin` is an index LIST, and the authority on its length is the scene's
    /// own `info.toml` (`[pin-<i>] pin = N`). This checks the reader against that
    /// declaration for every gauge scene that ships pins.
    ///
    /// It is discriminating: the reader previously expanded a 16-byte file as an
    /// inclusive range, which turns hang's declared 2 pins into 128.
    #[test]
    fn pin_indices_match_the_scenes_declared_counts() {
        let Some(gauge) = gauge_dir() else {
            eprintln!("no gauge capture (set PPF_GAUGE_DIR); skipping");
            return;
        };
        let goldens = std::path::Path::new(&gauge).join("goldens");
        let Ok(entries) = std::fs::read_dir(&goldens) else {
            eprintln!("no goldens dir; skipping");
            return;
        };
        let mut checked = 0;
        for e in entries.filter_map(Result::ok) {
            let scene = e.path();
            let Ok(info) = std::fs::read_to_string(scene.join("info.toml")) else {
                continue;
            };
            // Sum every `[pin-<i>]` block's declared `pin = N`. Only the block tables
            // declare `pin`; the `[pin-<i>-op-<j>]` tables do not.
            let mut declared = 0usize;
            let mut in_block = false;
            for line in info.lines() {
                let t = line.trim();
                if t.starts_with('[') {
                    in_block = t.starts_with("[pin-")
                        && !t.contains("-op-")
                        && t.trim_start_matches("[pin-")
                            .trim_end_matches(']')
                            .chars()
                            .all(|c| c.is_ascii_digit());
                } else if in_block {
                    if let Some(v) = t.strip_prefix("pin =") {
                        declared += v.trim().parse::<usize>().unwrap_or(0);
                    }
                }
            }
            if declared == 0 {
                continue;
            }
            let got = read_pin_indices(&scene).expect("pin indices must decode");
            assert_eq!(
                got.len(),
                declared,
                "{:?}: info.toml declares {declared} pinned vertices, reader returned {}",
                scene.file_name().unwrap(),
                got.len()
            );
            checked += 1;
        }
        assert!(checked > 0, "no gauge scene with pins was checked");
        eprintln!("verified pin counts for {checked} scenes");
    }

    #[test]
    fn decodes_synthetic_buffers() {
        // f64 xyz: two vertices.
        let mut fb = Vec::new();
        for x in [1.0f64, 2.0, 3.0, -4.0, 5.5, 6.25] {
            fb.extend_from_slice(&x.to_le_bytes());
        }
        let v = decode_vec3_f64(&fb).unwrap();
        assert_eq!(v, vec![[1.0, 2.0, 3.0], [-4.0, 5.5, 6.25]]);

        // f32 xyz: one vertex.
        let mut sb = Vec::new();
        for x in [0.5f32, -0.25, 10.0] {
            sb.extend_from_slice(&x.to_le_bytes());
        }
        assert_eq!(decode_vec3_f32(&sb).unwrap(), vec![[0.5, -0.25, 10.0]]);

        // u64 xyz: two triangles [0,1,2],[2,1,3].
        let mut tb = Vec::new();
        for i in [0u64, 1, 2, 2, 1, 3] {
            tb.extend_from_slice(&i.to_le_bytes());
        }
        assert_eq!(decode_tris_u64(&tb).unwrap(), vec![[0, 1, 2], [2, 1, 3]]);
    }

    #[test]
    fn compare_identical_is_zero_and_passes() {
        let f = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let r = compare_frames(&f, &f).unwrap();
        assert_eq!(r.max_rel, 0.0);
        assert_eq!(r.rms, 0.0);
        assert_eq!(r.centroid_rel, 0.0);
        assert!(r.passes());
    }

    #[test]
    fn compare_known_offset() {
        // Reference: unit-ish triangle, bbox diagonal = sqrt(1^2+1^2+0) = √2.
        let reference = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        // Shift every vertex +0.1 in x: each per-vertex distance = 0.1, centroid
        // shift = 0.1. diagonal = √2 ≈ 1.41421.
        let ours: Vec<[f32; 3]> = reference.iter().map(|p| [p[0] + 0.1, p[1], p[2]]).collect();
        let g = compare_frames(&ours, &reference).unwrap();
        let diag = 2.0f32.sqrt();
        assert!((g.diagonal - diag).abs() < 1e-5);
        // All distances equal 0.1 → max = rms = 0.1; centroid shift = 0.1. Divided by diag.
        assert!((g.max_rel - 0.1 / diag).abs() < 1e-5);
        assert!((g.rms - 0.1 / diag).abs() < 1e-5);
        assert!((g.centroid_rel - 0.1 / diag).abs() < 1e-5);
        // 0.1/√2 ≈ 0.0707 → max_rel passes (<8e-2) but rms fails (>2e-2).
        assert!(!g.passes(), "rms 0.0707 exceeds the 2e-2 gate");
    }

    #[test]
    fn compare_length_mismatch_errors() {
        assert!(compare_frames(&[[0.0; 3]], &[[0.0; 3], [1.0; 3]]).is_err());
    }

    #[test]
    fn parses_param_toml_from_text() {
        let dir = std::env::temp_dir().join("mguso_golden_param");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("param.toml"),
            "[param]\ndt = 0.001\nfps = 60.0\nframes = 200\n\
             gravity = [0.0, -9.8, 0.0]\nworld_scaling = 2.0\nbarrier = \"cubic\"\n",
        )
        .unwrap();
        let p = read_param_toml(&dir).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(p.dt, 0.001);
        assert_eq!(p.fps, 60.0);
        assert_eq!(p.frames, 200);
        assert_eq!(p.gravity, [0.0, -9.8, 0.0]);
        assert_eq!(p.world_scaling, 2.0);
    }

    #[test]
    fn reads_material_means_from_synthetic() {
        let dir = std::env::temp_dir().join("mguso_golden_mat");
        let p = dir.join("input_bin/param");
        std::fs::create_dir_all(&p).unwrap();
        let write = |name: &str, vals: &[f32]| {
            let mut b = Vec::new();
            for v in vals {
                b.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(p.join(name), b).unwrap();
        };
        write("tri-young-mod.bin", &[100.0, 100.0, 100.0]);
        write("tri-bend.bin", &[1.0, 3.0]); // mean 2
        write("tri-density.bin", &[1.0]);
        write("tri-strain-limit.bin", &[0.01, 0.03]); // mean 0.02
        let m = read_material(&dir).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(m.young_mod, 100.0);
        assert!((m.bend - 2.0).abs() < 1e-6);
        assert_eq!(m.density, 1.0);
        assert!((m.strain_limit - 0.02).abs() < 1e-6);
    }

    /// `pin-ind-*.bin` is a flat `u64` index LIST (`PORTING_data_layout.md:657`), not an
    /// inclusive range: two u64 `(10, 13)` mean two pinned vertices, not four. Blocks are
    /// concatenated in order.
    #[test]
    fn reads_pin_list_from_synthetic() {
        let dir = std::env::temp_dir().join("mguso_golden_pin");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("input_bin")).unwrap();
        let mut bytes = 10u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(&13u64.to_le_bytes());
        std::fs::write(dir.join("input_bin/pin-ind-0.bin"), &bytes).unwrap();
        // A second block of one index, to check concatenation across blocks.
        std::fs::write(
            dir.join("input_bin/pin-ind-1.bin"),
            7u64.to_le_bytes().as_slice(),
        )
        .unwrap();
        let idx = read_pin_indices(&dir).unwrap();
        // A file that is not a whole number of u64 must be rejected outright.
        std::fs::write(dir.join("input_bin/pin-ind-2.bin"), [0u8; 12]).unwrap();
        let bad = read_pin_indices(&dir);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(idx, vec![10, 13, 7]);
        assert!(bad.is_err(), "a misaligned pin file must be an error");
    }

    #[test]
    fn rejects_misaligned_and_oversized() {
        assert!(decode_vec3_f64(&[0u8; 23]).is_err());
        assert!(decode_vec3_f32(&[0u8; 10]).is_err());
        // A u64 index above u32::MAX must be rejected, not truncated.
        let mut tb = (u32::MAX as u64 + 1).to_le_bytes().to_vec();
        tb.extend_from_slice(&[0u8; 16]);
        assert!(decode_tris_u64(&tb).is_err());
    }
}
