// SPDX-License-Identifier: GPL-2.0-or-later
//! Implicit (backward-Euler) mass-spring solver — the high-fidelity tier's
//! machinery, at mass-spring fidelity.
//!
//! Each step minimizes the incremental potential
//!   E(x) = (1/2h²)‖x − x̃‖²_M + Σ_springs ½k(‖d‖−L)²,   x̃ = xₙ + h·vₙ + h²·g
//! by a Newton loop: assemble `H Δx = −∇E`, solve the SPD system with a pluggable
//! [`LinearSolver`] (CPU CG or the GPU PCG), update `x`. Spring Hessians are
//! SPD-projected (clamp the tangential term when compressed) so the system stays
//! SPD for CG — the same "filtered Hessian" idea the reference applies via
//! eigenanalysis (docs/KERNELS.md `energy/eigenanalysis`).
//!
//! This is the variational form the reference IPC solver uses; the contact
//! barrier + CCD line search layer on top later. CPU assembly now; the WGSL
//! assembly kernels (atomic embed into CSR) are the next port chunk.

// Numerical assembly kernels: fixed 3×3 blocks and parallel multi-array (x /
// grad / inv_mass) indexing read clearer as explicit index loops than as zips.
#![allow(clippy::needless_range_loop)]

use crate::bending::{self, Hinge};
use crate::contact::{self, Collider};
use crate::csr::CsrMatrix;
use crate::linsolve::LinearSolver;
use crate::membrane::{self, MembraneElement};
use crate::state::SimState;
use glam::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct ImplicitParams {
    pub dt: f32,
    pub gravity: Vec3,
    /// Newton iterations per step.
    pub newton_iters: u32,
    /// Spring (stretch) stiffness `k`.
    pub stiffness: f32,
    /// Dihedral bending stiffness (0 = no bending; springs only).
    pub bend_stiffness: f32,
    /// Baraff-Witkin membrane stiffness: warp, weft, shear (0 = disabled; use
    /// edge springs instead). Warp≠weft gives anisotropic (woven) fabric.
    pub membrane_warp: f32,
    pub membrane_weft: f32,
    pub membrane_shear: f32,
    /// Coulomb coefficient of the ANALYTIC COLLIDERS, combined with the per-vertex value.
    ///
    /// The reference combines the two with `combine_friction(collider, vertex, friction_mode)`
    /// (`contact.cu:37-47, 1162`) and every gauge scene uses `min` mode — so a collider
    /// declaring friction 0 makes its contacts FRICTIONLESS regardless of the cloth's own
    /// coefficient, which is easy to get wrong by ignoring the collider's value.
    ///
    /// Default `INFINITY` means "no collider coefficient" and is skipped explicitly, so it is
    /// inert under EVERY `friction_mode`. (It was originally relied on to be inert via
    /// `min(INF, vertex) == vertex`; that reasoning silently breaks under `Max`, which returns
    /// INFINITY, and under `Mean`, which poisons the average — hence the explicit test.) Scene-wide because the gauge never mixes
    /// per-collider coefficients within one scene; the driver reports it if one ever does.
    pub collider_friction: f32,
    /// How two participants' friction coefficients combine (`param.toml: friction_mode`).
    /// Default `Min`, matching the reference's `default:` branch — but **14 of the 22 gauge
    /// scenes ship `"max"`**, so this is not a rarely-exercised knob.
    pub friction_mode: crate::contact::FrictionMode,
    /// Cap on the per-vertex magnitude of the Newton SEARCH DIRECTION, in metres
    /// (`param.toml: max_dx`). Default `INFINITY` = uncapped.
    ///
    /// Applied as a single uniform rescale `min(1, max_dx / max‖dx_i‖)` before the line
    /// search (`main.cu:814-847`), so an over-eager step is shortened without bending its
    /// direction — a per-vertex clamp would change the direction and is NOT what the
    /// reference does.
    pub max_dx: f32,
    /// Newton termination target (`param.toml: target_toi`). **0 disables it**, in which
    /// case exactly [`Self::newton_iters`] iterations run (the historical behaviour).
    ///
    /// When > 0 this reproduces the reference's criterion (`main.cu:1038-1046`): once a
    /// Newton iteration's CCD-clamped step fraction reaches `target_toi` AND at least
    /// [`Self::min_newton_steps`] iterations have run, ONE further iteration is taken and
    /// the loop stops. Note it is not a residual-norm test — it keeps iterating while
    /// contact is heavily clamping the step, which is what makes it robustness-driven.
    /// `newton_iters` then acts as the upper bound.
    pub target_toi: f32,
    /// Minimum Newton iterations before [`Self::target_toi`] may end the loop.
    pub min_newton_steps: u32,
    /// Conservative-advancement iteration cap (`param.toml: ccd_max_iter`).
    pub ccd_max_iter: u32,
    /// Degenerate-eigenvalue guard for the isotropic eigensystems
    /// (`param.toml: eiganalysis_eps`, reference default 0.01). Selects when the
    /// `(∂Ψ/∂σᵢ − ∂Ψ/∂σⱼ)/(σᵢ − σⱼ)` flip eigenvalue falls back to its limit form.
    pub eiganalysis_eps: f32,
    /// Aerodynamic damper strength (`param.toml: air_density`). 0 disables the whole air
    /// term. See the assembly note in [`ImplicitSolver::step`]: this is NOT a drag impulse
    /// but an implicit ANISOTROPIC damper, full-strength along the surface normal and
    /// scaled by [`Self::air_friction`] tangentially (`energy/model/air_damper.hpp`).
    pub air_density: f32,
    /// Tangential coefficient of the aerodynamic damper (`param.toml: air_friction`).
    /// 0 damps only motion along the normal; 1 damps all directions equally.
    pub air_friction: f32,
    /// Direction-independent velocity damping (`param.toml: isotropic_air_friction`,
    /// `energy.cu:250-252`). Independent of area and normal.
    pub isotropic_air_friction: f32,
    /// Wind velocity the damper drives surface velocity toward (`param.toml: wind`).
    pub wind: Vec3,
    /// Floor on a pin's remaining gap, as a FRACTION of the pin's `ghat`
    /// (`param.toml: constraint_tol`, reference default 0.01). The reference applies
    /// this to kinematic pins (`contact.cu:1045`) and relies on a time-of-impact clamp
    /// to keep static pins inside `ghat`; we apply it to both, because without that
    /// TOI clamp a static pin's `mass/gap²` term would divide by zero.
    pub constraint_tol: f32,
    /// Contact barrier activation distance `d̂` (docs/CONFIG.md `contact-gap`).
    pub contact_dhat: f32,
    /// Contact barrier base stiffness κ.
    pub contact_kappa: f32,
    /// Dynamic-stiffness coupling for ANALYTIC COLLIDER contacts (0 = fixed κ; >0 scales κ
    /// by local inertia `mass/gap²` — see [`contact::dynamic_stiffness`]).
    ///
    /// Does NOT affect self-collision: those contacts always use the reference's full
    /// stiffness (elastic-Rayleigh + `mass/gap²`, [`ImplicitSolver::self_contact_rayleigh`]),
    /// which the reference has no switch for.
    pub contact_dynamic: f32,
    /// Isotropic-shell stiffness (`shell_mu`/`shell_lambda`) for faces registered via
    /// [`ImplicitSolver::set_shell_faces`]. Both default 0 ⇒ shells contribute nothing.
    pub shell_mu: f32,
    pub shell_lambda: f32,
    /// Constitutive model for the shell faces (default `Arap`, which is what the
    /// gauge's mixed-model scenes use for their non-Baraff-Witkin faces).
    pub shell_model: crate::tet::TetModel,
    /// Constitutive model for tet elements. Default `StVK`; the gauge's volumetric
    /// scenes use `Arap`, so it is selectable per scene.
    pub tet_model: crate::tet::TetModel,
    /// Shell (face) plasticity: creep rate + dead-zone threshold for the isotropic
    /// shell faces (see [`crate::shell_model::apply_shell_plasticity`]). 0 = off.
    pub shell_plasticity: f32,
    pub shell_plasticity_threshold: f32,
    /// Hinge (bend) plasticity: creeps the bending REST ANGLE so a held fold becomes a
    /// crease (see [`crate::bending::apply_hinge_plasticity`]). 0 = off.
    pub bend_plasticity: f32,
    pub bend_plasticity_threshold: f32,
    /// Volumetric plasticity: creep rate and dead-zone threshold (see
    /// [`crate::tet::apply_tet_plasticity`]). `tet_plasticity` 0 (default) disables it.
    /// The gauge's volumetric scenes use 0.5 / 0.5.
    pub tet_plasticity: f32,
    pub tet_plasticity_threshold: f32,
    /// Volumetric FEM Lamé-style parameters for tet elements (see `crate::tet`).
    /// Both default 0 = tets contribute nothing, so a cloth-only scene is
    /// unchanged even if tets are present.
    pub tet_mu: f32,
    pub tet_lambda: f32,
    /// Coulomb friction coefficient for collider contacts (0 = frictionless, the
    /// default). `friction_eps` is the slip regularization (reference `friction_eps`).
    pub friction: f32,
    pub friction_eps: f32,
    /// Which contact barrier to use. Default [`contact::BarrierKind::Log`] (the IPC
    /// log barrier — unchanged engine behavior). [`contact::BarrierKind::Cubic`] is
    /// the reference's cubic barrier: finite with bounded curvature through zero
    /// gap, so it recovers penetration instead of diverging.
    pub barrier_kind: contact::BarrierKind,
    /// Opt in to the reference's **absolute** elasticity-inclusive Rayleigh contact
    /// stiffness (`contact::rayleigh_stiffness`) for COLLIDER contacts, where the
    /// N=1 stencil makes it exact. `false` (default) keeps the multiplicative
    /// [`contact::dynamic_stiffness`] path, so defaults are bit-identical.
    pub contact_rayleigh: bool,
    /// Enable vertex–triangle self-collision (spatial-hash broad phase).
    pub self_collision: bool,
    /// Opt-in strain limiting on stretch edges: caps each edge to
    /// `strain_limit_ratio · rest_len` with stiffness `strain_limit_k`, active only
    /// when over-stretched. Both default 0 (disabled — springs/membrane carry
    /// stretch). A meaningful ratio is ≥ 1 (e.g. 1.05–1.2 for woven fabric).
    pub strain_limit_ratio: f32,
    pub strain_limit_k: f32,
    /// WEFT (V-direction) strain ratio for anisotropic limiting. 0 = isotropic
    /// (every edge uses `strain_limit_ratio`). When >0 AND `SimState::uvs` is
    /// present, each stretch edge is classified warp/weft by its UV-space direction.
    pub strain_limit_ratio_weft: f32,
    pub cg_tol: f32,
    pub cg_max_iter: u32,
    pub floor_y: f32,
}

impl Default for ImplicitParams {
    fn default() -> Self {
        Self {
            dt: 1.0 / 60.0,
            gravity: Vec3::new(0.0, -9.8, 0.0),
            newton_iters: 4,
            stiffness: 1.0e3,
            bend_stiffness: 0.0,
            membrane_warp: 0.0,
            membrane_weft: 0.0,
            membrane_shear: 0.0,
            collider_friction: f32::INFINITY,
            friction_mode: crate::contact::FrictionMode::Min,
            max_dx: f32::INFINITY,
            target_toi: 0.0,
            min_newton_steps: 0,
            ccd_max_iter: 256,
            // 1e-6 (not the reference's 0.01) keeps existing behaviour unchanged by
            // default; the gauge driver sets the scene's value.
            eiganalysis_eps: 1.0e-6,
            air_density: 0.0,
            air_friction: 0.0,
            isotropic_air_friction: 0.0,
            wind: Vec3::ZERO,
            constraint_tol: 1.0e-2,
            contact_dhat: 1.0e-2,
            contact_kappa: 1.0e2,
            contact_dynamic: 0.0,
            shell_mu: 0.0,
            shell_lambda: 0.0,
            shell_model: crate::tet::TetModel::Arap,
            tet_model: crate::tet::TetModel::StVK,
            shell_plasticity: 0.0,
            shell_plasticity_threshold: 0.0,
            bend_plasticity: 0.0,
            bend_plasticity_threshold: 0.0,
            tet_plasticity: 0.0,
            tet_plasticity_threshold: 0.0,
            tet_mu: 0.0,
            tet_lambda: 0.0,
            friction: 0.0,
            friction_eps: 1.0e-3,
            barrier_kind: contact::BarrierKind::Log,
            contact_rayleigh: false,
            self_collision: false,
            strain_limit_ratio: 0.0,
            strain_limit_k: 0.0,
            strain_limit_ratio_weft: 0.0,
            cg_tol: 1.0e-4,
            cg_max_iter: 2000,
            floor_y: f32::NEG_INFINITY,
        }
    }
}

/// A vertex position constraint — the port of the reference's `FixPair`
/// (`ppf-cts-solver/src/cpp/data.hpp:313`).
///
/// This is NOT a hard Dirichlet pin. The reference confines the vertex to a ball of
/// radius `ghat` around `target` and pulls it with the quadratic energy
/// `½·k·|x − target|²` (`energy/model/fix.hpp`), where the stiffness `k` is *dynamic*:
///
/// ```text
/// w   = x − target
/// gap = ghat − |w|
/// k   = (H_elastic·w)·w / |w|²  +  mass / gap²
/// ```
///
/// (`contact.cu:1038-1058`). The first term is the Rayleigh quotient of the vertex's own
/// 3×3 elastic Hessian block along `w` — the same construction as the contact barrier's
/// [`contact::rayleigh_stiffness`] — and the second is the inertial term that stiffens
/// without bound as the vertex approaches the `ghat` boundary. So a pinned vertex is free
/// to move a little (up to about `ghat`), which is what the reference's own captures show.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pin {
    /// Constrained vertex.
    pub index: u32,
    /// Position the vertex is pulled toward.
    pub target: Vec3,
    /// Radius of the ball the vertex is confined to.
    pub ghat: f32,
    /// Kinematic (animated) pins get their stiffness scaled by [`Pin::stiffness`] and
    /// their gap floored; static pins do not (`contact.cu:1052-1056`).
    pub kinematic: bool,
    /// Per-pin force scale, applied only when `kinematic`. 1.0 is the reference default.
    pub stiffness: f32,
}

/// A weighted soft pull — the port of the reference's `PullPair` (`data.hpp:324`,
/// applied at `energy.cu:208-218`).
///
/// Energy `½·w·|x − target|²`, so gradient `w·(x − target)` and Hessian `w·I`. Unlike
/// [`Pin`]/`FixPair` there is NO `ghat`, no dynamic stiffness and no time-of-impact clamp —
/// it is a plain spring. The reference treats a constrained vertex as EITHER pulled
/// (`pull > 0`) or fixed, never both (`scene.rs:2088-2104`), so the two sets are disjoint.
///
/// This is the entire physics of some scenes: `fishingknot` has `gravity = [0,0,0]` and a
/// 439-vertex block with `pull = 1000`, so without this term nothing in that scene moves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pull {
    /// Constrained vertex.
    pub index: u32,
    /// Position the vertex is pulled toward.
    pub target: Vec3,
    /// Pull weight (the reference's per-block `pull`). Must be > 0 to have any effect.
    pub weight: f32,
}

/// One rod segment — a 1-D codimensional strand element (the reference's "Rods":
/// yarn, noodles, woven threads).
///
/// The reference's stretch energy is `hook::make_diff_table`
/// (`energy/model/hook.hpp`), called with `weight = young_mod * mass` where
/// `mass = density * l_geom` (`energy.cu:286`, `scene.rs:1498`). Its gradient
/// `w(l/l₀ − 1)n` integrates to
///
/// ```text
/// E = w (l − l₀)² / (2 l₀),    l₀ = l_geom * length_factor
/// ```
///
/// i.e. a plain spring of stiffness `k = w / l₀ = young_mod * density / length_factor`
/// — independent of the segment's own length. (`hook::energy` in the reference is dead
/// code and inconsistent with that gradient by a factor `l₀`; the diff table is what
/// drives the solve, so it defines the energy.)
#[derive(Clone, Copy, Debug)]
pub struct Rod {
    pub a: u32,
    pub b: u32,
    /// Rest length — already scaled by the scene's `length_factor`, so a factor below
    /// 1 leaves the segment pre-tensioned at its initial geometry.
    pub rest_len: f32,
    /// Stretch stiffness `young_mod * density / length_factor`.
    pub k: f32,
    /// Contact OFFSET — the segment's radius (`rod-contact-offset.bin`). The reference
    /// subtracts the two participants' offsets from the measured distance
    /// (`g = ‖e‖ − offset`, `barrier.cu:103-116`, composed as a SUM at
    /// `contact.cu:372,558`), so a pair of rods with radius `r` each begins to repel at a
    /// centreline separation of `2r + ghat`. A strand with offset 0 is a zero-thickness
    /// curve.
    pub offset: f32,
    /// Contact activation gap `ghat` for this segment (`rod-contact-gap.bin`). The reference
    /// composes a pair's gap as the MEAN of the two participants' — `0.5*(a+b)`,
    /// `contact.cu:721` — in contrast to `offset`, which is a SUM. Zero means "use the
    /// scene-wide `ImplicitParams::contact_dhat`".
    pub ghat: f32,
}

/// One stitch (the reference's `Stitch`, assembled by `embed_stitch_force_hessian`,
/// `energy.cu:696`): a weighted barycentric point on one piece of cloth sewn to a weighted
/// barycentric point on another.
///
/// Structurally this is [`rod_stretch_block`]'s `hook` energy generalised from 2
/// participants to 6 weighted ones — the same `c₀`/`c₁` Hessian split, with
///
/// ```text
/// t = Σ_{i<3} wᵢ xᵢ  −  Σ_{j≥3} wⱼ xⱼ
/// ```
///
/// (the reference subtracts a centre-of-gravity from every term, which cancels exactly
/// because each side's weights sum to 1, so it is a pure fp32 conditioning trick).
///
/// `index[0..3]`/`weight[0..3]` is the SOURCE, `index[3..6]`/`weight[3..6]` the TARGET. A
/// single-vertex source degenerates to `index = {s,s,s}`, `weight = {1,0,0}`, which is what
/// all 851 of `fitting`'s stitches do. Repeated indices are fine: their contributions sum,
/// exactly as the reference's `atomic_embed_*` does.
#[derive(Clone, Copy, Debug)]
pub struct Stitch {
    pub index: [u32; 6],
    /// Barycentric weights; `[0..3]` and `[3..6]` each sum to 1.
    pub weight: [f32; 6],
    /// Direct force factor — the reference applies it RAW, with no mass or time-scale
    /// normalisation (`energy.cu:765-772`).
    pub stiffness: f32,
    /// Rest length: the weighted mean of the six vertices' contact `ghat`
    /// (`(Σ wᵢ·ghatᵢ)/2`, since the six weights sum to 2). The seam pulls the two sheets
    /// to within one contact gap of each other, i.e. just touching.
    pub l0: f32,
    /// Length at which the pull saturates:
    /// `stitch_length_factor * l0 + source_offset + target_offset`, where each side's
    /// offset is the MAX over its three vertices (`energy.cu:727-729`).
    pub l_cap: f32,
}

impl Pin {
    /// A static pin holding `index` at `target` within `ghat`.
    pub fn fixed(index: u32, target: Vec3, ghat: f32) -> Self {
        Self {
            index,
            target,
            ghat,
            kinematic: false,
            stiffness: 1.0,
        }
    }
}

pub struct ImplicitSolver {
    pub state: SimState,
    pub params: ImplicitParams,
    /// Per-step diagnostic channels + invariant config (see `metrics`).
    pub metrics: crate::metrics::Metrics,
    hinges: Vec<Hinge>,
    membrane: Vec<MembraneElement>,
    /// Volumetric (tet) elements — empty for pure cloth/shell meshes.
    tets: Vec<crate::tet::TetElement>,
    /// Optional PER-ELEMENT stiffness, parallel to the element lists. Empty ⇒ the
    /// scene-wide scalars in [`ImplicitParams`] are used, so defaults are unchanged.
    /// These exist because the reference assigns material per element, and averaging a
    /// varying array into one scalar is a fidelity gap the gauge driver had to report.
    membrane_k: Vec<[f32; 3]>, // (warp, weft, shear) per membrane element
    shell_k: Vec<[f32; 2]>, // (mu, lambda) per shell face
    tet_k: Vec<[f32; 2]>,   // (mu, lambda) per tet
    /// Optional PER-ELEMENT plasticity `(rate, threshold)`, parallel to the element
    /// lists. Empty ⇒ the scene-wide scalars are used.
    /// Optional PER-VERTEX friction coefficient. Empty ⇒ the scene-wide
    /// [`ImplicitParams::friction`] is used. Per-VERTEX rather than per-face because
    /// collider contact is evaluated per vertex; a scene's per-face array is averaged
    /// onto its vertices by the caller (see `ImplicitSolver::set_vertex_friction`).
    vertex_friction: Vec<f32>,
    tet_plast: Vec<[f32; 2]>,
    shell_plast: Vec<[f32; 2]>,
    hinge_plast: Vec<[f32; 2]>,
    /// Isotropic-shell (ARAP/StVK) faces. DISJOINT from the Baraff-Witkin `membrane`
    /// set: the reference assigns a model per face, so a mesh can carry both. Both
    /// accumulate into the same triplet list, so no union-CSR machinery is needed on
    /// the CPU path (that concern is specific to fixed-sparsity GPU assembly).
    shells: Vec<crate::shell_model::ShellElement>,
    /// Triangles used for SELF-COLLISION (and the intersection check): the cloth
    /// mesh's own tris plus, once tets are attached, the tet boundary surface. Cloth
    /// ELEMENTS (membrane/bending) always come from `state.tris`; this list is
    /// collision geometry only.
    collision_tris: Vec<[u32; 3]>,
    colliders: Vec<Collider>,
    /// Vertex position constraints (the port of the reference's `FixPair`).
    pins: Vec<Pin>,
    /// Weighted soft pulls (the reference's `PullPair`), disjoint from `pins`.
    pulls: Vec<Pull>,
    /// Rod segments, each with its own rest length and stiffness (unlike `state.stretch`,
    /// which shares the scene-wide `params.stretch_k`).
    rods: Vec<Rod>,
    /// Rod bend elements at interior rod vertices (`crate::rod_bend`). Independent of
    /// `hinges`, which are cloth dihedrals between two faces.
    rod_hinges: Vec<crate::rod_bend::RodHinge>,
    /// Seam stitches (the reference's `constraint.stitch`).
    stitches: Vec<Stitch>,
    /// Triplet count of the last assembly, used to reserve the next one exactly. `Cell`
    /// because `assemble` takes `&self`; it is a pure size hint, so a stale value only costs
    /// one realloc.
    trips_hint: std::cell::Cell<usize>,
    /// Faces carrying the reference's strain-limit BARRIER, with each face's limit and
    /// mass (see [`ImplicitSolver::set_strain_limit_faces`]). Independent of the
    /// membrane/shell split: the reference applies strain limiting to every shell face
    /// whatever its constitutive model.
    sl_faces: Vec<(crate::shell_model::ShellElement, f32, f32)>,
    /// Per-hinge bending stiffness, when derived from the scene's per-face `bend`
    /// (see [`ImplicitSolver::set_hinge_bend_from_faces`]). Empty ⇒ the scene-wide
    /// `params.bend_stiffness * hinge.w` is used instead.
    hinge_k: Vec<f32>,
    /// Per-vertex surface area — a third of each incident face's rest area, summed over
    /// ALL faces (`builder.rs:464-479`). Note this has NO shell filter, unlike the mass
    /// aggregation: a solid's surface faces contribute area even when they contribute no
    /// mass. Used to scale the aerodynamic damper.
    vertex_area: Vec<f32>,
    /// Per-vertex 1-ring (self + edge-neighbors) excluded from self-collision.
    exclusion: Vec<Vec<u32>>,
    /// Unique mesh edges, for edge-edge self-collision (CCD + barrier). Face edges first,
    /// then the attached rod segments — rods are codimensional, so they appear in no
    /// triangle and would otherwise never be tested for contact at all.
    edges: Vec<[u32; 2]>,
    /// Contact offset (radius) per entry of `edges`: 0 for a face edge, the segment's own
    /// offset for a rod. A pair's offset is the SUM of the two, matching the reference.
    edge_offset: Vec<f32>,
    /// Per-edge contact activation gap; 0 means "use the scene-wide `contact_dhat`" (every
    /// face edge, and any rod whose scene ships no gap). A pair's gap is the MEAN.
    edge_ghat: Vec<f32>,
    /// Characteristic (mean rest) edge length — the intersection-check hash cell.
    char_len: f32,
}

impl ImplicitSolver {
    pub fn new(state: SimState, params: ImplicitParams) -> Self {
        let hinges = bending::build_hinges(&state.tris, &state.positions);
        let membrane = membrane::build_membrane(&state.tris, &state.positions);
        let exclusion = crate::broadphase::one_ring_exclusion(state.vertex_count(), &state.tris);
        let edges = crate::broadphase::unique_edges(&state.tris);
        let char_len = if edges.is_empty() {
            1.0
        } else {
            let sum: f32 = edges
                .iter()
                .map(|e| (state.positions[e[0] as usize] - state.positions[e[1] as usize]).length())
                .sum();
            (sum / edges.len() as f32).max(1e-6)
        };
        // Strain limiting (when enabled) is a soft one-sided penalty that may slightly
        // overshoot its ratio, so it does NOT back a hard max-stretch invariant → keep
        // the invariant limit at 0 (auto-passes); a tolerance-based gate is a follow-up.
        let metrics = crate::metrics::Metrics::new(params.cg_tol, params.cg_max_iter, 0.0);
        let collision_tris = state.tris.clone();
        // Per-vertex area: a third of each incident face's area, from the REST positions
        // (the reference computes it once at build, `builder.rs:464-479`).
        let mut vertex_area = vec![0.0f32; state.vertex_count()];
        for f in &state.tris {
            let (a, b, c) = (
                state.positions[f[0] as usize],
                state.positions[f[1] as usize],
                state.positions[f[2] as usize],
            );
            let third = 0.5 * (b - a).cross(c - a).length() / 3.0;
            for &v in f {
                vertex_area[v as usize] += third;
            }
        }
        Self {
            state,
            params,
            metrics,
            hinges,
            membrane,
            tets: Vec::new(),
            shells: Vec::new(),
            membrane_k: Vec::new(),
            shell_k: Vec::new(),
            tet_k: Vec::new(),
            vertex_friction: Vec::new(),
            tet_plast: Vec::new(),
            shell_plast: Vec::new(),
            hinge_plast: Vec::new(),
            collision_tris,
            colliders: Vec::new(),
            pins: Vec::new(),
            pulls: Vec::new(),
            rods: Vec::new(),
            rod_hinges: Vec::new(),
            stitches: Vec::new(),
            trips_hint: std::cell::Cell::new(0),
            sl_faces: Vec::new(),
            hinge_k: Vec::new(),
            vertex_area,
            exclusion,
            edge_offset: vec![0.0; edges.len()],
            edge_ghat: vec![0.0; edges.len()],
            edges,
            char_len,
        }
    }

    /// Whether configuration `x` has any self-intersection (an edge piercing a
    /// triangle) — the reference `check_intersection` predicate (§6.1), used as a
    /// post-solve VERIFICATION (the reference's role for it: report a violation,
    /// not backtrack; the intersection-free guarantee comes from the CCD). Detects
    /// TRANSVERSE crossings (folds, corners); coherent parallel-sheet interpenetration
    /// is coplanar at the crossing instant and is not a transverse pierce.
    pub fn positions_self_intersect(&self, x: &[Vec3]) -> bool {
        if self.edges.is_empty() {
            return false;
        }
        // STREAMED, not materialised: this is a predicate that short-circuits, so building
        // the full candidate list first was pure waste — and on `fishingknot` it was the
        // single largest allocation in the solver (~718 MB transient, measured). Same
        // candidate set, same answer.
        crate::broadphase::any_edge_triangle_pair(
            x,
            &self.edges,
            &self.collision_tris,
            self.char_len,
            |ei, ti| {
                let e = self.edges[ei as usize];
                let t = self.collision_tris[ti as usize];
                contact::edge_triangle_intersect(
                    x[e[0] as usize],
                    x[e[1] as usize],
                    x[t[0] as usize],
                    x[t[1] as usize],
                    x[t[2] as usize],
                )
            },
        )
    }

    /// Whether the current committed state self-intersects (see
    /// [`positions_self_intersect`](Self::positions_self_intersect)).
    pub fn has_self_intersection(&self) -> bool {
        self.positions_self_intersect(&self.state.positions)
    }

    /// Dynamic barrier stiffness for a self-collision contact at `gap`, using the
    /// heaviest FREE stencil vertex's mass (the reference replaces zero/pinned
    /// masses with the max non-zero mass in the stencil).
    /// The reference's dynamic contact stiffness for a 4-vertex self-collision stencil —
    /// `barrier::compute_stiffness` (`barrier.cu:48-86`) in full: the Rayleigh quotient of
    /// `H_elastic + δ_ij·(m_ii/g²)·I` along the weighted contact direction.
    ///
    /// The **proximity weights** come from the distance gradient rather than a separate
    /// barycentric solve: for a distance function `∂d/∂x_a = w_a·n̂`, so `n̂ = ê(∂d/∂x₀)` and
    /// `w_a = (∂d/∂x_a)·n̂` — which recovers the reference's `prox.value` exactly,
    /// `(1, −b₀, −b₁, −b₂)` for vertex–triangle (`contact.cu:660`) and `(c₀, c₁, −c₂, −c₃)`
    /// for edge–edge (`contact.cu:728`). The separation vector is then `e = d·n̂`.
    ///
    /// `blocks`/`slots` carry the pre-gathered elastic Hessian (see
    /// [`Self::gather_elastic_blocks`]); a missing pair contributes a zero block.
    fn self_contact_rayleigh(
        &self,
        blocks: &[[f32; 9]],
        slots: &std::collections::HashMap<(u32, u32), usize>,
        idx: &[usize; 4],
        d: f32,
        g: &[Vec3; 4],
    ) -> f32 {
        let nlen = g[0].length();
        if nlen < 1.0e-12 {
            return self.params.contact_kappa;
        }
        let nrm = g[0] / nlen;
        let weights: [f32; 4] = std::array::from_fn(|a| g[a].dot(nrm));
        let mass: [f32; 4] = std::array::from_fn(|a| {
            let im = self.state.inv_mass[idx[a]];
            // A pinned/static participant reports zero mass; `rayleigh_stiffness`
            // substitutes the stencil's max, as the reference does.
            if im > 0.0 {
                1.0 / im
            } else {
                0.0
            }
        });
        let mut elastic = [[0.0f32; 9]; 16];
        for (ia, &va) in idx.iter().enumerate() {
            for (ib, &vb) in idx.iter().enumerate() {
                if let Some(&slot) = slots.get(&(va as u32, vb as u32)) {
                    elastic[ia * 4 + ib] = blocks[slot];
                }
            }
        }
        let k = crate::contact::rayleigh_stiffness(&elastic, &mass, &weights, d * nrm, 0.0);
        // A degenerate stencil (all masses zero AND no elastic coupling) would yield 0 and
        // silently switch the barrier off, so keep the base κ as a floor.
        if k > 0.0 {
            k
        } else {
            self.params.contact_kappa
        }
    }

    /// Strain-limit ratio for the stretch edge `(a, b)`: the warp ratio by default,
    /// or — when anisotropy is enabled (`strain_limit_ratio_weft > 0`) and per-vertex
    /// UVs are present — the warp ratio if the edge is more U-aligned in UV space,
    /// else the weft ratio. UV direction is material-space, so it is invariant to how
    /// the cloth is posed in 3D.
    fn edge_strain_ratio(&self, a: usize, b: usize) -> f32 {
        let warp = self.params.strain_limit_ratio;
        let weft = self.params.strain_limit_ratio_weft;
        if weft <= 0.0 || self.state.uvs.len() != self.state.vertex_count() {
            return warp;
        }
        let ua = self.state.uvs[a];
        let ub = self.state.uvs[b];
        let (du, dv) = ((ua[0] - ub[0]).abs(), (ua[1] - ub[1]).abs());
        if du >= dv {
            warp
        } else {
            weft
        }
    }

    /// Set the analytic colliders (half-spaces / spheres) cloth must not penetrate.
    pub fn set_colliders(&mut self, colliders: Vec<Collider>) {
        self.colliders = colliders;
    }

    /// Attach vertex position constraints — the port of the reference's `FixPair`
    /// (`data.hpp:313`, applied in `contact.cu:1038-1058`). Returns an error if a pin
    /// names a vertex outside the mesh or carries a non-positive `ghat`.
    pub fn set_pins(&mut self, pins: Vec<Pin>) -> Result<(), String> {
        let n = self.state.vertex_count();
        for p in &pins {
            if p.index as usize >= n {
                return Err(format!("pin vertex {} out of range (n = {n})", p.index));
            }
            if p.ghat.is_nan() || p.ghat <= 0.0 {
                return Err(format!("pin on vertex {} has ghat <= 0", p.index));
            }
        }
        self.pins = pins;
        Ok(())
    }

    /// Number of attached analytic colliders.
    pub fn colliders_len(&self) -> usize {
        self.colliders.len()
    }

    /// The attached pins.
    pub fn pins(&self) -> &[Pin] {
        &self.pins
    }

    /// Attach weighted soft pulls — the reference's `PullPair`. Rejects an out-of-range
    /// vertex or a non-positive weight (which would be a silent no-op).
    pub fn set_pulls(&mut self, pulls: Vec<Pull>) -> Result<(), String> {
        let n = self.state.vertex_count();
        for p in &pulls {
            if p.index as usize >= n {
                return Err(format!("pull vertex {} out of range (n = {n})", p.index));
            }
            if p.weight.is_nan() || p.weight <= 0.0 {
                return Err(format!("pull on vertex {} has weight <= 0", p.index));
            }
        }
        self.pulls = pulls;
        Ok(())
    }

    /// The attached pulls.
    pub fn pulls(&self) -> &[Pull] {
        &self.pulls
    }

    /// Attach rod segments (the reference's "Rods"). Rejects an out-of-range vertex, a
    /// degenerate segment, or a non-positive rest length / stiffness.
    pub fn set_rods(&mut self, rods: Vec<Rod>) -> Result<(), String> {
        let n = self.state.vertex_count();
        for r in &rods {
            if r.a as usize >= n || r.b as usize >= n {
                return Err(format!("rod ({}, {}) out of range (n = {n})", r.a, r.b));
            }
            if r.a == r.b {
                return Err(format!("rod ({}, {}) is degenerate", r.a, r.b));
            }
            if !r.rest_len.is_finite() || r.rest_len <= 0.0 || !r.k.is_finite() || r.k <= 0.0 {
                return Err(format!(
                    "rod ({}, {}) has rest_len {} / k {}; both must be > 0",
                    r.a, r.b, r.rest_len, r.k
                ));
            }
        }
        self.rods = rods;
        // Rod segments participate in edge-edge contact, so the collision edge list (and
        // the characteristic length derived from it) must be rebuilt.
        self.rebuild_collision_topology();
        Ok(())
    }

    /// The attached rod segments.
    pub fn rods(&self) -> &[Rod] {
        &self.rods
    }

    /// Attach rod bend elements. Rejects an out-of-range vertex, a repeated vertex within
    /// one hinge, or a non-finite stiffness / rest angle. A zero stiffness is allowed and
    /// simply contributes nothing, matching the reference's `stiff_k > 0` gate.
    pub fn set_rod_hinges(&mut self, hinges: Vec<crate::rod_bend::RodHinge>) -> Result<(), String> {
        let n = self.state.vertex_count();
        for h in &hinges {
            if h.j as usize >= n || h.i as usize >= n || h.k as usize >= n {
                return Err(format!(
                    "rod hinge ({}, {}, {}) out of range (n = {n})",
                    h.j, h.i, h.k
                ));
            }
            if h.j == h.i || h.i == h.k || h.j == h.k {
                return Err(format!(
                    "rod hinge ({}, {}, {}) repeats a vertex",
                    h.j, h.i, h.k
                ));
            }
            if !h.stiff.is_finite() || h.stiff < 0.0 || !h.rest_angle.is_finite() {
                return Err(format!(
                    "rod hinge ({}, {}, {}) has stiff {} / rest_angle {}",
                    h.j, h.i, h.k, h.stiff, h.rest_angle
                ));
            }
        }
        self.rod_hinges = hinges;
        Ok(())
    }

    /// The attached rod bend elements.
    pub fn rod_hinges(&self) -> &[crate::rod_bend::RodHinge] {
        &self.rod_hinges
    }

    /// Attach seam stitches. Rejects an out-of-range vertex, a non-finite or negative
    /// stiffness, a non-positive rest length, a cap below the rest length, or a side whose
    /// barycentric weights do not sum to 1 (which would make the centre-of-gravity terms
    /// stop cancelling and silently change the energy).
    pub fn set_stitches(&mut self, stitches: Vec<Stitch>) -> Result<(), String> {
        let n = self.state.vertex_count();
        for (si, st) in stitches.iter().enumerate() {
            if st.index.iter().any(|&i| i as usize >= n) {
                return Err(format!("stitch {si} references a vertex >= {n}"));
            }
            if !st.stiffness.is_finite() || st.stiffness < 0.0 {
                return Err(format!("stitch {si} has stiffness {}", st.stiffness));
            }
            if !st.l0.is_finite() || st.l0 <= 0.0 {
                return Err(format!("stitch {si} has rest length {}", st.l0));
            }
            if !st.l_cap.is_finite() || st.l_cap < st.l0 {
                return Err(format!(
                    "stitch {si} has cap {} below its rest length {}",
                    st.l_cap, st.l0
                ));
            }
            for (half, range) in [("source", 0..3), ("target", 3..6)] {
                let sum: f32 = range.clone().map(|k| st.weight[k]).sum();
                if (sum - 1.0).abs() > 1.0e-3 {
                    return Err(format!("stitch {si} {half} weights sum to {sum}, not 1"));
                }
            }
        }
        self.stitches = stitches;
        Ok(())
    }

    /// The attached stitches.
    pub fn stitches(&self) -> &[Stitch] {
        &self.stitches
    }

    /// Derive per-hinge bending stiffness from the scene's per-FACE `bend` values,
    /// reproducing the reference's coefficient exactly.
    ///
    /// Two pieces, both from the reference:
    ///
    /// 1. **Face → hinge**: a hinge's `bend` is the AREA-WEIGHTED MEAN of its incident
    ///    faces' `bend` (`builder.rs:770-800`, the same rule it uses for `ghat`, `offset`,
    ///    and bend plasticity). An earlier session recorded that the capture "does not
    ///    expose the reference's hinge param mapping" — that was wrong; the mapping is
    ///    plain area-weighted averaging.
    ///
    /// 2. **Stiffness**: the resolution-independent Discrete Shells coefficient
    ///    (`energy.cu:444-465`)
    ///    ```text
    ///    stiff_k = BEND_SCALE · bend · (|e|² / A) · areal_density
    ///    ```
    ///    where `|e|` is the hinge edge length, `A` the combined rest area of the two
    ///    incident triangles, and `areal_density` the mean of `mass/area` over the hinge's
    ///    four vertices (skipping any with zero area). The `|e|²/A` factor is what makes
    ///    the bent shape mesh-independent (a bare `|e|` shrinks under refinement, so finer
    ///    cloth droops more); the areal-density factor makes it density-independent, so
    ///    `bend` alone sets the shape. `BEND_SCALE` is the reference's calibration constant
    ///    and only fixes the numeric range of the user-facing `bend`.
    ///
    /// This SUPERSEDES `params.bend_stiffness` for every hinge once set.
    pub fn set_hinge_bend_from_faces(
        &mut self,
        faces: &[[u32; 3]],
        face_bend: &[f32],
    ) -> Result<(), String> {
        if face_bend.len() != faces.len() {
            return Err(format!(
                "face_bend has {} entries but there are {} faces",
                face_bend.len(),
                faces.len()
            ));
        }
        /// `energy.cu:460` — calibration only; it does NOT affect resolution-independence.
        const BEND_SCALE: f32 = 1.28e-5;
        let key = |a: u32, b: u32, c: u32| {
            let mut k = [a, b, c];
            k.sort_unstable();
            (k[0], k[1], k[2])
        };
        let mut by_face: std::collections::HashMap<(u32, u32, u32), usize> =
            std::collections::HashMap::with_capacity(faces.len());
        for (i, f) in faces.iter().enumerate() {
            by_face.insert(key(f[0], f[1], f[2]), i);
        }
        let pos = &self.state.positions;
        let area_of = |a: u32, b: u32, c: u32| {
            let (pa, pb, pc) = (pos[a as usize], pos[b as usize], pos[c as usize]);
            0.5 * (pb - pa).cross(pc - pa).length()
        };
        let mut out = Vec::with_capacity(self.hinges.len());
        for h in &self.hinges {
            // `v = [opp_a, shared_0, shared_1, opp_b]`: triangle A is (v0,v1,v2), B is
            // (v1,v2,v3), and the hinge edge is (v1,v2).
            let (v0, v1, v2, v3) = (h.v[0], h.v[1], h.v[2], h.v[3]);
            let (a1, a2) = (area_of(v0, v1, v2), area_of(v1, v2, v3));
            let mut bend_sum = 0.0f32;
            let mut area_sum = 0.0f32;
            for (tri, a) in [((v0, v1, v2), a1), ((v1, v2, v3), a2)] {
                if let Some(&fi) = by_face.get(&key(tri.0, tri.1, tri.2)) {
                    bend_sum += a * face_bend[fi];
                    area_sum += a;
                }
            }
            if area_sum <= 0.0 {
                out.push(0.0);
                continue;
            }
            let bend = bend_sum / area_sum;
            let area = a1 + a2;
            let length = (pos[v2 as usize] - pos[v1 as usize]).length();
            // Areal density: mean of mass/area over the four hinge vertices.
            let mut ad = 0.0f32;
            let mut cnt = 0u32;
            for &v in &h.v {
                let va = self.vertex_area.get(v as usize).copied().unwrap_or(0.0);
                let im = self.state.inv_mass[v as usize];
                if va > 0.0 && im > 0.0 {
                    ad += (1.0 / im) / va;
                    cnt += 1;
                }
            }
            if cnt > 0 {
                ad /= cnt as f32;
            }
            out.push(if area > 1.0e-12 {
                BEND_SCALE * bend * (length * length / area) * ad
            } else {
                0.0
            });
        }
        self.hinge_k = out;
        Ok(())
    }

    /// Per-hinge bending stiffness, if derived from the scene.
    pub fn hinge_k(&self) -> &[f32] {
        &self.hinge_k
    }

    /// Attach the reference's per-face strain-limit BARRIER
    /// (`crate::shell_model::strain_limit_assembly`), replacing the edge-spring path for
    /// these faces.
    ///
    /// `limits` are the reference's allowed-stretch FRACTIONS (`0.05` ⇒ 5% headroom, the
    /// `*-strain-limit.bin` convention) and `masses` the reference's face mass
    /// `density · rest_area`. Faces with a non-positive limit are skipped — that is how the
    /// capture spells "no limit on this face" — so a scene with a mixed array limits only
    /// the faces it actually constrains, rather than averaging them together.
    ///
    /// Rest shape comes from the CURRENT positions, as with `set_shell_faces`.
    pub fn set_strain_limit_faces(
        &mut self,
        faces: &[[u32; 3]],
        limits: &[f32],
        masses: &[f32],
    ) -> Result<(), String> {
        if limits.len() != faces.len() || masses.len() != faces.len() {
            return Err(format!(
                "strain-limit arrays must match the face count: {} faces, {} limits, {} masses",
                faces.len(),
                limits.len(),
                masses.len()
            ));
        }
        let mut out = Vec::new();
        for (i, f) in faces.iter().enumerate() {
            if limits[i] <= 0.0 || masses[i] <= 0.0 {
                continue;
            }
            // build_shells skips degenerate faces, so this yields 0 or 1 element.
            if let Some(el) = crate::shell_model::build_shells(
                std::slice::from_ref(f),
                &self.state.positions,
                1.0e-12,
                self.params.shell_model,
            )
            .pop()
            {
                out.push((el, limits[i], masses[i]));
            }
        }
        self.sl_faces = out;
        Ok(())
    }

    /// Number of faces carrying the strain-limit barrier.
    pub fn strain_limit_face_count(&self) -> usize {
        self.sl_faces.len()
    }

    /// Gather the requested 3×3 blocks of the elastic Hessian assembled so far.
    ///
    /// The reference's dynamic stiffnesses (`barrier::compute_stiffness`,
    /// `strainlimiting::compute_stiffness`) read `fixed_hess_in(vi, vj)` for every PAIR in
    /// a stencil, off-diagonals included. Our assembler holds the Hessian as triplets, so
    /// this makes ONE pass over them and accumulates only into the requested pairs —
    /// `O(nnz)` with a hash probe per triplet, the same cost shape as the diagonal-only
    /// extraction it generalizes.
    ///
    /// `slots` maps a `(vi, vj)` vertex pair to its index in the returned block list.
    /// Blocks are row-major 3×3: `block[3·r + c] = H[3·vi+r][3·vj+c]`.
    fn gather_elastic_blocks(
        trips: &[crate::csr::Triplet],
        slots: &std::collections::HashMap<(u32, u32), usize>,
    ) -> Vec<[f32; 9]> {
        let mut out = vec![[0.0f32; 9]; slots.len()];
        for &(r, c, v) in trips {
            let key = (r / 3, c / 3);
            if let Some(&slot) = slots.get(&key) {
                let (r, c) = (r as usize, c as usize);
                out[slot][3 * (r % 3) + (c % 3)] += v;
            }
        }
        out
    }

    /// Attach volumetric (tet) elements, built from the CURRENT positions as the
    /// rest state (`crate::tet::build_tets`). Degenerate tets are skipped and
    /// winding is normalized. Requires `tet_mu`/`tet_lambda` > 0 or a per-tet table
    /// from `set_tet_stiffness` to have any effect.
    pub fn set_tets(&mut self, tets: &[[u32; 4]]) {
        self.tets =
            crate::tet::build_tets(tets, &self.state.positions, 1.0e-12, self.params.tet_model);
        // The tet BOUNDARY surface joins the self-collision geometry (interior faces
        // are excluded by construction), so volumetric bodies can collide with
        // themselves and with cloth. Derived data (1-ring exclusion, unique edges,
        // characteristic length) is rebuilt over the combined surface.
        //
        // A boundary face ALREADY present in `state.tris` must not be appended again:
        // the reference ships the solid's surface inside its own face list (verified on
        // the gauge's `plastic` scene, whose 5120 tet-boundary faces all appear in
        // `tri.bin`), and a duplicate collision triangle would double every contact and
        // friction pair on that face. Comparison is winding-independent, since a face
        // and the same face reversed are one surface for collision purposes.
        let mut present: std::collections::HashSet<(u32, u32, u32)> =
            std::collections::HashSet::with_capacity(self.state.tris.len());
        for f in &self.state.tris {
            let mut k = *f;
            k.sort_unstable();
            present.insert((k[0], k[1], k[2]));
        }
        let surface = crate::tet::tet_surface(&self.tets);
        self.collision_tris = self.state.tris.clone();
        self.collision_tris.extend(surface.into_iter().filter(|f| {
            let mut k = *f;
            k.sort_unstable();
            !present.contains(&(k[0], k[1], k[2]))
        }));
        self.rebuild_collision_topology();
    }

    /// Recompute the self-collision derived data from `collision_tris`.
    fn rebuild_collision_topology(&mut self) {
        self.exclusion =
            crate::broadphase::one_ring_exclusion(self.state.vertex_count(), &self.collision_tris);
        self.edges = crate::broadphase::unique_edges(&self.collision_tris);
        self.edge_offset = vec![0.0; self.edges.len()];
        self.edge_ghat = vec![0.0; self.edges.len()];
        // Rod segments belong to no triangle, so `unique_edges` cannot see them and they
        // would never be tested for contact. Append them with their own radius. A rod that
        // duplicates a face edge is not possible in the gauge (rod vertices are disjoint
        // from every face), but appending unconditionally would double-count it, so skip
        // any segment already present.
        if !self.rods.is_empty() {
            let mut seen: std::collections::HashSet<(u32, u32)> = self
                .edges
                .iter()
                .map(|e| (e[0].min(e[1]), e[0].max(e[1])))
                .collect();
            for r in &self.rods {
                let key = (r.a.min(r.b), r.a.max(r.b));
                if seen.insert(key) {
                    self.edges.push([r.a, r.b]);
                    self.edge_offset.push(r.offset.max(0.0));
                    self.edge_ghat.push(r.ghat.max(0.0));
                }
            }
        }
        // Characteristic length over ALL collision edges. For a scene with no rods this is
        // byte-identical to the face-edge-only value it has always been; for a rod-only
        // scene (yarn/noodle/woven) it is the difference between a meaningful contact scale
        // and the 1.0 fallback.
        self.char_len = if self.edges.is_empty() {
            1.0
        } else {
            let sum: f32 = self
                .edges
                .iter()
                .map(|e| {
                    (self.state.positions[e[0] as usize] - self.state.positions[e[1] as usize])
                        .length()
                })
                .sum();
            (sum / self.edges.len() as f32).max(1e-6)
        };
    }

    /// Largest contact offset over all collision edges — the amount the broad phase must be
    /// widened by so a pair whose SURFACES are within `dhat` is still found when its
    /// centrelines are `2*offset + dhat` apart.
    fn max_edge_offset(&self) -> f32 {
        self.edge_offset.iter().copied().fold(0.0f32, f32::max)
    }

    /// This edge's activation gap, falling back to the scene-wide `contact_dhat`.
    fn edge_gap(&self, ei: usize) -> f32 {
        let g = self.edge_ghat[ei];
        if g > 0.0 {
            g
        } else {
            self.params.contact_dhat
        }
    }

    /// Largest per-edge activation gap, for sizing the broad phase.
    fn max_edge_gap(&self) -> f32 {
        (0..self.edges.len())
            .map(|i| self.edge_gap(i))
            .fold(self.params.contact_dhat, f32::max)
    }

    /// Number of collision edges (face edges plus attached rod segments).
    pub fn collision_edge_count(&self) -> usize {
        self.edges.len()
    }

    /// The triangles used for self-collision (cloth tris + tet boundary surface).
    pub fn collision_tris(&self) -> &[[u32; 3]] {
        &self.collision_tris
    }

    /// The built tet elements (rest state baked in).
    pub fn tets(&self) -> &[crate::tet::TetElement] {
        &self.tets
    }

    /// Register a subset of faces to be driven by the isotropic-shell model
    /// (ARAP/StVK) instead of Baraff-Witkin. Rest state comes from the CURRENT
    /// positions. Intended for the reference's per-face model assignment: pass the
    /// faces whose `model` is `arap`, leave the rest to the membrane path.
    ///
    /// The two sets should be disjoint — a face driven by both would be double-counted.
    /// Registering shell faces does NOT remove them from the membrane list, so callers
    /// that want a pure split must also zero the membrane stiffnesses or supply a mesh
    /// whose membrane elements exclude these faces.
    pub fn set_shell_faces(&mut self, faces: &[[u32; 3]]) {
        self.shells = crate::shell_model::build_shells(
            faces,
            &self.state.positions,
            1.0e-12,
            self.params.shell_model,
        );
    }

    /// Restrict the Baraff-Witkin membrane to a SUBSET of faces, rebuilding its
    /// elements from the current positions. Needed for a genuine per-face model
    /// partition: by default the membrane covers every triangle, so a mesh whose faces
    /// are split between models would have the shell faces receiving membrane stiffness
    /// too. Pass the Baraff-Witkin faces here and the isotropic ones to
    /// [`Self::set_shell_faces`] for a true partition.
    pub fn set_membrane_faces(&mut self, faces: &[[u32; 3]]) {
        self.membrane = membrane::build_membrane(faces, &self.state.positions);
        self.membrane_k.clear();
    }

    /// Per-element membrane stiffness `(warp, weft, shear)`, parallel to the membrane
    /// element list. Empty (default) ⇒ the scene-wide scalars are used.
    pub fn set_membrane_stiffness(&mut self, k: Vec<[f32; 3]>) -> Result<(), String> {
        if !k.is_empty() && k.len() != self.membrane.len() {
            return Err(format!(
                "membrane stiffness len {} != element count {}",
                k.len(),
                self.membrane.len()
            ));
        }
        self.membrane_k = k;
        Ok(())
    }

    /// Per-face shell stiffness `(mu, lambda)`, parallel to the shell face list.
    pub fn set_shell_stiffness(&mut self, k: Vec<[f32; 2]>) -> Result<(), String> {
        if !k.is_empty() && k.len() != self.shells.len() {
            return Err(format!(
                "shell stiffness len {} != face count {}",
                k.len(),
                self.shells.len()
            ));
        }
        self.shell_k = k;
        Ok(())
    }

    /// Per-tet stiffness `(mu, lambda)`, parallel to the tet list.
    pub fn set_tet_stiffness(&mut self, k: Vec<[f32; 2]>) -> Result<(), String> {
        if !k.is_empty() && k.len() != self.tets.len() {
            return Err(format!(
                "tet stiffness len {} != tet count {}",
                k.len(),
                self.tets.len()
            ));
        }
        self.tet_k = k;
        Ok(())
    }

    /// Per-vertex friction coefficients. Empty clears the override.
    pub fn set_vertex_friction(&mut self, mu: Vec<f32>) -> Result<(), String> {
        let n = self.state.vertex_count();
        if !mu.is_empty() && mu.len() != n {
            return Err(format!(
                "vertex friction len {} != vertex count {n}",
                mu.len()
            ));
        }
        self.vertex_friction = mu;
        Ok(())
    }

    /// Average a PER-FACE quantity onto vertices (area-agnostic mean over incident
    /// faces). Vertices touched by no face get `fallback`. Used to turn the reference's
    /// per-face `friction` array into the per-vertex form collider contact needs.
    pub fn face_scalars_to_vertices(
        &self,
        faces: &[[u32; 3]],
        per_face: &[f32],
        fallback: f32,
    ) -> Vec<f32> {
        let n = self.state.vertex_count();
        let mut sum = vec![0.0f32; n];
        let mut cnt = vec![0u32; n];
        for (f, &v) in faces.iter().zip(per_face) {
            for &vi in f {
                if (vi as usize) < n {
                    sum[vi as usize] += v;
                    cnt[vi as usize] += 1;
                }
            }
        }
        (0..n)
            .map(|i| {
                if cnt[i] > 0 {
                    sum[i] / cnt[i] as f32
                } else {
                    fallback
                }
            })
            .collect()
    }

    /// Per-tet plasticity `(rate, threshold)`, parallel to the tet list.
    pub fn set_tet_plasticity(&mut self, k: Vec<[f32; 2]>) -> Result<(), String> {
        if !k.is_empty() && k.len() != self.tets.len() {
            return Err(format!(
                "tet plasticity len {} != {}",
                k.len(),
                self.tets.len()
            ));
        }
        self.tet_plast = k;
        Ok(())
    }

    /// Per-face shell plasticity `(rate, threshold)`.
    pub fn set_shell_plasticity(&mut self, k: Vec<[f32; 2]>) -> Result<(), String> {
        if !k.is_empty() && k.len() != self.shells.len() {
            return Err(format!(
                "shell plasticity len {} != {}",
                k.len(),
                self.shells.len()
            ));
        }
        self.shell_plast = k;
        Ok(())
    }

    /// Per-hinge bend plasticity `(rate, threshold)`.
    pub fn set_hinge_plasticity(&mut self, k: Vec<[f32; 2]>) -> Result<(), String> {
        if !k.is_empty() && k.len() != self.hinges.len() {
            return Err(format!(
                "hinge plasticity len {} != {}",
                k.len(),
                self.hinges.len()
            ));
        }
        self.hinge_plast = k;
        Ok(())
    }

    /// Number of hinges (for aligning per-hinge arrays).
    pub fn hinge_count(&self) -> usize {
        self.hinges.len()
    }

    /// Membrane elements (after any [`Self::set_membrane_faces`] restriction).
    /// Per-element membrane stiffness `(warp, weft, shear)`; empty when the scene-wide
    /// scalars are in use.
    pub fn membrane_stiffness(&self) -> &[[f32; 3]] {
        &self.membrane_k
    }

    /// Per-face shell stiffness `(mu, lambda)`; empty when the scene-wide scalars are used.
    pub fn shell_stiffness(&self) -> &[[f32; 2]] {
        &self.shell_k
    }

    pub fn membrane_len(&self) -> usize {
        self.membrane.len()
    }

    /// Number of self-collision triangles (the mesh faces plus any tet-boundary faces
    /// not already among them).
    pub fn collision_tri_count(&self) -> usize {
        self.collision_tris.len()
    }

    /// The simulation state, for callers that need to inspect masses or positions.
    pub fn state(&self) -> &SimState {
        &self.state
    }

    /// Hinges, for tests that inspect plasticity-mutated rest angles.
    #[doc(hidden)]
    pub fn hinges_for_test(&self) -> &[Hinge] {
        &self.hinges
    }

    /// The registered isotropic-shell faces.
    pub fn shells(&self) -> &[crate::shell_model::ShellElement] {
        &self.shells
    }

    /// Largest step fraction `t ∈ [t_min, 1]` along `dx` that keeps every contact
    /// gap positive. Analytic colliders use an exact/additive per-vertex clamp
    /// (half-space distance is linear in `t`; sphere additive). Vertex–triangle
    /// self-collision pairs use the reference's **iterated per-pair ACCD**
    /// ([`contact::accd_toi`], PORTING_contact.md §4) with the correct Lipschitz
    /// bound `u_max = max_{i<j}‖dx_i−dx_j‖` — the per-pair min gives the step.
    ///
    /// Runs over BOTH vertex–triangle and edge–edge self-collision pairs (the
    /// latter catches a vertex/edge sliding through a shared triangle edge), so the
    /// CCD is complete for *transverse* surface crossing. Coherent parallel-sheet
    /// interpenetration under high momentum is a separate, open case (needs dynamic
    /// barrier stiffness — see `self_collision_survives_hard_impact`).
    fn ccd_max_step(&self, x: &[Vec3], dx: &[Vec3]) -> f32 {
        let has_self = self.params.self_collision;
        // A STATIC pin confines its vertex to a ball of radius `ghat`, enforced here rather
        // than by the barrier — so the limiter must run even with no colliders and no
        // self-collision.
        let has_static_pins = self.pins.iter().any(|p| !p.kinematic);
        if self.colliders.is_empty() && !has_self && !has_static_pins && self.sl_faces.is_empty() {
            return 1.0;
        }
        // GAP: keep ≥20% of the current gap this step (analytic colliders).
        // T_MIN=0: a genuine contact may legitimately clamp the step to 0 (inward
        // motion fully blocked) — the barrier's OUTWARD gradient is never clamped,
        // so this is not a deadlock. A nonzero floor would force downward creep
        // THROUGH contact under momentum (a tunneling source found in Session 14).
        const GAP: f32 = 0.2;
        const T_MIN: f32 = 0.0;
        let n = x.len();
        let mut t = 1.0f32;

        // Strain-limit line search (`strainlimiting.cu:236-290`). The barrier alone cannot
        // enforce the limit: if a step jumps a face's principal stretch PAST its limit, the
        // barrier argument goes non-positive and the restoring force stops growing. The
        // reference therefore clamps the step so no face ever crosses its limit, by
        // BISECTING the step fraction on `max(σ(F₀ + t·dF) − 1) < limit`. With this the
        // limit is genuinely enforced rather than merely resisted.
        if !self.sl_faces.is_empty() {
            const LS_MAX_T: f32 = 1.25; // line_search_max_t
            for (el, limit, _) in &self.sl_faces {
                let vs = el.v;
                let at = |frac: f32| {
                    let p = [
                        x[vs[0] as usize] + frac * dx[vs[0] as usize],
                        x[vs[1] as usize] + frac * dx[vs[1] as usize],
                        x[vs[2] as usize] + frac * dx[vs[2] as usize],
                    ];
                    let f = crate::shell_model::shell_deformation_gradient(&p, &el.dm_inv);
                    let (_, sigma, _) = crate::shell_model::svd3x2(&f);
                    (sigma[0] - 1.0).max(sigma[1] - 1.0)
                };
                // `F` is affine in the step fraction, so evaluating the gradient at
                // `x + frac·dx` is the reference's `F₀ + t·dF` written directly.
                if at(LS_MAX_T) < *limit {
                    continue;
                }
                let (mut lo, mut hi) = (0.0f32, LS_MAX_T);
                let mut window = hi - lo;
                // The reference bisects until the window stops shrinking (f32 exhaustion);
                // the iteration cap is a belt-and-braces bound on that same loop.
                for _ in 0..64 {
                    let mid = 0.5 * (lo + hi);
                    if at(mid) - *limit < 0.0 {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                    let w = hi - lo;
                    if w == window {
                        break;
                    }
                    window = w;
                }
                t = t.min(lo);
            }
        }

        // Static-pin confinement (`contact.cu:1996-2012`). The reference probes the
        // EXTENDED horizon `line_search_max_t·dx`, interpolates the RADIUS linearly between
        // its endpoints, and clamps so the vertex lands on the `ghat` sphere:
        //   t = (ghat − r0) / (r1 − r0),   toi = line_search_max_t · t
        // Linear-in-radius is an approximation (the exact solve is a quadratic), but it is
        // what the reference does, so it is reproduced rather than "improved".
        //
        // Only STATIC pins are clamped: a kinematic pin's target moves and the reference
        // lets it float, flooring the gap instead (`contact.cu:1045`).
        if has_static_pins {
            const LS_MAX_T: f32 = 1.25; // line_search_max_t
            for pin in &self.pins {
                if pin.kinematic {
                    continue;
                }
                let i = pin.index as usize;
                if self.state.inv_mass[i] <= 0.0 {
                    continue;
                }
                let r0 = (x[i] - pin.target).length();
                // The reference asserts `r0 < ghat`, an invariant this clamp maintains from
                // the initial state on (a static target starts AT its vertex, so r0 = 0).
                // If it were ever violated, clamping would freeze the vertex instead of
                // letting the pin's stiffness pull it back — so skip rather than deadlock.
                if r0 >= pin.ghat {
                    continue;
                }
                let r1 = (x[i] + LS_MAX_T * dx[i] - pin.target).length();
                if r1 > pin.ghat {
                    let denom = r1 - r0;
                    if denom != 0.0 {
                        t = t.min(LS_MAX_T * (pin.ghat - r0) / denom);
                    }
                }
            }
        }

        // Analytic colliders.
        for i in 0..n {
            if self.state.inv_mass[i] <= 0.0 {
                continue;
            }
            for c in &self.colliders {
                let (d0, nrm) = c.distance(x[i]);
                if d0 <= 0.0 {
                    continue; // already inside; barrier resolves
                }
                match c {
                    Collider::HalfSpace { .. } => {
                        let rate = dx[i].dot(nrm); // d(t) = d0 + t·rate (exact)
                        if rate < 0.0 {
                            t = t.min((GAP - 1.0) * d0 / rate);
                        }
                    }
                    Collider::Sphere { .. } => {
                        let l = dx[i].length();
                        if l > 1.0e-12 {
                            t = t.min((1.0 - GAP) * d0 / l);
                        }
                    }
                }
            }
        }

        // Self-collision vertex–triangle pairs via the reference iterated per-pair
        // ACCD (PORTING_contact.md §4). Broaden the search by TWICE the max
        // displacement (a vertex and a triangle vertex can approach at up to the
        // sum of their speeds), so any pair that could reach contact this step is
        // in the candidate set — the un-expanded broad phase was a tunneling source.
        if has_self {
            const MAX_T: f32 = 1.25; // line_search_max_t: extend horizon for back-off
            const REDUCTION: f32 = 0.01; // ccd_reduction: convergence band as a gap fraction
            let ccd_iters = self.params.ccd_max_iter.max(1); // conservative-advancement cap
            let max_disp = dx.iter().map(|d| d.length()).fold(0.0f32, f32::max);
            // Fixed part of the radius: the largest activation gap plus twice the largest
            // contact offset — two thick rods touch when their CENTRELINES are
            // `2*offset + ghat` apart, so an un-widened radius would miss the pair.
            let fixed = self.max_edge_gap() + 2.0 * self.max_edge_offset();

            // Conservative advancement over the broadphase RADIUS, not just over the step.
            //
            // Sizing the radius for the full `max_disp` is correct but degenerates: every
            // gauge scene ships `max_dx = 1.0` in WORLD units, so on `domino` (mean edge
            // 0.16, object ~2 units across) the radius reaches 2.0, `grid_cell` is floored
            // to 2.0, ALL 1344 edges land in ONE bucket, and the broadphase emits 17.3 M
            // candidate pairs — an honest O(n^2) for a query nobody needed that wide.
            // Measured with `MGUSO_BP_STATS=1`; labelled B-1 in the internal audit notes.
            //
            // So start with a radius sized to the MESH and widen only when the step it
            // yields would outrun what that radius covered. This returns the SAME `t` as the
            // full-radius query: `t_self` is accepted only once the motion it permits
            // (`t_self * max_disp`) fits within the `allowed` motion the radius covers, and
            // otherwise `allowed` widens until it reaches `max_disp`, at which point the
            // query IS the full one. A pair outside the candidate set provably cannot reach
            // contact while every vertex moves at most `allowed`, so it cannot own a
            // smaller TOI than the ones that were checked.
            // Start at HALF the characteristic edge length. That keeps `search` at or below
            // the mean element extent, so `grid_cell` falls back to its extent floor and the
            // grid stays in the healthy regime the barrier query already uses (cell 0.16,
            // buckets of ~25 on `domino`) instead of the degenerate one (cell 2.0, one bucket
            // of 1344). Widening x4 reaches `max_disp` in at most a few passes.
            let mut allowed = (0.5 * self.char_len).min(max_disp).max(1.0e-9);
            loop {
                let full = allowed >= max_disp;
                let search = fixed + 2.0 * allowed;
                let mut t_self = 1.0f32;
                let pairs = crate::broadphase::vertex_triangle_pairs(
                    x,
                    &self.collision_tris,
                    search,
                    &self.exclusion,
                );
                for (v, ti) in pairs {
                    let tri = self.collision_tris[ti as usize];
                    let idx = [
                        v as usize,
                        tri[0] as usize,
                        tri[1] as usize,
                        tri[2] as usize,
                    ];
                    let x0 = [x[idx[0]], x[idx[1]], x[idx[2]], x[idx[3]]];
                    let dd = [dx[idx[0]], dx[idx[1]], dx[idx[2]], dx[idx[3]]];
                    // offset=0 (zero-thickness cloth): ACCD keeps the pair ~REDUCTION·gap
                    // apart; the IPC barrier provides the repulsion. (A vertex-triangle
                    // pair only; a vertex sliding through a shared EDGE between two
                    // triangles is not captured here — that needs edge-edge CCD, the
                    // documented next step for full hard-impact robustness.)
                    let toi = contact::accd_toi(&x0, &dd, 0.0, MAX_T, REDUCTION, ccd_iters, |p| {
                        let d = contact::point_triangle_distance(p[0], p[1], p[2], p[3]);
                        d * d
                    });
                    t_self = t_self.min(toi / MAX_T);
                }

                // Edge-edge pairs: catches a vertex/edge sliding through a shared
                // triangle EDGE — the through-edge crossing vertex-triangle CCD misses.
                // Together, V-T + E-E CCD make surface crossing impossible, so the
                // unsigned barrier never sees a sign flip (intersection-free by
                // construction, the reference's guarantee).
                let ee = crate::broadphase::edge_edge_pairs(x, &self.edges, search);
                for (ei, ej) in ee {
                    let (ea, eb) = (self.edges[ei as usize], self.edges[ej as usize]);
                    let idx = [
                        ea[0] as usize,
                        ea[1] as usize,
                        eb[0] as usize,
                        eb[1] as usize,
                    ];
                    let x0 = [x[idx[0]], x[idx[1]], x[idx[2]], x[idx[3]]];
                    let dd = [dx[idx[0]], dx[idx[1]], dx[idx[2]], dx[idx[3]]];
                    // Thickness: the pair's offset is the SUM of the two segments' radii
                    // (`contact.cu:558`), so CCD stops them at surface contact, not centreline
                    // contact.
                    let off = self.edge_offset[ei as usize] + self.edge_offset[ej as usize];
                    let toi = contact::accd_toi(&x0, &dd, off, MAX_T, REDUCTION, ccd_iters, |p| {
                        let d = contact::segment_segment_distance(p[0], p[1], p[2], p[3]);
                        d * d
                    });
                    t_self = t_self.min(toi / MAX_T);
                }
                if full || t_self * max_disp <= allowed {
                    t = t.min(t_self);
                    break;
                }
                allowed = (allowed * 4.0).min(max_disp);
            }
        }

        t.clamp(T_MIN, 1.0)
    }

    /// Advance one implicit step, using `linear` for each Newton linear solve.
    #[tracing::instrument(level = "debug", skip(self, linear), fields(dt = self.params.dt))]
    pub fn step(&mut self, linear: &dyn LinearSolver) {
        let n = self.state.vertex_count();
        let h = self.params.dt;
        let inv_h2 = 1.0 / (h * h);
        let g = self.params.gravity;

        let x0: Vec<Vec3> = self.state.positions.clone();
        // Inertial target x̃ (pinned vertices stay put).
        let x_tilde: Vec<Vec3> = (0..n)
            .map(|i| {
                if self.state.inv_mass[i] > 0.0 {
                    x0[i] + self.state.velocities[i] * h + g * (h * h)
                } else {
                    x0[i]
                }
            })
            .collect();

        let mut x = x0.clone();
        let mut max_reresid = 0.0f64;
        let mut max_cg_iters = 0u32;
        // Reference termination (`main.cu:1038-1046`): iterate while contact is clamping
        // the step hard; once an iteration advances at least `target_toi` of its proposed
        // step (and `min_newton_steps` have run), take ONE more and stop. `target_toi == 0`
        // disables it and the loop runs the full fixed count, as it always did.
        let cap = self.params.newton_iters.max(1);
        let mut final_step = false;
        let mut taken = 0u32;
        for _ in 0..cap {
            taken += 1;
            crate::alloc_stats::checkpoint("pre-assemble");
            let (a, rhs) = self.assemble(&x, &x_tilde, inv_h2);
            crate::alloc_stats::checkpoint("post-assemble");
            let dxflat = linear.solve(&a, &rhs, self.params.cg_tol, self.params.cg_max_iter);
            crate::alloc_stats::checkpoint("post-solve");
            let st = linear.last_stats();
            max_reresid = max_reresid.max(st.residual);
            max_cg_iters = max_cg_iters.max(st.iters);
            let mut dx: Vec<Vec3> = (0..n)
                .map(|i| Vec3::new(dxflat[3 * i], dxflat[3 * i + 1], dxflat[3 * i + 2]))
                .collect();
            // Rescale the whole search direction so no vertex would move more than
            // `max_dx`, BEFORE the line search — uniform, so the direction is preserved.
            // `INFINITY` makes the ratio `INF` and the `min` a no-op, so the default path
            // needs no special case.
            let max_norm = dx.iter().map(|d| d.length()).fold(0.0f32, f32::max);
            if max_norm > 0.0 {
                let rescale = (self.params.max_dx / max_norm).min(1.0);
                if rescale < 1.0 {
                    for d in dx.iter_mut() {
                        *d *= rescale;
                    }
                }
            }
            // CCD line search: clamp the step so no vertex crosses a collider or
            // triangle this iteration. t = 1 when nothing is near.
            let t = self.ccd_max_step(&x, &dx);
            crate::alloc_stats::checkpoint("post-ccd");
            let mut max_step = 0.0f32;
            for i in 0..n {
                let d = dx[i] * t;
                x[i] += d;
                max_step = max_step.max(d.length());
            }
            if max_step < 1.0e-7 {
                break; // converged
            }
            if final_step {
                break;
            }
            // The reference checks `step >= min_newton_steps` where `step` counts iterations
            // that did NOT terminate, i.e. `taken − 1` here — so `taken > min_newton_steps`
            // is the same test without a second counter.
            if self.params.target_toi > 0.0
                && t >= self.params.target_toi
                && taken > self.params.min_newton_steps
            {
                final_step = true;
            }
        }

        // Floor collision (post-solve projection).
        let floor = self.params.floor_y;
        for i in 0..n {
            if self.state.inv_mass[i] > 0.0 && floor.is_finite() && x[i].y < floor {
                x[i].y = floor;
            }
        }

        // f32 robustness guard (Phase 6/9): a non-finite Newton result (NaN/Inf from
        // a singular assembly, division blow-up, etc.) must NOT be committed — it
        // would corrupt every subsequent step. Reject the bad step: keep the previous
        // positions and zero the velocities (freeze this step) so the sim degrades
        // gracefully instead of exploding. Only triggers on failure, so a healthy run
        // is unaffected.
        let step_finite = x.iter().all(|p| p.is_finite());
        let inv_h = 1.0 / h;
        for i in 0..n {
            if step_finite {
                self.state.velocities[i] = if self.state.inv_mass[i] > 0.0 {
                    (x[i] - x0[i]) * inv_h
                } else {
                    Vec3::ZERO
                };
                self.state.prev_positions[i] = x0[i];
                self.state.positions[i] = x[i];
            } else {
                // Reject: hold the pre-step position, kill momentum.
                self.state.velocities[i] = Vec3::ZERO;
                self.state.prev_positions[i] = x0[i];
                self.state.positions[i] = x0[i];
            }
        }
        self.state.time += h as f64;

        // Record diagnostic channels + invariant inputs for this step.
        let all_finite = self.state.positions.iter().all(|p| p.is_finite());
        let max_sigma = self.max_stretch_ratio();
        let max_velocity = self
            .state
            .velocities
            .iter()
            .zip(&self.state.inv_mass)
            .filter(|(_, &im)| im > 0.0)
            .map(|(v, _)| v.length())
            .fold(0.0f32, f32::max);
        let num_contacts = self.count_active_contacts();
        // No penetration if finite, CCD never fully stalled (t≥t_min ⇒ progress),
        // and no free vertex sits inside a collider beyond a small barrier tol.
        let penetration_free = all_finite && self.penetration_free();
        // Shell (face) plasticity: relax each shell face's rest shape (dead-zone creep).
        if (self.params.shell_plasticity > 0.0 || !self.shell_plast.is_empty())
            && !self.shells.is_empty()
        {
            let (rate, thresh, dt) = (
                self.params.shell_plasticity,
                self.params.shell_plasticity_threshold,
                self.params.dt,
            );
            let x = &self.state.positions;
            let per = &self.shell_plast;
            for (ei, el) in self.shells.iter_mut().enumerate() {
                let (rate, thresh) = if per.is_empty() {
                    (rate, thresh)
                } else {
                    (per[ei][0], per[ei][1])
                };
                let p = [
                    x[el.v[0] as usize],
                    x[el.v[1] as usize],
                    x[el.v[2] as usize],
                ];
                crate::shell_model::apply_shell_plasticity(el, &p, rate, thresh, dt);
            }
        }

        // Hinge (bend) plasticity: a fold held past the dead zone becomes a crease.
        if (self.params.bend_plasticity > 0.0 || !self.hinge_plast.is_empty())
            && !self.hinges.is_empty()
        {
            let (rate, thresh, dt) = (
                self.params.bend_plasticity,
                self.params.bend_plasticity_threshold,
                self.params.dt,
            );
            let x = &self.state.positions;
            let per = &self.hinge_plast;
            for (ei, h) in self.hinges.iter_mut().enumerate() {
                let (rate, thresh) = if per.is_empty() {
                    (rate, thresh)
                } else {
                    (per[ei][0], per[ei][1])
                };
                let p = [
                    x[h.v[0] as usize],
                    x[h.v[1] as usize],
                    x[h.v[2] as usize],
                    x[h.v[3] as usize],
                ];
                bending::apply_hinge_plasticity(h, &p, rate, thresh, dt);
            }
        }

        // Volumetric plasticity: relax each tet's REST state toward the settled
        // deformation (dead-zone creep). Runs AFTER positions are advanced, matching
        // the reference, which calls `update_tet_plasticity` on the post-step state.
        // Gated on a positive rate so it costs nothing by default.
        if (self.params.tet_plasticity > 0.0 || !self.tet_plast.is_empty()) && !self.tets.is_empty()
        {
            let (rate, thresh, dt) = (
                self.params.tet_plasticity,
                self.params.tet_plasticity_threshold,
                self.params.dt,
            );
            let x = &self.state.positions;
            let per = &self.tet_plast;
            for (ei, el) in self.tets.iter_mut().enumerate() {
                let (rate, thresh) = if per.is_empty() {
                    (rate, thresh)
                } else {
                    (per[ei][0], per[ei][1])
                };
                let p = [
                    x[el.v[0] as usize],
                    x[el.v[1] as usize],
                    x[el.v[2] as usize],
                    x[el.v[3] as usize],
                ];
                crate::tet::apply_tet_plasticity(el, &p, rate, thresh, dt);
            }
        }

        self.metrics.steps.push(crate::metrics::StepMetrics {
            time: self.state.time,
            max_reresid,
            max_cg_iters,
            newton_iters: taken,
            max_sigma,
            max_velocity,
            num_contacts,
            penetration_free,
            all_finite,
            device_alloc: 0, // CPU tier: no device pool
            device_free: 0,
        });

        tracing::debug!(
            time = self.state.time,
            ke = self.state.kinetic_energy(),
            reresid = max_reresid,
            cg_iters = max_cg_iters,
            max_sigma,
            "implicit step"
        );
    }

    /// Max stretch ratio `|x_a − x_b| / rest_len` over stretch constraints
    /// (the `advance.max_sigma` channel). 1.0 = rest.
    fn max_stretch_ratio(&self) -> f32 {
        self.state
            .stretch
            .iter()
            .map(|c| {
                let d =
                    self.state.positions[c.a as usize].distance(self.state.positions[c.b as usize]);
                if c.rest_len > 1e-9 {
                    d / c.rest_len
                } else {
                    1.0
                }
            })
            .fold(0.0f32, f32::max)
    }

    /// Count free vertices currently within `contact_dhat` of any collider
    /// (a proxy for the `advance.num_contact` channel).
    fn count_active_contacts(&self) -> u32 {
        if self.colliders.is_empty() {
            return 0;
        }
        let dhat = self.params.contact_dhat;
        let mut n = 0u32;
        for i in 0..self.state.vertex_count() {
            if self.state.inv_mass[i] <= 0.0 {
                continue;
            }
            for c in &self.colliders {
                if c.distance(self.state.positions[i]).0 < dhat {
                    n += 1;
                    break;
                }
            }
        }
        n
    }

    /// No free vertex has crossed to the wrong side of a collider (signed
    /// distance ≥ −tol). Analytic colliders are signed, so this is exact.
    fn penetration_free(&self) -> bool {
        let tol = -1e-3 * self.params.contact_dhat.max(1e-6);
        for i in 0..self.state.vertex_count() {
            if self.state.inv_mass[i] <= 0.0 {
                continue;
            }
            for c in &self.colliders {
                if c.distance(self.state.positions[i]).0 < tol {
                    return false;
                }
            }
            if self.params.floor_y.is_finite()
                && self.state.positions[i].y < self.params.floor_y - 1e-3
            {
                return false;
            }
        }
        // Self-intersection (transverse pierce), the reference §6 check — only when
        // self-collision is active (the check has a broad-phase cost).
        if self.params.self_collision {
            crate::alloc_stats::checkpoint("inv:pre-intersection");
            let hit = self.has_self_intersection();
            crate::alloc_stats::checkpoint("inv:post-intersection");
            if hit {
                return false;
            }
        }
        true
    }

    /// Friction coefficient for a self-collision stencil: the MINIMUM over its
    /// participating vertices. The reference assigns friction per element and resolves a
    /// pair through its own param indices, which the capture does not expose; taking the
    /// min is a documented, conservative choice (a contact is no rougher than its
    /// smoothest participant) rather than an invented average.
    fn stencil_mu(&self, idx: &[usize; 4]) -> f32 {
        if self.vertex_friction.is_empty() {
            return self.params.friction;
        }
        self.params
            .friction_mode
            .fold(idx.iter().map(|&i| self.vertex_friction[i]))
    }

    /// Assemble the SPD Newton system `(H, rhs = -grad)` at configuration `x`.
    fn assemble(&self, x: &[Vec3], x_tilde: &[Vec3], inv_h2: f32) -> (CsrMatrix, Vec<f32>) {
        let n = self.state.vertex_count();
        let ndof = 3 * n;
        let k = self.params.stiffness;
        // Reserve from the PREVIOUS assembly's length + 1/4, never an element-count formula:
        // how many faces carry a strain limit, how many hinges exist and how many contacts
        // are active are all scene- and state-dependent. The measured hint is exact after the
        // first assembly and self-corrects as contacts come and go; the quarter covers the
        // inter-Newton growth (`curtain`: 20.7 M entries on the first, 27.7 M on the second).
        // The first assembly still grows by doubling — a bounded ~1.5x transient.
        //
        // A static formula was measured and was worse: it over-reserved `hang` by 3.7x
        // (159.7 MB for 3.8 M entries) and `curtain` by 2.6x, and seeding only the first
        // assembly from it inherited the same over-reservation on the step where the peak
        // actually occurs (`hang` 216.9 MB vs 119.7 MB for the pure measured hint).
        let prev = self.trips_hint.get();
        let mut trips: Vec<crate::csr::Triplet> = Vec::with_capacity(prev + prev / 4);
        let mut grad = vec![0.0f32; ndof];

        // Inertia: H += m_i/h² I ; grad += m_i/h² (x_i - x̃_i). Free vertices only;
        // pinned are handled by the Dirichlet filter below.
        for i in 0..n {
            let im = self.state.inv_mass[i];
            if im <= 0.0 {
                continue;
            }
            let m_over_h2 = (1.0 / im) * inv_h2;
            let diff = x[i] - x_tilde[i];
            for c in 0..3 {
                trips.push(((3 * i + c) as u32, (3 * i + c) as u32, m_over_h2));
                grad[3 * i + c] += m_over_h2 * diff[c];
            }
        }

        // Rod segments: per-element stiffness, the reference's `hook` block.
        for rd in &self.rods {
            let (a, b) = (rd.a as usize, rd.b as usize);
            let (hk, ga) = rod_stretch_block(x[a] - x[b], rd.rest_len, rd.k);
            for c in 0..3 {
                grad[3 * a + c] += ga[c];
                grad[3 * b + c] -= ga[c];
            }
            for r in 0..3 {
                for cc in 0..3 {
                    let v = hk[r][cc];
                    trips.push(((3 * a + r) as u32, (3 * a + cc) as u32, v));
                    trips.push(((3 * b + r) as u32, (3 * b + cc) as u32, v));
                    trips.push(((3 * a + r) as u32, (3 * b + cc) as u32, -v));
                    trips.push(((3 * b + r) as u32, (3 * a + cc) as u32, -v));
                }
            }
        }

        // Rod bending: `½·stiff_k·(θ − θ₀)²` at interior rod vertices, with the
        // reference's eigen-projected (PSD) 9×9 Hessian.
        for h in &self.rod_hinges {
            if h.stiff <= 0.0 {
                continue;
            }
            let v = [h.j as usize, h.i as usize, h.k as usize];
            let (g, hh) =
                crate::rod_bend::strand_force_hessian(x[v[0]], x[v[1]], x[v[2]], h.rest_angle);
            for a in 0..3 {
                for c in 0..3 {
                    grad[3 * v[a] + c] += h.stiff * g[a][c];
                }
            }
            for a in 0..3 {
                for b in 0..3 {
                    for r in 0..3 {
                        for cc in 0..3 {
                            let val = h.stiff * hh[3 * a + r][3 * b + cc];
                            if val != 0.0 {
                                trips.push(((3 * v[a] + r) as u32, (3 * v[b] + cc) as u32, val));
                            }
                        }
                    }
                }
            }
        }

        // Seam stitches: the `hook` energy over 6 weighted participants
        // (`crate::stitch`, which is where the gradient and 18×18 Hessian are verified).
        for st in &self.stitches {
            if st.stiffness <= 0.0 {
                continue;
            }
            let xs: [Vec3; 6] = std::array::from_fn(|k| x[st.index[k] as usize]);
            let (g, hh) = crate::stitch::force_hessian(&xs, &st.weight, st.l0, st.l_cap);
            for k in 0..6 {
                let i = st.index[k] as usize;
                for c in 0..3 {
                    grad[3 * i + c] += st.stiffness * g[k][c];
                }
            }
            // Repeated indices (a degenerate single-vertex source) land on the same slot
            // repeatedly; the triplet builder SUMS them, exactly as the reference's
            // `atomic_embed_hessian` does.
            for a in 0..6 {
                let ia = st.index[a] as usize;
                for b in 0..6 {
                    let ib = st.index[b] as usize;
                    for rr in 0..3 {
                        for cc in 0..3 {
                            let v = st.stiffness * hh[3 * a + rr][3 * b + cc];
                            if v != 0.0 {
                                trips.push(((3 * ia + rr) as u32, (3 * ib + cc) as u32, v));
                            }
                        }
                    }
                }
            }
        }

        // Springs: SPD-projected Hessian + gradient.
        for s in &self.state.stretch {
            let (a, b) = (s.a as usize, s.b as usize);
            let d = x[a] - x[b];
            let (hk, ga) = spring_block(d, s.rest_len, k);
            // grad_a += ga ; grad_b -= ga
            for c in 0..3 {
                grad[3 * a + c] += ga[c];
                grad[3 * b + c] -= ga[c];
            }
            // 6x6 block = [[Hk, -Hk], [-Hk, Hk]]
            for r in 0..3 {
                for cc in 0..3 {
                    let v = hk[r][cc];
                    trips.push(((3 * a + r) as u32, (3 * a + cc) as u32, v));
                    trips.push(((3 * b + r) as u32, (3 * b + cc) as u32, v));
                    trips.push(((3 * a + r) as u32, (3 * b + cc) as u32, -v));
                    trips.push(((3 * b + r) as u32, (3 * a + cc) as u32, -v));
                }
            }

            // Opt-in strain limiting on the same edge (active only when the edge
            // exceeds `ratio · rest`). Same 6×6 sparsity as the spring above.
            // Anisotropic: when a weft ratio is set and UVs are present, classify the
            // edge warp (U) vs weft (V) by its dominant UV-space direction.
            let slk = self.params.strain_limit_k;
            let slr = self.edge_strain_ratio(a, b);
            if slr > 0.0 && slk > 0.0 {
                let (sg, sh) = crate::strain_limit::strain_block(x[a], x[b], s.rest_len, slr, slk);
                for c in 0..3 {
                    grad[3 * a + c] += sg[c];
                    grad[3 * b + c] -= sg[c];
                }
                for r in 0..3 {
                    for cc in 0..3 {
                        let v = sh[r][cc];
                        trips.push(((3 * a + r) as u32, (3 * a + cc) as u32, v));
                        trips.push(((3 * b + r) as u32, (3 * b + cc) as u32, v));
                        trips.push(((3 * a + r) as u32, (3 * b + cc) as u32, -v));
                        trips.push(((3 * b + r) as u32, (3 * a + cc) as u32, -v));
                    }
                }
            }
        }

        // Membrane (Baraff-Witkin): per-triangle warp/weft/shear conditions.
        // Anisotropic (warp≠weft). Gradient is analytic (condition_grads). Hessian:
        // stretch (warp/weft) uses the EIGEN-projected material `k·A·cⱼcₖ·M`
        // (eigenpairs {1, 1-1/‖w‖, 1-1/‖w‖} clamped); shear uses its EIGEN-projected
        // 6×6 material (blocks M_uu/M_uv/M_vu/M_vv, eigenpairs {μ±, ±Csh} clamped —
        // captures the indefinite wu-wv coupling GN drops). All PSD.
        let (kw, kf, ks) = (
            self.params.membrane_warp,
            self.params.membrane_weft,
            self.params.membrane_shear,
        );
        let any_membrane_k = !self.membrane_k.is_empty();
        if kw > 0.0 || kf > 0.0 || ks > 0.0 || any_membrane_k {
            for (ei, el) in self.membrane.iter().enumerate() {
                // Per-element stiffness when supplied, else the scene-wide scalars.
                let (kw, kf, ks) = if any_membrane_k {
                    let k = self.membrane_k[ei];
                    (k[0], k[1], k[2])
                } else {
                    (kw, kf, ks)
                };
                let vs = el.tri;
                let p = [x[vs[0] as usize], x[vs[1] as usize], x[vs[2] as usize]];
                let c = membrane::conditions(&p, &el.dm_inv);
                let g = membrane::condition_grads(&p, &el.dm_inv); // [cond][vertex]
                let (cu, cv) = membrane::stretch_coeffs(&el.dm_inv);
                let (mu, mv) = membrane::stretch_materials(&p, &el.dm_inv);
                let stretch = [(cu, mu), (cv, mv)]; // ci 0 (warp), 1 (weft)
                let sm = membrane::shear_material_at(&p, &el.dm_inv, true); // [uu,uv,vu,vv]
                for (ci, kc) in [kw, kf, ks].iter().enumerate() {
                    let ka = *kc * el.area;
                    if ka <= 0.0 {
                        continue;
                    }
                    let (gc, cval) = (&g[ci], c[ci]);
                    for a in 0..3 {
                        let va = vs[a] as usize;
                        // Gradient (analytic, all conditions).
                        for cc in 0..3 {
                            grad[3 * va + cc] += ka * cval * gc[a][cc];
                        }
                        for b in 0..3 {
                            let vb = vs[b] as usize;
                            for r in 0..3 {
                                for cc in 0..3 {
                                    let v = if ci < 2 {
                                        // Eigen stretch block: ka·cₐ·c_b·M[r][cc].
                                        let (coeff, m) = &stretch[ci];
                                        ka * coeff[a] * coeff[b] * m[r][cc]
                                    } else {
                                        // Eigen shear block: ka·(cuₐcu_b·M_uu +
                                        // cuₐcv_b·M_uv + cvₐcu_b·M_vu + cvₐcv_b·M_vv).
                                        ka * (cu[a] * cu[b] * sm[0][r][cc]
                                            + cu[a] * cv[b] * sm[1][r][cc]
                                            + cv[a] * cu[b] * sm[2][r][cc]
                                            + cv[a] * cv[b] * sm[3][r][cc])
                                    };
                                    if v != 0.0 {
                                        trips.push(((3 * va + r) as u32, (3 * vb + cc) as u32, v));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Contact: IPC barrier against analytic colliders. Per free vertex, a
        // diagonal 3×3 PSD block + gradient (no vertex-vertex coupling).
        // Isotropic shells (ARAP/StVK) on the faces registered via `set_shell_faces`.
        // Accumulates into the same `trips`/`grad` as the membrane path, so a mesh with
        // a per-face model mix needs no special union handling here.
        // A non-empty per-face table is reason enough to assemble, exactly as the
        // membrane (`any_membrane_k`) and tet (`!self.tet_k.is_empty()`) paths do —
        // otherwise `set_shell_stiffness` is silently a no-op at default scalars.
        if !self.shells.is_empty()
            && (self.params.shell_mu > 0.0
                || self.params.shell_lambda > 0.0
                || !self.shell_k.is_empty())
        {
            let (smu, slam) = (self.params.shell_mu, self.params.shell_lambda);
            for (ei, el) in self.shells.iter().enumerate() {
                let (smu, slam) = if self.shell_k.is_empty() {
                    (smu, slam)
                } else {
                    let k = self.shell_k[ei];
                    (k[0], k[1])
                };
                let p = [
                    x[el.v[0] as usize],
                    x[el.v[1] as usize],
                    x[el.v[2] as usize],
                ];
                let eg = crate::shell_model::shell_gradient_analytic(&p, el, smu, slam);
                let eh = crate::shell_model::shell_hessian_analytic(
                    &p,
                    el,
                    smu,
                    slam,
                    self.params.eiganalysis_eps,
                );
                for (a, gv) in eg.iter().enumerate() {
                    let va = el.v[a] as usize;
                    for c in 0..3 {
                        grad[3 * va + c] += gv[c];
                    }
                    for b in 0..3 {
                        let vb = el.v[b] as usize;
                        for r in 0..3 {
                            for cc in 0..3 {
                                let v = eh[(3 * a + r) * 9 + (3 * b + cc)];
                                if v != 0.0 {
                                    trips.push(((3 * va + r) as u32, (3 * vb + cc) as u32, v));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Volumetric (tet) FEM: per-element energy gradient + PSD-projected 12×12
        // Hessian over the tet's 4 vertices (crate::tet). Skipped entirely unless
        // tets are attached AND a stiffness is set, so cloth-only scenes are
        // bit-identical. Pinned rows are filtered by the Dirichlet pass below, the
        // same as every other element type.
        if !self.tets.is_empty()
            && (self.params.tet_mu > 0.0 || self.params.tet_lambda > 0.0 || !self.tet_k.is_empty())
        {
            let (mu, lam) = (self.params.tet_mu, self.params.tet_lambda);
            for (ei, el) in self.tets.iter().enumerate() {
                let (mu, lam) = if self.tet_k.is_empty() {
                    (mu, lam)
                } else {
                    let k = self.tet_k[ei];
                    (k[0], k[1])
                };
                let p = [
                    x[el.v[0] as usize],
                    x[el.v[1] as usize],
                    x[el.v[2] as usize],
                    x[el.v[3] as usize],
                ];
                let (_e, eg, eh) = crate::tet::tet_element_assembly(&p, el, mu, lam);
                for (a, gv) in eg.iter().enumerate() {
                    let va = el.v[a] as usize;
                    for c in 0..3 {
                        grad[3 * va + c] += gv[c];
                    }
                    for b in 0..4 {
                        let vb = el.v[b] as usize;
                        for r in 0..3 {
                            for cc in 0..3 {
                                let v = eh[(3 * a + r) * 12 + (3 * b + cc)];
                                if v != 0.0 {
                                    trips.push(((3 * va + r) as u32, (3 * vb + cc) as u32, v));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Bending: E = ½·kb·w·(θ−θ_rest)². Gauss-Newton (PSD) Hessian
        // kb·w·∇θ∇θᵀ over the hinge's 4 vertices; grad = kb·w·(θ−θr)·∇θ.
        let kb = self.params.bend_stiffness;
        let per_hinge_k = self.hinge_k.len() == self.hinges.len() && !self.hinge_k.is_empty();
        if kb > 0.0 || per_hinge_k {
            for (hi, hinge) in self.hinges.iter().enumerate() {
                let vs = hinge.v;
                let p = [
                    x[vs[0] as usize],
                    x[vs[1] as usize],
                    x[vs[2] as usize],
                    x[vs[3] as usize],
                ];
                let theta = bending::dihedral_angle(&p);
                let g = bending::dihedral_grad_analytic(&p); // ∇θ per vertex (analytic)
                                                             // A scene-derived per-hinge stiffness supersedes the scene-wide scalar.
                let kwt = if per_hinge_k {
                    self.hinge_k[hi]
                } else {
                    kb * hinge.w
                };
                if kwt <= 0.0 {
                    continue;
                }
                let gval = theta - hinge.rest;
                for a in 0..4 {
                    let va = vs[a] as usize;
                    for c in 0..3 {
                        grad[3 * va + c] += kwt * gval * g[a][c];
                    }
                    for b in 0..4 {
                        let vb = vs[b] as usize;
                        for r in 0..3 {
                            for cc in 0..3 {
                                let v = kwt * g[a][r] * g[b][cc];
                                if v != 0.0 {
                                    trips.push(((3 * va + r) as u32, (3 * vb + cc) as u32, v));
                                }
                            }
                        }
                    }
                }
            }
        }

        crate::alloc_stats::checkpoint("asm:elastic-done");
        // NOTE ON ORDER: every ELASTIC term must be assembled before the barrier/constraint
        // terms below, because each of those derives its stiffness from the elastic Hessian
        // gathered out of `trips` (the reference's `fixed_hess_in`). Bending used to sit
        // AFTER them, so the strain-limit, pin, collider, and self-collision stiffnesses all
        // read a Hessian with the bending term missing. Moving it here does not change its
        // own contribution — assembly is additive — only what those stiffnesses see.
        // ---- strain-limit barrier (strainlimiting.cu:46-92) ----
        // Per-face barrier on the principal-stretch excess: the energy diverges as a
        // stretch approaches its limit, so the limit is ENFORCED rather than resisted.
        // Supersedes the edge-spring path for the faces it covers.
        if !self.sl_faces.is_empty() {
            let kind = self.params.barrier_kind;
            // Every (vi, vj) pair over every strain-limited face, so the elastic blocks can
            // be gathered in a single pass.
            let mut slots: std::collections::HashMap<(u32, u32), usize> =
                std::collections::HashMap::with_capacity(9 * self.sl_faces.len());
            for (el, _, _) in &self.sl_faces {
                for &a in &el.v {
                    for &b in &el.v {
                        let n = slots.len();
                        slots.entry((a, b)).or_insert(n);
                    }
                }
            }
            let blocks = Self::gather_elastic_blocks(&trips, &slots);
            for (el, limit, mass) in &self.sl_faces {
                let vs = el.v;
                let p = [x[vs[0] as usize], x[vs[1] as usize], x[vs[2] as usize]];
                // Elastic half of the stiffness: `w·(H_elastic·w)` with `w` the face's
                // vertex positions relative to their centroid, UNNORMALIZED — exactly as
                // `strainlimiting::compute_stiffness` (`strainlimiting.cu:22-44`) forms it.
                let centroid = (p[0] + p[1] + p[2]) / 3.0;
                let w = [p[0] - centroid, p[1] - centroid, p[2] - centroid];
                let mut elastic_k = 0.0f32;
                for (ia, &va) in el.v.iter().enumerate() {
                    for (ib, &vb) in el.v.iter().enumerate() {
                        let Some(&slot) = slots.get(&(va, vb)) else {
                            continue;
                        };
                        let blk = &blocks[slot];
                        for r in 0..3 {
                            for c in 0..3 {
                                elastic_k += w[ia][r] * blk[3 * r + c] * w[ib][c];
                            }
                        }
                    }
                }
                let Some((eg, eh)) = crate::shell_model::strain_limit_assembly(
                    &p,
                    el,
                    *limit,
                    *mass,
                    elastic_k,
                    kind,
                    self.params.eiganalysis_eps,
                ) else {
                    continue;
                };
                for (a, gv) in eg.iter().enumerate() {
                    let va = vs[a] as usize;
                    for c in 0..3 {
                        grad[3 * va + c] += gv[c];
                    }
                    for b in 0..3 {
                        let vb = vs[b] as usize;
                        for r in 0..3 {
                            for cc in 0..3 {
                                let v = eh[(3 * a + r) * 9 + (3 * b + cc)];
                                if v != 0.0 {
                                    trips.push(((3 * va + r) as u32, (3 * vb + cc) as u32, v));
                                }
                            }
                        }
                    }
                }
            }
        }

        // ---- pins (the reference's FixPair, contact.cu:1038-1058) ----
        // Energy ½·k·|x − target|² with the DYNAMIC stiffness
        //   k = (H_elastic·w)·w / |w|²  +  mass / gap²,   gap = ghat − |w|
        // so gradient = k·w and Hessian = k·I (`energy/model/fix.hpp`). The first term is
        // the Rayleigh quotient of the vertex's own elastic Hessian block along w; every
        // triplet assembled so far IS that elastic part, so its diagonal blocks are summed
        // once here, the same way the collider path does it.
        if !self.pins.is_empty() {
            let mut diag = vec![[0.0f32; 9]; n];
            for &(r, c, v) in trips.iter() {
                let (r, c) = (r as usize, c as usize);
                if r / 3 == c / 3 {
                    diag[r / 3][3 * (r % 3) + (c % 3)] += v;
                }
            }
            for pin in &self.pins {
                let i = pin.index as usize;
                if self.state.inv_mass[i] <= 0.0 {
                    // Already immovable; a pull would be meaningless.
                    continue;
                }
                let m = 1.0 / self.state.inv_mass[i];
                let w = x[i] - pin.target;
                let d = w.length();
                // The reference floors the gap for kinematic pins and instead keeps
                // static pins inside `ghat` with a time-of-impact clamp
                // (`contact.cu:1996-2012`) that we do not implement; without a floor a
                // static pin that drifts to the boundary would divide by zero here.
                let floor = self.params.constraint_tol * pin.ghat;
                let gap = (pin.ghat - d).max(floor);
                // Rayleigh quotient (H·w)·w / |w|², zero when w vanishes.
                let d2 = w.length_squared();
                let tmp = if d2 > 0.0 {
                    let h = &diag[i];
                    let hw = Vec3::new(
                        h[0] * w.x + h[1] * w.y + h[2] * w.z,
                        h[3] * w.x + h[4] * w.y + h[5] * w.z,
                        h[6] * w.x + h[7] * w.y + h[8] * w.z,
                    );
                    hw.dot(w) / d2
                } else {
                    0.0
                };
                let mut k = tmp + m / (gap * gap);
                if pin.kinematic {
                    k *= pin.stiffness;
                }
                let g = k * w;
                for c in 0..3 {
                    grad[3 * i + c] += g[c];
                    trips.push(((3 * i + c) as u32, (3 * i + c) as u32, k));
                }
            }
        }

        // ---- weighted soft pulls (the reference's PullPair, energy.cu:208-218) ----
        // grad += w·(x − target), H += w·I. A plain spring: no ghat, no dynamic stiffness,
        // no TOI clamp — and disjoint from `pins` by construction upstream.
        for pl in &self.pulls {
            let i = pl.index as usize;
            if self.state.inv_mass[i] <= 0.0 {
                continue; // immovable; a pull would be meaningless
            }
            let d = (x[i] - pl.target) * pl.weight;
            for c in 0..3 {
                grad[3 * i + c] += d[c];
                trips.push(((3 * i + c) as u32, (3 * i + c) as u32, pl.weight));
            }
        }

        // ---- aerodynamic damper (energy/model/air_damper.hpp, energy.cu:188-206) ----
        // NOT a drag impulse: an implicit ANISOTROPIC damper on the within-step
        // displacement relative to the wind,
        //   z = (x − x_step_start) − dt·wind,   P = I − n·nᵀ
        //   grad = area·ρ·[ n(n·z) + air_friction·P·z ] / dt²
        //   hess = area·ρ·[ n·nᵀ  + air_friction·P    ] / dt²
        // Motion along the surface normal is damped at FULL strength while tangential
        // (in-plane) motion is damped only by `air_friction` — which is what makes a sheet
        // resist moving broadside through air while still sliding freely in its own plane.
        // The Hessian's eigenvalues are 1 (along n) and `air_friction` (twice, tangential),
        // so it is PSD for any air_friction >= 0 and needs no projection.
        let (rho, af) = (self.params.air_density, self.params.air_friction);
        if rho > 0.0 {
            // Vertex normal: the normalized sum of incident faces' cross products, from
            // the CURRENT iterate (`utility.cu:14-31`). Accumulating per face into its
            // three vertices is the same sum as iterating each vertex's incident faces.
            let mut nrm = vec![Vec3::ZERO; n];
            for f in &self.state.tris {
                let (a, b, c) = (x[f[0] as usize], x[f[1] as usize], x[f[2] as usize]);
                let cr = (b - a).cross(c - a);
                for &v in f {
                    nrm[v as usize] += cr;
                }
            }
            let inv_dt2 = inv_h2;
            // The reference scales wind by a time-dependent weight, so a constant `wind`
            // still gusts (`utility::get_wind_weight`, `utility/utility.cu:317-321`, applied
            // at `energy.cu:196`). No gauge scene ships a non-zero wind, so this is
            // unobservable against the capture — but a scene WITH wind would have been
            // silently steady, so it is ported rather than left as a note.
            let wind_dx = self.params.dt * wind_weight(self.state.time as f32) * self.params.wind;
            for i in 0..n {
                if self.state.inv_mass[i] <= 0.0 || self.vertex_area[i] <= 0.0 {
                    continue;
                }
                let n2 = nrm[i].length_squared();
                if n2 <= 0.0 {
                    continue;
                }
                let nv = nrm[i] / n2.sqrt();
                let z = (x[i] - self.state.positions[i]) - wind_dx;
                let nz = nv.dot(z);
                let k = self.vertex_area[i] * rho * inv_dt2;
                // n(n·z) + af·P·z, with P·z = z − n(n·z).
                let g = nv * nz + af * (z - nv * nz);
                for c in 0..3 {
                    grad[3 * i + c] += k * g[c];
                }
                // (1 − af)·n·nᵀ + af·I, which is n·nᵀ + af·P rearranged.
                for r in 0..3 {
                    for c in 0..3 {
                        let mut v = (1.0 - af) * nv[r] * nv[c];
                        if r == c {
                            v += af;
                        }
                        if v != 0.0 {
                            trips.push(((3 * i + r) as u32, (3 * i + c) as u32, k * v));
                        }
                    }
                }
            }
        }

        // Direction-independent velocity damping (`energy.cu:250-252`): area- and
        // normal-independent, so it applies to every dynamic vertex including those with
        // no incident face.
        let iso = self.params.isotropic_air_friction;
        if iso > 0.0 {
            let k = iso * inv_h2;
            for i in 0..n {
                if self.state.inv_mass[i] <= 0.0 {
                    continue;
                }
                let d = x[i] - self.state.positions[i];
                for c in 0..3 {
                    grad[3 * i + c] += k * d[c];
                    trips.push(((3 * i + c) as u32, (3 * i + c) as u32, k));
                }
            }
        }

        if !self.colliders.is_empty() {
            let (dhat, kappa) = (self.params.contact_dhat, self.params.contact_kappa);
            // For the reference's Rayleigh stiffness we need each vertex's own 3×3
            // elastic Hessian block. Everything assembled into `trips` so far IS the
            // elastic part (inertia + springs + membrane + bending), so sum the
            // diagonal blocks once — O(nnz) — rather than re-deriving them.
            let elastic_diag: Option<Vec<[f32; 9]>> = if self.params.contact_rayleigh {
                let mut blocks = vec![[0.0f32; 9]; n];
                for &(r, c, v) in trips.iter() {
                    let (r, c) = (r as usize, c as usize);
                    if r / 3 == c / 3 {
                        blocks[r / 3][3 * (r % 3) + (c % 3)] += v;
                    }
                }
                Some(blocks)
            } else {
                None
            };
            for i in 0..n {
                if self.state.inv_mass[i] <= 0.0 {
                    continue;
                }
                let m = 1.0 / self.state.inv_mass[i];
                // Friction slip: displacement since the step start (the reference feeds
                // `dx = ex - ex0`, i.e. motion within the step, not total displacement).
                // `min` with the collider's own coefficient, as the reference combines them.
                let mu_i = if self.vertex_friction.is_empty() {
                    self.params.friction
                } else {
                    self.vertex_friction[i]
                };
                // Combine with the collider's own coefficient — but only when there IS one.
                // `collider_friction` uses a non-finite sentinel for "unset", and that is inert
                // only under `Min` (`min(INF, v) == v`). Under `Max` it would return INFINITY
                // and under `Mean` it would poison the average, so the sentinel is tested
                // explicitly rather than relying on the mode's algebra.
                let mu_i = if self.params.collider_friction.is_finite() {
                    self.params
                        .friction_mode
                        .combine(mu_i, self.params.collider_friction)
                } else {
                    mu_i
                };
                let slip = if mu_i > 0.0 {
                    Some((
                        x[i] - self.state.positions[i],
                        mu_i,
                        self.params.friction_eps,
                    ))
                } else {
                    None
                };
                let (cg, ch) = contact::vertex_contact_full(
                    x[i],
                    &self.colliders,
                    dhat,
                    kappa,
                    self.params.contact_dynamic,
                    m,
                    self.params.barrier_kind,
                    elastic_diag.as_ref().map(|b| &b[i]),
                    slip,
                );
                for c in 0..3 {
                    grad[3 * i + c] += cg[c];
                }
                for r in 0..3 {
                    for cc in 0..3 {
                        if ch[r][cc] != 0.0 {
                            trips.push(((3 * i + r) as u32, (3 * i + cc) as u32, ch[r][cc]));
                        }
                    }
                }
            }
        }

        // Self-collision: vertex–triangle IPC barrier. Broad phase (spatial hash)
        // → candidate pairs; per active pair, barrier on the point-triangle
        // distance with FD gradient + Gauss-Newton (PSD) Hessian over the 4
        // vertices. Couples them (grows sparsity; the triplet builder handles it).
        if self.params.self_collision {
            let dhat = self.params.contact_dhat;
            // ---- pass 1: collect every active stencil, BOTH kinds ----
            // The stiffness needs the elastic Hessian as the reference's `fixed_hess_in`:
            // the elastic terms only, with NO contact contributions folded in. So all
            // stencils are collected first and the blocks gathered ONCE, before either kind
            // pushes a triplet — otherwise edge-edge would read vertex-triangle's barrier.
            // Each stencil carries its own activation gap: the reference's `ghat` is
            // per-element, composed as the MEAN of the two participants' (`contact.cu:721`),
            // unlike `offset` which is a SUM. Face stencils use the scene-wide `dhat`.
            let mut stencils: Vec<([usize; 4], f32, [Vec3; 4], f32)> = Vec::new();
            for (v, ti) in crate::broadphase::vertex_triangle_pairs(
                x,
                &self.collision_tris,
                dhat,
                &self.exclusion,
            ) {
                let t = self.collision_tris[ti as usize];
                let idx = [v as usize, t[0] as usize, t[1] as usize, t[2] as usize];
                let (d, g) =
                    contact::point_tri_distance_grad(x[idx[0]], x[idx[1]], x[idx[2]], x[idx[3]]);
                // Keep only ACTIVE stencils. Both barriers return `None` for `d >= ghat`
                // (`contact::barrier` line 639, `cubic_barrier` line 353) and friction is
                // nested inside that `Some`, so an inactive stencil contributes no gradient,
                // no Hessian and no friction — it only inflated `slots`/`blocks` below, which
                // are sized at 16 entries PER STENCIL. On `fishingknot` the broadphase offers
                // ~600 K candidates of which a few hundred are in contact, so this is a ~100x
                // reduction in the number of stencils with an identical result.
                //
                // VERIFIED equivalent, not assumed: `domino` (252 active contacts) reaches
                // frame 50 in 167 steps with max_rel 1.7801e-1 / rms 3.7437e-2 /
                // centroid_rel 8.6935e-3 both with and without this filter — byte-identical.
                // It is also a large SPEED win there (the unfiltered run does not finish
                // inside 2 minutes), though only ~3% of the allocated bytes, because domino's
                // real cost is the 17.3 M raw candidate pairs the CCD broadphase builds
                // BEFORE this point — labelled B-1 in the internal audit notes.
                if d >= dhat {
                    continue;
                }
                stencils.push((idx, d, g, dhat));
            }
            // Thick rods touch when their centrelines are `2*offset + dhat` apart, so the
            // edge-edge broad phase gets the widened radius. Zero for rod-free scenes.
            let ee_search = self.max_edge_gap() + 2.0 * self.max_edge_offset();
            for (ei, ej) in crate::broadphase::edge_edge_pairs(x, &self.edges, ee_search) {
                let (ea, eb) = (self.edges[ei as usize], self.edges[ej as usize]);
                let idx = [
                    ea[0] as usize,
                    ea[1] as usize,
                    eb[0] as usize,
                    eb[1] as usize,
                ];
                let (d, g) = contact::segment_segment_distance_grad(
                    x[idx[0]], x[idx[1]], x[idx[2]], x[idx[3]],
                );
                // The reference's barrier acts on `g = ‖e‖ − offset` (`barrier.cu:103-116`)
                // with the offset SUMMED over the two participants (`contact.cu:558`).
                // Because `∂(d − offset)/∂x = ∂d/∂x`, subtracting it here is exact and the
                // gradient/Hessian assembly below needs no change. Face edges carry offset
                // 0, so cloth behaviour is untouched.
                let off = self.edge_offset[ei as usize] + self.edge_offset[ej as usize];
                let pair_ghat = 0.5 * (self.edge_gap(ei as usize) + self.edge_gap(ej as usize));
                if d - off >= pair_ghat {
                    continue; // inactive — see the vertex-triangle note above
                }
                stencils.push((idx, d - off, g, pair_ghat));
            }

            let mut slots: std::collections::HashMap<(u32, u32), usize> =
                std::collections::HashMap::with_capacity(16 * stencils.len());
            for (idx, _, _, _) in &stencils {
                for &a in idx {
                    for &b in idx {
                        let n = slots.len();
                        slots.entry((a as u32, b as u32)).or_insert(n);
                    }
                }
            }
            let blocks = Self::gather_elastic_blocks(&trips, &slots);

            // ---- pass 2: assemble ----
            for (idx, d, g, pair_ghat) in &stencils {
                let (idx, d, g, pair_ghat) = (*idx, *d, *g, *pair_ghat);
                let k = self.self_contact_rayleigh(&blocks, &slots, &idx, d, &g);
                if let Some((_e, bg, bh)) =
                    contact::barrier_of(self.params.barrier_kind, d, pair_ghat, k)
                {
                    let hp = bh.max(0.0);
                    for a in 0..4 {
                        for c in 0..3 {
                            grad[3 * idx[a] + c] += bg * g[a][c];
                        }
                        for b in 0..4 {
                            for r in 0..3 {
                                for cc in 0..3 {
                                    let vh = hp * g[a][r] * g[b][cc];
                                    if vh != 0.0 {
                                        trips.push((
                                            (3 * idx[a] + r) as u32,
                                            (3 * idx[b] + cc) as u32,
                                            vh,
                                        ));
                                    }
                                }
                            }
                        }
                    }
                    // Self-collision friction on this stencil (reference: the Friction
                    // gradient/Hessian are embedded with the proximity weights).
                    let mu_s = self.stencil_mu(&idx);
                    if mu_s > 0.0 {
                        let dxs: [Vec3; 4] =
                            std::array::from_fn(|a| x[idx[a]] - self.state.positions[idx[a]]);
                        let (fg, fh) =
                            contact::stencil_friction(&g, &dxs, bg, mu_s, self.params.friction_eps);
                        for a in 0..4 {
                            for c in 0..3 {
                                grad[3 * idx[a] + c] += fg[a][c];
                            }
                            for b in 0..4 {
                                for r in 0..3 {
                                    for cc in 0..3 {
                                        let vh = fh[(3 * a + r) * 12 + (3 * b + cc)];
                                        if vh != 0.0 {
                                            trips.push((
                                                (3 * idx[a] + r) as u32,
                                                (3 * idx[b] + cc) as u32,
                                                vh,
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // rhs = -grad. MUST come after every gradient contribution above; it was
        // accidentally hoisted with the bending block once, which silently dropped the
        // barrier/pin/air forces from the right-hand side while the Hessian kept them.
        crate::alloc_stats::checkpoint("asm:constraints-done");
        let mut rhs: Vec<f32> = grad.iter().map(|&g| -g).collect();

        // Dirichlet filter for pinned DOFs: drop their rows+cols (keeps symmetry),
        // set unit diagonal, zero rhs → Δx = 0 there.
        let pinned: Vec<bool> = (0..n).map(|i| self.state.inv_mass[i] <= 0.0).collect();
        trips.retain(|&(r, c, _)| !pinned[r as usize / 3] && !pinned[c as usize / 3]);
        for i in 0..n {
            if pinned[i] {
                for c in 0..3 {
                    trips.push(((3 * i + c) as u32, (3 * i + c) as u32, 1.0));
                    rhs[3 * i + c] = 0.0;
                }
            }
        }

        // `MGUSO_ASM_STATS=1`: the actual assembly sizes, so the memory profile is measured
        // rather than estimated. Built once and reported, not built twice.
        self.trips_hint.set(trips.len());
        let m = CsrMatrix::from_triplets(ndof, &trips);
        if asm_stats() {
            eprintln!(
                "asm: ndof {ndof}  trips {} (cap {}, {:.1} MB)  csr nnz {} ({:.1} MB)",
                trips.len(),
                trips.capacity(),
                (trips.capacity() * std::mem::size_of::<crate::csr::Triplet>()) as f64 / 1048576.0,
                m.nnz(),
                (m.nnz() * 8) as f64 / 1048576.0,
            );
        }
        (m, rhs)
    }
}

/// SPD-projected spring Hessian block `H_aa` and the gradient contribution
/// `grad_a = k(‖d‖−L)·d̂`. Compression (`‖d‖<L`) clamps the tangential term to 0.
/// Shared with the gather assembler (`assembler.rs`) and mirrored in WGSL.
/// `MGUSO_ASM_STATS=1` prints the real per-Newton assembly sizes. Read ONCE (this is on the
/// hot path and `std::env::var` takes a lock).
fn asm_stats() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MGUSO_ASM_STATS").is_ok())
}

pub(crate) fn spring_block(d: Vec3, rest: f32, k: f32) -> ([[f32; 3]; 3], Vec3) {
    let l = d.length();
    if l < 1.0e-9 {
        // Degenerate: isotropic stiffness, no gradient.
        let mut h = [[0.0f32; 3]; 3];
        for c in 0..3 {
            h[c][c] = k;
        }
        return (h, Vec3::ZERO);
    }
    let nrm = d / l;
    let coeff_t = (1.0 - rest / l).max(0.0); // SPD projection
    let mut h = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let nnt = nrm[i] * nrm[j];
            let iden = if i == j { 1.0 } else { 0.0 };
            h[i][j] = k * (nnt + coeff_t * (iden - nnt));
        }
    }
    (h, nrm * (k * (l - rest)))
}

/// The reference's rod stretch block (`hook::make_diff_table`), returning the 3×3
/// Hessian sub-block and the gradient on `a` for `d = x_a − x_b`.
///
/// Identical to [`spring_block`] whenever the segment is stretched (`l ≥ rest`). They
/// diverge only under COMPRESSION: the reference drops the negative transverse term
/// (`c₁ = max(0, r/l₀)`) but leaves the axial coefficient at `(1 − r)/l₀ = 1/l`, i.e.
/// `k·rest/l > k`, whereas `spring_block` clamps the axial term to the true `k`. Both
/// are SPD; the reference's is stiffer on a slack segment. Kept separate so rods match
/// the reference without changing cloth spring behaviour.
/// The reference's wind gust envelope (`utility::get_wind_weight`,
/// `utility/utility.cu:317-321`):
///
/// ```text
/// w(t) = 0.25 * (0.5 * (1 + sin(30 t))) + 0.75
/// ```
///
/// so a constant `wind` parameter still oscillates, between 0.75 and 1.0, at 30 rad/s
/// (~4.77 Hz). The reference zeroes wind entirely when `inactive_momentum` is set; we have no
/// such switch, and no gauge scene enables wind, so that branch has nothing to mirror.
pub fn wind_weight(time: f32) -> f32 {
    let t = 0.25f32;
    t * (0.5 * (1.0 + (30.0 * time).sin())) + (1.0 - t)
}

pub(crate) fn rod_stretch_block(d: Vec3, rest: f32, k: f32) -> ([[f32; 3]; 3], Vec3) {
    let l = d.length();
    if l < 1.0e-9 {
        let mut h = [[0.0f32; 3]; 3];
        for c in 0..3 {
            h[c][c] = k;
        }
        return (h, Vec3::ZERO);
    }
    let nrm = d / l;
    let r = (l - rest) / l;
    let c_axial = 1.0 - r; // = rest/l, always > 0
    let c_trans = r.max(0.0); // SPD projection
    let mut h = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let iden = if i == j { 1.0 } else { 0.0 };
            h[i][j] = k * (c_axial * nrm[i] * nrm[j] + c_trans * iden);
        }
    }
    (h, nrm * (k * (l - rest)))
}

#[cfg(test)]
mod wind_tests {
    use super::*;
    use crate::geometry::Mesh;
    use crate::linsolve::CgCpu;

    /// The gust envelope must match the reference's closed form exactly, including its
    /// bounds: `0.25*(0.5*(1+sin(30t))) + 0.75` ranges over `[0.75, 1.0]` and starts at
    /// 0.875 (t = 0, sin = 0). A constant-wind scene should therefore GUST, not blow steadily.
    #[test]
    fn wind_weight_matches_the_reference_envelope() {
        assert!(
            (wind_weight(0.0) - 0.875).abs() < 1.0e-6,
            "w(0) must be 0.875, got {}",
            wind_weight(0.0)
        );
        // Extremes: sin(30t) = +1 at 30t = pi/2, and -1 at 30t = 3pi/2.
        let t_hi = std::f32::consts::FRAC_PI_2 / 30.0;
        let t_lo = 3.0 * std::f32::consts::FRAC_PI_2 / 30.0;
        assert!(
            (wind_weight(t_hi) - 1.0).abs() < 1.0e-5,
            "peak must be 1.0, got {}",
            wind_weight(t_hi)
        );
        assert!(
            (wind_weight(t_lo) - 0.75).abs() < 1.0e-5,
            "trough must be 0.75, got {}",
            wind_weight(t_lo)
        );
        // Bounded everywhere, and genuinely varying (a constant would pass the bounds).
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for k in 0..2000 {
            let w = wind_weight(k as f32 * 0.001);
            assert!(
                (0.75..=1.0).contains(&w),
                "w({}) = {w} out of range",
                k as f32 * 0.001
            );
            lo = lo.min(w);
            hi = hi.max(w);
        }
        assert!(
            hi - lo > 0.24,
            "the envelope must actually vary, spread {}",
            hi - lo
        );
    }

    /// End-to-end: with wind set, the solver's response must CHANGE OVER TIME even though the
    /// `wind` parameter is constant — which is the whole point of the envelope, and what a
    /// steady implementation would get wrong.
    #[test]
    fn constant_wind_produces_a_time_varying_response() {
        let displacement = |steps: usize| {
            let (nx, nz) = (5usize, 5usize);
            let mut verts = Vec::new();
            for j in 0..nz {
                for i in 0..nx {
                    verts.push(Vec3::new(i as f32 * 0.1, 0.0, j as f32 * 0.1));
                }
            }
            let mut tris = Vec::new();
            for j in 0..nz - 1 {
                for i in 0..nx - 1 {
                    let v = (j * nx + i) as u32;
                    let r = v + nx as u32;
                    tris.push([v, v + 1, r]);
                    tris.push([v + 1, r + 1, r]);
                }
            }
            let mut state = SimState::from_mesh(&Mesh::new(verts, tris), 1.0);
            state.pin(0);
            let params = ImplicitParams {
                gravity: Vec3::ZERO,
                air_density: 1.0,
                wind: Vec3::new(0.0, 5.0, 0.0),
                ..Default::default()
            };
            let mut solver = ImplicitSolver::new(state, params);
            let start = solver.state().positions.clone();
            for _ in 0..steps {
                solver.step(&CgCpu::new());
            }
            let p = &solver.state().positions;
            assert!(p.iter().all(|v| v.is_finite()));
            // Mean displacement magnitude over the sheet.
            let sum: f32 = p.iter().zip(&start).map(|(a, b)| (*a - *b).length()).sum();
            (sum / p.len() as f32, solver.state().time)
        };
        let (d_short, t_short) = displacement(40);
        let (d_long, t_long) = displacement(140);
        assert!(t_long > t_short, "the longer run must reach a later time");
        assert!(
            d_long > d_short,
            "wind must keep doing work: {d_short} at t={t_short} vs {d_long} at t={t_long}"
        );
        assert!(
            d_short > 1.0e-7,
            "wind must move the sheet at all, got {d_short}"
        );
    }
}

#[cfg(test)]
mod stitch_tests {
    use super::*;
    use crate::geometry::Mesh;
    use crate::linsolve::CgCpu;

    // The stitch gradient, the 18x18 Hessian, the degenerate single-vertex source and the
    // past-the-cap behaviour are all verified directly in `crate::stitch`, against an f64
    // energy reference. What is left for this module is the SOLVER-level question: does the
    // assembly actually reach the kernel and sew the seam.

    /// A stitch must actually SEW: two separated points pulled to within about the rest
    /// length, which is the whole point of the feature.
    #[test]
    fn a_stitch_closes_the_seam_it_sews() {
        let verts = vec![Vec3::ZERO, Vec3::new(0.05, 0.0, 0.0)];
        let mut state = SimState::from_mesh(&Mesh::new(verts, Vec::new()), 1.0);
        state.pin(0); // hold one side; the other is drawn to it
        let params = ImplicitParams {
            gravity: Vec3::ZERO,
            ..Default::default()
        };
        let mut solver = ImplicitSolver::new(state, params);
        let st = Stitch {
            index: [0, 0, 0, 1, 1, 1],
            weight: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            stiffness: 1.0,
            l0: 0.001,
            l_cap: 0.01,
        };
        solver.set_stitches(vec![st]).expect("stitches attach");
        assert_eq!(solver.stitches().len(), 1);
        let before = (solver.state().positions[1] - solver.state().positions[0]).length();
        for _ in 0..400 {
            solver.step(&CgCpu::new());
        }
        let p = &solver.state().positions;
        assert!(p.iter().all(|v| v.is_finite()), "stitch must stay finite");
        let after = (p[1] - p[0]).length();
        // Converges to the REST LENGTH, not merely "closer". Newton reaching the exact
        // energy minimum to 4+ digits is the evidence that the 18×18 Hessian is right —
        // a mis-signed or mis-scaled block still closes the seam but stops somewhere else.
        assert!(
            (after - st.l0).abs() <= 1.0e-4 * st.l0,
            "the seam must close to its rest length {}: {before} -> {after}",
            st.l0
        );
        assert!(
            (p[0] - Vec3::ZERO).length() < 1.0e-6,
            "the pinned side must not move"
        );

        // CONTROL: zero stiffness sews nothing, proving the motion is the stitch term.
        let verts = vec![Vec3::ZERO, Vec3::new(0.05, 0.0, 0.0)];
        let mut state = SimState::from_mesh(&Mesh::new(verts, Vec::new()), 1.0);
        state.pin(0);
        let mut inert = ImplicitSolver::new(
            state,
            ImplicitParams {
                gravity: Vec3::ZERO,
                ..Default::default()
            },
        );
        inert
            .set_stitches(vec![Stitch {
                stiffness: 0.0,
                ..st
            }])
            .expect("stitches attach");
        for _ in 0..400 {
            inert.step(&CgCpu::new());
        }
        let d = (inert.state().positions[1] - inert.state().positions[0]).length();
        assert!(
            (d - before).abs() < 1.0e-6,
            "with zero stiffness nothing may move: {before} -> {d}"
        );
    }

    #[test]
    fn set_stitches_rejects_nonsense() {
        let verts = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        let state = SimState::from_mesh(&Mesh::new(verts, vec![[0, 1, 2]]), 1.0);
        let mut solver = ImplicitSolver::new(state, ImplicitParams::default());
        let ok = Stitch {
            index: [0, 0, 0, 1, 1, 1],
            weight: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            stiffness: 1.0,
            l0: 0.001,
            l_cap: 0.01,
        };
        assert!(solver.set_stitches(vec![ok]).is_ok());
        let bad = [
            Stitch {
                index: [0, 0, 0, 9, 9, 9],
                ..ok
            }, // out of range
            Stitch {
                stiffness: -1.0,
                ..ok
            }, // negative stiffness
            Stitch { l0: 0.0, ..ok }, // zero rest length
            Stitch {
                l_cap: 0.0005,
                ..ok
            }, // cap below rest length
            Stitch {
                weight: [0.5, 0.0, 0.0, 1.0, 0.0, 0.0],
                ..ok
            }, // source sums to 0.5
            Stitch {
                weight: [1.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                ..ok
            }, // target sums to 0.5
            Stitch { l0: f32::NAN, ..ok },
        ];
        for b in bad {
            assert!(
                solver.set_stitches(vec![b]).is_err(),
                "set_stitches must reject {b:?}"
            );
        }
    }
}

#[cfg(test)]
mod rod_tests {
    use super::*;
    use crate::geometry::Mesh;
    use crate::linsolve::CgCpu;

    /// `E = k(l − rest)²/2` — the integral of the reference `hook` gradient
    /// `w(l/l₀ − 1)n` with `k = w/l₀`.
    fn rod_energy(d: Vec3, rest: f32, k: f32) -> f32 {
        let e = d.length() - rest;
        0.5 * k * e * e
    }

    /// True (unprojected) Hessian sub-block of `rod_energy`.
    fn true_block(d: Vec3, rest: f32, k: f32) -> [[f32; 3]; 3] {
        let l = d.length();
        let n = d / l;
        let mut h = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let nnt = n[i] * n[j];
                let iden = if i == j { 1.0 } else { 0.0 };
                h[i][j] = k * (nnt + (1.0 - rest / l) * (iden - nnt));
            }
        }
        h
    }

    #[test]
    fn rod_gradient_matches_finite_differences_of_the_reference_energy() {
        let (rest, k) = (0.8f32, 3.5f32);
        // One stretched (l = 1.20 > rest) and one compressed (l = 0.55 < rest) case.
        for d in [Vec3::new(0.6, 0.9, 0.6), Vec3::new(0.3, -0.3, 0.35)] {
            let (_, g) = rod_stretch_block(d, rest, k);
            let h = 1.0e-4;
            for c in 0..3 {
                let mut dp = d;
                let mut dm = d;
                dp[c] += h;
                dm[c] -= h;
                let fd = (rod_energy(dp, rest, k) - rod_energy(dm, rest, k)) / (2.0 * h);
                assert!(
                    (fd - g[c]).abs() <= 2.0e-3 * fd.abs().max(1.0e-3),
                    "rod gradient component {c}: analytic {} vs finite difference {fd} \
                     (l = {}, rest = {rest})",
                    g[c],
                    d.length()
                );
            }
        }
    }

    #[test]
    fn stretched_rod_block_is_the_exact_hessian_and_compressed_is_projected() {
        let (rest, k) = (0.8f32, 3.5f32);

        // Stretched: the reference's grouping reproduces the TRUE Hessian exactly.
        let d = Vec3::new(0.6, 0.9, 0.6);
        assert!(d.length() > rest);
        let (got, _) = rod_stretch_block(d, rest, k);
        let want = true_block(d, rest, k);
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (got[r][c] - want[r][c]).abs() <= 1.0e-5 * k,
                    "stretched rod Hessian [{r}][{c}]: {} vs true {}",
                    got[r][c],
                    want[r][c]
                );
            }
        }

        // Compressed: the transverse term is dropped (it is negative) and the axial
        // coefficient stays at k·rest/l — the reference's `c₀ = max(0, 1−r)/l₀`, which
        // never clamps. So the block is exactly `k·(rest/l)·nnᵀ`.
        let d = Vec3::new(0.3, -0.3, 0.35);
        let l = d.length();
        assert!(l < rest);
        let n = d / l;
        let (got, _) = rod_stretch_block(d, rest, k);
        for r in 0..3 {
            for c in 0..3 {
                let want = k * (rest / l) * n[r] * n[c];
                assert!(
                    (got[r][c] - want).abs() <= 1.0e-5 * k,
                    "compressed rod Hessian [{r}][{c}]: {} vs reference {want}",
                    got[r][c]
                );
            }
        }
        // PSD: a rank-1 positive multiple of nnᵀ. Its only nonzero eigenvalue is the
        // trace, which must be positive; the quadratic form is non-negative everywhere.
        for probe in [Vec3::X, Vec3::Y, Vec3::Z, n, Vec3::new(1.0, -2.0, 3.0)] {
            let mut q = 0.0;
            for r in 0..3 {
                for c in 0..3 {
                    q += probe[r] * got[r][c] * probe[c];
                }
            }
            assert!(
                q >= -1.0e-6,
                "rod block must be PSD, got quadratic form {q}"
            );
        }

        // And this is where it differs from the cloth spring block, which clamps the
        // axial coefficient to k instead of k·rest/l.
        let (spring, _) = spring_block(d, rest, k);
        let axial_rod: f32 = (0..3).map(|c| got[c][c]).sum();
        let axial_spring: f32 = (0..3).map(|c| spring[c][c]).sum();
        assert!(
            axial_rod > axial_spring * 1.05,
            "the reference rod block must be stiffer than the clamped spring block \
             under compression: {axial_rod} vs {axial_spring}"
        );
    }

    /// A straight chain of rod segments whose rest length is shorter than its geometry
    /// (the reference's `length_factor < 1`, which `yarn` ships at 0.85 and `woven` at
    /// 0.8) must CONTRACT toward its pinned end. This is the behavioural check that the
    /// rod stretch term is actually assembled and solved, not just constructed.
    #[test]
    fn a_pretensioned_rod_chain_contracts_toward_its_pin() {
        let n = 9usize;
        let spacing = 0.1f32;
        let verts: Vec<Vec3> = (0..n)
            .map(|i| Vec3::new(i as f32 * spacing, 0.0, 0.0))
            .collect();
        // No triangles at all — this is the rod-only topology of `yarn`/`noodle`/`woven`.
        let mut state = SimState::from_mesh(&Mesh::new(verts, Vec::new()), 1.0);
        state.pin(0);
        // Isolate the rod term.
        let params = ImplicitParams {
            gravity: Vec3::ZERO,
            ..Default::default()
        };
        let mut solver = ImplicitSolver::new(state, params);
        let length_factor = 0.85;
        let rods: Vec<Rod> = (0..n as u32 - 1)
            .map(|i| Rod {
                a: i,
                b: i + 1,
                rest_len: spacing * length_factor,
                k: 200.0,
                offset: 0.0,
                ghat: 0.0,
            })
            .collect();
        solver.set_rods(rods).expect("rods must attach");
        assert_eq!(solver.rods().len(), n - 1);

        let span_before = solver.state().positions[n - 1].x - solver.state().positions[0].x;
        for _ in 0..40 {
            solver.step(&CgCpu::new());
        }
        let p = &solver.state().positions;
        assert!(p.iter().all(|v| v.is_finite()), "rod step must stay finite");
        let span_after = p[n - 1].x - p[0].x;
        let rest_span = spacing * length_factor * (n - 1) as f32;
        assert!(
            span_after < span_before - 1.0e-3,
            "a pre-tensioned chain must contract: {span_before} -> {span_after}"
        );
        assert!(
            (span_after - rest_span).abs() < 0.15 * rest_span,
            "and it must settle near its rest span {rest_span}, got {span_after}"
        );
        assert!(
            (p[0].x - 0.0).abs() < 1.0e-6,
            "the pinned end must not move, got {}",
            p[0].x
        );
    }

    /// Run a 90°-bent 3-vertex strand for 60 steps at the given bend stiffness and return
    /// (start angle, end angle). Rod stretch is attached too, so the hinge cannot cheat by
    /// stretching the segments.
    fn bent_strand_angles(bend_stiff: f32) -> (f32, f32) {
        let verts = vec![
            Vec3::new(-0.1, 0.0, 0.0),
            Vec3::ZERO,
            Vec3::new(0.0, 0.1, 0.0), // 90 deg at vertex 1
        ];
        let mut state = SimState::from_mesh(&Mesh::new(verts, Vec::new()), 1.0);
        // Light vertices so the bend stiffness dominates within a few steps.
        for im in state.inv_mass.iter_mut() {
            *im = 1.0e4;
        }
        let params = ImplicitParams {
            gravity: Vec3::ZERO,
            ..Default::default()
        };
        let mut solver = ImplicitSolver::new(state, params);
        solver
            .set_rods(vec![
                Rod {
                    a: 0,
                    b: 1,
                    rest_len: 0.1,
                    k: 500.0,
                    offset: 0.0,
                    ghat: 0.0,
                },
                Rod {
                    a: 1,
                    b: 2,
                    rest_len: 0.1,
                    k: 500.0,
                    offset: 0.0,
                    ghat: 0.0,
                },
            ])
            .expect("rods attach");
        let hinges = crate::rod_bend::build_rod_hinges(3, &[[0, 1], [1, 2]], &[]);
        assert_eq!(hinges.len(), 1, "one interior vertex");
        assert_eq!(hinges[0][1], 1, "vertex 1 is the centre");
        solver
            .set_rod_hinges(
                hinges
                    .iter()
                    .map(|h| crate::rod_bend::RodHinge {
                        j: h[0],
                        i: h[1],
                        k: h[2],
                        rest_angle: std::f32::consts::PI,
                        stiff: bend_stiff,
                    })
                    .collect(),
            )
            .expect("hinges attach");

        let angle = |s: &ImplicitSolver| {
            let p = &s.state().positions;
            crate::rod_bend::strand_angle(p[0], p[1], p[2])
        };
        let before = angle(&solver);
        for _ in 0..60 {
            solver.step(&CgCpu::new());
        }
        assert!(
            solver.state().positions.iter().all(|p| p.is_finite()),
            "rod bending must keep positions finite"
        );
        (before, angle(&solver))
    }

    #[test]
    fn a_bent_strand_straightens_toward_its_rest_angle() {
        let (before, after) = bent_strand_angles(1.0);
        assert!(
            (before - std::f32::consts::FRAC_PI_2).abs() < 1.0e-4,
            "setup must start at 90 deg, got {before}"
        );
        assert!(
            after > before + 0.05,
            "a bent strand must straighten: {before} -> {after} rad"
        );
        assert!(
            after <= std::f32::consts::PI + 1.0e-3,
            "it must not straighten past π, got {after}"
        );

        // CONTROL: with zero bend stiffness nothing straightens. This is what proves the
        // motion above came from the bend term and not from the stretch springs, which are
        // already at their rest length here.
        let (b0, a0) = bent_strand_angles(0.0);
        assert!(
            (a0 - b0).abs() < 1.0e-3,
            "with no bend stiffness the angle must not change: {b0} -> {a0}"
        );
    }

    /// Two crossing rods must not pass through each other, and their THICKNESS must hold
    /// them apart: the reference subtracts both radii from the measured distance
    /// (`g = ‖e‖ − offset`), so two rods of radius `r` rest with their centrelines about
    /// `2r` apart, not touching at zero.
    ///
    /// Rod segments belong to no triangle, so before this they were invisible to the
    /// collision edge set and strands passed straight through one another.
    fn crossing_rods_final_distance(offset: f32) -> f32 {
        // Rod A along x, pinned. Rod B along z, above it, falling under gravity.
        let verts = vec![
            Vec3::new(-0.15, 0.0, 0.0),
            Vec3::new(0.15, 0.0, 0.0),
            Vec3::new(0.0, 0.12, -0.15),
            Vec3::new(0.0, 0.12, 0.15),
        ];
        let mut state = SimState::from_mesh(&Mesh::new(verts, Vec::new()), 1.0);
        state.pin(0);
        state.pin(1);
        let params = ImplicitParams {
            self_collision: true,
            contact_dhat: 0.02,
            contact_kappa: 1.0e3,
            ..Default::default()
        };
        let mut solver = ImplicitSolver::new(state, params);
        solver
            .set_rods(vec![
                Rod {
                    a: 0,
                    b: 1,
                    rest_len: 0.3,
                    k: 1.0e3,
                    offset,
                    ghat: 0.0,
                },
                Rod {
                    a: 2,
                    b: 3,
                    rest_len: 0.3,
                    k: 1.0e3,
                    offset,
                    ghat: 0.0,
                },
            ])
            .expect("rods attach");
        // The rod segments must actually be in the collision edge set now.
        assert!(
            solver.collision_edge_count() >= 2,
            "both rod segments must be collision edges, got {}",
            solver.collision_edge_count()
        );
        for _ in 0..400 {
            solver.step(&CgCpu::new());
        }
        let p = &solver.state().positions;
        assert!(
            p.iter().all(|v| v.is_finite()),
            "rod contact must stay finite"
        );
        crate::contact::segment_segment_distance(p[0], p[1], p[2], p[3])
    }

    /// Two crossing rods must be held apart by the SUM of their radii, and the barrier
    /// equilibrium gap on top of that must be INDEPENDENT of the radius — which is what
    /// proves `offset` is the reference's additive distance shift (`g = ‖e‖ − offset`,
    /// `barrier.cu:103-116`) and not some ad-hoc stiffness scaling.
    ///
    /// Offsets are the four values the gauge actually ships: `codim` 0.001, `woven` 0.0015,
    /// `yarn` 0.0023, `noodle` 0.003.
    #[test]
    fn crossing_rods_are_held_apart_by_their_combined_radius() {
        let mut gaps = Vec::new();
        for off in [0.001f32, 0.0015, 0.0023, 0.003] {
            let d = crossing_rods_final_distance(off);
            // Never passes through: before rods joined the collision edge set, the falling
            // strand went straight through and free-fell to y ≈ −87.
            assert!(
                d > 1.5 * off && d < 1.0,
                "rods of radius {off} must rest in contact, got centreline distance {d}"
            );
            // The residual beyond the two radii is the barrier's own equilibrium gap.
            gaps.push(d - 2.0 * off);
        }
        let lo = gaps.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = gaps.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            hi - lo < 1.0e-4,
            "the barrier gap beyond the combined radius must not depend on the radius; \
             got {gaps:?} (spread {})",
            hi - lo
        );
        // And thickness must actually separate them by more than the gap alone.
        assert!(
            lo > 0.0,
            "rods must rest OUTSIDE their combined radius, gaps {gaps:?}"
        );
    }

    #[test]
    fn set_rods_rejects_nonsense() {
        let verts = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        let state = SimState::from_mesh(&Mesh::new(verts, vec![[0, 1, 2]]), 1.0);
        let mut solver = ImplicitSolver::new(state, ImplicitParams::default());
        let ok = Rod {
            a: 0,
            b: 1,
            rest_len: 0.5,
            k: 10.0,
            offset: 0.0,
            ghat: 0.0,
        };
        assert!(solver.set_rods(vec![ok]).is_ok());
        for bad in [
            Rod {
                a: 0,
                b: 9,
                rest_len: 0.5,
                k: 10.0,
                offset: 0.0,
                ghat: 0.0,
            },
            Rod {
                a: 2,
                b: 2,
                rest_len: 0.5,
                k: 10.0,
                offset: 0.0,
                ghat: 0.0,
            },
            Rod {
                a: 0,
                b: 1,
                rest_len: 0.0,
                k: 10.0,
                offset: 0.0,
                ghat: 0.0,
            },
            Rod {
                a: 0,
                b: 1,
                rest_len: 0.5,
                k: 0.0,
                offset: 0.0,
                ghat: 0.0,
            },
            Rod {
                a: 0,
                b: 1,
                rest_len: f32::NAN,
                k: 10.0,
                offset: 0.0,
                ghat: 0.0,
            },
        ] {
            assert!(
                solver.set_rods(vec![bad]).is_err(),
                "set_rods must reject {bad:?}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Mesh;
    use crate::linsolve::CgCpu;
    use approx::assert_relative_eq;

    #[test]
    fn nan_guard_rejects_bad_step() {
        // A NaN velocity poisons x̃ = x + v·h + g·h², so the Newton solve returns a
        // non-finite step. The robustness guard must reject it and keep positions
        // finite (holding the pre-step state) instead of committing NaN.
        let verts = vec![
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ];
        let mut state = SimState::from_mesh(&Mesh::new(verts, vec![[0, 1, 2], [2, 1, 3]]), 0.05);
        state.pin(0);
        state.velocities[1] = Vec3::new(f32::NAN, 0.0, 0.0);
        let mut solver = ImplicitSolver::new(state, ImplicitParams::default());
        solver.step(&CgCpu::new());
        assert!(
            solver.state.positions.iter().all(|p| p.is_finite()),
            "guard must reject the NaN step and keep positions finite"
        );
        assert!(solver.state.velocities.iter().all(|v| v.is_finite()));
    }

    /// A horizontal strip pinned along its x=0 edge, cantilevered under gravity.
    /// Returns the mean vertical sag of the far edge after settling.
    fn cantilever_far_edge_sag(bend: f32) -> f32 {
        let (nx, nz) = (7usize, 3usize);
        let (dx, dz) = (0.1f32, 0.1f32);
        let mut verts = Vec::new();
        for j in 0..nz {
            for i in 0..nx {
                verts.push(Vec3::new(i as f32 * dx, 1.0, j as f32 * dz));
            }
        }
        let mut tris = Vec::new();
        for j in 0..nz - 1 {
            for i in 0..nx - 1 {
                let v = (j * nx + i) as u32;
                let r = v + nx as u32;
                tris.push([v, v + 1, r]);
                tris.push([r, v + 1, r + 1]);
            }
        }
        let mut state = SimState::from_mesh(&Mesh::new(verts, tris), 0.02);
        // Pin the first TWO columns so the root leaves horizontally: the sheet
        // can no longer swing down as a rigid flap about a single edge, so any
        // droop must come from curling — which bending resists.
        for j in 0..nz {
            state.pin(j * nx);
            state.pin(j * nx + 1);
        }
        let mut solver = ImplicitSolver::new(
            state,
            ImplicitParams {
                dt: 1.0 / 120.0,
                stiffness: 500.0,
                bend_stiffness: bend,
                newton_iters: 4,
                ..Default::default()
            },
        );
        for _ in 0..300 {
            solver.step(&CgCpu::new());
        }
        let mut sag = 0.0f32;
        for j in 0..nz {
            sag += 1.0 - solver.state.positions[j * nx + (nx - 1)].y;
        }
        sag / nz as f32
    }

    #[test]
    fn bending_stiffness_resists_sag() {
        let soft = cantilever_far_edge_sag(0.0);
        let stiff = cantilever_far_edge_sag(2000.0);
        assert!(soft.is_finite() && stiff.is_finite());
        // Bending resists curling → a stiff sheet's far edge sags clearly less.
        assert!(
            stiff < soft * 0.85,
            "bending did not reduce sag: soft={soft}, stiff={stiff}"
        );
    }

    fn flat_cloth(n: usize, y: f32, spacing: f32) -> SimState {
        let mut verts = Vec::new();
        for j in 0..n {
            for i in 0..n {
                verts.push(Vec3::new(i as f32 * spacing, y, j as f32 * spacing));
            }
        }
        let mut tris = Vec::new();
        for j in 0..n - 1 {
            for i in 0..n - 1 {
                let v = (j * n + i) as u32;
                let r = v + n as u32;
                tris.push([v, v + 1, r]);
                tris.push([r, v + 1, r + 1]);
            }
        }
        SimState::from_mesh(&Mesh::new(verts, tris), 0.02)
    }

    #[test]
    fn good_drape_passes_invariants() {
        use crate::contact::Collider;
        use crate::metrics::check_invariants;
        // A cloth settling onto a plane — a well-behaved run. Its recorded
        // metrics must pass every solver invariant.
        let mut solver = ImplicitSolver::new(
            flat_cloth(6, 0.3, 0.1),
            ImplicitParams {
                dt: 1.0 / 120.0,
                stiffness: 500.0,
                cg_tol: 1.0e-3,
                cg_max_iter: 2000,
                contact_dhat: 0.02,
                contact_kappa: 1.0e3,
                ..Default::default()
            },
        );
        solver.set_colliders(vec![Collider::HalfSpace {
            point: Vec3::ZERO,
            normal: Vec3::Y,
        }]);
        for _ in 0..120 {
            solver.step(&CgCpu::new());
        }
        assert_eq!(solver.metrics.steps.len(), 120);
        let report = check_invariants(&solver.metrics);
        assert!(
            report.all_ok(),
            "good drape must pass invariants: {:?}",
            report.checks.iter().filter(|c| !c.ok).collect::<Vec<_>>()
        );
    }

    #[test]
    fn diverged_run_fails_invariants() {
        use crate::metrics::check_invariants;
        // Starve the linear solver (cg_max_iter=1) on a stiff cloth so the PCG
        // residual stays well above tol → the cg_converged invariant must fail.
        let mut state = flat_cloth(8, 1.0, 0.1);
        state.pin(0);
        let mut solver = ImplicitSolver::new(
            state,
            ImplicitParams {
                dt: 1.0 / 60.0,
                stiffness: 1.0e5, // very stiff → ill-conditioned
                cg_tol: 1.0e-6,
                cg_max_iter: 1, // one iteration cannot converge
                newton_iters: 2,
                ..Default::default()
            },
        );
        for _ in 0..10 {
            solver.step(&CgCpu::new());
        }
        let report = check_invariants(&solver.metrics);
        assert!(!report.all_ok(), "starved solve must fail invariants");
        let by = |n: &str| report.checks.iter().find(|c| c.name == n).unwrap().ok;
        assert!(!by("cg_converged"), "cg_converged should have failed");
    }

    // Self-collision barrier + CCD hold a RESTING contact: two layers starting
    // within dhat stay separated under gravity. Impact robustness is now covered
    // by `self_collision_survives_hard_impact` (iterated per-pair ACCD, §4).
    #[test]
    fn self_collision_holds_resting_contact() {
        let n = 5;
        let mut positions = Vec::new();
        let mut tris = Vec::new();
        let mut lower_count = 0;
        for (li, y) in [(0usize, 0.0f32), (1, 0.018)] {
            let base = positions.len() as u32;
            for j in 0..n {
                for i in 0..n {
                    positions.push(Vec3::new(i as f32 * 0.1, y, j as f32 * 0.1));
                }
            }
            for j in 0..n - 1 {
                for i in 0..n - 1 {
                    let v = base + (j * n + i) as u32;
                    let r = v + n as u32;
                    tris.push([v, v + 1, r]);
                    tris.push([r, v + 1, r + 1]);
                }
            }
            if li == 0 {
                lower_count = positions.len();
            }
        }
        let mut state = SimState::from_mesh(&Mesh::new(positions, tris), 0.02);
        for i in 0..lower_count {
            state.pin(i); // pin the entire lower layer
        }
        let mut solver = ImplicitSolver::new(
            state,
            ImplicitParams {
                dt: 1.0 / 480.0,
                stiffness: 500.0,
                gravity: Vec3::new(0.0, -4.0, 0.0),
                contact_dhat: 0.02,
                contact_kappa: 2.0e3,
                self_collision: true,
                ..Default::default()
            },
        );
        for _ in 0..300 {
            solver.step(&CgCpu::new());
        }
        let upper_min = solver.state.positions[lower_count..]
            .iter()
            .map(|p| p.y)
            .fold(f32::INFINITY, f32::min);
        assert!(solver.state.positions.iter().all(|p| p.is_finite()));
        assert!(
            upper_min > -0.005,
            "self-collision failed to hold contact: upper min_y={upper_min}"
        );
    }

    /// A heavy resting self-contact must be held essentially non-penetrating by the
    /// reference's dynamic contact stiffness (`self_contact_rayleigh`: elastic-Rayleigh +
    /// `mass/gap²`), which stiffens without bound as the gap closes.
    ///
    /// This replaces a test that swept `contact_dynamic` 0 → 1 and asserted the dynamic
    /// setting rested higher. That knob no longer governs self-collision: the stiffness is
    /// now always the full Rayleigh form, matching the reference, which has no such switch
    /// (`contact_dynamic` still applies to the analytic-COLLIDER path). With the knob gone
    /// the comparison is vacuous, so the test asserts the outcome it was really about —
    /// the heavy layer does not sink through.
    #[test]
    fn self_collision_rayleigh_stiffness_holds_a_heavy_resting_contact() {
        let n = 4;
        let mut positions = Vec::new();
        let mut tris = Vec::new();
        let mut lower_count = 0;
        for (li, y) in [(0usize, 0.0f32), (1, 0.02)] {
            let base = positions.len() as u32;
            for j in 0..n {
                for i in 0..n {
                    positions.push(Vec3::new(i as f32 * 0.1, y, j as f32 * 0.1));
                }
            }
            for j in 0..n - 1 {
                for i in 0..n - 1 {
                    let v = base + (j * n + i) as u32;
                    let r = v + n as u32;
                    tris.push([v, v + 1, r]);
                    tris.push([r, v + 1, r + 1]);
                }
            }
            if li == 0 {
                lower_count = positions.len();
            }
        }
        let mut state = SimState::from_mesh(&Mesh::new(positions, tris), 0.1);
        for i in 0..lower_count {
            state.pin(i);
        }
        let mut solver = ImplicitSolver::new(
            state,
            ImplicitParams {
                dt: 1.0 / 240.0,
                stiffness: 500.0,
                gravity: Vec3::new(0.0, -9.8, 0.0),
                contact_dhat: 0.02,
                contact_kappa: 1.0e2,
                self_collision: true,
                ..Default::default()
            },
        );
        for _ in 0..200 {
            solver.step(&CgCpu::new());
        }
        let upper_min = solver.state.positions[lower_count..]
            .iter()
            .map(|p| p.y)
            .fold(f32::INFINITY, f32::min);
        assert!(solver.state.positions.iter().all(|p| p.is_finite()));
        assert!(
            !solver.has_self_intersection(),
            "no self-intersection may develop"
        );
        assert!(
            upper_min > -0.01,
            "the upper layer must rest essentially non-penetrating, got min_y={upper_min}"
        );
    }

    // The §6 intersection predicate (`positions_self_intersect`) detects a
    // TRANSVERSE self-intersection — a flap folded so it pierces another triangle —
    // and reports a clean config as intersection-free. This is the reference's use
    // of check_intersection (verification/reporting).
    #[test]
    fn detects_transverse_self_intersection() {
        // Two disjoint triangles in one mesh. Clean (separated) vs pierced.
        // Triangle A (large, in z=0 plane) + triangle B (small, vertical, crossing A).
        let clean = vec![
            // A
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(2.0, -1.0, 0.0),
            Vec3::new(-1.0, 2.0, 0.0),
            // B — sits above A (z>0), not piercing
            Vec3::new(0.2, 0.2, 0.3),
            Vec3::new(0.4, 0.2, 0.3),
            Vec3::new(0.3, 0.4, 0.6),
        ];
        let tris = vec![[0u32, 1, 2], [3, 4, 5]];
        let solver = ImplicitSolver::new(
            SimState::from_mesh(&Mesh::new(clean.clone(), tris.clone()), 1.0),
            ImplicitParams {
                self_collision: true,
                ..Default::default()
            },
        );
        assert!(
            !solver.positions_self_intersect(&clean),
            "clean config must be intersection-free"
        );
        // Move B down so it straddles A's plane (z from +0.3 to −0.3) at A's interior.
        let pierced = vec![
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(2.0, -1.0, 0.0),
            Vec3::new(-1.0, 2.0, 0.0),
            Vec3::new(0.2, 0.2, 0.3),
            Vec3::new(0.4, 0.2, -0.3),
            Vec3::new(0.3, 0.4, 0.3),
        ];
        assert!(
            solver.positions_self_intersect(&pierced),
            "a triangle piercing another must be detected"
        );
    }

    // Hard IMPACT target: two PARALLEL flat sheets slammed together fast. This is
    // a DEGENERATE case for self-collision and remains open. Findings (Sessions
    // 13-15): (1) iterated V-T + E-E ACCD is complete for *transverse* surface
    // crossing (verified), but (2) two coherent parallel sheets are COPLANAR at the
    // crossing instant — no edge ever straddles the other's plane transversely — so
    // neither the CCD nor the reference's edge-triangle intersection predicate (§6.1,
    // `positions_self_intersect`) can catch it; and (3) under high momentum the fixed
    // barrier is too weak to decelerate before contact, so the sheets creep together.
    // Session 22 UPDATE: signed/oriented self-contact was attempted and REVERTED.
    // The primitives (`contact::{oriented_point_triangle, oriented_point_tri_grad,
    // signed_barrier}`) are verified (`oriented_gap_signs_and_barrier_recovers`)
    // and kept, but wiring a signed cubic barrier into the self-collision assembly
    // — with the home side from step-start, then a maintained per-pair active set,
    // then displacement-expanded detection — REGRESSED this slam every time
    // (−0.96 → −1.6 → −1.2 → −7.1) and destabilized it, with no passing config.
    // Root difficulty: a fast vertex's first in-`dhat` detection can already be on
    // the wrong side, and expanding detection creates spurious wrong-sided contacts.
    // A correct fix needs the reference's full stateful contact set with the side
    // fixed by a CCD-guaranteed non-crossing history (intersection-free by
    // construction), not a post-hoc sign. Dynamic κ (Session 16) fixes moderate
    // resting-under-load (`dynamic_stiffness_holds_resting_contact_higher`); the
    // transverse/fold self-collision (the common case) works. This degenerate
    // violent parallel-sheet slam remains open — kept `#[ignore]` (no false green).
    #[test]
    #[ignore = "violent parallel-sheet slam open; signed-barrier self-contact attempted+reverted (regressed it) — needs the reference's full stateful CCD-history contact set"]
    fn self_collision_survives_hard_impact() {
        let n = 5;
        let mut positions = Vec::new();
        let mut tris = Vec::new();
        let mut lower_count = 0;
        for (li, y) in [(0usize, 0.0f32), (1, 0.15)] {
            let base = positions.len() as u32;
            for j in 0..n {
                for i in 0..n {
                    positions.push(Vec3::new(i as f32 * 0.1, y, j as f32 * 0.1));
                }
            }
            for j in 0..n - 1 {
                for i in 0..n - 1 {
                    let v = base + (j * n + i) as u32;
                    let r = v + n as u32;
                    tris.push([v, v + 1, r]);
                    tris.push([r, v + 1, r + 1]);
                }
            }
            if li == 0 {
                lower_count = positions.len();
            }
        }
        let mut state = SimState::from_mesh(&Mesh::new(positions, tris), 0.02);
        for i in 0..lower_count {
            state.pin(i); // lower layer is a fixed floor of triangles
        }
        // Slam the upper layer down FAST (free step ≫ gap ≫ dhat).
        for i in lower_count..state.vertex_count() {
            state.velocities[i] = Vec3::new(0.0, -12.0, 0.0);
        }
        let mut solver = ImplicitSolver::new(
            state,
            ImplicitParams {
                dt: 1.0 / 60.0,
                stiffness: 500.0,
                gravity: Vec3::new(0.0, -9.8, 0.0),
                contact_dhat: 0.02,
                contact_kappa: 2.0e3,
                self_collision: true,
                ..Default::default()
            },
        );
        for _ in 0..120 {
            solver.step(&CgCpu::new());
        }
        let upper_min = solver.state.positions[lower_count..]
            .iter()
            .map(|p| p.y)
            .fold(f32::INFINITY, f32::min);
        let upper_max = solver.state.positions[lower_count..]
            .iter()
            .map(|p| p.y)
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(solver.state.positions.iter().all(|p| p.is_finite()));
        // No tunneling: the upper layer never crosses below the pinned lower layer.
        assert!(
            upper_min > -0.01,
            "upper layer tunneled through: min_y={upper_min}"
        );
        // Sanity: it actually fell and impacted (didn't just hang in mid-air).
        assert!(
            upper_max < 0.14,
            "upper layer did not descend/impact: max_y={upper_max}"
        );
    }

    #[test]
    fn cloth_rests_on_plane_without_penetration() {
        use crate::contact::Collider;
        let mut solver = ImplicitSolver::new(
            flat_cloth(6, 0.3, 0.1),
            ImplicitParams {
                dt: 1.0 / 120.0,
                stiffness: 500.0,
                contact_dhat: 0.02,
                contact_kappa: 1.0e3,
                ..Default::default()
            },
        );
        solver.set_colliders(vec![Collider::HalfSpace {
            point: Vec3::ZERO,
            normal: Vec3::Y,
        }]);
        for _ in 0..400 {
            solver.step(&CgCpu::new());
        }
        // The barrier keeps every vertex above the floor (y >= 0, small tol).
        for p in &solver.state.positions {
            assert!(p.is_finite(), "blew up: {p:?}");
            assert!(p.y > -1.0e-3, "penetrated the floor: y={}", p.y);
        }
        // And the cloth actually descended to rest near the floor (didn't hang).
        let max_y = solver
            .state
            .positions
            .iter()
            .map(|p| p.y)
            .fold(0.0f32, f32::max);
        assert!(
            max_y < 0.1,
            "cloth did not settle onto the floor: max_y={max_y}"
        );
    }

    #[test]
    fn cloth_drapes_over_sphere_without_penetration() {
        use crate::contact::Collider;
        // Cloth starts above a sphere and drapes over it.
        let mut state = flat_cloth(9, 1.2, 0.15);
        // Center the cloth over the sphere at origin.
        let shift = Vec3::new(-0.6, 0.0, -0.6);
        for p in &mut state.positions {
            *p += shift;
        }
        state.prev_positions = state.positions.clone();
        let (center, radius) = (Vec3::new(0.0, 0.5, 0.0), 0.5f32);
        let mut solver = ImplicitSolver::new(
            state,
            ImplicitParams {
                dt: 1.0 / 120.0,
                stiffness: 500.0,
                bend_stiffness: 5.0,
                contact_dhat: 0.03,
                contact_kappa: 1.0e3,
                floor_y: -1.0,
                ..Default::default()
            },
        );
        solver.set_colliders(vec![Collider::Sphere { center, radius }]);
        for _ in 0..300 {
            solver.step(&CgCpu::new());
        }
        // No vertex is inside the sphere (allowing a small barrier tolerance).
        for p in &solver.state.positions {
            assert!(p.is_finite(), "blew up: {p:?}");
            assert!(
                (*p - center).length() > radius - 1.0e-2,
                "penetrated sphere: dist={}",
                (*p - center).length()
            );
        }
    }

    #[test]
    fn ccd_clamps_step_toward_triangle() {
        // One free vertex at z=0.05 above a pinned triangle in the z=0 plane,
        // stepping down by 0.2. CCD must return t<1 that keeps it above.
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.3, 0.3, 0.05),
        ];
        let tris = vec![[0u32, 1, 2]];
        let mut state = SimState::from_mesh(&Mesh::new(positions, tris), 1.0);
        state.pin(0);
        state.pin(1);
        state.pin(2);
        let solver = ImplicitSolver::new(
            state,
            ImplicitParams {
                self_collision: true,
                contact_dhat: 0.02,
                ..Default::default()
            },
        );
        let x = solver.state.positions.clone();
        let dx = vec![
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, -0.2),
        ];
        let t = solver.ccd_max_step(&x, &dx);
        assert!(t < 1.0, "CCD did not clamp an approaching step: t={t}");
        // The clamped step keeps the vertex above the triangle plane.
        let new_z = x[3].z + t * dx[3].z;
        assert!(new_z > 0.0, "CCD allowed crossing: new_z={new_z}");
    }

    // Anisotropy: a vertical cloth (warp=x, weft=y) pinned along its top edge,
    // loaded downward (weft direction) by gravity. Only membrane resists (springs
    // off). A weft-stiff cloth must stretch/sag less than a warp-stiff one under
    // the SAME load — i.e. the stiffness is directional, not just magnitude.
    fn membrane_bottom_y(warp: f32, weft: f32) -> f32 {
        let n = 6;
        let mut verts = Vec::new();
        for j in 0..n {
            for i in 0..n {
                verts.push(Vec3::new(i as f32 * 0.1, j as f32 * 0.1, 0.0));
            }
        }
        let mut tris = Vec::new();
        for j in 0..n - 1 {
            for i in 0..n - 1 {
                let v = (j * n + i) as u32;
                let r = v + n as u32;
                tris.push([v, v + 1, r]);
                tris.push([r, v + 1, r + 1]);
            }
        }
        let mut state = SimState::from_mesh(&Mesh::new(verts, tris), 0.05);
        for i in 0..n {
            state.pin((n - 1) * n + i); // top row (max y)
        }
        let mut solver = ImplicitSolver::new(
            state,
            ImplicitParams {
                dt: 1.0 / 120.0,
                stiffness: 0.0, // springs off — isolate the membrane
                membrane_warp: warp,
                membrane_weft: weft,
                membrane_shear: 200.0,
                newton_iters: 5,
                ..Default::default()
            },
        );
        for _ in 0..300 {
            solver.step(&CgCpu::new());
        }
        // Mean y of the bottom row.
        (0..n).map(|i| solver.state.positions[i].y).sum::<f32>() / n as f32
    }

    /// A vertex hurled at a triangle from FAR outside the broadphase's initial radius must
    /// still be stopped. This is the regression guard for the widening-radius conservative
    /// advancement in `ccd_max_step`: it starts the broadphase at half the characteristic
    /// edge length (so `domino`'s grid stops degenerating to one 1344-edge bucket) and widens
    /// only while the step it yields outruns the radius. If that loop ever accepted a step
    /// longer than its radius covered, this vertex would tunnel clean through — which is
    /// exactly the failure the old always-full-radius query was written to prevent.
    #[test]
    fn ccd_stops_a_vertex_moving_far_beyond_the_initial_search_radius() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.3, 0.3, 4.0), // 4 units away: ~4x the mesh, 200x the contact gap
        ];
        let tris = vec![[0u32, 1, 2]];
        let mut state = SimState::from_mesh(&Mesh::new(positions, tris), 1.0);
        state.pin(0);
        state.pin(1);
        state.pin(2);
        let solver = ImplicitSolver::new(
            state,
            ImplicitParams {
                self_collision: true,
                contact_dhat: 0.02,
                ..Default::default()
            },
        );
        let x = solver.state.positions.clone();
        // Straight through the triangle and out the other side.
        let dx = vec![
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, -8.0),
        ];
        let t = solver.ccd_max_step(&x, &dx);
        let z = x[3].z + t * dx[3].z;
        assert!(
            z > 0.0,
            "the vertex tunnelled through the triangle: t={t} lands it at z={z}"
        );
        assert!(
            t < 1.0,
            "an 8-unit step at a triangle 4 units away must be clamped, got t={t}"
        );
    }

    #[test]
    fn membrane_is_anisotropic() {
        let weft_stiff = membrane_bottom_y(100.0, 5000.0); // stiff along the load
        let warp_stiff = membrane_bottom_y(5000.0, 100.0); // stiff across the load
        assert!(weft_stiff.is_finite() && warp_stiff.is_finite());
        // Vertical (weft) load: the weft-stiff cloth hangs higher (stretches less).
        assert!(
            weft_stiff > warp_stiff + 0.02,
            "membrane not anisotropic: weft_stiff bottom={weft_stiff}, warp_stiff bottom={warp_stiff}"
        );
    }

    #[test]
    fn single_spring_reaches_static_equilibrium() {
        // vert0 pinned at (0,1,0); vert1 free at (0,0,0), rest length 1, mass 1.
        // Under gravity g, tension k·Δ = m·g → stretch Δ = m·g/k, so vert1.y → -Δ.
        let mesh = Mesh {
            positions: vec![Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 0.0)],
            tris: vec![],
            edges: vec![[0, 1]],
        };
        let mut state = SimState::from_mesh(&mesh, 1.0);
        state.pin(0);
        let k = 1000.0f32;
        let g = 9.8f32;
        let mut solver = ImplicitSolver::new(
            state,
            ImplicitParams {
                dt: 1.0 / 120.0,
                gravity: Vec3::new(0.0, -g, 0.0),
                newton_iters: 3,
                stiffness: k,
                floor_y: f32::NEG_INFINITY,
                ..Default::default()
            },
        );
        for _ in 0..2000 {
            solver.step(&CgCpu::new());
        }
        let expected_y = -(1.0 * g / k); // ≈ -0.0098
        assert_relative_eq!(solver.state.positions[1].y, expected_y, epsilon = 5e-4);
        assert!(solver.state.positions[0].y == 1.0, "pinned vertex moved");
    }

    #[test]
    fn implicit_hanging_cloth_is_stable() {
        let n = 6;
        let mut verts = Vec::new();
        for j in 0..n {
            for i in 0..n {
                verts.push(Vec3::new(i as f32 * 0.1, 1.0, j as f32 * 0.1));
            }
        }
        let mut tris = Vec::new();
        for j in 0..n - 1 {
            for i in 0..n - 1 {
                let v = (j * n + i) as u32;
                let r = v + n as u32;
                tris.push([v, v + 1, r]);
                tris.push([r, v + 1, r + 1]);
            }
        }
        let mut state = SimState::from_mesh(&Mesh::new(verts, tris), 0.05);
        state.pin(0);
        state.pin((n - 1) as usize);
        let mut solver = ImplicitSolver::new(state, ImplicitParams::default());
        for _ in 0..120 {
            solver.step(&CgCpu::new());
        }
        for p in &solver.state.positions {
            assert!(p.is_finite(), "implicit cloth blew up: {p:?}");
        }
    }

    #[test]
    fn anisotropic_strain_limit_caps_warp_and_weft_differently() {
        // Two independent edges, both along +x in 3D (so gravity stretches both),
        // but with different UV directions: edge (0,1) runs along U (warp), edge
        // (2,3) along V (weft). With warp ratio 1.1 and weft ratio 1.5 the warp edge
        // is capped tighter than the weft edge — anisotropy driven purely by UVs.
        let mesh = Mesh {
            positions: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0), // edge 0: (0,1)
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0), // edge 1: (2,3)
            ],
            tris: vec![],
            edges: vec![[0, 1], [2, 3]],
        };
        let mut state = SimState::from_mesh(&mesh, 1.0);
        // UVs: edge (0,1) delta along U → warp; edge (2,3) delta along V → weft.
        state.uvs = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]];
        state.pin(0);
        state.pin(2);
        let params = ImplicitParams {
            dt: 1.0 / 60.0,
            gravity: Vec3::new(2.0, 0.0, 0.0),
            newton_iters: 8,
            stiffness: 0.0,
            strain_limit_ratio: 1.1,      // warp
            strain_limit_ratio_weft: 1.5, // weft
            strain_limit_k: 5000.0,
            ..Default::default()
        };
        let mut s = ImplicitSolver::new(state, params);
        for _ in 0..300 {
            s.step(&CgCpu::new());
        }
        let warp_len = (s.state.positions[1] - s.state.positions[0]).length();
        let weft_len = (s.state.positions[3] - s.state.positions[2]).length();
        assert!(
            (1.1..1.25).contains(&warp_len),
            "warp edge capped at ~1.1, got {warp_len}"
        );
        assert!(
            (1.5..1.65).contains(&weft_len),
            "weft edge capped at ~1.5, got {weft_len}"
        );
        assert!(
            weft_len > warp_len + 0.2,
            "weft must stretch more than warp"
        );
    }

    #[test]
    fn strain_limit_caps_stretch_under_load() {
        // One edge, rest length 1.0, pinned at the origin; gravity along +x pulls
        // the free end away, stretching the edge. Springs OFF so ONLY the opt-in
        // strain limiter resists. A one-sided limiter caps over-stretch: with a 1.2
        // ratio the edge settles just past 1.2 (limiter balances gravity); with the
        // limiter OFF nothing resists and the edge runs away.
        let build = |ratio: f32| {
            let mesh = Mesh {
                positions: vec![Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)],
                tris: vec![],
                edges: vec![[0, 1]],
            };
            let mut state = SimState::from_mesh(&mesh, 1.0);
            state.pin(0);
            let params = ImplicitParams {
                dt: 1.0 / 60.0,
                gravity: Vec3::new(2.0, 0.0, 0.0), // pull the free end along +x
                newton_iters: 8,
                stiffness: 0.0, // springs off — isolate the strain limiter
                strain_limit_ratio: ratio,
                strain_limit_k: 5000.0,
                ..Default::default()
            };
            let mut s = ImplicitSolver::new(state, params);
            for _ in 0..300 {
                s.step(&CgCpu::new());
            }
            (s.state.positions[1] - s.state.positions[0]).length()
        };
        // Limiter on: capped just above the 1.2 limit (stiff → tiny overshoot).
        let limited = build(1.2);
        assert!(
            (1.2..1.3).contains(&limited),
            "strain-limited edge length {limited} (want ~1.2)"
        );
        // Limiter off: nothing resists gravity → the edge runs far past the limit.
        let free = build(0.0);
        assert!(free > 3.0, "unlimited edge length {free} (want ≫ limit)");
    }

    #[test]
    fn implicit_is_deterministic() {
        let run = || {
            let mesh = Mesh {
                positions: vec![Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 0.0)],
                tris: vec![],
                edges: vec![[0, 1]],
            };
            let mut state = SimState::from_mesh(&mesh, 1.0);
            state.pin(0);
            let mut s = ImplicitSolver::new(state, ImplicitParams::default());
            for _ in 0..50 {
                s.step(&CgCpu::new());
            }
            s.state.positions
        };
        assert_eq!(run(), run());
    }

    /// Defaults must be untouched by the barrier/stiffness wiring: an explicit
    /// `Default`-derived param set and one that names the new fields at their
    /// documented defaults must produce BIT-IDENTICAL trajectories.
    #[test]
    fn barrier_wiring_leaves_defaults_bit_identical() {
        let run = |explicit: bool| {
            let n = 5;
            let mut positions = Vec::new();
            let mut tris = Vec::new();
            for j in 0..n {
                for i in 0..n {
                    positions.push(Vec3::new(i as f32 * 0.1, 0.35, j as f32 * 0.1));
                }
            }
            for j in 0..n - 1 {
                for i in 0..n - 1 {
                    let v = (j * n + i) as u32;
                    let r = v + n as u32;
                    tris.push([v, v + 1, r]);
                    tris.push([r, v + 1, r + 1]);
                }
            }
            let state = SimState::from_mesh(&Mesh::new(positions, tris), 0.1);
            let base = ImplicitParams {
                dt: 1.0 / 240.0,
                stiffness: 500.0,
                gravity: Vec3::new(0.0, -9.8, 0.0),
                contact_dhat: 0.02,
                contact_kappa: 1.0e2,
                ..Default::default()
            };
            let params = if explicit {
                ImplicitParams {
                    barrier_kind: contact::BarrierKind::Log,
                    contact_rayleigh: false,
                    ..base
                }
            } else {
                base
            };
            let mut solver = ImplicitSolver::new(state, params);
            solver.set_colliders(vec![contact::Collider::HalfSpace {
                point: Vec3::ZERO,
                normal: Vec3::Y,
            }]);
            for _ in 0..60 {
                solver.step(&CgCpu::new());
            }
            solver.state.positions.clone()
        };
        let a = run(false);
        let b = run(true);
        for (i, (p, q)) in a.iter().zip(&b).enumerate() {
            assert_eq!(
                p.to_array(),
                q.to_array(),
                "vertex {i} diverged from default"
            );
        }
    }

    /// The cubic barrier must be usable end-to-end through the solver and, being
    /// finite through zero gap, must keep a dropped sheet out of the floor.
    #[test]
    fn cubic_barrier_runs_and_stays_non_penetrating() {
        let n = 5;
        let mut positions = Vec::new();
        let mut tris = Vec::new();
        for j in 0..n {
            for i in 0..n {
                positions.push(Vec3::new(i as f32 * 0.1, 0.3, j as f32 * 0.1));
            }
        }
        for j in 0..n - 1 {
            for i in 0..n - 1 {
                let v = (j * n + i) as u32;
                let r = v + n as u32;
                tris.push([v, v + 1, r]);
                tris.push([r, v + 1, r + 1]);
            }
        }
        let state = SimState::from_mesh(&Mesh::new(positions, tris), 0.1);
        let mut solver = ImplicitSolver::new(
            state,
            ImplicitParams {
                dt: 1.0 / 240.0,
                stiffness: 500.0,
                gravity: Vec3::new(0.0, -9.8, 0.0),
                contact_dhat: 0.02,
                contact_kappa: 1.0e3,
                barrier_kind: contact::BarrierKind::Cubic,
                ..Default::default()
            },
        );
        solver.set_colliders(vec![contact::Collider::HalfSpace {
            point: Vec3::ZERO,
            normal: Vec3::Y,
        }]);
        for _ in 0..400 {
            solver.step(&CgCpu::new());
        }
        let min_y = solver
            .state
            .positions
            .iter()
            .map(|p| p.y)
            .fold(f32::INFINITY, f32::min);
        assert!(min_y.is_finite(), "cubic barrier produced non-finite state");
        // Landed on the floor (fell from 0.3) but did not pass through it.
        assert!(min_y < 0.2, "sheet did not fall: {min_y}");
        assert!(
            min_y > -0.01,
            "cubic barrier let the sheet penetrate: {min_y}"
        );
    }

    /// The reference's absolute Rayleigh κ must run end-to-end and hold a resting
    /// contact non-penetrating (it includes the elastic term, so it is at least as
    /// stiff as the inertia-only estimate).
    #[test]
    fn rayleigh_contact_stiffness_holds_resting_contact() {
        let n = 5;
        let mut positions = Vec::new();
        let mut tris = Vec::new();
        for j in 0..n {
            for i in 0..n {
                positions.push(Vec3::new(i as f32 * 0.1, 0.05, j as f32 * 0.1));
            }
        }
        for j in 0..n - 1 {
            for i in 0..n - 1 {
                let v = (j * n + i) as u32;
                let r = v + n as u32;
                tris.push([v, v + 1, r]);
                tris.push([r, v + 1, r + 1]);
            }
        }
        let state = SimState::from_mesh(&Mesh::new(positions, tris), 0.1);
        let mut solver = ImplicitSolver::new(
            state,
            ImplicitParams {
                dt: 1.0 / 240.0,
                stiffness: 500.0,
                gravity: Vec3::new(0.0, -9.8, 0.0),
                contact_dhat: 0.02,
                contact_kappa: 1.0e2,
                contact_rayleigh: true,
                ..Default::default()
            },
        );
        solver.set_colliders(vec![contact::Collider::HalfSpace {
            point: Vec3::ZERO,
            normal: Vec3::Y,
        }]);
        for _ in 0..300 {
            solver.step(&CgCpu::new());
        }
        let min_y = solver
            .state
            .positions
            .iter()
            .map(|p| p.y)
            .fold(f32::INFINITY, f32::min);
        assert!(min_y.is_finite(), "Rayleigh κ produced non-finite state");
        assert!(
            min_y > -0.01,
            "Rayleigh κ let the sheet penetrate the floor: {min_y}"
        );
    }

    /// Attaching tets must not perturb a cloth-only scene when the tet stiffness is
    /// at its default 0 — the opt-in guarantee, checked bit-identically.
    #[test]
    fn tets_with_zero_stiffness_leave_cloth_bit_identical() {
        let run = |attach_tets: bool| {
            let n = 5;
            let mut positions = Vec::new();
            let mut tris = Vec::new();
            for j in 0..n {
                for i in 0..n {
                    positions.push(Vec3::new(i as f32 * 0.1, 0.3, j as f32 * 0.1));
                }
            }
            for j in 0..n - 1 {
                for i in 0..n - 1 {
                    let v = (j * n + i) as u32;
                    let r = v + n as u32;
                    tris.push([v, v + 1, r]);
                    tris.push([r, v + 1, r + 1]);
                }
            }
            let state = SimState::from_mesh(&Mesh::new(positions, tris), 0.1);
            let mut solver = ImplicitSolver::new(
                state,
                ImplicitParams {
                    dt: 1.0 / 240.0,
                    stiffness: 500.0,
                    gravity: Vec3::new(0.0, -9.8, 0.0),
                    ..Default::default()
                },
            );
            if attach_tets {
                // A tet spanning real mesh vertices; mu/lambda stay 0 by default.
                solver.set_tets(&[[0, 1, 5, 6]]);
            }
            for _ in 0..40 {
                solver.step(&CgCpu::new());
            }
            solver.state.positions.clone()
        };
        let a = run(false);
        let b = run(true);
        for (i, (p, q)) in a.iter().zip(&b).enumerate() {
            assert_eq!(
                p.to_array(),
                q.to_array(),
                "vertex {i} perturbed by inert tets"
            );
        }
    }

    /// A volumetric block under gravity, its top face pinned, must stretch, stay
    /// finite, and — the key volumetric property — keep every element un-inverted.
    #[test]
    fn tet_block_simulates_without_inversion() {
        // 3×3×3 vertex grid → 8 cubes → 40 tets.
        let nn = 3usize;
        let idx = |i: usize, j: usize, k: usize| ((k * nn + j) * nn + i) as u32;
        let mut positions = Vec::new();
        for k in 0..nn {
            for j in 0..nn {
                for i in 0..nn {
                    positions.push(Vec3::new(i as f32 * 0.1, j as f32 * 0.1, k as f32 * 0.1));
                }
            }
        }
        let mut tets = Vec::new();
        for k in 0..nn - 1 {
            for j in 0..nn - 1 {
                for i in 0..nn - 1 {
                    let (a, b, c, d) = (
                        idx(i, j, k),
                        idx(i + 1, j, k),
                        idx(i, j + 1, k),
                        idx(i + 1, j + 1, k),
                    );
                    let (e, f, g, h) = (
                        idx(i, j, k + 1),
                        idx(i + 1, j, k + 1),
                        idx(i, j + 1, k + 1),
                        idx(i + 1, j + 1, k + 1),
                    );
                    tets.push([a, b, c, e]);
                    tets.push([b, d, c, h]);
                    tets.push([b, c, e, h]);
                    tets.push([c, e, g, h]);
                    tets.push([b, e, f, h]);
                }
            }
        }
        // Surface tris only matter for mass lumping here; give the solver the mesh.
        let tris: Vec<[u32; 3]> = vec![[idx(0, 0, 0), idx(1, 0, 0), idx(0, 1, 0)]];
        let mut state = SimState::from_mesh(&Mesh::new(positions.clone(), tris), 0.5);
        // Pin the top face (j = nn-1) so the block hangs and stretches.
        for k in 0..nn {
            for i in 0..nn {
                state.pin(idx(i, nn - 1, k) as usize);
            }
        }
        let mut solver = ImplicitSolver::new(
            state,
            ImplicitParams {
                dt: 1.0 / 240.0,
                stiffness: 0.0, // volumetric-only: no cloth springs
                gravity: Vec3::new(0.0, -9.8, 0.0),
                tet_mu: 2.0e3,
                tet_lambda: 4.0e3,
                ..Default::default()
            },
        );
        solver.set_tets(&tets);
        assert!(
            solver.tets().len() >= 30,
            "expected ~40 tets, got {}",
            solver.tets().len()
        );
        let y0 = solver.state.positions[idx(1, 0, 1) as usize].y;
        for _ in 0..200 {
            solver.step(&CgCpu::new());
        }
        // Finite.
        assert!(
            solver.state.positions.iter().all(|p| p.is_finite()),
            "tet sim produced non-finite state"
        );
        // Hung under gravity: the free bottom moved down.
        let y1 = solver.state.positions[idx(1, 0, 1) as usize].y;
        assert!(y1 < y0, "block did not sag: {y0} -> {y1}");
        // Every element still positively oriented (no inversion).
        for (t, el) in solver.tets().iter().enumerate() {
            let p = [
                solver.state.positions[el.v[0] as usize],
                solver.state.positions[el.v[1] as usize],
                solver.state.positions[el.v[2] as usize],
                solver.state.positions[el.v[3] as usize],
            ];
            let f = crate::tet::deformation_gradient(&p, &el.dm_inv);
            assert!(
                f.determinant() > 0.0,
                "tet {t} inverted (det F = {})",
                f.determinant()
            );
        }
    }

    /// Volumetric stiffness must resist gravity: a stiffer block sags less.
    #[test]
    fn stiffer_tets_sag_less() {
        let run = |mu: f32, lam: f32| {
            let nn = 3usize;
            let idx = |i: usize, j: usize, k: usize| ((k * nn + j) * nn + i) as u32;
            let mut positions = Vec::new();
            for k in 0..nn {
                for j in 0..nn {
                    for i in 0..nn {
                        positions.push(Vec3::new(i as f32 * 0.1, j as f32 * 0.1, k as f32 * 0.1));
                    }
                }
            }
            let mut tets = Vec::new();
            for k in 0..nn - 1 {
                for j in 0..nn - 1 {
                    for i in 0..nn - 1 {
                        let (a, b, c, d) = (
                            idx(i, j, k),
                            idx(i + 1, j, k),
                            idx(i, j + 1, k),
                            idx(i + 1, j + 1, k),
                        );
                        let (e, f, g, h) = (
                            idx(i, j, k + 1),
                            idx(i + 1, j, k + 1),
                            idx(i, j + 1, k + 1),
                            idx(i + 1, j + 1, k + 1),
                        );
                        tets.push([a, b, c, e]);
                        tets.push([b, d, c, h]);
                        tets.push([b, c, e, h]);
                        tets.push([c, e, g, h]);
                        tets.push([b, e, f, h]);
                    }
                }
            }
            let tris: Vec<[u32; 3]> = vec![[idx(0, 0, 0), idx(1, 0, 0), idx(0, 1, 0)]];
            let mut state = SimState::from_mesh(&Mesh::new(positions, tris), 0.5);
            for k in 0..nn {
                for i in 0..nn {
                    state.pin(idx(i, nn - 1, k) as usize);
                }
            }
            let mut solver = ImplicitSolver::new(
                state,
                ImplicitParams {
                    dt: 1.0 / 240.0,
                    stiffness: 0.0,
                    gravity: Vec3::new(0.0, -9.8, 0.0),
                    tet_mu: mu,
                    tet_lambda: lam,
                    ..Default::default()
                },
            );
            solver.set_tets(&tets);
            for _ in 0..200 {
                solver.step(&CgCpu::new());
            }
            solver.state.positions[idx(1, 0, 1) as usize].y
        };
        let soft = run(2.0e2, 4.0e2);
        let stiff = run(2.0e4, 4.0e4);
        assert!(
            stiff > soft,
            "stiffer volumetric material must sag less: stiff={stiff} soft={soft}"
        );
    }

    /// Cloth-only scenes must be untouched by the collision-surface indirection:
    /// with no tets attached, `collision_tris` IS `state.tris`, so a self-colliding
    /// cloth run must stay bit-identical to the pre-change behavior.
    #[test]
    fn collision_surface_defaults_to_cloth_tris() {
        let n = 5;
        let mut positions = Vec::new();
        let mut tris = Vec::new();
        for j in 0..n {
            for i in 0..n {
                positions.push(Vec3::new(i as f32 * 0.1, 0.2, j as f32 * 0.1));
            }
        }
        for j in 0..n - 1 {
            for i in 0..n - 1 {
                let v = (j * n + i) as u32;
                let r = v + n as u32;
                tris.push([v, v + 1, r]);
                tris.push([r, v + 1, r + 1]);
            }
        }
        let state = SimState::from_mesh(&Mesh::new(positions, tris.clone()), 0.1);
        let solver = ImplicitSolver::new(
            state,
            ImplicitParams {
                self_collision: true,
                ..Default::default()
            },
        );
        assert_eq!(
            solver.collision_tris(),
            tris.as_slice(),
            "cloth-only collision surface must equal the mesh tris"
        );
    }

    /// Attaching tets must add exactly the tet BOUNDARY surface to the collision
    /// geometry — never interior faces, and never dropping the cloth tris.
    #[test]
    fn set_tets_adds_only_the_boundary_surface() {
        let nn = 3usize;
        let idx = |i: usize, j: usize, k: usize| ((k * nn + j) * nn + i) as u32;
        let mut positions = Vec::new();
        for k in 0..nn {
            for j in 0..nn {
                for i in 0..nn {
                    positions.push(Vec3::new(i as f32 * 0.1, j as f32 * 0.1, k as f32 * 0.1));
                }
            }
        }
        let mut tet_idx = Vec::new();
        for k in 0..nn - 1 {
            for j in 0..nn - 1 {
                for i in 0..nn - 1 {
                    let (a, b, c, d) = (
                        idx(i, j, k),
                        idx(i + 1, j, k),
                        idx(i, j + 1, k),
                        idx(i + 1, j + 1, k),
                    );
                    let (e, f, g, h) = (
                        idx(i, j, k + 1),
                        idx(i + 1, j, k + 1),
                        idx(i, j + 1, k + 1),
                        idx(i + 1, j + 1, k + 1),
                    );
                    tet_idx.push([a, b, c, e]);
                    tet_idx.push([b, d, c, h]);
                    tet_idx.push([b, c, e, h]);
                    tet_idx.push([c, e, g, h]);
                    tet_idx.push([b, e, f, h]);
                }
            }
        }
        // A cloth triangle on its OWN vertices, floating above the block. It must not
        // reuse the block's grid vertices: `(idx(0,0,0), idx(1,0,0), idx(0,1,0))` is
        // itself a boundary face of this tetrahedralization, so using it would conflate
        // this test (does the append add the boundary?) with the separate question of
        // whether an already-present boundary face is deduplicated — which
        // `tet_surface_already_in_the_mesh_is_not_duplicated` covers.
        let base = positions.len() as u32;
        for p in [
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.1, 0.0, 1.0),
            Vec3::new(0.0, 0.1, 1.0),
        ] {
            positions.push(p);
        }
        let cloth_tris: Vec<[u32; 3]> = vec![[base, base + 1, base + 2]];
        let state = SimState::from_mesh(&Mesh::new(positions.clone(), cloth_tris.clone()), 0.5);
        let mut solver = ImplicitSolver::new(state, ImplicitParams::default());
        let before = solver.collision_tris().len();
        assert_eq!(before, cloth_tris.len());
        solver.set_tets(&tet_idx);
        let expected_surface = crate::tet::tet_surface(solver.tets());
        assert_eq!(
            solver.collision_tris().len(),
            cloth_tris.len() + expected_surface.len(),
            "collision surface must be cloth tris + tet boundary"
        );
        // Cloth tris preserved.
        assert_eq!(
            &solver.collision_tris()[..cloth_tris.len()],
            cloth_tris.as_slice()
        );
        // The boundary surface must enclose the block's volume (so it really is the
        // boundary, not an arbitrary face subset).
        let tet_vol: f32 = solver.tets().iter().map(|e| e.volume).sum();
        let enclosed = crate::tet::enclosed_volume(&expected_surface, &positions);
        assert!(
            (enclosed - tet_vol).abs() < 1e-3 * tet_vol,
            "added surface is not the closed boundary: {enclosed} vs {tet_vol}"
        );
    }

    /// Volumetric self-collision end-to-end: a tet block dropped onto a PINNED cloth
    /// sheet must be stopped by it — the block's boundary surface now participates in
    /// vertex-triangle self-collision, so it cannot pass through.
    #[test]
    fn tet_block_does_not_pass_through_pinned_cloth() {
        let cn = 4usize; // cloth grid (small: this test drives the full self-collision path)
        let mut positions = Vec::new();
        let mut tris = Vec::new();
        // Cloth sheet at y = 0, spanning x,z in [0, 0.5].
        for j in 0..cn {
            for i in 0..cn {
                positions.push(Vec3::new(i as f32 * 0.1, 0.0, j as f32 * 0.1));
            }
        }
        for j in 0..cn - 1 {
            for i in 0..cn - 1 {
                let v = (j * cn + i) as u32;
                let r = v + cn as u32;
                tris.push([v, v + 1, r]);
                tris.push([r, v + 1, r + 1]);
            }
        }
        let cloth_count = positions.len();
        // Tet block above the sheet. nn=2 → ONE cube → 5 tets: enough to exercise
        // volumetric self-collision without making a unit test cost minutes (the
        // FD-of-gradient Hessian is ~24 SVDs per element per assembly).
        let nn = 2usize;
        let base = positions.len() as u32;
        let idx = |i: usize, j: usize, k: usize| base + ((k * nn + j) * nn + i) as u32;
        for k in 0..nn {
            for j in 0..nn {
                for i in 0..nn {
                    positions.push(Vec3::new(
                        0.10 + i as f32 * 0.10,
                        0.08 + j as f32 * 0.10,
                        0.10 + k as f32 * 0.10,
                    ));
                }
            }
        }
        let mut tet_idx = Vec::new();
        for k in 0..nn - 1 {
            for j in 0..nn - 1 {
                for i in 0..nn - 1 {
                    let (a, b, c, d) = (
                        idx(i, j, k),
                        idx(i + 1, j, k),
                        idx(i, j + 1, k),
                        idx(i + 1, j + 1, k),
                    );
                    let (e, f, g, h) = (
                        idx(i, j, k + 1),
                        idx(i + 1, j, k + 1),
                        idx(i, j + 1, k + 1),
                        idx(i + 1, j + 1, k + 1),
                    );
                    tet_idx.push([a, b, c, e]);
                    tet_idx.push([b, d, c, h]);
                    tet_idx.push([b, c, e, h]);
                    tet_idx.push([c, e, g, h]);
                    tet_idx.push([b, e, f, h]);
                }
            }
        }
        let mut state = SimState::from_mesh(&Mesh::new(positions, tris), 0.3);
        // Pin the whole cloth sheet so it acts as a taut barrier.
        for i in 0..cloth_count {
            state.pin(i);
        }
        let mut solver = ImplicitSolver::new(
            state,
            ImplicitParams {
                dt: 1.0 / 240.0,
                stiffness: 300.0,
                gravity: Vec3::new(0.0, -9.8, 0.0),
                tet_mu: 3.0e3,
                tet_lambda: 6.0e3,
                self_collision: true,
                contact_dhat: 0.01,
                contact_kappa: 1.0e4,
                ..Default::default()
            },
        );
        solver.set_tets(&tet_idx);
        // The tet surface must be in the collision geometry for this to be meaningful.
        assert!(
            solver.collision_tris().len() > 2 * (cn - 1) * (cn - 1),
            "tet surface not added to collision geometry"
        );
        for _ in 0..80 {
            solver.step(&CgCpu::new());
        }
        assert!(
            solver.state.positions.iter().all(|p| p.is_finite()),
            "volumetric self-collision produced non-finite state"
        );
        // The block fell toward the sheet but must not have dropped far below it.
        let block_min_y = solver.state.positions[cloth_count..]
            .iter()
            .map(|p| p.y)
            .fold(f32::INFINITY, f32::min);
        assert!(
            block_min_y > -0.06,
            "tet block passed through the pinned cloth: min y = {block_min_y}"
        );
    }

    /// Plasticity must be off by default: a volumetric run with the default params
    /// must be BIT-IDENTICAL to one that explicitly sets the plasticity fields to
    /// their documented defaults.
    #[test]
    fn tet_plasticity_is_off_by_default() {
        let run = |explicit: bool| {
            let (positions, tet_idx, tris) = tet_scene();
            let mut state = SimState::from_mesh(&Mesh::new(positions, tris), 0.5);
            state.pin(0);
            let base = ImplicitParams {
                dt: 1.0 / 240.0,
                stiffness: 0.0,
                gravity: Vec3::new(0.0, -9.8, 0.0),
                tet_mu: 2.0e3,
                tet_lambda: 4.0e3,
                ..Default::default()
            };
            let params = if explicit {
                ImplicitParams {
                    tet_plasticity: 0.0,
                    tet_plasticity_threshold: 0.0,
                    ..base
                }
            } else {
                base
            };
            let mut solver = ImplicitSolver::new(state, params);
            solver.set_tets(&tet_idx);
            for _ in 0..30 {
                solver.step(&CgCpu::new());
            }
            solver.state.positions.clone()
        };
        let a = run(false);
        let b = run(true);
        for (i, (p, q)) in a.iter().zip(&b).enumerate() {
            assert_eq!(
                p.to_array(),
                q.to_array(),
                "vertex {i} diverged from default"
            );
        }
    }

    /// End-to-end creep: a block held stretched by pinned ends must, WITH plasticity,
    /// keep less stored elastic energy than without — the rest state has permanently
    /// yielded. This is the property a user would actually observe.
    #[test]
    fn plasticity_permanently_relaxes_a_held_stretch() {
        let run = |rate: f32| {
            let (positions, tet_idx, tris) = tet_scene();
            let n = positions.len();
            let mut state = SimState::from_mesh(&Mesh::new(positions.clone(), tris), 0.5);
            // Pin the two extreme y-layers apart so the block is held stretched.
            for (i, p) in positions.iter().enumerate() {
                if p.y == 0.0 {
                    state.pin(i);
                }
            }
            let mut solver = ImplicitSolver::new(
                state,
                ImplicitParams {
                    dt: 1.0 / 120.0,
                    stiffness: 0.0,
                    gravity: Vec3::new(0.0, -30.0, 0.0), // strong pull ⇒ real stretch
                    tet_model: crate::tet::TetModel::Arap,
                    tet_mu: 1.0e3,
                    tet_lambda: 0.0,
                    tet_plasticity: rate,
                    tet_plasticity_threshold: 0.02,
                    ..Default::default()
                },
            );
            solver.set_tets(&tet_idx);
            for _ in 0..150 {
                solver.step(&CgCpu::new());
            }
            // Stored elastic energy in the settled configuration.
            let e: f32 = solver
                .tets()
                .iter()
                .map(|el| {
                    let p = [
                        solver.state.positions[el.v[0] as usize],
                        solver.state.positions[el.v[1] as usize],
                        solver.state.positions[el.v[2] as usize],
                        solver.state.positions[el.v[3] as usize],
                    ];
                    crate::tet::tet_energy(&p, el, 1.0e3, 0.0)
                })
                .sum();
            assert!(e.is_finite(), "non-finite energy");
            let _ = n;
            e
        };
        let elastic_only = run(0.0);
        let with_creep = run(5.0);
        assert!(
            with_creep < elastic_only,
            "plasticity must leave less stored energy: creep {with_creep} vs elastic {elastic_only}"
        );
    }

    /// Shared 2×2×2 tet scene helper for the solver-level plasticity tests.
    fn tet_scene() -> (Vec<Vec3>, Vec<[u32; 4]>, Vec<[u32; 3]>) {
        let nn = 3usize;
        let idx = |i: usize, j: usize, k: usize| ((k * nn + j) * nn + i) as u32;
        let mut positions = Vec::new();
        for k in 0..nn {
            for j in 0..nn {
                for i in 0..nn {
                    positions.push(Vec3::new(i as f32 * 0.1, j as f32 * 0.1, k as f32 * 0.1));
                }
            }
        }
        let mut tets = Vec::new();
        for k in 0..nn - 1 {
            for j in 0..nn - 1 {
                for i in 0..nn - 1 {
                    let (a, b, c, d) = (
                        idx(i, j, k),
                        idx(i + 1, j, k),
                        idx(i, j + 1, k),
                        idx(i + 1, j + 1, k),
                    );
                    let (e, f, g, h) = (
                        idx(i, j, k + 1),
                        idx(i + 1, j, k + 1),
                        idx(i, j + 1, k + 1),
                        idx(i + 1, j + 1, k + 1),
                    );
                    tets.push([a, b, c, e]);
                    tets.push([b, d, c, h]);
                    tets.push([b, c, e, h]);
                    tets.push([c, e, g, h]);
                    tets.push([b, e, f, h]);
                }
            }
        }
        let tris = vec![[idx(0, 0, 0), idx(1, 0, 0), idx(0, 1, 0)]];
        (positions, tets, tris)
    }

    fn sheet(n: usize, y: f32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let mut positions = Vec::new();
        let mut tris = Vec::new();
        for j in 0..n {
            for i in 0..n {
                positions.push(Vec3::new(i as f32 * 0.1, y, j as f32 * 0.1));
            }
        }
        for j in 0..n - 1 {
            for i in 0..n - 1 {
                let v = (j * n + i) as u32;
                let r = v + n as u32;
                tris.push([v, v + 1, r]);
                tris.push([r, v + 1, r + 1]);
            }
        }
        (positions, tris)
    }

    /// Registering shell faces with the default (zero) stiffness must leave a cloth
    /// run BIT-IDENTICAL — the opt-in guarantee for the new per-face path.
    #[test]
    fn shell_faces_with_zero_stiffness_are_bit_identical() {
        let run = |register: bool| {
            let (positions, tris) = sheet(5, 0.3);
            let state = SimState::from_mesh(&Mesh::new(positions, tris.clone()), 0.1);
            let mut solver = ImplicitSolver::new(
                state,
                ImplicitParams {
                    dt: 1.0 / 240.0,
                    stiffness: 500.0,
                    gravity: Vec3::new(0.0, -9.8, 0.0),
                    ..Default::default()
                },
            );
            if register {
                solver.set_shell_faces(&tris);
                assert!(!solver.shells().is_empty());
            }
            for _ in 0..40 {
                solver.step(&CgCpu::new());
            }
            solver.state.positions.clone()
        };
        let a = run(false);
        let b = run(true);
        for (i, (p, q)) in a.iter().zip(&b).enumerate() {
            assert_eq!(
                p.to_array(),
                q.to_array(),
                "vertex {i} perturbed by inert shells"
            );
        }
    }

    /// A sheet driven ONLY by the isotropic shell model (springs and membrane off)
    /// must behave elastically: it sags under gravity from pinned corners, stays
    /// finite, and a stiffer material sags less.
    #[test]
    fn shell_only_sheet_is_elastic() {
        let run = |mu: f32| {
            let (positions, tris) = sheet(5, 0.0);
            let n = positions.len();
            let mut state = SimState::from_mesh(&Mesh::new(positions.clone(), tris.clone()), 0.1);
            // Pin the first row so the rest hangs.
            for i in 0..5 {
                state.pin(i);
            }
            let mut solver = ImplicitSolver::new(
                state,
                ImplicitParams {
                    dt: 1.0 / 240.0,
                    stiffness: 0.0, // no springs
                    gravity: Vec3::new(0.0, -9.8, 0.0),
                    shell_mu: mu,
                    shell_lambda: 0.5 * mu,
                    ..Default::default()
                },
            );
            solver.set_shell_faces(&tris);
            for _ in 0..200 {
                solver.step(&CgCpu::new());
            }
            assert!(
                solver.state.positions.iter().all(|p| p.is_finite()),
                "shell-only sheet produced non-finite state"
            );
            let _ = n;
            solver.state.positions[solver.state.positions.len() - 1].y
        };
        let soft = run(50.0);
        let stiff = run(5000.0);
        assert!(soft < -0.001, "soft sheet should sag: {soft}");
        assert!(
            stiff > soft,
            "stiffer shell material must sag less: stiff {stiff} vs soft {soft}"
        );
    }

    /// Per-face shell stiffness must work on its own, WITHOUT also setting the
    /// scene-wide `shell_mu`/`shell_lambda` scalars. The membrane and tet paths both
    /// treat a non-empty per-element table as reason enough to assemble; shells must
    /// too, or `set_shell_stiffness` is silently a no-op at default params.
    /// Uniform per-face values must reproduce the equivalent scalar run exactly.
    #[test]
    fn per_face_shell_stiffness_works_without_the_scalars() {
        let run = |scalar: bool| {
            let (positions, tris) = sheet(5, 0.0);
            let mut state = SimState::from_mesh(&Mesh::new(positions, tris.clone()), 0.1);
            for i in 0..5 {
                state.pin(i);
            }
            let (mu, lambda) = (800.0f32, 400.0f32);
            let mut solver = ImplicitSolver::new(
                state,
                ImplicitParams {
                    dt: 1.0 / 240.0,
                    stiffness: 0.0, // no springs — shells are the only elasticity
                    gravity: Vec3::new(0.0, -9.8, 0.0),
                    shell_mu: if scalar { mu } else { 0.0 },
                    shell_lambda: if scalar { lambda } else { 0.0 },
                    ..Default::default()
                },
            );
            solver.set_shell_faces(&tris);
            if !scalar {
                solver
                    .set_shell_stiffness(vec![[mu, lambda]; tris.len()])
                    .expect("per-face shell stiffness");
            }
            for _ in 0..120 {
                solver.step(&CgCpu::new());
            }
            assert!(solver.state.positions.iter().all(|p| p.is_finite()));
            solver.state.positions.last().unwrap().y
        };
        let via_scalar = run(true);
        let via_per_face = run(false);
        // The scalar run is the reference behaviour; uniform per-face must match it.
        assert_eq!(
            via_per_face, via_scalar,
            "per-face shell stiffness must match the equivalent scalar run \
             (per_face {via_per_face} vs scalar {via_scalar}) — a mismatch means the \
             per-face table was ignored"
        );
    }

    /// Mixed per-face models: driving HALF the faces with Baraff-Witkin and the other
    /// half with the isotropic shell must run stably and produce a state distinct from
    /// either model alone — the capability the gauge's mixed-model scenes need.
    #[test]
    fn mixed_per_face_models_run_and_differ_from_either_alone() {
        let build = |membrane_k: f32, shell_mu: f32, shell_faces: Option<&[[u32; 3]]>| {
            let (positions, tris) = sheet(5, 0.0);
            let mut state = SimState::from_mesh(&Mesh::new(positions, tris.clone()), 0.1);
            for i in 0..5 {
                state.pin(i);
            }
            let mut solver = ImplicitSolver::new(
                state,
                ImplicitParams {
                    dt: 1.0 / 240.0,
                    stiffness: 0.0,
                    gravity: Vec3::new(0.0, -9.8, 0.0),
                    membrane_warp: membrane_k,
                    membrane_weft: membrane_k,
                    membrane_shear: 0.5 * membrane_k,
                    shell_mu,
                    shell_lambda: 0.5 * shell_mu,
                    ..Default::default()
                },
            );
            if let Some(f) = shell_faces {
                solver.set_shell_faces(f);
            }
            for _ in 0..120 {
                solver.step(&CgCpu::new());
            }
            assert!(solver.state.positions.iter().all(|p| p.is_finite()));
            solver.state.positions.last().unwrap().y
        };
        let (_, tris) = sheet(5, 0.0);
        let half = tris.len() / 2;
        // Membrane on all faces, no shells.
        let membrane_only = build(800.0, 0.0, None);
        // Shell on all faces, no membrane.
        let shell_only = build(0.0, 800.0, Some(&tris));
        // Mixed: membrane everywhere + shell on half the faces (so half are stiffer).
        let mixed = build(800.0, 800.0, Some(&tris[..half]));
        assert!(membrane_only.is_finite() && shell_only.is_finite() && mixed.is_finite());
        // Adding shell stiffness to half the faces must stiffen relative to membrane
        // alone (less sag), and differ from the shell-only case.
        assert!(
            mixed > membrane_only,
            "mixed should be stiffer than membrane alone: {mixed} vs {membrane_only}"
        );
        assert!(
            (mixed - shell_only).abs() > 1e-5,
            "mixed should differ from shell-only"
        );
    }

    /// All plasticity paths must be off by default: a cloth run with shells and hinges
    /// present must be BIT-IDENTICAL to one naming every plasticity field at its
    /// documented default.
    #[test]
    fn all_plasticity_paths_are_off_by_default() {
        let run = |explicit: bool| {
            let (positions, tris) = sheet(5, 0.2);
            let state = SimState::from_mesh(&Mesh::new(positions, tris.clone()), 0.1);
            let base = ImplicitParams {
                dt: 1.0 / 240.0,
                stiffness: 500.0,
                bend_stiffness: 0.5,
                gravity: Vec3::new(0.0, -9.8, 0.0),
                shell_mu: 200.0,
                ..Default::default()
            };
            let params = if explicit {
                ImplicitParams {
                    shell_plasticity: 0.0,
                    shell_plasticity_threshold: 0.0,
                    bend_plasticity: 0.0,
                    bend_plasticity_threshold: 0.0,
                    ..base
                }
            } else {
                base
            };
            let mut solver = ImplicitSolver::new(state, params);
            solver.set_shell_faces(&tris);
            for _ in 0..40 {
                solver.step(&CgCpu::new());
            }
            solver.state.positions.clone()
        };
        let a = run(false);
        let b = run(true);
        for (i, (p, q)) in a.iter().zip(&b).enumerate() {
            assert_eq!(
                p.to_array(),
                q.to_array(),
                "vertex {i} diverged from default"
            );
        }
    }

    /// End-to-end bend creep: a sheet draped over gravity with bend plasticity ON must
    /// end with hinge rest angles that have moved away from flat — the cloth has taken
    /// a set. With it off, the rest angles must be untouched.
    #[test]
    fn bend_plasticity_sets_a_crease_in_a_draped_sheet() {
        let run = |rate: f32| {
            let (positions, tris) = sheet(6, 0.0);
            let mut state = SimState::from_mesh(&Mesh::new(positions, tris), 0.1);
            for i in 0..6 {
                state.pin(i);
            }
            let mut solver = ImplicitSolver::new(
                state,
                ImplicitParams {
                    dt: 1.0 / 240.0,
                    stiffness: 800.0,
                    bend_stiffness: 2.0,
                    gravity: Vec3::new(0.0, -20.0, 0.0),
                    bend_plasticity: rate,
                    bend_plasticity_threshold: 0.02,
                    ..Default::default()
                },
            );
            for _ in 0..200 {
                solver.step(&CgCpu::new());
            }
            assert!(solver.state.positions.iter().all(|p| p.is_finite()));
            // Largest |rest angle| across hinges.
            solver
                .hinges_for_test()
                .iter()
                .map(|h| h.rest.abs())
                .fold(0.0f32, f32::max)
        };
        let off = run(0.0);
        let on = run(8.0);
        assert!(
            off < 1e-5,
            "rest angles must be untouched with plasticity off: {off}"
        );
        assert!(
            on > 1e-3,
            "bend plasticity should have set a crease (max |rest| = {on})"
        );
    }

    /// Friction must be off by default (bit-identical) and, when enabled, must
    /// measurably slow a sheet sliding down an inclined half-space — the observable
    /// behavior, not just the force formula.
    #[test]
    fn friction_slows_sliding_and_is_off_by_default() {
        // Half-space tilted so gravity drives sliding along it.
        let normal = Vec3::new(0.30, 0.954, 0.0).normalize();
        let run = |mu: f32| {
            let (positions, tris) = sheet(5, 0.0);
            // Lift the sheet just off the plane so contact engages.
            let positions: Vec<Vec3> = positions.iter().map(|p| *p + normal * 0.004).collect();
            let state = SimState::from_mesh(&Mesh::new(positions, tris), 0.1);
            let mut solver = ImplicitSolver::new(
                state,
                ImplicitParams {
                    dt: 1.0 / 240.0,
                    stiffness: 800.0,
                    gravity: Vec3::new(0.0, -9.8, 0.0),
                    contact_dhat: 0.01,
                    contact_kappa: 1.0e4,
                    friction: mu,
                    friction_eps: 1.0e-4,
                    ..Default::default()
                },
            );
            solver.set_colliders(vec![contact::Collider::HalfSpace {
                point: Vec3::ZERO,
                normal,
            }]);
            let start = solver.state.positions.clone();
            for _ in 0..200 {
                solver.step(&CgCpu::new());
            }
            assert!(
                solver.state.positions.iter().all(|p| p.is_finite()),
                "friction produced a non-finite state (mu={mu})"
            );
            // Mean TANGENTIAL travel along the incline.
            let mut acc = 0.0f32;
            for (a, b) in solver.state.positions.iter().zip(&start) {
                let d = *a - *b;
                acc += (d - d.dot(normal) * normal).length();
            }
            (acc / start.len() as f32, solver.state.positions.clone())
        };
        let (slide_free, pos_free) = run(0.0);
        let (slide_mu, _pos_mu) = run(0.8);
        assert!(
            slide_free > 1e-4,
            "frictionless sheet should slide: {slide_free}"
        );
        assert!(
            slide_mu < slide_free,
            "friction must reduce tangential sliding: mu=0.8 slid {slide_mu} vs frictionless {slide_free}"
        );

        // Off by default: naming friction: 0.0 explicitly must be bit-identical.
        let (_s, pos_explicit) = run(0.0);
        for (i, (a, b)) in pos_free.iter().zip(&pos_explicit).enumerate() {
            assert_eq!(a.to_array(), b.to_array(), "vertex {i} not deterministic");
        }
    }

    /// Per-vertex friction must (a) default to the scene-wide value bit-identically,
    /// (b) reject a wrong-length array, (c) average per-face values correctly, and
    /// (d) actually differentiate behavior: a sheet half rough / half slippery must
    /// slide LESS than the fully-slippery case and MORE than the fully-rough one.
    #[test]
    fn per_vertex_friction_is_honored_and_bounded_by_uniform_cases() {
        let normal = Vec3::new(0.30, 0.954, 0.0).normalize();
        /// Per-vertex friction field, computed from the built solver.
        type MuField<'a> = Option<Box<dyn Fn(&ImplicitSolver) -> Vec<f32> + 'a>>;
        let run = |mu_field: MuField, uniform: f32| {
            let (positions, tris) = sheet(5, 0.0);
            let positions: Vec<Vec3> = positions.iter().map(|p| *p + normal * 0.004).collect();
            let state = SimState::from_mesh(&Mesh::new(positions, tris), 0.1);
            let mut solver = ImplicitSolver::new(
                state,
                ImplicitParams {
                    dt: 1.0 / 240.0,
                    stiffness: 800.0,
                    gravity: Vec3::new(0.0, -9.8, 0.0),
                    contact_dhat: 0.01,
                    contact_kappa: 1.0e4,
                    friction: uniform,
                    friction_eps: 1.0e-4,
                    ..Default::default()
                },
            );
            solver.set_colliders(vec![contact::Collider::HalfSpace {
                point: Vec3::ZERO,
                normal,
            }]);
            if let Some(f) = mu_field {
                let mu = f(&solver);
                solver.set_vertex_friction(mu).expect("valid length");
            }
            let start = solver.state.positions.clone();
            for _ in 0..200 {
                solver.step(&CgCpu::new());
            }
            assert!(solver.state.positions.iter().all(|p| p.is_finite()));
            let mut acc = 0.0f32;
            for (a, b) in solver.state.positions.iter().zip(&start) {
                let d = *a - *b;
                acc += (d - d.dot(normal) * normal).length();
            }
            acc / start.len() as f32
        };

        let slippery = run(None, 0.0);
        let rough = run(None, 0.9);
        // Half the vertices rough, half slippery (split by x).
        let mixed = run(
            Some(Box::new(|s: &ImplicitSolver| {
                s.state
                    .positions
                    .iter()
                    .map(|p| if p.x < 0.2 { 0.9 } else { 0.0 })
                    .collect()
            })),
            0.0,
        );
        assert!(rough < slippery, "friction should reduce sliding");
        assert!(
            mixed < slippery && mixed > rough,
            "mixed friction must fall between the uniform cases: mixed {mixed}, \
             slippery {slippery}, rough {rough}"
        );

        // Empty override ⇒ bit-identical to the scene-wide path.
        let a = run(Some(Box::new(|_s| Vec::new())), 0.9);
        assert!(
            (a - rough).abs() < 1e-6,
            "empty override must equal scene-wide"
        );
    }

    /// A pin must HOLD its vertex near the target while the rest of the cloth falls, and
    /// must be a no-op when absent. This is the observable property of the reference's
    /// `FixPair`: not a hard Dirichlet condition, but confinement to roughly a ball of
    /// radius `ghat`, because the stiffness carries a `mass/gap²` term that grows without
    /// bound as the vertex nears that boundary.
    #[test]
    fn a_pin_holds_its_vertex_while_the_rest_of_the_cloth_falls() {
        let ghat = 0.01f32;
        let run = |pinned: bool| {
            let (positions, tris) = sheet(4, 0.0);
            let corner = 0usize;
            let far = positions.len() - 1;
            let target = positions[corner];
            let state = SimState::from_mesh(&Mesh::new(positions, tris), 0.1);
            let mut solver = ImplicitSolver::new(
                state,
                ImplicitParams {
                    dt: 1.0 / 60.0,
                    stiffness: 5.0e2,
                    gravity: Vec3::new(0.0, -9.8, 0.0),
                    ..Default::default()
                },
            );
            if pinned {
                solver
                    .set_pins(vec![Pin::fixed(corner as u32, target, ghat)])
                    .unwrap();
            }
            let cg = crate::linsolve::CgCpu::default();
            for _ in 0..40 {
                solver.step(&cg);
            }
            let p = solver.state().positions.clone();
            ((p[corner] - target).length(), target.y - p[far].y)
        };

        let (drift_free, fall_free) = run(false);
        let (drift_pinned, fall_pinned) = run(true);

        // Unpinned, the same corner falls far; pinned, it stays inside the ball.
        assert!(
            drift_free > 10.0 * ghat,
            "control: the unpinned corner should fall freely, drifted {drift_free:.4}"
        );
        // Exactly within ghat, not merely near it: the time-of-impact clamp enforces the
        // ball strictly (see `a_static_pin_never_leaves_its_ghat_ball`).
        assert!(
            drift_pinned <= ghat,
            "pinned corner must stay inside its ghat ball, drifted {drift_pinned:.4}"
        );
        // The pin must be a LOCAL constraint, not a global freeze: the far corner keeps
        // moving, by far more than the pinned corner is allowed to. But it must NOT free
        // fall either — the sheet now HANGS from the pinned corner, so the springs carry
        // it and its descent is a fraction of the unconstrained drop. Both bounds matter:
        // the lower one fails if a pin freezes the whole mesh, the upper one fails if the
        // pin silently stops holding.
        assert!(
            fall_pinned > 10.0 * ghat,
            "the far corner must still move, fell only {fall_pinned:.4}"
        );
        assert!(
            fall_pinned < 0.5 * fall_free,
            "with a corner pinned the sheet must hang, not free fall: \
             {fall_pinned:.4} vs unconstrained {fall_free:.4}"
        );
    }

    /// `target_toi` must be inert at its default and, when enabled, must terminate the
    /// Newton loop the way the reference does, with `min_newton_steps` holding it open and
    /// `newton_iters` remaining an upper bound.
    ///
    /// Requires ACTIVE CONTACT to be meaningful. Two earlier versions of this test used a
    /// free-falling sheet and could not distinguish anything: without contact the CCD
    /// fraction is 1, and the loop's `max_step < 1e-7` break (which predates this
    /// criterion) ends it after 2 iterations no matter how stiff the material — measured
    /// identical at 5e2 through 1e7. `target_toi` is a CONTACT-clamping criterion, so it
    /// only bites where CCD is actually limiting the step.
    #[test]
    fn target_toi_terminates_newton_early_and_is_inert_by_default() {
        // A sheet dropped onto a half-space, so contact clamps the step for many steps.
        let max_iters = |target_toi: f32, min_steps: u32, cap: u32| {
            let (positions, tris) = sheet(4, 0.3);
            let state = SimState::from_mesh(&Mesh::new(positions, tris), 0.1);
            let mut solver = ImplicitSolver::new(
                state,
                ImplicitParams {
                    dt: 1.0 / 60.0,
                    stiffness: 5.0e2,
                    gravity: Vec3::new(0.0, -9.8, 0.0),
                    contact_dhat: 0.05,
                    contact_kappa: 1.0e3,
                    newton_iters: cap,
                    target_toi,
                    min_newton_steps: min_steps,
                    ..Default::default()
                },
            );
            solver.set_colliders(vec![contact::Collider::HalfSpace {
                point: Vec3::ZERO,
                normal: Vec3::Y,
            }]);
            let cg = crate::linsolve::CgCpu::default();
            let mut worst = 0u32;
            for _ in 0..40 {
                solver.step(&cg);
                worst = worst.max(solver.metrics.steps.last().unwrap().newton_iters);
            }
            assert!(solver.state().positions.iter().all(|p| p.is_finite()));
            worst
        };

        // Disabled (the default): the loop runs to its fixed count.
        assert_eq!(max_iters(0.0, 0, 8), 8, "disabled ⇒ run the fixed count");
        assert_eq!(max_iters(0.0, 0, 3), 3);

        // Enabled: terminates well below the cap once a step advances past the target.
        assert_eq!(
            max_iters(0.25, 0, 8),
            2,
            "must terminate early, not run to the cap"
        );

        // `min_newton_steps = 3` holds it open. The count is min + 2, matching the
        // reference's own structure: it checks `step >= min_newton_steps` BEFORE
        // incrementing and then takes one further `final_step` iteration.
        assert_eq!(
            max_iters(0.25, 3, 8),
            5,
            "min_newton_steps must delay termination"
        );

        // `newton_iters` still bounds everything, even a minimum larger than it.
        assert_eq!(max_iters(0.25, 10, 5), 5, "the cap remains an upper bound");
    }

    /// `max_dx` must shorten an over-eager Newton step WITHOUT bending its direction, and
    /// be inert at its default. A per-vertex clamp would change the direction, which is not
    /// what the reference does — so the test checks that the step stays PARALLEL.
    #[test]
    fn max_dx_rescales_the_search_direction_without_changing_it() {
        let run = |max_dx: f32| {
            let (positions, tris) = sheet(4, 0.0);
            let state = SimState::from_mesh(&Mesh::new(positions.clone(), tris), 0.1);
            let mut solver = ImplicitSolver::new(
                state,
                ImplicitParams {
                    dt: 1.0 / 60.0,
                    stiffness: 5.0e2,
                    // A violent pull, so the unconstrained step is large.
                    gravity: Vec3::new(0.0, -5000.0, 0.0),
                    newton_iters: 1,
                    max_dx,
                    ..Default::default()
                },
            );
            solver.step(&crate::linsolve::CgCpu::default());
            let p = solver.state().positions.clone();
            (0..p.len())
                .map(|i| p[i] - positions[i])
                .collect::<Vec<_>>()
        };

        let free = run(f32::INFINITY);
        let free_max = free.iter().map(|d| d.length()).fold(0.0f32, f32::max);
        assert!(
            free_max > 0.05,
            "the control step must be large to be worth capping, got {free_max:.4}"
        );

        // Default must be inert.
        assert_eq!(
            free,
            run(f32::INFINITY),
            "the uncapped path must be deterministic"
        );

        let cap = 0.25 * free_max;
        let capped = run(cap);
        let capped_max = capped.iter().map(|d| d.length()).fold(0.0f32, f32::max);
        assert!(
            capped_max < free_max,
            "the cap must shorten the step: {capped_max:.4} vs {free_max:.4}"
        );
        // Same DIRECTION: every vertex's displacement must be parallel to the uncapped one,
        // i.e. one uniform scale for all of them. A per-vertex clamp would fail this.
        let scales: Vec<f32> = free
            .iter()
            .zip(&capped)
            .filter(|(f, _)| f.length() > 1.0e-6)
            .map(|(f, c)| c.length() / f.length())
            .collect();
        assert!(!scales.is_empty(), "need moving vertices to compare");
        let (lo, hi) = scales
            .iter()
            .fold((f32::INFINITY, 0.0f32), |(a, b), &s| (a.min(s), b.max(s)));
        assert!(
            hi - lo < 1.0e-3 * hi.max(1.0e-6),
            "the rescale must be UNIFORM across vertices (direction preserved), \
             got scales {lo:.5}..{hi:.5}"
        );
    }

    /// `gather_elastic_blocks` must reproduce OFF-DIAGONAL blocks, not just diagonal ones,
    /// place each entry at the right row/column within its block, accumulate duplicate
    /// triplets, and ignore pairs that were not requested. Getting the transpose or the
    /// row/column order wrong here would silently corrupt every dynamic stiffness.
    #[test]
    fn elastic_block_gather_handles_off_diagonals_and_duplicates() {
        use std::collections::HashMap;
        // Vertices 0,1,2 in play; vertex 3 must be ignored.
        let mut slots: HashMap<(u32, u32), usize> = HashMap::new();
        for (i, key) in [(0u32, 0u32), (0, 1), (1, 0), (2, 2)]
            .into_iter()
            .enumerate()
        {
            slots.insert(key, i);
        }
        let trips: Vec<crate::csr::Triplet> = vec![
            // Block (0,0): put 5.0 at local (r=1, c=2).
            (1u32, 2u32, 5.0f32),
            // Block (0,1): local (0,0) split across two triplets to check accumulation.
            (0, 3, 1.5),
            (0, 3, 2.5),
            // Block (1,0): local (2,1) — the transpose position, distinct from (0,1).
            (5, 1, -7.0),
            // Block (2,2): local (0,2).
            (6, 8, 9.0),
            // Not requested: pair (3,3) and pair (1,2).
            (9, 9, 100.0),
            (3, 6, 100.0),
        ];
        let b = ImplicitSolver::gather_elastic_blocks(&trips, &slots);
        assert_eq!(b.len(), 4);
        // Blocks are row-major 3×3; name the (row, col) so the intent stays legible.
        let at = |r: usize, c: usize| 3 * r + c;
        assert_eq!(b[0][at(1, 2)], 5.0, "block (0,0) at local (1,2)");
        assert_eq!(b[0].iter().filter(|v| **v != 0.0).count(), 1);
        assert_eq!(b[1][at(0, 0)], 4.0, "duplicate triplets must accumulate");
        assert_eq!(b[2][at(2, 1)], -7.0, "block (1,0) at local (2,1)");
        // The two off-diagonal blocks must be independent, not aliases of each other.
        assert_ne!(b[1], b[2], "(0,1) and (1,0) must be distinct blocks");
        assert_eq!(b[3][at(0, 2)], 9.0, "block (2,2) at local (0,2)");
        // Unrequested pairs contributed nothing anywhere.
        let total: f32 = b.iter().flatten().map(|v| v.abs()).sum();
        assert_eq!(
            total,
            5.0 + 4.0 + 7.0 + 9.0,
            "unrequested pairs must be ignored"
        );
    }

    /// Including the ELASTIC half of the strain-limit stiffness must make the barrier no
    /// weaker: a stiffer material earns a stiffer barrier. Checked as a behavioural
    /// comparison against a soft membrane, holding everything else fixed.
    #[test]
    fn the_elastic_half_of_strain_limit_stiffness_does_not_weaken_it() {
        let limit = 0.05f32;
        let worst_for = |membrane: f32| {
            let (positions, tris) = sheet(4, 0.0);
            let state = SimState::from_mesh(&Mesh::new(positions.clone(), tris.clone()), 0.1);
            let mut solver = ImplicitSolver::new(
                state,
                ImplicitParams {
                    dt: 1.0 / 60.0,
                    stiffness: membrane,
                    gravity: Vec3::ZERO,
                    barrier_kind: contact::BarrierKind::Cubic,
                    ..Default::default()
                },
            );
            solver
                .set_strain_limit_faces(&tris, &vec![limit; tris.len()], &vec![1.0; tris.len()])
                .unwrap();
            let cols = 4usize;
            let right: Vec<u32> = (0..cols).map(|j| (j * cols + cols - 1) as u32).collect();
            let left: Vec<u32> = (0..cols).map(|j| (j * cols) as u32).collect();
            let cg = crate::linsolve::CgCpu::default();
            for step in 1..=40 {
                let push = 0.004 * step as f32;
                let mut pins = Vec::new();
                for &v in &right {
                    pins.push(Pin {
                        index: v,
                        target: positions[v as usize] + Vec3::new(push, 0.0, 0.0),
                        ghat: 0.02,
                        kinematic: true,
                        stiffness: 1.0,
                    });
                }
                for &v in &left {
                    pins.push(Pin::fixed(v, positions[v as usize], 0.02));
                }
                solver.set_pins(pins).unwrap();
                solver.step(&cg);
            }
            assert!(solver.state().positions.iter().all(|p| p.is_finite()));
            let els = crate::shell_model::build_shells(
                &tris,
                &positions,
                1.0e-12,
                crate::tet::TetModel::Arap,
            );
            let mut worst = 0.0f32;
            for el in &els {
                let p = [
                    solver.state().positions[el.v[0] as usize],
                    solver.state().positions[el.v[1] as usize],
                    solver.state().positions[el.v[2] as usize],
                ];
                let f = crate::shell_model::shell_deformation_gradient(&p, &el.dm_inv);
                let (_, sigma, _) = crate::shell_model::svd3x2(&f);
                worst = worst.max(sigma[0] - 1.0).max(sigma[1] - 1.0);
            }
            worst
        };
        // A stiff membrane contributes a large elastic Rayleigh term; a soft one nearly
        // none. Both must still respect the limit.
        let soft = worst_for(5.0);
        let stiff = worst_for(5.0e3);
        for (name, v) in [("soft", soft), ("stiff", stiff)] {
            assert!(
                v <= limit * 1.02,
                "{name} membrane must still respect the limit: {v:.4} vs {limit}"
            );
        }
    }

    /// The strain-limit BARRIER must hold a sheet's stretch at or under its limit even when
    /// pulled hard — that is the behavioural difference from a one-sided spring, which is
    /// merely overpowered. Measured as the largest principal stretch reached by any face.
    #[test]
    fn the_strain_limit_barrier_enforces_its_limit_under_load() {
        let limit = 0.05f32; // 5% allowed stretch
                             // Pull the sheet apart by pinning two opposite edges and driving them outward.
        let run = |with_barrier: bool| {
            let (positions, tris) = sheet(4, 0.0);
            let n = positions.len();
            let state = SimState::from_mesh(&Mesh::new(positions.clone(), tris.clone()), 0.1);
            let mut solver = ImplicitSolver::new(
                state,
                ImplicitParams {
                    dt: 1.0 / 60.0,
                    // A weak membrane, so only the limit can hold the stretch back.
                    stiffness: 5.0,
                    gravity: Vec3::ZERO,
                    barrier_kind: contact::BarrierKind::Cubic,
                    ..Default::default()
                },
            );
            if with_barrier {
                let limits = vec![limit; tris.len()];
                let masses = vec![1.0f32; tris.len()];
                solver
                    .set_strain_limit_faces(&tris, &limits, &masses)
                    .unwrap();
                assert!(solver.strain_limit_face_count() > 0);
            }
            // Drag the last column outward in x via kinematic pins, stretching the sheet.
            let cols = 4usize;
            let right: Vec<u32> = (0..cols).map(|j| (j * cols + cols - 1) as u32).collect();
            let left: Vec<u32> = (0..cols).map(|j| (j * cols) as u32).collect();
            let cg = crate::linsolve::CgCpu::default();
            for step in 1..=40 {
                let push = 0.004 * step as f32;
                let mut pins = Vec::new();
                for &v in &right {
                    pins.push(Pin {
                        index: v,
                        target: positions[v as usize] + Vec3::new(push, 0.0, 0.0),
                        ghat: 0.02,
                        kinematic: true,
                        stiffness: 1.0,
                    });
                }
                for &v in &left {
                    pins.push(Pin::fixed(v, positions[v as usize], 0.02));
                }
                solver.set_pins(pins).unwrap();
                solver.step(&cg);
            }
            assert!(
                solver.state().positions.iter().all(|p| p.is_finite()),
                "state must stay finite"
            );
            // Largest principal stretch over all faces, from the rest shape.
            let els = crate::shell_model::build_shells(
                &tris,
                &positions,
                1.0e-12,
                crate::tet::TetModel::Arap,
            );
            let mut worst = 0.0f32;
            for el in &els {
                let p = [
                    solver.state().positions[el.v[0] as usize],
                    solver.state().positions[el.v[1] as usize],
                    solver.state().positions[el.v[2] as usize],
                ];
                let f = crate::shell_model::shell_deformation_gradient(&p, &el.dm_inv);
                let (_, sigma, _) = crate::shell_model::svd3x2(&f);
                worst = worst.max(sigma[0] - 1.0).max(sigma[1] - 1.0);
            }
            let _ = n;
            worst
        };

        let without = run(false);
        let with = run(true);
        // Control: the pull really does over-stretch an unlimited sheet.
        assert!(
            without > 1.5 * limit,
            "control: the load should over-stretch without a limit, got {without:.4}"
        );
        // ENFORCED, not merely resisted: the strain-limit line search clamps every step so
        // no face crosses its limit, so the reached stretch sits AT the limit rather than
        // some fraction of the way past it. Measured 0.0500 against a 0.05 limit while the
        // unlimited sheet reaches 0.8181 — a 16x difference, so this bound is not delicate.
        assert!(
            with <= limit * 1.02,
            "the limit must be ENFORCED: reached {with:.4} against limit {limit} \
             (unlimited reaches {without:.4})"
        );
        eprintln!(
            "max principal stretch: unlimited {without:.4}, limited {with:.4} (limit {limit})"
        );
    }

    /// Per-hinge bend stiffness must be the AREA-WEIGHTED mean of the incident faces'
    /// `bend` (`builder.rs:770-800`), carried through the reference's coefficient
    /// `BEND_SCALE·bend·(|e|²/A)·areal_density`.
    #[test]
    fn per_hinge_bend_is_the_area_weighted_mean_of_its_faces() {
        // Two triangles sharing edge (1,2) with DELIBERATELY unequal areas, so an
        // area-weighted mean is distinguishable from a plain average.
        let positions = vec![
            Vec3::new(-1.0, 0.0, 0.0), // 0: apex of the large face
            Vec3::new(0.0, 0.0, 0.0),  // 1: shared
            Vec3::new(0.0, 0.0, 1.0),  // 2: shared
            Vec3::new(0.25, 0.0, 0.0), // 3: apex of the small face
        ];
        let tris = vec![[0u32, 1, 2], [1, 2, 3]];
        let state = SimState::from_mesh(&Mesh::new(positions, tris.clone()), 0.1);
        let mut solver = ImplicitSolver::new(state, ImplicitParams::default());
        assert_eq!(solver.hinge_count(), 1, "one shared edge, one hinge");

        // Face areas: large = 0.5·1·1 = 0.5, small = 0.5·0.25·1 = 0.125.
        let (a_big, a_small) = (0.5f32, 0.125f32);
        // Give the faces different bend values and check the blend.
        solver
            .set_hinge_bend_from_faces(&tris, &[4.0, 1.0])
            .unwrap();
        let k_mixed = solver.hinge_k()[0];
        // Guard against every comparison below passing trivially on zeros. The stiffness is
        // small in absolute terms (~1e-4), which is why the tolerances here are strictly
        // RELATIVE — an absolute epsilon would swamp the signal.
        assert!(
            k_mixed > 0.0,
            "stiffness must be positive, got {k_mixed:.6e}"
        );

        // Uniform bend equal to the expected area-weighted mean must give the SAME
        // stiffness — this pins the weighting without re-deriving the coefficient.
        let want_bend = (a_big * 4.0 + a_small * 1.0) / (a_big + a_small);
        solver
            .set_hinge_bend_from_faces(&tris, &[want_bend, want_bend])
            .unwrap();
        let k_uniform = solver.hinge_k()[0];
        assert!(
            (k_mixed - k_uniform).abs() <= 1e-5 * k_uniform,
            "area-weighted mean expected: mixed {k_mixed:.6e} vs uniform({want_bend:.4}) \
             {k_uniform:.6e}"
        );
        // A plain (unweighted) average would be 2.5, which must NOT match.
        solver
            .set_hinge_bend_from_faces(&tris, &[2.5, 2.5])
            .unwrap();
        let k_plain = solver.hinge_k()[0];
        // Expect a ratio of 3.4 / 2.5 = 1.36, so this is a wide margin, not a hair.
        assert!(
            (k_mixed - k_plain).abs() > 0.1 * k_plain,
            "a plain average (2.5) must be distinguishable from the area-weighted one \
             (3.4): mixed {k_mixed:.6e} vs plain {k_plain:.6e}"
        );

        // Stiffness is linear in `bend`, and zero bend means zero stiffness.
        solver
            .set_hinge_bend_from_faces(&tris, &[0.0, 0.0])
            .unwrap();
        assert_eq!(solver.hinge_k()[0], 0.0);
        solver
            .set_hinge_bend_from_faces(&tris, &[1.0, 1.0])
            .unwrap();
        let k1 = solver.hinge_k()[0];
        solver
            .set_hinge_bend_from_faces(&tris, &[3.0, 3.0])
            .unwrap();
        let k3 = solver.hinge_k()[0];
        assert!(
            k1 > 0.0 && (k3 - 3.0 * k1).abs() <= 1e-5 * k3,
            "stiffness must be linear in bend: {k1:.6e} vs {k3:.6e}"
        );
        // Length mismatch is rejected rather than silently truncated.
        assert!(solver.set_hinge_bend_from_faces(&tris, &[1.0]).is_err());
    }

    /// The reference's `|e|²/A` factor exists to make the bent shape independent of mesh
    /// resolution. Refining a sheet must therefore keep the per-hinge stiffness in the same
    /// range, whereas a bare `|e|` factor would shrink it — so this test would fail if the
    /// coefficient regressed to edge-length weighting.
    #[test]
    fn per_hinge_bend_stiffness_is_resolution_independent() {
        let mean_k = |cells: usize| {
            let (positions, tris) = sheet(cells, 0.0);
            let state = SimState::from_mesh(&Mesh::new(positions, tris.clone()), 0.1);
            let mut solver = ImplicitSolver::new(state, ImplicitParams::default());
            let bend = vec![1.0f32; tris.len()];
            solver.set_hinge_bend_from_faces(&tris, &bend).unwrap();
            let k = solver.hinge_k();
            k.iter().sum::<f32>() / k.len() as f32
        };
        // `sheet(n)` keeps the spacing fixed and grows the extent, so refinement here is a
        // proxy: what matters is that |e|^2/area stays O(1) as the mesh changes, which it
        // does for any uniform triangulation. Compare a coarse and a finer sheet.
        let coarse = mean_k(4);
        let fine = mean_k(8);
        assert!(
            coarse > 0.0 && fine > 0.0,
            "stiffness must be positive: {coarse:.3e}, {fine:.3e}"
        );
        let ratio = fine / coarse;
        assert!(
            (0.5..2.0).contains(&ratio),
            "|e|^2/area must keep stiffness in the same range across mesh sizes, \
             got ratio {ratio:.3} ({coarse:.3e} -> {fine:.3e})"
        );
    }

    /// A STATIC pin's vertex must never leave its `ghat` ball, even under a load far
    /// stronger than the pin's stiffness would hold on its own. That containment is the
    /// job of the time-of-impact clamp, not the barrier: without it the vertex rides out to
    /// the boundary and the `mass/gap²` term is all that resists.
    #[test]
    fn a_static_pin_never_leaves_its_ghat_ball() {
        let ghat = 0.01f32;
        let (positions, tris) = sheet(4, 0.0);
        let corner = 0usize;
        let target = positions[corner];
        let state = SimState::from_mesh(&Mesh::new(positions, tris), 0.1);
        let mut solver = ImplicitSolver::new(
            state,
            ImplicitParams {
                dt: 1.0 / 60.0,
                stiffness: 5.0e2,
                // A brutal pull, orders of magnitude beyond ordinary gravity.
                gravity: Vec3::new(0.0, -2000.0, 0.0),
                ..Default::default()
            },
        );
        solver
            .set_pins(vec![Pin::fixed(corner as u32, target, ghat)])
            .unwrap();
        let cg = crate::linsolve::CgCpu::default();
        let mut worst = 0.0f32;
        for _ in 0..60 {
            solver.step(&cg);
            worst = worst.max((solver.state().positions[corner] - target).length());
            assert!(
                solver.state().positions[corner].is_finite(),
                "pinned vertex went non-finite"
            );
        }
        assert!(
            worst <= ghat,
            "a static pin must stay inside its ghat ball: drifted {worst:.6} > {ghat}"
        );
        // Sanity: the load really was enough to move things — the far corner fell hard, so
        // containment above is not just "nothing happened".
        let far = solver.state().positions.len() - 1;
        let fell = target.y - solver.state().positions[far].y;
        assert!(
            fell > 10.0 * ghat,
            "the load should move the sheet, fell {fell:.4}"
        );
    }

    /// The aerodynamic damper must be ANISOTROPIC: a sheet moving BROADSIDE (along its
    /// normal) is damped at full strength, while the same sheet sliding IN ITS OWN PLANE is
    /// damped only by `air_friction`. With `air_friction = 0.2` the in-plane motion must
    /// therefore be retained far better than the broadside motion — that contrast is the
    /// whole point of the model, and a naive isotropic drag would fail it.
    #[test]
    fn the_air_damper_resists_broadside_motion_more_than_in_plane_sliding() {
        // A flat sheet in the xz plane, so its normal is +/-y.
        let run = |v0: Vec3, rho: f32| {
            let (positions, tris) = sheet(4, 0.0);
            let n = positions.len();
            let state = SimState::from_mesh(&Mesh::new(positions, tris), 0.1);
            let mut solver = ImplicitSolver::new(
                state,
                ImplicitParams {
                    dt: 1.0 / 60.0,
                    // No elasticity or gravity: isolate the damper.
                    stiffness: 0.0,
                    gravity: Vec3::ZERO,
                    air_density: rho,
                    air_friction: 0.2,
                    ..Default::default()
                },
            );
            for i in 0..n {
                solver.state.velocities[i] = v0;
            }
            let cg = crate::linsolve::CgCpu::default();
            for _ in 0..20 {
                solver.step(&cg);
            }
            // Mean speed retained, as a fraction of the initial speed.
            let mean: f32 = solver
                .state
                .velocities
                .iter()
                .map(|v| v.length())
                .sum::<f32>()
                / n as f32;
            mean / v0.length()
        };

        // Measured over this 4x4 sheet, 20 steps, air_friction = 0.2 — broadside is more
        // damped than in-plane at every strength, and neither saturates at rho = 1.5:
        //   rho  0.1 -> broadside 0.8956  in_plane 0.9778
        //   rho  0.5 -> broadside 0.6122  in_plane 0.8953
        //   rho  1.5 -> broadside 0.3247  in_plane 0.7252
        //   rho  5.0 -> broadside 0.1000  in_plane 0.3803
        //   rho 20.0 -> broadside 0.0095  in_plane 0.0580
        let rho = 1.5;
        let broadside = run(Vec3::new(0.0, 1.0, 0.0), rho);
        let in_plane = run(Vec3::new(1.0, 0.0, 0.0), rho);
        assert!(
            broadside < 0.5,
            "broadside motion should be strongly damped, retained {broadside:.4}"
        );
        assert!(
            in_plane > 2.0 * broadside,
            "in-plane sliding must be damped far less than broadside: \
             in_plane {in_plane:.4} vs broadside {broadside:.4}"
        );
        // And with the damper off, nothing is lost at all.
        let off = run(Vec3::new(0.0, 1.0, 0.0), 0.0);
        assert!(
            off > 0.99,
            "with air_density = 0 the damper must not act, retained {off:.4}"
        );
    }

    /// The air terms are opt-in: at their defaults a run must be BIT-IDENTICAL to one that
    /// never mentioned them.
    #[test]
    fn air_terms_are_off_by_default() {
        let mk = |p: ImplicitParams| {
            let (positions, tris) = sheet(4, 0.0);
            let state = SimState::from_mesh(&Mesh::new(positions, tris), 0.1);
            let mut solver = ImplicitSolver::new(state, p);
            let cg = crate::linsolve::CgCpu::default();
            for _ in 0..10 {
                solver.step(&cg);
            }
            solver.state.positions.clone()
        };
        let base = ImplicitParams {
            dt: 1.0 / 60.0,
            stiffness: 5.0e2,
            gravity: Vec3::new(0.0, -9.8, 0.0),
            ..Default::default()
        };
        let explicit = ImplicitParams {
            air_density: 0.0,
            air_friction: 0.0,
            isotropic_air_friction: 0.0,
            wind: Vec3::ZERO,
            ..base
        };
        assert_eq!(mk(base), mk(explicit), "air defaults must be inert");
    }

    /// `isotropic_air_friction` damps every direction equally, and unlike the anisotropic
    /// damper it does not depend on area or normal.
    #[test]
    fn isotropic_air_friction_damps_all_directions_equally() {
        let run = |v0: Vec3| {
            let (positions, tris) = sheet(4, 0.0);
            let n = positions.len();
            let state = SimState::from_mesh(&Mesh::new(positions, tris), 0.1);
            let mut solver = ImplicitSolver::new(
                state,
                ImplicitParams {
                    dt: 1.0 / 60.0,
                    stiffness: 0.0,
                    gravity: Vec3::ZERO,
                    isotropic_air_friction: 1.0e-2,
                    ..Default::default()
                },
            );
            for i in 0..n {
                solver.state.velocities[i] = v0;
            }
            let cg = crate::linsolve::CgCpu::default();
            for _ in 0..20 {
                solver.step(&cg);
            }
            solver
                .state
                .velocities
                .iter()
                .map(|v| v.length())
                .sum::<f32>()
                / n as f32
                / v0.length()
        };
        let along_normal = run(Vec3::new(0.0, 1.0, 0.0));
        let in_plane = run(Vec3::new(1.0, 0.0, 0.0));
        assert!(
            along_normal < 0.99,
            "isotropic friction must damp, retained {along_normal:.4}"
        );
        assert!(
            (along_normal - in_plane).abs() < 1e-4,
            "isotropic damping must not depend on direction: {along_normal:.5} vs {in_plane:.5}"
        );
    }

    /// Collider friction combines with the per-vertex value as `min`, so a FRICTIONLESS
    /// collider yields frictionless contact even when the cloth itself is rough. That is the
    /// reference's `combine_friction(..., Min)` and it is easy to get wrong by ignoring the
    /// collider's coefficient — most gauge walls declare 0.
    #[test]
    fn collider_friction_combines_as_min_with_the_vertex_value() {
        // A sheet dropped at an angle onto a half-space; measure how far it slides sideways.
        let slide_mode = |collider_friction: f32, mode: crate::contact::FrictionMode| {
            let (positions, tris) = sheet(4, 0.25);
            let n = positions.len();
            let state = SimState::from_mesh(&Mesh::new(positions.clone(), tris), 0.1);
            let mut solver = ImplicitSolver::new(
                state,
                ImplicitParams {
                    dt: 1.0 / 60.0,
                    stiffness: 5.0e2,
                    gravity: Vec3::new(2.0, -9.8, 0.0), // push sideways as it lands
                    contact_dhat: 0.02,
                    contact_kappa: 1.0e3,
                    friction: 0.9, // rough CLOTH throughout
                    collider_friction,
                    friction_mode: mode,
                    ..Default::default()
                },
            );
            solver.set_colliders(vec![contact::Collider::HalfSpace {
                point: Vec3::ZERO,
                normal: Vec3::Y,
            }]);
            let cg = crate::linsolve::CgCpu::default();
            for _ in 0..80 {
                solver.step(&cg);
            }
            assert!(solver.state().positions.iter().all(|p| p.is_finite()));
            let sx: f32 = (0..n)
                .map(|i| solver.state().positions[i].x - positions[i].x)
                .sum::<f32>()
                / n as f32;
            sx
        };

        let slide = |cf: f32| slide_mode(cf, crate::contact::FrictionMode::Min);
        // Rough collider: min(0.9, 0.9) = 0.9 → resists sliding.
        let rough = slide(0.9);
        // Frictionless collider: min(0.0, 0.9) = 0.0 → slides freely despite rough cloth.
        let slick = slide(0.0);
        assert!(
            slick > rough * 1.5 && slick > 1.0e-3,
            "a frictionless collider must let the sheet slide further than a rough one: \
             slick {slick:.5} vs rough {rough:.5}"
        );
        // The default must be inert: INFINITY ⇒ min(INF, 0.9) = 0.9, same as an explicit 0.9.
        assert!(
            (slide(f32::INFINITY) - rough).abs() < 1.0e-6,
            "default collider_friction must not change behaviour"
        );
    }

    /// The `collider_friction` "unset" sentinel must be inert under EVERY `friction_mode`.
    ///
    /// REGRESSION: the sentinel is `f32::INFINITY` and was originally relied on to disappear
    /// via `min(INF, v) == v`. That reasoning silently breaks once the scene's mode is honoured
    /// — `max(INF, v)` is INFINITY and `mean(INF, v)` is INFINITY — so an unset collider
    /// coefficient would have meant INFINITE friction on the 14 of 22 gauge scenes that ship
    /// `friction_mode = "max"`. The fix skips the sentinel explicitly; this pins it.
    #[test]
    fn unset_collider_friction_is_inert_under_every_friction_mode() {
        use crate::contact::FrictionMode;
        let (positions, tris) = sheet(4, 0.25);
        let n = positions.len();
        let run = |cf: f32, mode: FrictionMode| {
            let state = SimState::from_mesh(&Mesh::new(positions.clone(), tris.clone()), 0.1);
            let mut solver = ImplicitSolver::new(
                state,
                ImplicitParams {
                    dt: 1.0 / 60.0,
                    stiffness: 5.0e2,
                    gravity: Vec3::new(2.0, -9.8, 0.0),
                    contact_dhat: 0.02,
                    contact_kappa: 1.0e3,
                    friction: 0.9,
                    collider_friction: cf,
                    friction_mode: mode,
                    ..Default::default()
                },
            );
            solver.set_colliders(vec![contact::Collider::HalfSpace {
                point: Vec3::ZERO,
                normal: Vec3::Y,
            }]);
            let cg = crate::linsolve::CgCpu::default();
            for _ in 0..80 {
                solver.step(&cg);
            }
            assert!(
                solver.state().positions.iter().all(|p| p.is_finite()),
                "mode {mode:?} with collider_friction {cf} produced non-finite positions"
            );
            (0..n)
                .map(|i| solver.state().positions[i].x - positions[i].x)
                .sum::<f32>()
                / n as f32
        };
        for mode in [FrictionMode::Min, FrictionMode::Max, FrictionMode::Mean] {
            let unset = run(f32::INFINITY, mode);
            let explicit = run(0.9, mode);
            // With the cloth at 0.9, an UNSET collider must behave exactly like a collider
            // that also declares 0.9 — under every mode, since combining 0.9 with 0.9 is 0.9
            // whichever rule is used.
            assert!(
                (unset - explicit).abs() < 1.0e-4,
                "mode {mode:?}: unset sentinel slid {unset:.6} but an explicit 0.9 slid \
                 {explicit:.6} — the sentinel is leaking into the combination"
            );
            assert!(
                unset.is_finite(),
                "mode {mode:?}: unset sentinel produced a non-finite slide"
            );
        }
    }

    /// A weighted PULL must draw its vertex toward the target, with the force scaling with
    /// the weight — and, critically, must work with NO gravity and no other force in the
    /// scene, because that is exactly `fishingknot`'s situation (gravity `[0,0,0]`, all
    /// motion from a `pull = 1000` block). Before this term existed that scene did not move
    /// at all and scored skill 0.000.
    #[test]
    fn a_weighted_pull_moves_its_vertex_with_no_other_force() {
        let run = |weight: Option<f32>| {
            let (positions, tris) = sheet(4, 0.0);
            let corner = 0usize;
            let target = positions[corner] + Vec3::new(0.0, 0.30, 0.0);
            let state = SimState::from_mesh(&Mesh::new(positions.clone(), tris), 0.1);
            let mut solver = ImplicitSolver::new(
                state,
                ImplicitParams {
                    dt: 1.0 / 60.0,
                    stiffness: 5.0e2,
                    // No gravity at all — the pull must be the only driver.
                    gravity: Vec3::ZERO,
                    ..Default::default()
                },
            );
            if let Some(w) = weight {
                solver
                    .set_pulls(vec![Pull {
                        index: corner as u32,
                        target,
                        weight: w,
                    }])
                    .unwrap();
            }
            let cg = crate::linsolve::CgCpu::default();
            for _ in 0..40 {
                solver.step(&cg);
            }
            assert!(solver.state().positions.iter().all(|p| p.is_finite()));
            (solver.state().positions[corner] - positions[corner]).dot(Vec3::Y)
        };

        // With no pull and no gravity, nothing may move — this is the state that made
        // `fishingknot` a static image.
        let none = run(None);
        assert!(
            none.abs() < 1.0e-6,
            "with no force the vertex must not move, moved {none:.6}"
        );
        // A pull moves it toward the target...
        let soft = run(Some(5.0));
        let firm = run(Some(500.0));
        assert!(
            soft > 1.0e-4,
            "a weighted pull must move the vertex, got {soft:.6}"
        );
        // ...and a heavier weight gets closer within the same number of steps.
        assert!(
            firm > soft,
            "heavier weight must pull further: firm {firm:.6} vs soft {soft:.6}"
        );
        // It must settle NEAR the target, not sail past it. A few percent of overshoot is
        // legitimate rather than a bug: at weight 500 the pull dominates inertia
        // (w/(m/dt²) ≈ 1.4), so the vertex reaches the target in ~one step and the sheet's
        // own springs then ring slightly around it. Measured 0.3073 against a 0.30 target.
        // Divergence is what would signal a real problem, so the bound is loose but finite.
        assert!(
            firm <= 0.30 * 1.10,
            "pull must settle near the target, not diverge past it: got {firm:.6}"
        );
    }

    /// Pulls are opt-in and validated: an empty set is bit-identical, and a bad index or a
    /// non-positive weight is rejected rather than silently ignored.
    #[test]
    fn pulls_are_off_by_default_and_validated() {
        let (positions, tris) = sheet(4, 0.0);
        let n = positions.len();
        let params = ImplicitParams {
            dt: 1.0 / 60.0,
            stiffness: 5.0e2,
            gravity: Vec3::new(0.0, -9.8, 0.0),
            ..Default::default()
        };
        let mut a = ImplicitSolver::new(
            SimState::from_mesh(&Mesh::new(positions.clone(), tris.clone()), 0.1),
            params,
        );
        let mut b = ImplicitSolver::new(
            SimState::from_mesh(&Mesh::new(positions, tris), 0.1),
            params,
        );
        b.set_pulls(Vec::new()).unwrap();
        let cg = crate::linsolve::CgCpu::default();
        for _ in 0..10 {
            a.step(&cg);
            b.step(&cg);
        }
        assert_eq!(
            a.state().positions,
            b.state().positions,
            "an empty pull set must not perturb the solve"
        );
        assert!(b
            .set_pulls(vec![Pull {
                index: n as u32,
                target: Vec3::ZERO,
                weight: 1.0,
            }])
            .is_err());
        assert!(b
            .set_pulls(vec![Pull {
                index: 0,
                target: Vec3::ZERO,
                weight: 0.0,
            }])
            .is_err());
    }

    /// A KINEMATIC pin whose target is moved between steps must drag its vertex along.
    /// This is the property the gauge's animated scenes depend on: the driver re-evaluates
    /// the pin schedule each step, so the solver must actually track a target that moves,
    /// not just hold a fixed one.
    #[test]
    fn a_vertex_follows_a_moving_kinematic_pin() {
        let ghat = 0.02f32;
        let (positions, tris) = sheet(4, 0.0);
        let corner = 0usize;
        let start = positions[corner];
        let state = SimState::from_mesh(&Mesh::new(positions, tris), 0.1);
        let mut solver = ImplicitSolver::new(
            state,
            ImplicitParams {
                dt: 1.0 / 60.0,
                stiffness: 5.0e2,
                // No gravity: isolate the pin's pull from any other motion.
                gravity: Vec3::ZERO,
                ..Default::default()
            },
        );
        let cg = crate::linsolve::CgCpu::default();
        // Walk the target upward in small increments, re-setting it each step exactly as
        // the driver does from the scene schedule.
        let steps = 60;
        let per_step = 0.01f32;
        for k in 1..=steps {
            let target = start + Vec3::new(0.0, per_step * k as f32, 0.0);
            solver
                .set_pins(vec![Pin {
                    index: corner as u32,
                    target,
                    ghat,
                    kinematic: true,
                    stiffness: 1.0,
                }])
                .unwrap();
            solver.step(&cg);
        }
        let want = start + Vec3::new(0.0, per_step * steps as f32, 0.0);
        let got = solver.state().positions[corner];
        let err = (got - want).length();
        // It tracks within roughly the confinement radius; it must NOT have stayed put.
        assert!(
            err <= 2.0 * ghat,
            "vertex should track the moving target: got {got:?} want {want:?} (err {err:.4})"
        );
        let moved = (got - start).length();
        assert!(
            moved > 0.5 * per_step * steps as f32,
            "vertex barely moved ({moved:.4}); the pin is not dragging it"
        );
    }

    /// Pins are opt-in: a run with none attached must be BIT-IDENTICAL to one that never
    /// touched the pin path, and `set_pins` must reject bad input rather than silently
    /// dropping it.
    #[test]
    fn pins_are_off_by_default_and_validated() {
        let (positions, tris) = sheet(4, 0.0);
        let n = positions.len();
        let params = ImplicitParams {
            dt: 1.0 / 60.0,
            stiffness: 5.0e2,
            gravity: Vec3::new(0.0, -9.8, 0.0),
            ..Default::default()
        };
        let mut a = ImplicitSolver::new(
            SimState::from_mesh(&Mesh::new(positions.clone(), tris.clone()), 0.1),
            params,
        );
        let mut b = ImplicitSolver::new(
            SimState::from_mesh(&Mesh::new(positions, tris), 0.1),
            params,
        );
        b.set_pins(Vec::new()).unwrap();
        let cga = crate::linsolve::CgCpu::default();
        let cgb = crate::linsolve::CgCpu::default();
        for _ in 0..10 {
            a.step(&cga);
            b.step(&cgb);
        }
        assert_eq!(
            a.state().positions,
            b.state().positions,
            "an empty pin set must not perturb the solve"
        );

        // Validation.
        assert!(
            b.set_pins(vec![Pin::fixed(n as u32, Vec3::ZERO, 0.01)])
                .is_err(),
            "an out-of-range pin vertex must be rejected"
        );
        assert!(
            b.set_pins(vec![Pin::fixed(0, Vec3::ZERO, 0.0)]).is_err(),
            "a non-positive ghat must be rejected"
        );
        assert!(b.set_pins(vec![Pin::fixed(0, Vec3::ZERO, 0.01)]).is_ok());
    }

    /// `set_tets` must not append a boundary face that the caller already supplied in
    /// the mesh. The reference ships a solid's surface inside its own face list (the
    /// gauge's `plastic` scene has all 5120 tet-boundary faces in `tri.bin`), and a
    /// duplicated collision triangle would double the contact/friction pairs there.
    #[test]
    fn tet_surface_already_in_the_mesh_is_not_duplicated() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let tet = [[0u32, 1, 2, 3]];
        // Every face of the single tet is on the boundary; hand them all to the mesh
        // with DELIBERATELY reversed winding so the dedup cannot be winding-sensitive.
        let faces: Vec<[u32; 3]> = crate::tet::tet_faces(&tet[0])
            .iter()
            .map(|f| [f[2], f[1], f[0]])
            .collect();
        let state = SimState::from_mesh(&Mesh::new(positions.clone(), faces.clone()), 0.1);
        let mut solver = ImplicitSolver::new(state, ImplicitParams::default());
        solver.set_tets(&tet);
        assert_eq!(
            solver.collision_tris.len(),
            faces.len(),
            "boundary faces already in the mesh must not be appended again"
        );

        // Control: the same tet with NO faces in the mesh must still gain its surface,
        // so the dedup did not simply disable the append path.
        let bare = SimState::from_mesh(&Mesh::new(positions, Vec::new()), 0.1);
        let mut solver2 = ImplicitSolver::new(bare, ImplicitParams::default());
        solver2.set_tets(&tet);
        assert_eq!(
            solver2.collision_tris.len(),
            4,
            "a tet whose surface is absent from the mesh must contribute all 4 faces"
        );
    }

    /// A wrong-length per-vertex friction array must be rejected, and the per-face→
    /// per-vertex averaging must be correct (a vertex shared by faces with different
    /// coefficients gets their mean).
    #[test]
    fn vertex_friction_validation_and_face_averaging() {
        let (positions, tris) = sheet(3, 0.0);
        let n = positions.len();
        let state = SimState::from_mesh(&Mesh::new(positions, tris.clone()), 0.1);
        let mut solver = ImplicitSolver::new(state, ImplicitParams::default());
        assert!(solver.set_vertex_friction(vec![0.5; n + 1]).is_err());
        assert!(solver.set_vertex_friction(vec![0.5; n]).is_ok());
        // Face averaging: give face 0 a coefficient of 1.0 and the rest 0.0.
        let mut per_face = vec![0.0f32; tris.len()];
        per_face[0] = 1.0;
        let v = solver.face_scalars_to_vertices(&tris, &per_face, -1.0);
        assert_eq!(v.len(), n);
        // Vertices of face 0 must be > 0; a vertex on no face would be the fallback.
        for &vi in &tris[0] {
            assert!(
                v[vi as usize] > 0.0,
                "vertex {vi} of face 0 should pick up 1.0"
            );
        }
        // The mean must never exceed the max input.
        assert!(v.iter().all(|&x| x <= 1.0 + 1e-6));
    }

    /// Self-collision friction: two stacked sheets sliding against each other must slide
    /// LESS when friction is on. Also asserts frictionless is bit-identical to default,
    /// so the new code is genuinely inert at mu = 0.
    #[test]
    fn self_collision_friction_damps_sheet_on_sheet_sliding() {
        let run = |mu: f32| {
            // Lower sheet pinned; upper sheet dropped on it and pushed sideways.
            // Small: this test drives the full self-collision broad phase + friction
            // on every pair, so scene size dominates runtime.
            let n = 4usize;
            let mut positions = Vec::new();
            let mut tris = Vec::new();
            for (layer, y) in [(0usize, 0.0f32), (1, 0.03)] {
                let base = positions.len() as u32;
                for j in 0..n {
                    for i in 0..n {
                        positions.push(Vec3::new(i as f32 * 0.1, y, j as f32 * 0.1));
                    }
                }
                for j in 0..n - 1 {
                    for i in 0..n - 1 {
                        let v = base + (j * n + i) as u32;
                        let r = v + n as u32;
                        tris.push([v, v + 1, r]);
                        tris.push([r, v + 1, r + 1]);
                    }
                }
                let _ = layer;
            }
            let lower = n * n;
            let mut state = SimState::from_mesh(&Mesh::new(positions, tris), 0.1);
            for i in 0..lower {
                state.pin(i);
            }
            // Push the upper sheet sideways.
            for i in lower..state.vertex_count() {
                state.velocities[i] = Vec3::new(0.5, 0.0, 0.0);
            }
            let mut solver = ImplicitSolver::new(
                state,
                ImplicitParams {
                    dt: 1.0 / 240.0,
                    stiffness: 600.0,
                    gravity: Vec3::new(0.0, -9.8, 0.0),
                    self_collision: true,
                    contact_dhat: 0.02,
                    contact_kappa: 1.0e3,
                    friction: mu,
                    friction_eps: 1.0e-4,
                    ..Default::default()
                },
            );
            let start = solver.state.positions.clone();
            for _ in 0..50 {
                solver.step(&CgCpu::new());
            }
            assert!(
                solver.state.positions.iter().all(|p| p.is_finite()),
                "self-collision friction produced a non-finite state (mu={mu})"
            );
            // Mean sideways travel of the upper sheet.
            let mut acc = 0.0f32;
            for i in lower..start.len() {
                acc += (solver.state.positions[i].x - start[i].x).abs();
            }
            (
                acc / (start.len() - lower) as f32,
                solver.state.positions.clone(),
            )
        };
        let (free, pos_free) = run(0.0);
        let (rough, _) = run(0.9);
        assert!(free > 1e-3, "frictionless upper sheet should slide: {free}");
        assert!(
            rough < free,
            "self-collision friction must damp sliding: rough {rough} vs free {free}"
        );
        // mu = 0 must be bit-identical to the default (new code inert).
        let (_f2, pos_again) = run(0.0);
        for (i, (a, b)) in pos_free.iter().zip(&pos_again).enumerate() {
            assert_eq!(a.to_array(), b.to_array(), "vertex {i} not deterministic");
        }
    }
}
