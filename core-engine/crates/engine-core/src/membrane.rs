// SPDX-License-Identifier: GPL-2.0-or-later
//! Baraff-Witkin membrane energy (Baraff & Witkin 1998, "Large Steps in Cloth
//! Simulation") — proper in-plane stretch + shear with **anisotropy** (separate
//! warp/weft stiffness). This is what makes fabric read as denim vs silk vs
//! jersey, which edge springs cannot capture (no shear, no
//! direction dependence). It is the reference's default `Tri` model.
//!
//! Per triangle the deformation gradient `F = Ds · Dm⁻¹` (3×2) maps material
//! (warp,weft) to world. Its columns `wu, wv` give three conditions:
//!   Cu = ‖wu‖ − 1 (warp stretch), Cv = ‖wv‖ − 1 (weft), Csh = wu·wv (shear),
//! all zero at the rest shape. Energy `E = ½·A·(kw·Cu² + kf·Cv² + ks·Csh²)`.
//!
//! Derivative policy: **analytic** condition gradients (Baraff & Witkin 1998,
//! §4.2–4.3; reference `baraffwitkin.hpp:27-35`/`:111-116`, and
//! `PORTING_energy_models.md §2.1` in the reference capture — see
//! `docs/GAUGE_FINDINGS.md`). `wu`/`wv` are linear in the vertices —
//! `wu = Σⱼ cⱼ xⱼ` (coeffs from `dm_inv`) — so `∂wu/∂xⱼ = cⱼ·I₃` and the gradients
//! are exact (verified vs central-FD: `analytic_grads_match_fd`).
//!
//! Hessian: the **stretch** (warp/weft) uses the reference's EIGEN-projected form
//! `k·A·cⱼcₖ·M`, `M = n̂n̂ᵀ + max(0,1−1/‖w‖)·(I−n̂n̂ᵀ)` (eigenpairs
//! `{1, 1−1/‖w‖, 1−1/‖w‖}` clamped, `baraffwitkin.hpp:37-109`) — it adds the
//! geometric perpendicular stiffness Gauss-Newton drops under tension, and clamps
//! it away (wrinkling) under compression. **Shear** uses its closed-form
//! eigen-projected 6×6 Hessian (`shear_material`, `H6=ggᵀ+Csh·J`, eigenpairs
//! `{μ±=(A±√(A²+12Csh²))/2, ±Csh}` clamped — captures the indefinite wu–wv coupling
//! GN drops). Both verified == the exact energy Hessian where nothing clamps
//! (`stretch_eigen_matches_energy_hessian_in_tension`,
//! `shear_eigen_block_matches_energy_hessian`) and PSD after clamping. All PSD.

use glam::Vec3;

/// Precomputed per-triangle rest data: inverse material shape matrix + area.
#[derive(Clone, Copy, Debug)]
pub struct MembraneElement {
    pub tri: [u32; 3],
    /// Inverse of the 2×2 material edge matrix (row-major).
    pub dm_inv: [[f32; 2]; 2],
    pub area: f32,
}

/// Deformation-gradient columns `(wu, wv)` for a triangle.
fn deformation(p: &[Vec3; 3], dm_inv: &[[f32; 2]; 2]) -> (Vec3, Vec3) {
    let ds0 = p[1] - p[0];
    let ds1 = p[2] - p[0];
    // F = [ds0 ds1] · dm_inv ; columns of F:
    let wu = ds0 * dm_inv[0][0] + ds1 * dm_inv[1][0];
    let wv = ds0 * dm_inv[0][1] + ds1 * dm_inv[1][1];
    (wu, wv)
}

/// The three Baraff-Witkin conditions (Cu, Cv, Csh). All zero at rest.
pub fn conditions(p: &[Vec3; 3], dm_inv: &[[f32; 2]; 2]) -> [f32; 3] {
    let (wu, wv) = deformation(p, dm_inv);
    [wu.length() - 1.0, wv.length() - 1.0, wu.dot(wv)]
}

/// Analytic gradients of each condition w.r.t. the 3 vertices:
/// `[Cu, Cv, Csh][vertex]` (each entry a 3-vector `∂C/∂xⱼ`).
///
/// `wu = Σⱼ cuⱼ xⱼ`, `wv = Σⱼ cvⱼ xⱼ` (scalar coefficients from `dm_inv`), so
/// `∂wu/∂xⱼ = cuⱼ I₃`. Hence, with `n̂u = wu/‖wu‖`:
///   ∂Cu/∂xⱼ = cuⱼ · n̂u,   ∂Cv/∂xⱼ = cvⱼ · n̂v,
///   ∂Csh/∂xⱼ = cuⱼ · wv + cvⱼ · wu.
pub fn condition_grads(p: &[Vec3; 3], dm_inv: &[[f32; 2]; 2]) -> [[Vec3; 3]; 3] {
    let (wu, wv) = deformation(p, dm_inv);
    // Per-vertex scalar coefficients: wu = ds0·a0 + ds1·a1 with ds0=x1-x0,
    // ds1=x2-x0 → wu = x0·(-a0-a1) + x1·a0 + x2·a1 (likewise wv).
    let (a0, a1) = (dm_inv[0][0], dm_inv[1][0]);
    let (b0, b1) = (dm_inv[0][1], dm_inv[1][1]);
    let cu = [-(a0 + a1), a0, a1];
    let cv = [-(b0 + b1), b0, b1];
    let nu = wu.normalize_or_zero();
    let nv = wv.normalize_or_zero();
    let mut g = [[Vec3::ZERO; 3]; 3]; // [condition][vertex]
    for j in 0..3 {
        g[0][j] = nu * cu[j]; // ∂Cu/∂xⱼ
        g[1][j] = nv * cv[j]; // ∂Cv/∂xⱼ
        g[2][j] = wv * cu[j] + wu * cv[j]; // ∂Csh/∂xⱼ
    }
    g
}

/// Per-vertex scalar coefficients `(cu, cv)` such that `wu = Σⱼ cuⱼ xⱼ`,
/// `wv = Σⱼ cvⱼ xⱼ` — constant per element (depend only on the rest shape `dm_inv`).
pub fn stretch_coeffs(dm_inv: &[[f32; 2]; 2]) -> ([f32; 3], [f32; 3]) {
    let (a0, a1) = (dm_inv[0][0], dm_inv[1][0]);
    let (b0, b1) = (dm_inv[0][1], dm_inv[1][1]);
    ([-(a0 + a1), a0, a1], [-(b0 + b1), b0, b1])
}

/// Eigen-projected material Hessian (3×3, PSD) for a stretch column with gradient
/// vector `w`: `n̂n̂ᵀ + max(0, 1−1/‖w‖)·(I − n̂n̂ᵀ)`, whose eigenvalues are
/// `{1, 1−1/‖w‖, 1−1/‖w‖}` clamped at 0 (Baraff-Witkin stretch,
/// `baraffwitkin.hpp:37-109`, PORTING_energy_models.md §2.1). The per-vertex block
/// `(j,k)` of the stretch Hessian is `stiffness·area·cⱼcₖ · M`. The Gauss-Newton
/// form keeps only the `n̂n̂ᵀ` (eigenvalue-1) part; this ADDS the perpendicular
/// geometric stiffness — positive (real resistance) under tension `‖w‖>1`, clamped
/// to 0 under compression `‖w‖<1` (lets the membrane wrinkle). PSD by construction.
pub fn stretch_eigen_material(w: Vec3) -> [[f32; 3]; 3] {
    let s = w.length();
    let n = w.normalize_or_zero();
    // 1−1/s: ≥0 in tension, <0 in compression (clamp), →−∞ as s→0 (clamp). At
    // s≈0, n̂=0 too, so M=0 (a collapsed column carries no membrane stiffness).
    let perp = if s > 1.0e-6 {
        (1.0 - 1.0 / s).max(0.0)
    } else {
        0.0
    };
    let mut m = [[0.0f32; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            let nnt = n[r] * n[c];
            let iden = if r == c { 1.0 } else { 0.0 };
            m[r][c] = nnt + perp * (iden - nnt);
        }
    }
    m
}

/// `s·(u vᵀ)` accumulated into a 3×3.
fn add_outer(m: &mut [[f32; 3]; 3], u: Vec3, v: Vec3, s: f32) {
    for r in 0..3 {
        for c in 0..3 {
            m[r][c] += s * u[r] * v[c];
        }
    }
}

/// `s·q` accumulated into a 3×3.
fn add_scaled(m: &mut [[f32; 3]; 3], q: &[[f32; 3]; 3], s: f32) {
    for r in 0..3 {
        for c in 0..3 {
            m[r][c] += s * q[r][c];
        }
    }
}

/// Orthogonal projector `I − ŵŵᵀ` onto the plane perpendicular to `w` (identity
/// if `w≈0`, i.e. the whole space is perpendicular).
fn proj_perp(w: Vec3) -> [[f32; 3]; 3] {
    let n = w.normalize_or_zero();
    let mut q = [[0.0f32; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            q[r][c] = if r == c { 1.0 } else { 0.0 } - n[r] * n[c];
        }
    }
    q
}

/// Eigen-projected **shear** material: the four 3×3 blocks `[M_uu, M_uv, M_vu,
/// M_vv]` of the 6×6 shear Hessian in `(wu;wv)` space. The per-vertex block `(j,k)`
/// is `ks·area·(cuⱼcuₖ·M_uu + cuⱼcvₖ·M_uv + cvⱼcuₖ·M_vu + cvⱼcvₖ·M_vv)`.
///
/// `H6 = g gᵀ + Csh·J`, `g=(wv;wu)`, `J=[[0,I],[I,0]]`, `Csh=wu·wv` (BW shear,
/// `baraffwitkin.hpp:124-225`). Eigensystem (derived from `Ψ_sh=½Csh²`; matches
/// spec §2.1's `√(I²+12·)` structure):
///   `μ± = (A ± √(A²+12·Csh²))/2`, `A=‖wu‖²+‖wv‖²`, eigvec `(μ·wv+Csh·wu; μ·wu+Csh·wv)`;
///   `μ=+Csh` (×2): `(p;p)`, `p⊥(wu+wv)`;   `μ=−Csh` (×2): `(p;−p)`, `p⊥(wv−wu)`.
/// `μ₋ ≤ 0` always; one of `±Csh` is negative by sign. `clamp`=true drops the
/// negative eigenvalues (PSD-project, for the solver); false keeps the exact
/// indefinite Hessian (only for verification against the energy FD Hessian).
pub fn shear_material(wu: Vec3, wv: Vec3, clamp: bool) -> [[[f32; 3]; 3]; 4] {
    let f = wu.dot(wv);
    let a = wu.length_squared() + wv.length_squared();
    let disc = (a * a + 12.0 * f * f).max(0.0).sqrt();
    let mu = [0.5 * (a + disc), 0.5 * (a - disc)];
    let ev = |x: f32| if clamp { x.max(0.0) } else { x };

    let mut m = [[[0.0f32; 3]; 3]; 4]; // [uu, uv, vu, vv]
                                       // Mixing eigenvectors v = (wv·μ + wu·f ; wu·μ + wv·f), eigenvalue μ.
    for &mui in &mu {
        let top = wv * mui + wu * f;
        let bot = wu * mui + wv * f;
        let n = top.length_squared() + bot.length_squared();
        if n > 1e-20 {
            let s = ev(mui) / n;
            add_outer(&mut m[0], top, top, s);
            add_outer(&mut m[1], top, bot, s);
            add_outer(&mut m[2], bot, top, s);
            add_outer(&mut m[3], bot, bot, s);
        }
    }
    // +Csh pair: ½·ev(Csh)·[[Q+,Q+],[Q+,Q+]], Q+ = I − ŵ₊ŵ₊ᵀ, ŵ₊ = norm(wu+wv).
    let fp = 0.5 * ev(f);
    if fp != 0.0 {
        let qp = proj_perp(wu + wv);
        for blk in m.iter_mut() {
            add_scaled(blk, &qp, fp);
        }
    }
    // −Csh pair: ½·ev(−Csh)·[[Q−,−Q−],[−Q−,Q−]], Q− = I − ŵ₋ŵ₋ᵀ, ŵ₋ = norm(wv−wu).
    let fm = 0.5 * ev(-f);
    if fm != 0.0 {
        let qm = proj_perp(wv - wu);
        add_scaled(&mut m[0], &qm, fm);
        add_scaled(&mut m[1], &qm, -fm);
        add_scaled(&mut m[2], &qm, -fm);
        add_scaled(&mut m[3], &qm, fm);
    }
    m
}

/// Eigen-projected material Hessians `(Mu, Mv)` for the warp/weft stretch columns
/// at configuration `p` (see [`stretch_eigen_material`]).
pub fn stretch_materials(p: &[Vec3; 3], dm_inv: &[[f32; 2]; 2]) -> ([[f32; 3]; 3], [[f32; 3]; 3]) {
    let (wu, wv) = deformation(p, dm_inv);
    (stretch_eigen_material(wu), stretch_eigen_material(wv))
}

/// Eigen-projected shear material blocks at configuration `p` (see
/// [`shear_material`]); `clamp` PSD-projects.
pub fn shear_material_at(p: &[Vec3; 3], dm_inv: &[[f32; 2]; 2], clamp: bool) -> [[[f32; 3]; 3]; 4] {
    let (wu, wv) = deformation(p, dm_inv);
    shear_material(wu, wv, clamp)
}

/// Per-triangle membrane contribution at positions `p`: the 3-vertex gradient
/// (`grad[9]`, vertex-major xyz) and the 9×9 element Hessian (`hess[81]`, row-major
/// with block `(a,b)[r][c]` at `(3a+r)*9 + (3b+c)`), for stiffnesses (kw,kf,ks) and
/// triangle `area`. This is the single source of truth shared by the CPU triplet
/// assembly, the [`crate::assembler`]-style gather, and the GPU kernel — all three
/// must agree.
pub fn element_assembly(
    p: &[Vec3; 3],
    dm_inv: &[[f32; 2]; 2],
    area: f32,
    kw: f32,
    kf: f32,
    ks: f32,
) -> ([f32; 9], [f32; 81]) {
    let c = conditions(p, dm_inv);
    let g = condition_grads(p, dm_inv);
    let (cu, cv) = stretch_coeffs(dm_inv);
    let (mu, mv) = stretch_materials(p, dm_inv);
    let sm = shear_material_at(p, dm_inv, true);
    let stretch = [(cu, mu), (cv, mv)];
    let mut grad = [0.0f32; 9];
    let mut hess = [0.0f32; 81];
    for (ci, kc) in [kw, kf, ks].iter().enumerate() {
        let ka = *kc * area;
        if ka <= 0.0 {
            continue;
        }
        for a in 0..3 {
            for cc in 0..3 {
                grad[3 * a + cc] += ka * c[ci] * g[ci][a][cc];
            }
            for b in 0..3 {
                for r in 0..3 {
                    for cc in 0..3 {
                        let v = if ci < 2 {
                            let (coeff, m) = &stretch[ci];
                            ka * coeff[a] * coeff[b] * m[r][cc]
                        } else {
                            ka * (cu[a] * cu[b] * sm[0][r][cc]
                                + cu[a] * cv[b] * sm[1][r][cc]
                                + cv[a] * cu[b] * sm[2][r][cc]
                                + cv[a] * cv[b] * sm[3][r][cc])
                        };
                        hess[(3 * a + r) * 9 + (3 * b + cc)] += v;
                    }
                }
            }
        }
    }
    (grad, hess)
}

/// Build membrane elements from a triangle mesh; rest data from `positions`.
pub fn build_membrane(tris: &[[u32; 3]], positions: &[Vec3]) -> Vec<MembraneElement> {
    let mut out = Vec::new();
    for t in tris {
        let (x0, x1, x2) = (
            positions[t[0] as usize],
            positions[t[1] as usize],
            positions[t[2] as usize],
        );
        let e1 = x1 - x0;
        let e2 = x2 - x0;
        let a = e1.length();
        let n = e1.cross(e2);
        let area = 0.5 * n.length();
        if a < 1.0e-9 || area < 1.0e-12 {
            continue; // degenerate triangle
        }
        // Material 2D frame of the rest triangle.
        let t1 = e1 / a;
        let t2 = (n.cross(e1)).normalize_or_zero();
        // Dm = [[a, e2·t1],[0, e2·t2]] (upper-triangular). Invert.
        let (b, d) = (e2.dot(t1), e2.dot(t2));
        let det = a * d;
        if det.abs() < 1.0e-12 {
            continue;
        }
        // inv([[a,b],[0,d]]) = [[1/a, -b/(a d)],[0, 1/d]]
        let dm_inv = [[1.0 / a, -b / (a * d)], [0.0, 1.0 / d]];
        out.push(MembraneElement {
            tri: *t,
            dm_inv,
            area,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tri(p0: Vec3, p1: Vec3, p2: Vec3) -> [Vec3; 3] {
        [p0, p1, p2]
    }

    /// Central finite-difference condition gradients — the oracle the analytic
    /// form is checked against (this is what `condition_grads` used to compute).
    fn condition_grads_fd(p: &[Vec3; 3], dm_inv: &[[f32; 2]; 2]) -> [[Vec3; 3]; 3] {
        let scale = (p[0].length() + p[1].length() + p[2].length()).max(1.0);
        let h = 1.0e-4 * scale;
        let inv = 1.0 / (2.0 * h);
        let mut g = [[Vec3::ZERO; 3]; 3];
        for vi in 0..3 {
            for c in 0..3 {
                let (mut pp, mut pm) = (*p, *p);
                pp[vi][c] += h;
                pm[vi][c] -= h;
                let cp = conditions(&pp, dm_inv);
                let cm = conditions(&pm, dm_inv);
                for k in 0..3 {
                    g[k][vi][c] = (cp[k] - cm[k]) * inv;
                }
            }
        }
        g
    }

    #[test]
    fn analytic_grads_match_fd() {
        // A rest triangle + a nontrivially deformed configuration (stretch + shear
        // + out-of-plane), so all three conditions are excited and F is generic.
        let rest = tri(
            Vec3::new(0.1, -0.2, 0.0),
            Vec3::new(1.3, 0.1, 0.0),
            Vec3::new(0.2, 0.9, 0.0),
        );
        let e = build_membrane(&[[0, 1, 2]], &rest);
        let def = tri(
            Vec3::new(0.15, -0.18, 0.05),
            Vec3::new(1.55, 0.22, -0.1),
            Vec3::new(0.05, 1.15, 0.2),
        );
        let ana = condition_grads(&def, &e[0].dm_inv);
        let fd = condition_grads_fd(&def, &e[0].dm_inv);
        let mut max = 0.0f32;
        for k in 0..3 {
            for j in 0..3 {
                max = max.max((ana[k][j] - fd[k][j]).length());
            }
        }
        assert!(max < 1e-3, "analytic vs FD condition-gradient drift {max}");
    }

    #[test]
    #[allow(clippy::needless_range_loop)] // r,c index m[r][c] vs the transpose m[c][r]
    fn stretch_eigen_material_eigenvalues() {
        // Tension ‖w‖=1.5 → eigenvalues {1, 1-1/1.5, 1-1/1.5}; along +x n̂=(1,0,0).
        let m = stretch_eigen_material(Vec3::new(1.5, 0.0, 0.0));
        assert!((m[0][0] - 1.0).abs() < 1e-5, "n̂ eigenvalue = 1");
        let perp = 1.0 - 1.0 / 1.5;
        assert!((m[1][1] - perp).abs() < 1e-5, "perp eigenvalue in tension");
        assert!((m[2][2] - perp).abs() < 1e-5);
        // Symmetric + PSD diagonal.
        for r in 0..3 {
            assert!(m[r][r] >= -1e-6);
            for c in 0..3 {
                assert!((m[r][c] - m[c][r]).abs() < 1e-6, "symmetric");
            }
        }
        // Compression ‖w‖=0.6 < 1 → perpendicular eigenvalue clamped to 0 → M=n̂n̂ᵀ
        // (exactly the Gauss-Newton material).
        let mc = stretch_eigen_material(Vec3::new(0.6, 0.0, 0.0));
        assert!((mc[0][0] - 1.0).abs() < 1e-6);
        assert!(
            mc[1][1].abs() < 1e-6 && mc[2][2].abs() < 1e-6,
            "compression clamps perp"
        );
    }

    #[test]
    #[allow(clippy::needless_range_loop)] // indices feed expect(blk,r,c)
    fn shear_material_unclamped_reconstructs_h6() {
        // With clamp=false the spectral sum must reproduce H6 = g gᵀ + Csh·J:
        //   uu=wv wvᵀ, uv=wv wuᵀ+Csh·I, vu=wu wvᵀ+Csh·I, vv=wu wuᵀ.
        let wu = Vec3::new(1.1, 0.3, -0.2);
        let wv = Vec3::new(0.2, 0.9, 0.4);
        let f = wu.dot(wv);
        let m = shear_material(wu, wv, false);
        let expect = |blk: usize, r: usize, c: usize| -> f32 {
            let ident = if r == c { f } else { 0.0 };
            match blk {
                0 => wv[r] * wv[c],
                1 => wv[r] * wu[c] + ident,
                2 => wu[r] * wv[c] + ident,
                _ => wu[r] * wu[c],
            }
        };
        for blk in 0..4 {
            for r in 0..3 {
                for c in 0..3 {
                    assert!(
                        (m[blk][r][c] - expect(blk, r, c)).abs() < 1e-4,
                        "block {blk}[{r}][{c}] = {} expected {}",
                        m[blk][r][c],
                        expect(blk, r, c)
                    );
                }
            }
        }
    }

    #[test]
    fn shear_eigen_block_matches_energy_hessian() {
        // The per-vertex shear block (clamp=false) must equal the exact FD Hessian
        // of Ψ_sh = ½·Csh². (clamp=false → the full indefinite Hessian.)
        let rest = tri(
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.2, 0.9, 0.0),
        );
        let e = build_membrane(&[[0, 1, 2]], &rest);
        let dm = e[0].dm_inv;
        let p = tri(
            Vec3::new(0.0, 0.05, 0.0),
            Vec3::new(1.2, 0.1, 0.1),
            Vec3::new(0.35, 1.0, -0.05),
        );
        assert!(conditions(&p, &dm)[2].abs() > 0.05, "need nonzero shear");
        let (cu, cv) = stretch_coeffs(&dm);
        let (wu, wv) = deformation(&p, &dm);
        let m = shear_material(wu, wv, false);
        // block(j,k) = cuⱼcuₖ M_uu + cuⱼcvₖ M_uv + cvⱼcuₖ M_vu + cvⱼcvₖ M_vv.
        let block = |j: usize, k: usize, r: usize, c: usize| -> f32 {
            cu[j] * cu[k] * m[0][r][c]
                + cu[j] * cv[k] * m[1][r][c]
                + cv[j] * cu[k] * m[2][r][c]
                + cv[j] * cv[k] * m[3][r][c]
        };
        let energy = |q: &[Vec3; 3]| {
            let sh = conditions(q, &dm)[2];
            0.5 * sh * sh
        };
        let h = 1.0e-3;
        for j in 0..3 {
            for k in 0..3 {
                for r in 0..3 {
                    for c in 0..3 {
                        let (mut pp, mut pm, mut mp, mut mm) = (p, p, p, p);
                        pp[j][r] += h;
                        pp[k][c] += h;
                        pm[j][r] += h;
                        pm[k][c] -= h;
                        mp[j][r] -= h;
                        mp[k][c] += h;
                        mm[j][r] -= h;
                        mm[k][c] -= h;
                        let fd =
                            (energy(&pp) - energy(&pm) - energy(&mp) + energy(&mm)) / (4.0 * h * h);
                        assert!(
                            (fd - block(j, k, r, c)).abs() < 3.0e-2,
                            "H({j},{k})[{r}][{c}] fd={fd} eigen={}",
                            block(j, k, r, c)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn shear_material_clamped_is_psd() {
        // Build the 6×6 from the clamped blocks and check xᵀMx ≥ 0 for many x.
        let cases = [
            (Vec3::new(1.2, 0.1, 0.0), Vec3::new(0.3, 1.1, 0.0)), // Csh>0
            (Vec3::new(1.0, -0.4, 0.2), Vec3::new(-0.5, 0.8, 0.1)), // Csh<0
        ];
        for (wu, wv) in cases {
            let m = shear_material(wu, wv, true);
            // 6×6 assembled from blocks [[uu,uv],[vu,vv]].
            let at = |i: usize, j: usize| -> f32 {
                let (blk, r, c) = (2 * (i / 3) + (j / 3), i % 3, j % 3);
                m[blk][r][c]
            };
            for seed in 0..40 {
                let mut x = [0.0f32; 6];
                for (t, xi) in x.iter_mut().enumerate() {
                    *xi = (((seed * 7 + t * 13) % 11) as f32 - 5.0) * 0.3;
                }
                let mut q = 0.0f32;
                for i in 0..6 {
                    for j in 0..6 {
                        q += x[i] * at(i, j) * x[j];
                    }
                }
                assert!(q > -1e-4, "not PSD: xᵀMx={q} for wu={wu} wv={wv}");
            }
        }
    }

    #[test]
    fn stretch_eigen_matches_energy_hessian_in_tension() {
        // Under tension no eigenvalue is clamped, so the eigen stretch Hessian block
        // must equal the exact FD Hessian of the warp energy Ψ_u = ½·Cu² (μ=1).
        let rest = tri(
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.2, 0.9, 0.0),
        );
        let e = build_membrane(&[[0, 1, 2]], &rest);
        let dm = e[0].dm_inv;
        let p = tri(
            Vec3::ZERO,
            Vec3::new(1.4, 0.05, 0.1),
            Vec3::new(0.25, 1.0, -0.05),
        );
        assert!(conditions(&p, &dm)[0] > 0.05, "need warp tension");
        let (cu, _) = stretch_coeffs(&dm);
        let (mu, _) = stretch_materials(&p, &dm);
        let energy = |q: &[Vec3; 3]| {
            let c = conditions(q, &dm)[0];
            0.5 * c * c
        };
        let h = 1.0e-3;
        for j in 0..3 {
            for k in 0..3 {
                for r in 0..3 {
                    for c in 0..3 {
                        let (mut pp, mut pm, mut mp, mut mm) = (p, p, p, p);
                        pp[j][r] += h;
                        pp[k][c] += h;
                        pm[j][r] += h;
                        pm[k][c] -= h;
                        mp[j][r] -= h;
                        mp[k][c] += h;
                        mm[j][r] -= h;
                        mm[k][c] -= h;
                        let fd =
                            (energy(&pp) - energy(&pm) - energy(&mp) + energy(&mm)) / (4.0 * h * h);
                        let eigen = cu[j] * cu[k] * mu[r][c];
                        assert!(
                            (fd - eigen).abs() < 3.0e-2,
                            "H({j},{k})[{r}][{c}] fd={fd} eigen={eigen}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn analytic_grad_matches_energy_derivative() {
        // ∇(½C²)·δ = C·(∇C·δ) — check dE/dx along a random direction via FD of the
        // energy vs the analytic condition gradient, per condition.
        let rest = tri(
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.2, 1.1, 0.0),
        );
        let e = build_membrane(&[[0, 1, 2]], &rest);
        let dm = e[0].dm_inv;
        let p = tri(
            Vec3::new(0.0, 0.05, 0.0),
            Vec3::new(1.2, -0.1, 0.1),
            Vec3::new(0.25, 1.2, -0.05),
        );
        let g = condition_grads(&p, &dm);
        // FD of each condition along +x of vertex 1.
        let h = 1e-4;
        let (mut pp, mut pm) = (p, p);
        pp[1].x += h;
        pm[1].x -= h;
        let (cp, cm) = (conditions(&pp, &dm), conditions(&pm, &dm));
        for k in 0..3 {
            let fd = (cp[k] - cm[k]) / (2.0 * h);
            assert!(
                (g[k][1].x - fd).abs() < 1e-2,
                "cond {k} ∂/∂x1: analytic {} vs fd {fd}",
                g[k][1].x
            );
        }
    }

    #[test]
    fn rest_shape_has_zero_conditions() {
        let rest = tri(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.3, 0.8, 0.0),
        );
        let e = build_membrane(&[[0, 1, 2]], &rest);
        let c = conditions(&rest, &e[0].dm_inv);
        for v in c {
            assert!(v.abs() < 1e-4, "rest condition nonzero: {c:?}");
        }
    }

    #[test]
    fn uniaxial_stretch_excites_only_warp() {
        // Rest is an axis-aligned right triangle; warp = +x, weft = +y.
        let rest = tri(
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let e = build_membrane(&[[0, 1, 2]], &rest);
        // Stretch along x by 20% only.
        let def = tri(
            Vec3::ZERO,
            Vec3::new(1.2, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let c = conditions(&def, &e[0].dm_inv);
        assert!(
            (c[0] - 0.2).abs() < 1e-3,
            "warp stretch Cu≈0.2, got {}",
            c[0]
        );
        assert!(
            c[1].abs() < 1e-3,
            "weft should be unstretched, got {}",
            c[1]
        );
        assert!(c[2].abs() < 1e-3, "no shear, got {}", c[2]);
    }

    #[test]
    fn shear_excites_shear_condition() {
        let rest = tri(
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let e = build_membrane(&[[0, 1, 2]], &rest);
        // Shear: move the +y vertex in +x.
        let def = tri(
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.3, 1.0, 0.0),
        );
        let c = conditions(&def, &e[0].dm_inv);
        assert!(
            c[2].abs() > 0.1,
            "shear condition should fire, got {}",
            c[2]
        );
    }
}
