// SPDX-License-Identifier: GPL-2.0-or-later
//! PSD projection for dense symmetric 12×12 blocks (the tet element Hessian), by
//! cyclic Jacobi eigendecomposition with negative eigenvalues clamped to zero.
//!
//! This is the CPU, N=12 counterpart of the 3×3 GPU device function
//! (`engine-gpu::sym_eig`): the reference likewise filters element Hessians to
//! positive-semidefinite via eigenanalysis (docs/KERNELS.md `energy/eigenanalysis`),
//! which is what keeps the assembled Newton system SPD for CG.

const N: usize = 12;

/// Symmetric eigendecomposition of a row-major `N×N` matrix by cyclic Jacobi.
/// Returns `(eigenvalues, eigenvectors_as_columns)`.
fn jacobi_eig(input: &[f32; N * N], sweeps: usize) -> ([f32; N], [f32; N * N]) {
    let mut a = *input;
    let mut v = [0.0f32; N * N];
    for i in 0..N {
        v[i * N + i] = 1.0;
    }
    for _ in 0..sweeps {
        for p in 0..N {
            for q in (p + 1)..N {
                let apq = a[p * N + q];
                if apq.abs() < 1e-12 {
                    continue;
                }
                let (app, aqq) = (a[p * N + p], a[q * N + q]);
                let phi = 0.5 * (2.0 * apq).atan2(aqq - app);
                let (c, s) = (phi.cos(), phi.sin());
                // Columns p,q.
                for k in 0..N {
                    let (akp, akq) = (a[k * N + p], a[k * N + q]);
                    a[k * N + p] = c * akp - s * akq;
                    a[k * N + q] = s * akp + c * akq;
                }
                // Rows p,q.
                for k in 0..N {
                    let (apk, aqk) = (a[p * N + k], a[q * N + k]);
                    a[p * N + k] = c * apk - s * aqk;
                    a[q * N + k] = s * apk + c * aqk;
                }
                // Accumulate eigenvectors: V ← V·R.
                for k in 0..N {
                    let (vkp, vkq) = (v[k * N + p], v[k * N + q]);
                    v[k * N + p] = c * vkp - s * vkq;
                    v[k * N + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut lam = [0.0f32; N];
    for i in 0..N {
        lam[i] = a[i * N + i];
    }
    (lam, v)
}

/// Nearest PSD matrix (Frobenius) to a symmetric 12×12: clamp negative eigenvalues
/// to zero and reconstruct `V·diag(max(λ,0))·Vᵀ`.
pub fn project_psd_12(input: &[f32; N * N]) -> [f32; N * N] {
    // Symmetrize first — FD Hessians are only symmetric to rounding.
    let mut sym = [0.0f32; N * N];
    for r in 0..N {
        for c in 0..N {
            sym[r * N + c] = 0.5 * (input[r * N + c] + input[c * N + r]);
        }
    }
    let (lam, v) = jacobi_eig(&sym, 16);
    if lam.iter().all(|&l| l >= 0.0) {
        return sym; // already PSD — skip the reconstruction rounding
    }
    let mut out = [0.0f32; N * N];
    for r in 0..N {
        for c in 0..N {
            let mut acc = 0.0f32;
            for k in 0..N {
                let l = lam[k].max(0.0);
                if l > 0.0 {
                    acc += v[r * N + k] * l * v[c * N + k];
                }
            }
            out[r * N + c] = acc;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad(m: &[f32; N * N], x: &[f32; N]) -> f32 {
        let mut q = 0.0;
        for r in 0..N {
            for c in 0..N {
                q += x[r] * m[r * N + c] * x[c];
            }
        }
        q
    }

    fn probe(seed: u64) -> [f32; N] {
        std::array::from_fn(|k| {
            ((seed
                .wrapping_mul(7919)
                .wrapping_add((k as u64).wrapping_mul(104729))
                % 200) as f32)
                / 100.0
                - 1.0
        })
    }

    /// Deterministic symmetric matrix with a controllable indefiniteness.
    fn sym_mat(seed: u64, indefinite: bool) -> [f32; N * N] {
        let mut m = [0.0f32; N * N];
        for r in 0..N {
            for c in r..N {
                let k = (r * N + c) as u64;
                let x = (seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(k.wrapping_mul(1442695040888963407)))
                    >> 33;
                let mut v = (x as f32 / u32::MAX as f32) * 2.0 - 1.0;
                if r == c && !indefinite {
                    v = v.abs() + 12.0; // diagonally dominant ⇒ PD
                }
                m[r * N + c] = v;
                m[c * N + r] = v;
            }
        }
        m
    }

    #[test]
    fn eig_reconstructs_symmetric_matrix() {
        for seed in 0..12u64 {
            let m = sym_mat(seed, true);
            let (lam, v) = jacobi_eig(&m, 16);
            // V·diag(λ)·Vᵀ == m
            for r in 0..N {
                for c in 0..N {
                    let mut acc = 0.0f32;
                    for k in 0..N {
                        acc += v[r * N + k] * lam[k] * v[c * N + k];
                    }
                    assert!(
                        (acc - m[r * N + c]).abs() < 2e-2,
                        "reconstruction failed seed={seed} at ({r},{c}): {acc} vs {}",
                        m[r * N + c]
                    );
                }
            }
            // Eigenvectors orthonormal.
            for i in 0..N {
                let norm: f32 = (0..N).map(|k| v[k * N + i] * v[k * N + i]).sum();
                assert!((norm - 1.0).abs() < 1e-2, "eigenvector {i} not unit");
            }
        }
    }

    #[test]
    fn projection_is_psd_and_idempotent() {
        for seed in 0..20u64 {
            let m = sym_mat(seed, true);
            let p = project_psd_12(&m);
            for t in 0..10u64 {
                let q = quad(&p, &probe(t));
                assert!(q >= -1e-2, "projection not PSD seed={seed}: {q}");
            }
            // Idempotent.
            let p2 = project_psd_12(&p);
            for i in 0..N * N {
                assert!(
                    (p2[i] - p[i]).abs() < 2e-2,
                    "not idempotent seed={seed} at {i}"
                );
            }
        }
    }

    #[test]
    fn already_psd_matrix_is_preserved() {
        for seed in 0..10u64 {
            let m = sym_mat(seed, false); // diagonally dominant ⇒ PD
            let p = project_psd_12(&m);
            for i in 0..N * N {
                assert!(
                    (p[i] - m[i]).abs() < 1e-3,
                    "PD matrix altered seed={seed} at {i}: {} vs {}",
                    p[i],
                    m[i]
                );
            }
        }
    }
}
