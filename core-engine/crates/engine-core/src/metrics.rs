// SPDX-License-Identifier: GPL-2.0-or-later
//! Per-step metric channels + the solver-invariant checker.
//!
//! Mirrors the reference gauge's log channels and `verify_invariants.py`
//! (see `docs/GAUGE_FINDINGS.md §1`). The invariants are the **primary**
//! acceptance gate — per-vertex position matching fails the reference itself
//! (5% self-divergence), so correctness is judged by these bounded invariants,
//! not tight numeric diffs.

/// One simulation step's diagnostic channels (subset of the gauge's 37).
#[derive(Clone, Copy, Debug)]
pub struct StepMetrics {
    pub time: f64,
    /// Max final relative PCG residual over the step's Newton iterations
    /// (gauge `advance.reresid`).
    pub max_reresid: f64,
    /// Max PCG iterations over the step's Newton iterations (`advance.iter`).
    pub max_cg_iters: u32,
    /// Newton iterations actually taken this step. Constant at `newton_iters` unless
    /// `target_toi` is enabled, in which case the loop terminates early — the reference
    /// logs the same quantity to show when the solver needed extra iterations.
    pub newton_iters: u32,
    /// Max stretch ratio (current/rest length) — `advance.max_sigma`.
    pub max_sigma: f32,
    /// Max non-pinned vertex speed (m/s) — `advance.max_u`; explosion detector.
    pub max_velocity: f32,
    /// Active contact/constraint pairs this step (`advance.num_contact`).
    pub num_contacts: u32,
    /// No penetration / CCD failure / non-finite this step.
    pub penetration_free: bool,
    /// Every position finite this step.
    pub all_finite: bool,
    /// Device allocations this step (0 = steady-state; CPU tier is always 0).
    pub device_alloc: u32,
    pub device_free: u32,
}

/// Accumulated run metrics + the config the invariant checker compares against.
#[derive(Clone, Debug)]
pub struct Metrics {
    pub steps: Vec<StepMetrics>,
    pub cg_tol: f32,
    pub cg_max_iter: u32,
    /// Max tensile strain cap; ≤0 disables the strain-limit invariant (matches
    /// the gauge's auto-pass when strain limiting is off).
    pub strain_limit: f32,
}

impl Metrics {
    pub fn new(cg_tol: f32, cg_max_iter: u32, strain_limit: f32) -> Self {
        Self {
            steps: Vec::new(),
            cg_tol,
            cg_max_iter,
            strain_limit,
        }
    }
}

/// One invariant check result.
#[derive(Clone, Debug)]
pub struct Invariant {
    pub name: &'static str,
    pub ok: bool,
    pub detail: String,
}

/// Full invariant report; `all_ok` is the acceptance verdict.
#[derive(Clone, Debug)]
pub struct InvariantReport {
    pub checks: Vec<Invariant>,
    pub all_ok: bool,
}

impl InvariantReport {
    pub fn all_ok(&self) -> bool {
        self.all_ok
    }
}

/// Run the solver invariants over accumulated [`Metrics`]. Thresholds mirror the
/// reference's own `verify_invariants.py` harness; they are tabulated, with the
/// reasoning for each, in `docs/GAUGE_FINDINGS.md` §1.
pub fn check_invariants(m: &Metrics) -> InvariantReport {
    let mut checks = Vec::new();
    let steps = &m.steps;

    // 1. penetration_free — any penetration/CCD-fail marker fails.
    let pen = steps.iter().all(|s| s.penetration_free);
    let bad = steps.iter().filter(|s| !s.penetration_free).count();
    checks.push(Invariant {
        name: "penetration_free",
        ok: pen,
        detail: format!("{bad} step(s) with penetration/CCD failure"),
    });

    // 2. all_frames_finite.
    let finite = steps.iter().all(|s| s.all_finite);
    checks.push(Invariant {
        name: "all_frames_finite",
        ok: finite,
        detail: format!(
            "{} non-finite step(s)",
            steps.iter().filter(|s| !s.all_finite).count()
        ),
    });

    // 3. strain_limit_respected — peak ≤ (1+limit)·1.02; auto-pass if disabled.
    if m.strain_limit <= 0.0 {
        checks.push(Invariant {
            name: "strain_limit_respected",
            ok: true,
            detail: "strain limiting disabled (auto-pass)".into(),
        });
    } else {
        let peak = steps.iter().map(|s| s.max_sigma).fold(0.0f32, f32::max);
        let ceiling = (1.0 + m.strain_limit) * 1.02;
        checks.push(Invariant {
            name: "strain_limit_respected",
            ok: peak <= ceiling,
            detail: format!("peak sigma {peak:.4} vs ceiling {ceiling:.4}"),
        });
    }

    // 4. cg_converged — worst reresid ≤ cg_tol·1.5.
    let worst = steps.iter().map(|s| s.max_reresid).fold(0.0f64, f64::max);
    let tol = m.cg_tol as f64 * 1.5;
    checks.push(Invariant {
        name: "cg_converged",
        ok: worst <= tol,
        detail: format!("worst reresid {worst:.3e} vs {tol:.3e}"),
    });

    // 5. cg_iter_under_cap — max iters < cg_max_iter.
    let max_it = steps.iter().map(|s| s.max_cg_iters).max().unwrap_or(0);
    checks.push(Invariant {
        name: "cg_iter_under_cap",
        ok: max_it < m.cg_max_iter,
        detail: format!("max cg iters {max_it} vs cap {}", m.cg_max_iter),
    });

    // 6. zero_steady_state_alloc — second half sums to 0.
    let half = steps.len() / 2;
    let (ta, tf): (u64, u64) = steps[half..].iter().fold((0, 0), |(a, f), s| {
        (a + s.device_alloc as u64, f + s.device_free as u64)
    });
    // These are DEVICE-pool counters only. The CPU tier hardwires them to 0
    // (`implicit.rs`: "CPU tier: no device pool"), so on CPU this check cannot fail
    // and says nothing about host allocation — which is NOT zero there (the triplet
    // list churns tens of MB per Newton iteration; see `MGUSO_MEM_TRACE=1`). Say so in
    // the detail rather than reporting a bare pass, following `strain_limit_respected`
    // above, which labels its own auto-pass.
    // ponytail: a string, not plumbing. If host churn needs gating, thread an
    // `alloc_stats` host peak into `StepMetrics` and check that instead.
    let vacuous = steps
        .iter()
        .all(|s| s.device_alloc == 0 && s.device_free == 0);
    checks.push(Invariant {
        name: "zero_steady_state_alloc",
        ok: ta == 0 && tf == 0,
        detail: if vacuous {
            "no device-pool counters recorded (CPU tier: auto-pass, host alloc unchecked)"
                .to_string()
        } else {
            format!("steady-state device alloc {ta} free {tf}")
        },
    });

    let all_ok = checks.iter().all(|c| c.ok);
    InvariantReport { checks, all_ok }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good_step(time: f64) -> StepMetrics {
        StepMetrics {
            time,
            max_reresid: 1e-4,
            max_cg_iters: 50,
            newton_iters: 4,
            max_sigma: 1.001,
            max_velocity: 0.5,
            num_contacts: 3,
            penetration_free: true,
            all_finite: true,
            device_alloc: 0,
            device_free: 0,
        }
    }

    #[test]
    fn good_run_passes_all_invariants() {
        let mut m = Metrics::new(1e-3, 10000, 0.0);
        for k in 0..10 {
            m.steps.push(good_step(k as f64));
        }
        let r = check_invariants(&m);
        assert!(r.all_ok(), "good run should pass: {:?}", r.checks);
    }

    #[test]
    fn diverged_run_fails() {
        let mut m = Metrics::new(1e-3, 100, 0.0);
        for k in 0..10 {
            let mut s = good_step(k as f64);
            if k >= 5 {
                s.all_finite = false; // NaN blowup halfway
                s.penetration_free = false;
                s.max_reresid = 0.9; // CG stalled
                s.max_cg_iters = 100; // hit cap
            }
            m.steps.push(s);
        }
        let r = check_invariants(&m);
        assert!(!r.all_ok(), "diverged run must fail");
        // The specific failing invariants are flagged.
        let by = |n: &str| r.checks.iter().find(|c| c.name == n).unwrap().ok;
        assert!(!by("all_frames_finite"));
        assert!(!by("penetration_free"));
        assert!(!by("cg_converged"));
        assert!(!by("cg_iter_under_cap"));
    }

    #[test]
    fn strain_limit_gate() {
        // With a 5% limit, peak sigma 1.10 exceeds ceiling 1.05·1.02=1.071 → fail.
        let mut m = Metrics::new(1e-3, 10000, 0.05);
        let mut s = good_step(0.0);
        s.max_sigma = 1.10;
        m.steps.push(s);
        assert!(!check_invariants(&m).all_ok());
        // Within the 2% slack passes.
        m.steps[0].max_sigma = 1.06;
        assert!(check_invariants(&m).all_ok());
    }
}
