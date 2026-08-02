// SPDX-License-Identifier: GPL-2.0-or-later
//! A 2D geometric constraint solver: safe Rust over FreeCAD's `planegcs`
//! (LGPL-2.1-or-later, vendored under `third_party/planegcs/`).
//!
//! This is the foundation of nguo-3d's parametric pattern layer: a pattern is
//! drafted as geometry plus constraints, so changing a body measurement
//! re-solves the geometry instead of requiring it to be redrawn.
//!
//! # Boundary
//!
//! Deliberately standalone. It takes geometry as plain `f64` coordinates and
//! hands back solved `f64` coordinates. It knows nothing about pattern pieces,
//! vector networks, seams or documents, so `nguo-pattern` owns the mapping
//! between its entity ids and the [`PointId`]/[`LineId`]/[`ConstraintId`]
//! handles here.
//!
//! # Shape of a session
//!
//! 1. Add parameters and geometry.
//! 2. Add constraints. Each returns a [`ConstraintId`].
//! 3. [`System::diagnose`] for DOF feedback ([`Diagnosis`], [`DofState`]).
//! 4. [`System::solve`], then read coordinates back.
//! 5. For a parameter change, [`System::set_param`] then [`System::solve`]
//!    again — no rebuild, no re-diagnosis.
//!
//! See `ui/docs/CONSTRAINTS.md` for the vendoring story and worked examples.
#![deny(unsafe_op_in_unsafe_fn)]

mod ffi;

use std::ffi::c_int;
use std::ptr::NonNull;

// ---------------------------------------------------------------------------
// Handles
// ---------------------------------------------------------------------------

macro_rules! handle {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
        pub struct $name(u32);

        impl $name {
            /// The raw arena index. Only meaningful to the system that issued it.
            #[must_use]
            pub const fn index(self) -> u32 {
                self.0
            }
        }
    };
}

handle!(
    /// A scalar the solver reads. Free parameters are unknowns it may move;
    /// driving parameters are the dimensions you set (§9.4).
    ParamId
);
handle!(
    /// A 2D point, backed by two [`ParamId`]s.
    PointId
);
handle!(
    /// A straight segment between two points.
    LineId
);
handle!(
    /// A full circle: centre point plus a radius parameter.
    CircleId
);
handle!(
    /// A circular arc.
    ArcId
);

/// Anything a point can lie on or a curve can be tangent to.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct CurveId(u32);

impl From<LineId> for CurveId {
    fn from(v: LineId) -> Self {
        Self(v.0)
    }
}
impl From<CircleId> for CurveId {
    fn from(v: CircleId) -> Self {
        Self(v.0)
    }
}
impl From<ArcId> for CurveId {
    fn from(v: ArcId) -> Self {
        Self(v.0)
    }
}

/// One user-level constraint. Maps to a planegcs tag, which may cover several
/// solver constraints (see [`System::collinear`]), and is what DOF diagnosis
/// reports back in [`Diagnosis`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ConstraintId(i32);

impl ConstraintId {
    /// The planegcs tag. Positive for every constraint this crate creates.
    #[must_use]
    pub const fn tag(self) -> i32 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// Errors and status
// ---------------------------------------------------------------------------

/// Why the solver stopped without a solution.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SolveStatus {
    /// Reached a minimum of the error function without zeroing it: the
    /// constraints are mutually unsatisfiable, or the solver found a local
    /// minimum it cannot escape.
    Converged,
    /// No minimum found within the iteration budget.
    Failed,
    /// A solution was found but the resulting geometry is degenerate.
    InvalidSolution,
    /// The system was never initialised (no unknowns, or no constraints).
    NotInitialised,
    /// The C++ solver raised an exception, which the shim absorbed.
    Exception,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, thiserror::Error)]
pub enum Error {
    /// A handle was never issued by this system.
    #[error("handle is not valid for this system")]
    BadHandle,
    /// The constraint is not defined for that kind of curve, e.g. tangency
    /// between two lines.
    #[error("constraint does not apply to that curve kind")]
    BadKind,
    /// The C++ layer raised an exception; the shim converted it rather than
    /// letting it unwind across the FFI boundary.
    #[error("the constraint solver raised an exception")]
    Exception,
    /// The solver ran and did not produce a solution. Parameters are left at
    /// the values they held before the attempt: never NaN, never half-solved.
    #[error("no solution: {0:?}")]
    NoSolution(SolveStatus),
    /// [`System::diagnose`] has not run since the last structural change.
    #[error("no diagnosis available; call diagnose() first")]
    NoDiagnosis,
}

pub type Result<T> = std::result::Result<T, Error>;

/// Which numerical method to use. planegcs' own default is [`Algorithm::DogLeg`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Algorithm {
    /// Broyden-Fletcher-Goldfarb-Shanno.
    Bfgs,
    /// Levenberg-Marquardt.
    LevenbergMarquardt,
    /// Powell's dog leg. planegcs' default and the most robust of the three.
    #[default]
    DogLeg,
}

impl Algorithm {
    const fn raw(self) -> c_int {
        match self {
            Self::Bfgs => ffi::GCS_ALG_BFGS,
            Self::LevenbergMarquardt => ffi::GCS_ALG_LEVENBERG_MARQUARDT,
            Self::DogLeg => ffi::GCS_ALG_DOGLEG,
        }
    }
}

// ---------------------------------------------------------------------------
// DOF analysis (§9.3)
// ---------------------------------------------------------------------------

/// How constrained a sketch is. This is the feedback a constraint UI cannot
/// work without: without it the user has no way to tell why the sketch will
/// not move.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DofState {
    /// Degrees of freedom remain; the sketch can still be dragged.
    UnderConstrained {
        /// How many.
        dof: u32,
    },
    /// Exactly determined. Every point has one solution.
    WellConstrained,
    /// Constraints contradict each other, or duplicate each other. Look at
    /// [`Diagnosis::conflicting`] and [`Diagnosis::redundant`].
    OverConstrained,
}

/// The result of planegcs' QR-based DOF analysis.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Diagnosis {
    /// Remaining degrees of freedom.
    ///
    /// Signed, because planegcs reports an over-determined system as
    /// `unknowns - independent constraints`, which goes negative. Zero means
    /// exactly determined, positive means draggable, negative means
    /// over-constrained by that many equations.
    pub dof: i32,
    /// Constraints that cannot all be satisfied at once. The sketch is broken
    /// until one of them is removed.
    pub conflicting: Vec<ConstraintId>,
    /// Constraints that are implied by the others. Harmless but worth
    /// flagging, since they hide the sketch's real structure.
    pub redundant: Vec<ConstraintId>,
    /// Constraints that are redundant only in part.
    pub partially_redundant: Vec<ConstraintId>,
}

impl Diagnosis {
    #[must_use]
    pub fn state(&self) -> DofState {
        if self.dof < 0 || !self.conflicting.is_empty() || !self.redundant.is_empty() {
            DofState::OverConstrained
        } else if self.dof > 0 {
            DofState::UnderConstrained {
                dof: self.dof.unsigned_abs(),
            }
        } else {
            DofState::WellConstrained
        }
    }

    /// True if the sketch has a unique solution and no wasted constraints.
    #[must_use]
    pub fn is_well_constrained(&self) -> bool {
        self.state() == DofState::WellConstrained
    }
}

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

/// A 2D constraint system.
///
/// Geometry goes in as plain `f64` coordinates and comes back out the same
/// way. The system deliberately knows nothing about pattern pieces, seams or
/// documents; wiring it to `nguo-pattern` is the caller's job.
///
/// ```
/// use nguo_constraints::System;
///
/// let mut s = System::new();
/// let a = s.add_fixed_point(0.0, 0.0);
/// let b = s.add_point(3.0, 4.0);
/// let l = s.add_line(a, b)?;
/// let len = s.add_driving_param(10.0);
///
/// s.horizontal(l)?;
/// s.length(l, len)?;
/// s.solve()?;
///
/// let (x, y) = s.point(b)?;
/// assert!((x - 10.0).abs() < 1e-9);
/// assert!(y.abs() < 1e-9);
/// # Ok::<(), nguo_constraints::Error>(())
/// ```
pub struct System {
    raw: NonNull<ffi::GcsSystem>,
    algorithm: Algorithm,
    /// Cleared by any structural change so a stale diagnosis cannot be read
    /// back through [`System::dof`].
    diagnosed: bool,
}

// SAFETY: `raw` points to a `GcsSystem` allocated by `gcs_new`, owned solely
// by this `System`, and never aliased — no other Rust value and no C++ global
// holds a copy of the pointer. planegcs keeps no process-wide mutable state;
// the only static in the build is the `Base::Console` singleton in our compat
// header, whose initialisation is thread-safe and which holds no state after
// init. Moving the owning `System` to another thread moves the whole object
// graph with it. `Sync` is deliberately not implemented.
unsafe impl Send for System {}

impl System {
    /// Aborts on allocation failure, like `Box::new`.
    #[must_use]
    pub fn new() -> Self {
        // SAFETY: `gcs_new` takes no arguments and returns either null or a
        // fresh, uniquely owned allocation.
        let raw = unsafe { ffi::gcs_new() };
        Self {
            raw: NonNull::new(raw).expect("nguo-constraints: out of memory allocating GcsSystem"),
            algorithm: Algorithm::default(),
            diagnosed: false,
        }
    }

    fn as_ptr(&self) -> *mut ffi::GcsSystem {
        self.raw.as_ptr()
    }

    /// Selects the numerical method used by [`Self::solve`] and
    /// [`Self::diagnose`].
    pub fn set_algorithm(&mut self, algorithm: Algorithm) {
        self.algorithm = algorithm;
    }

    /// Caps solver iterations. planegcs has its own default; set this lower to
    /// bound the worst case on an interactive path, or higher for a stiff
    /// system. There is no wall-clock timeout, deliberately: a time-based cut
    /// would make results depend on machine load and break determinism.
    pub fn set_max_iterations(&mut self, iters: i32) {
        // SAFETY: `self.raw` is a live system; the call only writes two `int`
        // fields on it.
        unsafe { ffi::gcs_set_max_iterations(self.as_ptr(), iters) }
    }

    /// Sets the residual below which a constraint counts as satisfied.
    pub fn set_convergence(&mut self, eps: f64) {
        // SAFETY: `self.raw` is a live system; the call only writes two
        // `double` fields on it.
        unsafe { ffi::gcs_set_convergence(self.as_ptr(), eps) }
    }

    // -- parameters --------------------------------------------------------

    /// Adds a value the solver may move.
    pub fn add_free_param(&mut self, value: f64) -> ParamId {
        self.diagnosed = false;
        // SAFETY: live system; `gcs_add_param` only appends to an owned deque
        // and reports allocation failure as `GCS_INVALID_ID`.
        let id = unsafe { ffi::gcs_add_param(self.as_ptr(), value, 1) };
        assert_ne!(id, ffi::GCS_INVALID_ID, "out of memory adding parameter");
        ParamId(id)
    }

    /// Adds a value the solver must respect. This is the anchor for
    /// "distance = parameter" (§9.2): bind it to a named parameter or a body
    /// measurement, change it, re-solve, and the geometry follows.
    pub fn add_driving_param(&mut self, value: f64) -> ParamId {
        self.diagnosed = false;
        // SAFETY: as `add_free_param`.
        let id = unsafe { ffi::gcs_add_param(self.as_ptr(), value, 0) };
        assert_ne!(id, ffi::GCS_INVALID_ID, "out of memory adding parameter");
        ParamId(id)
    }

    /// Changes a parameter's value without touching the system's structure.
    ///
    /// This is the incremental path: a following [`Self::solve`] reuses the
    /// existing subsystem partition and DOF analysis instead of rebuilding
    /// them, which is what makes dragging a slider affordable.
    pub fn set_param(&mut self, param: ParamId, value: f64) -> Result<()> {
        // SAFETY: live system; the shim range-checks `param` and returns
        // `GCS_ERR_BAD_HANDLE` rather than indexing out of bounds.
        let rc = unsafe { ffi::gcs_set_param(self.as_ptr(), param.0, value) };
        check_unit(rc)
    }

    /// Reads a parameter's current value.
    pub fn param(&self, param: ParamId) -> Result<f64> {
        let mut out = 0.0;
        // SAFETY: live system; `out` is a valid, writable `f64`. The shim
        // range-checks `param` and only writes `out` when it is in range.
        let rc = unsafe { ffi::gcs_get_param(self.as_ptr(), param.0, &mut out) };
        check_unit(rc).map(|()| out)
    }

    /// How many parameters exist, free and driving together.
    #[must_use]
    pub fn param_count(&self) -> u32 {
        // SAFETY: live system; read-only.
        unsafe { ffi::gcs_param_count(self.as_ptr()) }
    }

    // -- geometry ----------------------------------------------------------

    /// Adds a point the solver may move, seeded at `(x, y)`.
    pub fn add_point(&mut self, x: f64, y: f64) -> PointId {
        let px = self.add_free_param(x);
        let py = self.add_free_param(y);
        self.point_from(px, py).expect("freshly created parameters")
    }

    /// Adds a point pinned at `(x, y)`. Every sketch needs at least one, or
    /// the whole thing is free to translate and rotate.
    pub fn add_fixed_point(&mut self, x: f64, y: f64) -> PointId {
        let px = self.add_driving_param(x);
        let py = self.add_driving_param(y);
        self.point_from(px, py).expect("freshly created parameters")
    }

    /// Builds a point over parameters you already own, so two points can share
    /// a coordinate.
    pub fn point_from(&mut self, x: ParamId, y: ParamId) -> Result<PointId> {
        // SAFETY: live system; the shim range-checks both handles.
        let id = unsafe { ffi::gcs_add_point(self.as_ptr(), x.0, y.0) };
        check_handle(id).map(PointId)
    }

    /// Reads a point's current coordinates.
    pub fn point(&self, point: PointId) -> Result<(f64, f64)> {
        let mut x = 0.0;
        let mut y = 0.0;
        // SAFETY: live system; `x` and `y` are valid writable `f64`s, and the
        // shim writes them only after range-checking `point`.
        let rc = unsafe { ffi::gcs_point_coords(self.as_ptr(), point.0, &mut x, &mut y) };
        check_unit(rc).map(|()| (x, y))
    }

    pub fn add_line(&mut self, p1: PointId, p2: PointId) -> Result<LineId> {
        self.diagnosed = false;
        // SAFETY: live system; the shim range-checks both point handles.
        let id = unsafe { ffi::gcs_add_line(self.as_ptr(), p1.0, p2.0) };
        check_handle(id).map(LineId)
    }

    pub fn add_circle(&mut self, centre: PointId, radius: ParamId) -> Result<CircleId> {
        self.diagnosed = false;
        // SAFETY: live system; the shim range-checks both handles.
        let id = unsafe { ffi::gcs_add_circle(self.as_ptr(), centre.0, radius.0) };
        check_handle(id).map(CircleId)
    }

    /// Adds an arc. `start_point` and `end_point` are tied to
    /// `centre`/`radius`/the angles by an internal rule constraint, so they may
    /// be seeded roughly and will be corrected by the first solve. Angles are
    /// radians, counter-clockwise from +x.
    #[allow(clippy::too_many_arguments)]
    pub fn add_arc(
        &mut self,
        centre: PointId,
        radius: ParamId,
        start_angle: ParamId,
        end_angle: ParamId,
        start_point: PointId,
        end_point: PointId,
    ) -> Result<ArcId> {
        self.diagnosed = false;
        // SAFETY: live system; the shim range-checks all six handles before
        // dereferencing any of them.
        let id = unsafe {
            ffi::gcs_add_arc(
                self.as_ptr(),
                centre.0,
                radius.0,
                start_angle.0,
                end_angle.0,
                start_point.0,
                end_point.0,
            )
        };
        check_handle(id).map(ArcId)
    }

    // -- geometric constraints (§9.2) --------------------------------------

    /// Two points occupy the same position.
    pub fn coincident(&mut self, p1: PointId, p2: PointId) -> Result<ConstraintId> {
        // SAFETY: live system; handles are range-checked by the shim.
        let tag = unsafe { ffi::gcs_c_coincident(self.as_ptr(), p1.0, p2.0) };
        self.constraint(tag)
    }

    pub fn horizontal(&mut self, line: LineId) -> Result<ConstraintId> {
        // SAFETY: live system; handle range- and kind-checked by the shim.
        let tag = unsafe { ffi::gcs_c_horizontal(self.as_ptr(), line.0) };
        self.constraint(tag)
    }

    pub fn vertical(&mut self, line: LineId) -> Result<ConstraintId> {
        // SAFETY: live system; handle range- and kind-checked by the shim.
        let tag = unsafe { ffi::gcs_c_vertical(self.as_ptr(), line.0) };
        self.constraint(tag)
    }

    pub fn parallel(&mut self, l1: LineId, l2: LineId) -> Result<ConstraintId> {
        // SAFETY: live system; handles range- and kind-checked by the shim.
        let tag = unsafe { ffi::gcs_c_parallel(self.as_ptr(), l1.0, l2.0) };
        self.constraint(tag)
    }

    pub fn perpendicular(&mut self, l1: LineId, l2: LineId) -> Result<ConstraintId> {
        // SAFETY: live system; handles range- and kind-checked by the shim.
        let tag = unsafe { ffi::gcs_c_perpendicular(self.as_ptr(), l1.0, l2.0) };
        self.constraint(tag)
    }

    /// Two curves touch at exactly one point. At least one must be a circle or
    /// an arc; line-to-line tangency returns [`Error::BadKind`].
    pub fn tangent(
        &mut self,
        a: impl Into<CurveId>,
        b: impl Into<CurveId>,
    ) -> Result<ConstraintId> {
        let (a, b) = (a.into(), b.into());
        // SAFETY: live system; handles range- and kind-checked by the shim,
        // which returns `GCS_ERR_BAD_KIND` for combinations it cannot express.
        let tag = unsafe { ffi::gcs_c_tangent(self.as_ptr(), a.0, b.0) };
        self.constraint(tag)
    }

    /// `p1` and `p2` mirror across `line`.
    pub fn symmetric_about_line(
        &mut self,
        p1: PointId,
        p2: PointId,
        line: LineId,
    ) -> Result<ConstraintId> {
        // SAFETY: live system; handles range- and kind-checked by the shim.
        let tag = unsafe { ffi::gcs_c_symmetric_line(self.as_ptr(), p1.0, p2.0, line.0) };
        self.constraint(tag)
    }

    /// `p1` and `p2` mirror through `centre`.
    pub fn symmetric_about_point(
        &mut self,
        p1: PointId,
        p2: PointId,
        centre: PointId,
    ) -> Result<ConstraintId> {
        // SAFETY: live system; handles are range-checked by the shim.
        let tag = unsafe { ffi::gcs_c_symmetric_point(self.as_ptr(), p1.0, p2.0, centre.0) };
        self.constraint(tag)
    }

    pub fn equal_length(&mut self, l1: LineId, l2: LineId) -> Result<ConstraintId> {
        // SAFETY: live system; handles range- and kind-checked by the shim.
        let tag = unsafe { ffi::gcs_c_equal_length(self.as_ptr(), l1.0, l2.0) };
        self.constraint(tag)
    }

    /// Two lines lie on the same infinite line.
    ///
    /// planegcs has no single constraint for this, so the shim emits
    /// parallelism plus a point-on-line under one tag. It therefore removes
    /// two degrees of freedom while DOF diagnosis reports it as one
    /// constraint.
    pub fn collinear(&mut self, l1: LineId, l2: LineId) -> Result<ConstraintId> {
        // SAFETY: live system; handles range- and kind-checked by the shim.
        let tag = unsafe { ffi::gcs_c_collinear(self.as_ptr(), l1.0, l2.0) };
        self.constraint(tag)
    }

    /// A point lies on a line, circle or arc.
    pub fn point_on_curve(
        &mut self,
        point: PointId,
        curve: impl Into<CurveId>,
    ) -> Result<ConstraintId> {
        let curve = curve.into();
        // SAFETY: live system; handles range- and kind-checked by the shim.
        let tag = unsafe { ffi::gcs_c_point_on_curve(self.as_ptr(), point.0, curve.0) };
        self.constraint(tag)
    }

    // -- dimensional constraints (§9.2, §9.4) ------------------------------

    /// Distance between two points equals `dist`. Pass a driving parameter to
    /// bind it to a named value or a body measurement.
    pub fn distance(&mut self, p1: PointId, p2: PointId, dist: ParamId) -> Result<ConstraintId> {
        // SAFETY: live system; handles are range-checked by the shim.
        let tag = unsafe { ffi::gcs_c_distance_pp(self.as_ptr(), p1.0, p2.0, dist.0) };
        self.constraint(tag)
    }

    /// Perpendicular distance from a point to a line equals `dist`.
    pub fn distance_to_line(
        &mut self,
        point: PointId,
        line: LineId,
        dist: ParamId,
    ) -> Result<ConstraintId> {
        // SAFETY: live system; handles range- and kind-checked by the shim.
        let tag = unsafe { ffi::gcs_c_distance_pl(self.as_ptr(), point.0, line.0, dist.0) };
        self.constraint(tag)
    }

    /// A line's length equals `dist`.
    pub fn length(&mut self, line: LineId, dist: ParamId) -> Result<ConstraintId> {
        // SAFETY: live system; handles range- and kind-checked by the shim.
        let tag = unsafe { ffi::gcs_c_length(self.as_ptr(), line.0, dist.0) };
        self.constraint(tag)
    }

    /// Angle from `l1` to `l2`, in radians.
    pub fn angle(&mut self, l1: LineId, l2: LineId, angle: ParamId) -> Result<ConstraintId> {
        // SAFETY: live system; handles range- and kind-checked by the shim.
        let tag = unsafe { ffi::gcs_c_angle(self.as_ptr(), l1.0, l2.0, angle.0) };
        self.constraint(tag)
    }

    /// A circle's or arc's radius equals `radius`. A line returns
    /// [`Error::BadKind`].
    pub fn radius(&mut self, curve: impl Into<CurveId>, radius: ParamId) -> Result<ConstraintId> {
        let curve = curve.into();
        // SAFETY: live system; handles range- and kind-checked by the shim.
        let tag = unsafe { ffi::gcs_c_radius(self.as_ptr(), curve.0, radius.0) };
        self.constraint(tag)
    }

    // -- solving and diagnosis ---------------------------------------------

    /// Runs planegcs' QR-based DOF analysis.
    ///
    /// Cheap enough to call after every structural edit, which is what a
    /// constraint UI wants; not cheap enough to call per drag frame, which is
    /// why [`Self::set_param`] does not invalidate it.
    pub fn diagnose(&mut self) -> Result<Diagnosis> {
        let mut dof = 0i32;
        // SAFETY: live system; `dof` is a valid writable `i32`. `gcs_diagnose`
        // initialises the system if the structure changed, catches any C++
        // exception internally, and writes `dof` only when it returns 0.
        let rc = unsafe { ffi::gcs_diagnose(self.as_ptr(), self.algorithm.raw(), &mut dof) };
        check_unit(rc)?;
        self.diagnosed = true;
        Ok(Diagnosis {
            dof,
            conflicting: self.tags(ffi::gcs_conflicting),
            redundant: self.tags(ffi::gcs_redundant),
            partially_redundant: self.tags(ffi::gcs_partially_redundant),
        })
    }

    /// The remaining degrees of freedom from the last [`Self::diagnose`].
    /// See [`Diagnosis::dof`] for why this is signed.
    pub fn dof(&self) -> Result<i32> {
        if !self.diagnosed {
            return Err(Error::NoDiagnosis);
        }
        // SAFETY: live system; read-only accessor.
        Ok(unsafe { ffi::gcs_dof(self.as_ptr()) })
    }

    /// Solves and writes the result back into the parameters.
    ///
    /// On failure the parameters are restored to the values they held before
    /// the attempt, so a caller that ignores the error still never observes
    /// NaN or half-solved geometry.
    ///
    /// The solve always restarts from the geometry as it stood when the system
    /// was last initialised, not from the previous solution. That costs a
    /// little convergence speed and buys determinism: the same system solved
    /// twice gives the same answer, and dragging a parameter back and forth
    /// does not accumulate drift.
    pub fn solve(&mut self) -> Result<()> {
        // SAFETY: live system. `gcs_solve` initialises if needed, catches any
        // C++ exception, and applies or rolls back the solution itself.
        let status = unsafe { ffi::gcs_solve(self.as_ptr(), self.algorithm.raw()) };
        match status {
            ffi::GCS_STATUS_SUCCESS => Ok(()),
            ffi::GCS_STATUS_CONVERGED => Err(Error::NoSolution(SolveStatus::Converged)),
            ffi::GCS_STATUS_FAILED => Err(Error::NoSolution(SolveStatus::Failed)),
            ffi::GCS_STATUS_INVALID_SOLUTION => {
                Err(Error::NoSolution(SolveStatus::InvalidSolution))
            }
            ffi::GCS_STATUS_NOT_INITIALISED => Err(Error::NoSolution(SolveStatus::NotInitialised)),
            ffi::GCS_STATUS_EXCEPTION => Err(Error::NoSolution(SolveStatus::Exception)),
            _ => Err(Error::NoSolution(SolveStatus::Exception)),
        }
    }

    // -- internals ---------------------------------------------------------

    fn constraint(&mut self, tag: i32) -> Result<ConstraintId> {
        self.diagnosed = false;
        if tag < 0 {
            return Err(map_neg(tag));
        }
        Ok(ConstraintId(tag))
    }

    /// Two-call pattern: ask for the length, then fill a buffer of that size.
    fn tags(
        &self,
        get: unsafe extern "C" fn(*const ffi::GcsSystem, *mut i32, usize) -> usize,
    ) -> Vec<ConstraintId> {
        // SAFETY: live system; a null `out` with `cap == 0` is the documented
        // way to query the length without writing anything.
        let n = unsafe { get(self.as_ptr(), std::ptr::null_mut(), 0) };
        if n == 0 {
            return Vec::new();
        }
        let mut buf = vec![0i32; n];
        // SAFETY: `buf` holds exactly `n` writable `i32`s and `n` is passed as
        // the capacity, so the shim cannot write past the end.
        let written = unsafe { get(self.as_ptr(), buf.as_mut_ptr(), n) };
        buf.truncate(written.min(n));
        buf.into_iter().map(ConstraintId).collect()
    }
}

impl Default for System {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for System {
    fn drop(&mut self) {
        // SAFETY: `raw` came from `gcs_new`, is owned exclusively by this
        // value, and `Drop` runs at most once. `gcs_free` deletes the system,
        // which deletes every constraint planegcs owns plus every parameter
        // and geometry object the shim owns.
        unsafe { ffi::gcs_free(self.as_ptr()) }
    }
}

impl std::fmt::Debug for System {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("System")
            .field("params", &self.param_count())
            .field("algorithm", &self.algorithm)
            .finish_non_exhaustive()
    }
}

fn map_neg(code: i32) -> Error {
    match code {
        ffi::GCS_ERR_BAD_HANDLE => Error::BadHandle,
        ffi::GCS_ERR_BAD_KIND => Error::BadKind,
        ffi::GCS_ERR_EXCEPTION => Error::Exception,
        _ => Error::Exception,
    }
}

fn check_unit(rc: c_int) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(map_neg(rc))
    }
}

fn check_handle(id: u32) -> Result<u32> {
    if id == ffi::GCS_INVALID_ID {
        Err(Error::BadHandle)
    } else {
        Ok(id)
    }
}
