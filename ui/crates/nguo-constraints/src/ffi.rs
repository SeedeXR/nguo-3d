// SPDX-License-Identifier: GPL-2.0-or-later
//
// Raw declarations of the C shim in `cpp/nguo_gcs.h`. This module is the only
// place that names an `extern "C"` symbol; everything above it is safe.
//
// Hand-written on purpose. The header is ~30 functions of `uint32_t`/`double`
// and never changes shape, so bindgen would be a build-time dependency and a
// generated-code review burden for no benefit. Keep it in step with the header
// by hand; the integration tests exercise every entry point declared here.

#![allow(non_camel_case_types)]

use std::ffi::{c_double, c_int, c_void};

/// Opaque; matches `struct GcsSystem` in the shim.
#[repr(C)]
pub struct GcsSystem {
    _private: [u8; 0],
    /// Not `Send`/`Sync` by default and not constructible from Rust.
    _marker: core::marker::PhantomData<(*mut c_void, core::marker::PhantomPinned)>,
}

pub const GCS_INVALID_ID: u32 = 0xFFFF_FFFF;

pub const GCS_ERR_BAD_HANDLE: i32 = -1;
pub const GCS_ERR_BAD_KIND: i32 = -2;
pub const GCS_ERR_EXCEPTION: i32 = -3;

pub const GCS_STATUS_SUCCESS: c_int = 0;
pub const GCS_STATUS_CONVERGED: c_int = 1;
pub const GCS_STATUS_FAILED: c_int = 2;
pub const GCS_STATUS_INVALID_SOLUTION: c_int = 3;
pub const GCS_STATUS_NOT_INITIALISED: c_int = 4;
pub const GCS_STATUS_EXCEPTION: c_int = 5;

pub const GCS_ALG_BFGS: c_int = 0;
pub const GCS_ALG_LEVENBERG_MARQUARDT: c_int = 1;
pub const GCS_ALG_DOGLEG: c_int = 2;

extern "C" {
    pub fn gcs_new() -> *mut GcsSystem;
    pub fn gcs_free(sys: *mut GcsSystem);

    pub fn gcs_add_param(sys: *mut GcsSystem, value: c_double, is_free: c_int) -> u32;
    pub fn gcs_set_param(sys: *mut GcsSystem, param: u32, value: c_double) -> c_int;
    pub fn gcs_get_param(sys: *const GcsSystem, param: u32, out: *mut c_double) -> c_int;
    pub fn gcs_param_count(sys: *const GcsSystem) -> u32;

    pub fn gcs_add_point(sys: *mut GcsSystem, px: u32, py: u32) -> u32;
    pub fn gcs_point_coords(
        sys: *const GcsSystem,
        point: u32,
        x: *mut c_double,
        y: *mut c_double,
    ) -> c_int;

    pub fn gcs_add_line(sys: *mut GcsSystem, p1: u32, p2: u32) -> u32;
    pub fn gcs_add_circle(sys: *mut GcsSystem, centre: u32, radius: u32) -> u32;
    pub fn gcs_add_arc(
        sys: *mut GcsSystem,
        centre: u32,
        radius: u32,
        start_angle: u32,
        end_angle: u32,
        start_point: u32,
        end_point: u32,
    ) -> u32;

    pub fn gcs_c_coincident(sys: *mut GcsSystem, p1: u32, p2: u32) -> i32;
    pub fn gcs_c_horizontal(sys: *mut GcsSystem, line: u32) -> i32;
    pub fn gcs_c_vertical(sys: *mut GcsSystem, line: u32) -> i32;
    pub fn gcs_c_parallel(sys: *mut GcsSystem, l1: u32, l2: u32) -> i32;
    pub fn gcs_c_perpendicular(sys: *mut GcsSystem, l1: u32, l2: u32) -> i32;
    pub fn gcs_c_tangent(sys: *mut GcsSystem, c1: u32, c2: u32) -> i32;
    pub fn gcs_c_symmetric_line(sys: *mut GcsSystem, p1: u32, p2: u32, line: u32) -> i32;
    pub fn gcs_c_symmetric_point(sys: *mut GcsSystem, p1: u32, p2: u32, centre: u32) -> i32;
    pub fn gcs_c_equal_length(sys: *mut GcsSystem, l1: u32, l2: u32) -> i32;
    pub fn gcs_c_collinear(sys: *mut GcsSystem, l1: u32, l2: u32) -> i32;
    pub fn gcs_c_point_on_curve(sys: *mut GcsSystem, point: u32, curve: u32) -> i32;

    pub fn gcs_c_distance_pp(sys: *mut GcsSystem, p1: u32, p2: u32, dist: u32) -> i32;
    pub fn gcs_c_distance_pl(sys: *mut GcsSystem, point: u32, line: u32, dist: u32) -> i32;
    pub fn gcs_c_length(sys: *mut GcsSystem, line: u32, dist: u32) -> i32;
    pub fn gcs_c_angle(sys: *mut GcsSystem, l1: u32, l2: u32, angle: u32) -> i32;
    pub fn gcs_c_radius(sys: *mut GcsSystem, curve: u32, radius: u32) -> i32;

    pub fn gcs_set_max_iterations(sys: *mut GcsSystem, iters: c_int);
    pub fn gcs_set_convergence(sys: *mut GcsSystem, eps: c_double);

    pub fn gcs_diagnose(sys: *mut GcsSystem, algorithm: c_int, out_dof: *mut i32) -> c_int;
    pub fn gcs_dof(sys: *const GcsSystem) -> c_int;
    pub fn gcs_conflicting(sys: *const GcsSystem, out: *mut i32, cap: usize) -> usize;
    pub fn gcs_redundant(sys: *const GcsSystem, out: *mut i32, cap: usize) -> usize;
    pub fn gcs_partially_redundant(sys: *const GcsSystem, out: *mut i32, cap: usize) -> usize;

    pub fn gcs_solve(sys: *mut GcsSystem, algorithm: c_int) -> c_int;
}
