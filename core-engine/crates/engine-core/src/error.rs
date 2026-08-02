// SPDX-License-Identifier: GPL-2.0-or-later
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CoreError {
    #[error("empty mesh: {0}")]
    EmptyMesh(&'static str),
    #[error("index {index} out of bounds for {count} vertices in {what}")]
    IndexOutOfBounds {
        what: &'static str,
        index: u32,
        count: usize,
    },
    #[error("mismatched buffer lengths: {0}")]
    LengthMismatch(&'static str),
    #[error("solver diverged: {0}")]
    Diverged(String),
}

pub type Result<T> = std::result::Result<T, CoreError>;
