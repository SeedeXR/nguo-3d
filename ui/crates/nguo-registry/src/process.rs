// SPDX-License-Identifier: GPL-2.0-or-later
//! Process lifecycle types: what a running piece of work looks like from the
//! outside.
//!
//! A **process** here is one long-running, cancellable unit of work with an
//! observable outcome: a drape solve, an import, an export, a triangulation, a
//! constraint solve, a DCC round-trip. It is not a thread and not an OS
//! process. Short pure calls (a seam offset, an expression evaluation) are
//! features but not processes — `tracing` spans already cover those, and
//! putting them in the live list would drown it.

use std::time::Duration;

use crate::Feature;

/// Identifier of one process instance. Monotonic, assigned under the registry
/// lock, so a snapshot's ordering by id is also its creation order.
///
/// `ProcessId(0)` is the null id: an inert handle, from a disabled registry or
/// a stale epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProcessId(pub u64);

impl std::fmt::Display for ProcessId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// Where a process is in its lifecycle.
///
/// ```text
///   Queued ──► Running ──► Done
///     │           │
///     └───────────┴──────► Failed
///     └───────────┴──────► Cancelled
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ProcessState {
    /// Submitted, not started. Mirrors the one job `nguo-engine-bridge` holds
    /// in its `watch` slot.
    Queued,
    /// Actually executing.
    Running,
    /// Finished, successfully.
    Done,
    /// Finished, with an error chain recorded.
    Failed,
    /// Superseded by a newer epoch, cancelled by the user, or the handle was
    /// dropped without a terminal call.
    Cancelled,
}

impl ProcessState {
    /// Lower-case stable name, for logs and the debug overlay.
    pub const fn as_str(self) -> &'static str {
        match self {
            ProcessState::Queued => "queued",
            ProcessState::Running => "running",
            ProcessState::Done => "done",
            ProcessState::Failed => "failed",
            ProcessState::Cancelled => "cancelled",
        }
    }

    /// Nothing follows a terminal state.
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            ProcessState::Done | ProcessState::Failed | ProcessState::Cancelled
        )
    }

    /// The whole legality rule, in one place.
    ///
    /// `Queued -> Done` is **not** legal: you cannot finish work that never
    /// started. Call `run()` first — that is the point of having the state.
    pub const fn can_transition_to(self, to: ProcessState) -> bool {
        matches!(
            (self, to),
            (
                ProcessState::Queued,
                ProcessState::Running | ProcessState::Failed | ProcessState::Cancelled
            ) | (
                ProcessState::Running,
                ProcessState::Done | ProcessState::Failed | ProcessState::Cancelled
            )
        )
    }
}

impl std::fmt::Display for ProcessState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A process that has not finished, as seen by a snapshot.
#[derive(Debug, Clone)]
pub struct ActiveProcess {
    pub id: ProcessId,
    pub feature: &'static Feature,
    /// `Queued` or `Running` — an active process is never terminal.
    pub state: ProcessState,
    /// The generation id this work belongs to, raw. `0` means "not
    /// epoch-managed". See the crate docs on why this is a `u64` and not a
    /// second `Epoch` type.
    pub epoch: u64,
    /// `0.0..=1.0` if the work reports progress.
    pub progress: Option<f32>,
    /// Monotonic time since the process was registered.
    pub elapsed: Duration,
}

/// A process that has finished, as seen by a snapshot.
#[derive(Debug, Clone)]
pub struct CompletedProcess {
    pub id: ProcessId,
    pub feature: &'static Feature,
    /// Always one of `Done`, `Failed`, `Cancelled`.
    pub outcome: ProcessState,
    pub epoch: u64,
    /// Registration to terminal transition, monotonic.
    pub duration: Duration,
}

/// A failure, with the error chain walked to its root.
///
/// The UI shows [`FailureRecord::title`] / [`FailureRecord::detail`] in a
/// `nguo_ui_widgets::toast` and the log keeps the whole [`chain`]. Non-fatal by
/// construction (ui/docs/OBSERVABILITY.md §6): recording a failure is the *only*
/// thing this crate does about it.
///
/// [`chain`]: FailureRecord::chain
#[derive(Debug, Clone)]
pub struct FailureRecord {
    pub id: ProcessId,
    pub feature: &'static Feature,
    pub epoch: u64,
    /// `Display` of the error, then of each `source()` in turn, to the root.
    /// Never empty.
    pub chain: Vec<String>,
    pub duration: Duration,
}

impl FailureRecord {
    /// What to put in a toast title: the human feature name.
    pub fn title(&self) -> &'static str {
        self.feature.name
    }

    /// What to put in a toast body: the top-level message.
    pub fn detail(&self) -> &str {
        self.chain.first().map(String::as_str).unwrap_or("")
    }

    /// The bottom of the chain — usually the actionable bit.
    pub fn root_cause(&self) -> &str {
        self.chain.last().map(String::as_str).unwrap_or("")
    }
}

/// Walk `err` and every `source()` under it, `Display`-formatted.
pub(crate) fn error_chain(err: &(dyn std::error::Error + 'static)) -> Vec<String> {
    let mut chain = vec![err.to_string()];
    let mut cursor = err.source();
    // A cyclic `source()` chain is pathological but not impossible; cap it
    // rather than hang the logger.
    // ponytail: fixed cap, not cycle detection. Raise it if a real error
    // type ever nests deeper than this.
    while let Some(e) = cursor {
        if chain.len() >= 32 {
            chain.push("… (error chain truncated at 32)".to_owned());
            break;
        }
        chain.push(e.to_string());
        cursor = e.source();
    }
    chain
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_legal_transitions_are_legal() {
        use ProcessState::*;
        assert!(Queued.can_transition_to(Running));
        assert!(Queued.can_transition_to(Cancelled));
        assert!(Queued.can_transition_to(Failed));
        assert!(Running.can_transition_to(Done));
        assert!(Running.can_transition_to(Failed));
        assert!(Running.can_transition_to(Cancelled));

        // The ones that must be rejected.
        assert!(!Queued.can_transition_to(Done), "finished without running");
        assert!(!Queued.can_transition_to(Queued));
        assert!(!Running.can_transition_to(Running));
        assert!(!Running.can_transition_to(Queued));
        assert!(!Done.can_transition_to(Running), "done -> running");

        // Terminal is terminal, for every terminal state and every target.
        for from in [Done, Failed, Cancelled] {
            assert!(from.is_terminal());
            for to in [Queued, Running, Done, Failed, Cancelled] {
                assert!(
                    !from.can_transition_to(to),
                    "{from} -> {to} must be rejected"
                );
            }
        }
    }

    #[derive(Debug, thiserror::Error)]
    #[error("{msg}")]
    struct Layer {
        msg: &'static str,
        #[source]
        source: Option<Box<Layer>>,
    }

    #[test]
    fn error_chain_walks_to_the_root() {
        let err = Layer {
            msg: "could not drape `bodice`",
            source: Some(Box::new(Layer {
                msg: "mguso exited 1",
                source: Some(Box::new(Layer {
                    msg: "solver diverged at frame 12",
                    source: None,
                })),
            })),
        };
        assert_eq!(
            error_chain(&err),
            vec![
                "could not drape `bodice`",
                "mguso exited 1",
                "solver diverged at frame 12",
            ]
        );
    }

    #[test]
    fn error_chain_of_a_leaf_is_one_entry() {
        let err = Layer {
            msg: "lonely",
            source: None,
        };
        assert_eq!(error_chain(&err), vec!["lonely"]);
    }
}
