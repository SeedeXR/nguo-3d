// SPDX-License-Identifier: GPL-2.0-or-later
//! `nguo-mcp` — an MCP server exposing the nguo-3d operation surface so Claude
//! Code and other agentic tools can drive the cloth suite.
//!
//! The tool list is **generated** from [`nguo_cli::ops::SPECS`] and every call
//! goes through [`nguo_cli::ops::dispatch`] — the same code path the `nguo`
//! binary uses. There is no second copy of the command surface, and a test
//! fails if the two front ends ever diverge.
//!
//! Registration with Claude Code (stdio transport):
//! ```text
//! claude mcp add nguo -- /path/to/nguo-mcp
//! ```
//!
//! ## Long-running work
//! `drape` can run for minutes. Over MCP it therefore starts a background job
//! and returns a job id immediately; poll [`DRAPE_STATUS`] for the result. The
//! server keeps serving other tools while a drape runs.

use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};

use nguo_cli::ops::{self, Env, OpError, OpSpec, SPECS};
use rmcp::{
    handler::server::ServerHandler,
    model::{
        CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, Implementation,
        InitializeResult, JsonObject, ListToolsResult, PaginatedRequestParams, ServerCapabilities,
        Tool, ToolAnnotations,
    },
    service::{RequestContext, RoleServer},
    ErrorData as McpError,
};
use serde_json::{json, Value};

/// The one tool that is not an [`ops`] operation: the poll half of the async
/// `drape` wrapper. It only makes sense in a long-lived server, so the CLI has
/// no equivalent (a fresh CLI process holds no jobs).
pub const DRAPE_STATUS: &str = "drape_status";

/// Tools this server adds on top of [`SPECS`]. The surface-sync test uses it.
pub const MCP_ONLY_TOOLS: &[&str] = &[DRAPE_STATUS];

#[derive(Debug, Clone)]
enum JobState {
    Running,
    Done(Value),
    Failed(String),
}

/// The MCP server. Cheap to clone — all state is shared.
#[derive(Clone)]
pub struct NguoServer {
    env: Arc<Env>,
    jobs: Arc<Mutex<HashMap<String, JobState>>>,
    next_job: Arc<AtomicU64>,
}

impl Default for NguoServer {
    fn default() -> Self {
        Self::new(Env::default())
    }
}

impl NguoServer {
    pub fn new(env: Env) -> Self {
        Self {
            env: Arc::new(env),
            jobs: Arc::new(Mutex::new(HashMap::new())),
            next_job: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Every tool this server advertises, generated from [`SPECS`].
    pub fn tools() -> Vec<Tool> {
        SPECS
            .iter()
            .map(tool_from_spec)
            .chain(std::iter::once(drape_status_tool()))
            .collect()
    }

    fn start_drape(&self, args: Value) -> Value {
        let id = format!(
            "drape-{}",
            self.next_job.fetch_add(1, Ordering::Relaxed) + 1
        );
        self.jobs
            .lock()
            .expect("job map")
            .insert(id.clone(), JobState::Running);
        let env = self.env.clone();
        let jobs = self.jobs.clone();
        let job_id = id.clone();
        tokio::spawn(async move {
            let state = match ops::dispatch("drape", &args, &env).await {
                Ok(value) => JobState::Done(value),
                Err(OpError::Invalid(report)) => JobState::Failed(report.to_string()),
                Err(e) => JobState::Failed(e.to_string()),
            };
            jobs.lock().expect("job map").insert(job_id, state);
        });
        json!({
            "job_id": id,
            "status": "running",
            "poll_with": DRAPE_STATUS,
            "note": "The drape runs in the background. Call drape_status with this job_id until status is \"done\" or \"failed\"."
        })
    }

    fn drape_status(&self, args: &Value) -> Result<Value, OpError> {
        let Some(id) = args.get("job_id").and_then(Value::as_str) else {
            return Err(OpError::Args {
                op: DRAPE_STATUS.into(),
                detail: "`job_id` is required".into(),
            });
        };
        let state = self.jobs.lock().expect("job map").get(id).cloned();
        match state {
            None => Err(OpError::Failed(format!(
                "unknown job id `{id}`. Job ids live only in this server process; \
                 a restart forgets them."
            ))),
            Some(JobState::Running) => Ok(json!({"job_id": id, "status": "running"})),
            Some(JobState::Done(result)) => {
                Ok(json!({"job_id": id, "status": "done", "result": result}))
            }
            Some(JobState::Failed(error)) => {
                Err(OpError::Failed(format!("drape job `{id}` failed: {error}")))
            }
        }
    }
}

fn to_schema(value: Value) -> Arc<JsonObject> {
    Arc::new(match value {
        Value::Object(map) => map,
        other => {
            let mut map = JsonObject::new();
            map.insert("schema".into(), other);
            map
        }
    })
}

fn tool_from_spec(spec: &'static OpSpec) -> Tool {
    // The agent picks by description, so it must describe what the MCP tool
    // really does — including the async wrapper around `drape`.
    let mut description = format!("{} {}", spec.summary, spec.description);
    if spec.name == "drape" {
        description.push_str(
            " OVER MCP THIS CALL IS ASYNCHRONOUS: it returns a job_id immediately and the \
             drape runs in the background. Poll `drape_status` with that job_id for the \
             report. Each call starts a NEW job, so calling it twice runs the solve twice \
             (the result is identical; only the frame files in out_dir are rewritten).",
        );
    }
    Tool::new(spec.name, description, to_schema((spec.schema)())).with_annotations(
        ToolAnnotations::new()
            .read_only(spec.read_only)
            .destructive(false)
            .idempotent(spec.idempotent)
            .open_world(false),
    )
}

fn drape_status_tool() -> Tool {
    Tool::new(
        DRAPE_STATUS,
        "Poll a background drape started by the `drape` tool. Returns \
         {status: \"running\"} while the solver is working, {status: \"done\", result: ...} \
         with the engine's drape report when it finishes, and a tool error carrying the \
         engine's real message if it failed. Job ids exist only for the lifetime of this \
         server process.",
        to_schema(json!({
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "The job_id returned by `drape`."}
            },
            "required": ["job_id"],
            "additionalProperties": false
        })),
    )
    .with_annotations(
        ToolAnnotations::new()
            .read_only(true)
            .destructive(false)
            .idempotent(true)
            .open_world(false),
    )
}

fn ok_result(value: &Value) -> CallToolResult {
    CallToolResult::success(vec![ContentBlock::text(
        serde_json::to_string_pretty(value).unwrap_or_else(|e| e.to_string()),
    )])
}

/// Map an operation failure onto MCP's two failure modes: a bad request is a
/// protocol error, a real failure is a tool error the caller can read.
fn error_response(err: OpError) -> Result<CallToolResponse, McpError> {
    match err {
        OpError::Unknown(name) => Err(McpError::invalid_params(
            format!("unknown tool `{name}`"),
            None,
        )),
        OpError::Args { op, detail } => Err(McpError::invalid_params(
            format!("invalid arguments for `{op}`: {detail}"),
            None,
        )),
        OpError::Failed(message) => {
            Ok(CallToolResult::error(vec![ContentBlock::text(message)]).into())
        }
        OpError::Invalid(report) => Ok(CallToolResult::error(vec![ContentBlock::text(format!(
            "validation failed:\n{}",
            serde_json::to_string_pretty(&report).unwrap_or_else(|e| e.to_string())
        ))])
        .into()),
    }
}

impl ServerHandler for NguoServer {
    fn get_info(&self) -> InitializeResult {
        let mut info = InitializeResult::new(ServerCapabilities::builder().enable_tools().build());
        info.server_info = Implementation::new("nguo-mcp", env!("CARGO_PKG_VERSION"));
        info.instructions = Some(
            "Drives the nguo-3d cloth suite headlessly. Start with `schema` for the full \
             surface and the process contract. Scenes are engine-api Scene JSON files \
             (see core-engine/examples). Pattern-document editing (pieces, seams) is not \
             available yet and is deliberately absent rather than stubbed."
                .into(),
        );
        info
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        Ok(ListToolsResult::with_all_items(Self::tools()))
    }

    fn get_tool(&self, name: &str) -> Option<Tool> {
        Self::tools().into_iter().find(|t| t.name == name)
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, McpError> {
        let name = request.name.to_string();
        let args = request
            .arguments
            .map(Value::Object)
            .unwrap_or_else(|| json!({}));

        if name == DRAPE_STATUS {
            return match self.drape_status(&args) {
                Ok(value) => Ok(ok_result(&value).into()),
                Err(e) => error_response(e),
            };
        }
        // Long solves must not block the server: hand back a job id.
        if name == "drape" {
            // Fail a malformed request now, not inside a background job.
            if let Err(e) = ops::check_args("drape", &args) {
                return error_response(e);
            }
            return Ok(ok_result(&self.start_drape(args)).into());
        }
        match ops::dispatch(&name, &args, &self.env).await {
            Ok(value) => Ok(ok_result(&value).into()),
            Err(e) => error_response(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_op_becomes_a_tool_with_a_schema_and_honest_annotations() {
        let tools = NguoServer::tools();
        assert_eq!(tools.len(), SPECS.len() + MCP_ONLY_TOOLS.len());
        for spec in SPECS {
            let tool = tools
                .iter()
                .find(|t| t.name == spec.name)
                .unwrap_or_else(|| panic!("`{}` is missing from the MCP tool list", spec.name));
            let description = tool.description.as_deref().unwrap_or_default();
            assert!(
                description.contains(spec.summary),
                "`{}` lost its summary",
                spec.name
            );
            assert_eq!(tool.input_schema.get("type"), Some(&json!("object")));
            let annotations = tool.annotations.as_ref().expect("annotations");
            assert_eq!(annotations.read_only_hint, Some(spec.read_only));
        }
        assert!(tools.iter().any(|t| t.name == DRAPE_STATUS));
    }

    #[test]
    fn drape_advertises_that_it_is_asynchronous_over_mcp() {
        let drape = NguoServer::tools()
            .into_iter()
            .find(|t| t.name == "drape")
            .expect("drape tool");
        let description = drape.description.unwrap_or_default();
        assert!(
            description.contains("job_id"),
            "an agent would expect a report, not a job id"
        );
        assert!(description.contains(DRAPE_STATUS));
    }

    #[test]
    fn unknown_job_ids_fail_loudly_rather_than_reporting_success() {
        let server = NguoServer::default();
        let err = server
            .drape_status(&json!({"job_id": "drape-999"}))
            .expect_err("must not fabricate a result");
        assert!(err.to_string().contains("unknown job id"));
    }
}
