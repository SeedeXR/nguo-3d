// SPDX-License-Identifier: GPL-2.0-or-later
//! Drives the MCP server in-process over a duplex pipe — a real client, real
//! JSON-RPC, no external agent — and asserts that the MCP surface and the CLI
//! surface cannot drift apart.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use nguo_mcp::{NguoServer, DRAPE_STATUS, MCP_ONLY_TOOLS};
use rmcp::{model::CallToolRequestParams, ServiceExt};
use serde_json::Value;

fn example(name: &str) -> PathBuf {
    // ui/crates/nguo-mcp -> nguo-3d
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("repo root")
        .join("core-engine/examples")
        .join(name)
}

/// Serve `NguoServer` over an in-memory pipe and return a connected client.
async fn connect() -> rmcp::service::RunningService<rmcp::RoleClient, ()> {
    let (server_io, client_io) = tokio::io::duplex(1 << 16);
    tokio::spawn(async move {
        let service = NguoServer::default()
            .serve(server_io)
            .await
            .expect("server handshake");
        let _ = service.waiting().await;
    });
    ().serve(client_io).await.expect("client handshake")
}

fn args(value: Value) -> rmcp::model::JsonObject {
    value.as_object().cloned().expect("an object")
}

fn text_of(result: &rmcp::model::CallToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|c| match c {
            rmcp::model::ContentBlock::Text(t) => Some(t.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[tokio::test]
async fn lists_tools_with_real_schemas() {
    let client = connect().await;
    let tools = client.list_tools(None).await.expect("tools/list").tools;

    let names: BTreeSet<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
    for spec in nguo_cli::ops::SPECS {
        assert!(names.contains(spec.name), "`{}` is not exposed", spec.name);
    }
    assert!(names.contains(DRAPE_STATUS));

    for tool in &tools {
        let description = tool.description.as_deref().unwrap_or_default();
        assert!(
            description.len() > 40,
            "`{}` has a thin description; agents pick tools by these",
            tool.name
        );
        assert_eq!(
            tool.input_schema.get("type"),
            Some(&Value::String("object".into())),
            "`{}` has no object input schema",
            tool.name
        );
    }
    client.cancel().await.ok();
}

#[tokio::test]
async fn calls_a_tool_and_gets_the_real_result() {
    let client = connect().await;
    let result = client
        .call_tool(
            CallToolRequestParams::new("scene_info").with_arguments(args(
                serde_json::json!({"scene": example("sphere_drape.json").display().to_string()}),
            )),
        )
        .await
        .expect("tools/call");

    assert_eq!(result.is_error, Some(false));
    let payload: Value = serde_json::from_str(&text_of(&result)).expect("tool returned JSON");
    assert_eq!(payload["totals"]["verts"], 144);
    assert_eq!(payload["totals"]["tris"], 242);
    client.cancel().await.ok();
}

#[tokio::test]
async fn a_failing_tool_reports_the_real_error_not_a_fake_success() {
    let client = connect().await;
    let result = client
        .call_tool(
            CallToolRequestParams::new("scene_info").with_arguments(args(
                serde_json::json!({"scene": "/nonexistent/scene.json"}),
            )),
        )
        .await
        .expect("tools/call completes");
    assert_eq!(
        result.is_error,
        Some(true),
        "a missing file must not read as success"
    );
    assert!(text_of(&result).contains("/nonexistent/scene.json"));
    client.cancel().await.ok();
}

#[tokio::test]
async fn polling_an_unknown_job_is_an_error() {
    let client = connect().await;
    let result = client
        .call_tool(
            CallToolRequestParams::new(DRAPE_STATUS)
                .with_arguments(args(serde_json::json!({"job_id": "drape-404"}))),
        )
        .await
        .expect("tools/call completes");
    assert_eq!(result.is_error, Some(true));
    assert!(text_of(&result).contains("unknown job id"));
    client.cancel().await.ok();
}

/// The point of the shared `ops::SPECS`: neither front end can gain or lose an
/// operation without the other. If this fails, one surface drifted.
#[test]
fn the_cli_and_mcp_surfaces_stay_in_sync() {
    use clap::CommandFactory;

    let cli: BTreeSet<String> = nguo_cli::cli::Cli::command()
        .get_subcommands()
        .map(|c| c.get_name().to_string())
        .collect();
    let mcp: BTreeSet<String> = NguoServer::tools()
        .iter()
        .map(|t| t.name.to_string())
        .collect();

    let expected_cli: BTreeSet<String> =
        nguo_cli::ops::SPECS.iter().map(|s| s.cli_name()).collect();
    assert_eq!(
        cli, expected_cli,
        "the CLI has subcommands that are not operations (or is missing some)"
    );

    let expected_mcp: BTreeSet<String> = nguo_cli::ops::SPECS
        .iter()
        .map(|s| s.name.to_string())
        .chain(MCP_ONLY_TOOLS.iter().map(|s| s.to_string()))
        .collect();
    assert_eq!(
        mcp, expected_mcp,
        "the MCP server exposes tools that are neither operations nor declared MCP-only"
    );

    // And the two agree op-for-op, modulo the naming convention and the
    // explicitly declared MCP-only tools.
    let mcp_ops: BTreeSet<String> = mcp
        .iter()
        .filter(|n| !MCP_ONLY_TOOLS.contains(&n.as_str()))
        .map(|n| n.replace('_', "-"))
        .collect();
    assert_eq!(
        cli, mcp_ops,
        "one front end gained an operation the other lacks"
    );
}
