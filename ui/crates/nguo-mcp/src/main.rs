// SPDX-License-Identifier: GPL-2.0-or-later
//! `nguo-mcp` — MCP server (stdio transport) for the nguo-3d cloth suite.
//!
//! stdio carries the JSON-RPC protocol, so **all** logging goes to stderr;
//! writing anything else to stdout would corrupt the stream.
//!
//! Register with Claude Code:
//! ```text
//! claude mcp add nguo -- /path/to/nguo-mcp
//! ```

use rmcp::{transport::stdio, ServiceExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let server = nguo_mcp::NguoServer::default();
    tracing::info!(
        tools = nguo_mcp::NguoServer::tools().len(),
        "nguo-mcp serving on stdio"
    );
    let service = server.serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}
