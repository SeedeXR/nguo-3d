<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# AGENTIC.md — driving nguo-3d from a script or an agent

Two front ends, **one** implementation:

| Front end | Binary | Transport | Who uses it |
|---|---|---|---|
| CLI | `nguo` | argv → JSON on stdout | shells, CI, `make` |
| MCP server | `nguo-mcp` | JSON-RPC over stdio | Claude Code and other agentic tools |

Both are thin shells over `nguo_cli::ops` — one `SPECS` table, one `dispatch`.
A test (`nguo-mcp/tests/mcp_server.rs::the_cli_and_mcp_surfaces_stay_in_sync`)
fails the build if one side gains an operation the other lacks.

Under both sits `nguo-engine-bridge`, which runs the actual solver.

---

## 1. Process contract

Identical to the engine binary's (`core-engine/docs/PROCESS_CONTRACT.md`), on
purpose — one convention, not two.

* **stdout** — exactly one pretty JSON value per invocation. Nothing else ever.
* **stderr** — logs (`tracing`). `--log-format json` for one JSON record per
  line; `-v` / `-vv` or `RUST_LOG` for level.
* **exit codes**

| Code | Meaning |
|---|---|
| 0 | Success. stdout holds the result. |
| 1 | Failure: bad arguments, missing/unreadable file, engine failure, solver diverged. **stdout is empty**; the reason is an `ERROR` log on stderr. |
| 2 | The input was understood and is invalid (structural validation or solver invariants failed). stdout holds the full report — the report *is* the result. Same meaning as `mguso verify`'s exit 2. |

Nothing ever returns 0 on a failed simulation, and no operation fabricates a
success: if the engine fails, its own message is what you get back.

## 2. Discovering the surface

```console
$ nguo schema
```

Returns the machine-readable command surface: every operation's name, CLI
spelling, summary, honest description, `read_only` / `idempotent` flags, and a
JSON Schema for its arguments — plus the process contract above. This is the
same list the MCP `tools/list` response is generated from.

## 3. Operations

CLI subcommands are the operation name with `_` → `-`.

### `schema`
`nguo schema` — the above. No arguments. Read-only.

### `gpu_info`
`nguo gpu-info` — runs `mguso gpu-info`: picks an adapter, runs a no-op SAXPY
compute probe, returns the engine's report. Read-only, idempotent.

```json
{
  "backend": "metal",
  "adapter": "Apple M-series",
  "device_type": "IntegratedGpu",
  "noop_saxpy_ok": true
}
```

### `scene_info`
`nguo scene-info <scene.json>` — parses an `engine-api` Scene JSON document
locally (no solver) and summarises it. Read-only, idempotent.

```json
{
  "scene": "core-engine/examples/sphere_drape.json",
  "config": { "dt": 0.001, "fps": 60.0, "frames": 90, "iterations": 4,
              "gravity": [0.0, -9.8, 0.0], "floor-y": -1.0 },
  "objects": [{ "name": "drape", "verts": 144, "tris": 242, "pins": 0, "is_static": false }],
  "colliders": [{ "type": "sphere" }],
  "totals": { "objects": 1, "verts": 144, "tris": 242 }
}
```

### `scene_validate`
`nguo scene-validate <scene.json> [--solver-invariants] [--steps N]` —
structural checks, always local and fast: objects present, all positions
finite, every triangle and pin index in range, no degenerate triangles,
`dt > 0`, `fps > 0`. With `--solver-invariants` it also runs `mguso verify`,
which *simulates* the scene with the implicit solver and checks the solver's
own invariants — seconds to minutes. Read-only, idempotent.

```json
{
  "scene": "…/fabric_drape.json",
  "ok": true,
  "checks": [
    { "name": "has-objects",               "ok": true, "detail": "1 object(s)" },
    { "name": "positions-finite",          "ok": true, "detail": "0 non-finite vertex position(s)" },
    { "name": "triangle-indices-in-range", "ok": true, "detail": "0 triangle(s) index a vertex that does not exist" },
    { "name": "triangles-non-degenerate",  "ok": true, "detail": "0 triangle(s) repeat a vertex" },
    { "name": "pin-indices-in-range",      "ok": true, "detail": "0 pin(s) index a vertex that does not exist" },
    { "name": "timestep-positive",         "ok": true, "detail": "dt = 0.001" },
    { "name": "fps-positive",              "ok": true, "detail": "fps = 60" }
  ]
}
```

A failing scene prints the same shape with `"ok": false` and **exits 2**.

### `drape`
`nguo drape <scene.json> [--tier preview|bake] [--frames N] [--out DIR]` — runs
the solver through the engine bridge.

* `preview` (default) = XPBD, fast, interactive-grade.
* `bake` = implicit backward Euler, high fidelity, much slower.
* `--out DIR` keeps the per-frame OBJ sequence as `DIR/frame_0000.obj`… — this
  is also how you export a drape.

Deterministic: the same scene and tier give the same drape, so running it twice
is safe. The only side effect is rewriting the frame files in `--out`.

```json
{
  "scene": "…/sphere_drape.json",
  "tier": "preview",
  "frames_streamed": 3,
  "out_dir": "/tmp/frames",
  "engine": {
    "backend": "xpbd",
    "frames": 3,
    "verts": 144,
    "final_time": 0.0505,
    "final_kinetic_energy": 0.0537,
    "bbox_min": [-0.66, 1.2887, -0.66],
    "bbox_max": [0.66, 1.2887, 0.66]
  }
}
```

`engine` is the engine's own stdout summary, verbatim.

## 4. What is deliberately **not** here

`nguo-doc` (the project/pattern document model) does not exist yet, so there is
**no** `project_open`, `piece_list`, `seam_list` or `edit_*` operation. They are
absent rather than stubbed: a command that always fails is worse than a command
that is honestly missing, and a command that pretends to succeed is a bug. When
`nguo-doc` lands, those operations get added to `ops::SPECS` and both front ends
gain them at once.

Likewise there is no cancel for a running drape — see §7.

## 5. MCP server

### Register with Claude Code

```console
cargo build -p nguo-mcp --release          # in ui/
claude mcp add nguo -- /abs/path/to/ui/target/release/nguo-mcp
```

Point it at a specific engine build if needed:

```console
claude mcp add nguo --env NGUO_MGUSO=/abs/path/to/core-engine/target/release/mguso \
  -- /abs/path/to/ui/target/release/nguo-mcp
```

The server speaks stdio, so **stdout is the protocol** — all of its logging
goes to stderr (`RUST_LOG=debug` to see it).

### Tools

Every operation in §3 is an MCP tool under its snake_case name, with the JSON
Schema from `nguo schema` and annotations (`readOnlyHint`, `idempotentHint`,
`destructiveHint: false`, `openWorldHint: false`) taken from the same table.

One tool exists only over MCP, because it only makes sense in a long-lived
server:

### `drape_status`
`{ "job_id": "drape-1" }` → `{ "status": "running" }`, or
`{ "status": "done", "result": { …the §3 drape report… } }`, or a tool error
carrying the engine's real message. Job ids live in the server process only; a
restart forgets them, and polling an unknown id is an error, never an empty
success.

### Why `drape` is asynchronous over MCP

A bake can run for minutes and MCP is request/response — blocking the server
would freeze every other tool. So the MCP `drape` tool returns immediately:

```json
{
  "job_id": "drape-1",
  "status": "running",
  "poll_with": "drape_status",
  "note": "The drape runs in the background. Call drape_status with this job_id until status is \"done\" or \"failed\"."
}
```

The tool's description says so too, because an agent picks tools by their
description. The CLI's `drape` stays synchronous — a CLI process *is* the job.

Each `drape` call starts a new job. The solve is deterministic, so a repeat
gives an identical result; the only effect of calling twice is that the frame
files in `out_dir` are written again.

### Errors

MCP has two failure modes and this server uses both correctly:

* **protocol error** (`-32602 invalid params`) — the request itself is
  malformed: unknown tool, missing/wrongly-typed argument. Checked *before* a
  job is started.
* **tool error** (`isError: true` with text content) — the tool ran and failed:
  missing file, engine failure, solver diverged, validation failed. The content
  is the real error, including the engine's own log tail.

## 6. Worked example — an agent driving a drape end to end

```jsonc
// 1. Learn the surface.
→ tools/call  { "name": "schema", "arguments": {} }
← operations[…], process_contract{…}

// 2. Look at the scene before spending solver time on it.
→ tools/call  { "name": "scene_info",
                "arguments": { "scene": "core-engine/examples/sphere_drape.json" } }
← { "totals": { "objects": 1, "verts": 144, "tris": 242 }, … }

// 3. Check it is sane. (isError:true here means: do not bother draping.)
→ tools/call  { "name": "scene_validate",
                "arguments": { "scene": "core-engine/examples/sphere_drape.json" } }
← { "ok": true, "checks": [ … ] }

// 4. Start the drape. Returns immediately.
→ tools/call  { "name": "drape",
                "arguments": { "scene": "core-engine/examples/sphere_drape.json",
                               "tier": "preview", "frames": 30,
                               "out_dir": "/tmp/drape-out" } }
← { "job_id": "drape-1", "status": "running", "poll_with": "drape_status" }

// 5. Poll.
→ tools/call  { "name": "drape_status", "arguments": { "job_id": "drape-1" } }
← { "job_id": "drape-1", "status": "running" }
→ tools/call  { "name": "drape_status", "arguments": { "job_id": "drape-1" } }
← { "job_id": "drape-1", "status": "done",
    "result": { "frames_streamed": 30,
                "engine": { "frames": 30, "verts": 144, "final_time": 0.5, … } } }

// 6. /tmp/drape-out/frame_0000.obj … frame_0029.obj are the exported drape.
```

The same thing from a shell:

```console
$ nguo scene-validate core-engine/examples/sphere_drape.json && \
  nguo drape core-engine/examples/sphere_drape.json --frames 30 --out /tmp/drape-out
$ echo $?
0
```

## 7. `nguo-engine-bridge` (for UI code, not agents)

The async interface the desktop app uses:

```rust
let (bridge, mut updates) = EngineBridge::new(Arc::new(ProcessTransport::default()));
let epoch = bridge.submit(Submission { scene, tier: Tier::Preview, frames: None, out_dir: None });
while let Some(update) = updates.recv().await { /* Frame / Done / Failed */ }
```

* **Off-thread.** The solve runs on the blocking pool, never on the UI thread.
* **Epochs.** Every job carries a generation id. A newer `submit` supersedes the
  in-flight solve; nothing from a stale epoch is ever delivered — not a frame,
  not even a failure (guide §3.3). Rapid input coalesces to at most one solve in
  flight plus one queued (the newest); the ones in between never start.
* **Frame buffering.** A 3-deep channel of owned frame buffers (guide §3.1) —
  ownership transfer is what makes a torn read impossible. `Preview` drops
  frames rather than stalling the solver; `Bake` back-pressures so every frame
  reaches the export.
* **Determinism.** Same submission → same drape.

### The transport seam

`core-engine` is on `wgpu 30` while this workspace is pinned to `wgpu 27`
(`core-engine/docs/WGPU_PIN.md`). A path dependency on `engine-core` today
would pull two
incompatible wgpu majors into one binary. So the bridge is defined over a
`Transport` trait with exactly one implementation, `ProcessTransport`, which
spawns `mguso run` and reads its streamed frame files. When the wgpu pin
unifies, an in-process transport is a drop-in second impl and nothing above the
trait changes.

`ProcessTransport` finds the engine at `$NGUO_MGUSO`, else the sibling
`core-engine/target/{release,debug}/mguso`, else `mguso` on `PATH`.

**Known ceilings** (all marked `// ponytail:` in the source):

* Frames are discovered by polling the output directory, because `mguso run`
  streams frames as OBJ files and nothing else. Frame *N* is only read once
  *N+1* exists (or the engine has exited), so a half-written file is never
  parsed. Upgrade path: NDJSON frames on a pipe, or the in-process transport.
* A running drape cannot be cancelled through the MCP server. The bridge does
  cancel in-flight solves for the UI (epochs), but a `drape_status` cancel tool
  would have to kill the child process mid-flight, so it is not offered rather
  than offered and dishonest.
