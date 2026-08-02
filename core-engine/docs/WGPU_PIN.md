<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# wgpu version pin — 27

**Decision of record (2026-08-01): the whole product is pinned to `wgpu 27`.**
`core-engine` was ported 30 → 27; the UI stack is built against 27 from the start.

## Why 27

The product-wide rule is a **single** `wgpu::Device` shared by the 3D viewport,
iced, Vello and egui. wgpu majors are mutually incompatible — two majors in one
process means two devices, which is exactly what the rule forbids. So the pin is
whatever the four renderers can *simultaneously* satisfy:

| crate | version we use | requires | latest available | why not latest |
|---|---|---|---|---|
| `iced_wgpu` | 0.14.0 | `wgpu ^27.0` | 0.14.0 | already latest; caps everything at 27 |
| `vello` | **0.7.0** | `wgpu ^27.0.1` | 0.9.0 (`wgpu ^29.0.3`) | 0.9 needs 29, incompatible with iced 0.14 |
| `egui-wgpu` | **0.33.3** | `wgpu ^27.0.1` | 0.35.0 (`wgpu ^29.0`) | 0.35 needs 29 |
| `engine-gpu` | this port | `wgpu 27` | was `wgpu 30` | ported down; see deltas below |

`iced_wgpu` is the binding constraint. **27 is the only published set that
unifies all four.**

Resolved lockfile (`core-engine/Cargo.lock`):
`wgpu 27.0.1`, `wgpu-core 27.0.3`, `wgpu-hal 27.0.4`, `wgpu-types 27.0.1`,
`naga 27.0.3`.

## The rule

**All four renderers move together.** Nobody bumps `wgpu` in
`core-engine/Cargo.toml` or `ui/Cargo.toml` alone. The next move is to 29 (or
whatever the *next* version all four support is) and it happens in one change
that bumps engine + iced + vello + egui-wgpu at once, re-runs the engine-gpu
GPU-vs-CPU oracle tests, and updates this table.

## API deltas fixed in the 30 → 27 port

`cargo check --workspace` on 27 produced **46 errors in 5 distinct categories**,
all in `crates/engine-gpu`. Every replacement signature below was read from the
installed registry source, not from memory.

### 1. `PipelineLayoutDescriptor.bind_group_layouts` — 14 sites

Read from
`~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/wgpu-27.0.1/src/api/pipeline_layout.rs:33-45`.

```rust
// wgpu 30
bind_group_layouts: &[Some(&layout)],   // &[Option<&BindGroupLayout>]
// wgpu 27
bind_group_layouts: &[&layout],         // &'a [&'a BindGroupLayout]
```

30 made bind group layout slots sparse (`Option`); 27 has no such concept.
Our pipelines all use a single dense set 0, so the fix is a pure unwrap.

### 2. `PipelineLayoutDescriptor.immediate_size` → `push_constant_ranges` — 14 sites

Same file, same struct. 27 has no `immediate_size` field; the field list is
exactly `{ label, bind_group_layouts, push_constant_ranges }`.

```rust
// wgpu 30
immediate_size: 0,
// wgpu 27
push_constant_ranges: &[],
```

wgpu 30 renamed push constants to "immediates". We use neither, so the empty
slice is exact.

### 3. `BufferSlice::get_mapped_range()` no longer returns `Result` — 17 sites

Read from `wgpu-27.0.1/src/api/buffer.rs:598` — `pub fn get_mapped_range(&self)
-> BufferView` (and `:419` for the `Buffer` convenience method).

```rust
// wgpu 30
let data = slice.get_mapped_range().map_err(|e| GpuError::Readback(e.to_string()))?;
let data = slice.get_mapped_range().unwrap();
// wgpu 27
let data = slice.get_mapped_range();
```

30 turned map-range validation into a returned error; 27 panics on misuse
instead. Behaviour on the happy path is identical. `GpuError::Readback` is still
used — `map_async`'s callback result and the mpsc `recv()` still produce it.

### 4. `InstanceDescriptor::new_without_display_handle_from_env()` — 1 site

`crates/engine-gpu/src/lib.rs:88`. The 27 constructor is
`InstanceDescriptor::from_env_or_default()`, read from
`wgpu-types-27.0.1/src/instance.rs:42-58`. `Instance::new` takes a **reference**
in 27 (`wgpu-27.0.1/src/api/instance.rs:63`: `pub fn new(desc: &InstanceDescriptor) -> Self`).

```rust
// wgpu 30
wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env())
// wgpu 27
wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default())
```

`WGPU_BACKEND` is still honoured: `from_env_or_default()` → `with_env()` →
`Backends::with_env()`, which reads `WGPU_BACKEND`
(`wgpu-types-27.0.1/src/lib.rs:301-310`). The cross-backend CI matrix is
unaffected. The `without_display_handle` distinction is a 30-only surface-setup
concern and irrelevant to this headless compute engine.

### 5. Nothing else moved

These all compiled **unchanged** on 27 and needed zero edits, contrary to the
pre-port guess: `wgpu::PollType::wait_indefinitely()` (17 sites),
`wgpu::MapMode::Read` (17), `wgpu::Trace::Off`, `wgpu::ExperimentalFeatures`,
`wgpu::MemoryHints::Performance`, `DeviceDescriptor { experimental_features,
memory_hints, trace, .. }`, `adapter.request_device(&desc)` returning `Result`,
`request_adapter` returning `Result`, `copy_buffer_to_buffer`, and the entire
`Buffer` / `BufferUsages` / `BindGroup*` / `ComputePipeline` / `ShaderModule` /
`util::DeviceExt` surface.

## WGSL / naga

**No shader changes were needed.** All 24 files in
`crates/engine-gpu/src/shaders/` (2389 lines) that are actually compiled pass
naga 27.0.3 validation — proven by the 39 engine-gpu tests, which build real
Metal pipelines from every one of them.

The single exception is `subgroup_reduce.wgsl`, and it was *already* dead code
before this port:

- `wgpu::Features::SUBGROUP` exists in 27 (`wgpu-types-27.0.1/src/features.rs:1070`),
  and the Apple M4 / Metal adapter **does** expose it — measured.
- But naga 27.0.3 still classifies `subgroups` as
  `UnimplementedEnableExtension::Subgroups`
  (`naga-27.0.3/src/front/wgsl/parse/directive/enable_extension.rs:84,139`).
  Forcing `SUBGROUP_WGSL_SUPPORTED = true` and running the reduce test gives:
  `Shader 'reduce' parsing error: the 'subgroups' enable-extension is not yet supported`.
- This is **identical to the wgpu 30 behaviour** the code already documented, so
  `reduce.rs:21 const SUBGROUP_WGSL_SUPPORTED: bool = false` stays false and the
  Kahan tree-reduce fallback (`tree_reduce.wgsl`) remains the shipped path. Not
  a regression; not caused by the downgrade.

`Features::TIMESTAMP_QUERY` also still exists in 27
(`wgpu-types-27.0.1/src/features.rs:1389`), though note the engine does not
currently gate on it anywhere — grep finds no use outside comments.

## Sharing the device with the UI

The pin exists so one `wgpu::Device` can back the engine *and* iced *and* Vello
*and* egui. `ui/crates/nguo-gpu` owns that device (it must outlive the renderers
built against it); the engine **joins** it:

```rust
// in nguo-gpu, creating the one shared device
let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
    label: Some("nguo shared device"),
    required_features: GpuContext::required_features(&adapter),  // <- use these
    required_limits:   GpuContext::required_limits(&adapter),    // <- not your own
    experimental_features: Default::default(),
    memory_hints: wgpu::MemoryHints::Performance,
    trace: wgpu::Trace::Off,
}).await?;

let engine = GpuContext::from_device(&adapter, &device, &queue);
// device/queue/adapter are Arc-backed; nguo-gpu keeps its own handles.
```

Two requirements, both enforceable by calling the helpers instead of restating
the rules:

- **`required_limits`** must be `adapter.limits()`, **not**
  `Limits::downlevel_defaults()`. The solver's SpMV binds >4 storage buffers and
  downlevel caps `max_storage_buffers_per_shader_stage` at 4. This fails at
  *pipeline creation*, not startup — a nasty debug.
- **`required_features`** must be at least `adapter.features() & Features::SUBGROUP`.
  Fewer is not fatal but silently costs the subgroup reduction fast-path.

`GpuContext` has **no `instance` field** (removed in this port; it was never read
anywhere in the tree). It is not needed for lifetime: `CoreAdapter` and
`CoreDevice` each hold their own `Arc<Global>`
(`wgpu-27.0.1/src/backend/wgpu_core.rs:41,458-469`) and `ContextWgpuCore::drop`
is a no-op, so dropping the `Instance` cannot invalidate a live device. `nguo-gpu`
may hold or drop its instance as it likes.

`has_subgroups` is read from `device.features()`, not `adapter.features()` — what
matters is the feature actually enabled on the device you were handed.

## Verification of the port (run on wgpu 27, macOS / Metal / Apple M4)

```
cargo check --workspace --all-targets     0 errors, 0 warnings
cargo test -p engine-gpu --no-fail-fast   40 passed; 0 failed; 1 ignored
cargo test -p engine-core --no-fail-fast  215 passed; 0 failed; 1 ignored
cargo clippy -p engine-gpu --all-targets -- -D warnings   clean
cargo fmt --all --check                   clean
cargo run -p engine-cli -- gpu-info       Metal / Apple M4 / noop_saxpy_ok: true
```

39/39 of the pre-existing tests is the same count that passed on wgpu 30 before
the port; the 40th is the new `from_device_runs_on_a_caller_owned_device`, which
builds a device the way `nguo-gpu` will and solves a PCG system through it. The GPU-vs-CPU
oracle tests (`gpu_*_matches_cpu`, `gpu_resident_matches_cpu_implicit`,
`gpu_unified_step_matches_cpu`, `pcg_solves_known_system`, …) all still pass at
their original tolerances — **no numerical drift from the downgrade.**
