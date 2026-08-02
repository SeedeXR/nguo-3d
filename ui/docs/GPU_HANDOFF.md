<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# GPU_HANDOFF.md — two devices, one adapter, and the cost of the seam

Implementation: `ui/crates/nguo-gpu/src/handoff.rs`.
Benchmark: `ui/crates/nguo-gpu/tests/handoff_bench.rs`.

```
cargo test -p nguo-gpu --release --test handoff_bench -- --ignored --nocapture
```

---

## 1. The measurement first

Every number below was produced by the command above on **Apple M4 · Metal ·
IntegratedGpu** (unified memory), release profile, 10 untimed warmup iterations
followed by 100 timed iterations per cell, reported as **median / p95**. MB =
10<sup>6</sup> bytes. Frame budget:
**8.3 ms target (120 fps), 16.6 ms hard ceiling (60 fps)**.

### 1.1 Blocking path — positions only (12 B/vertex)

| vertices | payload | readback med/p95 ms | upload med/p95 ms | round trip med/p95 ms | round trip MB/s | vs frame budget |
|---:|---:|---:|---:|---:|---:|:--|
| 10 000 | 0.12 MB | 1.269 / 1.317 | 1.286 / 1.341 | 2.554 / 2.657 | 94 | within target |
| 50 000 | 0.60 MB | 1.358 / 1.389 | 1.525 / 2.685 | 2.865 / 4.035 | 419 | within target |
| 100 000 | 1.20 MB | 1.369 / 2.677 | 1.550 / 2.720 | 2.930 / 4.475 | 819 | within target |
| 250 000 | 3.00 MB | 1.444 / 2.851 | 2.080 / 4.397 | 3.886 / 5.869 | 1544 | within target |
| 500 000 | 6.00 MB | 1.778 / 4.312 | 3.360 / 7.079 | 5.668 / 8.890 | 2117 | median within target, p95 over |

### 1.2 Blocking path — the real payload: positions + normals + strain (28 B/vertex)

| vertices | payload | readback med/p95 ms | upload med/p95 ms | round trip med/p95 ms | round trip MB/s | vs frame budget |
|---:|---:|---:|---:|---:|---:|:--|
| 10 000 | 0.28 MB | 1.351 / 2.658 | 1.466 / 2.769 | 2.846 / 4.279 | 197 | within target |
| 50 000 | 1.40 MB | 1.431 / 3.841 | 1.786 / 4.098 | 3.234 / 5.764 | 866 | within target |
| 100 000 | 2.80 MB | 1.456 / 2.803 | 1.872 / 3.136 | 3.353 / 4.901 | 1670 | within target |
| 250 000 | 7.00 MB | 1.728 / 3.016 | 2.714 / 4.219 | 4.617 / 6.683 | 3032 | within target |
| 500 000 | 14.00 MB | 2.235 / 3.484 | 5.711 / 6.939 | **8.181 / 9.373** | 3422 | **eats the whole 8.3 ms budget** |

### 1.3 Pipelined path — the shipped `Handoff`, 3 slots, 28 B/vertex

| vertices | payload | ms per delivered frame | max sustainable fps | UI-thread med/p95 ms | vs frame budget |
|---:|---:|---:|---:|---:|:--|
| 10 000 | 0.28 MB | 0.215 | 4646 | 0.050 / 0.105 | within target |
| 50 000 | 1.40 MB | 0.236 | 4235 | 0.160 / 0.221 | within target |
| 100 000 | 2.80 MB | 0.501 | 1998 | 0.343 / 0.583 | within target |
| 250 000 | 7.00 MB | 1.410 | 709 | 1.133 / 2.298 | within target |
| 500 000 | 14.00 MB | 2.319 | 431 | 2.260 / 2.731 | within target |

### 1.4 Run-to-run spread — reported, not hidden

The benchmark was run four times. The blocking path at 500 k vertices / 28 B is
**not stable**; the pipelined path is:

| run | 500 k blocking round trip, median | 500 k pipelined, ms/frame |
|---|---:|---:|
| 1 (immediately after a 46 s compile) | 9.420 | 1.994 |
| 2 | 4.644 | 2.214 |
| 3 | 4.647 | 2.035 |
| 4 (the table above) | 8.181 | 2.319 |

The blocking number swings **4.6 – 9.4 ms** — it straddles the 8.3 ms target
depending on what else the machine is doing. The pipelined number stayed inside
**2.0 – 2.3 ms** across all four. That spread is the actual argument for
pipelining, and it is a stronger one than any single clean median.

### 1.5 Reading the numbers

- **There is a ~1.3–1.8 ms floor** on the blocking path regardless of size —
  0.12 MB costs 1.27 ms to read back. That is not bandwidth, it is one
  submit-and-wait round trip through the queue. It is exactly the cost
  pipelining removes, and it is why the 10 k row improves 13× (2.85 → 0.215 ms)
  even though it moves almost no data.
- **Pipelining removes the wait, not the copy.** At 500 k the UI thread still
  pays 2.26 ms median inside `pump` — the `write_buffer` memcpy of 14 MB. That
  is 27 % of the 8.3 ms budget and it is real. It is also the floor for *any*
  design that shows the drape at all.
- **Traffic is decoupled from frame rate.** The solver produces drape frames at
  its own rate; the UI re-renders the last completed one at 120 fps and only
  pays the upload cost on the frames where a new drape actually landed.
- **Not measured: discrete-GPU PCIe behaviour.** There is no discrete GPU on
  this machine. No number is extrapolated for it. What can be said without
  measuring is structural: on PCIe the *latency* of a readback grows and the
  triple buffering is what keeps that latency off the UI thread; the throughput
  ceiling would have to be re-measured on such a machine before any claim is
  made about it.

---

## 2. Why two devices — the maintainer's reasoning

### 2.1 What forced the question

`iced_wgpu-0.14.0/src/window/compositor.rs:151` creates iced's device with
`required_limits: wgpu::Limits::default()` (plus `max_bind_groups: 2`,
`max_non_sampler_bindings: 2048`) and `required_features: Features::empty()`.
`Compositor::request` takes no limits and no device parameter, and the
compositor's fields (`:11-18`) are private. **There is no injection point.**

`wgpu::Limits::default().max_storage_buffers_per_shader_stage` is **8**
(`wgpu-types-27.0.1/src/lib.rs:794`).
`core-engine/crates/engine-gpu/src/shaders/unified_gather.wgsl` declares **17**
storage buffers (`pcg_ops.wgsl` 15, `schwarz.wgsl` 14). The solver's pipelines
therefore **cannot be created** on iced's device. Not "would be slower" —
cannot be created, and the failure lands at pipeline-creation time, not startup.

All four *renderers* — iced, Vello, egui and the 3D viewport — do already share
iced's single device, verified running (`nguo-gpu/src/shared.rs`, `adopt`).
The one-device rule for the renderers holds unchanged.

### 2.2 The decision, and why it is not a performance decision

**Architecture of record: two devices, one `Instance`, one `Adapter`** — the
same physical GPU. An *engine device* created with `adapter.limits()` runs the
solver; *iced's UI device* runs the UI, the Vello canvas and the 3D viewport.

The reason is **not** the benchmark. The benchmark in §1 would have been
survivable either way. The reason is dependency direction:

> If the thing forcing the coupling is internal to one subsystem and the other
> never reads it, the coupling is wrong regardless of the benchmark.

Those 17 storage buffers are solver guts — PCG scratch vectors, contact pairs,
Hessian-apply intermediates. The UI never touches a single one of them and never
will. Making a UI framework's device policy answer to an internal solver detail
would mean that every future change to the solver's bind-group layout becomes a
question about iced, and every iced upgrade becomes a question about the solver.
That is the wrong dependency to own, and no measured millisecond buys it back.

What actually crosses the boundary is small and **output-only**: vertex
positions, normals, and optionally per-vertex strain for the heatmap
(the 3D viewport's strain-map display mode). One struct, 28 bytes, one
direction.

Consequently: **do not fork or patch `iced_wgpu`, and do not try to make the
engine run on iced's device.** Both were considered and rejected for the reason
above, not for a number.

### 2.3 One adapter is the design; two adapters would be a bug

Two devices on one adapter share the same physical GPU and, on unified memory,
the same physical RAM. Two *adapters* would mean cross-GPU copies and a
completely different cost model. So it is asserted, not assumed:

- `handoff::engine_device()` requests its adapter with byte-for-byte iced's own
  options (`PowerPreference::HighPerformance`,
  `iced_wgpu-0.14.0/src/window/compositor.rs:85`) and then calls
  `handoff::assert_one_adapter`, which compares the resulting `AdapterInfo`
  against `nguo_gpu::adapter_description()` — the same enumeration iced's pick
  resolves to.
- `Handoff::new` asserts the converse: the two devices must **not** be the same
  device. If they ever are, either iced grew an injection point (in which case
  this whole crate can be deleted) or the wiring is wrong.

> **ponytail:** the adapter check compares `AdapterInfo` strings rather than
> handles, because iced 0.14 keeps its `Adapter` private (`compositor.rs:13`)
> and `wgpu::Device` exposes no adapter handle. Exact on a single-GPU machine.
> On a dual-GPU laptop, take the real handle from a patched compositor instead.
> Same known ceiling as `shared.rs::adapter_description`, which it reuses.

---

## 3. The pipelining design

```text
engine device                                   UI device
─────────────                                   ─────────
solver output buffer (COPY_SRC)
      │  begin(epoch, src, n)
      │    copy_buffer_to_buffer  ─────┐
      │    engine_queue.submit         │
      │    map_async(Read, cb)         │  slot 0 ┐
      ▼                                ▼  slot 1 ├ COPY_DST | MAP_READ
   (returns immediately)                  slot 2 ┘
                                            │
   pump()  poll(PollType::Poll) ────────────┘
      │    newest READY slot only
      │    ui_queue.write_buffer(&mapped_range)  ──────►  vertex buffer
      │    unmap, slot → Free                             (VERTEX | COPY_DST)
      ▼
   returns Some(epoch) or None — never waits
```

- **Three slots**, the same depth and the same reasoning as
  `nguo_engine_bridge::FRAME_CHANNEL_CAPACITY`: the solver can be copying into
  one while the UI holds one and one waits.
- **`begin` never blocks.** If all three slots are in flight it returns `false`
  and the frame is dropped — the same choice `bridge.rs:190` makes for preview
  frames, for the same reason: a dropped preview frame is invisible because the
  next one supersedes it.
- **`pump` never blocks.** `PollType::Poll`, never `Wait`. A readback that is
  not done yet stays in flight and the UI redraws the previous drape.
- **No intermediate `Vec`.** `write_buffer` is fed the mapped range directly.
- **`drain()` does block**, on purpose, and is for tests and shutdown only.

### Epoch scheme — the bridge's, not a second one

`handoff` reuses `nguo_engine_bridge::Epoch` as the type, and mirrors the
bridge worker's two-point staleness check exactly:

| bridge | handoff | what it drops |
|---|---|---|
| `bridge.rs:127` — before starting a solve | `begin` — before any GPU work | an epoch already superseded |
| `bridge.rs:145` — before emitting a result | `pump` — before uploading | a result that completed but is now behind |

`Epoch(0)` means "none", matching `EngineBridge::latest_epoch`, whose first
handed-out epoch is 1.

### The strain slot

`DrapeVertex` is `{ position: [f32;3], normal: [f32;3], strain: f32 }` = 28 B.
`strain` is the slot for the 3D viewport's strain-map display mode (a
tension/compression heatmap): the solver fills it when the mode is on and leaves it `0.0` otherwise. One
payload shape, no second code path, no branch. **The shader that colours it is
not built** — this is the slot, not the feature.

---

## 4. Invariants

1. **Never render a half-written frame.** A slot's staging buffer is written by
   exactly one in-flight copy, and the UI vertex buffer is filled by a single
   `write_buffer` that is queue-ordered before the draw that reads it
   (`DATAFLOW.md` §8: uploads are pass-order step 1).
2. **A slot in flight is never handed out again.** `begin` only takes a `Free`
   slot, so a readback can never overwrite a buffer the GPU is still copying
   into or the UI is still uploading from.
3. **Stale epochs are dropped, never displayed.** Checked in `begin` and again
   in `pump`; when several readbacks complete in the same pump, only the newest
   is uploaded and the rest are unmapped and discarded.
4. **The UI thread never blocks on the GPU.** `pump` polls.
5. **One `Instance`, one `Adapter`, two devices** — asserted both ways (§2.3).
6. **Data goes one way.** Nothing is read back from the UI device in production;
   the only code that does is the round-trip test, which is what makes the test
   possible at all (and why the UI vertex buffer carries `COPY_SRC`).

## 5. Tests

`cargo test -p nguo-gpu` — 13 pass, 1 ignored (the benchmark). The handoff ones:

| test | proves |
|---|---|
| `round_trip_preserves_every_byte` | a known pattern arrives on the UI device byte-identical |
| `determinism_same_input_same_bytes` | same input → identical bytes, three runs |
| `rotation_never_reuses_a_slot_in_flight` | invariant 2: a 4th `begin` with 3 in flight is refused, and the slots all come back |
| `stale_epoch_is_refused_at_submit` | invariant 3 at submit: older and repeated epochs are refused, and the newest data is what lands |
| `only_the_newest_completed_frame_is_displayed` | invariant 3 at upload: 3 successful readbacks, only epoch 3 displayed |
| `pump_is_a_no_op_when_idle` | invariant 4: nothing ready → `None`, no wait |
| `iced_limits_cannot_host_the_solver` | the premise: 8 < 17, checked rather than asserted in prose |

They skip with a message rather than failing when no GPU adapter is present.
