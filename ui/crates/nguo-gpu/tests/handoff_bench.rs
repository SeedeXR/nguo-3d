// SPDX-License-Identifier: GPL-2.0-or-later
//! Measured cost of the engine-device → UI-device drape handoff.
//!
//! Not a throwaway script — re-run it with:
//!
//! ```text
//! cargo test -p nguo-gpu --release --test handoff_bench -- --ignored --nocapture
//! ```
//!
//! Method (this is what makes the numbers trustworthy):
//! * two devices on **one** `Instance` and **one** `Adapter`, the engine one
//!   with `adapter.limits()` and the UI one with iced's exact hardcoded limits
//!   (`nguo_gpu::handoff::iced_limits`);
//! * [`WARMUP`] untimed iterations before every measurement, so first-touch
//!   page faults and driver lazy-init are not in the sample;
//! * [`ITERS`] timed iterations, reported as **median and p95**, never a mean
//!   and never a single run;
//! * the blocking path and the triple-buffered pipelined path are measured
//!   separately, because the design claim is that pipelining hides the latency
//!   and a claim like that has to be falsifiable.
//!
//! Not measured, and deliberately not extrapolated: discrete-GPU PCIe
//! behaviour. This machine has no discrete GPU (Apple M4, Metal, unified
//! memory), so any number for it would be invented. The pipelined path is what
//! covers that case; its *latency* would grow on PCIe, its *throughput* is what
//! the triple buffering exists to keep off the UI thread.

use std::time::{Duration, Instant};

use nguo_engine_bridge::Epoch;
use nguo_gpu::handoff::{engine_device, iced_limits, DrapeVertex, Handoff, SLOTS, VERTEX_SIZE};

/// Garment complexity we actually ship against.
const COUNTS: [u32; 5] = [10_000, 50_000, 100_000, 250_000, 500_000];
const WARMUP: usize = 10;
const ITERS: usize = 100;
/// Delivered frames per pipelined measurement.
const PIPELINED_FRAMES: usize = 100;

/// Frame budget: 8.3 ms target (120 fps), 16.6 ms hard ceiling (60 fps).
const TARGET_MS: f64 = 8.3;
const CEILING_MS: f64 = 16.6;

/// Positions only — the floor of what has to cross.
const POSITIONS_ONLY: u64 = 12;

struct Rig {
    engine: wgpu::Device,
    engine_queue: wgpu::Queue,
    ui: wgpu::Device,
    ui_queue: wgpu::Queue,
    adapter: String,
}

impl Rig {
    fn new() -> Option<Self> {
        let (adapter, engine, engine_queue) = engine_device()?;
        let info = adapter.get_info();
        // One instance, one adapter, two devices: `ui` is requested from the
        // very same `Adapter` value the engine device came from, so "same
        // adapter" is true by construction here rather than by assertion.
        let (ui, ui_queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("iced-like UI device"),
            required_features: wgpu::Features::empty(),
            required_limits: iced_limits(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        }))
        .ok()?;
        Some(Self {
            engine,
            engine_queue,
            ui,
            ui_queue,
            adapter: format!(
                "{} · {:?} · {:?}",
                info.name, info.backend, info.device_type
            ),
        })
    }
}

/// median and p95, in milliseconds.
fn stats(mut samples: Vec<Duration>) -> (f64, f64) {
    samples.sort_unstable();
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let n = samples.len();
    (ms(samples[n / 2]), ms(samples[(n * 95 / 100).min(n - 1)]))
}

/// MB/s, MB = 10^6 bytes.
fn mbps(bytes: u64, ms: f64) -> f64 {
    (bytes as f64 / 1e6) / (ms / 1e3)
}

/// Blocking readback: copy → submit → map_async → **wait** → map → memcpy →
/// unmap. Everything is inside the timed region, including the memcpy out,
/// because that is what a caller actually pays.
fn readback_blocking(rig: &Rig, src: &wgpu::Buffer, staging: &wgpu::Buffer, out: &mut [u8]) {
    let bytes = out.len() as u64;
    let mut encoder = rig
        .engine
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(src, 0, staging, 0, bytes);
    rig.engine_queue.submit([encoder.finish()]);

    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(0..bytes)
        .map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
    rig.engine
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    rx.recv().expect("map callback").expect("map ok");

    out.copy_from_slice(&staging.slice(0..bytes).get_mapped_range());
    staging.unmap();
}

/// Upload into a UI-device vertex buffer, waited to completion so the number
/// is the real cost and not just the cost of queueing it.
fn upload(rig: &Rig, dst: &wgpu::Buffer, data: &[u8]) {
    rig.ui_queue.write_buffer(dst, 0, data);
    rig.ui_queue.submit([]);
    rig.ui
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
}

fn buffer(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

struct Row {
    verts: u32,
    bytes: u64,
    read: (f64, f64),
    write: (f64, f64),
    round: (f64, f64),
}

fn measure(rig: &Rig, verts: u32, stride: u64) -> Row {
    let bytes = u64::from(verts) * stride;
    let src = buffer(
        &rig.engine,
        "bench src",
        bytes,
        wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    let staging = buffer(
        &rig.engine,
        "bench staging",
        bytes,
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );
    let dst = buffer(
        &rig.ui,
        "bench ui vertices",
        bytes,
        wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    );
    let mut host = vec![0u8; bytes as usize];
    // Non-zero so nothing can be optimised into a zero-page trick.
    rig.engine_queue
        .write_buffer(&src, 0, &vec![0xA5u8; bytes as usize]);
    rig.engine_queue.submit([]);

    for _ in 0..WARMUP {
        readback_blocking(rig, &src, &staging, &mut host);
        upload(rig, &dst, &host);
    }

    let mut read = Vec::with_capacity(ITERS);
    let mut write = Vec::with_capacity(ITERS);
    let mut round = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        readback_blocking(rig, &src, &staging, &mut host);
        let after_read = t.elapsed();
        let t2 = Instant::now();
        upload(rig, &dst, &host);
        let after_write = t2.elapsed();
        read.push(after_read);
        write.push(after_write);
        round.push(after_read + after_write);
    }

    Row {
        verts,
        bytes,
        read: stats(read),
        write: stats(write),
        round: stats(round),
    }
}

/// Steady-state throughput of the shipped `Handoff`: up to [`SLOTS`] readbacks
/// in flight, `pump` never waiting. Returns
/// `(ms per delivered frame, median UI-thread ms in a delivering pump,
///   p95 UI-thread ms)`.
fn measure_pipelined(rig: &Rig, verts: u32) -> (f64, f64, f64) {
    let bytes = u64::from(verts) * VERTEX_SIZE;
    let src = buffer(
        &rig.engine,
        "bench src",
        bytes,
        wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    rig.engine_queue
        .write_buffer(&src, 0, &vec![0xA5u8; bytes as usize]);
    rig.engine_queue.submit([]);

    let mut handoff = Handoff::new(
        &rig.engine,
        &rig.engine_queue,
        &rig.ui,
        &rig.ui_queue,
        verts,
    );

    let mut epoch = 0u64;
    let mut run = |handoff: &mut Handoff, frames: usize| {
        let mut delivered = 0usize;
        let mut ui_cost = Vec::with_capacity(frames);
        let start = Instant::now();
        while delivered < frames {
            let t = Instant::now();
            if handoff.free_slots() > 0 {
                epoch += 1;
                handoff.begin(Epoch(epoch), &src, verts);
            }
            let got = handoff.pump().is_some();
            let cost = t.elapsed();
            if got {
                delivered += 1;
                ui_cost.push(cost);
            }
            assert!(
                start.elapsed() < Duration::from_secs(60),
                "pipelined handoff made no progress"
            );
        }
        (start.elapsed(), ui_cost)
    };

    run(&mut handoff, WARMUP);
    let (wall, ui_cost) = run(&mut handoff, PIPELINED_FRAMES);
    handoff.drain();

    let (median, p95) = stats(ui_cost);
    (
        wall.as_secs_f64() * 1e3 / PIPELINED_FRAMES as f64,
        median,
        p95,
    )
}

fn verdict(ms: f64) -> &'static str {
    if ms <= TARGET_MS {
        "within 8.3 ms target"
    } else if ms <= CEILING_MS {
        "over target, under 16.6 ms ceiling"
    } else {
        "OVER the 16.6 ms ceiling"
    }
}

#[test]
#[ignore = "benchmark: cargo test -p nguo-gpu --release --test handoff_bench -- --ignored --nocapture"]
fn handoff_cost() {
    let Some(rig) = Rig::new() else {
        eprintln!("no GPU adapter; skipping benchmark");
        return;
    };
    let profile = if cfg!(debug_assertions) {
        "debug (run with --release for the real numbers)"
    } else {
        "release"
    };

    println!("\n## Handoff cost — {}", rig.adapter);
    println!(
        "profile: {profile}; warmup {WARMUP}, {ITERS} timed iterations, median / p95; \
         MB = 10^6 bytes\n"
    );

    for (name, stride) in [
        ("positions only (12 B/vertex)", POSITIONS_ONLY),
        ("positions + normals + strain (28 B/vertex)", VERTEX_SIZE),
    ] {
        println!("### Blocking path — {name}\n");
        println!(
            "| vertices | payload | readback med/p95 ms | upload med/p95 ms | round trip med/p95 ms | round trip MB/s | vs frame budget |"
        );
        println!("|---:|---:|---:|---:|---:|---:|:--|");
        for verts in COUNTS {
            let r = measure(&rig, verts, stride);
            println!(
                "| {} | {:.2} MB | {:.3} / {:.3} | {:.3} / {:.3} | {:.3} / {:.3} | {:.0} | {} |",
                r.verts,
                r.bytes as f64 / 1e6,
                r.read.0,
                r.read.1,
                r.write.0,
                r.write.1,
                r.round.0,
                r.round.1,
                mbps(r.bytes * 2, r.round.0),
                verdict(r.round.0),
            );
        }
        println!();
    }

    println!("### Pipelined path — `Handoff`, {SLOTS} slots, 28 B/vertex\n");
    println!(
        "| vertices | payload | ms per delivered frame | max sustainable fps | UI-thread med/p95 ms | vs frame budget |"
    );
    println!("|---:|---:|---:|---:|---:|:--|");
    for verts in COUNTS {
        let bytes = u64::from(verts) * VERTEX_SIZE;
        let (per_frame, med, p95) = measure_pipelined(&rig, verts);
        println!(
            "| {} | {:.2} MB | {:.3} | {:.0} | {:.3} / {:.3} | {} |",
            verts,
            bytes as f64 / 1e6,
            per_frame,
            1e3 / per_frame,
            med,
            p95,
            verdict(per_frame),
        );
    }

    println!(
        "\nNot measured: discrete-GPU PCIe transfer. There is no discrete GPU on \
         this machine, so no number is extrapolated for it."
    );
    println!(
        "`DrapeVertex` = {VERTEX_SIZE} B ({}). Frame budget: {TARGET_MS} ms target, \
         {CEILING_MS} ms ceiling.\n",
        std::mem::size_of::<DrapeVertex>()
    );
}
