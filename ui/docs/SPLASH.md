<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# The splash screen (`nguo-splash`)

The first thing anyone ever sees of nguo-3d. Two panels, 1.685:1, split exactly
50/50: a black plate on the **left** carrying the wordmark, the butterfly mark
and a block of small grey type; the artwork filling the **right** edge to edge.

Reference render: `nguo-3d splash-screen.png`, 752x446, held in the
maintainer's local reference set and not distributed with this source. The
split was measured on it at x=376, i.e. 0.5000.

---

## 1. Integration — what `nguo-app` calls

```rust
use nguo_splash::Splash;

// 1. Register the vendored font ONCE, on the application builder.
iced::application(boot, update, view)
    .font(nguo_splash::INTER_TTF)      // required, or the type falls back
    .window_size(nguo_splash::SIZE)    // 900 x 534, the designed ratio
    // ...

// 2. Own one in your state.
struct App { splash: Option<Splash>, .. }
//                  ^ `None` once dismissed; that is your "splash is gone" flag.
App { splash: Some(Splash::new(Instant::now())), .. }

// 3. Drive it.
Splash::ticks().map(Message::Tick)     // subscription; window frames, no timer dep
```

### The window

A **fixed-size, centred, undecorated panel over the desktop** — Photoshop /
Lightroom, never a fullscreen takeover.

```rust
.window_size(nguo_splash::SIZE)                     // 900 x 534 logical pt
.resizable(false)
.decorations(false)
.level(iced::window::Level::AlwaysOnTop)
.centered()
```

`SIZE` is **logical points**, 900 x 534 = 1.685:1 — the reference render's
752x446 (1.686:1) scaled up to a comfortable size and kept in the 760–1000 pt
band. On a 2x display the surface is 1800x1068 physical while the layout stays
in the 900x534 logical space; iced handles that itself and nothing in this
crate does DPI arithmetic. No transparency and no rounded corners: the plate is
a hard-edged rectangle, so neither buys anything.

| Call | When | Notes |
|---|---|---|
| `Splash::new(now)` | at startup | `now` is injected, never read inside |
| `splash.report(stage, fraction)` | on each **real** milestone | monotonic, clamped `< 1.0` |
| `splash.loaded()` | when init is genuinely finished | sets 1.0; does **not** dismiss |
| `splash.tick(now) -> bool` | every frame | `true` on the tick that dismisses |
| `splash.view()` | in `view` | display only; nothing is clickable |
| `Splash::ticks()` | in `subscription` | `Subscription<Instant>` |
| `Splash::MIN_VISIBLE` | — | `Duration::from_secs(4)` |
| `Splash::SIZE` | — | `900.0 x 534.0` |
| `Splash::INTER_TTF` / `INTER` | — | font bytes and the `iced::Font` to match |

Wiring, in full:

```rust
Message::Tick(now) => {
    if let Some(splash) = &mut self.splash {
        if splash.tick(now) { self.splash = None; }
    }
}
Message::GpuReady   => report("GPU device", 0.20),
Message::Shaders    => report("Shaders",    0.45),
// ... theme, fonts, samples ...
Message::Ready      => { splash.loaded(); splash.tick(Instant::now()); }
//                                        ^ dismiss on this frame, not the next
```

Suggested stages, in the order `nguo-app` actually reaches them: **GPU device →
shaders → theme → fonts → samples**. The labels are free text; only report a
stage you have really passed.

---

## 2. Behaviour

Photoshop / Lightroom semantics.

1. **Minimum 4 s.** Even a 200 ms load holds the splash the full four seconds.
   A splash that flashes past reads as a rendering glitch, not as speed.
2. **Longer if loading is not done.** Dismissal is `max(shown_at + 4s,
   load_complete)`.
3. **Nothing else dismisses it.** In particular there is no click-to-dismiss.

### Why there is no click-to-dismiss

There used to be a `click()` and a latch for early clicks. It could not change
any outcome and was deleted rather than left as decoration.

An impatient click at 1 s must not tear down a half-initialised app, so a click
can never *rush* dismissal. And `tick` is driven by `iced::window::frames()`,
so the splash already goes on the first frame after `max(4 s, load)` — a click
cannot *bring that forward* either. The impatient user and the patient one got
identical behaviour, which meant the click handler was doing nothing.

If a click should ever mean something, the only honest version is "skip the
remaining 4 s minimum, but still never before loading is complete". That is a
product decision about the 4 s floor in §2.1, not a bug fix, so it is not made
here.

### Only `tick` dismisses

`report()` and `loaded()` are a data feed. They record what the app achieved
and return nothing. Dismissal — a state transition that tears down a window —
belongs to the frame loop, never to a progress callback that might be invoked
from the middle of the loader's call stack.

The cost: after `loaded()` the splash goes on the *next* frame. If you want it
gone on the same frame, call `tick(Instant::now())` right after (as above).

### Progress is real or it is not shown

`report()` clamps to `[previous, 0.99]`. The bar cannot go backwards and cannot
read 100 % before `loaded()`. There is no timer-driven animation anywhere in
this crate: if the app makes no progress, the line does not move. A fake
progress bar is a lie told to a user who is already waiting.

---

## 3. Copy — corrections that must not be reverted

The reference render contains two errors. Both are fixed in code and must stay
fixed:

| Reference render | Shipped | Why |
|---|---|---|
| "**Fore** more details…" | "**For** more details…" | typo |
| "…the About **Optima** Screen." | "…the About **nguo-3d** screen." | *Optima* is a different product |

Full copy, in order down the left panel:

```
Nguo - 3d                                        (wordmark, stroked SVG)
                                                 (butterfly mark)
Loading … <stage>  ·  <n>%                       (live; "Ready" when done)
Artwork by Alex Mkwizu & MidJourney AI
© 2026 Seede XR. All rights reserved.
For more details and legal notices, go to the About nguo-3d screen.
Built for the community
```

`Built for the community` is required by the maintainer and closes the block.

---

## 4. Typography and colour

- **Inter** for every line of type. Vendored at `ui/assets/fonts/Inter-Regular.ttf`
  (SIL OFL 1.1 — provenance and SHA-256 in `ui/assets/fonts/LICENSE.md`),
  embedded with `include_bytes!`, exposed as `nguo_splash::INTER_TTF`. The shell
  must register it; without that every line silently falls back.
- **One** exception: the **wordmark**, which is an SVG (below) because iced
  cannot stroke glyphs. Everything else — including the closing
  `Built for the community` line — is Inter at the same small size. That line
  is set apart only by a slightly dimmer grey (`#787878`), never by a different
  face. Do not reach for `Font::DEFAULT` + a weight here either: "sans-serif
  bold" resolves through each machine's fontdb and silently produced a
  *monospace* face on the dev machine. A splash must look the same everywhere,
  so every text widget names `nguo_splash::INTER` explicitly.
- **All type is small and grey** (`#909090`), stacked at the bottom of the
  panel. The wordmark and the butterfly are the only white, and the only
  things large enough to shout. This is the maintainer's call and it overrides
  the reference render, where the credits are white and larger.

### The wordmark is an SVG, on purpose

iced's text widget cannot stroke glyphs, and the wordmark is a hollow outlined
face. It is therefore a small inline SVG (`WORDMARK_SVG` in `src/lib.rs`) with
`fill="none"` and a white stroke: usvg converts `<text>` to glyph outlines, so
the stroke follows the real letterforms. iced's SVG rasteriser loads system
fonts (`iced_wgpu-0.14.0/src/image/vector.rs:56`), so the family list resolves
per platform. Editing the wordmark means editing that string.

### This screen is exempt from the token system

The project rule (`ui/docs/DEVELOPING.md` §4.7) is that nothing outside
`nguo-ui-theme` hardcodes a colour. **The splash is the single documented exception.** Three reasons:

1. It is a fixed brand plate, not a themed surface. It must be black-on-black
   whether the user runs light or dark; following the theme would break it.
2. It renders *before* the theme is necessarily loaded — the theme is one of
   the things the progress line reports on.
3. Its palette comes from the reference render, sampled, not from the scale:
   `#000000` (background, sampled at 5,5), `#ffffff` (wordmark stroke and
   mark), `#909090` (all type, sampled off the reference status line).

That is the whole exception. It does not extend to any other screen, and
"the splash does it" is not an argument for hardcoding a colour anywhere else.

---

## 5. Testing

`cargo test -p nguo-splash` — 8 tests, **no window and no real sleeps**. The
timing logic is pure and the clock is injected: tests build a `t0` once and
pass `t0 + Duration` into `tick`, so the whole suite runs in microseconds.
Covered: dismissal at exactly 4 s after a 0.2 s load; dismissal at
load-complete for a 9 s load (not at 4 s); dismissal on the first tick after
both conditions hold; single dismissal edge; progress monotonicity; never
100 % before completion; a configurable minimum that is still enforced.

### Looking at it

```
cargo run -p nguo-splash --example splash_demo
```

Shows only the splash with a simulated five-stage load that runs 6.5 s — longer
than the minimum on purpose, so the "waits for loading" branch is the one you
see. It prints the dismissal time on exit.

To capture it, prefer the app's **own framebuffer** over `screencapture`:
`iced::window::screenshot(id)` returns the exact rendered RGBA and cannot lose
a race with whatever else owns the front of the screen (on a busy machine,
`screencapture -x` repeatedly caught other apps' windows instead, and a
fullscreen Space belonging to another app hides the splash entirely).

---

## 6. Trap: an image handle rebuilt every frame never renders

`image::Handle::from_bytes` stamps a fresh `Id::unique()` on every call
(`iced_core-0.14.0/src/image.rs:136`) — the cache key is the *handle*, not the
bytes. Raster decoding is asynchronous on a worker thread
(`iced_wgpu-0.14.0/src/image/cache.rs:400`), so a handle constructed inside
`view` is a brand-new cache miss every frame: the decode is queued, the frame
draws nothing, the handle is dropped, repeat forever. The panel stays black and
**nothing errors** — `image::allocate()` on a separate handle even reports
`Ok(1024x1024)`, which is what makes this so slow to find.

`ARTWORK` is therefore a `LazyLock<image::Handle>` built once. SVG handles hash
their data (`iced_core-0.14.0/src/svg.rs:93`) and are immune, which is exactly
why the butterfly rendered while the artwork did not. **Any other crate drawing
a raster image must cache its handle the same way.**
