# LICENSES/

<!-- SPDX-License-Identifier: GPL-2.0-or-later -->

SPDX license texts for this project and its third-party dependencies.

## Upstream attribution — ZOZO `ppf-contact-solver` (Apache-2.0)

The solver in this repository is a **port of ZOZO's
[`st-tech/ppf-contact-solver`](https://github.com/st-tech/ppf-contact-solver)**,
which is licensed under the **Apache License, Version 2.0**. That attribution
is a licence obligation, not a courtesy: it must be preserved in any
redistribution of this work, in source or binary form.

Apache-2.0 is compatible with GPLv3 but not with GPLv2 alone, so the combined
work is conveyed under the GPL and this repository is
**GPL-2.0-or-later** rather than GPLv2-only. The Apache-2.0 licence text and
NOTICE conditions apply to the ported portions; the GPL applies to the combined
work. Do not remove, reword or relocate this notice.

- `GPL-2.0-or-later.txt` — the verbatim GNU GPL v2 text (the project is
  GPL-2.0-**or-later**; v2 is the base license text, the "or later" is the grant in
  each file's SPDX header + the `LICENSE` notice). Copied verbatim from the canonical
  copy shipped with the Rust toolchain (`rust/licenses/GPL-2.0-only.txt`).
  (https://www.gnu.org/licenses/old-licenses/gpl-2.0.txt). This is the project
  license.
- Third-party notices (all verified in our allowed set) collected as added;
  `cargo about` generates the full attribution list (`../about.toml`), and
  `make licenses` runs `cargo-deny` over the allow-list on every CI run. Key
  deps and their licenses (verified via `cargo metadata`):
  - wgpu → Apache-2.0/MIT · glam → MIT/Apache-2.0 · serde/serde_json → MIT/Apache-2.0
  - gltf, gltf-json, gltf-derive → MIT/Apache-2.0 · byteorder → Unlicense/MIT
  - roxmltree → MIT/Apache-2.0 · memchr (roxmltree's only dep) → Unlicense/MIT
  - fbxcel → MIT/Apache-2.0 · its deps libflate/libflate_lz77/rle-decode-fast → MIT
    (rle → MIT/Apache) · adler32 → Zlib (permissive, allowed)
  - upstream ppf-contact-solver (ported reference) → Apache-2.0

Every source file carries an `SPDX-License-Identifier: GPL-2.0-or-later` header.
Do not add a dependency whose license is GPLv2-only, non-commercial, or
research-only — see the rationale in `../LICENSE`.
