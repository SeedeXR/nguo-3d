#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Wrap the release binaries in a macOS .app bundle, with the app icon and the
# .nguo document association.
#
#   scripts/bundle-macos.sh [outdir]     default: output/
#
# Produces, in that directory (created if absent):
#   nguo.app              the application bundle
#   nguo-<version>.dmg    a compressed disk image for handing to someone
#   icons/                the generated .icns, kept so a rebuild can skip sips
#
# Expects `make build-release` (or scripts/build.sh) to have run first — this
# script BUNDLES, it does not compile. Keeping those separate means a failed
# bundle never looks like a failed build.
#
# `output/` is gitignored on purpose: everything here is derived from the source
# tree plus ui/assets/brand-icon/app-icon.png, and a committed binary artefact
# outlives the source it was built from.
#
# THE .nguo ICON. A file gets its icon from a UTI, and a UTI has to be declared
# by an installed application — there is nowhere else to put it. So:
#   * UTExportedTypeDeclarations declares `ai.seedexr.nguo.project` and binds it
#     to the `nguo` extension. "Exported" because we define the format (D-1).
#   * CFBundleDocumentTypes says this app edits that type, and names the icon.
# Finder then shows nguo-doc.icns on every .nguo once the app has been seen by
# Launch Services — which happens on first run, or via `lsregister`.
#
# The UTI conforms to `public.data`, NOT `public.zip-archive`, even though a
# .nguo IS a zip (D-1). Declaring the zip conformance is truthful but invites
# Archive Utility to claim the file and offer to expand it, which would turn a
# double-click into a folder of loose JSON instead of opening the project.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/output}"
APP="$OUT/nguo.app"
BIN="$ROOT/ui/target/release/nguo-app"
CLI="$ROOT/ui/target/release/nguo"
ICONS="$OUT/icons"

# The one place `output/` is created. `-p` so a caller-supplied nested path works
# and so a second run is a no-op.
mkdir -p "$OUT"

[[ "$(uname -s)" == "Darwin" ]] || { echo "bundle-macos: macOS only; nothing to do." >&2; exit 0; }
[[ -x "$BIN" ]] || { echo "bundle-macos: $BIN not built — run scripts/build.sh first" >&2; exit 1; }

# Version from the workspace manifest, so the bundle can never disagree with
# what cargo built: the first `version = "x"` inside [workspace.package].
#
# The `f=0` on any other `[section]` header is load-bearing. Without it the
# scan stays armed past the end of [workspace.package], so if that key is ever
# moved or renamed, awk happily returns a `version` from some LATER table and
# the bundle ships a confidently wrong CFBundleShortVersionString. Scoped, an
# absent key yields an empty string and the check below fails loudly instead.
VERSION="$(awk '/^\[/{f=0} /^\[workspace\.package\]/{f=1} f && /^version = /{gsub(/[":]/,"",$3); print $3; exit}' "$ROOT/ui/Cargo.toml")"
[[ -n "$VERSION" ]] || { echo "bundle-macos: could not read version from ui/Cargo.toml" >&2; exit 1; }

"$ROOT/scripts/make-icons.sh" "$ICONS"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/nguo-app"
# The CLI rides along so `nguo` and the GUI can never be different builds.
[[ -x "$CLI" ]] && cp "$CLI" "$APP/Contents/MacOS/nguo"
cp "$ICONS/nguo.icns" "$ICONS/nguo-doc.icns" "$APP/Contents/Resources/"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>                  <string>nguo</string>
  <key>CFBundleDisplayName</key>           <string>nguo 3D</string>
  <key>CFBundleExecutable</key>            <string>nguo-app</string>
  <key>CFBundleIdentifier</key>            <string>ai.seedexr.nguo</string>
  <key>CFBundlePackageType</key>           <string>APPL</string>
  <key>CFBundleShortVersionString</key>    <string>${VERSION}</string>
  <key>CFBundleVersion</key>               <string>${VERSION}</string>
  <key>CFBundleIconFile</key>              <string>nguo</string>
  <key>NSHighResolutionCapable</key>       <true/>
  <key>LSMinimumSystemVersion</key>        <string>11.0</string>
  <key>NSHumanReadableCopyright</key>      <string>© Seede XR. GPL-2.0-or-later.</string>

  <!-- What this app opens. Editor, not Viewer: nguo writes .nguo too. -->
  <key>CFBundleDocumentTypes</key>
  <array>
    <dict>
      <key>CFBundleTypeName</key>          <string>nguo project</string>
      <key>CFBundleTypeRole</key>          <string>Editor</string>
      <key>LSHandlerRank</key>             <string>Owner</string>
      <key>CFBundleTypeIconFile</key>      <string>nguo-doc</string>
      <key>LSItemContentTypes</key>
      <array><string>ai.seedexr.nguo.project</string></array>
    </dict>
  </array>

  <!-- We define the format, so the declaration is Exported, not Imported. -->
  <key>UTExportedTypeDeclarations</key>
  <array>
    <dict>
      <key>UTTypeIdentifier</key>          <string>ai.seedexr.nguo.project</string>
      <key>UTTypeDescription</key>         <string>nguo project</string>
      <key>UTTypeIconFile</key>            <string>nguo-doc</string>
      <key>UTTypeConformsTo</key>
      <array><string>public.data</string></array>
      <key>UTTypeTagSpecification</key>
      <dict>
        <key>public.filename-extension</key><array><string>nguo</string></array>
      </dict>
    </dict>
  </array>
</dict>
</plist>
PLIST

plutil -lint "$APP/Contents/Info.plist" >/dev/null || { echo "bundle-macos: generated Info.plist is malformed" >&2; exit 1; }

# Ad-hoc signature. Unsigned bundles are quarantined by Gatekeeper on download;
# this is enough for local use and CI, and is NOT a substitute for a Developer
# ID signature at release. Non-fatal: a build must not fail for want of signing.
codesign --force --deep --sign - "$APP" 2>/dev/null \
  && echo "bundle-macos: ad-hoc signed (not Developer ID — do not ship this to users)" \
  || echo "bundle-macos: WARNING codesign failed; bundle is unsigned"

printf 'bundle-macos: %s (%s)\n' "$APP" "$(du -sh "$APP" | cut -f1 | tr -d ' ')"

# ---------------------------------------------------------------------------
# The disk image.
#
# A .dmg is how a Mac app is handed to someone who is not going to run a build.
# The staging directory holds the app plus a symlink to /Applications, which is
# the whole of the "drag it across" convention — no background image, no
# AppleScript-positioned window. Those need a mounted read-write image and a
# Finder that will talk to you, which is exactly the step that breaks on a
# headless CI runner. This works everywhere and says the same thing.
#
# UDZO is zlib-compressed and read-only: the format every distributed dmg uses.
# `hdiutil` ships with macOS, so this adds no dependency.
#
# NOT notarised, and not Developer ID signed. On another machine Gatekeeper will
# refuse this on first open — right-click → Open, or `xattr -d
# com.apple.quarantine`. Shipping to users needs a real certificate; that is a
# release step, not a build step.
# ---------------------------------------------------------------------------
DMG="$OUT/nguo-${VERSION}.dmg"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/nguo-dmg.XXXXXX")"
# The trap covers every exit path: a failed hdiutil must not leave a staging
# directory (or, worse, a mounted image) behind.
trap 'rm -rf "$STAGE"' EXIT

cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"

rm -f "$DMG"
if hdiutil create \
     -volname "nguo 3D ${VERSION}" \
     -srcfolder "$STAGE" \
     -fs HFS+ \
     -format UDZO \
     -imagekey zlib-level=9 \
     -quiet \
     "$DMG"; then
  printf 'bundle-macos: %s (%s)\n' "$DMG" "$(du -sh "$DMG" | cut -f1 | tr -d ' ')"
  # Prove it mounts and carries the app, rather than trusting a zero exit.
  if hdiutil verify "$DMG" -quiet 2>/dev/null; then
    echo "bundle-macos: dmg checksum verifies"
  else
    echo "bundle-macos: WARNING dmg written but 'hdiutil verify' failed" >&2
  fi
else
  echo "bundle-macos: WARNING dmg creation failed; the .app above is still good" >&2
fi

echo "bundle-macos: to make Finder show the .nguo icon without installing:"
echo "  /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f \"$APP\""
