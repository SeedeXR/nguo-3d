#!/usr/bin/env python3
"""Generate a minimal valid rigged glTF binary (.glb): a skinned quad with 2
joints (root->bone1), per-vertex JOINTS_0/WEIGHTS_0, inverse-bind matrices, and
one rotation AnimationClip (2 keyframes). Used as a read-side fixture."""
import struct, json, sys

def f32(vals): return b"".join(struct.pack("<f", v) for v in vals)
def u32(vals): return b"".join(struct.pack("<I", v) for v in vals)
def u16(vals): return b"".join(struct.pack("<H", v) for v in vals)
def ident(): return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]  # column-major identity

positions = [0,0,0, 1,0,0, 0,1,0, 1,1,0]
indices   = [0,1,2, 2,1,3]
joints    = [0,1,0,0]*4          # each vert influenced by joints 0 and 1
weights   = [0.6,0.4,0,0]*4
ibm       = ident()+ident()      # 2 joints
anim_in   = [0.0, 1.0]
anim_out  = [0,0,0,1, 0,0,0.7071068,0.7071068]  # identity, 90deg about z

sections = [
    ("pos",     f32(positions)),
    ("idx",     u32(indices)),
    ("joints",  u16(joints)),
    ("weights", f32(weights)),
    ("ibm",     f32(ibm)),
    ("anim_in", f32(anim_in)),
    ("anim_out",f32(anim_out)),
]
bin_blob = b""
offs = {}
for name, data in sections:
    while len(bin_blob) % 4: bin_blob += b"\x00"
    offs[name] = len(bin_blob)
    bin_blob += data
while len(bin_blob) % 4: bin_blob += b"\x00"

doc = {
  "asset": {"version": "2.0", "generator": "gen_rigged_glb.py"},
  "buffers": [{"byteLength": len(bin_blob)}],
  "bufferViews": [
    {"buffer":0,"byteOffset":offs["pos"],"byteLength":48,"target":34962},
    {"buffer":0,"byteOffset":offs["idx"],"byteLength":24,"target":34963},
    {"buffer":0,"byteOffset":offs["joints"],"byteLength":32,"target":34962},
    {"buffer":0,"byteOffset":offs["weights"],"byteLength":64,"target":34962},
    {"buffer":0,"byteOffset":offs["ibm"],"byteLength":128},
    {"buffer":0,"byteOffset":offs["anim_in"],"byteLength":8},
    {"buffer":0,"byteOffset":offs["anim_out"],"byteLength":32},
  ],
  "accessors": [
    {"bufferView":0,"componentType":5126,"count":4,"type":"VEC3","min":[0,0,0],"max":[1,1,0]},
    {"bufferView":1,"componentType":5125,"count":6,"type":"SCALAR"},
    {"bufferView":2,"componentType":5123,"count":4,"type":"VEC4"},
    {"bufferView":3,"componentType":5126,"count":4,"type":"VEC4"},
    {"bufferView":4,"componentType":5126,"count":2,"type":"MAT4"},
    {"bufferView":5,"componentType":5126,"count":2,"type":"SCALAR","min":[0.0],"max":[1.0]},
    {"bufferView":6,"componentType":5126,"count":2,"type":"VEC4"},
  ],
  "meshes": [{"primitives":[{"attributes":{"POSITION":0,"JOINTS_0":2,"WEIGHTS_0":3},"indices":1,"mode":4}]}],
  "nodes": [
    {"mesh":0,"skin":0,"name":"skinnedMesh"},
    {"name":"root","children":[2]},
    {"name":"bone1"},
  ],
  "skins": [{"joints":[1,2],"inverseBindMatrices":4,"skeleton":1}],
  "animations": [{"name":"spin",
    "channels":[{"sampler":0,"target":{"node":1,"path":"rotation"}}],
    "samplers":[{"input":5,"output":6,"interpolation":"LINEAR"}]}],
  "scenes": [{"nodes":[0,1]}],
  "scene": 0,
}
js = json.dumps(doc, separators=(",",":")).encode("utf-8")
while len(js) % 4: js += b" "
blob = bin_blob
while len(blob) % 4: blob += b"\x00"
total = 12 + 8 + len(js) + 8 + len(blob)
out = struct.pack("<III", 0x46546C67, 2, total)
out += struct.pack("<II", len(js), 0x4E4F534A) + js
out += struct.pack("<II", len(blob), 0x004E4942) + blob
open(sys.argv[1], "wb").write(out)
print(f"wrote {sys.argv[1]} ({total} bytes)")
