// SPDX-License-Identifier: GPL-2.0-or-later
//
// Coordinate-system conversion between nguo-3d and Unity.
//
// This is a mirror of ui/crates/nguo-bridge-dcc/src/axis.rs, which is the
// authority and where the round-trip/winding tests live. If you change one,
// change both — Tests/Editor/NguoAxisTests.cs re-asserts the same properties
// on this side.

using UnityEngine;

namespace Seedexr.Nguo.Bridge
{
    /// <summary>
    /// nguo-3d is glTF space: right-handed, +Y up, +Z forward, <b>-X right</b>.
    /// Unity is left-handed, +Y up, +Z forward, <b>+X right</b>.
    ///
    /// The conversion that keeps "up" and "forward" meaning the same thing is
    /// therefore to negate X. That is also what Unity's own glTFast (>= 5.0)
    /// does — it moved from negate-Z to negate-X precisely to honour the glTF
    /// spec's "+Z is forward" — so a GLB imported through glTFast and an OBJ
    /// sequence imported through <see cref="NguoObj"/> land in the same place.
    ///
    /// Negating one axis has determinant -1: it <b>mirrors</b>. Two consequences
    /// that are not optional:
    ///   * triangle winding must be reversed, or the garment renders inside-out;
    ///   * normals go through the same matrix (it is orthogonal, so M^-T == M),
    ///     and it is the winding flip that puts the geometric normal back on the
    ///     same side as the shading normal.
    /// <see cref="ConvertMesh"/> does both. Prefer it over the per-element calls.
    ///
    /// UVs: glTF/nguo put the origin top-left (v grows down), Unity bottom-left.
    /// Both directions are v -> 1 - v.
    /// </summary>
    public static class NguoAxis
    {
        /// <summary>nguo -> Unity. Its own inverse.</summary>
        public static Vector3 Convert(Vector3 v) => new Vector3(-v.x, v.y, v.z);

        /// <summary>Unity -> nguo. Same map; negating X is an involution.</summary>
        public static Vector3 Invert(Vector3 v) => new Vector3(-v.x, v.y, v.z);

        /// <summary>glTF/nguo UV origin &lt;-&gt; Unity UV origin. Its own inverse.</summary>
        public static Vector2 ConvertUv(Vector2 uv) => new Vector2(uv.x, 1f - uv.y);

        /// <summary>True: the nguo &lt;-&gt; Unity map mirrors, so winding must flip.</summary>
        public const bool FlipsWinding = true;

        /// <summary>
        /// Convert positions, normals, UVs and winding together, in place.
        /// Getting three of the four right is the classic silent interop bug.
        /// </summary>
        public static void ConvertMesh(Vector3[] positions, Vector3[] normals, Vector2[] uvs, int[] triangles)
        {
            if (positions != null)
            {
                for (int i = 0; i < positions.Length; i++) positions[i] = Convert(positions[i]);
            }
            if (normals != null)
            {
                for (int i = 0; i < normals.Length; i++) normals[i] = Convert(normals[i]);
            }
            if (uvs != null)
            {
                for (int i = 0; i < uvs.Length; i++) uvs[i] = ConvertUv(uvs[i]);
            }
            if (triangles != null)
            {
                for (int t = 0; t + 2 < triangles.Length; t += 3)
                {
                    (triangles[t + 1], triangles[t + 2]) = (triangles[t + 2], triangles[t + 1]);
                }
            }
        }
    }
}
