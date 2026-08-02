// SPDX-License-Identifier: GPL-2.0-or-later
//
// Turn nguo-3d output into Unity assets.

using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace Seedexr.Nguo.Bridge
{
    public static class NguoImport
    {
        /// <summary>
        /// Import one nguo OBJ as a Unity <see cref="Mesh"/>, axis-converted.
        /// </summary>
        public static Mesh ImportMesh(string objPath)
        {
            var data = NguoObj.Read(objPath);
            var positions = data.Positions.ToArray();
            var normals = data.Normals.Count == positions.Length ? data.Normals.ToArray() : null;
            var uvs = data.Uvs.Count == positions.Length ? data.Uvs.ToArray() : null;
            var tris = data.Triangles.ToArray();

            NguoAxis.ConvertMesh(positions, normals, uvs, tris);

            var mesh = new Mesh { name = Path.GetFileNameWithoutExtension(objPath) };
            if (positions.Length > 65535) mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
            mesh.vertices = positions;
            if (uvs != null) mesh.uv = uvs;
            mesh.triangles = tris;
            if (normals != null) mesh.normals = normals; else mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        /// <summary>
        /// Import an nguo frame sequence as blend shapes on one mesh, plus an
        /// <see cref="AnimationClip"/> that plays them.
        ///
        /// Blend shapes rather than one mesh per frame: it is the Unity analogue
        /// of the shape keys the Blender bridge builds, it keeps a single
        /// material and UV set, and it needs no extra package. It requires the
        /// topology to be constant across the sequence — which it is, the solver
        /// never remeshes.
        ///
        /// <paramref name="vertexLimit"/> keeps only the leading N vertices:
        /// mguso lowers every scene object into one vertex space and writes them
        /// all to each frame, so a drape against a collider otherwise arrives as
        /// cloth+collider welded into one mesh. Pass the cloth vertex count.
        /// </summary>
        public static (Mesh mesh, AnimationClip clip) ImportSequence(
            string directory, float fps = 30f, int vertexLimit = 0)
        {
            var files = NguoObj.FrameFiles(directory);
            if (files.Length == 0)
            {
                throw new IOException($"no frame_*.obj files in {directory}");
            }

            var mesh = ImportMesh(files[0]);
            if (vertexLimit > 0 && vertexLimit < mesh.vertexCount)
            {
                mesh = Truncate(mesh, vertexLimit);
            }
            mesh.name = "nguo_drape";
            var basis = mesh.vertices;
            var n = basis.Length;

            var clip = new AnimationClip { name = "nguo_drape", frameRate = fps };
            var deltas = new Vector3[n];

            for (int f = 1; f < files.Length; f++)
            {
                var data = NguoObj.Read(files[f]);
                if (data.Positions.Count < n)
                {
                    throw new IOException(
                        $"frame {f} has {data.Positions.Count} vertices, frame 0 has {n} — " +
                        "topology must be constant across an nguo sequence");
                }
                for (int i = 0; i < n; i++)
                {
                    deltas[i] = NguoAxis.Convert(data.Positions[i]) - basis[i];
                }
                var name = $"frame_{f:D4}";
                mesh.AddBlendShapeFrame(name, 100f, deltas, null, null);

                // 0 -> 100 -> 0 across three frames, so exactly one shape is
                // fully on at each timeline frame. O(3n) keys, not O(n^2).
                var curve = new AnimationCurve(
                    new Keyframe((f - 1) / fps, 0f),
                    new Keyframe(f / fps, 100f),
                    new Keyframe((f + 1) / fps, 0f));
                clip.SetCurve("", typeof(SkinnedMeshRenderer), "blendShape." + name, curve);
            }

            mesh.RecalculateBounds();
            return (mesh, clip);
        }

        /// <summary>
        /// Put a sequence in the open scene as a <see cref="SkinnedMeshRenderer"/>.
        /// A SkinnedMeshRenderer is required — a MeshRenderer cannot play blend
        /// shapes at all.
        ///
        /// ponytail: this does NOT wire up playback. The clip returned by
        /// <see cref="ImportSequence"/> drives <c>blendShape.frame_NNNN</c>
        /// properties, which needs an Animator Controller — one drag in the
        /// Editor, and generating controllers from script is a pile of
        /// AnimatorController API for no gain. Assign the clip yourself, or ask
        /// and it gets built.
        /// </summary>
        public static GameObject Instantiate(Mesh mesh, Material material = null)
        {
            var go = new GameObject(mesh.name);
            var smr = go.AddComponent<SkinnedMeshRenderer>();
            smr.sharedMesh = mesh;
            smr.rootBone = go.transform;
            smr.localBounds = mesh.bounds;
            smr.sharedMaterial = material != null
                ? material
                : AssetDatabase.GetBuiltinExtraResource<Material>("Default-Material.mat");
            return go;
        }

        /// <summary>Write a mesh (+ optional clip) out as a .asset the project can reference.</summary>
        public static void SaveAssets(string assetPath, Mesh mesh, AnimationClip clip)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(assetPath)));
            AssetDatabase.CreateAsset(mesh, assetPath);
            if (clip != null) AssetDatabase.AddObjectToAsset(clip, mesh);
            AssetDatabase.SaveAssets();
        }

        // GLB import is deliberately NOT implemented here.
        //
        // engine-io's only material-carrying writer is `save_glb`
        // (gltf_io.rs:543) — it can *read* FBX/USD/COLLADA but writes none of
        // them — so GLB is the right format for a garment with materials. But
        // Unity's own com.unity.cloud.gltfast package already ships a
        // ScriptedImporter for .gltf/.glb: add the package, drop the file in
        // Assets/, done. Wrapping that in our own code would add a hard package
        // dependency and buy nothing. glTFast >= 5.0 converts by negating X,
        // which is exactly NguoAxis, so the two paths agree.

        static Mesh Truncate(Mesh src, int n)
        {
            var positions = src.vertices;
            var uvs = src.uv;
            var normals = src.normals;
            var tris = src.triangles;

            var keptTris = new List<int>(tris.Length);
            for (int t = 0; t + 2 < tris.Length; t += 3)
            {
                if (tris[t] < n && tris[t + 1] < n && tris[t + 2] < n)
                {
                    keptTris.Add(tris[t]);
                    keptTris.Add(tris[t + 1]);
                    keptTris.Add(tris[t + 2]);
                }
            }

            var mesh = new Mesh { name = src.name, indexFormat = src.indexFormat };
            mesh.vertices = Head(positions, n);
            if (uvs != null && uvs.Length >= n) mesh.uv = Head(uvs, n);
            mesh.triangles = keptTris.ToArray();
            if (normals != null && normals.Length >= n) mesh.normals = Head(normals, n);
            else mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        static T[] Head<T>(T[] src, int n)
        {
            var dst = new T[n];
            System.Array.Copy(src, dst, n);
            return dst;
        }
    }
}
