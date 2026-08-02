// SPDX-License-Identifier: GPL-2.0-or-later
//
// OBJ read/write in nguo space.
//
// Why our own reader rather than Unity's built-in model importer: the whole
// point of this package is that the axis conversion is explicit and tested
// (NguoAxis), and an AssetImporter applies a conversion we do not control.
// The files are also trivial — `mguso run --out DIR` writes frame_%04d.obj
// with nothing but `v` and `f` lines (engine-io/src/obj.rs:92 write_obj).

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace Seedexr.Nguo.Bridge
{
    /// <summary>Geometry as it sits in an nguo OBJ, before axis conversion.</summary>
    public sealed class NguoMeshData
    {
        public List<Vector3> Positions = new List<Vector3>();
        public List<Vector3> Normals = new List<Vector3>();
        public List<Vector2> Uvs = new List<Vector2>();
        public List<int> Triangles = new List<int>();
    }

    public static class NguoObj
    {
        static readonly char[] Space = { ' ', '\t' };
        static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

        /// <summary>
        /// Read an OBJ. Coordinates come back in <b>nguo</b> space — call
        /// <see cref="NguoAxis.ConvertMesh"/> before handing them to a Unity Mesh.
        /// </summary>
        public static NguoMeshData Read(string path)
        {
            var data = new NguoMeshData();
            // vt/vn are indexed separately from v in OBJ. mguso writes neither,
            // so we index them by vertex when the counts line up and drop them
            // otherwise rather than pretending to support the general case.
            var vt = new List<Vector2>();
            var vn = new List<Vector3>();

            foreach (var raw in File.ReadLines(path))
            {
                if (raw.Length < 2) continue;
                var line = raw.Trim();
                if (line.StartsWith("v ", StringComparison.Ordinal))
                {
                    var f = line.Split(Space, StringSplitOptions.RemoveEmptyEntries);
                    data.Positions.Add(new Vector3(Num(f[1]), Num(f[2]), Num(f[3])));
                }
                else if (line.StartsWith("vt ", StringComparison.Ordinal))
                {
                    var f = line.Split(Space, StringSplitOptions.RemoveEmptyEntries);
                    vt.Add(new Vector2(Num(f[1]), Num(f[2])));
                }
                else if (line.StartsWith("vn ", StringComparison.Ordinal))
                {
                    var f = line.Split(Space, StringSplitOptions.RemoveEmptyEntries);
                    vn.Add(new Vector3(Num(f[1]), Num(f[2]), Num(f[3])));
                }
                else if (line.StartsWith("f ", StringComparison.Ordinal))
                {
                    var f = line.Split(Space, StringSplitOptions.RemoveEmptyEntries);
                    // fan-triangulate n-gons
                    for (int k = 2; k < f.Length - 1; k++)
                    {
                        data.Triangles.Add(Index(f[1]));
                        data.Triangles.Add(Index(f[k]));
                        data.Triangles.Add(Index(f[k + 1]));
                    }
                }
            }

            if (vt.Count == data.Positions.Count) data.Uvs = vt;
            if (vn.Count == data.Positions.Count) data.Normals = vn;
            return data;
        }

        /// <summary>
        /// Write a Unity mesh out as an nguo-space OBJ — the collider / avatar
        /// target the solver consumes. Vertices are baked to world space,
        /// because nguo has no scene graph and would otherwise drop the
        /// transform silently.
        /// </summary>
        public static void Write(string path, Mesh mesh, Matrix4x4 localToWorld)
        {
            // Clone explicitly. Unity's Mesh.vertices getter does hand back a
            // fresh copy, but relying on that means a refactor to a
            // GetVertices(list) overload silently starts mutating the source
            // mesh — and exporting a collider must never move the collider.
            var positions = (Vector3[])mesh.vertices.Clone();
            var triangles = (int[])mesh.triangles.Clone();
            for (int i = 0; i < positions.Length; i++)
            {
                positions[i] = NguoAxis.Invert(localToWorld.MultiplyPoint3x4(positions[i]));
            }
            // Unity -> nguo mirrors just as nguo -> Unity does, so the winding
            // has to come back too.
            for (int t = 0; t + 2 < triangles.Length; t += 3)
            {
                (triangles[t + 1], triangles[t + 2]) = (triangles[t + 2], triangles[t + 1]);
            }

            var sb = new StringBuilder(positions.Length * 24);
            sb.Append("# nguo-3d Unity bridge export: ").Append(mesh.name).Append('\n');
            foreach (var p in positions)
            {
                sb.AppendFormat(Inv, "v {0} {1} {2}\n", p.x, p.y, p.z);
            }
            for (int t = 0; t + 2 < triangles.Length; t += 3)
            {
                sb.AppendFormat(Inv, "f {0} {1} {2}\n",
                    triangles[t] + 1, triangles[t + 1] + 1, triangles[t + 2] + 1);
            }
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path)));
            File.WriteAllText(path, sb.ToString());
        }

        /// <summary>Sorted frame_*.obj paths written by <c>mguso run --out</c>.</summary>
        public static string[] FrameFiles(string directory)
        {
            var files = Directory.GetFiles(directory, "frame_*.obj");
            Array.Sort(files, StringComparer.Ordinal);
            return files;
        }

        static float Num(string s) => float.Parse(s, Inv);

        static int Index(string token)
        {
            int slash = token.IndexOf('/');
            if (slash >= 0) token = token.Substring(0, slash);
            return int.Parse(token, Inv) - 1; // OBJ is 1-based
        }
    }
}
