// SPDX-License-Identifier: GPL-2.0-or-later
//
// The -executeMethod entry points. This is how CI and nguo-3d itself drive
// Unity; the menu items in NguoMenu.cs are a convenience on top.
//
//   Unity -batchmode -nographics -quit -logFile - \
//         -projectPath <p> -executeMethod Seedexr.Nguo.Bridge.NguoHeadless.<M> \
//         --key value ...
//
// Flags and their semantics are Unity's, from the Editor command line
// arguments reference: -batchmode runs without interaction and exits 1 on an
// uncaught exception; -nographics turns output logs OFF unless -logFile is
// given; -executeMethod runs a *static* method after the project opens, and
// "to return an error ... either throw an exception ... or call
// EditorApplication.Exit with a non-zero return code". Unity has no first-class
// parameter passing, so we read our own arguments back out of
// System.Environment.GetCommandLineArgs() — also per that reference.
//
// Exit codes here line up with nguo_bridge_dcc::classify:
//   0  success
//   1  Unity's own "an exception occurred"
//   2  bad arguments        -> BridgeError::Script
//   3  the operation failed -> BridgeError::Script
//   198 Unity has no licence (Unity's own) -> BridgeError::NoLicence

using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace Seedexr.Nguo.Bridge
{
    public static class NguoHeadless
    {
        /// <summary>
        /// Sentinel the Rust side greps for. Must stay byte-identical to
        /// nguo_bridge_dcc::RESULT_PREFIX and to the Blender bridge's.
        /// </summary>
        public const string ResultPrefix = "NGUO-BRIDGE-RESULT:";

        const int ExitBadArgs = 2;
        const int ExitFailed = 3;

        // ------------------------------------------------------------------
        // entry points
        // ------------------------------------------------------------------

        /// <summary>Register-and-report: proves the package compiled and loaded.</summary>
        public static void SelfTest()
        {
            Run(args =>
            {
                // Exercise the conversion so a broken build fails here rather
                // than silently mirroring somebody's garment later.
                var v = new Vector3(1f, 2f, 3f);
                if (NguoAxis.Invert(NguoAxis.Convert(v)) != v)
                {
                    throw new Exception("NguoAxis round trip is not the identity");
                }
                Emit($"{{\"unity\": \"{Application.unityVersion}\", \"ok\": true}}");
            });
        }

        /// <summary>
        /// <c>--dir DIR [--fps 30] [--vertex-limit N] [--asset Assets/x.asset]</c>
        /// — import a frame sequence as blend shapes.
        /// </summary>
        public static void ImportSequence()
        {
            Run(args =>
            {
                var dir = Require(args, "--dir");
                var fps = float.Parse(Get(args, "--fps", "30"), System.Globalization.CultureInfo.InvariantCulture);
                var limit = int.Parse(Get(args, "--vertex-limit", "0"));
                var (mesh, clip) = NguoImport.ImportSequence(dir, fps, limit);
                var asset = Get(args, "--asset", null);
                if (asset != null) NguoImport.SaveAssets(asset, mesh, clip);
                Emit($"{{\"vertices\": {mesh.vertexCount}, \"blendShapes\": {mesh.blendShapeCount}}}");
            });
        }

        /// <summary><c>--path X.obj [--asset Assets/x.asset]</c> — import a single frame.</summary>
        public static void ImportMesh()
        {
            Run(args =>
            {
                var mesh = NguoImport.ImportMesh(Require(args, "--path"));
                var asset = Get(args, "--asset", null);
                if (asset != null) NguoImport.SaveAssets(asset, mesh, null);
                Emit($"{{\"vertices\": {mesh.vertexCount}, \"triangles\": {mesh.triangles.Length / 3}}}");
            });
        }

        /// <summary>
        /// <c>--out X.obj [--object NAME]</c> — export scene meshes as an nguo
        /// collider / avatar target. Without <c>--object</c>, every MeshFilter
        /// in the open scene is written.
        /// </summary>
        public static void ExportCollider()
        {
            Run(args =>
            {
                var outPath = Require(args, "--out");
                var wanted = Get(args, "--object", null);
                var filters = new List<MeshFilter>();
                foreach (var mf in UnityEngine.Object.FindObjectsByType<MeshFilter>(FindObjectsSortMode.None))
                {
                    if (mf.sharedMesh != null && (wanted == null || mf.gameObject.name == wanted))
                    {
                        filters.Add(mf);
                    }
                }
                if (filters.Count == 0) Fail(ExitFailed, "no mesh to export");
                // ponytail: one object per call. nguo takes a list of objects in
                // its scene JSON, so exporting several is a loop on the caller's
                // side; merge them here only if a real workflow needs it.
                NguoObj.Write(outPath, filters[0].sharedMesh, filters[0].transform.localToWorldMatrix);
                Emit($"{{\"path\": {Quote(Path.GetFullPath(outPath))}, \"object\": {Quote(filters[0].gameObject.name)}}}");
            });
        }

        // ------------------------------------------------------------------
        // plumbing
        // ------------------------------------------------------------------

        static void Run(Action<string[]> body)
        {
            var args = Environment.GetCommandLineArgs();
            try
            {
                body(args);
            }
            catch (ExitException)
            {
                return; // Fail() already reported and called EditorApplication.Exit
            }
            catch (Exception e)
            {
                Debug.LogError($"nguo bridge: {e}");
                EditorApplication.Exit(ExitFailed);
                return;
            }
            EditorApplication.Exit(0);
        }

        // Emitted twice on purpose. Console.WriteLine lands on the process
        // stdout that `-logFile -` shares; Debug.Log lands in the Editor log.
        // Which of the two survives depends on how Unity was launched, and
        // parse_result takes the LAST matching line, so a duplicate is free
        // insurance against a silent no-result run.
        static void Emit(string json)
        {
            var line = $"{ResultPrefix} {json}";
            Console.WriteLine(line);
            Debug.Log(line);
        }

        static void Fail(int code, string message)
        {
            Debug.LogError($"nguo bridge: {message}");
            EditorApplication.Exit(code);
            throw new ExitException();
        }

        static string Get(string[] args, string key, string fallback)
        {
            for (int i = 0; i < args.Length - 1; i++)
            {
                if (args[i] == key) return args[i + 1];
            }
            return fallback;
        }

        static string Require(string[] args, string key)
        {
            var v = Get(args, key, null);
            if (v == null) Fail(ExitBadArgs, $"missing required argument {key}");
            return v;
        }

        static string Quote(string s) => "\"" + s.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";

        sealed class ExitException : Exception { }
    }
}
