// SPDX-License-Identifier: GPL-2.0-or-later
//
// The same properties ui/crates/nguo-bridge-dcc/src/axis.rs asserts in Rust,
// re-asserted on the C# side so the two cannot drift apart silently.
//
//   Unity -batchmode -nographics -projectPath <p> -runTests -testPlatform EditMode

using System.IO;
using NUnit.Framework;
using UnityEngine;

namespace Seedexr.Nguo.Bridge.Tests
{
    public class NguoAxisTests
    {
        static readonly Vector3[] Samples =
        {
            new Vector3(0f, 0f, 0f),
            new Vector3(1f, 2f, 3f),
            new Vector3(-0.5f, 12.25f, -7.125f),
        };

        [Test]
        public void RoundTripIsIdentity()
        {
            foreach (var v in Samples)
            {
                Assert.AreEqual(v, NguoAxis.Invert(NguoAxis.Convert(v)));
                Assert.AreEqual(v, NguoAxis.Convert(NguoAxis.Invert(v)));
            }
        }

        [Test]
        public void SemanticAxesSurviveTheTrip()
        {
            // up stays up, forward stays forward, and glTF's -X right becomes
            // Unity's +X right.
            Assert.AreEqual(Vector3.up, NguoAxis.Convert(Vector3.up));
            Assert.AreEqual(Vector3.forward, NguoAxis.Convert(Vector3.forward));
            Assert.AreEqual(Vector3.right, NguoAxis.Convert(Vector3.left));
        }

        [Test]
        public void UvRoundTripIsIdentityAndIsARealFlip()
        {
            var uv = new Vector2(0.25f, 0.75f);
            Assert.AreEqual(uv, NguoAxis.ConvertUv(NguoAxis.ConvertUv(uv)));
            Assert.AreEqual(new Vector2(0.25f, 0.25f), NguoAxis.ConvertUv(uv));
        }

        [Test]
        public void ConvertedMeshFacesTheSameWayItDidBefore()
        {
            // CCW in the XZ plane, normal at +Y.
            var positions = new[]
            {
                new Vector3(0f, 0f, 0f), new Vector3(1f, 0f, 0f), new Vector3(0f, 0f, -1f)
            };
            var normals = new[] { Vector3.up, Vector3.up, Vector3.up };
            var uvs = new Vector2[3];
            var tris = new[] { 0, 1, 2 };
            Assert.Greater(Vector3.Dot(GeoNormal(positions, tris, 0), normals[0]), 0f,
                "fixture is wound the wrong way");

            NguoAxis.ConvertMesh(positions, normals, uvs, tris);

            Assert.Greater(Vector3.Dot(GeoNormal(positions, tris, 0), normals[0]), 0f,
                "converted mesh is inside-out — the winding flip is missing");
        }

        [Test]
        public void WithoutTheWindingFlipItIsInsideOut()
        {
            var positions = new[]
            {
                NguoAxis.Convert(new Vector3(0f, 0f, 0f)),
                NguoAxis.Convert(new Vector3(1f, 0f, 0f)),
                NguoAxis.Convert(new Vector3(0f, 0f, -1f)),
            };
            var tris = new[] { 0, 1, 2 };
            Assert.Less(
                Vector3.Dot(GeoNormal(positions, tris, 0), NguoAxis.Convert(Vector3.up)), 0f);
        }

        [Test]
        public void ObjRoundTripsThroughDiskUnchanged()
        {
            var mesh = new Mesh
            {
                vertices = new[]
                {
                    new Vector3(0f, 0f, 0f), new Vector3(1f, 0f, 0f), new Vector3(0f, 2f, -1f)
                },
                triangles = new[] { 0, 1, 2 },
            };
            var path = Path.Combine(Path.GetTempPath(), "nguo_axis_round_trip.obj");
            NguoObj.Write(path, mesh, Matrix4x4.identity);

            var data = NguoObj.Read(path);
            var positions = data.Positions.ToArray();
            var tris = data.Triangles.ToArray();
            NguoAxis.ConvertMesh(positions, null, null, tris);

            Assert.AreEqual(3, positions.Length);
            for (int i = 0; i < 3; i++)
            {
                Assert.Less(Vector3.Distance(positions[i], mesh.vertices[i]), 1e-4f, $"vertex {i}");
            }
            CollectionAssert.AreEqual(mesh.triangles, tris);
            File.Delete(path);
        }

        static Vector3 GeoNormal(Vector3[] p, int[] tris, int t)
        {
            var a = p[tris[t * 3]];
            var b = p[tris[t * 3 + 1]];
            var c = p[tris[t * 3 + 2]];
            return Vector3.Cross(b - a, c - a);
        }
    }
}
