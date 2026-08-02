// SPDX-License-Identifier: GPL-2.0-or-later
//
// NOT UPSTREAM. Written for nguo-3d.
//
// GCS.cpp's `diagnose()` partitions the constraint system into decoupled
// components using exactly six Boost Graph symbols:
//
//     boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>
//     boost::add_vertex(g)
//     boost::add_edge(u, v, g)
//     boost::num_vertices(g)
//     boost::connected_components(g, &out[0])
//
// Boost Graph is header-only but not vendorable in isolation (it drags in
// mpl/type_traits/iterator/...). Rather than vendor ~150 MB of Boost or patch
// GCS.cpp, we satisfy those six symbols with a weighted union-find. Both
// headers GCS.cpp includes are angle-bracket, so this file simply shadows the
// real Boost on the include path; the real Boost is never on it.
//
// Component labels must match Boost's contract, which GCS.cpp relies on:
// labels are 0..n-1, dense, and assigned in ascending order of the smallest
// vertex index in each component. The relabelling pass below reproduces that.
#pragma once

#include <cstddef>
#include <numeric>
#include <vector>

namespace boost
{

struct vecS
{};
struct undirectedS
{};
struct directedS
{};

template<class OutEdgeListS = vecS, class VertexListS = vecS, class DirectedS = undirectedS>
class adjacency_list
{
public:
    std::vector<std::size_t> parent;
    std::vector<std::size_t> rank;

    std::size_t find(std::size_t x)
    {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];  // path halving
            x = parent[x];
        }
        return x;
    }

    void unite(std::size_t a, std::size_t b)
    {
        a = find(a);
        b = find(b);
        if (a == b) {
            return;
        }
        if (rank[a] < rank[b]) {
            std::swap(a, b);
        }
        parent[b] = a;
        if (rank[a] == rank[b]) {
            ++rank[a];
        }
    }
};

template<class O, class V, class D>
std::size_t add_vertex(adjacency_list<O, V, D>& g)
{
    const std::size_t v = g.parent.size();
    g.parent.push_back(v);
    g.rank.push_back(0);
    return v;
}

template<class O, class V, class D, class U, class W>
void add_edge(U u, W v, adjacency_list<O, V, D>& g)
{
    g.unite(static_cast<std::size_t>(u), static_cast<std::size_t>(v));
}

template<class O, class V, class D>
std::size_t num_vertices(const adjacency_list<O, V, D>& g)
{
    return g.parent.size();
}

/// Writes a dense component label per vertex into `out` and returns the number
/// of components. Labels are assigned in ascending order of first appearance,
/// matching Boost's behaviour for `vecS` vertex storage.
template<class O, class V, class D, class ComponentMap>
int connected_components(adjacency_list<O, V, D>& g, ComponentMap out)
{
    const std::size_t n = g.parent.size();
    std::vector<int> label(n, -1);
    int next = 0;
    for (std::size_t v = 0; v < n; ++v) {
        const std::size_t root = g.find(v);
        if (label[root] < 0) {
            label[root] = next++;
        }
        out[v] = label[root];
    }
    return next;
}

}  // namespace boost
