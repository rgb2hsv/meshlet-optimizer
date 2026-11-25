#include <core/Check.hpp>
#include <core/Types.hpp>
#include <hierarchy/RegionHierarchy.hpp>
#include <mlod/MlodWriter.hpp>

#include <meshoptimizer.h>
#define CLUSTERLOD_IMPLEMENTATION
#include <demo/clusterlod.h>

#include <cereal/archives/portable_binary.hpp>
#include <cereal/cereal.hpp>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <queue>
#include <tuple>
#include <type_traits>
#include <vector>

using namespace mopt;

namespace
{
// ---------------------------------------------------------------------------
// The .mlod layout, as defined by meshlet-viewer (src/model.h and
// src/meshlet_file.cpp there). Field order, types and sizes must match it.
namespace mlod
{
// A vertex of the .mlod model: position and normal.
struct Vertex {
    float px, py, pz;
    float nx, ny, nz;
};

// Texture coordinates of a vertex (version 2).
struct Texcoord {
    float u, v;
};

// The .mlod model: vertices, their texture coordinates (version 2; empty without), level-0 indices and bounds.
struct Model {
    std::vector<Vertex> vertices;
    std::vector<Texcoord> texcoords;
    std::vector<uint32_t> indices;
    float boundsMin[3] = {0, 0, 0};
    float boundsMax[3] = {0, 0, 0};
    float center[3] = {0, 0, 0};
    float radius = 0;
    float avgEdgeLength = 0;
};

enum class MeshletAlgorithm : int { Greedy = 0, Flex = 1, Spatial = 2, Scan = 3 };

// The meshlet builder settings the viewer records (written as the largest meshlet found).
struct MeshletSettings {
    MeshletAlgorithm algorithm = MeshletAlgorithm::Greedy;
    int maxVertices = 64;
    int maxTriangles = 124;
    int minTriangles = 32;
    float coneWeight = 0.25f;
    float splitFactor = 2.0f;
    float fillWeight = 0.5f;
    bool optimize = true;
    bool lod = true;
    bool lodSpatial = false;
    float lodSimplifyRatio = 0.5f;
    int lodPartitionSize = 16;
};

// A meshlet's own and coarser bounding spheres and errors, and its DAG depth.
struct MeshletLod {
    float selfCenter[3];
    float selfRadius;
    float coarseCenter[3];
    float coarseRadius;
    float selfError;
    float coarseError;
    uint32_t level;
    uint32_t pad;
};

// A meshlet's bounding sphere and normal cone.
struct MeshletBounds {
    float center[3];
    float radius;
    float coneAxis[3];
    float coneCutoff;
};

// A node of the LOD selection BVH.
struct LodNode {
    float center[3];
    float radius;
    float error;
    int32_t group;
    uint32_t childOffset;
    uint32_t childCount;
};

// A cluster group's range of meshlets.
struct LodGroupRange {
    uint32_t firstCluster;
    uint32_t clusterCount;
};

// Everything the viewer uploads: meshlets, their vertices and triangles, bounds, LOD data, BVH and statistics.
struct MeshletData {
    std::vector<meshopt_Meshlet> meshlets;
    std::vector<uint32_t> vertices;
    std::vector<uint8_t> triangles;
    std::vector<MeshletBounds> bounds;
    std::vector<MeshletLod> lods;
    std::vector<uint8_t> edgeMasks;
    double buildSeconds = 0;
    size_t triangleCount = 0;
    size_t baseTriangleCount = 0;
    bool lod = false;
    std::vector<size_t> levelClusters, levelTriangles;
    size_t coarsestTriangles = 0;
    std::vector<LodNode> lodNodes;
    std::vector<LodGroupRange> lodGroups;
    uint32_t lodRootCount = 0;
    std::vector<uint32_t> lodNodesPerDepth;
    float avgVertices = 0, avgTriangles = 0;
    float vertexFill = 0, triangleFill = 0;
    float avgConeAngleDeg = 0;
    size_t coneCullable = 0;
    float coneCullableFraction = 0;
};

static_assert(sizeof(Vertex) == 24 && sizeof(meshopt_Meshlet) == 16 && sizeof(MeshletBounds) == 32 &&
                  sizeof(MeshletLod) == 48 && sizeof(LodNode) == 32 && sizeof(LodGroupRange) == 8,
              ".mlod layout changed: update the writer to match meshlet-viewer");

constexpr uint32_t kMagic = 0x444f4c4d;  // "MLOD" in little endian
// Version 1: no texture coordinates. Version 2: a texture coordinate per vertex, right after the vertices.
constexpr uint32_t kVersion = 1;
constexpr uint32_t kTexcoordVersion = 2;

using OutputArchive = cereal::PortableBinaryOutputArchive;

template <class Word, class T>
void Array(OutputArchive& ar, const std::vector<T>& v)
{
    static_assert(std::is_trivially_copyable_v<T> && sizeof(T) % sizeof(Word) == 0);
    ar(uint64_t(v.size()));
    if (!v.empty()) ar(cereal::binary_data(reinterpret_cast<const Word*>(v.data()), v.size() * sizeof(T)));
}

void SizeField(OutputArchive& ar, size_t s) { ar(uint64_t(s)); }

void SizeArray(OutputArchive& ar, const std::vector<size_t>& v)
{
    const std::vector<uint64_t> wide(v.begin(), v.end());
    Array<uint64_t>(ar, wide);
}

void Archive(OutputArchive& ar, uint32_t version, Model& m, MeshletSettings& s, MeshletData& d)
{
    int32_t algorithm = int32_t(s.algorithm);
    ar(algorithm, s.maxVertices, s.maxTriangles, s.minTriangles, s.coneWeight, s.splitFactor, s.fillWeight, s.optimize);
    ar(s.lod, s.lodSpatial, s.lodSimplifyRatio, s.lodPartitionSize);

    Array<uint32_t>(ar, m.vertices);
    if (version >= kTexcoordVersion) Array<uint32_t>(ar, m.texcoords);
    Array<uint32_t>(ar, m.indices);
    ar(m.boundsMin, m.boundsMax, m.center, m.radius, m.avgEdgeLength);

    Array<uint32_t>(ar, d.meshlets);
    Array<uint32_t>(ar, d.vertices);
    Array<uint8_t>(ar, d.triangles);
    Array<uint32_t>(ar, d.bounds);
    Array<uint32_t>(ar, d.lods);
    Array<uint8_t>(ar, d.edgeMasks);

    ar(d.buildSeconds, d.lod);
    SizeField(ar, d.triangleCount);
    SizeField(ar, d.baseTriangleCount);
    SizeArray(ar, d.levelClusters);
    SizeArray(ar, d.levelTriangles);
    SizeField(ar, d.coarsestTriangles);

    Array<uint32_t>(ar, d.lodNodes);
    Array<uint32_t>(ar, d.lodGroups);
    ar(d.lodRootCount);
    Array<uint32_t>(ar, d.lodNodesPerDepth);

    ar(d.avgVertices, d.avgTriangles, d.vertexFill, d.triangleFill, d.avgConeAngleDeg);
    SizeField(ar, d.coneCullable);
    ar(d.coneCullableFraction);
}
}  // namespace mlod

// ---------------------------------------------------------------------------
using V3 = Vector3f;

// A bounding sphere (radius < 0: empty).
struct Sphere {
    V3 center = V3::Zero();
    float radius = -1.0f;  // empty
};

Sphere SphereOf(const std::vector<V3>& points)
{
    Sphere s;
    if (points.empty()) return s;
    V3 c = V3::Zero();
    for (const V3& p : points) c += p;
    c /= static_cast<float>(points.size());
    float r = 0.0f;
    for (const V3& p : points) r = std::max(r, (p - c).norm());
    s.center = c;
    s.radius = r;
    return s;
}

// Smallest sphere containing both.
Sphere Merge(const Sphere& a, const Sphere& b)
{
    if (a.radius < 0) return b;
    if (b.radius < 0) return a;
    const V3 d = b.center - a.center;
    const float dist = d.norm();
    if (dist + b.radius <= a.radius) return a;
    if (dist + a.radius <= b.radius) return b;
    Sphere s;
    s.radius = 0.5f * (dist + a.radius + b.radius);
    s.center = a.center + d * ((s.radius - a.radius) / std::max(dist, 1e-30f));
    return s;
}

// Distance from p to triangle abc (closest point, Ericson's method).
float PointTriangleDistance(const V3& p, const V3& a, const V3& b, const V3& c)
{
    const V3 ab = b - a, ac = c - a, ap = p - a;
    const float d1 = ab.dot(ap), d2 = ac.dot(ap);
    if (d1 <= 0 && d2 <= 0) return ap.norm();
    const V3 bp = p - b;
    const float d3 = ab.dot(bp), d4 = ac.dot(bp);
    if (d3 >= 0 && d4 <= d3) return bp.norm();
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) return (p - (a + ab * (d1 / (d1 - d3)))).norm();
    const V3 cp = p - c;
    const float d5 = ab.dot(cp), d6 = ac.dot(cp);
    if (d6 >= 0 && d5 <= d6) return cp.norm();
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) return (p - (a + ac * (d2 / (d2 - d6)))).norm();
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
        return (p - (b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))))).norm();
    }
    const float denom = 1.0f / (va + vb + vc);
    return (p - (a + ab * (vb * denom) + ac * (vc * denom))).norm();
}

// Largest distance from a vertex of `from` to the triangles of `to`.
float OneSidedDistance(const Mesh& fromMesh, const std::vector<int>& fromFaces, const Mesh& toMesh,
                       const std::vector<int>& toFaces)
{
    std::vector<uint32_t> verts;
    for (int f : fromFaces) {
        const Vector3u t = fromMesh.triIndexes[f];
        verts.insert(verts.end(), {t.x(), t.y(), t.z()});
    }
    std::sort(verts.begin(), verts.end());
    verts.erase(std::unique(verts.begin(), verts.end()), verts.end());
    float worst = 0.0f;
    for (uint32_t v : verts) {
        const V3 p = fromMesh.vertexLocations[v];
        float best = FLT_MAX;
        for (int f : toFaces) {
            const Vector3u t = toMesh.triIndexes[f];
            best = std::min(best, PointTriangleDistance(p, toMesh.vertexLocations[t.x()], toMesh.vertexLocations[t.y()],
                                                        toMesh.vertexLocations[t.z()]));
            if (best <= worst) break;  // cannot raise the maximum
        }
        worst = std::max(worst, best);
    }
    return worst;
}

std::vector<V3> AreaWeightedNormals(const Mesh& mesh)
{
    std::vector<V3> n(mesh.vertexLocations.size(), V3::Zero());
    for (const Vector3u& t : mesh.triIndexes) {
        const V3 a = mesh.vertexLocations[t.x()], b = mesh.vertexLocations[t.y()], c = mesh.vertexLocations[t.z()];
        const V3 fn = (b - a).cross(c - a);
        n[t.x()] += fn;
        n[t.y()] += fn;
        n[t.z()] += fn;
    }
    for (V3& v : n) {
        const float l = v.norm();
        v = l > 0 ? v / l : V3(0, 0, 1);
    }
    return n;
}

// Meshlets of a set of faces: meshlets index into vertices/triangles, and
// globalOf maps local vertex ids back to the level's output vertices.
struct FaceMeshlets {
    std::vector<meshopt_Meshlet> meshlets;
    std::vector<unsigned int> vertices;
    std::vector<unsigned char> triangles;
    std::vector<uint32_t> globalOf;
};

// The format's ceiling for one meshlet: local vertex indices are 8-bit, and
// the viewer's mesh shader emits at most 256 primitives per meshlet.
constexpr size_t kMeshletVertexCeiling = 256;
constexpr size_t kMeshletTriangleCeiling = 256;

// One meshlet holding exactly these faces, or an empty result when they
// exceed the format's ceiling.
// cornerVertex: the output vertex of every triangle corner of the level.
FaceMeshlets WholeMeshlet(const Mesh& mesh, const std::vector<uint32_t>& cornerVertex, const std::vector<int>& faces)
{
    FaceMeshlets out;
    if (faces.empty() || faces.size() > kMeshletTriangleCeiling) return out;
    std::map<uint32_t, uint32_t> local;
    for (int f : faces) {
        for (int c = 0; c < 3; c++) {
            const uint32_t v = cornerVertex[3 * f + c];
            const auto [it, inserted] = local.emplace(v, static_cast<uint32_t>(out.globalOf.size()));
            if (inserted) out.globalOf.push_back(v);
            out.triangles.push_back(static_cast<unsigned char>(it->second & 0xff));
        }
        if (out.globalOf.size() > kMeshletVertexCeiling) return FaceMeshlets{};
    }
    out.vertices.resize(out.globalOf.size());
    for (size_t i = 0; i < out.vertices.size(); i++) out.vertices[i] = static_cast<unsigned int>(i);
    meshopt_Meshlet m = {};
    m.vertex_offset = 0;
    m.triangle_offset = 0;
    m.vertex_count = static_cast<unsigned int>(out.globalOf.size());
    m.triangle_count = static_cast<unsigned int>(faces.size());
    out.meshlets.push_back(m);
    return out;
}

// The faces split into k edge-connected parts of about equal size: k seeds
// spread out (each the face farthest from those chosen so far), then k flood
// fills grown at once by centroid distance, the smallest part claiming the
// next face. Faces no fill reaches (pieces touching the rest only at a
// vertex) join the smallest part.
std::vector<std::vector<int>> PartitionEvenly(const Mesh& mesh, const std::vector<int>& faces, int k)
{
    const int n = static_cast<int>(faces.size());
    std::vector<Vector3f> centroid(n);
    std::map<uint64_t, std::vector<int>> edgeFaces;
    for (int i = 0; i < n; i++) {
        const Vector3u t = mesh.triIndexes[faces[i]];
        centroid[i] = (mesh.vertexLocations[t.x()] + mesh.vertexLocations[t.y()] + mesh.vertexLocations[t.z()]) / 3.0f;
        const uint32_t v[3] = {t.x(), t.y(), t.z()};
        for (int e = 0; e < 3; e++) {
            const uint32_t a = std::min(v[e], v[(e + 1) % 3]), b = std::max(v[e], v[(e + 1) % 3]);
            edgeFaces[(uint64_t(a) << 32) | b].push_back(i);
        }
    }
    std::vector<std::vector<int>> nbr(n);
    for (const auto& [key, fs] : edgeFaces) {
        for (int a : fs) {
            for (int b : fs) {
                if (a != b) nbr[a].push_back(b);
            }
        }
    }
    using Item = std::pair<float, int>;
    using Heap = std::priority_queue<Item, std::vector<Item>, std::greater<Item>>;
    // Distance from a set of seed faces, by centroid steps.
    const auto distanceFrom = [&](const std::vector<int>& seeds) {
        std::vector<float> d(n, FLT_MAX);
        Heap heap;
        for (int s : seeds) {
            d[s] = 0.0f;
            heap.push({0.0f, s});
        }
        while (!heap.empty()) {
            const auto [dist, f] = heap.top();
            heap.pop();
            if (dist > d[f]) continue;
            for (int g : nbr[f]) {
                const float nd = dist + (centroid[g] - centroid[f]).norm();
                if (nd < d[g]) {
                    d[g] = nd;
                    heap.push({nd, g});
                }
            }
        }
        return d;
    };
    std::vector<int> seeds;
    {
        const std::vector<float> d = distanceFrom({0});
        int farthest = 0;
        for (int i = 0; i < n; i++) {
            if (d[i] != FLT_MAX && d[i] > d[farthest]) farthest = i;
        }
        seeds.push_back(farthest);
    }
    while (static_cast<int>(seeds.size()) < k) {
        const std::vector<float> d = distanceFrom(seeds);
        int farthest = -1;
        for (int i = 0; i < n; i++) {
            if (d[i] != FLT_MAX && d[i] > 0.0f && (farthest < 0 || d[i] > d[farthest])) farthest = i;
        }
        if (farthest < 0) break;  // fewer faces reachable than parts
        seeds.push_back(farthest);
    }
    const int parts = static_cast<int>(seeds.size());
    std::vector<int> owner(n, -1), count(parts, 0);
    std::vector<Heap> heaps(parts);
    const auto claim = [&](int part, int f, float d) {
        owner[f] = part;
        count[part]++;
        for (int g : nbr[f]) {
            if (owner[g] < 0) heaps[part].push({d + (centroid[g] - centroid[f]).norm(), g});
        }
    };
    for (int p = 0; p < parts; p++) claim(p, seeds[p], 0.0f);
    for (;;) {
        int best = -1;
        for (int p = 0; p < parts; p++) {
            while (!heaps[p].empty() && owner[heaps[p].top().second] >= 0) heaps[p].pop();
            if (!heaps[p].empty() && (best < 0 || count[p] < count[best])) best = p;
        }
        if (best < 0) break;
        const auto [d, f] = heaps[best].top();
        heaps[best].pop();
        claim(best, f, d);
    }
    std::vector<std::vector<int>> out(parts);
    for (int i = 0; i < n; i++) {
        if (owner[i] < 0) owner[i] = static_cast<int>(std::min_element(count.begin(), count.end()) - count.begin());
        out[owner[i]].push_back(faces[i]);
    }
    return out;
}
}  // namespace

bool mopt::WriteMlodFile(const std::filesystem::path& file, const std::vector<LodLevel>& levels,
                         const RegionHierarchy& tree, std::string& error, MlodStats* stats)
{
    const auto start = std::chrono::steady_clock::now();
    const int levelCount = static_cast<int>(levels.size());
    if (levelCount == 0 || tree.levelCount() != levelCount) {
        error = "no level hierarchy to export";
        return false;
    }
    const int top = levelCount - 1;

    mlod::Model model;
    mlod::MeshletSettings settings;
    mlod::MeshletData data;
    data.lod = true;

    // Vertex buffer: every level's surface; index buffer: level 0. With texture coordinates, a vertex on a
    // texture seam becomes one output vertex per texture coordinate it has (same position and normal).
    const bool textured =
        std::all_of(levels.begin(), levels.end(), [](const LodLevel& l) { return l.simplified.hasTexcoords(); });
    std::vector<uint32_t> vertexBase(levelCount);
    std::vector<std::vector<uint32_t>> cornerVertex(levelCount);  // per level: corner -> output vertex - base
    for (int k = 0; k < levelCount; k++) {
        const Mesh& mesh = levels[k].simplified;
        vertexBase[k] = static_cast<uint32_t>(model.vertices.size());
        const std::vector<V3> normals = AreaWeightedNormals(mesh);
        const auto addVertex = [&](uint32_t v) {
            const V3 p = mesh.vertexLocations[v];
            model.vertices.push_back({p.x(), p.y(), p.z(), normals[v].x(), normals[v].y(), normals[v].z()});
        };
        std::vector<uint32_t>& corners = cornerVertex[k];
        corners.resize(3 * mesh.triIndexes.size());
        if (!textured) {
            for (size_t v = 0; v < mesh.vertexLocations.size(); v++) addVertex(static_cast<uint32_t>(v));
            for (size_t f = 0; f < mesh.triIndexes.size(); f++) {
                for (int c = 0; c < 3; c++) corners[3 * f + c] = mesh.triIndexes[f][c];
            }
            continue;
        }
        std::map<std::tuple<uint32_t, float, float>, uint32_t> outputOf;
        for (size_t f = 0; f < mesh.triIndexes.size(); f++) {
            for (int c = 0; c < 3; c++) {
                const uint32_t v = mesh.triIndexes[f][c];
                const Vector2f uv = mesh.texcoords[3 * f + c];
                const uint32_t next = static_cast<uint32_t>(model.vertices.size()) - vertexBase[k];
                const auto [it, inserted] = outputOf.emplace(std::make_tuple(v, uv.x(), uv.y()), next);
                if (inserted) {
                    addVertex(v);
                    model.texcoords.push_back({uv.x(), uv.y()});
                }
                corners[3 * f + c] = it->second;
            }
        }
    }
    const Mesh& base = levels[0].simplified;
    if (base.triIndexes.empty()) {
        error = "level 0 has no triangles";
        return false;
    }
    double area = 0.0, edgeSum = 0.0;
    V3 centroid = V3::Zero(), lo = V3::Constant(FLT_MAX), hi = V3::Constant(-FLT_MAX);
    for (const Vector3u& t : base.triIndexes) {
        const size_t f = &t - base.triIndexes.data();
        for (int c = 0; c < 3; c++) model.indices.push_back(vertexBase[0] + cornerVertex[0][3 * f + c]);
        const V3 a = base.vertexLocations[t.x()], b = base.vertexLocations[t.y()], c = base.vertexLocations[t.z()];
        const float triArea = 0.5f * ((b - a).cross(c - a)).norm();
        area += triArea;
        centroid += (a + b + c) * (triArea / 3.0f);
        edgeSum += (b - a).norm() + (c - b).norm() + (a - c).norm();
        for (const V3& p : {a, b, c}) {
            lo = V3(std::min(lo.x(), p.x()), std::min(lo.y(), p.y()), std::min(lo.z(), p.z()));
            hi = V3(std::max(hi.x(), p.x()), std::max(hi.y(), p.y()), std::max(hi.z(), p.z()));
        }
    }
    centroid = area > 0 ? Vector3f(centroid / static_cast<float>(area)) : Vector3f(0.5f * (lo + hi));
    for (int i = 0; i < 3; i++) {
        model.boundsMin[i] = lo[i];
        model.boundsMax[i] = hi[i];
        model.center[i] = centroid[i];
    }
    for (const V3& p : base.vertexLocations) model.radius = std::max(model.radius, (p - centroid).norm());
    model.avgEdgeLength = static_cast<float>(edgeSum / (3.0 * base.triIndexes.size()));

    // Faces of each node, per level; units (a node and the halves split
    // from it) are what LOD selection swaps.
    std::vector<std::vector<std::vector<int>>> facesOf(levelCount);
    for (int k = 0; k < levelCount; k++) {
        facesOf[k].resize(tree.nodeCount(k));
        const std::vector<int>& faceRegion = levels[k].simplifiedFaceRegion;
        for (size_t f = 0; f < faceRegion.size(); f++) facesOf[k][faceRegion[f]].push_back(static_cast<int>(f));
    }
    const auto unitOf = [&](int k, int node) { return tree.splitFrom[k][node]; };
    std::vector<std::map<int, std::vector<int>>> unitFaces(levelCount);
    for (int k = 0; k < levelCount; k++) {
        for (int node = 0; node < tree.nodeCount(k); node++) {
            auto& faces = unitFaces[k][unitOf(k, node)];
            faces.insert(faces.end(), facesOf[k][node].begin(), facesOf[k][node].end());
        }
    }

    // Bounds and error of each unit, bottom up.
    std::vector<std::map<int, Sphere>> sphere(levelCount);
    std::vector<std::map<int, float>> unitError(levelCount);
    for (int k = 0; k < levelCount; k++) {
        const Mesh& mesh = levels[k].simplified;
        // Children of each unit at level k: level k-1 nodes whose parent it is.
        std::map<int, std::vector<int>> children;
        if (k > 0) {
            for (int c = 0; c < tree.nodeCount(k - 1); c++) children[tree.parent[k - 1][c]].push_back(c);
        }
        std::vector<int> units;
        for (const auto& [u, faces] : unitFaces[k]) units.push_back(u);
        std::vector<Sphere> spheres(units.size());
        std::vector<float> errors(units.size(), 0.0f);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int i = 0; i < static_cast<int>(units.size()); i++) {
            const int u = units[i];
            const std::vector<int>& faces = unitFaces[k].at(u);
            std::vector<V3> points;
            for (int f : faces) {
                const Vector3u t = mesh.triIndexes[f];
                points.insert(points.end(),
                              {mesh.vertexLocations[t.x()], mesh.vertexLocations[t.y()], mesh.vertexLocations[t.z()]});
            }
            Sphere s = SphereOf(points);
            float e = 0.0f;
            if (k > 0) {
                std::vector<int> childFaces;
                for (int c : children[u]) {
                    childFaces.insert(childFaces.end(), facesOf[k - 1][c].begin(), facesOf[k - 1][c].end());
                    const int cu = unitOf(k - 1, c);
                    s = Merge(s, sphere[k - 1].at(cu));
                    e = std::max(e, unitError[k - 1].at(cu));
                }
                const Mesh& below = levels[k - 1].simplified;
                e = std::max(e, OneSidedDistance(below, childFaces, mesh, faces));
                e = std::max(e, OneSidedDistance(mesh, faces, below, childFaces));
            }
            spheres[i] = s;
            errors[i] = e;
        }
        for (size_t i = 0; i < units.size(); i++) {
            sphere[k][units[i]] = spheres[i];
            unitError[k][units[i]] = errors[i];
        }
    }

    // Groups: per depth k < top, the level-k nodes under each level k+1 unit;
    // at the top, each unit's nodes (terminal).
    const auto toBounds = [](const Sphere& s, float error) {
        clodBounds b = {};
        b.center[0] = s.center.x();
        b.center[1] = s.center.y();
        b.center[2] = s.center.z();
        b.radius = std::max(s.radius, 0.0f);
        b.error = error;
        return b;
    };
    std::vector<clodGroup> groups;
    std::vector<std::pair<int, std::vector<int>>> groupNodes;  // (depth, nodes)
    for (int k = 0; k < levelCount; k++) {
        std::map<int, std::vector<int>> byUnit;
        for (int node = 0; node < tree.nodeCount(k); node++) {
            byUnit[k < top ? tree.parent[k][node] : unitOf(k, node)].push_back(node);
        }
        for (const auto& [u, nodes] : byUnit) {
            clodGroup g = {};
            g.depth = k;
            g.simplified =
                k < top ? toBounds(sphere[k + 1].at(u), unitError[k + 1].at(u)) : toBounds(sphere[k].at(u), FLT_MAX);
            groups.push_back(g);
            groupNodes.push_back({k, nodes});
        }
    }

    // Clusters: each node's faces cut into meshlets, in group order.
    for (size_t g = 0; g < groups.size(); g++) {
        const int k = groupNodes[g].first;
        const Mesh& mesh = levels[k].simplified;
        data.lodGroups.push_back({uint32_t(data.meshlets.size()), 0});
        for (int node : groupNodes[g].second) {
            const std::vector<int>& faces = facesOf[k][node];
            if (faces.empty()) continue;
            // One meshlet per region, as large as the region. Only a region
            // beyond the format's ceiling is split, into the fewest parts of
            // about equal size that each fit (partitionEvenly).
            std::vector<FaceMeshlets> pieces;
            FaceMeshlets whole = WholeMeshlet(mesh, cornerVertex[k], faces);
            if (!whole.meshlets.empty()) {
                pieces.push_back(std::move(whole));
            } else {
                for (int parts = 2; parts <= static_cast<int>(faces.size()); parts++) {
                    std::vector<FaceMeshlets> attempt;
                    bool fits = true;
                    for (const std::vector<int>& part : PartitionEvenly(mesh, faces, parts)) {
                        if (part.empty()) continue;
                        attempt.push_back(WholeMeshlet(mesh, cornerVertex[k], part));
                        fits &= !attempt.back().meshlets.empty();
                    }
                    if (fits) {
                        pieces = std::move(attempt);
                        break;
                    }
                }
            }
            const int u = unitOf(k, node);
            for (FaceMeshlets& piece : pieces)
                for (meshopt_Meshlet m : piece.meshlets) {
                    meshopt_optimizeMeshlet(&piece.vertices[m.vertex_offset], &piece.triangles[m.triangle_offset],
                                            m.triangle_count, m.vertex_count);
                    meshopt_Meshlet out = m;
                    out.vertex_offset = static_cast<unsigned>(data.vertices.size());
                    out.triangle_offset = static_cast<unsigned>(data.triangles.size());
                    for (unsigned v = 0; v < m.vertex_count; v++) {
                        data.vertices.push_back(vertexBase[k] + piece.globalOf[piece.vertices[m.vertex_offset + v]]);
                    }
                    data.triangles.insert(data.triangles.end(), piece.triangles.begin() + m.triangle_offset,
                                          piece.triangles.begin() + m.triangle_offset + m.triangle_count * 3);
                    data.meshlets.push_back(out);

                    mlod::MeshletLod lod = {};
                    if (k > 0) {
                        const clodBounds self = toBounds(sphere[k].at(u), unitError[k].at(u));
                        std::memcpy(lod.selfCenter, self.center, sizeof(lod.selfCenter));
                        lod.selfRadius = self.radius;
                        lod.selfError = self.error;
                    }
                    std::memcpy(lod.coarseCenter, groups[g].simplified.center, sizeof(lod.coarseCenter));
                    lod.coarseRadius = groups[g].simplified.radius;
                    lod.coarseError = groups[g].simplified.error;
                    lod.level = static_cast<uint32_t>(k);
                    data.lods.push_back(lod);
                }
        }
        data.lodGroups.back().clusterCount =
            static_cast<uint32_t>(data.meshlets.size()) - data.lodGroups.back().firstCluster;
    }
    if (data.meshlets.empty()) {
        error = "no meshlets";
        return false;
    }
    // The file's meshlet sizes are the largest meshlets it holds (one per
    // region), not limits the meshlets were cut to.
    settings.maxVertices = 1;
    settings.maxTriangles = 1;
    for (const meshopt_Meshlet& m : data.meshlets) {
        settings.maxVertices = std::max(settings.maxVertices, static_cast<int>(m.vertex_count));
        settings.maxTriangles = std::max(settings.maxTriangles, static_cast<int>(m.triangle_count));
    }
    settings.minTriangles = std::min(settings.minTriangles, settings.maxTriangles);
    data.triangles.resize((data.triangles.size() + 3) & ~size_t(3));

    // Selection BVH: 8 wide, one tree per depth (as the viewer builds it).
    const size_t nodeWidth = 8;
    std::vector<clodNode> nodes(clodBuildHierarchyBound(groups.size(), nodeWidth, levelCount));
    nodes.resize(clodBuildHierarchy(nodes.data(), groups.data(), groups.size(), nodeWidth, levelCount));
    data.lodNodes.resize(nodes.size());
    for (size_t i = 0; i < nodes.size(); i++) {
        const clodNode& n = nodes[i];
        mlod::LodNode& o = data.lodNodes[i];
        std::memcpy(o.center, n.bounds.center, sizeof(o.center));
        o.radius = n.bounds.radius;
        o.error = n.bounds.error;
        o.group = n.group;
        o.childOffset = n.group < 0 ? n.child_offset : 0;
        o.childCount = n.group < 0 ? n.child_count : 0;
    }
    data.lodRootCount = static_cast<uint32_t>(levelCount);
    {
        std::vector<uint32_t> frontier, next;
        for (uint32_t root = 0; root < data.lodRootCount; root++) frontier.push_back(root);
        while (!frontier.empty()) {
            data.lodNodesPerDepth.push_back(static_cast<uint32_t>(frontier.size()));
            next.clear();
            for (uint32_t index : frontier) {
                const mlod::LodNode& n = data.lodNodes[index];
                for (uint32_t c = 0; c < n.childCount; c++) next.push_back(n.childOffset + c);
            }
            frontier.swap(next);
        }
    }

    // Cone bounds and perimeter edge masks per meshlet, then statistics (as
    // the viewer computes them).
    const size_t count = data.meshlets.size();
    data.bounds.resize(count);
    data.edgeMasks.assign(((data.triangles.size() + 2) / 3 + 3) & ~size_t(3), 0);
    std::vector<uint32_t> edges, sorted;
    for (size_t i = 0; i < count; i++) {
        const meshopt_Meshlet& m = data.meshlets[i];
        const uint8_t* tri = &data.triangles[m.triangle_offset];
        edges.clear();
        for (uint32_t t = 0; t < m.triangle_count * 3; t += 3) {
            for (uint32_t k = 0; k < 3; k++) {
                const uint32_t a = tri[t + (k + 1) % 3], b = tri[t + (k + 2) % 3];
                edges.push_back(std::min(a, b) << 8 | std::max(a, b));
            }
        }
        sorted = edges;
        std::sort(sorted.begin(), sorted.end());
        for (uint32_t t = 0; t < m.triangle_count; t++) {
            uint8_t mask = 0;
            for (uint32_t k = 0; k < 3; k++) {
                const auto range = std::equal_range(sorted.begin(), sorted.end(), edges[t * 3 + k]);
                if (range.second - range.first == 1) mask |= uint8_t(1u << k);
            }
            data.edgeMasks[m.triangle_offset / 3 + t] = mask;
        }
        const meshopt_Bounds b = meshopt_computeMeshletBounds(
            &data.vertices[m.vertex_offset], &data.triangles[m.triangle_offset], m.triangle_count,
            &model.vertices[0].px, model.vertices.size(), sizeof(mlod::Vertex));
        mlod::MeshletBounds& gb = data.bounds[i];
        std::memcpy(gb.center, b.center, sizeof(gb.center));
        gb.radius = b.radius;
        std::memcpy(gb.coneAxis, b.cone_axis, sizeof(gb.coneAxis));
        gb.coneCutoff = b.cone_cutoff;
    }
    size_t baseClusters = 0, baseVertices = 0;
    double coneAngles = 0.0;
    for (size_t i = 0; i < count; i++) {
        const meshopt_Meshlet& m = data.meshlets[i];
        const mlod::MeshletLod& lod = data.lods[i];
        data.triangleCount += m.triangle_count;
        if (lod.level >= data.levelClusters.size()) {
            data.levelClusters.resize(lod.level + 1);
            data.levelTriangles.resize(lod.level + 1);
        }
        data.levelClusters[lod.level]++;
        data.levelTriangles[lod.level] += m.triangle_count;
        if (lod.coarseError == FLT_MAX) data.coarsestTriangles += m.triangle_count;
        if (lod.level != 0) continue;
        baseClusters++;
        baseVertices += m.vertex_count;
        data.baseTriangleCount += m.triangle_count;
        const float cutoff = data.bounds[i].coneCutoff;
        if (cutoff < 1.f) {
            data.coneCullable++;
            coneAngles += 2.0 * std::asin(double(std::clamp(cutoff, 0.f, 1.f))) * 180.0 / 3.14159265358979323846;
        }
    }
    data.avgVertices = float(double(baseVertices) / double(baseClusters));
    data.avgTriangles = float(double(data.baseTriangleCount) / double(baseClusters));
    data.vertexFill = data.avgVertices / float(settings.maxVertices);
    data.triangleFill = data.avgTriangles / float(settings.maxTriangles);
    data.avgConeAngleDeg = data.coneCullable ? float(coneAngles / double(data.coneCullable)) : 0.f;
    data.coneCullableFraction = float(double(data.coneCullable) / double(baseClusters));
    data.buildSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    // Write next to the target and rename, as the viewer does.
    const std::filesystem::path temp = file.string() + ".tmp";
    try {
        std::error_code ec;
        std::filesystem::create_directories(file.parent_path(), ec);
        {
            std::ofstream out(temp, std::ios::binary | std::ios::trunc);
            if (!out) {
                error = "cannot open " + temp.string() + " for writing";
                return false;
            }
            mlod::OutputArchive ar(out);
            uint32_t magic = mlod::kMagic, version = textured ? mlod::kTexcoordVersion : mlod::kVersion;
            ar(magic, version);
            mlod::Archive(ar, version, model, settings, data);
            out.flush();
            if (!out) {
                error = "write failed: " + temp.string();
                return false;
            }
        }
        std::filesystem::rename(temp, file);
    } catch (const std::exception& e) {
        std::error_code ignored;
        std::filesystem::remove(temp, ignored);
        error = "saving " + file.string() + ": " + e.what();
        return false;
    }
    if (stats) {
        stats->vertices = model.vertices.size();
        stats->clusters = data.meshlets.size();
        stats->groups = groups.size();
        stats->triangles = data.triangleCount;
        stats->levels = levelCount;
    }
    return true;
}
