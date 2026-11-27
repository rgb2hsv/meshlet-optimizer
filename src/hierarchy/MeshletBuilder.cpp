#include <core/Check.hpp>
#include <core/TriangleGeometry.hpp>
#include <core/Types.hpp>
#include <flattening/ChartBinPack.hpp>
#include <flattening/ChartFlattening.hpp>
#include <hierarchy/BorderSimplification.hpp>
#include <hierarchy/DiskGroups.hpp>
#include <hierarchy/LevelCracks.hpp>
#include <hierarchy/MeshletBuilder.hpp>
#include <hierarchy/RegionHierarchy.hpp>
#include <hierarchy/RegionSplit.hpp>
#include <hierarchy/SeamStraightening.hpp>
#include <mlod/MlodWriter.hpp>

#include <LBFGSB.h>
#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

using namespace mopt;

namespace mopt
{
// Working state of the level being built, between the level stages.
struct LevelWork {
    int level = 0;
    // The previous level's simplified mesh, cut and flattened for this level.
    std::unique_ptr<ChartLayout> layout;
    std::unique_ptr<RegionResampler> resampler;
    ResampledSurface surface;
};
}  // namespace mopt

namespace
{

// Faces joined across their shared edges (both faces of a manifold edge), as a union-find; `joined(f, g)` decides.
template <class Joined>
std::vector<int> FaceComponents(const Mesh& mesh, const Joined& joined)
{
    const size_t F = mesh.triIndexes.size();
    std::vector<int> parent(F);
    for (size_t f = 0; f < F; f++) parent[f] = static_cast<int>(f);
    const auto find = [&](int f) {
        while (parent[f] != f) f = parent[f] = parent[parent[f]];
        return f;
    };
    std::unordered_map<uint64_t, int> firstFace;
    firstFace.reserve(3 * F);
    for (size_t f = 0; f < F; f++) {
        const Vector3u t = mesh.triIndexes[f];
        for (int k = 0; k < 3; k++) {
            const uint32_t a = std::min(t[k], t[(k + 1) % 3]), b = std::max(t[k], t[(k + 1) % 3]);
            const auto [it, inserted] = firstFace.emplace((uint64_t(a) << 32) | b, static_cast<int>(f));
            if (!inserted && joined(it->second, static_cast<int>(f), a, b))
                parent[find(static_cast<int>(f))] = find(it->second);
        }
    }
    // Components numbered in order of their first face.
    std::vector<int> id(F, -1), component(F);
    int count = 0;
    for (size_t f = 0; f < F; f++) {
        const int root = find(static_cast<int>(f));
        if (id[root] < 0) id[root] = count++;
        component[f] = id[root];
    }
    return component;
}

// Texture coordinates of vertex v in face f.
Vector2f CornerTexcoord(const Mesh& mesh, int f, uint32_t v)
{
    const Vector3u t = mesh.triIndexes[f];
    return mesh.texcoords[3 * f + (t[0] == v ? 0 : t[1] == v ? 1 : 2)];
}

// UV island of each face: faces joined across every edge whose ends have the same texture coordinates on both
// sides. Without texture coordinates the whole mesh is island 0.
std::vector<int> FaceIslands(const Mesh& mesh)
{
    if (!mesh.hasTexcoords()) return std::vector<int>(mesh.triIndexes.size(), 0);
    return FaceComponents(mesh, [&](int f, int g, uint32_t a, uint32_t b) {
        return CornerTexcoord(mesh, f, a) == CornerTexcoord(mesh, g, a) &&
               CornerTexcoord(mesh, f, b) == CornerTexcoord(mesh, g, b);
    });
}

// Island of every region (all its faces share one).
std::vector<int> RegionIslands(const std::vector<int>& faceRegion, const std::vector<int>& faceIsland, int regionCount)
{
    std::vector<int> island(regionCount, -1);
    for (size_t f = 0; f < faceRegion.size(); f++) {
        int& i = island[faceRegion[f]];
        Expects(i < 0 || i == faceIsland[f], "region {} spans UV islands {} and {}", faceRegion[f], i, faceIsland[f]);
        i = faceIsland[f];
    }
    return island;
}

// Only the adjacency between regions of the same island: regions never merge across a texture seam.
std::vector<std::vector<int>> SameIsland(std::vector<std::vector<int>> adjacency, const std::vector<int>& island)
{
    for (size_t r = 0; r < adjacency.size(); r++) {
        std::erase_if(adjacency[r], [&](int s) { return island[s] != island[r]; });
    }
    return adjacency;
}

// Texture coordinates at p on triangle (a, b, c) with corner texture coordinates (ua, ub, uc), from the
// closest point; returns the squared distance to it.
float TexcoordOnTriangle(const Vector3f& p, const Vector3f& a, const Vector3f& b, const Vector3f& c, const Vector2f& ua,
                         const Vector2f& ub, const Vector2f& uc, Vector2f& uv)
{
    Vector3f closest;
    const float d = DistanceToTriangleSqr(p, a, b, c, &closest);
    const Vector3f n = (b - a).cross(c - a);
    const float area = n.squaredNorm();
    if (!(area > 0.0f)) {
        uv = ua;
        return d;
    }
    const float wb = (closest - a).cross(c - a).dot(n) / area;
    const float wc = (b - a).cross(closest - a).dot(n) / area;
    uv = (1.0f - wb - wc) * ua + wb * ub + wc * uc;
    return d;
}

// Every level's vertex positions, to find the vertices a stage moved.
std::vector<std::vector<Vector3f>> PositionsOf(const std::vector<LodLevel>& levels)
{
    std::vector<std::vector<Vector3f>> positions;
    for (const LodLevel& level : levels) positions.push_back(level.simplified.vertexLocations);
    return positions;
}

// After vertices of a level moved within the surface (the triangles unchanged): each moved vertex's texture
// coordinates, in every region around it, from the closest point on its triangles in that region as they
// were before the move.
void ReprojectTexcoords(LodLevel& level, const std::vector<Vector3f>& before)
{
    Mesh& m = level.simplified;
    if (!m.hasTexcoords()) return;
    const size_t V = std::min(m.vertexLocations.size(), before.size());
    std::vector<char> moved(V, 0);
    bool any = false;
    for (size_t v = 0; v < V; v++) any |= (moved[v] = m.vertexLocations[v] != before[v]) != 0;
    if (!any) return;
    std::vector<std::vector<int>> facesOf(V);
    for (size_t f = 0; f < m.triIndexes.size(); f++) {
        for (int k = 0; k < 3; k++) {
            const uint32_t v = m.triIndexes[f][k];
            if (v < V && moved[v]) facesOf[v].push_back(static_cast<int>(f));
        }
    }
    const std::vector<Vector2f> old = m.texcoords;
    const std::vector<int>& region = level.simplifiedFaceRegion;
#pragma omp parallel for schedule(dynamic, 64)
    for (int v = 0; v < static_cast<int>(V); v++) {
        if (!moved[v]) continue;
        const Vector3f p = m.vertexLocations[v];
        const auto cornerOf = [&](int f) {
            const Vector3u t = m.triIndexes[f];
            return 3 * f + (t[0] == static_cast<uint32_t>(v) ? 0 : t[1] == static_cast<uint32_t>(v) ? 1 : 2);
        };
        for (int f : facesOf[v]) {
            float best = std::numeric_limits<float>::max();
            Vector2f uv = old[cornerOf(f)];
            for (int g : facesOf[v]) {
                // Only faces on the same side of any texture seam through v: same region and same UV at v.
                if (region[g] != region[f] || old[cornerOf(g)] != old[cornerOf(f)]) continue;
                const Vector3u t = m.triIndexes[g];
                Vector2f candidate;
                const float d = TexcoordOnTriangle(p, before[t[0]], before[t[1]], before[t[2]], old[3 * g],
                                                   old[3 * g + 1], old[3 * g + 2], candidate);
                if (d < best) {
                    best = d;
                    uv = candidate;
                }
            }
            m.texcoords[cornerOf(f)] = uv;
        }
    }
}

// A vertex belongs to a region when all its faces do; otherwise it is on a
// border (-1).
std::vector<int> VertexRegionsFromFaces(const Mesh& mesh, const std::vector<int>& faceRegion)
{
    std::vector<int> vertexRegion(mesh.vertexLocations.size(), -2);
    for (size_t f = 0; f < mesh.triIndexes.size(); f++) {
        const Vector3u t = mesh.triIndexes[f];
        const int g = faceRegion[f];
        for (uint32_t v : {t.x(), t.y(), t.z()}) {
            int& r = vertexRegion[v];
            r = r == -2 ? g : (r == g ? r : -1);
        }
    }
    for (int& r : vertexRegion) r = std::max(r, -1);
    return vertexRegion;
}
int TopLevel(const MeshletBuilder& builder) { return builder.regionHierarchy.levelCount() - 1; }

LevelWork& WorkAt(MeshletBuilder& builder, const char* step, bool needsResampler)
{
    LevelWork& work = builder.levelWork();
    Expects(work.layout != nullptr && work.level == TopLevel(builder), "{} needs LevelCut to have run for level {}",
            step, TopLevel(builder));
    if (needsResampler) {
        Expects(work.resampler != nullptr, "{} needs LevelBlueNoise to have run", step);
    }
    return work;
}
}  // namespace

namespace mopt
{
MeshletBuilder::MeshletBuilder(const MeshletParams& params) : params_(params) {}

MeshletBuilder::~MeshletBuilder() = default;

LevelWork& MeshletBuilder::levelWork()
{
    if (!levelWork_) levelWork_ = std::make_unique<LevelWork>();
    return *levelWork_;
}

void MeshletBuilder::build(const Mesh& input, const StatusHandler& status)
{
    mesh = input;
    pinned.clear();
    levels.clear();
    regionHierarchy = RegionHierarchy{};
    exportReport = ExportReport{};
    levelWork_.reset();
    bordersState_ = std::make_shared<LevelBordersState>();
    // Runs one stage and reports it with its duration, unless it had nothing to do (returned false).
    const auto stage = [&status](const std::string& name, const auto& run) {
        const auto start = std::chrono::steady_clock::now();
        const bool result = run();
        if (result && status) {
            status(name, std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        }
        return result;
    };

    stage("Base charts", [&] {
        buildBaseCharts();
        buildBaseLevel();
        if (params_.straightenBaseSeams) straightenBaseSeams();
        return true;
    });
    // Levels until at most stopAtRegions regions are left, or a level is no coarser than the one below it
    // (every vertex already on a border: e.g. a torus, which disk-shaped regions cannot cover with two).
    const auto keepGoing = [this] {
        const bool coarser = levels.size() < 2 || levels.back().simplified.vertexLocations.size() <
                                                      levels[levels.size() - 2].simplified.vertexLocations.size();
        return coarser && regionHierarchy.topNodeCount() > params_.stopAtRegions;
    };
    for (int pass = 0; pass < params_.maxDepth && keepGoing(); pass++) {
        const bool built = stage("Level " + std::to_string(pass + 1), [&] {
            if (!clusterLevel()) return false;
            cutLevel();
            flattenLevel();
            sampleLevel();
            relaxLevel();
            triangulateLevel();
            liftLevel();
            projectLevel();
            reconstituteLevel();
            if (params_.splitAboveFaces > 0) splitLevel();
            if (params_.simplifyBorders) simplifyLevelBorders();
            return true;
        });
        if (!built) break;
    }
    if (params_.simplifyBorders) {
        stage("Flat triangles", [&] { return fixFlatTriangles(); });
        if (params_.relaxBorders) stage("Border relaxation", [&] { return relaxBorders(); });
    }
    stage("UV atlases", [&] { return buildUvAtlases(); });
    if (params_.isometricUvs) stage("Isometric UVs", [&] { return makeIsometricUvs(); });
    stage("Write " + mlodFile_.string(), [&] { return writeMlod(); });
}
}  // namespace mopt

bool MeshletBuilder::buildBaseCharts()
{
    MeshletBuilder& builder = *this;
    CvtSegmentation cvt(builder.mesh);
    builder.chartCount = cvt.solve(params_.cvt, builder.vertLabels);
    builder.faceLabels.assign(builder.mesh.triIndexes.size(), -1);
    MakeOrderedVertexNeighborTable(builder.mesh, builder.vertNbrs);
    return true;
}

bool MeshletBuilder::buildBaseLevel()
{
    MeshletBuilder& builder = *this;
    int regionCount = 0;
    for (int l : builder.vertLabels) regionCount = std::max(regionCount, l + 1);
    std::vector<int> weights(regionCount, 0);
    for (int l : builder.vertLabels) {
        if (l >= 0) weights[l]++;
    }
    // Charts as sets of faces: each face goes to the builder holding most of its
    // corners (ties: lower id). Borders then run along existing mesh edges,
    // so no level ever adds border vertices, and every border a parent shares
    // with a neighbour is exactly the border its children share: levels can
    // be mixed without cracks.
    const Mesh& mesh = builder.mesh;
    std::vector<int> faceRegion(mesh.triIndexes.size(), -1);
    for (size_t f = 0; f < mesh.triIndexes.size(); f++) {
        const Vector3u t = mesh.triIndexes[f];
        const int l[3] = {builder.vertLabels[t.x()], builder.vertLabels[t.y()], builder.vertLabels[t.z()]};
        int best = -1, bestCount = 0;
        for (int k = 0; k < 3; k++) {
            if (l[k] < 0) continue;
            const int count = (l[0] == l[k]) + (l[1] == l[k]) + (l[2] == l[k]);
            if (count > bestCount || (count == bestCount && l[k] < best)) {
                best = l[k];
                bestCount = count;
            }
        }
        faceRegion[f] = best;
    }
    // Isolated triangles (no edge neighbour in their own builder) join the
    // smallest neighbouring builder. A builder left without faces is dropped and
    // the rest renumbered; its vertices take the builder its faces joined.
    const std::vector<int> labelled = faceRegion;
    const int isolated = RelabelIsolatedFaces(mesh, faceRegion);
    if (isolated > 0) {
        std::vector<int> faces(regionCount, 0);
        for (int r : faceRegion) {
            if (r >= 0) faces[r]++;
        }
        std::vector<int> joined(regionCount, -1);  // an emptied builder -> where its faces went
        for (size_t f = 0; f < faceRegion.size(); f++) {
            if (labelled[f] >= 0 && faces[labelled[f]] == 0) joined[labelled[f]] = faceRegion[f];
        }
        std::vector<int> remap(regionCount, -1);
        int kept = 0;
        for (int r = 0; r < regionCount; r++) {
            if (faces[r] > 0) remap[r] = kept++;
        }
        for (int r = 0; r < regionCount; r++) {
            if (remap[r] < 0 && joined[r] >= 0) remap[r] = remap[joined[r]];
        }
        if (kept < regionCount) {
            for (int& r : faceRegion) {
                if (r >= 0) r = remap[r];
            }
            for (int& l : builder.vertLabels) {
                if (l >= 0) l = remap[l];
            }
            weights.assign(kept, 0);
            for (int l : builder.vertLabels) {
                if (l >= 0) weights[l]++;
            }
            regionCount = kept;
            builder.chartCount = kept;
        }
    }
    // With texture coordinates no region may span a UV island: the charts are split along the texture seams
    // (and into edge-connected pieces), so every region of every level stays inside one island.
    const std::vector<int> faceIsland = FaceIslands(mesh);
    if (mesh.hasTexcoords()) {
        faceRegion = FaceComponents(mesh, [&](int f, int g, uint32_t, uint32_t) {
            return faceRegion[f] == faceRegion[g] && faceIsland[f] == faceIsland[g];
        });
        regionCount = 0;
        for (int r : faceRegion) regionCount = std::max(regionCount, r + 1);
        weights.assign(regionCount, 0);
        for (int r : faceRegion) weights[r]++;
        builder.chartCount = regionCount;
    }
    nodeIsland_.assign(1, RegionIslands(faceRegion, faceIsland, regionCount));
    std::vector<int> vertexRegion = VertexRegionsFromFaces(mesh, faceRegion);
    // Charts are adjacent when an edge has one face in each (and both lie in the same island).
    const std::vector<std::vector<int>> adjacency =
        SameIsland(FaceRegionAdjacency(mesh, faceRegion, regionCount), nodeIsland_[0]);
    builder.regionHierarchy =
        InitRegionHierarchy(adjacency, weights, params_.randomSeed, params_.maxCliqueSize, params_.absorbSingletons);
    builder.levels.clear();
    LodLevel base;
    base.level = 0;
    base.simplified = builder.mesh;
    base.simplifiedFaceRegion = std::move(faceRegion);
    base.simplifiedVertexRegion = std::move(vertexRegion);
    builder.levels.push_back(std::move(base));
    builder.levelWork() = LevelWork{};
    return true;
}

bool MeshletBuilder::clusterLevel()
{
    MeshletBuilder& builder = *this;
    RegionHierarchy& tree = builder.regionHierarchy;
    Expects(tree.levelCount() > 0, "ClusterLevel needs BaseLevel to have run");
    if (!AddRegionHierarchyLevel(tree)) return false;
    const int k = TopLevel(builder);
    if (!builder.levels.empty()) {
        // A group with no border at all covers a whole closed surface piece,
        // which cannot be flattened: it goes back to its members. If nothing
        // is merged any more, the level is undone and the levels stop here.
        const LodLevel& prev = builder.levels.back();
        const int regionCount = tree.nodeCount(k - 1);
        const std::vector<int>& parent = tree.parent[k - 1];
        std::vector<char> hasBorder(tree.nodeCount(k), 0);
        std::unordered_map<uint64_t, int> edgeGroup;
        const Mesh& mesh = prev.simplified;
        for (size_t f = 0; f < mesh.triIndexes.size(); f++) {
            const int g = parent[prev.simplifiedFaceRegion[f]];
            const Vector3u t = mesh.triIndexes[f];
            const uint32_t v[3] = {t.x(), t.y(), t.z()};
            for (int e = 0; e < 3; e++) {
                const uint32_t a = std::min(v[e], v[(e + 1) % 3]), b = std::max(v[e], v[(e + 1) % 3]);
                const auto [it, inserted] = edgeGroup.emplace((uint64_t(a) << 32) | b, g);
                if (!inserted && it->second != g) hasBorder[g] = hasBorder[it->second] = 1;
            }
        }
        std::vector<int> group = parent;
        std::vector<char> absorbed = tree.absorbed[k - 1];
        int dissolved = 0;
        int nextGroup = tree.nodeCount(k);
        for (int node = 0; node < regionCount; node++) {
            if (tree.groupSize[k][parent[node]] < 2 || hasBorder[parent[node]]) continue;
            group[node] = nextGroup++;  // its own group again
            absorbed[node] = 0;
            dissolved++;
        }
        if (dissolved > 0) {
            const std::vector<std::vector<int>> adjacency = SameIsland(
                FaceRegionAdjacency(prev.simplified, prev.simplifiedFaceRegion, regionCount), nodeIsland_[k - 1]);
            RegroupTopLevel(tree, group, absorbed, adjacency);
            if (tree.nodeCount(k) == regionCount) {
                RemoveTopLevel(tree, adjacency);
                return false;
            }
        }
    }
    if (params_.keepDisks && !builder.levels.empty()) {
        // Groups of the level below's (remeshed) regions, made disk-shaped:
        // small holes filled, then members shed (DiskGroups.hpp).
        const LodLevel& prev = builder.levels.back();
        const int regionCount = tree.nodeCount(k - 1);
        const std::vector<std::vector<int>> adjacency = SameIsland(
            FaceRegionAdjacency(prev.simplified, prev.simplifiedFaceRegion, regionCount), nodeIsland_[k - 1]);
        std::vector<int> group = tree.parent[k - 1];
        std::vector<char> absorbed = tree.absorbed[k - 1];
        DiskGroupParams params;
        params.maxHoleFraction = params_.maxHoleFraction;
        const bool changed = MakeGroupsDisks(prev.simplified, prev.simplifiedFaceRegion, regionCount, adjacency,
                                             tree.weight[k - 1], group, absorbed, params, nodeIsland_[k - 1]);
        if (changed) RegroupTopLevel(tree, group, absorbed, adjacency);
    }
    // Each new node lies in the island of its members.
    nodeIsland_.resize(k + 1);
    nodeIsland_[k].assign(tree.nodeCount(k), -1);
    for (int node = 0; node < tree.nodeCount(k - 1); node++) {
        int& island = nodeIsland_[k][tree.parent[k - 1][node]];
        Expects(island < 0 || island == nodeIsland_[k - 1][node], "level {} node {} spans UV islands", k,
                tree.parent[k - 1][node]);
        island = nodeIsland_[k - 1][node];
    }
    return true;
}

bool MeshletBuilder::cutLevel()
{
    MeshletBuilder& builder = *this;
    const RegionHierarchy& tree = builder.regionHierarchy;
    const int k = TopLevel(builder);
    Expects(k >= 1 && static_cast<int>(builder.levels.size()) == k,
            "LevelCut needs a new cluster level on top of {} stored levels", builder.levels.size());
    const LodLevel& prev = builder.levels.back();
    const std::vector<int>& parent = tree.parent[k - 1];

    LevelWork& work = builder.levelWork();
    work = LevelWork{};
    work.level = k;
    work.layout = std::make_unique<ChartLayout>();
    ChartLayout& level = *work.layout;
    level.mesh = prev.simplified;
    Mesh& mesh = level.mesh;
    const size_t V = mesh.vertexLocations.size();
    if (mesh.vertexNormals.size() != V) mesh.updateVertexNormals();

    // Group of each vertex: its region's parent. A vertex on a previous
    // region boundary joins a group when all faces around it are in it.
    level.vertLabels.assign(V, -1);
    for (size_t v = 0; v < V; v++) {
        const int r = prev.simplifiedVertexRegion[v];
        if (r >= 0) level.vertLabels[v] = parent[r];
    }
    const std::vector<int>& faceRegion = prev.simplifiedFaceRegion;
    if (!faceRegion.empty()) {
        std::vector<int> seen(V, -2);
        for (size_t f = 0; f < mesh.triIndexes.size(); f++) {
            const int g = parent[faceRegion[f]];
            const Vector3u t = mesh.triIndexes[f];
            for (uint32_t v : {t.x(), t.y(), t.z()}) {
                if (level.vertLabels[v] >= 0) continue;
                seen[v] = seen[v] == -2 ? g : (seen[v] == g ? g : -1);
            }
        }
        for (size_t v = 0; v < V; v++) {
            if (level.vertLabels[v] < 0 && seen[v] >= 0) level.vertLabels[v] = seen[v];
        }
    }

    // Group borders run along border vertices (-1): no edge joins two groups.
    for (const Vector3u& t : mesh.triIndexes) {
        for (int e = 0; e < 3; e++) {
            const int a = level.vertLabels[t[e]], b = level.vertLabels[t[(e + 1) % 3]];
            Expects(a < 0 || b < 0 || a == b, "Level {}: an edge crosses groups {} and {}", k, a, b);
        }
    }

    // Group of each face: from a labelled corner, else (all corners on seams)
    // from the face's previous region.
    level.faceLabels.assign(mesh.triIndexes.size(), -1);
    for (size_t f = 0; f < mesh.triIndexes.size(); f++) {
        const Vector3u t = mesh.triIndexes[f];
        for (uint32_t v : {t.x(), t.y(), t.z()}) {
            if (level.vertLabels[v] >= 0) {
                level.faceLabels[f] = level.vertLabels[v];
                break;
            }
        }
        if (level.faceLabels[f] < 0 && !faceRegion.empty()) {
            level.faceLabels[f] = parent[faceRegion[f]];
        }
        Expects(level.faceLabels[f] >= 0, "Level {} face {} has no group", k, f);
    }
    level.chartCount = tree.nodeCount(k);
    return true;
}

bool MeshletBuilder::flattenLevel()
{
    flattenCharts(*WorkAt(*this, "LevelLSCM", false).layout);
    return true;
}

void MeshletBuilder::flattenCharts(ChartLayout& layout) const
{
    BuildMosaics(layout, true);
    // Pad every region with rings of its neighbours' triangles; dilation
    // appends faces and vertices, so the region's own come first.
    std::vector<size_t> ownFaces(layout.mosaics.size()), ownVertices(layout.mosaics.size());
    for (size_t i = 0; i < layout.mosaics.size(); i++) {
        ownFaces[i] = layout.mosaics[i].faces.size();
        ownVertices[i] = layout.mosaics[i].vertexes.size();
    }
    if (params_.lscmPaddingRings > 0) {
        if (layout.vertNbrs.size() != layout.mesh.vertexLocations.size()) {
            MakeOrderedVertexNeighborTable(layout.mesh, layout.vertNbrs);
        }
        DilateMosaics(layout, params_.lscmPaddingRings);
        layout.totalVertexCount = layout.textureToMeshMap.size();
    }
    BuildMosaicNeighborTables(layout);
    ProjectMosaicsToPlanes(layout);
    SolveLscm(layout, params_.lscmIterations);
    if (params_.lscmPaddingRings > 0) {
        // Drop the padding: each region keeps exactly its own triangles.
        for (size_t i = 0; i < layout.mosaics.size(); i++) {
            Mosaic& mosaic = layout.mosaics[i];
            for (size_t v = ownVertices[i]; v < mosaic.vertexes.size(); v++) {
                mosaic.vertexMap.erase(mosaic.vertexes[v].mapId);
            }
            mosaic.faces.resize(ownFaces[i]);
            mosaic.vertexes.resize(ownVertices[i]);
        }
        // UV area back in proportion to each region's own 3D area.
        RescaleMosaics(layout);
    }
}

bool MeshletBuilder::buildUvAtlases()
{
    if (mesh.hasTexcoords()) return false;
    const ChartBinPack packer(params_.packPadding);
    for (LodLevel& level : levels) {
        ChartLayout layout;
        layout.mesh = level.simplified;
        layout.mesh.texcoords.clear();
        if (layout.mesh.vertexNormals.size() != layout.mesh.vertexLocations.size()) {
            layout.mesh.updateVertexNormals();
        }
        layout.faceLabels = level.simplifiedFaceRegion;
        layout.vertLabels = level.simplifiedVertexRegion;
        for (int r : layout.faceLabels) layout.chartCount = std::max(layout.chartCount, r + 1);
        flattenCharts(layout);
        packer.pack(layout.mosaics);

        Mesh& out = level.simplified;
        out.texcoords.assign(3 * out.triIndexes.size(), Vector2f::Zero());
        std::vector<char> covered(out.triIndexes.size(), 0);
        for (const Mosaic& mosaic : layout.mosaics) {
            for (const MosaicFace& face : mosaic.faces) {
                for (int c = 0; c < 3; c++) {
                    out.texcoords[3 * face.fid + c] = mosaic.getLocalVertex(face.mapFace[c]).uv.cast<float>();
                }
                covered[face.fid] = 1;
            }
        }
        const size_t missing = std::count(covered.begin(), covered.end(), 0);
        Expects(missing == 0, "Level {}: {} faces have no chart in the UV atlas", level.level, missing);
    }
    return true;
}

bool MeshletBuilder::makeIsometricUvs()
{
    // Only the atlases buildUvAtlases() made: a mesh's own texture coordinates are kept as they are.
    if (mesh.hasTexcoords()) return false;
    IsometricUvParams uvParams;
    uvParams.relaxSteps = params_.uvRelaxSteps;
    uvParams.fitToAtlas = params_.uvFitToAtlas;
    uvParams.enforceNoOverlaps = params_.uvEnforceNoOverlaps;
    bool any = false;
    for (LodLevel& level : levels) {
        if (!level.simplified.hasTexcoords()) continue;
        level.atlasUvs.resize(level.simplified.texcoords.size());
        for (size_t c = 0; c < level.atlasUvs.size(); c++) level.atlasUvs[c] = level.simplified.texcoords[c].cast<double>();
        MakeIsometricUvs(level.simplified, level.simplifiedFaceRegion, level.isometricUvs, uvParams);
        for (size_t c = 0; c < level.isometricUvs.size(); c++) level.simplified.texcoords[c] = level.isometricUvs[c].cast<float>();
        any = true;
    }
    return any;
}

bool MeshletBuilder::sampleLevel()
{
    MeshletBuilder& builder = *this;
    LevelWork& work = WorkAt(builder, "LevelBlueNoise", false);
    work.resampler = std::make_unique<RegionResampler>(work.layout->mesh, work.layout->mosaics, params_.resample);
    work.resampler->sampleBlueNoise();
    return true;
}

bool MeshletBuilder::relaxLevel()
{
    MeshletBuilder& builder = *this;
    WorkAt(builder, "LevelCVT", true).resampler->relaxCvt();
    return true;
}

bool MeshletBuilder::triangulateLevel()
{
    MeshletBuilder& builder = *this;
    WorkAt(builder, "LevelTriangulate", true).resampler->triangulate();
    return true;
}

bool MeshletBuilder::liftLevel()
{
    MeshletBuilder& builder = *this;
    WorkAt(builder, "LevelLift", true).resampler->lift();
    return true;
}

bool MeshletBuilder::projectLevel()
{
    MeshletBuilder& builder = *this;
    WorkAt(builder, "LevelProject", true).resampler->project(builder.projectToSurface);
    return true;
}

namespace
{
}  // namespace

bool MeshletBuilder::reconstituteLevel()
{
    MeshletBuilder& builder = *this;
    LevelWork& work = WorkAt(builder, "LevelReconstitute", true);
    work.surface = work.resampler->reconstitute();
    // Region index -> group (mosaic id).
    const std::vector<Mosaic>& mosaics = work.layout->mosaics;
    for (int& r : work.surface.faceRegion) r = mosaics[r].id;
    for (int& r : work.surface.vertexRegion) {
        if (r >= 0) r = mosaics[r].id;
    }

    LodLevel result;
    result.level = work.level;
    result.simplified = std::move(work.surface.mesh);
    result.simplifiedFaceRegion = std::move(work.surface.faceRegion);
    result.simplifiedVertexRegion = std::move(work.surface.vertexRegion);
    builder.levels.push_back(std::move(result));
    work = LevelWork{};
    return true;
}

bool MeshletBuilder::splitLevel()
{
    MeshletBuilder& builder = *this;
    RegionHierarchy& tree = builder.regionHierarchy;
    const int k = TopLevel(builder);
    Expects(k >= 1 && static_cast<int>(builder.levels.size()) == k + 1,
            "LevelSplit needs LevelReconstitute to have stored level {}", k);
    LodLevel& level = builder.levels.back();
    const int before = tree.topNodeCount();
    const size_t facesBefore = level.simplified.triIndexes.size();
    const std::vector<RegionSplit> splits =
        SplitLargeRegions(level.simplified, level.simplifiedFaceRegion, before, params_.splitAboveFaces);
    if (splits.empty()) return true;
    Expects(level.simplified.triIndexes.size() == facesBefore, "LevelSplit changed the triangle count");
    for (const RegionSplit& split : splits) {
        const int id = SplitTopNode(tree, split.region,
                                    split.movedFaces / static_cast<double>(split.keptFaces + split.movedFaces));
        Expects(id == split.newRegion, "LevelSplit: node {} for region {}", id, split.newRegion);
        nodeIsland_[k].resize(id + 1, -1);
        nodeIsland_[k][id] = nodeIsland_[k][split.region];
    }
    const int regionCount = tree.topNodeCount();
    // The new borders become border vertices (-1) like the others, so the
    // next level keeps them fixed.
    Mesh& mesh = level.simplified;
    level.simplifiedVertexRegion = VertexRegionsFromFaces(mesh, level.simplifiedFaceRegion);
    tree.topAdjacency = SameIsland(FaceRegionAdjacency(mesh, level.simplifiedFaceRegion, regionCount), nodeIsland_[k]);
    return true;
}

bool MeshletBuilder::straightenBaseSeams()
{
    MeshletBuilder& builder = *this;
    RegionHierarchy& tree = builder.regionHierarchy;
    Expects(tree.levelCount() == 1 && builder.levels.size() == 1, "BaseSeams needs BaseLevel to have run (alone)");
    LodLevel& base = builder.levels.front();
    StraightenRegionSeams(base.simplified, base.simplifiedFaceRegion, {}, nodeIsland_[0]);
    const Mesh& mesh = base.simplified;
    base.simplifiedVertexRegion = VertexRegionsFromFaces(mesh, base.simplifiedFaceRegion);
    tree.topAdjacency =
        SameIsland(FaceRegionAdjacency(mesh, base.simplifiedFaceRegion, tree.topNodeCount()), nodeIsland_[0]);
    return true;
}

// The finished levels' border vertices by position, and the removed
// vertices attached to the coarser edges that replaced them.
struct mopt::LevelBordersState {
    using Key = std::array<float, 3>;
    // One finished level as border simplification sees it: border vertices by position, faces per vertex and
    // the face normals when it was finished.
    struct Level {
        std::map<Key, uint32_t> border;  // border vertex by position
        std::vector<std::vector<int>> facesOf;
        std::vector<Vector3f> normal0;  // face normals when the level was finished
    };
    // Removed vertices of `level` under its chain edge a-b, each at its
    // parameter t along it.
    struct Attachment {
        int level = 0;
        Key a{}, b{};
        std::vector<std::pair<Key, float>> deps;
        bool alive = true;
    };
    std::vector<Level> levels;
    std::vector<Attachment> attachments;
    std::multimap<Key, int> byEnd;  // attachment ids by end position
    std::vector<std::function<void()>> undo;
    std::vector<std::pair<int, uint32_t>> moved;  // (level, vertex) moved in this trial
};

namespace
{
using BorderKey = LevelBordersState::Key;
BorderKey KeyOf(const Vector3f& p) { return {p.x(), p.y(), p.z()}; }
Vector3f PointOf(const BorderKey& k) { return Vector3f(k[0], k[1], k[2]); }

void RegisterLevel(LevelBordersState& st, const LodLevel& level)
{
    LevelBordersState::Level l;
    const Mesh& m = level.simplified;
    l.facesOf.assign(m.vertexLocations.size(), {});
    for (size_t t = 0; t < m.triIndexes.size(); t++) {
        const Vector3u tri = m.triIndexes[t];
        for (uint32_t v : {tri.x(), tri.y(), tri.z()}) l.facesOf[v].push_back(static_cast<int>(t));
        l.normal0.push_back((m.vertexLocations[tri.y()] - m.vertexLocations[tri.x()])
                                .cross(m.vertexLocations[tri.z()] - m.vertexLocations[tri.x()]));
    }
    for (size_t v = 0; v < m.vertexLocations.size(); v++) {
        if (level.simplifiedVertexRegion[v] < 0) l.border[KeyOf(m.vertexLocations[v])] = static_cast<uint32_t>(v);
    }
    st.levels.push_back(std::move(l));
}

// Move the copies of the border vertex at `from` on levels below `below` to
// `to`, and the vertices attached to edges ending there along with them.
// Every change is recorded in st.undo.
void MoveVertex(LevelBordersState& st, MeshletBuilder& builder, const BorderKey& from, const BorderKey& to, int below)
{
    if (from == to) return;
    for (int j = 0; j < below && j < static_cast<int>(st.levels.size()); j++) {
        auto& border = st.levels[j].border;
        const auto it = border.find(from);
        if (it == border.end()) continue;
        const uint32_t v = it->second;
        Vector3f& p = builder.levels[j].simplified.vertexLocations[v];
        const Vector3f old = p;
        p = PointOf(to);
        border.erase(it);
        border[to] = v;
        st.moved.push_back({j, v});
        st.undo.push_back([&st, &builder, j, v, old, from, to] {
            builder.levels[j].simplified.vertexLocations[v] = old;
            st.levels[j].border.erase(to);
            st.levels[j].border[from] = v;
        });
    }
    std::vector<int> ids;
    for (auto [it, end] = st.byEnd.equal_range(from); it != end; ++it) {
        if (st.attachments[it->second].alive && st.attachments[it->second].level < below) ids.push_back(it->second);
    }
    for (int id : ids) {
        LevelBordersState::Attachment& att = st.attachments[id];
        for (BorderKey* end : {&att.a, &att.b}) {
            if (*end != from) continue;
            *end = to;
            st.undo.push_back([end, from] { *end = from; });
        }
        const auto range = st.byEnd.equal_range(from);
        for (auto it = range.first; it != range.second; ++it) {
            if (it->second != id) continue;
            st.byEnd.erase(it);
            break;
        }
        st.byEnd.insert({to, id});
        st.undo.push_back([&st, id, from, to] {
            const auto range = st.byEnd.equal_range(to);
            for (auto it = range.first; it != range.second; ++it) {
                if (it->second != id) continue;
                st.byEnd.erase(it);
                break;
            }
            st.byEnd.insert({from, id});
        });
        const Vector3f a = PointOf(st.attachments[id].a), b = PointOf(st.attachments[id].b);
        for (size_t i = 0; i < st.attachments[id].deps.size(); i++) {
            const BorderKey old = st.attachments[id].deps[i].first;
            const float t = st.attachments[id].deps[i].second;
            const BorderKey next = KeyOf(a + t * (b - a));
            st.attachments[id].deps[i].first = next;
            st.undo.push_back([&st, id, i, old] { st.attachments[id].deps[i].first = old; });
            MoveVertex(st, builder, old, next, st.attachments[id].level);
        }
    }
}

// Did a moved vertex flip or turn one of its triangles too far?
bool FoldsAny(const LevelBordersState& st, const MeshletBuilder& builder, float cosMax)
{
    for (const auto& [j, v] : st.moved) {
        const Mesh& m = builder.levels[j].simplified;
        for (int t : st.levels[j].facesOf[v]) {
            const Vector3u tri = m.triIndexes[t];
            const Vector3f n = (m.vertexLocations[tri.y()] - m.vertexLocations[tri.x()])
                                   .cross(m.vertexLocations[tri.z()] - m.vertexLocations[tri.x()]);
            const Vector3f n0 = st.levels[j].normal0[t];
            if (n.dot(n0) <= cosMax * n.norm() * n0.norm()) return true;
        }
    }
    return false;
}
}  // namespace

bool MeshletBuilder::simplifyLevelBorders()
{
    MeshletBuilder& builder = *this;
    const int k = static_cast<int>(builder.levels.size()) - 1;
    Expects(k >= 1, "LevelBorders needs a level above the base");
    LevelBordersState& st = *bordersState_;
    if (k == 1) {
        st = LevelBordersState{};
        RegisterLevel(st, builder.levels[0]);
    }
    Expects(static_cast<int>(st.levels.size()) == k, "LevelBorders: level {} out of step", k);
    LodLevel& level = builder.levels[k];
    Mesh& mesh = level.simplified;
    const float cosMax = static_cast<float>(std::cos(params_.borders.maxNormalChangeDegrees * M_PI / 180.0));
    BorderSimplificationParams params = params_.borders;
    if (params_.snapFinerLevels) {
        params.trySnap = [&](const std::vector<std::pair<uint32_t, Vector3f>>& under, uint32_t ia, uint32_t ib) {
            // under[0] is the vertex being removed now; the others were
            // removed before, under the two chain edges that now merge.
            const BorderKey a = KeyOf(mesh.vertexLocations[ia]), b = KeyOf(mesh.vertexLocations[ib]);
            const BorderKey removed = KeyOf(mesh.vertexLocations[under[0].first]);
            st.undo.clear();
            st.moved.clear();
            // This level's attachments on the two merging edges.
            std::vector<BorderKey> deps{removed};
            std::vector<int> old;
            for (auto [it, end] = st.byEnd.equal_range(removed); it != end; ++it) {
                const LevelBordersState::Attachment& att = st.attachments[it->second];
                if (att.alive && att.level == k && (att.a == a || att.b == a || att.a == b || att.b == b)) {
                    old.push_back(it->second);
                }
            }
            for (int id : old) {
                for (const auto& [key, t] : st.attachments[id].deps) deps.push_back(key);
                st.attachments[id].alive = false;
                st.undo.push_back([&st, id] { st.attachments[id].alive = true; });
            }
            // The new attachment, its vertices moved onto a-b.
            const Vector3f pa = PointOf(a), ab = PointOf(b) - pa;
            LevelBordersState::Attachment att;
            att.level = k;
            att.a = a;
            att.b = b;
            for (const BorderKey& key : deps) {
                const float t = (PointOf(key) - pa).dot(ab) / std::max(1e-30f, ab.dot(ab));
                // A copy landing on (or past) an end of the edge would sit on
                // that end's vertex: refuse the removal instead.
                if (!(t > 1e-4f && t < 1.0f - 1e-4f)) {
                    for (auto it = st.undo.rbegin(); it != st.undo.rend(); ++it) (*it)();
                    st.undo.clear();
                    return false;
                }
                const BorderKey target = KeyOf(pa + t * ab);
                att.deps.push_back({target, t});
                MoveVertex(st, builder, key, target, k);
            }
            if (FoldsAny(st, builder, cosMax)) {
                for (auto it = st.undo.rbegin(); it != st.undo.rend(); ++it) (*it)();
                st.undo.clear();
                return false;
            }
            const int id = static_cast<int>(st.attachments.size());
            st.attachments.push_back(std::move(att));
            st.byEnd.insert({a, id});
            st.byEnd.insert({b, id});
            st.undo.clear();
            return true;
        };
    }
    const std::vector<std::vector<Vector3f>> before = PositionsOf(builder.levels);
    SimplifyRegionBorders(mesh, level.simplifiedFaceRegion, params);
    for (int j = 0; j < k; j++) ReprojectTexcoords(builder.levels[j], before[j]);
    level.simplifiedVertexRegion = VertexRegionsFromFaces(mesh, level.simplifiedFaceRegion);
    RegisterLevel(st, level);

    return true;
}

namespace
{
using V3d = Eigen::Vector3d;
V3d ToV3d(const Vector3f& p) { return V3d(p.x(), p.y(), p.z()); }

// Border edges (two faces of different regions) of a level, and its border
// junctions (3+ regions, border degree other than 2, or on an open edge).
struct LevelBorder {
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    std::vector<std::vector<uint32_t>> nbrs;
    std::vector<char> junction;
};

LevelBorder BorderOfLevel(const LodLevel& level)
{
    LevelBorder out;
    const Mesh& m = level.simplified;
    const size_t V = m.vertexLocations.size();
    std::unordered_map<uint64_t, std::vector<int>> edgeFaces;
    std::vector<std::set<int>> regions(V);
    for (size_t f = 0; f < m.triIndexes.size(); f++) {
        const Vector3u t = m.triIndexes[f];
        const uint32_t v[3] = {t.x(), t.y(), t.z()};
        for (int k = 0; k < 3; k++) {
            regions[v[k]].insert(level.simplifiedFaceRegion[f]);
            const uint32_t a = std::min(v[k], v[(k + 1) % 3]), b = std::max(v[k], v[(k + 1) % 3]);
            edgeFaces[(uint64_t(a) << 32) | b].push_back(static_cast<int>(f));
        }
    }
    out.nbrs.assign(V, {});
    out.junction.assign(V, 0);
    std::vector<char> open(V, 0);
    for (const auto& [key, fs] : edgeFaces) {
        const uint32_t a = static_cast<uint32_t>(key >> 32), b = static_cast<uint32_t>(key & 0xffffffffu);
        if (fs.size() != 2) {
            open[a] = open[b] = 1;
        } else if (level.simplifiedFaceRegion[fs[0]] != level.simplifiedFaceRegion[fs[1]]) {
            out.edges.push_back({a, b});
            out.nbrs[a].push_back(b);
            out.nbrs[b].push_back(a);
        }
    }
    for (size_t v = 0; v < V; v++) {
        if (!out.nbrs[v].empty()) out.junction[v] = open[v] || regions[v].size() >= 3 || out.nbrs[v].size() != 2;
    }
    return out;
}

}  // namespace

namespace
{
// Angles of a triangle in degrees, by corner.
std::array<double, 3> AnglesOf(const V3d& a, const V3d& b, const V3d& c)
{
    const V3d p[3] = {a, b, c};
    std::array<double, 3> out{};
    for (int k = 0; k < 3; k++) {
        const V3d u = p[(k + 1) % 3] - p[k], v = p[(k + 2) % 3] - p[k];
        const double d = u.norm() * v.norm();
        out[k] = d > 0.0 ? std::acos(std::clamp(u.dot(v) / d, -1.0, 1.0)) * 180.0 / M_PI : 0.0;
    }
    return out;
}

double SmallestOf(const V3d& a, const V3d& b, const V3d& c)
{
    const auto t = AnglesOf(a, b, c);
    return std::min({t[0], t[1], t[2]});
}

// Fix the flat caps of one level; returns the faces changed or added.
std::vector<int> FixFlatTriangles(LodLevel& level, double minAngle, double cosMax,
                                  const std::function<Vector3f(const Vector3f&)>& project)
{
    Mesh& m = level.simplified;
    std::vector<int>& region = level.simplifiedFaceRegion;
    std::set<int> changed;
    const auto P = [&](uint32_t v) { return ToV3d(m.vertexLocations[v]); };
    const auto tri = [&](int f) {
        const Vector3u t = m.triIndexes[f];
        return std::array<uint32_t, 3>{t.x(), t.y(), t.z()};
    };
    const auto normalOf = [&](uint32_t a, uint32_t b, uint32_t c) { return V3d((P(b) - P(a)).cross(P(c) - P(a))); };
    const auto good = [&](const std::array<std::array<uint32_t, 3>, 4>& faces, int n, const V3d& reference,
                          const std::vector<V3d>& extra, double better) {
        for (int i = 0; i < n; i++) {
            const auto& f = faces[i];
            const auto at = [&](uint32_t v) { return v == UINT32_MAX ? extra[0] : P(v); };
            const V3d a = at(f[0]), b = at(f[1]), c = at(f[2]);
            const V3d nn = (b - a).cross(c - a);
            if (nn.dot(reference) <= cosMax * nn.norm() * reference.norm()) return false;
            if (SmallestOf(a, b, c) <= better) return false;
        }
        return true;
    };
    bool any = true;
    for (int pass = 0; pass < 4 && any; pass++) {
        any = false;
        std::unordered_map<uint64_t, std::vector<int>> edgeFaces;
        std::unordered_set<uint64_t> edges;
        for (size_t f = 0; f < m.triIndexes.size(); f++) {
            const auto t = tri(static_cast<int>(f));
            for (int k = 0; k < 3; k++) {
                const uint32_t a = std::min(t[k], t[(k + 1) % 3]), b = std::max(t[k], t[(k + 1) % 3]);
                edgeFaces[(uint64_t(a) << 32) | b].push_back(static_cast<int>(f));
            }
        }
        std::vector<char> touched(m.triIndexes.size(), 0);
        const size_t faceCount = m.triIndexes.size();
        for (size_t fi = 0; fi < faceCount; fi++) {
            const int f = static_cast<int>(fi);
            if (touched[f]) continue;
            auto t = tri(f);
            const auto ang = AnglesOf(P(t[0]), P(t[1]), P(t[2]));
            if (std::min({ang[0], ang[1], ang[2]}) >= minAngle) continue;
            // The cap's obtuse corner m, its long edge p-q: rotate f to (p, m, q).
            const int k = static_cast<int>(std::max_element(ang.begin(), ang.end()) - ang.begin());
            if (ang[k] < 120.0) {
                continue;
            }
            const uint32_t mv = t[k], p = t[(k + 2) % 3], q = t[(k + 1) % 3];
            // f = (p, m, q) cyclic means edges p->m, m->q, q->p.
            const uint64_t key = (uint64_t(std::min(p, q)) << 32) | std::max(p, q);
            const auto& across = edgeFaces[key];
            if (across.size() != 2) {
                continue;
            }
            const int g = across[0] == f ? across[1] : across[0];
            if (touched[g]) continue;
            if (region[g] != region[f]) {
                continue;
            }
            if (m.hasTexcoords()) {
                // A texture seam can end inside a region: leave pairs that meet along one alone, as their shared
                // vertices have a different texture coordinate on each side.
                const auto cornerUv = [&](int h, uint32_t v) {
                    const auto ht = tri(h);
                    return m.texcoords[3 * h + (ht[0] == v ? 0 : ht[1] == v ? 1 : 2)];
                };
                if (cornerUv(f, p) != cornerUv(g, p) || cornerUv(f, q) != cornerUv(g, q)) continue;
            }
            const auto gt = tri(g);
            uint32_t w = UINT32_MAX;
            for (uint32_t v : gt) {
                if (v != p && v != q) w = v;
            }
            const V3d reference = normalOf(gt[0], gt[1], gt[2]);
            // Flip p-q to m-w: (m, q, w) and (w, p, m).
            const uint64_t mw = (uint64_t(std::min(mv, w)) << 32) | std::max(mv, w);
            if (!edgeFaces.count(mw)) {
                const std::array<std::array<uint32_t, 3>, 4> flip{{{mv, q, w}, {w, p, mv}, {}, {}}};
                if (good(flip, 2, reference, {}, minAngle)) {
                    if (m.hasTexcoords()) {
                        // The pair meets along no seam, so each of its vertices has one texture coordinate.
                        const auto uvOf = [&](uint32_t v) {
                            for (int h : {f, g}) {
                                const auto ht = tri(h);
                                for (int c = 0; c < 3; c++) {
                                    if (ht[c] == v) return m.texcoords[3 * h + c];
                                }
                            }
                            return Vector2f(Vector2f::Zero());
                        };
                        const Vector2f um = uvOf(mv), uq = uvOf(q), uw = uvOf(w), up = uvOf(p);
                        m.texcoords[3 * f] = um, m.texcoords[3 * f + 1] = uq, m.texcoords[3 * f + 2] = uw;
                        m.texcoords[3 * g] = uw, m.texcoords[3 * g + 1] = up, m.texcoords[3 * g + 2] = um;
                    }
                    m.triIndexes[f] = Vector3u(mv, q, w);
                    m.triIndexes[g] = Vector3u(w, p, mv);
                    touched[f] = touched[g] = 1;
                    changed.insert(f);
                    changed.insert(g);
                    any = true;
                    continue;
                }
            }
            // Split: a vertex s inside the pair, fan (p, m, s), (m, q, s),
            // (q, w, s), (w, p, s).
            V3d s3 = (P(p) + P(q) + P(w) + P(mv)) / 4.0;
            Vector3f s(static_cast<float>(s3.x()), static_cast<float>(s3.y()), static_cast<float>(s3.z()));
            if (project) s = project(s);
            const std::array<std::array<uint32_t, 3>, 4> fan{
                {{p, mv, UINT32_MAX}, {mv, q, UINT32_MAX}, {q, w, UINT32_MAX}, {w, p, UINT32_MAX}}};
            if (!good(fan, 4, reference, {ToV3d(s)}, minAngle)) {
                continue;
            }
            const uint32_t sv = static_cast<uint32_t>(m.vertexLocations.size());
            if (m.hasTexcoords()) {
                // The new vertex's texture coordinates from the closest point on the pair; the corners keep
                // their own.
                const auto uvOf = [&](uint32_t v) {
                    for (int h : {f, g}) {
                        const auto ht = tri(h);
                        for (int c = 0; c < 3; c++) {
                            if (ht[c] == v) return m.texcoords[3 * h + c];
                        }
                    }
                    return Vector2f(Vector2f::Zero());
                };
                Vector2f us, ug;
                const auto ft = tri(f);
                const float df =
                    TexcoordOnTriangle(s, m.vertexLocations[ft[0]], m.vertexLocations[ft[1]], m.vertexLocations[ft[2]],
                                       m.texcoords[3 * f], m.texcoords[3 * f + 1], m.texcoords[3 * f + 2], us);
                const float dg =
                    TexcoordOnTriangle(s, m.vertexLocations[gt[0]], m.vertexLocations[gt[1]], m.vertexLocations[gt[2]],
                                       m.texcoords[3 * g], m.texcoords[3 * g + 1], m.texcoords[3 * g + 2], ug);
                if (dg < df) us = ug;
                const Vector2f up = uvOf(p), um = uvOf(mv), uq = uvOf(q), uw = uvOf(w);
                m.texcoords[3 * f] = up, m.texcoords[3 * f + 1] = um, m.texcoords[3 * f + 2] = us;
                m.texcoords[3 * g] = um, m.texcoords[3 * g + 1] = uq, m.texcoords[3 * g + 2] = us;
                for (const Vector2f& uv : {uq, uw, us, uw, up, us}) m.texcoords.push_back(uv);
            }
            m.vertexLocations.push_back(s);
            if (m.vertexNormals.size() == sv) m.vertexNormals.push_back(m.vertexNormals[w]);
            level.simplifiedVertexRegion.push_back(region[f]);
            m.triIndexes[f] = Vector3u(p, mv, sv);
            m.triIndexes[g] = Vector3u(mv, q, sv);
            m.triIndexes.push_back(Vector3u(q, w, sv));
            m.triIndexes.push_back(Vector3u(w, p, sv));
            region.push_back(region[f]);
            region.push_back(region[f]);
            touched[f] = touched[g] = 1;
            touched.push_back(1);
            touched.push_back(1);
            for (int c : {f, g, static_cast<int>(m.triIndexes.size()) - 2, static_cast<int>(m.triIndexes.size()) - 1}) {
                changed.insert(c);
            }
            any = true;
        }
    }
    return std::vector<int>(changed.begin(), changed.end());
}
}  // namespace

bool MeshletBuilder::fixFlatTriangles()
{
    MeshletBuilder& builder = *this;
    const double cosMax = std::cos(params_.flatMaxNormalChangeDegrees * M_PI / 180.0);
    const bool withState = bordersState_ && bordersState_->levels.size() == builder.levels.size();
    for (size_t j = 0; j < builder.levels.size(); j++) {
        LodLevel& level = builder.levels[j];
        const std::vector<int> changed =
            FixFlatTriangles(level, params_.flatMinAngleDegrees, cosMax, builder.projectToSurface);
        if (withState && !changed.empty()) {
            // Changed and new faces take their own normals as reference; the
            // face lists are rebuilt.
            LevelBordersState::Level& l = bordersState_->levels[j];
            const Mesh& m = level.simplified;
            l.normal0.resize(m.triIndexes.size());
            for (int f : changed) {
                const Vector3u t = m.triIndexes[f];
                l.normal0[f] = (m.vertexLocations[t.y()] - m.vertexLocations[t.x()])
                                   .cross(m.vertexLocations[t.z()] - m.vertexLocations[t.x()]);
            }
            l.facesOf.assign(m.vertexLocations.size(), {});
            for (size_t f = 0; f < m.triIndexes.size(); f++) {
                const Vector3u t = m.triIndexes[f];
                for (uint32_t v : {t.x(), t.y(), t.z()}) l.facesOf[v].push_back(static_cast<int>(f));
            }
        }
    }
    return true;
}

bool MeshletBuilder::relaxBorders()
{
    MeshletBuilder& builder = *this;
    LevelBordersState& st = *bordersState_;
    const int levels = static_cast<int>(builder.levels.size());
    if (levels < 2 || static_cast<int>(st.levels.size()) != levels) {
        return true;
    }
    using Key = LevelBordersState::Key;
    // Every border vertex (by position) on every level is one identity.
    std::vector<LevelBorder> borders(levels);
    std::map<Key, int> ids;
    std::vector<Key> keys;
    std::vector<char> pinned;  // a junction on some level: fixed
    const auto idOf = [&](const Key& key) {
        const auto [it, added] = ids.emplace(key, static_cast<int>(keys.size()));
        if (added) {
            keys.push_back(key);
            pinned.push_back(0);
        }
        return it->second;
    };
    for (int j = 0; j < levels; j++) {
        borders[j] = BorderOfLevel(builder.levels[j]);
        const Mesh& m = builder.levels[j].simplified;
        for (size_t v = 0; v < m.vertexLocations.size(); v++) {
            if (borders[j].nbrs[v].empty()) continue;
            const int id = idOf(KeyOf(m.vertexLocations[v]));
            if (borders[j].junction[v]) pinned[id] = 1;
        }
    }
    const int I = static_cast<int>(keys.size());
    double edgeSum = 0.0;
    size_t edgeCount = 0;
    for (int j = 0; j < levels; j++) {
        const Mesh& m = builder.levels[j].simplified;
        for (const auto& [a, b] : borders[j].edges) {
            edgeSum += (m.vertexLocations[a] - m.vertexLocations[b]).norm();
            edgeCount++;
        }
    }
    const double L = edgeCount ? edgeSum / edgeCount : 1.0;
    std::vector<V3d> base(I);
    for (int id = 0; id < I; id++) base[id] = V3d(keys[id][0], keys[id][1], keys[id][2]) / L;

    // Variables: one per moving identity. An attached vertex slides along the
    // coarser edge it was snapped onto (its parameter t, kept between the
    // midpoints to its neighbours on that edge), so it stays on it; a free
    // vertex slides along its finest border, up to halfway to the next vertex
    // there that is not attached.
    struct Driven {
        int id, a, b, level, var;  // var < 0: fixed parameter t
        double t;
    };
    std::vector<Driven> driven;
    std::vector<int> drivenBy(I, -1);
    Eigen::VectorXd x0, lb, ub;
    std::vector<double> lo, hi, start;
    for (const LevelBordersState::Attachment& att : st.attachments) {
        if (!att.alive) continue;
        const auto ia = ids.find(att.a), ib = ids.find(att.b);
        if (ia == ids.end() || ib == ids.end()) continue;
        // Every attached vertex follows its edge; one that is a junction on
        // some level keeps its place t along it, the others slide.
        std::vector<std::pair<double, int>> along;  // (t, id)
        for (const auto& [key, t] : att.deps) {
            const auto it = ids.find(key);
            if (it != ids.end()) along.push_back({t, it->second});
        }
        std::sort(along.begin(), along.end());
        for (size_t q = 0; q < along.size(); q++) {
            const double t = along[q].first;
            const double before = q == 0 ? 0.0 : along[q - 1].first,
                         after = q + 1 == along.size() ? 1.0 : along[q + 1].first;
            int var = -1;
            if (!pinned[along[q].second]) {
                var = static_cast<int>(start.size());
                start.push_back(t);
                lo.push_back(0.5 * (before + t));
                hi.push_back(0.5 * (t + after));
            }
            drivenBy[along[q].second] = static_cast<int>(driven.size());
            driven.push_back({along[q].second, ia->second, ib->second, att.level, var, t});
        }
    }
    std::vector<int> drivenOrder(driven.size());
    for (size_t d = 0; d < driven.size(); d++) drivenOrder[d] = static_cast<int>(d);
    std::sort(drivenOrder.begin(), drivenOrder.end(), [&](int p, int q) { return driven[p].level > driven[q].level; });

    // A border vertex that slides along its border: its variable and the border polylines on either side.
    struct Slider {
        int id, var;
        std::vector<V3d> back, fwd;  // polylines from the vertex, each starting at it
    };
    std::vector<Slider> sliders;
    std::vector<int> sliderOf(I, -1);
    const auto arcLength = [](const std::vector<V3d>& line) {
        double l = 0.0;
        for (size_t q = 1; q < line.size(); q++) l += (line[q] - line[q - 1]).norm();
        return l;
    };
    for (int j = 0; j < levels; j++) {
        const Mesh& m = builder.levels[j].simplified;
        std::vector<int> idAt(m.vertexLocations.size(), -1);
        for (size_t v = 0; v < m.vertexLocations.size(); v++) {
            if (!borders[j].nbrs[v].empty()) idAt[v] = ids.at(KeyOf(m.vertexLocations[v]));
        }
        for (size_t v = 0; v < m.vertexLocations.size(); v++) {
            const int id = idAt[v];
            if (id < 0 || borders[j].junction[v] || pinned[id] || drivenBy[id] >= 0 || sliderOf[id] >= 0) continue;
            // Walk each way along the chain to the next vertex that is not
            // attached (a junction, or another free vertex).
            Slider sl;
            sl.id = id;
            for (int side = 0; side < 2; side++) {
                std::vector<V3d>& line = side == 0 ? sl.back : sl.fwd;
                line.push_back(base[id]);
                uint32_t prev = static_cast<uint32_t>(v), cur = borders[j].nbrs[v][side];
                while (true) {
                    line.push_back(base[idAt[cur]]);
                    const int cid = idAt[cur];
                    if (borders[j].junction[cur] || pinned[cid] || drivenBy[cid] < 0 ||
                        borders[j].nbrs[cur].size() != 2)
                        break;
                    const uint32_t next =
                        borders[j].nbrs[cur][0] == prev ? borders[j].nbrs[cur][1] : borders[j].nbrs[cur][0];
                    if (next == v) break;  // closed loop
                    prev = cur;
                    cur = next;
                }
            }
            const double lb0 = arcLength(sl.back), ub0 = arcLength(sl.fwd);
            if (lb0 <= 1e-12 || ub0 <= 1e-12) continue;
            sl.var = static_cast<int>(start.size());
            start.push_back(0.0);
            lo.push_back(-0.5 * lb0);
            hi.push_back(0.5 * ub0);
            sliderOf[id] = static_cast<int>(sliders.size());
            sliders.push_back(std::move(sl));
        }
    }
    // Interior vertices next to a border (sharing a triangle with a border
    // vertex), per level: they move in their tangent plane, at most half
    // their shortest edge along each tangent axis. They are not shared with
    // other levels' borders, so they cannot open cracks.
    struct Ring {
        int id, var;
        V3d origin, t1, t2;
    };
    std::vector<Ring> rings;
    std::vector<std::vector<int>> ringAt(levels);
    for (int j = 0; j < levels; j++) {
        const Mesh& m = builder.levels[j].simplified;
        const size_t V = m.vertexLocations.size();
        ringAt[j].assign(V, -1);
        std::vector<V3d> normal(V, V3d::Zero());
        std::vector<double> shortest(V, std::numeric_limits<double>::max());
        std::vector<char> nearBorder(V, 0);
        for (const auto& t : m.triIndexes) {
            const uint32_t v[3] = {t.x(), t.y(), t.z()};
            const V3d p[3] = {ToV3d(m.vertexLocations[t.x()]), ToV3d(m.vertexLocations[t.y()]),
                              ToV3d(m.vertexLocations[t.z()])};
            const V3d n = (p[1] - p[0]).cross(p[2] - p[0]);
            bool touches = false;
            for (int k = 0; k < 3; k++) touches |= !borders[j].nbrs[v[k]].empty();
            for (int k = 0; k < 3; k++) {
                normal[v[k]] += n;
                shortest[v[k]] =
                    std::min({shortest[v[k]], (p[(k + 1) % 3] - p[k]).norm(), (p[(k + 2) % 3] - p[k]).norm()});
                if (touches) nearBorder[v[k]] = 1;
            }
        }
        for (size_t v = 0; v < V; v++) {
            if (!nearBorder[v] || !borders[j].nbrs[v].empty() || normal[v].norm() <= 0.0) continue;
            Ring r;
            r.id = I + static_cast<int>(rings.size());
            r.origin = ToV3d(m.vertexLocations[v]) / L;
            const V3d n = normal[v].normalized();
            r.t1 = n.unitOrthogonal();
            r.t2 = n.cross(r.t1);
            r.var = static_cast<int>(start.size());
            const double bound = 0.5 * shortest[v] / L;
            for (int axis = 0; axis < 2; axis++) {
                start.push_back(0.0);
                lo.push_back(-bound);
                hi.push_back(bound);
            }
            ringAt[j][v] = r.id;
            rings.push_back(r);
        }
    }
    const int N = I + static_cast<int>(rings.size());
    base.resize(N);
    for (const Ring& r : rings) base[r.id] = r.origin;
    sliderOf.resize(N, -1);
    drivenBy.resize(N, -1);
    const int nv = static_cast<int>(start.size());
    if (nv == 0) return true;
    x0 = Eigen::Map<Eigen::VectorXd>(start.data(), nv);
    lb = Eigen::Map<Eigen::VectorXd>(lo.data(), nv);
    ub = Eigen::Map<Eigen::VectorXd>(hi.data(), nv);
    // Point at arclength s along a polyline, and the unit tangent there.
    const auto along = [](const std::vector<V3d>& line, double s, V3d& tangent) {
        for (size_t q = 1; q < line.size(); q++) {
            const V3d d = line[q] - line[q - 1];
            const double l = d.norm();
            if (l <= 0.0) continue;
            tangent = d / l;
            if (s <= l || q + 1 == line.size()) return V3d(line[q - 1] + std::min(s, l) * tangent);
            s -= l;
        }
        tangent = V3d::Zero();
        return line.front();
    };
    std::vector<V3d> tangentOf(sliders.size());
    const auto positions = [&](const Eigen::VectorXd& x, std::vector<V3d>& pos) {
        pos = base;
        for (size_t i = 0; i < sliders.size(); i++) {
            const Slider& sl = sliders[i];
            const double s = x[sl.var];
            V3d t;
            pos[sl.id] = s < 0.0 ? along(sl.back, -s, t) : along(sl.fwd, s, t);
            tangentOf[i] = s < 0.0 ? V3d(-t) : t;  // d pos / d s
        }
        for (int d : drivenOrder) {
            const Driven& dr = driven[d];
            const double t = dr.var >= 0 ? x[dr.var] : dr.t;
            pos[dr.id] = (1.0 - t) * pos[dr.a] + t * pos[dr.b];
        }
        for (const Ring& r : rings) pos[r.id] = r.origin + x[r.var] * r.t1 + x[r.var + 1] * r.t2;
    };
    // Every level's triangles with a border vertex, corners by identity (or
    // fixed, for interior vertices). Each level's energy is scaled by its mean
    // border edge length to the fourth (area times length squared), so every
    // level counts alike.
    struct Tri {
        int id[3];
        V3d fixed[3];
        double weight;
    };
    std::vector<Tri> tris;
    std::vector<std::vector<int>> trisOf(N);
    for (int j = 0; j < levels; j++) {
        const Mesh& m = builder.levels[j].simplified;
        double sum = 0.0;
        for (const auto& [a, b] : borders[j].edges) sum += (m.vertexLocations[a] - m.vertexLocations[b]).norm() / L;
        const double mean = borders[j].edges.empty() ? 1.0 : sum / borders[j].edges.size();
        const double weight = 1.0 / (mean * mean * mean * mean);
        for (const auto& t : m.triIndexes) {
            const uint32_t v[3] = {t.x(), t.y(), t.z()};
            Tri tri;
            bool any = false;
            for (int k = 0; k < 3; k++) {
                tri.id[k] = borders[j].nbrs[v[k]].empty() ? ringAt[j][v[k]] : ids.at(KeyOf(m.vertexLocations[v[k]]));
                tri.fixed[k] = ToV3d(m.vertexLocations[v[k]]) / L;
                any |= tri.id[k] >= 0;
            }
            if (!any) continue;
            tri.weight = weight;
            const int index = static_cast<int>(tris.size());
            for (int k = 0; k < 3; k++) {
                if (tri.id[k] >= 0) trisOf[tri.id[k]].push_back(index);
            }
            tris.push_back(tri);
        }
    }
    // The CVT energy of a triangle: each corner's share (the quad between it, its edge midpoints and the
    // centroid, split into two triangles) integrated against |x - corner|^2. halfQuadGrad gives the integral
    // of |x - q|^2 over the triangle (q, b, c), area / 6 (|b - q|^2 + |c - q|^2 + (b - q).(c - q)), and its
    // gradient with respect to q, b and c.
    const auto halfQuadGrad = [](const V3d& q, const V3d& b, const V3d& c, V3d& dq, V3d& db, V3d& dc) {
        const V3d bq = b - q, cq = c - q;
        const V3d w = bq.cross(cq);
        const double wn = w.norm();
        const double area = 0.5 * wn;
        const double s = bq.squaredNorm() + cq.squaredNorm() + bq.dot(cq);
        // d|bq x cq| / d bq = cq x w / |w|, d / d cq = w x bq / |w| (zero for a degenerate triangle).
        const V3d dAreaB = wn > 0.0 ? V3d(0.5 * cq.cross(w) / wn) : V3d::Zero();
        const V3d dAreaC = wn > 0.0 ? V3d(0.5 * w.cross(bq) / wn) : V3d::Zero();
        db = (s / 6.0) * dAreaB + (area / 6.0) * (2.0 * bq + cq);
        dc = (s / 6.0) * dAreaC + (area / 6.0) * (2.0 * cq + bq);
        dq = -(db + dc);
        return area / 6.0 * s;
    };
    // A triangle's energy and its gradient with respect to the corners, through the edge midpoints and centroid.
    const auto triEnergyGrad = [&](const V3d& p0, const V3d& p1, const V3d& p2, V3d grad[3]) {
        const V3d p[3] = {p0, p1, p2};
        const V3d g = (p0 + p1 + p2) / 3.0;
        V3d dG = V3d::Zero();
        grad[0] = grad[1] = grad[2] = V3d::Zero();
        double e = 0.0;
        for (int k = 0; k < 3; k++) {
            const int k1 = (k + 1) % 3, k2 = (k + 2) % 3;
            const V3d& a = p[k];
            const V3d mb = 0.5 * (a + p[k1]), mc = 0.5 * (a + p[k2]);
            V3d dq, db, dc;
            e += halfQuadGrad(a, mb, g, dq, db, dc);  // db: d/d mb, dc: d/d g
            grad[k] += dq + 0.5 * db;
            grad[k1] += 0.5 * db;
            dG += dc;
            e += halfQuadGrad(a, g, mc, dq, db, dc);  // db: d/d g, dc: d/d mc
            grad[k] += dq + 0.5 * dc;
            grad[k2] += 0.5 * dc;
            dG += db;
        }
        for (int k = 0; k < 3; k++) grad[k] += dG / 3.0;
        return e;
    };
    const auto cornerOf = [](const Tri& t, int k, const std::vector<V3d>& pos) {
        return t.id[k] >= 0 ? pos[t.id[k]] : t.fixed[k];
    };
    // Vertices whose position can change: the gradient of the others is not needed.
    std::vector<char> moves(N, 0);
    for (int id = 0; id < N; id++) moves[id] = !trisOf[id].empty() && !(id < I && sliderOf[id] < 0 && drivenBy[id] < 0);
    std::vector<V3d> pos, grad;
    std::vector<double> triE(tris.size());
    std::vector<std::array<V3d, 3>> triG(tris.size());
    const auto energy = [&](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
        positions(x, pos);
        const int T = static_cast<int>(tris.size());
#pragma omp parallel for schedule(static)
        for (int ti = 0; ti < T; ti++) {
            const Tri& t = tris[ti];
            V3d tg[3];
            triE[ti] = t.weight * triEnergyGrad(cornerOf(t, 0, pos), cornerOf(t, 1, pos), cornerOf(t, 2, pos), tg);
            for (int k = 0; k < 3; k++) triG[ti][k] = t.weight * tg[k];
        }
        // Summed in a fixed order, so results do not depend on the thread count.
        double f = 0.0;
        for (double e : triE) f += e;
        grad.assign(N, V3d::Zero());
#pragma omp parallel for schedule(dynamic, 256)
        for (int id = 0; id < N; id++) {
            if (!moves[id]) continue;
            V3d sum = V3d::Zero();
            for (int ti : trisOf[id]) {
                const Tri& t = tris[ti];
                for (int k = 0; k < 3; k++) {
                    if (t.id[k] == id) sum += triG[ti][k];
                }
            }
            grad[id] = sum;
        }
        g.setZero(nv);
        // Back through the attachments, finest first.
        for (auto it = drivenOrder.rbegin(); it != drivenOrder.rend(); ++it) {
            const Driven& dr = driven[*it];
            const double t = dr.var >= 0 ? x[dr.var] : dr.t;
            const V3d gd = grad[dr.id];
            if (dr.var >= 0) g[dr.var] += gd.dot(pos[dr.b] - pos[dr.a]);
            grad[dr.a] += (1.0 - t) * gd;
            grad[dr.b] += t * gd;
        }
        for (size_t i = 0; i < sliders.size(); i++) g[sliders[i].var] += grad[sliders[i].id].dot(tangentOf[i]);
        for (const Ring& r : rings) {
            g[r.var] += grad[r.id].dot(r.t1);
            g[r.var + 1] += grad[r.id].dot(r.t2);
        }
        return f;
    };
    Eigen::VectorXd x = x0, g0;
    const double f0 = energy(x, g0);
    LBFGSpp::LBFGSBParam<double> param;
    param.m = 6;
    param.max_iterations = params_.borderLloydIterations;
    param.epsilon = 1e-8;
    param.epsilon_rel = 1e-8;
    param.past = 1;
    param.delta = 1e-8;
    LBFGSpp::LBFGSBSolver<double> solver(param);
    double fx = f0;
    try {
        solver.minimize(energy, x, fx, lb, ub);
    } catch (const std::exception&) {
        // The line search can give up near the optimum; keep the last point, inside the bounds.
        x = x.cwiseMax(lb).cwiseMin(ub);
    }
    Eigen::VectorXd gx;
    const double fSolved = energy(x, gx);
    if (!(fSolved < f0)) {
        x = x0;
    }

    // Fold check on every level: pull back the variables behind a folded
    // triangle (its own vertices' and, through attachments, their ends').
    std::vector<std::vector<int>> varsBehind(N);
    for (const Slider& sl : sliders) varsBehind[sl.id] = {sl.var};
    for (const Ring& r : rings) varsBehind[r.id] = {r.var, r.var + 1};
    for (int d : drivenOrder) {
        const Driven& dr = driven[d];
        auto& to = varsBehind[dr.id];
        to.clear();
        if (dr.var >= 0) to.push_back(dr.var);
        to.insert(to.end(), varsBehind[dr.a].begin(), varsBehind[dr.a].end());
        to.insert(to.end(), varsBehind[dr.b].begin(), varsBehind[dr.b].end());
        std::sort(to.begin(), to.end());
        to.erase(std::unique(to.begin(), to.end()), to.end());
    }
    const float cosMax = static_cast<float>(std::cos(params_.borderLloydMaxNormalChangeDegrees * M_PI / 180.0));
    // Turns are measured from each level's normals as it was built (before
    // any snapping moved it), so snapping and relaxing together stay within
    // the limit.
    std::vector<std::vector<Vector3f>> normal0(levels);
    std::vector<std::vector<int>> vertexId(levels);
    for (int j = 0; j < levels; j++) {
        const Mesh& m = builder.levels[j].simplified;
        normal0[j] = st.levels[j].normal0;
        Expects(normal0[j].size() == m.triIndexes.size(), "BorderLloyd: level {} changed since it was built", j);
        vertexId[j].assign(m.vertexLocations.size(), -1);
        for (size_t v = 0; v < m.vertexLocations.size(); v++) {
            vertexId[j][v] = borders[j].nbrs[v].empty() ? ringAt[j][v] : ids.at(KeyOf(m.vertexLocations[v]));
        }
    }
    const auto pointOfId = [&](const std::vector<V3d>& p, int id) {
        const V3d q = p[id] * L;
        return Vector3f(static_cast<float>(q.x()), static_cast<float>(q.y()), static_cast<float>(q.z()));
    };
    // The relaxed positions, projected back onto the input surface (with
    // builder.projectToSurface): every slider and ring vertex that moved is
    // snapped to its closest point, then the attached vertices are placed again
    // on their (projected) coarser edges, so finer borders stay on coarser ones.
    // Vertices that did not move keep their place, so undoing every move gives
    // the original back.
    const auto settledPositions = [&](const Eigen::VectorXd& xs, std::vector<V3d>& p) {
        positions(xs, p);
        if (!builder.projectToSurface) return;
        const auto project = [&](int id) {
            const Vector3f q = builder.projectToSurface(pointOfId(p, id));
            p[id] = ToV3d(q) / L;
        };
        const int S = static_cast<int>(sliders.size());
#pragma omp parallel for schedule(static)
        for (int i = 0; i < S; i++) {
            if (xs[sliders[i].var] != x0[sliders[i].var]) project(sliders[i].id);
        }
        const int R = static_cast<int>(rings.size());
#pragma omp parallel for schedule(static)
        for (int i = 0; i < R; i++) {
            const Ring& r = rings[i];
            if (xs[r.var] != x0[r.var] || xs[r.var + 1] != x0[r.var + 1]) project(r.id);
        }
        for (int d : drivenOrder) {
            const Driven& dr = driven[d];
            const double t = dr.var >= 0 ? xs[dr.var] : dr.t;
            p[dr.id] = (1.0 - t) * p[dr.a] + t * p[dr.b];
        }
    };
    std::vector<int> halvings(nv, 0);
    while (true) {
        settledPositions(x, pos);
        std::vector<char> bad(nv, 0);
        bool any = false;
        for (int j = 0; j < levels; j++) {
            const Mesh& m = builder.levels[j].simplified;
            for (size_t f = 0; f < m.triIndexes.size(); f++) {
                const Vector3u t = m.triIndexes[f];
                const uint32_t v[3] = {t.x(), t.y(), t.z()};
                if (vertexId[j][v[0]] < 0 && vertexId[j][v[1]] < 0 && vertexId[j][v[2]] < 0) continue;
                Vector3f q[3];
                for (int k = 0; k < 3; k++) {
                    q[k] = vertexId[j][v[k]] >= 0 ? pointOfId(pos, vertexId[j][v[k]]) : m.vertexLocations[v[k]];
                }
                const Vector3f nn = (q[1] - q[0]).cross(q[2] - q[0]);
                const Vector3f n0 = normal0[j][f];
                // A fold: turned past the limit, or flattened (smallest angle
                // under 1 degree; fixFlatTriangles() removed those before).
                const bool turned = nn.dot(n0) <= cosMax * nn.norm() * n0.norm();
                const bool flattened = SmallestOf(ToV3d(q[0]), ToV3d(q[1]), ToV3d(q[2])) < 1.0;
                if (!turned && !flattened) continue;
                for (int k = 0; k < 3; k++) {
                    if (vertexId[j][v[k]] < 0) continue;
                    for (int var : varsBehind[vertexId[j][v[k]]]) {
                        if (x[var] != x0[var]) bad[var] = any = 1;
                    }
                }
            }
        }
        if (!any) break;
        for (int var = 0; var < nv; var++) {
            if (!bad[var]) continue;
            halvings[var]++;
            // Halve the move five times, then undo it: ends once every move
            // behind a folded triangle is undone (the original has no fold).
            x[var] = halvings[var] > 5 ? x0[var] : x0[var] + 0.5 * (x[var] - x0[var]);
        }
    }
    settledPositions(x, pos);
    const std::vector<std::vector<Vector3f>> before = PositionsOf(builder.levels);
    for (int j = 0; j < levels; j++) {
        Mesh& m = builder.levels[j].simplified;
        for (size_t v = 0; v < m.vertexLocations.size(); v++) {
            if (vertexId[j][v] >= 0) m.vertexLocations[v] = pointOfId(pos, vertexId[j][v]);
        }
        ReprojectTexcoords(builder.levels[j], before[j]);
    }
    return true;
}

bool MeshletBuilder::writeMlod()
{
    MeshletBuilder& builder = *this;
    if (builder.levels.empty()) return true;
    // Can any two levels be drawn side by side without cracks?
    if (builder.checkCracks()) {
        builder.exportReport.cracksChecked = true;
        const LevelCrackReport cracks = MeasureLevelCracks(builder.levels, builder.regionHierarchy);
        builder.exportReport.levelPairsWithCracks = cracks.pairsWithCracks();
        builder.exportReport.levelPairs = static_cast<int>(cracks.pairs.size());
    }
    // All of them as one meshlet LOD DAG.
    const std::filesystem::path& file = builder.mlodFile();
    if (file.empty()) return true;
    if (file.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(file.parent_path(), ec);
    }
    std::string error;
    MlodStats stats;
    if (WriteMlodFile(file, builder.levels, builder.regionHierarchy, error, &stats)) {
        builder.exportReport.written = true;
        builder.exportReport.stats = stats;
    } else {
        builder.exportReport.error = error;
    }
    return true;
}
