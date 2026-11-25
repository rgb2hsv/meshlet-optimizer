#include <core/Check.hpp>
#include <core/Types.hpp>
#include <hierarchy/LevelCracks.hpp>
#include <hierarchy/RegionHierarchy.hpp>
#include <limits>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

using namespace mopt;

namespace
{
uint64_t EdgeKey(uint32_t a, uint32_t b) { return (uint64_t(std::min(a, b)) << 32) | std::max(a, b); }

std::unordered_map<uint64_t, std::vector<int>> EdgeFacesOf(const Mesh& mesh)
{
    std::unordered_map<uint64_t, std::vector<int>> out;
    for (size_t f = 0; f < mesh.triIndexes.size(); f++) {
        const Vector3u t = mesh.triIndexes[f];
        const uint32_t v[3] = {t.x(), t.y(), t.z()};
        for (int k = 0; k < 3; k++) out[EdgeKey(v[k], v[(k + 1) % 3])].push_back(static_cast<int>(f));
    }
    return out;
}

// The level-k units each region of level i < k can be drawn under. A region's
// parent is a unit at the next level; when that unit was split, its halves
// can go to different units further up, so the region belongs to all of
// theirs. Sorted.
std::vector<std::vector<int>> UnitSetsAt(const RegionHierarchy& tree, int i, int k)
{
    std::vector<std::vector<int>> sets(tree.nodeCount(k));
    for (int n = 0; n < tree.nodeCount(k); n++) sets[n] = {tree.splitFrom[k][n]};
    for (int l = k - 1; l >= i; l--) {
        // Units of level l+1: the union over their halves.
        std::map<int, std::vector<int>> ofUnit;
        for (int h = 0; h < tree.nodeCount(l + 1); h++) {
            auto& u = ofUnit[tree.splitFrom[l + 1][h]];
            u.insert(u.end(), sets[h].begin(), sets[h].end());
        }
        std::vector<std::vector<int>> below(tree.nodeCount(l));
        for (int n = 0; n < tree.nodeCount(l); n++) {
            below[n] = ofUnit[tree.parent[l][n]];
            std::sort(below[n].begin(), below[n].end());
            below[n].erase(std::unique(below[n].begin(), below[n].end()), below[n].end());
        }
        sets = std::move(below);
    }
    return sets;
}

bool Disjoint(const std::vector<int>& a, const std::vector<int>& b)
{
    for (size_t i = 0, j = 0; i < a.size() && j < b.size();) {
        if (a[i] == b[j]) return false;
        a[i] < b[j] ? i++ : j++;
    }
    return true;
}

// Spatial hash of line segments, for finding the border edges near a point.
struct SegmentGrid {
    float cell = 1.0f;
    std::vector<std::pair<Vector3f, Vector3f>> segments;
    std::unordered_map<uint64_t, std::vector<int>> cells;

    static uint64_t key(int x, int y, int z)
    {
        return (uint64_t(uint32_t(x + (1 << 20))) << 42) | (uint64_t(uint32_t(y + (1 << 20))) << 21) |
               uint64_t(uint32_t(z + (1 << 20)));
    }
    int index(float v) const { return static_cast<int>(std::floor(v / cell)); }

    void add(const Vector3f& a, const Vector3f& b, float pad)
    {
        const int id = static_cast<int>(segments.size());
        segments.push_back({a, b});
        const Vector3f lo(std::min(a.x(), b.x()) - pad, std::min(a.y(), b.y()) - pad, std::min(a.z(), b.z()) - pad);
        const Vector3f hi(std::max(a.x(), b.x()) + pad, std::max(a.y(), b.y()) + pad, std::max(a.z(), b.z()) + pad);
        for (int x = index(lo.x()); x <= index(hi.x()); x++) {
            for (int y = index(lo.y()); y <= index(hi.y()); y++) {
                for (int z = index(lo.z()); z <= index(hi.z()); z++) cells[key(x, y, z)].push_back(id);
            }
        }
    }

    double distance(const Vector3f& p) const
    {
        double best = std::numeric_limits<double>::max();
        const auto it = cells.find(key(index(p.x()), index(p.y()), index(p.z())));
        if (it == cells.end()) return best;
        for (int id : it->second) {
            const Vector3f a = segments[id].first, ab = segments[id].second - a;
            const float t = std::clamp((p - a).dot(ab) / std::max(1e-30f, ab.dot(ab)), 0.0f, 1.0f);
            best = std::min(best, static_cast<double>((p - (a + t * ab)).norm()));
        }
        return best;
    }
};
}  // namespace

LevelCrackReport mopt::MeasureLevelCracks(const std::vector<LodLevel>& levels, const RegionHierarchy& tree)
{
    LevelCrackReport report;
    const int n = static_cast<int>(levels.size());
    std::vector<std::unordered_map<uint64_t, std::vector<int>>> edges(n);
    for (int l = 0; l < n; l++) edges[l] = EdgeFacesOf(levels[l].simplified);
    for (int i = 0; i < n; i++) {
        const LodLevel& fine = levels[i];
        std::set<std::array<float, 3>> fineBorder;
        for (size_t v = 0; v < fine.simplified.vertexLocations.size(); v++) {
            if (fine.simplifiedVertexRegion[v] >= 0) continue;
            const Vector3f p = fine.simplified.vertexLocations[v];
            fineBorder.insert({p.x(), p.y(), p.z()});
        }
        for (int k = i + 1; k < n; k++) {
            const LodLevel& coarse = levels[k];
            LevelPairCracks pair;
            pair.fine = i;
            pair.coarse = k;
            const std::vector<std::vector<int>> fineSets = UnitSetsAt(tree, i, k);
            // Coarser border edges between two units.
            std::vector<std::pair<uint32_t, uint32_t>> border;
            double length = 0.0;
            std::set<uint32_t> coarseVertices;
            for (const auto& [key, fs] : edges[k]) {
                if (fs.size() != 2) continue;
                const int ua = tree.splitFrom[k][coarse.simplifiedFaceRegion[fs[0]]];
                const int ub = tree.splitFrom[k][coarse.simplifiedFaceRegion[fs[1]]];
                if (ua == ub) continue;
                const uint32_t a = static_cast<uint32_t>(key >> 32), b = static_cast<uint32_t>(key & 0xffffffffu);
                border.push_back({a, b});
                length += (coarse.simplified.vertexLocations[a] - coarse.simplified.vertexLocations[b]).norm();
                coarseVertices.insert(a);
                coarseVertices.insert(b);
            }
            if (border.empty()) {
                report.pairs.push_back(pair);
                continue;
            }
            const float mean = static_cast<float>(length / border.size());
            const float tolerance = 1e-4f * mean;
            SegmentGrid coarseGrid;
            coarseGrid.cell = mean;
            for (const auto& [a, b] : border) {
                coarseGrid.add(coarse.simplified.vertexLocations[a], coarse.simplified.vertexLocations[b], tolerance);
            }
            // Finer border edges under a coarser border: their two sides can
            // only be drawn under different level-k units. (A region whose
            // parent was split can be drawn under either half's unit, and
            // only when neither is drawn.)
            std::set<uint32_t> under;
            SegmentGrid fineGrid;
            fineGrid.cell = mean;
            for (const auto& [key, fs] : edges[i]) {
                if (fs.size() != 2) continue;
                if (!Disjoint(fineSets[fine.simplifiedFaceRegion[fs[0]]], fineSets[fine.simplifiedFaceRegion[fs[1]]]))
                    continue;
                const uint32_t a = static_cast<uint32_t>(key >> 32), b = static_cast<uint32_t>(key & 0xffffffffu);
                under.insert(a);
                under.insert(b);
                fineGrid.add(fine.simplified.vertexLocations[a], fine.simplified.vertexLocations[b], tolerance);
            }
            for (uint32_t v : under) {
                pair.fineChecked++;
                const double d = coarseGrid.distance(fine.simplified.vertexLocations[v]);
                if (d > tolerance) {
                    pair.fineOff++;
                    if (pair.offSamples.size() < 8) pair.offSamples.push_back(fine.simplified.vertexLocations[v]);
                    pair.maxDistance = std::max(pair.maxDistance, d == std::numeric_limits<double>::max() ? -1.0 : d);
                }
            }
            // Coarser border vertices on such a finer border must be finer
            // vertices there (else a T-junction opens a crack). Coarser
            // border away from any finer one (e.g. along a split) is not drawn
            // next to level i.
            for (uint32_t v : coarseVertices) {
                const Vector3f p = coarse.simplified.vertexLocations[v];
                if (fineGrid.distance(p) > tolerance) continue;
                pair.coarseChecked++;
                pair.coarseMissing += fineBorder.count({p.x(), p.y(), p.z()}) == 0;
            }
            report.pairs.push_back(pair);
        }
    }
    return report;
}
