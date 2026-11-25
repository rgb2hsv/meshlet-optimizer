#include <core/Check.hpp>
#include <core/Types.hpp>
#include <hierarchy/RegionSplit.hpp>
#include <mesh/Mesh.hpp>

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <queue>
#include <unordered_map>
#include <utility>

using namespace mopt;

namespace
{
uint64_t EdgeKey(uint32_t a, uint32_t b) { return (uint64_t(a) << 32) | b; }

// Faces across each edge of each face (-1 on an open edge).
std::vector<std::array<int, 3>> FaceNeighbors(const Mesh& mesh)
{
    const size_t F = mesh.triIndexes.size();
    std::unordered_map<uint64_t, int> directed;
    directed.reserve(F * 3);
    for (size_t f = 0; f < F; f++) {
        const Vector3u t = mesh.triIndexes[f];
        const uint32_t v[3] = {t.x(), t.y(), t.z()};
        for (int e = 0; e < 3; e++) directed.emplace(EdgeKey(v[e], v[(e + 1) % 3]), static_cast<int>(f));
    }
    std::vector<std::array<int, 3>> nbr(F, {-1, -1, -1});
    for (size_t f = 0; f < F; f++) {
        const Vector3u t = mesh.triIndexes[f];
        const uint32_t v[3] = {t.x(), t.y(), t.z()};
        for (int e = 0; e < 3; e++) {
            // Consistently oriented: the neighbour has the reverse edge.
            auto it = directed.find(EdgeKey(v[(e + 1) % 3], v[e]));
            if (it == directed.end()) it = directed.find(EdgeKey(v[e], v[(e + 1) % 3]));
            if (it != directed.end() && it->second != static_cast<int>(f)) nbr[f][e] = it->second;
        }
    }
    return nbr;
}
}  // namespace

int mopt::RelabelIsolatedFaces(const Mesh& mesh, std::vector<int>& faceRegion)
{
    const std::vector<std::array<int, 3>> nbr = FaceNeighbors(mesh);
    int regionCount = 0;
    for (int r : faceRegion) regionCount = std::max(regionCount, r + 1);
    std::vector<int> size(regionCount, 0);
    for (int r : faceRegion) {
        if (r >= 0) size[r]++;
    }
    const auto isolated = [&](int f) {
        bool any = false;
        for (int g : nbr[f]) {
            if (g < 0) continue;
            any = true;
            if (faceRegion[g] == faceRegion[f]) return false;
        }
        return any;
    };
    int moved = 0;
    std::vector<int> queue;
    for (int f = 0; f < static_cast<int>(faceRegion.size()); f++) {
        if (isolated(f)) queue.push_back(f);
    }
    while (!queue.empty()) {
        std::vector<int> next;
        for (int f : queue) {
            if (!isolated(f)) continue;  // a neighbour joined it meanwhile
            int best = -1;
            for (int g : nbr[f]) {
                if (g < 0) continue;
                const int r = faceRegion[g];
                if (r < 0) continue;
                if (best < 0 || size[r] < size[best] || (size[r] == size[best] && r < best)) best = r;
            }
            if (best < 0) continue;
            const int old = faceRegion[f];
            if (old >= 0) size[old]--;
            faceRegion[f] = best;
            size[best]++;
            moved++;
            // Neighbours left in the old region may now be isolated.
            for (int g : nbr[f]) {
                if (g >= 0 && faceRegion[g] == old) next.push_back(g);
            }
        }
        queue.swap(next);
    }
    return moved;
}

std::vector<std::vector<int>> mopt::FaceRegionAdjacency(const Mesh& mesh, const std::vector<int>& faceRegion,
                                                        int regionCount)
{
    std::vector<std::vector<int>> adjacency(regionCount);
    std::unordered_map<uint64_t, int> edgeRegion;
    edgeRegion.reserve(mesh.triIndexes.size() * 3);
    for (size_t f = 0; f < mesh.triIndexes.size(); f++) {
        const int r = faceRegion[f];
        const Vector3u t = mesh.triIndexes[f];
        const uint32_t v[3] = {t.x(), t.y(), t.z()};
        for (int k = 0; k < 3; k++) {
            const uint32_t a = std::min(v[k], v[(k + 1) % 3]), b = std::max(v[k], v[(k + 1) % 3]);
            const auto [it, inserted] = edgeRegion.emplace(EdgeKey(a, b), r);
            if (!inserted && it->second != r && it->second >= 0 && r >= 0) {
                adjacency[it->second].push_back(r);
                adjacency[r].push_back(it->second);
            }
        }
    }
    for (auto& nbrs : adjacency) {
        std::sort(nbrs.begin(), nbrs.end());
        nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
    }
    return adjacency;
}

std::vector<RegionSplit> mopt::SplitLargeRegions(const Mesh& mesh, std::vector<int>& faceRegion, int regionCount,
                                                 int maxFaces)
{
    std::vector<RegionSplit> splits;
    const int F = static_cast<int>(mesh.triIndexes.size());
    if (maxFaces <= 0 || F == 0) return splits;
    std::vector<std::vector<int>> facesOf(regionCount);
    for (int f = 0; f < F; f++) facesOf[faceRegion[f]].push_back(f);
    bool any = false;
    for (const auto& faces : facesOf) any |= static_cast<int>(faces.size()) > maxFaces / 2;
    if (!any) return splits;

    const std::vector<std::array<int, 3>> nbr = FaceNeighbors(mesh);
    std::vector<Vector3f> centroid(F);
    for (int f = 0; f < F; f++) {
        const Vector3u t = mesh.triIndexes[f];
        centroid[f] = (mesh.vertexLocations[t.x()] + mesh.vertexLocations[t.y()] + mesh.vertexLocations[t.z()]) / 3.0f;
    }
    // Corner vertices: three or more regions around them.
    std::vector<std::vector<int>> regionsAt(mesh.vertexLocations.size());
    for (int f = 0; f < F; f++) {
        const Vector3u t = mesh.triIndexes[f];
        for (uint32_t v : {t.x(), t.y(), t.z()}) {
            auto& rs = regionsAt[v];
            if (std::find(rs.begin(), rs.end(), faceRegion[f]) == rs.end()) rs.push_back(faceRegion[f]);
        }
    }
    const auto isCornerFace = [&](int f) {
        const Vector3u t = mesh.triIndexes[f];
        return regionsAt[t.x()].size() >= 3 || regionsAt[t.y()].size() >= 3 || regionsAt[t.z()].size() >= 3;
    };

    // Dijkstra over the faces of `region` from `seed` by centroid distance;
    // the faces in the order they are settled, with their distances.
    std::vector<float> dist(F, std::numeric_limits<float>::max());
    const auto fill = [&](int region, int seed, std::vector<int>& order) {
        order.clear();
        for (int f : facesOf[region]) dist[f] = std::numeric_limits<float>::max();
        using Item = std::pair<float, int>;
        std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
        dist[seed] = 0.0f;
        heap.push({0.0f, seed});
        while (!heap.empty()) {
            const auto [d, f] = heap.top();
            heap.pop();
            if (d > dist[f]) continue;
            order.push_back(f);
            for (int g : nbr[f]) {
                if (g < 0 || faceRegion[g] != region) continue;
                const float nd = d + (centroid[g] - centroid[f]).norm();
                if (nd < dist[g]) {
                    dist[g] = nd;
                    heap.push({nd, g});
                }
            }
        }
    };

    // Edge-connected pieces of `faces` whose faces pass `keep`, largest first.
    std::vector<int> piece(F, -1);
    const auto piecesOf = [&](const std::vector<int>& faces, const auto& keep) {
        std::vector<std::vector<int>> pieces;
        for (int f : faces) {
            if (!keep(f) || piece[f] >= 0) continue;
            const int id = static_cast<int>(pieces.size());
            pieces.emplace_back();
            std::vector<int> stack{f};
            piece[f] = id;
            while (!stack.empty()) {
                const int g = stack.back();
                stack.pop_back();
                pieces[id].push_back(g);
                for (int h : nbr[g]) {
                    if (h >= 0 && piece[h] < 0 && faceRegion[h] == faceRegion[f] && keep(h)) {
                        piece[h] = id;
                        stack.push_back(h);
                    }
                }
            }
        }
        for (int f : faces) piece[f] = -1;
        std::stable_sort(pieces.begin(), pieces.end(),
                         [](const auto& x, const auto& y) { return x.size() > y.size(); });
        return pieces;
    };

    std::vector<int> order;
    std::vector<int> owner(F, -1);  // half (0 or 1) of each face while splitting
    std::vector<char> nearSecond(mesh.vertexLocations.size(), 0);
    int nextId = regionCount;
    for (int region = 0; region < regionCount; region++) {
        const std::vector<int>& faces = facesOf[region];
        const int n = static_cast<int>(faces.size());
        if (n <= maxFaces / 2) continue;
        // The region's main body. Remeshing can leave a few single triangles
        // that touch it only at a corner vertex (pinched outlines); they
        // follow whichever half they touch afterwards.
        const std::vector<std::vector<int>> parts = piecesOf(faces, [](int) { return true; });
        const std::vector<int>& body = parts.front();
        // Seed: the corner face farthest from the body's first face.
        fill(region, body.front(), order);
        int seed = -1;
        for (int f : order) {
            if (isCornerFace(f) && (seed < 0 || dist[f] > dist[seed])) seed = f;
        }
        if (seed < 0) seed = order.back();
        // Second seed: the body face farthest from the first.
        fill(region, seed, order);
        const int seedB = order.back();
        if (seedB == seed) continue;
        // Grow both halves at once, each from its own seed in order of
        // centroid distance; the smaller half always claims the next face.
        // Both stay edge-connected, nothing is cut off, and they end up the
        // same size unless one front is boxed in by the other.
        std::vector<int> second;
        {
            using Item = std::pair<float, int>;
            using Heap = std::priority_queue<Item, std::vector<Item>, std::greater<Item>>;
            Heap heap[2];
            int count[2] = {0, 0};
            const auto claim = [&](int side, int f, float d) {
                owner[f] = side;
                count[side]++;
                if (side == 1) second.push_back(f);
                for (int g : nbr[f]) {
                    if (g >= 0 && faceRegion[g] == region && owner[g] < 0) {
                        heap[side].push({d + (centroid[g] - centroid[f]).norm(), g});
                    }
                }
            };
            claim(0, seed, 0.0f);
            claim(1, seedB, 0.0f);
            for (;;) {
                for (int side : {0, 1}) {
                    while (!heap[side].empty() && owner[heap[side].top().second] >= 0) heap[side].pop();
                }
                int side = count[0] <= count[1] ? 0 : 1;
                if (heap[side].empty()) side = 1 - side;
                if (heap[side].empty()) break;
                const auto [d, f] = heap[side].top();
                heap[side].pop();
                claim(side, f, d);
            }
            for (int f : body) owner[f] = -1;
        }
        RegionSplit split;
        split.region = region;
        split.newRegion = nextId++;
        for (int f : second) {
            faceRegion[f] = split.newRegion;
            const Vector3u t = mesh.triIndexes[f];
            nearSecond[t.x()] = nearSecond[t.y()] = nearSecond[t.z()] = 1;
        }
        split.movedFaces = static_cast<int>(second.size());
        // Stray triangles go with the new half when they touch it.
        for (size_t p = 1; p < parts.size(); p++) {
            bool touches = false;
            for (int f : parts[p]) {
                const Vector3u t = mesh.triIndexes[f];
                touches |= nearSecond[t.x()] || nearSecond[t.y()] || nearSecond[t.z()];
            }
            if (!touches) continue;
            for (int f : parts[p]) faceRegion[f] = split.newRegion;
            split.movedFaces += static_cast<int>(parts[p].size());
        }
        for (int f : second) {
            const Vector3u t = mesh.triIndexes[f];
            nearSecond[t.x()] = nearSecond[t.y()] = nearSecond[t.z()] = 0;
        }
        split.keptFaces = n - split.movedFaces;
        splits.push_back(split);
    }
    return splits;
}
