#include <core/Check.hpp>
#include <core/Types.hpp>
#include <hierarchy/SeamStraightening.hpp>
#include <mesh/Mesh.hpp>

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>

using namespace mopt;

namespace
{
uint64_t EdgeKey(uint32_t a, uint32_t b) { return (uint64_t(std::min(a, b)) << 32) | std::max(a, b); }

// A seam between two junctions, as its vertices.
struct Chain {
    int a = -1, b = -1;
    std::vector<uint32_t> vertices;  // junction ... junction
};

// The seams between base regions; replaces each with the shortest edge path between its junctions that
// stays near the regions' bisector.
class Seams
{
public:
    Seams(const Mesh& mesh, std::vector<int>& faceRegion, double bisectorWeight)
        : mesh_(mesh), label_(faceRegion), bisectorWeight_(bisectorWeight)
    {
        const size_t F = mesh.triIndexes.size(), V = mesh.vertexLocations.size();
        edgeFaces_.reserve(F * 3 / 2 + 1);
        for (size_t f = 0; f < F; f++) {
            const Vector3u t = mesh.triIndexes[f];
            const uint32_t v[3] = {t.x(), t.y(), t.z()};
            for (int k = 0; k < 3; k++) edgeFaces_[EdgeKey(v[k], v[(k + 1) % 3])].push_back(static_cast<int>(f));
        }
        // Vertex -> faces and vertex -> neighbours (CSR).
        std::vector<std::vector<int>> vf(V);
        std::vector<std::vector<uint32_t>> vv(V);
        for (size_t f = 0; f < F; f++) {
            const Vector3u t = mesh.triIndexes[f];
            for (uint32_t v : {t.x(), t.y(), t.z()}) vf[v].push_back(static_cast<int>(f));
        }
        meshBorder_.assign(V, 0);
        for (const auto& [key, faces] : edgeFaces_) {
            const uint32_t a = static_cast<uint32_t>(key >> 32), b = static_cast<uint32_t>(key & 0xffffffffu);
            vv[a].push_back(b);
            vv[b].push_back(a);
            if (faces.size() != 2) meshBorder_[a] = meshBorder_[b] = 1;
        }
        vertexFaces_ = std::move(vf);
        neighbors_ = std::move(vv);
        int regions = 0;
        for (int r : label_) regions = std::max(regions, r + 1);
        facesOf_.resize(regions);
        for (size_t f = 0; f < F; f++) facesOf_[label_[f]].push_back(static_cast<int>(f));
        mark_.assign(V, 0);
        faceMark_.assign(F, 0);
        // Area-weighted region centroids: the seeds of a converged CVT sit at
        // their cells' centroids, so these stand in for them. Mean edge
        // length scales the distance to a bisector.
        centroid_.assign(regions, Vector3f::Zero());
        std::vector<double> area(regions, 0.0);
        for (size_t f = 0; f < F; f++) {
            const Vector3u t = mesh.triIndexes[f];
            const Vector3f a = mesh.vertexLocations[t.x()], b = mesh.vertexLocations[t.y()],
                           c = mesh.vertexLocations[t.z()];
            const float w = 0.5f * ((b - a).cross(c - a)).norm();
            centroid_[label_[f]] += (a + b + c) * (w / 3.0f);
            area[label_[f]] += w;
        }
        for (int r = 0; r < regions; r++) {
            if (area[r] > 0) centroid_[r] /= static_cast<float>(area[r]);
        }
        double edgeSum = 0.0;
        for (const auto& [key, faces] : edgeFaces_) {
            edgeSum += edgeLength(static_cast<uint32_t>(key >> 32), static_cast<uint32_t>(key & 0xffffffffu));
        }
        meanEdge_ = edgeFaces_.empty() ? 1.0f : static_cast<float>(edgeSum / edgeFaces_.size());
    }

    // Distance of the edge's midpoint from the bisector plane of the two
    // regions' centroids, in mean edge lengths.
    float bisectorDistance(int ra, int rb, uint32_t u, uint32_t v) const
    {
        const Vector3f ca = centroid_[ra], cb = centroid_[rb];
        const Vector3f axis = cb - ca;
        const float len = axis.norm();
        if (!(len > 0.0f)) return 0.0f;
        const Vector3f mid = 0.5f * (mesh_.vertexLocations[u] + mesh_.vertexLocations[v]);
        return std::abs((mid - 0.5f * (ca + cb)).dot(axis / len)) / meanEdge_;
    }

    // Path cost of an edge on a border between ra and rb: its length,
    // stretched by its distance from their bisector.
    float edgeCost(int ra, int rb, uint32_t u, uint32_t v) const
    {
        return edgeLength(u, v) * (1.0f + static_cast<float>(bisectorWeight_) * bisectorDistance(ra, rb, u, v));
    }

    // Length-weighted mean distance (in mean edge lengths) of the border
    // edges from the bisectors of the regions they separate.
    double meanBisectorDistance() const
    {
        double sum = 0.0, total = 0.0;
        for (const auto& [key, faces] : edgeFaces_) {
            if (!isBorderEdge(key)) continue;
            const uint32_t u = static_cast<uint32_t>(key >> 32), v = static_cast<uint32_t>(key & 0xffffffffu);
            const double len = edgeLength(u, v);
            sum += len * bisectorDistance(label_[faces[0]], label_[faces[1]], u, v);
            total += len;
        }
        return total > 0 ? sum / total : 0.0;
    }

    float edgeLength(uint32_t a, uint32_t b) const
    {
        return (mesh_.vertexLocations[a] - mesh_.vertexLocations[b]).norm();
    }

    // Two faces in different regions.
    bool isBorderEdge(uint64_t key) const
    {
        const auto it = edgeFaces_.find(key);
        return it != edgeFaces_.end() && it->second.size() == 2 && label_[it->second[0]] != label_[it->second[1]];
    }

    int regionCountAt(uint32_t v) const
    {
        int count = 0, seen[3] = {-1, -1, -1};
        for (int f : vertexFaces_[v]) {
            const int r = label_[f];
            bool known = false;
            for (int i = 0; i < count; i++) known |= seen[i] == r;
            if (known) continue;
            if (count == 3) return 4;
            seen[count++] = r;
        }
        return count;
    }

    int borderDegree(uint32_t v) const
    {
        int d = 0;
        for (uint32_t w : neighbors_[v]) d += isBorderEdge(EdgeKey(v, w));
        return d;
    }

    bool isJunction(uint32_t v) const { return meshBorder_[v] || regionCountAt(v) >= 3 || borderDegree(v) != 2; }

    std::vector<Chain> traceChains(int& loops, double& length)
    {
        std::unordered_set<uint64_t> visited;
        std::vector<Chain> chains;
        length = 0.0;
        for (const auto& [key, faces] : edgeFaces_) {
            if (isBorderEdge(key)) {
                length += edgeLength(static_cast<uint32_t>(key >> 32), static_cast<uint32_t>(key & 0xffffffffu));
            }
        }
        const auto walk = [&](uint32_t from, uint32_t to) {
            Chain chain;
            const auto& faces = edgeFaces_.at(EdgeKey(from, to));
            chain.a = std::min(label_[faces[0]], label_[faces[1]]);
            chain.b = std::max(label_[faces[0]], label_[faces[1]]);
            chain.vertices = {from, to};
            visited.insert(EdgeKey(from, to));
            uint32_t prev = from, cur = to;
            while (!isJunction(cur)) {
                uint32_t next = cur;
                for (uint32_t w : neighbors_[cur]) {
                    const uint64_t k = EdgeKey(cur, w);
                    if (w != prev && isBorderEdge(k) && !visited.count(k)) {
                        next = w;
                        break;
                    }
                }
                if (next == cur) break;  // closed back on itself
                visited.insert(EdgeKey(cur, next));
                chain.vertices.push_back(next);
                prev = cur;
                cur = next;
            }
            return chain;
        };
        const uint32_t V = static_cast<uint32_t>(mesh_.vertexLocations.size());
        for (uint32_t v = 0; v < V; v++) {
            if (!isJunction(v)) continue;
            for (uint32_t w : neighbors_[v]) {
                const uint64_t k = EdgeKey(v, w);
                if (isBorderEdge(k) && !visited.count(k)) chains.push_back(walk(v, w));
            }
        }
        // Border edges left over form closed loops without a junction.
        loops = 0;
        for (const auto& [key, faces] : edgeFaces_) {
            if (!isBorderEdge(key) || visited.count(key)) continue;
            loops++;
            walk(static_cast<uint32_t>(key >> 32), static_cast<uint32_t>(key & 0xffffffffu));
        }
        return chains;
    }

    // Shortest path from the chain's first junction to its last through
    // vertices inside the two regions or on the chain; empty if none.
    std::vector<uint32_t> shortestPath(const Chain& chain, float& pathLength)
    {
        const uint32_t start = chain.vertices.front(), goal = chain.vertices.back();
        stamp_++;
        for (uint32_t v : chain.vertices) mark_[v] = stamp_;
        const auto inPair = [&](int r) { return r == chain.a || r == chain.b; };
        const auto allowed = [&](uint32_t v) {
            if (v == goal || mark_[v] == stamp_) return true;
            if (meshBorder_[v]) return false;
            const int r = label_[vertexFaces_[v].front()];
            if (!inPair(r)) return false;
            for (int f : vertexFaces_[v]) {
                if (label_[f] != r) return false;  // on some border
            }
            return true;
        };
        std::unordered_map<uint32_t, float> dist;
        std::unordered_map<uint32_t, uint32_t> from;
        using Item = std::pair<float, uint32_t>;
        std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
        dist[start] = 0.0f;
        heap.push({0.0f, start});
        while (!heap.empty()) {
            const auto [d, v] = heap.top();
            heap.pop();
            if (d > dist[v]) continue;
            if (v == goal) break;
            for (uint32_t w : neighbors_[v]) {
                if (w == start || !allowed(w)) continue;
                // The edge must lie inside the two regions.
                const auto& faces = edgeFaces_.at(EdgeKey(v, w));
                if (faces.size() != 2 || !inPair(label_[faces[0]]) || !inPair(label_[faces[1]])) continue;
                const float nd = d + edgeCost(chain.a, chain.b, v, w);
                const auto it = dist.find(w);
                if (it == dist.end() || nd < it->second) {
                    dist[w] = nd;
                    from[w] = v;
                    heap.push({nd, w});
                }
            }
        }
        std::vector<uint32_t> path;
        if (!dist.count(goal)) return path;
        pathLength = dist[goal];
        for (uint32_t v = goal;; v = from.at(v)) {
            path.push_back(v);
            if (v == start) break;
        }
        std::reverse(path.begin(), path.end());
        return path;
    }

    // Replace the chain by the path when both regions survive unchanged in
    // topology. Returns the number of faces moved, or -1 when rejected.
    int replace(const Chain& chain, const std::vector<uint32_t>& path)
    {
        std::unordered_set<uint64_t> oldEdges, newEdges;
        for (size_t i = 0; i + 1 < chain.vertices.size(); i++)
            oldEdges.insert(EdgeKey(chain.vertices[i], chain.vertices[i + 1]));
        for (size_t i = 0; i + 1 < path.size(); i++) newEdges.insert(EdgeKey(path[i], path[i + 1]));
        std::vector<int> faces = facesOf_[chain.a];
        faces.insert(faces.end(), facesOf_[chain.b].begin(), facesOf_[chain.b].end());
        // Pieces of the two regions cut along the new path; other borders
        // between them still cut, the old chain no longer does.
        faceStamp_++;
        for (int f : faces) faceMark_[f] = faceStamp_;
        std::unordered_map<int, int> piece;
        std::vector<std::vector<int>> pieces;
        for (int f0 : faces) {
            if (piece.count(f0)) continue;
            const int id = static_cast<int>(pieces.size());
            pieces.emplace_back();
            std::vector<int> stack{f0};
            piece[f0] = id;
            while (!stack.empty()) {
                const int f = stack.back();
                stack.pop_back();
                pieces[id].push_back(f);
                const Vector3u t = mesh_.triIndexes[f];
                const uint32_t v[3] = {t.x(), t.y(), t.z()};
                for (int k = 0; k < 3; k++) {
                    const uint64_t key = EdgeKey(v[k], v[(k + 1) % 3]);
                    if (newEdges.count(key)) continue;
                    for (int g : edgeFaces_.at(key)) {
                        if (g == f || faceMark_[g] != faceStamp_ || piece.count(g)) continue;
                        if (label_[g] != label_[f] && !oldEdges.count(key)) continue;
                        piece[g] = id;
                        stack.push_back(g);
                    }
                }
            }
        }
        std::unordered_map<int, int> newLabel;
        for (const auto& p : pieces) {
            int countA = 0;
            for (int f : p) countA += label_[f] == chain.a;
            const int countB = static_cast<int>(p.size()) - countA;
            const int r = countA > countB ? chain.a : countB > countA ? chain.b : label_[p.front()];
            for (int f : p) newLabel[f] = r;
        }
        std::vector<int> newA, newB;
        int moved = 0;
        for (int f : faces) {
            (newLabel[f] == chain.a ? newA : newB).push_back(f);
            moved += newLabel[f] != label_[f];
        }
        if (moved == 0) return 0;
        if (newA.empty() || newB.empty()) return -1;
        if (euler(newA) != euler(facesOf_[chain.a]) || euler(newB) != euler(facesOf_[chain.b])) return -1;
        // No more pieces than before (base regions can already have stray
        // triangles that touch them only at a vertex).
        if (pieceCount(newA) > pieceCount(facesOf_[chain.a]) || pieceCount(newB) > pieceCount(facesOf_[chain.b])) {
            return -1;
        }
        // Junctions stay junctions and nothing else becomes one: around the
        // faces that move, every vertex keeps whether it touches 3+ regions.
        std::vector<uint32_t> around;
        for (int f : faces) {
            if (newLabel[f] == label_[f]) continue;
            const Vector3u t = mesh_.triIndexes[f];
            around.insert(around.end(), {t.x(), t.y(), t.z()});
        }
        std::sort(around.begin(), around.end());
        around.erase(std::unique(around.begin(), around.end()), around.end());
        std::vector<char> wasJunction(around.size());
        for (size_t i = 0; i < around.size(); i++) wasJunction[i] = regionCountAt(around[i]) >= 3;
        std::vector<std::pair<int, int>> undo;
        for (int f : faces) {
            if (newLabel[f] != label_[f]) {
                undo.push_back({f, label_[f]});
                label_[f] = newLabel[f];
            }
        }
        for (size_t i = 0; i < around.size(); i++) {
            if ((regionCountAt(around[i]) >= 3) != static_cast<bool>(wasJunction[i])) {
                for (const auto& [f, r] : undo) label_[f] = r;
                return -1;
            }
        }
        facesOf_[chain.a] = std::move(newA);
        facesOf_[chain.b] = std::move(newB);
        return moved;
    }

    double borderLength() const
    {
        double length = 0.0;
        for (const auto& [key, faces] : edgeFaces_) {
            if (isBorderEdge(key)) {
                length += edgeLength(static_cast<uint32_t>(key >> 32), static_cast<uint32_t>(key & 0xffffffffu));
            }
        }
        return length;
    }

private:
    long long euler(const std::vector<int>& faces) const
    {
        std::unordered_set<uint32_t> vertices;
        std::unordered_set<uint64_t> edges;
        for (int f : faces) {
            const Vector3u t = mesh_.triIndexes[f];
            const uint32_t v[3] = {t.x(), t.y(), t.z()};
            for (int k = 0; k < 3; k++) {
                vertices.insert(v[k]);
                edges.insert(EdgeKey(v[k], v[(k + 1) % 3]));
            }
        }
        return static_cast<long long>(vertices.size()) - static_cast<long long>(edges.size()) +
               static_cast<long long>(faces.size());
    }

    // Edge-connected pieces of a set of faces.
    int pieceCount(const std::vector<int>& faces)
    {
        faceStamp_++;
        for (int f : faces) faceMark_[f] = faceStamp_;
        int count = 0;
        for (int f0 : faces) {
            if (faceMark_[f0] != faceStamp_) continue;
            count++;
            std::vector<int> stack{f0};
            faceMark_[f0] = -faceStamp_;
            while (!stack.empty()) {
                const int f = stack.back();
                stack.pop_back();
                const Vector3u t = mesh_.triIndexes[f];
                const uint32_t v[3] = {t.x(), t.y(), t.z()};
                for (int k = 0; k < 3; k++) {
                    for (int g : edgeFaces_.at(EdgeKey(v[k], v[(k + 1) % 3]))) {
                        if (faceMark_[g] == faceStamp_) {
                            faceMark_[g] = -faceStamp_;
                            stack.push_back(g);
                        }
                    }
                }
            }
        }
        return count;
    }

    const Mesh& mesh_;
    std::vector<int>& label_;
    std::unordered_map<uint64_t, std::vector<int>> edgeFaces_;
    std::vector<std::vector<int>> vertexFaces_;
    std::vector<std::vector<uint32_t>> neighbors_;
    std::vector<char> meshBorder_;
    std::vector<std::vector<int>> facesOf_;
    std::vector<int> mark_;
    int stamp_ = 0;
    std::vector<int> faceMark_;
    int faceStamp_ = 0;
    double bisectorWeight_ = 0.0;
    std::vector<Vector3f> centroid_;
    float meanEdge_ = 1.0f;
};
}  // namespace

SeamStraighteningStats mopt::StraightenRegionSeams(const Mesh& mesh, std::vector<int>& faceRegion,
                                                   const SeamStraighteningParams& params)
{
    SeamStraighteningStats stats;
    if (mesh.triIndexes.empty() || faceRegion.size() != mesh.triIndexes.size()) return stats;
    Seams seams(mesh, faceRegion, std::max(0.0, params.bisectorWeight));
    const std::vector<Chain> chains = seams.traceChains(stats.loops, stats.lengthBefore);
    stats.bisectorDistanceBefore = seams.meanBisectorDistance();
    stats.chains = static_cast<int>(chains.size());
    for (const Chain& chain : chains) {
        if (chain.vertices.front() == chain.vertices.back() || chain.vertices.size() < 3) continue;
        float oldCost = 0.0f;
        for (size_t i = 0; i + 1 < chain.vertices.size(); i++) {
            oldCost += seams.edgeCost(chain.a, chain.b, chain.vertices[i], chain.vertices[i + 1]);
        }
        float pathCost = 0.0f;
        const std::vector<uint32_t> path = seams.shortestPath(chain, pathCost);
        if (path.empty() || pathCost >= oldCost * (1.0f - 1e-5f)) continue;  // already the cheapest
        const int moved = seams.replace(chain, path);
        if (moved < 0) {
            stats.rejected++;
        } else if (moved > 0) {
            stats.straightened++;
            stats.facesMoved += moved;
        }
    }
    stats.lengthAfter = seams.borderLength();
    stats.bisectorDistanceAfter = seams.meanBisectorDistance();
    return stats;
}
