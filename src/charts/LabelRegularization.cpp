/*
 * LabelRegularization.cpp
 *
 * See LabelRegularization.hpp for an overview.
 */
#include <charts/LabelRegularization.hpp>
#include <core/Check.hpp>
#include <core/Types.hpp>
#include <iostream>
#include <mesh/Mesh.hpp>

#include <algorithm>
#include <limits>
#include <queue>
#include <utility>

using namespace mopt;
using V3d = Eigen::Vector3d;

namespace
{
constexpr int kUnset = -1;
constexpr int kFar = std::numeric_limits<int>::max();

}  // namespace

LabelGraph::LabelGraph(const std::vector<Vec3>& positions, const std::vector<Eigen::Vector3i>& triangles)
    : positions_(positions), triangles_(triangles)
{
    build();
}

void LabelGraph::build()
{
    const int V = static_cast<int>(positions_.size());
    normals_.assign(V, V3d::Zero());
    area_ = 0.0;
    // Oriented edge opposite each corner: for triangle (a, b, c), vertex a
    // sees b -> c, b sees c -> a and c sees a -> b.
    std::vector<int> fanOffsets(V + 1, 0);
    for (const auto& t : triangles_) {
        for (int k = 0; k < 3; k++) fanOffsets[t[k] + 1]++;
        const V3d n = (positions_[t[1]] - positions_[t[0]]).cross(positions_[t[2]] - positions_[t[0]]);
        area_ += 0.5 * n.norm();
        for (int k = 0; k < 3; k++) normals_[t[k]] += n;
    }
    for (V3d& n : normals_) {
        const double len = n.norm();
        if (len > 0.0) n /= len;
    }
    for (int v = 0; v < V; v++) fanOffsets[v + 1] += fanOffsets[v];
    std::vector<std::pair<int, int>> fan(fanOffsets.back());
    {
        std::vector<int> fill(fanOffsets.begin(), fanOffsets.end() - 1);
        for (const auto& t : triangles_) {
            for (int k = 0; k < 3; k++) fan[fill[t[k]]++] = {t[(k + 1) % 3], t[(k + 2) % 3]};
        }
    }

    nbrOffsets_.assign(V + 1, 0);
    nbrs_.clear();
    ringOffsets_.assign(V + 1, 0);
    ring_.clear();
    ringKind_.assign(V, 0);
    std::vector<int> scratch;
    std::vector<char> used;
    for (int v = 0; v < V; v++) {
        const auto begin = fan.begin() + fanOffsets[v];
        const auto end = fan.begin() + fanOffsets[v + 1];
        const int count = static_cast<int>(end - begin);

        scratch.clear();
        for (auto it = begin; it != end; ++it) {
            scratch.push_back(it->first);
            scratch.push_back(it->second);
        }
        std::sort(scratch.begin(), scratch.end());
        scratch.erase(std::unique(scratch.begin(), scratch.end()), scratch.end());
        nbrs_.insert(nbrs_.end(), scratch.begin(), scratch.end());
        nbrOffsets_[v + 1] = static_cast<int>(nbrs_.size());

        // Chain the fan edges into an ordered ring. An edge start that is no
        // edge's end marks an open fan (mesh border).
        int start = count ? begin->first : -1;
        bool open = false;
        for (auto it = begin; it != end && !open; ++it) {
            const bool isEnd = std::any_of(begin, end, [&](const auto& e) { return e.second == it->first; });
            if (!isEnd) {
                start = it->first;
                open = true;
            }
        }
        const size_t ringStart = ring_.size();
        used.assign(count, 0);
        int usedCount = 0;
        int cur = start;
        if (count) ring_.push_back(cur);
        while (count) {
            int next = -1;
            for (int k = 0; k < count; k++) {
                if (!used[k] && (begin + k)->first == cur) {
                    used[k] = 1;
                    usedCount++;
                    next = (begin + k)->second;
                    break;
                }
            }
            if (next < 0 || (!open && next == start)) break;
            ring_.push_back(next);
            cur = next;
        }
        char kind = open ? 0 : 1;
        if (usedCount != count || static_cast<int>(ring_.size() - ringStart) != static_cast<int>(scratch.size())) {
            kind = 2;  // non-manifold or inconsistently oriented
        }
        ringKind_[v] = kind;
        ringOffsets_[v + 1] = static_cast<int>(ring_.size());
    }
}

int LabelGraph::compact(std::vector<int>& labels)
{
    int maxLabel = -1;
    for (int l : labels) maxLabel = std::max(maxLabel, l);
    std::vector<int> dense(maxLabel + 1, -1);
    int count = 0;
    for (int& l : labels) {
        if (l < 0) continue;
        if (dense[l] < 0) dense[l] = count++;
        l = dense[l];
    }
    return count;
}

bool LabelGraph::onLabelBoundary(int v, const std::vector<int>& labels) const
{
    for (int k = nbrOffsets_[v]; k < nbrOffsets_[v + 1]; k++) {
        if (labels[nbrs_[k]] != labels[v]) return true;
    }
    return false;
}

int LabelGraph::arcsAround(int v, int label, const std::vector<int>& labels) const
{
    if (ringKind_[v] == 2) return -1;
    const int b = ringOffsets_[v];
    const int m = ringOffsets_[v + 1] - b;
    if (m == 0) return 0;
    const bool closed = ringKind_[v] == 1;
    int runs = 0;
    int inside = 0;
    for (int k = 0; k < m; k++) {
        const bool in = labels[ring_[b + k]] == label;
        inside += in;
        const bool prevIn = k > 0 ? labels[ring_[b + k - 1]] == label : closed && labels[ring_[b + m - 1]] == label;
        if (in && !prevIn) runs++;
    }
    return inside == m ? 1 : runs;
}

int LabelGraph::open(std::vector<int>& labels, int rings) const
{
    const int V = static_cast<int>(vertexCount());
    const int L = compact(labels);
    if (rings <= 0 || L <= 1) return 0;

    // Depth of each vertex inside its chart, in rings from the label boundary.
    std::vector<int> depth(V, kFar);
    std::vector<int> frontier, next;
    for (int v = 0; v < V; v++) {
        if (onLabelBoundary(v, labels)) {
            depth[v] = 0;
            frontier.push_back(v);
        }
    }
    for (int d = 1; d <= rings && !frontier.empty(); d++) {
        next.clear();
        for (int v : frontier) {
            for (int k = nbrOffsets_[v]; k < nbrOffsets_[v + 1]; k++) {
                const int u = nbrs_[k];
                if (depth[u] == kFar && labels[u] == labels[v]) {
                    depth[u] = d;
                    next.push_back(u);
                }
            }
        }
        frontier.swap(next);
    }

    // Erode each chart by at most its own depth, so small compact charts keep
    // a core and only bands with no interior (depth 0) vanish.
    std::vector<int> labelDepth(L, 0);
    for (int v = 0; v < V; v++) {
        labelDepth[labels[v]] = std::max(labelDepth[labels[v]], std::min(depth[v], rings));
    }
    for (int v = 0; v < V; v++) {
        // Re-express depth as "is core" (1) or not (0) against the chart's own threshold.
        const int threshold = std::max(1, labelDepth[labels[v]]);
        depth[v] = std::min(depth[v], rings) >= threshold ? rings : 0;
    }

    // Largest connected core of each chart.
    std::vector<int> coreComponent(V, kUnset);
    std::vector<int> componentSize;
    std::vector<int> largest(L, kUnset);
    std::vector<int> stack;
    for (int s = 0; s < V; s++) {
        if (depth[s] < rings || coreComponent[s] != kUnset) continue;
        const int id = static_cast<int>(componentSize.size());
        componentSize.push_back(0);
        coreComponent[s] = id;
        stack.assign(1, s);
        while (!stack.empty()) {
            const int v = stack.back();
            stack.pop_back();
            componentSize[id]++;
            for (int k = nbrOffsets_[v]; k < nbrOffsets_[v + 1]; k++) {
                const int u = nbrs_[k];
                if (depth[u] >= rings && coreComponent[u] == kUnset && labels[u] == labels[v]) {
                    coreComponent[u] = id;
                    stack.push_back(u);
                }
            }
        }
        int& best = largest[labels[s]];
        if (best == kUnset || componentSize[id] > componentSize[best]) best = id;
    }

    // Competitive regrowth: geodesic Voronoi diagram of the kept cores.
    std::vector<int> owner(V, kUnset);
    std::vector<double> dist(V, std::numeric_limits<double>::infinity());
    using Item = std::pair<double, int>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> queue;
    for (int v = 0; v < V; v++) {
        const int c = coreComponent[v];
        if (c != kUnset && largest[labels[v]] == c) {
            owner[v] = labels[v];
            dist[v] = 0.0;
            queue.emplace(0.0, v);
        }
    }
    while (!queue.empty()) {
        const auto [d, v] = queue.top();
        queue.pop();
        if (d > dist[v]) continue;
        for (int k = nbrOffsets_[v]; k < nbrOffsets_[v + 1]; k++) {
            const int u = nbrs_[k];
            const double du = d + (positions_[u] - positions_[v]).norm();
            if (du < dist[u]) {
                dist[u] = du;
                owner[u] = owner[v];
                queue.emplace(du, u);
            }
        }
    }
    int dissolved = 0;
    for (int l = 0; l < L; l++) dissolved += largest[l] == kUnset;
    for (int v = 0; v < V; v++) {
        if (owner[v] == kUnset) owner[v] = labels[v];  // shell with no core at all
    }
    labels.swap(owner);
    compact(labels);
    return dissolved;
}

int LabelGraph::compete(std::vector<int>& labels, const LabelRegularizationParams& params) const
{
    const int V = static_cast<int>(vertexCount());
    const int L = compact(labels);
    if (L <= 1) return 0;
    const double s2 = std::max(area_ / L, std::numeric_limits<double>::min());

    std::vector<V3d> centroid(L), normal(L);
    std::vector<int> size(L);
    std::vector<int> candidates;
    int totalMoves = 0;
    for (int iter = 0; iter < params.competitionIterations; iter++) {
        std::fill(centroid.begin(), centroid.end(), V3d::Zero());
        std::fill(normal.begin(), normal.end(), V3d::Zero());
        std::fill(size.begin(), size.end(), 0);
        for (int v = 0; v < V; v++) {
            centroid[labels[v]] += positions_[v];
            normal[labels[v]] += normals_[v];
            size[labels[v]]++;
        }
        for (int l = 0; l < L; l++) {
            if (size[l]) centroid[l] /= size[l];
            const double len = normal[l].norm();
            if (len > 0.0) normal[l] /= len;
        }

        int moves = 0;
        for (int v = 0; v < V; v++) {
            const int cur = labels[v];
            if (size[cur] <= 1 || !onLabelBoundary(v, labels)) continue;
            const int deg = nbrOffsets_[v + 1] - nbrOffsets_[v];
            candidates.clear();
            for (int k = nbrOffsets_[v]; k < nbrOffsets_[v + 1]; k++) {
                const int l = labels[nbrs_[k]];
                if (l != cur && std::find(candidates.begin(), candidates.end(), l) == candidates.end()) {
                    candidates.push_back(l);
                }
            }
            const auto energy = [&](int l) {
                int differ = 0;
                for (int k = nbrOffsets_[v]; k < nbrOffsets_[v + 1]; k++) differ += labels[nbrs_[k]] != l;
                return params.compactnessWeight * (positions_[v] - centroid[l]).squaredNorm() / s2 +
                       params.normalWeight * (1.0 - normals_[v].dot(normal[l])) +
                       params.smoothnessWeight * double(differ) / std::max(1, deg);
            };
            // Leaving cur must not split it, joining l must not close a loop.
            const int curArcs = arcsAround(v, cur, labels);
            if (curArcs < 0 || curArcs > 1) continue;
            int best = cur;
            double bestEnergy = energy(cur) - 1e-9;
            for (int l : candidates) {
                const double e = energy(l);
                if (e < bestEnergy && arcsAround(v, l, labels) == 1) {
                    best = l;
                    bestEnergy = e;
                }
            }
            if (best != cur) {
                labels[v] = best;
                size[cur]--;
                size[best]++;
                moves++;
            }
        }
        totalMoves += moves;
        if (moves == 0) break;
    }
    return totalMoves;
}

std::vector<int> LabelGraph::eulerCharacteristics(const std::vector<int>& labels, int labelCount) const
{
    std::vector<int> chi(labelCount, 0);
    const int V = static_cast<int>(vertexCount());
    for (int v = 0; v < V; v++) {
        chi[labels[v]]++;
        for (int k = nbrOffsets_[v]; k < nbrOffsets_[v + 1]; k++) {
            const int u = nbrs_[k];
            if (u > v && labels[u] == labels[v]) chi[labels[v]]--;
        }
    }
    for (const auto& t : triangles_) {
        if (labels[t[0]] == labels[t[1]] && labels[t[1]] == labels[t[2]]) chi[labels[t[0]]]++;
    }
    return chi;
}

LabelTopologyStats LabelGraph::topology(const std::vector<int>& labels) const
{
    std::vector<int> dense = labels;
    LabelTopologyStats stats;
    stats.charts = compact(dense);
    for (int chi : eulerCharacteristics(dense, stats.charts)) {
        if (chi == 1) {
            stats.disks++;
        } else {
            stats.nonDisks++;
        }
    }
    return stats;
}

int LabelGraph::splitCharts(std::vector<int>& labels, int maxRounds, int minVertices,
                            const SplitSelector& needsSplit) const
{
    const int V = static_cast<int>(vertexCount());
    std::vector<int> hops(V, kFar);
    std::vector<double> dist(V, std::numeric_limits<double>::infinity());
    std::vector<int> side(V, kUnset);
    std::vector<int> frontier, next;
    using Item = std::pair<double, int>;
    int totalSplits = 0;

    for (int round = 0; round < maxRounds; round++) {
        int L = compact(labels);
        const std::vector<char> selected = needsSplit(labels, L);
        std::vector<std::vector<int>> members(L);
        for (int v = 0; v < V; v++) members[labels[v]].push_back(v);

        // Farthest vertex (in hops) from `from` within its chart.
        const auto farthest = [&](int from) {
            const int l = labels[from];
            for (int v : members[l]) hops[v] = kFar;
            hops[from] = 0;
            int last = from;
            frontier.assign(1, from);
            while (!frontier.empty()) {
                next.clear();
                for (int v : frontier) {
                    for (int k = nbrOffsets_[v]; k < nbrOffsets_[v + 1]; k++) {
                        const int u = nbrs_[k];
                        if (labels[u] == l && hops[u] == kFar) {
                            hops[u] = hops[v] + 1;
                            next.push_back(u);
                            if (hops[u] > hops[last] || (hops[u] == hops[last] && u < last)) last = u;
                        }
                    }
                }
                frontier.swap(next);
            }
            return last;
        };

        int splits = 0;
        for (int l = 0; l < static_cast<int>(members.size()); l++) {
            if (!selected[l] || static_cast<int>(members[l].size()) < minVertices) continue;
            const int a = farthest(members[l].front());
            const int b = farthest(a);
            if (a == b) continue;
            // Two-source geodesic Voronoi split inside the chart.
            for (int v : members[l]) {
                dist[v] = std::numeric_limits<double>::infinity();
                side[v] = kUnset;
            }
            std::priority_queue<Item, std::vector<Item>, std::greater<Item>> queue;
            dist[a] = dist[b] = 0.0;
            side[a] = 0;
            side[b] = 1;
            queue.emplace(0.0, a);
            queue.emplace(0.0, b);
            while (!queue.empty()) {
                const auto [d, v] = queue.top();
                queue.pop();
                if (d > dist[v]) continue;
                for (int k = nbrOffsets_[v]; k < nbrOffsets_[v + 1]; k++) {
                    const int u = nbrs_[k];
                    if (labels[u] != l) continue;
                    const double du = d + (positions_[u] - positions_[v]).norm();
                    if (du < dist[u]) {
                        dist[u] = du;
                        side[u] = side[v];
                        queue.emplace(du, u);
                    }
                }
            }
            const int newLabel = L + splits;
            for (int v : members[l]) {
                if (side[v] == 1) labels[v] = newLabel;
            }
            splits++;
        }
        totalSplits += splits;
        if (splits == 0) break;
    }
    compact(labels);
    return totalSplits;
}

int LabelGraph::splitNonDisks(std::vector<int>& labels, const LabelRegularizationParams& params) const
{
    return splitCharts(labels, params.maxSplitRounds, params.minSplitVertices,
                       [this](const std::vector<int>& l, int count) {
                           const std::vector<int> chi = eulerCharacteristics(l, count);
                           std::vector<char> out(count);
                           for (int i = 0; i < count; i++) out[i] = chi[i] != 1;
                           return out;
                       });
}

std::vector<int> LabelGraph::triangleCounts(const std::vector<int>& labels, int labelCount) const
{
    // A triangle counts toward every chart it touches: cutting the mesh along
    // chart boundaries leaves a piece of it in each of them.
    std::vector<int> counts(labelCount, 0);
    for (const auto& t : triangles_) {
        const int a = labels[t[0]], b = labels[t[1]], c = labels[t[2]];
        counts[a]++;
        if (b != a) counts[b]++;
        if (c != a && c != b) counts[c]++;
    }
    return counts;
}

int LabelGraph::capTriangles(std::vector<int>& labels, int maxTriangles) const
{
    if (maxTriangles <= 0) return 0;
    // Each round halves every oversized chart; 32 rounds is far more than any
    // mesh needs.
    return splitCharts(labels, 32, 2, [&](const std::vector<int>& l, int count) {
        const std::vector<int> tris = triangleCounts(l, count);
        std::vector<char> out(count);
        for (int i = 0; i < count; i++) out[i] = tris[i] > maxTriangles;
        return out;
    });
}

int LabelGraph::regularize(std::vector<int>& labels, const LabelRegularizationParams& params) const
{
    const LabelTopologyStats before = topology(labels);
    const int dissolved = open(labels, params.openingRings);
    int moves = params.competitionIterations > 0 ? compete(labels, params) : 0;
    int splits = 0;
    if (params.enforceDisks) {
        splits = splitNonDisks(labels, params);
        if (splits > 0 && params.competitionIterations > 0) moves += compete(labels, params);
    }
    // Last, so no later boundary move can push a chart back over the limit.
    const int capSplits = capTriangles(labels, params.maxTrianglesPerChart);
    const int count = compact(labels);
    const LabelTopologyStats after = topology(labels);
    std::cout << "Label regularization: " << before.charts << " charts (" << before.nonDisks << " non-disk) -> "
              << after.charts << " (" << after.nonDisks << " non-disk); dissolved " << dissolved << ", moved " << moves
              << ", split " << splits << ", size splits " << capSplits << '\n';
    if (params.maxTrianglesPerChart > 0) {
        const std::vector<int> tris = triangleCounts(labels, count);
        const int largest = tris.empty() ? 0 : *std::max_element(tris.begin(), tris.end());
        std::cout << "Largest chart: " << largest << " triangles (limit " << params.maxTrianglesPerChart << ")" << '\n';
    }
    return count;
}
