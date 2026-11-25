/*
 * RegionHierarchy.cpp
 *
 * See RegionHierarchy.hpp for an overview.
 */
#include <core/Check.hpp>
#include <core/Types.hpp>
#include <hierarchy/RegionHierarchy.hpp>
#include <iostream>
#include <mesh/Mesh.hpp>

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>
#include <random>
#include <tuple>
#include <unordered_map>

using namespace mopt;

namespace
{
bool Adjacent(const std::vector<std::vector<int>>& adj, int a, int b)
{
    return std::binary_search(adj[a].begin(), adj[a].end(), b);
}

// Largest clique (up to maxSize) containing `seed` among ungrouped nodes,
// lightest first among cliques of that size, then lowest ids.
std::vector<int> BestClique(int seed, const std::vector<std::vector<int>>& adj, const std::vector<int>& weight,
                            const std::vector<int>& group, int maxSize)
{
    std::vector<int> candidates;
    for (int n : adj[seed]) {
        if (group[n] < 0) candidates.push_back(n);
    }
    const int m = static_cast<int>(candidates.size());
    std::vector<int> best;
    std::vector<int> current;
    long long bestWeight = std::numeric_limits<long long>::max();
    // Depth-first search over subsets of the candidates, in increasing index
    // order, keeping every chosen node adjacent to all earlier ones.
    const auto search = [&](auto&& self, int start, int remaining, long long w) -> void {
        if (remaining == 0) {
            if (w < bestWeight) {
                bestWeight = w;
                best = current;
            }
            return;
        }
        for (int i = start; i <= m - remaining; i++) {
            const int c = candidates[i];
            bool ok = true;
            for (int chosen : current) {
                if (!Adjacent(adj, chosen, c)) {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;
            current.push_back(c);
            self(self, i + 1, remaining - 1, w + weight[c]);
            current.pop_back();
        }
    };
    for (int size = std::min(maxSize, m + 1); size >= 2; size--) {
        current.clear();
        bestWeight = std::numeric_limits<long long>::max();
        best.clear();
        search(search, 0, size - 1, weight[seed]);
        if (!best.empty()) {
            best.insert(best.begin(), seed);
            return best;
        }
    }
    return {seed};
}
}  // namespace

RegionHierarchy mopt::InitRegionHierarchy(const std::vector<std::vector<int>>& regionAdjacency,
                                          const std::vector<int>& regionWeights, uint32_t randomSeed, int maxCliqueSize,
                                          bool absorbSingletons)
{
    RegionHierarchy tree;
    const int regionCount = static_cast<int>(regionAdjacency.size());
    tree.randomSeed = randomSeed;
    tree.maxCliqueSize = std::max(1, maxCliqueSize);
    tree.absorbSingletons = absorbSingletons;
    std::vector<int> identity(regionCount);
    std::iota(identity.begin(), identity.end(), 0);
    tree.nodeOf.push_back(identity);
    tree.splitFrom.push_back(identity);
    tree.groupSize.push_back(std::vector<int>(regionCount, 1));
    tree.topAdjacency = regionAdjacency;
    tree.topWeight = regionWeights;
    tree.topWeight.resize(regionCount, 1);
    tree.weight.push_back(tree.topWeight);
    return tree;
}

bool mopt::AddRegionHierarchyLevel(RegionHierarchy& tree)
{
    if (tree.levelCount() == 0 || tree.topAdjacency.size() <= 1) return false;
    const int regionCount = static_cast<int>(tree.nodeOf[0].size());
    const int maxCliqueSize = tree.maxCliqueSize;
    const bool absorbSingletons = tree.absorbSingletons;
    std::vector<std::vector<int>>& adj = tree.topAdjacency;
    std::vector<int>& weight = tree.topWeight;
    // Its own random stream per level, so a level does not depend on how
    // the levels below it were built.
    std::mt19937 rng(tree.randomSeed + 7919u * static_cast<uint32_t>(tree.levelCount()));
    {
        const int n = static_cast<int>(adj.size());
        std::vector<int> group(n, -1);
        std::vector<char> queued(n, 0);
        std::vector<int> groupSize;

        // Random restart order, used whenever the queue runs dry.
        std::vector<int> restarts(n);
        std::iota(restarts.begin(), restarts.end(), 0);
        std::shuffle(restarts.begin(), restarts.end(), rng);
        size_t nextRestart = 0;
        std::deque<int> queue;

        while (true) {
            if (queue.empty()) {
                while (nextRestart < restarts.size() && group[restarts[nextRestart]] >= 0) nextRestart++;
                if (nextRestart == restarts.size()) break;
                queue.push_back(restarts[nextRestart]);
                queued[restarts[nextRestart]] = 1;
            }
            const int seed = queue.front();
            queue.pop_front();
            if (group[seed] >= 0) continue;
            const std::vector<int> clique = BestClique(seed, adj, weight, group, maxCliqueSize);
            const int id = static_cast<int>(groupSize.size());
            groupSize.push_back(static_cast<int>(clique.size()));
            for (int member : clique) group[member] = id;
            // The group's ungrouped neighbours are the next seeds.
            for (int member : clique) {
                for (int nbr : adj[member]) {
                    if (group[nbr] < 0 && !queued[nbr]) {
                        queued[nbr] = 1;
                        queue.push_back(nbr);
                    }
                }
            }
        }

        // A node left alone had every neighbour grouped before it was reached
        // (so its neighbours sit in groups of two or more). Rather than carry
        // it up a level on its own, it joins the smallest neighbouring group.
        std::vector<char> absorbed(n, 0);
        if (absorbSingletons) {
            std::vector<long long> groupWeight(groupSize.size(), 0);
            for (int a = 0; a < n; a++) groupWeight[group[a]] += weight[a];
            for (int a = 0; a < n; a++) {
                const int own = group[a];
                if (groupSize[own] != 1) continue;
                int best = -1;
                for (int b : adj[a]) {
                    const int g = group[b];
                    if (g == own || groupSize[g] < 2) continue;
                    if (best < 0 || std::tie(groupSize[g], groupWeight[g], g) <
                                        std::tie(groupSize[best], groupWeight[best], best)) {
                        best = g;
                    }
                }
                if (best < 0) continue;
                groupSize[own] = 0;
                group[a] = best;
                groupSize[best]++;
                groupWeight[best] += weight[a];
                absorbed[a] = 1;
            }
            // Drop the emptied groups and renumber the rest in order.
            std::vector<int> remap(groupSize.size(), -1);
            std::vector<int> kept;
            for (size_t g = 0; g < groupSize.size(); g++) {
                if (groupSize[g] > 0) {
                    remap[g] = static_cast<int>(kept.size());
                    kept.push_back(groupSize[g]);
                }
            }
            for (int& g : group) g = remap[g];
            groupSize = std::move(kept);
        }

        int groups = static_cast<int>(groupSize.size());
        if (groups == n) {
            // No two nodes touch: separate surface pieces, joined at the root.
            std::cout << "Region hierarchy: " << n << " disconnected nodes joined at the root" << '\n';
            std::fill(group.begin(), group.end(), 0);
            groupSize.assign(1, n);
            groups = 1;
        }

        std::vector<std::vector<int>> nextAdj(groups);
        std::vector<int> nextWeight(groups, 0);
        for (int a = 0; a < n; a++) {
            nextWeight[group[a]] += weight[a];
            for (int b : adj[a]) {
                if (group[a] != group[b]) nextAdj[group[a]].push_back(group[b]);
            }
        }
        for (auto& nbrs : nextAdj) {
            std::sort(nbrs.begin(), nbrs.end());
            nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
        }

        std::vector<int> nodeOf(regionCount);
        for (int r = 0; r < regionCount; r++) nodeOf[r] = group[tree.nodeOf.back()[r]];
        tree.parent.push_back(group);
        tree.absorbed.push_back(std::move(absorbed));
        tree.nodeOf.push_back(std::move(nodeOf));
        std::vector<int> self(groupSize.size());
        std::iota(self.begin(), self.end(), 0);
        tree.splitFrom.push_back(std::move(self));
        tree.groupSize.push_back(std::move(groupSize));
        tree.weight.push_back(nextWeight);
        adj = std::move(nextAdj);
        weight = std::move(nextWeight);
    }
    return true;
}

int mopt::SplitTopNode(RegionHierarchy& tree, int node, double fraction)
{
    Expects(tree.levelCount() > 0 && node >= 0 && node < tree.topNodeCount(), "SplitTopNode: no top node {}", node);
    const int id = tree.topNodeCount();
    tree.groupSize.back().push_back(0);
    tree.splitFrom.back().push_back(tree.splitFrom.back()[node]);
    tree.topAdjacency.resize(id + 1);
    tree.topWeight.resize(id + 1, 0);
    const int moved = static_cast<int>(std::lround(std::clamp(fraction, 0.0, 1.0) * tree.topWeight[node]));
    tree.topWeight[id] = std::max(1, moved);
    tree.topWeight[node] = std::max(1, tree.topWeight[node] - moved);
    tree.weight.back() = tree.topWeight;
    return id;
}

void mopt::RegroupTopLevel(RegionHierarchy& tree, const std::vector<int>& group, const std::vector<char>& absorbed,
                           const std::vector<std::vector<int>>& prevAdjacency)
{
    const int k = tree.levelCount() - 1;
    Expects(k >= 1, "RegroupTopLevel needs a level above the regions");
    const int n = tree.nodeCount(k - 1);
    Expects(static_cast<int>(group.size()) == n && static_cast<int>(absorbed.size()) == n &&
                static_cast<int>(prevAdjacency.size()) == n,
            "RegroupTopLevel: {} nodes one level down", n);
    std::unordered_map<int, int> renumber;
    std::vector<int> parent(n);
    for (int a = 0; a < n; a++) {
        parent[a] = renumber.emplace(group[a], static_cast<int>(renumber.size())).first->second;
    }
    const int groups = static_cast<int>(renumber.size());
    std::vector<int> groupSize(groups, 0), weight(groups, 0);
    std::vector<std::vector<int>> adjacency(groups);
    for (int a = 0; a < n; a++) {
        groupSize[parent[a]]++;
        weight[parent[a]] += tree.weight[k - 1][a];
        for (int b : prevAdjacency[a]) {
            if (parent[a] != parent[b]) adjacency[parent[a]].push_back(parent[b]);
        }
    }
    for (auto& nbrs : adjacency) {
        std::sort(nbrs.begin(), nbrs.end());
        nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
    }
    const int regionCount = static_cast<int>(tree.nodeOf[0].size());
    for (int r = 0; r < regionCount; r++) tree.nodeOf[k][r] = parent[tree.nodeOf[k - 1][r]];
    tree.parent[k - 1] = std::move(parent);
    tree.absorbed[k - 1] = absorbed;
    tree.groupSize[k] = std::move(groupSize);
    std::vector<int> self(groups);
    std::iota(self.begin(), self.end(), 0);
    tree.splitFrom[k] = std::move(self);
    tree.weight[k] = weight;
    tree.topWeight = std::move(weight);
    tree.topAdjacency = std::move(adjacency);
}

void mopt::RemoveTopLevel(RegionHierarchy& tree, const std::vector<std::vector<int>>& prevAdjacency)
{
    Expects(tree.levelCount() >= 2, "RemoveTopLevel needs a level above the regions");
    tree.nodeOf.pop_back();
    tree.parent.pop_back();
    tree.absorbed.pop_back();
    tree.groupSize.pop_back();
    tree.splitFrom.pop_back();
    tree.weight.pop_back();
    tree.topWeight = tree.weight.back();
    tree.topAdjacency = prevAdjacency;
}
