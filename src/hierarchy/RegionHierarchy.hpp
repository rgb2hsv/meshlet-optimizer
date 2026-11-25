/*
 * RegionHierarchy.hpp
 *
 * Bottom-up clustering of mesh regions (charts) into a tree.
 *
 * Level 0 has one node per region. To build the next level, grouping starts at
 * a randomly chosen node and spreads through the adjacency graph with a FIFO
 * queue: the node at the front of the queue, if not yet grouped, seeds a group
 * with the largest clique it belongs to among ungrouped nodes -- 4 mutually
 * adjacent nodes if possible, otherwise 3, 2, or the node alone -- and the
 * ungrouped neighbours of that group join the back of the queue. When the
 * queue runs dry (a separate surface piece), another random ungrouped node
 * restarts it. Among cliques of the same size the one with the least weight
 * (e.g. fewest vertices) is taken, which keeps groups balanced. A node left
 * on its own (all its neighbours were grouped first) then joins the smallest
 * neighbouring group, so groups are a clique plus such absorbed singles.
 *
 * Each group becomes one node of the next level, adjacent to the groups its
 * members touched. Levels are added one at a time (AddRegionHierarchyLevel),
 * or all the way to a single root (BuildRegionHierarchy); nodes on separate
 * surface pieces, which no edge connects, are joined at the root.
 */
#pragma once

#include <cstdint>
#include <vector>

namespace mopt
{

class Mesh;

// The tree of regions: level 0 holds the base charts, each higher level merges cliques of the level below.
// Holds every node's parent, weight and adjacency per level, and which nodes were split.
struct RegionHierarchy {
    // nodeOf[k][region]: the node at level k that contains the region.
    // Level 0 is the regions themselves; the last level is the top so far.
    std::vector<std::vector<int>> nodeOf;
    // parent[k][node]: the level k+1 node that node (at level k) merged into.
    // One entry fewer than levels: the top level has no parents yet.
    std::vector<std::vector<int>> parent;
    // absorbed[k][node]: 1 if the node joined its parent as a leftover single
    // rather than as part of the parent's clique.
    std::vector<std::vector<char>> absorbed;
    // groupSize[k][node]: how many level k-1 nodes merged into node, absorbed
    // singles included (1 at level 0).
    std::vector<std::vector<int>> groupSize;

    // splitFrom[k][node]: the node it was split from (SplitTopNode), else
    // the node itself. Both halves of a split share the original's children
    // (parent[k-1] points at the original id), so mixing levels k-1 and k
    // swaps the halves together: they are one unit, splitFrom[k][node].
    std::vector<std::vector<int>> splitFrom;

    // weight[k][node]: the node's weight, the sum of its regions' weights.
    std::vector<std::vector<int>> weight;

    // Adjacency and weights of the top level's nodes, for the next level.
    std::vector<std::vector<int>> topAdjacency;
    std::vector<int> topWeight;
    uint32_t randomSeed = 1;
    int maxCliqueSize = 4;
    bool absorbSingletons = true;

    int levelCount() const { return static_cast<int>(nodeOf.size()); }
    int nodeCount(int level) const { return static_cast<int>(groupSize[level].size()); }
    int topNodeCount() const { return levelCount() ? nodeCount(levelCount() - 1) : 0; }
};

// Level 0 only. regionWeights (e.g. vertex counts) break ties between cliques
// of equal size. With absorbSingletons, a node that found no free neighbour
// joins the smallest adjacent group instead of moving up a level on its own.
RegionHierarchy InitRegionHierarchy(const std::vector<std::vector<int>>& regionAdjacency,
                                    const std::vector<int>& regionWeights, uint32_t randomSeed, int maxCliqueSize = 4,
                                    bool absorbSingletons = true);

// Cluster the top level into one more level. Returns false, adding nothing,
// when the top is already a single node.
bool AddRegionHierarchyLevel(RegionHierarchy& tree);

// Split top-level node `node` in two after the fact (e.g. a remeshed region
// too large to keep): a new top node is appended and its id returned. The
// new node takes `fraction` of the node's weight, has no children of its own
// (groupSize 0) and records splitFrom = node. topAdjacency is left to the
// caller, who knows which nodes the halves touch.
int SplitTopNode(RegionHierarchy& tree, int node, double fraction);

// Replace the top level's grouping after the fact (e.g. to keep groups
// disk-shaped): group[node] gives the new group of each node one level down,
// any ids (renumbered in order of their first node); absorbed[node] marks
// nodes that joined their group outside its clique. prevAdjacency is the
// adjacency of the nodes one level down; the new top adjacency and weights
// follow from it.
void RegroupTopLevel(RegionHierarchy& tree, const std::vector<int>& group, const std::vector<char>& absorbed,
                     const std::vector<std::vector<int>>& prevAdjacency);

// Undo the last AddRegionHierarchyLevel: the level below becomes the top
// again, with prevAdjacency as its adjacency.
void RemoveTopLevel(RegionHierarchy& tree, const std::vector<std::vector<int>>& prevAdjacency);

}  // namespace mopt
