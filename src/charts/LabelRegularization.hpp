/*
 * LabelRegularization.hpp
 *
 * Clean-up of per-vertex chart labels on a triangle mesh, so every chart is a
 * compact topological disk that can be flattened:
 *
 *   open     Morphological opening on the label graph. Each chart is eroded by
 *            min(openingRings, its own depth) vertex rings from its label
 *            boundary: thin tails of large charts are cut off, and charts with
 *            no interior vertex at all (one- or two-triangle bands, rings
 *            around a thin feature) vanish, while small compact charts survive. Surviving cores then regrow
 * competitively: every eroded vertex joins the core that reaches it first along mesh edges (a geodesic Voronoi diagram
 * of the cores). compete  Region competition on the label boundaries. A boundary vertex moves to a neighbouring chart
 * when that lowers compactness |x - c_l|^2 / s^2 + normal (1 - n . N_l)
 *              + smoothness (fraction of differently labelled neighbours),
 *            and only when it is a simple vertex: its one-ring meets both the
 *            old and the new chart in a single arc. Simple moves cannot split
 *            a chart or open a hole in it, so they preserve chart topology.
 *   split    Charts whose Euler characteristic V - E + F is not 1 (annuli,
 *            handles, closed shells) are cut in two from their two farthest
 *            vertices, repeatedly, until they are disks.
 *   cap      Charts touching more than maxTrianglesPerChart triangles are
 *            split the same way until they fit.
 */
#pragma once

#include <Eigen/Core>

#include <functional>
#include <vector>

namespace mopt
{

class Mesh;

// Settings of the chart clean-up: opening depth, region-competition weights, disk splitting and chart size.
struct LabelRegularizationParams {
    // Erosion depth of the opening, in vertex rings. 0 disables the opening.
    int openingRings = 2;
    // Sweeps of region competition. 0 disables it.
    int competitionIterations = 20;
    double compactnessWeight = 1.0;
    double normalWeight = 1.0;
    double smoothnessWeight = 1.0;
    // Split charts that are not topological disks.
    bool enforceDisks = true;
    // Rounds of disk splitting (each round halves every non-disk chart).
    int maxSplitRounds = 6;
    // Non-disk charts smaller than this are not split.
    int minSplitVertices = 8;
    // Upper bound on the triangles touching each chart (every triangle that
    // has a vertex in it); larger charts are split. 0 = no limit.
    int maxTrianglesPerChart = 0;
};

// How many charts there are and how many of them are topological disks.
struct LabelTopologyStats {
    int charts = 0;
    int disks = 0;
    int nonDisks = 0;
};

// Vertex adjacency, triangle areas and normals of a mesh, for cleaning up vertex labels: morphological
// opening, region competition, merging small charts and splitting non-disk charts.
class LabelGraph
{
public:
    using Vec3 = Eigen::Vector3d;

    LabelGraph(const std::vector<Vec3>& positions, const std::vector<Eigen::Vector3i>& triangles);

    size_t vertexCount() const { return positions_.size(); }

    // Renumber labels densely (0..count-1) in first-seen vertex order.
    static int compact(std::vector<int>& labels);

    // Opening + competitive regrowth. Returns the number of charts dissolved.
    int open(std::vector<int>& labels, int rings) const;
    // Topology-preserving region competition. Returns the number of moves.
    int compete(std::vector<int>& labels, const LabelRegularizationParams& params) const;
    // Split non-disk charts. Returns the number of splits.
    int splitNonDisks(std::vector<int>& labels, const LabelRegularizationParams& params) const;
    // Split charts touching more than maxTriangles triangles until none do.
    // Returns the number of splits.
    int capTriangles(std::vector<int>& labels, int maxTriangles) const;
    // Triangles touching each chart.
    std::vector<int> triangleCounts(const std::vector<int>& labels, int labelCount) const;
    // Euler characteristic V - E + F of each chart's induced subcomplex.
    std::vector<int> eulerCharacteristics(const std::vector<int>& labels, int labelCount) const;
    LabelTopologyStats topology(const std::vector<int>& labels) const;

    // open -> compete -> split -> compete. Returns the final chart count;
    // labels are dense afterwards.
    int regularize(std::vector<int>& labels, const LabelRegularizationParams& params) const;

private:
    // Returns, per label, whether that chart should be split in two.
    using SplitSelector = std::function<std::vector<char>(const std::vector<int>& labels, int labelCount)>;
    // Repeatedly cut selected charts in two from their two farthest vertices.
    int splitCharts(std::vector<int>& labels, int maxRounds, int minVertices, const SplitSelector& needsSplit) const;
    void build();
    // Number of runs of `label` around v's one-ring; -1 for a non-manifold
    // vertex, and 1 when the whole ring has the label.
    int arcsAround(int v, int label, const std::vector<int>& labels) const;
    bool onLabelBoundary(int v, const std::vector<int>& labels) const;

    std::vector<Vec3> positions_;
    std::vector<Vec3> normals_;
    std::vector<Eigen::Vector3i> triangles_;
    double area_ = 0.0;
    // Unique vertex neighbours (CSR).
    std::vector<int> nbrOffsets_, nbrs_;
    // Ordered one-rings (CSR); ringKind_: 0 open fan, 1 closed, 2 non-manifold.
    std::vector<int> ringOffsets_, ring_;
    std::vector<char> ringKind_;
};

}  // namespace mopt
