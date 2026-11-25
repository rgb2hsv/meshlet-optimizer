/*
 * CvtSegmentation.hpp
 *
 * Mesh segmentation from a Restricted Centroidal Voronoi Tessellation (RCVT),
 * after Yan, Levy, Liu, Sun and Wang, "Isotropic Remeshing with Fast and Exact
 * Computation of Restricted Voronoi Diagram", SGP 2009.
 *
 *   1. Seeds are drawn as a blue-noise (Poisson disk) sample of the surface.
 *      The disk radius comes from the requested seed count, clamped below by
 *      the average edge length, and a hash grid with that cell size makes the
 *      dart-throwing and later nearest-neighbour queries local.
 *   2. The Restricted Voronoi Diagram (3D Voronoi cells intersected with the
 *      triangles) is computed exactly by clipping each triangle against the
 *      bisectors of the seeds whose cells touch it, found by propagating across
 *      the bisector edges of the clipped polygons. A security radius ends the
 *      clipping of a cell, so no Delaunay triangulation is needed.
 *   3. The CVT energy  F(X) = sum_i  integral over (Vor(x_i) n S) of |x - x_i|^2
 *      is minimised with a few Lloyd (EM) iterations -- assign the surface to
 *      its nearest seed, move each seed to its cell centroid -- followed by
 *      L-BFGS on F with gradient dF/dx_i = 2 (m_i x_i - integral of x).
 *   4. Each vertex takes the label of its nearest seed. Fragments that are not
 *      the largest connected piece of their cell, and charts smaller than the
 *      minimum, are absorbed by their neighbours.
 *   5. The labels are regularised (LabelGraph::regularize): an opening removes
 *      thin bands, region competition smooths the boundaries without changing
 *      chart topology, and charts that are not disks are split.
 */
#pragma once

#include <charts/LabelRegularization.hpp>
#include <mesh/Mesh.hpp>

#include <Eigen/Core>

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mopt
{

// Settings of the base-chart CVT: seed count or spacing, Lloyd / L-BFGS iterations and label clean-up.
struct CvtSegmentationParams {
    // Number of seeds (charts) to aim for. <= 0 derives it from `spacing`, or
    // from `verticesPerSeed` when that is also unset.
    int seedCount = 0;
    // Distance between neighbouring seeds. <= 0 derives it from the seed count.
    double spacing = 0.0;
    // Used only when seedCount and spacing are both unset and there is no
    // regularization.maxTrianglesPerChart (which then sets the seed count).
    int verticesPerSeed = 512;
    // Lower bound on the spacing, in multiples of the average edge length, so
    // every Voronoi cell captures some vertices.
    double minSpacingInEdgeLengths = 2.0;
    // Dart-throwing candidates generated per requested seed.
    int candidatesPerSeed = 30;
    // EM (Lloyd) warm-up iterations before L-BFGS.
    int lloydIterations = 4;
    // L-BFGS iterations; 0 skips the quasi-Newton stage.
    int lbfgsIterations = 40;
    // L-BFGS history length.
    int lbfgsHistory = 7;
    // Stop when the energy improves by less than this fraction per iteration.
    double relativeTolerance = 1e-6;
    // Charts with fewer vertices are merged into their neighbours.
    int minChartVertices = 16;
    // Opening, region competition and disk splitting of the final labels
    // (see LabelRegularization.hpp).
    bool regularize = true;
    LabelRegularizationParams regularization;
    uint32_t randomSeed = 1;
};

// Spatial hash over points with a fixed cell size, for nearest-neighbour and
// radius queries whose answer is close to the query point.
class PointHashGrid
{
public:
    using Vec3 = Eigen::Vector3d;

    void reset(double cellSize);
    void build(const std::vector<Vec3>& points, double cellSize);
    void insert(const Vec3& p, int id);

    // Nearest point id, or -1 when empty.
    int nearest(const Vec3& p, double* dist2 = nullptr) const;
    // Is any point closer than r to p?
    bool anyWithin(const Vec3& p, double r) const;
    // All points within r of p, sorted by increasing squared distance.
    void within(const Vec3& p, double r, std::vector<std::pair<double, int>>& out) const;
    // At least k nearest points to p (more if they fall in the searched shells),
    // sorted by increasing squared distance. `completeRadius` is the distance
    // up to which the list holds every point.
    void nearestK(const Vec3& p, int k, std::vector<std::pair<double, int>>& out, double& completeRadius) const;

private:
    // Integer coordinates of a hash-grid cell.
    struct Cell {
        int x, y, z;
    };
    Cell cellOf(const Vec3& p) const;
    static uint64_t keyOf(int x, int y, int z);
    // Visit ids in cells at Chebyshev distance exactly `ring` from c.
    template <class F>
    void forEachInShell(const Cell& c, int ring, F&& f) const;
    // Ring beyond which no more points can be found.
    int maxRing(const Cell& c) const;

    double cellSize_ = 1.0;
    std::vector<Vec3> points_;
    std::unordered_map<uint64_t, std::vector<int>> cells_;
    Cell lo_{0, 0, 0}, hi_{-1, -1, -1};
};

// Splits a triangle mesh into compact charts: blue-noise seeds on the surface, relaxed to a restricted
// centroidal Voronoi tessellation (Lloyd then L-BFGS), the cells regularized into disk-shaped charts.
class CvtSegmentation
{
public:
    using Vec3 = Eigen::Vector3d;

    // Per-seed integrals over its restricted Voronoi cell.
    struct CellIntegrals {
        std::vector<double> mass;    // area of the cell
        std::vector<Vec3> moment;    // integral of x over the cell
        std::vector<double> energy;  // integral of |x - x_i|^2 over the cell
        double totalEnergy = 0.0;
        Vec3 centroid(size_t i) const { return moment[i] / mass[i]; }
    };

    explicit CvtSegmentation(const Mesh& mesh);
    // Any triangle soup; e.g. a UV triangulation at z = 0 for a CVT restricted
    // to a 2D region.
    CvtSegmentation(std::vector<Vec3> positions, std::vector<Eigen::Vector3i> triangles);

    // Spacing of n seeds in a hexagonal packing of the surface.
    double spacingForSeedCount(int n) const;

    // Poisson-disk sample with minimum distance `radius`, at most maxSeeds.
    std::vector<Vec3> sampleBlueNoise(double radius, int maxSeeds, int candidates, uint32_t randomSeed) const;

    // Exact restricted Voronoi diagram of `seeds` on the mesh, integrated per
    // cell. Deterministic for any thread count.
    CellIntegrals restrictedVoronoi(const std::vector<Vec3>& seeds) const;

    // One EM step: seeds move to their cell centroids. Returns the energy of
    // the seeds before the move.
    double lloydStep(std::vector<Vec3>& seeds) const;

    // Minimise the CVT energy with L-BFGS. Returns the final energy.
    double optimizeLBFGS(std::vector<Vec3>& seeds, int maxIterations, int history, double relativeTolerance) const;

    // Label each vertex with its nearest seed, then absorb disconnected
    // fragments and charts below minChartVertices. Labels are 0..count-1.
    int labelVertices(const std::vector<Vec3>& seeds, int minChartVertices, std::vector<int>& vertexLabels) const;

    // Full pipeline. Returns the number of charts.
    int solve(const CvtSegmentationParams& params, std::vector<int>& vertexLabels);

private:
    void initialize();
    std::vector<Vec3> positions_;
    std::vector<Eigen::Vector3i> triangles_;
    std::vector<double> triangleAreas_;
    double area_ = 0.0;
    double averageEdge_ = 0.0;
};

}  // namespace mopt
