/*
 * CvtSegmentation.cpp
 *
 * See CvtSegmentation.hpp for an overview of the method.
 */
#include <charts/CvtSegmentation.hpp>
#include <core/Check.hpp>
#include <core/Types.hpp>
#include <iostream>

#include <LBFGS.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <deque>
#include <limits>
#include <random>

using namespace mopt;
using V3d = Eigen::Vector3d;

// ---------------------------------------------------------------------------
// PointHashGrid
// ---------------------------------------------------------------------------

void PointHashGrid::reset(double cellSize)
{
    cellSize_ = cellSize > 0.0 ? cellSize : 1.0;
    points_.clear();
    cells_.clear();
    lo_ = {INT_MAX, INT_MAX, INT_MAX};
    hi_ = {INT_MIN, INT_MIN, INT_MIN};
}

void PointHashGrid::build(const std::vector<V3d>& points, double cellSize)
{
    reset(cellSize);
    points_.reserve(points.size());
    cells_.reserve(points.size());
    for (size_t i = 0; i < points.size(); i++) insert(points[i], static_cast<int>(i));
}

void PointHashGrid::insert(const V3d& p, int id)
{
    if (id >= static_cast<int>(points_.size())) points_.resize(id + 1, p);
    points_[id] = p;
    const Cell c = cellOf(p);
    cells_[keyOf(c.x, c.y, c.z)].push_back(id);
    lo_ = {std::min(lo_.x, c.x), std::min(lo_.y, c.y), std::min(lo_.z, c.z)};
    hi_ = {std::max(hi_.x, c.x), std::max(hi_.y, c.y), std::max(hi_.z, c.z)};
}

PointHashGrid::Cell PointHashGrid::cellOf(const V3d& p) const
{
    return {static_cast<int>(std::floor(p.x() / cellSize_)), static_cast<int>(std::floor(p.y() / cellSize_)),
            static_cast<int>(std::floor(p.z() / cellSize_))};
}

uint64_t PointHashGrid::keyOf(int x, int y, int z)
{
    constexpr uint64_t kMask = (uint64_t(1) << 21) - 1;
    constexpr int64_t kBias = int64_t(1) << 20;
    return ((uint64_t(int64_t(x) + kBias) & kMask) << 42) | ((uint64_t(int64_t(y) + kBias) & kMask) << 21) |
           (uint64_t(int64_t(z) + kBias) & kMask);
}

int PointHashGrid::maxRing(const Cell& c) const
{
    if (points_.empty()) return -1;
    int r = 0;
    r = std::max(r, std::max(std::abs(c.x - lo_.x), std::abs(c.x - hi_.x)));
    r = std::max(r, std::max(std::abs(c.y - lo_.y), std::abs(c.y - hi_.y)));
    r = std::max(r, std::max(std::abs(c.z - lo_.z), std::abs(c.z - hi_.z)));
    return r;
}

template <class F>
void PointHashGrid::forEachInShell(const Cell& c, int ring, F&& f) const
{
    const auto visit = [&](int x, int y, int z) {
        if (x < lo_.x || x > hi_.x || y < lo_.y || y > hi_.y || z < lo_.z || z > hi_.z) return;
        const auto it = cells_.find(keyOf(x, y, z));
        if (it == cells_.end()) return;
        for (int id : it->second) f(id);
    };
    if (ring == 0) {
        visit(c.x, c.y, c.z);
        return;
    }
    for (int dx = -ring; dx <= ring; dx++) {
        for (int dy = -ring; dy <= ring; dy++) {
            if (std::abs(dx) == ring || std::abs(dy) == ring) {
                for (int dz = -ring; dz <= ring; dz++) visit(c.x + dx, c.y + dy, c.z + dz);
            } else {
                visit(c.x + dx, c.y + dy, c.z - ring);
                visit(c.x + dx, c.y + dy, c.z + ring);
            }
        }
    }
}

// After visiting shells 0..r every point closer than r * cellSize is found:
// such a point differs from the query by less than r cells along every axis.
int PointHashGrid::nearest(const V3d& p, double* dist2) const
{
    const Cell c = cellOf(p);
    const int last = maxRing(c);
    int best = -1;
    double bestD2 = std::numeric_limits<double>::max();
    for (int ring = 0; ring <= last; ring++) {
        forEachInShell(c, ring, [&](int id) {
            const double d2 = (points_[id] - p).squaredNorm();
            if (d2 < bestD2 || (d2 == bestD2 && id < best)) {
                bestD2 = d2;
                best = id;
            }
        });
        const double reach = ring * cellSize_;
        if (best >= 0 && bestD2 <= reach * reach) break;
    }
    if (dist2) *dist2 = bestD2;
    return best;
}

bool PointHashGrid::anyWithin(const V3d& p, double r) const
{
    const Cell c = cellOf(p);
    const int last = std::min(maxRing(c), static_cast<int>(std::ceil(r / cellSize_)));
    const double r2 = r * r;
    for (int ring = 0; ring <= last; ring++) {
        bool found = false;
        forEachInShell(c, ring, [&](int id) {
            if (!found && (points_[id] - p).squaredNorm() < r2) found = true;
        });
        if (found) return true;
    }
    return false;
}

void PointHashGrid::within(const V3d& p, double r, std::vector<std::pair<double, int>>& out) const
{
    out.clear();
    const Cell c = cellOf(p);
    const int last = std::min(maxRing(c), static_cast<int>(std::ceil(r / cellSize_)));
    const double r2 = r * r;
    for (int ring = 0; ring <= last; ring++) {
        forEachInShell(c, ring, [&](int id) {
            const double d2 = (points_[id] - p).squaredNorm();
            if (d2 <= r2) out.emplace_back(d2, id);
        });
    }
    std::sort(out.begin(), out.end());
}

void PointHashGrid::nearestK(const V3d& p, int k, std::vector<std::pair<double, int>>& out,
                             double& completeRadius) const
{
    out.clear();
    const Cell c = cellOf(p);
    const int last = maxRing(c);
    completeRadius = std::numeric_limits<double>::infinity();
    for (int ring = 0; ring <= last; ring++) {
        forEachInShell(c, ring, [&](int id) { out.emplace_back((points_[id] - p).squaredNorm(), id); });
        if (ring == last) break;
        const double reach = ring * cellSize_;
        const double reach2 = reach * reach;
        int inside = 0;
        for (const auto& e : out) inside += e.first <= reach2;
        if (inside >= k) {
            completeRadius = reach;
            break;
        }
    }
    std::sort(out.begin(), out.end());
    if (std::isfinite(completeRadius)) {
        const double r2 = completeRadius * completeRadius;
        out.erase(std::find_if(out.begin(), out.end(), [&](const auto& e) { return e.first > r2; }), out.end());
    }
}

// ---------------------------------------------------------------------------
// Restricted Voronoi diagram helpers
// ---------------------------------------------------------------------------
namespace
{

// Polygon vertex; `label` names the edge from this vertex to the next one:
// -1 for a piece of the triangle boundary, otherwise the seed whose bisector
// with the current seed carved it.
struct PolyVert {
    V3d p;
    int label;
};
using ClipPolygon = std::vector<PolyVert>;

// Keep the half of `in` closer to xi than to xj (Sutherland-Hodgman).
void ClipByBisector(const ClipPolygon& in, ClipPolygon& out, const V3d& xi, const V3d& xj, int j)
{
    out.clear();
    const V3d n = xj - xi;
    const V3d mid = 0.5 * (xi + xj);
    const size_t m = in.size();
    for (size_t k = 0; k < m; k++) {
        const PolyVert& a = in[k];
        const PolyVert& b = in[(k + 1) % m];
        const double fa = (a.p - mid).dot(n);
        const double fb = (b.p - mid).dot(n);
        const bool aIn = fa <= 0.0;
        const bool bIn = fb <= 0.0;
        if (aIn) {
            out.push_back(a);
            if (!bIn) out.push_back({a.p + (fa / (fa - fb)) * (b.p - a.p), j});
        } else if (bIn) {
            out.push_back({a.p + (fa / (fa - fb)) * (b.p - a.p), a.label});
        }
    }
}

double MaxDist2(const ClipPolygon& poly, const V3d& x)
{
    double r2 = 0.0;
    for (const PolyVert& v : poly) r2 = std::max(r2, (v.p - x).squaredNorm());
    return r2;
}

// One clipped Voronoi cell piece's share of a seed's CVT mass, first moment and energy.
struct Contribution {
    int seed;
    double mass;
    V3d moment;
    double energy;
};

// Integrals of 1, x and |x - s|^2 over a convex planar polygon, by fanning it
// into triangles. For a triangle with corners u, v, w relative to s,
//   integral |x - s|^2 = A/6 (u.u + v.v + w.w() + u.v + v.w() + w.u).
Contribution IntegratePolygon(const ClipPolygon& poly, const V3d& s, int seed)
{
    Contribution c{seed, 0.0, V3d::Zero(), 0.0};
    const V3d u = poly[0].p - s;
    for (size_t k = 1; k + 1 < poly.size(); k++) {
        const V3d v = poly[k].p - s;
        const V3d w = poly[k + 1].p - s;
        const double area = 0.5 * (v - u).cross(w - u).norm();
        c.mass += area;
        c.moment += area * ((u + v + w) / 3.0 + s);
        c.energy += area / 6.0 * (u.squaredNorm() + v.squaredNorm() + w.squaredNorm() + u.dot(v) + v.dot(w) + w.dot(u));
    }
    return c;
}

bool Report(const CvtSegmentation::StatusHandler& handler, const std::string& status, float progress)
{
    return handler ? handler(status, progress) : true;
}
}  // namespace

// ---------------------------------------------------------------------------
// CvtSegmentation
// ---------------------------------------------------------------------------

CvtSegmentation::CvtSegmentation(const Mesh& mesh)
{
    positions_.reserve(mesh.vertexLocations.size());
    for (const auto& p : mesh.vertexLocations) positions_.emplace_back(p.x(), p.y(), p.z());
    triangles_.reserve(mesh.triIndexes.size());
    for (const auto& t : mesh.triIndexes) {
        triangles_.emplace_back(static_cast<int>(t.x()), static_cast<int>(t.y()), static_cast<int>(t.z()));
    }
    initialize();
}

CvtSegmentation::CvtSegmentation(std::vector<V3d> positions, std::vector<Eigen::Vector3i> triangles)
    : positions_(std::move(positions)), triangles_(std::move(triangles))
{
    initialize();
}

void CvtSegmentation::initialize()
{
    triangleAreas_.clear();
    triangleAreas_.reserve(triangles_.size());
    area_ = 0.0;
    double edgeSum = 0.0;
    for (const Eigen::Vector3i& tri : triangles_) {
        const V3d& a = positions_[tri[0]];
        const V3d& b = positions_[tri[1]];
        const V3d& c = positions_[tri[2]];
        const double area = 0.5 * (b - a).cross(c - a).norm();
        triangleAreas_.push_back(area);
        area_ += area;
        edgeSum += (b - a).norm() + (c - b).norm() + (a - c).norm();
    }
    averageEdge_ = triangles_.empty() ? 0.0 : edgeSum / (3.0 * triangles_.size());
}

double CvtSegmentation::spacingForSeedCount(int n) const
{
    // A hexagonal packing with spacing r covers sqrt(3)/2 r^2 per point.
    return std::sqrt(2.0 * area_ / (std::sqrt(3.0) * std::max(1, n)));
}

std::vector<V3d> CvtSegmentation::sampleBlueNoise(double radius, int maxSeeds, int candidates,
                                                  uint32_t randomSeed) const
{
    std::vector<V3d> samples;
    if (triangles_.empty() || area_ <= 0.0) return samples;
    maxSeeds = std::max(1, maxSeeds);

    // Area-weighted triangle choice.
    std::vector<double> cdf(triangles_.size());
    double running = 0.0;
    for (size_t t = 0; t < triangles_.size(); t++) {
        running += triangleAreas_[t];
        cdf[t] = running;
    }
    std::mt19937 rng(randomSeed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    // Dart throwing: a candidate survives when no accepted sample lies within
    // `radius`. With the grid cell equal to the radius only the 27 cells around
    // the candidate are examined.
    PointHashGrid grid;
    grid.reset(radius);
    for (int c = 0; c < candidates && static_cast<int>(samples.size()) < maxSeeds; c++) {
        const double pick = uniform(rng) * running;
        const size_t t = std::min(static_cast<size_t>(std::upper_bound(cdf.begin(), cdf.end(), pick) - cdf.begin()),
                                  triangles_.size() - 1);
        const double r1 = std::sqrt(uniform(rng));
        const double r2 = uniform(rng);
        const Eigen::Vector3i& tri = triangles_[t];
        const V3d p =
            (1.0 - r1) * positions_[tri[0]] + r1 * (1.0 - r2) * positions_[tri[1]] + r1 * r2 * positions_[tri[2]];
        if (!grid.anyWithin(p, radius)) {
            grid.insert(p, static_cast<int>(samples.size()));
            samples.push_back(p);
        }
    }
    return samples;
}

CvtSegmentation::CellIntegrals CvtSegmentation::restrictedVoronoi(const std::vector<V3d>& seeds) const
{
    const int n = static_cast<int>(seeds.size());
    CellIntegrals result;
    result.mass.assign(n, 0.0);
    result.moment.assign(n, V3d::Zero());
    result.energy.assign(n, 0.0);
    if (n == 0 || triangles_.empty()) return result;

    PointHashGrid grid;
    grid.build(seeds, spacingForSeedCount(n));

    // Nearest neighbours of every seed, complete up to completeRadius[i]. The
    // clipping below asks for more only when a cell reaches beyond that.
    constexpr int kNeighbors = 16;
    std::vector<std::vector<std::pair<double, int>>> neighbors(n);
    std::vector<double> completeRadius(n);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
    for (int i = 0; i < n; i++) {
        grid.nearestK(seeds[i], kNeighbors + 1, neighbors[i], completeRadius[i]);
        std::erase_if(neighbors[i], [i](const auto& e) { return e.second == i; });
    }

    // Triangles are processed in fixed chunks whose contributions are summed
    // in chunk order, so the result does not depend on the thread count.
    constexpr int64_t kChunkSize = 256;
    const int64_t triCount = static_cast<int64_t>(triangles_.size());
    const int64_t chunkCount = (triCount + kChunkSize - 1) / kChunkSize;
    std::vector<std::vector<Contribution>> chunks(chunkCount);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (int64_t chunk = 0; chunk < chunkCount; chunk++) {
        ClipPolygon poly, scratch;
        std::vector<int> visited;
        std::deque<int> queue;
        std::vector<std::pair<double, int>> extra;
        std::vector<Contribution>& out = chunks[chunk];
        const int64_t end = std::min(triCount, (chunk + 1) * kChunkSize);
        for (int64_t t = chunk * kChunkSize; t < end; t++) {
            if (triangleAreas_[t] <= 0.0) continue;
            const Eigen::Vector3i& tri = triangles_[t];
            const V3d& a = positions_[tri[0]];
            const V3d& b = positions_[tri[1]];
            const V3d& c = positions_[tri[2]];

            // The seed nearest to the centroid owns the centroid, so its cell
            // meets the triangle. Other cells are reached across bisector edges.
            const int start = grid.nearest((a + b + c) / 3.0);
            visited.assign(1, start);
            queue.assign(1, start);
            while (!queue.empty()) {
                const int i = queue.front();
                queue.pop_front();
                const V3d& xi = seeds[i];
                poly.assign({{a, -1}, {b, -1}, {c, -1}});
                double r2 = MaxDist2(poly, xi);
                bool empty = false;
                bool secure = false;
                // Seeds farther than twice the cell's radius cannot clip it.
                const auto clipAll = [&](const std::vector<std::pair<double, int>>& list, double skipBelow2) {
                    for (const auto& [d2, j] : list) {
                        if (j == i || d2 <= skipBelow2) continue;
                        if (d2 > 4.0 * r2) {
                            secure = true;
                            return;
                        }
                        if (d2 <= 1e-24) {
                            // Coincident seeds: the lower id keeps the region.
                            if (j < i) {
                                empty = true;
                                return;
                            }
                            continue;
                        }
                        ClipByBisector(poly, scratch, xi, seeds[j], j);
                        poly.swap(scratch);
                        if (poly.size() < 3) {
                            empty = true;
                            return;
                        }
                        r2 = MaxDist2(poly, xi);
                    }
                };
                clipAll(neighbors[i], -1.0);
                const double complete2 = completeRadius[i] * completeRadius[i];
                if (!empty && !secure && 4.0 * r2 > complete2) {
                    grid.within(xi, 2.0 * std::sqrt(r2), extra);
                    clipAll(extra, complete2);
                }
                if (empty) continue;
                out.push_back(IntegratePolygon(poly, xi, i));
                for (const PolyVert& v : poly) {
                    if (v.label >= 0 && std::find(visited.begin(), visited.end(), v.label) == visited.end()) {
                        visited.push_back(v.label);
                        queue.push_back(v.label);
                    }
                }
            }
        }
    }

    for (const auto& chunk : chunks) {
        for (const Contribution& c : chunk) {
            result.mass[c.seed] += c.mass;
            result.moment[c.seed] += c.moment;
            result.energy[c.seed] += c.energy;
            result.totalEnergy += c.energy;
        }
    }
    return result;
}

double CvtSegmentation::lloydStep(std::vector<V3d>& seeds) const
{
    // E-step: restricted Voronoi cells. M-step: seeds to cell centroids.
    const CellIntegrals cells = restrictedVoronoi(seeds);
    for (size_t i = 0; i < seeds.size(); i++) {
        if (cells.mass[i] > 0.0) seeds[i] = cells.centroid(i);
    }
    return cells.totalEnergy;
}

double CvtSegmentation::optimizeLBFGS(std::vector<V3d>& seeds, int maxIterations, int history,
                                      double relativeTolerance) const
{
    const int n = static_cast<int>(seeds.size());
    if (n == 0 || maxIterations <= 0) return n ? restrictedVoronoi(seeds).totalEnergy : 0.0;

    // Work in units of the seed spacing and of the starting energy, so the
    // solver's tolerances and first step do not depend on the model's scale.
    const double scale = spacingForSeedCount(n);
    const double f0 = restrictedVoronoi(seeds).totalEnergy;
    if (!(f0 > 0.0)) return f0;

    Eigen::VectorXd y(3 * n);
    for (int i = 0; i < n; i++) y.segment<3>(3 * i) = seeds[i] / scale;

    std::vector<V3d> current(n);
    double bestF = std::numeric_limits<double>::max();
    Eigen::VectorXd bestY = y;
    auto energy = [&](const Eigen::VectorXd& yy, Eigen::VectorXd& grad) {
        for (int i = 0; i < n; i++) current[i] = yy.segment<3>(3 * i) * scale;
        const CellIntegrals cells = restrictedVoronoi(current);
        grad.resize(3 * n);
        for (int i = 0; i < n; i++) {
            // dF/dx_i = 2 (m_i x_i - integral of x over the cell)
            grad.segment<3>(3 * i) = (2.0 * scale / f0) * (cells.mass[i] * current[i] - cells.moment[i]);
        }
        const double f = cells.totalEnergy / f0;
        if (f < bestF) {
            bestF = f;
            bestY = yy;
        }
        return f;
    };

    LBFGSpp::LBFGSParam<double> param;
    param.m = std::max(3, history);
    param.max_iterations = maxIterations;
    param.epsilon = 1e-10;
    param.epsilon_rel = 1e-10;
    param.past = 1;
    param.delta = relativeTolerance;
    LBFGSpp::LBFGSSolver<double> solver(param);
    double fx = 0.0;
    try {
        const int iterations = solver.minimize(energy, y, fx);
        std::cout << "CVT L-BFGS: " << iterations << " iterations, energy " << f0 << " -> " << bestF * f0 << '\n';
    } catch (const std::exception& e) {
        // Line-search failures happen near the optimum where the energy is
        // flat to machine precision; the best point seen is kept.
        std::cout << "CVT L-BFGS stopped: " << e.what() << ", energy " << f0 << " -> " << bestF * f0 << '\n';
    }
    for (int i = 0; i < n; i++) seeds[i] = bestY.segment<3>(3 * i) * scale;
    return bestF * f0;
}

int CvtSegmentation::labelVertices(const std::vector<V3d>& seeds, int minChartVertices,
                                   std::vector<int>& vertexLabels) const
{
    const int64_t vertexCount = static_cast<int64_t>(positions_.size());
    vertexLabels.assign(vertexCount, -1);
    if (vertexCount == 0 || seeds.empty()) return 0;

    PointHashGrid grid;
    grid.build(seeds, spacingForSeedCount(static_cast<int>(seeds.size())));
    std::vector<int> nearest(vertexCount);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t v = 0; v < vertexCount; v++) nearest[v] = grid.nearest(positions_[v]);

    // Vertex adjacency (CSR) from the triangles.
    std::vector<int> offsets(vertexCount + 1, 0);
    for (const auto& tri : triangles_) {
        for (int k = 0; k < 3; k++) offsets[tri[k] + 1] += 2;
    }
    for (int64_t v = 0; v < vertexCount; v++) offsets[v + 1] += offsets[v];
    std::vector<int> adjacency(offsets.back());
    {
        std::vector<int> fill(offsets.begin(), offsets.end() - 1);
        for (const auto& tri : triangles_) {
            for (int k = 0; k < 3; k++) {
                adjacency[fill[tri[k]]++] = tri[(k + 1) % 3];
                adjacency[fill[tri[k]]++] = tri[(k + 2) % 3];
            }
        }
    }

    // Connected pieces of each Voronoi cell. Euclidean cells can wrap around
    // thin parts of the surface and leave several pieces.
    std::vector<int> component(vertexCount, -1);
    std::vector<int> componentSize;
    std::vector<int> componentLabel;
    std::vector<int> stack;
    for (int64_t s = 0; s < vertexCount; s++) {
        if (component[s] >= 0) continue;
        const int id = static_cast<int>(componentSize.size());
        const int label = nearest[s];
        componentSize.push_back(0);
        componentLabel.push_back(label);
        component[s] = id;
        stack.assign(1, static_cast<int>(s));
        while (!stack.empty()) {
            const int v = stack.back();
            stack.pop_back();
            componentSize[id]++;
            for (int k = offsets[v]; k < offsets[v + 1]; k++) {
                const int u = adjacency[k];
                if (component[u] < 0 && nearest[u] == label) {
                    component[u] = id;
                    stack.push_back(u);
                }
            }
        }
    }

    // Keep the largest piece of each cell if it is big enough.
    const int componentCount = static_cast<int>(componentSize.size());
    std::vector<int> largest(seeds.size(), -1);
    for (int c = 0; c < componentCount; c++) {
        int& best = largest[componentLabel[c]];
        if (best < 0 || componentSize[c] > componentSize[best]) best = c;
    }
    std::vector<char> keep(componentCount, 0);
    bool anyKept = false;
    for (int best : largest) {
        if (best >= 0 && componentSize[best] >= minChartVertices) {
            keep[best] = 1;
            anyKept = true;
        }
    }
    if (!anyKept) {
        for (int best : largest) {
            if (best >= 0) keep[best] = 1;
        }
    }

    // Absorb the rest breadth-first from the kept pieces, which keeps every
    // chart connected.
    std::vector<int> queue;
    queue.reserve(vertexCount);
    for (int64_t v = 0; v < vertexCount; v++) {
        if (keep[component[v]]) {
            vertexLabels[v] = nearest[v];
            queue.push_back(static_cast<int>(v));
        }
    }
    for (size_t head = 0; head < queue.size(); head++) {
        const int v = queue[head];
        for (int k = offsets[v]; k < offsets[v + 1]; k++) {
            const int u = adjacency[k];
            if (vertexLabels[u] < 0) {
                vertexLabels[u] = vertexLabels[v];
                queue.push_back(u);
            }
        }
    }
    // Pieces with no kept chart in reach (small separate shells) keep their
    // own cell label.
    for (int64_t v = 0; v < vertexCount; v++) {
        if (vertexLabels[v] < 0) vertexLabels[v] = nearest[v];
    }

    // Dense labels in first-seen vertex order.
    std::vector<int> dense(seeds.size(), -1);
    int count = 0;
    for (int& l : vertexLabels) {
        if (dense[l] < 0) dense[l] = count++;
        l = dense[l];
    }
    return count;
}

int CvtSegmentation::solve(const CvtSegmentationParams& params, std::vector<int>& vertexLabels,
                           const StatusHandler& statusHandler)
{
    energyHistory_.clear();
    seeds_.clear();
    if (triangles_.empty() || area_ <= 0.0) {
        vertexLabels.assign(positions_.size(), 0);
        return positions_.empty() ? 0 : 1;
    }

    int seedCount = params.seedCount;
    double spacing = params.spacing;
    const int maxTriangles = params.regularize ? params.regularization.maxTrianglesPerChart : 0;
    if (seedCount <= 0 && spacing <= 0.0 && maxTriangles > 0) {
        // Aim charts at ~50% of the limit: triangles on chart borders count
        // toward both sides, and the cap splits whatever still overflows.
        seedCount = std::max<int>(1, static_cast<int>(std::ceil(triangles_.size() / (0.375 * maxTriangles))));
    } else if (seedCount <= 0 && spacing <= 0.0) {
        seedCount = std::max<int>(1, static_cast<int>(positions_.size()) / std::max(1, params.verticesPerSeed));
    }
    if (spacing <= 0.0) spacing = spacingForSeedCount(seedCount);
    const double minSpacing = params.minSpacingInEdgeLengths * averageEdge_;
    if (spacing < minSpacing) {
        std::cout << "CVT spacing " << spacing << " is below " << params.minSpacingInEdgeLengths
                  << " average edge lengths; using " << minSpacing << '\n';
        spacing = minSpacing;
        seedCount = 0;
    }
    if (seedCount <= 0) {
        seedCount = std::max(1, static_cast<int>(std::lround(2.0 * area_ / (std::sqrt(3.0) * spacing * spacing))));
    }

    // Dart throwing stalls well short of a hexagonal packing, so the disk
    // radius is shrunk to make room for roughly the requested number of seeds.
    const double radius = 0.8 * spacing;
    const int candidates = std::max(1, params.candidatesPerSeed) * seedCount;
    Report(statusHandler, "CVT blue-noise sampling", 0.0f);
    std::vector<V3d> seeds = sampleBlueNoise(radius, seedCount, candidates, params.randomSeed);
    std::cout << "CVT seeds: " << seeds.size() << " / " << seedCount << " requested, spacing " << spacing
              << ", average edge " << averageEdge_ << '\n';

    const int totalSteps = std::max(1, params.lloydIterations + 1);
    for (int k = 0; k < params.lloydIterations; k++) {
        if (!Report(statusHandler, "CVT Lloyd iteration", float(k) / totalSteps)) break;
        energyHistory_.push_back(lloydStep(seeds));
    }
    if (params.lbfgsIterations > 0 && Report(statusHandler, "CVT L-BFGS", float(params.lloydIterations) / totalSteps)) {
        energyHistory_.push_back(
            optimizeLBFGS(seeds, params.lbfgsIterations, params.lbfgsHistory, params.relativeTolerance));
    } else {
        energyHistory_.push_back(restrictedVoronoi(seeds).totalEnergy);
    }

    Report(statusHandler, "CVT labeling", 1.0f);
    int count = labelVertices(seeds, params.minChartVertices, vertexLabels);
    if (params.regularize) {
        Report(statusHandler, "CVT label regularization", 1.0f);
        const LabelGraph graph(positions_, triangles_);
        count = graph.regularize(vertexLabels, params.regularization);
    }
    seeds_ = std::move(seeds);
    std::cout << "CVT charts: " << count << '\n';
    return count;
}
