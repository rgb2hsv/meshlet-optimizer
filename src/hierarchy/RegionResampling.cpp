/*
 * RegionResampling.cpp
 *
 * See RegionResampling.hpp for an overview.
 */
#include <charts/CvtSegmentation.hpp>
#include <core/Check.hpp>
#include <core/Types.hpp>
#include <hierarchy/RegionResampling.hpp>
#include <iostream>
#include <mesh/MeshDecimation.hpp>

#include <CDT.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <unordered_map>
#include <unordered_set>

using namespace mopt;

namespace
{
using V2 = RegionResampler::V2;
using V3d = RegionResampler::V3d;
using Tri = RegionResampler::Tri;

// Uniform grid of triangle buckets for point location in a 2D triangulation.
class TriangleGrid2D
{
public:
    TriangleGrid2D(const std::vector<V2>& pts, const std::vector<Tri>& tris) : pts_(pts), tris_(tris)
    {
        lo_ = V2(std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
        V2 hi = -lo_;
        double area = 0.0;
        for (const Tri& t : tris_) {
            for (int k = 0; k < 3; k++) {
                lo_ = lo_.cwiseMin(pts_[t[k]]);
                hi = hi.cwiseMax(pts_[t[k]]);
            }
            area += std::abs(signedArea(t));
        }
        const double cell = std::max(std::sqrt(4.0 * area / std::max<size_t>(1, tris_.size())), 1e-12);
        cell_ = cell;
        nx_ = std::clamp(static_cast<int>((hi.x() - lo_.x()) / cell) + 1, 1, 1024);
        ny_ = std::clamp(static_cast<int>((hi.y() - lo_.y()) / cell) + 1, 1, 1024);
        cell_ = std::max((hi.x() - lo_.x()) / nx_, (hi.y() - lo_.y()) / ny_);
        cell_ = std::max(cell_, 1e-12);
        buckets_.resize(static_cast<size_t>(nx_) * ny_);
        for (int i = 0; i < static_cast<int>(tris_.size()); i++) {
            V2 a = pts_[tris_[i][0]], b = a;
            for (int k = 1; k < 3; k++) {
                a = a.cwiseMin(pts_[tris_[i][k]]);
                b = b.cwiseMax(pts_[tris_[i][k]]);
            }
            const int x0 = cellX(a.x()), x1 = cellX(b.x()), y0 = cellY(a.y()), y1 = cellY(b.y());
            for (int y = y0; y <= y1; y++) {
                for (int x = x0; x <= x1; x++) buckets_[static_cast<size_t>(y) * nx_ + x].push_back(i);
            }
        }
    }

    // Triangle containing p and p's barycentric coordinates in it.
    bool locate(const V2& p, int& tri, V3d& bary) const
    {
        const int x = cellX(p.x()), y = cellY(p.y());
        if (p.x() < lo_.x() - cell_ || p.y() < lo_.y() - cell_) return false;
        double bestMin = -1e-9;
        tri = -1;
        for (int i : buckets_[static_cast<size_t>(y) * nx_ + x]) {
            const V3d b = barycentric(tris_[i], p);
            const double m = b.minCoeff();
            if (m >= bestMin) {
                bestMin = m;
                tri = i;
                bary = b;
            }
        }
        return tri >= 0;
    }

    // Like locate(), but for a point just outside the triangulation picks the
    // nearby triangle it is least outside of (barycentrics may be negative,
    // which extrapolates across that triangle's plane).
    bool locateNearest(const V2& p, int& tri, V3d& bary) const
    {
        if (locate(p, tri, bary)) return true;
        const int cx = cellX(p.x()), cy = cellY(p.y());
        double bestMin = -std::numeric_limits<double>::max();
        tri = -1;
        for (int y = std::max(0, cy - 1); y <= std::min(ny_ - 1, cy + 1); y++) {
            for (int x = std::max(0, cx - 1); x <= std::min(nx_ - 1, cx + 1); x++) {
                for (int i : buckets_[static_cast<size_t>(y) * nx_ + x]) {
                    const V3d b = barycentric(tris_[i], p);
                    if (b.minCoeff() > bestMin) {
                        bestMin = b.minCoeff();
                        tri = i;
                        bary = b;
                    }
                }
            }
        }
        return tri >= 0;
    }

    double signedArea(const Tri& t) const
    {
        const V2 &a = pts_[t[0]], &b = pts_[t[1]], &c = pts_[t[2]];
        return 0.5 * ((b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x()));
    }

private:
    int cellX(double v) const { return std::clamp(static_cast<int>((v - lo_.x()) / cell_), 0, nx_ - 1); }
    int cellY(double v) const { return std::clamp(static_cast<int>((v - lo_.y()) / cell_), 0, ny_ - 1); }
    V3d barycentric(const Tri& t, const V2& p) const
    {
        const V2 &a = pts_[t[0]], &b = pts_[t[1]], &c = pts_[t[2]];
        const double d = (b.y() - c.y()) * (a.x() - c.x()) + (c.x() - b.x()) * (a.y() - c.y());
        if (std::abs(d) < 1e-300) return V3d(-1, -1, -1);
        const double l1 = ((b.y() - c.y()) * (p.x() - c.x()) + (c.x() - b.x()) * (p.y() - c.y())) / d;
        const double l2 = ((c.y() - a.y()) * (p.x() - c.x()) + (a.x() - c.x()) * (p.y() - c.y())) / d;
        return V3d(l1, l2, 1.0 - l1 - l2);
    }

    const std::vector<V2>& pts_;
    const std::vector<Tri>& tris_;
    V2 lo_;
    double cell_ = 1.0;
    int nx_ = 1, ny_ = 1;
    std::vector<std::vector<int>> buckets_;
};

bool Report(const std::exception& e, int region)
{
    std::cout << "Region " << region << " kept its original triangles: " << e.what() << '\n';
    return false;
}
}  // namespace

RegionResampler::RegionResampler(const Mesh& cutMesh, const std::vector<Mosaic>& mosaics,
                                 const RegionResampleParams& params)
    : positions_(cutMesh.vertexLocations.begin(), cutMesh.vertexLocations.end()),
      normals_(cutMesh.vertexNormals.begin(), cutMesh.vertexNormals.end()),
      params_(params)
{
    regions_.resize(mosaics.size());
    for (size_t r = 0; r < mosaics.size(); r++) {
        const Mosaic& mosaic = mosaics[r];
        Region& region = regions_[r];
        const int n = static_cast<int>(mosaic.vertexes.size());
        region.uv.resize(n);
        region.meshId.resize(n);
        for (int i = 0; i < n; i++) {
            region.uv[i] = V2(mosaic.vertexes[i].uv.x(), mosaic.vertexes[i].uv.y());
            region.meshId[i] = mosaic.vertexes[i].meshId;
        }
        region.faces.reserve(mosaic.faces.size());
        for (const MosaicFace& f : mosaic.faces) {
            region.faces.push_back({static_cast<int>(mosaic.vertexMap.at(f.mapFace.x())),
                                    static_cast<int>(mosaic.vertexMap.at(f.mapFace.y())),
                                    static_cast<int>(mosaic.vertexMap.at(f.mapFace.z()))});
        }
        // Boundary: edges used by a single face of the region.
        const auto key = [](int a, int b) { return (uint64_t(uint32_t(a)) << 32) | uint32_t(b); };
        std::unordered_set<uint64_t> directed;
        for (const Tri& t : region.faces) {
            for (int k = 0; k < 3; k++) directed.insert(key(t[k], t[(k + 1) % 3]));
        }
        region.boundaryIndex.assign(n, -1);
        double signedTotal = 0.0;
        for (const Tri& t : region.faces) {
            for (int k = 0; k < 3; k++) {
                const int a = t[k], b = t[(k + 1) % 3];
                if (!directed.contains(key(b, a))) region.boundaryEdges.emplace_back(a, b);
            }
            const V2 &p = region.uv[t[0]], &q = region.uv[t[1]], &w = region.uv[t[2]];
            signedTotal += (q.x() - p.x()) * (w.y() - p.y()) - (q.y() - p.y()) * (w.x() - p.x());
        }
        region.mirrored = signedTotal < 0.0;  // LSCM may mirror a region
        for (const auto& [a, b] : region.boundaryEdges) {
            for (int v : {a, b}) {
                if (region.boundaryIndex[v] < 0) {
                    region.boundaryIndex[v] = static_cast<int>(region.boundary.size());
                    region.boundary.push_back(region.meshId[v]);
                }
            }
        }
        const int B = static_cast<int>(region.boundary.size());
        region.originalInterior = n - B;
        region.seeds.resize(B);
        for (int i = 0; i < n; i++) {
            const int b = region.boundaryIndex[i];
            if (b >= 0) region.seeds[b] = V3d(region.uv[i].x(), region.uv[i].y(), 0.0);
        }
    }
    const int regionCount = static_cast<int>(regions_.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int r = 0; r < regionCount; r++) {
        Region& region = regions_[r];
        region.folded = foldsOver(region);
    }
}

bool RegionResampler::foldsOver(const Region& region) const
{
    // Faces are oriented like the region (+1, or -1 if LSCM mirrored it); the
    // boundary edges, taken from the faces, then wind once around every point
    // of an unfolded region. Winding numbers by signed crossings of a ray
    // towards +x (the crossing-number form of the winding number).
    const double sign = region.mirrored ? -1.0 : 1.0;
    const auto cross = [](const V2& a, const V2& b, const V2& p) {
        return (b.x() - a.x()) * (p.y() - a.y()) - (p.x() - a.x()) * (b.y() - a.y());
    };
    for (const Tri& t : region.faces) {
        const V2 &a = region.uv[t[0]], &b = region.uv[t[1]], &c = region.uv[t[2]];
        if (sign * cross(a, b, c) <= 0.0) return true;  // flipped or degenerate face
        const V2 p = (a + b + c) / 3.0;
        int winding = 0;
        for (const auto& [i, j] : region.boundaryEdges) {
            const V2 &e0 = region.uv[i], &e1 = region.uv[j];
            if (e0.y() <= p.y()) {
                if (e1.y() > p.y() && cross(e0, e1, p) > 0.0) winding++;
            } else if (e1.y() <= p.y() && cross(e0, e1, p) < 0.0) {
                winding--;
            }
        }
        if (winding != static_cast<int>(sign)) return true;
    }
    return false;
}

bool RegionResampler::decimate(Region& region) const
{
    // The region as its own mesh, in 3D, boundary vertices locked.
    Mesh local;
    const int n = static_cast<int>(region.meshId.size());
    for (int i = 0; i < n; i++) local.vertexLocations.push_back(positions_[region.meshId[i]]);
    for (const Tri& t : region.faces) {
        local.triIndexes.push_back(Vector3u(uint32_t(t[0]), uint32_t(t[1]), uint32_t(t[2])));
    }
    std::vector<uint32_t> lockedIndexes;
    for (int i = 0; i < n; i++) {
        if (region.boundaryIndex[i] >= 0) lockedIndexes.push_back(static_cast<uint32_t>(i));
    }
    const int target =
        static_cast<int>(std::lround(std::clamp(params_.interiorFraction, 0.0, 1.0) * region.originalInterior));
    const int remove = region.originalInterior - target;
    MeshDecimation decimator;
    decimator.setLockedVertices(lockedIndexes);
    if (remove > 0 && n > 0) decimator.solve(local, static_cast<float>(remove) / n);
    const std::vector<uint32_t>& outIndex = decimator.outputIndexOf();
    if (outIndex.size() != static_cast<size_t>(n)) return false;

    // Output vertex -> boundary index, or interior slot.
    std::vector<int> corner(local.vertexLocations.size(), std::numeric_limits<int>::min());
    for (int i = 0; i < n; i++) {
        const int b = region.boundaryIndex[i];
        if (b < 0) continue;
        if (outIndex[i] == std::numeric_limits<uint32_t>::max()) return false;  // must survive
        corner[outIndex[i]] = b;
    }
    region.triangles.clear();
    region.interiorUv.clear();
    region.interiorPos.clear();
    // Output vertex -> the input vertex that survived as it, for its UV.
    std::vector<int> inputOf(local.vertexLocations.size(), -1);
    for (int i = 0; i < n; i++) {
        if (outIndex[i] != std::numeric_limits<uint32_t>::max()) inputOf[outIndex[i]] = i;
    }
    const auto addInterior = [&](uint32_t out) {
        region.interiorPos.push_back(local.vertexLocations[out]);
        region.interiorUv.push_back(inputOf[out] >= 0 ? region.uv[inputOf[out]] : V2(0, 0));
        return -static_cast<int>(region.interiorPos.size());
    };
    for (const auto& t : local.triIndexes) {
        Tri o;
        for (int k = 0; k < 3; k++) {
            int& c = corner[t[k]];
            if (c == std::numeric_limits<int>::min()) c = addInterior(t[k]);
            o[k] = c;
        }
        region.triangles.push_back(o);  // decimation keeps the 3D orientation
    }
    region.decimated = true;
    return !region.triangles.empty();
}

int RegionResampler::interiorPointCount() const
{
    int count = 0;
    for (const Region& r : regions_) count += static_cast<int>(r.seeds.size() - r.boundary.size());
    return count;
}

void RegionResampler::sampleBlueNoise()
{
    const int regionCount = static_cast<int>(regions_.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int r = 0; r < regionCount; r++) {
        Region& region = regions_[r];
        const int B = static_cast<int>(region.boundary.size());
        region.seeds.resize(B);
        const int target =
            static_cast<int>(std::lround(std::clamp(params_.interiorFraction, 0.0, 1.0) * region.originalInterior));
        if (target <= 0 || region.faces.empty() || region.folded) continue;
        const TriangleGrid2D locator(region.uv, region.faces);
        std::vector<double> cdf(region.faces.size());
        double area = 0.0;
        for (size_t t = 0; t < region.faces.size(); t++) {
            area += std::abs(locator.signedArea(region.faces[t]));
            cdf[t] = area;
        }
        if (!(area > 0.0)) continue;
        std::mt19937 rng(params_.randomSeed + 7919u * static_cast<uint32_t>(r));
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        // Hexagonal spacing for the interior points plus the boundary's share.
        double radius = 0.8 * std::sqrt(2.0 * area / (std::sqrt(3.0) * (target + 0.5 * B)));
        PointHashGrid grid;
        grid.reset(radius);
        for (int b = 0; b < B; b++) grid.insert(region.seeds[b], b);
        // Dart throwing stalls before maximal packing; shrink the radius until
        // the requested count fits.
        for (int pass = 0; pass < 6 && static_cast<int>(region.seeds.size()) - B < target; pass++) {
            const int candidates = std::max(1, params_.candidatesPerPoint) * target;
            for (int c = 0; c < candidates && static_cast<int>(region.seeds.size()) - B < target; c++) {
                const size_t t = std::min(
                    static_cast<size_t>(std::upper_bound(cdf.begin(), cdf.end(), uniform(rng) * area) - cdf.begin()),
                    region.faces.size() - 1);
                const double r1 = std::sqrt(uniform(rng)), r2 = uniform(rng);
                const Tri& f = region.faces[t];
                const V2 p =
                    (1.0 - r1) * region.uv[f[0]] + r1 * (1.0 - r2) * region.uv[f[1]] + r1 * r2 * region.uv[f[2]];
                const V3d p3(p.x(), p.y(), 0.0);
                if (!grid.anyWithin(p3, radius)) {
                    grid.insert(p3, static_cast<int>(region.seeds.size()));
                    region.seeds.push_back(p3);
                }
            }
            radius *= 0.85;
        }
    }
}

void RegionResampler::relaxCvt()
{
    const int regionCount = static_cast<int>(regions_.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int r = 0; r < regionCount; r++) {
        Region& region = regions_[r];
        const size_t B = region.boundary.size();
        if (region.folded || region.seeds.size() <= B || params_.lloydIterations <= 0) continue;
        std::vector<V3d> points(region.uv.size());
        for (size_t i = 0; i < region.uv.size(); i++) points[i] = V3d(region.uv[i].x(), region.uv[i].y(), 0.0);
        std::vector<Eigen::Vector3i> tris;
        tris.reserve(region.faces.size());
        for (const Tri& t : region.faces) tris.emplace_back(t[0], t[1], t[2]);
        const CvtSegmentation domain(std::move(points), std::move(tris));
        const TriangleGrid2D locator(region.uv, region.faces);
        int tri;
        V3d bary;
        for (int it = 0; it < params_.lloydIterations; it++) {
            const auto cells = domain.restrictedVoronoi(region.seeds);
            for (size_t i = B; i < region.seeds.size(); i++) {
                if (!(cells.mass[i] > 0.0)) continue;
                const V3d c = cells.centroid(i);
                if (locator.locate(V2(c.x(), c.y()), tri, bary)) region.seeds[i] = V3d(c.x(), c.y(), 0.0);
            }
        }
    }
}

void RegionResampler::keepOriginal(Region& region) const
{
    // interiorUv holds the original interior vertices; lift() maps each back
    // onto its own position (exactly, through the face it is a corner of).
    region.failed = true;
    region.decimated = false;
    region.triangles.clear();
    region.interiorUv.clear();
    region.interiorPos.clear();
    std::vector<int> interiorIndex(region.uv.size(), -1);
    for (const Tri& t : region.faces) {
        Tri o;
        for (int k = 0; k < 3; k++) {
            const int v = t[k];
            if (region.boundaryIndex[v] >= 0) {
                o[k] = region.boundaryIndex[v];
            } else {
                if (interiorIndex[v] < 0) {
                    interiorIndex[v] = static_cast<int>(region.interiorUv.size());
                    region.interiorUv.push_back(region.uv[v]);
                    region.interiorPos.push_back(positions_[region.meshId[v]]);
                }
                o[k] = -(interiorIndex[v] + 1);
            }
        }
        region.triangles.push_back(o);
    }
}

void RegionResampler::triangulate()
{
    const int regionCount = static_cast<int>(regions_.size());
    // Regions simplified by decimation instead: folded in UV, or with a
    // triangulation that would not reproduce the boundary.
    std::vector<char> needsDecimation(regionCount, 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int r = 0; r < regionCount; r++) {
        Region& region = regions_[r];
        region.failed = false;
        region.decimated = false;
        region.triangles.clear();
        region.interiorUv.clear();
        region.interiorPos.clear();
        const int B = static_cast<int>(region.boundary.size());
        if (region.faces.empty()) continue;
        if (region.folded || B < 3) {
            needsDecimation[r] = 1;
            continue;
        }
        const TriangleGrid2D locator(region.uv, region.faces);
        try {
            std::vector<CDT::V2d<double>> verts;
            verts.reserve(region.seeds.size());
            for (const V3d& p : region.seeds) verts.emplace_back(p.x(), p.y());
            std::vector<CDT::Edge> edges;
            edges.reserve(region.boundaryEdges.size());
            for (const auto& [a, b] : region.boundaryEdges) {
                edges.emplace_back(static_cast<CDT::VertInd>(region.boundaryIndex[a]),
                                   static_cast<CDT::VertInd>(region.boundaryIndex[b]));
            }
            // Points that coincide in UV (a pinched or folded region) would
            // merge boundary vertices the neighbouring region keeps separate.
            const CDT::DuplicatesInfo dup = CDT::RemoveDuplicatesAndRemapEdges(verts, edges);
            if (!dup.duplicates.empty()) {
                needsDecimation[r] = 1;
                continue;
            }
            CDT::Triangulation<double> cdt(CDT::VertexInsertionOrder::Auto,
                                           CDT::IntersectingConstraintEdges::TryResolve, 0.0);
            cdt.insertVertices(verts);
            cdt.insertEdges(edges);
            cdt.eraseOuterTrianglesAndHoles();
            // Crossing boundary edges (LSCM folded the region over itself) make
            // CDT add vertices the neighbour lacks and cover the wrong area;
            // either would crack the stitched mesh.
            double covered = 0.0;
            for (const CDT::Triangle& t : cdt.triangles) {
                const auto& a = cdt.vertices[t.vertices[0]];
                const auto& b = cdt.vertices[t.vertices[1]];
                const auto& c = cdt.vertices[t.vertices[2]];
                covered += 0.5 * std::abs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
            }
            // Area enclosed by the boundary loops (holes negative), which
            // is the region's area unless seams were simplified.
            double expected = 0.0;
            for (const auto& [a, b] : region.boundaryEdges) {
                expected += 0.5 * (region.uv[a].x() * region.uv[b].y() - region.uv[b].x() * region.uv[a].y());
            }
            expected = std::abs(expected);
            if (cdt.triangles.empty() || cdt.vertices.size() != verts.size() ||
                std::abs(covered - expected) > 1e-6 * expected + 1e-12) {
                needsDecimation[r] = 1;
                continue;
            }
            std::vector<int> corner(cdt.vertices.size(), std::numeric_limits<int>::min());
            for (const CDT::Triangle& t : cdt.triangles) {
                Tri o;
                for (int k = 0; k < 3; k++) {
                    const CDT::VertInd j = t.vertices[k];
                    int& c = corner[j];
                    if (c == std::numeric_limits<int>::min()) {
                        if (static_cast<int>(j) < B) {
                            c = static_cast<int>(j);
                        } else {
                            region.interiorUv.emplace_back(cdt.vertices[j].x, cdt.vertices[j].y);
                            c = -static_cast<int>(region.interiorUv.size());
                        }
                    }
                    o[k] = c;
                }
                if (region.mirrored) std::swap(o[1], o[2]);
                region.triangles.push_back(o);
            }
        } catch (const std::exception& e) {
            Report(e, r);
            needsDecimation[r] = 1;
        }
    }
    // Decimation builds Mesh objects, so it runs on this thread.
    for (int r = 0; r < regionCount; r++) {
        if (!needsDecimation[r]) continue;
        Region& region = regions_[r];
        region.triangles.clear();
        region.interiorUv.clear();
        if (!decimate(region)) keepOriginal(region);
    }
}

void RegionResampler::lift()
{
    // Interior points to 3D by barycentric interpolation through the UV
    // triangle that contains them (exactly the original vertices for a region
    // that kept its triangles). Decimated regions are already in 3D.
    const int regionCount = static_cast<int>(regions_.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int r = 0; r < regionCount; r++) {
        Region& region = regions_[r];
        if (region.decimated) continue;
        region.interiorPos.clear();
        region.interiorPos.reserve(region.interiorUv.size());
        const TriangleGrid2D locator(region.uv, region.faces);
        for (const V2& p : region.interiorUv) {
            int tri;
            V3d bary;
            if (locator.locateNearest(p, tri, bary)) {
                const Tri& f = region.faces[tri];
                region.interiorPos.push_back(float(bary[0]) * positions_[region.meshId[f[0]]] +
                                             float(bary[1]) * positions_[region.meshId[f[1]]] +
                                             float(bary[2]) * positions_[region.meshId[f[2]]]);
            } else {
                size_t nearest = 0;
                for (size_t i = 1; i < region.uv.size(); i++) {
                    if ((region.uv[i] - p).squaredNorm() < (region.uv[nearest] - p).squaredNorm()) nearest = i;
                }
                region.interiorPos.push_back(positions_[region.meshId[nearest]]);
            }
        }
        // Triangles that face against the surface under them: the
        // triangulation folds over in 3D (sharp seams cut by chords, strong
        // LSCM distortion). Such a region is decimated instead below.
        region.liftFolded = !region.failed && liftFoldsOver(region);
    }
    for (int r = 0; r < regionCount; r++) {
        Region& region = regions_[r];
        if (!region.liftFolded) continue;
        region.triangles.clear();
        region.interiorUv.clear();
        region.interiorPos.clear();
        if (!decimate(region)) keepOriginal(region);
    }
}

bool RegionResampler::liftFoldsOver(const Region& region) const
{
    const TriangleGrid2D locator(region.uv, region.faces);
    const auto corner3d = [&](int c) { return c >= 0 ? positions_[region.boundary[c]] : region.interiorPos[-c - 1]; };
    const auto cornerUv = [&](int c) {
        return c >= 0 ? V2(region.seeds[c].x(), region.seeds[c].y()) : region.interiorUv[-c - 1];
    };
    for (const Tri& t : region.triangles) {
        const Vector3f p0 = corner3d(t[0]), p1 = corner3d(t[1]), p2 = corner3d(t[2]);
        const Vector3f n = (p1 - p0).cross(p2 - p0);
        // The original face under the triangle's centroid, in UV.
        const V2 c = (cornerUv(t[0]) + cornerUv(t[1]) + cornerUv(t[2])) / 3.0;
        int f;
        V3d bary;
        if (!locator.locateNearest(c, f, bary)) continue;
        const Tri& o = region.faces[f];
        const Vector3f q0 = positions_[region.meshId[o[0]]];
        const Vector3f reference = (positions_[region.meshId[o[1]]] - q0).cross(positions_[region.meshId[o[2]]] - q0);
        if (n.dot(reference) <= 0.0f) return true;
    }
    return false;
}

void RegionResampler::project(const SurfaceProjector& projector)
{
    // New interior vertices only: kept boundary vertices are untouched, and a
    // region that kept its triangles keeps its vertices too.
    if (!projector) return;
    const int regionCount = static_cast<int>(regions_.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int r = 0; r < regionCount; r++) {
        Region& region = regions_[r];
        if (region.failed) continue;
        for (Vector3f& p : region.interiorPos) p = projector(p);
    }
}

ResampledSurface RegionResampler::reconstitute() const
{
    // Weld the regions back into one mesh: regions share their boundary
    // vertices (the same cut-mesh vertex), interiors are their own.
    ResampledSurface surface;
    std::unordered_map<uint32_t, uint32_t> boundaryOut;
    for (int r = 0; r < static_cast<int>(regions_.size()); r++) {
        const Region& region = regions_[r];
        std::vector<uint32_t> bIndex(region.boundary.size());
        for (size_t b = 0; b < region.boundary.size(); b++) {
            const uint32_t meshId = region.boundary[b];
            auto [it, inserted] =
                boundaryOut.emplace(meshId, static_cast<uint32_t>(surface.mesh.vertexLocations.size()));
            if (inserted) {
                surface.mesh.vertexLocations.push_back(positions_[meshId]);
                surface.vertexRegion.push_back(-1);
                surface.resampled.push_back(0);
            }
            bIndex[b] = it->second;
        }
        // Interior vertices that some triangle uses (a removed seam ear can
        // leave its vertex unused).
        std::vector<uint32_t> interiorOut(region.interiorPos.size(), std::numeric_limits<uint32_t>::max());
        for (const Tri& t : region.triangles) {
            for (int c : t) {
                if (c < 0) interiorOut[-c - 1] = 0;
            }
        }
        for (size_t i = 0; i < region.interiorPos.size(); i++) {
            if (interiorOut[i] == std::numeric_limits<uint32_t>::max()) continue;
            interiorOut[i] = static_cast<uint32_t>(surface.mesh.vertexLocations.size());
            surface.mesh.vertexLocations.push_back(region.interiorPos[i]);
            surface.vertexRegion.push_back(r);
            surface.resampled.push_back(region.failed ? 0 : 1);
        }
        const auto map = [&](int c) { return c >= 0 ? bIndex[c] : interiorOut[-c - 1]; };
        // UV of a corner in the region's layout.
        const auto uvOf = [&](int c) {
            const V2 p = c >= 0 ? V2(region.seeds[c].x(), region.seeds[c].y()) : region.interiorUv[-c - 1];
            return Vector2f(static_cast<float>(p.x()), static_cast<float>(p.y()));
        };
        for (const Tri& t : region.triangles) {
            surface.mesh.triIndexes.push_back(Vector3u(map(t[0]), map(t[1]), map(t[2])));
            surface.faceRegion.push_back(r);
            for (int c : t) surface.mesh.textureMap.push_back(uvOf(c));
        }
        surface.originalInterior += region.originalInterior;
        for (uint32_t o : interiorOut) surface.resampledInterior += o != std::numeric_limits<uint32_t>::max();
        surface.foldedRegions += region.folded;
        surface.decimatedRegions += region.decimated;
        surface.liftFoldedRegions += region.liftFolded;
        surface.failedRegions += region.failed;
    }
    surface.boundaryVertices = static_cast<int>(boundaryOut.size());
    // Normals with the input's sign convention, compared at kept boundary
    // vertices, which still have the input's normals.
    surface.mesh.updateVertexNormals();
    if (normals_.size() == positions_.size()) {
        double agreement = 0.0;
        for (const auto& [meshId, outId] : boundaryOut) {
            agreement += normals_[meshId].dot(surface.mesh.vertexNormals[outId]);
        }
        if (agreement < 0.0) {
            surface.flipNormals = true;
            surface.mesh.updateVertexNormals(true);
        }
    }
    // Every edge used once in each direction: the regions fit back together.
    std::unordered_map<uint64_t, int> directed;
    directed.reserve(surface.mesh.triIndexes.size() * 3);
    for (const auto& t : surface.mesh.triIndexes) {
        const uint32_t v[3] = {t.x(), t.y(), t.z()};
        for (int k = 0; k < 3; k++) directed[(uint64_t(v[k]) << 32) | v[(k + 1) % 3]]++;
    }
    for (const auto& [e, uses] : directed) {
        const auto reverse = directed.find((e << 32) | (e >> 32));
        if (uses != 1 || reverse == directed.end() || reverse->second != 1) surface.openEdges++;
    }
    // Redundant triangles: the same three vertices twice (a pair with opposite
    // windings would still pass the edge test above), or degenerate ones.
    {
        std::unordered_set<uint64_t> seen;
        seen.reserve(surface.mesh.triIndexes.size());
        double meanArea = 0.0;
        std::vector<double> areas(surface.mesh.triIndexes.size());
        for (size_t f = 0; f < surface.mesh.triIndexes.size(); f++) {
            const Vector3u t = surface.mesh.triIndexes[f];
            const Vector3f a = surface.mesh.vertexLocations[t.x()];
            areas[f] =
                0.5 * ((surface.mesh.vertexLocations[t.y()] - a).cross(surface.mesh.vertexLocations[t.z()] - a)).norm();
            meanArea += areas[f];
        }
        meanArea /= std::max<size_t>(1, areas.size());
        for (size_t f = 0; f < surface.mesh.triIndexes.size(); f++) {
            const Vector3u t = surface.mesh.triIndexes[f];
            uint32_t v[3] = {t.x(), t.y(), t.z()};
            std::sort(v, v + 3);
            if (v[0] == v[1] || v[1] == v[2] || !(areas[f] > 1e-9 * meanArea)) surface.degenerateTriangles++;
            // 21 bits per index is enough for these meshes; collisions only
            // over-report.
            const uint64_t key = (uint64_t(v[0]) << 42) | (uint64_t(v[1] & 0x1FFFFF) << 21) | (v[2] & 0x1FFFFF);
            if (!seen.insert(key).second) surface.duplicateTriangles++;
        }
    }
    {
        size_t undirectedEdges = 0;
        for (const auto& [e, uses] : directed) undirectedEdges += (e >> 32) < (e & 0xffffffffu) ? 1 : 0;
        // Edges whose reverse is missing count once more.
        for (const auto& [e, uses] : directed) {
            if ((e >> 32) > (e & 0xffffffffu) && !directed.count((e << 32) | (e >> 32))) undirectedEdges++;
        }
        surface.eulerCharacteristic = static_cast<long long>(surface.mesh.vertexLocations.size()) -
                                      static_cast<long long>(undirectedEdges) +
                                      static_cast<long long>(surface.mesh.triIndexes.size());
    }
    std::cout << "Reconstituted " << regions_.size() << " regions: interior " << surface.originalInterior << " -> "
              << surface.resampledInterior << " vertices, " << surface.boundaryVertices << " boundary kept; "
              << surface.foldedRegions << " folded, " << surface.liftFoldedRegions << " folded in 3D, "
              << surface.decimatedRegions << " decimated, " << surface.failedRegions << " kept original triangles; "
              << surface.openEdges << " unmatched edges, " << surface.duplicateTriangles << " duplicate and "
              << surface.degenerateTriangles << " degenerate triangles; V-E+F = " << surface.eulerCharacteristic
              << '\n';
    return surface;
}
