#include <flattening/IsometricUvs.hpp>

#include <core/Check.hpp>

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <tuple>
#include <utility>

namespace mopt
{
namespace
{
using Vec2d = Eigen::Vector2d;
using Triangle2 = std::array<Vec2d, 3>;

double SignedArea(const Triangle2& t) { return 0.5 * ((t[1] - t[0]).x() * (t[2] - t[0]).y() - (t[1] - t[0]).y() * (t[2] - t[0]).x()); }
Vec2d Centroid(const Triangle2& t) { return (t[0] + t[1] + t[2]) / 3.0; }
Vec2d Rotate(const Vec2d& v, double angle)
{
    const double c = std::cos(angle), s = std::sin(angle);
    return Vec2d(c * v.x() - s * v.y(), s * v.x() + c * v.y());
}
Vec2d Perp(const Vec2d& v) { return Vec2d(-v.y(), v.x()); }

// The 3D triangle laid flat: corner 0 at the origin, corner 1 on +x, corner 2 above (counter-clockwise).
Triangle2 FlattenTriangle(const Vector3f& p0, const Vector3f& p1, const Vector3f& p2)
{
    const Eigen::Vector3d d1 = (p1 - p0).cast<double>(), d2 = (p2 - p0).cast<double>();
    const Eigen::Vector3d x = d1.normalized();
    const Eigen::Vector3d y = x.cross(d2).cross(x).normalized();
    return {Vec2d::Zero(), Vec2d(d1.dot(x), 0.0), Vec2d(d2.dot(x), d2.dot(y))};
}

// Smallest t > 0 at which the ray from `from` along `direction` meets the boundary of triangle p.
double RayExit(const Vec2d& from, const Vec2d& direction, const Triangle2& p)
{
    double best = std::numeric_limits<double>::infinity();
    for (int k = 0; k < 3; k++) {
        const Vec2d a = p[k], e = p[(k + 1) % 3] - p[k];
        // from + t direction = a + s e
        const double det = direction.x() * (-e.y()) - direction.y() * (-e.x());
        if (std::abs(det) < 1e-300) continue;
        const Vec2d r = a - from;
        const double t = (r.x() * (-e.y()) - r.y() * (-e.x())) / det;
        const double s = (direction.x() * r.y() - direction.y() * r.x()) / det;
        if (t > 0.0 && s >= -1e-12 && s <= 1.0 + 1e-12) best = std::min(best, t);
    }
    return best;
}

double Median(std::vector<double> values)
{
    if (values.empty()) return 1.0;
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    return values[mid];
}

// Pairs of corners (face, corner) that are the same mesh vertex in two triangles of one region that share
// an edge.
struct CornerPair {
    uint32_t faceA, faceB;
    int cornerA, cornerB;
};

std::vector<CornerPair> SharedCorners(const Mesh& mesh, const std::vector<int>& faceRegion)
{
    std::map<std::pair<uint32_t, uint32_t>, std::vector<std::pair<uint32_t, int>>> edges;  // edge -> (face, corner of the edge's start)
    for (uint32_t f = 0; f < mesh.triIndexes.size(); f++) {
        const Vector3u& t = mesh.triIndexes[f];
        for (int c = 0; c < 3; c++) edges[std::minmax(t[c], t[(c + 1) % 3])].push_back({f, c});
    }
    std::vector<CornerPair> pairs;
    for (const auto& [edge, faces] : edges) {
        if (faces.size() != 2) continue;  // border of the mesh, or non-manifold: no pairing
        const auto [fa, ca] = faces[0];
        const auto [fb, cb] = faces[1];
        if (faceRegion[fa] != faceRegion[fb]) continue;
        const Vector3u &ta = mesh.triIndexes[fa], &tb = mesh.triIndexes[fb];
        for (uint32_t v : {edge.first, edge.second}) {
            const int ka = ta[0] == v ? 0 : ta[1] == v ? 1 : 2;
            const int kb = tb[0] == v ? 0 : tb[1] == v ? 1 : 2;
            pairs.push_back({fa, fb, ka, kb});
        }
    }
    return pairs;
}

double MeanGap(const std::vector<Triangle2>& uv, const std::vector<CornerPair>& pairs)
{
    double sum = 0.0;
    for (const CornerPair& p : pairs) sum += (uv[p.faceA][p.cornerA] - uv[p.faceB][p.cornerB]).norm();
    return pairs.empty() ? 0.0 : sum / pairs.size();
}

// Step 1: every triangle isometric, inscribed in its atlas triangle, at its region's median scale.
void InscribeIsometricTriangles(const Mesh& mesh, const std::vector<int>& faceRegion, std::vector<Triangle2>& uv)
{
    const size_t F = mesh.triIndexes.size();
    std::vector<Triangle2> flat(F);
    std::map<int, std::vector<double>> scaleSamples;
    for (size_t f = 0; f < F; f++) {
        const Vector3u& t = mesh.triIndexes[f];
        flat[f] = FlattenTriangle(mesh.vertexLocations[t[0]], mesh.vertexLocations[t[1]], mesh.vertexLocations[t[2]]);
        const double pArea = std::abs(SignedArea(uv[f])), qArea = SignedArea(flat[f]);
        if (pArea > 0.0 && qArea > 0.0) scaleSamples[faceRegion[f]].push_back(std::sqrt(pArea / qArea));
    }
    std::map<int, double> targetRatio;
    for (auto& [region, samples] : scaleSamples) targetRatio[region] = Median(std::move(samples));

    for (size_t f = 0; f < F; f++) {
        const Triangle2& p = uv[f];
        Triangle2 q = flat[f];
        const double pArea = std::abs(SignedArea(p)), qArea = SignedArea(q);
        if (!(pArea > 0.0) || !(qArea > 0.0)) continue;  // degenerate: left as it is
        const Vec2d pCenter = Centroid(p), qCenter = Centroid(q);
        // Turn corner 0 toward the atlas triangle's corner 0, and match the area.
        const Vec2d from = q[0] - qCenter, to = p[0] - pCenter;
        const double angle = std::atan2(from.x() * to.y() - from.y() * to.x(), from.dot(to));
        double scale = std::sqrt(pArea / qArea);
        for (Vec2d& c : q) c = scale * Rotate(c - qCenter, angle) + pCenter;
        // Scale about the centroid until a corner touches the atlas triangle.
        double fit = std::numeric_limits<double>::infinity();
        for (const Vec2d& c : q) fit = std::min(fit, RayExit(pCenter, c - pCenter, p));
        if (std::isfinite(fit)) {
            for (Vec2d& c : q) c = pCenter + fit * (c - pCenter);
            scale *= fit;
        }
        // No larger than the region's median scale, so every triangle of a region has about the same texel density.
        const auto target = targetRatio.find(faceRegion[f]);
        if (target != targetRatio.end() && scale > target->second) {
            const double shrink = target->second / scale;
            for (Vec2d& c : q) c = pCenter + shrink * (c - pCenter);
        }
        uv[f] = q;
    }
}

// Step 2 for one region: similarity motions of its triangles (translation, rotation, log scale) that bring
// corresponding corners together and make areas proportional to 3D areas.
int RelaxRegion(const std::vector<uint32_t>& faces, const std::vector<CornerPair>& pairs, const std::vector<double>& area3d,
                const IsometricUvParams& params, std::vector<Triangle2>& uv)
{
    if (faces.size() < 2) return 0;
    // The fixed triangle: UV centroid closest to the region's area-weighted UV centroid.
    Vec2d center = Vec2d::Zero();
    double area = 0.0;
    for (uint32_t f : faces) {
        const double a = std::abs(SignedArea(uv[f]));
        center += a * Centroid(uv[f]);
        area += a;
    }
    center = (area > 0.0) ? Vec2d(center / area) : Centroid(uv[faces.front()]);
    uint32_t fixedFace = faces.front();
    double bestDistance = std::numeric_limits<double>::infinity();
    for (uint32_t f : faces) {
        const double d = (Centroid(uv[f]) - center).squaredNorm();
        if (d < bestDistance) {
            bestDistance = d;
            fixedFace = f;
        }
    }
    // Log UV/3D scale of a triangle as it is now (its area is that scale squared times its 3D area).
    const auto logScale = [&](uint32_t f) {
        const double a = std::abs(SignedArea(uv[f]));
        return (a > 0.0 && area3d[f] > 0.0) ? 0.5 * std::log(a / area3d[f]) : 0.0;
    };
    // The authalic target: the region's average log UV/3D scale, held fixed by the solve (a constraint).
    double targetLogScale = 0.0;
    for (uint32_t f : faces) targetLogScale += logScale(f);
    targetLogScale /= static_cast<double>(faces.size());
    const bool authalic = params.areaWeight > 0.0;
    // Unknowns of every triangle, about its own centroid: translation x, y, rotation, log scale (only with the
    // authalic term). The fixed triangle's translation and rotation are not solved for (its centroid and
    // orientation stay); its scale is, like every other triangle's.
    const int perFace = authalic ? 4 : 3;
    std::map<uint32_t, int> unknown;
    for (uint32_t f : faces) {
        if (authalic || f != fixedFace) unknown.emplace(f, static_cast<int>(unknown.size()));
    }
    const int n = perFace * static_cast<int>(unknown.size());
    std::map<uint32_t, Triangle2> rest;  // corners relative to the centroid, at rotation 0 and scale 1
    std::map<uint32_t, Vec2d> origin;
    std::map<uint32_t, double> restLogScale, targetSize;
    for (uint32_t f : faces) {
        const Vec2d c = Centroid(uv[f]);
        origin[f] = c;
        rest[f] = {uv[f][0] - c, uv[f][1] - c, uv[f][2] - c};
        restLogScale[f] = logScale(f);
        targetSize[f] = std::sqrt(std::max(area3d[f], 0.0)) * std::exp(targetLogScale);
    }
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
    const auto scaleOf = [&](int i) { return authalic ? std::exp(x[i + 3]) : 1.0; };
    const auto corner = [&](uint32_t f, int k) -> Vec2d {
        const auto it = unknown.find(f);
        if (it == unknown.end()) return origin[f] + rest[f][k];
        const int i = perFace * it->second;
        return origin[f] + Vec2d(x[i], x[i + 1]) + scaleOf(i) * Rotate(rest[f][k], x[i + 2]);
    };

    const size_t rows = 2 * pairs.size() + (authalic ? unknown.size() : 0);
    // Residuals (corner gaps, then the authalic terms) at the current x, and their Jacobian when asked for.
    const auto assemble = [&](Eigen::VectorXd& r, std::vector<Eigen::Triplet<double>>* entries) {
        r = Eigen::VectorXd::Zero(static_cast<int>(rows));
        for (size_t p = 0; p < pairs.size(); p++) {
            const CornerPair& cp = pairs[p];
            const Vec2d gap = corner(cp.faceA, cp.cornerA) - corner(cp.faceB, cp.cornerB);
            const int row = static_cast<int>(2 * p);
            r[row] = gap.x();
            r[row + 1] = gap.y();
            if (!entries) continue;
            for (const auto& [f, k, sign] : {std::tuple(cp.faceA, cp.cornerA, 1.0), std::tuple(cp.faceB, cp.cornerB, -1.0)}) {
                const auto it = unknown.find(f);
                if (it == unknown.end()) continue;
                const int i = perFace * it->second;
                const Vec2d turned = scaleOf(i) * Rotate(rest[f][k], x[i + 2]);  // d corner / d log scale
                const Vec2d dRotation = Perp(turned);
                entries->emplace_back(row, i, sign);
                entries->emplace_back(row + 1, i + 1, sign);
                entries->emplace_back(row, i + 2, sign * dRotation.x());
                entries->emplace_back(row + 1, i + 2, sign * dRotation.y());
                if (authalic) {
                    entries->emplace_back(row, i + 3, sign * turned.x());
                    entries->emplace_back(row + 1, i + 3, sign * turned.y());
                }
            }
        }
        if (authalic) {
            // areaWeight * target size * (log UV/3D scale - the region's average), every triangle: linear in the log scale. The
            // target size (the triangle's 3D size at the average scale) does not shrink with the triangle, so
            // collapsing a triangle to close its gaps always costs.
            int row = static_cast<int>(2 * pairs.size());
            for (const auto& [f, index] : unknown) {
                const int i = perFace * index;
                const double w = params.areaWeight * targetSize[f];
                r[row] = w * (restLogScale[f] + x[i + 3] - targetLogScale);
                if (entries) entries->emplace_back(row, i + 3, w);
                row++;
            }
        }
    };

    // The average log scale stays the target: the log scales of the region's triangles sum to 0. The
    // constraint is eliminated: the last triangle's log scale is minus the sum of the others' (x = T y), so the
    // reduced normal equations stay symmetric positive definite for conjugate gradients. T also drops the fixed
    // triangle's translation and rotation.
    int dependent = -1;  // column of the eliminated log scale
    if (authalic && unknown.size() > 1) dependent = perFace * static_cast<int>(unknown.size() - 1) + 3;
    const auto unknownIt = unknown.find(fixedFace);
    const int fixedBase = (unknownIt != unknown.end()) ? perFace * unknownIt->second : -1;
    const auto solved = [&](int col) {
        if (col == dependent) return false;
        if (fixedBase >= 0 && col >= fixedBase && col < fixedBase + 3) return false;  // fixed triangle's translation, rotation
        if (authalic && unknown.size() == 1 && col % perFace == 3) return false;     // a lone triangle keeps its scale
        return true;
    };
    int m = 0;
    for (int col = 0; col < n; col++) m += solved(col);
    Eigen::SparseMatrix<double> T(n, m);
    {
        std::vector<Eigen::Triplet<double>> t;
        for (int col = 0, j = 0; col < n; col++) {
            if (!solved(col)) continue;
            t.emplace_back(col, j, 1.0);
            if (dependent >= 0 && col % perFace == 3) t.emplace_back(dependent, j, -1.0);
            j++;
        }
        T.setFromTriplets(t.begin(), t.end());
    }

    // Gauss-Newton: each iteration linearizes the rotations and scales and solves the sparse normal equations
    // with conjugate gradients.
    Eigen::VectorXd r;
    int iteration = 0;
    for (; iteration < params.iterations; iteration++) {
        std::vector<Eigen::Triplet<double>> entries;
        entries.reserve(pairs.size() * 2 * 8 + unknown.size());
        assemble(r, &entries);
        Eigen::SparseMatrix<double> J(static_cast<int>(rows), n);
        J.setFromTriplets(entries.begin(), entries.end());
        const Eigen::SparseMatrix<double> reduced = J * T;
        Eigen::SparseMatrix<double> normal = reduced.transpose() * reduced;
        const Eigen::VectorXd rhs = -(reduced.transpose() * r);
        // A triangle with no correspondences leaves its unknowns free: a tiny damping keeps the system definite.
        double meanDiagonal = 0.0;
        for (int i = 0; i < m; i++) meanDiagonal += normal.coeff(i, i);
        meanDiagonal = (m > 0) ? meanDiagonal / m : 1.0;
        for (int i = 0; i < m; i++) normal.coeffRef(i, i) += 1e-9 * meanDiagonal + 1e-30;
        Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
        cg.setMaxIterations(params.cgIterations);
        cg.setTolerance(1e-14);
        cg.compute(normal);
        const Eigen::VectorXd dy = cg.solve(rhs);
        if (!dy.allFinite()) break;
        const Eigen::VectorXd dx = T * dy;
        x += dx;
        if (dx.lpNorm<Eigen::Infinity>() < params.tolerance) {
            iteration++;
            break;
        }
    }
    for (uint32_t f : faces) {
        for (int k = 0; k < 3; k++) uv[f][k] = corner(f, k);
    }
    return iteration;
}
}  // namespace

IsometricUvStats MakeIsometricUvs(const Mesh& mesh, const std::vector<int>& faceRegion, std::vector<Vector2d>& out, const IsometricUvParams& params)
{
    const size_t F = mesh.triIndexes.size();
    Expects(mesh.texcoords.size() == 3 * F, "MakeIsometricUvs: {} texture coordinates for {} triangles", mesh.texcoords.size(), F);
    Expects(faceRegion.size() == F, "MakeIsometricUvs: {} face regions for {} triangles", faceRegion.size(), F);
    std::vector<Triangle2> uv(F);
    for (size_t f = 0; f < F; f++) {
        for (int c = 0; c < 3; c++) uv[f][c] = mesh.texcoords[3 * f + c].cast<double>();
    }
    InscribeIsometricTriangles(mesh, faceRegion, uv);

    std::vector<double> area3d(F);
    for (size_t f = 0; f < F; f++) {
        const Vector3u& t = mesh.triIndexes[f];
        area3d[f] = SignedArea(FlattenTriangle(mesh.vertexLocations[t[0]], mesh.vertexLocations[t[1]], mesh.vertexLocations[t[2]]));
    }
    const std::vector<CornerPair> pairs = SharedCorners(mesh, faceRegion);
    IsometricUvStats stats;
    stats.meanGapBefore = MeanGap(uv, pairs);
    std::map<int, std::vector<uint32_t>> regionFaces;
    std::map<int, std::vector<CornerPair>> regionPairs;
    for (uint32_t f = 0; f < F; f++) regionFaces[faceRegion[f]].push_back(f);
    for (const CornerPair& p : pairs) regionPairs[faceRegion[p.faceA]].push_back(p);
    for (auto& [region, faces] : regionFaces) {
        stats.iterations = std::max(stats.iterations, RelaxRegion(faces, regionPairs[region], area3d, params, uv));
    }
    stats.meanGapAfter = MeanGap(uv, pairs);

    out.resize(3 * F);
    for (size_t f = 0; f < F; f++) {
        for (int c = 0; c < 3; c++) out[3 * f + c] = uv[f][c];
    }
    return stats;
}

}  // namespace mopt
