#include <flattening/IsometricUvs.hpp>

#include <core/Check.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <tuple>
#include <unordered_map>
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
double Cross(const Vec2d& a, const Vec2d& b) { return a.x() * b.y() - a.y() * b.x(); }

// The 3D triangle laid flat: corner 0 at the origin, corner 1 on +x, corner 2 above (counter-clockwise).
Triangle2 FlattenTriangle(const Vector3f& p0, const Vector3f& p1, const Vector3f& p2)
{
    const Eigen::Vector3d d1 = (p1 - p0).cast<double>(), d2 = (p2 - p0).cast<double>();
    const Eigen::Vector3d x = d1.normalized();
    const Eigen::Vector3d y = x.cross(d2).cross(x).normalized();
    return {Vec2d::Zero(), Vec2d(d1.dot(x), 0.0), Vec2d(d2.dot(x), d2.dot(y))};
}

// A turn, uniform scale and offset: p -> scale * R(angle) p + offset.
struct FitTransform {
    double angle = 0.0, scale = 1.0;
    Vec2d offset = Vec2d::Zero();
};

// Convex hull (counter-clockwise, Andrew's monotone chain).
std::vector<Vec2d> ConvexHull(std::vector<Vec2d> p)
{
    std::sort(p.begin(), p.end(), [](const Vec2d& a, const Vec2d& b) { return a.x() < b.x() || (a.x() == b.x() && a.y() < b.y()); });
    p.erase(std::unique(p.begin(), p.end()), p.end());
    if (p.size() < 3) return p;
    const auto cross = [](const Vec2d& o, const Vec2d& a, const Vec2d& b) { return Cross(a - o, b - o); };
    std::vector<Vec2d> hull(2 * p.size());
    size_t k = 0;
    for (size_t i = 0; i < p.size(); i++) {
        while (k >= 2 && cross(hull[k - 2], hull[k - 1], p[i]) <= 0.0) k--;
        hull[k++] = p[i];
    }
    for (size_t i = p.size() - 1, lower = k + 1; i-- > 0;) {
        while (k >= lower && cross(hull[k - 2], hull[k - 1], p[i]) <= 0.0) k--;
        hull[k++] = p[i];
    }
    hull.resize(k - 1);
    return hull;
}

// The turn, uniform scale and offset that fit `points` as large as possible inside the box [lo, hi], centered:
// the scale at a turn is min(box size / turned points' box size) per axis, and the best of eight turns
// (k * 22.5 degrees) is kept.
FitTransform FitIntoBox(const std::vector<Vec2d>& points, const Vec2d& lo, const Vec2d& hi)
{
    FitTransform t;
    if (points.empty()) return t;
    const std::vector<Vec2d> hull = ConvexHull(points);
    const Vec2d size = hi - lo;
    const auto turnedBox = [&](double angle, Vec2d& boxLo, Vec2d& boxHi) {
        boxLo = Vec2d::Constant(std::numeric_limits<double>::infinity());
        boxHi = -boxLo;
        for (const Vec2d& p : hull) {
            const Vec2d q = Rotate(p, angle);
            boxLo = boxLo.cwiseMin(q);
            boxHi = boxHi.cwiseMax(q);
        }
    };
    const auto scaleAt = [&](double angle) {
        Vec2d a, b;
        turnedBox(angle, a, b);
        const Vec2d have = b - a;
        double scale = std::numeric_limits<double>::infinity();
        for (int axis = 0; axis < 2; axis++) {
            if (have[axis] > 0.0) scale = std::min(scale, size[axis] / have[axis]);
        }
        return (std::isfinite(scale) && scale > 0.0) ? scale : 1.0;
    };
    // Eight turns, k * 22.5 degrees (a box fit repeats every half turn, so these are eight distinct fits).
    constexpr int kTurns = 8;
    const double step = std::acos(-1.0) / kTurns;
    double best = 0.0, bestScale = scaleAt(0.0);
    for (int i = 1; i < kTurns; i++) {
        const double angle = i * step, scale = scaleAt(angle);
        if (scale > bestScale) {
            bestScale = scale;
            best = angle;
        }
    }
    t.angle = best;
    t.scale = scaleAt(best);
    Vec2d boxLo, boxHi;
    turnedBox(best, boxLo, boxHi);
    t.offset = 0.5 * (lo + hi) - t.scale * 0.5 * (boxLo + boxHi);
    return t;
}

// A spring of the simulation: corner cornerA of triangle faceA and corner cornerB of triangle faceB were at the
// same place in the atlas (the same vertex of one chart), weighted 1 / (N - 1) for a place N corners share, so
// every corner's springs weigh 1 in all.
struct CornerLink {
    uint32_t faceA, faceB;
    int cornerA, cornerB;
    double weight;
};

// Every pair of corners of one region that share an atlas location.
std::vector<CornerLink> SharedAtlasCorners(const Mesh& mesh, const std::vector<int>& faceRegion)
{
    std::map<std::tuple<int, float, float>, std::vector<std::pair<uint32_t, int>>> places;  // (region, atlas u, v) -> corners
    for (uint32_t f = 0; f < mesh.triIndexes.size(); f++) {
        for (int c = 0; c < 3; c++) {
            const auto& t = mesh.texcoords[3 * f + c];
            places[{faceRegion[f], static_cast<float>(t.x()), static_cast<float>(t.y())}].push_back({f, c});
        }
    }
    std::vector<CornerLink> links;
    for (const auto& [place, corners] : places) {
        if (corners.size() < 2) continue;
        const double weight = 1.0 / static_cast<double>(corners.size() - 1);
        for (size_t i = 0; i < corners.size(); i++) {
            for (size_t j = i + 1; j < corners.size(); j++) {
                links.push_back({corners[i].first, corners[j].first, corners[i].second, corners[j].second, weight});
            }
        }
    }
    return links;
}

// Step 1: every triangle its 3D triangle laid flat at its chart's scale, turned like and centered on its atlas
// triangle.
void PlaceIsometricTriangles(const Mesh& mesh, const std::vector<int>& faceRegion, std::vector<Triangle2>& uv)
{
    const size_t F = mesh.triIndexes.size();
    std::vector<Triangle2> flat(F);
    std::map<int, std::pair<double, double>> areas;  // region -> (atlas area, 3D area)
    for (size_t f = 0; f < F; f++) {
        const Vector3u& t = mesh.triIndexes[f];
        flat[f] = FlattenTriangle(mesh.vertexLocations[t[0]], mesh.vertexLocations[t[1]], mesh.vertexLocations[t[2]]);
        auto& [atlas, xyz] = areas[faceRegion[f]];
        atlas += std::abs(SignedArea(uv[f]));
        xyz += SignedArea(flat[f]);
    }
    for (size_t f = 0; f < F; f++) {
        const auto& [atlas, xyz] = areas[faceRegion[f]];
        const Triangle2& p = uv[f];
        const Triangle2& q = flat[f];
        if (!(atlas > 0.0) || !(xyz > 0.0) || !(SignedArea(q) > 0.0)) continue;  // degenerate: left as it is
        const double scale = std::sqrt(atlas / xyz);
        const Vec2d pCenter = Centroid(p), qCenter = Centroid(q);
        const Vec2d from = q[0] - qCenter, to = p[0] - pCenter;
        const double angle = std::atan2(Cross(from, to), from.dot(to));
        for (int c = 0; c < 3; c++) uv[f][c] = scale * Rotate(q[c] - qCenter, angle) + pCenter;
    }
}

// Two overlapping triangles f and g of a region: n, the axis along which they overlap least, pointing from f to
// g; depth, how far they overlap along it; point, the centroid of their overlap (where the impulse acts).
struct Contact {
    uint32_t f, g;
    Vec2d n, point;
    double depth;
};

// ---- Triangle-triangle intersection in 2D, specialized from Geometric Tools (David Eberly,
// GTE/Mathematics/IntrTriangle2Triangle2.h and IntrConvexPolygonHyperplane.h, Boost Software License 1.0) to
// double-precision Eigen vectors and fixed-size polygons. Both expect counter-clockwise triangles.

// GTE's Perp: (v.y, -v.x), the outward normal of a counter-clockwise edge v.
Vec2d GtePerp(const Vec2d& v) { return Vec2d(v.y(), -v.x()); }

// +1 if every vertex of t projects to >= 0 on P + s D, -1 if every one to <= 0, 0 if D's line splits t into two
// triangles of positive area.
int WhichSide(const Triangle2& t, const Vec2d& P, const Vec2d& D)
{
    int positive = 0, negative = 0;
    for (int i = 0; i < 3; i++) {
        const double s = D.dot(t[i] - P);
        if (s > 0.0)
            positive++;
        else if (s < 0.0)
            negative++;
        if (positive && negative) return 0;
    }
    return positive > 0 ? +1 : -1;
}

// Whether an edge of t0 separates t1 from it (t1 strictly on the edge's outer side; touching counts as separated).
bool Separated(const Triangle2& t0, const Triangle2& t1)
{
    for (int i0 = 2, i1 = 0; i1 < 3; i0 = i1++) {
        if (WhichSide(t1, t0[i0], GtePerp(t0[i1] - t0[i0])) > 0) return true;
    }
    return false;
}

// A convex polygon of at most 9 vertices (a triangle clipped by three lines has at most 6).
struct SmallPolygon {
    std::array<Vec2d, 9> v;
    int n = 0;
    void push(const Vec2d& p) { v[n++] = p; }
};

// The part of the convex polygon `in` on the positive side of the line N . x = c (GTE's convex polygon /
// hyperplane split, keeping its positive polygon); empty when no vertex is strictly on that side.
SmallPolygon PositivePart(const SmallPolygon& in, const Vec2d& N, double c)
{
    SmallPolygon out;
    const int n = in.n;
    if (n < 3) return out;
    std::array<double, 9> height{};
    int numPositive = 0, numNegative = 0, maxPos = -1, maxNeg = -1;
    double maxPosHeight = -std::numeric_limits<double>::max(), maxNegHeight = std::numeric_limits<double>::max();
    for (int i = 0; i < n; i++) {
        height[i] = N.dot(in.v[i]) - c;
        if (height[i] > 0.0) {
            numPositive++;
            if (height[i] > maxPosHeight) {
                maxPosHeight = height[i];
                maxPos = i;
            }
        } else if (height[i] < 0.0) {
            numNegative++;
            if (height[i] < maxNegHeight) {
                maxNegHeight = height[i];
                maxNeg = i;
            }
        }
    }
    if (numPositive == 0) return out;
    if (numNegative == 0) return in;

    // Split from the side whose farthest vertex is farther from the line (GTE's swap, for robustness): the
    // largest run of vertices with height >= 0 around `start`, and the crossings at its two ends.
    const bool swapped = maxPosHeight < -maxNegHeight;
    if (swapped) {
        for (int i = 0; i < n; i++) height[i] = -height[i];
    }
    const int start = swapped ? maxNeg : maxPos;
    int end0 = start, end0prev = start;
    for (int i = 0; i < n; i++) {
        end0prev = (end0 + n - 1) % n;
        if (height[end0prev] >= 0.0)
            end0 = end0prev;
        else
            break;
    }
    int end1 = start, end1next = start;
    for (int i = 0; i < n; i++) {
        end1next = (end1 + 1) % n;
        if (height[end1next] >= 0.0)
            end1 = end1next;
        else
            break;
    }
    const auto crossing = [&](int inside, int outside) {
        const double t = -height[outside] / (height[inside] - height[outside]);
        return Vec2d((1.0 - t) * in.v[outside] + t * in.v[inside]);
    };
    if (!swapped) {
        // Positive polygon: [crossing at end0] end0 .. end1 [crossing at end1].
        if (height[end0] > 0.0) out.push(crossing(end0, end0prev));
        for (int i = end0;; i = (i + 1) % n) {
            out.push(in.v[i]);
            if (i == end1) break;
        }
        if (height[end1] > 0.0) out.push(crossing(end1, end1next));
    } else {
        // The split's negative polygon, which is the positive one of the unswapped heights:
        // (crossing at end1 | end1) end1next .. end0prev (crossing at end0 | end0).
        out.push(height[end1] > 0.0 ? crossing(end1, end1next) : in.v[end1]);
        for (int i = end1next; i != end0; i = (i + 1) % n) out.push(in.v[i]);
        out.push(height[end0] > 0.0 ? crossing(end0, end0prev) : in.v[end0]);
    }
    return out;
}

// The intersection of triangles t0 and t1: t1 clipped by the three edges of t0 (GTE's find-intersection query).
SmallPolygon Intersection(const Triangle2& t0, const Triangle2& t1)
{
    SmallPolygon polygon;
    for (const Vec2d& p : t1) polygon.push(p);
    for (int i1 = 2, i0 = 0; i0 < 3; i1 = i0++) {
        const Vec2d N = GtePerp(t0[i1] - t0[i0]);  // points into t0
        polygon = PositivePart(polygon, N, N.dot(t0[i0]));
        if (polygon.n == 0) break;
    }
    return polygon;
}
// ---- end of the Geometric Tools port

Triangle2 CounterClockwise(Triangle2 t)
{
    if (SignedArea(t) < 0.0) std::swap(t[1], t[2]);
    return t;
}

// Every pair of a region's triangles (now: their current UVs, in the order of faces) that overlap at all.
std::vector<Contact> FindContacts(const std::vector<uint32_t>& faces, const std::vector<Triangle2>& now)
{
    std::vector<std::pair<Vec2d, Vec2d>> boxes(faces.size());
    for (size_t i = 0; i < faces.size(); i++) {
        const Triangle2& t = now[i];
        boxes[i] = {t[0].cwiseMin(t[1]).cwiseMin(t[2]), t[0].cwiseMax(t[1]).cwiseMax(t[2])};
    }
    std::vector<Contact> contacts;
    for (size_t i = 0; i < faces.size(); i++) {
        for (size_t j = i + 1; j < faces.size(); j++) {
            if ((boxes[i].second.array() < boxes[j].first.array()).any() || (boxes[j].second.array() < boxes[i].first.array()).any()) continue;
            const TriangleOverlap o = OverlapOf(now[i], now[j]);
            if (!(o.depth > 0.0)) continue;
            contacts.push_back({faces[i], faces[j], o.axis, o.center, o.depth});
        }
    }
    return contacts;
}

// Step 2 for one region (see IsometricUvs.hpp): the triangles as rigid bodies, held together by the springs
// and, with enforceNoOverlaps, repelling each other where they overlap.
void SimulateRegion(const std::vector<uint32_t>& faces, const std::vector<CornerLink>& links, const IsometricUvParams& params,
                    std::vector<Triangle2>& uv)
{
    const size_t count = faces.size();
    if (count < 2) return;
    struct Body {
        Vec2d center, velocity = Vec2d::Zero();
        double angle = 0.0, spin = 0.0, mass = 0.0, inverseMass = 0.0, inverseInertia = 0.0;
        Triangle2 offsets;  // corners about the centroid at angle 0
        // Accumulated in the compute phase, applied in the apply phase.
        Vec2d force = Vec2d::Zero(), impulse = Vec2d::Zero();
        double torque = 0.0, angularImpulse = 0.0;
    };
    std::vector<Body> bodies(count);
    std::unordered_map<uint32_t, size_t> index;
    double meanArea = 0.0;
    for (size_t i = 0; i < count; i++) meanArea += std::abs(SignedArea(uv[faces[i]]));
    meanArea = std::max(meanArea / static_cast<double>(count), 1e-300);
    for (size_t i = 0; i < count; i++) {
        const Triangle2& t = uv[faces[i]];
        Body& b = bodies[i];
        b.center = Centroid(t);
        double moment = 0.0;
        for (int k = 0; k < 3; k++) {
            b.offsets[k] = t[k] - b.center;
            moment += b.offsets[k].squaredNorm();
        }
        b.mass = std::abs(SignedArea(t)) / meanArea;  // in units of the region's mean
        b.inverseMass = (b.mass > 0.0) ? 1.0 / b.mass : 0.0;
        b.inverseInertia = (b.mass > 0.0 && moment > 0.0) ? 12.0 / (b.mass * moment) : 0.0;
        index[faces[i]] = i;
    }
    const auto pointVelocity = [](const Body& b, const Vec2d& r) { return Vec2d(b.velocity + b.spin * Perp(r)); };
    const double dt = 1.0;

    std::vector<Triangle2> now(count);
    for (int step = 0; step < params.relaxSteps + params.repelSteps; step++) {
        const bool relaxing = step < params.relaxSteps;
        for (size_t i = 0; i < count; i++) now[i] = uv[faces[i]];
        std::vector<Contact> contacts;
        // Springs alone while relaxing: repelling the heavily overlapping triangles of the placement would scatter
        // them before the springs have arranged them.
        if (params.enforceNoOverlaps && !relaxing) contacts = FindContacts(faces, now);
        if (!relaxing) {
            double deepest = 0.0;
            for (const Contact& h : contacts) deepest = std::max(deepest, h.depth);
            if (deepest <= params.overlapTolerance) break;
        }
        // Compute, from the state at the start of the step.
        for (Body& b : bodies) {
            b.force = b.impulse = Vec2d::Zero();
            b.torque = b.angularImpulse = 0.0;
        }
        const double stiffness = relaxing ? params.springStiffness : params.repelSpringStiffness;
        for (const CornerLink& l : links) {
            Body &A = bodies[index.at(l.faceA)], &B = bodies[index.at(l.faceB)];
            const Vec2d pa = uv[l.faceA][l.cornerA], pb = uv[l.faceB][l.cornerB];
            // On corner a, toward b; scaled by the lighter body's mass so a small triangle is not flung about.
            const Vec2d F = stiffness * l.weight * std::min(A.mass, B.mass) * (pb - pa);
            A.force += F;
            A.torque += Cross(pa - A.center, F);
            B.force -= F;
            B.torque -= Cross(pb - B.center, F);
        }
        for (const Contact& h : contacts) {
            Body &A = bodies[index.at(h.f)], &B = bodies[index.at(h.g)];
            const Vec2d rA = h.point - A.center, rB = h.point - B.center;
            const double vn = (pointVelocity(B, rB) - pointVelocity(A, rA)).dot(h.n);
            const double rAn = Cross(rA, h.n), rBn = Cross(rB, h.n);
            const double k = A.inverseMass + B.inverseMass + rAn * rAn * A.inverseInertia + rBn * rBn * B.inverseInertia;
            if (k <= 0.0) continue;
            const double j = std::max((-(1.0 + params.restitution) * vn + params.baumgarte * h.depth / dt) / k, 0.0);
            const Vec2d J = j * h.n;
            A.impulse -= J;
            A.angularImpulse -= Cross(rA, J);
            B.impulse += J;
            B.angularImpulse += Cross(rB, J);
        }
        // Apply, damp, integrate.
        const double keep = 1.0 - params.damping;
        for (Body& b : bodies) {
            b.velocity = keep * (b.velocity + b.inverseMass * (b.force * dt + b.impulse));
            b.spin = keep * (b.spin + b.inverseInertia * (b.torque * dt + b.angularImpulse));
            b.center += b.velocity * dt;
            b.angle += b.spin * dt;
        }
        for (size_t i = 0; i < count; i++) {
            for (int k = 0; k < 3; k++) uv[faces[i]][k] = bodies[i].center + Rotate(bodies[i].offsets[k], bodies[i].angle);
        }
    }
}
}  // namespace

TriangleOverlap OverlapOf(const std::array<Vector2d, 3>& p, const std::array<Vector2d, 3>& q)
{
    // Geometric Tools' test-intersection query decides whether the interiors meet (touching is not overlapping),
    // and its find-intersection query gives the overlap polygon.
    const Triangle2 a = CounterClockwise(p), b = CounterClockwise(q);
    if (Separated(a, b) || Separated(b, a)) return {};
    const SmallPolygon overlap = Intersection(a, b);
    if (overlap.n < 3) return {};
    TriangleOverlap o;
    // Where to push: the centroid of the overlap polygon.
    Vec2d sum = Vec2d::Zero(), mean = Vec2d::Zero();
    double area = 0.0;
    for (int i = 0; i < overlap.n; i++) {
        const Vec2d &u = overlap.v[i], &v = overlap.v[(i + 1) % overlap.n];
        const double c = Cross(u, v);
        area += c;
        sum += c * (u + v);
        mean += u;
    }
    o.center = (std::abs(area) > 1e-300) ? Vec2d(sum / (3.0 * area)) : Vec2d(mean / overlap.n);
    // How deep and which way: the separating axis (of the six edge normals) along which the projections overlap
    // least.
    o.depth = std::numeric_limits<double>::infinity();
    for (const Triangle2* t : {&a, &b}) {
        for (int e = 0; e < 3; e++) {
            const Vec2d edge = (*t)[(e + 1) % 3] - (*t)[e];
            const double length = edge.norm();
            if (length <= 0.0) continue;
            const Vec2d axis = Perp(edge) / length;
            double pMin = std::numeric_limits<double>::infinity(), pMax = -pMin, qMin = pMin, qMax = -pMin;
            for (const Vec2d& v : a) {
                pMin = std::min(pMin, axis.dot(v));
                pMax = std::max(pMax, axis.dot(v));
            }
            for (const Vec2d& v : b) {
                qMin = std::min(qMin, axis.dot(v));
                qMax = std::max(qMax, axis.dot(v));
            }
            // q ahead of p along the axis: they separate along +axis; behind it: along -axis.
            if (pMax - qMin < o.depth) {
                o.depth = pMax - qMin;
                o.axis = axis;
            }
            if (qMax - pMin < o.depth) {
                o.depth = qMax - pMin;
                o.axis = -axis;
            }
        }
    }
    if (!std::isfinite(o.depth) || !(o.depth > 0.0)) return {};
    return o;
}

void MakeIsometricUvs(const Mesh& mesh, const std::vector<int>& faceRegion, std::vector<Vector2d>& out, const IsometricUvParams& params)
{
    const size_t F = mesh.triIndexes.size();
    Expects(mesh.texcoords.size() == 3 * F, "MakeIsometricUvs: {} texture coordinates for {} triangles", mesh.texcoords.size(), F);
    Expects(faceRegion.size() == F, "MakeIsometricUvs: {} face regions for {} triangles", faceRegion.size(), F);
    std::vector<Triangle2> uv(F);
    for (size_t f = 0; f < F; f++) {
        for (int c = 0; c < 3; c++) uv[f][c] = mesh.texcoords[3 * f + c].cast<double>();
    }
    PlaceIsometricTriangles(mesh, faceRegion, uv);

    // Step 2, every region on its own thread: each owns its triangles' UVs.
    std::map<int, std::vector<uint32_t>> regionFaces;
    for (uint32_t f = 0; f < F; f++) regionFaces[faceRegion[f]].push_back(f);
    std::map<int, std::vector<CornerLink>> regionLinks;
    for (const CornerLink& l : SharedAtlasCorners(mesh, faceRegion)) regionLinks[faceRegion[l.faceA]].push_back(l);
    std::vector<std::pair<const std::vector<uint32_t>*, const std::vector<CornerLink>*>> regions;
    for (const auto& [region, faces] : regionFaces) regions.emplace_back(&faces, &regionLinks[region]);
    const long regionCount = static_cast<long>(regions.size());
#pragma omp parallel for schedule(dynamic)
    for (long i = 0; i < regionCount; i++) SimulateRegion(*regions[i].first, *regions[i].second, params, uv);

    if (params.fitToAtlas) {
        // Step 3: each region turned, scaled uniformly and moved so it fits tightly inside its atlas chart's
        // box: the turn is the one that allows the largest scale, and the region is centered in the box.
        std::map<int, std::pair<Vec2d, Vec2d>> atlas;  // region -> box (lo, hi)
        std::map<int, std::vector<Vec2d>> points;
        for (uint32_t f = 0; f < F; f++) {
            auto [it, inserted] = atlas.try_emplace(faceRegion[f], Vec2d::Constant(std::numeric_limits<double>::infinity()),
                                                    Vec2d::Constant(-std::numeric_limits<double>::infinity()));
            for (int c = 0; c < 3; c++) {
                const Vec2d a = mesh.texcoords[3 * f + c].cast<double>();
                it->second.first = it->second.first.cwiseMin(a);
                it->second.second = it->second.second.cwiseMax(a);
                points[faceRegion[f]].push_back(uv[f][c]);
            }
        }
        std::map<int, FitTransform> fit;
        for (const auto& [region, box] : atlas) fit[region] = FitIntoBox(points[region], box.first, box.second);
        for (uint32_t f = 0; f < F; f++) {
            const FitTransform& t = fit[faceRegion[f]];
            for (Vec2d& c : uv[f]) c = t.scale * Rotate(c, t.angle) + t.offset;
        }
    }

    out.resize(3 * F);
    for (size_t f = 0; f < F; f++) {
        for (int c = 0; c < 3; c++) out[3 * f + c] = uv[f][c];
    }
}

}  // namespace mopt
