#include <flattening/ChartBinPack.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace mopt
{
namespace
{
// An axis-aligned rectangle: corner (x, y), size w x h.
struct PackRect {
    double x = 0, y = 0, w = 0, h = 0;
};

// Convex hull of 2D points (monotone chain), counter-clockwise.
std::vector<Vector2d> ConvexHull(std::vector<Vector2d> pts)
{
    std::sort(pts.begin(), pts.end(),
              [](const Vector2d& a, const Vector2d& b) { return a.x() < b.x() || (a.x() == b.x() && a.y() < b.y()); });
    pts.erase(std::unique(pts.begin(), pts.end(),
                          [](const Vector2d& a, const Vector2d& b) { return a.x() == b.x() && a.y() == b.y(); }),
              pts.end());
    if (pts.size() < 3) return pts;
    const auto cross = [](const Vector2d& o, const Vector2d& a, const Vector2d& b) {
        return (a.x() - o.x()) * (b.y() - o.y()) - (a.y() - o.y()) * (b.x() - o.x());
    };
    std::vector<Vector2d> hull(2 * pts.size());
    size_t k = 0;
    for (size_t i = 0; i < pts.size(); i++) {
        while (k >= 2 && cross(hull[k - 2], hull[k - 1], pts[i]) <= 0) k--;
        hull[k++] = pts[i];
    }
    for (size_t i = pts.size() - 1, t = k + 1; i > 0; i--) {
        while (k >= t && cross(hull[k - 2], hull[k - 1], pts[i - 1]) <= 0) k--;
        hull[k++] = pts[i - 1];
    }
    hull.resize(k - 1);
    return hull;
}

// Angle that turns the hull's minimum-area enclosing rectangle axis-aligned. The optimal rectangle has a
// side along a hull edge, so each edge direction is tried.
double MinAreaRectAngle(const std::vector<Vector2d>& hull)
{
    double bestArea = std::numeric_limits<double>::max();
    double bestAngle = 0.0;
    const size_t n = hull.size();
    for (size_t i = 0; i < n; i++) {
        const Vector2d e = hull[(i + 1) % n] - hull[i];
        const double len = e.norm();
        if (len <= 0.0) continue;
        const double c = e.x() / len, s = e.y() / len;
        double minU = std::numeric_limits<double>::max(), maxU = -minU, minV = minU, maxV = -minU;
        for (const Vector2d& p : hull) {
            const double u = p.x() * c + p.y() * s;
            const double v = -p.x() * s + p.y() * c;
            minU = std::min(minU, u);
            maxU = std::max(maxU, u);
            minV = std::min(minV, v);
            maxV = std::max(maxV, v);
        }
        const double area = (maxU - minU) * (maxV - minV);
        if (area < bestArea) {
            bestArea = area;
            bestAngle = std::atan2(s, c);
        }
    }
    return bestAngle;
}

// MaxRects bin packing (Jylanki), best-short-side-fit, rectangles may be turned by 90 degrees. Keeps the
// maximal free rectangles of a square bin.
class MaxRects
{
public:
    explicit MaxRects(double side) : side_(side) { free_.push_back({0, 0, side, side}); }

    // Places a w x h rectangle; false if it does not fit.
    bool insert(double w, double h, PackRect& placed, bool& turned)
    {
        double bestShort = std::numeric_limits<double>::max();
        double bestLong = bestShort;
        bool found = false;
        for (const PackRect& f : free_) {
            for (int t = 0; t < 2; t++) {
                const double rw = t ? h : w, rh = t ? w : h;
                if (rw > f.w + eps() || rh > f.h + eps()) continue;
                const double dw = f.w - rw, dh = f.h - rh;
                const double shortFit = std::min(dw, dh), longFit = std::max(dw, dh);
                if (shortFit < bestShort || (shortFit == bestShort && longFit < bestLong)) {
                    bestShort = shortFit;
                    bestLong = longFit;
                    placed = {f.x, f.y, rw, rh};
                    turned = t == 1;
                    found = true;
                }
            }
        }
        if (!found) return false;
        place(placed);
        return true;
    }

private:
    double eps() const { return 1e-12 * side_; }

    void place(const PackRect& n)
    {
        // Split every free rectangle the new one overlaps into up to four maximal pieces around it.
        std::vector<PackRect> added;
        for (size_t i = 0; i < free_.size();) {
            const PackRect f = free_[i];
            const bool overlaps = n.x < f.x + f.w - eps() && n.x + n.w > f.x + eps() && n.y < f.y + f.h - eps() &&
                                  n.y + n.h > f.y + eps();
            if (!overlaps) {
                i++;
                continue;
            }
            if (n.x > f.x + eps()) added.push_back({f.x, f.y, n.x - f.x, f.h});
            if (n.x + n.w < f.x + f.w - eps()) added.push_back({n.x + n.w, f.y, f.x + f.w - (n.x + n.w), f.h});
            if (n.y > f.y + eps()) added.push_back({f.x, f.y, f.w, n.y - f.y});
            if (n.y + n.h < f.y + f.h - eps()) added.push_back({f.x, n.y + n.h, f.w, f.y + f.h - (n.y + n.h)});
            free_[i] = free_.back();
            free_.pop_back();
        }
        // Keep only maximal free rectangles: drop new ones inside any other, and old ones inside a new one.
        const auto contains = [this](const PackRect& a, const PackRect& b) {
            return b.x >= a.x - eps() && b.y >= a.y - eps() && b.x + b.w <= a.x + a.w + eps() &&
                   b.y + b.h <= a.y + a.h + eps();
        };
        std::vector<PackRect> kept;
        for (size_t i = 0; i < added.size(); i++) {
            bool inside = false;
            for (size_t j = 0; j < added.size() && !inside; j++) {
                if (i != j && contains(added[j], added[i]) && (!contains(added[i], added[j]) || j < i)) inside = true;
            }
            for (size_t j = 0; j < free_.size() && !inside; j++) inside = contains(free_[j], added[i]);
            if (!inside) kept.push_back(added[i]);
        }
        for (size_t j = 0; j < free_.size();) {
            bool inside = false;
            for (const PackRect& k : kept) {
                if (contains(k, free_[j])) {
                    inside = true;
                    break;
                }
            }
            if (inside) {
                free_[j] = free_.back();
                free_.pop_back();
            } else {
                j++;
            }
        }
        free_.insert(free_.end(), kept.begin(), kept.end());
    }

    double side_;
    std::vector<PackRect> free_;
};
}  // namespace

double ChartBinPack::pack(std::vector<Mosaic>& mosaics) const
{
    // 1. Turn each chart so its minimum-area enclosing rectangle is axis-aligned.
    const size_t count = mosaics.size();
    std::vector<Vector2d> lo(count), hi(count);
    std::vector<char> present(count, 0);
    double meanSide = 0.0;
    size_t nonEmpty = 0;
    for (size_t i = 0; i < count; i++) {
        Mosaic& mosaic = mosaics[i];
        if (mosaic.vertexes.empty()) continue;
        std::vector<Vector2d> pts;
        pts.reserve(mosaic.vertexes.size());
        for (const MosaicVertex& mv : mosaic.vertexes) pts.push_back(mv.uv);
        const std::vector<Vector2d> hull = ConvexHull(std::move(pts));
        if (hull.size() >= 3) {
            const double angle = MinAreaRectAngle(hull);
            const double c = std::cos(angle), s = std::sin(angle);
            for (MosaicVertex& mv : mosaic.vertexes) {
                mv.uv = Vector2d(mv.uv.x() * c + mv.uv.y() * s, -mv.uv.x() * s + mv.uv.y() * c);
            }
        }
        lo[i] = Vector2d::Constant(std::numeric_limits<double>::max());
        hi[i] = Vector2d::Constant(std::numeric_limits<double>::lowest());
        for (const MosaicVertex& mv : mosaic.vertexes) {
            lo[i] = lo[i].cwiseMin(mv.uv);
            hi[i] = hi[i].cwiseMax(mv.uv);
        }
        present[i] = 1;
        meanSide += 0.5 * ((hi[i].x() - lo[i].x()) + (hi[i].y() - lo[i].y()));
        nonEmpty++;
    }
    if (nonEmpty == 0) return 1.0;
    meanSide /= nonEmpty;
    const double pad = std::max(padding_ * meanSide, 1e-12);

    // 2. MaxRects into the smallest square found: start at the total area and grow 2% per failed attempt.
    //    Larger boxes go first.
    std::vector<size_t> order;
    double area = 0.0, maxSide = 0.0;
    for (size_t i = 0; i < count; i++) {
        if (!present[i]) continue;
        order.push_back(i);
        const double w = hi[i].x() - lo[i].x() + 2 * pad, h = hi[i].y() - lo[i].y() + 2 * pad;
        area += w * h;
        maxSide = std::max(maxSide, std::max(w, h));
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        const double sa = std::max(hi[a].x() - lo[a].x(), hi[a].y() - lo[a].y());
        const double sb = std::max(hi[b].x() - lo[b].x(), hi[b].y() - lo[b].y());
        return sa > sb;
    });
    std::vector<PackRect> placed(count);
    std::vector<char> turned(count, 0);
    double side = std::max(std::sqrt(area), maxSide);
    for (;;) {
        MaxRects bins(side);
        bool fits = true;
        for (size_t i : order) {
            bool t = false;
            if (!bins.insert(hi[i].x() - lo[i].x() + 2 * pad, hi[i].y() - lo[i].y() + 2 * pad, placed[i], t)) {
                fits = false;
                break;
            }
            turned[i] = t;
        }
        if (fits) break;
        side *= 1.02;
    }

    // 3. Move each chart into its rectangle (turned by 90 degrees if the packer turned it) and scale the
    //    square to [0,1]^2.
    const double scale = 1.0 / side;
    for (size_t i : order) {
        const PackRect& r = placed[i];
        const Vector2d corner(r.x + pad, r.y + pad);
        for (MosaicVertex& mv : mosaics[i].vertexes) {
            Vector2d p = mv.uv - lo[i];
            if (turned[i]) p = Vector2d(hi[i].y() - lo[i].y() - p.y(), p.x());  // 90 degrees counter-clockwise
            mv.uv = ((p + corner) * scale).cwiseMax(0.0).cwiseMin(1.0);
        }
        mosaics[i].scale = static_cast<float>(mosaics[i].scale * scale);
    }
    return scale;
}
}  // namespace mopt
