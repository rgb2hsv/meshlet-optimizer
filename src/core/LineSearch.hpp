#pragma once
// Brent's method: minimizes a 1D function on [minValue, maxValue] by golden-section and parabolic steps.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <tuple>
#include <utility>

namespace mopt
{
// value is set to the best point found. Returns {iterations, best score}.
template <class ScalarType>
std::tuple<size_t, double> SolveBrentLineSearch(ScalarType& value, const ScalarType& minValue,
                                                const ScalarType& maxValue,
                                                const std::function<double(const ScalarType& value)>& scoreFunc,
                                                ScalarType err = ScalarType(1e-5), size_t maxIterations = 100)
{
    constexpr double kGolden = 0.3819660112501051;  // (3 - sqrt(5)) / 2
    const double relEps = std::sqrt(std::numeric_limits<double>::epsilon());
    const double tol = std::max(double(err), 0.0);
    double a = double(minValue), b = double(maxValue);
    if (b < a) std::swap(a, b);
    auto f = [&](double t) { return scoreFunc(ScalarType(t)); };

    // x: best so far, w: second best, v: previous w.
    double x = a + kGolden * (b - a), w = x, v = x;
    double fx = f(x), fw = fx, fv = fx;
    double d = 0.0, e = 0.0;  // e: step before last
    size_t iter = 0;
    for (; iter < maxIterations; iter++) {
        const double xm = 0.5 * (a + b);
        const double tol1 = relEps * std::abs(x) + tol / 3.0;
        const double tol2 = 2.0 * tol1;
        if (std::abs(x - xm) <= tol2 - 0.5 * (b - a)) break;

        bool goldenStep = true;
        if (std::abs(e) > tol1) {
            // Parabola through x, w, v.
            double r = (x - w) * (fx - fv);
            double q = (x - v) * (fx - fw);
            double p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if (q > 0.0) p = -p;
            q = std::abs(q);
            const double eOld = e;
            e = d;
            if (std::abs(p) < std::abs(0.5 * q * eOld) && p > q * (a - x) && p < q * (b - x)) {
                d = p / q;
                const double u = x + d;
                if (u - a < tol2 || b - u < tol2) d = (x < xm) ? tol1 : -tol1;
                goldenStep = false;
            }
        }
        if (goldenStep) {
            e = (x < xm) ? b - x : a - x;
            d = kGolden * e;
        }
        const double u = (std::abs(d) >= tol1) ? x + d : x + (d > 0.0 ? tol1 : -tol1);
        const double fu = f(u);
        if (fu <= fx) {
            if (u < x)
                b = x;
            else
                a = x;
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if (u < x)
                a = u;
            else
                b = u;
            if (fu <= fw || w == x) {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if (fu <= fv || v == x || v == w) {
                v = u;
                fv = fu;
            }
        }
    }
    value = ScalarType(x);
    return {iter, fx};
}
}  // namespace mopt
