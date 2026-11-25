#pragma once
// Triangle area and point-to-triangle distance.
#include <core/Types.hpp>

#include <cmath>

namespace mopt
{
template <class T>
T TriangleArea(const Eigen::Matrix<T, 2, 1>& p1, const Eigen::Matrix<T, 2, 1>& p2, const Eigen::Matrix<T, 2, 1>& p3)
{
    return T(0.5) * std::abs(p1.x() * p2.y() - p1.y() * p2.x() + p2.x() * p3.y() - p2.y() * p3.x() + p3.x() * p1.y() -
                             p3.y() * p1.x());
}

template <class T>
T TriangleArea(const Eigen::Matrix<T, 3, 1>& p1, const Eigen::Matrix<T, 3, 1>& p2, const Eigen::Matrix<T, 3, 1>& p3)
{
    return T(0.5) * (p2 - p1).cross(p3 - p1).norm();
}

// Squared distance from p to the triangle (v0, v1, v2), and the closest point on it (Eberly's regions).
template <class T>
T DistanceToTriangleSqr(const Eigen::Matrix<T, 3, 1>& p, const Eigen::Matrix<T, 3, 1>& v0,
                        const Eigen::Matrix<T, 3, 1>& v1, const Eigen::Matrix<T, 3, 1>& v2,
                        Eigen::Matrix<T, 3, 1>* closestPoint = nullptr)
{
    using V = Eigen::Matrix<T, 3, 1>;
    const V e0 = v1 - v0, e1 = v2 - v0, dv = v0 - p;
    const T a = e0.dot(e0), b = e0.dot(e1), c = e1.dot(e1), d = e0.dot(dv), e = e1.dot(dv);
    const T det = a * c - b * b;
    T s = b * e - c * d;
    T t = b * d - a * e;
    int region;
    if (s + t <= det) {
        region = s < 0 ? (t < 0 ? 4 : 3) : (t < 0 ? 5 : 0);
    } else {
        region = s < 0 ? 2 : (t < 0 ? 6 : 1);
    }
    if (region == 0) {
        const T invDet = T(1) / det;
        s *= invDet;
        t *= invDet;
    } else if (region == 1) {
        const T numer = c + e - b - d;
        if (numer < 0) {
            s = 0;
        } else {
            const T denom = a - 2 * b + c;
            s = numer >= denom ? 1 : numer / denom;
        }
        t = 1 - s;
    } else if (region == 2) {
        const T tmp0 = b + d, tmp1 = c + e;
        if (tmp1 > tmp0) {
            const T numer = tmp1 - tmp0, denom = a - 2 * b + c;
            s = numer >= denom ? 1 : numer / denom;
            t = 1 - s;
        } else {
            s = 0;
            t = tmp1 <= 0 ? 1 : (e >= 0 ? 0 : -e / c);
        }
    } else if (region == 3) {
        s = 0;
        t = e >= 0 ? 0 : (-e >= c ? 1 : -e / c);
    } else if (region == 4) {
        const T tmp0 = c + e, tmp1 = a + d;
        if (tmp0 > tmp1) {
            s = 0;
            t = tmp1 <= 0 ? 1 : (e >= 0 ? 0 : -e / c);
        } else {
            t = 0;
            s = tmp1 <= 0 ? 1 : (d >= 0 ? 0 : -d / a);
        }
    } else if (region == 5) {
        t = 0;
        s = d >= 0 ? 0 : (-d >= a ? 1 : -d / a);
    } else {
        const T tmp0 = b + e, tmp1 = a + d;
        if (tmp1 > tmp0) {
            const T numer = tmp1 - tmp0, denom = c - 2 * b + a;
            t = numer >= denom ? 1 : numer / denom;
            s = 1 - t;
        } else {
            t = 0;
            s = tmp1 <= 0 ? 1 : (d >= 0 ? 0 : -d / a);
        }
    }
    const V closest = v0 + s * e0 + t * e1;
    if (closestPoint) *closestPoint = closest;
    return (p - closest).squaredNorm();
}
}  // namespace mopt
