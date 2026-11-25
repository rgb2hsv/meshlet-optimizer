#pragma once
// Vector and matrix types: Eigen's, plus the unsigned and byte vectors Eigen does not name.
// Built with EIGEN_INITIALIZE_MATRICES_BY_ZERO, so default-constructed vectors are zero.
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace mopt
{
using Eigen::Matrix2d;
using Eigen::Matrix2f;
using Eigen::Matrix3d;
using Eigen::Matrix3f;
using Eigen::Matrix4d;
using Eigen::Matrix4f;
using Eigen::Vector2d;
using Eigen::Vector2f;
using Eigen::Vector2i;
using Eigen::Vector3d;
using Eigen::Vector3f;
using Eigen::Vector3i;
using Eigen::Vector4d;
using Eigen::Vector4f;
using Eigen::Vector4i;
using Vector2u = Eigen::Matrix<uint32_t, 2, 1>;
using Vector3u = Eigen::Matrix<uint32_t, 3, 1>;
using Vector4u = Eigen::Matrix<uint32_t, 4, 1>;
using Vector3ub = Eigen::Matrix<uint8_t, 3, 1>;
using Vector4ub = Eigen::Matrix<uint8_t, 4, 1>;
using AlignedBox2f = Eigen::AlignedBox2f;
using AlignedBox3f = Eigen::AlignedBox3f;

// Hash of a small integer vector, for unordered containers.
struct VectorHash {
    template <class T, int N>
    size_t operator()(const Eigen::Matrix<T, N, 1>& v) const noexcept
    {
        size_t h = 0;
        for (int i = 0; i < N; i++) h ^= std::hash<T>()(v[i]) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        return h;
    }
};
}  // namespace mopt
