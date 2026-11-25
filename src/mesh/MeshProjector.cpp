#include <mesh/MeshProjector.hpp>

#include <vector>

namespace mopt
{
namespace
{
libkdtree::Vec3 ToTree(const Vector3f& p) { return {double(p.x()), double(p.y()), double(p.z())}; }
}  // namespace

MeshProjector::MeshProjector(const Mesh& mesh)
{
    std::vector<libkdtree::Triangle<3>> triangles;
    triangles.reserve(mesh.triIndexes.size());
    for (const Vector3u& t : mesh.triIndexes) {
        triangles.push_back({ToTree(mesh.vertexLocations[t[0]]), ToTree(mesh.vertexLocations[t[1]]),
                             ToTree(mesh.vertexLocations[t[2]])});
    }
    tree_ = libkdtree::ClosestPoint3D<>(std::move(triangles));
}

Vector3f MeshProjector::closestPoint(const Vector3f& p) const
{
    libkdtree::Vec3 closest;
    if (!tree_.closestPoint(ToTree(p), closest)) return p;
    return Vector3f(float(closest[0]), float(closest[1]), float(closest[2]));
}
}  // namespace mopt
