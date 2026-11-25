#pragma once
// Closest point on a triangle mesh, for snapping resampled vertices back onto the input surface
// (libkdtree's BoxTree over the mesh triangles).
#include <core/Types.hpp>
#include <mesh/Mesh.hpp>

#include <boxtree.hpp>

namespace mopt
{
// Closest point on a triangle mesh (libkdtree BoxTree over its triangles).
class MeshProjector
{
public:
    explicit MeshProjector(const Mesh& mesh);
    // The point of the mesh closest to p (p itself for a mesh without triangles). Thread-safe.
    Vector3f closestPoint(const Vector3f& p) const;

private:
    libkdtree::ClosestPoint3D<> tree_;
};
}  // namespace mopt
