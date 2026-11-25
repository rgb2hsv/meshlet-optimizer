#pragma once
// Indexed triangle mesh: positions, optional normals, optional per-corner texture coordinates.
#include <core/Check.hpp>
#include <core/Types.hpp>

#include <cstdint>
#include <filesystem>
#include <ostream>
#include <vector>

namespace mopt
{
// Indexed triangle mesh. Texture coordinates belong to triangle corners, so a vertex on a texture seam
// can have a different UV in each triangle around it.
class Mesh
{
public:
    std::vector<Vector3f> vertexLocations;
    std::vector<Vector3f> vertexNormals;
    std::vector<Vector3u> triIndexes;
    // Texture coordinates of the triangle corners (3 per triangle, in triangle order); empty when the mesh
    // has none.
    std::vector<Vector2f> texcoords;

    bool hasTexcoords() const { return !texcoords.empty() && texcoords.size() == 3 * triIndexes.size(); }
    void clear();
    // Area-weighted vertex normals (negated with flipSign).
    void updateVertexNormals(bool flipSign = false);
    virtual ~Mesh() = default;
};

template <class C, class R>
std::basic_ostream<C, R>& operator<<(std::basic_ostream<C, R>& ss, const Mesh& m)
{
    ss << "Mesh: " << m.vertexLocations.size() << " vertices, " << m.triIndexes.size() << " triangles";
    return ss;
}

using MeshListNeighborTable = std::vector<std::vector<uint32_t>>;
// Each vertex's neighbours in face order around it (the ring must be manifold).
void CreateOrderedVertexNeighborTable(const Mesh& mesh, MeshListNeighborTable& vertNbrs, bool leaveTail = false);
inline void MakeOrderedVertexNeighborTable(const Mesh& mesh, MeshListNeighborTable& vertNbrs, bool leaveTail = false)
{
    CreateOrderedVertexNeighborTable(mesh, vertNbrs, leaveTail);
}
}  // namespace mopt
