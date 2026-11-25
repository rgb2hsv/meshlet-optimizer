#pragma once
// Indexed surface mesh: positions, optional normals, triangles (and quads until
// convertQuadsToTriangles), optional per-corner texture coordinates.
#include <core/Check.hpp>
#include <core/Types.hpp>

#include <cstdint>
#include <filesystem>
#include <ostream>
#include <vector>

namespace mopt
{
// Indexed triangle mesh (quads only until convertQuadsToTriangles), with optional per-corner UVs.
class Mesh
{
public:
    std::vector<Vector3f> vertexLocations;
    std::vector<Vector3f> vertexNormals;

    std::vector<Vector4u> quadIndexes;
    std::vector<Vector3u> triIndexes;
    std::vector<Vector2f> textureMap;  // three (or four) entries per face, in face order

    void clear();
    void updateVertexNormals(bool flipSign = false, int smoothIterations = 0, float dotTolerance = 0.75f);
    bool convertQuadsToTriangles();
    virtual ~Mesh() = default;
};

template <class C, class R>
std::basic_ostream<C, R>& operator<<(std::basic_ostream<C, R>& ss, const Mesh& m)
{
    ss << "Mesh: " << m.vertexLocations.size() << " vertices, " << m.triIndexes.size() << " triangles";
    if (m.quadIndexes.size() > 0) ss << ", " << m.quadIndexes.size() << " quads";
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
