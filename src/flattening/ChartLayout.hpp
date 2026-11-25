#pragma once
// A mesh cut into labelled charts and their flattened UV layouts (mosaics), as built and read by the
// flattening functions (ChartFlattening.hpp).
#include <core/Types.hpp>
#include <flattening/Mosaic.hpp>
#include <mesh/Mesh.hpp>

#include <cstdint>
#include <set>
#include <vector>

namespace mopt
{
struct ChartLayout {
    Mesh mesh;
    // Chart of each vertex (-1 on a seam) and of each face.
    std::vector<int> vertLabels;
    std::vector<int> faceLabels;
    MeshListNeighborTable vertNbrs;
    int chartCount = 0;

    // One flattened chart per label, and the map from layout vertices (mapId) to mesh vertices.
    std::vector<Mosaic> mosaics;
    std::vector<uint32_t> textureToMeshMap;
    size_t totalTriangleCount = 0;
    size_t totalVertexCount = 0;
    size_t totalBoundaryCount = 0;
    // Smallest triangle area used to weight the LSCM rows.
    const double minArea = 1E-6;
    std::vector<bool> boundaryMap;
    std::vector<std::vector<uint32_t>> mosaicVertNbr;
    std::vector<std::vector<uint32_t>> mosaicFaceNbr;
    // Layout vertices held fixed by LSCM.
    std::set<uint32_t> pinned;
};
}  // namespace mopt
