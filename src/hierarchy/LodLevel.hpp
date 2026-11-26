#pragma once
// One level of detail of the region hierarchy (see MeshletBuilder.hpp): the level's
// remeshed surface and the hierarchy node of each face and vertex. Level 0 is the input mesh with the
// base charts as regions; level k > 0 is level k-1's surface remeshed inside the level-k regions.
#include <mesh/Mesh.hpp>

#include <vector>

namespace mopt
{
struct LodLevel {
    int level = 0;
    // The level's surface, stitched along the kept region boundaries. The next level starts here.
    Mesh simplified;
    // Region (node at this level) of each face.
    std::vector<int> simplifiedFaceRegion;
    // Region of each vertex, -1 for vertices on a region boundary.
    std::vector<int> simplifiedVertexRegion;
    // Isometric UVs in double precision, 3 per triangle (MeshletBuilder::makeIsometricUvs()); empty otherwise.
    // simplified.texcoords holds them rounded to float, for the .mlod.
    std::vector<Vector2d> isometricUvs;
};
}  // namespace mopt
