#pragma once
#include <core/Check.hpp>
#include <core/Types.hpp>
#include <filesystem>
#include <mesh/Mesh.hpp>
#include <unordered_map>
#include <vector>

namespace mopt
{
// A triangle of a mosaic: its mesh face, its corners as layout vertices, and fold / boundary flags.
struct MosaicFace {
    uint32_t id;
    size_t fid = 0;
    Vector3u meshFace;
    Vector3u mapFace;
    double angleError = 0;
    double areaError = 0;
    Vector3d edgeLengths;
    bool boundary;
    bool locked;
    bool flipped;
    MosaicFace() = default;
    MosaicFace(uint32_t id, size_t fid, Vector3u meshFace, Vector3u mapFace, bool boundary = false)
        : id(id), fid(fid), meshFace(meshFace), mapFace(mapFace), boundary(boundary), locked(false), flipped(false)
    {
    }
};
// A vertex of a mosaic: its UV position, the mesh vertex it copies (meshId) and its layout id (mapId).
struct MosaicVertex {
    Vector2d uv;
    bool locked;
    bool boundary;
    uint32_t meshId;
    uint32_t mapId;
    MosaicVertex() = default;
    MosaicVertex(uint32_t meshId, uint32_t mapId, bool boundary = false)
        : locked(false), boundary(boundary), meshId(meshId), mapId(mapId)
    {
    }
};
// One chart flattened to UV: its vertices and faces, the plane it was projected onto and its UV scale.
struct Mosaic {
    int id;
    Matrix4f pose = Matrix4f::Identity();
    float scale = 1.0f;
    std::vector<MosaicVertex> vertexes;
    std::vector<MosaicFace> faces;
    std::vector<Vector2u> edges;
    std::unordered_map<uint32_t, size_t> vertexMap;
    MosaicVertex& getLocalVertex(uint32_t uid);
    const MosaicVertex& getLocalVertex(uint32_t uid) const;
    Mosaic(int id = -1) : id(id) {}
};

}  // namespace mopt
