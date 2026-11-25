#pragma once
// Writes a triangle mesh as binary PLY (via happly).
#include <mesh/Mesh.hpp>

#include <filesystem>
#include <vector>

namespace mopt
{
// Positions, normals when present, faces, texture coordinates per face corner ("texcoord" lists, as
// ReadPlyMesh and MeshLab read them) when present, and, when given, an int "region" per face.
void WritePlyMesh(const std::filesystem::path& file, const Mesh& mesh, const std::vector<int>& faceRegion = {});
}  // namespace mopt
