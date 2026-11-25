#pragma once
// Reads a triangle mesh from PLY (ASCII or binary, via happly).
#include <mesh/Mesh.hpp>

#include <filesystem>

namespace mopt
{
// Positions, normals when present (computed otherwise), faces, and texture coordinates per face corner
// ("texcoord" lists) or per vertex (u/v, s/t, texture_u/texture_v). Quads are split along the shorter
// diagonal and larger polygons fanned. Throws std::runtime_error on a file without vertices or with bad indices.
void ReadPlyMesh(const std::filesystem::path& file, Mesh& mesh);
}  // namespace mopt
