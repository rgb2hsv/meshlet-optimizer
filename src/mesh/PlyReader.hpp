#pragma once
// Reads a triangle mesh from PLY (ASCII or binary, via happly).
#include <mesh/Mesh.hpp>

#include <filesystem>

namespace mopt
{
// Positions, normals when present (computed otherwise) and faces; quads are split in two and larger
// polygons fanned. Throws std::runtime_error on a file without vertices or with bad indices.
void ReadPlyMesh(const std::filesystem::path& file, Mesh& mesh);
}  // namespace mopt
