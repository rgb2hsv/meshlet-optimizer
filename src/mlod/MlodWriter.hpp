/*
 * MlodWriter.hpp
 *
 * Writes the hierarchy's levels of detail as a .mlod file, the meshlet LOD
 * format of meshlet-viewer (src/meshlet_file.* there):
 * the mesh, the builder settings and the full meshlet DAG (clusters, cone
 * bounds, per-cluster LOD errors, edge masks and the LOD selection BVH) in
 * one cereal portable binary archive (magic "MLOD"). Version 1 has no
 * texture coordinates; version 2 adds one per vertex, right after the
 * vertices, and splits vertices on texture seams (one per texture coordinate).
 *
 * The hierarchy maps onto the meshlet DAG as follows:
 *   - every level's surface is part of the vertex buffer; the index buffer
 *     is level 0 (the input mesh);
 *   - each region of each level is exactly one meshlet, as large as the
 *     region; only a region beyond the format's ceiling (8-bit local vertex
 *     indices: 256 vertices; the viewer's mesh shader: 256 triangles) is
 *     split, into the fewest edge-connected parts of about equal size. The
 *     file's meshlet size settings record the largest meshlet written;
 *   - a cluster group at DAG depth k is the set of level-k regions that
 *     merge into one region at level k+1; its "simplified" bounds and error
 *     are that region's (the remeshed geometry that replaces them). The
 *     halves of a split region share their children, so they count as one
 *     region here. The top level's regions form terminal groups (error
 *     FLT_MAX);
 *   - a region's error is the larger of its children's errors and the
 *     distance between its remeshed surface and its children's surface,
 *     measured both ways at vertices (an approximate Hausdorff distance), so
 *     errors only grow up the DAG; bounding spheres are merged upward the
 *     same way;
 *   - the selection BVH is clodBuildHierarchy's (meshoptimizer's
 *     demo/clusterlod.h), 8 wide, one tree per depth, as the viewer builds it.
 * Swapping a group for the region above it is crack-free because region
 * borders are kept across levels (see MeshletBuilder.hpp).
 */
#pragma once

#include <hierarchy/LodLevel.hpp>
#include <hierarchy/RegionHierarchy.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace mopt
{

// What WriteMlodFile wrote.
struct MlodStats {
    size_t vertices = 0;
    size_t clusters = 0;
    size_t groups = 0;
    size_t triangles = 0;  // all levels
    int levels = 0;
};

// The levels and their region hierarchy as built by MeshletBuilder.
// Returns false and sets error when nothing could be written.
bool WriteMlodFile(const std::filesystem::path& file, const std::vector<LodLevel>& levels, const RegionHierarchy& tree,
                   std::string& error, MlodStats* stats = nullptr);

}  // namespace mopt
