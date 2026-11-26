/*
 * IsometricUvs.hpp
 *
 * Turns a mesh's per-corner UVs (a packed atlas, one chart per region) into isometric ones, in two steps:
 *
 *  1. Isometric triangles (as wedge's ChartDisconnectedConformalOp): every triangle is replaced by its 3D
 *     triangle laid flat (same angles), turned to line up with its atlas triangle and scaled to its area
 *     about the atlas triangle's centroid, then scaled about that centroid until it touches the atlas
 *     triangle from inside, and shrunk if needed to its region's median UV/3D scale. Triangles no longer
 *     share UVs: each has its own three.
 *  2. Rigid relaxation: each triangle of a region moves rigidly (a translation and a rotation, no change of
 *     shape or size) so that the corners that triangles of the region share in 3D come together, the way the
 *     springls relax step pulls neighboring springls together. The triangle whose UV centroid is closest to
 *     the region's (area-weighted) UV centroid stays fixed. Least squares on the corner gaps, solved by
 *     Gauss-Newton: each iteration linearizes the rotations and solves the sparse normal equations with
 *     conjugate gradients.
 */
#pragma once

#include <mesh/Mesh.hpp>

#include <vector>

namespace mopt
{

struct IsometricUvParams {
    int rigidIterations = 20;   // Gauss-Newton iterations of step 2 (0: step 1 only)
    double tolerance = 1e-9;    // step 2 stops once no unknown changes by more than this (UV units / radians)
    int cgIterations = 1000;    // conjugate-gradient iterations per Gauss-Newton step
};

struct IsometricUvStats {
    double meanGapBefore = 0.0;  // mean distance between shared corners after step 1
    double meanGapAfter = 0.0;   // ... after step 2
    int iterations = 0;          // Gauss-Newton iterations run (the most of any region)
};

// mesh.texcoords (3 per triangle) are replaced; faceRegion gives the chart of every triangle.
IsometricUvStats MakeIsometricUvs(Mesh& mesh, const std::vector<int>& faceRegion, const IsometricUvParams& params = {});

}  // namespace mopt
