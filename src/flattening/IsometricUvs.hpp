/*
 * IsometricUvs.hpp
 *
 * Turns a mesh's per-corner UVs (a packed atlas, one chart per region) into isometric ones, in double
 * precision, in two steps:
 *
 *  1. Isometric triangles (as wedge's ChartDisconnectedConformalOp): every triangle is replaced by its 3D
 *     triangle laid flat (same angles), turned to line up with its atlas triangle and scaled to its area
 *     about the atlas triangle's centroid, then scaled about that centroid until it touches the atlas
 *     triangle from inside, and shrunk if needed to its region's median UV/3D scale. Triangles no longer
 *     share UVs: each has its own three.
 *  2. Relaxation: each triangle of a region moves as a similarity (a translation, a rotation and a uniform
 *     scale, so its angles stay its 3D angles) so that corresponding corners come together, the way the
 *     springls relax step pulls neighboring springls together, and so that every triangle's area becomes
 *     proportional to its 3D area (authalic). A corner corresponds to its vertex's copy in the triangle across
 *     each of its two edges, when that triangle is in the same region. The triangle whose UV centroid is
 *     closest to the region's area-weighted UV centroid stays fixed, and sets the region's UV/3D scale.
 *     Unknowns per triangle: translation, rotation and log scale. Least squares on the corner gaps plus, per
 *     triangle, areaWeight * (its 3D size at the fixed triangle's scale) * (its log UV/3D scale - the fixed
 *     triangle's), solved by Levenberg-Marquardt: each iteration linearizes the rotations and scales, solves
 *     the damped sparse normal equations with conjugate gradients, and keeps the step only if the cost drops. areaWeight 0 keeps every triangle's size (rigid motions).
 */
#pragma once

#include <mesh/Mesh.hpp>

#include <vector>

namespace mopt
{

struct IsometricUvParams {
    int iterations = 20;       // Gauss-Newton iterations of step 2 (0: step 1 only)
    double areaWeight = 1.0;   // weight of the authalic term (0: triangles keep their size)
    double tolerance = 1e-12;  // step 2 stops once no unknown changes by more than this
    int cgIterations = 2000;   // conjugate-gradient iterations per Gauss-Newton step
};

struct IsometricUvStats {
    double meanGapBefore = 0.0;  // mean distance between corresponding corners after step 1
    double meanGapAfter = 0.0;   // ... after step 2
    int iterations = 0;          // Gauss-Newton iterations run (the most of any region)
};

// UVs from mesh.texcoords (the atlas, 3 per triangle) to uv (3 per triangle, double); faceRegion gives the
// chart of every triangle.
IsometricUvStats MakeIsometricUvs(const Mesh& mesh, const std::vector<int>& faceRegion, std::vector<Vector2d>& uv,
                                  const IsometricUvParams& params = {});

}  // namespace mopt
