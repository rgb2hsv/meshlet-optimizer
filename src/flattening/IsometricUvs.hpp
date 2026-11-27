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
 *     closest to the region's area-weighted UV centroid stays fixed. The region's average log UV/3D scale is
 *     the authalic target and is held fixed as a constraint (the free triangles' log scales sum to zero,
 *     eliminated through one of them). Unknowns per triangle: translation, rotation and log scale. Least
 *     squares on the corner gaps plus, per triangle, areaWeight * (its 3D size at the average scale) * (its log
 *     UV/3D scale - the average), solved by Gauss-Newton: each iteration linearizes the rotations and scales
 *     and solves the sparse normal equations with conjugate gradients. areaWeight 0 keeps every triangle's
 *     size (rigid motions).
 *  2b. Constraint enforcement (enforceNoOverlaps): while two triangles of a region overlap, the relaxation runs
 *     again with one-sided springs between the centroids of neighboring triangles: a pair closer than its rest
 *     length is pushed apart. A pair sharing an edge starts at the centroid distance it has when the two lie flat
 *     side by side across that edge; a pair sharing only a vertex starts at its current centroid distance. Every
 *     round each rest length grows by springRestGrowth of its start and the spring strength doubles.
 *  3. Fit (fitToAtlas): each region is turned, scaled uniformly and moved so its bounding box fits
 *     tightly inside the bounding box of its chart in the atlas, centered, at whichever of eight turns (k * 22.5
 *     degrees) allows the largest scale: the packed charts' room is reused and
 *     relaxed charts cannot run into each other.
 */
#pragma once

#include <mesh/Mesh.hpp>

#include <vector>

namespace mopt
{

struct IsometricUvParams {
    int iterations = 20;       // Gauss-Newton iterations of step 2 (0: step 1 only)
    double areaWeight = 1.0;   // weight of the authalic term (0: triangles keep their size)
    // Step 2b: one-sided springs push neighboring triangles' centroids apart (sharing an edge or a vertex). Each
    // round every rest length grows by springRestGrowth of its starting value and the strength (from
    // springStrength) doubles, for at most springRounds rounds, until no two triangles of a region overlap.
    bool enforceNoOverlaps = true;
    double springStrength = 0.1;
    double springRestGrowth = 0.01;
    double springMaxStrength = 12.8;  // the strength stops doubling here (Gauss-Newton diverges on very stiff springs)
    int springRounds = 40;
    bool fitToAtlas = true;    // step 3: each region turned, scaled uniformly and moved to fit tightly inside its atlas chart's box
    double tolerance = 1e-12;  // step 2 stops once no unknown changes by more than this
    int cgIterations = 2000;   // conjugate-gradient iterations per Gauss-Newton step
};

struct IsometricUvStats {
    double meanGapBefore = 0.0;  // mean distance between corresponding corners after step 1
    double meanGapAfter = 0.0;   // ... after step 2
    int iterations = 0;          // Gauss-Newton iterations run (the most of any region)
    int springRounds = 0;        // enforcement rounds run (the most of any region)
    int overlappingRegions = 0;  // regions still overlapping after the last round
};

// UVs from mesh.texcoords (the atlas, 3 per triangle) to uv (3 per triangle, double); faceRegion gives the
// chart of every triangle.
IsometricUvStats MakeIsometricUvs(const Mesh& mesh, const std::vector<int>& faceRegion, std::vector<Vector2d>& uv,
                                  const IsometricUvParams& params = {});

}  // namespace mopt
