/*
 * IsometricUvs.hpp
 *
 * Turns a mesh's per-corner UVs (a packed atlas, one chart per region) into isometric ones, in double
 * precision. Every triangle gets its own three UVs, similar to its 3D triangle and, within a chart, at one
 * common scale, so every angle is its 3D angle and every area is the same multiple of its 3D area. The
 * triangles are then arranged as rigid bodies:
 *
 *  1. Placement: every triangle is its 3D triangle laid flat (same angles), scaled by its chart's scale
 *     sqrt(chart's atlas area / chart's 3D area), turned so its corner 0 points the way its atlas triangle's
 *     corner 0 does, and centered on its atlas triangle's centroid.
 *  2. Simulation: per chart, the triangles are rigid bodies (a centroid and a rotation; shape and size never
 *     change) with velocities that carry over from step to step. Every two corners of a chart that share an
 *     atlas location (the same vertex) are joined by a spring of weight 1 / (N - 1), N the number of corners
 *     there, so each corner's springs weigh 1 in all. Every step first computes, from the state at its start:
 *     the spring forces (stiffness * weight * the gap between the two corners, times the lighter body's mass)
 *     and their torques; and, for every pair of the chart's triangles that overlap (enforceNoOverlaps), a
 *     normal impulse at the centroid of their overlap along the axis where they overlap least,
 *       j = (-(1 + e) v_rel . n + baumgarte * depth / dt) / (1/m_A + 1/m_B + (r_A x n)^2 / I_A + (r_B x n)^2 / I_B),
 *     kept >= 0, -j n on A and +j n on B. Then it applies them to every body's velocity and spin, damps them, and
 *     integrates position and rotation (dt = 1; mass is UV area, I = m (|r_0|^2 + |r_1|^2 + |r_2|^2) / 12). The
 *     first relaxSteps steps use springStiffness, closing the gaps; then up to repelSteps steps use the weak
 *     repelSpringStiffness until no two triangles of the chart overlap by more than overlapTolerance.
 *  3. Fit (fitToAtlas): each chart is turned, scaled uniformly and moved so its bounding box fits tightly
 *     inside the bounding box of its chart in the atlas, centered, at whichever of eight turns (k * 22.5
 *     degrees) allows the largest scale: the packed charts' room is reused and charts cannot run into each other.
 */
#pragma once

#include <mesh/Mesh.hpp>

#include <array>
#include <vector>

namespace mopt
{

struct IsometricUvParams {
    int relaxSteps = 400;                 // simulation steps with springStiffness (0, and no overlaps: step 1 only)
    double springStiffness = 0.05;        // springs while closing the gaps
    bool enforceNoOverlaps = true;        // overlapping triangles repel each other (impulses)
    int repelSteps = 400;                 // at most this many further steps, until nothing overlaps
    double repelSpringStiffness = 0.001;  // springs while separating overlaps: weak against the repulsion
    double baumgarte = 0.5;               // the share of a pair's overlap its impulse separates per step
    double restitution = 0.0;             // e
    double damping = 0.1;                 // the share of every velocity lost per step
    // Two triangles overlap when they do by more than this (UV units): half a texel at 4096^2, which leaves room
    // for the fit into the atlas box (step 3) scaling a chart up.
    double overlapTolerance = 0.5 / 4096;
    bool fitToAtlas = true;  // step 3
};

// UVs from mesh.texcoords (the atlas, 3 per triangle) to uv (3 per triangle, double); faceRegion gives the
// chart of every triangle.
void MakeIsometricUvs(const Mesh& mesh, const std::vector<int>& faceRegion, std::vector<Vector2d>& uv, const IsometricUvParams& params = {});

// How two triangles overlap. Whether their interiors meet (touching is not overlapping) and the polygon where they
// do come from Geometric Tools' 2D triangle-triangle queries, ported in IsometricUvs.cpp. depth: the smallest
// overlap of their projections on their six edge normals (0 when they do not overlap); axis: the unit normal it
// is measured along, pointing from p to q; center: the centroid of the overlap polygon.
struct TriangleOverlap {
    double depth = 0.0;
    Vector2d axis = Vector2d::Zero();
    Vector2d center = Vector2d::Zero();
};
TriangleOverlap OverlapOf(const std::array<Vector2d, 3>& p, const std::array<Vector2d, 3>& q);

}  // namespace mopt
