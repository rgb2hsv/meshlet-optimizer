/*
 * BorderSimplification.hpp
 *
 * Coarsen the borders between face regions of a mesh by edge collapses
 * along them, so border edges get about as long as the regions' interior
 * edges.
 *
 * The borders (edges whose two faces lie in different regions) break into
 * chains, as in SeamStraightening.hpp: runs of border edges from junction to
 * junction, where a junction is a vertex where three or more regions meet,
 * where the border branches or pinches, or on the mesh's own border.
 * Border edges left over form closed loops.
 *
 * Each chain has a target edge length: the mean of its two regions' mean
 * interior edge lengths (an interior edge has both faces in one region),
 * times targetScale. A region without interior edges uses its other
 * region's value, or the mesh's mean interior edge length. While a chain's
 * mean edge length is below its target, its shortest edge is collapsed: the
 * chain vertex at one end (never a junction) moves onto the other, which
 * keeps its position, and the two faces on the edge go. Interior vertices
 * and junctions never move. A closed loop keeps at least three vertices.
 *
 * A collapse is refused when it would fold the mesh: a remaining triangle
 * around the moved vertex whose normal flips, turns more than
 * maxNormalChangeDegrees from where it started, or whose area vanishes; a
 * collapse breaking the link condition (the mesh would stop being
 * manifold); or one removing an ear, a face on the collapsed edge whose
 * other two edges are borders too (its region would change topology or
 * vanish). The vertex then stays.
 */
#pragma once

#include <core/Check.hpp>
#include <core/Types.hpp>

#include <functional>
#include <utility>
#include <vector>

namespace mopt
{

class Mesh;

// Settings of the border coarsening: target edge length, normal-change limit and the snap callback.
struct BorderSimplificationParams {
    // Chain target = targetScale * the regions' mean interior edge length.
    double targetScale = 1.0;
    // A remaining triangle may turn at most this far from its normal before
    // any collapse; beyond it the collapse counts as a fold. Without it a
    // triangle can turn by up to 90 degrees one collapse at a time and end
    // as a sliver standing on edge.
    double maxNormalChangeDegrees = 60.0;
    // Called before each collapse that passes the checks above, with every
    // removed vertex that would then lie under the merged chain edge a-b
    // (vertex indices in the mesh as given, before compaction) and its
    // closest point on that edge. Returning false refuses the collapse
    // (rejectedSnap). Used to move the finer levels' copies of removed
    // vertices onto the coarser border, and to refuse removals whose snap
    // would fold a finer triangle.
    std::function<bool(const std::vector<std::pair<uint32_t, Vector3f>>& snaps, uint32_t a, uint32_t b)> trySnap;
};

// faceRegion: one region id (>= 0) per face. On return the collapsed faces
// are gone and unused vertices dropped (vertexLocations, and vertexNormals /
// texcoords, when sized
// 3 * faces, keeps the remaining faces' corners), and faceRegion matches.
void SimplifyRegionBorders(Mesh& mesh, std::vector<int>& faceRegion, const BorderSimplificationParams& params = {});

}  // namespace mopt
