/*
 * DiskGroups.hpp
 *
 * Keep groups of regions disk-shaped. A group (a set of regions, each a set
 * of mesh faces) is a disk when its faces are connected and V - E + F = 1.
 * LSCM and the fold-free fallback both need disks; a group that rings a
 * hole, wraps a handle or pinches at a vertex is none.
 *
 *   Closing (graph morphology): remove the group from the group adjacency
 *   graph; every connected piece left over except the largest is a hole the
 *   group surrounds. Holes up to maxHoleFraction of the group's own size
 *   (in faces) are filled: their regions join the group. A hole is always a
 *   union of whole groups, so filling one never cuts another group apart.
 *
 *   Ejection: a group still not a disk sheds members one at a time -- the
 *   one whose removal leaves a connected disk, else the most disk-like
 *   remainder, lightest first -- up to half its members. Each shed member
 *   joins a neighbouring group that stays a disk with it (the lightest),
 *   else becomes a group of its own. If that never makes a disk the group is
 *   left as it was (e.g. the last few groups covering a torus cannot all be
 *   disks).
 */
#pragma once

#include <vector>

namespace mopt
{

class Mesh;

// Which disk repairs to run, and the largest hole that may be filled.
struct DiskGroupParams {
    bool fillHoles = true;
    // Largest hole filled, as a fraction of the group's face count.
    double maxHoleFraction = 0.5;
    bool ejectMembers = true;
};

// faceRegion: region of each mesh face, in [0, regionCount).
// regionAdjacency: regions sharing an edge. regionWeight: tie-breaks.
// group[region] (any ids) is updated in place, absorbed[region] set to 1 for
// regions that joined a group by filling or ejection and 0 for regions made
// groups of their own. New group ids are larger than any given one.
bool MakeGroupsDisks(const Mesh& mesh, const std::vector<int>& faceRegion, int regionCount,
                     const std::vector<std::vector<int>>& regionAdjacency, const std::vector<int>& regionWeight,
                     std::vector<int>& group, std::vector<char>& absorbed, const DiskGroupParams& params = {},
                     const std::vector<int>& regionIsland = {});

// V - E + F of the faces of the given regions, as one surface.
long long RegionsEulerCharacteristic(const Mesh& mesh, const std::vector<std::vector<int>>& facesOfRegion,
                                     const std::vector<int>& regions);

}  // namespace mopt
