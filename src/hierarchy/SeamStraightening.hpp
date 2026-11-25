/*
 * SeamStraightening.hpp
 *
 * Straighten the borders between face regions of a mesh, keeping junctions.
 *
 * The borders (edges whose two faces lie in different regions) break into
 * chains: runs of border edges between the same two regions, from one
 * junction to another. A junction is a vertex where three or more regions
 * meet, where the border branches or pinches, or on the mesh's own border;
 * junctions never move.
 *
 * Each chain is replaced by the cheapest path along mesh edges between its
 * two junctions, through vertices that lie inside the two regions or on the
 * chain itself -- never on another border -- so no other chain is touched.
 * An edge costs its 3D length, stretched by its distance from the bisector
 * plane of the two regions' centroids (bisectorWeight; 0: plain shortest
 * path). Region centroids stand in for CVT seeds (a converged CVT's seeds sit
 * at their cells' centroids), so the bisector cut through the surface is the
 * restricted Voronoi border between the two regions (Yan et al. 2009), and
 * seams drawn towards it keep the CVT's even cell shapes rather than just
 * being short. The faces between the old chain and the new path change
 * sides: the two regions' faces, cut along the new path, fall into pieces,
 * and each piece takes the region most of its faces had.
 *
 * A chain is only replaced when both regions stay non-empty, in no more
 * edge-connected pieces than before and of the same topology (V - E + F),
 * and no vertex around the moved faces starts or stops being a junction;
 * otherwise it is kept.
 * Closed border loops (an island inside one other region) have no junction
 * to keep and are left as they are.
 */
#pragma once

#include <vector>

namespace mopt
{

class Mesh;

// Settings of the seam straightening: how strongly paths are kept near the region bisector.
struct SeamStraighteningParams {
    // Extra edge cost per mean edge length of distance from the bisector of
    // the two regions' centroids; 0 gives the plain shortest path.
    // 2 keeps seams about as close to the Voronoi borders as the CVT labels
    // put them while still shortening them by about a third (torus test).
    double bisectorWeight = 2.0;
};

// What the seam straightening did: chains straightened or rejected, faces moved, border length.
struct SeamStraighteningStats {
    int chains = 0;             // junction-to-junction chains
    int straightened = 0;       // replaced by a shorter path
    int rejected = 0;           // a shorter path existed but broke a region
    int loops = 0;              // closed border loops, left as they are
    int facesMoved = 0;         // faces that changed region
    double lengthBefore = 0.0;  // total border length
    double lengthAfter = 0.0;
    // Length-weighted mean distance of the borders from the bisectors of the
    // regions they separate, in mean edge lengths.
    double bisectorDistanceBefore = 0.0;
    double bisectorDistanceAfter = 0.0;
};

// faceRegion: one region id per face (>= 0), updated in place.
SeamStraighteningStats StraightenRegionSeams(const Mesh& mesh, std::vector<int>& faceRegion,
                                             const SeamStraighteningParams& params = {});

}  // namespace mopt
