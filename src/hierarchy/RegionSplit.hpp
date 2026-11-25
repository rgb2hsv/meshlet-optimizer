/*
 * RegionSplit.hpp
 *
 * Splitting oversized face regions of a mesh in two (more than maxFaces / 2).
 *
 * A region's main body is split by growing two halves at once, each a flood
 * fill in order of distance between face centroids: one from a face on one
 * of its corners -- a face touching a vertex where three or more regions
 * meet, the one farthest from the body's first face, or simply the farthest
 * face when there is no such corner -- the other from the body face farthest
 * from that. The smaller half always claims the next face, so the halves
 * are edge-connected and about equal in triangles (unless one front is
 * boxed in by the other). The first half keeps the region id, the second
 * becomes a new region. Remeshing can leave single triangles that touch the
 * body only at a vertex (pinched outlines); they join the new half when they
 * share a vertex with it, else stay.
 */
#pragma once

#include <vector>

namespace mopt
{

class Mesh;

// A region split in two: the region, its new half and the faces that moved.
struct RegionSplit {
    int region = -1;     // the split region, which keeps the filled half
    int newRegion = -1;  // id given to the other half
    int keptFaces = 0;
    int movedFaces = 0;
};

// Regions with more than maxFaces / 2 faces are split.
// faceRegion holds one region id in [0, regionCount) per face and is updated
// in place; new regions get ids regionCount, regionCount + 1, ... in the
// order of the returned splits (ascending original id). Each region is split
// at most once, so a half may still exceed maxFaces.
std::vector<RegionSplit> SplitLargeRegions(const Mesh& mesh, std::vector<int>& faceRegion, int regionCount,
                                           int maxFaces);

// Isolated faces -- faces with edge neighbours, none of them in the face's
// own region -- join the neighbouring region with the fewest faces (ties:
// lower id). Repeated until none is left, since moving a face can isolate
// another. Returns the number of faces moved; a region can end up empty.
int RelabelIsolatedFaces(const Mesh& mesh, std::vector<int>& faceRegion);

// Regions whose faces share an edge, as sorted neighbour lists.
std::vector<std::vector<int>> FaceRegionAdjacency(const Mesh& mesh, const std::vector<int>& faceRegion,
                                                  int regionCount);

}  // namespace mopt
