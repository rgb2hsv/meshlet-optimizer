#pragma once
// Flattening the charts of a ChartLayout into UV mosaics, in the order the hierarchy runs them:
// BuildMosaics, DilateMosaics (padding), BuildMosaicNeighborTables, ProjectMosaicsToPlanes (initial
// layout), SolveLscm (one least-squares conformal solve over all mosaics), RescaleMosaics.
#include <flattening/ChartLayout.hpp>

namespace mopt
{
// One mosaic per chart. useFaceLabels: take layout.faceLabels as given instead of deriving them from the
// vertex labels (needed when a face can have all its vertices on a seam, as on a remeshed level).
void BuildMosaics(ChartLayout& layout, bool useFaceLabels);
// Pads every mosaic with `rings` rings of its neighbours' faces (appended after its own).
void DilateMosaics(ChartLayout& layout, int rings);
// Ordered neighbour rings of the layout vertices (layout.mosaicVertNbr).
void BuildMosaicNeighborTables(ChartLayout& layout, bool leaveTail = true);
// Each mosaic projected onto its best-fit plane: the starting UVs for LSCM.
void ProjectMosaicsToPlanes(ChartLayout& layout);
// Least-squares conformal maps of all mosaics in one sparse solve (BiCGSTAB, `iterations` at most).
void SolveLscm(ChartLayout& layout, int iterations);
// Scales each mosaic's UV layout so its area equals its 3D area.
void RescaleMosaics(ChartLayout& layout);
}  // namespace mopt
