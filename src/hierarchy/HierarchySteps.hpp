// The steps that build the level-of-detail hierarchy, in the order AddHierarchySteps runs them:
//
//   BaseChartsStep         CVT charts of the input mesh
//   BaseLevelStep          the charts become level 0 (as sets of faces, so borders follow mesh edges)
//   BaseSeamsStep          (straightenBaseSeams) borders between them straightened, junctions kept
//   repeated while the top level has more than stopAtRegions regions, at most maxDepth times:
//     ClusterLevelStep       clique-merge the top level into a new one, groups kept disk-shaped
//     LevelCutStep           the previous level's mesh labelled by the new groups
//     LevelFlattenStep       all groups flattened together (padded mosaics, planar start, one LSCM solve)
//     LevelBlueNoiseStep     new interior points per region, 2D blue noise
//     LevelCvtStep           restricted 2D CVT (Lloyd) with the boundary fixed
//     LevelTriangulateStep   constrained Delaunay triangulation per region
//     LevelLiftStep          barycentric lift of the new points to 3D
//     LevelProjectStep       new interior vertices snapped onto the input surface
//     LevelReconstituteStep  regions welded back into one mesh, stored as the new level
//     LevelSplitStep         regions over splitAboveFaces triangles split in two
//     LevelBordersStep       (simplifyBorders) the level's region borders coarsened, finer levels kept on them
//   FixFlatTrianglesStep   flat triangles left by snapping flipped or split
//   BorderLloydStep        (relaxBorders) border vertices of every level slid to even out border edges
//   ExportMlodStep         every level written as one .mlod
//
// Each level starts from the previous level's mesh, so the surface gets coarser as the regions grow.
// Every border at every level is a chain of edges of the levels below, so levels mix without cracks.
#pragma once

#include <charts/CvtSegmentation.hpp>
#include <core/Check.hpp>
#include <core/Types.hpp>
#include <flattening/ChartLayout.hpp>
#include <hierarchy/BorderSimplification.hpp>
#include <hierarchy/Pipeline.hpp>
#include <hierarchy/RegionResampling.hpp>

#include <cstdint>
#include <memory>

namespace mopt
{

class HierarchyBuilder;

// Working state of the level being built, between the steps above.
struct LevelWork {
    int level = 0;
    // The previous level's simplified mesh, cut and flattened for this level.
    std::unique_ptr<ChartLayout> layout;
    std::unique_ptr<RegionResampler> resampler;
    ResampledSurface surface;
};

// Base charts: the cells of a restricted centroidal Voronoi tessellation seeded with blue noise
// (CvtSegmentation.hpp), compact and of similar size.
class BaseChartsStep : public PipelineStep
{
public:
    explicit BaseChartsStep(const CvtSegmentationParams& params = {}) : params_(params) {}
    const char* name() const override { return "BaseCharts"; }
    int run(HierarchyBuilder& builder, StatusHandler handler) override;

private:
    CvtSegmentationParams params_;
};

// Level 0: the base charts become the regions of the input mesh (as sets of faces, so borders follow
// mesh edges) and the region hierarchy is started from their adjacency.
class BaseLevelStep : public PipelineStep
{
public:
    explicit BaseLevelStep(uint32_t randomSeed = 1, int maxCliqueSize = 4, bool absorbSingletons = true)
        : randomSeed_(randomSeed), maxCliqueSize_(maxCliqueSize), absorbSingletons_(absorbSingletons)
    {
    }
    const char* name() const override { return "BaseLevel"; }
    int run(HierarchyBuilder& builder, StatusHandler handler) override;

private:
    uint32_t randomSeed_;
    int maxCliqueSize_;
    bool absorbSingletons_;
};

// After BaseLevelStep: the borders between base regions straightened
// (StraightenRegionSeams: each border chain between two junctions replaced
// by the shortest edge path between them; junctions stay). Every level
// above inherits these borders, so levels still mix without cracks.
class BaseSeamsStep : public PipelineStep
{
public:
    const char* name() const override { return "BaseSeams"; }
    int run(HierarchyBuilder& builder, StatusHandler handler) override;
};

// keepDisks: after clustering, make every group disk-shaped where possible
// (MakeGroupsDisks: fill holes up to maxHoleFraction of the group's size,
// then shed members), so LSCM and the fold-free fallback get disks.
class ClusterLevelStep : public PipelineStep
{
public:
    explicit ClusterLevelStep(bool keepDisks = true, double maxHoleFraction = 0.5)
        : keepDisks_(keepDisks), maxHoleFraction_(maxHoleFraction)
    {
    }
    const char* name() const override { return "ClusterLevel"; }
    int run(HierarchyBuilder& builder, StatusHandler handler) override;

private:
    bool keepDisks_;
    double maxHoleFraction_;
};

// Starts a level: the previous level's mesh labelled by the new level's groups, as a ChartLayout.
class LevelCutStep : public PipelineStep
{
public:
    const char* name() const override { return "LevelCut"; }
    int run(HierarchyBuilder& builder, StatusHandler handler) override;
};

// paddingRings: each region is flattened together with that many rings of
// its neighbours' triangles around its border, which are dropped again after
// LSCM, so the region's own border triangles are not at the free boundary
// where LSCM distorts most. The region keeps exactly its own triangles.
class LevelFlattenStep : public PipelineStep
{
public:
    explicit LevelFlattenStep(int iterations = 1024, int paddingRings = 1)
        : iterations_(iterations), paddingRings_(paddingRings)
    {
    }
    const char* name() const override { return "LevelLSCM"; }
    int run(HierarchyBuilder& builder, StatusHandler handler) override;

private:
    int iterations_;
    int paddingRings_;
};

// New interior points for every region of the level: 2D blue noise in its UV layout.
class LevelBlueNoiseStep : public PipelineStep
{
public:
    explicit LevelBlueNoiseStep(const RegionResampleParams& params = {}) : params_(params) {}
    const char* name() const override { return "LevelBlueNoise"; }
    int run(HierarchyBuilder& builder, StatusHandler handler) override;

private:
    RegionResampleParams params_;
};

// Relaxes each region's interior points to a restricted 2D CVT (Lloyd), boundary fixed.
class LevelCvtStep : public PipelineStep
{
public:
    const char* name() const override { return "LevelCVT"; }
    int run(HierarchyBuilder& builder, StatusHandler handler) override;
};

// Constrained Delaunay triangulation of each region's boundary and interior points.
class LevelTriangulateStep : public PipelineStep
{
public:
    const char* name() const override { return "LevelTriangulate"; }
    int run(HierarchyBuilder& builder, StatusHandler handler) override;
};

// Lifts the new interior points to 3D by barycentric interpolation in the old UV triangles.
class LevelLiftStep : public PipelineStep
{
public:
    const char* name() const override { return "LevelLift"; }
    int run(HierarchyBuilder& builder, StatusHandler handler) override;
};

// Snaps the new interior vertices onto the input surface (HierarchyBuilder::projectToSurface).
class LevelProjectStep : public PipelineStep
{
public:
    const char* name() const override { return "LevelProject"; }
    int run(HierarchyBuilder& builder, StatusHandler handler) override;
};

// Welds the remeshed regions back into one mesh along their shared borders and stores it as the new level.
class LevelReconstituteStep : public PipelineStep
{
public:
    const char* name() const override { return "LevelReconstitute"; }
    int run(HierarchyBuilder& builder, StatusHandler handler) override;
};

// After reconstitution: each region of the new level with more than maxFaces
// triangles is split in two (SplitLargeRegions). The filled half keeps the
// node, the other becomes a new top node; the level's region labels and the
// hierarchy's top adjacency and weights are updated.
// A split cuts through the remeshed interior rather than along the
// children's borders, so the two halves can only be swapped for their
// children together (see RegionHierarchy::splitFrom).
class LevelSplitStep : public PipelineStep
{
public:
    explicit LevelSplitStep(int maxFaces = 256) : maxFaces_(maxFaces) {}
    const char* name() const override { return "LevelSplit"; }
    int run(HierarchyBuilder& builder, StatusHandler handler) override;

private:
    int maxFaces_;
};

struct RegionHierarchy;

struct LevelBordersState;

// In the level loop, after LevelSplit: the region borders of the level just
// built are coarsened by fold-free edge collapses along them until their
// edges are about as long as the regions' interior edges
// (SimplifyRegionBorders), before the next level is cut from it; so the next
// level is remeshed inside the simplified borders and its borders are made of
// them. The base (level 0) is not simplified.
// snapFinerLevels: a vertex removed from this level's border is still on the
// borders of the levels under it (collapses keep positions, so its copies
// are at the same place); its copies, the base's included, move onto the
// edge that replaced it and stay attached to it: when an end of that edge
// moves later (removed by a coarser level), they follow, and so on down. A
// collapse whose moves would flip a finer triangle or turn one past
// maxNormalChangeDegrees is refused. Every finer border then lies on the
// coarser borders, and the coarser borders' vertices on the finer borders.
class LevelBordersStep : public PipelineStep
{
public:
    explicit LevelBordersStep(const BorderSimplificationParams& params = {}, bool snapFinerLevels = true,
                              std::shared_ptr<LevelBordersState> state = nullptr);
    const char* name() const override { return "LevelBorders"; }
    int run(HierarchyBuilder& builder, StatusHandler handler) override;

private:
    BorderSimplificationParams params_;
    bool snapFinerLevels_;
    std::shared_ptr<LevelBordersState> state_;  // across the levels of one build
};

// After the levels are built, before the border relaxation: snapping leaves
// border vertices exactly on coarser edges, so three border vertices can be
// in a line with a flat triangle between them. Each triangle whose smallest
// angle is under minAngleDegrees and whose largest angle is over 120 degrees
// (a cap) has its long edge flipped with the triangle across it when that
// edge is inside a region and the two new triangles are better and do not
// turn past maxNormalChangeDegrees; otherwise a vertex is added inside the
// pair (projected onto the surface when builder.projectToSurface is set) and
// the pair split into four. Region borders and vertex indices stay as they
// are, so no level pair cracks. Updates LevelBordersStep's state.
class FixFlatTrianglesStep : public PipelineStep
{
public:
    explicit FixFlatTrianglesStep(std::shared_ptr<LevelBordersState> state, double minAngleDegrees = 1.0,
                                  double maxNormalChangeDegrees = 60.0)
        : state_(std::move(state)), minAngleDegrees_(minAngleDegrees), maxNormalChangeDegrees_(maxNormalChangeDegrees)
    {
    }
    const char* name() const override { return "FixFlatTriangles"; }
    int run(HierarchyBuilder& builder, StatusHandler handler) override;

private:
    std::shared_ptr<LevelBordersState> state_;
    double minAngleDegrees_;
    double maxNormalChangeDegrees_;
};

// After the levels are built, before export: every region border vertex of
// every level slides along its border to even out the border edges, the way
// Lloyd's algorithm relaxes CVT sites. A vertex and its copies on all levels
// are one variable, its arclength along the border it lies on at the finest
// level it exists on; it stays between the midpoints to its neighbours there.
// Vertices that LevelBordersStep attached to a coarser edge slide along it
// (between their neighbours there) and follow its ends; junctions stay. The
// interior vertices next to a border (sharing a triangle with a border vertex)
// move too, in their tangent plane, at most half their shortest edge along
// each tangent axis. The energy is the CVT energy of every triangle touching
// a moving vertex on every level (each corner's barycentric share integrated
// against its squared distance to the corner), each level scaled by its mean
// border edge length to the fourth, minimized with L-BFGS-B (LBFGSpp, as the
// CVT segmentation).
// Moves that would flip a triangle or turn one past maxNormalChangeDegrees on
// any level are halved, then undone. Uses LevelBordersStep's state.
class BorderLloydStep : public PipelineStep
{
public:
    explicit BorderLloydStep(std::shared_ptr<LevelBordersState> state, int maxIterations = 200,
                             double maxNormalChangeDegrees = 60.0)
        : state_(std::move(state)), maxIterations_(maxIterations), maxNormalChangeDegrees_(maxNormalChangeDegrees)
    {
    }
    const char* name() const override { return "BorderLloyd"; }
    int run(HierarchyBuilder& builder, StatusHandler handler) override;

private:
    std::shared_ptr<LevelBordersState> state_;
    int maxIterations_;
    double maxNormalChangeDegrees_;
};

// After the levels are built: checks every pair of levels for cracks and writes all of them as one
// meshlet LOD file (builder.mlodFile(), MlodWriter.hpp) for meshlet-viewer.
class ExportMlodStep : public PipelineStep
{
public:
    const char* name() const override { return "ExportMlod"; }
    int run(HierarchyBuilder& builder, StatusHandler handler) override;
};

// Every setting of the hierarchy build (AddHierarchySteps), with its default.
struct HierarchyParams {
    // Base level: blue-noise seeded restricted CVT charts (BaseChartsStep).
    CvtSegmentationParams cvt = [] {
        CvtSegmentationParams p;
        p.regularization.maxTrianglesPerChart = 256;
        return p;
    }();
    // Clustering of each level into the next (BaseLevelStep / RegionHierarchy).
    uint32_t randomSeed = 3429841;
    int maxCliqueSize = 4;
    bool absorbSingletons = true;
    // BaseSeamsStep after the base level.
    bool straightenBaseSeams = true;
    // Levels stop once the top level has at most stopAtRegions regions; at most maxDepth clustering passes.
    int stopAtRegions = 2;
    int maxDepth = 12;
    // ClusterLevelStep keeps groups disk-shaped, filling holes up to maxHoleFraction of a group.
    bool keepDisks = true;
    double maxHoleFraction = 0.5;
    // LevelFlattenStep.
    int lscmIterations = 1024;
    int lscmPaddingRings = 1;
    // Blue noise, CVT and remeshing of each region (LevelBlueNoiseStep .. LevelProjectStep).
    RegionResampleParams resample;
    // LevelSplitStep: remeshed regions with more triangles are split in two; 0 turns splitting off.
    int splitAboveFaces = 256;
    // LevelBordersStep at the end of each level, and FixFlatTrianglesStep after the levels.
    bool simplifyBorders = true;
    BorderSimplificationParams borders;
    bool snapFinerLevels = true;
    double flatMinAngleDegrees = 1.0;
    double flatMaxNormalChangeDegrees = 60.0;
    // BorderLloydStep before export (with simplifyBorders).
    bool relaxBorders = true;
    int borderLloydIterations = 200;
    double borderLloydMaxNormalChangeDegrees = 60.0;
};

// BaseChartsStep and BaseLevelStep, then the repeated level steps, stopping once the top level has
// at most params.stopAtRegions regions, or a level is no coarser than the one below it, then the border
// fixes and ExportMlodStep.
// Every border at every level is a chain of original mesh edges, so levels can be mixed without cracks.
void AddHierarchySteps(Pipeline& pipeline, const HierarchyParams& params = {});

}  // namespace mopt
