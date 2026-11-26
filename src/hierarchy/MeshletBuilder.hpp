// MeshletBuilder builds the level-of-detail hierarchy of a mesh. Its stages, in the order build() runs them:
//
//   buildBaseCharts()      CVT charts of the input mesh
//   buildBaseLevel()       the charts become level 0 (as sets of faces, so borders follow mesh edges)
//   straightenBaseSeams()  (straightenBaseSeams) borders between them straightened, junctions kept
//   repeated while the top level has more than stopAtRegions regions, at most maxDepth times:
//     clusterLevel()         clique-merge the top level into a new one, groups kept disk-shaped
//     cutLevel()             the previous level's mesh labelled by the new groups
//     flattenLevel()         all groups flattened together (padded mosaics, planar start, one LSCM solve)
//     sampleLevel()          new interior points per region, 2D blue noise
//     relaxLevel()           restricted 2D CVT (Lloyd) with the boundary fixed
//     triangulateLevel()     constrained Delaunay triangulation per region
//     liftLevel()            barycentric lift of the new points to 3D
//     projectLevel()         new interior vertices snapped onto the input surface
//     reconstituteLevel()    regions welded back into one mesh, stored as the new level
//     splitLevel()           regions over splitAboveFaces triangles split in two
//     simplifyLevelBorders() (simplifyBorders) the level's region borders coarsened, finer levels kept on them
//   fixFlatTriangles()     flat triangles left by snapping flipped or split
//   relaxBorders()         (relaxBorders) border vertices of every level slid to even out border edges
//   buildUvAtlases()       without input UVs: each level's regions flattened and packed into its own UV atlas
//   writeMlod()            every level written as one .mlod
//
// Each level starts from the previous level's mesh, so the surface gets coarser as the regions grow.
// Every border at every level is a chain of edges of the levels below, so levels mix without cracks.
#pragma once

#include <charts/CvtSegmentation.hpp>
#include <core/Types.hpp>
#include <flattening/ChartLayout.hpp>
#include <flattening/IsometricUvs.hpp>
#include <hierarchy/BorderSimplification.hpp>
#include <hierarchy/LodLevel.hpp>
#include <hierarchy/RegionHierarchy.hpp>
#include <hierarchy/RegionResampling.hpp>
#include <mlod/MlodWriter.hpp>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mopt
{
struct LevelWork;
struct LevelBordersState;

// Every setting of the build, with its default.
struct MeshletParams {
    // Base level: blue-noise seeded restricted CVT charts (buildBaseCharts()).
    CvtSegmentationParams cvt = [] {
        CvtSegmentationParams p;
        p.regularization.maxTrianglesPerChart = 256;
        return p;
    }();
    // Clustering of each level into the next (buildBaseLevel() / RegionHierarchy).
    uint32_t randomSeed = 3429841;
    int maxCliqueSize = 4;
    bool absorbSingletons = true;
    // straightenBaseSeams() after the base level.
    bool straightenBaseSeams = true;
    // Levels stop once the top level has at most stopAtRegions regions; at most maxDepth clustering passes.
    int stopAtRegions = 2;
    int maxDepth = 12;
    // clusterLevel() keeps groups disk-shaped, filling holes up to maxHoleFraction of a group.
    bool keepDisks = true;
    double maxHoleFraction = 0.5;
    // flattenLevel().
    int lscmIterations = 1024;
    int lscmPaddingRings = 1;
    // buildUvAtlases(): space around each packed chart, as a fraction of the mean chart size.
    double packPadding = 0.02;
    // Blue noise, CVT and remeshing of each region (sampleLevel() .. projectLevel()).
    RegionResampleParams resample;
    // splitLevel(): remeshed regions with more triangles are split in two; 0 turns splitting off.
    int splitAboveFaces = 256;
    // simplifyLevelBorders() at the end of each level, and fixFlatTriangles() after the levels.
    bool simplifyBorders = true;
    BorderSimplificationParams borders;
    bool snapFinerLevels = true;
    double flatMinAngleDegrees = 1.0;
    double flatMaxNormalChangeDegrees = 60.0;
    // relaxBorders() before export (with simplifyBorders).
    bool relaxBorders = true;
    int borderLloydIterations = 200;
    double borderLloydMaxNormalChangeDegrees = 60.0;
    // makeIsometricUvs() after the UV atlases: every triangle's UVs isometric to its 3D triangle, then the
    // triangles of each region moved rigidly to close the gaps between their shared corners (IsometricUvs.hpp).
    bool isometricUvs = true;
    int rigidUvIterations = 20;
};

// Builds the hierarchy and writes it as a .mlod. The base charts are a ChartLayout of the input mesh.
class MeshletBuilder : public ChartLayout
{
public:
    // Called after each stage with its name and duration.
    using StatusHandler = std::function<void(const std::string& stage, double seconds)>;

    explicit MeshletBuilder(const MeshletParams& params = {});
    ~MeshletBuilder();

    // Builds every level from a copy of mesh and writes the .mlod (see exportReport).
    void build(const Mesh& mesh, const StatusHandler& status = nullptr);

    // Where the .mlod is written (nothing when empty).
    void setMlodFile(const std::filesystem::path& file) { mlodFile_ = file; }
    const std::filesystem::path& mlodFile() const { return mlodFile_; }
    // Whether writeMlod() checks every pair of levels for cracks first (off by default: it is quadratic in
    // the level count and only verifies what the border stages already guarantee).
    void setCheckCracks(bool on) { checkCracks_ = on; }
    bool checkCracks() const { return checkCracks_; }

    // Clique-merge hierarchy of the base charts.
    RegionHierarchy regionHierarchy;
    // The levels built so far, base first.
    std::vector<LodLevel> levels;
    // Optional: moves a point near the surface onto it; resampled vertices are snapped with it.
    std::function<Vector3f(const Vector3f&)> projectToSurface;

    // Filled by writeMlod().
    struct ExportReport {
        bool written = false;
        std::string error;
        MlodStats stats;
        bool cracksChecked = false;
        int levelPairs = 0;
        int levelPairsWithCracks = 0;
    } exportReport;

    // Working state of the level being built.
    LevelWork& levelWork();

private:
    // Base charts: the cells of a restricted centroidal Voronoi tessellation seeded with blue noise
    // (CvtSegmentation.hpp), compact and of similar size.
    bool buildBaseCharts();

    // Level 0: the base charts become the regions of the input mesh (as sets of faces, so borders follow
    // mesh edges) and the region hierarchy is started from their adjacency.
    bool buildBaseLevel();

    // After buildBaseLevel(): the borders between base regions straightened
    // (StraightenRegionSeams: each border chain between two junctions replaced
    // by the shortest edge path between them; junctions stay). Every level
    // above inherits these borders, so levels still mix without cracks.
    bool straightenBaseSeams();

    // keepDisks: after clustering, make every group disk-shaped where possible
    // (MakeGroupsDisks: fill holes up to maxHoleFraction of the group's size,
    // then shed members), so LSCM and the fold-free fallback get disks.
    bool clusterLevel();

    // Starts a level: the previous level's mesh labelled by the new level's groups, as a ChartLayout.
    bool cutLevel();

    // paddingRings: each region is flattened together with that many rings of
    // its neighbours' triangles around its border, which are dropped again after
    // LSCM, so the region's own border triangles are not at the free boundary
    // where LSCM distorts most. The region keeps exactly its own triangles.
    bool flattenLevel();
    // Flattens every chart of the layout (padded mosaics, planar start, one LSCM solve, padding dropped,
    // UV area matched to 3D area); shared by flattenLevel() and buildUvAtlases().
    void flattenCharts(ChartLayout& layout) const;

    // New interior points for every region of the level: 2D blue noise in its UV layout.
    bool sampleLevel();

    // Relaxes each region's interior points to a restricted 2D CVT (Lloyd), boundary fixed.
    bool relaxLevel();

    // Constrained Delaunay triangulation of each region's boundary and interior points.
    bool triangulateLevel();

    // Lifts the new interior points to 3D by barycentric interpolation in the old UV triangles.
    bool liftLevel();

    // Snaps the new interior vertices onto the input surface (MeshletBuilder::projectToSurface).
    bool projectLevel();

    // Welds the remeshed regions back into one mesh along their shared borders and stores it as the new level.
    bool reconstituteLevel();

    // After reconstitution: each region of the new level with more than maxFaces
    // triangles is split in two (SplitLargeRegions). The filled half keeps the
    // node, the other becomes a new top node; the level's region labels and the
    // hierarchy's top adjacency and weights are updated.
    // A split cuts through the remeshed interior rather than along the
    // children's borders, so the two halves can only be swapped for their
    // children together (see RegionHierarchy::splitFrom).
    bool splitLevel();

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
    bool simplifyLevelBorders();

    // After the levels are built, before the border relaxation: snapping leaves
    // border vertices exactly on coarser edges, so three border vertices can be
    // in a line with a flat triangle between them. Each triangle whose smallest
    // angle is under minAngleDegrees and whose largest angle is over 120 degrees
    // (a cap) has its long edge flipped with the triangle across it when that
    // edge is inside a region and the two new triangles are better and do not
    // turn past maxNormalChangeDegrees; otherwise a vertex is added inside the
    // pair (projected onto the surface when builder.projectToSurface is set) and
    // the pair split into four. Region borders and vertex indices stay as they
    // are, so no level pair cracks. Updates simplifyLevelBorders()'s state.
    bool fixFlatTriangles();

    // After the levels are built, before export: every region border vertex of
    // every level slides along its border to even out the border edges, the way
    // Lloyd's algorithm relaxes CVT sites. A vertex and its copies on all levels
    // are one variable, its arclength along the border it lies on at the finest
    // level it exists on; it stays between the midpoints to its neighbours there.
    // Vertices that simplifyLevelBorders() attached to a coarser edge slide along it
    // (between their neighbours there) and follow its ends; junctions stay. The
    // interior vertices next to a border (sharing a triangle with a border vertex)
    // move too, in their tangent plane, at most half their shortest edge along
    // each tangent axis. The energy is the CVT energy of every triangle touching
    // a moving vertex on every level (each corner's barycentric share integrated
    // against its squared distance to the corner), each level scaled by its mean
    // border edge length to the fourth, minimized with L-BFGS-B (LBFGSpp, as the
    // CVT segmentation).
    // Moves that would flip a triangle or turn one past maxNormalChangeDegrees on
    // any level are halved, then undone. Uses simplifyLevelBorders()'s state.
    bool relaxBorders();

    // After the levels are built: writes all of them as one meshlet LOD file (builder.mlodFile(), MlodWriter.hpp)
    // for meshlet-viewer, after checking every pair of levels for cracks when builder.checkCracks().
    bool writeMlod();

    // When the input has no texture coordinates: each level's regions are flattened (flattenCharts()) and
    // packed (ChartBinPack) into the unit square, aspect ratios kept, and stored as the level's per-corner
    // UVs. Every level has its own atlas; atlases of different levels overlap in UV space.
    bool buildUvAtlases();

    // After buildUvAtlases() (isometricUvs): each level's per-corner UVs made isometric per triangle and
    // relaxed rigidly per region (MakeIsometricUvs). Triangles stop sharing UVs.
    bool makeIsometricUvs();

    MeshletParams params_;
    std::filesystem::path mlodFile_;
    bool checkCracks_ = false;
    std::unique_ptr<LevelWork> levelWork_;
    // UV island of every hierarchy node, per level (all 0 without texture coordinates): regions never merge
    // across islands, so texture coordinates stay continuous inside every region.
    std::vector<std::vector<int>> nodeIsland_;
    // Border vertices of the finished levels, shared by the border stages.
    std::shared_ptr<LevelBordersState> bordersState_;
};
}  // namespace mopt
