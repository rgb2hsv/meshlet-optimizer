#pragma once
// Builds the level-of-detail hierarchy of a mesh: CVT charts as the base level, then levels of clustered,
// flattened and remeshed regions (HierarchySteps.hpp), written as one .mlod file (MlodWriter.hpp).
#include <core/Types.hpp>
#include <flattening/ChartLayout.hpp>
#include <hierarchy/LodLevel.hpp>
#include <hierarchy/Pipeline.hpp>
#include <hierarchy/RegionHierarchy.hpp>
#include <mlod/MlodWriter.hpp>

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mopt
{
struct LevelWork;
struct HierarchyParams;

// The base charts are a ChartLayout of the input mesh.
class HierarchyBuilder : public ChartLayout
{
public:
    HierarchyBuilder();
    ~HierarchyBuilder();

    // Replaces the pipeline with the hierarchy steps for these settings.
    void buildPipeline(const HierarchyParams& params);
    // Runs the pipeline on a copy of mesh. Returns the final chart count.
    int build(const Mesh& mesh, const StatusHandler& handler = nullptr);

    // Where ExportMlodStep writes the .mlod (nothing when empty).
    void setMlodFile(const std::filesystem::path& file) { mlodFile_ = file; }
    const std::filesystem::path& mlodFile() const { return mlodFile_; }

    // Clique-merge hierarchy of the base charts.
    RegionHierarchy regionHierarchy;
    // The levels built so far, base first.
    std::vector<LodLevel> levels;
    // Optional: moves a point near the surface onto it; resampled vertices are snapped with it.
    std::function<Vector3f(const Vector3f&)> projectToSurface;

    // Filled by ExportMlodStep.
    struct ExportReport {
        bool written = false;
        std::string error;
        MlodStats stats;
        int levelPairs = 0;
        int levelPairsWithCracks = 0;
    } exportReport;

    // Working state of the level being built (HierarchySteps).
    LevelWork& levelWork();

private:
    std::filesystem::path mlodFile_;
    Pipeline pipeline_;
    std::unique_ptr<LevelWork> levelWork_;
};
}  // namespace mopt
