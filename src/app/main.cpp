// meshlet-optimizer: PLY in, meshlet LOD hierarchy (.mlod for meshlet-viewer) out.
#include "Options.hpp"

#include <hierarchy/HierarchyBuilder.hpp>
#include <hierarchy/HierarchySteps.hpp>
#include <mesh/MeshProjector.hpp>
#include <mesh/PlyReader.hpp>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace mopt;

namespace
{
// The cereal portable binary archive starts with the writer's endianness byte, then the magic and version.
bool HasMlodHeader(const std::filesystem::path& file)
{
    std::ifstream in(file, std::ios::binary);
    char header[9] = {};
    if (!in.read(header, sizeof(header))) return false;
    return std::string(header + 1, 4) == "MLOD";
}
}  // namespace

int main(int argc, char** argv)
{
    Options options;
    if (std::optional<int> exitCode = ParseOptions(argc, argv, options)) return *exitCode;
    if (options.quiet) std::cout.rdbuf(nullptr);  // progress goes to std::cout; errors still reach std::cerr
#if defined(_OPENMP)
    if (options.threads > 0) omp_set_num_threads(options.threads);
    const int threads = omp_get_max_threads();
#else
    const int threads = 1;
#endif

    const auto start = std::chrono::steady_clock::now();
    Mesh mesh;
    try {
        ReadPlyMesh(options.input, mesh);
    } catch (const std::exception& e) {
        std::cerr << "Could not read " << options.input.string() << ": " << e.what() << "\n";
        return 1;
    }
    if (mesh.triIndexes.size() == 0) {
        std::cerr << options.input.string() << " has no triangles\n";
        return 1;
    }
    std::cout << "Loaded " << options.input.string() << ": " << mesh.vertexLocations.size() << " vertices, "
              << mesh.triIndexes.size() << " triangles; " << threads << " thread(s)" << '\n';

    HierarchyBuilder builder;
    builder.setMlodFile(options.output);
    builder.buildPipeline(options.hierarchy);

    // Resampled vertices are snapped back onto the input surface.
    std::unique_ptr<MeshProjector> surface;
    if (options.project) {
        surface = std::make_unique<MeshProjector>(mesh);
        builder.projectToSurface = [&surface](const Vector3f& p) { return surface->closestPoint(p); };
    }

    try {
        builder.build(mesh, [](const std::string& label, float progress) {
            std::cout << label << " (" << static_cast<int>(progress * 100.0f) << "%)" << '\n';
            return true;
        });
    } catch (const std::exception& e) {
        std::cerr << "Hierarchy build failed: " << e.what() << "\n";
        return 1;
    }
    builder.projectToSurface = nullptr;

    const HierarchyBuilder::ExportReport& report = builder.exportReport;
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (!report.written) {
        std::cerr << "Could not write " << options.output.string() << ": "
                  << (report.error.empty() ? "no levels were built" : report.error) << "\n";
        return 1;
    }
    if (!options.quiet) {
        std::cout << options.output.string() << ": " << report.stats.levels << " levels, " << report.stats.clusters
                  << " meshlets in " << report.stats.groups << " groups, " << report.stats.vertices << " vertices, "
                  << report.stats.triangles << " triangles over all levels; " << report.levelPairsWithCracks << " of "
                  << report.levelPairs << " level pairs crack; " << seconds << " s\n";
    }
    if (options.check) {
        if (!HasMlodHeader(options.output)) {
            std::cerr << "Check failed: " << options.output.string() << " is not an MLOD file\n";
            return 1;
        }
        if (report.stats.levels < 2 || report.stats.clusters == 0) {
            std::cerr << "Check failed: expected at least two levels and one meshlet\n";
            return 1;
        }
        if (report.levelPairsWithCracks > 0) {
            std::cerr << "Check failed: " << report.levelPairsWithCracks << " level pairs crack\n";
            return 1;
        }
    }
    return 0;
}
