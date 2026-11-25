#pragma once
// Command-line options of meshlet-optimizer.

#include <hierarchy/MeshletBuilder.hpp>

#include <filesystem>
#include <optional>
#include <string>

namespace mopt
{
// Everything the command line sets: files, run options and the hierarchy settings.
struct Options {
    std::filesystem::path input;   // .ply
    std::filesystem::path output;  // .mlod
    bool project = true;           // new vertices onto the input surface
    bool check = false;            // fail unless the file was written (and, with checkCracks, no levels crack)
    bool checkCracks = false;      // check every pair of levels for cracks before writing
    bool quiet = false;
    int threads = 0;                   // 0: OpenMP default
    std::filesystem::path saveLevels;  // debugging: every level as <dir>/level_<k>.ply, empty: off
    MeshletParams hierarchy;
};

// Parses argv into options. Returns an exit code to stop with (help, version or a usage error, already
// printed), or nothing to run.
std::optional<int> ParseOptions(int argc, const char* const* argv, Options& options);
}  // namespace mopt
