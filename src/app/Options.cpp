#include "Options.hpp"

#include <argparse/argparse.hpp>

#include <iostream>

namespace mopt
{
namespace
{
constexpr const char* kVersion = "0.1.0";

// A boolean setting that defaults to on, turned off with --no-<name>.
void AddDisable(argparse::ArgumentParser& p, const std::string& name, const std::string& help)
{
    p.add_argument("--no-" + name).help(help).flag();
}
}  // namespace

std::optional<int> ParseOptions(int argc, const char* const* argv, Options& options)
{
    const MeshletParams d;  // defaults, shown in --help
    argparse::ArgumentParser p("meshlet-optimizer", kVersion);
    p.add_description(
        "Builds the chart hierarchy of a triangle mesh (CVT charts, then levels of clustered, flattened and "
        "remeshed regions) and writes it as a meshlet LOD file (.mlod) for meshlet-viewer.");

    p.add_argument("input").help("input mesh (.ply)");
    p.add_argument("-o", "--output").help("output .mlod file [default: the input's name with .mlod]");
    p.add_argument("-j", "--threads").help("OpenMP threads [default: all cores]").scan<'i', int>().default_value(0);
    p.add_argument("--no-project")
        .help("keep resampled vertices where they are instead of projecting them onto the input surface")
        .flag();
    p.add_argument("--check")
        .help("exit with an error unless the .mlod was written (and, with --check-cracks, no two levels crack)")
        .flag();
    p.add_argument("--check-cracks")
        .help("check every pair of levels for cracks before writing (slow with many levels)")
        .flag();
    p.add_argument("-q", "--quiet").help("only print errors").flag();
    p.add_argument("--save-levels")
        .help("debugging: also write every level as <dir>/level_<k>.ply, with its UVs and face regions")
        .metavar("DIR");

    p.add_group("Base charts (restricted CVT)");
    p.add_argument("--chart-triangles")
        .help("largest base chart, in triangles touching it")
        .scan<'i', int>()
        .default_value(d.cvt.regularization.maxTrianglesPerChart);
    p.add_argument("--cvt-lloyd")
        .help("Lloyd warm-up iterations")
        .scan<'i', int>()
        .default_value(d.cvt.lloydIterations);
    p.add_argument("--cvt-lbfgs")
        .help("L-BFGS iterations (0 skips)")
        .scan<'i', int>()
        .default_value(d.cvt.lbfgsIterations);
    p.add_argument("--min-chart-vertices")
        .help("smaller charts merge into a neighbour")
        .scan<'i', int>()
        .default_value(d.cvt.minChartVertices);
    p.add_argument("--cvt-seed")
        .help("seed of the blue-noise chart seeds")
        .scan<'u', uint32_t>()
        .default_value(d.cvt.randomSeed);
    AddDisable(p, "regularize", "skip opening, region competition and disk splitting of the base charts");

    p.add_group("Hierarchy");
    p.add_argument("--seed").help("seed of the region clustering").scan<'u', uint32_t>().default_value(d.randomSeed);
    p.add_argument("--max-clique")
        .help("most regions merged into one per level")
        .scan<'i', int>()
        .default_value(d.maxCliqueSize);
    p.add_argument("--stop-at-regions")
        .help("stop once the top level has at most this many regions")
        .scan<'i', int>()
        .default_value(d.stopAtRegions);
    p.add_argument("--max-depth")
        .help("most clustering passes (levels above the base)")
        .scan<'i', int>()
        .default_value(d.maxDepth);
    p.add_argument("--max-hole-fraction")
        .help("largest hole filled to keep a group a disk, as a fraction of the group")
        .scan<'g', double>()
        .default_value(d.maxHoleFraction);
    AddDisable(p, "absorb-singletons", "leave unmatched regions as their own nodes instead of joining a neighbour");
    AddDisable(p, "keep-disks", "do not make groups disk-shaped");
    AddDisable(p, "straighten-seams", "keep the base charts' borders as the CVT left them");

    p.add_group("Flattening");
    p.add_argument("--lscm-iterations")
        .help("conjugate-gradient iterations if the LSCM factorization fails")
        .scan<'i', int>()
        .default_value(d.lscmIterations);
    p.add_argument("--lscm-padding")
        .help("rings of neighbour triangles flattened with each region")
        .scan<'i', int>()
        .default_value(d.lscmPaddingRings);
    p.add_argument("--pack-padding")
        .help(
            "space around each chart in a generated UV atlas (input without UVs), as a fraction of the mean "
            "chart size")
        .scan<'g', double>()
        .default_value(d.packPadding);

    p.add_group("Remeshing");
    p.add_argument("--interior-fraction")
        .help("interior vertices kept per region and level")
        .scan<'g', double>()
        .default_value(d.resample.interiorFraction);
    p.add_argument("--resample-lloyd")
        .help("Lloyd iterations of each region's 2D CVT")
        .scan<'i', int>()
        .default_value(d.resample.lloydIterations);
    p.add_argument("--candidates-per-point")
        .help("dart-throwing candidates per new point")
        .scan<'i', int>()
        .default_value(d.resample.candidatesPerPoint);
    p.add_argument("--resample-seed")
        .help("seed of the blue-noise resampling")
        .scan<'u', uint32_t>()
        .default_value(d.resample.randomSeed);
    p.add_argument("--split-above-faces")
        .help("split remeshed regions with more triangles (0: never)")
        .scan<'i', int>()
        .default_value(d.splitAboveFaces);

    p.add_group("Region borders");
    p.add_argument("--border-target-scale")
        .help("border edge target, in mean interior edge lengths")
        .scan<'g', double>()
        .default_value(d.borders.targetScale);
    p.add_argument("--border-max-normal-change")
        .help("largest triangle turn allowed by a border collapse (degrees)")
        .scan<'g', double>()
        .default_value(d.borders.maxNormalChangeDegrees);
    p.add_argument("--flat-min-angle")
        .help("triangles with a smaller angle (and one over 120) are fixed (degrees)")
        .scan<'g', double>()
        .default_value(d.flatMinAngleDegrees);
    p.add_argument("--flat-max-normal-change")
        .help("largest triangle turn allowed when fixing flat triangles (degrees)")
        .scan<'g', double>()
        .default_value(d.flatMaxNormalChangeDegrees);
    p.add_argument("--border-lloyd-iterations")
        .help("L-BFGS-B iterations of the border relaxation")
        .scan<'i', int>()
        .default_value(d.borderLloydIterations);
    p.add_argument("--border-lloyd-max-normal-change")
        .help("largest triangle turn allowed by the relaxation (degrees)")
        .scan<'g', double>()
        .default_value(d.borderLloydMaxNormalChangeDegrees);
    AddDisable(
        p, "simplify-borders",
        "keep every level's region borders at full resolution (also skips the flat-triangle fix and relaxation)");
    AddDisable(p, "snap-finer-levels", "do not move finer levels' border vertices onto coarser borders");
    AddDisable(p, "relax-borders", "skip the border relaxation");
    AddDisable(p, "isometric-uvs", "keep the packed LSCM atlases instead of isometric per-triangle UVs");
    AddDisable(p, "uv-fit", "do not scale and move each relaxed chart to fit its atlas chart");
    AddDisable(p, "uv-overlap-springs", "skip the spring pass that pushes overlapping UV triangles apart");
    p.add_argument("--uv-iterations")
        .help("Gauss-Newton iterations of the UV relaxation (0: isometric triangles only)")
        .scan<'i', int>()
        .default_value(d.uvIterations);
    p.add_argument("--uv-area-weight")
        .help("weight of the equal-area term of the UV relaxation (0: rigid triangles)")
        .scan<'g', double>()
        .default_value(d.uvAreaWeight);

    try {
        p.parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n\n" << p;
        return 2;
    }

    Options& o = options;
    o.input = p.get<std::string>("input");
    if (auto out = p.present<std::string>("--output")) {
        o.output = *out;
    } else {
        o.output = o.input;
        o.output.replace_extension(".mlod");
    }
    o.threads = p.get<int>("--threads");
    o.project = !p.get<bool>("--no-project");
    o.check = p.get<bool>("--check");
    o.checkCracks = p.get<bool>("--check-cracks");
    o.quiet = p.get<bool>("--quiet");
    if (auto dir = p.present<std::string>("--save-levels")) o.saveLevels = *dir;

    MeshletParams& h = o.hierarchy;
    h.cvt.regularization.maxTrianglesPerChart = p.get<int>("--chart-triangles");
    h.cvt.lloydIterations = p.get<int>("--cvt-lloyd");
    h.cvt.lbfgsIterations = p.get<int>("--cvt-lbfgs");
    h.cvt.minChartVertices = p.get<int>("--min-chart-vertices");
    h.cvt.randomSeed = p.get<uint32_t>("--cvt-seed");
    h.cvt.regularize = !p.get<bool>("--no-regularize");

    h.randomSeed = p.get<uint32_t>("--seed");
    h.maxCliqueSize = p.get<int>("--max-clique");
    h.stopAtRegions = p.get<int>("--stop-at-regions");
    h.maxDepth = p.get<int>("--max-depth");
    h.maxHoleFraction = p.get<double>("--max-hole-fraction");
    h.absorbSingletons = !p.get<bool>("--no-absorb-singletons");
    h.keepDisks = !p.get<bool>("--no-keep-disks");
    h.straightenBaseSeams = !p.get<bool>("--no-straighten-seams");

    h.lscmIterations = p.get<int>("--lscm-iterations");
    h.lscmPaddingRings = p.get<int>("--lscm-padding");
    h.packPadding = p.get<double>("--pack-padding");

    h.resample.interiorFraction = p.get<double>("--interior-fraction");
    h.resample.lloydIterations = p.get<int>("--resample-lloyd");
    h.resample.candidatesPerPoint = p.get<int>("--candidates-per-point");
    h.resample.randomSeed = p.get<uint32_t>("--resample-seed");
    h.splitAboveFaces = p.get<int>("--split-above-faces");

    h.borders.targetScale = p.get<double>("--border-target-scale");
    h.borders.maxNormalChangeDegrees = p.get<double>("--border-max-normal-change");
    h.flatMinAngleDegrees = p.get<double>("--flat-min-angle");
    h.flatMaxNormalChangeDegrees = p.get<double>("--flat-max-normal-change");
    h.borderLloydIterations = p.get<int>("--border-lloyd-iterations");
    h.borderLloydMaxNormalChangeDegrees = p.get<double>("--border-lloyd-max-normal-change");
    h.simplifyBorders = !p.get<bool>("--no-simplify-borders");
    h.snapFinerLevels = !p.get<bool>("--no-snap-finer-levels");
    h.relaxBorders = !p.get<bool>("--no-relax-borders");
    h.isometricUvs = !p.get<bool>("--no-isometric-uvs");
    h.uvIterations = p.get<int>("--uv-iterations");
    h.uvAreaWeight = p.get<double>("--uv-area-weight");
    h.uvFitToAtlas = !p.get<bool>("--no-uv-fit");
    h.uvEnforceNoOverlaps = !p.get<bool>("--no-uv-overlap-springs");

    if (h.cvt.regularization.maxTrianglesPerChart < 0 || h.stopAtRegions < 1 || h.maxDepth < 0 || h.maxCliqueSize < 2 ||
        h.lscmIterations < 1 || h.lscmPaddingRings < 0 || !(h.packPadding >= 0.0) || h.splitAboveFaces < 0 ||
        !(h.resample.interiorFraction >= 0.0) || h.resample.interiorFraction > 1.0 || o.threads < 0) {
        std::cerr << "Invalid setting: --chart-triangles, --split-above-faces, --lscm-padding, --pack-padding and "
                     "--threads must be >= 0, "
                     "--stop-at-regions and --lscm-iterations >= 1, --max-clique >= 2, --interior-fraction in [0, 1]\n";
        return 2;
    }
    return std::nullopt;
}
}  // namespace mopt
