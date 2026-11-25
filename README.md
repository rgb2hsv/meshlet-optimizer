# meshlet-optimizer

Command-line tool that turns a triangle mesh (`.ply`) into a meshlet level-of-detail hierarchy saved as a
`.mlod` file for [meshlet-viewer](../meshlet-viewer). It is the mapping pipeline of wedge's toybox, stripped to
what the hierarchy build and the export need.

```
meshlet-optimizer assets/model/metasphere.ply -o metasphere.mlod
meshlet-viewer metasphere.mlod
```

## What it does

1. **Base charts.** Blue-noise seeds on the surface are relaxed to a restricted centroidal Voronoi
   tessellation (Lloyd, then L-BFGS). The cells are cleaned up into compact, disk-shaped charts of at most
   `--chart-triangles` triangles, which become level 0. Their borders are straightened along mesh edges.
2. **Levels.** Each level clique-merges the regions of the one below. It flattens every merged region
   (planar start, then one LSCM solve), samples fewer interior points as 2D blue noise, relaxes them with
   a CVT, re-triangulates, lifts to 3D and snaps them onto the input surface. Large regions are split, and
   region borders are coarsened while the finer levels' borders stay on them, so any two levels can be
   drawn side by side without cracks. This repeats until `--stop-at-regions` regions remain.
3. **UVs.** Texture coordinates of the input are carried through every level; regions never cross a UV
   seam. An input without UVs gets one atlas per level instead: the level's regions are flattened (LSCM)
   and bin-packed into the unit square at one common scale, so charts keep their shape and relative size.
   Atlases of different levels overlap in UV space.
4. **Export.** Every region of every level becomes a meshlet. Groups, bounds, errors and the LOD selection
   BVH are written in the viewer's `.mlod` layout (cereal portable binary, magic `MLOD`; version 2, which
   adds per-vertex texture coordinates, whenever the levels have UVs).

On `assets/model/metasphere.ply` (66K vertices, 132K triangles) this gives about 10 levels and
2,400 meshlets in about a minute on 20 threads.

## Building

Dependencies are git submodules under `external/`: Eigen, cereal, fmt, happly, argparse,
[libkdtree-cpp](https://github.com/rgb2hsv/kdtree-cpp) (BoxTree closest-point queries), CDT (constrained
Delaunay), LBFGSpp and meshoptimizer.

```
git submodule update --init
```

**Windows**: Visual Studio 2026 is the default preset; `vs2022` is there for machines without it.

```
cmake --preset vs2026
cmake --build --preset vs2026
ctest --preset vs2026
```

**macOS / Linux**: Ninja. Apple clang has no OpenMP runtime: install it with `brew install libomp` (found
automatically), or build single-threaded with `-DMOPT_OPENMP=OFF`.

```
cmake --preset ninja
cmake --build --preset ninja
ctest --preset ninja
```

The test runs the metasphere with `--check --check-cracks`, which fails unless the `.mlod` was written and no
pair of levels cracks.

## Options

`meshlet-optimizer --help` lists every option with its default. The groups are:

- **General**: `-o/--output`, `-j/--threads`, `-q/--quiet`, `--no-project` (leave new vertices in the
  flattened surface instead of snapping them onto the input), `--check` (fail unless the `.mlod` was written)
  and `--check-cracks` (check every pair of levels for cracks before writing; off by default, as its cost
  grows with the square of the level count), and `--save-levels DIR` (debugging: every level also written
  as `DIR/level_<k>.ply`, with its per-corner UVs and an int `region` per face).
- **Base charts**: chart size, CVT iterations, seed, label regularization.
- **Hierarchy**: clustering seed, clique size, when to stop, disk-shaped groups, seam straightening.
- **Flattening**: LSCM iterations and padding, and `--pack-padding` (space between charts in a generated
  UV atlas).
- **Remeshing**: interior vertices kept per level, CVT iterations, blue-noise candidates, splitting of
  large regions.
- **Region borders**: border coarsening, flat-triangle fix and border relaxation.

## Source layout

| Directory | Contents |
|---|---|
| `src/app` | Command line (`Options`) and `main` |
| `src/core` | Eigen type aliases, `Expects`/`Fail` (fmt), triangle geometry, Brent line search, mutable heap |
| `src/mesh` | `Mesh`, PLY reader and writer (happly), edge-collapse `MeshDecimation`, `MeshProjector` (closest point on the input) |
| `src/charts` | Base charts: `CvtSegmentation` and `LabelRegularization` |
| `src/flattening` | `ChartLayout`, `Mosaic`, and the flattening functions (mosaic build, planar start, LSCM), `ChartBinPack` (charts packed into the unit square) |
| `src/hierarchy` | `MeshletBuilder` (every stage of the build, in order), `RegionHierarchy`, remeshing, border simplification, crack check |
| `src/mlod` | `MlodWriter`: the levels as the viewer's meshlet DAG |
| `assets/model` | `metasphere.ply`, the test model |

## Conventions

- Types `PascalCase`; free functions `PascalCase`, methods `camelCase`; variables, parameters and public
  fields `camelCase`; private and protected members `camelCase_`; constants `kPascalCase`; macros
  `UPPER_CASE`. Files are named after the class or topic they hold. `.clang-tidy` checks the naming.
- Vectors and matrices are Eigen's (`Vector3f`, `Matrix4f`, `std::vector<Vector3u>` for storage); the
  build defines `EIGEN_INITIALIZE_MATRICES_BY_ZERO`.
- The library prints nothing: `MeshletBuilder::build` reports each stage (name and duration) to a callback,
  and the command line prints those lines. Broken invariants throw through `Expects`.
- Formatting is `.clang-format`, enforced by pre-commit:

  ```
  pip install pre-commit
  pre-commit install
  ```
