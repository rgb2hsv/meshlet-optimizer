/*
 * RegionResampling.hpp
 *
 * Remesh flattened regions with fewer interior vertices, in stages:
 *
 *   RegionResampler(cutMesh, mosaics)
 *            Per region (a mosaic with UV coordinates, e.g. after LSCM): its
 *            boundary vertices -- endpoints of edges used by a single face of
 *            the region -- are kept exactly, in 2D and 3D.
 *   sampleBlueNoise()
 *            interiorFraction x (original interior count) new interior points
 *            as 2D blue noise (dart throwing on a hash grid that already holds
 *            the boundary vertices) over the region's UV triangles.
 *   relaxCvt()
 *            Lloyd (EM) iterations of a CVT restricted to the region: exact
 *            Voronoi cells clipped to the UV triangles, boundary vertices as
 *            fixed generators, interior points moved to their cell centroids
 *            while those stay inside the region.
 *   triangulate()
 *            2D constrained Delaunay triangulation of boundary + interior
 *            points, boundary edges as constraints, holes removed.
 *
 *            Fold-overs: when the regions are built, each region's UV layout
 *            is checked with winding numbers -- every face must keep the
 *            region's orientation and every face centroid must have winding
 *            number +1 (-1 if LSCM mirrored the region) with respect to the
 *            boundary loops. A folded region skips blue
 *            noise and CVT and is instead simplified in 3D by MeshDecimation
 *            down to the same interior count, its boundary vertices locked.
 *            So is any region whose triangulation does not reproduce its
 *            boundary.
 *   lift()   Interior points to 3D by barycentric interpolation in the UV
 *            triangle containing them, region by region. A region where any
 *            lifted triangle faces against the original surface under it (a
 *            fold-over in 3D) is decimated instead.
 *   project()
 *            New interior vertices moved onto the true surface by a
 *            projector (meshlet-optimizer: the closest point on the input).
 *   reconstitute()
 *            The regions welded back into one mesh along their shared
 *            boundary vertices.
 */
#pragma once

#include <core/Check.hpp>
#include <core/Types.hpp>
#include <flattening/Mosaic.hpp>
#include <mesh/Mesh.hpp>

#include <Eigen/Core>

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

namespace mopt
{

// Settings of the per-region remeshing: interior vertices kept, CVT iterations, blue-noise candidates, seed.
struct RegionResampleParams {
    // Interior vertices kept, as a fraction of each region's original count
    // (clamped to [0, 1]). Triangulation needs no interior vertices at all: a
    // region with B boundary and I interior vertices always triangulates into
    // B + 2I - 2 triangles, so any fraction down to 0 is valid. Decimation
    // (folded regions) may stop above the target when every remaining
    // collapse would fold or exceed the valence limit.
    double interiorFraction = 0.35;  // remove 65% of each region's interior per level
    // Lloyd (EM) iterations of the restricted 2D CVT.
    int lloydIterations = 10;
    // Dart-throwing candidates per requested point.
    int candidatesPerPoint = 30;
    uint32_t randomSeed = 1;
};

// Moves a point near the surface onto it.
using SurfaceProjector = std::function<Vector3f(const Vector3f&)>;

// The remeshed level: the welded mesh, the region of each face and vertex, and statistics.
struct ResampledSurface {
    // The welded mesh; mesh.texcoords holds the texture coordinates of every triangle corner when the input had
    // them.
    Mesh mesh;
    // Region (mosaic index) of each output triangle.
    std::vector<int> faceRegion;
    // Region of each interior output vertex; -1 for kept boundary vertices.
    std::vector<int> vertexRegion;
    // Normals were recomputed with flipped sign to match the input's.
    bool flipNormals = false;
};

// Remeshes the regions of a flattened level with fewer interior vertices, one stage per call (see the
// file header): blue noise, CVT, triangulation, lift, projection, reconstitution.
class RegionResampler
{
public:
    using V2 = Eigen::Vector2d;
    using V3d = Eigen::Vector3d;
    using Tri = std::array<int, 3>;

    RegionResampler(const Mesh& cutMesh, const std::vector<Mosaic>& mosaics, const RegionResampleParams& params);

    void sampleBlueNoise();
    void relaxCvt();
    void triangulate();
    void lift();
    void project(const SurfaceProjector& projector);
    ResampledSurface reconstitute() const;

    int regionCount() const { return static_cast<int>(regions_.size()); }
    // Interior points currently held across all regions (after sampling).
    int interiorPointCount() const;

    // One region being remeshed: its UV layout and boundary, the new points and triangles, and fallback flags.
    struct Region {
        // Local vertices of the original region.
        std::vector<V2> uv;
        std::vector<uint32_t> meshId;
        std::vector<Tri> faces;
        // Kept boundary: local vertex -> boundary index, and cut-mesh ids.
        std::vector<int> boundaryIndex;
        std::vector<uint32_t> boundary;
        std::vector<int> boundaryLocal;                  // boundary index -> local vertex
        std::vector<std::pair<int, int>> boundaryEdges;  // local vertex pairs
        bool mirrored = false;                           // LSCM mirrored the region
        int originalInterior = 0;
        // Boundary points first, then interior points (UV, z = 0).
        std::vector<V3d> seeds;
        // After triangulate(): corners >= 0 index `boundary`, corners < 0 are
        // -(i + 1) into interiorUv (or into the original interior vertices
        // when the region failed).
        std::vector<Tri> triangles;
        std::vector<V2> interiorUv;
        // Interior vertices in 3D: from lift(), or from decimation.
        std::vector<Vector3f> interiorPos;
        // Texture coordinates of the local vertices (from the cut mesh's corners: a region lies in one UV
        // island, so each vertex has one), and of the interior vertices. Empty without texture coordinates.
        std::vector<Vector2f> texcoord;
        std::vector<Vector2f> interiorTexcoord;
        bool folded = false;  // the UV layout folds over
        bool liftFolded = false;
        bool decimated = false;
        bool failed = false;
    };

private:
    // Winding-number fold-over test of the region's UV layout.
    bool foldsOver(const Region& region) const;
    // Does any lifted triangle face against the original face under it?
    bool liftFoldsOver(const Region& region) const;
    // MeshDecimation of the region in 3D with its boundary locked; false if
    // it could not be done.
    bool decimate(Region& region) const;
    void keepOriginal(Region& region) const;
    // Texture coordinates at p, from the closest point on the region's original triangles.
    Vector2f texcoordAt(const Region& region, const Vector3f& p) const;
    std::vector<Vector3f> positions_;
    std::vector<Vector3f> normals_;
    bool hasTexcoords_ = false;
    std::vector<Region> regions_;
    RegionResampleParams params_;
};

}  // namespace mopt
