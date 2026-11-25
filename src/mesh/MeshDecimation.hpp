/*
 * MeshDecimation.hpp
 *
 * Greedy edge-collapse decimation (ported from alloy). Each vertex caches its
 * cheapest outgoing edge, cost = edge length * curvature (how far the faces
 * around the vertex turn away from the faces on that edge), in a mutable
 * min-heap. Collapsing an edge places the merged vertex along the edge (see
 * CollapsePlacement), removes the two faces on the edge and hands the rest of
 * u's faces to v. A collapse is
 * rejected if the edge is not interior-manifold (exactly two faces), if
 * either end already has MAX_VALENCE neighbours, or if it would fold the
 * surface over.
 *
 * Fold-over test: the faces around the merged vertex form a fan whose outer
 * edges, the link, should be one loop around the new position. The link is
 * projected onto the plane of the old star's area-weighted normal and its
 * winding number about the projected new position is taken, as the sum of
 * the signed angles each link edge subtends. The fan is unfolded only if
 * every angle is positive (no face flips in the plane) and the winding number
 * is exactly 1 (the fan goes around once, not zero or two times). A link that
 * is not a single loop (u and v share a third neighbour, or the star is
 * non-manifold) is rejected; on a boundary the link is one open path, which
 * must turn through positive angles summing to less than a full turn.
 *
 * All per-element records (DecimationVertex, DecimationTriangle, ArenaList)
 * are plain index structs with trivial destructors. The variable-size
 * adjacency (neighbours and faces of each vertex) lives in one flat uint32_t
 * arena owned by the solver; a list that outgrows its slot is moved to the end
 * of the arena with double the capacity.
 */
#pragma once

#include <core/Check.hpp>
#include <core/MutableHeap.hpp>
#include <core/Types.hpp>
#include <mesh/Mesh.hpp>

#include <cstdint>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

namespace mopt
{

// A list of indexes stored in a shared arena: arena[offset .. offset+count).
struct ArenaList {
    uint32_t offset = 0;
    uint32_t count = 0;
    uint32_t capacity = 0;
};

// Per-vertex decimation state: neighbours, faces and the cheapest collapse.
struct DecimationVertex {
    static constexpr uint32_t kNone = 0xFFFFFFFFu;
    ArenaList neighbors;           // adjacent vertices
    ArenaList faces;               // adjacent triangles
    uint32_t collapse = kNone;     // cheapest neighbour to collapse onto
    uint32_t outputIndex = kNone;  // index in the decimated mesh
    float cost = 0.0f;             // cost of collapsing onto `collapse`
    bool valid = true;
};

// Per-triangle decimation state: corners and normal.
struct DecimationTriangle {
    uint32_t vertex[3] = {0, 0, 0};
    Vector3f normal = Vector3f::Zero();
    bool valid = true;
    bool hasVertex(uint32_t v) const { return v == vertex[0] || v == vertex[1] || v == vertex[2]; }
};

static_assert(std::is_trivially_destructible_v<ArenaList>);
static_assert(std::is_trivially_destructible_v<DecimationVertex>);
static_assert(std::is_trivially_destructible_v<DecimationTriangle>);

// Edge-collapse decimation with locked vertices: collapses the cheapest edges (fan quality, no fold-overs,
// bounded valence) until the requested fraction of vertices is gone.
class MeshDecimation
{
public:
    // Indexes of input vertices that are never removed or moved: an edge into
    // a locked vertex collapses onto it exactly, so e.g. a patch boundary is
    // kept vertex for vertex. Applies to the following solve() calls.
    void setLockedVertices(std::vector<uint32_t> lockedVertexIndexes)
    {
        lockedIndexes_ = std::move(lockedVertexIndexes);
    }
    // After solve(): output index of each input vertex, or UINT32_MAX if it
    // was removed.
    const std::vector<uint32_t>& outputIndexOf() const { return outputIndex_; }

    // Removes about decimationAmount (0..1, default half) of the mesh's
    // vertices. Quads are converted to triangles first. Vertex locations,
    // normals and colors are kept; returning false from the status handler
    // stops early and keeps what has been decimated so far.
    static constexpr float kDefaultDecimationAmount = 0.5f;
    void solve(Mesh& mesh, float decimationAmount = kDefaultDecimationAmount);

protected:
    static constexpr uint32_t kMaxValence = 9;
    // Brent's method for adaptive placement: tolerance in t and iteration cap.
    static constexpr float kPlacementTolerance = 1e-3f;
    static constexpr size_t kPlacementMaxIterations = 24;
    // How much better (in worst-triangle radius ratio) an interior position
    // must be to beat an endpoint, whose attributes are exact.
    static constexpr float kHalfEdgePreference = 0.02f;

    std::vector<uint32_t> lockedIndexes_;
    std::vector<char> locked_;  // per vertex, expanded from lockedIndexes in solve()
    std::vector<uint32_t> outputIndex_;
    bool isLocked(uint32_t v) const { return v < locked_.size() && locked_[v]; }

    std::vector<DecimationVertex> vertexes_;
    std::vector<DecimationTriangle> triangles_;
    std::vector<uint32_t> arena_;
    DenseMutableHeap<float> heap_;
    // Scratch copies of lists that are iterated while the arena is modified.
    std::vector<uint32_t> scratchVerts_, scratchFaces_;
    // The link of the fan a collapse would create, built by buildLink():
    // edges (from, to), their walk order, and the projection plane.
    std::vector<uint32_t> linkFrom_, linkTo_, linkOrder_, starFaces_;
    Vector3f linkNormal_, linkT1_, linkT2_;
    bool linkOpen_ = false;

    // Builds the adjacency for the mesh's triangles.
    void initialize(const Mesh& mesh);
    // Builds the link of the fan that collapsing uv would create. False if it
    // is not a single loop or path (non-manifold result).
    bool buildLink(const Mesh& mesh, uint32_t u, uint32_t v);
    // With the link built: does the fan, centred at `position`, fold over?
    bool fanFolds(const Mesh& mesh, const Vector3f& position) const;
    // With the link built: worst signed quality of the fan's triangles,
    // 2*sqrt(3) * (area vector . plane normal) / (sum of squared edges). It is
    // the radius ratio (1 equilateral, 0 degenerate) for a flat fan, shrinks
    // with tilt, and goes negative continuously as a triangle folds over.
    float fanQuality(const Mesh& mesh, const Vector3f& position) const;
    // Picks where along uv to put the merged vertex, as t in [0, 1] from v to
    // u. False if every candidate folds over (or the link is non-manifold).
    bool choosePlacement(const Mesh& mesh, uint32_t u, uint32_t v, float& t);
    // True if collapsing edge uv to `position` would fold the surface over
    // (see the winding-number test above). Reads but does not move u and v.
    bool foldsOver(const Mesh& mesh, uint32_t u, uint32_t v, const Vector3f& position)
    {
        return !buildLink(mesh, u, v) || fanFolds(mesh, position);
    }

    // Arena list operations.
    const uint32_t* begin(const ArenaList& l) const { return arena_.data() + l.offset; }
    const uint32_t* end(const ArenaList& l) const { return arena_.data() + l.offset + l.count; }
    bool contains(const ArenaList& l, uint32_t value) const;
    void insert(ArenaList& l, uint32_t value);  // no-op if already present
    void erase(ArenaList& l, uint32_t value);   // no-op if absent

    void setTriangle(uint32_t t, uint32_t v0, uint32_t v1, uint32_t v2);
    Vector3f computeNormal(const Mesh& mesh, uint32_t t) const;
    void linkTriangle(uint32_t t);
    void removeTriangle(uint32_t t);
    void replaceVertex(const Mesh& mesh, uint32_t t, uint32_t vold, uint32_t vnew);
    void removeIfNonNeighbor(uint32_t v, uint32_t n);
    void removeVertex(uint32_t v);

    size_t decimateInternal(Mesh& mesh, size_t validCount, size_t targetCount);
    bool collapseEdge(Mesh& mesh, uint32_t u, uint32_t v);
    void computeEdgeCostAtVertex(const Mesh& mesh, uint32_t v);
    float computeEdgeCollapseCost(const Mesh& mesh, uint32_t u, uint32_t v) const;
};

}  // namespace mopt
