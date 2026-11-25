#include <core/Check.hpp>
#include <core/LineSearch.hpp>
#include <core/Types.hpp>
#include <mesh/MeshDecimation.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace mopt
{

// ---------------------------------------------------------------------------
// Arena lists
// ---------------------------------------------------------------------------

bool MeshDecimation::contains(const ArenaList& l, uint32_t value) const
{
    return std::find(begin(l), end(l), value) != end(l);
}

void MeshDecimation::insert(ArenaList& l, uint32_t value)
{
    if (contains(l, value)) return;
    if (l.count == l.capacity) {
        // Move the list to the end of the arena with twice the room. The old
        // slot is abandoned; total waste is bounded by the live size.
        const uint32_t newCapacity = std::max<uint32_t>(4, l.capacity * 2);
        const uint32_t newOffset = static_cast<uint32_t>(arena_.size());
        arena_.resize(arena_.size() + newCapacity);
        std::copy_n(arena_.begin() + l.offset, l.count, arena_.begin() + newOffset);
        l.offset = newOffset;
        l.capacity = newCapacity;
    }
    arena_[l.offset + l.count++] = value;
}

void MeshDecimation::erase(ArenaList& l, uint32_t value)
{
    uint32_t* first = arena_.data() + l.offset;
    uint32_t* last = first + l.count;
    uint32_t* it = std::find(first, last, value);
    if (it == last) return;
    *it = *(last - 1);  // order does not matter
    l.count--;
}

// ---------------------------------------------------------------------------
// Topology
// ---------------------------------------------------------------------------

Vector3f MeshDecimation::computeNormal(const Mesh& mesh, uint32_t t) const
{
    const DecimationTriangle& tri = triangles_[t];
    const Vector3f v0 = mesh.vertexLocations[tri.vertex[0]];
    const Vector3f v1 = mesh.vertexLocations[tri.vertex[1]];
    const Vector3f v2 = mesh.vertexLocations[tri.vertex[2]];
    return ((v2 - v0).cross(v1 - v0)).normalized();
}

// Adds the triangle's three edges to the neighbour lists of its corners.
void MeshDecimation::linkTriangle(uint32_t t)
{
    const uint32_t* v = triangles_[t].vertex;
    for (int i = 0; i < 3; i++) {
        insert(vertexes_[v[i]].neighbors, v[(i + 1) % 3]);
        insert(vertexes_[v[i]].neighbors, v[(i + 2) % 3]);
    }
}

void MeshDecimation::setTriangle(uint32_t t, uint32_t v0, uint32_t v1, uint32_t v2)
{
    DecimationTriangle& tri = triangles_[t];
    tri.vertex[0] = v0;
    tri.vertex[1] = v1;
    tri.vertex[2] = v2;
    tri.valid = true;
    for (int i = 0; i < 3; i++) insert(vertexes_[tri.vertex[i]].faces, t);
    linkTriangle(t);
}

// Drops n from v's neighbours unless some face of v still uses edge (v, n).
void MeshDecimation::removeIfNonNeighbor(uint32_t v, uint32_t n)
{
    DecimationVertex& vert = vertexes_[v];
    if (!contains(vert.neighbors, n)) return;
    for (const uint32_t* f = begin(vert.faces); f != end(vert.faces); ++f) {
        if (triangles_[*f].hasVertex(n)) return;
    }
    erase(vert.neighbors, n);
}

void MeshDecimation::removeTriangle(uint32_t t)
{
    DecimationTriangle& tri = triangles_[t];
    if (!tri.valid) return;
    tri.valid = false;
    for (int i = 0; i < 3; i++) erase(vertexes_[tri.vertex[i]].faces, t);
    for (int i = 0; i < 3; i++) {
        const uint32_t a = tri.vertex[i], b = tri.vertex[(i + 1) % 3];
        removeIfNonNeighbor(a, b);
        removeIfNonNeighbor(b, a);
    }
}

void MeshDecimation::replaceVertex(const Mesh& mesh, uint32_t t, uint32_t vold, uint32_t vnew)
{
    DecimationTriangle& tri = triangles_[t];
    if (vold == tri.vertex[0]) {
        tri.vertex[0] = vnew;
    } else if (vold == tri.vertex[1]) {
        tri.vertex[1] = vnew;
    } else {
        tri.vertex[2] = vnew;
    }
    erase(vertexes_[vold].faces, t);
    insert(vertexes_[vnew].faces, t);
    for (int i = 0; i < 3; i++) {
        removeIfNonNeighbor(vold, tri.vertex[i]);
        removeIfNonNeighbor(tri.vertex[i], vold);
    }
    linkTriangle(t);
    tri.normal = computeNormal(mesh, t);
}

void MeshDecimation::removeVertex(uint32_t v)
{
    DecimationVertex& vert = vertexes_[v];
    if (!vert.valid) return;
    vert.valid = false;
    for (const uint32_t* n = begin(vert.neighbors); n != end(vert.neighbors); ++n) {
        erase(vertexes_[*n].neighbors, v);
    }
}

// ---------------------------------------------------------------------------
// Costs
// ---------------------------------------------------------------------------

// Error of collapsing u onto v: edge length times how far the faces around u
// turn away from the faces on the edge. Small, coplanar regions go first.
float MeshDecimation::computeEdgeCollapseCost(const Mesh& mesh, uint32_t u, uint32_t v) const
{
    const float edgeLength = (mesh.vertexLocations[v] - mesh.vertexLocations[u]).norm();
    const ArenaList& faces = vertexes_[u].faces;
    uint32_t sides[2];
    int sideCount = 0;
    for (const uint32_t* f = begin(faces); f != end(faces); ++f) {
        if (triangles_[*f].hasVertex(v)) {
            if (sideCount < 2) sides[sideCount] = *f;
            sideCount++;
        }
    }
    sideCount = std::min(sideCount, 2);
    float curvature = 0.0f;
    for (const uint32_t* f = begin(faces); f != end(faces); ++f) {
        float minCurv = 1.0f;
        for (int j = 0; j < sideCount; j++) {
            const float d = (triangles_[*f].normal).dot(triangles_[sides[j]].normal);
            minCurv = std::min(minCurv, (1.0f - d) * 0.5f);
        }
        curvature = std::max(curvature, minCurv);
    }
    return edgeLength * curvature;
}

// Caches the cheapest edge out of v and its cost.
void MeshDecimation::computeEdgeCostAtVertex(const Mesh& mesh, uint32_t v)
{
    DecimationVertex& vert = vertexes_[v];
    vert.collapse = DecimationVertex::kNone;
    if (vert.neighbors.count == 0) {
        vert.cost = -0.01f;  // isolated: free to remove
        return;
    }
    vert.cost = std::numeric_limits<float>::max();
    if (isLocked(v)) return;  // never collapses away
    for (const uint32_t* n = begin(vert.neighbors); n != end(vert.neighbors); ++n) {
        const float c = computeEdgeCollapseCost(mesh, v, *n);
        if (c < vert.cost) {
            vert.collapse = *n;
            vert.cost = c;
        }
    }
}

// ---------------------------------------------------------------------------
// Fold-over test
// ---------------------------------------------------------------------------

bool MeshDecimation::buildLink(const Mesh& mesh, uint32_t u, uint32_t v)
{
    const std::vector<Vector3f>& loc = mesh.vertexLocations;

    // The old star: faces of u and v (each once).
    starFaces_.assign(begin(vertexes_[u].faces), end(vertexes_[u].faces));
    for (const uint32_t* f = begin(vertexes_[v].faces); f != end(vertexes_[v].faces); ++f) {
        if (!triangles_[*f].hasVertex(u)) starFaces_.push_back(*f);
    }

    // Projection plane from the old star's area-weighted normal, and the link
    // of the new fan: for each surviving face (c, a, b), with c the merged
    // vertex, the edge a -> b. Faces are counter-clockwise about their area
    // vector, so the link of an unfolded fan turns counter-clockwise about N.
    Vector3f N = Vector3f::Zero();
    linkFrom_.clear();
    linkTo_.clear();
    linkOrder_.clear();
    for (uint32_t f : starFaces_) {
        const DecimationTriangle& tri = triangles_[f];
        const Vector3f x0 = loc[tri.vertex[0]], x1 = loc[tri.vertex[1]], x2 = loc[tri.vertex[2]];
        N += (x1 - x0).cross(x2 - x0);
        if (tri.hasVertex(u) && tri.hasVertex(v)) continue;  // removed by the collapse
        int k = 0;
        while (tri.vertex[k] != u && tri.vertex[k] != v) k++;
        linkFrom_.push_back(tri.vertex[(k + 1) % 3]);
        linkTo_.push_back(tri.vertex[(k + 2) % 3]);
    }
    const float normLength = N.norm();
    if (!(normLength > 0.0f)) return false;  // degenerate star
    linkNormal_ = N / normLength;
    const size_t n = linkFrom_.size();
    if (n == 0) return false;

    // A manifold link starts and ends each edge at a distinct vertex. A repeat
    // means u and v share a third neighbour, or the star is pinched.
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            if (linkFrom_[i] == linkFrom_[j] || linkTo_[i] == linkTo_[j]) return false;
        }
    }
    // An edge whose start no edge ends at begins an open (boundary) path.
    size_t start = 0;
    linkOpen_ = false;
    for (size_t i = 0; i < n && !linkOpen_; i++) {
        if (std::find(linkTo_.begin(), linkTo_.end(), linkFrom_[i]) == linkTo_.end()) {
            start = i;
            linkOpen_ = true;
        }
    }
    // Walk it; every edge must be reached from the start.
    size_t cur = start;
    do {
        linkOrder_.push_back(static_cast<uint32_t>(cur));
        const auto next = std::find(linkFrom_.begin(), linkFrom_.end(), linkTo_[cur]);
        if (next == linkFrom_.end()) break;  // end of an open path
        cur = static_cast<size_t>(next - linkFrom_.begin());
    } while (cur != start && linkOrder_.size() <= n);
    if (linkOrder_.size() != n) return false;  // more than one loop or path

    // In-plane basis.
    const Vector3f axis = std::abs(linkNormal_.x()) < 0.9f ? Vector3f(1.0f, 0.0f, 0.0f) : Vector3f(0.0f, 1.0f, 0.0f);
    linkT1_ = (linkNormal_.cross(axis)).normalized();
    linkT2_ = linkNormal_.cross(linkT1_);
    return true;
}

bool MeshDecimation::fanFolds(const Mesh& mesh, const Vector3f& position) const
{
    // Sum the signed angle each link edge subtends at the projected position.
    // Every angle must be positive: a negative one is a face that flipped in
    // the plane.
    const std::vector<Vector3f>& loc = mesh.vertexLocations;
    double totalAngle = 0.0;
    for (uint32_t e : linkOrder_) {
        const Vector3f da = loc[linkFrom_[e]] - position;
        const Vector3f db = loc[linkTo_[e]] - position;
        const float ax = da.dot(linkT1_), ay = da.dot(linkT2_);
        const float bx = db.dot(linkT1_), by = db.dot(linkT2_);
        const float sinPart = ax * by - ay * bx;
        const float cosPart = ax * bx + ay * by;
        const float scale = std::sqrt((ax * ax + ay * ay) * (bx * bx + by * by));
        if (!(sinPart > 1e-6f * scale)) return true;  // flipped or degenerate
        totalAngle += std::atan2((double)sinPart, (double)cosPart);
    }
    constexpr double kTwoPi = 6.283185307179586;
    if (linkOpen_) return !(totalAngle < kTwoPi);
    return std::lround(totalAngle / kTwoPi) != 1;
}

float MeshDecimation::fanQuality(const Mesh& mesh, const Vector3f& position) const
{
    // Fan triangle (c, a, b) has area vector (a - c).cross(b - c), which points
    // along the plane normal while the triangle is unfolded.
    const std::vector<Vector3f>& loc = mesh.vertexLocations;
    constexpr float kK = 3.4641016f;  // 2 * sqrt(3)
    float worst = 1.0f;
    for (uint32_t e : linkOrder_) {
        const Vector3f a = loc[linkFrom_[e]] - position;
        const Vector3f b = loc[linkTo_[e]] - position;
        const float sumSq = a.dot(a) + b.dot(b) + (a - b).dot(a - b);
        const float q = sumSq > 0.0f ? kK * (a.cross(b)).dot(linkNormal_) / sumSq : 0.0f;
        worst = std::min(worst, q);
    }
    return worst;
}

bool MeshDecimation::choosePlacement(const Mesh& mesh, uint32_t u, uint32_t v, float& t)
{
    if (!buildLink(mesh, u, v)) return false;
    const Vector3f U = mesh.vertexLocations[u];
    const Vector3f V = mesh.vertexLocations[v];
    auto at = [&](float s) { return V + s * (U - V); };
    if (isLocked(v)) {
        // The merged vertex stays exactly where the locked one is.
        t = 0.0f;
        return !fanFolds(mesh, V);
    }
    // Endpoints first; an interior sample must beat the best by a margin.
    float bestQuality = -1.0f;
    for (float s : {0.0f, 1.0f}) {
        const Vector3f p = at(s);
        if (fanFolds(mesh, p)) continue;
        const float q = fanQuality(mesh, p);
        if (q > bestQuality) {
            bestQuality = q;
            t = s;
        }
    }
    // Interior optimum: maximize worst signed quality with Brent's method. The
    // objective is continuous through fold-overs (it goes negative), so the
    // parabolic steps steer away from them; the winding test confirms.
    float interiorT = 0.5f;
    SolveBrentLineSearch<float>(
        interiorT, 0.0f, 1.0f, [&](const float& s) { return -(double)fanQuality(mesh, at(s)); }, kPlacementTolerance,
        kPlacementMaxIterations);
    const Vector3f p = at(interiorT);
    if (!fanFolds(mesh, p)) {
        const float q = fanQuality(mesh, p);
        const float margin = bestQuality < 0.0f ? 0.0f : kHalfEdgePreference;
        if (q > bestQuality + margin) {
            bestQuality = q;
            t = interiorT;
        }
    }
    return bestQuality >= 0.0f;
}

// ---------------------------------------------------------------------------
// Collapse
// ---------------------------------------------------------------------------

bool MeshDecimation::collapseEdge(Mesh& mesh, uint32_t u, uint32_t v)
{
    if (isLocked(u)) return false;
    if (v == DecimationVertex::kNone) {
        removeVertex(u);  // isolated vertex
        return true;
    }
    // Only interior-manifold edges between low-valence vertices.
    uint32_t edgeFaces = 0;
    for (const uint32_t* f = begin(vertexes_[u].faces); f != end(vertexes_[u].faces); ++f) {
        if (triangles_[*f].hasVertex(v)) edgeFaces++;
    }
    if (edgeFaces != 2 || vertexes_[u].neighbors.count >= kMaxValence || vertexes_[v].neighbors.count >= kMaxValence) {
        return false;
    }

    // Place the merged vertex along the edge; give up if every placement folds.
    float t = 0.0f;
    if (!choosePlacement(mesh, u, v, t)) return false;
    Vector3f& U = mesh.vertexLocations[u];
    Vector3f& V = mesh.vertexLocations[v];
    V = V + t * (U - V);
    U = V;
    // Attributes at the merged vertex: exact at an endpoint, blended inside.
    if (t > 0.0f) {
        const size_t n = vertexes_.size();
        if (mesh.vertexNormals.size() == n) {
            Vector3f& N = mesh.vertexNormals[v];
            const Vector3f blended = N + t * (mesh.vertexNormals[u] - N);
            const float len = blended.norm();
            N = len > 0.0f ? blended / len : mesh.vertexNormals[t < 0.5f ? v : u];
        }
    }
    for (const uint32_t* f = begin(vertexes_[v].faces); f != end(vertexes_[v].faces); ++f) {
        triangles_[*f].normal = computeNormal(mesh, *f);
    }

    // Vertices whose cached cost may change: neighbours of both ends.
    scratchVerts_.assign(begin(vertexes_[u].neighbors), end(vertexes_[u].neighbors));
    scratchVerts_.insert(scratchVerts_.end(), begin(vertexes_[v].neighbors), end(vertexes_[v].neighbors));

    // Remove the faces on edge uv, then hand u's remaining faces to v.
    scratchFaces_.assign(begin(vertexes_[u].faces), end(vertexes_[u].faces));
    for (uint32_t f : scratchFaces_) {
        if (triangles_[f].hasVertex(v)) removeTriangle(f);
    }
    scratchFaces_.assign(begin(vertexes_[u].faces), end(vertexes_[u].faces));
    for (uint32_t f : scratchFaces_) replaceVertex(mesh, f, u, v);
    removeVertex(u);
    heap_.erase(u);

    std::sort(scratchVerts_.begin(), scratchVerts_.end());
    scratchVerts_.erase(std::unique(scratchVerts_.begin(), scratchVerts_.end()), scratchVerts_.end());
    for (uint32_t w : scratchVerts_) {
        if (w == u || !vertexes_[w].valid) continue;
        computeEdgeCostAtVertex(mesh, w);
        heap_.exchange(w, vertexes_[w].cost);  // no-op if already popped
    }
    return true;
}

size_t MeshDecimation::decimateInternal(Mesh& mesh, size_t validCount, size_t targetCount,
                                        const StatusHandler& statusHandler, bool& cancelled)
{
    const uint32_t vertexCount = static_cast<uint32_t>(vertexes_.size());
    heap_.clear();
    heap_.reserve(vertexCount);
    heap_.reserveKeys(vertexCount);
    for (uint32_t i = 0; i < vertexCount; i++) {
        if (vertexes_[i].valid) {
            computeEdgeCostAtVertex(mesh, i);
            heap_.push(i, vertexes_[i].cost);
        } else {
            vertexes_[i].collapse = DecimationVertex::kNone;
        }
    }
    size_t removeCount = vertexCount - validCount;
    size_t iteration = 0;
    while (!heap_.empty() && removeCount < targetCount) {
        if (statusHandler && (iteration++ & 1023) == 0 &&
            !statusHandler("Decimate", removeCount / (float)targetCount)) {
            cancelled = true;
            break;
        }
        const uint32_t u = heap_.top()->key;
        heap_.pop();
        if (vertexes_[u].valid && collapseEdge(mesh, u, vertexes_[u].collapse)) {
            removeCount++;
        }
    }
    heap_.clear();
    return removeCount;
}

// ---------------------------------------------------------------------------
// Solve
// ---------------------------------------------------------------------------

void MeshDecimation::initialize(const Mesh& mesh)
{
    const size_t vertexCount = mesh.vertexLocations.size();
    const size_t triCount = mesh.triIndexes.size();
    vertexes_.assign(vertexCount, DecimationVertex{});
    triangles_.assign(triCount, DecimationTriangle{});
    arena_.clear();
    arena_.reserve(vertexCount * 16);
    for (size_t i = 0; i < triCount; i++) {
        const Vector3u tri = mesh.triIndexes[i];
        setTriangle(static_cast<uint32_t>(i), tri.x(), tri.y(), tri.z());
        triangles_[i].normal = computeNormal(mesh, static_cast<uint32_t>(i));
    }
}

void MeshDecimation::solve(Mesh& mesh, float decimationAmount, const StatusHandler& statusHandler)
{
    outputIndex_.resize(mesh.vertexLocations.size());
    std::iota(outputIndex_.begin(), outputIndex_.end(), 0u);  // unchanged unless decimated
    locked_.assign(mesh.vertexLocations.size(), 0);
    for (uint32_t v : lockedIndexes_) {
        if (v < locked_.size()) locked_[v] = 1;
    }
    if (decimationAmount <= 0.0f) return;
    if (mesh.quadIndexes.size() > 0) mesh.convertQuadsToTriangles();
    if (mesh.triIndexes.size() == 0) return;
    initialize(mesh);
    const size_t vertexCount = mesh.vertexLocations.size();

    // Each pass recomputes all costs; repeat while passes still make progress.
    const size_t targetRemoveCount = static_cast<size_t>(std::min(decimationAmount, 1.0f) * vertexCount);
    size_t validCount = vertexCount;
    size_t lastRemoveCount = std::numeric_limits<size_t>::max();
    bool cancelled = false;
    while (vertexCount - validCount < targetRemoveCount && !cancelled) {
        decimateInternal(mesh, validCount, targetRemoveCount, statusHandler, cancelled);
        validCount = 0;
        for (const DecimationVertex& v : vertexes_) {
            if (v.valid) validCount++;
        }
        const size_t removeCount = vertexCount - validCount;
        if (removeCount == lastRemoveCount) break;
        lastRemoveCount = removeCount;
    }

    // Compact the surviving vertices and faces.
    const bool hasNormals = mesh.vertexNormals.size() == vertexCount;
    const std::vector<Vector3f> locations = mesh.vertexLocations;
    const std::vector<Vector3f> normals = hasNormals ? mesh.vertexNormals : std::vector<Vector3f>();
    mesh.vertexLocations.clear();
    mesh.vertexNormals.clear();
    for (size_t i = 0; i < vertexCount; i++) {
        DecimationVertex& vert = vertexes_[i];
        if (!vert.valid || vert.neighbors.count == 0) continue;
        vert.outputIndex = static_cast<uint32_t>(mesh.vertexLocations.size());
        mesh.vertexLocations.push_back(locations[i]);
        if (hasNormals) mesh.vertexNormals.push_back(normals[i]);
    }
    outputIndex_.assign(vertexCount, std::numeric_limits<uint32_t>::max());
    for (size_t i = 0; i < vertexCount; i++) {
        if (vertexes_[i].valid && vertexes_[i].neighbors.count > 0) outputIndex_[i] = vertexes_[i].outputIndex;
    }
    mesh.triIndexes.clear();
    for (const DecimationTriangle& tri : triangles_) {
        if (!tri.valid) continue;
        mesh.triIndexes.push_back(Vector3u(vertexes_[tri.vertex[0]].outputIndex, vertexes_[tri.vertex[1]].outputIndex,
                                           vertexes_[tri.vertex[2]].outputIndex));
    }
    // Per-corner texture coordinates no longer match the faces.
    mesh.textureMap.clear();

    vertexes_.clear();
    triangles_.clear();
    arena_.clear();
    scratchVerts_.clear();
    scratchFaces_.clear();
}

}  // namespace mopt
