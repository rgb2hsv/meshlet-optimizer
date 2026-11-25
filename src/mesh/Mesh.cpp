#include <core/Check.hpp>
#include <core/Types.hpp>
#include <mesh/Mesh.hpp>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mopt
{
using namespace std;

void Mesh::clear()
{
    vertexLocations.clear();
    vertexNormals.clear();
    triIndexes.clear();
    texcoords.clear();
}
void Mesh::updateVertexNormals(bool flipSign)
{
    vertexNormals.assign(vertexLocations.size(), Vector3f::Zero());
    for (const Vector3u& t : triIndexes) {
        const Vector3f v1 = vertexLocations[t.x()], v2 = vertexLocations[t.y()], v3 = vertexLocations[t.z()];
        const Vector3f norm = (v3 - v1).cross(v2 - v1);
        vertexNormals[t.x()] += norm;
        vertexNormals[t.y()] += norm;
        vertexNormals[t.z()] += norm;
    }
    const float sgn = flipSign ? -1.0f : 1.0f;
#pragma omp parallel for
    for (int n = 0; n < static_cast<int>(vertexNormals.size()); n++)
        vertexNormals[n] = sgn * vertexNormals[n].normalized();
}
namespace
{
size_t EstimatedHalfEdges(const Mesh& mesh)
{
    const size_t vertexCount = mesh.vertexLocations.size();
    if (vertexCount == 0) return 16;
    const size_t cornerCount = mesh.triIndexes.size() * 3;
    return 2 * std::max<size_t>(6, (cornerCount + vertexCount - 1) / vertexCount);
}

// For each corner, the two other corners in face order (previous, next).
void AppendTriangleHalfEdges(const Vector3u& face, std::vector<std::vector<uint32_t>>& rawNbrs)
{
    const auto append = [&](uint32_t vertex, uint32_t a, uint32_t b) {
        rawNbrs[vertex].push_back(a);
        rawNbrs[vertex].push_back(b);
    };
    append(face.x(), face.z(), face.y());
    append(face.y(), face.x(), face.z());
    append(face.z(), face.y(), face.x());
}

bool VertexNeighborRingIsManifold(int vertex, const std::vector<uint32_t>& nbrs)
{
    if (nbrs.empty()) {
        return true;
    }
    std::vector<uint32_t> sorted;
    sorted.reserve(nbrs.size());
    sorted = nbrs;
    std::sort(sorted.begin(), sorted.end());
    int count = 1;
    for (size_t i = 1; i < sorted.size(); ++i) {
        if (sorted[i] == sorted[i - 1]) {
            if (++count > 2) {
                return false;
            }
        } else {
            count = 1;
        }
    }
    return true;
}

void OrderVertexNeighborRing(int vertex, std::vector<uint32_t>& nbrs, std::vector<uint32_t>& out, bool leaveTail)
{
    out.clear();
    if (nbrs.empty()) {
        return;
    }
    out.reserve(nbrs.size());
    if (!VertexNeighborRingIsManifold(vertex, nbrs)) {
        return;
    }

    bool found;
    std::deque<uint32_t> chain;
    chain.push_back(nbrs[0]);
    chain.push_back(nbrs[1]);
    nbrs[0] = UINT32_MAX;
    nbrs[1] = UINT32_MAX;
    do {
        const uint32_t front = chain.front();
        const uint32_t back = chain.back();
        found = false;
        for (size_t i = 0; i + 1 < nbrs.size(); i += 2) {
            if (nbrs[i] == back) {
                chain.push_back(nbrs[i + 1]);
                nbrs[i + 1] = UINT32_MAX;
                found = true;
                break;
            }
        }

        if (!found) {
            for (size_t i = 1; i < nbrs.size(); i += 2) {
                if (nbrs[i] == front) {
                    chain.push_front(nbrs[i - 1]);
                    nbrs[i - 1] = UINT32_MAX;
                    found = true;
                    break;
                }
            }
        }
        if (!found) {
            if (!chain.empty()) {
                if (!leaveTail && chain.front() == chain.back()) {
                    chain.pop_back();
                }
                out.insert(out.end(), chain.begin(), chain.end());
                chain.clear();
            }
            for (size_t i = 0; i + 1 < nbrs.size(); i += 2) {
                if (nbrs[i] != UINT32_MAX && nbrs[i + 1] != UINT32_MAX) {
                    chain.push_back(nbrs[i]);
                    chain.push_back(nbrs[i + 1]);
                    found = true;
                    break;
                }
            }
        }
    } while (found);

    if (!chain.empty()) {
        if (!leaveTail && chain.front() == chain.back()) {
            chain.pop_back();
        }
        out.insert(out.end(), chain.begin(), chain.end());
    }
}

}  // namespace

void CreateOrderedVertexNeighborTable(const Mesh& mesh, MeshListNeighborTable& vertNbrsOut, bool leaveTail)
{
    // Leave tail means to not remove the duplicate vertex neighbor at the end of the neighbor list.
    // Non-manifold vertexes will not have a tail, so the tail can be used to detect them in simple (common) cases.
    const size_t vertexCount = mesh.vertexLocations.size();
    const size_t halfEdgeReserve = EstimatedHalfEdges(mesh);
    vertNbrsOut.assign(vertexCount, {});
    std::vector<std::vector<uint32_t>> vertNbrs(vertexCount);
    for (auto& nbrs : vertNbrs) nbrs.reserve(halfEdgeReserve);
    for (const Vector3u& face : mesh.triIndexes) AppendTriangleHalfEdges(face, vertNbrs);
    size_t nonManifoldCount = 0;
    for (size_t n = 0; n < vertexCount; n++) {
        if (!VertexNeighborRingIsManifold(static_cast<int>(n), vertNbrs[n])) {
            nonManifoldCount++;
            continue;
        }
        OrderVertexNeighborRing(static_cast<int>(n), vertNbrs[n], vertNbrsOut[n], leaveTail);
    }
    Expects(nonManifoldCount == 0, "{} non-manifold vertices", nonManifoldCount);
}

}  // namespace mopt
