#include <core/Check.hpp>
#include <core/Types.hpp>
#include <iostream>
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
    quadIndexes.clear();
    triIndexes.clear();
    textureMap.clear();
}
void Mesh::updateVertexNormals(bool flipSign, int smoothIterations, float dotTolerance)
{
    uint32_t sz = (uint32_t)triIndexes.size();
    if (triIndexes.size() == 0 && quadIndexes.size() == 0) return;
    Vector3f pt;
    vertexNormals.clear();
    vertexNormals.resize(vertexLocations.size(), Vector3f::Zero());
    for (uint32_t i = 0; i < sz; i++) {
        Vector3u verts = triIndexes[i];
        Vector3f v1 = vertexLocations[verts.x()];
        Vector3f v2 = vertexLocations[verts.y()];
        Vector3f v3 = vertexLocations[verts.z()];
        Vector3f norm = ((v3 - v1)).cross((v2 - v1));
        vertexNormals[verts.x()] += norm;
        vertexNormals[verts.y()] += norm;
        vertexNormals[verts.z()] += norm;
    }
    sz = (uint32_t)quadIndexes.size();
    for (uint32_t i = 0; i < sz; i++) {
        Vector4u verts = quadIndexes[i];
        Vector3f v1 = vertexLocations[verts.x()];
        Vector3f v2 = vertexLocations[verts.y()];
        Vector3f v3 = vertexLocations[verts.z()];
        Vector3f v4 = vertexLocations[verts.w()];

        vertexNormals[verts.x()] += ((v4 - v1)).cross((v2 - v1));
        vertexNormals[verts.y()] += ((v1 - v2)).cross((v3 - v2));
        vertexNormals[verts.z()] += ((v2 - v3)).cross((v4 - v3));
        vertexNormals[verts.w()] += ((v3 - v4)).cross((v1 - v4));
    }
    float sgn = (flipSign) ? -1.0f : 1.0f;
#pragma omp parallel for
    for (int n = 0; n < (int)vertexNormals.size(); n++) {
        vertexNormals[n] = sgn * vertexNormals[n].normalized();
    }
    if (smoothIterations > 0) {
        int vertCount = (int)vertexLocations.size();
        std::vector<Vector3f> tmp(vertCount);
        std::vector<std::list<int>> vertNbrs(vertCount);
        int indexCount = (int)quadIndexes.size();
        for (int i = 0; i < indexCount; i++) {
            Vector4u verts = quadIndexes[i];
            int v1 = verts.x();
            int v2 = verts.y();
            int v3 = verts.z();
            int v4 = verts.w();

            vertNbrs[v1].push_back(v2);
            vertNbrs[v2].push_back(v3);
            vertNbrs[v3].push_back(v1);

            vertNbrs[v3].push_back(v4);
            vertNbrs[v4].push_back(v1);
            vertNbrs[v1].push_back(v3);
        }
        indexCount = (int)triIndexes.size();
        for (int i = 0; i < indexCount; i++) {
            Vector3u verts = triIndexes[i];
            int v1 = verts.x();
            int v2 = verts.y();
            int v3 = verts.z();
            vertNbrs[v1].push_back(v2);
            vertNbrs[v2].push_back(v3);
            vertNbrs[v3].push_back(v1);
        }
        for (int iter = 0; iter < smoothIterations; iter++) {
#pragma omp parallel for
            for (int i = 0; i < vertCount; i++) {
                Vector3f norm = vertexNormals[i];
                Vector3f avg = Vector3f::Zero();
                for (int nbr : vertNbrs[i]) {
                    Vector3f nnorm = vertexNormals[nbr];
                    ;
                    if (norm.dot(nnorm) > dotTolerance) {
                        avg += nnorm;
                    } else {
                        avg += norm;
                    }
                }
                tmp[i] = sgn * avg.normalized();
            }
            vertexNormals = tmp;
        }
    }
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
                std::cerr << vertex << ") Non-manifold found: " << sorted[i] << " nbrs:";
                for (uint32_t nbr : nbrs) std::cerr << " " << nbr;
                std::cerr << '\n';
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
    Expects(mesh.quadIndexes.empty(), "neighbour tables need a triangle mesh");
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

bool Mesh::convertQuadsToTriangles()
{
    bool ret = (quadIndexes.size() > 0);
    if (textureMap.size() > 0 && quadIndexes.size() > 0) {
        std::vector<Vector2f> newTextureMap;
        uint32_t index = 0;
        for (int n = 0; n < (int)triIndexes.size(); n++) {
            Vector2f tx = textureMap[index++];
            Vector2f ty = textureMap[index++];
            Vector2f tz = textureMap[index++];
            newTextureMap.push_back(tx);
            newTextureMap.push_back(ty);
            newTextureMap.push_back(tz);
        }
        for (const Vector4u& face : quadIndexes) {
            Vector3f pt1 = vertexLocations[face.x()];
            Vector3f pt2 = vertexLocations[face.y()];
            Vector3f pt3 = vertexLocations[face.z()];
            Vector3f pt4 = vertexLocations[face.w()];

            Vector2f tx = textureMap[index++];
            Vector2f ty = textureMap[index++];
            Vector2f tz = textureMap[index++];
            Vector2f tw = textureMap[index++];
            if ((pt1 - pt3).squaredNorm() < (pt2 - pt4).squaredNorm()) {
                triIndexes.push_back(Vector3u(face.x(), face.y(), face.z()));
                newTextureMap.push_back(tx);
                newTextureMap.push_back(ty);
                newTextureMap.push_back(tz);
                triIndexes.push_back(Vector3u(face.z(), face.w(), face.x()));
                newTextureMap.push_back(tz);
                newTextureMap.push_back(tw);
                newTextureMap.push_back(tx);
            } else {
                triIndexes.push_back(Vector3u(face.x(), face.y(), face.w()));
                newTextureMap.push_back(tx);
                newTextureMap.push_back(ty);
                newTextureMap.push_back(tw);
                triIndexes.push_back(Vector3u(face.w(), face.y(), face.z()));
                newTextureMap.push_back(tw);
                newTextureMap.push_back(ty);
                newTextureMap.push_back(tz);
            }
        }

        textureMap = newTextureMap;
    } else {
        for (const Vector4u& face : quadIndexes) {
            Vector3f pt1 = vertexLocations[face.x()];
            Vector3f pt2 = vertexLocations[face.y()];
            Vector3f pt3 = vertexLocations[face.z()];
            Vector3f pt4 = vertexLocations[face.w()];
            if ((pt1 - pt3).squaredNorm() < (pt2 - pt4).squaredNorm()) {
                triIndexes.push_back(Vector3u(face.x(), face.y(), face.z()));
                triIndexes.push_back(Vector3u(face.z(), face.w(), face.x()));
            } else {
                triIndexes.push_back(Vector3u(face.x(), face.y(), face.w()));
                triIndexes.push_back(Vector3u(face.w(), face.y(), face.z()));
            }
        }
    }
    if (quadIndexes.size() > 0) quadIndexes.clear();
    if (ret && vertexNormals.size() > 0) {
        updateVertexNormals();
    }
    return ret;
}
}  // namespace mopt
