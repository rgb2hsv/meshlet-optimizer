#include <flattening/ChartFlattening.hpp>

#include <core/Check.hpp>
#include <core/TriangleGeometry.hpp>

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <algorithm>
#include <limits>
#include <list>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace mopt
{
namespace
{
// Quaternions are (x, y, z, w) vectors.
Vector4f QuaternionProduct(const Vector4f& a, const Vector4f& b)
{
    return Vector4f(a.x() * b.w() + a.w() * b.x() + a.y() * b.z() - a.z() * b.y(),
                    a.y() * b.w() + a.w() * b.y() + a.z() * b.x() - a.x() * b.z(),
                    a.z() * b.w() + a.w() * b.z() + a.x() * b.y() - a.y() * b.x(),
                    a.w() * b.w() - a.x() * b.x() - a.y() * b.y() - a.z() * b.z());
}

// Images of the x, y and z axes under the rotation q.
Vector3f RotatedX(const Vector4f& q)
{
    return Vector3f(q.w() * q.w() + q.x() * q.x() - q.y() * q.y() - q.z() * q.z(),
                    (q.x() * q.y() + q.z() * q.w()) * 2.0f, (q.z() * q.x() - q.y() * q.w()) * 2.0f);
}
Vector3f RotatedY(const Vector4f& q)
{
    return Vector3f((q.x() * q.y() - q.z() * q.w()) * 2.0f,
                    q.w() * q.w() - q.x() * q.x() + q.y() * q.y() - q.z() * q.z(),
                    (q.y() * q.z() + q.x() * q.w()) * 2.0f);
}
Vector3f RotatedZ(const Vector4f& q)
{
    return Vector3f((q.z() * q.x() + q.y() * q.w()) * 2.0f, (q.y() * q.z() - q.x() * q.w()) * 2.0f,
                    q.w() * q.w() - q.x() * q.x() - q.y() * q.y() + q.z() * q.z());
}

Matrix3f QuaternionMatrix(const Vector4f& q)
{
    Matrix3f m;
    m.col(0) = RotatedX(q);
    m.col(1) = RotatedY(q);
    m.col(2) = RotatedZ(q);
    return m;
}

// Jacobi diagonalization of a symmetric 3x3 matrix by quaternion rotations: Q^T A Q is diagonal for
// Q = quaternionMatrix(result). With sort, the eigenvalues come out in decreasing order and Q is a
// right-handed frame.
Vector4f Diagonalizer(const Matrix3f& A, bool sort = true)
{
    constexpr int kMaxsteps = 24;
    Vector4f q(0, 0, 0, 1);
    for (int i = 0; i < kMaxsteps; i++) {
        const Matrix3f Q = QuaternionMatrix(q);
        const Matrix3f D = Q.transpose() * A * Q;
        const Vector3f offdiag(D(2, 1), D(2, 0), D(1, 0));
        const Vector3f om(std::fabs(offdiag.x()), std::fabs(offdiag.y()), std::fabs(offdiag.z()));
        const int k = (om.x() > om.y() && om.x() > om.z()) ? 0 : (om.y() > om.z()) ? 1 : 2;
        const int k1 = (k + 1) % 3;
        const int k2 = (k + 2) % 3;
        if (offdiag[k] == 0.0f) break;
        float thet = (D(k2, k2) - D(k1, k1)) / (2.0f * offdiag[k]);
        const float sgn = (thet > 0.0f) ? 1.0f : -1.0f;
        thet *= sgn;
        const float t = sgn / (thet + ((thet < 1.E6f) ? std::sqrt(thet * thet + 1.0f) : thet));
        const float c = 1.0f / std::sqrt(t * t + 1.0f);
        if (c == 1.0f) break;
        Vector4f jr(0, 0, 0, 0);
        jr[k] = sgn * std::sqrt((1.0f - c) / 2.0f);
        jr[k] *= -1.0f;
        jr.w() = std::sqrt(1.0f - (jr[k] * jr[k]));
        if (jr.w() == 1.0f) break;
        q = QuaternionProduct(q, jr);
        q = q.normalized();
    }
    if (sort) {
        const float h = 1.0f / std::sqrt(2.0f);
        const auto e = [&q, &A]() -> Vector3f {
            const Matrix3f Q = QuaternionMatrix(q);
            return (Q.transpose() * A * Q).diagonal();
        };
        q = (e().x() < e().z()) ? QuaternionProduct(q, Vector4f(0, h, 0, h)) : q;
        q = (e().y() < e().z()) ? QuaternionProduct(q, Vector4f(h, 0, 0, h)) : q;
        q = (e().x() < e().y()) ? QuaternionProduct(q, Vector4f(0, 0, h, h)) : q;
        q = (RotatedZ(q).z() < 0) ? QuaternionProduct(q, Vector4f(1, 0, 0, 0)) : q;
        q = (RotatedY(q).y() < 0) ? QuaternionProduct(q, Vector4f(0, 0, 1, 0)) : q;
        q = (q.w() < 0) ? Vector4f(-q) : q;
    }
    return q;
}

void EigenSolve(const Matrix3f& A, Matrix3f& Q, Matrix3f& D, bool sort = true)
{
    Q = QuaternionMatrix(Diagonalizer(A, sort));
    D = Q.transpose() * A * Q;
}

bool FindAndRemoveEdge(const uint32_t& pivotVert, std::vector<Vector2u>& edges, uint32_t& otherVert)
{
    otherVert = uint32_t(-1);
    for (auto iter = edges.begin(); iter != edges.end(); iter++) {
        const Vector2u e = *iter;
        if (e.x() == pivotVert) {
            otherVert = e.y();
            edges.erase(iter);
            return true;
        }
        if (e.y() == pivotVert) {
            otherVert = e.x();
            edges.erase(iter);
            return true;
        }
    }
    return false;
}

void Repin(ChartLayout& ctx, bool left = true, bool right = true, bool top = true, bool bottom = true)
{
    ctx.pinned.clear();
    for (Mosaic& mosaic : ctx.mosaics) {
        int uMinIdx = -1;
        int uMaxIdx = -1;
        int vMinIdx = -1;
        int vMaxIdx = -1;
        float minu = std::numeric_limits<float>::max();
        float maxu = std::numeric_limits<float>::lowest();
        float minv = std::numeric_limits<float>::max();
        float maxv = std::numeric_limits<float>::lowest();
        int idx = 0;
        for (MosaicVertex& mv : mosaic.vertexes) {
            mv.locked = false;
            if (mv.uv.x() < minu) {
                uMinIdx = idx;
                minu = mv.uv.x();
            }
            if (mv.uv.x() > maxu) {
                uMaxIdx = idx;
                maxu = mv.uv.x();
            }
            if (mv.uv.y() < minv) {
                vMinIdx = idx;
                minv = mv.uv.y();
            }
            if (mv.uv.y() > maxv) {
                vMaxIdx = idx;
                maxv = mv.uv.y();
            }
            idx++;
        }
        if (left && uMinIdx >= 0) {
            mosaic.vertexes[uMinIdx].locked = true;
            ctx.pinned.insert(mosaic.vertexes[uMinIdx].mapId);
        }
        if (right && uMaxIdx >= 0) {
            mosaic.vertexes[uMaxIdx].locked = true;
            ctx.pinned.insert(mosaic.vertexes[uMaxIdx].mapId);
        }
        if (top && vMinIdx >= 0) {
            mosaic.vertexes[vMinIdx].locked = true;
            ctx.pinned.insert(mosaic.vertexes[vMinIdx].mapId);
        }
        if (bottom && vMaxIdx >= 0) {
            mosaic.vertexes[vMaxIdx].locked = true;
            ctx.pinned.insert(mosaic.vertexes[vMaxIdx].mapId);
        }
    }
}

Matrix4f FitPlane(const Mesh& mesh, const std::vector<MosaicVertex>& verts, Vector3f* deviations)
{
    Matrix3f AtA = Matrix3f::Zero();
    Vector3f centerPoint = Vector3f::Zero();
    for (const auto& vert : verts) {
        centerPoint += mesh.vertexLocations[vert.meshId];
    }
    const int N = static_cast<int>(verts.size());
    centerPoint = centerPoint * (1.0f / N);
    for (const auto& vert : verts) {
        Vector3f v = mesh.vertexLocations[vert.meshId] - centerPoint;
        AtA(0, 0) += v.x() * v.x();
        AtA(1, 0) += v.x() * v.y();
        AtA(2, 0) += v.x() * v.z();
        AtA(0, 1) += v.y() * v.x();
        AtA(1, 1) += v.y() * v.y();
        AtA(2, 1) += v.y() * v.z();
        AtA(0, 2) += v.z() * v.x();
        AtA(1, 2) += v.z() * v.y();
        AtA(2, 2) += v.z() * v.z();
    }
    AtA = (1.0f / N) * AtA;
    Matrix3f Q, D;
    EigenSolve(AtA, Q, D);
    *deviations = Vector3f(static_cast<float>(std::sqrt(D(2, 2))), static_cast<float>(std::sqrt(D(1, 1))),
                           static_cast<float>(std::sqrt(D(0, 0))));
    const Vector3f zaxis = Q.col(2);
    const Vector3f xaxis = Q.col(0);
    const Vector3f yaxis = zaxis.cross(xaxis);
    Matrix4f pose = Matrix4f::Identity();
    pose(0, 0) = xaxis.x();
    pose(1, 0) = xaxis.y();
    pose(2, 0) = xaxis.z();
    pose(0, 1) = yaxis.x();
    pose(1, 1) = yaxis.y();
    pose(2, 1) = yaxis.z();
    pose(0, 2) = zaxis.x();
    pose(1, 2) = zaxis.y();
    pose(2, 2) = zaxis.z();
    pose(0, 3) = centerPoint.x();
    pose(1, 3) = centerPoint.y();
    pose(2, 3) = centerPoint.z();
    pose(3, 3) = 1.0f;
    Matrix4f poseInv = pose.inverse();
    float dotSum = 0.0f;
    for (const auto& vert : verts) {
        Vector3f norm = mesh.vertexNormals[vert.meshId];
        Vector4f n4 = poseInv * Vector4f(norm.x(), norm.y(), norm.z(), 0.0f);
        dotSum += (n4.z() > 0.0f) ? 1.0f : -1.0f;
    }
    if (dotSum < 0.0f) {
        // Half a turn about x.
        const float c = std::cos(static_cast<float>(M_PI)), sn = std::sin(static_cast<float>(M_PI));
        Matrix4f flip = Matrix4f::Identity();
        flip(1, 1) = c;
        flip(1, 2) = -sn;
        flip(2, 1) = sn;
        flip(2, 2) = c;
        poseInv = flip * poseInv;
    }
    return poseInv;
}

void ProjectTriangle(const Vector3f& p0, const Vector3f& p1, const Vector3f& p2, Vector2d& z0, Vector2d& z1,
                     Vector2d& z2)
{
    const Vector3d d1 = (p1 - p0).cast<double>();
    const Vector3d x = d1.normalized();
    const Vector3d d2 = (p2 - p0).cast<double>();
    const Vector3d z = x.cross(d2);
    const Vector3d y = (z.cross(x)).normalized();
    z0 = Vector2d(0.0, 0.0);
    z1 = Vector2d(d1.dot(x), d1.dot(y));
    z2 = Vector2d(d2.dot(x), d2.dot(y));
}

void CreateOrderedVertexNeighborTable(ChartLayout& ctx, bool /*leaveTail*/)
{
    ctx.mosaicVertNbr.resize(ctx.textureToMeshMap.size(), std::vector<uint32_t>());
    std::vector<std::vector<Vector2u>> vertNbrs(ctx.textureToMeshMap.size());
    for (const Mosaic& mosaic : ctx.mosaics) {
        for (const MosaicFace& mface : mosaic.faces) {
            const Vector3u& face = mface.mapFace;
            vertNbrs[face.x()].push_back(Vector2u{face.z(), face.y()});
            vertNbrs[face.y()].push_back(Vector2u{face.x(), face.z()});
            vertNbrs[face.z()].push_back(Vector2u{face.y(), face.x()});
        }
    }
    const size_t N = vertNbrs.size();
    for (size_t n = 0; n < N; n++) {
        std::vector<Vector2u>& nbrs = vertNbrs[n];
        std::vector<uint32_t> chain;
        while (!nbrs.empty()) {
            const Vector2u edge = nbrs.back();
            chain.push_back(edge.x());
            chain.push_back(edge.y());
            nbrs.pop_back();
            bool found;
            do {
                found = false;
                uint32_t otherVert;
                if (FindAndRemoveEdge(chain.back(), nbrs, otherVert)) {
                    chain.push_back(otherVert);
                    found = true;
                }
                if (FindAndRemoveEdge(chain.front(), nbrs, otherVert)) {
                    chain.insert(chain.begin(), otherVert);
                    found = true;
                }
            } while (found);
            if (!chain.empty()) {
                ctx.mosaicVertNbr[n].insert(ctx.mosaicVertNbr[n].end(), chain.begin(), chain.end());
                chain.clear();
            }
        }
    }
}
}  // namespace

void BuildMosaics(ChartLayout& layout, bool useFaceLabels)
{
    const int chartCount = layout.chartCount;
    std::vector<Vector3u>& faceArray = layout.mesh.triIndexes;
    const size_t faceCount = faceArray.size();

    if (useFaceLabels) {
        Expects(layout.faceLabels.size() == faceCount, "Face labels {} for {} faces", layout.faceLabels.size(),
                faceCount);
    }
    layout.faceLabels.resize(faceCount, -1);
    size_t idx = 0;
    for (size_t count = 0; count < faceCount && !useFaceLabels; count++) {
        Vector3u tri = faceArray[count];
        int l1 = layout.vertLabels[tri.x()];
        int l2 = layout.vertLabels[tri.y()];
        int l3 = layout.vertLabels[tri.z()];
        if (l1 == l2 && l2 == l3 && l3 == l1) {
            layout.faceLabels[idx] = l1;
        } else if (l1 >= 0 && l2 < 0 && l3 < 0) {
            layout.faceLabels[idx] = l1;
        } else if (l2 >= 0 && l1 < 0 && l3 < 0) {
            layout.faceLabels[idx] = l2;
        } else if (l3 >= 0 && l1 < 0 && l2 < 0) {
            layout.faceLabels[idx] = l3;
        } else if (l1 >= 0 && l2 >= 0 && l3 < 0) {
            layout.faceLabels[idx] = l1;
        } else if (l2 >= 0 && l3 >= 0 && l1 < 0) {
            layout.faceLabels[idx] = l2;
        } else if (l3 >= 0 && l2 >= 0 && l1 < 0) {
            layout.faceLabels[idx] = l3;
        } else {
        }
        idx++;
    }

    layout.mosaics.resize(chartCount, {});
    layout.totalTriangleCount = 0;
    layout.totalVertexCount = 0;
    layout.textureToMeshMap.clear();

    std::vector<std::unordered_map<uint32_t, uint32_t>> vertexSet(layout.mosaics.size());
    for (size_t count = 0; count < layout.mosaics.size(); count++) {
        layout.mosaics[count].id = static_cast<int>(count);
    }

    std::vector<Vector2i> faceLabelList(faceCount);
    for (size_t count = 0; count < faceCount; count++) {
        int label = layout.faceLabels[count];
        Expects(label >= 0, "Face label {}", label);
        faceLabelList[count] = Vector2i(label, static_cast<int>(count));
    }
    std::sort(faceLabelList.begin(), faceLabelList.end(),
              [](const Vector2i& a, const Vector2i& b) { return a.x() < b.x() || (a.x() == b.x() && a.y() < b.y()); });

    for (size_t count = 0; count < faceCount; count++) {
        Vector2i pr = faceLabelList[count];
        int lid = pr.x();
        int fid = pr.y();
        auto& meshVertMap = vertexSet[lid];
        auto& mosaic = layout.mosaics[lid];
        Vector3u tri = faceArray[fid];
        Vector3u localTri;
        for (int i = 0; i < 3; i++) {
            uint32_t vid = tri[i];
            uint32_t nvid;
            if (auto pos = meshVertMap.find(vid); pos == meshVertMap.end()) {
                meshVertMap[vid] = nvid = static_cast<uint32_t>(layout.textureToMeshMap.size());
                layout.textureToMeshMap.push_back(vid);
                mosaic.vertexMap[nvid] = mosaic.vertexes.size();
                mosaic.vertexes.emplace_back(vid, nvid);
            } else {
                nvid = pos->second;
            }
            localTri[i] = nvid;
        }
        mosaic.faces.emplace_back(layout.totalTriangleCount++, fid, tri, localTri);
    }

    std::erase_if(layout.mosaics, [](const auto& mosaic) { return mosaic.vertexes.empty(); });
    layout.totalVertexCount = layout.textureToMeshMap.size();
}

void DilateMosaics(ChartLayout& layout, int n)
{
    if (n <= 0 || layout.mosaics.empty()) {
        return;
    }
    const auto& faceArray = layout.mesh.triIndexes;
    // Faces incident to each layout.mesh vertex (CSR), so each mosaic only looks at
    // faces around its dilated vertices instead of the whole layout.mesh.
    const size_t meshVertexCount = layout.mesh.vertexLocations.size();
    std::vector<uint32_t> faceOffsets(meshVertexCount + 1, 0);
    for (const Vector3u& tri : faceArray) {
        for (int i = 0; i < 3; i++) faceOffsets[tri[i] + 1]++;
    }
    for (size_t v = 0; v < meshVertexCount; v++) faceOffsets[v + 1] += faceOffsets[v];
    std::vector<uint32_t> incidentFaces(faceOffsets.back());
    {
        std::vector<uint32_t> fill(faceOffsets.begin(), faceOffsets.end() - 1);
        for (size_t fid = 0; fid < faceArray.size(); fid++) {
            for (int i = 0; i < 3; i++) incidentFaces[fill[faceArray[fid][i]]++] = static_cast<uint32_t>(fid);
        }
    }
    std::vector<uint32_t> candidates;
    for (Mosaic& mosaic : layout.mosaics) {
        std::unordered_set<uint32_t> vertSet;
        vertSet.reserve(mosaic.vertexes.size());
        for (const MosaicVertex& v : mosaic.vertexes) {
            vertSet.insert(v.meshId);
        }
        std::vector<uint32_t> frontier(vertSet.begin(), vertSet.end());
        for (int iter = 0; iter < n; iter++) {
            std::vector<uint32_t> nextFrontier;
            for (uint32_t vid : frontier) {
                Expects(vid < layout.vertNbrs.size(), "Vertex id must be {} < {}", vid, layout.vertNbrs.size());
                for (int nbr : layout.vertNbrs[vid]) {
                    if (vertSet.insert(static_cast<uint32_t>(nbr)).second) {
                        nextFrontier.push_back(static_cast<uint32_t>(nbr));
                    }
                }
            }
            frontier = std::move(nextFrontier);
            if (frontier.empty()) {
                break;
            }
        }
        std::unordered_set<size_t> existingFaces;
        existingFaces.reserve(mosaic.faces.size());
        for (const MosaicFace& face : mosaic.faces) {
            existingFaces.insert(face.fid);
        }
        std::unordered_map<uint32_t, uint32_t> meshVertMap;
        meshVertMap.reserve(mosaic.vertexes.size());
        for (const MosaicVertex& v : mosaic.vertexes) {
            meshVertMap.emplace(v.meshId, v.mapId);
        }
        const auto findOrCreateMapId = [&](uint32_t meshId) -> uint32_t {
            if (auto pos = meshVertMap.find(meshId); pos != meshVertMap.end()) {
                return pos->second;
            }
            const uint32_t mapId = static_cast<uint32_t>(layout.textureToMeshMap.size());
            layout.textureToMeshMap.push_back(meshId);
            layout.totalVertexCount++;
            mosaic.vertexMap[mapId] = mosaic.vertexes.size();
            mosaic.vertexes.emplace_back(meshId, mapId);
            meshVertMap.emplace(meshId, mapId);
            return mapId;
        };
        // Same faces, same ascending order as a scan over the whole layout.mesh.
        candidates.clear();
        for (uint32_t vid : vertSet) {
            if (vid >= meshVertexCount) continue;
            for (uint32_t k = faceOffsets[vid]; k < faceOffsets[vid + 1]; k++) {
                candidates.push_back(incidentFaces[k]);
            }
        }
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
        for (const size_t fid : candidates) {
            if (existingFaces.contains(fid)) {
                continue;
            }
            const Vector3u& tri = faceArray[fid];
            if (!vertSet.contains(tri.x()) || !vertSet.contains(tri.y()) || !vertSet.contains(tri.z())) {
                continue;
            }
            Vector3u localTri;
            for (int i = 0; i < 3; i++) {
                localTri[i] = findOrCreateMapId(tri[i]);
            }
            mosaic.faces.emplace_back(layout.totalTriangleCount++, fid, tri, localTri);
        }
    }
}

void BuildMosaicNeighborTables(ChartLayout& layout, bool leaveTail)
{
    CreateOrderedVertexNeighborTable(layout, leaveTail);
}

void ProjectMosaicsToPlanes(ChartLayout& layout)
{
    Vector3f deviations;
    layout.totalBoundaryCount = 0;
    layout.boundaryMap.resize(layout.totalVertexCount, false);
    std::vector<uint32_t> lookup(layout.totalVertexCount, std::numeric_limits<uint32_t>::max());
    int extraBoundaryLoops = 0;
    for (Mosaic& mosaic : layout.mosaics) {
        Matrix4f pose = FitPlane(layout.mesh, mosaic.vertexes, &deviations);
        mosaic.pose = pose;
        Mesh chartMesh;
        for (MosaicVertex& mv : mosaic.vertexes) {
            const Vector3f pt = layout.mesh.vertexLocations[mv.meshId];
            const Vector3f qt = (pose * pt.homogeneous()).hnormalized();
            lookup[mv.mapId] = static_cast<uint32_t>(chartMesh.vertexLocations.size());
            chartMesh.vertexLocations.push_back(Vector3f(qt.x(), qt.y(), 0.0f));
        }
        for (MosaicFace& mf : mosaic.faces) {
            const Vector3u tri = mf.mapFace;
            chartMesh.triIndexes.push_back(Vector3u(lookup[tri.x()], lookup[tri.y()], lookup[tri.z()]));
        }
        chartMesh.vertexNormals.resize(chartMesh.vertexLocations.size(), Vector3f(0, 0, -1));
        size_t idx = 0;
        for (MosaicVertex& mv : mosaic.vertexes) {
            const Vector3f qt = chartMesh.vertexLocations[idx++];
            mv.uv = Vector2d(qt.x(), qt.y());
            const auto& nbrs = layout.mosaicVertNbr[mv.mapId];
            if (nbrs.empty() || nbrs.front() != nbrs.back()) {
                mv.boundary = true;
                layout.boundaryMap[mv.mapId] = true;
                layout.totalBoundaryCount++;
            }
        }
        mosaic.edges.clear();
        // A boundary edge is used by a single face of the mosaic. Two boundary
        // endpoints are not enough: in small charts an interior edge (a chord)
        // often joins two boundary vertices and would break the loop below.
        const auto edgeKey = [](uint32_t s, uint32_t e) { return (static_cast<uint64_t>(s) << 32) | e; };
        std::unordered_set<uint64_t> directedEdges;
        directedEdges.reserve(mosaic.faces.size() * 3);
        for (const MosaicFace& mf : mosaic.faces) {
            const auto& f = mf.mapFace;
            for (int j = 0; j < 3; ++j) directedEdges.insert(edgeKey(f[j], f[(j + 1) % 3]));
        }
        // Multimap: a pinch vertex (where a dilated layout touches itself) has
        // two outgoing boundary edges.
        std::unordered_multimap<uint32_t, uint32_t> locator;
        for (const MosaicFace& mf : mosaic.faces) {
            const auto& f = mf.mapFace;
            for (int j = 0; j < 3; ++j) {
                const uint32_t s = f[j];
                const uint32_t e = f[(j + 1) % 3];
                if (!directedEdges.contains(edgeKey(e, s))) {
                    locator.emplace(mosaic.getLocalVertex(s).mapId, mosaic.getLocalVertex(e).mapId);
                }
            }
        }
        // Walk every boundary loop. Dilating a small layout can wrap it around a
        // hole, giving more than one loop; the edges are consumed as a set of
        // constraints, so all loops are kept back to back.
        while (!locator.empty()) {
            const uint32_t start = locator.begin()->first;
            uint32_t next = locator.begin()->second;
            locator.erase(locator.begin());
            mosaic.edges.emplace_back(start, next);
            while (next != start) {
                const auto pos = locator.find(next);
                if (pos == locator.end()) {
                    break;
                }
                mosaic.edges.emplace_back(pos->first, pos->second);
                next = pos->second;
                locator.erase(pos);
            }
            if (!locator.empty()) extraBoundaryLoops++;
        }
    }
    if (extraBoundaryLoops > 0) {
    }
    RescaleMosaics(layout);
    Repin(layout);
}

// Least-squares solution of A x = b through the normal equations, which are symmetric positive definite
// once every mosaic has pinned vertices: a sparse Cholesky factorization, or conjugate gradients from the
// guess in X (at most `iterations`) if the factorization fails.
void SolveLeastSquares(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, Eigen::VectorXd& X,
                       int iterations)
{
    if (A.cols() == 0) return;
    const Eigen::SparseMatrix<double> AtA = A.transpose() * A;
    const Eigen::VectorXd Atb = A.transpose() * b;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> direct(AtA);
    if (direct.info() == Eigen::Success) {
        X = direct.solve(Atb);
    }
    if (direct.info() != Eigen::Success) {
        Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
        cg.setMaxIterations(iterations);
        cg.setTolerance(1e-12);
        cg.compute(AtA);
        X = cg.solveWithGuess(Atb, X);
    }
}

void SolveLscm(ChartLayout& layout, int iterations)
{
    Repin(layout, true, true, true, true);
    const size_t constraints = layout.pinned.size();
    const auto& vertexLocations = layout.mesh.vertexLocations;
    Eigen::SparseMatrix<double> A(2 * layout.totalTriangleCount, 2 * (layout.totalVertexCount - constraints));
    Eigen::SparseMatrix<double> C(2 * layout.totalTriangleCount, 2 * constraints);
    Eigen::VectorXd b(2 * constraints);
    Eigen::VectorXd X(2 * (layout.totalVertexCount - constraints));
    b.setZero();
    X.setZero();

    size_t R = 0;
    Vector2d p1, p2, p3;
    int ui1, vi1, ui2, vi2, ui3, vi3;
    Vector2d w1, w2, w3;
    using Triple = Eigen::Triplet<double>;
    std::vector<Triple> freeTriples;
    std::vector<Triple> pinnedTriples;
    constexpr uint32_t kNotFound = uint32_t(-1);
    std::vector<uint32_t> freeMap(layout.totalVertexCount, kNotFound);
    {
        size_t fidx = 0;
        size_t cidx = 0;
        for (size_t n = 0; n < layout.totalVertexCount; n++) {
            if (layout.pinned.find(static_cast<uint32_t>(n)) == layout.pinned.end()) {
                freeMap[n] = static_cast<uint32_t>(fidx++);
            } else {
                freeMap[n] = static_cast<uint32_t>(cidx++);
            }
        }
    }
    for (Mosaic& mosaic : layout.mosaics) {
        for (MosaicVertex& mv : mosaic.vertexes) {
            if (!mv.locked) {
                X[2 * freeMap[mv.mapId]] = mv.uv.x();
                X[2 * freeMap[mv.mapId] + 1] = mv.uv.y();
            }
        }
        for (const MosaicFace& face : mosaic.faces) {
            const Vector3u& tri = face.mapFace;
            MosaicVertex& v1 = mosaic.getLocalVertex(tri.x());
            MosaicVertex& v2 = mosaic.getLocalVertex(tri.y());
            MosaicVertex& v3 = mosaic.getLocalVertex(tri.z());
            ProjectTriangle(vertexLocations[v1.meshId], vertexLocations[v2.meshId], vertexLocations[v3.meshId], p1, p2,
                            p3);
            w1.x() = p3.x() - p2.x();
            w1.y() = p3.y() - p2.y();
            w2.x() = p1.x() - p3.x();
            w2.y() = p1.y() - p3.y();
            w3.x() = p2.x() - p1.x();
            w3.y() = p2.y() - p1.y();
            const double area = (p1.x() * p2.y() - p1.y() * p2.x() + p2.x() * p3.y() - p2.y() * p3.x() +
                                 p3.x() * p1.y() - p3.y() * p1.x());
            const double sc = 1.0 / std::sqrt(std::max(layout.minArea, area));
            w1 *= sc;
            w2 *= sc;
            w3 *= sc;
            ui1 = 2 * static_cast<int>(freeMap[v1.mapId]);
            vi1 = ui1 + 1;
            ui2 = 2 * static_cast<int>(freeMap[v2.mapId]);
            vi2 = ui2 + 1;
            ui3 = 2 * static_cast<int>(freeMap[v3.mapId]);
            vi3 = ui3 + 1;
            if (v1.locked) {
                pinnedTriples.emplace_back(R, ui1, w1.x());
                pinnedTriples.emplace_back(R, vi1, -w1.y());
                pinnedTriples.emplace_back(R + 1, ui1, w1.y());
                pinnedTriples.emplace_back(R + 1, vi1, w1.x());
                b[ui1] = v1.uv.x();
                b[vi1] = v1.uv.y();
            } else {
                freeTriples.emplace_back(R, ui1, w1.x());
                freeTriples.emplace_back(R, vi1, -w1.y());
                freeTriples.emplace_back(R + 1, ui1, w1.y());
                freeTriples.emplace_back(R + 1, vi1, w1.x());
            }
            if (v2.locked) {
                pinnedTriples.emplace_back(R, ui2, w2.x());
                pinnedTriples.emplace_back(R, vi2, -w2.y());
                pinnedTriples.emplace_back(R + 1, ui2, w2.y());
                pinnedTriples.emplace_back(R + 1, vi2, w2.x());
                b[ui2] = v2.uv.x();
                b[vi2] = v2.uv.y();
            } else {
                freeTriples.emplace_back(R, ui2, w2.x());
                freeTriples.emplace_back(R, vi2, -w2.y());
                freeTriples.emplace_back(R + 1, ui2, w2.y());
                freeTriples.emplace_back(R + 1, vi2, w2.x());
            }
            if (v3.locked) {
                pinnedTriples.emplace_back(R, ui3, w3.x());
                pinnedTriples.emplace_back(R, vi3, -w3.y());
                pinnedTriples.emplace_back(R + 1, ui3, w3.y());
                pinnedTriples.emplace_back(R + 1, vi3, w3.x());
                b[ui3] = v3.uv.x();
                b[vi3] = v3.uv.y();
            } else {
                freeTriples.emplace_back(R, ui3, w3.x());
                freeTriples.emplace_back(R, vi3, -w3.y());
                freeTriples.emplace_back(R + 1, ui3, w3.y());
                freeTriples.emplace_back(R + 1, vi3, w3.x());
            }
            R += 2;
        }
    }
    A.setFromTriplets(freeTriples.begin(), freeTriples.end());
    C.setFromTriplets(pinnedTriples.begin(), pinnedTriples.end());
    b = -C * b;
    SolveLeastSquares(A, b, X, iterations);
    for (Mosaic& mosaic : layout.mosaics) {
        for (MosaicVertex& mv : mosaic.vertexes) {
            if (!mv.locked) {
                mv.uv = Vector2d(static_cast<float>(X[2 * freeMap[mv.mapId]]),
                                 static_cast<float>(X[2 * freeMap[mv.mapId] + 1]));
            }
        }
    }
    RescaleMosaics(layout);
    Repin(layout);
}

void RescaleMosaics(ChartLayout& layout)
{
    // Scale each mosaic's UV layout so its area equals its 3D area: every
    // chart then has the same size relative to the surface it maps.
    for (Mosaic& mosaic : layout.mosaics) {
        double area3d = 0.0;
        double areaUv = 0.0;
        for (MosaicFace& face : mosaic.faces) {
            const Vector3u& tri = face.mapFace;
            MosaicVertex& v1 = mosaic.getLocalVertex(tri.x());
            MosaicVertex& v2 = mosaic.getLocalVertex(tri.y());
            MosaicVertex& v3 = mosaic.getLocalVertex(tri.z());
            const Vector3f& p1 = layout.mesh.vertexLocations[v1.meshId];
            const Vector3f& p2 = layout.mesh.vertexLocations[v2.meshId];
            const Vector3f& p3 = layout.mesh.vertexLocations[v3.meshId];
            area3d += std::abs(TriangleArea(p1, p2, p3));
            areaUv += std::abs(TriangleArea(v1.uv, v2.uv, v3.uv));
        }
        if (!(areaUv > 0.0) || !(area3d > 0.0)) continue;
        double scale = std::sqrt(area3d / areaUv);
        for (MosaicVertex& v : mosaic.vertexes) {
            v.uv *= scale;
        }
        mosaic.scale = scale;
    }
}
}  // namespace mopt
