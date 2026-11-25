#include <mesh/PlyReader.hpp>

#include <happly.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace mopt
{
namespace
{
// A vertex property as floats, whatever its stored type; empty when absent.
std::vector<float> FloatProperty(happly::Element& vertex, const std::string& name)
{
    if (!vertex.hasProperty(name)) return {};
    return vertex.getProperty<float>(name);
}

}  // namespace

void ReadPlyMesh(const std::filesystem::path& file, Mesh& mesh)
{
    mesh.clear();
    happly::PLYData ply(file.string());
    happly::Element& vertex = ply.getElement("vertex");
    const std::vector<float> x = FloatProperty(vertex, "x"), y = FloatProperty(vertex, "y"),
                             z = FloatProperty(vertex, "z");
    if (x.empty() || x.size() != y.size() || x.size() != z.size())
        throw std::runtime_error("PLY has no vertex positions: " + file.string());
    const size_t n = x.size();
    mesh.vertexLocations.resize(n);
    for (size_t i = 0; i < n; i++) mesh.vertexLocations[i] = Vector3f(x[i], y[i], z[i]);

    const std::vector<float> nx = FloatProperty(vertex, "nx"), ny = FloatProperty(vertex, "ny"),
                             nz = FloatProperty(vertex, "nz");
    if (nx.size() == n && ny.size() == n && nz.size() == n) {
        mesh.vertexNormals.resize(n);
        for (size_t i = 0; i < n; i++) mesh.vertexNormals[i] = Vector3f(nx[i], ny[i], nz[i]);
    }

    // Texture coordinates per vertex (u/v, s/t or texture_u/texture_v), expanded to the corners below.
    std::vector<float> vertexU, vertexV;
    for (const auto& [un, vn] : {std::pair{"u", "v"}, std::pair{"s", "t"}, std::pair{"texture_u", "texture_v"}}) {
        vertexU = FloatProperty(vertex, un);
        vertexV = FloatProperty(vertex, vn);
        if (vertexU.size() == n && vertexV.size() == n) break;
        vertexU.clear();
        vertexV.clear();
    }

    if (ply.hasElement("face")) {
        happly::Element& face = ply.getElement("face");
        const std::string list = face.hasProperty("vertex_indices") ? "vertex_indices" : "vertex_index";
        const std::vector<std::vector<size_t>> faces = face.getListPropertyAnySign<size_t>(list);
        // Texture coordinates per face corner (u0 v0 u1 v1 ...), as MeshLab and VCG write them.
        std::vector<std::vector<float>> faceUv;
        if (face.hasProperty("texcoord")) faceUv = face.getListProperty<float>("texcoord");
        const bool cornerUv = faceUv.size() == faces.size();
        const bool hasUv = cornerUv || !vertexU.empty();
        const auto uvOf = [&](size_t f, size_t k) {
            if (cornerUv) {
                Expects(faceUv[f].size() == 2 * faces[f].size(), "face {} has {} texcoords for {} corners", f,
                        faceUv[f].size(), faces[f].size());
                return Vector2f(faceUv[f][2 * k], faceUv[f][2 * k + 1]);
            }
            const size_t v = faces[f][k];
            return Vector2f(vertexU[v], vertexV[v]);
        };
        const auto addTriangle = [&](size_t f, size_t a, size_t b, size_t c) {
            const std::vector<size_t>& p = faces[f];
            mesh.triIndexes.push_back(Vector3u(uint32_t(p[a]), uint32_t(p[b]), uint32_t(p[c])));
            if (!hasUv) return;
            for (size_t k : {a, b, c}) mesh.texcoords.push_back(uvOf(f, k));
        };
        for (size_t f = 0; f < faces.size(); f++) {
            const std::vector<size_t>& p = faces[f];
            for (size_t c : p) {
                if (c >= n) throw std::runtime_error("PLY face index out of range: " + file.string());
            }
            if (p.size() == 4) {
                // Quads split along the shorter diagonal.
                const auto at = [&](size_t k) { return mesh.vertexLocations[p[k]]; };
                if ((at(0) - at(2)).squaredNorm() < (at(1) - at(3)).squaredNorm()) {
                    addTriangle(f, 0, 1, 2);
                    addTriangle(f, 2, 3, 0);
                } else {
                    addTriangle(f, 0, 1, 3);
                    addTriangle(f, 3, 1, 2);
                }
            } else {
                for (size_t k = 1; k + 1 < p.size(); k++) addTriangle(f, 0, k, k + 1);
            }
        }
    }
    if (mesh.vertexNormals.size() == 0 && mesh.triIndexes.size() > 0) mesh.updateVertexNormals();
}

}  // namespace mopt
