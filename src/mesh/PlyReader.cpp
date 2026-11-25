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

    if (ply.hasElement("face")) {
        happly::Element& face = ply.getElement("face");
        const std::string list = face.hasProperty("vertex_indices") ? "vertex_indices" : "vertex_index";
        const std::vector<std::vector<size_t>> faces = face.getListPropertyAnySign<size_t>(list);
        for (const std::vector<size_t>& f : faces) {
            for (size_t c : f) {
                if (c >= n) throw std::runtime_error("PLY face index out of range: " + file.string());
            }
            if (f.size() == 3) {
                mesh.triIndexes.push_back(Vector3u(uint32_t(f[0]), uint32_t(f[1]), uint32_t(f[2])));
            } else if (f.size() == 4) {
                mesh.quadIndexes.push_back(Vector4u(uint32_t(f[0]), uint32_t(f[1]), uint32_t(f[2]), uint32_t(f[3])));
            } else {
                for (size_t k = 1; k + 1 < f.size(); k++) {
                    mesh.triIndexes.push_back(Vector3u(uint32_t(f[0]), uint32_t(f[k]), uint32_t(f[k + 1])));
                }
            }
        }
    }
    if (mesh.quadIndexes.size() > 0) mesh.convertQuadsToTriangles();
    if (mesh.vertexNormals.size() == 0 && mesh.triIndexes.size() > 0) mesh.updateVertexNormals();
}

}  // namespace mopt
