#include <mesh/PlyWriter.hpp>

#include <happly.h>

namespace mopt
{
void WritePlyMesh(const std::filesystem::path& file, const Mesh& mesh, const std::vector<int>& faceRegion)
{
    happly::PLYData ply;
    const size_t vertexCount = mesh.vertexLocations.size();
    ply.addElement("vertex", vertexCount);
    happly::Element& vertex = ply.getElement("vertex");
    const auto addVertexProperty = [&](const std::string& name, const auto& values, int axis) {
        std::vector<float> column(vertexCount);
        for (size_t i = 0; i < vertexCount; i++) column[i] = values[i][axis];
        vertex.addProperty<float>(name, column);
    };
    addVertexProperty("x", mesh.vertexLocations, 0);
    addVertexProperty("y", mesh.vertexLocations, 1);
    addVertexProperty("z", mesh.vertexLocations, 2);
    if (mesh.vertexNormals.size() == vertexCount) {
        addVertexProperty("nx", mesh.vertexNormals, 0);
        addVertexProperty("ny", mesh.vertexNormals, 1);
        addVertexProperty("nz", mesh.vertexNormals, 2);
    }

    const size_t faceCount = mesh.triIndexes.size();
    std::vector<std::vector<int>> indices(faceCount);
    for (size_t f = 0; f < faceCount; f++) {
        const Vector3u& t = mesh.triIndexes[f];
        indices[f] = {int(t.x()), int(t.y()), int(t.z())};
    }
    ply.addElement("face", faceCount);
    happly::Element& face = ply.getElement("face");
    face.addListProperty<int>("vertex_indices", indices);
    if (mesh.hasTexcoords()) {
        std::vector<std::vector<float>> texcoord(faceCount);
        for (size_t f = 0; f < faceCount; f++) {
            for (int c = 0; c < 3; c++) {
                const Vector2f& uv = mesh.texcoords[3 * f + c];
                texcoord[f].push_back(uv.x());
                texcoord[f].push_back(uv.y());
            }
        }
        face.addListProperty<float>("texcoord", texcoord);
    }
    if (faceRegion.size() == faceCount) face.addProperty<int>("region", faceRegion);
    ply.write(file.string(), happly::DataFormat::Binary);
}
}  // namespace mopt
