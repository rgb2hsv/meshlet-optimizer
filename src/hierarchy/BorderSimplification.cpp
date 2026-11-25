#include <core/Check.hpp>
#include <core/Types.hpp>
#include <hierarchy/BorderSimplification.hpp>
#include <mesh/Mesh.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <queue>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace mopt;

namespace
{
uint64_t EdgeKey(uint32_t a, uint32_t b) { return (uint64_t(std::min(a, b)) << 32) | std::max(a, b); }

using EdgeFaces = std::unordered_map<uint64_t, std::vector<int>>;

EdgeFaces BuildEdgeFaces(const Mesh& mesh)
{
    EdgeFaces edgeFaces;
    edgeFaces.reserve(mesh.triIndexes.size() * 3 / 2 + 1);
    for (size_t f = 0; f < mesh.triIndexes.size(); f++) {
        const Vector3u t = mesh.triIndexes[f];
        const uint32_t v[3] = {t.x(), t.y(), t.z()};
        for (int k = 0; k < 3; k++) edgeFaces[EdgeKey(v[k], v[(k + 1) % 3])].push_back(static_cast<int>(f));
    }
    return edgeFaces;
}

bool IsBorder(const std::vector<int>& faces, const std::vector<int>& label)
{
    return faces.size() == 2 && label[faces[0]] != label[faces[1]];
}

float EdgeLength(const Mesh& mesh, uint64_t key)
{
    return (mesh.vertexLocations[static_cast<uint32_t>(key >> 32)] -
            mesh.vertexLocations[static_cast<uint32_t>(key & 0xffffffffu)])
        .norm();
}

// Border vertex count and mean border / interior edge lengths.
struct Measure {
    int borderVertices = 0;
    double borderMean = 0.0, interiorMean = 0.0;
};

Measure MeasureBorders(const Mesh& mesh, const std::vector<int>& label, const EdgeFaces& edgeFaces)
{
    Measure m;
    std::vector<char> onBorder(mesh.vertexLocations.size(), 0);
    double borderSum = 0.0, interiorSum = 0.0;
    size_t borderCount = 0, interiorCount = 0;
    for (const auto& [key, faces] : edgeFaces) {
        if (IsBorder(faces, label)) {
            borderSum += EdgeLength(mesh, key);
            borderCount++;
            onBorder[key >> 32] = onBorder[key & 0xffffffffu] = 1;
        } else if (faces.size() == 2) {
            interiorSum += EdgeLength(mesh, key);
            interiorCount++;
        }
    }
    m.borderVertices = static_cast<int>(std::count(onBorder.begin(), onBorder.end(), 1));
    m.borderMean = borderCount ? borderSum / borderCount : 0.0;
    m.interiorMean = interiorCount ? interiorSum / interiorCount : 0.0;
    return m;
}

// A border chain between two regions (or a closed loop) and how far it still is from its target edge length.
struct Chain {
    int regionA = -1, regionB = -1;
    double length = 0.0;
    int edges = 0;
    // Closed loops, and chains from a junction back to itself, keep at
    // least three edges so the border does not pinch.
    bool loop = false;
    double target = 0.0;
    bool open() const { return edges > (loop ? 3 : 1) && length / edges < target; }
};

// Shortens region borders by collapsing border edges, shortest first, refusing collapses that fold
// triangles, break the link condition or that the caller's trySnap rejects.
class Collapser
{
public:
    Collapser(Mesh& mesh, std::vector<int>& label, const BorderSimplificationParams& params)
        : mesh_(mesh),
          label_(label),
          params_(params),
          cosMaxTurn_(static_cast<float>(std::cos(params.maxNormalChangeDegrees * M_PI / 180.0)))
    {
    }

    void run()
    {
        const EdgeFaces edgeFaces = BuildEdgeFaces(mesh_);
        const Measure before = MeasureBorders(mesh_, label_, edgeFaces);
        if (before.borderVertices == 0) return;
        setUp(edgeFaces, before.interiorMean);
        collapseAll();
        compact();
    }

private:
    // A candidate collapse in the queue: the border edge u-v and its length.
    struct Item {
        float length;
        uint32_t u, v;
        bool operator>(const Item& o) const { return length > o.length; }
    };

    void setUp(const EdgeFaces& edgeFaces, double globalInterior)
    {
        const size_t V = mesh_.vertexLocations.size(), F = mesh_.triIndexes.size();
        tris_.resize(F);
        normal0_.resize(F);
        vertexFaces_.assign(V, {});
        dead_.assign(F, 0);
        for (size_t f = 0; f < F; f++) {
            const Vector3u t = mesh_.triIndexes[f];
            tris_[f] = {t.x(), t.y(), t.z()};
            for (uint32_t v : tris_[f]) vertexFaces_[v].push_back(static_cast<int>(f));
            normal0_[f] = normalOf(tris_[f]);
        }
        // Region interior edge means.
        int regionCount = 0;
        for (int r : label_) regionCount = std::max(regionCount, r + 1);
        std::vector<double> sum(regionCount, 0.0);
        std::vector<int> count(regionCount, 0);
        std::vector<std::vector<uint32_t>> borderNbrs(V);
        std::vector<char> irregular(V, 0);  // on the mesh border or a non-manifold edge
        for (const auto& [key, faces] : edgeFaces) {
            const uint32_t a = static_cast<uint32_t>(key >> 32), b = static_cast<uint32_t>(key & 0xffffffffu);
            if (faces.size() != 2) {
                irregular[a] = irregular[b] = 1;
            } else if (IsBorder(faces, label_)) {
                borderNbrs[a].push_back(b);
                borderNbrs[b].push_back(a);
            } else {
                sum[label_[faces[0]]] += EdgeLength(mesh_, key);
                count[label_[faces[0]]]++;
            }
        }
        junction_.assign(V, 0);
        for (size_t v = 0; v < V; v++) {
            if (borderNbrs[v].empty()) continue;
            int regions[3] = {-1, -1, -1}, n = 0;
            for (int f : vertexFaces_[v]) {
                const int r = label_[f];
                if (std::find(regions, regions + n, r) != regions + n) continue;
                if (n == 3) {
                    n = 4;
                    break;
                }
                regions[n++] = r;
            }
            junction_[v] = irregular[v] || n >= 3 || borderNbrs[v].size() != 2;
        }
        // Chains: walk border edges between junctions; what is left are loops.
        chainOf_.assign(V, -1);
        nbr_.assign(V, {UINT32_MAX, UINT32_MAX});
        for (size_t v = 0; v < V; v++) {
            if (borderNbrs[v].size() == 2 && !junction_[v]) nbr_[v] = {borderNbrs[v][0], borderNbrs[v][1]};
        }
        const auto regionsOf = [&](uint32_t a, uint32_t b) {
            const std::vector<int>& faces = edgeFaces.at(EdgeKey(a, b));
            return std::make_pair(std::min(label_[faces[0]], label_[faces[1]]),
                                  std::max(label_[faces[0]], label_[faces[1]]));
        };
        std::unordered_map<uint64_t, char> visited;
        const auto walk = [&](uint32_t from, uint32_t to, bool loop) {
            Chain chain;
            chain.loop = loop;
            std::tie(chain.regionA, chain.regionB) = regionsOf(from, to);
            const int id = static_cast<int>(chains_.size());
            uint32_t prev = from, cur = to;
            while (true) {
                visited[EdgeKey(prev, cur)] = 1;
                chain.length += (mesh_.vertexLocations[prev] - mesh_.vertexLocations[cur]).norm();
                chain.edges++;
                if (junction_[cur] || (loop && cur == from)) break;
                chainOf_[cur] = id;
                const uint32_t next = nbr_[cur][0] == prev ? nbr_[cur][1] : nbr_[cur][0];
                prev = cur;
                cur = next;
            }
            if (loop) chainOf_[from] = id;
            chain.loop = loop || cur == from;
            chains_.push_back(chain);
        };
        for (uint32_t v = 0; v < V; v++) {
            if (!junction_[v]) continue;
            for (uint32_t w : borderNbrs[v]) {
                if (!visited.count(EdgeKey(v, w))) walk(v, w, false);
            }
        }
        for (uint32_t v = 0; v < V; v++) {
            if (junction_[v] || borderNbrs[v].size() != 2 || chainOf_[v] >= 0) continue;
            walk(v, nbr_[v][0], true);
        }
        // Targets.
        const auto mean = [&](int r) { return count[r] ? sum[r] / count[r] : -1.0; };
        for (Chain& c : chains_) {
            double a = mean(c.regionA), b = mean(c.regionB);
            if (a < 0) a = b;
            if (b < 0) b = a;
            const double base = a >= 0 ? 0.5 * (a + b) : globalInterior;
            c.target = params_.targetScale * base;
        }
        // Every chain edge with a movable end.
        for (uint32_t v = 0; v < V; v++) {
            if (chainOf_[v] < 0) continue;
            for (uint32_t w : nbr_[v]) {
                if (w > v || junction_[w]) push(v, w);
            }
        }
    }

    Vector3f normalOf(const std::array<uint32_t, 3>& t) const
    {
        const Vector3f a = mesh_.vertexLocations[t[0]];
        return (mesh_.vertexLocations[t[1]] - a).cross(mesh_.vertexLocations[t[2]] - a);
    }

    void push(uint32_t a, uint32_t b)
    {
        queue_.push({(mesh_.vertexLocations[a] - mesh_.vertexLocations[b]).norm(), a, b});
    }

    bool linked(uint32_t a, uint32_t b) const
    {
        if (!junction_[a]) return nbr_[a][0] == b || nbr_[a][1] == b;
        return !junction_[b] && (nbr_[b][0] == a || nbr_[b][1] == a);
    }

    uint32_t otherNbr(uint32_t v, uint32_t u) const { return nbr_[v][0] == u ? nbr_[v][1] : nbr_[v][0]; }

    // Edge ab of face f: is the face across it in another region, or absent?
    bool borderFor(int f, uint32_t a, uint32_t b) const
    {
        for (int g : vertexFaces_[a]) {
            if (g == f) continue;
            const std::array<uint32_t, 3>& t = tris_[g];
            if (std::find(t.begin(), t.end(), b) != t.end()) return label_[g] != label_[f];
        }
        return true;
    }

    enum class Verdict { Ok, Fold, Topology };

    // Collapse v (a chain vertex) onto u, its chain neighbour?
    Verdict check(uint32_t v, uint32_t u) const
    {
        // Link condition: common neighbours of u and v are exactly the third
        // vertices of the faces on edge uv.
        std::vector<uint32_t> nv, nu, third;
        for (int f : vertexFaces_[v]) {
            bool hasU = false;
            for (uint32_t w : tris_[f]) hasU |= w == u;
            for (uint32_t w : tris_[f]) {
                if (w == v || w == u) continue;
                nv.push_back(w);
                if (hasU) third.push_back(w);
            }
        }
        for (int f : vertexFaces_[u]) {
            for (uint32_t w : tris_[f]) {
                if (w != u && w != v) nu.push_back(w);
            }
        }
        for (auto* s : {&nv, &nu, &third}) {
            std::sort(s->begin(), s->end());
            s->erase(std::unique(s->begin(), s->end()), s->end());
        }
        std::vector<uint32_t> common;
        std::set_intersection(nv.begin(), nv.end(), nu.begin(), nu.end(), std::back_inserter(common));
        if (common != third || third.size() != 2) return Verdict::Topology;
        // Ears: a face on uv whose other two edges are both region borders
        // (or the mesh's border) is all its region has between them, so its
        // region would lose it and change topology, or vanish.
        for (int f : vertexFaces_[v]) {
            const std::array<uint32_t, 3>& t = tris_[f];
            if (std::find(t.begin(), t.end(), u) == t.end()) continue;
            const uint32_t w = t[0] != u && t[0] != v ? t[0] : t[1] != u && t[1] != v ? t[1] : t[2];
            if (borderFor(f, w, u) && borderFor(f, w, v)) return Verdict::Topology;
        }
        // Remaining faces around v: no flip against their current or
        // original normal, no vanishing area.
        for (int f : vertexFaces_[v]) {
            std::array<uint32_t, 3> t = tris_[f];
            if (std::find(t.begin(), t.end(), u) != t.end()) continue;
            const Vector3f now = normalOf(t);
            for (uint32_t& w : t) {
                if (w == v) w = u;
            }
            const Vector3f next = normalOf(t);
            float longest = 0.0f;
            for (int k = 0; k < 3; k++) {
                const Vector3f e = mesh_.vertexLocations[t[k]] - mesh_.vertexLocations[t[(k + 1) % 3]];
                longest = std::max(longest, e.dot(e));
            }
            if (next.norm() <= 1e-6f * longest) return Verdict::Fold;
            if (next.dot(now) <= 0.0f) return Verdict::Fold;
            const Vector3f n0 = normal0_[f];
            if (next.dot(n0) <= cosMaxTurn_ * next.norm() * n0.norm()) return Verdict::Fold;
        }
        return Verdict::Ok;
    }

    // With params_.trySnap: v and the vertices already removed under its two
    // chain edges are now all under the merged edge u-b; ask for them to be
    // snapped onto it. On success they are recorded under u-b.
    bool snapUnder(uint32_t v, uint32_t u)
    {
        if (!params_.trySnap) return true;
        const uint32_t b = otherNbr(v, u);
        std::vector<uint32_t> moved{v};
        for (uint64_t key : {EdgeKey(u, v), EdgeKey(v, b)}) {
            const auto it = under_.find(key);
            if (it != under_.end()) moved.insert(moved.end(), it->second.begin(), it->second.end());
        }
        const Vector3f a = mesh_.vertexLocations[u], ab = mesh_.vertexLocations[b] - a;
        std::vector<std::pair<uint32_t, Vector3f>> snaps;
        for (uint32_t w : moved) {
            const float t =
                std::clamp((mesh_.vertexLocations[w] - a).dot(ab) / std::max(1e-30f, ab.dot(ab)), 0.0f, 1.0f);
            snaps.push_back({w, a + t * ab});
        }
        if (!params_.trySnap(snaps, u, b)) return false;
        under_.erase(EdgeKey(u, v));
        under_.erase(EdgeKey(v, b));
        under_[EdgeKey(u, b)] = std::move(moved);
        return true;
    }

    void collapse(uint32_t v, uint32_t u)
    {
        const bool uvTexture = mesh_.texcoords.size() == 3 * tris_.size();
        // u's corner UV in each region, from u's own faces.
        std::unordered_map<int, Vector2f> uvOfU;
        if (uvTexture) {
            for (int f : vertexFaces_[u]) {
                for (int k = 0; k < 3; k++) {
                    if (tris_[f][k] == u) uvOfU.emplace(label_[f], mesh_.texcoords[3 * f + k]);
                }
            }
        }
        const std::vector<int> faces = vertexFaces_[v];
        for (int f : faces) {
            std::array<uint32_t, 3>& t = tris_[f];
            if (std::find(t.begin(), t.end(), u) != t.end()) {
                dead_[f] = 1;
                for (uint32_t w : t) {
                    if (w == v) continue;
                    auto& list = vertexFaces_[w];
                    list.erase(std::remove(list.begin(), list.end(), f), list.end());
                }
                continue;
            }
            for (int k = 0; k < 3; k++) {
                if (t[k] != v) continue;
                t[k] = u;
                if (uvTexture) {
                    const auto it = uvOfU.find(label_[f]);
                    if (it != uvOfU.end()) mesh_.texcoords[3 * f + k] = it->second;
                }
            }
            vertexFaces_[u].push_back(f);
        }
        vertexFaces_[v].clear();
        // Splice v out of its chain.
        const uint32_t b = otherNbr(v, u);
        Chain& chain = chains_[chainOf_[v]];
        const auto dist = [&](uint32_t p, uint32_t q) {
            return (mesh_.vertexLocations[p] - mesh_.vertexLocations[q]).norm();
        };
        chain.length += dist(u, b) - dist(u, v) - dist(v, b);
        chain.edges--;
        if (!junction_[u]) nbr_[u][nbr_[u][0] == v ? 0 : 1] = b;
        if (!junction_[b]) nbr_[b][nbr_[b][0] == v ? 0 : 1] = u;
        nbr_[v] = {UINT32_MAX, UINT32_MAX};
        chainOf_[v] = -1;
        if (!junction_[u] || !junction_[b]) push(u, b);
    }

    void collapseAll()
    {
        while (!queue_.empty()) {
            const Item item = queue_.top();
            queue_.pop();
            if (!linked(item.u, item.v)) continue;
            const uint32_t p = item.u, q = item.v;
            const int id = chainOf_[junction_[p] ? q : p];
            if (id < 0 || !chains_[id].open()) continue;
            // Candidates to move: non-junction ends, the one whose other
            // chain edge is shorter first.
            uint32_t order[2] = {p, q};
            int n = 0;
            if (!junction_[p] && !junction_[q]) {
                const auto other = [&](uint32_t v, uint32_t u) {
                    return (mesh_.vertexLocations[v] - mesh_.vertexLocations[otherNbr(v, u)]).norm();
                };
                if (other(q, p) < other(p, q)) std::swap(order[0], order[1]);
                n = 2;
            } else {
                order[0] = junction_[p] ? q : p;
                n = 1;
            }
            for (int i = 0; i < n; i++) {
                const uint32_t v = order[i], u = v == p ? q : p;
                const Verdict verdict = check(v, u);
                if (verdict == Verdict::Ok) {
                    if (!snapUnder(v, u)) {
                        continue;
                    }
                    collapse(v, u);
                    break;
                }
            }
        }
    }

    void compact()
    {
        const size_t V = mesh_.vertexLocations.size(), F = tris_.size();
        const bool uvTexture = mesh_.texcoords.size() == 3 * F;
        std::vector<uint32_t> remap(V, UINT32_MAX);
        std::vector<Vector3u> tris;
        std::vector<Vector2f> uv;
        std::vector<int> label;
        uint32_t next = 0;
        for (size_t f = 0; f < F; f++) {
            if (dead_[f]) continue;
            uint32_t t[3];
            for (int k = 0; k < 3; k++) {
                uint32_t& r = remap[tris_[f][k]];
                if (r == UINT32_MAX) r = next++;
                t[k] = r;
                if (uvTexture) uv.push_back(mesh_.texcoords[3 * f + k]);
            }
            tris.push_back(Vector3u(t[0], t[1], t[2]));
            label.push_back(label_[f]);
        }
        std::vector<Vector3f> locations, normals;
        locations.resize(next);
        const bool hasNormals = mesh_.vertexNormals.size() == V;
        if (hasNormals) normals.resize(next);
        for (size_t v = 0; v < V; v++) {
            const uint32_t r = remap[v];
            if (r == UINT32_MAX) continue;
            locations[r] = mesh_.vertexLocations[v];
            if (hasNormals) normals[r] = mesh_.vertexNormals[v];
        }
        mesh_.vertexLocations = std::move(locations);
        if (hasNormals) mesh_.vertexNormals = std::move(normals);
        mesh_.triIndexes = std::move(tris);
        if (uvTexture) mesh_.texcoords = std::move(uv);
        label_ = std::move(label);
    }

    Mesh& mesh_;
    std::vector<int>& label_;
    BorderSimplificationParams params_;
    float cosMaxTurn_;
    std::vector<std::array<uint32_t, 3>> tris_;
    std::vector<Vector3f> normal0_;
    std::vector<std::vector<int>> vertexFaces_;
    std::vector<char> dead_;
    std::vector<char> junction_;
    std::vector<int> chainOf_;
    std::vector<std::array<uint32_t, 2>> nbr_;
    // Removed vertices under each chain edge (by its two surviving ends),
    // for params_.trySnap.
    std::unordered_map<uint64_t, std::vector<uint32_t>> under_;
    std::vector<Chain> chains_;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> queue_;
};
}  // namespace

void mopt::SimplifyRegionBorders(Mesh& mesh, std::vector<int>& faceRegion, const BorderSimplificationParams& params)
{
    Collapser(mesh, faceRegion, params).run();
}
