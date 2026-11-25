#include <core/Check.hpp>
#include <core/Types.hpp>
#include <hierarchy/DiskGroups.hpp>
#include <mesh/Mesh.hpp>

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>

using namespace mopt;

long long mopt::RegionsEulerCharacteristic(const Mesh& mesh, const std::vector<std::vector<int>>& facesOfRegion,
                                           const std::vector<int>& regions)
{
    std::unordered_set<uint32_t> vertices;
    std::unordered_set<uint64_t> edges;
    long long faces = 0;
    for (int r : regions) {
        for (int f : facesOfRegion[r]) {
            const Vector3u t = mesh.triIndexes[f];
            const uint32_t v[3] = {t.x(), t.y(), t.z()};
            faces++;
            for (int k = 0; k < 3; k++) {
                vertices.insert(v[k]);
                const uint32_t a = std::min(v[k], v[(k + 1) % 3]), b = std::max(v[k], v[(k + 1) % 3]);
                edges.insert((uint64_t(a) << 32) | b);
            }
        }
    }
    return static_cast<long long>(vertices.size()) - static_cast<long long>(edges.size()) + faces;
}

namespace
{
// Region groups of one hierarchy level with their faces and weights; fills the holes of a group and sheds
// members until each group is a topological disk.
class Groups
{
public:
    Groups(const Mesh& mesh, const std::vector<int>& faceRegion, int regionCount,
           const std::vector<std::vector<int>>& adjacency, const std::vector<int>& weight, std::vector<int>& group)
        : mesh_(mesh),
          adjacency_(adjacency),
          weight_(weight),
          group_(group),
          facesOf_(regionCount),
          mark_(regionCount, 0)
    {
        for (size_t f = 0; f < faceRegion.size(); f++) facesOf_[faceRegion[f]].push_back(static_cast<int>(f));
        int maxId = -1;
        for (int g : group_) maxId = std::max(maxId, g);
        members_.resize(maxId + 1);
        for (int r = 0; r < regionCount; r++) members_[group_[r]].push_back(r);
    }

    int groupCount() const { return static_cast<int>(members_.size()); }
    const std::vector<int>& members(int g) const { return members_[g]; }
    int faces(const std::vector<int>& regions) const
    {
        int count = 0;
        for (int r : regions) count += static_cast<int>(facesOf_[r].size());
        return count;
    }
    long long weight(const std::vector<int>& regions) const
    {
        long long w = 0;
        for (int r : regions) w += weight_[r];
        return w;
    }
    bool connected(const std::vector<int>& regions)
    {
        if (regions.empty()) return false;
        stamp_++;
        for (int r : regions) mark_[r] = stamp_;
        std::vector<int> stack{regions.front()};
        mark_[regions.front()] = -stamp_;
        size_t reached = 0;
        while (!stack.empty()) {
            const int r = stack.back();
            stack.pop_back();
            reached++;
            for (int s : adjacency_[r]) {
                if (mark_[s] == stamp_) {
                    mark_[s] = -stamp_;
                    stack.push_back(s);
                }
            }
        }
        return reached == regions.size();
    }
    long long euler(const std::vector<int>& regions) const
    {
        return RegionsEulerCharacteristic(mesh_, facesOf_, regions);
    }
    bool isDisk(const std::vector<int>& regions) { return connected(regions) && euler(regions) == 1; }
    bool isDiskGroup(int g)
    {
        auto it = diskCache_.find(g);
        if (it != diskCache_.end()) return it->second;
        const bool disk = !members_[g].empty() && isDisk(members_[g]);
        diskCache_[g] = disk;
        return disk;
    }

    void move(int region, int to)
    {
        const int from = group_[region];
        auto& m = members_[from];
        m.erase(std::find(m.begin(), m.end(), region));
        members_[to].push_back(region);
        group_[region] = to;
        diskCache_.erase(from);
        diskCache_.erase(to);
    }
    int newGroup()
    {
        members_.emplace_back();
        return groupCount() - 1;
    }

    // Groups adjacent to g.
    std::vector<int> neighbors(int g) const
    {
        std::vector<int> out;
        for (int r : members_[g]) {
            for (int s : adjacency_[r]) {
                if (group_[s] != g) out.push_back(group_[s]);
            }
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }

    // Pieces of the group graph without g, as lists of groups, largest (in
    // faces) first.
    std::vector<std::vector<int>> piecesWithout(int g) const
    {
        std::vector<int> piece(groupCount(), -1);
        std::vector<std::vector<int>> pieces;
        std::vector<int> pieceFaces;
        for (int start = 0; start < groupCount(); start++) {
            if (start == g || members_[start].empty() || piece[start] >= 0) continue;
            const int id = static_cast<int>(pieces.size());
            pieces.emplace_back();
            pieceFaces.push_back(0);
            std::vector<int> stack{start};
            piece[start] = id;
            while (!stack.empty()) {
                const int h = stack.back();
                stack.pop_back();
                pieces[id].push_back(h);
                pieceFaces[id] += faces(members_[h]);
                for (int n : neighbors(h)) {
                    if (n != g && piece[n] < 0) {
                        piece[n] = id;
                        stack.push_back(n);
                    }
                }
            }
        }
        std::vector<int> order(pieces.size());
        for (size_t i = 0; i < order.size(); i++) order[i] = static_cast<int>(i);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return pieceFaces[a] > pieceFaces[b]; });
        std::vector<std::vector<int>> sorted;
        for (int i : order) sorted.push_back(std::move(pieces[i]));
        return sorted;
    }

private:
    const Mesh& mesh_;
    const std::vector<std::vector<int>>& adjacency_;
    const std::vector<int>& weight_;
    std::vector<int>& group_;
    std::vector<std::vector<int>> facesOf_;
    std::vector<std::vector<int>> members_;
    std::vector<int> mark_;
    int stamp_ = 0;
    std::unordered_map<int, bool> diskCache_;
};
}  // namespace

bool mopt::MakeGroupsDisks(const Mesh& mesh, const std::vector<int>& faceRegion, int regionCount,
                           const std::vector<std::vector<int>>& regionAdjacency, const std::vector<int>& regionWeight,
                           std::vector<int>& group, std::vector<char>& absorbed, const DiskGroupParams& params,
                           const std::vector<int>& regionIsland)
{
    bool changed = false;
    Groups groups(mesh, faceRegion, regionCount, regionAdjacency, regionWeight, group);

    // Closing: fill the small holes a group surrounds.
    if (params.fillHoles) {
        for (bool filled = true; filled;) {
            filled = false;
            for (int g = 0; g < groups.groupCount() && !filled; g++) {
                if (groups.members(g).size() < 2 || groups.isDiskGroup(g)) continue;
                const std::vector<std::vector<int>> pieces = groups.piecesWithout(g);
                const double limit = params.maxHoleFraction * groups.faces(groups.members(g));
                for (size_t p = 1; p < pieces.size(); p++) {  // pieces[0] is the outside
                    std::vector<int> regions;
                    for (int h : pieces[p]) {
                        regions.insert(regions.end(), groups.members(h).begin(), groups.members(h).end());
                    }
                    if (groups.faces(regions) > limit) continue;
                    // A hole in another UV island stays out: groups never span a texture seam.
                    const int island = regionIsland.empty() ? 0 : regionIsland[groups.members(g).front()];
                    if (!regionIsland.empty() &&
                        std::any_of(regions.begin(), regions.end(), [&](int r) { return regionIsland[r] != island; }))
                        continue;
                    for (int r : regions) {
                        groups.move(r, g);
                        absorbed[r] = 1;
                    }
                    filled = true;
                    changed = true;
                }
            }
        }
    }

    // Ejection: shed members until the group is a disk.
    if (params.ejectMembers) {
        const int initialGroups = groups.groupCount();
        for (int g = 0; g < initialGroups; g++) {
            if (groups.members(g).size() < 2 || groups.isDiskGroup(g)) continue;
            std::vector<int> kept = groups.members(g), shed;
            const size_t maxShed = kept.size() / 2;
            while (!groups.isDisk(kept) && shed.size() < maxShed) {
                int best = -1;
                bool bestDisk = false;
                long long bestEuler = std::numeric_limits<long long>::min();
                for (int m : kept) {
                    std::vector<int> rest;
                    for (int r : kept) {
                        if (r != m) rest.push_back(r);
                    }
                    if (!groups.connected(rest)) continue;
                    const long long chi = groups.euler(rest);
                    const bool disk = chi == 1;
                    const bool better =
                        best < 0 || (disk && !bestDisk) ||
                        (disk == bestDisk &&
                         (chi > bestEuler || (chi == bestEuler && regionWeight[m] < regionWeight[best])));
                    if (better) {
                        best = m;
                        bestDisk = disk;
                        bestEuler = chi;
                    }
                }
                if (best < 0) break;
                kept.erase(std::find(kept.begin(), kept.end(), best));
                shed.push_back(best);
            }
            if (shed.empty() || !groups.isDisk(kept)) continue;  // cannot be made a disk: leave it
            for (int m : shed) {
                // Join the lightest neighbouring group that stays a disk.
                int target = -1;
                long long targetWeight = 0;
                for (int s : regionAdjacency[m]) {
                    const int h = group[s];
                    if (h == g || h == group[m]) continue;
                    std::vector<int> joined = groups.members(h);
                    joined.push_back(m);
                    if (!groups.isDisk(joined)) continue;
                    const long long w = groups.weight(groups.members(h));
                    if (target < 0 || w < targetWeight) {
                        target = h;
                        targetWeight = w;
                    }
                }
                if (target >= 0) {
                    groups.move(m, target);
                    absorbed[m] = 1;
                } else {
                    groups.move(m, groups.newGroup());
                    absorbed[m] = 0;
                }
                changed = true;
            }
        }
    }
    return changed;
}
