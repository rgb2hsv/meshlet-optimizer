/*
 * LevelCracks.hpp
 *
 * Checks that any two levels of the builder hierarchy can be drawn next to
 * each other without cracks, as the meshlet LOD cut mixes them. For every
 * pair of levels i < k (units: the level-k nodes with split halves merged,
 * level-i regions mapped up through RegionHierarchy::parent):
 *   - every level-i border vertex on an edge between two different level-k
 *     units must lie on a level-k border edge between two units (within
 *     1e-4 of the level-k mean border edge length);
 *   - every level-k vertex on a border between two units that lies on such a
 *     level-i border must be a level-i border vertex at exactly the same
 *     position (no T-junction).
 * A level-i region whose ancestor was split can be drawn under either half's
 * level-k unit, and only when neither is drawn, so an edge between two
 * level-i regions is under a level-k border only when their possible units
 * are disjoint.
 */
#pragma once

#include <core/Types.hpp>
#include <hierarchy/LodLevel.hpp>
#include <hierarchy/RegionHierarchy.hpp>

#include <vector>

namespace mopt
{

// Crack check of one pair of levels: border vertices checked, off the other border, or missing.
struct LevelPairCracks {
    int fine = 0, coarse = 0;          // the two levels
    int fineChecked = 0;               // finer border vertices under a coarser border
    int fineOff = 0;                   // of those, off the coarser border
    double maxDistance = 0.0;          // largest distance of those off it
    int coarseChecked = 0;             // coarser border vertices on a finer border under it
    int coarseMissing = 0;             // of those, not a finer border vertex (T-junction)
    std::vector<Vector3f> offSamples;  // up to 8 finer vertices off the coarser border
};

// Crack checks of every pair of levels.
struct LevelCrackReport {
    std::vector<LevelPairCracks> pairs;  // (0,1), (0,2), .., (1,2), ..
    int pairsWithCracks() const
    {
        int n = 0;
        for (const LevelPairCracks& p : pairs) n += p.fineOff > 0 || p.coarseMissing > 0;
        return n;
    }
};

LevelCrackReport MeasureLevelCracks(const std::vector<LodLevel>& levels, const RegionHierarchy& tree);

}  // namespace mopt
