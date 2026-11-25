#pragma once
// Packs flattened charts (mosaics) into the unit square.
#include <flattening/Mosaic.hpp>

#include <vector>

namespace mopt
{
// Bin packing of flattened charts: each chart is turned so its minimum-area enclosing rectangle (over the
// directions of its convex hull's edges) is axis-aligned; the rectangles, grown by `padding` times the mean
// rectangle side on each side, are packed largest first by MaxRects (best short side fit, 90-degree turns
// allowed) into the smallest square that holds them, which is then scaled uniformly to [0,1]^2. Charts are
// only turned, moved and scaled, never mirrored or stretched, so their shapes and aspect ratios are kept.
class ChartBinPack
{
public:
    explicit ChartBinPack(double padding = 0.02) : padding_(padding) {}
    // Packs the mosaics' UVs in place; returns the scale applied.
    double pack(std::vector<Mosaic>& mosaics) const;

private:
    double padding_;
};
}  // namespace mopt
