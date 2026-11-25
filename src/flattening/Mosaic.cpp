#include <core/Check.hpp>
#include <core/Types.hpp>
#include <flattening/Mosaic.hpp>
#include <sstream>

#include <algorithm>

namespace mopt
{
MosaicVertex& Mosaic::getLocalVertex(uint32_t uid)
{
    auto pos = vertexMap.find(uid);
    if (pos == vertexMap.end()) {
        throw std::runtime_error(Format("{}", (std::ostringstream() << "Could not find vertex " << uid).str()));
    }
    MosaicVertex& ret = vertexes[pos->second];
    if (ret.mapId != uid) {
        throw std::runtime_error(
            Format("{}", (std::ostringstream() << "Ids do not match " << ret.mapId << " / " << uid).str()));
    }
    return ret;
}
const MosaicVertex& Mosaic::getLocalVertex(uint32_t uid) const
{
    auto pos = vertexMap.find(uid);
    if (pos == vertexMap.end()) {
        throw std::runtime_error(Format("{}", (std::ostringstream() << "Could not find vertex " << uid).str()));
    }
    const MosaicVertex& ret = vertexes[pos->second];
    if (ret.mapId != uid) {
        throw std::runtime_error(
            Format("{}", (std::ostringstream() << "Ids do not match " << ret.mapId << " / " << uid).str()));
    }
    return ret;
}
}  // namespace mopt
