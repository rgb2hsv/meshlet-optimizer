#include <hierarchy/HierarchyBuilder.hpp>

#include <hierarchy/HierarchySteps.hpp>

#include <iostream>

namespace mopt
{
HierarchyBuilder::HierarchyBuilder() = default;

HierarchyBuilder::~HierarchyBuilder() = default;

void HierarchyBuilder::buildPipeline(const HierarchyParams& params)
{
    pipeline_.clear();
    AddHierarchySteps(pipeline_, params);
}

int HierarchyBuilder::build(const Mesh& input, const StatusHandler& handler)
{
    mesh = input;
    pinned.clear();
    std::cout << "Building the hierarchy of " << mesh << '\n';
    return pipeline_.run(*this, handler);
}

LevelWork& HierarchyBuilder::levelWork()
{
    if (!levelWork_) levelWork_ = std::make_unique<LevelWork>();
    return *levelWork_;
}
}  // namespace mopt
