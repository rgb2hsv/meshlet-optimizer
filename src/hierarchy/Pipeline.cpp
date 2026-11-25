#include <hierarchy/Pipeline.hpp>

#include <hierarchy/HierarchyBuilder.hpp>

#include <chrono>
#include <iostream>

namespace mopt
{
void Pipeline::add(std::unique_ptr<PipelineStep> step) { steps_.push_back(std::move(step)); }

void Pipeline::clear() { steps_.clear(); }

int Pipeline::run(HierarchyBuilder& builder, StatusHandler handler)
{
    int result = 0;
    for (const auto& step : steps_) {
        std::cout << "Step " << step->name() << '\n';
        const auto start = std::chrono::steady_clock::now();
        result = step->run(builder, handler);
        std::cout << "Step " << step->name() << " took "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s\n";
        if (result == kStop) {
            std::cout << "Step " << step->name() << " ended the pipeline\n";
            break;
        }
    }
    return result;
}

int RepeatStep::run(HierarchyBuilder& builder, StatusHandler handler)
{
    int result = builder.chartCount;
    int pass = 0;
    for (; pass < maxPasses_ && keepGoing_(builder); pass++) {
        std::cout << name_ << " pass " << pass + 1 << '\n';
        const int r = body_.run(builder, handler);
        if (r == Pipeline::kStop) break;  // the body had nothing to do
        result = r;
    }
    std::cout << name_ << ": " << pass << " passes\n";
    return result;
}
}  // namespace mopt
