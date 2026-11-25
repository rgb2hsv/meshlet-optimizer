#pragma once
// The hierarchy build as a list of steps run in order on a HierarchyBuilder; RepeatStep loops a
// sub-pipeline while a condition holds.
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mopt
{
class HierarchyBuilder;

// Progress callback: a status line and the fraction done; returning false is ignored.
using StatusHandler = std::function<bool(const std::string& status, float progress)>;

// One stage of the hierarchy build.
class PipelineStep
{
public:
    virtual ~PipelineStep() = default;
    virtual const char* name() const = 0;
    // Returns the current chart count, or Pipeline::kStop to end the pipeline (and a RepeatStep's loop).
    virtual int run(HierarchyBuilder& builder, StatusHandler handler) = 0;
};

// Steps run in order on a HierarchyBuilder.
class Pipeline
{
public:
    static constexpr int kStop = -1;
    void add(std::unique_ptr<PipelineStep> step);
    void clear();
    // Runs the steps in order, printing each one's duration; stops at the first kStop.
    int run(HierarchyBuilder& builder, StatusHandler handler);

private:
    std::vector<std::unique_ptr<PipelineStep>> steps_;
};

// Runs `body` repeatedly while `keepGoing` holds (checked before each pass), at most maxPasses times.
class RepeatStep : public PipelineStep
{
public:
    using Condition = std::function<bool(const HierarchyBuilder&)>;

    RepeatStep(std::string name, Pipeline body, Condition keepGoing, int maxPasses)
        : name_(std::move(name)), body_(std::move(body)), keepGoing_(std::move(keepGoing)), maxPasses_(maxPasses)
    {
    }
    const char* name() const override { return name_.c_str(); }
    int run(HierarchyBuilder& builder, StatusHandler handler) override;

private:
    std::string name_;
    Pipeline body_;
    Condition keepGoing_;
    int maxPasses_;
};
}  // namespace mopt
