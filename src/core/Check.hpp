#pragma once
// Expects(cond, "format {}", args...) and Fail("format {}", args...): throw std::runtime_error with a
// fmt-formatted message and the source location. Arguments fmt cannot format (Eigen vectors) are
// printed with operator<<.
#include <fmt/format.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace mopt
{
namespace detail
{
template <class T>
// Wraps a Format argument so values fmt cannot format are printed with operator<<.
struct FormatArg {
    const T& value;
};
}  // namespace detail

template <class... Args>
std::string Format(std::string_view format, const Args&... args)
{
    try {
        return fmt::format(fmt::runtime(format), detail::FormatArg<Args>{args}...);
    } catch (const std::exception& e) {
        return std::string(format) + " <formatting error: " + e.what() + ">";
    }
}

namespace detail
{
[[noreturn]] inline void Throw(const char* condition, const char* file, int line, const std::string& message)
{
    std::string text = std::string(file) + ":" + std::to_string(line) + ": ";
    if (condition) text += std::string("expected ") + condition + (message.empty() ? "" : ": ");
    throw std::runtime_error(text + message);
}
[[noreturn]] inline void FailAt(const char* condition, const char* file, int line) { Throw(condition, file, line, {}); }
template <class... Args>
[[noreturn]] void FailAt(const char* condition, const char* file, int line, std::string_view format,
                         const Args&... args)
{
    Throw(condition, file, line, Format(format, args...));
}
}  // namespace detail
}  // namespace mopt

template <class T>
struct fmt::formatter<mopt::detail::FormatArg<T>, char> {
    static constexpr bool kFormattable = fmt::is_formattable<T, char>::value;
    fmt::formatter<std::conditional_t<kFormattable, T, std::string_view>, char> inner;
    constexpr auto parse(fmt::format_parse_context& ctx) { return inner.parse(ctx); }
    auto format(const mopt::detail::FormatArg<T>& arg, fmt::format_context& ctx) const
    {
        if constexpr (kFormattable) {
            return inner.format(arg.value, ctx);
        } else {
            std::ostringstream ss;
            ss << arg.value;
            return fmt::format_to(ctx.out(), "{}", ss.str());
        }
    }
};

#define Expects(cond, ...) \
    (static_cast<bool>(cond) ? void(0) : ::mopt::detail::FailAt(#cond, __FILE__, __LINE__, ##__VA_ARGS__))
#define Fail(...) ::mopt::detail::FailAt(nullptr, __FILE__, __LINE__, ##__VA_ARGS__)
