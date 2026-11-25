#pragma once
#include <core/Check.hpp>
#include <span>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mopt
{
namespace detail
{
/// Key -> heap slot lookup backed by a hash map. Works for any hashable key.
/// Key -> heap slot lookup backed by a flat array. Keys must be small dense integers
/// (e.g. vertex ids or linearized pixel/voxel indices); memory is O(max key).
struct DenseHeapIndex {
    static constexpr uint32_t kNpos = std::numeric_limits<uint32_t>::max();

    [[nodiscard]] uint32_t find(uint32_t key) const { return key < slots.size() ? slots[key] : kNpos; }
    void set(uint32_t key, uint32_t slot)
    {
        if (key >= slots.size()) {
            slots.resize(std::max<size_t>(size_t(key) + 1, slots.size() * 2), kNpos);
        }
        slots[key] = slot;
    }
    void erase(uint32_t key)
    {
        if (key < slots.size()) {
            slots[key] = kNpos;
        }
    }
    void reserve(size_t) {}
    /// Pre-size for keys in [0, n) so pushes never grow the table.
    void reserveKeys(size_t n)
    {
        if (n > slots.size()) {
            slots.resize(n, kNpos);
        }
    }
    /// Only resets slots that are live, so clearing is O(heap size), not O(key range).
    template <class Items>
    void clear(const Items& items)
    {
        for (const auto& item : items) {
            slots[item.key] = kNpos;
        }
    }

    std::vector<uint32_t> slots;
};
}  // namespace detail

/// Priority queue with O(log n) update-by-key. Indices 1..size are heap nodes; slot 0 is a sentinel.
/// `Index` maps a key to its heap slot; use the `MutableHeap` / `DenseMutableHeap` aliases below.
/// Requires `Key` and `T` to be default-constructible (scratch / growth slots).
template <class Key, class T, class Compare, class Index>
class BasicMutableHeap
{
    static_assert(std::is_default_constructible_v<Key> && std::is_default_constructible_v<T>,
                  "MutableHeap requires default-constructible Key and T");
    static constexpr uint32_t kNpos = Index::kNpos;

public:
    // A heap entry: the key it is found by and its priority.
    struct Item {
        Key key{};
        T value{};
        Item() = default;
        Item(const Key& keyIn, const T& valueIn) : key(keyIn), value(valueIn) {}
        /// Same ordering as a default (std::less) heap: smaller value first, ties by smaller key.
        bool operator<(const Item& item) const
        {
            if (value < item.value) return true;
            if (item.value < value) return false;
            return key < item.key;
        }
    };

    explicit BasicMutableHeap(const Compare& compare = Compare()) : mCompare_(compare), mSize_(0)
    {
        mStorage_.resize(1);  // index 0: sentinel / scratch
    }

    [[nodiscard]] bool empty() const noexcept { return mSize_ == 0; }

    /// Reserve heap slots and key->index capacity (reduces reallocation / rehash in bulk use).
    void reserve(size_t sz)
    {
        if (sz == 0) {
            return;
        }
        // +1 for one-based indexing (slot 0 unused for heap data)
        if (mStorage_.size() < sz + 1) {
            mStorage_.resize(sz + 1);
        }
        mIndex_.reserve(sz);
    }

    /// Reserve room for keys in [0, n). For a dense heap this pre-sizes the lookup table.
    void reserveKeys(size_t n) { mIndex_.reserveKeys(n); }

    [[nodiscard]] std::optional<Item> top() const
    {
        if (empty()) {
            return std::nullopt;
        }
        return mStorage_[1];
    }

    /// Heap items in storage order (not sorted); slot 0 is the sentinel and is excluded.
    [[nodiscard]] std::span<const Item> items() const noexcept { return {mStorage_.data() + 1, mSize_}; }

    [[nodiscard]] size_t size() const noexcept { return mSize_; }

    [[nodiscard]] bool contains(const Key& key) const { return mIndex_.find(key) != kNpos; }

    /// Change the value of an existing key. Returns false if the key is not in the heap.
    bool exchange(const Key& key, const T& value)
    {
        const uint32_t index = mIndex_.find(key);
        if (index == kNpos) {
            return false;
        }
        Item& v = mStorage_[index];
        // The key is unchanged, so only the value decides the direction.
        const bool up = mCompare_(value, v.value);
        v.value = value;
        if (up) {
            siftUp(index);
        } else {
            siftDown(index);
        }
        return true;
    }

    bool push(const Key& key, const T& value)
    {
        Expects(mIndex_.find(key) == kNpos, "Duplicate keys not allowed");
        Expects(mSize_ < size_t(kNpos) - 1, "MutableHeap is full");
        size_t hole = ++mSize_;
        if (hole + 1 > mStorage_.size()) {
            mStorage_.resize(std::max(size_t(2), mStorage_.size()) << 1);
        }
        const Item item{key, value};
        mStorage_[0] = item;  // sentinel stops the loop at the root
        size_t nextHole = hole >> 1;
        for (; compare(item, mStorage_[nextHole]); hole = nextHole, nextHole >>= 1) {
            mStorage_[hole] = std::move(mStorage_[nextHole]);
            mIndex_.set(mStorage_[hole].key, static_cast<uint32_t>(hole));
        }
        mStorage_[hole] = item;
        mIndex_.set(key, static_cast<uint32_t>(hole));
        return true;
    }

    void pop()
    {
        if (mSize_ == 0) {
            return;
        }
        mIndex_.erase(mStorage_[1].key);
        const size_t last = mSize_--;
        if (last > 1) {
            mStorage_[1] = std::move(mStorage_[last]);
            siftDown(1);
        }
    }

    bool eraseFirst(const std::function<bool(const Key& key)>& eraseFunc)
    {
        for (size_t n = 1; n <= mSize_; ++n) {
            if (eraseFunc(mStorage_[n].key)) {
                const Key key = mStorage_[n].key;
                return erase(key);
            }
        }
        return false;
    }

    bool erase(const Key& key)
    {
        const uint32_t idx = mIndex_.find(key);
        if (idx == kNpos) {
            return false;
        }
        mIndex_.erase(key);
        const size_t last = mSize_--;
        if (idx != last) {
            const bool up = compare(mStorage_[last], mStorage_[idx]);
            mStorage_[idx] = std::move(mStorage_[last]);
            if (up) {
                siftUp(idx);
            } else {
                siftDown(idx);
            }
        }
        return true;
    }

    /// Empties the heap but keeps its allocated storage.
    void clear()
    {
        mIndex_.clear(items());
        mSize_ = 0;
    }

    /// True if the heap order holds and every item's index entry points at its slot.
    bool isHeap() const
    {
        for (size_t n = 1; n <= mSize_; ++n) {
            const Item& it = mStorage_[n];
            if (mIndex_.find(it.key) != n) {
                return false;
            }
            if (n > 1 && compare(it, mStorage_[n >> 1])) {
                return false;
            }
        }
        return true;
    }

    std::string toString() const
    {
        std::stringstream ss;
        ss << "[";
        for (size_t n = 1; n <= mSize_; ++n) {
            const Item& it = mStorage_[n];
            ss << std::to_string(it.key) << ":" << std::to_string(it.value) << ((n < mSize_) ? ", " : "");
        }
        ss << "]";
        return ss.str();
    }

private:
    [[nodiscard]] bool compare(const Item& a, const Item& b) const
    {
        if (mCompare_(a.value, b.value)) {
            return true;
        }
        if (mCompare_(b.value, a.value)) {
            return false;
        }
        return a.key < b.key;
    }

    /// Sift node at index `i` toward the leaves. Always writes the final index entry.
    void siftDown(size_t i)
    {
        Item tmp = std::move(mStorage_[i]);
        for (;;) {
            size_t j = i << 1;
            if (j > mSize_) {
                break;
            }
            if (j < mSize_ && compare(mStorage_[j + 1], mStorage_[j])) {
                ++j;
            }
            if (!compare(mStorage_[j], tmp)) {
                break;
            }
            mStorage_[i] = std::move(mStorage_[j]);
            mIndex_.set(mStorage_[i].key, static_cast<uint32_t>(i));
            i = j;
        }
        mStorage_[i] = std::move(tmp);
        mIndex_.set(mStorage_[i].key, static_cast<uint32_t>(i));
    }

    /// Sift node at `k` toward the root. Always writes the final index entry.
    void siftUp(size_t k)
    {
        Item v = std::move(mStorage_[k]);
        size_t parent = k >> 1;
        while (parent > 0 && compare(v, mStorage_[parent])) {
            mStorage_[k] = std::move(mStorage_[parent]);
            mIndex_.set(mStorage_[k].key, static_cast<uint32_t>(k));
            k = parent;
            parent = k >> 1;
        }
        mStorage_[k] = std::move(v);
        mIndex_.set(mStorage_[k].key, static_cast<uint32_t>(k));
    }

    Compare mCompare_{};
    size_t mSize_{0};
    std::vector<Item> mStorage_;
    Index mIndex_;
};

/// Mutable heap for any hashable key.

/// Mutable heap for dense uint32_t keys (vertex ids, linear pixel/voxel indices).
/// Lookups are a flat array access instead of a hash; memory is O(max key).
template <class T, class Compare = std::less<T>>
using DenseMutableHeap = BasicMutableHeap<uint32_t, T, Compare, detail::DenseHeapIndex>;

}  // namespace mopt
