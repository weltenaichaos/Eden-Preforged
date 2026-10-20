// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <tuple>
#include <utility>
#include <vector>

namespace Dynarmic::Backend::Arm64 {

// Every emitted block keeps a few small maps and sets for as long as it stays in the code cache,
// and the code caches can hold hundreds of thousands of blocks. A flat hash table takes about
// 1 KiB even for one entry, which adds up to gigabytes, so these keep their entries in a vector
// sized to fit and look them up by going through it.

/// A map with few entries, in the order they were added.
template<typename Key, typename Value>
class CompactMap {
public:
    using value_type = std::pair<Key, Value>;
    using iterator = typename std::vector<value_type>::iterator;
    using const_iterator = typename std::vector<value_type>::const_iterator;

    iterator begin() { return entries.begin(); }
    iterator end() { return entries.end(); }
    const_iterator begin() const { return entries.begin(); }
    const_iterator end() const { return entries.end(); }

    bool empty() const { return entries.empty(); }
    std::size_t size() const { return entries.size(); }
    void clear() { entries.clear(); }

    iterator find(const Key& key) {
        return std::find_if(entries.begin(), entries.end(), [&](const value_type& entry) { return entry.first == key; });
    }

    const_iterator find(const Key& key) const {
        return std::find_if(entries.begin(), entries.end(), [&](const value_type& entry) { return entry.first == key; });
    }

    Value& operator[](const Key& key) {
        if (const auto iter = find(key); iter != entries.end()) {
            return iter->second;
        }
        return entries.emplace_back(std::piecewise_construct, std::forward_as_tuple(key), std::forward_as_tuple()).second;
    }

    /// Adds an entry unless the key is already there, like std::map::emplace.
    template<typename... Args>
    std::pair<iterator, bool> emplace(const Key& key, Args&&... args) {
        if (const auto iter = find(key); iter != entries.end()) {
            return {iter, false};
        }
        entries.emplace_back(std::piecewise_construct, std::forward_as_tuple(key), std::forward_as_tuple(std::forward<Args>(args)...));
        return {std::prev(entries.end()), true};
    }

private:
    std::vector<value_type> entries;
};

/// A set with few entries, in the order they were added.
template<typename T>
class CompactSet {
public:
    using value_type = T;
    using const_iterator = typename std::vector<T>::const_iterator;

    const_iterator begin() const { return entries.begin(); }
    const_iterator end() const { return entries.end(); }

    bool empty() const { return entries.empty(); }
    std::size_t size() const { return entries.size(); }
    void clear() { entries.clear(); }

    bool contains(const T& value) const {
        return std::find(entries.begin(), entries.end(), value) != entries.end();
    }

    /// Adds the value unless it is already there. Returns whether it was added.
    bool insert(const T& value) {
        if (contains(value)) {
            return false;
        }
        entries.push_back(value);
        return true;
    }

private:
    std::vector<T> entries;
};

}  // namespace Dynarmic::Backend::Arm64
