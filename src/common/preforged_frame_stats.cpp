// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <numeric>

#include <fmt/format.h>

#include "common/preforged_frame_stats.h"

namespace Common::Preforged::FrameStats {
namespace {

struct Counters {
    std::atomic<uint64_t> count{0};
    std::atomic<uint64_t> nanoseconds{0};
    std::atomic<uint64_t> longest{0};
    std::atomic<uint64_t> bytes{0};
};

std::atomic_bool enabled{false};
std::array<Counters, ITEM_COUNT> counters;

constexpr std::array<const char*, ITEM_COUNT> NAMES{
    "CPU code compiled",
    "CPU code cache cleared",
    "shaders translated",
    "pipelines compiled by the driver",
    "draws waited for a pipeline",
    "waits for the GPU",
    "waits for the GPU command worker",
    "game waits on GPU syncpoints",
    "texture uploads",
    "images created",
    "texture cache cleanups",
    "waits for a free frame",
    "swapchain presents",
};

} // Anonymous namespace

void SetEnabled(bool enabled_) {
    enabled.store(enabled_, std::memory_order_relaxed);
    (void)Take();
}

bool Enabled() {
    return enabled.load(std::memory_order_relaxed);
}

void Add(Item item, std::chrono::steady_clock::duration time, uint64_t bytes) {
    if (item >= Item::Count || !Enabled()) {
        return;
    }
    Counters& entry = counters[static_cast<size_t>(item)];
    const uint64_t nanoseconds = static_cast<uint64_t>(
        std::max<int64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(time).count(), 0));
    entry.count.fetch_add(1, std::memory_order_relaxed);
    entry.nanoseconds.fetch_add(nanoseconds, std::memory_order_relaxed);
    if (bytes != 0) {
        entry.bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
    uint64_t longest = entry.longest.load(std::memory_order_relaxed);
    while (nanoseconds > longest &&
           !entry.longest.compare_exchange_weak(longest, nanoseconds, std::memory_order_relaxed)) {
    }
}

Snapshot Take() {
    Snapshot snapshot;
    for (size_t index = 0; index < ITEM_COUNT; ++index) {
        Counters& entry = counters[index];
        snapshot.count[index] = entry.count.exchange(0, std::memory_order_relaxed);
        snapshot.nanoseconds[index] = entry.nanoseconds.exchange(0, std::memory_order_relaxed);
        snapshot.longest[index] = entry.longest.exchange(0, std::memory_order_relaxed);
        snapshot.bytes[index] = entry.bytes.exchange(0, std::memory_order_relaxed);
    }
    return snapshot;
}

std::string Describe(const Snapshot& snapshot) {
    std::array<size_t, ITEM_COUNT> order;
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return snapshot.nanoseconds[a] > snapshot.nanoseconds[b];
    });
    std::string text;
    for (const size_t index : order) {
        if (snapshot.count[index] == 0) {
            continue;
        }
        if (!text.empty()) {
            text += ", ";
        }
        text += fmt::format("{} {}x {:.1f} ms (longest {:.1f} ms", NAMES[index],
                            snapshot.count[index], snapshot.nanoseconds[index] / 1e6,
                            snapshot.longest[index] / 1e6);
        if (snapshot.bytes[index] != 0) {
            text += fmt::format(", {:.1f} MiB", snapshot.bytes[index] / 1048576.0);
        }
        text += ')';
    }
    return text.empty() ? std::string{"nothing counted"} : text;
}

} // namespace Common::Preforged::FrameStats
