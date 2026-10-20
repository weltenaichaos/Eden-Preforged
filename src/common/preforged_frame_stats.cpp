// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <numeric>
#include <utility>

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

struct Active {
    std::atomic<int64_t> running{0};
    /// When the item last started running while nothing else of it ran, in steady clock
    /// nanoseconds.
    std::atomic<int64_t> since{0};
};

std::atomic_bool enabled{false};
std::array<Counters, ITEM_COUNT> counters;
std::array<Active, ITEM_COUNT> active;

int64_t NowNanoseconds() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

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
    "GPU waits on a game semaphore",
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

void Begin(Item item) {
    if (item >= Item::Count) {
        return;
    }
    Active& entry = active[static_cast<size_t>(item)];
    if (entry.running.fetch_add(1, std::memory_order_acq_rel) == 0) {
        entry.since.store(NowNanoseconds(), std::memory_order_release);
    }
}

void End(Item item) {
    if (item >= Item::Count) {
        return;
    }
    active[static_cast<size_t>(item)].running.fetch_sub(1, std::memory_order_acq_rel);
}

std::string DescribeActive() {
    const int64_t now = NowNanoseconds();
    std::array<std::pair<int64_t, size_t>, ITEM_COUNT> running{};
    size_t count = 0;
    for (size_t index = 0; index < ITEM_COUNT; ++index) {
        const int64_t threads = active[index].running.load(std::memory_order_acquire);
        if (threads <= 0) {
            continue;
        }
        running[count++] = {now - active[index].since.load(std::memory_order_acquire), index};
    }
    std::sort(running.begin(), running.begin() + count, std::greater<>{});
    std::string text;
    for (size_t i = 0; i < count; ++i) {
        const size_t index = running[i].second;
        text += fmt::format("{}{} on {} thread{} for {:.1f} s", text.empty() ? "" : ", ",
                            NAMES[index], active[index].running.load(std::memory_order_relaxed),
                            active[index].running.load(std::memory_order_relaxed) == 1 ? "" : "s",
                            running[i].first / 1e9);
    }
    return text.empty() ? std::string{"nothing that is timed"} : text;
}

} // namespace Common::Preforged::FrameStats
