// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

namespace Common::Preforged::FrameStats {

/// Work that can hold up a frame, counted between two frames so a slow frame can show what ran
/// during it.
enum class Item : unsigned char {
    CpuCode,         ///< The CPU JIT compiling guest code
    CpuCodeClear,    ///< The CPU JIT throwing away its code cache when it is full
    ShaderTranslate, ///< Translating guest shaders on the GPU thread
    ShaderDriver,    ///< The driver compiling pipelines, on the shader worker threads
    PipelineWait,    ///< Drawing waiting for a pipeline that is still being compiled
    GpuWait,         ///< Waiting for the GPU to finish work
    WorkerWait,      ///< The GPU thread waiting for its command worker
    SyncpointWait,   ///< The game waiting on a GPU syncpoint
    TextureUpload,   ///< Copying guest texture data into images
    ImageCreate,     ///< Creating images in the texture cache, with overlapping ones merged
    TextureCleanup,  ///< The texture cache's garbage collector
    FrameWait,       ///< The GPU thread waiting for a free frame to draw into
    Present,         ///< Acquiring and presenting swapchain images, on the present thread
    SemaphoreAcquire, ///< The GPU thread waiting for a game semaphore to reach a value
    Count,
};

constexpr size_t ITEM_COUNT = static_cast<size_t>(Item::Count);

/// What was counted for each item over one frame.
struct Snapshot {
    std::array<uint64_t, ITEM_COUNT> count{};
    std::array<uint64_t, ITEM_COUNT> nanoseconds{};
    std::array<uint64_t, ITEM_COUNT> longest{};
    std::array<uint64_t, ITEM_COUNT> bytes{};
};

/// Turns counting on or off. Called when a game starts, from the "Log slow frames" switch.
void SetEnabled(bool enabled);

[[nodiscard]] bool Enabled();

/// Counts one event of `item` that took `time`, with `bytes` of data where that applies.
/// Item::Count counts nothing.
void Add(Item item, std::chrono::steady_clock::duration time, uint64_t bytes = 0);

/// Returns what was counted since the last call and starts over.
[[nodiscard]] Snapshot Take();

/// Describes a snapshot in one line, most time first, such as
/// "CPU code compiled 120x 35.2 ms (longest 4.1 ms)".
[[nodiscard]] std::string Describe(const Snapshot& snapshot);

/// Marks an item as running on this thread, so DescribeActive can tell what a freeze waits on.
void Begin(Item item);
void End(Item item);

/// Describes the items running right now and how long each has been running, oldest first.
[[nodiscard]] std::string DescribeActive();

/// Counts the time from construction to destruction as one event of `item`. Item::Count counts
/// nothing.
class Timer {
public:
    explicit Timer(Item item_, uint64_t bytes_ = 0)
        : item{item_}, bytes{bytes_}, running{item_ != Item::Count && Enabled()} {
        if (running) {
            start = std::chrono::steady_clock::now();
            Begin(item);
        }
    }

    ~Timer() {
        if (running) {
            End(item);
            Add(item, std::chrono::steady_clock::now() - start, bytes);
        }
    }

    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;

private:
    Item item;
    uint64_t bytes;
    bool running;
    std::chrono::steady_clock::time_point start{};
};

} // namespace Common::Preforged::FrameStats
