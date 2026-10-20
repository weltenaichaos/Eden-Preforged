// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#ifdef __linux__
#include <fcntl.h>
#include <malloc.h>
#include <unistd.h>
#endif
#ifdef __ANDROID__
#include <cxxabi.h>
#include <dlfcn.h>
#include <link.h>
#endif
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/logging.h"
#include "common/preforged_debug.h"
#include "common/preforged_frame_stats.h"
#include "common/preforged_heap_profile.h"
#include "common/settings.h"

namespace Common::Preforged {

DebugOptions& Options() {
    static DebugOptions options;
    return options;
}

void LoadOptions() {
    DebugOptions& options = Options();
    options.bake_driver_constants = Settings::values.preforged_bake_driver_constants.GetValue();
    options.scan_game_files = Settings::values.preforged_scan_game_files.GetValue();
    options.rescan_game_files = Settings::values.preforged_rescan_game_files.GetValue();
    options.precompile_game_shaders =
        Settings::values.preforged_precompile_game_shaders.GetValue();
    options.log_shader_compiles = Settings::values.preforged_log_shader_compiles.GetValue();
    options.precompile_percent =
        std::min(Settings::values.preforged_precompile_percent.GetValue(), 100u);
    options.store_any_state = Settings::values.preforged_store_any_state.GetValue();
    options.input_types_per_shader =
        Settings::values.preforged_input_types_per_shader.GetValue();
    options.log_memory = Settings::values.preforged_log_memory.GetValue();
    options.log_slow_frames = Settings::values.preforged_log_slow_frames.GetValue();
    options.log_allocation_sizes = Settings::values.preforged_log_allocation_sizes.GetValue();
    options.track_allocations = Settings::values.preforged_track_allocations.GetValue();
    options.precompile_cpu_code = Settings::values.preforged_precompile_cpu_code.GetValue();
    options.start_after_loading = Settings::values.preforged_start_after_loading.GetValue();
    options.limit_gpu_lag = Settings::values.preforged_limit_gpu_lag.GetValue();
    options.merge_gpu_work = Settings::values.preforged_merge_gpu_work.GetValue();
    options.unpack_unreal_packages = Settings::values.preforged_unpack_unreal_packages.GetValue();
    options.unpack_oodle = Settings::values.preforged_unpack_oodle.GetValue();
    options.read_unity_files = Settings::values.preforged_read_unity_files.GetValue();
    options.search_program = Settings::values.preforged_search_program.GetValue();
    options.zstd_dictionaries = Settings::values.preforged_zstd_dictionaries.GetValue();
    options.log_unreal_links = Settings::values.preforged_log_unreal_links.GetValue();
    options.unreal_learned_pairs = Settings::values.preforged_unreal_learned_pairs.GetValue();
    options.pipeline_libraries = Settings::values.preforged_pipeline_libraries.GetValue();
    options.optimize_joined = Settings::values.preforged_optimize_joined.GetValue();
    options.join_every_pipeline = Settings::values.preforged_join_every_pipeline.GetValue();
    FrameStats::SetEnabled(options.log_slow_frames);
    LOG_INFO(Common,
             "[Preforged] switches: bake constants {}, scan {}, rescan {}, pre-compile {}, "
             "log compiles {}",
             options.bake_driver_constants, options.scan_game_files, options.rescan_game_files,
             options.precompile_game_shaders, options.log_shader_compiles);
    LOG_INFO(Common,
             "[Preforged] pre-compile share {}%, stored code for any state {}, attribute types "
             "per vertex shader {}, log memory {}, log slow frames {}",
             options.precompile_percent, options.store_any_state, options.input_types_per_shader,
             options.log_memory, options.log_slow_frames);
    LOG_INFO(Common,
             "[Preforged] driver pipeline cache {}, log allocation sizes {}, track allocations {}, "
             "track driver memory {}, pre-compile CPU code {}, start after loading {}",
             Settings::values.use_vulkan_driver_pipeline_cache.GetValue(),
             options.log_allocation_sizes, options.track_allocations,
             Settings::values.preforged_track_driver_memory.GetValue(),
             options.precompile_cpu_code, options.start_after_loading);
    LOG_INFO(Common,
             "[Preforged] limit GPU lag {} (at most {} submits), merge GPU work {} (from {} "
             "submits left), unpack Unreal packages {}, unpack Oodle blocks {}, zstd "
             "dictionaries {}, log Unreal shader links {}, pre-compile Unreal pairs learned "
             "from played ones {}",
             options.limit_gpu_lag, MaxGpuLag, options.merge_gpu_work, MergeGpuLag,
             options.unpack_unreal_packages, options.unpack_oodle, options.zstd_dictionaries,
             options.log_unreal_links, options.unreal_learned_pairs);
    LOG_INFO(Common,
             "[Preforged] join pipelines from shader libraries {}, compile joined pipelines in "
             "full afterwards {}, test: join every pipeline {}, read Unity files {}, search "
             "the game's program {}",
             options.pipeline_libraries, options.optimize_joined, options.join_every_pipeline,
             options.read_unity_files, options.search_program);
    if (options.track_allocations) {
        HeapProfile::Start();
    } else if (HeapProfile::Sampling()) {
        HeapProfile::Stop();
    }
}

namespace {

constexpr std::size_t MaxMemoryTypes = 32;

struct GpuMemoryCounts {
    std::array<std::atomic<std::uint64_t>, MaxMemoryTypes> live_count{};
    std::array<std::atomic<std::uint64_t>, MaxMemoryTypes> live_bytes{};
    std::atomic<std::uint64_t> made{0};
    std::atomic<std::uint64_t> freed{0};
};

GpuMemoryCounts& GpuMemory() {
    static GpuMemoryCounts counts;
    return counts;
}

std::string DescribeGpuMemory() {
    const GpuMemoryCounts& counts = GpuMemory();
    std::uint64_t count = 0;
    std::uint64_t bytes = 0;
    std::string types;
    for (std::size_t type = 0; type < MaxMemoryTypes; ++type) {
        const std::uint64_t type_count = counts.live_count[type].load(std::memory_order_relaxed);
        if (type_count == 0) {
            continue;
        }
        const std::uint64_t type_bytes = counts.live_bytes[type].load(std::memory_order_relaxed);
        count += type_count;
        bytes += type_bytes;
        types += fmt::format("{}type {} {}x {} MiB", types.empty() ? "" : ", ", type, type_count,
                             type_bytes >> 20);
    }
    return fmt::format("{} allocations, {} MiB ({}); {} made and {} freed since the app started",
                       count, bytes >> 20, types.empty() ? "none" : types,
                       counts.made.load(std::memory_order_relaxed),
                       counts.freed.load(std::memory_order_relaxed));
}

constexpr std::size_t VulkanObjectTypes = static_cast<std::size_t>(VulkanObject::Count);

constexpr std::array<const char*, VulkanObjectTypes> VulkanObjectNames{
    "command pools",   "command buffers made", "descriptor pools", "descriptor sets made",
    "query pools",     "events",               "fences",           "semaphores",
    "direct memory allocations",               "buffers",          "buffer views",
    "images",          "image views",          "framebuffers",     "render passes",
    "samplers",        "pipelines",            "shader modules",
};

constexpr std::size_t SubmitCauses = static_cast<std::size_t>(SubmitCause::Count);

constexpr std::array<const char*, SubmitCauses> SubmitCauseNames{
    "other", "fences", "command lists", "draw batches", "waits for the GPU", "frames",
};

struct VulkanObjectCounts {
    std::array<std::atomic<std::int64_t>, VulkanObjectTypes> live{};
    std::atomic<std::uint64_t> submits{0};
    std::atomic<std::uint64_t> most_ahead{0};
    std::atomic<std::uint64_t> most_waiting{0};
    std::array<std::atomic<std::uint64_t>, SubmitCauses> by_cause{};
    std::atomic<std::uint64_t> merged{0};
    std::atomic<std::uint64_t> empty_skipped{0};
    std::atomic<std::uint64_t> lag_waits{0};
    std::atomic<std::uint64_t> lag_wait_nanoseconds{0};
    std::atomic<std::uint64_t> longest_lag_wait{0};
    /// Freeze report: the GPU thread's last submit and what it knew of the GPU's progress.
    std::atomic<std::uint64_t> last_tick{0};
    std::atomic<std::uint64_t> last_pending{0};
    std::atomic<std::uint64_t> known_tick{0};
    std::atomic<std::int64_t> last_submit_time{0};
    std::atomic<std::int64_t> last_frame_time{0};
    std::atomic<std::uint64_t> waiting_chunks{0};
};

thread_local SubmitCause current_cause = SubmitCause::Other;

std::int64_t NowNanoseconds() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void StoreMax(std::atomic<std::uint64_t>& most, std::uint64_t value) {
    std::uint64_t current = most.load(std::memory_order_relaxed);
    while (value > current &&
           !most.compare_exchange_weak(current, value, std::memory_order_relaxed)) {
    }
}

VulkanObjectCounts& VulkanObjects() {
    static VulkanObjectCounts counts;
    return counts;
}

/// Lists the live Vulkan objects by type with the change since the last call, then the queue
/// submits since the last call and how far the GPU fell behind them.
std::string DescribeVulkanObjects() {
    VulkanObjectCounts& counts = VulkanObjects();
    static std::array<std::int64_t, VulkanObjectTypes> last{};
    static std::uint64_t last_submits = 0;
    std::string text;
    for (std::size_t type = 0; type < VulkanObjectTypes; ++type) {
        const std::int64_t live = counts.live[type].load(std::memory_order_relaxed);
        const std::int64_t change = live - last[type];
        last[type] = live;
        if (live == 0 && change == 0) {
            continue;
        }
        text += fmt::format("{}{} {} ({}{})", text.empty() ? "" : ", ", VulkanObjectNames[type],
                            live, change >= 0 ? "+" : "", change);
    }
    const std::uint64_t submits = counts.submits.load(std::memory_order_relaxed);
    const std::uint64_t ahead = counts.most_ahead.exchange(0, std::memory_order_relaxed);
    const std::uint64_t waiting = counts.most_waiting.exchange(0, std::memory_order_relaxed);
    text += fmt::format("; {} queue submits (+{}), the GPU was up to {} submits behind, up to {} "
                        "command chunks waited to be submitted",
                        submits, submits - last_submits, ahead, waiting);
    last_submits = submits;

    std::string causes;
    for (std::size_t cause = 0; cause < SubmitCauses; ++cause) {
        const std::uint64_t count = counts.by_cause[cause].exchange(0, std::memory_order_relaxed);
        if (count != 0) {
            causes += fmt::format("{}{} {}", causes.empty() ? "" : ", ", SubmitCauseNames[cause],
                                  count);
        }
    }
    const std::uint64_t lag_waits = counts.lag_waits.exchange(0, std::memory_order_relaxed);
    const std::uint64_t lag_ns = counts.lag_wait_nanoseconds.exchange(0, std::memory_order_relaxed);
    const std::uint64_t longest = counts.longest_lag_wait.exchange(0, std::memory_order_relaxed);
    text += fmt::format("; submits since the last log by cause: {}; {} command lists and draw "
                        "batches merged into later submits, {} empty submits left out; waited {} "
                        "times for the GPU to catch up, {:.1f} ms in all (longest {:.1f} ms)",
                        causes.empty() ? std::string{"none"} : causes,
                        counts.merged.exchange(0, std::memory_order_relaxed),
                        counts.empty_skipped.exchange(0, std::memory_order_relaxed), lag_waits,
                        lag_ns / 1e6, longest / 1e6);
    return text;
}

#ifdef __linux__
/// Counts the lines of /proc/self/maps. Reads in small pieces, since the kernel holds the
/// mapping lock while it writes each piece, which holds up threads that map memory.
std::size_t CountMappings() {
    const int fd = open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return 0;
    }
    std::size_t count = 0;
    char buffer[16384];
    for (ssize_t size; (size = read(fd, buffer, sizeof(buffer))) > 0;) {
        count += static_cast<std::size_t>(std::count(buffer, buffer + size, '\n'));
    }
    close(fd);
    return count;
}
#endif

} // Anonymous namespace

void NoteGpuAllocation(std::uint32_t memory_type, std::uint64_t size) {
    GpuMemoryCounts& counts = GpuMemory();
    const std::size_t type = std::min<std::size_t>(memory_type, MaxMemoryTypes - 1);
    counts.live_count[type].fetch_add(1, std::memory_order_relaxed);
    counts.live_bytes[type].fetch_add(size, std::memory_order_relaxed);
    counts.made.fetch_add(1, std::memory_order_relaxed);
}

void NoteGpuFree(std::uint32_t memory_type, std::uint64_t size) {
    GpuMemoryCounts& counts = GpuMemory();
    const std::size_t type = std::min<std::size_t>(memory_type, MaxMemoryTypes - 1);
    counts.live_count[type].fetch_sub(1, std::memory_order_relaxed);
    counts.live_bytes[type].fetch_sub(size, std::memory_order_relaxed);
    counts.freed.fetch_add(1, std::memory_order_relaxed);
}

void NoteVulkanObjects(VulkanObject type, std::int64_t change) {
    const auto index = static_cast<std::size_t>(type);
    if (index < VulkanObjectTypes) {
        VulkanObjects().live[index].fetch_add(change, std::memory_order_relaxed);
    }
}

void NoteQueueSubmit(std::uint64_t ahead) {
    VulkanObjectCounts& counts = VulkanObjects();
    counts.submits.fetch_add(1, std::memory_order_relaxed);
    StoreMax(counts.most_ahead, ahead);
}

void NoteWorkQueue(std::uint64_t waiting) {
    VulkanObjectCounts& counts = VulkanObjects();
    StoreMax(counts.most_waiting, waiting);
    counts.waiting_chunks.store(waiting, std::memory_order_relaxed);
}

SubmitCauseScope::SubmitCauseScope(SubmitCause cause) : previous{current_cause} {
    current_cause = cause;
}

SubmitCauseScope::~SubmitCauseScope() {
    current_cause = previous;
}

void NoteSubmit(std::uint64_t tick, std::uint64_t pending) {
    VulkanObjectCounts& counts = VulkanObjects();
    const auto cause = static_cast<std::size_t>(current_cause);
    counts.by_cause[cause < SubmitCauses ? cause : 0].fetch_add(1, std::memory_order_relaxed);
    counts.last_tick.store(tick, std::memory_order_relaxed);
    counts.last_pending.store(pending, std::memory_order_relaxed);
    counts.last_submit_time.store(NowNanoseconds(), std::memory_order_relaxed);
}

void NoteMergedWork() {
    VulkanObjects().merged.fetch_add(1, std::memory_order_relaxed);
}

void NoteEmptySubmitSkipped() {
    VulkanObjects().empty_skipped.fetch_add(1, std::memory_order_relaxed);
}

void NoteGpuLagWait(std::chrono::steady_clock::duration time) {
    VulkanObjectCounts& counts = VulkanObjects();
    const auto nanoseconds = static_cast<std::uint64_t>(std::max<std::int64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(time).count(), 0));
    counts.lag_waits.fetch_add(1, std::memory_order_relaxed);
    counts.lag_wait_nanoseconds.fetch_add(nanoseconds, std::memory_order_relaxed);
    StoreMax(counts.longest_lag_wait, nanoseconds);
}

void NoteGpuTick(std::uint64_t known_tick) {
    VulkanObjects().known_tick.store(known_tick, std::memory_order_relaxed);
}

void NoteFrame() {
    VulkanObjects().last_frame_time.store(NowNanoseconds(), std::memory_order_relaxed);
}

void ResetFrameClock() {
    VulkanObjects().last_frame_time.store(NowNanoseconds(), std::memory_order_relaxed);
}

double SecondsSinceFrame() {
    const std::int64_t last = VulkanObjects().last_frame_time.load(std::memory_order_relaxed);
    return last == 0 ? -1.0 : (NowNanoseconds() - last) / 1e9;
}

std::string DescribeGpuProgress() {
    const VulkanObjectCounts& counts = VulkanObjects();
    const std::int64_t last_submit = counts.last_submit_time.load(std::memory_order_relaxed);
    return fmt::format(
        "last submit {:.1f} s ago (tick {}, {} earlier submits unfinished then), the GPU thread "
        "last saw tick {} finished, {} command chunks wait for the command worker; running now: "
        "{}",
        last_submit == 0 ? -1.0 : (NowNanoseconds() - last_submit) / 1e9,
        counts.last_tick.load(std::memory_order_relaxed),
        counts.last_pending.load(std::memory_order_relaxed),
        counts.known_tick.load(std::memory_order_relaxed),
        counts.waiting_chunks.load(std::memory_order_relaxed), FrameStats::DescribeActive());
}

void LogMemoryUse(const char* when, bool count_mappings) {
#ifdef __linux__
    long pages_total = 0;
    long pages_resident = 0;
    if (std::FILE* statm = std::fopen("/proc/self/statm", "r")) {
        if (std::fscanf(statm, "%ld %ld", &pages_total, &pages_resident) != 2) {
            pages_resident = 0;
        }
        std::fclose(statm);
    }
    std::string mappings{"not counted"};
    static std::size_t last_count = 0;
    static std::size_t next_summary = 20000;
    std::size_t count = 0;
    if (count_mappings) {
        count = CountMappings();
        mappings = count >= last_count ? fmt::format("{} (+{})", count, count - last_count)
                                       : fmt::format("{} (-{})", count, last_count - count);
        last_count = count;
    }
    const struct mallinfo heap = mallinfo();
    const long page_size = sysconf(_SC_PAGESIZE);
    LOG_INFO(Common,
             "[Preforged] memory {}: {} MiB resident, {} MiB virtual, {} MiB in malloc, {} "
             "mappings; malloc has {} MiB mapped and {} MiB free in its caches",
             when, pages_resident * page_size >> 20, pages_total * page_size >> 20,
             static_cast<size_t>(heap.uordblks) >> 20, mappings,
             static_cast<size_t>(heap.hblkhd) >> 20, static_cast<size_t>(heap.fordblks) >> 20);
    LOG_INFO(Common, "[Preforged] GPU memory {}: {}", when, DescribeGpuMemory());
    LOG_INFO(Common, "[Preforged] Vulkan objects {}: {}", when, DescribeVulkanObjects());
    if (count_mappings && count >= next_summary) {
        next_summary = count + 5000;
        LogMappingSummary(8);
    }
    HeapProfile::Log(when, 6);

    // Each time the heap grows past another 2 GiB, log what fills it.
    static size_t next_report = size_t{2} << 30;
    if (Options().log_allocation_sizes && static_cast<size_t>(heap.uordblks) >= next_report) {
        while (next_report <= static_cast<size_t>(heap.uordblks)) {
            next_report += size_t{2} << 30;
        }
        LogAllocationSizes(when);
        // Large allocations each get their own mapping, and the app ends once it holds about
        // 65,000 of them, so show which mappings it holds at the same moments.
        LogMappingSummary(12);
    }
#endif
}

#ifdef __ANDROID__
namespace {

// Walks every live allocation of Android's allocator with the calls that libmemunreachable
// uses. They are part of libc but not of the NDK's headers, so they are looked up at run time.
using ChunkCallback = void (*)(uintptr_t base, size_t size, void* arg);
using MallocIterate = int (*)(uintptr_t base, size_t size, ChunkCallback callback, void* arg);
using MallocToggle = void (*)();

// Mesa's Vulkan objects start with this value, followed by their VkObjectType.
constexpr uint64_t ICD_LOADER_MAGIC = 0x01CDC0DE;
constexpr uint64_t MESA_OBJECT_TAG = uint64_t{1} << 63;
// Android may keep a tag in the top byte of heap pointers.
constexpr uintptr_t ADDRESS_MASK = (uintptr_t{1} << 56) - 1;

constexpr size_t MAX_HEAP_REGIONS = 256;
constexpr size_t MAX_CODE_RANGES = 4096;
constexpr size_t GROUP_BITS = 16;
constexpr size_t GROUP_CAPACITY = size_t{1} << GROUP_BITS;
constexpr size_t GROUP_PROBES = 64;

struct HeapRegion {
    uintptr_t begin;
    uintptr_t end;
    uintptr_t last;
    uint64_t chunks;
    uint64_t bytes;
    uint64_t min_gap;
};

struct CodeRange {
    uintptr_t begin;
    uintptr_t end;
};

// Allocations of one size whose first word is the same pointer into a loaded library, which
// for C++ objects with virtual functions is their vtable, so it names their type.
struct ChunkGroup {
    uint64_t size;
    uint64_t tag;
    uint64_t count;
    uint64_t page_aligned;
    uint64_t line_aligned;
};

struct HeapSurvey {
    std::array<HeapRegion, MAX_HEAP_REGIONS> regions;
    size_t region_count;
    std::array<CodeRange, MAX_CODE_RANGES> code;
    size_t code_count;
    std::array<ChunkGroup, GROUP_CAPACITY> groups;
    size_t group_count;
    uint64_t chunks;
    uint64_t bytes;
    uint64_t ungrouped_chunks;
    uint64_t ungrouped_bytes;
    uint64_t outside_chunks;
    uint64_t outside_bytes;
};

HeapRegion* FindRegion(HeapSurvey& survey, uintptr_t address) {
    size_t low = 0;
    size_t high = survey.region_count;
    while (low < high) {
        const size_t middle = (low + high) / 2;
        HeapRegion& region = survey.regions[middle];
        if (address < region.begin) {
            high = middle;
        } else if (address >= region.end) {
            low = middle + 1;
        } else {
            return &region;
        }
    }
    return nullptr;
}

bool InCode(const HeapSurvey& survey, uintptr_t address) {
    size_t low = 0;
    size_t high = survey.code_count;
    while (low < high) {
        const size_t middle = (low + high) / 2;
        const CodeRange& range = survey.code[middle];
        if (address < range.begin) {
            high = middle;
        } else if (address >= range.end) {
            low = middle + 1;
        } else {
            return true;
        }
    }
    return false;
}

// Runs while the allocator is stopped, so it must not allocate.
void OnChunk(uintptr_t base, size_t size, void* arg) {
    HeapSurvey& survey = *static_cast<HeapSurvey*>(arg);
    const uintptr_t address = base & ADDRESS_MASK;
    ++survey.chunks;
    survey.bytes += size;
    if (HeapRegion* const region = FindRegion(survey, address)) {
        // The allocator walks each region upwards, so the closest two allocations come at
        // most one block apart.
        if (region->last != 0 && address > region->last) {
            region->min_gap = std::min<uint64_t>(region->min_gap, address - region->last);
        }
        region->last = address;
        ++region->chunks;
        region->bytes += size;
    } else {
        ++survey.outside_chunks;
        survey.outside_bytes += size;
    }

    uint64_t tag = 0;
    if (size >= sizeof(uint64_t)) {
        uint64_t word;
        std::memcpy(&word, reinterpret_cast<const void*>(base), sizeof(word));
        if (word == ICD_LOADER_MAGIC && size >= 2 * sizeof(uint64_t)) {
            uint32_t type;
            std::memcpy(&type, reinterpret_cast<const char*>(base) + 8, sizeof(type));
            tag = MESA_OBJECT_TAG | type;
        } else if (InCode(survey, static_cast<uintptr_t>(word) & ADDRESS_MASK)) {
            tag = word & ADDRESS_MASK;
        }
    }

    const uint64_t hash = (static_cast<uint64_t>(size) * 0x9E3779B97F4A7C15ULL) ^
                          (tag * 0xC2B2AE3D27D4EB4FULL);
    size_t slot = static_cast<size_t>(hash >> (64 - GROUP_BITS));
    for (size_t probe = 0; probe < GROUP_PROBES; ++probe, slot = (slot + 1) % GROUP_CAPACITY) {
        ChunkGroup& group = survey.groups[slot];
        if (group.count == 0) {
            if (survey.group_count >= GROUP_CAPACITY * 3 / 4) {
                break;
            }
            group.size = size;
            group.tag = tag;
            ++survey.group_count;
        } else if (group.size != size || group.tag != tag) {
            continue;
        }
        ++group.count;
        group.page_aligned += (address & 4095) == 0 ? 1 : 0;
        group.line_aligned += (address & 63) == 0 ? 1 : 0;
        return;
    }
    ++survey.ungrouped_chunks;
    survey.ungrouped_bytes += size;
}

int AddCodeRanges(dl_phdr_info* info, size_t, void* arg) {
    HeapSurvey& survey = *static_cast<HeapSurvey*>(arg);
    for (size_t index = 0; index < info->dlpi_phnum; ++index) {
        const ElfW(Phdr)& header = info->dlpi_phdr[index];
        if (header.p_type != PT_LOAD || survey.code_count == MAX_CODE_RANGES) {
            continue;
        }
        const uintptr_t begin = info->dlpi_addr + header.p_vaddr;
        survey.code[survey.code_count++] = CodeRange{begin, begin + header.p_memsz};
    }
    return 0;
}

const char* VulkanObjectName(uint32_t type) {
    static constexpr std::array<const char*, 26> names{
        "unknown",         "instance",         "physical device",    "device",
        "queue",           "semaphore",        "command buffer",     "fence",
        "device memory",   "buffer",           "image",              "event",
        "query pool",      "buffer view",      "image view",         "shader module",
        "pipeline cache",  "pipeline layout",  "render pass",        "pipeline",
        "descriptor set layout", "sampler",    "descriptor pool",    "descriptor set",
        "framebuffer",     "command pool",
    };
    return type < names.size() ? names[type] : nullptr;
}

std::string DescribeTag(uint64_t tag) {
    if (tag == 0) {
        return "no type found";
    }
    if ((tag & MESA_OBJECT_TAG) != 0) {
        const uint32_t type = static_cast<uint32_t>(tag);
        const char* const name = VulkanObjectName(type);
        return name ? fmt::format("Mesa Vulkan {}", name)
                    : fmt::format("Mesa Vulkan object of type {}", type);
    }
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(tag), &info) == 0 || info.dli_fname == nullptr) {
        return fmt::format("starts with {:#x}", tag);
    }
    const char* module = std::strrchr(info.dli_fname, '/');
    module = module ? module + 1 : info.dli_fname;
    std::string text = fmt::format("starts with {}+{:#x}", module,
                                   tag - reinterpret_cast<uintptr_t>(info.dli_fbase));
    if (info.dli_sname != nullptr) {
        int status = 0;
        char* const demangled = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &status);
        std::string name = status == 0 && demangled ? demangled : info.dli_sname;
        std::free(demangled);
        if (name.size() > 240) {
            name.resize(237);
            name += "...";
        }
        text += fmt::format(" ({}+{:#x})", name, tag - reinterpret_cast<uintptr_t>(info.dli_saddr));
    }
    return text;
}

// Lists what every live allocation is, by size, by type where it can tell, and by allocator
// region. Returns false where the allocator cannot be walked.
bool SurveyHeap(const char* when) {
    static const auto iterate =
        reinterpret_cast<MallocIterate>(dlsym(RTLD_DEFAULT, "malloc_iterate"));
    static const auto disable =
        reinterpret_cast<MallocToggle>(dlsym(RTLD_DEFAULT, "malloc_disable"));
    static const auto enable =
        reinterpret_cast<MallocToggle>(dlsym(RTLD_DEFAULT, "malloc_enable"));
    if (!iterate || !disable || !enable) {
        LOG_INFO(Common, "[Preforged] heap survey {}: the allocator cannot be walked here", when);
        return false;
    }
    // Too large for a thread's stack, and zeroed by make_unique.
    const auto survey = std::make_unique<HeapSurvey>();
    HeapSurvey& s = *survey;

    // Everything that allocates happens before the allocator is stopped.
    if (std::FILE* maps = std::fopen("/proc/self/maps", "r")) {
        char line[512];
        while (std::fgets(line, sizeof(line), maps) && s.region_count < MAX_HEAP_REGIONS) {
            unsigned long long begin = 0, end = 0;
            char perms[8]{};
            int name_offset = 0;
            if (std::sscanf(line, "%llx-%llx %7s %*s %*s %*s %n", &begin, &end, perms,
                            &name_offset) < 3 ||
                name_offset <= 0 || perms[0] != 'r' || perms[1] != 'w') {
                continue;
            }
            if (std::strncmp(line + name_offset, "[anon:scudo:primary]", 20) == 0) {
                s.regions[s.region_count++] = HeapRegion{static_cast<uintptr_t>(begin),
                                                         static_cast<uintptr_t>(end), 0, 0, 0,
                                                         UINT64_MAX};
            }
        }
        std::fclose(maps);
    }
    dl_iterate_phdr(AddCodeRanges, &s);
    std::sort(s.code.begin(), s.code.begin() + s.code_count,
              [](const CodeRange& a, const CodeRange& b) { return a.begin < b.begin; });
    const struct mallinfo before = mallinfo();

    const auto start = std::chrono::steady_clock::now();
    disable();
    iterate(0, SIZE_MAX, OnChunk, &s);
    enable();
    const std::chrono::duration<double, std::milli> took = std::chrono::steady_clock::now() - start;

    uint64_t block_bytes = 0;
    for (size_t index = 0; index < s.region_count; ++index) {
        const HeapRegion& region = s.regions[index];
        block_bytes += region.chunks > 1 ? region.chunks * region.min_gap : region.bytes;
    }
    LOG_INFO(Common,
             "[Preforged] heap survey {}: {} allocations asked for {} MiB; the blocks holding the "
             "small ones take about {} MiB, and {} large ones with their own mappings ask for {} "
             "MiB. malloc counts {} MiB allocated and {} MiB free in its caches. Walked in "
             "{:.0f} ms",
             when, s.chunks, s.bytes >> 20, block_bytes >> 20, s.outside_chunks,
             s.outside_bytes >> 20, static_cast<size_t>(before.uordblks) >> 20,
             static_cast<size_t>(before.fordblks) >> 20, took.count());

    // Regions of the small allocation heap, one per block size, most memory first.
    std::vector<HeapRegion> regions(s.regions.begin(), s.regions.begin() + s.region_count);
    std::sort(regions.begin(), regions.end(), [](const HeapRegion& a, const HeapRegion& b) {
        return a.end - a.begin > b.end - b.begin;
    });
    for (size_t index = 0; index < std::min<size_t>(regions.size(), 16); ++index) {
        const HeapRegion& region = regions[index];
        const uint64_t gap = region.chunks > 1 ? region.min_gap : 0;
        LOG_INFO(Common,
                 "[Preforged]   heap region {:#x}: {} MiB mapped, blocks of about {} bytes, {} "
                 "allocations asking for {} MiB, about {} MiB of blocks",
                 region.begin, (region.end - region.begin) >> 20, gap, region.chunks,
                 region.bytes >> 20, (region.chunks * gap) >> 20);
    }

    // Groups by size and type, most memory first, then the ones that grew most since the last
    // survey.
    struct Row {
        uint64_t size;
        uint64_t tag;
        uint64_t count;
        uint64_t page_aligned;
        uint64_t line_aligned;
        uint64_t bytes;
    };
    std::vector<Row> rows;
    rows.reserve(s.group_count);
    for (const ChunkGroup& group : s.groups) {
        if (group.count != 0) {
            rows.push_back(Row{group.size, group.tag, group.count, group.page_aligned,
                               group.line_aligned, group.size * group.count});
        }
    }
    static std::unordered_map<std::string, uint64_t> previous;
    const auto key_of = [](const Row& row) { return fmt::format("{}/{:x}", row.size, row.tag); };
    const auto print = [&](const char* label, size_t rank, const Row& row, int64_t growth) {
        LOG_INFO(Common,
                 "[Preforged]   {} #{}: {} bytes x {} = {} MiB ({:+} MiB since the last survey), "
                 "{}% on 4 KiB, {}% on 64 bytes, {}",
                 label, rank, row.size, row.count, row.bytes >> 20, growth / (int64_t{1} << 20),
                 row.page_aligned * 100 / row.count, row.line_aligned * 100 / row.count,
                 DescribeTag(row.tag));
    };
    const auto growth_of = [&](const Row& row) {
        const auto it = previous.find(key_of(row));
        return static_cast<int64_t>(row.bytes) -
               static_cast<int64_t>(it == previous.end() ? 0 : it->second);
    };
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.bytes > b.bytes; });
    const size_t shown = std::min<size_t>(rows.size(), 24);
    for (size_t index = 0; index < shown; ++index) {
        print("most held", index + 1, rows[index], growth_of(rows[index]));
    }
    if (!previous.empty()) {
        std::vector<std::pair<int64_t, size_t>> growth;
        for (size_t index = shown; index < rows.size(); ++index) {
            growth.emplace_back(growth_of(rows[index]), index);
        }
        std::sort(growth.begin(), growth.end(), std::greater<>{});
        for (size_t index = 0; index < std::min<size_t>(growth.size(), 8); ++index) {
            if (growth[index].first < (int64_t{4} << 20)) {
                break;
            }
            print("grew most", index + 1, rows[growth[index].second], growth[index].first);
        }
    }
    if (s.ungrouped_chunks != 0) {
        LOG_INFO(Common, "[Preforged]   {} allocations asking for {} MiB did not fit the list",
                 s.ungrouped_chunks, s.ungrouped_bytes >> 20);
    }
    previous.clear();
    for (const Row& row : rows) {
        previous.emplace(key_of(row), row.bytes);
    }
    return true;
}

} // Anonymous namespace
#endif

void LogAllocationSizes(const char* when) {
#ifdef __ANDROID__
    if (SurveyHeap(when)) {
        return;
    }
    // Android's allocator lists how many live allocations it holds of each size.
    char* text = nullptr;
    size_t length = 0;
    std::FILE* stream = open_memstream(&text, &length);
    if (!stream) {
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    const int result = malloc_info(0, stream);
    std::fclose(stream);
    const std::chrono::duration<double, std::milli> took = std::chrono::steady_clock::now() - start;
    if (result != 0 || !text) {
        std::free(text);
        LOG_INFO(Common, "[Preforged] allocation sizes {}: not available", when);
        return;
    }
    struct Size {
        size_t size;
        size_t count;
    };
    std::vector<Size> sizes;
    size_t total_bytes = 0;
    size_t total_count = 0;
    for (const char* line = text; (line = std::strstr(line, "<alloc size=\"")) != nullptr;) {
        size_t size = 0;
        size_t count = 0;
        if (std::sscanf(line, "<alloc size=\"%zu\" count=\"%zu\"", &size, &count) == 2) {
            sizes.push_back({size, count});
            total_bytes += size * count;
            total_count += count;
        }
        ++line;
    }
    std::free(text);
    std::sort(sizes.begin(), sizes.end(), [](const Size& a, const Size& b) {
        return a.size * a.count > b.size * b.count;
    });
    LOG_INFO(Common,
             "[Preforged] allocation sizes {}: {} MiB in {} small allocations, listed in {:.0f} ms "
             "(larger allocations have their own mappings)",
             when, total_bytes >> 20, total_count, took.count());
    for (size_t index = 0; index < std::min<size_t>(sizes.size(), 20); ++index) {
        LOG_INFO(Common, "[Preforged]   {} bytes x {} = {} MiB", sizes[index].size,
                 sizes[index].count, (sizes[index].size * sizes[index].count) >> 20);
    }
#else
    LOG_INFO(Common, "[Preforged] allocation sizes {}: not available", when);
#endif
}

void LogMappingSummary(size_t top) {
#ifdef __linux__
    std::FILE* maps = std::fopen("/proc/self/maps", "r");
    if (!maps) {
        return;
    }
    struct Group {
        size_t count = 0;
        size_t bytes = 0;
    };
    std::unordered_map<std::string, Group> groups;
    size_t total = 0;
    char line[512];
    while (std::fgets(line, sizeof(line), maps)) {
        unsigned long long begin = 0, end = 0;
        char perms[8]{};
        int name_offset = 0;
        if (std::sscanf(line, "%llx-%llx %7s %*s %*s %*s %n", &begin, &end, perms,
                        &name_offset) < 3) {
            continue;
        }
        std::string name = name_offset > 0 ? line + name_offset : "";
        while (!name.empty() && (name.back() == '\n' || name.back() == ' ')) {
            name.pop_back();
        }
        if (name.empty()) {
            name = "(anonymous)";
        }
        // Group by permissions too, since protected guest memory shows up as many small
        // pieces of the same mapping.
        name += " ";
        name += perms;
        Group& group = groups[name];
        ++group.count;
        group.bytes += end - begin;
        ++total;
    }
    std::fclose(maps);
    std::vector<std::pair<std::string, Group>> sorted(groups.begin(), groups.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second.count > b.second.count; });
    LOG_INFO(Common, "[Preforged] mappings: {} total, {} different names", total, sorted.size());
    for (size_t i = 0; i < std::min(top, sorted.size()); ++i) {
        LOG_INFO(Common, "[Preforged]   {} mappings, {} MiB: {}", sorted[i].second.count,
                 sorted[i].second.bytes >> 20, sorted[i].first);
    }
#endif
}

} // namespace Common::Preforged
