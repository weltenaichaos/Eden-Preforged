// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#ifdef __linux__
#include <malloc.h>
#include <unistd.h>
#endif
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/logging.h"
#include "common/preforged_debug.h"
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
    options.log_memory = Settings::values.preforged_log_memory.GetValue();
    options.log_slow_frames = Settings::values.preforged_log_slow_frames.GetValue();
    options.log_allocation_sizes = Settings::values.preforged_log_allocation_sizes.GetValue();
    options.track_allocations = Settings::values.preforged_track_allocations.GetValue();
    LOG_INFO(Common,
             "[Preforged] switches: bake constants {}, scan {}, rescan {}, pre-compile {}, "
             "log compiles {}",
             options.bake_driver_constants, options.scan_game_files, options.rescan_game_files,
             options.precompile_game_shaders, options.log_shader_compiles);
    LOG_INFO(Common,
             "[Preforged] pre-compile share {}%, stored code for any state {}, log memory {}, "
             "log slow frames {}",
             options.precompile_percent, options.store_any_state, options.log_memory,
             options.log_slow_frames);
    LOG_INFO(Common,
             "[Preforged] driver pipeline cache {}, log allocation sizes {}, track allocations {}",
             Settings::values.use_vulkan_driver_pipeline_cache.GetValue(),
             options.log_allocation_sizes, options.track_allocations);
    if (options.track_allocations) {
        HeapProfile::Start();
    } else if (HeapProfile::Sampling()) {
        HeapProfile::Stop();
    }
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
    // Counting mappings reads the whole mapping list, which holds up threads that change memory
    // mappings for up to a few hundred milliseconds, so it is left out during play.
    std::string mappings{"not counted"};
    if (count_mappings) {
        size_t count = 0;
        if (std::FILE* maps = std::fopen("/proc/self/maps", "r")) {
            for (int c; (c = std::fgetc(maps)) != EOF;) {
                count += c == '\n' ? 1 : 0;
            }
            std::fclose(maps);
        }
        mappings = std::to_string(count);
    }
    const struct mallinfo heap = mallinfo();
    const long page_size = sysconf(_SC_PAGESIZE);
    LOG_INFO(Common,
             "[Preforged] memory {}: {} MiB resident, {} MiB virtual, {} MiB in malloc, {} "
             "mappings",
             when, pages_resident * page_size >> 20, pages_total * page_size >> 20,
             static_cast<size_t>(heap.uordblks) >> 20, mappings);
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

void LogAllocationSizes(const char* when) {
#ifdef __ANDROID__
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
