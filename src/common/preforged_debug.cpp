// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdio>
#include <cstring>
#ifdef __linux__
#include <unistd.h>
#endif
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/logging.h"
#include "common/preforged_debug.h"
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
    options.log_memory = Settings::values.preforged_log_memory.GetValue();
    options.log_slow_frames = Settings::values.preforged_log_slow_frames.GetValue();
    LOG_INFO(Common,
             "[Preforged] switches: bake constants {}, scan {}, rescan {}, pre-compile {}, "
             "log compiles {}",
             options.bake_driver_constants, options.scan_game_files, options.rescan_game_files,
             options.precompile_game_shaders, options.log_shader_compiles);
    LOG_INFO(Common, "[Preforged] pre-compile share {}%, log memory {}, log slow frames {}",
             options.precompile_percent, options.log_memory, options.log_slow_frames);
    LOG_INFO(Common, "[Preforged] driver pipeline cache {}",
             Settings::values.use_vulkan_driver_pipeline_cache.GetValue());
}

void LogMemoryUse(const char* when) {
#ifdef __linux__
    long pages_total = 0;
    long pages_resident = 0;
    if (std::FILE* statm = std::fopen("/proc/self/statm", "r")) {
        if (std::fscanf(statm, "%ld %ld", &pages_total, &pages_resident) != 2) {
            pages_resident = 0;
        }
        std::fclose(statm);
    }
    size_t mappings = 0;
    if (std::FILE* maps = std::fopen("/proc/self/maps", "r")) {
        for (int c; (c = std::fgetc(maps)) != EOF;) {
            mappings += c == '\n' ? 1 : 0;
        }
        std::fclose(maps);
    }
    const long page_size = sysconf(_SC_PAGESIZE);
    LOG_INFO(Common, "[Preforged] memory {}: {} MiB resident, {} MiB virtual, {} mappings", when,
             pages_resident * page_size >> 20, pages_total * page_size >> 20, mappings);
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
