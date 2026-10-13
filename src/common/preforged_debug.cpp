// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <fstream>
#include <string>
#include <string_view>

#include "common/logging.h"
#include "common/preforged_debug.h"

namespace Common::Preforged {

namespace {
struct Entry {
    std::string_view key;
    bool DebugOptions::*value;
    std::string_view help;
};

constexpr Entry ENTRIES[] = {
    {"bake_driver_constants", &DebugOptions::bake_driver_constants,
     "Bake constant buffer 1 into shaders like upstream Eden (1) or not (0)"},
    {"scan_game_files", &DebugOptions::scan_game_files,
     "Search the game files for shaders once per game"},
    {"rescan_game_files", &DebugOptions::rescan_game_files,
     "Search the game files again on every start while 1"},
    {"precompile_game_shaders", &DebugOptions::precompile_game_shaders,
     "Compile shader pairs from the game files while the game loads"},
    {"log_shader_compiles", &DebugOptions::log_shader_compiles,
     "Log every shader compile during play"},
};

std::string_view Trim(std::string_view text) {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) {
        text.remove_prefix(1);
    }
    while (!text.empty() &&
           (text.back() == ' ' || text.back() == '\t' || text.back() == '\r')) {
        text.remove_suffix(1);
    }
    return text;
}
} // Anonymous namespace

DebugOptions& Options() {
    static DebugOptions options;
    return options;
}

void LoadOptions(const std::filesystem::path& path) {
    DebugOptions& options = Options();
    std::ifstream file{path};
    if (!file) {
        std::ofstream out{path};
        out << "# Eden Preforged test switches. Set a value to 1 or 0, then restart the game.\n";
        for (const Entry& entry : ENTRIES) {
            out << "\n# " << entry.help << "\n" << entry.key << " = "
                << (options.*entry.value ? 1 : 0) << "\n";
        }
    } else {
        std::string line;
        while (std::getline(file, line)) {
            const std::string_view text = Trim(line);
            const size_t equals = text.find('=');
            if (text.empty() || text.front() == '#' || equals == std::string_view::npos) {
                continue;
            }
            const std::string_view key = Trim(text.substr(0, equals));
            const std::string_view value = Trim(text.substr(equals + 1));
            for (const Entry& entry : ENTRIES) {
                if (entry.key == key) {
                    options.*entry.value = value == "1" || value == "true" || value == "on";
                }
            }
        }
    }
    for (const Entry& entry : ENTRIES) {
        LOG_INFO(Common, "[Preforged] option {} = {}", entry.key, options.*entry.value ? 1 : 0);
    }
}

} // namespace Common::Preforged
