// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

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
    LOG_INFO(Common,
             "[Preforged] switches: bake constants {}, scan {}, rescan {}, pre-compile {}, "
             "log compiles {}",
             options.bake_driver_constants, options.scan_game_files, options.rescan_game_files,
             options.precompile_game_shaders, options.log_shader_compiles);
}

} // namespace Common::Preforged
