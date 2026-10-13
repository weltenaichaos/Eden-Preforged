// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>

namespace Common::Preforged {

/// Switches for testing Eden Preforged without a new build, set in the debug settings.
struct DebugOptions {
    /// Bakes the values of constant buffer 1 into shaders, as upstream Eden does. Off makes
    /// shaders built from the game files match the game's own.
    bool bake_driver_constants = false;
    /// Searches the game files for shaders once per game.
    bool scan_game_files = true;
    /// Searches the game files again on every start while on.
    bool rescan_game_files = false;
    /// Compiles the shader pairs found in the game files while the game loads.
    bool precompile_game_shaders = true;
    /// Logs every shader compile during play, with its timing and where its shaders came from.
    bool log_shader_compiles = true;
};

[[nodiscard]] DebugOptions& Options();

/// Takes the switches from the settings and logs them. Call when a game starts.
void LoadOptions();

/// Logs how many memory mappings the app holds and which files or heaps hold the most, since
/// Android ends the app once it has about 65,000. Shows the top `top` names.
void LogMappingSummary(size_t top);

} // namespace Common::Preforged
