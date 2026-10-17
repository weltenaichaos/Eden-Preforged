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
    /// Share of the game file shader pairs to compile, in percent, most likely first. Compiled
    /// pairs are kept on storage and read when the game draws with them.
    unsigned precompile_percent = 100;
    /// Hands the driver a pair's stored code even when the game draws it with another state
    /// than it was compiled with, in case the driver can reuse part of it.
    bool store_any_state = false;
    /// Logs memory use every 30 seconds in game.
    bool log_memory = true;
    /// Logs every frame that took more than twice as long as the frame before it.
    bool log_slow_frames = true;
    /// Logs which allocation sizes fill the malloc heap and which mappings the app holds, each
    /// time the heap grows past another 2 GiB. Walking the heap holds up the emulator a moment.
    bool log_allocation_sizes = true;
    /// Records a sample of the heap allocations with the code that made them, and logs which
    /// code holds the most memory with every memory log. Costs a little speed.
    bool track_allocations = true;
};

[[nodiscard]] DebugOptions& Options();

/// Takes the switches from the settings and logs them. Call when a game starts.
void LoadOptions();

/// Logs how many memory mappings the app holds and which files or heaps hold the most, since
/// Android ends the app once it has about 65,000. Shows the top `top` names.
void LogMappingSummary(size_t top);

/// Logs resident, virtual and malloc memory, tagged with `when`, and the number of mappings
/// when `count_mappings` is set, which briefly holds up other threads.
void LogMemoryUse(const char* when, bool count_mappings = true);

/// Logs the allocation sizes that hold the most malloc memory, tagged with `when`. Walks the
/// whole heap, which holds up every thread that allocates while it runs.
void LogAllocationSizes(const char* when);

} // namespace Common::Preforged
