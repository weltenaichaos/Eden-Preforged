// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <stop_token>

#include "common/common_types.h"

namespace Core {
class System;
}

/// Searches the game's files for the shaders stored in its pipeline cache and logs how many of
/// them can be found there, to measure whether shaders can be compiled from game files.
/// Must run before the game starts, and runs once per game, within a time limit.
void RunShaderFileScan(Core::System& system, u64 program_id, std::stop_token stop);
