// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <functional>

#include "common/common_types.h"

namespace Kernel {
class KProcess;
}

namespace Core {
class System;
}

/// Compiles the CPU code a game ran in earlier sessions while it loads, so it does not stutter
/// compiling that code during play. The JIT keeps nothing between sessions, so this records the
/// blocks each CPU core compiles to a list next to the shader cache, then compiles that list on
/// the next start. Only for 64-bit games that run through Dynarmic, not natively (NCE).
namespace Core::Preforged::CpuCode {

/// Reads the list and starts compiling it on one thread per CPU core, and starts recording the
/// blocks the game compiles from now on. Call after the game loaded and before it starts; each
/// core waits for its compile before it runs.
void Start(System& system);

/// Waits for the compile Start began, calling `report(done, total)` about every 100 ms while it
/// runs, then logs what it compiled.
void Wait(const std::function<void(std::size_t done, std::size_t total)>& report);

/// Notes a block a CPU core compiled. Called by the JIT.
void NoteBlock(const Kernel::KProcess* process, std::size_t core, u64 location);

/// Cancels a compile that still runs, writes the rest of the recorded list and stops recording.
/// Call before the game shuts down.
void Stop();

} // namespace Core::Preforged::CpuCode
