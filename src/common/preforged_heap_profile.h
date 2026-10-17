// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>

namespace Common::Preforged::HeapProfile {

/// Starts recording a random sample of the C++ heap allocations made from now on, each with the
/// call stack that made it, so the log can show which code holds the memory. Android only; does
/// nothing elsewhere.
void Start();

/// Stops sampling new allocations. Allocations already sampled are still followed until freed.
void Stop();

/// Whether new allocations are being sampled.
[[nodiscard]] bool Sampling();

/// Logs the `top` call stacks that hold the most sampled heap memory, tagged with `when`.
void Log(const char* when, size_t top);

} // namespace Common::Preforged::HeapProfile
