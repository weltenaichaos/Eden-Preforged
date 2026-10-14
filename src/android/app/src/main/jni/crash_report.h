// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>

/// On a crash, appends the signal, the crashing thread and a backtrace to the log file, then lets
/// the previous handler (or the system) handle the crash as before.
void InstallCrashReport(const std::filesystem::path& log_file);
