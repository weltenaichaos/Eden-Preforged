// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>
#include <string>
#include <vector>

#include "common/common_types.h"

namespace Common {
class DynamicLibrary;
}

namespace Preforged {

/// A function a library takes from another library in a named version, like qsort_r@LIBC_36.
struct VersionedImport {
    std::string file;
    std::string version;
    std::string name;
};

/// Lists the functions an ELF library takes in a named version. Weak ones are left out, since a
/// missing weak function doesn't stop the library from loading. Returns nothing for a file that
/// isn't a 64-bit ELF library.
[[nodiscard]] std::vector<VersionedImport> ListVersionedImports(std::span<const u8> elf);

/// adrenotools only loads a custom driver when Vulkan starts, and uses the system driver without
/// a word when the custom one fails to load. This starts Vulkan through the library adrenotools
/// returned, logs which GPU driver answers, and if the custom driver file never got loaded, logs
/// why: functions it needs that this Android doesn't have, and the loader's own error.
void LogCustomDriverCheck(const Common::DynamicLibrary& library, const std::string& driver_dir,
                          const std::string& driver_name);

} // namespace Preforged
