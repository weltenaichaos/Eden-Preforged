// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "common/common_types.h"

// Reads the .pak files of Capcom's RE Engine (Monster Hunter Rise, Ghost Trick, Apollo Justice,
// Capcom Arcade Stadium on Switch): a "KPKA" header, then a table of files by name hash with
// where each is and how it is packed (stored, deflate without a header, or zstd).
namespace Preforged::RePak {

/// Reads `size` bytes at `offset` of the pak file. Returns fewer at the end of the file.
using ReadFn = std::function<std::vector<u8>(u64 offset, u64 size)>;

[[nodiscard]] bool IsPak(std::span<const u8> start);

struct Entry {
    u64 hash = 0; ///< Of the file's path, which the pak doesn't keep
    u64 offset = 0;
    u64 packed_size = 0;
    u64 size = 0;
    u32 compression = 0; ///< 0 stored, 1 deflate, 2 zstd
    u32 encryption = 0;  ///< Not 0 when the file's data is encrypted
};

struct Pak {
    u32 major = 0;
    u32 minor = 0;
    u32 features = 0;
    u32 declared = 0; ///< Files the header says the table has
    u64 table_offset = 0;
    /// Feature 8: the table is encrypted with a key that is itself encrypted (RSA), which
    /// isn't read.
    bool table_encrypted = false;
    std::vector<Entry> entries;
    size_t invalid = 0; ///< Table entries that point past the file
};

/// Reads the header and the table. Returns nothing, with `error` set, when the file isn't a
/// pak or the table doesn't read.
[[nodiscard]] std::optional<Pak> ReadPak(const ReadFn& read, u64 file_size, std::string& error);

[[nodiscard]] const char* CompressionName(u32 compression);

} // namespace Preforged::RePak
