// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "common/common_types.h"

// Reads Game Freak's .gfpak archives (Pokémon Let's Go, Sword and Shield, Legends: Arceus): a
// "GFLXPACK" header, then a table of files with where each is and how it is packed (stored,
// zlib, LZ4 blocks or Oodle).
namespace Preforged::GfPak {

/// Reads `size` bytes at `offset` of the archive. Returns fewer at the end of the file.
using ReadFn = std::function<std::vector<u8>(u64 offset, u64 size)>;

[[nodiscard]] bool IsPak(std::span<const u8> start);

struct Entry {
    u64 hash = 0; ///< Of the file's path, from the archive's hash list
    u64 offset = 0;
    u64 packed_size = 0;
    u64 size = 0;
    u32 compression = 0; ///< 0 stored, 1 zlib, 2 LZ4, 3 Oodle
};

struct Pak {
    u32 version = 0;
    u32 declared = 0;
    std::vector<Entry> entries;
    size_t invalid = 0; ///< Table entries that point past the file
};

[[nodiscard]] std::optional<Pak> ReadPak(const ReadFn& read, u64 file_size, std::string& error);

/// The method name Preforged::Unreal::UnpackBlock takes for a compression number.
[[nodiscard]] const char* MethodName(u32 compression);

} // namespace Preforged::GfPak
