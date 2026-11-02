// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "common/common_types.h"

// Reads Game Freak's data.trpfs (Pokémon Scarlet and Violet): an "ONEPACK" header pointing at
// a flatbuffer table of packs by name hash and offset. Each pack is a flatbuffer of files, each
// stored or packed with Oodle, with its unpacked size.
namespace Preforged::Trpfs {

/// Reads `size` bytes at `offset` of the file. Returns fewer at the end of the file.
using ReadFn = std::function<std::vector<u8>(u64 offset, u64 size)>;

[[nodiscard]] bool IsTrpfs(std::span<const u8> start);

struct Pack {
    u64 hash = 0;
    u64 offset = 0;
    u64 size = 0; ///< Up to the next pack, or the table for the last
};

struct FileSystem {
    u64 table_offset = 0;
    std::vector<Pack> packs;
};

[[nodiscard]] std::optional<FileSystem> ReadFileSystem(const ReadFn& read, u64 file_size,
                                                       std::string& error);

struct PackedFile {
    u64 hash = 0;
    s32 compression = -1; ///< -1 stored, 3 Oodle
    u64 size = 0;         ///< Unpacked
    std::span<const u8> data;
};

/// Reads the files of one pack, whose bytes are `pack`. The files' data point into `pack`.
[[nodiscard]] bool ReadPack(std::span<const u8> pack, std::vector<PackedFile>& files,
                            std::string& error);

} // namespace Preforged::Trpfs
