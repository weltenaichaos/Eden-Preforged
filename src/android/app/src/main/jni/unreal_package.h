// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "common/common_types.h"

// Unpacks the compressed parts of Unreal Engine packages, so the shader file scan can search
// what games built on the engine store in them. Unreal Engine 3 compresses a package either
// whole (console packages, often ".xxx") or in chunks listed in the package header. Both write
// each compressed part the same way: a tag, the block size, the packed and unpacked sizes, a
// size pair per block, then the blocks. Unreal Engine 4 writes the same with 64-bit fields.
namespace Preforged::Unreal {

/// How a compressed block is packed.
enum class Method : u8 {
    Zlib,
    Lzo,
    Lz4,
    Zstd,
    Stored,      ///< Packed size equals unpacked size: copied as it is
    Oodle,       ///< Oodle Kraken, Mermaid, Selkie or Leviathan (unpacked with ooz)
    OodleFailed, ///< Starts like Oodle, but did not unpack
    Unknown,     ///< None of the above
    Count,
};

constexpr size_t METHOD_COUNT = static_cast<size_t>(Method::Count);

[[nodiscard]] const char* MethodName(Method method);

/// Names the Oodle decoder numbered in the second byte of an Oodle block.
[[nodiscard]] const char* OodleCompressorName(size_t compressor);

struct Stats {
    size_t packages = 0;           ///< Buffers that start with the package tag
    size_t compressed_packages = 0; ///< Packages with at least one compressed part
    size_t parts = 0;              ///< Compressed parts found
    std::array<size_t, METHOD_COUNT> blocks{};
    size_t failed_blocks = 0;      ///< Blocks whose method was known but did not unpack
    u64 packed_bytes = 0;
    u64 unpacked_bytes = 0;
    /// Time spent unpacking blocks, without searching what they unpacked to.
    std::chrono::nanoseconds unpack_time{};
    /// The first bytes of a few blocks no method fit, as hex, to tell what packs them.
    std::vector<std::string> unknown_examples;
    /// The first bytes of a few package headers, as hex.
    std::vector<std::string> header_examples;
    /// Oodle blocks by the decoder named in their second byte (Kraken 6, Mermaid and Selkie 10,
    /// Leviathan 12, ...), whether they unpacked or not.
    std::array<size_t, 16> oodle_compressors{};
    /// The sizes and first bytes of a few Oodle blocks that did not unpack, as hex.
    std::vector<std::string> oodle_examples;
};

/// Returns true when the data starts with the Unreal package tag.
[[nodiscard]] bool IsPackage(std::span<const u8> data);

/// Finds the compressed parts of an Unreal package and unpacks them in file order. Hands the
/// unpacked data to `consume` in pieces of about `piece_size` bytes, each starting with the last
/// `overlap` bytes of the piece before it, with the unpacked offset of the piece's first byte.
/// Returns false when the package has no compressed part. With `unpack_oodle` false, Oodle
/// blocks are counted but left packed.
bool Unpack(std::span<const u8> data, size_t piece_size, size_t overlap, Stats& stats,
            const std::function<bool(std::span<const u8> piece, u64 offset)>& consume,
            bool unpack_oodle = true);

/// Unpacks LZO1X data into exactly `out.size()` bytes. Returns false when the data is not
/// LZO1X data of that size. Exposed for tests.
[[nodiscard]] bool DecompressLzo1x(std::span<const u8> in, std::span<u8> out);

} // namespace Preforged::Unreal
