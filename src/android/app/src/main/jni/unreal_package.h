// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
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

/// A block that did not unpack, and where it is.
struct FailedBlock {
    u64 file_offset;     ///< Where its packed bytes start in the file
    u64 unpacked_offset; ///< Where its bytes would start: in what Unpack hands out, or in the
                         ///< uncompressed layout for UnpackImage
    u32 packed;
    u32 unpacked;
    u8 decoder; ///< The Oodle decoder named in its second byte, or one of the values below
};

/// FailedBlock::decoder of a block that does not start like Oodle.
constexpr u8 FAILED_NOT_OODLE = 0xFF;
/// FailedBlock::decoder of a block the chunk list puts past the end of the layout.
constexpr u8 FAILED_PAST_END = 0xFE;

/// Blocks that did not unpack that one package keeps a record of.
constexpr size_t MAX_FAILED_BLOCKS_KEPT = 256;

/// Returns true when the data starts with the Unreal package tag.
[[nodiscard]] bool IsPackage(std::span<const u8> data);

/// Finds the compressed parts of an Unreal package and unpacks them in file order. Hands the
/// unpacked data to `consume` in pieces of about `piece_size` bytes, each starting with the last
/// `overlap` bytes of the piece before it, with the unpacked offset of the piece's first byte.
/// Returns false when the package has no compressed part. With `unpack_oodle` false, Oodle
/// blocks are counted but left packed. `failures`, when given, gets the blocks that did not
/// unpack (up to MAX_FAILED_BLOCKS_KEPT).
bool Unpack(std::span<const u8> data, size_t piece_size, size_t overlap, Stats& stats,
            const std::function<bool(std::span<const u8> piece, u64 offset)>& consume,
            bool unpack_oodle = true, std::vector<FailedBlock>* failures = nullptr);

/// Unpacks one block into exactly `out.size()` bytes, for Unreal Engine 4 pak files and shader
/// archives, which name their method: "Zlib", "Gzip", "LZ4", "Oodle" or "Zstd" (any case). An
/// empty or other name tries the methods in turn, as for packages. `out` must have
/// UNPACK_SLACK bytes of room after it, for Oodle. Returns the method that fit, or OodleFailed
/// or Unknown when none did.
[[nodiscard]] Method UnpackBlock(std::span<const u8> in, std::span<u8> out,
                                 std::string_view method_name, Method& last_method,
                                 bool unpack_oodle = true);

/// Room UnpackBlock needs after its output.
constexpr size_t UNPACK_SLACK = 64;

/// Unpacks LZO1X data into exactly `out.size()` bytes. Returns false when the data is not
/// LZO1X data of that size. Exposed for tests.
[[nodiscard]] bool DecompressLzo1x(std::span<const u8> in, std::span<u8> out);

/// What the shader search reads from the header of an Unreal Engine 3 package.
struct Summary {
    u32 version = 0;  ///< Engine file version
    u32 licensee = 0; ///< Version of the game's own changes to the format
    u32 header_size = 0;
    u32 name_count = 0;
    u32 name_offset = 0;
    u32 export_count = 0;
    u32 export_offset = 0;
    u32 import_count = 0;
    u32 import_offset = 0;
    /// Where the list of compressed chunks starts in the header, 0 when there is none.
    u32 chunk_list_offset = 0;
    /// The compressed chunks, in file order. Empty when the package is stored unpacked.
    struct Chunk {
        u32 unpacked_offset;
        u32 unpacked_size;
        u32 packed_offset;
        u32 packed_size;
    };
    std::vector<Chunk> chunks;
};

/// An object another package defines, which this one uses.
struct ObjectImport {
    s32 class_name = 0;  ///< Name index
    s32 outer = 0;       ///< Object index: negative for imports, positive for exports
    s32 object_name = 0; ///< Name index
};

/// An object this package defines, with where its data is.
struct ObjectExport {
    s32 class_index = 0; ///< Object index of the class; 0 for classes themselves
    s32 outer = 0;
    s32 object_name = 0;
    s32 object_number = 0; ///< Added to the name as "_N-1" when not 0
    u32 serial_size = 0;
    u32 serial_offset = 0;
};

/// The name, import and export tables of an Unreal Engine 3 package.
struct Tables {
    std::vector<std::string> names;
    std::vector<ObjectImport> imports;
    std::vector<ObjectExport> exports;
    /// Why reading stopped early, with the bytes it stopped at. Empty when all was read.
    std::string error;

    [[nodiscard]] std::string Name(s32 index) const;
    /// The name of an object index: an import for negative ones, an export for positive ones.
    [[nodiscard]] std::string ObjectName(s32 object_index) const;
    [[nodiscard]] std::string ClassName(const ObjectExport& object) const;
};

/// Reads the header at the start of a package file as stored. Finds the list of compressed
/// chunks by checking that each one points at a compressed part, so fields a game added to the
/// header are skipped. Returns nothing for data that is not an Unreal Engine 3 package.
[[nodiscard]] std::optional<Summary> ReadSummary(std::span<const u8> file);

/// A run of the data Unpack hands out and where it lies in the uncompressed layout.
struct ImageSpan {
    u64 unpacked_offset; ///< Offset in what Unpack hands out
    u64 image_offset;    ///< Offset in the uncompressed layout
    u64 size;
};

/// Unpacks a whole package into its uncompressed layout, where the offsets of the header and
/// tables apply: the header as stored, then each chunk at its unpacked offset. Blocks that do
/// not unpack stay zero and are counted in `failed_blocks`. Returns false when nothing unpacked.
/// `spans` gets where each block Unpack hands out lies in the layout. Unpack leaves out the
/// blocks that do not unpack, so its offsets move back by their size after each one.
/// `failures`, when given, gets the blocks that did not unpack (up to MAX_FAILED_BLOCKS_KEPT).
bool UnpackImage(std::span<const u8> file, const Summary& summary, std::vector<u8>& image,
                 size_t& failed_blocks, std::vector<ImageSpan>& spans,
                 std::vector<FailedBlock>* failures = nullptr);

/// Maps an offset in the data Unpack hands out to the offset in the uncompressed layout, with
/// the spans UnpackImage gave. Returns nothing for offsets past the last span.
[[nodiscard]] std::optional<u64> ImageOffset(std::span<const ImageSpan> spans,
                                             u64 unpacked_offset);

/// Reads the name, import and export tables from `data`, which holds the uncompressed layout
/// starting at `data_offset`.
[[nodiscard]] Tables ReadTables(std::span<const u8> data, u64 data_offset, const Summary& summary);

} // namespace Preforged::Unreal
