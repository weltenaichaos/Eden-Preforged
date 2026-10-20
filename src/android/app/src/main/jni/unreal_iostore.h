// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "common/common_types.h"
#include "jni/unreal_package.h"

// Reads the IoStore containers of Unreal Engine 4.25 and later and Unreal Engine 5 games, a pair
// of files per container: a .utoc table of contents and a .ucas (one file, or several
// "<name>.ucas", "<name>_s1.ucas", ... when the container is split into partitions) holding the
// data in compression blocks. The .utoc lists each chunk (a cooked file) by its offset and
// length in the uncompressed data, and lists the compression blocks that tile the .ucas. A
// container can be AES-encrypted; this reads only the ones that are not.
namespace Preforged::IoStore {

/// Bytes of the .utoc magic.
constexpr std::array<u8, 16> MAGIC{'-', '=', '=', '-', '-', '=', '=', '-',
                                   '-', '=', '=', '-', '-', '=', '=', '-'};

/// EIoStoreTocVersion values this reader knows.
enum class Version : u8 {
    Invalid = 0,
    Initial = 1,
    DirectoryIndex = 2,
    PartitionSize = 3,
    PerfectHash = 4,
    PerfectHashWithOverflow = 5,
};

/// EIoContainerFlags bits.
enum ContainerFlags : u8 {
    FlagCompressed = 1 << 0,
    FlagEncrypted = 1 << 1,
    FlagSigned = 1 << 2,
    FlagIndexed = 1 << 3,
    FlagOnDemand = 1 << 4,
};

struct Header {
    u8 version = 0;
    u32 header_size = 0;
    u32 entry_count = 0;             ///< Chunks
    u32 block_count = 0;             ///< Compression blocks
    u32 block_entry_size = 0;        ///< 12
    u32 method_name_count = 0;       ///< Named compression methods (method 0 is none)
    u32 method_name_length = 0;      ///< 32
    u32 compression_block_size = 0;  ///< Uncompressed bytes a block covers, usually 0x10000
    u32 directory_index_size = 0;
    u32 partition_count = 0;
    u64 container_id = 0;
    std::array<u8, 16> key_guid{};
    u8 flags = 0;
    u32 perfect_hash_seeds_count = 0;
    u64 partition_size = 0;
    u32 chunks_without_perfect_hash_count = 0;
};

/// A chunk's place in the uncompressed data, from FIoOffsetAndLength (5 big-endian bytes each).
struct Chunk {
    u64 offset = 0;
    u64 length = 0;
};

/// One compression block in the .ucas, from FIoStoreTocCompressedBlockEntry (little-endian).
struct Block {
    u64 offset = 0;         ///< In the .ucas, across partitions; 16-aligned
    u32 compressed_size = 0;
    u32 uncompressed_size = 0;
    u8 method = 0;          ///< 0 none, else the (method - 1)th name
};

struct Toc {
    Header header;
    std::vector<Chunk> chunks;
    std::vector<Block> blocks;
    /// The named methods, in order; method n of a block is methods[n - 1].
    std::vector<std::string> methods;
    /// True when the container is AES-encrypted, so its data can't be read without the key.
    bool encrypted = false;
    /// Why reading stopped early; empty when the whole table was read.
    std::string error;
};

/// Reads a .utoc into a table of chunks and compression blocks. Returns nothing when `data`
/// does not start with the magic or is too short for its header.
[[nodiscard]] std::optional<Toc> ReadToc(std::span<const u8> data);

/// Reads `size` bytes at `offset` in the .ucas, where `offset` runs across the container's
/// partitions (partition = offset / partition_size when there is more than one). Returns fewer
/// bytes at the end.
using ReadFn = std::function<std::vector<u8>(u64 offset, u64 size)>;

/// Called with each chunk's uncompressed bytes.
using ChunkFn = std::function<void(size_t chunk_index, std::span<const u8> bytes)>;

/// Returns true while the search should keep going.
using KeepGoing = std::function<bool()>;

struct ReadStats {
    size_t chunks_read = 0;
    size_t chunks_skipped = 0;   ///< Over the size limit
    size_t chunks_failed = 0;    ///< A block did not unpack
    u64 packed_bytes = 0;
    u64 unpacked_bytes = 0;
    std::array<size_t, Preforged::Unreal::METHOD_COUNT> methods{};
    std::string first_failure;
};

/// Decompresses each chunk's blocks from the .ucas and hands the chunk's bytes to `chunk`.
/// Chunks whose uncompressed length is over `max_chunk_bytes` are skipped. Stops when
/// `keep_going` returns false.
void ReadChunks(const Toc& toc, const ReadFn& read, u64 max_chunk_bytes, bool unpack_oodle,
                const ChunkFn& chunk, const KeepGoing& keep_going, ReadStats& stats);

} // namespace Preforged::IoStore
