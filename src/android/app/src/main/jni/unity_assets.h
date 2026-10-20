// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

#include "common/common_types.h"

// Reads the shaders of Unity games, so the shader file scan can search them. Unity keeps each
// Shader object's compiled programs per graphics platform in one blob, each platform's part
// packed with LZ4; the Switch's platform is 19. Shader objects sit in serialized files
// (sharedassets*.assets, resources.assets, level*, globalgamemanagers), which can themselves be
// inside UnityFS bundles (data.unity3d, Addressables bundles) packed in LZ4 blocks.
namespace Preforged::Unity {

/// ShaderCompilerPlatform of Nintendo Switch (NVN).
constexpr u32 SWITCH_PLATFORM = 19;

/// Reads `size` bytes at `offset` of a file. Returns fewer at the end of the file.
using ReadFn = std::function<std::vector<u8>(u64 offset, u64 size)>;

/// Whether data starts like a UnityFS bundle.
[[nodiscard]] bool IsBundle(std::span<const u8> start);

/// Whether data starts like a serialized file (the asset files Shader objects are kept in) of
/// `size` bytes: a known format version, the size it gives and a Unity version string.
[[nodiscard]] bool IsSerializedFile(std::span<const u8> start, u64 size);

/// Unpacks LZMA data as Unity packs bundle blocks: 5 bytes of properties, then the stream,
/// whose unpacked size is `out.size()`.
[[nodiscard]] bool UnpackLzma(std::span<const u8> in, std::span<u8> out);

struct BundleBlock {
    u64 file_offset;   ///< Where its packed bytes start in the file
    u64 stream_offset; ///< Where its bytes start in the unpacked data
    u32 packed_size;
    u32 unpacked_size;
    u16 flags; ///< Low 6 bits: 0 stored, 1 LZMA, 2 LZ4, 3 LZ4HC
};

/// A file inside a bundle, as a range of the unpacked data.
struct BundleNode {
    u64 offset;
    u64 size;
    u32 flags;
    std::string path;
};

struct Bundle {
    u32 version = 0;
    std::string unity_version;
    std::string revision;
    std::vector<BundleBlock> blocks;
    std::vector<BundleNode> nodes;
};

/// Reads a bundle's header and its tables of blocks and files. Returns nothing, with `error`
/// set, when the file isn't a bundle or its tables can't be read.
[[nodiscard]] std::optional<Bundle> ReadBundle(const ReadFn& read, u64 file_size,
                                               std::string& error);

/// Keeps the last block unpacked, as a bundle packed with LZMA is often one large block that
/// every read of a file in it needs.
struct BlockCache {
    size_t index = static_cast<size_t>(-1);
    std::vector<u8> data;
    /// Blocks unpacked, by method: stored, LZMA, LZ4.
    std::array<size_t, 3> unpacked{};
    u64 lzma_bytes = 0;
};

/// Largest block unpacked, for memory.
constexpr u32 MAX_BLOCK = 512u << 20;

/// Unpacks `size` bytes at `offset` of a bundle's unpacked data (a file in it) into `out`.
[[nodiscard]] bool ReadBundleData(const ReadFn& read, const Bundle& bundle, u64 offset, u64 size,
                                  std::vector<u8>& out, BlockCache& cache, std::string& error);

struct BlobStats {
    size_t candidates = 0; ///< Places that started like a Shader object's platform list
    size_t blobs = 0;      ///< Blobs whose Switch parts unpacked
    size_t parts = 0;      ///< Switch parts unpacked
    size_t failed = 0;     ///< Switch parts that did not unpack
    u64 unpacked = 0;
    std::string first_part; ///< The first bytes of the first part, as hex
    std::string first_failure;
};

/// Finds the program blobs of Shader objects in serialized data and unpacks their Switch
/// parts, handing each to `found` with the offset of its blob. `base_offset` is the offset of
/// `data` in its file. A blob that runs past the end of `data` is read with `fetch` when given.
/// `seen` skips blobs found before, for data searched in overlapping pieces.
void FindShaderBlobs(std::span<const u8> data, u64 base_offset,
                     const std::function<void(std::span<const u8> part, u64 offset)>& found,
                     BlobStats& stats, const ReadFn* fetch, std::unordered_set<u64>& seen);

} // namespace Preforged::Unity
