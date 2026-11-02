// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/common_types.h"

// Reads the pak files of Unreal Engine 4 games, so the shader file scan can search what they
// hold. A pak file ends with a footer that points at its index; the index lists each file with
// where its data is and how it is packed (stored, or in blocks of zlib, Oodle, LZ4, ...). The
// index, and sometimes the files, can be encrypted with AES-256 under a key compiled into the
// game's executable. The shaders are in ShaderArchive-*.ushaderbytecode files, where each shader
// is packed once more on its own.
namespace Preforged::UnrealPak {

constexpr u32 PAK_MAGIC = 0x5A6F12E1;
/// Bytes read from the end of a file to find the footer. The largest footer is 222 bytes.
constexpr size_t FOOTER_SEARCH = 0x200;

using Key = std::array<u8, 32>;

/// AES-256 decryption of 16-byte blocks, as Unreal Engine encrypts pak indexes and files (ECB).
class Aes256 {
public:
    explicit Aes256(const Key& key);

    void DecryptBlock(const u8* in, u8* out) const;

    /// Decrypts whole 16-byte blocks in place. A tail shorter than a block stays as it is.
    void Decrypt(std::span<u8> data) const;

private:
    std::array<u32, 60> round_keys{};
};

/// Decrypts whole 16-byte blocks in place.
using DecryptFn = std::function<void(std::span<u8>)>;
/// Reads `size` bytes at `offset` of the pak file. Returns fewer at the end of the file.
using ReadFn = std::function<std::vector<u8>(u64 offset, u64 size)>;

struct Footer {
    u32 version = 0;
    bool encrypted_index = false;
    /// Version 9 can store the index as a memory image, which isn't read.
    bool frozen_index = false;
    u64 index_offset = 0;
    u64 index_size = 0;
    /// Bytes from the footer's start to the end of the file.
    u32 size = 0;
    /// The compression methods entries name by number: method n is methods[n - 1]. Before
    /// version 8: Zlib, Gzip and the game's own (most often Oodle).
    std::vector<std::string> methods;
};

/// Finds the footer in `tail`, the last bytes of a file of `file_size` bytes.
[[nodiscard]] std::optional<Footer> ReadFooter(std::span<const u8> tail, u64 file_size);

/// The packed bytes of one block, as offsets in the pak file.
struct Block {
    u64 start;
    u64 end;
};

struct Entry {
    /// Mount point and file name. Empty when the index has no file names.
    std::string path;
    /// Where the entry's own copy of its record starts in the file, before its data.
    u64 offset = 0;
    u64 size = 0; ///< Packed bytes
    u64 uncompressed_size = 0;
    u32 method = 0; ///< 0 when stored, else method n of Footer::methods
    u32 block_size = 0;
    bool encrypted = false;
    std::vector<Block> blocks;

    /// Bytes of the record in front of the data.
    [[nodiscard]] u64 RecordSize(u32 version) const;
};

struct Index {
    std::string mount_point;
    /// Entries the index says it has.
    u64 declared = 0;
    std::vector<Entry> entries;
    /// False when the index has no directory of names, so the entries were read in order
    /// without them.
    bool named = true;
    /// Why reading stopped early. Empty when the whole index was read.
    std::string error;
};

/// Reads (and decrypts) part of the index the primary index points at, for version 10 on.
using IndexPartFn = std::function<std::vector<u8>(u64 offset, u64 size)>;

/// Reads the index from `primary`, its decrypted bytes.
[[nodiscard]] Index ReadIndex(const Footer& footer, std::span<const u8> primary,
                              const IndexPartFn& read_part);

/// Unpacks one entry into `out`. `decrypt` may be empty when the entry isn't encrypted.
/// `methods` counts the blocks by how they unpacked (Preforged::Unreal::Method).
[[nodiscard]] bool ReadEntry(const ReadFn& read, const Footer& footer, const Entry& entry,
                             const DecryptFn& decrypt, bool unpack_oodle, std::vector<u8>& out,
                             std::span<size_t> methods, std::string& error);

/// One shader in a shader archive.
struct ArchiveShader {
    u64 offset; ///< From the start of the code
    u32 size;   ///< Packed bytes
    u32 uncompressed_size;
    u8 frequency; ///< Stage: 0 vertex, 1 hull, 2 domain, 3 pixel, 4 geometry, 5 compute
};

/// The table of a ShaderArchive-*.ushaderbytecode file.
struct ShaderArchive {
    u32 version = 0;
    /// Where the code starts in the file; shader offsets count from here.
    u64 code_start = 0;
    std::vector<ArchiveShader> shaders;
    /// Shader maps (one per material and vertex factory) and the shaders each one uses, as
    /// first index into `shader_indices` and count.
    std::vector<std::pair<u32, u32>> maps;
    std::vector<u32> shader_indices;
};

/// Reads the table of a shader archive. Returns nothing, with `error` set, when `data` isn't
/// one or has a layout that isn't known.
[[nodiscard]] std::optional<ShaderArchive> ReadShaderArchive(std::span<const u8> data,
                                                             std::string& error);

/// Whether a file name is one of the shader archives (or global shader caches) that hold the
/// shaders of a game.
[[nodiscard]] bool IsShaderArchiveName(std::string_view lower_path);

/// Whether the first 16 bytes of an index look decrypted: the length of the mount point
/// string, then its first characters.
[[nodiscard]] bool LooksLikeIndexStart(std::span<const u8> block);

/// Looks through `image` (the game executable's read-only data and data) for the AES key that
/// decrypts `first_block`, the first 16 bytes of an encrypted index, to the start of an index,
/// and that `verify` accepts. Tries 32 bytes at each 4-byte step that look random, and each
/// string of 32 printable characters (older engines used one as the key). `tried` gets the keys
/// tried, `where` the offset in `image` of the one found.
[[nodiscard]] std::optional<Key> FindKey(std::span<const u8> image,
                                         std::span<const u8> first_block,
                                         const std::function<bool(const Key&)>& verify,
                                         size_t& tried, size_t& where);

} // namespace Preforged::UnrealPak
