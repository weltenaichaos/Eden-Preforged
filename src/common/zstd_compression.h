// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <map>
#include <memory>
#include <span>
#include <vector>

#include "common/common_types.h"

namespace Common::Compression {

/**
 * Compresses a source memory region with Zstandard and returns the compressed data in a vector.
 *
 * @param source            The uncompressed source memory region.
 * @param source_size       The size of the uncompressed source memory region.
 * @param compression_level The used compression level. Should be between 1 and 22.
 *
 * @return the compressed data.
 */
[[nodiscard]] std::vector<u8> CompressDataZSTD(const u8* source, std::size_t source_size,
                                               s32 compression_level);

/**
 * Compresses a source memory region with Zstandard with the default compression level and returns
 * the compressed data in a vector.
 *
 * @param source      The uncompressed source memory region.
 * @param source_size The size of the uncompressed source memory region.
 *
 * @return the compressed data.
 */
[[nodiscard]] std::vector<u8> CompressDataZSTDDefault(const u8* source, std::size_t source_size);

/**
 * Decompresses a source memory region with Zstandard and returns the uncompressed data in a vector.
 *
 * @param compressed the compressed source memory region.
 *
 * @return the decompressed data.
 */
[[nodiscard]] std::vector<u8> DecompressDataZSTD(std::span<const u8> compressed);

/// True when the data starts with the tag of a zstd dictionary.
[[nodiscard]] bool IsZstdDictionary(std::span<const u8> data);

/// The id of the dictionary a zstd frame was packed with, 0 when the frame names none.
[[nodiscard]] u32 ZstdFrameDictionaryId(std::span<const u8> frame);

/// The zstd dictionaries a game ships, to unpack the files it packed with them. Use it from one
/// thread at a time.
class ZstdDictionaries {
public:
    ZstdDictionaries();
    ~ZstdDictionaries();
    ZstdDictionaries(const ZstdDictionaries&) = delete;
    ZstdDictionaries& operator=(const ZstdDictionaries&) = delete;

    /// Adds a dictionary in zstd's format. Returns its id, or 0 when the data is not one or a
    /// dictionary with that id is there already.
    u32 Add(std::span<const u8> dictionary);

    [[nodiscard]] bool Has(u32 id) const;

    /// The ids of the dictionaries, with their sizes in bytes.
    [[nodiscard]] const std::map<u32, size_t>& Sizes() const;

    /// Unpacks zstd frames into at most `max_size` bytes: with the dictionary the first frame
    /// names, or when it names none, without a dictionary, and when that fails and the frame has
    /// a checksum, with each dictionary in turn. `used` is the id of the dictionary that unpacked
    /// it, 0 for none. Returns nothing when no way worked or the data unpacks to more than
    /// `max_size` bytes.
    [[nodiscard]] std::vector<u8> Decompress(std::span<const u8> compressed, size_t max_size,
                                             u32& used);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Common::Compression
