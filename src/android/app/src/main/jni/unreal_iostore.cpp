// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>

#include <fmt/format.h>

#include "jni/unreal_iostore.h"

namespace Preforged::IoStore {
namespace {

u64 LoadLE(const u8* data, size_t bytes) {
    u64 value = 0;
    for (size_t i = 0; i < bytes; ++i) {
        value |= u64{data[i]} << (i * 8);
    }
    return value;
}

u32 Load32(const u8* data) {
    u32 value;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

u64 Load64(const u8* data) {
    u64 value;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

/// A 5-byte big-endian number, as FIoOffsetAndLength stores an offset or a length.
u64 LoadBE40(const u8* data) {
    return (u64{data[0]} << 32) | (u64{data[1]} << 24) | (u64{data[2]} << 16) |
           (u64{data[3]} << 8) | u64{data[4]};
}

} // Anonymous namespace

std::optional<Toc> ReadToc(std::span<const u8> data) {
    if (data.size() < 144 || !std::equal(MAGIC.begin(), MAGIC.end(), data.begin())) {
        return std::nullopt;
    }
    Toc toc;
    Header& header = toc.header;
    header.version = data[16];
    header.header_size = Load32(&data[20]);
    header.entry_count = Load32(&data[24]);
    header.block_count = Load32(&data[28]);
    header.block_entry_size = Load32(&data[32]);
    header.method_name_count = Load32(&data[36]);
    header.method_name_length = Load32(&data[40]);
    header.compression_block_size = Load32(&data[44]);
    header.directory_index_size = Load32(&data[48]);
    header.partition_count = Load32(&data[52]);
    header.container_id = Load64(&data[56]);
    std::memcpy(header.key_guid.data(), &data[64], header.key_guid.size());
    header.flags = data[80];
    header.perfect_hash_seeds_count = Load32(&data[84]);
    header.partition_size = Load64(&data[88]);
    header.chunks_without_perfect_hash_count = Load32(&data[96]);

    toc.encrypted = (header.flags & FlagEncrypted) != 0 ||
                    std::any_of(header.key_guid.begin(), header.key_guid.end(),
                                [](u8 byte) { return byte != 0; });

    if (header.header_size != 144 || header.block_entry_size != 12 ||
        header.method_name_length == 0 || header.method_name_length > 256) {
        toc.error = fmt::format("header values out of range (size {}, block entry {}, method name "
                                "length {})",
                                header.header_size, header.block_entry_size,
                                header.method_name_length);
        return toc;
    }

    // The chunk ids, the chunk offset/length pairs, the perfect-hash arrays (when present), the
    // compression blocks, then the method names follow the header in that order.
    size_t pos = header.header_size;
    const auto take{[&](size_t count, size_t size) -> const u8* {
        if (count > data.size() / size || pos > data.size() - count * size) {
            return nullptr;
        }
        const u8* const at = data.data() + pos;
        pos += count * size;
        return at;
    }};

    take(header.entry_count, 12); // Chunk ids, not needed for the search.
    const u8* const offsets = take(header.entry_count, 10);
    take(header.perfect_hash_seeds_count, 4);
    take(header.chunks_without_perfect_hash_count, 4);
    const u8* const blocks = take(header.block_count, 12);
    const u8* const methods = take(header.method_name_count, header.method_name_length);
    if (!offsets || !blocks || !methods) {
        toc.error = "the table does not fit the file";
        return toc;
    }

    toc.chunks.reserve(header.entry_count);
    for (u32 i = 0; i < header.entry_count; ++i) {
        const u8* const at = offsets + i * 10;
        toc.chunks.push_back({LoadBE40(at), LoadBE40(at + 5)});
    }
    toc.blocks.reserve(header.block_count);
    for (u32 i = 0; i < header.block_count; ++i) {
        const u8* const at = blocks + i * 12;
        toc.blocks.push_back({LoadLE(at, 5), static_cast<u32>(LoadLE(at + 5, 3)),
                              static_cast<u32>(LoadLE(at + 8, 3)), at[11]});
    }
    toc.methods.reserve(header.method_name_count);
    for (u32 i = 0; i < header.method_name_count; ++i) {
        const u8* const at = methods + i * header.method_name_length;
        const u8* const end = std::find(at, at + header.method_name_length, u8{0});
        toc.methods.emplace_back(reinterpret_cast<const char*>(at),
                                 static_cast<size_t>(end - at));
    }
    return toc;
}

void ReadChunks(const Toc& toc, const ReadFn& read, u64 max_chunk_bytes, bool unpack_oodle,
                const ChunkFn& chunk, const KeepGoing& keep_going, ReadStats& stats) {
    const u32 block_size = toc.header.compression_block_size;
    if (block_size == 0 || toc.blocks.empty()) {
        return;
    }
    std::vector<u8> chunk_bytes;
    std::vector<u8> block_out;
    for (size_t index = 0; index < toc.chunks.size(); ++index) {
        if (!keep_going()) {
            return;
        }
        const Chunk& c = toc.chunks[index];
        if (c.length == 0) {
            continue;
        }
        if (c.length > max_chunk_bytes) {
            ++stats.chunks_skipped;
            continue;
        }
        const u64 first_block = c.offset / block_size;
        const u64 last_block = (c.offset + c.length - 1) / block_size;
        if (last_block >= toc.blocks.size()) {
            ++stats.chunks_failed;
            if (stats.first_failure.empty()) {
                stats.first_failure = fmt::format("chunk {} points past the blocks", index);
            }
            continue;
        }
        chunk_bytes.clear();
        bool ok = true;
        Preforged::Unreal::Method last_method = Preforged::Unreal::Method::Unknown;
        for (u64 b = first_block; b <= last_block; ++b) {
            const Block& block = toc.blocks[b];
            const std::vector<u8> packed = read(block.offset, block.compressed_size);
            if (packed.size() != block.compressed_size) {
                ok = false;
                if (stats.first_failure.empty()) {
                    stats.first_failure = fmt::format("block {} could not be read from the .ucas", b);
                }
                break;
            }
            stats.packed_bytes += block.compressed_size;
            block_out.assign(block.uncompressed_size + Preforged::Unreal::UNPACK_SLACK, 0);
            const std::span<u8> out{block_out.data(), block.uncompressed_size};
            std::string_view method_name;
            if (block.method != 0 && block.method <= toc.methods.size()) {
                method_name = toc.methods[block.method - 1];
            }
            Preforged::Unreal::Method used;
            if (block.compressed_size == block.uncompressed_size && block.method == 0) {
                std::memcpy(block_out.data(), packed.data(), block.uncompressed_size);
                used = Preforged::Unreal::Method::Stored;
            } else {
                used = Preforged::Unreal::UnpackBlock(packed, out, method_name, last_method,
                                                      unpack_oodle);
            }
            if (used == Preforged::Unreal::Method::Unknown ||
                used == Preforged::Unreal::Method::OodleFailed) {
                ok = false;
                if (stats.first_failure.empty()) {
                    stats.first_failure =
                        fmt::format("block {} did not unpack ({} bytes, method {})", b,
                                    block.compressed_size,
                                    method_name.empty() ? "none" : std::string{method_name});
                }
                break;
            }
            ++stats.methods[static_cast<size_t>(used)];
            chunk_bytes.insert(chunk_bytes.end(), block_out.begin(),
                               block_out.begin() + block.uncompressed_size);
        }
        if (!ok) {
            ++stats.chunks_failed;
            continue;
        }
        // The chunk starts partway into its first block.
        const u64 start = c.offset - first_block * u64{block_size};
        if (start + c.length > chunk_bytes.size()) {
            ++stats.chunks_failed;
            continue;
        }
        ++stats.chunks_read;
        stats.unpacked_bytes += c.length;
        chunk(index, std::span<const u8>{chunk_bytes.data() + start, c.length});
    }
}

} // namespace Preforged::IoStore
