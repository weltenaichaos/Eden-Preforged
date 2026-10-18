// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>
#include <optional>

#include <fmt/format.h>
#include <zlib.h>

#include "common/lz4_compression.h"
#include "common/zstd_compression.h"
#include "jni/unreal_package.h"

namespace Preforged::Unreal {
namespace {

constexpr u32 PACKAGE_TAG = 0x9E2A83C1;
constexpr u32 PACKAGE_TAG_SWAPPED = 0xC1832A9E;
constexpr u64 MIN_BLOCK_SIZE = 0x400;
constexpr u64 MAX_BLOCK_SIZE = 0x2000000;
constexpr u64 MAX_PART_SIZE = u64{2} << 30;
constexpr u64 MAX_BLOCKS = u64{1} << 20;
constexpr size_t EXAMPLES = 4;
/// A package whose first blocks all fail is most likely packed with something unsupported.
constexpr size_t GIVE_UP_AFTER_FAILURES = 4;

struct Block {
    u64 packed;
    u64 unpacked;
};

struct Part {
    u64 block_size = 0;
    u64 packed = 0;
    u64 unpacked = 0;
    std::vector<Block> blocks;
    size_t data_begin = 0;
    size_t end = 0;
};

u32 Load32(const u8* data, bool swap) {
    u32 value;
    std::memcpy(&value, data, sizeof(value));
    return swap ? __builtin_bswap32(value) : value;
}

u64 Load64(const u8* data, bool swap) {
    u64 value;
    std::memcpy(&value, data, sizeof(value));
    return swap ? __builtin_bswap64(value) : value;
}

std::string Hex(std::span<const u8> data, size_t length) {
    std::string out;
    for (size_t i = 0; i < std::min(data.size(), length); ++i) {
        out += fmt::format("{:02x}", data[i]);
    }
    return out;
}

/// Reads the header of a compressed part at `pos`, with 32-bit fields (Unreal Engine 3) or
/// 64-bit fields (Unreal Engine 4), and checks that its sizes add up.
std::optional<Part> ReadPart(std::span<const u8> data, size_t pos) {
    if (pos + 16 > data.size()) {
        return std::nullopt;
    }
    const u8* const base = data.data() + pos;
    const u32 tag = Load32(base, false);
    if (tag != PACKAGE_TAG && tag != PACKAGE_TAG_SWAPPED) {
        return std::nullopt;
    }
    const bool swap = tag == PACKAGE_TAG_SWAPPED;
    const bool wide = Load32(base + 4, swap) == 0;
    const size_t field = wide ? 8 : 4;
    const auto load = [&](size_t offset) {
        return wide ? Load64(base + offset, swap) : u64{Load32(base + offset, swap)};
    };
    if (pos + field * 4 > data.size()) {
        return std::nullopt;
    }
    Part part;
    part.block_size = load(field);
    part.packed = load(field * 2);
    part.unpacked = load(field * 3);
    if (part.block_size < MIN_BLOCK_SIZE || part.block_size > MAX_BLOCK_SIZE ||
        part.packed == 0 || part.unpacked == 0 || part.unpacked > MAX_PART_SIZE) {
        return std::nullopt;
    }
    const u64 count = (part.unpacked + part.block_size - 1) / part.block_size;
    const u64 table_begin = pos + field * 4;
    const u64 table_end = table_begin + count * field * 2;
    if (count > MAX_BLOCKS || table_end > data.size() || part.packed > data.size() - table_end) {
        return std::nullopt;
    }
    part.blocks.reserve(count);
    u64 packed_sum = 0;
    u64 unpacked_sum = 0;
    for (u64 i = 0; i < count; ++i) {
        const size_t offset = field * 4 + i * field * 2;
        const Block block{load(offset), load(offset + field)};
        if (block.packed == 0 || block.unpacked == 0 || block.unpacked > part.block_size) {
            return std::nullopt;
        }
        packed_sum += block.packed;
        unpacked_sum += block.unpacked;
        part.blocks.push_back(block);
    }
    if (packed_sum != part.packed || unpacked_sum != part.unpacked) {
        return std::nullopt;
    }
    part.data_begin = static_cast<size_t>(table_end);
    part.end = static_cast<size_t>(table_end + part.packed);
    return part;
}

bool IsZlibHeader(std::span<const u8> in) {
    return in.size() >= 2 && (in[0] & 0x0F) == 8 && (in[0] >> 4) <= 7 &&
           ((u32{in[0]} << 8) | in[1]) % 31 == 0;
}

bool IsZstdFrame(std::span<const u8> in) {
    return in.size() >= 4 && in[0] == 0x28 && in[1] == 0xB5 && in[2] == 0x2F && in[3] == 0xFD;
}

/// Oodle streams start with a byte whose low six bits are 0x0C, followed by the compressor
/// (Kraken 6, Mermaid 10, Selkie 11, Leviathan 12, Hydra 13, older ones below 6).
bool IsOodleHeader(std::span<const u8> in) {
    return in.size() >= 2 && (in[0] & 0x3F) == 0x0C && in[1] <= 0x0D;
}

bool Inflate(std::span<const u8> in, std::span<u8> out, bool raw) {
    z_stream stream{};
    if (inflateInit2(&stream, raw ? -MAX_WBITS : MAX_WBITS) != Z_OK) {
        return false;
    }
    stream.next_in = const_cast<Bytef*>(in.data());
    stream.avail_in = static_cast<uInt>(in.size());
    stream.next_out = out.data();
    stream.avail_out = static_cast<uInt>(out.size());
    const int result = inflate(&stream, Z_FINISH);
    const bool ok = result == Z_STREAM_END && stream.total_out == out.size();
    inflateEnd(&stream);
    return ok;
}

bool DecompressLz4(std::span<const u8> in, std::span<u8> out) {
    const int size =
        Common::Compression::DecompressDataLZ4(out.data(), out.size(), in.data(), in.size());
    return size >= 0 && static_cast<size_t>(size) == out.size();
}

bool DecompressZstd(std::span<const u8> in, std::span<u8> out) {
    const std::vector<u8> result = Common::Compression::DecompressDataZSTD(in);
    if (result.size() != out.size()) {
        return false;
    }
    std::memcpy(out.data(), result.data(), result.size());
    return true;
}

bool DecompressWith(Method method, std::span<const u8> in, std::span<u8> out) {
    switch (method) {
    case Method::Zlib:
        return Inflate(in, out, !IsZlibHeader(in));
    case Method::Lzo:
        return DecompressLzo1x(in, out);
    case Method::Lz4:
        return DecompressLz4(in, out);
    case Method::Zstd:
        return DecompressZstd(in, out);
    case Method::Stored:
        if (in.size() != out.size()) {
            return false;
        }
        std::memcpy(out.data(), in.data(), in.size());
        return true;
    default:
        return false;
    }
}

/// Unpacks one block, trying the method that worked for the package's earlier blocks first.
/// Returns the method that fit, or Oodle or Unknown when none did.
Method DecompressBlock(std::span<const u8> in, std::span<u8> out, Method& last_method) {
    if (in.size() == out.size()) {
        std::memcpy(out.data(), in.data(), in.size());
        return Method::Stored;
    }
    if (IsZlibHeader(in) && Inflate(in, out, false)) {
        return Method::Zlib;
    }
    if (IsZstdFrame(in) && DecompressZstd(in, out)) {
        return Method::Zstd;
    }
    if (last_method != Method::Unknown && DecompressWith(last_method, in, out)) {
        return last_method;
    }
    if (IsOodleHeader(in)) {
        return Method::Oodle;
    }
    for (const Method method : {Method::Lzo, Method::Lz4, Method::Zlib}) {
        if (method != last_method && DecompressWith(method, in, out)) {
            last_method = method;
            return method;
        }
    }
    return Method::Unknown;
}

/// Reads the length that follows a zero length code: each zero byte adds 255, and the first
/// non-zero byte ends it.
bool ReadExtendedLength(const u8*& ip, const u8* ip_end, size_t& length, size_t base) {
    size_t zeros = 0;
    while (true) {
        if (ip >= ip_end) {
            return false;
        }
        if (*ip != 0) {
            break;
        }
        ++ip;
        ++zeros;
    }
    length = zeros * 255 + base + *ip++;
    return true;
}

} // Anonymous namespace

const char* MethodName(Method method) {
    switch (method) {
    case Method::Zlib:
        return "zlib";
    case Method::Lzo:
        return "LZO";
    case Method::Lz4:
        return "LZ4";
    case Method::Zstd:
        return "zstd";
    case Method::Stored:
        return "stored";
    case Method::Oodle:
        return "Oodle (not supported)";
    default:
        return "unknown";
    }
}

bool IsPackage(std::span<const u8> data) {
    return data.size() >= 4 && (Load32(data.data(), false) == PACKAGE_TAG ||
                                Load32(data.data(), false) == PACKAGE_TAG_SWAPPED);
}

// LZO1X, as written by lzo1x_1_compress and lzo1x_999_compress. Instructions:
//   0-15 after a match or at the start: a run of literals (0: long run);
//   0-15 after 1-3 literals: a 2-byte match up to 1 KiB back;
//   0-15 after a run of 4+ literals: a 3-byte match 2-3 KiB back;
//   16-31: a match 16-48 KiB back (distance 0 ends the stream);
//   32-63: a match up to 16 KiB back;
//   64-255: a 3-8 byte match up to 2 KiB back.
// The low two bits of each match say how many literals (0-3) follow it.
bool DecompressLzo1x(std::span<const u8> in, std::span<u8> out) {
    constexpr size_t M2_MAX_OFFSET = 0x800;
    const u8* ip = in.data();
    const u8* const ip_end = in.data() + in.size();
    size_t op = 0;
    const size_t op_end = out.size();
    const auto ip_left = [&] { return static_cast<size_t>(ip_end - ip); };
    if (in.size() < 3) {
        return false;
    }

    size_t state = 0;
    size_t length = 0;
    size_t distance = 0;
    size_t trailing = 0;
    bool literal_run = false;
    if (*ip > 17) {
        length = *ip++ - 17;
        if (length < 4) {
            trailing = length;
        } else {
            literal_run = true;
        }
        if (!literal_run) {
            // Copy the literals below, as after a match.
            if (ip_left() < trailing + 3 || op_end - op < trailing) {
                return false;
            }
            std::memcpy(out.data() + op, ip, trailing);
            op += trailing;
            ip += trailing;
            state = trailing;
        }
    }
    while (true) {
        if (literal_run) {
            literal_run = false;
            if (ip_left() < length + 3 || op_end - op < length) {
                return false;
            }
            std::memcpy(out.data() + op, ip, length);
            op += length;
            ip += length;
            state = 4;
        }
        if (ip_left() < 1) {
            return false;
        }
        const size_t code = *ip++;
        if (code < 16) {
            if (state == 0) {
                length = code;
                if (length == 0 && !ReadExtendedLength(ip, ip_end, length, 15)) {
                    return false;
                }
                length += 3;
                literal_run = true;
                continue;
            }
            if (ip_left() < 1) {
                return false;
            }
            trailing = code & 3;
            if (state != 4) {
                distance = 1 + (code >> 2) + (size_t{*ip++} << 2);
                length = 2;
            } else {
                distance = 1 + M2_MAX_OFFSET + (code >> 2) + (size_t{*ip++} << 2);
                length = 3;
            }
        } else if (code >= 64) {
            if (ip_left() < 1) {
                return false;
            }
            trailing = code & 3;
            distance = 1 + ((code >> 2) & 7) + (size_t{*ip++} << 3);
            length = (code >> 5) + 1;
        } else if (code >= 32) {
            length = (code & 31) + 2;
            if (length == 2 && !ReadExtendedLength(ip, ip_end, length, 33)) {
                return false;
            }
            if (ip_left() < 2) {
                return false;
            }
            const size_t word = size_t{ip[0]} | (size_t{ip[1]} << 8);
            ip += 2;
            distance = 1 + (word >> 2);
            trailing = word & 3;
        } else {
            length = (code & 7) + 2;
            if (length == 2 && !ReadExtendedLength(ip, ip_end, length, 9)) {
                return false;
            }
            if (ip_left() < 2) {
                return false;
            }
            const size_t word = size_t{ip[0]} | (size_t{ip[1]} << 8);
            ip += 2;
            distance = ((code & 8) << 11) + (word >> 2);
            trailing = word & 3;
            if (distance == 0) {
                // End of the stream.
                return length == 3 && ip == ip_end && op == op_end;
            }
            distance += 0x4000;
        }
        if (distance > op || op_end - op < length) {
            return false;
        }
        // Matches can overlap what they write, so copy byte by byte.
        u8* const dst = out.data() + op;
        const u8* const src = dst - distance;
        for (size_t i = 0; i < length; ++i) {
            dst[i] = src[i];
        }
        op += length;
        if (ip_left() < trailing + 3 || op_end - op < trailing) {
            return false;
        }
        std::memcpy(out.data() + op, ip, trailing);
        op += trailing;
        ip += trailing;
        state = trailing;
    }
}

bool Unpack(std::span<const u8> data, size_t piece_size, size_t overlap, Stats& stats,
            const std::function<bool(std::span<const u8> piece, u64 offset)>& consume) {
    if (!IsPackage(data)) {
        return false;
    }
    ++stats.packages;
    if (stats.header_examples.size() < EXAMPLES) {
        stats.header_examples.push_back(Hex(data, 64));
    }
    std::vector<u8> buffer;
    u64 buffer_offset = 0;
    size_t fresh = 0;
    Method last_method = Method::Unknown;
    size_t failures_in_a_row = 0;
    u64 unpacked_here = 0;
    bool found = false;
    bool keep_going = true;

    const auto flush = [&] {
        if (fresh == 0) {
            return;
        }
        keep_going = consume(buffer, buffer_offset);
        const size_t keep = std::min(overlap, buffer.size());
        buffer_offset += buffer.size() - keep;
        buffer.erase(buffer.begin(), buffer.end() - static_cast<std::ptrdiff_t>(keep));
        fresh = 0;
    };

    size_t pos = 0;
    while (keep_going && pos + 16 <= data.size()) {
        // Find the next tag, in either byte order.
        const u8* const begin = data.data() + pos;
        const u8* const end = data.data() + data.size() - 3;
        const u8* hit = begin;
        for (; hit < end; ++hit) {
            if ((hit[0] == 0xC1 && hit[1] == 0x83 && hit[2] == 0x2A && hit[3] == 0x9E) ||
                (hit[0] == 0x9E && hit[1] == 0x2A && hit[2] == 0x83 && hit[3] == 0xC1)) {
                break;
            }
        }
        if (hit >= end) {
            break;
        }
        pos = static_cast<size_t>(hit - data.data());
        const std::optional<Part> part = ReadPart(data, pos);
        if (!part) {
            ++pos;
            continue;
        }
        found = true;
        ++stats.parts;
        size_t block_pos = part->data_begin;
        for (const Block& block : part->blocks) {
            const std::span<const u8> in = data.subspan(block_pos, block.packed);
            block_pos += block.packed;
            const size_t old_size = buffer.size();
            buffer.resize(old_size + block.unpacked);
            const std::span<u8> out{buffer.data() + old_size, block.unpacked};
            const Method method = DecompressBlock(in, out, last_method);
            ++stats.blocks[static_cast<size_t>(method)];
            stats.packed_bytes += block.packed;
            if (method == Method::Oodle || method == Method::Unknown) {
                buffer.resize(old_size);
                if (method == Method::Unknown && stats.unknown_examples.size() < EXAMPLES) {
                    stats.unknown_examples.push_back(fmt::format(
                        "{} -> {} bytes: {}", block.packed, block.unpacked, Hex(in, 16)));
                }
                if (++failures_in_a_row >= GIVE_UP_AFTER_FAILURES && unpacked_here == 0) {
                    keep_going = false;
                    break;
                }
                continue;
            }
            failures_in_a_row = 0;
            unpacked_here += block.unpacked;
            stats.unpacked_bytes += block.unpacked;
            fresh += block.unpacked;
            if (fresh >= piece_size) {
                flush();
                if (!keep_going) {
                    break;
                }
            }
        }
        pos = part->end;
    }
    if (keep_going) {
        flush();
    }
    if (found) {
        ++stats.compressed_packages;
    }
    return found;
}

} // namespace Preforged::Unreal
