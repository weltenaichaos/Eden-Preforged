// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <cstring>
#include <optional>
#include <string_view>

#include <fmt/format.h>
#include <zlib.h>

#include "common/lz4_compression.h"
#include "common/zstd_compression.h"
#include "jni/ooz/ooz.h"
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

/// Oodle streams start with a byte whose low six bits are 0x0C, followed by the decoder in the
/// low seven bits (Kraken 6, Mermaid and Selkie 10, Leviathan 12, LZNA 5, BitKnit 11). The top
/// bit marks streams with checksums, which unpack without checking them.
bool IsOodleHeader(std::span<const u8> in) {
    return in.size() >= 2 && (in[0] & 0x3F) == 0x0C && (in[1] & 0x7F) <= 0x0D;
}

/// Unpacks an Oodle block with ooz. The decoder may read a little past its input, so it gets a
/// copy with zeros after it. `out` must have Ooz::OUTPUT_SLACK bytes of room after it.
bool DecompressOodle(std::span<const u8> in, std::span<u8> out) {
    thread_local std::vector<u8> padded;
    padded.assign(in.begin(), in.end());
    padded.resize(in.size() + Ooz::OUTPUT_SLACK, 0);
    const int size = Ooz::Kraken_Decompress(padded.data(), in.size(), out.data(), out.size());
    return size >= 0 && static_cast<size_t>(size) == out.size();
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
    case Method::Oodle:
        return IsOodleHeader(in) && DecompressOodle(in, out);
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
/// Oodle comes after the methods that check their data, since ooz was written for files it
/// could trust: a block that only starts like Oodle reaches it only when nothing else unpacks
/// it. Returns the method that fit, or OodleFailed or Unknown when none did.
Method DecompressBlock(std::span<const u8> in, std::span<u8> out, Method& last_method,
                       bool unpack_oodle) {
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
    for (const Method method : {Method::Lzo, Method::Lz4, Method::Zlib}) {
        if (method != last_method && DecompressWith(method, in, out)) {
            last_method = method;
            return method;
        }
    }
    const bool oodle = IsOodleHeader(in);
    if (oodle && unpack_oodle && last_method != Method::Oodle && DecompressOodle(in, out)) {
        last_method = Method::Oodle;
        return Method::Oodle;
    }
    return oodle ? Method::OodleFailed : Method::Unknown;
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
        return "Oodle";
    case Method::OodleFailed:
        return "Oodle (did not unpack)";
    default:
        return "unknown";
    }
}

const char* OodleCompressorName(size_t compressor) {
    switch (compressor) {
    case 5:
        return "LZNA";
    case 6:
        return "Kraken";
    case 10:
        return "Mermaid or Selkie";
    case 11:
        return "BitKnit";
    case 12:
        return "Leviathan";
    default:
        return "other";
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
            const std::function<bool(std::span<const u8> piece, u64 offset)>& consume,
            bool unpack_oodle) {
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
            // Room for the Oodle decoder, which may write a little past the block.
            buffer.resize(old_size + block.unpacked + Ooz::OUTPUT_SLACK);
            const std::span<u8> out{buffer.data() + old_size, block.unpacked};
            const auto unpack_start = std::chrono::steady_clock::now();
            const Method method = DecompressBlock(in, out, last_method, unpack_oodle);
            stats.unpack_time += std::chrono::steady_clock::now() - unpack_start;
            buffer.resize(old_size + block.unpacked);
            ++stats.blocks[static_cast<size_t>(method)];
            stats.packed_bytes += block.packed;
            if (IsOodleHeader(in)) {
                ++stats.oodle_compressors[std::min<size_t>(in[1] & 0x7F, 15)];
            }
            if (method == Method::OodleFailed || method == Method::Unknown) {
                buffer.resize(old_size);
                if (method == Method::OodleFailed && unpack_oodle &&
                    stats.oodle_examples.size() < EXAMPLES) {
                    stats.oodle_examples.push_back(fmt::format(
                        "{} -> {} bytes: {}", block.packed, block.unpacked, Hex(in, 16)));
                }
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

namespace {

constexpr u32 MIN_UE3_VERSION = 491;
constexpr u32 MAX_UE3_VERSION = 900;
constexpr u32 MAX_CHUNKS = 100000;
constexpr u32 MAX_TABLE_ENTRIES = 4000000;
constexpr s32 MAX_NAME_LENGTH = 1024;

/// Reads little-endian values from a span, failing past its end.
class Reader {
public:
    Reader(std::span<const u8> data_, size_t pos_) : data{data_}, pos{pos_} {}

    bool U32(u32& value) {
        if (pos > data.size() || data.size() - pos < 4) {
            return false;
        }
        value = Load32(data.data() + pos, false);
        pos += 4;
        return true;
    }

    bool S32(s32& value) {
        u32 raw;
        if (!U32(raw)) {
            return false;
        }
        value = static_cast<s32>(raw);
        return true;
    }

    bool Skip(size_t bytes) {
        if (pos > data.size() || data.size() - pos < bytes) {
            return false;
        }
        pos += bytes;
        return true;
    }

    /// Reads a length-prefixed string: positive lengths count bytes, negative ones UTF-16
    /// units, both with the closing zero. Characters past ASCII become '?'.
    bool String(std::string& out) {
        s32 length;
        if (!S32(length) || length > MAX_NAME_LENGTH || length < -MAX_NAME_LENGTH) {
            return false;
        }
        out.clear();
        if (length >= 0) {
            if (!Fits(static_cast<size_t>(length))) {
                return false;
            }
            for (s32 i = 0; i + 1 < length; ++i) {
                out += static_cast<char>(data[pos + i]);
            }
            pos += static_cast<size_t>(length);
            return true;
        }
        const size_t units = static_cast<size_t>(-length);
        if (!Fits(units * 2)) {
            return false;
        }
        for (size_t i = 0; i + 1 < units; ++i) {
            const u16 unit = static_cast<u16>(data[pos + i * 2] | (data[pos + i * 2 + 1] << 8));
            out += unit < 0x80 ? static_cast<char>(unit) : '?';
        }
        pos += units * 2;
        return true;
    }

    size_t Pos() const {
        return pos;
    }

private:
    bool Fits(size_t bytes) const {
        return pos <= data.size() && data.size() - pos >= bytes;
    }

    std::span<const u8> data;
    size_t pos;
};

/// Checks for a list of compressed chunks at `pos`: a count, then per chunk its unpacked
/// offset and size and its packed offset and size. Each chunk must point at a compressed part
/// that unpacks to its size, and the unpacked ranges must follow one another.
std::optional<std::vector<Summary::Chunk>> ChunkListAt(std::span<const u8> file, size_t pos) {
    Reader reader{file, pos};
    u32 count;
    if (!reader.U32(count) || count == 0 || count > MAX_CHUNKS) {
        return std::nullopt;
    }
    std::vector<Summary::Chunk> chunks(count);
    for (Summary::Chunk& chunk : chunks) {
        if (!reader.U32(chunk.unpacked_offset) || !reader.U32(chunk.unpacked_size) ||
            !reader.U32(chunk.packed_offset) || !reader.U32(chunk.packed_size)) {
            return std::nullopt;
        }
    }
    for (size_t i = 0; i < chunks.size(); ++i) {
        const Summary::Chunk& chunk = chunks[i];
        if (u64{chunk.packed_offset} + chunk.packed_size > file.size() ||
            (i > 0 && chunk.unpacked_offset !=
                          chunks[i - 1].unpacked_offset + chunks[i - 1].unpacked_size)) {
            return std::nullopt;
        }
        const std::optional<Part> part = ReadPart(file, chunk.packed_offset);
        if (!part || part->unpacked != chunk.unpacked_size) {
            return std::nullopt;
        }
    }
    return chunks;
}

} // Anonymous namespace

std::string Tables::Name(s32 index) const {
    if (index >= 0 && static_cast<size_t>(index) < names.size()) {
        return names[static_cast<size_t>(index)];
    }
    return fmt::format("#{}", index);
}

std::string Tables::ObjectName(s32 object_index) const {
    if (object_index < 0 && static_cast<size_t>(-static_cast<s64>(object_index)) <= imports.size()) {
        return Name(imports[static_cast<size_t>(-static_cast<s64>(object_index) - 1)].object_name);
    }
    if (object_index > 0 && static_cast<size_t>(object_index) <= exports.size()) {
        const ObjectExport& object = exports[static_cast<size_t>(object_index - 1)];
        return object.object_number > 0
                   ? fmt::format("{}_{}", Name(object.object_name), object.object_number - 1)
                   : Name(object.object_name);
    }
    return object_index == 0 ? "None" : fmt::format("object {}", object_index);
}

std::string Tables::ClassName(const ObjectExport& object) const {
    return object.class_index == 0 ? "Class" : ObjectName(object.class_index);
}

std::optional<Summary> ReadSummary(std::span<const u8> file) {
    if (file.size() < 64 || Load32(file.data(), false) != PACKAGE_TAG) {
        return std::nullopt;
    }
    Summary summary;
    Reader reader{file, 4};
    u32 version_word;
    std::string folder;
    u32 flags;
    if (!reader.U32(version_word)) {
        return std::nullopt;
    }
    summary.version = version_word & 0xFFFF;
    summary.licensee = version_word >> 16;
    if (summary.version < MIN_UE3_VERSION || summary.version > MAX_UE3_VERSION) {
        return std::nullopt;
    }
    if (!reader.U32(summary.header_size) || !reader.String(folder) || !reader.U32(flags) ||
        !reader.U32(summary.name_count) || !reader.U32(summary.name_offset) ||
        !reader.U32(summary.export_count) || !reader.U32(summary.export_offset) ||
        !reader.U32(summary.import_count) || !reader.U32(summary.import_offset)) {
        return std::nullopt;
    }
    if (summary.name_count > MAX_TABLE_ENTRIES || summary.export_count > MAX_TABLE_ENTRIES ||
        summary.import_count > MAX_TABLE_ENTRIES) {
        return std::nullopt;
    }
    // Depends offset; import and export guids (version 623); thumbnail table (584); guid.
    u32 generations = 0;
    const bool skipped = reader.Skip(4) && (summary.version < 623 || reader.Skip(12)) &&
                         (summary.version < 584 || reader.Skip(4)) && reader.Skip(16) &&
                         reader.U32(generations) && generations <= 4096 &&
                         reader.Skip(size_t{generations} * 12) &&
                         // Engine version, cooked content version, compression flags.
                         reader.Skip(12);
    if (!skipped) {
        return std::nullopt;
    }
    const size_t expected = reader.Pos();
    // Games add fields to the header, so look a little before and after where the list
    // should be.
    for (size_t delta = 0; delta <= 256; ++delta) {
        for (const size_t pos : {expected + delta, expected - std::min(expected, delta)}) {
            if (auto chunks = ChunkListAt(file, pos)) {
                summary.chunk_list_offset = static_cast<u32>(pos);
                summary.chunks = std::move(*chunks);
                return summary;
            }
        }
    }
    u32 count = 0;
    if (Reader{file, expected}.U32(count) && count == 0) {
        return summary; // Stored unpacked
    }
    return std::nullopt;
}

bool UnpackImage(std::span<const u8> file, const Summary& summary, std::vector<u8>& image,
                 size_t& failed_blocks, std::vector<ImageSpan>& spans) {
    failed_blocks = 0;
    spans.clear();
    if (summary.chunks.empty()) {
        image.assign(file.begin(), file.end());
        spans.push_back({0, 0, file.size()});
        return true;
    }
    const Summary::Chunk& last = summary.chunks.back();
    const size_t first_offset = summary.chunks.front().unpacked_offset;
    image.assign(size_t{last.unpacked_offset} + last.unpacked_size + Ooz::OUTPUT_SLACK, 0);
    std::memcpy(image.data(), file.data(), std::min(first_offset, file.size()));
    // In file order, the order Unpack finds the parts in.
    std::vector<const Summary::Chunk*> order;
    for (const Summary::Chunk& chunk : summary.chunks) {
        order.push_back(&chunk);
    }
    std::stable_sort(order.begin(), order.end(), [](const auto* a, const auto* b) {
        return a->packed_offset < b->packed_offset;
    });
    Method last_method = Method::Unknown;
    bool any = false;
    u64 unpacked_pos = 0;
    for (const Summary::Chunk* const chunk : order) {
        const std::optional<Part> part = ReadPart(file, chunk->packed_offset);
        if (!part) {
            continue;
        }
        size_t block_pos = part->data_begin;
        size_t out_pos = chunk->unpacked_offset;
        for (const Block& block : part->blocks) {
            const std::span<const u8> in = file.subspan(block_pos, block.packed);
            // A chunk list that disagrees with its parts must not write past the image.
            if (out_pos + block.unpacked > image.size() - Ooz::OUTPUT_SLACK) {
                ++failed_blocks;
                break;
            }
            const std::span<u8> out{image.data() + out_pos, block.unpacked};
            const Method method = DecompressBlock(in, out, last_method, true);
            if (method == Method::OodleFailed || method == Method::Unknown) {
                std::memset(out.data(), 0, out.size());
                ++failed_blocks;
            } else {
                any = true;
                if (!spans.empty() &&
                    spans.back().image_offset + spans.back().size == out_pos) {
                    spans.back().size += block.unpacked;
                } else {
                    spans.push_back({unpacked_pos, out_pos, block.unpacked});
                }
                unpacked_pos += block.unpacked;
            }
            block_pos += block.packed;
            out_pos += block.unpacked;
        }
    }
    image.resize(size_t{last.unpacked_offset} + last.unpacked_size);
    return any;
}

std::optional<u64> ImageOffset(std::span<const ImageSpan> spans, u64 unpacked_offset) {
    // The last span that starts at or before the offset.
    const auto after = std::upper_bound(
        spans.begin(), spans.end(), unpacked_offset,
        [](u64 offset, const ImageSpan& span) { return offset < span.unpacked_offset; });
    if (after == spans.begin()) {
        return std::nullopt;
    }
    const ImageSpan& span = *(after - 1);
    if (unpacked_offset - span.unpacked_offset >= span.size) {
        return std::nullopt;
    }
    return span.image_offset + (unpacked_offset - span.unpacked_offset);
}

Tables ReadTables(std::span<const u8> data, u64 data_offset, const Summary& summary) {
    Tables tables;
    const auto start = [&](u32 offset) -> std::optional<size_t> {
        if (offset < data_offset || offset - data_offset > data.size()) {
            return std::nullopt;
        }
        return static_cast<size_t>(offset - data_offset);
    };
    const auto fail = [&](std::string_view what, size_t index, size_t pos) {
        tables.error = fmt::format("{} {} at 0x{:x}: {}", what, index, pos + data_offset,
                                   Hex(data.subspan(std::min(pos, data.size())), 64));
        return tables;
    };
    const auto names_pos = start(summary.name_offset);
    if (!names_pos) {
        return fail("names start outside the data, name", 0, 0);
    }
    Reader reader{data, *names_pos};
    tables.names.reserve(summary.name_count);
    for (u32 i = 0; i < summary.name_count; ++i) {
        const size_t pos = reader.Pos();
        std::string name;
        if (!reader.String(name) || !reader.Skip(8)) {
            return fail("name", i, pos);
        }
        tables.names.push_back(std::move(name));
    }
    const auto valid_name = [&](s32 index) {
        return index >= 0 && static_cast<size_t>(index) < tables.names.size();
    };
    const auto valid_object = [&](s32 index) {
        return index >= -static_cast<s64>(summary.import_count) &&
               index <= static_cast<s64>(summary.export_count);
    };
    const auto imports_pos = start(summary.import_offset);
    if (!imports_pos) {
        return fail("imports start outside the data, import", 0, 0);
    }
    reader = Reader{data, *imports_pos};
    tables.imports.reserve(summary.import_count);
    for (u32 i = 0; i < summary.import_count; ++i) {
        const size_t pos = reader.Pos();
        ObjectImport object;
        s32 class_package;
        s32 number;
        if (!reader.S32(class_package) || !reader.Skip(4) || !reader.S32(object.class_name) ||
            !reader.Skip(4) || !reader.S32(object.outer) || !reader.S32(object.object_name) ||
            !reader.S32(number) || !valid_name(class_package) || !valid_name(object.class_name) ||
            !valid_name(object.object_name) || !valid_object(object.outer)) {
            return fail("import", i, pos);
        }
        tables.imports.push_back(object);
    }
    const auto exports_pos = start(summary.export_offset);
    if (!exports_pos) {
        return fail("exports start outside the data, export", 0, 0);
    }
    reader = Reader{data, *exports_pos};
    tables.exports.reserve(summary.export_count);
    for (u32 i = 0; i < summary.export_count; ++i) {
        const size_t pos = reader.Pos();
        ObjectExport object;
        s32 super_index;
        s32 archetype;
        u32 net_objects = 0;
        // Class, super, outer, name, archetype, flags (64 bits), serial size and offset, export
        // flags, net object counts per generation, package guid, package flags.
        if (!reader.S32(object.class_index) || !reader.S32(super_index) ||
            !reader.S32(object.outer) || !reader.S32(object.object_name) ||
            !reader.S32(object.object_number) || !reader.S32(archetype) || !reader.Skip(8) ||
            !reader.U32(object.serial_size) || !reader.U32(object.serial_offset) ||
            !reader.Skip(4) || !reader.U32(net_objects) || net_objects > 1024 ||
            !reader.Skip(size_t{net_objects} * 4 + 16 + 4) || !valid_object(object.class_index) ||
            !valid_object(super_index) || !valid_object(object.outer) ||
            !valid_object(archetype) || !valid_name(object.object_name) ||
            (object.serial_size > 0 && object.serial_offset < summary.name_offset)) {
            return fail("export", i, pos);
        }
        tables.exports.push_back(object);
    }
    return tables;
}

} // namespace Preforged::Unreal
