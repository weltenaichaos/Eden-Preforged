// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>
#include <limits>

#include <zlib.h>

#include "jni/embedded_streams.h"

namespace Preforged::Embedded {
namespace {

/// Bytes read from the file at a time past the piece in memory.
constexpr u64 WINDOW = u64{4} << 20;
/// Largest block of a zstd frame, packed or not.
constexpr u32 ZSTD_MAX_BLOCK = 128u << 10;
constexpr u64 UNKNOWN = std::numeric_limits<u64>::max();

u32 Load24(const u8* data) {
    return u32{data[0]} | (u32{data[1]} << 8) | (u32{data[2]} << 16);
}

u32 Load32(const u8* data) {
    u32 value;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

u16 Load16(const u8* data) {
    u16 value;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

struct FrameInfo {
    u64 size = 0;
    u64 content = UNKNOWN; ///< The unpacked size the frame records, if it does
};

FrameInfo ReadZstdFrame(Source& source, u64 offset) {
    const auto header = source.Get(offset, 18);
    if (header.size() < 6 || Load32(header.data()) != 0xFD2FB528) {
        return {};
    }
    const u8 descriptor = header[4];
    const u32 content_flag = descriptor >> 6;
    const bool single_segment = (descriptor >> 5) & 1;
    const bool checksum = (descriptor >> 2) & 1;
    const u32 dictionary_flag = descriptor & 3;
    if ((descriptor >> 3) & 1) {
        return {};
    }
    size_t pos = 5;
    if (!single_segment) {
        // Window sizes past 2 GiB aren't valid.
        if (10 + (header[5] >> 3) > 31) {
            return {};
        }
        ++pos;
    }
    constexpr std::array<size_t, 4> DICTIONARY_BYTES{0, 1, 2, 4};
    pos += DICTIONARY_BYTES[dictionary_flag];
    constexpr std::array<size_t, 4> CONTENT_BYTES{0, 2, 4, 8};
    const size_t content_bytes =
        content_flag == 0 ? (single_segment ? 1 : 0) : CONTENT_BYTES[content_flag];
    if (header.size() < pos + content_bytes) {
        return {};
    }
    FrameInfo info;
    if (content_bytes != 0) {
        u64 content = 0;
        for (size_t i = 0; i < content_bytes; ++i) {
            content |= u64{header[pos + i]} << (8 * i);
        }
        info.content = content_bytes == 2 ? content + 256 : content;
    }
    pos += content_bytes;
    u64 at = offset + pos;
    while (true) {
        const auto block = source.Get(at, 3);
        if (block.size() < 3) {
            return {};
        }
        const u32 value = Load24(block.data());
        const bool last = value & 1;
        const u32 type = (value >> 1) & 3;
        const u32 size = value >> 3;
        if (type == 3 || size > ZSTD_MAX_BLOCK) {
            return {};
        }
        at += 3 + (type == 1 ? 1 : size);
        if (at > source.FileSize()) {
            return {};
        }
        if (last) {
            break;
        }
    }
    if (checksum) {
        at += 4;
        if (at > source.FileSize()) {
            return {};
        }
    }
    info.size = at - offset;
    return info;
}

/// Unpacks zlib streams and raw deflate streams read from a source.
class Inflater {
public:
    Inflater() {
        ok = inflateInit2(&stream, MAX_WBITS) == Z_OK;
    }

    ~Inflater() {
        if (ok) {
            inflateEnd(&stream);
        }
    }

    Inflater(const Inflater&) = delete;
    Inflater& operator=(const Inflater&) = delete;

    /// Unpacks the stream at `offset` into `out`, at most `max_size` bytes. `packed` is the
    /// bytes it took. Returns false when it isn't a complete stream there, or (with
    /// `too_large` set) when it unpacks to more than `max_size` bytes.
    bool Run(Source& source, u64 offset, bool raw, size_t max_size, std::vector<u8>& out,
             u64& packed, bool& too_large) {
        too_large = false;
        if (!ok || inflateReset2(&stream, raw ? -MAX_WBITS : MAX_WBITS) != Z_OK) {
            return false;
        }
        out.resize(std::min<size_t>(max_size, 64u << 10));
        size_t written = 0;
        u64 read_at = offset;
        stream.avail_in = 0;
        while (true) {
            if (stream.avail_in == 0) {
                const auto input = source.Get(read_at, std::min<u64>(WINDOW, u64{1} << 20));
                if (input.empty()) {
                    return false;
                }
                stream.next_in = const_cast<Bytef*>(input.data());
                stream.avail_in = static_cast<uInt>(input.size());
                read_at += input.size();
            }
            if (written == out.size()) {
                if (out.size() >= max_size) {
                    too_large = true;
                    return false;
                }
                out.resize(std::min(max_size, out.size() * 2));
            }
            stream.next_out = out.data() + written;
            stream.avail_out = static_cast<uInt>(out.size() - written);
            const int result = inflate(&stream, Z_NO_FLUSH);
            written = out.size() - stream.avail_out;
            if (result == Z_STREAM_END) {
                out.resize(written);
                packed = stream.total_in;
                return true;
            }
            if (result != Z_OK && result != Z_BUF_ERROR) {
                return false;
            }
            if (result == Z_BUF_ERROR && stream.avail_in != 0 && stream.avail_out != 0) {
                return false;
            }
        }
    }

private:
    z_stream stream{};
    bool ok = false;
};

/// The second byte of a zlib stream with a 32 KiB window and no preset dictionary.
bool IsZlibFlag(u8 flag) {
    return flag == 0x01 || flag == 0x5E || flag == 0x9C || flag == 0xDA;
}

class Finder {
public:
    Finder(Source& source_, size_t min_size_, size_t max_size_, const Handlers& handlers_,
           Stats& stats_)
        : source{source_}, min_size{min_size_}, max_size{max_size_}, handlers{handlers_},
          stats{stats_} {}

    /// The packed bytes of a stream at `offset`, 0 when there is none.
    u64 Try(u8 first, u64 offset) {
        const auto start = source.Get(offset, 4);
        if (start.size() < 4) {
            return 0;
        }
        switch (first) {
        case 0x28:
            return Load32(start.data()) == 0xFD2FB528 ? TryZstd(offset) : 0;
        case 0x78:
            return IsZlibFlag(start[1]) ? TryZlib(offset) : 0;
        case 'P':
            return start[1] == 'K' && start[2] == 3 && start[3] == 4 ? TryZip(offset) : 0;
        default:
            return std::memcmp(start.data(), "CRIL", 4) == 0 ? TryCrilayla(offset) : 0;
        }
    }

private:
    static constexpr size_t Index(Kind kind) {
        return static_cast<size_t>(kind);
    }

    u64 TryZstd(u64 offset) {
        const FrameInfo frame = ReadZstdFrame(source, offset);
        if (frame.size == 0) {
            return 0;
        }
        constexpr size_t index = Index(Kind::Zstd);
        ++stats.candidates[index];
        if (frame.size > max_size || (frame.content != UNKNOWN && frame.content > max_size)) {
            ++stats.too_large[index];
            return frame.size;
        }
        if (frame.content != UNKNOWN && frame.content < min_size) {
            return frame.size;
        }
        const auto bytes = source.Get(offset, frame.size);
        if (bytes.size() != frame.size) {
            return 0;
        }
        if (const u64 unpacked = handlers.zstd(offset, bytes); unpacked != 0) {
            ++stats.streams[index];
            stats.packed[index] += frame.size;
            stats.unpacked[index] += unpacked;
        }
        // Even a frame that did not unpack (packed with a dictionary the game files don't
        // have) holds nothing to search in its packed bytes.
        return frame.size;
    }

    u64 TryZlib(u64 offset) {
        constexpr size_t index = Index(Kind::Zlib);
        u64 packed = 0;
        bool too_large = false;
        if (!inflater.Run(source, offset, false, max_size, out, packed, too_large)) {
            if (too_large) {
                ++stats.candidates[index];
                ++stats.too_large[index];
            }
            return 0;
        }
        ++stats.candidates[index];
        if (out.size() >= min_size) {
            ++stats.streams[index];
            stats.packed[index] += packed;
            stats.unpacked[index] += out.size();
            handlers.unpacked(Kind::Zlib, offset, out);
        }
        return packed;
    }

    u64 TryZip(u64 offset) {
        const auto header = source.Get(offset, 30);
        if (header.size() < 30) {
            return 0;
        }
        const u16 version = Load16(header.data() + 4);
        const u16 flags = Load16(header.data() + 6);
        const u16 method = Load16(header.data() + 8);
        const u32 crc = Load32(header.data() + 14);
        const u32 packed_size = Load32(header.data() + 18);
        const u32 size = Load32(header.data() + 22);
        const u16 name_length = Load16(header.data() + 26);
        const u16 extra_length = Load16(header.data() + 28);
        // Deflate only: stored entries are searched as part of the file already. Encrypted
        // entries (flag 1) can't be read.
        if (version > 100 || method != 8 || (flags & 1) != 0 || name_length == 0 ||
            name_length > 1024 || extra_length > 4096) {
            return 0;
        }
        const auto name = source.Get(offset + 30, name_length);
        if (name.size() != name_length ||
            std::ranges::any_of(name, [](u8 c) { return c < 0x20 || c == 0x7F; })) {
            return 0;
        }
        constexpr size_t index = Index(Kind::Zip);
        ++stats.candidates[index];
        const u64 data_offset = 30 + u64{name_length} + extra_length;
        // Sizes and checksum follow the data instead (flag 8), or sit in the ZIP64 field.
        const bool sizes_known = (flags & 8) == 0 && size != 0xFFFFFFFF && packed_size != 0xFFFFFFFF;
        if (sizes_known && size > max_size) {
            ++stats.too_large[index];
            return data_offset + packed_size;
        }
        if (sizes_known && size < min_size) {
            return data_offset + packed_size;
        }
        u64 packed = 0;
        bool too_large = false;
        if (!inflater.Run(source, offset + data_offset, true,
                          sizes_known ? std::max<size_t>(size, 1) : max_size, out, packed,
                          too_large)) {
            if (too_large) {
                ++stats.too_large[index];
            }
            return 0;
        }
        if (sizes_known &&
            (out.size() != size ||
             crc32(crc32(0L, Z_NULL, 0), out.data(), static_cast<uInt>(out.size())) != crc)) {
            return 0;
        }
        if (out.size() >= min_size) {
            ++stats.streams[index];
            stats.packed[index] += packed;
            stats.unpacked[index] += out.size();
            handlers.unpacked(Kind::Zip, offset, out);
        }
        return data_offset + packed;
    }

    u64 TryCrilayla(u64 offset) {
        const auto header = source.Get(offset, 16);
        if (header.size() < 16 || std::memcmp(header.data(), "CRILAYLA", 8) != 0) {
            return 0;
        }
        constexpr size_t index = Index(Kind::Crilayla);
        const u32 size = Load32(header.data() + 8);
        const u32 packed_size = Load32(header.data() + 12);
        const u64 total = 16 + u64{packed_size} + 0x100;
        if (offset + total > source.FileSize() || packed_size == 0) {
            return 0;
        }
        ++stats.candidates[index];
        if (size + u64{0x100} > max_size) {
            ++stats.too_large[index];
            return total;
        }
        if (size + u64{0x100} < min_size) {
            return total;
        }
        const auto bytes = source.Get(offset, total);
        if (bytes.size() != total || !UnpackCrilayla(bytes, out)) {
            return 0;
        }
        ++stats.streams[index];
        stats.packed[index] += total;
        stats.unpacked[index] += out.size();
        handlers.unpacked(Kind::Crilayla, offset, out);
        return total;
    }

    Source& source;
    size_t min_size;
    size_t max_size;
    const Handlers& handlers;
    Stats& stats;
    Inflater inflater;
    std::vector<u8> out;
};

} // Anonymous namespace

std::string_view KindName(Kind kind) {
    switch (kind) {
    case Kind::Zstd:
        return "zstd frames";
    case Kind::Zlib:
        return "zlib streams";
    case Kind::Zip:
        return "ZIP entries";
    case Kind::Crilayla:
        return "CRILAYLA files";
    }
    return "?";
}

Source::Source(std::span<const u8> data_, u64 base_, u64 file_size_, const ReadFn* fetch_)
    : data{data_}, base{base_}, file_size{file_size_}, fetch{fetch_} {}

std::span<const u8> Source::Get(u64 offset, u64 size) {
    if (offset >= file_size) {
        return {};
    }
    size = std::min(size, file_size - offset);
    if (offset >= base && offset + size <= base + data.size()) {
        return data.subspan(static_cast<size_t>(offset - base), static_cast<size_t>(size));
    }
    if (!fetch) {
        // Only what is in memory.
        if (offset < base || offset >= base + data.size()) {
            return {};
        }
        return data.subspan(static_cast<size_t>(offset - base));
    }
    if (offset < window_offset || offset + size > window_offset + window.size()) {
        window = (*fetch)(offset, std::max(size, WINDOW));
        window_offset = offset;
    }
    if (offset + size > window_offset + window.size()) {
        size = window_offset + window.size() > offset ? window_offset + window.size() - offset : 0;
    }
    return std::span<const u8>{window}.subspan(static_cast<size_t>(offset - window_offset),
                                               static_cast<size_t>(size));
}

std::span<const u8> Source::Memory(u64 from, u64 to) const {
    from = std::max(from, base);
    to = std::min(to, base + data.size());
    if (from >= to) {
        return {};
    }
    return data.subspan(static_cast<size_t>(from - base), static_cast<size_t>(to - from));
}

bool UnpackCrilayla(std::span<const u8> in, std::vector<u8>& out) {
    if (in.size() < 16 + 0x100 || std::memcmp(in.data(), "CRILAYLA", 8) != 0) {
        return false;
    }
    const u32 size = Load32(in.data() + 8);
    const u32 packed_size = Load32(in.data() + 12);
    if (in.size() != 16 + u64{packed_size} + 0x100) {
        return false;
    }
    out.assign(size_t{size} + 0x100, 0);
    std::memcpy(out.data(), in.data() + 16 + packed_size, 0x100);
    // The packed bytes are read from their last one backwards, and fill the output from its
    // end backwards.
    s64 in_pos = 16 + s64{packed_size} - 1;
    u32 pool = 0;
    u32 bits_left = 0;
    bool overrun = false;
    const auto bits = [&](u32 count) {
        u32 value = 0;
        u32 produced = 0;
        while (produced < count) {
            if (bits_left == 0) {
                if (in_pos < 16) {
                    overrun = true;
                    return value;
                }
                pool = in[static_cast<size_t>(in_pos--)];
                bits_left = 8;
            }
            const u32 now = std::min(bits_left, count - produced);
            value = (value << now) | ((pool >> (bits_left - now)) & ((1u << now) - 1));
            bits_left -= now;
            produced += now;
        }
        return value;
    };
    const s64 output_end = 0x100 + s64{size} - 1;
    s64 written = 0;
    constexpr std::array<u32, 4> LENGTH_BITS{2, 3, 5, 8};
    while (written < s64{size}) {
        if (bits(1) != 0) {
            s64 from = output_end - written + bits(13) + 3;
            u32 length = 3;
            size_t level = 0;
            for (; level < LENGTH_BITS.size(); ++level) {
                const u32 part = bits(LENGTH_BITS[level]);
                length += part;
                if (part != (1u << LENGTH_BITS[level]) - 1) {
                    break;
                }
            }
            if (level == LENGTH_BITS.size()) {
                u32 part;
                do {
                    part = bits(8);
                    length += part;
                } while (part == 255 && !overrun);
            }
            if (overrun || from > output_end || length > s64{size} - written) {
                return false;
            }
            for (u32 i = 0; i < length; ++i) {
                out[static_cast<size_t>(output_end - written)] = out[static_cast<size_t>(from--)];
                ++written;
            }
        } else {
            const u32 value = bits(8);
            if (overrun) {
                return false;
            }
            out[static_cast<size_t>(output_end - written)] = static_cast<u8>(value);
            ++written;
        }
    }
    return true;
}

u64 ZstdFrameSize(Source& source, u64 offset) {
    return ReadZstdFrame(source, offset).size;
}

u64 FindStreams(Source& source, u64 from, u64 to, size_t min_size, size_t max_size,
                const Handlers& handlers, Stats& stats) {
    const std::span<const u8> memory = source.Memory(from, to);
    if (memory.empty()) {
        return from;
    }
    Finder finder{source, min_size, max_size, handlers, stats};
    const u64 start = std::max(from, source.Base());
    const u8* const begin = memory.data();
    const size_t end = memory.size();
    constexpr std::array<u8, 4> FIRST{0x28, 0x78, 'P', 'C'};
    std::array<const u8*, 4> next{};
    const auto find = [&](size_t which, size_t at) -> const u8* {
        return at < end ? static_cast<const u8*>(std::memchr(begin + at, FIRST[which], end - at))
                        : nullptr;
    };
    for (size_t which = 0; which < FIRST.size(); ++which) {
        next[which] = find(which, 0);
    }
    u64 found_end = from;
    size_t checks = 0;
    while (true) {
        const u8* place = nullptr;
        size_t which = 0;
        for (size_t i = 0; i < next.size(); ++i) {
            if (next[i] && (!place || next[i] < place)) {
                place = next[i];
                which = i;
            }
        }
        if (!place) {
            break;
        }
        if ((++checks & 0xFFF) == 0 && handlers.stop && handlers.stop()) {
            break;
        }
        const size_t at = static_cast<size_t>(place - begin);
        const u64 offset = start + at;
        const u64 packed = finder.Try(FIRST[which], offset);
        size_t resume = at + 1;
        if (packed != 0) {
            found_end = offset + packed;
            if (found_end >= to) {
                break;
            }
            resume = static_cast<size_t>(found_end - start);
        }
        for (size_t i = 0; i < next.size(); ++i) {
            if (next[i] && static_cast<size_t>(next[i] - begin) < resume) {
                next[i] = find(i, resume);
            }
        }
    }
    return found_end;
}

} // namespace Preforged::Embedded
