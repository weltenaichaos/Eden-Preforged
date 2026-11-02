// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>

#include <fmt/format.h>

#include "common/lz4_compression.h"
#include "jni/unity_assets.h"

namespace Preforged::Unity {
namespace {

constexpr u64 MAX_INFO_SIZE = u64{64} << 20;
constexpr u32 MAX_BLOCKS = 1u << 22;
constexpr u32 MAX_NODES = 1u << 20;
constexpr u32 MAX_PLATFORMS = 8;
constexpr u32 MAX_PARTS = 64;
constexpr u32 MAX_BLOB = 256u << 20;
constexpr u32 MAX_PART = 64u << 20;
/// Bundle flags.
constexpr u32 INFO_AT_END = 0x80;
constexpr u32 PADDING_BEFORE_DATA = 0x200;

u32 Load32(const u8* data) {
    u32 value;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

u64 AlignUp(u64 value, u64 alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

std::string Hex(std::span<const u8> data, size_t length) {
    std::string out;
    for (size_t i = 0; i < std::min(data.size(), length); ++i) {
        out += fmt::format("{:02x}", data[i]);
    }
    return out;
}

/// Reads the big-endian fields of a bundle header and its tables.
class BigEndianReader {
public:
    explicit BigEndianReader(std::span<const u8> data_) : data{data_} {}

    bool U16(u16& out) {
        if (data.size() - pos < 2) {
            return false;
        }
        out = static_cast<u16>((data[pos] << 8) | data[pos + 1]);
        pos += 2;
        return true;
    }

    bool U32(u32& out) {
        if (data.size() - pos < 4) {
            return false;
        }
        out = (u32{data[pos]} << 24) | (u32{data[pos + 1]} << 16) | (u32{data[pos + 2]} << 8) |
              u32{data[pos + 3]};
        pos += 4;
        return true;
    }

    bool U64(u64& out) {
        u32 high;
        u32 low;
        if (!U32(high) || !U32(low)) {
            return false;
        }
        out = (u64{high} << 32) | low;
        return true;
    }

    bool String(std::string& out) {
        const auto end = std::find(data.begin() + pos, data.end(), u8{0});
        if (end == data.end()) {
            return false;
        }
        out.assign(data.begin() + pos, end);
        pos = static_cast<size_t>(end - data.begin()) + 1;
        return true;
    }

    bool Skip(size_t count) {
        if (data.size() - pos < count) {
            return false;
        }
        pos += count;
        return true;
    }

    [[nodiscard]] size_t Position() const {
        return pos;
    }

private:
    std::span<const u8> data;
    size_t pos = 0;
};

bool UnpackLz4(std::span<const u8> in, std::span<u8> out) {
    const int size =
        Common::Compression::DecompressDataLZ4(out.data(), out.size(), in.data(), in.size());
    return size >= 0 && static_cast<size_t>(size) == out.size();
}

/// The fields of a Shader object after its platform list: per platform the offsets, packed
/// sizes and unpacked sizes of its parts in the blob, then the blob.
struct BlobLayout {
    std::vector<std::vector<u32>> offsets;
    std::vector<std::vector<u32>> packed;
    std::vector<std::vector<u32>> unpacked;
    size_t blob_start = 0; ///< In the data the layout was read from
    u32 blob_size = 0;
};

/// Reads the three lists in either layout: one part per platform (before Unity 2019.3), or a
/// list of parts per platform (2019.3 on).
std::optional<BlobLayout> ReadLayout(std::span<const u8> data, size_t pos, u32 platforms,
                                     bool nested) {
    BlobLayout layout;
    for (auto* list : {&layout.offsets, &layout.packed, &layout.unpacked}) {
        if (data.size() < pos + 4 || Load32(data.data() + pos) != platforms) {
            return std::nullopt;
        }
        pos += 4;
        list->resize(platforms);
        for (u32 platform = 0; platform < platforms; ++platform) {
            u32 parts = 1;
            if (nested) {
                if (data.size() < pos + 4) {
                    return std::nullopt;
                }
                parts = Load32(data.data() + pos);
                pos += 4;
                if (parts > MAX_PARTS) {
                    return std::nullopt;
                }
            }
            if (data.size() < pos + u64{parts} * 4) {
                return std::nullopt;
            }
            for (u32 part = 0; part < parts; ++part) {
                (*list)[platform].push_back(Load32(data.data() + pos));
                pos += 4;
            }
        }
    }
    for (u32 platform = 0; platform < platforms; ++platform) {
        if (layout.packed[platform].size() != layout.offsets[platform].size() ||
            layout.unpacked[platform].size() != layout.offsets[platform].size()) {
            return std::nullopt;
        }
    }
    if (data.size() < pos + 4) {
        return std::nullopt;
    }
    layout.blob_size = Load32(data.data() + pos);
    layout.blob_start = pos + 4;
    if (layout.blob_size == 0 || layout.blob_size > MAX_BLOB) {
        return std::nullopt;
    }
    for (u32 platform = 0; platform < platforms; ++platform) {
        for (size_t part = 0; part < layout.offsets[platform].size(); ++part) {
            const u64 offset = layout.offsets[platform][part];
            const u64 packed = layout.packed[platform][part];
            const u64 unpacked = layout.unpacked[platform][part];
            if (packed == 0 || unpacked == 0 || unpacked > MAX_PART ||
                offset + packed > layout.blob_size) {
                return std::nullopt;
            }
        }
    }
    return layout;
}

/// LZMA decoding as in the LZMA specification, into an output buffer that holds the whole
/// unpacked data and so serves as the dictionary.
class LzmaDecoder {
public:
    LzmaDecoder(std::span<const u8> in_, std::span<u8> out_) : in{in_}, out{out_} {}

    bool Decode(u32 lc_, u32 lp_, u32 pb_) {
        lc = lc_;
        lp = lp_;
        pb = pb_;
        literal_probs.assign(size_t{0x300} << (lc + lp), PROB_INIT);
        if (Byte() != 0) {
            return false;
        }
        for (int i = 0; i < 4; ++i) {
            code = (code << 8) | Byte();
        }
        if (code == range) {
            return false;
        }
        u32 state = 0;
        std::array<u32, 4> rep{};
        const u32 pb_mask = (1u << pb) - 1;
        while (pos < out.size()) {
            const u32 pos_state = static_cast<u32>(pos) & pb_mask;
            if (Bit(is_match[(state << 4) + pos_state]) == 0) {
                DecodeLiteral(state, rep[0]);
                state = state < 4 ? 0 : (state < 10 ? state - 3 : state - 6);
                continue;
            }
            u32 length;
            if (Bit(is_rep[state]) != 0) {
                if (pos == 0) {
                    return false;
                }
                if (Bit(is_rep_g0[state]) == 0) {
                    if (Bit(is_rep0_long[(state << 4) + pos_state]) == 0) {
                        state = state < 7 ? 9 : 11;
                        out[pos] = out[pos - rep[0] - 1];
                        ++pos;
                        continue;
                    }
                } else {
                    u32 distance;
                    if (Bit(is_rep_g1[state]) == 0) {
                        distance = rep[1];
                    } else {
                        if (Bit(is_rep_g2[state]) == 0) {
                            distance = rep[2];
                        } else {
                            distance = rep[3];
                            rep[3] = rep[2];
                        }
                        rep[2] = rep[1];
                    }
                    rep[1] = rep[0];
                    rep[0] = distance;
                }
                length = rep_length.Decode(*this, pos_state);
                state = state < 7 ? 8 : 11;
            } else {
                rep[3] = rep[2];
                rep[2] = rep[1];
                rep[1] = rep[0];
                length = match_length.Decode(*this, pos_state);
                state = state < 7 ? 7 : 10;
                rep[0] = DecodeDistance(length);
                if (rep[0] == 0xFFFFFFFF) {
                    // The end marker.
                    break;
                }
                if (rep[0] >= pos) {
                    return false;
                }
            }
            length += 2;
            const size_t from = pos - rep[0] - 1;
            const size_t count = std::min<size_t>(length, out.size() - pos);
            for (size_t i = 0; i < count; ++i) {
                out[pos + i] = out[from + i];
            }
            pos += count;
        }
        // The decoder reads as many bytes as the encoder wrote, its 5 flushed ones included.
        return pos == out.size() && overrun == 0;
    }

private:
    static constexpr u16 PROB_INIT = 1024;

    struct BitTree {
        std::vector<u16> probs;
        u32 bits;

        explicit BitTree(u32 bits_) : probs(size_t{1} << bits_, PROB_INIT), bits{bits_} {}

        u32 Decode(LzmaDecoder& decoder) {
            u32 m = 1;
            for (u32 i = 0; i < bits; ++i) {
                m = (m << 1) + decoder.Bit(probs[m]);
            }
            return m - (1u << bits);
        }
    };

    struct LengthDecoder {
        u16 choice = PROB_INIT;
        u16 choice2 = PROB_INIT;
        std::vector<BitTree> low = std::vector<BitTree>(16, BitTree{3});
        std::vector<BitTree> mid = std::vector<BitTree>(16, BitTree{3});
        BitTree high{8};

        u32 Decode(LzmaDecoder& decoder, u32 pos_state) {
            if (decoder.Bit(choice) == 0) {
                return low[pos_state].Decode(decoder);
            }
            if (decoder.Bit(choice2) == 0) {
                return 8 + mid[pos_state].Decode(decoder);
            }
            return 16 + high.Decode(decoder);
        }
    };

    u8 Byte() {
        if (in_pos < in.size()) {
            return in[in_pos++];
        }
        ++overrun;
        return 0;
    }

    void Normalize() {
        if (range < (1u << 24)) {
            range <<= 8;
            code = (code << 8) | Byte();
        }
    }

    u32 Bit(u16& prob) {
        const u32 bound = (range >> 11) * prob;
        u32 bit;
        if (code < bound) {
            prob = static_cast<u16>(prob + ((2048 - prob) >> 5));
            range = bound;
            bit = 0;
        } else {
            prob = static_cast<u16>(prob - (prob >> 5));
            code -= bound;
            range -= bound;
            bit = 1;
        }
        Normalize();
        return bit;
    }

    u32 DirectBits(u32 count) {
        u32 result = 0;
        for (u32 i = 0; i < count; ++i) {
            range >>= 1;
            code -= range;
            const u32 t = 0u - (code >> 31);
            code += range & t;
            Normalize();
            result = (result << 1) + (t + 1);
        }
        return result;
    }

    u32 ReverseBits(u16* probs, u32 count) {
        u32 m = 1;
        u32 symbol = 0;
        for (u32 i = 0; i < count; ++i) {
            const u32 bit = Bit(probs[m]);
            m = (m << 1) + bit;
            symbol |= bit << i;
        }
        return symbol;
    }

    void DecodeLiteral(u32 state, u32 rep0) {
        const u32 previous = pos > 0 ? out[pos - 1] : 0;
        const u32 literal_state =
            ((static_cast<u32>(pos) & ((1u << lp) - 1)) << lc) + (previous >> (8 - lc));
        u16* const probs = literal_probs.data() + size_t{0x300} * literal_state;
        u32 symbol = 1;
        if (state >= 7 && pos > rep0) {
            u32 match_byte = out[pos - rep0 - 1];
            do {
                const u32 match_bit = (match_byte >> 7) & 1;
                match_byte <<= 1;
                const u32 bit = Bit(probs[((1 + match_bit) << 8) + symbol]);
                symbol = (symbol << 1) | bit;
                if (match_bit != bit) {
                    break;
                }
            } while (symbol < 0x100);
        }
        while (symbol < 0x100) {
            symbol = (symbol << 1) | Bit(probs[symbol]);
        }
        out[pos++] = static_cast<u8>(symbol - 0x100);
    }

    u32 DecodeDistance(u32 length) {
        const u32 length_state = std::min<u32>(length, 3);
        const u32 slot = pos_slot[length_state].Decode(*this);
        if (slot < 4) {
            return slot;
        }
        const u32 direct = (slot >> 1) - 1;
        u32 distance = (2 | (slot & 1)) << direct;
        if (slot < 14) {
            distance += ReverseBits(pos_probs.data() + distance - slot, direct);
        } else {
            distance += DirectBits(direct - 4) << 4;
            distance += ReverseBits(align.data(), 4);
        }
        return distance;
    }

    std::span<const u8> in;
    std::span<u8> out;
    size_t in_pos = 0;
    size_t pos = 0;
    size_t overrun = 0;
    u32 range = 0xFFFFFFFF;
    u32 code = 0;
    u32 lc = 0;
    u32 lp = 0;
    u32 pb = 0;
    std::vector<u16> literal_probs;
    std::array<u16, 12 << 4> is_match = Probs<12 << 4>();
    std::array<u16, 12> is_rep = Probs<12>();
    std::array<u16, 12> is_rep_g0 = Probs<12>();
    std::array<u16, 12> is_rep_g1 = Probs<12>();
    std::array<u16, 12> is_rep_g2 = Probs<12>();
    std::array<u16, 12 << 4> is_rep0_long = Probs<12 << 4>();
    std::vector<BitTree> pos_slot = std::vector<BitTree>(4, BitTree{6});
    std::array<u16, 1 + 128 - 14> pos_probs = Probs<1 + 128 - 14>();
    std::array<u16, 16> align = Probs<16>();
    LengthDecoder match_length;
    LengthDecoder rep_length;

    template <size_t N>
    static constexpr std::array<u16, N> Probs() {
        std::array<u16, N> probs{};
        probs.fill(PROB_INIT);
        return probs;
    }
};

bool UnpackBlock(u32 method, std::span<const u8> in, std::span<u8> out) {
    switch (method) {
    case 0:
        if (in.size() != out.size()) {
            return false;
        }
        std::memcpy(out.data(), in.data(), in.size());
        return true;
    case 1:
        return UnpackLzma(in, out);
    case 2:
    case 3:
        return UnpackLz4(in, out);
    default:
        return false;
    }
}

} // Anonymous namespace

bool IsBundle(std::span<const u8> start) {
    return start.size() >= 8 && std::memcmp(start.data(), "UnityFS\0", 8) == 0;
}

bool IsSerializedFile(std::span<const u8> start, u64 size) {
    if (start.size() < 0x40) {
        return false;
    }
    BigEndianReader reader{start};
    u32 metadata_size;
    u32 file_size;
    u32 version;
    u32 data_offset;
    reader.U32(metadata_size);
    reader.U32(file_size);
    reader.U32(version);
    reader.U32(data_offset);
    // Unity 5 wrote version 9 on; 2020 and later write 21 and 22.
    if (version < 9 || version > 64) {
        return false;
    }
    reader.Skip(4);
    size_t string_start = 0x14;
    if (version >= 22) {
        u32 metadata_size_22;
        u64 file_size_22;
        reader.U32(metadata_size_22);
        if (!reader.U64(file_size_22) || file_size_22 != size) {
            return false;
        }
        string_start = 0x30;
    } else if (file_size != size) {
        return false;
    }
    // The Unity version, such as 2019.4.31f1.
    size_t end = string_start;
    while (end < start.size() && start[end] != 0) {
        const u8 c = start[end];
        if (c < 0x20 || c > 0x7E) {
            return false;
        }
        ++end;
    }
    const size_t length = end - string_start;
    return end < start.size() && length >= 3 && length <= 32 && start[string_start] >= '0' &&
           start[string_start] <= '9' &&
           std::find(start.begin() + string_start, start.begin() + end, '.') !=
               start.begin() + end;
}

bool UnpackLzma(std::span<const u8> in, std::span<u8> out) {
    if (in.size() < 5 + 5) {
        return false;
    }
    u32 properties = in[0];
    if (properties >= 9 * 5 * 5) {
        return false;
    }
    const u32 lc = properties % 9;
    properties /= 9;
    const u32 lp = properties % 5;
    const u32 pb = properties / 5;
    LzmaDecoder decoder{in.subspan(5), out};
    return decoder.Decode(lc, lp, pb);
}

std::optional<Bundle> ReadBundle(const ReadFn& read, u64 file_size, std::string& error) {
    const std::vector<u8> head = read(0, std::min<u64>(file_size, 0x1000));
    if (!IsBundle(head)) {
        error = "not a UnityFS bundle";
        return std::nullopt;
    }
    Bundle bundle;
    BigEndianReader reader{head};
    reader.Skip(8);
    u64 size;
    u32 packed_info;
    u32 unpacked_info;
    u32 flags;
    if (!reader.U32(bundle.version) || !reader.String(bundle.unity_version) ||
        !reader.String(bundle.revision) || !reader.U64(size) || !reader.U32(packed_info) ||
        !reader.U32(unpacked_info) || !reader.U32(flags)) {
        error = fmt::format("the header could not be read ({})", Hex(head, 64));
        return std::nullopt;
    }
    u64 pos = reader.Position();
    if (bundle.version >= 7) {
        pos = AlignUp(pos, 16);
    }
    if (packed_info == 0 || packed_info > MAX_INFO_SIZE || unpacked_info > MAX_INFO_SIZE ||
        packed_info > file_size) {
        error = fmt::format("the block table has {} -> {} bytes", packed_info, unpacked_info);
        return std::nullopt;
    }
    const bool info_at_end = (flags & INFO_AT_END) != 0;
    const u64 info_offset = info_at_end ? file_size - packed_info : pos;
    u64 data_start = info_at_end ? pos : pos + packed_info;
    if ((flags & PADDING_BEFORE_DATA) != 0) {
        data_start = AlignUp(data_start, 16);
    }
    const std::vector<u8> packed = read(info_offset, packed_info);
    if (packed.size() != packed_info) {
        error = "the block table runs past the end of the file";
        return std::nullopt;
    }
    std::vector<u8> info(unpacked_info);
    if (!UnpackBlock(flags & 0x3F, packed, info)) {
        error = fmt::format("the block table ({} -> {} bytes) did not unpack with method {} (0 "
                            "stored, 1 LZMA, 2 and 3 LZ4)",
                            packed_info, unpacked_info, flags & 0x3F);
        return std::nullopt;
    }
    BigEndianReader table{info};
    u32 block_count;
    if (!table.Skip(16) || !table.U32(block_count) || block_count > MAX_BLOCKS) {
        error = "the block table starts with no block count";
        return std::nullopt;
    }
    u64 file_offset = data_start;
    u64 stream_offset = 0;
    for (u32 i = 0; i < block_count; ++i) {
        BundleBlock block{};
        if (!table.U32(block.unpacked_size) || !table.U32(block.packed_size) ||
            !table.U16(block.flags)) {
            error = fmt::format("block {} of {} could not be read", i, block_count);
            return std::nullopt;
        }
        block.file_offset = file_offset;
        block.stream_offset = stream_offset;
        file_offset += block.packed_size;
        stream_offset += block.unpacked_size;
        bundle.blocks.push_back(block);
    }
    if (file_offset > file_size) {
        error = fmt::format("the blocks run past the end of the file ({} of {} bytes)",
                            file_offset, file_size);
        return std::nullopt;
    }
    u32 node_count;
    if (!table.U32(node_count) || node_count > MAX_NODES) {
        error = "the block table has no file count";
        return std::nullopt;
    }
    for (u32 i = 0; i < node_count; ++i) {
        BundleNode node;
        if (!table.U64(node.offset) || !table.U64(node.size) || !table.U32(node.flags) ||
            !table.String(node.path)) {
            error = fmt::format("file {} of {} could not be read", i, node_count);
            return std::nullopt;
        }
        if (node.offset > stream_offset || node.size > stream_offset - node.offset) {
            error = fmt::format("{} lies past the unpacked data", node.path);
            return std::nullopt;
        }
        bundle.nodes.push_back(std::move(node));
    }
    return bundle;
}

bool ReadBundleData(const ReadFn& read, const Bundle& bundle, u64 offset, u64 size,
                    std::vector<u8>& out, BlockCache& cache, std::string& error) {
    out.assign(size, 0);
    if (size == 0) {
        return true;
    }
    const auto after = std::ranges::upper_bound(bundle.blocks, offset, {},
                                                [](const BundleBlock& block) {
                                                    return block.stream_offset;
                                                });
    if (after == bundle.blocks.begin()) {
        error = "the data starts before the first block";
        return false;
    }
    std::vector<u8> packed;
    for (auto it = after - 1; it != bundle.blocks.end() && it->stream_offset < offset + size;
         ++it) {
        const BundleBlock& block = *it;
        const size_t index = static_cast<size_t>(it - bundle.blocks.begin());
        if (cache.index != index) {
            const u32 method = block.flags & 0x3F;
            if (block.unpacked_size > MAX_BLOCK) {
                error = fmt::format("a block at {} unpacks to {} MiB, more than is unpacked",
                                    block.file_offset, block.unpacked_size >> 20);
                return false;
            }
            packed = read(block.file_offset, block.packed_size);
            if (packed.size() != block.packed_size) {
                error =
                    fmt::format("a block at {} runs past the end of the file", block.file_offset);
                return false;
            }
            cache.index = static_cast<size_t>(-1);
            cache.data.resize(block.unpacked_size);
            if (!UnpackBlock(method, packed, cache.data)) {
                error = fmt::format("a block at {} ({} -> {} bytes, method {}: 0 stored, 1 LZMA, "
                                    "2 and 3 LZ4) did not unpack",
                                    block.file_offset, block.packed_size, block.unpacked_size,
                                    method);
                return false;
            }
            cache.index = index;
            ++cache.unpacked[std::min<u32>(method, 2)];
            if (method == 1) {
                cache.lzma_bytes += block.unpacked_size;
            }
        }
        const u64 from = std::max(offset, block.stream_offset);
        const u64 to = std::min(offset + size, block.stream_offset + block.unpacked_size);
        if (to > from) {
            std::memcpy(out.data() + (from - offset),
                        cache.data.data() + (from - block.stream_offset), to - from);
        }
    }
    return true;
}

void FindShaderBlobs(std::span<const u8> data, u64 base_offset,
                     const std::function<void(std::span<const u8> part, u64 offset)>& found,
                     BlobStats& stats, const ReadFn* fetch, std::unordered_set<u64>& seen) {
    if (data.size() < 16) {
        return;
    }
    std::vector<u8> fetched;
    std::vector<u8> part;
    for (size_t pos = 0; pos + 8 <= data.size(); pos += 4) {
        const u32 platforms = Load32(data.data() + pos);
        if (platforms == 0 || platforms > MAX_PLATFORMS ||
            data.size() < pos + 4 + u64{platforms} * 4) {
            continue;
        }
        std::vector<size_t> switch_platforms;
        bool small = true;
        for (u32 i = 0; i < platforms; ++i) {
            const u32 platform = Load32(data.data() + pos + 4 + i * 4);
            small &= platform < 64;
            if (platform == SWITCH_PLATFORM) {
                switch_platforms.push_back(i);
            }
        }
        if (!small || switch_platforms.empty()) {
            continue;
        }
        ++stats.candidates;
        const size_t lists = pos + 4 + size_t{platforms} * 4;
        for (const bool nested : {true, false}) {
            const auto layout = ReadLayout(data, lists, platforms, nested);
            if (!layout) {
                continue;
            }
            const u64 blob_offset = base_offset + layout->blob_start;
            if (seen.contains(blob_offset)) {
                break;
            }
            std::span<const u8> blob;
            if (layout->blob_start + u64{layout->blob_size} <= data.size()) {
                blob = data.subspan(layout->blob_start, layout->blob_size);
            } else if (fetch) {
                fetched = (*fetch)(blob_offset, layout->blob_size);
                if (fetched.size() != layout->blob_size) {
                    continue;
                }
                blob = fetched;
            } else {
                continue;
            }
            size_t unpacked_parts = 0;
            for (const size_t platform : switch_platforms) {
                for (size_t i = 0; i < layout->offsets[platform].size(); ++i) {
                    const auto in = blob.subspan(layout->offsets[platform][i],
                                                 layout->packed[platform][i]);
                    part.resize(layout->unpacked[platform][i]);
                    // A part as large packed as unpacked may have been stored.
                    const bool ok =
                        UnpackLz4(in, part) ||
                        (in.size() == part.size() &&
                         (std::memcpy(part.data(), in.data(), in.size()), true));
                    if (!ok) {
                        ++stats.failed;
                        if (stats.first_failure.empty()) {
                            stats.first_failure = fmt::format(
                                "{} -> {} bytes at {:#x}, starts {}", in.size(), part.size(),
                                blob_offset + layout->offsets[platform][i], Hex(in, 8));
                        }
                        continue;
                    }
                    ++unpacked_parts;
                    ++stats.parts;
                    stats.unpacked += part.size();
                    if (stats.first_part.empty()) {
                        stats.first_part = Hex(part, 64);
                    }
                    found(part, blob_offset);
                }
            }
            if (unpacked_parts == 0) {
                continue;
            }
            ++stats.blobs;
            seen.insert(blob_offset);
            // The blob holds no other Shader object.
            if (layout->blob_start + u64{layout->blob_size} <= data.size()) {
                pos = static_cast<size_t>(AlignUp(layout->blob_start + layout->blob_size, 4)) - 4;
            }
            break;
        }
    }
}

} // namespace Preforged::Unity
