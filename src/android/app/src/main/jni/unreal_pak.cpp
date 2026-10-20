// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>

#include <fmt/format.h>

#include "jni/unreal_package.h"
#include "jni/unreal_pak.h"

namespace Preforged::UnrealPak {
namespace {

using Preforged::Unreal::Method;

constexpr size_t AES_BLOCK = 16;
/// Bytes of the footer from the magic on: magic, version, index offset and size, index hash.
constexpr size_t FOOTER_CORE = 4 + 4 + 8 + 8 + 20;
constexpr size_t METHOD_NAME_SIZE = 32;
constexpr u32 MAX_VERSION = 12;
constexpr u64 MAX_INDEX_SIZE = u64{1} << 30;
constexpr u64 MAX_ENTRIES = u64{1} << 24;
constexpr u64 MAX_BLOCKS = u64{1} << 20;
constexpr s32 MAX_STRING = 1 << 16;
/// Entries whose packed bytes span more than this are read block by block.
constexpr u64 MAX_SINGLE_READ = u64{512} << 20;
constexpr u64 MAX_BLOCK_PACKED = u64{64} << 20;
constexpr u32 MAX_ARCHIVE_COUNT = 1u << 22;
constexpr u32 MAX_SHADER_SIZE = 16u << 20;

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

/// Reads the little-endian fields of an index with bounds checks.
class Reader {
public:
    explicit Reader(std::span<const u8> data_) : data{data_} {}

    template <typename T>
    bool Read(T& value) {
        if (data.size() - pos < sizeof(T)) {
            return false;
        }
        std::memcpy(&value, data.data() + pos, sizeof(T));
        pos += sizeof(T);
        return true;
    }

    bool Skip(u64 count) {
        if (data.size() - pos < count) {
            return false;
        }
        pos += count;
        return true;
    }

    /// An FString: its length with the terminating zero, then 8-bit characters, or for a
    /// negative length that many UTF-16 characters. Characters past ASCII become '?'.
    bool ReadString(std::string& out) {
        s32 length;
        if (!Read(length) || length > MAX_STRING || length < -MAX_STRING) {
            return false;
        }
        out.clear();
        if (length == 0) {
            return true;
        }
        if (length > 0) {
            if (data.size() - pos < static_cast<u64>(length)) {
                return false;
            }
            out.assign(reinterpret_cast<const char*>(data.data() + pos),
                       static_cast<size_t>(length));
            pos += static_cast<size_t>(length);
        } else {
            const u64 chars = static_cast<u64>(-static_cast<s64>(length));
            if (data.size() - pos < chars * 2) {
                return false;
            }
            for (u64 i = 0; i < chars; ++i) {
                const u16 c = static_cast<u16>(data[pos + i * 2] | (data[pos + i * 2 + 1] << 8));
                out.push_back(c < 0x80 ? static_cast<char>(c) : '?');
            }
            pos += chars * 2;
        }
        while (!out.empty() && out.back() == '\0') {
            out.pop_back();
        }
        return true;
    }

    [[nodiscard]] size_t Position() const {
        return pos;
    }

    [[nodiscard]] size_t Left() const {
        return data.size() - pos;
    }

    [[nodiscard]] std::span<const u8> Rest() const {
        return data.subspan(pos);
    }

private:
    std::span<const u8> data;
    size_t pos = 0;
};

// AES-256, the inverse cipher of FIPS-197, byte by byte. Fast enough for the key search and
// for the indexes; the scanner decrypts file data with the emulator's OpenSSL instead.
u8 Xtime(u8 x) {
    return static_cast<u8>((x << 1) ^ ((x & 0x80) != 0 ? 0x1B : 0));
}

u8 Multiply(u8 a, u8 b) {
    u8 result = 0;
    while (b != 0) {
        if ((b & 1) != 0) {
            result ^= a;
        }
        a = Xtime(a);
        b >>= 1;
    }
    return result;
}

struct AesTables {
    std::array<u8, 256> sbox{};
    std::array<u8, 256> inverse{};
    std::array<u8, 256> mul9{};
    std::array<u8, 256> mul11{};
    std::array<u8, 256> mul13{};
    std::array<u8, 256> mul14{};
};

AesTables MakeAesTables() {
    AesTables tables;
    // Powers of 3 walk the whole field, so each element's inverse is found from them.
    std::array<u8, 256> power{};
    std::array<u8, 256> log{};
    u8 value = 1;
    for (int i = 0; i < 255; ++i) {
        power[i] = value;
        log[value] = static_cast<u8>(i);
        value = static_cast<u8>(value ^ Xtime(value));
    }
    for (int i = 0; i < 256; ++i) {
        const u8 inverse = i == 0 ? 0 : power[(255 - log[i]) % 255];
        const u8 s = static_cast<u8>(inverse ^ std::rotl(inverse, 1) ^ std::rotl(inverse, 2) ^
                                     std::rotl(inverse, 3) ^ std::rotl(inverse, 4) ^ 0x63);
        tables.sbox[i] = s;
    }
    for (int i = 0; i < 256; ++i) {
        tables.inverse[tables.sbox[i]] = static_cast<u8>(i);
        tables.mul9[i] = Multiply(static_cast<u8>(i), 9);
        tables.mul11[i] = Multiply(static_cast<u8>(i), 11);
        tables.mul13[i] = Multiply(static_cast<u8>(i), 13);
        tables.mul14[i] = Multiply(static_cast<u8>(i), 14);
    }
    return tables;
}

const AesTables& Tables() {
    static const AesTables tables = MakeAesTables();
    return tables;
}

u32 SubWord(u32 word) {
    const auto& sbox = Tables().sbox;
    return (u32{sbox[word >> 24]} << 24) | (u32{sbox[(word >> 16) & 0xFF]} << 16) |
           (u32{sbox[(word >> 8) & 0xFF]} << 8) | u32{sbox[word & 0xFF]};
}

/// Bytes of a file record in front of an entry's data, and the record read from the index.
bool ReadRecord(Reader& reader, u32 version, Entry& entry, bool& deleted) {
    s64 offset;
    s64 size;
    s64 uncompressed_size;
    if (!reader.Read(offset) || !reader.Read(size) || !reader.Read(uncompressed_size) ||
        offset < 0 || size < 0 || uncompressed_size < 0) {
        return false;
    }
    entry.offset = static_cast<u64>(offset);
    entry.size = static_cast<u64>(size);
    entry.uncompressed_size = static_cast<u64>(uncompressed_size);
    if (version < 8) {
        // Flags: 1 zlib, 2 gzip, 4 the game's own method.
        s32 legacy;
        if (!reader.Read(legacy)) {
            return false;
        }
        entry.method = legacy == 0                ? 0
                       : (legacy & 0x01) != 0     ? 1
                       : (legacy & 0x02) != 0     ? 2
                                                  : 3;
    } else {
        u32 method;
        if (!reader.Read(method)) {
            return false;
        }
        entry.method = method;
    }
    if (version == 1 && !reader.Skip(8)) {
        return false;
    }
    if (!reader.Skip(20)) {
        return false;
    }
    entry.blocks.clear();
    deleted = false;
    if (version >= 3) {
        if (entry.method != 0) {
            s32 count;
            if (!reader.Read(count) || count < 0 || static_cast<u64>(count) > MAX_BLOCKS) {
                return false;
            }
            // Since version 5 the blocks count from the start of the entry's record.
            const u64 base = version >= 5 ? entry.offset : 0;
            for (s32 i = 0; i < count; ++i) {
                s64 start;
                s64 end;
                if (!reader.Read(start) || !reader.Read(end) || start < 0 || end < start) {
                    return false;
                }
                entry.blocks.push_back(
                    {base + static_cast<u64>(start), base + static_cast<u64>(end)});
            }
        }
        u8 flags;
        u32 block_size;
        if (!reader.Read(flags) || !reader.Read(block_size)) {
            return false;
        }
        entry.encrypted = (flags & 0x01) != 0;
        deleted = (flags & 0x02) != 0;
        entry.block_size = block_size;
    } else if (entry.method != 0) {
        // Before version 3 a packed entry is one block.
        entry.blocks.push_back({entry.offset + entry.RecordSize(version),
                                entry.offset + entry.RecordSize(version) + entry.size});
        entry.block_size = static_cast<u32>(
            std::min<u64>(entry.uncompressed_size, std::numeric_limits<u32>::max()));
    }
    return true;
}

/// Reads one entry of the bit-packed list of version 10 on (FPakFile::DecodePakEntry).
bool DecodeEntry(std::span<const u8> encoded, size_t& pos, u32 version, Entry& entry) {
    const auto take32{[&](u64& out) {
        if (encoded.size() - pos < 4) {
            return false;
        }
        out = Load32(encoded.data() + pos);
        pos += 4;
        return true;
    }};
    const auto take64{[&](u64& out) {
        if (encoded.size() - pos < 8) {
            return false;
        }
        out = Load64(encoded.data() + pos);
        pos += 8;
        return true;
    }};
    u64 value;
    if (pos > encoded.size() || !take32(value)) {
        return false;
    }
    u64 block_size = (value & 0x3F) << 11;
    if ((value & 0x3F) == 0x3F && !take32(block_size)) {
        return false;
    }
    entry.method = static_cast<u32>((value >> 23) & 0x3F);
    if (!(((value >> 31) & 1) != 0 ? take32(entry.offset) : take64(entry.offset))) {
        return false;
    }
    if (!(((value >> 30) & 1) != 0 ? take32(entry.uncompressed_size)
                                   : take64(entry.uncompressed_size))) {
        return false;
    }
    if (entry.method != 0) {
        if (!(((value >> 29) & 1) != 0 ? take32(entry.size) : take64(entry.size))) {
            return false;
        }
    } else {
        entry.size = entry.uncompressed_size;
    }
    entry.encrypted = ((value >> 22) & 1) != 0;
    const u64 count = (value >> 6) & 0xFFFF;
    entry.blocks.assign(count, Block{0, 0});
    entry.block_size = count == 0   ? 0
                       : count == 1 ? static_cast<u32>(std::min<u64>(
                                          entry.uncompressed_size, std::numeric_limits<u32>::max()))
                                    : static_cast<u32>(block_size);
    const u64 record = entry.RecordSize(version);
    if (count == 1 && !entry.encrypted) {
        entry.blocks[0] = {entry.offset + record, entry.offset + record + entry.size};
    } else if (count > 0) {
        u64 start = entry.offset + record;
        for (u64 i = 0; i < count; ++i) {
            u64 packed;
            if (!take32(packed)) {
                return false;
            }
            entry.blocks[i] = {start, start + packed};
            start += AlignUp(packed, entry.encrypted ? AES_BLOCK : 1);
        }
    }
    return true;
}

std::string JoinPath(const std::string& mount_point, std::string_view directory,
                     std::string_view name) {
    std::string path = mount_point;
    for (const std::string_view part : {directory, name}) {
        if (!path.empty() && path.back() == '/' && !part.empty() && part.front() == '/') {
            path.append(part.substr(1));
        } else {
            path.append(part);
        }
    }
    return path;
}

/// Whether 32 bytes have at least 24 different values, as a random key does almost always.
bool LooksRandom(const u8* data) {
    std::array<u64, 4> seen{};
    size_t distinct = 0;
    for (size_t i = 0; i < 32; ++i) {
        u64& word = seen[data[i] >> 6];
        const u64 bit = u64{1} << (data[i] & 63);
        distinct += (word & bit) == 0 ? 1 : 0;
        word |= bit;
    }
    return distinct >= 24;
}

bool IsPrintable(u8 c) {
    return c >= 0x20 && c < 0x7F;
}

} // Anonymous namespace

Aes256::Aes256(const Key& key) {
    for (size_t i = 0; i < 8; ++i) {
        round_keys[i] = (u32{key[i * 4]} << 24) | (u32{key[i * 4 + 1]} << 16) |
                        (u32{key[i * 4 + 2]} << 8) | u32{key[i * 4 + 3]};
    }
    u32 rcon = 0x01;
    for (size_t i = 8; i < round_keys.size(); ++i) {
        u32 word = round_keys[i - 1];
        if (i % 8 == 0) {
            word = SubWord(std::rotl(word, 8)) ^ (rcon << 24);
            rcon = Xtime(static_cast<u8>(rcon));
        } else if (i % 8 == 4) {
            word = SubWord(word);
        }
        round_keys[i] = round_keys[i - 8] ^ word;
    }
}

void Aes256::DecryptBlock(const u8* in, u8* out) const {
    const AesTables& tables = Tables();
    std::array<u8, 16> state;
    std::memcpy(state.data(), in, state.size());
    const auto add_round_key{[&](size_t round) {
        for (size_t column = 0; column < 4; ++column) {
            const u32 word = round_keys[round * 4 + column];
            state[column * 4] ^= static_cast<u8>(word >> 24);
            state[column * 4 + 1] ^= static_cast<u8>(word >> 16);
            state[column * 4 + 2] ^= static_cast<u8>(word >> 8);
            state[column * 4 + 3] ^= static_cast<u8>(word);
        }
    }};
    const auto inverse_shift_and_sub{[&] {
        std::array<u8, 16> shifted;
        for (size_t row = 0; row < 4; ++row) {
            for (size_t column = 0; column < 4; ++column) {
                shifted[((column + row) % 4) * 4 + row] = tables.inverse[state[column * 4 + row]];
            }
        }
        state = shifted;
    }};
    add_round_key(14);
    for (size_t round = 13; round >= 1; --round) {
        inverse_shift_and_sub();
        add_round_key(round);
        for (size_t column = 0; column < 4; ++column) {
            u8* const c = state.data() + column * 4;
            const u8 a0 = c[0];
            const u8 a1 = c[1];
            const u8 a2 = c[2];
            const u8 a3 = c[3];
            c[0] = tables.mul14[a0] ^ tables.mul11[a1] ^ tables.mul13[a2] ^ tables.mul9[a3];
            c[1] = tables.mul9[a0] ^ tables.mul14[a1] ^ tables.mul11[a2] ^ tables.mul13[a3];
            c[2] = tables.mul13[a0] ^ tables.mul9[a1] ^ tables.mul14[a2] ^ tables.mul11[a3];
            c[3] = tables.mul11[a0] ^ tables.mul13[a1] ^ tables.mul9[a2] ^ tables.mul14[a3];
        }
    }
    inverse_shift_and_sub();
    add_round_key(0);
    std::memcpy(out, state.data(), state.size());
}

void Aes256::Decrypt(std::span<u8> data) const {
    for (size_t pos = 0; pos + AES_BLOCK <= data.size(); pos += AES_BLOCK) {
        DecryptBlock(data.data() + pos, data.data() + pos);
    }
}

std::optional<Footer> ReadFooter(std::span<const u8> tail, u64 file_size) {
    if (tail.size() < FOOTER_CORE || tail.size() > file_size) {
        return std::nullopt;
    }
    for (size_t pos = tail.size() - FOOTER_CORE + 1; pos-- > 0;) {
        if (Load32(tail.data() + pos) != PAK_MAGIC) {
            continue;
        }
        Footer footer;
        footer.version = Load32(tail.data() + pos + 4);
        if (footer.version == 0 || footer.version > MAX_VERSION) {
            continue;
        }
        footer.index_offset = Load64(tail.data() + pos + 8);
        footer.index_size = Load64(tail.data() + pos + 16);
        size_t after = pos + FOOTER_CORE;
        if (footer.version == 9) {
            if (after >= tail.size()) {
                continue;
            }
            footer.frozen_index = tail[after] != 0;
            ++after;
        }
        const size_t rest = tail.size() - after;
        if (footer.version >= 8) {
            // 4 names up to engine 4.21, 5 from 4.22 on.
            if (rest % METHOD_NAME_SIZE != 0 || rest / METHOD_NAME_SIZE < 4 ||
                rest / METHOD_NAME_SIZE > 5) {
                continue;
            }
            bool names_ok = true;
            for (size_t name = 0; name < rest / METHOD_NAME_SIZE; ++name) {
                const u8* const start = tail.data() + after + name * METHOD_NAME_SIZE;
                const u8* const end = std::find(start, start + METHOD_NAME_SIZE, u8{0});
                names_ok &= std::all_of(start, end, IsPrintable);
                footer.methods.emplace_back(reinterpret_cast<const char*>(start),
                                            static_cast<size_t>(end - start));
            }
            if (!names_ok) {
                continue;
            }
        } else {
            if (rest != 0) {
                continue;
            }
            footer.methods = {"Zlib", "Gzip", "Oodle"};
        }
        size_t start = pos;
        if (footer.version >= 4) {
            if (pos < 1 || tail[pos - 1] > 1) {
                continue;
            }
            footer.encrypted_index = tail[pos - 1] != 0;
            --start;
        }
        if (footer.version >= 7) {
            // The key's GUID.
            if (start < 16) {
                continue;
            }
            start -= 16;
        }
        footer.size = static_cast<u32>(tail.size() - start);
        const u64 footer_start = file_size - footer.size;
        if (footer.index_size == 0 || footer.index_size > MAX_INDEX_SIZE ||
            footer.index_offset > footer_start ||
            footer.index_size > footer_start - footer.index_offset) {
            continue;
        }
        return footer;
    }
    return std::nullopt;
}

u64 Entry::RecordSize(u32 version) const {
    u64 bytes = 8 + 8 + 8 + 4 + 20;
    if (version == 1) {
        bytes += 8;
    }
    if (version >= 3) {
        bytes += 1 + 4;
        if (method != 0) {
            bytes += 4 + 16 * blocks.size();
        }
    }
    return bytes;
}

Index ReadIndex(const Footer& footer, std::span<const u8> primary, const IndexPartFn& read_part) {
    Index index;
    Reader reader{primary};
    s32 count;
    if (!reader.ReadString(index.mount_point) || !reader.Read(count) || count < 0 ||
        static_cast<u64>(count) > MAX_ENTRIES) {
        index.error = fmt::format("the index starts with no mount point and entry count ({})",
                                  Hex(primary, 32));
        return index;
    }
    index.declared = static_cast<u64>(count);
    if (footer.version < 10) {
        for (s32 i = 0; i < count; ++i) {
            std::string name;
            Entry entry;
            bool deleted;
            if (!reader.ReadString(name) || !ReadRecord(reader, footer.version, entry, deleted)) {
                index.error = fmt::format("entry {} of {} could not be read (at index byte {})",
                                          i, count, reader.Position());
                return index;
            }
            if (!deleted) {
                entry.path = JoinPath(index.mount_point, "", name);
                index.entries.push_back(std::move(entry));
            }
        }
        return index;
    }
    // Version 10 on: a list of bit-packed entries, the entries that didn't fit it, and a
    // directory of names elsewhere in the file pointing into both.
    u64 seed;
    u32 has_hash_index;
    if (!reader.Read(seed) || !reader.Read(has_hash_index)) {
        index.error = "the index ends before its path hash seed";
        return index;
    }
    if (has_hash_index != 0 && !reader.Skip(8 + 8 + 20)) {
        index.error = "the index ends in the path hash index's place";
        return index;
    }
    u32 has_directory;
    s64 directory_offset = 0;
    s64 directory_size = 0;
    if (!reader.Read(has_directory) ||
        (has_directory != 0 &&
         (!reader.Read(directory_offset) || !reader.Read(directory_size) || !reader.Skip(20)))) {
        index.error = "the index ends in the directory index's place";
        return index;
    }
    s32 encoded_size;
    if (!reader.Read(encoded_size) || encoded_size < 0 ||
        reader.Left() < static_cast<u64>(encoded_size)) {
        index.error = "the list of packed entries runs past the index";
        return index;
    }
    const std::span<const u8> encoded = reader.Rest().first(static_cast<size_t>(encoded_size));
    reader.Skip(static_cast<u64>(encoded_size));
    s32 other_count;
    if (!reader.Read(other_count) || other_count < 0 ||
        static_cast<u64>(other_count) > MAX_ENTRIES) {
        index.error = "the index ends before its count of other entries";
        return index;
    }
    std::vector<Entry> others(static_cast<size_t>(other_count));
    for (auto& entry : others) {
        bool deleted;
        if (!ReadRecord(reader, footer.version, entry, deleted)) {
            index.error = "an entry outside the packed list could not be read";
            return index;
        }
    }
    const auto by_location{[&](s32 location, Entry& entry) {
        if (location < 0) {
            const u64 other = static_cast<u64>(-(static_cast<s64>(location) + 1));
            if (other >= others.size()) {
                return false;
            }
            entry = others[other];
            return true;
        }
        size_t pos = static_cast<size_t>(location);
        return DecodeEntry(encoded, pos, footer.version, entry);
    }};
    std::vector<u8> directory;
    if (has_directory != 0 && directory_offset >= 0 && directory_size > 0 &&
        static_cast<u64>(directory_size) <= MAX_INDEX_SIZE) {
        directory = read_part(static_cast<u64>(directory_offset), static_cast<u64>(directory_size));
    }
    if (!directory.empty()) {
        Reader names{directory};
        s32 directories;
        if (!names.Read(directories) || directories < 0) {
            index.error = "the directory index starts with no count";
            return index;
        }
        for (s32 d = 0; d < directories; ++d) {
            std::string directory_name;
            s32 files;
            if (!names.ReadString(directory_name) || !names.Read(files) || files < 0) {
                index.error = fmt::format("directory {} of {} could not be read", d, directories);
                return index;
            }
            for (s32 f = 0; f < files; ++f) {
                std::string file_name;
                s32 location;
                if (!names.ReadString(file_name) || !names.Read(location)) {
                    index.error = fmt::format("a file name in directory {} could not be read",
                                              directory_name);
                    return index;
                }
                Entry entry;
                if (!by_location(location, entry)) {
                    index.error = fmt::format("{}{} points at no entry ({})", directory_name,
                                              file_name, location);
                    continue;
                }
                entry.path = JoinPath(index.mount_point, directory_name, file_name);
                index.entries.push_back(std::move(entry));
            }
        }
        return index;
    }
    // No names: the entries in list order.
    index.named = false;
    size_t pos = 0;
    while (pos < encoded.size() && index.entries.size() < MAX_ENTRIES) {
        Entry entry;
        if (!DecodeEntry(encoded, pos, footer.version, entry)) {
            index.error = fmt::format("packed entry at {} could not be read", pos);
            break;
        }
        index.entries.push_back(std::move(entry));
    }
    for (auto& entry : others) {
        index.entries.push_back(std::move(entry));
    }
    return index;
}

bool ReadEntry(const ReadFn& read, const Footer& footer, const Entry& entry,
               const DecryptFn& decrypt, bool unpack_oodle, std::vector<u8>& out,
               std::span<size_t> methods, std::string& error) {
    const auto count{[&](Method method) {
        if (static_cast<size_t>(method) < methods.size()) {
            ++methods[static_cast<size_t>(method)];
        }
    }};
    out.clear();
    if (entry.encrypted && !decrypt) {
        error = "encrypted, and no key";
        return false;
    }
    if (entry.method == 0) {
        const u64 start = entry.offset + entry.RecordSize(footer.version);
        const u64 length = entry.encrypted ? AlignUp(entry.size, AES_BLOCK) : entry.size;
        out = read(start, length);
        if (out.size() < length) {
            error = fmt::format("runs past the end of the file ({} of {} bytes)", out.size(),
                                length);
            return false;
        }
        if (entry.encrypted) {
            decrypt(out);
        }
        out.resize(entry.size);
        count(Method::Stored);
        return true;
    }
    if (entry.blocks.empty()) {
        error = "packed, with no blocks";
        return false;
    }
    const std::string_view method_name = entry.method <= footer.methods.size()
                                             ? std::string_view{footer.methods[entry.method - 1]}
                                             : std::string_view{};
    const u64 first = entry.blocks.front().start;
    const u64 last = entry.blocks.back().start +
                     AlignUp(entry.blocks.back().end - entry.blocks.back().start,
                             entry.encrypted ? AES_BLOCK : 1);
    const bool in_order = std::ranges::is_sorted(entry.blocks, {}, &Block::start);
    const bool single_read = in_order && last > first && last - first <= MAX_SINGLE_READ;
    std::vector<u8> packed_all;
    if (single_read) {
        packed_all = read(first, last - first);
        if (packed_all.size() < last - first) {
            error = fmt::format("runs past the end of the file ({} of {} bytes)",
                                packed_all.size(), last - first);
            return false;
        }
        if (entry.encrypted) {
            // Each block starts on a whole AES block, so decrypting the run decrypts each one.
            decrypt(packed_all);
        }
    }
    out.resize(entry.uncompressed_size + Preforged::Unreal::UNPACK_SLACK);
    u64 done = 0;
    Method last_method = Method::Unknown;
    std::vector<u8> packed_one;
    for (size_t i = 0; i < entry.blocks.size(); ++i) {
        const Block& block = entry.blocks[i];
        const u64 packed = block.end - block.start;
        const u64 left = entry.uncompressed_size - done;
        const u64 want = entry.block_size == 0 ? left : std::min<u64>(entry.block_size, left);
        if (packed == 0 || packed > MAX_BLOCK_PACKED || want == 0) {
            error = fmt::format("block {} of {} has {} packed bytes for {}", i,
                                entry.blocks.size(), packed, want);
            return false;
        }
        std::span<const u8> in;
        if (single_read) {
            in = std::span<const u8>{packed_all}.subspan(block.start - first, packed);
        } else {
            const u64 length = entry.encrypted ? AlignUp(packed, AES_BLOCK) : packed;
            packed_one = read(block.start, length);
            if (packed_one.size() < length) {
                error = fmt::format("block {} runs past the end of the file", i);
                return false;
            }
            if (entry.encrypted) {
                decrypt(packed_one);
            }
            in = std::span<const u8>{packed_one}.first(packed);
        }
        const Method method = Preforged::Unreal::UnpackBlock(
            in, std::span<u8>{out.data() + done, want}, method_name, last_method, unpack_oodle);
        count(method);
        if (method == Method::Unknown || method == Method::OodleFailed) {
            error = fmt::format("block {} of {} ({} -> {} bytes, named {}, starts {}) did not "
                                "unpack",
                                i, entry.blocks.size(), packed, want,
                                method_name.empty() ? std::string_view{"nothing"} : method_name,
                                Hex(in, 8));
            out.clear();
            return false;
        }
        done += want;
    }
    out.resize(done);
    if (done != entry.uncompressed_size) {
        error = fmt::format("the blocks unpacked to {} of {} bytes", done, entry.uncompressed_size);
        return false;
    }
    return true;
}

std::optional<ShaderArchive> ReadShaderArchive(std::span<const u8> data, std::string& error) {
    Reader reader{data};
    ShaderArchive archive;
    if (!reader.Read(archive.version)) {
        error = "too short";
        return std::nullopt;
    }
    const auto read_count{[&](u32& count, u64 element_size) {
        s32 value;
        if (!reader.Read(value) || value < 0 || static_cast<u32>(value) > MAX_ARCHIVE_COUNT ||
            reader.Left() / std::max<u64>(element_size, 1) < static_cast<u64>(value)) {
            return false;
        }
        count = static_cast<u32>(value);
        return true;
    }};
    const auto fail{[&](std::string_view what) {
        error = fmt::format("version {}: {} at byte {} (starts {})", archive.version, what,
                            reader.Position(), Hex(data, 48));
        return std::nullopt;
    }};
    u32 map_hashes;
    u32 shader_hashes;
    u32 map_count;
    u32 shader_count;
    if (archive.version == 2) {
        // FSerializedShaderArchive (engine 4.25 on): the hashes of the shader maps and of the
        // shaders, the shader maps, the shaders, the preload ranges and the shader indices.
        if (!read_count(map_hashes, 20) || !reader.Skip(u64{map_hashes} * 20) ||
            !read_count(shader_hashes, 20) || !reader.Skip(u64{shader_hashes} * 20)) {
            return fail("the hash lists don't fit");
        }
        if (!read_count(map_count, 16)) {
            return fail("the shader map list doesn't fit");
        }
        for (u32 i = 0; i < map_count; ++i) {
            u32 indices_offset;
            u32 shaders;
            reader.Read(indices_offset);
            reader.Read(shaders);
            reader.Skip(8);
            archive.maps.emplace_back(indices_offset, shaders);
        }
        if (!read_count(shader_count, 17)) {
            return fail("the shader list doesn't fit");
        }
        for (u32 i = 0; i < shader_count; ++i) {
            ArchiveShader shader;
            reader.Read(shader.offset);
            reader.Read(shader.size);
            reader.Read(shader.uncompressed_size);
            reader.Read(shader.frequency);
            archive.shaders.push_back(shader);
        }
        u32 preloads;
        u32 indices;
        if (!read_count(preloads, 16) || !reader.Skip(u64{preloads} * 16) ||
            !read_count(indices, 4)) {
            return fail("the preload or index list doesn't fit");
        }
        archive.shader_indices.resize(indices);
        for (u32& index : archive.shader_indices) {
            reader.Read(index);
        }
        if (shader_hashes != shader_count || map_hashes != map_count) {
            return fail(fmt::format("{} shader hashes for {} shaders, {} map hashes for {} maps",
                                    shader_hashes, shader_count, map_hashes, map_count));
        }
        archive.code_start = reader.Position();
    } else if (archive.version == 1) {
        // FShaderCodeArchive before engine 4.25: a map from shader hash to {offset, size,
        // unpacked size, stage}.
        if (!read_count(shader_count, 33)) {
            return fail("the shader map doesn't fit");
        }
        for (u32 i = 0; i < shader_count; ++i) {
            ArchiveShader shader;
            u32 offset;
            reader.Skip(20);
            reader.Read(offset);
            reader.Read(shader.size);
            reader.Read(shader.uncompressed_size);
            reader.Read(shader.frequency);
            shader.offset = offset;
            archive.shaders.push_back(shader);
        }
        // Anything the engine wrote after the map (such as a pipeline list) comes before the
        // code, which runs to the end of the file.
        u64 code_size = 0;
        for (const auto& shader : archive.shaders) {
            code_size = std::max<u64>(code_size, shader.offset + shader.size);
        }
        if (code_size > data.size() || data.size() - code_size < reader.Position()) {
            return fail(fmt::format("{} bytes of code don't fit after the table", code_size));
        }
        archive.code_start = data.size() - code_size;
    } else {
        return fail("version not known");
    }
    const u64 code_bytes = data.size() - archive.code_start;
    for (size_t i = 0; i < archive.shaders.size(); ++i) {
        const ArchiveShader& shader = archive.shaders[i];
        if (shader.size == 0 || shader.size > MAX_SHADER_SIZE ||
            shader.uncompressed_size > MAX_SHADER_SIZE || shader.size > shader.uncompressed_size ||
            shader.frequency >= 16 || shader.offset > code_bytes ||
            shader.size > code_bytes - shader.offset) {
            return fail(fmt::format("shader {} of {} is out of place: offset {}, {} -> {} bytes, "
                                    "stage {}, with {} bytes of code",
                                    i, archive.shaders.size(), shader.offset, shader.size,
                                    shader.uncompressed_size, shader.frequency, code_bytes));
        }
    }
    return archive;
}

bool IsShaderArchiveName(std::string_view lower_path) {
    return lower_path.ends_with(".ushaderbytecode") || lower_path.ends_with(".ushadercode") ||
           lower_path.find("globalshadercache") != std::string_view::npos ||
           lower_path.find("shaderarchive") != std::string_view::npos;
}

bool LooksLikeIndexStart(std::span<const u8> block) {
    if (block.size() < 16) {
        return false;
    }
    const s32 length = static_cast<s32>(Load32(block.data()));
    if (length > 0 && length <= 1024) {
        const size_t chars = std::min<size_t>(static_cast<size_t>(length) - 1, 12);
        if (chars == 0) {
            return false;
        }
        for (size_t i = 0; i < chars; ++i) {
            if (!IsPrintable(block[4 + i])) {
                return false;
            }
        }
        // A short mount point ends with its zero inside the block.
        return static_cast<size_t>(length) - 1 > 11 || block[4 + chars] == 0;
    }
    if (length < 0 && length >= -512) {
        const size_t chars = std::min<size_t>(static_cast<size_t>(-length) - 1, 6);
        if (chars == 0) {
            return false;
        }
        for (size_t i = 0; i < chars; ++i) {
            if (!IsPrintable(block[4 + i * 2]) || block[5 + i * 2] != 0) {
                return false;
            }
        }
        return true;
    }
    return false;
}

std::optional<Key> FindKey(std::span<const u8> image, std::span<const u8> first_block,
                           const std::function<bool(const Key&)>& verify, size_t& tried,
                           size_t& where) {
    tried = 0;
    if (first_block.size() < AES_BLOCK || image.size() < 32) {
        return std::nullopt;
    }
    std::array<u8, AES_BLOCK> plain;
    const auto attempt{[&](size_t pos) -> std::optional<Key> {
        Key key;
        std::memcpy(key.data(), image.data() + pos, key.size());
        ++tried;
        Aes256{key}.DecryptBlock(first_block.data(), plain.data());
        if (!LooksLikeIndexStart(plain) || !verify(key)) {
            return std::nullopt;
        }
        where = pos;
        return key;
    }};
    for (size_t pos = 0; pos + 32 <= image.size(); pos += 4) {
        if (LooksRandom(image.data() + pos)) {
            if (auto key = attempt(pos)) {
                return key;
            }
        }
    }
    // Strings of exactly 32 printable characters, at any byte.
    size_t run = 0;
    for (size_t pos = 0; pos < image.size(); ++pos) {
        if (IsPrintable(image[pos])) {
            ++run;
            continue;
        }
        if (run == 32 && image[pos] == 0) {
            if (auto key = attempt(pos - 32)) {
                return key;
            }
        }
        run = 0;
    }
    return std::nullopt;
}

} // namespace Preforged::UnrealPak
