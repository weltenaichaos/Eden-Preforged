// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>

#include <fmt/format.h>

#include "jni/re_pak.h"

namespace Preforged::RePak {
namespace {

constexpr u32 MAX_FILES = 1u << 22;
constexpr u32 FEATURE_ENCRYPTED_TABLE = 8;
/// Table entries checked to tell where the table starts.
constexpr size_t ENTRIES_CHECKED = 64;

template <typename T>
T Load(const u8* data) {
    T value;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

Entry ReadEntry(const u8* data, u32 major) {
    Entry entry;
    if (major == 2) {
        entry.offset = Load<u64>(data);
        entry.size = Load<u64>(data + 8);
        entry.packed_size = entry.size;
        entry.hash = (u64{Load<u32>(data + 20)} << 32) | Load<u32>(data + 16);
        return entry;
    }
    entry.hash = (u64{Load<u32>(data + 4)} << 32) | Load<u32>(data);
    entry.offset = Load<u64>(data + 8);
    entry.packed_size = Load<u64>(data + 16);
    entry.size = Load<u64>(data + 24);
    const u64 attributes = Load<u64>(data + 32);
    entry.compression = static_cast<u32>(attributes & 0xF);
    entry.encryption = static_cast<u32>((attributes >> 16) & 0xFF);
    return entry;
}

bool Valid(const Entry& entry, u64 table_end, u64 file_size) {
    return entry.offset >= table_end && entry.offset <= file_size &&
           entry.packed_size <= file_size - entry.offset && entry.compression <= 2 &&
           (entry.compression != 0 || entry.packed_size == entry.size);
}

} // Anonymous namespace

bool IsPak(std::span<const u8> start) {
    return start.size() >= 16 && std::memcmp(start.data(), "KPKA", 4) == 0;
}

std::optional<Pak> ReadPak(const ReadFn& read, u64 file_size, std::string& error) {
    const std::vector<u8> head = read(0, 32);
    if (!IsPak(head)) {
        error = "not a KPKA pak";
        return std::nullopt;
    }
    Pak pak;
    pak.major = head[4];
    pak.minor = head[5];
    pak.features = Load<u16>(head.data() + 6);
    pak.declared = Load<u32>(head.data() + 8);
    if ((pak.features & FEATURE_ENCRYPTED_TABLE) != 0) {
        pak.table_encrypted = true;
        return pak;
    }
    if (pak.major != 2 && pak.major != 4) {
        error = fmt::format("version {}.{} isn't known", pak.major, pak.minor);
        return std::nullopt;
    }
    const u64 entry_size = pak.major == 2 ? 24 : 48;
    if (pak.declared == 0 || pak.declared > MAX_FILES ||
        u64{pak.declared} * entry_size > file_size) {
        error = fmt::format("the header gives {} files", pak.declared);
        return std::nullopt;
    }
    // The table follows the 16-byte header; newer versions put a few more bytes between.
    const u64 table_size = u64{pak.declared} * entry_size;
    const std::vector<u8> table = read(16, table_size + 16);
    for (const u64 extra : {0, 4, 8, 16}) {
        if (table.size() < extra + table_size) {
            continue;
        }
        const u64 table_end = 16 + extra + table_size;
        bool valid = true;
        for (size_t i = 0; i < std::min<size_t>(pak.declared, ENTRIES_CHECKED) && valid; ++i) {
            valid = Valid(ReadEntry(table.data() + extra + i * entry_size, pak.major), table_end,
                          file_size);
        }
        if (!valid) {
            continue;
        }
        pak.table_offset = 16 + extra;
        for (u32 i = 0; i < pak.declared; ++i) {
            const Entry entry = ReadEntry(table.data() + extra + i * entry_size, pak.major);
            if (!Valid(entry, table_end, file_size)) {
                ++pak.invalid;
                continue;
            }
            pak.entries.push_back(entry);
        }
        return pak;
    }
    error = fmt::format("no table of {} files after the header points into the file",
                        pak.declared);
    return std::nullopt;
}

const char* CompressionName(u32 compression) {
    switch (compression) {
    case 0:
        return "stored";
    case 1:
        return "deflate";
    case 2:
        return "zstd";
    }
    return "?";
}

} // namespace Preforged::RePak
