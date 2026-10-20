// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>

#include <fmt/format.h>

#include "jni/gf_pak.h"

namespace Preforged::GfPak {
namespace {

constexpr u32 MAX_FILES = 1u << 22;
constexpr size_t HEADER_SIZE = 0x30;
constexpr size_t ENTRY_SIZE = 24;

template <typename T>
T Load(const u8* data) {
    T value;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

} // Anonymous namespace

bool IsPak(std::span<const u8> start) {
    return start.size() >= HEADER_SIZE && std::memcmp(start.data(), "GFLXPACK", 8) == 0;
}

std::optional<Pak> ReadPak(const ReadFn& read, u64 file_size, std::string& error) {
    const std::vector<u8> head = read(0, HEADER_SIZE);
    if (!IsPak(head)) {
        error = "not a GFLXPACK archive";
        return std::nullopt;
    }
    Pak pak;
    pak.version = Load<u32>(head.data() + 8);
    pak.declared = Load<u32>(head.data() + 16);
    const u64 info_offset = Load<u64>(head.data() + 24);
    const u64 hashes_offset = Load<u64>(head.data() + 32);
    if (pak.declared == 0 || pak.declared > MAX_FILES || info_offset > file_size ||
        u64{pak.declared} * ENTRY_SIZE > file_size - info_offset) {
        error = fmt::format("the header gives {} files at {:#x}", pak.declared, info_offset);
        return std::nullopt;
    }
    const std::vector<u8> info = read(info_offset, u64{pak.declared} * ENTRY_SIZE);
    std::vector<u8> hashes;
    if (hashes_offset <= file_size && u64{pak.declared} * 8 <= file_size - hashes_offset) {
        hashes = read(hashes_offset, u64{pak.declared} * 8);
    }
    if (info.size() != u64{pak.declared} * ENTRY_SIZE) {
        error = "the file table runs past the end of the file";
        return std::nullopt;
    }
    for (u32 i = 0; i < pak.declared; ++i) {
        const u8* const record = info.data() + i * ENTRY_SIZE;
        Entry entry;
        entry.compression = Load<u16>(record + 2);
        entry.size = Load<u32>(record + 4);
        entry.packed_size = Load<u32>(record + 8);
        entry.offset = Load<u64>(record + 16);
        if (hashes.size() == u64{pak.declared} * 8) {
            entry.hash = Load<u64>(hashes.data() + i * 8);
        }
        if (entry.compression > 3 || entry.offset > file_size ||
            entry.packed_size > file_size - entry.offset ||
            (entry.compression == 0 && entry.packed_size != entry.size)) {
            ++pak.invalid;
            continue;
        }
        pak.entries.push_back(entry);
    }
    if (pak.entries.empty()) {
        error = fmt::format("none of its {} files points into the file", pak.declared);
        return std::nullopt;
    }
    return pak;
}

const char* MethodName(u32 compression) {
    switch (compression) {
    case 1:
        return "zlib";
    case 2:
        return "lz4";
    case 3:
        return "oodle";
    }
    return "stored";
}

} // namespace Preforged::GfPak
