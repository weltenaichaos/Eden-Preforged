// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>

#include <fmt/format.h>

#include "jni/trpfs.h"

namespace Preforged::Trpfs {
namespace {

constexpr u64 MAX_TABLE = u64{64} << 20;
constexpr u32 MAX_COUNT = 1u << 24;

template <typename T>
bool Load(std::span<const u8> buffer, size_t pos, T& out) {
    if (pos > buffer.size() || buffer.size() - pos < sizeof(T)) {
        return false;
    }
    std::memcpy(&out, buffer.data() + pos, sizeof(T));
    return true;
}

/// A table of a flatbuffer, with bounds checks on every read.
class Table {
public:
    static std::optional<Table> Root(std::span<const u8> buffer) {
        u32 root;
        if (!Load(buffer, 0, root)) {
            return std::nullopt;
        }
        return At(buffer, root);
    }

    static std::optional<Table> At(std::span<const u8> buffer, size_t pos) {
        s32 to_vtable;
        if (!Load(buffer, pos, to_vtable)) {
            return std::nullopt;
        }
        const s64 vtable = static_cast<s64>(pos) - to_vtable;
        u16 vtable_size;
        if (vtable < 0 || !Load(buffer, static_cast<size_t>(vtable), vtable_size) ||
            vtable_size < 4 || static_cast<size_t>(vtable) + vtable_size > buffer.size()) {
            return std::nullopt;
        }
        return Table{buffer, pos, static_cast<size_t>(vtable), vtable_size};
    }

    /// Where field `index` is, 0 when the table leaves it out.
    [[nodiscard]] size_t Field(size_t index) const {
        const size_t entry = 4 + 2 * index;
        u16 offset = 0;
        if (entry + 2 > vtable_size || !Load(buffer, vtable + entry, offset) || offset == 0) {
            return 0;
        }
        return pos + offset;
    }

    template <typename T>
    [[nodiscard]] T Scalar(size_t index, T fallback) const {
        const size_t at = Field(index);
        T value = fallback;
        if (at != 0 && !Load(buffer, at, value)) {
            return fallback;
        }
        return value;
    }

    /// The bytes of vector field `index` of elements of `element_size` bytes, and its count.
    [[nodiscard]] std::optional<std::span<const u8>> Vector(size_t index, size_t element_size,
                                                            u32& count) const {
        const size_t at = Field(index);
        u32 to_vector;
        if (at == 0 || !Load(buffer, at, to_vector)) {
            return std::nullopt;
        }
        const size_t vector = at + to_vector;
        if (!Load(buffer, vector, count) || count > MAX_COUNT ||
            u64{count} * element_size > buffer.size() - std::min(buffer.size(), vector + 4)) {
            return std::nullopt;
        }
        return buffer.subspan(vector + 4, size_t{count} * element_size);
    }

    /// Table `element` of vector field `index` of tables.
    [[nodiscard]] std::optional<Table> VectorTable(size_t index, u32 element) const {
        u32 count;
        const auto elements = Vector(index, 4, count);
        if (!elements || element >= count) {
            return std::nullopt;
        }
        const size_t slot = static_cast<size_t>(elements->data() - buffer.data()) + element * 4;
        u32 to_table;
        if (!Load(buffer, slot, to_table)) {
            return std::nullopt;
        }
        return At(buffer, slot + to_table);
    }

private:
    Table(std::span<const u8> buffer_, size_t pos_, size_t vtable_, u16 vtable_size_)
        : buffer{buffer_}, pos{pos_}, vtable{vtable_}, vtable_size{vtable_size_} {}

    std::span<const u8> buffer;
    size_t pos;
    size_t vtable;
    u16 vtable_size;
};

u64 Load64(std::span<const u8> data, size_t index) {
    u64 value = 0;
    std::memcpy(&value, data.data() + index * 8, 8);
    return value;
}

} // Anonymous namespace

bool IsTrpfs(std::span<const u8> start) {
    return start.size() >= 16 && std::memcmp(start.data(), "ONEPACK\0", 8) == 0;
}

std::optional<FileSystem> ReadFileSystem(const ReadFn& read, u64 file_size, std::string& error) {
    const std::vector<u8> head = read(0, 16);
    if (!IsTrpfs(head)) {
        error = "not a trpfs";
        return std::nullopt;
    }
    FileSystem fs;
    std::memcpy(&fs.table_offset, head.data() + 8, 8);
    if (fs.table_offset < 16 || fs.table_offset >= file_size ||
        file_size - fs.table_offset > MAX_TABLE) {
        error = fmt::format("the table at {:#x} isn't in the file or is too large",
                            fs.table_offset);
        return std::nullopt;
    }
    const std::vector<u8> table_bytes = read(fs.table_offset, file_size - fs.table_offset);
    const auto table = Table::Root(table_bytes);
    u32 hash_count = 0;
    u32 offset_count = 0;
    const auto hashes = table ? table->Vector(0, 8, hash_count) : std::nullopt;
    const auto offsets = table ? table->Vector(1, 8, offset_count) : std::nullopt;
    if (!hashes || !offsets || hash_count != offset_count || offset_count == 0) {
        error = fmt::format("its table isn't a flatbuffer of pack hashes and offsets ({} and {})",
                            hash_count, offset_count);
        return std::nullopt;
    }
    for (u32 i = 0; i < offset_count; ++i) {
        fs.packs.push_back({.hash = Load64(*hashes, i), .offset = Load64(*offsets, i)});
    }
    std::vector<Pack> sorted = fs.packs;
    std::ranges::sort(sorted, {}, &Pack::offset);
    for (size_t i = 0; i < sorted.size(); ++i) {
        const u64 end = i + 1 < sorted.size() ? sorted[i + 1].offset : fs.table_offset;
        sorted[i].size = end > sorted[i].offset ? end - sorted[i].offset : 0;
    }
    std::erase_if(sorted, [&](const Pack& pack) {
        return pack.size == 0 || pack.offset < 16 || pack.offset >= fs.table_offset;
    });
    fs.packs = std::move(sorted);
    if (fs.packs.empty()) {
        error = "none of its packs lies in the file";
        return std::nullopt;
    }
    return fs;
}

bool ReadPack(std::span<const u8> pack, std::vector<PackedFile>& files, std::string& error) {
    files.clear();
    const auto table = Table::Root(pack);
    u32 hash_count = 0;
    u32 file_count = 0;
    const auto hashes = table ? table->Vector(0, 8, hash_count) : std::nullopt;
    if (!table || !table->Vector(1, 4, file_count)) {
        error = "not a flatbuffer of files";
        return false;
    }
    for (u32 i = 0; i < file_count; ++i) {
        const auto file = table->VectorTable(1, i);
        if (!file) {
            error = fmt::format("file {} of {} isn't a table", i, file_count);
            return false;
        }
        u32 size = 0;
        const auto data = file->Vector(4, 1, size);
        if (!data) {
            continue;
        }
        PackedFile out;
        out.hash = hashes && i < hash_count ? Load64(*hashes, i) : 0;
        out.compression = file->Scalar<s8>(1, -1);
        out.size = file->Scalar<u64>(3, data->size());
        out.data = *data;
        files.push_back(out);
    }
    return true;
}

} // namespace Preforged::Trpfs
