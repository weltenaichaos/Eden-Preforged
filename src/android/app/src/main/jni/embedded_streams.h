// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/common_types.h"

// Finds packed data inside files, wherever it starts: zstd frames, zlib streams, ZIP entries
// packed with deflate and CRI's CRILAYLA (the files in CPK archives). Many engines keep their files in large archives of their own
// whose tables the search doesn't read (Smash's data.arc, Xenoblade's xbc1 and .ard, Capcom's
// MT Framework .arc, RE Engine's .pak, CryEngine's ZIP paks), but pack each file in one of these
// common ways, which tell themselves by their first bytes and check themselves when unpacked.
namespace Preforged::Embedded {

enum class Kind : u8 {
    Zstd,
    Zlib,
    Zip,
    Crilayla,
};
constexpr size_t KIND_COUNT = 4;

[[nodiscard]] std::string_view KindName(Kind kind);

/// Reads `size` bytes at `offset` of a file. Returns fewer at the end of the file.
using ReadFn = std::function<std::vector<u8>(u64 offset, u64 size)>;

/// The bytes of a file: a piece of it in memory, and the file around it to read further.
class Source {
public:
    /// `data` is the bytes at `base` of a file of `file_size` bytes. `fetch` reads the rest of
    /// the file; it may be null when `data` is all of it.
    Source(std::span<const u8> data, u64 base, u64 file_size, const ReadFn* fetch);

    /// The bytes at [offset, offset + size) of the file, fewer at its end. They stay valid
    /// until the next call.
    [[nodiscard]] std::span<const u8> Get(u64 offset, u64 size);

    [[nodiscard]] u64 FileSize() const {
        return file_size;
    }

    [[nodiscard]] u64 Base() const {
        return base;
    }

    /// The part of [from, to) that is in memory.
    [[nodiscard]] std::span<const u8> Memory(u64 from, u64 to) const;

private:
    std::span<const u8> data;
    u64 base;
    u64 file_size;
    const ReadFn* fetch;
    std::vector<u8> window;
    u64 window_offset = 0;
};

struct Stats {
    std::array<size_t, KIND_COUNT> candidates{}; ///< Places whose first bytes looked like one
    std::array<size_t, KIND_COUNT> streams{};    ///< Unpacked
    std::array<size_t, KIND_COUNT> too_large{};  ///< Would unpack to more than allowed
    std::array<u64, KIND_COUNT> packed{};
    std::array<u64, KIND_COUNT> unpacked{};
};

/// Unpacks CRILAYLA data (header, packed bytes read from their end backwards, then the first
/// 0x100 bytes stored) into `out`. Returns false when it doesn't unpack.
[[nodiscard]] bool UnpackCrilayla(std::span<const u8> in, std::vector<u8>& out);

/// The size of the zstd frame that starts at `offset`: its header, blocks and checksum. 0 when
/// the bytes there aren't a valid frame or it runs past the end of the file.
[[nodiscard]] u64 ZstdFrameSize(Source& source, u64 offset);

struct Handlers {
    /// Unpacks the zstd frame `frame` at `offset` (the caller has the dictionaries) and
    /// searches what it holds. Returns the bytes it unpacked to, 0 when it did not unpack.
    std::function<u64(u64 offset, std::span<const u8> frame)> zstd;
    /// Searches what a zlib stream, ZIP entry or CRILAYLA file at `offset` unpacked to.
    std::function<void(Kind kind, u64 offset, std::span<const u8> unpacked)> unpacked;
    /// Whether to stop.
    std::function<bool()> stop;
};

/// Finds the packed streams that start in [from, to) of the file, as far as that is in memory,
/// and hands each to `handlers`.
/// Streams that unpack to fewer than `min_size` bytes are left out, and those that would unpack
/// to more than `max_size` bytes are counted only. Returns where the last stream found ends,
/// or `from` when none was found, so a search of the next piece of the file can start there.
u64 FindStreams(Source& source, u64 from, u64 to, size_t min_size, size_t max_size,
                const Handlers& handlers, Stats& stats);

} // namespace Preforged::Embedded
