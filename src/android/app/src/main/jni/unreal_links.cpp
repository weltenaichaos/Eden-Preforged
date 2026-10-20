// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>

#include "common/logging.h"
#include "jni/unreal_links.h"
#include "jni/unreal_package.h"

namespace Preforged::Unreal {
namespace {

constexpr u32 NVN_CODE_MAGIC = 0x12345678;
constexpr u64 SELF_BRANCH_A = 0xE2400FFFFF87000FULL;
constexpr u64 SELF_BRANCH_B = 0xE2400FFFFF07000FULL;
/// Hits kept per shader and kind of reference, to measure tables and pairs.
constexpr size_t MAX_OFFSET_HITS = 64;
constexpr size_t MAX_INDEX_HITS = 64;
/// How far apart a vertex and a fragment shader's references may be to count as one record.
constexpr s64 PAIR_WINDOW = 0x400;
constexpr s64 INDEX_PAIR_WINDOW = 0x40;
/// Largest step between table entries looked for.
constexpr size_t MAX_STRIDE = 0x200;
/// How far around a table entry 8-byte ids are looked for.
constexpr s32 ID_RANGE = 0x80;
constexpr size_t MAX_ID_WINDOWS = 16;
/// Small enough for the filter to stay in the CPU cache: it is read at every byte searched.
constexpr u32 ID_FILTER_BITS = 22;
constexpr u32 PAIR_FILTER_BITS = 22;
constexpr size_t MAX_PIECE_HITS = size_t{4} << 20;
/// How long the other packages are searched in all, so the shader search keeps its time.
constexpr auto SEARCH_TIME = std::chrono::seconds{60};
/// Index references below this are too common in other data to search other packages for.
constexpr u32 MIN_FOREIGN_INDEX = 64;
/// How long the bytes of other packages outside shader and material objects are searched for
/// the shaders' ids alone, and how many of the ids found there are shown.
constexpr auto ID_ONLY_SEARCH_TIME = std::chrono::seconds{20};
constexpr size_t ID_EXAMPLES = 12;
constexpr size_t ID_EXAMPLES_PER_CLASS = 3;
/// Shader cache objects read per package; bytes of each shown; references to the code kept;
/// how far before a reference its tag is looked for, and its ids before and after it.
constexpr size_t MAX_CACHE_OBJECTS = 2;
constexpr u64 CACHE_DUMP = 0x400;
constexpr size_t MAX_CACHE_REFS = size_t{1} << 21;
constexpr s32 CACHE_TAG_RANGE = 0x20;
constexpr s32 CACHE_LOOK_BACK = 0x80;
constexpr s32 CACHE_LOOK_AHEAD = 0x40;
constexpr size_t CACHE_ID_WINDOWS = 6;
constexpr size_t CACHE_SAMPLE = 2048;
constexpr size_t CACHE_FRONTS = 4096;
constexpr size_t CACHE_TAG_CANDIDATES = 3;
constexpr int CACHE_ID_MIN_BITS = 16;
/// The most ids between two shaders of a pair that are checked for being in one shader map.
constexpr u32 CACHE_MAP_HITS = 256;
/// How close the ids of a pair's two shaders are counted in a shader cache.
constexpr std::array<u64, 7> CACHE_LIMITS{0x40, 0x100, 0x400, 0x1000, 0x4000, 0x10000, 0x40000};
constexpr size_t NAMES_LOGGED = 4000;
constexpr size_t EXAMPLES = 3;
/// Bytes shown before and after the two references of a pair example.
constexpr u64 EXAMPLE_MARGIN = 0x80;
/// Examples kept at each distance; the most common distance shows this many.
constexpr size_t EXAMPLES_PER_DISTANCE = 3;
/// Bytes kept from before each code header, to look for the ids next to references in.
constexpr u64 CODE_WINDOW = 0x100;
/// Longest run of records walked around an example, and how many of its numbers are shown.
constexpr size_t RUN_MAX = 4096;
constexpr size_t RUN_SHOWN = 32;
/// The bytes around each code header whose 4-byte fields are compared between the two shaders
/// of a pair, and searched for 8-byte ids.
constexpr u64 FIELD_BEFORE = 0x200;
constexpr u64 FIELD_AFTER = 0x40;
/// Code headers looked at per shader that the package holds more than once.
constexpr size_t MAX_COPIES = 4;
/// How many headers on the moved control pairs are.
constexpr u32 CONTROL_SHIFT = 5;
/// Most pairs a learned link hands over for pre-compiling (the Thor compiles about 70 a
/// second), and most it may tie in all.
constexpr size_t MAX_LEARNED_PAIRS = 10000;
/// The shader cache names the pairs the game's materials draw with, so more of them are kept:
/// XCOM 2's gives about 17,000, which compile in about 90 s once on the Thor.
constexpr size_t MAX_CACHE_PAIRS = 30000;
constexpr size_t MAX_CANDIDATES = size_t{1} << 20;
constexpr size_t MAX_CODE_ID_WINDOWS = 8;
constexpr size_t MAX_HITS_PER_CODE = 16;
constexpr size_t MAX_ID_HITS = size_t{1} << 21;
/// Vertex and fragment shader pairs kept from the ids found close together in other packages,
/// to learn a link's pairs from: in all, and per kind of id and distance.
constexpr size_t MAX_ID_CANDIDATES = size_t{1} << 21;
constexpr size_t MAX_ID_CANDIDATES_AT = size_t{1} << 16;

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

std::string Hex(std::span<const u8> data, u64 begin, u64 length) {
    std::string out;
    const u64 end = std::min<u64>(data.size(), begin + length);
    for (u64 i = std::min<u64>(begin, data.size()); i < end; ++i) {
        out += fmt::format("{:02x}", data[i]);
    }
    return out;
}

u64 MixId(u64 value) {
    return (value * 0x9E3779B97F4A7C15ULL) >> (64 - ID_FILTER_BITS);
}

/// A second bit per id, for the filter of all shaders' ids: with two bits set per id, about 15
/// times fewer bytes of other data pass it, so the search for ids alone is about twice as fast.
u64 MixId2(u64 value) {
    return ((value ^ (value >> 31)) * 0xBF58476D1CE4E5B9ULL) >> (64 - ID_FILTER_BITS);
}

/// Whether 8 bytes look like an id: random bits, not a small number, a name or a float.
bool IdLike(u64 value) {
    return std::popcount(value) >= CACHE_ID_MIN_BITS;
}

/// 8 bytes as a name of the package (index and number) when they can be one, else as hex.
std::string KeyName(const Tables& tables, u64 key) {
    const u32 index = static_cast<u32>(key);
    const u32 number = static_cast<u32>(key >> 32);
    if (index < tables.names.size() && number < 0x10000) {
        return number == 0 ? tables.names[index]
                           : fmt::format("{}_{}", tables.names[index], number - 1);
    }
    return fmt::format("0x{:016x}", key);
}

std::string Lower(std::string_view text) {
    std::string out(text);
    for (char& c : out) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

/// Ways an object may refer to a piece of code. Kinds past these are 8-byte ids found next to
/// a table entry, one kind per place.
enum Kind : u32 {
    PackageOffset,   ///< Offset of the code header in the package's uncompressed layout
    OffsetFromFirst, ///< Offset from the first code header
    OffsetInObject,  ///< Offset from the start of the object holding the code
    Index,           ///< Position among the code headers, from 0
    IndexFromOne,    ///< Position from 1
    Slot,            ///< Offset from the first code header in 256-byte steps
    OffsetFromBase,  ///< Offset from the start of the part of the code holding it
    KIND_COUNT,
};

constexpr std::array<const char*, KIND_COUNT> KIND_NAMES{
    "package offset", "offset from the first code", "offset in its object", "index",
    "index from 1",   "256-byte slot",              "offset from the start of its part of the code",
};

bool IsIndexKind(u32 kind) {
    return kind == Index || kind == IndexFromOne || kind == Slot;
}

template <typename Key, typename Format>
std::string Top(const std::unordered_map<Key, size_t>& counts, size_t limit, Format&& format) {
    std::vector<std::pair<size_t, Key>> sorted;
    sorted.reserve(counts.size());
    for (const auto& [key, count] : counts) {
        sorted.emplace_back(count, key);
    }
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second;
    });
    std::string out;
    for (size_t i = 0; i < std::min(limit, sorted.size()); ++i) {
        out += out.empty() ? "" : ", ";
        out += format(sorted[i].second, sorted[i].first);
    }
    return out.empty() ? std::string{"none"} : out;
}

struct ClassStats {
    size_t count = 0;
    u64 bytes = 0;
    std::vector<std::string> examples;
};

/// A played pair found at some distance, as the log shows it.
struct PairExample {
    std::string text;
    /// The 16 bytes just before and just after each of the two references (vertex before,
    /// vertex after, fragment before, fragment after), which may be ids the code package also
    /// holds next to the code. Left out when all zero.
    std::array<std::array<u8, 16>, 4> around{};
    std::array<bool, 4> have{};
};

/// Played pairs whose vertex and fragment shader references were found close together.
struct PairFinds {
    std::unordered_set<u64> pairs;
    std::unordered_map<s64, size_t> distances;
    std::unordered_map<std::string, size_t> files;
    /// The first pairs found at each distance, for the first distances found.
    std::unordered_map<s64, std::vector<PairExample>> examples;
    /// Per group of pairs (played, shuffled, moved on): at each distance, how many different
    /// pairs of the group were found there.
    std::array<std::unordered_map<s64, u32>, 3> distinct_at;
    /// (group, pair, distance) counted already.
    std::unordered_set<u64> distinct_seen;
    /// Per distance, where in 4 bytes the vertex shader references of played pairs are.
    std::unordered_map<s64, std::array<u32, 4>> residues;
};

/// How many different played and control pairs one kind of reference lists at one distance.
struct LinkCount {
    std::pair<u32, u32> kinds;
    s64 distance;
    u32 played;
    u32 shuffled;
    u32 moved;
    s64 Score() const {
        return static_cast<s64>(played) - static_cast<s64>(std::max(shuffled, moved));
    }
};

/// The vertex and fragment shader pairs whose ids were found at one distance, by code header.
struct IdCandidates {
    std::vector<std::pair<u32, u32>> pairs;
    /// More were found than are kept.
    bool full = false;
};

/// A reference to a code header in a shader cache object.
struct CacheRef {
    u64 pos;
    u32 code;
};

/// Vertex and fragment shader pairs whose ids were found close together, by (first kind << 48) |
/// (second kind << 32) | distance, and the ids found by stage.
struct CandidateStore {
    std::unordered_map<u64, IdCandidates> buckets;
    size_t count = 0;
    /// More were found than are kept.
    bool capped = false;
    std::array<size_t, 3> hits{};
};

/// A shader or material object of the package being searched, as offsets in what Unpack
/// hands out.
struct ObjectSpan {
    u64 begin;
    u64 end;
    std::string what;
    std::string class_name;
};

constexpr size_t MAX_EXAMPLE_DISTANCES = 256;

struct Hit {
    u64 pos;
    u32 kind;
    u32 code;
};

/// How far apart the game's pairs are in the code package's order. Unreal may store a
/// material's shaders together, so the two shaders of a pair may be neighbours there.
/// `codes_of` lists the code headers of each shader hash in order.
void LogPairOrder(std::span<const std::pair<u64, u64>> played,
                  const std::unordered_map<u64, std::vector<u32>>& codes_of, u32 count) {
    const std::set<std::pair<u64, u64>> distinct(played.begin(), played.end());
    std::unordered_map<s64, size_t> deltas;
    size_t with_both = 0;
    std::array<size_t, 4> within{};
    for (const auto& [vertex, fragment] : distinct) {
        const auto v = codes_of.find(vertex);
        const auto f = codes_of.find(fragment);
        if (v == codes_of.end() || f == codes_of.end()) {
            continue;
        }
        ++with_both;
        const std::vector<u32>& fragments = f->second;
        s64 best = 0;
        bool have = false;
        for (const u32 vertex_code : v->second) {
            const auto it = std::lower_bound(fragments.begin(), fragments.end(), vertex_code);
            for (const auto near : {it, it == fragments.begin() ? fragments.end() : std::prev(it)}) {
                if (near == fragments.end()) {
                    continue;
                }
                const s64 delta = s64{vertex_code} - s64{*near};
                const s64 size = delta < 0 ? -delta : delta;
                const s64 best_size = best < 0 ? -best : best;
                if (!have || size < best_size || (size == best_size && delta > best)) {
                    best = delta;
                    have = true;
                }
            }
        }
        ++deltas[best];
        const s64 size = best < 0 ? -best : best;
        within[0] += size <= 1 ? 1 : 0;
        within[1] += size <= 2 ? 1 : 0;
        within[2] += size <= 4 ? 1 : 0;
        within[3] += size <= 16 ? 1 : 0;
    }
    LOG_INFO(Frontend,
             "[ShaderScan] Unreal links: the game's pairs in the code package's order: {} of {} "
             "pairs have both shaders there; vertex minus fragment shader position at their "
             "closest copies: {}; within 1: {}, 2: {}, 4: {}, 16: {}",
             with_both, distinct.size(),
             Top(deltas, 16, [](s64 d, size_t n) { return fmt::format("{:+d} x{}", d, n); }),
             within[0], within[1], within[2], within[3]);
    size_t repeated = 0;
    size_t repeated_headers = 0;
    size_t most = 0;
    for (const auto& [hash, list] : codes_of) {
        if (list.size() > 1) {
            ++repeated;
            repeated_headers += list.size();
        }
        most = std::max(most, list.size());
    }
    LOG_INFO(Frontend,
             "[ShaderScan] Unreal links: {} different shaders in {} code headers; {} of them are "
             "there more than once ({} headers), at most {} times",
             codes_of.size(), count, repeated, repeated_headers, most);
}

/// A played pair of shader hashes, by the code headers that hold each of the two shaders.
struct PairCopies {
    std::vector<u32> vertex;
    std::vector<u32> fragment;
};

/// The same shaders paired at random, and every pair moved CONTROL_SHIFT headers on, to tell a
/// link from chance. A field that only matches because a pair's shaders sit close together in
/// the package matches as often for the moved pairs.
struct Controls {
    std::vector<PairCopies> shuffled;
    std::vector<PairCopies> shifted;
};

Controls MakeControls(std::span<const PairCopies> played, u32 count) {
    Controls controls;
    controls.shuffled.assign(played.begin(), played.end());
    std::vector<size_t> order(played.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::mt19937 random{11};
    std::shuffle(order.begin(), order.end(), random);
    for (size_t i = 0; i < played.size(); ++i) {
        controls.shuffled[i].fragment = played[order[i]].fragment;
    }
    for (const PairCopies& pair : played) {
        PairCopies moved;
        for (const u32 code : pair.vertex) {
            if (code + CONTROL_SHIFT < count) {
                moved.vertex.push_back(code + CONTROL_SHIFT);
            }
        }
        for (const u32 code : pair.fragment) {
            if (code + CONTROL_SHIFT < count) {
                moved.fragment.push_back(code + CONTROL_SHIFT);
            }
        }
        if (!moved.vertex.empty() && !moved.fragment.empty()) {
            controls.shifted.push_back(std::move(moved));
        }
    }
    return controls;
}

/// Where the bytes before each code header stop being the code before it: just past the end
/// marker of the shader before. The fields and ids by a header are only read from there on.
std::vector<u64> PrefixStarts(std::span<const u8> image, const std::vector<u64>& codes) {
    std::vector<u64> starts(codes.size(), 0);
    for (size_t code = 1; code < codes.size(); ++code) {
        const u64 previous = codes[code - 1];
        starts[code] = previous;
        for (u64 pos = previous + 0x30; pos + 8 <= codes[code] && pos + 8 <= image.size();
             pos += 8) {
            const u64 value = Load64(image.data() + pos);
            if (value == SELF_BRANCH_A || value == SELF_BRANCH_B) {
                starts[code] = pos + 8;
                break;
            }
        }
    }
    if (!codes.empty()) {
        starts[0] = codes[0] - std::min(codes[0], FIELD_BEFORE);
    }
    return starts;
}

/// The bytes at the same place relative to each shader: by its code header, or in its record
/// in front of the code. Fields and ids are read from there.
struct Window {
    /// Where the bytes are, for the log.
    std::string what;
    /// The anchors are records apart from the code, which a field of another record may name.
    bool records = false;
    /// Per code header, what the places are counted from, 0 when it has none.
    std::vector<u64> anchors;
    /// Per code header, the bytes that belong to it: the bytes of other shaders are not read.
    std::vector<u64> lows;
    std::vector<u64> highs;
    s64 before = 0;
    s64 after = 0;
    /// Ids are looked for before this place only.
    s64 id_end = 0;
    /// Most fields apart the places of a field two shaders share may be.
    size_t max_shift = 0;

    size_t Slots() const {
        return static_cast<size_t>((after - before) / 4);
    }
    s64 Offset(size_t slot) const {
        return before + static_cast<s64>(slot) * 4;
    }
    std::optional<u64> At(u32 code, s64 offset, u64 size) const {
        const u64 anchor = anchors[code];
        if (anchor == 0) {
            return std::nullopt;
        }
        const s64 pos = static_cast<s64>(anchor) + offset;
        if (pos < static_cast<s64>(lows[code]) || static_cast<u64>(pos) + size > highs[code]) {
            return std::nullopt;
        }
        return static_cast<u64>(pos);
    }
    u32 Read32(std::span<const u8> image, u32 code, size_t slot) const {
        const auto pos = At(code, Offset(slot), 4);
        return pos ? Load32(image.data() + *pos) : 0;
    }
    std::optional<u64> Read64(std::span<const u8> image, u32 code, s64 offset) const {
        const auto pos = At(code, offset, 8);
        if (!pos) {
            return std::nullopt;
        }
        return Load64(image.data() + *pos);
    }
    std::vector<u32> Anchored() const {
        std::vector<u32> out;
        for (u32 code = 0; code < anchors.size(); ++code) {
            if (anchors[code] != 0) {
                out.push_back(code);
            }
        }
        return out;
    }
};

Window CodeHeaderWindow(std::span<const u8> image, const std::vector<u64>& codes) {
    Window window;
    window.what = "by the code headers";
    window.anchors = codes;
    window.lows = PrefixStarts(image, codes);
    window.highs.resize(codes.size());
    for (size_t code = 0; code < codes.size(); ++code) {
        window.highs[code] = std::min<u64>(image.size(), codes[code] + FIELD_AFTER);
    }
    window.before = -static_cast<s64>(FIELD_BEFORE);
    window.after = static_cast<s64>(FIELD_AFTER);
    window.id_end = 0;
    window.max_shift = window.Slots() - 1;
    return window;
}

/// The 4-byte fields of the shaders in played and control pairs, and which of them hold a
/// value few shaders have there, which can tie two shaders together.
struct Fields {
    size_t slots = 0;
    /// Per code header, its row of fields, or -1.
    std::vector<s32> row_of;
    std::vector<u32> values;
    std::vector<u8> rare;
    size_t rare_max = 0;

    const u32* Of(u32 code) const {
        return values.data() + static_cast<size_t>(row_of[code]) * slots;
    }
    const u8* RareOf(u32 code) const {
        return rare.data() + static_cast<size_t>(row_of[code]) * slots;
    }
};

/// Reads the fields of the code headers `needed` marks, and logs per place how the fields of
/// all shaders differ there.
Fields ReadFields(std::span<const u8> image, const Window& window, const std::vector<u8>& needed) {
    const u32 count = static_cast<u32>(window.anchors.size());
    Fields fields;
    fields.slots = window.Slots();
    fields.row_of.assign(count, -1);
    const std::vector<u32> anchored = window.Anchored();
    std::vector<u32> rows;
    for (u32 code = 0; code < count; ++code) {
        if (needed[code]) {
            fields.row_of[code] = static_cast<s32>(rows.size());
            rows.push_back(code);
        }
    }
    fields.values.assign(rows.size() * fields.slots, 0);
    fields.rare.assign(rows.size() * fields.slots, 0);
    fields.rare_max = std::max<size_t>(8, anchored.size() / 100);
    for (size_t row = 0; row < rows.size(); ++row) {
        for (size_t slot = 0; slot < fields.slots; ++slot) {
            fields.values[row * fields.slots + slot] = window.Read32(image, rows[row], slot);
        }
    }
    // A block of places at a time, so each shader's bytes are read in one go.
    constexpr size_t BLOCK = 64;
    const size_t n_codes = anchored.size();
    std::vector<u32> block(n_codes * BLOCK);
    std::vector<u32> column(n_codes);
    std::string profile;
    for (size_t first = 0; first < fields.slots; first += BLOCK) {
        const size_t n = std::min(BLOCK, fields.slots - first);
        for (size_t i = 0; i < n_codes; ++i) {
            for (size_t k = 0; k < n; ++k) {
                block[k * n_codes + i] = window.Read32(image, anchored[i], first + k);
            }
        }
        for (size_t k = 0; k < n; ++k) {
            const size_t slot = first + k;
            std::copy_n(block.begin() + static_cast<std::ptrdiff_t>(k * n_codes), n_codes,
                        column.begin());
            std::sort(column.begin(), column.end());
            size_t distinct = 0;
            size_t zeros = 0;
            size_t top = 0;
            u32 top_value = 0;
            for (size_t i = 0; i < column.size();) {
                size_t j = i;
                while (j < column.size() && column[j] == column[i]) {
                    ++j;
                }
                ++distinct;
                zeros += column[i] == 0 ? j - i : 0;
                if (j - i > top) {
                    top = j - i;
                    top_value = column[i];
                }
                i = j;
            }
            for (size_t row = 0; row < rows.size(); ++row) {
                const u32 value = fields.values[row * fields.slots + slot];
                if (value == 0 || value == 0xFFFFFFFF) {
                    continue;
                }
                const auto [low, high] = std::equal_range(column.begin(), column.end(), value);
                if (static_cast<size_t>(high - low) <= fields.rare_max) {
                    fields.rare[row * fields.slots + slot] = 1;
                }
            }
            if (zeros == n_codes) {
                continue;
            }
            profile += fmt::format("{}{:+#x}: {}% zero, {} values, most {:#x} x{}",
                                   profile.empty() ? "" : "; ", window.Offset(slot),
                                   zeros * 100 / std::max<size_t>(n_codes, 1), distinct,
                                   top_value, top);
            if (profile.size() > 3000) {
                LOG_INFO(Frontend, "[ShaderScan] Unreal links: fields {}: {}", window.what,
                         profile);
                profile.clear();
            }
        }
    }
    if (!profile.empty()) {
        LOG_INFO(Frontend, "[ShaderScan] Unreal links: fields {}: {}", window.what, profile);
    }
    return fields;
}

/// The two places of a cell of CountShared: the vertex shader's and the fragment shader's.
std::pair<size_t, size_t> CellSlots(size_t cell, size_t max_shift) {
    const size_t shifts = 2 * max_shift + 1;
    const size_t a = cell / shifts;
    return {a, a + cell % shifts - max_shift};
}

/// Per pair of fields (vertex shader field, fragment shader field at most `max_shift` fields
/// away): how many pairs hold the same rare value in both. A pair counts once, whichever of
/// its copies match.
std::vector<u32> CountShared(const Fields& fields, std::span<const PairCopies> pairs,
                             size_t max_shift, std::vector<u32>* examples) {
    const size_t slots = fields.slots;
    const size_t shifts = 2 * max_shift + 1;
    const size_t cells = slots * shifts;
    std::vector<u32> counts(cells, 0);
    if (examples) {
        examples->assign(cells, 0);
    }
    std::vector<u64> seen((cells + 63) / 64);
    for (const PairCopies& pair : pairs) {
        std::fill(seen.begin(), seen.end(), 0);
        for (const u32 v : pair.vertex) {
            const u32* vertex = fields.Of(v);
            const u8* vertex_rare = fields.RareOf(v);
            for (const u32 f : pair.fragment) {
                const u32* fragment = fields.Of(f);
                const u8* fragment_rare = fields.RareOf(f);
                for (size_t a = 0; a < slots; ++a) {
                    if (!vertex_rare[a]) {
                        continue;
                    }
                    const size_t low = a > max_shift ? a - max_shift : 0;
                    const size_t high = std::min(slots, a + max_shift + 1);
                    for (size_t b = low; b < high; ++b) {
                        if (fragment_rare[b] && fragment[b] == vertex[a]) {
                            const size_t cell = a * shifts + (b + max_shift - a);
                            seen[cell / 64] |= u64{1} << (cell % 64);
                            if (examples && (*examples)[cell] == 0) {
                                (*examples)[cell] = vertex[a];
                            }
                        }
                    }
                }
            }
        }
        for (size_t word = 0; word < seen.size(); ++word) {
            for (u64 bits = seen[word]; bits != 0; bits &= bits - 1) {
                ++counts[word * 64 + static_cast<size_t>(std::countr_zero(bits))];
            }
        }
    }
    return counts;
}

/// Ways one shader's field may name the other shader of its pair.
enum RefForm : u32 {
    RefIndex,
    RefIndexFromOne,
    RefPackageOffset,
    RefOffsetFromFirst,
    RefIndexStep,
    RefOffsetStep,
    RefRecordOffset,
    RefRecordStep,
    RefRecordFromFirst,
    REF_FORMS,
};

constexpr std::array<const char*, REF_FORMS> REF_FORM_NAMES{
    "index",
    "index from 1",
    "package offset",
    "offset from the first code",
    "headers between them",
    "bytes between them",
    "record's package offset",
    "bytes between the records",
    "record's offset from the first record",
};

/// Per direction (0: a vertex field names the fragment shader, 1: the other way), form and
/// field: how many pairs have it.
std::vector<u32> CountPartnerRefs(const Fields& fields, std::span<const PairCopies> pairs,
                                  const std::vector<u64>& codes, u64 first_code,
                                  const Window& window, u64 first_record) {
    const size_t slots = fields.slots;
    const size_t cells = 2 * REF_FORMS * slots;
    std::vector<u32> counts(cells, 0);
    std::vector<u8> seen(cells);
    for (const PairCopies& pair : pairs) {
        std::fill(seen.begin(), seen.end(), 0);
        for (const u32 v : pair.vertex) {
            for (const u32 f : pair.fragment) {
                for (u32 direction = 0; direction < 2; ++direction) {
                    const u32 holder = direction == 0 ? v : f;
                    const u32 other = direction == 0 ? f : v;
                    const bool records = window.records && window.anchors[holder] != 0 &&
                                         window.anchors[other] != 0;
                    const std::array<u32, REF_FORMS> forms{
                        other,
                        other + 1,
                        static_cast<u32>(codes[other]),
                        static_cast<u32>(codes[other] - first_code),
                        other - holder,
                        static_cast<u32>(codes[other] - codes[holder]),
                        records ? static_cast<u32>(window.anchors[other]) : 0U,
                        records ? static_cast<u32>(window.anchors[other] - window.anchors[holder])
                                : 0U,
                        records ? static_cast<u32>(window.anchors[other] - first_record) : 0U,
                    };
                    const u32* values = fields.Of(holder);
                    for (u32 form = 0; form < REF_FORMS; ++form) {
                        if (forms[form] == 0) {
                            continue;
                        }
                        for (size_t slot = 0; slot < slots; ++slot) {
                            if (values[slot] == forms[form]) {
                                seen[(direction * REF_FORMS + form) * slots + slot] = 1;
                            }
                        }
                    }
                }
            }
        }
        for (size_t cell = 0; cell < cells; ++cell) {
            counts[cell] += seen[cell];
        }
    }
    return counts;
}

/// The code headers of the played pairs and the controls.
std::vector<u8> PairCodes(u32 count, std::span<const PairCopies> played, const Controls& controls) {
    std::vector<u8> mask(count, 0);
    const auto mark = [&](std::span<const PairCopies> pairs) {
        for (const PairCopies& pair : pairs) {
            for (const u32 code : pair.vertex) {
                mask[code] = 1;
            }
            for (const u32 code : pair.fragment) {
                mask[code] = 1;
            }
        }
    };
    mark(played);
    mark(controls.shuffled);
    mark(controls.shifted);
    return mask;
}

/// Orders learned pairs for pre-compiling, those of the played vertex shaders first, keeps the
/// first `max_kept` and logs how well they hold the played pairs.
std::vector<std::pair<u64, u64>> RankLearned(std::vector<std::pair<u64, u64>> candidates,
                                             std::span<const PairCopies> played,
                                             const std::vector<u64>& hash_of,
                                             const std::string& what, bool capped,
                                             std::chrono::steady_clock::time_point start,
                                             size_t max_kept = MAX_LEARNED_PAIRS) {
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    // How well the link finds the played pairs, and how many fragment shaders it gives each
    // played vertex shader.
    std::unordered_set<u64> played_vertex;
    std::unordered_set<u64> played_fragment;
    std::set<std::pair<u64, u64>> played_hashes;
    for (const PairCopies& pair : played) {
        const u64 vertex = hash_of[pair.vertex.front()];
        const u64 fragment = hash_of[pair.fragment.front()];
        played_vertex.insert(vertex);
        played_fragment.insert(fragment);
        played_hashes.emplace(vertex, fragment);
    }
    size_t found = 0;
    std::unordered_map<u64, size_t> per_played_vertex;
    for (const auto& pair : candidates) {
        found += played_hashes.contains(pair) ? 1 : 0;
        if (played_vertex.contains(pair.first)) {
            ++per_played_vertex[pair.first];
        }
    }
    size_t per_vertex_total = 0;
    for (const auto& [vertex, n] : per_played_vertex) {
        per_vertex_total += n;
    }
    // Pairs whose vertex shader the game drew: the game is more likely to draw those.
    const auto rank = [&](const std::pair<u64, u64>& pair) {
        return played_vertex.contains(pair.first)       ? 0
               : played_fragment.contains(pair.second) ? 1
                                                       : 2;
    };
    std::stable_sort(candidates.begin(), candidates.end(),
                     [&](const auto& x, const auto& y) { return rank(x) < rank(y); });
    const size_t kept = std::min(candidates.size(), max_kept);
    std::set<u64> vertex_shaders;
    std::set<u64> fragment_shaders;
    for (const auto& [vertex, fragment] : candidates) {
        vertex_shaders.insert(vertex);
        fragment_shaders.insert(fragment);
    }
    LOG_INFO(Frontend,
             "[ShaderScan] Unreal links: {} ties {}{} pairs of {} vertex and {} fragment "
             "shaders; it holds {} of {} played pairs; {} played vertex shaders get {} fragment "
             "shaders each on average; {} pairs kept for pre-compiling, played vertex shaders "
             "first ({} ms)",
             what, capped ? "more than " : "", candidates.size(), vertex_shaders.size(),
             fragment_shaders.size(), found, played_hashes.size(), per_played_vertex.size(),
             per_played_vertex.empty() ? 0 : per_vertex_total / per_played_vertex.size(), kept,
             std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - start)
                 .count());
    candidates.resize(kept);
    return candidates;
}

/// Logs which fields of the window the two shaders of a played pair share, or use to name each
/// other, against the controls. When one field ties most played pairs together and few random
/// ones, returns the pairs it ties for every shader in the package.
std::vector<std::pair<u64, u64>> StudyFields(std::span<const u8> image, const Window& window,
                                             const std::vector<u64>& codes, u64 first_code,
                                             std::span<const PairCopies> played,
                                             const Controls& controls,
                                             const std::vector<u64>& hash_of,
                                             const std::vector<LinkSurvey::Stage>& stage_of) {
    const auto start = std::chrono::steady_clock::now();
    const u32 count = static_cast<u32>(codes.size());
    const Fields fields = ReadFields(image, window, PairCodes(count, played, controls));
    const size_t max_shift = window.max_shift;
    std::vector<u32> examples;
    const std::vector<u32> shared = CountShared(fields, played, max_shift, &examples);
    const std::vector<u32> shared_shuffled =
        CountShared(fields, controls.shuffled, max_shift, nullptr);
    const std::vector<u32> shared_shifted =
        CountShared(fields, controls.shifted, max_shift, nullptr);
    std::vector<size_t> cells(shared.size());
    std::iota(cells.begin(), cells.end(), size_t{0});
    const auto score = [&](size_t cell) {
        return static_cast<s64>(shared[cell]) -
               static_cast<s64>(std::max(shared_shuffled[cell], shared_shifted[cell]));
    };
    std::sort(cells.begin(), cells.end(), [&](size_t a, size_t b) {
        return score(a) != score(b) ? score(a) > score(b) : a < b;
    });
    std::string line;
    for (size_t i = 0; i < std::min<size_t>(cells.size(), 10); ++i) {
        const size_t cell = cells[i];
        if (shared[cell] < 3) {
            break;
        }
        const auto [a, b] = CellSlots(cell, max_shift);
        line += fmt::format("{}vertex {:+#x} = fragment {:+#x}: {} played, {} shuffled, {} "
                            "moved (e.g. {:#x})",
                            line.empty() ? "" : "; ", window.Offset(a), window.Offset(b),
                            shared[cell], shared_shuffled[cell], shared_shifted[cell],
                            examples[cell]);
    }
    LOG_INFO(Frontend,
             "[ShaderScan] Unreal links: fields {} the two shaders of a pair share (values at "
             "most {} shaders have there; {} played pairs, as many shuffled, {} moved {} headers "
             "on): {}",
             window.what, fields.rare_max, played.size(), controls.shifted.size(), CONTROL_SHIFT,
             line.empty() ? std::string{"none"} : line);

    u64 first_record = ~u64{0};
    for (const u64 anchor : window.anchors) {
        if (anchor != 0) {
            first_record = std::min(first_record, anchor);
        }
    }
    const std::vector<u32> refs =
        CountPartnerRefs(fields, played, codes, first_code, window, first_record);
    const std::vector<u32> refs_shuffled =
        CountPartnerRefs(fields, controls.shuffled, codes, first_code, window, first_record);
    const std::vector<u32> refs_shifted =
        CountPartnerRefs(fields, controls.shifted, codes, first_code, window, first_record);
    std::vector<size_t> ref_cells(refs.size());
    std::iota(ref_cells.begin(), ref_cells.end(), size_t{0});
    const auto ref_score = [&](size_t cell) {
        return static_cast<s64>(refs[cell]) -
               static_cast<s64>(std::max(refs_shuffled[cell], refs_shifted[cell]));
    };
    std::sort(ref_cells.begin(), ref_cells.end(), [&](size_t a, size_t b) {
        return ref_score(a) != ref_score(b) ? ref_score(a) > ref_score(b) : a < b;
    });
    line.clear();
    for (size_t i = 0; i < std::min<size_t>(ref_cells.size(), 6); ++i) {
        const size_t cell = ref_cells[i];
        if (refs[cell] < 3) {
            break;
        }
        const size_t direction = cell / (REF_FORMS * fields.slots);
        const size_t form = cell / fields.slots % REF_FORMS;
        line += fmt::format("{}{} field {:+#x} holds the {}'s {}: {} played, {} shuffled, {} "
                            "moved",
                            line.empty() ? "" : "; ", direction == 0 ? "vertex" : "fragment",
                            window.Offset(cell % fields.slots),
                            direction == 0 ? "fragment shader" : "vertex shader",
                            REF_FORM_NAMES[form], refs[cell], refs_shuffled[cell],
                            refs_shifted[cell]);
    }
    LOG_INFO(Frontend,
             "[ShaderScan] Unreal links: fields {} that name the other shader of a pair: {}",
             window.what, line.empty() ? std::string{"none"} : line);

    // A link ties most played pairs and few random ones.
    std::vector<std::pair<u64, u64>> learned;
    const size_t best = cells.empty() ? 0 : cells.front();
    const u32 strong = cells.empty() ? 0 : shared[best];
    const bool link = strong >= std::max<size_t>(10, played.size() / 4) &&
                      strong >= 5 * std::max<u32>(shared_shuffled[best], 1);
    if (!link) {
        LOG_INFO(Frontend,
                 "[ShaderScan] Unreal links: no field {} ties most played pairs and few shuffled "
                 "ones, so no pairs are learned from them ({} ms)",
                 window.what,
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - start)
                     .count());
        return learned;
    }
    const auto [a, b] = CellSlots(best, max_shift);
    // Values few shaders have at the two places, over all shaders.
    const auto column_counts = [&](size_t slot) {
        std::unordered_map<u32, u32> counts;
        for (u32 code = 0; code < count; ++code) {
            if (window.anchors[code] != 0) {
                ++counts[window.Read32(image, code, slot)];
            }
        }
        return counts;
    };
    const std::unordered_map<u32, u32> counts_a = column_counts(a);
    const std::unordered_map<u32, u32> counts_b = a == b ? counts_a : column_counts(b);
    const auto rare_at = [&](const std::unordered_map<u32, u32>& counts, u32 value) {
        if (value == 0 || value == 0xFFFFFFFF) {
            return false;
        }
        const auto it = counts.find(value);
        return it != counts.end() && it->second <= fields.rare_max;
    };
    std::unordered_map<u32, std::vector<u32>> vertex_by;
    std::unordered_map<u32, std::vector<u32>> fragment_by;
    for (u32 code = 0; code < count; ++code) {
        if (window.anchors[code] == 0) {
            continue;
        }
        if (stage_of[code] == LinkSurvey::Stage::Vertex) {
            const u32 value = window.Read32(image, code, a);
            if (rare_at(counts_a, value)) {
                vertex_by[value].push_back(code);
            }
        }
        if (stage_of[code] == LinkSurvey::Stage::Fragment) {
            const u32 value = window.Read32(image, code, b);
            if (rare_at(counts_b, value)) {
                fragment_by[value].push_back(code);
            }
        }
    }
    std::vector<std::pair<u64, u64>> candidates;
    bool capped = false;
    for (const auto& [value, vertices] : vertex_by) {
        const auto it = fragment_by.find(value);
        if (it == fragment_by.end()) {
            continue;
        }
        for (const u32 v : vertices) {
            for (const u32 f : it->second) {
                if (candidates.size() >= MAX_CANDIDATES) {
                    capped = true;
                    break;
                }
                candidates.emplace_back(hash_of[v], hash_of[f]);
            }
        }
    }
    learned = RankLearned(std::move(candidates), played, hash_of,
                          fmt::format("learned link vertex {:+#x} = fragment {:+#x} {}",
                                      window.Offset(a), window.Offset(b), window.what),
                          capped, start);
    return learned;
}

/// An 8-byte value by a code header, which may be the shader's id elsewhere.
struct IdEntry {
    u64 value;
    u32 code;
    u32 window;
};

/// Places in the window where most shaders hold a different 8-byte value: possible ids. With
/// `overlap`, places 4 bytes apart are all kept, since an id can start at either half of a
/// longer value.
std::vector<s32> ChooseIdWindows(std::span<const u8> image, const Window& window, size_t most,
                                 bool overlap) {
    struct Candidate {
        s32 at;
        double distinct;
    };
    std::vector<u32> sample = window.Anchored();
    if (sample.size() > 8192) {
        std::vector<u32> fewer;
        for (size_t i = 0; i < 8192; ++i) {
            fewer.push_back(sample[i * sample.size() / 8192]);
        }
        sample = std::move(fewer);
    }
    std::vector<Candidate> candidates;
    std::vector<u64> values;
    for (s64 at = window.before; at <= window.id_end - 8; at += 4) {
        values.clear();
        size_t zero_bytes = 0;
        for (const u32 code : sample) {
            const std::optional<u64> value = window.Read64(image, code, at);
            if (!value || *value == 0) {
                continue;
            }
            values.push_back(*value);
            for (u32 byte = 0; byte < 8; ++byte) {
                zero_bytes += ((*value >> (byte * 8)) & 0xFF) == 0 ? 1 : 0;
            }
        }
        if (values.size() < std::max<size_t>(16, sample.size() / 2) ||
            zero_bytes > 3 * values.size()) {
            continue;
        }
        std::sort(values.begin(), values.end());
        const size_t distinct = static_cast<size_t>(
            std::unique(values.begin(), values.end()) - values.begin());
        const double share = static_cast<double>(distinct) / static_cast<double>(sample.size());
        if (share >= 0.5) {
            candidates.push_back({static_cast<s32>(at), share});
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& x, const Candidate& y) {
        return x.distinct != y.distinct ? x.distinct > y.distinct : x.at > y.at;
    });
    std::vector<s32> kept;
    for (const Candidate& candidate : candidates) {
        if (kept.size() < most && std::ranges::none_of(kept, [&](s32 other) {
                return std::abs(other - candidate.at) < (overlap ? 4 : 8);
            })) {
            kept.push_back(candidate.at);
        }
    }
    std::sort(kept.begin(), kept.end());
    return kept;
}

/// Searches the code package outside the code for the ids in the window of every shader, and
/// logs whether the two shaders of a played pair are listed close together there, against
/// the controls.
void StudyCodeIds(std::span<const u8> image, const Window& window, u64 first_code, u64 code_end,
                  std::span<const s32> windows, std::span<const PairCopies> played,
                  const Controls& controls, const std::vector<u8>& played_mask,
                  const std::function<bool()>& stopped) {
    const auto start = std::chrono::steady_clock::now();
    const u32 count = static_cast<u32>(window.anchors.size());
    std::vector<IdEntry> entries;
    // Where the ids are in the window of each shader: another copy of a shader may have the
    // same id there, which is not a reference.
    std::vector<u64> own_places;
    for (u32 w = 0; w < windows.size(); ++w) {
        for (u32 code = 0; code < count; ++code) {
            const std::optional<u64> pos = window.At(code, windows[w], 8);
            if (!pos) {
                continue;
            }
            own_places.push_back(*pos);
            const u64 value = Load64(image.data() + *pos);
            if (value != 0) {
                entries.push_back({value, code, w});
            }
        }
    }
    std::sort(own_places.begin(), own_places.end());
    std::sort(entries.begin(), entries.end(), [](const IdEntry& x, const IdEntry& y) {
        return x.value != y.value ? x.value < y.value : x.code < y.code;
    });
    std::vector<u64> filter((size_t{1} << ID_FILTER_BITS) / 64, 0);
    for (const IdEntry& entry : entries) {
        const u64 bit = MixId(entry.value);
        filter[bit >> 6] |= u64{1} << (bit & 63);
    }
    struct IdHit {
        u64 pos;
        u32 code;
        u32 window;
    };
    std::vector<IdHit> hits;
    bool capped = false;
    u64 searched = 0;
    size_t copies = 0;
    const auto search = [&](u64 begin, u64 end) {
        for (u64 pos = begin; pos + 8 <= end; ++pos) {
            if ((pos & 0xFFFFF) == 0 && stopped()) {
                return;
            }
            const u64 value = Load64(image.data() + pos);
            const u64 bit = MixId(value);
            if (((filter[bit >> 6] >> (bit & 63)) & 1) == 0 || value == 0) {
                continue;
            }
            auto it = std::lower_bound(entries.begin(), entries.end(), value,
                                       [](const IdEntry& e, u64 v) { return e.value < v; });
            if (it == entries.end() || it->value != value) {
                continue;
            }
            if (std::binary_search(own_places.begin(), own_places.end(), pos)) {
                ++copies;
                continue;
            }
            for (; it != entries.end() && it->value == value; ++it) {
                const s64 own = static_cast<s64>(window.anchors[it->code]);
                const s64 at = static_cast<s64>(pos);
                if (at >= own + window.before && at < own + window.after) {
                    continue; // In the shader's own window.
                }
                if (hits.size() >= MAX_ID_HITS) {
                    capped = true;
                    return;
                }
                hits.push_back({pos, it->code, it->window});
            }
        }
        searched += end > begin ? end - begin : 0;
    };
    search(0, std::min<u64>(first_code, image.size()));
    search(std::min<u64>(code_end, image.size()), image.size());

    // Per place: how many codes were found, where the hits follow each other.
    std::string per_window;
    std::vector<std::vector<u64>> positions(count);
    for (u32 w = 0; w < windows.size(); ++w) {
        std::vector<u8> found(count, 0);
        size_t hit_count = 0;
        std::unordered_map<s64, size_t> steps;
        u64 previous = 0;
        bool have_previous = false;
        for (const IdHit& hit : hits) {
            if (hit.window != w) {
                continue;
            }
            ++hit_count;
            found[hit.code] = 1;
            if (have_previous) {
                ++steps[static_cast<s64>(hit.pos - previous)];
            }
            previous = hit.pos;
            have_previous = true;
            if (positions[hit.code].size() < MAX_HITS_PER_CODE) {
                positions[hit.code].push_back(hit.pos);
            }
        }
        size_t codes_found = 0;
        size_t played_found = 0;
        size_t played_codes = 0;
        for (u32 code = 0; code < count; ++code) {
            codes_found += found[code];
            played_found += found[code] && played_mask[code] ? 1 : 0;
            played_codes += played_mask[code];
        }
        per_window += fmt::format(
            "{}id at {:+#x}: {} found {} times ({} of {} played), steps between them {}",
            per_window.empty() ? "" : "; ", windows[w], codes_found, hit_count, played_found,
            played_codes,
            Top(steps, 5, [](s64 d, size_t n) { return fmt::format("+{:#x} x{}", d, n); }));
    }
    LOG_INFO(Frontend,
             "[ShaderScan] Unreal links: ids {}, looked for in {} MiB of the code package "
             "outside the code ({} code headers, {} hits{}, {} more at the same place of another "
             "shader, {} ms): {}",
             window.what, searched >> 20, count, hits.size(),
             capped ? ", stopped at the limit" : "", copies,
             std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - start)
                 .count(),
             per_window.empty() ? std::string{"no places with ids"} : per_window);
    if (hits.empty()) {
        return;
    }
    for (auto& list : positions) {
        std::sort(list.begin(), list.end());
    }

    // How close the ids of the two shaders of each pair are.
    static constexpr std::array<u64, 9> LIMITS{0,       0x40,    0x100,    0x400,   0x1000,
                                               0x4000,  0x10000, 0x40000, 0x100000};
    const auto closeness = [&](std::span<const PairCopies> pairs,
                               std::unordered_map<s64, size_t>* distances,
                               std::optional<std::pair<u64, u64>>* example) {
        std::array<size_t, LIMITS.size() + 2> buckets{};
        for (const PairCopies& pair : pairs) {
            std::optional<s64> best;
            std::pair<u64, u64> where{};
            for (const u32 v : pair.vertex) {
                for (const u32 f : pair.fragment) {
                    for (const u64 pv : positions[v]) {
                        for (const u64 pf : positions[f]) {
                            const s64 d = static_cast<s64>(pf) - static_cast<s64>(pv);
                            if (!best || std::abs(d) < std::abs(*best)) {
                                best = d;
                                where = {pv, pf};
                            }
                        }
                    }
                }
            }
            if (!best) {
                ++buckets.back();
                continue;
            }
            const u64 size = static_cast<u64>(std::abs(*best));
            size_t bucket = LIMITS.size();
            for (size_t i = 0; i < LIMITS.size(); ++i) {
                if (size <= LIMITS[i]) {
                    bucket = i;
                    break;
                }
            }
            ++buckets[bucket];
            if (distances) {
                ++(*distances)[*best];
            }
            if (example && !*example && size > 0 && size <= 0x1000) {
                *example = where;
            }
        }
        std::string out = fmt::format("same place {}", buckets[0]);
        for (size_t i = 1; i < LIMITS.size(); ++i) {
            out += fmt::format(", up to {:#x} {}", LIMITS[i], buckets[i]);
        }
        out += fmt::format(", farther {}, not both found {}", buckets[LIMITS.size()],
                           buckets.back());
        return out;
    };
    std::unordered_map<s64, size_t> distances;
    std::optional<std::pair<u64, u64>> example;
    const std::string played_line = closeness(played, &distances, &example);
    const std::string shuffled_line = closeness(controls.shuffled, nullptr, nullptr);
    const std::string shifted_line = closeness(controls.shifted, nullptr, nullptr);
    LOG_INFO(Frontend,
             "[ShaderScan] Unreal links: how far apart the ids {} of a pair's two shaders are, "
             "played pairs: {}; shuffled: {}; moved {} headers on: {}; played pairs at {}",
             window.what, played_line, shuffled_line, CONTROL_SHIFT, shifted_line,
             Top(distances, 8, [](s64 d, size_t n) { return fmt::format("{:+#x} x{}", d, n); }));
    if (example) {
        const u64 from = std::min(example->first, example->second);
        const u64 to = std::max(example->first, example->second) + 8;
        const u64 begin = from - std::min<u64>(from, 0x40);
        LOG_INFO(Frontend,
                 "[ShaderScan] Unreal links:   vertex id at 0x{:x}, fragment id at 0x{:x}, bytes "
                 "from 0x{:x}: {}",
                 example->first, example->second, begin,
                 Hex(image, begin, std::min<u64>(to + 0x40 - begin, 0x600)));
    }
}

/// Ways a shader's record may give the size of its code.
enum SizeKind : u32 {
    ToNextHeader,
    ToEnd,
    ToEndPastHeader,
    ToEndPastHeaders,
    ToEndRounded,
    ToEndPastHeaderRounded,
    ToEndPastHeadersRounded,
    SIZE_KINDS,
};

constexpr std::array<const char*, SIZE_KINDS> SIZE_KIND_NAMES{
    "bytes to the next code header",
    "bytes to the end of the code",
    "bytes to the end of the code past its 0x30-byte header",
    "bytes to the end of the code past its first 0x80 bytes",
    "bytes to the end of the code rounded up to 0x100",
    "bytes to the end of the code past its 0x30-byte header rounded up to 0x100",
    "bytes to the end of the code past its first 0x80 bytes rounded up to 0x100",
};

/// Values counted in front of the code, and how many of them are tried as record starts.
constexpr size_t MARKERS_COUNTED = 32;
constexpr size_t MARKERS_LOGGED = 20;
constexpr size_t MARKERS_TRIED = 6;
/// Places of a marker looked at for the next shader's record, and after how many shaders in a
/// row without one the next place with its size is looked for farther on.
constexpr size_t RECORD_LOOKAHEAD = 8;
constexpr size_t RECORD_RESYNC_AFTER = 20;
constexpr size_t RECORD_RESYNC = 1024;
/// Shaders matched before a way of tying records to them is given up when few match.
constexpr size_t RECORD_TRIAL = 300;
/// Bytes shown from the start of the package and before the code, per line.
constexpr u64 DUMP_BYTES = 0x4000;
constexpr u64 DUMP_LINE = 0x400;
/// Most bytes of a record read.
constexpr s64 MAX_RECORD = 0xA00;
constexpr size_t RECORD_ID_WINDOWS = 12;

std::array<std::vector<u32>, SIZE_KINDS> CodeSizes(std::span<const u8> image,
                                                   const std::vector<u64>& codes, u64 code_end) {
    const u32 count = static_cast<u32>(codes.size());
    std::array<std::vector<u32>, SIZE_KINDS> sizes;
    for (auto& list : sizes) {
        list.assign(count, 0);
    }
    const auto round = [](u64 n) { return static_cast<u32>((n + 0xFF) & ~u64{0xFF}); };
    for (u32 code = 0; code < count; ++code) {
        const u64 limit = code + 1 < count ? codes[code + 1] : std::min<u64>(image.size(), code_end);
        if (code + 1 < count) {
            sizes[ToNextHeader][code] = static_cast<u32>(codes[code + 1] - codes[code]);
        }
        u64 end = 0;
        for (u64 pos = codes[code] + 0x30; pos + 8 <= limit; pos += 8) {
            const u64 value = Load64(image.data() + pos);
            if (value == SELF_BRANCH_A || value == SELF_BRANCH_B) {
                end = pos + 8;
                break;
            }
        }
        if (end == 0) {
            continue;
        }
        const u64 length = end - codes[code];
        sizes[ToEnd][code] = static_cast<u32>(length);
        sizes[ToEndPastHeader][code] = static_cast<u32>(length - 0x30);
        sizes[ToEndRounded][code] = round(length);
        sizes[ToEndPastHeaderRounded][code] = round(length - 0x30);
        if (length > 0x80) {
            sizes[ToEndPastHeaders][code] = static_cast<u32>(length - 0x80);
            sizes[ToEndPastHeadersRounded][code] = round(length - 0x80);
        }
    }
    return sizes;
}

/// Walks the code headers and a marker's places in order, and ties each header to the next
/// place whose field (`fields`, one per place) holds its size. After many headers in a row
/// without one, looks farther on. Gives up after RECORD_TRIAL headers when `trial` and few
/// matched. Returns how many headers got a place.
size_t MatchRecords(const std::vector<u32>& fields, const std::vector<u32>& sizes, bool trial,
                    std::vector<s64>* matched) {
    size_t j = 0;
    size_t matches = 0;
    size_t misses = 0;
    for (u32 code = 0; code < sizes.size(); ++code) {
        const u32 want = sizes[code];
        bool found = false;
        if (want != 0) {
            const size_t last = std::min(fields.size(), j + RECORD_LOOKAHEAD);
            for (size_t k = j; k < last; ++k) {
                if (fields[k] == want) {
                    j = k;
                    found = true;
                    break;
                }
            }
            if (!found && misses >= RECORD_RESYNC_AFTER) {
                const size_t far = std::min(fields.size(), j + RECORD_RESYNC);
                for (size_t k = last; k < far; ++k) {
                    if (fields[k] == want) {
                        j = k;
                        found = true;
                        break;
                    }
                }
            }
        }
        if (found) {
            if (matched) {
                (*matched)[code] = static_cast<s64>(j);
            }
            ++j;
            ++matches;
            misses = 0;
        } else {
            ++misses;
        }
        if (trial && code + 1 == RECORD_TRIAL && matches * 10 < RECORD_TRIAL) {
            break;
        }
    }
    return matches;
}

/// Logs the most common 4-byte values in front of the code and dumps its start and end, then
/// looks for a record per shader there: places of one of those values whose field holds the
/// size of the shaders' code, in the order of the code. Returns the records found.
std::optional<Window> FindRecords(std::span<const u8> image, const std::vector<u64>& codes,
                                  u64 code_end, const std::function<bool()>& stopped) {
    const auto start = std::chrono::steady_clock::now();
    const u32 count = static_cast<u32>(codes.size());
    const u64 region_end = std::min<u64>(codes.front(), image.size());
    if (count < 16 || region_end < 0x10000) {
        return std::nullopt;
    }
    const u64 alignment = region_end % 4;
    // Values at every 7th place to find the common ones, then all places of those.
    std::vector<u32> sample;
    sample.reserve(static_cast<size_t>(region_end / 28) + 1);
    for (u64 pos = alignment; pos + 4 <= region_end; pos += 28) {
        const u32 value = Load32(image.data() + pos);
        if (value > 0xFF && value != 0xFFFFFFFF) {
            sample.push_back(value);
        }
    }
    std::sort(sample.begin(), sample.end());
    std::vector<std::pair<size_t, u32>> common;
    for (size_t i = 0; i < sample.size();) {
        size_t j = i;
        while (j < sample.size() && sample[j] == sample[i]) {
            ++j;
        }
        common.emplace_back(j - i, sample[i]);
        i = j;
    }
    sample = {};
    std::sort(common.rbegin(), common.rend());
    common.resize(std::min(common.size(), MARKERS_COUNTED));
    std::unordered_map<u32, size_t> index_of;
    std::vector<u64> low_bits(65536 / 64, 0);
    for (size_t i = 0; i < common.size(); ++i) {
        index_of.emplace(common[i].second, i);
        low_bits[(common[i].second & 0xFFFF) >> 6] |= u64{1} << (common[i].second & 63);
    }
    std::vector<size_t> exact(common.size(), 0);
    std::vector<std::vector<u64>> positions(common.size());
    const size_t keep = size_t{count} * 8 + 64;
    for (u64 pos = alignment; pos + 4 <= region_end; pos += 4) {
        if ((pos & 0xFFFFF) == alignment && stopped()) {
            return std::nullopt;
        }
        const u32 value = Load32(image.data() + pos);
        if (((low_bits[(value & 0xFFFF) >> 6] >> (value & 63)) & 1) == 0) {
            continue;
        }
        const auto it = index_of.find(value);
        if (it == index_of.end()) {
            continue;
        }
        ++exact[it->second];
        if (positions[it->second].size() < keep) {
            positions[it->second].push_back(pos);
        }
    }
    std::vector<size_t> order(common.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return exact[a] != exact[b] ? exact[a] > exact[b] : a < b; });
    std::string census;
    for (size_t rank = 0; rank < std::min(order.size(), MARKERS_LOGGED); ++rank) {
        const size_t i = order[rank];
        std::unordered_map<s64, size_t> gaps;
        for (size_t k = 1; k < positions[i].size(); ++k) {
            ++gaps[static_cast<s64>(positions[i][k] - positions[i][k - 1])];
        }
        std::string first;
        for (size_t k = 0; k < std::min<size_t>(positions[i].size(), 3); ++k) {
            first += fmt::format("{}0x{:x}", k == 0 ? "" : ", ", positions[i][k]);
        }
        census += fmt::format("{}{:#x} x{} (first at {}; gaps {})", census.empty() ? "" : "; ",
                              common[i].second, exact[i], first,
                              Top(gaps, 4, [](s64 d, size_t n) {
                                  return fmt::format("+{:#x} x{}", d, n);
                              }));
    }
    LOG_INFO(Frontend,
             "[ShaderScan] Unreal links: 4-byte values most common in the 0x{:x} bytes in front "
             "of the code, at places aligned like the code headers ({} code headers): {}",
             region_end, count, census.empty() ? std::string{"none"} : census);
    for (const auto& [what, from] : {std::pair<const char*, u64>{"from the start", 0},
                                     std::pair<const char*, u64>{
                                         "before the code",
                                         region_end - std::min(region_end, DUMP_BYTES)}}) {
        for (u64 at = from; at < std::min(from + DUMP_BYTES, region_end); at += DUMP_LINE) {
            LOG_INFO(Frontend, "[ShaderScan] Unreal links: bytes in front of the code, {}, at "
                               "0x{:x}: {}",
                     what, at, Hex(image, at, std::min(DUMP_LINE, region_end - at)));
        }
    }

    // Each marker, field and kind of size: how many shaders a record holds the size of.
    const std::array<std::vector<u32>, SIZE_KINDS> sizes = CodeSizes(image, codes, code_end);
    struct Way {
        size_t marker;
        u32 field;
        u32 kind;
        size_t matches;
        size_t control = 0;
    };
    std::vector<Way> ways;
    std::vector<u32> values;
    for (size_t rank = 0; rank < std::min(order.size(), MARKERS_TRIED); ++rank) {
        const size_t i = order[rank];
        if (exact[i] < std::max<size_t>(16, count / 2)) {
            continue;
        }
        for (u32 field = 4; field < 0x40; field += 4) {
            values.resize(positions[i].size());
            for (size_t k = 0; k < positions[i].size(); ++k) {
                const u64 at = positions[i][k] + field;
                values[k] = at + 4 <= region_end ? Load32(image.data() + at) : 0;
            }
            for (u32 kind = 0; kind < SIZE_KINDS; ++kind) {
                // A trial that is not given up runs through all shaders.
                const size_t matches = MatchRecords(values, sizes[kind], true, nullptr);
                if (matches * 10 >= std::min<size_t>(count, RECORD_TRIAL)) {
                    ways.push_back({i, field, kind, matches});
                }
            }
        }
    }
    std::sort(ways.begin(), ways.end(), [](const Way& a, const Way& b) {
        return a.matches != b.matches ? a.matches > b.matches
                                      : std::tie(a.marker, a.field, a.kind) <
                                            std::tie(b.marker, b.field, b.kind);
    });
    ways.resize(std::min<size_t>(ways.size(), 5));
    // The same with the sizes in a random order, for how many match by chance.
    std::vector<u32> order_of(count);
    std::iota(order_of.begin(), order_of.end(), 0U);
    std::mt19937 random{11};
    std::shuffle(order_of.begin(), order_of.end(), random);
    std::vector<u32> shuffled(count);
    for (Way& way : ways) {
        values.resize(positions[way.marker].size());
        for (size_t k = 0; k < positions[way.marker].size(); ++k) {
            const u64 at = positions[way.marker][k] + way.field;
            values[k] = at + 4 <= region_end ? Load32(image.data() + at) : 0;
        }
        for (u32 code = 0; code < count; ++code) {
            shuffled[code] = sizes[way.kind][order_of[code]];
        }
        way.control = MatchRecords(values, shuffled, false, nullptr);
    }
    std::string tried;
    for (const Way& way : ways) {
        tried += fmt::format("{}{:#x} with the {} at +{:#x}: {} shaders ({} with the sizes "
                             "shuffled)",
                             tried.empty() ? "" : "; ", common[way.marker].second,
                             SIZE_KIND_NAMES[way.kind], way.field, way.matches, way.control);
    }
    const auto elapsed = [&] {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - start)
            .count();
    };
    if (ways.empty() || ways.front().matches < std::max<size_t>(16, count / 2) ||
        ways.front().matches < 2 * ways.front().control) {
        LOG_INFO(Frontend,
                 "[ShaderScan] Unreal links: no records in front of the code that give the size "
                 "of each shader's code in order, so none are read ({}; {} ms)",
                 tried.empty() ? std::string{"no value found at enough places"} : tried, elapsed());
        return std::nullopt;
    }
    const Way& way = ways.front();
    values.resize(positions[way.marker].size());
    for (size_t k = 0; k < positions[way.marker].size(); ++k) {
        const u64 at = positions[way.marker][k] + way.field;
        values[k] = at + 4 <= region_end ? Load32(image.data() + at) : 0;
    }
    std::vector<s64> matched(count, -1);
    MatchRecords(values, sizes[way.kind], false, &matched);
    Window window;
    window.what = "in the shader records";
    window.records = true;
    window.anchors.assign(count, 0);
    for (u32 code = 0; code < count; ++code) {
        if (matched[code] >= 0) {
            window.anchors[code] = positions[way.marker][static_cast<size_t>(matched[code])];
        }
    }
    std::vector<u64> sorted;
    for (const u64 anchor : window.anchors) {
        if (anchor != 0) {
            sorted.push_back(anchor);
        }
    }
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    std::unordered_map<s64, size_t> gaps;
    for (size_t k = 1; k < sorted.size(); ++k) {
        ++gaps[static_cast<s64>(sorted[k] - sorted[k - 1])];
    }
    s64 typical = 0;
    size_t typical_count = 0;
    for (const auto& [gap, n] : gaps) {
        if (n > typical_count || (n == typical_count && gap < typical)) {
            typical = gap;
            typical_count = n;
        }
    }
    window.before = 0;
    window.after = std::clamp<s64>((typical + 3) & ~s64{3}, 0x100, MAX_RECORD);
    window.id_end = window.after;
    window.max_shift = 16;
    window.lows.assign(count, 0);
    window.highs.assign(count, 0);
    size_t in_order = 0;
    u64 previous = 0;
    std::string missing;
    size_t missing_count = 0;
    for (u32 code = 0; code < count; ++code) {
        const u64 anchor = window.anchors[code];
        if (anchor == 0) {
            if (missing_count++ < 8) {
                missing += fmt::format("{}#{}", missing.empty() ? "" : ", ", code);
            }
            continue;
        }
        in_order += anchor > previous ? 1 : 0;
        previous = anchor;
        const auto next = std::upper_bound(sorted.begin(), sorted.end(), anchor);
        window.lows[code] = anchor;
        window.highs[code] = std::min<u64>(
            {region_end, next == sorted.end() ? region_end : *next,
             anchor + static_cast<u64>(window.after)});
    }
    LOG_INFO(Frontend,
             "[ShaderScan] Unreal links: shader records in front of the code: they start with "
             "{:#x}, and the field at +{:#x} holds the {}; {} of {} code headers have one ({} "
             "when the sizes are shuffled), {} in the order of the code; records from 0x{:x} to "
             "0x{:x}, apart {}; read {:#x} bytes of each; headers without one: {}{}; ways tried: "
             "{} ({} ms)",
             common[way.marker].second, way.field, SIZE_KIND_NAMES[way.kind], way.matches,
             count, way.control, in_order, sorted.front(), sorted.back(),
             Top(gaps, 6, [](s64 d, size_t n) { return fmt::format("+{:#x} x{}", d, n); }),
             window.after, missing_count, missing.empty() ? "" : fmt::format(" ({})", missing),
             tried, elapsed());
    return window;
}

/// Where an id of a shader is: near its table entry, by its code header, or in its record.
/// Where an id place is: by a table entry, by the code header, in the shader record, or before
/// a shader cache entry's reference to the code (only read in the shader cache).
enum class IdSource : u8 { Entry, CodeHeader, Record, CacheEntry };

struct IdPlace {
    s32 at;
    IdSource source;
};

/// Groups of pairs: the played ones and the two controls.
enum PairGroup : u8 { PlayedGroup, ShuffledGroup, MovedGroup, PAIR_GROUPS };

/// Most parts of the code that offsets may count from.
constexpr size_t MAX_PARTS = 16;
constexpr u8 NO_PART = 0xFF;

} // Anonymous namespace

struct LinkSurvey::Impl {
    explicit Impl(size_t overlap_) : overlap{overlap_} {}

    size_t overlap;

    // Packages by class, from AddPackage.
    size_t packages = 0;
    size_t unreadable = 0;
    size_t table_errors = 0;
    std::vector<std::string> unreadable_examples;
    std::vector<std::string> table_error_examples;
    std::unordered_map<std::string, size_t> versions;
    std::unordered_map<std::string, ClassStats> classes;

    // What StudyCodePackage learned about the code package.
    bool studied = false;
    std::string code_package;
    /// Uncompressed offsets of the code headers, in order.
    std::vector<u64> codes;
    u32 residue = 0;
    u64 first_code = 0;
    u64 code_end = 0;
    u64 object_start = 0;
    bool have_object = false;
    /// Where the parts of the code start that offsets may count from, and the part of each
    /// code header (NO_PART before the first).
    std::vector<u64> part_starts;
    std::vector<u8> part_of;
    bool have_base = false;
    /// Per 256-byte slot of the package: the index + 1 of the code header in it.
    std::vector<u32> slots;
    std::vector<u8> played_mask;
    /// The code headers of the played pairs and of the control pairs, which index references
    /// and ids are looked for of.
    std::vector<u8> study_mask;
    /// Played and control pairs as (vertex index << 32) | fragment index: the groups each is in,
    /// with its number there.
    std::unordered_map<u64, std::vector<std::pair<u8, u32>>> pair_groups;
    std::array<size_t, PAIR_GROUPS> group_sizes{};
    /// Per code header: 1 when a played pair uses it as vertex shader, 2 as fragment shader.
    std::vector<u8> played_role;
    /// CODE_WINDOW bytes from before each code header, in order, zeros where the package
    /// starts.
    std::vector<u8> code_windows;
    /// Played pairs as (vertex index << 32) | fragment index.
    std::unordered_set<u64> played_pairs;
    std::vector<std::pair<u32, u32>> played_list;
    /// Where the table entry of each shader is in the code package, 0 when none was found.
    std::vector<u64> entries;
    /// The bytes by each code header, and each shader's record in front of the code when
    /// those were found.
    Window code_window;
    std::optional<Window> record_window;
    /// The places of ids, near the table entries, by the code headers or in the records.
    std::vector<IdPlace> windows;
    std::vector<u64> id_filter;
    /// The played and control shaders' ids: the shader and the place, per value. The bytes
    /// of one entry can be the id of a shader at one place and of its neighbour at another.
    std::unordered_map<u64, std::vector<std::pair<u32, u32>>> ids;
    /// The same for all shaders, which the other packages are searched for outside shader and
    /// material objects, to learn pairs from.
    std::vector<u64> all_id_filter;
    std::unordered_map<u64, std::vector<std::pair<u32, u32>>> all_ids;
    /// Per code header the shader's hash and stage (0 and Other when the search took no shader
    /// from it), and the played pairs by code header.
    std::vector<u64> code_hashes;
    std::vector<Stage> code_stages;
    std::vector<PairCopies> played_copies;
    /// Where the ids of all shaders are in the code package: an id found there is a shader's
    /// own, or another copy's, not a reference.
    std::vector<u64> own_id_places;
    /// Pairs learned from the played ones, for pre-compiling.
    std::vector<std::pair<u64, u64>> learned;

    /// Bits set for the played and control pairs, to skip most pairs without a hash lookup.
    std::vector<u64> pair_filter;

    // What SearchPiece found.
    size_t pieces = 0;
    u64 bytes = 0;
    size_t capped_pieces = 0;
    size_t skipped_pieces = 0;
    u64 skipped_bytes = 0;
    std::chrono::nanoseconds search_time{};
    /// The package AddPackage read last, which SearchPiece gets the pieces of, and the parts of
    /// it that hold objects of shader or material classes, as offsets in what Unpack hands out.
    /// Without readable tables all of it is searched.
    std::string current_package;
    bool ranges_known = false;
    std::vector<std::pair<u64, u64>> ranges;
    /// The same objects one by one, with their class and name, sorted by where they start.
    std::vector<ObjectSpan> objects;
    /// Bytes of other packages outside such objects, which are not searched.
    u64 other_class_bytes = 0;
    std::unordered_map<u32, size_t> hits_by_kind;
    std::unordered_map<u32, std::unordered_map<std::string, size_t>> files_by_kind;
    /// Ids found in other packages, by the class of the object holding them.
    std::unordered_map<std::string, size_t> id_classes;
    /// The same for the ids of all shaders found by the search for ids alone.
    std::unordered_map<std::string, size_t> all_id_classes;
    /// The tables of the package being searched, and all its objects as (start, end, export
    /// index) sorted by start, to tell what holds an id found outside the shader objects.
    Tables current_tables;
    std::vector<std::tuple<u64, u64, u32>> all_spans;
    /// The search for ids alone in the bytes outside shader and material objects.
    u64 id_only_bytes = 0;
    u64 id_only_skipped_bytes = 0;
    std::chrono::nanoseconds id_only_time{};
    std::vector<std::string> id_examples;
    std::unordered_map<std::string, size_t> id_example_classes;
    std::map<std::pair<u32, u32>, PairFinds> finds;
    /// What the search for ids alone found: the played and control pairs, and all vertex and
    /// fragment shader pairs close together.
    std::map<std::pair<u32, u32>, PairFinds> id_finds;
    CandidateStore id_store;
    /// Pairs learned from them, for pre-compiling.
    std::vector<std::pair<u64, u64>> learned_elsewhere;
    /// The control pairs, for the shader cache.
    Controls study_controls;
    /// Packages AddPackage saw a shader cache object in, and what reading those found.
    std::set<std::string> cache_packages;
    std::map<std::pair<u32, u32>, PairFinds> cache_finds;
    CandidateStore cache_store;
    std::vector<std::pair<u64, u64>> learned_cache;

    std::string KindName(u32 kind) const {
        if (kind < KIND_COUNT) {
            return KIND_NAMES[kind];
        }
        const IdPlace& place = windows[kind - KIND_COUNT];
        switch (place.source) {
        case IdSource::CodeHeader:
            return fmt::format("id at {:+#x} from the code header", place.at);
        case IdSource::Record:
            return fmt::format("id at {:+#x} in the shader record", place.at);
        case IdSource::CacheEntry:
            return fmt::format("id at {:+#x} from the shader cache entry's code offset", place.at);
        default:
            return fmt::format("id at {:+d} from the entry", place.at);
        }
    }

    /// A shader's id at a place, when it has one there.
    std::optional<u64> IdOf(std::span<const u8> data, u32 code, const IdPlace& place) const {
        switch (place.source) {
        case IdSource::CodeHeader:
            return code_window.Read64(data, code, place.at);
        case IdSource::Record:
            return record_window ? record_window->Read64(data, code, place.at) : std::nullopt;
        case IdSource::CacheEntry:
            return std::nullopt;
        default: {
            const u64 entry = entries.empty() ? 0 : entries[code];
            const s64 pos = static_cast<s64>(entry) + place.at;
            if (entry == 0 || pos < 0 || static_cast<u64>(pos) + 8 > data.size()) {
                return std::nullopt;
            }
            return Load64(data.data() + static_cast<u64>(pos));
        }
        }
    }

    /// Where a shader's id at a place is in the code package.
    std::optional<u64> IdPosition(u32 code, const IdPlace& place) const {
        switch (place.source) {
        case IdSource::CodeHeader:
            return code_window.At(code, place.at, 8);
        case IdSource::Record:
            return record_window ? record_window->At(code, place.at, 8) : std::nullopt;
        case IdSource::CacheEntry:
            return std::nullopt;
        default: {
            const u64 entry = entries.empty() ? 0 : entries[code];
            if (entry == 0 || static_cast<s64>(entry) + place.at < 0) {
                return std::nullopt;
            }
            return static_cast<u64>(static_cast<s64>(entry) + place.at);
        }
        }
    }

    /// The part of the code a code header is in, as an offset from its start, when the code
    /// is in parts.
    std::optional<u32> OffsetInPart(u32 code) const {
        if (part_of.empty() || part_of[code] == NO_PART) {
            return std::nullopt;
        }
        return static_cast<u32>(codes[code] - part_starts[part_of[code]]);
    }

    std::optional<u32> CodeAt(u64 offset) const {
        if ((offset & 0xFF) != residue) {
            return std::nullopt;
        }
        const u64 slot = offset >> 8;
        if (slot >= slots.size() || slots[slot] == 0) {
            return std::nullopt;
        }
        return slots[slot] - 1;
    }

    u32 Expected(u32 kind, u32 code) const {
        switch (kind) {
        case PackageOffset:
            return static_cast<u32>(codes[code]);
        case OffsetFromFirst:
            return static_cast<u32>(codes[code] - first_code);
        case OffsetInObject:
            return static_cast<u32>(codes[code] - object_start);
        case Index:
            return code;
        case IndexFromOne:
            return code + 1;
        case OffsetFromBase:
            // Code headers in no part get a number no field holds by chance.
            return OffsetInPart(code).value_or(0xFFFFFFFE);
        default:
            return static_cast<u32>((codes[code] - first_code) >> 8);
        }
    }

    /// Reports every place in data[begin, end) that may refer to a shader. Index references
    /// only count for played shaders, since small numbers are everywhere.
    /// Tables FindRefs reads at every byte, so most bytes take one look each: the low bytes an
    /// offset of some kind to a code header can end in, and per small number whether it is the
    /// index of a played shader (bit 0) or one more than one (bit 1).
    std::array<u8, 256> low_bytes{};
    std::vector<u8> index_bits;

    void Prepare() {
        low_bytes.fill(0);
        low_bytes[residue & 0xFF] = 1;
        low_bytes[(residue - first_code) & 0xFF] = 1;
        if (have_object) {
            low_bytes[(residue - object_start) & 0xFF] = 1;
        }
        if (have_base) {
            for (const u64 part : part_starts) {
                low_bytes[(residue - part) & 0xFF] = 1;
            }
        }
        const size_t count = codes.size();
        index_bits.assign(count + 1, 0);
        for (size_t value = 1; value <= count; ++value) {
            index_bits[value] = static_cast<u8>((value < count && study_mask[value] ? 1 : 0) |
                                                (study_mask[value - 1] ? 2 : 0));
        }
    }

    /// Reports every place in data[begin, end) that may refer to a shader. Index references
    /// only count for played shaders, since small numbers are everywhere. Call Prepare first.
    template <typename Emit>
    void FindRefs(std::span<const u8> data, u64 begin, u64 end, u32 min_index, bool all_kinds,
                  Emit&& emit) const {
        const u64 count = codes.size();
        const bool with_ids = !ids.empty();
        const u8 index_kinds = all_kinds ? 3 : 1;
        end = std::min<u64>(end, data.size());
        for (u64 pos = begin; pos + 4 <= end; ++pos) {
            const u32 value = Load32(data.data() + pos);
            if (low_bytes[value & 0xFF] != 0 && value != 0) {
                if (const auto code = CodeAt(value)) {
                    emit(pos, PackageOffset, *code);
                }
                if (value >= 0x100) {
                    if (const auto code = CodeAt(u64{value} + first_code)) {
                        emit(pos, OffsetFromFirst, *code);
                    }
                    if (have_object) {
                        if (const auto code = CodeAt(u64{value} + object_start)) {
                            emit(pos, OffsetInObject, *code);
                        }
                    }
                }
                if (have_base) {
                    for (size_t part = 0; part < part_starts.size(); ++part) {
                        if (const auto code = CodeAt(u64{value} + part_starts[part]);
                            code && part_of[*code] == part) {
                            emit(pos, OffsetFromBase, *code);
                        }
                    }
                }
            }
            // 0 for every value past the shaders, and for 0 itself.
            const u8 bits = index_bits[value <= count ? value : 0] & index_kinds;
            if (bits != 0 && value >= min_index) {
                if ((bits & 1) != 0) {
                    emit(pos, Index, value);
                }
                if ((bits & 2) != 0) {
                    emit(pos, IndexFromOne, value - 1);
                }
            }
            if (all_kinds && value != 0 && value >= min_index && value < slots.size()) {
                if (const auto code = CodeAt(first_code + u64{value} * 256);
                    code && study_mask[*code]) {
                    emit(pos, Slot, *code);
                }
            }
            if (with_ids && pos + 8 <= end) {
                const u64 id = Load64(data.data() + pos);
                const u64 bit = MixId(id);
                if (((id_filter[bit >> 6] >> (bit & 63)) & 1) != 0) {
                    if (const auto it = ids.find(id); it != ids.end()) {
                        for (const auto& [code, window] : it->second) {
                            emit(pos, KIND_COUNT + window, code);
                        }
                    }
                }
            }
        }
    }

    /// Reports every place in data[begin, end) that holds the id of any shader, and nothing
    /// else.
    template <typename Emit>
    void FindIds(std::span<const u8> data, u64 begin, u64 end, Emit&& emit) const {
        end = std::min<u64>(end, data.size());
        for (u64 pos = begin; pos + 8 <= end; ++pos) {
            const u64 id = Load64(data.data() + pos);
            const u64 bit = MixId(id);
            if (((all_id_filter[bit >> 6] >> (bit & 63)) & 1) == 0) {
                continue;
            }
            const u64 bit2 = MixId2(id);
            if (((all_id_filter[bit2 >> 6] >> (bit2 & 63)) & 1) == 0) {
                continue;
            }
            if (const auto it = all_ids.find(id); it != all_ids.end()) {
                for (const auto& [code, window] : it->second) {
                    emit(pos, KIND_COUNT + window, code);
                }
            }
        }
    }

    /// The groups a pair of code headers is in, or null when it is neither played nor a
    /// control pair.
    const std::vector<std::pair<u8, u32>>* PairGroups(u64 key) const {
        const u64 bit = MixId(key ^ 0x5851F42D4C957F2DULL) >> (ID_FILTER_BITS - PAIR_FILTER_BITS);
        if (((pair_filter[bit >> 6] >> (bit & 63)) & 1) == 0) {
            return nullptr;
        }
        const auto it = pair_groups.find(key);
        return it == pair_groups.end() ? nullptr : &it->second;
    }

    /// Finds played pairs whose vertex and fragment references are close in `hits`, which are
    /// sorted by position. Pairs whose vertex reference is before `min_first` don't count, and
    /// with `own_entries` neither do hits at a shader's own table entry. Index references are
    /// many, since small numbers are everywhere, so two of them pair only when much closer.
    /// What is found goes into `into`.
    void Pair(std::map<std::pair<u32, u32>, PairFinds>& into, const std::vector<Hit>& hits,
              std::span<const u8> data, std::string_view name, u64 offset, u64 min_first,
              bool own_entries) {
        std::vector<u32> sparse;
        std::vector<u32> dense;
        for (u32 i = 0; i < hits.size(); ++i) {
            if (own_entries && OwnEntry(hits[i])) {
                continue;
            }
            (IsIndexKind(hits[i].kind) ? dense : sparse).push_back(i);
        }
        std::map<std::pair<u32, u32>, size_t> found_here;
        const auto consider = [&](const Hit& first, const Hit& second) {
            if (first.pos < min_first || first.code == second.code || !study_mask[first.code]) {
                return;
            }
            const auto* groups = PairGroups((u64{first.code} << 32) | second.code);
            if (groups == nullptr) {
                return;
            }
            const s64 distance = static_cast<s64>(second.pos) - static_cast<s64>(first.pos);
            // Two fields can't share bytes: those are one number read at two places.
            const s64 first_size = first.kind >= KIND_COUNT ? 8 : 4;
            const s64 second_size = second.kind >= KIND_COUNT ? 8 : 4;
            if (distance > -second_size && distance < first_size) {
                return;
            }
            if (IsIndexKind(first.kind) && IsIndexKind(second.kind) &&
                (distance > INDEX_PAIR_WINDOW || distance < -INDEX_PAIR_WINDOW)) {
                return;
            }
            PairFinds& found = into[{first.kind, second.kind}];
            bool played = false;
            for (const auto& [group, number] : *groups) {
                const u64 seen = (u64{group} << 62) ^ (u64{number} << 32) ^
                                 static_cast<u32>(static_cast<s32>(distance));
                if (found.distinct_seen.insert(seen).second) {
                    ++found.distinct_at[group][distance];
                }
                played = played || group == PlayedGroup;
            }
            if (!played) {
                return;
            }
            ++found.residues[distance][(offset + first.pos) & 3];
            found.pairs.insert((u64{first.code} << 32) | second.code);
            ++found.distances[distance];
            ++found_here[{first.kind, second.kind}];
            const auto known = found.examples.find(distance);
            if (known == found.examples.end() ? found.examples.size() < MAX_EXAMPLE_DISTANCES
                                              : known->second.size() < EXAMPLES_PER_DISTANCE) {
                found.examples[distance].push_back(
                    MakeExample(first, second, distance, data, name, offset));
            }
        };
        // Every ordered pair of hits in `outer` and `inner` within `window` of each other.
        const auto pairs_of = [&](const std::vector<u32>& outer, const std::vector<u32>& inner,
                                  s64 window, bool both_orders) {
            size_t low = 0;
            for (const u32 a : outer) {
                const Hit& first = hits[a];
                while (low < inner.size() &&
                       static_cast<s64>(hits[inner[low]].pos) + window <
                           static_cast<s64>(first.pos)) {
                    ++low;
                }
                for (size_t j = low; j < inner.size() &&
                                     static_cast<s64>(hits[inner[j]].pos) <=
                                         static_cast<s64>(first.pos) + window;
                     ++j) {
                    if (inner[j] == a) {
                        continue;
                    }
                    consider(first, hits[inner[j]]);
                    if (both_orders) {
                        consider(hits[inner[j]], first);
                    }
                }
            }
        };
        pairs_of(sparse, sparse, PAIR_WINDOW, false);
        pairs_of(sparse, dense, PAIR_WINDOW, true);
        pairs_of(dense, dense, INDEX_PAIR_WINDOW, false);
        for (const auto& [kinds, count] : found_here) {
            into[kinds].files[std::string(name)] += count;
        }
    }

    /// Keeps every vertex and fragment shader pair whose ids are close together in `hits`,
    /// which are sorted by position, by the two kinds and the distance, to learn the pairs of
    /// a link from once all packages are searched. Pairs whose vertex id is before `min_first`
    /// were kept with the piece before.
    void KeepIdCandidates(CandidateStore& store, const std::vector<Hit>& hits, u64 min_first) {
        std::vector<u32> vertex;
        std::vector<u32> fragment;
        for (u32 i = 0; i < hits.size(); ++i) {
            const Stage stage = code_stages[hits[i].code];
            if (hits[i].pos >= min_first) {
                ++store.hits[static_cast<size_t>(stage)];
            }
            if (stage == Stage::Vertex) {
                vertex.push_back(i);
            } else if (stage == Stage::Fragment) {
                fragment.push_back(i);
            }
        }
        size_t low = 0;
        for (const u32 v : vertex) {
            const Hit& first = hits[v];
            while (low < fragment.size() &&
                   static_cast<s64>(hits[fragment[low]].pos) + PAIR_WINDOW <
                       static_cast<s64>(first.pos)) {
                ++low;
            }
            if (first.pos < min_first) {
                continue;
            }
            for (size_t j = low;
                 j < fragment.size() && static_cast<s64>(hits[fragment[j]].pos) <=
                                            static_cast<s64>(first.pos) + PAIR_WINDOW;
                 ++j) {
                const Hit& second = hits[fragment[j]];
                const s64 distance = static_cast<s64>(second.pos) - static_cast<s64>(first.pos);
                // Two ids can't share bytes.
                if (distance > -8 && distance < 8) {
                    continue;
                }
                if (store.count >= MAX_ID_CANDIDATES) {
                    store.capped = true;
                    return;
                }
                const u64 key = (u64{first.kind} << 48) | (u64{second.kind} << 32) |
                                static_cast<u32>(static_cast<s32>(distance));
                IdCandidates& at = store.buckets[key];
                if (at.full) {
                    continue;
                }
                if (at.pairs.size() >= MAX_ID_CANDIDATES_AT) {
                    // Lists repeat in many packages: keep each pair once.
                    const size_t before = at.pairs.size();
                    std::sort(at.pairs.begin(), at.pairs.end());
                    at.pairs.erase(std::unique(at.pairs.begin(), at.pairs.end()), at.pairs.end());
                    store.count -= before - at.pairs.size();
                    if (at.pairs.size() >= MAX_ID_CANDIDATES_AT * 3 / 4) {
                        at.full = true;
                        continue;
                    }
                }
                at.pairs.emplace_back(first.code, second.code);
                ++store.count;
            }
        }
    }

    /// Where in the package being searched an offset is: the object holding it, when known.
    std::string ObjectAt(std::string_view name, u64 at) const {
        if (!ranges_known || !name.starts_with(current_package)) {
            return {};
        }
        const auto it = std::upper_bound(objects.begin(), objects.end(), at,
                                         [](u64 value, const ObjectSpan& object) {
                                             return value < object.begin;
                                         });
        for (auto i = it; i != objects.begin();) {
            --i;
            if (at < i->end) {
                return fmt::format(", in {} at +0x{:x} of 0x{:x} bytes", i->what,
                                   at - i->begin, i->end - i->begin);
            }
            if (it - i > 8) {
                break;
            }
        }
        return {};
    }

    /// The class and the name of any object holding an offset of the package being searched,
    /// as a class name and "Class 'Name'"; empty when the offset is in no object.
    std::pair<std::string, std::string> AnyObjectAt(u64 at) const {
        const auto it = std::upper_bound(all_spans.begin(), all_spans.end(), at,
                                         [](u64 value, const std::tuple<u64, u64, u32>& span) {
                                             return value < std::get<0>(span);
                                         });
        for (auto i = it; i != all_spans.begin();) {
            --i;
            const auto& [begin, end, index] = *i;
            if (at < end) {
                const ObjectExport& object = current_tables.exports[index];
                const std::string class_name = current_tables.ClassName(object);
                return {class_name,
                        fmt::format("{} '{}'", class_name,
                                    current_tables.ObjectName(static_cast<s32>(index + 1)))};
            }
            if (it - i > 8) {
                break;
            }
        }
        return {};
    }

    /// The class of the shader or material object holding an offset of the package being
    /// searched, or empty when not known.
    std::string ClassAt(std::string_view name, u64 at) const {
        if (!ranges_known || !name.starts_with(current_package)) {
            return {};
        }
        const auto it = std::upper_bound(objects.begin(), objects.end(), at,
                                         [](u64 value, const ObjectSpan& object) {
                                             return value < object.begin;
                                         });
        for (auto i = it; i != objects.begin();) {
            --i;
            if (at < i->end) {
                return i->class_name;
            }
            if (it - i > 8) {
                break;
            }
        }
        return {};
    }

    /// Walks records `stride` bytes apart both ways from `pos`, for as long as the number at
    /// the same place in each is the index of a code header, and says what they hold.
    std::string RunAround(std::span<const u8> data, u64 pos, u64 stride, u64 offset) const {
        const u64 count = codes.size();
        const auto valid = [&](u64 at) {
            if (at + 4 > data.size()) {
                return false;
            }
            const u32 value = Load32(data.data() + at);
            return value != 0 && value < count;
        };
        u64 low = pos;
        size_t steps = 0;
        while (steps < RUN_MAX && low >= stride && valid(low - stride)) {
            low -= stride;
            ++steps;
        }
        u64 high = pos;
        while (steps < RUN_MAX && valid(high + stride)) {
            high += stride;
            ++steps;
        }
        const size_t records = static_cast<size_t>((high - low) / stride) + 1;
        std::string values;
        size_t plus_one = 0;
        u32 previous = 0;
        for (size_t i = 0; i < records; ++i) {
            const u32 value = Load32(data.data() + low + i * stride);
            if (i > 0 && value == previous + 1) {
                ++plus_one;
            }
            previous = value;
            if (i < RUN_SHOWN) {
                const u8 role = played_role[value];
                values += fmt::format("{}{}{}", i == 0 ? "" : " ", value,
                                      role == 1 ? "V" : role == 2 ? "F" : role == 3 ? "VF" : "");
            }
        }
        return fmt::format("; run of {} records {} bytes apart from 0x{:x}, numbers {}{} (V, F: "
                           "played vertex, fragment shader), {} steps of +1",
                           records, stride, offset + low, values,
                           records > RUN_SHOWN ? " ..." : "", plus_one);
    }

    PairExample MakeExample(const Hit& first, const Hit& second, s64 distance,
                            std::span<const u8> data, std::string_view name, u64 offset) const {
        PairExample example;
        const u64 from = std::min(first.pos, second.pos);
        const u64 to = std::max(first.pos, second.pos) + 8;
        const u64 start = from - std::min<u64>(from, EXAMPLE_MARGIN);
        const u64 stride = static_cast<u64>(distance < 0 ? -distance : distance);
        const bool walk = IsIndexKind(first.kind) && IsIndexKind(second.kind) &&
                          first.kind != Slot && second.kind != Slot && stride >= 8 &&
                          stride <= INDEX_PAIR_WINDOW && stride % 4 == 0;
        example.text = fmt::format(
            "vertex #{} at 0x{:x}, fragment #{} at 0x{:x} in {}{}, bytes from 0x{:x}: {}{}",
            first.code, offset + first.pos, second.code, offset + second.pos, name,
            ObjectAt(name, offset + from), offset + start,
            Hex(data, start, to + EXAMPLE_MARGIN - start),
            walk ? RunAround(data, first.pos, stride, offset) : std::string{});
        const std::array<u64, 2> places{first.pos, second.pos};
        const std::array<u32, 2> sizes{first.kind >= KIND_COUNT ? 8U : 4U,
                                       second.kind >= KIND_COUNT ? 8U : 4U};
        for (size_t i = 0; i < 2; ++i) {
            const u64 at = places[i];
            const std::array<std::pair<u64, bool>, 2> sides{
                std::pair<u64, bool>{at - std::min<u64>(at, 16), at >= 16},
                std::pair<u64, bool>{at + sizes[i], at + sizes[i] + 16 <= data.size()}};
            for (size_t side = 0; side < 2; ++side) {
                if (!sides[side].second) {
                    continue;
                }
                auto& near_bytes = example.around[i * 2 + side];
                std::memcpy(near_bytes.data(), data.data() + sides[side].first,
                            near_bytes.size());
                example.have[i * 2 + side] = std::any_of(near_bytes.begin(), near_bytes.end(),
                                                         [](u8 b) { return b != 0; });
            }
        }
        return example;
    }

    /// Where 16 bytes are in the bytes kept from before the code headers.
    std::string FindInCodeWindows(const std::array<u8, 16>& needle) const {
        if (code_windows.empty()) {
            return "not looked for";
        }
        const std::boyer_moore_horspool_searcher searcher(needle.begin(), needle.end());
        std::string out;
        size_t found = 0;
        auto at = code_windows.begin();
        while (found < 3) {
            const auto it = std::search(at, code_windows.end(), searcher);
            if (it == code_windows.end()) {
                break;
            }
            const u64 pos = static_cast<u64>(it - code_windows.begin());
            const u64 code = pos / CODE_WINDOW;
            out += fmt::format("{}0x{:x} before code #{}", found == 0 ? "" : ", ",
                               CODE_WINDOW - pos % CODE_WINDOW, code);
            ++found;
            at = it + 1;
        }
        return found == 0 ? "not before any code header" : out;
    }

    /// A reference in the code package by a shader's own table entry, or an id at the place a
    /// shader keeps its own: not a reference from elsewhere.
    bool OwnEntry(const Hit& hit) const {
        const u64 entry = entries.empty() ? 0 : entries[hit.code];
        if (entry != 0 && hit.pos + ID_RANGE >= entry && hit.pos <= entry + ID_RANGE) {
            return true;
        }
        return hit.kind >= KIND_COUNT &&
               std::binary_search(own_id_places.begin(), own_id_places.end(), hit.pos);
    }

    /// The code header a reference of a kind at `pos` of the code package names, for any
    /// shader, not only the played ones. `id_codes` has the shaders by their ids, for ids.
    std::optional<u32> DecodeAny(std::span<const u8> data, u64 pos, u32 kind,
                                 const std::unordered_map<u64, u32>& id_codes) const {
        if (kind >= KIND_COUNT) {
            if (pos + 8 > data.size()) {
                return std::nullopt;
            }
            const auto it = id_codes.find(Load64(data.data() + pos));
            if (it == id_codes.end()) {
                return std::nullopt;
            }
            return it->second;
        }
        if (pos + 4 > data.size()) {
            return std::nullopt;
        }
        const u32 value = Load32(data.data() + pos);
        const u64 count = codes.size();
        if (value == 0) {
            return std::nullopt;
        }
        switch (kind) {
        case PackageOffset:
            return CodeAt(value);
        case OffsetFromFirst:
            return value >= 0x100 ? CodeAt(u64{value} + first_code) : std::nullopt;
        case OffsetInObject:
            return have_object && value >= 0x100 ? CodeAt(u64{value} + object_start)
                                                 : std::nullopt;
        case Index:
            return value >= MIN_FOREIGN_INDEX && value < count ? std::optional<u32>{value}
                                                               : std::nullopt;
        case IndexFromOne:
            return value >= MIN_FOREIGN_INDEX && value <= count ? std::optional<u32>{value - 1}
                                                                : std::nullopt;
        case Slot:
            return value >= MIN_FOREIGN_INDEX && value < slots.size()
                       ? CodeAt(first_code + u64{value} * 256)
                       : std::nullopt;
        case OffsetFromBase:
            for (size_t part = 0; part < part_starts.size(); ++part) {
                if (const auto code = CodeAt(u64{value} + part_starts[part]);
                    code && part_of[*code] == part) {
                    return code;
                }
            }
            return std::nullopt;
        default:
            return std::nullopt;
        }
    }

    /// Logs the kinds of reference in `all` that list the two shaders of the most played pairs
    /// at one distance, and returns the first that lists enough of them and few control pairs.
    std::optional<LinkCount> ChooseLink(const std::map<std::pair<u32, u32>, PairFinds>& all,
                                        size_t played_count, std::string_view where) const {
        std::vector<LinkCount> links;
        for (const auto& [kinds, found] : all) {
            for (const auto& [distance, n] : found.distinct_at[PlayedGroup]) {
                const auto other = [&](u8 group) -> u32 {
                    const auto it = found.distinct_at[group].find(distance);
                    return it == found.distinct_at[group].end() ? 0 : it->second;
                };
                links.push_back({kinds, distance, n, other(ShuffledGroup), other(MovedGroup)});
            }
        }
        std::sort(links.begin(), links.end(), [](const LinkCount& a, const LinkCount& b) {
            return a.Score() != b.Score() ? a.Score() > b.Score()
                                          : std::tie(a.kinds, a.distance) <
                                                std::tie(b.kinds, b.distance);
        });
        std::string line;
        for (size_t i = 0; i < std::min<size_t>(links.size(), 8); ++i) {
            const LinkCount& link = links[i];
            line += fmt::format("{}vertex by {}, fragment by {} {:+d} bytes on: {} played, {} "
                                "shuffled, {} moved",
                                line.empty() ? "" : "; ", KindName(link.kinds.first),
                                KindName(link.kinds.second), link.distance, link.played,
                                link.shuffled, link.moved);
        }
        LOG_INFO(Frontend,
                 "[ShaderScan] Unreal links: references {} that list the two shaders of a pair "
                 "at one distance (different pairs; {} played, {} shuffled, {} moved {} headers "
                 "on): {}",
                 where, group_sizes[PlayedGroup], group_sizes[ShuffledGroup],
                 group_sizes[MovedGroup], CONTROL_SHIFT, line.empty() ? std::string{"none"} : line);
        const size_t need = std::max<size_t>(10, played_count / 10);
        for (const LinkCount& link : links) {
            if (link.played >= need &&
                link.played >= 5 * std::max({link.shuffled, link.moved, 1U})) {
                return link;
            }
        }
        LOG_INFO(Frontend,
                 "[ShaderScan] Unreal links: no kind of reference {} lists {} or more played "
                 "pairs at one distance and few control pairs, so no pairs are learned from them",
                 where, need);
        return std::nullopt;
    }

    /// When one kind of reference in the code package lists the two shaders of many played
    /// pairs at one distance, and few control pairs, returns the pairs it lists for all
    /// shaders. Call after the code package's references are paired.
    std::vector<std::pair<u64, u64>> LearnFromRefs(std::span<const u8> data,
                                                   std::span<const PairCopies> played,
                                                   const std::vector<u64>& hash_of,
                                                   const std::vector<Stage>& stage_of,
                                                   const std::function<bool()>& stopped) const {
        const auto start = std::chrono::steady_clock::now();
        const std::optional<LinkCount> chosen =
            ChooseLink(finds, played.size(), "in the code package");
        if (!chosen) {
            return {};
        }
        // Where in 4 bytes the vertex shader references of the played pairs are.
        const std::array<u32, 4>& residues = finds.at(chosen->kinds).residues.at(chosen->distance);
        const u64 lane =
            static_cast<u64>(std::max_element(residues.begin(), residues.end()) - residues.begin());
        // The ids of all shaders, for references by id.
        std::array<std::unordered_map<u64, u32>, 2> id_codes;
        for (size_t side = 0; side < 2; ++side) {
            const u32 kind = side == 0 ? chosen->kinds.first : chosen->kinds.second;
            if (kind < KIND_COUNT) {
                continue;
            }
            const u32 none = static_cast<u32>(codes.size());
            for (u32 code = 0; code < codes.size(); ++code) {
                const std::optional<u64> value = IdOf(data, code, windows[kind - KIND_COUNT]);
                if (!value || *value == 0) {
                    continue;
                }
                const auto [it, inserted] = id_codes[side].try_emplace(*value, code);
                if (!inserted && it->second != none && hash_of[it->second] != hash_of[code]) {
                    it->second = none;
                }
            }
            std::erase_if(id_codes[side], [&](const auto& entry) { return entry.second == none; });
        }
        std::vector<std::pair<u64, u64>> candidates;
        bool capped = false;
        const auto scan = [&](u64 begin, u64 end) {
            for (u64 pos = begin + ((lane - begin) & 3); pos < end; pos += 4) {
                if ((pos & 0xFFFFF) < 4 && stopped()) {
                    return;
                }
                const auto vertex = DecodeAny(data, pos, chosen->kinds.first, id_codes[0]);
                if (!vertex || stage_of[*vertex] != Stage::Vertex ||
                    OwnEntry({pos, chosen->kinds.first, *vertex})) {
                    continue;
                }
                const s64 other = static_cast<s64>(pos) + chosen->distance;
                if (other < 0 || static_cast<u64>(other) >= data.size() ||
                    (static_cast<u64>(other) >= first_code && static_cast<u64>(other) < code_end)) {
                    continue;
                }
                const auto fragment =
                    DecodeAny(data, static_cast<u64>(other), chosen->kinds.second, id_codes[1]);
                if (!fragment || *fragment == *vertex || stage_of[*fragment] != Stage::Fragment ||
                    OwnEntry({static_cast<u64>(other), chosen->kinds.second, *fragment})) {
                    continue;
                }
                if (candidates.size() >= MAX_CANDIDATES) {
                    capped = true;
                    return;
                }
                candidates.emplace_back(hash_of[*vertex], hash_of[*fragment]);
            }
        };
        scan(0, std::min<u64>(first_code, data.size()));
        scan(std::min<u64>(code_end, data.size()), data.size());
        return RankLearned(std::move(candidates), played, hash_of,
                           fmt::format("learned link: vertex by {}, fragment by {} {:+d} bytes on "
                                       "in the code package",
                                       KindName(chosen->kinds.first),
                                       KindName(chosen->kinds.second), chosen->distance),
                           capped, start);
    }

    /// When one kind of id outside the code package lists the two shaders of many played
    /// pairs at one distance, and few control pairs, returns the pairs it lists for all
    /// shaders. Call after all packages are searched.
    std::vector<std::pair<u64, u64>> LearnFromIds() const {
        const auto start = std::chrono::steady_clock::now();
        const std::optional<LinkCount> chosen =
            ChooseLink(id_finds, played_copies.size(), "in other packages (by ids alone)");
        if (!chosen) {
            return {};
        }
        const u64 key = (u64{chosen->kinds.first} << 48) | (u64{chosen->kinds.second} << 32) |
                        static_cast<u32>(static_cast<s32>(chosen->distance));
        std::vector<std::pair<u64, u64>> candidates;
        const auto it = id_store.buckets.find(key);
        if (it != id_store.buckets.end()) {
            for (const auto& [vertex, fragment] : it->second.pairs) {
                candidates.emplace_back(code_hashes[vertex], code_hashes[fragment]);
            }
        }
        return RankLearned(std::move(candidates), played_copies, code_hashes,
                           fmt::format("learned link: vertex by {}, fragment by {} {:+d} bytes on "
                                       "in other packages",
                                       KindName(chosen->kinds.first),
                                       KindName(chosen->kinds.second), chosen->distance),
                           id_store.capped || (it != id_store.buckets.end() && it->second.full),
                           start);
    }

    /// For each pair of `pairs`, the closest places of its vertex and fragment shader's ids in
    /// `places` (positions by code header, sorted): the distance and the two positions, or
    /// nothing when one of the two was not found.
    static std::vector<std::optional<std::tuple<u64, u64, u64>>> ClosestPlaces(
        std::span<const PairCopies> pairs,
        const std::unordered_map<u32, std::vector<u64>>& places) {
        std::vector<std::optional<std::tuple<u64, u64, u64>>> out;
        out.reserve(pairs.size());
        for (const PairCopies& pair : pairs) {
            std::optional<std::tuple<u64, u64, u64>> best;
            for (const u32 vertex : pair.vertex) {
                const auto v = places.find(vertex);
                if (v == places.end()) {
                    continue;
                }
                for (const u32 fragment : pair.fragment) {
                    const auto f = places.find(fragment);
                    if (f == places.end()) {
                        continue;
                    }
                    size_t j = 0;
                    for (const u64 pv : v->second) {
                        while (j + 1 < f->second.size() && f->second[j + 1] <= pv) {
                            ++j;
                        }
                        for (size_t k = j; k < std::min(j + 2, f->second.size()); ++k) {
                            const u64 pf = f->second[k];
                            const u64 distance = pv > pf ? pv - pf : pf - pv;
                            if (!best || distance < std::get<0>(*best)) {
                                best.emplace(distance, pv, pf);
                            }
                        }
                    }
                }
            }
            out.push_back(best);
        }
        return out;
    }

    /// Reads one object of class ShaderCache. Unreal Engine 3 keeps each shader there once,
    /// with an id, and the materials' shader maps list the ids of the shaders each material
    /// draws with. Finds the entries by their reference to the code, the places of their ids,
    /// where else in the object those ids are, and learns pairs when the played pairs' ids
    /// meet there more than the control pairs' do.
    void StudyCacheObject(std::span<const u8> data, const std::string& what, const Tables& tables,
                          const std::function<bool()>& stopped) {
        const auto start = std::chrono::steady_clock::now();
        const u32 count = static_cast<u32>(codes.size());
        for (u64 at = 0; at < std::min<u64>(data.size(), CACHE_DUMP); at += 0x80) {
            LOG_INFO(Frontend, "[ShaderScan] Unreal links: {}: bytes at 0x{:x}: {}", what, at,
                     Hex(data, at, 0x80));
        }
        // References to the code: 4 bytes holding a code header's offset from the first code.
        // The first code's offset is 0, which is everywhere, so it is left out.
        std::unordered_map<u32, u32> code_of_offset;
        std::vector<u64> filter((size_t{1} << ID_FILTER_BITS) / 64, 0);
        for (u32 code = 1; code < count; ++code) {
            const u64 offset = codes[code] - first_code;
            if (offset > 0xFFFFFFFFULL) {
                continue;
            }
            code_of_offset.emplace(static_cast<u32>(offset), code);
            const u64 bit = MixId(offset);
            filter[bit >> 6] |= u64{1} << (bit & 63);
        }
        std::vector<CacheRef> refs;
        size_t wide = 0;
        for (u64 pos = 0; pos + 4 <= data.size(); ++pos) {
            if ((pos & 0xFFFFF) == 0 && stopped()) {
                return;
            }
            const u32 value = Load32(data.data() + pos);
            if (value == 0) {
                continue;
            }
            const u64 bit = MixId(value);
            if (((filter[bit >> 6] >> (bit & 63)) & 1) == 0) {
                continue;
            }
            const auto it = code_of_offset.find(value);
            if (it == code_of_offset.end() || refs.size() >= MAX_CACHE_REFS) {
                continue;
            }
            refs.push_back({pos, it->second});
            wide += pos + 8 <= data.size() && Load32(data.data() + pos + 4) == 0 ? 1 : 0;
        }
        // The entries: the references with one 4-byte tag at one place before them, for the
        // most different shaders. Small numbers elsewhere in the object can equal a code's
        // offset too, but those name the same few codes again and again.
        struct TagCandidate {
            s32 at;
            u32 value;
            size_t count;
        };
        std::vector<TagCandidate> tag_candidates;
        for (s32 r = -CACHE_TAG_RANGE; r <= -4; ++r) {
            std::unordered_map<u32, size_t> values;
            for (const CacheRef& ref : refs) {
                if (static_cast<s64>(ref.pos) + r >= 0) {
                    const u32 value = Load32(data.data() + ref.pos - static_cast<u64>(-r));
                    if (value != 0) {
                        ++values[value];
                    }
                }
            }
            std::vector<std::pair<size_t, u32>> by_count;
            for (const auto& [value, n] : values) {
                by_count.emplace_back(n, value);
            }
            const size_t top = std::min(by_count.size(), CACHE_TAG_CANDIDATES);
            std::partial_sort(by_count.begin(), by_count.begin() + static_cast<s64>(top),
                              by_count.end(), std::greater<>());
            for (size_t i = 0; i < top; ++i) {
                tag_candidates.push_back({r, by_count[i].second, by_count[i].first});
            }
        }
        std::vector<std::vector<bool>> covered(tag_candidates.size(), std::vector<bool>(count));
        std::vector<size_t> distinct(tag_candidates.size(), 0);
        std::vector<bool> referenced(count);
        size_t codes_referenced = 0;
        for (const CacheRef& ref : refs) {
            if (!referenced[ref.code]) {
                referenced[ref.code] = true;
                ++codes_referenced;
            }
            for (size_t c = 0; c < tag_candidates.size(); ++c) {
                const TagCandidate& candidate = tag_candidates[c];
                if (static_cast<s64>(ref.pos) + candidate.at >= 0 && !covered[c][ref.code] &&
                    Load32(data.data() + ref.pos - static_cast<u64>(-candidate.at)) ==
                        candidate.value) {
                    covered[c][ref.code] = true;
                    ++distinct[c];
                }
            }
        }
        s32 tag_at = 0;
        u32 tag = 0;
        size_t tagged = 0;
        size_t tag_codes = 0;
        for (size_t c = 0; c < tag_candidates.size(); ++c) {
            if (distinct[c] > tag_codes ||
                (distinct[c] == tag_codes && tag_candidates[c].count > tagged)) {
                tag_codes = distinct[c];
                tagged = tag_candidates[c].count;
                tag = tag_candidates[c].value;
                tag_at = tag_candidates[c].at;
            }
        }
        std::vector<CacheRef> cache_entries;
        const bool have_tag = tag_codes >= std::max<size_t>(64, codes_referenced / 4);
        for (const CacheRef& ref : refs) {
            if (!have_tag || (static_cast<s64>(ref.pos) + tag_at >= 0 &&
                              Load32(data.data() + ref.pos - static_cast<u64>(-tag_at)) == tag)) {
                cache_entries.push_back(ref);
            }
        }
        std::unordered_map<s64, size_t> steps;
        std::unordered_set<u64> entry_shaders;
        std::unordered_set<u64> played_shaders;
        for (const PairCopies& pair : played_copies) {
            played_shaders.insert(code_hashes[pair.vertex.front()]);
            played_shaders.insert(code_hashes[pair.fragment.front()]);
        }
        size_t played_found = 0;
        size_t size_is_room = 0;
        size_t size_fits = 0;
        for (size_t i = 0; i < cache_entries.size(); ++i) {
            const CacheRef& entry = cache_entries[i];
            if (i > 0) {
                ++steps[static_cast<s64>(entry.pos) - static_cast<s64>(cache_entries[i - 1].pos)];
            }
            const u64 hash = code_hashes[entry.code];
            if (hash != 0 && entry_shaders.insert(hash).second && played_shaders.contains(hash)) {
                ++played_found;
            }
            // Whether the 8 bytes after an 8-byte offset hold the code's size.
            if (entry.pos + 16 <= data.size()) {
                const u64 size = Load64(data.data() + entry.pos + 8);
                const u64 room =
                    (entry.code + 1 < count ? codes[entry.code + 1] : code_end) - codes[entry.code];
                size_is_room += size == room ? 1 : 0;
                size_fits += size > 0 && size < room ? 1 : 0;
            }
        }
        std::string tag_text;
        for (u32 b = 0; b < 4; ++b) {
            const char c = static_cast<char>((tag >> (8 * b)) & 0xFF);
            tag_text += std::isprint(static_cast<unsigned char>(c)) ? c : '.';
        }
        LOG_INFO(Frontend,
                 "[ShaderScan] Unreal links: {}: {} references to code headers by their offset "
                 "from the first code ({} with 4 zero bytes after){}; {}: {} entries for {} "
                 "different shaders ({} of {} played ones); steps between them: {}; the 8 bytes "
                 "after the offset hold the room to the next code header for {}, a smaller size "
                 "for {}",
                 what, refs.size(), wide, refs.size() >= MAX_CACHE_REFS ? ", more not kept" : "",
                 have_tag ? fmt::format("{} of them, for {} of the {} code headers named, have "
                                        "0x{:08x} ('{}') {:+d} bytes before",
                                        tagged, tag_codes, codes_referenced, tag, tag_text, tag_at)
                          : fmt::format("no tag before the references to most code headers "
                                        "(best 0x{:08x} at {:+d}, {} times, for {} of the {} "
                                        "code headers named)",
                                        tag, tag_at, tagged, tag_codes, codes_referenced),
                 cache_entries.size(), entry_shaders.size(), played_found, played_shaders.size(),
                 Top(steps, 8, [](s64 d, size_t n) { return fmt::format("{:+#x} x{}", d, n); }),
                 size_is_room, size_fits);
        const size_t entries_shown = std::min<size_t>(cache_entries.size(), 3);
        for (size_t i = 0; i < entries_shown; ++i) {
            const CacheRef& entry = cache_entries[i * cache_entries.size() / entries_shown];
            const u64 from = entry.pos >= 0x80 ? entry.pos - 0x80 : 0;
            LOG_INFO(Frontend,
                     "[ShaderScan] Unreal links: {}: entry of code header #{} (hash {:016x}) at "
                     "0x{:x}: bytes from 0x{:x}: {}",
                     what, entry.code, code_hashes[entry.code], entry.pos, from,
                     Hex(data, from, 0xc0));
        }
        if (cache_entries.size() < 64 || stopped()) {
            return;
        }
        // The places near the offset whose 8 bytes differ per shader may be its ids: the
        // shader each value there belongs to, or TAKEN when two shaders have it.
        constexpr u64 TAKEN = ~u64{0};
        const auto owners_at = [&](s32 d) {
            std::unordered_map<u64, u64> owner;
            for (const CacheRef& entry : cache_entries) {
                const s64 pos = static_cast<s64>(entry.pos) + d;
                if (pos < 0 || static_cast<u64>(pos) + 8 > data.size()) {
                    continue;
                }
                const u64 value = Load64(data.data() + static_cast<u64>(pos));
                if (!IdLike(value)) {
                    continue;
                }
                const u64 hash = code_hashes[entry.code] != 0 ? code_hashes[entry.code]
                                                              : (u64{1} << 63) | entry.code;
                const auto [it, inserted] = owner.try_emplace(value, hash);
                if (!inserted && it->second != hash) {
                    it->second = TAKEN;
                }
            }
            return owner;
        };
        std::vector<std::pair<s32, size_t>> unique_at;
        for (s32 d = -CACHE_LOOK_BACK; d <= CACHE_LOOK_AHEAD && !stopped(); ++d) {
            if (d > -8 && d < 8) {
                continue;
            }
            size_t unique = 0;
            for (const auto& [value, hash] : owners_at(d)) {
                unique += hash != TAKEN ? 1 : 0;
            }
            if (unique * 10 >= entry_shaders.size() * 9) {
                unique_at.emplace_back(d, unique);
            }
        }
        // Which of those places' values the object holds again somewhere else, from a sample
        // of the entries, and how often the 8 bytes in front of them repeat. A shader map names
        // a shader's kind and then its id, so the place where an id starts has the same few
        // values in front of it wherever it is found again.
        struct SampleId {
            u64 value;
            u64 own;
            u32 place;
            u32 found;
            u64 front;
        };
        std::vector<SampleId> sample;
        const size_t sample_step = std::max<size_t>(1, cache_entries.size() / CACHE_SAMPLE);
        for (u32 p = 0; p < unique_at.size(); ++p) {
            for (size_t i = 0; i < cache_entries.size(); i += sample_step) {
                const s64 pos = static_cast<s64>(cache_entries[i].pos) + unique_at[p].first;
                if (pos < 0 || static_cast<u64>(pos) + 8 > data.size()) {
                    continue;
                }
                const u64 value = Load64(data.data() + static_cast<u64>(pos));
                if (IdLike(value)) {
                    sample.push_back({value, static_cast<u64>(pos), p, 0, 0});
                }
            }
        }
        // A value in the sample twice is no id of one shader.
        std::sort(sample.begin(), sample.end(),
                  [](const SampleId& x, const SampleId& y) { return x.value < y.value; });
        {
            std::vector<SampleId> single;
            single.reserve(sample.size());
            for (size_t i = 0; i < sample.size();) {
                size_t j = i + 1;
                while (j < sample.size() && sample[j].value == sample[i].value) {
                    ++j;
                }
                if (j == i + 1) {
                    single.push_back(sample[i]);
                }
                i = j;
            }
            sample = std::move(single);
        }
        std::vector<u64> sample_bits((size_t{1} << ID_FILTER_BITS) / 64, 0);
        for (const SampleId& id : sample) {
            const u64 bit = MixId(id.value);
            const u64 bit2 = MixId2(id.value);
            sample_bits[bit >> 6] |= u64{1} << (bit & 63);
            sample_bits[bit2 >> 6] |= u64{1} << (bit2 & 63);
        }
        for (u64 pos = 0; pos + 8 <= data.size(); ++pos) {
            if ((pos & 0xFFFFF) == 0 && stopped()) {
                return;
            }
            const u64 value = Load64(data.data() + pos);
            const u64 bit = MixId(value);
            const u64 bit2 = MixId2(value);
            if (((sample_bits[bit >> 6] >> (bit & 63)) & 1) == 0 ||
                ((sample_bits[bit2 >> 6] >> (bit2 & 63)) & 1) == 0) {
                continue;
            }
            const auto it = std::lower_bound(sample.begin(), sample.end(), value,
                                             [](const SampleId& x, u64 v) { return x.value < v; });
            if (it == sample.end() || it->value != value || it->own == pos) {
                continue;
            }
            if (it->found++ == 0 && pos >= 8) {
                it->front = Load64(data.data() + pos - 8);
            }
        }
        // Per place: how many of its sampled ids are found again, how often in all, and how
        // many of them have in front of them a value that is in front of another one too.
        std::vector<size_t> again(unique_at.size(), 0);
        std::vector<size_t> again_ids(unique_at.size(), 0);
        std::vector<std::unordered_map<u64, size_t>> fronts(unique_at.size());
        for (const SampleId& id : sample) {
            if (id.found == 0) {
                continue;
            }
            again[id.place] += id.found;
            ++again_ids[id.place];
            if (id.front != 0 && fronts[id.place].size() < CACHE_FRONTS) {
                ++fronts[id.place][id.front];
            }
        }
        std::vector<size_t> score(unique_at.size(), 0);
        for (size_t p = 0; p < unique_at.size(); ++p) {
            for (const auto& [front, n] : fronts[p]) {
                score[p] += n >= 2 ? n : 0;
            }
        }
        std::vector<size_t> order(unique_at.size());
        std::iota(order.begin(), order.end(), size_t{0});
        std::sort(order.begin(), order.end(), [&](size_t x, size_t y) {
            if (score[x] != score[y]) {
                return score[x] > score[y];
            }
            if (again_ids[x] != again_ids[y]) {
                return again_ids[x] > again_ids[y];
            }
            return std::abs(unique_at[x].first) < std::abs(unique_at[y].first);
        });
        std::string sampled;
        for (size_t i = 0; i < std::min<size_t>(order.size(), 8); ++i) {
            const size_t p = order[i];
            sampled += fmt::format(
                "{}{:+#x}: {} found again, {} times in all, {} with a value in front that "
                "another one has too ({})",
                sampled.empty() ? "" : "; ", unique_at[p].first, again_ids[p], again[p], score[p],
                Top(fronts[p], 3, [&](u64 key, size_t n) {
                    return fmt::format("{} x{}", KeyName(tables, key), n);
                }));
        }
        LOG_INFO(Frontend,
                 "[ShaderScan] Unreal links: {}: {} ids of {} of the entries looked for at {} "
                 "places, best first: {}",
                 what, sample.size(), cache_entries.size() / sample_step, unique_at.size(),
                 sampled.empty() ? std::string{"none"} : sampled);
        // The places whose ids are found again, those with kinds in front first. Places less
        // than 8 bytes from a chosen one hold the same field.
        const size_t best_score = order.empty() ? 0 : score[order.front()];
        const size_t min_score = best_score >= 16 ? std::max<size_t>(16, best_score / 10) : 0;
        std::vector<size_t> chosen;
        for (const size_t p : order) {
            if (chosen.size() >= CACHE_ID_WINDOWS) {
                break;
            }
            if (again_ids[p] == 0 || score[p] < min_score) {
                continue;
            }
            if (std::ranges::all_of(chosen, [&](size_t c) {
                    return std::abs(unique_at[c].first - unique_at[p].first) >= 8;
                })) {
                chosen.push_back(p);
            }
        }
        std::string places;
        for (const size_t c : chosen) {
            places += fmt::format("{}{:+#x}", places.empty() ? "" : ", ", unique_at[c].first);
        }
        LOG_INFO(Frontend,
                 "[ShaderScan] Unreal links: {}: 8 bytes that differ per shader at {} places "
                 "from {:+#x} to {:+#x} around the offset; ids looked for at {}",
                 what, unique_at.size(), -CACHE_LOOK_BACK, CACHE_LOOK_AHEAD,
                 places.empty() ? std::string{"none (none found again)"} : places);
        if (chosen.empty()) {
            return;
        }
        // The ids, and the places they hold in the entries, which are no references.
        std::unordered_map<u64, std::vector<std::pair<u32, u32>>> cache_ids;
        std::vector<u64> id_bits((size_t{1} << ID_FILTER_BITS) / 64, 0);
        std::vector<u64> own;
        std::vector<u32> kinds;
        for (const size_t c : chosen) {
            const s32 d = unique_at[c].first;
            const u32 kind = KIND_COUNT + static_cast<u32>(windows.size());
            windows.push_back({d, IdSource::CacheEntry});
            kinds.push_back(kind);
            const std::unordered_map<u64, u64> owner = owners_at(d);
            for (const CacheRef& entry : cache_entries) {
                const s64 pos = static_cast<s64>(entry.pos) + d;
                if (pos < 0 || static_cast<u64>(pos) + 8 > data.size()) {
                    continue;
                }
                own.push_back(static_cast<u64>(pos));
                const u64 value = Load64(data.data() + static_cast<u64>(pos));
                const auto it = owner.find(value);
                if (it == owner.end() || it->second == TAKEN) {
                    continue;
                }
                cache_ids[value].emplace_back(entry.code, kind);
                const u64 bit = MixId(value);
                const u64 bit2 = MixId2(value);
                id_bits[bit >> 6] |= u64{1} << (bit & 63);
                id_bits[bit2 >> 6] |= u64{1} << (bit2 & 63);
            }
        }
        std::sort(own.begin(), own.end());
        std::vector<Hit> hits;
        for (u64 pos = 0; pos + 8 <= data.size(); ++pos) {
            if ((pos & 0xFFFFF) == 0 && stopped()) {
                return;
            }
            const u64 value = Load64(data.data() + pos);
            const u64 bit = MixId(value);
            const u64 bit2 = MixId2(value);
            if (((id_bits[bit >> 6] >> (bit & 63)) & 1) == 0 ||
                ((id_bits[bit2 >> 6] >> (bit2 & 63)) & 1) == 0) {
                continue;
            }
            const auto it = cache_ids.find(value);
            if (it == cache_ids.end() || std::binary_search(own.begin(), own.end(), pos)) {
                continue;
            }
            for (const auto& [code, kind] : it->second) {
                if (hits.size() < MAX_PIECE_HITS) {
                    hits.push_back({pos, kind, code});
                }
            }
        }
        // Where each place's ids are found again, and what the 8 bytes before them are: a
        // shader map lists a shader by its kind's name and then its id.
        std::unordered_map<u32, std::vector<u64>> places_of;
        for (const u32 kind : kinds) {
            size_t found = 0;
            std::unordered_set<u32> shaders;
            size_t played_codes = 0;
            std::unordered_map<std::string, size_t> before;
            for (const Hit& hit : hits) {
                if (hit.kind != kind) {
                    continue;
                }
                ++found;
                if (shaders.insert(hit.code).second && played_role[hit.code] != 0) {
                    ++played_codes;
                }
                if (hit.pos >= 8) {
                    ++before[KeyName(tables, Load64(data.data() + hit.pos - 8))];
                }
            }
            LOG_INFO(Frontend,
                     "[ShaderScan] Unreal links: {}: {}: found again {} times, for {} code "
                     "headers ({} of them played); 8 bytes before: {}",
                     what, KindName(kind), found, shaders.size(), played_codes,
                     Top(before, 6, [](const std::string& key, size_t n) {
                         return fmt::format("{} x{}", key, n);
                     }));
        }
        for (const Hit& hit : hits) {
            places_of[hit.code].push_back(hit.pos);
        }
        for (auto& [code, list] : places_of) {
            std::sort(list.begin(), list.end());
            list.erase(std::unique(list.begin(), list.end()), list.end());
        }
        const size_t hits_shown = std::min<size_t>(hits.size(), 4);
        for (size_t i = 0; i < hits_shown; ++i) {
            const Hit& hit = hits[i * hits.size() / hits_shown];
            const u64 from = hit.pos >= 0x60 ? hit.pos - 0x60 : 0;
            LOG_INFO(Frontend,
                     "[ShaderScan] Unreal links: {}: {} of code header #{} found at 0x{:x}: "
                     "bytes from 0x{:x}: {}",
                     what, KindName(hit.kind), hit.code, hit.pos, from, Hex(data, from, 0xc0));
        }
        // How close the two shaders' ids of the played pairs are, against the controls.
        const auto closest_played = ClosestPlaces(played_copies, places_of);
        std::array<std::array<size_t, CACHE_LIMITS.size()>, PAIR_GROUPS> within{};
        std::array<size_t, PAIR_GROUPS> missing{};
        const std::array<std::span<const PairCopies>, PAIR_GROUPS> groups{
            played_copies, study_controls.shuffled, study_controls.shifted};
        for (size_t g = 0; g < PAIR_GROUPS; ++g) {
            const auto closest = g == 0 ? closest_played : ClosestPlaces(groups[g], places_of);
            for (const auto& best : closest) {
                if (!best) {
                    ++missing[g];
                    continue;
                }
                for (size_t l = 0; l < CACHE_LIMITS.size(); ++l) {
                    within[g][l] += std::get<0>(*best) <= CACHE_LIMITS[l] ? 1 : 0;
                }
            }
        }
        std::string closeness;
        static constexpr std::array<const char*, PAIR_GROUPS> GROUP_NAMES{"played", "shuffled",
                                                                          "moved"};
        for (size_t g = 0; g < PAIR_GROUPS; ++g) {
            closeness +=
                fmt::format("{}{} {}:", g == 0 ? "" : "; ", groups[g].size(), GROUP_NAMES[g]);
            for (size_t l = 0; l < CACHE_LIMITS.size(); ++l) {
                closeness += fmt::format(" up to 0x{:x} {},", CACHE_LIMITS[l], within[g][l]);
            }
            closeness += fmt::format(" not both found {}", missing[g]);
        }
        LOG_INFO(Frontend,
                 "[ShaderScan] Unreal links: {}: how close the ids of a pair's two shaders are "
                 "found: {}",
                 what, closeness);
        // A link at one distance, as in the other packages.
        Pair(cache_finds, hits, data, what, 0, 0, false);
        KeepIdCandidates(cache_store, hits, 0);
        const size_t need = std::max<size_t>(10, played_copies.size() / 10);
        // Pairs by closeness first: a link at one distance finds only the pairs of one kind, or
        // those whose shaders the maps list in one order.
        std::vector<std::pair<u64, u64>> candidates;
        std::vector<std::string> ways;
        bool capped = false;
        LearnByCloseness(data, what, tables, hits, closest_played, within, need, candidates, ways,
                         capped);
        if (const std::optional<LinkCount> link =
                ChooseLink(cache_finds, played_copies.size(), "in the shader cache");
            link && candidates.empty()) {
            const u64 key = (u64{link->kinds.first} << 48) | (u64{link->kinds.second} << 32) |
                            static_cast<u32>(static_cast<s32>(link->distance));
            const auto it = cache_store.buckets.find(key);
            if (it != cache_store.buckets.end()) {
                for (const auto& [vertex, fragment] : it->second.pairs) {
                    candidates.emplace_back(code_hashes[vertex], code_hashes[fragment]);
                }
            }
            capped = cache_store.capped || (it != cache_store.buckets.end() && it->second.full);
            ways.push_back(fmt::format("vertex by {}, fragment by {} {:+d} bytes on",
                                       KindName(link->kinds.first), KindName(link->kinds.second),
                                       link->distance));
        }
        if (candidates.empty()) {
            return;
        }
        std::string joined;
        for (const std::string& way : ways) {
            joined += fmt::format("{}{}", joined.empty() ? "" : " and ", way);
        }
        learned_cache = RankLearned(std::move(candidates), played_copies, code_hashes,
                                    fmt::format("learned link in the shader cache: {}", joined),
                                    capped, start, MAX_CACHE_PAIRS);
    }

    /// Whether the ids between two hits, both included, name one kind of shader twice: a
    /// shader map lists one shader of a kind, so then the two are in different maps.
    static bool CrossesMaps(std::span<const u8> data, const std::vector<Hit>& hits, u32 a, u32 b) {
        const u32 from = std::min(a, b);
        const u32 to = std::max(a, b);
        if (to - from > CACHE_MAP_HITS) {
            return false;
        }
        std::vector<u64> kinds;
        for (u32 i = from; i <= to; ++i) {
            if (hits[i].pos < 8 || (i > from && hits[i].pos == hits[i - 1].pos)) {
                continue;
            }
            const u64 kind = Load64(data.data() + hits[i].pos - 8);
            if (std::ranges::find(kinds, kind) != kinds.end()) {
                return true;
            }
            kinds.push_back(kind);
        }
        return false;
    }

    /// Shader maps list a material's shaders together, but not in a fixed order. When the
    /// played pairs' ids are much closer than the controls', the kinds of shader that pair,
    /// named by the 8 bytes before each id, tell which of the close ids pair: each vertex
    /// shader id pairs with the closest fragment shader id before it and after it of each kind
    /// its kind pairs with, as a map holds one shader of a kind.
    void LearnByCloseness(
        std::span<const u8> data, const std::string& what, const Tables& tables,
        const std::vector<Hit>& hits,
        const std::vector<std::optional<std::tuple<u64, u64, u64>>>& closest_played,
        const std::array<std::array<size_t, CACHE_LIMITS.size()>, PAIR_GROUPS>& within, size_t need,
        std::vector<std::pair<u64, u64>>& candidates, std::vector<std::string>& ways,
        bool& capped) {
        // The distance that holds the most played pairs while they are still five times the
        // controls.
        std::optional<size_t> limit;
        for (size_t l = 0; l < CACHE_LIMITS.size(); ++l) {
            const size_t controls = std::max(within[ShuffledGroup][l], within[MovedGroup][l]);
            if (within[PlayedGroup][l] >= need &&
                within[PlayedGroup][l] >= 5 * std::max<size_t>(controls, 1) &&
                (!limit || within[PlayedGroup][l] > within[PlayedGroup][*limit])) {
                limit = l;
            }
        }
        if (!limit) {
            LOG_INFO(Frontend,
                     "[ShaderScan] Unreal links: {}: the played pairs' ids are not {} or more "
                     "times close together where the controls' are few, so no pairs are learned "
                     "by closeness",
                     what, need);
            return;
        }
        const u64 window = CACHE_LIMITS[*limit];
        std::map<std::pair<u64, u64>, size_t> kinds_played;
        for (const auto& best : closest_played) {
            if (best && std::get<0>(*best) <= window && std::get<1>(*best) >= 8 &&
                std::get<2>(*best) >= 8) {
                ++kinds_played[{Load64(data.data() + std::get<1>(*best) - 8),
                                Load64(data.data() + std::get<2>(*best) - 8)}];
            }
        }
        std::set<std::pair<u64, u64>> kept_kinds;
        std::vector<std::pair<size_t, std::pair<u64, u64>>> by_count;
        for (const auto& [kinds_pair, n] : kinds_played) {
            if (n >= 2) {
                kept_kinds.insert(kinds_pair);
            }
            by_count.emplace_back(n, kinds_pair);
        }
        std::sort(by_count.rbegin(), by_count.rend());
        std::string shown;
        for (size_t i = 0; i < std::min<size_t>(by_count.size(), 8); ++i) {
            shown += fmt::format("{}{} + {} x{}", shown.empty() ? "" : "; ",
                                 KeyName(tables, by_count[i].second.first),
                                 KeyName(tables, by_count[i].second.second), by_count[i].first);
        }
        LOG_INFO(Frontend,
                 "[ShaderScan] Unreal links: {}: played pairs' ids within 0x{:x} bytes, by the 8 "
                 "bytes before the vertex and the fragment shader's id: {} kinds, {} of them "
                 "with 2 or more pairs: {}",
                 what, window, kinds_played.size(), kept_kinds.size(),
                 shown.empty() ? std::string{"none"} : shown);
        if (kept_kinds.empty()) {
            return;
        }
        std::map<u64, std::vector<u64>> fragment_kinds_of;
        for (const auto& [vertex_kind, fragment_kind] : kept_kinds) {
            fragment_kinds_of[vertex_kind].push_back(fragment_kind);
        }
        std::unordered_map<u64, std::vector<u32>> fragments_by_kind;
        std::vector<u32> vertex_hits;
        for (u32 i = 0; i < hits.size(); ++i) {
            if (hits[i].pos < 8) {
                continue;
            }
            const Stage stage = code_stages[hits[i].code];
            if (stage == Stage::Vertex) {
                vertex_hits.push_back(i);
            } else if (stage == Stage::Fragment) {
                fragments_by_kind[Load64(data.data() + hits[i].pos - 8)].push_back(i);
            }
        }
        for (const u32 v : vertex_hits) {
            const Hit& first = hits[v];
            const auto kinds_it = fragment_kinds_of.find(Load64(data.data() + first.pos - 8));
            if (kinds_it == fragment_kinds_of.end()) {
                continue;
            }
            for (const u64 fragment_kind : kinds_it->second) {
                const auto list = fragments_by_kind.find(fragment_kind);
                if (list == fragments_by_kind.end()) {
                    continue;
                }
                const std::vector<u32>& at = list->second;
                const auto after =
                    std::lower_bound(at.begin(), at.end(), first.pos,
                                     [&](u32 h, u64 pos) { return hits[h].pos < pos; });
                std::array<std::optional<u32>, 2> sides;
                if (after != at.end() && hits[*after].pos - first.pos <= window) {
                    sides[0] = *after;
                }
                if (after != at.begin() && first.pos - hits[*std::prev(after)].pos <= window) {
                    sides[1] = *std::prev(after);
                }
                for (const auto& side : sides) {
                    if (!side || CrossesMaps(data, hits, v, *side)) {
                        continue;
                    }
                    if (candidates.size() >= MAX_CANDIDATES) {
                        capped = true;
                        break;
                    }
                    candidates.emplace_back(code_hashes[first.code], code_hashes[hits[*side].code]);
                }
            }
        }
        ways.push_back(fmt::format("the closest fragment shader id before and after each "
                                   "vertex shader id within 0x{:x} bytes, of the {} kinds of "
                                   "pair the played ones show",
                                   window, kept_kinds.size()));
    }
};

LinkSurvey::LinkSurvey(size_t overlap) : impl{std::make_unique<Impl>(overlap)} {}

LinkSurvey::~LinkSurvey() = default;

void LinkSurvey::AddPackage(std::string_view name, std::span<const u8> file,
                            std::span<const u8> first_piece) {
    Impl& s = *impl;
    s.current_package = std::string(name);
    s.ranges_known = false;
    s.ranges.clear();
    s.objects.clear();
    s.all_spans.clear();
    s.current_tables = {};
    const std::optional<Summary> summary = ReadSummary(file);
    if (!summary) {
        ++s.unreadable;
        if (s.unreadable_examples.size() < EXAMPLES) {
            s.unreadable_examples.push_back(fmt::format("{}: {}", name, Hex(file, 0, 96)));
        }
        return;
    }
    ++s.packages;
    ++s.versions[fmt::format("{}/{}", summary->version, summary->licensee)];
    const u64 data_offset =
        summary->chunks.empty() ? 0 : summary->chunks.front().unpacked_offset;
    Tables tables = ReadTables(first_piece, data_offset, *summary);
    if (!tables.error.empty()) {
        ++s.table_errors;
        if (s.table_error_examples.size() < EXAMPLES) {
            s.table_error_examples.push_back(fmt::format(
                "{} (version {}/{}, {} names at 0x{:x}, {} imports at 0x{:x}, {} exports at "
                "0x{:x}): {}",
                name, summary->version, summary->licensee, summary->name_count,
                summary->name_offset, summary->import_count, summary->import_offset,
                summary->export_count, summary->export_offset, tables.error));
        }
    }
    // The objects that may name shaders: Unreal keeps a package's shader references with its
    // materials and in its shader cache objects.
    s.ranges_known = tables.error.empty() && !tables.exports.empty();
    for (size_t i = 0; i < tables.exports.size(); ++i) {
        const ObjectExport& object = tables.exports[i];
        const std::string class_name = tables.ClassName(object);
        const std::string lower = Lower(class_name);
        if (lower.find("shadercache") != std::string::npos && object.serial_size > 0) {
            s.cache_packages.emplace(name);
        }
        if (object.serial_size > 0 && object.serial_offset >= data_offset &&
            (lower.find("shader") != std::string::npos ||
             lower.find("material") != std::string::npos)) {
            const u64 begin = object.serial_offset - data_offset;
            s.ranges.emplace_back(begin, begin + object.serial_size);
            s.objects.push_back({begin, begin + object.serial_size,
                                 fmt::format("{} '{}'", class_name,
                                             tables.ObjectName(static_cast<s32>(i + 1))),
                                 class_name});
        }
    }
    std::sort(s.ranges.begin(), s.ranges.end());
    std::sort(s.objects.begin(), s.objects.end(),
              [](const ObjectSpan& a, const ObjectSpan& b) { return a.begin < b.begin; });
    std::vector<std::pair<u64, u64>> merged;
    for (const auto& range : s.ranges) {
        if (!merged.empty() && range.first <= merged.back().second) {
            merged.back().second = std::max(merged.back().second, range.second);
        } else {
            merged.push_back(range);
        }
    }
    s.ranges = std::move(merged);
    for (size_t i = 0; i < tables.exports.size(); ++i) {
        const ObjectExport& object = tables.exports[i];
        ClassStats& stats = s.classes[tables.ClassName(object)];
        ++stats.count;
        stats.bytes += object.serial_size;
        if (stats.examples.size() < 2 && object.serial_size > 0 &&
            object.serial_offset >= data_offset &&
            object.serial_offset - data_offset < first_piece.size()) {
            stats.examples.push_back(fmt::format(
                "{} in {}: 0x{:x} bytes: {}", tables.ObjectName(static_cast<s32>(i + 1)), name,
                object.serial_size,
                Hex(first_piece, object.serial_offset - data_offset,
                    std::min<u64>(object.serial_size, 0x100))));
        }
        if (s.ranges_known && object.serial_size > 0 && object.serial_offset >= data_offset) {
            const u64 begin = object.serial_offset - data_offset;
            s.all_spans.emplace_back(begin, begin + object.serial_size, static_cast<u32>(i));
        }
    }
    std::sort(s.all_spans.begin(), s.all_spans.end());
    if (s.ranges_known) {
        s.current_tables = std::move(tables);
    }
}

void LinkSurvey::StudyCodePackage(std::string_view name, std::span<const u8> file,
                                  std::span<const CodeHeader> headers,
                                  std::span<const std::pair<u64, u64>> played,
                                  const std::function<bool()>& stopped) {
    Impl& s = *impl;
    std::optional<Summary> summary = ReadSummary(file);
    std::vector<u8> image;
    std::vector<ImageSpan> spans;
    size_t failed_blocks = 0;
    std::vector<FailedBlock> failures;
    bool whole = false;
    if (!summary) {
        // A package compressed whole keeps its header inside the compressed data.
        Stats stats;
        Unpack(file, size_t{64} << 20, 0, stats, [&](std::span<const u8> piece, u64) {
            image.insert(image.end(), piece.begin(), piece.end());
            return true;
        });
        summary = ReadSummary(image);
        whole = true;
        if (!summary || !summary->chunks.empty()) {
            LOG_INFO(Frontend, "[ShaderScan] Unreal links: could not read the header of {}: {}",
                     name, Hex(file, 0, 128));
            return;
        }
    } else if (!UnpackImage(file, *summary, image, failed_blocks, spans, &failures)) {
        LOG_INFO(Frontend, "[ShaderScan] Unreal links: nothing in {} unpacked", name);
        return;
    }
    const std::span<const u8> data{image};
    const Tables tables = ReadTables(data, 0, *summary);
    LOG_INFO(Frontend,
             "[ShaderScan] Unreal links: code package {}: version {}/{}, {} names, {} imports, "
             "{} exports ({} read), {} chunks (list at 0x{:x}{}), {} MiB unpacked, {} blocks did "
             "not unpack{}{}",
             name, summary->version, summary->licensee, summary->name_count,
             summary->import_count, summary->export_count, tables.exports.size(),
             summary->chunks.size(), summary->chunk_list_offset,
             whole ? ", compressed whole" : "", image.size() >> 20, failed_blocks,
             tables.error.empty() ? "" : ", tables stopped at ", tables.error);
    LOG_INFO(Frontend, "[ShaderScan] Unreal links: header: {}", Hex(file, 0, 0x200));

    // Where the code headers the search found are in the uncompressed layout.
    std::unordered_map<u64, u64> hash_at;
    std::unordered_map<u64, Stage> stage_at;
    size_t misses = 0;
    for (const CodeHeader& header : headers) {
        const std::optional<u64> offset =
            whole ? std::optional<u64>{header.unpacked_offset}
                  : ImageOffset(spans, header.unpacked_offset);
        if (offset && *offset + 4 <= image.size() &&
            Load32(image.data() + *offset) == NVN_CODE_MAGIC) {
            hash_at.emplace(*offset, header.hash);
            stage_at.emplace(*offset, header.stage);
        } else if (misses++ < EXAMPLES) {
            LOG_INFO(Frontend,
                     "[ShaderScan] Unreal links: code header at unpacked 0x{:x} not found at "
                     "0x{:x}",
                     header.unpacked_offset, offset.value_or(0));
        }
    }
    if (hash_at.empty()) {
        LOG_INFO(Frontend, "[ShaderScan] Unreal links: none of the {} code headers found",
                 headers.size());
        return;
    }
    // Every code header in the package counts for indexes, also those the search could not
    // take a shader from. Only near the ones it found, so the tag in other data doesn't count.
    u64 lowest = ~u64{0};
    u64 highest = 0;
    for (const auto& [offset, hash] : hash_at) {
        lowest = std::min(lowest, offset);
        highest = std::max(highest, offset);
    }
    for (u64 pos = lowest - std::min<u64>(lowest & ~u64{3}, 0x100000);
         pos <= highest + 0x100000 && pos + 4 <= image.size(); pos += 4) {
        if (Load32(image.data() + pos) == NVN_CODE_MAGIC) {
            s.codes.push_back(pos);
        }
    }
    const u32 count = static_cast<u32>(s.codes.size());
    s.first_code = s.codes.front();
    s.code_end = s.codes.back() + 0x30;
    for (u64 pos = s.code_end; pos + 8 <= image.size() && pos < s.codes.back() + 0x100000;
         pos += 8) {
        const u64 value = Load64(image.data() + pos);
        if (value == SELF_BRANCH_A || value == SELF_BRANCH_B) {
            s.code_end = pos + 8;
            break;
        }
    }
    std::unordered_map<u32, size_t> residues;
    for (const u64 code : s.codes) {
        ++residues[static_cast<u32>(code & 0xFF)];
    }
    s.residue = std::max_element(residues.begin(), residues.end(), [](const auto& a,
                                                                      const auto& b) {
                    return a.second < b.second;
                })->first;
    s.slots.assign(image.size() / 256 + 1, 0);
    for (u32 code = 0; code < count; ++code) {
        if ((s.codes[code] & 0xFF) == s.residue) {
            s.slots[s.codes[code] >> 8] = code + 1;
        }
    }

    // The game's pairs, as indexes of code headers.
    std::unordered_map<u64, std::vector<u32>> codes_of;
    for (u32 code = 0; code < count; ++code) {
        if (const auto it = hash_at.find(s.codes[code]); it != hash_at.end()) {
            codes_of[it->second].push_back(code);
        }
    }
    s.played_mask.assign(count + 1, 0);
    for (const auto& [vertex, fragment] : played) {
        const auto v = codes_of.find(vertex);
        const auto f = codes_of.find(fragment);
        if (v == codes_of.end() || f == codes_of.end()) {
            continue;
        }
        for (size_t i = 0; i < std::min<size_t>(v->second.size(), 4); ++i) {
            for (size_t j = 0; j < std::min<size_t>(f->second.size(), 4); ++j) {
                const u64 key = (u64{v->second[i]} << 32) | f->second[j];
                if (s.played_pairs.insert(key).second) {
                    s.played_list.emplace_back(v->second[i], f->second[j]);
                    s.played_mask[v->second[i]] = 1;
                    s.played_mask[f->second[j]] = 1;
                }
            }
        }
    }
    size_t played_codes = 0;
    for (u32 code = 0; code < count; ++code) {
        played_codes += s.played_mask[code];
    }
    s.played_role.assign(count + 1, 0);
    for (const auto& [vertex_code, fragment_code] : s.played_list) {
        s.played_role[vertex_code] |= 1;
        s.played_role[fragment_code] |= 2;
    }
    s.code_windows.assign(size_t{count} * CODE_WINDOW, 0);
    for (u32 code = 0; code < count; ++code) {
        const u64 take = std::min<u64>(s.codes[code], CODE_WINDOW);
        std::memcpy(s.code_windows.data() + size_t{code} * CODE_WINDOW + (CODE_WINDOW - take),
                    image.data() + s.codes[code] - take, take);
    }
    LogPairOrder(played, codes_of, count);
    LOG_INFO(Frontend,
             "[ShaderScan] Unreal links: {} code headers in the package ({} with a shader the "
             "search took, {} not found where expected), first at 0x{:x}, code ends at 0x{:x}; "
             "offsets mod 256: {}; {} played pairs of {} shaders",
             count, hash_at.size(), misses, s.first_code, s.code_end,
             Top(residues, 4, [](u32 r, size_t n) { return fmt::format("0x{:02x}x{}", r, n); }),
             s.played_list.size(), played_codes);
    // Code in a block that did not unpack is missing from the search, and so are its shaders.
    for (const FailedBlock& block : failures) {
        const u64 begin = block.unpacked_offset;
        const u64 end = begin + block.unpacked;
        std::string where;
        if (end <= s.first_code) {
            where = "before the code";
        } else if (begin >= s.code_end) {
            where = "after the code";
        } else {
            // The code headers right before and after it.
            const auto next = std::ranges::lower_bound(s.codes, end);
            const auto previous = std::ranges::lower_bound(s.codes, begin);
            where = fmt::format("in the code{}{}",
                                previous != s.codes.begin()
                                    ? fmt::format(", after code header #{} at 0x{:x}",
                                                  previous - s.codes.begin() - 1, *(previous - 1))
                                    : std::string{},
                                next != s.codes.end() ? fmt::format(", before #{} at 0x{:x}",
                                                                    next - s.codes.begin(), *next)
                                                      : std::string{});
        }
        LOG_INFO(Frontend,
                 "[ShaderScan] Unreal links: block of {} that did not unpack: 0x{:x} to 0x{:x} "
                 "({} -> {} bytes at file 0x{:x}, {}), {}",
                 name, begin, end, block.packed, block.unpacked, block.file_offset,
                 block.decoder == FAILED_PAST_END ? std::string{"past the end of the layout"}
                 : block.decoder == FAILED_NOT_OODLE
                     ? std::string{"not Oodle"}
                     : fmt::format("Oodle decoder {} ({})", block.decoder,
                                   OodleCompressorName(block.decoder)),
                 where);
    }

    // The played pairs by the code headers of their shaders, for the fields and ids by them.
    std::vector<u64> hash_of(count, 0);
    std::vector<Stage> stage_of(count, Stage::Other);
    for (u32 code = 0; code < count; ++code) {
        if (const auto it = hash_at.find(s.codes[code]); it != hash_at.end()) {
            hash_of[code] = it->second;
            stage_of[code] = stage_at.at(s.codes[code]);
        }
    }
    std::vector<PairCopies> pair_copies;
    for (const auto& [vertex, fragment] :
         std::set<std::pair<u64, u64>>(played.begin(), played.end())) {
        const auto v = codes_of.find(vertex);
        const auto f = codes_of.find(fragment);
        if (v == codes_of.end() || f == codes_of.end()) {
            continue;
        }
        PairCopies pair;
        pair.vertex.assign(v->second.begin(),
                           v->second.begin() + static_cast<std::ptrdiff_t>(
                                                   std::min(v->second.size(), MAX_COPIES)));
        pair.fragment.assign(f->second.begin(),
                             f->second.begin() + static_cast<std::ptrdiff_t>(
                                                     std::min(f->second.size(), MAX_COPIES)));
        pair_copies.push_back(std::move(pair));
    }
    const Controls controls = MakeControls(pair_copies, count);
    s.code_hashes = hash_of;
    s.code_stages = stage_of;
    s.played_copies = pair_copies;
    s.study_controls = controls;
    // Played pairs and the controls, by the code headers of their copies, to tell how many
    // of each references in the packages list close together.
    s.study_mask = PairCodes(count, pair_copies, controls);
    s.pair_filter.assign((size_t{1} << PAIR_FILTER_BITS) / 64, 0);
    for (const auto& [group, pairs] :
         {std::pair<u8, std::span<const PairCopies>>{PlayedGroup, pair_copies},
          std::pair<u8, std::span<const PairCopies>>{ShuffledGroup, controls.shuffled},
          std::pair<u8, std::span<const PairCopies>>{MovedGroup, controls.shifted}}) {
        s.group_sizes[group] = pairs.size();
        for (u32 number = 0; number < pairs.size(); ++number) {
            for (const u32 v : pairs[number].vertex) {
                for (const u32 f : pairs[number].fragment) {
                    const u64 key = (u64{v} << 32) | f;
                    s.pair_groups[key].emplace_back(group, number);
                    const u64 bit = MixId(key ^ 0x5851F42D4C957F2DULL) >>
                                    (ID_FILTER_BITS - PAIR_FILTER_BITS);
                    s.pair_filter[bit >> 6] |= u64{1} << (bit & 63);
                }
            }
        }
    }
    size_t study_codes = 0;
    for (u32 code = 0; code < count; ++code) {
        study_codes += s.study_mask[code];
    }
    std::vector<size_t> shown_pairs;
    if (!pair_copies.empty()) {
        shown_pairs.push_back(0);
    }
    if (pair_copies.size() >= 2) {
        shown_pairs.push_back(pair_copies.size() / 2);
    }
    for (const size_t i : shown_pairs) {
        for (const auto& [what, code] :
             {std::pair{"vertex", pair_copies[i].vertex.front()},
              std::pair{"fragment", pair_copies[i].fragment.front()}}) {
            const u64 from = s.codes[code] - std::min(s.codes[code], FIELD_BEFORE);
            LOG_INFO(Frontend,
                     "[ShaderScan] Unreal links: played pair {}: {} shader #{} (hash {:016x}) "
                     "at 0x{:x}, bytes from {:+#x}: {}",
                     i, what, code, hash_of[code], s.codes[code],
                     static_cast<s64>(from) - static_cast<s64>(s.codes[code]),
                     Hex(data, from, s.codes[code] + FIELD_AFTER - from));
        }
    }
    s.code_window = CodeHeaderWindow(data, s.codes);
    // Pairs learned in each way, best first.
    std::vector<std::vector<std::pair<u64, u64>>> learned_sets;
    if (!pair_copies.empty() && !stopped()) {
        learned_sets.push_back(StudyFields(data, s.code_window, s.codes, s.first_code, pair_copies,
                                           controls, hash_of, stage_of));
    }
    // A record per shader in front of the code may say more about it than its code header.
    if (!stopped()) {
        s.record_window = FindRecords(data, s.codes, s.code_end, stopped);
    }
    if (s.record_window) {
        std::vector<std::pair<std::string, u32>> shown;
        for (u32 code = 0; code < count && shown.size() < 2; ++code) {
            if (s.record_window->anchors[code] != 0) {
                shown.emplace_back("the first shader with one", code);
                shown.emplace_back("the next shader", code + 1 < count ? code + 1 : code);
            }
        }
        for (const size_t i : shown_pairs) {
            shown.emplace_back(fmt::format("played pair {}'s vertex shader", i),
                               pair_copies[i].vertex.front());
            shown.emplace_back(fmt::format("played pair {}'s fragment shader", i),
                               pair_copies[i].fragment.front());
        }
        for (const auto& [what, code] : shown) {
            const u64 anchor = s.record_window->anchors[code];
            if (anchor == 0) {
                LOG_INFO(Frontend, "[ShaderScan] Unreal links: record of {} #{}: none", what,
                         code);
                continue;
            }
            LOG_INFO(Frontend,
                     "[ShaderScan] Unreal links: record of {} #{} (hash {:016x}, code at 0x{:x}) "
                     "at 0x{:x}: {}",
                     what, code, hash_of[code], s.codes[code], anchor,
                     Hex(data, anchor, s.record_window->highs[code] - anchor));
        }
        if (!pair_copies.empty() && !stopped()) {
            learned_sets.push_back(StudyFields(data, *s.record_window, s.codes, s.first_code,
                                               pair_copies, controls, hash_of, stage_of));
        }
    }

    // Objects, and which hold the code.
    std::vector<size_t> by_offset(tables.exports.size());
    for (size_t i = 0; i < by_offset.size(); ++i) {
        by_offset[i] = i;
    }
    std::sort(by_offset.begin(), by_offset.end(), [&](size_t a, size_t b) {
        return tables.exports[a].serial_offset < tables.exports[b].serial_offset;
    });
    std::vector<std::vector<u32>> codes_in(tables.exports.size());
    for (u32 code = 0; code < count; ++code) {
        const auto it = std::upper_bound(by_offset.begin(), by_offset.end(), s.codes[code],
                                         [&](u64 offset, size_t i) {
                                             return offset < tables.exports[i].serial_offset;
                                         });
        if (it == by_offset.begin()) {
            continue;
        }
        const ObjectExport& object = tables.exports[*std::prev(it)];
        if (s.codes[code] < u64{object.serial_offset} + object.serial_size) {
            codes_in[*std::prev(it)].push_back(code);
        }
    }
    size_t holder = tables.exports.size();
    for (size_t i = 0; i < tables.exports.size(); ++i) {
        if (!codes_in[i].empty() && (holder == tables.exports.size() ||
                                     codes_in[i].size() > codes_in[holder].size())) {
            holder = i;
        }
    }
    if (holder < tables.exports.size()) {
        s.have_object = true;
        s.object_start = tables.exports[holder].serial_offset;
    }
    std::vector<size_t> largest(tables.exports.size());
    for (size_t i = 0; i < largest.size(); ++i) {
        largest[i] = i;
    }
    std::sort(largest.begin(), largest.end(), [&](size_t a, size_t b) {
        return tables.exports[a].serial_size > tables.exports[b].serial_size;
    });
    std::unordered_map<std::string, size_t> class_counts;
    for (const ObjectExport& object : tables.exports) {
        ++class_counts[tables.ClassName(object)];
    }
    LOG_INFO(Frontend, "[ShaderScan] Unreal links: classes in the code package: {}",
             Top(class_counts, 30,
                 [](const std::string& c, size_t n) { return fmt::format("{} x{}", c, n); }));
    for (size_t rank = 0; rank < largest.size(); ++rank) {
        const size_t i = largest[rank];
        if (tables.exports.size() > 40 && rank >= 20 && codes_in[i].empty()) {
            continue;
        }
        const ObjectExport& object = tables.exports[i];
        const std::string held =
            codes_in[i].empty()
                ? std::string{}
                : fmt::format(", code headers #{} to #{} ({}), first at +0x{:x}, last at +0x{:x}",
                              codes_in[i].front(), codes_in[i].back(), codes_in[i].size(),
                              s.codes[codes_in[i].front()] - object.serial_offset,
                              s.codes[codes_in[i].back()] - object.serial_offset);
        LOG_INFO(Frontend,
                 "[ShaderScan] Unreal links: export #{} {} '{}' in '{}' at 0x{:x}, 0x{:x} "
                 "bytes{}",
                 i + 1, tables.ClassName(object), tables.ObjectName(static_cast<s32>(i + 1)),
                 tables.ObjectName(object.outer), object.serial_offset, object.serial_size,
                 held);
    }
    for (size_t first = 0; first < std::min(tables.names.size(), NAMES_LOGGED); first += 150) {
        std::string line;
        const size_t last = std::min({first + 150, tables.names.size(), NAMES_LOGGED});
        for (size_t i = first; i < last; ++i) {
            line += i == first ? "" : ",";
            line += tables.names[i];
        }
        LOG_INFO(Frontend, "[ShaderScan] Unreal links: names {}-{}: {}", first, last - 1, line);
    }
    u64 first_data = summary->name_offset;
    for (const size_t i : by_offset) {
        if (tables.exports[i].serial_size > 0) {
            first_data = tables.exports[i].serial_offset;
            break;
        }
    }
    for (const auto& [what, at, length] : std::initializer_list<std::tuple<const char*, u64, u64>>{
             {"at the start of the data", first_data, 0x200},
             {"before the first code", s.first_code - std::min<u64>(s.first_code, 0x200), 0x240},
             {"after the code", s.code_end, 0x200},
         }) {
        LOG_INFO(Frontend, "[ShaderScan] Unreal links: bytes {} (0x{:x}): {}", what, at,
                 Hex(data, at, length));
    }
    for (size_t rank = 0, shown = 0; rank < largest.size() && shown < 6; ++rank) {
        const ObjectExport& object = tables.exports[largest[rank]];
        if (object.serial_size == 0) {
            continue;
        }
        ++shown;
        LOG_INFO(Frontend, "[ShaderScan] Unreal links: export #{} starts: {}", largest[rank] + 1,
                 Hex(data, object.serial_offset, std::min<u64>(object.serial_size, 0x180)));
        if (object.serial_size > 0x180) {
            LOG_INFO(Frontend, "[ShaderScan] Unreal links: export #{} ends: {}",
                     largest[rank] + 1,
                     Hex(data, u64{object.serial_offset} + object.serial_size - 0x100, 0x100));
        }
    }

    // References to the code by offset or index, outside the code itself.
    std::array<std::vector<std::vector<u64>>, KIND_COUNT> where;
    std::array<std::vector<u32>, KIND_COUNT> hit_counts;
    for (u32 kind = 0; kind < KIND_COUNT; ++kind) {
        where[kind].resize(count);
        hit_counts[kind].assign(count, 0);
    }
    const auto record = [&](u64 pos, u32 kind, u32 code) {
        if (kind >= KIND_COUNT) {
            return;
        }
        hit_counts[kind][code] += hit_counts[kind][code] < 0xFFFFFFFF ? 1 : 0;
        const size_t cap = IsIndexKind(kind) ? MAX_INDEX_HITS : MAX_OFFSET_HITS;
        if (where[kind][code].size() < cap) {
            where[kind][code].push_back(pos);
        }
    };
    s.Prepare();
    s.FindRefs(data, 0, s.first_code, 1, true, record);
    s.FindRefs(data, s.code_end, image.size(), 1, true, record);

    // A table lists the shaders in order: the entry of the next shader is a fixed step on.
    struct Table {
        size_t stride = 0;
        size_t entries = 0;
    };
    std::array<Table, KIND_COUNT> tables_found{};
    const auto measure = [&](u32 kind) {
        size_t found = 0;
        u64 total = 0;
        std::unordered_map<size_t, size_t> strides;
        for (u32 code = 0; code < count; ++code) {
            found += hit_counts[kind][code] > 0 ? 1 : 0;
            total += hit_counts[kind][code];
            if (code + 1 >= count) {
                continue;
            }
            const u32 next = s.Expected(kind, code + 1);
            for (const u64 pos : where[kind][code]) {
                for (size_t stride = 1; stride <= MAX_STRIDE && pos + stride + 4 <= image.size();
                     ++stride) {
                    if (Load32(image.data() + pos + stride) == next) {
                        ++strides[stride];
                        break;
                    }
                }
            }
        }
        std::unordered_map<s64, size_t> distances;
        size_t pairs_close = 0;
        for (const auto& [vertex, fragment] : s.played_list) {
            bool close = false;
            for (const u64 v : where[kind][vertex]) {
                for (const u64 f : where[kind][fragment]) {
                    const s64 distance = static_cast<s64>(f) - static_cast<s64>(v);
                    if (distance <= PAIR_WINDOW && distance >= -PAIR_WINDOW) {
                        ++distances[distance];
                        close = true;
                    }
                }
            }
            pairs_close += close ? 1 : 0;
        }
        LOG_INFO(Frontend,
                 "[ShaderScan] Unreal links: by {} in the package: {} of {} shaders{} found, {} "
                 "times; next shader's entry at {}; played pairs within 1 KiB: {} of {}, at {}",
                 s.KindName(kind), found, IsIndexKind(kind) ? study_codes : count,
                 IsIndexKind(kind) ? " (played and control ones)" : "", total,
                 Top(strides, 6,
                     [](size_t d, size_t n) { return fmt::format("+0x{:x} x{}", d, n); }),
                 pairs_close, s.played_list.size(),
                 Top(distances, 8, [](s64 d, size_t n) { return fmt::format("{:+d} x{}", d, n); }));
        const auto best = std::max_element(
            strides.begin(), strides.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        if (best == strides.end()) {
            return;
        }
        tables_found[kind] = {best->first, best->second};
        size_t shown = 0;
        for (u32 code = 0; code + 1 < count && shown < 2; ++code) {
            for (const u64 pos : where[kind][code]) {
                if (pos + best->first + 4 <= image.size() &&
                    Load32(image.data() + pos + best->first) == s.Expected(kind, code + 1)) {
                    LOG_INFO(Frontend,
                             "[ShaderScan] Unreal links:   #{} at 0x{:x}, bytes from 0x{:x}: {}",
                             code, pos, pos - std::min<u64>(pos, 0x40),
                             Hex(data, pos - std::min<u64>(pos, 0x40),
                                 std::min<u64>(best->first * 3 + 0x40, 0x100)));
                    ++shown;
                    break;
                }
            }
        }
    };
    const auto is_table = [&](u32 kind) {
        const size_t base = IsIndexKind(kind) ? study_codes : count;
        return tables_found[kind].entries >= std::max<size_t>(16, base / 10);
    };
    for (u32 kind = 0; kind < OffsetFromBase; ++kind) {
        measure(kind);
    }

    // A table by the offsets above may list only the first part of the code.
    const size_t offset_entries = std::max({tables_found[PackageOffset].entries,
                                            tables_found[OffsetFromFirst].entries,
                                            tables_found[OffsetInObject].entries});
    if (offset_entries < count / 2 && !stopped()) {
        // No table of most shaders by the offsets above. Look for offsets from another base:
        // steps between the entries of three shaders in a row, or between a played pair's
        // entries, that equal the steps between their code.
        std::unordered_map<u64, std::vector<u32>> runs;
        u32 max_gap = 0;
        for (u32 code = 0; code + 3 < count; ++code) {
            const u32 a = static_cast<u32>(s.codes[code + 1] - s.codes[code]);
            const u32 b = static_cast<u32>(s.codes[code + 2] - s.codes[code + 1]);
            const u32 c = static_cast<u32>(s.codes[code + 3] - s.codes[code + 2]);
            max_gap = std::max(max_gap, a);
            auto& list = runs[(u64{a} << 40) ^ (u64{b} << 20) ^ c];
            if (list.size() < 5) {
                list.push_back(code);
            }
        }
        std::unordered_map<u32, std::vector<u32>> pair_steps;
        for (u32 i = 0; i < s.played_list.size(); ++i) {
            const auto& [vertex, fragment] = s.played_list[i];
            auto& list = pair_steps[static_cast<u32>(s.codes[fragment] - s.codes[vertex])];
            if (list.size() < 4) {
                list.push_back(i);
            }
        }
        std::unordered_map<u64, size_t> run_bases;
        std::unordered_map<u64, size_t> pair_bases;
        const auto key = [](u64 step, s64 base) {
            return (step << 48) ^ static_cast<u64>(base & 0xFFFFFFFFFFFFLL);
        };
        const auto base_of = [](u64 k) {
            s64 base = static_cast<s64>(k & 0xFFFFFFFFFFFFULL);
            if ((base & 0x800000000000LL) != 0) {
                base -= 0x1000000000000LL;
            }
            return base;
        };
        const auto search = [&](u64 begin, u64 end) {
            for (u64 pos = begin; pos + 4 * 64 + 4 <= end; ++pos) {
                if ((pos & 0xFFFFF) == 0 && stopped()) {
                    return;
                }
                const u32 a = Load32(data.data() + pos);
                for (u64 step = 4; step <= 64; step += 4) {
                    const u32 b = Load32(data.data() + pos + step);
                    const u32 diff = b - a;
                    if (diff == 0 || (diff & 0xFF) != 0) {
                        continue;
                    }
                    if (diff <= max_gap) {
                        const u32 c = Load32(data.data() + pos + step * 2) - b;
                        const u32 d = Load32(data.data() + pos + step * 3) -
                                      Load32(data.data() + pos + step * 2);
                        const auto it = runs.find((u64{diff} << 40) ^ (u64{c} << 20) ^ d);
                        if (it != runs.end() && it->second.size() < 5) {
                            for (const u32 code : it->second) {
                                ++run_bases[key(step, static_cast<s64>(a) -
                                                          static_cast<s64>(s.codes[code]))];
                            }
                        }
                    }
                    if (const auto it = pair_steps.find(diff); it != pair_steps.end()) {
                        for (const u32 i : it->second) {
                            ++pair_bases[key(step, static_cast<s64>(a) -
                                                       static_cast<s64>(
                                                           s.codes[s.played_list[i].first]))];
                        }
                    }
                }
            }
        };
        search(0, s.first_code);
        search(s.code_end, image.size());
        const auto format = [&](u64 k, size_t n) {
            return fmt::format("step {} base {:+#x} x{}", k >> 48, base_of(k), n);
        };
        LOG_INFO(Frontend,
                 "[ShaderScan] Unreal links: offsets from another base: three shaders in a row "
                 "{}; played pairs {}",
                 Top(run_bases, 6, format), Top(pair_bases, 6, format));
        // Offsets may count from the start of one of several parts of the code: the bases
        // the steps of three shaders in a row agree on, summed over the steps between entries.
        std::unordered_map<s64, size_t> by_base;
        for (const auto& [k, n] : run_bases) {
            by_base[base_of(k)] += n;
        }
        // A part starts at a code header, or before the first.
        std::vector<std::pair<size_t, s64>> bases;
        for (const auto& [b, n] : by_base) {
            if (b <= 0 && (static_cast<u64>(-b) <= s.first_code || s.CodeAt(static_cast<u64>(-b)))) {
                bases.emplace_back(n, b);
            }
        }
        std::sort(bases.rbegin(), bases.rend());
        std::vector<std::pair<u64, size_t>> parts;
        const size_t most = bases.empty() ? 0 : bases.front().first;
        for (const auto& [n, b] : bases) {
            if (parts.size() < MAX_PARTS && n >= 16 && n >= most / 32) {
                parts.emplace_back(static_cast<u64>(-b), n);
            }
        }
        if (parts.empty()) {
            // The base of played pairs, when far more steps agree on it than on any other.
            size_t best = 0;
            size_t second = 0;
            u64 best_key = 0;
            for (const auto& [k, n] : pair_bases) {
                if (n > best) {
                    second = best;
                    best = n;
                    best_key = k;
                } else {
                    second = std::max(second, n);
                }
            }
            if (best >= 16 && best >= second * 5 && base_of(best_key) <= 0) {
                parts.emplace_back(static_cast<u64>(-base_of(best_key)), best);
            }
        }
        if (!parts.empty()) {
            std::sort(parts.begin(), parts.end());
            // The first part may count from the first code header, which the offsets from the
            // first code look for already.
            if (parts.front().first > s.first_code) {
                parts.insert(parts.begin(), {s.first_code, 0});
            }
            for (const auto& [part_start, n] : parts) {
                s.part_starts.push_back(part_start);
            }
            s.part_of.assign(count, NO_PART);
            std::vector<size_t> in_part(parts.size(), 0);
            for (u32 code = 0; code < count; ++code) {
                const auto it =
                    std::upper_bound(s.part_starts.begin(), s.part_starts.end(), s.codes[code]);
                if (it != s.part_starts.begin()) {
                    const size_t part = static_cast<size_t>(it - s.part_starts.begin()) - 1;
                    s.part_of[code] = static_cast<u8>(part);
                    ++in_part[part];
                }
            }
            std::string listed;
            for (size_t part = 0; part < parts.size(); ++part) {
                const auto at = s.CodeAt(parts[part].first);
                listed += fmt::format("{}0x{:x} ({}, {} code headers, {} steps agree)",
                                      listed.empty() ? "" : ", ", parts[part].first,
                                      at ? fmt::format("code header #{}", *at)
                                         : std::string{"not a code header"},
                                      in_part[part], parts[part].second);
            }
            LOG_INFO(Frontend,
                     "[ShaderScan] Unreal links: offsets may count from the start of {} parts of "
                     "the code: {}",
                     parts.size(), listed);
            s.have_base = true;
            s.Prepare();
            const auto record_base = [&](u64 pos, u32 kind, u32 code) {
                if (kind == OffsetFromBase) {
                    record(pos, kind, code);
                }
            };
            s.FindRefs(data, 0, s.first_code, 1, true, record_base);
            s.FindRefs(data, s.code_end, image.size(), 1, true, record_base);
            measure(OffsetFromBase);
            // Without a table that counts from them, the parts only add chance matches, and
            // with many parts those make the search of the other packages slow.
            if (!is_table(OffsetFromBase)) {
                s.have_base = false;
                s.Prepare();
                LOG_INFO(Frontend,
                         "[ShaderScan] Unreal links: no table counts from the start of these "
                         "parts ({} entries found), so references are not looked for by them",
                         tables_found[OffsetFromBase].entries);
            }
        }
    }

    // Table entries: offsets before index kinds, which match small numbers by chance more.
    u32 table_kind = KIND_COUNT;
    const auto pick = [&](std::initializer_list<u32> kinds) {
        for (const u32 kind : kinds) {
            if (is_table(kind) && (table_kind == KIND_COUNT || tables_found[kind].entries >
                                                                   tables_found[table_kind].entries)) {
                table_kind = kind;
            }
        }
    };
    pick({PackageOffset, OffsetFromFirst, OffsetInObject, OffsetFromBase});
    if (table_kind == KIND_COUNT) {
        pick({Index, IndexFromOne, Slot});
    }

    // Ids next to each table entry, to look for in the other packages.
    s.entries.assign(count, 0);
    if (table_kind != KIND_COUNT) {
        const size_t stride = tables_found[table_kind].stride;
        size_t entries = 0;
        // An entry whose neighbours both hold the next and the one before comes first: offsets
        // in different parts of the code can be the same number for two shaders.
        for (const bool both : {true, false}) {
            for (u32 code = 0; code < count; ++code) {
                if (s.entries[code] != 0) {
                    continue;
                }
                for (const u64 pos : where[table_kind][code]) {
                    const bool before =
                        code + 1 < count && pos + stride + 4 <= image.size() &&
                        Load32(image.data() + pos + stride) == s.Expected(table_kind, code + 1);
                    const bool after =
                        code > 0 && pos >= stride &&
                        Load32(image.data() + pos - stride) == s.Expected(table_kind, code - 1);
                    if (both ? before && after : before || after) {
                        s.entries[code] = pos;
                        ++entries;
                        break;
                    }
                }
            }
        }
        // Each place near the entry that holds a different value for every shader may be an
        // id. Those of the entries around it are too, so the other packages tell which places
        // pair played shaders.
        const s32 range = static_cast<s32>(std::min<size_t>(stride + 8, ID_RANGE));
        struct EntryPlace {
            s32 at;
            double distinct;
            double zeros;
        };
        std::vector<EntryPlace> candidates;
        for (s32 at = -range; at <= range; ++at) {
            if (at > -8 && at < 4) {
                continue; // Overlaps the offset or index itself.
            }
            std::unordered_set<u64> values;
            size_t total = 0;
            size_t zeros = 0;
            for (u32 code = 0; code < count; ++code) {
                const u64 entry = s.entries[code];
                if (entry == 0 || static_cast<s64>(entry) + at < 0 ||
                    entry + at + 8 > image.size()) {
                    continue;
                }
                const u64 value = Load64(image.data() + entry + at);
                values.insert(value);
                ++total;
                for (u32 byte = 0; byte < 8; ++byte) {
                    zeros += ((value >> (byte * 8)) & 0xFF) == 0 ? 1 : 0;
                }
            }
            if (total < 16) {
                continue;
            }
            const double distinct = static_cast<double>(values.size()) / total;
            const double mean_zeros = static_cast<double>(zeros) / total;
            // Copies of one shader may share an id, as for the places by the code headers.
            if (distinct >= 0.5 && mean_zeros <= 3.0) {
                candidates.push_back({at, distinct, mean_zeros});
            }
        }
        // The game's data is 4-byte aligned, so ids at a multiple of 4 from the entry come
        // first, nearest to the entry first.
        std::sort(candidates.begin(), candidates.end(),
                  [](const EntryPlace& a, const EntryPlace& b) {
                      return std::make_tuple(a.at % 4 != 0, std::abs(a.at), a.at) <
                             std::make_tuple(b.at % 4 != 0, std::abs(b.at), b.at);
                  });
        // Places that overlap read the same id shifted, which finds the same links again, so
        // keep only places at least 8 bytes apart.
        std::vector<EntryPlace> kept;
        for (const EntryPlace& place : candidates) {
            if (kept.size() < MAX_ID_WINDOWS &&
                std::ranges::none_of(kept, [&](const EntryPlace& other) {
                    return std::abs(other.at - place.at) < 8;
                })) {
                kept.push_back(place);
            }
        }
        candidates = std::move(kept);
        std::sort(candidates.begin(), candidates.end(),
                  [](const EntryPlace& a, const EntryPlace& b) { return a.at < b.at; });
        std::string chosen;
        for (const EntryPlace& place : candidates) {
            s.windows.push_back({place.at, IdSource::Entry});
            chosen += fmt::format("{}{:+d}", chosen.empty() ? "" : ",", place.at);
        }
        LOG_INFO(Frontend,
                 "[ShaderScan] Unreal links: table by {} every 0x{:x} bytes, {} entries; 8-byte "
                 "values that differ per shader at {}",
                 s.KindName(table_kind), stride, entries, chosen.empty() ? "none" : chosen);
    }

    // The places by the code headers, and in the shader records, where each holds a different
    // 8-byte value may be ids that the package's other data and the other packages list
    // shaders by.
    std::vector<s32> by_code;
    if (!stopped()) {
        by_code = ChooseIdWindows(data, s.code_window, MAX_CODE_ID_WINDOWS, false);
        StudyCodeIds(data, s.code_window, s.first_code, s.code_end, by_code, pair_copies,
                     controls, s.played_mask, stopped);
    }
    std::vector<s32> in_records;
    if (s.record_window && !stopped()) {
        in_records = ChooseIdWindows(data, *s.record_window, RECORD_ID_WINDOWS, true);
        StudyCodeIds(data, *s.record_window, s.first_code, s.code_end, in_records, pair_copies,
                     controls, s.played_mask, stopped);
    }
    // The ids searched for: near the table entries; else (no table, or none near its entries)
    // in the records; else by the code headers.
    if (s.windows.empty()) {
        for (const s32 at : in_records) {
            s.windows.push_back({at, IdSource::Record});
        }
        if (in_records.empty()) {
            for (const s32 at : by_code) {
                s.windows.push_back({at, IdSource::CodeHeader});
            }
        }
    }
    // Only the played and control shaders' ids are searched for, to tell which places pair
    // them.
    s.id_filter.assign((size_t{1} << ID_FILTER_BITS) / 64, 0);
    s.all_id_filter.assign((size_t{1} << ID_FILTER_BITS) / 64, 0);
    for (u32 w = 0; w < s.windows.size(); ++w) {
        // A value two different shaders have there is not an id; copies of one shader may
        // share it.
        std::unordered_map<u64, u32> in_window;
        for (u32 code = 0; code < count; ++code) {
            if (const auto pos = s.IdPosition(code, s.windows[w])) {
                s.own_id_places.push_back(*pos);
            }
            const std::optional<u64> value = s.IdOf(data, code, s.windows[w]);
            if (!value || *value == 0) {
                continue;
            }
            const auto [it, inserted] = in_window.try_emplace(*value, code);
            if (!inserted && it->second != count && hash_of[it->second] != hash_of[code]) {
                it->second = count;
            }
        }
        for (u32 code = 0; code < count; ++code) {
            const std::optional<u64> value = s.IdOf(data, code, s.windows[w]);
            if (!value) {
                continue;
            }
            const auto it = in_window.find(*value);
            if (it == in_window.end() || it->second == count) {
                continue;
            }
            const u64 bit = MixId(*value);
            const u64 bit2 = MixId2(*value);
            s.all_ids[*value].emplace_back(code, w);
            s.all_id_filter[bit >> 6] |= u64{1} << (bit & 63);
            s.all_id_filter[bit2 >> 6] |= u64{1} << (bit2 & 63);
            if (s.study_mask[code]) {
                s.ids[*value].emplace_back(code, w);
                s.id_filter[bit >> 6] |= u64{1} << (bit & 63);
            }
        }
    }
    std::sort(s.own_id_places.begin(), s.own_id_places.end());

    // Played pairs whose references sit close together elsewhere in this package.
    s.code_package = std::string(name);
    s.studied = true;
    std::vector<Hit> hits;
    const auto collect = [&](u64 pos, u32 kind, u32 code) {
        if (hits.size() < MAX_PIECE_HITS) {
            hits.push_back({pos, kind, code});
        }
    };
    s.FindRefs(data, 0, s.first_code, MIN_FOREIGN_INDEX, true, collect);
    s.FindRefs(data, s.code_end, image.size(), MIN_FOREIGN_INDEX, true, collect);
    s.Pair(s.finds, hits, data, fmt::format("{} (the code package)", name), 0, 0, true);
    for (const Hit& hit : hits) {
        ++s.hits_by_kind[hit.kind];
        ++s.files_by_kind[hit.kind][fmt::format("{} (the code package)", name)];
    }
    hits = {};
    if (!pair_copies.empty() && !stopped()) {
        learned_sets.push_back(s.LearnFromRefs(data, pair_copies, hash_of, stage_of, stopped));
    }
    // All pairs learned, in the order of the ways, each once.
    std::set<std::pair<u64, u64>> kept;
    for (const auto& set : learned_sets) {
        for (const auto& pair : set) {
            if (s.learned.size() < MAX_LEARNED_PAIRS && kept.insert(pair).second) {
                s.learned.push_back(pair);
            }
        }
    }
}

void LinkSurvey::SearchPiece(std::string_view name, std::span<const u8> piece, u64 offset) {
    Impl& s = *impl;
    if (!s.studied || name.starts_with(s.code_package)) {
        return;
    }
    const u64 begin = offset > 0 ? std::min<u64>(s.overlap, piece.size()) : 0;
    // Only the objects that may name shaders, when the package's tables tell where they are.
    std::vector<std::pair<u64, u64>> parts;
    if (s.ranges_known && name.starts_with(s.current_package)) {
        for (const auto& [from, to] : s.ranges) {
            const u64 part_begin = std::max(from, offset + begin);
            const u64 part_end = std::min(to, offset + piece.size());
            if (part_begin < part_end) {
                parts.emplace_back(part_begin - offset, part_end - offset);
            }
        }
    } else {
        parts.emplace_back(begin, piece.size());
    }
    u64 part_bytes = 0;
    for (const auto& [from, to] : parts) {
        part_bytes += to - from;
    }
    s.other_class_bytes += piece.size() - begin - part_bytes;
    // An id can be held by an object of any class, so the bytes outside the shader and
    // material objects are searched for the shaders' ids alone, which is quick. The ids of
    // all shaders are looked for, to learn the pairs of a link found there.
    if (!s.all_ids.empty() && s.ranges_known && name.starts_with(s.current_package)) {
        std::vector<std::pair<u64, u64>> rest;
        u64 at = begin;
        for (const auto& [from, to] : parts) {
            if (from > at) {
                rest.emplace_back(at, from);
            }
            at = std::max(at, to);
        }
        if (at < piece.size()) {
            rest.emplace_back(at, piece.size());
        }
        u64 rest_bytes = 0;
        for (const auto& [from, to] : rest) {
            rest_bytes += to - from;
        }
        if (s.id_only_time >= ID_ONLY_SEARCH_TIME) {
            s.id_only_skipped_bytes += rest_bytes;
        } else {
            const auto id_start = std::chrono::steady_clock::now();
            s.id_only_bytes += rest_bytes;
            std::vector<Hit> id_hits;
            const auto collect_id = [&](u64 pos, u32 kind, u32 code) {
                if (id_hits.size() < MAX_PIECE_HITS) {
                    id_hits.push_back({pos, kind, code});
                }
            };
            for (const auto& [from, to] : rest) {
                s.FindIds(piece, from, to, collect_id);
            }
            s.KeepIdCandidates(s.id_store, id_hits, begin);
            for (const Hit& hit : id_hits) {
                const auto [class_name, holder] = s.AnyObjectAt(offset + hit.pos);
                ++s.all_id_classes[class_name.empty() ? std::string{"none (between objects)"}
                                                      : class_name];
                // A few from each class, since one list holds many.
                if (s.id_examples.size() < ID_EXAMPLES &&
                    s.id_example_classes[class_name]++ < ID_EXAMPLES_PER_CLASS) {
                    s.id_examples.push_back(fmt::format(
                        "{} of shader #{} at 0x{:x} in {}, in {}: bytes from 0x{:x}: {}",
                        s.KindName(hit.kind), hit.code, offset + hit.pos, name,
                        holder.empty() ? std::string{"no object"} : holder,
                        offset + (hit.pos >= 0x40 ? hit.pos - 0x40 : 0),
                        Hex(piece, hit.pos >= 0x40 ? hit.pos - 0x40 : 0, 0x88)));
                }
            }
            // The played and control shaders' ids, which tell a link from chance.
            std::erase_if(id_hits, [&](const Hit& hit) { return !s.study_mask[hit.code]; });
            const std::string file{name};
            for (const Hit& hit : id_hits) {
                ++s.hits_by_kind[hit.kind];
                ++s.files_by_kind[hit.kind][file];
            }
            s.Pair(s.id_finds, id_hits, piece, name, offset, begin, false);
            s.id_only_time += std::chrono::steady_clock::now() - id_start;
        }
    }
    if (parts.empty()) {
        return;
    }
    if (s.search_time >= SEARCH_TIME) {
        ++s.skipped_pieces;
        s.skipped_bytes += part_bytes;
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    ++s.pieces;
    s.bytes += part_bytes;
    std::vector<Hit> hits;
    bool capped = false;
    const auto collect = [&](u64 pos, u32 kind, u32 code) {
        if (hits.size() < MAX_PIECE_HITS) {
            hits.push_back({pos, kind, code});
        } else {
            capped = true;
        }
    };
    // Each piece repeats the end of the one before, where hits were counted already. Read a
    // little of it, for pairs whose other reference is there.
    for (const auto& [from, to] : parts) {
        const u64 lead = from == begin ? std::min<u64>(begin, PAIR_WINDOW) : 0;
        s.FindRefs(piece, from - lead, to, MIN_FOREIGN_INDEX, false, collect);
    }
    s.capped_pieces += capped ? 1 : 0;
    const std::string file{name};
    for (const Hit& hit : hits) {
        if (hit.pos >= begin) {
            ++s.hits_by_kind[hit.kind];
            ++s.files_by_kind[hit.kind][file];
            if (hit.kind >= KIND_COUNT) {
                const std::string class_name = s.ClassAt(name, offset + hit.pos);
                ++s.id_classes[class_name.empty() ? std::string{"unknown"} : class_name];
            }
        }
    }
    s.Pair(s.finds, hits, piece, name, offset, begin, false);
    s.search_time += std::chrono::steady_clock::now() - start;
}

void LinkSurvey::Report() const {
    const Impl& s = *impl;
    LOG_INFO(Frontend,
             "[ShaderScan] Unreal links: {} packages read ({} unreadable, {} with tables that "
             "stopped early); versions: {}",
             s.packages, s.unreadable, s.table_errors,
             Top(s.versions, 6,
                 [](const std::string& v, size_t n) { return fmt::format("{} x{}", v, n); }));
    for (const std::string& example : s.unreadable_examples) {
        LOG_INFO(Frontend, "[ShaderScan] Unreal links: unreadable: {}", example);
    }
    for (const std::string& example : s.table_error_examples) {
        LOG_INFO(Frontend, "[ShaderScan] Unreal links: tables stopped: {}", example);
    }
    std::unordered_map<std::string, size_t> counts;
    for (const auto& [name, stats] : s.classes) {
        counts[name] = stats.count;
    }
    LOG_INFO(Frontend, "[ShaderScan] Unreal links: classes in all packages: {}",
             Top(counts, 50, [&](const std::string& name, size_t n) {
                 return fmt::format("{} x{} ({} MiB)", name, n, s.classes.at(name).bytes >> 20);
             }));
    // All classes that may hold shaders or lists of them, however few objects they have.
    std::unordered_map<std::string, size_t> shader_like;
    for (const auto& [name, stats] : s.classes) {
        const std::string lower = Lower(name);
        if (lower.find("shader") != std::string::npos || lower.find("cache") != std::string::npos ||
            lower.find("pipeline") != std::string::npos) {
            shader_like[name] = stats.count;
        }
    }
    LOG_INFO(Frontend,
             "[ShaderScan] Unreal links: classes named like shaders, caches or pipelines: {}",
             Top(shader_like, 100, [&](const std::string& name, size_t n) {
                 return fmt::format("{} x{} ({} KiB)", name, n, s.classes.at(name).bytes >> 10);
             }));
    size_t shown = 0;
    std::vector<std::string> names;
    for (const auto& [name, stats] : s.classes) {
        names.push_back(name);
    }
    // Classes named like shaders or caches first, since materials are many.
    const auto rank = [](const std::string& name) {
        const std::string lower = Lower(name);
        if (lower.find("shader") != std::string::npos || lower.find("cache") != std::string::npos) {
            return 0;
        }
        return lower.find("material") != std::string::npos ? 1 : 2;
    };
    std::sort(names.begin(), names.end(), [&](const std::string& a, const std::string& b) {
        return std::make_pair(rank(a), a) < std::make_pair(rank(b), b);
    });
    for (const std::string& name : names) {
        if (rank(name) == 2) {
            continue;
        }
        if (shown++ >= 20) {
            break;
        }
        for (const std::string& example : s.classes.at(name).examples) {
            LOG_INFO(Frontend, "[ShaderScan] Unreal links: {} {}", name, example);
        }
    }
    if (!s.studied) {
        return;
    }
    LOG_INFO(Frontend,
             "[ShaderScan] Unreal links: searched {} MiB of shader and material objects in {} "
             "pieces of other packages in {} ms, {} with too many hits to keep all; {} pieces "
             "({} MiB) left out after {} s; {} MiB of objects of other classes not searched",
             s.bytes >> 20, s.pieces,
             std::chrono::duration_cast<std::chrono::milliseconds>(s.search_time).count(),
             s.capped_pieces, s.skipped_pieces, s.skipped_bytes >> 20, SEARCH_TIME.count(),
             s.other_class_bytes >> 20);
    if (!s.all_ids.empty()) {
        LOG_INFO(Frontend,
                 "[ShaderScan] Unreal links: searched {} MiB of the other packages outside "
                 "shader and material objects for the shaders' ids alone in {} ms; {} MiB left "
                 "out after {} s",
                 s.id_only_bytes >> 20,
                 std::chrono::duration_cast<std::chrono::milliseconds>(s.id_only_time).count(),
                 s.id_only_skipped_bytes >> 20, ID_ONLY_SEARCH_TIME.count());
        size_t buckets = 0;
        size_t full = 0;
        for (const auto& [key, at] : s.id_store.buckets) {
            buckets += at.pairs.empty() ? 0 : 1;
            full += at.full ? 1 : 0;
        }
        LOG_INFO(Frontend,
                 "[ShaderScan] Unreal links: ids of any shader found there: {} of vertex, {} of "
                 "fragment and {} of other shaders; {} vertex and fragment shader pairs of them "
                 "within 0x{:x} bytes kept, at {} kinds and distances ({} with more than kept){}; "
                 "by the class of the object holding them: {}",
                 s.id_store.hits[static_cast<size_t>(Stage::Vertex)],
                 s.id_store.hits[static_cast<size_t>(Stage::Fragment)],
                 s.id_store.hits[static_cast<size_t>(Stage::Other)], s.id_store.count, PAIR_WINDOW,
                 buckets, full, s.id_store.capped ? ", more not kept" : "",
                 Top(s.all_id_classes, 12,
                     [](const std::string& c, size_t n) { return fmt::format("{} x{}", c, n); }));
        for (const std::string& example : s.id_examples) {
            LOG_INFO(Frontend, "[ShaderScan] Unreal links: id found: {}", example);
        }
    }
    std::vector<u32> kinds;
    for (const auto& [kind, n] : s.hits_by_kind) {
        kinds.push_back(kind);
    }
    std::sort(kinds.begin(), kinds.end());
    std::string by_id;
    for (const u32 kind : kinds) {
        if (kind >= KIND_COUNT) {
            by_id += fmt::format("{}{} x{}", by_id.empty() ? "" : ", ", s.KindName(kind),
                                 s.hits_by_kind.at(kind));
            continue;
        }
        LOG_INFO(Frontend, "[ShaderScan] Unreal links: references by {}: {}, most in {}",
                 s.KindName(kind), s.hits_by_kind.at(kind),
                 Top(s.files_by_kind.at(kind), 6, [](const std::string& file, size_t n) {
                     return fmt::format("{} x{}", file, n);
                 }));
    }
    if (!by_id.empty()) {
        LOG_INFO(Frontend, "[ShaderScan] Unreal links: references by id, by place: {}", by_id);
        LOG_INFO(Frontend,
                 "[ShaderScan] Unreal links: ids found in other packages, by the class of the "
                 "object holding them: {}",
                 Top(s.id_classes, 12, [](const std::string& c, size_t n) {
                     return fmt::format("{} x{}", c, n);
                 }));
    }
    const auto log_finds = [&](const std::map<std::pair<u32, u32>, PairFinds>& all,
                               std::string_view where) {
        // A link shows as many pairs at one distance; chance meetings spread over all distances.
        std::vector<std::pair<size_t, std::pair<u32, u32>>> order;
        for (const auto& [kinds_pair, found] : all) {
            size_t peak = 0;
            for (const auto& [distance, n] : found.distances) {
                peak = std::max(peak, n);
            }
            order.emplace_back(peak, kinds_pair);
        }
        std::sort(order.rbegin(), order.rend());
        // Places that overlap the same id find the same pairs; those are shown once.
        std::set<std::pair<size_t, size_t>> shown_counts;
        size_t same = 0;
        for (size_t i = 0; i < order.size() && shown_counts.size() < 12; ++i) {
            const PairFinds& found = all.at(order[i].second);
            if (!shown_counts.emplace(order[i].first, found.pairs.size()).second) {
                ++same;
                continue;
            }
            size_t total = 0;
            for (const auto& [distance, n] : found.distances) {
                total += n;
            }
            // Different pairs at the distances most played pairs are at, against the controls.
            std::vector<std::pair<u32, s64>> by_pairs;
            for (const auto& [distance, n] : found.distinct_at[PlayedGroup]) {
                by_pairs.emplace_back(n, -distance);
            }
            std::sort(by_pairs.rbegin(), by_pairs.rend());
            std::string groups;
            for (size_t d = 0; d < std::min<size_t>(by_pairs.size(), 4); ++d) {
                const s64 distance = -by_pairs[d].second;
                const auto other = [&](u8 group) -> u32 {
                    const auto it = found.distinct_at[group].find(distance);
                    return it == found.distinct_at[group].end() ? 0 : it->second;
                };
                groups += fmt::format("{}{:+d}: {} played, {} shuffled, {} moved",
                                      groups.empty() ? "" : "; ", distance, by_pairs[d].first,
                                      other(ShuffledGroup), other(MovedGroup));
            }
            LOG_INFO(
                Frontend,
                "[ShaderScan] Unreal links: {}vertex by {}, fragment by {}: {} times at the most "
                "common distance ({}% of {}); {} of {} played pairs close together, at {}; "
                "different pairs of {} played, {} shuffled and {} moved at {}; in {}",
                where, s.KindName(order[i].second.first), s.KindName(order[i].second.second),
                order[i].first, total == 0 ? 0 : order[i].first * 100 / total, total,
                found.pairs.size(), s.played_list.size(),
                Top(found.distances, 8,
                    [](s64 d, size_t n) { return fmt::format("{:+d} x{}", d, n); }),
                s.group_sizes[PlayedGroup], s.group_sizes[ShuffledGroup], s.group_sizes[MovedGroup],
                groups.empty() ? std::string{"none"} : groups,
                Top(found.files, 4, [](const std::string& file, size_t n) {
                    return fmt::format("{} x{}", file, n);
                }));
            // Examples at the two most common distances.
            std::vector<std::pair<size_t, s64>> distances;
            for (const auto& [distance, n] : found.distances) {
                distances.emplace_back(n, -distance);
            }
            std::sort(distances.rbegin(), distances.rend());
            for (size_t d = 0; d < std::min<size_t>(distances.size(), 2); ++d) {
                const auto it = found.examples.find(-distances[d].second);
                if (it == found.examples.end()) {
                    continue;
                }
                const size_t shown_here = d == 0 ? EXAMPLES_PER_DISTANCE : 1;
                for (size_t e = 0; e < std::min(it->second.size(), shown_here); ++e) {
                    const PairExample& example = it->second[e];
                    LOG_INFO(Frontend, "[ShaderScan] Unreal links:   {}", example.text);
                    // Ids next to a reference that the code package holds next to the code would
                    // tie the two together without the index.
                    static constexpr std::array<const char*, 4> SIDES{
                        "before the vertex reference", "after the vertex reference",
                        "before the fragment reference", "after the fragment reference"};
                    std::string places;
                    for (size_t side = 0; side < 4; ++side) {
                        if (example.have[side]) {
                            places +=
                                fmt::format("{}16 bytes {} ({}): {}", places.empty() ? "" : "; ",
                                            SIDES[side], Hex(example.around[side], 0, 16),
                                            s.FindInCodeWindows(example.around[side]));
                        }
                    }
                    if (!places.empty()) {
                        LOG_INFO(Frontend, "[ShaderScan] Unreal links:     in the code package: {}",
                                 places);
                    }
                }
            }
        }
        if (same > 0) {
            LOG_INFO(
                Frontend,
                "[ShaderScan] Unreal links: {}{} more kinds of reference found the same numbers "
                "of pairs as one shown",
                where, same);
        }
    };
    log_finds(s.finds, "");
    log_finds(s.id_finds, "outside shader and material objects of other packages, by ids alone: ");
    log_finds(s.cache_finds, "in the shader cache, by ids: ");
}

const std::vector<std::pair<u64, u64>>& LinkSurvey::LearnedPairs() const {
    return impl->learned;
}

bool LinkSurvey::HoldsShaderCache(std::string_view name) const {
    return impl->cache_packages.contains(std::string(name));
}

void LinkSurvey::StudyShaderCache(std::string_view name, std::span<const u8> file,
                                  const std::function<bool()>& stopped) {
    Impl& s = *impl;
    const auto start = std::chrono::steady_clock::now();
    if (!s.studied) {
        LOG_INFO(Frontend,
                 "[ShaderScan] Unreal links: shader cache {} not read: the code package was not "
                 "read before it",
                 name);
        return;
    }
    std::optional<Summary> summary = ReadSummary(file);
    std::vector<u8> image;
    std::vector<ImageSpan> spans;
    size_t failed_blocks = 0;
    if (!summary) {
        // A package compressed whole keeps its header inside the compressed data.
        Stats stats;
        Unpack(file, size_t{64} << 20, 0, stats, [&](std::span<const u8> piece, u64) {
            image.insert(image.end(), piece.begin(), piece.end());
            return true;
        });
        summary = ReadSummary(image);
        if (!summary || !summary->chunks.empty()) {
            LOG_INFO(Frontend, "[ShaderScan] Unreal links: could not read the header of {}: {}",
                     name, Hex(file, 0, 128));
            return;
        }
    } else if (!UnpackImage(file, *summary, image, failed_blocks, spans)) {
        LOG_INFO(Frontend, "[ShaderScan] Unreal links: nothing in {} unpacked", name);
        return;
    }
    const Tables tables = ReadTables(image, 0, *summary);
    size_t read = 0;
    for (size_t i = 0; i < tables.exports.size() && read < MAX_CACHE_OBJECTS && !stopped(); ++i) {
        const ObjectExport& object = tables.exports[i];
        const std::string class_name = tables.ClassName(object);
        if (Lower(class_name).find("shadercache") == std::string::npos ||
            object.serial_size < 0x40 ||
            u64{object.serial_offset} + object.serial_size > image.size()) {
            continue;
        }
        ++read;
        s.StudyCacheObject(
            std::span<const u8>(image).subspan(object.serial_offset, object.serial_size),
            fmt::format("shader cache {} '{}' in {} (0x{:x} bytes)", class_name,
                        tables.ObjectName(static_cast<s32>(i + 1)), name, object.serial_size),
            tables, stopped);
    }
    LOG_INFO(Frontend,
             "[ShaderScan] Unreal links: read {} shader cache objects of {} ({} MiB unpacked, {} "
             "blocks did not unpack{}{}) in {} ms",
             read, name, image.size() >> 20, failed_blocks,
             tables.error.empty() ? "" : ", tables stopped at ", tables.error,
             std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - start)
                 .count());
}

const std::vector<std::pair<u64, u64>>& LinkSurvey::LearnedFromShaderCache() const {
    return impl->learned_cache;
}

const std::vector<std::pair<u64, u64>>& LinkSurvey::LearnFromOtherPackages() {
    Impl& s = *impl;
    if (s.studied && s.learned_elsewhere.empty() && !s.id_finds.empty()) {
        s.learned_elsewhere = s.LearnFromIds();
    }
    return s.learned_elsewhere;
}

} // namespace Preforged::Unreal
