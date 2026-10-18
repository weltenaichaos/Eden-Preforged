// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstring>
#include <map>
#include <optional>
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
constexpr size_t MAX_OFFSET_HITS = 16;
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
constexpr size_t NAMES_LOGGED = 4000;
constexpr size_t EXAMPLES = 3;

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
    OffsetFromBase,  ///< Offset from a base the package's own tables point to
    KIND_COUNT,
};

constexpr std::array<const char*, KIND_COUNT> KIND_NAMES{
    "package offset", "offset from the first code", "offset in its object", "index",
    "index from 1",   "256-byte slot",              "offset from another base",
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

/// Played pairs whose vertex and fragment shader references were found close together.
struct PairFinds {
    std::unordered_set<u64> pairs;
    std::unordered_map<s64, size_t> distances;
    std::unordered_map<std::string, size_t> files;
    /// The first pair found at each distance, for the first distances found.
    std::unordered_map<s64, std::string> examples;
};

constexpr size_t MAX_EXAMPLE_DISTANCES = 256;

struct Hit {
    u64 pos;
    u32 kind;
    u32 code;
};

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
    /// What offsets from another base add to the code's offset in the package.
    s64 base = 0;
    bool have_base = false;
    /// Per 256-byte slot of the package: the index + 1 of the code header in it.
    std::vector<u32> slots;
    std::vector<u8> played_mask;
    /// Played pairs as (vertex index << 32) | fragment index.
    std::unordered_set<u64> played_pairs;
    std::vector<std::pair<u32, u32>> played_list;
    /// Where the table entry of each shader is in the code package, 0 when none was found.
    std::vector<u64> entries;
    std::vector<s32> windows;
    std::vector<u64> id_filter;
    /// The played shaders' ids: the shader and the place near the entry, per value. The bytes
    /// of one entry can be the id of a shader at one place and of its neighbour at another.
    std::unordered_map<u64, std::vector<std::pair<u32, u32>>> ids;

    /// Bits set for the played pairs, to skip most pairs without a hash lookup.
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
    /// Bytes of other packages outside such objects, which are not searched.
    u64 other_class_bytes = 0;
    std::unordered_map<u32, size_t> hits_by_kind;
    std::unordered_map<u32, std::unordered_map<std::string, size_t>> files_by_kind;
    std::map<std::pair<u32, u32>, PairFinds> finds;

    std::string KindName(u32 kind) const {
        if (kind < KIND_COUNT) {
            return KIND_NAMES[kind];
        }
        return fmt::format("id at {:+d} from the entry", windows[kind - KIND_COUNT]);
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
            return static_cast<u32>(static_cast<s64>(codes[code]) + base);
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
            low_bytes[static_cast<u64>(residue + base) & 0xFF] = 1;
        }
        const size_t count = codes.size();
        index_bits.assign(count + 1, 0);
        for (size_t value = 1; value <= count; ++value) {
            index_bits[value] = static_cast<u8>((value < count && played_mask[value] ? 1 : 0) |
                                                (played_mask[value - 1] ? 2 : 0));
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
                if (have_base && static_cast<s64>(value) >= base) {
                    if (const auto code = CodeAt(static_cast<u64>(static_cast<s64>(value) - base))) {
                        emit(pos, OffsetFromBase, *code);
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
                    code && played_mask[*code]) {
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

    bool MayBePlayed(u64 key) const {
        const u64 bit = MixId(key ^ 0x5851F42D4C957F2DULL) >> (ID_FILTER_BITS - PAIR_FILTER_BITS);
        return ((pair_filter[bit >> 6] >> (bit & 63)) & 1) != 0 && played_pairs.contains(key);
    }

    /// Finds played pairs whose vertex and fragment references are close in `hits`, which are
    /// sorted by position. Pairs whose vertex reference is before `min_first` don't count, and
    /// with `own_entries` neither do hits at a shader's own table entry. Index references are
    /// many, since small numbers are everywhere, so two of them pair only when much closer.
    void Pair(const std::vector<Hit>& hits, std::span<const u8> data, std::string_view name,
              u64 offset, u64 min_first, bool own_entries) {
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
            if (first.pos < min_first || first.code == second.code || !played_mask[first.code] ||
                !MayBePlayed((u64{first.code} << 32) | second.code)) {
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
            PairFinds& found = finds[{first.kind, second.kind}];
            found.pairs.insert((u64{first.code} << 32) | second.code);
            ++found.distances[distance];
            ++found_here[{first.kind, second.kind}];
            if (found.examples.size() < MAX_EXAMPLE_DISTANCES &&
                !found.examples.contains(distance)) {
                const u64 from = std::min(first.pos, second.pos);
                const u64 to = std::max(first.pos, second.pos) + 8;
                const u64 start = from - std::min<u64>(from, 0x20);
                found.examples.emplace(distance, fmt::format(
                    "vertex #{} at 0x{:x}, fragment #{} at 0x{:x} in {}, bytes from 0x{:x}: {}",
                    first.code, offset + first.pos, second.code, offset + second.pos, name,
                    offset + start, Hex(data, start, std::min<u64>(to + 0x20 - start, 0x100))));
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
            finds[kinds].files[std::string(name)] += count;
        }
    }

    bool OwnEntry(const Hit& hit) const {
        const u64 entry = entries.empty() ? 0 : entries[hit.code];
        return entry != 0 && hit.pos + ID_RANGE >= entry && hit.pos <= entry + ID_RANGE;
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
    const Tables tables = ReadTables(first_piece, data_offset, *summary);
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
    for (const ObjectExport& object : tables.exports) {
        const std::string lower = Lower(tables.ClassName(object));
        if (object.serial_size > 0 && object.serial_offset >= data_offset &&
            (lower.find("shader") != std::string::npos ||
             lower.find("material") != std::string::npos)) {
            const u64 begin = object.serial_offset - data_offset;
            s.ranges.emplace_back(begin, begin + object.serial_size);
        }
    }
    std::sort(s.ranges.begin(), s.ranges.end());
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
    } else if (!UnpackImage(file, *summary, image, failed_blocks, spans)) {
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
    size_t misses = 0;
    for (const CodeHeader& header : headers) {
        const std::optional<u64> offset =
            whole ? std::optional<u64>{header.unpacked_offset}
                  : ImageOffset(spans, header.unpacked_offset);
        if (offset && *offset + 4 <= image.size() &&
            Load32(image.data() + *offset) == NVN_CODE_MAGIC) {
            hash_at.emplace(*offset, header.hash);
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
    s.pair_filter.assign((size_t{1} << PAIR_FILTER_BITS) / 64, 0);
    for (const u64 key : s.played_pairs) {
        const u64 bit =
            MixId(key ^ 0x5851F42D4C957F2DULL) >> (ID_FILTER_BITS - PAIR_FILTER_BITS);
        s.pair_filter[bit >> 6] |= u64{1} << (bit & 63);
    }
    size_t played_codes = 0;
    for (u32 code = 0; code < count; ++code) {
        played_codes += s.played_mask[code];
    }
    LOG_INFO(Frontend,
             "[ShaderScan] Unreal links: {} code headers in the package ({} with a shader the "
             "search took, {} not found where expected), first at 0x{:x}, code ends at 0x{:x}; "
             "offsets mod 256: {}; {} played pairs of {} shaders",
             count, hash_at.size(), misses, s.first_code, s.code_end,
             Top(residues, 4, [](u32 r, size_t n) { return fmt::format("0x{:02x}x{}", r, n); }),
             s.played_list.size(), played_codes);

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
                 s.KindName(kind), found, IsIndexKind(kind) ? played_codes : count,
                 IsIndexKind(kind) ? " (played ones)" : "", total,
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
        const size_t base = IsIndexKind(kind) ? played_codes : count;
        return tables_found[kind].entries >= std::max<size_t>(16, base / 10);
    };
    for (u32 kind = 0; kind < OffsetFromBase; ++kind) {
        measure(kind);
    }

    if (!is_table(PackageOffset) && !is_table(OffsetFromFirst) && !is_table(OffsetInObject) &&
        !stopped()) {
        // No table by the offsets above. Look for offsets from another base: steps between the
        // entries of three shaders in a row, or between a played pair's entries, that equal the
        // steps between their code.
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
        // Search the package once more by the base most steps agree on.
        // A base counts when far more steps agree on it than on any other.
        std::optional<s64> base;
        for (const auto* bases : {&run_bases, &pair_bases}) {
            size_t best = 0;
            size_t second = 0;
            u64 best_key = 0;
            for (const auto& [k, n] : *bases) {
                if (n > best) {
                    second = best;
                    best = n;
                    best_key = k;
                } else {
                    second = std::max(second, n);
                }
            }
            if (!base && best >= 16 && best >= second * 5) {
                base = base_of(best_key);
            }
        }
        if (base) {
            s.base = *base;
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
        for (u32 code = 0; code < count; ++code) {
            for (const u64 pos : where[table_kind][code]) {
                const bool before =
                    code + 1 < count && pos + stride + 4 <= image.size() &&
                    Load32(image.data() + pos + stride) == s.Expected(table_kind, code + 1);
                const bool after =
                    code > 0 && pos >= stride &&
                    Load32(image.data() + pos - stride) == s.Expected(table_kind, code - 1);
                if (before || after) {
                    s.entries[code] = pos;
                    ++entries;
                    break;
                }
            }
        }
        // Each place near the entry that holds a different value for every shader may be an
        // id. Those of the entries around it are too, so the other packages tell which places
        // pair played shaders.
        const s32 range = static_cast<s32>(std::min<size_t>(stride + 8, ID_RANGE));
        struct Window {
            s32 at;
            double distinct;
            double zeros;
        };
        std::vector<Window> candidates;
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
            if (distinct >= 0.98 && mean_zeros <= 3.0) {
                candidates.push_back({at, distinct, mean_zeros});
            }
        }
        std::sort(candidates.begin(), candidates.end(), [](const Window& a, const Window& b) {
            return std::abs(a.at) != std::abs(b.at) ? std::abs(a.at) < std::abs(b.at)
                                                    : a.at < b.at;
        });
        // Places that overlap read the same id shifted, which finds the same links again, so
        // keep only places at least 8 bytes apart, nearest to the entry first.
        std::vector<Window> kept;
        for (const Window& window : candidates) {
            if (kept.size() < MAX_ID_WINDOWS &&
                std::ranges::none_of(kept, [&](const Window& other) {
                    return std::abs(other.at - window.at) < 8;
                })) {
                kept.push_back(window);
            }
        }
        candidates = std::move(kept);
        std::sort(candidates.begin(), candidates.end(),
                  [](const Window& a, const Window& b) { return a.at < b.at; });
        std::string chosen;
        for (const Window& window : candidates) {
            s.windows.push_back(window.at);
            chosen += fmt::format("{}{:+d}", chosen.empty() ? "" : ",", window.at);
        }
        LOG_INFO(Frontend,
                 "[ShaderScan] Unreal links: table by {} every 0x{:x} bytes, {} entries; 8-byte "
                 "values that differ per shader at {}",
                 s.KindName(table_kind), stride, entries, chosen.empty() ? "none" : chosen);
        // Only the played shaders' ids are searched for, to tell which places pair them.
        s.id_filter.assign((size_t{1} << ID_FILTER_BITS) / 64, 0);
        for (u32 w = 0; w < s.windows.size(); ++w) {
            std::unordered_map<u64, u32> in_window;
            for (u32 code = 0; code < count; ++code) {
                const u64 entry = s.entries[code];
                if (entry == 0 || static_cast<s64>(entry) + s.windows[w] < 0 ||
                    entry + s.windows[w] + 8 > image.size()) {
                    continue;
                }
                const u64 value = Load64(image.data() + entry + s.windows[w]);
                const auto [it, inserted] = in_window.try_emplace(value, code);
                if (!inserted && it->second != code) {
                    it->second = count; // The same value for two shaders: not an id.
                }
            }
            for (const auto& [value, code] : in_window) {
                if (code < count && s.played_mask[code]) {
                    s.ids[value].emplace_back(code, w);
                    const u64 bit = MixId(value);
                    s.id_filter[bit >> 6] |= u64{1} << (bit & 63);
                }
            }
        }
    }

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
    s.Pair(hits, data, fmt::format("{} (the code package)", name), 0, 0, true);
    for (const Hit& hit : hits) {
        ++s.hits_by_kind[hit.kind];
        ++s.files_by_kind[hit.kind][fmt::format("{} (the code package)", name)];
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
        }
    }
    s.Pair(hits, piece, name, offset, begin, false);
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
    size_t shown = 0;
    std::vector<std::string> names;
    for (const auto& [name, stats] : s.classes) {
        names.push_back(name);
    }
    std::sort(names.begin(), names.end());
    for (const std::string& name : names) {
        const std::string lower = Lower(name);
        if (lower.find("shader") == std::string::npos &&
            lower.find("material") == std::string::npos) {
            continue;
        }
        if (shown++ >= 16) {
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
    std::vector<u32> kinds;
    for (const auto& [kind, n] : s.hits_by_kind) {
        kinds.push_back(kind);
    }
    std::sort(kinds.begin(), kinds.end());
    std::string by_id;
    for (const u32 kind : kinds) {
        if (kind >= KIND_COUNT) {
            by_id += fmt::format("{}{:+d} x{}", by_id.empty() ? "" : ", ",
                                 s.windows[kind - KIND_COUNT], s.hits_by_kind.at(kind));
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
    }
    // A link shows as many pairs at one distance; chance meetings spread over all distances.
    std::vector<std::pair<size_t, std::pair<u32, u32>>> order;
    for (const auto& [kinds_pair, found] : s.finds) {
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
        const PairFinds& found = s.finds.at(order[i].second);
        if (!shown_counts.emplace(order[i].first, found.pairs.size()).second) {
            ++same;
            continue;
        }
        size_t total = 0;
        for (const auto& [distance, n] : found.distances) {
            total += n;
        }
        LOG_INFO(Frontend,
                 "[ShaderScan] Unreal links: vertex by {}, fragment by {}: {} times at the most "
                 "common distance ({}% of {}); {} of {} played pairs close together, at {}; in {}",
                 s.KindName(order[i].second.first), s.KindName(order[i].second.second),
                 order[i].first, total == 0 ? 0 : order[i].first * 100 / total, total,
                 found.pairs.size(), s.played_list.size(),
                 Top(found.distances, 8,
                     [](s64 d, size_t n) { return fmt::format("{:+d} x{}", d, n); }),
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
            if (const auto it = found.examples.find(-distances[d].second);
                it != found.examples.end()) {
                LOG_INFO(Frontend, "[ShaderScan] Unreal links:   {}", it->second);
            }
        }
    }
    if (same > 0) {
        LOG_INFO(Frontend,
                 "[ShaderScan] Unreal links: {} more kinds of reference found the same numbers of "
                 "pairs as one shown",
                 same);
    }
}

} // namespace Preforged::Unreal
