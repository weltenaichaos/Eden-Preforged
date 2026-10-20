// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "common/cityhash.h"
#include "common/fs/file.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/preforged_debug.h"
#include "common/settings.h"
#include "common/thread.h"
#include "common/zstd_compression.h"
#include "core/core.h"
#include "core/file_sys/romfs.h"
#include "core/file_sys/romfs_factory.h"
#include "core/file_sys/vfs/vfs.h"
#include "core/hle/service/filesystem/filesystem.h"
#include "jni/shader_file_scan.h"
#include "jni/unreal_links.h"
#include "jni/unreal_package.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "video_core/preforged_shaders.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"

namespace {

using CachedShaderCode = Vulkan::PipelineCache::CachedShaderCode;

constexpr size_t SPH_SIZE = 0x50;
constexpr size_t NEEDLE_WORDS = 4;
constexpr size_t MAX_WHOLE_FILE = 64ULL << 20;
constexpr size_t CHUNK_SIZE = 64ULL << 20;
constexpr size_t CHUNK_OVERLAP = 1ULL << 20;
constexpr size_t MAX_DECOMPRESSED = 256ULL << 20;
/// zstd files that name no dictionary are unpacked to look for dictionaries up to this size,
/// and up to this much in all.
constexpr size_t MAX_DICTIONARY_ARCHIVE = 16ULL << 20;
constexpr u64 MAX_DICTIONARY_SEARCH_BYTES = 256ULL << 20;
/// How long the walk over all files may look for dictionaries when none is in a file named like
/// one. Tears of the Kingdom's 304,207 files took 187 s, most of the search's time.
constexpr auto DICTIONARY_SEARCH_TIME = std::chrono::seconds{45};
/// Files looked at to tell whether a game's zstd frames name dictionaries at all. XCOM 2 has no
/// zstd files, and looking at all its 33,550 files took 22 s.
constexpr size_t DICTIONARY_SAMPLE_FILES = 2000;
/// Unreal packages up to this size are read whole, so their compressed parts can be unpacked.
constexpr size_t MAX_WHOLE_PACKAGE = 384ULL << 20;
constexpr u32 FILTER_BITS = 22;
constexpr auto TIME_LIMIT = std::chrono::minutes{4};
// Raised whenever the search learns to read more of the game files, so a game an older version
// searched is searched once more. 10: Oodle blocks in Unreal packages. 11: zstd files packed
// with a dictionary (dictionaries looked for in files named like them first), the Unreal shader
// link log. 12: Unreal shader pairs learned from the played ones, in the code package and in
// the other packages. 13: Unreal shader caches read for the pairs. 14: up to 30,000 pairs
// kept from a shader cache. 15: every shader taken from the game files kept by hash, so the
// play log tells a missed pair from a shader the game files don't have.
constexpr u32 SEARCH_VERSION = 15;
constexpr auto TRANSLATE_TIME_LIMIT = std::chrono::seconds{90};
constexpr u32 BNSH_MAGIC = 0x48534E42; // "BNSH"
constexpr u32 FSHA_MAGIC = 0x41485346; // "FSHA"
/// Drawn shaders not among the game files' shaders that the log names one by one.
constexpr size_t MAX_UNMATCHED_LOGGED = 160;
/// Unreal packages with blocks that did not unpack that the log names, and blocks per package.
constexpr size_t FAILED_PACKAGES_LOGGED = 30;
constexpr size_t FAILED_BLOCKS_LOGGED_PER_FILE = 4;
constexpr u32 NVN_CODE_MAGIC = 0x12345678;
constexpr size_t NVN_CODE_HEADER_SIZE = 0x30;
constexpr size_t MAX_SHADER_SIZE = 0x100000;
constexpr size_t MAX_EXTRACTED_BYTES = 256ULL << 20;
constexpr u64 SELF_BRANCH_A = 0xE2400FFFFF87000FULL;
constexpr u64 SELF_BRANCH_B = 0xE2400FFFFF07000FULL;
constexpr u64 EXIT_VALUE = 0xE30000000007000FULL;
/// Added to the name of an Unreal package for what it unpacked to.
constexpr const char* UNPACKED_SUFFIX = " (unpacked)";

enum class MatchKind : u8 { None, Needle, CodeOnly, Exact };

constexpr std::string_view KindName(MatchKind kind) {
    switch (kind) {
    case MatchKind::Exact:
        return "exact";
    case MatchKind::CodeOnly:
        return "code-only";
    case MatchKind::Needle:
        return "start-only";
    default:
        return "none";
    }
}

u64 Load64(const u8* data) {
    u64 value;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

u64 Mix(u64 value) {
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33;
    value *= 0xc4ceb9fe1a85ec53ULL;
    value ^= value >> 33;
    return value;
}

u32 ReadBE32(const u8* data) {
    return (u32{data[0]} << 24) | (u32{data[1]} << 16) | (u32{data[2]} << 8) | u32{data[3]};
}

std::vector<u8> DecompressYaz0(std::span<const u8> in) {
    if (in.size() < 16) {
        return {};
    }
    const size_t out_size = ReadBE32(in.data() + 4);
    if (out_size > MAX_DECOMPRESSED) {
        return {};
    }
    std::vector<u8> out(out_size);
    size_t src = 16;
    size_t dst = 0;
    u8 code = 0;
    int bits = 0;
    while (dst < out_size) {
        if (bits == 0) {
            if (src >= in.size()) {
                return {};
            }
            code = in[src++];
            bits = 8;
        }
        if (code & 0x80) {
            if (src >= in.size()) {
                return {};
            }
            out[dst++] = in[src++];
        } else {
            if (src + 1 >= in.size()) {
                return {};
            }
            const u8 b1 = in[src++];
            const u8 b2 = in[src++];
            const size_t dist = (((b1 & 0xF) << 8) | b2) + 1;
            size_t count = b1 >> 4;
            if (count == 0) {
                if (src >= in.size()) {
                    return {};
                }
                count = size_t{in[src++]} + 0x12;
            } else {
                count += 2;
            }
            if (dist > dst) {
                return {};
            }
            for (size_t i = 0; i < count && dst < out_size; ++i, ++dst) {
                out[dst] = out[dst - dist];
            }
        }
        code <<= 1;
        --bits;
    }
    return out;
}

// Reads the content size from a zstd frame header, or 0 when the frame does not record it.
u64 ZstdContentSize(std::span<const u8> data) {
    if (data.size() < 6) {
        return 0;
    }
    const u8 descriptor = data[4];
    const u32 size_flag = descriptor >> 6;
    const bool single_segment = (descriptor >> 5) & 1;
    constexpr std::array<size_t, 4> dict_id_sizes{0, 1, 2, 4};
    size_t offset = 5 + (single_segment ? 0 : 1) + dict_id_sizes[descriptor & 3];
    constexpr std::array<size_t, 4> field_sizes{0, 2, 4, 8};
    const size_t field_size = size_flag == 0 ? (single_segment ? 1 : 0) : field_sizes[size_flag];
    if (field_size == 0 || offset + field_size > data.size()) {
        return 0;
    }
    u64 size = 0;
    for (size_t i = 0; i < field_size; ++i) {
        size |= u64{data[offset + i]} << (8 * i);
    }
    return field_size == 2 ? size + 256 : size;
}

bool IsYaz0(std::span<const u8> data) {
    return data.size() >= 4 && std::memcmp(data.data(), "Yaz0", 4) == 0;
}

bool IsZstd(std::span<const u8> data) {
    return data.size() >= 4 && data[0] == 0x28 && data[1] == 0xB5 && data[2] == 0x2F &&
           data[3] == 0xFD;
}

/// The first 4 bytes of a buffer, as the search counts what it could not unpack.
u32 Start32(std::span<const u8> data) {
    u32 value = 0;
    std::memcpy(&value, data.data(), std::min<size_t>(data.size(), sizeof(value)));
    return value;
}

/// The files of a SARC archive. Returns false when the data is not one.
bool SarcEntries(std::span<const u8> data, std::vector<std::span<const u8>>& entries) {
    entries.clear();
    if (data.size() < 0x14 || std::memcmp(data.data(), "SARC", 4) != 0) {
        return false;
    }
    u16 header_size;
    u16 bom;
    u32 data_offset;
    std::memcpy(&header_size, data.data() + 4, 2);
    std::memcpy(&bom, data.data() + 6, 2);
    std::memcpy(&data_offset, data.data() + 0xC, 4);
    if (bom != 0xFEFF || size_t{header_size} + 0xC > data.size() || data_offset > data.size()) {
        return false;
    }
    const u8* const sfat = data.data() + header_size;
    if (std::memcmp(sfat, "SFAT", 4) != 0) {
        return false;
    }
    u16 sfat_size;
    u16 node_count;
    std::memcpy(&sfat_size, sfat + 4, 2);
    std::memcpy(&node_count, sfat + 6, 2);
    const size_t nodes_offset = size_t{header_size} + sfat_size;
    if (nodes_offset + size_t{node_count} * 16 > data.size()) {
        return false;
    }
    for (size_t node = 0; node < node_count; ++node) {
        const u8* const entry = data.data() + nodes_offset + node * 16;
        u32 start;
        u32 end;
        std::memcpy(&start, entry + 8, 4);
        std::memcpy(&end, entry + 12, 4);
        if (end < start || size_t{data_offset} + end > data.size()) {
            entries.clear();
            return false;
        }
        entries.push_back(data.subspan(data_offset + start, end - start));
    }
    return true;
}

struct HeaderInfo {
    u64 hash;
    Shader::Stage stage;
};

struct ExtractedShader {
    Shader::Stage stage;
    u32 start_address;     ///< Keeps the game's alignment of scheduling words
    std::vector<u64> code; ///< Starts at the program header for graphics stages
};

// Reads the stage from a program header, if the words look like one.
std::optional<Shader::Stage> StageFromHeader(const Shader::ProgramHeader& sph) {
    const u32 sph_type = sph.common0.sph_type;
    switch (static_cast<u32>(sph.common0.shader_type)) {
    case 1:
        return sph_type == 1 ? std::optional{Shader::Stage::VertexB} : std::nullopt;
    case 2:
        return sph_type == 1 ? std::optional{Shader::Stage::TessellationControl} : std::nullopt;
    case 3:
        return sph_type == 1 ? std::optional{Shader::Stage::TessellationEval} : std::nullopt;
    case 4:
        return sph_type == 1 ? std::optional{Shader::Stage::Geometry} : std::nullopt;
    case 5:
        return sph_type == 2 ? std::optional{Shader::Stage::Fragment} : std::nullopt;
    default:
        return std::nullopt;
    }
}

using DriverConstants = std::unordered_map<u64, u32>;

// What a guessing environment may take from the real one, to find out which guess differs.
struct RealParts {
    bool cbuf = false;
    bool textures = false;
    bool misc = false;
};

// An environment for a shader taken from the game files, where nothing about the running game is
// known: every texture is guessed to be 2D, reads of the driver's constant buffer come from values
// seen in other shaders, and every other constant buffer read returns zero. For testing, parts can
// be taken from the environment the game really used.
class GuessEnvironment final : public Shader::Environment {
public:
    explicit GuessEnvironment(std::span<const u64> code_, Shader::Stage stage_, u32 start_address_,
                              u32 texture_bound_, const DriverConstants& driver_constants_,
                              Shader::Environment* real_ = nullptr, RealParts parts_ = {})
        : code{code_}, texture_bound{texture_bound_}, driver_constants{driver_constants_},
          real{real_}, parts{parts_} {
        stage = stage_;
        start_address = start_address_;
        is_proprietary_driver = texture_bound == 2;
        if (stage != Shader::Stage::Compute) {
            std::memcpy(&sph, code.data(), sizeof(sph));
            local_memory_size = static_cast<u32>(sph.LocalMemorySize()) +
                                sph.common3.shader_local_memory_crs_size;
        }
    }

    u64 ReadInstruction(u32 address) override {
        const size_t index = (address - start_address) / sizeof(u64);
        if (address < start_address || index >= code.size()) {
            throw Shader::LogicError("Read past the end of the shader");
        }
        return code[index];
    }
    u32 ReadCbufValue(u32 index, u32 offset) override {
        if (parts.cbuf) {
            try {
                return real->ReadCbufValue(index, offset);
            } catch (const std::exception&) {
            }
        }
        const auto it = driver_constants.find((u64{index} << 32) | offset);
        return it != driver_constants.end() ? it->second : 0;
    }
    Shader::TextureType ReadTextureType(u32 handle) override {
        if (parts.textures) {
            try {
                return real->ReadTextureType(handle);
            } catch (const std::exception&) {
            }
        }
        return Shader::TextureType::Color2D;
    }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override {
        if (parts.textures) {
            try {
                return real->ReadTexturePixelFormat(handle);
            } catch (const std::exception&) {
            }
        }
        return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }
    bool IsTexturePixelFormatInteger(u32 handle) override {
        if (parts.textures) {
            try {
                return real->IsTexturePixelFormatInteger(handle);
            } catch (const std::exception&) {
            }
        }
        return false;
    }
    u32 ReadViewportTransformState() override {
        return parts.misc ? real->ReadViewportTransformState() : 1;
    }
    u32 TextureBoundBuffer() const override {
        return texture_bound;
    }
    u32 LocalMemorySize() const override {
        return parts.misc ? real->LocalMemorySize() : local_memory_size;
    }
    u32 SharedMemorySize() const override {
        return parts.misc ? real->SharedMemorySize() : 0x8000;
    }
    std::array<u32, 3> WorkgroupSize() const override {
        return parts.misc ? real->WorkgroupSize() : std::array<u32, 3>{64, 1, 1};
    }
    bool HasHLEMacroState() const override {
        return parts.misc && real->HasHLEMacroState();
    }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32 bank, u32 offset) override {
        return parts.misc ? real->GetReplaceConstBuffer(bank, offset) : std::nullopt;
    }
    void Dump(u64, u64) override {}

private:
    std::span<const u64> code;
    u32 texture_bound;
    u32 local_memory_size = 0;
    const DriverConstants& driver_constants;
    Shader::Environment* real;
    RealParts parts;
};

class Scanner {
public:
    explicit Scanner(std::vector<CachedShaderCode> shaders_, std::stop_token stop_,
                     std::chrono::steady_clock::time_point deadline_)
        : shaders{std::move(shaders_)}, stop{stop_}, deadline{deadline_}, filter((size_t{1} << FILTER_BITS) / 64),
          results(shaders.size()) {
        if (Common::Preforged::Options().log_unreal_links) {
            links = std::make_unique<Preforged::Unreal::LinkSurvey>(CHUNK_OVERLAP);
        }
        for (size_t index = 0; index < shaders.size(); ++index) {
            const auto& shader = shaders[index];
            const size_t start = NeedleOffset(shader) / sizeof(u64);
            if (shader.code.size() < start + NEEDLE_WORDS) {
                continue;
            }
            const u64* const words = shader.code.data() + start;
            const u64 first = Mix(words[0]);
            filter[(first & ((1u << FILTER_BITS) - 1)) / 64] |= u64{1} << (first % 64);
            needles[NeedleKey(words)].push_back(index);
        }
        // The constant buffer slot for bindless textures is set once by the game, so the cached
        // shaders tell which one it uses.
        std::map<u32, size_t> bounds;
        for (const auto& shader : shaders) {
            if (shader.env) {
                ++bounds[shader.env->TextureBoundBuffer()];
            }
        }
        for (const auto& [bound, count] : bounds) {
            if (count > bounds[texture_bound]) {
                texture_bound = bound;
            }
        }
        LOG_INFO(Frontend, "[ShaderScan] game uses constant buffer {} for bindless textures",
                 texture_bound);

        // The official driver keeps its own values in constant buffer 1, the same for every
        // shader, so values read by cached shaders can stand in for shaders not seen yet.
        size_t conflicts = 0;
        for (const auto& shader : shaders) {
            if (!shader.env) {
                continue;
            }
            for (const auto& [key, value] : shader.env->CbufValues()) {
                if ((key >> 32) != 1) {
                    continue;
                }
                const auto [it, inserted] = driver_constants.emplace(key, value);
                if (!inserted && it->second != value) {
                    ++conflicts;
                }
            }
        }
        LOG_INFO(Frontend, "[ShaderScan] driver constants seen: {}, conflicting values: {}",
                 driver_constants.size(), conflicts);
    }

    void ScanDir(const FileSys::VirtualDir& dir, const std::string& path, bool shader_files) {
        if (!dir || Stopped()) {
            return;
        }
        for (const auto& file : dir->GetFiles()) {
            if (Stopped()) {
                return;
            }
            const std::string name = path + "/" + file->GetName();
            if (LooksLikeShaderFile(name) != shader_files || IsMedia(name)) {
                continue;
            }
            ScanFile(file, name);
            ++files_visited;
            ReportProgress(false);
        }
        for (const auto& subdir : dir->GetSubdirectories()) {
            ScanDir(subdir, path + "/" + subdir->GetName(), shader_files);
        }
    }

    // Counts the files the two passes of ScanDir visit, for the progress bar.
    static size_t CountFiles(const FileSys::VirtualDir& dir, const std::string& path) {
        if (!dir) {
            return 0;
        }
        size_t count = 0;
        for (const auto& file : dir->GetFiles()) {
            count += IsMedia(path + "/" + file->GetName()) ? 0 : 1;
        }
        for (const auto& subdir : dir->GetSubdirectories()) {
            count += CountFiles(subdir, path + "/" + subdir->GetName());
        }
        return count;
    }

    void SetProgress(const std::function<void(size_t, size_t)>* callback, size_t total) {
        progress_callback = callback;
        total_files = total;
    }

    void ReportProgress(bool force) {
        if (!progress_callback || !*progress_callback) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (!force && now - last_progress_report < std::chrono::milliseconds{250}) {
            return;
        }
        last_progress_report = now;
        // Stays one short of the end until the shaders are processed too.
        (*progress_callback)(std::min(files_visited, total_files - 1), total_files);
    }

    void Report(std::string_view phase) const {
        std::array<size_t, 4> by_kind{};
        std::map<u32, std::pair<size_t, size_t>> by_stage;
        for (size_t index = 0; index < shaders.size(); ++index) {
            const auto kind = results[index].kind;
            ++by_kind[static_cast<size_t>(kind)];
            auto& stage = by_stage[static_cast<u32>(shaders[index].stage)];
            ++stage.first;
            if (kind == MatchKind::Exact || kind == MatchKind::CodeOnly) {
                ++stage.second;
            }
        }
        LOG_INFO(Frontend,
                 "[ShaderScan] {}: {} files, {} MiB read, {} cached shaders: {} exact, {} "
                 "code-only, {} start-only, {} not found",
                 phase, files_scanned, bytes_scanned >> 20, shaders.size(),
                 by_kind[static_cast<size_t>(MatchKind::Exact)],
                 by_kind[static_cast<size_t>(MatchKind::CodeOnly)],
                 by_kind[static_cast<size_t>(MatchKind::Needle)],
                 by_kind[static_cast<size_t>(MatchKind::None)]);
        for (const auto& [stage, counts] : by_stage) {
            LOG_INFO(Frontend, "[ShaderScan] stage {}: {} of {} found", stage, counts.second,
                     counts.first);
        }
        std::vector<std::pair<size_t, std::string>> files;
        for (const auto& [name, hits] : hits_per_file) {
            files.emplace_back(hits.hits, name);
        }
        std::sort(files.rbegin(), files.rend());
        for (size_t i = 0; i < std::min<size_t>(files.size(), 15); ++i) {
            const auto& stats = hits_per_file.at(files[i].second);
            LOG_INFO(Frontend,
                     "[ShaderScan] file {} hits {} self-branches {} (shaders in file, estimate)",
                     files[i].second, stats.hits, stats.terminators);
        }
        size_t total_terminators = 0;
        for (const auto& [name, stats] : hits_per_file) {
            total_terminators += stats.terminators;
        }
        LOG_INFO(Frontend, "[ShaderScan] files with hits: {}, shaders in them (estimate): {}",
                 hits_per_file.size(), total_terminators);
    }

    // Finds the zstd dictionaries in the game files before the search, so the files packed with
    // them can be unpacked wherever they come. A dictionary is a file of its own, or a file in an
    // archive packed with zstd without one. Files named like dictionaries are read first; only
    // when none holds one are all files walked, for a limited time.
    void FindZstdDictionaries(const FileSys::VirtualDir& root) {
        if (!Common::Preforged::Options().zstd_dictionaries) {
            return;
        }
        using Clock = std::chrono::steady_clock;
        const auto start = Clock::now();
        const auto ms = [](Clock::duration time) {
            return std::chrono::duration_cast<std::chrono::milliseconds>(time).count();
        };
        const auto for_each_file =
            [&](const std::function<bool(const FileSys::VirtualFile&, const std::string&)>& visit) {
                std::function<bool(const FileSys::VirtualDir&, const std::string&)> walk =
                    [&](const FileSys::VirtualDir& dir, const std::string& path) {
                        if (!dir || Stopped()) {
                            return false;
                        }
                        for (const auto& file : dir->GetFiles()) {
                            if (!visit(file, path + "/" + file->GetName())) {
                                return false;
                            }
                        }
                        for (const auto& subdir : dir->GetSubdirectories()) {
                            if (!walk(subdir, path + "/" + subdir->GetName())) {
                                return false;
                            }
                        }
                        return true;
                    };
                walk(root, "");
            };
        const auto found_list = [&] {
            std::string found;
            for (const auto& [id, bytes] : zstd_dictionaries.Sizes()) {
                found +=
                    fmt::format("{}{:08x} ({} KiB)", found.empty() ? "" : ", ", id, bytes >> 10);
            }
            return found.empty() ? std::string{"none"} : found;
        };

        // Files named like a dictionary ("dic" anywhere in the name, as in .zsdic or ZsDic).
        size_t by_name = 0;
        u64 by_name_read = 0;
        std::string by_name_examples;
        size_t file_count = 0;
        for_each_file([&](const FileSys::VirtualFile& file, const std::string& name) {
            ++file_count;
            const size_t size = file->GetSize();
            if (size >= 8 && size <= MAX_DICTIONARY_ARCHIVE &&
                by_name_read + size <= MAX_DICTIONARY_SEARCH_BYTES &&
                Lower(file->GetName()).find("dic") != std::string::npos) {
                ++by_name;
                by_name_read += size;
                if (by_name <= 4) {
                    by_name_examples += fmt::format(
                        "{}{} ({} KiB)", by_name_examples.empty() ? "" : ", ", name, size >> 10);
                }
                FindDictionariesIn(file->ReadAllBytes(), name, 0);
            }
            return true;
        });
        const auto by_name_time = Clock::now() - start;
        if (!zstd_dictionaries.Sizes().empty()) {
            LOG_INFO(Frontend,
                     "[ShaderScan] zstd dictionaries: {} found: {}, in {} files named like "
                     "dictionaries ({}) read in {} ms",
                     zstd_dictionaries.Sizes().size(), found_list(), by_name, by_name_examples,
                     ms(by_name_time));
            return;
        }

        // None found by name. Look at the start of a sample of the files first, so a game whose
        // files name no dictionary doesn't pay for looking at all of them.
        const size_t step = std::max<size_t>(1, file_count / DICTIONARY_SAMPLE_FILES);
        size_t sampled = 0;
        size_t sample_naming = 0;
        size_t index = 0;
        for_each_file([&](const FileSys::VirtualFile& file, const std::string&) {
            if (index++ % step != 0) {
                return true;
            }
            const size_t size = file->GetSize();
            if (size < 8 || size > MAX_WHOLE_FILE) {
                return true;
            }
            ++sampled;
            const std::vector<u8> head = file->ReadBytes(std::min<size_t>(size, 18), 0);
            sample_naming += Common::Compression::IsZstdDictionary(head) ||
                                     (IsZstd(head) &&
                                      Common::Compression::ZstdFrameDictionaryId(head) != 0)
                                 ? 1
                                 : 0;
            return true;
        });
        const auto sample_time = Clock::now() - start - by_name_time;
        if (sample_naming == 0) {
            LOG_INFO(Frontend,
                     "[ShaderScan] zstd dictionaries: none in {} files named like dictionaries "
                     "({}) read in {} ms, and none named in a sample of {} of the {} files "
                     "looked at in {} ms, so not all files are looked at",
                     by_name, by_name_examples.empty() ? std::string{"none"} : by_name_examples,
                     ms(by_name_time), sampled, file_count, ms(sample_time));
            return;
        }

        // Look at the start of every file for dictionaries and for the ids of the dictionaries
        // zstd frames name, then unpack the small zstd files that name none, smallest first,
        // until every named dictionary is found.
        const auto walk_start = Clock::now();
        const auto walk_deadline = walk_start + DICTIONARY_SEARCH_TIME;
        std::map<u32, size_t> named;
        std::vector<std::tuple<size_t, std::string, FileSys::VirtualFile>> containers;
        size_t looked_at = 0;
        bool walk_done = true;
        for_each_file([&](const FileSys::VirtualFile& file, const std::string& name) {
            if (Clock::now() > walk_deadline) {
                walk_done = false;
                return false;
            }
            const size_t size = file->GetSize();
            if (size < 8 || size > MAX_WHOLE_FILE) {
                return true;
            }
            ++looked_at;
            // The longest zstd frame header.
            const std::vector<u8> head = file->ReadBytes(std::min<size_t>(size, 18), 0);
            if (Common::Compression::IsZstdDictionary(head)) {
                AddDictionary(file->ReadAllBytes(), name);
            } else if (IsZstd(head)) {
                const u32 id = Common::Compression::ZstdFrameDictionaryId(head);
                if (id != 0) {
                    ++named[id];
                } else if (size <= MAX_DICTIONARY_ARCHIVE) {
                    containers.emplace_back(size, name, file);
                }
            }
            return true;
        });
        const auto walk_time = Clock::now() - walk_start;
        const auto all_found = [&] {
            return std::ranges::all_of(
                named, [&](const auto& entry) { return zstd_dictionaries.Has(entry.first); });
        };
        std::sort(containers.begin(), containers.end(),
                  [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });
        size_t unpacked = 0;
        u64 read = 0;
        bool unpack_done = true;
        for (const auto& [size, name, file] : containers) {
            if (named.empty() || all_found()) {
                break;
            }
            if (read + size > MAX_DICTIONARY_SEARCH_BYTES || Clock::now() > walk_deadline ||
                Stopped()) {
                unpack_done = false;
                break;
            }
            read += size;
            ++unpacked;
            const std::vector<u8> data = file->ReadAllBytes();
            u32 used = 0;
            const std::vector<u8> out = zstd_dictionaries.Decompress(data, MAX_DECOMPRESSED, used);
            FindDictionariesIn(out, name, 0);
        }
        std::string wanted;
        for (const auto& [id, frames] : named) {
            wanted += fmt::format("{}{:08x} x{}{}", wanted.empty() ? "" : ", ", id, frames,
                                  zstd_dictionaries.Has(id) ? "" : " (not found)");
        }
        LOG_INFO(Frontend,
                 "[ShaderScan] zstd dictionaries: {} found: {}; {} files named like "
                 "dictionaries ({}) read in {} ms; {} of a sample of {} files name one; files "
                 "name {}; looked at the start of {} files in {} ms{}; unpacked {} of {} small "
                 "zstd files that name none ({} KiB) in {} ms{}",
                 zstd_dictionaries.Sizes().size(), found_list(), by_name,
                 by_name_examples.empty() ? std::string{"none"} : by_name_examples,
                 ms(by_name_time), sample_naming, sampled,
                 wanted.empty() ? std::string{"none"} : wanted, looked_at, ms(walk_time),
                 walk_done ? "" : " (stopped at the time limit)", unpacked, containers.size(),
                 read >> 10, ms(Clock::now() - walk_start - walk_time),
                 unpack_done ? "" : " (stopped at the time or size limit)");
    }

    void FindDictionariesIn(std::span<const u8> data, const std::string& name, int depth) {
        if (data.empty() || depth > 2) {
            return;
        }
        if (Common::Compression::IsZstdDictionary(data)) {
            AddDictionary(data, name);
            return;
        }
        if (IsZstd(data)) {
            u32 used = 0;
            FindDictionariesIn(zstd_dictionaries.Decompress(data, MAX_DECOMPRESSED, used), name,
                               depth + 1);
            return;
        }
        std::vector<std::span<const u8>> entries;
        if (SarcEntries(data, entries)) {
            for (size_t node = 0; node < entries.size(); ++node) {
                FindDictionariesIn(entries[node], fmt::format("{}#{}", name, node), depth + 1);
            }
        }
    }

    void AddDictionary(std::span<const u8> data, const std::string& name) {
        if (const u32 id = zstd_dictionaries.Add(data); id != 0) {
            LOG_INFO(Frontend, "[ShaderScan] zstd dictionary {:08x} ({} KiB) in {}", id,
                     data.size() >> 10, name);
        }
    }

    // Returns the decompressed contents when the data is a compressed container we understand.
    std::vector<u8> Decompress(std::span<const u8> data) {
        if (IsYaz0(data)) {
            std::vector<u8> out = DecompressYaz0(data);
            formats.yaz0 += out.empty() ? 0 : 1;
            return out;
        }
        if (!IsZstd(data)) {
            return {};
        }
        std::vector<u8> out;
        u32 used = 0;
        if (Common::Preforged::Options().zstd_dictionaries) {
            out = zstd_dictionaries.Decompress(data, MAX_DECOMPRESSED, used);
        } else if (const u64 size = ZstdContentSize(data); size != 0 && size <= MAX_DECOMPRESSED) {
            out = Common::Compression::DecompressDataZSTD(data);
        }
        if (!out.empty()) {
            if (used != 0) {
                ++formats.zstd_with_dictionary[used];
            } else {
                ++formats.zstd_plain;
            }
            return out;
        }
        const u32 id = Common::Compression::ZstdFrameDictionaryId(data);
        if (id != 0 && !zstd_dictionaries.Has(id)) {
            ++formats.zstd_missing_dictionary[id];
        } else {
            ++formats.zstd_failed;
            if (formats.zstd_failed_examples.size() < 4) {
                formats.zstd_failed_examples.push_back(
                    fmt::format("{} bytes, dictionary {:08x}, records {} bytes: {}", data.size(),
                                id, ZstdContentSize(data), Hex(data, 0, 32)));
            }
        }
        return {};
    }

    static std::string DescribeStart(u32 start) {
        std::string text;
        for (size_t i = 0; i < 4; ++i) {
            const char c = static_cast<char>((start >> (i * 8)) & 0xFF);
            text += std::isprint(static_cast<unsigned char>(c)) ? c : '.';
        }
        return fmt::format("{:02x}{:02x}{:02x}{:02x} '{}'", start & 0xFF, (start >> 8) & 0xFF,
                           (start >> 16) & 0xFF, start >> 24, text);
    }

    static std::string TopStarts(const std::map<u32, std::pair<size_t, u64>>& starts,
                                 size_t limit) {
        std::vector<std::pair<u64, u32>> by_bytes;
        for (const auto& [start, entry] : starts) {
            by_bytes.emplace_back(entry.second, start);
        }
        std::sort(by_bytes.rbegin(), by_bytes.rend());
        std::string out;
        for (size_t i = 0; i < std::min(limit, by_bytes.size()); ++i) {
            const auto& [count, bytes] = starts.at(by_bytes[i].second);
            out += fmt::format("{}{} x{} ({} MiB)", out.empty() ? "" : ", ",
                               DescribeStart(by_bytes[i].second), count, bytes >> 20);
        }
        return out.empty() ? std::string{"none"} : out;
    }

    /// Logs each drawn shader that is not among the shaders taken from the game files, so its
    /// pairs can't be compiled ahead: how much of its code was found, and where it differs.
    void ReportUnmatched() const {
        std::unordered_set<u64> drawn;
        for (const auto& shader : shaders) {
            drawn.insert(shader.unique_hash);
        }
        std::array<size_t, 4> by_kind{};
        size_t logged = 0;
        for (size_t index = 0; index < shaders.size(); ++index) {
            const auto& shader = shaders[index];
            if (extracted.contains(shader.unique_hash)) {
                continue;
            }
            const Result& result = results[index];
            ++by_kind[static_cast<size_t>(result.kind)];
            if (logged >= MAX_UNMATCHED_LOGGED) {
                continue;
            }
            ++logged;
            const size_t needle_offset = NeedleOffset(shader);
            const size_t code_bytes = shader.code.size() * sizeof(u64);
            std::string found;
            if (result.best_length > 0) {
                // The code header in front of the code, when the file has one there.
                std::string header = "no code header in front of it";
                const u64 code_start = result.best_offset - needle_offset;
                if (code_start >= NVN_CODE_HEADER_SIZE) {
                    const auto& file_headers = headers[result.best_file];
                    if (const auto it = file_headers.find(code_start - NVN_CODE_HEADER_SIZE);
                        it != file_headers.end()) {
                        const auto taken = extracted.find(it->second.hash);
                        header = fmt::format(
                            "its code header's shader {:016x} has {} bytes and was {}drawn",
                            it->second.hash,
                            taken != extracted.end() ? taken->second.code.size() * sizeof(u64) : 0,
                            drawn.contains(it->second.hash) ? "" : "not ");
                    }
                }
                found = fmt::format("{} of its bytes from +0x{:x} found in {} at 0x{:x}, {}; {}",
                                    result.best_length, needle_offset, file_names[result.best_file],
                                    code_start, result.best_difference, header);
            } else {
                std::string start;
                const auto* const code = reinterpret_cast<const u8*>(shader.code.data());
                for (size_t i = needle_offset; i < std::min(code_bytes, needle_offset + 32); ++i) {
                    start += fmt::format("{:02x}", code[i]);
                }
                found = fmt::format("its first instructions are {}", start);
            }
            LOG_INFO(Frontend,
                     "[ShaderScan] drawn shader {:016x} (stage {}, {} bytes of code) is not among "
                     "the game files' shaders ({}): {}",
                     shader.unique_hash, static_cast<u32>(shader.stage), code_bytes,
                     KindName(result.kind), found);
        }
        const size_t total = by_kind[0] + by_kind[1] + by_kind[2] + by_kind[3];
        const auto count = [&](MatchKind kind) { return by_kind[static_cast<size_t>(kind)]; };
        LOG_INFO(Frontend,
                 "[ShaderScan] drawn shaders not among the game files' shaders, so their new "
                 "pairs can't be compiled ahead: {} of {} ({} found whole, {} found with another "
                 "program header, {} only their start, {} not found){}",
                 total, shaders.size(), count(MatchKind::Exact), count(MatchKind::CodeOnly),
                 count(MatchKind::Needle), count(MatchKind::None),
                 logged < total ? fmt::format(", the first {} shown", logged) : std::string{});
    }

    // Logs what the search unpacked and what it could not, so a game whose shaders it does not
    // find shows which of its formats the search does not read.
    void ReportFormats() const {
        std::string with;
        for (const auto& [id, count] : formats.zstd_with_dictionary) {
            with += fmt::format("{}{:08x} x{}", with.empty() ? "" : ", ", id, count);
        }
        std::string missing;
        for (const auto& [id, count] : formats.zstd_missing_dictionary) {
            missing += fmt::format("{}{:08x} x{}", missing.empty() ? "" : ", ", id, count);
        }
        LOG_INFO(Frontend,
                 "[ShaderScan] unpacked: Yaz0 {}, zstd {} without a dictionary and {} with one "
                 "({}); not unpacked: zstd {} naming a dictionary the game files don't have ({}), "
                 "zstd {} that did not unpack",
                 formats.yaz0, formats.zstd_plain,
                 std::accumulate(formats.zstd_with_dictionary.begin(),
                                 formats.zstd_with_dictionary.end(), size_t{0},
                                 [](size_t sum, const auto& entry) { return sum + entry.second; }),
                 with.empty() ? "none" : with,
                 std::accumulate(formats.zstd_missing_dictionary.begin(),
                                 formats.zstd_missing_dictionary.end(), size_t{0},
                                 [](size_t sum, const auto& entry) { return sum + entry.second; }),
                 missing.empty() ? "none" : missing, formats.zstd_failed);
        for (const std::string& example : formats.zstd_failed_examples) {
            LOG_INFO(Frontend, "[ShaderScan] zstd that did not unpack: {}", example);
        }
        LOG_INFO(Frontend, "[ShaderScan] searched as stored, by their first bytes: {}",
                 TopStarts(formats.stored_starts, 16));
        LOG_INFO(Frontend,
                 "[ShaderScan] files too large to read whole, searched in pieces as stored: {}",
                 TopStarts(formats.large_starts, 8));
    }

    /// How many of the shaders the game drew the search found.
    size_t FoundCount() const {
        size_t found = 0;
        for (const auto& result : results) {
            found += result.kind == MatchKind::Exact || result.kind == MatchKind::CodeOnly ? 1 : 0;
        }
        return found;
    }

    /// One line on what the search found, kept in the done marker for later starts.
    /// `cut_short` says whether the search stopped before it read all files.
    std::string ResultSummary(bool cut_short) const {
        return fmt::format("found {} of the {} shaders the game had drawn, took {} shaders from "
                           "the files{}",
                           FoundCount(), shaders.size(), extracted.size(),
                           !cut_short              ? ""
                           : stop.stop_requested() ? ", stopped early because the game was closed"
                                                   : ", stopped early at the time limit");
    }

    // Says in one line how much of what the game drew the search found, and where to look when
    // it found little.
    void ReportResult() const {
        const size_t found = FoundCount();
        const size_t percent = shaders.empty() ? 0 : found * 100 / shaders.size();
        LOG_INFO(Frontend,
                 "[ShaderScan] result: {} of the {} shaders the game drew found in its files "
                 "({}%), {} shaders taken from the files{}",
                 found, shaders.size(), percent, extracted.size(),
                 percent >= 50 ? ""
                               : "; most were not found, so the rest are likely in files the "
                                 "search can't unpack (see the 'unpacked' and 'searched as "
                                 "stored' lines above)");
    }

    void ReportUnreal() const {
        const auto& stats = unreal_stats;
        if (stats.packages == 0) {
            return;
        }
        std::string blocks;
        for (size_t method = 0; method < Preforged::Unreal::METHOD_COUNT; ++method) {
            if (stats.blocks[method] != 0) {
                blocks += fmt::format(
                    "{}{} {}", blocks.empty() ? "" : ", ",
                    Preforged::Unreal::MethodName(static_cast<Preforged::Unreal::Method>(method)),
                    stats.blocks[method]);
            }
        }
        LOG_INFO(Frontend,
                 "[ShaderScan] Unreal packages: {} ({} with compressed parts), {} compressed "
                 "parts, blocks: {}; {} MiB packed unpacked into {} MiB in {} ms",
                 stats.packages, stats.compressed_packages, stats.parts,
                 blocks.empty() ? std::string{"none"} : blocks, stats.packed_bytes >> 20,
                 stats.unpacked_bytes >> 20,
                 std::chrono::duration_cast<std::chrono::milliseconds>(stats.unpack_time).count());
        std::string oodle;
        for (size_t compressor = 0; compressor < stats.oodle_compressors.size(); ++compressor) {
            if (stats.oodle_compressors[compressor] != 0) {
                oodle += fmt::format("{}{} ({}) {}", oodle.empty() ? "" : ", ",
                                     Preforged::Unreal::OodleCompressorName(compressor),
                                     compressor, stats.oodle_compressors[compressor]);
            }
        }
        if (!oodle.empty()) {
            LOG_INFO(Frontend, "[ShaderScan] Unreal Oodle blocks by decoder: {}", oodle);
        }
        for (const std::string& example : stats.oodle_examples) {
            LOG_INFO(Frontend, "[ShaderScan] Unreal Oodle block that did not unpack: {}", example);
        }
        ReportFailedPackages();
        for (const std::string& example : stats.header_examples) {
            LOG_INFO(Frontend, "[ShaderScan] Unreal package header: {}", example);
        }
        for (const std::string& example : stats.unknown_examples) {
            LOG_INFO(Frontend, "[ShaderScan] Unreal block packed with an unknown method: {}",
                     example);
        }
    }

    /// Logs the Unreal packages with blocks that did not unpack, most bytes lost first, so a
    /// shader the search does not find can be told to be in one of them.
    void ReportFailedPackages() const {
        if (failed_packages.empty()) {
            return;
        }
        std::vector<std::pair<std::string, const FailedPackage*>> order;
        size_t blocks = 0;
        u64 lost = 0;
        for (const auto& [name, failed] : failed_packages) {
            order.emplace_back(name, &failed);
            blocks += failed.blocks;
            lost += failed.lost;
        }
        std::ranges::sort(
            order, [](const auto& a, const auto& b) { return a.second->lost > b.second->lost; });
        LOG_INFO(Frontend,
                 "[ShaderScan] Unreal packages with blocks that did not unpack: {} packages, {} "
                 "blocks, {} KiB not searched; most first{}:",
                 order.size(), blocks, lost >> 10,
                 order.size() > FAILED_PACKAGES_LOGGED
                     ? fmt::format(" (the first {})", FAILED_PACKAGES_LOGGED)
                     : std::string{});
        for (size_t i = 0; i < std::min(order.size(), FAILED_PACKAGES_LOGGED); ++i) {
            const auto& [name, failed] = order[i];
            std::string decoders;
            for (const auto& [decoder, count] : failed->decoders) {
                decoders +=
                    fmt::format("{}{} x{}", decoders.empty() ? "" : ", ",
                                decoder == Preforged::Unreal::FAILED_NOT_OODLE
                                    ? std::string{"not Oodle"}
                                    : fmt::format("Oodle decoder {} ({})", decoder,
                                                  Preforged::Unreal::OodleCompressorName(decoder)),
                                count);
            }
            std::string where;
            for (const auto& block : failed->first) {
                where += fmt::format("{}0x{:x} (file 0x{:x}, {} -> {} bytes)",
                                     where.empty() ? "" : ", ", block.unpacked_offset,
                                     block.file_offset, block.packed, block.unpacked);
            }
            LOG_INFO(Frontend,
                     "[ShaderScan] Unreal package {}: {} blocks did not unpack ({}), {} KiB not "
                     "searched, at {}{}",
                     name, failed->blocks, decoders, failed->lost >> 10, where,
                     failed->blocks > failed->first.size() ? ", ..." : "");
        }
    }

    void ReportExtraction() const {
        std::map<u32, size_t> by_stage;
        for (const auto& [hash, shader] : extracted) {
            ++by_stage[static_cast<u32>(shader.stage)];
        }
        LOG_INFO(Frontend,
                 "[ShaderScan] extraction: {} code headers, {} distinct shaders, {} MiB",
                 code_headers, extracted.size(), extracted_bytes >> 20);
        for (const auto& [stage, count] : by_stage) {
            LOG_INFO(Frontend, "[ShaderScan] extracted stage {}: {}", stage, count);
        }
        std::map<u32, std::pair<size_t, size_t>> recall;
        size_t self_branch_recall = 0;
        for (const auto& shader : shaders) {
            auto& entry = recall[static_cast<u32>(shader.stage)];
            ++entry.first;
            if (extracted.contains(shader.unique_hash)) {
                ++entry.second;
            }
            if (self_branch_hashes.contains(shader.unique_hash)) {
                ++self_branch_recall;
            }
        }
        for (const auto& [stage, counts] : recall) {
            LOG_INFO(Frontend, "[ShaderScan] cached stage {}: {} of {} extracted from game files",
                     stage, counts.second, counts.first);
        }
        LOG_INFO(Frontend, "[ShaderScan] cached shaders found when ending at the self branch: {}",
                 self_branch_recall);
    }

    // How often the guesses used for game-file shaders match what the game really used.
    void ReportGuessAccuracy() const {
        size_t no_textures = 0;
        size_t all_2d = 0;
        size_t needs_cbuf = 0;
        std::array<size_t, Shader::NUM_TEXTURE_TYPES> type_counts{};
        for (const auto& shader : shaders) {
            if (shader.cbuf_value_count > 0) {
                ++needs_cbuf;
            }
            if (shader.texture_types.empty()) {
                ++no_textures;
            } else if (std::ranges::all_of(shader.texture_types, [](Shader::TextureType type) {
                           return type == Shader::TextureType::Color2D;
                       })) {
                ++all_2d;
            }
            for (const auto type : shader.texture_types) {
                ++type_counts[static_cast<size_t>(type)];
            }
        }
        LOG_INFO(Frontend,
                 "[ShaderScan] runtime info of {} cached shaders: {} use no texture types, {} use "
                 "only 2D textures, {} read constant buffers for bindless handles",
                 shaders.size(), no_textures, all_2d, needs_cbuf);
        LOG_INFO(Frontend, "[ShaderScan] texture types used: {}", fmt::join(type_counts, ","));
    }

    // Translates each cached shader twice, with what the game really used and with the guesses,
    // and compares the results. Equal results mean the guesses would have built the same shader.
    void CompareGuessesWithRuntime() const {
        const Shader::HostTranslateInfo host_info = MakeHostInfo();
        Vulkan::ShaderPools pools;
        const auto translate = [&](Shader::Environment& env) {
            const u32 cfg_offset =
                env.StartAddress() +
                (env.ShaderStage() == Shader::Stage::Compute ? 0 : sizeof(Shader::ProgramHeader));
            Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset);
            std::string dump = Shader::IR::DumpProgram(
                Shader::Maxwell::TranslateProgram(pools.inst, pools.block, env, cfg, host_info));
            pools.ReleaseContents();
            return dump;
        };
        struct Variant {
            std::string_view name;
            RealParts parts;
            size_t same = 0;
            size_t different = 0;
            size_t failed = 0;
        };
        std::array<Variant, 4> variants{{
            {"all guessed", {}},
            {"real constant buffers", {.cbuf = true}},
            {"real textures", {.textures = true}},
            {"real everything through the guess path", {.cbuf = true, .textures = true, .misc = true}},
        }};
        Shader::Optimization::SetDriverConstBufferFolding(true);
        size_t real_failed = 0;
        size_t logged = 0;
        for (const auto& shader : shaders) {
            if (!shader.env) {
                continue;
            }
            std::string real;
            try {
                real = translate(*shader.env);
            } catch (const std::exception&) {
                pools.ReleaseContents();
                ++real_failed;
                continue;
            }
            for (auto& variant : variants) {
                try {
                    GuessEnvironment guess{shader.code,     shader.stage, shader.env->StartAddress(),
                                           texture_bound,   driver_constants, shader.env.get(),
                                           variant.parts};
                    if (translate(guess) == real) {
                        ++variant.same;
                    } else {
                        ++variant.different;
                        if (&variant == &variants[0] && logged++ < 5) {
                            LOG_INFO(Frontend,
                                     "[ShaderScan] guess differs for {:016x} stage {}: {} "
                                     "textures, {} constant buffer reads",
                                     shader.unique_hash, static_cast<u32>(shader.stage),
                                     shader.texture_types.size(), shader.cbuf_value_count);
                        }
                    }
                } catch (const std::exception&) {
                    pools.ReleaseContents();
                    ++variant.failed;
                }
            }
        }
        for (const auto& variant : variants) {
            LOG_INFO(Frontend,
                     "[ShaderScan] {} vs runtime on cached shaders: {} identical, {} different, "
                     "{} failed",
                     variant.name, variant.same, variant.different, variant.failed);
        }
        LOG_INFO(Frontend, "[ShaderScan] runtime translation failed for {} cached shaders",
                 real_failed);

        // Without baking constant buffer 1 into the shader, translation no longer depends on
        // values only known while the game runs.
        Shader::Optimization::SetDriverConstBufferFolding(false);
        size_t unfolded_same = 0;
        size_t unfolded_different = 0;
        size_t unfolded_failed = 0;
        size_t changed_by_folding = 0;
        for (const auto& shader : shaders) {
            if (!shader.env) {
                continue;
            }
            try {
                const std::string real = translate(*shader.env);
                VideoCommon::PreforgedEnvironment guess{shader.code, shader.stage,
                                                        shader.env->StartAddress(), texture_bound};
                if (translate(guess) == real) {
                    ++unfolded_same;
                } else {
                    ++unfolded_different;
                }
                Shader::Optimization::SetDriverConstBufferFolding(true);
                const std::string folded = translate(*shader.env);
                Shader::Optimization::SetDriverConstBufferFolding(false);
                if (folded != real) {
                    ++changed_by_folding;
                }
            } catch (const std::exception&) {
                Shader::Optimization::SetDriverConstBufferFolding(false);
                pools.ReleaseContents();
                ++unfolded_failed;
            }
        }
        Shader::Optimization::SetDriverConstBufferFolding(false);
        LOG_INFO(Frontend,
                 "[ShaderScan] without constant buffer 1 folding, all guessed vs runtime: {} "
                 "identical, {} different, {} failed; folding changes {} runtime shaders",
                 unfolded_same, unfolded_different, unfolded_failed, changed_by_folding);
    }

    // Checks how the game's files keep the vertex and fragment shaders that the played
    // pipelines combine, to learn which shaders to pair when compiling ahead of time.
    void ReportPairing(const std::vector<Vulkan::GraphicsPipelineCacheKey>& keys) const {
        struct Occurrence {
            u32 file;
            size_t index;
            u64 offset;
        };
        std::unordered_map<u64, std::vector<Occurrence>> occurrences;
        std::vector<std::vector<HeaderInfo>> ordered(headers.size());
        std::vector<std::vector<u64>> offsets(headers.size());
        for (u32 file = 0; file < headers.size(); ++file) {
            for (const auto& [offset, header] : headers[file]) {
                occurrences[header.hash].push_back({file, ordered[file].size(), offset});
                ordered[file].push_back(header);
                offsets[file].push_back(offset);
            }
        }
        std::map<std::string, size_t> layout;
        std::unordered_set<u64> pairs;
        std::unordered_set<u64> shader_states;
        std::unordered_map<u64, std::unordered_set<u64>> states_per_pair;
        size_t vertex_fragment = 0;
        size_t other_layouts = 0;
        size_t both_extracted = 0;
        size_t same_file = 0;
        size_t next_fragment = 0;
        size_t previous_vertex = 0;
        size_t logged_same = 0;
        size_t logged_apart = 0;
        for (const auto& key : keys) {
            const u64 vertex = key.unique_hashes[1];
            const u64 fragment = key.unique_hashes[5];
            const bool others = key.unique_hashes[0] != 0 || key.unique_hashes[2] != 0 ||
                                key.unique_hashes[3] != 0 || key.unique_hashes[4] != 0;
            if (vertex == 0 || fragment == 0 || others) {
                ++other_layouts;
                continue;
            }
            ++vertex_fragment;
            const u64 pair = Mix(vertex) ^ fragment;
            pairs.insert(pair);
            const u64 shader_state = Vulkan::PipelineCache::ShaderStateHash(key);
            shader_states.insert(shader_state);
            states_per_pair[pair].insert(shader_state);
            const auto vertex_it = occurrences.find(vertex);
            const auto fragment_it = occurrences.find(fragment);
            if (vertex_it == occurrences.end() || fragment_it == occurrences.end()) {
                continue;
            }
            ++both_extracted;
            std::optional<std::pair<Occurrence, Occurrence>> best;
            for (const Occurrence& v : vertex_it->second) {
                for (const Occurrence& f : fragment_it->second) {
                    if (v.file != f.file) {
                        continue;
                    }
                    const auto distance = [](const Occurrence& a, const Occurrence& b) {
                        return a.index > b.index ? a.index - b.index : b.index - a.index;
                    };
                    if (!best || distance(v, f) < distance(best->first, best->second)) {
                        best = std::pair{v, f};
                    }
                }
            }
            if (!best) {
                if (logged_apart++ < 5) {
                    LOG_INFO(Frontend,
                             "[ShaderScan] pair {:016x}+{:016x} in different files: {} and {}",
                             vertex, fragment, file_names[vertex_it->second[0].file],
                             file_names[fragment_it->second[0].file]);
                }
                continue;
            }
            ++same_file;
            const auto& [v, f] = *best;
            const auto diff = static_cast<std::ptrdiff_t>(f.index) -
                              static_cast<std::ptrdiff_t>(v.index);
            ++layout[diff >= -4 && diff <= 4 ? fmt::format("{:+d}", diff)
                                             : (diff > 0 ? ">+4" : "<-4")];
            const auto& list = ordered[v.file];
            for (size_t i = v.index + 1; i < list.size(); ++i) {
                if (list[i].stage == Shader::Stage::Fragment) {
                    next_fragment += list[i].hash == fragment ? 1 : 0;
                    break;
                }
            }
            for (size_t i = f.index; i-- > 0;) {
                if (list[i].stage == Shader::Stage::VertexB) {
                    previous_vertex += list[i].hash == vertex ? 1 : 0;
                    break;
                }
            }
            if (logged_same++ < 10) {
                LOG_INFO(Frontend,
                         "[ShaderScan] pair {:016x}+{:016x} in {}: vertex #{} at 0x{:x}, "
                         "fragment #{} at 0x{:x}",
                         vertex, fragment, file_names[v.file], v.index, v.offset, f.index,
                         f.offset);
            }
        }
        size_t from_programs = 0;
        for (const auto& key : keys) {
            from_programs += program_pairs.contains({key.unique_hashes[1], key.unique_hashes[5]});
        }
        std::unordered_set<u64> program_vertex;
        std::unordered_set<u64> program_fragment;
        for (const auto& [vertex, fragment] : program_pairs) {
            program_vertex.insert(vertex);
            program_fragment.insert(fragment);
        }
        LOG_INFO(Frontend,
                 "[ShaderScan] programs: {} containers searched, {} pairs from program tables "
                 "({} vertex, {} fragment shaders), {} of {} played pipelines among them",
                 program_buffers, program_pairs.size(), program_vertex.size(),
                 program_fragment.size(), from_programs, keys.size());
        LOG_INFO(Frontend,
                 "[ShaderScan] program tables: pointer bases {}, record fields {}, fragment slots {}",
                 fmt::join(pointer_bases, ","), fmt::join(slot_fields, ","),
                 fmt::join(fragment_slots, ","));
        size_t multi_state_pairs = 0;
        for (const auto& [pair, states] : states_per_pair) {
            multi_state_pairs += states.size() > 1 ? 1 : 0;
        }
        LOG_INFO(Frontend,
                 "[ShaderScan] pipelines: {} vertex+fragment, {} other layouts; {} distinct "
                 "pairs, {} distinct shader states, {} pairs used with several states",
                 vertex_fragment, other_layouts, pairs.size(), shader_states.size(),
                 multi_state_pairs);
        LOG_INFO(Frontend,
                 "[ShaderScan] pairing: {} with both shaders extracted, {} in one file, {} where "
                 "the fragment is the next one after the vertex, {} where the vertex is the "
                 "last one before the fragment",
                 both_extracted, same_file, next_fragment, previous_vertex);
        for (const auto& [diff, count] : layout) {
            LOG_INFO(Frontend, "[ShaderScan] fragment position relative to vertex {}: {}", diff,
                     count);
        }
        // Shows the order of stages in the file with the most code headers.
        size_t biggest = 0;
        for (size_t file = 0; file < ordered.size(); ++file) {
            if (ordered[file].size() > ordered[biggest].size()) {
                biggest = file;
            }
        }
        if (!ordered.empty()) {
            std::string stages;
            for (size_t i = 0; i < std::min<size_t>(ordered[biggest].size(), 120); ++i) {
                stages += ordered[biggest][i].stage == Shader::Stage::VertexB    ? 'V'
                          : ordered[biggest][i].stage == Shader::Stage::Fragment ? 'F'
                                                                                 : '?';
            }
            LOG_INFO(Frontend, "[ShaderScan] stage order in {} ({} headers): {}",
                     file_names[biggest], ordered[biggest].size(), stages);
        }
    }

    static Shader::HostTranslateInfo MakeHostInfo() {
        Shader::HostTranslateInfo host_info{
            .support_float64 = false,
            .support_float16 = true,
            .support_int64 = true,
            .support_snorm_render_buffer = true,
        };
        host_info.ApplyDescriptorLimitPolicy();
        return host_info;
    }

    void TranslateExtracted() const {
        const Shader::HostTranslateInfo host_info = MakeHostInfo();
        Vulkan::ShaderPools pools;
        std::map<u32, std::pair<size_t, size_t>> by_stage;
        std::map<std::string, size_t> errors;
        const auto start = std::chrono::steady_clock::now();
        size_t attempted = 0;
        for (const auto& [hash, shader] : extracted) {
            if (std::chrono::steady_clock::now() - start > TRANSLATE_TIME_LIMIT) {
                break;
            }
            ++attempted;
            auto& counts = by_stage[static_cast<u32>(shader.stage)];
            ++counts.first;
            try {
                GuessEnvironment env{shader.code, shader.stage, shader.start_address, texture_bound,
                                     driver_constants};
                const u32 cfg_offset =
                    shader.start_address +
                    (shader.stage == Shader::Stage::Compute ? 0 : sizeof(Shader::ProgramHeader));
                Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset);
                void(Shader::Maxwell::TranslateProgram(pools.inst, pools.block, env, cfg,
                                                       host_info));
                ++counts.second;
            } catch (const std::exception& e) {
                ++errors[std::string(e.what()).substr(0, 80)];
            }
            pools.ReleaseContents();
        }
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);
        LOG_INFO(Frontend, "[ShaderScan] translated {} of {} extracted shaders in {} ms",
                 attempted, extracted.size(), ms.count());
        for (const auto& [stage, counts] : by_stage) {
            LOG_INFO(Frontend, "[ShaderScan] translate stage {}: {} of {} succeeded", stage,
                     counts.second, counts.first);
        }
        std::vector<std::pair<size_t, std::string>> sorted;
        for (const auto& [error, count] : errors) {
            sorted.emplace_back(count, error);
        }
        std::sort(sorted.rbegin(), sorted.rend());
        for (size_t i = 0; i < std::min<size_t>(sorted.size(), 5); ++i) {
            LOG_INFO(Frontend, "[ShaderScan] translate error x{}: {}", sorted[i].first,
                     sorted[i].second);
        }
    }

    // Reads the Unreal package that holds the most shader code once more, to log how its objects
    // refer to the code. Runs between the two passes, so the second searches the other packages
    // for what it learned.
    void StudyUnrealLinks(const std::vector<Vulkan::GraphicsPipelineCacheKey>& keys) {
        if (!links || links_studied) {
            return;
        }
        std::string best;
        size_t most = 0;
        for (const auto& [name, file] : unreal_code_files) {
            const auto it = file_ids.find(name + UNPACKED_SUFFIX);
            if (it != file_ids.end() && headers[it->second].size() > most) {
                most = headers[it->second].size();
                best = name;
            }
        }
        if (best.empty()) {
            return;
        }
        links_studied = true;
        const auto start = std::chrono::steady_clock::now();
        std::vector<Preforged::Unreal::LinkSurvey::CodeHeader> code;
        using Stage = Preforged::Unreal::LinkSurvey::Stage;
        for (const auto& [offset, header] : headers[file_ids.at(best + UNPACKED_SUFFIX)]) {
            code.push_back({offset, header.hash,
                            header.stage == Shader::Stage::VertexB    ? Stage::Vertex
                            : header.stage == Shader::Stage::Fragment ? Stage::Fragment
                                                                      : Stage::Other});
        }
        std::vector<std::pair<u64, u64>> played;
        for (const auto& key : keys) {
            if (key.unique_hashes[1] != 0 && key.unique_hashes[5] != 0) {
                played.emplace_back(key.unique_hashes[1], key.unique_hashes[5]);
            }
        }
        const std::vector<u8> data = unreal_code_files.at(best)->ReadAllBytes();
        links->StudyCodePackage(best + UNPACKED_SUFFIX, data, code, played,
                                [this] { return Stopped(); });
        LOG_INFO(Frontend, "[ShaderScan] Unreal links: read {} in {} ms", best,
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - start)
                     .count());
        // Pairs the played pairs taught, for compiling them ahead of time.
        const auto& learned = links->LearnedPairs();
        if (learned.empty()) {
            return;
        }
        if (!Common::Preforged::Options().unreal_learned_pairs) {
            LOG_INFO(Frontend,
                     "[ShaderScan] Unreal links: {} learned pairs not kept, the switch "
                     "\"Pre-compile Unreal pairs learned from played ones\" is off",
                     learned.size());
            return;
        }
        const size_t before = program_pairs.size();
        program_pairs.insert(learned.begin(), learned.end());
        LOG_INFO(Frontend,
                 "[ShaderScan] Unreal links: {} learned pairs kept for pre-compiling ({} new)",
                 learned.size(), program_pairs.size() - before);
    }

    // Reads the shader caches of the Unreal packages searched so far, once the code package is
    // read, and keeps the pairs they teach for compiling them ahead of time.
    void StudyUnrealCaches() {
        if (!links || !links_studied) {
            return;
        }
        for (const auto& [name, file] : unreal_cache_files) {
            if (stop.stop_requested() || !unreal_caches_read.insert(name).second) {
                continue;
            }
            const std::vector<u8> data = file->ReadAllBytes();
            links->StudyShaderCache(name, data, [this] { return stop.stop_requested(); });
        }
        const auto& learned = links->LearnedFromShaderCache();
        if (learned.empty() || unreal_cache_pairs_kept) {
            return;
        }
        unreal_cache_pairs_kept = true;
        if (!Common::Preforged::Options().unreal_learned_pairs) {
            LOG_INFO(Frontend,
                     "[ShaderScan] Unreal links: {} pairs learned from the shader cache not kept, "
                     "the switch \"Pre-compile Unreal pairs learned from played ones\" is off",
                     learned.size());
            return;
        }
        const size_t before = program_pairs.size();
        program_pairs.insert(learned.begin(), learned.end());
        LOG_INFO(Frontend,
                 "[ShaderScan] Unreal links: {} pairs learned from the shader cache kept for "
                 "pre-compiling ({} new)",
                 learned.size(), program_pairs.size() - before);
    }

    void ReportUnrealLinks() const {
        if (links) {
            links->Report();
        }
    }

    // Pairs that ids found in the other Unreal packages tie together, for compiling them ahead
    // of time. Runs after all packages are searched.
    void LearnUnrealLinksFromOtherPackages() {
        if (!links || !links_studied) {
            return;
        }
        const auto& learned = links->LearnFromOtherPackages();
        if (learned.empty()) {
            return;
        }
        if (!Common::Preforged::Options().unreal_learned_pairs) {
            LOG_INFO(Frontend,
                     "[ShaderScan] Unreal links: {} pairs learned from other packages not kept, "
                     "the switch \"Pre-compile Unreal pairs learned from played ones\" is off",
                     learned.size());
            return;
        }
        const size_t before = program_pairs.size();
        program_pairs.insert(learned.begin(), learned.end());
        LOG_INFO(Frontend,
                 "[ShaderScan] Unreal links: {} pairs learned from other packages kept for "
                 "pre-compiling ({} new)",
                 learned.size(), program_pairs.size() - before);
    }

    // Keeps the shader pairs from the game's program tables, for compiling them ahead of time.
    void SavePrograms(const std::filesystem::path& path) const {
        VideoCommon::PreforgedPrograms programs;
        programs.texture_bound = texture_bound;
        for (const auto& [vertex, fragment] : program_pairs) {
            const auto vertex_it = extracted.find(vertex);
            const auto fragment_it = extracted.find(fragment);
            if (vertex_it == extracted.end() || fragment_it == extracted.end()) {
                continue;
            }
            for (const auto& [hash, shader] : {*vertex_it, *fragment_it}) {
                programs.shaders.try_emplace(hash, VideoCommon::PreforgedShader{
                                                       shader.stage, shader.start_address,
                                                       shader.code});
            }
            programs.pairs.emplace_back(vertex, fragment);
        }
        for (const auto& [hash, shader] : extracted) {
            if (!programs.shaders.contains(hash)) {
                programs.other_shaders.push_back(hash);
            }
        }
        const bool written = VideoCommon::WritePreforgedPrograms(path, programs);
        LOG_INFO(Frontend,
                 "[ShaderScan] saved {} shader pairs with {} shaders, and the hashes of the {} "
                 "other shaders taken from the game files: {}",
                 programs.pairs.size(), programs.shaders.size(), programs.other_shaders.size(),
                 written ? "ok" : "failed");
    }

    bool Stopped() const {
        return stop.stop_requested() || std::chrono::steady_clock::now() > deadline;
    }

    bool AllFound() const {
        return std::ranges::all_of(results, [](const Result& result) {
            return result.kind == MatchKind::Exact || result.kind == MatchKind::CodeOnly;
        });
    }

private:
    struct Result {
        MatchKind kind = MatchKind::None;
        /// For a shader only its start was found of: the longest run of its code found from the
        /// start on, where, and the bytes where that run ends.
        size_t best_length = 0;
        u32 best_file = 0;
        u64 best_offset = 0;
        std::string best_difference;
    };

    struct FileStats {
        size_t hits = 0;
        size_t terminators = 0;
    };

    /// Blocks of one Unreal package that did not unpack.
    struct FailedPackage {
        size_t blocks = 0;
        u64 lost = 0; ///< Bytes they would have unpacked to
        std::map<u8, size_t> decoders;
        std::vector<Preforged::Unreal::FailedBlock> first;
    };

    static size_t NeedleOffset(const CachedShaderCode& shader) {
        return shader.stage == Shader::Stage::Compute ? 0 : SPH_SIZE;
    }

    static u64 NeedleKey(const u64* words) {
        u64 key = 0;
        for (size_t i = 0; i < NEEDLE_WORDS; ++i) {
            key = Mix(key ^ words[i]);
        }
        return key;
    }

    static bool EndsWith(std::string_view text, std::string_view suffix) {
        return text.size() >= suffix.size() &&
               text.substr(text.size() - suffix.size()) == suffix;
    }

    static std::string Lower(std::string_view text) {
        std::string out(text);
        std::ranges::transform(out, out.begin(), [](unsigned char c) { return std::tolower(c); });
        return out;
    }

    static bool LooksLikeShaderFile(std::string_view name) {
        const std::string lower = Lower(name);
        return lower.find("shader") != std::string::npos || lower.find("sha") != std::string::npos ||
               lower.find("bnsh") != std::string::npos || lower.find("nvn") != std::string::npos;
    }

    static bool IsMedia(std::string_view name) {
        const std::string lower = Lower(name);
        for (const std::string_view ext : {".bwav", ".bfstm", ".bfwav", ".bars", ".mp4", ".webm",
                                           ".usm", ".ogg", ".opus", ".wem", ".bnk", ".acb",
                                           ".awb"}) {
            if (EndsWith(lower, ext)) {
                return true;
            }
        }
        return false;
    }

    void ScanFile(const FileSys::VirtualFile& file, const std::string& name) {
        const size_t size = file->GetSize();
        if (size == 0) {
            return;
        }
        ++files_scanned;
        bool whole = size <= MAX_WHOLE_FILE;
        if (!whole && size <= MAX_WHOLE_PACKAGE &&
            Common::Preforged::Options().unpack_unreal_packages) {
            const std::vector<u8> start = file->ReadBytes(4, 0);
            whole = Preforged::Unreal::IsPackage(start);
        }
        if (whole) {
            const std::vector<u8> data = file->ReadAllBytes();
            bytes_scanned += data.size();
            ScanBuffer(data, name, 0);
            // Keeps Unreal packages that hold shader code, to read the one with the most again.
            if (links) {
                const auto it = file_ids.find(name + UNPACKED_SUFFIX);
                if (it != file_ids.end() && !headers[it->second].empty()) {
                    unreal_code_files.emplace(name, file);
                }
                // And those that hold a shader cache, to read once the code package is.
                if (links->HoldsShaderCache(name)) {
                    unreal_cache_files.emplace(name, file);
                }
            }
        } else {
            std::vector<u8> chunk;
            for (size_t offset = 0; offset < size && !Stopped();
                 offset += CHUNK_SIZE) {
                const size_t length = std::min(size - offset, CHUNK_SIZE + CHUNK_OVERLAP);
                chunk = file->ReadBytes(length, offset);
                bytes_scanned += std::min(length, CHUNK_SIZE);
                if (offset == 0) {
                    auto& large = formats.large_starts[Start32(chunk)];
                    ++large.first;
                    large.second += size;
                }
                SearchShaders(chunk, name, offset);
            }
        }
        LogProgress();
    }

    void ScanBuffer(std::span<const u8> data, const std::string& name, int depth) {
        if (Stopped() || depth > 4) {
            return;
        }
        if (std::vector<u8> decompressed = Decompress(data); !decompressed.empty()) {
            ScanBuffer(decompressed, name, depth + 1);
            return;
        }
        // Unreal Engine packages keep their contents in compressed parts. Search what they
        // unpack to, in pieces like large files.
        if (Common::Preforged::Options().unpack_unreal_packages &&
            Preforged::Unreal::IsPackage(data)) {
            const std::string unpacked_name = name + UNPACKED_SUFFIX;
            std::vector<Preforged::Unreal::FailedBlock> failures;
            const bool unpacked = Preforged::Unreal::Unpack(
                data, CHUNK_SIZE, CHUNK_OVERLAP, unreal_stats,
                [&](std::span<const u8> piece, u64 offset) {
                    if (links) {
                        if (offset == 0) {
                            links->AddPackage(name, data, piece);
                        }
                        links->SearchPiece(unpacked_name, piece, offset);
                    }
                    SearchShaders(piece, unpacked_name, offset);
                    return !Stopped();
                },
                Common::Preforged::Options().unpack_oodle, &failures);
            if (!failures.empty()) {
                auto& failed = failed_packages[unpacked_name];
                for (const auto& block : failures) {
                    ++failed.blocks;
                    failed.lost += block.unpacked;
                    ++failed.decoders[block.decoder];
                    if (failed.first.size() < FAILED_BLOCKS_LOGGED_PER_FILE) {
                        failed.first.push_back(block);
                    }
                }
            }
            if (unpacked) {
                return;
            }
            if (links && depth == 0) {
                links->AddPackage(name, data, data);
            }
        }
        if (data.size() >= 0x14 && std::memcmp(data.data(), "SARC", 4) == 0 &&
            ScanSarc(data, name, depth)) {
            return;
        }
        if (Common::Preforged::Options().zstd_dictionaries) {
            if (Common::Compression::IsZstdDictionary(data)) {
                AddDictionary(data, name);
            }
            // Packed bytes hold no shader code as it is, so searching them only costs time.
            // Decompress counted the file as naming a missing dictionary.
            if (IsZstd(data) && Common::Compression::ZstdFrameDictionaryId(data) != 0) {
                return;
            }
        }
        auto& stored = formats.stored_starts[Start32(data)];
        ++stored.first;
        stored.second += data.size();
        SearchShaders(data, name, 0);
    }

    // Walks the files inside a SARC archive so compressed entries are unpacked before searching.
    bool ScanSarc(std::span<const u8> data, const std::string& name, int depth) {
        std::vector<std::span<const u8>> entries;
        if (!SarcEntries(data, entries)) {
            return false;
        }
        for (size_t node = 0; node < entries.size(); ++node) {
            ScanBuffer(entries[node], fmt::format("{}#{}", name, node), depth + 1);
        }
        return true;
    }

    void SearchShaders(std::span<const u8> data, const std::string& name, size_t base_offset) {
        if (data.size() < NEEDLE_WORDS * sizeof(u64)) {
            return;
        }
        size_t hits = 0;
        size_t terminators = 0;
        const auto [file_it, inserted] = file_ids.try_emplace(name, file_names.size());
        if (inserted) {
            file_names.push_back(name);
            headers.emplace_back();
        }
        const u32 file_id = file_it->second;
        std::unordered_map<u64, HeaderInfo> local_headers;
        std::vector<u64> bnsh_starts;
        const size_t end = data.size() - NEEDLE_WORDS * sizeof(u64);
        for (size_t pos = 0; pos <= end; pos += 4) {
            if ((pos & 0xFFFFF) == 0 && Stopped()) {
                return;
            }
            const u64 first = Load64(data.data() + pos);
            if ((pos & 7) == 0 && (first == SELF_BRANCH_A || first == SELF_BRANCH_B)) {
                ++terminators;
            }
            if (static_cast<u32>(first) == NVN_CODE_MAGIC) {
                Extract(data, pos + NVN_CODE_HEADER_SIZE, file_id, base_offset + pos);
                const auto& file_headers = headers[file_id];
                if (const auto it = file_headers.find(base_offset + pos);
                    it != file_headers.end()) {
                    local_headers.emplace(pos, it->second);
                }
            }
            if ((static_cast<u32>(first) == BNSH_MAGIC || static_cast<u32>(first) == FSHA_MAGIC) &&
                (pos & 7) == 0) {
                bnsh_starts.push_back(pos);
            }
            const u64 mixed = Mix(first);
            if ((filter[(mixed & ((1u << FILTER_BITS) - 1)) / 64] & (u64{1} << (mixed % 64))) ==
                0) {
                continue;
            }
            std::array<u64, NEEDLE_WORDS> words;
            std::memcpy(words.data(), data.data() + pos, sizeof(words));
            const auto it = needles.find(NeedleKey(words.data()));
            if (it == needles.end()) {
                continue;
            }
            for (const size_t index : it->second) {
                const MatchKind kind = Verify(index, data, pos);
                auto& result = results[index];
                NotePartialMatch(index, data, pos, file_id, base_offset);
                if (kind > result.kind) {
                    if (result.kind < MatchKind::CodeOnly && kind >= MatchKind::CodeOnly) {
                        ++hits;
                        LogFirstMatches(index, kind, data, pos, name, base_offset);
                    }
                    result.kind = kind;
                }
            }
        }
        if (!local_headers.empty() && base_offset == 0) {
            FindPrograms(data, name, local_headers, bnsh_starts);
        }
        if (hits > 0) {
            auto& stats = hits_per_file[name];
            stats.hits += hits;
            stats.terminators += terminators;
        }
    }

    static std::string Hex(std::span<const u8> data, size_t start, size_t length) {
        std::string out;
        for (size_t i = start; i < std::min(data.size(), start + length); ++i) {
            out += fmt::format("{:02x}", data[i]);
        }
        return out;
    }

    // Shader containers (BNSH) describe programs as tables of pointers to per-stage shader
    // code. Follows those pointers back from each code header to learn which vertex and
    // fragment shaders the game combines.
    void FindPrograms(std::span<const u8> data, const std::string& name,
                      const std::unordered_map<u64, HeaderInfo>& local_headers,
                      const std::vector<u64>& bnsh_starts) {
        ++program_buffers;
        const auto base_of = [&](size_t pos) -> u64 {
            const auto it = std::upper_bound(bnsh_starts.begin(), bnsh_starts.end(), pos);
            return it == bnsh_starts.begin() ? 0 : *std::prev(it);
        };
        // Pointers to code headers, by where the pointer is stored.
        std::unordered_map<u64, HeaderInfo> code_pointers;
        for (size_t pos = 0; pos + 8 <= data.size(); pos += 8) {
            const u64 value = Load64(data.data() + pos);
            if (value == 0 || value >= data.size()) {
                continue;
            }
            const u64 base = base_of(pos);
            for (const u64 candidate_base : {base, u64{0}}) {
                const auto it = local_headers.find(candidate_base + value);
                if (it == local_headers.end()) {
                    continue;
                }
                code_pointers.emplace(pos, it->second);
                ++pointer_bases[candidate_base == 0 ? "buffer" : "container"];
                if (pointer_examples_logged < 3) {
                    ++pointer_examples_logged;
                    LOG_INFO(Frontend,
                             "[ShaderScan] pointer to code header in {} at 0x{:x} (container "
                             "0x{:x}): {}",
                             name, pos, base, Hex(data, pos - std::min<size_t>(pos, 0x40), 0x80));
                }
                break;
            }
        }
        if (code_pointers.empty()) {
            if (pointer_misses_logged < 2) {
                ++pointer_misses_logged;
                const u64 first_header = std::min_element(local_headers.begin(), local_headers.end(),
                                                          [](const auto& a, const auto& b) {
                                                              return a.first < b.first;
                                                          })->first;
                LOG_INFO(Frontend,
                         "[ShaderScan] no pointers to code headers in {}; container at 0x{:x}: "
                         "{}; first code header at 0x{:x}: {}",
                         name, bnsh_starts.empty() ? 0 : bnsh_starts[0],
                         Hex(data, bnsh_starts.empty() ? 0 : bnsh_starts[0], 0x100), first_header,
                         Hex(data, first_header - std::min<u64>(first_header, 0x40), 0x80));
            }
            return;
        }
        // Pointers to the records holding those pointers, which make up each program's table.
        std::unordered_map<u64, HeaderInfo> stage_slots;
        for (size_t pos = 0; pos + 8 <= data.size(); pos += 8) {
            const u64 value = Load64(data.data() + pos);
            if (value == 0 || value >= data.size()) {
                continue;
            }
            const u64 base = base_of(pos);
            bool found = false;
            for (const u64 candidate_base : {base, u64{0}}) {
                for (u64 field = 0; field <= 0x18 && !found; field += 8) {
                    const auto it = code_pointers.find(candidate_base + value + field);
                    if (it == code_pointers.end()) {
                        continue;
                    }
                    stage_slots.emplace(pos, it->second);
                    ++slot_fields[field];
                    found = true;
                }
                if (found) {
                    break;
                }
            }
        }
        for (const auto& [pos, header] : stage_slots) {
            if (header.stage != Shader::Stage::VertexB) {
                continue;
            }
            // Programs list their stages as vertex, hull, domain, geometry, pixel, compute.
            for (u64 slot = 1; slot <= 5; ++slot) {
                const auto it = stage_slots.find(pos + slot * 8);
                if (it != stage_slots.end() && it->second.stage == Shader::Stage::Fragment) {
                    if (slot == 4) {
                        program_pairs.insert({header.hash, it->second.hash});
                    }
                    ++fragment_slots[slot];
                    break;
                }
            }
        }
    }

    // Takes the shader that follows an NVN code header, using the same end marker and hash as the
    // emulator uses for shaders the game uploads.
    void Extract(std::span<const u8> data, size_t start, u32 file_id, u64 file_offset) {
        ++code_headers;
        if (start + SPH_SIZE + sizeof(u64) > data.size() || extracted_bytes > MAX_EXTRACTED_BYTES) {
            return;
        }
        Shader::ProgramHeader sph;
        std::memcpy(&sph, data.data() + start, sizeof(sph));
        const auto header_stage = StageFromHeader(sph);
        const Shader::Stage stage = header_stage.value_or(Shader::Stage::Compute);
        const size_t limit = std::min(data.size(), start + MAX_SHADER_SIZE);
        std::optional<size_t> exit_size;
        for (size_t pos = start; pos + sizeof(u64) <= limit; pos += sizeof(u64)) {
            const u64 inst = Load64(data.data() + pos);
            if (texture_bound != 2 && inst == EXIT_VALUE && !exit_size) {
                exit_size = pos - start + sizeof(u64);
            }
            if (inst != SELF_BRANCH_A && inst != SELF_BRANCH_B) {
                continue;
            }
            const size_t size = pos - start;
            if (size == 0) {
                return;
            }
            const auto* const bytes = reinterpret_cast<const char*>(data.data() + start);
            self_branch_hashes.insert(Common::CityHash64(bytes, size));
            // Like the emulator: unless the game uses the official driver's texture slot, the
            // shader ends after its first exit instruction.
            const size_t hashed = exit_size.value_or(size);
            const u64 hash = Common::CityHash64(bytes, hashed);
            headers[file_id].emplace(file_offset, HeaderInfo{hash, stage});
            if (extracted.contains(hash)) {
                return;
            }
            std::vector<u64> code(hashed / sizeof(u64) + 1);
            std::memcpy(code.data(), data.data() + start,
                        std::min(code.size() * sizeof(u64), data.size() - start));
            extracted_bytes += code.size() * sizeof(u64);
            // Game code places the program header 0x30 bytes after the code header.
            const u32 start_address = header_stage ? 0x30 : 0;
            extracted.emplace(hash, ExtractedShader{stage, start_address, std::move(code)});
            return;
        }
    }

    /// Keeps the longest run of a shader's code found from its start on, and where, to tell
    /// in the log why a drawn shader is not among the shaders taken from the game files.
    void NotePartialMatch(size_t index, std::span<const u8> data, size_t pos, u32 file_id,
                          u64 base_offset) {
        const auto& shader = shaders[index];
        const size_t needle_offset = NeedleOffset(shader);
        const size_t code_bytes = shader.code.size() * sizeof(u64);
        if (code_bytes <= needle_offset || pos < needle_offset) {
            return;
        }
        const auto* const code = reinterpret_cast<const u8*>(shader.code.data());
        const size_t tail = std::min(code_bytes - needle_offset, data.size() - pos);
        size_t length = 0;
        while (length < tail && data[pos + length] == code[needle_offset + length]) {
            ++length;
        }
        auto& result = results[index];
        if (length <= result.best_length) {
            return;
        }
        result.best_length = length;
        result.best_file = file_id;
        result.best_offset = base_offset + pos;
        const size_t at = (needle_offset + length) & ~size_t{7};
        std::string runtime;
        std::string file;
        for (size_t i = at; i < std::min(code_bytes, at + 16); ++i) {
            runtime += fmt::format("{:02x}", code[i]);
        }
        const size_t file_at = pos + at - needle_offset;
        for (size_t i = file_at; i < std::min(data.size(), file_at + 16); ++i) {
            file += fmt::format("{:02x}", data[i]);
        }
        result.best_difference =
            length == code_bytes - needle_offset
                ? fmt::format("all of it (file goes on with {})", file)
                : fmt::format("up to code +0x{:x}: drawn {} vs file {}", at, runtime, file);
    }

    MatchKind Verify(size_t index, std::span<const u8> data, size_t pos) const {
        const auto& shader = shaders[index];
        const size_t needle_offset = NeedleOffset(shader);
        const size_t code_bytes = shader.code.size() * sizeof(u64);
        const auto* const code = reinterpret_cast<const u8*>(shader.code.data());
        const size_t tail = code_bytes - needle_offset;
        if (pos + tail > data.size() ||
            std::memcmp(data.data() + pos, code + needle_offset, tail) != 0) {
            return MatchKind::Needle;
        }
        if (pos < needle_offset ||
            std::memcmp(data.data() + pos - needle_offset, code, needle_offset) != 0) {
            return MatchKind::CodeOnly;
        }
        return MatchKind::Exact;
    }

    void LogFirstMatches(size_t index, MatchKind kind, std::span<const u8> data, size_t pos,
                         const std::string& name, size_t base_offset) {
        const auto& shader = shaders[index];
        if (logged_per_stage[static_cast<u32>(shader.stage)]++ >= 3) {
            return;
        }
        const size_t start = pos - std::min(pos, NeedleOffset(shader) + 0x100);
        std::string bytes;
        for (size_t i = start; i < pos + 16 && i < data.size(); ++i) {
            bytes += fmt::format("{:02x}", data[i]);
        }
        LOG_INFO(Frontend,
                 "[ShaderScan] match {:016x} stage {} {} in {} at 0x{:x}, bytes from 0x{:x}: {}",
                 shader.unique_hash, static_cast<u32>(shader.stage), KindName(kind), name,
                 base_offset + pos, base_offset + start, bytes);
    }

    void LogProgress() {
        const auto now = std::chrono::steady_clock::now();
        if (now - last_progress < std::chrono::seconds{30}) {
            return;
        }
        last_progress = now;
        LOG_INFO(Frontend, "[ShaderScan] progress: {} files, {} MiB read", files_scanned,
                 bytes_scanned >> 20);
    }

    std::vector<CachedShaderCode> shaders;
    std::stop_token stop;
    std::chrono::steady_clock::time_point deadline;
    std::vector<u64> filter;
    std::unordered_map<u64, std::vector<size_t>> needles;
    std::vector<Result> results;
    std::map<std::string, FileStats> hits_per_file;
    std::map<std::string, FailedPackage> failed_packages;
    size_t files_scanned = 0;
    size_t bytes_scanned = 0;
    std::map<u32, size_t> logged_per_stage;
    std::unordered_map<u64, ExtractedShader> extracted;
    std::unordered_set<u64> self_branch_hashes;
    std::unordered_map<std::string, u32> file_ids;
    std::vector<std::string> file_names;
    /// Code headers of each file by offset, in the order the game stores them.
    std::vector<std::map<u64, HeaderInfo>> headers;
    const std::function<void(size_t, size_t)>* progress_callback = nullptr;
    size_t total_files = 1;
    size_t files_visited = 0;
    std::chrono::steady_clock::time_point last_progress_report{};
    size_t program_buffers = 0;
    size_t pointer_examples_logged = 0;
    size_t pointer_misses_logged = 0;
    std::map<std::string, size_t> pointer_bases;
    std::map<u64, size_t> slot_fields;
    std::map<u64, size_t> fragment_slots;
    std::set<std::pair<u64, u64>> program_pairs;
    u32 texture_bound = 2;
    DriverConstants driver_constants;
    size_t extracted_bytes = 0;
    size_t code_headers = 0;
    Preforged::Unreal::Stats unreal_stats;
    Common::Compression::ZstdDictionaries zstd_dictionaries;
    /// What the search unpacked, and what it searched as stored because it could not.
    struct FormatStats {
        size_t yaz0 = 0;
        size_t zstd_plain = 0;
        std::map<u32, size_t> zstd_with_dictionary;
        /// zstd frames that name a dictionary the game files don't have, by its id.
        std::map<u32, size_t> zstd_missing_dictionary;
        /// zstd frames that did not unpack otherwise: damaged, too large, or packed with a
        /// dictionary they don't name.
        size_t zstd_failed = 0;
        std::vector<std::string> zstd_failed_examples;
        /// Buffers searched as they are, by their first 4 bytes: how many, and their bytes.
        std::map<u32, std::pair<size_t, u64>> stored_starts;
        /// Files too large to read whole, searched in pieces as they are, by their first 4 bytes.
        std::map<u32, std::pair<size_t, u64>> large_starts;
    } formats;
    std::unique_ptr<Preforged::Unreal::LinkSurvey> links;
    bool links_studied = false;
    std::map<std::string, FileSys::VirtualFile> unreal_code_files;
    std::map<std::string, FileSys::VirtualFile> unreal_cache_files;
    std::set<std::string> unreal_caches_read;
    bool unreal_cache_pairs_kept = false;
    std::chrono::steady_clock::time_point last_progress = std::chrono::steady_clock::now();
};

/// Lists every file the game sees with its size, sorted by path, one per line. An update or a
/// mod changes the list.
void ListGameFiles(const FileSys::VirtualDir& dir, const std::string& path, std::string& out,
                   size_t& count) {
    if (!dir) {
        return;
    }
    std::vector<std::pair<std::string, size_t>> files;
    for (const auto& file : dir->GetFiles()) {
        files.emplace_back(file->GetName(), file->GetSize());
    }
    std::sort(files.begin(), files.end());
    for (const auto& [name, size] : files) {
        out += fmt::format("{}/{} {}\n", path, name, size);
    }
    count += files.size();
    std::vector<FileSys::VirtualDir> subdirs = dir->GetSubdirectories();
    std::sort(subdirs.begin(), subdirs.end(),
              [](const auto& a, const auto& b) { return a->GetName() < b->GetName(); });
    for (const auto& subdir : subdirs) {
        ListGameFiles(subdir, path + "/" + subdir->GetName(), out, count);
    }
}

/// What the last search of a game left in its shader folder.
struct SearchMarker {
    std::string files_id; ///< Id of the game file list it searched
    u32 version = 0;      ///< SEARCH_VERSION of the search
    std::string result;   ///< What it found, from ResultSummary
};

constexpr std::string_view MARKER_NAME = "shader_scan_done";
/// Before version 11 the marker held only the id of the game file list.
constexpr std::string_view OLD_MARKER_NAME = "shader_scan_done_v10";

std::optional<SearchMarker> ReadSearchMarker(const std::filesystem::path& shader_dir) {
    std::error_code error;
    const auto path = shader_dir / MARKER_NAME;
    if (std::filesystem::exists(path, error)) {
        SearchMarker marker;
        const std::string text =
            Common::FS::ReadStringFromFile(path, Common::FS::FileType::TextFile);
        size_t start = 0;
        while (start < text.size()) {
            size_t end = text.find('\n', start);
            if (end == std::string::npos) {
                end = text.size();
            }
            const std::string_view line{text.data() + start, end - start};
            start = end + 1;
            const size_t space = line.find(' ');
            if (space == std::string_view::npos) {
                continue;
            }
            const std::string_view key = line.substr(0, space);
            const std::string_view value = line.substr(space + 1);
            if (key == "files") {
                marker.files_id = value;
            } else if (key == "version") {
                std::from_chars(value.data(), value.data() + value.size(), marker.version);
            } else if (key == "result") {
                marker.result = value;
            }
        }
        return marker;
    }
    const auto old_path = shader_dir / OLD_MARKER_NAME;
    if (std::filesystem::exists(old_path, error)) {
        return SearchMarker{
            .files_id = Common::FS::ReadStringFromFile(old_path, Common::FS::FileType::TextFile),
            .version = 10,
            .result = "not kept by that version",
        };
    }
    return std::nullopt;
}

void WriteSearchMarker(const std::filesystem::path& shader_dir, const SearchMarker& marker) {
    void(Common::FS::WriteStringToFile(
        shader_dir / MARKER_NAME, Common::FS::FileType::TextFile,
        fmt::format("files {}\nversion {}\nresult {}\n", marker.files_id, marker.version,
                    marker.result)));
    std::error_code error;
    std::filesystem::remove(shader_dir / OLD_MARKER_NAME, error);
}

} // namespace

void RunShaderFileScan(Core::System& system, u64 program_id, std::stop_token stop,
                       const std::function<void(std::size_t, std::size_t)>& progress) {
    const auto start_time = std::chrono::steady_clock::now();
    const auto shader_dir = Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir) /
                            fmt::format("{:016x}", program_id);
    // The marker holds an id of the game files the search read and the search's version, so it
    // runs again when an update or a mod changes the files or when the search improves.
    const auto done_marker = shader_dir / MARKER_NAME;
    std::error_code error;
    const auto& options = Common::Preforged::Options();
    if (!options.scan_game_files) {
        LOG_INFO(Frontend, "[ShaderScan] switched off in the settings");
        return;
    }

    FileSys::RomFSFactory factory{system.GetAppLoader(), system.GetContentProvider(),
                                  system.GetFileSystemController()};
    const FileSys::VirtualFile romfs = factory.OpenCurrentProcess(program_id);
    const FileSys::VirtualDir root = romfs ? FileSys::ExtractRomFS(romfs) : nullptr;
    if (!root) {
        LOG_INFO(Frontend, "[ShaderScan] could not open the game's RomFS");
        return;
    }
    std::string file_list;
    size_t file_count = 0;
    ListGameFiles(root, "", file_list, file_count);
    const std::string files_id =
        fmt::format("{:016x}", Common::CityHash64(file_list.data(), file_list.size()));
    LOG_INFO(Frontend, "[ShaderScan] game files: {} files, id {}, opened and listed in {} ms",
             file_count, files_id,
             std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - start_time)
                 .count());
    if (const auto last = ReadSearchMarker(shader_dir)) {
        if (options.rescan_game_files) {
            LOG_INFO(Frontend,
                     "[ShaderScan] 'Search game files on every start' is on, so it runs again "
                     "(last search, version {}: {})",
                     last->version, last->result);
        } else if (last->files_id != files_id) {
            LOG_INFO(Frontend,
                     "[ShaderScan] the game files changed since the last search (id {}), for "
                     "example by an update or a mod, so it runs again",
                     last->files_id.empty() ? std::string{"none"} : last->files_id);
        } else if (last->version < SEARCH_VERSION) {
            LOG_INFO(Frontend,
                     "[ShaderScan] the search improved since it last ran for this game (version "
                     "{} then, {} now), so it runs once more; last result: {}",
                     last->version, SEARCH_VERSION, last->result);
        } else {
            LOG_INFO(Frontend,
                     "[ShaderScan] already ran for these game files with this search version "
                     "({}); last result: {}. Turn on 'Search game files on every start' or "
                     "delete {} to run it again",
                     last->version, last->result, Common::FS::PathToUTF8String(done_marker));
            return;
        }
    }

    // Read from a copy, because a damaged cache file is deleted while it is read.
    const auto copy = shader_dir / "shader_scan_copy.bin";
    std::filesystem::copy_file(shader_dir / "vulkan.bin", copy,
                               std::filesystem::copy_options::overwrite_existing, error);
    if (error) {
        LOG_INFO(Frontend,
                 "[ShaderScan] not searching on this start: the game has no shader cache yet "
                 "({}), as on its first start or after the cache was deleted. The search "
                 "compares the game files with the shaders the game drew and compiles them with "
                 "the states it drew them with, so it runs on the next start.",
                 error.message());
        return;
    }
    std::vector<Vulkan::GraphicsPipelineCacheKey> graphics_keys;
    std::vector<CachedShaderCode> shaders =
        Vulkan::PipelineCache::ReadCachedShaderCode(copy, stop, &graphics_keys);
    std::filesystem::remove(copy, error);
    if (shaders.empty()) {
        LOG_INFO(Frontend,
                 "[ShaderScan] not searching on this start: the shader cache holds no shaders "
                 "yet, so the search runs on the next start");
        return;
    }

    LOG_INFO(Frontend, "[ShaderScan] searching game files for {} cached shaders", shaders.size());

    // Mark the scan as done up front, so a crash during it never blocks the game twice.
    SearchMarker marker{
        .files_id = files_id,
        .version = SEARCH_VERSION,
        .result = "did not finish, Eden was closed or crashed during the search",
    };
    WriteSearchMarker(shader_dir, marker);

    Scanner scanner{std::move(shaders), stop, start_time + TIME_LIMIT};
    scanner.SetProgress(&progress, std::max<size_t>(Scanner::CountFiles(root, "") + 1, 2));
    scanner.ReportProgress(true);
    scanner.FindZstdDictionaries(root);
    scanner.ScanDir(root, "", true);
    scanner.Report("shader-named files");
    scanner.StudyUnrealLinks(graphics_keys);
    scanner.StudyUnrealCaches();
    if (!scanner.Stopped()) {
        scanner.ScanDir(root, "", false);
        scanner.Report("all files");
    }
    const bool cut_short = scanner.Stopped();
    scanner.ReportUnmatched();
    scanner.ReportFormats();
    // When the code is in a package the first pass didn't read, the other packages were
    // searched before it was known.
    scanner.StudyUnrealLinks(graphics_keys);
    scanner.StudyUnrealCaches();
    scanner.ReportUnreal();
    scanner.ReportUnrealLinks();
    scanner.LearnUnrealLinksFromOtherPackages();
    scanner.ReportExtraction();
    scanner.ReportPairing(graphics_keys);
    scanner.ReportGuessAccuracy();
    {
        // The translator logs every instruction it cannot decode, which floods the log when it
        // reads data that is not a shader. Keep those classes quiet while testing.
        Common::Log::Filter quiet;
        quiet.ParseFilterString(Settings::values.log_filter.GetValue());
        quiet.SetClassLevel(Common::Log::Class::Debug, Common::Log::Level::Count);
        quiet.SetClassLevel(Common::Log::Class::Shader, Common::Log::Level::Count);
        Common::Log::SetGlobalFilter(quiet);
        scanner.CompareGuessesWithRuntime();
        scanner.TranslateExtracted();
        Common::Log::Filter normal;
        normal.ParseFilterString(Settings::values.log_filter.GetValue());
        Common::Log::SetGlobalFilter(normal);
    }
    scanner.SavePrograms(shader_dir / "preforged_programs.bin");
    scanner.ReportResult();
    marker.result = scanner.ResultSummary(cut_short);
    WriteSearchMarker(shader_dir, marker);
    if (progress) {
        progress(1, 1);
    }
    const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start_time);
    LOG_INFO(Frontend, "[ShaderScan] finished in {} s{}", seconds.count(),
             scanner.Stopped() ? " (stopped at the time limit)" : "");
}
