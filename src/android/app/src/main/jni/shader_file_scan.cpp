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
#include "common/common_funcs.h"
#include "common/fs/file.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/preforged_debug.h"
#include "common/settings.h"
#include "common/lz4_compression.h"
#include "common/thread.h"
#include "common/zstd_compression.h"
#include "core/core.h"
#include "core/crypto/aes_util.h"
#include "core/crypto/key_manager.h"
#include "core/file_sys/content_archive.h"
#include "core/file_sys/nca_metadata.h"
#include "core/file_sys/patch_manager.h"
#include "core/file_sys/registered_cache.h"
#include "core/file_sys/romfs.h"
#include "core/file_sys/romfs_factory.h"
#include "core/file_sys/vfs/vfs.h"
#include "core/hle/service/filesystem/filesystem.h"
#include "core/loader/nso.h"
#include "jni/shader_file_scan.h"
#include "jni/unreal_links.h"
#include "jni/unreal_package.h"
#include "jni/unreal_pak.h"
#include "jni/embedded_streams.h"
#include "jni/gf_pak.h"
#include "jni/trpfs.h"
#include "jni/re_pak.h"
#include "jni/unity_assets.h"
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
// play log tells a missed pair from a shader the game files don't have. 16: the code of vertex
// and fragment shaders in no pair kept too, for pipeline libraries. 17: Unreal Engine 4 pak
// files unpacked (with the AES key from the game's executable when encrypted), and the shaders
// in their shader archives; the Switch shaders of Unity games' Shader objects, in their asset
// files and bundles; the read-only data and data of the game's program modules. 18: zstd frames,
// zlib streams, ZIP entries and CRILAYLA files found inside any file; RE Engine paks, Game
// Freak .gfpak archives and Pokémon Scarlet and Violet's data.trpfs (build 75 had 17 without
// them).
constexpr u32 SEARCH_VERSION = 18;
/// Packed streams found inside files that unpack to fewer bytes are left out: too small to hold
/// a shader worth the search.
constexpr size_t MIN_PACKED_STREAM = 256;
/// Files whose packed streams are logged one by one; the rest are only in the sums.
constexpr size_t PACKED_FILES_LOGGED = 16;
/// Bytes read from the start of a file to tell a Unity bundle or asset file.
constexpr size_t UNITY_START = 0x100;
/// Unity asset files and bundles logged one by one; the rest are only in the sums.
constexpr size_t UNITY_FILES_LOGGED = 24;
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
            if (IsMedia(name)) {
                continue;
            }
            // Unreal Engine 4 pak files: their shader archives in the first pass, their other
            // files in the second.
            if (Common::Preforged::Options().unpack_unreal_packages &&
                EndsWith(Lower(name), ".pak") && ScanPak(file, name, shader_files)) {
                if (!shader_files) {
                    ++files_visited;
                    ReportProgress(false);
                }
                continue;
            }
            if (LooksLikeShaderFile(name) != shader_files) {
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
        size_t unpaired_bytes = 0;
        for (const auto& [hash, shader] : extracted) {
            if (programs.shaders.contains(hash)) {
                continue;
            }
            programs.other_shaders.push_back(hash);
            // Vertex and fragment shaders in no pair are compiled on their own as pipeline
            // libraries, so a pair the game makes in play is joined instead of compiled.
            if (shader.stage == Shader::Stage::VertexB || shader.stage == Shader::Stage::Fragment) {
                programs.unpaired_shaders.try_emplace(
                    hash,
                    VideoCommon::PreforgedShader{shader.stage, shader.start_address, shader.code});
                unpaired_bytes += shader.code.size() * sizeof(u64);
            }
        }
        const bool written = VideoCommon::WritePreforgedPrograms(path, programs);
        LOG_INFO(Frontend,
                 "[ShaderScan] saved {} shader pairs with {} shaders, and the hashes of the {} "
                 "other shaders taken from the game files, with the code of the {} vertex and "
                 "fragment shaders among them ({} KiB): {}",
                 programs.pairs.size(), programs.shaders.size(), programs.other_shaders.size(),
                 programs.unpaired_shaders.size(), unpaired_bytes >> 10, written ? "ok" : "failed");
    }

    bool Stopped() const {
        return stop.stop_requested() || std::chrono::steady_clock::now() > deadline;
    }

    bool AllFound() const {
        return std::ranges::all_of(results, [](const Result& result) {
            return result.kind == MatchKind::Exact || result.kind == MatchKind::CodeOnly;
        });
    }

    /// Reads the game executable's read-only data and data, where Unreal Engine 4 games keep
    /// the AES key of their pak files. Called once, when the first encrypted pak needs a key.
    void SetExecutableReader(std::function<std::vector<u8>(std::string&)> reader) {
        executable_reader = std::move(reader);
    }

    using ModuleReader =
        std::function<std::vector<std::pair<std::string, std::vector<u8>>>(std::string&)>;

    void SetModuleReader(ModuleReader reader) {
        module_reader = std::move(reader);
    }

    /// Searches the read-only data and data of the game's program modules, where small engines
    /// and ports keep the shaders built into them.
    void ScanModules() {
        if (!module_reader || Stopped()) {
            return;
        }
        if (!Common::Preforged::Options().search_program) {
            LOG_INFO(Frontend, "[ShaderScan] 'Search the game's program' is off, so its modules "
                               "aren't searched");
            return;
        }
        std::string note;
        const auto modules = module_reader(note);
        std::vector<std::string> read;
        for (const auto& [module, data] : modules) {
            if (Stopped()) {
                break;
            }
            const size_t headers_before = code_headers;
            const size_t taken_before = extracted.size();
            bytes_scanned += data.size();
            SearchShaders(data, "exefs/" + module, 0);
            read.push_back(fmt::format("{} {} KiB ({} NVN code headers, {} new shaders)", module,
                                       data.size() >> 10, code_headers - headers_before,
                                       extracted.size() - taken_before));
        }
        LOG_INFO(Frontend,
                 "[ShaderScan] game program modules searched (read-only data and data): {}{}",
                 read.empty() ? std::string{"none"} : fmt::format("{}", fmt::join(read, ", ")),
                 note.empty() ? "" : fmt::format("; not read: {}", note));
    }

    /// The engines the game's file names point to, from LogGameEngines.
    void SetEngines(std::vector<std::string_view> engines_) {
        engines = std::move(engines_);
    }

    /// Sums up what the Unity bundles and asset files held, and says so when a Unity game's
    /// files held no Shader objects.
    void ReportUnity() const {
        const bool unity_game = std::ranges::find(engines, "Unity") != engines.end();
        if (!Common::Preforged::Options().read_unity_files) {
            if (unity_game) {
                LOG_INFO(Frontend, "[ShaderScan] Unity: 'Read Unity shader files' is off, so "
                                   "only shaders stored unpacked are found");
            }
            return;
        }
        const auto& totals = unity_totals;
        if (!unity_game && totals.bundles == 0 && totals.bundles_failed == 0 &&
            totals.serialized == 0) {
            return;
        }
        const auto& stats = unity_stats;
        LOG_INFO(Frontend,
                 "[ShaderScan] Unity: {} bundles read, {} not read{}; {} files in them read ({} "
                 "MiB unpacked; blocks stored {}, LZMA {} ({} MiB), LZ4 {}), {} could not be "
                 "read{}, {} texture and sound streams skipped ({} MiB); {} asset files searched "
                 "for Shader objects: {} places looked like one, {} Shader objects with Switch "
                 "shaders, {} Switch parts unpacked ({} MiB), {} did not unpack{}; {} NVN code "
                 "headers in them, {} new shaders taken; the first part starts {}",
                 totals.bundles, totals.bundles_failed,
                 totals.first_bundle_error.empty()
                     ? ""
                     : fmt::format(" (first: {})", totals.first_bundle_error),
                 totals.bundle_files, totals.bundle_bytes >> 20, totals.blocks[0],
                 totals.blocks[1], totals.lzma_bytes >> 20, totals.blocks[2], totals.read_failed,
                 totals.first_read_error.empty()
                     ? ""
                     : fmt::format(" (first: {})", totals.first_read_error),
                 totals.streams_skipped, totals.streams_skipped_bytes >> 20, totals.serialized,
                 stats.candidates, stats.blobs, stats.parts, stats.unpacked >> 20, stats.failed,
                 stats.first_failure.empty() ? "" : fmt::format(" (first: {})", stats.first_failure),
                 totals.code_headers, totals.taken,
                 stats.first_part.empty() ? "(none)" : stats.first_part);
        if (unity_game && stats.parts == 0) {
            LOG_INFO(Frontend,
                     "[ShaderScan] Unity: no Switch shaders found in the game files, though its "
                     "file names point to Unity. The lines above say whether its bundles and "
                     "asset files were read; shaders in bundles packed another way, or in a "
                     "layout of Shader objects that isn't known, aren't found");
        }
    }

    /// Sums up what the Unreal Engine 4 pak files held.
    void ReportPaks() const {
        if (pak_totals.paks == 0) {
            return;
        }
        std::string methods;
        for (size_t method = 0; method < pak_totals.methods.size(); ++method) {
            if (pak_totals.methods[method] != 0) {
                methods += fmt::format(
                    "{}{} {}", methods.empty() ? "" : ", ",
                    Preforged::Unreal::MethodName(static_cast<Preforged::Unreal::Method>(method)),
                    pak_totals.methods[method]);
            }
        }
        std::vector<std::pair<size_t, std::string>> by_type;
        for (const auto& [type, count] : pak_totals.headers_by_type) {
            by_type.emplace_back(count, type);
        }
        std::ranges::sort(by_type, std::greater{});
        std::string types;
        for (const auto& [count, type] : by_type) {
            types += fmt::format("{}{} {}", types.empty() ? "" : ", ", type, count);
        }
        LOG_INFO(Frontend,
                 "[ShaderScan] Unreal Engine 4 pak files: {} ({} readable, {} with an encrypted "
                 "index, {} keys found), {} files listed, {} read ({} MiB unpacked), {} did not "
                 "unpack, {} skipped (media or too large); {} shader archives with {} shaders, {} "
                 "NVN code headers in them, {} of them new; blocks by method: {}; NVN code "
                 "headers by file type: {}",
                 pak_totals.paks, pak_totals.readable, pak_totals.encrypted_index,
                 known_keys.size(), pak_totals.listed, pak_totals.read,
                 pak_totals.unpacked >> 20, pak_totals.failed, pak_totals.skipped,
                 pak_totals.archives, pak_totals.archive_shaders, pak_totals.archive_headers,
                 pak_totals.archive_taken, methods.empty() ? "none" : methods,
                 types.empty() ? "none" : types);
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
        // Archives read file by file: RE Engine paks, Unity bundles. Large Unity asset files are
        // searched for Shader objects piece by piece.
        bool unity_assets = false;
        const std::vector<u8> start = file->ReadBytes(UNITY_START, 0);
        if (Common::Preforged::Options().find_packed_data &&
            ((Preforged::RePak::IsPak(start) && ScanRePak(file, name)) ||
             (Preforged::GfPak::IsPak(start) && ScanGfPak(file, name)) ||
             (Preforged::Trpfs::IsTrpfs(start) && ScanTrpfs(file, name)))) {
            LogProgress();
            return;
        }
        if (Common::Preforged::Options().read_unity_files) {
            if (Preforged::Unity::IsBundle(start) && ScanUnityBundle(file, name)) {
                LogProgress();
                return;
            }
            unity_assets = size > MAX_WHOLE_FILE && Preforged::Unity::IsSerializedFile(start, size);
        }
        bool whole = size <= MAX_WHOLE_FILE;
        if (!whole && size <= MAX_WHOLE_PACKAGE &&
            Common::Preforged::Options().unpack_unreal_packages) {
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
            const Preforged::Unity::ReadFn fetch = [&file](u64 offset, u64 length) {
                return file->ReadBytes(length, offset);
            };
            UnitySearch unity{*this, name};
            u64 streams_end = 0;
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
                if (unity_assets) {
                    unity.Search(chunk, offset, &fetch);
                }
                // Streams that start in this piece; the overlap belongs to the next one.
                const u64 to = offset + length == size ? size : offset + CHUNK_SIZE;
                streams_end = SearchPackedStreams(chunk, name, offset,
                                                  std::max<u64>(offset, streams_end), to, size,
                                                  &fetch, 0);
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
        if (Common::Preforged::Options().read_unity_files &&
            Preforged::Unity::IsSerializedFile(data, data.size())) {
            UnitySearch{*this, name}.Search(data, 0, nullptr);
        }
        SearchPackedStreams(data, name, 0, 0, data.size(), data.size(), nullptr, depth);
        auto& stored = formats.stored_starts[Start32(data)];
        ++stored.first;
        stored.second += data.size();
        SearchShaders(data, name, 0);
    }

    /// Finds zstd frames, zlib streams, ZIP entries and CRILAYLA files that start in [from, to)
    /// of `data` (the bytes at `base` of a file of `file_size` bytes), unpacks them and searches
    /// what they hold like a file of their own. Returns where the last stream found ends.
    u64 SearchPackedStreams(std::span<const u8> data, const std::string& name, u64 base, u64 from,
                            u64 to, u64 file_size, const Preforged::Embedded::ReadFn* fetch,
                            int depth) {
        using namespace Preforged::Embedded;
        if (!Common::Preforged::Options().find_packed_data || depth >= 4 || Stopped()) {
            return from;
        }
        // Only the outermost search counts what is found, as streams can hold streams.
        const bool outermost = packed_nesting == 0;
        ++packed_nesting;
        const size_t headers_before = code_headers;
        const size_t taken_before = extracted.size();
        size_t found = 0;
        Source source{data, base, file_size, fetch};
        Handlers handlers;
        handlers.zstd = [&](u64 offset, std::span<const u8> frame) -> u64 {
            const std::vector<u8> out = Decompress(frame);
            if (out.empty()) {
                ++packed_totals.zstd_failed;
                return 0;
            }
            ++found;
            ScanBuffer(out, fmt::format("{}@{:x}", name, offset), depth + 1);
            return out.size();
        };
        handlers.unpacked = [&](Kind, u64 offset, std::span<const u8> out) {
            ++found;
            ScanBuffer(out, fmt::format("{}@{:x}", name, offset), depth + 1);
        };
        handlers.stop = [this] { return Stopped(); };
        const u64 end = FindStreams(source, from, to, MIN_PACKED_STREAM, MAX_DECOMPRESSED,
                                    handlers, packed_stats);
        --packed_nesting;
        if (outermost && found != 0) {
            const size_t new_headers = code_headers - headers_before;
            const size_t taken = extracted.size() - taken_before;
            packed_totals.code_headers += new_headers;
            packed_totals.taken += taken;
            auto& file = packed_totals.files[name];
            file.streams += found;
            file.code_headers += new_headers;
            file.taken += taken;
        }
        return end;
    }

public:
    /// Sums up the packed streams found inside files.
    void ReportPackedStreams() const {
        if (!Common::Preforged::Options().find_packed_data) {
            LOG_INFO(Frontend, "[ShaderScan] 'Find packed data inside files' is off");
            return;
        }
        using namespace Preforged::Embedded;
        const auto& stats = packed_stats;
        std::vector<std::string> kinds;
        for (size_t kind = 0; kind < KIND_COUNT; ++kind) {
            kinds.push_back(fmt::format(
                "{} {} of {} that looked like one ({} MiB packed, {} MiB unpacked, {} too large)",
                KindName(static_cast<Kind>(kind)), stats.streams[kind], stats.candidates[kind],
                stats.packed[kind] >> 20, stats.unpacked[kind] >> 20, stats.too_large[kind]));
        }
        std::vector<std::pair<size_t, std::string>> by_headers;
        for (const auto& [file, counts] : packed_totals.files) {
            by_headers.emplace_back(counts.code_headers * 1000000 + counts.streams, file);
        }
        std::ranges::sort(by_headers, std::greater{});
        std::vector<std::string> files;
        for (size_t i = 0; i < std::min(by_headers.size(), PACKED_FILES_LOGGED); ++i) {
            const auto& counts = packed_totals.files.at(by_headers[i].second);
            files.push_back(fmt::format("{} ({} streams, {} NVN code headers, {} new shaders)",
                                        by_headers[i].second, counts.streams,
                                        counts.code_headers, counts.taken));
        }
        LOG_INFO(Frontend,
                 "[ShaderScan] packed data inside files: {}; {} zstd frames did not unpack; {} "
                 "files held some; {} NVN code headers in them, {} new shaders taken; most: {}",
                 fmt::join(kinds, "; "), packed_totals.zstd_failed, packed_totals.files.size(),
                 packed_totals.code_headers, packed_totals.taken,
                 files.empty() ? std::string{"none"} : fmt::format("{}", fmt::join(files, ", ")));
    }

private:

    /// One file of an archive read by its table.
    struct TableEntry {
        u64 id = 0; ///< Hash of its path, for its name in the log
        u64 offset = 0;
        u64 packed_size = 0;
        u64 size = 0;
        const char* method = "stored"; ///< As Preforged::Unreal::UnpackBlock names it
        bool encrypted = false;
    };

    /// Reads the files of an archive by its table, in the order they are stored, unpacks and
    /// searches each. `what` describes the archive for the log line.
    void ScanTableEntries(const FileSys::VirtualFile& file, const std::string& name,
                          std::string_view what, std::vector<TableEntry> entries) {
        const size_t headers_before = code_headers;
        const size_t taken_before = extracted.size();
        std::ranges::sort(entries, {}, &TableEntry::offset);
        std::map<std::string_view, size_t> by_method;
        size_t read = 0;
        size_t encrypted = 0;
        size_t too_large = 0;
        size_t failed = 0;
        std::string first_failure;
        u64 unpacked = 0;
        Preforged::Unreal::Method last = Preforged::Unreal::Method::Unknown;
        std::vector<u8> out;
        for (const TableEntry& entry : entries) {
            if (Stopped()) {
                break;
            }
            if (entry.encrypted) {
                ++encrypted;
                continue;
            }
            if (entry.size > MAX_DECOMPRESSED || entry.packed_size > MAX_DECOMPRESSED) {
                ++too_large;
                continue;
            }
            const std::vector<u8> packed = file->ReadBytes(entry.packed_size, entry.offset);
            std::span<const u8> data = packed;
            const std::string_view method = entry.method;
            if (method != "stored") {
                out.resize(entry.size + Preforged::Unreal::UNPACK_SLACK);
                const auto used = Preforged::Unreal::UnpackBlock(
                    packed, std::span<u8>{out.data(), entry.size}, method, last,
                    Common::Preforged::Options().unpack_oodle);
                if (used == Preforged::Unreal::Method::Unknown ||
                    used == Preforged::Unreal::Method::OodleFailed) {
                    ++failed;
                    if (first_failure.empty()) {
                        first_failure = fmt::format("{} {} -> {} bytes, starts {}", method,
                                                    entry.packed_size, entry.size,
                                                    Hex(packed, 0, 8));
                    }
                    continue;
                }
                data = std::span<const u8>{out.data(), entry.size};
            }
            ++by_method[method];
            ++read;
            unpacked += data.size();
            bytes_scanned += data.size();
            ScanBuffer(data, fmt::format("{}#{:016x}", name, entry.id), 1);
        }
        std::string methods;
        for (const auto& [method, count] : by_method) {
            methods += fmt::format("{}{} {}", methods.empty() ? "" : ", ", method, count);
        }
        LOG_INFO(Frontend,
                 "[ShaderScan] {} {} ({} MiB): {} files read ({}; {} MiB unpacked), {} encrypted "
                 "and {} too large skipped, {} did not unpack{}; {} NVN code headers, {} new "
                 "shaders taken",
                 what, name, file->GetSize() >> 20, read, methods.empty() ? "none" : methods,
                 unpacked >> 20, encrypted, too_large, failed,
                 first_failure.empty() ? "" : fmt::format(" (first: {})", first_failure),
                 code_headers - headers_before, extracted.size() - taken_before);
    }

    /// Reads the files of an RE Engine pak one by one and searches each. Returns false when its
    /// table can't be read, so it is searched as it is (packed data inside it is still found).
    bool ScanRePak(const FileSys::VirtualFile& file, const std::string& name) {
        using namespace Preforged::RePak;
        const ReadFn read = [&file](u64 offset, u64 size) { return file->ReadBytes(size, offset); };
        std::string error;
        const auto pak = ReadPak(read, file->GetSize(), error);
        if (!pak || pak->table_encrypted) {
            LOG_INFO(Frontend,
                     "[ShaderScan] RE Engine pak {} ({} MiB) not read ({}), so it is searched as "
                     "it is",
                     name, file->GetSize() >> 20,
                     pak ? fmt::format("version {}.{}, features {:#x}: its file table is "
                                       "encrypted",
                                       pak->major, pak->minor, pak->features)
                         : error);
            return false;
        }
        std::vector<TableEntry> entries;
        for (const Entry& entry : pak->entries) {
            entries.push_back({
                .id = entry.hash,
                .offset = entry.offset,
                .packed_size = entry.packed_size,
                .size = entry.size,
                .method = entry.compression == 1 ? "zlib"
                          : entry.compression == 2 ? "zstd"
                                                   : "stored",
                .encrypted = entry.encryption != 0,
            });
        }
        ScanTableEntries(file, name,
                         fmt::format("RE Engine pak, version {}.{}, features {:#x}, {} files "
                                     "listed ({} point past the file):",
                                     pak->major, pak->minor, pak->features, pak->declared,
                                     pak->invalid),
                         std::move(entries));
        return true;
    }

    /// Reads the files of a Game Freak archive one by one and searches each. Returns false when
    /// its table can't be read, so it is searched as it is.
    bool ScanGfPak(const FileSys::VirtualFile& file, const std::string& name) {
        using namespace Preforged::GfPak;
        const ReadFn read = [&file](u64 offset, u64 size) { return file->ReadBytes(size, offset); };
        std::string error;
        const auto pak = ReadPak(read, file->GetSize(), error);
        if (!pak) {
            LOG_INFO(Frontend, "[ShaderScan] Game Freak archive {} not read ({}), so it is "
                               "searched as it is",
                     name, error);
            return false;
        }
        std::vector<TableEntry> entries;
        for (const Entry& entry : pak->entries) {
            entries.push_back({
                .id = entry.hash,
                .offset = entry.offset,
                .packed_size = entry.packed_size,
                .size = entry.size,
                .method = MethodName(entry.compression),
            });
        }
        ScanTableEntries(file, name,
                         fmt::format("Game Freak archive, version {:#x}, {} files listed ({} "
                                     "point past the file):",
                                     pak->version, pak->declared, pak->invalid),
                         std::move(entries));
        return true;
    }

    /// Reads the packs of a Pokémon Scarlet and Violet data.trpfs one by one, unpacks the files
    /// in each and searches them. Returns false when its table can't be read.
    bool ScanTrpfs(const FileSys::VirtualFile& file, const std::string& name) {
        using namespace Preforged::Trpfs;
        const ReadFn read = [&file](u64 offset, u64 size) { return file->ReadBytes(size, offset); };
        std::string error;
        const auto fs = ReadFileSystem(read, file->GetSize(), error);
        if (!fs) {
            LOG_INFO(Frontend, "[ShaderScan] Game Freak trpfs {} not read ({}), so it is searched "
                               "as it is",
                     name, error);
            return false;
        }
        const size_t headers_before = code_headers;
        const size_t taken_before = extracted.size();
        size_t packs_read = 0;
        size_t packs_failed = 0;
        size_t packs_too_large = 0;
        std::string first_pack_error;
        size_t stored = 0;
        size_t oodle = 0;
        size_t failed = 0;
        size_t too_large = 0;
        std::string first_failure;
        u64 unpacked = 0;
        std::vector<PackedFile> files;
        std::vector<u8> out;
        Preforged::Unreal::Method last = Preforged::Unreal::Method::Unknown;
        for (const Pack& pack : fs->packs) {
            if (Stopped()) {
                break;
            }
            if (pack.size > MAX_DECOMPRESSED * 2) {
                ++packs_too_large;
                continue;
            }
            const std::vector<u8> bytes = file->ReadBytes(pack.size, pack.offset);
            if (!ReadPack(bytes, files, error)) {
                ++packs_failed;
                if (first_pack_error.empty()) {
                    first_pack_error = fmt::format("pack at {:#x}: {}", pack.offset, error);
                }
                continue;
            }
            ++packs_read;
            for (const PackedFile& packed : files) {
                if (Stopped()) {
                    break;
                }
                std::span<const u8> data = packed.data;
                if (packed.size > MAX_DECOMPRESSED) {
                    ++too_large;
                    continue;
                }
                if (packed.compression != -1 || packed.data.size() != packed.size) {
                    out.resize(packed.size + Preforged::Unreal::UNPACK_SLACK);
                    const auto used = Preforged::Unreal::UnpackBlock(
                        packed.data, std::span<u8>{out.data(), packed.size}, "oodle", last,
                        Common::Preforged::Options().unpack_oodle);
                    if (used == Preforged::Unreal::Method::Unknown ||
                        used == Preforged::Unreal::Method::OodleFailed) {
                        ++failed;
                        if (first_failure.empty()) {
                            first_failure = fmt::format(
                                "type {}, {} -> {} bytes, starts {}", packed.compression,
                                packed.data.size(), packed.size, Hex(packed.data, 0, 8));
                        }
                        continue;
                    }
                    data = std::span<const u8>{out.data(), packed.size};
                    ++oodle;
                } else {
                    ++stored;
                }
                unpacked += data.size();
                bytes_scanned += data.size();
                ScanBuffer(data, fmt::format("{}#{:016x}/{:016x}", name, pack.hash, packed.hash),
                           1);
            }
        }
        LOG_INFO(Frontend,
                 "[ShaderScan] Game Freak trpfs {} ({} MiB): {} packs listed, {} read, {} not "
                 "read{}, {} too large; files: {} stored, {} unpacked with Oodle ({} MiB in "
                 "all), {} did not unpack{}, {} too large; {} NVN code headers, {} new shaders "
                 "taken",
                 name, file->GetSize() >> 20, fs->packs.size(), packs_read, packs_failed,
                 first_pack_error.empty() ? "" : fmt::format(" (first: {})", first_pack_error),
                 packs_too_large, stored, oodle, unpacked >> 20, failed,
                 first_failure.empty() ? "" : fmt::format(" (first: {})", first_failure),
                 too_large, code_headers - headers_before, extracted.size() - taken_before);
        return true;
    }

    /// Searches one Unity asset file, whole or piece by piece, for Shader objects, unpacks
    /// their Switch shaders and searches those; logs what the file held when done.
    class UnitySearch {
    public:
        UnitySearch(Scanner& scanner_, const std::string& name_)
            : scanner{scanner_}, name{name_}, shader_name{name_ + " (Unity shaders)"},
              stats_before{scanner_.unity_stats}, headers_before{scanner_.code_headers},
              taken_before{scanner_.extracted.size()} {}

        UnitySearch(const UnitySearch&) = delete;
        UnitySearch& operator=(const UnitySearch&) = delete;

        ~UnitySearch() {
            if (!searched) {
                return;
            }
            auto& totals = scanner.unity_totals;
            const auto& stats = scanner.unity_stats;
            ++totals.serialized;
            const size_t new_headers = scanner.code_headers - headers_before;
            const size_t taken = scanner.extracted.size() - taken_before;
            totals.code_headers += new_headers;
            totals.taken += taken;
            const size_t blobs = stats.blobs - stats_before.blobs;
            if (blobs == 0 && stats.failed == stats_before.failed) {
                return;
            }
            if (++totals.files_logged > UNITY_FILES_LOGGED) {
                return;
            }
            LOG_INFO(Frontend,
                     "[ShaderScan] Unity asset file {}: {} Shader objects with Switch shaders, "
                     "{} parts unpacked ({} KiB), {} did not unpack, {} NVN code headers in "
                     "them, {} new shaders taken",
                     name, blobs, stats.parts - stats_before.parts,
                     (stats.unpacked - stats_before.unpacked) >> 10,
                     stats.failed - stats_before.failed, new_headers, taken);
        }

        void Search(std::span<const u8> data, u64 base_offset,
                    const Preforged::Unity::ReadFn* fetch) {
            searched = true;
            Preforged::Unity::FindShaderBlobs(
                data, base_offset,
                [this](std::span<const u8> part, u64) {
                    // Each part is searched at its own offset, so the code headers in it are
                    // told apart from those of other parts.
                    scanner.SearchShaders(part, shader_name, part_offset);
                    part_offset += (part.size() + 15) / 16 * 16;
                },
                scanner.unity_stats, fetch, seen);
        }

    private:
        Scanner& scanner;
        const std::string& name;
        std::string shader_name;
        Preforged::Unity::BlobStats stats_before;
        size_t headers_before;
        size_t taken_before;
        std::unordered_set<u64> seen;
        u64 part_offset = 0;
        bool searched = false;
    };

    /// Reads the files in a UnityFS bundle (data.unity3d, asset bundles) and searches each,
    /// the asset files among them for Shader objects too. Returns false when the bundle can't
    /// be read, so it is searched as stored.
    bool ScanUnityBundle(const FileSys::VirtualFile& file, const std::string& name) {
        using namespace Preforged::Unity;
        const ReadFn read = [&file](u64 offset, u64 size) { return file->ReadBytes(size, offset); };
        auto& totals = unity_totals;
        std::string error;
        const auto bundle = ReadBundle(read, file->GetSize(), error);
        if (!bundle) {
            ++totals.bundles_failed;
            if (totals.first_bundle_error.empty()) {
                totals.first_bundle_error = fmt::format("{}: {}", name, error);
            }
            LOG_INFO(Frontend, "[ShaderScan] Unity bundle {} not read ({}), so it is searched "
                               "as stored",
                     name, error);
            return false;
        }
        ++totals.bundles;
        const size_t headers_before = code_headers;
        const size_t taken_before = extracted.size();
        const size_t serialized_before = totals.serialized;
        const size_t parts_before = unity_stats.parts;
        BlockCache cache;
        std::vector<u8> data;
        size_t files = 0;
        u64 unpacked = 0;
        std::array<size_t, 3> methods{};
        for (const auto& block : bundle->blocks) {
            ++methods[std::min<u32>(block.flags & 0x3F, 2)];
        }
        for (const auto& node : bundle->nodes) {
            if (Stopped()) {
                break;
            }
            const std::string lower = Lower(node.path);
            // Texture, mesh and sound data that asset files point into.
            if (EndsWith(lower, ".ress") || EndsWith(lower, ".resource")) {
                ++totals.streams_skipped;
                totals.streams_skipped_bytes += node.size;
                continue;
            }
            const std::string node_name = fmt::format("{}/{}", name, node.path);
            const auto failed = [&] {
                ++totals.read_failed;
                if (totals.first_read_error.empty()) {
                    totals.first_read_error = fmt::format("{}: {}", node_name, error);
                }
            };
            ++files;
            if (node.size <= MAX_WHOLE_FILE) {
                if (!ReadBundleData(read, *bundle, node.offset, node.size, data, cache, error)) {
                    failed();
                    continue;
                }
                unpacked += data.size();
                bytes_scanned += data.size();
                ScanBuffer(data, node_name, 1);
                continue;
            }
            // A large file, in pieces.
            const ReadFn fetch = [&](u64 offset, u64 size) {
                std::vector<u8> out;
                std::string fetch_error;
                if (offset >= node.size ||
                    !ReadBundleData(read, *bundle, node.offset + offset,
                                    std::min(size, node.size - offset), out, cache, fetch_error)) {
                    out.clear();
                }
                return out;
            };
            const std::vector<u8> start = fetch(0, UNITY_START);
            const bool assets = IsSerializedFile(start, node.size);
            UnitySearch unity{*this, node_name};
            for (u64 offset = 0; offset < node.size && !Stopped(); offset += CHUNK_SIZE) {
                const u64 length = std::min<u64>(node.size - offset, CHUNK_SIZE + CHUNK_OVERLAP);
                if (!ReadBundleData(read, *bundle, node.offset + offset, length, data, cache,
                                    error)) {
                    failed();
                    break;
                }
                unpacked += std::min<u64>(length, CHUNK_SIZE);
                bytes_scanned += std::min<u64>(length, CHUNK_SIZE);
                if (assets) {
                    unity.Search(data, offset, &fetch);
                }
                SearchShaders(data, node_name, offset);
            }
        }
        totals.bundle_files += files;
        totals.bundle_bytes += unpacked;
        for (size_t method = 0; method < cache.unpacked.size(); ++method) {
            totals.blocks[method] += cache.unpacked[method];
        }
        totals.lzma_bytes += cache.lzma_bytes;
        if (++totals.bundles_logged <= UNITY_FILES_LOGGED) {
            LOG_INFO(Frontend,
                     "[ShaderScan] Unity bundle {} ({} MiB): format {}, Unity {}, {} blocks "
                     "(stored {}, LZMA {}, LZ4 {}), {} files of {} read ({} MiB unpacked), {} of "
                     "them asset files; {} Switch shader parts, {} NVN code headers, {} new "
                     "shaders taken",
                     name, file->GetSize() >> 20, bundle->version, bundle->revision,
                     bundle->blocks.size(), methods[0], methods[1], methods[2], files,
                     bundle->nodes.size(), unpacked >> 20, totals.serialized - serialized_before,
                     unity_stats.parts - parts_before, code_headers - headers_before,
                     extracted.size() - taken_before);
        }
        return true;
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
        // A file gets an id when it is first found to hold shader code, so the many pieces of
        // data searched that hold none (each packed stream in an archive) take no memory.
        std::optional<u32> known_id;
        const auto file_id = [&] {
            if (!known_id) {
                const auto [file_it, inserted] = file_ids.try_emplace(name, file_names.size());
                if (inserted) {
                    file_names.push_back(name);
                    headers.emplace_back();
                }
                known_id = file_it->second;
            }
            return *known_id;
        };
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
                Extract(data, pos + NVN_CODE_HEADER_SIZE, file_id(), base_offset + pos);
                const auto& file_headers = headers[file_id()];
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
                NotePartialMatch(index, data, pos, file_id(), base_offset);
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

    /// An Unreal Engine 4 pak file, with its index once read.
    struct PakState {
        FileSys::VirtualFile file;
        std::string name;
        Preforged::UnrealPak::Footer footer;
        Preforged::UnrealPak::Index index;
        std::unique_ptr<Core::Crypto::AESCipher<Core::Crypto::Key256>> cipher;
        /// The index was read, so its files can be.
        bool readable = false;
        /// An unreadable pak searched as stored, as other large files are.
        bool stored_searched = false;
    };

    void DecryptPak(const PakState& pak, std::span<u8> data) const {
        const size_t whole = data.size() / 16 * 16;
        if (pak.cipher && whole != 0) {
            pak.cipher->Transcode(data.data(), whole, data.data(), Core::Crypto::Op::Decrypt);
        }
    }

    /// Finds the key of an encrypted pak index: a key found for another pak, or one in the
    /// game's executable.
    std::optional<Preforged::UnrealPak::Key> PakKey(const PakState& pak,
                                                    std::span<const u8> primary,
                                                    std::string& note) {
        using namespace Preforged::UnrealPak;
        if (primary.size() < 16) {
            note = "encrypted, and too short to test keys with";
            return std::nullopt;
        }
        // A wrong key gives an index start with a fitting string length and characters about
        // once in 10^11 tries, so the entry count is enough of a check after it.
        const auto verify{[&](const Key& key) {
            std::vector<u8> copy(primary.begin(), primary.end());
            Aes256{key}.Decrypt(copy);
            return ReadIndex(pak.footer, copy, [](u64, u64) { return std::vector<u8>{}; })
                       .declared > 0;
        }};
        for (const Key& key : known_keys) {
            std::array<u8, 16> plain;
            Aes256{key}.DecryptBlock(primary.data(), plain.data());
            if (LooksLikeIndexStart(plain) && verify(key)) {
                note = "encrypted, with the key found for another pak file";
                return key;
            }
        }
        if (!executable_read) {
            executable_read = true;
            const auto start = std::chrono::steady_clock::now();
            if (executable_reader) {
                executable = executable_reader(executable_note);
            } else {
                executable_note = "no reader";
            }
            LOG_INFO(Frontend, "[ShaderScan] game executable read for pak keys: {} ({} ms)",
                     executable_note,
                     std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - start)
                         .count());
        }
        if (executable.empty()) {
            note = fmt::format("encrypted, and the game's executable could not be read ({})",
                               executable_note);
            return std::nullopt;
        }
        const auto start = std::chrono::steady_clock::now();
        size_t tried = 0;
        size_t where = 0;
        const auto key = FindKey(executable, primary.first(16), verify, tried, where);
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count();
        if (!key) {
            note = fmt::format("encrypted, and none of the {} keys tried from the game's "
                               "executable ({} MiB) decrypts it ({} ms)",
                               tried, executable.size() >> 20, ms);
            return std::nullopt;
        }
        known_keys.push_back(*key);
        note = fmt::format("encrypted, key found in the game's executable at {:#x} of its data "
                           "after {} tries ({} ms)",
                           where, tried, ms);
        return key;
    }

    /// Reads the footer and index of a pak file once. Returns nothing when the file is not an
    /// Unreal Engine 4 pak.
    PakState* OpenPak(const FileSys::VirtualFile& file, const std::string& name) {
        using namespace Preforged::UnrealPak;
        if (const auto it = paks.find(name); it != paks.end()) {
            return it->second.get();
        }
        std::unique_ptr<PakState>& slot = paks[name];
        const u64 size = file->GetSize();
        if (size < 64) {
            return nullptr;
        }
        const size_t tail_size = static_cast<size_t>(std::min<u64>(size, FOOTER_SEARCH));
        const std::vector<u8> tail = file->ReadBytes(tail_size, size - tail_size);
        const auto footer = ReadFooter(tail, size);
        if (!footer) {
            return nullptr;
        }
        slot = std::make_unique<PakState>();
        PakState& pak = *slot;
        pak.file = file;
        pak.name = name;
        pak.footer = *footer;
        ++pak_totals.paks;
        std::string index_note = "plain";
        const auto log_pak{[&](std::string_view result) {
            LOG_INFO(Frontend,
                     "[ShaderScan] Unreal pak {} ({} MiB): version {}, index {} ({} KiB), "
                     "compression methods {}: {}",
                     name, size >> 20, footer->version, index_note, footer->index_size >> 10,
                     fmt::join(footer->methods, "/"), result);
        }};
        if (footer->frozen_index) {
            log_pak("its index is a frozen memory image, which isn't read");
            return &pak;
        }
        std::vector<u8> primary = file->ReadBytes(footer->index_size, footer->index_offset);
        if (primary.size() != footer->index_size) {
            log_pak("its index could not be read");
            return &pak;
        }
        if (footer->encrypted_index) {
            ++pak_totals.encrypted_index;
            const auto key = PakKey(pak, primary, index_note);
            if (!key) {
                log_pak("not read, so the file is searched as stored");
                return &pak;
            }
            pak.cipher =
                std::make_unique<Core::Crypto::AESCipher<Core::Crypto::Key256>>(
                    *key, Core::Crypto::Mode::ECB);
            DecryptPak(pak, primary);
        }
        pak.index = ReadIndex(*footer, primary, [&](u64 offset, u64 part_size) {
            std::vector<u8> part = file->ReadBytes(part_size, offset);
            if (footer->encrypted_index) {
                DecryptPak(pak, part);
            }
            return part;
        });
        pak.readable = !pak.index.entries.empty();
        size_t encrypted = 0;
        std::map<std::string, size_t> by_method;
        u64 unpacked = 0;
        for (const auto& entry : pak.index.entries) {
            encrypted += entry.encrypted ? 1 : 0;
            unpacked += entry.uncompressed_size;
            ++by_method[entry.method == 0                        ? std::string{"stored"}
                        : entry.method <= footer->methods.size() ? footer->methods[entry.method - 1]
                                                                 : fmt::format("#{}", entry.method)];
        }
        pak_totals.listed += pak.index.entries.size();
        if (pak.readable) {
            ++pak_totals.readable;
        }
        std::string methods;
        for (const auto& [method, count] : by_method) {
            methods += fmt::format("{}{} {}", methods.empty() ? "" : ", ", method, count);
        }
        log_pak(fmt::format(
            "mount point {}, {} of {} files listed{} ({} MiB unpacked; {}), {} encrypted{}{}",
            pak.index.mount_point, pak.index.entries.size(), pak.index.declared,
            pak.index.named ? "" : " without names", unpacked >> 20, methods, encrypted,
            encrypted != 0 && !pak.cipher ? ", and no key for them" : "",
            pak.index.error.empty() ? "" : fmt::format("; index read stopped: {}", pak.index.error)));
        return &pak;
    }

    /// Searches a pak file: its shader archives in the first pass, its other files in the
    /// second. Returns false when the file is not an Unreal Engine 4 pak.
    bool ScanPak(const FileSys::VirtualFile& file, const std::string& name, bool archives_pass) {
        PakState* const pak = OpenPak(file, name);
        if (!pak) {
            return false;
        }
        if (!pak->readable) {
            if (!archives_pass && !pak->stored_searched) {
                pak->stored_searched = true;
                ScanFile(file, name);
            }
            return true;
        }
        ScanPakEntries(*pak, archives_pass);
        return true;
    }

    static bool IsPakMedia(std::string_view lower) {
        for (const std::string_view ext : {".bik", ".bk2", ".wem", ".bnk", ".mp4", ".webm", ".ogg",
                                           ".wav", ".opus", ".ubulk", ".uptnl", ".ufont", ".ttf",
                                           ".otf", ".png", ".jpg", ".usm", ".awb", ".acb"}) {
            if (EndsWith(lower, ext)) {
                return true;
            }
        }
        return false;
    }

    static std::string FileType(std::string_view lower) {
        const size_t slash = lower.rfind('/');
        const size_t dot = lower.rfind('.');
        if (dot == std::string_view::npos || (slash != std::string_view::npos && dot < slash)) {
            return "no extension";
        }
        return std::string(lower.substr(dot));
    }

    void ScanPakEntries(PakState& pak, bool archives_pass) {
        using namespace Preforged::UnrealPak;
        const auto start = std::chrono::steady_clock::now();
        const ReadFn read = [&pak](u64 offset, u64 size) {
            return pak.file->ReadBytes(size, offset);
        };
        DecryptFn decrypt;
        if (pak.cipher) {
            decrypt = [this, &pak](std::span<u8> data) { DecryptPak(pak, data); };
        }
        // Without names, archives are told by their contents, so every file is read in the
        // first pass.
        if (!pak.index.named && !archives_pass) {
            return;
        }
        std::vector<size_t> order;
        size_t skipped = 0;
        for (size_t i = 0; i < pak.index.entries.size(); ++i) {
            const Entry& entry = pak.index.entries[i];
            const std::string lower = Lower(entry.path);
            if (pak.index.named && IsShaderArchiveName(lower) != archives_pass) {
                continue;
            }
            if (IsPakMedia(lower) || entry.uncompressed_size > MAX_DECOMPRESSED) {
                ++skipped;
                continue;
            }
            order.push_back(i);
        }
        // Material assets may hold their shaders inline, so they come first.
        const auto rank{[&pak](size_t i) {
            const std::string type = FileType(Lower(pak.index.entries[i].path));
            return type == ".uexp" ? 0 : type == ".uasset" ? 1 : type == ".umap" ? 2 : 3;
        }};
        std::ranges::stable_sort(order, {}, rank);
        size_t read_count = 0;
        size_t failed = 0;
        u64 packed = 0;
        u64 unpacked = 0;
        std::string first_failure;
        bool stopped = false;
        std::vector<u8> data;
        for (const size_t i : order) {
            if (Stopped()) {
                stopped = true;
                break;
            }
            const Entry& entry = pak.index.entries[i];
            std::string error;
            if (!ReadEntry(read, pak.footer, entry, decrypt,
                           Common::Preforged::Options().unpack_oodle, data, pak_totals.methods,
                           error)) {
                ++failed;
                if (first_failure.empty()) {
                    first_failure = fmt::format("{}: {}", entry.path, error);
                }
                continue;
            }
            ++read_count;
            packed += entry.size;
            unpacked += data.size();
            bytes_scanned += entry.size;
            const std::string entry_name =
                fmt::format("{}:{}", pak.name, entry.path.empty() ? fmt::format("#{}", i) : entry.path);
            const std::string lower = Lower(entry.path);
            const size_t headers_before = code_headers;
            const bool archive = (IsShaderArchiveName(lower) || !pak.index.named) &&
                                 ScanShaderArchive(data, entry_name, !pak.index.named);
            if (!archive) {
                if (Preforged::Unreal::IsPackage(data)) {
                    SearchShaders(data, entry_name, 0);
                } else {
                    ScanBuffer(data, entry_name, 1);
                }
            }
            if (code_headers != headers_before) {
                pak_totals.headers_by_type[FileType(lower)] += code_headers - headers_before;
            }
            LogProgress();
        }
        pak_totals.read += read_count;
        pak_totals.failed += failed;
        pak_totals.skipped += skipped;
        pak_totals.unpacked += unpacked;
        LOG_INFO(Frontend,
                 "[ShaderScan] Unreal pak {}: {} {} files read ({} MiB packed, {} MiB unpacked) "
                 "in {} ms, {} did not unpack{}, {} skipped (media or over {} MiB){}",
                 pak.name, read_count, archives_pass ? "shader archive" : "other",
                 packed >> 20, unpacked >> 20,
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - start)
                     .count(),
                 failed, first_failure.empty() ? "" : fmt::format(" (first: {})", first_failure),
                 skipped, MAX_DECOMPRESSED >> 20, stopped ? ", stopped at the time limit" : "");
    }

    /// Reads the shaders of an Unreal Engine 4 shader archive, each unpacked on its own, and
    /// searches them. Returns false when the data is not a shader archive.
    bool ScanShaderArchive(std::span<const u8> data, const std::string& name, bool quiet) {
        using Preforged::Unreal::Method;
        std::string error;
        const auto archive = Preforged::UnrealPak::ReadShaderArchive(data, error);
        if (!archive) {
            if (!quiet) {
                LOG_INFO(Frontend,
                         "[ShaderScan] Unreal shader archive {} not read ({}), so it is searched "
                         "as stored",
                         name, error);
            }
            return false;
        }
        const size_t headers_before = code_headers;
        const size_t taken_before = extracted.size();
        // Vertex, hull, domain, pixel, geometry, compute, other.
        std::array<size_t, 7> by_stage{};
        std::array<size_t, Preforged::Unreal::METHOD_COUNT> methods{};
        size_t failed = 0;
        std::string first_failure;
        std::string first_bytes;
        Method last = Method::Unknown;
        std::vector<u8> out;
        for (const auto& shader : archive->shaders) {
            if (Stopped()) {
                break;
            }
            ++by_stage[std::min<size_t>(shader.frequency, 6)];
            const std::span<const u8> in =
                data.subspan(archive->code_start + shader.offset, shader.size);
            std::span<const u8> code = in;
            if (shader.size != shader.uncompressed_size) {
                out.resize(shader.uncompressed_size + Preforged::Unreal::UNPACK_SLACK);
                const Method method = Preforged::Unreal::UnpackBlock(
                    in, std::span<u8>{out.data(), shader.uncompressed_size}, "LZ4", last,
                    Common::Preforged::Options().unpack_oodle);
                ++methods[static_cast<size_t>(method)];
                if (method == Method::Unknown || method == Method::OodleFailed) {
                    ++failed;
                    if (first_failure.empty()) {
                        first_failure = fmt::format("{} -> {} bytes, starts {}", shader.size,
                                                    shader.uncompressed_size, Hex(in, 0, 8));
                    }
                    continue;
                }
                code = std::span<const u8>{out.data(), shader.uncompressed_size};
            } else {
                ++methods[static_cast<size_t>(Method::Stored)];
            }
            if (first_bytes.empty()) {
                first_bytes = Hex(code, 0, 64);
            }
            SearchShaders(code, name, shader.offset);
        }
        std::string unpacked;
        for (size_t method = 0; method < methods.size(); ++method) {
            if (methods[method] != 0) {
                unpacked += fmt::format(
                    "{}{} {}", unpacked.empty() ? "" : ", ",
                    Preforged::Unreal::MethodName(static_cast<Method>(method)), methods[method]);
            }
        }
        const size_t new_headers = code_headers - headers_before;
        const size_t taken = extracted.size() - taken_before;
        ++pak_totals.archives;
        pak_totals.archive_shaders += archive->shaders.size();
        pak_totals.archive_headers += new_headers;
        pak_totals.archive_taken += taken;
        size_t largest_map = 0;
        for (const auto& [first, count] : archive->maps) {
            largest_map = std::max<size_t>(largest_map, count);
        }
        LOG_INFO(Frontend,
                 "[ShaderScan] Unreal shader archive {}: version {}, {} shaders (vertex {}, "
                 "pixel {}, compute {}, hull, domain and geometry {}, other {}) in {} shader "
                 "maps (largest {} shaders); unpacked: {}; {} did not unpack{}; {} NVN code "
                 "headers in them, {} new shaders taken; the first shader starts {}",
                 name, archive->version, archive->shaders.size(), by_stage[0], by_stage[3],
                 by_stage[5], by_stage[1] + by_stage[2] + by_stage[4], by_stage[6],
                 archive->maps.size(), largest_map, unpacked.empty() ? "none" : unpacked, failed,
                 first_failure.empty() ? "" : fmt::format(" (first: {})", first_failure),
                 new_headers, taken, first_bytes.empty() ? "(none)" : first_bytes);
        return true;
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
    std::map<std::string, std::unique_ptr<PakState>> paks;
    struct PakTotals {
        size_t paks = 0;
        size_t readable = 0;
        size_t encrypted_index = 0;
        size_t listed = 0;
        size_t read = 0;
        size_t failed = 0;
        size_t skipped = 0;
        u64 unpacked = 0;
        size_t archives = 0;
        size_t archive_shaders = 0;
        size_t archive_headers = 0;
        size_t archive_taken = 0;
        std::array<size_t, Preforged::Unreal::METHOD_COUNT> methods{};
        std::map<std::string, size_t> headers_by_type;
    } pak_totals;
    std::function<std::vector<u8>(std::string&)> executable_reader;
    bool executable_read = false;
    std::vector<u8> executable;
    std::string executable_note;
    std::vector<Preforged::UnrealPak::Key> known_keys;
    std::vector<std::string_view> engines;
    ModuleReader module_reader;
    struct UnityTotals {
        size_t bundles = 0;
        size_t bundles_failed = 0;
        std::string first_bundle_error;
        size_t bundle_files = 0;
        u64 bundle_bytes = 0;
        size_t read_failed = 0;
        std::string first_read_error;
        size_t streams_skipped = 0;
        u64 streams_skipped_bytes = 0;
        /// Blocks unpacked: stored, LZMA, LZ4.
        std::array<size_t, 3> blocks{};
        u64 lzma_bytes = 0;
        /// Asset files searched for Shader objects.
        size_t serialized = 0;
        size_t code_headers = 0;
        size_t taken = 0;
        size_t files_logged = 0;
        size_t bundles_logged = 0;
    } unity_totals;
    Preforged::Unity::BlobStats unity_stats;
    Preforged::Embedded::Stats packed_stats;
    struct PackedStreamFile {
        size_t streams = 0;
        size_t code_headers = 0;
        size_t taken = 0;
    };
    struct PackedTotals {
        size_t zstd_failed = 0;
        size_t code_headers = 0;
        size_t taken = 0;
        std::map<std::string, PackedStreamFile> files;
    } packed_totals;
    int packed_nesting = 0;
    std::chrono::steady_clock::time_point last_progress = std::chrono::steady_clock::now();
};

/// Opens the game's ExeFS, with an update's in place of the base game's. `note` says why
/// nothing was opened.
FileSys::VirtualDir OpenExeFS(Core::System& system, u64 program_id, std::string& note) {
    const auto nca =
        system.GetContentProvider().GetEntry(program_id, FileSys::ContentRecordType::Program);
    if (!nca) {
        note = "the game's program content was not found";
        return nullptr;
    }
    FileSys::VirtualDir exefs = nca->GetExeFS();
    if (!exefs) {
        note = "the game's program has no ExeFS";
        return nullptr;
    }
    const FileSys::PatchManager patch{program_id, system.GetFileSystemController(),
                                      system.GetContentProvider()};
    exefs = patch.PatchExeFS(exefs);
    if (!exefs) {
        note = "the patched ExeFS could not be opened";
    }
    return exefs;
}

/// Reads the read-only data and data of one NSO module (main, subsdk0, sdk, ...), unpacked.
/// The code segment is left out: it holds no shaders or keys in one piece. `note` says what
/// was read, or why nothing was.
std::vector<u8> ReadModuleData(const FileSys::VirtualFile& file, std::string& note) {
    const std::string name = file->GetName();
    const std::vector<u8> nso = file->ReadAllBytes();
    Loader::NSOHeader header;
    if (nso.size() < sizeof(header)) {
        note = fmt::format("{} is too short", name);
        return {};
    }
    std::memcpy(&header, nso.data(), sizeof(header));
    if (header.magic != Common::MakeMagic('N', 'S', 'O', '0')) {
        note = fmt::format("{} is not an NSO", name);
        return {};
    }
    if (header.IsZBICCompressed()) {
        note = fmt::format("{} is packed with ZBIC, which isn't read here", name);
        return {};
    }
    std::vector<u8> out;
    std::array<size_t, 3> sizes{};
    // Segments 1 and 2: read-only data and data.
    for (size_t segment = 1; segment < 3; ++segment) {
        const auto& info = header.segments[segment];
        const bool packed = header.IsSegmentCompressed(segment);
        const u64 stored = packed ? header.segments_compressed_size[segment] : info.size;
        if (info.offset > nso.size() || stored > nso.size() - info.offset) {
            note = fmt::format("segment {} of {} lies past its end", segment, name);
            return {};
        }
        const std::span<const u8> bytes{nso.data() + info.offset, static_cast<size_t>(stored)};
        if (packed) {
            const std::vector<u8> unpacked =
                Common::Compression::DecompressDataLZ4(bytes, info.size);
            if (unpacked.size() != info.size) {
                note = fmt::format("segment {} of {} did not unpack", segment, name);
                return {};
            }
            out.insert(out.end(), unpacked.begin(), unpacked.end());
        } else {
            out.insert(out.end(), bytes.begin(), bytes.end());
        }
        sizes[segment] = info.size;
    }
    note = fmt::format("{}: {} KiB of read-only data, {} KiB of data", name, sizes[1] >> 10,
                       sizes[2] >> 10);
    return out;
}

/// Reads the read-only data and data of the game's main executable, with an update's in place
/// of the base game's, for the AES key of Unreal Engine 4 pak files. `note` says what was read,
/// or why nothing was.
std::vector<u8> ReadExecutableData(Core::System& system, u64 program_id, std::string& note) {
    const FileSys::VirtualDir exefs = OpenExeFS(system, program_id, note);
    const FileSys::VirtualFile main = exefs ? exefs->GetFile("main") : nullptr;
    if (!main) {
        if (exefs) {
            note = "the ExeFS has no main";
        }
        return {};
    }
    return ReadModuleData(main, note);
}

/// Reads the read-only data and data of every module of the game's program: main, the
/// subsdk modules and sdk. Small engines and ports keep their few shaders built into them.
std::vector<std::pair<std::string, std::vector<u8>>> ReadProgramModules(Core::System& system,
                                                                        u64 program_id,
                                                                        std::string& note) {
    std::vector<std::pair<std::string, std::vector<u8>>> modules;
    const FileSys::VirtualDir exefs = OpenExeFS(system, program_id, note);
    if (!exefs) {
        return modules;
    }
    std::vector<std::string> notes;
    for (const auto& file : exefs->GetFiles()) {
        const std::string name = file->GetName();
        // rtld is the system's loader; main.npdm is the program's description, not code.
        if (name == "rtld" || name.find('.') != std::string::npos) {
            continue;
        }
        std::string module_note;
        std::vector<u8> data = ReadModuleData(file, module_note);
        if (data.empty()) {
            notes.push_back(module_note);
            continue;
        }
        modules.emplace_back(name, std::move(data));
    }
    note = fmt::format("{}", fmt::join(notes, ", "));
    return modules;
}

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

/// How far the search gets with an engine's shaders.
enum class EngineSupport {
    /// The search unpacks the engine's shader files and pairs their shaders.
    Read,
    /// The engine's archives aren't read by their tables, but their files are packed in ways
    /// the packed data finder unpacks wherever they are (zstd, zlib, ZIP, CRILAYLA).
    Streams,
    /// The shaders are in archives the search doesn't unpack yet, so it only finds the ones
    /// stored unpacked.
    Packed,
    /// The engine ships shader source and compiles it while the game runs, so the game files
    /// hold no compiled shaders to find.
    Source,
};

/// A game engine and the file names that give it away, from the engine survey in
/// notes/engine-survey.md. Matched against paths in lower case.
struct EngineRule {
    std::string_view name;
    EngineSupport support;
    /// Each matches a path that contains it.
    std::array<std::string_view, 4> markers;
};

constexpr std::array ENGINE_RULES{
    EngineRule{"Unity",
               EngineSupport::Read,
               {"/data/managed/metadata/global-metadata.dat", "/data/globalgamemanagers",
                "/data/sharedassets0.assets", "/data/data.unity3d"}},
    EngineRule{"Unreal Engine 4 (pak files)",
               EngineSupport::Read,
               {"/ue4commandline.txt", "/content/paks/", "", ""}},
    EngineRule{"Unreal Engine 4 or 5 IoStore containers",
               EngineSupport::Packed,
               {".utoc ", ".ucas ", "", ""}},
    EngineRule{"Unreal Engine 3", EngineSupport::Read, {".upk ", ".xxx ", ".umap ", "/cookednx"}},
    EngineRule{"Nintendo formats (BFRES, BFSHA, BNSH)",
               EngineSupport::Read,
               {".bfsha", ".bnsh", ".bfres", "/shaderdata/"}},
    EngineRule{"RE Engine", EngineSupport::Read, {"re_chunk_000.pak", "/natives/nsw/", "", ""}},
    EngineRule{"MT Framework", EngineSupport::Streams, {"/nativenx/", "/nativeswitch/", "", ""}},
    EngineRule{"Monolith Soft engine (Xenoblade)",
               EngineSupport::Streams,
               {"/monolib/shader/", ".wishp ", ".wismt ", ".ard "}},
    EngineRule{"Game Freak engine (Pokemon, .gfpak archives)",
               EngineSupport::Read,
               {".gfpak ", "", "", ""}},
    EngineRule{"Game Freak engine (Pokemon Scarlet and Violet, .trpfs)",
               EngineSupport::Read,
               {"/data.trpfs ", ".trpak ", "", ""}},
    EngineRule{"Smash Ultimate archive", EngineSupport::Streams, {"/data.arc ", "", "", ""}},
    EngineRule{"CRI CPK archives", EngineSupport::Streams, {".cpk ", "", "", ""}},
    EngineRule{"CryEngine",
               EngineSupport::Streams,
               {"/shadercache.pak ", "/shadercachestartup.pak ", "/shaders/shaderlist.txt", ""}},
    EngineRule{"UbiArt Framework", EngineSupport::Packed, {".ipk ", "", "", ""}},
    EngineRule{"Creation Engine", EngineSupport::Packed, {".bsa ", "", "", ""}},
    EngineRule{"Source engine", EngineSupport::Packed, {".vpk ", "", "", ""}},
    EngineRule{"PhyreEngine", EngineSupport::Packed, {".phyre ", "", "", ""}},
    EngineRule{"GameMaker", EngineSupport::Source, {"/game.win ", "/data.win ", "", ""}},
    EngineRule{"MonoGame or FNA", EngineSupport::Source, {".xnb ", "", "", ""}},
    EngineRule{"Heaps (HashLink)", EngineSupport::Source, {"/res/shaders.cache", "", "", ""}},
    EngineRule{"RPG Maker MV or MZ",
               EngineSupport::Source,
               {"/js/rpg_core.js ", "/js/rmmz_core.js ", "/js/rpg_managers.js ", ""}},
};

/// Logs which engines the game's file names point to and how far the search gets with each,
/// so a log tells why a game's shaders were or weren't found.
std::vector<std::string_view> LogGameEngines(std::string_view file_list) {
    struct Match {
        size_t files = 0;
        std::vector<std::string> examples;
    };
    std::array<Match, ENGINE_RULES.size()> matches{};
    std::string line;
    size_t begin = 0;
    while (begin < file_list.size()) {
        size_t end = file_list.find('\n', begin);
        if (end == std::string_view::npos) {
            end = file_list.size();
        }
        // Each line is the path, a space and the size.
        line.assign(file_list.substr(begin, end - begin));
        begin = end + 1;
        const size_t name_end = line.rfind(' ');
        if (name_end == std::string::npos) {
            continue;
        }
        line.resize(name_end);
        std::transform(line.begin(), line.end(), line.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        const std::string_view path = line;
        for (size_t rule = 0; rule < ENGINE_RULES.size(); ++rule) {
            for (const std::string_view marker : ENGINE_RULES[rule].markers) {
                if (marker.empty()) {
                    continue;
                }
                // A marker ending in a space matches the end of the path, so ".pak " only
                // matches a name ending in .pak.
                const bool hit = marker.back() == ' '
                                     ? path.ends_with(marker.substr(0, marker.size() - 1))
                                     : path.find(marker) != std::string_view::npos;
                if (!hit) {
                    continue;
                }
                Match& match = matches[rule];
                ++match.files;
                if (match.examples.size() < 2) {
                    match.examples.emplace_back(path);
                }
                break;
            }
        }
    }
    std::vector<std::string> found;
    std::vector<std::string_view> engines;
    for (size_t rule = 0; rule < ENGINE_RULES.size(); ++rule) {
        const Match& match = matches[rule];
        if (match.files == 0) {
            continue;
        }
        engines.push_back(ENGINE_RULES[rule].name);
        const char* const support = [&] {
            switch (ENGINE_RULES[rule].support) {
            case EngineSupport::Read:
                return "the search unpacks its shader files";
            case EngineSupport::Streams:
                return "its archives aren't read by their tables, but the packed data finder "
                       "unpacks the files in them";
            case EngineSupport::Packed:
                return "its archives aren't unpacked yet, so only shaders stored unpacked are "
                       "found";
            case EngineSupport::Source:
                return "it compiles shader source while the game runs, so the files hold no "
                       "compiled shaders";
            }
            return "";
        }();
        found.push_back(fmt::format("{} ({} files, such as {}; {})", ENGINE_RULES[rule].name,
                                    match.files, fmt::join(match.examples, ", "), support));
    }
    if (found.empty()) {
        LOG_INFO(Frontend,
                 "[ShaderScan] game engine: not recognized from the file names; the search looks "
                 "for shaders stored unpacked in every file");
        return engines;
    }
    LOG_INFO(Frontend, "[ShaderScan] game engine from the file names: {}", fmt::join(found, "; "));
    return engines;
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
    std::vector<std::string_view> engines = LogGameEngines(file_list);
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
    scanner.SetExecutableReader([&system, program_id](std::string& note) {
        return ReadExecutableData(system, program_id, note);
    });
    scanner.SetEngines(std::move(engines));
    scanner.SetModuleReader([&system, program_id](std::string& note) {
        return ReadProgramModules(system, program_id, note);
    });
    scanner.SetProgress(&progress, std::max<size_t>(Scanner::CountFiles(root, "") + 1, 2));
    scanner.ReportProgress(true);
    scanner.FindZstdDictionaries(root);
    scanner.ScanModules();
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
    scanner.ReportPaks();
    scanner.ReportUnity();
    scanner.ReportPackedStreams();
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
