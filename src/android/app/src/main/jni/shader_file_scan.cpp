// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstring>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "common/cityhash.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/settings.h"
#include "common/thread.h"
#include "common/zstd_compression.h"
#include "core/core.h"
#include "core/file_sys/romfs.h"
#include "core/file_sys/romfs_factory.h"
#include "core/file_sys/vfs/vfs.h"
#include "core/hle/service/filesystem/filesystem.h"
#include "jni/shader_file_scan.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"

namespace {

using CachedShaderCode = Vulkan::PipelineCache::CachedShaderCode;

constexpr size_t SPH_SIZE = 0x50;
constexpr size_t NEEDLE_WORDS = 4;
constexpr size_t MAX_WHOLE_FILE = 64ULL << 20;
constexpr size_t CHUNK_SIZE = 64ULL << 20;
constexpr size_t CHUNK_OVERLAP = 1ULL << 20;
constexpr size_t MAX_DECOMPRESSED = 256ULL << 20;
constexpr u32 FILTER_BITS = 22;
constexpr auto TIME_LIMIT = std::chrono::minutes{4};
constexpr auto TRANSLATE_TIME_LIMIT = std::chrono::seconds{90};
constexpr u32 NVN_CODE_MAGIC = 0x12345678;
constexpr size_t NVN_CODE_HEADER_SIZE = 0x30;
constexpr size_t MAX_SHADER_SIZE = 0x100000;
constexpr size_t MAX_EXTRACTED_BYTES = 256ULL << 20;
constexpr u64 SELF_BRANCH_A = 0xE2400FFFFF87000FULL;
constexpr u64 SELF_BRANCH_B = 0xE2400FFFFF07000FULL;
constexpr u64 EXIT_VALUE = 0xE30000000007000FULL;

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

// Returns the decompressed contents when the data is a compressed container we understand.
std::vector<u8> Decompress(std::span<const u8> data) {
    if (data.size() >= 4 && std::memcmp(data.data(), "Yaz0", 4) == 0) {
        return DecompressYaz0(data);
    }
    if (data.size() >= 4 && data[0] == 0x28 && data[1] == 0xB5 && data[2] == 0x2F &&
        data[3] == 0xFD) {
        const u64 size = ZstdContentSize(data);
        if (size == 0 || size > MAX_DECOMPRESSED) {
            return {};
        }
        return Common::Compression::DecompressDataZSTD(data);
    }
    return {};
}

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
        }
        for (const auto& subdir : dir->GetSubdirectories()) {
            ScanDir(subdir, path + "/" + subdir->GetName(), shader_files);
        }
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
                GuessEnvironment guess{shader.code,   shader.stage,     shader.env->StartAddress(),
                                       texture_bound, driver_constants};
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
        Shader::Optimization::SetDriverConstBufferFolding(true);
        LOG_INFO(Frontend,
                 "[ShaderScan] without constant buffer 1 folding, all guessed vs runtime: {} "
                 "identical, {} different, {} failed; folding changes {} runtime shaders",
                 unfolded_same, unfolded_different, unfolded_failed, changed_by_folding);
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
    };

    struct FileStats {
        size_t hits = 0;
        size_t terminators = 0;
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
        if (size <= MAX_WHOLE_FILE) {
            const std::vector<u8> data = file->ReadAllBytes();
            bytes_scanned += data.size();
            ScanBuffer(data, name, 0);
        } else {
            std::vector<u8> chunk;
            for (size_t offset = 0; offset < size && !Stopped();
                 offset += CHUNK_SIZE) {
                const size_t length = std::min(size - offset, CHUNK_SIZE + CHUNK_OVERLAP);
                chunk = file->ReadBytes(length, offset);
                bytes_scanned += std::min(length, CHUNK_SIZE);
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
        if (data.size() >= 0x14 && std::memcmp(data.data(), "SARC", 4) == 0 &&
            ScanSarc(data, name, depth)) {
            return;
        }
        SearchShaders(data, name, 0);
    }

    // Walks the files inside a SARC archive so compressed entries are unpacked before searching.
    bool ScanSarc(std::span<const u8> data, const std::string& name, int depth) {
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
        const size_t nodes_offset = header_size + sfat_size;
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
                return false;
            }
            ScanBuffer(data.subspan(data_offset + start, end - start),
                       fmt::format("{}#{}", name, node), depth + 1);
        }
        return true;
    }

    void SearchShaders(std::span<const u8> data, const std::string& name, size_t base_offset) {
        if (data.size() < NEEDLE_WORDS * sizeof(u64)) {
            return;
        }
        size_t hits = 0;
        size_t terminators = 0;
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
                Extract(data, pos + NVN_CODE_HEADER_SIZE);
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
                if (kind > result.kind) {
                    if (result.kind < MatchKind::CodeOnly && kind >= MatchKind::CodeOnly) {
                        ++hits;
                        LogFirstMatches(index, kind, data, pos, name, base_offset);
                    }
                    result.kind = kind;
                }
            }
        }
        if (hits > 0) {
            auto& stats = hits_per_file[name];
            stats.hits += hits;
            stats.terminators += terminators;
        }
    }

    // Takes the shader that follows an NVN code header, using the same end marker and hash as the
    // emulator uses for shaders the game uploads.
    void Extract(std::span<const u8> data, size_t start) {
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
    size_t files_scanned = 0;
    size_t bytes_scanned = 0;
    std::map<u32, size_t> logged_per_stage;
    std::unordered_map<u64, ExtractedShader> extracted;
    std::unordered_set<u64> self_branch_hashes;
    u32 texture_bound = 2;
    DriverConstants driver_constants;
    size_t extracted_bytes = 0;
    size_t code_headers = 0;
    std::chrono::steady_clock::time_point last_progress = std::chrono::steady_clock::now();
};

} // namespace

void RunShaderFileScan(Core::System& system, u64 program_id, std::stop_token stop) {
    const auto start_time = std::chrono::steady_clock::now();
    const auto shader_dir = Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir) /
                            fmt::format("{:016x}", program_id);
    const auto done_marker = shader_dir / "shader_scan_done_v5";
    std::error_code error;
    if (std::filesystem::exists(done_marker, error)) {
        LOG_INFO(Frontend, "[ShaderScan] already ran for this game, delete {} to run it again",
                 Common::FS::PathToUTF8String(done_marker));
        return;
    }

    // Read from a copy, because a damaged cache file is deleted while it is read.
    const auto copy = shader_dir / "shader_scan_copy.bin";
    std::filesystem::copy_file(shader_dir / "vulkan.bin", copy,
                               std::filesystem::copy_options::overwrite_existing, error);
    if (error) {
        LOG_INFO(Frontend, "[ShaderScan] no pipeline cache to compare against: {}",
                 error.message());
        return;
    }
    std::vector<CachedShaderCode> shaders =
        Vulkan::PipelineCache::ReadCachedShaderCode(copy, stop);
    std::filesystem::remove(copy, error);
    if (shaders.empty()) {
        LOG_INFO(Frontend, "[ShaderScan] pipeline cache has no shaders");
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
    LOG_INFO(Frontend, "[ShaderScan] searching game files for {} cached shaders", shaders.size());

    // Mark the scan as done up front, so a crash during it never blocks the game twice.
    void(Common::FS::NewFile(done_marker));

    Scanner scanner{std::move(shaders), stop, start_time + TIME_LIMIT};
    scanner.ScanDir(root, "", true);
    scanner.Report("shader-named files");
    if (!scanner.Stopped()) {
        scanner.ScanDir(root, "", false);
        scanner.Report("all files");
    }
    scanner.ReportExtraction();
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
    const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start_time);
    LOG_INFO(Frontend, "[ShaderScan] finished in {} s{}", seconds.count(),
             scanner.Stopped() ? " (stopped at the time limit)" : "");
}
