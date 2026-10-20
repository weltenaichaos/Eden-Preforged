// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>
#include <fstream>

#include "shader_recompiler/exception.h"
#include "video_core/preforged_shaders.h"

namespace VideoCommon {

namespace {
constexpr u32 FILE_MAGIC = 0x52504650; // "PFPR"
constexpr u32 FILE_VERSION = 1;
constexpr u32 MAX_SHADER_WORDS = 0x100000 / sizeof(u64);
constexpr u32 MAX_OTHER_SHADERS = 1u << 22;

template <typename T>
void Write(std::ofstream& file, const T& value) {
    file.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template <typename T>
bool Read(std::ifstream& file, T& value) {
    return static_cast<bool>(file.read(reinterpret_cast<char*>(&value), sizeof(value)));
}
} // Anonymous namespace

PreforgedEnvironment::PreforgedEnvironment(std::span<const u64> code_, Shader::Stage stage_,
                                           u32 start_address_, u32 texture_bound_)
    : code{code_}, texture_bound{texture_bound_} {
    stage = stage_;
    start_address = start_address_;
    is_proprietary_driver = texture_bound == 2;
    if (stage != Shader::Stage::Compute && code.size() * sizeof(u64) >= sizeof(sph)) {
        std::memcpy(&sph, code.data(), sizeof(sph));
        local_memory_size =
            static_cast<u32>(sph.LocalMemorySize()) + sph.common3.shader_local_memory_crs_size;
    }
}

u64 PreforgedEnvironment::ReadInstruction(u32 address) {
    const size_t index = (address - start_address) / sizeof(u64);
    if (address < start_address || index >= code.size()) {
        throw Shader::LogicError("Read past the end of the shader");
    }
    return code[index];
}

u32 PreforgedEnvironment::ReadCbufValue(u32, u32) {
    return 0;
}

Shader::TextureType PreforgedEnvironment::ReadTextureType(u32) {
    return Shader::TextureType::Color2D;
}

Shader::TexturePixelFormat PreforgedEnvironment::ReadTexturePixelFormat(u32) {
    return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
}

bool PreforgedEnvironment::IsTexturePixelFormatInteger(u32) {
    return false;
}

u32 PreforgedEnvironment::ReadViewportTransformState() {
    return 1;
}

u32 PreforgedEnvironment::TextureBoundBuffer() const {
    return texture_bound;
}

u32 PreforgedEnvironment::LocalMemorySize() const {
    return local_memory_size;
}

u32 PreforgedEnvironment::SharedMemorySize() const {
    return 0x8000;
}

std::array<u32, 3> PreforgedEnvironment::WorkgroupSize() const {
    return {64, 1, 1};
}

bool PreforgedEnvironment::HasHLEMacroState() const {
    return false;
}

std::optional<Shader::ReplaceConstant> PreforgedEnvironment::GetReplaceConstBuffer(u32, u32) {
    return std::nullopt;
}

bool WritePreforgedPrograms(const std::filesystem::path& path,
                            const PreforgedPrograms& programs) {
    std::ofstream file{path, std::ios::binary | std::ios::trunc};
    if (!file) {
        return false;
    }
    Write(file, FILE_MAGIC);
    Write(file, FILE_VERSION);
    Write(file, programs.texture_bound);
    Write(file, static_cast<u32>(programs.shaders.size()));
    Write(file, static_cast<u32>(programs.pairs.size()));
    for (const auto& [hash, shader] : programs.shaders) {
        Write(file, hash);
        Write(file, static_cast<u32>(shader.stage));
        Write(file, shader.start_address);
        Write(file, static_cast<u32>(shader.code.size()));
        file.write(reinterpret_cast<const char*>(shader.code.data()),
                   shader.code.size() * sizeof(u64));
    }
    for (const auto& [vertex, fragment] : programs.pairs) {
        Write(file, vertex);
        Write(file, fragment);
    }
    // After the pairs, so builds that don't read them still read the rest.
    Write(file, static_cast<u32>(programs.other_shaders.size()));
    file.write(reinterpret_cast<const char*>(programs.other_shaders.data()),
               programs.other_shaders.size() * sizeof(u64));
    return static_cast<bool>(file);
}

std::optional<PreforgedPrograms> ReadPreforgedPrograms(const std::filesystem::path& path) {
    std::ifstream file{path, std::ios::binary};
    if (!file) {
        return std::nullopt;
    }
    u32 magic{};
    u32 version{};
    u32 shader_count{};
    u32 pair_count{};
    PreforgedPrograms programs;
    if (!Read(file, magic) || !Read(file, version) || magic != FILE_MAGIC ||
        version != FILE_VERSION || !Read(file, programs.texture_bound) ||
        !Read(file, shader_count) || !Read(file, pair_count)) {
        return std::nullopt;
    }
    for (u32 index = 0; index < shader_count; ++index) {
        u64 hash{};
        u32 stage{};
        u32 word_count{};
        PreforgedShader shader{};
        if (!Read(file, hash) || !Read(file, stage) || !Read(file, shader.start_address) ||
            !Read(file, word_count) || word_count > MAX_SHADER_WORDS ||
            stage > static_cast<u32>(Shader::Stage::VertexA)) {
            return std::nullopt;
        }
        shader.stage = static_cast<Shader::Stage>(stage);
        shader.code.resize(word_count);
        if (!file.read(reinterpret_cast<char*>(shader.code.data()), word_count * sizeof(u64))) {
            return std::nullopt;
        }
        programs.shaders.emplace(hash, std::move(shader));
    }
    programs.pairs.resize(pair_count);
    for (auto& [vertex, fragment] : programs.pairs) {
        if (!Read(file, vertex) || !Read(file, fragment)) {
            return std::nullopt;
        }
    }
    // Files from builds before the list end here.
    u32 other_count{};
    if (Read(file, other_count) && other_count <= MAX_OTHER_SHADERS) {
        programs.other_shaders.resize(other_count);
        if (!file.read(reinterpret_cast<char*>(programs.other_shaders.data()),
                       other_count * sizeof(u64))) {
            programs.other_shaders.clear();
        }
    }
    return programs;
}

} // namespace VideoCommon
