// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/environment.h"

namespace VideoCommon {

/// An environment for a shader taken from the game files, where nothing about the running game
/// is known: every texture is guessed to be 2D and every constant buffer read returns zero.
/// With constant buffer 1 folding off, this builds the same shaders the game builds at runtime.
class PreforgedEnvironment final : public Shader::Environment {
public:
    explicit PreforgedEnvironment(std::span<const u64> code_, Shader::Stage stage_,
                                  u32 start_address_, u32 texture_bound_);

    u64 ReadInstruction(u32 address) override;
    u32 ReadCbufValue(u32 index, u32 offset) override;
    Shader::TextureType ReadTextureType(u32 handle) override;
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override;
    bool IsTexturePixelFormatInteger(u32 handle) override;
    u32 ReadViewportTransformState() override;
    u32 TextureBoundBuffer() const override;
    u32 LocalMemorySize() const override;
    u32 SharedMemorySize() const override;
    std::array<u32, 3> WorkgroupSize() const override;
    bool HasHLEMacroState() const override;
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32 bank, u32 offset) override;
    void Dump(u64, u64) override {}

private:
    std::span<const u64> code;
    u32 texture_bound;
    u32 local_memory_size = 0;
};

struct PreforgedShader {
    Shader::Stage stage;
    u32 start_address;
    std::vector<u64> code; ///< Starts at the program header for graphics stages
};

/// Shaders found in the game files and the vertex and fragment pairs the game combines them in.
struct PreforgedPrograms {
    u32 texture_bound = 2;
    std::unordered_map<u64, PreforgedShader> shaders;
    std::vector<std::pair<u64, u64>> pairs;
};

[[nodiscard]] bool WritePreforgedPrograms(const std::filesystem::path& path,
                                          const PreforgedPrograms& programs);

[[nodiscard]] std::optional<PreforgedPrograms> ReadPreforgedPrograms(
    const std::filesystem::path& path);

} // namespace VideoCommon
