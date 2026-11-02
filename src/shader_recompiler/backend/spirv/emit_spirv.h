// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"

namespace Shader::Backend::SPIRV {

constexpr u32 NUM_TEXTURE_SCALING_WORDS = 4;
constexpr u32 NUM_IMAGE_SCALING_WORDS = 2;
constexpr u32 NUM_TEXTURE_AND_IMAGE_SCALING_WORDS =
    NUM_TEXTURE_SCALING_WORDS + NUM_IMAGE_SCALING_WORDS;

struct RescalingLayout {
    alignas(16) std::array<u32, NUM_TEXTURE_SCALING_WORDS> rescaling_textures;
    alignas(16) std::array<u32, NUM_IMAGE_SCALING_WORDS> rescaling_images;
    u32 down_factor;
};
struct RenderAreaLayout {
    std::array<f32, 4> render_area;
};
// The alpha test reference, pushed live when preforged_dynamic_alpha_ref is on so one fragment
// shader serves every reference value. It sits past the rescaling and render-area push constants
// so it never overlaps them (an alpha-tested surface also samples textures, which uses rescaling).
struct AlphaRefLayout {
    f32 alpha_ref;
};
constexpr u32 RESCALING_LAYOUT_WORDS_OFFSET = offsetof(RescalingLayout, rescaling_textures);
constexpr u32 RESCALING_LAYOUT_DOWN_FACTOR_OFFSET = offsetof(RescalingLayout, down_factor);
constexpr u32 RENDERAREA_LAYOUT_OFFSET = offsetof(RenderAreaLayout, render_area);
constexpr u32 ALPHA_REF_LAYOUT_OFFSET =
    static_cast<u32>(sizeof(RescalingLayout) + sizeof(RenderAreaLayout));

[[nodiscard]] std::vector<u32> EmitSPIRV(const Profile& profile, const RuntimeInfo& runtime_info, IR::Program& program, Bindings& bindings);
[[nodiscard]] inline std::vector<u32> EmitSPIRV(const Profile& profile, IR::Program& program) {
    Bindings binding;
    return EmitSPIRV(profile, {}, program, binding);
}

} // namespace Shader::Backend::SPIRV
