// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <mutex>
#include <iostream>
#include <memory>
#include <span>
#include <thread>
#include <unordered_set>
#include <vector>
#include <bit>
#include <numeric>
#include "common/cityhash.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/thread_worker.h"
#include "core/core.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/program_header.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/preforged_shaders.h"
#include "common/preforged_debug.h"
#include "common/preforged_frame_stats.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/pipeline_statistics.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/shader_cache.h"
#include "video_core/shader_environment.h"
#include "video_core/shader_notify.h"
#include "video_core/surface.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/gpu_logging/gpu_logging.h"

#ifdef __ANDROID__
#include "../../android/app/src/main/jni/android_settings.h"
#endif

namespace Vulkan {

namespace {
using Shader::Backend::SPIRV::EmitSPIRV;
using Shader::Maxwell::ConvertLegacyToGeneric;
using Shader::Maxwell::GenerateGeometryPassthrough;
using Shader::Maxwell::MergeDualVertexPrograms;
using Shader::Maxwell::TranslateProgram;
using VideoCommon::ComputeEnvironment;
using VideoCommon::FileEnvironment;
using VideoCommon::GenericEnvironment;
using VideoCommon::GraphicsEnvironment;

constexpr u32 CACHE_VERSION = 18;
constexpr size_t VULKAN_CACHE_FLUSH_PIPELINES = 128;
constexpr size_t VULKAN_CACHE_FLUSH_MIN_SECONDS = 30;
constexpr std::array<char, 8> VULKAN_CACHE_MAGIC_NUMBER{'y', 'u', 'z', 'u', 'v', 'k', 'c', 'h'};

template <typename Container>
auto MakeSpan(Container& container) {
    return std::span(container.data(), container.size());
}

Shader::OutputTopology MaxwellToOutputTopology(Maxwell::PrimitiveTopology topology) {
    switch (topology) {
    case Maxwell::PrimitiveTopology::Points:
        return Shader::OutputTopology::PointList;
    case Maxwell::PrimitiveTopology::LineStrip:
        return Shader::OutputTopology::LineStrip;
    default:
        return Shader::OutputTopology::TriangleStrip;
    }
}

Shader::CompareFunction MaxwellToCompareFunction(Maxwell::ComparisonOp comparison) {
    switch (comparison) {
    case Maxwell::ComparisonOp::Never_D3D:
    case Maxwell::ComparisonOp::Never_GL:
        return Shader::CompareFunction::Never;
    case Maxwell::ComparisonOp::Less_D3D:
    case Maxwell::ComparisonOp::Less_GL:
        return Shader::CompareFunction::Less;
    case Maxwell::ComparisonOp::Equal_D3D:
    case Maxwell::ComparisonOp::Equal_GL:
        return Shader::CompareFunction::Equal;
    case Maxwell::ComparisonOp::LessEqual_D3D:
    case Maxwell::ComparisonOp::LessEqual_GL:
        return Shader::CompareFunction::LessThanEqual;
    case Maxwell::ComparisonOp::Greater_D3D:
    case Maxwell::ComparisonOp::Greater_GL:
        return Shader::CompareFunction::Greater;
    case Maxwell::ComparisonOp::NotEqual_D3D:
    case Maxwell::ComparisonOp::NotEqual_GL:
        return Shader::CompareFunction::NotEqual;
    case Maxwell::ComparisonOp::GreaterEqual_D3D:
    case Maxwell::ComparisonOp::GreaterEqual_GL:
        return Shader::CompareFunction::GreaterThanEqual;
    case Maxwell::ComparisonOp::Always_D3D:
    case Maxwell::ComparisonOp::Always_GL:
        return Shader::CompareFunction::Always;
    }
    UNIMPLEMENTED_MSG("Unimplemented comparison op={}", comparison);
    return {};
}

Shader::AttributeType CastAttributeType(const FixedPipelineState::VertexAttribute& attr) {
    if (attr.enabled == 0) {
        return Shader::AttributeType::Disabled;
    }
    switch (attr.Type()) {
    case Maxwell::VertexAttribute::Type::UnusedEnumDoNotUseBecauseItWillGoAway:
        ASSERT_MSG(false, "Invalid vertex attribute type!");
        return Shader::AttributeType::Disabled;
    case Maxwell::VertexAttribute::Type::SNorm:
    case Maxwell::VertexAttribute::Type::UNorm:
    case Maxwell::VertexAttribute::Type::Float:
        return Shader::AttributeType::Float;
    case Maxwell::VertexAttribute::Type::SInt:
        return Shader::AttributeType::SignedInt;
    case Maxwell::VertexAttribute::Type::UInt:
        return Shader::AttributeType::UnsignedInt;
    case Maxwell::VertexAttribute::Type::UScaled:
        return Shader::AttributeType::UnsignedScaled;
    case Maxwell::VertexAttribute::Type::SScaled:
        return Shader::AttributeType::SignedScaled;
    }
    return Shader::AttributeType::Float;
}

Shader::AttributeType AttributeType(const FixedPipelineState& state, size_t index) {
    switch (state.DynamicAttributeType(index)) {
    case 0:
        return Shader::AttributeType::Disabled;
    case 1:
        return Shader::AttributeType::Float;
    case 2:
        return Shader::AttributeType::SignedInt;
    case 3:
        return Shader::AttributeType::UnsignedInt;
    }
    return Shader::AttributeType::Disabled;
}

Shader::RuntimeInfo MakeRuntimeInfo(std::span<const Shader::IR::Program> programs,
                                    const GraphicsPipelineCacheKey& key,
                                    const Shader::IR::Program& program,
                                    const Shader::IR::Program* previous_program,
                                    const Vulkan::Device& device) {
    Shader::RuntimeInfo info;
    if (previous_program) {
        info.previous_stage_stores = previous_program->info.stores;
        info.previous_stage_legacy_stores_mapping = previous_program->info.legacy_stores_mapping;
        if (previous_program->is_geometry_passthrough) {
            info.previous_stage_stores.mask |= previous_program->info.passthrough.mask;
        }
    } else {
        info.previous_stage_stores.mask.set();
    }
    const Shader::Stage stage{program.stage};
    const bool has_geometry{key.unique_hashes[4] != 0 && !programs[4].is_geometry_passthrough};
    const bool gl_ndc{key.state.ndc_minus_one_to_one != 0};
    const float point_size{std::bit_cast<float>(key.state.point_size)};
    switch (stage) {
    case Shader::Stage::VertexB:
        if (!has_geometry) {
            if (key.state.topology == Maxwell::PrimitiveTopology::Points) {
                info.fixed_state_point_size = point_size;
            }
            if (key.state.xfb_enabled) {
                if (device.IsExtTransformFeedbackSupported()) {
                    auto [varyings, count] =
                        VideoCommon::MakeTransformFeedbackVaryings(key.state.xfb_state);
                    info.xfb_varyings = varyings;
                    info.xfb_count = count;
                } else {
                    LOG_WARNING(Render_Vulkan, "XFB requested in pipeline key but device lacks VK_EXT_transform_feedback; ignoring XFB decorations");
                }
            }
            info.convert_depth_mode = gl_ndc;
        }
        if (key.state.dynamic_vertex_input) {
            for (size_t index = 0; index < Maxwell::NumVertexAttributes; ++index) {
                info.generic_input_types[index] = AttributeType(key.state, index);
            }
        } else {
            std::ranges::transform(key.state.attributes, info.generic_input_types.begin(),
                                   &CastAttributeType);
        }
        break;
    case Shader::Stage::TessellationEval:
        info.tess_clockwise = key.state.tessellation_clockwise != 0;
        info.tess_primitive = [&key] {
            const u32 raw{key.state.tessellation_primitive.Value()};
            switch (static_cast<Maxwell::Tessellation::DomainType>(raw)) {
            case Maxwell::Tessellation::DomainType::Isolines:
                return Shader::TessPrimitive::Isolines;
            case Maxwell::Tessellation::DomainType::Triangles:
                return Shader::TessPrimitive::Triangles;
            case Maxwell::Tessellation::DomainType::Quads:
                return Shader::TessPrimitive::Quads;
            }
            ASSERT(false);
            return Shader::TessPrimitive::Triangles;
        }();
        info.tess_spacing = [&] {
            const u32 raw{key.state.tessellation_spacing};
            switch (static_cast<Maxwell::Tessellation::Spacing>(raw)) {
            case Maxwell::Tessellation::Spacing::Integer:
                return Shader::TessSpacing::Equal;
            case Maxwell::Tessellation::Spacing::FractionalOdd:
                return Shader::TessSpacing::FractionalOdd;
            case Maxwell::Tessellation::Spacing::FractionalEven:
                return Shader::TessSpacing::FractionalEven;
            }
            ASSERT(false);
            return Shader::TessSpacing::Equal;
        }();
        break;
    case Shader::Stage::Geometry:
        if (program.output_topology == Shader::OutputTopology::PointList) {
            info.fixed_state_point_size = point_size;
        }
        if (key.state.xfb_enabled != 0) {
            if (device.IsExtTransformFeedbackSupported()) {
                auto [varyings, count] =
                    VideoCommon::MakeTransformFeedbackVaryings(key.state.xfb_state);
                info.xfb_varyings = varyings;
                info.xfb_count = count;
            } else {
                LOG_WARNING(Render_Vulkan, "XFB requested in pipeline key but device lacks VK_EXT_transform_feedback; ignoring XFB decorations");
            }
        }
        info.convert_depth_mode = gl_ndc;
        break;
    case Shader::Stage::Fragment: {
        info.alpha_test_func = MaxwellToCompareFunction(
            key.state.UnpackComparisonOp(key.state.alpha_test_func.Value()));
        info.alpha_test_reference = std::bit_cast<float>(key.state.alpha_test_ref);

        info.dual_source_blend = key.state.attachment0_dual_source_blend != 0;

        if (device.IsMoltenVK()) {
            for (size_t i = 0; i < 8; ++i) {
                const auto format = static_cast<Tegra::RenderTargetFormat>(key.state.color_formats[i]);
                const auto pixel_format = VideoCore::Surface::PixelFormatFromRenderTargetFormat(format);
                if (VideoCore::Surface::IsPixelFormatInteger(pixel_format)) {
                    if (VideoCore::Surface::IsPixelFormatSignedInteger(pixel_format)) {
                        info.color_output_types[i] = Shader::AttributeType::SignedInt;
                    } else {
                        info.color_output_types[i] = Shader::AttributeType::UnsignedInt;
                    }
                } else {
                    info.color_output_types[i] = Shader::AttributeType::Float;
                }
            }
        }
        break;
    }
    default:
        break;
    }
    switch (key.state.topology) {
    case Maxwell::PrimitiveTopology::Points:
        info.input_topology = Shader::InputTopology::Points;
        break;
    case Maxwell::PrimitiveTopology::Lines:
    case Maxwell::PrimitiveTopology::LineLoop:
    case Maxwell::PrimitiveTopology::LineStrip:
        info.input_topology = Shader::InputTopology::Lines;
        break;
    case Maxwell::PrimitiveTopology::Triangles:
    case Maxwell::PrimitiveTopology::TriangleStrip:
    case Maxwell::PrimitiveTopology::TriangleFan:
    case Maxwell::PrimitiveTopology::Quads:
    case Maxwell::PrimitiveTopology::QuadStrip:
    case Maxwell::PrimitiveTopology::Polygon:
    case Maxwell::PrimitiveTopology::Patches:
        info.input_topology = Shader::InputTopology::Triangles;
        break;
    case Maxwell::PrimitiveTopology::LinesAdjacency:
    case Maxwell::PrimitiveTopology::LineStripAdjacency:
        info.input_topology = Shader::InputTopology::LinesAdjacency;
        break;
    case Maxwell::PrimitiveTopology::TrianglesAdjacency:
    case Maxwell::PrimitiveTopology::TriangleStripAdjacency:
        info.input_topology = Shader::InputTopology::TrianglesAdjacency;
        break;
    }
    info.force_early_z = key.state.early_z != 0;
    info.y_negate = key.state.y_negate != 0;
    return info;
}

size_t GetTotalPipelineWorkers() {
    const size_t max_core_threads =
        std::max<size_t>(static_cast<size_t>(std::thread::hardware_concurrency()), 2ULL) - 1ULL;
#ifdef __ANDROID__
    const int configured = AndroidSettings::values.pipeline_worker_count.GetValue();
    const int clamped = std::clamp(configured, 2, 8);
    const size_t desired = static_cast<size_t>(clamped);
    if (desired == 0) {
        return 1ULL;
    }
    return std::min(max_core_threads, desired);
#else
    return max_core_threads;
#endif
}

/// Why a pipeline can't be joined from pipeline libraries, and is compiled in full instead.
enum class LibraryFallback : u8 {
    NoFragmentShader,
    LegacyAttributes,
    UnwrittenInput,
    TooManyImages,
    TranslateFailed,
};
constexpr std::array<std::string_view, 5> LIBRARY_FALLBACK_NAMES{
    "no fragment shader",
    "the fragment shader reads legacy attributes",
    "the fragment shader reads an output the vertex half doesn't write",
    "too many textures or images for the rescaling bits",
    "the fragment shader failed to translate for its library",
};

/// Folds the hash of one stage's SPIR-V into the hash of the stages before it.
u64 FoldCodeHash(u64 previous, std::span<const u32> code) {
    return (previous * 0x9E3779B97F4A7C15ULL) ^
           Common::CityHash64(reinterpret_cast<const char*>(code.data()), code.size_bytes());
}

/// Whether the textures and images of a half fit the rescaling bits it gets in a pipeline joined
/// from libraries: bits 0 to 63 and 0 to 31 for the stages before the fragment shader, the bits
/// after those for the fragment shader.
bool FitsLibraryScaling(std::span<const Shader::Info* const> infos, LibraryHalf half) {
    size_t textures{};
    size_t images{};
    const size_t begin{half == LibraryHalf::Front ? 0U : infos.size() - 1};
    const size_t end{half == LibraryHalf::Front ? infos.size() - 1 : infos.size()};
    for (size_t stage = begin; stage < end; ++stage) {
        if (infos[stage]) {
            textures += infos[stage]->texture_descriptors.size();
            images += infos[stage]->image_descriptors.size();
        }
    }
    if (half == LibraryHalf::Front) {
        return textures <= LIBRARY_FRAGMENT_TEXTURE_SCALING_INDEX &&
               images <= LIBRARY_FRAGMENT_IMAGE_SCALING_INDEX;
    }
    return textures <= Shader::Backend::SPIRV::NUM_TEXTURE_SCALING_WORDS * 32 -
                           LIBRARY_FRAGMENT_TEXTURE_SCALING_INDEX &&
           images <= Shader::Backend::SPIRV::NUM_IMAGE_SCALING_WORDS * 32 -
                         LIBRARY_FRAGMENT_IMAGE_SCALING_INDEX;
}

/// Translates a fragment shader for the library of a joined pipeline: its resources in
/// descriptor set 1 from binding 0, its rescaling bits after the other half's, and an input for
/// every generic it reads, since the library doesn't know the stage before it. `front_stores`
/// is what that stage writes, when it is known, so a fragment shader reading more is caught.
std::optional<LibraryFallback> TranslateLibraryFragment(
    ShaderPools& pools, Shader::Environment& env, const Shader::HostTranslateInfo& host_info,
    const Shader::Profile& profile, std::span<const Shader::IR::Program> programs,
    const GraphicsPipelineCacheKey& key, const Device& device,
    const Shader::VaryingState* front_stores, Shader::IR::Program& program,
    std::vector<u32>& code) try {
    const u32 cfg_offset{static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader))};
    Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, false);
    program = TranslateProgram(pools.inst, pools.block, env, cfg, host_info);
    if (program.info.loads.Legacy()) {
        // Legacy attributes are matched to the generics the stage before left free.
        return LibraryFallback::LegacyAttributes;
    }
    if (front_stores) {
        for (size_t index = 0; index < Shader::IR::NUM_GENERICS; ++index) {
            if (program.info.loads.Generic(index) && !front_stores->Generic(index)) {
                return LibraryFallback::UnwrittenInput;
            }
        }
    }
    const std::array<const Shader::Info*, Maxwell::MaxShaderStage> infos{nullptr, nullptr, nullptr,
                                                                         nullptr, &program.info};
    if (!FitsLibraryScaling(infos, LibraryHalf::Back)) {
        return LibraryFallback::TooManyImages;
    }
    const Shader::RuntimeInfo runtime_info{
        MakeRuntimeInfo(programs, key, program, nullptr, device)};
    ConvertLegacyToGeneric(program, runtime_info);
    Shader::Backend::Bindings bindings{};
    bindings.texture_scaling_index = LIBRARY_FRAGMENT_TEXTURE_SCALING_INDEX;
    bindings.image_scaling_index = LIBRARY_FRAGMENT_IMAGE_SCALING_INDEX;
    bindings.descriptor_set = 1;
    code = EmitSPIRV(profile, runtime_info, program, bindings);
    return std::nullopt;
} catch (const Shader::Exception& exception) {
    LOG_DEBUG(Render_Vulkan, "[Preforged] fragment shader for a library failed: {}",
              exception.what());
    return LibraryFallback::TranslateFailed;
}

/// Names a library compiled from a shader of the game files with a state, so later boots skip
/// it without translating it again.
u64 LibrarySource(LibraryHalf half, u64 shader, u64 state_hash) {
    const std::array<u64, 3> words{static_cast<u64>(half) + 1, shader, state_hash};
    return Common::CityHash64(reinterpret_cast<const char*>(words.data()), sizeof(words));
}

/// x, y, z and w of a viewport as they are.
constexpr u16 IDENTITY_VIEWPORT_SWIZZLE = 0x6420;

/// The state the shaders from the game files are compiled with on a game's first start, before
/// it has drawn anything: triangles, one RGBA8 color target and a depth target, no MSAA, no alpha
/// test or transform feedback, the y direction and depth range the GPU starts with. Of the state,
/// a shader's SPIR-V only depends on the primitive class, alpha test, early Z, the y direction,
/// the depth range (without VK_EXT_depth_clip_control) and the attribute types, which are read
/// from each vertex shader's code. The rest is dynamic or doesn't change the shaders.
GraphicsPipelineCacheKey StarterKey(const DynamicFeatures& features) {
    GraphicsPipelineCacheKey key{};
    FixedPipelineState& state{key.state};
    state.driver_id = features.driver_id;
    state.driver_version = features.driver_version;
    state.extended_dynamic_state.Assign(features.has_extended_dynamic_state ? 1 : 0);
    state.extended_dynamic_state_2.Assign(features.has_extended_dynamic_state_2 ? 1 : 0);
    state.extended_dynamic_state_2_logic_op.Assign(
        features.has_extended_dynamic_state_2_logic_op ? 1 : 0);
    state.extended_dynamic_state_3_blend.Assign(features.has_extended_dynamic_state_3_blend ? 1
                                                                                          : 0);
    state.extended_dynamic_state_3_enables.Assign(
        features.has_extended_dynamic_state_3_enables ? 1 : 0);
    state.color_write_enable_dynamic.Assign(features.has_color_write_enable ? 1 : 0);
    state.dynamic_vertex_input.Assign(features.has_dynamic_vertex_input ? 1 : 0);
    state.polygon_mode.Assign(FixedPipelineState::PackPolygonMode(Maxwell::PolygonMode::Fill));
    state.topology.Assign(Maxwell::PrimitiveTopology::Triangles);
    state.msaa_mode.Assign(Tegra::Texture::MsaaMode::Msaa1x1);
    state.alpha_test_func.Assign(
        FixedPipelineState::PackComparisonOp(Maxwell::ComparisonOp::Always_GL));
    state.depth_enabled.Assign(1);
    state.depth_format.Assign(static_cast<u32>(Tegra::DepthFormat::Z24_UNORM_S8_UINT));
    // The first vertex, as the GPU starts, where the driver has it.
    state.provoking_vertex_last.Assign(features.has_provoking_vertex &&
                                               !features.has_provoking_vertex_first_mode &&
                                               features.has_provoking_vertex_last_mode
                                           ? 1
                                           : 0);
    state.color_formats.fill(static_cast<u8>(Tegra::RenderTargetFormat::NONE));
    state.color_formats[0] = static_cast<u8>(Tegra::RenderTargetFormat::A8B8G8R8_UNORM);
    state.point_size = std::bit_cast<u32>(1.0f);
    state.viewport_swizzles.fill(IDENTITY_VIEWPORT_SWIZZLE);
    for (FixedPipelineState::BlendingAttachment& attachment : state.attachments) {
        attachment.mask_r.Assign(1);
        attachment.mask_g.Assign(1);
        attachment.mask_b.Assign(1);
        attachment.mask_a.Assign(1);
    }
    if (!features.has_dynamic_vertex_input) {
        // Without dynamic vertex input the attributes are part of the pipeline: all float.
        for (FixedPipelineState::VertexAttribute& attribute : state.attributes) {
            attribute.enabled.Assign(1);
            attribute.type.Assign(static_cast<u32>(Maxwell::VertexAttribute::Type::Float));
            attribute.size.Assign(
                static_cast<u32>(Maxwell::VertexAttribute::Size::Size_R32_G32_B32_A32));
        }
    }
    return key;
}

/// Logs how the pipelines the game drew on earlier boots differ in the state that changes their
/// shaders, against the starter state a first start compiles with, to see how well it fits.
void LogPlayedStates(std::span<const GraphicsPipelineCacheKey> keys) {
    if (keys.empty()) {
        return;
    }
    using Topology = Maxwell::PrimitiveTopology;
    const u32 always{FixedPipelineState::PackComparisonOp(Maxwell::ComparisonOp::Always_GL)};
    size_t points{};
    size_t lines{};
    size_t patches{};
    size_t other_stages{};
    size_t lower_left{};
    size_t minus_one{};
    size_t alpha_test{};
    size_t early_z{};
    size_t msaa{};
    size_t xfb{};
    size_t dual_source{};
    size_t like_starter{};
    for (const GraphicsPipelineCacheKey& key : keys) {
        const FixedPipelineState& state{key.state};
        const Topology topology{state.topology.Value()};
        const bool is_points{topology == Topology::Points};
        points += is_points ? 1 : 0;
        lines += topology == Topology::Lines || topology == Topology::LineLoop ||
                         topology == Topology::LineStrip || topology == Topology::LinesAdjacency ||
                         topology == Topology::LineStripAdjacency
                     ? 1
                     : 0;
        patches += topology == Topology::Patches ? 1 : 0;
        const bool more_stages{key.unique_hashes[0] != 0 || key.unique_hashes[2] != 0 ||
                               key.unique_hashes[3] != 0 || key.unique_hashes[4] != 0};
        other_stages += more_stages ? 1 : 0;
        lower_left += state.y_negate != 0 ? 1 : 0;
        minus_one += state.ndc_minus_one_to_one != 0 ? 1 : 0;
        const bool alpha{state.alpha_test_func != always};
        alpha_test += alpha ? 1 : 0;
        early_z += state.early_z != 0 ? 1 : 0;
        msaa += state.msaa_mode.Value() != Tegra::Texture::MsaaMode::Msaa1x1 ? 1 : 0;
        xfb += state.xfb_enabled != 0 ? 1 : 0;
        dual_source += state.attachment0_dual_source_blend != 0 ? 1 : 0;
        like_starter += !more_stages && !is_points && !alpha && state.early_z == 0 &&
                                state.xfb_enabled == 0 &&
                                state.attachment0_dual_source_blend == 0
                            ? 1
                            : 0;
    }
    LOG_INFO(Render_Vulkan,
             "[Preforged] the {} pipelines the game drew: {} points, {} lines, {} patches, {} "
             "with more stages than a vertex and a fragment shader, {} with the window origin "
             "lower left, {} with depth -1 to 1, {} with alpha test, {} with early Z, {} with "
             "MSAA, {} with transform feedback, {} with dual source blending; {} change their "
             "shaders as the starter state of a first start does (by primitive, stages, alpha "
             "test, early Z, transform feedback and blending)",
             keys.size(), points, lines, patches, other_stages, lower_left, minus_one, alpha_test,
             early_z, msaa, xfb, dual_source, like_starter);
}

} // Anonymous namespace

size_t ComputePipelineCacheKey::Hash() const noexcept {
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(this), sizeof *this);
    return static_cast<size_t>(hash);
}

bool ComputePipelineCacheKey::operator==(const ComputePipelineCacheKey& rhs) const noexcept {
    return std::memcmp(&rhs, this, sizeof *this) == 0;
}

size_t GraphicsPipelineCacheKey::Hash() const noexcept {
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(this), Size());
    return static_cast<size_t>(hash);
}

bool GraphicsPipelineCacheKey::operator==(const GraphicsPipelineCacheKey& rhs) const noexcept {
    return std::memcmp(&rhs, this, Size()) == 0;
}

PipelineCache::PipelineCache(Tegra::MaxwellDeviceMemoryManager& device_memory_,
                             const Device& device_, Scheduler& scheduler_,
                             DescriptorPool& descriptor_pool_,
                             GuestDescriptorQueue& guest_descriptor_queue_,
                             DescriptorBufferRing& descriptor_buffer_ring_,
                             RenderPassCache& render_pass_cache_, BufferCache& buffer_cache_,
                             TextureCache& texture_cache_, VideoCore::ShaderNotify& shader_notify_)
    : VideoCommon::ShaderCache{device_memory_}, device{device_}, scheduler{scheduler_},
      descriptor_pool{descriptor_pool_}, guest_descriptor_queue{guest_descriptor_queue_},
      descriptor_buffer_ring{descriptor_buffer_ring_}, render_pass_cache{render_pass_cache_},
      buffer_cache{buffer_cache_}, texture_cache{texture_cache_}, shader_notify{shader_notify_},
      use_asynchronous_shaders{Settings::values.use_asynchronous_shaders.GetValue()},
      use_vulkan_pipeline_cache{Settings::values.use_vulkan_driver_pipeline_cache.GetValue()},
      pipeline_libraries{device_}, use_pipeline_libraries{device_.CanJoinPipelineLibraries()},
      join_every_pipeline{use_pipeline_libraries &&
                          Common::Preforged::Options().join_every_pipeline},
      workers(device.HasBrokenParallelShaderCompiling() ? 1ULL : GetTotalPipelineWorkers(),
              "VkPipelineBuilder", {}, Common::ThreadPlacement::Background),
      full_worker(1, "VkPipelineFull", {}, Common::ThreadPlacement::Background),
      serialization_thread(1, "VkPipelineSerialization", {}, Common::ThreadPlacement::Background) {
    GraphicsPipeline::ResetPlayTally();
    ComputePipeline::ResetPlayTally();
    const auto& float_control{device.FloatControlProperties()};
    const VkDriverId driver_id{device.GetDriverID()};
    const VkShaderStageFlags subgroup_stages{device.GetSubgroupSupportedStages()};
    const auto subgroup_stage_bit{[subgroup_stages](VkShaderStageFlags flag, Shader::Stage stage) {
        return (subgroup_stages & flag) != 0 ? (1u << static_cast<u32>(stage)) : 0u;
    }};
    const u32 supported_subgroup_stages{
        subgroup_stage_bit(VK_SHADER_STAGE_VERTEX_BIT, Shader::Stage::VertexA) |
        subgroup_stage_bit(VK_SHADER_STAGE_VERTEX_BIT, Shader::Stage::VertexB) |
        subgroup_stage_bit(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
                           Shader::Stage::TessellationControl) |
        subgroup_stage_bit(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
                           Shader::Stage::TessellationEval) |
        subgroup_stage_bit(VK_SHADER_STAGE_GEOMETRY_BIT, Shader::Stage::Geometry) |
        subgroup_stage_bit(VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Stage::Fragment) |
        subgroup_stage_bit(VK_SHADER_STAGE_COMPUTE_BIT, Shader::Stage::Compute)};
    profile = Shader::Profile{
        .supported_spirv = device.SupportedSpirvVersion(),
        .unified_descriptor_binding = true,
        .support_descriptor_aliasing = device.IsDescriptorAliasingSupported(),
        .support_int8 = device.IsInt8Supported(),
        .support_uniform_and_storage_buffer_8bit =
            device.IsUniformAndStorageBuffer8BitAccessSupported(),
        .support_storage_buffer_8bit = device.IsStorageBuffer8BitAccessSupported(),
        .support_int16 = device.IsShaderInt16Supported(),
        .support_uniform_and_storage_buffer_16bit =
            device.IsUniformAndStorageBuffer16BitAccessSupported(),
        .support_storage_buffer_16bit = device.IsStorageBuffer16BitAccessSupported(),
        .support_int64 = device.IsShaderInt64Supported(),
        .support_vertex_instance_id = false,
        .support_float_controls = device.IsKhrShaderFloatControlsSupported(),
        .support_separate_denorm_behavior =
            float_control.denormBehaviorIndependence == VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
        .support_separate_rounding_mode =
            float_control.roundingModeIndependence == VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
        .support_fp16_denorm_preserve = float_control.shaderDenormPreserveFloat16 != VK_FALSE,
        .support_fp32_denorm_preserve = float_control.shaderDenormPreserveFloat32 != VK_FALSE,
        .support_fp16_denorm_flush = float_control.shaderDenormFlushToZeroFloat16 != VK_FALSE,
        .support_fp32_denorm_flush = float_control.shaderDenormFlushToZeroFloat32 != VK_FALSE,
        .support_fp16_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat16 != VK_FALSE,
        .support_fp32_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat32 != VK_FALSE,
        .support_fp64_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat64 != VK_FALSE,
        .support_explicit_workgroup_layout = device.IsKhrWorkgroupMemoryExplicitLayoutSupported(),
        .support_workgroup_layout_8bit_access =
            device.IsWorkgroupMemoryExplicitLayout8BitAccessSupported(),
        .support_workgroup_layout_16bit_access =
            device.IsWorkgroupMemoryExplicitLayout16BitAccessSupported(),
        .support_shader_quad_control = device.IsKhrShaderQuadControlSupported(),
        .support_quad_shuffles = device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_QUAD_BIT),
        .support_vote = device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_VOTE_BIT),
        .supported_subgroup_stages = supported_subgroup_stages,
        .support_viewport_index_layer_non_geometry =
            device.IsExtShaderViewportIndexLayerSupported(),
        .support_viewport_mask = device.IsNvViewportArray2Supported(),
        .support_typeless_image_loads = device.IsFormatlessImageLoadSupported(),
        .support_demote_to_helper_invocation =
            device.IsExtShaderDemoteToHelperInvocationSupported(),
        .support_int64_atomics = device.IsExtShaderAtomicInt64Supported(),
        .support_shared_int64_atomics = device.IsSharedInt64AtomicsSupported(),
        .support_derivative_control = true,
        .support_geometry_shader_passthrough = device.IsNvGeometryShaderPassthroughSupported(),
        .support_native_ndc = device.IsExtDepthClipControlSupported(),
        .support_scaled_attributes = !device.MustEmulateScaledFormats(),
        .support_multi_viewport = device.SupportsMultiViewport(),
        .support_geometry_streams = device.AreTransformFeedbackGeometryStreamsSupported(),
        .support_sampled_image_array_nonuniform_indexing =
            device.IsSampledImageArrayNonUniformIndexingSupported(),
        .support_storage_image_array_nonuniform_indexing =
            device.IsStorageImageArrayNonUniformIndexingSupported(),
        .support_uniform_texel_buffer_array_nonuniform_indexing =
            device.IsUniformTexelBufferArrayNonUniformIndexingSupported(),
        .support_storage_texel_buffer_array_nonuniform_indexing =
            device.IsStorageTexelBufferArrayNonUniformIndexingSupported(),

        .warp_size_potentially_larger_than_guest = device.IsWarpSizePotentiallyBiggerThanGuest(),

        .lower_left_origin_mode = false,
        .need_declared_frag_colors = false,
        .need_gather_subpixel_offset = driver_id == VK_DRIVER_ID_AMD_PROPRIETARY ||
                                       driver_id == VK_DRIVER_ID_AMD_OPEN_SOURCE ||
                                       driver_id == VK_DRIVER_ID_MESA_RADV ||
                                       driver_id == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS ||
                                       driver_id == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA,

        .has_broken_spirv_clamp = driver_id == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS,
        .has_broken_spirv_position_input = driver_id == false,
        .has_broken_unsigned_image_offsets = false,
        .has_broken_signed_operations = false,
        .has_broken_fp16_float_controls = driver_id == VK_DRIVER_ID_NVIDIA_PROPRIETARY,
        .has_broken_fp32_denorm_flush = driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY,
        .ignore_nan_fp_comparisons = false,
        .has_broken_spirv_subgroup_mask_vector_extract_dynamic = false,
        .has_broken_robust =
            device.IsNvidia() && device.GetNvidiaArch() <= NvidiaArchitecture::Arch_Pascal,
        .min_ssbo_alignment = device.GetStorageBufferAlignment(),
        .max_user_clip_distances = device.GetMaxUserClipDistances()
    };

    host_info = Shader::HostTranslateInfo{
        .min_ssbo_alignment = device.GetStorageBufferAlignment(),
        .max_per_stage_descriptor_sampled_images = device.GetMaxPerStageDescriptorSampledImages(),
        .max_per_stage_resources = device.GetMaxPerStageResources(),
        .max_descriptor_set_samplers = device.GetMaxDescriptorSetSamplers(),
        .max_descriptor_set_uniform_buffers = device.GetMaxDescriptorSetUniformBuffers(),
        .max_descriptor_set_uniform_buffers_dynamic = device.GetMaxDescriptorSetUniformBuffersDynamic(),
        .max_descriptor_set_storage_buffers = device.GetMaxDescriptorSetStorageBuffers(),
        .max_descriptor_set_storage_buffers_dynamic = device.GetMaxDescriptorSetStorageBuffersDynamic(),
        .max_descriptor_set_sampled_images = device.GetMaxDescriptorSetSampledImages(),
        .max_descriptor_set_storage_images = device.GetMaxDescriptorSetStorageImages(),
        .max_descriptor_set_input_attachements = device.GetMaxDescriptorSetInputAttachments(),
        .support_float64 = device.IsFloat64Supported(),
        .support_float16 = device.IsFloat16Supported(),
        .support_int64 = device.IsShaderInt64Supported(),
        .needs_demote_reorder = driver_id == VK_DRIVER_ID_AMD_PROPRIETARY ||
                                driver_id == VK_DRIVER_ID_AMD_OPEN_SOURCE ||
                                driver_id == VK_DRIVER_ID_SAMSUNG_PROPRIETARY,
        .support_snorm_render_buffer = true,
        .support_viewport_index_layer = device.IsExtShaderViewportIndexLayerSupported(),
        .support_geometry_shader_passthrough = device.IsNvGeometryShaderPassthroughSupported(),
        .support_conditional_barrier = device.SupportsConditionalBarriers(),
    };
    host_info.ApplyDescriptorLimitPolicy();

    if (device.GetMaxVertexInputAttributes() < Maxwell::NumVertexAttributes) {
        LOG_WARNING(Render_Vulkan, "maxVertexInputAttributes is too low: {} < {}",
                    device.GetMaxVertexInputAttributes(), Maxwell::NumVertexAttributes);
    }
    if (device.GetMaxVertexInputBindings() < Maxwell::NumVertexArrays) {
        LOG_WARNING(Render_Vulkan, "maxVertexInputBindings is too low: {} < {}",
                    device.GetMaxVertexInputBindings(), Maxwell::NumVertexArrays);
    }

    LOG_INFO(Render_Vulkan, "DynamicState setting value: {}", u32(Settings::values.dyna_state.GetValue()));
    LOG_INFO(Render_Vulkan, "[ShaderTiming] VK_EXT_graphics_pipeline_library available: {}",
             device.GetAvailableExtensions().contains(
                 VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME));
    {
        const auto& library_properties{device.GraphicsPipelineLibraryProperties()};
        LOG_INFO(Render_Vulkan,
                 "[Preforged] joining pipelines from shader libraries: {} (fast linking {}, "
                 "interpolation decorations only in the fragment shader {}, setting {}); "
                 "compiling joined pipelines in full afterwards: {}; test, join every pipeline: "
                 "{}",
                 use_pipeline_libraries ? "on" : "off",
                 library_properties.graphicsPipelineLibraryFastLinking != VK_FALSE,
                 library_properties.graphicsPipelineLibraryIndependentInterpolationDecoration !=
                     VK_FALSE,
                 Common::Preforged::Options().pipeline_libraries,
                 use_pipeline_libraries && Common::Preforged::Options().optimize_joined &&
                     !join_every_pipeline,
                 join_every_pipeline ? "on, so joined pipelines are kept as they are" : "off");
    }

    dynamic_features = {};
    dynamic_features.driver_id = device.GetDriverID();
    dynamic_features.driver_version = device.GetDriverVersion();

    // User granularity enforced in vulkan_device.cpp switch statement:
    //   Level 0: Core Dynamic States only
    //   Level 1: Core + EDS1
    //   Level 2: Core + EDS1 + EDS2 (accumulative)
    //   Level 3: Core + EDS1 + EDS2 + EDS3 (accumulative)
    // Here we only verify if extensions were successfully loaded by the device

    dynamic_features.has_extended_dynamic_state =
        device.IsExtExtendedDynamicStateSupported();

    dynamic_features.has_extended_dynamic_state_2 =
        device.IsExtExtendedDynamicState2Supported();
    dynamic_features.has_extended_dynamic_state_2_logic_op =
        device.IsExtExtendedDynamicState2ExtrasSupported();
    dynamic_features.has_extended_dynamic_state_2_patch_control_points = false;

    dynamic_features.has_extended_dynamic_state_3_blend =
        device.IsExtExtendedDynamicState3BlendingSupported();
    dynamic_features.has_extended_dynamic_state_3_enables =
        device.IsExtExtendedDynamicState3EnablesSupported();
    dynamic_features.has_color_write_enable =
        device.IsExtColorWriteEnableSupported();
    dynamic_features.has_dynamic_state3_depth_clamp_enable =
        dynamic_features.has_extended_dynamic_state_3_enables &&
        device.SupportsDynamicState3DepthClampEnable();
    dynamic_features.has_dynamic_state3_logic_op_enable =
        dynamic_features.has_extended_dynamic_state_3_enables &&
        device.SupportsDynamicState3LogicOpEnable();
    dynamic_features.has_dynamic_state3_line_stipple_enable =
        dynamic_features.has_extended_dynamic_state_3_enables &&
        device.SupportsDynamicState3LineStippleEnable();

    // VIDS: Independent toggle (not affected by dyna_state levels)
    dynamic_features.has_dynamic_vertex_input =
        device.IsExtVertexInputDynamicStateSupported() &&
        Settings::values.vertex_input_dynamic_state.GetValue();

    dynamic_features.has_provoking_vertex = device.IsExtProvokingVertexSupported();
    dynamic_features.has_provoking_vertex_first_mode =
        device.SupportsProvokingVertexFirstMode();
    dynamic_features.has_provoking_vertex_last_mode =
        device.SupportsProvokingVertexLastMode();
    dynamic_features.has_provoking_vertex_tf_preserve =
        device.SupportsTransformFeedbackProvokingVertexPreservation();
}

PipelineCache::~PipelineCache() {
    if (use_vulkan_pipeline_cache && !vulkan_pipeline_cache_filename.empty()) {
        SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                     CACHE_VERSION);
    }
}

GraphicsPipeline* PipelineCache::CurrentGraphicsPipeline() {

    if (!RefreshStages(graphics_key.unique_hashes)) {
        current_pipeline = nullptr;
        return nullptr;
    }
    graphics_key.state.Refresh(*maxwell3d, dynamic_features);

    if (current_pipeline) {
        GraphicsPipeline* const next{current_pipeline->Next(graphics_key)};
        if (next) {
            current_pipeline = next;
            return BuiltPipeline(current_pipeline);
        }
    }
    return CurrentGraphicsPipelineSlowPath();
}

ComputePipeline* PipelineCache::CurrentComputePipeline() {

    const ShaderInfo* const shader{ComputeShader()};
    if (!shader) {
        return nullptr;
    }
    const auto& qmd{kepler_compute->launch_description};
    const ComputePipelineCacheKey key{
        .unique_hash = shader->unique_hash,
        .shared_memory_size = qmd.shared_alloc,
        .workgroup_size{qmd.block_dim_x, qmd.block_dim_y, qmd.block_dim_z},
    };
    const auto [pair, is_new]{compute_cache.try_emplace(key)};
    auto& pipeline{pair->second};
    if (!is_new) {
        return pipeline.get();
    }
    pipeline = CreateComputePipeline(key, shader);
    return pipeline.get();
}

void PipelineCache::LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                                      const VideoCore::DiskResourceLoadCallback& callback) {
    if (title_id == 0) {
        return;
    }
    if (!pipeline_cache_filename.empty()) {
        serialization_thread.WaitForRequests();
        if (use_vulkan_pipeline_cache && !vulkan_pipeline_cache_filename.empty()) {
            SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                         CACHE_VERSION);
        }
    }

    const auto shader_dir{Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir)};
    const auto base_dir{shader_dir / fmt::format("{:016x}", title_id)};
    if (!Common::FS::CreateDir(shader_dir) || !Common::FS::CreateDir(base_dir)) {
        LOG_ERROR(Common_Filesystem, "Failed to create pipeline cache directories");
        return;
    }
    pipeline_cache_filename = base_dir / "vulkan.bin";

    OpenPreforgedStore(base_dir);
    if (use_pipeline_libraries) {
        pipeline_libraries.OpenStore(base_dir);
    }
    if (use_vulkan_pipeline_cache) {
        vulkan_pipeline_cache_filename = base_dir / "vulkan_pipelines.bin";
        Common::Preforged::LogMemoryUse("before loading the driver pipeline cache");
        vulkan_pipeline_cache =
            LoadVulkanPipelineCache(vulkan_pipeline_cache_filename, CACHE_VERSION);
        Common::Preforged::LogMemoryUse("after loading the driver pipeline cache");
    }

    struct {
        std::mutex mutex;
        size_t total{};
        size_t built{};
        bool has_loaded{};
        std::unique_ptr<PipelineStatistics> statistics;
    } state;

    if (device.IsKhrPipelineExecutablePropertiesEnabled()) {
        state.statistics = std::make_unique<PipelineStatistics>(device);
    }
    // Keys queued for building, so entries that collapse onto the same key after retargeting
    // are only built once. Only touched from this thread while the cache file is parsed.
    std::unordered_set<GraphicsPipelineCacheKey> queued_graphics_keys;
    std::vector<GraphicsPipelineCacheKey> played_keys;
    preforged_attribute_votes = {};
    size_t retargeted_graphics{};
    const auto load_compute{[&](std::ifstream& file, FileEnvironment env) {
        ComputePipelineCacheKey key;
        file.read(reinterpret_cast<char*>(&key), sizeof(key));

        workers.QueueWork([this, key, env_ = std::move(env), &state, &callback]() mutable {
            ShaderPools pools;
            auto pipeline{CreateComputePipeline(pools, key, env_, state.statistics.get(), false)};
            std::scoped_lock lock{state.mutex};
            if (pipeline) {
                compute_cache.emplace(key, std::move(pipeline));
            }
            ++state.built;
            if (state.has_loaded) {
                callback(VideoCore::LoadCallbackStage::Build, state.built, state.total);
            }
        });
        ++state.total;
    }};
    const auto load_graphics{[&](std::ifstream& file, std::vector<FileEnvironment> envs) {
        GraphicsPipelineCacheKey key;
        file.read(reinterpret_cast<char*>(&key), sizeof(key));

        if ((key.state.extended_dynamic_state != 0) !=
                dynamic_features.has_extended_dynamic_state ||
            (key.state.extended_dynamic_state_2 != 0) !=
                dynamic_features.has_extended_dynamic_state_2 ||
            (key.state.extended_dynamic_state_2_logic_op != 0) !=
                dynamic_features.has_extended_dynamic_state_2_logic_op ||
            (key.state.extended_dynamic_state_3_blend != 0) !=
                dynamic_features.has_extended_dynamic_state_3_blend ||
            (key.state.extended_dynamic_state_3_enables != 0) !=
                dynamic_features.has_extended_dynamic_state_3_enables ||
            (key.state.color_write_enable_dynamic != 0) !=
                dynamic_features.has_color_write_enable ||
            (key.state.dynamic_vertex_input != 0) !=
                dynamic_features.has_dynamic_vertex_input) {
            return;
        }

        const bool key_requests_provoking_last = key.state.provoking_vertex_last != 0;
        if (key_requests_provoking_last && !dynamic_features.has_provoking_vertex_last_mode) {
            return;
        }

        const bool key_uses_transform_feedback = key.state.xfb_enabled != 0;
        if (key_uses_transform_feedback && key_requests_provoking_last &&
            !dynamic_features.has_provoking_vertex_tf_preserve) {
            return;
        }

        // The key records the driver it was created with, but nothing else reads those fields.
        // Entries saved under an older driver (or another machine's) would be built here and then
        // never match at runtime, so every one of them would recompile mid-game. Retarget them
        // to the current driver so the preload is actually used.
        if (key.state.driver_id != dynamic_features.driver_id ||
            key.state.driver_version != dynamic_features.driver_version) {
            key.state.driver_id = dynamic_features.driver_id;
            key.state.driver_version = dynamic_features.driver_version;
            ++retargeted_graphics;
        }
        if (!queued_graphics_keys.insert(key).second) {
            return;
        }
        played_keys.push_back(key);

        workers.QueueWork([this, key, envs_ = std::move(envs), &state, &callback]() mutable {
            ShaderPools pools;
            boost::container::static_vector<Shader::Environment*, 5> env_ptrs;
            for (auto& env : envs_) {
                env_ptrs.push_back(&env);
            }
            PreforgedCode code;
            // Also keeps the libraries of the pipeline's two halves, so the game can join its
            // shaders with others in play.
            auto pipeline{CreateGraphicsPipeline(pools, key, MakeSpan(env_ptrs),
                                                 state.statistics.get(), false, {}, &code, true)};

            std::scoped_lock lock{state.mutex};
            if (pipeline) {
                graphics_cache.emplace(key, std::move(pipeline));
                CountPreforgedAttributeTypes(key, code);
            }
            ++state.built;
            if (state.has_loaded) {
                callback(VideoCore::LoadCallbackStage::Build, state.built, state.total);
            }
        });
        ++state.total;
    }};
    VideoCommon::LoadPipelines(stop_loading, pipeline_cache_filename, CACHE_VERSION, load_compute,
                               load_graphics);

    LOG_INFO(Render_Vulkan, "Total Pipeline Count: {}", state.total);
    if (retargeted_graphics != 0) {
        LOG_INFO(Render_Vulkan, "Retargeted {} cached pipelines to the current driver",
                 retargeted_graphics);
    }

    std::unique_lock lock{state.mutex};
    callback(VideoCore::LoadCallbackStage::Build, 0, state.total);
    state.has_loaded = true;
    lock.unlock();

    workers.WaitForRequests(stop_loading);
    Common::Preforged::LogMemoryUse("after building the played pipelines");
    if (join_every_pipeline) {
        LOG_INFO(Render_Vulkan,
                 "[Preforged] test, join every pipeline: of the {} pipelines the game drew, {}",
                 played_keys.size(), LibraryJoinSummary());
        // The counts in play start again.
        library_joins = 0;
        library_not_needed = 0;
        for (auto& count : library_fallbacks) {
            count = 0;
        }
        pipeline_libraries.ResetCounts();
    }

    PrecompilePreforgedPairs(base_dir, played_keys, stop_loading, callback);
    Common::Preforged::LogMemoryUse("after compiling game file pairs");

    if (use_vulkan_pipeline_cache) {
        SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                     CACHE_VERSION);
        size_t size = 0;
        vulkan_pipeline_cache.Read(&size, nullptr);
        last_cache_size.store(size, std::memory_order_relaxed);
        last_flush = std::chrono::steady_clock::now();
    }

    if (state.statistics) {
        state.statistics->Report();
    }
}

std::vector<PipelineCache::CachedShaderCode> PipelineCache::ReadCachedShaderCode(
    const std::filesystem::path& filename, std::stop_token stop_loading,
    std::vector<GraphicsPipelineCacheKey>* graphics_keys) {
    std::vector<CachedShaderCode> result;
    std::unordered_set<u64> seen;
    const auto add{[&](FileEnvironment& env) {
        const std::span<const u64> code{env.Code()};
        if (code.size() < 3) {
            return;
        }
        // The unique hash covers the code up to the terminating instruction. Pipelines store the
        // cached size, which adds the terminator, plus one more word from SetCachedSize.
        const u64 hash{Common::CityHash64(reinterpret_cast<const char*>(code.data()),
                                          (code.size() - 2) * sizeof(u64))};
        if (seen.insert(hash).second) {
            std::vector<Shader::TextureType> texture_types;
            for (const auto& [handle, type] : env.TextureTypes()) {
                texture_types.push_back(type);
            }
            std::vector<u64> code_copy(code.begin(), code.end());
            const size_t cbuf_value_count{env.CbufValueCount()};
            const Shader::Stage stage{env.ShaderStage()};
            result.push_back({hash, stage, std::move(code_copy), std::move(texture_types),
                              cbuf_value_count,
                              std::make_shared<FileEnvironment>(std::move(env))});
        }
    }};
    VideoCommon::LoadPipelines(
        stop_loading, filename, CACHE_VERSION,
        [&](std::ifstream& file, FileEnvironment env) {
            ComputePipelineCacheKey key;
            file.read(reinterpret_cast<char*>(&key), sizeof(key));
            add(env);
        },
        [&](std::ifstream& file, std::vector<FileEnvironment> envs) {
            GraphicsPipelineCacheKey key;
            file.read(reinterpret_cast<char*>(&key), sizeof(key));
            if (graphics_keys) {
                graphics_keys->push_back(key);
            }
            for (FileEnvironment& env : envs) {
                add(env);
            }
        });
    return result;
}

u64 PipelineCache::ShaderStateHash(const GraphicsPipelineCacheKey& key) {
    const FixedPipelineState& state{key.state};
    std::array<u32, 8 + Maxwell::NumVertexAttributes> words{};
    words[0] = static_cast<u32>(state.topology.Value());
    words[1] = state.topology == Maxwell::PrimitiveTopology::Points ? state.point_size : 0;
    words[2] = state.xfb_enabled;
    words[3] = state.ndc_minus_one_to_one;
    words[4] = state.alpha_test_func;
    words[5] = state.alpha_test_ref;
    words[6] = state.attachment0_dual_source_blend;
    words[7] = static_cast<u32>(state.msaa_mode.Value());
    for (size_t index = 0; index < Maxwell::NumVertexAttributes; ++index) {
        words[8 + index] = static_cast<u32>(
            state.dynamic_vertex_input ? AttributeType(state, index)
                                       : CastAttributeType(state.attributes[index]));
    }
    return Common::CityHash64(reinterpret_cast<const char*>(words.data()), sizeof(words));
}

namespace {
constexpr std::array<char, 4> PREFORGED_STORE_MAGIC{'P', 'F', 'D', 'S'};
// Version 2 records what each pair was translated to, so stored code is matched by SPIR-V.
// Version 3 compiles each vertex shader with its own attribute types; pairs stored by earlier
// builds have the most common types, so the store starts over once.
constexpr u32 PREFORGED_STORE_VERSION = 3;

struct PreforgedStoreHeader {
    std::array<char, 4> magic;
    u32 version;
    u32 driver_id;
    u32 driver_version;
};

struct PreforgedStoreRecord {
    u64 pair_id;
    u64 state_hash;
    u64 code_hash;
    u32 size;
    /// A bit for each attribute whose type was read from the vertex shader's code. Earlier builds
    /// wrote 0 here.
    u32 code_typed;
    PipelineCache::PreforgedShaderInputs inputs;
};

PipelineCache::PreforgedShaderInputs MakePreforgedInputs(const GraphicsPipelineCacheKey& key,
                                                         const Shader::Info& vertex_info) {
    const FixedPipelineState& state{key.state};
    PipelineCache::PreforgedShaderInputs inputs{};
    inputs.attributes.fill(0xFF);
    for (size_t index = 0; index < inputs.attributes.size(); ++index) {
        if (!vertex_info.loads.Generic(index)) {
            continue;
        }
        inputs.attributes[index] = static_cast<u8>(
            state.dynamic_vertex_input ? AttributeType(state, index)
                                       : CastAttributeType(state.attributes[index]));
    }
    inputs.points = state.topology == Maxwell::PrimitiveTopology::Points;
    inputs.point_size = inputs.points ? state.point_size : 0;
    inputs.xfb = state.xfb_enabled != 0;
    inputs.ndc_minus_one_to_one = state.ndc_minus_one_to_one != 0;
    inputs.alpha_test_func = static_cast<u8>(state.alpha_test_func.Value());
    inputs.alpha_test_ref = state.alpha_test_ref;
    inputs.dual_source_blend = state.attachment0_dual_source_blend != 0;
    inputs.early_z = state.early_z != 0;
    inputs.y_negate = state.y_negate != 0;
    return inputs;
}

std::string_view AttributeTypeName(u8 type) {
    switch (static_cast<Shader::AttributeType>(type)) {
    case Shader::AttributeType::Float:
        return "float";
    case Shader::AttributeType::SignedInt:
        return "int";
    case Shader::AttributeType::UnsignedInt:
        return "uint";
    case Shader::AttributeType::SignedScaled:
        return "int scaled";
    case Shader::AttributeType::UnsignedScaled:
        return "uint scaled";
    case Shader::AttributeType::Disabled:
        return "off";
    }
    return "?";
}

/// Lists what differs between the inputs a pair was compiled with and the ones the game uses.
std::string DescribeInputDifference(const PipelineCache::PreforgedShaderInputs& stored,
                                    const PipelineCache::PreforgedShaderInputs& game) {
    std::vector<std::string> parts;
    for (size_t index = 0; index < stored.attributes.size(); ++index) {
        if (stored.attributes[index] != game.attributes[index]) {
            parts.push_back(fmt::format("attribute {} {} instead of {}", index,
                                        AttributeTypeName(game.attributes[index]),
                                        AttributeTypeName(stored.attributes[index])));
        }
    }
    if (stored.points != game.points || stored.point_size != game.point_size) {
        parts.emplace_back("point size");
    }
    if (stored.xfb != game.xfb) {
        parts.emplace_back("transform feedback");
    }
    if (stored.ndc_minus_one_to_one != game.ndc_minus_one_to_one) {
        parts.emplace_back("depth range");
    }
    if (stored.alpha_test_func != game.alpha_test_func) {
        parts.push_back(fmt::format("alpha test {} instead of {}", game.alpha_test_func,
                                    stored.alpha_test_func));
    } else if (stored.alpha_test_ref != game.alpha_test_ref) {
        parts.emplace_back("alpha test reference");
    }
    if (stored.dual_source_blend != game.dual_source_blend) {
        parts.emplace_back("dual source blend");
    }
    if (stored.early_z != game.early_z) {
        parts.emplace_back("early depth test");
    }
    if (stored.y_negate != game.y_negate) {
        parts.emplace_back("y flip");
    }
    if (parts.empty()) {
        // Same state, so what the game files guessed for the shaders, such as texture types,
        // differs from what the game uses.
        return "same state, shader guesses differ";
    }
    return fmt::format("{}", fmt::join(parts, ", "));
}

/// What a vertex shader's code does with an attribute it reads: float math, or it takes the bits
/// as an integer. The game feeds an attribute an integer type when the shader works on its bits
/// as an integer, and a float type when the shader does float math with it.
constexpr u8 INPUT_USED_AS_FLOAT = 1;
constexpr u8 INPUT_USED_AS_INTEGER = 2;

struct InputUse {
    /// INPUT_USED_AS_FLOAT and INPUT_USED_AS_INTEGER for each attribute, both when the code does
    /// both, neither when it only passes the value on.
    std::array<u8, 32> kinds{};
    /// A bit for each attribute the shader reads.
    u32 read{};
};

/// Whether an instruction does float math with a float value it takes.
bool IsFloatMath(Shader::IR::Opcode opcode) {
    using Shader::IR::Opcode;
    switch (opcode) {
    case Opcode::FPAbs32:
    case Opcode::FPAdd32:
    case Opcode::FPFma32:
    case Opcode::FPMax32:
    case Opcode::FPMin32:
    case Opcode::FPMul32:
    case Opcode::FPNeg32:
    case Opcode::FPRecip32:
    case Opcode::FPRecipSqrt32:
    case Opcode::FPSqrt:
    case Opcode::FPSin:
    case Opcode::FPExp2:
    case Opcode::FPCos:
    case Opcode::FPLog2:
    case Opcode::FPSaturate32:
    case Opcode::FPClamp32:
    case Opcode::FPRoundEven32:
    case Opcode::FPFloor32:
    case Opcode::FPCeil32:
    case Opcode::FPTrunc32:
    case Opcode::FPOrdEqual32:
    case Opcode::FPUnordEqual32:
    case Opcode::FPOrdNotEqual32:
    case Opcode::FPUnordNotEqual32:
    case Opcode::FPOrdLessThan32:
    case Opcode::FPUnordLessThan32:
    case Opcode::FPOrdGreaterThan32:
    case Opcode::FPUnordGreaterThan32:
    case Opcode::FPOrdLessThanEqual32:
    case Opcode::FPUnordLessThanEqual32:
    case Opcode::FPOrdGreaterThanEqual32:
    case Opcode::FPUnordGreaterThanEqual32:
    case Opcode::FPIsNan32:
    case Opcode::ConvertS16F32:
    case Opcode::ConvertS32F32:
    case Opcode::ConvertS64F32:
    case Opcode::ConvertU16F32:
    case Opcode::ConvertU32F32:
    case Opcode::ConvertU64F32:
    case Opcode::ConvertF16F32:
    case Opcode::ConvertF64F32:
        return true;
    default:
        return false;
    }
}

/// Whether an instruction works on the bits of a 32-bit value it takes as an integer: integer
/// math, integer to float conversion, or an offset or address, as in a constant buffer read
/// indexed by a bone number.
bool IsIntegerMath(Shader::IR::Opcode opcode) {
    using Shader::IR::Opcode;
    switch (opcode) {
    case Opcode::IAdd32:
    case Opcode::ISub32:
    case Opcode::IMul32:
    case Opcode::SDiv32:
    case Opcode::UDiv32:
    case Opcode::INeg32:
    case Opcode::IAbs32:
    case Opcode::ShiftLeftLogical32:
    case Opcode::ShiftRightLogical32:
    case Opcode::ShiftRightArithmetic32:
    case Opcode::BitwiseAnd32:
    case Opcode::BitwiseOr32:
    case Opcode::BitwiseXor32:
    case Opcode::BitwiseNot32:
    case Opcode::BitFieldInsert:
    case Opcode::BitFieldSExtract:
    case Opcode::BitFieldUExtract:
    case Opcode::BitReverse32:
    case Opcode::BitCount32:
    case Opcode::FindSMsb32:
    case Opcode::FindUMsb32:
    case Opcode::SMin32:
    case Opcode::UMin32:
    case Opcode::SMax32:
    case Opcode::UMax32:
    case Opcode::SClamp32:
    case Opcode::UClamp32:
    case Opcode::SLessThan:
    case Opcode::ULessThan:
    case Opcode::IEqual:
    case Opcode::SLessThanEqual:
    case Opcode::ULessThanEqual:
    case Opcode::SGreaterThan:
    case Opcode::UGreaterThan:
    case Opcode::INotEqual:
    case Opcode::SGreaterThanEqual:
    case Opcode::UGreaterThanEqual:
    case Opcode::ConvertF32S8:
    case Opcode::ConvertF32S16:
    case Opcode::ConvertF32S32:
    case Opcode::ConvertF32U8:
    case Opcode::ConvertF32U16:
    case Opcode::ConvertF32U32:
    case Opcode::GetCbufU8:
    case Opcode::GetCbufS8:
    case Opcode::GetCbufU16:
    case Opcode::GetCbufS16:
    case Opcode::GetCbufU32:
    case Opcode::GetCbufF32:
    case Opcode::GetCbufU32x2:
    case Opcode::LoadLocal:
    case Opcode::LoadStorage32:
    case Opcode::LoadStorage64:
    case Opcode::LoadStorage128:
        return true;
    default:
        return false;
    }
}

/// Reads from a translated vertex shader whether it does float math with each attribute it
/// reads or works on its bits as an integer. Each read is followed through bit casts, selects
/// and phis to the instructions that use it.
InputUse InputUseFromCode(const Shader::IR::Program& program) {
    using Shader::IR::Inst;
    using Shader::IR::Opcode;
    InputUse use{};
    for (size_t index = 0; index < use.kinds.size(); ++index) {
        use.read |= program.info.loads.Generic(index) ? 1u << index : 0u;
    }
    std::unordered_map<const Inst*, std::vector<const Inst*>> users;
    std::vector<std::pair<const Inst*, u32>> reads;
    for (const Shader::IR::Block* const block : program.blocks) {
        for (const Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() == Opcode::GetAttribute) {
                const Shader::IR::Value attribute{inst.Arg(0)};
                if (attribute.Type() == Shader::IR::Type::Attribute &&
                    Shader::IR::IsGeneric(attribute.Attribute())) {
                    reads.emplace_back(&inst,
                                       Shader::IR::GenericAttributeIndex(attribute.Attribute()));
                }
            }
            for (size_t arg = 0; arg < inst.NumArgs(); ++arg) {
                if (const Inst* const producer{inst.Arg(arg).TryInstRecursive()}) {
                    users[producer].push_back(&inst);
                }
            }
        }
    }
    for (const auto& [read, index] : reads) {
        // Each value the read flows into, and whether it holds the bits as an integer.
        std::vector<std::pair<const Inst*, bool>> work{{read, false}};
        std::unordered_set<const Inst*> visited{read};
        while (!work.empty()) {
            const auto [value, bits] = work.back();
            work.pop_back();
            const auto it{users.find(value)};
            if (it == users.end()) {
                continue;
            }
            for (const Inst* const user : it->second) {
                const Opcode opcode{user->GetOpcode()};
                const bool same{opcode == Opcode::Phi || opcode == Opcode::Identity ||
                                opcode == (bits ? Opcode::SelectU32 : Opcode::SelectF32)};
                const bool cast{opcode == (bits ? Opcode::BitCastF32U32 : Opcode::BitCastU32F32)};
                if (same || cast) {
                    if (visited.insert(user).second) {
                        work.emplace_back(user, cast ? !bits : bits);
                    }
                    continue;
                }
                if (bits ? IsIntegerMath(opcode) : IsFloatMath(opcode)) {
                    use.kinds[index] |= bits ? INPUT_USED_AS_INTEGER : INPUT_USED_AS_FLOAT;
                }
            }
        }
    }
    return use;
}

/// Gives each attribute whose use the code makes clear float, or an integer type (the integer
/// type the game feeds more often there, else unsigned). Attributes the guess turned off stay
/// off. Returns a bit for each attribute it typed.
u32 ApplyInputUse(GraphicsPipelineCacheKey& key, const InputUse& use,
                  const std::array<std::array<size_t, 4>, 32>& votes) {
    u32 typed{};
    for (size_t index = 0; index < Maxwell::NumVertexAttributes; ++index) {
        const u8 kind{use.kinds[index]};
        const u64 current{key.state.DynamicAttributeType(index)};
        if (((use.read >> index) & 1) == 0 || current == 0 ||
            (kind != INPUT_USED_AS_FLOAT && kind != INPUT_USED_AS_INTEGER)) {
            continue;
        }
        u64 type{1};
        if (kind == INPUT_USED_AS_INTEGER) {
            type = current != 1 ? current : votes[index][2] > votes[index][3] ? 2 : 3;
        }
        key.state.attribute_types &= ~(u64{3} << (index * 2));
        key.state.attribute_types |= type << (index * 2);
        typed |= 1u << index;
    }
    return typed;
}
} // Anonymous namespace

void PipelineCache::OpenPreforgedStore(const std::filesystem::path& base_dir) {
    std::scoped_lock lock{preforged_store_mutex};
    preforged_store.clear();
    preforged_store_path = base_dir / "preforged_driver_store.bin";
    std::error_code ec;

    // Earlier builds compiled the pairs into the driver pipeline cache, which the driver keeps
    // in memory as a whole. Start that cache over once; the played pipelines are built again
    // from vulkan.bin on this boot and land in it.
    const auto warmed_path{base_dir / "preforged_warmed.bin"};
    if (std::filesystem::exists(warmed_path, ec)) {
        std::filesystem::remove(warmed_path, ec);
        std::filesystem::remove(base_dir / "preforged_percent.txt", ec);
        std::filesystem::remove(base_dir / "vulkan_pipelines.bin", ec);
        LOG_INFO(Render_Vulkan, "[Preforged] cleared the driver pipeline cache of pairs compiled "
                                "by earlier builds, they now go to storage");
    }

    const PreforgedStoreHeader expected{
        .magic = PREFORGED_STORE_MAGIC,
        .version = PREFORGED_STORE_VERSION,
        .driver_id = dynamic_features.driver_id,
        .driver_version = dynamic_features.driver_version,
    };
    u64 valid_end{0};
    u64 bytes{0};
    {
        const u64 file_size{std::filesystem::file_size(preforged_store_path, ec)};
        std::ifstream file{preforged_store_path, std::ios::binary};
        PreforgedStoreHeader header{};
        if (file && file.read(reinterpret_cast<char*>(&header), sizeof(header)) &&
            std::memcmp(&header, &expected, sizeof(header)) == 0) {
            valid_end = sizeof(header);
            PreforgedStoreRecord record{};
            while (valid_end + sizeof(record) <= file_size &&
                   file.read(reinterpret_cast<char*>(&record), sizeof(record))) {
                const u64 offset{valid_end + sizeof(record)};
                if (offset + record.size > file_size) {
                    break;
                }
                preforged_store.insert_or_assign(
                    record.pair_id,
                    PreforgedStoreEntry{offset, record.state_hash, record.size,
                                        PreforgedCode{record.code_hash, record.inputs},
                                        record.code_typed});
                valid_end = offset + record.size;
                bytes += record.size;
                file.seekg(static_cast<std::streamoff>(valid_end));
            }
        } else if (file) {
            LOG_INFO(Render_Vulkan,
                     "[Preforged] pair store is from another driver or build, starting over");
        }
    }
    if (valid_end == 0) {
        std::ofstream file{preforged_store_path, std::ios::binary | std::ios::trunc};
        file.write(reinterpret_cast<const char*>(&expected), sizeof(expected));
        valid_end = sizeof(expected);
    } else if (std::filesystem::file_size(preforged_store_path, ec) != valid_end) {
        // A write that was cut off, for example by a crash, is dropped.
        std::filesystem::resize_file(preforged_store_path, valid_end, ec);
    }
    LOG_INFO(Render_Vulkan, "[Preforged] pair store on storage: {} pairs, {} MiB",
             preforged_store.size(), bytes >> 20);
}

void PipelineCache::AppendPreforgedStore(u64 pair_id, u64 state_hash, const PreforgedCode& code,
                                         u32 code_typed, const vk::PipelineCache& cache) {
    size_t size{};
    if (cache.Read(&size, nullptr) != VK_SUCCESS || size == 0) {
        return;
    }
    std::vector<char> data(size);
    if (cache.Read(&size, data.data()) != VK_SUCCESS) {
        return;
    }
    std::scoped_lock lock{preforged_store_mutex};
    if (!preforged_store_out.is_open()) {
        return;
    }
    const PreforgedStoreRecord record{
        pair_id, state_hash, code.code_hash, static_cast<u32>(size), code_typed, code.inputs};
    const u64 offset{static_cast<u64>(preforged_store_out.tellp()) + sizeof(record)};
    preforged_store_out.write(reinterpret_cast<const char*>(&record), sizeof(record));
    preforged_store_out.write(data.data(), size);
    if (preforged_store_out) {
        preforged_store.insert_or_assign(
            pair_id, PreforgedStoreEntry{offset, state_hash, record.size, code, code_typed});
    }
}

void PipelineCache::CountPreforgedAttributeTypes(const GraphicsPipelineCacheKey& key,
                                                 const PreforgedCode& code) {
    if (!key.state.dynamic_vertex_input || key.unique_hashes[0] != 0 ||
        key.unique_hashes[1] == 0) {
        return;
    }
    for (size_t index = 0; index < code.inputs.attributes.size(); ++index) {
        if (code.inputs.attributes[index] != 0xFF) {
            ++preforged_attribute_votes[index][key.state.DynamicAttributeType(index)];
        }
    }
}

std::shared_ptr<vk::PipelineCache> PipelineCache::LoadPreforgedStoreEntry(
    const GraphicsPipelineCacheKey& key, const PreforgedCode& code, std::string& match) {
    if (key.unique_hashes[0] != 0 || key.unique_hashes[2] != 0 || key.unique_hashes[3] != 0 ||
        key.unique_hashes[4] != 0 || key.unique_hashes[1] == 0 || key.unique_hashes[5] == 0) {
        return {};
    }
    const u64 pair_id{(key.unique_hashes[1] * 0x9E3779B97F4A7C15ULL) ^ key.unique_hashes[5]};
    PreforgedStoreEntry entry{};
    std::filesystem::path path;
    {
        std::scoped_lock lock{preforged_store_mutex};
        const auto it{preforged_store.find(pair_id)};
        if (it == preforged_store.end()) {
            match = "not stored";
            return {};
        }
        entry = it->second;
        path = preforged_store_path;
    }
    // Driver code only helps when the driver gets the same SPIR-V. Handing it code for other
    // SPIR-V crashed the driver, so that stays behind its switch.
    if (entry.code.code_hash == code.code_hash) {
        match = "same code";
    } else {
        match = fmt::format("other code: {}",
                            DescribeInputDifference(entry.code.inputs, code.inputs));
        if (!Common::Preforged::Options().store_any_state) {
            return {};
        }
    }
    std::vector<char> data(entry.size);
    std::ifstream file{path, std::ios::binary};
    if (!file.seekg(static_cast<std::streamoff>(entry.offset)) ||
        !file.read(data.data(), entry.size)) {
        return {};
    }
    ++preforged_store_reads;
    return std::make_shared<vk::PipelineCache>(device.GetLogical().CreatePipelineCache({
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .initialDataSize = data.size(),
        .pInitialData = data.data(),
    }));
}

void PipelineCache::PrecompilePreforgedPairs(
    const std::filesystem::path& base_dir, std::span<const GraphicsPipelineCacheKey> played_keys,
    std::stop_token stop_loading, const VideoCore::DiskResourceLoadCallback& callback) {
    const auto programs_ptr{std::make_shared<std::optional<VideoCommon::PreforgedPrograms>>(
        VideoCommon::ReadPreforgedPrograms(base_dir / "preforged_programs.bin"))};
    LogPlayedStates(played_keys);
    if (!*programs_ptr) {
        return;
    }
    const VideoCommon::PreforgedPrograms& programs{**programs_ptr};
    const auto pair_id{[](u64 vertex, u64 fragment) {
        return (vertex * 0x9E3779B97F4A7C15ULL) ^ fragment;
    }};

    // Each vertex shader gets its own attribute types: the types the game drew it with, or for a
    // shader it hasn't drawn yet, float or integer by what its code does with each attribute.
    // Off gives every shader the type the game feeds most often at each attribute.
    const bool per_shader_types{Common::Preforged::Options().input_types_per_shader};

    // Pairs compiled on earlier boots, with the attribute types their vertex shader was
    // compiled with. A pair whose types differ from what would be guessed now is compiled again.
    // Types that were read from the shader's code stay as they are.
    struct Warmed {
        std::array<u8, 32> attributes;
        u32 code_typed;
    };
    std::unordered_map<u64, Warmed> warmed;
    {
        std::scoped_lock lock{preforged_store_mutex};
        for (const auto& [id, entry] : preforged_store) {
            warmed.emplace(id, Warmed{entry.code.inputs.attributes,
                                      per_shader_types ? entry.code_typed : 0u});
        }
    }
    size_t stored_with_other_attributes{};
    std::unordered_set<u64> played;

    // A pair compiled with a state that changes its shaders differently than the game will is
    // compiled again by the driver, so use a state the game drew this vertex shader with, or
    // else the state most of its pipelines use.
    std::unordered_map<u64, const GraphicsPipelineCacheKey*> vertex_templates;
    std::unordered_map<u64, const GraphicsPipelineCacheKey*> played_pair_keys;
    std::unordered_map<u64, std::pair<size_t, const GraphicsPipelineCacheKey*>> state_counts;
    for (const GraphicsPipelineCacheKey& key : played_keys) {
        played.insert(pair_id(key.unique_hashes[1], key.unique_hashes[5]));
        if (key.unique_hashes[0] != 0 || key.unique_hashes[2] != 0 ||
            key.unique_hashes[3] != 0 || key.unique_hashes[4] != 0) {
            continue;
        }
        played_pair_keys.try_emplace(pair_id(key.unique_hashes[1], key.unique_hashes[5]), &key);
        vertex_templates.try_emplace(key.unique_hashes[1], &key);
        auto& [count, example] = state_counts[ShaderStateHash(key)];
        ++count;
        example = &key;
    }
    // On a game's first start it hasn't drawn anything yet, so the starter state stands in for
    // the states it draws. Later starts use the states it drew.
    std::optional<GraphicsPipelineCacheKey> starter;
    const GraphicsPipelineCacheKey* common_template{};
    if (state_counts.empty()) {
        if (!Common::Preforged::Options().first_start_compile) {
            LOG_INFO(Render_Vulkan,
                     "[Preforged] the game hasn't drawn any pipeline of a vertex and a fragment "
                     "shader yet (first start), and compiling on a first start is switched off");
            return;
        }
        starter = StarterKey(dynamic_features);
        common_template = &*starter;
        LOG_INFO(Render_Vulkan,
                 "[Preforged] the game hasn't drawn any pipeline of a vertex and a fragment "
                 "shader yet (first start): compiling the shaders from the game files with the "
                 "starter state (triangles, one RGBA8 color target with depth, no MSAA, no alpha "
                 "test, window origin upper left, depth 0 to 1), fragment shader libraries also "
                 "with the origin lower left{}",
                 device.IsExtDepthClipControlSupported()
                     ? std::string{}
                     : std::string{", vertex shader libraries also with depth -1 to 1"});
    }
    size_t common_count{};
    for (const auto& [hash, entry] : state_counts) {
        if (entry.first > common_count) {
            common_count = entry.first;
            common_template = entry.second;
        }
    }

    // Each attribute gets the type the game feeds most often when a played vertex shader reads
    // it, off included. Only the attributes a shader reads change its SPIR-V, so the others don't
    // matter, and with dynamic vertex input the types aren't part of the pipeline's vertex input.
    // An attribute no played shader reads stays as the template has it, or float if it's off.
    const auto votes{preforged_attribute_votes};
    const auto guess_attributes{[&votes](GraphicsPipelineCacheKey& key) {
        if (!key.state.dynamic_vertex_input) {
            return;
        }
        for (size_t index = 0; index < Maxwell::NumVertexAttributes; ++index) {
            const auto& counts{votes[index]};
            u64 type{key.state.DynamicAttributeType(index) != 0 ? key.state.DynamicAttributeType(index)
                                                                : 1};
            if (counts[0] + counts[1] + counts[2] + counts[3] != 0) {
                type = 0;
                for (u64 candidate = 1; candidate < 4; ++candidate) {
                    if (counts[candidate] > counts[type]) {
                        type = candidate;
                    }
                }
            }
            key.state.attribute_types &= ~(u64{3} << (index * 2));
            key.state.attribute_types |= type << (index * 2);
        }
    }};
    {
        std::vector<std::string> parts;
        for (size_t index = 0; index < votes.size(); ++index) {
            const auto& counts{votes[index]};
            if (counts[0] + counts[1] + counts[2] + counts[3] == 0) {
                continue;
            }
            parts.push_back(fmt::format("{}: {} off {} float {} int {} uint", index, counts[0],
                                        counts[1], counts[2], counts[3]));
        }
        LOG_INFO(Render_Vulkan, "[Preforged] attribute types played shaders read: {}",
                 parts.empty() ? std::string{"none"} : fmt::format("{}", fmt::join(parts, "; ")));
    }
    const auto same_attributes{[](const std::array<u8, 32>& stored, u32 trusted,
                                  const GraphicsPipelineCacheKey& key) {
        for (size_t index = 0; index < stored.size(); ++index) {
            if (stored[index] != 0xFF && ((trusted >> index) & 1) == 0 &&
                stored[index] != static_cast<u8>(AttributeType(key.state, index))) {
                return false;
            }
        }
        return true;
    }};

    preforged_pairs.clear();
    preforged_skipped_pairs.clear();
    preforged_shaders.clear();
    for (const auto& [hash, shader] : programs.shaders) {
        preforged_shaders.insert(hash);
    }
    preforged_other_shaders = {programs.other_shaders.begin(), programs.other_shaders.end()};
    std::unordered_set<u64> played_fragments;
    for (const GraphicsPipelineCacheKey& key : played_keys) {
        played_fragments.insert(key.unique_hashes[5]);
    }
    // Pairs that share a shader with what the game drew so far come first, since they belong to
    // the same areas and effects; a pair whose vertex shader was drawn also gets its real state.
    struct Candidate {
        GraphicsPipelineCacheKey key;
        /// The attribute types are the ones the game drew the vertex shader with.
        bool drawn_types;
    };
    std::array<std::vector<Candidate>, 3> candidates;
    std::unordered_set<u64> seen;
    size_t warmed_count{};
    size_t played_not_stored{};
    size_t played_other_attributes{};
    for (const auto& [vertex, fragment] : programs.pairs) {
        if (!programs.shaders.contains(vertex) || !programs.shaders.contains(fragment)) {
            continue;
        }
        const u64 id{pair_id(vertex, fragment)};
        if (!seen.insert(id).second) {
            continue;
        }
        const auto it{vertex_templates.find(vertex)};
        const bool drawn_types{per_shader_types && it != vertex_templates.end()};
        GraphicsPipelineCacheKey key{it != vertex_templates.end() ? *it->second : *common_template};
        key.unique_hashes = {};
        key.unique_hashes[1] = vertex;
        key.unique_hashes[5] = fragment;
        if (!drawn_types) {
            guess_attributes(key);
        }
        if (played.contains(id)) {
            preforged_pairs.try_emplace(id, ShaderStateHash(key));
            // The game draws a pair with more states as it goes, and each new one looks for the
            // pair's stored code. Without stored code, or with code compiled with other attribute
            // types than the game drew the pair with, the driver compiles the new state in full,
            // so the pair is compiled with the state the game drew.
            if (per_shader_types) {
                const auto stored{warmed.find(id)};
                const auto drawn{played_pair_keys.find(id)};
                if (drawn != played_pair_keys.end() &&
                    (stored == warmed.end() ||
                     (drawn->second->state.dynamic_vertex_input &&
                      !same_attributes(stored->second.attributes, 0, *drawn->second)))) {
                    ++(stored == warmed.end() ? played_not_stored : played_other_attributes);
                    candidates[0].push_back(Candidate{*drawn->second, true});
                }
            }
            continue;
        }
        if (const auto stored{warmed.find(id)}; stored != warmed.end()) {
            // Types the game drew the vertex shader with count over types read from its code.
            const u32 trusted{drawn_types ? 0u : stored->second.code_typed};
            if (!key.state.dynamic_vertex_input ||
                same_attributes(stored->second.attributes, trusted, key)) {
                preforged_pairs.try_emplace(id, ShaderStateHash(key));
                ++warmed_count;
                continue;
            }
            ++stored_with_other_attributes;
        }
        const size_t rank{it != vertex_templates.end() ? 0u
                          : played_fragments.contains(fragment) ? 1u
                                                                : 2u};
        candidates[rank].push_back(Candidate{key, drawn_types});
    }
    const size_t candidate_count{candidates[0].size() + candidates[1].size() +
                                 candidates[2].size()};
    // Pairs compiled on earlier boots count against the share, so it doesn't grow every boot.
    const size_t wanted{((warmed_count + candidate_count) *
                             Common::Preforged::Options().precompile_percent +
                         99) /
                        100};
    std::vector<Candidate> todo;
    for (const auto& rank : candidates) {
        for (const Candidate& candidate : rank) {
            const GraphicsPipelineCacheKey& key{candidate.key};
            if (warmed_count + todo.size() >= wanted) {
                preforged_skipped_pairs.insert(
                    pair_id(key.unique_hashes[1], key.unique_hashes[5]));
                continue;
            }
            todo.push_back(candidate);
            preforged_pairs.try_emplace(pair_id(key.unique_hashes[1], key.unique_hashes[5]),
                                        ShaderStateHash(key));
        }
    }
    LOG_INFO(Render_Vulkan,
             "[Preforged] {} shader pairs from the game files, {} already played, {} compiled "
             "on earlier boots, {} not compiled yet ({} with a played vertex shader, {} with a "
             "played fragment shader, {} others), {} to compile now at {}%, {} states seen, {} "
             "stored pairs compiled again with other attribute types",
             programs.pairs.size(), played.size(), warmed_count, candidate_count,
             candidates[0].size(), candidates[1].size(), candidates[2].size(), todo.size(),
             Common::Preforged::Options().precompile_percent, state_counts.size(),
             stored_with_other_attributes);
    LOG_INFO(Render_Vulkan,
             "[Preforged] attribute types per vertex shader {}: {} drawn vertex shaders keep "
             "the types the game drew them with; played pairs to compile with the state the "
             "game drew: {} not stored yet, {} stored with other attribute types",
             per_shader_types, per_shader_types ? vertex_templates.size() : 0,
             played_not_stored, played_other_attributes);
    if (!Common::Preforged::Options().precompile_game_shaders) {
        LOG_INFO(Render_Vulkan, "[Preforged] compiling switched off in the settings");
        return;
    }
    struct {
        std::mutex mutex;
        size_t built{};
        size_t failed{};
        size_t slow{};
        double total_ms{};
        std::vector<u64> warmed;
        // Attribute types read from the vertex shaders' code, counted once per vertex shader.
        size_t typed_shaders{};
        size_t typed_attributes{};
        size_t typed_changed{};
        size_t typed_unclear{};
        size_t typed_failed{};
        size_t typed_pairs{};
        // The same reading on vertex shaders the game drew, against the types it drew them with.
        size_t check_shaders{};
        size_t check_agree{};
        size_t check_differ{};
        size_t check_off{};
        size_t check_unclear{};
        std::vector<std::string> check_examples;
        std::unordered_map<u64, InputUse> input_uses;
    } state;

    // Reads, once per vertex shader, what its code does with each attribute, and counts how that
    // compares with the types it is compiled with.
    const auto input_use_of{[this, &state](u64 hash, const VideoCommon::PreforgedShader& shader,
                                           u32 texture_bound, const GraphicsPipelineCacheKey& key,
                                           bool drawn_types) {
        {
            std::scoped_lock lock{state.mutex};
            if (const auto it{state.input_uses.find(hash)}; it != state.input_uses.end()) {
                return it->second;
            }
        }
        InputUse use{};
        bool translated{};
        try {
            VideoCommon::PreforgedEnvironment env{shader.code, shader.stage, shader.start_address,
                                                  texture_bound};
            ShaderPools pools;
            const u32 cfg_offset{
                static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader))};
            Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, false);
            const Shader::IR::Program program{
                TranslateProgram(pools.inst, pools.block, env, cfg, host_info)};
            use = InputUseFromCode(program);
            translated = true;
        } catch (const std::exception& e) {
            LOG_DEBUG(Render_Vulkan, "[Preforged] vertex shader {:016x} failed: {}", hash,
                      e.what());
        }
        std::scoped_lock lock{state.mutex};
        const auto [it, is_new]{state.input_uses.try_emplace(hash, use)};
        if (!is_new) {
            return it->second;
        }
        if (!translated) {
            ++state.typed_failed;
            return use;
        }
        if (drawn_types) {
            ++state.check_shaders;
        } else {
            ++state.typed_shaders;
        }
        for (size_t index = 0; index < use.kinds.size(); ++index) {
            if (((use.read >> index) & 1) == 0) {
                continue;
            }
            const u8 kind{use.kinds[index]};
            const u32 type{key.state.DynamicAttributeType(index)};
            const bool clear{kind == INPUT_USED_AS_FLOAT || kind == INPUT_USED_AS_INTEGER};
            if (!drawn_types) {
                if (!clear || type == 0) {
                    ++state.typed_unclear;
                    continue;
                }
                ++state.typed_attributes;
                state.typed_changed += (kind == INPUT_USED_AS_FLOAT) != (type == 1) ? 1 : 0;
                continue;
            }
            if (type == 0) {
                ++state.check_off;
            } else if (!clear) {
                ++state.check_unclear;
            } else if ((kind == INPUT_USED_AS_FLOAT) == (type == 1)) {
                ++state.check_agree;
            } else {
                ++state.check_differ;
                if (state.check_examples.size() < 8) {
                    state.check_examples.push_back(
                        fmt::format("{:016x} attribute {} drawn as {}, code reads it as {}", hash,
                                    index, type == 1 ? "float" : type == 2 ? "int" : "uint",
                                    kind == INPUT_USED_AS_FLOAT ? "float" : "integer"));
                }
            }
        }
        return use;
    }};

    if (!todo.empty()) {
        std::scoped_lock lock{preforged_store_mutex};
        preforged_store_out.open(preforged_store_path, std::ios::binary | std::ios::app);
    }
    const auto start{std::chrono::steady_clock::now()};
    if (!todo.empty()) {
        callback(VideoCore::LoadCallbackStage::PrecompileGameShaders, 0, todo.size());
    }
    for (const Candidate& candidate : todo) {
        workers.QueueWork([this, candidate, programs_ptr, pair_id, per_shader_types, &votes,
                           &input_use_of, total = todo.size(), &state, &callback] {
            const VideoCommon::PreforgedPrograms& shared{**programs_ptr};
            GraphicsPipelineCacheKey key{candidate.key};
            const auto& vertex{shared.shaders.at(key.unique_hashes[1])};
            const auto& fragment{shared.shaders.at(key.unique_hashes[5])};
            // A vertex shader the game hasn't drawn yet gets float or an integer type at each
            // attribute by what its code does with it; one it drew keeps the types it drew.
            u32 code_typed{};
            if (per_shader_types && key.state.dynamic_vertex_input) {
                const InputUse use{input_use_of(key.unique_hashes[1], vertex,
                                                shared.texture_bound, key, candidate.drawn_types)};
                if (!candidate.drawn_types) {
                    code_typed = ApplyInputUse(key, use, votes);
                }
            }
            if (code_typed != 0) {
                std::scoped_lock lock{state.mutex};
                ++state.typed_pairs;
            }
            VideoCommon::PreforgedEnvironment vertex_env{vertex.code, vertex.stage,
                                                         vertex.start_address,
                                                         shared.texture_bound};
            VideoCommon::PreforgedEnvironment fragment_env{fragment.code, fragment.stage,
                                                           fragment.start_address,
                                                           shared.texture_bound};
            std::array<Shader::Environment*, 2> envs{&vertex_env, &fragment_env};
            const auto pipeline_start{std::chrono::steady_clock::now()};
            bool built{};
            try {
                // Each pair gets its own driver cache, so its code can be stored on its own and
                // nothing piles up in the driver pipeline cache the driver keeps in memory.
                const auto cache{std::make_shared<vk::PipelineCache>(
                    device.GetLogical().CreatePipelineCache({
                        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
                        .pNext = nullptr,
                        .flags = 0,
                        .initialDataSize = 0,
                        .pInitialData = nullptr,
                    }))};
                ShaderPools pools;
                PreforgedCode code;
                built = CreateGraphicsPipeline(pools, key, envs, nullptr, false, cache, &code) !=
                        nullptr;
                if (built) {
                    AppendPreforgedStore(pair_id(key.unique_hashes[1], key.unique_hashes[5]),
                                         ShaderStateHash(key), code, code_typed, *cache);
                }
            } catch (const std::exception& e) {
                LOG_DEBUG(Render_Vulkan, "[Preforged] pair failed: {}", e.what());
            }
            const std::chrono::duration<double, std::milli> time{
                std::chrono::steady_clock::now() - pipeline_start};
            std::scoped_lock lock{state.mutex};
            if (built) {
                state.total_ms += time.count();
                state.slow += time.count() > 5.0 ? 1 : 0;
                state.warmed.push_back(pair_id(key.unique_hashes[1], key.unique_hashes[5]));
            } else {
                ++state.failed;
            }
            ++state.built;
            callback(VideoCore::LoadCallbackStage::PrecompileGameShaders, state.built, total);
        });
    }
    workers.WaitForRequests(stop_loading);
    {
        std::scoped_lock lock{preforged_store_mutex};
        preforged_store_out.close();
    }
    const auto seconds{std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start)};
    LOG_INFO(Render_Vulkan,
             "[Preforged] compiled {} shader pairs in {} s ({} failed, {} over 5 ms, {:.1f} ms "
             "average per pair)",
             state.warmed.size(), seconds.count(), state.failed, state.slow,
             state.warmed.empty() ? 0.0 : state.total_ms / state.warmed.size());

    // Pipeline libraries of every vertex and fragment shader from the game files, in a pair or
    // not, so a pipeline the game needs in play that no compiled pair has is joined from the
    // libraries of its two shaders instead of compiled.
    if (use_pipeline_libraries && !stop_loading.stop_requested()) {
        std::unordered_map<u64, const GraphicsPipelineCacheKey*> fragment_templates;
        for (const GraphicsPipelineCacheKey& key : played_keys) {
            if (key.unique_hashes[0] == 0 && key.unique_hashes[2] == 0 &&
                key.unique_hashes[3] == 0 && key.unique_hashes[4] == 0) {
                fragment_templates.try_emplace(key.unique_hashes[5], &key);
            }
        }
        struct LibraryJob {
            u64 hash;
            const VideoCommon::PreforgedShader* shader;
            LibraryHalf half;
            GraphicsPipelineCacheKey key;
            bool drawn_types;
            u64 source;
        };
        // Shaders the game drew come first, since a new pipeline in play most often pairs a
        // drawn shader with another.
        std::array<std::vector<LibraryJob>, 2> library_candidates;
        size_t library_stored{};
        size_t vertex_shaders{};
        size_t fragment_shaders{};
        size_t unpaired_shaders{};
        // Libraries for the other window origin or depth range, on a first start.
        size_t starter_variants{};
        const auto add_library{
            [&](u64 hash, const VideoCommon::PreforgedShader& shader, bool unpaired) {
                LibraryHalf half{};
                if (shader.stage == Shader::Stage::VertexB) {
                    half = LibraryHalf::Front;
                    ++vertex_shaders;
                } else if (shader.stage == Shader::Stage::Fragment) {
                    half = LibraryHalf::Back;
                    ++fragment_shaders;
                } else {
                    return;
                }
                unpaired_shaders += unpaired ? 1 : 0;
                const bool front{half == LibraryHalf::Front};
                const auto& templates{front ? vertex_templates : fragment_templates};
                const auto it{templates.find(hash)};
                const bool drawn{it != templates.end()};
                GraphicsPipelineCacheKey key{drawn ? *it->second : *common_template};
                key.unique_hashes = {};
                key.unique_hashes[front ? 1 : 5] = hash;
                const bool drawn_types{front && per_shader_types && drawn};
                if (front && !drawn_types) {
                    guess_attributes(key);
                }
                const u64 source{LibrarySource(half, hash, ShaderStateHash(key))};
                if (pipeline_libraries.HasSource(source)) {
                    ++library_stored;
                    return;
                }
                library_candidates[drawn ? 0 : 1].push_back(
                    LibraryJob{hash, &shader, half, key, drawn_types, source});
                if (!starter) {
                    return;
                }
                // Before the game has drawn, its window origin isn't known, nor its depth range,
                // which changes vertex shaders without VK_EXT_depth_clip_control. Both change the
                // SPIR-V of the shaders that read them, so the other one is compiled too. A
                // shader whose SPIR-V stays the same is stored once.
                GraphicsPipelineCacheKey other{key};
                if (!front) {
                    other.state.y_negate.Assign(1);
                } else if (!device.IsExtDepthClipControlSupported()) {
                    other.state.ndc_minus_one_to_one.Assign(1);
                } else {
                    return;
                }
                const u64 other_source{LibrarySource(half, hash, ShaderStateHash(other))};
                if (other_source != source && pipeline_libraries.HasSource(other_source)) {
                    return;
                }
                library_candidates[1].push_back(
                    LibraryJob{hash, &shader, half, other, drawn_types, other_source});
                ++starter_variants;
            }};
        for (const auto& [hash, shader] : programs.shaders) {
            add_library(hash, shader, false);
        }
        for (const auto& [hash, shader] : programs.unpaired_shaders) {
            add_library(hash, shader, true);
        }
        const size_t library_count{library_candidates[0].size() + library_candidates[1].size()};
        const size_t library_wanted{
            ((library_stored + library_count) * Common::Preforged::Options().precompile_percent +
             99) /
            100};
        std::vector<LibraryJob> library_todo;
        for (const auto& rank : library_candidates) {
            for (const LibraryJob& job : rank) {
                if (library_stored + library_todo.size() >= library_wanted) {
                    break;
                }
                library_todo.push_back(job);
            }
        }
        LOG_INFO(Render_Vulkan,
                 "[Preforged] pipeline libraries: {} vertex and {} fragment shaders from the game "
                 "files ({} of them in no pair{}), {} compiled on earlier boots, {} not compiled "
                 "yet ({} drawn by the game), {} to compile now at {}%",
                 vertex_shaders, fragment_shaders, unpaired_shaders,
                 programs.unpaired_shaders.empty() && !programs.other_shaders.empty()
                     ? "; the code of shaders in no pair is kept from search version 16 on"
                     : "",
                 library_stored, library_count, library_candidates[0].size(), library_todo.size(),
                 Common::Preforged::Options().precompile_percent);

        struct {
            std::mutex mutex;
            size_t done{};
            size_t compiled{};
            size_t failed{};
            std::array<size_t, LIBRARY_FALLBACK_NAMES.size()> fallbacks{};
            double total_ms{};
            double longest_ms{};
        } library_state;
        const auto library_start{std::chrono::steady_clock::now()};
        const size_t stored_before{pipeline_libraries.StoredCount()};
        if (!library_todo.empty()) {
            callback(VideoCore::LoadCallbackStage::PrecompileGameShaders, 0, library_todo.size());
        }
        for (const LibraryJob& job : library_todo) {
            workers.QueueWork([this, job, programs_ptr, per_shader_types, &votes, &input_use_of,
                               total = library_todo.size(), &library_state, &callback] {
                const VideoCommon::PreforgedPrograms& shared{**programs_ptr};
                GraphicsPipelineCacheKey key{job.key};
                const bool front{job.half == LibraryHalf::Front};
                // Attribute types for a vertex shader the game hasn't drawn, as for the pairs.
                if (front && per_shader_types && key.state.dynamic_vertex_input) {
                    const InputUse use{input_use_of(job.hash, *job.shader, shared.texture_bound,
                                                    key, job.drawn_types)};
                    if (!job.drawn_types) {
                        ApplyInputUse(key, use, votes);
                    }
                }
                VideoCommon::PreforgedEnvironment env{job.shader->code, job.shader->stage,
                                                      job.shader->start_address,
                                                      shared.texture_bound};
                const auto job_start{std::chrono::steady_clock::now()};
                std::optional<LibraryFallback> fallback;
                bool compiled{};
                try {
                    ShaderPools pools;
                    std::array<Shader::IR::Program, Maxwell::MaxShaderProgram> stage_programs;
                    std::array<const Shader::Info*, Maxwell::MaxShaderStage> infos{};
                    std::array<vk::ShaderModule, Maxwell::MaxShaderStage> modules;
                    std::vector<u32> code;
                    const size_t stage{front ? 0U : Maxwell::MaxShaderStage - 1};
                    if (front) {
                        // The vertex shader as a pipeline of only it and a fragment shader
                        // translates it: in set 0 from binding 0, the stage first in line.
                        Shader::IR::Program& program{stage_programs[1]};
                        const u32 cfg_offset{
                            static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader))};
                        Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, false);
                        program = TranslateProgram(pools.inst, pools.block, env, cfg, host_info);
                        infos[stage] = &program.info;
                        if (!FitsLibraryScaling(infos, LibraryHalf::Front)) {
                            fallback = LibraryFallback::TooManyImages;
                        } else {
                            const Shader::RuntimeInfo runtime_info{
                                MakeRuntimeInfo(stage_programs, key, program, nullptr, device)};
                            ConvertLegacyToGeneric(program, runtime_info);
                            Shader::Backend::Bindings bindings{};
                            code = EmitSPIRV(profile, runtime_info, program, bindings);
                        }
                    } else {
                        Shader::IR::Program& program{stage_programs[Maxwell::MaxShaderProgram - 1]};
                        fallback =
                            TranslateLibraryFragment(pools, env, host_info, profile, stage_programs,
                                                     key, device, nullptr, program, code);
                        infos[stage] = &program.info;
                    }
                    if (!fallback) {
                        modules[stage] = BuildShader(device, code);
                        compiled = pipeline_libraries.CompileIntoStore(
                            job.half, FoldCodeHash(0, code), job.source,
                            [&](VkPipelineCache cache) {
                                return GraphicsPipeline::CreateLibrary(device, render_pass_cache,
                                                                       key, job.half, modules,
                                                                       infos, cache);
                            });
                    }
                } catch (const std::exception& e) {
                    LOG_DEBUG(Render_Vulkan, "[Preforged] library of {:016x} failed: {}", job.hash,
                              e.what());
                    fallback = LibraryFallback::TranslateFailed;
                }
                if (fallback || !compiled) {
                    // Not tried again on later boots.
                    pipeline_libraries.NoteSource(job.source);
                }
                const std::chrono::duration<double, std::milli> time{
                    std::chrono::steady_clock::now() - job_start};
                std::scoped_lock lock{library_state.mutex};
                if (fallback) {
                    ++library_state.fallbacks[static_cast<size_t>(*fallback)];
                } else if (compiled) {
                    ++library_state.compiled;
                    library_state.total_ms += time.count();
                    library_state.longest_ms = std::max(library_state.longest_ms, time.count());
                } else {
                    ++library_state.failed;
                }
                ++library_state.done;
                callback(VideoCore::LoadCallbackStage::PrecompileGameShaders, library_state.done,
                         total);
            });
        }
        workers.WaitForRequests(stop_loading);
        std::vector<std::string> skipped;
        for (size_t index = 0; index < library_state.fallbacks.size(); ++index) {
            if (library_state.fallbacks[index] != 0) {
                skipped.push_back(fmt::format("{} {}", library_state.fallbacks[index],
                                              LIBRARY_FALLBACK_NAMES[index]));
            }
        }
        LOG_INFO(
            Render_Vulkan,
            "[Preforged] compiled {} pipeline libraries in {} s ({:.1f} ms average, longest "
            "{:.1f} ms), {} failed in the driver, skipped: {}; the library store holds {}",
            library_state.compiled,
            std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() -
                                                             library_start)
                .count(),
            library_state.compiled == 0 ? 0.0 : library_state.total_ms / library_state.compiled,
            library_state.longest_ms, library_state.failed,
            skipped.empty() ? std::string{"none"} : fmt::format("{}", fmt::join(skipped, ", ")),
            pipeline_libraries.StoredCount());
        if (starter) {
            LOG_INFO(Render_Vulkan,
                     "[Preforged] first start: {} of the libraries were for the other window "
                     "origin or depth range; {} new libraries were stored in all, the rest had "
                     "the SPIR-V of another",
                     starter_variants, pipeline_libraries.StoredCount() - stored_before);
        }
    }
    if (per_shader_types) {
        LOG_INFO(Render_Vulkan,
                 "[Preforged] attribute types from the code of {} vertex shaders the game hasn't "
                 "drawn: {} attributes set to float or integer by the code, {} of them other "
                 "than the most common type, {} unclear (both or neither in the code, or off); "
                 "{} pairs got types from the code; {} vertex shaders failed to translate",
                 state.typed_shaders, state.typed_attributes, state.typed_changed,
                 state.typed_unclear, state.typed_pairs, state.typed_failed);
        LOG_INFO(Render_Vulkan,
                 "[Preforged] check on {} vertex shaders the game drew: the code gives the type "
                 "the game drew for {} attributes, another type for {}, {} were off in the game, "
                 "{} unclear in the code",
                 state.check_shaders, state.check_agree, state.check_differ, state.check_off,
                 state.check_unclear);
        for (const std::string& example : state.check_examples) {
            LOG_INFO(Render_Vulkan, "[Preforged]   other type: {}", example);
        }
    }
}

void PipelineCache::QueueVulkanPipelineCacheFlush() {
    if (!use_vulkan_pipeline_cache || vulkan_pipeline_cache_filename.empty()) {
        return;
    }
    if (++pipelines_since_flush < VULKAN_CACHE_FLUSH_PIPELINES) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    const auto megabytes = last_cache_size.load(std::memory_order_relaxed) / (1024 * 1024);
    const std::chrono::seconds interval{
        std::max<size_t>(VULKAN_CACHE_FLUSH_MIN_SECONDS, megabytes)};
    if (last_flush.time_since_epoch().count() != 0 && now - last_flush < interval) {
        return;
    }
    if (flush_in_flight.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    pipelines_since_flush = 0;
    last_flush = now;
    serialization_thread.QueueWork([this] {
        SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                     CACHE_VERSION);
        size_t size = 0;
        vulkan_pipeline_cache.Read(&size, nullptr);
        last_cache_size.store(size, std::memory_order_relaxed);
        flush_in_flight.store(false, std::memory_order_release);
    });
}

GraphicsPipeline* PipelineCache::CurrentGraphicsPipelineSlowPath() {
    const auto [pair, is_new]{graphics_cache.try_emplace(graphics_key)};
    auto& pipeline{pair->second};
    if (is_new) {
        if (Common::Preforged::Options().log_shader_compiles) {
            LogStateVariant(graphics_key);
        }
        pipeline = CreateGraphicsPipeline();
    }
    if (!pipeline) {
        return nullptr;
    }
    if (current_pipeline) {
        current_pipeline->AddTransition(pipeline.get());
    }
    current_pipeline = pipeline.get();
    return BuiltPipeline(current_pipeline);
}

void PipelineCache::LogStateVariant(const GraphicsPipelineCacheKey& key) {
    // How many variants are shown one by one, and how often the counts are logged.
    static constexpr size_t STATE_EXAMPLES = 24;
    static constexpr size_t SUMMARY_EVERY = 100;
    static constexpr auto SUMMARY_INTERVAL = std::chrono::seconds{30};
    const auto start{std::chrono::steady_clock::now()};
    std::vector<const FixedPipelineState*> states;
    for (const auto& [other, other_pipeline] : graphics_cache) {
        if (other.unique_hashes == key.unique_hashes && other != key) {
            states.push_back(&other.state);
        }
    }
    if (states.empty()) {
        ++state_tally.new_shaders;
    } else {
        const auto [closest, parts]{PreforgedState::Closest(key.state, states)};
        state_tally.Add(parts);
        if (state_examples < STATE_EXAMPLES) {
            ++state_examples;
            LOG_INFO(Render_Vulkan,
                     "[Preforged] state variant: shaders {:016x} had {} pipelines; the closest "
                     "differs in {}: {}; they would share one with {}",
                     fmt::join(key.unique_hashes, ","), states.size(),
                     PreforgedState::PartNames(parts),
                     PreforgedState::DescribeDifference(key.state, *closest, parts),
                     PreforgedState::WAY_NAMES[PreforgedState::WayFor(parts)]);
        }
    }
    const auto now{std::chrono::steady_clock::now()};
    state_search_time += now - start;
    const size_t total{state_tally.new_shaders + state_tally.variants};
    if (total % SUMMARY_EVERY != 0 && now - last_state_summary < SUMMARY_INTERVAL) {
        return;
    }
    last_state_summary = now;
    const auto yes = [](bool value) { return value ? "yes" : "no"; };
    LOG_INFO(Render_Vulkan,
             "[Preforged] pipelines compiled in play: {}; {} (in use now: Extended Dynamic "
             "State 1 {}, 2 {}, 3 blending {}, Vertex Input Dynamic State {}, from the setting "
             "level {}; {} pipelines known, looked through in {} ms in all)",
             total, state_tally.Summary(), yes(dynamic_features.has_extended_dynamic_state),
             yes(dynamic_features.has_extended_dynamic_state_2),
             yes(dynamic_features.has_extended_dynamic_state_3_blend),
             yes(dynamic_features.has_dynamic_vertex_input),
             static_cast<u32>(Settings::values.dyna_state.GetValue()), graphics_cache.size(),
             std::chrono::duration_cast<std::chrono::milliseconds>(state_search_time).count());
    if (use_pipeline_libraries) {
        LOG_INFO(Render_Vulkan, "[Preforged] pipeline libraries in play: {}",
                 LibraryJoinSummary());
    }
}

std::string PipelineCache::LibraryJoinSummary() {
    std::vector<std::string> not_joined;
    for (size_t index = 0; index < library_fallbacks.size(); ++index) {
        if (const size_t count{library_fallbacks[index].load()}; count != 0) {
            not_joined.push_back(fmt::format("{} {}", count, LIBRARY_FALLBACK_NAMES[index]));
        }
    }
    const PipelineLibraries::Counts counts{pipeline_libraries.PlayCounts()};
    return fmt::format("{} pipelines joined, {} didn't need them (the driver had their code), "
                       "not joined: {}; libraries {} from memory, {} from storage ({} of them "
                       "slow, so the driver compiled them again), {} compiled, {} failed",
                       library_joins.load(), library_not_needed.load(),
                       not_joined.empty() ? std::string{"none"}
                                          : fmt::format("{}", fmt::join(not_joined, ", ")),
                       counts.memory, counts.storage, counts.slow_storage, counts.compiled,
                       counts.failed);
}

GraphicsPipeline* PipelineCache::BuiltPipeline(GraphicsPipeline* pipeline) const noexcept {
    if (pipeline->IsBuilt()) {
        return pipeline;
    }
    if (!use_asynchronous_shaders) {
        return pipeline;
    }
    // If games are using a small index count, we can assume these are full screen quads.
    // Usually these shaders are only used once for building textures so we can assume they
    // can't be built async
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    if (draw_state.index_buffer.count <= 6 || draw_state.vertex_buffer.count <= 6) {
        return pipeline;
    }
    return nullptr;
}

std::unique_ptr<GraphicsPipeline> PipelineCache::CreateGraphicsPipeline(
    ShaderPools& pools, const GraphicsPipelineCacheKey& key,
    std::span<Shader::Environment* const> envs, PipelineStatistics* statistics,
    bool build_in_parallel, std::shared_ptr<vk::PipelineCache> build_cache, PreforgedCode* code_out,
    bool store_libraries) try {
    const auto translate_start{std::chrono::steady_clock::now()};
    auto hash = key.Hash();
    LOG_DEBUG(Render_Vulkan, "{:#016x}", hash);
    size_t env_index{0};
    std::array<Shader::IR::Program, Maxwell::MaxShaderProgram> programs;
    const bool uses_vertex_a{key.unique_hashes[0] != 0};
    const bool uses_vertex_b{key.unique_hashes[1] != 0};

    // Layer passthrough generation for devices without VK_EXT_shader_viewport_index_layer
    Shader::IR::Program* layer_source_program{};

    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        const bool is_emulated_stage = layer_source_program != nullptr &&
                                       index == static_cast<u32>(Maxwell::ShaderType::Geometry);
        if (key.unique_hashes[index] == 0 && is_emulated_stage) {
            auto topology = MaxwellToOutputTopology(key.state.topology);
            programs[index] = GenerateGeometryPassthrough(pools.inst, pools.block, host_info,
                                                          *layer_source_program, topology);
            continue;
        }
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        Shader::Environment& env{*envs[env_index]};
        ++env_index;

        const u32 cfg_offset{static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader))};
        Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, index == 0);
        if (!uses_vertex_a || index != 1) {
            // Normal path
            programs[index] = TranslateProgram(pools.inst, pools.block, env, cfg, host_info);
        } else {
            // VertexB path when VertexA is present.
            auto& program_va{programs[0]};
            auto program_vb{TranslateProgram(pools.inst, pools.block, env, cfg, host_info)};
            programs[index] = MergeDualVertexPrograms(program_va, program_vb, env);
        }

        if (device.GetDriverID() != VK_DRIVER_ID_NVIDIA_PROPRIETARY && Shader::Optimization::HasBrokenPattern(programs[index])) {
            LOG_WARNING(Render_Vulkan, "0x{:016x}: Skipping shader with known issues", key.unique_hashes[index]);
            return nullptr;
        }

        if (Settings::values.dump_guest_shaders) {
            env.Dump(hash, key.unique_hashes[index]);
        }

        if (programs[index].info.requires_layer_emulation) {
            layer_source_program = &programs[index];
        }
    }
    std::array<const Shader::Info*, Maxwell::MaxShaderStage> infos{};
    std::array<vk::ShaderModule, Maxwell::MaxShaderStage> modules;

    const Shader::IR::Program* previous_stage{};
    Shader::Backend::Bindings binding;
    PreforgedCode preforged_code;
    // The SPIR-V of the stages before the fragment shader, which names their library, and the
    // last of those stages.
    u64 front_code{};
    const Shader::IR::Program* front_last{};
    for (size_t index = uses_vertex_a && uses_vertex_b ? 1 : 0; index < Maxwell::MaxShaderProgram;
         ++index) {
        const bool is_emulated_stage = layer_source_program != nullptr &&
                                       index == static_cast<u32>(Maxwell::ShaderType::Geometry);
        if (key.unique_hashes[index] == 0 && !is_emulated_stage) {
            continue;
        }
        UNIMPLEMENTED_IF(index == 0);

        Shader::IR::Program& program{programs[index]};
        const size_t stage_index{index - 1};
        infos[stage_index] = &program.info;

        const auto runtime_info{MakeRuntimeInfo(programs, key, program, previous_stage, device)};
        ConvertLegacyToGeneric(program, runtime_info);
        const std::vector<u32> code{EmitSPIRV(profile, runtime_info, program, binding)};
        device.SaveShader(code);
        preforged_code.code_hash = FoldCodeHash(preforged_code.code_hash, code);
        if (stage_index + 1 < Maxwell::MaxShaderStage) {
            front_code = FoldCodeHash(front_code, code);
        } else {
            front_last = previous_stage;
        }
        modules[stage_index] = BuildShader(device, code);

        // Text log + .spv dump. Text log is gated by gpu_log_level != Off; .spv dump
        // is independent and gated only by gpu_log_shader_dumps.
        const bool should_log = GPU::Logging::IsActive();
        const bool should_dump = Settings::values.gpu_log_shader_dumps.GetValue();
        if (should_log || should_dump) {
            static constexpr std::array stage_names{"vertex", "tess_control", "tess_eval", "geometry", "fragment"};
            const std::string shader_name = fmt::format("shader_{:016x}_{}", key.unique_hashes[index], stage_names[stage_index]);
            if (should_log) {
                const std::string shader_info = fmt::format("SPIR-V size: {} bytes, hash: {:016x}",
                    code.size() * sizeof(u32), key.unique_hashes[index]);
                GPU::Logging::GPULogger::GetInstance().LogShaderCompilation(shader_name, shader_info);
            }
            if (should_dump) {
                GPU::Logging::DumpSpirvShader(key.unique_hashes[index],
                                              std::span<const u32>(code.data(), code.size()));
            }
        }

        if (device.HasDebuggingToolAttached()) {
            const std::string name{fmt::format("Shader {:016x}", key.unique_hashes[index])};
            modules[stage_index].SetObjectNameEXT(name.c_str());
        }
        previous_stage = &program;
    }
    if (uses_vertex_b && !uses_vertex_a) {
        preforged_code.inputs = MakePreforgedInputs(key, programs[1].info);
    }
    if (code_out) {
        *code_out = preforged_code;
    }
    const bool given_build_cache{build_cache != nullptr};
    std::string store_match{"not stored"};
    if (!build_cache) {
        build_cache = LoadPreforgedStoreEntry(key, preforged_code, store_match);
    }

    // A pipeline the driver would compile in play is joined from pipeline libraries of its two
    // halves instead: the stages before the fragment shader as they are, and the fragment shader
    // translated once more for its own descriptor set. The driver needs no libraries when it has
    // the pipeline's code already: a stored pair with the same SPIR-V, or a pipeline with the
    // same SPIR-V compiled this session.
    // The test switch joins every pipeline the game drew, while it loads, and every pipeline
    // built in play, but not the pairs compiled into the pair store.
    const bool join_for_test{join_every_pipeline && (build_in_parallel || store_libraries)};
    bool built_before{};
    if (use_pipeline_libraries && build_in_parallel && !build_cache && !join_for_test) {
        std::scoped_lock lock{built_codes_mutex};
        built_before = built_codes.contains(preforged_code.code_hash);
    }
    const bool join{join_for_test || (use_pipeline_libraries && build_in_parallel &&
                                      !build_cache && !built_before)};
    // A pipeline the game drew whose libraries were kept on an earlier boot isn't translated
    // for them again.
    const u64 played_source{store_libraries ? LibrarySource(LibraryHalf::Back, hash, 1) : 0};
    const bool keep_libraries{use_pipeline_libraries && store_libraries &&
                              !pipeline_libraries.HasSource(played_source)};
    std::string library_note{use_pipeline_libraries ? "not needed, the driver has the code"
                                                    : "off"};
    std::optional<GraphicsPipelineLibraryInfo> library_info;
    Shader::IR::Program library_fragment;
    vk::ShaderModule library_module;
    u64 back_code{};
    if (join || keep_libraries) {
        std::optional<LibraryFallback> fallback;
        if (key.unique_hashes[5] == 0 || !modules[Maxwell::MaxShaderStage - 1]) {
            fallback = LibraryFallback::NoFragmentShader;
        } else if (!FitsLibraryScaling(infos, LibraryHalf::Front)) {
            fallback = LibraryFallback::TooManyImages;
        } else {
            // In play the fragment shader's inputs are checked against the stage before it.
            // Libraries kept while the game loads are for any stage before, as the game file
            // shaders' are.
            Shader::VaryingState front_stores{};
            if (front_last) {
                front_stores = front_last->info.stores;
                if (front_last->is_geometry_passthrough) {
                    front_stores.mask |= front_last->info.passthrough.mask;
                }
            }
            std::vector<u32> code;
            fallback = TranslateLibraryFragment(
                pools, *envs.back(), host_info, profile, programs, key, device,
                join && front_last ? &front_stores : nullptr, library_fragment, code);
            if (!fallback) {
                back_code = FoldCodeHash(0, code);
                library_module = BuildShader(device, code);
            }
        }
        if (fallback) {
            library_note = LIBRARY_FALLBACK_NAMES[static_cast<size_t>(*fallback)];
            if (join) {
                ++library_fallbacks[static_cast<size_t>(*fallback)];
            }
        }
        if (!fallback && keep_libraries) {
            // A pipeline the game drew, built while the game loads: its libraries go to
            // storage, so the game can join its shaders with others in play.
            constexpr size_t fragment_stage{Maxwell::MaxShaderStage - 1};
            std::swap(modules[fragment_stage], library_module);
            const Shader::Info* const fragment_info{
                std::exchange(infos[fragment_stage], &library_fragment.info)};
            bool kept{true};
            for (const auto& [half, code] : {std::pair{LibraryHalf::Front, front_code},
                                             std::pair{LibraryHalf::Back, back_code}}) {
                const LibraryHalf library_half{half};
                kept &= pipeline_libraries.CompileIntoStore(
                    library_half, code, 0, [&](VkPipelineCache cache) {
                        return GraphicsPipeline::CreateLibrary(device, render_pass_cache, key,
                                                               library_half, modules, infos, cache);
                    });
            }
            std::swap(modules[fragment_stage], library_module);
            infos[fragment_stage] = fragment_info;
            if (kept) {
                pipeline_libraries.NoteSource(played_source);
            }
        } else if (fallback && keep_libraries) {
            pipeline_libraries.NoteSource(played_source);
        }
        if (!fallback && join) {
            constexpr size_t fragment_stage{Maxwell::MaxShaderStage - 1};
            modules[fragment_stage] = std::move(library_module);
            infos[fragment_stage] = &library_fragment.info;
            library_info = GraphicsPipelineLibraryInfo{
                .libraries = &pipeline_libraries,
                .front_code = front_code,
                .back_code = back_code,
                .full_worker = Common::Preforged::Options().optimize_joined && !join_every_pipeline
                                   ? &full_worker
                                   : nullptr,
            };
            library_note = "joined";
            ++library_joins;
        }
    } else if (use_pipeline_libraries && build_in_parallel) {
        ++library_not_needed;
    }
    if (!build_cache && !library_info) {
        // Compiled into the driver pipeline cache, which other states of the same SPIR-V find.
        std::scoped_lock lock{built_codes_mutex};
        built_codes.insert(preforged_code.code_hash);
    }
    if (build_in_parallel) {
        Common::Preforged::FrameStats::Add(Common::Preforged::FrameStats::Item::ShaderTranslate,
                                           std::chrono::steady_clock::now() - translate_start);
    }
    if (build_in_parallel && Common::Preforged::Options().log_shader_compiles) {
        // Shader timing test: guest shader translation to SPIR-V, done on the GPU thread in game
        const std::chrono::duration<double, std::milli> translate_time{
            std::chrono::steady_clock::now() - translate_start};
        // Whether the shaders came from the game files and the pair was compiled ahead of time,
        // to tell missed pairs from pairs the driver compiled again.
        std::string game_files{"no list"};
        if (!preforged_shaders.empty()) {
            const u64 vertex{key.unique_hashes[1]};
            const u64 fragment{key.unique_hashes[5]};
            const auto pair{preforged_pairs.find((vertex * 0x9E3779B97F4A7C15ULL) ^ fragment)};
            // Files from searches before the list of other shaders can't tell the last two apart.
            const auto where{[this](u64 shader) {
                return preforged_shaders.contains(shader)         ? "in a pair"
                       : preforged_other_shaders.contains(shader) ? "in no pair"
                       : preforged_other_shaders.empty()          ? "not in a pair"
                                                                  : "not among them";
            }};
            game_files = fmt::format(
                "vertex {}, fragment {}, pair {}", where(vertex), where(fragment),
                pair != preforged_pairs.end() ? fmt::format("yes, {}", store_match)
                : preforged_skipped_pairs.contains((vertex * 0x9E3779B97F4A7C15ULL) ^ fragment)
                    ? std::string{"skipped by the pre-compile share"}
                    : std::string{"no"});
        }
        LOG_INFO(Render_Vulkan,
                 "[ShaderTiming] {:016x} translate {:.2f} ms shaders {:016x} game files: {}, "
                 "from storage {}, libraries: {}",
                 hash, translate_time.count(), fmt::join(key.unique_hashes, ","), game_files,
                 build_cache && !given_build_cache ? "yes" : "no", library_note);
    }
    Common::ThreadWorker* const thread_worker{build_in_parallel ? &workers : nullptr};
    return std::make_unique<GraphicsPipeline>(
        scheduler, buffer_cache, texture_cache, vulkan_pipeline_cache, &shader_notify, device,
        descriptor_pool, guest_descriptor_queue, descriptor_buffer_ring, thread_worker, statistics,
        render_pass_cache, key, std::move(modules), infos, std::move(build_cache),
        library_info ? &*library_info : nullptr);

} catch (const Shader::Exception& exception) {
    auto hash = key.Hash();
    size_t env_index{0};
    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        Shader::Environment& env{*envs[env_index]};
        ++env_index;

        const u32 cfg_offset{static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader))};
        Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, index == 0);
        env.Dump(hash, key.unique_hashes[index]);
    }
    LOG_ERROR(Render_Vulkan, "{}", exception.what());
    return nullptr;
}

std::unique_ptr<GraphicsPipeline> PipelineCache::CreateGraphicsPipeline() {
    GraphicsEnvironments environments;
    GetGraphicsEnvironments(environments, graphics_key.unique_hashes);

    main_pools.ReleaseContents();
    auto pipeline{
        CreateGraphicsPipeline(main_pools, graphics_key, environments.Span(), nullptr, true)};
    if (!pipeline || pipeline_cache_filename.empty()) {
        return pipeline;
    }
    serialization_thread.QueueWork([this, key = graphics_key, envs = std::move(environments.envs)] {
        boost::container::static_vector<const GenericEnvironment*, Maxwell::MaxShaderProgram>
            env_ptrs;
        for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] != 0) {
                env_ptrs.push_back(&envs[index]);
            }
        }
        SerializePipeline(key, env_ptrs, pipeline_cache_filename, CACHE_VERSION);
    });
    QueueVulkanPipelineCacheFlush();
    return pipeline;
}

std::unique_ptr<ComputePipeline> PipelineCache::CreateComputePipeline(
    const ComputePipelineCacheKey& key, const ShaderInfo* shader) {
    const Common::Preforged::FrameStats::Timer timer{
        Common::Preforged::FrameStats::Item::ShaderTranslate};
    const GPUVAddr program_base{kepler_compute->regs.code_loc.Address()};
    const auto& qmd{kepler_compute->launch_description};
    ComputeEnvironment env{*kepler_compute, *gpu_memory, program_base, qmd.program_start};
    env.SetCachedSize(shader->size_bytes);

    main_pools.ReleaseContents();
    const auto translate_start{std::chrono::steady_clock::now()};
    auto pipeline{CreateComputePipeline(main_pools, key, env, nullptr, true)};
    if (Common::Preforged::Options().log_shader_compiles) {
        // Compute shaders from the game files aren't compiled ahead yet: their workgroup size
        // and shared memory come from the dispatch, not the code. This shows how many the game
        // needs in play and whether the search has them.
        const std::chrono::duration<double, std::milli> translate_time{
            std::chrono::steady_clock::now() - translate_start};
        const bool listed{!preforged_shaders.empty() || !preforged_other_shaders.empty()};
        const bool found{preforged_shaders.contains(key.unique_hash) ||
                         preforged_other_shaders.contains(key.unique_hash)};
        LOG_INFO(Render_Vulkan,
                 "[ShaderTiming] compute {:016x} translate {:.2f} ms, in the game files: {}, "
                 "workgroup {}x{}x{}, shared memory {} bytes",
                 key.unique_hash, translate_time.count(),
                 !listed ? "no list" : found ? "yes" : "no", key.workgroup_size[0],
                 key.workgroup_size[1], key.workgroup_size[2], key.shared_memory_size);
    }
    if (!pipeline || pipeline_cache_filename.empty()) {
        return pipeline;
    }
    serialization_thread.QueueWork([this, key, env_ = std::move(env)] {
        SerializePipeline(key, std::array<const GenericEnvironment*, 1>{&env_},
                          pipeline_cache_filename, CACHE_VERSION);
    });
    QueueVulkanPipelineCacheFlush();
    return pipeline;
}

std::unique_ptr<ComputePipeline> PipelineCache::CreateComputePipeline(
    ShaderPools& pools, const ComputePipelineCacheKey& key, Shader::Environment& env,
    PipelineStatistics* statistics, bool build_in_parallel) try {
    auto hash = key.Hash();
    if (device.HasBrokenCompute()) {
        LOG_ERROR(Render_Vulkan, "Skipping {:#016x}", hash);
        return nullptr;
    }

    LOG_DEBUG(Render_Vulkan, "{:#016x}", hash);

    Shader::Maxwell::Flow::CFG cfg{env, pools.flow_block, env.StartAddress()};

    // Dump it before error.
    if (Settings::values.dump_guest_shaders) {
        env.Dump(hash, key.unique_hash);
    }

    auto program{TranslateProgram(pools.inst, pools.block, env, cfg, host_info)};
    const VkDriverIdKHR driver_id = device.GetDriverID();
    const bool needs_shared_mem_clamp =
        driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY ||
        driver_id == VK_DRIVER_ID_ARM_PROPRIETARY;
    const u32 max_shared_memory = device.GetMaxComputeSharedMemorySize();
    if (needs_shared_mem_clamp && program.shared_memory_size > max_shared_memory) {
        LOG_WARNING(Render_Vulkan,
                    "Compute shader {:#016x} requests {}KB shared memory but device max is {}KB - clamping",
                    key.unique_hash,
                    program.shared_memory_size / 1024,
                    max_shared_memory / 1024);
        program.shared_memory_size = max_shared_memory;
    }
    const std::vector<u32> code{EmitSPIRV(profile, program)};
    device.SaveShader(code);
    vk::ShaderModule spv_module{BuildShader(device, code)};

    // Text log + .spv dump. Same split as the graphics path.
    const bool should_log = GPU::Logging::IsActive();
    const bool should_dump = Settings::values.gpu_log_shader_dumps.GetValue();
    if (should_log || should_dump) {
        const std::string shader_name = fmt::format("shader_{:016x}_compute", key.unique_hash);
        if (should_log) {
            const std::string shader_info = fmt::format("SPIR-V size: {} bytes, hash: {:016x}",
                code.size() * sizeof(u32), key.unique_hash);
            GPU::Logging::GPULogger::GetInstance().LogShaderCompilation(shader_name, shader_info);
        }
        if (should_dump) {
            GPU::Logging::DumpSpirvShader(key.unique_hash,
                                          std::span<const u32>(code.data(), code.size()));
        }
    }

    if (device.HasDebuggingToolAttached()) {
        const auto name{fmt::format("Shader {:016x}", key.unique_hash)};
        spv_module.SetObjectNameEXT(name.c_str());
    }
    Common::ThreadWorker* const thread_worker{build_in_parallel ? &workers : nullptr};
    return std::make_unique<ComputePipeline>(device, scheduler, vulkan_pipeline_cache, descriptor_pool,
                                             guest_descriptor_queue, descriptor_buffer_ring,
                                             thread_worker, statistics,
                                             &shader_notify, program.info, std::move(spv_module),
                                             key.unique_hash);

} catch (const Shader::Exception& exception) {
    LOG_ERROR(Render_Vulkan, "{}", exception.what());
    return nullptr;
}

void PipelineCache::SerializeVulkanPipelineCache(const std::filesystem::path& filename,
                                                 const vk::PipelineCache& pipeline_cache,
                                                 u32 cache_version) try {
    std::ofstream file(filename, std::ios::binary);
    file.exceptions(std::ifstream::failbit);
    if (!file.is_open()) {
        LOG_ERROR(Common_Filesystem, "Failed to open Vulkan driver pipeline cache file {}",
                  Common::FS::PathToUTF8String(filename));
        return;
    }
    file.write(VULKAN_CACHE_MAGIC_NUMBER.data(), VULKAN_CACHE_MAGIC_NUMBER.size())
        .write(reinterpret_cast<const char*>(&cache_version), sizeof(cache_version));

    size_t cache_size = 0;
    std::vector<char> cache_data;
    if (pipeline_cache) {
        pipeline_cache.Read(&cache_size, nullptr);
        cache_data.resize(cache_size);
        pipeline_cache.Read(&cache_size, cache_data.data());
    }
    file.write(cache_data.data(), cache_size);

    LOG_INFO(Render_Vulkan, "Vulkan driver pipelines cached at: {}",
             Common::FS::PathToUTF8String(filename));

} catch (const std::ios_base::failure& e) {
    LOG_ERROR(Common_Filesystem, "{}", e.what());
    if (!Common::FS::RemoveFile(filename)) {
        LOG_ERROR(Common_Filesystem, "Failed to delete Vulkan driver pipeline cache file {}",
                  Common::FS::PathToUTF8String(filename));
    }
}

vk::PipelineCache PipelineCache::LoadVulkanPipelineCache(const std::filesystem::path& filename,
                                                         u32 expected_cache_version) {
    const auto create_pipeline_cache = [this](size_t data_size, const void* data) {
        VkPipelineCacheCreateInfo pipeline_cache_ci = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .initialDataSize = data_size,
            .pInitialData = data};
        return device.GetLogical().CreatePipelineCache(pipeline_cache_ci);
    };
    try {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            return create_pipeline_cache(0, nullptr);
        }
        file.exceptions(std::ifstream::failbit);
        const auto end{file.tellg()};
        file.seekg(0, std::ios::beg);

        std::array<char, 8> magic_number;
        u32 cache_version;
        file.read(magic_number.data(), magic_number.size())
            .read(reinterpret_cast<char*>(&cache_version), sizeof(cache_version));
        if (magic_number != VULKAN_CACHE_MAGIC_NUMBER || cache_version != expected_cache_version) {
            file.close();
            if (Common::FS::RemoveFile(filename)) {
                if (magic_number != VULKAN_CACHE_MAGIC_NUMBER) {
                    LOG_ERROR(Common_Filesystem, "Invalid Vulkan driver pipeline cache file");
                }
                if (cache_version != expected_cache_version) {
                    LOG_INFO(Common_Filesystem, "Deleting old Vulkan driver pipeline cache");
                }
            } else {
                LOG_ERROR(Common_Filesystem,
                          "Invalid Vulkan pipeline cache file and failed to delete it in \"{}\"",
                          Common::FS::PathToUTF8String(filename));
            }
            return create_pipeline_cache(0, nullptr);
        }

        static constexpr size_t header_size = magic_number.size() + sizeof(cache_version);
        const size_t cache_size = static_cast<size_t>(end) - header_size;
        std::vector<char> cache_data(cache_size);
        file.read(cache_data.data(), cache_size);

        LOG_INFO(Render_Vulkan, "Loaded Vulkan driver pipeline cache: {} MiB from {}",
                 cache_size >> 20, Common::FS::PathToUTF8String(filename));

        return create_pipeline_cache(cache_size, cache_data.data());

    } catch (const std::ios_base::failure& e) {
        LOG_ERROR(Common_Filesystem, "{}", e.what());
        if (!Common::FS::RemoveFile(filename)) {
            LOG_ERROR(Common_Filesystem, "Failed to delete Vulkan driver pipeline cache file {}",
                      Common::FS::PathToUTF8String(filename));
        }

        return create_pipeline_cache(0, nullptr);
    }
}

} // namespace Vulkan
