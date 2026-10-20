// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <type_traits>
#include <vector>

#include "common/thread_worker.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_descriptor_buffer.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace VideoCore {
class ShaderNotify;
}

namespace Vulkan {

struct GraphicsPipelineCacheKey {
    std::array<u64, 6> unique_hashes;
    FixedPipelineState state;

    size_t Hash() const noexcept;

    bool operator==(const GraphicsPipelineCacheKey& rhs) const noexcept;

    bool operator!=(const GraphicsPipelineCacheKey& rhs) const noexcept {
        return !operator==(rhs);
    }

    size_t Size() const noexcept {
        return sizeof(unique_hashes) + state.Size();
    }
};
static_assert(std::has_unique_object_representations_v<GraphicsPipelineCacheKey>);
static_assert(std::is_trivially_copyable_v<GraphicsPipelineCacheKey>);
static_assert(std::is_trivially_constructible_v<GraphicsPipelineCacheKey>);

} // namespace Vulkan

namespace std {
template <>
struct hash<Vulkan::GraphicsPipelineCacheKey> {
    size_t operator()(const Vulkan::GraphicsPipelineCacheKey& k) const noexcept {
        return k.Hash();
    }
};
} // namespace std

namespace Vulkan {

class Device;
class PipelineLibraries;
struct PipelineLibrary;
class PipelineStatistics;
class RenderPassCache;
class RescalingPushConstant;
class RenderAreaPushConstant;
class Scheduler;

/// Which half of a graphics pipeline a pipeline library holds: the front has the vertex input
/// and the stages before the fragment shader, the back has the fragment shader and its output.
enum class LibraryHalf : u8 {
    Front,
    Back,
};

/// What a pipeline joined from libraries needs from the pipeline cache.
struct GraphicsPipelineLibraryInfo {
    /// Hands out the libraries of both halves, compiling the ones nobody has yet.
    PipelineLibraries* libraries{};
    /// Hashes of the SPIR-V of the front stages and of the fragment shader, which name the
    /// libraries.
    u64 front_code{};
    u64 back_code{};
    /// Compiles the full pipeline after joining, or nullptr to keep the joined one.
    Common::ThreadWorker* full_worker{};
};

class GraphicsPipeline {
    static constexpr size_t NUM_STAGES = Tegra::Engines::Maxwell3D::Regs::MaxShaderStage;

public:
    /// Starts the count of the pipelines a game needs in play anew, for a new game.
    static void ResetPlayTally();

    explicit GraphicsPipeline(
        Scheduler& scheduler, BufferCache& buffer_cache, TextureCache& texture_cache,
        vk::PipelineCache& pipeline_cache, VideoCore::ShaderNotify* shader_notify,
        const Device& device, DescriptorPool& descriptor_pool,
        GuestDescriptorQueue& guest_descriptor_queue, DescriptorBufferRing& descriptor_buffer_ring,
        Common::ThreadWorker* worker_thread, PipelineStatistics* pipeline_statistics,
        RenderPassCache& render_pass_cache, const GraphicsPipelineCacheKey& key,
        std::array<vk::ShaderModule, NUM_STAGES> stages,
        const std::array<const Shader::Info*, NUM_STAGES>& infos,
        std::shared_ptr<vk::PipelineCache> build_cache = {},
        const GraphicsPipelineLibraryInfo* library_info = nullptr);

    /// Compiles one half of the pipeline of `key` as a pipeline library, for the game file
    /// shaders compiled while the game loads. `modules` and `infos` are those of a pipeline
    /// joined from libraries, where the fragment shader is on set 1.
    [[nodiscard]] static PipelineLibrary CreateLibrary(
        const Device& device, RenderPassCache& render_pass_cache,
        const GraphicsPipelineCacheKey& key, LibraryHalf half,
        const std::array<vk::ShaderModule, NUM_STAGES>& modules,
        const std::array<const Shader::Info*, NUM_STAGES>& infos, VkPipelineCache cache);

    bool HasDynamicVertexInput() const noexcept { return key.state.dynamic_vertex_input; }
    bool SupportsAlphaToCoverage() const noexcept {
        return fragment_has_color0_output;
    }

    bool SupportsAlphaToOne() const noexcept {
        return fragment_has_color0_output;
    }

    bool UsesExtendedDynamicState() const noexcept {
        return key.state.extended_dynamic_state != 0;
    }
    GraphicsPipeline& operator=(GraphicsPipeline&&) noexcept = delete;
    GraphicsPipeline(GraphicsPipeline&&) noexcept = delete;

    GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;
    GraphicsPipeline(const GraphicsPipeline&) = delete;

    void AddTransition(GraphicsPipeline* transition);

    bool Configure(bool is_indexed) {
        return configure_func(this, is_indexed);
    }

    [[nodiscard]] GraphicsPipeline* Next(const GraphicsPipelineCacheKey& current_key) noexcept {
        if (key == current_key) {
            return this;
        }
        const auto it{std::find(transition_keys.begin(), transition_keys.end(), current_key)};
        return it != transition_keys.end() ? transitions[std::distance(transition_keys.begin(), it)]
                                           : nullptr;
    }

    [[nodiscard]] bool IsBuilt() const noexcept {
        return is_built.load(std::memory_order::relaxed);
    }

    /// Whether the pipeline was joined from libraries.
    [[nodiscard]] bool UsesLibraries() const noexcept {
        return uses_libraries;
    }

    template <typename Spec>
    static auto MakeConfigureSpecFunc() {
        return [](GraphicsPipeline* pl, bool is_indexed) { return pl->ConfigureImpl<Spec>(is_indexed); };
    }

    void SetEngine(Tegra::Engines::Maxwell3D* maxwell3d_, Tegra::MemoryManager* gpu_memory_) {
        maxwell3d = maxwell3d_;
        gpu_memory = gpu_memory_;
    }

private:
    template <typename Spec>
    bool ConfigureImpl(bool is_indexed);

    bool ConfigureDraw(const RescalingPushConstant& rescaling,
                       const RenderAreaPushConstant& render_are);

    void MakePipeline(VkRenderPass render_pass);

    /// Joins the pipeline from the libraries of its two halves. Returns false when the driver
    /// failed to.
    bool JoinPipeline(VkRenderPass render_pass, const GraphicsPipelineLibraryInfo& library_info);

    /// Compiles the full pipeline of one joined from libraries, from the same shaders and with
    /// the same layout.
    [[nodiscard]] vk::Pipeline CompileFullPipeline(VkRenderPass render_pass);

    void Validate();

    const GraphicsPipelineCacheKey key;
    Tegra::Engines::Maxwell3D* maxwell3d;
    Tegra::MemoryManager* gpu_memory;
    const Device& device;
    TextureCache& texture_cache;
    BufferCache& buffer_cache;
    vk::PipelineCache& pipeline_cache;
    /// Driver cache used instead of pipeline_cache for the build, then let go of. Holds code
    /// compiled ahead of time from the game files, read from storage for just this pipeline.
    std::shared_ptr<vk::PipelineCache> build_cache;
    Scheduler& scheduler;
    GuestDescriptorQueue& guest_descriptor_queue;
    DescriptorBufferRing& descriptor_buffer_ring;

    bool (*configure_func)(GraphicsPipeline*, bool){};

    std::vector<GraphicsPipelineCacheKey> transition_keys;
    std::vector<GraphicsPipeline*> transitions;

    std::array<vk::ShaderModule, NUM_STAGES> spv_modules;

    std::array<Shader::Info, NUM_STAGES> stage_infos;
    std::array<u32, 5> enabled_uniform_buffer_masks{};
    VideoCommon::UniformBufferSizes uniform_buffer_sizes{};
    u32 num_descriptor_entries{};
    size_t num_image_elements{};
    u32 num_textures{};
    bool fragment_has_color0_output{};

    vk::DescriptorSetLayout descriptor_set_layout;
    DescriptorAllocator descriptor_allocator;
    vk::PipelineLayout pipeline_layout;
    vk::DescriptorUpdateTemplate descriptor_update_template;
    vk::Pipeline pipeline;
    /// The pipeline draws bind: the joined or compiled one, then the full one compiled after
    /// joining.
    std::atomic<VkPipeline> bound_pipeline{VK_NULL_HANDLE};

    /// For a pipeline joined from libraries: descriptor_set_layout and its allocator and
    /// template are set 0, for the stages before the fragment shader, and these are set 1, for
    /// the fragment shader.
    bool uses_libraries{false};
    vk::DescriptorSetLayout back_set_layout;
    DescriptorAllocator back_descriptor_allocator;
    vk::DescriptorUpdateTemplate back_update_template;
    /// The full pipeline compiled after joining.
    vk::Pipeline full_pipeline;

    DescriptorBufferLayout descriptor_buffer_layout;
    std::vector<DescriptorUpdateEntry> last_descriptor_payload;
    VkDeviceSize last_descriptor_buffer_offset{};
    u32 last_descriptor_buffer_chunk{};
    u64 last_descriptor_buffer_generation{};

    std::condition_variable build_condvar;
    std::mutex build_mutex;
    std::atomic_bool is_built{false};
    bool uses_push_descriptor{false};
    bool uses_descriptor_buffer{false};
};

} // namespace Vulkan
